"""CPU-only oracle for exact endpoint-to-boundary sparse reduction.

This module starts after exact ordered-transfer endpoint cotangents exist.  It
does not fit or validate a temporal atlas, and it does not upgrade the separate
sampled atlas error diagnostic into a continuous or VJP certificate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from compiled_transfer_adjoint import DTYPE, FAR_CUT_ID, NEAR_CUT_ID, StableCellWord


@dataclass(frozen=True)
class SampleEndpointCotangents:
    """One exact P0 transfer sample lowered to finite-cut depth cotangents."""

    transfer: torch.Tensor
    finite_cut_ids: torch.Tensor
    depth_coordinate_cotangents: torch.Tensor
    grad_site_density: torch.Tensor
    grad_site_color: torch.Tensor
    grad_ray_metric: torch.Tensor | None


@dataclass(frozen=True)
class SparseIncidenceReduction:
    """Endpoint events reduced through one four-scalar row per track/cut."""

    grad_depth_coefficients: torch.Tensor
    grad_boundary: torch.Tensor
    grad_ray_coefficients: torch.Tensor | None
    accounting: dict[str, int | bool]


@dataclass(frozen=True)
class DirectEndpointReduction:
    """Independent event-by-event implicit-plane VJP oracle."""

    grad_boundary: torch.Tensor
    grad_ray_coefficients: torch.Tensor | None


def sparse_factorized_depth_coefficients(
    boundary: torch.Tensor,
    ray_coefficients: torch.Tensor,
    incidence: torch.Tensor,
) -> torch.Tensor:
    """Lower only referenced ``(track, boundary)`` pairs to ``[A,B,C,D]``.

    For ``x(z,t)=o0+t*o1+z*(d0+t*d1)`` and
    ``n.x+n_t*t+b=0``, the cut depth is

    ``z(t) = (A+B*t)/(C+D*t)``.
    """

    boundary_f64, rays_f64, incidence_i64 = _validate_sparse_program(
        boundary,
        ray_coefficients,
        incidence,
    )
    if incidence_i64.shape[0] == 0:
        return torch.empty((0, 4), dtype=DTYPE)
    track_ids = incidence_i64[:, 0]
    boundary_ids = incidence_i64[:, 1]
    active_boundary = boundary_f64[boundary_ids]
    active_rays = rays_f64[track_ids]
    normal = active_boundary[:, :3]
    return torch.stack(
        (
            -(active_rays[:, 0:3] * normal).sum(dim=1) - active_boundary[:, 4],
            -(active_rays[:, 3:6] * normal).sum(dim=1) - active_boundary[:, 3],
            (active_rays[:, 6:9] * normal).sum(dim=1),
            (active_rays[:, 9:12] * normal).sum(dim=1),
        ),
        dim=1,
    )


def sample_word_endpoint_cotangents(
    *,
    word: StableCellWord,
    cut_coefficients: dict[int, torch.Tensor],
    ray_coefficient: torch.Tensor,
    time: float | torch.Tensor,
    site_density: torch.Tensor,
    site_color: torch.Tensor,
    grad_transfer: torch.Tensor,
    near: float,
    far: float,
    compute_ray_grad: bool = False,
    denominator_epsilon: float = 1.0e-9,
    physical_length_epsilon: float = 1.0e-8,
) -> SampleEndpointCotangents:
    """Apply the exact constant-state transfer VJP at one ray/time sample.

    ``grad_transfer`` follows ``[beta, m_r, m_g, m_b]``.  For a segment with
    optical-depth cotangent ``tau_bar``, the finite endpoint cotangents are

    ``left_z_bar  = -||d(t)|| * density * tau_bar``
    ``right_z_bar = +||d(t)|| * density * tau_bar``.

    The speed factor is required even when ray gradients are disabled: it is
    the ordinary-depth fiber Jacobian that makes world gradients invariant to
    an orientation-preserving rescaling of the ray-depth coordinate.
    """

    _validate_word(word)
    ray = _f64_vector("ray_coefficient", ray_coefficient, length=12)
    density = _f64_vector("site_density", site_density)
    color = _f64_matrix("site_color", site_color, columns=3)
    transfer_bar = _f64_vector("grad_transfer", grad_transfer, length=4)
    t = torch.as_tensor(time, dtype=DTYPE).reshape(())
    _require_finite("time", t)
    if density.numel() != color.shape[0]:
        raise ValueError("site_density and site_color must share one site count")
    if bool((density < 0.0).any().item()):
        raise ValueError("site_density must be nonnegative")
    if not math.isfinite(near) or not math.isfinite(far) or far <= near:
        raise ValueError("expected finite near/far with far > near")
    if denominator_epsilon <= 0.0 or physical_length_epsilon <= 0.0:
        raise ValueError("safety epsilons must be positive")
    if int(word.owners.min()) < 0 or int(word.owners.max()) >= int(density.numel()):
        raise ValueError("word owner lies outside the site field")

    direction = ray[6:9] + t * ray[9:12]
    fiber_speed = torch.linalg.vector_norm(direction)
    if not bool(torch.isfinite(fiber_speed).item()) or float(fiber_speed.item()) <= 0.0:
        raise ValueError("ray direction has unsafe physical fiber speed")
    speed_jacobian = None
    if compute_ray_grad:
        speed_jacobian = torch.zeros_like(ray)
        speed_jacobian[6:9] = direction / fiber_speed
        speed_jacobian[9:12] = t * direction / fiber_speed

    segments: list[tuple[int, int, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    total_beta = torch.ones((), dtype=DTYPE)
    total_m = torch.zeros(3, dtype=DTYPE)
    for owner_raw, left_raw, right_raw in zip(
        word.owners.tolist(),
        word.left_cut_ids.tolist(),
        word.right_cut_ids.tolist(),
        strict=True,
    ):
        owner = int(owner_raw)
        left_id = int(left_raw)
        right_id = int(right_raw)
        left_depth = _cut_depth(
            cut_coefficients,
            left_id,
            t,
            near=near,
            far=far,
            denominator_epsilon=denominator_epsilon,
        )
        right_depth = _cut_depth(
            cut_coefficients,
            right_id,
            t,
            near=near,
            far=far,
            denominator_epsilon=denominator_epsilon,
        )
        coordinate_length = right_depth - left_depth
        physical_length = fiber_speed * coordinate_length
        if float(coordinate_length.item()) <= 0.0:
            raise ValueError("word produced a non-positive coordinate length")
        if not bool(torch.isfinite(physical_length).item()) or float(physical_length.item()) <= physical_length_epsilon:
            raise ValueError("word produced an unsafe physical segment length")
        segment_beta = torch.exp(-density[owner] * physical_length)
        segments.append(
            (owner, left_id, right_id, coordinate_length, physical_length, total_beta, total_m)
        )
        total_m = total_m + total_beta * (1.0 - segment_beta) * color[owner]
        total_beta = total_beta * segment_beta

    grad_density = torch.zeros_like(density)
    grad_color = torch.zeros_like(color)
    grad_ray_metric = torch.zeros_like(ray) if compute_ray_grad else None
    cut_ids: list[int] = []
    cut_bars: list[torch.Tensor] = []
    beta_bar = transfer_bar[0]
    m_bar = transfer_bar[1:]
    for owner, left_id, right_id, coordinate_length, physical_length, prefix_beta, prefix_m in segments:
        segment_beta = torch.exp(-density[owner] * physical_length)
        tau_bar = torch.dot(m_bar, prefix_m + prefix_beta * color[owner] - total_m) - total_beta * beta_bar
        endpoint_bar = fiber_speed * density[owner] * tau_bar
        if left_id >= 0:
            cut_ids.append(left_id)
            cut_bars.append(-endpoint_bar)
        if right_id >= 0:
            cut_ids.append(right_id)
            cut_bars.append(endpoint_bar)
        grad_density[owner] += physical_length * tau_bar
        grad_color[owner] += prefix_beta * (1.0 - segment_beta) * m_bar
        if grad_ray_metric is not None and speed_jacobian is not None:
            grad_ray_metric += density[owner] * coordinate_length * tau_bar * speed_jacobian

    finite_cut_ids = torch.tensor(cut_ids, dtype=torch.int64)
    depth_coordinate_cotangents = (
        torch.stack(cut_bars).to(dtype=DTYPE)
        if cut_bars
        else torch.empty((0,), dtype=DTYPE)
    )
    return SampleEndpointCotangents(
        transfer=torch.cat((total_beta.reshape(1), total_m)),
        finite_cut_ids=finite_cut_ids,
        depth_coordinate_cotangents=depth_coordinate_cotangents,
        grad_site_density=grad_density,
        grad_site_color=grad_color,
        grad_ray_metric=grad_ray_metric,
    )


def reduce_endpoint_cotangents_via_sparse_incidence(
    *,
    boundary: torch.Tensor,
    ray_coefficients: torch.Tensor,
    incidence: torch.Tensor,
    event_incidence_ids: torch.Tensor,
    event_times: torch.Tensor,
    event_depth_coordinate_cotangents: torch.Tensor,
    event_block_size: int = 1024,
    compute_ray_grad: bool = False,
    grad_ray_metric: torch.Tensor | None = None,
    denominator_epsilon: float = 1.0e-9,
) -> SparseIncidenceReduction:
    """Reduce endpoint cotangents through sparse Möbius coefficients.

    Repeated events for the same incidence are intentional and are summed with
    ``index_add_``.  The event block bounds temporary storage; the persistent
    adjoint is exactly ``[incidence_count, 4]`` and is independent of frame
    count once events are streamed by the caller.
    """

    if event_block_size < 1:
        raise ValueError("event_block_size must be positive")
    if denominator_epsilon <= 0.0:
        raise ValueError("denominator_epsilon must be positive")
    boundary_f64, rays_f64, incidence_i64 = _validate_sparse_program(
        boundary,
        ray_coefficients,
        incidence,
    )
    event_ids, times, depth_bars = _validate_events(
        event_incidence_ids,
        event_times,
        event_depth_coordinate_cotangents,
        incidence_count=int(incidence_i64.shape[0]),
    )
    coefficients = sparse_factorized_depth_coefficients(boundary_f64, rays_f64, incidence_i64)
    grad_coefficients = torch.zeros_like(coefficients)
    peak_event_block = min(event_block_size, int(event_ids.numel()))
    for event_start in range(0, int(event_ids.numel()), event_block_size):
        event_end = min(event_start + event_block_size, int(event_ids.numel()))
        ids = event_ids[event_start:event_end]
        t = times[event_start:event_end]
        bar = depth_bars[event_start:event_end]
        coeff = coefficients[ids]
        numerator = coeff[:, 0] + coeff[:, 1] * t
        denominator = coeff[:, 2] + coeff[:, 3] * t
        denominator_scale = coeff[:, 2].abs() + (coeff[:, 3] * t).abs()
        relative_margin = denominator.abs() / denominator_scale
        if bool(
            (
                (~torch.isfinite(numerator))
                | (~torch.isfinite(denominator))
                | (denominator_scale <= 0.0)
                | (~torch.isfinite(relative_margin))
                | (relative_margin <= denominator_epsilon)
            ).any().item()
        ):
            raise ValueError("an endpoint event has an unsafe depth denominator")
        jacobian = torch.stack(
            (
                1.0 / denominator,
                t / denominator,
                -numerator / denominator.square(),
                -t * numerator / denominator.square(),
            ),
            dim=1,
        )
        grad_coefficients.index_add_(0, ids, bar.unsqueeze(1) * jacobian)

    grad_boundary, grad_ray = _sparse_coefficient_vjp(
        boundary_f64,
        rays_f64,
        incidence_i64,
        grad_coefficients,
        compute_ray_grad=compute_ray_grad,
    )
    if compute_ray_grad:
        if grad_ray_metric is not None:
            metric = _f64_matrix("grad_ray_metric", grad_ray_metric, columns=12)
            if metric.shape != rays_f64.shape:
                raise ValueError("grad_ray_metric must match ray_coefficients")
            grad_ray = grad_ray + metric
    else:
        grad_ray = None

    atomics = sparse_incidence_atomic_accounting(
        finite_endpoint_event_count=int(event_ids.numel()),
        incidence_count=int(incidence_i64.shape[0]),
    )
    return SparseIncidenceReduction(
        grad_depth_coefficients=grad_coefficients,
        grad_boundary=grad_boundary,
        grad_ray_coefficients=grad_ray,
        accounting={
            **atomics,
            "event_count": int(event_ids.numel()),
            "incidence_count": int(incidence_i64.shape[0]),
            "event_block_size": peak_event_block,
            "coefficient_adjoint_bytes": int(grad_coefficients.numel() * grad_coefficients.element_size()),
            "ray_gradient_bytes": 0 if grad_ray is None else int(grad_ray.numel() * grad_ray.element_size()),
        },
    )


def direct_boundary_vjp_from_endpoint_cotangents(
    *,
    boundary: torch.Tensor,
    ray_coefficients: torch.Tensor,
    incidence: torch.Tensor,
    event_incidence_ids: torch.Tensor,
    event_times: torch.Tensor,
    event_depth_coordinate_cotangents: torch.Tensor,
    event_block_size: int = 1024,
    compute_ray_grad: bool = False,
    grad_ray_metric: torch.Tensor | None = None,
    denominator_epsilon: float = 1.0e-9,
) -> DirectEndpointReduction:
    """Scatter endpoint cotangents directly through the implicit plane.

    This implementation deliberately bypasses ``[A,B,C,D]`` and therefore is
    an independent oracle for the sparse-incidence reduction.
    """

    if event_block_size < 1:
        raise ValueError("event_block_size must be positive")
    boundary_f64, rays_f64, incidence_i64 = _validate_sparse_program(
        boundary,
        ray_coefficients,
        incidence,
    )
    event_ids, times, depth_bars = _validate_events(
        event_incidence_ids,
        event_times,
        event_depth_coordinate_cotangents,
        incidence_count=int(incidence_i64.shape[0]),
    )
    grad_boundary = torch.zeros_like(boundary_f64)
    grad_ray = torch.zeros_like(rays_f64) if compute_ray_grad else None
    for event_start in range(0, int(event_ids.numel()), event_block_size):
        event_end = min(event_start + event_block_size, int(event_ids.numel()))
        ids = event_ids[event_start:event_end]
        t = times[event_start:event_end]
        bar = depth_bars[event_start:event_end]
        pairs = incidence_i64[ids]
        track_ids = pairs[:, 0]
        boundary_ids = pairs[:, 1]
        active_boundary = boundary_f64[boundary_ids]
        active_rays = rays_f64[track_ids]
        normal = active_boundary[:, :3]
        origin = active_rays[:, 0:3] + t.unsqueeze(1) * active_rays[:, 3:6]
        direction = active_rays[:, 6:9] + t.unsqueeze(1) * active_rays[:, 9:12]
        denominator = (normal * direction).sum(dim=1)
        denominator_scale = torch.linalg.vector_norm(normal, dim=1) * torch.linalg.vector_norm(direction, dim=1)
        relative_margin = denominator.abs() / denominator_scale
        if bool(
            (
                (~torch.isfinite(denominator))
                | (denominator_scale <= 0.0)
                | (~torch.isfinite(relative_margin))
                | (relative_margin <= denominator_epsilon)
            ).any().item()
        ):
            raise ValueError("an endpoint event has an unsafe implicit-plane denominator")
        numerator = -(
            (normal * origin).sum(dim=1)
            + active_boundary[:, 3] * t
            + active_boundary[:, 4]
        )
        depth = numerator / denominator
        inv_denominator = 1.0 / denominator
        boundary_event_grad = torch.cat(
            (
                -(origin + depth.unsqueeze(1) * direction) * inv_denominator.unsqueeze(1),
                (-t * inv_denominator).unsqueeze(1),
                (-inv_denominator).unsqueeze(1),
            ),
            dim=1,
        ) * bar.unsqueeze(1)
        grad_boundary.index_add_(0, boundary_ids, boundary_event_grad)
        if grad_ray is not None:
            normal_over_denominator = normal * inv_denominator.unsqueeze(1)
            ray_event_grad = torch.cat(
                (
                    -normal_over_denominator,
                    -t.unsqueeze(1) * normal_over_denominator,
                    -depth.unsqueeze(1) * normal_over_denominator,
                    -(t * depth).unsqueeze(1) * normal_over_denominator,
                ),
                dim=1,
            ) * bar.unsqueeze(1)
            grad_ray.index_add_(0, track_ids, ray_event_grad)
    if grad_ray is not None and grad_ray_metric is not None:
        metric = _f64_matrix("grad_ray_metric", grad_ray_metric, columns=12)
        if metric.shape != rays_f64.shape:
            raise ValueError("grad_ray_metric must match ray_coefficients")
        grad_ray += metric
    return DirectEndpointReduction(
        grad_boundary=grad_boundary,
        grad_ray_coefficients=grad_ray,
    )


def sparse_incidence_atomic_accounting(
    *,
    finite_endpoint_event_count: int,
    incidence_count: int,
    scalar_bytes: int = 4,
) -> dict[str, int | bool]:
    """Model logical boundary-path atomic adds and minimum payload bytes.

    This is not a measured memory-traffic model: a real atomic read/modify/write
    moves cache lines and may contend.  It compares the scalar payload of two
    exact strategies:

    - direct: five plane-parameter atomics per finite endpoint;
    - sparse: four coefficient atomics per endpoint, then five plane atomics
      per unique ``(track, boundary)`` incidence.

    Sparse incidence wins this narrow count exactly when average endpoint reuse
    exceeds five events per incidence.
    """

    if finite_endpoint_event_count < 0 or incidence_count < 0 or scalar_bytes < 1:
        raise ValueError("counts must be nonnegative and scalar_bytes positive")
    direct_adds = 5 * finite_endpoint_event_count
    coefficient_adds = 4 * finite_endpoint_event_count
    finalize_adds = 5 * incidence_count
    sparse_adds = coefficient_adds + finalize_adds
    return {
        "direct_boundary_scalar_atomic_adds": direct_adds,
        "sparse_coefficient_scalar_atomic_adds": coefficient_adds,
        "sparse_boundary_finalize_scalar_atomic_adds": finalize_adds,
        "sparse_total_scalar_atomic_adds": sparse_adds,
        "direct_minimum_atomic_payload_bytes": direct_adds * scalar_bytes,
        "sparse_minimum_atomic_payload_bytes": sparse_adds * scalar_bytes,
        "modeled_sparse_incidence_adjoint_bytes": 4 * incidence_count * scalar_bytes,
        "sparse_wins_boundary_atomic_count": sparse_adds < direct_adds,
    }


def _sparse_coefficient_vjp(
    boundary: torch.Tensor,
    ray_coefficients: torch.Tensor,
    incidence: torch.Tensor,
    grad_coefficients: torch.Tensor,
    *,
    compute_ray_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    grad_boundary = torch.zeros_like(boundary)
    grad_ray = torch.zeros_like(ray_coefficients) if compute_ray_grad else None
    if incidence.shape[0] == 0:
        return grad_boundary, grad_ray
    track_ids = incidence[:, 0]
    boundary_ids = incidence[:, 1]
    active_boundary = boundary[boundary_ids]
    active_rays = ray_coefficients[track_ids]
    normal = active_boundary[:, :3]
    grad_a, grad_b, grad_c, grad_d = grad_coefficients.unbind(dim=1)
    boundary_event_grad = torch.cat(
        (
            -grad_a.unsqueeze(1) * active_rays[:, 0:3]
            - grad_b.unsqueeze(1) * active_rays[:, 3:6]
            + grad_c.unsqueeze(1) * active_rays[:, 6:9]
            + grad_d.unsqueeze(1) * active_rays[:, 9:12],
            -grad_b.unsqueeze(1),
            -grad_a.unsqueeze(1),
        ),
        dim=1,
    )
    grad_boundary.index_add_(0, boundary_ids, boundary_event_grad)
    if grad_ray is not None:
        ray_event_grad = torch.cat(
            (
                -grad_a.unsqueeze(1) * normal,
                -grad_b.unsqueeze(1) * normal,
                grad_c.unsqueeze(1) * normal,
                grad_d.unsqueeze(1) * normal,
            ),
            dim=1,
        )
        grad_ray.index_add_(0, track_ids, ray_event_grad)
    return grad_boundary, grad_ray


def _cut_depth(
    cut_coefficients: dict[int, torch.Tensor],
    cut_id: int,
    time: torch.Tensor,
    *,
    near: float,
    far: float,
    denominator_epsilon: float,
) -> torch.Tensor:
    if cut_id == NEAR_CUT_ID:
        return torch.as_tensor(near, dtype=DTYPE)
    if cut_id == FAR_CUT_ID:
        return torch.as_tensor(far, dtype=DTYPE)
    if cut_id < 0 or cut_id not in cut_coefficients:
        raise ValueError(f"invalid cut id {cut_id}")
    coeff = _f64_vector(f"cut_coefficients[{cut_id}]", cut_coefficients[cut_id], length=4)
    numerator = coeff[0] + coeff[1] * time
    denominator = coeff[2] + coeff[3] * time
    scale = coeff[2].abs() + (coeff[3] * time).abs()
    relative_margin = denominator.abs() / scale
    if (
        not bool(torch.isfinite(numerator).item())
        or not bool(torch.isfinite(relative_margin).item())
        or float(scale.item()) <= 0.0
        or float(relative_margin.item()) <= denominator_epsilon
    ):
        raise ValueError("cut has an unsafe depth denominator")
    depth = numerator / denominator
    if not bool(torch.isfinite(depth).item()):
        raise ValueError("cut produced a non-finite depth")
    return depth


def _validate_sparse_program(
    boundary: torch.Tensor,
    ray_coefficients: torch.Tensor,
    incidence: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    boundary_f64 = _f64_matrix("boundary", boundary, columns=5)
    rays_f64 = _f64_matrix("ray_coefficients", ray_coefficients, columns=12)
    incidence_i64 = torch.as_tensor(incidence, dtype=torch.int64).detach()
    if incidence_i64.device.type != "cpu":
        raise ValueError("the exact sparse-incidence oracle is CPU-only")
    if incidence_i64.ndim != 2 or incidence_i64.shape[1] != 2:
        raise ValueError("incidence must have shape [I,2]")
    if incidence_i64.numel():
        if int(incidence_i64[:, 0].min()) < 0 or int(incidence_i64[:, 0].max()) >= int(rays_f64.shape[0]):
            raise ValueError("incidence track id is out of range")
        if int(incidence_i64[:, 1].min()) < 0 or int(incidence_i64[:, 1].max()) >= int(boundary_f64.shape[0]):
            raise ValueError("incidence boundary id is out of range")
        if len({tuple(row) for row in incidence_i64.tolist()}) != int(incidence_i64.shape[0]):
            raise ValueError("incidence rows must be unique; repeated events reference one canonical row")
    return boundary_f64, rays_f64, incidence_i64


def _validate_events(
    event_incidence_ids: torch.Tensor,
    event_times: torch.Tensor,
    event_depth_coordinate_cotangents: torch.Tensor,
    *,
    incidence_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    event_ids = torch.as_tensor(event_incidence_ids, dtype=torch.int64).reshape(-1).detach()
    times = torch.as_tensor(event_times, dtype=DTYPE).reshape(-1).detach()
    depth_bars = torch.as_tensor(event_depth_coordinate_cotangents, dtype=DTYPE).reshape(-1).detach()
    if event_ids.device.type != "cpu" or times.device.type != "cpu" or depth_bars.device.type != "cpu":
        raise ValueError("the exact sparse-incidence oracle is CPU-only")
    if not (event_ids.shape == times.shape == depth_bars.shape):
        raise ValueError("event ids, times, and cotangents must have matching 1D shapes")
    _require_finite("event_times", times)
    _require_finite("event_depth_coordinate_cotangents", depth_bars)
    if event_ids.numel() and (int(event_ids.min()) < 0 or int(event_ids.max()) >= incidence_count):
        raise ValueError("event incidence id is out of range")
    return event_ids, times, depth_bars


def _validate_word(word: StableCellWord) -> None:
    tensors = (word.owners, word.left_cut_ids, word.right_cut_ids)
    if any(tensor.device.type != "cpu" or tensor.dtype != torch.int64 or tensor.ndim != 1 for tensor in tensors):
        raise ValueError("word tensors must be one-dimensional CPU int64")
    if not (tensors[0].shape == tensors[1].shape == tensors[2].shape) or tensors[0].numel() == 0:
        raise ValueError("word tensors must have one shared nonempty shape")


def _f64_matrix(name: str, value: torch.Tensor, *, columns: int) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=DTYPE).detach()
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU")
    if tensor.ndim != 2 or tensor.shape[1] != columns:
        raise ValueError(f"{name} must have shape [N,{columns}]")
    _require_finite(name, tensor)
    return tensor


def _f64_vector(name: str, value: torch.Tensor, *, length: int | None = None) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=DTYPE).reshape(-1).detach()
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU")
    if length is not None and tensor.numel() != length:
        raise ValueError(f"{name} must have {length} values")
    _require_finite(name, tensor)
    return tensor


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must contain only finite values")
