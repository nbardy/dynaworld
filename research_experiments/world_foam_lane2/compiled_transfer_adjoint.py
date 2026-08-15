from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

DTYPE = torch.float64
NEAR_CUT_ID = -1
FAR_CUT_ID = -2


@dataclass(frozen=True)
class StableCellWord:
    """One caller-supplied front-to-back owner word for a time chart."""

    owners: torch.Tensor
    left_cut_ids: torch.Tensor
    right_cut_ids: torch.Tensor


@dataclass(frozen=True)
class ChebyshevTransferAtlas:
    """Experimental transfer coefficients for caller-supplied fixed words.

    The last channel convention is ``[beta, m_r, m_g, m_b]``.  The atlas is
    compiled from a shared world at a fixed number of nodes; requested frame
    count does not appear in its shape.
    """

    t_min: float
    t_max: float
    near: float
    far: float
    node_times: torch.Tensor
    fit_matrix: torch.Tensor
    coefficients: torch.Tensor
    depth_coefficient_incidence: torch.Tensor
    words: tuple[StableCellWord, ...]
    supplied_word_ordering_check: dict[str, float | int | bool]

    @property
    def node_count(self) -> int:
        return int(self.node_times.numel())

    @property
    def track_count(self) -> int:
        return int(self.coefficients.shape[0])

    @property
    def structural_bytes(self) -> int:
        tensors = [self.node_times, self.fit_matrix, self.coefficients, self.depth_coefficient_incidence]
        tensors.extend(tensor for word in self.words for tensor in _word_tensors(word))
        return _tensor_bytes(tensors)


@dataclass(frozen=True)
class CompiledTransferVJP:
    loss: torch.Tensor
    predictions: torch.Tensor | None
    atlas: ChebyshevTransferAtlas
    grad_site_density: torch.Tensor
    grad_site_color: torch.Tensor
    grad_depth_coefficients: torch.Tensor
    grad_boundary: torch.Tensor
    grad_ray_coefficients: torch.Tensor | None
    sampled_validation_error: float
    accounting: dict[str, int]


@dataclass(frozen=True)
class TrackBlockedCompiledTransferVJP:
    """Compiled VJP whose per-step atlas/scratch is bounded by a track block."""

    loss: torch.Tensor
    predictions: torch.Tensor | None
    grad_site_density: torch.Tensor
    grad_site_color: torch.Tensor
    grad_boundary: torch.Tensor
    grad_ray_coefficients: torch.Tensor | None
    sampled_validation_error: float
    accounting: dict[str, int]


@dataclass(frozen=True)
class StreamedWordVJP:
    """Exact fixed-word replay with constant reverse state and chunked samples."""

    loss: torch.Tensor
    predictions: torch.Tensor | None
    supplied_word_ordering_check: dict[str, float | int | bool]
    depth_coefficient_incidence: torch.Tensor
    grad_site_density: torch.Tensor
    grad_site_color: torch.Tensor
    grad_depth_coefficients: torch.Tensor
    grad_boundary: torch.Tensor
    grad_ray_coefficients: torch.Tensor | None
    accounting: dict[str, int]


@dataclass(frozen=True)
class CompiledPowerCellVJP:
    transfer: CompiledTransferVJP | TrackBlockedCompiledTransferVJP
    grad_site_geometry: torch.Tensor
    accounting: dict[str, int]


def make_stable_cell_word(
    owners: Sequence[int] | torch.Tensor,
    left_cut_ids: Sequence[int] | torch.Tensor,
    right_cut_ids: Sequence[int] | torch.Tensor,
) -> StableCellWord:
    word = StableCellWord(
        owners=torch.as_tensor(owners, dtype=torch.int64).reshape(-1).contiguous(),
        left_cut_ids=torch.as_tensor(left_cut_ids, dtype=torch.int64).reshape(-1).contiguous(),
        right_cut_ids=torch.as_tensor(right_cut_ids, dtype=torch.int64).reshape(-1).contiguous(),
    )
    _validate_word_shape(word)
    return word


def chebyshev_nodes(node_count: int, *, t_min: float, t_max: float) -> torch.Tensor:
    if node_count < 2:
        raise ValueError("node_count must be at least 2")
    if not math.isfinite(t_min) or not math.isfinite(t_max) or t_max <= t_min:
        raise ValueError("expected a finite interval with t_max > t_min")
    indices = torch.arange(node_count, dtype=DTYPE)
    normalized = torch.cos(math.pi * (2.0 * indices + 1.0) / (2.0 * node_count))
    return (0.5 * (t_max - t_min) * normalized + 0.5 * (t_max + t_min)).contiguous()


def chebyshev_basis(
    times: torch.Tensor | Sequence[float],
    *,
    t_min: float,
    t_max: float,
    node_count: int,
) -> torch.Tensor:
    times_f64 = torch.as_tensor(times, dtype=DTYPE).reshape(-1)
    if t_max <= t_min:
        raise ValueError("expected t_max > t_min")
    x = (2.0 * times_f64 - (t_max + t_min)) / (t_max - t_min)
    columns = [torch.ones_like(x)]
    if node_count > 1:
        columns.append(x)
    for _ in range(2, node_count):
        columns.append(2.0 * x * columns[-1] - columns[-2])
    return torch.stack(columns, dim=1)


def factorized_depth_coefficients(boundary: torch.Tensor, ray_coefficients: torch.Tensor) -> torch.Tensor:
    """Lower shared 4D planes and affine ray tracks to Mobius depth coefficients.

    ``boundary`` is ``[B,5] = [n_x,n_y,n_z,n_t,b]`` and each ray track is
    ``[12] = [o_0,o_1,d_0,d_1]``.  The returned rows are ``[A,B,C,D]`` for
    ``z(t) = (A + B t) / (C + D t)``.
    """

    boundary = _require_f64_matrix("boundary", boundary, columns=5)
    ray_coefficients = _require_f64_matrix("ray_coefficients", ray_coefficients, columns=12)
    normal = boundary[:, :3]
    normal_t = boundary[:, 3]
    bias = boundary[:, 4]
    origin_base = ray_coefficients[:, 0:3]
    origin_slope = ray_coefficients[:, 3:6]
    direction_base = ray_coefficients[:, 6:9]
    direction_slope = ray_coefficients[:, 9:12]
    numer_base = -(origin_base @ normal.T + bias.unsqueeze(0))
    numer_slope = -(origin_slope @ normal.T + normal_t.unsqueeze(0))
    denom_base = direction_base @ normal.T
    denom_slope = direction_slope @ normal.T
    return torch.stack((numer_base, numer_slope, denom_base, denom_slope), dim=2)


def factorized_depth_coefficients_vjp(
    boundary: torch.Tensor,
    ray_coefficients: torch.Tensor,
    grad_depth_coefficients: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the shared boundary/ray VJP once after temporal reduction."""

    boundary = _require_f64_matrix("boundary", boundary, columns=5)
    ray_coefficients = _require_f64_matrix("ray_coefficients", ray_coefficients, columns=12)
    grad = torch.as_tensor(grad_depth_coefficients, dtype=DTYPE)
    expected = (ray_coefficients.shape[0], boundary.shape[0], 4)
    if tuple(grad.shape) != expected:
        raise ValueError(f"grad_depth_coefficients must have shape {expected}")

    normal = boundary[:, :3]
    origin_base = ray_coefficients[:, 0:3]
    origin_slope = ray_coefficients[:, 3:6]
    direction_base = ray_coefficients[:, 6:9]
    direction_slope = ray_coefficients[:, 9:12]
    grad_a, grad_b, grad_c, grad_d = grad.unbind(dim=2)

    grad_normal = (
        -torch.einsum("mb,mi->bi", grad_a, origin_base)
        - torch.einsum("mb,mi->bi", grad_b, origin_slope)
        + torch.einsum("mb,mi->bi", grad_c, direction_base)
        + torch.einsum("mb,mi->bi", grad_d, direction_slope)
    )
    grad_boundary = torch.cat(
        (
            grad_normal,
            -grad_b.sum(dim=0, keepdim=False).unsqueeze(1),
            -grad_a.sum(dim=0, keepdim=False).unsqueeze(1),
        ),
        dim=1,
    )
    grad_ray = torch.cat(
        (
            -torch.einsum("mb,bi->mi", grad_a, normal),
            -torch.einsum("mb,bi->mi", grad_b, normal),
            torch.einsum("mb,bi->mi", grad_c, normal),
            torch.einsum("mb,bi->mi", grad_d, normal),
        ),
        dim=1,
    )
    return grad_boundary, grad_ray


def power_boundary_parameters(sites: torch.Tensor, boundary_pairs: torch.Tensor) -> torch.Tensor:
    """Build only the active 4D power-cell faces, never the all-pairs graph."""

    sites = _require_f64_matrix("sites", sites, columns=5)
    pairs = torch.as_tensor(boundary_pairs, dtype=torch.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("boundary_pairs must have shape [B,2]")
    if pairs.numel() and (int(pairs.min()) < 0 or int(pairs.max()) >= int(sites.shape[0])):
        raise ValueError("boundary_pairs contains a site id outside sites")
    left = sites[pairs[:, 0]]
    right = sites[pairs[:, 1]]
    normal = 2.0 * (right[:, :4] - left[:, :4])
    bias = left[:, :4].square().sum(dim=1) - right[:, :4].square().sum(dim=1) - left[:, 4] + right[:, 4]
    return torch.cat((normal, bias.unsqueeze(1)), dim=1)


def power_boundary_parameters_vjp(
    sites: torch.Tensor,
    boundary_pairs: torch.Tensor,
    grad_boundary: torch.Tensor,
) -> torch.Tensor:
    """Scatter one active-face adjoint into the shared 4D sites and weights."""

    sites = _require_f64_matrix("sites", sites, columns=5)
    pairs = torch.as_tensor(boundary_pairs, dtype=torch.int64)
    grad_boundary = _require_f64_matrix("grad_boundary", grad_boundary, columns=5)
    if pairs.ndim != 2 or pairs.shape != (grad_boundary.shape[0], 2):
        raise ValueError("boundary_pairs must have shape [grad_boundary rows,2]")
    if pairs.numel() and (int(pairs.min()) < 0 or int(pairs.max()) >= int(sites.shape[0])):
        raise ValueError("boundary_pairs contains a site id outside sites")
    left_ids = pairs[:, 0]
    right_ids = pairs[:, 1]
    grad_normal = grad_boundary[:, :4]
    grad_bias = grad_boundary[:, 4:5]
    left_grad = torch.cat(
        (-2.0 * grad_normal + 2.0 * sites[left_ids, :4] * grad_bias, -grad_bias),
        dim=1,
    )
    right_grad = torch.cat(
        (2.0 * grad_normal - 2.0 * sites[right_ids, :4] * grad_bias, grad_bias),
        dim=1,
    )
    grad_sites = torch.zeros_like(sites)
    grad_sites.index_add_(0, left_ids, left_grad)
    grad_sites.index_add_(0, right_ids, right_grad)
    return grad_sites


def check_power_word_adjacency(
    *,
    sites: torch.Tensor,
    boundary_pairs: torch.Tensor,
    ray_coefficients: torch.Tensor,
    words: Sequence[StableCellWord],
    t_min: float,
    t_max: float,
) -> None:
    """Check that each supplied internal cut has the claimed oriented pair.

    This catches mismatched boundary ids and reversed adjacent owners.  It does
    not prove that no third power cell undercuts a supplied owner; complete
    owner-word discovery/verification remains a separate compiler obligation.
    """

    sites_f64 = _require_f64_matrix("sites", sites, columns=5)
    pairs = torch.as_tensor(boundary_pairs, dtype=torch.int64)
    rays = _require_f64_matrix("ray_coefficients", ray_coefficients, columns=12)
    if len(words) != int(rays.shape[0]):
        raise ValueError("words must contain one stable cell word per ray track")
    boundary = power_boundary_parameters(sites_f64, pairs)
    midpoint = 0.5 * (t_min + t_max)
    for track_id, word in enumerate(words):
        direction = rays[track_id, 6:9] + midpoint * rays[track_id, 9:12]
        for run_id in range(int(word.owners.numel()) - 1):
            cut_id = int(word.right_cut_ids[run_id])
            if cut_id < 0 or cut_id != int(word.left_cut_ids[run_id + 1]):
                raise ValueError("adjacent power-word runs must share one internal boundary")
            if cut_id >= int(pairs.shape[0]):
                raise ValueError("power-word cut id is outside boundary_pairs")
            left_site, right_site = (int(value) for value in pairs[cut_id].tolist())
            denominator = torch.dot(boundary[cut_id, :3], direction)
            if float(denominator.item()) == 0.0:
                raise ValueError("power-word boundary is tangent to the ray at chart midpoint")
            expected = (left_site, right_site) if float(denominator.item()) > 0.0 else (right_site, left_site)
            observed = (int(word.owners[run_id]), int(word.owners[run_id + 1]))
            if observed != expected:
                raise ValueError(
                    f"track {track_id} cut {cut_id} owner transition {observed} does not match {expected}"
                )


def _referenced_depth_coefficient_incidence(words: Sequence[StableCellWord]) -> torch.Tensor:
    rows: list[tuple[int, int]] = []
    for track_id, word in enumerate(words):
        cut_ids = sorted(
            {int(cut_id) for cut_id in torch.cat((word.left_cut_ids, word.right_cut_ids)).tolist() if int(cut_id) >= 0}
        )
        rows.extend((track_id, cut_id) for cut_id in cut_ids)
    if not rows:
        return torch.empty((0, 2), dtype=torch.int64)
    return torch.tensor(rows, dtype=torch.int64)


def _sparse_factorized_depth_coefficients(
    boundary: torch.Tensor,
    ray_coefficients: torch.Tensor,
    incidence: torch.Tensor,
) -> torch.Tensor:
    if incidence.ndim != 2 or incidence.shape[1] != 2:
        raise ValueError("incidence must have shape [I,2]")
    if incidence.shape[0] == 0:
        return torch.empty((0, 4), dtype=DTYPE)
    track_ids = incidence[:, 0]
    boundary_ids = incidence[:, 1]
    active_boundary = boundary[boundary_ids]
    active_ray = ray_coefficients[track_ids]
    normal = active_boundary[:, :3]
    return torch.stack(
        (
            -(active_ray[:, 0:3] * normal).sum(dim=1) - active_boundary[:, 4],
            -(active_ray[:, 3:6] * normal).sum(dim=1) - active_boundary[:, 3],
            (active_ray[:, 6:9] * normal).sum(dim=1),
            (active_ray[:, 9:12] * normal).sum(dim=1),
        ),
        dim=1,
    )


def _sparse_factorized_depth_coefficients_vjp(
    boundary: torch.Tensor,
    ray_coefficients: torch.Tensor,
    incidence: torch.Tensor,
    grad_coefficients: torch.Tensor,
    grad_ray: torch.Tensor | None = None,
    compute_ray_grad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if incidence.ndim != 2 or incidence.shape[1] != 2 or grad_coefficients.shape != (incidence.shape[0], 4):
        raise ValueError("sparse coefficient VJP tensors have incompatible shapes")
    grad_boundary = torch.zeros_like(boundary)
    if not compute_ray_grad:
        grad_ray = None
    elif grad_ray is None:
        grad_ray = torch.zeros_like(ray_coefficients)
    elif grad_ray.shape != ray_coefficients.shape:
        raise ValueError("grad_ray must match ray_coefficients")
    if incidence.shape[0] == 0:
        return grad_boundary, grad_ray
    track_ids = incidence[:, 0]
    boundary_ids = incidence[:, 1]
    active_boundary = boundary[boundary_ids]
    active_ray = ray_coefficients[track_ids]
    normal = active_boundary[:, :3]
    grad_a, grad_b, grad_c, grad_d = grad_coefficients.unbind(dim=1)
    incidence_boundary_grad = torch.cat(
        (
            -grad_a.unsqueeze(1) * active_ray[:, 0:3]
            - grad_b.unsqueeze(1) * active_ray[:, 3:6]
            + grad_c.unsqueeze(1) * active_ray[:, 6:9]
            + grad_d.unsqueeze(1) * active_ray[:, 9:12],
            -grad_b.unsqueeze(1),
            -grad_a.unsqueeze(1),
        ),
        dim=1,
    )
    grad_boundary.index_add_(0, boundary_ids, incidence_boundary_grad)
    if grad_ray is not None:
        incidence_ray_grad = torch.cat(
            (
                -grad_a.unsqueeze(1) * normal,
                -grad_b.unsqueeze(1) * normal,
                grad_c.unsqueeze(1) * normal,
                grad_d.unsqueeze(1) * normal,
            ),
            dim=1,
        )
        grad_ray.index_add_(0, track_ids, incidence_ray_grad)
    return grad_boundary, grad_ray


def _track_cut_coefficient_maps(
    incidence: torch.Tensor,
    coefficients: torch.Tensor,
    *,
    track_count: int,
) -> tuple[dict[int, torch.Tensor], ...]:
    maps: list[dict[int, torch.Tensor]] = [dict() for _ in range(track_count)]
    for incidence_id, (track_id, boundary_id) in enumerate(incidence.tolist()):
        maps[int(track_id)][int(boundary_id)] = coefficients[incidence_id]
    return tuple(maps)


def _incidence_index_maps(incidence: torch.Tensor, *, track_count: int) -> tuple[dict[int, int], ...]:
    maps: list[dict[int, int]] = [dict() for _ in range(track_count)]
    for incidence_id, (track_id, boundary_id) in enumerate(incidence.tolist()):
        maps[int(track_id)][int(boundary_id)] = incidence_id
    return tuple(maps)


def check_supplied_word_ordering(
    *,
    boundary: torch.Tensor,
    ray_coefficients: torch.Tensor,
    words: Sequence[StableCellWord],
    site_count: int,
    t_min: float,
    t_max: float,
    near: float,
    far: float,
    denominator_epsilon: float = 1.0e-9,
    length_epsilon: float = 1.0e-8,
    fiber_speed_epsilon: float = 1.0e-9,
) -> dict[str, float | int | bool]:
    """Analytically check cut denominators and ordering for a supplied word.

    For affine 4D faces and affine ray tracks every cut is Mobius in time.
    Segment length is a quadratic-over-quadratic rational function, so its
    continuous minimum is found from interval endpoints and derivative roots.
    This float64 diagnostic neither discovers/proves cell ownership nor provides
    an outward-rounded interval certificate.
    """

    if not all(math.isfinite(value) for value in (t_min, t_max, near, far)) or t_max <= t_min:
        raise ValueError("expected finite bounds with t_min < t_max")
    if far <= near:
        raise ValueError("far must be greater than near")
    boundary_f64 = _require_f64_matrix("boundary", boundary, columns=5).detach().cpu()
    ray_f64 = _require_f64_matrix("ray_coefficients", ray_coefficients, columns=12).detach().cpu()
    _require_finite("boundary", boundary_f64)
    _require_finite("ray_coefficients", ray_f64)
    if len(words) != int(ray_f64.shape[0]):
        raise ValueError("words must contain one stable cell word per ray track")
    plane_scale = boundary_f64.abs().amax(dim=1, keepdim=True)
    if bool((plane_scale <= 0.0).any().item()):
        raise ValueError("boundary contains a zero homogeneous plane")
    normalized_boundary = boundary_f64 / plane_scale
    incidence = _referenced_depth_coefficient_incidence(words)
    sparse_coefficients = _sparse_factorized_depth_coefficients(normalized_boundary, ray_f64, incidence)
    coefficient_maps = _track_cut_coefficient_maps(
        incidence,
        sparse_coefficients,
        track_count=int(ray_f64.shape[0]),
    )
    min_denominator = math.inf
    min_physical_length_lower_bound = math.inf
    min_fiber_speed = math.inf
    referenced_boundaries: set[tuple[int, int]] = set()
    run_count = 0

    for track_id, word in enumerate(words):
        _validate_word(word, site_count=site_count, boundary_count=int(boundary_f64.shape[0]))
        if int(word.left_cut_ids[0]) != NEAR_CUT_ID or int(word.right_cut_ids[-1]) != FAR_CUT_ID:
            raise ValueError("a stable word must span from the near cut to the far cut")
        if word.owners.numel() > 1 and not torch.equal(word.right_cut_ids[:-1], word.left_cut_ids[1:]):
            raise ValueError("adjacent stable-word runs must share the same cut id")
        run_count += int(word.owners.numel())
        track_min_speed = _minimum_affine_vector_norm(
            ray_f64[track_id, 6:9],
            ray_f64[track_id, 9:12],
            t_min=t_min,
            t_max=t_max,
        )
        min_fiber_speed = min(min_fiber_speed, track_min_speed)
        if track_min_speed <= fiber_speed_epsilon:
            raise ValueError(f"track {track_id} has an unsafe fiber speed over the chart")

        for cut_id in torch.cat((word.left_cut_ids, word.right_cut_ids)).tolist():
            if cut_id < 0:
                continue
            referenced_boundaries.add((track_id, int(cut_id)))
            denom_min = _minimum_affine_cosine_margin(
                normalized_boundary[int(cut_id), :3],
                ray_f64[track_id, 6:9],
                ray_f64[track_id, 9:12],
                t_min=t_min,
                t_max=t_max,
            )
            min_denominator = min(min_denominator, denom_min)
            if denom_min <= denominator_epsilon:
                raise ValueError(f"track {track_id} boundary {cut_id} has an unsafe depth denominator over the chart")

        for left_id, right_id in zip(word.left_cut_ids.tolist(), word.right_cut_ids.tolist(), strict=True):
            left = _cut_coefficients(coefficient_maps[track_id], int(left_id), near=near, far=far)
            right = _cut_coefficients(coefficient_maps[track_id], int(right_id), near=near, far=far)
            length_min = _minimum_rational_difference(right, left, t_min=t_min, t_max=t_max)
            if length_min <= 0.0:
                raise ValueError(
                    f"track {track_id} word loses strict endpoint order over the chart; min coordinate length={length_min}"
                )
            physical_length_lower_bound = length_min * track_min_speed
            min_physical_length_lower_bound = min(
                min_physical_length_lower_bound,
                physical_length_lower_bound,
            )
            if physical_length_lower_bound <= length_epsilon:
                raise ValueError(
                    f"track {track_id} word has an unsafe physical segment-length lower bound over the chart"
                )

    return {
        "passed": True,
        "track_count": int(ray_f64.shape[0]),
        "run_count": run_count,
        "referenced_track_boundaries": len(referenced_boundaries),
        "minimum_relative_denominator_margin": 1.0 if not referenced_boundaries else float(min_denominator),
        "minimum_physical_segment_length_lower_bound": float(min_physical_length_lower_bound),
        "minimum_fiber_speed": float(min_fiber_speed),
    }


def compile_transfer_atlas(
    *,
    boundary: torch.Tensor,
    ray_coefficients: torch.Tensor,
    words: Sequence[StableCellWord],
    site_density: torch.Tensor,
    site_color: torch.Tensor,
    t_min: float,
    t_max: float,
    near: float,
    far: float,
    node_count: int,
    denominator_epsilon: float = 1.0e-9,
    length_epsilon: float = 1.0e-8,
    differentiable: bool = False,
) -> ChebyshevTransferAtlas:
    boundary, ray_coefficients, site_density, site_color, words_tuple = _validate_world_inputs(
        boundary=boundary,
        ray_coefficients=ray_coefficients,
        words=words,
        site_density=site_density,
        site_color=site_color,
    )
    if not differentiable:
        boundary = boundary.detach()
        ray_coefficients = ray_coefficients.detach()
        site_density = site_density.detach()
        site_color = site_color.detach()
    ordering_check = check_supplied_word_ordering(
        boundary=boundary,
        ray_coefficients=ray_coefficients,
        words=words_tuple,
        site_count=int(site_density.numel()),
        t_min=t_min,
        t_max=t_max,
        near=near,
        far=far,
        denominator_epsilon=denominator_epsilon,
        length_epsilon=length_epsilon,
    )
    nodes = chebyshev_nodes(node_count, t_min=t_min, t_max=t_max)
    node_basis = chebyshev_basis(nodes, t_min=t_min, t_max=t_max, node_count=node_count)
    fit_matrix = torch.linalg.inv(node_basis)
    incidence = _referenced_depth_coefficient_incidence(words_tuple)
    sparse_depth_coefficients = _sparse_factorized_depth_coefficients(
        boundary,
        ray_coefficients,
        incidence,
    )
    coefficient_maps = _track_cut_coefficient_maps(
        incidence,
        sparse_depth_coefficients,
        track_count=int(ray_coefficients.shape[0]),
    )
    node_transfer = torch.stack(
        [
            torch.stack(
                [
                    _word_transfer(
                        words_tuple[track_id],
                        coefficient_maps[track_id],
                        ray_coefficients[track_id],
                        time,
                        site_density,
                        site_color,
                        near=near,
                        far=far,
                        denominator_epsilon=denominator_epsilon,
                        length_epsilon=length_epsilon,
                    )
                    for time in nodes
                ],
                dim=0,
            )
            for track_id in range(int(ray_coefficients.shape[0]))
        ],
        dim=0,
    )
    coefficients = torch.einsum("kn,mnc->mkc", fit_matrix, node_transfer)
    return ChebyshevTransferAtlas(
        t_min=float(t_min),
        t_max=float(t_max),
        near=float(near),
        far=float(far),
        node_times=nodes,
        fit_matrix=fit_matrix,
        coefficients=coefficients,
        depth_coefficient_incidence=incidence,
        words=words_tuple,
        supplied_word_ordering_check=ordering_check,
    )


def evaluate_transfer_atlas(
    atlas: ChebyshevTransferAtlas,
    times: torch.Tensor | Sequence[float],
    *,
    background: torch.Tensor | Sequence[float],
) -> torch.Tensor:
    times_f64 = _validate_times_in_chart(times, t_min=atlas.t_min, t_max=atlas.t_max)
    basis = chebyshev_basis(
        times_f64,
        t_min=atlas.t_min,
        t_max=atlas.t_max,
        node_count=atlas.node_count,
    )
    transfer = torch.einsum("fk,mkc->mfc", basis, atlas.coefficients)
    background_f64 = torch.as_tensor(background, dtype=DTYPE).reshape(3)
    return transfer[:, :, 1:] + transfer[:, :, :1] * background_f64.reshape(1, 1, 3)


def direct_word_render(
    *,
    boundary: torch.Tensor,
    ray_coefficients: torch.Tensor,
    words: Sequence[StableCellWord],
    site_density: torch.Tensor,
    site_color: torch.Tensor,
    times: torch.Tensor | Sequence[float],
    background: torch.Tensor | Sequence[float],
    near: float,
    far: float,
) -> torch.Tensor:
    transfer = direct_word_transfer(
        boundary=boundary,
        ray_coefficients=ray_coefficients,
        words=words,
        site_density=site_density,
        site_color=site_color,
        times=times,
        near=near,
        far=far,
    )
    background_f64 = torch.as_tensor(background, dtype=DTYPE).reshape(3)
    return transfer[:, :, 1:] + transfer[:, :, :1] * background_f64.reshape(1, 1, 3)


def direct_word_transfer(
    *,
    boundary: torch.Tensor,
    ray_coefficients: torch.Tensor,
    words: Sequence[StableCellWord],
    site_density: torch.Tensor,
    site_color: torch.Tensor,
    times: torch.Tensor | Sequence[float],
    near: float,
    far: float,
) -> torch.Tensor:
    boundary, ray_coefficients, site_density, site_color, words_tuple = _validate_world_inputs(
        boundary=boundary,
        ray_coefficients=ray_coefficients,
        words=words,
        site_density=site_density,
        site_color=site_color,
    )
    times_f64 = _require_nonempty_finite_times(times)
    incidence = _referenced_depth_coefficient_incidence(words_tuple)
    sparse_depth_coefficients = _sparse_factorized_depth_coefficients(
        boundary,
        ray_coefficients,
        incidence,
    )
    coefficient_maps = _track_cut_coefficient_maps(
        incidence,
        sparse_depth_coefficients,
        track_count=int(ray_coefficients.shape[0]),
    )
    rows = []
    for track_id, word in enumerate(words_tuple):
        track_rows = []
        for time in times_f64:
            track_rows.append(
                _word_transfer(
                    word,
                    coefficient_maps[track_id],
                    ray_coefficients[track_id],
                    time,
                    site_density,
                    site_color,
                    near=near,
                    far=far,
                )
            )
        rows.append(torch.stack(track_rows, dim=0))
    return torch.stack(rows, dim=0)


def sampled_transfer_error(
    atlas: ChebyshevTransferAtlas,
    *,
    boundary: torch.Tensor,
    ray_coefficients: torch.Tensor,
    site_density: torch.Tensor,
    site_color: torch.Tensor,
    validation_count: int = 257,
    track_block_size: int = 64,
    time_block_size: int = 32,
) -> float:
    """Return a blockwise sampled error; this is not an error certificate."""

    if validation_count < 2 or track_block_size < 1 or time_block_size < 1:
        raise ValueError("validation_count must be >=2 and block sizes must be positive")
    times = torch.linspace(atlas.t_min, atlas.t_max, validation_count, dtype=DTYPE)
    rays = _require_f64_matrix("ray_coefficients", ray_coefficients, columns=12).detach()
    max_error = 0.0
    for track_start in range(0, atlas.track_count, track_block_size):
        track_end = min(track_start + track_block_size, atlas.track_count)
        for time_start in range(0, validation_count, time_block_size):
            time_end = min(time_start + time_block_size, validation_count)
            time_block = times[time_start:time_end]
            direct = direct_word_transfer(
                boundary=boundary,
                ray_coefficients=rays[track_start:track_end],
                words=atlas.words[track_start:track_end],
                site_density=site_density,
                site_color=site_color,
                times=time_block,
                near=atlas.near,
                far=atlas.far,
            )
            basis = chebyshev_basis(
                time_block,
                t_min=atlas.t_min,
                t_max=atlas.t_max,
                node_count=atlas.node_count,
            )
            compiled = torch.einsum(
                "fk,mkc->mfc",
                basis,
                atlas.coefficients[track_start:track_end],
            )
            max_error = max(max_error, float((compiled - direct).abs().max().detach().cpu().item()))
    return max_error


def streamed_word_mse_vjp(
    *,
    boundary: torch.Tensor,
    ray_coefficients: torch.Tensor,
    words: Sequence[StableCellWord],
    site_density: torch.Tensor,
    site_color: torch.Tensor,
    times: torch.Tensor | Sequence[float],
    targets: torch.Tensor,
    background: torch.Tensor | Sequence[float],
    t_min: float,
    t_max: float,
    near: float,
    far: float,
    frame_block_size: int = 16,
    return_predictions: bool = False,
    compute_ray_grad: bool = False,
) -> StreamedWordVJP:
    """Exact fixed-topology replay with frame-independent reverse interaction memory.

    This is the conservative first production target.  It still scans the
    supplied cell word at every requested sample, but processes caller-resident
    samples in blocks and uses the prefix-only second-pass identity, so no
    frame-by-run or suffix tape is retained.
    """

    if frame_block_size < 1:
        raise ValueError("frame_block_size must be positive")
    boundary, ray_coefficients, site_density, site_color, words_tuple = _validate_world_inputs(
        boundary=boundary,
        ray_coefficients=ray_coefficients,
        words=words,
        site_density=site_density,
        site_color=site_color,
    )
    boundary = boundary.detach()
    ray_coefficients = ray_coefficients.detach()
    site_density = site_density.detach()
    site_color = site_color.detach()
    times_f64 = _validate_times_in_chart(times, t_min=t_min, t_max=t_max).detach()
    targets_f64 = torch.as_tensor(targets, dtype=DTYPE).detach()
    expected_targets = (ray_coefficients.shape[0], times_f64.numel(), 3)
    if tuple(targets_f64.shape) != expected_targets:
        raise ValueError(f"targets must have shape {expected_targets}")
    background_f64 = torch.as_tensor(background, dtype=DTYPE).reshape(3).detach()
    _require_finite("targets", targets_f64)
    _require_finite("background", background_f64)
    ordering_check = check_supplied_word_ordering(
        boundary=boundary,
        ray_coefficients=ray_coefficients,
        words=words_tuple,
        site_count=int(site_density.numel()),
        t_min=t_min,
        t_max=t_max,
        near=near,
        far=far,
    )
    incidence = _referenced_depth_coefficient_incidence(words_tuple)
    sparse_coefficients = _sparse_factorized_depth_coefficients(boundary, ray_coefficients, incidence)
    coefficient_maps = _track_cut_coefficient_maps(
        incidence,
        sparse_coefficients,
        track_count=int(ray_coefficients.shape[0]),
    )
    incidence_index_maps = _incidence_index_maps(incidence, track_count=int(ray_coefficients.shape[0]))
    grad_site_density = torch.zeros_like(site_density)
    grad_site_color = torch.zeros_like(site_color)
    grad_depth_coefficients = torch.zeros_like(sparse_coefficients)
    grad_ray_metric = torch.zeros_like(ray_coefficients) if compute_ray_grad else None
    predictions = torch.empty(expected_targets, dtype=DTYPE) if return_predictions else None
    loss = torch.zeros((), dtype=DTYPE)
    inv_element_count = 1.0 / float(targets_f64.numel())

    for frame_start in range(0, int(times_f64.numel()), frame_block_size):
        frame_end = min(frame_start + frame_block_size, int(times_f64.numel()))
        for track_id, word in enumerate(words_tuple):
            for frame_id in range(frame_start, frame_end):
                transfer = _word_transfer(
                    word,
                    coefficient_maps[track_id],
                    ray_coefficients[track_id],
                    times_f64[frame_id],
                    site_density,
                    site_color,
                    near=near,
                    far=far,
                )
                prediction = transfer[1:] + transfer[0] * background_f64
                if predictions is not None:
                    predictions[track_id, frame_id] = prediction
                residual = prediction - targets_f64[track_id, frame_id]
                loss += residual.square().sum() * inv_element_count
                grad_prediction = 2.0 * residual * inv_element_count
                grad_transfer = torch.cat((torch.dot(grad_prediction, background_f64).reshape(1), grad_prediction))
                density_grad, color_grad, coefficient_grad, ray_metric_grad = _word_transfer_vjp(
                    word,
                    coefficient_maps[track_id],
                    ray_coefficients[track_id],
                    times_f64[frame_id],
                    site_density,
                    site_color,
                    grad_transfer,
                    near=near,
                    far=far,
                    compute_ray_grad=compute_ray_grad,
                )
                grad_site_density += density_grad
                grad_site_color += color_grad
                for boundary_id, boundary_coefficient_grad in coefficient_grad.items():
                    grad_depth_coefficients[incidence_index_maps[track_id][boundary_id]] += boundary_coefficient_grad
                if grad_ray_metric is not None and ray_metric_grad is not None:
                    grad_ray_metric[track_id] += ray_metric_grad

    grad_boundary, grad_ray = _sparse_factorized_depth_coefficients_vjp(
        boundary,
        ray_coefficients,
        incidence,
        grad_depth_coefficients,
        grad_ray_metric,
        compute_ray_grad=compute_ray_grad,
    )
    scalar_bytes = torch.tensor([], dtype=DTYPE).element_size()
    block_size = min(frame_block_size, int(times_f64.numel()))
    structural_bytes = _tensor_bytes((incidence, *[tensor for word in words_tuple for tensor in _word_tensors(word)]))
    reverse_interaction_bytes = (
        _tensor_bytes(
            (
                grad_site_density,
                grad_site_color,
                grad_depth_coefficients,
                grad_boundary,
                *(() if grad_ray is None else (grad_ray,)),
            )
        )
        + 12 * scalar_bytes
        + int(ray_coefficients.shape[0]) * block_size * 10 * scalar_bytes
    )
    sample_io_bytes = int(ray_coefficients.shape[0]) * int(times_f64.numel()) * 3 * scalar_bytes
    if return_predictions:
        sample_io_bytes *= 2
    return StreamedWordVJP(
        loss=loss,
        predictions=predictions,
        supplied_word_ordering_check=ordering_check,
        depth_coefficient_incidence=incidence,
        grad_site_density=grad_site_density,
        grad_site_color=grad_site_color,
        grad_depth_coefficients=grad_depth_coefficients,
        grad_boundary=grad_boundary,
        grad_ray_coefficients=grad_ray,
        accounting={
            "frame_count": int(times_f64.numel()),
            "frame_block_size": block_size,
            "world_parameter_bytes": _tensor_bytes((boundary, site_density, site_color)),
            "camera_program_bytes": _tensor_bytes((ray_coefficients,)),
            "structural_program_bytes": structural_bytes,
            "reverse_interaction_bytes": reverse_interaction_bytes,
            "sample_io_bytes": sample_io_bytes,
        },
    )


def compiled_transfer_mse_vjp(
    *,
    boundary: torch.Tensor,
    ray_coefficients: torch.Tensor,
    words: Sequence[StableCellWord],
    site_density: torch.Tensor,
    site_color: torch.Tensor,
    times: torch.Tensor | Sequence[float],
    targets: torch.Tensor,
    background: torch.Tensor | Sequence[float],
    t_min: float,
    t_max: float,
    near: float,
    far: float,
    node_count: int,
    frame_block_size: int = 16,
    return_predictions: bool = False,
    compute_ray_grad: bool = False,
    sampled_validation_count: int | None = None,
    sampled_error_tolerance: float = 1.0e-6,
) -> CompiledTransferVJP:
    """Compile, sample, reduce to atlas coefficients, and run one world VJP.

    Cell-word replay occurs at ``node_count`` compile nodes, not at every
    requested time.  The requested-time pass performs only basis evaluation,
    RGB loss, and coefficient reduction.  Logical reverse interaction storage
    is independent of requested frame count; this low-level function still
    receives a resident target tensor and retains a full-track atlas.
    """

    if frame_block_size < 1:
        raise ValueError("frame_block_size must be positive")
    boundary, ray_coefficients, site_density, site_color, words_tuple = _validate_world_inputs(
        boundary=boundary,
        ray_coefficients=ray_coefficients,
        words=words,
        site_density=site_density,
        site_color=site_color,
    )
    boundary = boundary.detach()
    ray_coefficients = ray_coefficients.detach()
    site_density = site_density.detach()
    site_color = site_color.detach()
    times_f64 = _validate_times_in_chart(times, t_min=t_min, t_max=t_max).detach()
    targets_f64 = torch.as_tensor(targets, dtype=DTYPE).detach()
    expected_targets = (ray_coefficients.shape[0], times_f64.numel(), 3)
    if tuple(targets_f64.shape) != expected_targets:
        raise ValueError(f"targets must have shape {expected_targets}")
    background_f64 = torch.as_tensor(background, dtype=DTYPE).reshape(3).detach()
    _require_finite("targets", targets_f64)
    _require_finite("background", background_f64)
    atlas = compile_transfer_atlas(
        boundary=boundary,
        ray_coefficients=ray_coefficients,
        words=words_tuple,
        site_density=site_density,
        site_color=site_color,
        t_min=t_min,
        t_max=t_max,
        near=near,
        far=far,
        node_count=node_count,
    )
    validation_count = max(2 * node_count + 1, 17) if sampled_validation_count is None else sampled_validation_count
    validation_error = sampled_transfer_error(
        atlas,
        boundary=boundary,
        ray_coefficients=ray_coefficients,
        site_density=site_density,
        site_color=site_color,
        validation_count=validation_count,
        track_block_size=min(64, int(ray_coefficients.shape[0])),
        time_block_size=min(32, validation_count),
    )
    if not math.isfinite(sampled_error_tolerance) or sampled_error_tolerance <= 0.0:
        raise ValueError("sampled_error_tolerance must be finite and positive")
    if validation_error > sampled_error_tolerance:
        raise ValueError(
            "fixed-rank transfer atlas exceeds its sampled forward-error gate; "
            f"observed={validation_error}, tolerance={sampled_error_tolerance}"
        )

    grad_coefficients = torch.zeros_like(atlas.coefficients)
    predictions = torch.empty(expected_targets, dtype=DTYPE) if return_predictions else None
    loss_sum = torch.zeros((), dtype=DTYPE)
    inv_element_count = 1.0 / float(targets_f64.numel())
    max_block = min(frame_block_size, int(times_f64.numel()))
    for frame_start in range(0, int(times_f64.numel()), frame_block_size):
        frame_end = min(frame_start + frame_block_size, int(times_f64.numel()))
        basis = chebyshev_basis(
            times_f64[frame_start:frame_end],
            t_min=t_min,
            t_max=t_max,
            node_count=node_count,
        )
        transfer = torch.einsum("fk,mkc->mfc", basis, atlas.coefficients)
        prediction = transfer[:, :, 1:] + transfer[:, :, :1] * background_f64.reshape(1, 1, 3)
        if predictions is not None:
            predictions[:, frame_start:frame_end] = prediction
        residual = prediction - targets_f64[:, frame_start:frame_end]
        loss_sum += residual.square().sum() * inv_element_count
        grad_prediction = 2.0 * residual * inv_element_count
        grad_beta = torch.einsum("mfc,c->mf", grad_prediction, background_f64)
        grad_coefficients[:, :, 0] += torch.einsum("fk,mf->mk", basis, grad_beta)
        grad_coefficients[:, :, 1:] += torch.einsum("fk,mfc->mkc", basis, grad_prediction)

    grad_node_transfer = torch.einsum("kn,mkc->mnc", atlas.fit_matrix, grad_coefficients)
    incidence = atlas.depth_coefficient_incidence
    sparse_depth_coefficients = _sparse_factorized_depth_coefficients(
        boundary,
        ray_coefficients,
        incidence,
    )
    coefficient_maps = _track_cut_coefficient_maps(
        incidence,
        sparse_depth_coefficients,
        track_count=int(ray_coefficients.shape[0]),
    )
    incidence_index_maps = _incidence_index_maps(incidence, track_count=int(ray_coefficients.shape[0]))
    grad_site_density = torch.zeros_like(site_density)
    grad_site_color = torch.zeros_like(site_color)
    grad_depth_coefficients = torch.zeros_like(sparse_depth_coefficients)
    grad_ray_metric = torch.zeros_like(ray_coefficients) if compute_ray_grad else None
    for track_id, word in enumerate(words_tuple):
        for node_id, time in enumerate(atlas.node_times):
            density_grad, color_grad, coefficient_grad, ray_metric_grad = _word_transfer_vjp(
                word,
                coefficient_maps[track_id],
                ray_coefficients[track_id],
                time,
                site_density,
                site_color,
                grad_node_transfer[track_id, node_id],
                near=near,
                far=far,
                compute_ray_grad=compute_ray_grad,
            )
            grad_site_density += density_grad
            grad_site_color += color_grad
            for boundary_id, boundary_coefficient_grad in coefficient_grad.items():
                grad_depth_coefficients[incidence_index_maps[track_id][boundary_id]] += boundary_coefficient_grad
            if grad_ray_metric is not None and ray_metric_grad is not None:
                grad_ray_metric[track_id] += ray_metric_grad

    grad_boundary, grad_ray = _sparse_factorized_depth_coefficients_vjp(
        boundary,
        ray_coefficients,
        incidence,
        grad_depth_coefficients,
        grad_ray_metric,
        compute_ray_grad=compute_ray_grad,
    )
    accounting = compiled_memory_accounting(
        atlas=atlas,
        boundary=boundary,
        ray_coefficients=ray_coefficients,
        site_density=site_density,
        site_color=site_color,
        frame_count=int(times_f64.numel()),
        frame_block_size=max_block,
        return_predictions=return_predictions,
        compute_ray_grad=compute_ray_grad,
    )
    return CompiledTransferVJP(
        loss=loss_sum,
        predictions=predictions,
        atlas=atlas,
        grad_site_density=grad_site_density,
        grad_site_color=grad_site_color,
        grad_depth_coefficients=grad_depth_coefficients,
        grad_boundary=grad_boundary,
        grad_ray_coefficients=grad_ray,
        sampled_validation_error=validation_error,
        accounting=accounting,
    )


def track_blocked_compiled_transfer_mse_vjp(
    *,
    boundary: torch.Tensor,
    ray_coefficients: torch.Tensor,
    words: Sequence[StableCellWord],
    site_density: torch.Tensor,
    site_color: torch.Tensor,
    times: torch.Tensor | Sequence[float],
    targets: torch.Tensor,
    background: torch.Tensor | Sequence[float],
    t_min: float,
    t_max: float,
    near: float,
    far: float,
    node_count: int,
    track_block_size: int = 64,
    frame_block_size: int = 16,
    return_predictions: bool = False,
    compute_ray_grad: bool = False,
    sampled_validation_count: int | None = None,
    sampled_error_tolerance: float = 1.0e-6,
) -> TrackBlockedCompiledTransferVJP:
    """Bound the experimental atlas and adjoints in both track and time.

    The caller still owns the full topology/ray program and target tensor in
    this CPU reference.  This function demonstrates that modeled per-step transfer
    coefficients and reverse scratch need scale with ``track_block_size`` and
    ``frame_block_size`` rather than full image tracks or frame count.
    """

    if track_block_size < 1:
        raise ValueError("track_block_size must be positive")
    boundary_f64, rays_f64, density_f64, color_f64, words_tuple = _validate_world_inputs(
        boundary=boundary,
        ray_coefficients=ray_coefficients,
        words=words,
        site_density=site_density,
        site_color=site_color,
    )
    boundary_f64 = boundary_f64.detach()
    rays_f64 = rays_f64.detach()
    density_f64 = density_f64.detach()
    color_f64 = color_f64.detach()
    times_f64 = _validate_times_in_chart(times, t_min=t_min, t_max=t_max).detach()
    targets_f64 = torch.as_tensor(targets, dtype=DTYPE).detach()
    expected_targets = (rays_f64.shape[0], times_f64.numel(), 3)
    if tuple(targets_f64.shape) != expected_targets:
        raise ValueError(f"targets must have shape {expected_targets}")
    _require_finite("targets", targets_f64)

    track_count = int(rays_f64.shape[0])
    effective_track_block = min(track_block_size, track_count)
    loss = torch.zeros((), dtype=DTYPE)
    predictions = torch.empty(expected_targets, dtype=DTYPE) if return_predictions else None
    grad_density = torch.zeros_like(density_f64)
    grad_color = torch.zeros_like(color_f64)
    grad_boundary = torch.zeros_like(boundary_f64)
    grad_rays = torch.zeros_like(rays_f64) if compute_ray_grad else None
    max_validation_error = 0.0
    peak_block_atlas_bytes = 0
    peak_block_reverse_bytes = 0

    for track_start in range(0, track_count, effective_track_block):
        track_end = min(track_start + effective_track_block, track_count)
        result = compiled_transfer_mse_vjp(
            boundary=boundary_f64,
            ray_coefficients=rays_f64[track_start:track_end],
            words=words_tuple[track_start:track_end],
            site_density=density_f64,
            site_color=color_f64,
            times=times_f64,
            targets=targets_f64[track_start:track_end],
            background=background,
            t_min=t_min,
            t_max=t_max,
            near=near,
            far=far,
            node_count=node_count,
            frame_block_size=frame_block_size,
            return_predictions=return_predictions,
            compute_ray_grad=compute_ray_grad,
            sampled_validation_count=sampled_validation_count,
            sampled_error_tolerance=sampled_error_tolerance,
        )
        weight = float(track_end - track_start) / float(track_count)
        loss += weight * result.loss
        grad_density += weight * result.grad_site_density
        grad_color += weight * result.grad_site_color
        grad_boundary += weight * result.grad_boundary
        if grad_rays is not None and result.grad_ray_coefficients is not None:
            grad_rays[track_start:track_end] = weight * result.grad_ray_coefficients
        if predictions is not None and result.predictions is not None:
            predictions[track_start:track_end] = result.predictions
        max_validation_error = max(max_validation_error, result.sampled_validation_error)
        peak_block_atlas_bytes = max(peak_block_atlas_bytes, result.accounting["atlas_structural_bytes"])
        peak_block_reverse_bytes = max(peak_block_reverse_bytes, result.accounting["reverse_interaction_bytes"])

    topology_bytes = _tensor_bytes(tuple(tensor for word in words_tuple for tensor in _word_tensors(word)))
    global_result_tensors = (grad_density, grad_color, grad_boundary) + (() if grad_rays is None else (grad_rays,))
    global_result_bytes = _tensor_bytes(global_result_tensors)
    if predictions is not None:
        global_result_bytes += _tensor_bytes((predictions,))
    sample_io_bytes = _tensor_bytes((targets_f64, times_f64))
    return TrackBlockedCompiledTransferVJP(
        loss=loss,
        predictions=predictions,
        grad_site_density=grad_density,
        grad_site_color=grad_color,
        grad_boundary=grad_boundary,
        grad_ray_coefficients=grad_rays,
        sampled_validation_error=max_validation_error,
        accounting={
            "frame_count": int(times_f64.numel()),
            "frame_block_size": min(frame_block_size, int(times_f64.numel())),
            "track_count": track_count,
            "track_block_size": effective_track_block,
            "world_parameter_bytes": _tensor_bytes((boundary_f64, density_f64, color_f64)),
            "camera_program_bytes": _tensor_bytes((rays_f64,)),
            "caller_resident_topology_bytes": topology_bytes,
            "caller_resident_sample_io_bytes": sample_io_bytes,
            "global_result_bytes": global_result_bytes,
            "peak_block_atlas_bytes": peak_block_atlas_bytes,
            "peak_block_reverse_bytes": peak_block_reverse_bytes,
            "logical_peak_step_bytes_excluding_allocator_and_caller_inputs": (
                global_result_bytes + peak_block_atlas_bytes + peak_block_reverse_bytes
            ),
        },
    )


def compiled_power_cell_mse_vjp(
    *,
    sites: torch.Tensor,
    boundary_pairs: torch.Tensor,
    ray_coefficients: torch.Tensor,
    words: Sequence[StableCellWord],
    site_density: torch.Tensor,
    site_color: torch.Tensor,
    times: torch.Tensor | Sequence[float],
    targets: torch.Tensor,
    background: torch.Tensor | Sequence[float],
    t_min: float,
    t_max: float,
    near: float,
    far: float,
    node_count: int,
    track_block_size: int = 64,
    frame_block_size: int = 16,
    return_predictions: bool = False,
    compute_ray_grad: bool = False,
) -> CompiledPowerCellVJP:
    """Run the fixed-topology atlas VJP through sparse power-cell faces."""

    sites_f64 = _require_f64_matrix("sites", sites, columns=5).detach()
    pairs_i64 = torch.as_tensor(boundary_pairs, dtype=torch.int64)
    rays_f64 = _require_f64_matrix("ray_coefficients", ray_coefficients, columns=12).detach()
    density_f64 = torch.as_tensor(site_density, dtype=DTYPE).reshape(-1).detach()
    color_f64 = _require_f64_matrix("site_color", site_color, columns=3).detach()
    if not (sites_f64.shape[0] == density_f64.numel() == color_f64.shape[0]):
        raise ValueError("geometry, density, and color must share one site count")
    boundary = power_boundary_parameters(sites_f64, pairs_i64)
    check_power_word_adjacency(
        sites=sites_f64,
        boundary_pairs=pairs_i64,
        ray_coefficients=rays_f64,
        words=words,
        t_min=t_min,
        t_max=t_max,
    )
    transfer = track_blocked_compiled_transfer_mse_vjp(
        boundary=boundary,
        ray_coefficients=rays_f64,
        words=words,
        site_density=density_f64,
        site_color=color_f64,
        times=times,
        targets=targets,
        background=background,
        t_min=t_min,
        t_max=t_max,
        near=near,
        far=far,
        node_count=node_count,
        track_block_size=track_block_size,
        frame_block_size=frame_block_size,
        return_predictions=return_predictions,
        compute_ray_grad=compute_ray_grad,
    )
    accounting = dict(transfer.accounting)
    accounting["world_parameter_bytes"] = _tensor_bytes((sites_f64, density_f64, color_f64))
    accounting["caller_supplied_active_pair_bytes"] = _tensor_bytes((pairs_i64,))
    accounting["derived_active_boundary_bytes"] = _tensor_bytes((boundary,))
    accounting["caller_resident_topology_bytes"] += accounting["caller_supplied_active_pair_bytes"]
    grad_site_geometry = power_boundary_parameters_vjp(sites_f64, pairs_i64, transfer.grad_boundary)
    returned_gradients = (
        grad_site_geometry,
        transfer.grad_site_density,
        transfer.grad_site_color,
    ) + (() if transfer.grad_ray_coefficients is None else (transfer.grad_ray_coefficients,))
    accounting["returned_world_gradient_bytes"] = _tensor_bytes(returned_gradients)
    accounting["logical_peak_step_bytes_excluding_allocator_and_caller_inputs"] += (
        accounting["derived_active_boundary_bytes"] + _tensor_bytes((grad_site_geometry,))
    )
    return CompiledPowerCellVJP(
        transfer=transfer,
        grad_site_geometry=grad_site_geometry,
        accounting=accounting,
    )


def compiled_memory_accounting(
    *,
    atlas: ChebyshevTransferAtlas,
    boundary: torch.Tensor,
    ray_coefficients: torch.Tensor,
    site_density: torch.Tensor,
    site_color: torch.Tensor,
    frame_count: int,
    frame_block_size: int,
    return_predictions: bool,
    compute_ray_grad: bool = False,
) -> dict[str, int]:
    """Model selected logical tensor payloads, not measured/live peak memory."""

    track_count = atlas.track_count
    node_count = atlas.node_count
    boundary_count = int(boundary.shape[0])
    incidence_count = int(atlas.depth_coefficient_incidence.shape[0])
    site_count = int(site_density.numel())
    scalar_bytes = torch.tensor([], dtype=DTYPE).element_size()
    world_parameter_bytes = _tensor_bytes((boundary, site_density, site_color))
    camera_program_bytes = _tensor_bytes((ray_coefficients,))
    coefficient_adjoint_bytes = track_count * node_count * 4 * scalar_bytes
    node_adjoint_bytes = coefficient_adjoint_bytes
    depth_coefficient_adjoint_bytes = incidence_count * 4 * scalar_bytes
    world_gradient_scalars = boundary_count * 5 + site_count * 4
    if compute_ray_grad:
        world_gradient_scalars += track_count * 12
    world_gradient_bytes = world_gradient_scalars * scalar_bytes
    constant_state_word_vjp_bytes = 12 * scalar_bytes
    streamed_block_bytes = (
        frame_block_size * node_count + track_count * frame_block_size * (4 + 3 + 3 + 3 + 1)
    ) * scalar_bytes
    reverse_interaction_bytes = (
        coefficient_adjoint_bytes
        + node_adjoint_bytes
        + depth_coefficient_adjoint_bytes
        + world_gradient_bytes
        + constant_state_word_vjp_bytes
        + streamed_block_bytes
    )
    sample_io_bytes = track_count * frame_count * 3 * scalar_bytes
    if return_predictions:
        sample_io_bytes *= 2
    return {
        "frame_count": int(frame_count),
        "frame_block_size": int(frame_block_size),
        "world_parameter_bytes": int(world_parameter_bytes),
        "camera_program_bytes": int(camera_program_bytes),
        "atlas_structural_bytes": int(atlas.structural_bytes),
        "reverse_interaction_bytes": int(reverse_interaction_bytes),
        "sample_io_bytes": int(sample_io_bytes),
        "output_and_target_bytes_excluded_from_reverse_contract": int(sample_io_bytes),
    }


def _word_transfer(
    word: StableCellWord,
    cut_coefficients: dict[int, torch.Tensor],
    ray_coefficient: torch.Tensor,
    time: torch.Tensor,
    site_density: torch.Tensor,
    site_color: torch.Tensor,
    *,
    near: float,
    far: float,
    denominator_epsilon: float = 1.0e-9,
    length_epsilon: float = 1.0e-8,
) -> torch.Tensor:
    beta_total = torch.ones((), dtype=DTYPE)
    m_total = torch.zeros(3, dtype=DTYPE)
    fiber_speed, _ = _fiber_speed_and_jacobian(ray_coefficient, time, compute_jacobian=False)
    for owner, left_id, right_id in zip(
        word.owners.tolist(),
        word.left_cut_ids.tolist(),
        word.right_cut_ids.tolist(),
        strict=True,
    ):
        left_depth, _ = _cut_depth_and_jacobian(
            cut_coefficients,
            int(left_id),
            time,
            near=near,
            far=far,
            denominator_epsilon=denominator_epsilon,
        )
        right_depth, _ = _cut_depth_and_jacobian(
            cut_coefficients,
            int(right_id),
            time,
            near=near,
            far=far,
            denominator_epsilon=denominator_epsilon,
        )
        coordinate_length = right_depth - left_depth
        if float(coordinate_length.detach().cpu().item()) <= 0.0:
            raise ValueError("stable cell word produced a non-positive segment length")
        physical_length = fiber_speed * coordinate_length
        if float(physical_length.detach().cpu().item()) <= length_epsilon:
            raise ValueError("stable cell word produced an unsafe physical segment length")
        beta = torch.exp(-site_density[int(owner)] * physical_length)
        m_total = m_total + beta_total * (1.0 - beta) * site_color[int(owner)]
        beta_total = beta_total * beta
    return torch.cat((beta_total.reshape(1), m_total))


def _word_transfer_vjp(
    word: StableCellWord,
    cut_coefficients: dict[int, torch.Tensor],
    ray_coefficient: torch.Tensor,
    time: torch.Tensor,
    site_density: torch.Tensor,
    site_color: torch.Tensor,
    grad_transfer: torch.Tensor,
    *,
    near: float,
    far: float,
    compute_ray_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[int, torch.Tensor], torch.Tensor | None]:
    total_transfer = _word_transfer(
        word,
        cut_coefficients,
        ray_coefficient,
        time,
        site_density,
        site_color,
        near=near,
        far=far,
    )
    total_beta = total_transfer[0]
    total_m = total_transfer[1:]
    grad_site_density = torch.zeros_like(site_density)
    grad_site_color = torch.zeros_like(site_color)
    grad_coefficients = {cut_id: torch.zeros(4, dtype=DTYPE) for cut_id in cut_coefficients}
    grad_ray_metric = torch.zeros_like(ray_coefficient) if compute_ray_grad else None
    grad_beta_total = grad_transfer[0]
    grad_m_total = grad_transfer[1:]
    prefix_beta = torch.ones((), dtype=DTYPE)
    prefix_m = torch.zeros(3, dtype=DTYPE)
    fiber_speed, fiber_speed_jacobian = _fiber_speed_and_jacobian(
        ray_coefficient,
        time,
        compute_jacobian=compute_ray_grad,
    )
    for owner_raw, left_id_raw, right_id_raw in zip(
        word.owners.tolist(), word.left_cut_ids.tolist(), word.right_cut_ids.tolist(), strict=True
    ):
        owner = int(owner_raw)
        left_id = int(left_id_raw)
        right_id = int(right_id_raw)
        left = _cut_depth_and_jacobian(cut_coefficients, left_id, time, near=near, far=far)
        right = _cut_depth_and_jacobian(cut_coefficients, right_id, time, near=near, far=far)
        coordinate_length = right[0] - left[0]
        physical_length = fiber_speed * coordinate_length
        beta = torch.exp(-site_density[owner] * physical_length)
        tau_bar = (
            torch.dot(grad_m_total, prefix_m + prefix_beta * site_color[owner] - total_m) - total_beta * grad_beta_total
        )
        physical_length_bar = site_density[owner] * tau_bar
        coordinate_length_bar = fiber_speed * physical_length_bar
        grad_site_density[owner] += physical_length * tau_bar
        grad_site_color[owner] += prefix_beta * (1.0 - beta) * grad_m_total
        if left_id >= 0:
            grad_coefficients[left_id] -= coordinate_length_bar * left[1]
        if right_id >= 0:
            grad_coefficients[right_id] += coordinate_length_bar * right[1]
        if grad_ray_metric is not None and fiber_speed_jacobian is not None:
            grad_ray_metric += physical_length_bar * coordinate_length * fiber_speed_jacobian
        prefix_m = prefix_m + prefix_beta * (1.0 - beta) * site_color[owner]
        prefix_beta = prefix_beta * beta
    return grad_site_density, grad_site_color, grad_coefficients, grad_ray_metric


def _fiber_speed_and_jacobian(
    ray_coefficient: torch.Tensor,
    time: torch.Tensor,
    *,
    compute_jacobian: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return ``||d(t)||`` and its VJP row for ``[o0,o1,d0,d1]``.

    This is the ordinary-depth fiber Jacobian.  Keeping the fitted affine
    direction unnormalized preserves Mobius cut depths while making optical
    depth invariant to orientation-preserving affine rescaling of the depth
    coordinate.
    """

    direction = ray_coefficient[6:9] + time * ray_coefficient[9:12]
    speed = torch.linalg.vector_norm(direction)
    if float(speed.detach().cpu().item()) <= 0.0:
        raise ValueError("ray direction has zero fiber speed")
    if not compute_jacobian:
        return speed, None
    jacobian = torch.zeros_like(ray_coefficient)
    unit_direction = direction / speed
    jacobian[6:9] = unit_direction
    jacobian[9:12] = time * unit_direction
    return speed, jacobian


def _cut_depth_and_jacobian(
    cut_coefficients: dict[int, torch.Tensor],
    cut_id: int,
    time: torch.Tensor,
    *,
    near: float,
    far: float,
    denominator_epsilon: float = 1.0e-9,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cut_id == NEAR_CUT_ID:
        return torch.as_tensor(near, dtype=DTYPE), torch.zeros(4, dtype=DTYPE)
    if cut_id == FAR_CUT_ID:
        return torch.as_tensor(far, dtype=DTYPE), torch.zeros(4, dtype=DTYPE)
    if cut_id < 0 or cut_id not in cut_coefficients:
        raise ValueError(f"invalid cut id {cut_id}")
    coeff = cut_coefficients[cut_id]
    numerator = coeff[0] + coeff[1] * time
    denominator = coeff[2] + coeff[3] * time
    denominator_scale = coeff[2].abs() + (coeff[3] * time).abs()
    denominator_abs = denominator.detach().abs()
    if (
        not bool(torch.isfinite(denominator_abs).cpu().item())
        or float(denominator_scale.detach().cpu().item()) == 0.0
        or float((denominator_abs / denominator_scale.detach()).cpu().item()) <= denominator_epsilon
    ):
        raise ValueError("depth denominator is unsafe at a requested time")
    depth = numerator / denominator
    jacobian = torch.stack(
        (
            1.0 / denominator,
            time / denominator,
            -numerator / denominator.square(),
            -time * numerator / denominator.square(),
        )
    )
    return depth, jacobian


def _cut_coefficients(
    cut_coefficients: dict[int, torch.Tensor],
    cut_id: int,
    *,
    near: float,
    far: float,
) -> tuple[float, float, float, float]:
    if cut_id == NEAR_CUT_ID:
        return float(near), 0.0, 1.0, 0.0
    if cut_id == FAR_CUT_ID:
        return float(far), 0.0, 1.0, 0.0
    if cut_id < 0 or cut_id not in cut_coefficients:
        raise ValueError(f"invalid cut id {cut_id}")
    return tuple(float(value) for value in cut_coefficients[cut_id].tolist())


def _minimum_affine_cosine_margin(
    normal: torch.Tensor,
    direction_base: torch.Tensor,
    direction_slope: torch.Tensor,
    *,
    t_min: float,
    t_max: float,
) -> float:
    normal_squared = float(torch.dot(normal, normal).item())
    if normal_squared == 0.0:
        return 0.0
    a = float(torch.dot(normal, direction_base).item())
    b = float(torch.dot(normal, direction_slope).item())
    c = float(torch.dot(direction_base, direction_base).item())
    d = 2.0 * float(torch.dot(direction_base, direction_slope).item())
    e = float(torch.dot(direction_slope, direction_slope).item())
    candidates = [t_min, t_max]
    if b != 0.0:
        denominator_root = -a / b
        if t_min < denominator_root < t_max:
            candidates.append(denominator_root)
    stationary_base = 2.0 * b * c - a * d
    stationary_slope = b * d - 2.0 * a * e
    if stationary_slope != 0.0:
        stationary = -stationary_base / stationary_slope
        if t_min < stationary < t_max:
            candidates.append(stationary)
    margins = []
    for time in candidates:
        direction_squared = c + d * time + e * time * time
        if direction_squared <= 0.0:
            margins.append(0.0)
            continue
        denominator = a + b * time
        margins.append(abs(denominator) / math.sqrt(normal_squared * direction_squared))
    return min(margins)


def _minimum_affine_vector_norm(
    base: torch.Tensor,
    slope: torch.Tensor,
    *,
    t_min: float,
    t_max: float,
) -> float:
    slope_squared = float(torch.dot(slope, slope).item())
    candidates = [t_min, t_max]
    if slope_squared > 0.0:
        stationary = -float(torch.dot(base, slope).item()) / slope_squared
        if t_min < stationary < t_max:
            candidates.append(stationary)
    return min(float(torch.linalg.vector_norm(base + time * slope).item()) for time in candidates)


def _minimum_rational_difference(
    right: tuple[float, float, float, float],
    left: tuple[float, float, float, float],
    *,
    t_min: float,
    t_max: float,
) -> float:
    ar, br, cr, dr = right
    al, bl, cl, dl = left
    numerator = (
        ar * cl - al * cr,
        ar * dl + br * cl - al * dr - bl * cr,
        br * dl - bl * dr,
    )
    denominator = (cr * cl, cr * dl + dr * cl, dr * dl)
    derivative = (
        numerator[1] * denominator[0] - numerator[0] * denominator[1],
        2.0 * (numerator[2] * denominator[0] - numerator[0] * denominator[2]),
        numerator[2] * denominator[1] - numerator[1] * denominator[2],
    )
    candidates = [t_min, t_max]
    candidates.extend(root for root in _real_polynomial_roots(derivative) if t_min < root < t_max)
    values = [_poly_eval(numerator, t) / _poly_eval(denominator, t) for t in candidates]
    if not all(math.isfinite(value) for value in values):
        return -math.inf
    return min(values)


def _real_polynomial_roots(coefficients: Sequence[float], epsilon: float = 1.0e-14) -> list[float]:
    coeff = list(coefficients)
    scale = max((abs(value) for value in coeff), default=0.0)
    if scale == 0.0:
        return []
    while len(coeff) > 1 and abs(coeff[-1]) <= epsilon * scale:
        coeff.pop()
    if len(coeff) == 1:
        return []
    if len(coeff) == 2:
        return [-coeff[0] / coeff[1]]
    if len(coeff) == 3:
        c, b, a = coeff
        discriminant = b * b - 4.0 * a * c
        discriminant_scale = b * b + abs(4.0 * a * c)
        if discriminant < -epsilon * discriminant_scale:
            return []
        sqrt_discriminant = math.sqrt(max(discriminant, 0.0))
        return [(-b - sqrt_discriminant) / (2.0 * a), (-b + sqrt_discriminant) / (2.0 * a)]
    raise ValueError("root helper only supports degree <= 2 after cancellation")


def _poly_eval(coefficients: Sequence[float], value: float) -> float:
    result = 0.0
    for coefficient in reversed(coefficients):
        result = result * value + coefficient
    return result


def _validate_world_inputs(
    *,
    boundary: torch.Tensor,
    ray_coefficients: torch.Tensor,
    words: Sequence[StableCellWord],
    site_density: torch.Tensor,
    site_color: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[StableCellWord, ...]]:
    boundary = _require_f64_matrix("boundary", boundary, columns=5)
    ray_coefficients = _require_f64_matrix("ray_coefficients", ray_coefficients, columns=12)
    site_density = torch.as_tensor(site_density, dtype=DTYPE).reshape(-1)
    site_color = _require_f64_matrix("site_color", site_color, columns=3)
    _require_finite("boundary", boundary)
    _require_finite("ray_coefficients", ray_coefficients)
    _require_finite("site_density", site_density)
    _require_finite("site_color", site_color)
    if site_color.shape[0] != site_density.shape[0]:
        raise ValueError("site_density and site_color must have the same site count")
    if ray_coefficients.shape[0] == 0 or site_density.numel() == 0:
        raise ValueError("world inputs require at least one ray track and one site")
    if bool((site_density < 0.0).any().detach().cpu().item()):
        raise ValueError("site_density must be nonnegative")
    words_tuple = tuple(words)
    if len(words_tuple) != int(ray_coefficients.shape[0]):
        raise ValueError("words must contain one stable cell word per ray track")
    for word in words_tuple:
        _validate_word(word, site_count=int(site_density.numel()), boundary_count=int(boundary.shape[0]))
    return boundary, ray_coefficients, site_density, site_color, words_tuple


def _validate_word_shape(word: StableCellWord) -> None:
    tensors = _word_tensors(word)
    if any(tensor.dtype != torch.int64 or tensor.ndim != 1 for tensor in tensors):
        raise ValueError("stable word owner/cut tensors must be 1D int64")
    if not (tensors[0].shape == tensors[1].shape == tensors[2].shape):
        raise ValueError("stable word owner/cut tensors must have matching shapes")
    if tensors[0].numel() == 0:
        raise ValueError("stable word must contain at least one run")


def _validate_word(word: StableCellWord, *, site_count: int, boundary_count: int) -> None:
    _validate_word_shape(word)
    if int(word.owners.min()) < 0 or int(word.owners.max()) >= site_count:
        raise ValueError("stable word owner id is outside the shared site field")
    for cut_id in torch.cat((word.left_cut_ids, word.right_cut_ids)).tolist():
        if cut_id not in {NEAR_CUT_ID, FAR_CUT_ID} and not 0 <= int(cut_id) < boundary_count:
            raise ValueError(f"stable word cut id {cut_id} is invalid")


def _require_f64_matrix(name: str, tensor: torch.Tensor, *, columns: int) -> torch.Tensor:
    tensor_f64 = torch.as_tensor(tensor, dtype=DTYPE)
    if tensor_f64.ndim != 2 or tensor_f64.shape[1] != columns:
        raise ValueError(f"{name} must have shape [N,{columns}]")
    return tensor_f64


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    if not bool(torch.isfinite(tensor).all().detach().cpu().item()):
        raise ValueError(f"{name} must contain only finite values")


def _require_nonempty_finite_times(times: torch.Tensor | Sequence[float]) -> torch.Tensor:
    times_f64 = torch.as_tensor(times, dtype=DTYPE).reshape(-1)
    if times_f64.numel() == 0:
        raise ValueError("times must contain at least one sample")
    _require_finite("times", times_f64)
    return times_f64


def _validate_times_in_chart(
    times: torch.Tensor | Sequence[float],
    *,
    t_min: float,
    t_max: float,
) -> torch.Tensor:
    times_f64 = _require_nonempty_finite_times(times)
    if not math.isfinite(t_min) or not math.isfinite(t_max) or t_max <= t_min:
        raise ValueError("expected finite chart bounds with t_min < t_max")
    if bool(((times_f64 < t_min) | (times_f64 > t_max)).any().detach().cpu().item()):
        raise ValueError("requested time lies outside the checked chart")
    return times_f64


def _word_tensors(word: StableCellWord) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return word.owners, word.left_cut_ids, word.right_cut_ids


def _tensor_bytes(tensors: Sequence[torch.Tensor]) -> int:
    return int(sum(tensor.numel() * tensor.element_size() for tensor in tensors))
