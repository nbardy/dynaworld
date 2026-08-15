"""Structure-preserving temporal closure for ordered RGB transfer.

An optical-transfer element ``G = (beta, m)`` acts on the radiance behind it
as ``G(c) = m + beta * c``.  For ``0 < beta <= 1`` its affine-group logarithm
is represented by

``kappa = -log(beta),  v = kappa * m / (1 - beta)``.

The inverse chart is ``beta = exp(-kappa)`` and
``m = ((1 - exp(-kappa)) / kappa) * v``.  The removable singularity at
``kappa = 0`` is evaluated with a Taylor branch, including its derivative.

Fitting a fixed-rank Chebyshev atlas in this chart is an *approximate*
``O(J R + F J)`` temporal closure for ``R`` ordered runs, ``J`` nodes and
``F`` requested samples.  Its rank must follow the physical chart complexity,
not the requested frame count.  Cone checks are deliberately exposed so a
caller can split a chart or raise its rank instead of silently decoding an
unphysical polynomial overshoot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

DTYPE = torch.float64
ChartKind = Literal["raw", "lie"]


@dataclass(frozen=True)
class TemporalTransferAtlas:
    """Chebyshev coefficients for total transfer in one temporal chart."""

    t_min: float
    t_max: float
    node_times: torch.Tensor
    fit_matrix: torch.Tensor
    coefficients: torch.Tensor
    chart: ChartKind

    @property
    def rank(self) -> int:
        return int(self.node_times.numel())


@dataclass(frozen=True)
class ConeReport:
    """Fail-closed report for a transfer or Lie-chart physical cone."""

    passed: bool
    element_count: int
    violation_count: int
    maximum_violation: float


@dataclass(frozen=True)
class HardFixtureComparison:
    """Raw-versus-Lie closure errors for the hard moving-opacity fixture."""

    rank: int
    raw_max_transfer_error: float
    lie_max_transfer_error: float
    raw_max_parameter_vjp_error: float
    lie_max_parameter_vjp_error: float
    raw_transfer_cone: ConeReport
    lie_chart_cone: ConeReport
    lie_transfer_cone: ConeReport


def affine_transfer_compose(front: torch.Tensor, back: torch.Tensor) -> torch.Tensor:
    """Compose front-to-back transfers: ``front(back(c))``."""

    front_f64 = _require_transfer("front", front)
    back_f64 = _require_transfer("back", back)
    if front_f64.shape != back_f64.shape:
        raise ValueError("front and back must have identical shapes")
    beta = front_f64[..., :1] * back_f64[..., :1]
    moment = front_f64[..., 1:] + front_f64[..., :1] * back_f64[..., 1:]
    return torch.cat((beta, moment), dim=-1)


def transfer_lie_encode(transfer: torch.Tensor) -> torch.Tensor:
    """Map physical affine transfers ``[beta,m_rgb]`` to ``[kappa,v_rgb]``."""

    transfer_f64 = _require_transfer("transfer", transfer)
    beta = transfer_f64[..., :1]
    if bool(torch.any(beta <= 0.0).item()):
        raise ValueError("Lie encoding requires beta > 0")
    kappa = -torch.log(beta)
    inverse_phi, _ = _inverse_phi_and_derivative(kappa)
    return torch.cat((kappa, inverse_phi * transfer_f64[..., 1:]), dim=-1)


def transfer_lie_decode(chart: torch.Tensor) -> torch.Tensor:
    """Map ``[kappa,v_rgb]`` back to affine transfer ``[beta,m_rgb]``."""

    chart_f64 = _require_chart("chart", chart)
    kappa = chart_f64[..., :1]
    phi, _ = _phi_and_derivative(kappa)
    return torch.cat((torch.exp(-kappa), phi * chart_f64[..., 1:]), dim=-1)


def transfer_lie_encode_vjp(
    transfer: torch.Tensor,
    grad_chart: torch.Tensor,
) -> torch.Tensor:
    """Analytic VJP of :func:`transfer_lie_encode`."""

    transfer_f64 = _require_transfer("transfer", transfer)
    grad_f64 = _require_chart("grad_chart", grad_chart)
    if transfer_f64.shape != grad_f64.shape:
        raise ValueError("transfer and grad_chart must have identical shapes")
    beta = transfer_f64[..., :1]
    if bool(torch.any(beta <= 0.0).item()):
        raise ValueError("Lie encoding requires beta > 0")
    moment = transfer_f64[..., 1:]
    kappa = -torch.log(beta)
    inverse_phi, inverse_phi_prime = _inverse_phi_and_derivative(kappa)
    grad_kappa = (
        grad_f64[..., :1]
        + (grad_f64[..., 1:] * moment).sum(
            dim=-1,
            keepdim=True,
        )
        * inverse_phi_prime
    )
    grad_beta = -grad_kappa / beta
    grad_moment = inverse_phi * grad_f64[..., 1:]
    return torch.cat((grad_beta, grad_moment), dim=-1)


def transfer_lie_decode_vjp(
    chart: torch.Tensor,
    grad_transfer: torch.Tensor,
) -> torch.Tensor:
    """Analytic VJP of :func:`transfer_lie_decode`."""

    chart_f64 = _require_chart("chart", chart)
    grad_f64 = _require_transfer("grad_transfer", grad_transfer)
    if chart_f64.shape != grad_f64.shape:
        raise ValueError("chart and grad_transfer must have identical shapes")
    kappa = chart_f64[..., :1]
    velocity = chart_f64[..., 1:]
    beta = torch.exp(-kappa)
    phi, phi_prime = _phi_and_derivative(kappa)
    grad_kappa = (
        -beta * grad_f64[..., :1]
        + (grad_f64[..., 1:] * velocity).sum(
            dim=-1,
            keepdim=True,
        )
        * phi_prime
    )
    grad_velocity = phi * grad_f64[..., 1:]
    return torch.cat((grad_kappa, grad_velocity), dim=-1)


def lie_chart_word_cotangents(
    kappa_total: torch.Tensor,
    total_moment: torch.Tensor,
    grad_chart: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lower a total Lie-chart cotangent without a raw ``beta`` cotangent.

    This is the stable seam for a prefix-only ordered-word adjoint.  If
    ``r(kappa) = kappa / (1-exp(-kappa))``, then

    ``bar_m = r * bar_v`` and
    ``bar_kappa_word = bar_kappa + r' * <m,bar_v>``.

    A run with optical-depth cotangent formula
    ``<bar_m, prefix_m + prefix_beta*c_i - total_m> + bar_kappa_word``
    therefore never forms the potentially enormous raw ``bar_beta``.  Passing
    ``kappa_total`` directly also remains valid after ``exp(-kappa)`` has
    underflowed to zero.
    """

    kappa = torch.as_tensor(kappa_total, dtype=DTYPE)
    moment = torch.as_tensor(total_moment, dtype=DTYPE)
    chart_grad = _require_chart("grad_chart", grad_chart)
    if kappa.shape != chart_grad.shape[:-1]:
        raise ValueError("kappa_total must match the grad_chart batch shape")
    if moment.shape != chart_grad.shape[:-1] + (3,):
        raise ValueError("total_moment must match grad_chart[...,1:]")
    inverse_phi, inverse_phi_prime = _inverse_phi_and_derivative(kappa.unsqueeze(-1))
    grad_moment = inverse_phi * chart_grad[..., 1:]
    grad_kappa_word = chart_grad[..., 0] + inverse_phi_prime.squeeze(-1) * (moment * chart_grad[..., 1:]).sum(dim=-1)
    return grad_moment, grad_kappa_word


def chebyshev_nodes(rank: int, *, t_min: float, t_max: float) -> torch.Tensor:
    if rank < 2:
        raise ValueError("rank must be at least 2")
    _validate_interval(t_min, t_max)
    index = torch.arange(rank, dtype=DTYPE)
    normalized = torch.cos(math.pi * (2.0 * index + 1.0) / (2.0 * rank))
    return (0.5 * (t_max - t_min) * normalized + 0.5 * (t_max + t_min)).contiguous()


def chebyshev_basis(
    times: torch.Tensor,
    *,
    t_min: float,
    t_max: float,
    rank: int,
) -> torch.Tensor:
    _validate_interval(t_min, t_max)
    if rank < 1:
        raise ValueError("rank must be positive")
    times_f64 = torch.as_tensor(times, dtype=DTYPE).reshape(-1)
    if not bool(torch.isfinite(times_f64).all().item()):
        raise ValueError("times must be finite")
    x = (2.0 * times_f64 - (t_max + t_min)) / (t_max - t_min)
    columns = [torch.ones_like(x)]
    if rank > 1:
        columns.append(x)
    for _ in range(2, rank):
        columns.append(2.0 * x * columns[-1] - columns[-2])
    return torch.stack(columns, dim=1)


def fit_transfer_atlas(
    node_transfers: torch.Tensor,
    *,
    t_min: float,
    t_max: float,
    chart: ChartKind,
) -> TemporalTransferAtlas:
    """Fit node transfers in either raw or affine-Lie coordinates.

    ``node_transfers`` may have any batch prefix and must end in ``[J,4]``.
    Nodes are the canonical Chebyshev roots returned by :func:`chebyshev_nodes`.
    """

    values = _require_transfer("node_transfers", node_transfers)
    if values.ndim < 2:
        raise ValueError("node_transfers must end in [J,4]")
    rank = int(values.shape[-2])
    nodes = chebyshev_nodes(rank, t_min=t_min, t_max=t_max)
    node_basis = chebyshev_basis(nodes, t_min=t_min, t_max=t_max, rank=rank)
    fit_matrix = torch.linalg.inv(node_basis)
    chart_values = _to_chart(values, chart)
    coefficients = torch.einsum("kn,...nc->...kc", fit_matrix, chart_values)
    return TemporalTransferAtlas(
        t_min=float(t_min),
        t_max=float(t_max),
        node_times=nodes,
        fit_matrix=fit_matrix,
        coefficients=coefficients,
        chart=chart,
    )


def evaluate_transfer_atlas(
    atlas: TemporalTransferAtlas,
    times: torch.Tensor,
) -> torch.Tensor:
    """Evaluate total transfer at requested times."""

    chart_values = evaluate_transfer_atlas_chart(atlas, times)
    return _from_chart(chart_values, atlas.chart)


def evaluate_transfer_atlas_chart(
    atlas: TemporalTransferAtlas,
    times: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the stored coordinates before Lie decoding."""

    basis = chebyshev_basis(
        times,
        t_min=atlas.t_min,
        t_max=atlas.t_max,
        rank=atlas.rank,
    )
    return torch.einsum("fk,...kc->...fc", basis, atlas.coefficients)


def evaluate_transfer_atlas_vjp(
    atlas: TemporalTransferAtlas,
    times: torch.Tensor,
    grad_transfer: torch.Tensor,
) -> torch.Tensor:
    """Analytic atlas VJP returning cotangents for the fitted node transfers."""

    chart_values = evaluate_transfer_atlas_chart(atlas, times)
    grad_output = _require_transfer("grad_transfer", grad_transfer)
    if grad_output.shape != chart_values.shape:
        raise ValueError("grad_transfer must match the evaluated transfer shape")
    grad_chart_values = grad_output if atlas.chart == "raw" else transfer_lie_decode_vjp(chart_values, grad_output)
    basis = chebyshev_basis(
        times,
        t_min=atlas.t_min,
        t_max=atlas.t_max,
        rank=atlas.rank,
    )
    grad_coefficients = torch.einsum("fk,...fc->...kc", basis, grad_chart_values)
    grad_node_chart = torch.einsum("kn,...kc->...nc", atlas.fit_matrix, grad_coefficients)
    if atlas.chart == "raw":
        return grad_node_chart
    node_chart = torch.einsum("nk,...kc->...nc", torch.linalg.inv(atlas.fit_matrix), atlas.coefficients)
    node_transfer = transfer_lie_decode(node_chart)
    return transfer_lie_encode_vjp(node_transfer, grad_node_chart)


def check_lie_chart_cone(chart: torch.Tensor, *, tolerance: float = 1.0e-12) -> ConeReport:
    """Check ``kappa >= 0`` and ``0 <= v_c <= kappa`` for bounded RGB."""

    chart_f64 = _require_chart("chart", chart)
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    kappa = chart_f64[..., :1]
    velocity = chart_f64[..., 1:]
    violation = torch.maximum(
        torch.maximum((-kappa).clamp_min(0.0).expand_as(velocity), (-velocity).clamp_min(0.0)),
        (velocity - kappa).clamp_min(0.0),
    )
    return _cone_report(violation, tolerance=tolerance)


def check_transfer_cone(transfer: torch.Tensor, *, tolerance: float = 1.0e-12) -> ConeReport:
    """Check ``0 <= beta <= 1`` and ``0 <= m_c <= 1-beta``."""

    transfer_f64 = _require_transfer("transfer", transfer)
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    beta = transfer_f64[..., :1]
    moment = transfer_f64[..., 1:]
    beta_violation = torch.maximum((-beta).clamp_min(0.0), (beta - 1.0).clamp_min(0.0))
    moment_violation = torch.maximum((-moment).clamp_min(0.0), (moment - (1.0 - beta)).clamp_min(0.0))
    violation = torch.maximum(beta_violation.expand_as(moment), moment_violation)
    return _cone_report(violation, tolerance=tolerance)


def hard_opacity_moving_boundary_transfer(
    times: torch.Tensor,
    parameters: torch.Tensor | None = None,
) -> torch.Tensor:
    """Faithful scalar form of the existing hard moving-boundary fixture.

    Parameters are ``[density, boundary_intercept, boundary_slope, r, g, b]``.
    The ray has unit fiber speed, the near plane is ``0.05``, and the second
    cell is transparent.  Defaults reproduce ``test_fixed_rank_atlas...`` in
    ``test_compiled_transfer_adjoint.py``.
    """

    times_f64 = torch.as_tensor(times, dtype=DTYPE).reshape(-1)
    if parameters is None:
        parameters = torch.tensor([50.0, 1.0, 0.9, 1.0, 0.0, 0.0], dtype=DTYPE)
    params = torch.as_tensor(parameters, dtype=DTYPE).reshape(6)
    density, intercept, slope = params[:3]
    color = params[3:]
    length = intercept + slope * times_f64 - 0.05
    if bool(torch.any(length <= 0.0).detach().item()):
        raise ValueError("hard fixture left the fixed positive-length chart")
    beta = torch.exp(-density * length)
    moment = (-torch.expm1(-density * length)).unsqueeze(1) * color.unsqueeze(0)
    return torch.cat((beta.unsqueeze(1), moment), dim=1)


def compare_hard_fixture_charts(
    ranks: tuple[int, ...] = (2, 4, 8, 16, 32),
    *,
    validation_count: int = 257,
) -> tuple[HardFixtureComparison, ...]:
    """Falsification report for raw and Lie closure on the hard fixture."""

    if validation_count < 2:
        raise ValueError("validation_count must be at least 2")
    times = torch.linspace(-1.0, 1.0, validation_count, dtype=DTYPE)
    exact_parameters = torch.tensor(
        [50.0, 1.0, 0.9, 1.0, 0.0, 0.0],
        dtype=DTYPE,
        requires_grad=True,
    )
    exact_transfer = hard_opacity_moving_boundary_transfer(times, exact_parameters)
    phase = torch.linspace(0.0, 2.0 * math.pi, validation_count, dtype=DTYPE)
    cotangent = torch.stack(
        (
            0.17 + 0.11 * torch.cos(phase),
            0.23 + 0.07 * torch.sin(phase),
            -0.13 + 0.05 * torch.cos(2.0 * phase),
            0.09 - 0.04 * torch.sin(3.0 * phase),
        ),
        dim=1,
    ) / float(validation_count)
    exact_parameter_vjp = torch.autograd.grad(
        (exact_transfer * cotangent).sum(),
        exact_parameters,
    )[0]

    reports = []
    for rank in ranks:
        nodes = chebyshev_nodes(rank, t_min=-1.0, t_max=1.0)
        for_report_parameters = exact_parameters.detach().clone().requires_grad_(True)
        node_transfer = hard_opacity_moving_boundary_transfer(nodes, for_report_parameters)
        chart_results: dict[str, tuple[torch.Tensor, torch.Tensor, TemporalTransferAtlas]] = {}
        for chart in ("raw", "lie"):
            atlas = fit_transfer_atlas(
                node_transfer.detach(),
                t_min=-1.0,
                t_max=1.0,
                chart=chart,
            )
            prediction = evaluate_transfer_atlas(atlas, times)
            grad_nodes = evaluate_transfer_atlas_vjp(atlas, times, cotangent)
            parameter_vjp = torch.autograd.grad(
                node_transfer,
                for_report_parameters,
                grad_outputs=grad_nodes,
                retain_graph=chart == "raw",
            )[0]
            chart_results[chart] = (prediction, parameter_vjp, atlas)

        raw_prediction, raw_vjp, raw_atlas = chart_results["raw"]
        lie_prediction, lie_vjp, lie_atlas = chart_results["lie"]
        reports.append(
            HardFixtureComparison(
                rank=rank,
                raw_max_transfer_error=_max_abs(raw_prediction - exact_transfer.detach()),
                lie_max_transfer_error=_max_abs(lie_prediction - exact_transfer.detach()),
                raw_max_parameter_vjp_error=_max_abs(raw_vjp - exact_parameter_vjp.detach()),
                lie_max_parameter_vjp_error=_max_abs(lie_vjp - exact_parameter_vjp.detach()),
                raw_transfer_cone=check_transfer_cone(raw_prediction),
                lie_chart_cone=check_lie_chart_cone(evaluate_transfer_atlas_chart(lie_atlas, times)),
                lie_transfer_cone=check_transfer_cone(lie_prediction),
            )
        )
        del raw_atlas
    return tuple(reports)


def closure_operation_counts(*, run_count: int, rank: int, frame_count: int) -> dict[str, int]:
    """Expose the idealized temporal work terms without claiming wall time."""

    if run_count < 0 or rank < 2 or frame_count < 0:
        raise ValueError("expected run_count >= 0, rank >= 2 and frame_count >= 0")
    return {
        "world_compile_interactions": int(rank * run_count),
        "sample_basis_interactions": int(frame_count * rank),
        "total_interactions": int(rank * run_count + frame_count * rank),
    }


def _phi_and_derivative(kappa: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(1-exp(-k))/k`` and its derivative, including ``k=0``."""

    small = kappa.abs() < 1.0e-4
    k2 = kappa * kappa
    k3 = k2 * kappa
    k4 = k3 * kappa
    k5 = k4 * kappa
    k6 = k5 * kappa
    series = 1.0 - kappa / 2.0 + k2 / 6.0 - k3 / 24.0 + k4 / 120.0 - k5 / 720.0 + k6 / 5040.0
    series_prime = -0.5 + kappa / 3.0 - k2 / 8.0 + k3 / 30.0 - k4 / 144.0 + k5 / 840.0
    safe_kappa = torch.where(small, torch.ones_like(kappa), kappa)
    numerator = -torch.expm1(-kappa)
    direct = numerator / safe_kappa
    direct_prime = (safe_kappa * torch.exp(-kappa) - numerator) / safe_kappa.square()
    return torch.where(small, series, direct), torch.where(small, series_prime, direct_prime)


def _inverse_phi_and_derivative(kappa: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``k/(1-exp(-k))`` and its derivative, including ``k=0``."""

    small = kappa.abs() < 1.0e-4
    k2 = kappa * kappa
    k3 = k2 * kappa
    k4 = k3 * kappa
    k5 = k4 * kappa
    k6 = k5 * kappa
    series = 1.0 + kappa / 2.0 + k2 / 12.0 - k4 / 720.0 + k6 / 30240.0
    series_prime = 0.5 + kappa / 6.0 - k3 / 180.0 + k5 / 5040.0
    denominator = -torch.expm1(-kappa)
    safe_denominator = torch.where(small, torch.ones_like(denominator), denominator)
    direct = kappa / safe_denominator
    direct_prime = (denominator - kappa * torch.exp(-kappa)) / safe_denominator.square()
    return torch.where(small, series, direct), torch.where(small, series_prime, direct_prime)


def _to_chart(transfer: torch.Tensor, chart: ChartKind) -> torch.Tensor:
    if chart == "raw":
        return transfer
    if chart == "lie":
        return transfer_lie_encode(transfer)
    raise ValueError("chart must be 'raw' or 'lie'")


def _from_chart(chart_values: torch.Tensor, chart: ChartKind) -> torch.Tensor:
    if chart == "raw":
        return chart_values
    if chart == "lie":
        return transfer_lie_decode(chart_values)
    raise ValueError("chart must be 'raw' or 'lie'")


def _require_transfer(name: str, value: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=DTYPE)
    if tensor.ndim < 1 or tensor.shape[-1] != 4:
        raise ValueError(f"{name} must end in four channels [beta,m_rgb]")
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must be finite")
    return tensor


def _require_chart(name: str, value: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=DTYPE)
    if tensor.ndim < 1 or tensor.shape[-1] != 4:
        raise ValueError(f"{name} must end in four channels [kappa,v_rgb]")
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must be finite")
    return tensor


def _validate_interval(t_min: float, t_max: float) -> None:
    if not math.isfinite(t_min) or not math.isfinite(t_max) or t_max <= t_min:
        raise ValueError("expected a finite interval with t_max > t_min")


def _cone_report(violation: torch.Tensor, *, tolerance: float) -> ConeReport:
    violating = violation > tolerance
    return ConeReport(
        passed=not bool(torch.any(violating).item()),
        element_count=int(violation.numel()),
        violation_count=int(violating.sum().item()),
        maximum_violation=_max_abs(violation),
    )


def _max_abs(value: torch.Tensor) -> float:
    if value.numel() == 0:
        return 0.0
    return float(value.detach().abs().max().cpu().item())
