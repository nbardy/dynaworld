"""Template-free temporal schedule for compact WorldFoam track blocks.

The adaptive CPU compiler returns a full ``P``-track atlas because its
coefficients, node charts, sparse depth coefficients, and words are useful to
the reference evaluator.  A streamed native/training step does not need to
retain those tensors after chart selection.  It needs only the temporal chart
partition and interpolation schedule, whose storage is ``O(sum(J_c^2))`` and
independent of the global track count.

This module copies that small schedule out of a selected atlas and assigns it
a content digest.  Compact ``B_p`` blocks can then be compiled independently
and a global gradient ledger can compare the digest instead of holding the
full ``P``-track atlas alive.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Literal

import torch
from compiled_lie_world_adjoint import (
    AdaptiveCompiledLieWorldAtlas,
    AdaptiveLieChartSelection,
    AdaptiveLieWorldCompilePolicy,
)
from transfer_lie_chart import ChartKind, chebyshev_basis, chebyshev_nodes

_Scalar = bool | int | float | str

LINEAR_SAMPLE_WEIGHT_EVALUATION = "verified_fit_derived_second_form_barycentric"


@dataclass(frozen=True)
class SampleToNodeWeightResult:
    """One bounded sample block's interpolation weights and cost provenance."""

    weights: torch.Tensor
    evaluation: Literal[
        "verified_fit_derived_second_form_barycentric",
        "verified_fit_derived_second_form_barycentric_with_dense_fallback",
    ]
    sample_count: int
    node_count: int
    linear_weight_interactions: int
    dense_fallback_interactions: int
    exact_node_row_count: int
    dense_fallback_row_count: int


@dataclass(frozen=True)
class CompactLieChartSchedule:
    """One immutable interval/rank/interpolation schedule."""

    t_min: float
    t_max: float
    near: float
    far: float
    node_times: torch.Tensor
    fit_matrix: torch.Tensor
    barycentric_weights: torch.Tensor
    chart: ChartKind
    tensor_signatures: tuple[tuple[object, ...], ...]

    @property
    def sample_weight_evaluation(self) -> str:
        return LINEAR_SAMPLE_WEIGHT_EVALUATION

    @property
    def node_count(self) -> int:
        return int(self.node_times.numel())

    @property
    def resident_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.node_times, self.fit_matrix, self.barycentric_weights)
        )

    def sample_to_node_weights(self, times: torch.Tensor) -> SampleToNodeWeightResult:
        """Evaluate the verified second-form interpolant in ``O(K J)`` work."""

        self.assert_current()
        return fit_derived_sample_to_node_weights(
            times,
            t_min=self.t_min,
            t_max=self.t_max,
            node_times=self.node_times,
            fit_matrix=self.fit_matrix,
            barycentric_weights=self.barycentric_weights,
        )

    def assert_current(self) -> None:
        if not math.isfinite(self.t_min) or not math.isfinite(self.t_max) or self.t_max <= self.t_min:
            raise ValueError("compact chart schedule interval must be finite and increasing")
        if not math.isfinite(self.near) or not math.isfinite(self.far) or self.far <= self.near:
            raise ValueError("compact chart schedule near/far interval must be finite and increasing")
        if self.chart not in {"raw", "lie"}:
            raise ValueError("compact chart schedule has an unsupported transfer chart")
        if (
            self.node_count < 2
            or tuple(self.fit_matrix.shape) != (self.node_count, self.node_count)
            or tuple(self.barycentric_weights.shape) != (self.node_count,)
        ):
            raise ValueError("compact chart schedule node times and fit matrix have inconsistent ranks")
        if (
            self.node_times.device.type != "cpu"
            or self.fit_matrix.device.type != "cpu"
            or self.barycentric_weights.device.type != "cpu"
            or self.node_times.dtype != torch.float64
            or self.fit_matrix.dtype != torch.float64
            or self.barycentric_weights.dtype != torch.float64
            or not self.node_times.is_contiguous()
            or not self.fit_matrix.is_contiguous()
            or not self.barycentric_weights.is_contiguous()
        ):
            raise ValueError("compact chart schedule tensors must be contiguous CPU float64")
        if not all(
            bool(torch.isfinite(tensor).all().item())
            for tensor in (self.node_times, self.fit_matrix, self.barycentric_weights)
        ):
            raise ValueError("compact chart schedule tensors must be finite")
        if not bool(torch.all(self.barycentric_weights != 0.0).item()):
            raise ValueError("compact chart schedule barycentric weights must be nonzero")
        expected_weights = self.fit_matrix[-1] / self.fit_matrix[-1].abs().max()
        if not torch.equal(self.barycentric_weights, expected_weights):
            raise ValueError("compact chart schedule barycentric weights changed provenance")
        if tuple(
            _tensor_signature(tensor)
            for tensor in (self.node_times, self.fit_matrix, self.barycentric_weights)
        ) != self.tensor_signatures:
            raise ValueError("compact chart schedule tensors changed after extraction")


@dataclass(frozen=True)
class CompactLieChartSpec:
    """Predeclared chart shape that does not require any track-sized atlas."""

    t_min: float
    t_max: float
    near: float
    far: float
    node_count: int
    chart: ChartKind = "lie"


@dataclass(frozen=True)
class CompactLieWorldSchedule:
    """Small global chart identity with no track-sized atlas tensors."""

    global_track_count: int
    charts: tuple[CompactLieChartSchedule, ...]
    selections: tuple[AdaptiveLieChartSelection, ...]
    policy: AdaptiveLieWorldCompilePolicy
    supplied_word_ordering_facts: tuple[tuple[str, _Scalar], ...]
    selection_provenance: str
    generation_digest: str

    @property
    def chart_count(self) -> int:
        return len(self.charts)

    @property
    def total_node_count(self) -> int:
        return sum(chart.node_count for chart in self.charts)

    @property
    def resident_bytes(self) -> int:
        return sum(chart.resident_bytes for chart in self.charts)

    @property
    def selection_signature(self) -> tuple[tuple[float, float, int], ...]:
        return tuple((chart.t_min, chart.t_max, chart.node_count) for chart in self.charts)

    @property
    def supplied_word_ordering_check(self) -> dict[str, _Scalar]:
        return dict(self.supplied_word_ordering_facts)

    def assert_current(self) -> None:
        if self.global_track_count < 1:
            raise ValueError("compact schedule global_track_count must be positive")
        if not self.charts:
            raise ValueError("compact schedule must contain at least one chart")
        for previous, chart in zip(self.charts[:-1], self.charts[1:], strict=True):
            if previous.t_max != chart.t_min:
                raise ValueError("compact chart schedule must be ordered and exactly contiguous")
        for chart in self.charts:
            chart.assert_current()
        first = self.charts[0]
        if any(chart.near != first.near or chart.far != first.far for chart in self.charts[1:]):
            raise ValueError("compact chart schedules must share one physical near/far interval")
        if self.selections and len(self.selections) != len(self.charts):
            raise ValueError("compact schedule selections must be empty or cover every chart")
        if (
            self.selections
            and tuple((selection.t_min, selection.t_max, selection.node_count) for selection in self.selections)
            != self.selection_signature
        ):
            raise ValueError("compact schedule selections disagree with its chart partition")
        if not self.selection_provenance.strip():
            raise ValueError("compact schedule selection provenance must be nonempty")
        expected = _schedule_generation_digest(
            global_track_count=self.global_track_count,
            charts=self.charts,
            selections=self.selections,
            policy=self.policy,
            supplied_word_ordering_facts=self.supplied_word_ordering_facts,
            selection_provenance=self.selection_provenance,
        )
        if self.generation_digest != expected:
            raise ValueError("compact chart schedule generation digest changed")


def compact_lie_world_schedule_from_atlas(
    atlas: AdaptiveCompiledLieWorldAtlas,
) -> CompactLieWorldSchedule:
    """Copy only temporal chart metadata out of a full selected atlas."""

    if not atlas.charts:
        raise ValueError("adaptive atlas must contain at least one chart")
    if atlas.track_count < 1:
        raise ValueError("adaptive atlas track count must be positive")
    first = atlas.charts[0]
    for previous, chart in zip(atlas.charts[:-1], atlas.charts[1:], strict=True):
        if previous.transfer_atlas.t_max != chart.transfer_atlas.t_min:
            raise ValueError("adaptive atlas charts must be ordered and exactly contiguous")
    for chart in atlas.charts:
        if chart.track_count != atlas.track_count:
            raise ValueError("adaptive atlas charts must share one global track count")
        if chart.transfer_atlas.chart != first.transfer_atlas.chart:
            raise ValueError("adaptive atlas charts must share one transfer chart kind")

    charts = tuple(_copy_chart_schedule(chart) for chart in atlas.charts)
    ordering_facts = tuple(
        sorted((str(name), _require_scalar(value)) for name, value in atlas.supplied_word_ordering_check.items())
    )
    digest = _schedule_generation_digest(
        global_track_count=atlas.track_count,
        charts=charts,
        selections=atlas.selections,
        policy=atlas.policy,
        supplied_word_ordering_facts=ordering_facts,
        selection_provenance="extracted_from_selected_adaptive_atlas",
    )
    schedule = CompactLieWorldSchedule(
        global_track_count=atlas.track_count,
        charts=charts,
        selections=atlas.selections,
        policy=atlas.policy,
        supplied_word_ordering_facts=ordering_facts,
        selection_provenance="extracted_from_selected_adaptive_atlas",
        generation_digest=digest,
    )
    schedule.assert_current()
    return schedule


def compact_lie_world_schedule_from_specs(
    chart_specs: tuple[CompactLieChartSpec, ...],
    *,
    global_track_count: int,
    selection_provenance: str,
    policy: AdaptiveLieWorldCompilePolicy | None = None,
) -> CompactLieWorldSchedule:
    """Build a production-count schedule without constructing a full atlas.

    This is a schedule constructor, not a rank/topology certificate.  A caller
    must give a durable provenance label (for example a checked-in protocol
    digest), and every compact block must still pass the appropriate strict or
    owner-topology binding before native execution.  Predeclaring ranks this
    way removes the otherwise circular need to allocate a ``P``-track atlas
    merely to describe an ``O(sum(J_c^2))`` interpolation schedule.
    """

    if global_track_count < 1:
        raise ValueError("global_track_count must be positive")
    if not chart_specs:
        raise ValueError("chart_specs must be nonempty")
    if not selection_provenance.strip():
        raise ValueError("selection_provenance must be nonempty")
    charts = tuple(_chart_schedule_from_spec(spec) for spec in chart_specs)
    normalized_policy = (
        AdaptiveLieWorldCompilePolicy(node_count_schedule=tuple(dict.fromkeys(spec.node_count for spec in chart_specs)))
        if policy is None
        else policy
    )
    if any(spec.node_count not in normalized_policy.node_count_schedule for spec in chart_specs):
        raise ValueError("compact chart spec rank is absent from the declared compile policy")
    digest = _schedule_generation_digest(
        global_track_count=global_track_count,
        charts=charts,
        selections=(),
        policy=normalized_policy,
        supplied_word_ordering_facts=(),
        selection_provenance=selection_provenance,
    )
    schedule = CompactLieWorldSchedule(
        global_track_count=global_track_count,
        charts=charts,
        selections=(),
        policy=normalized_policy,
        supplied_word_ordering_facts=(),
        selection_provenance=selection_provenance,
        generation_digest=digest,
    )
    schedule.assert_current()
    return schedule


def _copy_chart_schedule(chart: object) -> CompactLieChartSchedule:
    transfer = chart.transfer_atlas
    node_times = transfer.node_times.detach().to(device="cpu", dtype=torch.float64).clone().contiguous()
    fit_matrix = transfer.fit_matrix.detach().to(device="cpu", dtype=torch.float64).clone().contiguous()
    barycentric_weights = certify_fit_derived_barycentric_weights(
        node_times,
        fit_matrix,
        t_min=float(transfer.t_min),
        t_max=float(transfer.t_max),
    )
    return CompactLieChartSchedule(
        t_min=float(transfer.t_min),
        t_max=float(transfer.t_max),
        near=float(chart.near),
        far=float(chart.far),
        node_times=node_times,
        fit_matrix=fit_matrix,
        barycentric_weights=barycentric_weights,
        chart=transfer.chart,
        tensor_signatures=tuple(
            _tensor_signature(tensor)
            for tensor in (node_times, fit_matrix, barycentric_weights)
        ),
    )


def _chart_schedule_from_spec(spec: CompactLieChartSpec) -> CompactLieChartSchedule:
    if spec.node_count < 2:
        raise ValueError("compact chart spec node_count must be at least two")
    if spec.chart != "lie":
        raise ValueError("fixed-word WorldFoam schedule specs currently require the affine-Lie chart")
    node_times = chebyshev_nodes(
        spec.node_count,
        t_min=spec.t_min,
        t_max=spec.t_max,
    ).contiguous()
    fit_matrix = torch.linalg.inv(
        chebyshev_basis(
            node_times,
            t_min=spec.t_min,
            t_max=spec.t_max,
            rank=spec.node_count,
        )
    ).contiguous()
    barycentric_weights = certify_fit_derived_barycentric_weights(
        node_times,
        fit_matrix,
        t_min=float(spec.t_min),
        t_max=float(spec.t_max),
    )
    return CompactLieChartSchedule(
        t_min=float(spec.t_min),
        t_max=float(spec.t_max),
        near=float(spec.near),
        far=float(spec.far),
        node_times=node_times,
        fit_matrix=fit_matrix,
        barycentric_weights=barycentric_weights,
        chart=spec.chart,
        tensor_signatures=tuple(
            _tensor_signature(tensor)
            for tensor in (node_times, fit_matrix, barycentric_weights)
        ),
    )


def _schedule_generation_digest(
    *,
    global_track_count: int,
    charts: tuple[CompactLieChartSchedule, ...],
    selections: tuple[AdaptiveLieChartSelection, ...],
    policy: AdaptiveLieWorldCompilePolicy,
    supplied_word_ordering_facts: tuple[tuple[str, _Scalar], ...],
    selection_provenance: str,
) -> str:
    payload = {
        "schema": "worldfoam-compact-lie-world-schedule-v2",
        "global_track_count": global_track_count,
        "charts": [
            {
                "t_min": chart.t_min,
                "t_max": chart.t_max,
                "near": chart.near,
                "far": chart.far,
                "node_count": chart.node_count,
                "chart": chart.chart,
                "node_times": _tensor_content_digest(chart.node_times),
                "fit_matrix": _tensor_content_digest(chart.fit_matrix),
                "barycentric_weights": _tensor_content_digest(chart.barycentric_weights),
                "sample_weight_evaluation": chart.sample_weight_evaluation,
            }
            for chart in charts
        ],
        "selections": [asdict(selection) for selection in selections],
        "policy": asdict(policy),
        "supplied_word_ordering_facts": supplied_word_ordering_facts,
        "selection_provenance": selection_provenance,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def certify_fit_derived_barycentric_weights(
    node_times: torch.Tensor,
    fit_matrix: torch.Tensor,
    *,
    t_min: float,
    t_max: float,
) -> torch.Tensor:
    """Certify a Chebyshev fit and cache barycentric weights for its actual nodes.

    The final row of the inverse Chebyshev Vandermonde matrix is proportional
    to the polynomial barycentric weights: it is the highest-order Chebyshev
    coefficient of every cardinal polynomial.  Deriving the weights from the
    stored fit, rather than from ideal analytic roots, is important when a
    large time offset rounds the physical node locations.

    This ``O(J^3)`` identity check runs only when a compact schedule or sealed
    native binding is built.  Sample blocks use :func:`fit_derived_sample_to_node_weights`
    and perform ``O(K J)`` work in the verified common case.
    """

    if not math.isfinite(t_min) or not math.isfinite(t_max) or t_max <= t_min:
        raise ValueError("barycentric schedule requires a finite increasing interval")
    nodes = torch.as_tensor(node_times, dtype=torch.float64, device="cpu").reshape(-1).detach()
    fit = torch.as_tensor(fit_matrix, dtype=torch.float64, device="cpu").detach()
    rank = int(nodes.numel())
    if rank < 2 or tuple(fit.shape) != (rank, rank):
        raise ValueError("barycentric schedule requires J nodes and a J x J fit matrix")
    if not bool(torch.isfinite(nodes).all().item()) or not bool(torch.isfinite(fit).all().item()):
        raise ValueError("barycentric schedule tensors must be finite")
    node_basis = chebyshev_basis(nodes, t_min=t_min, t_max=t_max, rank=rank)
    normalized_nodes = (2.0 * nodes - (t_max + t_min)) / (t_max - t_min)
    if int(torch.unique(normalized_nodes).numel()) != rank:
        raise ValueError("interpolation nodes collide after Chebyshev normalization")
    identity_error = (node_basis @ fit - torch.eye(rank, dtype=torch.float64)).abs().max()
    identity_tolerance = 512.0 * torch.finfo(torch.float64).eps * max(1, rank)
    if not math.isfinite(float(identity_error.item())) or float(identity_error.item()) > identity_tolerance:
        raise ValueError("fit matrix is not a verified inverse for the stored interpolation nodes")
    weights = fit[-1].clone().contiguous()
    scale = weights.abs().max()
    if not bool(torch.isfinite(scale).item()) or float(scale.item()) == 0.0:
        raise ValueError("fit-derived barycentric weights have no finite nonzero scale")
    weights.div_(scale)
    if not bool(torch.isfinite(weights).all().item()) or not bool(torch.all(weights != 0.0).item()):
        raise ValueError("fit-derived barycentric weights must be finite and nonzero")
    if rank > 1 and not bool(torch.all(weights[:-1] * weights[1:] < 0.0).item()):
        raise ValueError("fit-derived barycentric weights do not alternate signs")
    return weights


def dense_sample_to_node_weights(
    times: torch.Tensor,
    *,
    t_min: float,
    t_max: float,
    fit_matrix: torch.Tensor,
) -> torch.Tensor:
    """Retained ``O(K J^2)`` Chebyshev oracle and exceptional-row fallback."""

    fit = torch.as_tensor(fit_matrix, dtype=torch.float64, device="cpu").detach()
    if fit.ndim != 2 or fit.shape[0] != fit.shape[1] or fit.shape[0] < 2:
        raise ValueError("dense sample weights require a square rank-at-least-two fit matrix")
    times_f64 = torch.as_tensor(times, dtype=torch.float64, device="cpu").reshape(-1).detach()
    if times_f64.numel() < 1 or not bool(torch.isfinite(times_f64).all().item()):
        raise ValueError("sample times must be nonempty and finite")
    weights = chebyshev_basis(
        times_f64,
        t_min=t_min,
        t_max=t_max,
        rank=int(fit.shape[0]),
    ) @ fit
    if not bool(torch.isfinite(weights).all().item()):
        raise ValueError("dense sample-to-node fallback produced nonfinite weights")
    row_sum_tolerance = 512.0 * torch.finfo(torch.float64).eps * max(1, int(fit.shape[0]))
    if bool(torch.any((weights.sum(dim=1) - 1.0).abs() > row_sum_tolerance).item()):
        raise ValueError("dense sample-to-node fallback violates the cardinal partition of unity")
    return weights.contiguous()


def fit_derived_sample_to_node_weights(
    times: torch.Tensor,
    *,
    t_min: float,
    t_max: float,
    node_times: torch.Tensor,
    fit_matrix: torch.Tensor,
    barycentric_weights: torch.Tensor,
) -> SampleToNodeWeightResult:
    """Evaluate actual-node cardinal weights with second-form barycentric interpolation.

    Exact node rows are emitted as exact one-hot vectors.  A nonfinite or
    cancellation-dominated denominator is evaluated by the retained dense
    Chebyshev oracle for only that exceptional row; a nonfinite dense result
    raises instead of entering a native launch.
    """

    if not math.isfinite(t_min) or not math.isfinite(t_max) or t_max <= t_min:
        raise ValueError("sample weights require a finite increasing interval")
    times_f64 = torch.as_tensor(times, dtype=torch.float64, device="cpu").reshape(-1).detach()
    nodes = torch.as_tensor(node_times, dtype=torch.float64, device="cpu").reshape(-1).detach()
    fit = torch.as_tensor(fit_matrix, dtype=torch.float64, device="cpu").detach()
    barycentric = torch.as_tensor(
        barycentric_weights,
        dtype=torch.float64,
        device="cpu",
    ).reshape(-1).detach()
    sample_count = int(times_f64.numel())
    node_count = int(nodes.numel())
    if sample_count < 1 or not bool(torch.isfinite(times_f64).all().item()):
        raise ValueError("sample times must be nonempty and finite")
    if (
        node_count < 2
        or tuple(fit.shape) != (node_count, node_count)
        or tuple(barycentric.shape) != (node_count,)
    ):
        raise ValueError("sample weights require consistent node, fit, and barycentric ranks")
    if not all(
        bool(torch.isfinite(tensor).all().item())
        for tensor in (nodes, fit, barycentric)
    ) or not bool(torch.all(barycentric != 0.0).item()):
        raise ValueError("sample-weight schedule tensors must be finite with nonzero barycentric weights")
    expected_barycentric = fit[-1] / fit[-1].abs().max()
    if not torch.equal(barycentric, expected_barycentric):
        raise ValueError("sample-weight barycentric vector does not come from the certified fit matrix")

    normalized_times = (2.0 * times_f64 - (t_max + t_min)) / (t_max - t_min)
    normalized_nodes = (2.0 * nodes - (t_max + t_min)) / (t_max - t_min)
    interval_tolerance = 32.0 * torch.finfo(torch.float64).eps
    if bool(
        torch.any(normalized_times < -1.0 - interval_tolerance).item()
        or torch.any(normalized_times > 1.0 + interval_tolerance).item()
    ):
        raise ValueError("sample times leave the interpolation chart interval")
    delta = normalized_times[:, None] - normalized_nodes[None, :]
    exact = delta == 0.0
    exact_rows = exact.any(dim=1)
    if bool((exact.sum(dim=1) > 1).any().item()):
        raise ValueError("interpolation schedule has duplicate normalized nodes")

    safe_delta = torch.where(exact, torch.ones_like(delta), delta)
    raw_terms = barycentric[None, :] / safe_delta
    row_scale = raw_terms.abs().max(dim=1).values
    safe_scale = torch.where(row_scale > 0.0, row_scale, torch.ones_like(row_scale))
    terms = raw_terms / safe_scale[:, None]
    denominator = terms.sum(dim=1)
    term_scale = terms.abs().sum(dim=1)
    roundoff_scale = torch.maximum(
        torch.ones_like(delta),
        torch.maximum(normalized_times[:, None].abs(), normalized_nodes[None, :].abs()),
    )
    near_nonexact_rows = ((delta.abs() <= 16.0 * torch.finfo(torch.float64).eps * roundoff_scale) & ~exact).any(
        dim=1
    )
    condition_floor = 64.0 * torch.finfo(torch.float64).eps * node_count
    stable_rows = (~exact_rows) & (~near_nonexact_rows) & torch.isfinite(raw_terms).all(dim=1)
    stable_rows &= torch.isfinite(terms).all(dim=1) & torch.isfinite(denominator) & torch.isfinite(term_scale)
    stable_rows &= denominator.abs() > condition_floor * term_scale
    fallback_rows = (~exact_rows) & (~stable_rows)

    result = torch.empty((sample_count, node_count), dtype=torch.float64)
    if bool(exact_rows.any().item()):
        exact_ids = exact[exact_rows].to(dtype=torch.float64)
        result[exact_rows] = exact_ids
    if bool(stable_rows.any().item()):
        result[stable_rows] = terms[stable_rows] / denominator[stable_rows, None]
    fallback_count = int(fallback_rows.sum().item())
    if fallback_count:
        result[fallback_rows] = dense_sample_to_node_weights(
            times_f64[fallback_rows],
            t_min=t_min,
            t_max=t_max,
            fit_matrix=fit,
        )
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError("sample-to-node interpolation produced nonfinite weights")
    return SampleToNodeWeightResult(
        weights=result.contiguous(),
        evaluation=(
            "verified_fit_derived_second_form_barycentric_with_dense_fallback"
            if fallback_count
            else LINEAR_SAMPLE_WEIGHT_EVALUATION
        ),
        sample_count=sample_count,
        node_count=node_count,
        linear_weight_interactions=sample_count * node_count,
        dense_fallback_interactions=fallback_count * node_count * node_count,
        exact_node_row_count=int(exact_rows.sum().item()),
        dense_fallback_row_count=fallback_count,
    )


def _tensor_signature(tensor: torch.Tensor) -> tuple[object, ...]:
    return (
        tensor.untyped_storage().data_ptr(),
        tensor._version,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.storage_offset(),
        str(tensor.dtype),
        str(tensor.device),
    )


def _tensor_content_digest(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().to(device="cpu").contiguous()
    header = json.dumps(
        {"dtype": str(cpu.dtype), "shape": list(cpu.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + b"\0" + cpu.numpy().tobytes(order="C")).hexdigest()


def _require_scalar(value: object) -> _Scalar:
    if isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("compact schedule ordering facts must be finite")
        return value
    raise ValueError("compact schedule ordering facts must contain only JSON scalars")


__all__ = [
    "CompactLieChartSpec",
    "CompactLieChartSchedule",
    "CompactLieWorldSchedule",
    "LINEAR_SAMPLE_WEIGHT_EVALUATION",
    "SampleToNodeWeightResult",
    "certify_fit_derived_barycentric_weights",
    "compact_lie_world_schedule_from_atlas",
    "compact_lie_world_schedule_from_specs",
    "dense_sample_to_node_weights",
    "fit_derived_sample_to_node_weights",
]
