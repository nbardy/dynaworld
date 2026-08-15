"""Frame-independent CPU transfer program over all kinetic owner charts.

The outer program retains exact rational event guards and dispatches every
binary64 sample with the right-continuous half-open convention.  Each owner
chart is compiled exactly once at a fixed number of compact transfer nodes.
Requested samples are streamed in bounded blocks: their Lie-chart
cotangents accumulate into ``O(sum J_c)`` node state, followed by one
prefix-only material reverse per compile node.

An irrational algebraic seam is exactly orientable by its polynomial, but
the current float64 compact atlas cannot represent the algebraic endpoint.
Samples strictly inside its certified root isolator therefore fail closed;
they are never silently extrapolated across the missing endpoint
neighborhood.  Geometry, ray, weight, dispatch, and event-time derivatives
remain outside this material-only bridge.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from fractions import Fraction

import torch
from kinetic_chart_transfer_bridge import (
    BoundKineticOwnerProgram,
    KineticChartP0Geometry,
    KineticChartP0Transfer,
    KineticEventGuardLike,
    KineticOwnerProgramLike,
    bind_kinetic_owner_program,
    compile_bound_kinetic_chart_p0_geometry,
    kinetic_chart_p0_node_material_vjp,
    ordered_p0_transfer,
    refresh_kinetic_chart_p0_transfer,
)
from kinetic_power_word_compiler import (
    AffineKineticPowerSites,
    discover_kinetic_power_word_at_time,
)
from transfer_lie_chart import (
    check_lie_chart_cone,
    transfer_lie_decode,
    transfer_lie_decode_vjp,
    transfer_lie_encode,
    transfer_lie_encode_vjp,
)

DTYPE = torch.float64


@dataclass(frozen=True)
class KineticMultiChartP0Program:
    """Provenance-bound chart partition and compact geometry payloads."""

    binding: BoundKineticOwnerProgram
    charts: tuple[KineticChartP0Geometry, ...]
    generation_digest: str
    seam_policy_id: str = "right_continuous_half_open_v1"
    exact_binary_sample_dispatch: bool = True
    algebraic_endpoint_neighborhood_policy: str = "fail_closed"
    requested_frame_sampling_used: bool = False
    dense_track_chart_refinement_used: bool = False
    temporal_closure_kind: str = "fixed_rank_affine_lie_approximation"
    continuous_forward_error_certified: bool = False
    geometry_vjp_implemented: bool = False
    event_time_vjp_implemented: bool = False

    @property
    def chart_count(self) -> int:
        return len(self.charts)

    @property
    def total_node_count(self) -> int:
        return sum(chart.node_count for chart in self.charts)

    @property
    def structural_tensor_bytes(self) -> int:
        return sum(chart.structural_tensor_bytes for chart in self.charts)

    @property
    def unresolved_algebraic_endpoint_count(self) -> int:
        return sum(not guard.exact for guard in self.binding.program.active_event_guards)

    def assert_current(self) -> None:
        self.binding.assert_current()
        if self.seam_policy_id != self.binding.program.seam_policy_id:
            raise ValueError("multi-chart transfer seam policy changed")
        if not self.exact_binary_sample_dispatch or self.algebraic_endpoint_neighborhood_policy != "fail_closed":
            raise ValueError("multi-chart transfer dispatch policy changed")
        if self.temporal_closure_kind != "fixed_rank_affine_lie_approximation":
            raise ValueError("multi-chart transfer temporal-closure semantics changed")
        if self.continuous_forward_error_certified:
            raise ValueError("fixed-rank kinetic transfer has no continuous forward-error certificate")
        if len(self.charts) != len(self.binding.program.charts) or not self.charts:
            raise ValueError("multi-chart transfer does not cover every certified owner chart")
        for chart_id, chart in enumerate(self.charts):
            if chart.chart_id != chart_id:
                raise ValueError("multi-chart transfer charts are not in canonical order")
            if chart.binding_digest != self.binding.source_content_digest:
                raise ValueError("multi-chart geometry has stale source provenance")
            chart.schedule.assert_current()
            if tuple(chart.owners.tolist()) != chart.owner_word:
                raise ValueError("multi-chart owner tensor changed after compilation")
            if tuple(chart.node_physical_lengths.shape) != (chart.node_count, chart.run_count):
                raise ValueError("multi-chart physical-length payload has the wrong shape")
        for guard in self.binding.program.active_event_guards:
            _require_dispatchable_guard(guard)
        if _multi_program_digest(self.binding, self.charts) != self.generation_digest:
            raise ValueError("multi-chart transfer generation digest mismatch")


@dataclass(frozen=True)
class KineticMultiChartP0Transfer:
    """One global material snapshot and per-chart compact node transfers."""

    program: KineticMultiChartP0Program
    site_density: torch.Tensor
    site_color: torch.Tensor
    chart_node_transfers: tuple[torch.Tensor, ...]
    requested_frame_sampling_used: bool = False

    @property
    def resident_tensor_bytes(self) -> int:
        return self.program.structural_tensor_bytes + _tensor_bytes(
            (self.site_density, self.site_color, *self.chart_node_transfers)
        )

    def chart_transfer(self, chart_id: int) -> KineticChartP0Transfer:
        geometry = self.program.charts[chart_id]
        return KineticChartP0Transfer(
            geometry=geometry,
            site_density=self.site_density,
            site_color=self.site_color,
            node_transfers=self.chart_node_transfers[chart_id],
            node_transfer_cone_passed=True,
            node_lie_cone_passed=True,
        )


@dataclass(frozen=True)
class KineticMultiChartNodeReduction:
    """Blocked MSE reduced to public ``O(sum J_c)`` transfer cotangents."""

    loss: torch.Tensor
    predictions: torch.Tensor | None
    grad_chart_node_transfers: tuple[torch.Tensor, ...]
    accounting: dict[str, int | bool]
    frame_dependent_structural_bytes: int = 0
    frozen_program_semantics: bool = True
    chart_endpoint_vjp_implemented: bool = False
    node_time_vjp_implemented: bool = False
    sample_weight_vjp_implemented: bool = False


@dataclass(frozen=True)
class KineticMultiChartMaterialVJP:
    """Global blocked MSE and material VJP with no retained sample tape."""

    loss: torch.Tensor
    predictions: torch.Tensor | None
    grad_site_density: torch.Tensor
    grad_site_color: torch.Tensor
    grad_chart_node_transfers: tuple[torch.Tensor, ...]
    accounting: dict[str, int | bool]
    geometry_gradients: None = None
    event_time_gradients: None = None
    geometry_vjp_implemented: bool = False
    event_time_vjp_implemented: bool = False


def compile_kinetic_multichart_p0_program(
    owner_program: KineticOwnerProgramLike,
    sites: AffineKineticPowerSites,
    ray_coefficients: torch.Tensor,
    *,
    node_count: int,
) -> KineticMultiChartP0Program:
    """Bind provenance once and compile every certified chart once."""

    binding = bind_kinetic_owner_program(owner_program, sites, ray_coefficients)
    return compile_bound_kinetic_multichart_p0_program(
        binding,
        node_count=node_count,
    )


def compile_bound_kinetic_multichart_p0_program(
    binding: BoundKineticOwnerProgram,
    *,
    node_count: int,
) -> KineticMultiChartP0Program:
    """Compile fixed-rank chart geometry from an already source-bound proof."""

    binding.assert_current()
    charts = tuple(
        compile_bound_kinetic_chart_p0_geometry(
            binding,
            chart_id=chart_id,
            node_count=node_count,
        )
        for chart_id in range(len(binding.program.charts))
    )
    return assemble_bound_kinetic_multichart_p0_program(binding, charts)


def assemble_bound_kinetic_multichart_p0_program(
    binding: BoundKineticOwnerProgram,
    charts: tuple[KineticChartP0Geometry, ...],
) -> KineticMultiChartP0Program:
    """Seal already-selected per-chart ranks without recompiling geometry."""

    binding.assert_current()
    result = KineticMultiChartP0Program(
        binding=binding,
        charts=charts,
        generation_digest=_multi_program_digest(binding, charts),
    )
    result.assert_current()
    return result


def refresh_kinetic_multichart_p0_transfer(
    program: KineticMultiChartP0Program,
    site_density: torch.Tensor,
    site_color: torch.Tensor,
) -> KineticMultiChartP0Transfer:
    """Refresh material transfer nodes without duplicating global material state."""

    program.assert_current()
    density = None
    color = None
    node_transfers = []
    for chart in program.charts:
        refreshed = refresh_kinetic_chart_p0_transfer(chart, site_density, site_color)
        if density is None:
            density = refreshed.site_density
            color = refreshed.site_color
        node_transfers.append(refreshed.node_transfers)
    if density is None or color is None:
        raise ArithmeticError("a passed multi-chart program unexpectedly contains no charts")
    result = KineticMultiChartP0Transfer(
        program=program,
        site_density=density,
        site_color=color,
        chart_node_transfers=tuple(node_transfers),
    )
    _assert_transfer_current(result)
    return result


def dispatch_kinetic_chart_index(
    program: KineticMultiChartP0Program,
    time: Fraction | float | int,
) -> int:
    """Return the exact right-continuous chart index for one rational sample."""

    program.assert_current()
    point = _as_fraction(time, name="sample time")
    return _dispatch_chart_index_current(program, point)


def dispatch_prevalidated_kinetic_chart_index(
    program: KineticMultiChartP0Program,
    time: Fraction | float | int,
    *,
    expected_generation_digest: str,
) -> int:
    """Warm-dispatch after a caller has sealed program tensor identity/version.

    Unlike :func:`dispatch_kinetic_chart_index`, this function deliberately
    does not re-hash the program's world tensors.  The caller must have run
    ``program.assert_current()`` at a cold preparation boundary and must guard
    every source tensor by identity/layout/mutation version during reuse.  The
    expected generation digest makes that prevalidated provenance explicit.
    Exact event predicates still provide right-continuous seam orientation,
    and unresolved algebraic endpoint neighborhoods still fail closed.
    """

    if not isinstance(program, KineticMultiChartP0Program):
        raise TypeError("prevalidated kinetic dispatch requires KineticMultiChartP0Program")
    if expected_generation_digest != program.generation_digest:
        raise ValueError("prevalidated kinetic dispatch generation is stale")
    if (
        not program.exact_binary_sample_dispatch
        or program.seam_policy_id != "right_continuous_half_open_v1"
        or program.algebraic_endpoint_neighborhood_policy != "fail_closed"
    ):
        raise ValueError("prevalidated kinetic dispatch policy changed")
    point = _as_fraction(time, name="sample time")
    _reject_unresolved_algebraic_neighborhood(program, point)
    return _dispatch_chart_index_current(program, point)


def _dispatch_chart_index_current(
    program: KineticMultiChartP0Program,
    point: Fraction,
) -> int:
    owner_program = program.binding.program
    if point < owner_program.t_min or point > owner_program.t_max:
        raise ValueError("sample time leaves the kinetic program domain")
    guards = owner_program.active_event_guards
    lower = 0
    upper = len(guards)
    while lower < upper:
        middle = (lower + upper) // 2
        if _compare_sample_to_guard(point, guards[middle]) < 0:
            upper = middle
        else:
            lower = middle + 1
    chart_id = lower
    if not 0 <= chart_id < program.chart_count:
        raise ArithmeticError("exact kinetic seam dispatch produced an invalid chart index")
    return chart_id


def evaluate_kinetic_multichart_p0_transfer(
    transfer: KineticMultiChartP0Transfer,
    times: torch.Tensor,
    *,
    sample_block_size: int = 32,
) -> torch.Tensor:
    """Stream exact outer dispatch and compact inner transfer evaluation."""

    _assert_transfer_current(transfer)
    times_f64 = _finite_sample_times(times)
    _require_positive_block_size(sample_block_size, name="sample_block_size")
    result = torch.empty((times_f64.numel(), 4), dtype=DTYPE)
    node_charts = tuple(transfer_lie_encode(nodes) for nodes in transfer.chart_node_transfers)
    for start in range(0, int(times_f64.numel()), sample_block_size):
        stop = min(start + sample_block_size, int(times_f64.numel()))
        block = times_f64[start:stop]
        chart_ids = _dispatch_supported_block(transfer.program, block)
        for chart_id, local_rows in _block_rows_by_chart(chart_ids).items():
            local_index = torch.tensor(local_rows, dtype=torch.int64)
            local_times = block[local_index]
            schedule = transfer.program.charts[chart_id].schedule
            weights = schedule.sample_to_node_weights(local_times).weights
            sample_chart = weights @ node_charts[chart_id]
            _require_lie_cone(sample_chart)
            result[start + local_index] = transfer_lie_decode(sample_chart)
    return result


def kinetic_multichart_material_mse_vjp(
    transfer: KineticMultiChartP0Transfer,
    times: torch.Tensor,
    targets: torch.Tensor,
    *,
    background: torch.Tensor,
    sample_block_size: int = 16,
    return_predictions: bool = False,
) -> KineticMultiChartMaterialVJP:
    """Reduce streamed samples to chart nodes, then reverse materials once."""

    reduction = reduce_kinetic_multichart_mse_to_node_transfers(
        transfer,
        times,
        targets,
        background=background,
        sample_block_size=sample_block_size,
        return_predictions=return_predictions,
    )
    grad_density = torch.zeros_like(transfer.site_density)
    grad_color = torch.zeros_like(transfer.site_color)
    for chart_id, grad_node_transfer in enumerate(reduction.grad_chart_node_transfers):
        chart_density_grad, chart_color_grad = kinetic_chart_p0_node_material_vjp(
            transfer.chart_transfer(chart_id),
            grad_node_transfer,
        )
        grad_density += chart_density_grad
        grad_color += chart_color_grad

    accounting = dict(reduction.accounting)
    accounting.update(
        {
            "world_node_replay_count": transfer.program.total_node_count,
            "material_prefix_reverse_node_count": transfer.program.total_node_count,
            "reverse_structural_tensor_bytes": int(accounting["reverse_structural_tensor_bytes"])
            + _tensor_bytes((grad_density, grad_color)),
        }
    )
    return KineticMultiChartMaterialVJP(
        loss=reduction.loss,
        predictions=reduction.predictions,
        grad_site_density=grad_density,
        grad_site_color=grad_color,
        grad_chart_node_transfers=reduction.grad_chart_node_transfers,
        accounting=accounting,
    )


def reduce_kinetic_multichart_mse_to_node_transfers(
    transfer: KineticMultiChartP0Transfer,
    times: torch.Tensor,
    targets: torch.Tensor,
    *,
    background: torch.Tensor,
    sample_block_size: int = 16,
    return_predictions: bool = False,
) -> KineticMultiChartNodeReduction:
    """Reduce sample loss to transfer nodes without replaying materials.

    The returned cotangents are the public seam between temporal
    rasterization and downstream material or stable-stratum geometry reverse.
    Chart endpoints, node times, and sample weights are frozen; their
    derivatives are deliberately absent.  Requested samples are consumed in
    bounded blocks; only ``O(sum J_c)`` reverse state survives the reduction.
    """

    _assert_transfer_current(transfer)
    times_f64 = _finite_sample_times(times)
    _require_positive_block_size(sample_block_size, name="sample_block_size")
    target = torch.as_tensor(targets, dtype=DTYPE, device="cpu").detach()
    if tuple(target.shape) != (int(times_f64.numel()), 3) or not bool(torch.isfinite(target).all().item()):
        raise ValueError("targets must be finite and have shape [F,3]")
    background_f64 = torch.as_tensor(background, dtype=DTYPE, device="cpu").detach().reshape(-1)
    if tuple(background_f64.shape) != (3,) or not bool(torch.isfinite(background_f64).all().item()):
        raise ValueError("background must be a finite RGB vector")

    node_charts = tuple(transfer_lie_encode(nodes) for nodes in transfer.chart_node_transfers)
    grad_node_charts = tuple(torch.zeros_like(nodes) for nodes in node_charts)
    predictions = torch.empty_like(target) if return_predictions else None
    loss = torch.zeros((), dtype=DTYPE)
    inv_element_count = 1.0 / float(target.numel())
    linear_interactions = 0
    dense_fallback_interactions = 0
    peak_block = min(sample_block_size, int(times_f64.numel()))
    for start in range(0, int(times_f64.numel()), sample_block_size):
        stop = min(start + sample_block_size, int(times_f64.numel()))
        block_times = times_f64[start:stop]
        block_targets = target[start:stop]
        chart_ids = _dispatch_supported_block(transfer.program, block_times)
        for chart_id, local_rows in _block_rows_by_chart(chart_ids).items():
            local_index = torch.tensor(local_rows, dtype=torch.int64)
            local_times = block_times[local_index]
            weight_result = transfer.program.charts[chart_id].schedule.sample_to_node_weights(local_times)
            linear_interactions += weight_result.linear_weight_interactions
            dense_fallback_interactions += weight_result.dense_fallback_interactions
            sample_chart = weight_result.weights @ node_charts[chart_id]
            _require_lie_cone(sample_chart)
            sample_transfer = transfer_lie_decode(sample_chart)
            prediction = sample_transfer[:, 1:] + sample_transfer[:, :1] * background_f64
            if predictions is not None:
                predictions[start + local_index] = prediction
            residual = prediction - block_targets[local_index]
            loss += residual.square().sum() * inv_element_count
            grad_prediction = 2.0 * residual * inv_element_count
            grad_sample_transfer = torch.cat(
                (
                    (grad_prediction * background_f64).sum(dim=1, keepdim=True),
                    grad_prediction,
                ),
                dim=1,
            )
            grad_sample_chart = transfer_lie_decode_vjp(sample_chart, grad_sample_transfer)
            grad_node_charts[chart_id].add_(weight_result.weights.T @ grad_sample_chart)

    grad_node_transfers = []
    for chart_id, grad_node_chart in enumerate(grad_node_charts):
        grad_node_transfer = transfer_lie_encode_vjp(
            transfer.chart_node_transfers[chart_id],
            grad_node_chart,
        )
        grad_node_transfers.append(grad_node_transfer)
    grad_node_transfer_tuple = tuple(grad_node_transfers)

    accounting: dict[str, int | bool] = {
        "requested_sample_count": int(times_f64.numel()),
        "chart_count": transfer.program.chart_count,
        "compile_node_count": transfer.program.total_node_count,
        "world_node_replay_count": 0,
        "material_prefix_reverse_node_count": 0,
        "sample_to_node_linear_interactions": linear_interactions,
        "sample_to_node_dense_fallback_interactions": dense_fallback_interactions,
        "structural_tensor_bytes": transfer.resident_tensor_bytes,
        "reverse_structural_tensor_bytes": _tensor_bytes((*grad_node_charts, *grad_node_transfer_tuple)),
        "returned_node_transfer_cotangent_bytes": _tensor_bytes(grad_node_transfer_tuple),
        "peak_sample_block_bytes": peak_block
        * (max(chart.node_count for chart in transfer.program.charts) + 4 + 4 + 3 + 3 + 1)
        * 8,
        "frame_dependent_structural_bytes": 0,
        "dense_track_chart_refinement_bytes": 0,
        "requested_frame_sampling_used_for_compile": False,
        "geometry_gradients_emitted": False,
        "event_time_gradients_emitted": False,
        "frozen_program_semantics": True,
        "chart_endpoint_gradients_emitted": False,
        "node_time_gradients_emitted": False,
        "sample_weight_gradients_emitted": False,
    }
    return KineticMultiChartNodeReduction(
        loss=loss,
        predictions=predictions,
        grad_chart_node_transfers=grad_node_transfer_tuple,
        accounting=accounting,
    )


def exact_streamed_kinetic_p0_replay(
    transfer: KineticMultiChartP0Transfer,
    times: torch.Tensor,
    *,
    sample_block_size: int = 32,
) -> torch.Tensor:
    """Linear-work CPU oracle; never use this as the compact training path."""

    _assert_transfer_current(transfer)
    times_f64 = _finite_sample_times(times)
    _require_positive_block_size(sample_block_size, name="sample_block_size")
    result = torch.empty((times_f64.numel(), 4), dtype=DTYPE)
    binding = transfer.program.binding
    for start in range(0, int(times_f64.numel()), sample_block_size):
        stop = min(start + sample_block_size, int(times_f64.numel()))
        block = times_f64[start:stop]
        _dispatch_supported_block(transfer.program, block)
        for local_id, time in enumerate(block):
            exact_time = Fraction.from_float(float(time.item()))
            fixed_word = discover_kinetic_power_word_at_time(
                binding.sites,
                binding.ray_coefficients,
                time=exact_time,
                near=binding.program.near,
                far=binding.program.far,
            )
            cuts = (binding.program.near, *fixed_word.transition_depths, binding.program.far)
            coordinate_lengths = tuple(right - left for left, right in zip(cuts, cuts[1:], strict=False))
            direction = binding.ray_coefficients[6:9] + time * binding.ray_coefficients[9:12]
            speed = torch.linalg.vector_norm(direction)
            physical_lengths = speed * torch.tensor(
                [float(length) for length in coordinate_lengths],
                dtype=DTYPE,
            )
            result[start + local_id] = ordered_p0_transfer(
                fixed_word.word.owners,
                physical_lengths,
                transfer.site_density,
                transfer.site_color,
            )
    return result


def _dispatch_supported_block(
    program: KineticMultiChartP0Program,
    times: torch.Tensor,
) -> list[int]:
    result = []
    for time in times:
        point = Fraction.from_float(float(time.item()))
        _reject_unresolved_algebraic_neighborhood(program, point)
        chart_id = _dispatch_chart_index_current(program, point)
        chart = program.charts[chart_id]
        value = float(time.item())
        if value < chart.schedule.t_min or value > chart.schedule.t_max:
            raise ValueError("sample leaves the selected chart's certified float64 interval")
        result.append(chart_id)
    return result


def _block_rows_by_chart(chart_ids: list[int]) -> dict[int, list[int]]:
    rows: dict[int, list[int]] = {}
    for index, chart_id in enumerate(chart_ids):
        rows.setdefault(chart_id, []).append(index)
    return rows


def _reject_unresolved_algebraic_neighborhood(
    program: KineticMultiChartP0Program,
    point: Fraction,
) -> None:
    for guard in program.binding.program.active_event_guards:
        if not guard.exact and guard.lower_bound < point < guard.upper_bound:
            raise ValueError(
                "sample lies inside an unresolved algebraic endpoint neighborhood; "
                "compact float64 chart evaluation fails closed"
            )


def _compare_sample_to_guard(point: Fraction, guard: KineticEventGuardLike) -> int:
    _require_dispatchable_guard(guard)
    if guard.exact:
        return (point > guard.lower_bound) - (point < guard.lower_bound)
    if point <= guard.lower_bound:
        return -1
    if point >= guard.upper_bound:
        return 1
    for source in guard.sources:
        lower_sign = _sign(source.polynomial.evaluate(guard.lower_bound))
        upper_sign = _sign(source.polynomial.evaluate(guard.upper_bound))
        if lower_sign == 0 or upper_sign == 0 or lower_sign == upper_sign:
            continue
        value_sign = _sign(source.polynomial.evaluate(point))
        if value_sign == 0:
            raise ValueError("a nonexact algebraic guard was hit by a rational sample")
        if value_sign == lower_sign:
            return -1
        if value_sign == upper_sign:
            return 1
    raise ValueError("algebraic event predicates cannot orient this rational sample")


def _require_dispatchable_guard(guard: KineticEventGuardLike) -> None:
    if not guard.active_owner_change:
        raise ValueError("multi-chart dispatch received an inactive event guard")
    if guard.exact:
        if guard.lower_bound != guard.upper_bound:
            raise ValueError("exact kinetic event guard has a nonzero interval")
        return
    if guard.lower_bound >= guard.upper_bound:
        raise ValueError("nonexact kinetic event guard has no isolating interval")
    if not any(
        _strict_sign_change(
            source.polynomial.evaluate(guard.lower_bound),
            source.polynomial.evaluate(guard.upper_bound),
        )
        for source in guard.sources
    ):
        raise ValueError("nonexact active event has no sign-changing dispatch predicate")


def _strict_sign_change(left: Fraction, right: Fraction) -> bool:
    return left != 0 and right != 0 and (left > 0) != (right > 0)


def _sign(value: Fraction) -> int:
    return (value > 0) - (value < 0)


def _as_fraction(value: Fraction | float | int, *, name: str) -> Fraction:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite rational, float, or integer")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if not isinstance(value, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite rational, float, or integer")
    return Fraction.from_float(value)


def _finite_sample_times(times: torch.Tensor) -> torch.Tensor:
    result = torch.as_tensor(times, dtype=DTYPE, device="cpu").detach().reshape(-1).contiguous()
    if result.numel() < 1 or not bool(torch.isfinite(result).all().item()):
        raise ValueError("times must be nonempty finite CPU-compatible data")
    return result


def _require_positive_block_size(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _require_lie_cone(chart: torch.Tensor) -> None:
    report = check_lie_chart_cone(chart)
    if not report.passed:
        raise ValueError("interpolated kinetic affine-Lie chart left the physical cone")


def _assert_transfer_current(transfer: KineticMultiChartP0Transfer) -> None:
    transfer.program.assert_current()
    if len(transfer.chart_node_transfers) != transfer.program.chart_count:
        raise ValueError("multi-chart material snapshot has the wrong chart count")
    if tuple(transfer.site_density.shape) != (transfer.program.binding.sites.site_count,):
        raise ValueError("multi-chart density snapshot has the wrong shape")
    if tuple(transfer.site_color.shape) != (transfer.program.binding.sites.site_count, 3):
        raise ValueError("multi-chart color snapshot has the wrong shape")
    for geometry, nodes in zip(transfer.program.charts, transfer.chart_node_transfers, strict=True):
        if tuple(nodes.shape) != (geometry.node_count, 4) or not bool(torch.isfinite(nodes).all().item()):
            raise ValueError("multi-chart transfer nodes are stale or malformed")


def _multi_program_digest(
    binding: BoundKineticOwnerProgram,
    charts: tuple[KineticChartP0Geometry, ...],
) -> str:
    digest = hashlib.sha256()
    parts: tuple[object, ...] = (
        "kinetic-multichart-p0-program-v1",
        binding.source_content_digest,
        binding.program_semantic_digest,
        tuple(
            (
                chart.chart_id,
                chart.owner_word,
                chart.schedule.t_min,
                chart.schedule.t_max,
                chart.node_count,
                _tensor_digest(chart.schedule.node_times),
                _tensor_digest(chart.schedule.fit_matrix),
                _tensor_digest(chart.schedule.barycentric_weights),
                _tensor_digest(chart.owners),
                _tensor_digest(chart.node_physical_lengths),
            )
            for chart in charts
        ),
    )
    for part in parts:
        encoded = repr(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(repr((tuple(value.shape), str(value.dtype))).encode("utf-8"))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _tensor_bytes(tensors: tuple[torch.Tensor, ...]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


__all__ = [
    "KineticMultiChartMaterialVJP",
    "KineticMultiChartNodeReduction",
    "KineticMultiChartP0Program",
    "KineticMultiChartP0Transfer",
    "assemble_bound_kinetic_multichart_p0_program",
    "compile_bound_kinetic_multichart_p0_program",
    "compile_kinetic_multichart_p0_program",
    "dispatch_kinetic_chart_index",
    "dispatch_prevalidated_kinetic_chart_index",
    "evaluate_kinetic_multichart_p0_transfer",
    "exact_streamed_kinetic_p0_replay",
    "kinetic_multichart_material_mse_vjp",
    "reduce_kinetic_multichart_mse_to_node_transfers",
    "refresh_kinetic_multichart_p0_transfer",
]
