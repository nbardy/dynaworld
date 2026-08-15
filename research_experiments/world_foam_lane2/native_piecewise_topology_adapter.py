"""Stream topology-changing WorldFoam charts through the native lifecycle.

The lower-level :mod:`native_track_adapter` deliberately handles one fixed
owner topology.  This module composes several such worlds without pretending
that their CSR words or active power faces are shared.  It preserves the
systems shape needed by the paper:

* one caller-owned global site-gradient ledger and one ``P * F * 3`` loss
  denominator;
* one spatial ``B_p`` block and one temporal ``K`` block resident at a time;
* one prepared topology/binding/native world payload resident at a time;
* exact binary-sample dispatch against rational polynomials and certified
  algebraic-root isolators; and
* a right one-sided frozen-topology VJP at every seam, with no claim that the
  event time or discrete dispatch has been differentiated.

An irrational event cannot be represented exactly by a binary floating-point
atlas endpoint.  The polynomial guard remains exact for requested-sample
dispatch, but the result is not continuous-real paper evidence until native
chart domains themselves can retain algebraic endpoints.  This distinction is
intentional and machine readable.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal

import torch
from native_track_adapter import (
    _INFLIGHT_LEDGER_ATTRIBUTE,
    NativeTrackAdapterUnavailableError,
    _assert_binding_matches_prepared,
    _registered_track_range,
    _resolve_staging_layout,
    _sample_partition_generation_id,
    _synchronize_device,
    _validate_staged_camera_block,
    resolve_native_fixed_word_p0_ops,
)
from power_topology_event_predicates import (
    RIGHT_CONTINUOUS_SEAM_POLICY,
    ZERO_RUN_DELETION_EQUIVALENCE,
    TopologyEventIsolation,
)
from powerfoam_track_staging import PowerFoamTrackStagingPlan
from prepared_track_block import accumulate_prepared_rows_, gather_prepared_rows
from staged_compiled_lie_adjoint import (
    CompactSpatialStepLedger,
    PreparedCompactStagedLieWorld,
    _assert_compact_spatial_step_current,
    _tensor_signature,
)

FractionLike = Fraction | float | int


@dataclass(frozen=True)
class NativeAlgebraicTopologyEventGuard:
    """Exact event predicate retained beyond rational root isolation."""

    event_id: str
    left_chart_id: str
    right_chart_id: str
    source_track_id: int
    predicate_kind: str
    site_ids: tuple[int, ...]
    polynomial_coefficients: tuple[Fraction, ...]
    certified_domain_t_min: Fraction
    certified_domain_t_max: Fraction
    root_lower_bound: Fraction
    root_upper_bound: Fraction
    root_exact: bool
    root_multiplicity: int
    sturm_root_count: int
    polynomial_sign_at_lower: int
    polynomial_sign_at_upper: int
    seam_policy_id: str
    geometry_ray_content_digest: str
    compiler_provenance: str
    guard_digest: str
    zero_run_transfer_identity_certified: bool = True
    classical_geometry_derivative_at_event_certified: bool = False

    @property
    def dispatch_kind(self) -> str:
        return "exact_rational_root" if self.root_exact else "polynomial_guarded_algebraic_root"

    def assert_current(self) -> None:
        if not self.event_id.strip() or not self.left_chart_id.strip() or not self.right_chart_id.strip():
            raise ValueError("event and adjacent chart ids must be nonempty")
        if self.left_chart_id == self.right_chart_id:
            raise ValueError("an event guard must separate two distinct charts")
        if self.source_track_id < 0:
            raise ValueError("event guard source track id must be nonnegative")
        if not self.compiler_provenance.strip():
            raise ValueError("event guard compiler provenance must be nonempty")
        _require_sha256(self.geometry_ray_content_digest, name="event geometry/ray provenance")
        if self.seam_policy_id != RIGHT_CONTINUOUS_SEAM_POLICY.policy_id:
            raise ValueError("event guard uses an unsupported seam policy")
        if not self.polynomial_coefficients or self.polynomial_coefficients[-1] == 0:
            raise ValueError("event guard polynomial must be nonzero and canonically trimmed")
        if self.root_multiplicity < 1 or self.sturm_root_count != 1:
            raise ValueError("event guard must identify exactly one certified root")
        if self.root_upper_bound < self.root_lower_bound:
            raise ValueError("event guard root interval is reversed")
        if (
            self.certified_domain_t_max <= self.certified_domain_t_min
            or self.root_lower_bound < self.certified_domain_t_min
            or self.root_upper_bound > self.certified_domain_t_max
        ):
            raise ValueError("event guard root leaves its certified predicate domain")
        lower_value = _evaluate_polynomial(self.polynomial_coefficients, self.root_lower_bound)
        upper_value = _evaluate_polynomial(self.polynomial_coefficients, self.root_upper_bound)
        if _sign(lower_value) != self.polynomial_sign_at_lower or _sign(upper_value) != (self.polynomial_sign_at_upper):
            raise ValueError("event guard endpoint signs do not match its exact polynomial")
        if self.root_exact:
            if self.root_lower_bound != self.root_upper_bound or lower_value != 0:
                raise ValueError("exact event guard does not contain an exact rational root")
        elif (
            self.root_lower_bound >= self.root_upper_bound
            or self.polynomial_sign_at_lower == 0
            or self.polynomial_sign_at_upper == 0
            or self.polynomial_sign_at_lower == self.polynomial_sign_at_upper
        ):
            raise ValueError("algebraic event guard lacks a sign-changing rational isolator")
        if not self.zero_run_transfer_identity_certified:
            raise ValueError("native topology streaming currently supports only zero-run transfer seams")
        if self.classical_geometry_derivative_at_event_certified:
            raise ValueError("topology-changing event geometry derivatives must remain uncertified")
        if _guard_digest(self) != self.guard_digest:
            raise ValueError("event guard provenance digest is stale or fabricated")

    def compare_binary_sample(self, time: FractionLike) -> int:
        """Return ``-1/0/+1`` for a rational sample relative to the event."""

        self.assert_current()
        point = _as_fraction(time, name="sample time")
        if self.root_exact:
            return (point > self.root_lower_bound) - (point < self.root_lower_bound)
        if point <= self.root_lower_bound:
            return -1
        if point >= self.root_upper_bound:
            return 1
        value_sign = _sign(_evaluate_polynomial(self.polynomial_coefficients, point))
        if value_sign == 0:
            raise ValueError("an allegedly irrational event root was hit by a rational sample")
        if value_sign == self.polynomial_sign_at_lower:
            return -1
        if value_sign == self.polynomial_sign_at_upper:
            return 1
        raise ValueError("event polynomial cannot orient a sample inside its isolating interval")


@dataclass(frozen=True)
class NativeFixedTopologySubchartSpec:
    chart_index: int
    t_min: float
    t_max: float
    node_count: int
    chart_digest: str


@dataclass(frozen=True)
class NativePiecewiseTopologyChartSpec:
    """Payload-free identity needed to dispatch before loading a CPU atlas."""

    chart_id: str
    source_track_start: int
    source_track_end: int
    schedule_generation_digest: str
    topology_content_digest: str
    geometry_ray_content_digest: str
    certificate_binding_digest: str
    binding_mode: str
    binding_paper_evidence_eligible: bool
    native_subcharts: tuple[NativeFixedTopologySubchartSpec, ...]
    chart_provenance: str
    chart_spec_digest: str

    def assert_current(self) -> None:
        if not self.chart_id.strip() or not self.chart_provenance.strip():
            raise ValueError("topology chart id and provenance must be nonempty")
        if self.source_track_start < 0 or self.source_track_end <= self.source_track_start:
            raise ValueError("topology chart track range must be nonempty")
        for digest in (
            self.schedule_generation_digest,
            self.topology_content_digest,
            self.geometry_ray_content_digest,
            self.certificate_binding_digest,
            self.chart_spec_digest,
        ):
            _require_sha256(digest, name="topology chart digest")
        if self.binding_mode not in {"strict_frozen_evaluation", "training_owner_topology_only"}:
            raise ValueError("topology chart uses an unknown binding mode")
        if self.binding_mode == "training_owner_topology_only" and self.binding_paper_evidence_eligible:
            raise ValueError("training topology charts cannot be paper evidence")
        if not self.native_subcharts:
            raise ValueError("topology chart must contain at least one native transfer chart")
        for index, chart in enumerate(self.native_subcharts):
            if chart.chart_index != index or not math.isfinite(chart.t_min) or not math.isfinite(chart.t_max):
                raise ValueError("native subchart identities must be finite and contiguous by index")
            if chart.t_max <= chart.t_min or chart.node_count < 1 or not chart.chart_digest.strip():
                raise ValueError("native subchart metadata is invalid")
            if index and self.native_subcharts[index - 1].t_max != chart.t_min:
                raise ValueError("native subcharts must form one contiguous half-open partition")
        if _chart_spec_digest(self) != self.chart_spec_digest:
            raise ValueError("topology chart spec digest is stale or fabricated")


@dataclass(frozen=True)
class NativePiecewiseTopologyChartPayload:
    """One provider-owned prepared topology and its sealed native binding."""

    prepared: PreparedCompactStagedLieWorld
    certificate_binding: Any


@dataclass(frozen=True)
class NativePiecewiseTopologyProgram:
    """Exact outer dispatch plus payload-free native chart identities."""

    domain_t_min: Fraction
    domain_t_max: Fraction
    charts: tuple[NativePiecewiseTopologyChartSpec, ...]
    event_guards: tuple[NativeAlgebraicTopologyEventGuard, ...]
    compiler_provenance: str
    generation_digest: str
    seam_policy_id: str = RIGHT_CONTINUOUS_SEAM_POLICY.policy_id
    exact_binary_sample_dispatch: bool = True
    event_time_vjp: Literal["not_implemented"] = "not_implemented"
    algebraic_event_dispatch_vjp: Literal["unresolved"] = "unresolved"
    frozen_topology_parameter_vjp: Literal["right_one_sided"] = "right_one_sided"

    @property
    def continuous_real_native_boundary_equivalence_certified(self) -> bool:
        return _continuous_real_native_boundary_equivalence(self)

    @property
    def paper_evidence_eligible(self) -> bool:
        return self.continuous_real_native_boundary_equivalence_certified and all(
            chart.binding_paper_evidence_eligible for chart in self.charts
        )

    def assert_current(self) -> None:
        if self.domain_t_max <= self.domain_t_min:
            raise ValueError("piecewise topology domain must have positive width")
        if not self.compiler_provenance.strip() or not self.charts:
            raise ValueError("piecewise topology program needs provenance and charts")
        if self.seam_policy_id != RIGHT_CONTINUOUS_SEAM_POLICY.policy_id:
            raise ValueError("piecewise topology program uses an unsupported seam policy")
        if not self.exact_binary_sample_dispatch:
            raise ValueError("piecewise topology program must retain exact binary-sample dispatch")
        if len(self.event_guards) + 1 != len(self.charts):
            raise ValueError("piecewise topology program needs one event guard between adjacent charts")
        chart_ids = tuple(chart.chart_id for chart in self.charts)
        if len(set(chart_ids)) != len(chart_ids):
            raise ValueError("piecewise topology chart ids must be unique")
        for chart in self.charts:
            chart.assert_current()
        for index, guard in enumerate(self.event_guards):
            guard.assert_current()
            if (guard.left_chart_id, guard.right_chart_id) != chart_ids[index : index + 2]:
                raise ValueError("event guard adjacency disagrees with topology chart order")
            adjacent = self.charts[index : index + 2]
            if any(chart.geometry_ray_content_digest != guard.geometry_ray_content_digest for chart in adjacent):
                raise ValueError("event guard does not bind the adjacent charts' geometry/ray snapshot")
            if any(
                not chart.source_track_start <= guard.source_track_id < chart.source_track_end for chart in adjacent
            ):
                raise ValueError("event guard source track leaves an adjacent topology chart")
            if index and not _event_guard_strictly_before(self.event_guards[index - 1], guard):
                raise ValueError("topology event guards are not strictly ordered")
        if self.event_guards:
            if self.event_guards[0].root_lower_bound <= self.domain_t_min:
                raise ValueError("first topology event must lie inside the program domain")
            if self.event_guards[-1].root_upper_bound >= self.domain_t_max:
                raise ValueError("last topology event must lie inside the program domain")
        _require_sha256(self.generation_digest, name="piecewise topology program generation")
        if _program_digest(self) != self.generation_digest:
            raise ValueError("piecewise topology program generation is stale or fabricated")

    def chart_index_for_binary_sample(self, time: FractionLike) -> int:
        self.assert_current()
        point = _as_fraction(time, name="sample time")
        if point < self.domain_t_min or point > self.domain_t_max:
            raise NativeTrackAdapterUnavailableError("sample time leaves the algebraic topology program domain")
        for index, guard in enumerate(self.event_guards):
            if guard.compare_binary_sample(point) < 0:
                return index
        return len(self.charts) - 1


@dataclass(frozen=True)
class NativeTopologyEventVJPMetadata:
    event_id: str
    guard_digest: str
    dispatch_kind: str
    root_lower_bound: Fraction
    root_upper_bound: Fraction
    seam_sample_assignment: Literal["right_chart"] = "right_chart"
    frozen_topology_parameter_vjp: Literal["right_one_sided"] = "right_one_sided"
    event_time_vjp: Literal["not_implemented"] = "not_implemented"
    algebraic_event_dispatch_vjp: Literal["unresolved"] = "unresolved"
    differentiability: Literal["nondifferentiable_or_stratified"] = "nondifferentiable_or_stratified"


@dataclass(frozen=True)
class NativePiecewiseTopologyBlockResult:
    block_id: str
    program_generation_digest: str
    sample_partition_generation_id: str
    block_loss: torch.Tensor
    global_loss_element_count: int
    topology_chart_count: int
    selected_topology_chart_count: int
    native_subchart_count: int
    sample_block_count: int
    device_barrier_count: int
    maximum_resident_payload_count: int
    maximum_resident_target_elements: int
    maximum_resident_sample_time_bytes: int
    event_gradients: tuple[NativeTopologyEventVJPMetadata, ...]
    exact_binary_sample_dispatch: bool
    continuous_real_native_boundary_equivalence_certified: bool
    paper_evidence_eligible: bool

    @property
    def resident_output_bytes(self) -> int:
        return self.block_loss.numel() * self.block_loss.element_size()


@dataclass(frozen=True)
class _NativeSubchartPartition:
    chart_index: int
    global_start: int
    global_end: int


@dataclass(frozen=True)
class _NativeTopologyPartition:
    topology_chart_index: int
    global_start: int
    global_end: int
    subcharts: tuple[_NativeSubchartPartition, ...]


def make_native_algebraic_topology_event_guard(
    isolation: TopologyEventIsolation,
    *,
    root_index: int,
    event_id: str,
    left_chart_id: str,
    right_chart_id: str,
    source_track_id: int,
    geometry_ray_content_digest: str,
    compiler_provenance: str,
) -> NativeAlgebraicTopologyEventGuard:
    """Seal one exact rational root or polynomial-guarded algebraic root."""

    if root_index < 0 or root_index >= len(isolation.roots):
        raise ValueError("event root index is outside the certified isolation")
    if isolation.seam_policy_id != RIGHT_CONTINUOUS_SEAM_POLICY.policy_id:
        raise ValueError("event isolation uses an unsupported seam policy")
    root = isolation.roots[root_index]
    values = dict(
        event_id=event_id,
        left_chart_id=left_chart_id,
        right_chart_id=right_chart_id,
        source_track_id=source_track_id,
        predicate_kind=isolation.predicate.kind,
        site_ids=isolation.predicate.site_ids,
        polynomial_coefficients=isolation.predicate.polynomial.coefficients,
        certified_domain_t_min=isolation.t_min,
        certified_domain_t_max=isolation.t_max,
        root_lower_bound=root.lower_bound,
        root_upper_bound=root.upper_bound,
        root_exact=root.exact,
        root_multiplicity=root.multiplicity,
        sturm_root_count=root.sturm_root_count,
        polynomial_sign_at_lower=root.polynomial_sign_at_lower,
        polynomial_sign_at_upper=root.polynomial_sign_at_upper,
        seam_policy_id=isolation.seam_policy_id,
        geometry_ray_content_digest=geometry_ray_content_digest,
        compiler_provenance=compiler_provenance,
        zero_run_transfer_identity_certified=(
            ZERO_RUN_DELETION_EQUIVALENCE.forward_value_equivalent
            and ZERO_RUN_DELETION_EQUIVALENCE.insertion_or_deletion_preserves_ordered_product
        ),
        classical_geometry_derivative_at_event_certified=(
            ZERO_RUN_DELETION_EQUIVALENCE.classical_geometry_derivative_at_event_certified
        ),
    )
    provisional = NativeAlgebraicTopologyEventGuard(**values, guard_digest="")
    guard = NativeAlgebraicTopologyEventGuard(**values, guard_digest=_guard_digest(provisional))
    guard.assert_current()
    return guard


def describe_native_piecewise_topology_chart(
    *,
    chart_id: str,
    prepared: PreparedCompactStagedLieWorld,
    certificate_binding: Any,
    chart_provenance: str,
) -> NativePiecewiseTopologyChartSpec:
    """Create payload-free identity metadata before releasing a CPU payload."""

    _assert_binding_matches_prepared(certificate_binding, prepared)
    track_ids = prepared.topology.source_track_ids
    expected_track_ids = torch.arange(
        int(track_ids[0].item()),
        int(track_ids[-1].item()) + 1,
        dtype=track_ids.dtype,
        device=track_ids.device,
    )
    if not torch.equal(track_ids, expected_track_ids):
        raise ValueError("native topology chart tracks must form one contiguous spatial block")
    native_subcharts = tuple(
        NativeFixedTopologySubchartSpec(
            chart_index=index,
            t_min=float(chart.transfer_atlas.t_min),
            t_max=float(chart.transfer_atlas.t_max),
            node_count=chart.node_count,
            chart_digest=certificate_binding.charts[index].chart_digest,
        )
        for index, chart in enumerate(prepared.world_snapshot.atlas.charts)
    )
    values = dict(
        chart_id=chart_id,
        source_track_start=int(track_ids[0].item()),
        source_track_end=int(track_ids[-1].item()) + 1,
        schedule_generation_digest=prepared.schedule.generation_digest,
        topology_content_digest=_topology_content_digest(prepared),
        geometry_ray_content_digest=native_geometry_ray_content_digest(
            prepared.source_tensors[0],
            prepared.source_tensors[1],
        ),
        certificate_binding_digest=str(certificate_binding.canonical_digest),
        binding_mode=str(certificate_binding.binding_mode),
        binding_paper_evidence_eligible=bool(certificate_binding.paper_evidence_eligible),
        native_subcharts=native_subcharts,
        chart_provenance=chart_provenance,
    )
    provisional = NativePiecewiseTopologyChartSpec(**values, chart_spec_digest="")
    spec = NativePiecewiseTopologyChartSpec(**values, chart_spec_digest=_chart_spec_digest(provisional))
    spec.assert_current()
    return spec


def make_native_piecewise_topology_program(
    charts: tuple[NativePiecewiseTopologyChartSpec, ...],
    event_guards: tuple[NativeAlgebraicTopologyEventGuard, ...],
    *,
    domain_t_min: FractionLike,
    domain_t_max: FractionLike,
    compiler_provenance: str,
) -> NativePiecewiseTopologyProgram:
    values = dict(
        domain_t_min=_as_fraction(domain_t_min, name="domain_t_min"),
        domain_t_max=_as_fraction(domain_t_max, name="domain_t_max"),
        charts=tuple(charts),
        event_guards=tuple(event_guards),
        compiler_provenance=compiler_provenance,
    )
    provisional = NativePiecewiseTopologyProgram(**values, generation_digest="")
    program = NativePiecewiseTopologyProgram(**values, generation_digest=_program_digest(provisional))
    program.assert_current()
    return program


def execute_native_piecewise_topology_track_block(
    ledger: CompactSpatialStepLedger,
    *,
    block_id: str,
    program: NativePiecewiseTopologyProgram,
    payload_provider: Callable[
        [NativePiecewiseTopologyChartSpec],
        AbstractContextManager[NativePiecewiseTopologyChartPayload],
    ],
    staging_plan: PowerFoamTrackStagingPlan,
    background_rgb: torch.Tensor | tuple[float, float, float] | list[float],
    replay_config: Any,
    sample_block_size: int,
    native_ops: Any | None = None,
    max_in_flight_sample_blocks: int = 1,
    device_synchronize: Callable[[torch.device], None] | None = None,
    physical_length_epsilon: float = 1.0e-8,
    cone_tolerance: float = 1.0e-6,
) -> NativePiecewiseTopologyBlockResult:
    """Execute and consume one spatial block through streamed topology worlds.

    Gradient rows are scattered into ``ledger`` immediately after each native
    topology world is finalized, so no ``number_of_topology_charts * B_p``
    result is retained.  If a provider or launch fails after that point the
    in-flight marker remains set and the step must be discarded; retrying a
    partially accumulated ledger is intentionally forbidden.
    """

    _assert_compact_spatial_step_current(ledger)
    program.assert_current()
    if ledger.finalized:
        raise ValueError("compact spatial step was already finalized")
    track_start, track_end = _registered_track_range(ledger, block_id)
    if block_id in ledger.consumed_block_ids:
        raise ValueError("compact spatial block was already consumed")
    if getattr(ledger, _INFLIGHT_LEDGER_ATTRIBUTE, None) is not None:
        raise ValueError("consume or discard the previous native spatial operation first")
    if sample_block_size < 1:
        raise ValueError("sample_block_size must be positive")
    if max_in_flight_sample_blocks != 1:
        raise ValueError("piecewise native topology requires exactly one in-flight K block")
    expected_generation = tuple(
        digest
        for expected_block_id, digest in ledger.expected_block_schedule_generations
        if expected_block_id == block_id
    )
    if expected_generation != (program.generation_digest,):
        raise ValueError("piecewise topology program is not registered for this spatial block")
    if any((chart.source_track_start, chart.source_track_end) != (track_start, track_end) for chart in program.charts):
        raise ValueError("every topology chart must describe the registered spatial block")

    staging = _resolve_staging_layout(
        staging_plan,
        ledger,
        track_start=track_start,
        track_end=track_end,
    )
    ordered_plan, partitions = _ordered_piecewise_staging_plan(
        staging.plan,
        program,
        device=ledger.source_tensors[0].device,
    )
    if not partitions:
        raise ValueError("piecewise topology block requires at least one selected chart")
    sample_partition_generation_id = (
        _sample_partition_generation_id(
            ordered_plan,
            loss_normalization_id=ledger.loss_normalization_id,
            global_track_count=ledger.global_track_count,
            global_sample_count=ledger.global_frame_count,
        )
        + f":topology:{program.generation_digest}"
    )
    geometry, global_rays, density, color = ledger.source_tensors
    if any(tensor.dtype != torch.float32 for tensor in (geometry, global_rays, density, color)):
        raise ValueError("native topology worlds require float32 live tensors")
    if len({tensor.device for tensor in (geometry, global_rays, density, color)}) != 1:
        raise ValueError("native topology world tensors must share one device")
    background = torch.as_tensor(background_rgb, dtype=torch.float32, device=geometry.device).reshape(3).contiguous()
    native = resolve_native_fixed_word_p0_ops(native_ops)
    block_loss = torch.zeros((), dtype=torch.float32, device=geometry.device)
    block_loss_storage = block_loss.untyped_storage().data_ptr()
    selected_topology_charts = 0
    native_subchart_count = 0
    sample_block_count = 0
    device_barrier_count = 0
    maximum_resident_target_elements = 0
    maximum_resident_sample_time_bytes = 0
    setattr(ledger, _INFLIGHT_LEDGER_ATTRIBUTE, f"piecewise-topology:{block_id}")

    for partition in partitions:
        spec = program.charts[partition.topology_chart_index]
        with payload_provider(spec) as payload:
            _validate_payload(spec, payload, ledger, block_id=block_id)
            prepared = payload.prepared
            binding = payload.certificate_binding
            topology = prepared.topology
            source_site_ids = topology.source_site_ids.to(device=geometry.device, dtype=torch.long)
            source_track_ids = topology.source_track_ids.to(device=geometry.device, dtype=torch.long)
            compact_sites = gather_prepared_rows(geometry, source_site_ids)
            compact_rays = gather_prepared_rows(global_rays, source_track_ids)
            compact_density = gather_prepared_rows(density, source_site_ids)
            compact_color = gather_prepared_rows(color, source_site_ids)
            compact_rgba = torch.cat((compact_color, compact_density[:, None]), dim=1).contiguous()
            first_subchart = partition.subcharts[0]
            first_range = next(
                _native_subchart_sample_blocks(
                    first_subchart,
                    topology_chart_index=partition.topology_chart_index,
                    sample_block_size=sample_block_size,
                    block_id=block_id,
                )
            )
            first_stage = ordered_plan.stage(
                track_start=staging.stage_track_start,
                track_end=staging.stage_track_end,
                sample_start=first_range[1],
                sample_end=first_range[2],
                require_affine_ray_program=True,
            )
            _validate_staged_camera_block(
                first_stage,
                ordered_plan,
                track_start=staging.stage_track_start,
                track_end=staging.stage_track_end,
                sample_start=first_range[1],
                sample_end=first_range[2],
                compact_rays=compact_rays,
                validate_static_camera_program=True,
                global_track_count=ledger.global_track_count,
                global_sample_count=ledger.global_frame_count,
                view_factor=staging.view_factor,
            )
            topology_token = native.prepare_fixed_word_p0_topology_token(
                topology.word_offsets_i32.to(device=geometry.device, dtype=torch.int32),
                topology.word_owner_i32.to(device=geometry.device, dtype=torch.int32),
                topology.word_left_incidence_i32.to(device=geometry.device, dtype=torch.int32),
                topology.word_right_incidence_i32.to(device=geometry.device, dtype=torch.int32),
                topology.track_incidence_offsets_i32.to(device=geometry.device, dtype=torch.int32),
                topology.incidence_boundary_i32.to(device=geometry.device, dtype=torch.int32),
                topology.boundary_site_pairs_i32.to(device=geometry.device, dtype=torch.int32),
                track_count=topology.track_count,
                site_count=topology.site_count,
                certificate_binding=binding,
            )
            world_token = native.refresh_fixed_word_p0_world_token(
                topology_token,
                compact_sites,
                compact_rgba,
                compact_rays,
                replay_config,
                physical_length_epsilon=physical_length_epsilon,
                cone_tolerance=cone_tolerance,
            )
            expected_native_partitions = tuple(
                (
                    spec.native_subcharts[subchart.chart_index].chart_digest,
                    subchart.global_start,
                    subchart.global_end,
                )
                for subchart in partition.subcharts
            )
            world_grad = native.fixed_word_p0_lie_world_grad_init_launch_only(
                world_token,
                expected_chart_partitions=expected_native_partitions,
                global_track_count=ledger.global_track_count,
                global_sample_count=ledger.global_frame_count,
                global_loss_element_count=ledger.global_loss_element_count,
                loss_normalization_id=ledger.loss_normalization_id,
                sample_partition_generation_id=sample_partition_generation_id,
                resident_sample_start=partition.global_start,
                resident_sample_end=partition.global_end,
            )
            for subchart in partition.subcharts:
                chart_token = native.prepare_fixed_word_p0_chart_token(
                    world_token,
                    prepared.world_snapshot.atlas.charts[subchart.chart_index].transfer_atlas.node_times.to(
                        device=geometry.device, dtype=torch.float32
                    ),
                    chart_index=subchart.chart_index,
                )
                expected_chart = spec.native_subcharts[subchart.chart_index]
                if (
                    chart_token.world is not world_token
                    or chart_token.chart_index != subchart.chart_index
                    or chart_token.chart_generation_id != expected_chart.chart_digest
                    or chart_token.node_count != expected_chart.node_count
                ):
                    raise ValueError("native chart token disagrees with its streamed topology spec")
                sample_state = native.prepare_fixed_word_p0_sample_state_token(
                    chart_token,
                    global_track_count=ledger.global_track_count,
                    global_sample_count=ledger.global_frame_count,
                    global_sample_start=subchart.global_start,
                    global_sample_end=subchart.global_end,
                    global_loss_element_count=ledger.global_loss_element_count,
                    loss_normalization_id=ledger.loss_normalization_id,
                    sample_partition_generation_id=sample_partition_generation_id,
                    sample_block_size=sample_block_size,
                )
                for sample_block_id, sample_start, sample_end in _native_subchart_sample_blocks(
                    subchart,
                    topology_chart_index=partition.topology_chart_index,
                    sample_block_size=sample_block_size,
                    block_id=block_id,
                ):
                    if first_stage is not None and sample_start == first_range[1] and sample_end == first_range[2]:
                        staged = first_stage
                    else:
                        staged = ordered_plan.stage(
                            track_start=staging.stage_track_start,
                            track_end=staging.stage_track_end,
                            sample_start=sample_start,
                            sample_end=sample_end,
                            require_affine_ray_program=True,
                        )
                        _validate_staged_camera_block(
                            staged,
                            ordered_plan,
                            track_start=staging.stage_track_start,
                            track_end=staging.stage_track_end,
                            sample_start=sample_start,
                            sample_end=sample_end,
                            compact_rays=compact_rays,
                            validate_static_camera_program=False,
                            global_track_count=ledger.global_track_count,
                            global_sample_count=ledger.global_frame_count,
                            view_factor=staging.view_factor,
                        )
                    maximum_resident_target_elements = max(
                        maximum_resident_target_elements,
                        int(staged.targets.numel()),
                    )
                    block_sample_times_f64 = ordered_plan.sample_times[sample_start:sample_end].to(
                        device="cpu", dtype=torch.float64
                    )
                    maximum_resident_sample_time_bytes = max(
                        maximum_resident_sample_time_bytes,
                        block_sample_times_f64.numel() * block_sample_times_f64.element_size(),
                    )
                    sample_block = native.prepare_fixed_word_p0_sample_block_token(
                        sample_state,
                        staged.targets,
                        background,
                        sample_t_f64=block_sample_times_f64,
                        sample_block_id=sample_block_id,
                        global_sample_start=sample_start,
                        global_sample_end=sample_end,
                    )
                    native.fixed_word_p0_lie_sample_accumulate_loss_only_launch_only(
                        sample_block,
                        sample_state,
                    )
                    _synchronize_device(geometry.device, device_synchronize)
                    device_barrier_count += 1
                    del sample_block, staged, block_sample_times_f64
                    first_stage = None
                    sample_block_count += 1
                block_loss.add_(sample_state.loss_f32)
                native.fixed_word_p0_lie_node_vjp_accumulate_launch_only(
                    chart_token,
                    sample_state,
                    world_grad,
                )
                _synchronize_device(geometry.device, device_synchronize)
                device_barrier_count += 1
                del sample_state, chart_token
                native_subchart_count += 1
            native.fixed_word_p0_sparse_mobius_boundary_finalize_launch_only(world_grad)
            grad_sites = native.fixed_word_p0_site_geometry_finalize_launch_only(world_grad)
            grad_rgba = world_grad.grad_site_rgba_f32
            _synchronize_device(geometry.device, device_synchronize)
            device_barrier_count += 1
            _validate_native_gradients(ledger, prepared, grad_sites, grad_rgba)
            accumulate_prepared_rows_(
                ledger.gradients.grad_site_geometry,
                grad_sites[:, :4],
                source_site_ids,
            )
            accumulate_prepared_rows_(
                ledger.gradients.grad_site_weight,
                grad_sites[:, 4],
                source_site_ids,
            )
            accumulate_prepared_rows_(
                ledger.gradients.grad_site_color,
                grad_rgba[:, :3],
                source_site_ids,
            )
            accumulate_prepared_rows_(
                ledger.gradients.grad_site_density,
                grad_rgba[:, 3],
                source_site_ids,
            )
            ledger.compact_site_rows_accumulated += int(source_site_ids.numel())
            ledger.state_tensor_signatures = tuple(
                _tensor_signature(tensor) for tensor in (*ledger.gradients.tensors, ledger.loss)
            )
            del grad_sites, grad_rgba, world_grad, world_token, topology_token, first_stage
            selected_topology_charts += 1
            # Do not let Python loop locals retain the previous chart's CPU
            # payload or compact accelerator gathers while the provider enters
            # the next chart.  Without this explicit release, two topology
            # payloads can overlap transiently even though the context manager
            # itself reports only one active payload.
            del (
                compact_sites,
                compact_rays,
                compact_density,
                compact_color,
                compact_rgba,
                source_site_ids,
                source_track_ids,
                expected_native_partitions,
                topology,
                binding,
                prepared,
                payload,
            )

    if block_loss.untyped_storage().data_ptr() != block_loss_storage:
        raise ValueError("piecewise topology block loss tensor identity changed")
    if not bool(torch.isfinite(block_loss).item()):
        raise ValueError("piecewise topology block loss is non-finite")
    ledger.loss.add_(block_loss)
    ledger.consumed_block_ids.add(block_id)
    ledger.state_tensor_signatures = tuple(
        _tensor_signature(tensor) for tensor in (*ledger.gradients.tensors, ledger.loss)
    )
    delattr(ledger, _INFLIGHT_LEDGER_ATTRIBUTE)
    _assert_compact_spatial_step_current(ledger)
    event_gradients = tuple(
        NativeTopologyEventVJPMetadata(
            event_id=guard.event_id,
            guard_digest=guard.guard_digest,
            dispatch_kind=guard.dispatch_kind,
            root_lower_bound=guard.root_lower_bound,
            root_upper_bound=guard.root_upper_bound,
        )
        for guard in program.event_guards
    )
    return NativePiecewiseTopologyBlockResult(
        block_id=block_id,
        program_generation_digest=program.generation_digest,
        sample_partition_generation_id=sample_partition_generation_id,
        block_loss=block_loss,
        global_loss_element_count=ledger.global_loss_element_count,
        topology_chart_count=len(program.charts),
        selected_topology_chart_count=selected_topology_charts,
        native_subchart_count=native_subchart_count,
        sample_block_count=sample_block_count,
        device_barrier_count=device_barrier_count,
        maximum_resident_payload_count=1,
        maximum_resident_target_elements=maximum_resident_target_elements,
        maximum_resident_sample_time_bytes=maximum_resident_sample_time_bytes,
        event_gradients=event_gradients,
        exact_binary_sample_dispatch=True,
        continuous_real_native_boundary_equivalence_certified=(
            program.continuous_real_native_boundary_equivalence_certified
        ),
        paper_evidence_eligible=program.paper_evidence_eligible,
    )


def _ordered_piecewise_staging_plan(
    plan: PowerFoamTrackStagingPlan,
    program: NativePiecewiseTopologyProgram,
    *,
    device: torch.device,
) -> tuple[PowerFoamTrackStagingPlan, tuple[_NativeTopologyPartition, ...]]:
    assignments = []
    for time in plan.sample_times.tolist():
        outer = program.chart_index_for_binary_sample(float(time))
        inner = _native_subchart_for_binary_sample(
            Fraction.from_float(float(time)),
            program.charts[outer].native_subcharts,
        )
        assignments.append((outer, inner))
    permutation = sorted(range(len(assignments)), key=lambda index: assignments[index])
    order = torch.tensor(permutation, dtype=torch.long)
    ordered = PowerFoamTrackStagingPlan(
        target_provider=plan.target_provider,
        ray_provider=plan.ray_provider,
        pixel_indices=plan.pixel_indices,
        sample_indices=plan.sample_indices.index_select(0, order),
        height=plan.height,
        width=plan.width,
        sample_times=plan.sample_times.index_select(0, order),
        device=device,
    )
    ordered_assignments = tuple(assignments[index] for index in permutation)
    partitions = []
    cursor = 0
    while cursor < len(ordered_assignments):
        outer = ordered_assignments[cursor][0]
        outer_end = cursor + 1
        while outer_end < len(ordered_assignments) and ordered_assignments[outer_end][0] == outer:
            outer_end += 1
        subcharts = []
        subcursor = cursor
        while subcursor < outer_end:
            inner = ordered_assignments[subcursor][1]
            subend = subcursor + 1
            while subend < outer_end and ordered_assignments[subend] == (outer, inner):
                subend += 1
            subcharts.append(
                _NativeSubchartPartition(
                    chart_index=inner,
                    global_start=subcursor,
                    global_end=subend,
                )
            )
            subcursor = subend
        partitions.append(
            _NativeTopologyPartition(
                topology_chart_index=outer,
                global_start=cursor,
                global_end=outer_end,
                subcharts=tuple(subcharts),
            )
        )
        cursor = outer_end
    return ordered, tuple(partitions)


def _native_subchart_sample_blocks(
    partition: _NativeSubchartPartition,
    *,
    topology_chart_index: int,
    sample_block_size: int,
    block_id: str,
):
    """Yield one deterministic K partition without retaining O(F/K) records."""

    for start in range(partition.global_start, partition.global_end, sample_block_size):
        end = min(start + sample_block_size, partition.global_end)
        yield (
            f"{block_id}:topology-{topology_chart_index}:chart-{partition.chart_index}:samples-{start}-{end}",
            start,
            end,
        )


def _native_subchart_for_binary_sample(
    time: Fraction,
    charts: tuple[NativeFixedTopologySubchartSpec, ...],
) -> int:
    matches = []
    for index, chart in enumerate(charts):
        lower = Fraction.from_float(chart.t_min)
        upper = Fraction.from_float(chart.t_max)
        if lower <= time and (time < upper or (index + 1 == len(charts) and time <= upper)):
            matches.append(index)
    if len(matches) != 1:
        raise NativeTrackAdapterUnavailableError(
            "algebraically dispatched sample is not covered by exactly one native transfer chart"
        )
    return matches[0]


def _validate_payload(
    spec: NativePiecewiseTopologyChartSpec,
    payload: NativePiecewiseTopologyChartPayload,
    ledger: CompactSpatialStepLedger,
    *,
    block_id: str,
) -> None:
    if type(payload) is not NativePiecewiseTopologyChartPayload:
        raise ValueError("topology provider returned an unknown payload type")
    prepared = payload.prepared
    binding = payload.certificate_binding
    _assert_binding_matches_prepared(binding, prepared)
    if prepared.schedule.generation_digest != spec.schedule_generation_digest:
        raise ValueError("streamed topology payload changed its chart schedule")
    if _topology_content_digest(prepared) != spec.topology_content_digest:
        raise ValueError("streamed topology payload changed its compact CSR")
    if (
        native_geometry_ray_content_digest(
            prepared.source_tensors[0],
            prepared.source_tensors[1],
        )
        != spec.geometry_ray_content_digest
    ):
        raise ValueError("streamed topology payload changed its event geometry/ray provenance")
    if str(binding.canonical_digest) != spec.certificate_binding_digest:
        raise ValueError("streamed topology payload changed its certificate binding")
    if str(binding.binding_mode) != spec.binding_mode or bool(binding.paper_evidence_eligible) != (
        spec.binding_paper_evidence_eligible
    ):
        raise ValueError("streamed topology payload changed its certification mode")
    actual_subcharts = tuple(
        NativeFixedTopologySubchartSpec(
            chart_index=index,
            t_min=float(chart.transfer_atlas.t_min),
            t_max=float(chart.transfer_atlas.t_max),
            node_count=chart.node_count,
            chart_digest=binding.charts[index].chart_digest,
        )
        for index, chart in enumerate(prepared.world_snapshot.atlas.charts)
    )
    if actual_subcharts != spec.native_subcharts:
        raise ValueError("streamed topology payload changed its native subcharts")
    expected_tracks = torch.arange(
        spec.source_track_start,
        spec.source_track_end,
        dtype=prepared.topology.source_track_ids.dtype,
        device=prepared.topology.source_track_ids.device,
    )
    if not torch.equal(prepared.topology.source_track_ids, expected_tracks):
        raise ValueError("streamed topology payload changed its spatial track range")
    registered_start, registered_end = _registered_track_range(ledger, block_id)
    if (registered_start, registered_end) != (spec.source_track_start, spec.source_track_end):
        raise ValueError("streamed topology payload does not match the registered spatial block")
    if prepared.schedule.global_track_count != ledger.global_track_count:
        raise ValueError("streamed topology schedule changed the global track count")
    if len(prepared.source_tensors) != len(ledger.source_tensors) or any(
        actual is not expected for actual, expected in zip(prepared.source_tensors, ledger.source_tensors, strict=True)
    ):
        raise ValueError("streamed topology payload does not share the logical step's live world tensors")


def _validate_native_gradients(
    ledger: CompactSpatialStepLedger,
    prepared: PreparedCompactStagedLieWorld,
    grad_sites: torch.Tensor,
    grad_rgba: torch.Tensor,
) -> None:
    expected = (
        (grad_sites, (prepared.topology.site_count, 5)),
        (grad_rgba, (prepared.topology.site_count, 4)),
    )
    for tensor, shape in expected:
        if tuple(tensor.shape) != shape:
            raise ValueError("native topology gradient has an incompatible compact shape")
        if tensor.dtype != ledger.loss.dtype or tensor.device != ledger.loss.device:
            raise ValueError("native topology gradients must match the global ledger")
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError("native topology gradients must be finite")


def _topology_content_digest(prepared: PreparedCompactStagedLieWorld) -> str:
    digest = hashlib.sha256()
    topology = prepared.topology
    for tensor in (
        topology.source_track_ids,
        topology.source_boundary_ids,
        topology.source_site_ids,
        topology.word_offsets_i32,
        topology.word_owner_i32,
        topology.word_left_incidence_i32,
        topology.word_right_incidence_i32,
        topology.track_incidence_offsets_i32,
        topology.incidence_boundary_i32,
        topology.boundary_site_pairs_i32,
    ):
        cpu = tensor.detach().to(device="cpu").contiguous()
        digest.update(str(cpu.dtype).encode("ascii"))
        digest.update(json.dumps(tuple(cpu.shape)).encode("ascii"))
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def native_geometry_ray_content_digest(
    site_geometry: torch.Tensor,
    ray_coefficients: torch.Tensor,
) -> str:
    """Bind exact event predicates to the world/rays from which they arose."""

    digest = hashlib.sha256()
    for tensor in (torch.as_tensor(site_geometry), torch.as_tensor(ray_coefficients)):
        if not tensor.dtype.is_floating_point or not bool(torch.isfinite(tensor).all().item()):
            raise ValueError("event geometry/ray provenance tensors must be finite floating point")
        cpu = tensor.detach().to(device="cpu").contiguous()
        digest.update(str(cpu.dtype).encode("ascii"))
        digest.update(json.dumps(tuple(cpu.shape)).encode("ascii"))
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def _continuous_real_native_boundary_equivalence(program: NativePiecewiseTopologyProgram) -> bool:
    expected_bounds: list[Fraction | None] = [program.domain_t_min]
    expected_bounds.extend(guard.root_lower_bound if guard.root_exact else None for guard in program.event_guards)
    expected_bounds.append(program.domain_t_max)
    for index, chart in enumerate(program.charts):
        lower = Fraction.from_float(chart.native_subcharts[0].t_min)
        upper = Fraction.from_float(chart.native_subcharts[-1].t_max)
        if expected_bounds[index] is None or expected_bounds[index + 1] is None:
            return False
        if lower != expected_bounds[index] or upper != expected_bounds[index + 1]:
            return False
    return True


def _event_guard_strictly_before(
    left: NativeAlgebraicTopologyEventGuard,
    right: NativeAlgebraicTopologyEventGuard,
) -> bool:
    return left.root_upper_bound < right.root_lower_bound


def _guard_digest(guard: NativeAlgebraicTopologyEventGuard) -> str:
    payload = {
        "schema": "worldfoam-native-algebraic-event-guard-v1",
        "event_id": guard.event_id,
        "left_chart_id": guard.left_chart_id,
        "right_chart_id": guard.right_chart_id,
        "source_track_id": guard.source_track_id,
        "predicate_kind": guard.predicate_kind,
        "site_ids": guard.site_ids,
        "polynomial_coefficients": tuple(_fraction_payload(value) for value in guard.polynomial_coefficients),
        "certified_domain_t_min": _fraction_payload(guard.certified_domain_t_min),
        "certified_domain_t_max": _fraction_payload(guard.certified_domain_t_max),
        "root_lower_bound": _fraction_payload(guard.root_lower_bound),
        "root_upper_bound": _fraction_payload(guard.root_upper_bound),
        "root_exact": guard.root_exact,
        "root_multiplicity": guard.root_multiplicity,
        "sturm_root_count": guard.sturm_root_count,
        "polynomial_sign_at_lower": guard.polynomial_sign_at_lower,
        "polynomial_sign_at_upper": guard.polynomial_sign_at_upper,
        "seam_policy_id": guard.seam_policy_id,
        "geometry_ray_content_digest": guard.geometry_ray_content_digest,
        "compiler_provenance": guard.compiler_provenance,
        "zero_run_transfer_identity_certified": guard.zero_run_transfer_identity_certified,
        "classical_geometry_derivative_at_event_certified": (guard.classical_geometry_derivative_at_event_certified),
    }
    return _json_digest(payload)


def _chart_spec_digest(spec: NativePiecewiseTopologyChartSpec) -> str:
    payload = {
        "schema": "worldfoam-native-piecewise-topology-chart-v1",
        "chart_id": spec.chart_id,
        "source_track_range": (spec.source_track_start, spec.source_track_end),
        "schedule_generation_digest": spec.schedule_generation_digest,
        "topology_content_digest": spec.topology_content_digest,
        "geometry_ray_content_digest": spec.geometry_ray_content_digest,
        "certificate_binding_digest": spec.certificate_binding_digest,
        "binding_mode": spec.binding_mode,
        "binding_paper_evidence_eligible": spec.binding_paper_evidence_eligible,
        "native_subcharts": tuple(
            (chart.chart_index, chart.t_min, chart.t_max, chart.node_count, chart.chart_digest)
            for chart in spec.native_subcharts
        ),
        "chart_provenance": spec.chart_provenance,
    }
    return _json_digest(payload)


def _program_digest(program: NativePiecewiseTopologyProgram) -> str:
    payload = {
        "schema": "worldfoam-native-piecewise-topology-program-v1",
        "domain_t_min": _fraction_payload(program.domain_t_min),
        "domain_t_max": _fraction_payload(program.domain_t_max),
        "chart_spec_digests": tuple(chart.chart_spec_digest for chart in program.charts),
        "event_guard_digests": tuple(guard.guard_digest for guard in program.event_guards),
        "compiler_provenance": program.compiler_provenance,
        "seam_policy_id": program.seam_policy_id,
        "exact_binary_sample_dispatch": program.exact_binary_sample_dispatch,
        "event_time_vjp": program.event_time_vjp,
        "algebraic_event_dispatch_vjp": program.algebraic_event_dispatch_vjp,
        "frozen_topology_parameter_vjp": program.frozen_topology_parameter_vjp,
    }
    return _json_digest(payload)


def _json_digest(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fraction_payload(value: Fraction) -> tuple[int, int]:
    return value.numerator, value.denominator


def _evaluate_polynomial(coefficients: tuple[Fraction, ...], point: Fraction) -> Fraction:
    value = Fraction(0)
    for coefficient in reversed(coefficients):
        value = value * point + coefficient
    return value


def _as_fraction(value: FractionLike, *, name: str) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return Fraction.from_float(value)


def _sign(value: Fraction) -> int:
    return (value > 0) - (value < 0)


def _require_sha256(value: str, *, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


__all__ = [
    "NativeAlgebraicTopologyEventGuard",
    "NativeFixedTopologySubchartSpec",
    "NativePiecewiseTopologyBlockResult",
    "NativePiecewiseTopologyChartPayload",
    "NativePiecewiseTopologyChartSpec",
    "NativePiecewiseTopologyProgram",
    "NativeTopologyEventVJPMetadata",
    "describe_native_piecewise_topology_chart",
    "execute_native_piecewise_topology_track_block",
    "make_native_algebraic_topology_event_guard",
    "make_native_piecewise_topology_program",
    "native_geometry_ray_content_digest",
]
