"""Block-scoped geometry reduction for full native equal-rank VJPs.

The native equal-rank reverse returns one bounded ``[J, W_b]`` cotangent for
the physical lengths of all CSR rows in a block.  This module maps those
columns back to the frame-free kinetic source programs and reduces them to
world geometry and affine-ray parameter bars:

* one analytic ``kinetic_p0_node_physical_length_geometry_vjp`` per CSR row;
* additive global ``positions0``, velocity, and polynomial-weight bars;
* optional additive compact affine-ray bars keyed by
  ``(view_index, track_id)``, including repeated track ids from multiple
  charts; fixed-camera training omits their aggregate storage;
* no requested frame, sample, target, prediction, or material state.

The caller-supplied device completion fence and both fence/native-result
provenance ids are mandatory.  No native tensor value is copied, reduced, or
inspected before the single callback invocation.  This source bridge does not
verify that the callback implements real device/stream fence semantics.  The
returned object retains only CPU parameter bars and scalar provenance; it
never retains the native result, world, runtime, block, sampler, or row
scratch.

Selected logical-byte accounting covers the native length-bar input, its CPU
float64 copy, returned parameter bars, and the largest row-result/frozen-word
scratch visible at this bridge boundary.  It is not a measured or exhaustive
peak: validation temporaries, internal tensors inside the analytic geometry
kernel, allocator storage/peak, and Python objects remain explicitly
unmeasured.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from types import MappingProxyType

import torch
from kinetic_native_equal_rank_lowering import KineticNativeEqualRankRowSpec
from kinetic_native_equal_rank_runtime_adapter import (
    KineticNativeEqualRankVJPResult,
)
from kinetic_stable_stratum_vjp import (
    DERIVATIVE_SCOPE as ANALYTIC_GEOMETRY_DERIVATIVE_SCOPE,
)
from kinetic_stable_stratum_vjp import (
    KineticP0NodePhysicalLengthGeometryVJP,
    StableStratumThresholds,
    kinetic_p0_node_physical_length_geometry_vjp,
    make_frozen_kinetic_owner_word,
)
from paper_kinetic_ragged_sample_plan import (
    PaperKineticRowBinding,
    PaperKineticRowRaggedSampler,
)

GEOMETRY_REDUCTION_PROVENANCE = "kinetic-native-equal-rank-geometry-reduction-v1"
GEOMETRY_DERIVATIVE_SCOPE = "frozen_owner_topology_chart_rank_node_times"

_RESULT_SEAL = object()


@dataclass(frozen=True)
class KineticNativeEqualRankGeometryReductionMemory:
    """F-labelled logical bytes; only ``requested_frame_count`` may vary."""

    requested_frame_count: int
    native_length_bar_input_tensor_bytes: int
    cpu_length_bar_copy_tensor_bytes: int
    grad_positions0_tensor_bytes: int
    grad_velocities_tensor_bytes: int
    grad_weight_coefficients_tensor_bytes: int
    grad_track_ray_coefficients_tensor_bytes: int
    output_parameter_bar_tensor_bytes: int
    maximum_row_result_tensor_bytes: int
    maximum_frozen_word_tensor_bytes: int
    bridge_visible_peak_logical_tensor_bytes: int
    persistent_frame_tensor_bytes: int
    persistent_sample_tensor_bytes: int
    persistent_target_tensor_bytes: int
    persistent_prediction_tensor_bytes: int
    persistent_material_tensor_bytes: int
    dense_row_by_global_time_tensor_bytes: int
    frame_by_word_reverse_state_tensor_bytes: int
    analytic_kernel_internal_allocator_peak_measured: bool
    allocator_storage_bytes_measured: bool
    allocator_peak_measured: bool
    python_object_bytes_measured: bool


@dataclass(frozen=True)
class KineticNativeEqualRankGeometryReduction:
    """CPU geometry bars from exactly one fenced native equal-rank block."""

    sampler_generation_digest: str
    native_block_generation_digest: str
    native_world_generation_id: str
    native_vjp_provenance_id: str
    device_completion_fence_provenance: str
    view_index: int
    track_ids: tuple[int, ...]
    ray_gradients_included: bool
    ray_bar_keys: tuple[tuple[int, int], ...]
    grad_positions0_f64: torch.Tensor = field(repr=False)
    grad_velocities_f64: torch.Tensor = field(repr=False)
    grad_weight_coefficients_f64: torch.Tensor = field(repr=False)
    grad_track_ray_coefficients_f64: torch.Tensor = field(repr=False)
    row_count: int
    node_count: int
    word_count: int
    site_count: int
    weight_coefficient_count: int
    row_geometry_vjp_call_count: int
    device_completion_fence_call_count: int
    differentiable_word_reverse_interactions: int
    dense_global_site_accumulation_elements: int
    all_site_owner_validation_evaluations: int
    memory: KineticNativeEqualRankGeometryReductionMemory
    accounting: MappingProxyType
    generation_digest: str
    tensor_signatures: tuple[tuple[object, ...], ...] = field(repr=False)
    tensor_content_digests: tuple[str, ...] = field(repr=False)
    provenance: str = GEOMETRY_REDUCTION_PROVENANCE
    derivative_scope: str = GEOMETRY_DERIVATIVE_SCOPE
    requested_frame_sampling_used: bool = False
    frame_or_sample_axis_retained: bool = False
    target_prediction_or_material_retained: bool = False
    native_result_retained: bool = False
    native_world_or_runtime_retained: bool = False
    row_scratch_retained: bool = False
    geometry_vjp_implemented: bool = True
    real_device_completion_fence_semantics_verified: bool = False
    upstream_request_receipt_bound: bool = False
    allocator_peak_measured: bool = False
    event_time_derivatives_included: bool = False
    chart_endpoint_derivatives_included: bool = False
    node_time_or_rank_derivatives_included: bool = False
    compiler_choice_derivatives_included: bool = False
    _seal: object = field(default=None, repr=False)

    @property
    def track_count(self) -> int:
        return len(self.track_ids)

    @property
    def output_parameter_bar_tensor_bytes(self) -> int:
        return _tensor_bytes(self._tensors())

    def _tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.grad_positions0_f64,
            self.grad_velocities_f64,
            self.grad_weight_coefficients_f64,
            self.grad_track_ray_coefficients_f64,
        )

    def memory_report(
        self,
        requested_frame_count: int,
    ) -> KineticNativeEqualRankGeometryReductionMemory:
        _require_positive_int(requested_frame_count, name="requested_frame_count")
        return KineticNativeEqualRankGeometryReductionMemory(
            **{
                **self.memory.__dict__,
                "requested_frame_count": requested_frame_count,
            }
        )

    def assert_current(self) -> None:
        if (
            self._seal is not _RESULT_SEAL
            or self.provenance != GEOMETRY_REDUCTION_PROVENANCE
            or self.derivative_scope != GEOMETRY_DERIVATIVE_SCOPE
            or self.requested_frame_sampling_used
            or self.frame_or_sample_axis_retained
            or self.target_prediction_or_material_retained
            or self.native_result_retained
            or self.native_world_or_runtime_retained
            or self.row_scratch_retained
            or not self.geometry_vjp_implemented
            or not isinstance(self.ray_gradients_included, bool)
            or self.real_device_completion_fence_semantics_verified
            or self.upstream_request_receipt_bound
            or self.allocator_peak_measured
            or self.event_time_derivatives_included
            or self.chart_endpoint_derivatives_included
            or self.node_time_or_rank_derivatives_included
            or self.compiler_choice_derivatives_included
            or self.device_completion_fence_call_count != 1
            or self.row_geometry_vjp_call_count != self.row_count
            or self.row_count < 1
            or self.node_count < 2
            or self.word_count < self.row_count
            or self.site_count < 1
            or self.weight_coefficient_count < 1
            or self.view_index < 0
            or not self.track_ids
            or tuple(sorted(set(self.track_ids))) != self.track_ids
            or self.ray_bar_keys
            != (
                tuple((self.view_index, track_id) for track_id in self.track_ids)
                if self.ray_gradients_included
                else ()
            )
            or self.differentiable_word_reverse_interactions != self.node_count * self.word_count
            or self.dense_global_site_accumulation_elements
            != self.row_count * self.site_count * (6 + self.weight_coefficient_count)
            or self.all_site_owner_validation_evaluations
            != self.node_count * 3 * self.word_count * (self.site_count - 1)
        ):
            raise ValueError("equal-rank geometry reduction execution/memory contract changed")
        for name, value in (
            ("sampler_generation_digest", self.sampler_generation_digest),
            ("native_block_generation_digest", self.native_block_generation_digest),
            ("native_vjp_provenance_id", self.native_vjp_provenance_id),
        ):
            _require_sha256(value, name=name)
        _require_provenance(
            self.native_world_generation_id,
            name="native_world_generation_id",
        )
        _require_provenance(
            self.device_completion_fence_provenance,
            name="device_completion_fence_provenance",
        )
        tensors = self._tensors()
        if tuple(_tensor_signature(tensor) for tensor in tensors) != self.tensor_signatures:
            raise ValueError("equal-rank geometry reduction tensor identity/layout/version changed")
        expected_shapes = (
            (self.site_count, 3),
            (self.site_count, 3),
            (self.site_count, self.weight_coefficient_count),
            (self.track_count if self.ray_gradients_included else 0, 12),
        )
        for tensor, shape in zip(tensors, expected_shapes, strict=True):
            _require_cpu_f64_tensor(tensor, shape=shape)
        if tuple(_tensor_digest(tensor) for tensor in tensors) != self.tensor_content_digests:
            raise ValueError("equal-rank geometry reduction tensor content changed")
        if self.output_parameter_bar_tensor_bytes != self.memory.output_parameter_bar_tensor_bytes:
            raise ValueError("equal-rank geometry reduction output-byte accounting changed")
        if dict(self.accounting) != _result_accounting(self):
            raise ValueError("equal-rank geometry reduction accounting changed")
        if self.generation_digest != _result_digest(self):
            raise ValueError("equal-rank geometry reduction generation changed")


def kinetic_native_equal_rank_vjp_provenance_id(
    native_vjp: KineticNativeEqualRankVJPResult,
) -> str:
    """Bind one sealed native result without reading its tensor values."""

    if not isinstance(native_vjp, KineticNativeEqualRankVJPResult):
        raise TypeError("native_vjp must be KineticNativeEqualRankVJPResult")
    native_vjp.assert_warm_layout()
    world = native_vjp.world
    runtime = world.runtime
    return _digest_parts(
        GEOMETRY_REDUCTION_PROVENANCE,
        "native-vjp",
        world.generation_id,
        runtime.generation_id,
        runtime.payload.block.generation_digest,
        runtime.native_abi_identity,
        _tensor_signature(native_vjp.grad_node_physical_length_f32),
        native_vjp.accounting,
    )


@torch.no_grad()
def reduce_kinetic_native_equal_rank_geometry_vjp(
    native_vjp: KineticNativeEqualRankVJPResult,
    sampler: PaperKineticRowRaggedSampler,
    *,
    expected_native_vjp_provenance_id: str,
    device_completion_fence: Callable[[], None],
    device_completion_fence_provenance: str,
    maximum_bridge_visible_peak_logical_tensor_bytes: int,
    include_ray_gradients: bool = True,
    thresholds: StableStratumThresholds = StableStratumThresholds(),
) -> KineticNativeEqualRankGeometryReduction:
    """Fence, copy, and reduce one full native ``[J,W_b]`` geometry bar."""

    if not isinstance(native_vjp, KineticNativeEqualRankVJPResult):
        raise TypeError("native_vjp must be KineticNativeEqualRankVJPResult")
    if not isinstance(sampler, PaperKineticRowRaggedSampler):
        raise TypeError("sampler must be PaperKineticRowRaggedSampler")
    if not isinstance(thresholds, StableStratumThresholds):
        raise TypeError("thresholds must be StableStratumThresholds")
    if not isinstance(include_ray_gradients, bool):
        raise TypeError("include_ray_gradients must be bool")
    if not callable(device_completion_fence):
        raise TypeError("device_completion_fence must be callable")
    _require_sha256(
        expected_native_vjp_provenance_id,
        name="expected_native_vjp_provenance_id",
    )
    _require_provenance(
        device_completion_fence_provenance,
        name="device_completion_fence_provenance",
    )
    _require_positive_int(
        maximum_bridge_visible_peak_logical_tensor_bytes,
        name="maximum_bridge_visible_peak_logical_tensor_bytes",
    )

    # Warm validation and scalar metadata only: no device tensor value read.
    native_vjp.assert_warm_layout()
    sampler.assert_cold_current()
    actual_native_provenance = kinetic_native_equal_rank_vjp_provenance_id(native_vjp)
    if actual_native_provenance != expected_native_vjp_provenance_id:
        raise ValueError("native equal-rank VJP provenance changed")
    block = native_vjp.world.runtime.payload.block
    row_bindings, row_specs = _block_rows(sampler, block.generation_digest)
    _validate_block_mapping(
        native_vjp,
        sampler,
        row_bindings=row_bindings,
        row_specs=row_specs,
    )
    first_program = row_bindings[0].program
    sites = first_program.binding.sites
    if any(binding.program.binding.sites is not sites for binding in row_bindings):
        raise ValueError("one native block cannot mix kinetic world tensor identities")
    track_ids = tuple(sorted({binding.track_id for binding in row_bindings}))
    weight_count = int(sites.weight_coefficients.shape[1])
    memory = _preflight_memory(
        native_vjp,
        site_count=sites.site_count,
        weight_coefficient_count=weight_count,
        track_count=len(track_ids),
        include_ray_gradients=include_ray_gradients,
        maximum_row_word_count=max(row.word_count for row in row_specs),
    )
    if memory.bridge_visible_peak_logical_tensor_bytes > maximum_bridge_visible_peak_logical_tensor_bytes:
        raise MemoryError("equal-rank geometry reduction exceeds its preflight byte budget")

    # This is the sole caller-supplied completion-callback invocation.
    # Everything above used only sealed Python metadata, shapes, identities,
    # layouts, and versions; real device/stream synchronization is not proven
    # by this source bridge.
    fence_result = device_completion_fence()
    fence_call_count = 1
    if fence_result is not None:
        raise TypeError("device_completion_fence must return None")
    native_vjp.assert_warm_layout()
    if kinetic_native_equal_rank_vjp_provenance_id(native_vjp) != actual_native_provenance:
        raise ValueError("native equal-rank VJP changed across the completion fence")
    sampler.assert_cold_current()
    grad_lengths_cpu_f64 = (
        native_vjp.grad_node_physical_length_f32.detach().to(device="cpu", dtype=torch.float64).contiguous()
    )
    if not bool(torch.isfinite(grad_lengths_cpu_f64).all().item()):
        raise ValueError("native equal-rank physical-length bar is nonfinite")

    grad_positions0 = torch.zeros_like(sites.positions0, device="cpu", dtype=torch.float64)
    grad_velocities = torch.zeros_like(sites.velocities, device="cpu", dtype=torch.float64)
    grad_weights = torch.zeros_like(
        sites.weight_coefficients,
        device="cpu",
        dtype=torch.float64,
    )
    grad_rays = torch.zeros(
        (len(track_ids) if include_ray_gradients else 0, 12),
        dtype=torch.float64,
    )
    track_to_compact = (
        {track_id: index for index, track_id in enumerate(track_ids)}
        if include_ray_gradients
        else {}
    )

    word_start = 0
    row_call_count = 0
    differentiable_word_reverse_interactions = 0
    dense_global_site_accumulation_elements = 0
    all_site_owner_validation_evaluations = 0
    for binding, row in zip(row_bindings, row_specs, strict=True):
        word_end = word_start + row.word_count
        source = binding.source
        chart = binding.program.charts[binding.chart_index]
        topology_chart = source.lowering.charts[binding.chart_index]
        _require_sha256(
            topology_chart.owner_topology_certificate_digest,
            name="continuous_topology_certificate_digest",
        )
        row_cotangent_provenance = _digest_parts(
            GEOMETRY_REDUCTION_PROVENANCE,
            expected_native_vjp_provenance_id,
            device_completion_fence_provenance,
            block.generation_digest,
            row.row_identity_digest,
        )
        row_result = kinetic_p0_node_physical_length_geometry_vjp(
            binding.program.binding.sites,
            binding.program.binding.ray_coefficients,
            chart.schedule.node_times,
            (make_frozen_kinetic_owner_word(row.owner_word),),
            grad_lengths_cpu_f64[:, word_start:word_end],
            near=row.near,
            far=row.far,
            continuous_topology_certificate_id=(topology_chart.owner_topology_certificate_digest),
            node_physical_length_cotangent_provenance_id=(row_cotangent_provenance),
            thresholds=thresholds,
        )
        _validate_analytic_row_result(
            row_result,
            continuous_topology_certificate_id=(topology_chart.owner_topology_certificate_digest),
            node_physical_length_cotangent_provenance_id=(row_cotangent_provenance),
        )
        expected_lengths = chart.node_physical_lengths.to(dtype=torch.float64, device="cpu")
        if not torch.allclose(
            row_result.node_physical_lengths,
            expected_lengths,
            rtol=2.0e-6,
            atol=2.0e-7,
        ):
            raise ValueError("geometry row recompute disagrees with compiled native lengths")
        grad_positions0.add_(row_result.grad_positions0)
        grad_velocities.add_(row_result.grad_velocities)
        grad_weights.add_(row_result.grad_weight_coefficients)
        if include_ray_gradients:
            grad_rays[track_to_compact[binding.track_id]].add_(
                row_result.grad_ray_coefficients
            )
        differentiable_word_reverse_interactions += int(row_result.accounting["physical_length_reverse_interactions"])
        dense_global_site_accumulation_elements += sites.site_count * (6 + weight_count)
        all_site_owner_validation_evaluations += int(row_result.accounting["owner_margin_evaluations"])
        word_start = word_end
        row_call_count += 1
        del row_result, source, chart, topology_chart
    if word_start != block.word_count:
        raise ValueError("geometry reduction did not consume the exact native CSR word bar")
    del grad_lengths_cpu_f64

    tensors = (grad_positions0, grad_velocities, grad_weights, grad_rays)
    tensor_signatures = tuple(_tensor_signature(tensor) for tensor in tensors)
    tensor_digests = tuple(_tensor_digest(tensor) for tensor in tensors)
    provisional = KineticNativeEqualRankGeometryReduction(
        sampler_generation_digest=sampler.generation_digest,
        native_block_generation_digest=block.generation_digest,
        native_world_generation_id=native_vjp.world.generation_id,
        native_vjp_provenance_id=expected_native_vjp_provenance_id,
        device_completion_fence_provenance=device_completion_fence_provenance,
        view_index=sampler.view_index,
        track_ids=track_ids,
        ray_gradients_included=include_ray_gradients,
        ray_bar_keys=(
            tuple((sampler.view_index, track_id) for track_id in track_ids)
            if include_ray_gradients
            else ()
        ),
        grad_positions0_f64=grad_positions0,
        grad_velocities_f64=grad_velocities,
        grad_weight_coefficients_f64=grad_weights,
        grad_track_ray_coefficients_f64=grad_rays,
        row_count=len(row_bindings),
        node_count=block.node_count,
        word_count=block.word_count,
        site_count=sites.site_count,
        weight_coefficient_count=weight_count,
        row_geometry_vjp_call_count=row_call_count,
        device_completion_fence_call_count=fence_call_count,
        differentiable_word_reverse_interactions=(differentiable_word_reverse_interactions),
        dense_global_site_accumulation_elements=(dense_global_site_accumulation_elements),
        all_site_owner_validation_evaluations=(all_site_owner_validation_evaluations),
        memory=memory,
        accounting=MappingProxyType({}),
        generation_digest="",
        tensor_signatures=tensor_signatures,
        tensor_content_digests=tensor_digests,
        _seal=_RESULT_SEAL,
    )
    accounting = MappingProxyType(_result_accounting(provisional))
    with_accounting = replace(provisional, accounting=accounting)
    result = replace(
        with_accounting,
        generation_digest=_result_digest(with_accounting),
    )
    result.assert_current()
    return result


def _validate_analytic_row_result(
    result: KineticP0NodePhysicalLengthGeometryVJP,
    *,
    continuous_topology_certificate_id: str,
    node_physical_length_cotangent_provenance_id: str,
) -> None:
    if not isinstance(result, KineticP0NodePhysicalLengthGeometryVJP):
        raise TypeError("analytic geometry row VJP returned an invalid result type")
    if (
        result.derivative_scope != ANALYTIC_GEOMETRY_DERIVATIVE_SCOPE
        or not result.geometry_vjp_implemented
        or result.material_gradients_included
        or result.event_time_derivatives_included
        or result.chart_endpoint_derivatives_included
        or result.node_time_or_rank_derivatives_included
        or result.compiler_choice_derivatives_included
        or int(result.accounting.get("material_gradient_tensors_emitted", -1)) != 0
        or hasattr(result, "grad_site_density")
        or hasattr(result, "grad_site_color")
    ):
        raise ValueError("analytic geometry row VJP derivative/omission contract changed")
    if result.continuous_topology_certificate_id != continuous_topology_certificate_id:
        raise ValueError("analytic geometry row VJP topology certificate changed")
    if result.node_physical_length_cotangent_provenance_id != node_physical_length_cotangent_provenance_id:
        raise ValueError("analytic geometry row VJP cotangent provenance changed")


def _block_rows(
    sampler: PaperKineticRowRaggedSampler,
    block_generation_digest: str,
) -> tuple[
    tuple[PaperKineticRowBinding, ...],
    tuple[KineticNativeEqualRankRowSpec, ...],
]:
    bindings = tuple(
        sorted(
            (row for row in sampler.rows if row.native_block_generation_digest == block_generation_digest),
            key=lambda row: row.native_local_row_index,
        )
    )
    blocks = tuple(
        block
        for bucket in sampler.lowering.buckets
        for block in bucket.blocks
        if block.generation_digest == block_generation_digest
    )
    if len(blocks) != 1 or not bindings:
        raise ValueError("native VJP block is foreign to the ragged sampler")
    block = blocks[0]
    rows_by_index = {row.global_row_index: row for row in sampler.lowering.rows}
    rows = tuple(rows_by_index[index] for index in block.global_row_indices)
    return bindings, rows


def _validate_block_mapping(
    native_vjp: KineticNativeEqualRankVJPResult,
    sampler: PaperKineticRowRaggedSampler,
    *,
    row_bindings: tuple[PaperKineticRowBinding, ...],
    row_specs: tuple[KineticNativeEqualRankRowSpec, ...],
) -> None:
    runtime = native_vjp.world.runtime
    block = runtime.payload.block
    if (
        len(row_bindings) != block.row_count
        or len(row_specs) != block.row_count
        or tuple(binding.native_local_row_index for binding in row_bindings) != tuple(range(block.row_count))
        or tuple(binding.global_row_index for binding in row_bindings) != block.global_row_indices
        or tuple(binding.row_identity_digest for binding in row_bindings) != block.row_identity_digests
        or tuple(row.global_row_index for row in row_specs) != block.global_row_indices
        or sum(row.word_count for row in row_specs) != block.word_count
        or sampler.lowering.global_site_count != runtime.global_site_count
    ):
        raise ValueError("native VJP CSR rows do not match the ragged sampler")
    for binding, row in zip(row_bindings, row_specs, strict=True):
        source = binding.source
        if (
            binding.track_id != row.track_id
            or binding.chart_index != row.chart_index
            or binding.node_count != row.node_count
            or row.node_count != block.node_count
            or source.row_identity != row.row_identity
            or source.program is not binding.program
            or source.program.generation_digest != row.kinetic_program_generation_digest
            or source.lowering.generation_digest != row.topology_lowering_generation_digest
        ):
            raise ValueError("native VJP row source/program provenance changed")


def _preflight_memory(
    native_vjp: KineticNativeEqualRankVJPResult,
    *,
    site_count: int,
    weight_coefficient_count: int,
    track_count: int,
    include_ray_gradients: bool,
    maximum_row_word_count: int,
) -> KineticNativeEqualRankGeometryReductionMemory:
    block = native_vjp.world.runtime.payload.block
    input_bytes = block.node_count * block.word_count * 4
    cpu_copy_bytes = block.node_count * block.word_count * 8
    positions_bytes = site_count * 3 * 8
    velocities_bytes = positions_bytes
    weights_bytes = site_count * weight_coefficient_count * 8
    rays_bytes = track_count * 12 * 8 if include_ray_gradients else 0
    output_bytes = positions_bytes + velocities_bytes + weights_bytes + rays_bytes
    maximum_row_result_bytes = (
        block.node_count * maximum_row_word_count * 8 + positions_bytes + velocities_bytes + weights_bytes + 12 * 8
    )
    frozen_word_bytes = maximum_row_word_count * 8
    peak = input_bytes + cpu_copy_bytes + output_bytes + maximum_row_result_bytes + frozen_word_bytes
    return KineticNativeEqualRankGeometryReductionMemory(
        requested_frame_count=1,
        native_length_bar_input_tensor_bytes=input_bytes,
        cpu_length_bar_copy_tensor_bytes=cpu_copy_bytes,
        grad_positions0_tensor_bytes=positions_bytes,
        grad_velocities_tensor_bytes=velocities_bytes,
        grad_weight_coefficients_tensor_bytes=weights_bytes,
        grad_track_ray_coefficients_tensor_bytes=rays_bytes,
        output_parameter_bar_tensor_bytes=output_bytes,
        maximum_row_result_tensor_bytes=maximum_row_result_bytes,
        maximum_frozen_word_tensor_bytes=frozen_word_bytes,
        bridge_visible_peak_logical_tensor_bytes=peak,
        persistent_frame_tensor_bytes=0,
        persistent_sample_tensor_bytes=0,
        persistent_target_tensor_bytes=0,
        persistent_prediction_tensor_bytes=0,
        persistent_material_tensor_bytes=0,
        dense_row_by_global_time_tensor_bytes=0,
        frame_by_word_reverse_state_tensor_bytes=0,
        analytic_kernel_internal_allocator_peak_measured=False,
        allocator_storage_bytes_measured=False,
        allocator_peak_measured=False,
        python_object_bytes_measured=False,
    )


def _result_accounting(
    result: KineticNativeEqualRankGeometryReduction,
) -> dict[str, int | str | bool]:
    return {
        "provenance": result.provenance,
        "derivative_scope": result.derivative_scope,
        "view_index": result.view_index,
        "ray_bar_key_kind": (
            "(view_index, track_id)"
            if result.ray_gradients_included
            else "disabled/fixed_camera"
        ),
        "ray_gradients_included": result.ray_gradients_included,
        "row_count": result.row_count,
        "track_count": result.track_count,
        "node_count": result.node_count,
        "word_count": result.word_count,
        "row_geometry_vjp_call_count": result.row_geometry_vjp_call_count,
        "device_completion_fence_call_count": result.device_completion_fence_call_count,
        "differentiable_word_reverse_interactions": (result.differentiable_word_reverse_interactions),
        "differentiable_word_reverse_scaling": "O(J * sum_rows R_row)",
        "dense_global_site_accumulation_elements": (result.dense_global_site_accumulation_elements),
        "dense_global_site_accumulation_scaling": ("O(row_count * S * (6 + weight_coefficient_count))"),
        "all_site_owner_validation_evaluations": (result.all_site_owner_validation_evaluations),
        "all_site_owner_validation_scaling": "O(J * S * sum_rows R_row)",
        "aggregate_reverse_runtime_scaling_claimed": False,
        "native_length_bar_device_to_cpu_copy_count": 1,
        "native_result_retained": False,
        "native_world_or_runtime_retained": False,
        "row_scratch_retained": False,
        "requested_frame_count_used": 0,
        "requested_sample_count_used": 0,
        "persistent_frame_tensor_bytes": 0,
        "persistent_sample_tensor_bytes": 0,
        "persistent_target_tensor_bytes": 0,
        "persistent_prediction_tensor_bytes": 0,
        "persistent_material_tensor_bytes": 0,
        "frame_by_word_reverse_state_tensor_bytes": 0,
        "output_parameter_bar_tensor_bytes": result.output_parameter_bar_tensor_bytes,
        "bridge_visible_peak_logical_tensor_bytes": (result.memory.bridge_visible_peak_logical_tensor_bytes),
        "frame_scaling": "independent of requested F after node reduction",
        "geometry_vjp_implemented": True,
        "event_time_derivatives_included": False,
        "chart_endpoint_derivatives_included": False,
        "node_time_or_rank_derivatives_included": False,
        "compiler_choice_derivatives_included": False,
        "real_device_completion_fence_semantics_verified": False,
        "upstream_request_receipt_bound": False,
        "analytic_kernel_internal_allocator_peak_measured": False,
        "allocator_peak_measured": False,
    }


def _result_digest(result: KineticNativeEqualRankGeometryReduction) -> str:
    return _digest_parts(
        GEOMETRY_REDUCTION_PROVENANCE,
        result.sampler_generation_digest,
        result.native_block_generation_digest,
        result.native_world_generation_id,
        result.native_vjp_provenance_id,
        result.device_completion_fence_provenance,
        result.view_index,
        result.track_ids,
        result.ray_gradients_included,
        result.ray_bar_keys,
        result.row_count,
        result.node_count,
        result.word_count,
        result.site_count,
        result.weight_coefficient_count,
        result.row_geometry_vjp_call_count,
        result.device_completion_fence_call_count,
        result.differentiable_word_reverse_interactions,
        result.dense_global_site_accumulation_elements,
        result.all_site_owner_validation_evaluations,
        result.memory,
        tuple(sorted(dict(result.accounting).items())),
        result.tensor_content_digests,
    )


def _tensor_signature(tensor: torch.Tensor) -> tuple[object, ...]:
    return (
        id(tensor),
        int(getattr(tensor, "_version", 0)),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
        tensor.layout,
        tensor.requires_grad,
        tensor.is_contiguous(),
    )


def _tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(repr(tuple(value.shape)).encode("utf-8"))
    if value.numel():
        digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()


def _tensor_bytes(tensors: tuple[torch.Tensor, ...]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _require_cpu_f64_tensor(tensor: torch.Tensor, *, shape: tuple[int, ...]) -> None:
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.device.type != "cpu"
        or tensor.dtype != torch.float64
        or tensor.layout != torch.strided
        or tuple(tensor.shape) != shape
        or not tensor.is_contiguous()
        or tensor.requires_grad
        or not bool(torch.isfinite(tensor).all().item())
    ):
        raise ValueError("geometry reduction tensor has invalid CPU float64 layout/content")


def _digest_parts(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = repr(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _require_sha256(value: str, *, name: str) -> None:
    if len(value) != 64:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    try:
        parsed = bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest") from error
    if len(parsed) != 32 or value != value.lower():
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_provenance(value: str, *, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty")


def _require_positive_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


__all__ = [
    "GEOMETRY_DERIVATIVE_SCOPE",
    "GEOMETRY_REDUCTION_PROVENANCE",
    "KineticNativeEqualRankGeometryReduction",
    "KineticNativeEqualRankGeometryReductionMemory",
    "kinetic_native_equal_rank_vjp_provenance_id",
    "reduce_kinetic_native_equal_rank_geometry_vjp",
]
