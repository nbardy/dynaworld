"""CPU/fake-native-only bounded full-geometry finalization candidate.

The existing paper scheduler reduces arbitrarily many bounded temporal chunks
to one ``[row, node, 4]`` cotangent per active equal-rank native block.  This
module implements only the missing finalization seam, rather than cloning that
scheduler:

``complete node bar -> one full native VJP -> one fence -> geometry reduction``.

Material bars use the existing union-local assembly and outer global material
coordinator.  Physical-length bars use the frozen-stratum geometry bridge and
add into shared site-position, velocity, polynomial-weight, and compact affine
ray bars.  Rays are keyed by ``(view_index, track_id)`` so repeated charts and
rank blocks add rather than overwrite.

The structural program is fixed.  Event/chart/rank/node-time/compiler-choice
derivatives are excluded.  This seam retains no frame, sample, target,
prediction, interpolation-weight, native result, world, or runtime after the
request is consumed. Every public entry and finalization boundary rejects
accelerator tensors; this module is not an MPS/CUDA lifetime seam. The full
reverse and its exact block sample coverage must
arrive in an executor-sealed execution receipt and match an independently
prepared canonical manifest; this module accepts no free-standing node bar or
caller-reported coverage count. Native build, real fence semantics, allocator
peaks, and the whole-step dataset/trainer coordinator remain external gates.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType

import torch
from kinetic_native_equal_rank_geometry_reduction import (
    GEOMETRY_DERIVATIVE_SCOPE,
    kinetic_native_equal_rank_vjp_provenance_id,
    reduce_kinetic_native_equal_rank_geometry_vjp,
)
from kinetic_native_equal_rank_runtime_adapter import (
    KineticNativeEqualRankVJPResult,
)
from kinetic_native_material_step_executor import (
    EXECUTOR_PROVENANCE,
    KineticNativeFullGeometryVJPExecution,
    KineticNativeMaterialStepSession,
    KineticNativeMaterialStepTelemetry,
)
from paper_kinetic_ragged_sample_plan import (
    iter_paper_kinetic_row_ragged_request_blocks,
)
from paper_kinetic_union_local_bar_assembly import (
    PaperKineticUnionLocalBarAssembly,
    PaperKineticUnionLocalRequestWork,
    begin_paper_kinetic_union_local_bar_assembly,
    consume_paper_kinetic_union_local_native_contribution,
    finalize_paper_kinetic_union_local_bar_assembly,
    seal_paper_kinetic_native_block_vjp_contribution,
)
from paper_ragged_material_bar_coordinator import (
    PaperRaggedMaterialBarRequest,
    PaperRaggedMaterialBarStepLedger,
    _assert_ledger_current as _assert_material_ledger_current,
    consume_paper_ragged_compact_material_bar_result,
)

FULL_GEOMETRY_PROVENANCE = "paper-kinetic-full-geometry-request-v1"
FULL_GEOMETRY_STATUS = "cpu_fake_native_only/no_accelerator_admission"

_ASSEMBLY_SEAL = object()
_BLOCK_MANIFEST_SEAL = object()
_BLOCK_RECEIPT_SEAL = object()
_RECEIPT_SEAL = object()


@dataclass(frozen=True)
class PaperKineticFullGeometryMemoryReport:
    """Selected logical bytes; not an allocator or exhaustive peak claim."""

    requested_frame_count: int
    request_union_material_bar_tensor_bytes: int
    request_site_geometry_bar_tensor_bytes: int
    request_compact_ray_bar_tensor_bytes: int
    maximum_native_length_bar_tensor_bytes: int
    maximum_geometry_bridge_visible_tensor_bytes: int
    persistent_frame_tensor_bytes: int = 0
    persistent_sample_tensor_bytes: int = 0
    persistent_target_tensor_bytes: int = 0
    persistent_prediction_tensor_bytes: int = 0
    frame_by_word_reverse_state_tensor_bytes: int = 0
    requested_frame_count_affects_reported_bytes: bool = False
    allocator_storage_bytes_measured: bool = False
    allocator_peak_measured: bool = False
    python_object_bytes_measured: bool = False
    hot_native_block_state_tensor_bytes_included: bool = False
    whole_step_peak_measured: bool = False


@dataclass(frozen=True)
class _PaperKineticExpectedBlockManifest:
    """Scalar-only canonical sample coverage for one request/native block."""

    work_generation_digest: str
    request_generation_id: str
    executor_session_generation_id: str
    runtime_generation_id: str
    native_block_generation_digest: str
    sample_chunk_count: int
    sample_count: int
    sample_manifest_digest: str
    first_flat_sample_index: int
    last_flat_sample_index: int
    generation_digest: str
    _seal: object = field(default=None, repr=False)

    def assert_current(self) -> None:
        if (
            self._seal is not _BLOCK_MANIFEST_SEAL
            or not self.work_generation_digest.strip()
            or not self.request_generation_id.strip()
            or not self.executor_session_generation_id.strip()
            or not self.runtime_generation_id.strip()
            or not self.native_block_generation_digest.strip()
            or self.sample_chunk_count < 1
            or self.sample_count < 1
            or len(self.sample_manifest_digest) != 64
            or self.first_flat_sample_index < 0
            or self.last_flat_sample_index < self.first_flat_sample_index
            or self.generation_digest != _expected_block_manifest_digest(self)
        ):
            raise ValueError("full-geometry expected sample manifest changed")


@dataclass(frozen=True)
class PaperKineticFullGeometryBlockReceipt:
    """Scalar receipt for one consumed full VJP; retains no native tensors."""

    native_block_generation_digest: str
    request_generation_id: str
    work_generation_digest: str
    executor_session_generation_id: str
    executor_runtime_generation_id: str
    executor_execution_generation_digest: str
    reduced_sample_chunk_count: int
    reduced_sample_count: int
    global_loss_element_count: int
    loss_scale: float
    loss_normalization_id: str
    sample_manifest_digest: str
    first_flat_sample_index: int
    last_flat_sample_index: int
    native_vjp_provenance_id: str
    geometry_reduction_generation_digest: str
    row_geometry_vjp_call_count: int
    differentiable_word_reverse_interactions: int
    dense_global_site_accumulation_elements: int
    all_site_owner_validation_evaluations: int
    native_length_bar_tensor_bytes: int
    geometry_bridge_visible_tensor_bytes: int
    generation_digest: str
    native_full_vjp_invocation_count: int = 1
    device_completion_fence_call_count: int = 1
    native_result_retained: bool = False
    native_world_or_runtime_retained: bool = False
    frame_sample_target_or_prediction_retained: bool = False
    execution_consumed: bool = True
    _seal: object = field(default=None, repr=False)

    def assert_current(self) -> None:
        if (
            self._seal is not _BLOCK_RECEIPT_SEAL
            or not self.native_block_generation_digest.strip()
            or not self.request_generation_id.strip()
            or not self.work_generation_digest.strip()
            or not self.executor_session_generation_id.strip()
            or not self.executor_runtime_generation_id.strip()
            or len(self.executor_execution_generation_digest) != 64
            or self.reduced_sample_chunk_count < 1
            or self.reduced_sample_count < 1
            or self.global_loss_element_count < self.reduced_sample_count * 3
            or self.loss_scale != 1.0 / float(self.global_loss_element_count)
            or not self.loss_normalization_id.strip()
            or len(self.sample_manifest_digest) != 64
            or self.first_flat_sample_index < 0
            or self.last_flat_sample_index < self.first_flat_sample_index
            or not self.native_vjp_provenance_id.strip()
            or len(self.geometry_reduction_generation_digest) != 64
            or self.row_geometry_vjp_call_count < 1
            or self.differentiable_word_reverse_interactions < 0
            or self.dense_global_site_accumulation_elements < 0
            or self.all_site_owner_validation_evaluations < 0
            or self.native_length_bar_tensor_bytes < 1
            or self.geometry_bridge_visible_tensor_bytes < 1
            or self.native_full_vjp_invocation_count != 1
            or self.device_completion_fence_call_count != 1
            or self.native_result_retained
            or self.native_world_or_runtime_retained
            or self.frame_sample_target_or_prediction_retained
            or not self.execution_consumed
            or self.generation_digest != _block_receipt_digest(self)
        ):
            raise ValueError("full-geometry block receipt changed")


@dataclass
class PaperKineticFullGeometryRequestAssembly:
    """Request-local geometry bars plus the existing union-material ledger."""

    material: PaperKineticUnionLocalBarAssembly | None
    work_generation_digest: str
    request_generation_id: str
    request_step_generation_id: str
    active_native_block_count: int
    total_sample_count: int
    global_loss_element_count: int
    loss_scale: float
    loss_normalization_id: str
    expected_block_manifests: tuple[_PaperKineticExpectedBlockManifest, ...]
    ray_bar_keys: tuple[tuple[int, int], ...]
    grad_positions0_f64: torch.Tensor
    grad_velocities_f64: torch.Tensor
    grad_weight_coefficients_f64: torch.Tensor
    grad_track_ray_coefficients_f64: torch.Tensor
    generation_id: str
    tensor_signatures: tuple[tuple[object, ...], ...]
    executor_session_identity: int
    executor_session_generation_id: str
    executor_step_generation_id: str
    executor_generation_id: str
    _block_receipts: tuple[PaperKineticFullGeometryBlockReceipt, ...] = field(
        default_factory=tuple,
        repr=False,
    )
    request_references_released: bool = False
    poisoned: bool = False
    finalized: bool = False
    _seal: object = field(default=None, repr=False)

    @property
    def work(self) -> PaperKineticUnionLocalRequestWork:
        material = self.material
        if material is None:
            raise ValueError("full-geometry assembly released its request work")
        return material.work

    @property
    def block_receipts(self) -> tuple[PaperKineticFullGeometryBlockReceipt, ...]:
        return self._block_receipts

    def _tensors(self) -> tuple[torch.Tensor, ...]:
        material = self.material
        if material is None:
            raise ValueError("full-geometry assembly released its request tensors")
        return (
            material.grad_union_site_rgba_f32,
            material.loss_f32,
            self.grad_positions0_f64,
            self.grad_velocities_f64,
            self.grad_weight_coefficients_f64,
            self.grad_track_ray_coefficients_f64,
        )

    def assert_open(self) -> None:
        if self._seal is not _ASSEMBLY_SEAL:
            raise ValueError("full-geometry assembly was not sealed by its opener")
        if self.poisoned:
            raise ValueError("full-geometry assembly is poisoned by an earlier failure")
        if self.finalized:
            raise ValueError("full-geometry assembly was already finalized")
        if self.request_references_released or self.material is None:
            raise ValueError("full-geometry assembly released its request state")
        self.work.assert_warm_layout()
        for manifest in self.expected_block_manifests:
            manifest.assert_current()
        for receipt in self.block_receipts:
            receipt.assert_current()
        if (
            tuple(_tensor_signature(tensor) for tensor in self._tensors())
            != self.tensor_signatures
            or self.work.generation_digest != self.work_generation_digest
            or self.work.request.request_generation_id != self.request_generation_id
            or self.work.request.step_generation_id != self.request_step_generation_id
            or self.work.active_native_block_count != self.active_native_block_count
            or self.work.total_sample_count != self.total_sample_count
            or self.work.request.global_loss_element_count
            != self.global_loss_element_count
            or self.work.request.global_loss_scale != self.loss_scale
            or self.work.request.loss_normalization_id != self.loss_normalization_id
            or len(self.expected_block_manifests) != self.active_native_block_count
            or any(
                manifest.work_generation_digest != self.work_generation_digest
                or manifest.request_generation_id != self.request_generation_id
                or manifest.executor_session_generation_id
                != self.executor_session_generation_id
                for manifest in self.expected_block_manifests
            )
            or tuple(
                manifest.native_block_generation_digest
                for manifest in self.expected_block_manifests
            )
            != tuple(
                block.native_block_generation_digest
                for block in self.work.active_blocks
            )
            or len(self.block_receipts) != self.material.next_active_block_index
            or len(self.block_receipts) != self.material.consumed_native_block_count
            or len(self.block_receipts) > self.active_native_block_count
            or self.material.consumed_sample_chunk_count
            != sum(
                receipt.reduced_sample_chunk_count
                for receipt in self.block_receipts
            )
            or self.material.consumed_sample_count
            != sum(receipt.reduced_sample_count for receipt in self.block_receipts)
            or any(
                receipt.global_loss_element_count != self.global_loss_element_count
                or receipt.loss_scale != self.loss_scale
                or receipt.loss_normalization_id != self.loss_normalization_id
                for receipt in self.block_receipts
            )
            or any(
                not _block_receipt_matches_manifest(receipt, manifest)
                for receipt, manifest in zip(
                    self.block_receipts,
                    self.expected_block_manifests[: len(self.block_receipts)],
                    strict=True,
                )
            )
            or self.ray_bar_keys != _expected_ray_bar_keys(self.work)
            or self.executor_session_identity < 1
            or not self.executor_session_generation_id.strip()
            or self.executor_step_generation_id != self.request_generation_id
            or not self.executor_generation_id.strip()
            or self.generation_id != _assembly_generation_id(self)
        ):
            raise ValueError("full-geometry assembly identity/coverage changed")


@dataclass(frozen=True)
class PaperKineticFullGeometryRequestReceipt:
    """Target-free receipt after material and geometry reach global bars."""

    step_generation_id: str
    request_generation_id: str
    view_index: int
    block_id: str
    ray_bar_keys: tuple[tuple[int, int], ...]
    accounting: Mapping[str, int | float | str | bool]
    generation_digest: str
    provenance: str = FULL_GEOMETRY_PROVENANCE
    runtime_status: str = FULL_GEOMETRY_STATUS
    derivative_scope: str = GEOMETRY_DERIVATIVE_SCOPE
    structural_program_fixed: bool = True
    geometry_vjp_implemented: bool = True
    material_routed_through_existing_union_and_global_coordinators: bool = True
    node_bar_lifecycle_bound_to_sample_executor: bool = True
    cpu_fake_native_only: bool = True
    accelerator_tensor_admission_allowed: bool = False
    production_promotion_allowed: bool = False
    frame_sample_target_prediction_or_native_state_retained: bool = False
    event_time_derivatives_included: bool = False
    chart_endpoint_derivatives_included: bool = False
    node_time_or_rank_derivatives_included: bool = False
    compiler_choice_derivatives_included: bool = False
    native_runtime_verified: bool = False
    real_device_completion_fence_semantics_verified: bool = False
    allocator_peak_measured: bool = False
    _seal: object = field(default=None, repr=False)

    def memory_report(self, requested_frame_count: int) -> PaperKineticFullGeometryMemoryReport:
        _require_positive_int(requested_frame_count, name="requested_frame_count")
        return PaperKineticFullGeometryMemoryReport(
            requested_frame_count=requested_frame_count,
            request_union_material_bar_tensor_bytes=int(
                self.accounting["request_union_material_bar_tensor_bytes"]
            ),
            request_site_geometry_bar_tensor_bytes=int(
                self.accounting["request_site_geometry_bar_tensor_bytes"]
            ),
            request_compact_ray_bar_tensor_bytes=int(
                self.accounting["request_compact_ray_bar_tensor_bytes"]
            ),
            maximum_native_length_bar_tensor_bytes=int(
                self.accounting["maximum_native_length_bar_tensor_bytes"]
            ),
            maximum_geometry_bridge_visible_tensor_bytes=int(
                self.accounting["maximum_geometry_bridge_visible_tensor_bytes"]
            ),
        )

    def assert_current(self) -> None:
        if (
            self._seal is not _RECEIPT_SEAL
            or self.provenance != FULL_GEOMETRY_PROVENANCE
            or self.runtime_status != FULL_GEOMETRY_STATUS
            or self.derivative_scope != GEOMETRY_DERIVATIVE_SCOPE
            or not self.structural_program_fixed
            or not self.geometry_vjp_implemented
            or not self.material_routed_through_existing_union_and_global_coordinators
            or not self.node_bar_lifecycle_bound_to_sample_executor
            or not self.cpu_fake_native_only
            or self.accelerator_tensor_admission_allowed
            or self.production_promotion_allowed
            or self.frame_sample_target_prediction_or_native_state_retained
            or self.event_time_derivatives_included
            or self.chart_endpoint_derivatives_included
            or self.node_time_or_rank_derivatives_included
            or self.compiler_choice_derivatives_included
            or self.native_runtime_verified
            or self.real_device_completion_fence_semantics_verified
            or self.allocator_peak_measured
            or not self.step_generation_id.strip()
            or not self.request_generation_id.strip()
            or self.view_index < 0
            or not self.block_id.strip()
            or tuple(sorted(set(self.ray_bar_keys))) != self.ray_bar_keys
            or int(self.accounting["native_full_vjp_invocation_count"])
            != int(self.accounting["active_native_block_count"])
            or int(self.accounting["device_completion_fence_call_count"])
            != int(self.accounting["active_native_block_count"])
            or int(self.accounting["reduced_sample_count"])
            != int(self.accounting["request_total_sample_count"])
            or int(self.accounting["reduced_sample_chunk_count"])
            != int(self.accounting["expected_sample_chunk_count"])
            or not str(self.accounting["request_work_generation_digest"]).strip()
            or not str(self.accounting["executor_session_generation_id"]).strip()
            or len(str(self.accounting["executor_telemetry_generation_digest"])) != 64
            or int(self.accounting["global_loss_element_count"])
            < int(self.accounting["reduced_sample_count"]) * 3
            or float(self.accounting["loss_scale"])
            != 1.0 / float(int(self.accounting["global_loss_element_count"]))
            or not str(self.accounting["loss_normalization_id"]).strip()
            or len(str(self.accounting["ordered_block_receipt_digest"])) != 64
            or len(str(self.accounting["ordered_sample_manifest_digest"])) != 64
            or not bool(
                self.accounting["request_references_released_before_receipt_return"]
            )
            or bool(self.accounting["hot_native_block_state_tensor_bytes_included"])
            or bool(self.accounting["whole_step_peak_measured"])
            or not bool(
                self.accounting["cpu_only_entry_and_finalization_enforced"]
            )
            or bool(self.accounting["accelerator_tensor_admission_allowed"])
            or self.generation_digest != _receipt_digest(self)
        ):
            raise ValueError("full-geometry request receipt changed")


@torch.no_grad()
def begin_paper_kinetic_full_geometry_request(
    work: PaperKineticUnionLocalRequestWork,
    session: KineticNativeMaterialStepSession,
    *,
    grad_union_site_rgba_f32: torch.Tensor,
    loss_f32: torch.Tensor,
    grad_positions0_f64: torch.Tensor,
    grad_velocities_f64: torch.Tensor,
    grad_weight_coefficients_f64: torch.Tensor,
    ray_bar_keys: Sequence[tuple[int, int]],
    grad_track_ray_coefficients_f64: torch.Tensor,
) -> PaperKineticFullGeometryRequestAssembly:
    """Bind one empty executor session and open zeroed request-local bars."""

    if not isinstance(work, PaperKineticUnionLocalRequestWork):
        raise TypeError("work must be PaperKineticUnionLocalRequestWork")
    _require_cpu_fake_native_work(work)
    for tensor in (
        grad_union_site_rgba_f32,
        loss_f32,
        grad_positions0_f64,
        grad_velocities_f64,
        grad_weight_coefficients_f64,
        grad_track_ray_coefficients_f64,
    ):
        _require_cpu_only_tensor(
            tensor,
            name="full_geometry_begin_tensor",
        )
    work.assert_warm_layout()
    _require_executor_session_for_work(session, work, require_empty=True)
    expected_manifests = _prepare_expected_block_manifests(work, session)
    sites = _shared_sites(work)
    keys = tuple((int(view), int(track)) for view, track in ray_bar_keys)
    if keys != _expected_ray_bar_keys(work):
        raise ValueError("request ray bars must use exact canonical (view,track) keys")
    _require_cpu_f64(
        grad_positions0_f64,
        name="grad_positions0_f64",
        shape=(sites.site_count, 3),
        require_finite=False,
    )
    _require_cpu_f64(
        grad_velocities_f64,
        name="grad_velocities_f64",
        shape=(sites.site_count, 3),
        require_finite=False,
    )
    _require_cpu_f64(
        grad_weight_coefficients_f64,
        name="grad_weight_coefficients_f64",
        shape=tuple(sites.weight_coefficients.shape),
        require_finite=False,
    )
    _require_cpu_f64(
        grad_track_ray_coefficients_f64,
        name="grad_track_ray_coefficients_f64",
        shape=(len(keys), 12),
        require_finite=False,
    )
    _require_tensor(
        grad_union_site_rgba_f32,
        name="grad_union_site_rgba_f32",
        dtype=torch.float32,
        device=work.bundle.device,
        shape=(work.bundle.union_site_count, 4),
    )
    _require_tensor(
        loss_f32,
        name="union_local_loss_f32",
        dtype=torch.float32,
        device=work.bundle.device,
        shape=(1,),
    )
    geometry_tensors = (
        grad_positions0_f64,
        grad_velocities_f64,
        grad_weight_coefficients_f64,
        grad_track_ray_coefficients_f64,
    )
    _require_distinct_storage(
        grad_union_site_rgba_f32,
        loss_f32,
        *geometry_tensors,
    )
    material = begin_paper_kinetic_union_local_bar_assembly(
        work,
        grad_union_site_rgba_f32=grad_union_site_rgba_f32,
        loss_f32=loss_f32,
    )
    for tensor in geometry_tensors:
        tensor.zero_()
    provisional = PaperKineticFullGeometryRequestAssembly(
        material=material,
        work_generation_digest=work.generation_digest,
        request_generation_id=work.request.request_generation_id,
        request_step_generation_id=work.request.step_generation_id,
        active_native_block_count=work.active_native_block_count,
        total_sample_count=work.total_sample_count,
        global_loss_element_count=work.request.global_loss_element_count,
        loss_scale=work.request.global_loss_scale,
        loss_normalization_id=work.request.loss_normalization_id,
        expected_block_manifests=expected_manifests,
        ray_bar_keys=keys,
        grad_positions0_f64=grad_positions0_f64,
        grad_velocities_f64=grad_velocities_f64,
        grad_weight_coefficients_f64=grad_weight_coefficients_f64,
        grad_track_ray_coefficients_f64=grad_track_ray_coefficients_f64,
        generation_id="",
        tensor_signatures=(),
        executor_session_identity=id(session),
        executor_session_generation_id=session.generation_id,
        executor_step_generation_id=session.step_generation_id,
        executor_generation_id=session.executor.generation_id,
        _seal=_ASSEMBLY_SEAL,
    )
    provisional.tensor_signatures = tuple(
        _tensor_signature(tensor) for tensor in provisional._tensors()
    )
    provisional.generation_id = _assembly_generation_id(provisional)
    provisional.assert_open()
    return provisional


@torch.no_grad()
def consume_paper_kinetic_full_geometry_native_block(
    assembly: PaperKineticFullGeometryRequestAssembly,
    session: KineticNativeMaterialStepSession,
    execution: KineticNativeFullGeometryVJPExecution,
    *,
    loss_f32: torch.Tensor,
    device_completion_fence: Callable[[], None],
    device_completion_fence_provenance: str,
    maximum_geometry_bridge_visible_peak_logical_tensor_bytes: int,
) -> PaperKineticFullGeometryBlockReceipt:
    """Consume exactly the next expected block and retain only scalar receipt."""

    if not isinstance(assembly, PaperKineticFullGeometryRequestAssembly):
        raise TypeError("assembly must be PaperKineticFullGeometryRequestAssembly")
    assembly.assert_open()
    if not isinstance(session, KineticNativeMaterialStepSession):
        raise TypeError("session must be KineticNativeMaterialStepSession")
    if not isinstance(execution, KineticNativeFullGeometryVJPExecution):
        raise TypeError("execution must be KineticNativeFullGeometryVJPExecution")
    _require_cpu_only_tensor(loss_f32, name="full_geometry_block_loss_f32")
    _bind_executor_session(assembly, session)
    execution.assert_current(session)
    native_vjp = execution.native_vjp_result
    if not isinstance(native_vjp, KineticNativeEqualRankVJPResult):
        raise TypeError("executor full-geometry receipt has an invalid native VJP")
    native_vjp.assert_warm_layout()
    world = native_vjp.world
    _require_positive_int(
        maximum_geometry_bridge_visible_peak_logical_tensor_bytes,
        name="maximum_geometry_bridge_visible_peak_logical_tensor_bytes",
    )
    material = assembly.material
    if material is None:
        raise ValueError("full-geometry assembly released its material state")
    expected_index = material.next_active_block_index
    expected = assembly.work.active_blocks[expected_index]
    expected_manifest = assembly.expected_block_manifests[expected_index]
    digest = world.runtime.payload.block.generation_digest
    if digest != expected.native_block_generation_digest:
        raise ValueError("full-geometry native block is duplicate, out of order, or foreign")
    if (
        execution.reduced_sample_chunk_count != expected.sample_chunk_count
        or execution.reduced_sample_count != expected.sample_count
        or not _execution_matches_manifest(execution, expected_manifest)
    ):
        raise ValueError(
            "executor receipt does not match exact ordered request sample coverage"
        )
    _require_tensor(
        loss_f32,
        name="native_block_loss_f32",
        dtype=torch.float32,
        device=world.runtime.device,
        shape=(1,),
    )
    execution.assert_current(session, loss_f32=loss_f32)
    try:
        native_provenance = kinetic_native_equal_rank_vjp_provenance_id(native_vjp)
        geometry = reduce_kinetic_native_equal_rank_geometry_vjp(
            native_vjp,
            assembly.work.sampler,
            expected_native_vjp_provenance_id=native_provenance,
            device_completion_fence=device_completion_fence,
            device_completion_fence_provenance=device_completion_fence_provenance,
            maximum_bridge_visible_peak_logical_tensor_bytes=(
                maximum_geometry_bridge_visible_peak_logical_tensor_bytes
            ),
        )
        if (
            geometry.native_block_generation_digest != digest
            or geometry.ray_bar_keys
            != tuple(key for key in assembly.ray_bar_keys if key in set(geometry.ray_bar_keys))
        ):
            _fail_value("geometry reduction block/ray provenance changed")
        assembly.grad_positions0_f64.add_(geometry.grad_positions0_f64)
        assembly.grad_velocities_f64.add_(geometry.grad_velocities_f64)
        assembly.grad_weight_coefficients_f64.add_(
            geometry.grad_weight_coefficients_f64
        )
        ray_position = {key: index for index, key in enumerate(assembly.ray_bar_keys)}
        for compact_index, key in enumerate(geometry.ray_bar_keys):
            assembly.grad_track_ray_coefficients_f64[ray_position[key]].add_(
                geometry.grad_track_ray_coefficients_f64[compact_index]
            )
        contribution = seal_paper_kinetic_native_block_vjp_contribution(
            material,
            native_vjp_result=native_vjp,
            loss_f32=loss_f32,
            reduced_sample_chunk_count=execution.reduced_sample_chunk_count,
            reduced_sample_count=execution.reduced_sample_count,
        )
        consume_paper_kinetic_union_local_native_contribution(
            material,
            contribution,
        )
        completion = session.consume_full_geometry_vjp_execution(
            execution,
            geometry_reduction=geometry,
            expected_device_completion_fence_provenance=(
                device_completion_fence_provenance
            ),
        )
        completion.assert_current()
        if (
            completion.execution_generation_digest != execution.generation_digest
            or completion.geometry_reduction_generation_digest
            != geometry.generation_digest
            or completion.global_accumulation_proven
            or completion.completion_semantics
            != "fenced_and_reduced_not_globally_committed"
        ):
            _fail_value("executor fenced-reduction completion changed")
        provisional = PaperKineticFullGeometryBlockReceipt(
            native_block_generation_digest=digest,
            request_generation_id=assembly.request_generation_id,
            work_generation_digest=assembly.work_generation_digest,
            executor_session_generation_id=session.generation_id,
            executor_runtime_generation_id=execution.runtime_generation_id,
            executor_execution_generation_digest=execution.generation_digest,
            reduced_sample_chunk_count=execution.reduced_sample_chunk_count,
            reduced_sample_count=execution.reduced_sample_count,
            global_loss_element_count=assembly.global_loss_element_count,
            loss_scale=assembly.loss_scale,
            loss_normalization_id=assembly.loss_normalization_id,
            sample_manifest_digest=execution.sample_manifest_digest,
            first_flat_sample_index=execution.first_flat_sample_index,
            last_flat_sample_index=execution.last_flat_sample_index,
            native_vjp_provenance_id=native_provenance,
            geometry_reduction_generation_digest=geometry.generation_digest,
            row_geometry_vjp_call_count=geometry.row_geometry_vjp_call_count,
            differentiable_word_reverse_interactions=(
                geometry.differentiable_word_reverse_interactions
            ),
            dense_global_site_accumulation_elements=(
                geometry.dense_global_site_accumulation_elements
            ),
            all_site_owner_validation_evaluations=(
                geometry.all_site_owner_validation_evaluations
            ),
            native_length_bar_tensor_bytes=_tensor_bytes(
                native_vjp.grad_node_physical_length_f32
            ),
            geometry_bridge_visible_tensor_bytes=(
                geometry.memory.bridge_visible_peak_logical_tensor_bytes
            ),
            generation_digest="",
            _seal=_BLOCK_RECEIPT_SEAL,
        )
        receipt = replace(
            provisional,
            generation_digest=_block_receipt_digest(provisional),
        )
        receipt.assert_current()
        assembly._block_receipts += (receipt,)
        assembly.tensor_signatures = tuple(
            _tensor_signature(tensor) for tensor in assembly._tensors()
        )
        assembly.generation_id = _assembly_generation_id(assembly)
        assembly.assert_open()
    except BaseException:
        try:
            session.abort(
                device_completion_fence=device_completion_fence,
                device_completion_fence_provenance=(
                    device_completion_fence_provenance
                ),
            )
        finally:
            _poison_and_clear(assembly)
        raise
    else:
        return receipt


@torch.no_grad()
def finalize_and_consume_paper_kinetic_full_geometry_request(
    assembly: PaperKineticFullGeometryRequestAssembly,
    ledger: PaperRaggedMaterialBarStepLedger,
    request: PaperRaggedMaterialBarRequest,
    *,
    executor_telemetry: KineticNativeMaterialStepTelemetry,
    global_grad_positions0_f64: torch.Tensor,
    global_grad_velocities_f64: torch.Tensor,
    global_grad_weight_coefficients_f64: torch.Tensor,
    global_ray_bar_keys: Sequence[tuple[int, int]],
    global_grad_track_ray_coefficients_f64: torch.Tensor,
) -> PaperKineticFullGeometryRequestReceipt:
    """Preflight completely, then commit or fail-stop every step-global bar."""

    assembly.assert_open()
    _require_cpu_fake_native_work(assembly.work)
    if not isinstance(ledger, PaperRaggedMaterialBarStepLedger):
        raise TypeError("ledger must be PaperRaggedMaterialBarStepLedger")
    for tensor in assembly._tensors() + (
        ledger.global_grad_site_rgba_f32,
        ledger.loss_f32,
        global_grad_positions0_f64,
        global_grad_velocities_f64,
        global_grad_weight_coefficients_f64,
        global_grad_track_ray_coefficients_f64,
    ):
        _require_cpu_only_tensor(
            tensor,
            name="full_geometry_finalize_tensor",
        )
    if assembly.work.request is not request:
        raise ValueError("full-geometry assembly belongs to a different request")
    _prevalidate_complete_assembly(assembly)
    _prevalidate_executor_telemetry(assembly, executor_telemetry)
    _assert_material_ledger_current(ledger)
    if getattr(ledger, "_full_geometry_poisoned", False):
        raise ValueError("ragged material step is poisoned by a failed geometry commit")
    if ledger.finalized:
        raise ValueError("ragged material step was already finalized")
    if ledger.active_request is not request:
        raise ValueError("full-geometry request is not the active material request")
    keys = tuple((int(view), int(track)) for view, track in global_ray_bar_keys)
    if tuple(sorted(set(keys))) != keys or not set(assembly.ray_bar_keys).issubset(keys):
        raise ValueError("global ray keys must canonically contain the request ray keys")
    _require_cpu_f64(
        global_grad_positions0_f64,
        name="global_grad_positions0_f64",
        shape=tuple(assembly.grad_positions0_f64.shape),
    )
    _require_cpu_f64(
        global_grad_velocities_f64,
        name="global_grad_velocities_f64",
        shape=tuple(assembly.grad_velocities_f64.shape),
    )
    _require_cpu_f64(
        global_grad_weight_coefficients_f64,
        name="global_grad_weight_coefficients_f64",
        shape=tuple(assembly.grad_weight_coefficients_f64.shape),
    )
    _require_cpu_f64(
        global_grad_track_ray_coefficients_f64,
        name="global_grad_track_ray_coefficients_f64",
        shape=(len(keys), 12),
    )
    local_tensors = assembly._tensors()
    global_geometry_tensors = (
        global_grad_positions0_f64,
        global_grad_velocities_f64,
        global_grad_weight_coefficients_f64,
        global_grad_track_ray_coefficients_f64,
    )
    _require_distinct_storage(
        *local_tensors,
        ledger.global_grad_site_rgba_f32,
        ledger.loss_f32,
        *global_geometry_tensors,
    )
    for tensor in local_tensors + (
        ledger.global_grad_site_rgba_f32,
        ledger.loss_f32,
    ) + global_geometry_tensors:
        _require_finite(tensor, name="full_geometry_commit_tensor")
    global_ray_position = {key: index for index, key in enumerate(keys)}
    accounting = MappingProxyType(
        _request_accounting(assembly, executor_telemetry)
    )
    provisional = PaperKineticFullGeometryRequestReceipt(
        step_generation_id=request.step_generation_id,
        request_generation_id=request.request_generation_id,
        view_index=request.view_index,
        block_id=request.block.block_id,
        ray_bar_keys=assembly.ray_bar_keys,
        accounting=accounting,
        generation_digest="",
        _seal=_RECEIPT_SEAL,
    )
    receipt = replace(
        provisional,
        generation_digest=_receipt_digest(provisional),
    )
    receipt.assert_current()

    global_commit_started = False
    try:
        material = assembly.material
        if material is None:
            raise ValueError("full-geometry assembly released its material state")
        compact = finalize_paper_kinetic_union_local_bar_assembly(material)
        compact.assert_current()
        global_commit_started = True
        global_grad_positions0_f64.add_(assembly.grad_positions0_f64)
        global_grad_velocities_f64.add_(assembly.grad_velocities_f64)
        global_grad_weight_coefficients_f64.add_(
            assembly.grad_weight_coefficients_f64
        )
        for compact_index, key in enumerate(assembly.ray_bar_keys):
            global_grad_track_ray_coefficients_f64[
                global_ray_position[key]
            ].add_(assembly.grad_track_ray_coefficients_f64[compact_index])
        consume_paper_ragged_compact_material_bar_result(ledger, request, compact)
        for tensor in (
            ledger.global_grad_site_rgba_f32,
            ledger.loss_f32,
        ) + global_geometry_tensors:
            _require_finite(tensor, name="committed_full_geometry_tensor")
        assembly.finalized = True
        _release_request_references(assembly)
    except BaseException:
        if global_commit_started:
            _fail_stop_global_commit(ledger, global_geometry_tensors)
        _poison_and_clear(assembly)
        raise
    else:
        return receipt


def _request_accounting(
    assembly: PaperKineticFullGeometryRequestAssembly,
    executor_telemetry: KineticNativeMaterialStepTelemetry,
) -> dict[str, int | float | str | bool]:
    receipts = assembly.block_receipts
    for receipt in receipts:
        receipt.assert_current()
    material = assembly.material
    if material is None:
        raise ValueError("full-geometry accounting requires live request material")
    return {
        "request_work_generation_digest": assembly.work_generation_digest,
        "executor_session_generation_id": assembly.executor_session_generation_id,
        "executor_telemetry_generation_digest": executor_telemetry.generation_digest,
        "global_loss_element_count": assembly.global_loss_element_count,
        "loss_scale": assembly.loss_scale,
        "loss_normalization_id": assembly.loss_normalization_id,
        "ordered_block_receipt_digest": _digest_parts(
            FULL_GEOMETRY_PROVENANCE,
            "ordered-block-receipts",
            assembly.request_generation_id,
            assembly.work_generation_digest,
            tuple(receipt.generation_digest for receipt in receipts),
        ),
        "ordered_sample_manifest_digest": _digest_parts(
            FULL_GEOMETRY_PROVENANCE,
            "ordered-sample-manifests",
            assembly.request_generation_id,
            assembly.executor_session_generation_id,
            tuple(receipt.sample_manifest_digest for receipt in receipts),
        ),
        "active_native_block_count": assembly.active_native_block_count,
        "request_total_sample_count": assembly.total_sample_count,
        "expected_sample_chunk_count": sum(
            manifest.sample_chunk_count
            for manifest in assembly.expected_block_manifests
        ),
        "native_full_vjp_invocation_count": sum(
            item.native_full_vjp_invocation_count for item in receipts
        ),
        "device_completion_fence_call_count": sum(
            item.device_completion_fence_call_count for item in receipts
        ),
        "row_geometry_vjp_call_count": sum(
            item.row_geometry_vjp_call_count for item in receipts
        ),
        "reduced_sample_chunk_count": sum(
            item.reduced_sample_chunk_count for item in receipts
        ),
        "reduced_sample_count": sum(item.reduced_sample_count for item in receipts),
        "differentiable_word_reverse_interactions": sum(
            item.differentiable_word_reverse_interactions for item in receipts
        ),
        "dense_global_site_accumulation_elements": sum(
            item.dense_global_site_accumulation_elements for item in receipts
        ),
        "all_site_owner_validation_evaluations": sum(
            item.all_site_owner_validation_evaluations for item in receipts
        ),
        "request_union_material_bar_tensor_bytes": _tensor_bytes(
            material.grad_union_site_rgba_f32
        ),
        "request_site_geometry_bar_tensor_bytes": _tensor_bytes(
            assembly.grad_positions0_f64,
            assembly.grad_velocities_f64,
            assembly.grad_weight_coefficients_f64,
        ),
        "request_compact_ray_bar_tensor_bytes": _tensor_bytes(
            assembly.grad_track_ray_coefficients_f64
        ),
        "maximum_native_length_bar_tensor_bytes": max(
            item.native_length_bar_tensor_bytes for item in receipts
        ),
        "maximum_geometry_bridge_visible_tensor_bytes": max(
            item.geometry_bridge_visible_tensor_bytes for item in receipts
        ),
        "ray_bars_keyed_by_view_and_track": True,
        "material_global_accounting_consumed": True,
        "structural_program_fixed": True,
        "node_bar_lifecycle_bound_to_sample_executor": True,
        "production_promotion_allowed": False,
        "request_references_released_before_receipt_return": True,
        "hot_native_block_state_tensor_bytes_included": False,
        "whole_step_peak_measured": False,
        "persistent_frame_tensor_bytes": 0,
        "persistent_sample_tensor_bytes": 0,
        "persistent_target_tensor_bytes": 0,
        "persistent_prediction_tensor_bytes": 0,
        "frame_by_word_reverse_state_tensor_bytes": 0,
        "word_reverse_scaling": "O(sum_active_blocks J_b * W_b)",
        "geometry_validation_scaling": "O(sum_rows J * S * R_row)",
        "dense_geometry_accumulation_scaling": "O(sum_rows S * (6 + Q))",
        "native_runtime_verified": False,
        "real_device_completion_fence_semantics_verified": False,
        "cpu_only_entry_and_finalization_enforced": True,
        "accelerator_tensor_admission_allowed": False,
        "allocator_peak_measured": False,
    }


def _shared_sites(work: PaperKineticUnionLocalRequestWork):
    """Return the common global site namespace represented by this request.

    Track compilation is allowed to freeze an equal-content site table into a
    distinct Python object.  Object identity is therefore not the world
    identity: the lowering's sealed ``site_namespace_digest`` is.  The warm
    sampler check immediately before this helper already binds every row to
    that lowering and rejects tensor identity/layout/version drift.  Requiring
    one Python object here made the ordinary multi-track compiler output
    impossible to execute even though all rows had the same sealed global site
    namespace.
    """

    if not work.sampler.rows:
        raise ValueError("one full-geometry request requires at least one kinetic row")
    first = work.sampler.rows[0].program.binding.sites
    expected_site_count = int(work.sampler.lowering.global_site_count)
    expected_weight_shape = tuple(first.weight_coefficients.shape)
    if (
        first.site_count != expected_site_count
        or any(
            row.program.binding.sites.site_count != expected_site_count
            or tuple(row.program.binding.sites.positions0.shape)
            != tuple(first.positions0.shape)
            or tuple(row.program.binding.sites.velocities.shape)
            != tuple(first.velocities.shape)
            or tuple(row.program.binding.sites.weight_coefficients.shape)
            != expected_weight_shape
            for row in work.sampler.rows
        )
    ):
        raise ValueError(
            "one full-geometry request must share one sealed kinetic site namespace"
        )
    return first


def _expected_ray_bar_keys(
    work: PaperKineticUnionLocalRequestWork,
) -> tuple[tuple[int, int], ...]:
    active = {
        block.native_block_generation_digest for block in work.active_blocks
    }
    return tuple(
        sorted(
            {
                (work.bundle.view_index, row.track_id)
                for row in work.sampler.rows
                if row.native_block_generation_digest in active
            }
        )
    )


def _bind_executor_session(
    assembly: PaperKineticFullGeometryRequestAssembly,
    session: KineticNativeMaterialStepSession,
) -> None:
    _require_executor_session_for_work(session, assembly.work, require_empty=False)
    if (
        assembly.executor_session_identity != id(session)
        or assembly.executor_session_generation_id != session.generation_id
        or assembly.executor_step_generation_id != session.step_generation_id
        or assembly.executor_generation_id != session.executor.generation_id
    ):
        raise ValueError(
            "one full-geometry request cannot mix executor sessions"
        )


def _require_executor_session_for_work(
    session: KineticNativeMaterialStepSession,
    work: PaperKineticUnionLocalRequestWork,
    *,
    require_empty: bool,
) -> None:
    if not isinstance(session, KineticNativeMaterialStepSession):
        raise TypeError("session must be KineticNativeMaterialStepSession")
    _require_cpu_fake_native_work(work, session=session)
    session._assert_open()
    session.executor.assert_current()
    if (
        session.step_generation_id != work.request.request_generation_id
        or session.requested_observation_count != work.total_sample_count
        or session._sealed
        or session._failed
        or (require_empty and bool(session._states))
        or (
            require_empty
            and (
                session._native_node_forward_launch_count != 0
                or session._native_sample_prepare_count != 0
                or session._native_sample_launch_count != 0
                or session._streamed_sample_count != 0
                or session._native_material_vjp_launch_count != 0
                or session._native_full_geometry_vjp_launch_count != 0
                or session._native_length_bar_tensor_bytes != 0
                or session._reverse_mode is not None
                or bool(session._node_bar_owner_by_storage)
                or bool(session._loss_owner_by_storage)
            )
        )
        or (
            require_empty
            and any(
                value is not None
                for value in (
                    session._global_loss_element_count,
                    session._loss_scale,
                    session._loss_normalization_id,
                )
            )
        )
        or (
            not require_empty
            and (
                session._global_loss_element_count
                != work.request.global_loss_element_count
                or session._loss_scale != work.request.global_loss_scale
                or session._loss_normalization_id
                != work.request.loss_normalization_id
            )
        )
    ):
        raise ValueError(
            "full-geometry executor session belongs to a different or active request"
        )


def _require_cpu_fake_native_work(
    work: PaperKineticUnionLocalRequestWork,
    *,
    session: KineticNativeMaterialStepSession | None = None,
) -> None:
    if work.bundle.device.type != "cpu" or (
        session is not None and session.executor.device.type != "cpu"
    ):
        raise ValueError(
            "standalone full-geometry seam is CPU/fake-native-only"
        )


def _prepare_expected_block_manifests(
    work: PaperKineticUnionLocalRequestWork,
    session: KineticNativeMaterialStepSession,
) -> tuple[_PaperKineticExpectedBlockManifest, ...]:
    """Independently replay canonical dispatch into O(active-blocks) digests."""

    bindings = {}
    for active in work.active_blocks:
        matches = tuple(
            binding
            for binding in session.executor.bindings
            if binding.native_block_generation_digest
            == active.native_block_generation_digest
        )
        if len(matches) != 1:
            raise ValueError(
                "full-geometry request has no unique executor runtime binding"
            )
        binding = matches[0]
        bindings[active.native_block_generation_digest] = binding

    states: dict[str, dict[str, int | str | None]] = {}
    for active in work.active_blocks:
        binding = bindings[active.native_block_generation_digest]
        states[active.native_block_generation_digest] = {
            "manifest": _digest_parts(
                EXECUTOR_PROVENANCE,
                "block-sample-manifest",
                session.generation_id,
                binding.runtime_generation_id,
                binding.sampler_generation_digest,
                binding.native_block_generation_digest,
            ),
            "chunk_count": 0,
            "sample_count": 0,
            "first": None,
            "last": None,
        }

    for sample_block in iter_paper_kinetic_row_ragged_request_blocks(
        work.sampler,
        work.request,
        maximum_samples_per_launch=work.maximum_samples_per_launch,
    ):
        state = states.get(sample_block.native_block_generation_digest)
        if state is None:
            raise ValueError("canonical sample dispatch selected an inactive native block")
        flat_indices = tuple(
            int(value) for value in sample_block.flat_sample_index_i64.tolist()
        )
        previous_last = state["last"]
        if (
            len(flat_indices) != sample_block.sample_count
            or any(
                right <= left
                for left, right in zip(flat_indices, flat_indices[1:], strict=False)
            )
            or (
                isinstance(previous_last, int)
                and flat_indices[0] <= previous_last
            )
        ):
            raise ValueError("canonical request sample manifest is not strictly ordered")
        state["manifest"] = _digest_parts(
            EXECUTOR_PROVENANCE,
            "block-sample-launch",
            state["manifest"],
            sample_block.generation_digest,
            sample_block.dispatch_generation_digest,
            flat_indices,
        )
        state["chunk_count"] = int(state["chunk_count"]) + 1
        state["sample_count"] = int(state["sample_count"]) + sample_block.sample_count
        if state["first"] is None:
            state["first"] = flat_indices[0]
        state["last"] = flat_indices[-1]

    manifests = []
    for active in work.active_blocks:
        binding = bindings[active.native_block_generation_digest]
        state = states[active.native_block_generation_digest]
        first = state["first"]
        last = state["last"]
        if (
            int(state["chunk_count"]) != active.sample_chunk_count
            or int(state["sample_count"]) != active.sample_count
            or not isinstance(first, int)
            or not isinstance(last, int)
        ):
            raise ValueError("canonical request sample manifest coverage changed")
        provisional = _PaperKineticExpectedBlockManifest(
            work_generation_digest=work.generation_digest,
            request_generation_id=work.request.request_generation_id,
            executor_session_generation_id=session.generation_id,
            runtime_generation_id=binding.runtime_generation_id,
            native_block_generation_digest=active.native_block_generation_digest,
            sample_chunk_count=active.sample_chunk_count,
            sample_count=active.sample_count,
            sample_manifest_digest=str(state["manifest"]),
            first_flat_sample_index=first,
            last_flat_sample_index=last,
            generation_digest="",
            _seal=_BLOCK_MANIFEST_SEAL,
        )
        manifest = replace(
            provisional,
            generation_digest=_expected_block_manifest_digest(provisional),
        )
        manifest.assert_current()
        manifests.append(manifest)
    return tuple(manifests)


def _execution_matches_manifest(
    execution: KineticNativeFullGeometryVJPExecution,
    manifest: _PaperKineticExpectedBlockManifest,
) -> bool:
    manifest.assert_current()
    return (
        execution.session_generation_id
        == manifest.executor_session_generation_id
        and execution.runtime_generation_id == manifest.runtime_generation_id
        and execution.native_block_generation_digest
        == manifest.native_block_generation_digest
        and execution.reduced_sample_chunk_count == manifest.sample_chunk_count
        and execution.reduced_sample_count == manifest.sample_count
        and execution.sample_manifest_digest == manifest.sample_manifest_digest
        and execution.first_flat_sample_index == manifest.first_flat_sample_index
        and execution.last_flat_sample_index == manifest.last_flat_sample_index
    )


def _block_receipt_matches_manifest(
    receipt: PaperKineticFullGeometryBlockReceipt,
    manifest: _PaperKineticExpectedBlockManifest,
) -> bool:
    receipt.assert_current()
    manifest.assert_current()
    return (
        receipt.request_generation_id == manifest.request_generation_id
        and receipt.work_generation_digest == manifest.work_generation_digest
        and receipt.executor_session_generation_id
        == manifest.executor_session_generation_id
        and receipt.executor_runtime_generation_id == manifest.runtime_generation_id
        and receipt.native_block_generation_digest
        == manifest.native_block_generation_digest
        and receipt.reduced_sample_chunk_count == manifest.sample_chunk_count
        and receipt.reduced_sample_count == manifest.sample_count
        and receipt.sample_manifest_digest == manifest.sample_manifest_digest
        and receipt.first_flat_sample_index == manifest.first_flat_sample_index
        and receipt.last_flat_sample_index == manifest.last_flat_sample_index
    )


def _prevalidate_complete_assembly(
    assembly: PaperKineticFullGeometryRequestAssembly,
) -> None:
    material = assembly.material
    if material is None:
        raise ValueError("full-geometry assembly released its material state")
    receipts = assembly.block_receipts
    if (
        len(receipts) != assembly.active_native_block_count
        or material.next_active_block_index != assembly.active_native_block_count
        or material.consumed_native_block_count != assembly.active_native_block_count
        or material.consumed_sample_chunk_count
        != sum(
            manifest.sample_chunk_count
            for manifest in assembly.expected_block_manifests
        )
        or material.consumed_sample_count != assembly.total_sample_count
        or any(
            receipt.global_loss_element_count != assembly.global_loss_element_count
            or receipt.loss_scale != assembly.loss_scale
            or receipt.loss_normalization_id != assembly.loss_normalization_id
            for receipt in receipts
        )
        or any(
            not _block_receipt_matches_manifest(receipt, manifest)
            for receipt, manifest in zip(
                receipts,
                assembly.expected_block_manifests,
                strict=True,
            )
        )
    ):
        raise ValueError("full-geometry request cannot finalize with incomplete provenance")


def _prevalidate_executor_telemetry(
    assembly: PaperKineticFullGeometryRequestAssembly,
    telemetry: KineticNativeMaterialStepTelemetry,
) -> None:
    if not isinstance(telemetry, KineticNativeMaterialStepTelemetry):
        raise TypeError("executor_telemetry must be KineticNativeMaterialStepTelemetry")
    telemetry.assert_current()
    receipts = assembly.block_receipts
    if (
        telemetry.executor_generation_id != assembly.executor_generation_id
        or telemetry.step_generation_id != assembly.executor_step_generation_id
        or telemetry.session_generation_id
        != assembly.executor_session_generation_id
        or telemetry.requested_observation_count != assembly.total_sample_count
        or telemetry.active_native_block_count != assembly.active_native_block_count
        or telemetry.native_full_geometry_vjp_launch_count
        != assembly.active_native_block_count
        or telemetry.reverse_mode != "full_geometry"
        or telemetry.global_loss_element_count != assembly.global_loss_element_count
        or telemetry.loss_scale != assembly.loss_scale
        or telemetry.loss_normalization_id != assembly.loss_normalization_id
        or len(telemetry.blocks) != len(receipts)
        or any(
            block.runtime_generation_id != receipt.executor_runtime_generation_id
            or block.sampler_generation_digest
            != assembly.work.sampler.generation_digest
            or block.native_block_generation_digest
            != receipt.native_block_generation_digest
            or block.native_sample_launch_count
            != receipt.reduced_sample_chunk_count
            or block.streamed_sample_count != receipt.reduced_sample_count
            or block.sample_manifest_digest != receipt.sample_manifest_digest
            or block.first_flat_sample_index != receipt.first_flat_sample_index
            or block.last_flat_sample_index != receipt.last_flat_sample_index
            for block, receipt in zip(telemetry.blocks, receipts, strict=True)
        )
    ):
        raise ValueError(
            "executor telemetry does not match full-geometry request provenance"
        )


@torch.no_grad()
def _fail_stop_global_commit(
    ledger: PaperRaggedMaterialBarStepLedger,
    global_geometry_tensors: tuple[torch.Tensor, ...],
) -> None:
    """Invalidate the entire caller-owned step after any partial global write."""

    for tensor in (
        ledger.global_grad_site_rgba_f32,
        ledger.loss_f32,
    ) + global_geometry_tensors:
        try:
            tensor.zero_()
        except BaseException:
            pass
    ledger.active_request = None
    ledger.finalized = True
    ledger.authorization_issued = False
    setattr(ledger, "_full_geometry_poisoned", True)


def _release_request_references(
    assembly: PaperKineticFullGeometryRequestAssembly,
) -> None:
    assembly.material = None
    assembly.tensor_signatures = ()
    assembly.request_references_released = True


@torch.no_grad()
def _poison_and_clear(assembly: PaperKineticFullGeometryRequestAssembly) -> None:
    if assembly.material is not None:
        for tensor in assembly._tensors():
            tensor.zero_()
    assembly.poisoned = True
    _release_request_references(assembly)


def _assembly_generation_id(
    assembly: PaperKineticFullGeometryRequestAssembly,
) -> str:
    material = assembly.material
    if material is None:
        raise ValueError("released full-geometry assembly has no live generation")
    return _digest_parts(
        FULL_GEOMETRY_PROVENANCE,
        assembly.work_generation_digest,
        assembly.request_generation_id,
        assembly.request_step_generation_id,
        assembly.active_native_block_count,
        assembly.total_sample_count,
        assembly.global_loss_element_count,
        assembly.loss_scale,
        assembly.loss_normalization_id,
        assembly.expected_block_manifests,
        material.generation_id,
        assembly.ray_bar_keys,
        tuple(id(tensor) for tensor in assembly._tensors()),
        assembly.executor_session_identity,
        assembly.executor_session_generation_id,
        assembly.executor_step_generation_id,
        assembly.executor_generation_id,
        assembly.block_receipts,
    )


def _expected_block_manifest_digest(
    manifest: _PaperKineticExpectedBlockManifest,
) -> str:
    return _digest_parts(
        FULL_GEOMETRY_PROVENANCE,
        "expected-block-sample-manifest",
        manifest.work_generation_digest,
        manifest.request_generation_id,
        manifest.executor_session_generation_id,
        manifest.runtime_generation_id,
        manifest.native_block_generation_digest,
        manifest.sample_chunk_count,
        manifest.sample_count,
        manifest.sample_manifest_digest,
        manifest.first_flat_sample_index,
        manifest.last_flat_sample_index,
    )


def _block_receipt_digest(receipt: PaperKineticFullGeometryBlockReceipt) -> str:
    return _digest_parts(
        FULL_GEOMETRY_PROVENANCE,
        "consumed-full-geometry-block",
        receipt.native_block_generation_digest,
        receipt.request_generation_id,
        receipt.work_generation_digest,
        receipt.executor_session_generation_id,
        receipt.executor_runtime_generation_id,
        receipt.executor_execution_generation_digest,
        receipt.reduced_sample_chunk_count,
        receipt.reduced_sample_count,
        receipt.global_loss_element_count,
        receipt.loss_scale,
        receipt.loss_normalization_id,
        receipt.sample_manifest_digest,
        receipt.first_flat_sample_index,
        receipt.last_flat_sample_index,
        receipt.native_vjp_provenance_id,
        receipt.geometry_reduction_generation_digest,
        receipt.row_geometry_vjp_call_count,
        receipt.differentiable_word_reverse_interactions,
        receipt.dense_global_site_accumulation_elements,
        receipt.all_site_owner_validation_evaluations,
        receipt.native_length_bar_tensor_bytes,
        receipt.geometry_bridge_visible_tensor_bytes,
        receipt.native_full_vjp_invocation_count,
        receipt.device_completion_fence_call_count,
        receipt.execution_consumed,
    )


def _receipt_digest(receipt: PaperKineticFullGeometryRequestReceipt) -> str:
    return _digest_parts(
        FULL_GEOMETRY_PROVENANCE,
        receipt.step_generation_id,
        receipt.request_generation_id,
        receipt.view_index,
        receipt.block_id,
        receipt.ray_bar_keys,
        tuple(sorted(dict(receipt.accounting).items())),
    )


def _require_distinct_storage(*tensors: torch.Tensor) -> None:
    identities = {
        (str(tensor.device), tensor.untyped_storage().data_ptr()) for tensor in tensors
    }
    if len(identities) != len(tensors):
        raise ValueError("full-geometry material and geometry bars must not alias")


def _require_finite(tensor: torch.Tensor, *, name: str) -> None:
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must be finite")


def _require_cpu_only_tensor(tensor: torch.Tensor, *, name: str) -> None:
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
        raise ValueError(f"{name} must remain on CPU in the fake-native seam")


def _require_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    dtype: torch.dtype,
    device: torch.device,
    shape: tuple[int, ...],
) -> None:
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.dtype != dtype
        or tensor.device != device
        or tensor.layout != torch.strided
        or tuple(tensor.shape) != shape
        or not tensor.is_contiguous()
        or tensor.requires_grad
    ):
        raise ValueError(f"{name} has invalid dtype/device/layout/shape")


def _require_cpu_f64(
    tensor: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, ...],
    require_finite: bool = True,
) -> None:
    _require_tensor(
        tensor,
        name=name,
        dtype=torch.float64,
        device=torch.device("cpu"),
        shape=shape,
    )
    if require_finite and not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must be finite")


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


def _tensor_bytes(*tensors: torch.Tensor) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _require_positive_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _fail_value(message: str) -> None:
    raise ValueError(message)


def _digest_parts(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = repr(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


__all__ = [
    "FULL_GEOMETRY_PROVENANCE",
    "FULL_GEOMETRY_STATUS",
    "PaperKineticFullGeometryBlockReceipt",
    "PaperKineticFullGeometryMemoryReport",
    "PaperKineticFullGeometryRequestAssembly",
    "PaperKineticFullGeometryRequestReceipt",
    "begin_paper_kinetic_full_geometry_request",
    "consume_paper_kinetic_full_geometry_native_block",
    "finalize_and_consume_paper_kinetic_full_geometry_request",
]
