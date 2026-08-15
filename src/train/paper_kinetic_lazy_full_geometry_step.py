"""Memory-light lazy spatial WorldFoam full-geometry step contract.

This module is the public sibling of the material-only lazy coordinator.  It
keeps the camera fixed, streams only selected observations, and exposes one
device material bar plus CPU-owned global position/velocity/weight bars.  The
two reverse modes are deliberately distinct:

``staged_sparse``
    one fenced native length cotangent and certified compact geometry
    reduction per active equal-rank block;

``fused_union_v2``
    one exact ``P_b = P_U Q_b`` request-union transaction per spatial bundle,
    followed by one bounded union-to-CPU readback and CPU ``index_add_``.

The public result and its later device bridge are not material-only results by
duck typing.  Their exact types, seals, generation digests, and geometry D2H
receipts remain separate.  Native runtime parity and allocator peaks are not
claimed by this source-complete seam.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch
from paper_kinetic_lazy_program_bundles import (
    PaperKineticLazyProgramBundleProvider,
    PaperKineticObservation,
)
from paper_kinetic_runtime_paths import ensure_worldfoam_lane2_research_path

ensure_worldfoam_lane2_research_path()

from kinetic_native_equal_rank_sparse_geometry_reduction import (  # noqa: E402
    KineticNativeEqualRankSparseGeometryReduction,
)


STAGED_SPARSE = "staged_sparse"
FUSED_UNION_V2 = "fused_union_v2"
FULL_GEOMETRY_REVERSE_MODES = frozenset((STAGED_SPARSE, FUSED_UNION_V2))

STEP_PROVENANCE = "paper-kinetic-lazy-native-full-geometry-step-v1"
STEP_STATUS = "source_only/native_runtime_unverified"
GEOMETRY_D2H_RECEIPT_PROVENANCE = (
    "paper-kinetic-lazy-full-geometry-d2h-receipt-v1"
)

# Literal source-capability seal consumed by the fresh-process paper driver.
# Runtime execution, allocator peaks, RSS, and MPS measurements are deliberately
# absent: those are attested only by the producer after a real worker exits.
PAPER_KINETIC_LAZY_FULL_GEOMETRY_CAPABILITY_SEAL = {
    "seal_id": "paper-kinetic-lazy-full-geometry-step-capability-seal-v1",
    "real_native_spatial_block_coordinator": True,
    "full_geometry_trainable": True,
    "all_competitor_active_owner_certification": True,
    "post_certification_compact_device_lowering": True,
    "production_device_material_gradient_receipt": True,
    "production_geometry_device_to_host_reduction_receipt": True,
    "geometry_optimizer_authorization_receipt": True,
    "cpu_manual_sgd_mutation": True,
    "checkpoint_restart_lifecycle": True,
    "staged_sparse_mode": True,
    "fused_union_v2_mode": True,
    "direct_selected_pixel_target_stream": True,
    "zero_full_frame_target_materialization": True,
}

_RESULT_SEAL = object()
_D2H_RECEIPT_SEAL = object()
_CONTEXT_SEAL = object()


@dataclass(frozen=True)
class PaperKineticLazyFullGeometryMemoryPolicy:
    """Full-geometry bounds layered over the existing lazy memory policy."""

    maximum_global_geometry_bar_logical_tensor_bytes: int
    maximum_geometry_bridge_visible_peak_logical_tensor_bytes: int
    maximum_fused_union_transaction_scratch_tensor_bytes: int

    def assert_valid(self, *, reverse_mode: str) -> None:
        if reverse_mode not in FULL_GEOMETRY_REVERSE_MODES:
            raise ValueError(
                "full-geometry reverse_mode must be staged_sparse or fused_union_v2"
            )
        values = (
            self.maximum_global_geometry_bar_logical_tensor_bytes,
            self.maximum_geometry_bridge_visible_peak_logical_tensor_bytes,
            self.maximum_fused_union_transaction_scratch_tensor_bytes,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in values
        ):
            raise TypeError("full-geometry memory-policy limits must be integers")
        if any(value < 1 for value in values[:2]):
            raise ValueError("full-geometry memory-policy limits must be positive")
        if reverse_mode == FUSED_UNION_V2:
            if self.maximum_fused_union_transaction_scratch_tensor_bytes < 1:
                raise ValueError(
                    "fused_union_v2 requires a positive transaction scratch bound"
                )
        elif self.maximum_fused_union_transaction_scratch_tensor_bytes != 0:
            raise ValueError(
                "staged_sparse requires a zero fused-union scratch bound"
            )


@dataclass(frozen=True)
class PaperKineticLazyGeometryD2HReceipt:
    """Tensor-free proof of one bounded geometry device-to-host boundary."""

    reverse_mode: str
    bundle_index: int
    completion_fence_sequence: int
    completion_launch_generation_digest: str
    completion_receipt_generation_digest: str
    source_index_space: str
    source_site_count: int
    global_site_count: int
    weight_coefficient_count: int
    active_native_block_count: int
    source_site_ids_digest: str
    source_transaction_generation_id: str
    source_tensor_bytes: int
    device_to_host_tensor_bytes: int
    cpu_tensor_bytes: int
    device_to_host_tensor_count: int
    cpu_geometry_output_tensor_count: int
    generation_digest: str
    exact_request_union_identity_certified: bool
    compact_owner_certificate_consumed: bool
    fixed_camera: bool = True
    ray_gradient_tensor_count: int = 0
    persistent_frame_tensor_bytes: int = 0
    persistent_sample_tensor_bytes: int = 0
    persistent_target_tensor_bytes: int = 0
    persistent_prediction_tensor_bytes: int = 0
    native_runtime_verified: bool = False
    allocator_peak_measured: bool = False
    provenance: str = GEOMETRY_D2H_RECEIPT_PROVENANCE
    _seal: object = field(default=None, repr=False)

    def assert_current(self) -> None:
        staged = self.reverse_mode == STAGED_SPARSE
        fused = self.reverse_mode == FUSED_UNION_V2
        expected_space = "block_compact" if staged else "request_union"
        if (
            self._seal is not _D2H_RECEIPT_SEAL
            or self.provenance != GEOMETRY_D2H_RECEIPT_PROVENANCE
            or not (staged or fused)
            or self.bundle_index < 0
            or self.completion_fence_sequence < 1
            or not _is_sha256(self.completion_launch_generation_digest)
            or not _is_sha256(self.completion_receipt_generation_digest)
            or self.source_index_space != expected_space
            or self.source_site_count < 1
            or self.global_site_count < self.source_site_count
            or self.weight_coefficient_count < 1
            or self.active_native_block_count < 1
            or not _is_sha256(self.source_site_ids_digest)
            or not self.source_transaction_generation_id.strip()
            or self.source_tensor_bytes < 1
            or self.device_to_host_tensor_bytes < 1
            or self.cpu_tensor_bytes < 1
            or self.device_to_host_tensor_count < 1
            or self.cpu_geometry_output_tensor_count != 3
            or self.exact_request_union_identity_certified != fused
            or self.compact_owner_certificate_consumed != staged
            or not self.fixed_camera
            or self.ray_gradient_tensor_count != 0
            or any(
                value != 0
                for value in (
                    self.persistent_frame_tensor_bytes,
                    self.persistent_sample_tensor_bytes,
                    self.persistent_target_tensor_bytes,
                    self.persistent_prediction_tensor_bytes,
                )
            )
            or self.native_runtime_verified
            or self.allocator_peak_measured
            or self.generation_digest != _geometry_d2h_receipt_digest(self)
        ):
            raise ValueError("lazy full-geometry D2H receipt changed")

    def accounting(self) -> dict[str, Any]:
        self.assert_current()
        return {
            "provenance": self.provenance,
            "generation_digest": self.generation_digest,
            "reverse_mode": self.reverse_mode,
            "bundle_index": self.bundle_index,
            "completion_fence_sequence": self.completion_fence_sequence,
            "completion_launch_generation_digest": (
                self.completion_launch_generation_digest
            ),
            "completion_receipt_generation_digest": (
                self.completion_receipt_generation_digest
            ),
            "source_index_space": self.source_index_space,
            "source_site_count": self.source_site_count,
            "global_site_count": self.global_site_count,
            "weight_coefficient_count": self.weight_coefficient_count,
            "active_native_block_count": self.active_native_block_count,
            "source_site_ids_digest": self.source_site_ids_digest,
            "source_transaction_generation_id": (
                self.source_transaction_generation_id
            ),
            "source_tensor_bytes": self.source_tensor_bytes,
            "device_to_host_tensor_bytes": self.device_to_host_tensor_bytes,
            "cpu_tensor_bytes": self.cpu_tensor_bytes,
            "device_to_host_tensor_count": self.device_to_host_tensor_count,
            "cpu_geometry_output_tensor_count": (
                self.cpu_geometry_output_tensor_count
            ),
            "exact_request_union_identity_certified": (
                self.exact_request_union_identity_certified
            ),
            "compact_owner_certificate_consumed": (
                self.compact_owner_certificate_consumed
            ),
            "fixed_camera": True,
            "ray_gradient_tensor_count": 0,
            "persistent_frame_tensor_bytes": 0,
            "persistent_sample_tensor_bytes": 0,
            "persistent_target_tensor_bytes": 0,
            "persistent_prediction_tensor_bytes": 0,
            "native_runtime_verified": False,
            "allocator_peak_measured": False,
        }


@dataclass
class PaperKineticLazyNativeFullGeometryStepResult:
    """Distinct full-geometry authorization for one external CPU updater."""

    step_index: int
    step_generation_id: str
    provider_generation_digest: str
    world_generation_digest: str
    sites_content_digest: str
    geometry_generation_id: str
    loss_normalization_id: str
    material_generation_id: str
    background_generation_id: str
    reverse_mode: str
    loss_f32: torch.Tensor = field(repr=False)
    grad_global_site_rgba_f32: torch.Tensor = field(repr=False)
    grad_positions0_f64_cpu: torch.Tensor = field(repr=False)
    grad_velocities_f64_cpu: torch.Tensor = field(repr=False)
    grad_weight_coefficients_f64_cpu: torch.Tensor = field(repr=False)
    geometry_d2h_receipts: tuple[PaperKineticLazyGeometryD2HReceipt, ...]
    accounting: Mapping[str, int | float | str | bool]
    generation_digest: str
    _tensor_signatures: tuple[tuple[object, ...], ...] = field(repr=False)
    _cpu_geometry_content_digests: tuple[str, ...] = field(repr=False)
    _material_tensor_identity: int = field(repr=False)
    _material_tensor_signature: tuple[object, ...] = field(repr=False)
    _background_tensor_identity: int = field(repr=False)
    _background_tensor_signature: tuple[object, ...] = field(repr=False)
    _sealed_completion_fence: Any = field(repr=False)
    _sealed_completion_fence_identity: int = field(repr=False)
    _sealed_completion_fence_generation_digest: str = field(repr=False)
    issued_bridge_receipt_identity: int = 0
    issued_bridge_receipt_generation_digest: str = ""
    fixed_camera: bool = True
    camera_ray_gradients_enabled: bool = False
    ray_gradient_tensor_count: int = 0
    native_runtime_verified: bool = False
    allocator_peak_measured: bool = False
    provenance: str = STEP_PROVENANCE
    runtime_status: str = STEP_STATUS
    _seal: object = field(default=None, repr=False)

    @property
    def site_count(self) -> int:
        return int(self.grad_positions0_f64_cpu.shape[0])

    @property
    def weight_coefficient_count(self) -> int:
        return int(self.grad_weight_coefficients_f64_cpu.shape[1])

    def assert_current(self) -> None:
        from kinetic_sealed_completion_fence import (  # local research import
            PaperKineticSealedCompletionFence,
        )

        tensors = (
            self.loss_f32,
            self.grad_global_site_rgba_f32,
            self.grad_positions0_f64_cpu,
            self.grad_velocities_f64_cpu,
            self.grad_weight_coefficients_f64_cpu,
        )
        if (
            self._seal is not _RESULT_SEAL
            or self.provenance != STEP_PROVENANCE
            or self.runtime_status != STEP_STATUS
            or self.reverse_mode not in FULL_GEOMETRY_REVERSE_MODES
            or self.step_index < 0
            or not self.step_generation_id.strip()
            or not _is_sha256(self.provider_generation_digest)
            or not _is_sha256(self.world_generation_digest)
            or not _is_sha256(self.sites_content_digest)
            or not _is_sha256(self.geometry_generation_id)
            or not self.loss_normalization_id.strip()
            or not _is_sha256(self.material_generation_id)
            or not _is_sha256(self.background_generation_id)
            or not self.geometry_d2h_receipts
            or any(
                receipt.reverse_mode != self.reverse_mode
                for receipt in self.geometry_d2h_receipts
            )
            or not self.fixed_camera
            or self.camera_ray_gradients_enabled
            or self.ray_gradient_tensor_count != 0
            or self.native_runtime_verified
            or self.allocator_peak_measured
            or not isinstance(self.accounting, MappingProxyType)
            or type(self._sealed_completion_fence)
            is not PaperKineticSealedCompletionFence
            or id(self._sealed_completion_fence)
            != self._sealed_completion_fence_identity
            or self._sealed_completion_fence.generation_digest
            != self._sealed_completion_fence_generation_digest
            or bool(self.issued_bridge_receipt_identity)
            != bool(self.issued_bridge_receipt_generation_digest)
            or self.issued_bridge_receipt_identity < 0
            or self.issued_bridge_receipt_generation_digest
            and not _is_sha256(self.issued_bridge_receipt_generation_digest)
            or tuple(_tensor_signature(tensor) for tensor in tensors)
            != self._tensor_signatures
            or self._cpu_geometry_content_digests
            != tuple(
                _tensor_content_digest(tensor)
                for tensor in tensors[2:]
            )
            or self.generation_digest != _full_geometry_result_digest(self)
        ):
            raise ValueError("lazy native full-geometry step result changed")
        _require_device_f32(
            self.loss_f32,
            device=self.grad_global_site_rgba_f32.device,
            shape=(1,),
            name="full-geometry loss",
        )
        _require_device_f32(
            self.grad_global_site_rgba_f32,
            device=self.grad_global_site_rgba_f32.device,
            shape=(self.site_count, 4),
            name="full-geometry material bar",
        )
        _require_cpu_f64_geometry_bars(
            self.grad_positions0_f64_cpu,
            self.grad_velocities_f64_cpu,
            self.grad_weight_coefficients_f64_cpu,
            site_count=self.site_count,
        )
        # The material/loss bars remain device resident until the separately
        # fenced bridge.  Do not smuggle an eager MPS readback into this warm
        # assertion.  Geometry is already CPU owned at this boundary and may
        # be checked without creating another device synchronization.
        if any(
            not bool(torch.isfinite(tensor).all().item())
            for tensor in tensors[2:]
        ):
            raise FloatingPointError("lazy full-geometry result is nonfinite")
        for receipt in self.geometry_d2h_receipts:
            receipt.assert_current()
        if (
            self.accounting.get("reverse_mode") != self.reverse_mode
            or self.accounting.get("full_geometry") is not True
            or self.accounting.get("fixed_camera") is not True
            or self.accounting.get("camera_ray_gradients_enabled") is not False
            or self.accounting.get("ray_gradient_tensor_count") != 0
            or self.accounting.get("geometry_d2h_receipt_count")
            != len(self.geometry_d2h_receipts)
            or self.accounting.get("native_runtime_verified") is not False
            or self.accounting.get("allocator_peak_measured") is not False
        ):
            raise ValueError("lazy full-geometry result accounting changed")

    def assert_device_snapshot_tensors(
        self,
        *,
        material_tensor: torch.Tensor,
        background_tensor: torch.Tensor,
    ) -> None:
        self.assert_current()
        if (
            id(material_tensor) != self._material_tensor_identity
            or _tensor_signature(material_tensor)
            != self._material_tensor_signature
            or id(background_tensor) != self._background_tensor_identity
            or _tensor_signature(background_tensor)
            != self._background_tensor_signature
        ):
            raise ValueError("lazy full-geometry result has a foreign device snapshot")

    def claim_bridge_receipt(
        self,
        *,
        receipt_identity: int,
        receipt_generation_digest: str,
    ) -> None:
        self.assert_current()
        if self.issued_bridge_receipt_identity:
            raise ValueError("full-geometry result already issued a bridge receipt")
        if receipt_identity < 1 or not _is_sha256(receipt_generation_digest):
            raise ValueError("full-geometry bridge receipt identity is invalid")
        self.issued_bridge_receipt_identity = receipt_identity
        self.issued_bridge_receipt_generation_digest = receipt_generation_digest
        self.assert_current()


@dataclass
class _PaperKineticLazyFullGeometryExecutionContext:
    """Internal CPU bar owner used by the shared lazy scheduler."""

    provider: PaperKineticLazyProgramBundleProvider = field(repr=False)
    reverse_mode: str
    policy: PaperKineticLazyFullGeometryMemoryPolicy
    geometry_generation_id: str
    grad_positions0_f64_cpu: torch.Tensor = field(repr=False)
    grad_velocities_f64_cpu: torch.Tensor = field(repr=False)
    grad_weight_coefficients_f64_cpu: torch.Tensor = field(repr=False)
    initial_tensor_signatures: tuple[tuple[object, ...], ...] = field(repr=False)
    d2h_receipts: list[PaperKineticLazyGeometryD2HReceipt] = field(
        default_factory=list,
        repr=False,
    )
    native_full_geometry_vjp_launch_count: int = 0
    native_fused_union_v2_transaction_count: int = 0
    geometry_compact_to_global_scatter_row_count: int = 0
    geometry_union_to_global_scatter_row_count: int = 0
    maximum_geometry_bridge_visible_peak_logical_tensor_bytes: int = 0
    _seal: object = field(default=None, repr=False)

    @property
    def site_count(self) -> int:
        return self.provider.world.site_count

    @property
    def weight_coefficient_count(self) -> int:
        return int(self.grad_weight_coefficients_f64_cpu.shape[1])

    def assert_current(self) -> None:
        if (
            self._seal is not _CONTEXT_SEAL
            or self.reverse_mode not in FULL_GEOMETRY_REVERSE_MODES
            or not _is_sha256(self.geometry_generation_id)
            or tuple(
                _writable_tensor_binding_signature(tensor)
                for tensor in self._geometry_bars()
            )
            != self.initial_tensor_signatures
            or self.native_full_geometry_vjp_launch_count < 0
            or self.native_fused_union_v2_transaction_count < 0
            or self.geometry_compact_to_global_scatter_row_count < 0
            or self.geometry_union_to_global_scatter_row_count < 0
            or self.maximum_geometry_bridge_visible_peak_logical_tensor_bytes < 0
        ):
            raise ValueError("lazy full-geometry execution context changed")
        self.provider.assert_warm_current()
        self.policy.assert_valid(reverse_mode=self.reverse_mode)
        _require_cpu_f64_geometry_bars(
            *self._geometry_bars(),
            site_count=self.site_count,
        )
        if _tensor_bytes(*self._geometry_bars()) > (
            self.policy.maximum_global_geometry_bar_logical_tensor_bytes
        ):
            raise MemoryError("global CPU geometry bars exceed their bound")
        for receipt in self.d2h_receipts:
            receipt.assert_current()

    def _geometry_bars(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.grad_positions0_f64_cpu,
            self.grad_velocities_f64_cpu,
            self.grad_weight_coefficients_f64_cpu,
        )

    @torch.no_grad()
    def accumulate_staged_sparse(
        self,
        reduction: KineticNativeEqualRankSparseGeometryReduction,
        *,
        bundle_index: int,
        completion_fence_sequence: int,
        completion_launch_generation_digest: str,
        completion_receipt_generation_digest: str,
    ) -> None:
        self.assert_current()
        if self.reverse_mode != STAGED_SPARSE:
            raise ValueError("staged geometry contribution used in another mode")
        reduction.assert_current()
        if reduction.ray_gradients_included or reduction.ray_bar_keys:
            raise ValueError("fixed-camera staged reduction returned ray bars")
        source_ids = reduction.source_site_ids_i64
        if source_ids.device.type != "cpu" or source_ids.dtype != torch.int64:
            raise ValueError("staged geometry source ids must be CPU int64")
        self.grad_positions0_f64_cpu.index_add_(
            0,
            source_ids,
            reduction.grad_compact_positions0_f64,
        )
        self.grad_velocities_f64_cpu.index_add_(
            0,
            source_ids,
            reduction.grad_compact_velocities_f64,
        )
        self.grad_weight_coefficients_f64_cpu.index_add_(
            0,
            source_ids,
            reduction.grad_compact_weight_coefficients_f64,
        )
        self.native_full_geometry_vjp_launch_count += 1
        self.geometry_compact_to_global_scatter_row_count += int(source_ids.numel())
        self.maximum_geometry_bridge_visible_peak_logical_tensor_bytes = max(
            self.maximum_geometry_bridge_visible_peak_logical_tensor_bytes,
            reduction.memory.bridge_visible_peak_logical_tensor_bytes,
        )
        self._append_receipt(
            reverse_mode=STAGED_SPARSE,
            bundle_index=bundle_index,
            completion_fence_sequence=completion_fence_sequence,
            completion_launch_generation_digest=(
                completion_launch_generation_digest
            ),
            completion_receipt_generation_digest=(
                completion_receipt_generation_digest
            ),
            source_index_space="block_compact",
            source_site_ids=tuple(int(value) for value in source_ids.tolist()),
            active_native_block_count=1,
            source_tensor_bytes=(
                reduction.memory.native_full_length_bar_tensor_bytes
            ),
            cpu_tensor_bytes=_tensor_bytes(
                reduction.grad_compact_positions0_f64,
                reduction.grad_compact_velocities_f64,
                reduction.grad_compact_weight_coefficients_f64,
            ),
            exact_request_union_identity_certified=False,
            compact_owner_certificate_consumed=True,
            device_to_host_tensor_count=(
                reduction.native_length_bar_row_copy_count
            ),
            device_to_host_tensor_bytes=(
                reduction.memory.native_full_length_bar_tensor_bytes
            ),
            source_transaction_generation_id=reduction.generation_digest,
        )

    @torch.no_grad()
    def consume_and_accumulate_fused_union_v2(
        self,
        transaction_result: Any,
        *,
        bundle_index: int,
        settle_device_outputs: Callable[
            [tuple[torch.Tensor, ...]], Any
        ],
    ) -> None:
        from kinetic_native_equal_rank_runtime_adapter import (
            KineticNativeEqualRankFusedUnionFullVjpV2TransactionResult,
        )
        from kinetic_sealed_completion_fence import (
            PaperKineticCompletionFenceReceipt,
        )

        self.assert_current()
        if self.reverse_mode != FUSED_UNION_V2:
            raise ValueError("union-v2 contribution used in another mode")
        if type(transaction_result) is not (
            KineticNativeEqualRankFusedUnionFullVjpV2TransactionResult
        ):
            raise TypeError("union-v2 accumulation requires its exact accepted result")
        transaction_result.assert_current()
        (
            union_source_site_ids,
            compact_material_bars,
            grad_union_positions_f32,
            grad_union_velocities_f32,
            grad_union_weights_f32,
        ) = transaction_result.consume_bars_once()
        retained_device_source_tensor_bytes = _tensor_bytes(
            grad_union_positions_f32,
            grad_union_velocities_f32,
            grad_union_weights_f32,
        )
        # Exactly three bounded union-local bars cross the device boundary.
        # Compact material bars remain on device for the caller's exact
        # block-local scatter into the sole global material bar.
        grad_union_positions0_f64_cpu = (
            grad_union_positions_f32.detach().to(device="cpu", dtype=torch.float64).contiguous()
        )
        grad_union_velocities_f64_cpu = (
            grad_union_velocities_f32.detach().to(device="cpu", dtype=torch.float64).contiguous()
        )
        grad_union_weight_coefficients_f64_cpu = (
            grad_union_weights_f32.detach().to(device="cpu", dtype=torch.float64).contiguous()
        )
        completion_receipt = settle_device_outputs(compact_material_bars)
        if type(completion_receipt) is not PaperKineticCompletionFenceReceipt:
            raise TypeError("union-v2 settlement returned a foreign receipt")
        completion_receipt.assert_current()
        if (
            not union_source_site_ids
            or tuple(sorted(set(union_source_site_ids)))
            != union_source_site_ids
            or union_source_site_ids[0] < 0
            or union_source_site_ids[-1] >= self.site_count
        ):
            raise ValueError("fused union source identity is not exact")
        union_count = len(union_source_site_ids)
        _require_cpu_f64_geometry_bars(
            grad_union_positions0_f64_cpu,
            grad_union_velocities_f64_cpu,
            grad_union_weight_coefficients_f64_cpu,
            site_count=union_count,
        )
        ids = torch.tensor(union_source_site_ids, dtype=torch.int64)
        self.grad_positions0_f64_cpu.index_add_(
            0, ids, grad_union_positions0_f64_cpu
        )
        self.grad_velocities_f64_cpu.index_add_(
            0, ids, grad_union_velocities_f64_cpu
        )
        self.grad_weight_coefficients_f64_cpu.index_add_(
            0, ids, grad_union_weight_coefficients_f64_cpu
        )
        self.native_fused_union_v2_transaction_count += 1
        self.geometry_union_to_global_scatter_row_count += union_count
        cpu_bytes = _tensor_bytes(
            grad_union_positions0_f64_cpu,
            grad_union_velocities_f64_cpu,
            grad_union_weight_coefficients_f64_cpu,
        )
        self.maximum_geometry_bridge_visible_peak_logical_tensor_bytes = max(
            self.maximum_geometry_bridge_visible_peak_logical_tensor_bytes,
            retained_device_source_tensor_bytes + cpu_bytes,
        )
        self._append_receipt(
            reverse_mode=FUSED_UNION_V2,
            bundle_index=bundle_index,
            completion_fence_sequence=completion_receipt.fence_sequence,
            completion_launch_generation_digest=(
                completion_receipt.launch_generation_digest
            ),
            completion_receipt_generation_digest=(
                completion_receipt.generation_digest
            ),
            source_index_space="request_union",
            source_site_ids=union_source_site_ids,
            active_native_block_count=transaction_result.block_count,
            source_tensor_bytes=retained_device_source_tensor_bytes,
            cpu_tensor_bytes=cpu_bytes,
            exact_request_union_identity_certified=True,
            compact_owner_certificate_consumed=False,
            device_to_host_tensor_count=3,
            device_to_host_tensor_bytes=retained_device_source_tensor_bytes,
            source_transaction_generation_id=(
                transaction_result.transaction_generation_id
            ),
        )

    def _append_receipt(
        self,
        *,
        reverse_mode: str,
        bundle_index: int,
        completion_fence_sequence: int,
        completion_launch_generation_digest: str,
        completion_receipt_generation_digest: str,
        source_index_space: str,
        source_site_ids: tuple[int, ...],
        active_native_block_count: int,
        source_tensor_bytes: int,
        cpu_tensor_bytes: int,
        exact_request_union_identity_certified: bool,
        compact_owner_certificate_consumed: bool,
        device_to_host_tensor_count: int,
        device_to_host_tensor_bytes: int,
        source_transaction_generation_id: str,
    ) -> None:
        provisional = PaperKineticLazyGeometryD2HReceipt(
            reverse_mode=reverse_mode,
            bundle_index=bundle_index,
            completion_fence_sequence=completion_fence_sequence,
            completion_launch_generation_digest=(
                completion_launch_generation_digest
            ),
            completion_receipt_generation_digest=(
                completion_receipt_generation_digest
            ),
            source_index_space=source_index_space,
            source_site_count=len(source_site_ids),
            global_site_count=self.site_count,
            weight_coefficient_count=self.weight_coefficient_count,
            active_native_block_count=active_native_block_count,
            source_site_ids_digest=_digest_parts(source_site_ids),
            source_transaction_generation_id=source_transaction_generation_id,
            source_tensor_bytes=source_tensor_bytes,
            device_to_host_tensor_bytes=device_to_host_tensor_bytes,
            cpu_tensor_bytes=cpu_tensor_bytes,
            device_to_host_tensor_count=device_to_host_tensor_count,
            cpu_geometry_output_tensor_count=3,
            generation_digest="",
            exact_request_union_identity_certified=(
                exact_request_union_identity_certified
            ),
            compact_owner_certificate_consumed=(
                compact_owner_certificate_consumed
            ),
            _seal=_D2H_RECEIPT_SEAL,
        )
        receipt = PaperKineticLazyGeometryD2HReceipt(
            **{
                **provisional.__dict__,
                "generation_digest": _geometry_d2h_receipt_digest(provisional),
            }
        )
        receipt.assert_current()
        self.d2h_receipts.append(receipt)

    def build_result(
        self,
        *,
        step_index: int,
        step_generation_id: str,
        provider_generation_digest: str,
        world_generation_digest: str,
        sites_content_digest: str,
        loss_normalization_id: str,
        material_generation_id: str,
        background_generation_id: str,
        loss_f32: torch.Tensor,
        grad_global_site_rgba_f32: torch.Tensor,
        material_tensor: torch.Tensor,
        background_tensor: torch.Tensor,
        sealed_completion_fence: Any,
        accounting: Mapping[str, int | float | str | bool],
    ) -> PaperKineticLazyNativeFullGeometryStepResult:
        self.assert_current()
        if not self.d2h_receipts:
            raise ValueError("full-geometry step has no geometry D2H receipt")
        result_accounting = MappingProxyType(
            {
                **dict(accounting),
                "reverse_mode": self.reverse_mode,
                "full_geometry": True,
                "fixed_camera": True,
                "camera_ray_gradients_enabled": False,
                "ray_gradient_tensor_count": 0,
                "geometry_d2h_receipt_count": len(self.d2h_receipts),
                "geometry_d2h_source_tensor_bytes": sum(
                    receipt.source_tensor_bytes for receipt in self.d2h_receipts
                ),
                "geometry_d2h_cpu_tensor_bytes": sum(
                    receipt.cpu_tensor_bytes for receipt in self.d2h_receipts
                ),
                "geometry_device_to_host_tensor_bytes": sum(
                    receipt.device_to_host_tensor_bytes
                    for receipt in self.d2h_receipts
                ),
                "geometry_device_to_host_tensor_count": sum(
                    receipt.device_to_host_tensor_count
                    for receipt in self.d2h_receipts
                ),
                "geometry_cpu_output_tensor_count": sum(
                    receipt.cpu_geometry_output_tensor_count
                    for receipt in self.d2h_receipts
                ),
                "native_full_geometry_vjp_launch_count": (
                    self.native_full_geometry_vjp_launch_count
                ),
                "native_fused_union_v2_transaction_count": (
                    self.native_fused_union_v2_transaction_count
                ),
                "geometry_compact_to_global_scatter_row_count": (
                    self.geometry_compact_to_global_scatter_row_count
                ),
                "geometry_union_to_global_scatter_row_count": (
                    self.geometry_union_to_global_scatter_row_count
                ),
                "global_cpu_geometry_bar_logical_tensor_bytes": _tensor_bytes(
                    *self._geometry_bars()
                ),
                "maximum_geometry_bridge_visible_peak_logical_tensor_bytes": (
                    self.maximum_geometry_bridge_visible_peak_logical_tensor_bytes
                ),
                "persistent_frame_tensor_bytes": 0,
                "persistent_sample_tensor_bytes": 0,
                "persistent_target_tensor_bytes": 0,
                "persistent_prediction_tensor_bytes": 0,
                "native_runtime_verified": False,
                "allocator_peak_measured": False,
            }
        )
        tensors = (
            loss_f32,
            grad_global_site_rgba_f32,
            *self._geometry_bars(),
        )
        provisional = PaperKineticLazyNativeFullGeometryStepResult(
            step_index=step_index,
            step_generation_id=step_generation_id,
            provider_generation_digest=provider_generation_digest,
            world_generation_digest=world_generation_digest,
            sites_content_digest=sites_content_digest,
            geometry_generation_id=self.geometry_generation_id,
            loss_normalization_id=loss_normalization_id,
            material_generation_id=material_generation_id,
            background_generation_id=background_generation_id,
            reverse_mode=self.reverse_mode,
            loss_f32=loss_f32,
            grad_global_site_rgba_f32=grad_global_site_rgba_f32,
            grad_positions0_f64_cpu=self.grad_positions0_f64_cpu,
            grad_velocities_f64_cpu=self.grad_velocities_f64_cpu,
            grad_weight_coefficients_f64_cpu=(
                self.grad_weight_coefficients_f64_cpu
            ),
            geometry_d2h_receipts=tuple(self.d2h_receipts),
            accounting=result_accounting,
            generation_digest="",
            _tensor_signatures=tuple(_tensor_signature(tensor) for tensor in tensors),
            _cpu_geometry_content_digests=tuple(
                _tensor_content_digest(tensor) for tensor in tensors[2:]
            ),
            _material_tensor_identity=id(material_tensor),
            _material_tensor_signature=_tensor_signature(material_tensor),
            _background_tensor_identity=id(background_tensor),
            _background_tensor_signature=_tensor_signature(background_tensor),
            _sealed_completion_fence=sealed_completion_fence,
            _sealed_completion_fence_identity=id(sealed_completion_fence),
            _sealed_completion_fence_generation_digest=(
                sealed_completion_fence.generation_digest
            ),
            _seal=_RESULT_SEAL,
        )
        provisional.generation_digest = _full_geometry_result_digest(provisional)
        provisional.assert_current()
        return provisional


def prepare_paper_kinetic_lazy_full_geometry_execution_context(
    provider: PaperKineticLazyProgramBundleProvider,
    *,
    reverse_mode: str,
    policy: PaperKineticLazyFullGeometryMemoryPolicy,
    geometry_generation_id: str,
    grad_positions0_f64_cpu: torch.Tensor,
    grad_velocities_f64_cpu: torch.Tensor,
    grad_weight_coefficients_f64_cpu: torch.Tensor,
) -> _PaperKineticLazyFullGeometryExecutionContext:
    """Prepare and zero the three sole CPU-owned global geometry bars."""

    if not isinstance(provider, PaperKineticLazyProgramBundleProvider):
        raise TypeError("full-geometry context requires a lazy program provider")
    provider.assert_current()
    policy.assert_valid(reverse_mode=reverse_mode)
    if not _is_sha256(geometry_generation_id):
        raise ValueError("geometry_generation_id must be SHA-256")
    from paper_kinetic_fixed_camera_full_geometry_step import (
        paper_kinetic_fixed_camera_provider_geometry_generation_id,
    )

    if geometry_generation_id != (
        paper_kinetic_fixed_camera_provider_geometry_generation_id(provider)
    ):
        raise ValueError(
            "geometry_generation_id is foreign to the live fixed-camera world"
        )
    bars = (
        grad_positions0_f64_cpu,
        grad_velocities_f64_cpu,
        grad_weight_coefficients_f64_cpu,
    )
    _require_cpu_f64_geometry_bars(*bars, site_count=provider.world.site_count)
    if len({tensor.untyped_storage().data_ptr() for tensor in bars}) != 3:
        raise ValueError("global geometry bars must not alias")
    if _tensor_bytes(*bars) > policy.maximum_global_geometry_bar_logical_tensor_bytes:
        raise MemoryError("global CPU geometry bars exceed their preflight bound")
    with torch.no_grad():
        for tensor in bars:
            tensor.zero_()
    context = _PaperKineticLazyFullGeometryExecutionContext(
        provider=provider,
        reverse_mode=reverse_mode,
        policy=policy,
        geometry_generation_id=geometry_generation_id,
        grad_positions0_f64_cpu=grad_positions0_f64_cpu,
        grad_velocities_f64_cpu=grad_velocities_f64_cpu,
        grad_weight_coefficients_f64_cpu=grad_weight_coefficients_f64_cpu,
        initial_tensor_signatures=tuple(
            _writable_tensor_binding_signature(tensor) for tensor in bars
        ),
        _seal=_CONTEXT_SEAL,
    )
    context.assert_current()
    return context


@torch.no_grad()
def run_paper_kinetic_lazy_native_full_geometry_step(
    state: Any,
    provider: PaperKineticLazyProgramBundleProvider,
    observations: Iterable[PaperKineticObservation],
    *,
    step_index: int,
    expected_observation_count: int,
    expected_observation_manifest_digest: str,
    loss_normalization_id: str,
    material_generation_id: str,
    geometry_generation_id: str,
    background_generation_id: str,
    global_site_rgba_f32: torch.Tensor,
    global_grad_site_rgba_f32: torch.Tensor,
    grad_positions0_f64_cpu: torch.Tensor,
    grad_velocities_f64_cpu: torch.Tensor,
    grad_weight_coefficients_f64_cpu: torch.Tensor,
    background_rgb_f32: torch.Tensor,
    native_ops: Any,
    maximum_samples_per_launch: int,
    memory_policy: Any,
    full_geometry_memory_policy: PaperKineticLazyFullGeometryMemoryPolicy,
    reverse_mode: str,
    optimizer_update: Callable[
        [PaperKineticLazyNativeFullGeometryStepResult], None
    ],
    cone_tolerance: float = 1.0e-5,
) -> PaperKineticLazyNativeFullGeometryStepResult:
    """Run the shared selected-observation scheduler with full geometry."""

    if not callable(optimizer_update):
        raise TypeError("full-geometry optimizer_update must be callable")
    context = prepare_paper_kinetic_lazy_full_geometry_execution_context(
        provider,
        reverse_mode=reverse_mode,
        policy=full_geometry_memory_policy,
        geometry_generation_id=geometry_generation_id,
        grad_positions0_f64_cpu=grad_positions0_f64_cpu,
        grad_velocities_f64_cpu=grad_velocities_f64_cpu,
        grad_weight_coefficients_f64_cpu=grad_weight_coefficients_f64_cpu,
    )
    from kinetic_lazy_native_material_step import (  # local to avoid cycle
        run_paper_kinetic_lazy_native_material_step,
    )

    result = run_paper_kinetic_lazy_native_material_step(
        state,
        provider,
        observations,
        step_index=step_index,
        expected_observation_count=expected_observation_count,
        expected_observation_manifest_digest=expected_observation_manifest_digest,
        loss_normalization_id=loss_normalization_id,
        material_generation_id=material_generation_id,
        background_generation_id=background_generation_id,
        global_site_rgba_f32=global_site_rgba_f32,
        global_grad_site_rgba_f32=global_grad_site_rgba_f32,
        background_rgb_f32=background_rgb_f32,
        native_ops=native_ops,
        maximum_samples_per_launch=maximum_samples_per_launch,
        memory_policy=memory_policy,
        optimizer_update=optimizer_update,
        cone_tolerance=cone_tolerance,
        _full_geometry_context=context,
    )
    if type(result) is not PaperKineticLazyNativeFullGeometryStepResult:
        raise TypeError("shared lazy scheduler returned a material-only result")
    result.assert_current()
    return result


def run_worldfoam_training_memory_ablation_adapter(
    context: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Load the paper-ablation deployment adapter without creating a cycle."""

    from worldfoam_training_memory_ablation_adapter import (
        run_worldfoam_training_memory_ablation_adapter as run_adapter,
    )

    return run_adapter(context)


def _geometry_d2h_receipt_digest(
    receipt: PaperKineticLazyGeometryD2HReceipt,
) -> str:
    return _digest_parts(
        receipt.provenance,
        receipt.reverse_mode,
        receipt.bundle_index,
        receipt.completion_fence_sequence,
        receipt.completion_launch_generation_digest,
        receipt.completion_receipt_generation_digest,
        receipt.source_index_space,
        receipt.source_site_count,
        receipt.global_site_count,
        receipt.weight_coefficient_count,
        receipt.active_native_block_count,
        receipt.source_site_ids_digest,
        receipt.source_transaction_generation_id,
        receipt.source_tensor_bytes,
        receipt.device_to_host_tensor_bytes,
        receipt.cpu_tensor_bytes,
        receipt.device_to_host_tensor_count,
        receipt.cpu_geometry_output_tensor_count,
        receipt.exact_request_union_identity_certified,
        receipt.compact_owner_certificate_consumed,
        receipt.fixed_camera,
        receipt.ray_gradient_tensor_count,
    )


def _full_geometry_result_digest(
    result: PaperKineticLazyNativeFullGeometryStepResult,
) -> str:
    return _digest_parts(
        result.provenance,
        result.runtime_status,
        result.step_index,
        result.step_generation_id,
        result.provider_generation_digest,
        result.world_generation_digest,
        result.sites_content_digest,
        result.geometry_generation_id,
        result.loss_normalization_id,
        result.material_generation_id,
        result.background_generation_id,
        result.reverse_mode,
        tuple(receipt.generation_digest for receipt in result.geometry_d2h_receipts),
        tuple(sorted(result.accounting.items())),
        result._tensor_signatures,
        result._cpu_geometry_content_digests,
        result._material_tensor_identity,
        result._material_tensor_signature,
        result._background_tensor_identity,
        result._background_tensor_signature,
        result._sealed_completion_fence_identity,
        result._sealed_completion_fence_generation_digest,
    )


def _require_cpu_f64_geometry_bars(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    weights: torch.Tensor,
    *,
    site_count: int,
) -> None:
    weight_count = int(weights.shape[1]) if weights.ndim == 2 else 0
    for tensor, shape, name in (
        (positions, (site_count, 3), "position bar"),
        (velocities, (site_count, 3), "velocity bar"),
        (weights, (site_count, weight_count), "weight bar"),
    ):
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or tensor.dtype != torch.float64
            or tensor.layout != torch.strided
            or tuple(tensor.shape) != shape
            or not tensor.is_contiguous()
            or tensor.requires_grad
        ):
            raise ValueError(f"{name} must be contiguous non-autograd CPU float64")
    if weight_count < 1:
        raise ValueError("weight bar must have at least one coefficient")


def _require_device_f32(
    tensor: torch.Tensor,
    *,
    device: torch.device,
    shape: tuple[int, ...],
    name: str,
) -> None:
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.device != device
        or tensor.dtype != torch.float32
        or tensor.layout != torch.strided
        or tuple(tensor.shape) != shape
        or not tensor.is_contiguous()
        or tensor.requires_grad
    ):
        raise ValueError(f"{name} has invalid device/dtype/layout/shape")


def _tensor_signature(tensor: torch.Tensor) -> tuple[object, ...]:
    return (
        id(tensor),
        int(getattr(tensor, "_version", 0)),
        int(tensor.untyped_storage().data_ptr()),
        int(tensor.storage_offset()),
        tuple(int(value) for value in tensor.shape),
        tuple(int(value) for value in tensor.stride()),
        tensor.dtype,
        tensor.device,
        tensor.layout,
    )


def _writable_tensor_binding_signature(
    tensor: torch.Tensor,
) -> tuple[object, ...]:
    """Bind an explicitly writable bar without freezing its mutation version."""

    signature = _tensor_signature(tensor)
    return (signature[0], *signature[2:], bool(tensor.requires_grad))


def _tensor_content_digest(tensor: torch.Tensor) -> str:
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        raise ValueError("tensor content digest requires contiguous CPU storage")
    return hashlib.sha256(tensor.detach().numpy().tobytes(order="C")).hexdigest()


def _tensor_bytes(*tensors: torch.Tensor) -> int:
    return sum(int(tensor.numel()) * int(tensor.element_size()) for tensor in tensors)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _digest_parts(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = repr(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


__all__ = (
    "PAPER_KINETIC_LAZY_FULL_GEOMETRY_CAPABILITY_SEAL",
    "FUSED_UNION_V2",
    "FULL_GEOMETRY_REVERSE_MODES",
    "PaperKineticLazyFullGeometryMemoryPolicy",
    "PaperKineticLazyGeometryD2HReceipt",
    "PaperKineticLazyNativeFullGeometryStepResult",
    "STAGED_SPARSE",
    "prepare_paper_kinetic_lazy_full_geometry_execution_context",
    "run_paper_kinetic_lazy_native_full_geometry_step",
    "run_worldfoam_training_memory_ablation_adapter",
)
