"""Exact device-to-CPU bridge for one lazy full-geometry WorldFoam step.

The lazy scheduler intentionally leaves the scalar loss and the sole global
``[S,4]`` material cotangent on its execution device.  Geometry cotangents have
already crossed their separately receipted, bounded union/block boundary and
are CPU-owned.  This module performs exactly one final material/loss readback,
binds all five bars to the current immutable combined generation, and exposes
one one-shot authorization to the existing combined SGD/cold-recompile
transaction.

This is a source/runtime contract, not native evidence.  In particular it does
not claim that the currently built Metal extension exports the new full-
geometry ABI or that allocator peaks have been measured.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch
import paper_kinetic_fixed_camera_combined_state as _combined_state
from paper_kinetic_fixed_camera_combined_state import (
    PaperKineticFixedCameraColdRecompileManifest,
    PaperKineticFixedCameraColdRecompileReceipt,
    PaperKineticFixedCameraCombinedSGDPolicy,
    PaperKineticFixedCameraCombinedState,
)
from paper_kinetic_fixed_site_material_device_bridge import (
    PaperKineticFixedSiteMaterialDeviceSnapshot,
)
from paper_kinetic_lazy_full_geometry_step import (
    PaperKineticLazyNativeFullGeometryStepResult,
)
from paper_kinetic_lazy_program_bundles import (
    PaperKineticLazyProgramBundleProvider,
    prepare_paper_kinetic_lazy_program_bundle_provider,
)
from paper_kinetic_runtime_paths import ensure_worldfoam_lane2_research_path

ensure_worldfoam_lane2_research_path()

from kinetic_compiled_cpu_artifact_store import (  # noqa: E402
    PaperKineticCompiledCpuArtifactStore,
    PaperKineticCompiledCpuArtifactStorePolicy,
)
from kinetic_power_word_compiler import AffineKineticPowerSites  # noqa: E402
from kinetic_sealed_completion_fence import (  # noqa: E402
    PaperKineticCompletionFenceReceipt,
    PaperKineticSealedCompletionFence,
    prepare_paper_kinetic_completion_subject_binding,
)


RECEIPT_PROVENANCE = (
    "paper-kinetic-lazy-full-geometry-device-gradient-receipt-v1"
)
READBACK_SUBJECT_KIND = (
    "paper-kinetic-lazy-full-geometry-gradient-readback-subject-v1"
)
RUNTIME_STATUS = "source_integrated/native_runtime_unverified"
UPDATE_PROVENANCE = (
    "paper-kinetic-lazy-full-geometry-combined-sgd-transaction-v1"
)
READY_PROVENANCE = (
    "paper-kinetic-lazy-full-geometry-ready-generation-v1"
)

_SUBJECT_SEAL = object()
_RECEIPT_SEAL = object()
_UPDATE_SEAL = object()
_READY_SEAL = object()
_TRANSACTION_CONSUMPTION_AUTHORITY = object()


@dataclass
class _FullGeometryReadbackSubject:
    result: PaperKineticLazyNativeFullGeometryStepResult = field(repr=False)
    snapshot: PaperKineticFixedSiteMaterialDeviceSnapshot = field(repr=False)
    subject_identity: int
    result_identity: int
    snapshot_identity: int
    material_bar_signature: tuple[object, ...]
    loss_signature: tuple[object, ...]
    generation_digest: str
    phase: str = "installed"
    _seal: object = field(default=None, repr=False)

    def assert_current(self) -> None:
        if (
            type(self) is not _FullGeometryReadbackSubject
            or self._seal is not _SUBJECT_SEAL
            or id(self) != self.subject_identity
            or id(self.result) != self.result_identity
            or id(self.snapshot) != self.snapshot_identity
            or self.phase not in {"installed", "copied"}
            or _tensor_signature(self.result.grad_global_site_rgba_f32)
            != self.material_bar_signature
            or _tensor_signature(self.result.loss_f32) != self.loss_signature
            or self.generation_digest != _subject_digest(self)
        ):
            raise ValueError("lazy full-geometry readback subject changed")
        self.result.assert_current()


@dataclass
class PaperKineticLazyFullGeometryDeviceGradientReceipt:
    """One-shot CPU authorization for combined material/geometry SGD."""

    source_state_identity: int
    source_state_generation_digest: str
    source_provider_identity: int
    source_artifact_store_identity: int
    step_result_identity: int
    step_result_generation_digest: str
    snapshot_identity: int
    snapshot_generation_digest: str
    provider_generation_digest: str
    world_generation_digest: str
    sites_content_digest: str
    geometry_generation_id: str
    material_generation_id_before: str
    background_generation_id: str
    source_step_index: int
    step_generation_id: str
    reverse_mode: str
    source_device: torch.device
    grad_site_rgba_f32_cpu: torch.Tensor | None = field(repr=False)
    loss_f32_cpu: torch.Tensor | None = field(repr=False)
    grad_positions0_f64_cpu: torch.Tensor | None = field(repr=False)
    grad_velocities_f64_cpu: torch.Tensor | None = field(repr=False)
    grad_weight_coefficients_f64_cpu: torch.Tensor | None = field(repr=False)
    tensor_signatures: tuple[tuple[object, ...], ...] = field(repr=False)
    tensor_content_digests: tuple[str, ...]
    authorization_logical_tensor_bytes: int
    completion_capability_generation_digest: str
    completion_launch_generation_digest: str
    completion_fence_sequence: int
    completion_receipt_generation_digest: str
    authorization_generation_digest: str
    generation_digest: str
    receipt_identity: int
    consumed: bool = False
    revoked: bool = False
    released_after_consumption: bool = False
    promoted_state_generation_digest: str = ""
    update_receipt_generation_digest: str = ""
    consumption_generation_digest: str = ""
    persistent_frame_tensor_bytes: int = 0
    persistent_sample_tensor_bytes: int = 0
    persistent_target_tensor_bytes: int = 0
    persistent_prediction_tensor_bytes: int = 0
    fixed_camera: bool = True
    ray_gradient_tensor_count: int = 0
    native_runtime_verified: bool = False
    allocator_peak_measured: bool = False
    provenance: str = RECEIPT_PROVENANCE
    runtime_status: str = RUNTIME_STATUS
    _snapshot: PaperKineticFixedSiteMaterialDeviceSnapshot | None = field(
        default=None,
        repr=False,
    )
    _result: PaperKineticLazyNativeFullGeometryStepResult | None = field(
        default=None,
        repr=False,
    )
    _seal: object = field(default=None, repr=False)

    @property
    def site_count(self) -> int:
        return int(self.tensor_signatures[0][3][0])

    def _live_tensors(self) -> tuple[torch.Tensor, ...]:
        tensors = (
            self.grad_site_rgba_f32_cpu,
            self.loss_f32_cpu,
            self.grad_positions0_f64_cpu,
            self.grad_velocities_f64_cpu,
            self.grad_weight_coefficients_f64_cpu,
        )
        if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise ValueError("lazy full-geometry receipt lost a CPU tensor")
        return tensors  # type: ignore[return-value]

    def assert_current(
        self,
        state: PaperKineticFixedCameraCombinedState,
        provider: PaperKineticLazyProgramBundleProvider,
        artifact_store: PaperKineticCompiledCpuArtifactStore,
    ) -> None:
        if self.consumed or self.revoked or self.released_after_consumption:
            raise ValueError("lazy full-geometry receipt was already consumed")
        if (
            self._seal is not _RECEIPT_SEAL
            or self.provenance != RECEIPT_PROVENANCE
            or self.runtime_status != RUNTIME_STATUS
            or id(self) != self.receipt_identity
            or id(state) != self.source_state_identity
            or id(provider) != self.source_provider_identity
            or id(artifact_store) != self.source_artifact_store_identity
            or state.generation_digest != self.source_state_generation_digest
            or self.generation_digest != _receipt_digest(self)
            or not self.fixed_camera
            or self.ray_gradient_tensor_count
            or self.native_runtime_verified
            or self.allocator_peak_measured
            or any(
                (
                    self.persistent_frame_tensor_bytes,
                    self.persistent_sample_tensor_bytes,
                    self.persistent_target_tensor_bytes,
                    self.persistent_prediction_tensor_bytes,
                )
            )
        ):
            raise ValueError("lazy full-geometry receipt changed or is foreign")
        state.assert_current(provider, artifact_store)
        result = self._result
        snapshot = self._snapshot
        if type(result) is not PaperKineticLazyNativeFullGeometryStepResult:
            raise ValueError("lazy full-geometry receipt lost its exact result")
        if not isinstance(snapshot, PaperKineticFixedSiteMaterialDeviceSnapshot):
            raise ValueError("lazy full-geometry receipt lost its snapshot")
        result.assert_current()
        snapshot.assert_current(state.material_state)
        tensors = self._live_tensors()
        if (
            id(result) != self.step_result_identity
            or result.generation_digest != self.step_result_generation_digest
            or id(snapshot) != self.snapshot_identity
            or snapshot.generation_digest != self.snapshot_generation_digest
            or result.issued_bridge_receipt_identity != id(self)
            or result.issued_bridge_receipt_generation_digest
            != self.generation_digest
            or snapshot.issued_gradient_receipt_identity != id(self)
            or snapshot.issued_gradient_receipt_generation_digest
            != self.generation_digest
            or provider.generation_digest != self.provider_generation_digest
            or state.world_generation_digest != self.world_generation_digest
            or state.sites_content_digest != self.sites_content_digest
            or state.geometry_generation_id != self.geometry_generation_id
            or state.material_state.material_generation_id
            != self.material_generation_id_before
            or state.geometry_update_count != self.source_step_index
            or result.step_generation_id != self.step_generation_id
            or result.reverse_mode != self.reverse_mode
            or tuple(_tensor_signature(tensor) for tensor in tensors)
            != self.tensor_signatures
            or tuple(_tensor_content_digest(tensor) for tensor in tensors)
            != self.tensor_content_digests
            or _tensor_bytes(*tensors) != self.authorization_logical_tensor_bytes
        ):
            raise ValueError("lazy full-geometry receipt generation changed")
        _validate_cpu_bars(tensors, state.site_count, state.positions0_f64.shape[1], state.weight_coefficients_f64.shape[1])

    def _revoke_after_validated_retirement(self, authority: object) -> int:
        """Drop optimizer roots after the transaction's commit point."""

        if authority is not _TRANSACTION_CONSUMPTION_AUTHORITY:
            raise PermissionError("lazy full-geometry receipt revocation is transaction-only")
        if self.consumed or self.revoked or self.released_after_consumption:
            raise ValueError("lazy full-geometry receipt was already revoked")
        snapshot = self._snapshot
        result = self._result
        if not isinstance(snapshot, PaperKineticFixedSiteMaterialDeviceSnapshot):
            raise ValueError("lazy full-geometry receipt lost its snapshot")
        if type(result) is not PaperKineticLazyNativeFullGeometryStepResult:
            raise ValueError("lazy full-geometry receipt lost its exact result")
        released = self.authorization_logical_tensor_bytes
        if released < 1:
            raise ArithmeticError("lazy full-geometry receipt retained no authorization")

        self.grad_site_rgba_f32_cpu = None
        self.loss_f32_cpu = None
        self.grad_positions0_f64_cpu = None
        self.grad_velocities_f64_cpu = None
        self.grad_weight_coefficients_f64_cpu = None
        result._seal = None
        result.loss_f32 = None  # type: ignore[assignment]
        result.grad_global_site_rgba_f32 = None  # type: ignore[assignment]
        result.grad_positions0_f64_cpu = None  # type: ignore[assignment]
        result.grad_velocities_f64_cpu = None  # type: ignore[assignment]
        result.grad_weight_coefficients_f64_cpu = None  # type: ignore[assignment]
        self._result = None
        snapshot._release_after_consumption()
        self.released_after_consumption = True
        self.revoked = True
        self.assert_revoked()
        return released

    def _commit_promoted_consumption(
        self,
        authority: object,
        *,
        promoted_state_generation_digest: str,
        update_receipt_generation_digest: str,
    ) -> None:
        if authority is not _TRANSACTION_CONSUMPTION_AUTHORITY:
            raise PermissionError("lazy full-geometry receipt consumption is transaction-only")
        self.assert_revoked()
        _require_sha256(
            promoted_state_generation_digest,
            name="promoted_state_generation_digest",
        )
        _require_sha256(
            update_receipt_generation_digest,
            name="update_receipt_generation_digest",
        )
        self.promoted_state_generation_digest = promoted_state_generation_digest
        self.update_receipt_generation_digest = update_receipt_generation_digest
        self.consumed = True
        self.consumption_generation_digest = _consumption_digest(self)
        self.assert_consumed()

    def assert_revoked(self) -> None:
        snapshot = self._snapshot
        if (
            self._seal is not _RECEIPT_SEAL
            or id(self) != self.receipt_identity
            or self.generation_digest != _receipt_digest(self)
            or not self.revoked
            or self.consumed
            or not self.released_after_consumption
            or any(
                tensor is not None
                for tensor in (
                    self.grad_site_rgba_f32_cpu,
                    self.loss_f32_cpu,
                    self.grad_positions0_f64_cpu,
                    self.grad_velocities_f64_cpu,
                    self.grad_weight_coefficients_f64_cpu,
                )
            )
            or self._result is not None
            or not isinstance(snapshot, PaperKineticFixedSiteMaterialDeviceSnapshot)
            or not snapshot.released_after_consumption
            or self.promoted_state_generation_digest
            or self.update_receipt_generation_digest
            or self.consumption_generation_digest
        ):
            raise ValueError("lazy full-geometry receipt revocation changed")

    def assert_consumed(self) -> None:
        snapshot = self._snapshot
        if (
            self._seal is not _RECEIPT_SEAL
            or id(self) != self.receipt_identity
            or self.generation_digest != _receipt_digest(self)
            or not self.revoked
            or not self.consumed
            or not self.released_after_consumption
            or any(
                tensor is not None
                for tensor in (
                    self.grad_site_rgba_f32_cpu,
                    self.loss_f32_cpu,
                    self.grad_positions0_f64_cpu,
                    self.grad_velocities_f64_cpu,
                    self.grad_weight_coefficients_f64_cpu,
                )
            )
            or self._result is not None
            or not isinstance(snapshot, PaperKineticFixedSiteMaterialDeviceSnapshot)
            or not snapshot.released_after_consumption
            or not _is_sha256(self.promoted_state_generation_digest)
            or not _is_sha256(self.update_receipt_generation_digest)
            or self.consumption_generation_digest != _consumption_digest(self)
        ):
            raise ValueError("lazy full-geometry receipt consumption changed")

    def accounting(self) -> dict[str, int | str | bool]:
        return {
            "provenance": self.provenance,
            "runtime_status": self.runtime_status,
            "generation_digest": self.generation_digest,
            "reverse_mode": self.reverse_mode,
            "site_count": self.site_count,
            "source_device": str(self.source_device),
            "device_to_host_copy_phase_count": 1,
            "device_to_host_tensor_count": 2,
            "global_material_bar_bytes": self.site_count * 16,
            "authorization_logical_tensor_bytes": (
                0 if self.released_after_consumption else self.authorization_logical_tensor_bytes
            ),
            "fixed_camera": True,
            "ray_gradient_tensor_count": 0,
            "consumed": self.consumed,
            "revoked": self.revoked,
            "released_after_consumption": self.released_after_consumption,
            "native_runtime_verified": False,
            "allocator_peak_measured": False,
        }


@dataclass(frozen=True)
class _LazyCombinedAuthorization:
    generation_digest: str
    step_generation_id: str
    grad_site_rgba_f32: torch.Tensor = field(repr=False)
    loss_f32: torch.Tensor = field(repr=False)
    grad_positions0_f64: torch.Tensor = field(repr=False)
    grad_velocities_f64: torch.Tensor = field(repr=False)
    grad_weight_coefficients_f64: torch.Tensor = field(repr=False)

    def _tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.grad_site_rgba_f32,
            self.loss_f32,
            self.grad_positions0_f64,
            self.grad_velocities_f64,
            self.grad_weight_coefficients_f64,
        )

    @property
    def logical_tensor_bytes(self) -> int:
        return _tensor_bytes(*self._tensors())


@dataclass(frozen=True)
class PaperKineticLazyFullGeometryCombinedUpdateReceipt:
    """Tensor-free receipt for one lazy combined SGD/cold-recompile commit."""

    step_index: int
    step_generation_id: str
    bridge_receipt_generation_digest: str
    bridge_completion_receipt_generation_digest: str
    bridge_completion_fence_sequence: int
    step_result_generation_digest: str
    authorization_generation_digest: str
    geometry_d2h_receipt_generation_digests: tuple[str, ...]
    reverse_mode: str
    policy_generation_digest: str
    material_generation_id_before: str
    material_generation_id_after: str
    geometry_generation_id_before: str
    geometry_generation_id_after: str
    old_state_generation_digest: str
    new_state_generation_digest: str
    old_provider_generation_digest: str
    new_provider_generation_digest: str
    old_world_generation_digest: str
    new_world_generation_digest: str
    loss: float
    grad_site_rgba_l2_norm: float
    grad_site_rgba_max_abs: float
    grad_positions0_l2_norm: float
    grad_positions0_max_abs: float
    grad_velocities_l2_norm: float
    grad_velocities_max_abs: float
    grad_weight_coefficients_l2_norm: float
    grad_weight_coefficients_max_abs: float
    raw_color_parameter_delta_l2_norm: float
    raw_color_parameter_delta_max_abs: float
    raw_density_parameter_delta_l2_norm: float
    raw_density_parameter_delta_max_abs: float
    positions0_parameter_delta_l2_norm: float
    positions0_parameter_delta_max_abs: float
    velocities_parameter_delta_l2_norm: float
    velocities_parameter_delta_max_abs: float
    weight_coefficients_parameter_delta_l2_norm: float
    weight_coefficients_parameter_delta_max_abs: float
    combined_state_logical_tensor_bytes: int
    update_candidate_logical_tensor_bytes: int
    authorization_logical_tensor_bytes: int
    released_authorization_logical_tensor_bytes: int
    candidate_world_geometry_clone_logical_tensor_bytes: int
    update_validation_scratch_logical_tensor_bytes_upper_bound: int
    old_candidate_authorization_logical_tensor_bytes: int
    old_store_resident_accounted_bytes_before_retirement: int
    fresh_store_resident_accounted_bytes_upper_bound: int
    transaction_tracked_logical_and_store_accounted_bytes_upper_bound: int
    transaction_tracked_policy_bound: int
    stale_provider_store_retirement_count: int
    provider_store_retirement_receipt_chain_sha256: str
    fresh_full_interval_recompile_count: int
    fresh_full_interval_recompile_receipt_generation_digest: str
    cold_compiled_request_count: int
    geometry_d2h_receipt_count: int
    core_accounting: MappingProxyType
    generation_digest: str
    provenance: str = UPDATE_PROVENANCE
    runtime_status: str = RUNTIME_STATUS
    fixed_camera: bool = True
    ray_gradient_tensor_count: int = 0
    native_runtime_verified: bool = False
    allocator_peak_measured: bool = False
    _seal: object = field(default=None, repr=False)

    def assert_current(self) -> None:
        numeric = (
            self.loss,
            self.grad_site_rgba_l2_norm,
            self.grad_site_rgba_max_abs,
            self.grad_positions0_l2_norm,
            self.grad_positions0_max_abs,
            self.grad_velocities_l2_norm,
            self.grad_velocities_max_abs,
            self.grad_weight_coefficients_l2_norm,
            self.grad_weight_coefficients_max_abs,
            self.raw_color_parameter_delta_l2_norm,
            self.raw_color_parameter_delta_max_abs,
            self.raw_density_parameter_delta_l2_norm,
            self.raw_density_parameter_delta_max_abs,
            self.positions0_parameter_delta_l2_norm,
            self.positions0_parameter_delta_max_abs,
            self.velocities_parameter_delta_l2_norm,
            self.velocities_parameter_delta_max_abs,
            self.weight_coefficients_parameter_delta_l2_norm,
            self.weight_coefficients_parameter_delta_max_abs,
        )
        digests = (
            self.bridge_receipt_generation_digest,
            self.bridge_completion_receipt_generation_digest,
            self.step_result_generation_digest,
            self.authorization_generation_digest,
            *self.geometry_d2h_receipt_generation_digests,
            self.policy_generation_digest,
            self.material_generation_id_before,
            self.material_generation_id_after,
            self.geometry_generation_id_before,
            self.geometry_generation_id_after,
            self.old_state_generation_digest,
            self.new_state_generation_digest,
            self.old_provider_generation_digest,
            self.new_provider_generation_digest,
            self.old_world_generation_digest,
            self.new_world_generation_digest,
            self.provider_store_retirement_receipt_chain_sha256,
            self.fresh_full_interval_recompile_receipt_generation_digest,
        )
        if (
            self._seal is not _UPDATE_SEAL
            or self.provenance != UPDATE_PROVENANCE
            or self.runtime_status != RUNTIME_STATUS
            or self.step_index < 1
            or self.bridge_completion_fence_sequence < 1
            or not self.step_generation_id.strip()
            or self.reverse_mode not in {"staged_sparse", "fused_union_v2"}
            or any(not _is_sha256(value) for value in digests)
            or self.material_generation_id_before == self.material_generation_id_after
            or self.geometry_generation_id_before == self.geometry_generation_id_after
            or self.old_state_generation_digest == self.new_state_generation_digest
            or self.old_provider_generation_digest == self.new_provider_generation_digest
            or self.old_world_generation_digest == self.new_world_generation_digest
            or any(not math.isfinite(value) or value < 0.0 for value in numeric)
            or min(
                self.combined_state_logical_tensor_bytes,
                self.update_candidate_logical_tensor_bytes,
                self.authorization_logical_tensor_bytes,
                self.released_authorization_logical_tensor_bytes,
                self.candidate_world_geometry_clone_logical_tensor_bytes,
                self.update_validation_scratch_logical_tensor_bytes_upper_bound,
                self.old_candidate_authorization_logical_tensor_bytes,
                self.fresh_store_resident_accounted_bytes_upper_bound,
                self.transaction_tracked_policy_bound,
            ) < 1
            or self.update_candidate_logical_tensor_bytes
            != self.combined_state_logical_tensor_bytes
            or self.released_authorization_logical_tensor_bytes
            != self.authorization_logical_tensor_bytes
            or self.old_candidate_authorization_logical_tensor_bytes
            != self.combined_state_logical_tensor_bytes * 2
            + self.authorization_logical_tensor_bytes
            or self.old_store_resident_accounted_bytes_before_retirement < 0
            or self.transaction_tracked_logical_and_store_accounted_bytes_upper_bound
            != self.old_candidate_authorization_logical_tensor_bytes
            + self.candidate_world_geometry_clone_logical_tensor_bytes
            + self.update_validation_scratch_logical_tensor_bytes_upper_bound
            + self.old_store_resident_accounted_bytes_before_retirement
            + self.fresh_store_resident_accounted_bytes_upper_bound
            or self.transaction_tracked_logical_and_store_accounted_bytes_upper_bound
            > self.transaction_tracked_policy_bound
            or self.stale_provider_store_retirement_count != 1
            or self.fresh_full_interval_recompile_count != 1
            or self.cold_compiled_request_count < 1
            or self.geometry_d2h_receipt_count
            != len(self.geometry_d2h_receipt_generation_digests)
            or self.geometry_d2h_receipt_count < 1
            or not isinstance(self.core_accounting, MappingProxyType)
            or self.core_accounting.get("reverse_mode") != self.reverse_mode
            or self.core_accounting.get("full_geometry") is not True
            or not self.fixed_camera
            or self.ray_gradient_tensor_count
            or self.native_runtime_verified
            or self.allocator_peak_measured
            or self.generation_digest != _update_digest(self)
        ):
            raise ValueError("lazy combined update receipt changed")

    def accounting(self) -> dict[str, Any]:
        self.assert_current()
        return {
            **{
                name: value
                for name, value in self.__dict__.items()
                if name not in {"_seal", "core_accounting"}
            },
            "geometry_d2h_receipt_generation_digests": list(
                self.geometry_d2h_receipt_generation_digests
            ),
            "core_accounting": dict(self.core_accounting),
            "checkpoint_payload_supported": True,
            # This is current source capability, proven by the sealed restore
            # receipt/ready-generation path in the combined-state module.  Do
            # not infer it from the legacy false-valued fields serialized in a
            # checkpoint payload; those remain schema compatibility metadata.
            "checkpoint_restore_resume_supported": True,
            "checkpoint_restore_resume_api": (
                "restore_paper_kinetic_fixed_camera_combined_generation_from_payload"
            ),
            "checkpoint_restore_requires_fresh_runtime_inputs": True,
        }


@dataclass
class PaperKineticLazyFullGeometryReadyGeneration:
    """Fresh combined generation after receipt revocation and cold compile."""

    state: PaperKineticFixedCameraCombinedState = field(repr=False)
    provider: PaperKineticLazyProgramBundleProvider = field(repr=False)
    artifact_store: PaperKineticCompiledCpuArtifactStore = field(repr=False)
    update_receipt: PaperKineticLazyFullGeometryCombinedUpdateReceipt
    recompile_receipt: PaperKineticFixedCameraColdRecompileReceipt
    manifest: PaperKineticFixedCameraColdRecompileManifest
    generation_digest: str
    provenance: str = READY_PROVENANCE
    runtime_status: str = RUNTIME_STATUS
    _seal: object = field(default=None, repr=False)

    def assert_current(self) -> None:
        if (
            self._seal is not _READY_SEAL
            or self.provenance != READY_PROVENANCE
            or self.runtime_status != RUNTIME_STATUS
        ):
            raise ValueError("lazy combined ready generation changed")
        self.state.assert_current(self.provider, self.artifact_store)
        self.update_receipt.assert_current()
        self.recompile_receipt.assert_current(
            self.provider,
            self.artifact_store,
            self.manifest,
        )
        if (
            self.state.generation_digest
            != self.update_receipt.new_state_generation_digest
            or self.state.cold_recompile_seal_generation_digest
            != self.recompile_receipt.generation_digest
            or self.generation_digest != _ready_digest(self)
        ):
            raise ValueError("lazy combined ready-generation components disagree")


def claim_paper_kinetic_lazy_full_geometry_ready_generation_for_next_step(
    ready: PaperKineticLazyFullGeometryReadyGeneration,
    *,
    caller_retained_untracked_logical_and_accounted_bytes: int,
    device: torch.device | str,
) -> Any:
    """Consume one lazy promotion seal into its next lazy trainer state."""

    if not isinstance(ready, PaperKineticLazyFullGeometryReadyGeneration):
        raise TypeError("lazy next-step claim requires a lazy ready generation")
    retained = caller_retained_untracked_logical_and_accounted_bytes
    if (
        isinstance(retained, bool)
        or not isinstance(retained, int)
        or retained < 0
    ):
        raise ValueError("caller-retained next-step bytes must be nonnegative")
    ready.assert_current()
    if retained:
        raise MemoryError(
            "lazy next-step claim requires zero caller-retained retired-generation "
            "logical or store-accounted bytes"
        )
    from kinetic_lazy_native_material_step import (
        prepare_paper_kinetic_lazy_native_trainer_state,
    )

    trainer_state = prepare_paper_kinetic_lazy_native_trainer_state(
        ready.provider,
        device=device,
        initial_step_index=ready.state.geometry_update_count,
    )
    # Revoke the one-shot wrapper capability. Its explicitly carried live
    # state/provider/store remain the generation used by ``trainer_state``.
    ready._seal = None
    return trainer_state


@torch.no_grad()
def seal_paper_kinetic_lazy_full_geometry_device_gradient_receipt(
    state: PaperKineticFixedCameraCombinedState,
    provider: PaperKineticLazyProgramBundleProvider,
    artifact_store: PaperKineticCompiledCpuArtifactStore,
    snapshot: PaperKineticFixedSiteMaterialDeviceSnapshot,
    result: PaperKineticLazyNativeFullGeometryStepResult,
) -> PaperKineticLazyFullGeometryDeviceGradientReceipt:
    """Fence exactly once and publish one CPU combined-gradient receipt."""

    if type(result) is not PaperKineticLazyNativeFullGeometryStepResult:
        raise TypeError("full-geometry bridge requires its exact lazy result")
    state.assert_current(provider, artifact_store)
    result.assert_current()
    snapshot.assert_current(state.material_state, require_unissued=True)
    material = snapshot.site_rgba_f32_device
    background = snapshot.background_rgb_f32_device
    if not isinstance(material, torch.Tensor) or not isinstance(background, torch.Tensor):
        raise ValueError("full-geometry bridge snapshot lost device tensors")
    result.assert_device_snapshot_tensors(
        material_tensor=material,
        background_tensor=background,
    )
    if (
        result.provider_generation_digest != provider.generation_digest
        or result.world_generation_digest != state.world_generation_digest
        or result.sites_content_digest != state.sites_content_digest
        or result.geometry_generation_id != state.geometry_generation_id
        or result.material_generation_id
        != state.material_state.material_generation_id
        or result.material_generation_id != snapshot.material_generation_id
        or result.background_generation_id != snapshot.background_generation_id
        or result.step_index != state.geometry_update_count
        or result.step_index != snapshot.source_step_index
        or result.grad_global_site_rgba_f32.device != snapshot.device
        or result.loss_f32.device != snapshot.device
        or result.issued_bridge_receipt_identity
    ):
        raise ValueError("lazy full-geometry result is stale or foreign")
    capability = result._sealed_completion_fence
    if (
        type(capability) is not PaperKineticSealedCompletionFence
        or capability.device != snapshot.device
    ):
        raise ValueError("lazy full-geometry result has no bridge authority")

    subject = _FullGeometryReadbackSubject(
        result=result,
        snapshot=snapshot,
        subject_identity=0,
        result_identity=id(result),
        snapshot_identity=id(snapshot),
        material_bar_signature=_tensor_signature(result.grad_global_site_rgba_f32),
        loss_signature=_tensor_signature(result.loss_f32),
        generation_digest="",
        _seal=_SUBJECT_SEAL,
    )
    subject.subject_identity = id(subject)
    subject.generation_digest = _subject_digest(subject)
    subject.assert_current()
    binding = prepare_paper_kinetic_completion_subject_binding(
        capability,
        subject,
        kind=READBACK_SUBJECT_KIND,
        subject_generation_digest=subject.generation_digest,
    )
    launch_digest = _digest_parts(
        RECEIPT_PROVENANCE,
        "combined-optimizer-gradient-readback",
        result.generation_digest,
        snapshot.generation_digest,
        subject.generation_digest,
        binding.generation_digest,
    )
    fence_sequence = capability.next_fence_sequence
    epoch = capability.register_launch(
        stage="combined-optimizer-gradient-readback",
        launch_generation_digest=launch_digest,
        subject_binding=binding,
    )
    grad_cpu = _owned_cpu_f32(result.grad_global_site_rgba_f32)
    loss_cpu = _owned_cpu_f32(result.loss_f32)
    subject.phase = "copied"
    completion = capability.fence(epoch)
    if type(completion) is not PaperKineticCompletionFenceReceipt:
        raise TypeError("full-geometry bridge completion has a foreign type")
    completion.assert_current()
    subject.assert_current()
    if (
        completion.fence_sequence != fence_sequence
        or completion.subject_binding is not binding
    ):
        raise ValueError("full-geometry readback completion relation changed")

    tensors = (
        grad_cpu,
        loss_cpu,
        result.grad_positions0_f64_cpu,
        result.grad_velocities_f64_cpu,
        result.grad_weight_coefficients_f64_cpu,
    )
    _validate_cpu_bars(
        tensors,
        state.site_count,
        state.positions0_f64.shape[1],
        state.weight_coefficients_f64.shape[1],
    )
    content_digests = tuple(_tensor_content_digest(tensor) for tensor in tensors)
    authorization_digest = _digest_parts(
        RECEIPT_PROVENANCE,
        "combined-cpu-optimizer-authorization-v1",
        state.generation_digest,
        result.generation_digest,
        result.step_generation_id,
        content_digests,
    )
    provisional = PaperKineticLazyFullGeometryDeviceGradientReceipt(
        source_state_identity=id(state),
        source_state_generation_digest=state.generation_digest,
        source_provider_identity=id(provider),
        source_artifact_store_identity=id(artifact_store),
        step_result_identity=id(result),
        step_result_generation_digest=result.generation_digest,
        snapshot_identity=id(snapshot),
        snapshot_generation_digest=snapshot.generation_digest,
        provider_generation_digest=provider.generation_digest,
        world_generation_digest=state.world_generation_digest,
        sites_content_digest=state.sites_content_digest,
        geometry_generation_id=state.geometry_generation_id,
        material_generation_id_before=state.material_state.material_generation_id,
        background_generation_id=result.background_generation_id,
        source_step_index=state.geometry_update_count,
        step_generation_id=result.step_generation_id,
        reverse_mode=result.reverse_mode,
        source_device=snapshot.device,
        grad_site_rgba_f32_cpu=grad_cpu,
        loss_f32_cpu=loss_cpu,
        grad_positions0_f64_cpu=result.grad_positions0_f64_cpu,
        grad_velocities_f64_cpu=result.grad_velocities_f64_cpu,
        grad_weight_coefficients_f64_cpu=(
            result.grad_weight_coefficients_f64_cpu
        ),
        tensor_signatures=tuple(_tensor_signature(tensor) for tensor in tensors),
        tensor_content_digests=content_digests,
        authorization_logical_tensor_bytes=_tensor_bytes(*tensors),
        completion_capability_generation_digest=capability.generation_digest,
        completion_launch_generation_digest=launch_digest,
        completion_fence_sequence=fence_sequence,
        completion_receipt_generation_digest=completion.generation_digest,
        authorization_generation_digest=authorization_digest,
        generation_digest="",
        receipt_identity=0,
        _snapshot=snapshot,
        _result=result,
        _seal=_RECEIPT_SEAL,
    )
    provisional.receipt_identity = id(provisional)
    provisional.generation_digest = _receipt_digest(provisional)
    snapshot.issued_gradient_receipt_identity = id(provisional)
    snapshot.issued_gradient_receipt_generation_digest = provisional.generation_digest
    result.claim_bridge_receipt(
        receipt_identity=id(provisional),
        receipt_generation_digest=provisional.generation_digest,
    )
    provisional.assert_current(state, provider, artifact_store)
    completion.consume_for_subject(
        capability,
        binding,
        subject=subject,
        consumer="lazy-full-geometry-device-gradient-readback",
    )
    return provisional


@torch.no_grad()
def apply_paper_kinetic_lazy_full_geometry_combined_sgd_transaction(
    state: PaperKineticFixedCameraCombinedState,
    current_provider: PaperKineticLazyProgramBundleProvider,
    current_artifact_store: PaperKineticCompiledCpuArtifactStore,
    receipt: PaperKineticLazyFullGeometryDeviceGradientReceipt,
    *,
    policy: PaperKineticFixedCameraCombinedSGDPolicy,
    cold_recompile_manifest: PaperKineticFixedCameraColdRecompileManifest,
    fresh_store_policy: PaperKineticCompiledCpuArtifactStorePolicy | None = None,
    runtime_measurements: dict[str, float] | None = None,
) -> PaperKineticLazyFullGeometryReadyGeneration:
    """Apply lazy material+geometry SGD, retire stale state, and cold compile."""

    if not isinstance(state, PaperKineticFixedCameraCombinedState):
        raise TypeError("lazy combined update requires its combined state")
    if not isinstance(receipt, PaperKineticLazyFullGeometryDeviceGradientReceipt):
        raise TypeError("lazy combined update requires its exact bridge receipt")
    if not isinstance(
        cold_recompile_manifest,
        PaperKineticFixedCameraColdRecompileManifest,
    ):
        raise TypeError("lazy combined update requires a cold-recompile manifest")
    if not isinstance(
        current_artifact_store,
        PaperKineticCompiledCpuArtifactStore,
    ):
        raise TypeError("lazy combined update requires its bounded artifact store")
    policy.assert_valid()
    resolved_store_policy = (
        current_artifact_store.policy
        if fresh_store_policy is None
        else fresh_store_policy
    )
    if not isinstance(
        resolved_store_policy,
        PaperKineticCompiledCpuArtifactStorePolicy,
    ):
        raise TypeError("lazy combined update requires a fresh-store policy")
    if runtime_measurements is not None and (
        not isinstance(runtime_measurements, dict) or runtime_measurements
    ):
        raise ValueError("runtime_measurements must be an empty mutable dictionary")

    receipt.assert_current(state, current_provider, current_artifact_store)
    cold_recompile_manifest.assert_compatible(current_provider)
    if (
        state.geometry_update_count
        and state.last_update_policy_generation_digest != policy.generation_digest
    ):
        raise ValueError(
            "lazy combined update policy changed; start a fresh generation"
        )
    result = receipt._result
    if type(result) is not PaperKineticLazyNativeFullGeometryStepResult:
        raise ValueError("lazy combined update lost its exact step result")
    tensors = receipt._live_tensors()
    authorization = _LazyCombinedAuthorization(
        generation_digest=receipt.authorization_generation_digest,
        step_generation_id=receipt.step_generation_id,
        grad_site_rgba_f32=tensors[0],
        loss_f32=tensors[1],
        grad_positions0_f64=tensors[2],
        grad_velocities_f64=tensors[3],
        grad_weight_coefficients_f64=tensors[4],
    )
    old_store_report = current_artifact_store.report()
    memory_preflight = _combined_state._preflight_transaction(
        state,
        authorization,
        policy=policy,
        manifest=cold_recompile_manifest,
        store_policy=resolved_store_policy,
        old_store_resident_accounted_bytes=(
            old_store_report.current_resident_accounted_bytes
        ),
    )
    if authorization.logical_tensor_bytes != receipt.authorization_logical_tensor_bytes:
        raise ArithmeticError("lazy authorization changed its receipted tensor layout")

    old_state_generation = state.generation_digest
    old_material_generation = state.material_state.material_generation_id
    old_geometry_generation = state.geometry_generation_id
    old_provider_generation = current_provider.generation_digest
    old_world_generation = current_provider.world.generation_digest
    geometry_d2h_digests = tuple(
        item.generation_digest for item in result.geometry_d2h_receipts
    )
    core_accounting = MappingProxyType(dict(result.accounting))
    loss_value = float(tensors[1].item())
    bar_stats = tuple(_tensor_stats(tensor) for tensor in authorization._tensors()[::])

    candidates = _combined_state._build_update_candidates(
        state,
        authorization,
        policy=policy,
    )
    if (
        candidates.logical_tensor_bytes
        != memory_preflight.update_candidate_logical_tensor_bytes
    ):
        raise ArithmeticError("lazy combined candidate changed its preflighted layout")
    parameter_delta_stats = (
        _tensor_stats(candidates.raw_color_f32 - state.material_state.raw_color_f32),
        _tensor_stats(candidates.raw_density_f32 - state.material_state.raw_density_f32),
        _tensor_stats(candidates.positions0_f64 - state.positions0_f64),
        _tensor_stats(candidates.velocities_f64 - state.velocities_f64),
        _tensor_stats(
            candidates.weight_coefficients_f64 - state.weight_coefficients_f64
        ),
    )
    initializer_generation_digest = _digest_parts(
        UPDATE_PROVENANCE,
        "owned-candidate-world-initializer",
        state.geometry_generation_id,
        receipt.authorization_generation_digest,
        receipt.step_generation_id,
        _tensor_content_digest(candidates.positions0_f64),
        _tensor_content_digest(candidates.velocities_f64),
        _tensor_content_digest(candidates.weight_coefficients_f64),
    )
    initializer = _combined_state._OwnedCandidateWorldInitializer(
        sites=AffineKineticPowerSites(
            positions0=candidates.positions0_f64,
            velocities=candidates.velocities_f64,
            weight_coefficients=candidates.weight_coefficients_f64,
        ),
        generation_digest=initializer_generation_digest,
    )
    fresh_provider = prepare_paper_kinetic_lazy_program_bundle_provider(
        dataset_generation_digest=current_provider.dataset_generation_digest,
        target_provider=current_provider.target_provider,
        ray_provider=current_provider.ray_provider,
        frame_times=current_provider.frame_times,
        height=current_provider.height,
        width=current_provider.width,
        maximum_tracks_per_bundle=current_provider.maximum_tracks_per_bundle,
        maximum_observations_per_bundle=(
            current_provider.maximum_observations_per_bundle
        ),
        maximum_rows_per_native_block=current_provider.maximum_rows_per_native_block,
        world_initializer=initializer,
        program_factory=current_provider.program_factory,
    )
    if not initializer.consumed:
        raise ArithmeticError("fresh provider did not consume lazy candidate geometry")
    if (
        _tensor_bytes(
            fresh_provider.world.sites.positions0,
            fresh_provider.world.sites.velocities,
            fresh_provider.world.sites.weight_coefficients,
        )
        != memory_preflight.candidate_world_geometry_clone_logical_tensor_bytes
    ):
        raise ArithmeticError("lazy candidate world clone changed its tensor layout")
    cold_recompile_manifest.assert_compatible(fresh_provider)
    geometry_generation_after = (
        _combined_state.paper_kinetic_fixed_camera_provider_geometry_generation_id(
            fresh_provider
        )
    )
    fresh_material_state = _combined_state._build_fresh_material_state(
        state.material_state,
        fresh_provider,
        authorization,
        candidates,
    )
    fresh_store = PaperKineticCompiledCpuArtifactStore(resolved_store_policy)

    stage = "retire_old_generation"
    new_state: PaperKineticFixedCameraCombinedState | None = None
    try:
        _combined_state._retire_combined_generation(
            state,
            current_provider,
            current_artifact_store,
        )
        retired_store_report = current_artifact_store.report()
        retirement_chain = _digest_parts(
            UPDATE_PROVENANCE,
            "stale-provider-store-retirement",
            old_state_generation,
            old_provider_generation,
            old_world_generation,
            old_store_report.current_entry_count,
            old_store_report.current_resident_accounted_bytes,
            retired_store_report.current_entry_count,
            retired_store_report.current_resident_accounted_bytes,
        )
        stage = "revoke_lazy_full_geometry_receipt"
        released_authorization_bytes = (
            receipt._revoke_after_validated_retirement(
                _TRANSACTION_CONSUMPTION_AUTHORITY
            )
        )
        del authorization, tensors, result
        stage = "cold_recompile"
        cold_compile_started = time.perf_counter()
        recompile_receipt = _combined_state._cold_recompile_and_seal(
            fresh_provider,
            fresh_store,
            cold_recompile_manifest,
            maximum_artifact_accounted_bytes=policy.maximum_artifact_accounted_bytes,
        )
        cold_compile_wall_time_seconds = time.perf_counter() - cold_compile_started
        if (
            not math.isfinite(cold_compile_wall_time_seconds)
            or cold_compile_wall_time_seconds < 0.0
        ):
            raise ArithmeticError("cold-compile wall-time measurement is invalid")
        if runtime_measurements is not None:
            runtime_measurements["cold_cpu_compile_wall_time_seconds"] = (
                cold_compile_wall_time_seconds
            )
        new_state = PaperKineticFixedCameraCombinedState(
            material_state=fresh_material_state,
            positions0_f64=fresh_provider.world.sites.positions0,
            velocities_f64=fresh_provider.world.sites.velocities,
            weight_coefficients_f64=(
                fresh_provider.world.sites.weight_coefficients
            ),
            provider_generation_digest=fresh_provider.generation_digest,
            world_generation_digest=fresh_provider.world.generation_digest,
            sites_content_digest=fresh_provider.world.sites_content_digest,
            geometry_generation_parent_digest=old_geometry_generation,
            geometry_generation_id=geometry_generation_after,
            last_authorization_generation_digest=(
                receipt.authorization_generation_digest
            ),
            last_step_generation_id=receipt.step_generation_id,
            last_update_policy_generation_digest=policy.generation_digest,
            geometry_update_count=state.geometry_update_count + 1,
            cold_recompile_seal_generation_digest=(
                recompile_receipt.generation_digest
            ),
            tensor_signatures=tuple(
                _combined_state._tensor_signature(tensor)
                for tensor in (
                    fresh_provider.world.sites.positions0,
                    fresh_provider.world.sites.velocities,
                    fresh_provider.world.sites.weight_coefficients,
                )
            ),
            generation_digest="",
            active=True,
            retired=False,
            poisoned=False,
            _provider_identity=id(fresh_provider),
            _artifact_store_identity=id(fresh_store),
            _seal=_combined_state._STATE_SEAL,
        )
        new_state.generation_digest = _combined_state._combined_state_digest(new_state)
        new_state.assert_current(fresh_provider, fresh_store)

        provisional_update = PaperKineticLazyFullGeometryCombinedUpdateReceipt(
            step_index=new_state.geometry_update_count,
            step_generation_id=receipt.step_generation_id,
            bridge_receipt_generation_digest=receipt.generation_digest,
            bridge_completion_receipt_generation_digest=(
                receipt.completion_receipt_generation_digest
            ),
            bridge_completion_fence_sequence=receipt.completion_fence_sequence,
            step_result_generation_digest=receipt.step_result_generation_digest,
            authorization_generation_digest=receipt.authorization_generation_digest,
            geometry_d2h_receipt_generation_digests=geometry_d2h_digests,
            reverse_mode=receipt.reverse_mode,
            policy_generation_digest=policy.generation_digest,
            material_generation_id_before=old_material_generation,
            material_generation_id_after=(
                new_state.material_state.material_generation_id
            ),
            geometry_generation_id_before=old_geometry_generation,
            geometry_generation_id_after=new_state.geometry_generation_id,
            old_state_generation_digest=old_state_generation,
            new_state_generation_digest=new_state.generation_digest,
            old_provider_generation_digest=old_provider_generation,
            new_provider_generation_digest=new_state.provider_generation_digest,
            old_world_generation_digest=old_world_generation,
            new_world_generation_digest=new_state.world_generation_digest,
            loss=loss_value,
            grad_site_rgba_l2_norm=bar_stats[0][0],
            grad_site_rgba_max_abs=bar_stats[0][1],
            grad_positions0_l2_norm=bar_stats[2][0],
            grad_positions0_max_abs=bar_stats[2][1],
            grad_velocities_l2_norm=bar_stats[3][0],
            grad_velocities_max_abs=bar_stats[3][1],
            grad_weight_coefficients_l2_norm=bar_stats[4][0],
            grad_weight_coefficients_max_abs=bar_stats[4][1],
            raw_color_parameter_delta_l2_norm=parameter_delta_stats[0][0],
            raw_color_parameter_delta_max_abs=parameter_delta_stats[0][1],
            raw_density_parameter_delta_l2_norm=parameter_delta_stats[1][0],
            raw_density_parameter_delta_max_abs=parameter_delta_stats[1][1],
            positions0_parameter_delta_l2_norm=parameter_delta_stats[2][0],
            positions0_parameter_delta_max_abs=parameter_delta_stats[2][1],
            velocities_parameter_delta_l2_norm=parameter_delta_stats[3][0],
            velocities_parameter_delta_max_abs=parameter_delta_stats[3][1],
            weight_coefficients_parameter_delta_l2_norm=(
                parameter_delta_stats[4][0]
            ),
            weight_coefficients_parameter_delta_max_abs=(
                parameter_delta_stats[4][1]
            ),
            combined_state_logical_tensor_bytes=(
                memory_preflight.combined_state_logical_tensor_bytes
            ),
            update_candidate_logical_tensor_bytes=(
                memory_preflight.update_candidate_logical_tensor_bytes
            ),
            authorization_logical_tensor_bytes=(
                memory_preflight.authorization_logical_tensor_bytes
            ),
            released_authorization_logical_tensor_bytes=(
                released_authorization_bytes
            ),
            candidate_world_geometry_clone_logical_tensor_bytes=(
                memory_preflight.candidate_world_geometry_clone_logical_tensor_bytes
            ),
            update_validation_scratch_logical_tensor_bytes_upper_bound=(
                memory_preflight
                .update_validation_scratch_logical_tensor_bytes_upper_bound
            ),
            old_candidate_authorization_logical_tensor_bytes=(
                memory_preflight.old_candidate_authorization_logical_tensor_bytes
            ),
            old_store_resident_accounted_bytes_before_retirement=(
                memory_preflight.old_store_resident_accounted_bytes
            ),
            fresh_store_resident_accounted_bytes_upper_bound=(
                memory_preflight.fresh_store_resident_accounted_bytes_upper_bound
            ),
            transaction_tracked_logical_and_store_accounted_bytes_upper_bound=(
                memory_preflight
                .transaction_tracked_logical_and_store_accounted_bytes_upper_bound
            ),
            transaction_tracked_policy_bound=(
                policy.maximum_transaction_tracked_logical_and_store_accounted_bytes
            ),
            stale_provider_store_retirement_count=1,
            provider_store_retirement_receipt_chain_sha256=retirement_chain,
            fresh_full_interval_recompile_count=1,
            fresh_full_interval_recompile_receipt_generation_digest=(
                recompile_receipt.generation_digest
            ),
            cold_compiled_request_count=recompile_receipt.cold_compile_count,
            geometry_d2h_receipt_count=len(geometry_d2h_digests),
            core_accounting=core_accounting,
            generation_digest="",
            _seal=_UPDATE_SEAL,
        )
        update_receipt = PaperKineticLazyFullGeometryCombinedUpdateReceipt(
            **{
                **provisional_update.__dict__,
                "generation_digest": _update_digest(provisional_update),
            }
        )
        update_receipt.assert_current()
        receipt._commit_promoted_consumption(
            _TRANSACTION_CONSUMPTION_AUTHORITY,
            promoted_state_generation_digest=new_state.generation_digest,
            update_receipt_generation_digest=update_receipt.generation_digest,
        )
        provisional_ready = PaperKineticLazyFullGeometryReadyGeneration(
            state=new_state,
            provider=fresh_provider,
            artifact_store=fresh_store,
            update_receipt=update_receipt,
            recompile_receipt=recompile_receipt,
            manifest=cold_recompile_manifest,
            generation_digest="",
            _seal=_READY_SEAL,
        )
        provisional_ready.generation_digest = _ready_digest(provisional_ready)
        provisional_ready.assert_current()
        return provisional_ready
    except BaseException as error:
        cleanup_errors = _combined_state._invalidate_candidate_generation(
            fresh_material_state,
            fresh_provider,
            fresh_store,
            combined_state=new_state,
        )
        failure = _combined_state.PaperKineticFixedCameraCombinedTransactionFailure(
            stage,
            error,
        )
        for cleanup_error in cleanup_errors:
            failure.add_note(cleanup_error)
        raise failure from error


def _subject_digest(subject: _FullGeometryReadbackSubject) -> str:
    return _digest_parts(
        READBACK_SUBJECT_KIND,
        subject.subject_identity,
        subject.result_identity,
        subject.snapshot_identity,
        subject.result.generation_digest,
        subject.snapshot.generation_digest,
        subject.material_bar_signature,
        subject.loss_signature,
    )


def _receipt_digest(
    receipt: PaperKineticLazyFullGeometryDeviceGradientReceipt,
) -> str:
    return _digest_parts(
        RECEIPT_PROVENANCE,
        receipt.receipt_identity,
        receipt.source_state_identity,
        receipt.source_state_generation_digest,
        receipt.source_provider_identity,
        receipt.source_artifact_store_identity,
        receipt.step_result_identity,
        receipt.step_result_generation_digest,
        receipt.snapshot_identity,
        receipt.snapshot_generation_digest,
        receipt.provider_generation_digest,
        receipt.world_generation_digest,
        receipt.sites_content_digest,
        receipt.geometry_generation_id,
        receipt.material_generation_id_before,
        receipt.background_generation_id,
        receipt.source_step_index,
        receipt.step_generation_id,
        receipt.reverse_mode,
        str(receipt.source_device),
        receipt.tensor_content_digests,
        receipt.authorization_logical_tensor_bytes,
        receipt.completion_capability_generation_digest,
        receipt.completion_launch_generation_digest,
        receipt.completion_fence_sequence,
        receipt.completion_receipt_generation_digest,
        receipt.authorization_generation_digest,
    )


def _consumption_digest(
    receipt: PaperKineticLazyFullGeometryDeviceGradientReceipt,
) -> str:
    return _digest_parts(
        RECEIPT_PROVENANCE,
        "revoked-and-consumed-for-promoted-combined-generation",
        receipt.generation_digest,
        receipt.promoted_state_generation_digest,
        receipt.update_receipt_generation_digest,
        receipt.released_after_consumption,
        receipt.revoked,
        receipt.consumed,
    )


def _update_digest(
    receipt: PaperKineticLazyFullGeometryCombinedUpdateReceipt,
) -> str:
    return _digest_parts(
        UPDATE_PROVENANCE,
        tuple(
            (name, value)
            for name, value in receipt.__dict__.items()
            if name not in {"generation_digest", "core_accounting", "_seal"}
        ),
        tuple(sorted(receipt.core_accounting.items())),
    )


def _ready_digest(ready: PaperKineticLazyFullGeometryReadyGeneration) -> str:
    return _digest_parts(
        READY_PROVENANCE,
        ready.state.generation_digest,
        ready.provider.generation_digest,
        ready.update_receipt.generation_digest,
        ready.recompile_receipt.generation_digest,
        ready.manifest.generation_digest,
    )


def _tensor_stats(tensor: torch.Tensor) -> tuple[float, float]:
    flat = tensor.detach().reshape(-1)
    if flat.numel() < 1:
        raise ValueError("tensor statistics require a nonempty tensor")
    return (
        float(torch.linalg.vector_norm(flat).item()),
        float(flat.abs().max().item()),
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, *, name: str) -> None:
    if not _is_sha256(value):
        raise ValueError(f"{name} must be a SHA-256 digest")


def _owned_cpu_f32(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.float32, copy=True).contiguous()


def _validate_cpu_bars(
    tensors: tuple[torch.Tensor, ...],
    site_count: int,
    spatial_dimension: int,
    weight_coefficient_count: int,
) -> None:
    expected = (
        (torch.float32, (site_count, 4)),
        (torch.float32, (1,)),
        (torch.float64, (site_count, spatial_dimension)),
        (torch.float64, (site_count, spatial_dimension)),
        (torch.float64, (site_count, weight_coefficient_count)),
    )
    for index, (tensor, (dtype, shape)) in enumerate(zip(tensors, expected, strict=True)):
        if (
            tensor.device.type != "cpu"
            or tensor.dtype != dtype
            or tuple(tensor.shape) != shape
            or not tensor.is_contiguous()
            or tensor.requires_grad
            or not bool(torch.isfinite(tensor).all().item())
        ):
            raise ValueError(f"lazy full-geometry CPU bar {index} is invalid")


def _tensor_signature(tensor: torch.Tensor) -> tuple[object, ...]:
    return (
        id(tensor),
        str(tensor.device),
        str(tensor.dtype),
        tuple(int(value) for value in tensor.shape),
        tuple(int(value) for value in tensor.stride()),
        int(tensor.storage_offset()),
        int(tensor.untyped_storage().data_ptr()),
        int(tensor._version),
    )


def _tensor_content_digest(tensor: torch.Tensor) -> str:
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        raise ValueError("content digest requires a contiguous CPU tensor")
    return hashlib.sha256(tensor.numpy().tobytes(order="C")).hexdigest()


def _tensor_bytes(*tensors: torch.Tensor) -> int:
    return sum(int(tensor.numel()) * int(tensor.element_size()) for tensor in tensors)


def _digest_parts(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = repr(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


__all__ = (
    "PaperKineticLazyFullGeometryCombinedUpdateReceipt",
    "PaperKineticLazyFullGeometryDeviceGradientReceipt",
    "PaperKineticLazyFullGeometryReadyGeneration",
    "apply_paper_kinetic_lazy_full_geometry_combined_sgd_transaction",
    "claim_paper_kinetic_lazy_full_geometry_ready_generation_for_next_step",
    "seal_paper_kinetic_lazy_full_geometry_device_gradient_receipt",
)
