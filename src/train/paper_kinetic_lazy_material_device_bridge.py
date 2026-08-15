"""Exact lazy-native WorldFoam device-bar to CPU optimizer bridge.

The block-outer lazy coordinator returns one frame-invariant ``[S, 4]``
material bar and one scalar loss on its execution device.  This module copies
those two tensors exactly once, under the coordinator's sealed completion
capability, and consumes them through the canonical CPU fixed-site manual-SGD
path.  It is deliberately distinct from the dense fixed-site step bridge: a
lazy result is not a ``PaperKineticFixedSiteMaterialStepResult`` and cannot be
made one by duck typing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import torch
from paper_kinetic_fixed_site_material_device_bridge import (
    PaperKineticFixedSiteMaterialDeviceSnapshot,
)
from paper_kinetic_fixed_site_material_state import (
    PaperKineticFixedSiteMaterialState,
    PaperKineticFixedSiteMaterialStepReceipt,
    _apply_validated_paper_kinetic_fixed_site_material_bars,
)
from paper_kinetic_runtime_paths import ensure_worldfoam_lane2_research_path

ensure_worldfoam_lane2_research_path()

from kinetic_lazy_native_material_step import (  # noqa: E402
    PaperKineticLazyNativeMaterialStepResult,
)
from kinetic_sealed_completion_fence import (  # noqa: E402
    PaperKineticCompletionFenceReceipt,
    PaperKineticCompletionSubjectBinding,
    PaperKineticSealedCompletionFence,
    prepare_paper_kinetic_completion_subject_binding,
)


RECEIPT_PROVENANCE = "paper-kinetic-lazy-material-device-gradient-receipt-v1"
READBACK_SUBJECT_KIND = "paper-kinetic-lazy-gradient-readback-subject-v1"

_READBACK_SUBJECT_SEAL = object()
_RECEIPT_SEAL = object()


@dataclass
class _LazyGradientReadbackSubject:
    """Stable prelaunch root set for one exact device-to-host copy."""

    result: PaperKineticLazyNativeMaterialStepResult = field(repr=False)
    snapshot: PaperKineticFixedSiteMaterialDeviceSnapshot = field(repr=False)
    grad_device: torch.Tensor = field(repr=False)
    loss_device: torch.Tensor = field(repr=False)
    subject_identity: int
    generation_digest: str
    result_identity: int
    snapshot_identity: int
    grad_signature: tuple[object, ...]
    loss_signature: tuple[object, ...]
    grad_cpu: torch.Tensor | None = field(default=None, repr=False)
    loss_cpu: torch.Tensor | None = field(default=None, repr=False)
    cpu_signatures: tuple[tuple[object, ...], ...] = field(
        default=(),
        repr=False,
    )
    phase: str = "installed"
    _seal: object = field(default=None, repr=False)

    def assert_current(self) -> None:
        if (
            type(self) is not _LazyGradientReadbackSubject
            or self._seal is not _READBACK_SUBJECT_SEAL
            or id(self) != self.subject_identity
            or id(self.result) != self.result_identity
            or id(self.snapshot) != self.snapshot_identity
            or _tensor_signature(self.grad_device) != self.grad_signature
            or _tensor_signature(self.loss_device) != self.loss_signature
            or self.phase not in {"installed", "copied"}
            or self.generation_digest != _readback_subject_digest(self)
        ):
            raise ValueError("lazy gradient readback subject changed")
        self.result.assert_current()
        copied = self.phase == "copied"
        if copied != isinstance(self.grad_cpu, torch.Tensor) or copied != isinstance(
            self.loss_cpu,
            torch.Tensor,
        ):
            raise ValueError("lazy gradient readback CPU publication changed")
        if copied and tuple(
            _tensor_signature(tensor) for tensor in (self.grad_cpu, self.loss_cpu)
        ) != self.cpu_signatures:
            raise ValueError("lazy gradient readback CPU tensors changed")

    def publish_cpu(self, grad_cpu: torch.Tensor, loss_cpu: torch.Tensor) -> None:
        self.assert_current()
        if self.phase != "installed":
            raise ValueError("lazy gradient readback was already published")
        self.grad_cpu = grad_cpu
        self.loss_cpu = loss_cpu
        self.cpu_signatures = tuple(
            _tensor_signature(tensor) for tensor in (grad_cpu, loss_cpu)
        )
        self.phase = "copied"
        self.assert_current()


@dataclass
class PaperKineticLazyMaterialDeviceGradientReceipt:
    """One consumed fence plus owned CPU bar/loss awaiting one SGD apply."""

    step_result_identity: int
    step_result_generation_digest: str
    snapshot_identity: int
    snapshot_generation_digest: str
    source_state_identity: int
    world_generation_digest: str
    sites_content_digest: str
    material_generation_id_before: str
    background_generation_id: str
    source_step_index: int
    step_generation_id: str
    source_device: torch.device
    grad_site_rgba_f32_cpu: torch.Tensor | None = field(repr=False)
    loss_f32_cpu: torch.Tensor | None = field(repr=False)
    grad_content_digest: str
    loss_content_digest: str
    tensor_signatures: tuple[tuple[object, ...], ...] = field(repr=False)
    completion_capability_generation_digest: str
    completion_launch_generation_digest: str
    completion_fence_sequence: int
    completion_receipt_generation_digest: str
    authorization_generation_digest: str
    generation_digest: str
    receipt_identity: int
    material_generation_id_after: str = ""
    step_receipt_generation_digest: str = ""
    consumption_generation_digest: str = ""
    consumed: bool = False
    released_after_consumption: bool = False
    persistent_frame_tensor_bytes: int = 0
    persistent_sample_tensor_bytes: int = 0
    persistent_target_tensor_bytes: int = 0
    persistent_prediction_tensor_bytes: int = 0
    provenance: str = RECEIPT_PROVENANCE
    _snapshot: PaperKineticFixedSiteMaterialDeviceSnapshot | None = field(
        default=None,
        repr=False,
    )
    _result: PaperKineticLazyNativeMaterialStepResult | None = field(
        default=None,
        repr=False,
    )
    _seal: object = field(default=None, repr=False)

    @property
    def site_count(self) -> int:
        return int(self.tensor_signatures[0][3][0])

    @property
    def live_cpu_receipt_tensor_bytes(self) -> int:
        return 0 if self.released_after_consumption else self.site_count * 16 + 4

    def accounting(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance,
            "generation_digest": self.generation_digest,
            "step_result_generation_digest": self.step_result_generation_digest,
            "snapshot_generation_digest": self.snapshot_generation_digest,
            "site_count": self.site_count,
            "source_device": str(self.source_device),
            "completion_fence_sequence": self.completion_fence_sequence,
            "device_to_host_copy_phase_count": 1,
            "device_to_host_tensor_count": 2,
            "global_material_bar_shape": [self.site_count, 4],
            "global_material_bar_bytes": self.site_count * 16,
            "live_cpu_receipt_tensor_bytes": self.live_cpu_receipt_tensor_bytes,
            "consumed": self.consumed,
            "released_after_consumption": self.released_after_consumption,
            "persistent_frame_tensor_bytes": 0,
            "persistent_sample_tensor_bytes": 0,
            "persistent_target_tensor_bytes": 0,
            "persistent_prediction_tensor_bytes": 0,
        }

    def assert_current(
        self,
        state: PaperKineticFixedSiteMaterialState,
        *,
        require_unconsumed: bool = True,
    ) -> None:
        if not isinstance(state, PaperKineticFixedSiteMaterialState):
            raise TypeError("lazy material receipt requires fixed-site state")
        snapshot = self._snapshot
        if (
            self._seal is not _RECEIPT_SEAL
            or self.provenance != RECEIPT_PROVENANCE
            or id(self) != self.receipt_identity
            or not isinstance(snapshot, PaperKineticFixedSiteMaterialDeviceSnapshot)
            or id(snapshot) != self.snapshot_identity
            or snapshot.generation_digest != self.snapshot_generation_digest
            or snapshot.issued_gradient_receipt_identity != id(self)
            or snapshot.issued_gradient_receipt_generation_digest
            != self.generation_digest
            or id(state) != self.source_state_identity
            or self.generation_digest != _receipt_digest(self)
            or self.persistent_frame_tensor_bytes != 0
            or self.persistent_sample_tensor_bytes != 0
            or self.persistent_target_tensor_bytes != 0
            or self.persistent_prediction_tensor_bytes != 0
        ):
            raise ValueError("lazy material device receipt changed or is foreign")
        for name, value in (
            ("world_generation_digest", self.world_generation_digest),
            ("sites_content_digest", self.sites_content_digest),
            ("material_generation_id_before", self.material_generation_id_before),
            ("step_result_generation_digest", self.step_result_generation_digest),
            ("completion_capability_generation_digest", self.completion_capability_generation_digest),
            ("completion_launch_generation_digest", self.completion_launch_generation_digest),
            ("completion_receipt_generation_digest", self.completion_receipt_generation_digest),
            ("authorization_generation_digest", self.authorization_generation_digest),
            ("grad_content_digest", self.grad_content_digest),
            ("loss_content_digest", self.loss_content_digest),
            ("generation_digest", self.generation_digest),
        ):
            _require_sha256(value, name=name)
        if self.consumed:
            if require_unconsumed:
                raise ValueError("lazy material device receipt was already consumed")
            if (
                not self.released_after_consumption
                or self.grad_site_rgba_f32_cpu is not None
                or self.loss_f32_cpu is not None
                or self._result is not None
                or not snapshot.released_after_consumption
                or state.generation_parent_digest
                != self.material_generation_id_before
                or state.material_generation_id != self.material_generation_id_after
                or state.step_index != self.source_step_index + 1
                or state.last_step_generation_id != self.step_generation_id
                or state.last_authorization_generation_digest
                != self.authorization_generation_digest
                or not self.step_receipt_generation_digest
                or self.consumption_generation_digest
                != _consumption_digest(self)
            ):
                raise ValueError("consumed lazy material device receipt changed")
            state.assert_current()
            return
        if self.released_after_consumption or self.material_generation_id_after:
            raise ValueError("unconsumed lazy material receipt has commit metadata")
        result = self._result
        if not isinstance(result, PaperKineticLazyNativeMaterialStepResult):
            raise ValueError("unconsumed lazy material receipt lost its result")
        result.assert_current()
        snapshot.assert_current(state)
        if (
            id(result) != self.step_result_identity
            or result.generation_digest != self.step_result_generation_digest
            or result.issued_bridge_receipt_identity != id(self)
            or result.issued_bridge_receipt_generation_digest
            != self.generation_digest
            or state.world_generation_digest != self.world_generation_digest
            or state.sites_content_digest != self.sites_content_digest
            or state.material_generation_id != self.material_generation_id_before
            or state.step_index != self.source_step_index
        ):
            raise ValueError("lazy material receipt generation is stale")
        tensors = (self.grad_site_rgba_f32_cpu, self.loss_f32_cpu)
        if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise ValueError("lazy material receipt lost its CPU tensors")
        gradient, loss = tensors
        _require_cpu_f32(gradient, shape=(self.site_count, 4), name="lazy CPU bar")
        _require_cpu_f32(loss, shape=(1,), name="lazy CPU loss")
        if tuple(_tensor_signature(tensor) for tensor in tensors) != self.tensor_signatures:
            raise ValueError("lazy material receipt CPU tensor identity changed")
        if (
            _tensor_content_digest(gradient) != self.grad_content_digest
            or _tensor_content_digest(loss) != self.loss_content_digest
            or not bool(torch.isfinite(gradient).all().item())
            or not bool(torch.isfinite(loss).all().item())
        ):
            raise ValueError("lazy material receipt CPU numeric content changed")


@torch.no_grad()
def seal_paper_kinetic_lazy_material_device_gradient_receipt(
    state: PaperKineticFixedSiteMaterialState,
    snapshot: PaperKineticFixedSiteMaterialDeviceSnapshot,
    result: PaperKineticLazyNativeMaterialStepResult,
) -> PaperKineticLazyMaterialDeviceGradientReceipt:
    """Fence and copy the exact lazy result into one CPU-owned receipt."""

    if not isinstance(result, PaperKineticLazyNativeMaterialStepResult):
        raise TypeError("lazy material bridge requires its exact step result")
    result.assert_current()
    snapshot.assert_current(state, require_unissued=True)
    material = snapshot.site_rgba_f32_device
    background = snapshot.background_rgb_f32_device
    if not isinstance(material, torch.Tensor) or not isinstance(background, torch.Tensor):
        raise ValueError("lazy material bridge snapshot lost device tensors")
    result.assert_device_snapshot_tensors(
        material_tensor=material,
        background_tensor=background,
    )
    if (
        result.world_generation_digest != snapshot.world_generation_digest
        or result.sites_content_digest != snapshot.sites_content_digest
        or result.material_generation_id != snapshot.material_generation_id
        or result.background_generation_id != snapshot.background_generation_id
        or result.step_index != snapshot.source_step_index
        or result.grad_global_site_rgba_f32.device != snapshot.device
        or result.loss_f32.device != snapshot.device
    ):
        raise ValueError("lazy step result is stale or foreign to the snapshot")
    capability = result._sealed_completion_fence
    if (
        type(capability) is not PaperKineticSealedCompletionFence
        or capability.device != snapshot.device
        or result.issued_bridge_receipt_identity
    ):
        raise ValueError("lazy step result has no current bridge authority")
    _require_f32_device(
        result.grad_global_site_rgba_f32,
        shape=(snapshot.site_count, 4),
        device=snapshot.device,
        name="lazy device bar",
    )
    _require_f32_device(
        result.loss_f32,
        shape=(1,),
        device=snapshot.device,
        name="lazy device loss",
    )

    subject = _LazyGradientReadbackSubject(
        result=result,
        snapshot=snapshot,
        grad_device=result.grad_global_site_rgba_f32,
        loss_device=result.loss_f32,
        subject_identity=0,
        generation_digest="",
        result_identity=id(result),
        snapshot_identity=id(snapshot),
        grad_signature=_tensor_signature(result.grad_global_site_rgba_f32),
        loss_signature=_tensor_signature(result.loss_f32),
        _seal=_READBACK_SUBJECT_SEAL,
    )
    subject.subject_identity = id(subject)
    subject.generation_digest = _readback_subject_digest(subject)
    subject.assert_current()
    binding = prepare_paper_kinetic_completion_subject_binding(
        capability,
        subject,
        kind=READBACK_SUBJECT_KIND,
        subject_generation_digest=subject.generation_digest,
    )
    launch_digest = _digest_parts(
        RECEIPT_PROVENANCE,
        "optimizer-gradient-readback",
        result.generation_digest,
        snapshot.generation_digest,
        subject.generation_digest,
        binding.generation_digest,
    )
    fence_sequence = capability.next_fence_sequence
    epoch = capability.register_launch(
        stage="optimizer-gradient-readback",
        launch_generation_digest=launch_digest,
        subject_binding=binding,
    )
    grad_cpu = _owned_cpu_f32(result.grad_global_site_rgba_f32)
    loss_cpu = _owned_cpu_f32(result.loss_f32)
    subject.publish_cpu(grad_cpu, loss_cpu)
    completion_receipt = capability.fence(epoch)
    subject.assert_current()
    if (
        completion_receipt.fence_sequence != fence_sequence
        or completion_receipt.subject_binding is not binding
    ):
        raise ValueError("lazy gradient readback completion relation changed")
    if not bool(torch.isfinite(grad_cpu).all().item()) or not bool(
        torch.isfinite(loss_cpu).all().item()
    ):
        raise FloatingPointError("lazy material device bar/loss is nonfinite")
    tensors = (grad_cpu, loss_cpu)
    grad_content_digest = _tensor_content_digest(grad_cpu)
    loss_content_digest = _tensor_content_digest(loss_cpu)
    authorization_generation_digest = _digest_parts(
        RECEIPT_PROVENANCE,
        "canonical-cpu-material-optimizer-authorization-v1",
        snapshot.world_generation_digest,
        snapshot.sites_content_digest,
        snapshot.material_generation_id,
        snapshot.background_generation_id,
        result.step_generation_id,
        grad_content_digest,
        loss_content_digest,
    )
    provisional = PaperKineticLazyMaterialDeviceGradientReceipt(
        step_result_identity=id(result),
        step_result_generation_digest=result.generation_digest,
        snapshot_identity=id(snapshot),
        snapshot_generation_digest=snapshot.generation_digest,
        source_state_identity=id(state),
        world_generation_digest=snapshot.world_generation_digest,
        sites_content_digest=snapshot.sites_content_digest,
        material_generation_id_before=snapshot.material_generation_id,
        background_generation_id=snapshot.background_generation_id,
        source_step_index=snapshot.source_step_index,
        step_generation_id=result.step_generation_id,
        source_device=snapshot.device,
        grad_site_rgba_f32_cpu=grad_cpu,
        loss_f32_cpu=loss_cpu,
        grad_content_digest=grad_content_digest,
        loss_content_digest=loss_content_digest,
        tensor_signatures=tuple(_tensor_signature(tensor) for tensor in tensors),
        completion_capability_generation_digest=capability.generation_digest,
        completion_launch_generation_digest=launch_digest,
        completion_fence_sequence=fence_sequence,
        completion_receipt_generation_digest=completion_receipt.generation_digest,
        authorization_generation_digest=authorization_generation_digest,
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
    provisional.assert_current(state)
    completion_receipt.consume_for_subject(
        capability,
        binding,
        subject=subject,
        consumer="lazy-material-device-gradient-readback",
    )
    return provisional


@torch.no_grad()
def apply_paper_kinetic_lazy_material_device_gradient_receipt(
    state: PaperKineticFixedSiteMaterialState,
    receipt: PaperKineticLazyMaterialDeviceGradientReceipt,
) -> PaperKineticFixedSiteMaterialStepReceipt:
    """Consume one lazy receipt through the sole CPU manual-SGD mutation."""

    if not isinstance(receipt, PaperKineticLazyMaterialDeviceGradientReceipt):
        raise TypeError("lazy material apply requires its exact receipt")
    receipt.assert_current(state, require_unconsumed=True)
    gradient = receipt.grad_site_rgba_f32_cpu
    loss = receipt.loss_f32_cpu
    if not isinstance(gradient, torch.Tensor) or not isinstance(loss, torch.Tensor):
        raise ValueError("lazy material receipt lost its CPU tensors")
    step_receipt = _apply_validated_paper_kinetic_fixed_site_material_bars(
        state,
        grad_site_rgba_f32=gradient,
        loss_f32=loss,
        authorization_generation_digest=receipt.authorization_generation_digest,
        step_generation_id=receipt.step_generation_id,
    )
    snapshot = receipt._snapshot
    if not isinstance(snapshot, PaperKineticFixedSiteMaterialDeviceSnapshot):
        state.poisoned = True
        raise ValueError("lazy material receipt lost its snapshot at commit")
    receipt.material_generation_id_after = step_receipt.material_generation_id_after
    receipt.step_receipt_generation_digest = step_receipt.generation_digest
    receipt.grad_site_rgba_f32_cpu = None
    receipt.loss_f32_cpu = None
    receipt._result = None
    receipt.released_after_consumption = True
    receipt.consumed = True
    snapshot._release_after_consumption()
    receipt.consumption_generation_digest = _consumption_digest(receipt)
    try:
        receipt.assert_current(state, require_unconsumed=False)
    except BaseException:
        state.poisoned = True
        raise
    return step_receipt


def _readback_subject_digest(subject: _LazyGradientReadbackSubject) -> str:
    return _digest_parts(
        READBACK_SUBJECT_KIND,
        subject.subject_identity,
        subject.result_identity,
        subject.snapshot_identity,
        subject.result.generation_digest,
        subject.snapshot.generation_digest,
        subject.grad_signature,
        subject.loss_signature,
    )


def _receipt_digest(receipt: PaperKineticLazyMaterialDeviceGradientReceipt) -> str:
    return _digest_parts(
        RECEIPT_PROVENANCE,
        receipt.step_result_identity,
        receipt.step_result_generation_digest,
        receipt.snapshot_identity,
        receipt.snapshot_generation_digest,
        receipt.source_state_identity,
        receipt.world_generation_digest,
        receipt.sites_content_digest,
        receipt.material_generation_id_before,
        receipt.background_generation_id,
        receipt.source_step_index,
        receipt.step_generation_id,
        str(receipt.source_device),
        receipt.grad_content_digest,
        receipt.loss_content_digest,
        receipt.completion_capability_generation_digest,
        receipt.completion_launch_generation_digest,
        receipt.completion_fence_sequence,
        receipt.completion_receipt_generation_digest,
        receipt.authorization_generation_digest,
        receipt.receipt_identity,
        "one_fenced_d2h_then_one_cpu_sgd",
    )


def _consumption_digest(
    receipt: PaperKineticLazyMaterialDeviceGradientReceipt,
) -> str:
    return _digest_parts(
        RECEIPT_PROVENANCE,
        "consumed",
        receipt.generation_digest,
        receipt.material_generation_id_after,
        receipt.step_receipt_generation_digest,
        receipt.released_after_consumption,
    )


def _owned_cpu_f32(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.float32, copy=True).contiguous()


def _require_f32_device(
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...],
    device: torch.device,
    name: str,
) -> None:
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.dtype != torch.float32
        or tensor.device != device
        or tuple(tensor.shape) != shape
        or not tensor.is_contiguous()
        or tensor.requires_grad
    ):
        raise ValueError(f"{name} must be contiguous non-autograd float32 on {device}")


def _require_cpu_f32(
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...],
    name: str,
) -> None:
    _require_f32_device(
        tensor,
        shape=shape,
        device=torch.device("cpu"),
        name=name,
    )


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


def _require_sha256(value: str, *, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _digest_parts(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = repr(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


__all__ = (
    "PaperKineticLazyMaterialDeviceGradientReceipt",
    "apply_paper_kinetic_lazy_material_device_gradient_receipt",
    "seal_paper_kinetic_lazy_material_device_gradient_receipt",
)
