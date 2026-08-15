"""Generation-bound CPU/device material snapshots for fixed-site WorldFoam.

The canonical fixed-site material state remains CPU-owned and checkpointable.
This module lends one exact physical snapshot to a selected accelerator, then
returns one fenced material-gradient receipt to the shared manual-SGD updater.
It retains no frame/sample tape and releases bridge tensors after consumption.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from kinetic_dense_cached_native_material_request import (
    MPS_DEVICE_COMPLETION_FENCE_PROVENANCE,
    synchronize_mps_device_completion_fence,
)
from paper_kinetic_fixed_site_material_state import (
    PaperKineticFixedSiteMaterialState,
    PaperKineticFixedSiteMaterialStepReceipt,
    _apply_validated_paper_kinetic_fixed_site_material_bars,
)
from paper_kinetic_fixed_site_material_step import (
    PaperKineticFixedSiteMaterialStepResult,
)


SNAPSHOT_PROVENANCE = "paper-kinetic-fixed-site-material-device-snapshot-v1"
GRADIENT_RECEIPT_PROVENANCE = (
    "paper-kinetic-fixed-site-material-device-gradient-receipt-v1"
)

_SNAPSHOT_SEAL = object()
_GRADIENT_RECEIPT_SEAL = object()


@dataclass
class PaperKineticFixedSiteMaterialDeviceSnapshot:
    """One exact CPU material/background generation copied to one device."""

    world_generation_digest: str
    sites_content_digest: str
    material_generation_id: str
    background_generation_id: str
    background_content_digest: str
    source_step_index: int
    source_material_tensor_signature: tuple[object, ...] = field(repr=False)
    device: torch.device
    site_rgba_f32_device: torch.Tensor | None = field(repr=False)
    background_rgb_f32_device: torch.Tensor | None = field(repr=False)
    tensor_signatures: tuple[tuple[object, ...], ...] = field(repr=False)
    completion_fence_provenance: str
    generation_digest: str
    snapshot_identity: int
    source_state_identity: int = field(repr=False)
    issued_gradient_receipt_identity: int = 0
    issued_gradient_receipt_generation_digest: str = ""
    released_after_consumption: bool = False
    completion_fence_call_count: int = 1
    persistent_frame_tensor_bytes: int = 0
    persistent_sample_tensor_bytes: int = 0
    persistent_target_tensor_bytes: int = 0
    persistent_prediction_tensor_bytes: int = 0
    provenance: str = SNAPSHOT_PROVENANCE
    _seal: object = field(default=None, repr=False)

    @property
    def site_count(self) -> int:
        material = self.site_rgba_f32_device
        if material is not None:
            return int(material.shape[0])
        return int(self.tensor_signatures[0][5][0])

    @property
    def gradient_receipt_issued(self) -> bool:
        return self.issued_gradient_receipt_identity != 0

    @property
    def live_device_material_tensor_bytes(self) -> int:
        return 0 if self.released_after_consumption else self.site_count * 4 * 4

    @property
    def live_device_background_tensor_bytes(self) -> int:
        return 0 if self.released_after_consumption else 3 * 4

    @property
    def live_snapshot_tensor_bytes(self) -> int:
        return (
            self.live_device_material_tensor_bytes
            + self.live_device_background_tensor_bytes
        )

    def accounting(self) -> dict[str, Any]:
        """Return O(1) bridge accounting without reading tensor contents."""

        return {
            "provenance": self.provenance,
            "generation_digest": self.generation_digest,
            "material_generation_id": self.material_generation_id,
            "site_count": self.site_count,
            "device": str(self.device),
            "released_after_consumption": self.released_after_consumption,
            "live_device_material_tensor_bytes": (
                self.live_device_material_tensor_bytes
            ),
            "live_device_background_tensor_bytes": (
                self.live_device_background_tensor_bytes
            ),
            "live_snapshot_tensor_bytes": self.live_snapshot_tensor_bytes,
            "live_cpu_receipt_tensor_bytes": 0,
            "device_material_copy_count": 1,
            "device_material_content_digest_count": 0,
            "device_material_readback_count": 0,
            "persistent_frame_tensor_bytes": self.persistent_frame_tensor_bytes,
            "persistent_sample_tensor_bytes": self.persistent_sample_tensor_bytes,
            "persistent_target_tensor_bytes": self.persistent_target_tensor_bytes,
            "persistent_prediction_tensor_bytes": (
                self.persistent_prediction_tensor_bytes
            ),
        }

    def _assert_self_consistent(self) -> None:
        if (
            self._seal is not _SNAPSHOT_SEAL
            or self.provenance != SNAPSHOT_PROVENANCE
            or id(self) != self.snapshot_identity
            or self.device.type not in {"cpu", "mps", "cuda"}
            or self.source_step_index < 0
            or self.completion_fence_call_count != 1
            or not self.completion_fence_provenance.strip()
            or not self.background_generation_id.strip()
            or self.persistent_frame_tensor_bytes != 0
            or self.persistent_sample_tensor_bytes != 0
            or self.persistent_target_tensor_bytes != 0
            or self.persistent_prediction_tensor_bytes != 0
            or bool(self.issued_gradient_receipt_identity)
            != bool(self.issued_gradient_receipt_generation_digest)
        ):
            raise ValueError("fixed-site material device snapshot changed")
        for name, value in (
            ("world_generation_digest", self.world_generation_digest),
            ("sites_content_digest", self.sites_content_digest),
            ("material_generation_id", self.material_generation_id),
            ("background_content_digest", self.background_content_digest),
            ("generation_digest", self.generation_digest),
        ):
            _require_sha256(value, name=name)
        if self.generation_digest != _snapshot_digest(self):
            raise ValueError("fixed-site material device snapshot generation changed")
        tensors = (self.site_rgba_f32_device, self.background_rgb_f32_device)
        if self.released_after_consumption:
            if any(tensor is not None for tensor in tensors):
                raise ValueError("consumed material snapshot retained device tensors")
            return
        if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise ValueError("live material snapshot lost its device tensors")
        material, background = tensors
        _require_f32(
            material,
            name="device material snapshot",
            shape=(self.site_count, 4),
            device=self.device,
        )
        _require_f32(
            background,
            name="device background snapshot",
            shape=(3,),
            device=self.device,
        )
        if material.requires_grad or background.requires_grad:
            raise ValueError("device material snapshots must be non-autograd")
        if tuple(_tensor_signature(tensor) for tensor in tensors) != self.tensor_signatures:
            raise ValueError("device material snapshot tensor identity/layout changed")
        _require_distinct_storage(material, background)

    def assert_current(
        self,
        state: PaperKineticFixedSiteMaterialState,
        *,
        require_unissued: bool = False,
    ) -> None:
        if not isinstance(state, PaperKineticFixedSiteMaterialState):
            raise TypeError("material device snapshot requires fixed-site state")
        state.assert_current()
        self._assert_self_consistent()
        if self.released_after_consumption:
            raise ValueError("material device snapshot was already consumed")
        if (
            id(state) != self.source_state_identity
            or state.world_generation_digest != self.world_generation_digest
            or state.sites_content_digest != self.sites_content_digest
            or state.material_generation_id != self.material_generation_id
            or state.step_index != self.source_step_index
            or _tensor_signature(state.site_rgba_f32)
            != self.source_material_tensor_signature
        ):
            raise ValueError("material device snapshot is stale or foreign")
        if require_unissued and self.gradient_receipt_issued:
            raise ValueError("material device snapshot already issued a gradient receipt")

    def _release_after_consumption(self) -> None:
        self.site_rgba_f32_device = None
        self.background_rgb_f32_device = None
        self.released_after_consumption = True


@dataclass
class PaperKineticFixedSiteMaterialDeviceGradientReceipt:
    """One fenced device gradient copied to CPU and bound to its snapshot."""

    snapshot_generation_digest: str
    world_generation_digest: str
    sites_content_digest: str
    material_generation_id_before: str
    background_generation_id: str
    background_content_digest: str
    source_step_index: int
    step_generation_id: str
    step_result_generation_digest: str
    authorization_generation_digest: str
    accumulator_generation_digest: str
    replay_receipt_generation_digest: str
    optimizer_commit_generation_digest: str
    step_result_identity: int = field(repr=False)
    authorization_identity: int = field(repr=False)
    accumulator_identity: int = field(repr=False)
    replay_receipt_identity: int = field(repr=False)
    production_step_result_bound: bool
    source_device: torch.device
    grad_site_rgba_f32_cpu: torch.Tensor | None = field(repr=False)
    loss_f32_cpu: torch.Tensor | None = field(repr=False)
    grad_content_digest: str
    loss_content_digest: str
    tensor_signatures: tuple[tuple[object, ...], ...] = field(repr=False)
    completion_fence_provenance: str
    generation_digest: str
    receipt_identity: int
    snapshot_identity: int = field(repr=False)
    source_state_identity: int = field(repr=False)
    material_generation_id_after: str = ""
    step_receipt_generation_digest: str = ""
    consumption_generation_digest: str = ""
    consumed: bool = False
    released_after_consumption: bool = False
    completion_fence_call_count: int = 1
    persistent_frame_tensor_bytes: int = 0
    persistent_sample_tensor_bytes: int = 0
    persistent_target_tensor_bytes: int = 0
    persistent_prediction_tensor_bytes: int = 0
    provenance: str = GRADIENT_RECEIPT_PROVENANCE
    _snapshot: PaperKineticFixedSiteMaterialDeviceSnapshot | None = field(
        default=None,
        repr=False,
    )
    _seal: object = field(default=None, repr=False)

    @property
    def site_count(self) -> int:
        return int(self.tensor_signatures[0][5][0])

    @property
    def live_cpu_gradient_tensor_bytes(self) -> int:
        return 0 if self.released_after_consumption else self.site_count * 4 * 4

    @property
    def live_cpu_loss_tensor_bytes(self) -> int:
        return 0 if self.released_after_consumption else 4

    @property
    def live_cpu_receipt_tensor_bytes(self) -> int:
        return self.live_cpu_gradient_tensor_bytes + self.live_cpu_loss_tensor_bytes

    @property
    def live_device_snapshot_tensor_bytes(self) -> int:
        snapshot = self._snapshot
        if not isinstance(snapshot, PaperKineticFixedSiteMaterialDeviceSnapshot):
            return 0
        return snapshot.live_snapshot_tensor_bytes

    @property
    def live_bridge_tensor_bytes(self) -> int:
        return (
            self.live_device_snapshot_tensor_bytes
            + self.live_cpu_receipt_tensor_bytes
        )

    @property
    def applied_authorization_generation_digest(self) -> str:
        return self.optimizer_commit_generation_digest

    def accounting(self) -> dict[str, Any]:
        """Return O(1) live receipt/snapshot bytes without tensor readback."""

        return {
            "provenance": self.provenance,
            "generation_digest": self.generation_digest,
            "snapshot_generation_digest": self.snapshot_generation_digest,
            "step_result_generation_digest": self.step_result_generation_digest,
            "authorization_generation_digest": (
                self.authorization_generation_digest
            ),
            "accumulator_generation_digest": self.accumulator_generation_digest,
            "replay_receipt_generation_digest": (
                self.replay_receipt_generation_digest
            ),
            "optimizer_commit_generation_digest": (
                self.optimizer_commit_generation_digest
            ),
            "production_step_result_bound": self.production_step_result_bound,
            "material_generation_id_before": self.material_generation_id_before,
            "material_generation_id_after": self.material_generation_id_after,
            "site_count": self.site_count,
            "source_device": str(self.source_device),
            "consumed": self.consumed,
            "released_after_consumption": self.released_after_consumption,
            "live_device_snapshot_tensor_bytes": (
                self.live_device_snapshot_tensor_bytes
            ),
            "live_cpu_gradient_tensor_bytes": self.live_cpu_gradient_tensor_bytes,
            "live_cpu_loss_tensor_bytes": self.live_cpu_loss_tensor_bytes,
            "live_cpu_receipt_tensor_bytes": self.live_cpu_receipt_tensor_bytes,
            "live_bridge_tensor_bytes": self.live_bridge_tensor_bytes,
            "device_to_host_copy_phase_count": 1,
            "device_to_host_tensor_count": 2,
            "device_material_content_digest_count": 0,
            "persistent_frame_tensor_bytes": self.persistent_frame_tensor_bytes,
            "persistent_sample_tensor_bytes": self.persistent_sample_tensor_bytes,
            "persistent_target_tensor_bytes": self.persistent_target_tensor_bytes,
            "persistent_prediction_tensor_bytes": (
                self.persistent_prediction_tensor_bytes
            ),
        }

    def assert_current(
        self,
        state: PaperKineticFixedSiteMaterialState,
        *,
        require_unconsumed: bool = True,
    ) -> None:
        if not isinstance(state, PaperKineticFixedSiteMaterialState):
            raise TypeError("material gradient receipt requires fixed-site state")
        snapshot = self._snapshot
        binding_digests = (
            self.step_result_generation_digest,
            self.authorization_generation_digest,
            self.accumulator_generation_digest,
            self.replay_receipt_generation_digest,
        )
        binding_identities = (
            self.step_result_identity,
            self.authorization_identity,
            self.accumulator_identity,
            self.replay_receipt_identity,
        )
        complete_production_binding = all(binding_digests) and all(
            identity > 0 for identity in binding_identities
        )
        empty_test_binding = not any(binding_digests) and not any(
            binding_identities
        )
        if (
            self._seal is not _GRADIENT_RECEIPT_SEAL
            or self.provenance != GRADIENT_RECEIPT_PROVENANCE
            or id(self) != self.receipt_identity
            or not isinstance(snapshot, PaperKineticFixedSiteMaterialDeviceSnapshot)
            or id(snapshot) != self.snapshot_identity
            or snapshot.issued_gradient_receipt_identity != id(self)
            or snapshot.issued_gradient_receipt_generation_digest
            != self.generation_digest
            or self.snapshot_generation_digest != snapshot.generation_digest
            or self.source_state_identity != snapshot.source_state_identity
            or self.world_generation_digest != snapshot.world_generation_digest
            or self.sites_content_digest != snapshot.sites_content_digest
            or self.material_generation_id_before != snapshot.material_generation_id
            or self.background_generation_id != snapshot.background_generation_id
            or self.background_content_digest != snapshot.background_content_digest
            or self.source_step_index != snapshot.source_step_index
            or self.source_device != snapshot.device
            or not self.step_generation_id.strip()
            or self.production_step_result_bound
            != complete_production_binding
            or not self.production_step_result_bound
            and not empty_test_binding
            or self.completion_fence_call_count != 1
            or not self.completion_fence_provenance.strip()
            or self.persistent_frame_tensor_bytes != 0
            or self.persistent_sample_tensor_bytes != 0
            or self.persistent_target_tensor_bytes != 0
            or self.persistent_prediction_tensor_bytes != 0
            or self.generation_digest != _gradient_receipt_digest(self)
        ):
            raise ValueError("fixed-site material gradient receipt changed or is foreign")
        snapshot._assert_self_consistent()
        for name, value in (
            ("grad_content_digest", self.grad_content_digest),
            ("loss_content_digest", self.loss_content_digest),
            (
                "optimizer_commit_generation_digest",
                self.optimizer_commit_generation_digest,
            ),
            ("generation_digest", self.generation_digest),
        ):
            _require_sha256(value, name=name)
        if self.production_step_result_bound:
            for name, value in zip(
                (
                    "step_result_generation_digest",
                    "authorization_generation_digest",
                    "accumulator_generation_digest",
                    "replay_receipt_generation_digest",
                ),
                binding_digests,
                strict=True,
            ):
                _require_sha256(value, name=name)
        if self.consumed:
            if require_unconsumed:
                raise ValueError("fixed-site material gradient receipt was already consumed")
            if (
                not self.released_after_consumption
                or self.grad_site_rgba_f32_cpu is not None
                or self.loss_f32_cpu is not None
                or not snapshot.released_after_consumption
                or id(state) != self.source_state_identity
                or state.generation_parent_digest
                != self.material_generation_id_before
                or state.material_generation_id
                != self.material_generation_id_after
                or state.last_authorization_generation_digest
                != self.applied_authorization_generation_digest
                or state.last_step_generation_id != self.step_generation_id
                or state.step_index != self.source_step_index + 1
                or not self.step_receipt_generation_digest
                or self.consumption_generation_digest
                != _gradient_consumption_digest(self)
            ):
                raise ValueError("consumed material gradient receipt changed")
            state.assert_current()
            return
        if (
            self.released_after_consumption
            or self.material_generation_id_after
            or self.step_receipt_generation_digest
            or self.consumption_generation_digest
        ):
            raise ValueError("unconsumed material gradient receipt has commit metadata")
        snapshot.assert_current(state)
        tensors = (self.grad_site_rgba_f32_cpu, self.loss_f32_cpu)
        if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise ValueError("unconsumed material gradient receipt lost CPU tensors")
        gradient, loss = tensors
        _require_f32(
            gradient,
            name="CPU material gradient receipt",
            shape=(self.site_count, 4),
            device=torch.device("cpu"),
        )
        _require_f32(
            loss,
            name="CPU material loss receipt",
            shape=(1,),
            device=torch.device("cpu"),
        )
        if gradient.requires_grad or loss.requires_grad:
            raise ValueError("CPU material gradient receipt must be non-autograd")
        if tuple(_tensor_signature(tensor) for tensor in tensors) != self.tensor_signatures:
            raise ValueError("CPU material gradient receipt tensor identity/layout changed")
        _require_distinct_storage(*state._tensors(), gradient, loss)
        if (
            _tensor_content_digest(gradient) != self.grad_content_digest
            or _tensor_content_digest(loss) != self.loss_content_digest
            or not bool(torch.isfinite(gradient).all().item())
            or not bool(torch.isfinite(loss).all().item())
        ):
            raise ValueError("CPU material gradient receipt numeric content changed")

    def _commit_consumption(
        self,
        step_receipt: PaperKineticFixedSiteMaterialStepReceipt,
    ) -> None:
        snapshot = self._snapshot
        if snapshot is None:
            raise ValueError("material gradient receipt lost its source snapshot")
        self.material_generation_id_after = step_receipt.material_generation_id_after
        self.step_receipt_generation_digest = step_receipt.generation_digest
        self.grad_site_rgba_f32_cpu = None
        self.loss_f32_cpu = None
        self.released_after_consumption = True
        self.consumed = True
        snapshot._release_after_consumption()
        self.consumption_generation_digest = _gradient_consumption_digest(self)


@torch.no_grad()
def snapshot_paper_kinetic_fixed_site_material_to_device(
    state: PaperKineticFixedSiteMaterialState,
    *,
    background_rgb_f32_cpu: torch.Tensor,
    background_generation_id: str,
    device: torch.device | str,
    device_completion_fence: Callable[[], None],
    device_completion_fence_provenance: str,
) -> PaperKineticFixedSiteMaterialDeviceSnapshot:
    """Copy and seal one exact CPU state/background generation on ``device``."""

    if not isinstance(state, PaperKineticFixedSiteMaterialState):
        raise TypeError("material device snapshot requires fixed-site state")
    state.assert_current()
    if state.device.type != "cpu":
        raise ValueError("material device snapshots require a CPU-owned source state")
    if not isinstance(background_generation_id, str) or not background_generation_id.strip():
        raise ValueError("background_generation_id must be nonempty")
    resolved_device = torch.device(device)
    if resolved_device.type not in {"cpu", "mps", "cuda"}:
        raise ValueError("material bridge supports CPU, MPS, or CUDA devices")
    _validate_completion_fence(
        resolved_device,
        device_completion_fence,
        device_completion_fence_provenance,
    )
    _require_f32(
        background_rgb_f32_cpu,
        name="CPU background snapshot",
        shape=(3,),
        device=torch.device("cpu"),
    )
    if background_rgb_f32_cpu.requires_grad or not bool(
        torch.isfinite(background_rgb_f32_cpu).all().item()
    ):
        raise ValueError("CPU background snapshot must be finite and non-autograd")

    source_material_tensor_signature = _tensor_signature(state.site_rgba_f32)
    background_content_digest = _tensor_content_digest(background_rgb_f32_cpu)
    # ``copy=True`` creates the one owned destination directly.  In particular,
    # do not follow a cross-device ``to`` with ``clone``: that temporarily keeps
    # two O(S) device material buffers alive and defeats this bridge's memory
    # contract.
    material_device = state.site_rgba_f32.detach().to(
        device=resolved_device,
        dtype=torch.float32,
        copy=True,
    ).contiguous()
    background_device = background_rgb_f32_cpu.detach().to(
        device=resolved_device,
        dtype=torch.float32,
        copy=True,
    ).contiguous()
    returned = device_completion_fence()
    if returned is not None:
        raise TypeError("device completion fence must return None")
    if _tensor_signature(state.site_rgba_f32) != source_material_tensor_signature:
        raise ValueError("CPU material source changed during the fenced device copy")
    tensors = (material_device, background_device)
    provisional = PaperKineticFixedSiteMaterialDeviceSnapshot(
        world_generation_digest=state.world_generation_digest,
        sites_content_digest=state.sites_content_digest,
        material_generation_id=state.material_generation_id,
        background_generation_id=background_generation_id,
        background_content_digest=background_content_digest,
        source_step_index=state.step_index,
        source_material_tensor_signature=source_material_tensor_signature,
        device=resolved_device,
        site_rgba_f32_device=material_device,
        background_rgb_f32_device=background_device,
        tensor_signatures=tuple(_tensor_signature(tensor) for tensor in tensors),
        completion_fence_provenance=device_completion_fence_provenance,
        generation_digest="",
        snapshot_identity=0,
        source_state_identity=id(state),
        _seal=_SNAPSHOT_SEAL,
    )
    provisional.snapshot_identity = id(provisional)
    provisional.generation_digest = _snapshot_digest(provisional)
    provisional.assert_current(state, require_unissued=True)
    return provisional


@torch.no_grad()
def seal_paper_kinetic_fixed_site_material_device_gradient_receipt(
    state: PaperKineticFixedSiteMaterialState,
    snapshot: PaperKineticFixedSiteMaterialDeviceSnapshot,
    step_result: PaperKineticFixedSiteMaterialStepResult,
    *,
    device_completion_fence: Callable[[], None],
    device_completion_fence_provenance: str,
) -> PaperKineticFixedSiteMaterialDeviceGradientReceipt:
    """Seal the exact coordinator capability for one CPU optimizer apply.

    This is the production entry point.  It proves that the dense coordinator's
    sealed accumulator was created against this exact device material snapshot;
    callers cannot substitute arbitrary bars, loss, or step identity.
    """

    if not isinstance(step_result, PaperKineticFixedSiteMaterialStepResult):
        raise TypeError("device gradient receipt requires a sealed material step result")
    step_result.assert_current()
    snapshot.assert_current(state, require_unissued=True)
    authorization = step_result.authorization
    accumulator = step_result.accumulator
    replay_receipt = step_result.replay_receipt
    if (
        accumulator._material_tensor_ref is not snapshot.site_rgba_f32_device
        or accumulator.material_tensor_identity
        != id(snapshot.site_rgba_f32_device)
        or accumulator.material_generation_id != snapshot.material_generation_id
        or accumulator.background_generation_id != snapshot.background_generation_id
        or accumulator._background_tensor_ref
        is not snapshot.background_rgb_f32_device
        or accumulator.background_tensor_identity
        != id(snapshot.background_rgb_f32_device)
        or accumulator.world_generation_digest != snapshot.world_generation_digest
        or accumulator.world_sites_content_digest != snapshot.sites_content_digest
        or authorization.step_generation_id != accumulator.step_generation_id
        or authorization.grad_site_rgba_f32 is not accumulator.grad_site_rgba_f32
        or authorization.loss_f32 is not accumulator.loss_f32
    ):
        raise ValueError(
            "material step result is not bound to the exact device snapshot"
        )
    return _seal_paper_kinetic_fixed_site_material_device_gradient_receipt_raw_for_test(
        state,
        snapshot,
        grad_site_rgba_f32_device=authorization.grad_site_rgba_f32,
        loss_f32_device=authorization.loss_f32,
        step_generation_id=authorization.step_generation_id,
        device_completion_fence=device_completion_fence,
        device_completion_fence_provenance=device_completion_fence_provenance,
        step_result_generation_digest=step_result.generation_digest,
        authorization_generation_digest=authorization.generation_digest,
        accumulator_generation_digest=accumulator.generation_digest,
        replay_receipt_generation_digest=replay_receipt.generation_digest,
        step_result_identity=id(step_result),
        authorization_identity=id(authorization),
        accumulator_identity=id(accumulator),
        replay_receipt_identity=id(replay_receipt),
        production_step_result_bound=True,
    )


@torch.no_grad()
def _seal_paper_kinetic_fixed_site_material_device_gradient_receipt_raw_for_test(
    state: PaperKineticFixedSiteMaterialState,
    snapshot: PaperKineticFixedSiteMaterialDeviceSnapshot,
    *,
    grad_site_rgba_f32_device: torch.Tensor,
    loss_f32_device: torch.Tensor,
    step_generation_id: str,
    device_completion_fence: Callable[[], None],
    device_completion_fence_provenance: str,
    step_result_generation_digest: str = "",
    authorization_generation_digest: str = "",
    accumulator_generation_digest: str = "",
    replay_receipt_generation_digest: str = "",
    step_result_identity: int = 0,
    authorization_identity: int = 0,
    accumulator_identity: int = 0,
    replay_receipt_identity: int = 0,
    production_step_result_bound: bool = False,
) -> PaperKineticFixedSiteMaterialDeviceGradientReceipt:
    """Test-only raw bridge primitive beneath the sealed-result entry point."""

    if not isinstance(snapshot, PaperKineticFixedSiteMaterialDeviceSnapshot):
        raise TypeError("device gradient receipt requires a sealed material snapshot")
    snapshot.assert_current(state, require_unissued=True)
    if not isinstance(step_generation_id, str) or not step_generation_id.strip():
        raise ValueError("step_generation_id must be nonempty")
    _validate_completion_fence(
        snapshot.device,
        device_completion_fence,
        device_completion_fence_provenance,
    )
    _require_f32(
        grad_site_rgba_f32_device,
        name="device material gradient",
        shape=(snapshot.site_count, 4),
        device=snapshot.device,
    )
    _require_f32(
        loss_f32_device,
        name="device material loss",
        shape=(1,),
        device=snapshot.device,
    )
    if grad_site_rgba_f32_device.requires_grad or loss_f32_device.requires_grad:
        raise ValueError("device material gradient/loss must be non-autograd")
    _require_distinct_storage(
        snapshot.site_rgba_f32_device,
        snapshot.background_rgb_f32_device,
        grad_site_rgba_f32_device,
        loss_f32_device,
    )
    returned = device_completion_fence()
    if returned is not None:
        raise TypeError("device completion fence must return None")

    gradient_cpu = _owned_cpu_f32(grad_site_rgba_f32_device)
    loss_cpu = _owned_cpu_f32(loss_f32_device)
    if not bool(torch.isfinite(gradient_cpu).all().item()) or not bool(
        torch.isfinite(loss_cpu).all().item()
    ):
        raise ValueError("fenced device material gradient/loss is nonfinite")
    tensors = (gradient_cpu, loss_cpu)
    grad_content_digest = _tensor_content_digest(gradient_cpu)
    loss_content_digest = _tensor_content_digest(loss_cpu)
    # Runtime result/authorization identities remain bound into the enclosing
    # receipt, but the CPU material version chain needs a restart-stable commit
    # identity.  The exact sealed coordinator has already certified coverage;
    # this canonical digest binds that semantic step to the copied numeric bars
    # without importing process-local tensor identities into checkpoints.
    optimizer_commit_generation_digest = _digest_parts(
        GRADIENT_RECEIPT_PROVENANCE,
        "canonical-optimizer-commit-v1",
        snapshot.world_generation_digest,
        snapshot.sites_content_digest,
        snapshot.material_generation_id,
        snapshot.background_generation_id,
        snapshot.background_content_digest,
        step_generation_id,
        grad_content_digest,
        loss_content_digest,
        production_step_result_bound,
    )
    provisional = PaperKineticFixedSiteMaterialDeviceGradientReceipt(
        snapshot_generation_digest=snapshot.generation_digest,
        world_generation_digest=snapshot.world_generation_digest,
        sites_content_digest=snapshot.sites_content_digest,
        material_generation_id_before=snapshot.material_generation_id,
        background_generation_id=snapshot.background_generation_id,
        background_content_digest=snapshot.background_content_digest,
        source_step_index=snapshot.source_step_index,
        step_generation_id=step_generation_id,
        step_result_generation_digest=step_result_generation_digest,
        authorization_generation_digest=authorization_generation_digest,
        accumulator_generation_digest=accumulator_generation_digest,
        replay_receipt_generation_digest=replay_receipt_generation_digest,
        optimizer_commit_generation_digest=(
            optimizer_commit_generation_digest
        ),
        step_result_identity=step_result_identity,
        authorization_identity=authorization_identity,
        accumulator_identity=accumulator_identity,
        replay_receipt_identity=replay_receipt_identity,
        production_step_result_bound=production_step_result_bound,
        source_device=snapshot.device,
        grad_site_rgba_f32_cpu=gradient_cpu,
        loss_f32_cpu=loss_cpu,
        grad_content_digest=grad_content_digest,
        loss_content_digest=loss_content_digest,
        tensor_signatures=tuple(_tensor_signature(tensor) for tensor in tensors),
        completion_fence_provenance=device_completion_fence_provenance,
        generation_digest="",
        receipt_identity=0,
        snapshot_identity=id(snapshot),
        source_state_identity=id(state),
        _snapshot=snapshot,
        _seal=_GRADIENT_RECEIPT_SEAL,
    )
    provisional.receipt_identity = id(provisional)
    provisional.generation_digest = _gradient_receipt_digest(provisional)
    snapshot.issued_gradient_receipt_identity = id(provisional)
    snapshot.issued_gradient_receipt_generation_digest = (
        provisional.generation_digest
    )
    try:
        provisional.assert_current(state)
    except BaseException:
        snapshot.issued_gradient_receipt_identity = 0
        snapshot.issued_gradient_receipt_generation_digest = ""
        raise
    return provisional


@torch.no_grad()
def apply_paper_kinetic_fixed_site_material_device_gradient_receipt(
    state: PaperKineticFixedSiteMaterialState,
    receipt: PaperKineticFixedSiteMaterialDeviceGradientReceipt,
) -> PaperKineticFixedSiteMaterialStepReceipt:
    """Consume one receipt through the canonical fixed-site SGD mutation path."""

    if not isinstance(receipt, PaperKineticFixedSiteMaterialDeviceGradientReceipt):
        raise TypeError("device material apply requires its exact gradient receipt")
    receipt.assert_current(state, require_unconsumed=True)
    state_identity = id(state)
    gradient = receipt.grad_site_rgba_f32_cpu
    loss = receipt.loss_f32_cpu
    if not isinstance(gradient, torch.Tensor) or not isinstance(loss, torch.Tensor):
        raise ValueError("device material receipt lost its CPU bars")
    step_receipt = _apply_validated_paper_kinetic_fixed_site_material_bars(
        state,
        grad_site_rgba_f32=gradient,
        loss_f32=loss,
        authorization_generation_digest=(
            receipt.applied_authorization_generation_digest
        ),
        step_generation_id=receipt.step_generation_id,
    )
    if id(state) != state_identity:
        state.poisoned = True
        raise ArithmeticError("fixed-site material updater replaced state identity")
    receipt._commit_consumption(step_receipt)
    try:
        receipt.assert_current(state, require_unconsumed=False)
    except BaseException:
        state.poisoned = True
        raise
    return step_receipt


def _snapshot_digest(snapshot: PaperKineticFixedSiteMaterialDeviceSnapshot) -> str:
    return _digest_parts(
        SNAPSHOT_PROVENANCE,
        snapshot.world_generation_digest,
        snapshot.sites_content_digest,
        snapshot.material_generation_id,
        snapshot.background_generation_id,
        snapshot.background_content_digest,
        snapshot.source_step_index,
        str(snapshot.device),
        snapshot.completion_fence_provenance,
        snapshot.completion_fence_call_count,
        "no_frame_or_sample_state",
    )


def _gradient_receipt_digest(
    receipt: PaperKineticFixedSiteMaterialDeviceGradientReceipt,
) -> str:
    return _digest_parts(
        GRADIENT_RECEIPT_PROVENANCE,
        receipt.snapshot_generation_digest,
        receipt.world_generation_digest,
        receipt.sites_content_digest,
        receipt.material_generation_id_before,
        receipt.background_generation_id,
        receipt.background_content_digest,
        receipt.source_step_index,
        receipt.step_generation_id,
        receipt.step_result_generation_digest,
        receipt.authorization_generation_digest,
        receipt.accumulator_generation_digest,
        receipt.replay_receipt_generation_digest,
        receipt.optimizer_commit_generation_digest,
        receipt.step_result_identity,
        receipt.authorization_identity,
        receipt.accumulator_identity,
        receipt.replay_receipt_identity,
        receipt.production_step_result_bound,
        str(receipt.source_device),
        receipt.grad_content_digest,
        receipt.loss_content_digest,
        receipt.completion_fence_provenance,
        receipt.completion_fence_call_count,
        "one_shot_cpu_apply",
    )


def _gradient_consumption_digest(
    receipt: PaperKineticFixedSiteMaterialDeviceGradientReceipt,
) -> str:
    return _digest_parts(
        GRADIENT_RECEIPT_PROVENANCE,
        "consumed",
        receipt.generation_digest,
        receipt.material_generation_id_after,
        receipt.step_receipt_generation_digest,
        receipt.released_after_consumption,
    )


def _validate_completion_fence(
    device: torch.device,
    fence: Callable[[], None],
    provenance: str,
) -> None:
    if not callable(fence):
        raise TypeError("device completion fence must be callable")
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError("device completion fence provenance must be nonempty")
    if device.type == "mps" and (
        fence is not synchronize_mps_device_completion_fence
        or provenance != MPS_DEVICE_COMPLETION_FENCE_PROVENANCE
    ):
        raise ValueError("MPS material bridge requires the canonical completion fence")


def _owned_cpu_f32(tensor: torch.Tensor) -> torch.Tensor:
    # One device-to-host transfer produces the owned receipt.  Inputs are
    # required contiguous, so ``contiguous`` is metadata-only here.
    return tensor.detach().to(
        device="cpu",
        dtype=torch.float32,
        copy=True,
    ).contiguous()


def _tensor_content_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(repr(tuple(value.shape)).encode("utf-8"))
    if value.numel():
        digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()


def _tensor_signature(tensor: torch.Tensor) -> tuple[object, ...]:
    storage = tensor.untyped_storage()
    return (
        id(tensor),
        int(getattr(tensor, "_version", 0)),
        int(storage.data_ptr()),
        int(storage.nbytes()),
        int(tensor.storage_offset()),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
        bool(tensor.requires_grad),
    )


def _require_f32(
    tensor: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    device: torch.device,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if (
        tensor.dtype != torch.float32
        or tensor.device != device
        or tuple(tensor.shape) != shape
        or not tensor.is_contiguous()
    ):
        raise ValueError(f"{name} must be contiguous float32 {shape} on {device}")


def _require_distinct_storage(*tensors: torch.Tensor | None) -> None:
    if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise ValueError("material bridge live tensors must be present")
    keys = tuple(
        (str(tensor.device), int(tensor.untyped_storage().data_ptr()))
        for tensor in tensors
    )
    if len(set(keys)) != len(keys):
        raise ValueError("material bridge tensors must own distinct storage")


def _require_sha256(value: Any, *, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    try:
        parsed = bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest") from error
    if len(parsed) != 32 or value != value.lower():
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _digest_parts(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = repr(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


__all__ = [
    "GRADIENT_RECEIPT_PROVENANCE",
    "SNAPSHOT_PROVENANCE",
    "PaperKineticFixedSiteMaterialDeviceGradientReceipt",
    "PaperKineticFixedSiteMaterialDeviceSnapshot",
    "apply_paper_kinetic_fixed_site_material_device_gradient_receipt",
    "seal_paper_kinetic_fixed_site_material_device_gradient_receipt",
    "snapshot_paper_kinetic_fixed_site_material_to_device",
]
