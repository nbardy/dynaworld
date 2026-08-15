"""Exact-type completion capability for the lazy kinetic WorldFoam lane.

The lazy native coordinator currently accepts an arbitrary Python callable as
its completion fence.  A provenance string and callback identity cannot prove
that such a callable synchronizes the backend, device, or dispatch domain that
produced the tensors whose lifetimes are about to be released.

This module is the isolated replacement contract.  It deliberately contains
no caller-supplied synchronization callable.  The capability owns the
backend-specific synchronization operation:

* CPU: native/Torch calls are required to have completed before returning;
* MPS: ``torch.mps.synchronize()`` is a device-wide completion fence;
* CUDA: ``torch.cuda.synchronize(bound_device)`` is a device-wide fence.

Device-wide completion is stronger than stream-local completion, so it cannot
accidentally synchronize a sibling stream while leaving the producer stream
live.  The capability is additionally bound to one exact native-ops object,
its lazy-lane ABI identities, one normalized device, one owner generation, and
one creating thread.  It is sequentially reusable, but each invocation has a
unique monotone sequence number and cannot be re-entered.  A synchronization
exception permanently poisons the capability and makes completion unknown;
the caller must quarantine every live producer/consumer root and restart.

MPS construction is admitted only for the exact canonical WorldFoam native
module after its compiled ABI/source-freshness check and selected Metal-kernel
resource attestation have both run against the exact live dispatch anchor.
CUDA remains fail-closed because it has no equivalent launch-domain
attestation in this lane yet.  Minting a capability is still only authority to
fence work; it is not paper evidence that a particular training run completed.
"""

from __future__ import annotations

import hashlib
import secrets
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

import torch


CAPABILITY_PROVENANCE = "paper-kinetic-sealed-device-completion-v1"
CAPABILITY_STATUS = "cpu_contract_or_attested_mps/cuda_promotion_gate_closed"

CANONICAL_MPS_NATIVE_OPS_MODULE = "torch_world_foam_lane2_fused_slab.ops"
MPS_LAUNCH_DOMAIN_ATTESTATION = (
    "worldfoam-lane2-selected-material-kernels-and-device-wide-mps-fence-v1"
)

CPU_CALL_RETURN_SCOPE = "cpu/call-return-synchronous"
MPS_DEVICE_WIDE_SCOPE = "mps/torch-device-wide"
CUDA_DEVICE_WIDE_SCOPE = "cuda/torch-device-wide"

LAZY_NATIVE_REQUIRED_OP_NAMES = (
    "kinetic_precompiled_length_p0_lie_node_forward_into_launch_only_v1",
    "kinetic_precompiled_length_p0_lie_material_node_vjp_accumulate_launch_only",
    "prepare_kinetic_ragged_p0_lie_sample_block",
    "kinetic_ragged_p0_lie_sample_accumulate_loss_only_launch_only",
)

_CAPABILITY_SEAL = object()
_SUBJECT_BINDING_SEAL = object()
_LAUNCH_EPOCH_SEAL = object()
_RECEIPT_SEAL = object()
_RECEIPT_CONSUMPTION_SEAL = object()
_LOCK_TYPE = type(threading.Lock())
_TORCH_MPS_SYNCHRONIZE = getattr(getattr(torch, "mps", None), "synchronize", None)
_TORCH_CUDA_SYNCHRONIZE = getattr(getattr(torch, "cuda", None), "synchronize", None)


class PaperKineticCompletionUnknownError(RuntimeError):
    """The sole device-wide fence attempt failed; every root must survive."""


@dataclass(frozen=True, slots=True)
class PaperKineticCompletionSubjectBinding:
    """Exact owner-bound identity for one stable prelaunch subject slot."""

    capability_identity: int
    capability_generation_digest: str
    owner_generation_digest: str
    capability_nonce_digest: str
    subject_kind: str
    subject_identity: int
    subject_generation_digest: str
    binding_identity: int
    generation_digest: str
    _subject: Any = field(repr=False, compare=False)
    provenance: str = CAPABILITY_PROVENANCE
    _seal: object = field(default=None, repr=False, compare=False)

    def __init_subclass__(cls, **kwargs: Any) -> NoReturn:
        raise TypeError("sealed completion subject bindings cannot be subclassed")

    def assert_current(self) -> None:
        subject = self._subject
        if type(self) is not PaperKineticCompletionSubjectBinding:
            raise TypeError("completion subject binding has a foreign exact type")
        if (
            self._seal is not _SUBJECT_BINDING_SEAL
            or self.provenance != CAPABILITY_PROVENANCE
            or self.capability_identity < 1
            or not _is_sha256(self.capability_generation_digest)
            or not _is_sha256(self.owner_generation_digest)
            or not _is_sha256(self.capability_nonce_digest)
            or not self.subject_kind.strip()
            or self.subject_identity < 1
            or not _is_sha256(self.subject_generation_digest)
            or subject is None
            or id(subject) != self.subject_identity
            or getattr(subject, "generation_digest", None)
            != self.subject_generation_digest
            or self.binding_identity < 1
            or id(self) != self.binding_identity
            or self.generation_digest != _subject_binding_digest(self)
        ):
            raise ValueError("sealed completion subject binding changed")

    def assert_for(
        self,
        capability: PaperKineticSealedCompletionFence,
        *,
        subject: Any,
    ) -> None:
        """Validate the exact capability and immutable stable subject object."""

        self.assert_current()
        if type(capability) is not PaperKineticSealedCompletionFence:
            raise TypeError("completion subject requires the exact capability")
        capability.assert_current()
        if (
            id(capability) != self.capability_identity
            or capability.generation_digest
            != self.capability_generation_digest
            or capability.owner_generation_digest
            != self.owner_generation_digest
            or capability.capability_nonce_digest
            != self.capability_nonce_digest
            or subject is not self._subject
            or getattr(subject, "generation_digest", None)
            != self.subject_generation_digest
        ):
            raise ValueError("completion subject binding is foreign or stale")


@dataclass(slots=True)
class PaperKineticCompletionLaunchEpoch:
    """Exact one-shot capability-owned launch registration."""

    capability_generation_digest: str
    launch_epoch_sequence: int
    stage: str
    launch_generation_digest: str
    generation_digest: str
    subject_binding: PaperKineticCompletionSubjectBinding | None = field(
        default=None,
        repr=False,
    )
    subject_binding_identity: int | None = None
    subject_binding_generation_digest: str = ""
    fenced: bool = False
    _seal: object = field(default=None, repr=False)

    def __init_subclass__(cls, **kwargs: Any) -> NoReturn:
        raise TypeError("sealed completion launch epochs cannot be subclassed")

    def assert_current(self) -> None:
        binding = self.subject_binding
        if binding is not None:
            binding.assert_current()
        if (
            type(self) is not PaperKineticCompletionLaunchEpoch
            or self._seal is not _LAUNCH_EPOCH_SEAL
            or not _is_sha256(self.capability_generation_digest)
            or isinstance(self.launch_epoch_sequence, bool)
            or not isinstance(self.launch_epoch_sequence, int)
            or self.launch_epoch_sequence < 1
            or not self.stage.strip()
            or not _is_sha256(self.launch_generation_digest)
            or (binding is None) != (self.subject_binding_identity is None)
            or (binding is None)
            != (self.subject_binding_generation_digest == "")
            or binding is not None
            and (
                id(binding) != self.subject_binding_identity
                or binding.generation_digest
                != self.subject_binding_generation_digest
                or binding.capability_generation_digest
                != self.capability_generation_digest
            )
            or not isinstance(self.fenced, bool)
            or self.generation_digest != _launch_epoch_digest(self)
        ):
            raise ValueError("sealed completion launch epoch changed")


@dataclass(slots=True)
class _PaperKineticCompletionReceiptConsumption:
    """Private one-shot ledger shared only by one sealed receipt."""

    consumed: bool = False
    consumer: str | None = None
    _seal: object = field(default=None, repr=False)

    def assert_current(self) -> None:
        if (
            self._seal is not _RECEIPT_CONSUMPTION_SEAL
            or not isinstance(self.consumed, bool)
            or self.consumed != (self.consumer is not None)
            or self.consumer is not None
            and not self.consumer.strip()
        ):
            raise ValueError("sealed completion-fence receipt consumption changed")


@dataclass(frozen=True, slots=True)
class PaperKineticCompletionFenceReceipt:
    """Tensor-free proof of one successful capability-owned fence call."""

    capability_generation_digest: str
    owner_generation_digest: str
    capability_nonce_digest: str
    receipt_identity: int
    launch_epoch_sequence: int
    fence_sequence: int
    stage: str
    launch_generation_digest: str
    backend_type: str
    normalized_device: str
    completion_scope: str
    native_ops_identity: int
    native_abi_identity_digest: str
    creating_thread_identity: int
    subject_binding_identity: int | None
    subject_binding_generation_digest: str
    subject_kind: str
    subject_identity: int | None
    subject_generation_digest: str
    generation_digest: str
    completion_known: bool = True
    completion_domain_drained: bool = True
    fence_call_count: int = 1
    provenance: str = CAPABILITY_PROVENANCE
    runtime_status: str = CAPABILITY_STATUS
    subject_binding: PaperKineticCompletionSubjectBinding | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _consumption: _PaperKineticCompletionReceiptConsumption = field(
        default=None,
        repr=False,
        compare=False,
    )
    _seal: object = field(default=None, repr=False, compare=False)

    def __init_subclass__(cls, **kwargs: Any) -> NoReturn:
        raise TypeError("sealed completion-fence receipts cannot be subclassed")

    def assert_current(self) -> None:
        binding = self.subject_binding
        if binding is not None:
            binding.assert_current()
        if isinstance(self._consumption, _PaperKineticCompletionReceiptConsumption):
            self._consumption.assert_current()
        if type(self) is not PaperKineticCompletionFenceReceipt:
            raise TypeError("completion-fence receipt has a foreign exact type")
        if (
            self._seal is not _RECEIPT_SEAL
            or self.provenance != CAPABILITY_PROVENANCE
            or self.runtime_status != CAPABILITY_STATUS
            or not _is_sha256(self.capability_generation_digest)
            or not _is_sha256(self.owner_generation_digest)
            or not _is_sha256(self.capability_nonce_digest)
            or self.receipt_identity < 1
            or id(self) != self.receipt_identity
            or isinstance(self.launch_epoch_sequence, bool)
            or not isinstance(self.launch_epoch_sequence, int)
            or self.launch_epoch_sequence < 1
            or isinstance(self.fence_sequence, bool)
            or not isinstance(self.fence_sequence, int)
            or self.fence_sequence < 1
            or not self.stage.strip()
            or not _is_sha256(self.launch_generation_digest)
            or self.backend_type not in {"cpu", "mps", "cuda"}
            or not self.normalized_device.strip()
            or self.completion_scope
            != _completion_scope_for_backend(self.backend_type)
            or self.native_ops_identity < 1
            or not _is_sha256(self.native_abi_identity_digest)
            or self.creating_thread_identity < 1
            or (binding is None) != (self.subject_binding_identity is None)
            or (binding is None)
            != (self.subject_binding_generation_digest == "")
            or (binding is None) != (self.subject_kind == "")
            or (binding is None) != (self.subject_identity is None)
            or (binding is None)
            != (self.subject_generation_digest == "")
            or binding is not None
            and (
                id(binding) != self.subject_binding_identity
                or binding.generation_digest
                != self.subject_binding_generation_digest
                or binding.subject_kind != self.subject_kind
                or binding.subject_identity != self.subject_identity
                or binding.subject_generation_digest
                != self.subject_generation_digest
                or binding.capability_generation_digest
                != self.capability_generation_digest
                or binding.owner_generation_digest
                != self.owner_generation_digest
                or binding.capability_nonce_digest
                != self.capability_nonce_digest
            )
            or not self.completion_known
            or not self.completion_domain_drained
            or self.fence_call_count != 1
            or not isinstance(
                self._consumption,
                _PaperKineticCompletionReceiptConsumption,
            )
            or self.generation_digest != _receipt_digest(self)
        ):
            raise ValueError("sealed completion-fence receipt changed")

    @property
    def consumed(self) -> bool:
        self.assert_current()
        return self._consumption.consumed

    def assert_for(
        self,
        capability: PaperKineticSealedCompletionFence,
        *,
        stage: str,
        launch_generation_digest: str,
        fence_sequence: int,
        require_unconsumed: bool = True,
    ) -> None:
        """Prove exact capability, owner, stage, launch, and sequence binding."""

        self.assert_current()
        if type(capability) is not PaperKineticSealedCompletionFence:
            raise TypeError("completion receipt requires the exact sealed capability")
        capability.assert_current(require_healthy=False)
        if (
            not isinstance(stage, str)
            or not stage.strip()
            or not _is_sha256(launch_generation_digest)
            or isinstance(fence_sequence, bool)
            or not isinstance(fence_sequence, int)
            or fence_sequence < 1
            or self.capability_generation_digest != capability.generation_digest
            or self.owner_generation_digest != capability.owner_generation_digest
            or self.capability_nonce_digest != capability.capability_nonce_digest
            or self.launch_epoch_sequence != fence_sequence
            or self.stage != stage
            or self.launch_generation_digest != launch_generation_digest
            or self.fence_sequence != fence_sequence
            or self.backend_type != capability.backend_type
            or self.normalized_device != capability.normalized_device
            or self.completion_scope != capability.completion_scope
            or self.native_ops_identity != capability.native_ops_identity
            or self.native_abi_identity_digest
            != capability.native_abi_identity_digest
            or self.creating_thread_identity
            != capability.creating_thread_identity
            or require_unconsumed
            and self._consumption.consumed
        ):
            raise ValueError("sealed completion-fence receipt is foreign or consumed")

    def consume_for(
        self,
        capability: PaperKineticSealedCompletionFence,
        *,
        stage: str,
        launch_generation_digest: str,
        fence_sequence: int,
        consumer: str,
    ) -> None:
        """Consume this receipt exactly once before its non-throwing commit."""

        if self.subject_binding is not None:
            raise ValueError(
                "subject-bound completion receipt requires consume_for_subject"
            )
        if not isinstance(consumer, str) or not consumer.strip():
            raise ValueError("completion receipt consumer must be nonempty")
        self.assert_for(
            capability,
            stage=stage,
            launch_generation_digest=launch_generation_digest,
            fence_sequence=fence_sequence,
            require_unconsumed=True,
        )
        capability._consume_published_receipt(self, consumer=consumer)

    def assert_for_subject(
        self,
        capability: PaperKineticSealedCompletionFence,
        binding: PaperKineticCompletionSubjectBinding,
        *,
        subject: Any,
        require_unconsumed: bool = True,
    ) -> None:
        """Prove the internally recorded launch relation and exact subject."""

        self.assert_for(
            capability,
            stage=self.stage,
            launch_generation_digest=self.launch_generation_digest,
            fence_sequence=self.fence_sequence,
            require_unconsumed=require_unconsumed,
        )
        if (
            type(binding) is not PaperKineticCompletionSubjectBinding
            or self.subject_binding is not binding
            or id(binding) != self.subject_binding_identity
            or binding.generation_digest
            != self.subject_binding_generation_digest
        ):
            raise ValueError("completion receipt has a foreign subject binding")
        binding.assert_for(capability, subject=subject)

    def consume_for_subject(
        self,
        capability: PaperKineticSealedCompletionFence,
        binding: PaperKineticCompletionSubjectBinding,
        *,
        subject: Any,
        consumer: str,
    ) -> None:
        """Consume one exact subject-bound receipt without relation strings."""

        if not isinstance(consumer, str) or not consumer.strip():
            raise ValueError("completion receipt consumer must be nonempty")
        self.assert_for_subject(
            capability,
            binding,
            subject=subject,
            require_unconsumed=True,
        )
        capability._consume_published_receipt(self, consumer=consumer)


@dataclass(slots=True)
class PaperKineticSealedCompletionFence:
    """Backend/device/generation-bound fence with no injected callback.

    One object may settle multiple sequential launch epochs.  It cannot be
    invoked from a foreign thread, entered concurrently or recursively, or
    retried after a failed backend synchronization.  No receipt history is
    retained, so Python metadata remains O(1) in the number of launches.
    """

    native_ops: Any = field(repr=False)
    device: torch.device
    backend_provenance: str
    owner_generation_digest: str
    capability_nonce: str = field(repr=False)
    capability_nonce_digest: str
    capability_identity: int
    native_ops_identity: int
    native_abi_identity: tuple[tuple[str, int], ...]
    native_abi_identity_digest: str
    launch_domain_attestation_digest: str
    dispatch_anchor_identity: int | None
    dispatch_anchor_signature: tuple[object, ...]
    backend_type: str
    normalized_device: str
    completion_scope: str
    creating_thread_identity: int
    generation_digest: str
    runtime_status: str = CAPABILITY_STATUS
    fence_attempt_count: int = 0
    successful_fence_count: int = 0
    consumed_fence_count: int = 0
    last_consumed_fence_sequence: int = 0
    outstanding_receipt_sequence: int | None = None
    outstanding_receipt_identity: int | None = None
    outstanding_subject_binding_identity: int | None = None
    outstanding_subject_binding_generation_digest: str = ""
    registered_launch_epoch: PaperKineticCompletionLaunchEpoch | None = field(
        default=None,
        repr=False,
    )
    registered_launch_epoch_identity: int | None = None
    last_registered_launch_epoch_sequence: int = 0
    next_launch_epoch_sequence: int = 1
    next_fence_sequence: int = 1
    invocation_in_progress: bool = False
    completion_unknown: bool = False
    poisoned: bool = False
    failed_stage: str | None = None
    failed_launch_generation_digest: str | None = None
    failure: BaseException | None = field(default=None, repr=False)
    failure_traceback: Any = field(default=None, repr=False)
    provenance: str = CAPABILITY_PROVENANCE
    _invocation_lock: threading.Lock = field(
        default_factory=threading.Lock,
        repr=False,
    )
    _seal: object = field(default=None, repr=False)

    def __init_subclass__(cls, **kwargs: Any) -> NoReturn:
        raise TypeError("sealed completion-fence capabilities cannot be subclassed")

    def assert_current(
        self,
        *,
        native_ops: Any | None = None,
        device: torch.device | str | None = None,
        backend_provenance: str | None = None,
        owner_generation_digest: str | None = None,
        require_healthy: bool = True,
    ) -> None:
        """Validate the immutable binding plus the O(1) lifecycle ledger."""

        if type(self) is not PaperKineticSealedCompletionFence:
            raise TypeError("completion-fence capability has a foreign exact type")
        registered_epoch = self.registered_launch_epoch
        if registered_epoch is not None:
            registered_epoch.assert_current()
        expected_device = (
            self.normalized_device
            if device is None
            else _normalize_device(torch.device(device))
        )
        expected_backend = (
            self.backend_provenance
            if backend_provenance is None
            else backend_provenance
        )
        expected_owner = (
            self.owner_generation_digest
            if owner_generation_digest is None
            else owner_generation_digest
        )
        expected_ops = self.native_ops if native_ops is None else native_ops
        if (
            self._seal is not _CAPABILITY_SEAL
            or self.provenance != CAPABILITY_PROVENANCE
            or self.runtime_status != CAPABILITY_STATUS
            or id(self.native_ops) != self.native_ops_identity
            or expected_ops is not self.native_ops
            or _native_abi_identity(self.native_ops)
            != self.native_abi_identity
            or self.native_abi_identity_digest
            != _digest_parts(self.native_abi_identity)
            or not _is_sha256(self.launch_domain_attestation_digest)
            or self.backend_type == "mps"
            and (
                self.dispatch_anchor_identity is None
                or self.dispatch_anchor_identity < 1
                or not self.dispatch_anchor_signature
            )
            or self.backend_type != "mps"
            and (
                self.dispatch_anchor_identity is not None
                or self.dispatch_anchor_signature
            )
            or self.device.type != self.backend_type
            or self.backend_type not in {"cpu", "mps", "cuda"}
            or self.normalized_device != _normalize_device(self.device)
            or expected_device != self.normalized_device
            or self.completion_scope
            != _completion_scope_for_backend(self.backend_type)
            or not self.backend_provenance.strip()
            or expected_backend != self.backend_provenance
            or not _is_sha256(self.owner_generation_digest)
            or not _is_sha256(self.capability_nonce)
            or self.capability_nonce_digest
            != hashlib.sha256(self.capability_nonce.encode("ascii")).hexdigest()
            or self.capability_identity < 1
            or id(self) != self.capability_identity
            or expected_owner != self.owner_generation_digest
            or self.creating_thread_identity < 1
            or not isinstance(self._invocation_lock, _LOCK_TYPE)
            or isinstance(self.fence_attempt_count, bool)
            or not isinstance(self.fence_attempt_count, int)
            or self.fence_attempt_count < 0
            or isinstance(self.successful_fence_count, bool)
            or not isinstance(self.successful_fence_count, int)
            or self.successful_fence_count < 0
            or self.successful_fence_count > self.fence_attempt_count
            or isinstance(self.consumed_fence_count, bool)
            or not isinstance(self.consumed_fence_count, int)
            or self.consumed_fence_count < 0
            or self.consumed_fence_count > self.successful_fence_count
            or self.successful_fence_count - self.consumed_fence_count
            not in {0, 1}
            or self.last_consumed_fence_sequence != self.consumed_fence_count
            or self.outstanding_receipt_sequence
            != (
                self.successful_fence_count
                if self.successful_fence_count > self.consumed_fence_count
                else None
            )
            or (self.outstanding_receipt_sequence is None)
            != (self.outstanding_receipt_identity is None)
            or self.outstanding_receipt_identity is not None
            and self.outstanding_receipt_identity < 1
            or self.outstanding_receipt_sequence is None
            and (
                self.outstanding_subject_binding_identity is not None
                or self.outstanding_subject_binding_generation_digest != ""
            )
            or self.outstanding_receipt_sequence is not None
            and (
                (self.outstanding_subject_binding_identity is None)
                != (self.outstanding_subject_binding_generation_digest == "")
                or self.outstanding_subject_binding_identity is not None
                and (
                    self.outstanding_subject_binding_identity < 1
                    or not _is_sha256(
                        self.outstanding_subject_binding_generation_digest
                    )
                )
            )
            or self.last_registered_launch_epoch_sequence
            != self.next_launch_epoch_sequence - 1
            or self.next_launch_epoch_sequence
            != self.next_fence_sequence
            + (1 if self.registered_launch_epoch is not None else 0)
            or (self.registered_launch_epoch is None)
            != (self.registered_launch_epoch_identity is None)
            or self.registered_launch_epoch is not None
            and (
                id(self.registered_launch_epoch)
                != self.registered_launch_epoch_identity
                or self.registered_launch_epoch.fenced
                or self.registered_launch_epoch.capability_generation_digest
                != self.generation_digest
                or self.registered_launch_epoch.launch_epoch_sequence
                != self.next_fence_sequence
                or self.registered_launch_epoch.subject_binding is not None
                and (
                    self.registered_launch_epoch.subject_binding.capability_identity
                    != id(self)
                    or self.registered_launch_epoch.subject_binding.owner_generation_digest
                    != self.owner_generation_digest
                )
            )
            or isinstance(self.next_fence_sequence, bool)
            or not isinstance(self.next_fence_sequence, int)
            or self.next_fence_sequence != self.successful_fence_count + 1
            or not isinstance(self.invocation_in_progress, bool)
            or not isinstance(self.completion_unknown, bool)
            or not isinstance(self.poisoned, bool)
            or self.completion_unknown != (self.failure is not None)
            or self.poisoned != self.completion_unknown
            or self.completion_unknown
            != (
                self.failed_stage is not None
                and self.failed_launch_generation_digest is not None
                and self.failure_traceback is not None
            )
            or self.generation_digest != _capability_digest(self)
        ):
            raise ValueError("sealed completion-fence capability changed")
        if require_healthy and (self.poisoned or self.invocation_in_progress):
            raise RuntimeError("completion-fence capability is not healthy")

    def register_launch(
        self,
        *,
        stage: str,
        launch_generation_digest: str,
        subject_binding: PaperKineticCompletionSubjectBinding | None = None,
    ) -> PaperKineticCompletionLaunchEpoch:
        """Register exactly one monotone launch epoch before fencing it."""

        self.assert_current()
        if threading.get_ident() != self.creating_thread_identity:
            raise RuntimeError("completion launch registered from a foreign thread")
        if not self._invocation_lock.acquire(blocking=False):
            raise RuntimeError("completion launch registration cannot be re-entered")
        try:
            return self._register_launch_locked(
                stage=stage,
                launch_generation_digest=launch_generation_digest,
                subject_binding=subject_binding,
            )
        finally:
            self._invocation_lock.release()

    def _register_launch_locked(
        self,
        *,
        stage: str,
        launch_generation_digest: str,
        subject_binding: PaperKineticCompletionSubjectBinding | None,
    ) -> PaperKineticCompletionLaunchEpoch:
        """Lock-held launch registration implementation."""

        self.assert_current()
        if (
            self.outstanding_receipt_sequence is not None
            or self.registered_launch_epoch is not None
        ):
            raise RuntimeError(
                "previous launch epoch/receipt must settle before registration"
            )
        if not isinstance(stage, str) or not stage.strip():
            raise ValueError("completion launch stage must be nonempty")
        if not _is_sha256(launch_generation_digest):
            raise ValueError("launch generation must be a SHA-256 digest")
        if subject_binding is not None:
            if type(subject_binding) is not PaperKineticCompletionSubjectBinding:
                raise TypeError("launch subject binding has a foreign exact type")
            subject_binding.assert_current()
            if (
                subject_binding.capability_identity != id(self)
                or subject_binding.capability_generation_digest
                != self.generation_digest
                or subject_binding.owner_generation_digest
                != self.owner_generation_digest
                or subject_binding.capability_nonce_digest
                != self.capability_nonce_digest
            ):
                raise ValueError("launch subject binding is foreign")
        elif stage == "sample-completion" or self.backend_type != "cpu":
            raise ValueError(
                "sample or accelerator completion requires an exact prelaunch "
                "subject binding"
            )
        sequence = self.next_launch_epoch_sequence
        epoch = PaperKineticCompletionLaunchEpoch(
            capability_generation_digest=self.generation_digest,
            launch_epoch_sequence=sequence,
            stage=stage,
            launch_generation_digest=launch_generation_digest,
            generation_digest="",
            subject_binding=subject_binding,
            subject_binding_identity=(
                None if subject_binding is None else id(subject_binding)
            ),
            subject_binding_generation_digest=(
                ""
                if subject_binding is None
                else subject_binding.generation_digest
            ),
            _seal=_LAUNCH_EPOCH_SEAL,
        )
        epoch.generation_digest = _launch_epoch_digest(epoch)
        epoch.assert_current()
        self.registered_launch_epoch = epoch
        self.registered_launch_epoch_identity = id(epoch)
        self.last_registered_launch_epoch_sequence = sequence
        self.next_launch_epoch_sequence += 1
        return epoch

    def fence(
        self,
        launch_epoch: PaperKineticCompletionLaunchEpoch,
    ) -> PaperKineticCompletionFenceReceipt:
        """Perform one internally selected fence for an exact launch epoch.

        The receipt is completely constructed and validated before the backend
        call, then kept private until synchronization returns.  Consequently,
        no fallible receipt construction occurs after known completion.  If
        synchronization raises, this capability is permanently poisoned and
        the unpublished receipt is discarded.
        """

        self.assert_current()
        if type(launch_epoch) is not PaperKineticCompletionLaunchEpoch:
            raise TypeError("completion fence requires the exact launch epoch")
        launch_epoch.assert_current()
        if (
            self.registered_launch_epoch is not launch_epoch
            or id(launch_epoch) != self.registered_launch_epoch_identity
            or launch_epoch.capability_generation_digest != self.generation_digest
            or launch_epoch.launch_epoch_sequence != self.next_fence_sequence
            or launch_epoch.fenced
        ):
            raise ValueError("completion launch epoch is foreign, stale, or reused")
        stage = launch_epoch.stage
        launch_generation_digest = launch_epoch.launch_generation_digest
        subject_binding = launch_epoch.subject_binding
        if threading.get_ident() != self.creating_thread_identity:
            raise RuntimeError("completion fence invoked from a foreign thread")
        if not isinstance(stage, str) or not stage.strip():
            raise ValueError("completion fence stage must be nonempty")
        if not _is_sha256(launch_generation_digest):
            raise ValueError("launch generation must be a SHA-256 digest")
        if not self._invocation_lock.acquire(blocking=False):
            raise RuntimeError("completion fence cannot be re-entered")

        try:
            self.assert_current()
            sequence = self.next_fence_sequence
            provisional = PaperKineticCompletionFenceReceipt(
                capability_generation_digest=self.generation_digest,
                owner_generation_digest=self.owner_generation_digest,
                capability_nonce_digest=self.capability_nonce_digest,
                receipt_identity=0,
                launch_epoch_sequence=launch_epoch.launch_epoch_sequence,
                fence_sequence=sequence,
                stage=stage,
                launch_generation_digest=launch_generation_digest,
                backend_type=self.backend_type,
                normalized_device=self.normalized_device,
                completion_scope=self.completion_scope,
                native_ops_identity=self.native_ops_identity,
                native_abi_identity_digest=self.native_abi_identity_digest,
                creating_thread_identity=self.creating_thread_identity,
                subject_binding_identity=(
                    None if subject_binding is None else id(subject_binding)
                ),
                subject_binding_generation_digest=(
                    ""
                    if subject_binding is None
                    else subject_binding.generation_digest
                ),
                subject_kind=(
                    "" if subject_binding is None else subject_binding.subject_kind
                ),
                subject_identity=(
                    None
                    if subject_binding is None
                    else subject_binding.subject_identity
                ),
                subject_generation_digest=(
                    ""
                    if subject_binding is None
                    else subject_binding.subject_generation_digest
                ),
                generation_digest="",
                subject_binding=subject_binding,
                _consumption=_PaperKineticCompletionReceiptConsumption(
                    _seal=_RECEIPT_CONSUMPTION_SEAL,
                ),
                _seal=_RECEIPT_SEAL,
            )
            object.__setattr__(
                provisional,
                "receipt_identity",
                id(provisional),
            )
            object.__setattr__(
                provisional,
                "generation_digest",
                _receipt_digest(provisional),
            )
            receipt = provisional
            receipt.assert_current()
            receipt.assert_for(
                self,
                stage=stage,
                launch_generation_digest=launch_generation_digest,
                fence_sequence=sequence,
            )

            self.invocation_in_progress = True
            self.fence_attempt_count += 1
            try:
                self._synchronize_bound_device_wide()
            except BaseException as error:
                self.completion_unknown = True
                self.poisoned = True
                self.failed_stage = stage
                self.failed_launch_generation_digest = launch_generation_digest
                self.failure = error
                self.failure_traceback = error.__traceback__
                raise PaperKineticCompletionUnknownError(
                    "device-wide completion is unknown; retain every bound "
                    "launch root and restart the process"
                ) from error
            else:
                self.successful_fence_count += 1
                self.next_fence_sequence += 1
                self.outstanding_receipt_sequence = sequence
                self.outstanding_receipt_identity = id(receipt)
                self.outstanding_subject_binding_identity = (
                    receipt.subject_binding_identity
                )
                self.outstanding_subject_binding_generation_digest = (
                    receipt.subject_binding_generation_digest
                )
                launch_epoch.fenced = True
                self.registered_launch_epoch = None
                self.registered_launch_epoch_identity = None
                return receipt
            finally:
                self.invocation_in_progress = False
        finally:
            self._invocation_lock.release()

    def _consume_published_receipt(
        self,
        receipt: PaperKineticCompletionFenceReceipt,
        *,
        consumer: str,
    ) -> None:
        """Atomically advance the O(1) capability and receipt ledgers."""

        if threading.get_ident() != self.creating_thread_identity:
            raise RuntimeError("completion receipt consumed from a foreign thread")
        if not self._invocation_lock.acquire(blocking=False):
            raise RuntimeError("completion receipt consumption cannot be re-entered")
        try:
            self.assert_current(require_healthy=False)
            receipt.assert_for(
                self,
                stage=receipt.stage,
                launch_generation_digest=receipt.launch_generation_digest,
                fence_sequence=receipt.fence_sequence,
                require_unconsumed=True,
            )
            if (
                receipt.capability_generation_digest != self.generation_digest
                or receipt.capability_nonce_digest != self.capability_nonce_digest
                or self.outstanding_receipt_sequence != receipt.fence_sequence
                or self.outstanding_receipt_identity != id(receipt)
                or self.outstanding_subject_binding_identity
                != receipt.subject_binding_identity
                or self.outstanding_subject_binding_generation_digest
                != receipt.subject_binding_generation_digest
                or receipt.fence_sequence != self.last_consumed_fence_sequence + 1
                or receipt._consumption.consumed
                or not isinstance(consumer, str)
                or not consumer.strip()
            ):
                raise ValueError("completion receipt is not the outstanding sequence")
            receipt._consumption.consumer = consumer
            receipt._consumption.consumed = True
            self.consumed_fence_count += 1
            self.last_consumed_fence_sequence = receipt.fence_sequence
            self.outstanding_receipt_sequence = None
            self.outstanding_receipt_identity = None
            self.outstanding_subject_binding_identity = None
            self.outstanding_subject_binding_generation_digest = ""
        finally:
            self._invocation_lock.release()

    def _synchronize_bound_device_wide(self) -> None:
        """Use only module-owned backend synchronizers; never call user code."""

        if self.backend_type == "cpu":
            # The CPU contract admits only producers whose call-return boundary
            # means completion.  No asynchronous CPU producer is promotable.
            return
        if self.backend_type == "mps":
            if self.normalized_device != "mps:0":
                raise RuntimeError("MPS completion capability is bound to mps:0")
            if _TORCH_MPS_SYNCHRONIZE is None or not torch.backends.mps.is_available():
                raise RuntimeError("bound MPS completion backend is unavailable")
            returned = _TORCH_MPS_SYNCHRONIZE()
        elif self.backend_type == "cuda":
            if _TORCH_CUDA_SYNCHRONIZE is None or not torch.cuda.is_available():
                raise RuntimeError("bound CUDA completion backend is unavailable")
            returned = _TORCH_CUDA_SYNCHRONIZE(self.device)
        else:  # pragma: no cover - assert_current rejects this before dispatch.
            raise RuntimeError("unsupported completion backend")
        if returned is not None:
            raise TypeError("canonical device-wide synchronizer returned a value")


def prepare_paper_kinetic_sealed_completion_fence(
    native_ops: Any,
    *,
    device: torch.device | str,
    owner_generation_digest: str,
    dispatch_anchor: torch.Tensor | None = None,
) -> PaperKineticSealedCompletionFence:
    """Mint one CPU or canonically attested MPS completion capability.

    ``dispatch_anchor`` is not a caller assertion.  For MPS it must be the
    already-live tensor whose exact dispatch domain is queried by the compiled
    native module.  The constructor derives and seals the attestation itself;
    no caller boolean, callback, provenance string, or stream name is accepted.
    """

    resolved = torch.device(device)
    if resolved.type not in {"cpu", "mps", "cuda"}:
        raise ValueError("sealed completion capability supports cpu, mps, or cuda")
    if resolved.type == "cuda":
        _reject_unattested_accelerator(resolved)
    if resolved.type == "cpu" and dispatch_anchor is not None:
        raise ValueError("CPU completion capability does not accept a dispatch anchor")
    if not _is_sha256(owner_generation_digest):
        raise ValueError("owner generation must be a SHA-256 digest")
    abi_identity = _native_abi_identity(native_ops)
    if resolved.type == "mps":
        launch_domain_attestation_digest = _attest_canonical_mps_launch_domain(
            native_ops,
            dispatch_anchor=dispatch_anchor,
            abi_identity=abi_identity,
        )
        dispatch_anchor_identity = id(dispatch_anchor)
        dispatch_anchor_signature = _tensor_signature(dispatch_anchor)
    else:
        launch_domain_attestation_digest = _digest_parts(
            CAPABILITY_PROVENANCE,
            "cpu-call-return-launch-domain-v1",
            abi_identity,
        )
        dispatch_anchor_identity = None
        dispatch_anchor_signature = ()
    backend_provenance = _canonical_backend_provenance(
        resolved,
        abi_identity=abi_identity,
        launch_domain_attestation_digest=launch_domain_attestation_digest,
    )
    creating_thread = threading.get_ident()
    provisional = PaperKineticSealedCompletionFence(
        native_ops=native_ops,
        device=resolved,
        backend_provenance=backend_provenance,
        owner_generation_digest=owner_generation_digest,
        capability_nonce=secrets.token_hex(32),
        capability_nonce_digest="",
        capability_identity=0,
        native_ops_identity=id(native_ops),
        native_abi_identity=abi_identity,
        native_abi_identity_digest=_digest_parts(abi_identity),
        launch_domain_attestation_digest=launch_domain_attestation_digest,
        dispatch_anchor_identity=dispatch_anchor_identity,
        dispatch_anchor_signature=dispatch_anchor_signature,
        backend_type=resolved.type,
        normalized_device=_normalize_device(resolved),
        completion_scope=_completion_scope_for_backend(resolved.type),
        creating_thread_identity=creating_thread,
        generation_digest="",
        _seal=_CAPABILITY_SEAL,
    )
    provisional.capability_nonce_digest = hashlib.sha256(
        provisional.capability_nonce.encode("ascii")
    ).hexdigest()
    provisional.capability_identity = id(provisional)
    provisional.generation_digest = _capability_digest(provisional)
    capability = provisional
    capability.assert_current(
        native_ops=native_ops,
        device=resolved,
        backend_provenance=backend_provenance,
        owner_generation_digest=owner_generation_digest,
    )
    return capability


def prepare_paper_kinetic_completion_subject_binding(
    capability: PaperKineticSealedCompletionFence,
    subject: Any,
    *,
    kind: str,
    subject_generation_digest: str,
) -> PaperKineticCompletionSubjectBinding:
    """Bind one stable prelaunch slot to the exact completion capability."""

    if type(capability) is not PaperKineticSealedCompletionFence:
        raise TypeError("subject binding requires the exact sealed capability")
    capability.assert_current()
    if threading.get_ident() != capability.creating_thread_identity:
        raise RuntimeError("completion subject bound from a foreign thread")
    if subject is None:
        raise ValueError("completion subject must be a stable object")
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError("completion subject kind must be nonempty")
    if not _is_sha256(subject_generation_digest):
        raise ValueError("completion subject generation must be a SHA-256 digest")
    if (
        getattr(subject, "generation_digest", None)
        != subject_generation_digest
    ):
        raise ValueError("completion subject generation is foreign or stale")
    provisional = PaperKineticCompletionSubjectBinding(
        capability_identity=id(capability),
        capability_generation_digest=capability.generation_digest,
        owner_generation_digest=capability.owner_generation_digest,
        capability_nonce_digest=capability.capability_nonce_digest,
        subject_kind=kind,
        subject_identity=id(subject),
        subject_generation_digest=subject_generation_digest,
        binding_identity=0,
        generation_digest="",
        _subject=subject,
        _seal=_SUBJECT_BINDING_SEAL,
    )
    object.__setattr__(provisional, "binding_identity", id(provisional))
    object.__setattr__(
        provisional,
        "generation_digest",
        _subject_binding_digest(provisional),
    )
    binding = provisional
    binding.assert_for(capability, subject=subject)
    return binding


def _canonical_backend_provenance(
    device: torch.device,
    *,
    abi_identity: tuple[tuple[str, int], ...],
    launch_domain_attestation_digest: str,
) -> str:
    """Derive telemetry from the exact bound device/ABI, never caller text."""

    return (
        f"{_completion_scope_for_backend(device.type)}/"
        f"native-abi-sha256:{_digest_parts(abi_identity)}/"
        f"launch-domain-sha256:{launch_domain_attestation_digest}"
    )


def _attest_canonical_mps_launch_domain(
    native_ops: Any,
    *,
    dispatch_anchor: torch.Tensor | None,
    abi_identity: tuple[tuple[str, int], ...],
) -> str:
    """Derive the MPS launch-domain seal from the exact loaded extension."""

    if not isinstance(dispatch_anchor, torch.Tensor):
        _reject_unattested_accelerator(torch.device("mps"))
    if dispatch_anchor.device.type != "mps":
        raise ValueError("MPS completion dispatch anchor must be a live MPS tensor")
    if (
        getattr(native_ops, "__name__", None) != CANONICAL_MPS_NATIVE_OPS_MODULE
        or sys.modules.get(CANONICAL_MPS_NATIVE_OPS_MODULE) is not native_ops
    ):
        _reject_unattested_accelerator(torch.device("mps"))
    module_file = Path(str(getattr(native_ops, "__file__", ""))).resolve()
    if module_file.name != "ops.py" or module_file.parent.name != (
        "torch_world_foam_lane2_fused_slab"
    ):
        _reject_unattested_accelerator(torch.device("mps"))
    compiled_abi_attestation = getattr(
        native_ops,
        "assert_kinetic_memory_light_compiled_abi_registered",
        None,
    )
    resource_attestation = getattr(
        native_ops,
        "kinetic_memory_light_selected_kernel_resource_attestation",
        None,
    )
    if not callable(compiled_abi_attestation) or not callable(resource_attestation):
        _reject_unattested_accelerator(torch.device("mps"))
    returned = compiled_abi_attestation()
    if returned is not None:
        raise TypeError("canonical compiled-ABI attestation returned a value")
    report = resource_attestation(dispatch_anchor)
    kernels = getattr(report, "kernels", None)
    expected_operators = (
        "kinetic_precompiled_length_p0_lie_node_forward_into_launch_only_v1",
        "kinetic_ragged_p0_lie_sample_accumulate_loss_only_launch_only",
        "kinetic_precompiled_length_p0_lie_material_node_vjp_accumulate_launch_only",
    )
    if (
        type(kernels) is not tuple
        or tuple(getattr(kernel, "operator_name", None) for kernel in kernels)
        != expected_operators
        or getattr(report, "abi_namespace", None)
        != "world_foam_lane2_fused_slab_v0"
        or getattr(report, "selected_execution_path", None)
        != "kinetic_material_only"
        or getattr(report, "compiled_abi_schema_verified", None) is not True
        or getattr(report, "compiled_source_mtime_gate_passed", None) is not True
        or getattr(report, "optional_full_geometry_vjp_included", None) is not False
    ):
        raise RuntimeError("canonical selected-kernel launch-domain attestation changed")
    extension_path = getattr(native_ops, "_EXTENSION_LIBRARY_PATH", None)
    if not isinstance(extension_path, Path):
        raise RuntimeError("canonical native module did not expose its loaded extension")
    extension_path = extension_path.resolve()
    if not extension_path.is_file() or extension_path.suffix != ".so":
        raise RuntimeError("canonical native extension path is unavailable")
    source_paths = getattr(native_ops, "_compiled_source_paths", None)
    if not callable(source_paths):
        raise RuntimeError("canonical native module did not expose compiled sources")
    compiled_sources = tuple(Path(path).resolve() for path in source_paths())
    if not compiled_sources or any(not path.is_file() for path in compiled_sources):
        raise RuntimeError("canonical native compiled-source closure is unavailable")
    if _TORCH_MPS_SYNCHRONIZE is None or not torch.backends.mps.is_available():
        raise RuntimeError("attested MPS completion backend is unavailable")
    synchronized = _TORCH_MPS_SYNCHRONIZE()
    if synchronized is not None:
        raise TypeError("canonical MPS attestation synchronizer returned a value")
    return _digest_parts(
        MPS_LAUNCH_DOMAIN_ATTESTATION,
        CANONICAL_MPS_NATIVE_OPS_MODULE,
        str(module_file),
        _file_sha256(module_file),
        str(extension_path),
        _file_sha256(extension_path),
        tuple((str(path), _file_sha256(path)) for path in compiled_sources),
        abi_identity,
        tuple(
            (
                kernel.operator_name,
                kernel.metal_function_name,
                kernel.max_threads_per_threadgroup,
                kernel.thread_execution_width,
                kernel.static_threadgroup_memory_length_bytes,
            )
            for kernel in kernels
        ),
        _tensor_signature(dispatch_anchor),
    )


def _reject_unattested_accelerator(device: torch.device) -> NoReturn:
    """Closed promotion gate; source presence is not runtime attestation."""

    raise RuntimeError(
        f"{device.type.upper()} completion capability is source-defined but "
        "not promotable until the canonical native module proves its dispatch "
        "domain and safe-host runtime evidence validates the device-wide fence"
    )


def _native_abi_identity(native_ops: Any) -> tuple[tuple[str, int], ...]:
    identity: list[tuple[str, int]] = []
    for name in LAZY_NATIVE_REQUIRED_OP_NAMES:
        operation = getattr(native_ops, name, None)
        if not callable(operation):
            raise TypeError(f"native ops object is missing callable {name}")
        implementation = getattr(operation, "__func__", operation)
        identity.append((name, id(implementation)))
    return tuple(identity)


def _normalize_device(device: torch.device) -> str:
    if device.type == "cpu":
        if device.index not in {None, 0}:
            raise ValueError("CPU completion capability supports only cpu or cpu:0")
        return "cpu"
    if device.type == "mps":
        if device.index not in {None, 0}:
            raise ValueError("MPS completion capability supports only mps or mps:0")
        return "mps:0"
    if device.type == "cuda":
        if device.index is None:
            raise ValueError("CUDA completion capability requires an explicit device index")
        if device.index < 0:
            raise ValueError("CUDA completion capability device index must be nonnegative")
        return f"cuda:{device.index}"
    raise ValueError("unsupported completion device")


def _completion_scope_for_backend(backend_type: str) -> str:
    scopes = {
        "cpu": CPU_CALL_RETURN_SCOPE,
        "mps": MPS_DEVICE_WIDE_SCOPE,
        "cuda": CUDA_DEVICE_WIDE_SCOPE,
    }
    try:
        return scopes[backend_type]
    except KeyError as error:
        raise ValueError("unsupported completion backend") from error


def _capability_digest(capability: PaperKineticSealedCompletionFence) -> str:
    return _digest_parts(
        CAPABILITY_PROVENANCE,
        capability.runtime_status,
        capability.backend_provenance,
        capability.owner_generation_digest,
        capability.capability_nonce_digest,
        capability.capability_identity,
        capability.native_ops_identity,
        capability.native_abi_identity,
        capability.native_abi_identity_digest,
        capability.launch_domain_attestation_digest,
        capability.dispatch_anchor_identity,
        capability.dispatch_anchor_signature,
        capability.backend_type,
        capability.normalized_device,
        capability.completion_scope,
        capability.creating_thread_identity,
    )


def _receipt_digest(receipt: PaperKineticCompletionFenceReceipt) -> str:
    return _digest_parts(
        CAPABILITY_PROVENANCE,
        receipt.runtime_status,
        receipt.capability_generation_digest,
        receipt.owner_generation_digest,
        receipt.capability_nonce_digest,
        receipt.receipt_identity,
        receipt.launch_epoch_sequence,
        receipt.fence_sequence,
        receipt.stage,
        receipt.launch_generation_digest,
        receipt.backend_type,
        receipt.normalized_device,
        receipt.completion_scope,
        receipt.native_ops_identity,
        receipt.native_abi_identity_digest,
        receipt.creating_thread_identity,
        receipt.subject_binding_identity,
        receipt.subject_binding_generation_digest,
        receipt.subject_kind,
        receipt.subject_identity,
        receipt.subject_generation_digest,
        receipt.completion_known,
        receipt.completion_domain_drained,
        receipt.fence_call_count,
    )


def _launch_epoch_digest(epoch: PaperKineticCompletionLaunchEpoch) -> str:
    return _digest_parts(
        CAPABILITY_PROVENANCE,
        "launch-epoch-v1",
        epoch.capability_generation_digest,
        epoch.launch_epoch_sequence,
        epoch.stage,
        epoch.launch_generation_digest,
        epoch.subject_binding_identity,
        epoch.subject_binding_generation_digest,
    )


def _subject_binding_digest(
    binding: PaperKineticCompletionSubjectBinding,
) -> str:
    return _digest_parts(
        CAPABILITY_PROVENANCE,
        "completion-subject-binding-v1",
        binding.capability_identity,
        binding.capability_generation_digest,
        binding.owner_generation_digest,
        binding.capability_nonce_digest,
        binding.subject_kind,
        binding.subject_identity,
        binding.subject_generation_digest,
        binding.binding_identity,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_signature(tensor: torch.Tensor) -> tuple[object, ...]:
    return (
        id(tensor),
        str(tensor.device),
        str(tensor.dtype),
        tuple(int(value) for value in tensor.shape),
        tuple(int(value) for value in tensor.stride()),
        int(tensor.storage_offset()),
        int(tensor.untyped_storage().data_ptr()),
    )


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
    "CAPABILITY_PROVENANCE",
    "CAPABILITY_STATUS",
    "CPU_CALL_RETURN_SCOPE",
    "CUDA_DEVICE_WIDE_SCOPE",
    "LAZY_NATIVE_REQUIRED_OP_NAMES",
    "MPS_DEVICE_WIDE_SCOPE",
    "PaperKineticCompletionFenceReceipt",
    "PaperKineticCompletionLaunchEpoch",
    "PaperKineticCompletionSubjectBinding",
    "PaperKineticCompletionUnknownError",
    "PaperKineticSealedCompletionFence",
    "prepare_paper_kinetic_completion_subject_binding",
    "prepare_paper_kinetic_sealed_completion_fence",
)
