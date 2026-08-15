"""One-bundle-at-a-time kinetic WorldFoam material optimizer step.

This module composes the previously separate production-shaped seams:

* a dataset-bound lazy program-bundle provider;
* one droppable native runtime lane per spatial bundle;
* frame-major sealed selected-pixel/sample launches;
* exactly one ordered-word forward and material VJP per active native block;
* one caller-owned global material bar and one optimizer authorization.

The coordinator, rather than the low-level executor, owns the logical-step
contract.  It zeroes and binds all mutable buffers, gathers compact material
rows through the runtime's sealed source-site map, proves canonical bundle and
track coverage, and prevents a completed step index from being reopened.  A
failure with proven device completion clears the partial global bar.  An
unknown sample/reverse/lane completion instead retains one bounded lifetime on
the poisoned trainer without enqueueing cleanup.  A failure inside the
optimizer callback also poisons the trainer because an external optimizer may
already have mutated parameters.

The current data provider intentionally supports bounded sparse sampled
observations only.  Dense-F execution still needs a replayable observation
source that decouples one compiled track bundle from arbitrarily many sample
chunks.  Native build, accelerator parity, and allocator peaks remain
unverified.  Two-phase lane and sample-transfer lifetimes now preserve every
source-visible predecessor inside pre-lane bundle construction, the native lane,
the compact-material gather, and the sparse sample stream.  Accelerator
execution remains fail-closed until a canonical device-specific fence capability
exists, the caller-owned forward-into ABI is rebuilt and parity-verified, and
native allocator/runtime lifetimes are measured; this file is the CPU/source
integration boundary.
"""

from __future__ import annotations

import hashlib
import math
import threading
from collections.abc import Callable, Iterable, Mapping
from contextlib import closing
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, NoReturn

import torch
from kinetic_native_lazy_bundle_lane import (
    estimate_paper_kinetic_native_lazy_bundle_lane_resident_bytes,
    materialize_paper_kinetic_native_lazy_bundle_lane,
    prepare_paper_kinetic_native_lazy_bundle_lane_construction_lifetime,
)
from kinetic_native_material_step_executor import (
    KineticNativeMaterialStepSession,
    KineticNativeMaterialStepWorldToken,
    KineticNativeNodeForwardIntoLifetime,
    KineticNativePendingSampleLaunchCompletion,
    KineticNativeSampleLaunchLifetime,
    prepare_kinetic_native_node_forward_into_lifetime,
)
from kinetic_sealed_completion_fence import (
    CAPABILITY_PROVENANCE,
    LAZY_NATIVE_REQUIRED_OP_NAMES,
    PaperKineticCompletionFenceReceipt,
    PaperKineticCompletionLaunchEpoch,
    PaperKineticCompletionSubjectBinding,
    PaperKineticCompletionUnknownError,
    PaperKineticSealedCompletionFence,
    prepare_paper_kinetic_completion_subject_binding,
    prepare_paper_kinetic_sealed_completion_fence,
)
from paper_kinetic_lazy_program_bundles import (
    PaperKineticLazyBundleConstructionLifetimeSlot,
    PaperKineticLazyProgramBundle,
    PaperKineticLazyProgramBundleProvider,
    PaperKineticObservation,
    prepare_paper_kinetic_lazy_bundle_construction_lifetime_slot,
)
from paper_kinetic_sparse_sample_blocks import (
    PaperKineticSparseSampleBlockStream,
    PaperKineticSparseSampleMaterializationLifetime,
    PaperKineticSparseSamplePlan,
    iter_paper_kinetic_sparse_sample_blocks,
    prepare_paper_kinetic_sparse_sample_plan,
)
from paper_kinetic_ragged_sample_plan import PaperKineticRowRaggedSampleBlock
from paper_kinetic_step_target_frame_cache import (
    PaperKineticStepTargetFrameCache,
    prepare_paper_kinetic_step_target_frame_cache,
)

STEP_PROVENANCE = "paper-kinetic-lazy-native-material-step-v4"
STEP_STATUS = "cpu_contract_only/accelerator_async_quarantine_required"
TARGET_FRAME_STREAM_ONCE = "one_frame_streaming"
TARGET_FRAME_STEP_CACHE = "step_decode_once_bounded_cache"
TARGET_FRAME_ACCESS_MODES = frozenset((TARGET_FRAME_STREAM_ONCE, TARGET_FRAME_STEP_CACHE))

_TRAINER_STATE_SEAL = object()
_RESULT_SEAL = object()
_COMPACT_GATHER_LIFETIME_SEAL = object()
_TOP_LEVEL_DEVICE_TRANSACTION_LIFETIME_SEAL = object()
_SAMPLE_COMPOSITE_SETTLEMENT_SLOT_SEAL = object()
_ACCELERATOR_STAGE_SETTLEMENT_SLOT_SEAL = object()
_LOCK_TYPE = type(threading.Lock())

SAMPLE_COMPOSITE_SUBJECT_KIND = "kinetic-sample-composite-settlement-v1"
ACCELERATOR_STAGE_SUBJECT_KIND = "kinetic-general-stage-settlement-v1"

# These roots are exact destinations of work covered by the registered epoch.
# Their identity/layout/storage must remain fixed, while their tensor version is
# expected to advance when the native launch writes the destination.  Every
# other tensor root remains version-bound.  Keep this role list narrow: it is
# the capability boundary between "rooted writable output" and immutable input.
_WRITABLE_SETTLEMENT_TENSOR_ROLES = frozenset(
    {
        "global_bar",
        "global_loss",
        "loss",
        "cone_diagnostic",
        "forward_node_chart_out",
        "active_block_grad_node_chart",
        "active_block_loss",
        "existing_active_block_grad_node_chart",
        "existing_active_block_loss",
    }
)


@dataclass
class _AcceleratorStageSettlementSlot:
    """One reusable exact subject for every non-sample accelerator epoch.

    A device-wide fence proves completion only for the dispatch domain.  This
    stable state-owned slot additionally roots the exact Python/tensor subjects
    of the current epoch until its one receipt is consumed.  The slot is O(1)
    in queue depth and is reused only after an assignment-only release.
    """

    owner_generation_digest: str
    capability: PaperKineticSealedCompletionFence = field(repr=False)
    slot_identity: int
    generation_digest: str
    subject_binding: PaperKineticCompletionSubjectBinding | None = field(
        default=None,
        repr=False,
    )
    stage: str = ""
    launch_generation_digest: str = ""
    root_roles: tuple[str, ...] = ()
    root_identities: tuple[int, ...] = ()
    root_signatures: tuple[tuple[object, ...], ...] = field(
        default=(),
        repr=False,
    )
    roots: tuple[Any, ...] = field(default=(), repr=False)
    launch_epoch: PaperKineticCompletionLaunchEpoch | None = field(
        default=None,
        repr=False,
    )
    phase: str = "installed"
    _seal: object = field(default=None, repr=False)

    def assert_current(self) -> None:
        if type(self) is not _AcceleratorStageSettlementSlot:
            raise TypeError("accelerator stage settlement slot has a foreign type")
        if (
            self._seal is not _ACCELERATOR_STAGE_SETTLEMENT_SLOT_SEAL
            or not _is_sha256(self.owner_generation_digest)
            or self.slot_identity < 1
            or id(self) != self.slot_identity
            or self.generation_digest != _accelerator_stage_slot_digest(self)
            or self.capability.backend_type == "cpu"
            or self.phase not in {"installed", "bound", "armed", "registered"}
        ):
            raise ValueError("accelerator stage settlement slot changed")
        self.capability.assert_current()
        binding_expected = self.phase != "installed"
        if binding_expected != (
            type(self.subject_binding) is PaperKineticCompletionSubjectBinding
        ):
            raise ValueError("accelerator stage subject publication changed")
        if self.subject_binding is not None:
            self.subject_binding.assert_for(self.capability, subject=self)
            if self.subject_binding.subject_kind != ACCELERATOR_STAGE_SUBJECT_KIND:
                raise ValueError("accelerator stage subject kind changed")
        active = self.phase in {"armed", "registered"}
        if active != bool(self.stage):
            raise ValueError("accelerator stage publication changed")
        if active:
            if (
                not _is_sha256(self.launch_generation_digest)
                or not self.root_roles
                or len(
                    {
                        len(self.root_roles),
                        len(self.root_identities),
                        len(self.root_signatures),
                        len(self.roots),
                    }
                )
                != 1
                or len(set(self.root_roles)) != len(self.root_roles)
                or any(not role.strip() for role in self.root_roles)
                or tuple(id(root) for root in self.roots) != self.root_identities
                or tuple(
                    _settlement_root_signature_for_role(role, root)
                    for role, root in zip(
                        self.root_roles,
                        self.roots,
                        strict=True,
                    )
                )
                != self.root_signatures
            ):
                raise ValueError("accelerator stage roots changed")
        elif any(
            (
                self.stage,
                self.launch_generation_digest,
                self.root_roles,
                self.root_identities,
                self.root_signatures,
                self.roots,
            )
        ):
            raise ValueError("idle accelerator stage retained roots")
        epoch_expected = self.phase == "registered"
        if epoch_expected != (
            type(self.launch_epoch) is PaperKineticCompletionLaunchEpoch
        ):
            raise ValueError("accelerator stage epoch publication changed")
        if self.launch_epoch is not None and (
            self.launch_epoch.subject_binding is not self.subject_binding
            or self.launch_epoch.stage != self.stage
            or self.launch_epoch.launch_generation_digest
            != self.launch_generation_digest
        ):
            raise ValueError("accelerator stage epoch changed")

    def bind_subject(self) -> None:
        self.assert_current()
        if self.phase != "installed":
            raise ValueError("accelerator stage subject was already bound")
        binding = prepare_paper_kinetic_completion_subject_binding(
            self.capability,
            self,
            kind=ACCELERATOR_STAGE_SUBJECT_KIND,
            subject_generation_digest=self.generation_digest,
        )
        self.subject_binding = binding
        self.phase = "bound"
        self.assert_current()

    def arm(
        self,
        *,
        stage: str,
        launch_generation_digest: str,
        roots: tuple[tuple[str, Any], ...],
    ) -> None:
        self.assert_current()
        if self.phase != "bound":
            raise ValueError("accelerator stage slot is already active")
        if not isinstance(stage, str) or not stage.strip():
            raise ValueError("accelerator stage must be nonempty")
        if not _is_sha256(launch_generation_digest):
            raise ValueError("accelerator stage launch digest must be SHA-256")
        self.stage = stage
        self.launch_generation_digest = launch_generation_digest
        self._replace_roots(roots)
        self.phase = "armed"
        self.assert_current()

    def extend_roots(self, roots: tuple[tuple[str, Any], ...]) -> None:
        self.assert_current()
        if self.phase not in {"armed", "registered"}:
            raise ValueError("accelerator stage roots extended while idle")
        self._replace_roots(
            tuple(zip(self.root_roles, self.roots, strict=True)) + roots
        )
        self.assert_current()

    def register_epoch(self, epoch: PaperKineticCompletionLaunchEpoch) -> None:
        self.assert_current()
        if self.phase != "armed" or epoch.subject_binding is not self.subject_binding:
            raise ValueError("accelerator stage epoch is foreign")
        self.launch_epoch = epoch
        self.phase = "registered"
        self.assert_current()

    def consume_receipt(
        self,
        receipt: PaperKineticCompletionFenceReceipt,
        *,
        consumer: str,
    ) -> None:
        self.assert_current()
        if self.phase != "registered" or self.subject_binding is None:
            raise ValueError("accelerator stage has no registered epoch")
        receipt.consume_for_subject(
            self.capability,
            self.subject_binding,
            subject=self,
            consumer=consumer,
        )
        self.stage = ""
        self.launch_generation_digest = ""
        self.root_roles = ()
        self.root_identities = ()
        self.root_signatures = ()
        self.roots = ()
        self.launch_epoch = None
        self.phase = "bound"
        self.assert_current()

    def _replace_roots(self, roots: tuple[tuple[str, Any], ...]) -> None:
        if not roots:
            raise ValueError("accelerator stage requires at least one rooted subject")
        roles = tuple(role for role, _ in roots)
        references = tuple(root for _, root in roots)
        if (
            len(set(roles)) != len(roles)
            or any(not isinstance(role, str) or not role.strip() for role in roles)
            or any(root is None for root in references)
        ):
            raise ValueError("accelerator stage roots are empty or duplicated")
        self.root_roles = roles
        self.roots = references
        self.root_identities = tuple(id(root) for root in references)
        self.root_signatures = tuple(
            _settlement_root_signature_for_role(role, root)
            for role, root in zip(roles, references, strict=True)
        )


def _prepare_accelerator_stage_settlement_slot(
    capability: PaperKineticSealedCompletionFence,
    *,
    owner_generation_digest: str,
) -> _AcceleratorStageSettlementSlot | None:
    if capability.backend_type == "cpu":
        return None
    provisional = _AcceleratorStageSettlementSlot(
        owner_generation_digest=owner_generation_digest,
        capability=capability,
        slot_identity=0,
        generation_digest="",
        _seal=_ACCELERATOR_STAGE_SETTLEMENT_SLOT_SEAL,
    )
    provisional.slot_identity = id(provisional)
    provisional.generation_digest = _accelerator_stage_slot_digest(provisional)
    provisional.assert_current()
    provisional.bind_subject()
    return provisional


def _register_general_completion_epoch(
    capability: PaperKineticSealedCompletionFence,
    settlement_slot: _AcceleratorStageSettlementSlot | None,
    *,
    stage: str,
    launch_generation_digest: str,
    roots: tuple[tuple[str, Any], ...],
) -> PaperKineticCompletionLaunchEpoch:
    if settlement_slot is None:
        return capability.register_launch(
            stage=stage,
            launch_generation_digest=launch_generation_digest,
        )
    settlement_slot.arm(
        stage=stage,
        launch_generation_digest=launch_generation_digest,
        roots=roots,
    )
    epoch = capability.register_launch(
        stage=stage,
        launch_generation_digest=launch_generation_digest,
        subject_binding=settlement_slot.subject_binding,
    )
    settlement_slot.register_epoch(epoch)
    return epoch


def _extend_general_completion_roots(
    settlement_slot: _AcceleratorStageSettlementSlot | None,
    roots: tuple[tuple[str, Any], ...],
) -> None:
    if settlement_slot is not None:
        settlement_slot.extend_roots(roots)


def _consume_general_completion_receipt(
    capability: PaperKineticSealedCompletionFence,
    settlement_slot: _AcceleratorStageSettlementSlot | None,
    receipt: PaperKineticCompletionFenceReceipt,
    *,
    consumer: str,
) -> None:
    if settlement_slot is None:
        receipt.consume_for(
            capability,
            stage=receipt.stage,
            launch_generation_digest=receipt.launch_generation_digest,
            fence_sequence=receipt.fence_sequence,
            consumer=consumer,
        )
        return
    settlement_slot.consume_receipt(receipt, consumer=consumer)


@dataclass
class _SampleCompositeSettlementSlot:
    """Stable prelaunch subject for one transfer + native sample settlement."""

    step_generation_id: str
    bundle_generation_digest: str
    plan_generation_digest: str
    session_generation_id: str
    session_identity: int
    plan_identity: int
    stream_identity: int
    launch_ordinal: int
    covered_sample_count_before_launch: int
    slot_identity: int
    generation_digest: str
    session: KineticNativeMaterialStepSession | None = field(
        default=None,
        repr=False,
    )
    plan: PaperKineticSparseSamplePlan | None = field(
        default=None,
        repr=False,
    )
    stream: PaperKineticSparseSampleBlockStream | None = field(
        default=None,
        repr=False,
    )
    subject_binding: PaperKineticCompletionSubjectBinding | None = field(
        default=None,
        repr=False,
    )
    launch_epoch: PaperKineticCompletionLaunchEpoch | None = field(
        default=None,
        repr=False,
    )
    transfer_lifetime: (
        PaperKineticSparseSampleMaterializationLifetime | None
    ) = field(default=None, repr=False)
    sample_block: PaperKineticRowRaggedSampleBlock | None = field(
        default=None,
        repr=False,
    )
    executor_lifetime: KineticNativeSampleLaunchLifetime | None = field(
        default=None,
        repr=False,
    )
    pending_completion: KineticNativePendingSampleLaunchCompletion | None = field(
        default=None,
        repr=False,
    )
    sample_block_identity: int | None = None
    sample_block_generation_digest: str = ""
    transfer_lifetime_identity: int | None = None
    transfer_dispatch_generation_digest: str = ""
    executor_lifetime_identity: int | None = None
    executor_lifetime_generation_digest: str = ""
    pending_completion_identity: int | None = None
    pending_completion_generation_digest: str = ""
    additional_root_roles: tuple[str, ...] = ()
    additional_root_identities: tuple[int, ...] = ()
    additional_root_signatures: tuple[tuple[object, ...], ...] = field(
        default=(),
        repr=False,
    )
    additional_roots: tuple[Any, ...] = field(default=(), repr=False)
    phase: str = "installed"
    _seal: object = field(default=None, repr=False)

    def assert_current(self) -> None:
        self._assert_integrity(allow_completion_unknown=False)

    def assert_quarantine_retained(self) -> None:
        """Validate the exact slot without weakening its live-path contract."""

        self._assert_integrity(allow_completion_unknown=True)

    def _assert_integrity(self, *, allow_completion_unknown: bool) -> None:
        if type(self) is not _SampleCompositeSettlementSlot:
            raise TypeError("sample composite slot has a foreign exact type")
        if (
            self._seal is not _SAMPLE_COMPOSITE_SETTLEMENT_SLOT_SEAL
            or not self.step_generation_id.strip()
            or not _is_sha256(self.bundle_generation_digest)
            or not _is_sha256(self.plan_generation_digest)
            or not self.session_generation_id.strip()
            or self.session_identity < 1
            or self.plan_identity < 1
            or self.stream_identity < 1
            or self.launch_ordinal < 0
            or self.covered_sample_count_before_launch < 0
            or self.slot_identity < 1
            or id(self) != self.slot_identity
            or self.generation_digest != _sample_composite_slot_digest(self)
            or self.phase
            not in {
                "installed",
                "bound",
                "registered",
                "materialized",
                "launched",
                "pending",
                "committed",
            }
        ):
            raise ValueError("sample composite settlement slot changed")
        if self.phase == "committed":
            if any(
                value is not None
                for value in (
                    self.stream,
                    self.session,
                    self.plan,
                    self.subject_binding,
                    self.launch_epoch,
                    self.transfer_lifetime,
                    self.sample_block,
                    self.executor_lifetime,
                    self.pending_completion,
                )
            ):
                raise ValueError("committed sample composite slot retained roots")
            if any(
                (
                    self.additional_root_roles,
                    self.additional_root_identities,
                    self.additional_root_signatures,
                    self.additional_roots,
                )
            ):
                raise ValueError("committed sample composite slot retained additions")
            return
        if (
            type(self.session) is not KineticNativeMaterialStepSession
            or id(self.session) != self.session_identity
            or self.session.generation_id != self.session_generation_id
            or type(self.plan) is not PaperKineticSparseSamplePlan
            or id(self.plan) != self.plan_identity
            or self.plan.generation_digest != self.plan_generation_digest
            or self.plan.bundle.generation_digest
            != self.bundle_generation_digest
            or type(self.stream) is not PaperKineticSparseSampleBlockStream
            or id(self.stream) != self.stream_identity
            or self.stream.plan is not self.plan
            or not self.stream.require_explicit_transfer_settlement
        ):
            raise ValueError("sample composite stream identity changed")
        binding_expected = self.phase not in {"installed"}
        epoch_expected = self.phase not in {"installed", "bound"}
        materialized_expected = self.phase in {
            "materialized",
            "launched",
            "pending",
        }
        launched_expected = self.phase in {"launched", "pending"}
        pending_expected = self.phase == "pending"
        if binding_expected != (
            type(self.subject_binding)
            is PaperKineticCompletionSubjectBinding
        ):
            raise ValueError("sample composite subject publication changed")
        if epoch_expected != (
            type(self.launch_epoch) is PaperKineticCompletionLaunchEpoch
        ):
            raise ValueError("sample composite epoch publication changed")
        if materialized_expected != isinstance(
            self.sample_block,
            PaperKineticRowRaggedSampleBlock,
        ) or materialized_expected != isinstance(
            self.transfer_lifetime,
            PaperKineticSparseSampleMaterializationLifetime,
        ):
            raise ValueError("sample composite materialization changed")
        if launched_expected != (
            type(self.executor_lifetime)
            is KineticNativeSampleLaunchLifetime
        ):
            raise ValueError("sample composite executor lifetime changed")
        if pending_expected != (
            type(self.pending_completion)
            is KineticNativePendingSampleLaunchCompletion
        ):
            raise ValueError("sample composite pending completion changed")
        if self.subject_binding is not None:
            self.subject_binding.assert_current()
            if (
                self.subject_binding.subject_kind
                != SAMPLE_COMPOSITE_SUBJECT_KIND
                or self.subject_binding.subject_identity != id(self)
                or self.subject_binding.subject_generation_digest
                != self.generation_digest
            ):
                raise ValueError("sample composite subject binding changed")
        if self.launch_epoch is not None:
            self.launch_epoch.assert_current()
            if self.launch_epoch.subject_binding is not self.subject_binding:
                raise ValueError("sample composite epoch subject changed")
        if materialized_expected:
            if (
                id(self.sample_block) != self.sample_block_identity
                or self.sample_block.generation_digest
                != self.sample_block_generation_digest
                or id(self.transfer_lifetime)
                != self.transfer_lifetime_identity
                or self.transfer_lifetime.dispatch_generation_digest
                != self.transfer_dispatch_generation_digest
            ):
                raise ValueError("sample composite materialized subject changed")
            self.transfer_lifetime.assert_releasable_after_consumed_receipt()
            self.sample_block.assert_warm_layout()
        if launched_expected and (
            id(self.executor_lifetime) != self.executor_lifetime_identity
            or getattr(self.executor_lifetime, "generation_digest", None)
            != self.executor_lifetime_generation_digest
            or self.executor_lifetime.sample_block is not self.sample_block
            or self.executor_lifetime._session_identity
            != self.session_identity
        ):
            raise ValueError("sample composite executor subject changed")
        if launched_expected:
            if (
                allow_completion_unknown
                and self.executor_lifetime.phase == "completion_unknown"
                and self.executor_lifetime.completion_unknown
            ):
                self.executor_lifetime.assert_retained(self.session)
            else:
                self.executor_lifetime.assert_current(self.session)
        if pending_expected and (
            id(self.pending_completion) != self.pending_completion_identity
            or getattr(self.pending_completion, "generation_digest", None)
            != self.pending_completion_generation_digest
            or self.pending_completion._session_identity
            != self.session_identity
            or self.pending_completion._sample_lifetime
            is not self.executor_lifetime
            or self.pending_completion.subject_binding
            is not self.subject_binding
            or self.pending_completion.subject_identity != id(self)
            or self.pending_completion.subject_generation_digest
            != self.generation_digest
            or self.pending_completion._launch_epoch is not self.launch_epoch
        ):
            raise ValueError("sample composite pending subject changed")
        if (
            len(
                {
                    len(self.additional_root_roles),
                    len(self.additional_root_identities),
                    len(self.additional_root_signatures),
                    len(self.additional_roots),
                }
            )
            != 1
            or len(set(self.additional_root_roles))
            != len(self.additional_root_roles)
            or tuple(id(root) for root in self.additional_roots)
            != self.additional_root_identities
            or tuple(
                _settlement_root_signature_for_role(role, root)
                for role, root in zip(
                    self.additional_root_roles,
                    self.additional_roots,
                    strict=True,
                )
            )
            != self.additional_root_signatures
        ):
            raise ValueError("sample composite additional roots changed")

    def bind_subject(
        self,
        binding: PaperKineticCompletionSubjectBinding,
    ) -> None:
        self.assert_current()
        if self.phase != "installed":
            raise ValueError("sample composite subject was already bound")
        binding.assert_current()
        if (
            binding.subject_kind != SAMPLE_COMPOSITE_SUBJECT_KIND
            or binding.subject_identity != id(self)
            or binding.subject_generation_digest != self.generation_digest
        ):
            raise ValueError("sample composite subject binding is foreign")
        self.subject_binding = binding
        self.phase = "bound"
        self.assert_current()

    def register_epoch(self, epoch: PaperKineticCompletionLaunchEpoch) -> None:
        self.assert_current()
        if self.phase != "bound" or epoch.subject_binding is not self.subject_binding:
            raise ValueError("sample composite launch epoch is foreign")
        self.launch_epoch = epoch
        self.phase = "registered"
        self.assert_current()

    def publish_materialization(
        self,
        sample_block: PaperKineticRowRaggedSampleBlock,
        transfer_lifetime: PaperKineticSparseSampleMaterializationLifetime,
    ) -> None:
        self.assert_current()
        if self.phase != "registered":
            raise ValueError("sample composite materialized out of order")
        self.stream.assert_active_releasable_after_consumed_receipt(
            sample_block,
            expected_lifetime=transfer_lifetime,
        )
        self.sample_block = sample_block
        self.sample_block_identity = id(sample_block)
        self.sample_block_generation_digest = sample_block.generation_digest
        self.transfer_lifetime = transfer_lifetime
        self.transfer_lifetime_identity = id(transfer_lifetime)
        self.transfer_dispatch_generation_digest = (
            transfer_lifetime.dispatch_generation_digest
        )
        self.phase = "materialized"
        self.assert_current()

    def publish_executor_lifetime(
        self,
        lifetime: KineticNativeSampleLaunchLifetime,
    ) -> None:
        self.assert_current()
        if self.phase != "materialized":
            raise ValueError("sample composite executor lifetime is out of order")
        if type(lifetime) is not KineticNativeSampleLaunchLifetime:
            raise TypeError("sample composite executor lifetime is foreign")
        lifetime.assert_current(self.session)
        if (
            lifetime.sample_block is not self.sample_block
            or lifetime._session_identity != self.session_identity
            or lifetime.session_generation_id != self.session_generation_id
        ):
            raise ValueError(
                "sample executor lifetime is not the composite materialization"
            )
        generation_digest = getattr(lifetime, "generation_digest", None)
        if not _is_sha256(generation_digest):
            raise ValueError("sample executor lifetime has no sealed generation")
        self.executor_lifetime = lifetime
        self.executor_lifetime_identity = id(lifetime)
        self.executor_lifetime_generation_digest = generation_digest
        self.phase = "launched"
        self.assert_current()

    def publish_pending_completion(
        self,
        pending: KineticNativePendingSampleLaunchCompletion,
    ) -> None:
        self.assert_current()
        if self.phase != "launched":
            raise ValueError("sample pending completion is out of order")
        if type(pending) is not KineticNativePendingSampleLaunchCompletion:
            raise TypeError("sample pending completion is foreign")
        if (
            pending._session_identity != self.session_identity
            or pending.session_generation_id != self.session_generation_id
            or pending._sample_lifetime is not self.executor_lifetime
            or pending.subject_binding is not self.subject_binding
            or pending.subject_identity != id(self)
            or pending.subject_generation_digest != self.generation_digest
            or pending._launch_epoch is not self.launch_epoch
        ):
            raise ValueError("sample pending completion is not this composite")
        generation_digest = getattr(pending, "generation_digest", None)
        if not _is_sha256(generation_digest):
            raise ValueError("sample pending completion has no sealed generation")
        self.pending_completion = pending
        self.pending_completion_identity = id(pending)
        self.pending_completion_generation_digest = generation_digest
        self.phase = "pending"
        self.assert_current()

    def extend_roots(self, roots: tuple[tuple[str, Any], ...]) -> None:
        self.assert_current()
        if self.phase not in {"registered", "materialized", "launched"}:
            raise ValueError("sample composite roots extended out of order")
        combined = tuple(
            zip(self.additional_root_roles, self.additional_roots, strict=True)
        ) + roots
        roles = tuple(role for role, _ in combined)
        references = tuple(root for _, root in combined)
        if (
            not roots
            or len(set(roles)) != len(roles)
            or any(not isinstance(role, str) or not role.strip() for role in roles)
            or any(root is None for root in references)
        ):
            raise ValueError("sample composite roots are empty or duplicated")
        self.additional_root_roles = roles
        self.additional_roots = references
        self.additional_root_identities = tuple(id(root) for root in references)
        self.additional_root_signatures = tuple(
            _settlement_root_signature_for_role(role, root)
            for role, root in zip(roles, references, strict=True)
        )
        self.assert_current()

    def _commit_after_consumed_receipt(self) -> None:
        """Assignment-only release of the stable subject slot itself."""

        self.stream = None
        self.session = None
        self.plan = None
        self.subject_binding = None
        self.launch_epoch = None
        self.transfer_lifetime = None
        self.sample_block = None
        self.executor_lifetime = None
        self.pending_completion = None
        self.additional_root_roles = ()
        self.additional_root_identities = ()
        self.additional_root_signatures = ()
        self.additional_roots = ()
        self.phase = "committed"


def _prepare_sample_composite_settlement_slot(
    *,
    step_generation_id: str,
    bundle_generation_digest: str,
    plan: PaperKineticSparseSamplePlan,
    session: KineticNativeMaterialStepSession,
    stream: PaperKineticSparseSampleBlockStream,
    launch_ordinal: int,
    covered_sample_count_before_launch: int,
) -> _SampleCompositeSettlementSlot:
    slot = _SampleCompositeSettlementSlot(
        step_generation_id=step_generation_id,
        bundle_generation_digest=bundle_generation_digest,
        plan_generation_digest=plan.generation_digest,
        session_generation_id=session.generation_id,
        session_identity=id(session),
        plan_identity=id(plan),
        stream_identity=id(stream),
        launch_ordinal=launch_ordinal,
        covered_sample_count_before_launch=covered_sample_count_before_launch,
        slot_identity=0,
        generation_digest="",
        session=session,
        plan=plan,
        stream=stream,
        _seal=_SAMPLE_COMPOSITE_SETTLEMENT_SLOT_SEAL,
    )
    slot.slot_identity = id(slot)
    slot.generation_digest = _sample_composite_slot_digest(slot)
    slot.assert_current()
    return slot


@dataclass(frozen=True)
class _LazyAsyncFailureQuarantine:
    """Trainer-owned fail-stop roots whose device completion is unknown.

    This is deliberately a single bounded carrier, not a failure history.  A
    quarantined trainer can never authorize an optimizer or retry a step; its
    only valid recovery is process restart, which keeps Python from dropping
    objects that an asynchronous backend may still be reading or writing.
    """

    stage: str
    original_error: BaseException = field(repr=False)
    original_traceback: Any = field(repr=False)
    failed_completion_fence_error: BaseException = field(repr=False)
    retained_reference_roles: tuple[str, ...]
    retained_references: tuple[Any, ...] = field(repr=False)
    completion_fence_generation_digest: str
    generation_digest: str
    restart_required: bool = True

    def assert_current(self) -> None:
        retained_by_role = dict(
            zip(
                self.retained_reference_roles,
                self.retained_references,
                strict=True,
            )
        )
        if (
            not self.stage.strip()
            or self.original_traceback is None
            or len(self.retained_reference_roles) != len(self.retained_references)
            or len(set(self.retained_reference_roles))
            != len(self.retained_reference_roles)
            or any(not role.strip() for role in self.retained_reference_roles)
            or not _is_sha256(self.completion_fence_generation_digest)
            or not self.restart_required
            or self.generation_digest != _async_failure_quarantine_digest(self)
        ):
            raise ValueError("lazy async failure quarantine changed")
        session = retained_by_role.get("native_session")
        lifetime = retained_by_role.get("session_outstanding_sample_lifetime")
        if lifetime is None:
            lifetime = retained_by_role.get("current_sample_lifetime")
        if lifetime is not None and session is not None and not lifetime.consumed:
            lifetime.assert_retained(session)
        lane_construction = retained_by_role.get("lane_construction_lifetime")
        if lane_construction is not None:
            lane_construction.assert_retained()
        bundle_slot = retained_by_role.get("bundle_construction_lifetime_slot")
        if bundle_slot is not None:
            bundle_slot.assert_current()
        bundle_lifetime = retained_by_role.get("bundle_construction_lifetime")
        if bundle_lifetime is not None:
            bundle_lifetime.assert_retained()
        sample_stream = retained_by_role.get("sample_iterator")
        transfer_lifetime = retained_by_role.get(
            "current_sample_transfer_lifetime"
        )
        if transfer_lifetime is None and sample_stream is not None:
            transfer_lifetime = sample_stream.active_transfer_lifetime
        if transfer_lifetime is not None:
            transfer_lifetime.assert_retained()
        sample_composite_slot = retained_by_role.get(
            "current_sample_composite_slot"
        )
        if sample_composite_slot is not None:
            sample_composite_slot.assert_quarantine_retained()
        pending_sample_completion = retained_by_role.get(
            "current_pending_sample_completion"
        )
        if pending_sample_completion is not None:
            if (
                type(pending_sample_completion)
                is not KineticNativePendingSampleLaunchCompletion
                or sample_composite_slot is None
                or session is None
            ):
                raise ValueError(
                    "quarantined pending sample lost its exact composite roots"
                )
            completion_fence = retained_by_role.get(
                "sealed_completion_fence"
            )
            pending_sample_completion.assert_exact_sealed_receipt_relation(
                session,
                completion_fence,
                subject=sample_composite_slot,
                require_unconsumed=(
                    pending_sample_completion.phase
                    == "pending_receipt_consumption"
                ),
            )
        active = retained_by_role.get("active_blocks")
        if isinstance(active, dict):
            for block_state in active.values():
                block_state.compact_gather_lifetime.assert_retained()
                block_state.forward_into_lifetime.assert_retained(session)
        compact_gather = retained_by_role.get("current_compact_gather_lifetime")
        if compact_gather is not None:
            compact_gather.assert_retained()
        top_level = retained_by_role.get("top_level_device_transaction_lifetime")
        if top_level is not None:
            top_level.assert_retained()
        completion_fence = retained_by_role.get("sealed_completion_fence")
        if completion_fence is not None:
            completion_fence.assert_current(require_healthy=False)
            if (
                completion_fence.generation_digest
                != self.completion_fence_generation_digest
            ):
                raise ValueError("quarantined completion capability changed")
        forward_lifetime = retained_by_role.get("current_forward_into_lifetime")
        if forward_lifetime is not None:
            forward_lifetime.assert_retained(session)
        reverse_block = retained_by_role.get("current_reverse_block_state")
        if (
            reverse_block is not None
            and (
                not isinstance(active, dict)
                or all(value is not reverse_block for value in active.values())
            )
        ):
            raise ValueError("lazy reverse block escaped its retained active lane")


@dataclass
class _TopLevelDeviceTransactionLifetime:
    """State-owned roots installed before the first step-local device call."""

    global_site_rgba_f32: torch.Tensor | None = field(repr=False)
    global_grad_site_rgba_f32: torch.Tensor | None = field(repr=False)
    background_rgb_f32: torch.Tensor | None = field(repr=False)
    sealed_completion_fence: PaperKineticSealedCompletionFence | None = field(
        repr=False,
    )
    loss_f32: torch.Tensor | None = field(default=None, repr=False)
    cone_diagnostic_i32: torch.Tensor | None = field(default=None, repr=False)
    global_bar_zero_result: torch.Tensor | None = field(default=None, repr=False)
    phase: str = "installed"
    completion_fenced: bool = False
    _global_material_identity: int = field(default=0, repr=False)
    _global_bar_identity: int = field(default=0, repr=False)
    _background_identity: int = field(default=0, repr=False)
    _loss_identity: int | None = field(default=None, repr=False)
    _diagnostic_identity: int | None = field(default=None, repr=False)
    _seal: object = field(default=None, repr=False)

    def assert_retained(self) -> None:
        if self._seal is not _TOP_LEVEL_DEVICE_TRANSACTION_LIFETIME_SEAL:
            raise ValueError("top-level device transaction lifetime was not sealed")
        if self.phase == "released":
            if (
                not self.completion_fenced
                or self.global_site_rgba_f32 is not None
                or self.global_grad_site_rgba_f32 is not None
                or self.background_rgb_f32 is not None
                or self.loss_f32 is not None
                or self.cone_diagnostic_i32 is not None
                or self.global_bar_zero_result is not None
                or self.sealed_completion_fence is not None
            ):
                raise ValueError("released top-level device transaction retained roots")
            return
        if self.phase not in {
            "installed",
            "loss_published",
            "diagnostic_published",
            "global_bar_zeroing",
            "active",
            "cleanup_zeroing",
        }:
            raise ValueError("top-level device transaction lifetime phase changed")
        if (
            self.completion_fenced
            or not isinstance(self.global_site_rgba_f32, torch.Tensor)
            or id(self.global_site_rgba_f32) != self._global_material_identity
            or not isinstance(self.global_grad_site_rgba_f32, torch.Tensor)
            or id(self.global_grad_site_rgba_f32) != self._global_bar_identity
            or not isinstance(self.background_rgb_f32, torch.Tensor)
            or id(self.background_rgb_f32) != self._background_identity
            or type(self.sealed_completion_fence)
            is not PaperKineticSealedCompletionFence
        ):
            raise ValueError("top-level device transaction predecessor roots changed")
        loss_expected = self.phase != "installed"
        diagnostic_expected = self.phase not in {"installed", "loss_published"}
        if loss_expected != isinstance(self.loss_f32, torch.Tensor):
            raise ValueError("top-level loss output publication changed")
        if loss_expected and id(self.loss_f32) != self._loss_identity:
            raise ValueError("top-level loss output identity changed")
        if diagnostic_expected != isinstance(self.cone_diagnostic_i32, torch.Tensor):
            raise ValueError("top-level diagnostic output publication changed")
        if diagnostic_expected and id(self.cone_diagnostic_i32) != self._diagnostic_identity:
            raise ValueError("top-level diagnostic output identity changed")
        zero_result_expected = self.phase in {"active", "cleanup_zeroing"}
        if zero_result_expected != isinstance(
            self.global_bar_zero_result,
            torch.Tensor,
        ):
            raise ValueError("top-level global-bar zero result publication changed")

    def publish_loss(self, loss_f32: torch.Tensor) -> None:
        self.assert_retained()
        if self.phase != "installed" or not isinstance(loss_f32, torch.Tensor):
            raise ValueError("top-level loss publication is not current")
        self.loss_f32 = loss_f32
        self._loss_identity = id(loss_f32)
        self.phase = "loss_published"
        self.assert_retained()

    def publish_diagnostic(self, diagnostic_i32: torch.Tensor) -> None:
        self.assert_retained()
        if self.phase != "loss_published" or not isinstance(
            diagnostic_i32,
            torch.Tensor,
        ):
            raise ValueError("top-level diagnostic publication is not current")
        self.cone_diagnostic_i32 = diagnostic_i32
        self._diagnostic_identity = id(diagnostic_i32)
        self.phase = "diagnostic_published"
        self.assert_retained()

    def begin_global_bar_zero(self) -> None:
        self.assert_retained()
        if self.phase != "diagnostic_published":
            raise ValueError("top-level global-bar zero began out of order")
        self.phase = "global_bar_zeroing"
        self.assert_retained()

    def publish_global_bar_zero_result(self, result: torch.Tensor) -> None:
        if self.phase != "global_bar_zeroing":
            raise ValueError("top-level global-bar zero result is not current")
        self.global_bar_zero_result = result
        self.phase = "active"
        self.assert_retained()

    def begin_cleanup_zero(self) -> None:
        self.assert_retained()
        if self.phase not in {"global_bar_zeroing", "active"}:
            raise ValueError("top-level cleanup zero began out of order")
        self.global_bar_zero_result = self.global_grad_site_rgba_f32
        self.phase = "cleanup_zeroing"
        self.assert_retained()

    def retire_after_sealed_completion_receipt(
        self,
        receipt: PaperKineticCompletionFenceReceipt,
        *,
        stage: str,
        launch_generation_digest: str,
        expected_fence_sequence: int,
    ) -> None:
        self.assert_releasable_for_sealed_completion_receipt(
            receipt,
            stage=stage,
            launch_generation_digest=launch_generation_digest,
            expected_fence_sequence=expected_fence_sequence,
        )
        capability = self.sealed_completion_fence
        receipt.consume_for(
            capability,
            stage=stage,
            launch_generation_digest=launch_generation_digest,
            fence_sequence=expected_fence_sequence,
            consumer="top-level-device-transaction-release",
        )
        self._commit_release_after_consumed_receipt()

    def assert_releasable_for_sealed_completion_receipt(
        self,
        receipt: PaperKineticCompletionFenceReceipt,
        *,
        stage: str,
        launch_generation_digest: str,
        expected_fence_sequence: int,
    ) -> None:
        self.assert_retained()
        capability = self.sealed_completion_fence
        if type(capability) is not PaperKineticSealedCompletionFence:
            raise TypeError("top-level transaction lost its sealed completion fence")
        receipt.assert_for(
            capability,
            stage=stage,
            launch_generation_digest=launch_generation_digest,
            fence_sequence=expected_fence_sequence,
            require_unconsumed=True,
        )

    def _commit_release_after_consumed_receipt(self) -> None:
        """Assignment-only root clear after exact authority is consumed."""

        self.completion_fenced = True
        self.global_site_rgba_f32 = None
        self.global_grad_site_rgba_f32 = None
        self.background_rgb_f32 = None
        self.loss_f32 = None
        self.cone_diagnostic_i32 = None
        self.global_bar_zero_result = None
        self.sealed_completion_fence = None
        self.phase = "released"

    def retire_after_all_completion_receipts(
        self,
        capability: PaperKineticSealedCompletionFence,
        *,
        expected_last_consumed_sequence: int,
    ) -> None:
        """Release after the exact capability proves no launch epoch is live."""

        self.assert_retained()
        if capability is not self.sealed_completion_fence:
            raise ValueError("top-level transaction received a foreign capability")
        capability.assert_current()
        if (
            isinstance(expected_last_consumed_sequence, bool)
            or not isinstance(expected_last_consumed_sequence, int)
            or expected_last_consumed_sequence < 1
            or capability.registered_launch_epoch is not None
            or capability.outstanding_receipt_sequence is not None
            or capability.successful_fence_count
            != expected_last_consumed_sequence
            or capability.consumed_fence_count
            != expected_last_consumed_sequence
            or capability.last_consumed_fence_sequence
            != expected_last_consumed_sequence
        ):
            raise ValueError("top-level completion ledger is not fully consumed")
        self._commit_release_after_consumed_receipt()


@dataclass
class PaperKineticLazyNativeTrainerState:
    """O(1) logical-step ledger that prevents completed-step reuse."""

    provider_identity: int
    provider_generation_digest: str
    global_site_count: int
    device: torch.device
    next_step_index: int = 0
    last_completed_step_index: int = -1
    active_step_index: int | None = None
    optimizer_callback_count: int = 0
    poisoned: bool = False
    _active_device_transaction_lifetime: (
        _TopLevelDeviceTransactionLifetime | None
    ) = field(default=None, repr=False)
    _async_failure_quarantine: _LazyAsyncFailureQuarantine | None = field(
        default=None,
        repr=False,
    )
    provenance: str = STEP_PROVENANCE
    _execution_lock: threading.Lock = field(
        default_factory=threading.Lock,
        repr=False,
    )
    _seal: object = field(default=None, repr=False)

    def assert_current(
        self,
        provider: PaperKineticLazyProgramBundleProvider,
    ) -> None:
        if self._seal is not _TRAINER_STATE_SEAL:
            raise ValueError("lazy native trainer state was not sealed by its preparer")
        if self._async_failure_quarantine is not None:
            self._async_failure_quarantine.assert_current()
        if (
            self.provenance != STEP_PROVENANCE
            or id(provider) != self.provider_identity
            or provider.generation_digest != self.provider_generation_digest
            or self.global_site_count != provider.world.site_count
            or self.global_site_count < 1
            or self.next_step_index != self.last_completed_step_index + 1
            or self.optimizer_callback_count != self.next_step_index
            or self.active_step_index is not None
            and self.active_step_index != self.next_step_index
            or self._async_failure_quarantine is not None
            and (
                not self.poisoned
                or self.active_step_index != self.next_step_index
            )
            or self.active_step_index is None
            and self._active_device_transaction_lifetime is not None
            or not isinstance(self._execution_lock, _LOCK_TYPE)
        ):
            raise ValueError("lazy native trainer state identity/step contract changed")
        provider.assert_warm_current()


@dataclass(frozen=True)
class PaperKineticLazyNativeMemoryPolicy:
    """Explicit fail-before-launch logical tensor budgets for one step."""

    max_global_material_and_bar_tensor_bytes: int
    max_bundle_observation_count: int
    max_lane_resident_logical_tensor_bytes: int
    max_active_node_and_vjp_tensor_bytes: int
    max_decoded_frame_scratch_tensor_bytes: int
    max_selected_frame_target_tensor_bytes: int
    max_sample_launch_tensor_bytes: int
    max_coordinator_visible_live_tensor_bytes: int
    target_frame_access_mode: str = TARGET_FRAME_STREAM_ONCE
    max_step_target_frame_cache_tensor_bytes: int = 0

    def assert_valid(self) -> None:
        values = (
            self.max_global_material_and_bar_tensor_bytes,
            self.max_bundle_observation_count,
            self.max_lane_resident_logical_tensor_bytes,
            self.max_active_node_and_vjp_tensor_bytes,
            self.max_decoded_frame_scratch_tensor_bytes,
            self.max_selected_frame_target_tensor_bytes,
            self.max_sample_launch_tensor_bytes,
            self.max_coordinator_visible_live_tensor_bytes,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("lazy native memory-policy limits must be integers")
        if any(value < 1 for value in values):
            raise ValueError("lazy native memory-policy limits must be positive")
        if self.target_frame_access_mode not in TARGET_FRAME_ACCESS_MODES:
            raise ValueError("lazy native target-frame access mode is unsupported")
        cache_bytes = self.max_step_target_frame_cache_tensor_bytes
        if isinstance(cache_bytes, bool) or not isinstance(cache_bytes, int):
            raise TypeError("lazy native target-frame cache limit must be an integer")
        if self.target_frame_access_mode == TARGET_FRAME_STREAM_ONCE:
            if cache_bytes != 0:
                raise ValueError("one-frame streaming requires a zero step target-cache budget")
        elif cache_bytes < 1:
            raise ValueError("decode-once target caching requires a positive explicit byte budget")


@dataclass
class PaperKineticLazyNativeMaterialStepResult:
    """Final caller-owned loss/bar views and source-level execution proof."""

    step_index: int
    step_generation_id: str
    provider_generation_digest: str
    world_generation_digest: str
    sites_content_digest: str
    loss_normalization_id: str
    material_generation_id: str
    background_generation_id: str
    loss_f32: torch.Tensor = field(repr=False)
    grad_global_site_rgba_f32: torch.Tensor = field(repr=False)
    accounting: Mapping[str, int | float | str | bool]
    generation_digest: str
    _tensor_signatures: tuple[tuple[object, ...], ...] = field(repr=False)
    _material_tensor_identity: int = field(repr=False)
    _material_tensor_signature: tuple[object, ...] = field(repr=False)
    _background_tensor_identity: int = field(repr=False)
    _background_tensor_signature: tuple[object, ...] = field(repr=False)
    _sealed_completion_fence: PaperKineticSealedCompletionFence = field(
        repr=False,
    )
    _sealed_completion_fence_identity: int = field(repr=False)
    _sealed_completion_fence_generation_digest: str = field(repr=False)
    _sealed_generation_digest: str = field(repr=False)
    issued_bridge_receipt_identity: int = 0
    issued_bridge_receipt_generation_digest: str = ""
    provenance: str = STEP_PROVENANCE
    runtime_status: str = STEP_STATUS
    native_runtime_verified: bool = False
    allocator_peak_measured: bool = False
    _seal: object = field(default=None, repr=False)

    def assert_current(self) -> None:
        if self._seal is not _RESULT_SEAL:
            raise ValueError("lazy native step result was not sealed by its coordinator")
        if (
            self.provenance != STEP_PROVENANCE
            or self.runtime_status != STEP_STATUS
            or self.native_runtime_verified
            or self.allocator_peak_measured
            or self.step_index < 0
            or not self.step_generation_id.strip()
            or not self.loss_normalization_id.strip()
            or not _is_sha256(self.world_generation_digest)
            or not _is_sha256(self.sites_content_digest)
            or not self.material_generation_id.strip()
            or not self.background_generation_id.strip()
            or self._material_tensor_identity < 1
            or not self._material_tensor_signature
            or self._background_tensor_identity < 1
            or not self._background_tensor_signature
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
            or tuple(_tensor_signature(tensor) for tensor in (self.loss_f32, self.grad_global_site_rgba_f32))
            != self._tensor_signatures
            or self.generation_digest != self._sealed_generation_digest
            or self.generation_digest != _result_digest(self)
        ):
            raise ValueError("lazy native step result identity/execution proof changed")
        self._sealed_completion_fence.assert_current()

    def assert_device_snapshot_tensors(
        self,
        *,
        material_tensor: torch.Tensor,
        background_tensor: torch.Tensor,
    ) -> None:
        """Prove that a bridge snapshot is the exact step input generation."""

        self.assert_current()
        if (
            id(material_tensor) != self._material_tensor_identity
            or _tensor_signature(material_tensor) != self._material_tensor_signature
            or id(background_tensor) != self._background_tensor_identity
            or _tensor_signature(background_tensor)
            != self._background_tensor_signature
        ):
            raise ValueError("lazy step result is not bound to the device snapshot")

    def claim_bridge_receipt(
        self,
        *,
        receipt_identity: int,
        receipt_generation_digest: str,
    ) -> None:
        self.assert_current()
        if self.issued_bridge_receipt_identity:
            raise ValueError("lazy step result already issued a bridge receipt")
        if receipt_identity < 1 or not _is_sha256(receipt_generation_digest):
            raise ValueError("lazy bridge receipt identity/generation is invalid")
        self.issued_bridge_receipt_identity = receipt_identity
        self.issued_bridge_receipt_generation_digest = (
            receipt_generation_digest
        )
        self.assert_current()


@dataclass
class _CompactMaterialGatherLifetime:
    """Roots a gather result before a following contiguous operation."""

    global_site_rgba_f32: torch.Tensor | None = field(repr=False)
    source_site_ids_i64: torch.Tensor | None = field(repr=False)
    index_select_result_f32: torch.Tensor | None = field(default=None, repr=False)
    compact_site_rgba_f32: torch.Tensor | None = field(default=None, repr=False)
    phase: str = "installed"
    _global_identity: int = field(default=0, repr=False)
    _indices_identity: int = field(default=0, repr=False)
    _seal: object = field(default=None, repr=False)

    def assert_retained(self) -> None:
        if self.phase == "released":
            if (
                self._seal is not _COMPACT_GATHER_LIFETIME_SEAL
                or self.global_site_rgba_f32 is not None
                or self.source_site_ids_i64 is not None
                or self.index_select_result_f32 is not None
                or self.compact_site_rgba_f32 is not None
            ):
                raise ValueError("released compact material gather retained roots")
            return
        if (
            self._seal is not _COMPACT_GATHER_LIFETIME_SEAL
            or self.phase not in {"installed", "gathered", "materialized"}
            or id(self.global_site_rgba_f32) != self._global_identity
            or id(self.source_site_ids_i64) != self._indices_identity
            or (self.phase in {"gathered", "materialized"})
            != isinstance(self.index_select_result_f32, torch.Tensor)
            or (self.phase == "materialized")
            != isinstance(self.compact_site_rgba_f32, torch.Tensor)
        ):
            raise ValueError("compact material gather lifetime changed")

    def retire_after_sealed_completion_receipt(
        self,
        receipt: PaperKineticCompletionFenceReceipt,
        capability: PaperKineticSealedCompletionFence,
        *,
        stage: str,
        launch_generation_digest: str,
        expected_fence_sequence: int,
    ) -> None:
        self.assert_releasable_for_sealed_completion_receipt(
            receipt,
            capability,
            stage=stage,
            launch_generation_digest=launch_generation_digest,
            expected_fence_sequence=expected_fence_sequence,
        )
        receipt.consume_for(
            capability,
            stage=stage,
            launch_generation_digest=launch_generation_digest,
            fence_sequence=expected_fence_sequence,
            consumer="compact-material-gather-release",
        )
        self._commit_release_after_consumed_receipt()

    def assert_releasable_for_sealed_completion_receipt(
        self,
        receipt: PaperKineticCompletionFenceReceipt,
        capability: PaperKineticSealedCompletionFence,
        *,
        stage: str,
        launch_generation_digest: str,
        expected_fence_sequence: int,
    ) -> None:
        self.assert_retained()
        receipt.assert_for(
            capability,
            stage=stage,
            launch_generation_digest=launch_generation_digest,
            fence_sequence=expected_fence_sequence,
            require_unconsumed=True,
        )

    def _commit_release_after_consumed_receipt(self) -> None:
        """Assignment-only root clear after exact authority is consumed."""

        self.global_site_rgba_f32 = None
        self.source_site_ids_i64 = None
        self.index_select_result_f32 = None
        self.compact_site_rgba_f32 = None
        self.phase = "released"

    def retire_after_all_completion_receipts(
        self,
        capability: PaperKineticSealedCompletionFence,
        *,
        expected_last_consumed_sequence: int,
    ) -> None:
        self.assert_retained()
        capability.assert_current()
        if (
            capability.registered_launch_epoch is not None
            or capability.outstanding_receipt_sequence is not None
            or capability.successful_fence_count
            != expected_last_consumed_sequence
            or capability.consumed_fence_count
            != expected_last_consumed_sequence
        ):
            raise ValueError("compact-gather completion ledger is not consumed")
        self._commit_release_after_consumed_receipt()


@dataclass
class _ActiveNativeBlockState:
    token: KineticNativeMaterialStepWorldToken
    grad_node_chart_f32: torch.Tensor
    loss_f32: torch.Tensor
    compact_gather_lifetime: _CompactMaterialGatherLifetime
    forward_into_lifetime: KineticNativeNodeForwardIntoLifetime


@dataclass(frozen=True)
class _BundleExecutionResult:
    eligible_native_block_count: int
    active_native_block_count: int
    native_node_forward_launch_count: int
    native_sample_prepare_count: int
    native_sample_launch_count: int
    native_sample_completion_fence_count: int
    native_reverse_completion_fence_count: int
    native_lane_construction_completion_fence_count: int
    lane_release_completion_fence_count: int
    native_material_word_vjp_launch_count: int
    native_full_geometry_vjp_launch_count: int
    native_fused_union_v2_vjp_launch_count: int
    native_length_bar_tensor_bytes: int
    native_union_construction_completion_fence_count: int
    geometry_d2h_completion_fence_count: int
    streamed_sample_count: int
    ordered_word_node_interactions: int
    sample_to_node_linear_interactions: int
    sample_to_node_dense_fallback_interactions: int
    lane_resident_logical_tensor_bytes: int
    peak_active_node_state_tensor_bytes: int
    peak_sample_launch_tensor_bytes: int
    target_read_accounting: Mapping[str, int | bool | str]
    executor_generation_id: str
    native_abi_identity: tuple[tuple[str, int], ...]
    lane_generation_digest: str
    fence_call_count: int


def prepare_paper_kinetic_lazy_native_trainer_state(
    provider: PaperKineticLazyProgramBundleProvider,
    *,
    device: torch.device | str,
    initial_step_index: int = 0,
) -> PaperKineticLazyNativeTrainerState:
    """Bind an O(1) step ledger to one immutable provider/world generation.

    ``initial_step_index`` is the absolute optimizer generation.  A freshly
    promoted or restored provider must continue at its combined state's
    ``geometry_update_count`` rather than silently restarting the ledger at
    zero.
    """

    if not isinstance(provider, PaperKineticLazyProgramBundleProvider):
        raise TypeError("lazy native trainer state requires a kinetic provider")
    if (
        isinstance(initial_step_index, bool)
        or not isinstance(initial_step_index, int)
        or initial_step_index < 0
    ):
        raise ValueError("initial_step_index must be a nonnegative integer")
    provider.assert_current()
    state = PaperKineticLazyNativeTrainerState(
        provider_identity=id(provider),
        provider_generation_digest=provider.generation_digest,
        global_site_count=provider.world.site_count,
        device=torch.device(device),
        next_step_index=initial_step_index,
        last_completed_step_index=initial_step_index - 1,
        optimizer_callback_count=initial_step_index,
        _seal=_TRAINER_STATE_SEAL,
    )
    state.assert_current(provider)
    return state


@torch.no_grad()
def run_paper_kinetic_lazy_native_material_step(
    state: PaperKineticLazyNativeTrainerState,
    provider: PaperKineticLazyProgramBundleProvider,
    observations: Iterable[PaperKineticObservation],
    *,
    step_index: int,
    expected_observation_count: int,
    expected_observation_manifest_digest: str,
    loss_normalization_id: str,
    material_generation_id: str,
    background_generation_id: str,
    global_site_rgba_f32: torch.Tensor,
    global_grad_site_rgba_f32: torch.Tensor,
    background_rgb_f32: torch.Tensor,
    native_ops: Any,
    maximum_samples_per_launch: int,
    memory_policy: PaperKineticLazyNativeMemoryPolicy,
    optimizer_update: Callable[[PaperKineticLazyNativeMaterialStepResult], None],
    cone_tolerance: float = 1.0e-5,
    _full_geometry_context: Any | None = None,
) -> Any:
    """Run one sparse material step with bundle-maximum native residency."""

    if not isinstance(state, PaperKineticLazyNativeTrainerState):
        raise TypeError("state must be PaperKineticLazyNativeTrainerState")
    execution_lock = state._execution_lock
    if not isinstance(execution_lock, _LOCK_TYPE):
        raise TypeError("lazy native trainer execution lock changed")
    if not execution_lock.acquire(blocking=False):
        raise ValueError("lazy native trainer already has an active or reentrant step")

    loss_f32: torch.Tensor | None = None
    cone_diagnostic_i32: torch.Tensor | None = None
    target_frame_cache: PaperKineticStepTargetFrameCache | None = None
    bundle_construction_lifetime_slot: (
        PaperKineticLazyBundleConstructionLifetimeSlot | None
    ) = None
    bundle_iterator = None
    step_generation_id = ""
    immutable_input_signatures: tuple[tuple[object, ...], ...] = ()
    global_loss_element_count = 0
    owns_active_state = False
    optimizer_authorization_started = False
    device_transaction_lifetime: _TopLevelDeviceTransactionLifetime | None = None
    sealed_completion_fence: PaperKineticSealedCompletionFence | None = None
    accelerator_stage_settlement_slot: (
        _AcceleratorStageSettlementSlot | None
    ) = None
    top_initialization_epoch: PaperKineticCompletionLaunchEpoch | None = None
    top_initialization_receipt: PaperKineticCompletionFenceReceipt | None = None
    top_initialization_sequence: int | None = None
    top_initialization_launch_digest = ""
    canonical_backend_provenance = ""
    bundle_materialization_epoch: PaperKineticCompletionLaunchEpoch | None = None
    bundle_materialization_receipt: PaperKineticCompletionFenceReceipt | None = None
    bundle_materialization_sequence: int | None = None
    bundle_materialization_launch_digest = ""
    bundle_materialization_spatial_lifetime = None
    bundle_materialization_release_kind = ""

    bundle_count = 0
    processed_observation_count = 0
    eligible_native_block_count = 0
    active_native_block_count = 0
    native_node_forward_launch_count = 0
    native_sample_prepare_count = 0
    native_sample_launch_count = 0
    native_sample_completion_fence_count = 0
    native_reverse_completion_fence_count = 0
    bundle_materialization_completion_fence_count = 0
    bundle_exhaustion_probe_completion_fence_count = 0
    native_lane_construction_completion_fence_count = 0
    lane_release_completion_fence_count = 0
    native_material_word_vjp_launch_count = 0
    native_full_geometry_vjp_launch_count = 0
    native_fused_union_v2_vjp_launch_count = 0
    native_length_bar_tensor_bytes = 0
    native_union_construction_completion_fence_count = 0
    geometry_d2h_completion_fence_count = 0
    streamed_sample_count = 0
    ordered_word_node_interactions = 0
    sample_to_node_linear_interactions = 0
    sample_to_node_dense_fallback_interactions = 0
    peak_lane_resident_logical_tensor_bytes = 0
    peak_active_node_state_tensor_bytes = 0
    peak_sample_launch_tensor_bytes = 0
    selected_pixel_read_call_count = 0
    selected_pixel_read_observation_count = 0
    direct_selected_pixel_observation_count = 0
    bounded_region_selected_pixel_observation_count = 0
    full_frame_fallback_observation_count = 0
    full_frame_target_materialization_count = 0
    bounded_region_target_materialization_count = 0
    peak_full_frame_materialization_tensor_bytes = 0
    peak_bounded_region_materialization_tensor_bytes = 0
    peak_source_visible_target_read_tensor_bytes = 0
    peak_transient_mapped_address_space_bytes = 0
    mapped_selected_pixel_read_call_count = 0
    mapping_closed_before_return_count = 0
    selected_pixel_read_modes: set[str] = set()
    selected_pixel_source_provenance_manifest = _ManifestAccumulator(
        "selected-pixel-source-provenance"
    )
    peak_decoded_frame_scratch_upper_bound_bytes = 0
    peak_selected_frame_target_tensor_upper_bound_bytes = 0
    peak_coordinator_visible_live_tensor_upper_bound_bytes = 0
    target_frame_request_count = 0
    last_track_identity: tuple[int, int] | None = None
    fence_call_count = 0
    compile_track_count = 0
    compiler_work_receipt_count = 0
    compiler_work_receipt_chain_link_count = 0
    root_complement_witness_count = 0
    candidate_source_attempt_count = 0
    all_site_witness_check_count = 0
    unique_pair_difference_count = 0
    per_witness_candidate_bound_verified = True
    exhaustive_triple_enumeration_used = False
    requested_frame_sampling_used = False
    compiler_accounting_complete = True
    all_track_receipt_digests_verified = True
    compiler_work_receipt_provenance: str | None = None
    compiler_work_receipt_provenance_mixed = False
    camera_ray_slice_work_count = 0
    camera_ray_slice_scalar_count = 0
    step_native_abi_identity: tuple[tuple[str, int], ...] | None = None
    native_generation_accumulator = _ManifestAccumulator("native-lane-generations")
    observation_accumulator = _ManifestAccumulator("observation-identities")
    compiler_receipt_accumulator = _ManifestAccumulator(
        "active-compiler-work-receipts"
    )

    try:
        _validate_step_inputs(
            state,
            provider,
            step_index=step_index,
            expected_observation_count=expected_observation_count,
            expected_observation_manifest_digest=(expected_observation_manifest_digest),
            loss_normalization_id=loss_normalization_id,
            material_generation_id=material_generation_id,
            background_generation_id=background_generation_id,
            global_site_rgba_f32=global_site_rgba_f32,
            global_grad_site_rgba_f32=global_grad_site_rgba_f32,
            background_rgb_f32=background_rgb_f32,
            maximum_samples_per_launch=maximum_samples_per_launch,
            memory_policy=memory_policy,
            optimizer_update=optimizer_update,
            cone_tolerance=cone_tolerance,
        )
        if _full_geometry_context is not None:
            from paper_kinetic_lazy_full_geometry_step import (
                _PaperKineticLazyFullGeometryExecutionContext,
            )

            if type(_full_geometry_context) is not (
                _PaperKineticLazyFullGeometryExecutionContext
            ):
                raise TypeError("full-geometry scheduler context has a foreign type")
            _full_geometry_context.assert_current()
            if _full_geometry_context.provider is not provider:
                raise ValueError("full-geometry context is foreign to the provider")
        if state.active_step_index is not None:
            _fail_value("lazy native trainer already has an active or interrupted step")
        completion_owner_generation_digest = _digest_parts(
            "paper-kinetic-lazy-completion-owner-v1",
            STEP_PROVENANCE,
            provider.generation_digest,
            step_index,
            expected_observation_count,
            expected_observation_manifest_digest,
            loss_normalization_id,
            material_generation_id,
            background_generation_id,
            str(state.device),
            _lazy_native_abi_identity(native_ops),
            _tensor_signature(global_site_rgba_f32),
            _tensor_signature(global_grad_site_rgba_f32),
            _tensor_signature(background_rgb_f32),
            memory_policy,
            (
                None
                if _full_geometry_context is None
                else (
                    _full_geometry_context.reverse_mode,
                    _full_geometry_context.geometry_generation_id,
                    _full_geometry_context.policy,
                )
            ),
        )
        sealed_completion_fence = prepare_paper_kinetic_sealed_completion_fence(
            native_ops,
            device=state.device,
            owner_generation_digest=completion_owner_generation_digest,
            dispatch_anchor=(
                global_site_rgba_f32 if state.device.type == "mps" else None
            ),
        )
        canonical_backend_provenance = (
            sealed_completion_fence.backend_provenance
        )
        top_initialization_launch_digest = _digest_parts(
            "lazy-top-level-initialization-v1",
            completion_owner_generation_digest,
            _tensor_signature(global_site_rgba_f32),
            _tensor_signature(global_grad_site_rgba_f32),
            _tensor_signature(background_rgb_f32),
        )
        top_initialization_sequence = sealed_completion_fence.next_fence_sequence
        state.active_step_index = step_index
        owns_active_state = True
        _assert_nonretaining_factory(provider)
        global_loss_element_count = expected_observation_count * 3
        step_generation_id = _digest_parts(
            completion_owner_generation_digest,
            sealed_completion_fence.generation_digest,
            id(optimizer_update),
        )
        device_transaction_lifetime = _TopLevelDeviceTransactionLifetime(
            global_site_rgba_f32=global_site_rgba_f32,
            global_grad_site_rgba_f32=global_grad_site_rgba_f32,
            background_rgb_f32=background_rgb_f32,
            sealed_completion_fence=sealed_completion_fence,
            _global_material_identity=id(global_site_rgba_f32),
            _global_bar_identity=id(global_grad_site_rgba_f32),
            _background_identity=id(background_rgb_f32),
            _seal=_TOP_LEVEL_DEVICE_TRANSACTION_LIFETIME_SEAL,
        )
        state._active_device_transaction_lifetime = device_transaction_lifetime
        device_transaction_lifetime.assert_retained()
        accelerator_stage_settlement_slot = (
            _prepare_accelerator_stage_settlement_slot(
                sealed_completion_fence,
                owner_generation_digest=completion_owner_generation_digest,
            )
        )
        top_initialization_epoch = _register_general_completion_epoch(
            sealed_completion_fence,
            accelerator_stage_settlement_slot,
            stage="top-level-initialization",
            launch_generation_digest=top_initialization_launch_digest,
            roots=(
                ("trainer_state", state),
                ("provider", provider),
                ("device_transaction_lifetime", device_transaction_lifetime),
                ("global_material", global_site_rgba_f32),
                ("global_bar", global_grad_site_rgba_f32),
                ("background", background_rgb_f32),
            ),
        )
        allocated_loss_f32 = torch.zeros(
            (1,),
            dtype=torch.float32,
            device=state.device,
        )
        device_transaction_lifetime.publish_loss(allocated_loss_f32)
        loss_f32 = allocated_loss_f32
        allocated_cone_diagnostic_i32 = torch.zeros(
            (3,),
            dtype=torch.int32,
            device=state.device,
        )
        device_transaction_lifetime.publish_diagnostic(
            allocated_cone_diagnostic_i32
        )
        cone_diagnostic_i32 = allocated_cone_diagnostic_i32
        device_transaction_lifetime.begin_global_bar_zero()
        global_bar_zero_result = global_grad_site_rgba_f32.zero_()
        device_transaction_lifetime.publish_global_bar_zero_result(
            global_bar_zero_result
        )
        if global_bar_zero_result is not global_grad_site_rgba_f32:
            _fail_value("top-level global-bar zero returned foreign storage")
        _extend_general_completion_roots(
            accelerator_stage_settlement_slot,
            (
                ("loss", loss_f32),
                ("cone_diagnostic", cone_diagnostic_i32),
            ),
        )
        top_initialization_receipt = _fence_registered_completion_epoch(
            sealed_completion_fence,
            top_initialization_epoch,
            expected_fence_sequence=top_initialization_sequence,
        )
        _consume_general_completion_receipt(
            sealed_completion_fence,
            accelerator_stage_settlement_slot,
            top_initialization_receipt,
            consumer="lazy-top-level-initialization-settlement",
        )
        fence_call_count += 1
        top_initialization_epoch = None
        immutable_input_signatures = tuple(
            _tensor_signature(tensor) for tensor in (global_site_rgba_f32, background_rgb_f32)
        )
        if memory_policy.target_frame_access_mode == TARGET_FRAME_STEP_CACHE:
            target_frame_cache = prepare_paper_kinetic_step_target_frame_cache(
                provider,
                maximum_resident_bytes=(memory_policy.max_step_target_frame_cache_tensor_bytes),
            )

        bundle_construction_lifetime_slot = (
            prepare_paper_kinetic_lazy_bundle_construction_lifetime_slot()
        )
        bundle_iterator = provider.iter_canonical_spatial_bundles(
            observations,
            device=state.device,
            construction_lifetime_slot=bundle_construction_lifetime_slot,
        )

        def iter_prelaunched_bundles(bundles):
            nonlocal bundle_materialization_completion_fence_count
            nonlocal bundle_exhaustion_probe_completion_fence_count
            nonlocal bundle_materialization_epoch
            nonlocal bundle_materialization_launch_digest
            nonlocal bundle_materialization_receipt
            nonlocal bundle_materialization_release_kind
            nonlocal bundle_materialization_sequence
            nonlocal bundle_materialization_spatial_lifetime
            nonlocal fence_call_count
            bundle_ordinal = 0
            while processed_observation_count < expected_observation_count:
                bundle_materialization_launch_digest = _digest_parts(
                    "lazy-bundle-device-materialization-v2",
                    step_generation_id,
                    provider.generation_digest,
                    bundle_ordinal,
                    processed_observation_count,
                    id(bundles),
                    str(state.device),
                )
                bundle_materialization_sequence = (
                    sealed_completion_fence.next_fence_sequence
                )
                bundle_materialization_epoch = (
                    _register_general_completion_epoch(
                        sealed_completion_fence,
                        accelerator_stage_settlement_slot,
                        stage="bundle-materialization",
                        launch_generation_digest=(
                            bundle_materialization_launch_digest
                        ),
                        roots=(
                            ("bundle_iterator", bundles),
                            (
                                "bundle_construction_lifetime_slot",
                                bundle_construction_lifetime_slot,
                            ),
                            ("provider", provider),
                            ("global_material", global_site_rgba_f32),
                            ("global_bar", global_grad_site_rgba_f32),
                        ),
                    )
                )
                try:
                    bundle = next(bundles)
                except StopIteration:
                    bundle_materialization_receipt = (
                        sealed_completion_fence.fence(
                            bundle_materialization_epoch
                        )
                    )
                    _consume_general_completion_receipt(
                        sealed_completion_fence,
                        accelerator_stage_settlement_slot,
                        bundle_materialization_receipt,
                        consumer="lazy-bundle-premature-exhaustion-settlement",
                    )
                    bundle_materialization_completion_fence_count += 1
                    fence_call_count += 1
                    bundle_materialization_epoch = None
                    bundle_materialization_receipt = None
                    bundle_materialization_sequence = None
                    bundle_materialization_launch_digest = ""
                    _fail_value(
                        "lazy native bundle stream ended before the declared "
                        "global observations were covered"
                    )
                bundle_materialization_spatial_lifetime = (
                    bundle.spatial_bundle._construction_lifetime
                )
                _extend_general_completion_roots(
                    accelerator_stage_settlement_slot,
                    (
                        ("materialized_bundle", bundle),
                        (
                            "materialized_spatial_lifetime",
                            bundle_materialization_spatial_lifetime,
                        ),
                    ),
                )
                if state.device.type != "cpu":
                    bundle.assert_accelerator_transfer_pending(provider)
                    if (
                        bundle_construction_lifetime_slot.active_lifetime
                        is not bundle_materialization_spatial_lifetime
                    ):
                        _fail_value(
                            "accelerator bundle materialization lost its caller-visible lifetime"
                        )
                remaining_observation_count = (
                    expected_observation_count - processed_observation_count
                )
                if bundle.observation_count > remaining_observation_count:
                    bundle_materialization_release_kind = "retire"
                    bundle_materialization_spatial_lifetime.assert_retirable_after_consumed_receipt()
                else:
                    bundle_materialization_release_kind = "transfer-predecessors"
                    bundle_materialization_spatial_lifetime.assert_transfer_predecessors_releasable_after_consumed_receipt()
                bundle_materialization_receipt = sealed_completion_fence.fence(
                    bundle_materialization_epoch
                )
                if bundle_materialization_release_kind == "retire":
                    bundle_materialization_spatial_lifetime.assert_retirable_after_consumed_receipt()
                else:
                    if state.device.type == "cpu":
                        bundle_materialization_spatial_lifetime.assert_transfer_predecessors_releasable_after_consumed_receipt()
                    else:
                        bundle_materialization_spatial_lifetime.assert_accelerator_transfer_releasable_after_completion_fence(
                            bundle.spatial_bundle
                        )
                _consume_general_completion_receipt(
                    sealed_completion_fence,
                    accelerator_stage_settlement_slot,
                    bundle_materialization_receipt,
                    consumer="lazy-bundle-materialization-predecessor-release",
                )
                if bundle_materialization_release_kind == "retire":
                    bundle_materialization_spatial_lifetime._commit_retire_after_consumed_receipt()
                else:
                    bundle_materialization_spatial_lifetime._commit_transfer_predecessors_after_consumed_receipt()
                if state.device.type != "cpu":
                    if bundle_materialization_release_kind != "retire":
                        bundle.assert_cold_current(provider)
                    bundle_construction_lifetime_slot.complete(
                        bundle_materialization_spatial_lifetime
                    )
                bundle_materialization_completion_fence_count += 1
                fence_call_count += 1
                bundle_materialization_epoch = None
                bundle_materialization_receipt = None
                bundle_materialization_release_kind = ""
                bundle_materialization_sequence = None
                bundle_materialization_launch_digest = ""
                bundle_materialization_spatial_lifetime = None
                if bundle.observation_count > remaining_observation_count:
                    _fail_value(
                        "lazy native bundle stream exceeded declared coverage"
                    )
                bundle_ordinal += 1
                yield bundle
                del bundle

            bundle_materialization_launch_digest = _digest_parts(
                "lazy-bundle-device-exhaustion-probe-v1",
                step_generation_id,
                provider.generation_digest,
                bundle_ordinal,
                processed_observation_count,
                id(bundles),
                str(state.device),
            )
            bundle_materialization_sequence = (
                sealed_completion_fence.next_fence_sequence
            )
            bundle_materialization_epoch = _register_general_completion_epoch(
                sealed_completion_fence,
                accelerator_stage_settlement_slot,
                stage="bundle-exhaustion-probe",
                launch_generation_digest=bundle_materialization_launch_digest,
                roots=(
                    ("bundle_iterator", bundles),
                    (
                        "bundle_construction_lifetime_slot",
                        bundle_construction_lifetime_slot,
                    ),
                    ("provider", provider),
                ),
            )
            try:
                extra_bundle = next(bundles)
            except StopIteration:
                extra_bundle = None
            if extra_bundle is not None:
                bundle_materialization_spatial_lifetime = (
                    extra_bundle.spatial_bundle._construction_lifetime
                )
                _extend_general_completion_roots(
                    accelerator_stage_settlement_slot,
                    (
                        ("extra_bundle", extra_bundle),
                        (
                            "extra_bundle_spatial_lifetime",
                            bundle_materialization_spatial_lifetime,
                        ),
                    ),
                )
                bundle_materialization_release_kind = "retire"
                bundle_materialization_spatial_lifetime.assert_retirable_after_consumed_receipt()
            bundle_materialization_receipt = sealed_completion_fence.fence(
                bundle_materialization_epoch
            )
            if bundle_materialization_spatial_lifetime is not None:
                bundle_materialization_spatial_lifetime.assert_retirable_after_consumed_receipt()
            _consume_general_completion_receipt(
                sealed_completion_fence,
                accelerator_stage_settlement_slot,
                bundle_materialization_receipt,
                consumer="lazy-bundle-exhaustion-proof",
            )
            if bundle_materialization_spatial_lifetime is not None:
                bundle_materialization_spatial_lifetime._commit_retire_after_consumed_receipt()
            bundle_exhaustion_probe_completion_fence_count += 1
            fence_call_count += 1
            bundle_materialization_epoch = None
            bundle_materialization_receipt = None
            bundle_materialization_release_kind = ""
            bundle_materialization_sequence = None
            bundle_materialization_launch_digest = ""
            bundle_materialization_spatial_lifetime = None
            if extra_bundle is not None:
                _fail_value(
                    "lazy native bundle stream contains undeclared trailing observations"
                )

        with closing(bundle_iterator) as bundles:
            prelaunched_bundles = iter_prelaunched_bundles(bundles)
            expected_bundle_index = 0
            while True:
                try:
                    bundle = next(prelaunched_bundles)
                except StopIteration:
                    break
                _assert_immutable_step_inputs(
                    immutable_input_signatures,
                    global_site_rgba_f32,
                    background_rgb_f32,
                )
                if bundle.bundle_index != expected_bundle_index:
                    _fail_value("lazy native step bundle order is not canonical")
                first_track = (bundle.view_index, bundle.track_ids[0])
                final_track = (bundle.view_index, bundle.track_ids[-1])
                if last_track_identity is not None and first_track <= last_track_identity:
                    _fail_value("lazy native step repeated or reordered a compiled track")
                if tuple(sorted(set(bundle.track_ids))) != bundle.track_ids:
                    _fail_value("lazy native step bundle contains duplicate track ids")
                last_track_identity = final_track
                if bundle.observation_count > memory_policy.max_bundle_observation_count:
                    _fail_value("lazy native bundle exceeds its explicit observation-record budget")
                compile_receipt = bundle.compile_receipt
                compile_receipt.assert_current(
                    track_ids=bundle.track_ids,
                    program_generation_digests=bundle.program_generation_digests,
                    request_generation_digests=(
                        bundle.factory_request_generation_digests
                    ),
                )
                if (
                    compile_receipt.provider_generation_digest
                    != provider.generation_digest
                    or compile_receipt.view_index != bundle.view_index
                ):
                    _fail_value(
                        "lazy native bundle compiler receipt belongs to another provider/view"
                    )
                compile_track_count += compile_receipt.compile_track_count
                compiler_work_receipt_count += (
                    compile_receipt.compiler_work_receipt_count
                )
                compiler_work_receipt_chain_link_count += (
                    compile_receipt.compiler_work_receipt_chain_link_count
                )
                root_complement_witness_count += (
                    compile_receipt.root_complement_witness_count
                )
                candidate_source_attempt_count += (
                    compile_receipt.candidate_source_attempt_count
                )
                all_site_witness_check_count += (
                    compile_receipt.all_site_witness_check_count
                )
                unique_pair_difference_count += (
                    compile_receipt.unique_pair_difference_count
                )
                per_witness_candidate_bound_verified = (
                    per_witness_candidate_bound_verified
                    and compile_receipt.per_witness_candidate_bound_verified
                )
                exhaustive_triple_enumeration_used = (
                    exhaustive_triple_enumeration_used
                    or compile_receipt.exhaustive_triple_enumeration_used
                )
                requested_frame_sampling_used = (
                    requested_frame_sampling_used
                    or compile_receipt.requested_frame_sampling_used
                )
                compiler_accounting_complete = (
                    compiler_accounting_complete
                    and compile_receipt.compiler_accounting_complete
                )
                all_track_receipt_digests_verified = (
                    all_track_receipt_digests_verified
                    and compile_receipt.all_track_receipt_digests_verified
                )
                if compiler_work_receipt_provenance is None:
                    compiler_work_receipt_provenance = (
                        compile_receipt.compiler_work_receipt_provenance
                    )
                elif (
                    compiler_work_receipt_provenance
                    != compile_receipt.compiler_work_receipt_provenance
                ):
                    compiler_work_receipt_provenance_mixed = True
                compiler_receipt_accumulator.add(
                    (
                        bundle.bundle_index,
                        bundle.generation_digest,
                        compile_receipt.generation_digest,
                        compile_receipt.compiler_work_receipt_chain_digest,
                        compile_receipt.compile_track_count,
                        compile_receipt.compiler_work_receipt_count,
                        compile_receipt.root_complement_witness_count,
                        compile_receipt.candidate_source_attempt_count,
                        compile_receipt.all_site_witness_check_count,
                        compile_receipt.unique_pair_difference_count,
                    )
                )
                camera_ray_slice_work_count += bundle.observation_count
                camera_ray_slice_scalar_count += bundle.selected_ray_scalar_count
                for record in bundle.observations:
                    observation_accumulator.add(record.observation.sample_identity)

                plan = prepare_paper_kinetic_sparse_sample_plan(
                    bundle,
                    provider,
                    global_loss_element_count=global_loss_element_count,
                    loss_normalization_id=loss_normalization_id,
                    maximum_samples_per_launch=maximum_samples_per_launch,
                )
                plan_memory = plan.memory_report()
                target_frame_request_count += plan_memory.unique_selected_frame_count
                lane_preflight_bytes = estimate_paper_kinetic_native_lazy_bundle_lane_resident_bytes(
                    bundle,
                    device=state.device,
                )
                active_node_and_vjp_upper_bound_bytes = _active_node_and_vjp_upper_bound_bytes(bundle)
                if target_frame_cache is None:
                    # The selected-pixel source enforces this cap before its
                    # own allocation and returns the exact observed peak in a
                    # sealed read receipt.  Using the cap here keeps the
                    # coordinator-visible preflight conservative for sources
                    # whose mapped-page footprint is not knowable from the
                    # tensor-free plan.
                    decoded_frame_scratch_upper_bound_bytes = (
                        memory_policy.max_decoded_frame_scratch_tensor_bytes
                    )
                else:
                    decoded_frame_scratch_upper_bound_bytes = (
                        plan_memory.decoded_frame_scratch_upper_bound_bytes * 2
                    )
                coordinator_live_upper_bound = _enforce_bundle_memory_policy(
                    memory_policy,
                    global_site_rgba_f32=global_site_rgba_f32,
                    global_grad_site_rgba_f32=global_grad_site_rgba_f32,
                    background_rgb_f32=background_rgb_f32,
                    loss_f32=loss_f32,
                    cone_diagnostic_i32=cone_diagnostic_i32,
                    lane_preflight_bytes=lane_preflight_bytes,
                    active_node_and_vjp_upper_bound_bytes=(active_node_and_vjp_upper_bound_bytes),
                    decoded_frame_scratch_upper_bound_bytes=(decoded_frame_scratch_upper_bound_bytes),
                    selected_frame_target_upper_bound_bytes=(
                        plan_memory.selected_frame_target_tensor_upper_bound_bytes
                    ),
                    sample_launch_upper_bound_bytes=(plan_memory.launch_tensor_upper_bound_bytes),
                )

                outcome = _execute_native_bundle(
                    state,
                    bundle,
                    provider,
                    plan=plan,
                    target_frame_cache=target_frame_cache,
                    global_site_rgba_f32=global_site_rgba_f32,
                    global_grad_site_rgba_f32=global_grad_site_rgba_f32,
                    background_rgb_f32=background_rgb_f32,
                    loss_f32=loss_f32,
                    cone_diagnostic_i32=cone_diagnostic_i32,
                    native_ops=native_ops,
                    backend_provenance=canonical_backend_provenance,
                    step_generation_id=step_generation_id,
                    memory_policy=memory_policy,
                    sealed_completion_fence=sealed_completion_fence,
                    accelerator_stage_settlement_slot=(
                        accelerator_stage_settlement_slot
                    ),
                    top_level_device_transaction_lifetime=(
                        device_transaction_lifetime
                    ),
                    cone_tolerance=cone_tolerance,
                    full_geometry_context=_full_geometry_context,
                )
                _assert_immutable_step_inputs(
                    immutable_input_signatures,
                    global_site_rgba_f32,
                    background_rgb_f32,
                )
                provider.assert_warm_current()
                _assert_nonretaining_factory(provider)
                if step_native_abi_identity is None:
                    step_native_abi_identity = outcome.native_abi_identity
                elif outcome.native_abi_identity != step_native_abi_identity:
                    _fail_value("native operation ABI changed between spatial lanes")
                native_generation_accumulator.add(
                    (
                        outcome.executor_generation_id,
                        outcome.native_abi_identity,
                        outcome.lane_generation_digest,
                    )
                )
                processed_observation_count += bundle.observation_count
                bundle_count += 1
                eligible_native_block_count += outcome.eligible_native_block_count
                active_native_block_count += outcome.active_native_block_count
                native_node_forward_launch_count += outcome.native_node_forward_launch_count
                native_sample_prepare_count += outcome.native_sample_prepare_count
                native_sample_launch_count += outcome.native_sample_launch_count
                native_sample_completion_fence_count += (
                    outcome.native_sample_completion_fence_count
                )
                native_reverse_completion_fence_count += (
                    outcome.native_reverse_completion_fence_count
                )
                native_lane_construction_completion_fence_count += (
                    outcome.native_lane_construction_completion_fence_count
                )
                lane_release_completion_fence_count += (
                    outcome.lane_release_completion_fence_count
                )
                native_material_word_vjp_launch_count += outcome.native_material_word_vjp_launch_count
                native_full_geometry_vjp_launch_count += (
                    outcome.native_full_geometry_vjp_launch_count
                )
                native_fused_union_v2_vjp_launch_count += (
                    outcome.native_fused_union_v2_vjp_launch_count
                )
                native_length_bar_tensor_bytes += (
                    outcome.native_length_bar_tensor_bytes
                )
                native_union_construction_completion_fence_count += (
                    outcome.native_union_construction_completion_fence_count
                )
                geometry_d2h_completion_fence_count += (
                    outcome.geometry_d2h_completion_fence_count
                )
                streamed_sample_count += outcome.streamed_sample_count
                ordered_word_node_interactions += outcome.ordered_word_node_interactions
                sample_to_node_linear_interactions += outcome.sample_to_node_linear_interactions
                sample_to_node_dense_fallback_interactions += outcome.sample_to_node_dense_fallback_interactions
                peak_lane_resident_logical_tensor_bytes = max(
                    peak_lane_resident_logical_tensor_bytes,
                    outcome.lane_resident_logical_tensor_bytes,
                )
                peak_active_node_state_tensor_bytes = max(
                    peak_active_node_state_tensor_bytes,
                    outcome.peak_active_node_state_tensor_bytes,
                )
                peak_sample_launch_tensor_bytes = max(
                    peak_sample_launch_tensor_bytes,
                    outcome.peak_sample_launch_tensor_bytes,
                )
                target_read = outcome.target_read_accounting
                selected_pixel_read_call_count += int(
                    target_read["selected_pixel_read_call_count"]
                )
                selected_pixel_read_observation_count += int(
                    target_read["selected_pixel_read_observation_count"]
                )
                direct_selected_pixel_observation_count += int(
                    target_read["direct_selected_pixel_observation_count"]
                )
                bounded_region_selected_pixel_observation_count += int(
                    target_read[
                        "bounded_region_selected_pixel_observation_count"
                    ]
                )
                full_frame_fallback_observation_count += int(
                    target_read["full_frame_fallback_observation_count"]
                )
                full_frame_target_materialization_count += int(
                    target_read["full_frame_target_materialization_count"]
                )
                bounded_region_target_materialization_count += int(
                    target_read[
                        "bounded_region_target_materialization_count"
                    ]
                )
                peak_full_frame_materialization_tensor_bytes = max(
                    peak_full_frame_materialization_tensor_bytes,
                    int(
                        target_read[
                            "peak_full_frame_materialization_tensor_bytes"
                        ]
                    ),
                )
                peak_bounded_region_materialization_tensor_bytes = max(
                    peak_bounded_region_materialization_tensor_bytes,
                    int(
                        target_read[
                            "peak_bounded_region_materialization_tensor_bytes"
                        ]
                    ),
                )
                peak_source_visible_target_read_tensor_bytes = max(
                    peak_source_visible_target_read_tensor_bytes,
                    int(
                        target_read[
                            "peak_source_visible_target_read_logical_tensor_bytes_upper_bound"
                        ]
                    ),
                )
                peak_transient_mapped_address_space_bytes = max(
                    peak_transient_mapped_address_space_bytes,
                    int(
                        target_read[
                            "peak_transient_mapped_address_space_bytes"
                        ]
                    ),
                )
                mapped_selected_pixel_read_call_count += int(
                    target_read["mapped_selected_pixel_read_call_count"]
                )
                mapping_closed_before_return_count += int(
                    target_read["mapping_closed_before_return_count"]
                )
                selected_pixel_read_modes.add(
                    str(target_read["selected_pixel_read_mode"])
                )
                selected_pixel_source_provenance_manifest.add(
                    (
                        target_read[
                            "selected_pixel_read_source_provenance_digest"
                        ],
                        target_read[
                            "selected_pixel_read_source_provenance_count"
                        ],
                    )
                )
                peak_decoded_frame_scratch_upper_bound_bytes = max(
                    peak_decoded_frame_scratch_upper_bound_bytes,
                    decoded_frame_scratch_upper_bound_bytes,
                )
                peak_selected_frame_target_tensor_upper_bound_bytes = max(
                    peak_selected_frame_target_tensor_upper_bound_bytes,
                    plan_memory.selected_frame_target_tensor_upper_bound_bytes,
                )
                peak_coordinator_visible_live_tensor_upper_bound_bytes = max(
                    peak_coordinator_visible_live_tensor_upper_bound_bytes,
                    coordinator_live_upper_bound,
                )
                fence_call_count += outcome.fence_call_count
                del outcome, plan, plan_memory, bundle
                expected_bundle_index += 1

        if bundle_count < 1:
            _fail_value("lazy native material step produced no spatial bundle")
        compiler_work_receipt_chain_digest = compiler_receipt_accumulator.finish(
            bundle_count
        )
        if compiler_work_receipt_provenance_mixed:
            resolved_compiler_work_receipt_provenance = "mixed"
        else:
            resolved_compiler_work_receipt_provenance = (
                compiler_work_receipt_provenance or "unavailable"
            )
        if compiler_work_receipt_chain_link_count != compile_track_count:
            _fail_arithmetic(
                "lazy native compiler receipt chain lost a compiled-track link"
            )
        expected_all_track_receipts_verified = (
            compile_track_count > 0
            and compiler_work_receipt_count == compile_track_count
        )
        if (
            all_track_receipt_digests_verified
            != expected_all_track_receipts_verified
        ):
            _fail_arithmetic(
                "lazy native compiler receipt digest coverage changed"
            )
        compiler_accounting_complete = (
            compiler_accounting_complete
            and expected_all_track_receipts_verified
            and resolved_compiler_work_receipt_provenance
            not in {"unavailable", "mixed"}
        )
        per_witness_candidate_bound_verified = (
            compiler_work_receipt_count > 0
            and per_witness_candidate_bound_verified
        )
        if (
            camera_ray_slice_work_count != processed_observation_count
            or camera_ray_slice_scalar_count != 6 * camera_ray_slice_work_count
        ):
            _fail_arithmetic(
                "lazy native calibrated camera-ray slice accounting changed"
            )
        if (
            processed_observation_count != expected_observation_count
            or streamed_sample_count != expected_observation_count
        ):
            _fail_value("lazy native material step did not cover the declared global observations")
        observed_manifest_digest = observation_accumulator.finish(processed_observation_count)
        if observed_manifest_digest != expected_observation_manifest_digest:
            _fail_value("lazy native material step observation manifest differs from the declared batch")
        if (
            selected_pixel_read_observation_count
            != expected_observation_count
            or direct_selected_pixel_observation_count
            + bounded_region_selected_pixel_observation_count
            + full_frame_fallback_observation_count
            != expected_observation_count
            or selected_pixel_read_call_count != target_frame_request_count
            or not selected_pixel_read_modes
        ):
            _fail_arithmetic(
                "lazy native selected-target receipts lost exact observation coverage"
            )
        selected_pixel_read_mode = (
            next(iter(selected_pixel_read_modes))
            if len(selected_pixel_read_modes) == 1
            else "mixed"
        )
        selected_pixel_read_acceptance_capable = (
            selected_pixel_read_modes
            <= {"direct_pixels", "certified_bounded_region"}
            and full_frame_fallback_observation_count == 0
            and full_frame_target_materialization_count == 0
        )
        selected_pixel_source_provenance_manifest_digest = (
            selected_pixel_source_provenance_manifest.finish(bundle_count)
        )
        staged_full_geometry = (
            _full_geometry_context is not None
            and _full_geometry_context.reverse_mode == "staged_sparse"
        )
        fused_union_v2_full_geometry = (
            _full_geometry_context is not None
            and _full_geometry_context.reverse_mode == "fused_union_v2"
        )
        expected_material_reverse_count = (
            0
            if staged_full_geometry or fused_union_v2_full_geometry
            else active_native_block_count
        )
        expected_staged_reverse_count = (
            active_native_block_count if staged_full_geometry else 0
        )
        if not (
            native_node_forward_launch_count == active_native_block_count
            and native_material_word_vjp_launch_count
            == expected_material_reverse_count
            and native_full_geometry_vjp_launch_count
            == expected_staged_reverse_count
            and native_fused_union_v2_vjp_launch_count
            == (
                active_native_block_count
                if fused_union_v2_full_geometry
                else 0
            )
        ):
            _fail_arithmetic("lazy native material step violated once-per-active-word execution")
        if not (
            native_sample_prepare_count
            == native_sample_launch_count
            == native_sample_completion_fence_count
        ):
            _fail_arithmetic("lazy native material step sample launch counts differ")
        if (
            native_reverse_completion_fence_count
            != (
                native_material_word_vjp_launch_count
                + native_full_geometry_vjp_launch_count
                + (
                    bundle_count if fused_union_v2_full_geometry else 0
                )
            )
            or lane_release_completion_fence_count != bundle_count
            or bundle_materialization_completion_fence_count != bundle_count
            or bundle_exhaustion_probe_completion_fence_count != 1
            or fence_call_count
            != 1
            + bundle_materialization_completion_fence_count
            + bundle_exhaustion_probe_completion_fence_count
            + native_sample_completion_fence_count
            + native_reverse_completion_fence_count
            + native_lane_construction_completion_fence_count
            + native_union_construction_completion_fence_count
            + geometry_d2h_completion_fence_count
        ):
            _fail_arithmetic(
                "lazy native material step did not fence every sample, "
                "reverse scratch, and released lane"
            )
        _assert_immutable_step_inputs(
            immutable_input_signatures,
            global_site_rgba_f32,
            background_rgb_f32,
        )

        if target_frame_cache is None:
            target_cache_accounting: Mapping[str, int | bool | str] = {
                "request_count": target_frame_request_count,
                "hit_count": 0,
                "decode_attempt_count": full_frame_target_materialization_count,
                "decode_count": full_frame_target_materialization_count,
                "preflight_rejection_count": 0,
                "peak_resident_frame_tensor_bytes": 0,
                "resident_frame_tensor_bytes": 0,
                "cached_frame_count": 0,
                "closed": True,
                "close_count": 0,
                "maximum_resident_bytes": 0,
            }
        else:
            target_frame_cache.close()
            target_cache_accounting = target_frame_cache.accounting()
            if (
                target_cache_accounting["request_count"] != target_frame_request_count
                or target_cache_accounting["closed"] is not True
                or target_cache_accounting["resident_frame_tensor_bytes"] != 0
                or target_cache_accounting["cached_frame_count"] != 0
            ):
                _fail_arithmetic("step target-frame cache close/accounting contract changed")
        provider.assert_warm_current()
        _assert_nonretaining_factory(provider)
        _assert_immutable_step_inputs(
            immutable_input_signatures,
            global_site_rgba_f32,
            background_rgb_f32,
        )

        if sealed_completion_fence is None or device_transaction_lifetime is None:
            _fail_arithmetic("sealed top-level completion authority disappeared")

        if step_native_abi_identity is None:
            _fail_arithmetic("lazy native step has no sealed native ABI")
        native_generation_manifest_digest = native_generation_accumulator.finish(bundle_count)
        target_frame_cache_enabled = target_frame_cache is not None
        if target_frame_cache_enabled:
            target_frame_residency_scaling = (
                "O(unique selected frames * H * W) until the explicit no-eviction "
                "step-cache byte cap; the step fails before decoding beyond that cap"
            )
            target_decode_work_scaling = "one decode per unique selected (view,frame)"
        else:
            if selected_pixel_read_acceptance_capable:
                target_frame_residency_scaling = (
                    "O(selected observations in one bundle-local frame read), "
                    "bounded by the source-decode cap and its sealed "
                    "source-visible receipt"
                )
                target_decode_work_scaling = (
                    "one selected-pixel read per bundle-local selected frame; "
                    "zero full-frame decodes"
                )
            else:
                target_frame_residency_scaling = (
                    "compatibility fallback materialized O(H*W) frame state; "
                    "the exact count/peak is reported and is not acceptance-capable"
                )
                target_decode_work_scaling = (
                    "one compatibility full-frame materialization per reported "
                    "fallback receipt/cache miss"
                )

        accounting: dict[str, int | float | str | bool] = {
            "expected_observation_count": expected_observation_count,
            "processed_observation_count": processed_observation_count,
            "global_loss_element_count": global_loss_element_count,
            "global_site_count": state.global_site_count,
            "spatial_bundle_count": bundle_count,
            "compile_track_count": compile_track_count,
            "compiler_work_receipt_count": compiler_work_receipt_count,
            "compiler_work_receipt_bundle_count": bundle_count,
            "compiler_work_receipt_chain_link_count": (
                compiler_work_receipt_chain_link_count
            ),
            "root_complement_witness_count": root_complement_witness_count,
            "candidate_source_attempt_count": candidate_source_attempt_count,
            "all_site_witness_check_count": all_site_witness_check_count,
            "unique_pair_difference_count": unique_pair_difference_count,
            "per_witness_candidate_bound_verified": (
                per_witness_candidate_bound_verified
            ),
            "exhaustive_triple_enumeration_used": (
                exhaustive_triple_enumeration_used
            ),
            "requested_frame_sampling_used": requested_frame_sampling_used,
            "active_compiler_accounting_complete": (
                compiler_accounting_complete
            ),
            "all_track_receipt_digests_verified": (
                all_track_receipt_digests_verified
            ),
            "compiler_work_receipt_provenance": (
                resolved_compiler_work_receipt_provenance
            ),
            "compiler_work_receipt_chain_digest": (
                compiler_work_receipt_chain_digest
            ),
            "retained_compiled_program_count": 0,
            "retained_compiler_receipt_entry_count": 0,
            "retained_compiler_tensor_bytes": 0,
            "compiler_receipt_state_scaling": (
                "O(1) rolling digest plus scalar totals; no programs, tracks, "
                "frames, per-track receipts, or tensors retained"
            ),
            "camera_ray_slice_work_count": camera_ray_slice_work_count,
            "camera_ray_slice_scalar_count": camera_ray_slice_scalar_count,
            "eligible_native_block_count": eligible_native_block_count,
            "active_native_block_count": active_native_block_count,
            "native_node_forward_launch_count": native_node_forward_launch_count,
            "native_sample_prepare_count": native_sample_prepare_count,
            "native_sample_launch_count": native_sample_launch_count,
            "native_sample_completion_fence_count": (
                native_sample_completion_fence_count
            ),
            "sample_completion_fence_call_count": (
                native_sample_completion_fence_count
            ),
            "reverse_completion_fence_call_count": (
                native_reverse_completion_fence_count
            ),
            "lane_construction_completion_fence_call_count": (
                native_lane_construction_completion_fence_count
            ),
            "bundle_materialization_completion_fence_call_count": (
                bundle_materialization_completion_fence_count
            ),
            "bundle_exhaustion_probe_completion_fence_call_count": (
                bundle_exhaustion_probe_completion_fence_count
            ),
            "top_initialization_completion_fence_call_count": 1,
            "lane_release_fence_call_count": 0,
            "lane_release_completion_boundary_count": (
                lane_release_completion_fence_count
            ),
            "lane_release_reuses_final_reverse_receipt": True,
            "optimizer_authorization_fence_call_count": 0,
            "optimizer_authorization_uses_fully_consumed_capability_ledger": True,
            "native_material_word_vjp_launch_count": (native_material_word_vjp_launch_count),
            "native_full_geometry_vjp_launch_count": (
                native_full_geometry_vjp_launch_count
            ),
            "native_fused_union_v2_vjp_launch_count": (
                native_fused_union_v2_vjp_launch_count
            ),
            "native_fused_union_v2_transaction_count": (
                bundle_count if fused_union_v2_full_geometry else 0
            ),
            "native_length_bar_tensor_bytes": native_length_bar_tensor_bytes,
            "native_union_construction_completion_fence_count": (
                native_union_construction_completion_fence_count
            ),
            "geometry_d2h_completion_fence_count": (
                geometry_d2h_completion_fence_count
            ),
            "streamed_sample_count": streamed_sample_count,
            "ordered_word_node_interactions": ordered_word_node_interactions,
            "sample_to_node_linear_interactions": (sample_to_node_linear_interactions),
            "sample_to_node_dense_fallback_interactions": (sample_to_node_dense_fallback_interactions),
            "peak_lane_resident_logical_tensor_bytes": (peak_lane_resident_logical_tensor_bytes),
            "peak_active_node_state_tensor_bytes": (peak_active_node_state_tensor_bytes),
            "peak_sample_launch_tensor_bytes": peak_sample_launch_tensor_bytes,
            "peak_decoded_frame_scratch_upper_bound_bytes": (peak_decoded_frame_scratch_upper_bound_bytes),
            "peak_selected_frame_target_tensor_upper_bound_bytes": (
                peak_selected_frame_target_tensor_upper_bound_bytes
            ),
            "selected_pixel_read_mode": selected_pixel_read_mode,
            "selected_pixel_read_source_provenance_manifest_digest": (
                selected_pixel_source_provenance_manifest_digest
            ),
            "selected_pixel_read_call_count": selected_pixel_read_call_count,
            "selected_pixel_read_observation_count": (
                selected_pixel_read_observation_count
            ),
            "selected_pixel_read_acceptance_capable": (
                selected_pixel_read_acceptance_capable
            ),
            "direct_selected_pixel_observation_count": (
                direct_selected_pixel_observation_count
            ),
            "bounded_region_selected_pixel_observation_count": (
                bounded_region_selected_pixel_observation_count
            ),
            "full_frame_fallback_observation_count": (
                full_frame_fallback_observation_count
            ),
            "full_frame_target_materialization_count": (
                full_frame_target_materialization_count
            ),
            "bounded_region_target_materialization_count": (
                bounded_region_target_materialization_count
            ),
            "peak_full_frame_materialization_tensor_bytes": (
                peak_full_frame_materialization_tensor_bytes
            ),
            "peak_bounded_region_materialization_tensor_bytes": (
                peak_bounded_region_materialization_tensor_bytes
            ),
            "peak_source_visible_target_read_logical_tensor_bytes_upper_bound": (
                peak_source_visible_target_read_tensor_bytes
            ),
            "peak_transient_mapped_address_space_bytes": (
                peak_transient_mapped_address_space_bytes
            ),
            "mapped_selected_pixel_read_call_count": (
                mapped_selected_pixel_read_call_count
            ),
            "mapping_closed_before_return_count": (
                mapping_closed_before_return_count
            ),
            "all_selected_pixel_mappings_closed_before_return": (
                mapping_closed_before_return_count
                == mapped_selected_pixel_read_call_count
            ),
            "target_source_decode_budget_enforced_before_allocation": True,
            "peak_coordinator_visible_live_tensor_upper_bound_bytes": (
                peak_coordinator_visible_live_tensor_upper_bound_bytes
            ),
            "target_frame_access_mode": memory_policy.target_frame_access_mode,
            "target_frame_cache_enabled": target_frame_cache_enabled,
            "target_frame_request_count": int(target_cache_accounting["request_count"]),
            "target_frame_decode_attempt_count": int(target_cache_accounting["decode_attempt_count"]),
            "target_frame_decode_count": int(target_cache_accounting["decode_count"]),
            "target_frame_cache_hit_count": int(target_cache_accounting["hit_count"]),
            "target_frame_cache_peak_resident_tensor_bytes": int(
                target_cache_accounting["peak_resident_frame_tensor_bytes"]
            ),
            "target_frame_cache_maximum_resident_tensor_bytes": int(target_cache_accounting["maximum_resident_bytes"]),
            "target_frame_cache_resident_tensor_bytes_after_close": int(
                target_cache_accounting["resident_frame_tensor_bytes"]
            ),
            "target_frame_cache_closed_before_optimizer": bool(target_cache_accounting["closed"]),
            "target_frame_cache_close_count": int(target_cache_accounting["close_count"]),
            "target_frame_cache_generation_digest": (
                target_frame_cache.generation_digest if target_frame_cache is not None else "none"
            ),
            "target_frame_cache_provenance": (
                target_frame_cache.provenance if target_frame_cache is not None else "none"
            ),
            "target_frame_cache_residency_scaling": (target_frame_residency_scaling),
            "decode_once_cache_target_residency_bound": (
                "O(unique_selected_frames * H * W * 3 * sizeof(float32)); "
                "no eviction, so every unique selected frame must fit the explicit "
                "step-cache cap or the step fails before the next decode"
            ),
            "target_frame_cache_no_eviction": target_frame_cache_enabled,
            "target_frame_cache_success_requires_all_unique_frames_fit_budget": (target_frame_cache_enabled),
            "target_frame_decode_work_scaling": target_decode_work_scaling,
            "caller_global_material_tensor_bytes": _unique_tensor_storage_bytes((global_site_rgba_f32,)),
            "caller_global_material_bar_tensor_bytes": _unique_tensor_storage_bytes((global_grad_site_rgba_f32,)),
            "caller_global_material_bar_count": 1,
            "optimizer_update_authorization_count": 1,
            "observation_manifest_digest": observed_manifest_digest,
            "native_lane_generation_manifest_digest": (native_generation_manifest_digest),
            "native_abi_identity_digest": _digest_parts(step_native_abi_identity),
            "material_generation_id": material_generation_id,
            "background_generation_id": background_generation_id,
            "device_completion_fence_call_count": fence_call_count,
            "device_completion_fence_provenance": CAPABILITY_PROVENANCE,
            "sealed_completion_fence_generation_digest": (
                sealed_completion_fence.generation_digest
            ),
            "sealed_completion_owner_generation_digest": (
                sealed_completion_fence.owner_generation_digest
            ),
            "sealed_completion_fence_success_count": (
                sealed_completion_fence.successful_fence_count
            ),
            "sealed_completion_receipt_consumption_count": (
                sealed_completion_fence.consumed_fence_count
            ),
            "sealed_completion_outstanding_receipt_count": 0,
            "caller_supplied_completion_callback_count": 0,
            "caller_supplied_completion_provenance_count": 0,
            "block_major_bundle_streaming": True,
            "frame_major_selected_pixel_streaming_inside_bundle": True,
            "frame_major_target_streaming_inside_bundle": True,
            "provider_owned_retained_bundle_count": 0,
            "consumer_release_required_for_one_lane_peak": True,
            "factory_nonretention_self_attested_source_contract": True,
            "provider_outer_cold_certification_count": 1,
            "provider_inner_checks_are_warm": True,
            "memory_policy_preflight_enforced": True,
            "max_lane_resident_logical_tensor_bytes": (memory_policy.max_lane_resident_logical_tensor_bytes),
            "max_active_node_and_vjp_tensor_bytes": (memory_policy.max_active_node_and_vjp_tensor_bytes),
            "max_coordinator_visible_live_tensor_bytes": (memory_policy.max_coordinator_visible_live_tensor_bytes),
            "max_step_target_frame_cache_tensor_bytes": (memory_policy.max_step_target_frame_cache_tensor_bytes),
            "persistent_frame_tensor_bytes": 0,
            "persistent_target_tensor_bytes": 0,
            "persistent_prediction_tensor_bytes": 0,
            "dense_track_frame_tensor_bytes": 0,
            "sparse_sampled_observations_only": True,
            "dense_f_replayable_observation_source_implemented": False,
            "geometry_parameter_vjp_implemented": (
                _full_geometry_context is not None
            ),
            "native_runtime_verified": False,
            "accelerator_execution_fail_closed_until_async_quarantine": True,
            "bounded_trainer_owned_async_failure_quarantine_implemented": True,
            "async_failure_quarantine_is_single_carrier_not_history": True,
            "sample_reverse_lane_fence_failures_require_restart": True,
            "optimizer_authorized_only_after_all_completion_fences": True,
            "partial_lane_construction_provisional_lease_implemented": True,
            "sparse_sample_transfer_predecessor_lease_implemented": True,
            "pre_lane_bundle_device_construction_lease_implemented": True,
            "bundle_transfer_predecessors_released_after_proven_fence": True,
            "bundle_construction_lifetime_strong_reference_cycle": False,
            "spent_bundle_retired_after_proven_fence": True,
            "forward_compact_gather_provisional_lease_implemented": True,
            "top_level_device_zero_transaction_lifetime_implemented": True,
            "union_cold_device_to_host_receipt_lifetime_implemented": False,
            "caller_owned_native_forward_output_lifetime_implemented": True,
            "native_forward_into_binding_shader_source_implemented": True,
            "native_forward_into_compiled_registration_verified": False,
            "native_forward_internal_output_lifetime_verified": False,
            "accelerator_gate_blocker": (
                "union-map cold device-to-host receipts, native caller-owned "
                "forward-into rebuild/registration/parity, "
                "canonical device-specific completion-fence capability, and "
                "native allocator/runtime behavior remain unproven"
            ),
            "allocator_peak_measured": False,
            "device_async_release_peak_measured": False,
            "programs_recompiled_each_optimizer_step": True,
            "bounded_precompiled_lane_store_implemented": False,
            "program_compile_work_scaling": (
                "current source coordinator recompiles each selected track/bundle per step"
            ),
            "word_work_scaling": (
                "O(sum_active_blocks J_b*W_b), independent of sparse sample density at fixed compiled bundles"
            ),
            "sample_work_scaling": "O(streamed_samples*selected_J)",
            "live_tensor_scaling": (
                "coordinator-visible logical tensors: O(S global material/bar + "
                "max one fenced spatial lane/native-node state + one sealed "
                "selected-pixel read + bounded sample state); the compatibility "
                "decode-once target cache is separately reported as O(unique frames*H*W)"
            ),
        }
        if _full_geometry_context is None:
            provisional = PaperKineticLazyNativeMaterialStepResult(
                step_index=step_index,
                step_generation_id=step_generation_id,
                provider_generation_digest=provider.generation_digest,
                world_generation_digest=provider.world.generation_digest,
                sites_content_digest=provider.world.sites_content_digest,
                loss_normalization_id=loss_normalization_id,
                material_generation_id=material_generation_id,
                background_generation_id=background_generation_id,
                loss_f32=loss_f32,
                grad_global_site_rgba_f32=global_grad_site_rgba_f32,
                accounting=MappingProxyType(accounting),
                generation_digest="",
                _tensor_signatures=tuple(
                    _tensor_signature(tensor)
                    for tensor in (loss_f32, global_grad_site_rgba_f32)
                ),
                _material_tensor_identity=id(global_site_rgba_f32),
                _material_tensor_signature=_tensor_signature(global_site_rgba_f32),
                _background_tensor_identity=id(background_rgb_f32),
                _background_tensor_signature=_tensor_signature(background_rgb_f32),
                _sealed_completion_fence=sealed_completion_fence,
                _sealed_completion_fence_identity=id(sealed_completion_fence),
                _sealed_completion_fence_generation_digest=(
                    sealed_completion_fence.generation_digest
                ),
                _sealed_generation_digest="",
                _seal=_RESULT_SEAL,
            )
            generation_digest = _result_digest(provisional)
            result = PaperKineticLazyNativeMaterialStepResult(
                **{
                    **provisional.__dict__,
                    "generation_digest": generation_digest,
                    "_sealed_generation_digest": generation_digest,
                }
            )
        else:
            result = _full_geometry_context.build_result(
                step_index=step_index,
                step_generation_id=step_generation_id,
                provider_generation_digest=provider.generation_digest,
                world_generation_digest=provider.world.generation_digest,
                sites_content_digest=provider.world.sites_content_digest,
                loss_normalization_id=loss_normalization_id,
                material_generation_id=material_generation_id,
                background_generation_id=background_generation_id,
                loss_f32=loss_f32,
                grad_global_site_rgba_f32=global_grad_site_rgba_f32,
                material_tensor=global_site_rgba_f32,
                background_tensor=background_rgb_f32,
                sealed_completion_fence=sealed_completion_fence,
                accounting=accounting,
            )
        result.assert_current()
        optimizer_authorization_started = True
        optimizer_update(result)
        result.assert_current()
        final_consumed_sequence = sealed_completion_fence.consumed_fence_count
        device_transaction_lifetime.retire_after_all_completion_receipts(
            sealed_completion_fence,
            expected_last_consumed_sequence=final_consumed_sequence,
        )
        state._active_device_transaction_lifetime = None
        state.optimizer_callback_count += 1
        state.last_completed_step_index = step_index
        state.next_step_index = step_index + 1
        state.active_step_index = None
        owns_active_state = False
        state.assert_current(provider)
    except BaseException as error:
        cleanup_errors: list[BaseException] = []
        async_quarantine = state._async_failure_quarantine
        partial_bundle_lifetime = (
            bundle_construction_lifetime_slot.active_lifetime
            if bundle_construction_lifetime_slot is not None
            else None
        )
        original_traceback = error.__traceback__
        retained_step_references = (
            ("provider", provider),
            ("bundle_iterator", bundle_iterator),
            (
                "bundle_construction_lifetime_slot",
                bundle_construction_lifetime_slot,
            ),
            ("bundle_construction_lifetime", partial_bundle_lifetime),
            (
                "top_level_device_transaction_lifetime",
                device_transaction_lifetime,
            ),
            ("native_ops", native_ops),
            ("global_site_rgba_f32", global_site_rgba_f32),
            ("global_grad_site_rgba_f32", global_grad_site_rgba_f32),
            ("background_rgb_f32", background_rgb_f32),
            ("loss_f32", loss_f32),
            ("cone_diagnostic_i32", cone_diagnostic_i32),
            ("sealed_completion_fence", sealed_completion_fence),
            (
                "accelerator_stage_settlement_slot",
                accelerator_stage_settlement_slot,
            ),
            ("bundle_materialization_epoch", bundle_materialization_epoch),
            ("bundle_materialization_receipt", bundle_materialization_receipt),
            (
                "bundle_materialization_spatial_lifetime",
                bundle_materialization_spatial_lifetime,
            ),
        )
        foreign_subject_epoch = (
            sealed_completion_fence is not None
            and sealed_completion_fence.registered_launch_epoch is not None
            and sealed_completion_fence.registered_launch_epoch.subject_binding
            is not None
            and (
                accelerator_stage_settlement_slot is None
                or sealed_completion_fence.registered_launch_epoch.subject_binding
                is not accelerator_stage_settlement_slot.subject_binding
            )
        )
        foreign_subject_receipt = (
            sealed_completion_fence is not None
            and sealed_completion_fence.outstanding_subject_binding_identity
            is not None
            and (
                accelerator_stage_settlement_slot is None
                or sealed_completion_fence.outstanding_subject_binding_identity
                != id(accelerator_stage_settlement_slot.subject_binding)
            )
        )
        if (
            async_quarantine is None
            and (
                foreign_subject_epoch
                or foreign_subject_receipt
                or accelerator_stage_settlement_slot is not None
                and accelerator_stage_settlement_slot.phase != "bound"
            )
        ):
            _quarantine_lazy_async_failure(
                state,
                stage=(
                    "optimizer-gradient-readback"
                    if foreign_subject_epoch or foreign_subject_receipt
                    else accelerator_stage_settlement_slot.stage
                    or "accelerator-general-stage"
                ),
                original_error=error,
                original_traceback=original_traceback,
                failed_completion_fence_error=error,
                retained_references=retained_step_references,
                completion_fence_generation_digest=(
                    sealed_completion_fence.generation_digest
                ),
            )
            error.add_note(
                "lazy accelerator stage failed before its exact subject-bound "
                "receipt commit; roots are quarantined and restart is required"
            )
            raise error.with_traceback(original_traceback)
        error_settlement_receipt: PaperKineticCompletionFenceReceipt | None = None
        error_settlement_sequence: int | None = None
        error_settlement_stage = (
            "lazy-bundle-construction"
            if partial_bundle_lifetime is not None
            else "top-level-device-transaction"
        )
        error_settlement_launch_digest = ""
        needs_error_settlement = (
            async_quarantine is None
            and device_transaction_lifetime is not None
            and device_transaction_lifetime.phase != "released"
        )
        if needs_error_settlement:
            if sealed_completion_fence is None:
                raise RuntimeError(
                    "top-level device work lost its sealed completion authority"
                ) from error
            try:
                registered_epoch = sealed_completion_fence.registered_launch_epoch
                if registered_epoch is not None:
                    error_settlement_stage = registered_epoch.stage
                    error_settlement_launch_digest = (
                        registered_epoch.launch_generation_digest
                    )
                    error_settlement_sequence = (
                        registered_epoch.launch_epoch_sequence
                    )
                    error_settlement_receipt = (
                        _fence_registered_completion_epoch(
                            sealed_completion_fence,
                            registered_epoch,
                            expected_fence_sequence=error_settlement_sequence,
                        )
                    )
                elif sealed_completion_fence.outstanding_receipt_sequence is not None:
                    candidate_receipt = bundle_materialization_receipt
                    candidate_launch_digest = bundle_materialization_launch_digest
                    if candidate_receipt is None or candidate_receipt.consumed:
                        candidate_receipt = top_initialization_receipt
                        candidate_launch_digest = top_initialization_launch_digest
                    if candidate_receipt is None or candidate_receipt.consumed:
                        raise RuntimeError(
                            "outstanding top-level completion receipt is unavailable"
                        )
                    error_settlement_receipt = candidate_receipt
                    error_settlement_stage = candidate_receipt.stage
                    error_settlement_launch_digest = candidate_launch_digest
                    error_settlement_sequence = (
                        sealed_completion_fence.outstanding_receipt_sequence
                    )
                else:
                    sealed_completion_fence.assert_current()
            except BaseException as fence_error:
                _quarantine_lazy_async_failure(
                    state,
                    stage=error_settlement_stage,
                    original_error=error,
                    original_traceback=original_traceback,
                    failed_completion_fence_error=fence_error,
                    retained_references=retained_step_references,
                    completion_fence_generation_digest=(
                        sealed_completion_fence.generation_digest
                    ),
                )
                error.add_note(
                    "lazy step error fence failed; device transaction roots "
                    "are quarantined and the process must restart"
                )
                raise error.with_traceback(original_traceback) from fence_error
        construction_cleanup_required = (
            partial_bundle_lifetime is not None
            or bundle_materialization_spatial_lifetime is not None
        )
        if async_quarantine is None and construction_cleanup_required:
            try:
                if partial_bundle_lifetime is not None:
                    bundle_construction_lifetime_slot.assert_active_releasable_after_consumed_receipt()
                if bundle_materialization_spatial_lifetime is not None:
                    if bundle_materialization_release_kind == "retire":
                        bundle_materialization_spatial_lifetime.assert_retirable_after_consumed_receipt()
                    elif (
                        bundle_materialization_release_kind
                        == "transfer-predecessors"
                    ):
                        bundle_materialization_spatial_lifetime.assert_transfer_predecessors_releasable_after_consumed_receipt()
                    else:
                        raise RuntimeError(
                            "bundle materialization cleanup lost its release kind"
                        )
                if error_settlement_receipt is not None:
                    error_settlement_receipt.assert_for(
                        sealed_completion_fence,
                        stage=error_settlement_stage,
                        launch_generation_digest=error_settlement_launch_digest,
                        fence_sequence=error_settlement_sequence,
                    )
                    error_settlement_receipt.consume_for(
                        sealed_completion_fence,
                        stage=error_settlement_stage,
                        launch_generation_digest=error_settlement_launch_digest,
                        fence_sequence=error_settlement_sequence,
                        consumer="lazy-bundle-construction-composite-release",
                    )
                    error_settlement_receipt = None
                else:
                    sealed_completion_fence.assert_current()
                    if (
                        sealed_completion_fence.registered_launch_epoch
                        is not None
                        or sealed_completion_fence.outstanding_receipt_sequence
                        is not None
                    ):
                        raise RuntimeError(
                            "bundle construction cleanup lost completion authority"
                        )
                if partial_bundle_lifetime is not None:
                    bundle_construction_lifetime_slot._commit_active_release_after_consumed_receipt()
                if bundle_materialization_spatial_lifetime is not None:
                    if bundle_materialization_release_kind == "retire":
                        bundle_materialization_spatial_lifetime._commit_retire_after_consumed_receipt()
                    else:
                        bundle_materialization_spatial_lifetime._commit_transfer_predecessors_after_consumed_receipt()
                bundle_materialization_spatial_lifetime = None
                bundle_materialization_release_kind = ""
            except BaseException as fence_error:
                _quarantine_lazy_async_failure(
                    state,
                    stage=error_settlement_stage,
                    original_error=error,
                    original_traceback=original_traceback,
                    failed_completion_fence_error=fence_error,
                    retained_references=retained_step_references,
                    completion_fence_generation_digest=(
                        sealed_completion_fence.generation_digest
                    ),
                )
                error.add_note(
                    "lazy bundle-construction settlement failed; retained "
                    "roots are quarantined and the process must restart"
                )
                raise error.with_traceback(original_traceback) from fence_error
        if (
            async_quarantine is None
            and target_frame_cache is not None
            and not target_frame_cache.closed
        ):
            try:
                target_frame_cache.close()
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        cleanup_zero_requires_fence = False
        cleanup_bar_published = False
        cleanup_epoch: PaperKineticCompletionLaunchEpoch | None = None
        cleanup_sequence: int | None = None
        cleanup_launch_digest = ""
        if error_settlement_receipt is not None:
            error_settlement_receipt.consume_for(
                sealed_completion_fence,
                stage=error_settlement_stage,
                launch_generation_digest=error_settlement_launch_digest,
                fence_sequence=error_settlement_sequence,
                consumer="lazy-error-settlement-release",
            )
        if (
            async_quarantine is None
            and device_transaction_lifetime is not None
            and device_transaction_lifetime.phase
            in {"global_bar_zeroing", "active"}
        ):
            cleanup_launch_digest = _digest_parts(
                "lazy-top-level-cleanup-zero-v1",
                step_generation_id,
                _tensor_signature(global_grad_site_rgba_f32),
                None if loss_f32 is None else _tensor_signature(loss_f32),
            )
            cleanup_sequence = sealed_completion_fence.next_fence_sequence
            device_transaction_lifetime.begin_cleanup_zero()
            cleanup_epoch = _register_general_completion_epoch(
                sealed_completion_fence,
                accelerator_stage_settlement_slot,
                stage="top-level-cleanup-zero",
                launch_generation_digest=cleanup_launch_digest,
                roots=tuple(
                    root
                    for root in (
                        (
                            "device_transaction_lifetime",
                            device_transaction_lifetime,
                        ),
                        ("global_bar", global_grad_site_rgba_f32),
                        ("loss", loss_f32),
                    )
                    if root[1] is not None
                ),
            )
            try:
                cleanup_zero_requires_fence = True
                cleanup_bar_result = global_grad_site_rgba_f32.zero_()
                device_transaction_lifetime.global_bar_zero_result = (
                    cleanup_bar_result
                )
                device_transaction_lifetime.assert_retained()
                cleanup_bar_published = True
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if (
            async_quarantine is None
            and cleanup_bar_published
            and loss_f32 is not None
        ):
            try:
                cleanup_loss_result = loss_f32.zero_()
                if cleanup_loss_result is not loss_f32:
                    raise ValueError("top-level loss zero returned foreign storage")
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if async_quarantine is None and cleanup_zero_requires_fence:
            try:
                cleanup_receipt = _fence_registered_completion_epoch(
                    sealed_completion_fence,
                    cleanup_epoch,
                    expected_fence_sequence=cleanup_sequence,
                )
            except BaseException as fence_error:
                _quarantine_lazy_async_failure(
                    state,
                    stage="top-level-cleanup-zero",
                    original_error=error,
                    original_traceback=original_traceback,
                    failed_completion_fence_error=fence_error,
                    retained_references=retained_step_references,
                    completion_fence_generation_digest=(
                        sealed_completion_fence.generation_digest
                    ),
                )
                error.add_note(
                    "lazy cleanup-zero fence failed; device transaction roots "
                    "are quarantined and the process must restart"
                )
                raise error.with_traceback(original_traceback) from fence_error
            device_transaction_lifetime.assert_releasable_for_sealed_completion_receipt(
                cleanup_receipt,
                stage="top-level-cleanup-zero",
                launch_generation_digest=cleanup_launch_digest,
                expected_fence_sequence=cleanup_sequence,
            )
            _consume_general_completion_receipt(
                sealed_completion_fence,
                accelerator_stage_settlement_slot,
                cleanup_receipt,
                consumer="top-level-device-transaction-release",
            )
            device_transaction_lifetime._commit_release_after_consumed_receipt()
            state._active_device_transaction_lifetime = None
        elif (
            async_quarantine is None
            and device_transaction_lifetime is not None
            and device_transaction_lifetime.phase != "released"
        ):
            settled_sequence = sealed_completion_fence.consumed_fence_count
            device_transaction_lifetime.retire_after_all_completion_receipts(
                sealed_completion_fence,
                expected_last_consumed_sequence=settled_sequence,
            )
            state._active_device_transaction_lifetime = None
        if async_quarantine is None and owns_active_state:
            state.active_step_index = None
        if async_quarantine is not None:
            # A failed completion fence means these tensors may still be in
            # use.  No close/zero/release command is safe, and retrying this
            # trainer would let the same storage race unknown native work.
            state.poisoned = True
            async_quarantine.assert_current()
        elif optimizer_authorization_started:
            state.poisoned = True
        for cleanup_error in cleanup_errors:
            error.add_note(f"lazy native step cleanup also failed: {type(cleanup_error).__name__}: {cleanup_error}")
        raise
    else:
        return result
    finally:
        execution_lock.release()


def paper_kinetic_observation_manifest_digest(
    observations: Iterable[PaperKineticObservation],
) -> str:
    """Hash one replayable canonical observation stream without retaining it."""

    accumulator = _ManifestAccumulator("observation-identities")
    count = 0
    for observation in observations:
        if not isinstance(observation, PaperKineticObservation):
            raise TypeError("observation manifest requires PaperKineticObservation values")
        accumulator.add(observation.sample_identity)
        count += 1
    if count < 1:
        raise ValueError("observation manifest cannot be empty")
    return accumulator.finish(count)


class _ManifestAccumulator:
    """Constant-memory ordered SHA-256 accumulator with an exact count seal."""

    def __init__(self, domain: str) -> None:
        if not isinstance(domain, str) or not domain.strip():
            raise ValueError("manifest domain must be nonempty")
        self._digest = hashlib.sha256()
        self._count = 0
        self._finished = False
        self._update((STEP_PROVENANCE, domain))

    def add(self, part: object) -> None:
        if self._finished:
            raise ValueError("manifest accumulator is already sealed")
        self._update(part)
        self._count += 1

    def finish(self, expected_count: int) -> str:
        if self._finished:
            raise ValueError("manifest accumulator is already sealed")
        if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count < 0:
            raise ValueError("manifest expected count must be a nonnegative integer")
        if self._count != expected_count:
            raise ArithmeticError("manifest accumulator count changed")
        self._update(("final-count", expected_count))
        self._finished = True
        return self._digest.hexdigest()

    def _update(self, part: object) -> None:
        encoded = repr(part).encode("utf-8")
        self._digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        self._digest.update(encoded)


def _execute_native_bundle(
    state: PaperKineticLazyNativeTrainerState,
    bundle: PaperKineticLazyProgramBundle,
    provider: PaperKineticLazyProgramBundleProvider,
    *,
    plan: PaperKineticSparseSamplePlan,
    target_frame_cache: PaperKineticStepTargetFrameCache | None,
    global_site_rgba_f32: torch.Tensor,
    global_grad_site_rgba_f32: torch.Tensor,
    background_rgb_f32: torch.Tensor,
    loss_f32: torch.Tensor,
    cone_diagnostic_i32: torch.Tensor,
    native_ops: Any,
    backend_provenance: str,
    step_generation_id: str,
    memory_policy: PaperKineticLazyNativeMemoryPolicy,
    sealed_completion_fence: PaperKineticSealedCompletionFence,
    accelerator_stage_settlement_slot: (
        _AcceleratorStageSettlementSlot | None
    ),
    top_level_device_transaction_lifetime: (
        _TopLevelDeviceTransactionLifetime
    ),
    cone_tolerance: float,
    full_geometry_context: Any | None = None,
) -> _BundleExecutionResult:
    """Fence samples, each reverse scratch, and the lane before releasing roots.

    A failed completion fence has unknown semantics.  It is never retried and
    no local root is cleared: the complete bounded lane lifetime is moved to
    the trainer-owned restart-required quarantine instead.
    """

    lane = None
    lane_construction_lifetime = None
    session = None
    sample_iterator = None
    sample_iterator_closed = False
    prelaunched_sample_blocks = None
    prelaunched_sample_blocks_closed = False
    active: dict[str, _ActiveNativeBlockState] = {}
    current_sample_block = None
    current_sample_lifetime = None
    current_sample_transfer_lifetime = None
    current_sample_composite_slot = None
    current_pending_sample_completion = None
    current_sample_release_commit_plan = None
    current_forward_runtime = None
    current_compact_gather_lifetime = None
    current_forward_compact_material = None
    current_forward_into_lifetime = None
    current_forward_node_chart_out = None
    current_forward_token = None
    current_forward_grad_node_chart = None
    current_forward_loss = None
    current_reverse_runtime = None
    current_reverse_block_state = None
    current_reverse_compact_bar = None
    current_material_execution = None
    current_completion_receipt: PaperKineticCompletionFenceReceipt | None = None
    current_completion_launch_epoch: PaperKineticCompletionLaunchEpoch | None = None
    current_completion_stage = ""
    current_completion_launch_generation_digest = ""
    current_completion_expected_fence_sequence: int | None = None
    completion_unknown_stage: str | None = None
    direct_completion_fence_call_count = 0
    ordered_word_node_interactions = 0
    sample_to_node_linear_interactions = 0
    sample_to_node_dense_fallback_interactions = 0
    peak_active_node_state_tensor_bytes = 0
    peak_sample_launch_tensor_bytes = 0
    sample_completion_fence_call_count = 0
    reverse_completion_fence_call_count = 0
    final_reverse_receipt: PaperKineticCompletionFenceReceipt | None = None
    final_reverse_launch_digest = ""
    final_reverse_sequence: int | None = None

    def fence_registered_completion_epoch(
        launch_epoch: PaperKineticCompletionLaunchEpoch,
        *,
        expected_fence_sequence: int,
    ) -> tuple[PaperKineticCompletionFenceReceipt, int]:
        nonlocal completion_unknown_stage
        nonlocal direct_completion_fence_call_count
        direct_completion_fence_call_count += 1
        try:
            receipt = sealed_completion_fence.fence(launch_epoch)
            return receipt, expected_fence_sequence
        except BaseException:
            completion_unknown_stage = launch_epoch.stage
            raise

    def lifetime_references() -> tuple[tuple[str, Any], ...]:
        outstanding = (
            session._outstanding_sample_lifetime
            if session is not None
            else None
        )
        return (
            ("provider", provider),
            ("bundle", bundle),
            ("sample_plan", plan),
            ("target_frame_cache", target_frame_cache),
            ("sample_iterator", sample_iterator),
            ("native_ops", native_ops),
            ("lane_construction_lifetime", lane_construction_lifetime),
            ("native_lane", lane),
            ("native_session", session),
            ("active_blocks", active),
            ("current_sample_block", current_sample_block),
            (
                "current_sample_transfer_lifetime",
                current_sample_transfer_lifetime,
            ),
            ("current_sample_lifetime", current_sample_lifetime),
            (
                "current_sample_composite_slot",
                current_sample_composite_slot,
            ),
            (
                "current_pending_sample_completion",
                current_pending_sample_completion,
            ),
            (
                "current_sample_release_commit_plan",
                current_sample_release_commit_plan,
            ),
            ("session_outstanding_sample_lifetime", outstanding),
            ("current_forward_runtime", current_forward_runtime),
            (
                "current_compact_gather_lifetime",
                current_compact_gather_lifetime,
            ),
            (
                "current_forward_compact_material",
                current_forward_compact_material,
            ),
            (
                "current_forward_into_lifetime",
                current_forward_into_lifetime,
            ),
            (
                "current_forward_node_chart_out",
                current_forward_node_chart_out,
            ),
            ("current_forward_token", current_forward_token),
            (
                "current_forward_grad_node_chart",
                current_forward_grad_node_chart,
            ),
            ("current_forward_loss", current_forward_loss),
            ("current_reverse_runtime", current_reverse_runtime),
            ("current_reverse_block_state", current_reverse_block_state),
            ("current_reverse_compact_bar", current_reverse_compact_bar),
            ("current_material_execution", current_material_execution),
            ("current_completion_receipt", current_completion_receipt),
            ("current_completion_launch_epoch", current_completion_launch_epoch),
            ("sealed_completion_fence", sealed_completion_fence),
            (
                "accelerator_stage_settlement_slot",
                accelerator_stage_settlement_slot,
            ),
            ("global_site_rgba_f32", global_site_rgba_f32),
            ("global_grad_site_rgba_f32", global_grad_site_rgba_f32),
            ("background_rgb_f32", background_rgb_f32),
            ("loss_f32", loss_f32),
            ("cone_diagnostic_i32", cone_diagnostic_i32),
            (
                "top_level_device_transaction_lifetime",
                top_level_device_transaction_lifetime,
            ),
        )

    def close_sample_iterator_after_proven_completion() -> None:
        nonlocal sample_iterator_closed
        nonlocal prelaunched_sample_blocks_closed
        if (
            prelaunched_sample_blocks is not None
            and not prelaunched_sample_blocks_closed
        ):
            prelaunched_sample_blocks.close()
            prelaunched_sample_blocks_closed = True
        if sample_iterator is not None and not sample_iterator_closed:
            if sample_iterator.active_transfer_lifetime is not None:
                raise RuntimeError(
                    "sample iterator close requires its prevalidated commit"
                )
            sample_iterator.close()
            sample_iterator_closed = True

    def unique_forward_and_gather_lifetimes() -> tuple[list[Any], list[Any]]:
        forward_lifetimes = [
            block_state.forward_into_lifetime for block_state in active.values()
        ]
        gather_lifetimes = [
            block_state.compact_gather_lifetime for block_state in active.values()
        ]
        if isinstance(
            current_forward_into_lifetime,
            KineticNativeNodeForwardIntoLifetime,
        ):
            forward_lifetimes.append(current_forward_into_lifetime)
        if isinstance(current_compact_gather_lifetime, _CompactMaterialGatherLifetime):
            gather_lifetimes.append(current_compact_gather_lifetime)
        return (
            list({id(value): value for value in forward_lifetimes}.values()),
            list({id(value): value for value in gather_lifetimes}.values()),
        )

    try:
        lane_construction_launch_digest = _digest_parts(
            "lazy-native-lane-construction-v2",
            step_generation_id,
            bundle.generation_digest,
            provider.generation_digest,
            _lazy_native_abi_identity(native_ops),
            memory_policy.max_lane_resident_logical_tensor_bytes,
            str(global_site_rgba_f32.device),
        )
        lane_construction_sequence = sealed_completion_fence.next_fence_sequence
        lane_construction_epoch = _register_general_completion_epoch(
            sealed_completion_fence,
            accelerator_stage_settlement_slot,
            stage="lane-construction",
            launch_generation_digest=lane_construction_launch_digest,
            roots=(
                ("bundle", bundle),
                ("provider", provider),
                ("sample_plan", plan),
                ("native_ops", native_ops),
                ("global_material", global_site_rgba_f32),
                ("global_bar", global_grad_site_rgba_f32),
                (
                    "top_level_device_transaction_lifetime",
                    top_level_device_transaction_lifetime,
                ),
            ),
        )
        current_completion_launch_epoch = lane_construction_epoch
        current_completion_stage = "lane-construction"
        current_completion_launch_generation_digest = (
            lane_construction_launch_digest
        )
        current_completion_expected_fence_sequence = lane_construction_sequence
        lane_construction_lifetime = (
            prepare_paper_kinetic_native_lazy_bundle_lane_construction_lifetime(
                bundle,
                provider,
                native_ops,
                device=global_site_rgba_f32.device,
                backend_provenance=backend_provenance,
                max_resident_logical_tensor_bytes=(
                    memory_policy.max_lane_resident_logical_tensor_bytes
                ),
            )
        )
        _extend_general_completion_roots(
            accelerator_stage_settlement_slot,
            (("lane_construction_lifetime", lane_construction_lifetime),),
        )
        lane = materialize_paper_kinetic_native_lazy_bundle_lane(
            lane_construction_lifetime
        )
        _extend_general_completion_roots(
            accelerator_stage_settlement_slot,
            (("native_lane", lane),),
        )
        lane.assert_cold_current(provider)
        lane_construction_receipt, observed_construction_sequence = (
            fence_registered_completion_epoch(
                lane_construction_epoch,
                expected_fence_sequence=lane_construction_sequence,
            )
        )
        current_completion_receipt = lane_construction_receipt
        if observed_construction_sequence != lane_construction_sequence:
            _fail_arithmetic("lane construction completion sequence changed")
        spatial_construction_lifetime = (
            bundle.spatial_bundle._construction_lifetime
        )
        spatial_construction_lifetime.assert_transfer_predecessors_releasable_after_consumed_receipt()
        _consume_general_completion_receipt(
            sealed_completion_fence,
            accelerator_stage_settlement_slot,
            lane_construction_receipt,
            consumer="lazy-native-lane-construction-predecessor-release",
        )
        spatial_construction_lifetime._commit_transfer_predecessors_after_consumed_receipt()
        current_completion_receipt = None
        current_completion_launch_epoch = None
        current_completion_stage = ""
        current_completion_launch_generation_digest = ""
        current_completion_expected_fence_sequence = None
        session = lane.executor.begin_step(
            step_generation_id=(
                f"{step_generation_id}:bundle:{bundle.bundle_index}"
            ),
            requested_observation_count=bundle.observation_count,
        )
        sample_iterator = iter_paper_kinetic_sparse_sample_blocks(
            plan,
            target_frame_cache=target_frame_cache,
            maximum_source_decode_tensor_bytes=(
                memory_policy.max_decoded_frame_scratch_tensor_bytes
            ),
            require_explicit_transfer_settlement=True,
        )

        def iter_prelaunched_sample_blocks():
            nonlocal current_sample_block
            nonlocal current_sample_composite_slot
            nonlocal current_sample_transfer_lifetime
            nonlocal current_completion_expected_fence_sequence
            nonlocal current_completion_launch_epoch
            nonlocal current_completion_launch_generation_digest
            nonlocal current_completion_stage
            covered_sample_count = 0
            launch_ordinal = 0
            while covered_sample_count < plan.observation_count:
                sample_composite_slot = (
                    _prepare_sample_composite_settlement_slot(
                        step_generation_id=step_generation_id,
                        bundle_generation_digest=bundle.generation_digest,
                        plan=plan,
                        session=session,
                        stream=sample_iterator,
                        launch_ordinal=launch_ordinal,
                        covered_sample_count_before_launch=(
                            covered_sample_count
                        ),
                    )
                )
                current_sample_composite_slot = sample_composite_slot
                subject_binding = (
                    prepare_paper_kinetic_completion_subject_binding(
                        sealed_completion_fence,
                        sample_composite_slot,
                        kind=SAMPLE_COMPOSITE_SUBJECT_KIND,
                        subject_generation_digest=(
                            sample_composite_slot.generation_digest
                        ),
                    )
                )
                sample_composite_slot.bind_subject(subject_binding)
                sample_launch_digest = _digest_parts(
                    "lazy-native-sample-materialize-and-launch-v2",
                    session.generation_id,
                    plan.generation_digest,
                    launch_ordinal,
                    covered_sample_count,
                    id(sample_iterator),
                    sample_composite_slot.generation_digest,
                    subject_binding.generation_digest,
                    _tensor_signature(global_site_rgba_f32),
                    _tensor_signature(background_rgb_f32),
                    _tensor_signature(loss_f32),
                    _tensor_signature(cone_diagnostic_i32),
                )
                expected_sequence = sealed_completion_fence.next_fence_sequence
                launch_epoch = sealed_completion_fence.register_launch(
                    stage="sample-completion",
                    launch_generation_digest=sample_launch_digest,
                    subject_binding=subject_binding,
                )
                sample_composite_slot.register_epoch(launch_epoch)
                sample_composite_slot.extend_roots(
                    (
                        ("native_lane", lane),
                        ("active_blocks", active),
                        ("global_material", global_site_rgba_f32),
                        ("global_bar", global_grad_site_rgba_f32),
                        ("background", background_rgb_f32),
                        ("global_loss", loss_f32),
                        ("cone_diagnostic", cone_diagnostic_i32),
                    )
                )
                current_completion_launch_epoch = launch_epoch
                current_completion_stage = "sample-completion"
                current_completion_launch_generation_digest = sample_launch_digest
                current_completion_expected_fence_sequence = expected_sequence
                sample_block = next(sample_iterator)
                transfer_lifetime = sample_iterator.active_lifetime_for(
                    sample_block
                )
                current_sample_block = sample_block
                current_sample_transfer_lifetime = transfer_lifetime
                sample_composite_slot.publish_materialization(
                    sample_block,
                    transfer_lifetime,
                )
                covered_sample_count += sample_block.sample_count
                if covered_sample_count > plan.observation_count:
                    _fail_arithmetic(
                        "sparse sample stream exceeded its predeclared coverage"
                    )
                yield (
                    sample_block,
                    sample_composite_slot,
                    launch_epoch,
                    sample_launch_digest,
                    expected_sequence,
                )
                del sample_block, sample_composite_slot, transfer_lifetime
                launch_ordinal += 1
            if covered_sample_count != plan.observation_count:
                _fail_arithmetic("sparse sample stream ended before exact coverage")

        prelaunched_sample_blocks = iter_prelaunched_sample_blocks()
        while True:
            try:
                (
                    sample_block,
                    sample_composite_slot,
                    sample_launch_epoch,
                    sample_launch_digest,
                    expected_sample_fence_sequence,
                ) = next(prelaunched_sample_blocks)
            except StopIteration:
                break
            if current_sample_composite_slot is not sample_composite_slot:
                raise ValueError("sample composite slot publication changed")
            digest = sample_block.native_block_generation_digest
            block_state = active.get(digest)
            if block_state is None:
                runtime = lane.runtime_for_native_block_digest(digest)
                current_forward_runtime = runtime
                sample_composite_slot.extend_roots(
                    (("forward_runtime", runtime),)
                )
                compact_gather_lifetime = _CompactMaterialGatherLifetime(
                    global_site_rgba_f32=global_site_rgba_f32,
                    source_site_ids_i64=runtime.source_site_ids_i64,
                    _global_identity=id(global_site_rgba_f32),
                    _indices_identity=id(runtime.source_site_ids_i64),
                    _seal=_COMPACT_GATHER_LIFETIME_SEAL,
                )
                compact_gather_lifetime.assert_retained()
                current_compact_gather_lifetime = compact_gather_lifetime
                gathered_material = global_site_rgba_f32.index_select(
                    0,
                    runtime.source_site_ids_i64,
                )
                compact_gather_lifetime.index_select_result_f32 = (
                    gathered_material
                )
                compact_gather_lifetime.phase = "gathered"
                compact_gather_lifetime.assert_retained()
                compact_material = gathered_material.contiguous()
                compact_gather_lifetime.compact_site_rgba_f32 = compact_material
                compact_gather_lifetime.phase = "materialized"
                compact_gather_lifetime.assert_retained()
                current_forward_compact_material = compact_material
                forward_into_lifetime = (
                    prepare_kinetic_native_node_forward_into_lifetime(
                        session,
                        runtime,
                        compact_material,
                    )
                )
                current_forward_into_lifetime = forward_into_lifetime
                node_chart_out = torch.empty(
                    (runtime.row_count, runtime.node_count, 4),
                    dtype=torch.float32,
                    device=global_site_rgba_f32.device,
                )
                forward_into_lifetime.publish_output(node_chart_out)
                current_forward_node_chart_out = node_chart_out
                sample_composite_slot.extend_roots(
                    (
                        ("compact_gather_lifetime", compact_gather_lifetime),
                        ("compact_material", compact_material),
                        ("forward_into_lifetime", forward_into_lifetime),
                        ("forward_node_chart_out", node_chart_out),
                    )
                )
                token = session.launch_node_forward_into(
                    forward_into_lifetime
                )
                current_forward_token = token
                current_forward_grad_node_chart = torch.zeros_like(
                    token.world.node_chart_f32,
                )
                current_forward_loss = torch.zeros(
                    (1,),
                    dtype=torch.float32,
                    device=global_site_rgba_f32.device,
                )
                block_state = _ActiveNativeBlockState(
                    token=token,
                    grad_node_chart_f32=current_forward_grad_node_chart,
                    loss_f32=current_forward_loss,
                    compact_gather_lifetime=compact_gather_lifetime,
                    forward_into_lifetime=forward_into_lifetime,
                )
                active[digest] = block_state
                sample_composite_slot.extend_roots(
                    (
                        ("forward_token", token),
                        ("active_block_state", block_state),
                        (
                            "active_block_grad_node_chart",
                            block_state.grad_node_chart_f32,
                        ),
                        ("active_block_loss", block_state.loss_f32),
                    )
                )
                ordered_word_node_interactions += (
                    runtime.node_count * runtime.word_count
                )
                current_forward_loss = None
                current_forward_grad_node_chart = None
                current_forward_token = None
                current_forward_node_chart_out = None
                current_forward_into_lifetime = None
                current_forward_compact_material = None
                current_compact_gather_lifetime = None
                current_forward_runtime = None
                del gathered_material, compact_material
                del (
                    compact_gather_lifetime,
                    forward_into_lifetime,
                    node_chart_out,
                    token,
                    runtime,
                )
            if block_state is not None and not any(
                root is block_state for root in sample_composite_slot.additional_roots
            ):
                sample_composite_slot.extend_roots(
                    (
                        ("existing_active_block_state", block_state),
                        (
                            "existing_active_block_grad_node_chart",
                            block_state.grad_node_chart_f32,
                        ),
                        ("existing_active_block_loss", block_state.loss_f32),
                    )
                )
            sample_lifetime = session.launch_sample_accumulate(
                block_state.token,
                sample_block,
                sampler=bundle.sampler,
                background_rgb_f32=background_rgb_f32,
                loss_f32=block_state.loss_f32,
                grad_node_chart_f32=block_state.grad_node_chart_f32,
                cone_diagnostic_i32=cone_diagnostic_i32,
                cone_tolerance=cone_tolerance,
            )
            current_sample_lifetime = sample_lifetime
            sample_composite_slot.publish_executor_lifetime(sample_lifetime)
            pending_sample_completion = session.settle_sample_accumulate(
                sample_lifetime,
                sealed_completion_fence=sealed_completion_fence,
                sealed_completion_launch_epoch=sample_launch_epoch,
            )
            if (
                type(pending_sample_completion)
                is not KineticNativePendingSampleLaunchCompletion
            ):
                raise TypeError(
                    "sealed sample settlement did not return its pending "
                    "completion"
                )
            current_pending_sample_completion = pending_sample_completion
            current_completion_receipt = (
                pending_sample_completion.sealed_completion_receipt
            )
            sample_composite_slot.publish_pending_completion(
                pending_sample_completion
            )
            if (
                pending_sample_completion.launch_epoch_sequence
                != expected_sample_fence_sequence
                or pending_sample_completion.launch_generation_digest
                != sample_launch_digest
            ):
                raise ValueError(
                    "sample pending completion changed its prelaunch relation"
                )

            # Complete every root/digest/tensor validation, accounting read,
            # and commit-plan allocation before consuming the sole receipt.
            # The tail has only the sealed executor plan's constant-time
            # authorization guard followed by executor/transfer/slot assigns.
            next_sample_to_node_linear_interactions = (
                sample_to_node_linear_interactions
                + sample_block.linear_weight_interactions
            )
            next_sample_to_node_dense_fallback_interactions = (
                sample_to_node_dense_fallback_interactions
                + sample_block.dense_fallback_interactions
            )
            next_peak_sample_launch_tensor_bytes = max(
                peak_sample_launch_tensor_bytes,
                sample_block.retained_tensor_bytes,
            )
            next_peak_active_node_state_tensor_bytes = max(
                peak_active_node_state_tensor_bytes,
                _active_state_tensor_bytes(active),
            )
            sample_composite_slot.assert_current()
            sample_iterator.assert_active_releasable_after_consumed_receipt(
                sample_block,
                expected_lifetime=current_sample_transfer_lifetime,
            )
            session.assert_pending_sample_accumulate_releasable(
                pending_sample_completion,
                sealed_completion_fence,
                subject=sample_composite_slot,
            )
            current_sample_release_commit_plan = (
                pending_sample_completion.consume_sealed_receipt_for_outer_composite(
                    session,
                    sealed_completion_fence,
                    subject=sample_composite_slot,
                    consumer="lazy-native-sample-composite-release",
                )
            )
            sample_completion = (
                session.commit_sample_accumulate_after_consumed_sealed_receipt(
                    current_sample_release_commit_plan
                )
            )
            sample_iterator._commit_active_release_after_consumed_receipt(
                expected_lifetime=current_sample_transfer_lifetime,
            )
            sample_composite_slot._commit_after_consumed_receipt()
            current_completion_receipt = None
            current_completion_launch_epoch = None
            current_completion_stage = ""
            current_completion_launch_generation_digest = ""
            current_completion_expected_fence_sequence = None
            current_pending_sample_completion = None
            sample_release_commit_plan = current_sample_release_commit_plan
            current_sample_release_commit_plan = None
            current_sample_composite_slot = None
            current_sample_lifetime = None
            current_sample_transfer_lifetime = None
            current_sample_block = None

            sample_completion_fence_call_count += (
                sample_completion.device_completion_fence_call_count
            )
            sample_to_node_linear_interactions = (
                next_sample_to_node_linear_interactions
            )
            sample_to_node_dense_fallback_interactions = (
                next_sample_to_node_dense_fallback_interactions
            )
            peak_sample_launch_tensor_bytes = (
                next_peak_sample_launch_tensor_bytes
            )
            peak_active_node_state_tensor_bytes = (
                next_peak_active_node_state_tensor_bytes
            )
            del (
                pending_sample_completion,
                sample_completion,
                sample_release_commit_plan,
                sample_lifetime,
                sample_block,
                sample_composite_slot,
            )
        close_sample_iterator_after_proven_completion()
        target_read_accounting = sample_iterator.target_read_accounting()
        if (
            int(target_read_accounting["selected_pixel_read_observation_count"])
            != bundle.observation_count
            or int(target_read_accounting["selected_pixel_read_call_count"])
            != len(plan.unique_selected_frames)
        ):
            _fail_arithmetic(
                "selected-pixel read receipts lost exact bundle coverage"
            )

        if (
            full_geometry_context is not None
            and full_geometry_context.reverse_mode == "fused_union_v2"
        ):
            from kinetic_native_equal_rank_runtime_adapter import (
                execute_kinetic_native_equal_rank_fused_union_full_vjp_transaction_v2,
                materialize_kinetic_native_equal_rank_fused_union_full_vjp_transaction_v2,
                prepare_kinetic_native_equal_rank_fused_direct_full_vjp_v1,
                prepare_kinetic_native_equal_rank_fused_union_full_vjp_construction_lifetime_v2,
            )

            ordered_active = tuple(
                (runtime, active[runtime.payload.block.generation_digest])
                for runtime in lane.runtimes
                if runtime.payload.block.generation_digest in active
            )
            if len(ordered_active) != len(lane.runtimes):
                raise ValueError(
                    "fused_union_v2 requires the exact all-block request union; "
                    "a selected observation left a spatial block inactive"
                )

            construction_digest = _digest_parts(
                "lazy-fused-union-v2-construction-v1",
                session.generation_id,
                bundle.spatial_bundle.generation_digest,
                tuple(runtime.generation_id for runtime, _ in ordered_active),
            )
            construction_sequence = sealed_completion_fence.next_fence_sequence
            construction_epoch = _register_general_completion_epoch(
                sealed_completion_fence,
                accelerator_stage_settlement_slot,
                stage="fused-union-v2-construction",
                launch_generation_digest=construction_digest,
                roots=(
                    ("native_lane", lane),
                    ("native_session", session),
                    ("active_blocks", active),
                    ("spatial_bundle", bundle.spatial_bundle),
                ),
            )
            current_completion_launch_epoch = construction_epoch
            current_completion_stage = "fused-union-v2-construction"
            current_completion_launch_generation_digest = construction_digest
            current_completion_expected_fence_sequence = construction_sequence
            construction_receipts: list[PaperKineticCompletionFenceReceipt] = []

            def settle_union_construction() -> None:
                receipt, observed = fence_registered_completion_epoch(
                    construction_epoch,
                    expected_fence_sequence=construction_sequence,
                )
                if observed != construction_sequence:
                    _fail_arithmetic("union-v2 construction sequence changed")
                construction_receipts.append(receipt)

            prepared_blocks = tuple(
                prepare_kinetic_native_equal_rank_fused_direct_full_vjp_v1(
                    block_state.token.world,
                    lowering=bundle.sampler.lowering,
                    sources=bundle.sampler.sources,
                )
                for _, block_state in ordered_active
            )
            union_construction_lifetime = (
                prepare_kinetic_native_equal_rank_fused_union_full_vjp_construction_lifetime_v2(
                    prepared_blocks,
                    tuple(
                        block_state.grad_node_chart_f32
                        for _, block_state in ordered_active
                    ),
                    spatial_bundle=bundle.spatial_bundle,
                    active_block_manifest_generation_id=(
                        bundle.spatial_bundle.generation_digest
                    ),
                    max_transaction_scratch_tensor_bytes=(
                        full_geometry_context.policy.maximum_fused_union_transaction_scratch_tensor_bytes
                    ),
                )
            )
            union_transaction = (
                materialize_kinetic_native_equal_rank_fused_union_full_vjp_transaction_v2(
                    union_construction_lifetime,
                    construction_completion_fence=settle_union_construction,
                    construction_completion_fence_provenance=CAPABILITY_PROVENANCE,
                )
            )
            if not construction_receipts:
                settle_union_construction()
            if len(construction_receipts) != 1:
                _fail_arithmetic("union-v2 construction fenced more than once")
            construction_receipt = construction_receipts[0]
            current_completion_receipt = construction_receipt
            _consume_general_completion_receipt(
                sealed_completion_fence,
                accelerator_stage_settlement_slot,
                construction_receipt,
                consumer="lazy-fused-union-v2-construction-settlement",
            )
            current_completion_receipt = None
            current_completion_launch_epoch = None

            execution_digest = _digest_parts(
                "lazy-fused-union-v2-execution-v1",
                session.generation_id,
                union_transaction.generation_id,
                _tensor_signature(global_grad_site_rgba_f32),
                _tensor_signature(loss_f32),
            )
            execution_sequence = sealed_completion_fence.next_fence_sequence
            execution_epoch = _register_general_completion_epoch(
                sealed_completion_fence,
                accelerator_stage_settlement_slot,
                stage="fused-union-v2-execution",
                launch_generation_digest=execution_digest,
                roots=(
                    ("native_lane", lane),
                    ("native_session", session),
                    ("active_blocks", active),
                    ("union_transaction", union_transaction),
                    ("global_bar", global_grad_site_rgba_f32),
                    ("global_loss", loss_f32),
                ),
            )
            current_completion_launch_epoch = execution_epoch
            current_completion_stage = "fused-union-v2-execution"
            current_completion_launch_generation_digest = execution_digest
            current_completion_expected_fence_sequence = execution_sequence
            execution_receipts: list[PaperKineticCompletionFenceReceipt] = []

            def settle_union_execution() -> None:
                receipt, observed = fence_registered_completion_epoch(
                    execution_epoch,
                    expected_fence_sequence=execution_sequence,
                )
                if observed != execution_sequence:
                    _fail_arithmetic("union-v2 execution sequence changed")
                execution_receipts.append(receipt)

            for _, block_state in ordered_active:
                loss_f32.add_(block_state.loss_f32)
            union_result = (
                execute_kinetic_native_equal_rank_fused_union_full_vjp_transaction_v2(
                    union_transaction,
                    device_completion_fence=settle_union_execution,
                    device_completion_fence_provenance=CAPABILITY_PROVENANCE,
                )
            )
            if len(execution_receipts) != 1:
                _fail_arithmetic("union-v2 execution did not fence exactly once")
            execution_receipt = execution_receipts[0]
            current_completion_receipt = execution_receipt
            session.accept_fused_union_v2_full_geometry_result(union_result)
            _consume_general_completion_receipt(
                sealed_completion_fence,
                accelerator_stage_settlement_slot,
                execution_receipt,
                consumer="lazy-fused-union-v2-execution-settlement",
            )
            current_completion_receipt = None
            current_completion_launch_epoch = None

            d2h_digest = _digest_parts(
                "lazy-fused-union-v2-d2h-and-material-scatter-v1",
                session.generation_id,
                union_result.transaction_generation_id,
                _tensor_signature(global_grad_site_rgba_f32),
            )
            d2h_sequence = sealed_completion_fence.next_fence_sequence
            d2h_epoch = _register_general_completion_epoch(
                sealed_completion_fence,
                accelerator_stage_settlement_slot,
                stage="fused-union-v2-d2h",
                launch_generation_digest=d2h_digest,
                roots=(
                    ("native_lane", lane),
                    ("native_session", session),
                    ("active_blocks", active),
                    ("union_result", union_result),
                    ("global_bar", global_grad_site_rgba_f32),
                    ("full_geometry_context", full_geometry_context),
                ),
            )
            current_completion_launch_epoch = d2h_epoch
            current_completion_stage = "fused-union-v2-d2h"
            current_completion_launch_generation_digest = d2h_digest
            current_completion_expected_fence_sequence = d2h_sequence
            d2h_receipts: list[PaperKineticCompletionFenceReceipt] = []

            def scatter_material_and_settle_d2h(
                compact_material_bars: tuple[torch.Tensor, ...],
            ) -> PaperKineticCompletionFenceReceipt:
                if len(compact_material_bars) != len(ordered_active):
                    raise ValueError("union-v2 compact material-bar count changed")
                for (runtime, _), compact_bar in zip(
                    ordered_active,
                    compact_material_bars,
                    strict=True,
                ):
                    global_grad_site_rgba_f32.index_add_(
                        0,
                        runtime.source_site_ids_i64,
                        compact_bar,
                    )
                receipt, observed = fence_registered_completion_epoch(
                    d2h_epoch,
                    expected_fence_sequence=d2h_sequence,
                )
                if observed != d2h_sequence:
                    _fail_arithmetic("union-v2 D2H sequence changed")
                d2h_receipts.append(receipt)
                return receipt

            full_geometry_context.consume_and_accumulate_fused_union_v2(
                union_result,
                bundle_index=bundle.bundle_index,
                settle_device_outputs=scatter_material_and_settle_d2h,
            )
            if len(d2h_receipts) != 1:
                _fail_arithmetic("union-v2 D2H did not fence exactly once")
            d2h_receipt = d2h_receipts[0]
            current_completion_receipt = d2h_receipt
            telemetry = session.seal()
            forward_lifetimes, gather_lifetimes = (
                unique_forward_and_gather_lifetimes()
            )
            for lifetime in forward_lifetimes:
                lifetime.assert_releasable_for_sealed_completion_receipt(
                    d2h_receipt,
                    sealed_completion_fence,
                    stage="fused-union-v2-d2h",
                    launch_generation_digest=d2h_digest,
                    expected_fence_sequence=d2h_sequence,
                )
            for lifetime in gather_lifetimes:
                lifetime.assert_releasable_for_sealed_completion_receipt(
                    d2h_receipt,
                    sealed_completion_fence,
                    stage="fused-union-v2-d2h",
                    launch_generation_digest=d2h_digest,
                    expected_fence_sequence=d2h_sequence,
                )
            spatial_construction_lifetime = (
                bundle.spatial_bundle._construction_lifetime
            )
            spatial_construction_lifetime.assert_retirable_after_consumed_receipt()
            _consume_general_completion_receipt(
                sealed_completion_fence,
                accelerator_stage_settlement_slot,
                d2h_receipt,
                consumer="lazy-fused-union-v2-d2h-and-lane-release",
            )
            for lifetime in forward_lifetimes:
                lifetime._commit_retire_after_consumed_receipt()
            for lifetime in gather_lifetimes:
                lifetime._commit_release_after_consumed_receipt()
            spatial_construction_lifetime._commit_retire_after_consumed_receipt()
            active.clear()
            current_completion_receipt = None
            current_completion_launch_epoch = None
            current_completion_stage = ""
            current_completion_launch_generation_digest = ""
            current_completion_expected_fence_sequence = None
            outcome = _BundleExecutionResult(
                eligible_native_block_count=lane.native_runtime_count,
                active_native_block_count=telemetry.active_native_block_count,
                native_node_forward_launch_count=(
                    telemetry.native_node_forward_launch_count
                ),
                native_sample_prepare_count=telemetry.native_sample_prepare_count,
                native_sample_launch_count=telemetry.native_sample_launch_count,
                native_sample_completion_fence_count=(
                    telemetry.native_sample_completion_fence_count
                ),
                native_reverse_completion_fence_count=1,
                native_lane_construction_completion_fence_count=1,
                lane_release_completion_fence_count=1,
                native_material_word_vjp_launch_count=0,
                native_full_geometry_vjp_launch_count=0,
                native_fused_union_v2_vjp_launch_count=(
                    telemetry.native_fused_full_geometry_vjp_launch_count
                ),
                native_length_bar_tensor_bytes=0,
                native_union_construction_completion_fence_count=1,
                geometry_d2h_completion_fence_count=1,
                streamed_sample_count=telemetry.streamed_sample_count,
                ordered_word_node_interactions=ordered_word_node_interactions,
                sample_to_node_linear_interactions=(
                    sample_to_node_linear_interactions
                ),
                sample_to_node_dense_fallback_interactions=(
                    sample_to_node_dense_fallback_interactions
                ),
                lane_resident_logical_tensor_bytes=(
                    lane.resident_logical_tensor_bytes
                ),
                peak_active_node_state_tensor_bytes=max(
                    peak_active_node_state_tensor_bytes,
                    union_result.retained_output_tensor_bytes,
                ),
                peak_sample_launch_tensor_bytes=peak_sample_launch_tensor_bytes,
                target_read_accounting=MappingProxyType(
                    dict(target_read_accounting)
                ),
                executor_generation_id=lane.executor.generation_id,
                native_abi_identity=lane.executor.native_abi_identity,
                lane_generation_digest=lane.generation_digest,
                fence_call_count=sample_completion_fence_call_count + 4,
            )
            if direct_completion_fence_call_count != 4:
                _fail_arithmetic("union-v2 direct completion count changed")
            del (
                telemetry,
                union_result,
                union_transaction,
                union_construction_lifetime,
                prepared_blocks,
            )
            return outcome

        for runtime in lane.runtimes:
            digest = runtime.payload.block.generation_digest
            block_state = active.get(digest)
            if block_state is None:
                continue
            current_reverse_runtime = runtime
            current_reverse_block_state = block_state
            reverse_launch_digest = _digest_parts(
                "lazy-material-reverse-launch-v1",
                session.generation_id,
                runtime.generation_id,
                block_state.token.generation_id,
                _tensor_signature(block_state.grad_node_chart_f32),
                _tensor_signature(block_state.loss_f32),
                _tensor_signature(global_grad_site_rgba_f32),
                _tensor_signature(loss_f32),
            )
            reverse_sequence = sealed_completion_fence.next_fence_sequence
            reverse_launch_epoch = _register_general_completion_epoch(
                sealed_completion_fence,
                accelerator_stage_settlement_slot,
                stage="reverse-completion",
                launch_generation_digest=reverse_launch_digest,
                roots=(
                    ("native_lane", lane),
                    ("native_session", session),
                    ("reverse_runtime", runtime),
                    ("reverse_block_state", block_state),
                    ("active_blocks", active),
                    ("global_bar", global_grad_site_rgba_f32),
                    ("global_loss", loss_f32),
                ),
            )
            current_completion_launch_epoch = reverse_launch_epoch
            current_completion_stage = "reverse-completion"
            current_completion_launch_generation_digest = reverse_launch_digest
            current_completion_expected_fence_sequence = reverse_sequence
            compact_bar = torch.empty(
                (runtime.compact_site_count, 4),
                dtype=torch.float32,
                device=global_site_rgba_f32.device,
            )
            current_reverse_compact_bar = compact_bar
            staged_geometry = (
                full_geometry_context is not None
                and full_geometry_context.reverse_mode == "staged_sparse"
            )
            if staged_geometry:
                from kinetic_native_equal_rank_geometry_reduction import (
                    kinetic_native_equal_rank_vjp_provenance_id,
                )
                from kinetic_native_equal_rank_sparse_geometry_reduction import (
                    reduce_kinetic_native_equal_rank_sparse_geometry_vjp,
                )

                vjp_result = session.launch_full_geometry_vjp(
                    block_state.token,
                    block_state.grad_node_chart_f32,
                    compact_grad_site_rgba_f32=compact_bar,
                    global_grad_site_rgba_f32=global_grad_site_rgba_f32,
                )
            else:
                vjp_result = session.launch_material_vjp(
                    block_state.token,
                    block_state.grad_node_chart_f32,
                    compact_grad_site_rgba_f32=compact_bar,
                    global_grad_site_rgba_f32=global_grad_site_rgba_f32,
                )
            current_material_execution = vjp_result
            loss_f32.add_(block_state.loss_f32)
            _extend_general_completion_roots(
                accelerator_stage_settlement_slot,
                (
                    ("reverse_compact_bar", compact_bar),
                    ("reverse_material_execution", vjp_result),
                ),
            )
            peak_active_node_state_tensor_bytes = max(
                peak_active_node_state_tensor_bytes,
                _active_state_tensor_bytes(active)
                + _unique_tensor_storage_bytes((compact_bar,)),
            )
            # One reverse scratch is retained at a time.  Its device-wide
            # completion proof precedes deletion from ``active`` and reuse of
            # the next compact bar, preventing queue-depth growth by block.
            if staged_geometry:
                staged_fence_receipts: list[
                    PaperKineticCompletionFenceReceipt
                ] = []

                def staged_reduction_completion_fence() -> None:
                    receipt, observed_sequence = (
                        fence_registered_completion_epoch(
                            reverse_launch_epoch,
                            expected_fence_sequence=reverse_sequence,
                        )
                    )
                    if observed_sequence != reverse_sequence:
                        _fail_arithmetic(
                            "staged geometry reverse completion sequence changed"
                        )
                    staged_fence_receipts.append(receipt)

                reduction = (
                    reduce_kinetic_native_equal_rank_sparse_geometry_vjp(
                        vjp_result.native_vjp_result,
                        bundle.sampler,
                        expected_native_vjp_provenance_id=(
                            kinetic_native_equal_rank_vjp_provenance_id(
                                vjp_result.native_vjp_result
                            )
                        ),
                        device_completion_fence=(
                            staged_reduction_completion_fence
                        ),
                        device_completion_fence_provenance=(
                            CAPABILITY_PROVENANCE
                        ),
                        maximum_bridge_visible_peak_logical_tensor_bytes=(
                            full_geometry_context.policy.maximum_geometry_bridge_visible_peak_logical_tensor_bytes
                        ),
                        include_ray_gradients=False,
                    )
                )
                if len(staged_fence_receipts) != 1:
                    _fail_arithmetic(
                        "staged geometry reduction did not fence exactly once"
                    )
                reverse_receipt = staged_fence_receipts[0]
                observed_reverse_sequence = reverse_receipt.fence_sequence
                session.consume_full_geometry_vjp_execution(
                    vjp_result,
                    geometry_reduction=reduction,
                    expected_device_completion_fence_provenance=(
                        CAPABILITY_PROVENANCE
                    ),
                )
                full_geometry_context.accumulate_staged_sparse(
                    reduction,
                    bundle_index=bundle.bundle_index,
                    completion_fence_sequence=reverse_receipt.fence_sequence,
                    completion_launch_generation_digest=(
                        reverse_receipt.launch_generation_digest
                    ),
                    completion_receipt_generation_digest=(
                        reverse_receipt.generation_digest
                    ),
                )
            else:
                reverse_receipt, observed_reverse_sequence = (
                    fence_registered_completion_epoch(
                        reverse_launch_epoch,
                        expected_fence_sequence=reverse_sequence,
                    )
                )
            current_completion_receipt = reverse_receipt
            if observed_reverse_sequence != reverse_sequence:
                _fail_arithmetic("reverse completion sequence changed")
            reverse_completion_fence_call_count += 1
            if len(active) > 1:
                block_state.forward_into_lifetime.assert_releasable_for_sealed_completion_receipt(
                    reverse_receipt,
                    sealed_completion_fence,
                    stage=current_completion_stage,
                    launch_generation_digest=reverse_launch_digest,
                    expected_fence_sequence=reverse_sequence,
                )
                block_state.compact_gather_lifetime.assert_releasable_for_sealed_completion_receipt(
                    reverse_receipt,
                    sealed_completion_fence,
                    stage=current_completion_stage,
                    launch_generation_digest=reverse_launch_digest,
                    expected_fence_sequence=reverse_sequence,
                )
                _consume_general_completion_receipt(
                    sealed_completion_fence,
                    accelerator_stage_settlement_slot,
                    reverse_receipt,
                    consumer="lazy-native-reverse-scratch-and-forward-release",
                )
                block_state.forward_into_lifetime._commit_retire_after_consumed_receipt()
                block_state.compact_gather_lifetime._commit_release_after_consumed_receipt()
                del active[digest]
                current_completion_receipt = None
                current_completion_launch_epoch = None
                current_completion_stage = ""
                current_completion_launch_generation_digest = ""
                current_completion_expected_fence_sequence = None
            else:
                final_reverse_receipt = reverse_receipt
                final_reverse_launch_digest = reverse_launch_digest
                final_reverse_sequence = reverse_sequence
            current_material_execution = None
            current_reverse_compact_bar = None
            current_reverse_block_state = None
            current_reverse_runtime = None
            del vjp_result, compact_bar, block_state
        if len(active) != 1:
            _fail_arithmetic("lazy native step lost its final reverse release carrier")

        telemetry = session.seal()
        if (
            telemetry.streamed_sample_count != bundle.observation_count
            or telemetry.native_sample_completion_fence_count
            != sample_completion_fence_call_count
            or telemetry.native_sample_launch_count
            != sample_completion_fence_call_count
            or (
                telemetry.native_material_word_vjp_launch_count
                + telemetry.native_full_geometry_vjp_launch_count
                + telemetry.native_fused_full_geometry_vjp_launch_count
            )
            != reverse_completion_fence_call_count
        ):
            _fail_arithmetic(
                "lazy native step bundle samples/reverses differ from exact observations"
            )
        if final_reverse_receipt is None or final_reverse_sequence is None:
            _fail_arithmetic("lazy lane has no final reverse completion receipt")
        final_block_state = next(iter(active.values()))
        final_block_state.forward_into_lifetime.assert_releasable_for_sealed_completion_receipt(
            final_reverse_receipt,
            sealed_completion_fence,
            stage="reverse-completion",
            launch_generation_digest=final_reverse_launch_digest,
            expected_fence_sequence=final_reverse_sequence,
        )
        final_block_state.compact_gather_lifetime.assert_releasable_for_sealed_completion_receipt(
            final_reverse_receipt,
            sealed_completion_fence,
            stage="reverse-completion",
            launch_generation_digest=final_reverse_launch_digest,
            expected_fence_sequence=final_reverse_sequence,
        )
        spatial_construction_lifetime = (
            bundle.spatial_bundle._construction_lifetime
        )
        spatial_construction_lifetime.assert_retirable_after_consumed_receipt()
        _consume_general_completion_receipt(
            sealed_completion_fence,
            accelerator_stage_settlement_slot,
            final_reverse_receipt,
            consumer="lazy-native-final-reverse-and-lane-release",
        )
        final_block_state.forward_into_lifetime._commit_retire_after_consumed_receipt()
        final_block_state.compact_gather_lifetime._commit_release_after_consumed_receipt()
        spatial_construction_lifetime._commit_retire_after_consumed_receipt()
        active.clear()
        current_completion_receipt = None
        current_completion_stage = ""
        current_completion_launch_generation_digest = ""
        current_completion_expected_fence_sequence = None
        outcome = _BundleExecutionResult(
            eligible_native_block_count=lane.native_runtime_count,
            active_native_block_count=telemetry.active_native_block_count,
            native_node_forward_launch_count=(
                telemetry.native_node_forward_launch_count
            ),
            native_sample_prepare_count=telemetry.native_sample_prepare_count,
            native_sample_launch_count=telemetry.native_sample_launch_count,
            native_sample_completion_fence_count=(
                telemetry.native_sample_completion_fence_count
            ),
            native_reverse_completion_fence_count=(
                reverse_completion_fence_call_count
            ),
            native_lane_construction_completion_fence_count=1,
            lane_release_completion_fence_count=1,
            native_material_word_vjp_launch_count=(
                telemetry.native_material_word_vjp_launch_count
            ),
            native_full_geometry_vjp_launch_count=(
                telemetry.native_full_geometry_vjp_launch_count
            ),
            native_fused_union_v2_vjp_launch_count=0,
            native_length_bar_tensor_bytes=(
                telemetry.native_length_bar_tensor_bytes
            ),
            native_union_construction_completion_fence_count=0,
            geometry_d2h_completion_fence_count=0,
            streamed_sample_count=telemetry.streamed_sample_count,
            ordered_word_node_interactions=ordered_word_node_interactions,
            sample_to_node_linear_interactions=(
                sample_to_node_linear_interactions
            ),
            sample_to_node_dense_fallback_interactions=(
                sample_to_node_dense_fallback_interactions
            ),
            lane_resident_logical_tensor_bytes=(
                lane.resident_logical_tensor_bytes
            ),
            peak_active_node_state_tensor_bytes=(
                peak_active_node_state_tensor_bytes
            ),
            peak_sample_launch_tensor_bytes=peak_sample_launch_tensor_bytes,
            target_read_accounting=MappingProxyType(
                dict(target_read_accounting)
            ),
            executor_generation_id=lane.executor.generation_id,
            native_abi_identity=lane.executor.native_abi_identity,
            lane_generation_digest=lane.generation_digest,
            fence_call_count=(
                sample_completion_fence_call_count
                + reverse_completion_fence_call_count
                + 1
            ),
        )
        del telemetry
        if direct_completion_fence_call_count != (
            reverse_completion_fence_call_count + 1
        ):
            _fail_arithmetic(
                "lazy native bundle direct completion-fence count changed"
            )
    except BaseException as error:
        original_traceback = error.__traceback__
        if (
            current_pending_sample_completion is not None
            and current_sample_composite_slot is not None
            and current_sample_composite_slot.phase == "pending"
        ):
            # Completion is known, but the sole subject-bound receipt has not
            # been consumed and outer precommit validation rejected the
            # composite.  Retain both owners intact: bypassing the rejected
            # validation through a different abort path would defeat the
            # exact one-shot ownership contract.
            _quarantine_lazy_async_failure(
                state,
                stage="sample-composite-settlement",
                original_error=error,
                original_traceback=original_traceback,
                failed_completion_fence_error=error,
                retained_references=lifetime_references(),
                completion_fence_generation_digest=(
                    sealed_completion_fence.generation_digest
                ),
            )
            error.add_note(
                "lazy subject-bound sample failed after known completion but "
                "before its exact composite commit; roots require restart"
            )
            raise
        if (
            accelerator_stage_settlement_slot is not None
            and accelerator_stage_settlement_slot.phase != "bound"
        ):
            _quarantine_lazy_async_failure(
                state,
                stage=(
                    accelerator_stage_settlement_slot.stage
                    or "accelerator-general-stage"
                ),
                original_error=error,
                original_traceback=original_traceback,
                failed_completion_fence_error=error,
                retained_references=lifetime_references(),
                completion_fence_generation_digest=(
                    sealed_completion_fence.generation_digest
                ),
            )
            error.add_note(
                "lazy accelerator lane stage failed before its exact "
                "subject-bound commit; roots require restart"
            )
            raise
        sample_completion_unknown = bool(
            session is not None and session._sample_completion_unknown
        )
        if (
            sealed_completion_fence.poisoned
            or completion_unknown_stage is not None
            or sample_completion_unknown
        ):
            stage = (
                completion_unknown_stage
                if completion_unknown_stage is not None
                else "sample-completion"
            )
            _quarantine_lazy_async_failure(
                state,
                stage=stage,
                original_error=error,
                original_traceback=original_traceback,
                failed_completion_fence_error=error,
                retained_references=lifetime_references(),
                completion_fence_generation_digest=(
                    sealed_completion_fence.generation_digest
                ),
            )
            error.add_note(
                f"lazy {stage} fence failed; lane roots are quarantined "
                "and the process must restart"
            )
            raise
        cleanup_receipt = current_completion_receipt
        cleanup_stage = current_completion_stage
        cleanup_launch_digest = current_completion_launch_generation_digest
        cleanup_sequence = current_completion_expected_fence_sequence
        session_abort_commit_plan = None
        try:
            if cleanup_receipt is not None and cleanup_receipt.consumed:
                cleanup_receipt = None
                cleanup_sequence = None
            if cleanup_receipt is None and (
                current_completion_launch_epoch is not None
                and sealed_completion_fence.registered_launch_epoch
                is current_completion_launch_epoch
            ):
                cleanup_stage = current_completion_stage
                cleanup_launch_digest = (
                    current_completion_launch_generation_digest
                )
                cleanup_sequence = (
                    current_completion_expected_fence_sequence
                )
                if cleanup_sequence is None:
                    _fail_arithmetic("registered cleanup epoch lost its sequence")
                cleanup_receipt, observed_cleanup_sequence = (
                    fence_registered_completion_epoch(
                        current_completion_launch_epoch,
                        expected_fence_sequence=cleanup_sequence,
                    )
                )
                if observed_cleanup_sequence != cleanup_sequence:
                    _fail_arithmetic("registered cleanup sequence changed")
            if cleanup_receipt is None and session is not None and not session._sealed:
                prior_sample_receipt = (
                    session._failed_sample_completion_sealed_receipt
                )
                if prior_sample_receipt is not None:
                    cleanup_receipt = prior_sample_receipt
                    cleanup_stage = "sample-completion"
                    cleanup_launch_digest = (
                        session._failed_sample_completion_launch_generation_digest
                    )
                    if not _is_sha256(cleanup_launch_digest):
                        _fail_arithmetic(
                            "failed sample settlement lost its launch generation"
                        )
                    cleanup_sequence = prior_sample_receipt.fence_sequence

            spatial_lifetime = bundle.spatial_bundle._construction_lifetime
            sample_composite_slot = current_sample_composite_slot
            if (
                sample_composite_slot is not None
                and sample_composite_slot.phase != "committed"
            ):
                sample_composite_slot.assert_current()
            forward_lifetimes, gather_lifetimes = (
                unique_forward_and_gather_lifetimes()
            )
            active_transfer = (
                None
                if sample_iterator is None
                else sample_iterator.active_transfer_lifetime
            )
            spatial_lifetime.assert_retirable_after_consumed_receipt()
            if active_transfer is not None:
                sample_iterator.assert_active_releasable_after_consumed_receipt(
                    current_sample_block
                )

            if cleanup_receipt is not None:
                if cleanup_sequence is None:
                    _fail_arithmetic("outstanding completion receipt lost its sequence")
                cleanup_receipt.assert_for(
                    sealed_completion_fence,
                    stage=cleanup_stage,
                    launch_generation_digest=cleanup_launch_digest,
                    fence_sequence=cleanup_sequence,
                )
                if session is not None and not session._sealed:
                    session_abort_commit_plan = (
                        session.assert_abort_releasable_for_sealed_completion_receipt(
                            cleanup_receipt,
                            sealed_completion_fence,
                            stage=cleanup_stage,
                            launch_generation_digest=cleanup_launch_digest,
                            expected_fence_sequence=cleanup_sequence,
                        )
                    )
                for lifetime in forward_lifetimes:
                    if lifetime.phase != "released":
                        lifetime.assert_releasable_for_sealed_completion_receipt(
                            cleanup_receipt,
                            sealed_completion_fence,
                            stage=cleanup_stage,
                            launch_generation_digest=cleanup_launch_digest,
                            expected_fence_sequence=cleanup_sequence,
                        )
                for lifetime in gather_lifetimes:
                    if lifetime.phase != "released":
                        lifetime.assert_releasable_for_sealed_completion_receipt(
                            cleanup_receipt,
                            sealed_completion_fence,
                            stage=cleanup_stage,
                            launch_generation_digest=cleanup_launch_digest,
                            expected_fence_sequence=cleanup_sequence,
                        )
                if cleanup_receipt.subject_binding is not None:
                    if (
                        sample_composite_slot is None
                        or sample_composite_slot.subject_binding
                        is not cleanup_receipt.subject_binding
                    ):
                        _fail_arithmetic(
                            "failed sample cleanup lost its exact bound subject"
                        )
                    cleanup_receipt.consume_for_subject(
                        sealed_completion_fence,
                        sample_composite_slot.subject_binding,
                        subject=sample_composite_slot,
                        consumer="lazy-native-failed-sample-composite-release",
                    )
                else:
                    cleanup_receipt.consume_for(
                        sealed_completion_fence,
                        stage=cleanup_stage,
                        launch_generation_digest=cleanup_launch_digest,
                        fence_sequence=cleanup_sequence,
                        consumer="lazy-native-failed-lane-composite-release",
                    )
            else:
                sealed_completion_fence.assert_current()
                if (
                    sealed_completion_fence.registered_launch_epoch is not None
                    or sealed_completion_fence.outstanding_receipt_sequence is not None
                ):
                    _fail_arithmetic(
                        "failed lane lost its outstanding completion authority"
                    )
                for lifetime in forward_lifetimes:
                    if lifetime.phase != "released":
                        lifetime.assert_retained(session)
                for lifetime in gather_lifetimes:
                    if lifetime.phase != "released":
                        lifetime.assert_retained()

            if session is not None and not session._sealed:
                if cleanup_receipt is None:
                    session.abort_after_all_sealed_completion_receipts(
                        sealed_completion_fence,
                        expected_last_consumed_sequence=(
                            sealed_completion_fence.consumed_fence_count
                        ),
                    )
                else:
                    session._commit_abort_release_after_consumed_receipt(
                        session_abort_commit_plan
                    )
            spatial_lifetime._commit_retire_after_consumed_receipt()
            if active_transfer is not None:
                sample_iterator._commit_active_release_after_consumed_receipt()
            for lifetime in forward_lifetimes:
                if lifetime.phase != "released":
                    lifetime._commit_retire_after_consumed_receipt()
            for lifetime in gather_lifetimes:
                if lifetime.phase != "released":
                    lifetime._commit_release_after_consumed_receipt()
            if (
                sample_composite_slot is not None
                and sample_composite_slot.phase != "committed"
            ):
                sample_composite_slot._commit_after_consumed_receipt()
            active.clear()
            close_sample_iterator_after_proven_completion()
        except BaseException as fence_error:
            _quarantine_lazy_async_failure(
                state,
                stage=(completion_unknown_stage or "cleanup-completion"),
                original_error=error,
                original_traceback=original_traceback,
                failed_completion_fence_error=fence_error,
                retained_references=lifetime_references(),
                completion_fence_generation_digest=(
                    sealed_completion_fence.generation_digest
                ),
            )
            error.add_note(
                "lazy cleanup completion fence failed; lane roots are "
                "quarantined and the process must restart"
            )
            raise error.with_traceback(original_traceback) from fence_error
        raise
    else:
        close_sample_iterator_after_proven_completion()
        active.clear()
        return outcome


def _active_node_and_vjp_upper_bound_bytes(
    bundle: PaperKineticLazyProgramBundle,
) -> int:
    """Conservative dynamic material/node/bar bytes if every block is active."""

    blocks = tuple(block for bucket in bundle.sampler.lowering.buckets for block in bucket.blocks)
    if not blocks:
        raise ValueError("lazy native bundle has no equal-rank block")
    active_bytes = sum(
        16 * len(block.source_site_ids)
        + 32 * block.row_count * block.node_count
        + 4  # one executor-bound block-local loss scalar
        for block in blocks
    )
    one_vjp_bar_bytes = 16 * max(len(block.source_site_ids) for block in blocks)
    return active_bytes + one_vjp_bar_bytes


def _enforce_bundle_memory_policy(
    memory_policy: PaperKineticLazyNativeMemoryPolicy,
    *,
    global_site_rgba_f32: torch.Tensor,
    global_grad_site_rgba_f32: torch.Tensor,
    background_rgb_f32: torch.Tensor,
    loss_f32: torch.Tensor,
    cone_diagnostic_i32: torch.Tensor,
    lane_preflight_bytes: int,
    active_node_and_vjp_upper_bound_bytes: int,
    decoded_frame_scratch_upper_bound_bytes: int,
    selected_frame_target_upper_bound_bytes: int,
    sample_launch_upper_bound_bytes: int,
) -> int:
    """Fail before lane/decode allocation and return the conservative live bound."""

    component_limits = (
        (
            "native lane",
            lane_preflight_bytes,
            memory_policy.max_lane_resident_logical_tensor_bytes,
        ),
        (
            "active node/material VJP state",
            active_node_and_vjp_upper_bound_bytes,
            memory_policy.max_active_node_and_vjp_tensor_bytes,
        ),
        (
            "target decode transient",
            decoded_frame_scratch_upper_bound_bytes,
            memory_policy.max_decoded_frame_scratch_tensor_bytes,
        ),
        (
            "selected frame target",
            selected_frame_target_upper_bound_bytes,
            memory_policy.max_selected_frame_target_tensor_bytes,
        ),
        (
            "sample launch",
            sample_launch_upper_bound_bytes,
            memory_policy.max_sample_launch_tensor_bytes,
        ),
    )
    for name, actual, limit in component_limits:
        if actual > limit:
            raise MemoryError(
                f"lazy native {name} preflight exceeds its explicit budget: estimated={actual}, budget={limit}"
            )
    fixed_bytes = _unique_tensor_storage_bytes(
        (
            global_site_rgba_f32,
            global_grad_site_rgba_f32,
            background_rgb_f32,
            loss_f32,
            cone_diagnostic_i32,
        )
    )
    target_cache_bytes = (
        memory_policy.max_step_target_frame_cache_tensor_bytes
        if memory_policy.target_frame_access_mode == TARGET_FRAME_STEP_CACHE
        else 0
    )
    live_upper_bound = (
        fixed_bytes
        + lane_preflight_bytes
        + active_node_and_vjp_upper_bound_bytes
        + target_cache_bytes
        + decoded_frame_scratch_upper_bound_bytes
        + selected_frame_target_upper_bound_bytes
        + sample_launch_upper_bound_bytes
    )
    if live_upper_bound > memory_policy.max_coordinator_visible_live_tensor_bytes:
        raise MemoryError(
            "lazy native coordinator-visible tensor preflight exceeds its "
            "explicit aggregate budget: "
            f"estimated={live_upper_bound}, "
            f"budget={memory_policy.max_coordinator_visible_live_tensor_bytes}"
        )
    return live_upper_bound


def _assert_immutable_step_inputs(
    expected_signatures: tuple[tuple[object, ...], ...],
    global_site_rgba_f32: torch.Tensor,
    background_rgb_f32: torch.Tensor,
) -> None:
    if expected_signatures != tuple(_tensor_signature(tensor) for tensor in (global_site_rgba_f32, background_rgb_f32)):
        raise ValueError("global material or background changed during native step execution")


def _validate_step_inputs(
    state: PaperKineticLazyNativeTrainerState,
    provider: PaperKineticLazyProgramBundleProvider,
    *,
    step_index: int,
    expected_observation_count: int,
    expected_observation_manifest_digest: str,
    loss_normalization_id: str,
    material_generation_id: str,
    background_generation_id: str,
    global_site_rgba_f32: torch.Tensor,
    global_grad_site_rgba_f32: torch.Tensor,
    background_rgb_f32: torch.Tensor,
    maximum_samples_per_launch: int,
    memory_policy: PaperKineticLazyNativeMemoryPolicy,
    optimizer_update: Callable[[PaperKineticLazyNativeMaterialStepResult], None],
    cone_tolerance: float,
) -> None:
    state.assert_current(provider)
    if state.poisoned:
        raise ValueError("lazy native trainer state is poisoned")
    if state.device.type not in {"cpu", "mps"}:
        raise RuntimeError(
            "lazy native CUDA execution is fail-closed until a canonical "
            "launch-domain attestation and completion-fence binding exist"
        )
    if isinstance(step_index, bool) or not isinstance(step_index, int):
        raise TypeError("step_index must be an integer")
    if step_index != state.next_step_index:
        raise ValueError("lazy native trainer step index is stale, skipped, or reused")
    _require_positive_int(
        expected_observation_count,
        name="expected_observation_count",
    )
    _require_positive_int(
        maximum_samples_per_launch,
        name="maximum_samples_per_launch",
    )
    _require_sha256(
        expected_observation_manifest_digest,
        name="expected_observation_manifest_digest",
    )
    if not isinstance(loss_normalization_id, str) or not loss_normalization_id.strip():
        raise ValueError("loss_normalization_id must be nonempty")
    for name, value in (
        ("material_generation_id", material_generation_id),
        ("background_generation_id", background_generation_id),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be nonempty")
    if not isinstance(memory_policy, PaperKineticLazyNativeMemoryPolicy):
        raise TypeError("memory_policy must be PaperKineticLazyNativeMemoryPolicy")
    memory_policy.assert_valid()
    frame_tensor_bytes = provider.height * provider.width * 3 * 4
    if memory_policy.target_frame_access_mode == TARGET_FRAME_STEP_CACHE:
        if memory_policy.max_step_target_frame_cache_tensor_bytes < frame_tensor_bytes:
            raise MemoryError("step target-frame cache cannot admit one frame before decode")
        required_decode_transient_bytes = 2 * frame_tensor_bytes
    else:
        # Direct selected-pixel sources have source-specific temporary state
        # (advanced-index vectors, bounded regions, or mapped pages).  They
        # receive this exact cap and must reject before allocation, then seal
        # the observed source-visible peak.  Requiring a full frame here would
        # silently reintroduce the memory floor this path removes.
        required_decode_transient_bytes = 1
    if required_decode_transient_bytes > memory_policy.max_decoded_frame_scratch_tensor_bytes:
        raise MemoryError("target decode transient exceeds its explicit memory-policy budget")
    if not callable(optimizer_update):
        raise TypeError("optimizer_update must be callable")
    if not math.isfinite(float(cone_tolerance)) or float(cone_tolerance) < 0.0:
        raise ValueError("cone_tolerance must be finite and nonnegative")
    _require_f32_tensor(
        global_site_rgba_f32,
        name="global_site_rgba_f32",
        device=state.device,
        shape=(state.global_site_count, 4),
    )
    _require_f32_tensor(
        global_grad_site_rgba_f32,
        name="global_grad_site_rgba_f32",
        device=state.device,
        shape=(state.global_site_count, 4),
    )
    _require_f32_tensor(
        background_rgb_f32,
        name="background_rgb_f32",
        device=state.device,
        shape=(3,),
    )
    storages = {
        tensor.untyped_storage().data_ptr()
        for tensor in (
            global_site_rgba_f32,
            global_grad_site_rgba_f32,
            background_rgb_f32,
        )
    }
    if len(storages) != 3:
        raise ValueError("global material, bar, and background must not alias")
    global_material_and_bar_bytes = _unique_tensor_storage_bytes((global_site_rgba_f32, global_grad_site_rgba_f32))
    if global_material_and_bar_bytes > memory_policy.max_global_material_and_bar_tensor_bytes:
        raise ValueError("global material/bar tensors exceed their explicit memory-policy budget")
    preallocation_fixed_bytes = global_material_and_bar_bytes + (
        _unique_tensor_storage_bytes((background_rgb_f32,)) + 4 + 3 * 4
    )
    target_cache_bytes = (
        memory_policy.max_step_target_frame_cache_tensor_bytes
        if memory_policy.target_frame_access_mode == TARGET_FRAME_STEP_CACHE
        else 0
    )
    if (
        preallocation_fixed_bytes + target_cache_bytes + required_decode_transient_bytes
        > memory_policy.max_coordinator_visible_live_tensor_bytes
    ):
        raise MemoryError("fixed step/target tensors exceed the coordinator-visible budget")
    # This is the sole O(VF) content certification for a valid logical step.
    # Every provider/bundle/cache check after this boundary is deliberately
    # warm so frame-linear metadata hashing cannot multiply by bundle count.
    provider.assert_current()


def _assert_nonretaining_factory(
    provider: PaperKineticLazyProgramBundleProvider,
) -> None:
    report_fn = getattr(provider.program_factory, "memory_light_residency", None)
    if not callable(report_fn):
        raise TypeError("memory-light execution requires program_factory.memory_light_residency()")
    report = report_fn()
    if not isinstance(report, Mapping):
        raise TypeError("program factory memory-light residency must be a mapping")
    expected_zero = (
        "retained_compile_request_count",
        "retained_compiled_program_count",
        "retained_observation_record_count",
        "retained_tensor_bytes",
    )
    if any(report.get(key) != 0 for key in expected_zero) or report.get("unbounded_cache_enabled") is not False:
        raise ValueError("program factory retains bundle/sample state or an unbounded cache")


def _active_state_tensor_bytes(
    active: Mapping[str, _ActiveNativeBlockState],
) -> int:
    tensors = tuple(
        tensor
        for block_state in active.values()
        for tensor in (
            block_state.token.world.compact_site_rgba_f32,
            block_state.compact_gather_lifetime.index_select_result_f32,
            block_state.token.world.node_chart_f32,
            block_state.grad_node_chart_f32,
            block_state.loss_f32,
        )
        if isinstance(tensor, torch.Tensor)
    )
    return _unique_tensor_storage_bytes(tensors)


def _require_f32_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    device: torch.device,
    shape: tuple[int, ...],
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
        raise ValueError(f"{name} has an invalid device/dtype/layout/shape")


def _unique_tensor_storage_bytes(tensors: Iterable[torch.Tensor]) -> int:
    storages: dict[tuple[str, int, int], int] = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        storage_bytes = int(storage.nbytes())
        storages.setdefault(
            (str(tensor.device), int(storage.data_ptr()), storage_bytes),
            storage_bytes,
        )
    return sum(storages.values())


def _tensor_signature(tensor: torch.Tensor) -> tuple[object, ...]:
    return (
        id(tensor),
        int(getattr(tensor, "_version", 0)),
        tensor.untyped_storage().data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
        tensor.layout,
        tensor.requires_grad,
    )


def _quarantine_lazy_async_failure(
    state: PaperKineticLazyNativeTrainerState,
    *,
    stage: str,
    original_error: BaseException,
    original_traceback: Any,
    failed_completion_fence_error: BaseException,
    retained_references: tuple[tuple[str, Any], ...],
    completion_fence_generation_digest: str,
) -> None:
    """Install the sole bounded quarantine without enqueueing cleanup work."""

    if state._async_failure_quarantine is not None:
        state._async_failure_quarantine.assert_current()
        return
    retained = tuple(
        (role, reference)
        for role, reference in retained_references
        if reference is not None
    )
    provisional = _LazyAsyncFailureQuarantine(
        stage=stage,
        original_error=original_error,
        original_traceback=original_traceback,
        failed_completion_fence_error=failed_completion_fence_error,
        retained_reference_roles=tuple(role for role, _ in retained),
        retained_references=tuple(reference for _, reference in retained),
        completion_fence_generation_digest=(
            completion_fence_generation_digest
        ),
        generation_digest="",
    )
    quarantine = replace(
        provisional,
        generation_digest=_async_failure_quarantine_digest(provisional),
    )
    state._async_failure_quarantine = quarantine
    state.poisoned = True
    quarantine.assert_current()


def _sample_composite_slot_digest(
    slot: _SampleCompositeSettlementSlot,
) -> str:
    return _digest_parts(
        STEP_PROVENANCE,
        SAMPLE_COMPOSITE_SUBJECT_KIND,
        slot.step_generation_id,
        slot.bundle_generation_digest,
        slot.plan_generation_digest,
        slot.session_generation_id,
        slot.session_identity,
        slot.plan_identity,
        slot.stream_identity,
        slot.launch_ordinal,
        slot.covered_sample_count_before_launch,
        slot.slot_identity,
    )


def _accelerator_stage_slot_digest(
    slot: _AcceleratorStageSettlementSlot,
) -> str:
    return _digest_parts(
        STEP_PROVENANCE,
        ACCELERATOR_STAGE_SUBJECT_KIND,
        slot.owner_generation_digest,
        id(slot.capability),
        slot.capability.generation_digest,
        slot.slot_identity,
    )


def _settlement_root_signature(root: Any) -> tuple[object, ...]:
    if isinstance(root, torch.Tensor):
        return ("tensor", *_tensor_signature(root))
    generation = getattr(root, "generation_digest", None)
    if generation is None:
        generation = getattr(root, "generation_id", None)
    return (
        "object",
        type(root).__module__,
        type(root).__qualname__,
        id(root),
        generation if isinstance(generation, str) else None,
    )


def _settlement_root_signature_for_role(
    role: str,
    root: Any,
) -> tuple[object, ...]:
    """Bind declared outputs by exact storage while inputs remain immutable.

    A tensor's ``_version`` is mutation evidence, not identity evidence.  It is
    therefore retained for every ordinary root and omitted only for the small
    role allow-list whose whole purpose is to be written by the fenced epoch.
    """

    signature = _settlement_root_signature(root)
    if (
        isinstance(root, torch.Tensor)
        and role in _WRITABLE_SETTLEMENT_TENSOR_ROLES
    ):
        # _tensor_signature is (id, version, storage, offset, shape, stride,
        # dtype, device, layout).  Preserve every identity/layout field and
        # omit exactly the mutation counter.
        return (
            signature[0],
            signature[1],
            *signature[3:],
            "declared-writable-output",
        )
    return signature


def _async_failure_quarantine_digest(
    quarantine: _LazyAsyncFailureQuarantine,
) -> str:
    return _digest_parts(
        STEP_PROVENANCE,
        "async-failure-quarantine-v1",
        quarantine.stage,
        type(quarantine.original_error).__qualname__,
        str(quarantine.original_error),
        id(quarantine.original_traceback),
        type(quarantine.failed_completion_fence_error).__qualname__,
        str(quarantine.failed_completion_fence_error),
        quarantine.retained_reference_roles,
        tuple(id(reference) for reference in quarantine.retained_references),
        quarantine.completion_fence_generation_digest,
        quarantine.restart_required,
    )


def _result_digest(result: PaperKineticLazyNativeMaterialStepResult) -> str:
    return _digest_parts(
        STEP_PROVENANCE,
        result.step_index,
        result.step_generation_id,
        result.provider_generation_digest,
        result.world_generation_digest,
        result.sites_content_digest,
        result.loss_normalization_id,
        result.material_generation_id,
        result.background_generation_id,
        result._material_tensor_identity,
        result._material_tensor_signature,
        result._background_tensor_identity,
        result._background_tensor_signature,
        result._sealed_completion_fence_generation_digest,
        tuple(sorted(result.accounting.items())),
        result.runtime_status,
    )


def _fence_registered_completion_epoch(
    capability: PaperKineticSealedCompletionFence,
    launch_epoch: PaperKineticCompletionLaunchEpoch,
    *,
    expected_fence_sequence: int,
) -> PaperKineticCompletionFenceReceipt:
    """Fence one exact epoch that was registered before its native work."""

    if capability.registered_launch_epoch is not launch_epoch:
        raise ValueError("completion launch epoch is not capability-current")
    receipt = capability.fence(launch_epoch)
    return receipt


def _lazy_native_abi_identity(native_ops: Any) -> tuple[tuple[str, int], ...]:
    identity = []
    for name in LAZY_NATIVE_REQUIRED_OP_NAMES:
        operation = getattr(native_ops, name, None)
        if not callable(operation):
            raise TypeError(f"native ops object is missing callable {name}")
        implementation = getattr(operation, "__func__", operation)
        identity.append((name, id(implementation)))
    return tuple(identity)


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


def _require_positive_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _require_sha256(value: str, *, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest") from error


def _fail_value(message: str) -> NoReturn:
    raise ValueError(message)


def _fail_arithmetic(message: str) -> NoReturn:
    raise ArithmeticError(message)


def _fail_type(message: str) -> NoReturn:
    raise TypeError(message)


__all__ = [
    "STEP_PROVENANCE",
    "STEP_STATUS",
    "TARGET_FRAME_ACCESS_MODES",
    "TARGET_FRAME_STEP_CACHE",
    "TARGET_FRAME_STREAM_ONCE",
    "PaperKineticLazyNativeMaterialStepResult",
    "PaperKineticLazyNativeMemoryPolicy",
    "PaperKineticLazyNativeTrainerState",
    "paper_kinetic_observation_manifest_digest",
    "prepare_paper_kinetic_lazy_native_trainer_state",
    "run_paper_kinetic_lazy_native_material_step",
]
