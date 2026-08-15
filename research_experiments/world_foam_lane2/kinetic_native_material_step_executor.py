"""Production-facing native executor for one memory-light kinetic step.

The mathematical compiler and equal-rank runtime adapter already seal the
ordered word, node lengths, and compact material mapping.  This module owns the
remaining launch lifecycle.  It binds one native-ops object (the extension
module in production) and permits only this sequence for each active block::

    node forward once
      -> one or more bounded row-ragged sample reductions
      -> exactly one of:
           material-only ordered-word VJP
           staged material-plus-physical-length VJP
           fused all-block material-plus-world-geometry VJP

The executor is intended to be constructed for one lazily resident spatial
lane.  It retains the lane's already-resident runtime blocks, but neither
copies their tensors nor retains the larger sampler object used for cold
binding.  It does not retain a frame axis, targets, sample weights, or
predictions after a sample launch.  A step session retains only the native node
chart, its accumulated cotangent, one block-local loss scalar, and
O(active-lane-blocks)
provenance/counters. Exactly one bounded sample launch may additionally remain
live between enqueue and its executor-owned completion fence; that lifetime
aliases the already-accounted sample/prepared tensors and is released
immediately after settlement, with no token history. Its own world reference is released immediately after a
successful material-only VJP. A staged full-geometry world remains
executor-owned until a sealed completion operation verifies the canonical
fence-backed ``[J,W]`` reduction; global accumulation remains the higher
request layer's responsibility. The distinct fixed-camera fused mode consumes
the session's exact active-block manifest in one
validation/accumulation/finalization transaction. It returns compact material
and global world-geometry bars without allocating ``[J,W]``. Those accepted
bars are still inputs to a separate out-of-place optimizer commit; this
executor does not claim optimizer fail-atomicity. An error poisons the session
without releasing potentially asynchronous native references. The caller must
explicitly abort with a successful completion fence before those references
are released.
The reverse mode is session-wide and fail-closed. Material-only remains the
default caller path and cannot allocate the optional ``[J,W]`` length
cotangent. Staged or fused full geometry must be selected explicitly and every
active block in that session must use exactly one mode; mixing modes is
rejected.

This is a source/CPU-contract integration seam.  Passing the real
``torch_world_foam_lane2_fused_slab`` ops module binds the exact production
Python ABI, but native build, device execution, parity, and allocator telemetry
remain separate gates.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any, NoReturn

import torch
from kinetic_native_equal_rank_geometry_reduction import (
    KineticNativeEqualRankGeometryReduction,
    kinetic_native_equal_rank_vjp_provenance_id,
)
from kinetic_native_equal_rank_sparse_geometry_reduction import (
    KineticNativeEqualRankSparseGeometryReduction,
)
from kinetic_native_equal_rank_runtime_adapter import (
    FORWARD_INTO_OP_NAME,
    FORWARD_OP_NAME,
    MATERIAL_VJP_OP_NAME,
    VJP_OP_NAME,
    KineticNativeEqualRankFusedDirectFullVjpV1,
    KineticNativeEqualRankFusedDirectFullVjpV1Transaction,
    KineticNativeEqualRankFusedDirectFullVjpV1TransactionResult,
    KineticNativeEqualRankFusedUnionFullVjpV2TransactionResult,
    KineticNativeEqualRankMaterialVJPResult,
    KineticNativeEqualRankRuntimeBlock,
    KineticNativeEqualRankVJPResult,
    KineticNativeEqualRankWorld,
    execute_kinetic_native_equal_rank_fused_direct_full_vjp_transaction_v1,
    execute_kinetic_native_equal_rank_node_vjp,
    execute_kinetic_native_equal_rank_material_node_vjp,
    prepare_kinetic_native_equal_rank_fused_direct_full_vjp_transaction_v1,
    refresh_kinetic_native_equal_rank_world,
    refresh_kinetic_native_equal_rank_world_into,
)
from kinetic_sealed_completion_fence import (
    CAPABILITY_PROVENANCE,
    PaperKineticCompletionFenceReceipt,
    PaperKineticCompletionLaunchEpoch,
    PaperKineticCompletionSubjectBinding,
    PaperKineticSealedCompletionFence,
)
from paper_kinetic_ragged_sample_plan import (
    PaperKineticRowRaggedSampleBlock,
    PaperKineticRowRaggedSampler,
)

EXECUTOR_PROVENANCE = "kinetic-native-step-executor-v3"
EXECUTOR_STATUS = "native_ops_bound/source_runtime_unverified"
SAMPLE_PREPARE_OP_NAME = "prepare_kinetic_ragged_p0_lie_sample_block"
SAMPLE_LAUNCH_OP_NAME = (
    "kinetic_ragged_p0_lie_sample_accumulate_loss_only_launch_only"
)

_REQUIRED_NATIVE_OP_NAMES = (
    FORWARD_OP_NAME,
    VJP_OP_NAME,
    MATERIAL_VJP_OP_NAME,
    SAMPLE_PREPARE_OP_NAME,
    SAMPLE_LAUNCH_OP_NAME,
)
_BINDING_SEAL = object()
_EXECUTOR_SEAL = object()
_WORLD_TOKEN_SEAL = object()
_FORWARD_INTO_LIFETIME_SEAL = object()
_SAMPLE_LAUNCH_LIFETIME_SEAL = object()
_SAMPLE_COMPLETION_RECEIPT_SEAL = object()
_PENDING_SAMPLE_COMPLETION_SEAL = object()
_SAMPLE_RELEASE_COMMIT_PLAN_SEAL = object()
_SAMPLE_RELEASE_COMMIT_AUTHORIZATION_SEAL = object()
_FULL_GEOMETRY_EXECUTION_SEAL = object()
_FULL_GEOMETRY_COMPLETION_SEAL = object()
_FUSED_FULL_GEOMETRY_EXECUTION_SEAL = object()
_SESSION_SEAL = object()
_TELEMETRY_SEAL = object()
_ABORT_RELEASE_COMMIT_PLAN_SEAL = object()


@dataclass(frozen=True)
class KineticNativeMaterialRuntimeBinding:
    """Cold proof that one runtime and one ragged sampler describe one block."""

    runtime: KineticNativeEqualRankRuntimeBlock = field(repr=False)
    runtime_identity: int
    sampler_identity: int
    sampler_view_index: int
    runtime_generation_id: str
    sampler_generation_digest: str
    native_block_generation_digest: str
    generation_id: str
    _sealed_generation_id: str = field(repr=False)
    provenance: str = EXECUTOR_PROVENANCE
    _seal: object = field(default=None, repr=False)

    def assert_current(self) -> None:
        """Warm identity/layout check; full content was checked when bound."""

        if self._seal is not _BINDING_SEAL or self.provenance != EXECUTOR_PROVENANCE:
            raise ValueError("native material runtime binding was not sealed by its preparer")
        if (
            id(self.runtime) != self.runtime_identity
            or self.runtime.generation_id != self.runtime_generation_id
            or self.runtime.payload.block.generation_digest
            != self.native_block_generation_digest
            or self.generation_id != self._sealed_generation_id
            or self.generation_id != _binding_generation_id(
                self.runtime,
                sampler_generation_digest=self.sampler_generation_digest,
            )
        ):
            raise ValueError("native material runtime binding generation/provenance changed")
        self.runtime.assert_warm_layout()

    def assert_sampler_current(self, sampler: PaperKineticRowRaggedSampler) -> None:
        """Rebind the caller-owned sampler without retaining it in the executor."""

        if not isinstance(sampler, PaperKineticRowRaggedSampler):
            raise TypeError("sampler must be PaperKineticRowRaggedSampler")
        self.assert_current()
        sampler.assert_warm_layout()
        if (
            id(sampler) != self.sampler_identity
            or sampler.view_index != self.sampler_view_index
            or sampler.generation_digest != self.sampler_generation_digest
        ):
            raise ValueError("native material runtime sampler generation/provenance changed")
        matching_blocks = tuple(
            block
            for bucket in sampler.lowering.buckets
            for block in bucket.blocks
            if block.generation_digest == self.native_block_generation_digest
        )
        if len(matching_blocks) != 1 or matching_blocks[0] is not self.runtime.payload.block:
            raise ValueError("native material runtime sampler block identity changed")


@dataclass(frozen=True)
class KineticNativeMaterialExecutorMemory:
    """Logical references/bytes; not an allocator or Python-heap measurement."""

    requested_observation_count: int
    eligible_native_block_count: int
    summed_runtime_unique_retained_tensor_bytes_upper_bound: int
    executor_owned_persistent_tensor_bytes: int
    runtime_tensor_copy_bytes_allocated_by_executor: int
    retained_sampler_count: int
    persistent_frame_tensor_bytes: int
    persistent_sample_tensor_bytes: int
    persistent_target_tensor_bytes: int
    persistent_prediction_tensor_bytes: int
    caller_selected_runtime_subset: bool = True
    intended_scope: str = "one_lazily_resident_spatial_lane"
    requested_observation_count_affects_retained_bytes: bool = False
    allocator_storage_bytes_measured: bool = False
    allocator_peak_measured: bool = False


@dataclass(frozen=True)
class KineticNativeMaterialStepExecutor:
    """Cold-bound native ABI and all runtime/sampler provenance for a lane."""

    native_ops: Any = field(repr=False)
    bindings: tuple[KineticNativeMaterialRuntimeBinding, ...] = field(repr=False)
    device: torch.device
    backend_provenance: str
    native_ops_identity: int
    native_abi_identity: tuple[tuple[str, int], ...]
    binding_identities: tuple[int, ...]
    generation_id: str
    _sealed_generation_id: str = field(repr=False)
    provenance: str = EXECUTOR_PROVENANCE
    runtime_status: str = EXECUTOR_STATUS
    frame_or_sample_axis_retained: bool = False
    target_or_prediction_retained: bool = False
    geometry_vjp_exposed: bool = False
    full_geometry_vjp_available: bool = True
    native_runtime_verified: bool = False
    _seal: object = field(default=None, repr=False)

    def assert_current(self) -> None:
        if self._seal is not _EXECUTOR_SEAL:
            raise ValueError("native material executor was not sealed by its preparer")
        if (
            self.provenance != EXECUTOR_PROVENANCE
            or self.runtime_status != EXECUTOR_STATUS
            or not self.backend_provenance.strip()
            or self.frame_or_sample_axis_retained
            or self.target_or_prediction_retained
            or self.geometry_vjp_exposed
            or not self.full_geometry_vjp_available
            or self.native_runtime_verified
            or id(self.native_ops) != self.native_ops_identity
            or _native_abi_identity(self.native_ops) != self.native_abi_identity
            or tuple(id(binding) for binding in self.bindings) != self.binding_identities
            or self.generation_id != self._sealed_generation_id
            or self.generation_id != _executor_generation_id(
                native_ops=self.native_ops,
                native_abi_identity=self.native_abi_identity,
                bindings=self.bindings,
                device=self.device,
                backend_provenance=self.backend_provenance,
            )
        ):
            raise ValueError("native material executor ABI/generation contract changed")
        for binding in self.bindings:
            binding.assert_current()

    def binding_for_runtime(
        self,
        runtime: KineticNativeEqualRankRuntimeBlock,
    ) -> KineticNativeMaterialRuntimeBinding:
        matches = tuple(binding for binding in self.bindings if binding.runtime is runtime)
        if len(matches) != 1:
            raise ValueError("runtime is not uniquely bound to this native material executor")
        return matches[0]

    def memory_accounting(
        self,
        requested_observation_count: int,
    ) -> KineticNativeMaterialExecutorMemory:
        """Report cold lane-runtime residency and zero executor tensor copies.

        A live session's refreshed node charts, caller node bars, and one
        retained four-byte loss scalar per active block belong to the outer
        coordinator's active-state/peak report; they are deliberately not
        hidden in this cold executor report.
        """

        self.assert_current()
        _require_positive_int(
            requested_observation_count,
            name="requested_observation_count",
        )
        runtime_bytes = sum(
            binding.runtime.memory_accounting(
                requested_observation_count
            ).unique_retained_tensor_bytes
            for binding in self.bindings
        )
        return KineticNativeMaterialExecutorMemory(
            requested_observation_count=requested_observation_count,
            eligible_native_block_count=len(self.bindings),
            summed_runtime_unique_retained_tensor_bytes_upper_bound=runtime_bytes,
            executor_owned_persistent_tensor_bytes=0,
            runtime_tensor_copy_bytes_allocated_by_executor=0,
            retained_sampler_count=0,
            persistent_frame_tensor_bytes=0,
            persistent_sample_tensor_bytes=0,
            persistent_target_tensor_bytes=0,
            persistent_prediction_tensor_bytes=0,
        )

    def begin_step(
        self,
        *,
        step_generation_id: str,
        requested_observation_count: int,
    ) -> KineticNativeMaterialStepSession:
        """Open one single-threaded step session with no frame-sized state."""

        self.assert_current()
        if not isinstance(step_generation_id, str) or not step_generation_id.strip():
            raise ValueError("step_generation_id must be nonempty")
        _require_positive_int(
            requested_observation_count,
            name="requested_observation_count",
        )
        generation_id = _digest_parts(
            EXECUTOR_PROVENANCE,
            self.generation_id,
            step_generation_id,
            requested_observation_count,
        )
        return KineticNativeMaterialStepSession(
            executor=self,
            step_generation_id=step_generation_id,
            requested_observation_count=requested_observation_count,
            generation_id=generation_id,
            _executor_identity=id(self),
            _executor_generation_id=self.generation_id,
            _seal=_SESSION_SEAL,
        )


@dataclass(frozen=True)
class KineticNativeMaterialStepWorldToken:
    """Step-owned native world; foreign or stale sessions cannot consume it."""

    world: KineticNativeEqualRankWorld = field(repr=False)
    runtime_binding: KineticNativeMaterialRuntimeBinding = field(repr=False)
    session_generation_id: str
    runtime_generation_id: str
    native_block_generation_digest: str
    world_generation_id: str
    generation_id: str
    _session_identity: int = field(repr=False)
    _world_identity: int = field(repr=False)
    provenance: str = EXECUTOR_PROVENANCE
    _seal: object = field(default=None, repr=False)

    def assert_current(self, session: KineticNativeMaterialStepSession) -> None:
        if self._seal is not _WORLD_TOKEN_SEAL or self.provenance != EXECUTOR_PROVENANCE:
            raise ValueError("native material step world was not sealed by its executor")
        if (
            id(session) != self._session_identity
            or session.generation_id != self.session_generation_id
            or id(self.world) != self._world_identity
            or self.world.generation_id != self.world_generation_id
            or self.runtime_binding.runtime.generation_id != self.runtime_generation_id
            or self.runtime_binding.native_block_generation_digest
            != self.native_block_generation_digest
            or self.generation_id
            != _world_token_generation_id(
                session,
                self.runtime_binding,
                self.world,
            )
        ):
            raise ValueError("native material step world generation/provenance changed")
        self.runtime_binding.assert_current()
        self.world.assert_current()


@dataclass
class KineticNativeSampleLaunchLifetime:
    """Single in-flight sample launch whose roots survive its completion fence.

    The session owns at most one instance.  It installs the carrier before the
    native prepare call, fills the opaque prepared payload after preparation,
    and returns it only after the launch has been enqueued and revalidated.
    Successful settlement consumes every tensor/object reference immediately;
    the session retains no per-sample token history.
    """

    prepared_payload: Any | None = field(repr=False)
    sample_block: PaperKineticRowRaggedSampleBlock | None = field(repr=False)
    world_token: KineticNativeMaterialStepWorldToken | None = field(repr=False)
    background_rgb_f32: torch.Tensor | None = field(repr=False)
    loss_f32: torch.Tensor | None = field(repr=False)
    grad_node_chart_f32: torch.Tensor | None = field(repr=False)
    cone_diagnostic_i32: torch.Tensor | None = field(repr=False)
    session_generation_id: str
    runtime_generation_id: str
    sampler_generation_digest: str
    native_block_generation_digest: str
    sample_block_generation_digest: str
    sample_dispatch_generation_digest: str
    prior_sample_manifest_digest: str
    next_sample_manifest_digest: str
    flat_sample_identity_digest: str
    sample_count: int
    first_flat_sample_index: int
    last_flat_sample_index: int
    read_only_tensor_signatures: tuple[tuple[object, ...], ...] = field(
        repr=False
    )
    prepared_payload_signature: tuple[object, ...] = field(repr=False)
    writable_tensor_signatures_after_launch: tuple[
        tuple[object, ...], ...
    ] = field(repr=False)
    generation_digest: str
    _session_identity: int = field(repr=False)
    _block_state_identity: int = field(repr=False)
    _prepared_payload_identity: int = field(repr=False)
    _sample_block_identity: int = field(repr=False)
    _world_token_identity: int = field(repr=False)
    phase: str = "preparing"
    completion_fence_attempt_count: int = 0
    completion_unknown: bool = False
    consumed: bool = False
    _seal: object = field(default=None, repr=False)

    def assert_current(self, session: KineticNativeMaterialStepSession) -> None:
        if not isinstance(session, KineticNativeMaterialStepSession):
            raise TypeError("session must be KineticNativeMaterialStepSession")
        state = session._states.get(self.runtime_generation_id)
        sample_block = self.sample_block
        world_token = self.world_token
        prepared_payload = self.prepared_payload
        read_only_tensors = (
            world_token.world.node_chart_f32
            if isinstance(world_token, KineticNativeMaterialStepWorldToken)
            else None,
            sample_block.sample_to_node_f32
            if isinstance(sample_block, PaperKineticRowRaggedSampleBlock)
            else None,
            sample_block.target_rgb_f32
            if isinstance(sample_block, PaperKineticRowRaggedSampleBlock)
            else None,
            self.background_rgb_f32,
        )
        writable_tensors = (
            self.loss_f32,
            self.grad_node_chart_f32,
            self.cone_diagnostic_i32,
        )
        if (
            self._seal is not _SAMPLE_LAUNCH_LIFETIME_SEAL
            or self.consumed
            or self.phase != "launched"
            or id(session) != self._session_identity
            or session.generation_id != self.session_generation_id
            or session._outstanding_sample_lifetime is not self
            or state is None
            or id(state) != self._block_state_identity
            or state.token is not world_token
            or id(world_token) != self._world_token_identity
            or not isinstance(sample_block, PaperKineticRowRaggedSampleBlock)
            or id(sample_block) != self._sample_block_identity
            or prepared_payload is None
            or id(prepared_payload) != self._prepared_payload_identity
            or self.sample_block_generation_digest
            != sample_block.generation_digest
            or self.sample_dispatch_generation_digest
            != sample_block.dispatch_generation_digest
            or self.sampler_generation_digest
            != sample_block.sampler_generation_digest
            or self.native_block_generation_digest
            != sample_block.native_block_generation_digest
            or state.native_block_generation_digest
            != self.native_block_generation_digest
            or state.sampler_generation_digest
            != self.sampler_generation_digest
            or state.sample_manifest_digest != self.prior_sample_manifest_digest
            or self.sample_count != sample_block.sample_count
            or self.first_flat_sample_index < 0
            or self.last_flat_sample_index < self.first_flat_sample_index
            or len(self.flat_sample_identity_digest) != 64
            or any(not isinstance(tensor, torch.Tensor) for tensor in read_only_tensors)
            or any(not isinstance(tensor, torch.Tensor) for tensor in writable_tensors)
            or tuple(_tensor_signature(tensor) for tensor in read_only_tensors)
            != self.read_only_tensor_signatures
            or _prepared_sample_payload_signature(prepared_payload)
            != self.prepared_payload_signature
            or tuple(_tensor_signature(tensor) for tensor in writable_tensors)
            != self.writable_tensor_signatures_after_launch
            or self.generation_digest != _sample_launch_lifetime_digest(self)
        ):
            raise ValueError("native sample launch lifetime changed or is foreign")
        world_token.assert_current(session)
        sample_block.assert_warm_layout()

    def assert_retained(self, session: KineticNativeMaterialStepSession) -> None:
        """Validate a provisional/launched/unknown carrier for quarantine."""

        state = session._states.get(self.runtime_generation_id)
        if (
            self._seal is not _SAMPLE_LAUNCH_LIFETIME_SEAL
            or self.consumed
            or self.phase
            not in {"preparing", "prepared", "launched", "completion_unknown"}
            or id(session) != self._session_identity
            or session.generation_id != self.session_generation_id
            or session._outstanding_sample_lifetime is not self
            or state is None
            or id(state) != self._block_state_identity
            or state.token is not self.world_token
            or id(self.world_token) != self._world_token_identity
            or id(self.sample_block) != self._sample_block_identity
            or not isinstance(
                self.sample_block,
                PaperKineticRowRaggedSampleBlock,
            )
            or (
                self.generation_digest
                and self.generation_digest != _sample_launch_lifetime_digest(self)
            )
        ):
            raise ValueError("native sample launch lifetime is not retained")
        if self.phase in {"prepared", "launched", "completion_unknown"}:
            if (
                self.prepared_payload is None
                or id(self.prepared_payload) != self._prepared_payload_identity
                or _prepared_sample_payload_signature(self.prepared_payload)
                != self.prepared_payload_signature
            ):
                raise ValueError("native prepared sample payload is not retained")
        if self.phase == "launched":
            self.assert_current(session)

    def _commit_release_roots_after_consumed_receipt(self) -> None:
        """Assignment-only commit after sealed authority was consumed."""

        self.prepared_payload = None
        self.sample_block = None
        self.world_token = None
        self.background_rgb_f32 = None
        self.loss_f32 = None
        self.grad_node_chart_f32 = None
        self.cone_diagnostic_i32 = None
        self.phase = "released"
        self.consumed = True

    def _release_roots(self) -> None:
        """Legacy release alias; sealed code uses the explicit commit above."""

        self._commit_release_roots_after_consumed_receipt()


@dataclass(frozen=True)
class KineticNativeSampleLaunchCompletionReceipt:
    """Tensor-free proof that one sample launch was fenced and released."""

    session_generation_id: str
    runtime_generation_id: str
    native_block_generation_digest: str
    sample_lifetime_generation_digest: str
    sample_manifest_digest: str
    sample_count: int
    first_flat_sample_index: int
    last_flat_sample_index: int
    device_completion_fence_provenance: str
    generation_digest: str
    sealed_completion_capability_generation_digest: str = ""
    sealed_completion_receipt_generation_digest: str = ""
    sealed_completion_fence_sequence: int = 0
    sealed_completion_scope: str = ""
    sealed_completion_normalized_device: str = ""
    sealed_completion_launch_generation_digest: str = ""
    sealed_completion_receipt: PaperKineticCompletionFenceReceipt | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    device_completion_fence_call_count: int = 1
    maximum_simultaneous_sample_lifetime_count: int = 1
    sample_roots_released: bool = True
    retained_tensor_or_sample_reference_count: int = 0
    provenance: str = EXECUTOR_PROVENANCE
    _seal: object = field(default=None, repr=False)

    def assert_current(self) -> None:
        sealed = self.sealed_completion_receipt
        if sealed is not None:
            sealed.assert_current()
        if (
            self._seal is not _SAMPLE_COMPLETION_RECEIPT_SEAL
            or self.provenance != EXECUTOR_PROVENANCE
            or len(self.sample_lifetime_generation_digest) != 64
            or len(self.sample_manifest_digest) != 64
            or self.sample_count < 1
            or self.first_flat_sample_index < 0
            or self.last_flat_sample_index < self.first_flat_sample_index
            or not self.device_completion_fence_provenance.strip()
            or (sealed is None)
            != (
                self.sealed_completion_capability_generation_digest == ""
                and self.sealed_completion_receipt_generation_digest == ""
                and self.sealed_completion_fence_sequence == 0
                and self.sealed_completion_scope == ""
                and self.sealed_completion_normalized_device == ""
                and self.sealed_completion_launch_generation_digest == ""
            )
            or sealed is not None
            and (
                self.device_completion_fence_provenance
                != CAPABILITY_PROVENANCE
                or self.sealed_completion_capability_generation_digest
                != sealed.capability_generation_digest
                or self.sealed_completion_receipt_generation_digest
                != sealed.generation_digest
                or self.sealed_completion_fence_sequence
                != sealed.fence_sequence
                or self.sealed_completion_scope != sealed.completion_scope
                or self.sealed_completion_normalized_device
                != sealed.normalized_device
                or self.sealed_completion_launch_generation_digest
                != sealed.launch_generation_digest
            )
            or self.device_completion_fence_call_count != 1
            or self.maximum_simultaneous_sample_lifetime_count != 1
            or not self.sample_roots_released
            or self.retained_tensor_or_sample_reference_count != 0
            or self.generation_digest != _sample_completion_receipt_digest(self)
        ):
            raise ValueError("native sample completion receipt changed")


@dataclass(slots=True)
class KineticNativePendingSampleLaunchCompletion:
    """One fenced sample whose exact receipt remains unconsumed by design."""

    session_generation_id: str
    runtime_generation_id: str
    sample_lifetime_generation_digest: str
    pending_identity: int
    capability_identity: int
    capability_generation_digest: str
    capability_owner_generation_digest: str
    subject_binding_identity: int
    subject_binding_generation_digest: str
    subject_identity: int
    subject_generation_digest: str
    launch_epoch_identity: int
    launch_epoch_generation_digest: str
    launch_stage: str
    launch_generation_digest: str
    launch_epoch_sequence: int
    receipt_identity: int
    receipt_generation_digest: str
    completion_receipt_identity: int
    completion_receipt_generation_digest: str
    next_grad_node_chart_signature: tuple[object, ...] = field(repr=False)
    next_loss_signature: tuple[object, ...] = field(repr=False)
    next_state_prepare_count: int
    next_state_launch_count: int
    next_state_fence_count: int
    next_state_streamed_count: int
    next_sample_manifest_digest: str
    next_first_flat_sample_index: int
    next_last_flat_sample_index: int
    next_native_prepare_count: int
    next_native_launch_count: int
    next_native_fence_count: int
    next_streamed_count: int
    generation_digest: str
    _session_identity: int = field(repr=False)
    _sample_lifetime_identity: int = field(repr=False)
    _block_state_identity: int = field(repr=False)
    _sealed_completion_fence: PaperKineticSealedCompletionFence | None = field(
        repr=False
    )
    _subject_binding: PaperKineticCompletionSubjectBinding = field(repr=False)
    _launch_epoch: PaperKineticCompletionLaunchEpoch | None = field(repr=False)
    _sealed_completion_receipt: PaperKineticCompletionFenceReceipt = field(
        repr=False
    )
    _completion_receipt: KineticNativeSampleLaunchCompletionReceipt = field(
        repr=False
    )
    _sample_lifetime: KineticNativeSampleLaunchLifetime | None = field(repr=False)
    _block_state: _BlockExecutionState | None = field(repr=False)
    phase: str = "pending_receipt_consumption"
    receipt_consumer: str | None = None
    provenance: str = EXECUTOR_PROVENANCE
    _seal: object = field(default=None, repr=False)

    def __init_subclass__(cls, **kwargs: Any) -> NoReturn:
        raise TypeError("pending sample completions cannot be subclassed")

    @property
    def sealed_completion_receipt(self) -> PaperKineticCompletionFenceReceipt:
        """Read-only access to the exact receipt owned by this transaction."""

        return self._sealed_completion_receipt

    @property
    def subject_binding(self) -> PaperKineticCompletionSubjectBinding:
        return self._subject_binding

    def _assert_exact_relation(
        self,
        session: KineticNativeMaterialStepSession,
        capability: PaperKineticSealedCompletionFence,
        *,
        subject: Any | None,
        subject_required: bool,
        require_unconsumed: bool,
        require_installed: bool,
    ) -> None:
        if type(self) is not KineticNativePendingSampleLaunchCompletion:
            raise TypeError("pending sample completion has a foreign exact type")
        if type(session) is not KineticNativeMaterialStepSession:
            raise TypeError("pending sample completion requires its exact session")
        if type(capability) is not PaperKineticSealedCompletionFence:
            raise TypeError("pending sample completion requires its exact capability")
        lifetime = self._sample_lifetime
        state = self._block_state
        binding = self._subject_binding
        epoch = self._launch_epoch
        receipt = self._sealed_completion_receipt
        completion = self._completion_receipt
        if (
            type(lifetime) is not KineticNativeSampleLaunchLifetime
            or type(state) is not _BlockExecutionState
            or type(binding) is not PaperKineticCompletionSubjectBinding
            or type(epoch) is not PaperKineticCompletionLaunchEpoch
            or type(receipt) is not PaperKineticCompletionFenceReceipt
            or type(completion)
            is not KineticNativeSampleLaunchCompletionReceipt
        ):
            raise TypeError("pending sample completion roots changed exact type")
        capability.assert_current(
            native_ops=session.executor.native_ops,
            device=session.executor.device,
            owner_generation_digest=self.capability_owner_generation_digest,
        )
        binding.assert_current()
        epoch.assert_current()
        receipt.assert_current()
        lifetime.assert_current(session)
        expected_phase = (
            "pending_receipt_consumption"
            if require_unconsumed
            else "receipt_consumed"
        )
        if (
            self._seal is not _PENDING_SAMPLE_COMPLETION_SEAL
            or self.provenance != EXECUTOR_PROVENANCE
            or self.pending_identity < 1
            or id(self) != self.pending_identity
            or id(session) != self._session_identity
            or session.generation_id != self.session_generation_id
            or (session._pending_sample_completion is self)
            != require_installed
            or session._outstanding_sample_lifetime is not lifetime
            or id(lifetime) != self._sample_lifetime_identity
            or lifetime.generation_digest
            != self.sample_lifetime_generation_digest
            or session._states.get(self.runtime_generation_id) is not state
            or id(state) != self._block_state_identity
            or id(capability) != self.capability_identity
            or capability is not self._sealed_completion_fence
            or capability.generation_digest
            != self.capability_generation_digest
            or capability.owner_generation_digest
            != self.capability_owner_generation_digest
            or id(binding) != self.subject_binding_identity
            or binding.generation_digest
            != self.subject_binding_generation_digest
            or binding.subject_identity != self.subject_identity
            or binding.subject_generation_digest
            != self.subject_generation_digest
            or binding.capability_identity != self.capability_identity
            or id(epoch) != self.launch_epoch_identity
            or epoch.generation_digest != self.launch_epoch_generation_digest
            or not epoch.fenced
            or epoch.stage != self.launch_stage
            or epoch.launch_generation_digest
            != self.launch_generation_digest
            or epoch.launch_epoch_sequence != self.launch_epoch_sequence
            or epoch.subject_binding is not binding
            or id(receipt) != self.receipt_identity
            or receipt.generation_digest != self.receipt_generation_digest
            or receipt.receipt_identity != self.receipt_identity
            or receipt.subject_binding is not binding
            or receipt.stage != self.launch_stage
            or receipt.launch_generation_digest
            != self.launch_generation_digest
            or receipt.fence_sequence != self.launch_epoch_sequence
            or id(completion) != self.completion_receipt_identity
            or completion.generation_digest
            != self.completion_receipt_generation_digest
            or completion._seal is not None
            or completion.sealed_completion_receipt is not receipt
            or completion.session_generation_id != self.session_generation_id
            or completion.runtime_generation_id != self.runtime_generation_id
            or completion.native_block_generation_digest
            != state.native_block_generation_digest
            or completion.sample_lifetime_generation_digest
            != self.sample_lifetime_generation_digest
            or completion.sample_manifest_digest
            != lifetime.next_sample_manifest_digest
            or completion.sample_count != lifetime.sample_count
            or completion.first_flat_sample_index
            != lifetime.first_flat_sample_index
            or completion.last_flat_sample_index
            != lifetime.last_flat_sample_index
            or completion.device_completion_fence_provenance
            != CAPABILITY_PROVENANCE
            or completion.device_completion_fence_call_count != 1
            or not completion.sample_roots_released
            or completion.retained_tensor_or_sample_reference_count != 0
            or completion.generation_digest
            != _sample_completion_receipt_digest(completion)
            or lifetime.completion_fence_attempt_count != 1
            or state.native_sample_prepare_count + 1
            != self.next_state_prepare_count
            or state.native_sample_launch_count + 1
            != self.next_state_launch_count
            or state.native_sample_completion_fence_count + 1
            != self.next_state_fence_count
            or state.streamed_sample_count + lifetime.sample_count
            != self.next_state_streamed_count
            or lifetime.next_sample_manifest_digest
            != self.next_sample_manifest_digest
            or (
                lifetime.first_flat_sample_index
                if state.first_flat_sample_index is None
                else state.first_flat_sample_index
            )
            != self.next_first_flat_sample_index
            or lifetime.last_flat_sample_index
            != self.next_last_flat_sample_index
            or session._native_sample_prepare_count + 1
            != self.next_native_prepare_count
            or session._native_sample_launch_count + 1
            != self.next_native_launch_count
            or session._native_sample_completion_fence_count + 1
            != self.next_native_fence_count
            or session._streamed_sample_count + lifetime.sample_count
            != self.next_streamed_count
            or _tensor_signature(lifetime.grad_node_chart_f32)
            != self.next_grad_node_chart_signature
            or _tensor_signature(lifetime.loss_f32)
            != self.next_loss_signature
            or session._failed_sample_completion_sealed_receipt is not receipt
            or session._failed_sample_completion_launch_generation_digest
            != self.launch_generation_digest
            or not session._failed_sample_completion_fence_succeeded
            or self.phase != expected_phase
            or (receipt.consumed is False) != require_unconsumed
            or (self.receipt_consumer is None) != require_unconsumed
            or self.generation_digest
            != _pending_sample_completion_digest(self)
        ):
            raise ValueError("pending sample completion changed or is foreign")
        receipt.assert_for(
            capability,
            stage=self.launch_stage,
            launch_generation_digest=self.launch_generation_digest,
            fence_sequence=self.launch_epoch_sequence,
            require_unconsumed=require_unconsumed,
        )
        if subject_required:
            receipt.assert_for_subject(
                capability,
                binding,
                subject=subject,
                require_unconsumed=require_unconsumed,
            )

    def assert_exact_sealed_receipt_relation(
        self,
        session: KineticNativeMaterialStepSession,
        capability: PaperKineticSealedCompletionFence,
        *,
        subject: Any,
        require_unconsumed: bool = True,
    ) -> PaperKineticCompletionFenceReceipt:
        """Read-only validation with no caller-supplied launch relation."""

        self._assert_exact_relation(
            session,
            capability,
            subject=subject,
            subject_required=True,
            require_unconsumed=require_unconsumed,
            require_installed=True,
        )
        return self._sealed_completion_receipt

    def consume_sealed_receipt_for_outer_composite(
        self,
        session: KineticNativeMaterialStepSession,
        capability: PaperKineticSealedCompletionFence,
        *,
        subject: Any,
        consumer: str,
    ) -> _KineticNativeSampleReleaseCommitPlan:
        """Consume once and return the already-prepared non-fallible commit."""

        if not isinstance(consumer, str) or not consumer.strip():
            raise ValueError("pending sample receipt consumer must be nonempty")
        self._assert_exact_relation(
            session,
            capability,
            subject=subject,
            subject_required=True,
            require_unconsumed=True,
            require_installed=True,
        )
        commit_plan = _KineticNativeSampleReleaseCommitPlan(
            session_identity=id(session),
            plan_identity=0,
            pending_identity=id(self),
            pending=self,
            lifetime=self._sample_lifetime,
            state=self._block_state,
            completion=self._completion_receipt,
            next_grad_node_chart_signature=(
                self.next_grad_node_chart_signature
            ),
            next_loss_signature=self.next_loss_signature,
            next_state_prepare_count=self.next_state_prepare_count,
            next_state_launch_count=self.next_state_launch_count,
            next_state_fence_count=self.next_state_fence_count,
            next_state_streamed_count=self.next_state_streamed_count,
            next_sample_manifest_digest=self.next_sample_manifest_digest,
            next_first_flat_sample_index=self.next_first_flat_sample_index,
            next_last_flat_sample_index=self.next_last_flat_sample_index,
            next_native_prepare_count=self.next_native_prepare_count,
            next_native_launch_count=self.next_native_launch_count,
            next_native_fence_count=self.next_native_fence_count,
            next_streamed_count=self.next_streamed_count,
            _authorization=_KineticNativeSampleReleaseCommitAuthorization(
                _seal=_SAMPLE_RELEASE_COMMIT_AUTHORIZATION_SEAL,
            ),
            _seal=_SAMPLE_RELEASE_COMMIT_PLAN_SEAL,
        )
        object.__setattr__(commit_plan, "plan_identity", id(commit_plan))
        self._sealed_completion_receipt.consume_for_subject(
            capability,
            self._subject_binding,
            subject=subject,
            consumer=consumer,
        )
        commit_plan._authorization.authorized_after_receipt_consume = True
        self.receipt_consumer = consumer
        self.phase = "receipt_consumed"
        return commit_plan


@dataclass
class KineticNativeFullGeometryVJPExecution:
    """Executor-sealed full reverse plus its actual block sample/loss coverage."""

    native_vjp_result: KineticNativeEqualRankVJPResult | None = field(repr=False)
    session_generation_id: str
    runtime_generation_id: str
    native_block_generation_digest: str
    reduced_sample_chunk_count: int
    reduced_sample_count: int
    sample_manifest_digest: str
    first_flat_sample_index: int
    last_flat_sample_index: int
    node_bar_identity: int
    node_bar_signature: tuple[object, ...] = field(repr=False)
    loss_identity: int
    loss_signature: tuple[object, ...] = field(repr=False)
    native_length_bar_tensor_bytes: int
    generation_digest: str
    _session_identity: int = field(repr=False)
    _native_vjp_result_identity: int = field(repr=False)
    provenance: str = EXECUTOR_PROVENANCE
    consumed: bool = False
    _seal: object = field(default=None, repr=False)

    def assert_current(
        self,
        session: KineticNativeMaterialStepSession,
        *,
        loss_f32: torch.Tensor | None = None,
    ) -> None:
        if not isinstance(session, KineticNativeMaterialStepSession):
            raise TypeError("session must be KineticNativeMaterialStepSession")
        native_vjp_result = self.native_vjp_result
        if not isinstance(native_vjp_result, KineticNativeEqualRankVJPResult):
            raise ValueError("full-geometry VJP execution no longer owns its native result")
        native_vjp_result.assert_warm_layout()
        state = session._states.get(self.runtime_generation_id)
        if (
            self._seal is not _FULL_GEOMETRY_EXECUTION_SEAL
            or self.provenance != EXECUTOR_PROVENANCE
            or self.consumed
            or id(session) != self._session_identity
            or session.generation_id != self.session_generation_id
            or id(native_vjp_result) != self._native_vjp_result_identity
            or state is None
            or state.full_geometry_execution is not self
            or not state.full_geometry_execution_outstanding
            or state.full_geometry_execution_consumed
            or state.full_geometry_vjp_launch_count != 1
            or state.material_vjp_launch_count != 0
            or state.reverse_result_identity != id(self.native_vjp_result)
            or state.native_block_generation_digest
            != self.native_block_generation_digest
            or state.native_sample_launch_count != self.reduced_sample_chunk_count
            or state.streamed_sample_count != self.reduced_sample_count
            or state.sample_manifest_digest != self.sample_manifest_digest
            or state.first_flat_sample_index != self.first_flat_sample_index
            or state.last_flat_sample_index != self.last_flat_sample_index
            or state.grad_node_chart_identity != self.node_bar_identity
            or state.grad_node_chart_signature != self.node_bar_signature
            or state.loss_identity != self.loss_identity
            or state.loss_signature != self.loss_signature
            or (
                loss_f32 is not None
                and (
                    id(loss_f32) != self.loss_identity
                    or _tensor_signature(loss_f32) != self.loss_signature
                )
            )
            or state.native_length_bar_tensor_bytes
            != self.native_length_bar_tensor_bytes
            or self.native_length_bar_tensor_bytes
            != native_vjp_result.grad_node_physical_length_f32.numel()
            * native_vjp_result.grad_node_physical_length_f32.element_size()
            or self.generation_digest != _full_geometry_execution_digest(self)
        ):
            raise ValueError("full-geometry VJP execution receipt changed or is foreign")


@dataclass(frozen=True)
class KineticNativeFullGeometryVJPCompletionReceipt:
    """Tensor-free proof that this execution was fenced and fully reduced."""

    session_generation_id: str
    runtime_generation_id: str
    native_block_generation_digest: str
    execution_generation_digest: str
    native_vjp_provenance_id: str
    native_length_bar_shape: tuple[int, int]
    native_length_bar_tensor_bytes: int
    native_length_bar_signature: tuple[object, ...] = field(repr=False)
    geometry_reduction_identity: int
    geometry_reduction_generation_digest: str
    reduction_completion_fence_provenance: str
    generation_digest: str
    geometry_reduction_success_count: int = 1
    reduction_completion_fence_call_count: int = 1
    execution_consumed: bool = True
    native_or_geometry_tensors_retained: bool = False
    global_accumulation_proven: bool = False
    completion_semantics: str = "fenced_and_reduced_not_globally_committed"
    provenance: str = EXECUTOR_PROVENANCE
    _seal: object = field(default=None, repr=False)

    def assert_current(self) -> None:
        if (
            self._seal is not _FULL_GEOMETRY_COMPLETION_SEAL
            or self.provenance != EXECUTOR_PROVENANCE
            or not self.session_generation_id.strip()
            or not self.runtime_generation_id.strip()
            or not self.native_block_generation_digest.strip()
            or len(self.execution_generation_digest) != 64
            or len(self.native_vjp_provenance_id) != 64
            or len(self.native_length_bar_shape) != 2
            or min(self.native_length_bar_shape) < 1
            or self.native_length_bar_tensor_bytes
            != self.native_length_bar_shape[0]
            * self.native_length_bar_shape[1]
            * 4
            or len(self.native_length_bar_signature) != 10
            or self.native_length_bar_signature[4]
            != self.native_length_bar_shape
            or self.native_length_bar_signature[6] != torch.float32
            or self.geometry_reduction_identity < 1
            or len(self.geometry_reduction_generation_digest) != 64
            or not self.reduction_completion_fence_provenance.strip()
            or self.geometry_reduction_success_count != 1
            or self.reduction_completion_fence_call_count != 1
            or not self.execution_consumed
            or self.native_or_geometry_tensors_retained
            or self.global_accumulation_proven
            or self.completion_semantics
            != "fenced_and_reduced_not_globally_committed"
            or self.generation_digest != _full_geometry_completion_digest(self)
        ):
            raise ValueError("full-geometry completion receipt changed")


@dataclass(frozen=True)
class KineticNativeFusedFullGeometryVJPExecutionReceipt:
    """Executor proof that one fused transaction covered the active manifest.

    The wrapped adapter receipt deliberately does not certify active-manifest
    coverage.  This session-level receipt adds that fact from ``_states`` and
    the executor's canonical binding order.  It is still only an accepted
    gradient payload for a later out-of-place optimizer commit.
    """

    transaction_result: (
        KineticNativeEqualRankFusedDirectFullVjpV1TransactionResult
    ) = field(repr=False)
    session_generation_id: str
    active_runtime_generation_ids: tuple[str, ...]
    active_block_generation_ids: tuple[str, ...]
    active_world_generation_ids: tuple[str, ...]
    node_bar_identities: tuple[int, ...]
    node_bar_signatures: tuple[tuple[object, ...], ...] = field(repr=False)
    sample_manifest_digests: tuple[str, ...]
    reduced_sample_count: int
    transaction_result_identity: int
    transaction_generation_id: str
    retained_output_tensor_bytes: int
    device_completion_fence_provenance: str
    generation_digest: str
    active_block_count: int
    block_reverse_count: int
    transaction_count: int = 1
    device_completion_fence_call_count: int = 1
    active_manifest_coverage_certified: bool = True
    exact_token_world_node_bar_identity_certified: bool = True
    length_cotangent_allocated: bool = False
    optimizer_fail_atomicity_certified: bool = False
    optimizer_commit_performed: bool = False
    native_runtime_verified: bool = False
    camera_mode: str = "fixed"
    ray_cotangent_surface_exposed: bool = False
    provenance: str = EXECUTOR_PROVENANCE
    _session_identity: int = field(default=0, repr=False)
    _seal: object = field(default=None, repr=False)

    def assert_current(
        self,
        session: KineticNativeMaterialStepSession | None = None,
    ) -> None:
        result = self.transaction_result
        if not isinstance(
            result,
            KineticNativeEqualRankFusedDirectFullVjpV1TransactionResult,
        ):
            raise ValueError("fused full-geometry receipt lost its transaction result")
        result.assert_current()
        if (
            self._seal is not _FUSED_FULL_GEOMETRY_EXECUTION_SEAL
            or self.provenance != EXECUTOR_PROVENANCE
            or not self.session_generation_id.strip()
            or self.active_block_count < 1
            or self.block_reverse_count != self.active_block_count
            or self.transaction_count != 1
            or self.device_completion_fence_call_count != 1
            or len(self.active_runtime_generation_ids) != self.active_block_count
            or len(set(self.active_runtime_generation_ids)) != self.active_block_count
            or len(self.active_block_generation_ids) != self.active_block_count
            or len(set(self.active_block_generation_ids)) != self.active_block_count
            or len(self.active_world_generation_ids) != self.active_block_count
            or len(self.node_bar_identities) != self.active_block_count
            or len(set(self.node_bar_identities)) != self.active_block_count
            or len(self.node_bar_signatures) != self.active_block_count
            or len(self.sample_manifest_digests) != self.active_block_count
            or any(len(digest) != 64 for digest in self.sample_manifest_digests)
            or self.reduced_sample_count < self.active_block_count
            or id(result) != self.transaction_result_identity
            or self.transaction_generation_id != result.transaction_generation_id
            or self.active_block_generation_ids
            != result.active_block_generation_ids
            or self.retained_output_tensor_bytes
            != result.retained_output_tensor_bytes
            or self.device_completion_fence_provenance
            != result.device_completion_fence_provenance
            or result.device_completion_fence_call_count != 1
            or result.active_manifest_coverage_certified
            or result.length_cotangent_allocated
            or not self.active_manifest_coverage_certified
            or not self.exact_token_world_node_bar_identity_certified
            or self.length_cotangent_allocated
            or self.optimizer_fail_atomicity_certified
            or self.optimizer_commit_performed
            or self.native_runtime_verified
            or self.camera_mode != "fixed"
            or self.ray_cotangent_surface_exposed
            or self.generation_digest != _fused_full_geometry_execution_digest(self)
        ):
            raise ValueError("fused full-geometry execution receipt changed")
        if session is not None:
            if not isinstance(session, KineticNativeMaterialStepSession):
                raise TypeError("session must be KineticNativeMaterialStepSession")
            ordered_states = session._ordered_active_states()
            if (
                id(session) != self._session_identity
                or session.generation_id != self.session_generation_id
                or session._reverse_mode != "fused_full_geometry"
                or session._fused_full_geometry_execution_receipt is not self
                or tuple(state.runtime_generation_id for state in ordered_states)
                != self.active_runtime_generation_ids
                or tuple(
                    state.native_block_generation_digest for state in ordered_states
                )
                != self.active_block_generation_ids
                or tuple(state.world_generation_id for state in ordered_states)
                != self.active_world_generation_ids
                or tuple(
                    state.grad_node_chart_identity for state in ordered_states
                )
                != self.node_bar_identities
                or tuple(
                    state.grad_node_chart_signature for state in ordered_states
                )
                != self.node_bar_signatures
                or tuple(
                    state.sample_manifest_digest for state in ordered_states
                )
                != self.sample_manifest_digests
                or sum(
                    state.streamed_sample_count for state in ordered_states
                )
                != self.reduced_sample_count
                or any(
                    state.fused_full_geometry_vjp_launch_count != 1
                    or state.material_vjp_launch_count != 0
                    or state.full_geometry_vjp_launch_count != 0
                    or state.fused_transaction_generation_id
                    != self.transaction_generation_id
                    for state in ordered_states
                )
            ):
                raise ValueError(
                    "fused full-geometry receipt no longer matches its active session"
                )


@dataclass(frozen=True)
class KineticNativeMaterialBlockCallTelemetry:
    runtime_generation_id: str
    sampler_generation_digest: str
    native_block_generation_digest: str
    world_generation_id: str
    native_node_forward_launch_count: int
    native_sample_prepare_count: int
    native_sample_launch_count: int
    native_sample_completion_fence_count: int
    streamed_sample_count: int
    sample_manifest_digest: str
    first_flat_sample_index: int
    last_flat_sample_index: int
    reverse_mode: str
    native_material_word_vjp_launch_count: int
    executor_world_reference_released_after_reverse_completion: bool
    native_full_geometry_vjp_launch_count: int = 0
    native_full_geometry_fenced_reduction_count: int = 0
    native_fused_full_geometry_vjp_launch_count: int = 0
    native_length_bar_tensor_bytes: int = 0
    full_geometry_fenced_reduction_generation_digest: str = ""
    geometry_reduction_generation_digest: str = ""
    reduction_completion_fence_provenance: str = ""
    fused_transaction_generation_id: str = ""


@dataclass(frozen=True)
class KineticNativeMaterialStepTelemetry:
    executor_generation_id: str
    step_generation_id: str
    session_generation_id: str
    requested_observation_count: int
    eligible_native_block_count: int
    active_native_block_count: int
    native_node_forward_launch_count: int
    native_sample_prepare_count: int
    native_sample_launch_count: int
    native_sample_completion_fence_count: int
    streamed_sample_count: int
    native_material_word_vjp_launch_count: int
    native_full_geometry_vjp_launch_count: int
    native_full_geometry_fenced_reduction_count: int
    native_length_bar_tensor_bytes: int
    reverse_mode: str
    global_loss_element_count: int
    loss_scale: float
    loss_normalization_id: str
    blocks: tuple[KineticNativeMaterialBlockCallTelemetry, ...]
    generation_digest: str
    _sealed_generation_digest: str = field(repr=False)
    provenance: str = EXECUTOR_PROVENANCE
    runtime_status: str = EXECUTOR_STATUS
    call_count_scope: str = "python_executor_launch_boundary"
    exactly_one_forward_per_active_block: bool = True
    exactly_one_material_vjp_per_active_block: bool = True
    exactly_one_full_geometry_vjp_per_active_block: bool = False
    exactly_one_reverse_per_active_block: bool = True
    frame_or_sample_axis_retained: bool = False
    geometry_vjp_exposed: bool = False
    executor_retained_world_reference_count: int = 0
    full_geometry_global_accumulation_proven: bool = False
    full_geometry_completion_semantics: str = (
        "fenced_and_reduced_not_globally_committed"
    )
    native_runtime_verified: bool = False
    allocator_peak_measured: bool = False
    native_fused_full_geometry_vjp_launch_count: int = 0
    native_fused_full_geometry_transaction_count: int = 0
    native_fused_full_geometry_completion_fence_count: int = 0
    fused_full_geometry_output_tensor_bytes: int = 0
    fused_full_geometry_transaction_generation_id: str = ""
    fused_full_geometry_completion_fence_provenance: str = ""
    exactly_one_fused_full_geometry_vjp_per_active_block: bool = False
    fused_full_geometry_active_manifest_certified: bool = False
    fused_full_geometry_length_cotangent_allocated: bool = False
    optimizer_fail_atomicity_certified: bool = False
    sample_completion_fence_provenance: str = ""
    maximum_simultaneous_sample_lifetime_count: int = 1
    outstanding_sample_lifetime_count_at_seal: int = 0
    sample_lifetime_history_retained: bool = False
    sample_lifetime_additional_logical_tensor_bytes: int = 0
    sample_lifetime_python_heap_bytes_measured: bool = False
    _seal: object = field(default=None, repr=False)

    def assert_current(self) -> None:
        if self._seal is not _TELEMETRY_SEAL:
            raise ValueError("native step telemetry was not sealed by its executor")
        if self.reverse_mode not in {
            "material_only",
            "full_geometry",
            "fused_full_geometry",
            "fused_union_v2_full_geometry",
        }:
            raise ValueError("native step telemetry has an invalid reverse mode")
        material_only = self.reverse_mode == "material_only"
        staged_full_geometry = self.reverse_mode == "full_geometry"
        fused_full_geometry = self.reverse_mode in {
            "fused_full_geometry",
            "fused_union_v2_full_geometry",
        }
        expected_material_vjps = self.active_native_block_count if material_only else 0
        expected_full_vjps = (
            self.active_native_block_count if staged_full_geometry else 0
        )
        expected_fused_vjps = (
            self.active_native_block_count if fused_full_geometry else 0
        )
        expected_completion_semantics = (
            "one_fenced_all_block_transaction_not_optimizer_committed"
            if fused_full_geometry
            else "fenced_and_reduced_not_globally_committed"
        )
        if (
            self.provenance != EXECUTOR_PROVENANCE
            or self.runtime_status != EXECUTOR_STATUS
            or self.call_count_scope != "python_executor_launch_boundary"
            or not self.exactly_one_forward_per_active_block
            or self.exactly_one_material_vjp_per_active_block != material_only
            or self.exactly_one_full_geometry_vjp_per_active_block
            != staged_full_geometry
            or self.exactly_one_fused_full_geometry_vjp_per_active_block
            != fused_full_geometry
            or not self.exactly_one_reverse_per_active_block
            or self.frame_or_sample_axis_retained
            or self.geometry_vjp_exposed == material_only
            or self.executor_retained_world_reference_count != 0
            or self.full_geometry_global_accumulation_proven
            or self.full_geometry_completion_semantics
            != expected_completion_semantics
            or self.native_runtime_verified
            or self.allocator_peak_measured
            or self.optimizer_fail_atomicity_certified
            or self.active_native_block_count != len(self.blocks)
            or not 0 < self.active_native_block_count <= self.eligible_native_block_count
            or self.native_node_forward_launch_count != self.active_native_block_count
            or self.native_material_word_vjp_launch_count != expected_material_vjps
            or self.native_full_geometry_vjp_launch_count != expected_full_vjps
            or self.native_full_geometry_fenced_reduction_count
            != expected_full_vjps
            or self.native_fused_full_geometry_vjp_launch_count
            != expected_fused_vjps
            or self.native_fused_full_geometry_transaction_count
            != (1 if fused_full_geometry else 0)
            or self.native_fused_full_geometry_completion_fence_count
            != (1 if fused_full_geometry else 0)
            or self.native_length_bar_tensor_bytes
            != (
                sum(block.native_length_bar_tensor_bytes for block in self.blocks)
                if staged_full_geometry
                else 0
            )
            or (staged_full_geometry and self.native_length_bar_tensor_bytes < 1)
            or self.fused_full_geometry_output_tensor_bytes
            < (1 if fused_full_geometry else 0)
            or (
                not fused_full_geometry
                and self.fused_full_geometry_output_tensor_bytes != 0
            )
            or bool(self.fused_full_geometry_transaction_generation_id)
            != fused_full_geometry
            or bool(self.fused_full_geometry_completion_fence_provenance)
            != fused_full_geometry
            or self.fused_full_geometry_active_manifest_certified
            != fused_full_geometry
            or self.fused_full_geometry_length_cotangent_allocated
            or self.native_sample_prepare_count != self.native_sample_launch_count
            or self.native_sample_launch_count
            != self.native_sample_completion_fence_count
            or not self.sample_completion_fence_provenance.strip()
            or self.maximum_simultaneous_sample_lifetime_count != 1
            or self.outstanding_sample_lifetime_count_at_seal != 0
            or self.sample_lifetime_history_retained
            or self.sample_lifetime_additional_logical_tensor_bytes != 0
            or self.sample_lifetime_python_heap_bytes_measured
            or self.streamed_sample_count != self.requested_observation_count
            or self.global_loss_element_count < self.requested_observation_count * 3
            or self.loss_scale != 1.0 / float(self.global_loss_element_count)
            or not self.loss_normalization_id.strip()
            or self.native_sample_prepare_count
            != sum(block.native_sample_prepare_count for block in self.blocks)
            or self.native_sample_launch_count
            != sum(block.native_sample_launch_count for block in self.blocks)
            or self.native_sample_completion_fence_count
            != sum(
                block.native_sample_completion_fence_count
                for block in self.blocks
            )
            or self.streamed_sample_count
            != sum(block.streamed_sample_count for block in self.blocks)
            or self.native_full_geometry_fenced_reduction_count
            != sum(
                block.native_full_geometry_fenced_reduction_count
                for block in self.blocks
            )
            or self.native_fused_full_geometry_vjp_launch_count
            != sum(
                block.native_fused_full_geometry_vjp_launch_count
                for block in self.blocks
            )
            or len({block.runtime_generation_id for block in self.blocks})
            != self.active_native_block_count
            or any(
                block.native_node_forward_launch_count != 1
                or block.native_sample_prepare_count != block.native_sample_launch_count
                or block.native_sample_launch_count
                != block.native_sample_completion_fence_count
                or block.native_sample_launch_count < 1
                or block.streamed_sample_count < 1
                or len(block.sample_manifest_digest) != 64
                or block.first_flat_sample_index < 0
                or block.last_flat_sample_index < block.first_flat_sample_index
                or block.reverse_mode != self.reverse_mode
                or block.native_material_word_vjp_launch_count
                != (1 if material_only else 0)
                or block.native_full_geometry_vjp_launch_count
                != (1 if staged_full_geometry else 0)
                or block.native_full_geometry_fenced_reduction_count
                != (1 if staged_full_geometry else 0)
                or block.native_fused_full_geometry_vjp_launch_count
                != (1 if fused_full_geometry else 0)
                or not block.executor_world_reference_released_after_reverse_completion
                or (
                    not staged_full_geometry
                    and block.native_length_bar_tensor_bytes != 0
                )
                or (staged_full_geometry and block.native_length_bar_tensor_bytes < 1)
                or (
                    not staged_full_geometry
                    and (
                        block.full_geometry_fenced_reduction_generation_digest
                        or block.geometry_reduction_generation_digest
                        or block.reduction_completion_fence_provenance
                    )
                )
                or (
                    staged_full_geometry
                    and (
                        len(block.full_geometry_fenced_reduction_generation_digest)
                        != 64
                        or len(block.geometry_reduction_generation_digest) != 64
                        or not block.reduction_completion_fence_provenance.strip()
                    )
                )
                or bool(block.fused_transaction_generation_id)
                != fused_full_geometry
                or (
                    fused_full_geometry
                    and block.fused_transaction_generation_id
                    != self.fused_full_geometry_transaction_generation_id
                )
                for block in self.blocks
            )
            or self.generation_digest != self._sealed_generation_digest
            or self.generation_digest != _telemetry_generation_digest(self)
        ):
            raise ValueError("native step telemetry contract changed")


@dataclass
class KineticNativeNodeForwardIntoLifetime:
    """Caller-visible roots for the explicit into-output node forward.

    The coordinator installs this carrier before allocating the node-chart
    output.  Every returned output/world/token is published here before any
    later validation.  A reverse or abort completion fence must succeed before
    ``retire_after_completion_fence`` drops the roots.  The lifetime is
    intentionally one-block bounded and contains no frame/sample history.
    """

    session_generation_id: str
    runtime: KineticNativeEqualRankRuntimeBlock | None = field(repr=False)
    compact_site_rgba_f32: torch.Tensor | None = field(repr=False)
    node_chart_out_f32: torch.Tensor | None = field(default=None, repr=False)
    world: KineticNativeEqualRankWorld | None = field(default=None, repr=False)
    token: KineticNativeMaterialStepWorldToken | None = field(
        default=None,
        repr=False,
    )
    phase: str = "installed"
    completion_fenced: bool = False
    _runtime_identity: int = field(default=0, repr=False)
    _compact_identity: int = field(default=0, repr=False)
    _forward_into_implementation_id: int = field(default=0, repr=False)
    _output_identity: int | None = field(default=None, repr=False)
    _world_identity: int | None = field(default=None, repr=False)
    _token_identity: int | None = field(default=None, repr=False)
    _seal: object = field(default=None, repr=False)

    def assert_retained(
        self,
        session: KineticNativeMaterialStepSession | None = None,
    ) -> None:
        if self._seal is not _FORWARD_INTO_LIFETIME_SEAL:
            raise ValueError("native forward-into lifetime was not sealed")
        if session is not None and self.session_generation_id != session.generation_id:
            raise ValueError("native forward-into lifetime belongs to another session")
        if self.phase == "released":
            if (
                not self.completion_fenced
                or self.runtime is not None
                or self.compact_site_rgba_f32 is not None
                or self.node_chart_out_f32 is not None
                or self.world is not None
                or self.token is not None
            ):
                raise ValueError("released native forward-into lifetime retained roots")
            return
        if self.phase not in {
            "installed",
            "output_published",
            "world_published",
            "token_published",
            "active",
        }:
            raise ValueError("native forward-into lifetime has an invalid phase")
        if (
            not isinstance(self.runtime, KineticNativeEqualRankRuntimeBlock)
            or id(self.runtime) != self._runtime_identity
            or not isinstance(self.compact_site_rgba_f32, torch.Tensor)
            or id(self.compact_site_rgba_f32) != self._compact_identity
            or self.completion_fenced
        ):
            raise ValueError("native forward-into predecessor roots changed")
        forward_into = getattr(self.runtime.native_ops, FORWARD_INTO_OP_NAME, None)
        implementation = getattr(forward_into, "__func__", forward_into)
        if (
            not callable(forward_into)
            or id(implementation) != self._forward_into_implementation_id
        ):
            raise ValueError("native forward-into callable identity changed")
        output_expected = self.phase != "installed"
        world_expected = self.phase in {"world_published", "token_published", "active"}
        token_expected = self.phase in {"token_published", "active"}
        if output_expected != isinstance(self.node_chart_out_f32, torch.Tensor):
            raise ValueError("native forward-into output publication changed")
        if output_expected and id(self.node_chart_out_f32) != self._output_identity:
            raise ValueError("native forward-into output identity changed")
        if world_expected != isinstance(self.world, KineticNativeEqualRankWorld):
            raise ValueError("native forward-into world publication changed")
        if world_expected and id(self.world) != self._world_identity:
            raise ValueError("native forward-into world identity changed")
        if token_expected != isinstance(self.token, KineticNativeMaterialStepWorldToken):
            raise ValueError("native forward-into token publication changed")
        if token_expected and id(self.token) != self._token_identity:
            raise ValueError("native forward-into token identity changed")

    def publish_output(self, node_chart_out_f32: torch.Tensor) -> None:
        self.assert_retained()
        if self.phase != "installed" or not isinstance(node_chart_out_f32, torch.Tensor):
            raise ValueError("native forward-into output publication is not current")
        self.node_chart_out_f32 = node_chart_out_f32
        self._output_identity = id(node_chart_out_f32)
        self.phase = "output_published"
        self.assert_retained()

    def publish_world(self, world: KineticNativeEqualRankWorld) -> None:
        self.assert_retained()
        if self.phase != "output_published" or not isinstance(
            world,
            KineticNativeEqualRankWorld,
        ):
            raise ValueError("native forward-into world publication is not current")
        self.world = world
        self._world_identity = id(world)
        self.phase = "world_published"
        self.assert_retained()

    def publish_token(self, token: KineticNativeMaterialStepWorldToken) -> None:
        self.assert_retained()
        if self.phase != "world_published" or not isinstance(
            token,
            KineticNativeMaterialStepWorldToken,
        ):
            raise ValueError("native forward-into token publication is not current")
        self.token = token
        self._token_identity = id(token)
        self.phase = "token_published"
        self.assert_retained()

    def retire_after_completion_fence(self) -> None:
        self.assert_retained()
        if self.compact_site_rgba_f32.device.type != "cpu":
            raise RuntimeError(
                "authority-free forward lifetime release is CPU-only"
            )
        self._commit_retire_after_consumed_receipt()

    def _commit_retire_after_consumed_receipt(self) -> None:
        """Assignment-only root clear after exact completion authority."""

        self.completion_fenced = True
        self.runtime = None
        self.compact_site_rgba_f32 = None
        self.node_chart_out_f32 = None
        self.world = None
        self.token = None
        self.phase = "released"

    def retire_after_sealed_completion_receipt(
        self,
        receipt: PaperKineticCompletionFenceReceipt,
        capability: PaperKineticSealedCompletionFence,
        *,
        stage: str,
        launch_generation_digest: str,
        expected_fence_sequence: int,
    ) -> None:
        """Release only under the exact outstanding lazy-lane receipt."""

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
            consumer="native-forward-into-lifetime-release",
        )
        self._commit_retire_after_consumed_receipt()

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
        if (
            capability.backend_type != "cpu"
            or self.compact_site_rgba_f32.device.type != "cpu"
        ):
            raise RuntimeError(
                "unbound forward lifetime receipt release is CPU-only"
            )
        receipt.assert_for(
            capability,
            stage=stage,
            launch_generation_digest=launch_generation_digest,
            fence_sequence=expected_fence_sequence,
            require_unconsumed=True,
        )

    def retire_after_all_sealed_completion_receipts(
        self,
        capability: PaperKineticSealedCompletionFence,
        *,
        expected_last_consumed_sequence: int,
    ) -> None:
        """Release when the bound capability has no live epoch or receipt."""

        self.assert_retained()
        if (
            capability.backend_type != "cpu"
            or self.compact_site_rgba_f32.device.type != "cpu"
        ):
            raise RuntimeError(
                "unbound forward lifetime ledger release is CPU-only"
            )
        capability.assert_current(native_ops=self.runtime.native_ops)
        if (
            capability.registered_launch_epoch is not None
            or capability.outstanding_receipt_sequence is not None
            or capability.consumed_fence_count
            != expected_last_consumed_sequence
            or capability.successful_fence_count
            != expected_last_consumed_sequence
        ):
            raise ValueError("forward-into completion ledger is not fully consumed")
        self._commit_retire_after_consumed_receipt()


@dataclass
class _BlockExecutionState:
    runtime_binding: KineticNativeMaterialRuntimeBinding
    runtime_generation_id: str
    sampler_generation_digest: str
    native_block_generation_digest: str
    world_generation_id: str
    token: KineticNativeMaterialStepWorldToken | None
    token_identity: int
    native_sample_prepare_count: int = 0
    native_sample_launch_count: int = 0
    native_sample_completion_fence_count: int = 0
    streamed_sample_count: int = 0
    sample_manifest_digest: str = ""
    first_flat_sample_index: int | None = None
    last_flat_sample_index: int | None = None
    material_vjp_launch_count: int = 0
    full_geometry_vjp_launch_count: int = 0
    fused_full_geometry_vjp_launch_count: int = 0
    fused_transaction_generation_id: str = ""
    native_length_bar_tensor_bytes: int = 0
    reverse_result_identity: int | None = None
    grad_node_chart_identity: int | None = None
    grad_node_chart_signature: tuple[object, ...] | None = None
    grad_node_chart_f32: torch.Tensor | None = field(default=None, repr=False)
    loss_identity: int | None = None
    loss_signature: tuple[object, ...] | None = None
    loss_f32: torch.Tensor | None = field(default=None, repr=False)
    full_geometry_execution: KineticNativeFullGeometryVJPExecution | None = None
    full_geometry_completion_receipt: (
        KineticNativeFullGeometryVJPCompletionReceipt | None
    ) = None
    full_geometry_execution_outstanding: bool = False
    full_geometry_execution_consumed: bool = False


@dataclass(slots=True)
class _KineticNativeSampleReleaseCommitAuthorization:
    """Private two-state ledger shared only by one exact commit plan."""

    authorized_after_receipt_consume: bool = False
    consumed: bool = False
    _seal: object = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class _KineticNativeSampleReleaseCommitPlan:
    """Preallocated exact assignments for the post-consume sample commit."""

    session_identity: int
    plan_identity: int
    pending_identity: int
    pending: KineticNativePendingSampleLaunchCompletion = field(repr=False)
    lifetime: KineticNativeSampleLaunchLifetime = field(repr=False)
    state: _BlockExecutionState = field(repr=False)
    completion: KineticNativeSampleLaunchCompletionReceipt = field(repr=False)
    next_grad_node_chart_signature: tuple[object, ...] = field(repr=False)
    next_loss_signature: tuple[object, ...] = field(repr=False)
    next_state_prepare_count: int
    next_state_launch_count: int
    next_state_fence_count: int
    next_state_streamed_count: int
    next_sample_manifest_digest: str
    next_first_flat_sample_index: int
    next_last_flat_sample_index: int
    next_native_prepare_count: int
    next_native_launch_count: int
    next_native_fence_count: int
    next_streamed_count: int
    _authorization: _KineticNativeSampleReleaseCommitAuthorization = field(
        repr=False,
        compare=False,
    )
    _seal: object = field(default=None, repr=False)

    def __init_subclass__(cls, **kwargs: Any) -> NoReturn:
        raise TypeError("sample release commit plans cannot be subclassed")


@dataclass(frozen=True)
class _KineticNativeAbortReleaseCommitPlan:
    """Exact references prevalidated before sealed receipt consumption."""

    session_identity: int
    outstanding_sample_lifetime: KineticNativeSampleLaunchLifetime | None = field(
        repr=False
    )
    state_execution_pairs: tuple[
        tuple[
            _BlockExecutionState,
            KineticNativeFullGeometryVJPExecution | None,
        ],
        ...,
    ] = field(repr=False)
    fused_full_geometry_execution_receipt: (
        KineticNativeFusedFullGeometryVJPExecutionReceipt | None
    ) = field(repr=False)
    failed_fused_full_geometry_transaction: (
        KineticNativeEqualRankFusedDirectFullVjpV1Transaction | None
    ) = field(repr=False)
    failed_sample_completion_receipt: (
        PaperKineticCompletionFenceReceipt | None
    ) = field(repr=False)
    pending_sample_completion: (
        KineticNativePendingSampleLaunchCompletion | None
    ) = field(repr=False)
    _seal: object = field(default=None, repr=False)


@dataclass
class KineticNativeMaterialStepSession:
    """Mutable, fail-closed launch ledger for exactly one optimizer step."""

    executor: KineticNativeMaterialStepExecutor = field(repr=False)
    step_generation_id: str
    requested_observation_count: int
    generation_id: str
    _executor_identity: int = field(repr=False)
    _executor_generation_id: str = field(repr=False)
    _states: dict[str, _BlockExecutionState] = field(default_factory=dict, repr=False)
    _native_node_forward_launch_count: int = 0
    _native_sample_prepare_count: int = 0
    _native_sample_launch_count: int = 0
    _native_sample_completion_fence_count: int = 0
    _streamed_sample_count: int = 0
    _native_material_vjp_launch_count: int = 0
    _native_full_geometry_vjp_launch_count: int = 0
    _native_full_geometry_fenced_reduction_count: int = 0
    _native_fused_full_geometry_vjp_launch_count: int = 0
    _native_fused_full_geometry_transaction_count: int = 0
    _native_fused_full_geometry_completion_fence_count: int = 0
    _fused_union_v2_transaction_generation_id: str = ""
    _fused_union_v2_retained_output_tensor_bytes: int = 0
    _fused_union_v2_completion_fence_provenance: str = ""
    _native_length_bar_tensor_bytes: int = 0
    _fused_full_geometry_execution_receipt: (
        KineticNativeFusedFullGeometryVJPExecutionReceipt | None
    ) = field(default=None, repr=False)
    _failed_fused_full_geometry_transaction: (
        KineticNativeEqualRankFusedDirectFullVjpV1Transaction | None
    ) = field(default=None, repr=False)
    _failed_fused_full_geometry_error: BaseException | None = field(
        default=None,
        repr=False,
    )
    _failed_fused_full_geometry_fence_provenance: str | None = None
    _fused_full_geometry_completion_unknown: bool = False
    _fused_transaction_in_progress: bool = False
    _reverse_mode: str | None = None
    _global_loss_element_count: int | None = None
    _loss_scale: float | None = None
    _loss_normalization_id: str | None = None
    _node_bar_owner_by_storage: dict[tuple[object, ...], str] = field(
        default_factory=dict,
        repr=False,
    )
    _loss_owner_by_storage: dict[tuple[object, ...], str] = field(
        default_factory=dict,
        repr=False,
    )
    _outstanding_sample_lifetime: KineticNativeSampleLaunchLifetime | None = field(
        default=None,
        repr=False,
    )
    _pending_sample_completion: (
        KineticNativePendingSampleLaunchCompletion | None
    ) = field(default=None, repr=False)
    _sample_settlement_in_progress: bool = False
    _sample_completion_unknown: bool = False
    _failed_sample_completion_error: BaseException | None = field(
        default=None,
        repr=False,
    )
    _failed_sample_completion_sealed_receipt: (
        PaperKineticCompletionFenceReceipt | None
    ) = field(default=None, repr=False)
    _failed_sample_completion_launch_generation_digest: str | None = None
    _failed_sample_completion_fence_succeeded: bool = False
    _sample_completion_fence_provenance: str | None = None
    _maximum_simultaneous_sample_lifetime_count: int = 0
    _sealed: bool = False
    _failed: bool = False
    _abort_fence_in_progress: bool = False
    _abort_release_completed: bool = False
    _abort_completion_fence_call_count: int = 0
    _abort_completion_fence_provenance: str | None = None
    _seal: object = field(default=None, repr=False)

    @torch.no_grad()
    def launch_node_forward(
        self,
        runtime: KineticNativeEqualRankRuntimeBlock,
        compact_site_rgba_f32: torch.Tensor,
    ) -> KineticNativeMaterialStepWorldToken:
        """Launch the sole node forward for one active native block."""

        with self._poison_on_error():
            if self._fused_full_geometry_execution_receipt is not None:
                raise ValueError(
                    "native blocks cannot launch after fused manifest acceptance"
                )
            binding = self.executor.binding_for_runtime(runtime)
            binding.assert_current()
            if binding.runtime_generation_id in self._states:
                raise ValueError("native block node forward was already launched this step")
            _require_tensor(
                compact_site_rgba_f32,
                name="compact_site_rgba_f32",
                device=self.executor.device,
                dtype=torch.float32,
                shape=(runtime.compact_site_count, 4),
            )
            world = refresh_kinetic_native_equal_rank_world(
                runtime,
                compact_site_rgba_f32,
            )
            token_generation = _world_token_generation_id(self, binding, world)
            token = KineticNativeMaterialStepWorldToken(
                world=world,
                runtime_binding=binding,
                session_generation_id=self.generation_id,
                runtime_generation_id=binding.runtime_generation_id,
                native_block_generation_digest=binding.native_block_generation_digest,
                world_generation_id=world.generation_id,
                generation_id=token_generation,
                _session_identity=id(self),
                _world_identity=id(world),
                _seal=_WORLD_TOKEN_SEAL,
            )
            token.assert_current(self)
            self._states[binding.runtime_generation_id] = _BlockExecutionState(
                runtime_binding=binding,
                runtime_generation_id=binding.runtime_generation_id,
                sampler_generation_digest=binding.sampler_generation_digest,
                native_block_generation_digest=binding.native_block_generation_digest,
                world_generation_id=world.generation_id,
                token=token,
                token_identity=id(token),
                sample_manifest_digest=_digest_parts(
                    EXECUTOR_PROVENANCE,
                    "block-sample-manifest",
                    self.generation_id,
                    binding.runtime_generation_id,
                    binding.sampler_generation_digest,
                    binding.native_block_generation_digest,
                ),
            )
            self._native_node_forward_launch_count += 1
        return token

    @torch.no_grad()
    def launch_node_forward_into(
        self,
        lifetime: KineticNativeNodeForwardIntoLifetime,
    ) -> KineticNativeMaterialStepWorldToken:
        """Launch the lifetime-safe caller-owned-output node forward."""

        if not isinstance(lifetime, KineticNativeNodeForwardIntoLifetime):
            raise TypeError(
                "lifetime must be KineticNativeNodeForwardIntoLifetime"
            )
        lifetime.assert_retained(self)
        if lifetime.phase != "output_published":
            raise ValueError("native forward-into output was not published prelaunch")
        runtime = lifetime.runtime
        compact_site_rgba_f32 = lifetime.compact_site_rgba_f32
        node_chart_out_f32 = lifetime.node_chart_out_f32
        if (
            not isinstance(runtime, KineticNativeEqualRankRuntimeBlock)
            or not isinstance(compact_site_rgba_f32, torch.Tensor)
            or not isinstance(node_chart_out_f32, torch.Tensor)
        ):
            raise ValueError("native forward-into lifetime lost a launch root")
        with self._poison_on_error():
            if self._fused_full_geometry_execution_receipt is not None:
                raise ValueError(
                    "native blocks cannot launch after fused manifest acceptance"
                )
            binding = self.executor.binding_for_runtime(runtime)
            binding.assert_current()
            if binding.runtime_generation_id in self._states:
                raise ValueError("native block node forward was already launched this step")
            _require_tensor(
                compact_site_rgba_f32,
                name="compact_site_rgba_f32",
                device=self.executor.device,
                dtype=torch.float32,
                shape=(runtime.compact_site_count, 4),
            )
            world = refresh_kinetic_native_equal_rank_world_into(
                runtime,
                compact_site_rgba_f32,
                node_chart_out_f32,
            )
            lifetime.publish_world(world)
            token_generation = _world_token_generation_id(self, binding, world)
            token = KineticNativeMaterialStepWorldToken(
                world=world,
                runtime_binding=binding,
                session_generation_id=self.generation_id,
                runtime_generation_id=binding.runtime_generation_id,
                native_block_generation_digest=(
                    binding.native_block_generation_digest
                ),
                world_generation_id=world.generation_id,
                generation_id=token_generation,
                _session_identity=id(self),
                _world_identity=id(world),
                _seal=_WORLD_TOKEN_SEAL,
            )
            lifetime.publish_token(token)
            token.assert_current(self)
            self._states[binding.runtime_generation_id] = _BlockExecutionState(
                runtime_binding=binding,
                runtime_generation_id=binding.runtime_generation_id,
                sampler_generation_digest=binding.sampler_generation_digest,
                native_block_generation_digest=(
                    binding.native_block_generation_digest
                ),
                world_generation_id=world.generation_id,
                token=token,
                token_identity=id(token),
                sample_manifest_digest=_digest_parts(
                    EXECUTOR_PROVENANCE,
                    "block-sample-manifest",
                    self.generation_id,
                    binding.runtime_generation_id,
                    binding.sampler_generation_digest,
                    binding.native_block_generation_digest,
                ),
            )
            self._native_node_forward_launch_count += 1
            lifetime.phase = "active"
            lifetime.assert_retained(self)
        return token

    @torch.no_grad()
    def launch_sample_accumulate(
        self,
        token: KineticNativeMaterialStepWorldToken,
        sample_block: PaperKineticRowRaggedSampleBlock,
        *,
        sampler: PaperKineticRowRaggedSampler,
        background_rgb_f32: torch.Tensor,
        loss_f32: torch.Tensor,
        grad_node_chart_f32: torch.Tensor,
        cone_diagnostic_i32: torch.Tensor,
        cone_tolerance: float = 1.0e-5,
    ) -> KineticNativeSampleLaunchLifetime:
        """Enqueue one sample reduction while retaining its exact async roots.

        Coverage is not committed here.  The returned carrier is also owned by
        the session and must be passed to :meth:`settle_sample_accumulate`,
        which performs the sole completion fence before releasing its roots.
        """

        with self._poison_on_error():
            state = self._state_for_token(token)
            if (
                state.material_vjp_launch_count
                or state.full_geometry_vjp_launch_count
                or state.fused_full_geometry_vjp_launch_count
            ):
                raise ValueError("native samples cannot launch after the block VJP")
            binding = token.runtime_binding
            binding.assert_sampler_current(sampler)
            sample_block.assert_cold_current(sampler)
            if (
                sample_block.native_block_generation_digest
                != binding.native_block_generation_digest
                or sample_block.sampler_generation_digest
                != binding.sampler_generation_digest
                or sample_block.view_index != binding.sampler_view_index
                or sample_block.row_count != token.world.row_count
                or sample_block.node_count != token.world.node_count
            ):
                raise ValueError("ragged sample block belongs to a different native world")
            _require_tensor(
                background_rgb_f32,
                name="background_rgb_f32",
                device=self.executor.device,
                dtype=torch.float32,
                shape=(3,),
            )
            _require_tensor(
                loss_f32,
                name="loss_f32",
                device=self.executor.device,
                dtype=torch.float32,
                shape=(1,),
            )
            _require_tensor(
                grad_node_chart_f32,
                name="grad_node_chart_f32",
                device=self.executor.device,
                dtype=torch.float32,
                shape=(token.world.row_count, token.world.node_count, 4),
            )
            _require_tensor(
                cone_diagnostic_i32,
                name="cone_diagnostic_i32",
                device=self.executor.device,
                dtype=torch.int32,
                shape=(3,),
            )
            self._bind_or_validate_loss_normalization(sample_block)
            self._bind_or_validate_node_bar(state, grad_node_chart_f32)
            self._bind_or_validate_loss(state, loss_f32)
            flat_sample_indices = tuple(
                int(value)
                for value in sample_block.flat_sample_index_i64.tolist()
            )
            if (
                len(flat_sample_indices) != sample_block.sample_count
                or any(
                    right <= left
                    for left, right in zip(
                        flat_sample_indices,
                        flat_sample_indices[1:],
                        strict=False,
                    )
                )
                or (
                    state.last_flat_sample_index is not None
                    and flat_sample_indices[0] <= state.last_flat_sample_index
                )
            ):
                raise ValueError(
                    "native block sample identities must be unique and canonical"
                )
            next_sample_manifest_digest = _digest_parts(
                EXECUTOR_PROVENANCE,
                "block-sample-launch",
                state.sample_manifest_digest,
                sample_block.generation_digest,
                sample_block.dispatch_generation_digest,
                flat_sample_indices,
            )
            read_only_signatures = tuple(
                _tensor_signature(tensor)
                for tensor in (
                    token.world.node_chart_f32,
                    sample_block.sample_to_node_f32,
                    sample_block.target_rgb_f32,
                    background_rgb_f32,
                )
            )
            lifetime = KineticNativeSampleLaunchLifetime(
                prepared_payload=None,
                sample_block=sample_block,
                world_token=token,
                background_rgb_f32=background_rgb_f32,
                loss_f32=loss_f32,
                grad_node_chart_f32=grad_node_chart_f32,
                cone_diagnostic_i32=cone_diagnostic_i32,
                session_generation_id=self.generation_id,
                runtime_generation_id=state.runtime_generation_id,
                sampler_generation_digest=state.sampler_generation_digest,
                native_block_generation_digest=(
                    state.native_block_generation_digest
                ),
                sample_block_generation_digest=sample_block.generation_digest,
                sample_dispatch_generation_digest=(
                    sample_block.dispatch_generation_digest
                ),
                prior_sample_manifest_digest=state.sample_manifest_digest,
                next_sample_manifest_digest=next_sample_manifest_digest,
                flat_sample_identity_digest=_digest_parts(
                    EXECUTOR_PROVENANCE,
                    "flat-sample-identities",
                    flat_sample_indices,
                ),
                sample_count=sample_block.sample_count,
                first_flat_sample_index=flat_sample_indices[0],
                last_flat_sample_index=flat_sample_indices[-1],
                read_only_tensor_signatures=read_only_signatures,
                prepared_payload_signature=(),
                writable_tensor_signatures_after_launch=(),
                generation_digest="",
                _session_identity=id(self),
                _block_state_identity=id(state),
                _prepared_payload_identity=0,
                _sample_block_identity=id(sample_block),
                _world_token_identity=id(token),
                _seal=_SAMPLE_LAUNCH_LIFETIME_SEAL,
            )
            self._outstanding_sample_lifetime = lifetime
            self._maximum_simultaneous_sample_lifetime_count = max(
                self._maximum_simultaneous_sample_lifetime_count,
                1,
            )
            prepared = getattr(self.executor.native_ops, SAMPLE_PREPARE_OP_NAME)(
                token.world.node_chart_f32,
                sample_block.sample_row_i32,
                sample_block.sample_to_node_f32,
                sample_block.target_rgb_f32,
                background_rgb_f32,
                loss_scale=sample_block.loss_scale,
                cone_tolerance=cone_tolerance,
            )
            lifetime.prepared_payload = prepared
            lifetime._prepared_payload_identity = id(prepared)
            lifetime.prepared_payload_signature = (
                _prepared_sample_payload_signature(prepared)
            )
            lifetime.phase = "prepared"
            if read_only_signatures != tuple(
                _tensor_signature(tensor)
                for tensor in (
                    token.world.node_chart_f32,
                    sample_block.sample_to_node_f32,
                    sample_block.target_rgb_f32,
                    background_rgb_f32,
                )
            ):
                raise ValueError("native sample prepare mutated a read-only input")
            returned = getattr(self.executor.native_ops, SAMPLE_LAUNCH_OP_NAME)(
                prepared,
                loss_f32,
                grad_node_chart_f32,
                cone_diagnostic_i32,
            )
            if returned is not None:
                raise TypeError("loss-only native sample launch must return None")
            if read_only_signatures != tuple(
                _tensor_signature(tensor)
                for tensor in (
                    token.world.node_chart_f32,
                    sample_block.sample_to_node_f32,
                    sample_block.target_rgb_f32,
                    background_rgb_f32,
                )
            ):
                raise ValueError("native sample launch mutated a read-only input")
            sample_block.assert_warm_layout()
            token.world.assert_current()
            lifetime.writable_tensor_signatures_after_launch = tuple(
                _tensor_signature(tensor)
                for tensor in (
                    loss_f32,
                    grad_node_chart_f32,
                    cone_diagnostic_i32,
                )
            )
            lifetime.phase = "launched"
            lifetime.generation_digest = _sample_launch_lifetime_digest(
                lifetime
            )
            lifetime.assert_current(self)
        return lifetime

    @torch.no_grad()
    def settle_sample_accumulate(
        self,
        lifetime: KineticNativeSampleLaunchLifetime,
        *,
        device_completion_fence: Callable[[], None] | None = None,
        device_completion_fence_provenance: str | None = None,
        sealed_completion_fence: PaperKineticSealedCompletionFence | None = None,
        sealed_completion_launch_epoch: PaperKineticCompletionLaunchEpoch | None = None,
    ) -> (
        KineticNativeSampleLaunchCompletionReceipt
        | KineticNativePendingSampleLaunchCompletion
    ):
        """Fence and validate; sealed callers defer one composite commit."""

        fence_succeeded = False
        sealed_completion_receipt: PaperKineticCompletionFenceReceipt | None = None
        with self._poison_on_error(allow_outstanding_sample=True):
            if not isinstance(lifetime, KineticNativeSampleLaunchLifetime):
                raise TypeError(
                    "lifetime must be KineticNativeSampleLaunchLifetime"
                )
            if self._outstanding_sample_lifetime is not lifetime:
                raise ValueError("sample launch lifetime is foreign or already settled")
            if self._pending_sample_completion is not None:
                raise RuntimeError(
                    "pending sample completion must commit before settlement"
                )
            sealed_authority = (
                type(sealed_completion_fence)
                is PaperKineticSealedCompletionFence
            )
            legacy_authority = callable(device_completion_fence)
            if sealed_authority == legacy_authority:
                raise TypeError(
                    "sample settlement requires exactly one completion authority"
                )
            if sealed_authority:
                if device_completion_fence_provenance is not None:
                    raise ValueError(
                        "sealed sample settlement forbids caller provenance"
                    )
                sealed_completion_fence.assert_current(
                    native_ops=self.executor.native_ops,
                    device=lifetime.sample_block.device,
                )
                if (
                    type(sealed_completion_launch_epoch)
                    is not PaperKineticCompletionLaunchEpoch
                    or sealed_completion_launch_epoch.stage
                    != "sample-completion"
                    or sealed_completion_fence.registered_launch_epoch
                    is not sealed_completion_launch_epoch
                    or type(sealed_completion_launch_epoch.subject_binding)
                    is not PaperKineticCompletionSubjectBinding
                ):
                    raise ValueError(
                        "sealed sample settlement requires its pre-launch epoch"
                    )
                authority_provenance = CAPABILITY_PROVENANCE
            else:
                if sealed_completion_launch_epoch is not None:
                    raise ValueError("legacy sample settlement forbids a launch epoch")
                if (
                    not isinstance(device_completion_fence_provenance, str)
                    or not device_completion_fence_provenance.strip()
                ):
                    raise ValueError(
                        "device_completion_fence_provenance must be nonempty"
                    )
                authority_provenance = device_completion_fence_provenance
            if self._sample_completion_unknown:
                raise RuntimeError(
                    "sample completion is unknown; restart is required"
                ) from self._failed_sample_completion_error
            if (
                self._sample_completion_fence_provenance is not None
                and self._sample_completion_fence_provenance
                != authority_provenance
            ):
                raise ValueError(
                    "sample completion fence provenance changed within the step"
                )
            lifetime.assert_current(self)
            if legacy_authority and (
                self.executor.device.type != "cpu"
                or lifetime.sample_block.device.type != "cpu"
            ):
                raise RuntimeError(
                    "callback-authorized sample settlement is CPU-only; "
                    "accelerator roots require exact subject-bound authority"
                )
            state = self._states[lifetime.runtime_generation_id]
            pre_fence_snapshot = _sample_settlement_snapshot(
                self,
                state,
                lifetime,
            )
            lifetime.completion_fence_attempt_count += 1
            self._sample_settlement_in_progress = True
            try:
                if sealed_authority:
                    expected_fence_sequence = (
                        sealed_completion_launch_epoch.launch_epoch_sequence
                    )
                    sealed_completion_receipt = sealed_completion_fence.fence(
                        sealed_completion_launch_epoch
                    )
                else:
                    returned = device_completion_fence()
                    if returned is not None:
                        raise TypeError("device_completion_fence must return None")
                fence_succeeded = True
            except BaseException as error:
                # A completion call that did not return a receipt is
                # intrinsically unknown.  Do not rely on the authority to
                # have published its own poison bit: an exception can escape
                # at any point inside a backend/capability implementation,
                # including after the device was synchronized but before its
                # Python ledger was updated.
                lifetime.phase = "completion_unknown"
                lifetime.completion_unknown = True
                self._sample_completion_unknown = True
                self._failed_sample_completion_error = error
                raise
            finally:
                self._sample_settlement_in_progress = False

            # Capture the callback-visible ledger before publishing the
            # executor's own successful-settlement bookkeeping below.  The
            # latter deliberately changes receipt/provenance fields and must
            # not be mistaken for a callback mutation.
            post_fence_snapshot: tuple[object, ...] | None = None
            post_fence_snapshot_error: BaseException | None = None
            try:
                post_fence_snapshot = _sample_settlement_snapshot(
                    self,
                    state,
                    lifetime,
                )
            except BaseException as error:
                post_fence_snapshot_error = error
            if sealed_authority:
                # Publish the exact known-completion receipt before any
                # subsequent validation.  A pre-commit rejection therefore
                # preserves the sole authority and every launch root for the
                # outer composite abort; it never invents a post-hoc epoch.
                self._failed_sample_completion_fence_succeeded = True
                self._sample_completion_fence_provenance = CAPABILITY_PROVENANCE
                self._failed_sample_completion_sealed_receipt = (
                    sealed_completion_receipt
                )
                self._failed_sample_completion_launch_generation_digest = (
                    sealed_completion_launch_epoch.launch_generation_digest
                )

            try:
                if post_fence_snapshot_error is not None:
                    raise post_fence_snapshot_error
                lifetime.assert_current(self)
                if pre_fence_snapshot != post_fence_snapshot:
                    raise ValueError(
                        "sample completion callback mutated the session ledger"
                    )
                if lifetime.completion_fence_attempt_count != 1:
                    raise ArithmeticError(
                        "sample lifetime did not receive exactly one completion fence"
                    )
                provisional_receipt = KineticNativeSampleLaunchCompletionReceipt(
                    session_generation_id=self.generation_id,
                    runtime_generation_id=state.runtime_generation_id,
                    native_block_generation_digest=(
                        state.native_block_generation_digest
                    ),
                    sample_lifetime_generation_digest=(
                        lifetime.generation_digest
                    ),
                    sample_manifest_digest=(
                        lifetime.next_sample_manifest_digest
                    ),
                    sample_count=lifetime.sample_count,
                    first_flat_sample_index=(
                        lifetime.first_flat_sample_index
                    ),
                    last_flat_sample_index=lifetime.last_flat_sample_index,
                    device_completion_fence_provenance=(
                        authority_provenance
                    ),
                    generation_digest="",
                    sealed_completion_capability_generation_digest=(
                        ""
                        if sealed_completion_receipt is None
                        else sealed_completion_receipt.capability_generation_digest
                    ),
                    sealed_completion_receipt_generation_digest=(
                        ""
                        if sealed_completion_receipt is None
                        else sealed_completion_receipt.generation_digest
                    ),
                    sealed_completion_fence_sequence=(
                        0
                        if sealed_completion_receipt is None
                        else sealed_completion_receipt.fence_sequence
                    ),
                    sealed_completion_scope=(
                        ""
                        if sealed_completion_receipt is None
                        else sealed_completion_receipt.completion_scope
                    ),
                    sealed_completion_normalized_device=(
                        ""
                        if sealed_completion_receipt is None
                        else sealed_completion_receipt.normalized_device
                    ),
                    sealed_completion_launch_generation_digest=(
                        ""
                        if sealed_completion_receipt is None
                        else sealed_completion_launch_epoch.launch_generation_digest
                    ),
                    sealed_completion_receipt=sealed_completion_receipt,
                    _seal=(
                        None
                        if sealed_authority
                        else _SAMPLE_COMPLETION_RECEIPT_SEAL
                    ),
                )
                receipt = replace(
                    provisional_receipt,
                    generation_digest=_sample_completion_receipt_digest(
                        provisional_receipt
                    ),
                )
                if not sealed_authority:
                    receipt.assert_current()
                next_grad_node_chart_signature = _tensor_signature(
                    lifetime.grad_node_chart_f32
                )
                next_loss_signature = _tensor_signature(lifetime.loss_f32)
                next_state_prepare_count = state.native_sample_prepare_count + 1
                next_state_launch_count = state.native_sample_launch_count + 1
                next_state_fence_count = (
                    state.native_sample_completion_fence_count + 1
                )
                next_state_streamed_count = (
                    state.streamed_sample_count + lifetime.sample_count
                )
                next_sample_manifest_digest = (
                    lifetime.next_sample_manifest_digest
                )
                next_first_flat_sample_index = (
                    lifetime.first_flat_sample_index
                    if state.first_flat_sample_index is None
                    else state.first_flat_sample_index
                )
                next_native_prepare_count = self._native_sample_prepare_count + 1
                next_native_launch_count = self._native_sample_launch_count + 1
                next_native_fence_count = (
                    self._native_sample_completion_fence_count + 1
                )
                next_streamed_count = self._streamed_sample_count + lifetime.sample_count
                next_last_flat_sample_index = lifetime.last_flat_sample_index

                if sealed_authority:
                    subject_binding = (
                        sealed_completion_launch_epoch.subject_binding
                    )
                    sealed_completion_receipt.assert_for(
                        sealed_completion_fence,
                        stage="sample-completion",
                        launch_generation_digest=(
                            sealed_completion_launch_epoch.launch_generation_digest
                        ),
                        fence_sequence=expected_fence_sequence,
                    )
                    if (
                        sealed_completion_receipt.subject_binding
                        is not subject_binding
                    ):
                        raise ValueError(
                            "sealed sample receipt lost its prelaunch subject"
                        )
                    pending = KineticNativePendingSampleLaunchCompletion(
                        session_generation_id=self.generation_id,
                        runtime_generation_id=state.runtime_generation_id,
                        sample_lifetime_generation_digest=(
                            lifetime.generation_digest
                        ),
                        pending_identity=0,
                        capability_identity=id(sealed_completion_fence),
                        capability_generation_digest=(
                            sealed_completion_fence.generation_digest
                        ),
                        capability_owner_generation_digest=(
                            sealed_completion_fence.owner_generation_digest
                        ),
                        subject_binding_identity=id(subject_binding),
                        subject_binding_generation_digest=(
                            subject_binding.generation_digest
                        ),
                        subject_identity=subject_binding.subject_identity,
                        subject_generation_digest=(
                            subject_binding.subject_generation_digest
                        ),
                        launch_epoch_identity=id(
                            sealed_completion_launch_epoch
                        ),
                        launch_epoch_generation_digest=(
                            sealed_completion_launch_epoch.generation_digest
                        ),
                        launch_stage=sealed_completion_launch_epoch.stage,
                        launch_generation_digest=(
                            sealed_completion_launch_epoch.launch_generation_digest
                        ),
                        launch_epoch_sequence=expected_fence_sequence,
                        receipt_identity=id(sealed_completion_receipt),
                        receipt_generation_digest=(
                            sealed_completion_receipt.generation_digest
                        ),
                        completion_receipt_identity=id(receipt),
                        completion_receipt_generation_digest=(
                            receipt.generation_digest
                        ),
                        next_grad_node_chart_signature=(
                            next_grad_node_chart_signature
                        ),
                        next_loss_signature=next_loss_signature,
                        next_state_prepare_count=next_state_prepare_count,
                        next_state_launch_count=next_state_launch_count,
                        next_state_fence_count=next_state_fence_count,
                        next_state_streamed_count=next_state_streamed_count,
                        next_sample_manifest_digest=(
                            next_sample_manifest_digest
                        ),
                        next_first_flat_sample_index=(
                            next_first_flat_sample_index
                        ),
                        next_last_flat_sample_index=(
                            next_last_flat_sample_index
                        ),
                        next_native_prepare_count=next_native_prepare_count,
                        next_native_launch_count=next_native_launch_count,
                        next_native_fence_count=next_native_fence_count,
                        next_streamed_count=next_streamed_count,
                        generation_digest="",
                        _session_identity=id(self),
                        _sample_lifetime_identity=id(lifetime),
                        _block_state_identity=id(state),
                        _sealed_completion_fence=sealed_completion_fence,
                        _subject_binding=subject_binding,
                        _launch_epoch=sealed_completion_launch_epoch,
                        _sealed_completion_receipt=(
                            sealed_completion_receipt
                        ),
                        _completion_receipt=receipt,
                        _sample_lifetime=lifetime,
                        _block_state=state,
                        _seal=_PENDING_SAMPLE_COMPLETION_SEAL,
                    )
                    pending.pending_identity = id(pending)
                    pending.generation_digest = (
                        _pending_sample_completion_digest(pending)
                    )
                    pending._assert_exact_relation(
                        self,
                        sealed_completion_fence,
                        subject=None,
                        subject_required=False,
                        require_unconsumed=True,
                        require_installed=False,
                    )
                    self._pending_sample_completion = pending
                    return pending

                # Legacy callbacks remain an immediate synchronous commit.
                state.grad_node_chart_signature = next_grad_node_chart_signature
                state.loss_signature = next_loss_signature
                state.native_sample_prepare_count = next_state_prepare_count
                state.native_sample_launch_count = next_state_launch_count
                state.native_sample_completion_fence_count = next_state_fence_count
                state.streamed_sample_count = next_state_streamed_count
                state.sample_manifest_digest = next_sample_manifest_digest
                state.first_flat_sample_index = next_first_flat_sample_index
                state.last_flat_sample_index = next_last_flat_sample_index
                self._native_sample_prepare_count = next_native_prepare_count
                self._native_sample_launch_count = next_native_launch_count
                self._native_sample_completion_fence_count = next_native_fence_count
                self._streamed_sample_count = next_streamed_count
                self._sample_completion_fence_provenance = authority_provenance
                self._failed_sample_completion_fence_succeeded = False
                self._failed_sample_completion_sealed_receipt = None
                self._failed_sample_completion_launch_generation_digest = None
                lifetime.prepared_payload = None
                lifetime.sample_block = None
                lifetime.world_token = None
                lifetime.background_rgb_f32 = None
                lifetime.loss_f32 = None
                lifetime.grad_node_chart_f32 = None
                lifetime.cone_diagnostic_i32 = None
                lifetime.phase = "released"
                lifetime.consumed = True
                self._outstanding_sample_lifetime = None
                return receipt
            except BaseException:
                # Completion is known.  Even if provenance/layout validation
                # rejects the request, the exact receipt and every root remain
                # published for one outer consume-then-commit abort.
                if (
                    fence_succeeded
                    and not sealed_authority
                    and self._outstanding_sample_lifetime is lifetime
                ):
                    self._failed_sample_completion_fence_succeeded = True
                    self._sample_completion_fence_provenance = (
                        authority_provenance
                    )
                    self._failed_sample_completion_sealed_receipt = (
                        sealed_completion_receipt
                    )
                    self._failed_sample_completion_launch_generation_digest = (
                        None
                    )
                    lifetime._release_roots()
                    self._outstanding_sample_lifetime = None
                raise

    def assert_pending_sample_accumulate_releasable(
        self,
        pending: KineticNativePendingSampleLaunchCompletion,
        capability: PaperKineticSealedCompletionFence,
        *,
        subject: Any,
    ) -> None:
        """Prevalidate executor roots before the outer composite consumes."""

        if type(pending) is not KineticNativePendingSampleLaunchCompletion:
            raise TypeError("sample commit requires the exact pending completion")
        pending.assert_exact_sealed_receipt_relation(
            self,
            capability,
            subject=subject,
            require_unconsumed=True,
        )

    def commit_sample_accumulate_after_consumed_sealed_receipt(
        self,
        commit_plan: _KineticNativeSampleReleaseCommitPlan,
    ) -> KineticNativeSampleLaunchCompletionReceipt:
        """Assignment-only commit of a pre-consume exact release plan."""

        if type(commit_plan) is not _KineticNativeSampleReleaseCommitPlan:
            raise TypeError("sample release commit plan has a foreign exact type")
        authorization = commit_plan._authorization
        if (
            commit_plan._seal is not _SAMPLE_RELEASE_COMMIT_PLAN_SEAL
            or type(authorization)
            is not _KineticNativeSampleReleaseCommitAuthorization
            or authorization._seal
            is not _SAMPLE_RELEASE_COMMIT_AUTHORIZATION_SEAL
            or commit_plan.plan_identity < 1
            or id(commit_plan) != commit_plan.plan_identity
            or commit_plan.session_identity != id(self)
            or commit_plan.pending_identity != id(commit_plan.pending)
            or self._pending_sample_completion is not commit_plan.pending
            or not authorization.authorized_after_receipt_consume
            or authorization.consumed
        ):
            raise ValueError(
                "sample release commit plan is foreign, unauthorized, or consumed"
            )
        authorization.consumed = True
        pending = commit_plan.pending
        lifetime = commit_plan.lifetime
        state = commit_plan.state
        completion = commit_plan.completion
        state.grad_node_chart_signature = (
            commit_plan.next_grad_node_chart_signature
        )
        state.loss_signature = commit_plan.next_loss_signature
        state.native_sample_prepare_count = commit_plan.next_state_prepare_count
        state.native_sample_launch_count = commit_plan.next_state_launch_count
        state.native_sample_completion_fence_count = (
            commit_plan.next_state_fence_count
        )
        state.streamed_sample_count = commit_plan.next_state_streamed_count
        state.sample_manifest_digest = commit_plan.next_sample_manifest_digest
        state.first_flat_sample_index = commit_plan.next_first_flat_sample_index
        state.last_flat_sample_index = commit_plan.next_last_flat_sample_index
        self._native_sample_prepare_count = commit_plan.next_native_prepare_count
        self._native_sample_launch_count = commit_plan.next_native_launch_count
        self._native_sample_completion_fence_count = (
            commit_plan.next_native_fence_count
        )
        self._streamed_sample_count = commit_plan.next_streamed_count
        self._sample_completion_fence_provenance = CAPABILITY_PROVENANCE
        self._failed_sample_completion_fence_succeeded = False
        self._failed_sample_completion_sealed_receipt = None
        self._failed_sample_completion_launch_generation_digest = None
        lifetime.prepared_payload = None
        lifetime.sample_block = None
        lifetime.world_token = None
        lifetime.background_rgb_f32 = None
        lifetime.loss_f32 = None
        lifetime.grad_node_chart_f32 = None
        lifetime.cone_diagnostic_i32 = None
        lifetime.phase = "released"
        lifetime.consumed = True
        self._outstanding_sample_lifetime = None
        self._pending_sample_completion = None
        pending._sample_lifetime = None
        pending._block_state = None
        pending._sealed_completion_fence = None
        pending._launch_epoch = None
        pending.phase = "committed"
        # The provisional frozen receipt is identity-bound into ``pending``;
        # seal that exact object only after the one-shot outer authorization
        # commits instead of replacing it with a new dataclass instance.
        object.__setattr__(completion, "_seal", _SAMPLE_COMPLETION_RECEIPT_SEAL)
        return completion

    @torch.no_grad()
    def launch_material_vjp(
        self,
        token: KineticNativeMaterialStepWorldToken,
        grad_node_chart_f32: torch.Tensor,
        *,
        compact_grad_site_rgba_f32: torch.Tensor,
        global_grad_site_rgba_f32: torch.Tensor | None = None,
    ) -> KineticNativeEqualRankMaterialVJPResult:
        """Launch the sole material-only ordered-word reverse for a block."""

        with self._poison_on_error():
            state = self._state_for_token(token)
            if state.native_sample_launch_count < 1:
                raise ValueError("native material VJP requires at least one sample reduction")
            if (
                state.material_vjp_launch_count
                or state.full_geometry_vjp_launch_count
                or state.fused_full_geometry_vjp_launch_count
            ):
                raise ValueError("native reverse was already launched for this block")
            self._bind_reverse_mode("material_only")
            self._bind_or_validate_node_bar(state, grad_node_chart_f32)
            result = execute_kinetic_native_equal_rank_material_node_vjp(
                token.world,
                grad_node_chart_f32,
                compact_grad_site_rgba_f32=compact_grad_site_rgba_f32,
                global_grad_site_rgba_f32=global_grad_site_rgba_f32,
            )
            state.grad_node_chart_signature = _tensor_signature(grad_node_chart_f32)
            state.material_vjp_launch_count = 1
            state.reverse_result_identity = id(result)
            state.token = None
            state.grad_node_chart_f32 = None
            self._native_material_vjp_launch_count += 1
        return result

    @torch.no_grad()
    def launch_full_geometry_vjp(
        self,
        token: KineticNativeMaterialStepWorldToken,
        grad_node_chart_f32: torch.Tensor,
        *,
        compact_grad_site_rgba_f32: torch.Tensor,
        global_grad_site_rgba_f32: torch.Tensor | None = None,
    ) -> KineticNativeFullGeometryVJPExecution:
        """Launch the sole material-plus-length reverse for one active block.

        The returned sealed result owns the bounded ``[J,W]`` physical-length
        bar. The caller must pass the resulting trusted geometry reduction back
        to this session before the executor releases the result; no frame or
        sample axis is introduced by this operation.
        """

        with self._poison_on_error():
            state = self._state_for_token(token)
            if state.native_sample_launch_count < 1:
                raise ValueError("native full-geometry VJP requires at least one sample reduction")
            if (
                state.material_vjp_launch_count
                or state.full_geometry_vjp_launch_count
                or state.fused_full_geometry_vjp_launch_count
            ):
                raise ValueError("native reverse was already launched for this block")
            self._bind_reverse_mode("full_geometry")
            self._bind_or_validate_node_bar(state, grad_node_chart_f32)
            result = execute_kinetic_native_equal_rank_node_vjp(
                token.world,
                grad_node_chart_f32,
                compact_grad_site_rgba_f32=compact_grad_site_rgba_f32,
                global_grad_site_rgba_f32=global_grad_site_rgba_f32,
            )
            length_bar_bytes = (
                result.grad_node_physical_length_f32.numel()
                * result.grad_node_physical_length_f32.element_size()
            )
            if length_bar_bytes < 1:
                raise ArithmeticError("native full-geometry VJP returned an empty length bar")
            state.grad_node_chart_signature = _tensor_signature(grad_node_chart_f32)
            state.full_geometry_vjp_launch_count = 1
            state.native_length_bar_tensor_bytes = length_bar_bytes
            state.reverse_result_identity = id(result)
            state.token = None
            state.grad_node_chart_f32 = None
            self._native_full_geometry_vjp_launch_count += 1
            self._native_length_bar_tensor_bytes += length_bar_bytes
            if (
                state.grad_node_chart_identity is None
                or state.grad_node_chart_signature is None
                or state.loss_identity is None
                or state.loss_signature is None
                or state.first_flat_sample_index is None
                or state.last_flat_sample_index is None
            ):
                raise ArithmeticError(
                    "full-geometry VJP lost its sealed node-bar/loss identity"
                )
            provisional = KineticNativeFullGeometryVJPExecution(
                native_vjp_result=result,
                session_generation_id=self.generation_id,
                runtime_generation_id=state.runtime_generation_id,
                native_block_generation_digest=(
                    state.native_block_generation_digest
                ),
                reduced_sample_chunk_count=state.native_sample_launch_count,
                reduced_sample_count=state.streamed_sample_count,
                sample_manifest_digest=state.sample_manifest_digest,
                first_flat_sample_index=state.first_flat_sample_index,
                last_flat_sample_index=state.last_flat_sample_index,
                node_bar_identity=state.grad_node_chart_identity,
                node_bar_signature=state.grad_node_chart_signature,
                loss_identity=state.loss_identity,
                loss_signature=state.loss_signature,
                native_length_bar_tensor_bytes=length_bar_bytes,
                generation_digest="",
                _session_identity=id(self),
                _native_vjp_result_identity=id(result),
                _seal=_FULL_GEOMETRY_EXECUTION_SEAL,
            )
            execution = replace(
                provisional,
                generation_digest=_full_geometry_execution_digest(provisional),
            )
            state.full_geometry_execution = execution
            state.full_geometry_execution_outstanding = True
            state.full_geometry_execution_consumed = False
            execution.assert_current(self)
        return execution

    @torch.no_grad()
    def execute_fused_full_geometry_vjp_transaction(
        self,
        prepared_blocks: Sequence[KineticNativeEqualRankFusedDirectFullVjpV1],
        *,
        max_output_scratch_tensor_bytes: int,
        device_completion_fence: Callable[[], None],
        device_completion_fence_provenance: str,
    ) -> KineticNativeFusedFullGeometryVJPExecutionReceipt:
        """Execute one fused reverse over the exact active session manifest.

        ``_states`` is the sole authority for which blocks are active.  The
        caller supplies cold-prepared fused tokens, but cannot omit, duplicate,
        or reorder an active block: their worlds and block generations must
        match the executor-binding order exactly.  Node cotangents are taken
        from the aliases bound by sample launches, not from a second
        caller-authored manifest.  The adapter owns fresh zero scratch and one
        completion fence.  Only after its accepted receipt is checked are all
        active states marked reversed together.
        """

        with self._poison_on_error():
            if self.executor.device.type != "cpu":
                raise RuntimeError(
                    "callback-authorized fused geometry settlement is CPU-only; "
                    "accelerator roots require exact subject-bound authority"
                )
            if self._fused_full_geometry_execution_receipt is not None:
                raise ValueError("fused full-geometry transaction already executed")
            ordered_states = self._ordered_active_states()
            if not ordered_states:
                raise ValueError(
                    "fused full-geometry transaction requires an active block"
                )
            blocks = tuple(prepared_blocks)
            if len(blocks) != len(ordered_states) or any(
                not isinstance(
                    block,
                    KineticNativeEqualRankFusedDirectFullVjpV1,
                )
                for block in blocks
            ):
                raise ValueError(
                    "fused prepared blocks must cover the exact active manifest"
                )
            if (
                self._native_sample_prepare_count
                != self._native_sample_launch_count
                or self._streamed_sample_count
                != self.requested_observation_count
            ):
                raise ValueError(
                    "fused reverse requires complete requested sample coverage"
                )
            node_bars: list[torch.Tensor] = []
            for block, state in zip(blocks, ordered_states, strict=True):
                token = state.token
                if not isinstance(token, KineticNativeMaterialStepWorldToken):
                    raise ValueError(
                        "fused active state no longer owns its exact world token"
                    )
                token.assert_current(self)
                block.assert_cold_current()
                node_bar = state.grad_node_chart_f32
                if not isinstance(node_bar, torch.Tensor):
                    raise ValueError(
                        "fused active state no longer owns its exact node cotangent"
                    )
                if (
                    state.native_sample_prepare_count
                    != state.native_sample_launch_count
                    or state.native_sample_launch_count < 1
                    or state.streamed_sample_count < 1
                    or len(state.sample_manifest_digest) != 64
                    or state.first_flat_sample_index is None
                    or state.last_flat_sample_index is None
                    or state.last_flat_sample_index < state.first_flat_sample_index
                    or state.material_vjp_launch_count != 0
                    or state.full_geometry_vjp_launch_count != 0
                    or state.fused_full_geometry_vjp_launch_count != 0
                    or state.reverse_result_identity is not None
                    or state.full_geometry_execution is not None
                    or state.full_geometry_execution_outstanding
                    or state.full_geometry_execution_consumed
                    or state.full_geometry_completion_receipt is not None
                    or state.token_identity != id(token)
                    or token.world is not block.world
                    or token.world.runtime is not state.runtime_binding.runtime
                    or token.world.generation_id != state.world_generation_id
                    or block.world.runtime.payload.block.generation_digest
                    != state.native_block_generation_digest
                    or id(node_bar) != state.grad_node_chart_identity
                    or _tensor_signature(node_bar)
                    != state.grad_node_chart_signature
                    or state.loss_f32 is None
                    or id(state.loss_f32) != state.loss_identity
                    or _tensor_signature(state.loss_f32) != state.loss_signature
                ):
                    raise ValueError(
                        "fused prepared block does not match its sampled active state"
                    )
                node_bars.append(node_bar)
            active_block_generation_ids = tuple(
                state.native_block_generation_digest for state in ordered_states
            )
            if tuple(
                block.world.runtime.payload.block.generation_digest
                for block in blocks
            ) != active_block_generation_ids:
                raise ValueError(
                    "fused prepared block order differs from the active manifest"
                )
            self._bind_reverse_mode("fused_full_geometry")
            transaction = (
                prepare_kinetic_native_equal_rank_fused_direct_full_vjp_transaction_v1(
                    blocks,
                    tuple(node_bars),
                    max_output_scratch_tensor_bytes=(
                        max_output_scratch_tensor_bytes
                    ),
                )
            )
            if not isinstance(
                transaction,
                KineticNativeEqualRankFusedDirectFullVjpV1Transaction,
            ):
                raise TypeError("fused transaction preparer returned a foreign token")
            transaction.assert_ready()
            if (
                transaction.active_block_generation_ids
                != active_block_generation_ids
                or transaction.prepared_block_identities
                != tuple(id(block) for block in blocks)
                or tuple(
                    id(tensor)
                    for tensor in transaction._state.grad_node_chart_f32_by_block
                )
                != tuple(
                    state.grad_node_chart_identity for state in ordered_states
                )
                or tuple(
                    _tensor_signature(tensor)
                    for tensor in transaction._state.grad_node_chart_f32_by_block
                )
                != tuple(
                    state.grad_node_chart_signature for state in ordered_states
                )
                or transaction.active_manifest_coverage_certified
            ):
                raise ValueError(
                    "fused transaction does not match the authoritative active manifest"
                )
            # Root the transaction before the external completion capability is
            # invoked.  The same fields remain bound through receipt sealing;
            # on any later rejection abort observes a settled transaction and
            # cannot issue a duplicate fence.
            self._failed_fused_full_geometry_transaction = transaction
            self._failed_fused_full_geometry_fence_provenance = (
                device_completion_fence_provenance
            )
            pre_fence_session_snapshot = _fused_session_snapshot(
                self,
                ordered_states,
            )
            self._fused_transaction_in_progress = True
            try:
                result = (
                    execute_kinetic_native_equal_rank_fused_direct_full_vjp_transaction_v1(
                        transaction,
                        device_completion_fence=device_completion_fence,
                        device_completion_fence_provenance=(
                            device_completion_fence_provenance
                        ),
                    )
                )
            except BaseException as error:
                self._failed_fused_full_geometry_transaction = transaction
                self._failed_fused_full_geometry_error = error
                self._failed_fused_full_geometry_fence_provenance = (
                    device_completion_fence_provenance
                )
                self._fused_full_geometry_completion_unknown = bool(
                    transaction._state.completion_unknown
                )
                raise
            finally:
                self._fused_transaction_in_progress = False
            # Keep the settled transaction rooted until the executor receipt
            # is sealed. If any later provenance check fails, abort must not
            # issue a redundant second fence.
            if (
                _fused_session_snapshot(self, self._ordered_active_states())
                != pre_fence_session_snapshot
            ):
                raise RuntimeError(
                    "fused completion callback changed the authoritative session manifest"
                )
            if not isinstance(
                result,
                KineticNativeEqualRankFusedDirectFullVjpV1TransactionResult,
            ):
                raise TypeError("fused transaction returned a foreign receipt")
            result.assert_current()
            if (
                result.active_block_generation_ids
                != active_block_generation_ids
                or result.transaction_generation_id != transaction.generation_id
                or result.block_count != len(ordered_states)
                or result.device_completion_fence_call_count != 1
                or result.length_cotangent_allocated
                or result.active_manifest_coverage_certified
                or result.optimizer_fail_atomicity_certified
            ):
                raise ValueError(
                    "fused transaction receipt does not preserve the session contract"
                )
            provisional = KineticNativeFusedFullGeometryVJPExecutionReceipt(
                transaction_result=result,
                session_generation_id=self.generation_id,
                active_runtime_generation_ids=tuple(
                    state.runtime_generation_id for state in ordered_states
                ),
                active_block_generation_ids=active_block_generation_ids,
                active_world_generation_ids=tuple(
                    state.world_generation_id for state in ordered_states
                ),
                node_bar_identities=tuple(
                    state.grad_node_chart_identity for state in ordered_states
                ),
                node_bar_signatures=tuple(
                    state.grad_node_chart_signature for state in ordered_states
                ),
                sample_manifest_digests=tuple(
                    state.sample_manifest_digest for state in ordered_states
                ),
                reduced_sample_count=sum(
                    state.streamed_sample_count for state in ordered_states
                ),
                transaction_result_identity=id(result),
                transaction_generation_id=result.transaction_generation_id,
                retained_output_tensor_bytes=result.retained_output_tensor_bytes,
                device_completion_fence_provenance=(
                    result.device_completion_fence_provenance
                ),
                generation_digest="",
                active_block_count=len(ordered_states),
                block_reverse_count=len(ordered_states),
                _session_identity=id(self),
                _seal=_FUSED_FULL_GEOMETRY_EXECUTION_SEAL,
            )
            receipt = replace(
                provisional,
                generation_digest=(
                    _fused_full_geometry_execution_digest(provisional)
                ),
            )
            receipt.assert_current()
            for state in ordered_states:
                state.fused_full_geometry_vjp_launch_count = 1
                state.fused_transaction_generation_id = (
                    result.transaction_generation_id
                )
                state.reverse_result_identity = id(result)
                state.token = None
                state.grad_node_chart_f32 = None
            self._native_fused_full_geometry_vjp_launch_count += len(
                ordered_states
            )
            self._native_fused_full_geometry_transaction_count += 1
            self._native_fused_full_geometry_completion_fence_count += 1
            self._fused_full_geometry_execution_receipt = receipt
            receipt.assert_current(self)
            self._failed_fused_full_geometry_transaction = None
            self._failed_fused_full_geometry_error = None
            self._failed_fused_full_geometry_fence_provenance = None
        return receipt

    def consume_full_geometry_vjp_execution(
        self,
        execution: KineticNativeFullGeometryVJPExecution,
        *,
        geometry_reduction: (
            KineticNativeEqualRankGeometryReduction
            | KineticNativeEqualRankSparseGeometryReduction
        ),
        expected_device_completion_fence_provenance: str,
    ) -> KineticNativeFullGeometryVJPCompletionReceipt:
        """Verify one canonical fenced reduction, then release ``[J,W]``."""

        with self._poison_on_error():
            if not isinstance(execution, KineticNativeFullGeometryVJPExecution):
                raise TypeError("execution must be KineticNativeFullGeometryVJPExecution")
            execution.assert_current(self)
            if not isinstance(
                geometry_reduction,
                (
                    KineticNativeEqualRankGeometryReduction,
                    KineticNativeEqualRankSparseGeometryReduction,
                ),
            ):
                raise TypeError(
                    "geometry_reduction must be a sealed dense-or-sparse equal-rank reduction"
                )
            if self.executor.device.type != "cpu":
                raise RuntimeError(
                    "callback-authorized staged geometry settlement is CPU-only; "
                    "accelerator roots require exact subject-bound authority"
                )
            geometry_reduction.assert_current()
            if (
                not isinstance(expected_device_completion_fence_provenance, str)
                or not expected_device_completion_fence_provenance.strip()
            ):
                raise ValueError("full-geometry fence provenance must be nonempty")
            state = self._states.get(execution.runtime_generation_id)
            if (
                state is None
                or state.full_geometry_execution is not execution
                or not state.full_geometry_execution_outstanding
                or state.full_geometry_execution_consumed
                or state.full_geometry_completion_receipt is not None
            ):
                raise ValueError("full-geometry execution is foreign or already consumed")
            native_vjp = execution.native_vjp_result
            if not isinstance(native_vjp, KineticNativeEqualRankVJPResult):
                raise ValueError("full-geometry execution lost its native VJP")
            native_vjp_provenance = kinetic_native_equal_rank_vjp_provenance_id(
                native_vjp
            )
            length_bar = native_vjp.grad_node_physical_length_f32
            length_shape = tuple(int(value) for value in length_bar.shape)
            if (
                len(length_shape) != 2
                or geometry_reduction.native_vjp_provenance_id
                != native_vjp_provenance
                or geometry_reduction.native_block_generation_digest
                != execution.native_block_generation_digest
                or geometry_reduction.native_world_generation_id
                != native_vjp.world.generation_id
                or geometry_reduction.device_completion_fence_call_count != 1
                or geometry_reduction.device_completion_fence_provenance
                != expected_device_completion_fence_provenance
            ):
                raise ValueError(
                    "full-geometry reduction does not prove this execution/fence"
                )
            execution.assert_current(self)
            geometry_reduction.assert_current()
            provisional = KineticNativeFullGeometryVJPCompletionReceipt(
                session_generation_id=self.generation_id,
                runtime_generation_id=execution.runtime_generation_id,
                native_block_generation_digest=(
                    execution.native_block_generation_digest
                ),
                execution_generation_digest=execution.generation_digest,
                native_vjp_provenance_id=native_vjp_provenance,
                native_length_bar_shape=(length_shape[0], length_shape[1]),
                native_length_bar_tensor_bytes=(
                    execution.native_length_bar_tensor_bytes
                ),
                native_length_bar_signature=_tensor_signature(length_bar),
                geometry_reduction_identity=id(geometry_reduction),
                geometry_reduction_generation_digest=(
                    geometry_reduction.generation_digest
                ),
                reduction_completion_fence_provenance=(
                    expected_device_completion_fence_provenance
                ),
                generation_digest="",
                _seal=_FULL_GEOMETRY_COMPLETION_SEAL,
            )
            completion = replace(
                provisional,
                generation_digest=_full_geometry_completion_digest(provisional),
            )
            completion.assert_current()
            state.full_geometry_execution_outstanding = False
            state.full_geometry_execution_consumed = True
            state.full_geometry_execution = None
            state.full_geometry_completion_receipt = completion
            execution.native_vjp_result = None
            execution.consumed = True
            self._native_full_geometry_fenced_reduction_count += 1
        return completion

    @torch.no_grad()
    def accept_fused_union_v2_full_geometry_result(
        self,
        result: KineticNativeEqualRankFusedUnionFullVjpV2TransactionResult,
    ) -> None:
        """Bind one exact accepted union-v2 result to the active session.

        The adapter already performed its single fenced all-block transaction.
        This method adds only the executor's exact active-manifest relation; it
        neither consumes the result bars nor performs a global write.
        """

        with self._poison_on_error():
            if type(result) is not (
                KineticNativeEqualRankFusedUnionFullVjpV2TransactionResult
            ):
                raise TypeError("union-v2 result has a foreign exact type")
            result.assert_current()
            ordered_states = self._ordered_active_states()
            if not ordered_states:
                raise ValueError("union-v2 requires an active block")
            if (
                result.active_block_generation_ids
                != tuple(
                    state.native_block_generation_digest
                    for state in ordered_states
                )
                or result.block_count != len(ordered_states)
                or self._native_sample_prepare_count
                != self._native_sample_launch_count
                or self._streamed_sample_count
                != self.requested_observation_count
            ):
                raise ValueError(
                    "union-v2 result does not cover the exact active session manifest"
                )
            if any(
                state.native_sample_launch_count < 1
                or state.material_vjp_launch_count
                or state.full_geometry_vjp_launch_count
                or state.fused_full_geometry_vjp_launch_count
                or state.token is None
                or state.grad_node_chart_f32 is None
                for state in ordered_states
            ):
                raise ValueError("union-v2 active state is not reverse-ready")
            self._bind_reverse_mode("fused_union_v2_full_geometry")
            for state in ordered_states:
                state.fused_full_geometry_vjp_launch_count = 1
                state.fused_transaction_generation_id = (
                    result.transaction_generation_id
                )
                state.reverse_result_identity = id(result)
                state.token = None
                state.grad_node_chart_f32 = None
            self._native_fused_full_geometry_vjp_launch_count += len(
                ordered_states
            )
            self._native_fused_full_geometry_transaction_count += 1
            self._native_fused_full_geometry_completion_fence_count += 1
            self._fused_union_v2_transaction_generation_id = (
                result.transaction_generation_id
            )
            self._fused_union_v2_retained_output_tensor_bytes = (
                result.retained_output_tensor_bytes
            )
            self._fused_union_v2_completion_fence_provenance = (
                result.device_completion_fence_provenance
            )

    def seal(self) -> KineticNativeMaterialStepTelemetry:
        """Prove exact active-block coverage and permanently close the session."""

        with self._poison_on_error():
            if not self._states:
                raise ValueError("native step cannot seal without an active block")
            if self._reverse_mode not in {
                "material_only",
                "full_geometry",
                "fused_full_geometry",
                "fused_union_v2_full_geometry",
            }:
                raise ValueError("native step cannot seal without one reverse mode")
            if (
                self._global_loss_element_count is None
                or self._loss_scale is None
                or self._loss_normalization_id is None
            ):
                raise ValueError("native step cannot seal without one loss normalization")
            material_only = self._reverse_mode == "material_only"
            staged_full_geometry = self._reverse_mode == "full_geometry"
            fused_direct_full_geometry = (
                self._reverse_mode == "fused_full_geometry"
            )
            fused_union_v2_full_geometry = (
                self._reverse_mode == "fused_union_v2_full_geometry"
            )
            fused_full_geometry = (
                fused_direct_full_geometry or fused_union_v2_full_geometry
            )
            ordered_states = self._ordered_active_states()
            fused_receipt = self._fused_full_geometry_execution_receipt
            if fused_direct_full_geometry:
                if fused_receipt is None:
                    raise ValueError(
                        "fused full-geometry step has no accepted transaction receipt"
                    )
                fused_receipt.assert_current(self)
            elif fused_receipt is not None:
                raise ValueError(
                    "non-fused step retained a fused full-geometry receipt"
                )
            if fused_union_v2_full_geometry:
                if (
                    not self._fused_union_v2_transaction_generation_id
                    or self._fused_union_v2_retained_output_tensor_bytes < 1
                    or not self._fused_union_v2_completion_fence_provenance
                ):
                    raise ValueError(
                        "union-v2 step has no accepted transaction telemetry"
                    )
            elif any(
                (
                    self._fused_union_v2_transaction_generation_id,
                    self._fused_union_v2_retained_output_tensor_bytes,
                    self._fused_union_v2_completion_fence_provenance,
                )
            ):
                raise ValueError("non-union step retained union-v2 telemetry")
            for state in ordered_states:
                completion = state.full_geometry_completion_receipt
                if completion is not None:
                    completion.assert_current()
                    if (
                        completion.session_generation_id != self.generation_id
                        or completion.runtime_generation_id
                        != state.runtime_generation_id
                        or completion.native_block_generation_digest
                        != state.native_block_generation_digest
                    ):
                        raise ValueError(
                            "native step contains a foreign geometry completion"
                        )
            if any(
                state.native_sample_prepare_count != state.native_sample_launch_count
                or state.native_sample_launch_count
                != state.native_sample_completion_fence_count
                or state.native_sample_launch_count < 1
                or state.streamed_sample_count < 1
                or len(state.sample_manifest_digest) != 64
                or state.first_flat_sample_index is None
                or state.last_flat_sample_index is None
                or state.last_flat_sample_index < state.first_flat_sample_index
                or state.material_vjp_launch_count
                + state.full_geometry_vjp_launch_count
                + state.fused_full_geometry_vjp_launch_count
                != 1
                or state.material_vjp_launch_count != (1 if material_only else 0)
                or state.full_geometry_vjp_launch_count
                != (1 if staged_full_geometry else 0)
                or state.fused_full_geometry_vjp_launch_count
                != (1 if fused_full_geometry else 0)
                or (
                    staged_full_geometry
                    and state.native_length_bar_tensor_bytes < 1
                )
                or (
                    not staged_full_geometry
                    and state.native_length_bar_tensor_bytes != 0
                )
                or state.full_geometry_execution_outstanding
                or state.full_geometry_execution is not None
                or state.full_geometry_execution_consumed
                != staged_full_geometry
                or (
                    state.full_geometry_completion_receipt is not None
                    if not staged_full_geometry
                    else state.full_geometry_completion_receipt is None
                )
                or state.token is not None
                or state.grad_node_chart_f32 is not None
                or state.loss_f32 is None
                or id(state.loss_f32) != state.loss_identity
                or _tensor_signature(state.loss_f32) != state.loss_signature
                or bool(state.fused_transaction_generation_id)
                != fused_full_geometry
                or (
                    fused_full_geometry
                    and fused_direct_full_geometry
                    and fused_receipt is not None
                    and state.fused_transaction_generation_id
                    != fused_receipt.transaction_generation_id
                )
                or (
                    fused_union_v2_full_geometry
                    and state.fused_transaction_generation_id
                    != self._fused_union_v2_transaction_generation_id
                )
                for state in ordered_states
            ):
                raise ValueError("native step has incomplete block launch coverage")
            active_count = len(ordered_states)
            if (
                self._native_node_forward_launch_count != active_count
                or self._native_material_vjp_launch_count
                != (active_count if material_only else 0)
                or self._native_full_geometry_vjp_launch_count
                != (active_count if staged_full_geometry else 0)
                or self._native_full_geometry_fenced_reduction_count
                != (active_count if staged_full_geometry else 0)
                or self._native_fused_full_geometry_vjp_launch_count
                != (active_count if fused_full_geometry else 0)
                or self._native_fused_full_geometry_transaction_count
                != (1 if fused_full_geometry else 0)
                or self._native_fused_full_geometry_completion_fence_count
                != (1 if fused_full_geometry else 0)
                or self._native_length_bar_tensor_bytes
                != sum(state.native_length_bar_tensor_bytes for state in ordered_states)
                or self._native_sample_prepare_count != self._native_sample_launch_count
                or self._native_sample_launch_count
                != self._native_sample_completion_fence_count
                or self._streamed_sample_count != self.requested_observation_count
                or not isinstance(
                    self._sample_completion_fence_provenance,
                    str,
                )
                or not self._sample_completion_fence_provenance.strip()
                or self._maximum_simultaneous_sample_lifetime_count != 1
                or self._outstanding_sample_lifetime is not None
                or self._pending_sample_completion is not None
            ):
                raise ArithmeticError("native step launch counts are inconsistent")
            blocks = tuple(
                KineticNativeMaterialBlockCallTelemetry(
                    runtime_generation_id=state.runtime_generation_id,
                    sampler_generation_digest=state.sampler_generation_digest,
                    native_block_generation_digest=state.native_block_generation_digest,
                    world_generation_id=state.world_generation_id,
                    native_node_forward_launch_count=1,
                    native_sample_prepare_count=state.native_sample_prepare_count,
                    native_sample_launch_count=state.native_sample_launch_count,
                    native_sample_completion_fence_count=(
                        state.native_sample_completion_fence_count
                    ),
                    streamed_sample_count=state.streamed_sample_count,
                    sample_manifest_digest=state.sample_manifest_digest,
                    first_flat_sample_index=(
                        state.first_flat_sample_index
                        if state.first_flat_sample_index is not None
                        else -1
                    ),
                    last_flat_sample_index=(
                        state.last_flat_sample_index
                        if state.last_flat_sample_index is not None
                        else -1
                    ),
                    reverse_mode=self._reverse_mode,
                    native_material_word_vjp_launch_count=(
                        state.material_vjp_launch_count
                    ),
                    executor_world_reference_released_after_reverse_completion=True,
                    native_full_geometry_vjp_launch_count=(
                        state.full_geometry_vjp_launch_count
                    ),
                    native_full_geometry_fenced_reduction_count=(
                        0
                        if state.full_geometry_completion_receipt is None
                        else 1
                    ),
                    native_fused_full_geometry_vjp_launch_count=(
                        state.fused_full_geometry_vjp_launch_count
                    ),
                    native_length_bar_tensor_bytes=(
                        state.native_length_bar_tensor_bytes
                    ),
                    full_geometry_fenced_reduction_generation_digest=(
                        ""
                        if state.full_geometry_completion_receipt is None
                        else state.full_geometry_completion_receipt.generation_digest
                    ),
                    geometry_reduction_generation_digest=(
                        ""
                        if state.full_geometry_completion_receipt is None
                        else state.full_geometry_completion_receipt.geometry_reduction_generation_digest
                    ),
                    reduction_completion_fence_provenance=(
                        ""
                        if state.full_geometry_completion_receipt is None
                        else state.full_geometry_completion_receipt.reduction_completion_fence_provenance
                    ),
                    fused_transaction_generation_id=(
                        state.fused_transaction_generation_id
                    ),
                )
                for state in ordered_states
            )
            provisional = KineticNativeMaterialStepTelemetry(
                executor_generation_id=self.executor.generation_id,
                step_generation_id=self.step_generation_id,
                session_generation_id=self.generation_id,
                requested_observation_count=self.requested_observation_count,
                eligible_native_block_count=len(self.executor.bindings),
                active_native_block_count=active_count,
                native_node_forward_launch_count=self._native_node_forward_launch_count,
                native_sample_prepare_count=self._native_sample_prepare_count,
                native_sample_launch_count=self._native_sample_launch_count,
                native_sample_completion_fence_count=(
                    self._native_sample_completion_fence_count
                ),
                streamed_sample_count=self._streamed_sample_count,
                native_material_word_vjp_launch_count=(
                    self._native_material_vjp_launch_count
                ),
                native_full_geometry_vjp_launch_count=(
                    self._native_full_geometry_vjp_launch_count
                ),
                native_full_geometry_fenced_reduction_count=(
                    self._native_full_geometry_fenced_reduction_count
                ),
                native_fused_full_geometry_vjp_launch_count=(
                    self._native_fused_full_geometry_vjp_launch_count
                ),
                native_fused_full_geometry_transaction_count=(
                    self._native_fused_full_geometry_transaction_count
                ),
                native_fused_full_geometry_completion_fence_count=(
                    self._native_fused_full_geometry_completion_fence_count
                ),
                native_length_bar_tensor_bytes=self._native_length_bar_tensor_bytes,
                reverse_mode=self._reverse_mode,
                global_loss_element_count=self._global_loss_element_count,
                loss_scale=self._loss_scale,
                loss_normalization_id=self._loss_normalization_id,
                blocks=blocks,
                generation_digest="",
                _sealed_generation_digest="",
                exactly_one_material_vjp_per_active_block=material_only,
                exactly_one_full_geometry_vjp_per_active_block=(
                    staged_full_geometry
                ),
                exactly_one_fused_full_geometry_vjp_per_active_block=(
                    fused_full_geometry
                ),
                geometry_vjp_exposed=(not material_only),
                full_geometry_completion_semantics=(
                    "one_fenced_all_block_transaction_not_optimizer_committed"
                    if fused_full_geometry
                    else "fenced_and_reduced_not_globally_committed"
                ),
                fused_full_geometry_output_tensor_bytes=(
                    self._fused_union_v2_retained_output_tensor_bytes
                    if fused_union_v2_full_geometry
                    else (
                        0
                        if fused_receipt is None
                        else fused_receipt.retained_output_tensor_bytes
                    )
                ),
                fused_full_geometry_transaction_generation_id=(
                    self._fused_union_v2_transaction_generation_id
                    if fused_union_v2_full_geometry
                    else (
                        ""
                        if fused_receipt is None
                        else fused_receipt.transaction_generation_id
                    )
                ),
                fused_full_geometry_completion_fence_provenance=(
                    self._fused_union_v2_completion_fence_provenance
                    if fused_union_v2_full_geometry
                    else (
                        ""
                        if fused_receipt is None
                        else fused_receipt.device_completion_fence_provenance
                    )
                ),
                fused_full_geometry_active_manifest_certified=(
                    fused_full_geometry
                ),
                sample_completion_fence_provenance=(
                    self._sample_completion_fence_provenance
                ),
                maximum_simultaneous_sample_lifetime_count=(
                    self._maximum_simultaneous_sample_lifetime_count
                ),
                outstanding_sample_lifetime_count_at_seal=0,
                sample_lifetime_history_retained=False,
                sample_lifetime_additional_logical_tensor_bytes=0,
                sample_lifetime_python_heap_bytes_measured=False,
            )
            generation_digest = _telemetry_generation_digest(provisional)
            telemetry = KineticNativeMaterialStepTelemetry(
                **{
                    **provisional.__dict__,
                    "generation_digest": generation_digest,
                    "_sealed_generation_digest": generation_digest,
                    "_seal": _TELEMETRY_SEAL,
                }
            )
            telemetry.assert_current()
            for state in ordered_states:
                state.loss_f32 = None
                state.full_geometry_completion_receipt = None
            self._fused_full_geometry_execution_receipt = None
            self._sealed = True
        return telemetry

    def abort(
        self,
        *,
        device_completion_fence: Callable[[], None] | None = None,
        device_completion_fence_provenance: str | None = None,
        sealed_completion_fence: PaperKineticSealedCompletionFence | None = None,
        sealed_launch_generation_digest: str | None = None,
    ) -> PaperKineticCompletionFenceReceipt | None:
        """Poison, fence, then release every executor-owned native reference."""

        if self._seal is not _SESSION_SEAL:
            raise ValueError("native material step session was not opened by its executor")
        if self._sealed:
            raise ValueError("sealed native step session cannot be aborted")
        if self._fused_transaction_in_progress:
            raise RuntimeError(
                "native material step session cannot abort reentrantly during a fused transaction"
            )
        if self._sample_settlement_in_progress:
            raise RuntimeError(
                "native material step session cannot abort reentrantly during sample settlement"
            )
        if self._pending_sample_completion is not None:
            raise RuntimeError(
                "subject-bound pending sample completion requires its exact "
                "outer composite settlement"
            )
        self._failed = True
        if self._abort_release_completed:
            raise ValueError("native material step session was already abort-released")
        if self._abort_fence_in_progress:
            raise ValueError("native material step abort fence is already in progress")
        sealed_authority = (
            type(sealed_completion_fence) is PaperKineticSealedCompletionFence
        )
        legacy_authority = callable(device_completion_fence)
        if sealed_authority == legacy_authority:
            raise TypeError("abort requires exactly one completion authority")
        if sealed_authority:
            if device_completion_fence_provenance is not None:
                raise ValueError("sealed abort forbids caller provenance")
            if (
                not isinstance(sealed_launch_generation_digest, str)
                or len(sealed_launch_generation_digest) != 64
            ):
                raise ValueError("sealed abort requires one launch-generation digest")
            sealed_completion_fence.assert_current(
                native_ops=self.executor.native_ops,
                device=self.executor.device,
            )
            authority_provenance = CAPABILITY_PROVENANCE
        else:
            if sealed_launch_generation_digest is not None:
                raise ValueError("legacy abort cannot accept a sealed launch digest")
            if self.executor.device.type != "cpu":
                raise RuntimeError(
                    "callback-authorized abort release is CPU-only; accelerator "
                    "roots require exact subject-bound authority"
                )
            if (
                not isinstance(device_completion_fence_provenance, str)
                or not device_completion_fence_provenance.strip()
            ):
                raise ValueError("device_completion_fence_provenance must be nonempty")
            authority_provenance = device_completion_fence_provenance
        if self._sample_completion_unknown:
            raise RuntimeError(
                "sample completion is unknown; restart is required and live references cannot be abort-released"
            ) from self._failed_sample_completion_error
        if self._failed_sample_completion_fence_succeeded:
            # Settlement already performed the one device-wide completion
            # fence.  Its later structural rejection cannot justify a second.
            self._abort_completion_fence_provenance = (
                self._sample_completion_fence_provenance
            )
            self._release_executor_world_references_after_abort_fence()
            self._abort_release_completed = True
            receipt = self._failed_sample_completion_sealed_receipt
            self._failed_sample_completion_sealed_receipt = None
            self._failed_sample_completion_launch_generation_digest = None
            return receipt
        failed_fused_transaction = self._failed_fused_full_geometry_transaction
        if failed_fused_transaction is not None:
            if self._fused_full_geometry_completion_unknown:
                raise RuntimeError(
                    "fused transaction completion is unknown; restart is required and live references cannot be abort-released"
                ) from self._failed_fused_full_geometry_error
            if failed_fused_transaction._state.settled:
                # The transaction adapter already performed the sole required
                # completion fence before reporting its settled rejection.
                # A second fence here would falsify the one-fence lifecycle.
                self._abort_completion_fence_provenance = (
                    self._failed_fused_full_geometry_fence_provenance
                )
                self._release_executor_world_references_after_abort_fence()
                self._failed_fused_full_geometry_transaction = None
                self._failed_fused_full_geometry_error = None
                self._abort_release_completed = True
                return None
        self._abort_fence_in_progress = True
        self._abort_completion_fence_call_count += 1
        sealed_receipt: PaperKineticCompletionFenceReceipt | None = None
        try:
            if sealed_authority:
                raise RuntimeError(
                    "sealed abort cannot mint a post-hoc launch epoch; pass an "
                    "existing completion receipt or a fully consumed capability"
                )
            else:
                returned = device_completion_fence()
                if returned is not None:
                    raise TypeError("device_completion_fence must return None")
        finally:
            self._abort_fence_in_progress = False
        self._abort_completion_fence_provenance = (
            authority_provenance
        )
        self._release_executor_world_references_after_abort_fence()
        self._abort_release_completed = True
        return sealed_receipt

    def abort_after_sealed_completion_receipt(
        self,
        receipt: PaperKineticCompletionFenceReceipt,
        capability: PaperKineticSealedCompletionFence,
        *,
        stage: str,
        launch_generation_digest: str,
        expected_fence_sequence: int,
    ) -> None:
        """Release a failed session under an already-successful exact receipt."""

        commit_plan = self.assert_abort_releasable_for_sealed_completion_receipt(
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
            consumer="native-session-abort-release",
        )
        self._commit_abort_release_after_consumed_receipt(commit_plan)

    def assert_abort_releasable_for_sealed_completion_receipt(
        self,
        receipt: PaperKineticCompletionFenceReceipt,
        capability: PaperKineticSealedCompletionFence,
        *,
        stage: str,
        launch_generation_digest: str,
        expected_fence_sequence: int,
    ) -> _KineticNativeAbortReleaseCommitPlan:
        """Validate every session root while exact authority is outstanding."""

        if self._seal is not _SESSION_SEAL or self._sealed:
            raise ValueError("only an open native session can consume abort release")
        if self._sample_completion_unknown:
            raise RuntimeError(
                "unknown sample completion cannot use a successful receipt"
            ) from self._failed_sample_completion_error
        receipt.assert_for(
            capability,
            stage=stage,
            launch_generation_digest=launch_generation_digest,
            fence_sequence=expected_fence_sequence,
            require_unconsumed=True,
        )
        return self._prepare_abort_release_commit_plan()

    def _prepare_abort_release_commit_plan(
        self,
    ) -> _KineticNativeAbortReleaseCommitPlan:
        """Freeze every exact release target before authority is consumed."""

        if (
            self._seal is not _SESSION_SEAL
            or self._sealed
            or self._sample_completion_unknown
            or self._fused_transaction_in_progress
            or self._sample_settlement_in_progress
            or self._abort_fence_in_progress
            or self._abort_release_completed
            or type(self._states) is not dict
        ):
            raise ValueError("native abort release state is not precommittable")
        lifetime = self._outstanding_sample_lifetime
        if lifetime is not None:
            if type(lifetime) is not KineticNativeSampleLaunchLifetime:
                raise TypeError("native abort sample lifetime type changed")
            lifetime.assert_retained(self)
        pending_sample_completion = self._pending_sample_completion
        if pending_sample_completion is not None and (
            type(pending_sample_completion)
            is not KineticNativePendingSampleLaunchCompletion
            or pending_sample_completion._session_identity != id(self)
            or pending_sample_completion.session_generation_id
            != self.generation_id
        ):
            raise ValueError("native abort pending sample completion changed")
        state_execution_pairs: list[
            tuple[
                _BlockExecutionState,
                KineticNativeFullGeometryVJPExecution | None,
            ]
        ] = []
        for runtime_generation_id, state in self._states.items():
            if (
                type(runtime_generation_id) is not str
                or not runtime_generation_id
                or type(state) is not _BlockExecutionState
                or state.runtime_generation_id != runtime_generation_id
            ):
                raise ValueError("native abort block-state map changed")
            execution = state.full_geometry_execution
            if (
                execution is not None
                and type(execution) is not KineticNativeFullGeometryVJPExecution
            ):
                raise TypeError("native abort full-geometry execution type changed")
            state_execution_pairs.append((state, execution))
        return _KineticNativeAbortReleaseCommitPlan(
            session_identity=id(self),
            outstanding_sample_lifetime=lifetime,
            state_execution_pairs=tuple(state_execution_pairs),
            fused_full_geometry_execution_receipt=(
                self._fused_full_geometry_execution_receipt
            ),
            failed_fused_full_geometry_transaction=(
                self._failed_fused_full_geometry_transaction
            ),
            failed_sample_completion_receipt=(
                self._failed_sample_completion_sealed_receipt
            ),
            pending_sample_completion=pending_sample_completion,
            _seal=_ABORT_RELEASE_COMMIT_PLAN_SEAL,
        )

    def _commit_abort_release_after_consumed_receipt(
        self,
        commit_plan: _KineticNativeAbortReleaseCommitPlan,
    ) -> None:
        """Non-validating assignment-only commit after authority is spent."""

        self._failed = True
        self._abort_completion_fence_provenance = CAPABILITY_PROVENANCE
        lifetime = commit_plan.outstanding_sample_lifetime
        if lifetime is not None:
            lifetime.prepared_payload = None
            lifetime.sample_block = None
            lifetime.world_token = None
            lifetime.background_rgb_f32 = None
            lifetime.loss_f32 = None
            lifetime.grad_node_chart_f32 = None
            lifetime.cone_diagnostic_i32 = None
            lifetime.phase = "released"
            lifetime.consumed = True
        self._outstanding_sample_lifetime = None
        for state, execution in commit_plan.state_execution_pairs:
            state.token = None
            state.grad_node_chart_f32 = None
            state.loss_f32 = None
            if execution is not None:
                execution.native_vjp_result = None
                execution.consumed = True
            state.full_geometry_execution = None
            state.full_geometry_execution_outstanding = False
            state.full_geometry_completion_receipt = None
        self._fused_full_geometry_execution_receipt = None
        self._failed_fused_full_geometry_transaction = None
        self._failed_fused_full_geometry_error = None
        self._failed_fused_full_geometry_fence_provenance = None
        self._failed_sample_completion_error = None
        self._failed_sample_completion_sealed_receipt = None
        self._failed_sample_completion_launch_generation_digest = None
        self._failed_sample_completion_fence_succeeded = False
        pending_sample_completion = commit_plan.pending_sample_completion
        if pending_sample_completion is not None:
            pending_sample_completion._sample_lifetime = None
            pending_sample_completion._block_state = None
            pending_sample_completion._sealed_completion_fence = None
            pending_sample_completion._launch_epoch = None
            pending_sample_completion.phase = "aborted"
        self._pending_sample_completion = None
        self._abort_release_completed = True

    def abort_after_all_sealed_completion_receipts(
        self,
        capability: PaperKineticSealedCompletionFence,
        *,
        expected_last_consumed_sequence: int,
    ) -> None:
        """Abort host-side after every registered native epoch was consumed."""

        if self._seal is not _SESSION_SEAL or self._sealed:
            raise ValueError("only an open native session can be abort-released")
        capability.assert_current(
            native_ops=self.executor.native_ops,
            device=self.executor.device,
        )
        if (
            capability.registered_launch_epoch is not None
            or capability.outstanding_receipt_sequence is not None
            or capability.successful_fence_count
            != expected_last_consumed_sequence
            or capability.consumed_fence_count
            != expected_last_consumed_sequence
        ):
            raise ValueError("native abort completion ledger is not fully consumed")
        commit_plan = self._prepare_abort_release_commit_plan()
        self._commit_abort_release_after_consumed_receipt(commit_plan)

    @contextmanager
    def _poison_on_error(self, *, allow_outstanding_sample: bool = False):
        self._assert_open(allow_outstanding_sample=allow_outstanding_sample)
        try:
            yield
        except BaseException:
            self._failed = True
            raise

    def _release_executor_world_references_after_abort_fence(self) -> None:
        pending_sample_completion = self._pending_sample_completion
        if pending_sample_completion is not None:
            pending_sample_completion._sample_lifetime = None
            pending_sample_completion._block_state = None
            pending_sample_completion._sealed_completion_fence = None
            pending_sample_completion._launch_epoch = None
            pending_sample_completion.phase = "aborted"
            self._pending_sample_completion = None
        lifetime = self._outstanding_sample_lifetime
        if lifetime is not None:
            lifetime._release_roots()
            self._outstanding_sample_lifetime = None
        for state in self._states.values():
            state.token = None
            state.grad_node_chart_f32 = None
            state.loss_f32 = None
            execution = state.full_geometry_execution
            if execution is not None:
                execution.native_vjp_result = None
                execution.consumed = True
                state.full_geometry_execution = None
                state.full_geometry_execution_outstanding = False
            state.full_geometry_completion_receipt = None
        self._fused_full_geometry_execution_receipt = None
        self._failed_fused_full_geometry_transaction = None
        self._failed_fused_full_geometry_error = None
        self._failed_fused_full_geometry_fence_provenance = None
        self._failed_sample_completion_error = None

    def _assert_open(self, *, allow_outstanding_sample: bool = False) -> None:
        if self._seal is not _SESSION_SEAL:
            raise ValueError("native material step session was not opened by its executor")
        if self._sealed:
            raise ValueError("native material step session is already sealed")
        if self._failed:
            raise ValueError("native material step session is poisoned by an earlier failure")
        if self._fused_transaction_in_progress:
            raise RuntimeError(
                "native material step session cannot be reentered during a fused transaction"
            )
        if self._sample_settlement_in_progress:
            raise RuntimeError(
                "native material step session cannot be reentered during sample settlement"
            )
        if (
            self._outstanding_sample_lifetime is not None
            and not allow_outstanding_sample
        ):
            raise RuntimeError(
                "native sample launch must settle before another session operation"
            )
        if self._pending_sample_completion is not None:
            raise RuntimeError(
                "pending sample receipt must composite-commit before another operation"
            )
        if (
            id(self.executor) != self._executor_identity
            or self.executor.generation_id != self._executor_generation_id
            or self.generation_id
            != _digest_parts(
                EXECUTOR_PROVENANCE,
                self.executor.generation_id,
                self.step_generation_id,
                self.requested_observation_count,
            )
        ):
            raise ValueError("native material step session generation/provenance changed")
        if (
            id(self.executor.native_ops) != self.executor.native_ops_identity
            or _native_abi_identity(self.executor.native_ops)
            != self.executor.native_abi_identity
        ):
            raise ValueError("native material step native ABI changed during execution")

    def _state_for_token(
        self,
        token: KineticNativeMaterialStepWorldToken,
    ) -> _BlockExecutionState:
        if not isinstance(token, KineticNativeMaterialStepWorldToken):
            raise TypeError("token must be KineticNativeMaterialStepWorldToken")
        token.assert_current(self)
        state = self._states.get(token.runtime_generation_id)
        if (
            state is None
            or state.token_identity != id(token)
            or state.token is not token
        ):
            raise ValueError("native material world token is foreign to this step")
        return state

    def _ordered_active_states(self) -> tuple[_BlockExecutionState, ...]:
        """Return the complete active manifest in canonical binding order."""

        ordered = tuple(
            self._states[binding.runtime_generation_id]
            for binding in self.executor.bindings
            if binding.runtime_generation_id in self._states
        )
        if (
            len(ordered) != len(self._states)
            or len({state.runtime_generation_id for state in ordered})
            != len(ordered)
            or {state.runtime_generation_id for state in ordered}
            != set(self._states)
        ):
            raise ValueError(
                "native session states are not the executor's exact active manifest"
            )
        return ordered

    def _bind_reverse_mode(self, reverse_mode: str) -> None:
        if reverse_mode not in {
            "material_only",
            "full_geometry",
            "fused_full_geometry",
            "fused_union_v2_full_geometry",
        }:
            raise ValueError("native step reverse mode is invalid")
        if self._reverse_mode is None:
            self._reverse_mode = reverse_mode
        elif self._reverse_mode != reverse_mode:
            raise ValueError("one native step cannot mix reverse modes")

    def _bind_or_validate_loss_normalization(
        self,
        sample_block: PaperKineticRowRaggedSampleBlock,
    ) -> None:
        if sample_block.global_loss_element_count < self.requested_observation_count * 3:
            raise ValueError(
                "native step loss denominator is smaller than its requested RGB coverage"
            )
        if self._global_loss_element_count is None:
            self._global_loss_element_count = sample_block.global_loss_element_count
            self._loss_scale = sample_block.loss_scale
            self._loss_normalization_id = sample_block.loss_normalization_id
            return
        if (
            sample_block.global_loss_element_count
            != self._global_loss_element_count
            or sample_block.loss_scale != self._loss_scale
            or sample_block.loss_normalization_id != self._loss_normalization_id
        ):
            raise ValueError("one native step cannot mix loss normalizations")

    def _bind_or_validate_node_bar(
        self,
        state: _BlockExecutionState,
        grad_node_chart_f32: torch.Tensor,
    ) -> None:
        if state.grad_node_chart_identity is None:
            storage_identity = _tensor_storage_identity(grad_node_chart_f32)
            prior_owner = self._node_bar_owner_by_storage.get(storage_identity)
            if prior_owner is not None:
                raise ValueError(
                    "native blocks must not share one node-bar accumulator"
                )
            if storage_identity in self._loss_owner_by_storage:
                raise ValueError("native node and loss accumulators must not alias storage")
            grad_node_chart_f32.zero_()
            state.grad_node_chart_identity = id(grad_node_chart_f32)
            state.grad_node_chart_signature = _tensor_signature(
                grad_node_chart_f32
            )
            state.grad_node_chart_f32 = grad_node_chart_f32
            self._node_bar_owner_by_storage[storage_identity] = (
                state.runtime_generation_id
            )
            return
        signature = _tensor_signature(grad_node_chart_f32)
        if (
            state.grad_node_chart_f32 is not grad_node_chart_f32
            or id(grad_node_chart_f32) != state.grad_node_chart_identity
            or signature != state.grad_node_chart_signature
        ):
            raise ValueError("native block must accumulate into one caller-owned node bar")

    def _bind_or_validate_loss(
        self,
        state: _BlockExecutionState,
        loss_f32: torch.Tensor,
    ) -> None:
        if state.loss_identity is None:
            storage_identity = _tensor_storage_identity(loss_f32)
            prior_owner = self._loss_owner_by_storage.get(storage_identity)
            if prior_owner is not None:
                raise ValueError(
                    "native blocks must not share one loss accumulator"
                )
            if storage_identity in self._node_bar_owner_by_storage:
                raise ValueError("native loss and node accumulators must not alias storage")
            loss_f32.zero_()
            state.loss_identity = id(loss_f32)
            state.loss_signature = _tensor_signature(loss_f32)
            state.loss_f32 = loss_f32
            self._loss_owner_by_storage[storage_identity] = (
                state.runtime_generation_id
            )
            return
        signature = _tensor_signature(loss_f32)
        if (
            state.loss_f32 is not loss_f32
            or id(loss_f32) != state.loss_identity
            or signature != state.loss_signature
        ):
            raise ValueError("native block must accumulate into one caller-owned loss scalar")


def prepare_kinetic_native_node_forward_into_lifetime(
    session: KineticNativeMaterialStepSession,
    runtime: KineticNativeEqualRankRuntimeBlock,
    compact_site_rgba_f32: torch.Tensor,
) -> KineticNativeNodeForwardIntoLifetime:
    """Install one bounded forward lifetime before output allocation."""

    if not isinstance(session, KineticNativeMaterialStepSession):
        raise TypeError("session must be KineticNativeMaterialStepSession")
    session._assert_open()
    if not isinstance(runtime, KineticNativeEqualRankRuntimeBlock):
        raise TypeError("runtime must be KineticNativeEqualRankRuntimeBlock")
    runtime.assert_warm_layout()
    _require_tensor(
        compact_site_rgba_f32,
        name="compact_site_rgba_f32",
        device=session.executor.device,
        dtype=torch.float32,
        shape=(runtime.compact_site_count, 4),
    )
    forward_into = getattr(runtime.native_ops, FORWARD_INTO_OP_NAME, None)
    if not callable(forward_into):
        raise RuntimeError(
            "lifetime-safe lazy execution requires the distinct caller-owned "
            f"forward ABI {FORWARD_INTO_OP_NAME}"
        )
    lifetime = KineticNativeNodeForwardIntoLifetime(
        session_generation_id=session.generation_id,
        runtime=runtime,
        compact_site_rgba_f32=compact_site_rgba_f32,
        _runtime_identity=id(runtime),
        _compact_identity=id(compact_site_rgba_f32),
        _forward_into_implementation_id=id(
            getattr(forward_into, "__func__", forward_into)
        ),
        _seal=_FORWARD_INTO_LIFETIME_SEAL,
    )
    lifetime.assert_retained(session)
    return lifetime


def prepare_kinetic_native_material_step_executor(
    native_ops: Any,
    runtime_sampler_pairs: Sequence[
        tuple[KineticNativeEqualRankRuntimeBlock, PaperKineticRowRaggedSampler]
    ],
    *,
    backend_provenance: str,
) -> KineticNativeMaterialStepExecutor:
    """Cold-bind one real native-ops object to compiled runtime/sampler pairs."""

    if native_ops is None:
        raise TypeError("native_ops must be provided as one object")
    if not isinstance(backend_provenance, str) or not backend_provenance.strip():
        raise ValueError("backend_provenance must be nonempty")
    abi_identity = _require_native_abi(native_ops)
    pairs = tuple(runtime_sampler_pairs)
    if not pairs:
        raise ValueError("native material executor requires at least one runtime/sampler pair")
    bindings = []
    runtime_identities: set[int] = set()
    runtime_generation_ids: set[str] = set()
    validated_samplers: dict[int, PaperKineticRowRaggedSampler] = {}
    devices: set[torch.device] = set()
    for pair in pairs:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise TypeError("runtime_sampler_pairs must contain (runtime, sampler) tuples")
        runtime, sampler = pair
        if not isinstance(runtime, KineticNativeEqualRankRuntimeBlock):
            raise TypeError("runtime/sampler pair has an invalid runtime")
        if not isinstance(sampler, PaperKineticRowRaggedSampler):
            raise TypeError("runtime/sampler pair has an invalid sampler")
        runtime.assert_warm_layout()
        prior_sampler = validated_samplers.get(id(sampler))
        if prior_sampler is None:
            sampler.assert_cold_current()
            validated_samplers[id(sampler)] = sampler
        elif prior_sampler is not sampler:
            raise ValueError("sampler identity collision changed the cold binding")
        if runtime.native_ops is not native_ops:
            raise ValueError("every runtime must bind the executor native_ops object")
        if id(runtime) in runtime_identities or runtime.generation_id in runtime_generation_ids:
            raise ValueError("native material executor received a duplicate runtime")
        matching_blocks = tuple(
            block
            for bucket in sampler.lowering.buckets
            for block in bucket.blocks
            if block.generation_digest == runtime.payload.block.generation_digest
        )
        if len(matching_blocks) != 1 or matching_blocks[0] is not runtime.payload.block:
            raise ValueError("runtime payload is not the sampler's unique native block")
        if (
            runtime.global_site_count != sampler.lowering.global_site_count
            or runtime.row_count != matching_blocks[0].row_count
            or runtime.node_count != matching_blocks[0].node_count
        ):
            raise ValueError("runtime and sampler native block shape/provenance differ")
        binding_generation = _binding_generation_id(
            runtime,
            sampler_generation_digest=sampler.generation_digest,
        )
        bindings.append(
            KineticNativeMaterialRuntimeBinding(
                runtime=runtime,
                runtime_identity=id(runtime),
                sampler_identity=id(sampler),
                sampler_view_index=sampler.view_index,
                runtime_generation_id=runtime.generation_id,
                sampler_generation_digest=sampler.generation_digest,
                native_block_generation_digest=(
                    runtime.payload.block.generation_digest
                ),
                generation_id=binding_generation,
                _sealed_generation_id=binding_generation,
                _seal=_BINDING_SEAL,
            )
        )
        runtime_identities.add(id(runtime))
        runtime_generation_ids.add(runtime.generation_id)
        devices.add(runtime.device)
    if len(devices) != 1:
        raise ValueError("one native material executor cannot span multiple devices")
    normalized_bindings = tuple(bindings)
    for binding in normalized_bindings:
        binding.assert_current()
    device = next(iter(devices))
    generation_id = _executor_generation_id(
        native_ops=native_ops,
        native_abi_identity=abi_identity,
        bindings=normalized_bindings,
        device=device,
        backend_provenance=backend_provenance,
    )
    executor = KineticNativeMaterialStepExecutor(
        native_ops=native_ops,
        bindings=normalized_bindings,
        device=device,
        backend_provenance=backend_provenance,
        native_ops_identity=id(native_ops),
        native_abi_identity=abi_identity,
        binding_identities=tuple(id(binding) for binding in normalized_bindings),
        generation_id=generation_id,
        _sealed_generation_id=generation_id,
        _seal=_EXECUTOR_SEAL,
    )
    executor.assert_current()
    return executor


def _require_native_abi(native_ops: Any) -> tuple[tuple[str, int], ...]:
    identity = _native_abi_identity(native_ops)
    for name, _implementation_id in identity:
        if not callable(getattr(native_ops, name, None)):
            raise TypeError(f"native_ops does not expose callable {name}")
    return identity


def _native_abi_identity(native_ops: Any) -> tuple[tuple[str, int], ...]:
    identities = []
    for name in _REQUIRED_NATIVE_OP_NAMES:
        callable_value = getattr(native_ops, name, None)
        implementation = getattr(callable_value, "__func__", callable_value)
        identities.append((name, id(implementation)))
    return tuple(identities)


def _world_token_generation_id(
    session: KineticNativeMaterialStepSession,
    binding: KineticNativeMaterialRuntimeBinding,
    world: KineticNativeEqualRankWorld,
) -> str:
    return _digest_parts(
        EXECUTOR_PROVENANCE,
        session.generation_id,
        binding.generation_id,
        world.generation_id,
    )


def _binding_generation_id(
    runtime: KineticNativeEqualRankRuntimeBlock,
    *,
    sampler_generation_digest: str,
) -> str:
    return _digest_parts(
        EXECUTOR_PROVENANCE,
        runtime.generation_id,
        sampler_generation_digest,
        runtime.payload.block.generation_digest,
    )


def _executor_generation_id(
    *,
    native_ops: Any,
    native_abi_identity: tuple[tuple[str, int], ...],
    bindings: tuple[KineticNativeMaterialRuntimeBinding, ...],
    device: torch.device,
    backend_provenance: str,
) -> str:
    return _digest_parts(
        EXECUTOR_PROVENANCE,
        backend_provenance,
        str(device),
        id(native_ops),
        native_abi_identity,
        tuple(binding.generation_id for binding in bindings),
    )


def _sample_launch_lifetime_digest(
    lifetime: KineticNativeSampleLaunchLifetime,
) -> str:
    return _digest_parts(
        EXECUTOR_PROVENANCE,
        "native-sample-launch-lifetime",
        lifetime.session_generation_id,
        lifetime.runtime_generation_id,
        lifetime.sampler_generation_digest,
        lifetime.native_block_generation_digest,
        lifetime.sample_block_generation_digest,
        lifetime.sample_dispatch_generation_digest,
        lifetime.prior_sample_manifest_digest,
        lifetime.next_sample_manifest_digest,
        lifetime.flat_sample_identity_digest,
        lifetime.sample_count,
        lifetime.first_flat_sample_index,
        lifetime.last_flat_sample_index,
        lifetime.read_only_tensor_signatures,
        lifetime.prepared_payload_signature,
        lifetime.writable_tensor_signatures_after_launch,
        lifetime._session_identity,
        lifetime._block_state_identity,
        lifetime._prepared_payload_identity,
        lifetime._sample_block_identity,
        lifetime._world_token_identity,
    )


def _sample_completion_receipt_digest(
    receipt: KineticNativeSampleLaunchCompletionReceipt,
) -> str:
    return _digest_parts(
        EXECUTOR_PROVENANCE,
        "native-sample-launch-completion",
        receipt.session_generation_id,
        receipt.runtime_generation_id,
        receipt.native_block_generation_digest,
        receipt.sample_lifetime_generation_digest,
        receipt.sample_manifest_digest,
        receipt.sample_count,
        receipt.first_flat_sample_index,
        receipt.last_flat_sample_index,
        receipt.device_completion_fence_provenance,
        receipt.sealed_completion_capability_generation_digest,
        receipt.sealed_completion_receipt_generation_digest,
        receipt.sealed_completion_fence_sequence,
        receipt.sealed_completion_scope,
        receipt.sealed_completion_normalized_device,
        receipt.sealed_completion_launch_generation_digest,
        receipt.device_completion_fence_call_count,
        receipt.maximum_simultaneous_sample_lifetime_count,
        receipt.sample_roots_released,
        receipt.retained_tensor_or_sample_reference_count,
        receipt.provenance,
    )


def _pending_sample_completion_digest(
    pending: KineticNativePendingSampleLaunchCompletion,
) -> str:
    return _digest_parts(
        EXECUTOR_PROVENANCE,
        "native-pending-sample-completion-v1",
        pending.session_generation_id,
        pending.runtime_generation_id,
        pending.sample_lifetime_generation_digest,
        pending.pending_identity,
        pending.capability_identity,
        pending.capability_generation_digest,
        pending.capability_owner_generation_digest,
        pending.subject_binding_identity,
        pending.subject_binding_generation_digest,
        pending.subject_identity,
        pending.subject_generation_digest,
        pending.launch_epoch_identity,
        pending.launch_epoch_generation_digest,
        pending.launch_stage,
        pending.launch_generation_digest,
        pending.launch_epoch_sequence,
        pending.receipt_identity,
        pending.receipt_generation_digest,
        pending.completion_receipt_identity,
        pending.completion_receipt_generation_digest,
        pending.next_grad_node_chart_signature,
        pending.next_loss_signature,
        pending.next_state_prepare_count,
        pending.next_state_launch_count,
        pending.next_state_fence_count,
        pending.next_state_streamed_count,
        pending.next_sample_manifest_digest,
        pending.next_first_flat_sample_index,
        pending.next_last_flat_sample_index,
        pending.next_native_prepare_count,
        pending.next_native_launch_count,
        pending.next_native_fence_count,
        pending.next_streamed_count,
        pending._session_identity,
        pending._sample_lifetime_identity,
        pending._block_state_identity,
        pending.provenance,
    )


def _telemetry_generation_digest(
    telemetry: KineticNativeMaterialStepTelemetry,
) -> str:
    return _digest_parts(
        EXECUTOR_PROVENANCE,
        telemetry.executor_generation_id,
        telemetry.step_generation_id,
        telemetry.session_generation_id,
        telemetry.requested_observation_count,
        telemetry.eligible_native_block_count,
        telemetry.active_native_block_count,
        telemetry.native_node_forward_launch_count,
        telemetry.native_sample_prepare_count,
        telemetry.native_sample_launch_count,
        telemetry.native_sample_completion_fence_count,
        telemetry.streamed_sample_count,
        telemetry.native_material_word_vjp_launch_count,
        telemetry.native_full_geometry_vjp_launch_count,
        telemetry.native_full_geometry_fenced_reduction_count,
        telemetry.native_fused_full_geometry_vjp_launch_count,
        telemetry.native_fused_full_geometry_transaction_count,
        telemetry.native_fused_full_geometry_completion_fence_count,
        telemetry.native_length_bar_tensor_bytes,
        telemetry.reverse_mode,
        telemetry.global_loss_element_count,
        telemetry.loss_scale,
        telemetry.loss_normalization_id,
        telemetry.blocks,
        telemetry.call_count_scope,
        telemetry.full_geometry_global_accumulation_proven,
        telemetry.full_geometry_completion_semantics,
        telemetry.fused_full_geometry_output_tensor_bytes,
        telemetry.fused_full_geometry_transaction_generation_id,
        telemetry.fused_full_geometry_completion_fence_provenance,
        telemetry.fused_full_geometry_active_manifest_certified,
        telemetry.fused_full_geometry_length_cotangent_allocated,
        telemetry.optimizer_fail_atomicity_certified,
        telemetry.sample_completion_fence_provenance,
        telemetry.maximum_simultaneous_sample_lifetime_count,
        telemetry.outstanding_sample_lifetime_count_at_seal,
        telemetry.sample_lifetime_history_retained,
        telemetry.sample_lifetime_additional_logical_tensor_bytes,
        telemetry.sample_lifetime_python_heap_bytes_measured,
        telemetry.runtime_status,
    )


def _full_geometry_execution_digest(
    execution: KineticNativeFullGeometryVJPExecution,
) -> str:
    return _digest_parts(
        EXECUTOR_PROVENANCE,
        "full-geometry-vjp-execution",
        execution.session_generation_id,
        execution.runtime_generation_id,
        execution.native_block_generation_digest,
        execution.reduced_sample_chunk_count,
        execution.reduced_sample_count,
        execution.sample_manifest_digest,
        execution.first_flat_sample_index,
        execution.last_flat_sample_index,
        execution.node_bar_identity,
        execution.node_bar_signature,
        execution.loss_identity,
        execution.loss_signature,
        execution.native_length_bar_tensor_bytes,
        execution._session_identity,
        execution._native_vjp_result_identity,
    )


def _full_geometry_completion_digest(
    completion: KineticNativeFullGeometryVJPCompletionReceipt,
) -> str:
    return _digest_parts(
        EXECUTOR_PROVENANCE,
        "full-geometry-vjp-completion",
        completion.session_generation_id,
        completion.runtime_generation_id,
        completion.native_block_generation_digest,
        completion.execution_generation_digest,
        completion.native_vjp_provenance_id,
        completion.native_length_bar_shape,
        completion.native_length_bar_tensor_bytes,
        completion.native_length_bar_signature,
        completion.geometry_reduction_identity,
        completion.geometry_reduction_generation_digest,
        completion.reduction_completion_fence_provenance,
        completion.geometry_reduction_success_count,
        completion.reduction_completion_fence_call_count,
        completion.execution_consumed,
        completion.native_or_geometry_tensors_retained,
        completion.global_accumulation_proven,
        completion.completion_semantics,
    )


def _fused_full_geometry_execution_digest(
    receipt: KineticNativeFusedFullGeometryVJPExecutionReceipt,
) -> str:
    return _digest_parts(
        EXECUTOR_PROVENANCE,
        "fused-full-geometry-vjp-execution",
        receipt.session_generation_id,
        receipt.active_runtime_generation_ids,
        receipt.active_block_generation_ids,
        receipt.active_world_generation_ids,
        receipt.node_bar_identities,
        receipt.node_bar_signatures,
        receipt.sample_manifest_digests,
        receipt.reduced_sample_count,
        receipt.transaction_result_identity,
        receipt.transaction_generation_id,
        receipt.retained_output_tensor_bytes,
        receipt.device_completion_fence_provenance,
        receipt.active_block_count,
        receipt.block_reverse_count,
        receipt.transaction_count,
        receipt.device_completion_fence_call_count,
        receipt.active_manifest_coverage_certified,
        receipt.exact_token_world_node_bar_identity_certified,
        receipt.length_cotangent_allocated,
        receipt.optimizer_fail_atomicity_certified,
        receipt.optimizer_commit_performed,
        receipt.native_runtime_verified,
        receipt.camera_mode,
        receipt.ray_cotangent_surface_exposed,
        receipt._session_identity,
    )


def _require_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> None:
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.device != device
        or tensor.dtype != dtype
        or tensor.layout != torch.strided
        or tuple(tensor.shape) != shape
        or not tensor.is_contiguous()
        or tensor.requires_grad
    ):
        raise ValueError(f"{name} has an invalid device/dtype/layout/shape")


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


def _prepared_sample_payload_signature(prepared: Any) -> tuple[object, ...]:
    """Bind the production prepared ABI without requiring its concrete type."""

    tensor_fields = (
        "node_chart_f32",
        "sample_row_i32",
        "sample_to_node_f32",
        "target_rgb_f32",
        "background_rgb_f32",
        "config_i32",
        "config_f32",
    )
    scalar_fields = ("row_count", "node_count", "sample_count")
    tensor_bindings: list[tuple[object, ...]] = []
    for name in tensor_fields:
        value = getattr(prepared, name, None)
        if isinstance(value, torch.Tensor):
            tensor_bindings.append((name, _tensor_signature(value)))
        elif hasattr(prepared, name):
            tensor_bindings.append((name, type(value).__qualname__, repr(value)))
    scalar_bindings = tuple(
        (name, getattr(prepared, name))
        for name in scalar_fields
        if hasattr(prepared, name)
    )
    declared_signatures = getattr(prepared, "tensor_signatures", None)
    if declared_signatures is not None:
        declared_signatures = tuple(tuple(item) for item in declared_signatures)
    return (
        type(prepared).__module__,
        type(prepared).__qualname__,
        tuple(tensor_bindings),
        scalar_bindings,
        declared_signatures,
    )


def _sample_settlement_snapshot(
    session: KineticNativeMaterialStepSession,
    state: _BlockExecutionState,
    lifetime: KineticNativeSampleLaunchLifetime,
) -> tuple[object, ...]:
    """Snapshot all ledger state a completion callback may not mutate."""

    return (
        _fused_session_snapshot(session, session._ordered_active_states()),
        id(state),
        state.runtime_generation_id,
        state.native_sample_prepare_count,
        state.native_sample_launch_count,
        state.native_sample_completion_fence_count,
        state.streamed_sample_count,
        state.sample_manifest_digest,
        state.first_flat_sample_index,
        state.last_flat_sample_index,
        state.grad_node_chart_identity,
        state.grad_node_chart_signature,
        state.loss_identity,
        state.loss_signature,
        id(lifetime),
        lifetime.generation_digest,
        lifetime.phase,
        lifetime.completion_unknown,
        lifetime.consumed,
    )


def _fused_session_snapshot(
    session: KineticNativeMaterialStepSession,
    ordered_states: Sequence[_BlockExecutionState],
) -> tuple[object, ...]:
    """Bind every sampled input the completion callback must not change."""

    return (
        id(session),
        id(session.executor),
        id(session._states),
        session._executor_identity,
        session._executor_generation_id,
        session.step_generation_id,
        session.requested_observation_count,
        session.generation_id,
        tuple(session._states),
        session._native_node_forward_launch_count,
        session._native_sample_prepare_count,
        session._native_sample_launch_count,
        session._native_sample_completion_fence_count,
        session._streamed_sample_count,
        session._native_material_vjp_launch_count,
        session._native_full_geometry_vjp_launch_count,
        session._native_full_geometry_fenced_reduction_count,
        session._native_fused_full_geometry_vjp_launch_count,
        session._native_fused_full_geometry_transaction_count,
        session._native_fused_full_geometry_completion_fence_count,
        session._native_length_bar_tensor_bytes,
        session._reverse_mode,
        session._global_loss_element_count,
        session._loss_scale,
        session._loss_normalization_id,
        id(session._outstanding_sample_lifetime)
        if session._outstanding_sample_lifetime is not None
        else None,
        id(session._pending_sample_completion)
        if session._pending_sample_completion is not None
        else None,
        session._sample_settlement_in_progress,
        session._sample_completion_unknown,
        id(session._failed_sample_completion_error)
        if session._failed_sample_completion_error is not None
        else None,
        session._failed_sample_completion_fence_succeeded,
        session._sample_completion_fence_provenance,
        session._maximum_simultaneous_sample_lifetime_count,
        id(session._fused_full_geometry_execution_receipt)
        if session._fused_full_geometry_execution_receipt is not None
        else None,
        id(session._failed_fused_full_geometry_transaction)
        if session._failed_fused_full_geometry_transaction is not None
        else None,
        id(session._failed_fused_full_geometry_error)
        if session._failed_fused_full_geometry_error is not None
        else None,
        session._failed_fused_full_geometry_fence_provenance,
        session._fused_full_geometry_completion_unknown,
        session._fused_transaction_in_progress,
        id(session._node_bar_owner_by_storage),
        tuple(session._node_bar_owner_by_storage.items()),
        id(session._loss_owner_by_storage),
        tuple(session._loss_owner_by_storage.items()),
        session._sealed,
        session._failed,
        session._abort_fence_in_progress,
        session._abort_release_completed,
        session._abort_completion_fence_call_count,
        session._abort_completion_fence_provenance,
        id(session._seal),
        tuple(
            (
                id(state),
                id(state.runtime_binding),
                state.runtime_generation_id,
                state.sampler_generation_digest,
                state.native_block_generation_digest,
                state.world_generation_id,
                state.token_identity,
                id(state.token),
                state.native_sample_prepare_count,
                state.native_sample_launch_count,
                state.native_sample_completion_fence_count,
                state.streamed_sample_count,
                state.sample_manifest_digest,
                state.first_flat_sample_index,
                state.last_flat_sample_index,
                state.material_vjp_launch_count,
                state.full_geometry_vjp_launch_count,
                state.fused_full_geometry_vjp_launch_count,
                state.fused_transaction_generation_id,
                state.native_length_bar_tensor_bytes,
                state.reverse_result_identity,
                state.grad_node_chart_identity,
                state.grad_node_chart_signature,
                _tensor_signature(state.grad_node_chart_f32)
                if isinstance(state.grad_node_chart_f32, torch.Tensor)
                else None,
                state.loss_identity,
                state.loss_signature,
                _tensor_signature(state.loss_f32)
                if isinstance(state.loss_f32, torch.Tensor)
                else None,
                id(state.full_geometry_execution)
                if state.full_geometry_execution is not None
                else None,
                id(state.full_geometry_completion_receipt)
                if state.full_geometry_completion_receipt is not None
                else None,
                state.full_geometry_execution_outstanding,
                state.full_geometry_execution_consumed,
            )
            for state in ordered_states
        ),
    )


def _tensor_storage_identity(tensor: torch.Tensor) -> tuple[object, ...]:
    storage = tensor.untyped_storage()
    return (
        tensor.device,
        storage.data_ptr(),
        int(storage.nbytes()),
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


__all__ = [
    "EXECUTOR_PROVENANCE",
    "EXECUTOR_STATUS",
    "KineticNativeFusedFullGeometryVJPExecutionReceipt",
    "KineticNativeFullGeometryVJPCompletionReceipt",
    "KineticNativeFullGeometryVJPExecution",
    "KineticNativeMaterialBlockCallTelemetry",
    "KineticNativeMaterialExecutorMemory",
    "KineticNativeMaterialRuntimeBinding",
    "KineticNativeMaterialStepExecutor",
    "KineticNativeMaterialStepSession",
    "KineticNativeMaterialStepTelemetry",
    "KineticNativeMaterialStepWorldToken",
    "KineticNativeNodeForwardIntoLifetime",
    "KineticNativePendingSampleLaunchCompletion",
    "KineticNativeSampleLaunchCompletionReceipt",
    "KineticNativeSampleLaunchLifetime",
    "SAMPLE_LAUNCH_OP_NAME",
    "SAMPLE_PREPARE_OP_NAME",
    "prepare_kinetic_native_node_forward_into_lifetime",
    "prepare_kinetic_native_material_step_executor",
]
