from __future__ import annotations

import gc
import inspect
import threading
import weakref
from dataclasses import replace

import pytest

from kinetic_sealed_completion_fence import (
    CPU_CALL_RETURN_SCOPE,
    PaperKineticCompletionLaunchEpoch,
    PaperKineticCompletionUnknownError,
    PaperKineticCompletionFenceReceipt,
    PaperKineticCompletionSubjectBinding,
    PaperKineticSealedCompletionFence,
    prepare_paper_kinetic_completion_subject_binding,
    prepare_paper_kinetic_sealed_completion_fence,
)


_OWNER_GENERATION = "a" * 64
_LAUNCH_GENERATION_0 = "b" * 64
_LAUNCH_GENERATION_1 = "c" * 64
_SUBJECT_GENERATION_0 = "d" * 64
_SUBJECT_GENERATION_1 = "e" * 64


class _SynchronousCpuNativeOps:
    def kinetic_precompiled_length_p0_lie_node_forward_into_launch_only_v1(
        self,
    ) -> None:
        return None

    def kinetic_precompiled_length_p0_lie_material_node_vjp_accumulate_launch_only(
        self,
    ) -> None:
        return None

    def prepare_kinetic_ragged_p0_lie_sample_block(self) -> None:
        return None

    def kinetic_ragged_p0_lie_sample_accumulate_loss_only_launch_only(
        self,
    ) -> None:
        return None


class _StableCompletionSubject:
    def __init__(self, generation_digest: str) -> None:
        self.generation_digest = generation_digest


def _prepare_cpu_capability(
    native_ops: _SynchronousCpuNativeOps,
) -> PaperKineticSealedCompletionFence:
    return prepare_paper_kinetic_sealed_completion_fence(
        native_ops,
        device="cpu",
        owner_generation_digest=_OWNER_GENERATION,
    )


def _register_sample_epoch(
    capability: PaperKineticSealedCompletionFence,
    *,
    launch_generation_digest: str,
    subject_generation_digest: str = _SUBJECT_GENERATION_0,
):
    subject = _StableCompletionSubject(subject_generation_digest)
    binding = prepare_paper_kinetic_completion_subject_binding(
        capability,
        subject,
        kind="test-sample-composite",
        subject_generation_digest=subject_generation_digest,
    )
    epoch = capability.register_launch(
        stage="sample-completion",
        launch_generation_digest=launch_generation_digest,
        subject_binding=binding,
    )
    return subject, binding, epoch


def test_cpu_completion_capability_returns_monotone_exact_receipts() -> None:
    native_ops = _SynchronousCpuNativeOps()
    capability = _prepare_cpu_capability(native_ops)

    first_subject, first_binding, first_epoch = _register_sample_epoch(
        capability,
        launch_generation_digest=_LAUNCH_GENERATION_0,
    )
    first = capability.fence(first_epoch)
    first.consume_for_subject(
        capability,
        first_binding,
        subject=first_subject,
        consumer="test-first-release",
    )
    second_epoch = capability.register_launch(
        stage="reverse-completion",
        launch_generation_digest=_LAUNCH_GENERATION_1,
    )
    second = capability.fence(second_epoch)
    second.consume_for(
        capability,
        stage="reverse-completion",
        launch_generation_digest=_LAUNCH_GENERATION_1,
        fence_sequence=2,
        consumer="test-second-release",
    )

    assert type(capability) is PaperKineticSealedCompletionFence
    assert type(first) is type(second) is PaperKineticCompletionFenceReceipt
    assert first.fence_sequence == 1
    assert second.fence_sequence == 2
    assert first.completion_scope == second.completion_scope == CPU_CALL_RETURN_SCOPE
    assert first.completion_domain_drained is True
    assert first.capability_generation_digest == capability.generation_digest
    assert capability.fence_attempt_count == capability.successful_fence_count == 2
    assert capability.consumed_fence_count == 2
    assert capability.outstanding_receipt_sequence is None
    assert capability.next_fence_sequence == 3
    assert capability.completion_unknown is False
    first.assert_current()
    second.assert_current()
    capability.assert_current(native_ops=native_ops, device="cpu")


def test_capability_constructor_exposes_no_caller_provenance_or_callback() -> None:
    parameters = inspect.signature(
        prepare_paper_kinetic_sealed_completion_fence
    ).parameters

    assert "backend_provenance" not in parameters
    assert "completion_callback" not in parameters
    assert "device_completion_fence" not in parameters


def test_native_abi_mutation_invalidates_capability_before_another_fence() -> None:
    native_ops = _SynchronousCpuNativeOps()
    capability = _prepare_cpu_capability(native_ops)
    native_ops.prepare_kinetic_ragged_p0_lie_sample_block = lambda: None

    with pytest.raises(ValueError, match="capability changed"):
        capability.assert_current()
    with pytest.raises(ValueError, match="capability changed"):
        epoch = capability.register_launch(
            stage="forbidden-after-abi-mutation",
            launch_generation_digest=_LAUNCH_GENERATION_0,
        )
        capability.fence(epoch)
    assert capability.fence_attempt_count == 0


def test_unconsumed_receipt_blocks_every_later_launch_and_cross_capability_use() -> None:
    native_ops = _SynchronousCpuNativeOps()
    first_capability = _prepare_cpu_capability(native_ops)
    second_capability = _prepare_cpu_capability(native_ops)
    assert first_capability.generation_digest != second_capability.generation_digest
    assert (
        first_capability.capability_nonce_digest
        != second_capability.capability_nonce_digest
    )

    subject, binding, epoch = _register_sample_epoch(
        first_capability,
        launch_generation_digest=_LAUNCH_GENERATION_0,
    )
    receipt = first_capability.fence(epoch)
    with pytest.raises(RuntimeError, match="must settle before registration"):
        first_capability.register_launch(
            stage="forbidden-overlap",
            launch_generation_digest=_LAUNCH_GENERATION_1,
        )
    with pytest.raises(ValueError, match="foreign or consumed"):
        receipt.assert_for(
            second_capability,
            stage="sample-completion",
            launch_generation_digest=_LAUNCH_GENERATION_0,
            fence_sequence=1,
        )
    receipt.consume_for_subject(
        first_capability,
        binding,
        subject=subject,
        consumer="test-exact-capability-release",
    )
    with pytest.raises(ValueError, match="foreign or consumed"):
        receipt.consume_for_subject(
            first_capability,
            binding,
            subject=subject,
            consumer="forbidden-double-consume",
        )


def test_registered_epoch_is_single_outstanding_and_foreign_epochs_fail_closed() -> None:
    native_ops = _SynchronousCpuNativeOps()
    first_capability = _prepare_cpu_capability(native_ops)
    second_capability = _prepare_cpu_capability(native_ops)
    first_subject, first_binding, first_epoch = _register_sample_epoch(
        first_capability,
        launch_generation_digest=_LAUNCH_GENERATION_0,
    )
    cloned_epoch = replace(first_epoch)

    with pytest.raises(RuntimeError, match="must settle before registration"):
        first_capability.register_launch(
            stage="forbidden-overlap",
            launch_generation_digest=_LAUNCH_GENERATION_1,
        )
    with pytest.raises(ValueError, match="foreign, stale, or reused"):
        first_capability.fence(cloned_epoch)
    foreign_subject, foreign_binding, foreign_epoch = _register_sample_epoch(
        second_capability,
        launch_generation_digest=_LAUNCH_GENERATION_0,
        subject_generation_digest=_SUBJECT_GENERATION_1,
    )
    with pytest.raises(ValueError, match="foreign, stale, or reused"):
        first_capability.fence(foreign_epoch)

    first_receipt = first_capability.fence(first_epoch)
    with pytest.raises(ValueError, match="foreign, stale, or reused"):
        first_capability.fence(first_epoch)
    first_receipt.consume_for_subject(
        first_capability,
        first_binding,
        subject=first_subject,
        consumer="test-stale-epoch-rejection",
    )
    foreign_receipt = second_capability.fence(foreign_epoch)
    foreign_receipt.consume_for_subject(
        second_capability,
        foreign_binding,
        subject=foreign_subject,
        consumer="test-foreign-capability-settlement",
    )


def test_sample_receipt_requires_its_exact_prelaunch_subject_and_binding() -> None:
    native_ops = _SynchronousCpuNativeOps()
    capability = _prepare_cpu_capability(native_ops)
    foreign_capability = _prepare_cpu_capability(native_ops)
    subject = _StableCompletionSubject(_SUBJECT_GENERATION_0)
    lookalike_subject = _StableCompletionSubject(_SUBJECT_GENERATION_0)
    binding = prepare_paper_kinetic_completion_subject_binding(
        capability,
        subject,
        kind="test-sample-composite",
        subject_generation_digest=_SUBJECT_GENERATION_0,
    )
    foreign_subject = _StableCompletionSubject(_SUBJECT_GENERATION_1)
    foreign_binding = prepare_paper_kinetic_completion_subject_binding(
        foreign_capability,
        foreign_subject,
        kind="test-sample-composite",
        subject_generation_digest=_SUBJECT_GENERATION_1,
    )

    with pytest.raises(ValueError, match="exact prelaunch subject"):
        capability.register_launch(
            stage="sample-completion",
            launch_generation_digest=_LAUNCH_GENERATION_0,
        )
    with pytest.raises(ValueError, match="subject binding is foreign"):
        capability.register_launch(
            stage="sample-completion",
            launch_generation_digest=_LAUNCH_GENERATION_0,
            subject_binding=foreign_binding,
        )

    epoch = capability.register_launch(
        stage="sample-completion",
        launch_generation_digest=_LAUNCH_GENERATION_0,
        subject_binding=binding,
    )
    receipt = capability.fence(epoch)
    with pytest.raises(ValueError, match="foreign or stale"):
        receipt.consume_for_subject(
            capability,
            binding,
            subject=lookalike_subject,
            consumer="forbidden-lookalike-subject",
        )
    with pytest.raises(ValueError, match="foreign subject binding"):
        receipt.consume_for_subject(
            capability,
            foreign_binding,
            subject=foreign_subject,
            consumer="forbidden-foreign-binding",
        )
    with pytest.raises(ValueError, match="requires consume_for_subject"):
        receipt.consume_for(
            capability,
            stage="sample-completion",
            launch_generation_digest=_LAUNCH_GENERATION_0,
            fence_sequence=1,
            consumer="forbidden-unbound-consumption",
        )

    assert receipt.consumed is False
    assert capability.outstanding_receipt_identity == id(receipt)
    receipt.consume_for_subject(
        capability,
        binding,
        subject=subject,
        consumer="exact-subject-consumption",
    )
    assert receipt.consumed is True


def test_epoch_and_receipt_strongly_retain_exact_subject_and_reject_binding_clone() -> None:
    capability = _prepare_cpu_capability(_SynchronousCpuNativeOps())
    subject, binding, epoch = _register_sample_epoch(
        capability,
        launch_generation_digest=_LAUNCH_GENERATION_0,
    )
    subject_ref = weakref.ref(subject)
    binding_identity = id(binding)
    del subject, binding
    gc.collect()

    assert subject_ref() is not None
    assert id(epoch.subject_binding) == binding_identity
    epoch.subject_binding.assert_for(capability, subject=subject_ref())

    receipt = capability.fence(epoch)
    del epoch
    gc.collect()
    retained_subject = subject_ref()
    retained_binding = receipt.subject_binding
    assert retained_subject is not None
    assert retained_binding is not None
    assert id(retained_binding) == binding_identity
    cloned_binding = replace(retained_binding)
    with pytest.raises(ValueError, match="subject binding changed"):
        cloned_binding.assert_current()
    with pytest.raises(ValueError, match="foreign subject binding"):
        receipt.consume_for_subject(
            capability,
            cloned_binding,
            subject=retained_subject,
            consumer="forbidden-cloned-binding",
        )

    assert receipt.consumed is False
    receipt.consume_for_subject(
        capability,
        retained_binding,
        subject=retained_subject,
        consumer="retained-exact-subject",
    )
    assert receipt.consumed is True


def test_receipt_rejects_wrong_stage_launch_digest_and_sequence() -> None:
    capability = _prepare_cpu_capability(_SynchronousCpuNativeOps())
    subject, binding, epoch = _register_sample_epoch(
        capability,
        launch_generation_digest=_LAUNCH_GENERATION_0,
    )
    receipt = capability.fence(epoch)

    for assertion in (
        {
            "stage": "reverse-completion",
            "launch_generation_digest": _LAUNCH_GENERATION_0,
            "fence_sequence": 1,
        },
        {
            "stage": "sample-completion",
            "launch_generation_digest": _LAUNCH_GENERATION_1,
            "fence_sequence": 1,
        },
        {
            "stage": "sample-completion",
            "launch_generation_digest": _LAUNCH_GENERATION_0,
            "fence_sequence": 2,
        },
    ):
        with pytest.raises(ValueError, match="foreign or consumed"):
            receipt.assert_for(capability, **assertion)

    receipt.consume_for_subject(
        capability,
        binding,
        subject=subject,
        consumer="test-exact-receipt-relation",
    )


def test_dataclass_clones_cannot_replace_exact_capability_or_receipt_identity() -> None:
    capability = _prepare_cpu_capability(_SynchronousCpuNativeOps())
    subject, binding, epoch = _register_sample_epoch(
        capability,
        launch_generation_digest=_LAUNCH_GENERATION_0,
    )
    receipt = capability.fence(epoch)
    capability_clone = replace(capability)
    receipt_clone = replace(receipt)

    with pytest.raises(ValueError, match="capability changed"):
        capability_clone.assert_current(require_healthy=False)
    with pytest.raises(ValueError, match="receipt changed"):
        receipt_clone.assert_current()
    with pytest.raises(ValueError, match="receipt changed"):
        receipt_clone.consume_for_subject(
            capability,
            binding,
            subject=subject,
            consumer="forbidden-cloned-receipt",
        )
    with pytest.raises(ValueError, match="capability changed"):
        receipt.consume_for_subject(
            capability_clone,
            binding,
            subject=subject,
            consumer="forbidden-cloned-capability",
        )

    assert receipt.consumed is False
    assert capability.outstanding_receipt_identity == id(receipt)
    receipt.consume_for_subject(
        capability,
        binding,
        subject=subject,
        consumer="original-identity-still-consumable",
    )
    assert receipt.consumed is True
    assert capability.outstanding_receipt_identity is None


def test_failed_owned_fence_poisoning_retains_exact_registered_epoch(
    monkeypatch,
) -> None:
    capability = _prepare_cpu_capability(_SynchronousCpuNativeOps())
    _subject, _binding, epoch = _register_sample_epoch(
        capability,
        launch_generation_digest=_LAUNCH_GENERATION_0,
    )

    def fail_owned_fence(self) -> None:
        assert self is capability
        raise RuntimeError("synthetic owned synchronizer failure")

    monkeypatch.setattr(
        PaperKineticSealedCompletionFence,
        "_synchronize_bound_device_wide",
        fail_owned_fence,
    )
    with pytest.raises(PaperKineticCompletionUnknownError) as failure:
        capability.fence(epoch)

    assert isinstance(failure.value.__cause__, RuntimeError)
    assert "synthetic owned synchronizer failure" in str(failure.value.__cause__)
    assert capability.completion_unknown is True
    assert capability.poisoned is True
    assert capability.fence_attempt_count == 1
    assert capability.successful_fence_count == 0
    assert capability.consumed_fence_count == 0
    assert capability.registered_launch_epoch is epoch
    assert epoch.fenced is False
    with pytest.raises(RuntimeError, match="not healthy"):
        capability.register_launch(
            stage="forbidden-post-failure-epoch",
            launch_generation_digest=_LAUNCH_GENERATION_1,
        )
    with pytest.raises(RuntimeError, match="not healthy"):
        capability.fence(epoch)
    capability.assert_current(require_healthy=False)


def test_foreign_thread_cannot_register_a_launch_epoch() -> None:
    capability = _prepare_cpu_capability(_SynchronousCpuNativeOps())
    failures: list[BaseException] = []

    def invoke_from_foreign_thread() -> None:
        try:
            epoch = capability.register_launch(
                stage="foreign-thread",
                launch_generation_digest=_LAUNCH_GENERATION_0,
            )
            capability.fence(epoch)
        except BaseException as error:
            failures.append(error)

    worker = threading.Thread(target=invoke_from_foreign_thread)
    worker.start()
    worker.join()

    assert len(failures) == 1
    assert isinstance(failures[0], RuntimeError)
    assert "foreign thread" in str(failures[0])
    assert capability.fence_attempt_count == 0
    assert capability.successful_fence_count == 0
    capability.assert_current()


@pytest.mark.parametrize("device", ("mps", "cuda:0"))
def test_accelerator_capability_minting_is_fail_closed(device: str) -> None:
    with pytest.raises(RuntimeError, match="source-defined but not promotable"):
        prepare_paper_kinetic_sealed_completion_fence(
            _SynchronousCpuNativeOps(),
            device=device,
            owner_generation_digest=_OWNER_GENERATION,
        )


def test_capability_binding_and_receipt_subclassing_is_rejected() -> None:
    with pytest.raises(TypeError, match="cannot be subclassed"):

        class _ForgedCapability(PaperKineticSealedCompletionFence):
            pass

    with pytest.raises(TypeError, match="cannot be subclassed"):

        class _ForgedReceipt(PaperKineticCompletionFenceReceipt):
            pass

    with pytest.raises(TypeError, match="cannot be subclassed"):

        class _ForgedSubjectBinding(PaperKineticCompletionSubjectBinding):
            pass

    with pytest.raises(TypeError, match="cannot be subclassed"):

        class _ForgedLaunchEpoch(PaperKineticCompletionLaunchEpoch):
            pass
