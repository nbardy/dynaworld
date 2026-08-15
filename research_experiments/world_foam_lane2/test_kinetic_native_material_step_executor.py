from __future__ import annotations

import gc
import weakref
from dataclasses import replace

import pytest
import torch
import kinetic_native_material_step_executor as executor_module
from kinetic_native_equal_rank_geometry_reduction import (
    kinetic_native_equal_rank_vjp_provenance_id,
    reduce_kinetic_native_equal_rank_geometry_vjp,
)
from kinetic_native_material_step_executor import (
    EXECUTOR_STATUS,
    KineticNativePendingSampleLaunchCompletion,
    prepare_kinetic_native_material_step_executor,
)
from kinetic_sealed_completion_fence import (
    prepare_paper_kinetic_completion_subject_binding,
    prepare_paper_kinetic_sealed_completion_fence,
)
from paper_kinetic_ragged_sample_plan import (
    iter_paper_kinetic_row_ragged_sample_blocks,
)
from test_kinetic_ragged_paper_step_cpu_fake_native import (
    _adapted,
    _compiled_case,
    _direct_oracle,
)


_SAMPLE_COMPLETION_FENCE_PROVENANCE = (
    "cpu-contract-double/sample-completion-fence-v1"
)
_SEALED_OWNER_GENERATION = "7" * 64
_SEALED_SUBJECT_GENERATION = "8" * 64


class _StableSampleCompositeSubject:
    def __init__(self, generation_digest: str) -> None:
        self.generation_digest = generation_digest


def _settle_sample(session, lifetime):
    receipt = session.settle_sample_accumulate(
        lifetime,
        device_completion_fence=lambda: None,
        device_completion_fence_provenance=(
            _SAMPLE_COMPLETION_FENCE_PROVENANCE
        ),
    )
    receipt.assert_current()
    return receipt


def _first_sample_launch_fixture():
    compiled = _compiled_case()
    adapted = _adapted(3)
    executor = prepare_kinetic_native_material_step_executor(
        compiled.native_ops,
        tuple(
            (runtime, compiled.sampler)
            for runtime in compiled.lane.runtimes
        ),
        backend_provenance="cpu-contract-double/sample-lifetime",
    )
    session = executor.begin_step(
        step_generation_id="sample-lifetime-step",
        requested_observation_count=(
            adapted.pixel_count * adapted.observation_count
        ),
    )
    group = adapted.groups[0]
    staged = group.stage_targets(
        track_start=0,
        track_end=adapted.pixel_count,
        sample_start=0,
        sample_end=group.observation_count,
    )
    sample_block = next(
        iter_paper_kinetic_row_ragged_sample_blocks(
            compiled.sampler,
            staged,
            loss_normalization_id=adapted.loss_normalization_id,
            maximum_samples_per_launch=1,
        )
    )
    runtime = next(
        value
        for value in compiled.lane.runtimes
        if value.payload.block.generation_digest
        == sample_block.native_block_generation_digest
    )
    token = session.launch_node_forward(
        runtime,
        compiled.global_site_rgba_f32.index_select(
            0,
            runtime.source_site_ids_i64,
        ).contiguous(),
    )
    return {
        "compiled": compiled,
        "session": session,
        "token": token,
        "sample_block": sample_block,
        "launch_kwargs": {
            "sampler": compiled.sampler,
            "background_rgb_f32": compiled.background_rgb_f32,
            "loss_f32": torch.zeros((1,), dtype=torch.float32),
            "grad_node_chart_f32": torch.zeros_like(
                token.world.node_chart_f32
            ),
            "cone_diagnostic_i32": torch.zeros((3,), dtype=torch.int32),
        },
    }


def _settle_sample_sealed(case, lifetime):
    capability = prepare_paper_kinetic_sealed_completion_fence(
        case["compiled"].native_ops,
        device=lifetime.sample_block.device,
        owner_generation_digest=_SEALED_OWNER_GENERATION,
    )
    subject = _StableSampleCompositeSubject(_SEALED_SUBJECT_GENERATION)
    binding = prepare_paper_kinetic_completion_subject_binding(
        capability,
        subject,
        kind="test-native-sample-composite",
        subject_generation_digest=_SEALED_SUBJECT_GENERATION,
    )
    epoch = capability.register_launch(
        stage="sample-completion",
        launch_generation_digest=lifetime.generation_digest,
        subject_binding=binding,
    )
    pending = case["session"].settle_sample_accumulate(
        lifetime,
        sealed_completion_fence=capability,
        sealed_completion_launch_epoch=epoch,
    )
    assert type(pending) is KineticNativePendingSampleLaunchCompletion
    return capability, subject, binding, epoch, pending


def _run_executor(
    frame_count: int,
    *,
    maximum_samples_per_launch: int = 2,
    reverse_mode: str = "material_only",
    dirty_accumulators: bool = False,
):
    if reverse_mode not in {"material_only", "full_geometry"}:
        raise ValueError("test reverse_mode is invalid")
    compiled = _compiled_case()
    adapted = _adapted(frame_count)
    runtimes = compiled.lane.runtimes
    executor = prepare_kinetic_native_material_step_executor(
        compiled.native_ops,
        tuple((runtime, compiled.sampler) for runtime in runtimes),
        backend_provenance="cpu-contract-double/exact-production-op-object",
    )
    session = executor.begin_step(
        step_generation_id=adapted.loss_normalization_id,
        requested_observation_count=(
            adapted.pixel_count * adapted.observation_count
        ),
    )
    runtime_by_digest = {
        runtime.payload.block.generation_digest: runtime for runtime in runtimes
    }
    tokens = {}
    node_bars = {}
    block_losses = {}
    full_geometry_sample_count = 0
    full_geometry_length_bar_bytes = 0
    full_geometry_fenced_reduction_count = 0
    loss = torch.zeros((1,), dtype=torch.float32)
    global_bar = torch.zeros_like(compiled.global_site_rgba_f32)
    cone = torch.zeros((3,), dtype=torch.int32)
    group = adapted.groups[0]
    staged = group.stage_targets(
        track_start=0,
        track_end=adapted.pixel_count,
        sample_start=0,
        sample_end=group.observation_count,
    )
    for sample_block in iter_paper_kinetic_row_ragged_sample_blocks(
        compiled.sampler,
        staged,
        loss_normalization_id=adapted.loss_normalization_id,
        maximum_samples_per_launch=maximum_samples_per_launch,
    ):
        digest = sample_block.native_block_generation_digest
        if digest not in tokens:
            runtime = runtime_by_digest[digest]
            compact_material = compiled.global_site_rgba_f32.index_select(
                0,
                runtime.source_site_ids_i64,
            ).contiguous()
            token = session.launch_node_forward(runtime, compact_material)
            tokens[digest] = token
            fill = 7.0 if dirty_accumulators else 0.0
            node_bars[digest] = torch.full_like(
                token.world.node_chart_f32,
                fill,
            )
            block_losses[digest] = torch.full(
                (1,),
                fill,
                dtype=torch.float32,
            )
        lifetime = session.launch_sample_accumulate(
            tokens[digest],
            sample_block,
            sampler=compiled.sampler,
            background_rgb_f32=compiled.background_rgb_f32,
            loss_f32=block_losses[digest],
            grad_node_chart_f32=node_bars[digest],
            cone_diagnostic_i32=cone,
        )
        _settle_sample(session, lifetime)
    for runtime in runtimes:
        digest = runtime.payload.block.generation_digest
        if digest in tokens:
            reverse = (
                session.launch_material_vjp
                if reverse_mode == "material_only"
                else session.launch_full_geometry_vjp
            )
            result = reverse(
                tokens[digest],
                node_bars[digest],
                compact_grad_site_rgba_f32=torch.empty(
                    (runtime.compact_site_count, 4),
                    dtype=torch.float32,
                ),
                global_grad_site_rgba_f32=global_bar,
            )
            if reverse_mode == "full_geometry":
                full_geometry_sample_count += result.reduced_sample_count
                full_geometry_length_bar_bytes += (
                    result.native_length_bar_tensor_bytes
                )
                native_vjp = result.native_vjp_result
                native_provenance = kinetic_native_equal_rank_vjp_provenance_id(
                    native_vjp
                )
                geometry = reduce_kinetic_native_equal_rank_geometry_vjp(
                    native_vjp,
                    compiled.sampler,
                    expected_native_vjp_provenance_id=native_provenance,
                    device_completion_fence=lambda: None,
                    device_completion_fence_provenance=(
                        "cpu-contract-double/executor-reduction-fence-v1"
                    ),
                    maximum_bridge_visible_peak_logical_tensor_bytes=10_000_000,
                )
                completion = session.consume_full_geometry_vjp_execution(
                    result,
                    geometry_reduction=geometry,
                    expected_device_completion_fence_provenance=(
                        "cpu-contract-double/executor-reduction-fence-v1"
                    ),
                )
                completion.assert_current()
                assert not completion.global_accumulation_proven
                assert (
                    completion.completion_semantics
                    == "fenced_and_reduced_not_globally_committed"
                )
                full_geometry_fenced_reduction_count += 1
                with pytest.raises(
                    ValueError,
                    match="no longer owns its native result",
                ):
                    result.assert_current(session)
            loss.add_(block_losses[digest])
    telemetry = session.seal()
    if reverse_mode == "full_geometry":
        assert full_geometry_sample_count == adapted.pixel_count * adapted.observation_count
        assert full_geometry_length_bar_bytes == telemetry.native_length_bar_tensor_bytes
        assert (
            full_geometry_fenced_reduction_count
            == telemetry.native_full_geometry_fenced_reduction_count
        )
    return compiled, adapted, executor, telemetry, loss, global_bar


def test_one_native_ops_object_matches_oracle_and_seals_exact_block_counts() -> None:
    compiled, adapted, executor, telemetry, loss, global_bar = _run_executor(9)
    oracle_loss, oracle_bar = _direct_oracle(compiled, adapted)

    torch.testing.assert_close(loss, oracle_loss, rtol=2.0e-5, atol=2.0e-6)
    torch.testing.assert_close(global_bar, oracle_bar, rtol=3.0e-5, atol=3.0e-6)
    telemetry.assert_current()
    assert executor.runtime_status == telemetry.runtime_status == EXECUTOR_STATUS
    assert telemetry.active_native_block_count == len(compiled.lane.runtimes)
    assert (
        telemetry.native_node_forward_launch_count
        == telemetry.native_material_word_vjp_launch_count
        == telemetry.active_native_block_count
    )
    assert (
        telemetry.native_sample_prepare_count
        == telemetry.native_sample_launch_count
        == telemetry.native_sample_completion_fence_count
    )
    assert telemetry.maximum_simultaneous_sample_lifetime_count == 1
    assert telemetry.outstanding_sample_lifetime_count_at_seal == 0
    assert not telemetry.sample_lifetime_history_retained
    assert telemetry.sample_lifetime_additional_logical_tensor_bytes == 0
    assert not telemetry.sample_lifetime_python_heap_bytes_measured
    assert telemetry.streamed_sample_count == adapted.pixel_count * adapted.observation_count
    assert telemetry.global_loss_element_count == adapted.pixel_count * adapted.observation_count * 3
    assert telemetry.loss_scale == 1.0 / float(telemetry.global_loss_element_count)
    assert telemetry.loss_normalization_id == adapted.loss_normalization_id
    assert telemetry.native_full_geometry_vjp_launch_count == 0
    assert telemetry.native_full_geometry_fenced_reduction_count == 0
    assert telemetry.native_length_bar_tensor_bytes == 0
    assert telemetry.reverse_mode == "material_only"
    assert telemetry.exactly_one_material_vjp_per_active_block
    assert not telemetry.exactly_one_full_geometry_vjp_per_active_block
    assert not telemetry.geometry_vjp_exposed
    assert compiled.native_ops.forward_calls == telemetry.native_node_forward_launch_count
    assert compiled.native_ops.material_vjp_calls == telemetry.native_material_word_vjp_launch_count
    assert compiled.native_ops.vjp_calls == 0
    assert compiled.native_ops.sample_prepare_calls == telemetry.native_sample_prepare_count
    assert compiled.native_ops.sample_launch_calls == telemetry.native_sample_launch_count
    assert all(
        block.native_node_forward_launch_count
        == block.native_material_word_vjp_launch_count
        == 1
        and block.native_sample_prepare_count
        == block.native_sample_launch_count
        == block.native_sample_completion_fence_count
        for block in telemetry.blocks
    )

    stale = replace(
        telemetry,
        streamed_sample_count=telemetry.streamed_sample_count + 1,
    )
    with pytest.raises(ValueError, match="telemetry contract changed"):
        stale.assert_current()


def test_full_geometry_mode_uses_same_sealed_sample_lifecycle_and_material_oracle() -> None:
    compiled, adapted, _executor, telemetry, loss, global_bar = _run_executor(
        9,
        reverse_mode="full_geometry",
    )
    oracle_loss, oracle_bar = _direct_oracle(compiled, adapted)

    torch.testing.assert_close(loss, oracle_loss, rtol=2.0e-5, atol=2.0e-6)
    torch.testing.assert_close(global_bar, oracle_bar, rtol=3.0e-5, atol=3.0e-6)
    telemetry.assert_current()
    assert telemetry.reverse_mode == "full_geometry"
    assert telemetry.native_material_word_vjp_launch_count == 0
    assert (
        telemetry.native_full_geometry_vjp_launch_count
        == telemetry.active_native_block_count
    )
    assert telemetry.native_length_bar_tensor_bytes > 0
    assert (
        telemetry.native_full_geometry_fenced_reduction_count
        == telemetry.active_native_block_count
    )
    assert not telemetry.full_geometry_global_accumulation_proven
    assert (
        telemetry.full_geometry_completion_semantics
        == "fenced_and_reduced_not_globally_committed"
    )
    assert not telemetry.exactly_one_material_vjp_per_active_block
    assert telemetry.exactly_one_full_geometry_vjp_per_active_block
    assert telemetry.exactly_one_reverse_per_active_block
    assert telemetry.geometry_vjp_exposed
    assert compiled.native_ops.material_vjp_calls == 0
    assert compiled.native_ops.vjp_calls == telemetry.active_native_block_count
    assert all(
        block.reverse_mode == "full_geometry"
        and block.native_material_word_vjp_launch_count == 0
        and block.native_full_geometry_vjp_launch_count == 1
        and block.native_full_geometry_fenced_reduction_count == 1
        and block.native_length_bar_tensor_bytes > 0
        and len(block.full_geometry_fenced_reduction_generation_digest) == 64
        and len(block.geometry_reduction_generation_digest) == 64
        and block.reduction_completion_fence_provenance
        == "cpu-contract-double/executor-reduction-fence-v1"
        for block in telemetry.blocks
    )


@pytest.mark.parametrize("reverse_mode", ("material_only", "full_geometry"))
def test_executor_zero_initializes_each_block_accumulator(reverse_mode: str) -> None:
    compiled, adapted, _executor, _telemetry, loss, global_bar = _run_executor(
        9,
        reverse_mode=reverse_mode,
        dirty_accumulators=True,
    )
    oracle_loss, oracle_bar = _direct_oracle(compiled, adapted)

    torch.testing.assert_close(loss, oracle_loss, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(global_bar, oracle_bar, rtol=3e-5, atol=3e-6)


def test_full_geometry_f_density_changes_samples_not_word_or_length_bar_work() -> None:
    _sparse_compiled, _sparse_batch, _executor, sparse, _loss, _bar = (
        _run_executor(5, reverse_mode="full_geometry")
    )
    _dense_compiled, _dense_batch, _executor, dense, _loss, _bar = (
        _run_executor(41, reverse_mode="full_geometry")
    )

    assert dense.native_sample_launch_count > sparse.native_sample_launch_count
    assert (
        dense.native_node_forward_launch_count
        == sparse.native_node_forward_launch_count
    )
    assert (
        dense.native_full_geometry_vjp_launch_count
        == sparse.native_full_geometry_vjp_launch_count
    )
    assert dense.native_length_bar_tensor_bytes == sparse.native_length_bar_tensor_bytes


def test_requested_f_changes_only_streamed_sample_launches_not_word_launches() -> None:
    _sparse_compiled, sparse_batch, sparse_executor, sparse, _loss, _bar = (
        _run_executor(5)
    )
    _dense_compiled, dense_batch, dense_executor, dense, _loss, _bar = (
        _run_executor(41)
    )

    # The executor ledger counts flat pixel-ray observations.  ``F`` is the
    # per-track temporal density, so this fixture requests ``P * F`` samples.
    assert (
        sparse.requested_observation_count
        == sparse_batch.pixel_count * sparse_batch.observation_count
    )
    assert (
        dense.requested_observation_count
        == dense_batch.pixel_count * dense_batch.observation_count
    )
    assert sparse.active_native_block_count == dense.active_native_block_count
    assert (
        sparse.native_node_forward_launch_count
        == dense.native_node_forward_launch_count
    )
    assert (
        sparse.native_material_word_vjp_launch_count
        == dense.native_material_word_vjp_launch_count
    )
    assert dense.native_sample_launch_count > sparse.native_sample_launch_count
    assert dense.streamed_sample_count == dense_batch.pixel_count * 41
    assert sparse.streamed_sample_count == sparse_batch.pixel_count * 5
    sparse_memory = sparse_executor.memory_accounting(5)
    dense_memory = dense_executor.memory_accounting(41)
    assert (
        sparse_memory.summed_runtime_unique_retained_tensor_bytes_upper_bound
        == dense_memory.summed_runtime_unique_retained_tensor_bytes_upper_bound
    )
    assert sparse_memory.executor_owned_persistent_tensor_bytes == 0
    assert dense_memory.runtime_tensor_copy_bytes_allocated_by_executor == 0
    assert dense_memory.retained_sampler_count == 0
    assert dense_memory.persistent_frame_tensor_bytes == 0
    assert dense_memory.persistent_sample_tensor_bytes == 0
    assert dense_memory.persistent_target_tensor_bytes == 0
    assert dense_memory.persistent_prediction_tensor_bytes == 0
    assert not dense_memory.requested_observation_count_affects_retained_bytes


def test_distinct_loss_views_sharing_storage_cannot_cross_native_blocks() -> None:
    compiled = _compiled_case()
    adapted = _adapted(9)
    runtimes = {
        runtime.payload.block.generation_digest: runtime
        for runtime in compiled.lane.runtimes
    }
    executor = prepare_kinetic_native_material_step_executor(
        compiled.native_ops,
        tuple((runtime, compiled.sampler) for runtime in compiled.lane.runtimes),
        backend_provenance="cpu-contract-double/exact-production-op-object",
    )
    session = executor.begin_step(
        step_generation_id="shared-loss-storage",
        requested_observation_count=adapted.pixel_count * adapted.observation_count,
    )
    group = adapted.groups[0]
    staged = group.stage_targets(
        track_start=0,
        track_end=adapted.pixel_count,
        sample_start=0,
        sample_end=group.observation_count,
    )
    first_by_digest = {}
    for block in iter_paper_kinetic_row_ragged_sample_blocks(
        compiled.sampler,
        staged,
        loss_normalization_id=adapted.loss_normalization_id,
        maximum_samples_per_launch=2,
    ):
        first_by_digest.setdefault(block.native_block_generation_digest, block)
        if len(first_by_digest) == 2:
            break
    assert len(first_by_digest) == 2

    shared_storage = torch.zeros((2,), dtype=torch.float32)
    loss_views = (shared_storage[:1], shared_storage[1:])
    for index, (digest, block) in enumerate(first_by_digest.items()):
        runtime = runtimes[digest]
        token = session.launch_node_forward(
            runtime,
            compiled.global_site_rgba_f32.index_select(
                0,
                runtime.source_site_ids_i64,
            ).contiguous(),
        )
        kwargs = dict(
            sampler=compiled.sampler,
            background_rgb_f32=compiled.background_rgb_f32,
            loss_f32=loss_views[index],
            grad_node_chart_f32=torch.zeros_like(token.world.node_chart_f32),
            cone_diagnostic_i32=torch.zeros((3,), dtype=torch.int32),
        )
        if index == 0:
            lifetime = session.launch_sample_accumulate(token, block, **kwargs)
            _settle_sample(session, lifetime)
        else:
            with pytest.raises(ValueError, match="share one loss accumulator"):
                session.launch_sample_accumulate(token, block, **kwargs)
    with pytest.raises(ValueError, match="poisoned"):
        session.seal()


def test_foreign_generation_duplicate_forward_and_incomplete_step_fail_closed() -> None:
    compiled = _compiled_case()
    runtime = compiled.lane.runtimes[0]
    executor = prepare_kinetic_native_material_step_executor(
        compiled.native_ops,
        tuple((value, compiled.sampler) for value in compiled.lane.runtimes),
        backend_provenance="cpu-contract-double/exact-production-op-object",
    )
    session = executor.begin_step(
        step_generation_id="step-a",
        requested_observation_count=3,
    )
    compact = compiled.global_site_rgba_f32.index_select(
        0,
        runtime.source_site_ids_i64,
    ).contiguous()
    token = session.launch_node_forward(runtime, compact)
    before = compiled.native_ops.forward_calls
    with pytest.raises(ValueError, match="already launched"):
        session.launch_node_forward(runtime, compact)
    assert compiled.native_ops.forward_calls == before
    with pytest.raises(ValueError, match="poisoned"):
        session.seal()
    state = session._states[runtime.generation_id]
    assert state.token is token
    failed_fence_calls = 0

    def failing_abort_fence() -> None:
        nonlocal failed_fence_calls
        failed_fence_calls += 1
        raise RuntimeError("injected abort fence failure")

    with pytest.raises(RuntimeError, match="injected abort fence failure"):
        session.abort(
            device_completion_fence=failing_abort_fence,
            device_completion_fence_provenance="cpu-test-failing-abort-fence-v1",
        )
    assert failed_fence_calls == 1
    assert session._abort_completion_fence_call_count == 1
    assert session._failed and not session._abort_release_completed
    assert state.token is token
    completed_fence_calls = 0

    def completed_abort_fence() -> None:
        nonlocal completed_fence_calls
        completed_fence_calls += 1

    session.abort(
        device_completion_fence=completed_abort_fence,
        device_completion_fence_provenance="cpu-test-completed-abort-fence-v1",
    )
    assert completed_fence_calls == 1
    assert session._abort_completion_fence_call_count == 2
    assert session._abort_release_completed and state.token is None

    foreign = executor.begin_step(
        step_generation_id="step-b",
        requested_observation_count=3,
    )
    with pytest.raises(ValueError, match="generation/provenance changed"):
        foreign.launch_material_vjp(
            token,
            torch.zeros_like(token.world.node_chart_f32),
            compact_grad_site_rgba_f32=torch.empty(
                (runtime.compact_site_count, 4),
                dtype=torch.float32,
            ),
        )
    with pytest.raises(ValueError, match="poisoned"):
        foreign.seal()


def test_stale_sample_and_native_callable_identity_fail_before_native_launch() -> None:
    compiled = _compiled_case()
    adapted = _adapted(3)
    executor = prepare_kinetic_native_material_step_executor(
        compiled.native_ops,
        tuple((runtime, compiled.sampler) for runtime in compiled.lane.runtimes),
        backend_provenance="cpu-contract-double/exact-production-op-object",
    )
    group = adapted.groups[0]
    staged = group.stage_targets(
        track_start=0,
        track_end=adapted.pixel_count,
        sample_start=0,
        sample_end=group.observation_count,
    )
    block = next(
        iter_paper_kinetic_row_ragged_sample_blocks(
            compiled.sampler,
            staged,
            loss_normalization_id=adapted.loss_normalization_id,
            maximum_samples_per_launch=2,
        )
    )
    runtime = next(
        value
        for value in compiled.lane.runtimes
        if value.payload.block.generation_digest
        == block.native_block_generation_digest
    )
    session = executor.begin_step(
        step_generation_id="stale-sample",
        requested_observation_count=3,
    )
    token = session.launch_node_forward(
        runtime,
        compiled.global_site_rgba_f32.index_select(
            0,
            runtime.source_site_ids_i64,
        ).contiguous(),
    )
    stale = replace(block, sampler_generation_digest="stale")
    before_prepare = compiled.native_ops.sample_prepare_calls
    before_launch = compiled.native_ops.sample_launch_calls
    with pytest.raises(ValueError, match="stale sampler"):
        session.launch_sample_accumulate(
            token,
            stale,
            sampler=compiled.sampler,
            background_rgb_f32=compiled.background_rgb_f32,
            loss_f32=torch.zeros((1,), dtype=torch.float32),
            grad_node_chart_f32=torch.zeros_like(token.world.node_chart_f32),
            cone_diagnostic_i32=torch.zeros((3,), dtype=torch.int32),
        )
    assert compiled.native_ops.sample_prepare_calls == before_prepare
    assert compiled.native_ops.sample_launch_calls == before_launch

    compiled.native_ops.prepare_kinetic_ragged_p0_lie_sample_block = lambda *args, **kwargs: None
    with pytest.raises(ValueError, match="ABI/generation contract changed"):
        executor.assert_current()


def test_sample_lifetime_roots_prepared_payload_until_fence_then_releases() -> None:
    case = _first_sample_launch_fixture()
    lifetime = case["session"].launch_sample_accumulate(
        case["token"],
        case["sample_block"],
        **case["launch_kwargs"],
    )
    prepared_ref = weakref.ref(lifetime.prepared_payload)
    fence_observations = []

    def completion_fence() -> None:
        lifetime.assert_current(case["session"])
        fence_observations.append(
            prepared_ref() is lifetime.prepared_payload
            and not lifetime.consumed
        )

    receipt = case["session"].settle_sample_accumulate(
        lifetime,
        device_completion_fence=completion_fence,
        device_completion_fence_provenance=(
            _SAMPLE_COMPLETION_FENCE_PROVENANCE
        ),
    )
    receipt.assert_current()
    assert fence_observations == [True]
    assert lifetime.consumed and lifetime.phase == "released"
    assert all(
        value is None
        for value in (
            lifetime.prepared_payload,
            lifetime.sample_block,
            lifetime.world_token,
            lifetime.background_rgb_f32,
            lifetime.loss_f32,
            lifetime.grad_node_chart_f32,
            lifetime.cone_diagnostic_i32,
        )
    )
    gc.collect()
    assert prepared_ref() is None


def test_sealed_sample_settlement_retains_one_pending_payload_then_commits_once() -> None:
    case = _first_sample_launch_fixture()
    lifetime = case["session"].launch_sample_accumulate(
        case["token"],
        case["sample_block"],
        **case["launch_kwargs"],
    )
    prepared = lifetime.prepared_payload
    capability, subject, _binding, _epoch, pending = _settle_sample_sealed(
        case,
        lifetime,
    )
    sealed_receipt = pending.assert_exact_sealed_receipt_relation(
        case["session"],
        capability,
        subject=subject,
    )

    assert sealed_receipt is pending.sealed_completion_receipt
    assert sealed_receipt.consumed is False
    assert capability.outstanding_receipt_identity == id(sealed_receipt)
    assert case["session"]._outstanding_sample_lifetime is lifetime
    assert case["session"]._pending_sample_completion is pending
    assert lifetime.prepared_payload is prepared
    assert all(
        value is not None
        for value in (
            lifetime.sample_block,
            lifetime.world_token,
            lifetime.background_rgb_f32,
            lifetime.loss_f32,
            lifetime.grad_node_chart_f32,
            lifetime.cone_diagnostic_i32,
        )
    )
    before = (
        case["compiled"].native_ops.sample_prepare_calls,
        case["compiled"].native_ops.sample_launch_calls,
    )
    with pytest.raises(RuntimeError, match="must settle"):
        case["session"].launch_sample_accumulate(
            case["token"],
            case["sample_block"],
            **case["launch_kwargs"],
        )
    assert (
        case["compiled"].native_ops.sample_prepare_calls,
        case["compiled"].native_ops.sample_launch_calls,
    ) == before

    case["session"].assert_pending_sample_accumulate_releasable(
        pending,
        capability,
        subject=subject,
    )
    commit_plan = pending.consume_sealed_receipt_for_outer_composite(
        case["session"],
        capability,
        subject=subject,
        consumer="test-exact-outer-composite",
    )
    with pytest.raises(ValueError, match="changed or is foreign"):
        pending.consume_sealed_receipt_for_outer_composite(
            case["session"],
            capability,
            subject=subject,
            consumer="forbidden-second-consumption",
        )
    cloned_commit_plan = replace(commit_plan)
    with pytest.raises(ValueError, match="foreign, unauthorized, or consumed"):
        case["session"].commit_sample_accumulate_after_consumed_sealed_receipt(
            cloned_commit_plan
        )
    assert lifetime.prepared_payload is prepared
    completion = (
        case["session"].commit_sample_accumulate_after_consumed_sealed_receipt(
            commit_plan
        )
    )
    completion.assert_current()

    assert sealed_receipt.consumed is True
    assert capability.outstanding_receipt_identity is None
    assert case["session"]._outstanding_sample_lifetime is None
    assert case["session"]._pending_sample_completion is None
    assert lifetime.consumed and lifetime.phase == "released"
    assert all(
        value is None
        for value in (
            lifetime.prepared_payload,
            lifetime.sample_block,
            lifetime.world_token,
            lifetime.background_rgb_f32,
            lifetime.loss_f32,
            lifetime.grad_node_chart_f32,
            lifetime.cone_diagnostic_i32,
        )
    )
    with pytest.raises(ValueError, match="foreign, unauthorized, or consumed"):
        case["session"].commit_sample_accumulate_after_consumed_sealed_receipt(
            commit_plan
        )


def test_outer_revalidation_rejects_foreign_subject_without_consuming_or_releasing() -> None:
    case = _first_sample_launch_fixture()
    lifetime = case["session"].launch_sample_accumulate(
        case["token"],
        case["sample_block"],
        **case["launch_kwargs"],
    )
    prepared = lifetime.prepared_payload
    capability, subject, _binding, _epoch, pending = _settle_sample_sealed(
        case,
        lifetime,
    )
    foreign_subject = _StableSampleCompositeSubject(
        _SEALED_SUBJECT_GENERATION
    )

    with pytest.raises(ValueError, match="foreign or stale"):
        case["session"].assert_pending_sample_accumulate_releasable(
            pending,
            capability,
            subject=foreign_subject,
        )

    sealed_receipt = pending.sealed_completion_receipt
    assert sealed_receipt.consumed is False
    assert capability.outstanding_receipt_identity == id(sealed_receipt)
    assert case["session"]._outstanding_sample_lifetime is lifetime
    assert case["session"]._pending_sample_completion is pending
    assert lifetime.prepared_payload is prepared
    assert lifetime.consumed is False
    assert lifetime.phase == "launched"

    case["session"].assert_pending_sample_accumulate_releasable(
        pending,
        capability,
        subject=subject,
    )
    commit_plan = pending.consume_sealed_receipt_for_outer_composite(
        case["session"],
        capability,
        subject=subject,
        consumer="test-recovery-after-foreign-subject-rejection",
    )
    case["session"].commit_sample_accumulate_after_consumed_sealed_receipt(
        commit_plan
    )


def test_accelerator_legacy_sample_callback_is_rejected_before_invocation() -> None:
    case = _first_sample_launch_fixture()
    lifetime = case["session"].launch_sample_accumulate(
        case["token"],
        case["sample_block"],
        **case["launch_kwargs"],
    )
    calls = 0

    def forbidden_callback() -> None:
        nonlocal calls
        calls += 1

    object.__setattr__(case["session"].executor, "device", torch.device("mps"))
    with pytest.raises(RuntimeError, match="sample settlement is CPU-only"):
        case["session"].settle_sample_accumulate(
            lifetime,
            device_completion_fence=forbidden_callback,
            device_completion_fence_provenance="forbidden-accelerator-callback",
        )

    assert calls == 0
    assert lifetime.consumed is False
    assert lifetime.prepared_payload is not None
    assert case["session"]._outstanding_sample_lifetime is lifetime


def test_accelerator_legacy_fused_and_abort_callbacks_fail_before_invocation() -> None:
    fused_case = _first_sample_launch_fixture()
    fused_calls = 0

    def forbidden_fused_callback() -> None:
        nonlocal fused_calls
        fused_calls += 1

    object.__setattr__(
        fused_case["session"].executor,
        "device",
        torch.device("mps"),
    )
    with pytest.raises(RuntimeError, match="fused geometry settlement is CPU-only"):
        fused_case["session"].execute_fused_full_geometry_vjp_transaction(
            (),
            max_output_scratch_tensor_bytes=1,
            device_completion_fence=forbidden_fused_callback,
            device_completion_fence_provenance="forbidden-accelerator-callback",
        )
    assert fused_calls == 0

    abort_case = _first_sample_launch_fixture()
    abort_calls = 0

    def forbidden_abort_callback() -> None:
        nonlocal abort_calls
        abort_calls += 1

    object.__setattr__(
        abort_case["session"].executor,
        "device",
        torch.device("mps"),
    )
    with pytest.raises(RuntimeError, match="abort release is CPU-only"):
        abort_case["session"].abort(
            device_completion_fence=forbidden_abort_callback,
            device_completion_fence_provenance="forbidden-accelerator-callback",
        )
    assert abort_calls == 0


def test_outstanding_sample_lifetime_rejects_another_session_operation() -> None:
    case = _first_sample_launch_fixture()
    lifetime = case["session"].launch_sample_accumulate(
        case["token"],
        case["sample_block"],
        **case["launch_kwargs"],
    )
    before = (
        case["compiled"].native_ops.sample_prepare_calls,
        case["compiled"].native_ops.sample_launch_calls,
    )
    with pytest.raises(RuntimeError, match="must settle"):
        case["session"].launch_sample_accumulate(
            case["token"],
            case["sample_block"],
            **case["launch_kwargs"],
        )
    assert (
        case["compiled"].native_ops.sample_prepare_calls,
        case["compiled"].native_ops.sample_launch_calls,
    ) == before
    _settle_sample(case["session"], lifetime)


def test_unknown_sample_completion_cannot_retry_or_abort_release_roots() -> None:
    case = _first_sample_launch_fixture()
    lifetime = case["session"].launch_sample_accumulate(
        case["token"],
        case["sample_block"],
        **case["launch_kwargs"],
    )
    prepared_ref = weakref.ref(lifetime.prepared_payload)
    failed_fence_calls = 0

    def failing_fence() -> None:
        nonlocal failed_fence_calls
        failed_fence_calls += 1
        assert prepared_ref() is lifetime.prepared_payload
        raise RuntimeError("injected sample completion failure")

    with pytest.raises(RuntimeError, match="injected sample completion failure"):
        case["session"].settle_sample_accumulate(
            lifetime,
            device_completion_fence=failing_fence,
            device_completion_fence_provenance=(
                _SAMPLE_COMPLETION_FENCE_PROVENANCE
            ),
        )
    lifetime.assert_retained(case["session"])
    assert failed_fence_calls == lifetime.completion_fence_attempt_count == 1
    assert lifetime.completion_unknown and lifetime.phase == "completion_unknown"
    assert prepared_ref() is lifetime.prepared_payload

    forbidden_retry_calls = 0

    def forbidden_retry() -> None:
        nonlocal forbidden_retry_calls
        forbidden_retry_calls += 1

    with pytest.raises(ValueError, match="poisoned"):
        case["session"].settle_sample_accumulate(
            lifetime,
            device_completion_fence=forbidden_retry,
            device_completion_fence_provenance=(
                _SAMPLE_COMPLETION_FENCE_PROVENANCE
            ),
        )
    with pytest.raises(RuntimeError, match="sample completion is unknown"):
        case["session"].abort(
            device_completion_fence=forbidden_retry,
            device_completion_fence_provenance=(
                _SAMPLE_COMPLETION_FENCE_PROVENANCE
            ),
        )
    assert forbidden_retry_calls == 0
    lifetime.assert_retained(case["session"])
    assert prepared_ref() is lifetime.prepared_payload


class _FakeFusedPreparedBlock:
    def __init__(self, world) -> None:
        self.world = world

    def assert_cold_current(self) -> None:
        self.world.assert_current()


class _FakeFusedTransactionState:
    def __init__(self, node_bars) -> None:
        self.grad_node_chart_f32_by_block = tuple(node_bars)
        self.settled = False
        self.completion_unknown = False


class _FakeFusedTransaction:
    def __init__(self, blocks, node_bars) -> None:
        self._state = _FakeFusedTransactionState(node_bars)
        self.active_block_generation_ids = tuple(
            block.world.runtime.payload.block.generation_digest
            for block in blocks
        )
        self.prepared_block_identities = tuple(id(block) for block in blocks)
        self.node_bar_signatures = tuple(
            executor_module._tensor_signature(tensor) for tensor in node_bars
        )
        self.active_manifest_coverage_certified = False
        self.generation_id = "cpu-fake-fused-transaction-v1"

    def assert_ready(self) -> None:
        if self._state.settled:
            raise ValueError("fake fused transaction already settled")


class _FakeFusedTransactionResult:
    def __init__(self, transaction, blocks, *, fence_provenance: str) -> None:
        global_site_count = blocks[0].world.runtime.global_site_count
        self.grad_compact_site_rgba_f32_by_block = tuple(
            torch.zeros(
                (block.world.compact_site_count, 4),
                dtype=torch.float32,
            )
            for block in blocks
        )
        self.grad_global_positions0_f32 = torch.zeros(
            (global_site_count, 3),
            dtype=torch.float32,
        )
        self.grad_global_velocities_f32 = torch.zeros(
            (global_site_count, 3),
            dtype=torch.float32,
        )
        self.grad_global_weight_coefficients_f32 = torch.zeros(
            (global_site_count, 1),
            dtype=torch.float32,
        )
        bars = (
            *self.grad_compact_site_rgba_f32_by_block,
            self.grad_global_positions0_f32,
            self.grad_global_velocities_f32,
            self.grad_global_weight_coefficients_f32,
        )
        self.active_block_generation_ids = (
            transaction.active_block_generation_ids
        )
        self.transaction_generation_id = transaction.generation_id
        self.block_count = len(blocks)
        self.device_completion_fence_call_count = 1
        self.device_completion_fence_provenance = fence_provenance
        self.length_cotangent_allocated = False
        self.active_manifest_coverage_certified = False
        self.optimizer_fail_atomicity_certified = False
        self.retained_output_tensor_bytes = sum(
            tensor.numel() * tensor.element_size() for tensor in bars
        )

    def assert_current(self) -> None:
        if self.length_cotangent_allocated:
            raise ValueError("fake fused result grew a length cotangent")


def _sampled_fused_session(frame_count: int = 9):
    compiled = _compiled_case()
    adapted = _adapted(frame_count)
    executor = prepare_kinetic_native_material_step_executor(
        compiled.native_ops,
        tuple(
            (runtime, compiled.sampler) for runtime in compiled.lane.runtimes
        ),
        backend_provenance="cpu-contract-double/fused-session-manifest",
    )
    session = executor.begin_step(
        step_generation_id=adapted.loss_normalization_id,
        requested_observation_count=(
            adapted.pixel_count * adapted.observation_count
        ),
    )
    runtime_by_digest = {
        runtime.payload.block.generation_digest: runtime
        for runtime in compiled.lane.runtimes
    }
    tokens = {}
    node_bars = {}
    block_losses = {}
    group = adapted.groups[0]
    staged = group.stage_targets(
        track_start=0,
        track_end=adapted.pixel_count,
        sample_start=0,
        sample_end=group.observation_count,
    )
    for sample_block in iter_paper_kinetic_row_ragged_sample_blocks(
        compiled.sampler,
        staged,
        loss_normalization_id=adapted.loss_normalization_id,
        maximum_samples_per_launch=2,
    ):
        digest = sample_block.native_block_generation_digest
        if digest not in tokens:
            runtime = runtime_by_digest[digest]
            tokens[digest] = session.launch_node_forward(
                runtime,
                compiled.global_site_rgba_f32.index_select(
                    0,
                    runtime.source_site_ids_i64,
                ).contiguous(),
            )
            node_bars[digest] = torch.zeros_like(
                tokens[digest].world.node_chart_f32
            )
            block_losses[digest] = torch.zeros((1,), dtype=torch.float32)
        lifetime = session.launch_sample_accumulate(
            tokens[digest],
            sample_block,
            sampler=compiled.sampler,
            background_rgb_f32=compiled.background_rgb_f32,
            loss_f32=block_losses[digest],
            grad_node_chart_f32=node_bars[digest],
            cone_diagnostic_i32=torch.zeros((3,), dtype=torch.int32),
        )
        _settle_sample(session, lifetime)
    ordered_states = session._ordered_active_states()
    assert len(ordered_states) == len(session._states) > 0
    return compiled, adapted, session, ordered_states


def _install_fused_session_cpu_fake(
    monkeypatch,
    *,
    reject_after_fence=False,
    completion_unknown=False,
):
    captured = {}

    def fake_prepare(blocks, node_bars, *, max_output_scratch_tensor_bytes):
        assert max_output_scratch_tensor_bytes > 0
        captured["blocks"] = tuple(blocks)
        captured["node_bars"] = tuple(node_bars)
        transaction = _FakeFusedTransaction(blocks, node_bars)
        captured["transaction"] = transaction
        return transaction

    def fake_execute(
        transaction,
        *,
        device_completion_fence,
        device_completion_fence_provenance,
    ):
        if completion_unknown:
            transaction._state.completion_unknown = True
            raise RuntimeError("injected unknown fused completion")
        device_completion_fence()
        transaction._state.settled = True
        if reject_after_fence:
            raise RuntimeError("injected settled fused rejection")
        return _FakeFusedTransactionResult(
            transaction,
            captured["blocks"],
            fence_provenance=device_completion_fence_provenance,
        )

    monkeypatch.setattr(
        executor_module,
        "KineticNativeEqualRankFusedDirectFullVjpV1",
        _FakeFusedPreparedBlock,
    )
    monkeypatch.setattr(
        executor_module,
        "KineticNativeEqualRankFusedDirectFullVjpV1Transaction",
        _FakeFusedTransaction,
    )
    monkeypatch.setattr(
        executor_module,
        "KineticNativeEqualRankFusedDirectFullVjpV1TransactionResult",
        _FakeFusedTransactionResult,
    )
    monkeypatch.setattr(
        executor_module,
        "prepare_kinetic_native_equal_rank_fused_direct_full_vjp_transaction_v1",
        fake_prepare,
    )
    monkeypatch.setattr(
        executor_module,
        "execute_kinetic_native_equal_rank_fused_direct_full_vjp_transaction_v1",
        fake_execute,
    )
    return captured


def test_fused_session_uses_exact_states_manifest_and_one_fence(monkeypatch) -> None:
    _compiled, adapted, session, ordered_states = _sampled_fused_session()
    captured = _install_fused_session_cpu_fake(monkeypatch)
    blocks = tuple(
        _FakeFusedPreparedBlock(state.token.world) for state in ordered_states
    )
    node_bar_identities = tuple(
        id(state.grad_node_chart_f32) for state in ordered_states
    )
    fence_calls = 0

    def completion_fence() -> None:
        nonlocal fence_calls
        fence_calls += 1

    receipt = session.execute_fused_full_geometry_vjp_transaction(
        blocks,
        max_output_scratch_tensor_bytes=10_000_000,
        device_completion_fence=completion_fence,
        device_completion_fence_provenance="cpu-fake-fused-one-fence-v1",
    )
    receipt.assert_current(session)
    assert fence_calls == 1
    assert captured["blocks"] == blocks
    assert tuple(id(tensor) for tensor in captured["node_bars"]) == (
        node_bar_identities
    )
    assert receipt.active_runtime_generation_ids == tuple(
        state.runtime_generation_id for state in ordered_states
    )
    assert receipt.active_block_generation_ids == tuple(
        state.native_block_generation_digest for state in ordered_states
    )
    assert receipt.reduced_sample_count == (
        adapted.pixel_count * adapted.observation_count
    )
    assert receipt.active_manifest_coverage_certified
    assert not receipt.length_cotangent_allocated
    assert not receipt.optimizer_fail_atomicity_certified
    assert all(
        state.fused_full_geometry_vjp_launch_count == 1
        and state.material_vjp_launch_count == 0
        and state.full_geometry_vjp_launch_count == 0
        and state.token is None
        and state.grad_node_chart_f32 is None
        for state in ordered_states
    )

    telemetry = session.seal()
    receipt.assert_current()
    telemetry.assert_current()
    assert telemetry.reverse_mode == "fused_full_geometry"
    assert telemetry.native_fused_full_geometry_vjp_launch_count == len(
        ordered_states
    )
    assert telemetry.native_fused_full_geometry_transaction_count == 1
    assert telemetry.native_fused_full_geometry_completion_fence_count == 1
    assert telemetry.native_full_geometry_vjp_launch_count == 0
    assert telemetry.native_full_geometry_fenced_reduction_count == 0
    assert telemetry.native_length_bar_tensor_bytes == 0
    assert telemetry.fused_full_geometry_active_manifest_certified
    assert not telemetry.fused_full_geometry_length_cotangent_allocated
    assert not telemetry.optimizer_fail_atomicity_certified


@pytest.mark.parametrize("reentry_kind", ("execute", "abort"))
def test_fused_completion_callback_rejects_session_reentry(
    monkeypatch,
    reentry_kind,
) -> None:
    _compiled, _adapted, session, ordered_states = _sampled_fused_session()
    captured = _install_fused_session_cpu_fake(monkeypatch)
    blocks = tuple(
        _FakeFusedPreparedBlock(state.token.world) for state in ordered_states
    )
    outer_fence_calls = 0
    nested_fence_calls = 0

    def nested_fence() -> None:
        nonlocal nested_fence_calls
        nested_fence_calls += 1

    def completion_fence() -> None:
        nonlocal outer_fence_calls
        outer_fence_calls += 1
        if reentry_kind == "execute":
            with pytest.raises(RuntimeError, match="cannot be reentered"):
                session.execute_fused_full_geometry_vjp_transaction(
                    blocks,
                    max_output_scratch_tensor_bytes=10_000_000,
                    device_completion_fence=nested_fence,
                    device_completion_fence_provenance=(
                        "cpu-fake-forbidden-nested-fused-v1"
                    ),
                )
        else:
            with pytest.raises(RuntimeError, match="cannot abort reentrantly"):
                session.abort(
                    device_completion_fence=nested_fence,
                    device_completion_fence_provenance=(
                        "cpu-fake-forbidden-nested-abort-v1"
                    ),
                )

    receipt = session.execute_fused_full_geometry_vjp_transaction(
        blocks,
        max_output_scratch_tensor_bytes=10_000_000,
        device_completion_fence=completion_fence,
        device_completion_fence_provenance="cpu-fake-reentry-guard-v1",
    )
    receipt.assert_current(session)
    assert outer_fence_calls == 1
    assert nested_fence_calls == 0
    assert captured["blocks"] == blocks
    assert (
        receipt.transaction_generation_id
        == captured["transaction"].generation_id
    )
    assert not session._fused_transaction_in_progress
    assert not session._failed

    telemetry = session.seal()
    telemetry.assert_current()
    assert telemetry.native_fused_full_geometry_transaction_count == 1
    assert telemetry.native_fused_full_geometry_completion_fence_count == 1
    assert telemetry.native_fused_full_geometry_vjp_launch_count == len(
        ordered_states
    )


def test_fused_receipt_revalidates_sample_manifest_and_count(monkeypatch) -> None:
    _compiled, _adapted, session, ordered_states = _sampled_fused_session()
    _install_fused_session_cpu_fake(monkeypatch)
    receipt = session.execute_fused_full_geometry_vjp_transaction(
        tuple(
            _FakeFusedPreparedBlock(state.token.world)
            for state in ordered_states
        ),
        max_output_scratch_tensor_bytes=10_000_000,
        device_completion_fence=lambda: None,
        device_completion_fence_provenance="cpu-fake-fused-manifest-recheck-v1",
    )
    state = ordered_states[0]
    original_digest = state.sample_manifest_digest
    state.sample_manifest_digest = (
        "0" * 64 if original_digest != "0" * 64 else "1" * 64
    )
    with pytest.raises(ValueError, match="active session"):
        receipt.assert_current(session)
    state.sample_manifest_digest = original_digest
    state.streamed_sample_count += 1
    with pytest.raises(ValueError, match="active session"):
        receipt.assert_current(session)


def test_fused_acceptance_freezes_the_active_block_manifest(monkeypatch) -> None:
    compiled, _adapted, session, ordered_states = _sampled_fused_session()
    _install_fused_session_cpu_fake(monkeypatch)
    session.execute_fused_full_geometry_vjp_transaction(
        tuple(
            _FakeFusedPreparedBlock(state.token.world)
            for state in ordered_states
        ),
        max_output_scratch_tensor_bytes=10_000_000,
        device_completion_fence=lambda: None,
        device_completion_fence_provenance="cpu-fake-fused-freeze-v1",
    )
    runtime = compiled.lane.runtimes[0]
    with pytest.raises(ValueError, match="cannot launch after fused manifest"):
        session.launch_node_forward(
            runtime,
            compiled.global_site_rgba_f32.index_select(
                0,
                runtime.source_site_ids_i64,
            ).contiguous(),
        )


def test_fused_session_rejects_a_second_transaction(monkeypatch) -> None:
    _compiled, _adapted, session, ordered_states = _sampled_fused_session()
    _install_fused_session_cpu_fake(monkeypatch)
    blocks = tuple(
        _FakeFusedPreparedBlock(state.token.world)
        for state in ordered_states
    )
    session.execute_fused_full_geometry_vjp_transaction(
        blocks,
        max_output_scratch_tensor_bytes=10_000_000,
        device_completion_fence=lambda: None,
        device_completion_fence_provenance="cpu-fake-fused-single-use-v1",
    )
    with pytest.raises(ValueError, match="already executed"):
        session.execute_fused_full_geometry_vjp_transaction(
            blocks,
            max_output_scratch_tensor_bytes=10_000_000,
            device_completion_fence=lambda: None,
            device_completion_fence_provenance="cpu-fake-fused-single-use-v1",
        )


def test_fused_session_rejects_an_omitted_active_state_before_prepare(
    monkeypatch,
) -> None:
    _compiled, _adapted_batch, session, ordered_states = _sampled_fused_session()
    captured = _install_fused_session_cpu_fake(monkeypatch)
    blocks = tuple(
        _FakeFusedPreparedBlock(state.token.world)
        for state in ordered_states[:-1]
    )
    with pytest.raises(ValueError, match="exact active manifest"):
        session.execute_fused_full_geometry_vjp_transaction(
            blocks,
            max_output_scratch_tensor_bytes=10_000_000,
            device_completion_fence=lambda: None,
            device_completion_fence_provenance="cpu-fake-omission-v1",
        )
    assert "transaction" not in captured
    assert session._failed
    assert all(
        state.fused_full_geometry_vjp_launch_count == 0
        for state in ordered_states
    )


def test_fused_session_rejects_caller_reordering_before_prepare(monkeypatch) -> None:
    _compiled, _adapted_batch, session, ordered_states = _sampled_fused_session()
    assert len(ordered_states) > 1
    captured = _install_fused_session_cpu_fake(monkeypatch)
    blocks = tuple(
        _FakeFusedPreparedBlock(state.token.world)
        for state in reversed(ordered_states)
    )
    with pytest.raises(ValueError, match="sampled active state"):
        session.execute_fused_full_geometry_vjp_transaction(
            blocks,
            max_output_scratch_tensor_bytes=10_000_000,
            device_completion_fence=lambda: None,
            device_completion_fence_provenance="cpu-fake-reorder-v1",
        )
    assert "transaction" not in captured
    assert session._failed


def test_settled_fused_rejection_is_not_double_fenced_on_abort(monkeypatch) -> None:
    _compiled, _adapted_batch, session, ordered_states = _sampled_fused_session()
    captured = _install_fused_session_cpu_fake(
        monkeypatch,
        reject_after_fence=True,
    )
    blocks = tuple(
        _FakeFusedPreparedBlock(state.token.world) for state in ordered_states
    )
    transaction_fence_calls = 0

    def transaction_fence() -> None:
        nonlocal transaction_fence_calls
        transaction_fence_calls += 1

    with pytest.raises(RuntimeError, match="settled fused rejection"):
        session.execute_fused_full_geometry_vjp_transaction(
            blocks,
            max_output_scratch_tensor_bytes=10_000_000,
            device_completion_fence=transaction_fence,
            device_completion_fence_provenance="cpu-fake-settled-rejection-v1",
        )
    assert transaction_fence_calls == 1
    assert session._failed_fused_full_geometry_transaction is captured["transaction"]
    assert session._failed_fused_full_geometry_error is not None
    abort_fence_calls = 0

    def forbidden_second_fence() -> None:
        nonlocal abort_fence_calls
        abort_fence_calls += 1

    session.abort(
        device_completion_fence=forbidden_second_fence,
        device_completion_fence_provenance="cpu-fake-unused-second-fence-v1",
    )
    assert abort_fence_calls == 0
    assert session._abort_release_completed
    assert all(
        state.token is None and state.grad_node_chart_f32 is None
        for state in ordered_states
    )


def test_unknown_fused_completion_keeps_live_roots_and_requires_restart(
    monkeypatch,
) -> None:
    _compiled, _adapted_batch, session, ordered_states = _sampled_fused_session()
    captured = _install_fused_session_cpu_fake(
        monkeypatch,
        completion_unknown=True,
    )
    blocks = tuple(
        _FakeFusedPreparedBlock(state.token.world) for state in ordered_states
    )
    with pytest.raises(RuntimeError, match="unknown fused completion"):
        session.execute_fused_full_geometry_vjp_transaction(
            blocks,
            max_output_scratch_tensor_bytes=10_000_000,
            device_completion_fence=lambda: None,
            device_completion_fence_provenance="cpu-fake-unknown-completion-v1",
        )
    assert session._failed_fused_full_geometry_transaction is captured["transaction"]
    assert session._fused_full_geometry_completion_unknown
    with pytest.raises(RuntimeError, match="restart is required"):
        session.abort(
            device_completion_fence=lambda: None,
            device_completion_fence_provenance="cpu-fake-forbidden-abort-v1",
        )
    assert not session._abort_release_completed
    assert all(
        state.token is not None and state.grad_node_chart_f32 is not None
        for state in ordered_states
    )
