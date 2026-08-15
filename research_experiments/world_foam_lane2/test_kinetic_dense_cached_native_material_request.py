from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import kinetic_dense_cached_native_material_request as dense_request_module
from kinetic_compiled_cpu_artifact_store import (
    PaperKineticCompiledCpuArtifactStore,
    PaperKineticCompiledCpuArtifactStorePolicy,
)
from kinetic_dense_cached_native_material_request import (
    FUSED_DIRECT_V1_FULL_GEOMETRY_REVERSE,
    PaperKineticDenseCachedNativeMemoryPolicy,
    STAGED_SPARSE_FULL_GEOMETRY_REVERSE,
    authorize_paper_kinetic_dense_optimizer_step,
    consume_paper_kinetic_dense_request_delta,
    fail_stop_paper_kinetic_dense_step,
    prepare_paper_kinetic_dense_chunk_target_loader,
    prepare_paper_kinetic_dense_chunk_target_loader_test_fault,
    prepare_paper_kinetic_dense_step_gradient_accumulator,
    run_paper_kinetic_dense_cached_native_request,
    run_paper_kinetic_dense_cached_native_material_request,
)
from paper_kinetic_union_local_bar_assembly import (
    prepare_paper_kinetic_union_local_spatial_bundle,
)
from paper_kinetic_replayable_observations import (
    OBSERVATION_IDENTITY_LOGICAL_BYTES,
    TRACK_ID_LOGICAL_BYTES,
    PaperKineticDenseObservationMemoryPolicy,
    prepare_paper_kinetic_replayable_dense_observation_source,
)
from paper_training_types import SpacetimeBatch, SpacetimeSample
from test_kinetic_compiled_cpu_artifact_store import (
    _compile_artifact,
    _observations,
    _provider as _artifact_provider,
)
from test_kinetic_ragged_paper_step_cpu_fake_native import _FakeNativeOps


class _CoverageCheckingNativeOps(_FakeNativeOps):
    def __init__(self) -> None:
        super().__init__()
        self.replay_session = None
        self.vjp_saw_complete_request: list[bool] = []
        self.full_vjp_saw_complete_request: list[bool] = []
        self.return_allocating_forward_calls = 0
        self.forward_into_calls = 0

    def _request_complete(self) -> bool:
        session = self.replay_session
        return bool(
            session is not None
            and session.request_count >= 1
            and session.emitted_observation_count > 0
            and not session._active_request
            and not session.poisoned
        )

    def kinetic_precompiled_length_p0_lie_material_node_vjp_accumulate_launch_only(
        self,
        *args,
        **kwargs,
    ):
        self.vjp_saw_complete_request.append(self._request_complete())
        return super().kinetic_precompiled_length_p0_lie_material_node_vjp_accumulate_launch_only(
            *args,
            **kwargs,
        )

    def kinetic_precompiled_length_p0_lie_node_forward_launch_only(
        self,
        *args,
        **kwargs,
    ):
        self.return_allocating_forward_calls += 1
        return super().kinetic_precompiled_length_p0_lie_node_forward_launch_only(
            *args,
            **kwargs,
        )

    def kinetic_precompiled_length_p0_lie_node_forward_into_launch_only_v1(
        self,
        *args,
        **kwargs,
    ):
        self.forward_into_calls += 1
        return super().kinetic_precompiled_length_p0_lie_node_forward_into_launch_only_v1(
            *args,
            **kwargs,
        )

    def kinetic_precompiled_length_p0_lie_node_vjp_accumulate_launch_only(
        self,
        *args,
        **kwargs,
    ):
        self.full_vjp_saw_complete_request.append(self._request_complete())
        return super().kinetic_precompiled_length_p0_lie_node_vjp_accumulate_launch_only(
            *args,
            **kwargs,
        )


@dataclass
class _Fence:
    calls: int = 0

    def __call__(self) -> None:
        self.calls += 1


def _dense_policy(chunk_capacity: int) -> PaperKineticDenseObservationMemoryPolicy:
    return PaperKineticDenseObservationMemoryPolicy(
        maximum_persistent_observation_count=0,
        maximum_persistent_observation_logical_bytes=0,
        maximum_retained_frame_metadata_count=64,
        maximum_retained_frame_metadata_logical_bytes=4096,
        maximum_live_generated_observation_count=1,
        maximum_live_generated_observation_logical_bytes=(OBSERVATION_IDENTITY_LOGICAL_BYTES),
        maximum_request_track_count=2,
        maximum_request_track_logical_bytes=2 * TRACK_ID_LOGICAL_BYTES,
        maximum_chunk_observation_count=chunk_capacity,
        maximum_chunk_observation_logical_bytes=(chunk_capacity * OBSERVATION_IDENTITY_LOGICAL_BYTES),
    )


def _request_policy(**overrides) -> PaperKineticDenseCachedNativeMemoryPolicy:
    values = {
        "maximum_lane_resident_logical_tensor_bytes": 10_000_000,
        "maximum_active_node_and_union_bar_tensor_bytes": 10_000_000,
        "maximum_decoded_frame_scratch_tensor_bytes": 10_000_000,
        "maximum_chunk_target_tensor_bytes": 10_000_000,
        "maximum_target_decode_bridge_peak_logical_tensor_bytes": 10_000_000,
        "maximum_sample_materialization_logical_tensor_bytes": 10_000_000,
        "maximum_sample_launch_tensor_bytes": 10_000_000,
        "maximum_request_geometry_bar_tensor_bytes": 10_000_000,
        "maximum_geometry_bridge_visible_peak_logical_tensor_bytes": 10_000_000,
    }
    values.update(overrides)
    return PaperKineticDenseCachedNativeMemoryPolicy(**values)


def _target_loader(
    source,
    request,
    *,
    policy=None,
    target_generation_id="fixture-targets-v1",
    source_test_fault=None,
):
    selected = _request_policy() if policy is None else policy
    return prepare_paper_kinetic_dense_chunk_target_loader(
        source,
        request,
        device="cpu",
        target_generation_id=target_generation_id,
        maximum_decoded_frame_scratch_tensor_bytes=(
            selected.maximum_decoded_frame_scratch_tensor_bytes
        ),
        maximum_chunk_target_tensor_bytes=(
            selected.maximum_chunk_target_tensor_bytes
        ),
        maximum_target_decode_bridge_peak_logical_tensor_bytes=(
            selected.maximum_target_decode_bridge_peak_logical_tensor_bytes
        ),
        source_test_fault=source_test_fault,
    )


def _batch(frame_indices: tuple[int, ...]) -> SpacetimeBatch:
    return SpacetimeBatch(
        samples=tuple(SpacetimeSample(view_index=0, frame_index=frame_index) for frame_index in frame_indices),
        epoch=7,
        batch_index=3,
        completes_epoch=False,
    )


def _provider():
    """Two-pixel fixture whose canonical track request is the full manifest."""

    return _artifact_provider(height=1, width=2)


def _case(*, chunk_capacity: int, frame_indices: tuple[int, ...]):
    target_source, factory, provider = _provider()
    store = PaperKineticCompiledCpuArtifactStore(
        PaperKineticCompiledCpuArtifactStorePolicy(
            maximum_entries=1,
            maximum_resident_accounted_bytes=10_000_000,
        )
    )
    acquisition = store.acquire(
        provider,
        view_index=0,
        track_ids=(0, 1),
        maximum_artifact_accounted_bytes=10_000_000,
        compile_artifact=lambda key: _compile_artifact(
            provider,
            _observations((0, 0), (0, 1)),
            key,
        ),
    )
    source = prepare_paper_kinetic_replayable_dense_observation_source(
        provider,
        _batch(frame_indices),
        memory_policy=_dense_policy(chunk_capacity),
    )
    replay = source.open_session()
    request = source.prepare_track_request(view_index=0, track_ids=(0, 1))
    return (
        target_source,
        factory,
        provider,
        store,
        acquisition.artifact,
        source,
        replay,
        request,
    )


def _run(
    *,
    chunk_capacity: int,
    frame_indices: tuple[int, ...],
    policy=None,
    full_geometry: bool = False,
    optimize_camera_rays: bool | None = None,
    before_consume=None,
    before_authorize=None,
    full_geometry_reverse_mode: str = STAGED_SPARSE_FULL_GEOMETRY_REVERSE,
):
    (
        target_source,
        factory,
        provider,
        store,
        artifact,
        source,
        replay,
        request,
    ) = _case(chunk_capacity=chunk_capacity, frame_indices=frame_indices)
    native_ops = _CoverageCheckingNativeOps()
    native_ops.replay_session = replay
    fence = _Fence()
    material = torch.tensor([[0.21, 0.37, 0.16, 0.8]], dtype=torch.float32)
    background = torch.tensor((0.03, 0.05, 0.07), dtype=torch.float32)
    compile_count_before = factory.compile_count
    selected_policy = _request_policy() if policy is None else policy
    accumulator = prepare_paper_kinetic_dense_step_gradient_accumulator(
        source,
        replay,
        step_generation_id="dense-cached-step-v1",
        loss_normalization_id="global-rgb-mean-v1",
        material_generation_id="fixture-material-v1",
        background_generation_id="fixture-background-v1",
        global_site_rgba_f32=material,
        background_rgb_f32=background,
        device="cpu",
        full_geometry=full_geometry,
        optimize_camera_rays=(
            full_geometry
            if optimize_camera_rays is None
            else optimize_camera_rays
        ),
    )

    loader = _target_loader(source, request, policy=selected_policy)

    result = run_paper_kinetic_dense_cached_native_request(
        source,
        replay,
        request,
        artifact,
        accumulator,
        step_generation_id="dense-cached-step-v1",
        loss_normalization_id="global-rgb-mean-v1",
        material_generation_id="fixture-material-v1",
        background_generation_id="fixture-background-v1",
        global_site_rgba_f32=material,
        background_rgb_f32=background,
        native_ops=native_ops,
        backend_provenance="cpu-fake-native/exact-op-surface",
        maximum_samples_per_launch=2,
        memory_policy=selected_policy,
        load_chunk_targets=loader,
        device_completion_fence=fence,
        device_completion_fence_provenance="cpu-synchronous-fence-v1",
        full_geometry_reverse_mode=full_geometry_reverse_mode,
    )
    if before_consume is not None:
        before_consume(result.delta, accumulator, source, request, artifact, replay)
    commit_receipt = consume_paper_kinetic_dense_request_delta(
        accumulator,
        source,
        replay,
        request,
        artifact,
        result.delta,
        device_completion_fence=fence,
        device_completion_fence_provenance="cpu-synchronous-fence-v1",
    )
    replay_receipt = replay.seal()
    if before_authorize is not None:
        before_authorize(material, background, accumulator)
    authorization = authorize_paper_kinetic_dense_optimizer_step(
        accumulator,
        source,
        replay,
        replay_receipt,
    )
    assert factory.compile_count == compile_count_before
    return {
        "target_source": target_source,
        "provider": provider,
        "store": store,
        "artifact": artifact,
        "source": source,
        "replay": replay,
        "request": request,
        "native_ops": native_ops,
        "fence": fence,
        "material": material,
        "bar": authorization.grad_site_rgba_f32,
        "loss": authorization.loss_f32,
        "result": result,
        "accumulator": accumulator,
        "authorization": authorization,
        "commit_receipt": commit_receipt,
        "replay_receipt": replay_receipt,
    }


def test_optimizer_authorization_rejects_post_request_material_mutation() -> None:
    captured = {}

    def mutate_material(material, _background, accumulator) -> None:
        captured["accumulator"] = accumulator
        material.add_(0.125)

    with pytest.raises(ValueError, match="step accumulator changed"):
        _run(
            chunk_capacity=2,
            frame_indices=(0, 2),
            before_authorize=mutate_material,
        )
    accumulator = captured["accumulator"]
    assert accumulator.poisoned
    assert not accumulator.optimizer_authorized
    assert all(
        not bool(torch.any(tensor != 0.0).item())
        for tensor in accumulator._tensors()
    )


def test_cached_lane_replays_all_chunks_then_launches_one_vjp_per_active_block() -> None:
    case = _run(chunk_capacity=1, frame_indices=(0, 2))
    result = case["result"]
    telemetry = result.telemetry

    result.assert_current(
        case["source"],
        case["request"],
        case["artifact"],
        case["replay"],
        case["accumulator"],
    )
    assert result.receipt.expected_observation_count == 4
    assert result.receipt.replay_chunk_count == 4
    assert telemetry.streamed_sample_count == 4
    assert telemetry.native_sample_launch_count == 4
    assert (
        telemetry.native_node_forward_launch_count
        == telemetry.native_material_word_vjp_launch_count
        == telemetry.active_native_block_count
    )
    assert telemetry.active_native_block_count == 2
    assert case["native_ops"].forward_calls == 2
    assert case["native_ops"].return_allocating_forward_calls == 0
    assert case["native_ops"].forward_into_calls == 2
    assert case["native_ops"].material_vjp_calls == 2
    assert case["native_ops"].vjp_saw_complete_request == [True, True]
    assert case["fence"].calls == (
        telemetry.native_sample_launch_count
        + telemetry.active_native_block_count
        + 2
    )
    case["commit_receipt"].assert_current(
        case["accumulator"],
        case["source"],
        case["replay"],
        case["request"],
        case["artifact"],
        result.delta,
    )
    assert case["commit_receipt"].device_completion_fence_call_count == 1
    assert case["commit_receipt"].persistent_tensor_bytes == 0
    assert case["replay"].request_count == 1
    assert case["replay"].emitted_observation_count == 4
    assert not case["replay"].poisoned
    assert torch.isfinite(case["loss"]).all()
    assert torch.isfinite(case["bar"]).all()
    assert float(case["loss"].item()) > 0.0
    assert bool(torch.any(case["bar"] != 0.0).item())

    accounting = result.accounting
    assert accounting["native_lane_prepare_count"] == 1
    assert accounting["native_lane_two_phase_construction"] is True
    assert accounting[
        "union_and_runtime_construction_lifetimes_retained_through_lane_fence"
    ] is True
    assert accounting["accelerator_release_capability_integrated"] is False
    blocks = tuple(
        block
        for bucket in case["artifact"].sampler.lowering.buckets
        for block in bucket.blocks
    )
    union_count = len(
        {
            source_id
            for block in blocks
            for source_id in block.source_site_ids
        }
    )
    expected_construction_predecessor_bytes = (
        8 * union_count
        + 8 * sum(len(block.source_site_ids) for block in blocks)
        + 4 * len(blocks)
    )
    assert accounting[
        "lane_two_phase_construction_predecessor_logical_tensor_bytes_upper_bound"
    ] == expected_construction_predecessor_bytes
    assert accounting[
        "lane_two_phase_construction_predecessors_overlap_active_request"
    ] is True
    assert accounting["sample_node_interaction_count"] >= (
        accounting["streamed_observation_count"]
    )
    assert accounting["transferred_target_payload_bytes"] == 4 * 12
    assert accounting["peak_sample_launch_node_count"] == max(
        accounting["chart_node_ranks"]
    )
    assert accounting["active_native_block_count"] == 2
    assert accounting["node_forward_launch_count"] == 2
    assert accounting["node_forward_abi"] == "caller_preallocated_into_v1"
    assert accounting["return_allocating_node_forward_launch_count"] == 0
    assert accounting["caller_preallocated_node_forward_launch_count"] == 2
    assert accounting["forward_into_lifetime_install_count"] == 2
    assert accounting["forward_into_lifetime_retire_count"] == 2
    assert accounting["compact_gather_lifetime_install_count"] == 2
    assert accounting["compact_gather_lifetime_retire_count"] == 2
    assert accounting["retained_forward_into_lifetime_count_after_request"] == 0
    assert accounting["retained_compact_gather_lifetime_count_after_request"] == 0
    assert accounting["forward_into_lifetime_additional_logical_tensor_bytes"] == 0
    assert accounting["compact_gather_lifetime_additional_logical_tensor_bytes"] == 0
    assert accounting[
        "forward_predecessor_and_output_roots_released_only_after_reverse_or_abort_fence"
    ] is True
    assert accounting["caller_preallocated_node_forward_output_bytes"] > 0
    assert accounting["node_forward_thread_count"] > 0
    assert accounting["node_forward_interaction_count"] >= accounting[
        "node_forward_thread_count"
    ]
    assert accounting["node_forward_interaction_count"] == accounting[
        "material_word_vjp_interaction_count"
    ]
    assert accounting["active_material_exact_model_bytes"] > 0
    assert len(accounting["artifact_structural_signature_sha256"]) == 64
    assert accounting["lane_reused_across_all_chunks"] is True
    assert accounting["structural_compile_track_count_during_request"] == 0
    assert accounting["persistent_frame_tensor_bytes"] == 0
    assert accounting["persistent_sample_tensor_bytes"] == 0
    assert accounting["persistent_target_tensor_bytes"] == 0
    assert accounting["retained_observation_count_after_request"] == 0
    assert accounting["request_returns_one_combined_uncommitted_delta"] is True
    assert accounting["step_accumulator_world_bound_not_sampler_bound"] is True
    assert accounting["step_accumulator_retains_frame_axis"] is False
    assert accounting["optimizer_authorization_requires_full_manifest_seal"] is True
    assert accounting["optimizer_authorization_is_point_in_time"] is True
    assert accounting["post_authorization_snapshot_mutation_zeroes_bars"] is False
    assert accounting["target_loader_is_arbitrary_callable"] is False
    assert accounting["target_loader_partial_failure_lifetime_certified"] is True
    assert accounting["target_loader_retained_closure_state_measured"] is True
    assert accounting["target_loader_completed_load_count"] == accounting[
        "replay_chunk_count"
    ]
    assert accounting["target_loader_failed_after_enqueue_count"] == 0
    assert accounting["target_loader_maximum_outstanding_lifetime_count"] == 1
    assert accounting["target_loader_retained_lifetime_count_after_request"] == 0
    assert accounting[
        "target_loader_transfer_roots_released_only_after_completion_fence"
    ] is True
    assert accounting["target_loader_lifetime_additional_logical_tensor_bytes"] == 0
    assert accounting["target_loader_lifetime_python_heap_bytes_measured"] is False
    assert accounting["whole_pipeline_target_loader_memory_proven"] is False
    assert accounting["decoder_allocator_peak_measured"] is False
    assert accounting["sample_materialization_float64_scratch_measured"] is False
    assert accounting[
        "sample_materialization_source_visible_logical_tensors_accounted"
    ] is True
    assert accounting["whole_step_python_object_peak_measured"] is False
    assert not result.target_loader_is_arbitrary_callable
    assert result.target_loader_partial_failure_lifetime_certified
    assert not result.decoder_allocator_peak_measured
    assert not result.sample_materialization_float64_scratch_measured
    assert not result.whole_step_python_object_peak_measured
    assert accounting["sample_completion_fence_call_count"] == telemetry.native_sample_launch_count
    assert (
        accounting["native_sample_lifetime_token_count"]
        == accounting["native_sample_lifetime_settle_count"]
        == accounting["native_sample_completion_fence_count"]
        == telemetry.native_sample_prepare_count
        == telemetry.native_sample_launch_count
        == telemetry.native_sample_completion_fence_count
    )
    assert accounting["maximum_in_flight_sample_lifetime_token_count"] == 1
    assert accounting["retained_sample_lifetime_token_count_after_seal"] == 0
    assert accounting["sample_lifetime_token_history_retained"] is False
    assert accounting["sample_lifetime_additional_logical_tensor_bytes"] == 0
    assert accounting["sample_lifetime_python_heap_bytes_measured"] is False
    assert accounting[
        "sample_lifetime_roots_released_only_after_completion_fence"
    ] is True
    assert accounting[
        "sample_materialization_predecessor_roots_leased_until_fence"
    ] is True
    assert accounting[
        "chunk_cpu_transfer_source_retained_through_sample_fences"
    ] is True
    assert accounting["active_block_commit_fence_call_count"] == telemetry.active_native_block_count
    assert accounting["active_block_commit_fenced_before_scratch_release"] is True
    assert accounting["maximum_in_flight_active_block_commit_scratch_count"] == 1
    assert accounting["sample_launch_fence_requested_after_every_launch"] is True
    assert accounting["maximum_requested_in_flight_sample_launches"] == 1
    assert accounting["real_device_fence_semantics_verified"] is False
    assert accounting["decoded_frame_device_type"] == "cpu"
    assert accounting["decoded_frame_mps_completion_fence_call_count"] == 0
    assert accounting["single_bounded_chunk_transfer_per_replay_chunk"] is True
    assert accounting["cpu_to_device_chunk_transfer_requested_non_blocking"] is False
    assert accounting["real_device_transfer_completion_verified"] is False
    assert int(accounting["peak_cpu_decoded_frame_tensor_bytes"]) > 0
    assert int(accounting["peak_cpu_chunk_target_tensor_bytes"]) > 0
    assert int(accounting["peak_device_chunk_target_tensor_bytes"]) > 0
    assert int(accounting["peak_target_decode_bridge_logical_tensor_bytes"]) > 0
    assert accounting["peak_native_prepared_sample_scratch_tensor_bytes"] == 24
    assert accounting["native_prepared_sample_scratch_formula"] == "4*N+20"
    assert accounting["native_prepared_sample_public_tensor_scratch_accounted"] is True
    assert int(
        accounting[
            "peak_sample_materialization_logical_tensor_bytes_upper_bound"
        ]
    ) > 0
    assert int(
        accounting[
            "peak_interpolation_evaluator_scratch_logical_tensor_bytes_upper_bound"
        ]
    ) > 0
    assert int(accounting["maximum_interpolation_rows_per_subchunk"]) > 0
    assert accounting["requested_maximum_samples_per_launch"] == 2
    assert accounting["effective_maximum_samples_per_launch"] == 2
    assert accounting["interpolation_evaluator_scratch_formula"] == (
        "4096+512*J+8*J^2+K_sub*(1024+512*J)"
    )
    assert accounting["native_driver_allocator_scratch_measured"] is False
    assert accounting["selected_pixel_read_mode"] == "full_frame_fallback"
    assert accounting["selected_pixel_read_acceptance_capable"] is False
    assert accounting["full_frame_target_materialization_count"] > 0
    assert accounting["direct_selected_pixel_observation_count"] == 0
    assert accounting["full_frame_fallback_observation_count"] == 4
    assert accounting["total_pf_sample_work_is_linear"] is False
    assert (
        accounting[
            "structural_node_word_work_invariance_requires_cross_row_verification"
        ]
        is True
    )
    assert accounting["full_geometry_vjp_integrated"] is False
    assert accounting["production_trainer_integrated"] is False
    assert not result.native_runtime_verified
    assert not result.allocator_peak_measured


def test_float64_interpolation_scratch_subchunks_without_changing_step() -> None:
    wide = _run(chunk_capacity=4, frame_indices=(0, 2))
    bounded = _run(
        chunk_capacity=4,
        frame_indices=(0, 2),
        policy=_request_policy(
            maximum_sample_materialization_logical_tensor_bytes=8_000,
        ),
    )

    torch.testing.assert_close(bounded["loss"], wide["loss"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(bounded["bar"], wide["bar"], rtol=0.0, atol=0.0)
    accounting = bounded["result"].accounting
    assert accounting["requested_maximum_samples_per_launch"] == 2
    assert accounting["effective_maximum_samples_per_launch"] == 2
    assert accounting["maximum_interpolation_rows_per_subchunk"] == 1
    assert (
        accounting["peak_sample_materialization_logical_tensor_bytes_upper_bound"]
        <= accounting["maximum_sample_materialization_logical_tensor_bytes"]
        == 8_000
    )
    assert accounting[
        "sample_materialization_source_visible_logical_tensors_accounted"
    ] is True


def test_dense_full_geometry_reuses_replay_and_authorizes_one_bounded_step() -> None:
    full = _run(
        chunk_capacity=1,
        frame_indices=(0, 2),
        full_geometry=True,
    )
    material = _run(chunk_capacity=1, frame_indices=(0, 2))
    result = full["result"]
    telemetry = result.telemetry
    accumulator = full["accumulator"]
    authorization = full["authorization"]

    torch.testing.assert_close(full["loss"], material["loss"], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(full["bar"], material["bar"], rtol=2e-6, atol=2e-6)
    authorization.assert_current(accumulator, full["replay_receipt"])
    assert accumulator.consumed_request_count == 1
    assert accumulator.optimizer_authorized
    assert result.delta.consumed
    assert result.delta.logical_tensor_bytes == 0
    assert result.full_geometry_vjp_integrated
    assert telemetry.reverse_mode == "full_geometry"
    assert telemetry.native_material_word_vjp_launch_count == 0
    assert (
        telemetry.native_full_geometry_vjp_launch_count
        == telemetry.active_native_block_count
    )
    assert full["native_ops"].material_vjp_calls == 0
    assert full["native_ops"].vjp_calls == telemetry.active_native_block_count
    assert full["native_ops"].full_vjp_saw_complete_request == [True, True]
    assert full["fence"].calls == (
        telemetry.native_sample_launch_count
        + 2 * telemetry.active_native_block_count
        + 2
    )
    assert result.accounting["geometry_committed_after_executor_seal"] is False
    assert result.accounting["caller_bars_mutated_by_request"] is False
    assert result.accounting["full_geometry_vjp_integrated"] is True
    assert (
        result.accounting["geometry_completion_receipt_count"]
        == telemetry.active_native_block_count
    )
    assert result.accounting["geometry_completion_receipt_retains_native_tensors"] is False
    assert int(result.accounting["maximum_native_length_bar_tensor_bytes"]) > 0
    assert int(result.accounting["request_geometry_bar_tensor_bytes"]) > 0
    expected_bridge = (
        dense_request_module._staged_sparse_geometry_bridge_visible_peak_logical_tensor_bytes_upper_bound(
            full["artifact"],
            include_ray_gradients=True,
        )
    )
    assert result.accounting[
        "staged_sparse_geometry_bridge_visible_peak_logical_tensor_bytes_upper_bound"
    ] == expected_bridge
    assert result.accounting["staged_sparse_geometry_bridge_included_in_main_active_peak"]
    assert result.accounting["active_request_logical_tensor_bytes_upper_bound"] == (
        dense_request_module._active_state_upper_bound_bytes(
            full["artifact"],
            include_full_geometry=False,
        )
        + result.accounting["request_geometry_bar_tensor_bytes"]
        + accumulator.logical_tensor_bytes
        + expected_bridge
    )
    assert result.accounting[
        "reverse_lane_plus_active_logical_tensor_bytes_upper_bound"
    ] == (
        result.accounting["lane_resident_logical_tensor_bytes_upper_bound"]
        + result.accounting["active_request_logical_tensor_bytes_upper_bound"]
    )
    assert result.accounting[
        "reverse_lane_plus_active_logical_tensor_bytes_upper_bound"
    ] <= result.accounting["reverse_lane_plus_active_policy_cap_sum"]
    assert not result.accounting["reverse_lane_plus_active_is_allocator_peak"]
    assert any(
        bool(torch.any(tensor != 0.0).item())
        for tensor in (
            authorization.grad_positions0_f64,
            authorization.grad_velocities_f64,
            authorization.grad_weight_coefficients_f64,
            authorization.grad_track_ray_coefficients_f64,
        )
    )


def test_staged_bridge_is_composed_before_native_lane_build(monkeypatch) -> None:
    (
        _target_source,
        _factory,
        _provider,
        _store,
        artifact,
        source,
        replay,
        request,
    ) = _case(chunk_capacity=1, frame_indices=(0, 2))
    material = torch.tensor([[0.21, 0.37, 0.16, 0.8]], dtype=torch.float32)
    background = torch.tensor((0.03, 0.05, 0.07), dtype=torch.float32)
    accumulator = prepare_paper_kinetic_dense_step_gradient_accumulator(
        source,
        replay,
        step_generation_id="staged-memory-preflight-v1",
        loss_normalization_id="global-rgb-mean-v1",
        material_generation_id="fixture-material-v1",
        background_generation_id="fixture-background-v1",
        global_site_rgba_f32=material,
        background_rgb_f32=background,
        device="cpu",
        full_geometry=True,
        optimize_camera_rays=True,
    )
    old_incomplete_bound = (
        dense_request_module._active_state_upper_bound_bytes(
            artifact,
            include_full_geometry=False,
        )
        + dense_request_module._request_geometry_bar_bytes(
            artifact.sampler,
            request,
            include_ray_gradients=True,
        )
        + accumulator.logical_tensor_bytes
    )
    bridge = (
        dense_request_module._staged_sparse_geometry_bridge_visible_peak_logical_tensor_bytes_upper_bound(
            artifact,
            include_ray_gradients=True,
        )
    )
    lane_prepare_calls = 0

    def reject_lane_prepare(*_args, **_kwargs):
        nonlocal lane_prepare_calls
        lane_prepare_calls += 1
        raise AssertionError("native lane build started before staged memory admission")

    monkeypatch.setattr(
        dense_request_module,
        "_prepare_dense_cached_native_lane",
        reject_lane_prepare,
    )
    with pytest.raises(MemoryError, match="active-state budget"):
        _run(
            chunk_capacity=1,
            frame_indices=(0, 2),
            policy=_request_policy(
                maximum_active_node_and_union_bar_tensor_bytes=(
                    old_incomplete_bound + bridge - 1
                ),
            ),
            full_geometry=True,
        )

    assert bridge > 0
    assert lane_prepare_calls == 0


def test_fixed_camera_full_geometry_omits_global_and_request_ray_bars() -> None:
    trainable_camera = _run(
        chunk_capacity=1,
        frame_indices=(0, 2),
        full_geometry=True,
        optimize_camera_rays=True,
    )
    fixed_camera = _run(
        chunk_capacity=1,
        frame_indices=(0, 2),
        full_geometry=True,
        optimize_camera_rays=False,
    )

    fixed_authorization = fixed_camera["authorization"]
    fixed_accounting = fixed_camera["result"].accounting
    torch.testing.assert_close(
        fixed_camera["loss"],
        trainable_camera["loss"],
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        fixed_camera["bar"],
        trainable_camera["bar"],
        rtol=2e-6,
        atol=2e-6,
    )
    for fixed_bar, trainable_bar in zip(
        (
            fixed_authorization.grad_positions0_f64,
            fixed_authorization.grad_velocities_f64,
            fixed_authorization.grad_weight_coefficients_f64,
        ),
        (
            trainable_camera["authorization"].grad_positions0_f64,
            trainable_camera["authorization"].grad_velocities_f64,
            trainable_camera["authorization"].grad_weight_coefficients_f64,
        ),
        strict=True,
    ):
        torch.testing.assert_close(fixed_bar, trainable_bar, rtol=8e-5, atol=8e-6)
    assert fixed_authorization.optimize_camera_rays is False
    assert fixed_authorization.ray_bar_keys == ()
    assert fixed_authorization.grad_track_ray_coefficients_f64 is None
    assert fixed_accounting["camera_ray_gradients_enabled"] is False
    assert fixed_accounting["fixed_camera_avoids_global_ray_bar"] is True
    assert fixed_accounting["step_ray_bar_key_logical_bytes"] == 0
    assert fixed_accounting["request_delta_ray_bar_key_logical_bytes"] == 0
    assert fixed_accounting["step_accumulator_logical_tensor_bytes"] < (
        trainable_camera["result"].accounting[
            "step_accumulator_logical_tensor_bytes"
        ]
    )
    assert fixed_accounting["request_geometry_bar_tensor_bytes"] < (
        trainable_camera["result"].accounting[
            "request_geometry_bar_tensor_bytes"
        ]
    )


def test_request_delta_rejects_same_count_wrong_mode_tensor_pattern() -> None:
    def replace_site_bar_with_forbidden_ray_bar(
        delta,
        accumulator,
        source,
        request,
        artifact,
        replay,
    ) -> None:
        broken = replace(
            delta,
            grad_positions0_f64=None,
            grad_track_ray_coefficients_f64=delta.grad_positions0_f64,
        )
        with pytest.raises(ValueError, match="mode/tensors disagree"):
            broken.assert_current(accumulator, source, request, artifact, replay)

    _run(
        chunk_capacity=1,
        frame_indices=(0, 2),
        full_geometry=True,
        optimize_camera_rays=False,
        before_consume=replace_site_bar_with_forbidden_ray_bar,
    )


def test_full_geometry_chunk_size_changes_sampling_not_geometry_or_word_work() -> None:
    one = _run(
        chunk_capacity=1,
        frame_indices=(0, 2),
        full_geometry=True,
    )
    four = _run(
        chunk_capacity=4,
        frame_indices=(0, 2),
        full_geometry=True,
    )

    torch.testing.assert_close(one["loss"], four["loss"], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(one["bar"], four["bar"], rtol=2e-6, atol=2e-6)
    for left, right in zip(
        (
            one["authorization"].grad_positions0_f64,
            one["authorization"].grad_velocities_f64,
            one["authorization"].grad_weight_coefficients_f64,
            one["authorization"].grad_track_ray_coefficients_f64,
        ),
        (
            four["authorization"].grad_positions0_f64,
            four["authorization"].grad_velocities_f64,
            four["authorization"].grad_weight_coefficients_f64,
            four["authorization"].grad_track_ray_coefficients_f64,
        ),
        strict=True,
    ):
        torch.testing.assert_close(left, right, rtol=8e-5, atol=8e-6)
    assert (
        one["result"].telemetry.native_sample_launch_count
        > four["result"].telemetry.native_sample_launch_count
    )
    assert (
        one["result"].telemetry.native_full_geometry_vjp_launch_count
        == four["result"].telemetry.native_full_geometry_vjp_launch_count
    )
    assert (
        one["result"].accounting["maximum_native_length_bar_tensor_bytes"]
        == four["result"].accounting["maximum_native_length_bar_tensor_bytes"]
    )


def test_chunk_size_changes_sample_launches_not_loss_bar_or_word_work() -> None:
    one = _run(chunk_capacity=1, frame_indices=(0, 2))
    four = _run(chunk_capacity=4, frame_indices=(0, 2))

    torch.testing.assert_close(one["loss"], four["loss"], rtol=1.0e-6, atol=1.0e-6)
    torch.testing.assert_close(one["bar"], four["bar"], rtol=2.0e-6, atol=2.0e-6)
    assert one["result"].receipt.replay_chunk_count == 4
    assert four["result"].receipt.replay_chunk_count == 1
    assert one["result"].telemetry.native_sample_launch_count > four["result"].telemetry.native_sample_launch_count
    assert (
        one["result"].telemetry.native_node_forward_launch_count
        == four["result"].telemetry.native_node_forward_launch_count
    )
    assert (
        one["result"].telemetry.native_material_word_vjp_launch_count
        == four["result"].telemetry.native_material_word_vjp_launch_count
    )
    assert (
        one["result"].accounting["lane_resident_logical_tensor_bytes_upper_bound"]
        == four["result"].accounting["lane_resident_logical_tensor_bytes_upper_bound"]
    )


def test_more_frames_grow_only_chunk_sample_and_target_work() -> None:
    sparse = _run(chunk_capacity=2, frame_indices=(0,))
    dense = _run(chunk_capacity=2, frame_indices=(0, 1, 2))

    assert sparse["result"].telemetry.streamed_sample_count == 2
    assert dense["result"].telemetry.streamed_sample_count == 6
    assert dense["result"].telemetry.native_sample_launch_count > sparse["result"].telemetry.native_sample_launch_count
    assert (
        sparse["result"].telemetry.native_node_forward_launch_count
        == dense["result"].telemetry.native_node_forward_launch_count
    )
    assert (
        sparse["result"].telemetry.native_material_word_vjp_launch_count
        == dense["result"].telemetry.native_material_word_vjp_launch_count
    )
    assert (
        sparse["result"].accounting["lane_resident_logical_tensor_bytes_upper_bound"]
        == dense["result"].accounting["lane_resident_logical_tensor_bytes_upper_bound"]
    )
    assert sparse["artifact"].key.generation_digest == dense["artifact"].key.generation_digest
    assert sparse["artifact"].program_generation_digests == dense["artifact"].program_generation_digests
    assert (
        sparse["result"].accounting["artifact_structural_signature_sha256"]
        == dense["result"].accounting["artifact_structural_signature_sha256"]
    )
    assert (
        sparse["result"].accounting["compiled_camera_path_signature_sha256"]
        == dense["result"].accounting["compiled_camera_path_signature_sha256"]
    )
    assert (
        sparse["result"].accounting["sample_node_interaction_count"]
        < dense["result"].accounting["sample_node_interaction_count"]
    )


def test_full_geometry_more_frames_do_not_grow_word_or_geometry_bar_state() -> None:
    sparse = _run(
        chunk_capacity=2,
        frame_indices=(0,),
        full_geometry=True,
    )
    dense = _run(
        chunk_capacity=2,
        frame_indices=(0, 1, 2),
        full_geometry=True,
    )

    assert dense["result"].telemetry.streamed_sample_count > sparse["result"].telemetry.streamed_sample_count
    assert (
        dense["result"].telemetry.native_sample_launch_count
        > sparse["result"].telemetry.native_sample_launch_count
    )
    assert (
        dense["result"].telemetry.native_full_geometry_vjp_launch_count
        == sparse["result"].telemetry.native_full_geometry_vjp_launch_count
    )
    for key in (
        "lane_resident_logical_tensor_bytes_upper_bound",
        "request_geometry_bar_tensor_bytes",
        "step_accumulator_logical_tensor_bytes",
        "step_ray_bar_key_logical_bytes",
        "maximum_native_length_bar_tensor_bytes",
    ):
        assert dense["result"].accounting[key] == sparse["result"].accounting[key]
    for run in (sparse, dense):
        accounting = run["result"].accounting
        assert accounting["geometry_reduction_mode"] == "certified_sparse_compact"
        assert accounting["geometry_dense_global_site_accumulation_elements"] == 0
        assert accounting["geometry_all_site_owner_validation_evaluations"] == 0
        assert accounting["maximum_simultaneous_geometry_jw_length_bar_tensors"] == 1
        assert accounting["geometry_compact_to_global_scatter_elements"] > 0


def test_budget_failure_precedes_target_decode_poisons_replay_and_commits_nothing() -> None:
    (
        target_source,
        _factory,
        _provider_value,
        _store,
        artifact,
        source,
        replay,
        request,
    ) = _case(chunk_capacity=1, frame_indices=(0, 2))
    native_ops = _CoverageCheckingNativeOps()
    native_ops.replay_session = replay
    fence = _Fence()
    material = torch.tensor([[0.21, 0.37, 0.16, 0.8]], dtype=torch.float32)
    bar = torch.full_like(material, 7.0)
    loss = torch.tensor((5.0,), dtype=torch.float32)
    before_bar = bar.clone()
    before_loss = loss.clone()
    policy = _request_policy(maximum_chunk_target_tensor_bytes=11)
    loader = _target_loader(source, request, policy=policy)

    with pytest.raises(MemoryError, match="target budget fails before target load"):
        run_paper_kinetic_dense_cached_native_material_request(
            source,
            replay,
            request,
            artifact,
            step_generation_id="budget-failure",
            loss_normalization_id="global-rgb-mean-v1",
            material_generation_id="fixture-material-v1",
            background_generation_id="fixture-background-v1",
            global_site_rgba_f32=material,
            global_grad_site_rgba_f32=bar,
            loss_f32=loss,
            background_rgb_f32=torch.zeros((3,), dtype=torch.float32),
            native_ops=native_ops,
            backend_provenance="cpu-fake-native/exact-op-surface",
            maximum_samples_per_launch=1,
            memory_policy=policy,
            load_chunk_targets=loader,
            device_completion_fence=fence,
            device_completion_fence_provenance="cpu-synchronous-fence-v1",
        )
    assert loader.completed_load_count == 0
    assert loader._active_lifetime is None
    assert target_source.calls == []
    assert native_ops.material_vjp_calls == 0
    assert fence.calls == native_ops.sample_launch_calls + 1
    assert replay.poisoned
    torch.testing.assert_close(bar, before_bar)
    torch.testing.assert_close(loss, before_loss)


def test_lower_level_request_rejects_arbitrary_target_loader_before_replay() -> None:
    (
        _target_source,
        _factory,
        _provider_value,
        _store,
        artifact,
        source,
        replay,
        request,
    ) = _case(chunk_capacity=1, frame_indices=(0, 2))
    material = torch.tensor([[0.21, 0.37, 0.16, 0.8]], dtype=torch.float32)
    background = torch.zeros((3,), dtype=torch.float32)
    accumulator = prepare_paper_kinetic_dense_step_gradient_accumulator(
        source,
        replay,
        step_generation_id="reject-arbitrary-loader",
        loss_normalization_id="global-rgb-mean-v1",
        material_generation_id="fixture-material-v1",
        background_generation_id="fixture-background-v1",
        global_site_rgba_f32=material,
        background_rgb_f32=background,
        device="cpu",
        full_geometry=False,
    )

    with pytest.raises(TypeError, match="sealed target loader"):
        run_paper_kinetic_dense_cached_native_request(
            source,
            replay,
            request,
            artifact,
            accumulator,
            step_generation_id="reject-arbitrary-loader",
            loss_normalization_id="global-rgb-mean-v1",
            material_generation_id="fixture-material-v1",
            background_generation_id="fixture-background-v1",
            global_site_rgba_f32=material,
            background_rgb_f32=background,
            native_ops=_CoverageCheckingNativeOps(),
            backend_provenance="cpu-fake-native/exact-op-surface",
            maximum_samples_per_launch=1,
            memory_policy=_request_policy(),
            load_chunk_targets=lambda _chunk: None,
            device_completion_fence=_Fence(),
            device_completion_fence_provenance="cpu-synchronous-fence-v1",
        )
    assert replay.request_count == 0
    assert replay.emitted_observation_count == 0
    assert not replay.poisoned
    assert not accumulator.poisoned


def test_mid_replay_target_failure_has_no_partial_caller_commit() -> None:
    (
        _target_source,
        _factory,
        _provider_value,
        _store,
        artifact,
        source,
        replay,
        request,
    ) = _case(chunk_capacity=1, frame_indices=(0, 2))
    native_ops = _CoverageCheckingNativeOps()
    native_ops.replay_session = replay
    fence = _Fence()
    material = torch.tensor([[0.21, 0.37, 0.16, 0.8]], dtype=torch.float32)
    background = torch.zeros((3,), dtype=torch.float32)
    accumulator = prepare_paper_kinetic_dense_step_gradient_accumulator(
        source,
        replay,
        step_generation_id="mid-replay-failure",
        loss_normalization_id="global-rgb-mean-v1",
        material_generation_id="fixture-material-v1",
        background_generation_id="fixture-background-v1",
        global_site_rgba_f32=material,
        background_rgb_f32=background,
        device="cpu",
        full_geometry=True,
        optimize_camera_rays=True,
    )
    loader = _target_loader(
        source,
        request,
        source_test_fault=(
            prepare_paper_kinetic_dense_chunk_target_loader_test_fault(
                message="injected target failure",
                fail_on_load_number=2,
            )
        ),
    )

    with pytest.raises(RuntimeError, match="injected target failure"):
        run_paper_kinetic_dense_cached_native_request(
            source,
            replay,
            request,
            artifact,
            accumulator,
            step_generation_id="mid-replay-failure",
            loss_normalization_id="global-rgb-mean-v1",
            material_generation_id="fixture-material-v1",
            background_generation_id="fixture-background-v1",
            global_site_rgba_f32=material,
            background_rgb_f32=background,
            native_ops=native_ops,
            backend_provenance="cpu-fake-native/exact-op-surface",
            maximum_samples_per_launch=1,
            memory_policy=_request_policy(),
            load_chunk_targets=loader,
            device_completion_fence=fence,
            device_completion_fence_provenance="cpu-synchronous-fence-v1",
        )
    assert loader.completed_load_count == 1
    assert loader.failed_after_enqueue_count == 1
    assert loader._active_lifetime is None
    assert native_ops.sample_launch_calls == 1
    assert native_ops.material_vjp_calls == 0
    assert native_ops.vjp_calls == 0
    assert replay.poisoned
    assert fence.calls == native_ops.sample_launch_calls + 1
    assert accumulator.poisoned
    assert not accumulator.optimizer_authorized
    assert all(
        not bool(torch.any(tensor != 0.0).item())
        for tensor in accumulator._tensors()
    )


def test_sample_settlement_fence_failure_quarantines_launch_roots_and_rejects_abort(
) -> None:
    (
        _target_source,
        _factory,
        _provider_value,
        _store,
        artifact,
        source,
        replay,
        request,
    ) = _case(chunk_capacity=2, frame_indices=(0, 2))
    native_ops = _CoverageCheckingNativeOps()
    native_ops.replay_session = replay
    material = torch.tensor([[0.21, 0.37, 0.16, 0.8]], dtype=torch.float32)
    background = torch.tensor((0.03, 0.05, 0.07), dtype=torch.float32)
    accumulator = prepare_paper_kinetic_dense_step_gradient_accumulator(
        source,
        replay,
        step_generation_id="sample-settlement-failure",
        loss_normalization_id="global-rgb-mean-v1",
        material_generation_id="fixture-material-v1",
        background_generation_id="fixture-background-v1",
        global_site_rgba_f32=material,
        background_rgb_f32=background,
        device="cpu",
        full_geometry=False,
    )
    fence_calls = 0

    loader = _target_loader(
        source,
        request,
        target_generation_id="sample-settlement-failure-targets",
    )

    def fail_sample_settlement_fence() -> None:
        nonlocal fence_calls
        fence_calls += 1
        raise RuntimeError("injected sample settlement fence failure")

    with pytest.raises(
        RuntimeError,
        match="injected sample settlement fence failure",
    ) as caught:
        run_paper_kinetic_dense_cached_native_request(
            source,
            replay,
            request,
            artifact,
            accumulator,
            step_generation_id="sample-settlement-failure",
            loss_normalization_id="global-rgb-mean-v1",
            material_generation_id="fixture-material-v1",
            background_generation_id="fixture-background-v1",
            global_site_rgba_f32=material,
            background_rgb_f32=background,
            native_ops=native_ops,
            backend_provenance="cpu-fake-native/exact-op-surface",
            maximum_samples_per_launch=2,
            memory_policy=_request_policy(),
            load_chunk_targets=loader,
            device_completion_fence=fail_sample_settlement_fence,
            device_completion_fence_provenance=(
                "injected-sample-settlement-failing-fence-v1"
            ),
        )

    assert native_ops.sample_prepare_calls == 1
    assert native_ops.sample_launch_calls == 1
    # Abort rejects unknown completion without issuing a second fence.
    assert fence_calls == 1
    assert accumulator.poisoned
    assert not accumulator.optimizer_authorized
    quarantine = accumulator._async_failure_quarantine
    assert quarantine is not None
    quarantine.assert_current()
    assert quarantine.stage == "native-session-abort"
    assert str(quarantine.original_error) == (
        "injected sample settlement fence failure"
    )
    assert "sample completion is unknown" in str(quarantine.cleanup_fence_error)
    assert caught.value.__cause__ is quarantine.cleanup_fence_error
    assert quarantine.restart_required

    retained = dict(
        zip(
            quarantine.retained_reference_roles,
            quarantine.retained_references,
            strict=True,
        )
    )
    required_roles = {
        "native_session",
        "current_chunk_targets",
        "current_chunk_cpu_transfer_source",
        "current_sample_materialization",
        "current_sample_block",
        "current_sample_lifetime",
        "session_outstanding_sample_lifetime",
        "active_blocks",
    }
    assert required_roles.issubset(retained)

    targets = retained["current_chunk_targets"]
    targets.assert_transfer_retained()
    assert (
        retained["current_chunk_cpu_transfer_source"]
        is targets._cpu_transfer_source_ref
    )
    active_blocks = retained["active_blocks"]
    assert active_blocks
    for block_state in active_blocks.values():
        block_state.compact_gather_lifetime.assert_retained()
        block_state.forward_into_lifetime.assert_retained(
            retained["native_session"]
        )
    materialization = retained["current_sample_materialization"]
    materialization.assert_retained()
    sample_block = retained["current_sample_block"]
    assert materialization.sample_block is sample_block
    assert materialization.chunk_target_rgb_f32 is targets.target_rgb_f32
    assert not materialization.released_after_completion_fence

    native_session = retained["native_session"]
    lifetime = retained["current_sample_lifetime"]
    assert lifetime is retained["session_outstanding_sample_lifetime"]
    assert lifetime is native_session._outstanding_sample_lifetime
    lifetime.assert_retained(native_session)
    assert lifetime.phase == "completion_unknown"
    assert lifetime.completion_unknown
    assert lifetime.completion_fence_attempt_count == 1
    assert not lifetime.consumed
    assert lifetime.sample_block is sample_block

    prepared = lifetime.prepared_payload
    assert prepared is not None
    assert prepared.node_chart_f32 is lifetime.world_token.world.node_chart_f32
    assert prepared.sample_row_i32 is sample_block.sample_row_i32
    assert prepared.sample_to_node_f32 is sample_block.sample_to_node_f32
    assert prepared.target_rgb_f32 is sample_block.target_rgb_f32
    assert prepared.background_rgb_f32 is lifetime.background_rgb_f32


def test_forward_into_enqueue_failure_and_failed_abort_fence_quarantine_all_roots(
) -> None:
    class _ForwardIntoRaisesAfterWrite(_CoverageCheckingNativeOps):
        def kinetic_precompiled_length_p0_lie_node_forward_into_launch_only_v1(
            self,
            *args,
            **kwargs,
        ):
            super().kinetic_precompiled_length_p0_lie_node_forward_into_launch_only_v1(
                *args,
                **kwargs,
            )
            raise RuntimeError("injected forward-into enqueue failure")

    (
        _target_source,
        _factory,
        _provider_value,
        _store,
        artifact,
        source,
        replay,
        request,
    ) = _case(chunk_capacity=2, frame_indices=(0, 2))
    native_ops = _ForwardIntoRaisesAfterWrite()
    native_ops.replay_session = replay
    material = torch.tensor([[0.21, 0.37, 0.16, 0.8]], dtype=torch.float32)
    background = torch.tensor((0.03, 0.05, 0.07), dtype=torch.float32)
    accumulator = prepare_paper_kinetic_dense_step_gradient_accumulator(
        source,
        replay,
        step_generation_id="forward-into-enqueue-failure",
        loss_normalization_id="global-rgb-mean-v1",
        material_generation_id="fixture-material-v1",
        background_generation_id="fixture-background-v1",
        global_site_rgba_f32=material,
        background_rgb_f32=background,
        device="cpu",
        full_geometry=False,
    )
    loader = _target_loader(
        source,
        request,
        target_generation_id="forward-into-enqueue-failure-targets",
    )
    fence_calls = 0

    def fail_abort_fence() -> None:
        nonlocal fence_calls
        fence_calls += 1
        raise RuntimeError("injected forward-into abort fence failure")

    with pytest.raises(
        RuntimeError,
        match="injected forward-into enqueue failure",
    ) as caught:
        run_paper_kinetic_dense_cached_native_request(
            source,
            replay,
            request,
            artifact,
            accumulator,
            step_generation_id="forward-into-enqueue-failure",
            loss_normalization_id="global-rgb-mean-v1",
            material_generation_id="fixture-material-v1",
            background_generation_id="fixture-background-v1",
            global_site_rgba_f32=material,
            background_rgb_f32=background,
            native_ops=native_ops,
            backend_provenance="cpu-fake-native/exact-op-surface",
            maximum_samples_per_launch=2,
            memory_policy=_request_policy(),
            load_chunk_targets=loader,
            device_completion_fence=fail_abort_fence,
            device_completion_fence_provenance=(
                "injected-forward-into-abort-failing-fence-v1"
            ),
        )

    assert fence_calls == 1
    assert native_ops.return_allocating_forward_calls == 0
    assert native_ops.forward_into_calls == 1
    assert native_ops.sample_launch_calls == 0
    quarantine = accumulator._async_failure_quarantine
    assert quarantine is not None
    quarantine.assert_current()
    assert quarantine.stage == "native-session-abort"
    assert str(quarantine.original_error) == (
        "injected forward-into enqueue failure"
    )
    assert str(quarantine.cleanup_fence_error) == (
        "injected forward-into abort fence failure"
    )
    assert caught.value.__cause__ is quarantine.cleanup_fence_error
    retained = dict(
        zip(
            quarantine.retained_reference_roles,
            quarantine.retained_references,
            strict=True,
        )
    )
    assert retained["active_blocks"] == {}
    assert retained["current_forward_runtime"] is not None
    assert retained["current_forward_compact_material"] is not None
    assert retained["current_forward_node_chart_out"] is not None
    assert "current_forward_token" not in retained
    retained["current_compact_gather_lifetime"].assert_retained()
    retained["current_forward_into_lifetime"].assert_retained(
        retained["native_session"]
    )


def test_partial_lane_failure_and_failed_fence_quarantine_construction_roots(
    monkeypatch,
) -> None:
    (
        _target_source,
        _factory,
        _provider_value,
        _store,
        artifact,
        source,
        replay,
        request,
    ) = _case(chunk_capacity=2, frame_indices=(0, 2))
    native_ops = _CoverageCheckingNativeOps()
    material = torch.tensor([[0.21, 0.37, 0.16, 0.8]], dtype=torch.float32)
    background = torch.zeros((3,), dtype=torch.float32)
    accumulator = prepare_paper_kinetic_dense_step_gradient_accumulator(
        source,
        replay,
        step_generation_id="partial-lane-failure",
        loss_normalization_id="global-rgb-mean-v1",
        material_generation_id="fixture-material-v1",
        background_generation_id="fixture-background-v1",
        global_site_rgba_f32=material,
        background_rgb_f32=background,
        device="cpu",
        full_geometry=False,
    )
    fence_calls = 0

    def fail_runtime_prepare(*_args, **_kwargs):
        raise RuntimeError("injected partial lane construction")

    def fail_completion_fence() -> None:
        nonlocal fence_calls
        fence_calls += 1
        raise RuntimeError("injected partial-lane fence failure")

    loader = _target_loader(source, request)

    monkeypatch.setattr(
        dense_request_module,
        "prepare_kinetic_native_equal_rank_runtime_block",
        fail_runtime_prepare,
    )
    with pytest.raises(RuntimeError, match="injected partial lane construction"):
        run_paper_kinetic_dense_cached_native_request(
            source,
            replay,
            request,
            artifact,
            accumulator,
            step_generation_id="partial-lane-failure",
            loss_normalization_id="global-rgb-mean-v1",
            material_generation_id="fixture-material-v1",
            background_generation_id="fixture-background-v1",
            global_site_rgba_f32=material,
            background_rgb_f32=background,
            native_ops=native_ops,
            backend_provenance="cpu-fake-native/exact-op-surface",
            maximum_samples_per_launch=2,
            memory_policy=_request_policy(),
            load_chunk_targets=loader,
            device_completion_fence=fail_completion_fence,
            device_completion_fence_provenance=(
                "injected-partial-lane-failing-fence-v1"
            ),
        )
    quarantine = accumulator._async_failure_quarantine
    assert fence_calls == 1
    assert accumulator.poisoned
    assert quarantine is not None
    quarantine.assert_current()
    assert quarantine.stage == "partial-lane-construction"
    assert quarantine.original_traceback is not None
    assert str(quarantine.original_error) == "injected partial lane construction"
    assert str(quarantine.cleanup_fence_error) == "injected partial-lane fence failure"
    assert quarantine.restart_required
    assert loader.completed_load_count == 0
    assert loader._active_lifetime is None
    assert {
        "dense_lane_construction_lifetime",
        "spatial_bundle_construction_lifetime",
        "spatial_bundle",
        "payloads",
        "runtime_construction_lifetimes",
        "runtimes",
        "native_ops",
    }.issubset(
        quarantine.retained_reference_roles
    )
    retained = dict(
        zip(
            quarantine.retained_reference_roles,
            quarantine.retained_references,
            strict=True,
        )
    )
    retained["dense_lane_construction_lifetime"].assert_retained()


def test_post_replay_release_failure_poisons_session_and_step() -> None:
    (
        _target_source,
        _factory,
        _provider_value,
        _store,
        artifact,
        source,
        replay,
        request,
    ) = _case(chunk_capacity=2, frame_indices=(0, 2))
    native_ops = _CoverageCheckingNativeOps()
    native_ops.replay_session = replay
    material = torch.tensor([[0.21, 0.37, 0.16, 0.8]], dtype=torch.float32)
    background = torch.tensor((0.03, 0.05, 0.07), dtype=torch.float32)
    accumulator = prepare_paper_kinetic_dense_step_gradient_accumulator(
        source,
        replay,
        step_generation_id="release-failure",
        loss_normalization_id="global-rgb-mean-v1",
        material_generation_id="fixture-material-v1",
        background_generation_id="fixture-background-v1",
        global_site_rgba_f32=material,
        background_rgb_f32=background,
        device="cpu",
        full_geometry=False,
    )
    accumulator_tensor_versions_before = tuple(
        int(tensor._version) for tensor in accumulator._tensors()
    )
    fence_calls = 0

    loader = _target_loader(
        source,
        request,
        target_generation_id="release-failure-targets",
    )

    def failing_fence() -> None:
        nonlocal fence_calls
        fence_calls += 1
        if fence_calls == (
            native_ops.sample_launch_calls + native_ops.material_vjp_calls + 1
        ):
            raise RuntimeError("injected release fence failure")

    with pytest.raises(RuntimeError, match="injected release fence failure"):
        run_paper_kinetic_dense_cached_native_request(
            source,
            replay,
            request,
            artifact,
            accumulator,
            step_generation_id="release-failure",
            loss_normalization_id="global-rgb-mean-v1",
            material_generation_id="fixture-material-v1",
            background_generation_id="fixture-background-v1",
            global_site_rgba_f32=material,
            background_rgb_f32=background,
            native_ops=native_ops,
            backend_provenance="cpu-fake-native/exact-op-surface",
            maximum_samples_per_launch=2,
            memory_policy=_request_policy(),
            load_chunk_targets=loader,
            device_completion_fence=failing_fence,
            device_completion_fence_provenance="injected-failing-fence-v1",
        )
    assert fence_calls == (
        native_ops.sample_launch_calls + native_ops.material_vjp_calls + 1
    )
    assert replay.request_count == 1
    assert replay.emitted_observation_count == source.observation_count
    assert replay.poisoned
    assert accumulator.poisoned
    assert not accumulator.pending_delta_generation_digest
    quarantine = accumulator._async_failure_quarantine
    assert quarantine is not None
    quarantine.assert_current()
    assert quarantine.stage == "outer-lane-release"
    assert quarantine.original_error is quarantine.cleanup_fence_error
    assert quarantine.restart_required
    assert {"lane", "native_session", "active_blocks"}.issubset(
        quarantine.retained_reference_roles
    )
    fail_stop_paper_kinetic_dense_step(accumulator, source, replay)
    assert tuple(
        int(tensor._version) for tensor in accumulator._tensors()
    ) == accumulator_tensor_versions_before
    assert all(
        not bool(torch.any(tensor != 0.0).item())
        for tensor in accumulator._tensors()
    )


def test_post_enqueue_target_failure_and_failed_abort_fence_quarantines_loader_lifetime(
) -> None:
    (
        _target_source,
        _factory,
        _provider_value,
        _store,
        artifact,
        source,
        replay,
        request,
    ) = _case(chunk_capacity=2, frame_indices=(0, 2))
    native_ops = _CoverageCheckingNativeOps()
    material = torch.tensor([[0.21, 0.37, 0.16, 0.8]], dtype=torch.float32)
    background = torch.zeros((3,), dtype=torch.float32)
    accumulator = prepare_paper_kinetic_dense_step_gradient_accumulator(
        source,
        replay,
        step_generation_id="abort-fence-failure",
        loss_normalization_id="global-rgb-mean-v1",
        material_generation_id="fixture-material-v1",
        background_generation_id="fixture-background-v1",
        global_site_rgba_f32=material,
        background_rgb_f32=background,
        device="cpu",
        full_geometry=False,
    )
    fence_calls = 0

    loader = _target_loader(
        source,
        request,
        target_generation_id="post-enqueue-failure-targets",
        source_test_fault=(
            prepare_paper_kinetic_dense_chunk_target_loader_test_fault(
                message="injected request body failure"
            )
        ),
    )

    def fail_abort_fence() -> None:
        nonlocal fence_calls
        fence_calls += 1
        raise RuntimeError("injected abort fence failure")

    with pytest.raises(RuntimeError, match="injected request body failure"):
        run_paper_kinetic_dense_cached_native_request(
            source,
            replay,
            request,
            artifact,
            accumulator,
            step_generation_id="abort-fence-failure",
            loss_normalization_id="global-rgb-mean-v1",
            material_generation_id="fixture-material-v1",
            background_generation_id="fixture-background-v1",
            global_site_rgba_f32=material,
            background_rgb_f32=background,
            native_ops=native_ops,
            backend_provenance="cpu-fake-native/exact-op-surface",
            maximum_samples_per_launch=2,
            memory_policy=_request_policy(),
            load_chunk_targets=loader,
            device_completion_fence=fail_abort_fence,
            device_completion_fence_provenance=(
                "injected-abort-failing-fence-v1"
            ),
        )
    quarantine = accumulator._async_failure_quarantine
    assert fence_calls == 1
    assert accumulator.poisoned
    assert quarantine is not None
    quarantine.assert_current()
    assert quarantine.stage == "native-session-abort"
    assert str(quarantine.original_error) == "injected request body failure"
    assert str(quarantine.cleanup_fence_error) == "injected abort fence failure"
    assert quarantine.restart_required
    assert {
        "lane",
        "native_session",
        "active_blocks",
        "target_loader",
        "target_loader_active_lifetime",
    }.issubset(
        quarantine.retained_reference_roles
    )
    retained = dict(
        zip(
            quarantine.retained_reference_roles,
            quarantine.retained_references,
            strict=True,
        )
    )
    lifetime = retained["target_loader_active_lifetime"]
    assert retained["target_loader"] is loader
    assert loader._active_lifetime is lifetime
    assert lifetime.phase == "failed_after_enqueue"
    assert lifetime.chunk_ref.generation_digest == lifetime.chunk_generation_digest
    assert lifetime.selected_read_ref is not None
    assert lifetime.cpu_transfer_source_ref is not None
    assert lifetime.device_tensor_refs
    assert lifetime.failure_ref is quarantine.original_error
    assert loader.failed_after_enqueue_count == 1
    assert not lifetime.completion_fence_proven
    assert not lifetime.released


def test_two_track_local_samplers_share_one_world_bound_step_and_authorize_once() -> None:
    _target_source, factory, provider = _provider()
    store = PaperKineticCompiledCpuArtifactStore(
        PaperKineticCompiledCpuArtifactStorePolicy(
            maximum_entries=2,
            maximum_resident_accounted_bytes=20_000_000,
        )
    )
    artifacts = tuple(
        store.acquire(
            provider,
            view_index=0,
            track_ids=(track_id,),
            maximum_artifact_accounted_bytes=10_000_000,
            compile_artifact=lambda key, selected=track_id: _compile_artifact(
                provider,
                _observations((0, selected)),
                key,
            ),
        ).artifact
        for track_id in (0, 1)
    )
    assert artifacts[0].sampler.generation_digest != artifacts[1].sampler.generation_digest
    source = prepare_paper_kinetic_replayable_dense_observation_source(
        provider,
        _batch((0, 2)),
        memory_policy=_dense_policy(2),
    )
    replay = source.open_session()
    requests = tuple(
        source.prepare_track_request(view_index=0, track_ids=(track_id,))
        for track_id in (0, 1)
    )
    material = torch.tensor([[0.21, 0.37, 0.16, 0.8]], dtype=torch.float32)
    background = torch.tensor((0.03, 0.05, 0.07), dtype=torch.float32)
    accumulator = prepare_paper_kinetic_dense_step_gradient_accumulator(
        source,
        replay,
        step_generation_id="two-request-step-v1",
        loss_normalization_id="global-rgb-mean-v1",
        material_generation_id="fixture-material-v1",
        background_generation_id="fixture-background-v1",
        global_site_rgba_f32=material,
        background_rgb_f32=background,
        device="cpu",
        full_geometry=True,
        optimize_camera_rays=True,
    )
    native_ops = _CoverageCheckingNativeOps()
    native_ops.replay_session = replay
    fence = _Fence()

    def run_one(index: int):
        request = requests[index]
        loader = _target_loader(
            source,
            request,
            target_generation_id=f"two-request-targets-{index}",
        )

        return run_paper_kinetic_dense_cached_native_request(
            source,
            replay,
            request,
            artifacts[index],
            accumulator,
            step_generation_id="two-request-step-v1",
            loss_normalization_id="global-rgb-mean-v1",
            material_generation_id="fixture-material-v1",
            background_generation_id="fixture-background-v1",
            global_site_rgba_f32=material,
            background_rgb_f32=background,
            native_ops=native_ops,
            backend_provenance="cpu-fake-native/exact-op-surface",
            maximum_samples_per_launch=2,
            memory_policy=_request_policy(),
            load_chunk_targets=loader,
            device_completion_fence=fence,
            device_completion_fence_provenance="cpu-synchronous-fence-v1",
        )

    first = run_one(0)
    with pytest.raises(ValueError, match="consume its pending request"):
        run_one(1)
    first_commit_receipt = consume_paper_kinetic_dense_request_delta(
        accumulator,
        source,
        replay,
        requests[0],
        artifacts[0],
        first.delta,
        device_completion_fence=fence,
        device_completion_fence_provenance="cpu-synchronous-fence-v1",
    )
    with pytest.raises(ValueError, match="already consumed"):
        consume_paper_kinetic_dense_request_delta(
            accumulator,
            source,
            replay,
            requests[0],
            artifacts[0],
            first.delta,
            device_completion_fence=fence,
            device_completion_fence_provenance="cpu-synchronous-fence-v1",
        )
    original_material = material
    material = material.clone()
    with pytest.raises(ValueError, match="material/background snapshot"):
        run_one(1)
    assert replay.request_count == accumulator.consumed_request_count == 1
    assert not accumulator.poisoned
    material = original_material
    second = run_one(1)
    second_commit_receipt = consume_paper_kinetic_dense_request_delta(
        accumulator,
        source,
        replay,
        requests[1],
        artifacts[1],
        second.delta,
        device_completion_fence=fence,
        device_completion_fence_provenance="cpu-synchronous-fence-v1",
    )
    assert accumulator.consumed_request_count == 2
    assert accumulator.fenced_request_commit_count == 2
    assert accumulator.consumed_observation_count == source.observation_count
    assert not accumulator.optimizer_authorized
    replay_receipt = replay.seal()
    authorization = authorize_paper_kinetic_dense_optimizer_step(
        accumulator,
        source,
        replay,
        replay_receipt,
    )
    authorization.assert_current(accumulator, replay_receipt)
    first_commit_receipt.assert_current(
        accumulator,
        source,
        replay,
        requests[0],
        artifacts[0],
        first.delta,
    )
    second_commit_receipt.assert_current(
        accumulator,
        source,
        replay,
        requests[1],
        artifacts[1],
        second.delta,
    )
    assert authorization.request_count == 2
    assert authorization.observation_count == source.observation_count
    assert authorization.ray_bar_keys == ((0, 0), (0, 1))
    assert first.delta.logical_tensor_bytes == second.delta.logical_tensor_bytes == 0


def test_fused_selector_requires_explicit_caps_before_lane_or_target_work() -> None:
    (
        _target_source,
        _factory,
        _provider_value,
        _store,
        artifact,
        source,
        replay,
        request,
    ) = _case(chunk_capacity=1, frame_indices=(0, 2))
    material = torch.tensor([[0.21, 0.37, 0.16, 0.8]], dtype=torch.float32)
    background = torch.tensor((0.03, 0.05, 0.07), dtype=torch.float32)
    accumulator = prepare_paper_kinetic_dense_step_gradient_accumulator(
        source,
        replay,
        step_generation_id="dense-fused-cap-step-v1",
        loss_normalization_id="global-rgb-mean-v1",
        material_generation_id="fixture-material-v1",
        background_generation_id="fixture-background-v1",
        global_site_rgba_f32=material,
        background_rgb_f32=background,
        device="cpu",
        full_geometry=True,
        optimize_camera_rays=False,
    )
    loader = _target_loader(source, request)

    with pytest.raises(MemoryError, match="fused prepared payload"):
        run_paper_kinetic_dense_cached_native_request(
            source,
            replay,
            request,
            artifact,
            accumulator,
            step_generation_id="dense-fused-cap-step-v1",
            loss_normalization_id="global-rgb-mean-v1",
            material_generation_id="fixture-material-v1",
            background_generation_id="fixture-background-v1",
            global_site_rgba_f32=material,
            background_rgb_f32=background,
            native_ops=_CoverageCheckingNativeOps(),
            backend_provenance="cpu-fake-native/exact-op-surface",
            maximum_samples_per_launch=2,
            memory_policy=_request_policy(),
            load_chunk_targets=loader,
            device_completion_fence=_Fence(),
            device_completion_fence_provenance="cpu-synchronous-fence-v1",
            full_geometry_reverse_mode=(
                FUSED_DIRECT_V1_FULL_GEOMETRY_REVERSE
            ),
        )
    assert loader.completed_load_count == 0
    assert loader._active_lifetime is None
    assert replay.request_count == 0
    assert replay.emitted_observation_count == 0
    assert accumulator.poisoned
    assert not accumulator.optimizer_authorized


def test_cpu_fake_fused_bridge_and_compact_union_scatter_preserve_public_abi() -> None:
    (
        _target_source,
        _factory,
        provider,
        _store,
        artifact,
        _source,
        _replay,
        request,
    ) = _case(chunk_capacity=2, frame_indices=(0, 2))
    bundle = prepare_paper_kinetic_union_local_spatial_bundle(
        artifact.sampler,
        track_ids=request.track_ids,
        device="cpu",
    )
    block_ids = tuple(
        binding.native_block_generation_digest
        for binding in bundle.native_blocks
    )
    compact_bars = tuple(
        torch.full(
            (binding.compact_site_count, 4),
            float(index + 1),
            dtype=torch.float32,
        )
        for index, binding in enumerate(bundle.native_blocks)
    )
    active = {
        block_id: SimpleNamespace(
            loss_f32=torch.tensor((0.25 * (index + 1),), dtype=torch.float32)
        )
        for index, block_id in enumerate(block_ids)
    }
    union = torch.zeros((bundle.union_site_count, 4), dtype=torch.float32)
    loss = torch.zeros((1,), dtype=torch.float32)
    expected_union = torch.zeros_like(union)
    for binding, compact in zip(bundle.native_blocks, compact_bars, strict=True):
        expected_union.index_add_(0, binding.compact_to_union_i64, compact)
    scatter_elements = dense_request_module._accumulate_fused_compact_material_and_loss(
        bundle,
        active,
        active_block_generation_ids=block_ids,
        grad_compact_site_rgba_f32_by_block=compact_bars,
        local_union_bar=union,
        local_loss=loss,
    )
    torch.testing.assert_close(union, expected_union)
    assert float(loss.item()) == sum(
        float(state.loss_f32.item()) for state in active.values()
    )
    assert scatter_elements == sum(tensor.numel() for tensor in compact_bars)

    sites = provider.world.sites
    source_geometry = (
        torch.arange(sites.positions0.numel(), dtype=torch.float32).reshape(
            sites.positions0.shape
        ),
        torch.arange(sites.velocities.numel(), dtype=torch.float32).reshape(
            sites.velocities.shape
        ).add_(0.5),
        torch.arange(
            sites.weight_coefficients.numel(), dtype=torch.float32
        ).reshape(sites.weight_coefficients.shape).sub_(0.25),
    )
    fake_result = SimpleNamespace(
        grad_global_positions0_f32=source_geometry[0],
        grad_global_velocities_f32=source_geometry[1],
        grad_global_weight_coefficients_f32=source_geometry[2],
    )
    bridge_cap = (
        dense_request_module._fused_geometry_bridge_visible_peak_logical_tensor_bytes(
            artifact.sampler
        )
    )
    _, global_f32_output_bytes, _ = (
        dense_request_module._fused_output_scratch_logical_tensor_bytes_upper_bound(
            artifact
        )
    )
    request_f64_geometry_bytes = dense_request_module._request_geometry_bar_bytes(
        artifact.sampler,
        request,
        include_ray_gradients=False,
    )
    assert bridge_cap == global_f32_output_bytes + request_f64_geometry_bytes
    bridged, bridge_bytes = (
        dense_request_module._bridge_fused_global_geometry_bars_to_cpu_f64(
            fake_result,
            artifact.sampler,
            maximum_bridge_visible_peak_logical_tensor_bytes=bridge_cap,
        )
    )
    assert bridge_bytes == bridge_cap
    assert all(tensor.dtype == torch.float64 for tensor in bridged)
    assert all(tensor.device.type == "cpu" for tensor in bridged)
    for actual, expected in zip(bridged, source_geometry, strict=True):
        torch.testing.assert_close(actual, expected.to(dtype=torch.float64))


def test_dense_request_source_has_one_fused_transaction_and_no_staged_fallback() -> None:
    source = Path(dense_request_module.__file__).read_text(encoding="utf-8")
    fused_branch = source.split(
        "        if fused_full_geometry:\n            prepared_blocks = []",
        1,
    )[1].split("\n        for runtime in lane.runtimes:", 1)[0]
    assert fused_branch.count(
        "native_session.execute_fused_full_geometry_vjp_transaction("
    ) == 1
    assert "prepare_kinetic_native_equal_rank_fused_direct_full_vjp_v1(" in fused_branch
    assert "_accumulate_fused_compact_material_and_loss(" in fused_branch
    assert "_bridge_fused_global_geometry_bars_to_cpu_f64(" in fused_branch
    assert "invoke_device_completion_fence()" in fused_branch
    assert "active.clear()" in fused_branch
    assert "launch_full_geometry_vjp(" not in fused_branch
    assert "launch_material_vjp(" not in fused_branch
    lifetime_roots = source.split(
        "    def request_lifetime_references()",
        1,
    )[1].split("    try:", 1)[0]
    for role in (
        "fused_prepared_blocks",
        "fused_execution_receipt",
        "fused_transaction_result",
    ):
        assert role in lifetime_roots
    post_seal = source.split(
        "        telemetry = native_session.seal()",
        1,
    )[1].split("        if (", 1)[0]
    assert "fused_prepared_blocks = ()" in post_seal
    assert "fused_execution_receipt = None" in post_seal
    assert "fused_transaction_result = None" in post_seal
