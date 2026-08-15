from __future__ import annotations

import gc
import weakref
from dataclasses import dataclass
from pathlib import Path

import paper_kinetic_fixed_site_material_step as step_module
import pytest
import torch
from kinetic_compiled_cpu_artifact_store import (
    PaperKineticCompiledCpuArtifactStore,
    PaperKineticCompiledCpuArtifactStorePolicy,
)
from kinetic_dense_cached_native_material_request import (
    PaperKineticDenseCachedNativeMemoryPolicy,
)
from paper_kinetic_fixed_site_material_step import (
    GLOBAL_RGB_MEAN_LOSS_NORMALIZATION_ID,
    PaperKineticFixedSiteMaterialOnlyGenerationPolicy,
    PaperKineticFixedSiteMaterialOnlyStepPolicy,
    PaperKineticFixedSiteMaterialStepPartialFailure,
    prepare_paper_kinetic_fixed_site_material_step_state,
    run_paper_kinetic_fixed_site_material_only_step,
)
from paper_kinetic_fixed_site_material_state import (
    PaperKineticFixedSiteMaterialParameterization,
    PaperKineticFixedSiteMaterialSGDPolicy,
    apply_paper_kinetic_fixed_site_material_sgd_step,
    checkpoint_paper_kinetic_fixed_site_material_state,
    paper_kinetic_fixed_site_material_checkpoint_from_payload,
    prepare_paper_kinetic_fixed_site_material_state,
    restore_paper_kinetic_fixed_site_material_state,
)
from paper_kinetic_fixed_site_material_device_bridge import (
    apply_paper_kinetic_fixed_site_material_device_gradient_receipt,
    seal_paper_kinetic_fixed_site_material_device_gradient_receipt,
    snapshot_paper_kinetic_fixed_site_material_to_device,
)
from paper_kinetic_replayable_observations import (
    OBSERVATION_IDENTITY_LOGICAL_BYTES,
    TRACK_ID_LOGICAL_BYTES,
    PaperKineticDenseObservationMemoryPolicy,
)
from paper_training_types import SpacetimeBatch, SpacetimeSample
from paper_kinetic_world_initializer import (
    prepare_paper_kinetic_point_cloud_world_initializer,
)
from test_kinetic_compiled_cpu_artifact_store import _provider
from test_kinetic_ragged_paper_step_cpu_fake_native import _FakeNativeOps


@dataclass
class _Fence:
    calls: int = 0

    def __call__(self) -> None:
        self.calls += 1


def _batch() -> SpacetimeBatch:
    return SpacetimeBatch(
        samples=(
            SpacetimeSample(view_index=0, frame_index=0),
            SpacetimeSample(view_index=1, frame_index=1),
        ),
        epoch=3,
        batch_index=4,
        completes_epoch=False,
    )


def _policy(
    *,
    maximum_tracks_per_request: int = 2,
) -> PaperKineticFixedSiteMaterialOnlyStepPolicy:
    observation = PaperKineticDenseObservationMemoryPolicy(
        maximum_persistent_observation_count=0,
        maximum_persistent_observation_logical_bytes=0,
        maximum_retained_frame_metadata_count=16,
        maximum_retained_frame_metadata_logical_bytes=4096,
        maximum_live_generated_observation_count=1,
        maximum_live_generated_observation_logical_bytes=(
            OBSERVATION_IDENTITY_LOGICAL_BYTES
        ),
        maximum_request_track_count=max(2, maximum_tracks_per_request),
        maximum_request_track_logical_bytes=(
            max(2, maximum_tracks_per_request) * TRACK_ID_LOGICAL_BYTES
        ),
        maximum_chunk_observation_count=2,
        maximum_chunk_observation_logical_bytes=(
            2 * OBSERVATION_IDENTITY_LOGICAL_BYTES
        ),
    )
    request = PaperKineticDenseCachedNativeMemoryPolicy(
        maximum_lane_resident_logical_tensor_bytes=10_000_000,
        maximum_active_node_and_union_bar_tensor_bytes=10_000_000,
        maximum_decoded_frame_scratch_tensor_bytes=10_000_000,
        maximum_chunk_target_tensor_bytes=10_000_000,
        maximum_target_decode_bridge_peak_logical_tensor_bytes=10_000_000,
        maximum_sample_materialization_logical_tensor_bytes=10_000_000,
        maximum_sample_launch_tensor_bytes=10_000_000,
        maximum_request_geometry_bar_tensor_bytes=1,
        maximum_geometry_bridge_visible_peak_logical_tensor_bytes=1,
    )
    return PaperKineticFixedSiteMaterialOnlyStepPolicy(
        observation_memory_policy=observation,
        request_memory_policy=request,
        maximum_world_site_count=100,
        maximum_material_state_logical_tensor_bytes=10_000_000,
        maximum_material_checkpoint_logical_tensor_bytes=10_000_000,
        maximum_step_accumulator_logical_tensor_bytes=10_000_000,
        maximum_tracks_per_request=maximum_tracks_per_request,
        maximum_artifact_accounted_bytes=10_000_000,
        maximum_samples_per_launch=2,
        cone_tolerance=1.0e-5,
    )


def _generation(
    *,
    step_index: int = 0,
    material_generation_id: str = "fixed-site-material-generation-0",
) -> PaperKineticFixedSiteMaterialOnlyGenerationPolicy:
    return PaperKineticFixedSiteMaterialOnlyGenerationPolicy(
        step_index=step_index,
        material_generation_id=material_generation_id,
        background_generation_id="fixed-site-background-generation-0",
        target_generation_id="fixed-site-target-generation-0",
    )


def _case(
    *,
    maximum_store_entries: int = 2,
    maximum_store_bytes: int = 20_000_000,
):
    target_source, factory, provider = _provider()
    store = PaperKineticCompiledCpuArtifactStore(
        PaperKineticCompiledCpuArtifactStorePolicy(
            maximum_entries=maximum_store_entries,
            maximum_resident_accounted_bytes=maximum_store_bytes,
        )
    )
    state = prepare_paper_kinetic_fixed_site_material_step_state(
        provider,
        store,
        device="cpu",
    )
    material = torch.tensor(
        ((0.21, 0.37, 0.16, 0.8),),
        dtype=torch.float32,
    )
    background = torch.tensor((0.03, 0.05, 0.07), dtype=torch.float32)
    return target_source, factory, provider, store, state, material, background


def _material_state_for_provider(tmp_path: Path, provider):
    asset = tmp_path / "fixed_site_coordinator_world.ply"
    asset.write_text(
        "\n".join(
            (
                "ply",
                "format ascii 1.0",
                "element vertex 1",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header",
                "0.0 0.0 0.0 64 96 128",
                "",
            )
        ),
        encoding="utf-8",
    )
    initializer = prepare_paper_kinetic_point_cloud_world_initializer(
        {
            "source_path": asset,
            "source_coordinate_frame": "model",
            "point_transform": None,
            "maximum_source_asset_bytes": 1_000_000,
            "maximum_source_point_count": 1,
            "site_count": 1,
            "sample_mode": "first",
            "sample_seed": 0,
            "coordinate_quantization_step": 0.25,
            "weight_coefficients": (0.0,),
            "weight_quantization_step": 0.0625,
            "initial_density": 0.8,
        }
    )
    initialization = initializer.initialize_p0_material(provider.world.sites)
    return prepare_paper_kinetic_fixed_site_material_state(
        initialization,
        provider.world,
        parameterization=PaperKineticFixedSiteMaterialParameterization(),
        optimizer_policy=PaperKineticFixedSiteMaterialSGDPolicy(
            color_learning_rate=0.01,
            density_learning_rate=0.01,
        ),
        device="cpu",
        maximum_material_state_logical_tensor_bytes=1_000_000,
    )


def test_material_step_authorizes_exact_multiview_manifest_without_parameter_update(
    monkeypatch,
) -> None:
    target_source, factory, provider, store, state, material, background = _case()
    material_before = material.clone()
    background_before = background.clone()
    fence = _Fence()
    capability_roots = {}
    original_prepare = step_module.prepare_paper_kinetic_dense_step_gradient_accumulator

    def capture_capability_roots(source, session, **kwargs):
        capability_roots["source"] = weakref.ref(source)
        capability_roots["session"] = weakref.ref(session)
        return original_prepare(source, session, **kwargs)

    monkeypatch.setattr(
        step_module,
        "prepare_paper_kinetic_dense_step_gradient_accumulator",
        capture_capability_roots,
    )

    result = run_paper_kinetic_fixed_site_material_only_step(
        state,
        provider,
        _batch(),
        policy=_policy(),
        generation_policy=_generation(),
        global_site_rgba_f32=material,
        background_rgb_f32=background,
        native_ops=_FakeNativeOps(),
        backend_provenance="cpu-fake-native/exact-op-surface",
        device_completion_fence=fence,
        device_completion_fence_provenance="cpu-synchronous-fence-v1",
    )

    result.assert_current()
    result.authorization.assert_current(
        result.accumulator,
        result.replay_receipt,
    )
    torch.testing.assert_close(material, material_before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(background, background_before, rtol=0.0, atol=0.0)
    assert result.authorization.full_geometry is False
    assert result.accumulator.full_geometry is False
    assert result.authorization.grad_positions0_f64 is None
    assert result.authorization.grad_velocities_f64 is None
    assert result.authorization.grad_weight_coefficients_f64 is None
    assert result.authorization.grad_track_ray_coefficients_f64 is None
    assert result.authorization.request_count == 6
    assert result.authorization.observation_count == 12
    assert result.accounting["exact_request_count"] == 6
    assert result.accounting["exact_observation_count"] == 12
    assert result.accounting["image_height"] == 2
    assert result.accounting["image_width"] == provider.width == 3
    assert result.accounting["sample_node_interaction_count"] >= 12
    assert result.accounting["transferred_target_payload_bytes"] == 12 * 12
    assert result.accounting["peak_sample_launch_node_count"] == max(
        result.accounting["chart_node_ranks"]
    )
    assert result.accounting["active_native_block_count"] == (
        result.accounting["native_material_vjp_launch_count"]
    )
    assert result.accounting["node_forward_launch_count"] == (
        result.accounting["active_native_block_count"]
    )
    assert result.accounting["node_forward_thread_count"] > 0
    assert result.accounting["node_forward_interaction_count"] >= result.accounting[
        "node_forward_thread_count"
    ]
    assert result.accounting["node_forward_interaction_count"] == result.accounting[
        "material_word_vjp_interaction_count"
    ]
    assert result.accounting["active_material_exact_model_bytes"] > 0
    assert len(result.accounting["structural_signature_sha256"]) == 64
    assert result.accounting["provider_world_generation_digest"] == (
        provider.world.generation_digest
    )
    assert result.accounting["world_generation_digest"] == (
        result.accounting["world_sites_content_digest"]
    )
    assert result.accounting["full_dense_observation_replay"]
    assert result.accounting["sample_and_target_payloads_streamed"]
    assert result.accounting[
        "structural_node_word_work_invariance_requires_cross_row_verification"
    ]
    assert not result.accounting["full_video_target_tensor_retained"]
    assert result.accounting["global_loss_element_count"] == 36
    assert result.accounting["loss_normalization_id"] == (
        GLOBAL_RGB_MEAN_LOSS_NORMALIZATION_ID
    )
    assert result.accounting["global_rgb_mean_application_count"] == 1
    assert result.accounting[
        "accumulator_initialization_fence_call_count"
    ] == 1
    assert fence.calls == result.accounting[
        "total_step_completion_fence_call_count"
    ]
    assert result.accounting["built_in_bounded_target_decoder"]
    assert not result.accounting["arbitrary_external_target_loader"]
    assert result.accounting["artifact_store_cold_compile_count"] == 6
    assert result.accounting["artifact_store_cold_compiled_track_count"] == 12
    assert result.accounting[
        "artifact_store_step_metrics_derived_from_acquisition_receipts"
    ]
    assert result.accounting["artifact_store_peak_resident_accounted_bytes"] > 0
    assert result.accounting["peak_cpu_decoded_frame_tensor_bytes"] > 0
    assert result.accounting["peak_cpu_chunk_target_tensor_bytes"] > 0
    assert result.accounting[
        "peak_sample_materialization_logical_tensor_bytes_upper_bound"
    ] > 0
    assert result.accounting[
        "peak_interpolation_evaluator_scratch_logical_tensor_bytes_upper_bound"
    ] > 0
    assert result.accounting["maximum_interpolation_rows_per_subchunk"] > 0
    assert result.accounting[
        "sample_materialization_source_visible_logical_tensors_accounted"
    ]
    assert not result.accounting["sample_materialization_float64_scratch_measured"]
    assert result.accounting["peak_native_prepared_sample_scratch_tensor_bytes"] > 0
    assert result.accounting["maximum_simultaneously_decoded_target_frame_count"] == 1
    assert not result.accounting["step_accumulator_retains_frame_axis"]
    assert result.accounting["reachable_autograd_tensor_count"] == 0
    assert not result.accounting["autograd_graph_retained"]
    assert not result.accounting["autograd_saved_tensor_peak_measured"]
    for key in (
        "persistent_frame_tensor_bytes",
        "persistent_sample_tensor_bytes",
        "persistent_target_tensor_bytes",
        "persistent_prediction_tensor_bytes",
    ):
        assert result.accounting[key] == 0
    assert not result.accounting["artifact_working_set_fits_entry_bound"]
    assert result.accounting["parameter_mutation_count"] == 0
    assert factory.compile_count == 12
    assert len(target_source.calls) == 6
    assert store.report().current_entry_count == 2
    assert state.authorized_step_count == 1
    assert not state.poisoned
    assert not state.restart_required
    assert state.accounting()["successful_step_local_object_count_retained"] == 0
    assert "source" not in result.__dict__
    assert "session" not in result.__dict__
    assert "artifact" not in result.__dict__
    gc.collect()
    assert capability_roots["source"]() is None
    assert capability_roots["session"]() is None


def test_real_sealed_result_applies_to_fixed_site_material_state(
    tmp_path: Path,
) -> None:
    (
        _target_source,
        _factory,
        provider,
        store,
        coordinator_state,
        _material,
        background,
    ) = _case(
        maximum_store_entries=6,
        maximum_store_bytes=60_000_000,
    )
    material_state = _material_state_for_provider(tmp_path, provider)
    material_before = material_state.site_rgba_f32.clone()
    material_version_before = int(material_state.site_rgba_f32._version)
    material_generation_before = material_state.material_generation_id

    first_result = run_paper_kinetic_fixed_site_material_only_step(
        coordinator_state,
        provider,
        _batch(),
        policy=_policy(),
        generation_policy=_generation(
            material_generation_id=material_state.material_generation_id,
        ),
        global_site_rgba_f32=material_state.site_rgba_f32,
        background_rgb_f32=background,
        native_ops=_FakeNativeOps(),
        backend_provenance="cpu-fake-native/exact-op-surface",
        device_completion_fence=_Fence(),
        device_completion_fence_provenance="cpu-synchronous-fence-v1",
    )

    first_result.assert_current()
    torch.testing.assert_close(
        material_state.site_rgba_f32,
        material_before,
        rtol=0.0,
        atol=0.0,
    )
    assert int(material_state.site_rgba_f32._version) == material_version_before
    receipt = apply_paper_kinetic_fixed_site_material_sgd_step(
        material_state,
        first_result.authorization,
        first_result.accumulator,
        first_result.replay_receipt,
    )

    receipt.assert_current(material_state)
    assert receipt.step_generation_id == (
        first_result.authorization.step_generation_id
    )
    assert receipt.authorization_generation_digest == (
        first_result.authorization.generation_digest
    )
    assert receipt.material_generation_id_before == material_generation_before
    assert receipt.material_generation_id_after == (
        material_state.material_generation_id
    )
    assert material_state.step_index == 1
    assert material_state.material_generation_id != material_generation_before

    checkpoint = checkpoint_paper_kinetic_fixed_site_material_state(material_state)
    parsed_checkpoint = paper_kinetic_fixed_site_material_checkpoint_from_payload(
        checkpoint.payload(),
        expected_world_site_count=provider.world.site_count,
        maximum_checkpoint_logical_tensor_bytes=1_000_000,
    )
    restored_state = restore_paper_kinetic_fixed_site_material_state(
        parsed_checkpoint,
        world=provider.world,
        device="cpu",
        maximum_material_state_logical_tensor_bytes=1_000_000,
    )
    assert restored_state.step_index == 1
    assert restored_state.material_generation_id == material_state.material_generation_id
    assert restored_state.restart_checkpoint_generation_digest == (
        parsed_checkpoint.generation_digest
    )
    assert restored_state.last_authorization_generation_digest == (
        receipt.authorization_generation_digest
    )
    assert restored_state.last_step_generation_id == receipt.step_generation_id
    with pytest.raises(ValueError, match="authorization changed or is foreign"):
        apply_paper_kinetic_fixed_site_material_sgd_step(
            restored_state,
            first_result.authorization,
            first_result.accumulator,
            first_result.replay_receipt,
        )
    restored_state.assert_current()
    del first_result

    # A real restore cannot inherit the old coordinator's in-memory counter.
    # Bootstrap a fresh coordinator from the sealed material history while
    # retaining the caller-owned structural artifact store.
    restored_coordinator_state = (
        prepare_paper_kinetic_fixed_site_material_step_state(
            provider,
            store,
            device="cpu",
            resume_material_state=restored_state,
        )
    )
    assert restored_coordinator_state.authorized_step_count == 1
    assert restored_coordinator_state.last_step_generation_id == (
        restored_state.last_step_generation_id
    )
    assert restored_coordinator_state.last_authorized_material_generation_id == (
        restored_state.generation_parent_digest
    )
    second_material_generation_before = restored_state.material_generation_id
    second_result = run_paper_kinetic_fixed_site_material_only_step(
        restored_coordinator_state,
        provider,
        _batch(),
        policy=_policy(),
        generation_policy=_generation(
            step_index=1,
            material_generation_id=restored_state.material_generation_id,
        ),
        global_site_rgba_f32=restored_state.site_rgba_f32,
        background_rgb_f32=background,
        native_ops=_FakeNativeOps(),
        backend_provenance="cpu-fake-native/exact-op-surface",
        device_completion_fence=_Fence(),
        device_completion_fence_provenance="cpu-synchronous-fence-v1",
    )
    assert second_result.accounting["artifact_store_cold_compile_count"] == 0
    assert second_result.accounting["artifact_store_warm_hit_count"] == 6
    assert second_result.accounting["artifact_working_set_fits_entry_bound"]
    assert _factory.compile_count == 12
    second_receipt = apply_paper_kinetic_fixed_site_material_sgd_step(
        restored_state,
        second_result.authorization,
        second_result.accumulator,
        second_result.replay_receipt,
    )
    second_receipt.assert_current(restored_state)
    assert second_receipt.step_index == 2
    assert second_receipt.material_generation_id_before == (
        second_material_generation_before
    )
    assert second_receipt.material_generation_id_after == (
        restored_state.material_generation_id
    )
    assert restored_state.step_index == 2
    assert restored_state.generation_parent_digest == (
        second_material_generation_before
    )
    assert restored_state.last_authorization_generation_digest == (
        second_receipt.authorization_generation_digest
    )
    assert restored_state.last_step_generation_id == (
        second_receipt.step_generation_id
    )
    assert restored_coordinator_state.authorized_step_count == 2


def test_device_bridge_seals_only_the_exact_coordinator_result(
    tmp_path: Path,
) -> None:
    (
        _target_source,
        _factory,
        provider,
        _store,
        coordinator_state,
        _unused_material,
        background,
    ) = _case(maximum_store_entries=6, maximum_store_bytes=60_000_000)
    material_state = _material_state_for_provider(tmp_path, provider)
    snapshot = snapshot_paper_kinetic_fixed_site_material_to_device(
        material_state,
        background_rgb_f32_cpu=background,
        background_generation_id="fixed-site-background-generation-0",
        device="cpu",
        device_completion_fence=_Fence(),
        device_completion_fence_provenance="cpu-synchronous-fence-v1",
    )
    coordinator_result = run_paper_kinetic_fixed_site_material_only_step(
        coordinator_state,
        provider,
        _batch(),
        policy=_policy(),
        generation_policy=_generation(
            material_generation_id=material_state.material_generation_id,
        ),
        global_site_rgba_f32=snapshot.site_rgba_f32_device,
        background_rgb_f32=snapshot.background_rgb_f32_device,
        native_ops=_FakeNativeOps(),
        backend_provenance="cpu-fake-native/exact-op-surface",
        device_completion_fence=_Fence(),
        device_completion_fence_provenance="cpu-synchronous-fence-v1",
    )
    assert (
        coordinator_result.accumulator._material_tensor_ref
        is snapshot.site_rgba_f32_device
    )
    receipt = seal_paper_kinetic_fixed_site_material_device_gradient_receipt(
        material_state,
        snapshot,
        coordinator_result,
        device_completion_fence=_Fence(),
        device_completion_fence_provenance="cpu-synchronous-fence-v1",
    )

    assert receipt.production_step_result_bound
    assert receipt.step_result_identity == id(coordinator_result)
    assert receipt.authorization_identity == id(coordinator_result.authorization)
    assert receipt.accumulator_identity == id(coordinator_result.accumulator)
    assert receipt.replay_receipt_identity == id(coordinator_result.replay_receipt)
    assert receipt.step_result_generation_digest == (
        coordinator_result.generation_digest
    )
    assert receipt.authorization_generation_digest == (
        coordinator_result.authorization.generation_digest
    )
    assert receipt.accumulator_generation_digest == (
        coordinator_result.accumulator.generation_digest
    )
    assert receipt.replay_receipt_generation_digest == (
        coordinator_result.replay_receipt.generation_digest
    )
    assert "_step_result" not in receipt.__dict__
    step_receipt = apply_paper_kinetic_fixed_site_material_device_gradient_receipt(
        material_state,
        receipt,
    )
    assert step_receipt.authorization_generation_digest == (
        receipt.optimizer_commit_generation_digest
    )
    assert material_state.last_authorization_generation_digest == (
        receipt.optimizer_commit_generation_digest
    )


def test_device_bridge_rejects_result_for_another_material_tensor(
    tmp_path: Path,
) -> None:
    (
        _target_source,
        _factory,
        provider,
        _store,
        coordinator_state,
        _unused_material,
        background,
    ) = _case(maximum_store_entries=6, maximum_store_bytes=60_000_000)
    material_state = _material_state_for_provider(tmp_path, provider)
    snapshot = snapshot_paper_kinetic_fixed_site_material_to_device(
        material_state,
        background_rgb_f32_cpu=background,
        background_generation_id="fixed-site-background-generation-0",
        device="cpu",
        device_completion_fence=_Fence(),
        device_completion_fence_provenance="cpu-synchronous-fence-v1",
    )
    foreign_material = snapshot.site_rgba_f32_device.clone()
    coordinator_result = run_paper_kinetic_fixed_site_material_only_step(
        coordinator_state,
        provider,
        _batch(),
        policy=_policy(),
        generation_policy=_generation(
            material_generation_id=material_state.material_generation_id,
        ),
        global_site_rgba_f32=foreign_material,
        background_rgb_f32=snapshot.background_rgb_f32_device,
        native_ops=_FakeNativeOps(),
        backend_provenance="cpu-fake-native/exact-op-surface",
        device_completion_fence=_Fence(),
        device_completion_fence_provenance="cpu-synchronous-fence-v1",
    )

    with pytest.raises(ValueError, match="exact device snapshot"):
        seal_paper_kinetic_fixed_site_material_device_gradient_receipt(
            material_state,
            snapshot,
            coordinator_result,
            device_completion_fence=_Fence(),
            device_completion_fence_provenance="cpu-synchronous-fence-v1",
        )
    assert not snapshot.gradient_receipt_issued


def test_partial_second_request_failure_poison_retains_only_safety_roots(
    monkeypatch,
) -> None:
    target_source, _factory, provider, _store, state, material, background = _case()
    material_before = material.clone()
    original_run = step_module.run_paper_kinetic_dense_cached_native_request
    request_call_count = 0

    def fail_second_request(*args, **kwargs):
        nonlocal request_call_count
        request_call_count += 1
        if request_call_count == 2:
            raise RuntimeError("injected second-request failure")
        return original_run(*args, **kwargs)

    monkeypatch.setattr(
        step_module,
        "run_paper_kinetic_dense_cached_native_request",
        fail_second_request,
    )
    with pytest.raises(
        PaperKineticFixedSiteMaterialStepPartialFailure,
        match="process restart is required",
    ) as failure:
        run_paper_kinetic_fixed_site_material_only_step(
            state,
            provider,
            _batch(),
            policy=_policy(),
            generation_policy=_generation(),
            global_site_rgba_f32=material,
            background_rgb_f32=background,
            native_ops=_FakeNativeOps(),
            backend_provenance="cpu-fake-native/exact-op-surface",
            device_completion_fence=_Fence(),
            device_completion_fence_provenance="cpu-synchronous-fence-v1",
        )

    assert failure.value.state is state
    assert request_call_count == 2
    assert state.poisoned
    assert state.restart_required
    assert state.failure_fail_stop_completed
    assert state.failed_accumulator is not None
    assert state.failed_accumulator.poisoned
    assert not state.failed_accumulator.optimizer_authorized
    assert not bool(torch.any(state.failed_accumulator.grad_site_rgba_f32).item())
    assert not bool(torch.any(state.failed_accumulator.loss_f32).item())
    assert state.failed_replay_session is not None
    assert state.failed_replay_session.poisoned
    assert state.accounting()["failed_source_retained"]
    assert state.accounting()["failed_session_retained"]
    assert state.accounting()["failed_accumulator_retained"]
    assert state.accounting()["failed_request_lifetime_root_count"] == 2
    assert state.failure_lifetime_root_roles == ("request", "artifact")
    state.assert_current(provider)
    torch.testing.assert_close(material, material_before, rtol=0.0, atol=0.0)
    assert len(target_source.calls) == 1
    with pytest.raises(RuntimeError, match="process restart"):
        run_paper_kinetic_fixed_site_material_only_step(
            state,
            provider,
            _batch(),
            policy=_policy(),
            generation_policy=_generation(),
            global_site_rgba_f32=material,
            background_rgb_f32=background,
            native_ops=_FakeNativeOps(),
            backend_provenance="cpu-fake-native/exact-op-surface",
            device_completion_fence=_Fence(),
            device_completion_fence_provenance="cpu-synchronous-fence-v1",
        )


def test_failed_fail_stop_still_retains_restart_only_lifetime_roots(
    monkeypatch,
) -> None:
    _target_source, _factory, provider, _store, state, material, background = _case()
    original_run = step_module.run_paper_kinetic_dense_cached_native_request
    request_call_count = 0

    def fail_second_request(*args, **kwargs):
        nonlocal request_call_count
        request_call_count += 1
        if request_call_count == 2:
            raise RuntimeError("injected request failure before fail-stop")
        return original_run(*args, **kwargs)

    def fail_fail_stop(*_args, **_kwargs) -> None:
        raise RuntimeError("injected whole-step fail-stop failure")

    monkeypatch.setattr(
        step_module,
        "run_paper_kinetic_dense_cached_native_request",
        fail_second_request,
    )
    monkeypatch.setattr(
        step_module,
        "fail_stop_paper_kinetic_dense_step",
        fail_fail_stop,
    )
    with pytest.raises(PaperKineticFixedSiteMaterialStepPartialFailure):
        run_paper_kinetic_fixed_site_material_only_step(
            state,
            provider,
            _batch(),
            policy=_policy(),
            generation_policy=_generation(),
            global_site_rgba_f32=material,
            background_rgb_f32=background,
            native_ops=_FakeNativeOps(),
            backend_provenance="cpu-fake-native/exact-op-surface",
            device_completion_fence=_Fence(),
            device_completion_fence_provenance="cpu-synchronous-fence-v1",
        )

    assert state.poisoned
    assert state.restart_required
    assert not state.failure_fail_stop_completed
    assert state.failed_accumulator is not None
    assert not state.failed_accumulator.poisoned
    assert state.failure_lifetime_root_roles == ("request", "artifact")
    state.assert_current(provider)


def test_accumulator_initialization_fence_failure_retains_without_tensor_mutation(
    monkeypatch,
) -> None:
    target_source, factory, provider, _store, state, material, background = _case()
    original_prepare = step_module.prepare_paper_kinetic_dense_step_gradient_accumulator
    captured_versions = ()

    def capture_accumulator(source, session, **kwargs):
        nonlocal captured_versions
        accumulator = original_prepare(source, session, **kwargs)
        captured_versions = tuple(
            int(tensor._version) for tensor in accumulator._tensors()
        )
        return accumulator

    def fail_initialization_fence() -> None:
        raise RuntimeError("injected accumulator initialization fence failure")

    monkeypatch.setattr(
        step_module,
        "prepare_paper_kinetic_dense_step_gradient_accumulator",
        capture_accumulator,
    )
    with pytest.raises(
        PaperKineticFixedSiteMaterialStepPartialFailure,
        match="initialization fence failure",
    ):
        run_paper_kinetic_fixed_site_material_only_step(
            state,
            provider,
            _batch(),
            policy=_policy(),
            generation_policy=_generation(),
            global_site_rgba_f32=material,
            background_rgb_f32=background,
            native_ops=_FakeNativeOps(),
            backend_provenance="cpu-fake-native/exact-op-surface",
            device_completion_fence=fail_initialization_fence,
            device_completion_fence_provenance="injected-failing-fence-v1",
        )

    assert state.poisoned
    assert state.restart_required
    assert not state.failure_fail_stop_completed
    assert state.failed_accumulator is not None
    assert not state.failed_accumulator.poisoned
    assert tuple(
        int(tensor._version) for tensor in state.failed_accumulator._tensors()
    ) == captured_versions
    assert state.failure_lifetime_root_roles == ()
    assert factory.compile_count == 0
    assert target_source.calls == []
    state.assert_current(provider)


def test_policy_failure_precedes_artifact_compile_target_decode_and_poison() -> None:
    target_source, factory, provider, _store, state, material, background = _case()
    with pytest.raises(ValueError, match="source/compiler bound"):
        run_paper_kinetic_fixed_site_material_only_step(
            state,
            provider,
            _batch(),
            policy=_policy(maximum_tracks_per_request=3),
            generation_policy=_generation(),
            global_site_rgba_f32=material,
            background_rgb_f32=background,
            native_ops=_FakeNativeOps(),
            backend_provenance="cpu-fake-native/exact-op-surface",
            device_completion_fence=_Fence(),
            device_completion_fence_provenance="cpu-synchronous-fence-v1",
        )
    assert factory.compile_count == 0
    assert target_source.calls == []
    assert not state.poisoned
    assert not state.restart_required
    state.assert_current(provider)


def test_state_rejects_async_device_without_canonical_fence_contract() -> None:
    _target_source, _factory, provider = _provider()
    store = PaperKineticCompiledCpuArtifactStore(
        PaperKineticCompiledCpuArtifactStorePolicy(
            maximum_entries=2,
            maximum_resident_accounted_bytes=20_000_000,
        )
    )
    with pytest.raises(NotImplementedError, match="explicit fence contract"):
        prepare_paper_kinetic_fixed_site_material_step_state(
            provider,
            store,
            device="cuda",
        )
