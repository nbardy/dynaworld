from __future__ import annotations

from dataclasses import replace

import paper_kinetic_fixed_camera_combined_state as combined_module
import kinetic_dense_cached_native_material_request as dense_request_module
import pytest
import torch
from kinetic_compiled_cpu_artifact_store import (
    PaperKineticCompiledCpuArtifactStore,
    PaperKineticCompiledCpuArtifactStorePolicy,
    compile_paper_kinetic_compiled_cpu_artifact,
)
from kinetic_dense_cached_native_material_request import (
    FUSED_DIRECT_V1_FULL_GEOMETRY_REVERSE,
    STAGED_SPARSE_FULL_GEOMETRY_REVERSE,
)
from paper_kinetic_fixed_camera_combined_state import (
    PaperKineticFixedCameraCombinedSGDPolicy,
    PaperKineticFixedCameraCombinedTransactionFailure,
    apply_paper_kinetic_fixed_camera_combined_sgd_transaction,
    checkpoint_paper_kinetic_fixed_camera_combined_state,
    claim_paper_kinetic_fixed_camera_ready_generation_for_next_step,
    claim_paper_kinetic_fixed_camera_restored_ready_generation_for_next_step,
    paper_kinetic_fixed_camera_combined_checkpoint_from_payload,
    prepare_paper_kinetic_fixed_camera_cold_recompile_manifest,
    prepare_paper_kinetic_fixed_camera_selected_tracks_cold_recompile_manifest,
    prepare_paper_kinetic_fixed_camera_combined_state,
    restore_paper_kinetic_fixed_camera_combined_generation_from_payload,
)
from paper_kinetic_fixed_camera_full_geometry_step import (
    PaperKineticFixedCameraFullGeometryGenerationPolicy,
    prepare_paper_kinetic_fixed_camera_full_geometry_step_state,
    run_paper_kinetic_fixed_camera_full_geometry_step,
)
from test_kinetic_compiled_cpu_artifact_store import _provider
from test_kinetic_ragged_paper_step_cpu_fake_native import _FakeNativeOps
from test_kinetic_native_material_step_executor import (
    _FakeFusedPreparedBlock,
    _install_fused_session_cpu_fake,
)
from test_paper_kinetic_fixed_camera_full_geometry_step import (
    _Fence,
    _batch,
    _policy as _full_geometry_policy,
)
from test_paper_kinetic_fixed_site_material_step import (
    _material_state_for_provider,
)


def _transaction_policy() -> PaperKineticFixedCameraCombinedSGDPolicy:
    return PaperKineticFixedCameraCombinedSGDPolicy(
        position_learning_rate=1.0e-5,
        velocity_learning_rate=1.0e-5,
        weight_learning_rate=1.0e-5,
        maximum_absolute_position_update=1.0e6,
        maximum_absolute_velocity_update=1.0e6,
        maximum_absolute_weight_update=1.0e6,
        maximum_absolute_position_value=1.0e9,
        maximum_absolute_velocity_value=1.0e9,
        maximum_absolute_weight_value=1.0e9,
        maximum_combined_state_logical_tensor_bytes=1_000_000,
        maximum_update_candidate_logical_tensor_bytes=1_000_000,
        maximum_candidate_world_geometry_clone_logical_tensor_bytes=1_000_000,
        maximum_update_validation_scratch_logical_tensor_bytes=1_000_000,
        maximum_old_candidate_authorization_logical_tensor_bytes=2_000_000,
        maximum_checkpoint_logical_tensor_bytes=1_000_000,
        maximum_state_checkpoint_logical_tensor_bytes=2_000_000,
        maximum_state_checkpoint_payload_logical_tensor_bytes=3_000_000,
        maximum_transaction_tracked_logical_and_store_accounted_bytes=(
            128_000_000
        ),
        maximum_recompile_request_count=6,
        maximum_recompile_track_id_logical_bytes=1_000_000,
        maximum_artifact_accounted_bytes=10_000_000,
    )


def _install_fused_coordinator_cpu_fake(monkeypatch) -> None:
    _install_fused_session_cpu_fake(monkeypatch)

    def prepare_fake_fused(world, **_kwargs):
        prepared = _FakeFusedPreparedBlock(world)
        prepared.memory = type(
            "FakeFusedMemory",
            (),
            {"owned_persistent_tensor_bytes": 4},
        )()
        return prepared

    monkeypatch.setattr(
        dense_request_module,
        "prepare_kinetic_native_equal_rank_fused_direct_full_vjp_v1",
        prepare_fake_fused,
    )


def _authorized_case(
    tmp_path,
    *,
    full_geometry_reverse_mode: str = STAGED_SPARSE_FULL_GEOMETRY_REVERSE,
):
    target_source, factory, provider = _provider()
    store_policy = PaperKineticCompiledCpuArtifactStorePolicy(
        maximum_entries=6,
        maximum_resident_accounted_bytes=60_000_000,
    )
    store = PaperKineticCompiledCpuArtifactStore(store_policy)
    material_state = _material_state_for_provider(tmp_path, provider)
    combined_state = prepare_paper_kinetic_fixed_camera_combined_state(
        material_state,
        provider,
        store,
        maximum_combined_state_logical_tensor_bytes=1_000_000,
    )
    coordinator_state = prepare_paper_kinetic_fixed_camera_full_geometry_step_state(
        provider,
        store,
        device="cpu",
        resume_material_state=material_state,
    )
    generation_policy = PaperKineticFixedCameraFullGeometryGenerationPolicy(
        step_index=material_state.step_index,
        material_generation_id=material_state.material_generation_id,
        background_generation_id="combined-background-generation-0",
        target_generation_id="combined-target-generation-0",
        geometry_generation_id=combined_state.geometry_generation_id,
    )
    result = run_paper_kinetic_fixed_camera_full_geometry_step(
        coordinator_state,
        provider,
        _batch(),
        policy=_full_geometry_policy(
            full_geometry_reverse_mode=full_geometry_reverse_mode
        ),
        generation_policy=generation_policy,
        global_site_rgba_f32=material_state.site_rgba_f32,
        background_rgb_f32=torch.tensor(
            (0.03, 0.05, 0.07),
            dtype=torch.float32,
        ),
        native_ops=_FakeNativeOps(),
        backend_provenance="cpu-fake-native/exact-op-surface",
        device_completion_fence=_Fence(),
        device_completion_fence_provenance="cpu-synchronous-fence-v1",
    )
    manifest = prepare_paper_kinetic_fixed_camera_cold_recompile_manifest(
        provider,
        view_indices=(0, 1),
        maximum_tracks_per_request=2,
    )
    return {
        "target_source": target_source,
        "factory": factory,
        "provider": provider,
        "store": store,
        "store_policy": store_policy,
        "material_state": material_state,
        "combined_state": combined_state,
        "step_result": result,
        "manifest": manifest,
    }


def _expected_material_candidates(material_state, authorization):
    color_grad = torch.empty_like(material_state.raw_color_f32)
    density_grad = torch.empty_like(material_state.raw_density_f32)
    runtime = material_state.parameterization.runtime_parameterization
    runtime.color_vjp_(
        color_grad,
        material_state.site_rgba_f32[:, :3],
        authorization.grad_site_rgba_f32[:, :3],
    )
    runtime.density_vjp_(
        density_grad,
        material_state.raw_density_f32,
        authorization.grad_site_rgba_f32[:, 3],
    )
    return (
        material_state.raw_color_f32
        - material_state.optimizer_policy.color_learning_rate * color_grad,
        material_state.raw_density_f32
        - material_state.optimizer_policy.density_learning_rate * density_grad,
    )


def test_selected_track_manifest_never_expands_unselected_pixels() -> None:
    _source, _factory, provider = _provider()
    manifest = (
        prepare_paper_kinetic_fixed_camera_selected_tracks_cold_recompile_manifest(
            provider,
            selected_track_ids_by_view=((0, (0, 2, 3)), (1, (5,))),
            maximum_tracks_per_request=2,
            maximum_request_count=4,
            maximum_track_id_logical_bytes=128,
        )
    )

    assert tuple(
        (request.view_index, request.track_start, request.track_stop)
        for request in manifest.requests
    ) == ((0, 0, 1), (0, 2, 4), (1, 5, 6))
    assert manifest.track_count == 4
    assert manifest.request_count == 3
    assert sum(request.track_count for request in manifest.requests) == 4
    with pytest.raises(MemoryError, match="track-id policy"):
        prepare_paper_kinetic_fixed_camera_selected_tracks_cold_recompile_manifest(
            provider,
            selected_track_ids_by_view=((0, (0, 1, 2)),),
            maximum_tracks_per_request=2,
            maximum_request_count=4,
            maximum_track_id_logical_bytes=16,
        )


def test_combined_state_rejects_digest_consistent_foreign_geometry_generation(
    tmp_path,
) -> None:
    case = _authorized_case(tmp_path)
    provisional = replace(
        case["combined_state"],
        geometry_generation_id="0" * 64,
        generation_digest="",
    )
    forged = replace(
        provisional,
        generation_digest=combined_module._combined_state_digest(provisional),
    )

    with pytest.raises(ValueError, match="geometry generation"):
        forged.assert_current(case["provider"], case["store"])
    case["combined_state"].assert_current(case["provider"], case["store"])


def test_combined_transaction_updates_both_states_and_cold_reseals(tmp_path) -> None:
    case = _authorized_case(tmp_path)
    policy = _transaction_policy()
    old_provider = case["provider"]
    old_store = case["store"]
    old_state = case["combined_state"]
    step_result = case["step_result"]
    step_result_generation_digest = step_result.generation_digest
    authorization = step_result.authorization
    accumulator = step_result.accumulator
    authorization_logical_tensor_bytes = accumulator.logical_tensor_bytes
    expected_raw_color, expected_raw_density = _expected_material_candidates(
        case["material_state"],
        authorization,
    )
    expected_geometry = tuple(
        old - learning_rate * bar
        for old, bar, learning_rate in zip(
            old_state._geometry_tensors(),
            (
                authorization.grad_positions0_f64,
                authorization.grad_velocities_f64,
                authorization.grad_weight_coefficients_f64,
            ),
            (
                policy.position_learning_rate,
                policy.velocity_learning_rate,
                policy.weight_learning_rate,
            ),
            strict=True,
        )
    )
    stale_artifact = old_store.acquire(
        old_provider,
        view_index=0,
        track_ids=(0, 1),
        maximum_artifact_accounted_bytes=policy.maximum_artifact_accounted_bytes,
        compile_artifact=lambda key: (
            compile_paper_kinetic_compiled_cpu_artifact(old_provider, key)
        ),
    ).artifact

    ready = apply_paper_kinetic_fixed_camera_combined_sgd_transaction(
        old_state,
        old_provider,
        old_store,
        step_result,
        policy=policy,
        cold_recompile_manifest=case["manifest"],
        fresh_store_policy=case["store_policy"],
    )

    ready.assert_current()
    torch.testing.assert_close(
        ready.state.material_state.raw_color_f32,
        expected_raw_color,
    )
    torch.testing.assert_close(
        ready.state.material_state.raw_density_f32,
        expected_raw_density,
    )
    for current, expected in zip(
        ready.state._geometry_tensors(),
        expected_geometry,
        strict=True,
    ):
        torch.testing.assert_close(current, expected, rtol=0.0, atol=1.0e-12)
    assert ready.state.geometry_update_count == 1
    assert ready.state.material_state.step_index == 1
    assert ready.update_receipt.ray_updates_enabled is False
    assert ready.state.camera_ray_parameter_tensor_bytes == 0
    assert ready.update_receipt.compiled_tensor_bytes_retained == 0
    assert ready.update_receipt.cold_compile_scratch_peak_measured is False
    assert (
        ready.update_receipt
        .transaction_tracked_logical_and_store_accounted_bytes_upper_bound
        == ready.update_receipt.old_candidate_authorization_logical_tensor_bytes
        + ready.update_receipt.candidate_world_geometry_clone_logical_tensor_bytes
        + ready.update_receipt
        .update_validation_scratch_logical_tensor_bytes_upper_bound
        + ready.update_receipt.old_store_resident_accounted_bytes_before_retirement
        + ready.update_receipt.fresh_store_resident_accounted_bytes_upper_bound
    )
    assert (
        ready.update_receipt
        .transaction_tracked_logical_and_store_accounted_bytes_upper_bound
        <= policy.maximum_transaction_tracked_logical_and_store_accounted_bytes
    )
    assert ready.update_receipt.transaction_accounting_scope == (
        "transaction-owned-state-candidate-authorization-geometry-"
        "clone-validation-scratch-plus-"
        "store-owned-accounted-entries"
    )
    assert ready.update_receipt.authorization_logical_tensor_bytes == (
        authorization_logical_tensor_bytes
    )
    assert ready.update_receipt.released_authorization_logical_tensor_bytes == (
        authorization_logical_tensor_bytes
    )
    assert ready.update_receipt.authorization_capability_revoked
    assert ready.update_receipt.authorization_accumulator_revoked
    assert ready.update_receipt.authorization_tensor_references_released
    assert ready.update_receipt.full_geometry_step_result_revoked
    assert (
        ready.update_receipt.full_geometry_step_result_generation_digest
        == step_result_generation_digest
    )
    assert (
        ready.update_receipt.full_geometry_reverse_mode
        == STAGED_SPARSE_FULL_GEOMETRY_REVERSE
    )
    assert not ready.update_receipt.caller_retained_untracked_bytes_included
    assert ready.recompile_receipt.compiled_tensors_retained_by_receipt == 0

    with pytest.raises(ValueError):
        step_result.assert_current()
    for tensor_name in (
        "grad_site_rgba_f32",
        "loss_f32",
        "grad_positions0_f64",
        "grad_velocities_f64",
        "grad_weight_coefficients_f64",
        "grad_track_ray_coefficients_f64",
    ):
        assert getattr(authorization, tensor_name) is None
        assert getattr(accumulator, tensor_name) is None
    assert accumulator._material_tensor_ref is None
    assert accumulator._background_tensor_ref is None

    fresh_report = ready.artifact_store.report()
    assert case["manifest"].track_id_logical_bytes == (
        case["manifest"].request_count * 3 * 8
    )
    assert all(
        "track_ids" not in request.__dict__
        for request in case["manifest"].requests
    )
    assert fresh_report.current_entry_count == case["manifest"].request_count
    assert fresh_report.cold_compile_count == case["manifest"].request_count
    assert fresh_report.cold_compiled_track_count == case["manifest"].track_count
    assert fresh_report.hit_count == 0
    assert fresh_report.eviction_count == 0
    assert ready.recompile_receipt.eviction_count == 0
    assert ready.recompile_receipt.full_manifest_digest_chain_bound
    assert ready.recompile_receipt.final_store_is_bounded_lru_working_set
    assert old_store.report().current_entry_count == 0
    old_state.assert_retired()
    provisional_forged_retired = replace(
        old_state,
        geometry_generation_id="0" * 64,
        generation_digest="",
    )
    forged_retired = replace(
        provisional_forged_retired,
        generation_digest=combined_module._combined_state_digest(
            provisional_forged_retired
        ),
    )
    with pytest.raises(ValueError, match="retired combined state"):
        forged_retired.assert_retired()
    with pytest.raises(ValueError):
        old_provider.assert_current()
    with pytest.raises(ValueError):
        stale_artifact.assert_warm_reusable_with_provider(ready.provider)
    with pytest.raises(RuntimeError, match="closed"):
        old_store.acquire(
            ready.provider,
            view_index=0,
            track_ids=(0, 1),
            maximum_artifact_accounted_bytes=policy.maximum_artifact_accounted_bytes,
            compile_artifact=lambda key: (
                compile_paper_kinetic_compiled_cpu_artifact(ready.provider, key)
            ),
        )

    assert not hasattr(ready, "checkpoint")
    checkpoint = checkpoint_paper_kinetic_fixed_camera_combined_state(
        ready.state,
        ready.provider,
        ready.artifact_store,
        manifest=ready.manifest,
        recompile_receipt=ready.recompile_receipt,
        policy=policy,
        initializer_generation_digest=ready.provider.initializer_generation_digest,
    )
    checkpoint.assert_current()
    provisional_forged_checkpoint = replace(
        checkpoint,
        geometry_generation_id="0" * 64,
        generation_digest="",
    )
    forged_checkpoint = replace(
        provisional_forged_checkpoint,
        generation_digest=combined_module._combined_checkpoint_digest(
            provisional_forged_checkpoint
        ),
    )
    with pytest.raises(ValueError, match="geometry generation"):
        forged_checkpoint.assert_current()
    assert checkpoint.combined_sgd_policy == policy
    assert checkpoint.last_update_policy_generation_digest == policy.generation_digest
    assert checkpoint.checkpoint_tensor_bytes <= (
        policy.maximum_checkpoint_logical_tensor_bytes
    )
    assert checkpoint.state_checkpoint_logical_tensor_bytes == (
        ready.state.total_persistent_tensor_bytes
        + checkpoint.checkpoint_tensor_bytes
    )
    assert checkpoint.state_checkpoint_payload_peak_logical_tensor_bytes == (
        ready.state.total_persistent_tensor_bytes
        + 2 * checkpoint.checkpoint_tensor_bytes
    )
    assert checkpoint.compiled_tensor_bytes == 0
    assert checkpoint.camera_ray_parameter_tensor_bytes == 0
    assert not checkpoint.combined_checkpoint_restore_integrated
    assert not checkpoint.production_trainer_integrated
    for saved, live in zip(
        (
            checkpoint.positions0_f64_cpu,
            checkpoint.velocities_f64_cpu,
            checkpoint.weight_coefficients_f64_cpu,
        ),
        ready.state._geometry_tensors(),
        strict=True,
    ):
        assert saved.untyped_storage().data_ptr() != live.untyped_storage().data_ptr()
    payload = checkpoint.payload()
    assert payload["compiled_tensor_bytes"] == 0
    assert payload["camera_ray_parameter_tensor_bytes"] == 0
    assert {
        key for key, value in payload.items() if isinstance(value, torch.Tensor)
    } == {
        "positions0_f64_cpu",
        "velocities_f64_cpu",
        "weight_coefficients_f64_cpu",
    }
    parsed_checkpoint = paper_kinetic_fixed_camera_combined_checkpoint_from_payload(
        payload,
        expected_world_site_count=ready.state.site_count,
        expected_combined_sgd_policy=policy,
    )
    parsed_checkpoint.assert_current()
    assert parsed_checkpoint.generation_digest == checkpoint.generation_digest
    assert parsed_checkpoint.cold_recompile_manifest == checkpoint.cold_recompile_manifest
    assert parsed_checkpoint.combined_sgd_policy == policy
    for parsed, serialized in zip(
        (
            parsed_checkpoint.positions0_f64_cpu,
            parsed_checkpoint.velocities_f64_cpu,
            parsed_checkpoint.weight_coefficients_f64_cpu,
        ),
        (
            payload["positions0_f64_cpu"],
            payload["velocities_f64_cpu"],
            payload["weight_coefficients_f64_cpu"],
        ),
        strict=True,
    ):
        torch.testing.assert_close(parsed, serialized)
        assert parsed.untyped_storage().data_ptr() != serialized.untyped_storage().data_ptr()
    lying_accounting_payload = checkpoint.payload()
    lying_accounting_payload[
        "live_state_logical_tensor_bytes_at_checkpoint"
    ] += 1
    with pytest.raises(ValueError, match="serialized byte accounting"):
        paper_kinetic_fixed_camera_combined_checkpoint_from_payload(
            lying_accounting_payload,
            expected_world_site_count=ready.state.site_count,
            expected_combined_sgd_policy=policy,
        )
    oversized_storage_payload = checkpoint.payload()
    backing_rows = policy.maximum_checkpoint_logical_tensor_bytes // 24 + 2
    oversized_positions_backing = torch.empty(
        (backing_rows, 3),
        dtype=torch.float64,
    )
    oversized_positions = oversized_positions_backing[: ready.state.site_count]
    oversized_positions.copy_(oversized_storage_payload["positions0_f64_cpu"])
    assert oversized_positions.is_contiguous()
    oversized_storage_payload["positions0_f64_cpu"] = oversized_positions
    with pytest.raises(MemoryError, match="source storage"):
        paper_kinetic_fixed_camera_combined_checkpoint_from_payload(
            oversized_storage_payload,
            expected_world_site_count=ready.state.site_count,
            expected_combined_sgd_policy=policy,
        )
    oversized_manifest_payload = checkpoint.payload()
    raw_manifest = dict(oversized_manifest_payload["cold_recompile_manifest"])
    oversized_track_count = (
        policy.maximum_recompile_track_id_logical_bytes // 8 + 1
    )
    raw_manifest.update(
        {
            "height": 1,
            "width": oversized_track_count,
            "maximum_tracks_per_bundle": oversized_track_count,
            "requests": ((0, 0, oversized_track_count),),
            "request_count": 1,
            "track_count": oversized_track_count,
            "persistent_partition_logical_bytes": 24,
        }
    )
    oversized_manifest_payload["cold_recompile_manifest"] = raw_manifest
    with pytest.raises(MemoryError, match="request-local track-id bound"):
        paper_kinetic_fixed_camera_combined_checkpoint_from_payload(
            oversized_manifest_payload,
            expected_world_site_count=ready.state.site_count,
            expected_combined_sgd_policy=policy,
        )
    changed_geometry_payload = checkpoint.payload()
    changed_geometry_payload["positions0_f64_cpu"][0, 0].add_(1.0)
    with pytest.raises(ValueError, match="combined fixed-camera checkpoint changed"):
        paper_kinetic_fixed_camera_combined_checkpoint_from_payload(
            changed_geometry_payload,
            expected_world_site_count=ready.state.site_count,
            expected_combined_sgd_policy=policy,
        )
    stricter_restart_policy = replace(
        policy,
        maximum_checkpoint_logical_tensor_bytes=(
            checkpoint.checkpoint_tensor_bytes - 1
        ),
    )
    with pytest.raises(ValueError, match="differs from the restart policy"):
        paper_kinetic_fixed_camera_combined_checkpoint_from_payload(
            checkpoint.payload(),
            expected_world_site_count=ready.state.site_count,
            expected_combined_sgd_policy=stricter_restart_policy,
        )
    with pytest.raises(MemoryError, match="artifact/checkpoint/retired-generation"):
        claim_paper_kinetic_fixed_camera_ready_generation_for_next_step(
            ready,
            caller_retained_untracked_logical_and_accounted_bytes=(
                2 * checkpoint.checkpoint_tensor_bytes
            ),
        )
    ready.assert_current()
    # Persistence is caller-owned; release both clone layers before resuming.
    del payload, checkpoint, parsed_checkpoint
    del lying_accounting_payload, oversized_storage_payload
    del oversized_positions, oversized_positions_backing
    del oversized_manifest_payload, raw_manifest
    del parsed, serialized, changed_geometry_payload
    del saved, live

    # The lifecycle cannot discover arbitrary caller roots.  Drop every old
    # generation/artifact/checkpoint expectation before the zero-byte claim.
    del current, expected
    del expected_geometry, expected_raw_color, expected_raw_density
    del authorization, accumulator, step_result
    del stale_artifact, old_provider, old_store, old_state, case

    next_coordinator = claim_paper_kinetic_fixed_camera_ready_generation_for_next_step(
        ready,
        caller_retained_untracked_logical_and_accounted_bytes=0,
    )
    next_coordinator.assert_current(ready.provider)
    assert next_coordinator.authorized_step_count == 1
    with pytest.raises(ValueError, match="already claimed"):
        claim_paper_kinetic_fixed_camera_ready_generation_for_next_step(
            ready,
            caller_retained_untracked_logical_and_accounted_bytes=0,
        )

    second_generation_policy = PaperKineticFixedCameraFullGeometryGenerationPolicy(
        step_index=ready.state.material_state.step_index,
        material_generation_id=ready.state.material_state.material_generation_id,
        background_generation_id="combined-background-generation-1",
        target_generation_id="combined-target-generation-1",
        geometry_generation_id=ready.state.geometry_generation_id,
    )
    second_step_result = run_paper_kinetic_fixed_camera_full_geometry_step(
        next_coordinator,
        ready.provider,
        _batch(),
        policy=_full_geometry_policy(),
        generation_policy=second_generation_policy,
        global_site_rgba_f32=ready.state.material_state.site_rgba_f32,
        background_rgb_f32=torch.tensor(
            (0.03, 0.05, 0.07),
            dtype=torch.float32,
        ),
        native_ops=_FakeNativeOps(),
        backend_provenance="cpu-fake-native/exact-op-surface",
        device_completion_fence=_Fence(),
        device_completion_fence_provenance="cpu-synchronous-fence-v1",
    )
    second_ready = apply_paper_kinetic_fixed_camera_combined_sgd_transaction(
        ready.state,
        ready.provider,
        ready.artifact_store,
        second_step_result,
        policy=policy,
        cold_recompile_manifest=ready.manifest,
        fresh_store_policy=ready.artifact_store.policy,
    )

    second_ready.assert_current()
    assert second_ready.state.geometry_update_count == 2
    assert second_ready.state.material_state.step_index == 2
    assert second_ready.update_receipt.step_index == 2
    assert next_coordinator.authorized_step_count == 2
    ready.state.assert_retired()
    with pytest.raises(ValueError):
        second_step_result.assert_current()


def test_combined_transaction_preserves_fused_reverse_provenance(
    tmp_path,
    monkeypatch,
) -> None:
    _install_fused_coordinator_cpu_fake(monkeypatch)
    case = _authorized_case(
        tmp_path,
        full_geometry_reverse_mode=FUSED_DIRECT_V1_FULL_GEOMETRY_REVERSE,
    )
    step_result_generation_digest = case["step_result"].generation_digest

    ready = apply_paper_kinetic_fixed_camera_combined_sgd_transaction(
        case["combined_state"],
        case["provider"],
        case["store"],
        case["step_result"],
        policy=_transaction_policy(),
        cold_recompile_manifest=case["manifest"],
        fresh_store_policy=case["store_policy"],
    )

    ready.assert_current()
    assert (
        ready.update_receipt.full_geometry_step_result_generation_digest
        == step_result_generation_digest
    )
    assert (
        ready.update_receipt.full_geometry_reverse_mode
        == FUSED_DIRECT_V1_FULL_GEOMETRY_REVERSE
    )
    assert ready.update_receipt.full_geometry_step_result_revoked
    with pytest.raises(ValueError):
        case["step_result"].assert_current()


def test_combined_checkpoint_restore_matches_uninterrupted_next_step(
    tmp_path,
) -> None:
    case = _authorized_case(tmp_path)
    policy = _transaction_policy()
    uninterrupted = apply_paper_kinetic_fixed_camera_combined_sgd_transaction(
        case["combined_state"],
        case["provider"],
        case["store"],
        case["step_result"],
        policy=policy,
        cold_recompile_manifest=case["manifest"],
        fresh_store_policy=case["store_policy"],
    )
    checkpoint = checkpoint_paper_kinetic_fixed_camera_combined_state(
        uninterrupted.state,
        uninterrupted.provider,
        uninterrupted.artifact_store,
        manifest=uninterrupted.manifest,
        recompile_receipt=uninterrupted.recompile_receipt,
        policy=policy,
        initializer_generation_digest=(
            uninterrupted.provider.initializer_generation_digest
        ),
    )
    payload = checkpoint.payload()

    _fresh_source, _fresh_factory, runtime_template = _provider()
    target_provider = runtime_template.target_provider
    ray_provider = runtime_template.ray_provider
    program_factory = runtime_template.program_factory
    frame_times = runtime_template.frame_times
    maximum_observations_per_bundle = (
        runtime_template.maximum_observations_per_bundle
    )
    maximum_rows_per_native_block = (
        runtime_template.maximum_rows_per_native_block
    )
    object.__setattr__(runtime_template, "_seal", None)
    del runtime_template

    restored = restore_paper_kinetic_fixed_camera_combined_generation_from_payload(
        payload,
        expected_world_site_count=uninterrupted.state.site_count,
        expected_combined_sgd_policy=policy,
        target_provider=target_provider,
        ray_provider=ray_provider,
        frame_times=frame_times,
        maximum_observations_per_bundle=maximum_observations_per_bundle,
        maximum_rows_per_native_block=maximum_rows_per_native_block,
        program_factory=program_factory,
        fresh_store_policy=case["store_policy"],
        device="cpu",
    )

    restored.assert_current()
    assert restored.state.generation_digest == uninterrupted.state.generation_digest
    assert restored.provider.generation_digest == uninterrupted.provider.generation_digest
    assert (
        restored.recompile_receipt.generation_digest
        == uninterrupted.recompile_receipt.generation_digest
        == checkpoint.cold_recompile_seal_generation_digest
    )
    assert restored.restore_receipt.checkpoint_generation_digest == (
        checkpoint.generation_digest
    )
    assert restored.restore_receipt.source_payload_tensor_bytes == (
        checkpoint.checkpoint_tensor_bytes
    )
    assert (
        restored.restore_receipt
        .restore_tracked_logical_and_store_accounted_bytes_upper_bound
        == checkpoint.state_checkpoint_payload_peak_logical_tensor_bytes
        + case["store_policy"].maximum_resident_accounted_bytes
    )
    for restored_tensor, uninterrupted_tensor in zip(
        restored.state._geometry_tensors(),
        uninterrupted.state._geometry_tensors(),
        strict=True,
    ):
        torch.testing.assert_close(
            restored_tensor,
            uninterrupted_tensor,
            rtol=0.0,
            atol=0.0,
        )
        assert (
            restored_tensor.untyped_storage().data_ptr()
            != uninterrupted_tensor.untyped_storage().data_ptr()
        )
    torch.testing.assert_close(
        restored.state.material_state.raw_color_f32,
        uninterrupted.state.material_state.raw_color_f32,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        restored.state.material_state.raw_density_f32,
        uninterrupted.state.material_state.raw_density_f32,
        rtol=0.0,
        atol=0.0,
    )

    del payload, checkpoint
    uninterrupted_coordinator = (
        claim_paper_kinetic_fixed_camera_ready_generation_for_next_step(
            uninterrupted,
            caller_retained_untracked_logical_and_accounted_bytes=0,
        )
    )
    restored_coordinator = (
        claim_paper_kinetic_fixed_camera_restored_ready_generation_for_next_step(
            restored,
            caller_retained_untracked_logical_and_accounted_bytes=0,
        )
    )
    generation_policy = PaperKineticFixedCameraFullGeometryGenerationPolicy(
        step_index=uninterrupted.state.material_state.step_index,
        material_generation_id=(
            uninterrupted.state.material_state.material_generation_id
        ),
        background_generation_id="combined-restore-background-generation-1",
        target_generation_id="combined-restore-target-generation-1",
        geometry_generation_id=uninterrupted.state.geometry_generation_id,
    )

    def run_next(coordinator, ready_generation):
        return run_paper_kinetic_fixed_camera_full_geometry_step(
            coordinator,
            ready_generation.provider,
            _batch(),
            policy=_full_geometry_policy(),
            generation_policy=generation_policy,
            global_site_rgba_f32=(
                ready_generation.state.material_state.site_rgba_f32
            ),
            background_rgb_f32=torch.tensor(
                (0.03, 0.05, 0.07),
                dtype=torch.float32,
            ),
            native_ops=_FakeNativeOps(),
            backend_provenance="cpu-fake-native/exact-op-surface",
            device_completion_fence=_Fence(),
            device_completion_fence_provenance="cpu-synchronous-fence-v1",
        )

    uninterrupted_step = run_next(uninterrupted_coordinator, uninterrupted)
    restored_step = run_next(restored_coordinator, restored)
    assert len(uninterrupted_step.authorization.generation_digest) == 64
    assert len(restored_step.authorization.generation_digest) == 64
    assert restored_step.authorization.generation_digest != (
        uninterrupted_step.authorization.generation_digest
    )
    assert restored_step.generation_digest != uninterrupted_step.generation_digest
    torch.testing.assert_close(
        restored_step.authorization.loss_f32,
        uninterrupted_step.authorization.loss_f32,
        rtol=0.0,
        atol=0.0,
    )
    for restored_bar, uninterrupted_bar in zip(
        (
            restored_step.authorization.grad_site_rgba_f32,
            restored_step.authorization.grad_positions0_f64,
            restored_step.authorization.grad_velocities_f64,
            restored_step.authorization.grad_weight_coefficients_f64,
        ),
        (
            uninterrupted_step.authorization.grad_site_rgba_f32,
            uninterrupted_step.authorization.grad_positions0_f64,
            uninterrupted_step.authorization.grad_velocities_f64,
            uninterrupted_step.authorization.grad_weight_coefficients_f64,
        ),
        strict=True,
    ):
        torch.testing.assert_close(
            restored_bar,
            uninterrupted_bar,
            rtol=0.0,
            atol=0.0,
        )
    uninterrupted_after = apply_paper_kinetic_fixed_camera_combined_sgd_transaction(
        uninterrupted.state,
        uninterrupted.provider,
        uninterrupted.artifact_store,
        uninterrupted_step,
        policy=policy,
        cold_recompile_manifest=uninterrupted.manifest,
        fresh_store_policy=case["store_policy"],
    )
    restored_after = apply_paper_kinetic_fixed_camera_combined_sgd_transaction(
        restored.state,
        restored.provider,
        restored.artifact_store,
        restored_step,
        policy=policy,
        cold_recompile_manifest=restored.manifest,
        fresh_store_policy=case["store_policy"],
    )

    assert restored_after.state.geometry_update_count == 2
    assert uninterrupted_after.state.geometry_update_count == 2
    assert restored_after.update_receipt.loss == (
        uninterrupted_after.update_receipt.loss
    )
    assert (
        restored_after.update_receipt.raw_color_gradient_norm,
        restored_after.update_receipt.raw_density_gradient_norm,
        restored_after.update_receipt.position_gradient_norm,
        restored_after.update_receipt.velocity_gradient_norm,
        restored_after.update_receipt.weight_gradient_norm,
    ) == (
        uninterrupted_after.update_receipt.raw_color_gradient_norm,
        uninterrupted_after.update_receipt.raw_density_gradient_norm,
        uninterrupted_after.update_receipt.position_gradient_norm,
        uninterrupted_after.update_receipt.velocity_gradient_norm,
        uninterrupted_after.update_receipt.weight_gradient_norm,
    )
    for restored_tensor, uninterrupted_tensor in zip(
        (
            *restored_after.state._geometry_tensors(),
            restored_after.state.material_state.raw_color_f32,
            restored_after.state.material_state.raw_density_f32,
        ),
        (
            *uninterrupted_after.state._geometry_tensors(),
            uninterrupted_after.state.material_state.raw_color_f32,
            uninterrupted_after.state.material_state.raw_density_f32,
        ),
        strict=True,
    ):
        torch.testing.assert_close(
            restored_tensor,
            uninterrupted_tensor,
            rtol=0.0,
            atol=0.0,
        )


def test_cold_recompile_streams_full_manifest_through_smaller_lru(tmp_path) -> None:
    case = _authorized_case(tmp_path)
    fresh_store_policy = PaperKineticCompiledCpuArtifactStorePolicy(
        maximum_entries=2,
        maximum_resident_accounted_bytes=20_000_000,
    )

    ready = apply_paper_kinetic_fixed_camera_combined_sgd_transaction(
        case["combined_state"],
        case["provider"],
        case["store"],
        case["step_result"],
        policy=_transaction_policy(),
        cold_recompile_manifest=case["manifest"],
        fresh_store_policy=fresh_store_policy,
    )

    ready.assert_current()
    report = ready.artifact_store.report()
    receipt = ready.recompile_receipt
    assert case["manifest"].request_count == 6
    assert report.lookup_count == case["manifest"].request_count
    assert report.miss_count == case["manifest"].request_count
    assert report.cold_compile_count == case["manifest"].request_count
    assert report.cold_compiled_track_count == case["manifest"].track_count
    assert report.hit_count == 0
    assert report.current_entry_count == fresh_store_policy.maximum_entries
    assert report.eviction_count == (
        case["manifest"].request_count - report.current_entry_count
    )
    assert receipt.store_current_entry_count == report.current_entry_count
    assert receipt.store_current_resident_accounted_bytes == (
        report.current_resident_accounted_bytes
    )
    assert receipt.eviction_count == report.eviction_count
    assert receipt.evicted_accounted_bytes == report.evicted_accounted_bytes
    assert receipt.all_acquisitions_cold
    assert receipt.full_manifest_digest_chain_bound
    assert receipt.final_store_is_bounded_lru_working_set
    assert (
        ready.update_receipt.fresh_store_resident_accounted_bytes_upper_bound
        == fresh_store_policy.maximum_resident_accounted_bytes
    )
    assert (
        ready.update_receipt
        .transaction_tracked_logical_and_store_accounted_bytes_upper_bound
        == ready.update_receipt.old_candidate_authorization_logical_tensor_bytes
        + ready.update_receipt.candidate_world_geometry_clone_logical_tensor_bytes
        + ready.update_receipt
        .update_validation_scratch_logical_tensor_bytes_upper_bound
        + ready.update_receipt.old_store_resident_accounted_bytes_before_retirement
        + fresh_store_policy.maximum_resident_accounted_bytes
    )


def test_transaction_memory_preflight_preserves_the_old_generation(tmp_path) -> None:
    case = _authorized_case(tmp_path)
    too_small = replace(
        _transaction_policy(),
        maximum_update_candidate_logical_tensor_bytes=1,
    )

    with pytest.raises(MemoryError, match="candidate"):
        apply_paper_kinetic_fixed_camera_combined_sgd_transaction(
            case["combined_state"],
            case["provider"],
            case["store"],
            case["step_result"],
            policy=too_small,
            cold_recompile_manifest=case["manifest"],
            fresh_store_policy=case["store_policy"],
        )

    geometry_clone_too_small = replace(
        _transaction_policy(),
        maximum_candidate_world_geometry_clone_logical_tensor_bytes=1,
    )
    with pytest.raises(MemoryError, match="geometry clone"):
        apply_paper_kinetic_fixed_camera_combined_sgd_transaction(
            case["combined_state"],
            case["provider"],
            case["store"],
            case["step_result"],
            policy=geometry_clone_too_small,
            cold_recompile_manifest=case["manifest"],
            fresh_store_policy=case["store_policy"],
        )

    validation_scratch_too_small = replace(
        _transaction_policy(),
        maximum_update_validation_scratch_logical_tensor_bytes=1,
    )
    with pytest.raises(MemoryError, match="validation scratch"):
        apply_paper_kinetic_fixed_camera_combined_sgd_transaction(
            case["combined_state"],
            case["provider"],
            case["store"],
            case["step_result"],
            policy=validation_scratch_too_small,
            cold_recompile_manifest=case["manifest"],
            fresh_store_policy=case["store_policy"],
        )

    payload_peak_too_small = replace(
        _transaction_policy(),
        maximum_state_checkpoint_payload_logical_tensor_bytes=1,
    )
    with pytest.raises(MemoryError, match="payload clone"):
        apply_paper_kinetic_fixed_camera_combined_sgd_transaction(
            case["combined_state"],
            case["provider"],
            case["store"],
            case["step_result"],
            policy=payload_peak_too_small,
            cold_recompile_manifest=case["manifest"],
            fresh_store_policy=case["store_policy"],
        )

    tracked_peak_too_small = replace(
        _transaction_policy(),
        maximum_transaction_tracked_logical_and_store_accounted_bytes=1,
    )
    with pytest.raises(MemoryError, match="tracked state/store peak"):
        apply_paper_kinetic_fixed_camera_combined_sgd_transaction(
            case["combined_state"],
            case["provider"],
            case["store"],
            case["step_result"],
            policy=tracked_peak_too_small,
            cold_recompile_manifest=case["manifest"],
            fresh_store_policy=case["store_policy"],
        )

    case["combined_state"].assert_current(case["provider"], case["store"])
    case["step_result"].assert_current()
    assert case["store"].report().current_entry_count == 6


def test_transaction_rejects_content_identical_foreign_provider(tmp_path) -> None:
    case = _authorized_case(tmp_path)
    foreign_provider = replace(case["provider"])
    foreign_provider.assert_current()
    assert foreign_provider.generation_digest == case["provider"].generation_digest
    assert id(foreign_provider) != id(case["provider"])
    foreign_store = PaperKineticCompiledCpuArtifactStore(case["store_policy"])
    foreign_state = prepare_paper_kinetic_fixed_camera_combined_state(
        case["material_state"],
        foreign_provider,
        foreign_store,
        maximum_combined_state_logical_tensor_bytes=1_000_000,
    )

    with pytest.raises(ValueError, match="stale/foreign"):
        apply_paper_kinetic_fixed_camera_combined_sgd_transaction(
            foreign_state,
            foreign_provider,
            foreign_store,
            case["step_result"],
            policy=_transaction_policy(),
            cold_recompile_manifest=case["manifest"],
            fresh_store_policy=case["store_policy"],
        )

    foreign_state.assert_current(foreign_provider, foreign_store)
    case["step_result"].assert_current()


def test_cold_recompile_failure_poison_retires_both_generations(
    tmp_path,
    monkeypatch,
) -> None:
    case = _authorized_case(tmp_path)

    def fail_compile(_provider, _key):
        raise RuntimeError("injected cold compile failure")

    monkeypatch.setattr(
        combined_module,
        "compile_paper_kinetic_compiled_cpu_artifact",
        fail_compile,
    )
    with pytest.raises(PaperKineticFixedCameraCombinedTransactionFailure) as raised:
        apply_paper_kinetic_fixed_camera_combined_sgd_transaction(
            case["combined_state"],
            case["provider"],
            case["store"],
            case["step_result"],
            policy=_transaction_policy(),
            cold_recompile_manifest=case["manifest"],
            fresh_store_policy=case["store_policy"],
        )

    failure = raised.value
    assert failure.stage == "cold_recompile"
    assert failure.restart_required
    assert failure.old_generation_unusable
    assert failure.candidate_generation_unusable
    case["combined_state"].assert_retired()
    assert case["store"].report().current_entry_count == 0
    with pytest.raises(ValueError):
        case["step_result"].assert_current()
    assert case["step_result"].authorization.grad_site_rgba_f32 is None
    assert case["step_result"].accumulator.grad_site_rgba_f32 is None
    with pytest.raises(ValueError):
        case["provider"].assert_current()
