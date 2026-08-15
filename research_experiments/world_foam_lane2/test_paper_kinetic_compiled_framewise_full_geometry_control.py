from __future__ import annotations

import hashlib

import torch
from kinetic_compiled_cpu_artifact_store import (
    PaperKineticCompiledCpuArtifactStore,
    PaperKineticCompiledCpuArtifactStorePolicy,
)
from paper_kinetic_active_track_program_factory import (
    PaperKineticActiveP0TrackProgramFactoryConfig,
    prepare_paper_kinetic_active_p0_track_program_factory,
)
from paper_kinetic_compiled_framewise_full_geometry_control import (
    prepare_paper_kinetic_compiled_framewise_program_provider,
    run_paper_kinetic_compiled_framewise_full_geometry_control,
)
from paper_kinetic_fixed_camera_combined_state import (
    PaperKineticFixedCameraCombinedSGDPolicy,
    prepare_paper_kinetic_fixed_camera_combined_state,
)
from paper_kinetic_fixed_site_material_state import (
    PaperKineticFixedSiteMaterialParameterization,
    PaperKineticFixedSiteMaterialSGDPolicy,
    prepare_paper_kinetic_fixed_site_material_state,
)
from paper_kinetic_lazy_full_geometry_step import (
    PaperKineticLazyFullGeometryMemoryPolicy,
)
from paper_kinetic_world_initializer import (
    prepare_paper_kinetic_p0_material_initialization,
)
from test_kinetic_lazy_native_material_step import (
    _FakeNativeOps,
    _background,
    _material,
    _memory_policy,
    _provider,
)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _combined_policy() -> PaperKineticFixedCameraCombinedSGDPolicy:
    return PaperKineticFixedCameraCombinedSGDPolicy(
        position_learning_rate=1.0e-3,
        velocity_learning_rate=1.0e-3,
        weight_learning_rate=1.0e-3,
        maximum_absolute_position_update=1.0,
        maximum_absolute_velocity_update=1.0,
        maximum_absolute_weight_update=1.0,
        maximum_absolute_position_value=10.0,
        maximum_absolute_velocity_value=10.0,
        maximum_absolute_weight_value=10.0,
        maximum_combined_state_logical_tensor_bytes=1_000_000,
        maximum_update_candidate_logical_tensor_bytes=1_000_000,
        maximum_candidate_world_geometry_clone_logical_tensor_bytes=1_000_000,
        maximum_update_validation_scratch_logical_tensor_bytes=1_000_000,
        maximum_old_candidate_authorization_logical_tensor_bytes=2_000_000,
        maximum_checkpoint_logical_tensor_bytes=1_000_000,
        maximum_state_checkpoint_logical_tensor_bytes=2_000_000,
        maximum_state_checkpoint_payload_logical_tensor_bytes=3_000_000,
        maximum_transaction_tracked_logical_and_store_accounted_bytes=64_000_000,
        maximum_recompile_request_count=1,
        maximum_recompile_track_id_logical_bytes=1_000_000,
        maximum_artifact_accounted_bytes=10_000_000,
    )


def test_compiled_framewise_control_compiles_once_replays_frames_and_mutates_once() -> None:
    factory = prepare_paper_kinetic_active_p0_track_program_factory(
        PaperKineticActiveP0TrackProgramFactoryConfig(
            near=0.0,
            far=2.0,
            node_count=2,
            maximum_sites_per_track_compile=8,
            maximum_charts_per_track=16,
            maximum_owner_runs_per_chart=8,
            rank_selection_provenance="compiled-framewise-control-test-v1",
        )
    )
    _, _, base_provider = _provider(
        maximum_tracks_per_bundle=1,
        maximum_observations_per_bundle=1,
        factory=factory,
    )
    store = PaperKineticCompiledCpuArtifactStore(
        PaperKineticCompiledCpuArtifactStorePolicy(
            maximum_entries=2,
            maximum_resident_accounted_bytes=10_000_000,
        )
    )
    provider = prepare_paper_kinetic_compiled_framewise_program_provider(
        base_provider,
        store,
        selected_track_ids=(0, 2),
        maximum_artifact_accounted_bytes_per_entry=5_000_000,
    )
    initialization = prepare_paper_kinetic_p0_material_initialization(
        _material(),
        provider.world.sites,
        initializer_generation_digest=_sha256("compiled-framewise-material-init"),
        source_material_seed_digest=_sha256("compiled-framewise-material-seed"),
    )
    material_state = prepare_paper_kinetic_fixed_site_material_state(
        initialization,
        provider.world,
        parameterization=PaperKineticFixedSiteMaterialParameterization(),
        optimizer_policy=PaperKineticFixedSiteMaterialSGDPolicy(
            color_learning_rate=1.0e-2,
            density_learning_rate=1.0e-2,
        ),
        device="cpu",
        maximum_material_state_logical_tensor_bytes=1_000_000,
    )
    state = prepare_paper_kinetic_fixed_camera_combined_state(
        material_state,
        provider,
        store,
        maximum_combined_state_logical_tensor_bytes=1_000_000,
    )
    before = tuple(
        tensor.clone()
        for tensor in (
            state.material_state.raw_color_f32,
            state.material_state.raw_density_f32,
            state.positions0_f64,
            state.velocities_f64,
            state.weight_coefficients_f64,
        )
    )
    result = run_paper_kinetic_compiled_framewise_full_geometry_control(
        state,
        provider,
        store,
        selected_frame_indices=(0, 1),
        selected_track_ids=(0, 2),
        global_site_rgba_f32=state.material_state.site_rgba_f32,
        background_rgb_f32=_background(),
        background_generation_id=_sha256("compiled-framewise-background"),
        native_ops=_FakeNativeOps(),
        maximum_samples_per_launch=1,
        cone_tolerance=1.0e-5,
        memory_policy=_memory_policy(provider),
        full_geometry_memory_policy=PaperKineticLazyFullGeometryMemoryPolicy(
            maximum_global_geometry_bar_logical_tensor_bytes=1_000_000,
            maximum_geometry_bridge_visible_peak_logical_tensor_bytes=10_000_000,
            maximum_fused_union_transaction_scratch_tensor_bytes=0,
        ),
        combined_sgd_policy=_combined_policy(),
        device_completion_fence=lambda: None,
        device_completion_fence_provenance="cpu-synchronous-fake-native-v1",
        emit_parity_payload=True,
    )
    result.assert_current()
    accounting = result.accounting
    assert result.precompile_receipt.compile_pass_count == 1
    assert result.precompile_receipt.request_count == 2
    assert result.precompile_receipt.track_count == 2
    assert accounting["per_frame_replay_count"] == 2
    assert accounting["compiled_artifact_warm_hit_count"] == 4
    assert accounting["per_frame_continuous_recompile_count"] == 0
    assert accounting["maximum_simultaneously_live_frame_count"] == 1
    assert accounting["cpu_optimizer_mutation_count"] == 1
    assert accounting["fresh_selected_track_recompile_count"] == 0
    coordinator = accounting[
        "frame_coordinator_visible_logical_tensor_bytes_upper_bound"
    ]
    geometry_bridge = accounting[
        "frame_geometry_bridge_visible_logical_tensor_bytes_upper_bound"
    ]
    material_bar = accounting["frame_material_bar_logical_tensor_bytes"]
    geometry_bars = accounting["frame_geometry_bar_logical_tensor_bytes"]
    readback_and_loss = accounting[
        "frame_material_readback_and_loss_logical_tensor_bytes"
    ]
    corrected_bound = accounting[
        "maximum_frame_local_logical_tensor_bytes_upper_bound"
    ]
    assert accounting["frame_material_bar_included_in_coordinator_bound"] is True
    assert accounting["frame_geometry_bridge_may_overlap_coordinator"] is True
    assert material_bar == state.site_count * 4 * 4
    assert geometry_bars == state.geometry_tensor_bytes
    assert readback_and_loss == material_bar + 4
    assert corrected_bound == (
        coordinator + geometry_bridge + geometry_bars + readback_and_loss
    )
    old_omitting_bound = max(coordinator, geometry_bridge) + readback_and_loss
    assert corrected_bound > old_omitting_bound
    assert result.parity_payload is not None
    assert not state.active
    assert state.poisoned
    after = (
        state.material_state.raw_color_f32,
        state.material_state.raw_density_f32,
        state.positions0_f64,
        state.velocities_f64,
        state.weight_coefficients_f64,
    )
    assert any(not torch.equal(left, right) for left, right in zip(before, after, strict=True))
