from __future__ import annotations

import hashlib

import pytest
import tests.test_paper_kinetic_lazy_program_bundles as lazy_fixture
import torch
from kinetic_compiled_cpu_artifact_store import (
    PaperKineticCompiledCpuArtifactStore,
    PaperKineticCompiledCpuArtifactStorePolicy,
)
from kinetic_power_word_compiler import AffineKineticPowerSites
from kinetic_sealed_completion_fence import (
    prepare_paper_kinetic_sealed_completion_fence,
)
from paper_kinetic_fixed_camera_combined_state import (
    PaperKineticFixedCameraCombinedSGDPolicy,
    checkpoint_paper_kinetic_fixed_camera_combined_state,
    prepare_paper_kinetic_fixed_camera_cold_recompile_manifest,
    prepare_paper_kinetic_fixed_camera_combined_state,
)
from paper_kinetic_fixed_camera_full_geometry_step import (
    paper_kinetic_fixed_camera_provider_geometry_generation_id,
)
from paper_kinetic_fixed_site_material_device_bridge import (
    snapshot_paper_kinetic_fixed_site_material_to_device,
)
from paper_kinetic_fixed_site_material_state import (
    PaperKineticFixedSiteMaterialParameterization,
    PaperKineticFixedSiteMaterialSGDPolicy,
    prepare_paper_kinetic_fixed_site_material_state,
)
from paper_kinetic_lazy_full_geometry_device_bridge import (
    apply_paper_kinetic_lazy_full_geometry_combined_sgd_transaction,
    seal_paper_kinetic_lazy_full_geometry_device_gradient_receipt,
)
from paper_kinetic_lazy_full_geometry_step import (
    STAGED_SPARSE,
    PaperKineticLazyFullGeometryMemoryPolicy,
    prepare_paper_kinetic_lazy_full_geometry_execution_context,
)
from paper_kinetic_lazy_program_bundles import (
    PaperKineticWorldInitializationRequest,
    prepare_paper_kinetic_lazy_program_bundle_provider,
)
from paper_kinetic_world_initializer import (
    prepare_paper_kinetic_p0_material_initialization,
)
from powerfoam_training_data import PowerFoamRayProvider, PowerFoamTargetProvider
from test_kinetic_compiled_cpu_artifact_store import (
    _NonRetainingStaticProgramFactory,
)
from test_kinetic_lazy_native_material_step import _FakeNativeOps


def _sha256(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class _TwoSiteWorldInitializer:
    provenance = "lazy-combined-two-site-world-v1"

    def __init__(self) -> None:
        self.generation_digest = _sha256(self.provenance)
        self.sites = AffineKineticPowerSites(
            positions0=torch.tensor(
                ((-0.18, 0.0, 0.6), (0.24, 0.05, 1.1)),
                dtype=torch.float64,
            ),
            velocities=torch.tensor(
                ((0.01, -0.02, 0.0), (-0.015, 0.01, 0.005)),
                dtype=torch.float64,
            ),
            weight_coefficients=torch.tensor(
                ((-0.04,), (0.07,)),
                dtype=torch.float64,
            ),
        )

    def initialize_world(
        self,
        request: PaperKineticWorldInitializationRequest,
    ) -> AffineKineticPowerSites:
        request.assert_self_consistent()
        return self.sites


def _provider():
    source = lazy_fixture.LazyTargetSource()
    target_provider = PowerFoamTargetProvider(source=source, device="cpu")
    ray_provider = PowerFoamRayProvider(
        cameras=lazy_fixture._camera_grid(),
        height=source.height,
        width=source.width,
        device="cpu",
    )
    provider = prepare_paper_kinetic_lazy_program_bundle_provider(
        dataset_generation_digest=_sha256("lazy-combined-two-site-dataset"),
        target_provider=target_provider,
        ray_provider=ray_provider,
        frame_times=(0.0, 0.4, 1.0),
        height=source.height,
        width=source.width,
        maximum_tracks_per_bundle=2,
        maximum_observations_per_bundle=4,
        maximum_rows_per_native_block=1,
        world_initializer=_TwoSiteWorldInitializer(),
        program_factory=_NonRetainingStaticProgramFactory(),
    )
    return provider


def _combined_policy() -> PaperKineticFixedCameraCombinedSGDPolicy:
    return PaperKineticFixedCameraCombinedSGDPolicy(
        position_learning_rate=1.0e-3,
        velocity_learning_rate=2.0e-3,
        weight_learning_rate=3.0e-3,
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
        maximum_recompile_request_count=3,
        maximum_recompile_track_id_logical_bytes=1_000_000,
        maximum_artifact_accounted_bytes=10_000_000,
    )


def test_lazy_combined_sgd_mutates_two_sites_and_retires_stale_generation() -> None:
    provider = _provider()
    store_policy = PaperKineticCompiledCpuArtifactStorePolicy(
        maximum_entries=3,
        maximum_resident_accounted_bytes=30_000_000,
    )
    store = PaperKineticCompiledCpuArtifactStore(store_policy)
    physical_material = torch.tensor(
        ((0.25, 0.35, 0.45, 0.8), (0.65, 0.55, 0.30, 0.6)),
        dtype=torch.float32,
    )
    initialization = prepare_paper_kinetic_p0_material_initialization(
        physical_material,
        provider.world.sites,
        initializer_generation_digest=_sha256("lazy-combined-material-initializer"),
        source_material_seed_digest=_sha256("lazy-combined-material-seed"),
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
    background_generation_id = _sha256("lazy-combined-background")
    snapshot = snapshot_paper_kinetic_fixed_site_material_to_device(
        material_state,
        background_rgb_f32_cpu=torch.tensor((0.03, 0.05, 0.07)),
        background_generation_id=background_generation_id,
        device="cpu",
        device_completion_fence=lambda: None,
        device_completion_fence_provenance="cpu-synchronous-fake-native-v1",
    )
    position_bar = torch.empty((2, 3), dtype=torch.float64)
    velocity_bar = torch.empty((2, 3), dtype=torch.float64)
    weight_bar = torch.empty((2, 1), dtype=torch.float64)
    context = prepare_paper_kinetic_lazy_full_geometry_execution_context(
        provider,
        reverse_mode=STAGED_SPARSE,
        policy=PaperKineticLazyFullGeometryMemoryPolicy(
            maximum_global_geometry_bar_logical_tensor_bytes=1_000_000,
            maximum_geometry_bridge_visible_peak_logical_tensor_bytes=1_000_000,
            maximum_fused_union_transaction_scratch_tensor_bytes=0,
        ),
        geometry_generation_id=state.geometry_generation_id,
        grad_positions0_f64_cpu=position_bar,
        grad_velocities_f64_cpu=velocity_bar,
        grad_weight_coefficients_f64_cpu=weight_bar,
    )
    position_bar.copy_(
        torch.tensor(((0.4, -0.2, 0.1), (-0.3, 0.5, -0.4)), dtype=torch.float64)
    )
    velocity_bar.copy_(
        torch.tensor(((-0.2, 0.3, 0.6), (0.7, -0.1, 0.2)), dtype=torch.float64)
    )
    weight_bar.copy_(torch.tensor(((0.8,), (-0.9,)), dtype=torch.float64))
    geometry_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (position_bar, velocity_bar, weight_bar)
    )
    context.native_full_geometry_vjp_launch_count = 1
    context.geometry_compact_to_global_scatter_row_count = 2
    context.maximum_geometry_bridge_visible_peak_logical_tensor_bytes = geometry_bytes
    context._append_receipt(
        reverse_mode=STAGED_SPARSE,
        bundle_index=0,
        completion_fence_sequence=1,
        completion_launch_generation_digest=_sha256("lazy-combined-geometry-launch"),
        completion_receipt_generation_digest=_sha256("lazy-combined-geometry-fence"),
        source_index_space="block_compact",
        source_site_ids=(0, 1),
        active_native_block_count=1,
        source_tensor_bytes=geometry_bytes,
        cpu_tensor_bytes=geometry_bytes,
        exact_request_union_identity_certified=False,
        compact_owner_certificate_consumed=True,
        device_to_host_tensor_count=3,
        device_to_host_tensor_bytes=geometry_bytes,
        source_transaction_generation_id=_sha256("lazy-combined-geometry-transaction"),
    )
    native_ops = _FakeNativeOps()
    completion_fence = prepare_paper_kinetic_sealed_completion_fence(
        native_ops,
        device="cpu",
        owner_generation_digest=_sha256("lazy-combined-step-owner"),
    )
    material_bar = torch.tensor(
        ((0.2, -0.1, 0.3, 0.4), (-0.25, 0.15, -0.2, 0.35)),
        dtype=torch.float32,
    )
    result = context.build_result(
        step_index=0,
        step_generation_id="lazy-combined-step-0",
        provider_generation_digest=provider.generation_digest,
        world_generation_digest=provider.world.generation_digest,
        sites_content_digest=provider.world.sites_content_digest,
        loss_normalization_id="lazy-combined-two-site-loss",
        material_generation_id=material_state.material_generation_id,
        background_generation_id=background_generation_id,
        loss_f32=torch.tensor((1.25,), dtype=torch.float32),
        grad_global_site_rgba_f32=material_bar,
        material_tensor=snapshot.site_rgba_f32_device,
        background_tensor=snapshot.background_rgb_f32_device,
        sealed_completion_fence=completion_fence,
        accounting={},
    )
    bridge_receipt = seal_paper_kinetic_lazy_full_geometry_device_gradient_receipt(
        state,
        provider,
        store,
        snapshot,
        result,
    )
    manifest = prepare_paper_kinetic_fixed_camera_cold_recompile_manifest(
        provider,
        view_indices=(0,),
        maximum_tracks_per_request=2,
    )
    old_positions = state.positions0_f64.clone()
    old_velocities = state.velocities_f64.clone()
    old_weights = state.weight_coefficients_f64.clone()
    old_material = state.material_state.site_rgba_f32.clone()

    runtime_measurements = {}
    ready = apply_paper_kinetic_lazy_full_geometry_combined_sgd_transaction(
        state,
        provider,
        store,
        bridge_receipt,
        policy=_combined_policy(),
        cold_recompile_manifest=manifest,
        fresh_store_policy=store_policy,
        runtime_measurements=runtime_measurements,
    )

    ready.assert_current()
    assert set(runtime_measurements) == {"cold_cpu_compile_wall_time_seconds"}
    assert runtime_measurements["cold_cpu_compile_wall_time_seconds"] >= 0.0
    assert ready.state.site_count == 2
    assert ready.state.geometry_update_count == 1
    assert not torch.equal(ready.state.positions0_f64, old_positions)
    assert not torch.equal(ready.state.velocities_f64, old_velocities)
    assert not torch.equal(ready.state.weight_coefficients_f64, old_weights)
    assert not torch.equal(ready.state.material_state.site_rgba_f32, old_material)
    update = ready.update_receipt
    for value in (
        update.grad_site_rgba_l2_norm,
        update.grad_positions0_l2_norm,
        update.grad_velocities_l2_norm,
        update.grad_weight_coefficients_l2_norm,
        update.raw_color_parameter_delta_l2_norm,
        update.raw_density_parameter_delta_l2_norm,
        update.positions0_parameter_delta_l2_norm,
        update.velocities_parameter_delta_l2_norm,
        update.weight_coefficients_parameter_delta_l2_norm,
    ):
        assert value > 0.0
    assert update.geometry_d2h_receipt_count == 1
    assert update.stale_provider_store_retirement_count == 1
    assert update.fresh_full_interval_recompile_count == 1
    assert update.cold_compiled_request_count == manifest.request_count
    assert update.core_accounting["native_full_geometry_vjp_launch_count"] == 1
    update_accounting = update.accounting()
    assert update_accounting["checkpoint_payload_supported"]
    assert update_accounting["checkpoint_restore_resume_supported"]
    assert update_accounting["checkpoint_restore_requires_fresh_runtime_inputs"]
    assert update_accounting["checkpoint_restore_resume_api"] == (
        "restore_paper_kinetic_fixed_camera_combined_generation_from_payload"
    )
    checkpoint = checkpoint_paper_kinetic_fixed_camera_combined_state(
        ready.state,
        ready.provider,
        ready.artifact_store,
        manifest=ready.manifest,
        recompile_receipt=ready.recompile_receipt,
        policy=_combined_policy(),
        initializer_generation_digest=ready.provider.initializer_generation_digest,
    )
    checkpoint.assert_current()
    assert checkpoint.payload()["combined_checkpoint_restore_integrated"] is False
    bridge_receipt.assert_consumed()
    state.assert_retired()
    assert store.report().current_entry_count == 0
    with pytest.raises(ValueError):
        provider.assert_current()
    with pytest.raises(ValueError):
        result.assert_current()
