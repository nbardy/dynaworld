from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType

import kinetic_dense_cached_native_material_request as dense_request_module
import paper_kinetic_fixed_camera_full_geometry_step as step_module
import pytest
import torch
from kinetic_compiled_cpu_artifact_store import (
    PaperKineticCompiledCpuArtifactStore,
    PaperKineticCompiledCpuArtifactStorePolicy,
)
from kinetic_dense_cached_native_material_request import (
    FUSED_DIRECT_V1_FULL_GEOMETRY_REVERSE,
    STAGED_SPARSE_FULL_GEOMETRY_REVERSE,
    PaperKineticDenseCachedNativeMemoryPolicy,
)
from paper_kinetic_fixed_camera_full_geometry_step import (
    PaperKineticFixedCameraFullGeometryGenerationPolicy,
    PaperKineticFixedCameraFullGeometryStepPartialFailure,
    PaperKineticFixedCameraFullGeometryStepPolicy,
    paper_kinetic_fixed_camera_provider_geometry_generation_id,
    prepare_paper_kinetic_fixed_camera_full_geometry_step_state,
    run_paper_kinetic_fixed_camera_full_geometry_step,
)
from paper_kinetic_replayable_observations import (
    OBSERVATION_IDENTITY_LOGICAL_BYTES,
    TRACK_ID_LOGICAL_BYTES,
    PaperKineticDenseObservationMemoryPolicy,
)
from paper_training_types import SpacetimeBatch, SpacetimeSample
from test_kinetic_compiled_cpu_artifact_store import _provider
from test_kinetic_ragged_paper_step_cpu_fake_native import _FakeNativeOps
from test_kinetic_native_material_step_executor import (
    _FakeFusedPreparedBlock,
    _install_fused_session_cpu_fake,
)


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
    full_geometry_reverse_mode: str = STAGED_SPARSE_FULL_GEOMETRY_REVERSE,
) -> PaperKineticFixedCameraFullGeometryStepPolicy:
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
        maximum_request_geometry_bar_tensor_bytes=10_000_000,
        maximum_geometry_bridge_visible_peak_logical_tensor_bytes=10_000_000,
        maximum_fused_prepared_owned_logical_tensor_bytes=(
            10_000_000
            if full_geometry_reverse_mode
            == FUSED_DIRECT_V1_FULL_GEOMETRY_REVERSE
            else 0
        ),
        maximum_fused_output_scratch_logical_tensor_bytes=(
            10_000_000
            if full_geometry_reverse_mode
            == FUSED_DIRECT_V1_FULL_GEOMETRY_REVERSE
            else 0
        ),
        maximum_fused_geometry_bridge_visible_peak_logical_tensor_bytes=(
            10_000_000
            if full_geometry_reverse_mode
            == FUSED_DIRECT_V1_FULL_GEOMETRY_REVERSE
            else 0
        ),
    )
    return PaperKineticFixedCameraFullGeometryStepPolicy(
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
        maximum_geometry_bar_logical_tensor_bytes=10_000_000,
        full_geometry_reverse_mode=full_geometry_reverse_mode,
    )


def _generation(provider) -> PaperKineticFixedCameraFullGeometryGenerationPolicy:
    return PaperKineticFixedCameraFullGeometryGenerationPolicy(
        step_index=0,
        material_generation_id="fixed-camera-material-generation-0",
        background_generation_id="fixed-camera-background-generation-0",
        target_generation_id="fixed-camera-target-generation-0",
        geometry_generation_id=(
            paper_kinetic_fixed_camera_provider_geometry_generation_id(provider)
        ),
    )


def _case():
    target_source, factory, provider = _provider()
    store = PaperKineticCompiledCpuArtifactStore(
        PaperKineticCompiledCpuArtifactStorePolicy(
            maximum_entries=6,
            maximum_resident_accounted_bytes=60_000_000,
        )
    )
    state = prepare_paper_kinetic_fixed_camera_full_geometry_step_state(
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


def _run(
    state,
    provider,
    material,
    background,
    *,
    policy=None,
    generation_policy=None,
    fence=None,
):
    return run_paper_kinetic_fixed_camera_full_geometry_step(
        state,
        provider,
        _batch(),
        policy=_policy() if policy is None else policy,
        generation_policy=(
            _generation(provider) if generation_policy is None else generation_policy
        ),
        global_site_rgba_f32=material,
        background_rgb_f32=background,
        native_ops=_FakeNativeOps(),
        backend_provenance="cpu-fake-native/exact-op-surface",
        device_completion_fence=_Fence() if fence is None else fence,
        device_completion_fence_provenance="cpu-synchronous-fence-v1",
    )


def test_fixed_camera_step_authorizes_one_fenced_full_vjp_per_active_block(
    monkeypatch,
) -> None:
    (
        target_source,
        factory,
        provider,
        store,
        state,
        material,
        background,
    ) = _case()
    world_before = tuple(
        tensor.clone()
        for tensor in (
            provider.world.sites.positions0,
            provider.world.sites.velocities,
            provider.world.sites.weight_coefficients,
        )
    )
    prepare_kwargs = {}
    original_prepare = step_module.prepare_paper_kinetic_dense_step_gradient_accumulator

    def capture_prepare(source, session, **kwargs):
        prepare_kwargs.update(kwargs)
        return original_prepare(source, session, **kwargs)

    monkeypatch.setattr(
        step_module,
        "prepare_paper_kinetic_dense_step_gradient_accumulator",
        capture_prepare,
    )
    fence = _Fence()
    result = _run(
        state,
        provider,
        material,
        background,
        fence=fence,
    )

    result.assert_current()
    assert prepare_kwargs["full_geometry"] is True
    assert prepare_kwargs["optimize_camera_rays"] is False
    assert result.authorization.full_geometry is True
    assert result.authorization.optimize_camera_rays is False
    assert result.authorization.ray_bar_keys == ()
    assert result.authorization.grad_track_ray_coefficients_f64 is None
    assert result.accumulator.ray_bar_keys == ()
    assert result.accumulator.grad_track_ray_coefficients_f64 is None
    for bar, geometry in zip(
        (
            result.authorization.grad_positions0_f64,
            result.authorization.grad_velocities_f64,
            result.authorization.grad_weight_coefficients_f64,
        ),
        world_before,
        strict=True,
    ):
        assert bar is not None
        assert bar.dtype == torch.float64
        assert bar.device.type == "cpu"
        assert tuple(bar.shape) == tuple(geometry.shape)
    for current, before in zip(
        (
            provider.world.sites.positions0,
            provider.world.sites.velocities,
            provider.world.sites.weight_coefficients,
        ),
        world_before,
        strict=True,
    ):
        torch.testing.assert_close(current, before, rtol=0.0, atol=0.0)

    accounting = result.accounting
    assert accounting["world_generation_digest"] == provider.world.generation_digest
    assert accounting["world_sites_content_digest"] == (
        provider.world.sites_content_digest
    )
    assert accounting["world_site_count"] == provider.world.site_count
    assert accounting["site_table_identity"] == id(provider.world.sites)
    assert accounting["geometry_generation_id"] == (
        paper_kinetic_fixed_camera_provider_geometry_generation_id(provider)
    )
    active_blocks = accounting["active_native_block_count"]
    assert accounting["native_material_vjp_launch_count"] == 0
    assert accounting["native_full_geometry_vjp_launch_count"] == active_blocks
    assert accounting["native_full_geometry_fenced_reduction_count"] == active_blocks
    assert accounting["geometry_reduction_fence_call_count"] == active_blocks
    assert accounting["geometry_completion_receipt_count"] == active_blocks
    assert accounting["exactly_one_full_geometry_vjp_per_active_block"]
    assert not accounting["geometry_completion_receipt_retains_native_tensors"]
    assert accounting["native_length_bar_released_after_fenced_reduction"]
    assert accounting["geometry_row_vjp_call_count"] > 0
    assert accounting["maximum_native_length_bar_tensor_bytes"] > 0
    assert accounting["maximum_geometry_bridge_visible_tensor_bytes"] > 0
    assert accounting[
        "staged_sparse_geometry_bridge_visible_peak_logical_tensor_bytes_upper_bound"
    ] >= accounting["maximum_geometry_bridge_visible_tensor_bytes"]
    assert accounting["staged_sparse_geometry_bridge_included_in_main_active_peak"]
    assert accounting["maximum_request_geometry_bar_tensor_bytes"] > 0
    assert accounting["geometry_bar_tensor_bytes"] == sum(
        tensor.numel() * 8 for tensor in world_before
    )
    assert accounting["step_accumulator_logical_tensor_bytes"] == (
        accounting[
            "fixed_camera_full_geometry_step_accumulator_preflight_logical_tensor_bytes"
        ]
    )
    assert accounting["maximum_request_delta_ray_bar_key_logical_bytes"] == 0
    assert accounting["peak_ray_payload_logical_tensor_bytes"] == 0
    assert accounting["geometry_bar_memory_receipt_kind"] == "logical_tensor_bytes"
    assert not accounting["geometry_bar_allocator_peak_measured"]
    assert accounting["fixed_camera_full_geometry_step_coordinator_integrated"]
    assert not accounting["production_trainer_integrated"]
    assert not accounting["geometry_update_executed"]
    assert not accounting["fresh_world_recompile_executed"]
    assert not accounting["stale_structure_reuse_prevention_integrated"]
    assert fence.calls == accounting["total_step_completion_fence_call_count"]
    assert result.authorization.request_count == 6
    assert result.authorization.observation_count == 12
    assert factory.compile_count == 12
    assert len(target_source.calls) == 6
    assert store.report().current_entry_count == 6
    assert state.authorized_step_count == 1
    assert not state.poisoned


def test_fixed_camera_step_can_authorize_fused_direct_v1_without_jw(
    monkeypatch,
) -> None:
    (
        _target_source,
        _factory,
        provider,
        _store,
        state,
        material,
        background,
    ) = _case()
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
    fence = _Fence()
    result = _run(
        state,
        provider,
        material,
        background,
        policy=_policy(
            full_geometry_reverse_mode=FUSED_DIRECT_V1_FULL_GEOMETRY_REVERSE
        ),
        fence=fence,
    )

    result.assert_current()
    accounting = result.accounting
    request_count = accounting["exact_request_count"]
    assert accounting["full_geometry_reverse_mode"] == (
        FUSED_DIRECT_V1_FULL_GEOMETRY_REVERSE
    )
    assert accounting["native_full_geometry_vjp_launch_count"] == 0
    assert accounting["native_full_geometry_fenced_reduction_count"] == 0
    assert accounting["geometry_reduction_fence_call_count"] == 0
    assert accounting["geometry_completion_receipt_count"] == 0
    assert accounting["maximum_native_length_bar_tensor_bytes"] == 0
    assert accounting["maximum_simultaneous_geometry_jw_length_bar_tensors"] == 0
    assert accounting["native_fused_full_geometry_vjp_launch_count"] == (
        accounting["active_native_block_count"]
    )
    assert accounting["native_fused_full_geometry_transaction_count"] == (
        request_count
    )
    assert accounting[
        "native_fused_full_geometry_completion_fence_count"
    ] == request_count
    assert accounting["fused_transaction_fence_call_count"] == request_count
    assert accounting["fused_post_accept_commit_fence_call_count"] == request_count
    assert accounting["fused_active_manifest_coverage_certified"]
    assert accounting[
        "maximum_request_fused_prepared_owned_logical_tensor_bytes"
    ] > 0
    assert accounting[
        "maximum_request_fused_output_scratch_logical_tensor_bytes"
    ] > 0
    assert accounting[
        "maximum_request_fused_geometry_bridge_visible_tensor_bytes"
    ] > 0
    assert fence.calls == accounting["total_step_completion_fence_call_count"]


def test_geometry_bar_budget_fails_before_compile_decode_or_accumulator_allocation(
    monkeypatch,
) -> None:
    target_source, factory, provider, _store, state, material, background = _case()
    prepare_call_count = 0
    original_prepare = step_module.prepare_paper_kinetic_dense_step_gradient_accumulator

    def count_prepare(*args, **kwargs):
        nonlocal prepare_call_count
        prepare_call_count += 1
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(
        step_module,
        "prepare_paper_kinetic_dense_step_gradient_accumulator",
        count_prepare,
    )
    policy = replace(
        _policy(),
        maximum_geometry_bar_logical_tensor_bytes=1,
    )
    with pytest.raises(MemoryError, match="geometry bars exceed"):
        _run(
            state,
            provider,
            material,
            background,
            policy=policy,
        )

    assert prepare_call_count == 0
    assert factory.compile_count == 0
    assert target_source.calls == []
    assert not state.poisoned
    assert state.authorized_step_count == 0
    state.assert_current(provider)


def test_geometry_generation_is_bound_to_live_provider_before_work() -> None:
    target_source, factory, provider, _store, state, material, background = _case()
    foreign_generation = replace(
        _generation(provider),
        geometry_generation_id="0" * 64,
    )

    with pytest.raises(ValueError, match="foreign to the live provider world"):
        _run(
            state,
            provider,
            material,
            background,
            generation_policy=foreign_generation,
        )

    assert factory.compile_count == 0
    assert target_source.calls == []
    assert state.authorized_step_count == 0
    state.assert_current(provider)


def test_sealed_result_fails_closed_if_any_ray_payload_is_claimed() -> None:
    _target_source, _factory, provider, _store, state, material, background = _case()
    result = _run(state, provider, material, background)
    forged_accounting = MappingProxyType(
        {
            **result.accounting,
            "camera_ray_gradients_enabled": True,
            "maximum_request_delta_ray_bar_key_logical_bytes": 16,
        }
    )
    provisional = replace(
        result,
        accounting=forged_accounting,
        generation_digest="",
    )
    forged = replace(
        provisional,
        generation_digest=step_module._result_digest(provisional),
    )

    with pytest.raises(ValueError, match="retained ray bars"):
        forged.assert_current()


def test_sealed_result_rejects_forged_geometry_binding() -> None:
    _target_source, _factory, provider, _store, state, material, background = _case()
    result = _run(state, provider, material, background)
    forged_accounting = MappingProxyType(
        {
            **result.accounting,
            "geometry_generation_id": "0" * 64,
        }
    )
    provisional = replace(
        result,
        accounting=forged_accounting,
        generation_digest="",
    )
    forged = replace(
        provisional,
        generation_digest=step_module._result_digest(provisional),
    )

    with pytest.raises(ValueError, match="step result changed"):
        forged.assert_current()


def test_invalid_full_geometry_request_receipt_poison_stops_the_step(
    monkeypatch,
) -> None:
    _target_source, _factory, provider, _store, state, material, background = _case()
    original_run = step_module.run_paper_kinetic_dense_cached_native_request

    def forge_ray_accounting(*args, **kwargs):
        result = original_run(*args, **kwargs)
        # The coordinator must not trust a caller-visible request object merely
        # because it was returned by the expected function.  This deliberately
        # breaks the fixed-camera relation before request-delta commit.
        return replace(
            result,
            accounting={
                **result.accounting,
                "camera_ray_gradients_enabled": True,
            },
        )

    monkeypatch.setattr(
        step_module,
        "run_paper_kinetic_dense_cached_native_request",
        forge_ray_accounting,
    )
    with pytest.raises(
        PaperKineticFixedCameraFullGeometryStepPartialFailure,
        match="process restart is required",
    ):
        _run(state, provider, material, background)

    assert state.poisoned
    assert state.restart_required
    assert state.failure_fail_stop_completed
    assert state.failed_accumulator is not None
    assert state.failed_accumulator.poisoned
    assert not state.failed_accumulator.optimizer_authorized
    assert state.authorized_step_count == 0
    state.assert_current(provider)
