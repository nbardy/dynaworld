from __future__ import annotations

import gc
import inspect
import weakref
from dataclasses import dataclass, replace

import kinetic_lazy_native_material_step as step_module
import paper_kinetic_lazy_program_bundles as lazy_bundle_module
import paper_kinetic_sparse_sample_blocks as sparse_sample_module
import paper_kinetic_union_local_bar_assembly as union_assembly_module
import pytest
import tests.test_paper_kinetic_lazy_program_bundles as lazy_fixture
import torch
from kinetic_lazy_native_material_step import (
    TARGET_FRAME_STEP_CACHE,
    TARGET_FRAME_STREAM_ONCE,
    PaperKineticLazyNativeMemoryPolicy,
    paper_kinetic_observation_manifest_digest,
    prepare_paper_kinetic_lazy_native_trainer_state,
    run_paper_kinetic_lazy_native_material_step,
)
from kinetic_sealed_completion_fence import (
    PaperKineticCompletionUnknownError,
    PaperKineticSealedCompletionFence,
)
from kinetic_multichart_transfer_program import (
    compile_kinetic_multichart_p0_program,
)
from kinetic_native_lazy_bundle_lane import (
    prepare_paper_kinetic_native_lazy_bundle_lane,
)
from kinetic_native_material_step_executor import (
    KineticNativeMaterialStepSession,
    KineticNativePendingSampleLaunchCompletion,
)
from kinetic_owner_chart_compiler import compile_exact_kinetic_owner_charts
from paper_kinetic_lazy_program_bundles import (
    PaperKineticLazyProgramBundleProvider,
    PaperKineticObservation,
    PaperKineticTrackProgramRequest,
    prepare_paper_kinetic_lazy_program_bundle_provider,
)
from paper_kinetic_sparse_sample_blocks import (
    PaperKineticSparseSampleBlockStream,
    iter_paper_kinetic_sparse_sample_blocks,
    prepare_paper_kinetic_sparse_sample_plan,
)
from powerfoam_training_data import PowerFoamRayProvider, PowerFoamTargetProvider
from test_kinetic_ragged_paper_step_cpu_fake_native import (
    _FakeNativeOps,
    _node_charts,
    _phi,
)

_BACKEND = "cpu-contract-double/exact-production-op-object"


class _NonRetainingExactStaticRayProgramFactory:
    """Exact fixture compiler whose only mutable state is a scalar counter."""

    provenance = "nonretaining-exact-static-ray-program-factory-v1"

    def __init__(self) -> None:
        self.generation_digest = lazy_fixture._sha256(self.provenance)
        self.compile_count = 0

    def compile_track(self, request: PaperKineticTrackProgramRequest):
        request.assert_self_consistent()
        self.compile_count += 1
        ray = request.observations[0].ray_origin_direction
        coefficients = torch.tensor(
            [*ray[0:3], 0.0, 0.0, 0.0, *ray[3:6], 0.0, 0.0, 0.0],
            dtype=torch.float64,
        )
        owner_program = compile_exact_kinetic_owner_charts(
            request.world.sites,
            coefficients,
            t_min=request.frame_times[0],
            t_max=request.frame_times[-1],
            near=0.0,
            far=2.0,
        )
        assert owner_program.passed
        return compile_kinetic_multichart_p0_program(
            owner_program,
            request.world.sites,
            coefficients,
            node_count=2,
        )

    def memory_light_residency(self) -> dict[str, int | bool]:
        return {
            "retained_compile_request_count": 0,
            "retained_compiled_program_count": 0,
            "retained_observation_record_count": 0,
            "retained_tensor_bytes": 0,
            "unbounded_cache_enabled": False,
        }


class _BecomesRetainingFactory(_NonRetainingExactStaticRayProgramFactory):
    """Starts admissible, then proves the coordinator rechecks after use."""

    provenance = "becomes-retaining-static-ray-program-factory-v1"

    def __init__(self) -> None:
        super().__init__()
        self.generation_digest = lazy_fixture._sha256(self.provenance)
        self.retained_program = None

    def compile_track(self, request: PaperKineticTrackProgramRequest):
        program = super().compile_track(request)
        self.retained_program = program
        return program

    def memory_light_residency(self) -> dict[str, int | bool]:
        report = super().memory_light_residency()
        report["retained_compiled_program_count"] = int(self.retained_program is not None)
        return report


@dataclass
class _OptimizerCapture:
    calls: int = 0
    loss_f32: torch.Tensor | None = None
    gradient_f32: torch.Tensor | None = None

    def __call__(self, result) -> None:
        result.assert_current()
        self.calls += 1
        self.loss_f32 = result.loss_f32.clone()
        self.gradient_f32 = result.grad_global_site_rgba_f32.clone()


def _fail_owned_fence_at_stage(
    monkeypatch,
    *,
    stage: str,
    occurrence: int = 1,
) -> dict[str, object]:
    """Inject failure only inside the class-owned synchronizer.

    The production step never accepts this callable.  Patching the sealed
    capability's owned synchronization boundary is the narrow adversarial way
    to exercise completion-unknown quarantine without reopening callback
    injection in the lazy API.
    """

    original = PaperKineticSealedCompletionFence._synchronize_bound_device_wide
    audit: dict[str, object] = {"calls": 0, "matching_calls": 0, "capability": None}

    def synchronize_or_fail(capability) -> None:
        audit["calls"] = int(audit["calls"]) + 1
        epoch = capability.registered_launch_epoch
        assert epoch is not None
        audit["capability"] = capability
        if epoch.stage == stage:
            audit["matching_calls"] = int(audit["matching_calls"]) + 1
            if audit["matching_calls"] == occurrence:
                raise RuntimeError(f"synthetic owned {stage} fence failure")
        original(capability)

    monkeypatch.setattr(
        PaperKineticSealedCompletionFence,
        "_synchronize_bound_device_wide",
        synchronize_or_fail,
    )
    return audit


class _InterruptingNativeOps(_FakeNativeOps):
    def kinetic_ragged_p0_lie_sample_accumulate_loss_only_launch_only(
        self,
        prepared,
        loss_f32,
        grad_node_chart_f32,
        cone_diagnostic_i32,
    ):
        super().kinetic_ragged_p0_lie_sample_accumulate_loss_only_launch_only(
            prepared,
            loss_f32,
            grad_node_chart_f32,
            cone_diagnostic_i32,
        )
        raise KeyboardInterrupt("synthetic native interruption")


class _InterruptingForwardNativeOps(_FakeNativeOps):
    def kinetic_precompiled_length_p0_lie_node_forward_into_launch_only_v1(
        self,
        *args,
        **kwargs,
    ):
        super().kinetic_precompiled_length_p0_lie_node_forward_into_launch_only_v1(
            *args,
            **kwargs,
        )
        raise KeyboardInterrupt("synthetic forward enqueue interruption")


@dataclass
class _InterruptingOptimizer:
    calls: int = 0

    def __call__(self, result) -> None:
        result.assert_current()
        self.calls += 1
        raise KeyboardInterrupt("synthetic optimizer interruption")


def _provider(
    *,
    maximum_tracks_per_bundle: int,
    maximum_observations_per_bundle: int,
    factory: _NonRetainingExactStaticRayProgramFactory | None = None,
):
    source = lazy_fixture.LazyTargetSource()
    target_provider = PowerFoamTargetProvider(source=source, device="cpu")
    ray_provider = PowerFoamRayProvider(
        cameras=lazy_fixture._camera_grid(),
        height=source.height,
        width=source.width,
        device="cpu",
    )
    selected_factory = factory or _NonRetainingExactStaticRayProgramFactory()
    provider = prepare_paper_kinetic_lazy_program_bundle_provider(
        dataset_generation_digest=lazy_fixture._sha256("lazy-native-material-step-test-dataset"),
        target_provider=target_provider,
        ray_provider=ray_provider,
        frame_times=(0.0, 0.4, 1.0),
        height=source.height,
        width=source.width,
        maximum_tracks_per_bundle=maximum_tracks_per_bundle,
        maximum_observations_per_bundle=maximum_observations_per_bundle,
        maximum_rows_per_native_block=1,
        world_initializer=lazy_fixture.OneSiteWorldInitializer(),
        program_factory=selected_factory,
    )
    return source, selected_factory, provider


def _observations(
    view_frame_pixel: tuple[tuple[int, int, int], ...],
) -> tuple[PaperKineticObservation, ...]:
    canonical = sorted(
        view_frame_pixel,
        key=lambda item: (item[0], item[2], item[1]),
    )
    return tuple(
        PaperKineticObservation(
            observation_id=index,
            view_index=view,
            frame_index=frame,
            pixel_index=pixel,
        )
        for index, (view, frame, pixel) in enumerate(canonical)
    )


def _material() -> torch.Tensor:
    return torch.tensor([[0.18, 0.31, 0.47, 0.73]], dtype=torch.float32)


def _background() -> torch.Tensor:
    return torch.tensor([0.03, 0.07, 0.11], dtype=torch.float32)


def _memory_policy(
    provider,
    *,
    target_frame_access_mode: str = TARGET_FRAME_STREAM_ONCE,
    cache_frame_capacity: int | None = None,
) -> PaperKineticLazyNativeMemoryPolicy:
    frame_bytes = provider.height * provider.width * 3 * 4
    if target_frame_access_mode == TARGET_FRAME_STEP_CACHE:
        cache_bytes = (
            provider.view_count * provider.frame_count * frame_bytes
            if cache_frame_capacity is None
            else cache_frame_capacity * frame_bytes
        )
    else:
        cache_bytes = 0
    return PaperKineticLazyNativeMemoryPolicy(
        max_global_material_and_bar_tensor_bytes=1_000_000,
        max_bundle_observation_count=provider.maximum_observations_per_bundle,
        max_lane_resident_logical_tensor_bytes=10_000_000,
        max_active_node_and_vjp_tensor_bytes=10_000_000,
        max_decoded_frame_scratch_tensor_bytes=10_000_000,
        max_selected_frame_target_tensor_bytes=10_000_000,
        max_sample_launch_tensor_bytes=10_000_000,
        max_coordinator_visible_live_tensor_bytes=100_000_000,
        target_frame_access_mode=target_frame_access_mode,
        max_step_target_frame_cache_tensor_bytes=cache_bytes,
    )


def _direct_oracle(
    provider,
    observations: tuple[PaperKineticObservation, ...],
    *,
    global_site_rgba_f32: torch.Tensor,
    background_rgb_f32: torch.Tensor,
    maximum_samples_per_launch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rgba = global_site_rgba_f32.detach().clone().requires_grad_(True)
    loss = torch.zeros((), dtype=torch.float32)
    for bundle in provider.iter_canonical_spatial_bundles(
        observations,
        device="cpu",
    ):
        lane = prepare_paper_kinetic_native_lazy_bundle_lane(
            bundle,
            provider,
            _FakeNativeOps(),
            device="cpu",
            backend_provenance=_BACKEND,
        )
        nodes_by_digest = {
            runtime.payload.block.generation_digest: _node_charts(
                runtime.payload.word_offsets_i32,
                runtime.payload.word_owner_i32,
                runtime.payload.node_physical_length_f32,
                rgba.index_select(0, runtime.source_site_ids_i64),
            )
            for runtime in lane.runtimes
        }
        plan = prepare_paper_kinetic_sparse_sample_plan(
            bundle,
            provider,
            global_loss_element_count=len(observations) * 3,
            loss_normalization_id="direct-oracle",
            maximum_samples_per_launch=maximum_samples_per_launch,
        )
        for sample_block in iter_paper_kinetic_sparse_sample_blocks(plan):
            rows = sample_block.sample_row_i32.to(dtype=torch.int64)
            selected = nodes_by_digest[sample_block.native_block_generation_digest].index_select(0, rows)
            chart = torch.sum(
                selected * sample_block.sample_to_node_f32[:, :, None],
                dim=1,
            )
            kappa = chart[:, 0]
            prediction = _phi(kappa)[:, None] * chart[:, 1:] + torch.exp(-kappa)[:, None] * background_rgb_f32
            loss = loss + ((prediction - sample_block.target_rgb_f32).square().sum() * sample_block.loss_scale)
    loss.backward()
    assert rgba.grad is not None
    return loss.detach().reshape(1), rgba.grad.detach()


def _lane_statistics(
    provider,
    observations: tuple[PaperKineticObservation, ...],
) -> tuple[tuple[int, ...], int]:
    lane_bytes = []
    native_block_count = 0
    for bundle in provider.iter_canonical_spatial_bundles(
        observations,
        device="cpu",
    ):
        lane = prepare_paper_kinetic_native_lazy_bundle_lane(
            bundle,
            provider,
            _FakeNativeOps(),
            device="cpu",
            backend_provenance=_BACKEND,
        )
        lane_bytes.append(lane.resident_logical_tensor_bytes)
        native_block_count += lane.native_runtime_count
    return tuple(lane_bytes), native_block_count


def _run(
    state,
    provider,
    observations,
    *,
    step_index: int,
    native_ops: _FakeNativeOps,
    capture: _OptimizerCapture,
    material: torch.Tensor,
    gradient: torch.Tensor,
    maximum_samples_per_launch: int,
    expected_observation_count: int | None = None,
    expected_observation_manifest_digest: str | None = None,
    memory_policy: PaperKineticLazyNativeMemoryPolicy | None = None,
    background: torch.Tensor | None = None,
):
    return run_paper_kinetic_lazy_native_material_step(
        state,
        provider,
        observations,
        step_index=step_index,
        expected_observation_count=(
            len(observations) if expected_observation_count is None else expected_observation_count
        ),
        expected_observation_manifest_digest=(
            paper_kinetic_observation_manifest_digest(observations)
            if expected_observation_manifest_digest is None
            else expected_observation_manifest_digest
        ),
        loss_normalization_id=f"lazy-native-step-{step_index}",
        material_generation_id=f"material-generation-{step_index}",
        background_generation_id=f"background-generation-{step_index}",
        global_site_rgba_f32=material,
        global_grad_site_rgba_f32=gradient,
        background_rgb_f32=_background() if background is None else background,
        native_ops=native_ops,
        maximum_samples_per_launch=maximum_samples_per_launch,
        memory_policy=(_memory_policy(provider) if memory_policy is None else memory_policy),
        optimizer_update=capture,
    )


def test_exact_oracle_multi_bundle_peak_once_per_block_and_step_reuse_guard() -> None:
    source, factory, provider = _provider(
        maximum_tracks_per_bundle=2,
        maximum_observations_per_bundle=6,
    )
    observations = _observations(
        (
            (0, 0, 0),
            (0, 2, 0),
            (0, 1, 1),
            (0, 2, 2),
            (1, 1, 3),
            (1, 2, 4),
        )
    )
    material = _material()
    oracle_loss, oracle_bar = _direct_oracle(
        provider,
        observations,
        global_site_rgba_f32=material,
        background_rgb_f32=_background(),
        maximum_samples_per_launch=2,
    )
    source.calls.clear()
    lane_bytes, eligible_block_count = _lane_statistics(provider, observations)
    assert len(lane_bytes) > 1

    state = prepare_paper_kinetic_lazy_native_trainer_state(
        provider,
        device="cpu",
    )
    native_ops = _FakeNativeOps()
    capture = _OptimizerCapture()
    gradient = torch.full_like(material, 91.0)
    result = _run(
        state,
        provider,
        observations,
        step_index=0,
        native_ops=native_ops,
        capture=capture,
        material=material,
        gradient=gradient,
        maximum_samples_per_launch=2,
        memory_policy=_memory_policy(
            provider,
            target_frame_access_mode=TARGET_FRAME_STEP_CACHE,
        ),
    )

    torch.testing.assert_close(result.loss_f32, oracle_loss, rtol=2.0e-5, atol=2.0e-6)
    torch.testing.assert_close(gradient, oracle_bar, rtol=3.0e-5, atol=3.0e-6)
    torch.testing.assert_close(capture.loss_f32, oracle_loss, rtol=2.0e-5, atol=2.0e-6)
    torch.testing.assert_close(capture.gradient_f32, oracle_bar, rtol=3.0e-5, atol=3.0e-6)

    accounting = result.accounting
    active = accounting["active_native_block_count"]
    assert accounting["spatial_bundle_count"] == len(lane_bytes)
    assert accounting["eligible_native_block_count"] == eligible_block_count
    assert accounting["peak_lane_resident_logical_tensor_bytes"] == max(lane_bytes)
    assert accounting["peak_lane_resident_logical_tensor_bytes"] < sum(lane_bytes)
    assert native_ops.forward_calls == active
    assert native_ops.material_vjp_calls == active
    assert native_ops.vjp_calls == 0
    assert native_ops.sample_prepare_calls == accounting["native_sample_prepare_count"]
    assert native_ops.sample_launch_calls == accounting["native_sample_launch_count"]
    assert accounting["native_node_forward_launch_count"] == active
    assert accounting["native_material_word_vjp_launch_count"] == active
    assert accounting["caller_global_material_bar_count"] == 1
    assert accounting["optimizer_update_authorization_count"] == 1
    unique_frames = {(observation.view_index, observation.frame_index) for observation in observations}
    assert accounting["target_frame_access_mode"] == TARGET_FRAME_STEP_CACHE
    assert accounting["target_frame_cache_enabled"] is True
    assert accounting["target_frame_decode_count"] == len(unique_frames)
    assert accounting["target_frame_cache_hit_count"] > 0
    frame_bytes = provider.height * provider.width * 3 * 4
    assert accounting["target_frame_cache_peak_resident_tensor_bytes"] == (len(unique_frames) * frame_bytes)
    assert accounting["target_frame_cache_maximum_resident_tensor_bytes"] == (
        provider.view_count * provider.frame_count * frame_bytes
    )
    assert "O(unique_selected_frames * H * W" in accounting["decode_once_cache_target_residency_bound"]
    assert accounting["target_frame_cache_no_eviction"] is True
    assert accounting["target_frame_cache_closed_before_optimizer"] is True
    assert accounting["target_frame_cache_resident_tensor_bytes_after_close"] == 0
    assert accounting["selected_pixel_read_mode"] == "full_frame_cache"
    assert accounting["selected_pixel_read_acceptance_capable"] is False
    assert accounting["direct_selected_pixel_observation_count"] == 0
    assert accounting["full_frame_fallback_observation_count"] == len(
        observations
    )
    assert accounting["full_frame_target_materialization_count"] == len(
        unique_frames
    )
    assert accounting["sample_completion_fence_call_count"] == accounting[
        "native_sample_launch_count"
    ]
    assert accounting["reverse_completion_fence_call_count"] == accounting[
        "native_material_word_vjp_launch_count"
    ]
    assert accounting["lane_release_fence_call_count"] == 0
    assert accounting["lane_release_completion_boundary_count"] == accounting[
        "spatial_bundle_count"
    ]
    assert accounting["lane_release_reuses_final_reverse_receipt"] is True
    assert accounting["bundle_materialization_completion_fence_call_count"] == (
        accounting["spatial_bundle_count"]
    )
    assert accounting["bundle_exhaustion_probe_completion_fence_call_count"] == 1
    assert accounting["sealed_completion_fence_success_count"] == accounting[
        "device_completion_fence_call_count"
    ]
    assert accounting["sealed_completion_receipt_consumption_count"] == accounting[
        "sealed_completion_fence_success_count"
    ]
    assert accounting["sealed_completion_outstanding_receipt_count"] == 0
    assert accounting["caller_supplied_completion_callback_count"] == 0
    assert accounting["caller_supplied_completion_provenance_count"] == 0
    assert accounting[
        "bounded_trainer_owned_async_failure_quarantine_implemented"
    ] is True
    assert accounting[
        "async_failure_quarantine_is_single_carrier_not_history"
    ] is True
    assert accounting[
        "optimizer_authorized_only_after_all_completion_fences"
    ] is True
    assert accounting[
        "partial_lane_construction_provisional_lease_implemented"
    ] is True
    assert accounting[
        "sparse_sample_transfer_predecessor_lease_implemented"
    ] is True
    assert accounting[
        "pre_lane_bundle_device_construction_lease_implemented"
    ] is True
    assert accounting[
        "bundle_transfer_predecessors_released_after_proven_fence"
    ] is True
    assert accounting[
        "forward_compact_gather_provisional_lease_implemented"
    ] is True
    assert accounting["top_level_device_zero_transaction_lifetime_implemented"] is True
    assert accounting[
        "caller_owned_native_forward_output_lifetime_implemented"
    ] is True
    assert accounting[
        "native_forward_into_binding_shader_source_implemented"
    ] is True
    assert accounting["native_forward_into_compiled_registration_verified"] is False
    assert accounting["native_forward_internal_output_lifetime_verified"] is False
    assert "forward-into rebuild/registration/parity" in accounting[
        "accelerator_gate_blocker"
    ]
    assert state._active_device_transaction_lifetime is None
    assert capture.calls == state.optimizer_callback_count == 1
    assert factory.memory_light_residency() == {
        "retained_compile_request_count": 0,
        "retained_compiled_program_count": 0,
        "retained_observation_record_count": 0,
        "retained_tensor_bytes": 0,
        "unbounded_cache_enabled": False,
    }

    calls_before_reopen = (
        native_ops.forward_calls,
        native_ops.material_vjp_calls,
        native_ops.sample_launch_calls,
        capture.calls,
    )
    with pytest.raises(ValueError, match="stale, skipped, or reused"):
        _run(
            state,
            provider,
            observations,
            step_index=0,
            native_ops=native_ops,
            capture=capture,
            material=material,
            gradient=gradient,
            maximum_samples_per_launch=2,
        )
    assert calls_before_reopen == (
        native_ops.forward_calls,
        native_ops.material_vjp_calls,
        native_ops.sample_launch_calls,
        capture.calls,
    )
    assert state.next_step_index == 1
    assert state.last_completed_step_index == 0


def test_trainer_state_binds_promoted_absolute_step_index() -> None:
    _source, _factory, provider = _provider(
        maximum_tracks_per_bundle=1,
        maximum_observations_per_bundle=1,
    )
    state = prepare_paper_kinetic_lazy_native_trainer_state(
        provider,
        device="cpu",
        initial_step_index=3,
    )
    assert state.next_step_index == 3
    assert state.last_completed_step_index == 2
    assert state.optimizer_callback_count == 3
    state.assert_current(provider)
    with pytest.raises(ValueError, match="initial_step_index"):
        prepare_paper_kinetic_lazy_native_trainer_state(
            provider,
            device="cpu",
            initial_step_index=-1,
        )


def test_more_selected_frames_only_grows_sparse_sample_and_decode_work() -> None:
    source, factory, provider = _provider(
        maximum_tracks_per_bundle=2,
        maximum_observations_per_bundle=6,
    )
    sparse = _observations(((0, 0, 0), (0, 0, 1)))
    denser = _observations(tuple((0, frame, pixel) for pixel in (0, 1) for frame in (0, 1, 2)))
    state = prepare_paper_kinetic_lazy_native_trainer_state(
        provider,
        device="cpu",
    )
    material = _material()
    gradient = torch.empty_like(material)
    native_ops = _FakeNativeOps()
    capture = _OptimizerCapture()

    compile_before = factory.compile_count
    sparse_result = _run(
        state,
        provider,
        sparse,
        step_index=0,
        native_ops=native_ops,
        capture=capture,
        material=material,
        gradient=gradient,
        maximum_samples_per_launch=1,
    )
    sparse_compile_work = factory.compile_count - compile_before
    sparse_decode_work = len(source.calls)
    sparse_forward_work = native_ops.forward_calls
    sparse_vjp_work = native_ops.material_vjp_calls
    assert capture.calls == 1

    compile_before = factory.compile_count
    decode_before = len(source.calls)
    forward_before = native_ops.forward_calls
    vjp_before = native_ops.material_vjp_calls
    sample_before = native_ops.sample_launch_calls
    denser_result = _run(
        state,
        provider,
        denser,
        step_index=1,
        native_ops=native_ops,
        capture=capture,
        material=material,
        gradient=gradient,
        maximum_samples_per_launch=1,
    )
    denser_compile_work = factory.compile_count - compile_before
    denser_decode_work = len(source.calls) - decode_before
    denser_forward_work = native_ops.forward_calls - forward_before
    denser_vjp_work = native_ops.material_vjp_calls - vjp_before
    denser_sample_work = native_ops.sample_launch_calls - sample_before

    sparse_accounting = sparse_result.accounting
    denser_accounting = denser_result.accounting
    for key in (
        "spatial_bundle_count",
        "eligible_native_block_count",
        "active_native_block_count",
        "native_node_forward_launch_count",
        "native_material_word_vjp_launch_count",
        "ordered_word_node_interactions",
        "peak_lane_resident_logical_tensor_bytes",
        "peak_active_node_state_tensor_bytes",
        "peak_sample_launch_tensor_bytes",
        "peak_decoded_frame_scratch_upper_bound_bytes",
        "peak_selected_frame_target_tensor_upper_bound_bytes",
    ):
        assert sparse_accounting[key] == denser_accounting[key]
    assert sparse_compile_work == denser_compile_work == 2
    assert sparse_forward_work == denser_forward_work
    assert sparse_vjp_work == denser_vjp_work
    assert sparse_decode_work == 1
    assert denser_decode_work == 3
    assert sparse_accounting["streamed_sample_count"] == len(sparse)
    assert denser_accounting["streamed_sample_count"] == len(denser)
    assert denser_accounting["native_sample_launch_count"] > sparse_accounting["native_sample_launch_count"]
    assert denser_sample_work == len(denser)
    assert (
        denser_accounting["sample_to_node_linear_interactions"]
        > (sparse_accounting["sample_to_node_linear_interactions"])
    )
    assert denser_accounting["persistent_frame_tensor_bytes"] == 0
    assert denser_accounting["persistent_target_tensor_bytes"] == 0
    assert denser_accounting["persistent_prediction_tensor_bytes"] == 0
    assert denser_accounting["target_frame_access_mode"] == TARGET_FRAME_STREAM_ONCE
    assert denser_accounting["target_frame_cache_enabled"] is False
    assert denser_accounting["target_frame_cache_peak_resident_tensor_bytes"] == 0
    assert denser_accounting["target_frame_decode_count"] == 0
    assert denser_accounting["selected_pixel_read_call_count"] == denser_decode_work
    assert denser_accounting["selected_pixel_read_mode"] == "direct_pixels"
    assert denser_accounting["selected_pixel_read_acceptance_capable"] is True
    assert denser_accounting["direct_selected_pixel_observation_count"] == len(
        denser
    )
    assert denser_accounting["full_frame_fallback_observation_count"] == 0
    assert denser_accounting["full_frame_target_materialization_count"] == 0
    assert denser_accounting["sparse_sampled_observations_only"] is True
    assert denser_accounting["dense_f_replayable_observation_source_implemented"] is False
    assert capture.calls == state.optimizer_callback_count == 2


def test_pre_authorization_coverage_failure_zeros_global_bar_and_can_retry() -> None:
    _source, _factory, provider = _provider(
        maximum_tracks_per_bundle=2,
        maximum_observations_per_bundle=6,
    )
    observations = _observations(((0, 0, 0), (0, 2, 0), (0, 1, 1)))
    state = prepare_paper_kinetic_lazy_native_trainer_state(
        provider,
        device="cpu",
    )
    material = _material()
    gradient = torch.full_like(material, 37.0)
    native_ops = _FakeNativeOps()
    capture = _OptimizerCapture()

    with pytest.raises(ValueError, match="manifest differs"):
        _run(
            state,
            provider,
            observations,
            step_index=0,
            native_ops=native_ops,
            capture=capture,
            material=material,
            gradient=gradient,
            maximum_samples_per_launch=2,
            expected_observation_manifest_digest="0" * 64,
        )
    assert torch.count_nonzero(gradient) == 0
    assert capture.calls == 0
    assert state.active_step_index is None
    assert state.next_step_index == 0
    assert not state.poisoned

    gradient.fill_(37.0)
    with pytest.raises(ValueError, match="declared global observations"):
        _run(
            state,
            provider,
            observations,
            step_index=0,
            native_ops=native_ops,
            capture=capture,
            material=material,
            gradient=gradient,
            maximum_samples_per_launch=2,
            expected_observation_count=len(observations) + 1,
        )
    assert torch.count_nonzero(gradient) == 0
    assert capture.calls == 0
    assert state.active_step_index is None
    assert state.next_step_index == 0
    assert state.last_completed_step_index == -1
    assert not state.poisoned

    _run(
        state,
        provider,
        observations,
        step_index=0,
        native_ops=native_ops,
        capture=capture,
        material=material,
        gradient=gradient,
        maximum_samples_per_launch=2,
    )
    assert capture.calls == state.optimizer_callback_count == 1
    assert state.next_step_index == 1


def test_declared_prefix_rejects_a_valid_trailing_bundle_before_optimizer() -> None:
    _source, _factory, provider = _provider(
        maximum_tracks_per_bundle=1,
        maximum_observations_per_bundle=1,
    )
    observations = _observations(((0, 0, 0), (0, 1, 1)))
    declared_prefix = observations[:1]
    state = prepare_paper_kinetic_lazy_native_trainer_state(provider, device="cpu")
    material = _material()
    gradient = torch.full_like(material, 38.0)
    capture = _OptimizerCapture()
    native_ops = _FakeNativeOps()

    with pytest.raises(ValueError, match="undeclared trailing observations"):
        _run(
            state,
            provider,
            observations,
            step_index=0,
            native_ops=native_ops,
            capture=capture,
            material=material,
            gradient=gradient,
            maximum_samples_per_launch=1,
            expected_observation_count=len(declared_prefix),
            expected_observation_manifest_digest=(
                paper_kinetic_observation_manifest_digest(declared_prefix)
            ),
        )

    assert native_ops.sample_launch_calls == len(declared_prefix)
    assert capture.calls == 0
    assert torch.count_nonzero(gradient) == 0
    assert state.active_step_index is None
    assert state.next_step_index == 0
    assert state.last_completed_step_index == -1
    assert state.poisoned is False


def test_factory_that_retains_compiled_program_fails_before_optimizer() -> None:
    factory = _BecomesRetainingFactory()
    _source, _factory, provider = _provider(
        maximum_tracks_per_bundle=2,
        maximum_observations_per_bundle=6,
        factory=factory,
    )
    observations = _observations(((0, 0, 0), (0, 0, 1)))
    state = prepare_paper_kinetic_lazy_native_trainer_state(
        provider,
        device="cpu",
    )
    material = _material()
    gradient = torch.full_like(material, 11.0)
    capture = _OptimizerCapture()

    assert factory.memory_light_residency()["retained_compiled_program_count"] == 0
    with pytest.raises(ValueError, match="retains bundle/sample state"):
        _run(
            state,
            provider,
            observations,
            step_index=0,
            native_ops=_FakeNativeOps(),
            capture=capture,
            material=material,
            gradient=gradient,
            maximum_samples_per_launch=1,
        )
    assert factory.retained_program is not None
    assert factory.memory_light_residency()["retained_compiled_program_count"] == 1
    assert torch.count_nonzero(gradient) == 0
    assert capture.calls == 0
    assert state.next_step_index == 0
    assert not state.poisoned


def test_target_cache_budget_rejects_before_allocation_without_touching_bar() -> None:
    source, _factory, provider = _provider(
        maximum_tracks_per_bundle=1,
        maximum_observations_per_bundle=3,
    )
    observations = _observations(((0, 0, 0),))
    state = prepare_paper_kinetic_lazy_native_trainer_state(provider, device="cpu")
    material = _material()
    gradient = torch.full_like(material, 19.0)
    capture = _OptimizerCapture()
    frame_bytes = provider.height * provider.width * 3 * 4
    undersized = replace(
        _memory_policy(
            provider,
            target_frame_access_mode=TARGET_FRAME_STEP_CACHE,
            cache_frame_capacity=1,
        ),
        max_step_target_frame_cache_tensor_bytes=frame_bytes - 1,
    )

    with pytest.raises(MemoryError, match="cannot admit one frame"):
        _run(
            state,
            provider,
            observations,
            step_index=0,
            native_ops=_FakeNativeOps(),
            capture=capture,
            material=material,
            gradient=gradient,
            maximum_samples_per_launch=1,
            memory_policy=undersized,
        )
    torch.testing.assert_close(gradient, torch.full_like(material, 19.0))
    assert source.calls == []
    assert capture.calls == 0
    assert state.active_step_index is None
    assert state.next_step_index == 0
    assert not state.poisoned

    _run(
        state,
        provider,
        observations,
        step_index=0,
        native_ops=_FakeNativeOps(),
        capture=capture,
        material=material,
        gradient=gradient,
        maximum_samples_per_launch=1,
    )
    assert capture.calls == 1


def test_multi_bundle_cached_step_has_one_outer_cold_provider_certification(
    monkeypatch,
) -> None:
    _source, _factory, provider = _provider(
        maximum_tracks_per_bundle=1,
        maximum_observations_per_bundle=3,
    )
    observations = _observations(((0, 0, 0), (0, 1, 1), (0, 2, 2)))
    state = prepare_paper_kinetic_lazy_native_trainer_state(provider, device="cpu")
    calls = {"cold": 0}
    original_cold = PaperKineticLazyProgramBundleProvider.assert_current

    def tracked_cold(self) -> None:
        calls["cold"] += 1
        original_cold(self)

    monkeypatch.setattr(
        PaperKineticLazyProgramBundleProvider,
        "assert_current",
        tracked_cold,
    )
    result = _run(
        state,
        provider,
        observations,
        step_index=0,
        native_ops=_FakeNativeOps(),
        capture=_OptimizerCapture(),
        material=_material(),
        gradient=torch.empty_like(_material()),
        maximum_samples_per_launch=1,
        memory_policy=_memory_policy(
            provider,
            target_frame_access_mode=TARGET_FRAME_STEP_CACHE,
        ),
    )
    assert result.accounting["spatial_bundle_count"] == 3
    assert calls["cold"] == 1


def test_only_one_bundle_and_sample_payload_is_live_before_next_materialization(
    monkeypatch,
) -> None:
    _source, _factory, provider = _provider(
        maximum_tracks_per_bundle=1,
        maximum_observations_per_bundle=2,
    )
    observations = _observations(
        (
            (0, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (0, 1, 1),
        )
    )
    state = prepare_paper_kinetic_lazy_native_trainer_state(provider, device="cpu")
    bundle_release_checks: list[bool] = []
    sample_release_checks: list[bool] = []
    original_bundles = (
        PaperKineticLazyProgramBundleProvider.iter_canonical_spatial_bundles
    )
    original_sample_next = PaperKineticSparseSampleBlockStream.__next__
    original_sample_close = PaperKineticSparseSampleBlockStream.close
    previous_sample_by_stream: dict[int, weakref.ReferenceType] = {}

    def tracked_bundles(
        self,
        selected,
        *,
        device,
        construction_lifetime_slot=None,
    ):
        inner = original_bundles(
            self,
            selected,
            device=device,
            construction_lifetime_slot=construction_lifetime_slot,
        )
        try:
            current = next(inner)
            while True:
                previous = weakref.ref(current)
                yield current
                del current
                try:
                    current = next(inner)
                except StopIteration:
                    gc.collect()
                    bundle_release_checks.append(previous() is None)
                    break
                gc.collect()
                bundle_release_checks.append(previous() is None)
        finally:
            inner.close()

    def tracked_sample_next(self):
        previous = previous_sample_by_stream.get(id(self))
        if previous is not None:
            gc.collect()
            sample_release_checks.append(previous() is None)
        sample_block = original_sample_next(self)
        previous_sample_by_stream[id(self)] = weakref.ref(sample_block)
        return sample_block

    def tracked_sample_close(self) -> None:
        original_sample_close(self)
        previous = previous_sample_by_stream.pop(id(self), None)
        gc.collect()
        if previous is not None:
            sample_release_checks.append(previous() is None)

    monkeypatch.setattr(
        PaperKineticLazyProgramBundleProvider,
        "iter_canonical_spatial_bundles",
        tracked_bundles,
    )
    monkeypatch.setattr(
        PaperKineticSparseSampleBlockStream,
        "__next__",
        tracked_sample_next,
    )
    monkeypatch.setattr(
        PaperKineticSparseSampleBlockStream,
        "close",
        tracked_sample_close,
    )
    result = _run(
        state,
        provider,
        observations,
        step_index=0,
        native_ops=_FakeNativeOps(),
        capture=_OptimizerCapture(),
        material=_material(),
        gradient=torch.empty_like(_material()),
        maximum_samples_per_launch=1,
    )

    assert result.accounting["spatial_bundle_count"] == 2
    assert bundle_release_checks
    assert sample_release_checks
    assert all(bundle_release_checks)
    assert all(sample_release_checks)


def test_sample_outer_composite_prevalidates_then_consumes_and_commits_once(
    monkeypatch,
) -> None:
    _source, _factory, provider = _provider(
        maximum_tracks_per_bundle=1,
        maximum_observations_per_bundle=2,
    )
    observations = _observations(((0, 0, 0),))
    state = prepare_paper_kinetic_lazy_native_trainer_state(provider, device="cpu")
    events: list[str] = []
    original_stream_validate = (
        PaperKineticSparseSampleBlockStream.assert_active_releasable_after_consumed_receipt
    )
    original_executor_validate = (
        KineticNativeMaterialStepSession.assert_pending_sample_accumulate_releasable
    )
    original_consume = (
        KineticNativePendingSampleLaunchCompletion.consume_sealed_receipt_for_outer_composite
    )
    original_executor_commit = (
        KineticNativeMaterialStepSession.commit_sample_accumulate_after_consumed_sealed_receipt
    )
    original_stream_commit = (
        PaperKineticSparseSampleBlockStream._commit_active_release_after_consumed_receipt
    )
    original_slot_commit = (
        step_module._SampleCompositeSettlementSlot._commit_after_consumed_receipt
    )

    def tracked_stream_validate(
        stream,
        sample_block=None,
        expected_lifetime=None,
    ):
        events.append("stream-validate")
        return original_stream_validate(
            stream,
            sample_block,
            expected_lifetime=expected_lifetime,
        )

    def tracked_executor_validate(
        session,
        pending,
        capability,
        *,
        subject,
    ):
        assert pending.sealed_completion_receipt.consumed is False
        events.append("executor-validate")
        return original_executor_validate(
            session,
            pending,
            capability,
            subject=subject,
        )

    def tracked_consume(
        pending,
        session,
        capability,
        *,
        subject,
        consumer,
    ):
        assert pending.sealed_completion_receipt.consumed is False
        events.append("consume")
        return original_consume(
            pending,
            session,
            capability,
            subject=subject,
            consumer=consumer,
        )

    def tracked_executor_commit(session, commit_plan):
        pending = commit_plan.pending
        assert pending.sealed_completion_receipt.consumed is True
        events.append("executor-commit")
        return original_executor_commit(session, commit_plan)

    def tracked_stream_commit(stream, expected_lifetime=None):
        assert stream.active_transfer_lifetime is expected_lifetime
        events.append("stream-commit")
        return original_stream_commit(
            stream,
            expected_lifetime=expected_lifetime,
        )

    def tracked_slot_commit(slot):
        assert slot.pending_completion.phase == "committed"
        assert slot.transfer_lifetime.released_after_completion_fence
        events.append("slot-commit")
        return original_slot_commit(slot)

    monkeypatch.setattr(
        PaperKineticSparseSampleBlockStream,
        "assert_active_releasable_after_consumed_receipt",
        tracked_stream_validate,
    )
    monkeypatch.setattr(
        KineticNativeMaterialStepSession,
        "assert_pending_sample_accumulate_releasable",
        tracked_executor_validate,
    )
    monkeypatch.setattr(
        KineticNativePendingSampleLaunchCompletion,
        "consume_sealed_receipt_for_outer_composite",
        tracked_consume,
    )
    monkeypatch.setattr(
        KineticNativeMaterialStepSession,
        "commit_sample_accumulate_after_consumed_sealed_receipt",
        tracked_executor_commit,
    )
    monkeypatch.setattr(
        PaperKineticSparseSampleBlockStream,
        "_commit_active_release_after_consumed_receipt",
        tracked_stream_commit,
    )
    monkeypatch.setattr(
        step_module._SampleCompositeSettlementSlot,
        "_commit_after_consumed_receipt",
        tracked_slot_commit,
    )

    material = _material()
    _run(
        state,
        provider,
        observations,
        step_index=0,
        native_ops=_FakeNativeOps(),
        capture=_OptimizerCapture(),
        material=material,
        gradient=torch.empty_like(material),
        maximum_samples_per_launch=1,
    )

    assert events == [
        "stream-validate",
        "stream-validate",
        "executor-validate",
        "consume",
        "executor-commit",
        "stream-commit",
        "slot-commit",
    ]


def test_post_fence_outer_revalidation_failure_retains_both_sample_owners(
    monkeypatch,
) -> None:
    _source, _factory, provider = _provider(
        maximum_tracks_per_bundle=1,
        maximum_observations_per_bundle=2,
    )
    observations = _observations(((0, 0, 0),))
    state = prepare_paper_kinetic_lazy_native_trainer_state(provider, device="cpu")
    material = _material()
    gradient = torch.full_like(material, 31.0)
    capture = _OptimizerCapture()
    captured: dict[str, object] = {}
    original_validate = (
        KineticNativeMaterialStepSession.assert_pending_sample_accumulate_releasable
    )

    def reject_after_fence(
        session,
        pending,
        capability,
        *,
        subject,
    ):
        original_validate(
            session,
            pending,
            capability,
            subject=subject,
        )
        captured.update(
            session=session,
            pending=pending,
            capability=capability,
            subject=subject,
        )
        raise RuntimeError("injected post-fence outer revalidation failure")

    monkeypatch.setattr(
        KineticNativeMaterialStepSession,
        "assert_pending_sample_accumulate_releasable",
        reject_after_fence,
    )

    with pytest.raises(RuntimeError, match="post-fence outer revalidation"):
        _run(
            state,
            provider,
            observations,
            step_index=0,
            native_ops=_FakeNativeOps(),
            capture=capture,
            material=material,
            gradient=gradient,
            maximum_samples_per_launch=1,
        )

    quarantine = state._async_failure_quarantine
    assert quarantine is not None
    quarantine.assert_current()
    retained = dict(
        zip(
            quarantine.retained_reference_roles,
            quarantine.retained_references,
            strict=True,
        )
    )
    pending = captured["pending"]
    capability = captured["capability"]
    session = captured["session"]
    subject = captured["subject"]
    assert retained["current_pending_sample_completion"] is pending
    assert retained["current_sample_composite_slot"] is subject
    assert retained["native_session"] is session
    sealed_receipt = pending.assert_exact_sealed_receipt_relation(
        session,
        capability,
        subject=subject,
    )
    sample_lifetime = retained["current_sample_lifetime"]
    transfer_lifetime = retained["current_sample_transfer_lifetime"]
    sample_stream = retained["sample_iterator"]

    assert sealed_receipt.consumed is False
    assert capability.outstanding_receipt_identity == id(sealed_receipt)
    assert session._pending_sample_completion is pending
    assert session._outstanding_sample_lifetime is sample_lifetime
    assert sample_lifetime.phase == "launched"
    assert sample_lifetime.consumed is False
    assert sample_lifetime.prepared_payload is not None
    assert sample_stream.active_transfer_lifetime is transfer_lifetime
    assert transfer_lifetime.released_after_completion_fence is False
    assert transfer_lifetime.sample_block is retained["current_sample_block"]
    assert capture.calls == 0
    assert torch.count_nonzero(gradient) == 0
    assert state.poisoned


def test_cache_cap_failure_fences_settled_samples_and_every_built_lane() -> None:
    source, _factory, provider = _provider(
        maximum_tracks_per_bundle=1,
        maximum_observations_per_bundle=3,
    )
    observations = _observations(((0, 0, 0), (0, 1, 1)))
    state = prepare_paper_kinetic_lazy_native_trainer_state(provider, device="cpu")
    material = _material()
    gradient = torch.full_like(material, 23.0)
    capture = _OptimizerCapture()
    native_ops = _FakeNativeOps()

    with pytest.raises(MemoryError, match="before decode"):
        _run(
            state,
            provider,
            observations,
            step_index=0,
            native_ops=native_ops,
            capture=capture,
            material=material,
            gradient=gradient,
            maximum_samples_per_launch=1,
            memory_policy=_memory_policy(
                provider,
                target_frame_access_mode=TARGET_FRAME_STEP_CACHE,
                cache_frame_capacity=1,
            ),
        )
    assert len(source.calls) == 1
    assert native_ops.sample_launch_calls == 1
    assert torch.count_nonzero(gradient) == 0
    assert capture.calls == 0
    assert state.active_step_index is None
    assert state.next_step_index == 0
    assert not state.poisoned


def test_baseexception_closes_both_generators_fences_lane_and_allows_retry(
    monkeypatch,
) -> None:
    _source, _factory, provider = _provider(
        maximum_tracks_per_bundle=1,
        maximum_observations_per_bundle=3,
    )
    observations = _observations(((0, 0, 0), (0, 1, 1)))
    state = prepare_paper_kinetic_lazy_native_trainer_state(provider, device="cpu")
    material = _material()
    gradient = torch.full_like(material, 29.0)
    capture = _OptimizerCapture()
    closed = {"bundles": 0, "samples": 0}
    registered_stages: list[str] = []
    original_bundles = PaperKineticLazyProgramBundleProvider.iter_canonical_spatial_bundles
    original_sample_close = PaperKineticSparseSampleBlockStream.close
    original_register = PaperKineticSealedCompletionFence.register_launch

    def tracked_register(
        capability,
        *,
        stage,
        launch_generation_digest,
        subject_binding=None,
    ):
        registered_stages.append(stage)
        return original_register(
            capability,
            stage=stage,
            launch_generation_digest=launch_generation_digest,
            subject_binding=subject_binding,
        )

    def tracked_bundles(
        self,
        selected,
        *,
        device,
        construction_lifetime_slot=None,
    ):
        inner = original_bundles(
            self,
            selected,
            device=device,
            construction_lifetime_slot=construction_lifetime_slot,
        )
        try:
            yield from inner
        finally:
            inner.close()
            closed["bundles"] += 1

    def tracked_sample_close(self) -> None:
        original_sample_close(self)
        closed["samples"] += 1

    monkeypatch.setattr(
        PaperKineticLazyProgramBundleProvider,
        "iter_canonical_spatial_bundles",
        tracked_bundles,
    )
    monkeypatch.setattr(
        PaperKineticSparseSampleBlockStream,
        "close",
        tracked_sample_close,
    )
    monkeypatch.setattr(
        PaperKineticSealedCompletionFence,
        "register_launch",
        tracked_register,
    )

    with pytest.raises(KeyboardInterrupt, match="synthetic native interruption"):
        _run(
            state,
            provider,
            observations,
            step_index=0,
            native_ops=_InterruptingNativeOps(),
            capture=capture,
            material=material,
            gradient=gradient,
            maximum_samples_per_launch=1,
        )
    assert closed == {"bundles": 1, "samples": 1}
    assert registered_stages.count("sample-completion") == 1
    assert all("abort" not in stage for stage in registered_stages)
    assert torch.count_nonzero(gradient) == 0
    assert capture.calls == 0
    assert state.active_step_index is None
    assert state.next_step_index == 0
    assert not state.poisoned

    _run(
        state,
        provider,
        observations,
        step_index=0,
        native_ops=_FakeNativeOps(),
        capture=capture,
        material=material,
        gradient=gradient,
        maximum_samples_per_launch=1,
    )
    assert capture.calls == 1


def test_optimizer_baseexception_poisoning_happens_only_after_authorization() -> None:
    _source, _factory, provider = _provider(
        maximum_tracks_per_bundle=2,
        maximum_observations_per_bundle=4,
    )
    observations = _observations(((0, 0, 0), (0, 1, 1)))
    state = prepare_paper_kinetic_lazy_native_trainer_state(provider, device="cpu")
    material = _material()
    gradient = torch.full_like(material, 31.0)
    optimizer = _InterruptingOptimizer()
    native_ops = _FakeNativeOps()

    with pytest.raises(KeyboardInterrupt, match="synthetic optimizer interruption"):
        _run(
            state,
            provider,
            observations,
            step_index=0,
            native_ops=native_ops,
            capture=optimizer,
            material=material,
            gradient=gradient,
            maximum_samples_per_launch=1,
        )
    assert optimizer.calls == 1
    # Optimizer authorization happens only after every sample completion,
    # every active-block reverse scratch, and the sole lane are fenced.
    assert native_ops.sample_launch_calls > 0
    assert torch.count_nonzero(gradient) == 0
    assert state.active_step_index is None
    assert state.optimizer_callback_count == 0
    assert state.next_step_index == 0
    assert state.poisoned

    with pytest.raises(ValueError, match="poisoned"):
        _run(
            state,
            provider,
            observations,
            step_index=0,
            native_ops=_FakeNativeOps(),
            capture=_OptimizerCapture(),
            material=material,
            gradient=gradient,
            maximum_samples_per_launch=1,
        )


def test_public_lazy_step_exposes_no_injected_completion_authority() -> None:
    parameters = inspect.signature(
        run_paper_kinetic_lazy_native_material_step
    ).parameters

    assert "device_completion_fence" not in parameters
    assert "device_completion_fence_provenance" not in parameters
    assert "backend_provenance" not in parameters


def test_sample_slot_is_subject_bound_to_exact_plan_session_and_stream_before_next(
    monkeypatch,
) -> None:
    _source, _factory, provider = _provider(
        maximum_tracks_per_bundle=1,
        maximum_observations_per_bundle=2,
    )
    observations = _observations(((0, 0, 0),))
    state = prepare_paper_kinetic_lazy_native_trainer_state(provider, device="cpu")
    captured: dict[str, object] = {}
    observations_at_next: list[tuple[int, ...]] = []
    original_prepare_binding = (
        step_module.prepare_paper_kinetic_completion_subject_binding
    )
    original_next = PaperKineticSparseSampleBlockStream.__next__

    def tracked_prepare_binding(
        capability,
        subject,
        *,
        kind,
        subject_generation_digest,
    ):
        binding = original_prepare_binding(
            capability,
            subject,
            kind=kind,
            subject_generation_digest=subject_generation_digest,
        )
        if kind == step_module.SAMPLE_COMPOSITE_SUBJECT_KIND:
            captured.update(
                capability=capability,
                slot=subject,
                binding=binding,
            )
        return binding

    def tracked_next(stream):
        slot = captured["slot"]
        binding = captured["binding"]
        capability = captured["capability"]
        slot.assert_current()
        assert slot.phase == "registered"
        assert slot.stream is stream
        assert slot.plan is stream.plan
        assert id(slot.plan) == slot.plan_identity
        assert id(slot.session) == slot.session_identity
        assert slot.session.generation_id == slot.session_generation_id
        assert id(stream) == slot.stream_identity
        assert slot.subject_binding is binding
        assert binding._subject is slot
        assert binding.subject_identity == id(slot)
        epoch = slot.launch_epoch
        assert capability.registered_launch_epoch is epoch
        assert epoch.subject_binding is binding
        assert stream.active_transfer_lifetime is None
        observations_at_next.append(
            (
                id(slot),
                id(slot.plan),
                id(slot.session),
                id(stream),
                id(binding),
                id(epoch),
            )
        )
        return original_next(stream)

    monkeypatch.setattr(
        step_module,
        "prepare_paper_kinetic_completion_subject_binding",
        tracked_prepare_binding,
    )
    monkeypatch.setattr(
        PaperKineticSparseSampleBlockStream,
        "__next__",
        tracked_next,
    )

    material = _material()
    _run(
        state,
        provider,
        observations,
        step_index=0,
        native_ops=_FakeNativeOps(),
        capture=_OptimizerCapture(),
        material=material,
        gradient=torch.empty_like(material),
        maximum_samples_per_launch=1,
    )

    assert len(observations_at_next) == 1
    assert len(set(observations_at_next[0])) == len(observations_at_next[0])


def test_native_work_observes_an_exact_pre_registered_launch_epoch(
    monkeypatch,
) -> None:
    _source, _factory, provider = _provider(
        maximum_tracks_per_bundle=1,
        maximum_observations_per_bundle=2,
    )
    observations = _observations(((0, 0, 0),))
    state = prepare_paper_kinetic_lazy_native_trainer_state(provider, device="cpu")
    audit: dict[str, object] = {
        "capability": None,
        "registrations": [],
        "work": [],
    }
    original_register = PaperKineticSealedCompletionFence.register_launch
    original_bundle_materialization = (
        lazy_bundle_module.prepare_paper_kinetic_union_local_spatial_bundle
    )
    original_sample_materialization = (
        sparse_sample_module._materialize_sparse_launch
    )

    def tracked_register(
        capability,
        *,
        stage,
        launch_generation_digest,
        subject_binding=None,
    ):
        epoch = original_register(
            capability,
            stage=stage,
            launch_generation_digest=launch_generation_digest,
            subject_binding=subject_binding,
        )
        audit["capability"] = capability
        audit["registrations"].append(
            (stage, epoch.launch_epoch_sequence, id(epoch))
        )
        return epoch

    def observe_materialization(expected_stage, operation) -> None:
        capability = audit["capability"]
        assert type(capability) is PaperKineticSealedCompletionFence
        epoch = capability.registered_launch_epoch
        assert epoch is not None
        assert epoch.stage == expected_stage
        assert epoch.fenced is False
        audit["work"].append(
            (operation, epoch.launch_epoch_sequence, id(epoch))
        )

    def tracked_bundle_materialization(*args, **kwargs):
        observe_materialization(
            "bundle-materialization",
            "bundle-materialization",
        )
        return original_bundle_materialization(*args, **kwargs)

    def tracked_sample_materialization(lifetime):
        observe_materialization(
            "sample-completion",
            "sample-materialization",
        )
        return original_sample_materialization(lifetime)

    monkeypatch.setattr(
        PaperKineticSealedCompletionFence,
        "register_launch",
        tracked_register,
    )
    monkeypatch.setattr(
        lazy_bundle_module,
        "prepare_paper_kinetic_union_local_spatial_bundle",
        tracked_bundle_materialization,
    )
    monkeypatch.setattr(
        sparse_sample_module,
        "_materialize_sparse_launch",
        tracked_sample_materialization,
    )

    class SequencingNativeOps(_FakeNativeOps):
        def _observe_epoch(self, expected_stage: str, operation: str) -> None:
            capability = audit["capability"]
            assert type(capability) is PaperKineticSealedCompletionFence
            epoch = capability.registered_launch_epoch
            assert epoch is not None
            assert epoch.stage == expected_stage
            assert epoch.fenced is False
            audit["work"].append(
                (operation, epoch.launch_epoch_sequence, id(epoch))
            )

        def kinetic_precompiled_length_p0_lie_node_forward_into_launch_only_v1(
            self,
            *args,
            **kwargs,
        ):
            self._observe_epoch("sample-completion", "forward")
            return super().kinetic_precompiled_length_p0_lie_node_forward_into_launch_only_v1(
                *args,
                **kwargs,
            )

        def prepare_kinetic_ragged_p0_lie_sample_block(self, *args, **kwargs):
            self._observe_epoch("sample-completion", "sample-prepare")
            return super().prepare_kinetic_ragged_p0_lie_sample_block(
                *args,
                **kwargs,
            )

        def kinetic_ragged_p0_lie_sample_accumulate_loss_only_launch_only(
            self,
            *args,
            **kwargs,
        ):
            self._observe_epoch("sample-completion", "sample-launch")
            return super().kinetic_ragged_p0_lie_sample_accumulate_loss_only_launch_only(
                *args,
                **kwargs,
            )

        def kinetic_precompiled_length_p0_lie_material_node_vjp_accumulate_launch_only(
            self,
            *args,
            **kwargs,
        ):
            self._observe_epoch("reverse-completion", "reverse")
            return super().kinetic_precompiled_length_p0_lie_material_node_vjp_accumulate_launch_only(
                *args,
                **kwargs,
            )

    result = _run(
        state,
        provider,
        observations,
        step_index=0,
        native_ops=SequencingNativeOps(),
        capture=_OptimizerCapture(),
        material=_material(),
        gradient=torch.empty_like(_material()),
        maximum_samples_per_launch=1,
    )

    assert {operation for operation, _sequence, _identity in audit["work"]} == {
        "bundle-materialization",
        "forward",
        "sample-materialization",
        "sample-prepare",
        "sample-launch",
        "reverse",
    }
    registration_identities = {
        (sequence, identity)
        for _stage, sequence, identity in audit["registrations"]
    }
    assert all(
        (sequence, identity) in registration_identities
        for _operation, sequence, identity in audit["work"]
    )
    capability = audit["capability"]
    assert type(capability) is PaperKineticSealedCompletionFence
    assert capability.registered_launch_epoch is None
    assert capability.outstanding_receipt_sequence is None
    assert capability.successful_fence_count == capability.consumed_fence_count
    assert result.accounting["sealed_completion_outstanding_receipt_count"] == 0


def test_failed_owned_sample_fence_quarantines_roots_without_retry(
    monkeypatch,
) -> None:
    _source, _factory, provider = _provider(
        maximum_tracks_per_bundle=1,
        maximum_observations_per_bundle=2,
    )
    observations = _observations(((0, 0, 0),))
    state = prepare_paper_kinetic_lazy_native_trainer_state(provider, device="cpu")
    material = _material()
    gradient = torch.full_like(material, 43.0)
    capture = _OptimizerCapture()
    native_ops = _FakeNativeOps()
    audit = _fail_owned_fence_at_stage(
        monkeypatch,
        stage="sample-completion",
    )

    with pytest.raises(PaperKineticCompletionUnknownError):
        _run(
            state,
            provider,
            observations,
            step_index=0,
            native_ops=native_ops,
            capture=capture,
            material=material,
            gradient=gradient,
            maximum_samples_per_launch=1,
        )

    quarantine = state._async_failure_quarantine
    assert quarantine is not None
    quarantine.assert_current()
    retained = dict(
        zip(
            quarantine.retained_reference_roles,
            quarantine.retained_references,
            strict=True,
        )
    )
    lifetime = retained["session_outstanding_sample_lifetime"]
    transfer_lifetime = retained["current_sample_transfer_lifetime"]
    assert quarantine.stage == "sample-completion"
    assert quarantine.restart_required is True
    assert lifetime is retained["current_sample_lifetime"]
    assert lifetime.phase == "completion_unknown"
    assert lifetime.completion_unknown is True
    assert lifetime.consumed is False
    assert lifetime.prepared_payload is not None
    assert retained["current_sample_block"] is lifetime.sample_block
    assert retained["native_session"]._outstanding_sample_lifetime is lifetime
    assert retained["sample_iterator"] is not None
    assert (
        retained["sample_iterator"].active_transfer_lifetime
        is transfer_lifetime
    )
    assert transfer_lifetime.sample_block is retained["current_sample_block"]
    assert transfer_lifetime.released_after_completion_fence is False
    transfer_lifetime.assert_retained()
    assert retained["active_blocks"]
    capability = retained["sealed_completion_fence"]
    assert capability is audit["capability"]
    assert capability.completion_unknown is True
    assert capability.poisoned is True
    assert capability.registered_launch_epoch is not None
    assert capability.registered_launch_epoch.stage == "sample-completion"
    assert audit["matching_calls"] == 1
    assert capture.calls == 0
    assert state.poisoned
    assert state.active_step_index == state.next_step_index == 0
    state.assert_current(provider)

    with pytest.raises(ValueError, match="poisoned"):
        _run(
            state,
            provider,
            observations,
            step_index=0,
            native_ops=_FakeNativeOps(),
            capture=_OptimizerCapture(),
            material=material,
            gradient=gradient,
            maximum_samples_per_launch=1,
        )
    assert audit["matching_calls"] == 1


def test_partial_lane_construction_failure_quarantines_preinstalled_lifetime(
    monkeypatch,
) -> None:
    _source, _factory, provider = _provider(
        maximum_tracks_per_bundle=1,
        maximum_observations_per_bundle=2,
    )
    observations = _observations(((0, 0, 0),))
    state = prepare_paper_kinetic_lazy_native_trainer_state(provider, device="cpu")
    material = _material()
    gradient = torch.full_like(material, 44.0)
    audit = _fail_owned_fence_at_stage(
        monkeypatch,
        stage="lane-construction",
    )

    def interrupt_after_lifetime_install(lifetime):
        lifetime.assert_retained()
        assert lifetime.phase == "installed"
        lifetime.phase = "materializing"
        raise KeyboardInterrupt("synthetic partial lane construction")

    monkeypatch.setattr(
        step_module,
        "materialize_paper_kinetic_native_lazy_bundle_lane",
        interrupt_after_lifetime_install,
    )
    with pytest.raises(KeyboardInterrupt, match="partial lane construction"):
        _run(
            state,
            provider,
            observations,
            step_index=0,
            native_ops=_FakeNativeOps(),
            capture=_OptimizerCapture(),
            material=material,
            gradient=gradient,
            maximum_samples_per_launch=1,
        )

    quarantine = state._async_failure_quarantine
    assert quarantine is not None
    quarantine.assert_current()
    retained = dict(
        zip(
            quarantine.retained_reference_roles,
            quarantine.retained_references,
            strict=True,
        )
    )
    construction = retained["lane_construction_lifetime"]
    assert quarantine.stage == "lane-construction"
    assert construction.phase == "materializing"
    assert construction.provider is provider
    construction.assert_retained()
    assert "native_lane" not in retained
    capability = retained["sealed_completion_fence"]
    assert capability is audit["capability"]
    assert capability.registered_launch_epoch is not None
    assert capability.registered_launch_epoch.stage == "lane-construction"
    assert audit["matching_calls"] == 1
    assert state.poisoned
    assert state.active_step_index == 0


def test_cpu_partial_bundle_construction_failure_releases_slot_and_allows_retry(
    monkeypatch,
) -> None:
    _source, _factory, provider = _provider(
        maximum_tracks_per_bundle=1,
        maximum_observations_per_bundle=2,
    )
    observations = _observations(((0, 0, 0),))
    state = prepare_paper_kinetic_lazy_native_trainer_state(provider, device="cpu")
    material = _material()
    gradient = torch.full_like(material, 44.25)
    capture = _OptimizerCapture()
    original_prepare = (
        lazy_bundle_module.prepare_paper_kinetic_union_local_spatial_bundle
    )
    calls = 0

    def interrupt_after_materialization(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 1:
            return original_prepare(*args, **kwargs)
        lifetime = kwargs["construction_lifetime"]
        bundle = (
            union_assembly_module.materialize_paper_kinetic_union_local_spatial_bundle(
                lifetime
            )
        )
        assert lifetime.bundle_identity == id(bundle)
        lifetime.assert_retained()
        raise KeyboardInterrupt("synthetic partial bundle construction")

    monkeypatch.setattr(
        lazy_bundle_module,
        "prepare_paper_kinetic_union_local_spatial_bundle",
        interrupt_after_materialization,
    )
    with pytest.raises(KeyboardInterrupt, match="partial bundle construction"):
        _run(
            state,
            provider,
            observations,
            step_index=0,
            native_ops=_FakeNativeOps(),
            capture=capture,
            material=material,
            gradient=gradient,
            maximum_samples_per_launch=1,
        )

    assert state._async_failure_quarantine is None
    assert state.active_step_index is None
    assert state.next_step_index == 0
    assert state.poisoned is False
    assert torch.count_nonzero(gradient) == 0

    _run(
        state,
        provider,
        observations,
        step_index=0,
        native_ops=_FakeNativeOps(),
        capture=capture,
        material=material,
        gradient=gradient,
        maximum_samples_per_launch=1,
    )
    assert capture.calls == 1


def test_cpu_sparse_transfer_failure_releases_sources_and_allows_retry(
    monkeypatch,
) -> None:
    _source, _factory, provider = _provider(
        maximum_tracks_per_bundle=1,
        maximum_observations_per_bundle=2,
    )
    observations = _observations(((0, 0, 0),))
    state = prepare_paper_kinetic_lazy_native_trainer_state(provider, device="cpu")
    material = _material()
    gradient = torch.full_like(material, 44.5)
    capture = _OptimizerCapture()
    original_materialize = sparse_sample_module._materialize_sparse_launch
    calls = 0

    def interrupt_after_transfer(lifetime):
        nonlocal calls
        calls += 1
        sample_block = original_materialize(lifetime)
        if calls > 1:
            return sample_block
        lifetime.assert_retained()
        raise KeyboardInterrupt("synthetic sparse transfer interruption")

    monkeypatch.setattr(
        sparse_sample_module,
        "_materialize_sparse_launch",
        interrupt_after_transfer,
    )
    with pytest.raises(KeyboardInterrupt, match="sparse transfer interruption"):
        _run(
            state,
            provider,
            observations,
            step_index=0,
            native_ops=_FakeNativeOps(),
            capture=capture,
            material=material,
            gradient=gradient,
            maximum_samples_per_launch=1,
        )

    assert state._async_failure_quarantine is None
    assert state.active_step_index is None
    assert state.next_step_index == 0
    assert state.poisoned is False
    assert torch.count_nonzero(gradient) == 0

    _run(
        state,
        provider,
        observations,
        step_index=0,
        native_ops=_FakeNativeOps(),
        capture=capture,
        material=material,
        gradient=gradient,
        maximum_samples_per_launch=1,
    )
    assert capture.calls == 1


def test_forward_enqueue_error_and_failed_owned_fence_retain_predecessors(
    monkeypatch,
) -> None:
    _source, _factory, provider = _provider(
        maximum_tracks_per_bundle=1,
        maximum_observations_per_bundle=2,
    )
    observations = _observations(((0, 0, 0),))
    state = prepare_paper_kinetic_lazy_native_trainer_state(provider, device="cpu")
    material = _material()
    gradient = torch.full_like(material, 45.0)
    capture = _OptimizerCapture()
    native_ops = _InterruptingForwardNativeOps()
    audit = _fail_owned_fence_at_stage(
        monkeypatch,
        stage="sample-completion",
    )

    with pytest.raises(KeyboardInterrupt, match="forward enqueue interruption"):
        _run(
            state,
            provider,
            observations,
            step_index=0,
            native_ops=native_ops,
            capture=capture,
            material=material,
            gradient=gradient,
            maximum_samples_per_launch=1,
        )

    quarantine = state._async_failure_quarantine
    assert quarantine is not None
    quarantine.assert_current()
    retained = dict(
        zip(
            quarantine.retained_reference_roles,
            quarantine.retained_references,
            strict=True,
        )
    )
    assert quarantine.stage == "sample-completion"
    assert retained["current_forward_runtime"] is not None
    assert retained["current_forward_compact_material"] is not None
    gather_lifetime = retained["current_compact_gather_lifetime"]
    gather_lifetime.assert_retained()
    assert gather_lifetime.phase == "materialized"
    assert (
        gather_lifetime.compact_site_rgba_f32
        is retained["current_forward_compact_material"]
    )
    assert gather_lifetime.index_select_result_f32 is not None
    forward_lifetime = retained["current_forward_into_lifetime"]
    forward_lifetime.assert_retained(retained["native_session"])
    assert forward_lifetime.phase == "output_published"
    assert forward_lifetime.runtime is retained["current_forward_runtime"]
    assert (
        forward_lifetime.compact_site_rgba_f32
        is retained["current_forward_compact_material"]
    )
    assert (
        forward_lifetime.node_chart_out_f32
        is retained["current_forward_node_chart_out"]
    )
    assert forward_lifetime.world is None
    assert forward_lifetime.token is None
    assert "current_forward_token" not in retained
    top_level = retained["top_level_device_transaction_lifetime"]
    top_level.assert_retained()
    assert top_level.phase == "active"
    assert state._active_device_transaction_lifetime is top_level
    assert retained["current_sample_block"] is not None
    transfer_lifetime = retained["current_sample_transfer_lifetime"]
    assert transfer_lifetime.sample_block is retained["current_sample_block"]
    assert (
        retained["sample_iterator"].active_transfer_lifetime
        is transfer_lifetime
    )
    transfer_lifetime.assert_retained()
    assert retained["sample_iterator"] is not None
    assert retained["native_session"]._sealed is False
    capability = retained["sealed_completion_fence"]
    assert capability is audit["capability"]
    assert capability.completion_unknown is True
    assert capability.registered_launch_epoch is not None
    assert capability.registered_launch_epoch.stage == "sample-completion"
    assert audit["matching_calls"] == 1
    assert native_ops.forward_calls == 1
    assert native_ops.sample_launch_calls == 0
    assert capture.calls == 0
    assert state.poisoned
    assert state.active_step_index == 0


def test_top_level_output_publication_failure_quarantines_preinstalled_lifetime(
    monkeypatch,
) -> None:
    _source, _factory, provider = _provider(
        maximum_tracks_per_bundle=1,
        maximum_observations_per_bundle=2,
    )
    observations = _observations(((0, 0, 0),))
    state = prepare_paper_kinetic_lazy_native_trainer_state(provider, device="cpu")
    material = _material()
    gradient = torch.full_like(material, 45.25)
    audit = _fail_owned_fence_at_stage(
        monkeypatch,
        stage="top-level-initialization",
    )
    original_publish = step_module._TopLevelDeviceTransactionLifetime.publish_loss

    def interrupt_after_publish(lifetime, loss_f32):
        original_publish(lifetime, loss_f32)
        lifetime.assert_retained()
        raise KeyboardInterrupt("synthetic top-level publication interruption")

    monkeypatch.setattr(
        step_module._TopLevelDeviceTransactionLifetime,
        "publish_loss",
        interrupt_after_publish,
    )
    with pytest.raises(
        KeyboardInterrupt,
        match="top-level publication interruption",
    ):
        _run(
            state,
            provider,
            observations,
            step_index=0,
            native_ops=_FakeNativeOps(),
            capture=_OptimizerCapture(),
            material=material,
            gradient=gradient,
            maximum_samples_per_launch=1,
        )

    quarantine = state._async_failure_quarantine
    assert quarantine is not None
    quarantine.assert_current()
    retained = dict(
        zip(
            quarantine.retained_reference_roles,
            quarantine.retained_references,
            strict=True,
        )
    )
    lifetime = retained["top_level_device_transaction_lifetime"]
    lifetime.assert_retained()
    assert quarantine.stage == "top-level-initialization"
    assert lifetime.phase == "loss_published"
    assert lifetime.loss_f32 is not None
    assert "loss_f32" not in retained
    assert state._active_device_transaction_lifetime is lifetime
    capability = retained["sealed_completion_fence"]
    assert capability is audit["capability"]
    assert capability.completion_unknown is True
    assert capability.registered_launch_epoch is not None
    assert capability.registered_launch_epoch.stage == "top-level-initialization"
    assert audit["matching_calls"] == 1
    assert state.poisoned


def test_cleanup_zero_has_its_own_failed_fence_and_bounded_quarantine(
    monkeypatch,
) -> None:
    _source, _factory, provider = _provider(
        maximum_tracks_per_bundle=1,
        maximum_observations_per_bundle=2,
    )
    observations = _observations(((0, 0, 0),))
    state = prepare_paper_kinetic_lazy_native_trainer_state(provider, device="cpu")
    material = _material()
    gradient = torch.full_like(material, 45.5)
    capture = _OptimizerCapture()
    audit = _fail_owned_fence_at_stage(
        monkeypatch,
        stage="top-level-cleanup-zero",
    )

    with pytest.raises(KeyboardInterrupt, match="synthetic native interruption"):
        _run(
            state,
            provider,
            observations,
            step_index=0,
            native_ops=_InterruptingNativeOps(),
            capture=capture,
            material=material,
            gradient=gradient,
            maximum_samples_per_launch=1,
        )

    quarantine = state._async_failure_quarantine
    assert quarantine is not None
    quarantine.assert_current()
    retained = dict(
        zip(
            quarantine.retained_reference_roles,
            quarantine.retained_references,
            strict=True,
        )
    )
    lifetime = retained["top_level_device_transaction_lifetime"]
    lifetime.assert_retained()
    assert quarantine.stage == "top-level-cleanup-zero"
    assert lifetime.phase == "cleanup_zeroing"
    assert lifetime.global_grad_site_rgba_f32 is gradient
    assert lifetime.loss_f32 is retained["loss_f32"]
    assert state._active_device_transaction_lifetime is lifetime
    capability = retained["sealed_completion_fence"]
    assert capability is audit["capability"]
    assert capability.completion_unknown is True
    assert capability.registered_launch_epoch is not None
    assert capability.registered_launch_epoch.stage == "top-level-cleanup-zero"
    assert audit["matching_calls"] == 1
    assert capture.calls == 0
    assert state.poisoned


def test_failed_owned_reverse_fence_quarantines_bounded_scratch(
    monkeypatch,
) -> None:
    _source, _factory, provider = _provider(
        maximum_tracks_per_bundle=1,
        maximum_observations_per_bundle=2,
    )
    observations = _observations(((0, 0, 0),))
    state = prepare_paper_kinetic_lazy_native_trainer_state(provider, device="cpu")
    material = _material()
    gradient = torch.full_like(material, 47.0)
    capture = _OptimizerCapture()
    native_ops = _FakeNativeOps()
    audit = _fail_owned_fence_at_stage(
        monkeypatch,
        stage="reverse-completion",
    )

    with pytest.raises(PaperKineticCompletionUnknownError):
        _run(
            state,
            provider,
            observations,
            step_index=0,
            native_ops=native_ops,
            capture=capture,
            material=material,
            gradient=gradient,
            maximum_samples_per_launch=1,
        )

    quarantine = state._async_failure_quarantine
    assert quarantine is not None
    quarantine.assert_current()
    retained = dict(
        zip(
            quarantine.retained_reference_roles,
            quarantine.retained_references,
            strict=True,
        )
    )
    reverse_block = retained["current_reverse_block_state"]
    assert quarantine.stage == "reverse-completion"
    assert retained["current_reverse_runtime"] is not None
    assert retained["current_reverse_compact_bar"] is not None
    assert retained["current_material_execution"] is not None
    assert any(
        block is reverse_block
        for block in retained["active_blocks"].values()
    )
    assert retained["native_session"]._sealed is False
    capability = retained["sealed_completion_fence"]
    assert capability is audit["capability"]
    assert capability.completion_unknown is True
    assert capability.registered_launch_epoch is not None
    assert capability.registered_launch_epoch.stage == "reverse-completion"
    assert audit["matching_calls"] == 1
    assert native_ops.sample_launch_calls == 1
    assert native_ops.material_vjp_calls == 1
    assert capture.calls == 0
    assert state.poisoned
    assert state.active_step_index == 0


def test_failed_receipt_consumption_precedes_and_blocks_root_release(
    monkeypatch,
) -> None:
    _source, _factory, provider = _provider(
        maximum_tracks_per_bundle=1,
        maximum_observations_per_bundle=2,
    )
    observations = _observations(((0, 0, 0),))
    state = prepare_paper_kinetic_lazy_native_trainer_state(provider, device="cpu")
    material = _material()
    gradient = torch.full_like(material, 53.0)
    capture = _OptimizerCapture()
    native_ops = _FakeNativeOps()
    original_consume = (
        PaperKineticSealedCompletionFence._consume_published_receipt
    )
    audit: dict[str, object] = {"capability": None, "calls": 0}

    def reject_reverse_consumption(capability, receipt, *, consumer) -> None:
        if receipt.stage == "reverse-completion":
            audit["capability"] = capability
            audit["calls"] = int(audit["calls"]) + 1
            raise RuntimeError("synthetic sealed receipt consumption failure")
        original_consume(capability, receipt, consumer=consumer)

    monkeypatch.setattr(
        PaperKineticSealedCompletionFence,
        "_consume_published_receipt",
        reject_reverse_consumption,
    )

    with pytest.raises(
        RuntimeError,
        match="synthetic sealed receipt consumption failure",
    ):
        _run(
            state,
            provider,
            observations,
            step_index=0,
            native_ops=native_ops,
            capture=capture,
            material=material,
            gradient=gradient,
            maximum_samples_per_launch=1,
        )

    quarantine = state._async_failure_quarantine
    assert quarantine is not None
    quarantine.assert_current()
    retained = dict(
        zip(
            quarantine.retained_reference_roles,
            quarantine.retained_references,
            strict=True,
        )
    )
    assert quarantine.restart_required is True
    assert retained["native_lane"] is not None
    assert retained["native_session"]._sealed is True
    assert retained["active_blocks"]
    assert retained["global_grad_site_rgba_f32"] is gradient
    assert retained["loss_f32"] is not None
    capability = retained["sealed_completion_fence"]
    assert capability is audit["capability"]
    assert capability.completion_unknown is False
    assert capability.poisoned is False
    assert capability.outstanding_receipt_sequence is not None
    assert audit["calls"] >= 1
    assert native_ops.sample_launch_calls == 1
    assert native_ops.material_vjp_calls == 1
    assert capture.calls == 0
    assert state.optimizer_callback_count == 0
    assert state.poisoned
    assert state.active_step_index == 0


def test_accelerator_stays_fail_closed_before_native_or_fence_work() -> None:
    _source, _factory, provider = _provider(
        maximum_tracks_per_bundle=1,
        maximum_observations_per_bundle=2,
    )
    observations = _observations(((0, 0, 0),))
    state = prepare_paper_kinetic_lazy_native_trainer_state(provider, device="cpu")
    state.device = torch.device("mps")
    material = _material()
    gradient = torch.full_like(material, 59.0)
    capture = _OptimizerCapture()
    native_ops = _FakeNativeOps()

    with pytest.raises(RuntimeError, match="accelerator execution is fail-closed"):
        _run(
            state,
            provider,
            observations,
            step_index=0,
            native_ops=native_ops,
            capture=capture,
            material=material,
            gradient=gradient,
            maximum_samples_per_launch=1,
        )
    torch.testing.assert_close(gradient, torch.full_like(material, 59.0))
    assert native_ops.forward_calls == 0
    assert native_ops.sample_prepare_calls == 0
    assert native_ops.sample_launch_calls == 0
    assert native_ops.material_vjp_calls == 0
    assert capture.calls == 0
    assert state.active_step_index is None
    assert state._async_failure_quarantine is None
    assert not state.poisoned
