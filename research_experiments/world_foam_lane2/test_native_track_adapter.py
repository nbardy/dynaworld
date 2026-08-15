from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from camera import CameraSpec
from compact_lie_schedule import compact_lie_world_schedule_from_atlas
from compiled_lie_world_adjoint import (
    AdaptiveCompiledLieWorldAtlas,
    AdaptiveLieWorldCompilePolicy,
    compile_lie_world_atlas,
)
from compiled_transfer_adjoint import make_stable_cell_word
from native_track_adapter import (
    NativeTrackAdapterUnavailableError,
    assert_native_fixed_word_p0_validated_topology_token,
    consume_native_fixed_word_p0_track_block_result,
    execute_native_fixed_word_p0_track_block,
)
from powerfoam_track_staging import PowerFoamTrackStagingPlan
from powerfoam_training_data import PowerFoamRayProvider, PowerFoamTargetProvider
from prepared_track_block import prepare_worldfoam_track_block
from staged_compiled_lie_adjoint import (
    allocate_compact_spatial_gradient_buffers,
    begin_compact_spatial_step_v2,
    finalize_compact_spatial_step,
    prepare_compact_staged_lie_world_snapshot_v2,
)


@dataclass(frozen=True)
class _FakeChartBinding:
    chart_digest: str


class _FakeBinding:
    def __init__(self, prepared) -> None:
        self._prepared = prepared
        self.binding_mode = "strict_frozen_evaluation"
        self.canonical_digest = f"binding-{prepared.topology.source_track_ids.tolist()}"
        self.charts = tuple(
            _FakeChartBinding(f"chart-{index}") for index in range(len(prepared.world_snapshot.atlas.charts))
        )

    def assert_current(self) -> None:
        self._prepared.assert_current()


class _FakeTrainingBinding:
    """Material-refresh capability that owns no prepared/template object."""

    binding_mode = "training_owner_topology_only"
    paper_evidence_eligible = False
    transfer_jacobian_certified = False
    approximation_error_certified = False

    def __init__(self, prepared) -> None:
        self.canonical_digest = f"training-{prepared.topology.source_track_ids.tolist()}"
        self.topology_snapshot_generation = f"topology-{prepared.topology.source_track_ids.tolist()}"
        self.charts = tuple(
            _FakeChartBinding(f"training-chart-{index}") for index in range(len(prepared.world_snapshot.atlas.charts))
        )
        self._schedule_generation = prepared.schedule.generation_digest
        self._topology = tuple(
            tensor.clone()
            for tensor in (
                prepared.topology.source_track_ids,
                prepared.topology.source_boundary_ids,
                prepared.topology.source_site_ids,
                prepared.topology.word_offsets_i32,
                prepared.topology.word_owner_i32,
                prepared.topology.word_left_incidence_i32,
                prepared.topology.word_right_incidence_i32,
                prepared.topology.track_incidence_offsets_i32,
                prepared.topology.incidence_boundary_i32,
                prepared.topology.boundary_site_pairs_i32,
            )
        )
        self._site_geometry = prepared.site_geometry.clone()
        self._rays = prepared.world_snapshot.ray_coefficients.clone()

    def assert_current(self) -> None:
        assert self.canonical_digest

    def assert_prepared_immutable(self, prepared) -> None:
        prepared.schedule.assert_current()
        assert prepared.schedule.generation_digest == self._schedule_generation
        actual_topology = (
            prepared.topology.source_track_ids,
            prepared.topology.source_boundary_ids,
            prepared.topology.source_site_ids,
            prepared.topology.word_offsets_i32,
            prepared.topology.word_owner_i32,
            prepared.topology.word_left_incidence_i32,
            prepared.topology.word_right_incidence_i32,
            prepared.topology.track_incidence_offsets_i32,
            prepared.topology.incidence_boundary_i32,
            prepared.topology.boundary_site_pairs_i32,
        )
        assert all(
            torch.equal(actual, expected) for actual, expected in zip(actual_topology, self._topology, strict=True)
        )
        assert torch.equal(prepared.site_geometry, self._site_geometry)
        assert torch.equal(prepared.world_snapshot.ray_coefficients, self._rays)


class _FakeNativeLifecycle:
    """CPU lifecycle double with the same caller-visible token boundaries."""

    def __init__(self, *, wrong_chart_identity: bool = False) -> None:
        self.sample_states = []
        self.sample_targets = []
        self.sample_times = []
        self.chart_node_times = []
        self.world_grad_calls = []
        self.material_world_grad_calls = []
        self.full_node_vjp_count = 0
        self.material_node_vjp_count = 0
        self.full_geometry_finalize_count = 0
        self.material_world_grad_finalize_count = 0
        self.wrong_chart_identity = wrong_chart_identity
        self.active_chart_index = None
        self.topology_prepare_count = 0
        self.world_refresh_count = 0

    def prepare_fixed_word_p0_topology_token(self, *tensors, **kwargs):
        self.topology_prepare_count += 1
        binding = kwargs["certificate_binding"]
        return SimpleNamespace(
            tensors=tensors,
            track_count=kwargs["track_count"],
            site_count=kwargs["site_count"],
            certificate_binding=binding,
            topology_generation_id=getattr(binding, "topology_snapshot_generation", ""),
            training_binding_digest=(
                binding.canonical_digest if binding.binding_mode == "training_owner_topology_only" else None
            ),
        )

    def refresh_fixed_word_p0_world_token(
        self,
        topology,
        sites_f32,
        site_rgba_f32,
        track_ray_coeff_f32,
        replay_config,
        **_kwargs,
    ):
        self.world_refresh_count += 1
        return SimpleNamespace(
            topology=topology,
            sites_f32=sites_f32,
            site_rgba_f32=site_rgba_f32,
            track_ray_coeff_f32=track_ray_coeff_f32,
            replay_config=replay_config,
        )

    def prepare_fixed_word_p0_chart_token(self, world, compiler_node_t_f32, *, chart_index):
        assert self.active_chart_index is None
        self.active_chart_index = chart_index
        expected = world.topology.certificate_binding.charts[chart_index]
        self.chart_node_times.append((chart_index, compiler_node_t_f32.clone()))
        return SimpleNamespace(
            world=world,
            chart_index=chart_index,
            chart_generation_id=("wrong-chart" if self.wrong_chart_identity else expected.chart_digest),
            node_count=int(compiler_node_t_f32.numel()),
        )

    def fixed_word_p0_lie_world_grad_init_launch_only(self, world, **kwargs):
        self.world_grad_calls.append(kwargs)
        return SimpleNamespace(
            world=world,
            grad_site_rgba_f32=torch.zeros_like(world.site_rgba_f32),
            boundary_finalized=False,
        )

    def fixed_word_p0_lie_material_world_grad_init_launch_only(self, world, **kwargs):
        self.material_world_grad_calls.append(kwargs)
        return SimpleNamespace(
            world=world,
            grad_site_rgba_f32=torch.zeros_like(world.site_rgba_f32),
            finalized=False,
        )

    def prepare_fixed_word_p0_sample_state_token(self, chart, **kwargs):
        state = SimpleNamespace(
            chart=chart,
            loss_f32=torch.zeros((), dtype=torch.float32),
            global_loss_scale=1.0 / float(kwargs["global_loss_element_count"]),
            sample_block_size=kwargs["sample_block_size"],
            kwargs=kwargs,
        )
        self.sample_states.append(state)
        return state

    # Deliberately no interpolation-weight argument. The real lifecycle derives
    # it from this bounded block's times and does not retain a chart-wide tape.
    def prepare_fixed_word_p0_sample_block_token(
        self,
        sample_state,
        target_rgb_f32,
        background_rgb_f32,
        *,
        sample_t_f64,
        sample_block_id,
        global_sample_start,
        global_sample_end,
    ):
        self.sample_targets.append(target_rgb_f32.clone())
        self.sample_times.append(sample_t_f64.clone())
        sample_count = global_sample_end - global_sample_start
        assert tuple(sample_t_f64.shape) == (sample_count,)
        return SimpleNamespace(
            sample_state=sample_state,
            target_rgb_f32=target_rgb_f32,
            background_rgb_f32=background_rgb_f32,
            sample_block_id=sample_block_id,
            global_sample_start=global_sample_start,
            global_sample_end=global_sample_end,
            sample_weight_evaluation="verified_fit_derived_second_form_barycentric",
            sample_weight_linear_interactions=sample_count * sample_state.chart.node_count,
            sample_weight_dense_fallback_interactions=0,
            sample_weight_exact_node_rows=0,
            sample_weight_dense_fallback_rows=0,
        )

    def fixed_word_p0_lie_sample_accumulate_loss_only_launch_only(
        self,
        sample_block,
        sample_state,
    ):
        sample_state.loss_f32.add_(sample_block.target_rgb_f32.square().sum() * sample_state.global_loss_scale)

    def fixed_word_p0_lie_node_vjp_accumulate_launch_only(self, chart, _sample_state, world_grad):
        assert self.active_chart_index == chart.chart_index
        self.full_node_vjp_count += 1
        world_grad.grad_site_rgba_f32.add_(float(chart.chart_index + 1))
        self.active_chart_index = None

    def fixed_word_p0_lie_material_node_vjp_accumulate_launch_only(
        self,
        chart,
        _sample_state,
        world_grad,
    ):
        assert self.active_chart_index == chart.chart_index
        self.material_node_vjp_count += 1
        world_grad.grad_site_rgba_f32.add_(float(chart.chart_index + 1))
        self.active_chart_index = None

    def fixed_word_p0_lie_material_world_grad_finalize_launch_only(self, world_grad):
        assert not world_grad.finalized
        self.material_world_grad_finalize_count += 1
        world_grad.finalized = True
        return world_grad.grad_site_rgba_f32

    def fixed_word_p0_sparse_mobius_boundary_finalize_launch_only(self, world_grad):
        world_grad.boundary_finalized = True
        return torch.empty((0, 5), dtype=torch.float32)

    def fixed_word_p0_site_geometry_finalize_launch_only(self, world_grad):
        assert world_grad.boundary_finalized
        self.full_geometry_finalize_count += 1
        values = torch.arange(
            world_grad.world.topology.site_count * 5,
            dtype=torch.float32,
        )
        return values.reshape(world_grad.world.topology.site_count, 5)


def _camera(x: float = 0.0) -> CameraSpec:
    transform = torch.eye(4, dtype=torch.float32)
    transform[0, 3] = x
    return CameraSpec(
        fx=3.0,
        fy=3.0,
        cx=1.0,
        cy=1.0,
        camera_to_world=transform,
    )


def _compiled_blocks_and_ledger(
    rays: torch.Tensor,
    *,
    global_sample_count: int,
    blocks: tuple[tuple[str, int, int], ...],
    node_counts: tuple[int, int] = (2, 2),
):
    sites = torch.tensor([[0.0, 0.0, 0.3, 0.0, 0.0]], dtype=torch.float32)
    density = torch.tensor([0.5], dtype=torch.float32)
    color = torch.tensor([[0.2, 0.4, 0.7]], dtype=torch.float32)
    boundary = torch.empty((0, 5), dtype=torch.float64)
    words = tuple(make_stable_cell_word([0], [-1], [-2]) for _ in range(rays.shape[0]))
    charts = tuple(
        compile_lie_world_atlas(
            boundary=boundary,
            ray_coefficients=rays.to(dtype=torch.float64),
            words=words,
            site_density=density.to(dtype=torch.float64),
            site_color=color.to(dtype=torch.float64),
            t_min=t_min,
            t_max=t_max,
            near=0.1,
            far=1.0,
            node_count=node_count,
        )
        for (t_min, t_max), node_count in zip(
            ((0.0, 0.5), (0.5, 1.0)),
            node_counts,
            strict=True,
        )
    )
    atlas = AdaptiveCompiledLieWorldAtlas(
        charts=charts,
        selections=(),
        policy=AdaptiveLieWorldCompilePolicy(),
        supplied_word_ordering_check=charts[0].supplied_word_ordering_check,
    )
    schedule = compact_lie_world_schedule_from_atlas(atlas)
    pairs = torch.empty((0, 2), dtype=torch.int64)
    prepared = tuple(
        prepare_compact_staged_lie_world_snapshot_v2(
            schedule,
            prepare_worldfoam_track_block(
                words,
                pairs,
                site_count=1,
                track_start=track_start,
                track_end=track_end,
            ),
            site_geometry=sites,
            ray_coefficients=rays,
            site_density=density,
            site_color=color,
        )
        for _block_id, track_start, track_end in blocks
    )
    gradients = allocate_compact_spatial_gradient_buffers(
        site_geometry=sites,
        site_density=density,
        site_color=color,
    )
    ledger = begin_compact_spatial_step_v2(
        schedule=schedule,
        site_geometry=sites,
        ray_coefficients=rays,
        site_density=density,
        site_color=color,
        gradients=gradients,
        global_track_count=int(rays.shape[0]),
        global_frame_count=global_sample_count,
        loss_normalization_id="paper-logical-batch",
        expected_blocks=blocks,
    )
    return prepared, ledger


def _case():
    frames = torch.arange(2 * 4 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 4, 3, 2, 2)
    frames = frames / float(frames.numel())
    target_provider = PowerFoamTargetProvider.from_resident_frames(frames, device=torch.device("cpu"))
    ray_provider = PowerFoamRayProvider(
        (
            tuple(_camera(0.0) for _ in range(4)),
            tuple(_camera(0.25) for _ in range(4)),
        ),
        height=2,
        width=2,
        device=torch.device("cpu"),
    )
    plan = PowerFoamTrackStagingPlan(
        target_provider,
        ray_provider,
        torch.tensor([0, 1, 2, 3]),
        torch.tensor([3, 0, 2, 1]),
        sample_times=torch.tensor([1.0, 0.0, 2.0 / 3.0, 1.0 / 3.0]),
    )
    staged = plan.stage(require_affine_ray_program=True)
    assert staged.affine_ray_program is not None
    rays = staged.affine_ray_program.coefficients[0].to(dtype=torch.float32)
    prepared, ledger = _compiled_blocks_and_ledger(
        rays,
        global_sample_count=4,
        blocks=(("pixels-0-2", 0, 2), ("pixels-2-4", 2, 4)),
    )
    return plan, prepared, ledger


def _multiview_case():
    base_plan, _prepared, _ledger = _case()
    plan = PowerFoamTrackStagingPlan(
        base_plan.target_provider,
        base_plan.ray_provider,
        base_plan.pixel_indices,
        torch.tensor([7, 0, 5, 2, 4, 1, 6, 3]),
    )
    view_tracks = plan.stage(require_affine_ray_program=True).as_view_tracks()
    blocks = (("view-0", 0, 4), ("view-1", 4, 8))
    prepared, ledger = _compiled_blocks_and_ledger(
        view_tracks.ray_coefficients,
        global_sample_count=4,
        blocks=blocks,
    )
    return plan, view_tracks, prepared, ledger


def test_bp_and_k_blocks_keep_one_denominator_and_feed_global_gradient_ledger() -> None:
    plan, prepared_blocks, ledger = _case()
    assert ledger.template is None
    assert all(prepared.template is None for prepared in prepared_blocks)
    assert all(prepared.schedule is ledger.schedule for prepared in prepared_blocks)
    native = _FakeNativeLifecycle()
    pointers = tuple(tensor.untyped_storage().data_ptr() for tensor in ledger.gradients.tensors)
    partition_generation_ids = []
    barrier_counts = []
    barriers = []
    for block_id, prepared in zip(("pixels-0-2", "pixels-2-4"), prepared_blocks, strict=True):
        result = execute_native_fixed_word_p0_track_block(
            ledger,
            block_id=block_id,
            prepared=prepared,
            staging_plan=plan,
            certificate_binding=_FakeBinding(prepared),
            background_rgb=(0.0, 0.0, 0.0),
            replay_config=SimpleNamespace(near=0.1, far=1.0),
            sample_block_size=1,
            native_ops=native,
            device_synchronize=barriers.append,
        )
        partition_generation_ids.append(result.sample_partition_generation_id)
        barrier_counts.append(result.device_barrier_count)
        assert result.sample_block_count == 4
        assert result.sample_weight_evaluation == "verified_fit_derived_second_form_barycentric"
        assert result.sample_weight_linear_interactions == 4 * 2
        assert result.sample_weight_dense_fallback_interactions == 0
        assert result.geometry_vjp_executed
        assert result.grad_site_geometry_f32 is not None
        assert result.sample_payload_layout == "target_plus_explicit_rays"
        assert result.peak_staged_target_bytes == 2 * 1 * 3 * 4
        assert result.peak_staged_explicit_ray_bytes == 2 * 1 * 6 * 4
        assert result.peak_staged_sample_time_bytes == 1 * 8
        assert "prediction" not in vars(result)
        consume_native_fixed_word_p0_track_block_result(ledger, result)
        assert tuple(tensor.untyped_storage().data_ptr() for tensor in ledger.gradients.tensors) == pointers
        del result

    final = finalize_compact_spatial_step(ledger)
    full_targets = plan.stage().targets
    expected_loss = full_targets.square().sum() / float(4 * 4 * 3)
    torch.testing.assert_close(final.loss, expected_loss)
    torch.testing.assert_close(
        final.gradients.grad_site_geometry,
        torch.tensor([[0.0, 2.0, 4.0, 6.0]]),
    )
    torch.testing.assert_close(final.gradients.grad_site_weight, torch.tensor([8.0]))
    torch.testing.assert_close(final.gradients.grad_site_color, torch.full((1, 3), 6.0))
    torch.testing.assert_close(final.gradients.grad_site_density, torch.tensor([6.0]))
    assert {call["global_loss_element_count"] for call in native.world_grad_calls} == {4 * 4 * 3}
    assert {state.kwargs["global_loss_element_count"] for state in native.sample_states} == {4 * 4 * 3}
    assert [state.kwargs["global_sample_start"] for state in native.sample_states] == [0, 2, 0, 2]
    assert [state.kwargs["global_sample_end"] for state in native.sample_states] == [2, 4, 2, 4]
    assert {state.kwargs["sample_block_size"] for state in native.sample_states} == {1}
    assert all("expected_sample_blocks" not in state.kwargs for state in native.sample_states)
    assert all("global_sample_t_f64" not in state.kwargs for state in native.sample_states)
    assert all(tuple(times.shape) == (1,) for times in native.sample_times)
    assert partition_generation_ids[0] == partition_generation_ids[1]
    assert barrier_counts == [6, 6]
    assert barriers == [torch.device("cpu")] * 12
    assert final.accounting["global_loss_element_count"] == 4 * 4 * 3


def test_rectangular_fixed_views_expand_to_view_major_tracks_without_restaging_other_views() -> None:
    plan, view_tracks, prepared_blocks, ledger = _multiview_case()
    native = _FakeNativeLifecycle()
    partition_ids = []
    for block_id, prepared in zip(("view-0", "view-1"), prepared_blocks, strict=True):
        result = execute_native_fixed_word_p0_track_block(
            ledger,
            block_id=block_id,
            prepared=prepared,
            staging_plan=plan,
            certificate_binding=_FakeBinding(prepared),
            background_rgb=(0.0, 0.0, 0.0),
            replay_config=SimpleNamespace(near=0.1, far=1.0),
            sample_block_size=1,
            native_ops=native,
        )
        partition_ids.append(result.sample_partition_generation_id)
        consume_native_fixed_word_p0_track_block_result(ledger, result)
        del result

    final = finalize_compact_spatial_step(ledger)
    torch.testing.assert_close(
        final.loss,
        view_tracks.targets.square().sum() / float(view_tracks.global_rgb_element_count),
    )
    assert partition_ids[0] == partition_ids[1]
    assert all(tuple(target.shape) == (4, 1, 3) for target in native.sample_targets)
    assert {call["global_track_count"] for call in native.world_grad_calls} == {8}
    assert {call["global_sample_count"] for call in native.world_grad_calls} == {4}
    torch.testing.assert_close(final.gradients.grad_site_color, torch.full((1, 3), 6.0))
    torch.testing.assert_close(final.gradients.grad_site_density, torch.tensor([6.0]))


def test_nonrectangular_mixed_views_fail_before_native_lifecycle() -> None:
    plan, _view_tracks, prepared_blocks, ledger = _multiview_case()
    unbalanced = PowerFoamTrackStagingPlan(
        plan.target_provider,
        plan.ray_provider,
        plan.pixel_indices,
        torch.tensor([0, 4, 1]),
    )
    native = _FakeNativeLifecycle()
    with pytest.raises(NativeTrackAdapterUnavailableError, match="rectangular frame/time grid"):
        execute_native_fixed_word_p0_track_block(
            ledger,
            block_id="view-0",
            prepared=prepared_blocks[0],
            staging_plan=unbalanced,
            certificate_binding=_FakeBinding(prepared_blocks[0]),
            background_rgb=(0.0, 0.0, 0.0),
            replay_config=SimpleNamespace(near=0.1, far=1.0),
            sample_block_size=1,
            native_ops=native,
        )
    assert native.world_grad_calls == []


def test_spatial_result_must_be_consumed_before_the_next_binding_is_executed() -> None:
    plan, prepared_blocks, ledger = _case()
    first = execute_native_fixed_word_p0_track_block(
        ledger,
        block_id="pixels-0-2",
        prepared=prepared_blocks[0],
        staging_plan=plan,
        certificate_binding=_FakeBinding(prepared_blocks[0]),
        background_rgb=(0.0, 0.0, 0.0),
        replay_config=SimpleNamespace(near=0.1, far=1.0),
        sample_block_size=2,
        native_ops=_FakeNativeLifecycle(),
    )
    with pytest.raises(ValueError, match="consume and drop the previous native spatial result"):
        execute_native_fixed_word_p0_track_block(
            ledger,
            block_id="pixels-2-4",
            prepared=prepared_blocks[1],
            staging_plan=plan,
            certificate_binding=_FakeBinding(prepared_blocks[1]),
            background_rgb=(0.0, 0.0, 0.0),
            replay_config=SimpleNamespace(near=0.1, far=1.0),
            sample_block_size=2,
            native_ops=_FakeNativeLifecycle(),
        )
    consume_native_fixed_word_p0_track_block_result(ledger, first)


def test_moving_camera_and_live_ray_mismatch_fail_closed() -> None:
    plan, prepared_blocks, ledger = _case()
    moved_cameras = list(plan.ray_provider.cameras[0])
    moved_cameras[2] = _camera(0.1)
    moving_provider = PowerFoamRayProvider(
        (tuple(moved_cameras), plan.ray_provider.cameras[1]),
        height=2,
        width=2,
        device=torch.device("cpu"),
    )
    moving = PowerFoamTrackStagingPlan(
        plan.target_provider,
        moving_provider,
        plan.pixel_indices,
        plan.sample_indices,
        sample_times=plan.sample_times,
    )
    with pytest.raises(ValueError, match="piecewise-affine/projective camera-gauge compiler"):
        execute_native_fixed_word_p0_track_block(
            ledger,
            block_id="pixels-0-2",
            prepared=prepared_blocks[0],
            staging_plan=moving,
            certificate_binding=_FakeBinding(prepared_blocks[0]),
            background_rgb=(0.0, 0.0, 0.0),
            replay_config=SimpleNamespace(near=0.1, far=1.0),
            sample_block_size=2,
            native_ops=_FakeNativeLifecycle(),
        )
    with pytest.raises(ValueError, match="piecewise-affine/projective camera-gauge compiler"):
        execute_native_fixed_word_p0_track_block(
            ledger,
            block_id="pixels-0-2",
            prepared=prepared_blocks[0],
            staging_plan=moving,
            certificate_binding=_FakeTrainingBinding(prepared_blocks[0]),
            background_rgb=(0.0, 0.0, 0.0),
            replay_config=SimpleNamespace(near=0.1, far=1.0),
            sample_block_size=2,
            native_ops=_FakeNativeLifecycle(),
            immutable_generation_id="moving-camera-material-program",
        )

    plan, prepared_blocks, _ledger = _case()
    sites, rays, density, color = _ledger.source_tensors
    mismatched_rays = rays.clone()
    mismatched_rays[0, 0] += 0.25
    gradients = allocate_compact_spatial_gradient_buffers(
        site_geometry=sites,
        site_density=density,
        site_color=color,
    )
    mismatch_ledger = begin_compact_spatial_step_v2(
        schedule=_ledger.schedule,
        site_geometry=sites,
        ray_coefficients=mismatched_rays,
        site_density=density,
        site_color=color,
        gradients=gradients,
        global_track_count=4,
        global_frame_count=4,
        loss_normalization_id="paper-logical-batch",
        expected_blocks=(("pixels-0-2", 0, 2), ("pixels-2-4", 2, 4)),
    )
    with pytest.raises(ValueError, match="does not match the certified live track rays"):
        execute_native_fixed_word_p0_track_block(
            mismatch_ledger,
            block_id="pixels-0-2",
            prepared=prepared_blocks[0],
            staging_plan=plan,
            certificate_binding=_FakeBinding(prepared_blocks[0]),
            background_rgb=(0.0, 0.0, 0.0),
            replay_config=SimpleNamespace(near=0.1, far=1.0),
            sample_block_size=2,
            native_ops=_FakeNativeLifecycle(),
        )
    with pytest.raises(ValueError, match="does not match the certified live track rays"):
        execute_native_fixed_word_p0_track_block(
            mismatch_ledger,
            block_id="pixels-0-2",
            prepared=prepared_blocks[0],
            staging_plan=plan,
            certificate_binding=_FakeTrainingBinding(prepared_blocks[0]),
            background_rgb=(0.0, 0.0, 0.0),
            replay_config=SimpleNamespace(near=0.1, far=1.0),
            sample_block_size=2,
            native_ops=_FakeNativeLifecycle(),
            immutable_generation_id="mismatched-ray-material-program",
        )


def test_certificate_binding_identity_is_not_replaceable_by_matching_chart_metadata() -> None:
    plan, prepared_blocks, ledger = _case()
    with pytest.raises(ValueError, match="max_in_flight_sample_blocks=1"):
        execute_native_fixed_word_p0_track_block(
            ledger,
            block_id="pixels-0-2",
            prepared=prepared_blocks[0],
            staging_plan=plan,
            certificate_binding=_FakeBinding(prepared_blocks[0]),
            background_rgb=(0.0, 0.0, 0.0),
            replay_config=SimpleNamespace(near=0.1, far=1.0),
            sample_block_size=2,
            native_ops=_FakeNativeLifecycle(),
            max_in_flight_sample_blocks=2,
        )

    wrong = _FakeBinding(prepared_blocks[1])
    with pytest.raises(ValueError, match="different compact prepared snapshot"):
        execute_native_fixed_word_p0_track_block(
            ledger,
            block_id="pixels-0-2",
            prepared=prepared_blocks[0],
            staging_plan=plan,
            certificate_binding=wrong,
            background_rgb=(0.0, 0.0, 0.0),
            replay_config=SimpleNamespace(near=0.1, far=1.0),
            sample_block_size=2,
            native_ops=_FakeNativeLifecycle(),
        )

    with pytest.raises(ValueError, match="sealed compact chart identity"):
        execute_native_fixed_word_p0_track_block(
            ledger,
            block_id="pixels-0-2",
            prepared=prepared_blocks[0],
            staging_plan=plan,
            certificate_binding=_FakeBinding(prepared_blocks[0]),
            background_rgb=(0.0, 0.0, 0.0),
            replay_config=SimpleNamespace(near=0.1, far=1.0),
            sample_block_size=2,
            native_ops=_FakeNativeLifecycle(wrong_chart_identity=True),
        )


def test_native_template_free_blocks_compare_chart_schedule_generation() -> None:
    plan, prepared_blocks, ledger = _case()
    rays = ledger.source_tensors[1]
    wrong_prepared, _ = _compiled_blocks_and_ledger(
        rays,
        global_sample_count=4,
        blocks=(("pixels-0-2", 0, 2), ("pixels-2-4", 2, 4)),
        node_counts=(2, 3),
    )
    assert wrong_prepared[0].schedule.generation_digest != ledger.schedule.generation_digest
    with pytest.raises(ValueError, match="different chart schedule generation"):
        execute_native_fixed_word_p0_track_block(
            ledger,
            block_id="pixels-0-2",
            prepared=wrong_prepared[0],
            staging_plan=plan,
            certificate_binding=_FakeBinding(wrong_prepared[0]),
            background_rgb=(0.0, 0.0, 0.0),
            replay_config=SimpleNamespace(near=0.1, far=1.0),
            sample_block_size=2,
            native_ops=_FakeNativeLifecycle(),
        )


def test_global_ledger_allows_registered_per_block_rank_schedules() -> None:
    plan, prepared_blocks, base_ledger = _case()
    alternate_blocks, alternate_ledger = _compiled_blocks_and_ledger(
        base_ledger.source_tensors[1],
        global_sample_count=4,
        blocks=(("pixels-0-2", 0, 2), ("pixels-2-4", 2, 4)),
        node_counts=(3, 4),
    )
    geometry, rays, density, color = base_ledger.source_tensors
    gradients = allocate_compact_spatial_gradient_buffers(
        site_geometry=geometry,
        site_density=density,
        site_color=color,
    )
    mixed = begin_compact_spatial_step_v2(
        schedule=base_ledger.schedule,
        site_geometry=geometry,
        ray_coefficients=rays,
        site_density=density,
        site_color=color,
        gradients=gradients,
        global_track_count=4,
        global_frame_count=4,
        loss_normalization_id="paper-logical-batch",
        expected_blocks=(("pixels-0-2", 0, 2), ("pixels-2-4", 2, 4)),
        expected_block_schedule_generations=(
            ("pixels-0-2", base_ledger.schedule.generation_digest),
            ("pixels-2-4", alternate_ledger.schedule.generation_digest),
        ),
    )
    native = _FakeNativeLifecycle()
    for block_id, prepared in (
        ("pixels-0-2", prepared_blocks[0]),
        ("pixels-2-4", alternate_blocks[1]),
    ):
        result = execute_native_fixed_word_p0_track_block(
            mixed,
            block_id=block_id,
            prepared=prepared,
            staging_plan=plan,
            certificate_binding=_FakeBinding(prepared),
            background_rgb=(0.0, 0.0, 0.0),
            replay_config=SimpleNamespace(near=0.1, far=1.0),
            sample_block_size=2,
            native_ops=native,
        )
        consume_native_fixed_word_p0_track_block_result(mixed, result)
    final = finalize_compact_spatial_step(mixed)
    assert final.accounting["distinct_expected_chart_schedule_count"] == 2
    torch.testing.assert_close(
        final.loss,
        plan.stage().targets.square().sum() / float(4 * 4 * 3),
    )


def test_material_training_reuses_prepared_topology_across_in_place_rgba_steps() -> None:
    plan, prepared_blocks, first_ledger = _case()
    bindings = tuple(_FakeTrainingBinding(prepared) for prepared in prepared_blocks)
    native = _FakeNativeLifecycle()
    validated_topologies = {}
    immutable_generation_id = "material-program-generation"

    def run_step(ledger):
        for block_id, prepared, binding in zip(
            ("pixels-0-2", "pixels-2-4"),
            prepared_blocks,
            bindings,
            strict=True,
        ):
            result = execute_native_fixed_word_p0_track_block(
                ledger,
                block_id=block_id,
                prepared=prepared,
                staging_plan=plan,
                certificate_binding=binding,
                background_rgb=(0.0, 0.0, 0.0),
                replay_config=SimpleNamespace(near=0.1, far=1.0),
                sample_block_size=2,
                native_ops=native,
                validated_topology_token=validated_topologies.get(block_id),
                immutable_generation_id=immutable_generation_id,
            )
            assert result.validated_topology_token is not None
            assert not result.geometry_vjp_executed
            assert result.grad_site_geometry_f32 is None
            assert result.sample_payload_layout == "target_only"
            assert result.peak_staged_target_bytes == 2 * 2 * 3 * 4
            assert result.peak_staged_explicit_ray_bytes == 0
            assert result.peak_staged_sample_time_bytes == 2 * 8
            validated_topologies[block_id] = result.validated_topology_token
            consume_native_fixed_word_p0_track_block_result(ledger, result)
        return finalize_compact_spatial_step(ledger)

    first = run_step(first_ledger)
    geometry, rays, density, color = first_ledger.source_tensors
    density.add_(0.125)
    color.mul_(0.75)
    with pytest.raises(ValueError, match="source world tensors changed"):
        prepared_blocks[0].assert_current()

    gradients = allocate_compact_spatial_gradient_buffers(
        site_geometry=geometry,
        site_density=density,
        site_color=color,
    )
    second_ledger = begin_compact_spatial_step_v2(
        schedule=first_ledger.schedule,
        site_geometry=geometry,
        ray_coefficients=rays,
        site_density=density,
        site_color=color,
        gradients=gradients,
        global_track_count=4,
        global_frame_count=4,
        loss_normalization_id="paper-logical-batch",
        expected_blocks=(("pixels-0-2", 0, 2), ("pixels-2-4", 2, 4)),
    )
    second = run_step(second_ledger)
    torch.testing.assert_close(second.loss, first.loss)
    torch.testing.assert_close(
        first.gradients.grad_site_geometry,
        torch.zeros_like(first.gradients.grad_site_geometry),
    )
    torch.testing.assert_close(
        second.gradients.grad_site_geometry,
        torch.zeros_like(second.gradients.grad_site_geometry),
    )
    torch.testing.assert_close(
        first.gradients.grad_site_weight,
        torch.zeros_like(first.gradients.grad_site_weight),
    )
    torch.testing.assert_close(
        second.gradients.grad_site_weight,
        torch.zeros_like(second.gradients.grad_site_weight),
    )
    torch.testing.assert_close(second.gradients.grad_site_density, first.gradients.grad_site_density)
    assert native.topology_prepare_count == 2
    assert native.world_refresh_count == 4
    assert native.world_grad_calls == []
    assert len(native.material_world_grad_calls) == 4
    assert native.full_node_vjp_count == 0
    assert native.material_node_vjp_count == 8
    assert native.full_geometry_finalize_count == 0
    assert native.material_world_grad_finalize_count == 4
    assert len(validated_topologies) == 2
    assert all(prepared.template is None for prepared in prepared_blocks)

    cached = validated_topologies["pixels-0-2"]
    for kwargs, error in (
        ({"block_id": "pixels-2-4"}, "stale or mismatched"),
        ({"certificate_binding": bindings[1]}, "stale or mismatched"),
        ({"native_ops": _FakeNativeLifecycle()}, "stale or mismatched"),
        ({"device": torch.device("meta")}, "stale or mismatched"),
        ({"immutable_generation_id": "different-generation"}, "stale or mismatched"),
    ):
        arguments = {
            "block_id": "pixels-0-2",
            "prepared": prepared_blocks[0],
            "certificate_binding": bindings[0],
            "native_ops": native,
            "device": torch.device("cpu"),
            "immutable_generation_id": immutable_generation_id,
            **kwargs,
        }
        with pytest.raises(ValueError, match=error):
            assert_native_fixed_word_p0_validated_topology_token(cached, **arguments)
