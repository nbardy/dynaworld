from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from camera import CameraSpec
from compact_lie_schedule import (
    CompactLieChartSpec,
    compact_lie_world_schedule_from_specs,
)
from compiled_transfer_adjoint import make_stable_cell_word
from material_training_step import (
    WorldFoamMaterialParameterization,
    WorldFoamMaterialTrainingBlock,
    WorldFoamNativeTopologyCachePolicy,
    bind_worldfoam_material_parameters,
    prepare_worldfoam_material_training_program,
    report_worldfoam_material_training_program_persistent_memory,
    run_worldfoam_material_training_step,
)
from native_track_adapter import estimate_native_fixed_word_p0_topology_token_resident_bytes
from powerfoam_track_staging import PowerFoamTrackStagingPlan
from powerfoam_training_data import PowerFoamRayProvider, PowerFoamTargetProvider
from prepared_track_block import prepare_worldfoam_track_block
from staged_compiled_lie_adjoint import prepare_compact_staged_lie_world_snapshot_v2
from torch_world_foam_lane2_fused_slab.certificate_binding import (
    certify_and_bind_native_fixed_word_p0_training_topology,
)


@dataclass(frozen=True)
class _ReplayConfig:
    near: float = 0.1
    far: float = 0.9


class _DifferentiableFakeNativeLifecycle:
    """CPU lifecycle double with analytic gradients for a tiny material model."""

    def __init__(self) -> None:
        self.topology_prepare_count = 0
        self.world_refresh_count = 0
        self.sample_block_count = 0
        self.full_world_grad_init_count = 0
        self.full_node_vjp_count = 0
        self.full_geometry_finalize_count = 0
        self.material_world_grad_init_count = 0
        self.material_node_vjp_count = 0
        self.material_world_grad_finalize_count = 0
        self.rgba_snapshots: list[torch.Tensor] = []

    def prepare_fixed_word_p0_topology_token(self, *tensors, **kwargs):
        binding = kwargs["certificate_binding"]
        binding.assert_native_topology(
            word_offsets_i32=tensors[0],
            word_owner_i32=tensors[1],
            word_left_incidence_i32=tensors[2],
            word_right_incidence_i32=tensors[3],
            track_incidence_offsets_i32=tensors[4],
            incidence_boundary_i32=tensors[5],
            boundary_site_pairs_i32=tensors[6],
        )
        self.topology_prepare_count += 1
        return SimpleNamespace(
            tensors=tensors,
            certificate_binding=binding,
            track_count=kwargs["track_count"],
            site_count=kwargs["site_count"],
            topology_generation_id=binding.topology_snapshot_generation,
            training_binding_digest=binding.canonical_digest,
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
        topology.certificate_binding.assert_native_world(
            sites_f32=sites_f32,
            site_rgba_f32=site_rgba_f32,
            track_ray_coeff_f32=track_ray_coeff_f32,
        )
        topology.certificate_binding.assert_replay_interval(
            near=replay_config.near,
            far=replay_config.far,
        )
        self.world_refresh_count += 1
        self.rgba_snapshots.append(site_rgba_f32.clone())
        return SimpleNamespace(
            topology=topology,
            sites_f32=sites_f32,
            site_rgba_f32=site_rgba_f32,
            track_ray_coeff_f32=track_ray_coeff_f32,
        )

    def prepare_fixed_word_p0_chart_token(self, world, compiler_node_t_f32, *, chart_index):
        chart = world.topology.certificate_binding.assert_native_chart(
            chart_index,
            compiler_node_t_f32,
        )
        return SimpleNamespace(
            world=world,
            chart_index=chart_index,
            chart_generation_id=chart.chart_digest,
            node_count=int(compiler_node_t_f32.numel()),
        )

    def fixed_word_p0_lie_world_grad_init_launch_only(self, world, **_kwargs):
        self.full_world_grad_init_count += 1
        return SimpleNamespace(
            world=world,
            grad_site_rgba_f32=torch.zeros_like(world.site_rgba_f32),
            boundary_finalized=False,
        )

    def fixed_word_p0_lie_material_world_grad_init_launch_only(self, world, **_kwargs):
        self.material_world_grad_init_count += 1
        return SimpleNamespace(
            world=world,
            grad_site_rgba_f32=torch.zeros_like(world.site_rgba_f32),
            finalized=False,
        )

    def prepare_fixed_word_p0_sample_state_token(self, chart, **kwargs):
        return SimpleNamespace(
            chart=chart,
            loss_f32=torch.zeros((), dtype=torch.float32),
            grad_site_rgba_f32=torch.zeros_like(chart.world.site_rgba_f32),
            global_loss_scale=1.0 / float(kwargs["global_loss_element_count"]),
        )

    def prepare_fixed_word_p0_sample_block_token(
        self,
        sample_state,
        target_rgb_f32,
        background_rgb_f32,
        **_kwargs,
    ):
        self.sample_block_count += 1
        weight_result = sample_state.chart.world.topology.certificate_binding.sample_to_node_weight_result(
            sample_state.chart.chart_index,
            _kwargs["sample_t_f64"],
        )
        return SimpleNamespace(
            sample_state=sample_state,
            target_rgb_f32=target_rgb_f32,
            background_rgb_f32=background_rgb_f32,
            sample_weight_evaluation=weight_result.evaluation,
            sample_weight_linear_interactions=weight_result.linear_weight_interactions,
            sample_weight_dense_fallback_interactions=(weight_result.dense_fallback_interactions),
            sample_weight_exact_node_rows=weight_result.exact_node_row_count,
            sample_weight_dense_fallback_rows=weight_result.dense_fallback_row_count,
        )

    def fixed_word_p0_lie_sample_accumulate_loss_only_launch_only(
        self,
        sample_block,
        sample_state,
    ):
        rgba = sample_state.chart.world.site_rgba_f32
        site_count = int(rgba.shape[0])
        predicted_rgb = rgba[:, :3].mean(dim=0) + rgba[:, 3].mean()
        prediction = predicted_rgb.view(1, 1, 3).expand_as(sample_block.target_rgb_f32)
        residual = prediction - sample_block.target_rgb_f32
        sample_state.loss_f32.add_(residual.square().sum() * sample_state.global_loss_scale)
        grad_prediction = 2.0 * residual * sample_state.global_loss_scale
        sample_state.grad_site_rgba_f32[:, :3].add_(grad_prediction.sum(dim=(0, 1)).view(1, 3) / float(site_count))
        sample_state.grad_site_rgba_f32[:, 3].add_(grad_prediction.sum() / float(site_count))

    def fixed_word_p0_lie_node_vjp_accumulate_launch_only(self, _chart, sample_state, world_grad):
        self.full_node_vjp_count += 1
        world_grad.grad_site_rgba_f32.add_(sample_state.grad_site_rgba_f32)

    def fixed_word_p0_lie_material_node_vjp_accumulate_launch_only(
        self,
        _chart,
        sample_state,
        world_grad,
    ):
        self.material_node_vjp_count += 1
        world_grad.grad_site_rgba_f32.add_(sample_state.grad_site_rgba_f32)

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
        return torch.zeros(
            (world_grad.world.topology.site_count, 5),
            dtype=torch.float32,
        )


def _camera() -> CameraSpec:
    return CameraSpec(
        fx=1_000.0,
        fy=1_000.0,
        cx=1.5,
        cy=0.0,
        camera_to_world=torch.eye(4, dtype=torch.float32),
    )


def _cache_policy(
    *,
    entries: int = 2,
    cached_bytes: int = 1 << 20,
    live_bytes: int = 1 << 20,
) -> WorldFoamNativeTopologyCachePolicy:
    return WorldFoamNativeTopologyCachePolicy(
        max_cached_entries=entries,
        max_cached_tensor_bytes=cached_bytes,
        max_live_topology_tensor_bytes=live_bytes,
    )


def _training_case():
    target_rgb = torch.tensor([0.4, 0.6, 0.8], dtype=torch.float32)
    frames = target_rgb.view(1, 1, 3, 1, 1).expand(1, 4, 3, 1, 4).clone()
    target_provider = PowerFoamTargetProvider.from_resident_frames(
        frames,
        device=torch.device("cpu"),
    )
    ray_provider = PowerFoamRayProvider(
        (tuple(_camera() for _ in range(4)),),
        height=1,
        width=4,
        device=torch.device("cpu"),
    )
    plan = PowerFoamTrackStagingPlan(
        target_provider,
        ray_provider,
        torch.arange(4),
        torch.tensor([3, 0, 2, 1]),
    )
    initial_stage = plan.stage(
        track_start=0,
        track_end=4,
        sample_start=0,
        sample_end=1,
        require_affine_ray_program=True,
    )
    assert initial_stage.affine_ray_program is not None
    ray_coefficients = initial_stage.affine_ray_program.coefficients[0].clone()
    site_geometry = torch.tensor(
        [
            [0.0, 0.0, 0.2, 0.0, 0.0],
            [0.0, 0.0, 0.6, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    pairs = torch.tensor([[0, 1]], dtype=torch.int64)
    words = tuple(make_stable_cell_word([0, 1], [-1, 0], [0, -2]) for _ in range(4))
    schedule = compact_lie_world_schedule_from_specs(
        (
            CompactLieChartSpec(
                t_min=0.0,
                t_max=1.0,
                near=0.1,
                far=0.9,
                node_count=4,
            ),
        ),
        global_track_count=4,
        selection_provenance="unit-test-fixed-rank-owner-certified-v1",
    )
    parameterization = WorldFoamMaterialParameterization()
    raw_density = torch.nn.Parameter(torch.zeros((2,), dtype=torch.float32))
    raw_color = torch.nn.Parameter(torch.zeros((2, 3), dtype=torch.float32))
    initial_density = torch.empty_like(raw_density, requires_grad=False)
    initial_color = torch.empty_like(raw_color, requires_grad=False)
    parameterization.decode_density_(initial_density, raw_density)
    parameterization.decode_color_(initial_color, raw_color)
    blocks = []
    for block_id, track_start, track_end in (
        ("pixels-0-2", 0, 2),
        ("pixels-2-4", 2, 4),
    ):
        topology = prepare_worldfoam_track_block(
            words,
            pairs,
            site_count=2,
            track_start=track_start,
            track_end=track_end,
        )
        initial_prepared = prepare_compact_staged_lie_world_snapshot_v2(
            schedule,
            topology,
            site_geometry=site_geometry,
            ray_coefficients=ray_coefficients,
            site_density=initial_density,
            site_color=initial_color,
        )
        owner_binding = certify_and_bind_native_fixed_word_p0_training_topology(
            initial_prepared,
            max_split_depth=8,
            max_leaf_count=256,
            max_work_units=100_000,
            arithmetic_fraction_bits=64,
        )
        blocks.append(
            WorldFoamMaterialTrainingBlock(
                block_id=block_id,
                topology=topology,
                schedule=schedule,
                owner_binding=owner_binding,
            )
        )
    program = prepare_worldfoam_material_training_program(
        staging_plan=plan,
        blocks=tuple(blocks),
        site_geometry=site_geometry,
        ray_coefficients=ray_coefficients,
        background_rgb=(0.0, 0.0, 0.0),
        replay_config=_ReplayConfig(),
        sample_block_size=2,
        parameterization=parameterization,
    )
    return program, raw_density, raw_color


def _unique_storage_bytes(tensors: tuple[torch.Tensor, ...]) -> int:
    storages = {
        (
            str(tensor.device),
            int(tensor.untyped_storage().data_ptr()),
            int(tensor.untyped_storage().nbytes()),
        ): int(tensor.untyped_storage().nbytes())
        for tensor in tensors
    }
    return sum(storages.values())


def test_persistent_memory_report_measures_the_actual_lightweight_program() -> None:
    program, _, _ = _training_case()

    report = report_worldfoam_material_training_program_persistent_memory(program)

    global_tensors = [
        program.site_geometry,
        program.ray_coefficients,
        program.staging_plan.pixel_indices,
        program.staging_plan.sample_indices,
        program.staging_plan.sample_times,
    ]
    for cameras in program.staging_plan.ray_provider.cameras:
        for camera in cameras:
            global_tensors.extend(
                value
                for value in (
                    camera.fx,
                    camera.fy,
                    camera.cx,
                    camera.cy,
                    camera.camera_to_world,
                    camera.distortion,
                )
                if torch.is_tensor(value)
            )
    topology_names = (
        "source_track_ids",
        "source_boundary_ids",
        "source_site_ids",
        "word_offsets_i32",
        "word_owner_i32",
        "word_left_incidence_i32",
        "word_right_incidence_i32",
        "track_incidence_offsets_i32",
        "incidence_boundary_i32",
        "boundary_site_pairs_i32",
    )
    topology_tensors = tuple(getattr(block.topology, name) for block in program.blocks for name in topology_names)
    binding_tensors = tuple(
        tensor
        for block in program.blocks
        for tensor in (
            *block.owner_binding._bound_tensors,
            *block.owner_binding._sample_barycentric_weights,
        )
    )
    schedule = program.blocks[0].schedule
    schedule_tensors = tuple(
        tensor
        for chart in schedule.charts
        for tensor in (chart.node_times, chart.fit_matrix, chart.barycentric_weights)
    )
    all_program_tensors = (
        *global_tensors,
        *topology_tensors,
        *binding_tensors,
        *schedule_tensors,
    )
    target_residency = program.staging_plan.target_provider.residency()

    assert report.block_count == 2
    assert report.unique_schedule_count == 1
    assert program.blocks[0].schedule is program.blocks[1].schedule
    assert report.program_global_model_staging_tensor_bytes == _unique_storage_bytes(tuple(global_tensors))
    assert report.retained_block_topology_tensor_bytes == _unique_storage_bytes(topology_tensors)
    assert report.retained_training_binding_private_tensor_bytes == _unique_storage_bytes(binding_tensors)
    assert report.unique_schedule_tensor_bytes == _unique_storage_bytes(schedule_tensors)
    assert report.unique_schedule_tensor_bytes == schedule.resident_bytes
    assert report.unique_schedule_tensor_bytes * 2 == sum(block.schedule.resident_bytes for block in program.blocks)
    assert report.unique_program_tensor_storage_bytes == _unique_storage_bytes(all_program_tensors)
    independent_category_sum = (
        report.program_global_model_staging_tensor_bytes
        + report.retained_block_topology_tensor_bytes
        + report.retained_training_binding_private_tensor_bytes
        + report.unique_schedule_tensor_bytes
    )
    assert report.cross_category_shared_tensor_storage_bytes == (
        independent_category_sum - report.unique_program_tensor_storage_bytes
    )
    assert report.target_provider_residency_available
    assert report.target_provider_residency == target_residency
    assert report.target_provider_resident_bytes == target_residency["resident_bytes"]
    assert report.total_source_level_persistent_bytes == (
        report.unique_program_tensor_storage_bytes + target_residency["resident_bytes"]
    )
    assert report.tensor_storage_deduplicated
    assert report.retained_compiled_cpu_atlas_block_count == 0
    assert report.excluded_byte_classes == (
        "python_objects",
        "json_strings",
        "allocator_metadata_and_reservations",
        "optimizer_state",
        "native_transient_buffers",
    )


def test_two_material_steps_decrease_loss_without_retaining_compiled_cpu_atlases() -> None:
    program, raw_density, raw_color = _training_case()
    geometry_before = program.site_geometry.clone()
    rays_before = program.ray_coefficients.clone()
    immutable_ids = tuple((id(block.topology), id(block.schedule), id(block.owner_binding)) for block in program.blocks)
    session = bind_worldfoam_material_parameters(
        program,
        raw_density=raw_density,
        raw_color=raw_color,
        native_topology_cache_policy=_cache_policy(),
    )
    assert not hasattr(session, "prepared_blocks")
    gradient_pointers = session.gradient_storage_pointers
    density_pointer = raw_density.untyped_storage().data_ptr()
    color_pointer = raw_color.untyped_storage().data_ptr()
    initial_physical_density = session.site_density.clone()
    initial_physical_color = session.site_color.clone()
    native = _DifferentiableFakeNativeLifecycle()
    optimizer = torch.optim.SGD([raw_density, raw_color], lr=0.1)

    first = run_worldfoam_material_training_step(session, optimizer, native_ops=native)
    first_density_grad = raw_density.grad.clone()
    first_color_grad = raw_color.grad.clone()
    second = run_worldfoam_material_training_step(session, optimizer, native_ops=native)

    assert second.loss < first.loss
    initial_prediction = float(initial_physical_density.mean() + initial_physical_color[:, 0].mean())
    residual = torch.tensor([initial_prediction - 0.4, initial_prediction - 0.6, initial_prediction - 0.8])
    assert first.loss == pytest.approx(float(residual.square().mean()))
    assert first.step_index == 1
    assert second.step_index == 2
    assert first.immutable_generation_id == second.immutable_generation_id == program.immutable_generation_id
    assert first.accounting["global_loss_element_count"] == 4 * 4 * 3
    assert second.accounting["global_loss_element_count"] == 4 * 4 * 3
    assert first.accounting["sample_block_count"] == 4
    assert second.accounting["sample_block_count"] == 4
    assert first.accounting["sample_payload_layout"] == "target_only"
    assert second.accounting["sample_payload_layout"] == "target_only"
    assert first.accounting["peak_staged_target_bytes"] == 2 * 2 * 3 * 4
    assert second.accounting["peak_staged_target_bytes"] == 2 * 2 * 3 * 4
    assert first.accounting["peak_staged_explicit_ray_bytes"] == 0
    assert second.accounting["peak_staged_explicit_ray_bytes"] == 0
    assert first.accounting["peak_staged_sample_time_bytes"] == 2 * 8
    assert second.accounting["peak_staged_sample_time_bytes"] == 2 * 8
    assert first.accounting["chart_or_global_sample_time_clone_bytes"] == 0
    assert second.accounting["chart_or_global_sample_time_clone_bytes"] == 0
    assert first.accounting["explicit_ray_staging_omitted"] is True
    assert second.accounting["explicit_ray_staging_omitted"] is True
    assert (
        first.accounting["sample_weight_evaluation"]
        == second.accounting["sample_weight_evaluation"]
        == "verified_fit_derived_second_form_barycentric"
    )
    expected_weight_interactions = (
        len(program.blocks) * program.global_sample_count * program.blocks[0].schedule.charts[0].node_count
    )
    expected_material_reverse_bytes_omitted = sum(
        16 * block.topology.incidence_count + 20 * block.topology.boundary_count + 20 * block.topology.site_count
        for block in program.blocks
    )
    assert first.accounting["sample_weight_linear_interactions"] == expected_weight_interactions
    assert second.accounting["sample_weight_linear_interactions"] == expected_weight_interactions
    assert first.accounting["sample_weight_common_path_complexity"] == "O(spatial_blocks*F*J)"
    assert second.accounting["sample_weight_common_path_complexity"] == "O(spatial_blocks*F*J)"
    assert first.accounting["sample_weight_dense_fallback_complexity"] == ("O(spatial_blocks*F_fallback*J^2)")
    assert second.accounting["sample_weight_spatial_block_count"] == len(program.blocks)
    assert first.accounting["sample_weight_dense_fallback_interactions"] == 0
    assert second.accounting["sample_weight_dense_fallback_interactions"] == 0
    assert first.accounting["cpu_compact_atlas_compile_count_per_step"] == 0
    assert second.accounting["cpu_compact_atlas_compile_count_per_step"] == 0
    assert first.accounting["prepared_block_compile_count_per_step"] == 0
    assert second.accounting["prepared_block_compile_count_per_step"] == 0
    assert first.accounting["native_topology_prepare_count"] == 2
    assert second.accounting["native_topology_prepare_count"] == 0
    assert first.accounting["native_topology_cache_hit_count"] == 0
    assert second.accounting["native_topology_cache_hit_count"] == 2
    assert first.accounting["native_topology_cache_entry_count"] == 2
    assert second.accounting["native_topology_cache_entry_count"] == 2
    assert first.accounting["native_topology_cache_bounded_by_spatial_block_count"] is False
    assert first.accounting["native_topology_cache_bounded_by_explicit_policy"] is True
    assert first.accounting["native_topology_cache_max_entries"] == 2
    assert first.accounting["native_topology_cache_max_tensor_bytes"] == 1 << 20
    assert first.accounting["native_topology_max_live_tensor_bytes"] == 1 << 20
    assert first.accounting["native_topology_cache_resident_tensor_bytes"] > 0
    assert first.accounting["native_topology_cache_resident_tensor_bytes"] <= 1 << 20
    assert first.accounting["native_topology_peak_live_tensor_bytes"] <= 1 << 20
    assert first.accounting["native_topology_peak_actual_token_tensor_bytes"] <= first.accounting[
        "native_topology_peak_preflight_token_tensor_bytes"
    ]
    assert first.accounting["binding_construction_compiled_snapshot_count_lower_bound"] == 2
    assert second.accounting["binding_construction_compiled_snapshot_count_lower_bound"] == 2
    assert first.accounting["retained_compiled_cpu_atlas_block_count"] == 0
    assert second.accounting["retained_compiled_cpu_atlas_block_count"] == 0
    assert first.accounting["lightweight_topology_schedule_block_count"] == 2
    assert second.accounting["lightweight_topology_schedule_block_count"] == 2
    assert first.accounting["density_parameterization"] == "softplus"
    assert first.accounting["color_parameterization"] == "sigmoid"
    assert first.accounting["manual_parameter_chain_rule"] is True
    assert first.accounting["geometry_vjp_executed"] is False
    assert second.accounting["geometry_vjp_executed"] is False
    assert (
        first.accounting["material_only_reverse_tensor_bytes_omitted"]
        == second.accounting["material_only_reverse_tensor_bytes_omitted"]
        == expected_material_reverse_bytes_omitted
    )
    assert first.frozen_geometry_gradient_norm == 0.0
    assert second.frozen_geometry_gradient_norm == 0.0
    assert first.frozen_weight_gradient_norm == 0.0
    assert second.frozen_weight_gradient_norm == 0.0
    assert not first.paper_evidence_eligible
    assert not first.transfer_jacobian_certified
    assert not first.approximation_error_certified
    assert first.binding_mode == "training_owner_topology_only"
    session.assert_current()
    assert (
        tuple((id(block.topology), id(block.schedule), id(block.owner_binding)) for block in program.blocks)
        == immutable_ids
    )
    assert session.gradient_storage_pointers == gradient_pointers
    assert raw_density.untyped_storage().data_ptr() == density_pointer
    assert raw_color.untyped_storage().data_ptr() == color_pointer
    assert raw_density.grad is session.raw_density_gradient
    assert raw_color.grad is session.raw_color_gradient
    assert bool(torch.any(first_density_grad != 0.0))
    assert bool(torch.any(first_color_grad != 0.0))
    physical_density_grad = torch.full((2,), float(residual.sum() / 3.0))
    physical_color_grad = (residual / 3.0).view(1, 3).expand(2, 3)
    torch.testing.assert_close(first_density_grad, physical_density_grad * 0.5)
    torch.testing.assert_close(
        first_color_grad,
        physical_color_grad * 0.25,
    )
    assert native.topology_prepare_count == 2
    assert native.world_refresh_count == 4
    assert native.sample_block_count == 8
    assert native.material_world_grad_init_count == 4
    assert native.material_node_vjp_count == 4
    assert native.material_world_grad_finalize_count == 4
    assert native.full_world_grad_init_count == 0
    assert native.full_node_vjp_count == 0
    assert native.full_geometry_finalize_count == 0
    assert torch.count_nonzero(session.gradients.grad_site_geometry).item() == 0
    assert torch.count_nonzero(session.gradients.grad_site_weight).item() == 0
    assert not torch.equal(native.rgba_snapshots[0], native.rgba_snapshots[2])
    assert all(bool(torch.all(snapshot[:, 3] >= 0.0).item()) for snapshot in native.rgba_snapshots)
    assert all(
        bool(torch.all((snapshot[:, :3] >= 0.0) & (snapshot[:, :3] <= 1.0)).item())
        for snapshot in native.rgba_snapshots
    )
    torch.testing.assert_close(session.site_density, F.softplus(raw_density))
    torch.testing.assert_close(session.site_color, torch.sigmoid(raw_color))
    torch.testing.assert_close(program.site_geometry, geometry_before)
    torch.testing.assert_close(program.ray_coefficients, rays_before)
    assert not program.site_geometry.requires_grad
    assert not program.ray_coefficients.requires_grad


def test_material_topology_cache_rejects_a_stale_validated_token() -> None:
    program, raw_density, raw_color = _training_case()
    session = bind_worldfoam_material_parameters(
        program,
        raw_density=raw_density,
        raw_color=raw_color,
        native_topology_cache_policy=_cache_policy(),
    )
    native = _DifferentiableFakeNativeLifecycle()
    optimizer = torch.optim.SGD([raw_density, raw_color], lr=0.1)
    run_worldfoam_material_training_step(session, optimizer, native_ops=native)
    key, cached = next(iter(session.native_topology_token_cache.items()))
    session.native_topology_token_cache[key] = replace(
        cached,
        cache_key=replace(cached.cache_key, binding_digest="0" * 64),
    )

    with pytest.raises(ValueError, match="cache identity was corrupted"):
        run_worldfoam_material_training_step(session, optimizer, native_ops=native)
    assert native.topology_prepare_count == 2


def test_material_topology_cache_replaces_each_block_when_native_ops_changes() -> None:
    program, raw_density, raw_color = _training_case()
    session = bind_worldfoam_material_parameters(
        program,
        raw_density=raw_density,
        raw_color=raw_color,
        native_topology_cache_policy=_cache_policy(),
    )
    optimizer = torch.optim.SGD([raw_density, raw_color], lr=0.1)
    first_native = _DifferentiableFakeNativeLifecycle()
    second_native = _DifferentiableFakeNativeLifecycle()

    run_worldfoam_material_training_step(session, optimizer, native_ops=first_native)
    second_result = run_worldfoam_material_training_step(
        session,
        optimizer,
        native_ops=second_native,
    )

    assert first_native.topology_prepare_count == len(program.blocks)
    assert second_native.topology_prepare_count == len(program.blocks)
    assert second_result.accounting["native_topology_cache_hit_count"] == 0
    assert second_result.accounting["native_topology_cache_miss_count"] == len(program.blocks)
    assert len(session.native_topology_token_cache) == len(program.blocks)
    assert {key.native_ops_identity for key in session.native_topology_token_cache} == {id(second_native)}
    assert {id(cached.native_ops) for cached in session.native_topology_token_cache.values()} == {id(second_native)}


def test_material_topology_cache_streams_one_live_token_under_explicit_budget() -> None:
    program, raw_density, raw_color = _training_case()
    one_token_budget = max(
        estimate_native_fixed_word_p0_topology_token_resident_bytes(block)
        for block in program.blocks
    )
    session = bind_worldfoam_material_parameters(
        program,
        raw_density=raw_density,
        raw_color=raw_color,
        native_topology_cache_policy=_cache_policy(
            entries=1,
            cached_bytes=1 << 20,
            live_bytes=one_token_budget,
        ),
    )
    native = _DifferentiableFakeNativeLifecycle()
    optimizer = torch.optim.SGD([raw_density, raw_color], lr=0.1)

    first = run_worldfoam_material_training_step(session, optimizer, native_ops=native)
    second = run_worldfoam_material_training_step(session, optimizer, native_ops=native)

    assert len(session.native_topology_token_cache) == 1
    assert first.accounting["native_topology_cache_entry_count"] == 1
    assert second.accounting["native_topology_cache_entry_count"] == 1
    assert first.accounting["native_topology_cache_eviction_count"] == 1
    assert second.accounting["native_topology_cache_eviction_count"] == 2
    assert first.accounting["native_topology_cache_skip_count"] == 0
    assert second.accounting["native_topology_cache_skip_count"] == 0
    assert second.accounting["native_topology_peak_live_tensor_bytes"] <= one_token_budget
    assert second.accounting["native_topology_cache_resident_tensor_bytes"] <= one_token_budget
    assert native.topology_prepare_count == 2 * len(program.blocks)
    assert native.material_node_vjp_count == 2 * len(program.blocks)
    session.assert_current()


def test_material_topology_budget_rejects_oversized_block_before_native_prepare() -> None:
    program, raw_density, raw_color = _training_case()
    minimum_token_preflight = min(
        estimate_native_fixed_word_p0_topology_token_resident_bytes(block)
        for block in program.blocks
    )
    session = bind_worldfoam_material_parameters(
        program,
        raw_density=raw_density,
        raw_color=raw_color,
        native_topology_cache_policy=_cache_policy(
            entries=0,
            cached_bytes=0,
            live_bytes=minimum_token_preflight - 1,
        ),
    )
    native = _DifferentiableFakeNativeLifecycle()
    optimizer = torch.optim.SGD([raw_density, raw_color], lr=0.1)
    density_before = raw_density.detach().clone()
    color_before = raw_color.detach().clone()

    with pytest.raises(ValueError, match="preflight exceeds max_live_topology_tensor_bytes"):
        run_worldfoam_material_training_step(session, optimizer, native_ops=native)

    assert native.topology_prepare_count == 0
    assert session.steps_completed == 0
    torch.testing.assert_close(raw_density, density_before)
    torch.testing.assert_close(raw_color, color_before)


def test_material_step_rejects_optimizer_parameters_outside_density_and_color() -> None:
    program, raw_density, raw_color = _training_case()
    session = bind_worldfoam_material_parameters(
        program,
        raw_density=raw_density,
        raw_color=raw_color,
        native_topology_cache_policy=_cache_policy(),
    )
    unrelated = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.SGD([raw_density, raw_color, unrelated], lr=0.1)
    with pytest.raises(ValueError, match="exactly the caller raw-density and raw-color"):
        run_worldfoam_material_training_step(
            session,
            optimizer,
            native_ops=_DifferentiableFakeNativeLifecycle(),
        )


def test_softplus_sigmoid_manual_vjps_match_autograd_across_transform_branches() -> None:
    parameterization = WorldFoamMaterialParameterization(
        density_beta=1.3,
        density_threshold=10.0,
        minimum_density=0.01,
    )
    raw_density = torch.tensor([-20.0, -1.0, 0.0, 5.0, 12.0], dtype=torch.float64)
    raw_color = torch.tensor(
        [[-8.0, -1.0, 0.0], [1.0, 4.0, 8.0]],
        dtype=torch.float64,
    )
    density_bar = torch.tensor([0.3, -0.2, 0.7, -0.5, 0.9], dtype=torch.float64)
    color_bar = torch.tensor(
        [[0.4, -0.3, 0.2], [-0.1, 0.6, -0.8]],
        dtype=torch.float64,
    )
    physical_density = torch.empty_like(raw_density)
    physical_color = torch.empty_like(raw_color)
    manual_density_bar = torch.empty_like(raw_density)
    manual_color_bar = torch.empty_like(raw_color)
    parameterization.decode_density_(physical_density, raw_density)
    parameterization.decode_color_(physical_color, raw_color)
    parameterization.density_vjp_(manual_density_bar, raw_density, density_bar)
    parameterization.color_vjp_(manual_color_bar, physical_color, color_bar)

    density_reference = raw_density.clone().requires_grad_()
    density_loss = (
        (
            F.softplus(
                density_reference,
                beta=parameterization.density_beta,
                threshold=parameterization.density_threshold,
            )
            + parameterization.minimum_density
        )
        * density_bar
    ).sum()
    density_loss.backward()
    color_reference = raw_color.clone().requires_grad_()
    color_loss = (torch.sigmoid(color_reference) * color_bar).sum()
    color_loss.backward()

    torch.testing.assert_close(manual_density_bar, density_reference.grad)
    torch.testing.assert_close(manual_color_bar, color_reference.grad)
    assert bool(torch.all(physical_density >= parameterization.minimum_density).item())
    assert bool(torch.all((physical_color >= 0.0) & (physical_color <= 1.0)).item())
