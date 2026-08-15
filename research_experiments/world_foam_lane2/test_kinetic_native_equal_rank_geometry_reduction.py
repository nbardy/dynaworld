from __future__ import annotations

import inspect
from dataclasses import dataclass, replace

import kinetic_native_equal_rank_geometry_reduction as geometry_reduction
import pytest
import test_kinetic_native_equal_rank_runtime_adapter as runtime_fixture
import torch
from kinetic_native_equal_rank_geometry_reduction import (
    kinetic_native_equal_rank_vjp_provenance_id,
    reduce_kinetic_native_equal_rank_geometry_vjp,
)
from kinetic_native_equal_rank_runtime_adapter import (
    execute_kinetic_native_equal_rank_node_vjp,
)
from kinetic_native_equal_rank_sparse_geometry_reduction import (
    preflight_kinetic_native_equal_rank_sparse_geometry_reduction_memory,
    reduce_kinetic_native_equal_rank_sparse_geometry_vjp,
)
from kinetic_stable_stratum_vjp import (
    kinetic_p0_node_physical_length_geometry_vjp,
    make_frozen_kinetic_owner_word,
)
from paper_kinetic_ragged_sample_plan import (
    prepare_paper_kinetic_row_ragged_sampler,
)


@dataclass
class _Fence:
    calls: int = 0

    def __call__(self) -> None:
        self.calls += 1


def _full_vjp_case(*, view_index: int = 0):
    sources, lowering, payloads, global_rgba = runtime_fixture._case(maximum_rows_per_block=2)
    assert len(payloads) == 1 and payloads[0].row_count == 2
    runtime, world = runtime_fixture._runtime_world(
        payloads[0],
        lowering,
        sources,
        global_rgba,
        runtime_fixture._FakeNativeOps(),
    )
    grad_node = torch.linspace(
        -0.53,
        0.71,
        runtime.row_count * runtime.node_count * 4,
        dtype=torch.float32,
    ).reshape(runtime.row_count, runtime.node_count, 4)
    native_vjp = execute_kinetic_native_equal_rank_node_vjp(
        world,
        grad_node,
        compact_grad_site_rgba_f32=torch.zeros_like(world.compact_site_rgba_f32),
    )
    sampler = prepare_paper_kinetic_row_ragged_sampler(
        view_index=view_index,
        lowering=lowering,
        sources=sources,
    )
    return native_vjp, sampler


def _reduce(
    native_vjp,
    sampler,
    fence: _Fence,
    *,
    byte_cap: int = 10_000_000,
    include_ray_gradients: bool = True,
):
    return reduce_kinetic_native_equal_rank_geometry_vjp(
        native_vjp,
        sampler,
        expected_native_vjp_provenance_id=(kinetic_native_equal_rank_vjp_provenance_id(native_vjp)),
        device_completion_fence=fence,
        device_completion_fence_provenance="cpu-fake-native-completion-fence-v1",
        maximum_bridge_visible_peak_logical_tensor_bytes=byte_cap,
        include_ray_gradients=include_ray_gradients,
    )


def _reduce_sparse(
    native_vjp,
    sampler,
    fence: _Fence,
    *,
    byte_cap: int = 10_000_000,
    include_ray_gradients: bool = True,
):
    return reduce_kinetic_native_equal_rank_sparse_geometry_vjp(
        native_vjp,
        sampler,
        expected_native_vjp_provenance_id=(
            kinetic_native_equal_rank_vjp_provenance_id(native_vjp)
        ),
        device_completion_fence=fence,
        device_completion_fence_provenance="cpu-fake-native-completion-fence-v1",
        maximum_bridge_visible_peak_logical_tensor_bytes=byte_cap,
        include_ray_gradients=include_ray_gradients,
    )


def _independent_row_sum(native_vjp, sampler):
    block = native_vjp.world.runtime.payload.block
    bindings = tuple(
        sorted(
            (row for row in sampler.rows if row.native_block_generation_digest == block.generation_digest),
            key=lambda row: row.native_local_row_index,
        )
    )
    rows_by_index = {row.global_row_index: row for row in sampler.lowering.rows}
    row_specs = tuple(rows_by_index[index] for index in block.global_row_indices)
    sites = bindings[0].program.binding.sites
    positions = torch.zeros_like(sites.positions0)
    velocities = torch.zeros_like(sites.velocities)
    weights = torch.zeros_like(sites.weight_coefficients)
    rays_by_track: dict[int, torch.Tensor] = {}
    row_results = []
    word_start = 0
    for binding, row in zip(bindings, row_specs, strict=True):
        word_end = word_start + row.word_count
        chart = binding.program.charts[binding.chart_index]
        topology_chart = binding.source.lowering.charts[binding.chart_index]
        result = kinetic_p0_node_physical_length_geometry_vjp(
            sites,
            binding.program.binding.ray_coefficients,
            chart.schedule.node_times,
            (make_frozen_kinetic_owner_word(row.owner_word),),
            native_vjp.grad_node_physical_length_f32[:, word_start:word_end].to(dtype=torch.float64),
            near=row.near,
            far=row.far,
            continuous_topology_certificate_id=(topology_chart.owner_topology_certificate_digest),
            node_physical_length_cotangent_provenance_id="independent-test-row",
        )
        positions.add_(result.grad_positions0)
        velocities.add_(result.grad_velocities)
        weights.add_(result.grad_weight_coefficients)
        rays_by_track.setdefault(
            binding.track_id,
            torch.zeros_like(result.grad_ray_coefficients),
        ).add_(result.grad_ray_coefficients)
        row_results.append(result)
        word_start = word_end
    return (
        bindings,
        row_specs,
        row_results,
        positions,
        velocities,
        weights,
        torch.stack([rays_by_track[track_id] for track_id in sorted(rays_by_track)]),
    )


def test_multirow_multichart_geometry_reduction_sums_repeated_sites_and_track() -> None:
    native_vjp, sampler = _full_vjp_case(view_index=3)
    (
        bindings,
        row_specs,
        row_results,
        expected_positions,
        expected_velocities,
        expected_weights,
        expected_rays,
    ) = _independent_row_sum(native_vjp, sampler)
    assert tuple(binding.track_id for binding in bindings) == (7, 7)
    assert tuple(binding.chart_index for binding in bindings) == (0, 1)
    assert all(0 in row.owner_word for row in row_specs)

    fence = _Fence()
    result = _reduce(native_vjp, sampler, fence)

    assert fence.calls == result.device_completion_fence_call_count == 1
    assert result.row_geometry_vjp_call_count == result.row_count == 2
    assert result.view_index == sampler.view_index == 3
    assert result.track_ids == (7,)
    assert result.ray_bar_keys == ((3, 7),)
    torch.testing.assert_close(result.grad_positions0_f64, expected_positions)
    torch.testing.assert_close(result.grad_velocities_f64, expected_velocities)
    torch.testing.assert_close(result.grad_weight_coefficients_f64, expected_weights)
    torch.testing.assert_close(result.grad_track_ray_coefficients_f64, expected_rays)
    torch.testing.assert_close(
        result.grad_track_ray_coefficients_f64[0],
        row_results[0].grad_ray_coefficients + row_results[1].grad_ray_coefficients,
    )
    assert not hasattr(result, "native_vjp")
    assert not hasattr(result, "sampler")
    assert not result.native_result_retained
    assert not result.native_world_or_runtime_retained
    assert not result.row_scratch_retained
    assert not result.frame_or_sample_axis_retained
    assert not result.target_prediction_or_material_retained
    assert not result.real_device_completion_fence_semantics_verified
    assert not result.upstream_request_receipt_bound
    assert not result.allocator_peak_measured
    assert result.differentiable_word_reverse_interactions == (result.node_count * result.word_count)
    assert result.dense_global_site_accumulation_elements == (
        result.row_count * result.site_count * (6 + result.weight_coefficient_count)
    )
    assert result.all_site_owner_validation_evaluations == (
        result.node_count * 3 * result.word_count * (result.site_count - 1)
    )
    assert result.accounting["view_index"] == 3
    assert result.accounting["ray_bar_key_kind"] == "(view_index, track_id)"
    assert not result.accounting["aggregate_reverse_runtime_scaling_claimed"]
    assert "reverse_scaling" not in result.accounting
    result.assert_current()
    with pytest.raises(ValueError, match="execution/memory contract changed"):
        replace(result, view_index=4).assert_current()
    with pytest.raises(ValueError, match="execution/memory contract changed"):
        replace(result, ray_bar_keys=((3, 8),)).assert_current()


def test_fixed_camera_geometry_reduction_omits_compact_ray_bar_storage() -> None:
    native_vjp, sampler = _full_vjp_case(view_index=3)
    trainable_camera = _reduce(native_vjp, sampler, _Fence())
    fixed_camera = _reduce(
        native_vjp,
        sampler,
        _Fence(),
        include_ray_gradients=False,
    )

    torch.testing.assert_close(
        fixed_camera.grad_positions0_f64,
        trainable_camera.grad_positions0_f64,
    )
    torch.testing.assert_close(
        fixed_camera.grad_velocities_f64,
        trainable_camera.grad_velocities_f64,
    )
    torch.testing.assert_close(
        fixed_camera.grad_weight_coefficients_f64,
        trainable_camera.grad_weight_coefficients_f64,
    )
    assert fixed_camera.ray_gradients_included is False
    assert fixed_camera.ray_bar_keys == ()
    assert tuple(fixed_camera.grad_track_ray_coefficients_f64.shape) == (0, 12)
    assert fixed_camera.grad_track_ray_coefficients_f64.numel() == 0
    assert fixed_camera.memory.grad_track_ray_coefficients_tensor_bytes == 0
    assert fixed_camera.output_parameter_bar_tensor_bytes < (
        trainable_camera.output_parameter_bar_tensor_bytes
    )
    assert fixed_camera.memory.bridge_visible_peak_logical_tensor_bytes < (
        trainable_camera.memory.bridge_visible_peak_logical_tensor_bytes
    )
    assert fixed_camera.accounting["ray_gradients_included"] is False
    assert fixed_camera.accounting["ray_bar_key_kind"] == "disabled/fixed_camera"
    fixed_camera.assert_current()


def test_sparse_geometry_public_preflight_matches_result_memory() -> None:
    native_vjp, sampler = _full_vjp_case(view_index=3)
    block = native_vjp.world.runtime.payload.block
    rows_by_index = {
        row.global_row_index: row for row in sampler.lowering.rows
    }
    rows = tuple(rows_by_index[index] for index in block.global_row_indices)
    bindings = tuple(
        row
        for row in sampler.rows
        if row.native_block_generation_digest == block.generation_digest
    )
    compact_site_count = len(block.source_site_ids)
    weight_count = int(
        bindings[0].program.binding.sites.weight_coefficients.shape[1]
    )
    track_count = len({binding.track_id for binding in bindings})
    maximum_row_word_count = max(row.word_count for row in rows)

    fixed_preflight = (
        preflight_kinetic_native_equal_rank_sparse_geometry_reduction_memory(
            sampler,
            block_generation_digest=block.generation_digest,
            include_ray_gradients=False,
        )
    )
    trainable_preflight = (
        preflight_kinetic_native_equal_rank_sparse_geometry_reduction_memory(
            sampler,
            block_generation_digest=block.generation_digest,
            include_ray_gradients=True,
        )
    )
    result = _reduce_sparse(
        native_vjp,
        sampler,
        _Fence(),
        include_ray_gradients=False,
    )

    def expected_peak(*, include_ray_gradients: bool) -> tuple[int, int]:
        input_bytes = 4 * block.node_count * block.word_count
        maximum_row_copy_bytes = (
            8 * block.node_count * maximum_row_word_count
        )
        output_bytes = (
            8 * compact_site_count
            + 8 * compact_site_count * (6 + weight_count)
            + (96 * track_count if include_ray_gradients else 0)
        )
        row_source_bytes = (
            8 * maximum_row_word_count
            + 8 * maximum_row_word_count * (6 + weight_count)
            + 96
            + 8 * block.node_count
            + 8 * block.node_count * maximum_row_word_count
        )
        row_parameter_bytes = (
            8 * maximum_row_word_count * (6 + weight_count)
            + (96 if include_ray_gradients else 0)
        )
        node_scratch_bytes = (24 * maximum_row_word_count + 64) * 8
        validation_scratch_bytes = 1 + max(
            block.node_count * maximum_row_word_count,
            3 * compact_site_count,
            weight_count * compact_site_count,
            12 * track_count if include_ray_gradients else 0,
        )
        return (
            input_bytes
            + maximum_row_copy_bytes
            + output_bytes
            + row_source_bytes
            + row_parameter_bytes
            + node_scratch_bytes
            + validation_scratch_bytes,
            validation_scratch_bytes,
        )

    fixed_peak, fixed_validation = expected_peak(include_ray_gradients=False)
    trainable_peak, trainable_validation = expected_peak(
        include_ray_gradients=True
    )
    assert fixed_preflight == result.memory
    assert fixed_preflight.bridge_visible_peak_logical_tensor_bytes == fixed_peak
    assert (
        fixed_preflight.maximum_validation_scratch_tensor_bytes
        == fixed_validation
    )
    assert (
        trainable_preflight.bridge_visible_peak_logical_tensor_bytes
        == trainable_peak
    )
    assert (
        trainable_preflight.maximum_validation_scratch_tensor_bytes
        == trainable_validation
    )
    assert fixed_preflight.bridge_visible_peak_logical_tensor_bytes > (
        fixed_preflight.native_full_length_bar_tensor_bytes
    )


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        ({"derivative_scope": "changed"}, "derivative/omission contract"),
        ({"geometry_vjp_implemented": False}, "derivative/omission contract"),
        ({"material_gradients_included": True}, "derivative/omission contract"),
        ({"event_time_derivatives_included": True}, "derivative/omission contract"),
        ({"chart_endpoint_derivatives_included": True}, "derivative/omission contract"),
        ({"node_time_or_rank_derivatives_included": True}, "derivative/omission contract"),
        ({"compiler_choice_derivatives_included": True}, "derivative/omission contract"),
        ({"continuous_topology_certificate_id": "foreign"}, "topology certificate"),
        (
            {"node_physical_length_cotangent_provenance_id": "foreign"},
            "cotangent provenance",
        ),
    ),
)
def test_analytic_row_result_contract_is_checked_before_accumulation(
    monkeypatch: pytest.MonkeyPatch,
    updates: dict[str, object],
    message: str,
) -> None:
    native_vjp, sampler = _full_vjp_case()
    original = geometry_reduction.kinetic_p0_node_physical_length_geometry_vjp

    def altered_result(*args, **kwargs):
        return replace(original(*args, **kwargs), **updates)

    monkeypatch.setattr(
        geometry_reduction,
        "kinetic_p0_node_physical_length_geometry_vjp",
        altered_result,
    )
    fence = _Fence()
    with pytest.raises(ValueError, match=message):
        _reduce(native_vjp, sampler, fence)
    assert fence.calls == 1


def test_provenance_stale_inputs_and_preflight_fail_before_fence() -> None:
    native_vjp, sampler = _full_vjp_case()
    fence = _Fence()
    with pytest.raises(ValueError, match="provenance changed"):
        reduce_kinetic_native_equal_rank_geometry_vjp(
            native_vjp,
            sampler,
            expected_native_vjp_provenance_id="0" * 64,
            device_completion_fence=fence,
            device_completion_fence_provenance="cpu-fake-fence",
            maximum_bridge_visible_peak_logical_tensor_bytes=10_000_000,
        )
    assert fence.calls == 0

    with pytest.raises(MemoryError, match="preflight byte budget"):
        _reduce(native_vjp, sampler, fence, byte_cap=1)
    assert fence.calls == 0

    with torch.no_grad():
        native_vjp.grad_node_physical_length_f32.add_(1.0)
    with pytest.raises(ValueError, match="identity/layout/version changed"):
        _reduce(native_vjp, sampler, fence)
    assert fence.calls == 0

    native_vjp, sampler = _full_vjp_case()
    fence = _Fence()
    with torch.no_grad():
        sampler.rows[0].program.binding.ray_coefficients.add_(0.01)
    with pytest.raises(
        ValueError,
        match="source (content digest mismatch|tensor identity/layout/version changed)",
    ):
        _reduce(native_vjp, sampler, fence)
    assert fence.calls == 0


def test_fence_contract_and_memory_are_independent_of_requested_frame_count() -> None:
    native_vjp, sampler = _full_vjp_case()
    provenance = kinetic_native_equal_rank_vjp_provenance_id(native_vjp)
    with pytest.raises(TypeError, match="must be callable"):
        reduce_kinetic_native_equal_rank_geometry_vjp(
            native_vjp,
            sampler,
            expected_native_vjp_provenance_id=provenance,
            device_completion_fence=None,
            device_completion_fence_provenance="cpu-fake-fence",
            maximum_bridge_visible_peak_logical_tensor_bytes=10_000_000,
        )
    with pytest.raises(ValueError, match="must be nonempty"):
        reduce_kinetic_native_equal_rank_geometry_vjp(
            native_vjp,
            sampler,
            expected_native_vjp_provenance_id=provenance,
            device_completion_fence=_Fence(),
            device_completion_fence_provenance="",
            maximum_bridge_visible_peak_logical_tensor_bytes=10_000_000,
        )

    wrong_return_calls = 0

    def wrong_return_fence():
        nonlocal wrong_return_calls
        wrong_return_calls += 1
        return "not-a-completion-fence-result"

    with pytest.raises(TypeError, match="must return None"):
        reduce_kinetic_native_equal_rank_geometry_vjp(
            native_vjp,
            sampler,
            expected_native_vjp_provenance_id=provenance,
            device_completion_fence=wrong_return_fence,
            device_completion_fence_provenance="cpu-fake-fence",
            maximum_bridge_visible_peak_logical_tensor_bytes=10_000_000,
        )
    assert wrong_return_calls == 1

    stale_fence_calls = 0

    def stale_during_fence() -> None:
        nonlocal stale_fence_calls
        stale_fence_calls += 1
        native_vjp.grad_node_physical_length_f32.add_(1.0)

    with pytest.raises(
        ValueError,
        match="(changed across the completion fence|identity/layout/version changed)",
    ):
        reduce_kinetic_native_equal_rank_geometry_vjp(
            native_vjp,
            sampler,
            expected_native_vjp_provenance_id=provenance,
            device_completion_fence=stale_during_fence,
            device_completion_fence_provenance="cpu-fake-fence",
            maximum_bridge_visible_peak_logical_tensor_bytes=10_000_000,
        )
    assert stale_fence_calls == 1

    native_vjp, sampler = _full_vjp_case()
    fence = _Fence()
    result = _reduce(native_vjp, sampler, fence)
    small = result.memory_report(2)
    large = result.memory_report(1_000_000)
    assert fence.calls == 1
    assert {key: value for key, value in small.__dict__.items() if key != "requested_frame_count"} == {
        key: value for key, value in large.__dict__.items() if key != "requested_frame_count"
    }
    assert small.requested_frame_count == 2
    assert large.requested_frame_count == 1_000_000
    assert small.bridge_visible_peak_logical_tensor_bytes == 704
    assert small.persistent_frame_tensor_bytes == 0
    assert small.persistent_sample_tensor_bytes == 0
    assert small.persistent_target_tensor_bytes == 0
    assert small.persistent_prediction_tensor_bytes == 0
    assert small.persistent_material_tensor_bytes == 0
    assert small.frame_by_word_reverse_state_tensor_bytes == 0
    assert result.accounting["frame_scaling"] == ("independent of requested F after node reduction")

    parameters = inspect.signature(reduce_kinetic_native_equal_rank_geometry_vjp).parameters
    assert (
        not {
            "requested_frame_count",
            "sample_count",
            "targets",
            "predictions",
            "material",
        }
        & parameters.keys()
    )


def test_certified_sparse_reducer_matches_dense_oracle_with_compact_bars() -> None:
    native_vjp, sampler = _full_vjp_case(view_index=3)
    dense = _reduce(native_vjp, sampler, _Fence())
    fence = _Fence()
    sparse = _reduce_sparse(native_vjp, sampler, fence)

    global_positions = torch.zeros_like(dense.grad_positions0_f64)
    global_velocities = torch.zeros_like(dense.grad_velocities_f64)
    global_weights = torch.zeros_like(dense.grad_weight_coefficients_f64)
    global_positions.index_add_(
        0,
        sparse.source_site_ids_i64,
        sparse.grad_compact_positions0_f64,
    )
    global_velocities.index_add_(
        0,
        sparse.source_site_ids_i64,
        sparse.grad_compact_velocities_f64,
    )
    global_weights.index_add_(
        0,
        sparse.source_site_ids_i64,
        sparse.grad_compact_weight_coefficients_f64,
    )

    assert fence.calls == sparse.device_completion_fence_call_count == 1
    torch.testing.assert_close(global_positions, dense.grad_positions0_f64)
    torch.testing.assert_close(global_velocities, dense.grad_velocities_f64)
    torch.testing.assert_close(global_weights, dense.grad_weight_coefficients_f64)
    torch.testing.assert_close(
        sparse.grad_track_ray_coefficients_f64,
        dense.grad_track_ray_coefficients_f64,
    )
    assert sparse.geometry_reduction_mode == "certified_sparse_compact"
    assert sparse.dense_global_site_accumulation_elements == 0
    assert sparse.all_site_owner_validation_evaluations == 0
    assert sparse.maximum_simultaneous_jw_length_bar_tensors == 1
    assert sparse.memory.cpu_full_length_bar_copy_tensor_bytes == 0
    assert sparse.native_length_bar_row_copy_count == sparse.row_count
    assert sparse.compact_owner_scatter_rows == sparse.word_count
    assert sparse.accounting["equal_rank_implies_equal_node_times"] is False
    assert sparse.accounting["future_native_row_node_time_shape"] == "[Q_block,J]"
    assert sparse.accounting["future_native_owner_index_space"] == (
        "compact word_owner_i32"
    )
    assert sparse.accounting["preferred_native_full_jw_length_bar_allocated"] is False
    sparse.assert_current()


def test_certified_sparse_reducer_fails_before_fence_on_certificate_tamper() -> None:
    native_vjp, sampler = _full_vjp_case()
    topology = sampler.rows[0].source.lowering.charts[sampler.rows[0].chart_index]
    object.__setattr__(topology, "owner_topology_certificate_digest", "0" * 64)
    fence = _Fence()

    with pytest.raises(ValueError, match="certificate digest mismatch"):
        _reduce_sparse(native_vjp, sampler, fence)
    assert fence.calls == 0
