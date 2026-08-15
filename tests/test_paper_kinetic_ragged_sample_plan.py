from __future__ import annotations

from fractions import Fraction

import pytest
import torch
from kinetic_multichart_transfer_program import (
    KineticMultiChartP0Program,
    compile_kinetic_multichart_p0_program,
    dispatch_kinetic_chart_index,
    dispatch_prevalidated_kinetic_chart_index,
)
from kinetic_native_equal_rank_lowering import (
    kinetic_native_equal_rank_chart_sources_for_track,
    lower_kinetic_native_equal_rank_buckets,
)
from kinetic_native_topology_lowering import lower_kinetic_multichart_to_native_topology
from kinetic_owner_chart_compiler import compile_exact_kinetic_owner_charts
from kinetic_power_word_compiler import AffineKineticPowerSites
from paper_kinetic_ragged_sample_plan import (
    diagnose_paper_kinetic_coordinator_bar_assembly,
    iter_paper_kinetic_row_ragged_sample_blocks,
    prepare_paper_kinetic_row_ragged_sampler,
)
from paper_ragged_track_staging import PaperRaggedTrackTargetStageBlock
from powerfoam_track_staging import (
    PowerFoamTrackLossNormalization,
    PowerFoamTrackTargetStageBlock,
)

DTYPE = torch.float64


def _shared_sites() -> AffineKineticPowerSites:
    slopes = [(0, 0), (-2, 0)]
    intercepts = [(0, 0, 0), (1, -1, 0)]
    positions = []
    velocities = []
    weights = []
    for (slope0, slope1), (bias0, bias1, bias2) in zip(
        slopes,
        intercepts,
        strict=True,
    ):
        position = -Fraction(slope0) / 2
        velocity = -Fraction(slope1) / 2
        positions.append((position, Fraction(0), Fraction(0)))
        velocities.append((velocity, Fraction(0), Fraction(0)))
        weights.append(
            (
                position * position - Fraction(bias0),
                2 * position * velocity - Fraction(bias1),
                velocity * velocity - Fraction(bias2),
            )
        )
    return AffineKineticPowerSites(
        positions0=torch.tensor(
            [[float(value) for value in row] for row in positions],
            dtype=DTYPE,
        ),
        velocities=torch.tensor(
            [[float(value) for value in row] for row in velocities],
            dtype=DTYPE,
        ),
        weight_coefficients=torch.tensor(
            [[float(value) for value in row] for row in weights],
            dtype=DTYPE,
        ),
    )


def _program(
    sites: AffineKineticPowerSites,
    *,
    ray_origin_x: int,
    node_count: int,
):
    ray = torch.tensor(
        [ray_origin_x, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        dtype=DTYPE,
    )
    owner_program = compile_exact_kinetic_owner_charts(
        sites,
        ray,
        t_min=-2,
        t_max=2,
        near=0,
        far=1,
    )
    return compile_kinetic_multichart_p0_program(
        owner_program,
        sites,
        ray,
        node_count=node_count,
    )


def _sampler_fixture():
    sites = _shared_sites()
    programs = {
        10: _program(sites, ray_origin_x=-2, node_count=3),
        20: _program(sites, ray_origin_x=-1, node_count=4),
        30: _program(sites, ray_origin_x=0, node_count=5),
    }
    assert [programs[track_id].chart_count for track_id in (10, 20, 30)] == [1, 2, 3]
    sources = []
    for track_id in (30, 10, 20):
        program = programs[track_id]
        sources.extend(
            kinetic_native_equal_rank_chart_sources_for_track(
                track_id,
                program,
                lowering=lower_kinetic_multichart_to_native_topology(program),
            )
        )
    sources = tuple(reversed(sources))
    lowering = lower_kinetic_native_equal_rank_buckets(
        sources,
        maximum_rows_per_block=2,
    )
    sampler = prepare_paper_kinetic_row_ragged_sampler(
        view_index=7,
        lowering=lowering,
        sources=sources,
    )
    return programs, sources, lowering, sampler


def _staged_block(
    times: list[float],
    *,
    track_ids: tuple[int, ...] = (10, 20, 30),
    view_index: int = 7,
) -> PaperRaggedTrackTargetStageBlock:
    sample_count = len(times)
    track_count = len(track_ids)
    pixels = torch.tensor(track_ids, dtype=torch.long)
    frame_indices = torch.tensor(
        [(3 * index + 1) % max(1, sample_count) for index in range(sample_count)],
        dtype=torch.long,
    )
    sample_indices = frame_indices + view_index * max(1, sample_count)
    targets = torch.empty((track_count, sample_count, 3), dtype=torch.float32)
    for track_position in range(track_count):
        for observation_position in range(sample_count):
            value = float(100 * track_position + 10 * observation_position)
            targets[track_position, observation_position] = torch.tensor(
                [value + 1.0, value + 2.0, value + 3.0],
                dtype=torch.float32,
            )
    normalization = PowerFoamTrackLossNormalization(
        global_track_count=track_count,
        global_sample_count=sample_count,
        block_track_count=track_count,
        block_sample_count=sample_count,
    )
    staged = PowerFoamTrackTargetStageBlock(
        pixel_indices=pixels,
        sample_indices=sample_indices,
        view_indices=torch.full((sample_count,), view_index, dtype=torch.long),
        frame_indices=frame_indices,
        sample_times=torch.tensor(times, dtype=torch.float32),
        targets=targets,
        normalization=normalization,
        accounting={"ray_bytes": 0, "explicit_rays_staged": False},
    )
    batch_positions = torch.tensor(
        list(reversed(range(sample_count))),
        dtype=torch.long,
    )
    return PaperRaggedTrackTargetStageBlock(
        view_index=view_index,
        batch_positions=batch_positions,
        logical_sample_start=0,
        logical_sample_end=sample_count,
        staged=staged,
        normalization=normalization,
    )


def _forbid(*_args, **_kwargs):
    raise AssertionError("warm validation performed a forbidden content/allocation operation")


def test_sampler_is_frame_free_and_warm_validation_is_metadata_only(monkeypatch) -> None:
    _programs, _sources, lowering, sampler = _sampler_fixture()

    assert tuple(row.row_identity for row in sampler.rows) == (
        (10, 0),
        (20, 0),
        (20, 1),
        (30, 0),
        (30, 1),
        (30, 2),
    )
    assert tuple(row.node_count for row in sampler.rows) == (3, 4, 4, 5, 5, 5)
    assert tuple(row.native_local_row_index for row in sampler.rows) == (0, 0, 1, 0, 1, 0)
    small = sampler.memory_report(4)
    huge = sampler.memory_report(10_000_000)
    for name in small.__dataclass_fields__:
        if name != "requested_frame_count":
            assert getattr(small, name) == getattr(huge, name)
    assert small.descriptor_tensor_bytes == 0
    assert small.descriptor_canonical_metadata_bytes > 0
    assert small.source_kinetic_program_tensor_bytes == lowering.source_kinetic_program_tensor_bytes
    assert small.persistent_sample_tensor_bytes == 0
    assert small.persistent_target_tensor_bytes == 0
    assert small.persistent_interpolation_weight_tensor_bytes == 0
    assert small.dense_row_by_global_time_tensor_bytes == 0
    assert not small.global_common_temporal_refinement_used
    assert not small.requested_frame_sampling_used_for_compile

    import paper_kinetic_ragged_sample_plan as sample_plan

    monkeypatch.setattr(sample_plan, "_tensor_digest", _forbid)
    monkeypatch.setattr(torch.Tensor, "cpu", _forbid)
    monkeypatch.setattr(torch.Tensor, "item", _forbid)
    monkeypatch.setattr(torch.Tensor, "tolist", _forbid)
    monkeypatch.setattr(torch, "empty", _forbid)
    monkeypatch.setattr(torch, "zeros", _forbid)
    monkeypatch.setattr(torch, "tensor", _forbid)
    monkeypatch.setattr(torch, "as_tensor", _forbid)
    sampler.assert_warm_layout()


def test_arbitrary_paper_order_dispatches_seams_and_terminal_closure_without_padding() -> None:
    programs, _sources, lowering, sampler = _sampler_fixture()
    staged = _staged_block([1.0, -2.0, -1.0, 2.0])
    blocks = tuple(
        iter_paper_kinetic_row_ragged_sample_blocks(
            sampler,
            staged,
            loss_normalization_id="paper-step-seams",
            maximum_samples_per_launch=2,
        )
    )

    assert blocks
    assert {block.node_count for block in blocks} == {3, 4, 5}
    assert all(block.sample_count <= 2 for block in blocks)
    assert all(block.loss_scale == 1.0 / staged.normalization.global_rgb_element_count for block in blocks)
    assert all(block.loss_normalization_id == "paper-step-seams" for block in blocks)
    assert all(not block.global_common_temporal_refinement_used for block in blocks)
    assert all(block.dense_row_by_global_time_tensor_bytes == 0 for block in blocks)
    assert all(block.persistent_after_launch_tensor_bytes == 0 for block in blocks)

    native_blocks = {block.generation_digest: block for bucket in lowering.buckets for block in bucket.blocks}
    row_by_index = {row.global_row_index: row for row in sampler.rows}
    flat_to_identity = {}
    covered = []
    target_flat = staged.targets.reshape(-1, 3)
    for block in blocks:
        block.assert_warm_layout()
        block.assert_cold_current(sampler)
        assert block.sample_row_i32.device.type == "cpu"
        assert block.sample_row_i32.dtype == torch.int32
        assert block.sample_to_node_f32.dtype == torch.float32
        assert block.target_rgb_f32.dtype == torch.float32
        assert block.sample_to_node_f32.shape == (block.sample_count, block.node_count)
        assert block.target_rgb_f32.shape == (block.sample_count, 3)
        native = native_blocks[block.native_block_generation_digest]
        for sample_index, flat_index in enumerate(block.flat_sample_index_i64.tolist()):
            covered.append(flat_index)
            track_position, observation_position = divmod(flat_index, staged.sample_times.numel())
            track_id = int(staged.pixel_indices[track_position])
            time = float(staged.sample_times[observation_position])
            chart_index = dispatch_kinetic_chart_index(programs[track_id], time)
            local_row = int(block.sample_row_i32[sample_index])
            global_row = native.global_row_indices[local_row]
            row = row_by_index[global_row]
            assert row.row_identity == (track_id, chart_index)
            flat_to_identity[flat_index] = row.row_identity
            torch.testing.assert_close(block.target_rgb_f32[sample_index], target_flat[flat_index])
            expected_weight = (
                programs[track_id]
                .charts[chart_index]
                .schedule.sample_to_node_weights(staged.sample_times[observation_position : observation_position + 1])
                .weights.to(torch.float32)[0]
            )
            torch.testing.assert_close(
                block.sample_to_node_f32[sample_index],
                expected_weight,
                rtol=2.0e-6,
                atol=2.0e-6,
            )

    assert sorted(covered) == list(range(staged.targets.shape[0] * staged.targets.shape[1]))
    assert len(covered) == len(set(covered))
    sample_count = int(staged.sample_times.numel())
    assert flat_to_identity[1 * sample_count + 0] == (20, 1)  # t=1 is right chart.
    assert flat_to_identity[2 * sample_count + 2] == (30, 1)  # t=-1 is right chart.
    assert flat_to_identity[2 * sample_count + 0] == (30, 2)  # t=1 is right chart.
    assert flat_to_identity[2 * sample_count + 3] == (30, 2)  # terminal t=2 is closed.
    assert (
        dispatch_prevalidated_kinetic_chart_index(
            programs[30],
            1.0,
            expected_generation_digest=programs[30].generation_digest,
        )
        == 2
    )
    with pytest.raises(ValueError, match="generation is stale"):
        dispatch_prevalidated_kinetic_chart_index(
            programs[30],
            1.0,
            expected_generation_digest="stale-generation",
        )


def test_launch_block_warm_seal_uses_no_content_or_tensor_allocation(monkeypatch) -> None:
    _programs, _sources, _lowering, sampler = _sampler_fixture()
    block = next(
        iter_paper_kinetic_row_ragged_sample_blocks(
            sampler,
            _staged_block([-2.0, -1.0, 1.0, 2.0]),
            loss_normalization_id="paper-step-warm",
            maximum_samples_per_launch=3,
        )
    )

    import paper_kinetic_ragged_sample_plan as sample_plan

    monkeypatch.setattr(sample_plan, "_tensor_digest", _forbid)
    monkeypatch.setattr(torch.Tensor, "cpu", _forbid)
    monkeypatch.setattr(torch.Tensor, "item", _forbid)
    monkeypatch.setattr(torch.Tensor, "tolist", _forbid)
    monkeypatch.setattr(torch, "empty", _forbid)
    monkeypatch.setattr(torch, "zeros", _forbid)
    monkeypatch.setattr(torch, "tensor", _forbid)
    monkeypatch.setattr(torch, "as_tensor", _forbid)
    block.assert_warm_layout()


def test_streamed_request_does_not_recertify_world_or_inspect_target_content(monkeypatch) -> None:
    _programs, _sources, _lowering, sampler = _sampler_fixture()
    staged = _staged_block([-2.0, -1.0, 1.0, 2.0])
    target_pointer = staged.targets.untyped_storage().data_ptr()
    original_isfinite = torch.isfinite

    def guarded_isfinite(tensor, *args, **kwargs):
        if tensor.untyped_storage().data_ptr() == target_pointer:
            raise AssertionError("streamed request inspected accelerator/provider target content")
        return original_isfinite(tensor, *args, **kwargs)

    import paper_kinetic_ragged_sample_plan as sample_plan

    monkeypatch.setattr(sample_plan, "_tensor_digest", _forbid)
    monkeypatch.setattr(KineticMultiChartP0Program, "assert_current", _forbid)
    monkeypatch.setattr(torch, "isfinite", guarded_isfinite)
    blocks = tuple(
        iter_paper_kinetic_row_ragged_sample_blocks(
            sampler,
            staged,
            loss_normalization_id="no-world-recertification",
            maximum_samples_per_launch=2,
        )
    )
    assert sum(block.sample_count for block in blocks) == staged.targets.shape[0] * staged.targets.shape[1]
    for block in blocks:
        block.assert_warm_layout()
        block.assert_cold_current(sampler)


def test_fixed_launch_bound_keeps_peak_sample_state_invariant_as_time_density_grows() -> None:
    _programs, _sources, _lowering, sampler = _sampler_fixture()
    sparse_times = [-2.0, -1.0, 1.0, 2.0] * 5
    dense_times = [-2.0, -1.0, 1.0, 2.0] * 50

    def materialize(times: list[float]):
        return tuple(
            iter_paper_kinetic_row_ragged_sample_blocks(
                sampler,
                _staged_block(times),
                loss_normalization_id=f"density-{len(times)}",
                maximum_samples_per_launch=4,
            )
        )

    sparse = materialize(sparse_times)
    dense = materialize(dense_times)
    assert sum(block.sample_count for block in sparse) == 3 * len(sparse_times)
    assert sum(block.sample_count for block in dense) == 3 * len(dense_times)
    assert max(block.sample_count for block in sparse) == 4
    assert max(block.sample_count for block in dense) == 4
    assert max(block.retained_tensor_bytes for block in sparse) == max(block.retained_tensor_bytes for block in dense)
    assert sum(block.retained_tensor_bytes for block in dense) > sum(block.retained_tensor_bytes for block in sparse)
    assert all(block.accounting["persistent_after_launch_tensor_bytes"] == 0 for block in dense)
    assert all(block.accounting["dense_row_by_global_time_tensor_bytes"] == 0 for block in dense)
    assert all(block.accounting["global_denominator_preserved"] for block in dense)


def test_multi_rank_request_reports_bounded_union_local_bar_merge_ready() -> None:
    _programs, _sources, _lowering, sampler = _sampler_fixture()
    blocks = tuple(
        iter_paper_kinetic_row_ragged_sample_blocks(
            sampler,
            _staged_block([-2.0, -1.0, 1.0, 2.0]),
            loss_normalization_id="bar-assembly-boundary",
            maximum_samples_per_launch=2,
        )
    )

    diagnostic = diagnose_paper_kinetic_coordinator_bar_assembly(sampler, blocks)
    assert diagnostic.native_block_count > 1
    assert diagnostic.cross_native_block_merge_required
    assert diagnostic.union_source_site_count <= diagnostic.summed_native_compact_site_count
    assert diagnostic.global_site_count == sampler.lowering.global_site_count
    assert diagnostic.bounded_union_local_mapping_implemented
    assert diagnostic.exactly_one_coordinator_compact_bar_proven
    assert not diagnostic.per_request_global_site_bar_allocated
    assert all(block.accounting["coordinator_compact_bar_assembly_implemented"] for block in blocks)
    assert all(not block.accounting["per_request_global_site_bar_allocated"] for block in blocks)
    assert "native VJP numerical correctness" in diagnostic.proof_boundary
    diagnostic.require_ready()


def test_dispatch_and_cold_provenance_fail_closed_on_invalid_or_silent_mutation() -> None:
    _programs, _sources, _lowering, sampler = _sampler_fixture()
    with pytest.raises(ValueError, match="program domain"):
        tuple(
            iter_paper_kinetic_row_ragged_sample_blocks(
                sampler,
                _staged_block([3.0]),
                loss_normalization_id="outside-domain",
                maximum_samples_per_launch=1,
            )
        )
    with pytest.raises(ValueError, match="no compiled track"):
        tuple(
            iter_paper_kinetic_row_ragged_sample_blocks(
                sampler,
                _staged_block([0.0], track_ids=(10, 99)),
                loss_normalization_id="missing-track",
                maximum_samples_per_launch=1,
            )
        )

    block = next(
        iter_paper_kinetic_row_ragged_sample_blocks(
            sampler,
            _staged_block([-2.0, -1.0, 1.0, 2.0]),
            loss_normalization_id="silent-content-drift",
            maximum_samples_per_launch=4,
        )
    )
    block.sample_to_node_f32.numpy()[0, 0] += 0.125
    block.assert_warm_layout()  # Raw storage edit does not increment Tensor._version.
    block.assert_cold_current(sampler)  # Production structural validation remains synchronization-free.
    with pytest.raises(ValueError, match="partition of unity"):
        block.diagnose_cpu_content()
