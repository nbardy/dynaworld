from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest
import torch
from kinetic_multichart_transfer_program import compile_kinetic_multichart_p0_program
from kinetic_native_equal_rank_lowering import (
    KineticNativeEqualRankChartSource,
    iter_materialize_kinetic_native_equal_rank_blocks,
    kinetic_native_equal_rank_chart_sources_for_track,
    lower_kinetic_native_equal_rank_buckets,
    pack_equal_rank_compact_site_rgba,
)
from kinetic_native_topology_lowering import lower_kinetic_multichart_to_native_topology
from kinetic_owner_chart_compiler import compile_exact_kinetic_owner_charts
from kinetic_power_word_compiler import AffineKineticPowerSites

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


def _heterogeneous_sources():
    sites = _shared_sites()
    programs = {
        # Same world sites, but different rays, chart counts, and actual ranks.
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
    return programs, tuple(reversed(sources))


def test_equal_rank_descriptor_preserves_heterogeneous_rows_without_common_refinement() -> None:
    programs, sources = _heterogeneous_sources()
    lowering = lower_kinetic_native_equal_rank_buckets(
        sources,
        maximum_rows_per_block=2,
    )

    assert tuple(row.row_identity for row in lowering.rows) == (
        (10, 0),
        (20, 0),
        (20, 1),
        (30, 0),
        (30, 1),
        (30, 2),
    )
    assert tuple(bucket.node_count for bucket in lowering.buckets) == (3, 4, 5)
    assert tuple(bucket.row_count for bucket in lowering.buckets) == (1, 2, 3)
    assert tuple(len(bucket.blocks) for bucket in lowering.buckets) == (1, 1, 2)
    assert lowering.row_count == sum(program.chart_count for program in programs.values())
    assert tuple(row.right_closed for row in lowering.rows) == (
        True,
        False,
        True,
        False,
        False,
        True,
    )
    assert {row.word_count for row in lowering.rows} == {1, 2}
    assert not lowering.global_common_temporal_refinement_used
    assert not lowering.jmax_padding_used
    assert not lowering.requested_frame_sampling_used
    assert lowering.descriptor_tensor_bytes == 0
    assert lowering.descriptor_canonical_metadata_bytes > 0
    assert not lowering.descriptor_python_allocator_bytes_measured

    # A global common refinement would take every track across every distinct
    # union interval.  The descriptor instead retains exactly the six source
    # charts and their own endpoints.
    union_endpoints = sorted({value for row in lowering.rows for value in (row.t_min, row.t_max)})
    common_refinement_rows = len(programs) * (len(union_endpoints) - 1)
    assert common_refinement_rows > lowering.row_count
    lowering.assert_current(tuple(reversed(sources)))


def test_bounded_materialization_compacts_global_sites_and_packs_native_lengths() -> None:
    programs, sources = _heterogeneous_sources()
    lowering = lower_kinetic_native_equal_rank_buckets(
        sources,
        maximum_rows_per_block=2,
    )
    blocks = tuple(iter_materialize_kinetic_native_equal_rank_blocks(lowering, sources))
    source_by_identity = {source.row_identity: source for source in sources}
    row_by_index = {row.global_row_index: row for row in lowering.rows}

    assert len(blocks) == 4
    assert max(block.row_count for block in blocks) == 2
    for payload in blocks:
        payload.assert_warm_layout()
        payload.assert_cold_current(lowering, sources)
        assert payload.row_count <= lowering.maximum_rows_per_block
        assert payload.node_physical_length_f32.shape == (
            payload.node_count,
            payload.word_count,
        )
        assert tuple(payload.config_i32.tolist()) == (
            payload.row_count,
            payload.node_count,
            payload.compact_site_count,
            payload.word_count,
        )
        assert payload.retained_tensor_bytes == payload.block.expected_retained_tensor_bytes
        accounting = payload.byte_accounting
        assert accounting["retained_tensor_bytes"] == sum(
            int(accounting[key])
            for key in (
                "identity_domain_tensor_bytes",
                "topology_tensor_bytes",
                "node_physical_length_tensor_bytes",
                "duplicated_schedule_tensor_bytes",
            )
        )
        assert accounting["persistent_frame_tensor_bytes"] == 0
        assert accounting["persistent_sample_tensor_bytes"] == 0
        assert accounting["persistent_target_tensor_bytes"] == 0
        assert accounting["persistent_prediction_tensor_bytes"] == 0
        assert accounting["dense_row_by_global_time_tensor_bytes"] == 0
        assert accounting["allocator_peak_measured"] is False

        length_start = 0
        for local_row, global_row in enumerate(payload.block.global_row_indices):
            row = row_by_index[global_row]
            word_start = int(payload.word_offsets_i32[local_row].item())
            word_end = int(payload.word_offsets_i32[local_row + 1].item())
            compact_ids = payload.word_owner_i32[word_start:word_end].to(torch.int64)
            source_ids = payload.source_site_ids_i64.index_select(0, compact_ids)
            assert tuple(source_ids.tolist()) == row.owner_word
            source = source_by_identity[row.row_identity]
            expected_length = source.program.charts[source.chart_index].node_physical_lengths.to(
                torch.float32
            )
            torch.testing.assert_close(
                payload.node_physical_length_f32[
                    :,
                    length_start : length_start + row.word_count,
                ],
                expected_length,
                rtol=0,
                atol=0,
            )
            assert bool(payload.row_right_closed_bool[local_row].item()) == row.right_closed
            length_start += row.word_count

        # Sample interpolation schedules stay in the provenance-bound source
        # program and are not duplicated in the native world-node block.
        assert accounting["duplicated_schedule_tensor_bytes"] == 0
        assert not hasattr(payload, "fit_matrix_f64")

        density = torch.tensor([0.25, 0.75], dtype=torch.float32)
        color = torch.tensor([[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]], dtype=torch.float32)
        compact_rgba = pack_equal_rank_compact_site_rgba(payload, density, color)
        expected_ids = payload.source_site_ids_i64
        torch.testing.assert_close(compact_rgba[:, :3], color.index_select(0, expected_ids))
        torch.testing.assert_close(compact_rgba[:, 3], density.index_select(0, expected_ids))

    # The rank-4 block combines [0] and [0,1] through one global compact table.
    rank4 = next(payload for payload in blocks if payload.node_count == 4)
    assert tuple(rank4.source_site_ids_i64.tolist()) == (0, 1)
    assert tuple(rank4.word_owner_i32.tolist()) == (0, 0, 1)
    assert tuple(rank4.word_offsets_i32.tolist()) == (0, 1, 3)


def test_frame_density_does_not_change_descriptor_or_streamed_block_bytes() -> None:
    _, sources = _heterogeneous_sources()
    lowering = lower_kinetic_native_equal_rank_buckets(
        sources,
        maximum_rows_per_block=2,
    )
    small = lowering.memory_report(3)
    huge = lowering.memory_report(10_000_000)

    differing = {"requested_frame_count"}
    for name in small.__dataclass_fields__:
        if name not in differing:
            assert getattr(small, name) == getattr(huge, name)
    assert small.descriptor_tensor_bytes == 0
    assert small.descriptor_canonical_metadata_bytes > 0
    assert small.descriptor_row_count == 6
    assert small.descriptor_rank_bucket_count == 3
    assert small.descriptor_block_count == 4
    assert small.total_materialized_block_tensor_bytes >= small.maximum_materialized_block_tensor_bytes
    assert small.source_plus_maximum_block_tensor_bytes == (
        small.source_kinetic_program_tensor_bytes + small.maximum_materialized_block_tensor_bytes
    )
    assert not small.allocator_peak_measured
    assert not small.descriptor_python_allocator_bytes_measured
    assert small.persistent_frame_tensor_bytes == 0
    assert small.persistent_sample_tensor_bytes == 0
    assert small.persistent_target_tensor_bytes == 0
    assert small.persistent_prediction_tensor_bytes == 0
    assert small.dense_row_by_global_time_tensor_bytes == 0

    one_row = lower_kinetic_native_equal_rank_buckets(
        sources,
        maximum_rows_per_block=1,
    )
    all_rows_of_rank = lower_kinetic_native_equal_rank_buckets(
        sources,
        maximum_rows_per_block=100,
    )
    assert one_row.maximum_materialized_block_tensor_bytes < (
        all_rows_of_rank.maximum_materialized_block_tensor_bytes
    )
    assert one_row.row_count == all_rows_of_rank.row_count


def test_duplicate_identity_and_mixed_track_provenance_fail_closed() -> None:
    _, sources = _heterogeneous_sources()
    with pytest.raises(ValueError, match=r"duplicate \(track_id, chart_index\)"):
        lower_kinetic_native_equal_rank_buckets(
            (*sources, sources[0]),
            maximum_rows_per_block=2,
        )

    source_10 = next(source for source in sources if source.track_id == 10)
    source_20_chart1 = next(
        source for source in sources if source.track_id == 20 and source.chart_index == 1
    )
    mixed = replace(source_20_chart1, track_id=10)
    with pytest.raises(ValueError, match="cannot mix different kinetic programs"):
        lower_kinetic_native_equal_rank_buckets(
            (source_10, mixed),
            maximum_rows_per_block=2,
        )


def test_descriptor_and_payload_tampering_fail_at_the_correct_cold_or_warm_gate() -> None:
    _, sources = _heterogeneous_sources()
    lowering = lower_kinetic_native_equal_rank_buckets(
        sources,
        maximum_rows_per_block=2,
    )
    payload = next(iter_materialize_kinetic_native_equal_rank_blocks(lowering, sources))

    with pytest.raises(ValueError, match="generation digest mismatch"):
        replace(lowering, generation_digest="0" * 64).assert_current(sources)
    with pytest.raises(ValueError, match="canonical metadata accounting"):
        replace(
            lowering,
            descriptor_canonical_metadata_bytes=lowering.descriptor_canonical_metadata_bytes + 1,
        ).assert_current(sources)
    with pytest.raises(ValueError, match="generation digest mismatch"):
        replace(payload, generation_digest="f" * 64).assert_cold_current(lowering, sources)
    with pytest.raises(ValueError, match="not sealed"):
        replace(payload, _seal=None).assert_warm_layout()

    # Warm validation is intentionally provenance-free and allocation-free: it
    # needs no source program and rejects an ordinary in-place write by version.
    assert payload.warm_validation_device_to_host_syncs == 0
    assert payload.warm_validation_tensor_allocations == 0
    payload.node_physical_length_f32[0, 0].add_(0.125)
    with pytest.raises(ValueError, match="identity/layout/version changed"):
        payload.assert_warm_layout()

    source = sources[0]
    source.program.binding.ray_coefficients[0].add_(0.25)
    with pytest.raises(ValueError, match="source content digest mismatch"):
        lowering.assert_current(sources)


def test_global_site_namespace_mismatch_fails_closed() -> None:
    sites = _shared_sites()
    first = _program(sites, ray_origin_x=-2, node_count=3)
    changed_sites = AffineKineticPowerSites(
        positions0=sites.positions0 + torch.tensor([[0.0, 0.0, 0.0], [0.125, 0.0, 0.0]]),
        velocities=sites.velocities,
        weight_coefficients=sites.weight_coefficients,
    )
    second = _program(changed_sites, ray_origin_x=2, node_count=3)
    first_source = kinetic_native_equal_rank_chart_sources_for_track(0, first)[0]
    second_source = kinetic_native_equal_rank_chart_sources_for_track(1, second)[0]
    assert isinstance(first_source, KineticNativeEqualRankChartSource)
    with pytest.raises(ValueError, match="share one global kinetic site namespace"):
        lower_kinetic_native_equal_rank_buckets(
            (first_source, second_source),
            maximum_rows_per_block=2,
        )
