from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest
import torch
from compiled_transfer_adjoint import FAR_CUT_ID, NEAR_CUT_ID
from kinetic_multichart_transfer_program import compile_kinetic_multichart_p0_program
from kinetic_native_topology_lowering import (
    NATIVE_RUNTIME_REMAINING_GATE,
    lower_kinetic_multichart_to_native_topology,
    materialize_kinetic_native_topology_chart,
)
from kinetic_owner_chart_compiler import compile_exact_kinetic_owner_charts
from kinetic_power_word_compiler import AffineKineticPowerSites

DTYPE = torch.float64


def _sites_from_ray_lines(
    slopes: list[tuple[int | Fraction, int | Fraction]],
    intercepts: list[tuple[int | Fraction, int | Fraction, int | Fraction]],
) -> AffineKineticPowerSites:
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


def _three_chart_program(node_count: int = 4):
    sites = _sites_from_ray_lines(
        slopes=[(0, 0), (-2, 0)],
        intercepts=[(0, 0, 0), (1, -1, 0)],
    )
    ray = torch.tensor(
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
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


def test_every_kinetic_chart_lowers_exactly_to_native_word_and_schedule_abis() -> None:
    program = _three_chart_program(node_count=5)
    lowering = lower_kinetic_multichart_to_native_topology(program)

    assert lowering.chart_count == program.chart_count == 3
    assert len(lowering.event_guard_digests) == 2
    assert not lowering.native_execution_ready
    assert lowering.native_node_physical_length_source_abi_available
    assert not lowering.native_node_physical_length_runtime_verified
    assert not lowering.static_s5_site_geometry_fabricated
    assert not lowering.requested_frame_sampling_used
    assert lowering.remaining_native_runtime_gate == NATIVE_RUNTIME_REMAINING_GATE
    assert "extension rebuild" in NATIVE_RUNTIME_REMAINING_GATE
    assert "real Metal forward/VJP parity" in NATIVE_RUNTIME_REMAINING_GATE
    assert "trainer/session integration" in NATIVE_RUNTIME_REMAINING_GATE

    for chart_index, chart in enumerate(program.charts):
        payload = materialize_kinetic_native_topology_chart(
            lowering,
            program,
            chart_index,
        )
        topology = payload.topology
        schedule = payload.schedule.charts[0]
        word_start = int(topology.word_offsets_i32[0].item())
        word_end = int(topology.word_offsets_i32[1].item())
        compact_owner_ids = topology.word_owner_i32[word_start:word_end].to(dtype=torch.long)
        source_owners = topology.source_site_ids.index_select(0, compact_owner_ids)
        assert tuple(source_owners.tolist()) == chart.owner_word
        assert tuple(topology.word_left_incidence_i32.tolist()) == (
            NEAR_CUT_ID,
            *range(chart.run_count - 1),
        )
        assert tuple(topology.word_right_incidence_i32.tolist()) == (
            *range(chart.run_count - 1),
            FAR_CUT_ID,
        )
        if chart.run_count > 1:
            source_pairs = topology.source_site_ids[topology.boundary_site_pairs_i32.to(dtype=torch.long)]
            assert tuple(map(tuple, source_pairs.tolist())) == tuple(
                zip(chart.owner_word[:-1], chart.owner_word[1:], strict=True)
            )
        else:
            assert tuple(topology.boundary_site_pairs_i32.shape) == (0, 2)

        torch.testing.assert_close(schedule.node_times, chart.schedule.node_times)
        torch.testing.assert_close(schedule.fit_matrix, chart.schedule.fit_matrix)
        torch.testing.assert_close(
            schedule.barycentric_weights,
            chart.schedule.barycentric_weights,
        )
        torch.testing.assert_close(
            payload.node_physical_lengths,
            chart.node_physical_lengths,
        )
        assert payload.spec.owner_word == chart.owner_word
        assert payload.spec.owner_topology_certificate_digest
        assert payload.retained_tensor_bytes == (
            topology.resident_bytes
            + payload.schedule.resident_bytes
            + chart.node_physical_lengths.numel() * chart.node_physical_lengths.element_size()
        )
        assert payload.persistent_sample_time_tensor_bytes == 0
        assert payload.persistent_frame_or_chart_sample_state_bytes == 0
        assert payload.dense_sample_by_chart_tensor_bytes == 0
        assert not payload.spec.requested_frame_sampling_used
        assert payload.spec.native_node_physical_length_source_abi_available
        assert not payload.spec.native_node_physical_length_runtime_verified
        assert payload.spec.remaining_native_runtime_gate == NATIVE_RUNTIME_REMAINING_GATE


def test_streamed_lowering_retained_bytes_are_invariant_to_requested_frames() -> None:
    program = _three_chart_program(node_count=6)
    lowering = lower_kinetic_multichart_to_native_topology(program)
    small = lowering.memory_report(9)
    large = lowering.memory_report(1_000_000)

    assert small.requested_frame_count == 9
    assert large.requested_frame_count == 1_000_000
    for field in (
        "descriptor_tensor_bytes",
        "source_kinetic_program_tensor_bytes",
        "maximum_resident_chart_payload_bytes",
        "maximum_in_process_retained_tensor_bytes",
        "persistent_sample_time_tensor_bytes",
        "persistent_frame_or_chart_sample_state_bytes",
        "dense_sample_by_chart_tensor_bytes",
        "frame_dependent_retained_tensor_bytes",
    ):
        assert getattr(small, field) == getattr(large, field)
    assert small.descriptor_tensor_bytes == 0
    assert small.source_kinetic_program_tensor_bytes > 0
    assert small.maximum_resident_chart_payload_bytes > 0
    assert small.maximum_in_process_retained_tensor_bytes == (
        small.source_kinetic_program_tensor_bytes + small.maximum_resident_chart_payload_bytes
    )
    assert small.persistent_sample_time_tensor_bytes == 0
    assert small.persistent_frame_or_chart_sample_state_bytes == 0
    assert small.dense_sample_by_chart_tensor_bytes == 0
    assert small.frame_dependent_retained_tensor_bytes == 0


def test_lowering_and_materialized_payload_fail_closed_on_stale_provenance() -> None:
    program = _three_chart_program()
    lowering = lower_kinetic_multichart_to_native_topology(program)
    payload = materialize_kinetic_native_topology_chart(lowering, program, 1)

    stale_generation = replace(lowering, generation_digest="0" * 64)
    with pytest.raises(ValueError, match="generation digest mismatch"):
        stale_generation.assert_current(program)

    stale_chart = replace(
        lowering,
        charts=(
            replace(lowering.charts[0], payload_digest="f" * 64),
            *lowering.charts[1:],
        ),
    )
    with pytest.raises(ValueError, match="chart descriptors changed"):
        stale_chart.assert_current(program)

    payload.node_physical_lengths[0, 0].add_(0.125)
    with pytest.raises(ValueError, match="payload tensors changed"):
        payload.assert_current()

    program.binding.ray_coefficients[0].add_(0.25)
    with pytest.raises(ValueError, match="source content digest mismatch"):
        lowering.assert_current(program)
