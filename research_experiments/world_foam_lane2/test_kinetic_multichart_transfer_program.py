from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest
import torch
from kinetic_multichart_transfer_program import (
    compile_kinetic_multichart_p0_program,
    dispatch_kinetic_chart_index,
    evaluate_kinetic_multichart_p0_transfer,
    exact_streamed_kinetic_p0_replay,
    kinetic_multichart_material_mse_vjp,
    reduce_kinetic_multichart_mse_to_node_transfers,
    refresh_kinetic_multichart_p0_transfer,
)
from kinetic_owner_chart_compiler import compile_exact_kinetic_owner_charts
from kinetic_power_word_compiler import AffineKineticPowerSites

DTYPE = torch.float64


def _static_x_ray() -> torch.Tensor:
    return torch.tensor(
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        dtype=DTYPE,
    )


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
        positions0=torch.tensor([[float(value) for value in row] for row in positions], dtype=DTYPE),
        velocities=torch.tensor([[float(value) for value in row] for row in velocities], dtype=DTYPE),
        weight_coefficients=torch.tensor([[float(value) for value in row] for row in weights], dtype=DTYPE),
    )


def _three_chart_fixture(node_count: int = 3):
    sites = _sites_from_ray_lines(
        slopes=[(0, 0), (-2, 0)],
        intercepts=[(0, 0, 0), (1, -1, 0)],
    )
    ray = _static_x_ray()
    owner_program = compile_exact_kinetic_owner_charts(
        sites,
        ray,
        t_min=-2,
        t_max=2,
        near=0,
        far=1,
    )
    assert owner_program.passed
    program = compile_kinetic_multichart_p0_program(
        owner_program,
        sites,
        ray,
        node_count=node_count,
    )
    color = torch.tensor([[0.72, 0.31, 0.12], [0.72, 0.31, 0.12]], dtype=DTYPE)
    transfer = refresh_kinetic_multichart_p0_transfer(
        program,
        torch.tensor([0.25, 0.8], dtype=DTYPE),
        color,
    )
    return program, transfer


def test_exact_right_continuous_seam_dispatch_and_streamed_replay_parity() -> None:
    program, transfer = _three_chart_fixture(node_count=2)
    epsilon = Fraction(1, 1 << 20)
    points = (
        Fraction(-2),
        Fraction(-1) - epsilon,
        Fraction(-1),
        Fraction(0),
        Fraction(1) - epsilon,
        Fraction(1),
        Fraction(2),
    )

    assert tuple(dispatch_kinetic_chart_index(program, point) for point in points) == (
        0,
        0,
        1,
        1,
        1,
        2,
        2,
    )
    times = torch.tensor([float(point) for point in points], dtype=DTYPE)
    compact = evaluate_kinetic_multichart_p0_transfer(
        transfer,
        times,
        sample_block_size=3,
    )
    exact = exact_streamed_kinetic_p0_replay(
        transfer,
        times,
        sample_block_size=2,
    )
    # Equal run colors make the Lie coordinates affine in this moving-cut
    # fixture, so rank two is an exact temporal closure up to float64 error.
    torch.testing.assert_close(compact, exact, rtol=3.0e-13, atol=3.0e-13)


def test_source_and_program_content_digests_fail_closed_when_replaced() -> None:
    program, _ = _three_chart_fixture()
    assert len(program.binding.source_content_digest) == 64
    assert len(program.binding.program_semantic_digest) == 64

    bad_binding = replace(program.binding, source_content_digest="0" * 64)
    stale = replace(program, binding=bad_binding)
    with pytest.raises(ValueError, match="source content digest mismatch"):
        stale.assert_current()

    bad_semantics = replace(program.binding, program_semantic_digest="f" * 64)
    stale = replace(program, binding=bad_semantics)
    with pytest.raises(ValueError, match="semantic digest mismatch"):
        stale.assert_current()

    # Frozen dataclasses cannot make tensors immutable, so the content digest
    # must also reject in-place source mutation.
    program.binding.ray_coefficients[0].add_(1.0)
    with pytest.raises(ValueError, match="source content digest mismatch"):
        program.assert_current()


def test_multichart_reverse_structure_is_independent_of_requested_frame_count() -> None:
    program, transfer = _three_chart_fixture(node_count=4)
    background = torch.tensor([0.02, 0.03, 0.05], dtype=DTYPE)
    results = []
    for frame_count in (9, 105):
        times = torch.linspace(-2, 2, frame_count, dtype=DTYPE)
        targets = torch.stack(
            (
                0.28 + 0.01 * times,
                0.24 - 0.02 * times,
                torch.full_like(times, 0.35),
            ),
            dim=1,
        )
        results.append(
            kinetic_multichart_material_mse_vjp(
                transfer,
                times,
                targets,
                background=background,
                sample_block_size=4,
            )
        )

    small, large = (result.accounting for result in results)
    for key in (
        "chart_count",
        "compile_node_count",
        "world_node_replay_count",
        "material_prefix_reverse_node_count",
        "structural_tensor_bytes",
        "reverse_structural_tensor_bytes",
        "returned_node_transfer_cotangent_bytes",
        "peak_sample_block_bytes",
        "frame_dependent_structural_bytes",
        "dense_track_chart_refinement_bytes",
    ):
        assert small[key] == large[key]
    assert small["requested_sample_count"] == 9
    assert large["requested_sample_count"] == 105
    assert large["sample_to_node_linear_interactions"] > small["sample_to_node_linear_interactions"]
    assert small["frame_dependent_structural_bytes"] == 0
    assert small["dense_track_chart_refinement_bytes"] == 0
    assert program.total_node_count == 3 * 4
    assert tuple(tuple(gradient.shape) for gradient in results[0].grad_chart_node_transfers) == ((4, 4), (4, 4), (4, 4))
    assert all(torch.isfinite(result.grad_site_density).all() for result in results)
    assert all(torch.isfinite(result.grad_site_color).all() for result in results)


def test_accumulated_multichart_material_vjp_matches_finite_differences() -> None:
    program, _ = _three_chart_fixture(node_count=5)
    density = torch.tensor([0.3, 0.72], dtype=DTYPE)
    color = torch.tensor([[0.83, 0.2, 0.11], [0.12, 0.61, 0.91]], dtype=DTYPE)
    times = torch.tensor([-1.6, -0.7, -0.1, 0.65, 1.4], dtype=DTYPE)
    targets = torch.tensor(
        [
            [0.21, 0.31, 0.27],
            [0.25, 0.28, 0.32],
            [0.3, 0.24, 0.35],
            [0.34, 0.22, 0.38],
            [0.37, 0.2, 0.4],
        ],
        dtype=DTYPE,
    )
    background = torch.tensor([0.02, 0.04, 0.06], dtype=DTYPE)
    transfer = refresh_kinetic_multichart_p0_transfer(program, density, color)
    analytic = kinetic_multichart_material_mse_vjp(
        transfer,
        times,
        targets,
        background=background,
        sample_block_size=2,
    )

    def loss_for(candidate_density: torch.Tensor, candidate_color: torch.Tensor) -> torch.Tensor:
        candidate = refresh_kinetic_multichart_p0_transfer(
            program,
            candidate_density,
            candidate_color,
        )
        return kinetic_multichart_material_mse_vjp(
            candidate,
            times,
            targets,
            background=background,
            sample_block_size=3,
        ).loss

    epsilon = 1.0e-6
    density_fd = torch.zeros_like(density)
    for index in range(density.numel()):
        plus = density.clone()
        minus = density.clone()
        plus[index] += epsilon
        minus[index] -= epsilon
        density_fd[index] = (loss_for(plus, color) - loss_for(minus, color)) / (2.0 * epsilon)
    color_fd = torch.zeros_like(color)
    for row in range(color.shape[0]):
        for column in range(color.shape[1]):
            plus = color.clone()
            minus = color.clone()
            plus[row, column] += epsilon
            minus[row, column] -= epsilon
            color_fd[row, column] = (loss_for(density, plus) - loss_for(density, minus)) / (2.0 * epsilon)

    torch.testing.assert_close(analytic.grad_site_density, density_fd, rtol=3.0e-6, atol=3.0e-8)
    torch.testing.assert_close(analytic.grad_site_color, color_fd, rtol=3.0e-6, atol=3.0e-8)


def test_reduction_only_api_exposes_same_node_cotangent_without_material_replay() -> None:
    _, transfer = _three_chart_fixture(node_count=4)
    times = torch.tensor([-1.7, -0.6, 0.2, 1.3], dtype=DTYPE)
    targets = torch.tensor(
        [
            [0.21, 0.31, 0.27],
            [0.25, 0.28, 0.32],
            [0.3, 0.24, 0.35],
            [0.37, 0.2, 0.4],
        ],
        dtype=DTYPE,
    )
    background = torch.tensor([0.02, 0.04, 0.06], dtype=DTYPE)
    reduction = reduce_kinetic_multichart_mse_to_node_transfers(
        transfer,
        times,
        targets,
        background=background,
        sample_block_size=2,
        return_predictions=True,
    )
    full = kinetic_multichart_material_mse_vjp(
        transfer,
        times,
        targets,
        background=background,
        sample_block_size=2,
        return_predictions=True,
    )

    torch.testing.assert_close(reduction.loss, full.loss)
    assert reduction.predictions is not None and full.predictions is not None
    torch.testing.assert_close(reduction.predictions, full.predictions)
    for reduced, material in zip(
        reduction.grad_chart_node_transfers,
        full.grad_chart_node_transfers,
        strict=True,
    ):
        torch.testing.assert_close(reduced, material)
    assert reduction.accounting["world_node_replay_count"] == 0
    assert reduction.accounting["material_prefix_reverse_node_count"] == 0
    assert full.accounting["world_node_replay_count"] == transfer.program.total_node_count
    assert full.accounting["material_prefix_reverse_node_count"] == transfer.program.total_node_count
    assert reduction.accounting["frame_dependent_structural_bytes"] == 0
    assert reduction.frame_dependent_structural_bytes == 0
    assert reduction.frozen_program_semantics
    assert not reduction.chart_endpoint_vjp_implemented
    assert not reduction.node_time_vjp_implemented
    assert not reduction.sample_weight_vjp_implemented
    assert reduction.accounting["frozen_program_semantics"]
    assert not reduction.accounting["chart_endpoint_gradients_emitted"]
    assert not reduction.accounting["node_time_gradients_emitted"]
    assert not reduction.accounting["sample_weight_gradients_emitted"]


def test_irrational_endpoint_neighborhood_is_oriented_but_evaluation_fails_closed() -> None:
    sites = _sites_from_ray_lines(
        slopes=[(0, 0), (-2, 0)],
        intercepts=[(0, 0, 0), (1, 0, -2)],
    )
    ray = _static_x_ray()
    owner_program = compile_exact_kinetic_owner_charts(
        sites,
        ray,
        t_min=0,
        t_max=1,
        near=0,
        far=2,
    )
    program = compile_kinetic_multichart_p0_program(
        owner_program,
        sites,
        ray,
        node_count=4,
    )
    transfer = refresh_kinetic_multichart_p0_transfer(
        program,
        torch.tensor([0.2, 0.6], dtype=DTYPE),
        torch.tensor([[0.8, 0.2, 0.1], [0.1, 0.5, 0.9]], dtype=DTYPE),
    )
    algebraic_sample = 2.0**-0.5

    assert dispatch_kinetic_chart_index(program, algebraic_sample) in {0, 1}
    assert program.unresolved_algebraic_endpoint_count == 1
    with pytest.raises(ValueError, match="unresolved algebraic endpoint neighborhood"):
        evaluate_kinetic_multichart_p0_transfer(
            transfer,
            torch.tensor([algebraic_sample], dtype=DTYPE),
        )
