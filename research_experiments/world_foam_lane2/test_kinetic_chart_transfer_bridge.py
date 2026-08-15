from __future__ import annotations

import inspect
from fractions import Fraction

import pytest
import torch
from cell_path_optical_transfer_fixture import constant_run_element, scan
from kinetic_chart_transfer_bridge import (
    compile_kinetic_chart_p0_geometry,
    evaluate_kinetic_chart_p0_transfer,
    kinetic_chart_material_mse_vjp,
    refresh_kinetic_chart_p0_transfer,
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


def _moving_cut_fixture() -> tuple[AffineKineticPowerSites, torch.Tensor, object]:
    sites = _sites_from_ray_lines(
        slopes=[(0, 0), (-2, 0)],
        intercepts=[(0, 0, 0), (1, -1, 0)],
    )
    ray = _static_x_ray()
    program = compile_exact_kinetic_owner_charts(
        sites,
        ray,
        t_min=-2,
        t_max=2,
        near=0,
        far=1,
    )
    assert program.passed
    return sites, ray, program


def _compile_transfer(node_count: int = 5):
    sites, ray, program = _moving_cut_fixture()
    geometry = compile_kinetic_chart_p0_geometry(
        program,
        sites,
        ray,
        chart_id=1,
        node_count=node_count,
    )
    density = torch.tensor([0.35, 0.8], dtype=DTYPE)
    color = torch.tensor([[0.9, 0.2, 0.1], [0.1, 0.65, 0.95]], dtype=DTYPE)
    return geometry, refresh_kinetic_chart_p0_transfer(geometry, density, color)


def test_compile_nodes_match_independent_ordered_p0_replay() -> None:
    geometry, transfer = _compile_transfer()

    assert geometry.owner_word == (0, 1)
    assert geometry.exact_owner_and_cut_discovery_at_nodes
    assert not geometry.requested_frame_sampling_used
    for node_id, lengths in enumerate(geometry.node_physical_lengths):
        expected = scan(
            [
                constant_run_element(
                    transfer.site_density[owner],
                    length,
                    transfer.site_color[owner],
                )
                for owner, length in zip(geometry.owner_word, lengths, strict=True)
            ]
        )
        expected_tensor = torch.cat((expected.beta.reshape(1), expected.m))
        torch.testing.assert_close(transfer.node_transfers[node_id], expected_tensor)

    # The compact cardinal evaluator must reproduce every exact node row.
    evaluated = evaluate_kinetic_chart_p0_transfer(transfer, geometry.schedule.node_times)
    torch.testing.assert_close(evaluated, transfer.node_transfers, rtol=2.0e-13, atol=2.0e-13)


def test_compact_cross_time_material_vjp_matches_central_differences() -> None:
    geometry, transfer = _compile_transfer(node_count=6)
    times = torch.linspace(-0.8, 0.8, 9, dtype=DTYPE)
    targets = torch.stack(
        (
            0.25 + 0.03 * times,
            0.31 - 0.02 * times,
            0.42 + 0.01 * times,
        ),
        dim=1,
    )
    background = torch.tensor([0.02, 0.03, 0.05], dtype=DTYPE)
    result = kinetic_chart_material_mse_vjp(
        transfer,
        times,
        targets,
        background=background,
        frame_block_size=3,
        return_predictions=True,
    )

    epsilon = 1.0e-6

    def loss_for(density: torch.Tensor, color: torch.Tensor) -> torch.Tensor:
        candidate = refresh_kinetic_chart_p0_transfer(geometry, density, color)
        return kinetic_chart_material_mse_vjp(
            candidate,
            times,
            targets,
            background=background,
            frame_block_size=4,
        ).loss

    density_fd = torch.zeros_like(transfer.site_density)
    for index in range(density_fd.numel()):
        plus = transfer.site_density.clone()
        minus = transfer.site_density.clone()
        plus[index] += epsilon
        minus[index] -= epsilon
        density_fd[index] = (loss_for(plus, transfer.site_color) - loss_for(minus, transfer.site_color)) / (
            2.0 * epsilon
        )
    color_fd = torch.zeros_like(transfer.site_color)
    for row in range(color_fd.shape[0]):
        for column in range(color_fd.shape[1]):
            plus = transfer.site_color.clone()
            minus = transfer.site_color.clone()
            plus[row, column] += epsilon
            minus[row, column] -= epsilon
            color_fd[row, column] = (loss_for(transfer.site_density, plus) - loss_for(transfer.site_density, minus)) / (
                2.0 * epsilon
            )

    torch.testing.assert_close(result.grad_site_density, density_fd, rtol=2.0e-6, atol=2.0e-8)
    torch.testing.assert_close(result.grad_site_color, color_fd, rtol=2.0e-6, atol=2.0e-8)
    assert result.predictions is not None and result.predictions.shape == targets.shape
    assert result.geometry_gradients is None and result.event_time_gradients is None
    assert not result.geometry_vjp_implemented and not result.event_time_vjp_implemented


def test_structural_and_reverse_state_do_not_scale_with_requested_samples() -> None:
    geometry, transfer = _compile_transfer(node_count=5)
    background = torch.tensor([0.01, 0.02, 0.03], dtype=DTYPE)

    results = []
    for frame_count in (7, 103):
        times = torch.linspace(-0.75, 0.75, frame_count, dtype=DTYPE)
        targets = torch.full((frame_count, 3), 0.25, dtype=DTYPE)
        results.append(
            kinetic_chart_material_mse_vjp(
                transfer,
                times,
                targets,
                background=background,
                frame_block_size=3,
            )
        )

    small, large = (result.accounting for result in results)
    for key in (
        "compile_node_count",
        "world_node_replay_count",
        "structural_tensor_bytes",
        "reverse_node_tensor_bytes",
        "peak_sample_block_bytes",
        "frame_dependent_structural_bytes",
    ):
        assert small[key] == large[key]
    assert small["requested_sample_count"] == 7
    assert large["requested_sample_count"] == 103
    assert large["sample_to_node_linear_interactions"] > small["sample_to_node_linear_interactions"]
    assert small["frame_dependent_structural_bytes"] == 0


def test_irrational_seam_compiles_only_a_certified_safe_subset() -> None:
    sites = _sites_from_ray_lines(
        slopes=[(0, 0), (-2, 0)],
        intercepts=[(0, 0, 0), (1, 0, -2)],
    )
    ray = _static_x_ray()
    program = compile_exact_kinetic_owner_charts(
        sites,
        ray,
        t_min=0,
        t_max=1,
        near=0,
        far=2,
    )

    geometry = compile_kinetic_chart_p0_geometry(
        program,
        sites,
        ray,
        chart_id=0,
        node_count=4,
    )

    seam = 2.0**-0.5
    assert geometry.schedule.t_max < seam
    assert geometry.right_boundary_uncertainty > 0
    assert not geometry.full_algebraic_boundary_coverage
    assert geometry.safe_interval_is_certified_inside_owner_chart
    assert not geometry.seam_dispatch_implemented


def test_bridge_rejects_unbound_program_and_right_open_seam() -> None:
    sites, ray, program = _moving_cut_fixture()
    different_sites = _sites_from_ray_lines(
        slopes=[(0, 0), (-2, 0)],
        intercepts=[(0, 0, 0), (-100, 0, 0)],
    )
    with pytest.raises(ValueError, match="not bound|different active seam"):
        compile_kinetic_chart_p0_geometry(
            program,
            different_sites,
            ray,
            chart_id=1,
            node_count=4,
        )

    geometry = compile_kinetic_chart_p0_geometry(
        program,
        sites,
        ray,
        chart_id=1,
        node_count=4,
    )
    transfer = refresh_kinetic_chart_p0_transfer(
        geometry,
        torch.tensor([0.2, 0.3], dtype=DTYPE),
        torch.tensor([[0.2, 0.3, 0.4], [0.6, 0.5, 0.4]], dtype=DTYPE),
    )
    with pytest.raises(ValueError, match="right-open"):
        evaluate_kinetic_chart_p0_transfer(
            transfer,
            torch.tensor([geometry.schedule.t_max], dtype=DTYPE),
        )


def test_compile_and_refresh_apis_have_no_requested_frame_parameter() -> None:
    forbidden = {"frame_count", "requested_frame_count", "sample_count"}
    for function in (
        compile_kinetic_chart_p0_geometry,
        refresh_kinetic_chart_p0_transfer,
    ):
        assert not forbidden.intersection(inspect.signature(function).parameters)
