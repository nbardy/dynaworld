from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest
import torch
from kinetic_active_owner_chart_compiler import compile_active_kinetic_owner_charts
from kinetic_multichart_stable_stratum_vjp import (
    DERIVATIVE_SCOPE,
    bind_kinetic_multichart_stable_stratum_vjp,
    kinetic_multichart_stable_stratum_mse_vjp,
)
from kinetic_multichart_transfer_program import (
    compile_kinetic_multichart_p0_program,
    dispatch_kinetic_chart_index,
    refresh_kinetic_multichart_p0_transfer,
)
from kinetic_power_word_compiler import AffineKineticPowerSites
from transfer_lie_chart import transfer_lie_decode, transfer_lie_encode

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


def _fixture(node_count: int = 4):
    sites = _sites_from_ray_lines(
        slopes=[(0, 0), (-2, 0)],
        intercepts=[(0, 0, 0), (1, -1, 0)],
    )
    ray = torch.tensor(
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        dtype=DTYPE,
    )
    owner_program = compile_active_kinetic_owner_charts(
        sites,
        ray,
        t_min=-2,
        t_max=2,
        near=0,
        far=1,
    )
    program = compile_kinetic_multichart_p0_program(
        owner_program,
        sites,
        ray,
        node_count=node_count,
    )
    transfer = refresh_kinetic_multichart_p0_transfer(
        program,
        torch.tensor([0.31, 0.77], dtype=DTYPE),
        torch.tensor(
            [[0.82, 0.18, 0.11], [0.14, 0.63, 0.91]],
            dtype=DTYPE,
        ),
    )
    return program, transfer


def _frozen_chart_node_transfers(
    chart,
    positions0: torch.Tensor,
    velocities: torch.Tensor,
    weight_coefficients: torch.Tensor,
    ray: torch.Tensor,
    density: torch.Tensor,
    color: torch.Tensor,
    *,
    near: float,
    far: float,
) -> torch.Tensor:
    rows = []
    owners = tuple(int(owner) for owner in chart.owners.tolist())
    for time in chart.schedule.node_times:
        positions = positions0 + time * velocities
        powers = torch.stack((torch.ones_like(time), time, time.square()))[: weight_coefficients.shape[1]]
        weights = weight_coefficients @ powers
        origin = ray[:3] + time * ray[3:6]
        direction = ray[6:9] + time * ray[9:12]
        cuts = [torch.as_tensor(near, dtype=DTYPE)]
        for left, right in zip(owners[:-1], owners[1:], strict=True):
            normal = 2.0 * (positions[right] - positions[left])
            intercept = (
                torch.dot(normal, origin)
                + torch.dot(positions[left], positions[left])
                - torch.dot(positions[right], positions[right])
                - weights[left]
                + weights[right]
            )
            cuts.append(-intercept / torch.dot(normal, direction))
        cuts.append(torch.as_tensor(far, dtype=DTYPE))
        lengths = torch.linalg.vector_norm(direction) * (torch.stack(cuts)[1:] - torch.stack(cuts)[:-1])
        beta_total = torch.ones((), dtype=DTYPE)
        moment_total = torch.zeros(3, dtype=DTYPE)
        for run_id, owner in enumerate(owners):
            optical_depth = density[owner] * lengths[run_id]
            beta = torch.exp(-optical_depth)
            alpha = -torch.expm1(-optical_depth)
            moment_total = moment_total + beta_total * alpha * color[owner]
            beta_total = beta_total * beta
        rows.append(torch.cat((beta_total.reshape(1), moment_total)))
    return torch.stack(rows)


def _frozen_compact_mse(
    program,
    times: torch.Tensor,
    targets: torch.Tensor,
    background: torch.Tensor,
    positions0: torch.Tensor,
    velocities: torch.Tensor,
    weight_coefficients: torch.Tensor,
    ray: torch.Tensor,
    density: torch.Tensor,
    color: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    node_charts = tuple(
        transfer_lie_encode(
            _frozen_chart_node_transfers(
                chart,
                positions0,
                velocities,
                weight_coefficients,
                ray,
                density,
                color,
                near=float(program.binding.program.near),
                far=float(program.binding.program.far),
            )
        )
        for chart in program.charts
    )
    predictions = []
    for time in times:
        chart_id = dispatch_kinetic_chart_index(program, float(time.item()))
        weights = program.charts[chart_id].schedule.sample_to_node_weights(time.reshape(1)).weights
        sample_transfer = transfer_lie_decode(weights @ node_charts[chart_id])[0]
        predictions.append(sample_transfer[1:] + sample_transfer[0] * background)
    prediction = torch.stack(predictions)
    return (prediction - targets).square().mean(), prediction


def test_end_to_end_vjp_matches_frozen_program_autograd() -> None:
    program, transfer = _fixture(node_count=5)
    provenance = bind_kinetic_multichart_stable_stratum_vjp(program)
    times = torch.tensor([-1.72, -1.21, -0.64, 0.14, 0.73, 1.28, 1.81], dtype=DTYPE)
    targets = torch.tensor(
        [
            [0.18, 0.25, 0.31],
            [0.22, 0.29, 0.27],
            [0.31, 0.23, 0.36],
            [0.35, 0.19, 0.42],
            [0.29, 0.33, 0.38],
            [0.24, 0.37, 0.34],
            [0.2, 0.4, 0.3],
        ],
        dtype=DTYPE,
    )
    background = torch.tensor([0.03, 0.05, 0.07], dtype=DTYPE)
    analytic = kinetic_multichart_stable_stratum_mse_vjp(
        transfer,
        provenance,
        times,
        targets,
        background=background,
        sample_block_size=3,
        return_predictions=True,
    )

    leaves = [
        value.clone().requires_grad_(True)
        for value in (
            program.binding.sites.positions0,
            program.binding.sites.velocities,
            program.binding.sites.weight_coefficients,
            program.binding.ray_coefficients,
            transfer.site_density,
            transfer.site_color,
        )
    ]
    oracle_loss, oracle_prediction = _frozen_compact_mse(
        program,
        times,
        targets,
        background,
        *leaves,
    )
    oracle_loss.backward()

    torch.testing.assert_close(analytic.loss, oracle_loss.detach())
    assert analytic.predictions is not None
    torch.testing.assert_close(analytic.predictions, oracle_prediction.detach())
    for observed, leaf in zip(
        (
            analytic.grad_positions0,
            analytic.grad_velocities,
            analytic.grad_weight_coefficients,
            analytic.grad_ray_coefficients,
            analytic.grad_site_density,
            analytic.grad_site_color,
        ),
        leaves,
        strict=True,
    ):
        assert leaf.grad is not None
        torch.testing.assert_close(observed, leaf.grad, rtol=7.0e-12, atol=7.0e-12)

    assert analytic.derivative_scope == DERIVATIVE_SCOPE
    assert analytic.geometry_vjp_implemented
    assert analytic.material_vjp_implemented
    assert not analytic.event_time_derivatives_included
    assert not analytic.chart_endpoint_derivatives_included
    assert not analytic.sample_dispatch_derivatives_included
    assert not analytic.node_time_or_rank_derivatives_included
    assert not analytic.continuous_geometry_jacobian_certified
    assert not analytic.continuous_geometry_approximation_certified
    assert analytic.accounting["frame_dependent_reverse_tape_bytes"] == 0
    assert analytic.accounting["retained_requested_sample_bytes"] == 0
    assert analytic.accounting["dense_sample_by_chart_state_bytes"] == 0
    assert analytic.accounting["returned_prediction_bytes"] == times.numel() * 3 * 8


def test_world_reverse_shape_and_work_are_independent_of_frame_density() -> None:
    program, transfer = _fixture(node_count=4)
    provenance = bind_kinetic_multichart_stable_stratum_vjp(program)
    background = torch.tensor([0.02, 0.04, 0.06], dtype=DTYPE)
    results = []
    for frame_count in (9, 105):
        times = torch.linspace(-1.9, 1.9, frame_count, dtype=DTYPE)
        targets = torch.stack(
            (
                0.28 + 0.01 * times,
                0.24 - 0.02 * times,
                torch.full_like(times, 0.35),
            ),
            dim=1,
        )
        results.append(
            kinetic_multichart_stable_stratum_mse_vjp(
                transfer,
                provenance,
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
        "peak_chart_node_count",
        "peak_chart_run_count",
        "active_run_node_interactions",
        "active_cut_node_interactions",
        "owner_margin_evaluations",
        "world_geometry_material_reverse_node_count",
        "node_transfer_cotangent_bytes",
        "parameter_gradient_bytes",
        "peak_sample_block_bytes",
        "frame_dependent_reverse_tape_bytes",
        "retained_requested_sample_bytes",
        "dense_sample_by_chart_state_bytes",
    ):
        assert small[key] == large[key]
    assert small["requested_sample_count"] == 9
    assert large["requested_sample_count"] == 105
    assert large["sample_to_node_linear_interactions"] > small["sample_to_node_linear_interactions"]
    assert small["world_reverse_independent_of_requested_frame_count"]
    assert small["world_reverse_scaling"] == "O(sum_c J_c * R_c)"
    assert small["frame_dependent_reverse_tape_bytes"] == 0
    assert results[0].predictions is None
    assert results[1].predictions is None
    assert tuple(result.grad_ray_coefficients.shape for result in results) == (
        torch.Size([12]),
        torch.Size([12]),
    )


def test_binding_rejects_stale_program_and_certificate_provenance() -> None:
    program, transfer = _fixture()
    provenance = bind_kinetic_multichart_stable_stratum_vjp(program)
    assert len(provenance.binding_digest) == 64
    assert len(provenance.chart_topology_certificate_ids) == program.chart_count
    assert all(len(identifier) == 64 for identifier in provenance.chart_topology_certificate_ids)

    stale_digest = replace(provenance, binding_digest="0" * 64)
    with pytest.raises(ValueError, match="binding digest mismatch"):
        stale_digest.assert_current(program)

    stale_certificate = replace(
        provenance,
        chart_topology_certificate_ids=(
            "f" * 64,
            *provenance.chart_topology_certificate_ids[1:],
        ),
    )
    with pytest.raises(ValueError, match="provenance does not match"):
        stale_certificate.assert_current(program)

    program.binding.sites.positions0[0, 0].add_(0.125)
    with pytest.raises(ValueError, match="source content digest mismatch"):
        kinetic_multichart_stable_stratum_mse_vjp(
            transfer,
            provenance,
            torch.tensor([-1.5, 0.0, 1.5], dtype=DTYPE),
            torch.zeros((3, 3), dtype=DTYPE),
            background=torch.zeros(3, dtype=DTYPE),
        )
