from __future__ import annotations

import inspect
from dataclasses import replace

import pytest
import torch
from kinetic_chart_transfer_bridge import kinetic_chart_p0_node_material_vjp
from kinetic_multichart_transfer_program import (
    compile_kinetic_multichart_p0_program,
    refresh_kinetic_multichart_p0_transfer,
)
from kinetic_native_precompiled_length_oracle import (
    kinetic_native_precompiled_length_node_vjp,
    refresh_kinetic_native_precompiled_length_world,
)
from kinetic_native_topology_lowering import (
    lower_kinetic_multichart_to_native_topology,
    materialize_kinetic_native_topology_chart,
)
from kinetic_owner_chart_compiler import compile_exact_kinetic_owner_charts
from kinetic_power_word_compiler import AffineKineticPowerSites
from transfer_lie_chart import transfer_lie_encode, transfer_lie_encode_vjp

DTYPE = torch.float64


def _fixture(node_count: int = 5):
    # The active source word is deliberately (3,1,2), while compact source
    # rows are sorted (1,2,3). Site zero is globally present but unreferenced.
    sites = AffineKineticPowerSites(
        positions0=torch.tensor(
            [
                [100.0, 100.0, 100.0],
                [1.5, 0.0, 0.0],
                [2.5, 0.0, 0.0],
                [0.5, 0.0, 0.0],
            ],
            dtype=DTYPE,
        ),
        velocities=torch.zeros((4, 3), dtype=DTYPE),
        weight_coefficients=torch.zeros((4, 1), dtype=DTYPE),
    )
    ray = torch.tensor(
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        dtype=DTYPE,
    )
    owner_program = compile_exact_kinetic_owner_charts(
        sites,
        ray,
        t_min=-1,
        t_max=1,
        near=0,
        far=3,
    )
    program = compile_kinetic_multichart_p0_program(
        owner_program,
        sites,
        ray,
        node_count=node_count,
    )
    assert program.chart_count == 1
    assert program.charts[0].owner_word == (3, 1, 2)
    density = torch.tensor([0.91, 0.37, 0.68, 0.24], dtype=DTYPE)
    color = torch.tensor(
        [
            [0.02, 0.03, 0.04],
            [0.82, 0.16, 0.11],
            [0.12, 0.71, 0.91],
            [0.46, 0.31, 0.77],
        ],
        dtype=DTYPE,
    )
    transfer = refresh_kinetic_multichart_p0_transfer(program, density, color)
    lowering = lower_kinetic_multichart_to_native_topology(program)
    payload = materialize_kinetic_native_topology_chart(lowering, program, 0)
    return program, transfer, payload, density, color


def _independent_node_transfers(
    compact_rgba: torch.Tensor,
    lengths: torch.Tensor,
    owners: tuple[int, ...],
) -> torch.Tensor:
    rows = []
    for node_lengths in lengths:
        # Compose 4x4 affine RGB transfer matrices. This is intentionally a
        # different oracle shape from the prefix recurrence under test.
        total = torch.eye(4, dtype=DTYPE)
        for run_index, owner in enumerate(owners):
            optical_depth = compact_rgba[owner, 3] * node_lengths[run_index]
            beta = torch.exp(-optical_depth)
            alpha = -torch.expm1(-optical_depth)
            segment = torch.eye(4, dtype=DTYPE)
            segment[:3, :3] = beta * torch.eye(3, dtype=DTYPE)
            segment[:3, 3] = alpha * compact_rgba[owner, :3]
            total = total @ segment
        rows.append(torch.cat((total[0, 0].reshape(1), total[:3, 3])))
    return torch.stack(rows)


def _independent_node_charts(
    compact_rgba: torch.Tensor,
    lengths: torch.Tensor,
    owners: tuple[int, ...],
) -> torch.Tensor:
    transfers = _independent_node_transfers(compact_rgba, lengths, owners)
    ordered_density = compact_rgba[list(owners), 3]
    kappa = torch.sum(lengths * ordered_density.unsqueeze(0), dim=1)
    small = kappa.abs() < 1.0e-4
    kappa2 = kappa.square()
    series = 1.0 + 0.5 * kappa + kappa2 / 12.0 - kappa2.square() / 720.0 + kappa2.pow(3) / 30240.0
    denominator = -torch.expm1(-kappa)
    inverse_phi = torch.where(
        small,
        series,
        kappa / torch.where(small, torch.ones_like(denominator), denominator),
    )
    return torch.cat((kappa[:, None], inverse_phi[:, None] * transfers[:, 1:]), dim=1)


def test_compact_rgba_packing_and_forward_match_existing_kinetic_nodes() -> None:
    _, transfer, payload, density, color = _fixture()
    world = refresh_kinetic_native_precompiled_length_world(
        payload,
        density,
        color,
    )

    assert tuple(payload.topology.source_site_ids.tolist()) == (1, 2, 3)
    compact_owners = payload.topology.word_owner_i32.to(dtype=torch.long)
    source_owners = payload.topology.source_site_ids.index_select(0, compact_owners)
    assert tuple(source_owners.tolist()) == (3, 1, 2)
    expected_rgba = torch.cat(
        (
            color.index_select(0, payload.topology.source_site_ids),
            density.index_select(0, payload.topology.source_site_ids)[:, None],
        ),
        dim=1,
    )
    torch.testing.assert_close(world.compact_site_rgba, expected_rgba)
    expected_chart = transfer_lie_encode(transfer.chart_node_transfers[0])
    torch.testing.assert_close(world.node_charts, expected_chart, rtol=2.0e-15, atol=2.0e-15)
    assert not torch.allclose(world.node_charts, transfer.chart_node_transfers[0])
    assert tuple(world.node_charts.shape) == (payload.spec.node_count, 4)
    assert not world.requested_frame_sampling_used
    assert not world.frame_or_sample_axis_retained
    assert not world.native_execution_ready


def test_manual_rgba_and_length_vjp_match_autograd_and_finite_differences() -> None:
    _, transfer, payload, density, color = _fixture(node_count=4)
    world = refresh_kinetic_native_precompiled_length_world(
        payload,
        density,
        color,
    )
    grad_node_chart = torch.linspace(
        -0.41,
        0.73,
        world.node_count * 4,
        dtype=DTYPE,
    ).reshape(world.node_count, 4)
    analytic = kinetic_native_precompiled_length_node_vjp(
        world,
        grad_node_chart,
    )

    compact_rgba = world.compact_site_rgba.clone().requires_grad_(True)
    lengths = payload.node_physical_lengths.clone().requires_grad_(True)
    owners = tuple(int(owner) for owner in payload.topology.word_owner_i32.tolist())
    oracle_nodes = _independent_node_charts(compact_rgba, lengths, owners)
    objective = torch.sum(oracle_nodes * grad_node_chart)
    objective.backward()
    assert compact_rgba.grad is not None
    assert lengths.grad is not None
    torch.testing.assert_close(
        analytic.grad_compact_site_rgba,
        compact_rgba.grad,
        rtol=4.0e-12,
        atol=4.0e-12,
    )
    torch.testing.assert_close(
        analytic.grad_node_physical_lengths,
        lengths.grad,
        rtol=4.0e-12,
        atol=4.0e-12,
    )

    raw_node_cotangent = transfer_lie_encode_vjp(
        transfer.chart_node_transfers[0],
        grad_node_chart,
    )
    material_density_grad, material_color_grad = kinetic_chart_p0_node_material_vjp(
        transfer.chart_transfer(0),
        raw_node_cotangent,
    )
    global_density_grad = torch.zeros_like(density)
    global_color_grad = torch.zeros_like(color)
    global_density_grad.index_add_(
        0,
        payload.topology.source_site_ids,
        analytic.grad_compact_site_rgba[:, 3],
    )
    global_color_grad.index_add_(
        0,
        payload.topology.source_site_ids,
        analytic.grad_compact_site_rgba[:, :3],
    )
    torch.testing.assert_close(global_density_grad, material_density_grad)
    torch.testing.assert_close(global_color_grad, material_color_grad)

    material_direction = torch.sin(torch.arange(compact_rgba.numel(), dtype=DTYPE).reshape(compact_rgba.shape) + 0.37)
    length_direction = torch.cos(torch.arange(lengths.numel(), dtype=DTYPE).reshape(lengths.shape) + 0.19)

    def objective_for(
        candidate_rgba: torch.Tensor,
        candidate_lengths: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sum(
            _independent_node_charts(
                candidate_rgba,
                candidate_lengths,
                owners,
            )
            * grad_node_chart
        )

    epsilon = 2.0e-6
    material_fd = (
        objective_for(compact_rgba.detach() + epsilon * material_direction, lengths.detach())
        - objective_for(compact_rgba.detach() - epsilon * material_direction, lengths.detach())
    ) / (2.0 * epsilon)
    length_fd = (
        objective_for(compact_rgba.detach(), lengths.detach() + epsilon * length_direction)
        - objective_for(compact_rgba.detach(), lengths.detach() - epsilon * length_direction)
    ) / (2.0 * epsilon)
    torch.testing.assert_close(
        torch.sum(analytic.grad_compact_site_rgba * material_direction),
        material_fd,
        rtol=2.0e-9,
        atol=2.0e-10,
    )
    torch.testing.assert_close(
        torch.sum(analytic.grad_node_physical_lengths * length_direction),
        length_fd,
        rtol=2.0e-9,
        atol=2.0e-10,
    )

    assert analytic.accounting["requested_frame_count_used"] == 0
    assert analytic.accounting["persistent_sample_time_tensor_bytes"] == 0
    assert analytic.accounting["persistent_frame_or_sample_tensor_bytes"] == 0
    assert not analytic.accounting["frame_by_run_reverse_state_allocated"]
    assert analytic.accounting["reverse_scaling"] == "O(J * R)"
    parameters = inspect.signature(kinetic_native_precompiled_length_node_vjp).parameters
    assert "times" not in parameters
    assert "targets" not in parameters
    assert "frame_count" not in parameters


@pytest.mark.parametrize("referenced_density", (1.0e-18, 1.0e4))
def test_direct_lie_kappa_and_vjp_stay_finite_at_tiny_and_underflowing_optical_depth(
    referenced_density: float,
) -> None:
    _, _, payload, density, color = _fixture(node_count=4)
    density = density.clone()
    density[payload.topology.source_site_ids] = referenced_density
    world = refresh_kinetic_native_precompiled_length_world(payload, density, color)
    owners = tuple(int(owner) for owner in payload.topology.word_owner_i32.tolist())
    compact_density = world.compact_site_rgba[:, 3]
    expected_kappa = torch.sum(
        payload.node_physical_lengths * compact_density[list(owners)].unsqueeze(0),
        dim=1,
    )

    torch.testing.assert_close(world.node_charts[:, 0], expected_kappa, rtol=2.0e-15, atol=0.0)
    assert bool(torch.all(world.node_charts[:, 0] > 0.0).item())
    assert bool(torch.isfinite(world.node_charts).all().item())
    if referenced_density > 1.0:
        assert bool(torch.all(torch.exp(-expected_kappa) == 0.0).item())

    grad_node_chart = torch.linspace(-0.37, 0.61, world.node_count * 4, dtype=DTYPE).reshape(
        world.node_count,
        4,
    )
    analytic = kinetic_native_precompiled_length_node_vjp(world, grad_node_chart)
    compact_rgba = world.compact_site_rgba.clone().requires_grad_(True)
    lengths = payload.node_physical_lengths.clone().requires_grad_(True)
    objective = torch.sum(_independent_node_charts(compact_rgba, lengths, owners) * grad_node_chart)
    objective.backward()
    assert compact_rgba.grad is not None
    assert lengths.grad is not None
    torch.testing.assert_close(analytic.grad_compact_site_rgba, compact_rgba.grad, rtol=5.0e-12, atol=5.0e-12)
    torch.testing.assert_close(analytic.grad_node_physical_lengths, lengths.grad, rtol=5.0e-12, atol=5.0e-12)
    assert bool(torch.isfinite(analytic.grad_compact_site_rgba).all().item())
    assert bool(torch.isfinite(analytic.grad_node_physical_lengths).all().item())


def test_world_and_payload_provenance_fail_closed_after_mutation() -> None:
    _, _, payload, density, color = _fixture()
    world = refresh_kinetic_native_precompiled_length_world(
        payload,
        density,
        color,
    )
    stale = replace(world, world_generation_digest="0" * 64)
    with pytest.raises(ValueError, match="world generation mismatch"):
        stale.assert_current()

    world.compact_site_rgba[0, 0].add_(0.125)
    with pytest.raises(ValueError, match="world tensors changed"):
        world.assert_current()

    _, _, second_payload, second_density, second_color = _fixture()
    second_world = refresh_kinetic_native_precompiled_length_world(
        second_payload,
        second_density,
        second_color,
    )
    second_payload.node_physical_lengths[0, 0].add_(0.25)
    with pytest.raises(ValueError, match="payload tensors changed"):
        kinetic_native_precompiled_length_node_vjp(
            second_world,
            torch.ones_like(second_world.node_charts),
        )
