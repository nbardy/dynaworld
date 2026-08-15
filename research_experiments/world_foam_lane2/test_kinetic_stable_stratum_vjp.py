from __future__ import annotations

import inspect

import pytest
import torch
from kinetic_power_word_compiler import (
    AffineKineticPowerSites,
    discover_kinetic_power_word_at_time,
)
from kinetic_stable_stratum_vjp import (
    StableStratumError,
    StableStratumThresholds,
    kinetic_p0_compiler_node_vjp,
    kinetic_p0_node_physical_length_geometry_vjp,
    make_frozen_kinetic_owner_word,
)

DTYPE = torch.float64


def _stable_fixture() -> tuple[
    AffineKineticPowerSites,
    torch.Tensor,
    torch.Tensor,
    tuple,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    sites = AffineKineticPowerSites(
        positions0=torch.tensor(
            [
                [-0.25, 0.12, 0.0],
                [0.18, -0.15, 2.1],
                [-0.12, 0.22, 4.2],
            ],
            dtype=DTYPE,
        ),
        velocities=torch.tensor(
            [
                [0.025, -0.010, 0.040],
                [-0.018, 0.015, -0.025],
                [0.012, -0.020, 0.030],
            ],
            dtype=DTYPE,
        ),
        weight_coefficients=torch.tensor(
            [
                [0.08, -0.025, 0.012],
                [-0.04, 0.018, -0.008],
                [0.06, 0.011, 0.006],
            ],
            dtype=DTYPE,
        ),
    )
    rays = torch.tensor(
        [
            [
                0.04,
                -0.03,
                -1.0,
                0.012,
                -0.008,
                0.018,
                0.03,
                0.02,
                1.0,
                -0.006,
                0.004,
                0.012,
            ],
            [
                -0.08,
                0.06,
                -0.9,
                -0.009,
                0.011,
                -0.014,
                -0.025,
                0.035,
                0.97,
                0.005,
                -0.007,
                0.009,
            ],
        ],
        dtype=DTYPE,
    )
    node_times = torch.tensor([-0.35, 0.10, 0.45], dtype=DTYPE)
    words = []
    for ray in rays:
        discovered = [
            discover_kinetic_power_word_at_time(
                sites,
                ray,
                time=float(time),
                near=0.0,
                far=6.0,
            )
            for time in node_times
        ]
        owners = tuple(int(owner) for owner in discovered[0].word.owners.tolist())
        assert owners == (0, 1, 2)
        assert all(tuple(int(owner) for owner in item.word.owners.tolist()) == owners for item in discovered)
        words.append(make_frozen_kinetic_owner_word(owners))
    density = torch.tensor([0.37, 0.68, 0.29], dtype=DTYPE)
    color = torch.tensor(
        [[0.82, 0.16, 0.11], [0.12, 0.71, 0.91], [0.46, 0.31, 0.77]],
        dtype=DTYPE,
    )
    grad_transfer = torch.linspace(
        -0.43,
        0.67,
        rays.shape[0] * node_times.numel() * 4,
        dtype=DTYPE,
    ).reshape(rays.shape[0], node_times.numel(), 4)
    return sites, rays, node_times, tuple(words), density, color, grad_transfer


def _direct_fixed_word_transfer(
    positions0: torch.Tensor,
    velocities: torch.Tensor,
    weight_coefficients: torch.Tensor,
    rays: torch.Tensor,
    node_times: torch.Tensor,
    words: tuple,
    density: torch.Tensor,
    color: torch.Tensor,
    *,
    near: float,
    far: float,
) -> torch.Tensor:
    """Small independent differentiable oracle for the frozen owner program."""

    track_rows = []
    for track_id, word in enumerate(words):
        node_rows = []
        for time in node_times:
            positions = positions0 + time * velocities
            powers = torch.stack((torch.ones_like(time), time, time.square()))[: weight_coefficients.shape[1]]
            weights = weight_coefficients @ powers
            origin = rays[track_id, :3] + time * rays[track_id, 3:6]
            direction = rays[track_id, 6:9] + time * rays[track_id, 9:12]
            owners = tuple(int(owner) for owner in word.owners.tolist())
            cuts = [torch.as_tensor(near, dtype=DTYPE)]
            for left, right in zip(owners[:-1], owners[1:], strict=True):
                normal = 2.0 * (positions[right] - positions[left])
                denominator = torch.dot(normal, direction)
                intercept = (
                    torch.dot(normal, origin)
                    + torch.dot(positions[left], positions[left])
                    - torch.dot(positions[right], positions[right])
                    - weights[left]
                    + weights[right]
                )
                cuts.append(-intercept / denominator)
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
            node_rows.append(torch.cat((beta_total.reshape(1), moment_total)))
        track_rows.append(torch.stack(node_rows))
    return torch.stack(track_rows)


def _direct_fixed_word_physical_lengths(
    positions0: torch.Tensor,
    velocities: torch.Tensor,
    weight_coefficients: torch.Tensor,
    rays: torch.Tensor,
    node_times: torch.Tensor,
    words: tuple,
    *,
    near: float,
    far: float,
) -> torch.Tensor:
    """Independent differentiable geometry oracle returning ``[P,J,R]``."""

    track_rows = []
    for track_id, word in enumerate(words):
        node_rows = []
        for time in node_times:
            positions = positions0 + time * velocities
            powers = torch.stack((torch.ones_like(time), time, time.square()))[: weight_coefficients.shape[1]]
            weights = weight_coefficients @ powers
            origin = rays[track_id, :3] + time * rays[track_id, 3:6]
            direction = rays[track_id, 6:9] + time * rays[track_id, 9:12]
            owners = tuple(int(owner) for owner in word.owners.tolist())
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
            depths = torch.stack(cuts)
            node_rows.append(torch.linalg.vector_norm(direction) * (depths[1:] - depths[:-1]))
        track_rows.append(torch.stack(node_rows))
    return torch.stack(track_rows)


def _direct_node_transfers_from_lengths(
    physical_lengths: torch.Tensor,
    word,
    density: torch.Tensor,
    color: torch.Tensor,
) -> torch.Tensor:
    rows = []
    owners = tuple(int(owner) for owner in word.owners.tolist())
    for lengths in physical_lengths:
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


def _manual_result():
    fixture = _stable_fixture()
    sites, rays, node_times, words, density, color, grad_transfer = fixture
    result = kinetic_p0_compiler_node_vjp(
        sites,
        rays,
        node_times,
        words,
        density,
        color,
        grad_transfer,
        near=0.0,
        far=6.0,
        continuous_topology_certificate_id="test-certified-stable-chart",
    )
    return fixture, result


def _one_track_fixture():
    sites, rays, node_times, words, density, color, grad_transfer = _stable_fixture()
    return (
        sites,
        rays[0].clone(),
        node_times,
        words[:1],
        density,
        color,
        grad_transfer[:1].clone(),
    )


def test_manual_sparse_vjp_matches_independent_autograd_and_directional_difference() -> None:
    (sites, rays, node_times, words, density, color, grad_transfer), result = _manual_result()
    leaves = [
        value.clone().requires_grad_(True)
        for value in (
            sites.positions0,
            sites.velocities,
            sites.weight_coefficients,
            rays,
            density,
            color,
        )
    ]
    oracle_transfer = _direct_fixed_word_transfer(
        *leaves[:4],
        node_times,
        words,
        *leaves[4:],
        near=0.0,
        far=6.0,
    )
    objective = torch.sum(oracle_transfer * grad_transfer)
    objective.backward()

    torch.testing.assert_close(result.node_transfers, oracle_transfer.detach())
    manual_gradients = (
        result.grad_positions0,
        result.grad_velocities,
        result.grad_weight_coefficients,
        result.grad_ray_coefficients,
        result.grad_site_density,
        result.grad_site_color,
    )
    for manual, leaf in zip(manual_gradients, leaves, strict=True):
        assert leaf.grad is not None
        torch.testing.assert_close(manual, leaf.grad, rtol=3.0e-12, atol=3.0e-12)

    directions = tuple(
        torch.sin(torch.arange(value.numel(), dtype=DTYPE).reshape(value.shape) + 0.37 * index)
        for index, value in enumerate(leaves, start=1)
    )
    predicted_directional = sum(
        torch.sum(gradient * direction) for gradient, direction in zip(manual_gradients, directions, strict=True)
    )

    def perturbed_objective(sign: float, epsilon: float) -> torch.Tensor:
        values = tuple(
            leaf.detach() + sign * epsilon * direction for leaf, direction in zip(leaves, directions, strict=True)
        )
        transfer = _direct_fixed_word_transfer(
            *values[:4],
            node_times,
            words,
            *values[4:],
            near=0.0,
            far=6.0,
        )
        return torch.sum(transfer * grad_transfer)

    epsilon = 2.0e-6
    observed_directional = (perturbed_objective(1.0, epsilon) - perturbed_objective(-1.0, epsilon)) / (2.0 * epsilon)
    torch.testing.assert_close(
        predicted_directional,
        observed_directional,
        rtol=2.0e-8,
        atol=2.0e-9,
    )


def test_node_physical_length_geometry_bridge_matches_directional_finite_difference() -> None:
    sites, rays, node_times, words, *_ = _one_track_fixture()
    grad_lengths = torch.linspace(
        -0.61,
        0.79,
        node_times.numel() * words[0].run_count,
        dtype=DTYPE,
    ).reshape(node_times.numel(), words[0].run_count)
    result = kinetic_p0_node_physical_length_geometry_vjp(
        sites,
        rays,
        node_times,
        words,
        grad_lengths,
        near=0.0,
        far=6.0,
        continuous_topology_certificate_id="test-certified-stable-chart",
        node_physical_length_cotangent_provenance_id="native-node-transfer-vjp-generation",
    )
    expected_lengths = _direct_fixed_word_physical_lengths(
        sites.positions0,
        sites.velocities,
        sites.weight_coefficients,
        rays.unsqueeze(0),
        node_times,
        words,
        near=0.0,
        far=6.0,
    )[0]
    torch.testing.assert_close(result.node_physical_lengths, expected_lengths)

    values = (
        sites.positions0,
        sites.velocities,
        sites.weight_coefficients,
        rays,
    )
    gradients = (
        result.grad_positions0,
        result.grad_velocities,
        result.grad_weight_coefficients,
        result.grad_ray_coefficients,
    )
    directions = tuple(
        torch.sin(torch.arange(value.numel(), dtype=DTYPE).reshape(value.shape) + 0.29 * index)
        for index, value in enumerate(values, start=1)
    )
    predicted_directional = sum(
        torch.sum(gradient * direction) for gradient, direction in zip(gradients, directions, strict=True)
    )

    def perturbed_objective(sign: float, epsilon: float) -> torch.Tensor:
        perturbed = tuple(
            value + sign * epsilon * direction for value, direction in zip(values, directions, strict=True)
        )
        lengths = _direct_fixed_word_physical_lengths(
            *perturbed[:3],
            perturbed[3].unsqueeze(0),
            node_times,
            words,
            near=0.0,
            far=6.0,
        )[0]
        return torch.sum(lengths * grad_lengths)

    epsilon = 2.0e-6
    observed_directional = (perturbed_objective(1.0, epsilon) - perturbed_objective(-1.0, epsilon)) / (2.0 * epsilon)
    torch.testing.assert_close(
        predicted_directional,
        observed_directional,
        rtol=2.0e-8,
        atol=2.0e-9,
    )
    assert result.accounting["node_geometry_recompute_count"] == node_times.numel()
    assert result.accounting["node_geometry_recomputed_once_per_node"]
    assert result.accounting["requested_sample_count_used"] == 0
    assert result.accounting["requested_frame_count_used"] == 0
    assert not result.accounting["frame_by_run_reverse_state_allocated"]
    assert result.accounting["reverse_interaction_scaling"] == "O(J * R)"
    assert result.geometry_vjp_implemented
    assert not result.material_gradients_included
    assert not result.event_time_derivatives_included
    assert not result.chart_endpoint_derivatives_included
    assert not result.node_time_or_rank_derivatives_included
    assert not result.compiler_choice_derivatives_included
    assert "grad_site_density" not in vars(result)
    assert "grad_site_color" not in vars(result)


def test_node_physical_length_bridge_matches_full_transfer_vjp_geometry() -> None:
    sites, rays, node_times, words, density, color, grad_transfer = _one_track_fixture()
    independent_lengths = _direct_fixed_word_physical_lengths(
        sites.positions0,
        sites.velocities,
        sites.weight_coefficients,
        rays.unsqueeze(0),
        node_times,
        words,
        near=0.0,
        far=6.0,
    )[0].detach()
    differentiable_lengths = independent_lengths.clone().requires_grad_(True)
    node_transfers = _direct_node_transfers_from_lengths(
        differentiable_lengths,
        words[0],
        density,
        color,
    )
    torch.sum(node_transfers * grad_transfer[0]).backward()
    assert differentiable_lengths.grad is not None

    bridge = kinetic_p0_node_physical_length_geometry_vjp(
        sites,
        rays,
        node_times,
        words,
        differentiable_lengths.grad,
        near=0.0,
        far=6.0,
        continuous_topology_certificate_id="test-certified-stable-chart",
        node_physical_length_cotangent_provenance_id="same-transfer-vjp-generation",
    )
    full = kinetic_p0_compiler_node_vjp(
        sites,
        rays.unsqueeze(0),
        node_times,
        words,
        density,
        color,
        grad_transfer,
        near=0.0,
        far=6.0,
        continuous_topology_certificate_id="test-certified-stable-chart",
    )
    torch.testing.assert_close(full.node_transfers[0], node_transfers.detach())
    torch.testing.assert_close(bridge.node_physical_lengths, independent_lengths)
    for bridge_gradient, full_gradient in zip(
        (
            bridge.grad_positions0,
            bridge.grad_velocities,
            bridge.grad_weight_coefficients,
            bridge.grad_ray_coefficients,
        ),
        (
            full.grad_positions0,
            full.grad_velocities,
            full.grad_weight_coefficients,
            full.grad_ray_coefficients[0],
        ),
        strict=True,
    ):
        torch.testing.assert_close(bridge_gradient, full_gradient, rtol=4.0e-12, atol=4.0e-12)


def test_node_physical_length_bridge_requires_provenance_and_native_chart_shape() -> None:
    sites, rays, node_times, words, *_ = _one_track_fixture()
    bars = torch.ones((node_times.numel(), words[0].run_count), dtype=DTYPE)
    kwargs = {
        "near": 0.0,
        "far": 6.0,
        "continuous_topology_certificate_id": "test-certified-stable-chart",
        "node_physical_length_cotangent_provenance_id": "native-length-bars-v1",
    }
    with pytest.raises(ValueError, match="node_physical_length_cotangent_provenance_id"):
        kinetic_p0_node_physical_length_geometry_vjp(
            sites,
            rays,
            node_times,
            words,
            bars,
            **{**kwargs, "node_physical_length_cotangent_provenance_id": ""},
        )
    with pytest.raises(ValueError, match=r"shape \[12\]"):
        kinetic_p0_node_physical_length_geometry_vjp(
            sites,
            rays.unsqueeze(0),
            node_times,
            words,
            bars,
            **kwargs,
        )
    with pytest.raises(ValueError, match="grad_node_physical_lengths must have shape"):
        kinetic_p0_node_physical_length_geometry_vjp(
            sites,
            rays,
            node_times,
            words,
            bars[:, :-1],
            **kwargs,
        )

    parameters = inspect.signature(kinetic_p0_node_physical_length_geometry_vjp).parameters
    for forbidden in (
        "sample_times",
        "requested_frame_count",
        "targets",
        "site_density",
        "site_color",
        "grad_node_transfer",
    ):
        assert forbidden not in parameters


def test_node_accumulation_is_additive_and_has_no_requested_frame_axis() -> None:
    (sites, rays, node_times, words, density, color, grad_transfer), combined = _manual_result()
    individual = [
        kinetic_p0_compiler_node_vjp(
            sites,
            rays,
            node_times[node_id : node_id + 1],
            words,
            density,
            color,
            grad_transfer[:, node_id : node_id + 1],
            near=0.0,
            far=6.0,
            continuous_topology_certificate_id="test-certified-stable-chart",
        )
        for node_id in range(node_times.numel())
    ]
    for name in (
        "grad_positions0",
        "grad_velocities",
        "grad_weight_coefficients",
        "grad_ray_coefficients",
        "grad_site_density",
        "grad_site_color",
    ):
        expected = sum((getattr(item, name) for item in individual), torch.zeros_like(getattr(combined, name)))
        torch.testing.assert_close(getattr(combined, name), expected)

    parameters = inspect.signature(kinetic_p0_compiler_node_vjp).parameters
    assert "frame_count" not in parameters
    assert "requested_frame_count" not in parameters
    assert combined.accounting["requested_frame_count_used"] == 0
    assert not combined.accounting["frame_by_run_reverse_state_allocated"]
    assert combined.accounting["reverse_interaction_scaling"] == "O(J * sum_p R_p)"
    assert combined.accounting["active_run_node_interactions"] == 18
    assert combined.accounting["active_cut_node_interactions"] == 12
    assert combined.accounting["owner_margin_evaluations"] == 108
    assert combined.accounting["validation_interaction_scaling"] == "O(J * S * sum_p R_p)"
    assert not combined.event_time_derivatives_included
    assert not combined.chart_endpoint_derivatives_included
    assert not combined.node_time_or_rank_derivatives_included


def test_tiny_optical_depth_matches_stable_expm1_transfer_and_color_vjp() -> None:
    sites = AffineKineticPowerSites(
        positions0=torch.zeros((1, 3), dtype=DTYPE),
        velocities=torch.zeros((1, 3), dtype=DTYPE),
        weight_coefficients=torch.zeros((1, 1), dtype=DTYPE),
    )
    density = torch.tensor([1.0e-18], dtype=DTYPE)
    color = torch.tensor([[0.5, 0.25, 0.75]], dtype=DTYPE)
    result = kinetic_p0_compiler_node_vjp(
        sites,
        torch.tensor([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]], dtype=DTYPE),
        [0.0],
        (make_frozen_kinetic_owner_word((0,)),),
        density,
        color,
        torch.tensor([[[0.0, 1.0, 0.0, 0.0]]], dtype=DTYPE),
        near=0.0,
        far=1.0,
        continuous_topology_certificate_id="tiny-optical-depth-chart",
    )

    alpha = -torch.expm1(-density[0])
    torch.testing.assert_close(result.node_transfers[0, 0, 1:], alpha * color[0])
    torch.testing.assert_close(
        result.grad_site_color[0],
        torch.tensor([alpha, 0.0, 0.0], dtype=DTYPE),
    )
    assert result.node_transfers[0, 0, 1] > 0.0
    assert result.grad_site_color[0, 0] > 0.0


def test_small_cut_denominator_fails_closed() -> None:
    sites = AffineKineticPowerSites(
        positions0=torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=DTYPE),
        velocities=torch.zeros((2, 3), dtype=DTYPE),
        weight_coefficients=torch.zeros((2, 1), dtype=DTYPE),
    )
    ray = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]], dtype=DTYPE)
    with pytest.raises(StableStratumError, match="absolute denominator"):
        kinetic_p0_compiler_node_vjp(
            sites,
            ray,
            [0.0],
            (make_frozen_kinetic_owner_word((0, 1)),),
            torch.ones(2, dtype=DTYPE),
            torch.ones((2, 3), dtype=DTYPE),
            torch.ones((1, 1, 4), dtype=DTYPE),
            near=0.0,
            far=1.0,
            continuous_topology_certificate_id="degenerate-fixture",
        )


def test_all_competitor_check_rejects_a_missing_middle_owner() -> None:
    sites = AffineKineticPowerSites(
        positions0=torch.tensor([[0, 0, 0], [0, 0, 2], [0, 0, 4]], dtype=DTYPE),
        velocities=torch.zeros((3, 3), dtype=DTYPE),
        weight_coefficients=torch.zeros((3, 1), dtype=DTYPE),
    )
    ray = torch.tensor([[0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0]], dtype=DTYPE)
    with pytest.raises(StableStratumError, match="owner/topology margin"):
        kinetic_p0_compiler_node_vjp(
            sites,
            ray,
            [0.0],
            (make_frozen_kinetic_owner_word((0, 2)),),
            torch.ones(3, dtype=DTYPE),
            torch.ones((3, 3), dtype=DTYPE),
            torch.ones((1, 1, 4), dtype=DTYPE),
            near=0.0,
            far=6.0,
            continuous_topology_certificate_id="incorrect-word-fixture",
        )


def test_observed_owner_gap_can_be_promoted_to_a_stricter_trust_gate() -> None:
    (sites, rays, node_times, words, density, color, grad_transfer), accepted = _manual_result()
    assert accepted.margins.minimum_owner_gap > 0.0
    with pytest.raises(StableStratumError, match="owner/topology margin"):
        kinetic_p0_compiler_node_vjp(
            sites,
            rays,
            node_times,
            words,
            density,
            color,
            grad_transfer,
            near=0.0,
            far=6.0,
            continuous_topology_certificate_id="test-certified-stable-chart",
            thresholds=StableStratumThresholds(
                minimum_owner_gap=1.01 * accepted.margins.minimum_owner_gap,
            ),
        )


def test_continuous_topology_certificate_provenance_is_mandatory() -> None:
    sites, rays, node_times, words, density, color, grad_transfer = _stable_fixture()
    with pytest.raises(ValueError, match="continuous_topology_certificate_id"):
        kinetic_p0_compiler_node_vjp(
            sites,
            rays,
            node_times,
            words,
            density,
            color,
            grad_transfer,
            near=0.0,
            far=6.0,
            continuous_topology_certificate_id="",
        )
