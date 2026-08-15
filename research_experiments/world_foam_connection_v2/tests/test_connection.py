from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from world_foam_connection_v2.affine import (
    AffineGenerator,
    compose,
    generator_exponential,
)
from world_foam_connection_v2.connection import (
    EndpointTransportHistory,
    P0FlowSamples,
    P0Ray,
    P0RayRate,
    evaluate_connection,
    scan_endpoint_transports,
)
from world_foam_connection_v2.fixtures import FIXTURE_BUILDERS, build_fixture


@pytest.mark.parametrize("name", tuple(FIXTURE_BUILDERS))
def test_all_stable_p0_fixtures_close_the_connection_identity(name: str) -> None:
    fixture = build_fixture(name, 0.0)

    result = evaluate_connection(
        fixture.ray,
        fixture.rate,
        fixture.flow,
        fixture.endpoint_history,
        quadrature_order=fixture.quadrature_order,
        flow_declaration=fixture.flow_declaration,
    )

    torch.testing.assert_close(
        result.core.direct_time_derivative.as_vector(),
        result.core.predicted_time_derivative.as_vector(),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    torch.testing.assert_close(
        result.core.reconstructed_transfer.as_vector(),
        result.core.ordered.total.as_vector(),
        rtol=1.0e-11,
        atol=1.0e-11,
    )
    assert result.direct_physical_cone.passed
    assert result.reconstructed_physical_cone.passed


def test_pinned_moving_boundary_is_nonzero_singular_curvature() -> None:
    fixture = build_fixture("moving_noncommuting_boundary", 0.0)
    result = evaluate_connection(
        fixture.ray,
        fixture.rate,
        fixture.flow,
        fixture.endpoint_history,
        quadrature_order=fixture.quadrature_order,
    )

    torch.testing.assert_close(
        result.core.curvature.singular_total.as_vector(),
        result.core.direct_time_derivative.as_vector(),
        rtol=1.0e-11,
        atol=1.0e-11,
    )
    assert torch.linalg.vector_norm(
        result.core.curvature.singular_total.moment
    ) > 0.1


def test_discontinuous_flow_fails_continuity_policy_but_not_theorem() -> None:
    fixture = build_fixture("discontinuous_flow", 0.0)
    result = evaluate_connection(
        fixture.ray,
        fixture.rate,
        fixture.flow,
        fixture.endpoint_history,
        quadrature_order=fixture.quadrature_order,
        flow_declaration=fixture.flow_declaration,
    )

    assert result.flow_admissibility is not None
    assert not result.flow_admissibility.continuity_passed
    assert not result.flow_admissibility.passed
    assert result.flow_admissibility.probe_grid_only
    assert not result.flow_admissibility.continuous_bound_certified
    torch.testing.assert_close(
        result.core.theorem_residual.as_vector(),
        torch.zeros(4, dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-10,
    )


def test_flow_capacity_charges_retained_bytes_when_budget_is_declared() -> None:
    fixture = build_fixture("front_red_back_blue", 0.0)
    declaration = replace(
        fixture.flow_declaration,
        source_motion_retained_bytes=4,
    )

    result = evaluate_connection(
        fixture.ray,
        fixture.rate,
        fixture.flow,
        fixture.endpoint_history,
        quadrature_order=fixture.quadrature_order,
        flow_declaration=declaration,
    )

    report = result.flow_admissibility
    assert report is not None
    assert report.declared_retained_bytes == 8
    assert report.source_motion_retained_bytes == 4
    assert report.retained_byte_capacity_ratio == 2.0
    assert not report.capacity_passed
    assert not report.passed


def test_noncommuting_multistep_endpoint_history_preserves_time_order() -> None:
    reference = torch.tensor(0.0, dtype=torch.float64)
    ray = P0Ray(
        cuts=reference.new_tensor((0.0, 1.0)),
        extinction=reference.new_tensor((1.0,)),
        emission_density=reference.new_tensor(((0.5, 0.2, 0.1),)),
    )
    history = EndpointTransportHistory(
        durations=reference.new_tensor((0.2, 0.3)),
        near_scalar=reference.new_tensor((-1.0, -0.4)),
        near_source=reference.new_tensor(((1.0, 0.0, 0.0), (0.0, 0.4, 0.0))),
        far_scalar=reference.new_zeros((2,)),
        far_source=reference.new_zeros((2, 3)),
    )
    first = generator_exponential(
        AffineGenerator(history.near_scalar[0], history.near_source[0]),
        history.durations[0],
    )
    second = generator_exponential(
        AffineGenerator(history.near_scalar[1], history.near_source[1]),
        history.durations[1],
    )

    result = scan_endpoint_transports(ray, history)

    torch.testing.assert_close(
        result.near.as_vector(),
        compose(first, second).as_vector(),
    )
    assert not torch.allclose(
        result.near.as_vector(),
        compose(second, first).as_vector(),
    )


@pytest.mark.parametrize(
    ("cut_velocity", "expected_near_nonzero", "expected_far_nonzero"),
    (((0.25, 0.0), True, False), ((0.0, -0.3), False, True)),
)
def test_near_only_and_far_only_endpoint_motion(
    cut_velocity: tuple[float, float],
    expected_near_nonzero: bool,
    expected_far_nonzero: bool,
) -> None:
    fixture = build_fixture("material_evolution", 0.0)
    rate = P0RayRate(
        cut_velocity=fixture.ray.cuts.new_tensor(cut_velocity),
        extinction_time=torch.zeros_like(fixture.rate.extinction_time),
        emission_density_time=torch.zeros_like(
            fixture.rate.emission_density_time
        ),
    )
    result = evaluate_connection(
        fixture.ray,
        rate,
        fixture.flow,
        fixture.endpoint_history,
        quadrature_order=fixture.quadrature_order,
    )

    assert bool(
        torch.linalg.vector_norm(
            result.core.endpoint_kinematics.near_generator.as_vector()
        )
        > 0.0
    ) is expected_near_nonzero
    assert bool(
        torch.linalg.vector_norm(
            result.core.endpoint_kinematics.far_generator.as_vector()
        )
        > 0.0
    ) is expected_far_nonzero
    torch.testing.assert_close(
        result.core.direct_time_derivative.as_vector(),
        result.core.endpoint_kinematics.flux.as_vector(),
        rtol=1.0e-11,
        atol=1.0e-11,
    )


def test_vacuum_run_is_identity_and_has_finite_zero_derivative() -> None:
    reference = torch.tensor(0.0, dtype=torch.float64)
    ray = P0Ray(
        cuts=reference.new_tensor((0.0, 2.0)),
        extinction=reference.new_zeros((1,)),
        emission_density=reference.new_zeros((1, 3)),
    )
    rate = P0RayRate(
        cut_velocity=reference.new_tensor((0.1, -0.2)),
        extinction_time=reference.new_zeros((1,)),
        emission_density_time=reference.new_zeros((1, 3)),
    )
    flow = P0FlowSamples(
        bulk_value=reference.new_zeros((1, 4)),
        bulk_d_dz=reference.new_zeros((1, 4)),
        cell_left_value=reference.new_zeros((1,)),
        cell_right_value=reference.new_zeros((1,)),
    )
    history = EndpointTransportHistory(
        durations=reference.new_empty((0,)),
        near_scalar=reference.new_empty((0,)),
        near_source=reference.new_empty((0, 3)),
        far_scalar=reference.new_empty((0,)),
        far_source=reference.new_empty((0, 3)),
    )

    result = evaluate_connection(
        ray,
        rate,
        flow,
        history,
        quadrature_order=4,
    )

    torch.testing.assert_close(
        result.core.ordered.total.as_vector(),
        reference.new_tensor((1.0, 0.0, 0.0, 0.0)),
    )
    torch.testing.assert_close(
        result.core.direct_time_derivative.as_vector(),
        reference.new_zeros((4,)),
    )
