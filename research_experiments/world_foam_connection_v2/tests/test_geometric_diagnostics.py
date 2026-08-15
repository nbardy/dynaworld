from __future__ import annotations

import torch

from world_foam_connection_v2.affine import AffineGenerator
from world_foam_connection_v2.connection import diagnose_sensor_depth_lift
from world_foam_connection_v2.fixtures import (
    cosine_depth_curvature_cancellation,
    sideways_pinhole_lift,
)
from world_foam_connection_v2.holonomy import positive_rectangle_holonomy
from world_foam_connection_v2.oracle import (
    evaluate_corrected_derivative_oracle,
    evaluate_endpoint_history_direction_oracle,
)


def test_nonzero_cosine_curvature_cancels_only_after_transport_and_depth() -> None:
    fixture = cosine_depth_curvature_cancellation(sample_count=32)

    assert torch.amax(torch.abs(fixture.curvature.source)) > 0.01
    torch.testing.assert_close(
        fixture.transported_curvature_integral.as_vector(),
        torch.zeros(4, dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-11,
    )


def test_sideways_pinhole_motion_needs_sensor_plane_lift() -> None:
    inputs = sideways_pinhole_lift()

    report = diagnose_sensor_depth_lift(inputs)

    assert torch.all(report.full_lift_passed)
    assert torch.all(~report.supplied_scalar_flow_exact)
    torch.testing.assert_close(
        report.full_lift[:, 0],
        torch.reciprocal(inputs.gamma_u[:, 0]),
    )
    torch.testing.assert_close(
        report.full_lift[:, 1:],
        torch.zeros_like(report.full_lift[:, 1:]),
    )


def test_constant_noncommuting_rectangle_has_declared_curvature_sign() -> None:
    reference = torch.tensor(0.0, dtype=torch.float64)
    report = positive_rectangle_holonomy(
        depth_generator=AffineGenerator(
            reference.new_tensor(-1.0),
            reference.new_tensor((1.0, 0.0, 0.0)),
        ),
        time_generator=AffineGenerator(
            reference.new_tensor(-2.0),
            reference.new_tensor((0.0, 0.0, 2.0)),
        ),
        depth_extent=reference.new_tensor(1.0e-5),
        time_extent=reference.new_tensor(1.0e-5),
    )

    assert torch.linalg.vector_norm(report.predicted_curvature.source) > 0.1
    torch.testing.assert_close(
        report.area_scaled_holonomy.as_vector(),
        report.predicted_curvature.as_vector(),
        rtol=1.0e-4,
        atol=1.0e-4,
    )


def test_recomputed_u_tilde_derivative_catches_endpoint_scan_order() -> None:
    report = evaluate_corrected_derivative_oracle()

    assert report.passed
    assert report.maximum_absolute_error <= 1.0e-6


def test_endpoint_history_selected_direction_matches_finite_difference() -> None:
    report = evaluate_endpoint_history_direction_oracle()

    assert report.checked_observable_count == 16
    assert report.passed
