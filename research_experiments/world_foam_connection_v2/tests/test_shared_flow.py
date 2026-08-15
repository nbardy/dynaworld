from __future__ import annotations

import torch

from world_foam_connection_v2.shared_flow import (
    FlowDomain,
    SharedChebyshevFlow,
)


def _flow(*, temporal_degree: int = 1, depth_degree: int = 1):
    return SharedChebyshevFlow(
        domain=FlowDomain(t_min=0.0, t_max=2.0, z_min=1.0, z_max=5.0),
        temporal_degree=temporal_degree,
        depth_degree=depth_degree,
        maximum_speed=3.0,
        dtype=torch.float64,
    )


def test_zero_shared_flow_has_zero_value_and_derivatives() -> None:
    flow = _flow()
    t = torch.tensor([0.0, 0.5, 2.0], dtype=torch.float64)
    z = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float64)

    evaluation = flow.evaluate(t, z)

    torch.testing.assert_close(evaluation.value, torch.zeros_like(t))
    torch.testing.assert_close(evaluation.d_dt, torch.zeros_like(t))
    torch.testing.assert_close(evaluation.d_dz, torch.zeros_like(t))


def test_depth_chebyshev_coefficient_has_analytic_midpoint_slope() -> None:
    flow = _flow(temporal_degree=0, depth_degree=1)
    with torch.no_grad():
        flow.coefficients[0, 1] = 0.25
    t = torch.tensor(1.0, dtype=torch.float64)
    z = torch.tensor(3.0, dtype=torch.float64)

    evaluation = flow.evaluate(t, z)

    # At normalized z=0, tanh(series)=0 and dT_1/dz=2/(5-1).
    torch.testing.assert_close(
        evaluation.d_dz,
        torch.tensor(3.0 * 0.25 * 0.5, dtype=torch.float64),
    )
    torch.testing.assert_close(evaluation.value, torch.zeros_like(t))


def test_capacity_receipt_has_no_frame_or_ray_indexed_state() -> None:
    flow = _flow(temporal_degree=2, depth_degree=3)

    report = flow.capacity_report(reference_temporal_dof=12)

    assert report.coefficient_count == 12
    assert report.capacity_ratio == 1.0
    assert report.within_reference_capacity
    assert report.requested_frame_indexed_state_count == 0
    assert report.ray_indexed_state_count == 0


def test_flow_gradients_reach_every_used_coefficient() -> None:
    flow = _flow(temporal_degree=2, depth_degree=2)
    t = torch.tensor([0.2, 0.7, 1.4], dtype=torch.float64)
    z = torch.tensor([1.3, 2.8, 4.6], dtype=torch.float64)

    loss = flow(t, z).square().sum() + flow.evaluate(t, z).d_dz.square().sum()
    loss.backward()

    assert flow.coefficients.grad is not None
    assert torch.isfinite(flow.coefficients.grad).all()
    assert torch.count_nonzero(flow.coefficients.grad) > 0


def test_nonpositive_local_orientation_margin_is_observable() -> None:
    flow = _flow(temporal_degree=0, depth_degree=1)
    with torch.no_grad():
        flow.coefficients[0, 1] = -4.0
    t = torch.tensor([1.0], dtype=torch.float64)
    z = torch.tensor([3.0], dtype=torch.float64)

    margin = flow.minimum_euler_orientation_margin(t, z, delta_t=1.0)

    assert margin < 0.0
