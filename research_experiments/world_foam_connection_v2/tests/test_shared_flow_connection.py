from __future__ import annotations

import pytest
import torch

from world_foam_connection_v2.oracle import evaluate_shared_flow_direction_oracle
from world_foam_connection_v2.shared_flow import FlowDomain, SharedChebyshevFlow


def test_end_to_end_cut_time_and_flow_coefficient_jvp_matches_finite_difference() -> None:
    report = evaluate_shared_flow_direction_oracle()

    assert report.covers_cut_resampling
    assert report.covers_flow_coefficients
    assert report.covers_time_coordinate
    assert report.checked_observable_count == 9
    assert report.passed


def test_shared_flow_rejects_queries_outside_declared_domain() -> None:
    flow = SharedChebyshevFlow(
        domain=FlowDomain(0.0, 1.0, 0.5, 3.0),
        temporal_degree=1,
        depth_degree=1,
        maximum_speed=1.0,
        dtype=torch.float64,
    )

    with pytest.raises(ValueError, match="outside"):
        flow(
            torch.tensor(0.5, dtype=torch.float64),
            torch.tensor(3.1, dtype=torch.float64),
        )
