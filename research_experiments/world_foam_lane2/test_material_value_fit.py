from __future__ import annotations

import torch

from research_experiments.world_foam_lane2.fit_finite_element_materials import (
    TARGET_COLOR,
    TARGET_CONTROLS,
    _positive_p2_integral,
    make_targets,
    run_gate,
)


def test_independent_target_integral_matches_dense_quadrature() -> None:
    x = torch.linspace(0.0, 1.0, 200_001, dtype=torch.float64)
    sigma = (
        TARGET_CONTROLS[0] * (1.0 - x).square()
        + 2.0 * TARGET_CONTROLS[1] * x * (1.0 - x)
        + TARGET_CONTROLS[2] * x.square()
    )
    for target in make_targets():
        mask = (x >= target.start) & (x <= target.stop)
        x_local = x[mask]
        sigma_local = sigma[mask]
        dense_tau = torch.trapezoid(sigma_local, x_local)
        exact_tau = _positive_p2_integral(
            TARGET_CONTROLS, target.start, target.stop
        )
        # Uniform samples do not always land on both arbitrary endpoints, so
        # this is a quadrature sanity check rather than the exact oracle.
        assert torch.allclose(dense_tau, exact_tau, rtol=0.0, atol=6.0e-6)
        assert torch.allclose(
            target.m,
            (1.0 - target.beta) * TARGET_COLOR,
            rtol=0.0,
            atol=1.0e-14,
        )


def test_material_value_gate_passes() -> None:
    report = run_gate(steps=450, seed=17)
    assert report["status"] == "pass"
    assert all(report["checks"].values())
