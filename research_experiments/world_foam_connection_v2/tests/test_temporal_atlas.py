from __future__ import annotations

import pytest
import torch

from world_foam_connection_v2.temporal_atlas import (
    AtlasKind,
    LinearTemporalAtlas,
    compile_probe_certified_linear_atlas,
    physical_chart_to_transfer,
    transfer_to_physical_chart,
    transfer_to_unrestricted_log_chart,
    unrestricted_log_chart_to_transfer,
)


def test_unrestricted_chart_roundtrips_attenuating_amplifying_and_identity() -> None:
    transfer = torch.tensor(
        [
            [0.4, 0.2, -0.1, 0.3],
            [1.0, 0.0, 0.0, 0.0],
            [1.7, -0.4, 0.2, 1.1],
        ],
        dtype=torch.float64,
    )

    reconstructed = unrestricted_log_chart_to_transfer(
        transfer_to_unrestricted_log_chart(transfer)
    )

    torch.testing.assert_close(reconstructed, transfer, rtol=1.0e-12, atol=1.0e-12)


def test_identity_chart_has_correct_first_derivative() -> None:
    kappa = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    chart = torch.stack(
        (
            kappa,
            torch.tensor(0.2, dtype=torch.float64),
            torch.tensor(-0.3, dtype=torch.float64),
            torch.tensor(0.4, dtype=torch.float64),
        )
    )

    transfer = unrestricted_log_chart_to_transfer(chart)
    transfer[1:].sum().backward()

    # d[(1-exp(-k))/k]/dk at k=0 is -1/2.
    torch.testing.assert_close(
        kappa.grad,
        torch.tensor(-0.5 * (0.2 - 0.3 + 0.4), dtype=torch.float64),
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_physical_chart_rejects_nonphysical_group_transfer() -> None:
    amplifying = torch.tensor([1.2, 0.0, 0.0, 0.0], dtype=torch.float64)
    signed_moment = torch.tensor([0.8, -0.1, 0.0, 0.0], dtype=torch.float64)

    with pytest.raises(ValueError, match="beta<=1"):
        transfer_to_physical_chart(amplifying)
    with pytest.raises(ValueError, match="nonnegative moment"):
        transfer_to_physical_chart(signed_moment)


def test_physical_chart_roundtrip_preserves_cone() -> None:
    transfer = torch.tensor(
        [[0.7, 0.1, 0.2, 0.05], [1.0, 0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )

    chart = transfer_to_physical_chart(transfer)
    reconstructed = physical_chart_to_transfer(chart)

    torch.testing.assert_close(reconstructed, transfer, rtol=1.0e-12, atol=1.0e-12)


def test_linear_atlas_evaluation_derivative_and_integral_are_exact() -> None:
    knots = torch.tensor([0.0, 1.0, 3.0], dtype=torch.float64)
    slope = torch.tensor([2.0, -1.0, 0.5, 3.0], dtype=torch.float64)
    offset = torch.tensor([0.2, 0.1, -0.4, 1.0], dtype=torch.float64)
    values = offset + knots[:, None] * slope
    atlas = LinearTemporalAtlas(AtlasKind.SIGNED_K_F, knots, values)
    query = torch.tensor([0.0, 0.5, 2.0, 3.0], dtype=torch.float64)

    torch.testing.assert_close(
        atlas.evaluate(query),
        offset + query[:, None] * slope,
    )
    torch.testing.assert_close(
        atlas.derivative(query),
        slope.expand(query.numel(), -1),
    )
    torch.testing.assert_close(
        atlas.integral_from_start(query),
        offset * query[:, None] + 0.5 * slope * query[:, None].square(),
    )


def test_probe_compiler_uses_two_nodes_for_linear_primal_and_tangent() -> None:
    times = torch.linspace(0.0, 1.0, 33, dtype=torch.float64)
    slope = torch.tensor([0.2, -0.1, 0.4, 0.3], dtype=torch.float64)
    offset = torch.tensor([1.0, 0.0, 0.2, -0.3], dtype=torch.float64)

    atlas, certificate = compile_probe_certified_linear_atlas(
        kind=AtlasKind.GROUP_U_TILDE,
        probe_times=times,
        value_evaluator=lambda value: offset + value[:, None] * slope,
        primal_tolerance=1.0e-12,
        tangent_evaluator=lambda value: slope.expand(value.numel(), -1),
        tangent_tolerance=1.0e-12,
    )

    assert atlas.node_count == 2
    assert certificate.verified
    assert certificate.probe_grid_only


def test_probe_compiler_fails_closed_at_node_budget() -> None:
    times = torch.linspace(0.0, 1.0, 65, dtype=torch.float64)

    _, certificate = compile_probe_certified_linear_atlas(
        kind=AtlasKind.SIGNED_K_F,
        probe_times=times,
        value_evaluator=lambda value: torch.stack(
            (
                torch.sin(19.0 * value),
                torch.cos(17.0 * value),
                torch.sin(13.0 * value),
                torch.cos(11.0 * value),
            ),
            dim=-1,
        ),
        primal_tolerance=1.0e-8,
        maximum_nodes=3,
    )

    assert certificate.node_count == 3
    assert not certificate.verified
