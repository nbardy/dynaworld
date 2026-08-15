from __future__ import annotations

import pytest
import torch

from world_foam_connection_v2.fitting import (
    AtlasFitConfig,
    TrainableConnectionAtlas,
    fit_connection_atlas,
)
from world_foam_connection_v2.oracle import build_flat_translation_probe_series
from world_foam_connection_v2.representation_benchmark import (
    compile_equal_family_representation,
)
from world_foam_connection_v2.temporal_atlas import AtlasKind


def _compile(
    kind: AtlasKind,
    *,
    variant: str,
    flow_bytes: int,
):
    series, run_count, _ = build_flat_translation_probe_series(probe_count=33)
    return compile_equal_family_representation(
        series,
        kind=kind,
        variant=variant,
        primal_tolerance=1.0e-8,
        secant_tolerance=1.0e-7,
        run_count=run_count,
        shared_flow_payload_bytes=flow_bytes,
        maximum_nodes=33,
    )


def test_u_tilde_and_k_f_reconstruct_the_same_physical_u() -> None:
    group = _compile(
        AtlasKind.GROUP_U_TILDE,
        variant="A1_group_U_tilde",
        flow_bytes=8,
    )
    curvature = _compile(
        AtlasKind.SIGNED_K_F,
        variant="A2_signed_K_F",
        flow_bytes=8,
    )

    assert group.certificate.probe_primal_secant_verified
    assert curvature.certificate.probe_primal_secant_verified
    assert group.certificate.node_count == 2
    assert curvature.certificate.node_count == 2
    assert not group.certificate.canonical_primal_tangent_verified
    assert not curvature.certificate.promotion_eligible


def test_direct_a0_and_capacity_matched_a0c_are_reported_separately() -> None:
    direct = _compile(
        AtlasKind.PHYSICAL_U,
        variant="A0_direct_U",
        flow_bytes=0,
    )
    matched = _compile(
        AtlasKind.PHYSICAL_U,
        variant="A0c_direct_U_capacity_matched_flow",
        flow_bytes=64,
    )

    assert direct.certificate.variant != matched.certificate.variant
    assert (
        matched.certificate.total_retained_bytes
        - direct.certificate.total_retained_bytes
        == 64
    )
    assert matched.certificate.compile_flow_run_evaluations > 0
    assert direct.certificate.compile_flow_run_evaluations == 0


def test_k_f_additive_fit_fails_closed_at_group_boundary() -> None:
    knots = torch.tensor((0.0, 1.0), dtype=torch.float64)
    tangent = torch.tensor(
        ((-2.0, 0.0, 0.0, 0.0), (-2.0, 0.0, 0.0, 0.0)),
        dtype=torch.float64,
    )
    model = TrainableConnectionAtlas(
        kind=AtlasKind.SIGNED_K_F,
        knots=knots,
        initial_values=tangent,
        base_group_transfer=torch.tensor(
            (1.0, 0.0, 0.0, 0.0),
            dtype=torch.float64,
        ),
    )
    identity_endpoints = torch.tensor(
        ((1.0, 0.0, 0.0, 0.0),),
        dtype=torch.float64,
    )

    with pytest.raises(ValueError, match="crossed beta"):
        model.physical_transfer(
            torch.tensor((1.0,), dtype=torch.float64),
            near_endpoint_transport=identity_endpoints,
            far_endpoint_transport=identity_endpoints,
        )


def test_nonphysical_u_tilde_reconstructs_a_physical_u() -> None:
    knots = torch.tensor((0.0, 1.0), dtype=torch.float64)
    corrected = torch.tensor(
        ((2.0, 0.4, 0.2, 0.6), (2.0, 0.4, 0.2, 0.6)),
        dtype=torch.float64,
    )
    model = TrainableConnectionAtlas(
        kind=AtlasKind.GROUP_U_TILDE,
        knots=knots,
        initial_values=corrected,
    )
    near = torch.tensor(((2.0, 0.0, 0.0, 0.0),), dtype=torch.float64)
    far = torch.tensor(((0.5, 0.0, 0.0, 0.0),), dtype=torch.float64)

    reconstructed = model.physical_transfer(
        torch.tensor((0.5,), dtype=torch.float64),
        near_endpoint_transport=near,
        far_endpoint_transport=far,
    )

    torch.testing.assert_close(
        reconstructed,
        torch.tensor(((0.5, 0.2, 0.1, 0.3),), dtype=torch.float64),
    )
    assert corrected[0, 0] > 1.0


def test_direct_u_fit_performs_a_finite_decreasing_optimizer_step_series() -> None:
    knots = torch.tensor((0.0, 1.0), dtype=torch.float64)
    query_times = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    initial = torch.tensor(
        ((0.90, 0.02, 0.02, 0.02), (0.90, 0.02, 0.02, 0.02)),
        dtype=torch.float64,
    )
    target = torch.stack(
        (
            0.82 - 0.07 * query_times,
            0.05 + 0.02 * query_times,
            0.08 + 0.03 * query_times,
            0.04 + 0.01 * query_times,
        ),
        dim=-1,
    )
    model = TrainableConnectionAtlas(
        kind=AtlasKind.PHYSICAL_U,
        knots=knots,
        initial_values=initial,
    )

    report = fit_connection_atlas(
        model,
        query_times=query_times,
        target_physical_transfer=target,
        config=AtlasFitConfig(steps=40, learning_rate=3.0e-2),
    )

    assert report.finite
    assert report.loss_decreased
    assert report.final_loss < report.initial_loss


def test_signed_k_f_fit_reconstructs_before_scoring_and_decreases_loss() -> None:
    knots = torch.tensor((0.0, 1.0), dtype=torch.float64)
    query_times = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    base = torch.tensor((0.85, 0.04, 0.06, 0.03), dtype=torch.float64)
    target_rate = torch.tensor((-0.08, 0.03, 0.02, 0.01), dtype=torch.float64)
    target = base[None, :] + query_times[:, None] * target_rate[None, :]
    identity_endpoints = torch.tensor(
        (1.0, 0.0, 0.0, 0.0),
        dtype=torch.float64,
    ).expand(query_times.numel(), -1)
    model = TrainableConnectionAtlas(
        kind=AtlasKind.SIGNED_K_F,
        knots=knots,
        initial_values=torch.zeros((2, 4), dtype=torch.float64),
        base_group_transfer=base,
    )

    report = fit_connection_atlas(
        model,
        query_times=query_times,
        target_physical_transfer=target,
        config=AtlasFitConfig(steps=40, learning_rate=3.0e-2),
        near_endpoint_transport=identity_endpoints,
        far_endpoint_transport=identity_endpoints,
    )

    assert report.finite
    assert report.loss_decreased
    assert report.final_loss < report.initial_loss
