from __future__ import annotations

import math

import pytest
import torch
from compact_lie_schedule import (
    LINEAR_SAMPLE_WEIGHT_EVALUATION,
    certify_fit_derived_barycentric_weights,
    dense_sample_to_node_weights,
    fit_derived_sample_to_node_weights,
)
from transfer_lie_chart import DTYPE, chebyshev_basis, chebyshev_nodes


def _schedule(rank: int, t_min: float, t_max: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nodes = chebyshev_nodes(rank, t_min=t_min, t_max=t_max)
    fit = torch.linalg.inv(chebyshev_basis(nodes, t_min=t_min, t_max=t_max, rank=rank))
    barycentric = certify_fit_derived_barycentric_weights(
        nodes,
        fit,
        t_min=t_min,
        t_max=t_max,
    )
    return nodes, fit, barycentric


@pytest.mark.parametrize("rank", (2, 3, 4, 7, 8, 16, 32))
@pytest.mark.parametrize(
    ("t_min", "t_max"),
    ((-1.0, 1.0), (0.123, 0.456), (1.0e9, 1.0e9 + 1.0), (1.0e12, 1.0e12 + 1.0)),
)
def test_fit_derived_weights_match_dense_forward_and_node_cotangent(
    rank: int,
    t_min: float,
    t_max: float,
) -> None:
    nodes, fit, barycentric = _schedule(rank, t_min, t_max)
    interior = torch.linspace(t_min, t_max, 11, dtype=DTYPE)
    samples = torch.cat((torch.tensor([t_min, t_max], dtype=DTYPE), interior, nodes))
    result = fit_derived_sample_to_node_weights(
        samples,
        t_min=t_min,
        t_max=t_max,
        node_times=nodes,
        fit_matrix=fit,
        barycentric_weights=barycentric,
    )
    dense = dense_sample_to_node_weights(
        samples,
        t_min=t_min,
        t_max=t_max,
        fit_matrix=fit,
    )
    tolerance = 128.0 * torch.finfo(DTYPE).eps * rank
    torch.testing.assert_close(result.weights, dense, atol=tolerance, rtol=tolerance)

    exact_rows = result.weights[-rank:]
    torch.testing.assert_close(exact_rows, torch.eye(rank, dtype=DTYPE), atol=0.0, rtol=0.0)
    assert result.exact_node_row_count >= rank
    assert result.linear_weight_interactions == samples.numel() * rank

    generator = torch.Generator().manual_seed(17 + rank)
    node_chart = torch.randn((rank, 4), generator=generator, dtype=DTYPE)
    sample_bar = torch.randn((samples.numel(), 4), generator=generator, dtype=DTYPE)
    torch.testing.assert_close(
        result.weights @ node_chart,
        dense @ node_chart,
        atol=tolerance * samples.numel(),
        rtol=tolerance * samples.numel(),
    )
    expected_node_bar = dense.T @ sample_bar
    torch.testing.assert_close(
        result.weights.T @ sample_bar,
        expected_node_bar,
        atol=tolerance * samples.numel(),
        rtol=tolerance * samples.numel(),
    )
    differentiable_nodes = node_chart.clone().requires_grad_(True)
    ((result.weights @ differentiable_nodes) * sample_bar).sum().backward()
    torch.testing.assert_close(
        differentiable_nodes.grad,
        expected_node_bar,
        atol=tolerance * samples.numel(),
        rtol=tolerance * samples.numel(),
    )


def test_exact_and_near_node_rows_are_one_hot_or_explicit_dense_fallback() -> None:
    rank = 8
    nodes, fit, barycentric = _schedule(rank, -1.0, 1.0)
    near_left = torch.nextafter(nodes[3], torch.tensor(float("-inf"), dtype=DTYPE))
    near_right = torch.nextafter(nodes[3], torch.tensor(float("inf"), dtype=DTYPE))
    samples = torch.stack((nodes[3], near_left, near_right, torch.tensor(-1.0), torch.tensor(1.0)))
    result = fit_derived_sample_to_node_weights(
        samples,
        t_min=-1.0,
        t_max=1.0,
        node_times=nodes,
        fit_matrix=fit,
        barycentric_weights=barycentric,
    )
    dense = dense_sample_to_node_weights(samples, t_min=-1.0, t_max=1.0, fit_matrix=fit)

    assert result.evaluation == "verified_fit_derived_second_form_barycentric_with_dense_fallback"
    assert result.exact_node_row_count == 1
    assert result.dense_fallback_row_count == 2
    assert result.dense_fallback_interactions == 2 * rank * rank
    torch.testing.assert_close(result.weights[0], torch.eye(rank, dtype=DTYPE)[3], atol=0.0, rtol=0.0)
    torch.testing.assert_close(result.weights[1:], dense[1:], atol=2.0e-14, rtol=2.0e-14)


def test_fit_row_weights_preserve_large_offset_nodes_where_analytic_root_weights_do_not() -> None:
    rank = 32
    t_min, t_max = 1.0e12, 1.0e12 + 1.0
    nodes, fit, barycentric = _schedule(rank, t_min, t_max)
    samples = torch.linspace(t_min, t_max, 129, dtype=DTYPE)
    dense = dense_sample_to_node_weights(samples, t_min=t_min, t_max=t_max, fit_matrix=fit)
    actual = fit_derived_sample_to_node_weights(
        samples,
        t_min=t_min,
        t_max=t_max,
        node_times=nodes,
        fit_matrix=fit,
        barycentric_weights=barycentric,
    )
    torch.testing.assert_close(actual.weights, dense, atol=1.0e-13, rtol=1.0e-13)
    assert actual.evaluation == LINEAR_SAMPLE_WEIGHT_EVALUATION

    index = torch.arange(rank, dtype=DTYPE)
    theta = math.pi * (2.0 * index + 1.0) / (2.0 * rank)
    analytic = ((-1.0) ** index) * torch.sin(theta)
    analytic /= analytic.abs().max()
    x = (2.0 * samples - (t_max + t_min)) / (t_max - t_min)
    x_nodes = (2.0 * nodes - (t_max + t_min)) / (t_max - t_min)
    q = analytic[None, :] / (x[:, None] - x_nodes[None, :])
    analytic_weights = q / q.sum(dim=1, keepdim=True)
    finite_rows = torch.isfinite(analytic_weights).all(dim=1)
    assert float((analytic_weights[finite_rows] - dense[finite_rows]).abs().max().item()) > 1.0e-3


def test_invalid_fit_barycentric_provenance_and_interval_fail_closed() -> None:
    nodes, fit, barycentric = _schedule(4, -1.0, 1.0)
    bad_fit = fit.clone()
    bad_fit[0, 0] += 0.1
    with pytest.raises(ValueError, match="verified inverse"):
        certify_fit_derived_barycentric_weights(nodes, bad_fit, t_min=-1.0, t_max=1.0)
    with pytest.raises(ValueError, match="certified fit matrix"):
        fit_derived_sample_to_node_weights(
            torch.tensor([0.0], dtype=DTYPE),
            t_min=-1.0,
            t_max=1.0,
            node_times=nodes,
            fit_matrix=fit,
            barycentric_weights=barycentric.roll(1),
        )
    with pytest.raises(ValueError, match="leave the interpolation chart"):
        fit_derived_sample_to_node_weights(
            torch.tensor([1.01], dtype=DTYPE),
            t_min=-1.0,
            t_max=1.0,
            node_times=nodes,
            fit_matrix=fit,
            barycentric_weights=barycentric,
        )
