from __future__ import annotations

import torch

from research_experiments.softmax_gs.reference import (
    softmax_gs_bounded_contribution_tape,
    softmax_gs_composite,
    softmax_gs_contribution_tape,
    vanilla_alpha_over,
    vanilla_alpha_over_weights,
)


def test_softmax_gs_disabled_matches_vanilla_alpha_over() -> None:
    absorbance = torch.tensor([0.2, 0.35, 0.15], dtype=torch.float64)
    exponent = torch.tensor([-0.2, -0.7, -0.1], dtype=torch.float64)
    depth = torch.tensor([1.0, 1.1, 2.0], dtype=torch.float64)
    features = torch.tensor(
        [[1.0, 0.0, 0.2], [0.0, 1.0, 0.4], [0.5, 0.3, 1.0]],
        dtype=torch.float64,
    )

    actual_color, actual_alpha = softmax_gs_composite(
        absorbance,
        exponent,
        depth,
        features,
        beta=8.0,
        gamma=5.0,
        enabled=False,
    )
    expected_color, expected_alpha = vanilla_alpha_over(absorbance, features)

    torch.testing.assert_close(actual_color, expected_color)
    torch.testing.assert_close(actual_alpha, expected_alpha)


def test_softmax_gs_two_same_depth_splats_are_order_invariant() -> None:
    absorbance = torch.tensor([0.6, 0.35], dtype=torch.float64)
    exponent = torch.tensor([-0.05, -1.5], dtype=torch.float64)
    depth = torch.tensor([2.0, 2.0], dtype=torch.float64)
    features = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.25, 1.0]], dtype=torch.float64)

    color_a, alpha_a = softmax_gs_composite(absorbance, exponent, depth, features, beta=6.0, gamma=8.0)
    order = torch.tensor([1, 0])
    color_b, alpha_b = softmax_gs_composite(
        absorbance[order],
        exponent[order],
        depth[order],
        features[order],
        beta=6.0,
        gamma=8.0,
    )

    torch.testing.assert_close(color_a, color_b)
    torch.testing.assert_close(alpha_a, alpha_b)


def test_softmax_gs_separated_depth_splats_approach_vanilla() -> None:
    absorbance = torch.tensor([0.55, 0.45], dtype=torch.float64)
    exponent = torch.tensor([-0.02, -0.08], dtype=torch.float64)
    depth = torch.tensor([1.0, 10.0], dtype=torch.float64)
    features = torch.tensor([[0.9, 0.1, 0.0], [0.0, 0.2, 1.0]], dtype=torch.float64)

    actual_color, actual_alpha = softmax_gs_composite(absorbance, exponent, depth, features, beta=12.0, gamma=25.0)
    expected_color, expected_alpha = vanilla_alpha_over(absorbance, features)

    torch.testing.assert_close(actual_color, expected_color, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(actual_alpha, expected_alpha, rtol=1e-12, atol=1e-12)


def test_softmax_gs_preserves_original_final_transmittance() -> None:
    absorbance = torch.tensor([0.2, 0.4, 0.3, 0.1], dtype=torch.float64)
    exponent = torch.tensor([-0.2, -0.1, -1.0, -0.5], dtype=torch.float64)
    depth = torch.tensor([1.0, 1.02, 1.04, 1.5], dtype=torch.float64)
    features = torch.eye(4, dtype=torch.float64)

    _color, actual_alpha, rows = softmax_gs_composite(
        absorbance,
        exponent,
        depth,
        features,
        beta=5.0,
        gamma=7.0,
        return_debug=True,
    )
    _expected_color, expected_alpha = vanilla_alpha_over(absorbance, features)

    assert len(rows) == absorbance.shape[0]
    torch.testing.assert_close(actual_alpha, expected_alpha)


def test_softmax_gs_reference_has_finite_gradients() -> None:
    absorbance = torch.tensor([0.2, 0.4, 0.3], dtype=torch.float64, requires_grad=True)
    exponent = torch.tensor([-0.2, -0.1, -0.6], dtype=torch.float64, requires_grad=True)
    depth = torch.tensor([1.0, 1.1, 1.2], dtype=torch.float64)
    features = torch.tensor(
        [[0.2, 0.6, 0.1], [0.8, 0.0, 0.4], [0.1, 0.4, 0.7]],
        dtype=torch.float64,
    )
    beta = torch.tensor(4.0, dtype=torch.float64, requires_grad=True)
    gamma = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)

    color, alpha = softmax_gs_composite(absorbance, exponent, depth, features, beta=beta, gamma=gamma)
    loss = color.square().sum() + alpha.square()
    loss.backward()

    for grad in (absorbance.grad, exponent.grad, beta.grad, gamma.grad):
        assert grad is not None
        assert torch.isfinite(grad).all()


def test_softmax_gs_contribution_tape_reconstructs_any_feature_output() -> None:
    absorbance = torch.tensor([0.2, 0.4, 0.3, 0.1], dtype=torch.float64)
    exponent = torch.tensor([-0.2, -0.1, -1.0, -0.5], dtype=torch.float64)
    depth = torch.tensor([1.0, 1.02, 1.04, 1.5], dtype=torch.float64)
    features = torch.tensor(
        [[0.2, 0.6, 0.1], [0.8, 0.0, 0.4], [0.1, 0.4, 0.7], [0.7, 0.2, 0.3]],
        dtype=torch.float64,
    )

    expected_color, expected_alpha = softmax_gs_composite(
        absorbance,
        exponent,
        depth,
        features,
        beta=5.0,
        gamma=7.0,
    )
    weights, actual_alpha, rows = softmax_gs_contribution_tape(
        absorbance,
        exponent,
        depth,
        beta=5.0,
        gamma=7.0,
    )

    torch.testing.assert_close(weights @ features, expected_color)
    torch.testing.assert_close(actual_alpha, expected_alpha)
    torch.testing.assert_close(weights.sum(), actual_alpha)
    assert tuple(row.index for row in rows) == tuple(range(absorbance.shape[0]))
    torch.testing.assert_close(torch.stack([row.final_contribution_weight for row in rows]), weights)


def test_softmax_gs_contribution_tape_matches_vanilla_when_disabled() -> None:
    absorbance = torch.tensor([0.2, 0.35, 0.15], dtype=torch.float64)
    exponent = torch.tensor([-0.2, -0.7, -0.1], dtype=torch.float64)
    depth = torch.tensor([1.0, 1.1, 2.0], dtype=torch.float64)

    expected_weights, expected_alpha = vanilla_alpha_over_weights(absorbance)
    actual_weights, actual_alpha, _rows = softmax_gs_contribution_tape(
        absorbance,
        exponent,
        depth,
        beta=8.0,
        gamma=5.0,
        enabled=False,
    )

    torch.testing.assert_close(actual_weights, expected_weights)
    torch.testing.assert_close(actual_alpha, expected_alpha)


def test_softmax_gs_contribution_tape_gives_color_gradients() -> None:
    absorbance = torch.tensor([0.2, 0.4, 0.3], dtype=torch.float64, requires_grad=True)
    exponent = torch.tensor([-0.2, -0.1, -0.6], dtype=torch.float64, requires_grad=True)
    depth = torch.tensor([1.0, 1.1, 1.2], dtype=torch.float64)
    features = torch.tensor(
        [[0.2, 0.6, 0.1], [0.8, 0.0, 0.4], [0.1, 0.4, 0.7]],
        dtype=torch.float64,
        requires_grad=True,
    )
    grad_color = torch.tensor([0.3, -0.2, 0.7], dtype=torch.float64)

    color, _alpha = softmax_gs_composite(absorbance, exponent, depth, features, beta=4.0, gamma=3.0)
    color.backward(grad_color, retain_graph=True)
    expected_grad = features.grad.detach().clone()
    features.grad.zero_()

    weights, _alpha_from_weights, _rows = softmax_gs_contribution_tape(
        absorbance,
        exponent,
        depth,
        beta=4.0,
        gamma=3.0,
    )
    weighted_color = weights @ features
    weighted_color.backward(grad_color)

    torch.testing.assert_close(features.grad, expected_grad)
    torch.testing.assert_close(features.grad, weights.detach().unsqueeze(-1) * grad_color)


def test_softmax_gs_bounded_tape_selects_exact_topk_weights_in_ray_order() -> None:
    absorbance = torch.tensor([0.12, 0.5, 0.15, 0.4, 0.08], dtype=torch.float64)
    exponent = torch.tensor([-0.4, -0.05, -0.7, -0.1, -0.6], dtype=torch.float64)
    depth = torch.tensor([1.0, 1.01, 1.03, 1.04, 1.5], dtype=torch.float64)

    weights, final_alpha, rows = softmax_gs_contribution_tape(
        absorbance,
        exponent,
        depth,
        beta=6.0,
        gamma=4.0,
    )
    bounded = softmax_gs_bounded_contribution_tape(
        absorbance,
        exponent,
        depth,
        beta=6.0,
        gamma=4.0,
        k_limit=3,
    )
    expected_indices = torch.sort(torch.topk(weights, 3).indices).values

    torch.testing.assert_close(bounded.selected_indices, expected_indices)
    torch.testing.assert_close(bounded.selected_weights, weights[expected_indices])
    torch.testing.assert_close(bounded.residual_weight, final_alpha - weights[expected_indices].sum())
    assert tuple(row.index for row in bounded.selected_rows) == tuple(int(i) for i in expected_indices.tolist())
    torch.testing.assert_close(
        torch.stack([row.final_contribution_weight for row in bounded.selected_rows]),
        bounded.selected_weights,
    )
    torch.testing.assert_close(torch.stack([row.final_contribution_weight for row in rows]), weights)


def test_softmax_gs_bounded_tape_residual_bounds_unit_feature_error() -> None:
    torch.manual_seed(5)
    absorbance = torch.tensor([0.1, 0.35, 0.22, 0.4, 0.18, 0.08], dtype=torch.float64)
    exponent = torch.tensor([-0.5, -0.1, -0.35, -0.2, -0.8, -0.6], dtype=torch.float64)
    depth = torch.tensor([1.0, 1.02, 1.03, 1.08, 1.3, 1.6], dtype=torch.float64)
    features = torch.rand(absorbance.shape[0], 4, dtype=torch.float64)

    weights, final_alpha, _rows = softmax_gs_contribution_tape(
        absorbance,
        exponent,
        depth,
        beta=5.0,
        gamma=3.0,
    )
    bounded = softmax_gs_bounded_contribution_tape(
        absorbance,
        exponent,
        depth,
        beta=5.0,
        gamma=3.0,
        k_limit=2,
    )
    exact = weights @ features
    approx = bounded.selected_weights @ features[bounded.selected_indices]

    torch.testing.assert_close(bounded.final_alpha, final_alpha)
    assert (exact - approx).abs().max() <= bounded.residual_weight + 1.0e-12


def test_softmax_gs_bounded_tape_rejects_empty_limit() -> None:
    absorbance = torch.tensor([0.2], dtype=torch.float64)
    exponent = torch.tensor([-0.1], dtype=torch.float64)
    depth = torch.tensor([1.0], dtype=torch.float64)

    try:
        softmax_gs_bounded_contribution_tape(
            absorbance,
            exponent,
            depth,
            beta=1.0,
            gamma=1.0,
            k_limit=0,
        )
    except ValueError as exc:
        assert "k_limit" in str(exc)
    else:
        raise AssertionError("Expected k_limit validation to fail")
