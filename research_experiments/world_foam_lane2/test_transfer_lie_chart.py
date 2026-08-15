from __future__ import annotations

import unittest

import torch
from transfer_lie_chart import (
    DTYPE,
    affine_transfer_compose,
    chebyshev_nodes,
    check_lie_chart_cone,
    check_transfer_cone,
    closure_operation_counts,
    compare_hard_fixture_charts,
    evaluate_transfer_atlas,
    evaluate_transfer_atlas_vjp,
    fit_transfer_atlas,
    hard_opacity_moving_boundary_transfer,
    lie_chart_word_cotangents,
    transfer_lie_decode,
    transfer_lie_decode_vjp,
    transfer_lie_encode,
    transfer_lie_encode_vjp,
)


def _central_difference(function, value: torch.Tensor, *, epsilon: float = 2.0e-6) -> torch.Tensor:
    gradient = torch.zeros_like(value)
    flat_value = value.reshape(-1)
    flat_gradient = gradient.reshape(-1)
    for index in range(flat_value.numel()):
        plus = value.clone()
        minus = value.clone()
        plus.reshape(-1)[index] += epsilon
        minus.reshape(-1)[index] -= epsilon
        flat_gradient[index] = (function(plus) - function(minus)) / (2.0 * epsilon)
    return gradient


class TransferLieChartTest(unittest.TestCase):
    def test_stable_round_trip_including_identity_and_high_opacity(self) -> None:
        kappa = torch.tensor([0.0, 1.0e-12, 1.0e-6, 0.7, 20.0, 100.0], dtype=DTYPE)
        effective_color = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.2, 0.7, 1.0],
                [1.0, 0.3, 0.1],
                [0.4, 0.8, 0.2],
                [0.9, 0.1, 0.6],
                [0.3, 0.5, 0.7],
            ],
            dtype=DTYPE,
        )
        chart = torch.cat((kappa.unsqueeze(1), kappa.unsqueeze(1) * effective_color), dim=1)
        transfer = transfer_lie_decode(chart)
        recovered = transfer_lie_encode(transfer)
        torch.testing.assert_close(recovered, chart, atol=3.0e-14, rtol=3.0e-14)
        self.assertTrue(check_lie_chart_cone(chart).passed)
        self.assertTrue(check_transfer_cone(transfer).passed)

    def test_affine_transfer_composition_matches_sequential_action(self) -> None:
        front = torch.tensor([0.7, 0.1, 0.04, 0.02], dtype=DTYPE)
        back = torch.tensor([0.4, 0.03, 0.2, 0.01], dtype=DTYPE)
        background = torch.tensor([0.8, 0.5, 0.2], dtype=DTYPE)
        composed = affine_transfer_compose(front, back)
        sequential = front[1:] + front[0] * (back[1:] + back[0] * background)
        direct = composed[1:] + composed[0] * background
        torch.testing.assert_close(direct, sequential)

    def test_encode_vjp_matches_autograd_and_finite_difference(self) -> None:
        beta = torch.tensor([1.0, 1.0 - 1.0e-12, 0.7, 1.0e-12], dtype=DTYPE)
        color = torch.tensor(
            [[0.0, 0.0, 0.0], [0.2, 0.7, 0.4], [0.8, 0.1, 0.5], [0.3, 0.6, 0.9]],
            dtype=DTYPE,
        )
        transfer = torch.cat((beta.unsqueeze(1), (1.0 - beta).unsqueeze(1) * color), dim=1)
        grad_chart = torch.tensor(
            [
                [0.3, -0.2, 0.4, 0.7],
                [-0.1, 0.5, 0.2, -0.6],
                [0.8, -0.4, 0.3, 0.1],
                [-0.7, 0.9, -0.2, 0.4],
            ],
            dtype=DTYPE,
        )
        differentiable = transfer.clone().requires_grad_(True)
        expected = torch.autograd.grad(
            (transfer_lie_encode(differentiable) * grad_chart).sum(),
            differentiable,
        )[0]
        actual = transfer_lie_encode_vjp(transfer, grad_chart)
        torch.testing.assert_close(actual, expected, atol=3.0e-12, rtol=3.0e-12)

        moderate = transfer[2].clone()
        moderate_grad = grad_chart[2]
        finite_difference = _central_difference(
            lambda value: (transfer_lie_encode(value) * moderate_grad).sum(),
            moderate,
        )
        torch.testing.assert_close(
            transfer_lie_encode_vjp(moderate, moderate_grad),
            finite_difference,
            atol=2.0e-9,
            rtol=2.0e-9,
        )

    def test_decode_vjp_matches_autograd_and_finite_difference(self) -> None:
        kappa = torch.tensor([0.0, 1.0e-12, 0.7, 40.0], dtype=DTYPE)
        color = torch.tensor(
            [[0.0, 0.0, 0.0], [0.2, 0.7, 0.4], [0.8, 0.1, 0.5], [0.3, 0.6, 0.9]],
            dtype=DTYPE,
        )
        chart = torch.cat((kappa.unsqueeze(1), kappa.unsqueeze(1) * color), dim=1)
        grad_transfer = torch.tensor(
            [
                [0.3, -0.2, 0.4, 0.7],
                [-0.1, 0.5, 0.2, -0.6],
                [0.8, -0.4, 0.3, 0.1],
                [-0.7, 0.9, -0.2, 0.4],
            ],
            dtype=DTYPE,
        )
        differentiable = chart.clone().requires_grad_(True)
        expected = torch.autograd.grad(
            (transfer_lie_decode(differentiable) * grad_transfer).sum(),
            differentiable,
        )[0]
        actual = transfer_lie_decode_vjp(chart, grad_transfer)
        torch.testing.assert_close(actual, expected, atol=3.0e-12, rtol=3.0e-12)

        moderate = chart[2].clone()
        moderate_grad = grad_transfer[2]
        finite_difference = _central_difference(
            lambda value: (transfer_lie_decode(value) * moderate_grad).sum(),
            moderate,
        )
        torch.testing.assert_close(
            transfer_lie_decode_vjp(moderate, moderate_grad),
            finite_difference,
            atol=2.0e-10,
            rtol=2.0e-9,
        )

    def test_atlas_vjp_matches_autograd_for_both_coordinate_choices(self) -> None:
        rank = 7
        nodes = chebyshev_nodes(rank, t_min=-1.0, t_max=1.0)
        node_transfer = hard_opacity_moving_boundary_transfer(nodes)
        times = torch.linspace(-1.0, 1.0, 23, dtype=DTYPE)
        cotangent = torch.linspace(-0.4, 0.7, 23 * 4, dtype=DTYPE).reshape(23, 4)
        for chart in ("raw", "lie"):
            differentiable = node_transfer.clone().requires_grad_(True)
            differentiable_atlas = fit_transfer_atlas(
                differentiable,
                t_min=-1.0,
                t_max=1.0,
                chart=chart,
            )
            expected = torch.autograd.grad(
                (evaluate_transfer_atlas(differentiable_atlas, times) * cotangent).sum(),
                differentiable,
            )[0]
            atlas = fit_transfer_atlas(
                node_transfer,
                t_min=-1.0,
                t_max=1.0,
                chart=chart,
            )
            actual = evaluate_transfer_atlas_vjp(atlas, times, cotangent)
            torch.testing.assert_close(actual, expected, atol=2.0e-10, rtol=2.0e-10)

    def test_high_opacity_word_cotangent_avoids_raw_beta_adjoint(self) -> None:
        optical_depth = torch.tensor([40.0, 35.0, 25.0], dtype=DTYPE, requires_grad=True)
        colors = torch.tensor(
            [[0.9, 0.1, 0.3], [0.2, 0.8, 0.4], [0.1, 0.3, 0.9]],
            dtype=DTYPE,
        )
        prefix_beta = torch.ones((), dtype=DTYPE)
        total_moment = torch.zeros(3, dtype=DTYPE)
        prefix_states = []
        for tau, color in zip(optical_depth, colors, strict=True):
            prefix_states.append((prefix_beta, total_moment, color))
            beta = torch.exp(-tau)
            total_moment = total_moment + prefix_beta * (1.0 - beta) * color
            prefix_beta = prefix_beta * beta
        kappa_total = optical_depth.sum()
        encoded = torch.cat(
            (
                kappa_total.reshape(1),
                kappa_total / (-torch.expm1(-kappa_total)) * total_moment,
            )
        )
        grad_chart = torch.tensor([0.4, -0.2, 0.7, 0.1], dtype=DTYPE)
        expected = torch.autograd.grad((encoded * grad_chart).sum(), optical_depth)[0]

        grad_moment, grad_kappa_word = lie_chart_word_cotangents(
            kappa_total.detach(),
            total_moment.detach(),
            grad_chart,
        )
        actual = torch.stack(
            [
                torch.dot(
                    grad_moment,
                    prefix_m + beta_prefix * color - total_moment.detach(),
                )
                + grad_kappa_word
                for beta_prefix, prefix_m, color in prefix_states
            ]
        )
        self.assertTrue(bool(torch.isfinite(actual).all().item()))
        torch.testing.assert_close(actual, expected, atol=2.0e-14, rtol=2.0e-14)

    def test_hard_fixture_lie_chart_wins_but_is_not_a_universal_winner(self) -> None:
        reports = compare_hard_fixture_charts((2, 8, 16, 32))
        for report in reports:
            self.assertLess(report.lie_max_transfer_error, 4.0e-14)
            self.assertLess(report.lie_max_parameter_vjp_error, 2.0e-14)
            self.assertTrue(report.lie_chart_cone.passed)
            self.assertTrue(report.lie_transfer_cone.passed)
            self.assertFalse(report.raw_transfer_cone.passed)
        self.assertGreater(reports[2].raw_max_transfer_error, 2.0e-3)
        self.assertGreater(reports[2].raw_max_parameter_vjp_error, 1.0e-6)

        times = torch.linspace(-1.0, 1.0, 101, dtype=DTYPE)
        nodes = chebyshev_nodes(2, t_min=-1.0, t_max=1.0)

        def raw_linear_fixture(t: torch.Tensor) -> torch.Tensor:
            beta = 0.6 + 0.1 * t
            moment = torch.stack((0.10 + 0.02 * t, 0.05 + 0.01 * t, 0.03 - 0.01 * t), dim=1)
            return torch.cat((beta.unsqueeze(1), moment), dim=1)

        exact = raw_linear_fixture(times)
        raw = evaluate_transfer_atlas(
            fit_transfer_atlas(raw_linear_fixture(nodes), t_min=-1.0, t_max=1.0, chart="raw"),
            times,
        )
        lie = evaluate_transfer_atlas(
            fit_transfer_atlas(raw_linear_fixture(nodes), t_min=-1.0, t_max=1.0, chart="lie"),
            times,
        )
        self.assertLess(float((raw - exact).abs().max().item()), 2.0e-16)
        self.assertGreater(float((lie - exact).abs().max().item()), 1.0e-4)

    def test_cone_checks_fail_closed_and_complexity_does_not_hide_frame_replay(self) -> None:
        self.assertFalse(check_lie_chart_cone(torch.tensor([[-0.1, 0.0, 0.0, 0.0]], dtype=DTYPE)).passed)
        self.assertFalse(check_lie_chart_cone(torch.tensor([[0.2, 0.3, 0.0, 0.0]], dtype=DTYPE)).passed)
        self.assertFalse(check_transfer_cone(torch.tensor([[0.8, 0.3, 0.0, 0.0]], dtype=DTYPE)).passed)
        report = closure_operation_counts(run_count=11, rank=8, frame_count=300)
        self.assertEqual(report["world_compile_interactions"], 88)
        self.assertEqual(report["sample_basis_interactions"], 2400)
        self.assertEqual(report["total_interactions"], 2488)


if __name__ == "__main__":
    unittest.main()
