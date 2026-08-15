from __future__ import annotations

import unittest
from dataclasses import replace

import torch
from compiled_lie_world_adjoint import (
    DTYPE,
    AdaptiveLieWorldCompilePolicy,
    compile_adaptive_lie_world_atlas,
    compile_lie_world_atlas,
    compiled_lie_world_mse_vjp,
    piecewise_compiled_lie_world_mse_vjp,
    referenced_depth_coefficient_incidence,
    refresh_fixed_topology_lie_world_atlas,
    sampled_lie_world_tangent_error,
    sampled_lie_world_transfer_error,
    sparse_factorized_depth_coefficients,
    sparse_factorized_depth_coefficients_boundary_vjp,
)
from compiled_transfer_adjoint import (
    direct_word_transfer,
    make_stable_cell_word,
    streamed_word_mse_vjp,
)
from transfer_lie_chart import check_lie_chart_cone, evaluate_transfer_atlas, evaluate_transfer_atlas_chart


def _smooth_fixture() -> dict[str, object]:
    return {
        "boundary": torch.tensor(
            [
                [0.12, -0.05, 1.00, -0.08, -0.95],
                [-0.07, 0.08, 1.00, 0.04, -1.85],
            ],
            dtype=DTYPE,
        ),
        "ray_coefficients": torch.tensor(
            [
                [
                    0.05,
                    -0.02,
                    0.10,
                    0.02,
                    0.01,
                    -0.01,
                    0.02,
                    -0.03,
                    1.00,
                    0.01,
                    0.02,
                    0.03,
                ],
                [
                    -0.10,
                    0.04,
                    0.02,
                    -0.01,
                    0.03,
                    0.02,
                    -0.03,
                    0.01,
                    0.95,
                    0.02,
                    -0.01,
                    0.01,
                ],
            ],
            dtype=DTYPE,
        ),
        "words": (
            make_stable_cell_word([0, 1, 2], [-1, 0, 1], [0, 1, -2]),
            make_stable_cell_word([2, 1, 0], [-1, 0, 1], [0, 1, -2]),
        ),
        "site_density": torch.tensor([0.45, 0.82, 0.31], dtype=DTYPE),
        "site_color": torch.tensor(
            [[0.91, 0.16, 0.08], [0.10, 0.66, 0.93], [0.38, 0.84, 0.25]],
            dtype=DTYPE,
        ),
        "background": torch.tensor([0.03, 0.04, 0.06], dtype=DTYPE),
        "t_min": -0.8,
        "t_max": 0.9,
        "near": 0.1,
        "far": 3.0,
    }


def _hard_fixture(*, density: float = 50.0) -> dict[str, object]:
    return {
        "boundary": torch.tensor([[0.0, 0.0, 1.0, -0.9, -1.0]], dtype=DTYPE),
        "ray_coefficients": torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
            dtype=DTYPE,
        ),
        "words": (make_stable_cell_word([0, 1], [-1, 0], [0, -2]),),
        "site_density": torch.tensor([density, 0.0], dtype=DTYPE),
        "site_color": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=DTYPE),
        "background": torch.tensor([0.07, 0.03, 0.11], dtype=DTYPE),
        "t_min": -1.0,
        "t_max": 1.0,
        "near": 0.05,
        "far": 2.0,
    }


def _common(fixture: dict[str, object]) -> dict[str, object]:
    return {
        key: fixture[key]
        for key in (
            "boundary",
            "ray_coefficients",
            "words",
            "site_density",
            "site_color",
            "background",
            "t_min",
            "t_max",
            "near",
            "far",
        )
    }


def _adaptive_policy(**overrides: object) -> AdaptiveLieWorldCompilePolicy:
    values: dict[str, object] = {
        "node_count_schedule": (2, 4, 8, 16),
        "probe_validation_count": 17,
        "heldout_validation_count": 16,
        "probe_direction_count": 2,
        "heldout_direction_count": 2,
        "forward_absolute_tolerance": 1.0e-10,
        "forward_relative_tolerance": 1.0e-6,
        "tangent_absolute_tolerance": 1.0e-10,
        "tangent_relative_tolerance": 1.0e-3,
        "max_split_depth": 3,
        "max_chart_count": 8,
    }
    values.update(overrides)
    return AdaptiveLieWorldCompilePolicy(**values)


class CompiledLieWorldAdjointTest(unittest.TestCase):
    def test_sparse_incidence_coefficients_and_boundary_vjp_match_dense_autograd(self) -> None:
        fixture = _smooth_fixture()
        incidence = referenced_depth_coefficient_incidence(fixture["words"])
        self.assertEqual(tuple(incidence.shape), (4, 2))
        sparse = sparse_factorized_depth_coefficients(
            fixture["boundary"],
            fixture["ray_coefficients"],
            incidence,
        )

        boundary = fixture["boundary"].clone().requires_grad_(True)
        reference_sparse = sparse_factorized_depth_coefficients(
            boundary,
            fixture["ray_coefficients"],
            incidence,
        )
        cotangent = torch.linspace(-0.4, 0.7, sparse.numel(), dtype=DTYPE).reshape_as(sparse)
        expected = torch.autograd.grad((reference_sparse * cotangent).sum(), boundary)[0]
        actual = sparse_factorized_depth_coefficients_boundary_vjp(
            fixture["boundary"],
            fixture["ray_coefficients"],
            incidence,
            cotangent,
        )
        torch.testing.assert_close(actual, expected, atol=2.0e-15, rtol=2.0e-14)

    def test_smooth_fixture_matches_exact_streamed_world_vjp(self) -> None:
        fixture = _smooth_fixture()
        times = torch.linspace(fixture["t_min"], fixture["t_max"], 65, dtype=DTYPE)
        targets = torch.full((2, 65, 3), 0.29, dtype=DTYPE)
        exact = streamed_word_mse_vjp(
            **_common(fixture),
            times=times,
            targets=targets,
            frame_block_size=7,
            return_predictions=True,
            compute_ray_grad=False,
        )
        compiled = compiled_lie_world_mse_vjp(
            **_common(fixture),
            times=times,
            targets=targets,
            node_count=16,
            frame_block_size=7,
            track_block_size=1,
            validation_count=65,
            return_predictions=True,
        )
        self.assertLess(compiled.sampled_validation_error, 2.0e-10)
        torch.testing.assert_close(compiled.loss, exact.loss, atol=2.0e-12, rtol=2.0e-10)
        for observed, reference in (
            (compiled.predictions, exact.predictions),
            (compiled.grad_boundary, exact.grad_boundary),
            (compiled.grad_site_density, exact.grad_site_density),
            (compiled.grad_site_color, exact.grad_site_color),
            (compiled.grad_depth_coefficients, exact.grad_depth_coefficients),
        ):
            torch.testing.assert_close(observed, reference, atol=2.0e-10, rtol=2.0e-9)

    def test_hard_opacity_two_node_primal_is_exact_but_dormant_material_tangent_is_not(self) -> None:
        fixture = _hard_fixture()
        times = torch.linspace(-1.0, 1.0, 97, dtype=DTYPE)
        phase = torch.linspace(0.0, 3.0, 97, dtype=DTYPE)
        targets = torch.stack(
            (
                0.2 + 0.1 * torch.sin(phase),
                0.3 + 0.05 * torch.cos(phase),
                0.1 + 0.03 * torch.sin(2.0 * phase),
            ),
            dim=1,
        ).unsqueeze(0)
        exact = streamed_word_mse_vjp(
            **_common(fixture),
            times=times,
            targets=targets,
            frame_block_size=11,
            return_predictions=True,
            compute_ray_grad=False,
        )
        compiled = compiled_lie_world_mse_vjp(
            **_common(fixture),
            times=times,
            targets=targets,
            node_count=2,
            frame_block_size=11,
            track_block_size=1,
            validation_count=65,
            return_predictions=True,
        )
        self.assertLess(compiled.sampled_validation_error, 3.0e-14)
        torch.testing.assert_close(compiled.loss, exact.loss, atol=2.0e-14, rtol=2.0e-13)
        for observed, reference in (
            (compiled.predictions, exact.predictions),
            (compiled.grad_boundary, exact.grad_boundary),
            (compiled.grad_site_color, exact.grad_site_color),
        ):
            torch.testing.assert_close(observed, reference, atol=4.0e-13, rtol=4.0e-12)
        torch.testing.assert_close(
            compiled.grad_site_density[0],
            exact.grad_site_density[0],
            atol=4.0e-13,
            rtol=4.0e-12,
        )
        self.assertGreater(
            float((compiled.grad_site_density[1] - exact.grad_site_density[1]).abs().item()),
            1.0e-3,
        )
        self.assertGreater(
            float((compiled.grad_depth_coefficients - exact.grad_depth_coefficients).abs().max().item()),
            1.0e-2,
        )
        self.assertGreater(compiled.sampled_tangent_validation.maximum_world_gradient_error, 1.0e-2)

    def test_hard_dormant_tail_tangent_error_falls_with_rank_and_has_a_separate_gate(self) -> None:
        fixture = _hard_fixture()
        times = torch.linspace(-1.0, 1.0, 65, dtype=DTYPE)
        targets = torch.full((1, 65, 3), 0.2, dtype=DTYPE)
        reports = []
        for node_count in (2, 4, 8, 16, 32):
            result = compiled_lie_world_mse_vjp(
                **_common(fixture),
                times=times,
                targets=targets,
                node_count=node_count,
                frame_block_size=8,
                track_block_size=1,
                validation_count=65,
                sampled_forward_tolerance=1.0e-12,
                sampled_tangent_tolerance=1.0e-8 if node_count == 32 else None,
            )
            reports.append(result.sampled_tangent_validation.maximum_world_gradient_error)
            self.assertLess(result.sampled_validation_error, 4.0e-14)
        for previous, current in zip(reports[:-1], reports[1:], strict=True):
            self.assertLess(current, previous)
        self.assertGreater(reports[0], 1.0e-2)
        self.assertLess(reports[-1], 1.0e-8)
        with self.assertRaisesRegex(ValueError, "tangent/VJP rank gate failed"):
            compiled_lie_world_mse_vjp(
                **_common(fixture),
                times=times,
                targets=targets,
                node_count=2,
                frame_block_size=8,
                track_block_size=1,
                validation_count=65,
                sampled_forward_tolerance=1.0e-12,
                sampled_tangent_tolerance=1.0e-4,
            )

    def test_adaptive_compile_uses_multiple_probe_and_heldout_directions_and_splits(self) -> None:
        fixture = _hard_fixture()
        atlas = compile_adaptive_lie_world_atlas(
            **{
                key: fixture[key]
                for key in (
                    "boundary",
                    "ray_coefficients",
                    "words",
                    "site_density",
                    "site_color",
                    "t_min",
                    "t_max",
                    "near",
                    "far",
                )
            },
            policy=_adaptive_policy(),
            track_block_size=1,
            frame_block_size=8,
        )
        self.assertEqual(
            atlas.selection_signature,
            ((-1.0, -0.5, 16), (-0.5, 0.0, 2), (0.0, 1.0, 2)),
        )
        self.assertEqual(atlas.chart_count, 3)
        self.assertEqual(atlas.total_node_count, 20)
        self.assertTrue(atlas.supplied_word_ordering_check["passed"])
        for selection in atlas.selections:
            self.assertTrue(selection.probe_validation.passed)
            self.assertTrue(selection.heldout_validation.passed)
            self.assertEqual(selection.probe_validation.tangent.direction_count, 2)
            self.assertEqual(selection.heldout_validation.tangent.direction_count, 2)
            self.assertEqual(
                {direction.split for direction in selection.probe_validation.tangent.directions},
                {"probe"},
            )
            self.assertEqual(
                {direction.split for direction in selection.heldout_validation.tangent.directions},
                {"heldout"},
            )
            self.assertLessEqual(
                selection.probe_validation.tangent.maximum_normalized_world_gradient_error,
                1.0,
            )
            self.assertLessEqual(
                selection.heldout_validation.tangent.maximum_normalized_world_gradient_error,
                1.0,
            )

    def test_piecewise_warm_vjp_is_frame_count_independent_on_world_side(self) -> None:
        fixture = _hard_fixture()
        atlas = compile_adaptive_lie_world_atlas(
            **{
                key: fixture[key]
                for key in (
                    "boundary",
                    "ray_coefficients",
                    "words",
                    "site_density",
                    "site_color",
                    "t_min",
                    "t_max",
                    "near",
                    "far",
                )
            },
            policy=_adaptive_policy(),
            track_block_size=1,
            frame_block_size=8,
        )
        signature = atlas.selection_signature
        refreshed = refresh_fixed_topology_lie_world_atlas(
            atlas,
            assume_fixed_topology=True,
            boundary=fixture["boundary"],
            ray_coefficients=fixture["ray_coefficients"],
            site_density=fixture["site_density"] + torch.tensor([0.1, 0.0], dtype=DTYPE),
            site_color=fixture["site_color"],
        )
        self.assertEqual(refreshed.selection_signature, signature)
        self.assertIs(refreshed.selections, atlas.selections)
        self.assertFalse(
            torch.equal(
                refreshed.charts[0].transfer_atlas.coefficients,
                atlas.charts[0].transfer_atlas.coefficients,
            )
        )
        results = []
        for frame_count in (13, 101):
            times = torch.linspace(-1.0, 1.0, frame_count, dtype=DTYPE)
            targets = torch.full((1, frame_count, 3), 0.2, dtype=DTYPE)
            result = piecewise_compiled_lie_world_mse_vjp(
                atlas,
                boundary=fixture["boundary"],
                ray_coefficients=fixture["ray_coefficients"],
                site_density=fixture["site_density"],
                site_color=fixture["site_color"],
                times=times,
                targets=targets,
                background=fixture["background"],
                frame_block_size=7,
                track_block_size=1,
                return_predictions=True,
            )
            exact = streamed_word_mse_vjp(
                **_common(fixture),
                times=times,
                targets=targets,
                frame_block_size=7,
                return_predictions=True,
                compute_ray_grad=False,
            )
            self.assertEqual(result.atlas.selection_signature, signature)
            self.assertEqual(result.accounting["sampled_validation_count"], 0)
            self.assertEqual(result.accounting["validation_exact_run_interactions"], 0)
            self.assertEqual(result.accounting["frame_run_reverse_state_elements"], 0)
            self.assertEqual(result.accounting["per_sample_run_tape_bytes"], 0)
            torch.testing.assert_close(result.predictions, exact.predictions, atol=8.0e-14, rtol=8.0e-13)
            torch.testing.assert_close(result.loss, exact.loss, atol=8.0e-14, rtol=8.0e-13)
            torch.testing.assert_close(
                result.grad_site_density,
                exact.grad_site_density,
                atol=5.0e-8,
                rtol=5.0e-7,
            )
            results.append(result)
        small, large = results
        self.assertEqual(
            small.accounting["refresh_world_forward_run_interactions"],
            large.accounting["refresh_world_forward_run_interactions"],
        )
        self.assertEqual(
            small.accounting["step_world_reverse_run_interactions"],
            large.accounting["step_world_reverse_run_interactions"],
        )
        self.assertGreater(
            large.accounting["sample_basis_interactions"],
            small.accounting["sample_basis_interactions"],
        )

    def test_adaptive_compile_fails_closed_on_holdout_topology_and_resource_limits(self) -> None:
        fixture = _hard_fixture()
        compile_inputs = {
            key: fixture[key]
            for key in (
                "boundary",
                "ray_coefficients",
                "words",
                "site_density",
                "site_color",
                "t_min",
                "t_max",
                "near",
                "far",
            )
        }
        with self.assertRaisesRegex(ValueError, "held-out audit failed"):
            compile_adaptive_lie_world_atlas(
                **compile_inputs,
                policy=_adaptive_policy(tangent_relative_tolerance=1.0e-2),
                track_block_size=1,
                frame_block_size=8,
            )
        with self.assertRaisesRegex(ValueError, "exhausted the rank schedule and split depth"):
            compile_adaptive_lie_world_atlas(
                **compile_inputs,
                policy=_adaptive_policy(
                    node_count_schedule=(2,),
                    tangent_relative_tolerance=1.0e-6,
                    max_split_depth=0,
                ),
                track_block_size=1,
                frame_block_size=8,
            )
        with self.assertRaisesRegex(ValueError, "midpoint heldout samples are disjoint"):
            compile_adaptive_lie_world_atlas(
                **compile_inputs,
                policy=_adaptive_policy(
                    probe_validation_count=5,
                    heldout_validation_count=2,
                ),
                track_block_size=1,
                frame_block_size=8,
            )

        topology_crossing = fixture["boundary"].clone()
        topology_crossing[0, 3] = -1.2
        with self.assertRaisesRegex(ValueError, "loses strict endpoint order"):
            compile_adaptive_lie_world_atlas(
                **{**compile_inputs, "boundary": topology_crossing},
                policy=_adaptive_policy(max_split_depth=3),
                track_block_size=1,
                frame_block_size=8,
            )

    def test_tangent_report_exposes_scale_normalized_parameter_blocks(self) -> None:
        fixture = _hard_fixture()
        atlas = compile_lie_world_atlas(
            boundary=fixture["boundary"],
            ray_coefficients=fixture["ray_coefficients"],
            words=fixture["words"],
            site_density=fixture["site_density"],
            site_color=fixture["site_color"],
            t_min=fixture["t_min"],
            t_max=fixture["t_max"],
            near=fixture["near"],
            far=fixture["far"],
            node_count=2,
        )
        report = sampled_lie_world_tangent_error(
            atlas,
            boundary=fixture["boundary"],
            ray_coefficients=fixture["ray_coefficients"],
            site_density=fixture["site_density"],
            site_color=fixture["site_color"],
            validation_count=17,
            track_block_size=1,
            frame_block_size=8,
            direction_ids=(0, 1, 2),
            direction_split="probe",
            absolute_tolerance=1.0e-10,
            relative_tolerance=1.0e-4,
        )
        self.assertEqual(report.direction_count, 3)
        self.assertEqual([direction.direction_id for direction in report.directions], [0, 1, 2])
        self.assertGreater(report.grad_site_density_error, 1.0e-4)
        self.assertGreater(report.grad_depth_coefficient_normalized_error, 1.0)
        self.assertTrue(
            all(
                torch.isfinite(torch.tensor(value)).item()
                for value in (
                    report.grad_boundary_normalized_error,
                    report.grad_site_density_normalized_error,
                    report.grad_site_color_normalized_error,
                    report.grad_depth_coefficient_normalized_error,
                )
            )
        )

    def test_high_opacity_chart_reverse_remains_finite_after_beta_underflow(self) -> None:
        fixture = _hard_fixture(density=1000.0)
        times = torch.linspace(-1.0, 1.0, 41, dtype=DTYPE)
        targets = torch.full((1, 41, 3), 0.17, dtype=DTYPE)
        result = compiled_lie_world_mse_vjp(
            **_common(fixture),
            times=times,
            targets=targets,
            node_count=2,
            frame_block_size=5,
            track_block_size=1,
            validation_count=33,
            return_predictions=True,
        )
        self.assertEqual(float(torch.exp(torch.tensor(-1000.0, dtype=DTYPE)).item()), 0.0)
        for tensor in (
            result.loss,
            result.predictions,
            result.grad_boundary,
            result.grad_site_density,
            result.grad_site_color,
            result.grad_depth_coefficients,
        ):
            self.assertTrue(bool(torch.isfinite(tensor).all().item()))
        self.assertLess(result.sampled_validation_error, 2.0e-13)

    def test_tiny_optical_depth_preserves_chart_color_and_active_density_vjp(self) -> None:
        times = torch.linspace(-1.0, 1.0, 9, dtype=DTYPE)
        for density in (1.0e-16, 1.0e-18):
            fixture = _hard_fixture(density=density)
            atlas = compile_lie_world_atlas(
                boundary=fixture["boundary"],
                ray_coefficients=fixture["ray_coefficients"],
                words=fixture["words"],
                site_density=fixture["site_density"],
                site_color=fixture["site_color"],
                t_min=fixture["t_min"],
                t_max=fixture["t_max"],
                near=fixture["near"],
                far=fixture["far"],
                node_count=2,
            )
            node_chart = evaluate_transfer_atlas_chart(
                atlas.transfer_atlas,
                atlas.transfer_atlas.node_times,
            )[0]
            self.assertTrue(bool(torch.all(node_chart[:, 0] > 0.0).item()))
            torch.testing.assert_close(
                node_chart[:, 1] / node_chart[:, 0],
                torch.ones(2, dtype=DTYPE),
                atol=3.0e-15,
                rtol=3.0e-15,
            )
            targets = torch.full((1, 9, 3), 0.17, dtype=DTYPE)
            result = compiled_lie_world_mse_vjp(
                **_common(fixture),
                times=times,
                targets=targets,
                node_count=2,
                frame_block_size=3,
                track_block_size=1,
                validation_count=17,
            )
            differentiable_density = torch.tensor(density, dtype=DTYPE, requires_grad=True)
            length = 0.95 + 0.9 * times
            tau = differentiable_density * length
            beta = torch.exp(-tau)
            alpha = -torch.expm1(-tau)
            prediction = alpha.unsqueeze(1) * fixture["site_color"][0] + beta.unsqueeze(1) * fixture["background"]
            expected_density_grad = torch.autograd.grad(
                (prediction - targets[0]).square().mean(),
                differentiable_density,
            )[0]
            torch.testing.assert_close(
                result.grad_site_density[0],
                expected_density_grad,
                atol=3.0e-14,
                rtol=3.0e-13,
            )

    def test_manual_vjp_detaches_autograd_inputs_and_retains_no_graph(self) -> None:
        fixture = _smooth_fixture()
        differentiable = {
            key: fixture[key].clone().requires_grad_(True)
            for key in ("boundary", "ray_coefficients", "site_density", "site_color", "background")
        }
        times = torch.linspace(fixture["t_min"], fixture["t_max"], 9, dtype=DTYPE).requires_grad_(True)
        targets = torch.full((2, 9, 3), 0.23, dtype=DTYPE, requires_grad=True)
        result = compiled_lie_world_mse_vjp(
            boundary=differentiable["boundary"],
            ray_coefficients=differentiable["ray_coefficients"],
            words=fixture["words"],
            site_density=differentiable["site_density"],
            site_color=differentiable["site_color"],
            times=times,
            targets=targets,
            background=differentiable["background"],
            t_min=fixture["t_min"],
            t_max=fixture["t_max"],
            near=fixture["near"],
            far=fixture["far"],
            node_count=8,
            frame_block_size=3,
            track_block_size=1,
            validation_count=17,
            return_predictions=True,
        )
        outputs = (
            result.loss,
            result.predictions,
            result.grad_boundary,
            result.grad_site_density,
            result.grad_site_color,
            result.grad_depth_coefficients,
            result.atlas.transfer_atlas.coefficients,
        )
        for output in outputs:
            self.assertFalse(output.requires_grad)
            self.assertIsNone(output.grad_fn)
        for source in (*differentiable.values(), times, targets):
            self.assertIsNone(source.grad)

    def test_sample_chart_cone_and_forward_error_gates_fail_closed(self) -> None:
        fixture = _smooth_fixture()
        times = torch.linspace(fixture["t_min"], fixture["t_max"], 9, dtype=DTYPE)
        targets = torch.full((2, 9, 3), 0.23, dtype=DTYPE)
        with self.assertRaisesRegex(ValueError, "forward rank gate failed"):
            compiled_lie_world_mse_vjp(
                **_common(fixture),
                times=times,
                targets=targets,
                node_count=2,
                frame_block_size=3,
                track_block_size=1,
                validation_count=65,
                sampled_forward_tolerance=1.0e-5,
            )

        cone_fixture = _hard_fixture()
        base = compile_lie_world_atlas(
            boundary=cone_fixture["boundary"],
            ray_coefficients=cone_fixture["ray_coefficients"],
            words=cone_fixture["words"],
            site_density=cone_fixture["site_density"],
            site_color=cone_fixture["site_color"],
            t_min=cone_fixture["t_min"],
            t_max=cone_fixture["t_max"],
            near=cone_fixture["near"],
            far=cone_fixture["far"],
            node_count=2,
        )
        coefficients = torch.tensor(
            [[[1.0, 0.5, 0.5, 0.5], [1.3, 0.65, 0.65, 0.65]]],
            dtype=DTYPE,
        )
        overshooting = replace(
            base,
            transfer_atlas=replace(base.transfer_atlas, coefficients=coefficients),
        )
        node_chart = evaluate_transfer_atlas_chart(
            overshooting.transfer_atlas,
            overshooting.transfer_atlas.node_times,
        )
        self.assertTrue(check_lie_chart_cone(node_chart).passed)
        with self.assertRaisesRegex(ValueError, "left the physical cone between nodes"):
            sampled_lie_world_transfer_error(
                overshooting,
                boundary=cone_fixture["boundary"],
                ray_coefficients=cone_fixture["ray_coefficients"],
                site_density=cone_fixture["site_density"],
                site_color=cone_fixture["site_color"],
                validation_count=17,
                track_block_size=1,
                frame_block_size=3,
            )

    def test_reverse_state_has_no_frame_by_run_tape(self) -> None:
        fixture = _smooth_fixture()
        results = []
        for frame_count in (8, 128):
            times = torch.linspace(fixture["t_min"], fixture["t_max"], frame_count, dtype=DTYPE)
            targets = torch.full((2, frame_count, 3), 0.23, dtype=DTYPE)
            results.append(
                compiled_lie_world_mse_vjp(
                    **_common(fixture),
                    times=times,
                    targets=targets,
                    node_count=8,
                    frame_block_size=4,
                    track_block_size=1,
                    validation_count=17,
                )
            )
        small, large = results
        for result in results:
            self.assertEqual(result.accounting["frame_run_reverse_state_elements"], 0)
            self.assertEqual(result.accounting["per_sample_run_tape_bytes"], 0)
            self.assertEqual(
                result.accounting["world_reverse_run_interactions"],
                result.accounting["node_count"] * result.accounting["run_count"],
            )
            self.assertEqual(
                result.accounting["validation_exact_forward_run_interactions"],
                17 * result.accounting["run_count"],
            )
            self.assertEqual(
                result.accounting["validation_exact_tangent_run_interactions"],
                3 * 17 * result.accounting["run_count"],
            )
        self.assertEqual(
            small.accounting["reverse_state_bytes_excluding_targets_and_predictions"],
            large.accounting["reverse_state_bytes_excluding_targets_and_predictions"],
        )
        self.assertEqual(
            small.accounting["world_reverse_run_interactions"],
            large.accounting["world_reverse_run_interactions"],
        )
        self.assertGreater(
            large.accounting["sample_basis_interactions"],
            small.accounting["sample_basis_interactions"],
        )

        default_times = torch.linspace(fixture["t_min"], fixture["t_max"], 3, dtype=DTYPE)
        default_result = compiled_lie_world_mse_vjp(
            **_common(fixture),
            times=default_times,
            targets=torch.full((2, 3, 3), 0.23, dtype=DTYPE),
            node_count=8,
            frame_block_size=3,
            track_block_size=1,
        )
        self.assertIsNone(default_result.sampled_validation_error)
        self.assertIsNone(default_result.sampled_tangent_validation)
        self.assertEqual(default_result.accounting["sampled_validation_count"], 0)
        self.assertEqual(default_result.accounting["validation_exact_forward_run_interactions"], 0)
        self.assertEqual(default_result.accounting["validation_exact_tangent_run_interactions"], 0)

    def test_compiled_atlas_exposes_sampled_error_gate_inputs(self) -> None:
        fixture = _smooth_fixture()
        atlas = compile_lie_world_atlas(
            boundary=fixture["boundary"],
            ray_coefficients=fixture["ray_coefficients"],
            words=fixture["words"],
            site_density=fixture["site_density"],
            site_color=fixture["site_color"],
            t_min=fixture["t_min"],
            t_max=fixture["t_max"],
            near=fixture["near"],
            far=fixture["far"],
            node_count=12,
        )
        sampled_error = sampled_lie_world_transfer_error(
            atlas,
            boundary=fixture["boundary"],
            ray_coefficients=fixture["ray_coefficients"],
            site_density=fixture["site_density"],
            site_color=fixture["site_color"],
            validation_count=65,
            track_block_size=1,
            frame_block_size=7,
        )
        exact = direct_word_transfer(
            boundary=fixture["boundary"],
            ray_coefficients=fixture["ray_coefficients"],
            words=fixture["words"],
            site_density=fixture["site_density"],
            site_color=fixture["site_color"],
            times=atlas.transfer_atlas.node_times,
            near=fixture["near"],
            far=fixture["far"],
        )
        compiled_nodes = evaluate_transfer_atlas(
            atlas.transfer_atlas,
            atlas.transfer_atlas.node_times,
        )
        torch.testing.assert_close(compiled_nodes, exact, atol=3.0e-14, rtol=3.0e-13)
        self.assertLess(sampled_error, 2.0e-8)


if __name__ == "__main__":
    unittest.main()
