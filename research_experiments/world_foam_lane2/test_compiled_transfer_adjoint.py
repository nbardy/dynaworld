from __future__ import annotations

import unittest

import torch
from compiled_transfer_adjoint import (
    DTYPE,
    check_power_word_adjacency,
    check_supplied_word_ordering,
    compile_transfer_atlas,
    compiled_memory_accounting,
    compiled_power_cell_mse_vjp,
    compiled_transfer_mse_vjp,
    direct_word_render,
    evaluate_transfer_atlas,
    factorized_depth_coefficients,
    factorized_depth_coefficients_vjp,
    make_stable_cell_word,
    power_boundary_parameters,
    power_boundary_parameters_vjp,
    sampled_transfer_error,
    streamed_word_mse_vjp,
    track_blocked_compiled_transfer_mse_vjp,
)


def _fixture() -> dict[str, object]:
    boundary = torch.tensor(
        [
            [0.12, -0.05, 1.00, -0.08, -0.95],
            [-0.07, 0.08, 1.00, 0.04, -1.85],
        ],
        dtype=DTYPE,
    )
    ray_coefficients = torch.tensor(
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
    )
    words = (
        make_stable_cell_word([0, 1, 2], [-1, 0, 1], [0, 1, -2]),
        make_stable_cell_word([2, 1, 0], [-1, 0, 1], [0, 1, -2]),
    )
    return {
        "boundary": boundary,
        "ray_coefficients": ray_coefficients,
        "words": words,
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


class CompiledTransferAdjointTest(unittest.TestCase):
    def test_analytic_supplied_word_ordering_check_and_sampled_forward_accuracy(self) -> None:
        fixture = _fixture()
        ordering_check = check_supplied_word_ordering(
            boundary=fixture["boundary"],
            ray_coefficients=fixture["ray_coefficients"],
            words=fixture["words"],
            site_count=3,
            t_min=fixture["t_min"],
            t_max=fixture["t_max"],
            near=fixture["near"],
            far=fixture["far"],
        )
        self.assertTrue(ordering_check["passed"])
        self.assertGreater(ordering_check["minimum_relative_denominator_margin"], 0.8)
        self.assertGreater(ordering_check["minimum_physical_segment_length_lower_bound"], 0.4)
        self.assertGreater(ordering_check["minimum_fiber_speed"], 0.9)

        atlas = compile_transfer_atlas(
            boundary=fixture["boundary"],
            ray_coefficients=fixture["ray_coefficients"],
            words=fixture["words"],
            site_density=fixture["site_density"],
            site_color=fixture["site_color"],
            t_min=fixture["t_min"],
            t_max=fixture["t_max"],
            near=fixture["near"],
            far=fixture["far"],
            node_count=16,
        )
        self.assertLess(
            sampled_transfer_error(
                atlas,
                boundary=fixture["boundary"],
                ray_coefficients=fixture["ray_coefficients"],
                site_density=fixture["site_density"],
                site_color=fixture["site_color"],
            ),
            1.0e-10,
        )

    def test_shared_coefficient_vjp_matches_autograd(self) -> None:
        fixture = _fixture()
        times = torch.linspace(fixture["t_min"], fixture["t_max"], 19, dtype=DTYPE)
        target = direct_word_render(
            boundary=fixture["boundary"],
            ray_coefficients=fixture["ray_coefficients"],
            words=fixture["words"],
            site_density=fixture["site_density"] * 1.07,
            site_color=fixture["site_color"].roll(1, dims=0),
            times=times,
            background=fixture["background"],
            near=fixture["near"],
            far=fixture["far"],
        ).detach()

        boundary = fixture["boundary"].clone().requires_grad_(True)
        rays = fixture["ray_coefficients"].clone().requires_grad_(True)
        density = fixture["site_density"].clone().requires_grad_(True)
        color = fixture["site_color"].clone().requires_grad_(True)
        atlas = compile_transfer_atlas(
            boundary=boundary,
            ray_coefficients=rays,
            words=fixture["words"],
            site_density=density,
            site_color=color,
            t_min=fixture["t_min"],
            t_max=fixture["t_max"],
            near=fixture["near"],
            far=fixture["far"],
            node_count=12,
            differentiable=True,
        )
        prediction = evaluate_transfer_atlas(atlas, times, background=fixture["background"])
        autograd_loss = (prediction - target).square().mean()
        expected = torch.autograd.grad(autograd_loss, (boundary, rays, density, color))

        actual = compiled_transfer_mse_vjp(
            boundary=fixture["boundary"],
            ray_coefficients=fixture["ray_coefficients"],
            words=fixture["words"],
            site_density=fixture["site_density"],
            site_color=fixture["site_color"],
            times=times,
            targets=target,
            background=fixture["background"],
            t_min=fixture["t_min"],
            t_max=fixture["t_max"],
            near=fixture["near"],
            far=fixture["far"],
            node_count=12,
            frame_block_size=5,
            compute_ray_grad=True,
        )
        torch.testing.assert_close(actual.loss, autograd_loss.detach(), atol=2.0e-14, rtol=2.0e-13)
        for observed, reference in zip(
            (
                actual.grad_boundary,
                actual.grad_ray_coefficients,
                actual.grad_site_density,
                actual.grad_site_color,
            ),
            expected,
            strict=True,
        ):
            torch.testing.assert_close(observed, reference, atol=3.0e-12, rtol=3.0e-10)

    def test_exact_streamed_prefix_only_vjp_matches_direct_autograd(self) -> None:
        fixture = _fixture()
        times = torch.linspace(fixture["t_min"], fixture["t_max"], 17, dtype=DTYPE)
        targets = torch.full((2, 17, 3), 0.31, dtype=DTYPE)
        boundary = fixture["boundary"].clone().requires_grad_(True)
        rays = fixture["ray_coefficients"].clone().requires_grad_(True)
        density = fixture["site_density"].clone().requires_grad_(True)
        color = fixture["site_color"].clone().requires_grad_(True)
        prediction = direct_word_render(
            boundary=boundary,
            ray_coefficients=rays,
            words=fixture["words"],
            site_density=density,
            site_color=color,
            times=times,
            background=fixture["background"],
            near=fixture["near"],
            far=fixture["far"],
        )
        loss = (prediction - targets).square().mean()
        expected = torch.autograd.grad(loss, (boundary, rays, density, color))
        actual = streamed_word_mse_vjp(
            boundary=fixture["boundary"],
            ray_coefficients=fixture["ray_coefficients"],
            words=fixture["words"],
            site_density=fixture["site_density"],
            site_color=fixture["site_color"],
            times=times,
            targets=targets,
            background=fixture["background"],
            t_min=fixture["t_min"],
            t_max=fixture["t_max"],
            near=fixture["near"],
            far=fixture["far"],
            frame_block_size=4,
            return_predictions=True,
            compute_ray_grad=True,
        )
        torch.testing.assert_close(actual.loss, loss.detach(), atol=2.0e-15, rtol=2.0e-14)
        torch.testing.assert_close(actual.predictions, prediction.detach())
        for observed, reference in zip(
            (
                actual.grad_boundary,
                actual.grad_ray_coefficients,
                actual.grad_site_density,
                actual.grad_site_color,
            ),
            expected,
            strict=True,
        ):
            torch.testing.assert_close(observed, reference, atol=3.0e-12, rtol=3.0e-10)
        self.assertEqual(actual.grad_depth_coefficients.shape, (4, 4))

    def test_stream_block_size_changes_no_result(self) -> None:
        fixture = _fixture()
        times = torch.linspace(fixture["t_min"], fixture["t_max"], 23, dtype=DTYPE)
        targets = torch.full((2, 23, 3), 0.27, dtype=DTYPE)
        results = [
            compiled_transfer_mse_vjp(
                boundary=fixture["boundary"],
                ray_coefficients=fixture["ray_coefficients"],
                words=fixture["words"],
                site_density=fixture["site_density"],
                site_color=fixture["site_color"],
                times=times,
                targets=targets,
                background=fixture["background"],
                t_min=fixture["t_min"],
                t_max=fixture["t_max"],
                near=fixture["near"],
                far=fixture["far"],
                node_count=12,
                frame_block_size=block_size,
                return_predictions=True,
                compute_ray_grad=True,
            )
            for block_size in (1, 7, 23)
        ]
        for result in results[1:]:
            torch.testing.assert_close(result.loss, results[0].loss, atol=3.0e-16, rtol=3.0e-15)
            torch.testing.assert_close(result.predictions, results[0].predictions)
            torch.testing.assert_close(result.grad_boundary, results[0].grad_boundary, atol=2.0e-15, rtol=2.0e-14)
            torch.testing.assert_close(
                result.grad_ray_coefficients,
                results[0].grad_ray_coefficients,
                atol=2.0e-15,
                rtol=2.0e-14,
            )
            torch.testing.assert_close(result.grad_site_density, results[0].grad_site_density)
            torch.testing.assert_close(result.grad_site_color, results[0].grad_site_color)

    def test_compiled_transfer_forward_and_vjp_track_exact_replay_on_smooth_chart(self) -> None:
        fixture = _fixture()
        times = torch.linspace(fixture["t_min"], fixture["t_max"], 65, dtype=DTYPE)
        targets = torch.full((2, 65, 3), 0.29, dtype=DTYPE)
        exact = streamed_word_mse_vjp(
            boundary=fixture["boundary"],
            ray_coefficients=fixture["ray_coefficients"],
            words=fixture["words"],
            site_density=fixture["site_density"],
            site_color=fixture["site_color"],
            times=times,
            targets=targets,
            background=fixture["background"],
            t_min=fixture["t_min"],
            t_max=fixture["t_max"],
            near=fixture["near"],
            far=fixture["far"],
            frame_block_size=8,
            return_predictions=True,
            compute_ray_grad=True,
        )
        compiled = compiled_transfer_mse_vjp(
            boundary=fixture["boundary"],
            ray_coefficients=fixture["ray_coefficients"],
            words=fixture["words"],
            site_density=fixture["site_density"],
            site_color=fixture["site_color"],
            times=times,
            targets=targets,
            background=fixture["background"],
            t_min=fixture["t_min"],
            t_max=fixture["t_max"],
            near=fixture["near"],
            far=fixture["far"],
            node_count=16,
            frame_block_size=8,
            return_predictions=True,
            compute_ray_grad=True,
        )
        for observed, reference in (
            (compiled.predictions, exact.predictions),
            (compiled.grad_boundary, exact.grad_boundary),
            (compiled.grad_ray_coefficients, exact.grad_ray_coefficients),
            (compiled.grad_site_density, exact.grad_site_density),
            (compiled.grad_site_color, exact.grad_site_color),
        ):
            torch.testing.assert_close(observed, reference, atol=2.0e-10, rtol=2.0e-9)

    def test_structural_and_reverse_payload_do_not_scale_with_frame_count(self) -> None:
        fixture = _fixture()
        atlas = compile_transfer_atlas(
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
        reports = [
            compiled_memory_accounting(
                atlas=atlas,
                boundary=fixture["boundary"],
                ray_coefficients=fixture["ray_coefficients"],
                site_density=fixture["site_density"],
                site_color=fixture["site_color"],
                frame_count=frame_count,
                frame_block_size=8,
                return_predictions=False,
            )
            for frame_count in (8, 128)
        ]
        for key in (
            "world_parameter_bytes",
            "camera_program_bytes",
            "atlas_structural_bytes",
            "reverse_interaction_bytes",
        ):
            self.assertEqual(reports[0][key], reports[1][key])
        self.assertEqual(reports[1]["sample_io_bytes"], 16 * reports[0]["sample_io_bytes"])
        self.assertEqual(atlas.coefficients.shape, (2, 12, 4))
        self.assertEqual(atlas.depth_coefficient_incidence.shape, (4, 2))

    def test_track_blocked_compiled_vjp_matches_full_reference(self) -> None:
        fixture = _fixture()
        times = torch.linspace(fixture["t_min"], fixture["t_max"], 21, dtype=DTYPE)
        targets = torch.full((2, 21, 3), 0.24, dtype=DTYPE)
        common = {
            "boundary": fixture["boundary"],
            "ray_coefficients": fixture["ray_coefficients"],
            "words": fixture["words"],
            "site_density": fixture["site_density"],
            "site_color": fixture["site_color"],
            "times": times,
            "targets": targets,
            "background": fixture["background"],
            "t_min": fixture["t_min"],
            "t_max": fixture["t_max"],
            "near": fixture["near"],
            "far": fixture["far"],
            "node_count": 12,
            "frame_block_size": 5,
        }
        full = compiled_transfer_mse_vjp(**common, compute_ray_grad=True)
        blocked = track_blocked_compiled_transfer_mse_vjp(**common, track_block_size=1, compute_ray_grad=True)
        torch.testing.assert_close(blocked.loss, full.loss, atol=3.0e-16, rtol=3.0e-15)
        for observed, reference in (
            (blocked.grad_boundary, full.grad_boundary),
            (blocked.grad_ray_coefficients, full.grad_ray_coefficients),
            (blocked.grad_site_density, full.grad_site_density),
            (blocked.grad_site_color, full.grad_site_color),
        ):
            torch.testing.assert_close(observed, reference, atol=3.0e-12, rtol=3.0e-10)
        self.assertEqual(blocked.accounting["track_block_size"], 1)
        self.assertLess(
            blocked.accounting["peak_block_atlas_bytes"],
            full.accounting["atlas_structural_bytes"],
        )

    def test_factorized_coefficient_vjp_matches_autograd(self) -> None:
        fixture = _fixture()
        boundary = fixture["boundary"].clone().requires_grad_(True)
        rays = fixture["ray_coefficients"].clone().requires_grad_(True)
        coefficients = factorized_depth_coefficients(boundary, rays)
        cotangent = torch.linspace(-0.4, 0.7, coefficients.numel(), dtype=DTYPE).reshape_as(coefficients)
        expected = torch.autograd.grad((coefficients * cotangent).sum(), (boundary, rays))
        actual = factorized_depth_coefficients_vjp(boundary.detach(), rays.detach(), cotangent)
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])

    def test_sparse_power_boundary_scatter_matches_autograd(self) -> None:
        sites = torch.tensor(
            [
                [0.0, -0.2, 0.4, -0.1, 0.05],
                [0.1, 0.3, 1.2, 0.2, -0.02],
                [-0.2, 0.1, 2.0, 0.5, 0.11],
                [0.4, -0.1, 2.6, 0.8, -0.07],
            ],
            dtype=DTYPE,
            requires_grad=True,
        )
        pairs = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.int64)
        boundary = power_boundary_parameters(sites, pairs)
        cotangent = torch.linspace(-0.3, 0.5, boundary.numel(), dtype=DTYPE).reshape_as(boundary)
        expected = torch.autograd.grad((boundary * cotangent).sum(), sites)[0]
        actual = power_boundary_parameters_vjp(sites.detach(), pairs, cotangent)
        torch.testing.assert_close(actual, expected)

    def test_physical_transfer_is_invariant_to_affine_depth_rescaling(self) -> None:
        fixture = _fixture()
        times = torch.linspace(fixture["t_min"], fixture["t_max"], 31, dtype=DTYPE)
        reference = direct_word_render(
            boundary=fixture["boundary"],
            ray_coefficients=fixture["ray_coefficients"],
            words=fixture["words"],
            site_density=fixture["site_density"],
            site_color=fixture["site_color"],
            times=times,
            background=fixture["background"],
            near=fixture["near"],
            far=fixture["far"],
        )
        scale = 7.0
        rescaled_rays = fixture["ray_coefficients"].clone()
        rescaled_rays[:, 6:12] /= scale
        rescaled = direct_word_render(
            boundary=fixture["boundary"],
            ray_coefficients=rescaled_rays,
            words=fixture["words"],
            site_density=fixture["site_density"],
            site_color=fixture["site_color"],
            times=times,
            background=fixture["background"],
            near=scale * fixture["near"],
            far=scale * fixture["far"],
        )
        torch.testing.assert_close(rescaled, reference, atol=2.0e-15, rtol=2.0e-14)

    def test_manual_vjp_does_not_retain_an_autograd_graph(self) -> None:
        fixture = _fixture()
        result = streamed_word_mse_vjp(
            boundary=fixture["boundary"].clone().requires_grad_(True),
            ray_coefficients=fixture["ray_coefficients"].clone().requires_grad_(True),
            words=fixture["words"],
            site_density=fixture["site_density"].clone().requires_grad_(True),
            site_color=fixture["site_color"].clone().requires_grad_(True),
            times=torch.linspace(fixture["t_min"], fixture["t_max"], 5, dtype=DTYPE),
            targets=torch.full((2, 5, 3), 0.2, dtype=DTYPE),
            background=fixture["background"],
            t_min=fixture["t_min"],
            t_max=fixture["t_max"],
            near=fixture["near"],
            far=fixture["far"],
            return_predictions=False,
            compute_ray_grad=True,
        )
        for tensor in (
            result.loss,
            result.grad_boundary,
            result.grad_ray_coefficients,
            result.grad_site_density,
            result.grad_site_color,
        ):
            self.assertFalse(tensor.requires_grad)
            self.assertIsNone(tensor.grad_fn)

    def test_atlas_and_chunked_vjp_fail_closed_outside_chart(self) -> None:
        fixture = _fixture()
        atlas = compile_transfer_atlas(
            boundary=fixture["boundary"],
            ray_coefficients=fixture["ray_coefficients"],
            words=fixture["words"],
            site_density=fixture["site_density"],
            site_color=fixture["site_color"],
            t_min=fixture["t_min"],
            t_max=fixture["t_max"],
            near=fixture["near"],
            far=fixture["far"],
            node_count=8,
        )
        with self.assertRaisesRegex(ValueError, "outside the checked chart"):
            evaluate_transfer_atlas(atlas, [fixture["t_max"] + 0.01], background=fixture["background"])
        with self.assertRaisesRegex(ValueError, "outside the checked chart"):
            streamed_word_mse_vjp(
                boundary=fixture["boundary"],
                ray_coefficients=fixture["ray_coefficients"],
                words=fixture["words"],
                site_density=fixture["site_density"],
                site_color=fixture["site_color"],
                times=[fixture["t_max"] + 0.01],
                targets=torch.zeros((2, 1, 3), dtype=DTYPE),
                background=fixture["background"],
                t_min=fixture["t_min"],
                t_max=fixture["t_max"],
                near=fixture["near"],
                far=fixture["far"],
            )

    def test_ordering_check_rejects_interior_crossing_at_homogeneous_scales(self) -> None:
        left = [0.7825342496411134, 0.6923451248082364, 1.0, 0.6225289126535452]
        right = [0.9196658082983348, 0.5049172647983071, 1.0, -0.3008824897997976]
        base_boundary = torch.tensor(
            [
                [left[2], 0.0, left[3], -left[1], -left[0]],
                [right[2], 0.0, right[3], -right[1], -right[0]],
            ],
            dtype=DTYPE,
        )
        rays = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
            dtype=DTYPE,
        )
        words = (make_stable_cell_word([0, 1, 2], [-1, 0, 1], [0, 1, -2]),)
        for scale in (1.0, 1.0e-4):
            with self.assertRaisesRegex(ValueError, "loses strict endpoint order"):
                check_supplied_word_ordering(
                    boundary=base_boundary * scale,
                    ray_coefficients=rays,
                    words=words,
                    site_count=3,
                    t_min=-1.0,
                    t_max=1.0,
                    near=0.05,
                    far=3.0,
                )

    def test_fixed_rank_atlas_exposes_hard_chart_error(self) -> None:
        boundary = torch.tensor([[0.0, 0.0, 1.0, -0.9, -1.0]], dtype=DTYPE)
        rays = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
            dtype=DTYPE,
        )
        words = (make_stable_cell_word([0, 1], [-1, 0], [0, -2]),)
        density = torch.tensor([50.0, 0.0], dtype=DTYPE)
        color = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=DTYPE)
        atlas = compile_transfer_atlas(
            boundary=boundary,
            ray_coefficients=rays,
            words=words,
            site_density=density,
            site_color=color,
            t_min=-1.0,
            t_max=1.0,
            near=0.05,
            far=2.0,
            node_count=16,
        )
        error = sampled_transfer_error(
            atlas,
            boundary=boundary,
            ray_coefficients=rays,
            site_density=density,
            site_color=color,
        )
        self.assertGreater(error, 2.0e-3)
        with self.assertRaisesRegex(ValueError, "exceeds its sampled forward-error gate"):
            compiled_transfer_mse_vjp(
                boundary=boundary,
                ray_coefficients=rays,
                words=words,
                site_density=density,
                site_color=color,
                times=torch.linspace(-1.0, 1.0, 9, dtype=DTYPE),
                targets=torch.zeros((1, 9, 3), dtype=DTYPE),
                background=[0.0, 0.0, 0.0],
                t_min=-1.0,
                t_max=1.0,
                near=0.05,
                far=2.0,
                node_count=16,
                sampled_error_tolerance=1.0e-3,
            )

    def test_composed_power_cell_vjp_and_oriented_adjacency(self) -> None:
        sites = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 4.0, 0.0, 0.0],
            ],
            dtype=DTYPE,
        )
        pairs = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64)
        rays = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=DTYPE)
        word = make_stable_cell_word([0, 1, 2], [-1, 0, 1], [0, 1, -2])
        density = torch.tensor([0.3, 0.5, 0.8], dtype=DTYPE)
        color = torch.tensor([[0.9, 0.1, 0.2], [0.2, 0.8, 0.3], [0.1, 0.2, 0.9]], dtype=DTYPE)
        times = torch.linspace(-0.5, 0.5, 7, dtype=DTYPE)
        targets = torch.full((1, 7, 3), 0.25, dtype=DTYPE)

        sites_autograd = sites.clone().requires_grad_(True)
        prediction = direct_word_render(
            boundary=power_boundary_parameters(sites_autograd, pairs),
            ray_coefficients=rays,
            words=(word,),
            site_density=density,
            site_color=color,
            times=times,
            background=[0.0, 0.0, 0.0],
            near=0.1,
            far=4.5,
        )
        expected = torch.autograd.grad((prediction - targets).square().mean(), sites_autograd)[0]
        actual = compiled_power_cell_mse_vjp(
            sites=sites,
            boundary_pairs=pairs,
            ray_coefficients=rays,
            words=(word,),
            site_density=density,
            site_color=color,
            times=times,
            targets=targets,
            background=[0.0, 0.0, 0.0],
            t_min=-0.5,
            t_max=0.5,
            near=0.1,
            far=4.5,
            node_count=6,
        )
        torch.testing.assert_close(actual.grad_site_geometry, expected, atol=3.0e-12, rtol=3.0e-10)
        self.assertIsNone(actual.transfer.grad_ray_coefficients)
        self.assertEqual(actual.accounting["track_block_size"], 1)

        reversed_word = make_stable_cell_word([1, 0, 2], [-1, 0, 1], [0, 1, -2])
        with self.assertRaisesRegex(ValueError, "owner transition"):
            check_power_word_adjacency(
                sites=sites,
                boundary_pairs=pairs,
                ray_coefficients=rays,
                words=(reversed_word,),
                t_min=-0.5,
                t_max=0.5,
            )

    def test_certificate_fails_closed_on_a_denominator_event(self) -> None:
        fixture = _fixture()
        bad_rays = fixture["ray_coefficients"].clone()
        bad_rays[0, 6:9] = torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE)
        bad_rays[0, 9:12] = torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE)
        with self.assertRaisesRegex(ValueError, "unsafe depth denominator"):
            check_supplied_word_ordering(
                boundary=fixture["boundary"],
                ray_coefficients=bad_rays,
                words=fixture["words"],
                site_count=3,
                t_min=-0.8,
                t_max=0.9,
                near=fixture["near"],
                far=fixture["far"],
            )


if __name__ == "__main__":
    unittest.main()
