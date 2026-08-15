from __future__ import annotations

import unittest

import torch
from compiled_transfer_adjoint import DTYPE, direct_word_render, make_stable_cell_word, streamed_word_mse_vjp
from exact_sparse_incidence_oracle import (
    direct_boundary_vjp_from_endpoint_cotangents,
    reduce_endpoint_cotangents_via_sparse_incidence,
    sample_word_endpoint_cotangents,
    sparse_factorized_depth_coefficients,
    sparse_incidence_atomic_accounting,
)


def _fixture() -> dict[str, object]:
    boundary = torch.tensor(
        [
            [0.12, -0.05, 1.00, -0.08, -0.95],
            [-0.07, 0.08, 1.00, 0.04, -1.85],
        ],
        dtype=DTYPE,
    )
    rays = torch.tensor(
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
    return {
        "boundary": boundary,
        "rays": rays,
        "words": (
            make_stable_cell_word([0, 1, 2], [-1, 0, 1], [0, 1, -2]),
            make_stable_cell_word([2, 1, 0], [-1, 0, 1], [0, 1, -2]),
        ),
        "density": torch.tensor([0.45, 0.82, 0.31], dtype=DTYPE),
        "color": torch.tensor(
            [[0.91, 0.16, 0.08], [0.10, 0.66, 0.93], [0.38, 0.84, 0.25]],
            dtype=DTYPE,
        ),
        "background": torch.tensor([0.03, 0.04, 0.06], dtype=DTYPE),
        "near": 0.1,
        "far": 3.0,
        "t_min": -0.8,
        "t_max": 0.9,
    }


def _incidence_program(
    boundary: torch.Tensor,
    rays: torch.Tensor,
    words: tuple,
) -> tuple[torch.Tensor, tuple[dict[int, torch.Tensor], ...], dict[tuple[int, int], int]]:
    rows: list[tuple[int, int]] = []
    for track_id, word in enumerate(words):
        cut_ids = sorted(
            {
                int(cut_id)
                for cut_id in torch.cat((word.left_cut_ids, word.right_cut_ids)).tolist()
                if int(cut_id) >= 0
            }
        )
        rows.extend((track_id, cut_id) for cut_id in cut_ids)
    incidence = torch.tensor(rows, dtype=torch.int64)
    coefficients = sparse_factorized_depth_coefficients(boundary, rays, incidence)
    maps: list[dict[int, torch.Tensor]] = [dict() for _ in range(int(rays.shape[0]))]
    index: dict[tuple[int, int], int] = {}
    for incidence_id, (track_id, boundary_id) in enumerate(rows):
        maps[track_id][boundary_id] = coefficients[incidence_id]
        index[(track_id, boundary_id)] = incidence_id
    return incidence, tuple(maps), index


def _collect_exact_mse_events(
    *,
    boundary: torch.Tensor,
    rays: torch.Tensor,
    words: tuple,
    density: torch.Tensor,
    color: torch.Tensor,
    times: torch.Tensor,
    targets: torch.Tensor,
    background: torch.Tensor,
    near: float,
    far: float,
    compute_ray_grad: bool,
) -> dict[str, torch.Tensor | tuple | dict]:
    incidence, coefficient_maps, incidence_index = _incidence_program(boundary, rays, words)
    prediction = direct_word_render(
        boundary=boundary,
        ray_coefficients=rays,
        words=words,
        site_density=density,
        site_color=color,
        times=times,
        background=background,
        near=near,
        far=far,
    )
    residual = prediction - targets
    inv_element_count = 1.0 / float(targets.numel())
    grad_density = torch.zeros_like(density)
    grad_color = torch.zeros_like(color)
    grad_ray_metric = torch.zeros_like(rays) if compute_ray_grad else None
    event_ids: list[int] = []
    event_times: list[torch.Tensor] = []
    event_bars: list[torch.Tensor] = []
    sample_transfers: list[torch.Tensor] = []
    for track_id, word in enumerate(words):
        for frame_id, time in enumerate(times):
            grad_prediction = 2.0 * residual[track_id, frame_id] * inv_element_count
            grad_transfer = torch.cat(
                (
                    torch.dot(grad_prediction, background).reshape(1),
                    grad_prediction,
                )
            )
            sample = sample_word_endpoint_cotangents(
                word=word,
                cut_coefficients=coefficient_maps[track_id],
                ray_coefficient=rays[track_id],
                time=time,
                site_density=density,
                site_color=color,
                grad_transfer=grad_transfer,
                near=near,
                far=far,
                compute_ray_grad=compute_ray_grad,
            )
            sample_transfers.append(sample.transfer)
            grad_density += sample.grad_site_density
            grad_color += sample.grad_site_color
            if grad_ray_metric is not None and sample.grad_ray_metric is not None:
                grad_ray_metric[track_id] += sample.grad_ray_metric
            for cut_id, depth_bar in zip(
                sample.finite_cut_ids.tolist(),
                sample.depth_coordinate_cotangents,
                strict=True,
            ):
                event_ids.append(incidence_index[(track_id, int(cut_id))])
                event_times.append(time)
                event_bars.append(depth_bar)
    return {
        "incidence": incidence,
        "event_ids": torch.tensor(event_ids, dtype=torch.int64),
        "event_times": torch.stack(event_times),
        "event_bars": torch.stack(event_bars),
        "grad_density": grad_density,
        "grad_color": grad_color,
        "grad_ray_metric": grad_ray_metric,
        "prediction": prediction,
        "sample_transfers": tuple(sample_transfers),
    }


class ExactSparseIncidenceOracleTest(unittest.TestCase):
    def test_physical_endpoint_events_match_direct_boundary_and_existing_exact_vjp(self) -> None:
        fixture = _fixture()
        times = torch.linspace(fixture["t_min"], fixture["t_max"], 11, dtype=DTYPE)
        targets = torch.linspace(0.13, 0.73, 2 * 11 * 3, dtype=DTYPE).reshape(2, 11, 3)
        events = _collect_exact_mse_events(
            boundary=fixture["boundary"],
            rays=fixture["rays"],
            words=fixture["words"],
            density=fixture["density"],
            color=fixture["color"],
            times=times,
            targets=targets,
            background=fixture["background"],
            near=fixture["near"],
            far=fixture["far"],
            compute_ray_grad=True,
        )
        sparse = reduce_endpoint_cotangents_via_sparse_incidence(
            boundary=fixture["boundary"],
            ray_coefficients=fixture["rays"],
            incidence=events["incidence"],
            event_incidence_ids=events["event_ids"],
            event_times=events["event_times"],
            event_depth_coordinate_cotangents=events["event_bars"],
            event_block_size=7,
            compute_ray_grad=True,
            grad_ray_metric=events["grad_ray_metric"],
        )
        direct = direct_boundary_vjp_from_endpoint_cotangents(
            boundary=fixture["boundary"],
            ray_coefficients=fixture["rays"],
            incidence=events["incidence"],
            event_incidence_ids=events["event_ids"],
            event_times=events["event_times"],
            event_depth_coordinate_cotangents=events["event_bars"],
            event_block_size=5,
            compute_ray_grad=True,
            grad_ray_metric=events["grad_ray_metric"],
        )
        existing = streamed_word_mse_vjp(
            boundary=fixture["boundary"],
            ray_coefficients=fixture["rays"],
            words=fixture["words"],
            site_density=fixture["density"],
            site_color=fixture["color"],
            times=times,
            targets=targets,
            background=fixture["background"],
            t_min=fixture["t_min"],
            t_max=fixture["t_max"],
            near=fixture["near"],
            far=fixture["far"],
            compute_ray_grad=True,
        )

        torch.testing.assert_close(sparse.grad_boundary, direct.grad_boundary, atol=2.0e-14, rtol=2.0e-13)
        torch.testing.assert_close(
            sparse.grad_ray_coefficients,
            direct.grad_ray_coefficients,
            atol=2.0e-14,
            rtol=2.0e-13,
        )
        for observed, reference in (
            (sparse.grad_boundary, existing.grad_boundary),
            (sparse.grad_ray_coefficients, existing.grad_ray_coefficients),
            (events["grad_density"], existing.grad_site_density),
            (events["grad_color"], existing.grad_site_color),
        ):
            torch.testing.assert_close(observed, reference, atol=3.0e-12, rtol=3.0e-10)

        # Every internal cut is both the right endpoint of one run and the
        # left endpoint of the next run at every sample.
        event_counts = torch.bincount(events["event_ids"], minlength=events["incidence"].shape[0])
        torch.testing.assert_close(event_counts, torch.full((4,), 2 * times.numel(), dtype=torch.int64))

    def test_repeated_incidence_events_match_autograd_and_are_block_invariant(self) -> None:
        boundary = torch.tensor(
            [[0.2, -0.1, 1.1, 0.07, -0.8], [-0.3, 0.2, 0.9, -0.05, -1.7]],
            dtype=DTYPE,
        )
        rays = torch.tensor(
            [
                [0.1, -0.2, 0.0, 0.02, 0.01, -0.03, 0.05, -0.02, 1.0, 0.01, 0.02, 0.04],
                [-0.1, 0.1, 0.2, 0.01, -0.02, 0.02, -0.04, 0.03, 0.93, 0.03, -0.01, 0.02],
            ],
            dtype=DTYPE,
        )
        # Boundary zero is shared across two tracks; event ids deliberately
        # repeat and cross every tested block boundary.
        incidence = torch.tensor([[0, 0], [0, 1], [1, 0]], dtype=torch.int64)
        event_ids = torch.tensor([0, 0, 2, 1, 0, 2, 2, 1, 0, 2, 1, 1, 0], dtype=torch.int64)
        times = torch.tensor([-0.7, -0.2, 0.4, 0.1, 0.8, -0.5, 0.2, 0.7, -0.1, 0.55, -0.6, 0.3, 0.0], dtype=DTYPE)
        depth_bars = torch.tensor([0.3, -0.8, 0.2, 0.7, 0.5, -0.4, 0.9, -0.1, -0.6, 0.25, 0.45, -0.35, 0.15], dtype=DTYPE)
        ray_metric = torch.linspace(-0.03, 0.04, rays.numel(), dtype=DTYPE).reshape_as(rays)

        reductions = [
            reduce_endpoint_cotangents_via_sparse_incidence(
                boundary=boundary,
                ray_coefficients=rays,
                incidence=incidence,
                event_incidence_ids=event_ids,
                event_times=times,
                event_depth_coordinate_cotangents=depth_bars,
                event_block_size=block_size,
                compute_ray_grad=True,
                grad_ray_metric=ray_metric,
            )
            for block_size in (1, 4, event_ids.numel())
        ]
        for result in reductions[1:]:
            torch.testing.assert_close(result.grad_depth_coefficients, reductions[0].grad_depth_coefficients)
            torch.testing.assert_close(result.grad_boundary, reductions[0].grad_boundary, atol=3.0e-15, rtol=3.0e-14)
            torch.testing.assert_close(
                result.grad_ray_coefficients,
                reductions[0].grad_ray_coefficients,
                atol=3.0e-15,
                rtol=3.0e-14,
            )

        boundary_ad = boundary.clone().requires_grad_(True)
        rays_ad = rays.clone().requires_grad_(True)
        pair_rows = incidence[event_ids]
        normal = boundary_ad[pair_rows[:, 1], :3]
        active_rays = rays_ad[pair_rows[:, 0]]
        origin = active_rays[:, 0:3] + times.unsqueeze(1) * active_rays[:, 3:6]
        direction = active_rays[:, 6:9] + times.unsqueeze(1) * active_rays[:, 9:12]
        depth = -(
            (normal * origin).sum(dim=1)
            + boundary_ad[pair_rows[:, 1], 3] * times
            + boundary_ad[pair_rows[:, 1], 4]
        ) / (normal * direction).sum(dim=1)
        objective = (depth * depth_bars).sum() + (rays_ad * ray_metric).sum()
        expected = torch.autograd.grad(objective, (boundary_ad, rays_ad))
        torch.testing.assert_close(reductions[0].grad_boundary, expected[0], atol=3.0e-14, rtol=3.0e-13)
        torch.testing.assert_close(
            reductions[0].grad_ray_coefficients,
            expected[1],
            atol=3.0e-14,
            rtol=3.0e-13,
        )

        with self.assertRaisesRegex(ValueError, "incidence rows must be unique"):
            sparse_factorized_depth_coefficients(boundary, rays, torch.tensor([[0, 0], [0, 0]]))

    def test_ray_depth_gauge_rescaling_preserves_physical_world_vjp(self) -> None:
        fixture = _fixture()
        times = torch.linspace(fixture["t_min"], fixture["t_max"], 9, dtype=DTYPE)
        targets = torch.full((2, 9, 3), 0.37, dtype=DTYPE)
        reference_events = _collect_exact_mse_events(
            boundary=fixture["boundary"],
            rays=fixture["rays"],
            words=fixture["words"],
            density=fixture["density"],
            color=fixture["color"],
            times=times,
            targets=targets,
            background=fixture["background"],
            near=fixture["near"],
            far=fixture["far"],
            compute_ray_grad=False,
        )
        reference = reduce_endpoint_cotangents_via_sparse_incidence(
            boundary=fixture["boundary"],
            ray_coefficients=fixture["rays"],
            incidence=reference_events["incidence"],
            event_incidence_ids=reference_events["event_ids"],
            event_times=reference_events["event_times"],
            event_depth_coordinate_cotangents=reference_events["event_bars"],
            event_block_size=3,
            compute_ray_grad=False,
        )

        scale = 13.0
        scaled_rays = fixture["rays"].clone()
        scaled_rays[:, 6:12] /= scale
        scaled_events = _collect_exact_mse_events(
            boundary=fixture["boundary"],
            rays=scaled_rays,
            words=fixture["words"],
            density=fixture["density"],
            color=fixture["color"],
            times=times,
            targets=targets,
            background=fixture["background"],
            near=scale * fixture["near"],
            far=scale * fixture["far"],
            compute_ray_grad=False,
        )
        scaled = reduce_endpoint_cotangents_via_sparse_incidence(
            boundary=fixture["boundary"],
            ray_coefficients=scaled_rays,
            incidence=scaled_events["incidence"],
            event_incidence_ids=scaled_events["event_ids"],
            event_times=scaled_events["event_times"],
            event_depth_coordinate_cotangents=scaled_events["event_bars"],
            event_block_size=17,
            compute_ray_grad=False,
            # No-ray mode must not validate, copy, or retain this optional term.
            grad_ray_metric=torch.full_like(scaled_rays, float("nan")),
        )
        torch.testing.assert_close(scaled_events["prediction"], reference_events["prediction"], atol=2.0e-15, rtol=2.0e-14)
        torch.testing.assert_close(scaled_events["event_bars"], reference_events["event_bars"] / scale)
        torch.testing.assert_close(scaled.grad_boundary, reference.grad_boundary, atol=3.0e-14, rtol=3.0e-13)
        torch.testing.assert_close(scaled_events["grad_density"], reference_events["grad_density"])
        torch.testing.assert_close(scaled_events["grad_color"], reference_events["grad_color"])
        self.assertIsNone(reference.grad_ray_coefficients)
        self.assertIsNone(scaled.grad_ray_coefficients)
        self.assertEqual(reference.accounting["ray_gradient_bytes"], 0)

        with_ray = reduce_endpoint_cotangents_via_sparse_incidence(
            boundary=fixture["boundary"],
            ray_coefficients=fixture["rays"],
            incidence=reference_events["incidence"],
            event_incidence_ids=reference_events["event_ids"],
            event_times=reference_events["event_times"],
            event_depth_coordinate_cotangents=reference_events["event_bars"],
            compute_ray_grad=True,
        )
        torch.testing.assert_close(with_ray.grad_boundary, reference.grad_boundary)
        self.assertIsNotNone(with_ray.grad_ray_coefficients)

    def test_atomic_and_byte_model_states_the_reuse_break_even(self) -> None:
        report = sparse_incidence_atomic_accounting(
            finite_endpoint_event_count=1280,
            incidence_count=32,
            scalar_bytes=4,
        )
        self.assertEqual(report["direct_boundary_scalar_atomic_adds"], 6400)
        self.assertEqual(report["sparse_coefficient_scalar_atomic_adds"], 5120)
        self.assertEqual(report["sparse_boundary_finalize_scalar_atomic_adds"], 160)
        self.assertEqual(report["sparse_total_scalar_atomic_adds"], 5280)
        self.assertEqual(report["direct_minimum_atomic_payload_bytes"], 25_600)
        self.assertEqual(report["sparse_minimum_atomic_payload_bytes"], 21_120)
        self.assertEqual(report["modeled_sparse_incidence_adjoint_bytes"], 512)
        self.assertTrue(report["sparse_wins_boundary_atomic_count"])

        # Do not oversell sparse reduction: without temporal/endpoint reuse its
        # finalize pass costs more atomics than direct five-scalar scattering.
        low_reuse = sparse_incidence_atomic_accounting(
            finite_endpoint_event_count=4,
            incidence_count=4,
        )
        self.assertFalse(low_reuse["sparse_wins_boundary_atomic_count"])
        self.assertGreater(
            low_reuse["sparse_total_scalar_atomic_adds"],
            low_reuse["direct_boundary_scalar_atomic_adds"],
        )


if __name__ == "__main__":
    unittest.main()
