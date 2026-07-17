from __future__ import annotations

import unittest
from unittest import mock
from types import SimpleNamespace

import torch

import probe_gate4_affine_candidate_csr_capacity as probe


def _row(frame_count: int, *, storage: int, candidates: int, max_candidates: int = 224) -> dict[str, object]:
    return {
        "status": "ok",
        "frame_count": frame_count,
        "storage_bytes": storage,
        "candidate_count": candidates,
        "candidate_replay_iterations": candidates * frame_count,
        "direct_boundary_iterations": 1_000 * frame_count,
        "compiled_boundary_tests": 2_000,
        "max_candidates_per_row": max_candidates,
        "max_origin_residual": 0.0,
        "max_direction_residual": 0.0,
    }


class Gate4AffineCandidateCSRCapacityProbeTests(unittest.TestCase):
    def test_storage_breakdown_matches_train_eval_mps_layout(self) -> None:
        tape = SimpleNamespace(
            row_index=torch.zeros((8192,), dtype=torch.int32),
            row_offsets=torch.zeros((8193,), dtype=torch.int32),
            candidate_count=1_339_555,
            ray_coeff=torch.zeros((8192, 12), dtype=torch.float32),
            frame_t=torch.zeros((2,), dtype=torch.float32),
        )

        storage = probe.candidate_csr_storage_breakdown(tape=tape, site_count=24)

        self.assertEqual(storage["affine_candidate_depth_num_f32"], 10_716_440)
        self.assertEqual(storage["affine_candidate_depth_den_f16"], 5_358_220)
        self.assertEqual(storage["affine_ray_f32"], 393_216)
        self.assertEqual(storage["total_bytes"], 16_533_904)

    def test_summarize_rows_accepts_flat_topology_across_frame_growth(self) -> None:
        rows = [
            _row(2, storage=1_048_324, candidates=84_930),
            _row(4, storage=1_044_480, candidates=84_609),
            _row(8, storage=1_039_540, candidates=84_196),
            _row(16, storage=1_039_920, candidates=84_225),
        ]

        summary = probe.summarize_rows(
            rows,
            max_storage_scale=1.10,
            max_candidate_scale=1.10,
            max_candidates_per_row=256,
            max_fit_residual=1.0e-5,
        )

        self.assertEqual(summary["status"], "ok", summary["failures"])
        self.assertTrue(summary["scale_gate_required"])
        self.assertLess(summary["candidate_count_scale_first_to_last"], 1.10)
        self.assertLess(summary["storage_scale_first_to_last"], 1.10)
        self.assertLess(
            summary["compiled_boundary_test_scale_first_to_last"],
            summary["direct_boundary_iteration_scale_first_to_last"],
        )

    def test_summarize_rows_rejects_candidate_growth_or_cap_violation(self) -> None:
        rows = [
            _row(2, storage=1_000, candidates=1_000),
            _row(16, storage=1_050, candidates=3_000, max_candidates=300),
        ]

        summary = probe.summarize_rows(
            rows,
            max_storage_scale=1.10,
            max_candidate_scale=1.10,
            max_candidates_per_row=256,
            max_fit_residual=1.0e-5,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("candidate_count_scale_within_limit", summary["failures"])
        self.assertIn("candidate_rows_under_cap", summary["failures"])

    def test_run_probe_threads_site_initialization_to_each_profile_row(self) -> None:
        captured_initializations: list[str] = []

        def fake_profile_frame_count(**kwargs: object) -> dict[str, object]:
            captured_initializations.append(str(kwargs["site_initialization"]))
            return {
                "status": "ok",
                "frame_count": int(kwargs["frame_count"]),
                "storage_bytes": 1_000,
                "candidate_count": 1_000,
                "candidate_replay_iterations": 1_000,
                "direct_boundary_iterations": 10_000,
                "compiled_boundary_tests": 1_000,
                "max_candidates_per_row": 128,
                "max_origin_residual": 0.0,
                "max_direction_residual": 0.0,
                "site_initialization": str(kwargs["site_initialization"]),
            }

        args = SimpleNamespace(
            frame_counts="2,4",
            config=probe.DEFAULT_CONFIG,
            render_size=8,
            site_count=4,
            site_initialization="legacy_pixel_mean",
            near=0.1,
            far=6.0,
            density=10.0,
            invalid_epsilon=1.0e-6,
            gate4_residual_depth_padding=0.001,
            gate4_time_slabs=1,
            origin_velocity_x=0.08,
            origin_velocity_y=0.0,
            origin_velocity_z=0.02,
            direction_velocity_x=0.02,
            direction_velocity_y=0.0,
            direction_velocity_z=0.0,
            sample_validation="skip",
            repeat_loaded_frames=False,
            max_storage_scale=1.10,
            max_candidate_scale=1.10,
            max_candidates_per_row=256,
            max_fit_residual=1.0e-5,
        )

        with mock.patch.object(probe, "profile_frame_count", side_effect=fake_profile_frame_count):
            payload = probe.run_probe(args)

        self.assertEqual(captured_initializations, ["legacy_pixel_mean", "legacy_pixel_mean"])
        self.assertEqual(payload["site_initialization"], "legacy_pixel_mean")
        self.assertEqual([row["site_initialization"] for row in payload["rows"]], ["legacy_pixel_mean", "legacy_pixel_mean"])


if __name__ == "__main__":
    unittest.main()
