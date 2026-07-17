from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate4_affine_candidate_csr_promotion_gate as runner


def _args(tmpdir: str, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "run_id": "unit_candidate_csr",
        "config": runner.DEFAULT_CONFIG,
        "results_dir": Path(tmpdir),
        "out_summary": Path(tmpdir) / "summary.json",
        "frame_counts": "2,4,8,16",
        "render_size": 16,
        "site_count": 24,
        "near": 0.1,
        "far": 6.0,
        "density": 10.0,
        "invalid_epsilon": 1.0e-6,
        "transmittance_threshold": 1.0e-4,
        "origin_velocity_x": 0.08,
        "origin_velocity_y": 0.0,
        "origin_velocity_z": 0.02,
        "direction_velocity_x": 0.02,
        "direction_velocity_y": 0.0,
        "direction_velocity_z": 0.0,
        "steps": 5,
        "warmup_steps": 2,
        "lr": 0.03,
        "beta1": 0.9,
        "beta2": 0.999,
        "adam_eps": 1.0e-8,
        "gate4_time_slabs": 1,
        "gate4_residual_depth_padding": 0.001,
        "defer_heldout_device": True,
        "max_promotion_attempts": 2,
        "wait_for_benchmark_environment_ok": False,
        "wait_timeout_s": 0.0,
        "wait_interval_s": 0.0,
        "stable_preflight_checks": 1,
        "min_train_psnr": 8.0,
        "min_heldout_psnr": 8.0,
        "max_total_scale": 2.0,
        "max_backward_scale": 2.0,
        "max_total_median_scale": 2.0,
        "max_backward_median_scale": 2.0,
        "max_storage_scale": 1.10,
        "max_noncoeff_storage_scale": 1.10,
        "max_candidate_scale": 1.10,
        "max_candidates_per_row": 256,
        "max_row_mean_to_median": 2.0,
        "max_row_max_to_median": 4.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _environment(status: str) -> dict[str, object]:
    return {
        "status": status,
        "blocking_processes": [] if status == "background" else [{"pid": 123, "pcpu": 99.0}],
    }


def _step_summary(frame: int) -> dict[str, dict[str, float | int]]:
    total_s = {2: 0.0045, 4: 0.0048, 8: 0.0042, 16: 0.0044}[frame]
    backward_s = {2: 0.0039, 4: 0.0040, 8: 0.0036, 16: 0.0038}[frame]
    return {
        "render": {"count": 5, "mean_s": 0.0, "median_s": 0.0, "min_s": 0.0, "max_s": 0.0},
        "loss_eval": {"count": 5, "mean_s": 0.0, "median_s": 0.0, "min_s": 0.0, "max_s": 0.0},
        "backward": {
            "count": 5,
            "mean_s": backward_s,
            "median_s": backward_s,
            "min_s": backward_s * 0.95,
            "max_s": backward_s * 1.05,
        },
        "optimizer": {"count": 5, "mean_s": 0.0005, "median_s": 0.0005, "min_s": 0.00045, "max_s": 0.00055},
        "total": {
            "count": 5,
            "mean_s": total_s,
            "median_s": total_s,
            "min_s": total_s * 0.95,
            "max_s": total_s * 1.05,
        },
    }


def _row(frame: int) -> dict[str, object]:
    storage = {2: 1_048_324, 4: 1_044_480, 8: 1_039_540, 16: 1_039_920}[frame]
    candidates = {2: 84_930, 4: 84_609, 8: 84_196, 16: 84_225}[frame]
    return {
        "frame_count": frame,
        "status": "ok",
        "tape_mode": runner.GATE4_AFFINE_CANDIDATE_NUM32_DEN16_FUSED_MSE_MODE,
        "render_size": 16,
        "site_count": 24,
        "endpoint_record_source": "gate4-affine",
        "gate4_affine_candidate_csr_fused_mse": True,
        "final_train_psnr": 14.0,
        "final_heldout_psnr": 13.5,
        "first_grad_abs_sum": 0.3,
        "parameter_update_abs_max": 0.1,
        "step_summary": _step_summary(frame),
        "train_selected_tape_mps_resident_storage_bytes": storage,
        "train_selected_tape_mps_resident_noncoeff_storage_bytes": storage,
        "gate4_endpoint_train_metadata": {
            "candidate_count": candidates,
            "max_candidates_per_row": 224,
        },
        "acceptance": {
            "loss_decreased": True,
            "gradients_nonzero": True,
            "parameters_updated": True,
            "selected_tape_segments_below_full": True,
            "owner_run_segments_below_full": True,
            "selected_tape_vjp_under_segment_cap": True,
            "owner_run_vjp_under_segment_cap": True,
            "outputs_are_finite": True,
        },
    }


def _payload(environment_status: str) -> dict[str, object]:
    return {
        "benchmark": "world_foam_lane2_segment_tape_train_eval_mps",
        "status": "ok",
        "tape_mode": runner.GATE4_AFFINE_CANDIDATE_NUM32_DEN16_FUSED_MSE_MODE,
        "gate4_affine_candidate_csr_fused_mse": True,
        "endpoint_record_source": "gate4-affine",
        "optimizer_mode": "manual-vjp",
        "full_trainer_claim": False,
        "full_geometry_gradient_claim": False,
        "quality_claim": False,
        "render_size": 16,
        "site_count": 24,
        "frame_counts": [2, 4, 8, 16],
        "benchmark_environment": {"status": environment_status},
        "rows": [_row(frame) for frame in (2, 4, 8, 16)],
    }


class RunGate4AffineCandidateCSRPromotionGateTests(unittest.TestCase):
    def test_compact_environment_records_counts_and_short_processes(self) -> None:
        environment = {
            "status": "contended",
            "pid": 999,
            "blocking_cpu_threshold": 5.0,
            "blocking_process_count": 9,
            "contending_process_count": 3,
            "background_process_count": 4,
            "blocking_processes": [
                {
                    "pid": 100 + index,
                    "ppid": 1,
                    "pcpu": 90.0 - index,
                    "pmem": 1.5,
                    "command": f"python busy_{index}.py",
                    "extra": "drop me",
                }
                for index in range(7)
            ],
            "contending_processes": [{"pid": 200, "ppid": 1, "pcpu": 10.0, "pmem": 0.5, "command": "pytest"}],
            "background_processes": [{"pid": 300}, {"pid": 301}],
        }

        compact = runner._compact_environment(environment)

        self.assertEqual(compact["status"], "contended")
        self.assertEqual(compact["blocking_process_count"], 9)
        self.assertEqual(compact["contending_process_count"], 3)
        self.assertEqual(compact["background_process_count"], 4)
        self.assertEqual(len(compact["blocking_processes"]), 5)
        self.assertEqual(set(compact["blocking_processes"][0]), {"pid", "pcpu", "pmem", "command"})

    def test_blocks_when_preflight_is_contended_without_wait(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _args(tmpdir)
            with patch.object(runner, "_capture_benchmark_environment", return_value=_environment("contended")):
                with patch.object(runner, "run_train_eval") as train_eval:
                    summary = runner.run_promotion(args)

        self.assertEqual(summary["status"], "preflight_blocked")
        train_eval.assert_not_called()

    def test_promotes_clean_first_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _args(tmpdir)
            with patch.object(runner, "_capture_benchmark_environment", return_value=_environment("background")):
                with patch.object(runner, "run_train_eval", return_value=_payload("background")) as train_eval:
                    summary = runner.run_promotion(args)

            summary_path = Path(tmpdir) / "summary.json"
            saved = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertTrue(Path(summary["attempts"][0]["artifact"]).exists())
            self.assertTrue(Path(summary["attempts"][0]["verify_artifact"]).exists())

        self.assertEqual(summary["status"], "promoted")
        self.assertEqual(saved["status"], "promoted")
        self.assertEqual(saved["endpoint_record_source"], "gate4-affine")
        self.assertEqual(summary["attempt_count"], 1)
        self.assertEqual(train_eval.call_count, 1)
        self.assertEqual(train_eval.call_args.kwargs["endpoint_record_source"], "gate4-affine")

    def test_retries_contaminated_attempt_then_promotes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _args(tmpdir, wait_for_benchmark_environment_ok=True)
            train_eval = Mock(side_effect=[_payload("contended"), _payload("background")])
            with patch.object(runner, "_capture_benchmark_environment", return_value=_environment("background")):
                with patch.object(runner, "run_train_eval", train_eval):
                    summary = runner.run_promotion(args)

        self.assertEqual(summary["status"], "promoted")
        self.assertEqual(summary["attempt_count"], 2)
        self.assertEqual(summary["attempts"][0]["verify_status"], "failed")
        self.assertEqual(summary["attempts"][1]["verify_status"], "ok")
        self.assertEqual(train_eval.call_count, 2)


if __name__ == "__main__":
    unittest.main()
