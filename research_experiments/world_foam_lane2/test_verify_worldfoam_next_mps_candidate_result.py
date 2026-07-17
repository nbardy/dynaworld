from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import run_worldfoam_next_mps_candidate as launcher
import verify_worldfoam_next_mps_candidate_result as verifier


FRAME_COUNTS = [2, 4, 8, 16, 32]


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _benchmark_environment_snapshot(*, status: str = "background") -> dict[str, object]:
    return {
        "status": status,
        "pid": 12345,
        "keywords": ["python", "pytest", "torch", "metal", "mps", "modal"],
        "hard_keywords": ["pytest", "torch", "metal", "mps"],
        "blocking_cpu_threshold": verifier.EXPECTED_BLOCKING_CPU_THRESHOLD,
        "general_blocking_cpu_threshold": verifier.EXPECTED_GENERAL_BLOCKING_CPU_THRESHOLD,
        "blocking_process_count": 0,
        "contending_process_count": 0,
        "background_process_count": 0,
        "blocking_processes": [],
        "background_processes": [],
        "contending_processes": [],
    }


def _merged_benchmark_environment(*, status: str = "background") -> dict[str, object]:
    return {
        "status": status,
        "start": _benchmark_environment_snapshot(status="background"),
        "end": _benchmark_environment_snapshot(status="background"),
    }


def _artifact_payload(
    *,
    site_initialization: str = "legacy_pixel_mean",
    fused_render_timing: bool = False,
) -> dict[str, object]:
    rows = []
    for index, frame_count in enumerate(FRAME_COUNTS):
        scale_offset = 0.001 * index
        backward_mean = 0.004 + scale_offset
        step_summary = {
            "total": {"mean_s": 0.010 + scale_offset},
            "backward": {"mean_s": backward_mean},
            "render": {"mean_s": 0.0 if fused_render_timing else 0.001 + scale_offset},
        }
        if fused_render_timing:
            step_summary["fused_loss_vjp"] = {"mean_s": backward_mean}
        rows.append(
            {
                "status": "ok",
                "frame_count": frame_count,
                "loaded_frame_count": frame_count,
                "render_size": 64,
                "site_count": 24,
                "steps": 8,
                "warmup_steps": 4,
                "repeat_loaded_frames": False,
                "final_train_psnr": 13.0 + scale_offset,
                "final_heldout_psnr": 14.0 + scale_offset,
                "final_train_l1": 0.18,
                "final_heldout_l1": 0.15,
                "step_summary": step_summary,
            }
        )
    return {
        "status": "ok",
        "benchmark": verifier.EXPECTED_BENCHMARK,
        "benchmark_environment": _merged_benchmark_environment(),
        "device": "mps",
        "frame_counts": FRAME_COUNTS,
        "frame_scale_first_to_last": 16.0,
        "total_step_scale_first_to_last": 1.4,
        "backward_scale_first_to_last": 2.0,
        "render_scale_first_to_last": 0.0 if fused_render_timing else 5.0,
        "render_timing_scope": "fused_loss_vjp_includes_render" if fused_render_timing else "separate_render",
        "fused_loss_vjp_scale_first_to_last": 2.0 if fused_render_timing else None,
        "render_size": 64,
        "site_count": 24,
        "site_initialization": site_initialization,
        "tape_mode": launcher.DEFAULT_TAPE_MODE,
        "optimizer_mode": "manual-vjp",
        "endpoint_record_source": "slow-owner-run",
        "experimental_selected_only_owner_run_delta_prep": True,
        "experimental_native_owner_run_cutwalk_delta": True,
        "allow_repeat_loaded_frames": False,
        "acceptance": {
            "all_rows_ok": True,
            "total_step_sublinear_vs_frames": True,
            "render_sublinear_vs_frames": True,
            "backward_sublinear_vs_frames": True,
            "selected_tape_segments_below_full_at_max_frame": True,
            "selected_tape_storage_below_full_at_max_frame": True,
            "owner_run_segments_below_full_at_max_frame": True,
        },
        "rows": rows,
    }


def _summary_payload(artifact_path: Path, *, status: str = "train_eval_ok") -> dict[str, object]:
    return {
        "benchmark": "world_foam_next_mps_candidate_launch",
        "status": status,
        "execute": True,
        "failures": [],
        "readiness_status": "ok",
        "ready_for_quiet_mps_quality_speed_run": True,
        "quality_claim": False,
        "speed_claim": False,
        "mps_quality_speed_artifact_required": True,
        "next_mps_candidate": "legacy_pixel_mean",
        "preflight_returncode": 0,
        "preflight_benchmark_environment_status": "background",
        "preflight_benchmark_environment": _benchmark_environment_snapshot(),
        "preflight_blocking_process_count": 0,
        "preflight_contending_process_count": 0,
        "preflight_blocking_reasons": [],
        "preflight_external_blocker_summary": {
            "blocking_reason_counts": {},
            "blocking_kind_counts": {},
            "manual_next_actions": [],
            "requires_external_quiet_window": False,
        },
        "preflight_blocking_processes": [],
        "preflight_stability_samples_requested": 3,
        "preflight_stability_samples_completed": 3,
        "preflight_stability_ok": True,
        "train_eval_returncode": 0,
        "planned_worldfoam_artifact": str(artifact_path),
        "train_eval_command": [
            "python",
            "train_eval_owner_run_tape.py",
            "--config",
            "real32.jsonc",
            "--frame-counts",
            "2,4,8,16,32",
            "--render-size",
            "64",
            "--site-count",
            "24",
            "--site-initialization",
            "legacy_pixel_mean",
            "--steps",
            "8",
            "--warmup-steps",
            "4",
            "--optimizer-mode",
            "manual-vjp",
            "--tape-mode",
            launcher.DEFAULT_TAPE_MODE,
            "--endpoint-record-source",
            "slow-owner-run",
            "--require-benchmark-environment-ok",
            "--out-json",
            str(artifact_path),
        ],
    }


class VerifyWorldFoamNextMpsCandidateResultTests(unittest.TestCase):
    def test_clean_candidate_result_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            artifact_path = _write_json(tmp / "candidate.worldfoam.json", _artifact_payload())
            summary_path = _write_json(tmp / "summary.json", _summary_payload(artifact_path))

            report = verifier.verify_summary(summary_path)

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["failures"], [])
        self.assertEqual(report["expected_frame_counts"], FRAME_COUNTS)
        self.assertFalse(report["artifact_checks_skipped"])

    def test_fused_loss_vjp_candidate_result_passes_with_zero_render_timing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            artifact_path = _write_json(
                tmp / "candidate.worldfoam.json",
                _artifact_payload(fused_render_timing=True),
            )
            summary_path = _write_json(tmp / "summary.json", _summary_payload(artifact_path))

            report = verifier.verify_summary(summary_path)

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["failures"], [])

    def test_preflight_contended_summary_fails_before_artifact_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            artifact_path = tmp / "missing.worldfoam.json"
            summary = _summary_payload(artifact_path, status="preflight_contended")
            summary["preflight_returncode"] = 2
            summary["preflight_benchmark_environment_status"] = "contended"
            summary["preflight_blocking_process_count"] = 1
            summary["preflight_contending_process_count"] = 1
            summary["preflight_blocking_reasons"] = ["high_cpu"]
            summary["preflight_external_blocker_summary"] = {
                "blocking_reason_counts": {"high_cpu": 1},
                "blocking_kind_counts": {"high_cpu_external_job": 1},
                "manual_next_actions": ["wait for or manually pause high-CPU external training/export jobs"],
                "requires_external_quiet_window": True,
            }
            summary["preflight_blocking_processes"] = [
                {
                    "pid": 7002,
                    "ppid": 6978,
                    "stat": "R",
                    "elapsed": "50:09",
                    "block_reason": "high_cpu",
                    "pcpu": 195.1,
                    "pmem": 1.9,
                    "command": "python train_node_curve_program_flow_v2.py",
                }
            ]
            summary["preflight_stability_samples_completed"] = 1
            summary["preflight_stability_ok"] = False
            summary["train_eval_returncode"] = None
            summary_path = _write_json(tmp / "summary.json", summary)

            report = verifier.verify_summary(summary_path)

        self.assertEqual(report["status"], "failed")
        self.assertIn("summary status is not train_eval_ok: 'preflight_contended'", report["failures"])
        self.assertIn("preflight_stability_ok is not true", report["failures"])
        self.assertTrue(report["artifact_checks_skipped"])
        self.assertNotIn("WorldFoam artifact is missing or invalid JSON", report["failures"])
        self.assertEqual(report["preflight_benchmark_environment_status"], "contended")
        self.assertEqual(report["preflight_blocking_process_count"], 1)
        self.assertEqual(report["preflight_contending_process_count"], 1)
        self.assertEqual(report["preflight_blocking_reasons"], ["high_cpu"])
        self.assertEqual(
            report["preflight_external_blocker_summary"],
            {
                "blocking_reason_counts": {"high_cpu": 1},
                "blocking_kind_counts": {"high_cpu_external_job": 1},
                "manual_next_actions": ["wait for or manually pause high-CPU external training/export jobs"],
                "requires_external_quiet_window": True,
            },
        )
        self.assertEqual(report["preflight_blocking_processes"][0]["pid"], 7002)
        self.assertEqual(report["preflight_blocking_processes"][0]["stat"], "R")

    def test_site_initialization_and_frame_coverage_must_match_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            artifact = _artifact_payload(site_initialization="legacy_sparse")
            artifact["frame_counts"] = [2, 4, 8]
            artifact["rows"] = artifact["rows"][:3]  # type: ignore[index]
            artifact_path = _write_json(tmp / "candidate.worldfoam.json", artifact)
            summary_path = _write_json(tmp / "summary.json", _summary_payload(artifact_path))

            report = verifier.verify_summary(summary_path)

        self.assertEqual(report["status"], "failed")
        self.assertIn("WorldFoam artifact site_initialization does not match candidate", report["failures"])
        self.assertIn("WorldFoam artifact frame_counts [2, 4, 8] do not match [2, 4, 8, 16, 32]", report["failures"])
        self.assertIn("WorldFoam row frame counts [2, 4, 8] do not match expected", report["failures"])

    def test_non_sublinear_timing_fails_even_if_rows_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            artifact = _artifact_payload()
            artifact["total_step_scale_first_to_last"] = 16.0
            artifact["render_scale_first_to_last"] = 16.0
            artifact_path = _write_json(tmp / "candidate.worldfoam.json", artifact)
            summary_path = _write_json(tmp / "summary.json", _summary_payload(artifact_path))

            report = verifier.verify_summary(summary_path)

        self.assertEqual(report["status"], "failed")
        self.assertIn("total_step_scale_first_to_last is not sublinear versus frame scale", report["failures"])
        self.assertIn("render_scale_first_to_last is not sublinear versus frame scale", report["failures"])

    def test_partial_frame_scale_matrix_fails_even_when_artifact_matches_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            artifact = _artifact_payload()
            artifact["frame_counts"] = [2, 4, 8]
            artifact["rows"] = artifact["rows"][:3]  # type: ignore[index]
            artifact_path = _write_json(tmp / "candidate.worldfoam.json", artifact)
            summary = _summary_payload(artifact_path)
            command = summary["train_eval_command"]
            assert isinstance(command, list)
            command[command.index("--frame-counts") + 1] = "2,4,8"
            summary_path = _write_json(tmp / "summary.json", summary)

            report = verifier.verify_summary(summary_path)

        self.assertEqual(report["status"], "failed")
        self.assertIn(
            "train_eval_command frame_counts [2, 4, 8] do not match required [2, 4, 8, 16, 32]",
            report["failures"],
        )

    def test_reordered_or_duplicate_frame_matrix_fails_even_when_artifact_is_full(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            artifact_path = _write_json(tmp / "candidate.worldfoam.json", _artifact_payload())
            summary = _summary_payload(artifact_path)
            command = summary["train_eval_command"]
            assert isinstance(command, list)
            command[command.index("--frame-counts") + 1] = "2,4,4,8,16,32"
            summary_path = _write_json(tmp / "summary.json", summary)

            duplicate_report = verifier.verify_summary(summary_path)

            command[command.index("--frame-counts") + 1] = "32,16,8,4,2"
            summary_path = _write_json(tmp / "summary.json", summary)
            reordered_report = verifier.verify_summary(summary_path)

        self.assertEqual(duplicate_report["status"], "failed")
        self.assertIn(
            "train_eval_command frame_counts [2, 4, 4, 8, 16, 32] do not match required [2, 4, 8, 16, 32]",
            duplicate_report["failures"],
        )
        self.assertEqual(reordered_report["status"], "failed")
        self.assertIn(
            "train_eval_command frame_counts [32, 16, 8, 4, 2] do not match required [2, 4, 8, 16, 32]",
            reordered_report["failures"],
        )

    def test_duplicate_or_invalid_artifact_rows_fail_even_when_frame_set_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            artifact = _artifact_payload()
            rows = artifact["rows"]
            assert isinstance(rows, list)
            duplicate_row = dict(rows[0])
            duplicate_row["frame_count"] = 4
            rows.append(duplicate_row)
            invalid_row = dict(rows[0])
            invalid_row["frame_count"] = True
            rows.append(invalid_row)
            artifact_path = _write_json(tmp / "candidate.worldfoam.json", artifact)
            summary_path = _write_json(tmp / "summary.json", _summary_payload(artifact_path))

            report = verifier.verify_summary(summary_path)

        self.assertEqual(report["status"], "failed")
        self.assertIn("WorldFoam row count 7 does not match expected 5", report["failures"])
        self.assertIn("WorldFoam duplicate row frame counts [4]", report["failures"])
        self.assertIn("WorldFoam artifact row has invalid frame_count True", report["failures"])

    def test_wrong_render_site_or_step_shape_fails_even_when_rows_are_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            artifact = _artifact_payload()
            artifact["render_size"] = 32
            artifact["site_count"] = 12
            rows = artifact["rows"]
            assert isinstance(rows, list)
            first_row = rows[0]
            assert isinstance(first_row, dict)
            first_row["render_size"] = 32
            first_row["site_count"] = 12
            first_row["steps"] = 4
            first_row["warmup_steps"] = 1
            artifact_path = _write_json(tmp / "candidate.worldfoam.json", artifact)
            summary = _summary_payload(artifact_path)
            command = summary["train_eval_command"]
            assert isinstance(command, list)
            command[command.index("--render-size") + 1] = "32"
            command[command.index("--site-count") + 1] = "12"
            command[command.index("--steps") + 1] = "4"
            command[command.index("--warmup-steps") + 1] = "1"
            summary_path = _write_json(tmp / "summary.json", summary)

            report = verifier.verify_summary(summary_path)

        self.assertEqual(report["status"], "failed")
        self.assertIn("train_eval_command --render-size 32 does not match required 64", report["failures"])
        self.assertIn("train_eval_command --site-count 12 does not match required 24", report["failures"])
        self.assertIn("train_eval_command --steps 4 does not match required 8", report["failures"])
        self.assertIn("train_eval_command --warmup-steps 1 does not match required 4", report["failures"])
        self.assertIn("WorldFoam artifact render_size 32 does not match required 64", report["failures"])
        self.assertIn("WorldFoam artifact site_count 12 does not match required 24", report["failures"])
        self.assertIn("WorldFoam row 2f render_size 32 does not match required 64", report["failures"])
        self.assertIn("WorldFoam row 2f site_count 12 does not match required 24", report["failures"])
        self.assertIn("WorldFoam row 2f steps 4 does not match required 8", report["failures"])
        self.assertIn("WorldFoam row 2f warmup_steps 1 does not match required 4", report["failures"])

    def test_nonfinite_quality_timing_or_scale_values_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            artifact = _artifact_payload()
            artifact["frame_scale_first_to_last"] = float("nan")
            artifact["total_step_scale_first_to_last"] = float("inf")
            artifact["render_scale_first_to_last"] = float("nan")
            artifact["backward_scale_first_to_last"] = float("nan")
            rows = artifact["rows"]
            assert isinstance(rows, list)
            first_row = rows[0]
            assert isinstance(first_row, dict)
            first_row["final_train_psnr"] = float("nan")
            first_row["final_heldout_psnr"] = float("inf")
            first_row["final_train_l1"] = float("-inf")
            first_row["final_heldout_l1"] = float("nan")
            step_summary = first_row["step_summary"]
            assert isinstance(step_summary, dict)
            total = step_summary["total"]
            render = step_summary["render"]
            backward = step_summary["backward"]
            assert isinstance(total, dict)
            assert isinstance(render, dict)
            assert isinstance(backward, dict)
            total["mean_s"] = float("nan")
            render["mean_s"] = float("-inf")
            backward["mean_s"] = float("inf")
            artifact_path = _write_json(tmp / "candidate.worldfoam.json", artifact)
            summary_path = _write_json(tmp / "summary.json", _summary_payload(artifact_path))

            report = verifier.verify_summary(summary_path)

        self.assertEqual(report["status"], "failed")
        self.assertIn("WorldFoam artifact frame_scale_first_to_last is not finite positive", report["failures"])
        self.assertIn("WorldFoam artifact total_step_scale_first_to_last is not finite positive", report["failures"])
        self.assertIn("WorldFoam artifact render_scale_first_to_last is not finite positive", report["failures"])
        self.assertIn("WorldFoam artifact backward_scale_first_to_last is not finite positive", report["failures"])
        self.assertIn("WorldFoam row 2f missing finite numeric final_train_psnr", report["failures"])
        self.assertIn("WorldFoam row 2f missing finite numeric final_heldout_psnr", report["failures"])
        self.assertIn("WorldFoam row 2f missing finite numeric final_train_l1", report["failures"])
        self.assertIn("WorldFoam row 2f missing finite numeric final_heldout_l1", report["failures"])
        self.assertIn("WorldFoam row 2f missing positive total mean_s", report["failures"])
        self.assertIn("WorldFoam row 2f missing positive render mean_s", report["failures"])
        self.assertIn("WorldFoam row 2f missing positive backward mean_s", report["failures"])

    def test_stale_environment_contract_fails_even_when_status_is_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            artifact = _artifact_payload()
            artifact["benchmark_environment"] = {"status": "background"}
            artifact_path = _write_json(tmp / "candidate.worldfoam.json", artifact)
            summary = _summary_payload(artifact_path)
            summary["preflight_benchmark_environment"] = {"status": "background"}
            summary_path = _write_json(tmp / "summary.json", summary)

            report = verifier.verify_summary(summary_path)

        self.assertEqual(report["status"], "failed")
        self.assertIn(
            "summary preflight benchmark environment missing current general_blocking_cpu_threshold",
            report["failures"],
        )
        self.assertIn(
            "WorldFoam artifact benchmark environment missing current general_blocking_cpu_threshold",
            report["failures"],
        )

    def test_dirty_environment_counts_fail_even_when_status_is_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            artifact = _artifact_payload()
            artifact_environment = _benchmark_environment_snapshot()
            artifact_environment["blocking_process_count"] = 1
            artifact["benchmark_environment"] = artifact_environment
            artifact_path = _write_json(tmp / "candidate.worldfoam.json", artifact)
            summary = _summary_payload(artifact_path)
            summary["preflight_contending_process_count"] = 1
            preflight_environment = _benchmark_environment_snapshot()
            preflight_environment["contending_process_count"] = 1
            summary["preflight_benchmark_environment"] = preflight_environment
            summary_path = _write_json(tmp / "summary.json", summary)

            report = verifier.verify_summary(summary_path)

        self.assertEqual(report["status"], "failed")
        self.assertIn("preflight recorded contending processes", report["failures"])
        self.assertIn(
            "summary preflight benchmark environment recorded contending processes",
            report["failures"],
        )
        self.assertIn(
            "WorldFoam artifact benchmark environment recorded blocking processes",
            report["failures"],
        )


if __name__ == "__main__":
    unittest.main()
