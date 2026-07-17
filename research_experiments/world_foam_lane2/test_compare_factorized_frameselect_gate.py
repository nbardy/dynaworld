from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import compare_factorized_frameselect_gate as gate


def _row(
    frame: int,
    *,
    total_s: float,
    backward_s: float,
    schema_bytes: int,
    topology_bytes: int,
    noncoeff_bytes: int,
    psnr: float = 13.0,
) -> dict[str, object]:
    return {
        "frame_count": frame,
        "status": "ok",
        "step_summary": {
            "total": {"median_s": total_s},
            "backward": {"median_s": backward_s},
        },
        "train_selected_tape_schema_storage_bytes": schema_bytes,
        "train_selected_tape_schema_topology_storage_bytes": topology_bytes,
        "train_selected_tape_mps_resident_noncoeff_storage_bytes": noncoeff_bytes,
        "final_train_psnr": psnr,
    }


def _payload(*, environment_status: str, rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "status": "ok",
        "benchmark_environment": {"status": environment_status},
        "rows": rows,
    }


class FactorizedFrameSelectGateTests(unittest.TestCase):
    def test_compare_payloads_marks_clean_frameselect_candidate(self) -> None:
        regular = _payload(
            environment_status="background",
            rows=[
                _row(2, total_s=0.0020, backward_s=0.0016, schema_bytes=1000, topology_bytes=400, noncoeff_bytes=500),
                _row(4, total_s=0.0040, backward_s=0.0032, schema_bytes=2000, topology_bytes=800, noncoeff_bytes=900),
            ],
        )
        frameselect = _payload(
            environment_status="background",
            rows=[
                _row(2, total_s=0.0018, backward_s=0.0015, schema_bytes=900, topology_bytes=360, noncoeff_bytes=460),
                _row(4, total_s=0.0041, backward_s=0.0033, schema_bytes=1800, topology_bytes=720, noncoeff_bytes=850),
            ],
        )

        comparison = gate.compare_payloads(
            regular,
            frameselect,
            max_total_ratio=1.10,
            max_backward_ratio=1.10,
            max_storage_ratio=1.00,
        )

        self.assertEqual(comparison["status"], "ok")
        self.assertEqual(comparison["recommendation"], "frameselect_candidate")
        self.assertTrue(comparison["clean_speedscale_artifact"])
        self.assertAlmostEqual(comparison["max_total_median_ratio"], 1.025)
        self.assertAlmostEqual(comparison["max_schema_storage_ratio"], 0.9)
        self.assertEqual(comparison["frames_compared"], [2, 4])

    def test_compare_payloads_rejects_contaminated_artifacts(self) -> None:
        regular = _payload(
            environment_status="contended",
            rows=[_row(2, total_s=0.0020, backward_s=0.0016, schema_bytes=1000, topology_bytes=400, noncoeff_bytes=500)],
        )
        frameselect = _payload(
            environment_status="background",
            rows=[_row(2, total_s=0.0018, backward_s=0.0015, schema_bytes=900, topology_bytes=360, noncoeff_bytes=460)],
        )

        comparison = gate.compare_payloads(
            regular,
            frameselect,
            max_total_ratio=1.10,
            max_backward_ratio=1.10,
            max_storage_ratio=1.00,
        )

        self.assertEqual(comparison["status"], "failed")
        self.assertEqual(comparison["recommendation"], "rerun_clean")
        self.assertFalse(comparison["clean_speedscale_artifact"])
        self.assertIn("benchmark artifacts are not clean", comparison["failures"][0])

    def test_compare_candidate_set_selects_framebitmask_when_frameselect_storage_regresses(self) -> None:
        regular = _payload(
            environment_status="background",
            rows=[
                _row(16, total_s=0.0030, backward_s=0.0025, schema_bytes=67000, topology_bytes=42000, noncoeff_bytes=43000),
            ],
        )
        frameselect = _payload(
            environment_status="background",
            rows=[
                _row(16, total_s=0.0031, backward_s=0.0026, schema_bytes=74000, topology_bytes=49000, noncoeff_bytes=50000),
            ],
        )
        framebitmask = _payload(
            environment_status="background",
            rows=[
                _row(16, total_s=0.0030, backward_s=0.0025, schema_bytes=61760, topology_bytes=36624, noncoeff_bytes=36736),
            ],
        )

        comparison = gate.compare_candidate_set(
            regular,
            {
                "frameselect": frameselect,
                "framebitmask": framebitmask,
            },
            max_total_ratio=1.10,
            max_backward_ratio=1.10,
            max_storage_ratio=1.00,
        )

        self.assertEqual(comparison["status"], "ok")
        self.assertEqual(comparison["recommendation"], "framebitmask_candidate")
        self.assertEqual(comparison["best_candidate"], "framebitmask")
        self.assertEqual(comparison["passing_candidates"], ["framebitmask"])
        self.assertEqual(
            comparison["candidate_comparisons"]["frameselect"]["recommendation"],
            "keep_regular_or_fork_again",
        )

    def test_main_writes_preflight_failure_without_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            with (
                mock.patch.object(
                    gate.promotion_gate,
                    "_run_preflight",
                    return_value=(
                        2,
                        [
                            {
                                "returncode": 2,
                                "status": "contended",
                                "success_streak": 0,
                            }
                        ],
                    ),
                ),
                mock.patch.object(gate.promotion_gate, "_run") as train_run,
            ):
                status = gate.main(
                    [
                        "--run-id",
                        "unit_preflight_failure",
                        "--summary-json",
                        str(summary_path),
                        "--stable-preflight-checks",
                        "1",
                    ]
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 2)
        self.assertEqual(summary["status"], "preflight_failed_before_regular")
        self.assertEqual(summary["regular_preflight_status"], 2)
        train_run.assert_not_called()

    def test_main_marks_interrupted_preflight_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            with (
                mock.patch.object(gate.promotion_gate, "_run_preflight", side_effect=KeyboardInterrupt),
                mock.patch.object(gate.promotion_gate, "_run") as train_run,
            ):
                status = gate.main(
                    [
                        "--run-id",
                        "unit_interrupted_preflight",
                        "--summary-json",
                        str(summary_path),
                    ]
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 130)
        self.assertEqual(summary["status"], "interrupted")
        self.assertEqual(summary["interrupted_reason"], "KeyboardInterrupt")
        self.assertEqual(summary["interrupted_previous_status"], "regular_attempt_pending")
        self.assertEqual(summary["current_attempt_index"], 1)
        train_run.assert_not_called()

    def test_dry_run_writes_both_train_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            status = gate.main(
                [
                    "--run-id",
                    "unit_dry_run",
                    "--summary-json",
                    str(summary_path),
                    "--dry-run",
                ]
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertEqual(summary["status"], "dry_run")
        self.assertIn(gate.REGULAR_MODE, summary["regular_train_command"])
        self.assertIn(gate.FRAMESELECT_MODE, summary["frameselect_train_command"])

    def test_dry_run_can_include_framebitmask_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            status = gate.main(
                [
                    "--run-id",
                    "unit_dry_run_bitmask",
                    "--summary-json",
                    str(summary_path),
                    "--dry-run",
                    "--include-framebitmask",
                ]
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertEqual(summary["status"], "dry_run")
        self.assertEqual(summary["candidate_labels"], ["frameselect", "framebitmask"])
        self.assertIn(gate.FRAMEBITMASK_MODE, summary["framebitmask_train_command"])
        self.assertIn("framebitmask_out_json", summary["attempt_artifacts"][0])

    def test_dry_run_can_select_only_framebitmask_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            status = gate.main(
                [
                    "--run-id",
                    "unit_dry_run_only_bitmask",
                    "--summary-json",
                    str(summary_path),
                    "--dry-run",
                    "--candidate-labels",
                    "framebitmask",
                ]
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertEqual(summary["status"], "dry_run")
        self.assertEqual(summary["candidate_labels"], ["framebitmask"])
        self.assertIn(gate.FRAMEBITMASK_MODE, summary["framebitmask_train_command"])
        self.assertNotIn("frameselect_train_command", summary)
        self.assertNotIn("frameselect_out_json", summary["attempt_artifacts"][0])

    def test_main_retries_after_frameselect_artifact_contamination(self) -> None:
        regular = _payload(
            environment_status="background",
            rows=[
                _row(2, total_s=0.0020, backward_s=0.0016, schema_bytes=1000, topology_bytes=400, noncoeff_bytes=500),
            ],
        )
        frameselect_contended = _payload(
            environment_status="contended",
            rows=[
                _row(2, total_s=0.0018, backward_s=0.0015, schema_bytes=900, topology_bytes=360, noncoeff_bytes=460),
            ],
        )
        frameselect_clean = _payload(
            environment_status="background",
            rows=[
                _row(2, total_s=0.0018, backward_s=0.0015, schema_bytes=900, topology_bytes=360, noncoeff_bytes=460),
            ],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            with (
                mock.patch.object(gate.promotion_gate, "_run_preflight", return_value=(0, [])),
                mock.patch.object(gate.promotion_gate, "_run", return_value=0) as train_run,
                mock.patch.object(
                    gate,
                    "_load_payload",
                    side_effect=[
                        regular,
                        frameselect_contended,
                        frameselect_clean,
                    ],
                ),
            ):
                status = gate.main(
                    [
                        "--run-id",
                        "unit_retry_contaminated",
                        "--summary-json",
                        str(summary_path),
                        "--max-comparison-attempts",
                        "2",
                    ]
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(len(summary["attempts"]), 3)
        self.assertEqual(summary["attempts"][1]["retry_reason"], "frameselect_artifact_contaminated")
        self.assertEqual(summary["regular_accepted_attempt_index"], 1)
        self.assertEqual(summary["frameselect_accepted_attempt_index"], 2)
        self.assertEqual(summary["current_mode_label"], "frameselect")
        self.assertEqual(summary["current_attempt_index"], 2)
        self.assertEqual(train_run.call_count, 3)
        self.assertEqual(summary["comparison"]["recommendation"], "frameselect_candidate")

    def test_main_retries_after_nonzero_regular_contaminated_artifact(self) -> None:
        regular_contended = _payload(
            environment_status="contended",
            rows=[
                _row(2, total_s=0.0020, backward_s=0.0016, schema_bytes=1000, topology_bytes=400, noncoeff_bytes=500),
            ],
        )
        regular_clean = _payload(
            environment_status="background",
            rows=[
                _row(2, total_s=0.0020, backward_s=0.0016, schema_bytes=1000, topology_bytes=400, noncoeff_bytes=500),
            ],
        )
        frameselect_clean = _payload(
            environment_status="background",
            rows=[
                _row(2, total_s=0.0018, backward_s=0.0015, schema_bytes=900, topology_bytes=360, noncoeff_bytes=460),
            ],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            regular_path = Path(tmpdir) / "regular.json"
            regular_path.with_name("regular.attempt1.json").write_text("{}", encoding="utf-8")
            regular_path.with_name("regular.attempt2.json").write_text("{}", encoding="utf-8")
            with (
                mock.patch.object(gate.promotion_gate, "_run_preflight", return_value=(0, [])),
                mock.patch.object(gate.promotion_gate, "_run", side_effect=[1, 0, 0]) as train_run,
                mock.patch.object(
                    gate,
                    "_load_payload",
                    side_effect=[
                        regular_contended,
                        regular_clean,
                        frameselect_clean,
                    ],
                ),
            ):
                status = gate.main(
                    [
                        "--run-id",
                        "unit_retry_nonzero_regular_contaminated",
                        "--summary-json",
                        str(summary_path),
                        "--regular-out-json",
                        str(regular_path),
                        "--allow-overwrite-artifacts",
                        "--max-comparison-attempts",
                        "2",
                    ]
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["attempts"][0]["retry_reason"], "regular_artifact_contaminated")
        self.assertEqual(summary["regular_accepted_attempt_index"], 2)
        self.assertEqual(summary["frameselect_accepted_attempt_index"], 1)
        self.assertEqual(summary["current_mode_label"], "frameselect")
        self.assertEqual(summary["current_attempt_index"], 1)
        self.assertEqual(train_run.call_count, 3)

    def test_main_retries_after_child_start_environment_contended_without_artifact(self) -> None:
        regular_clean = _payload(
            environment_status="background",
            rows=[
                _row(2, total_s=0.0020, backward_s=0.0016, schema_bytes=1000, topology_bytes=400, noncoeff_bytes=500),
            ],
        )
        frameselect_clean = _payload(
            environment_status="background",
            rows=[
                _row(2, total_s=0.0018, backward_s=0.0015, schema_bytes=900, topology_bytes=360, noncoeff_bytes=460),
            ],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            regular_path = Path(tmpdir) / "regular.json"
            with (
                mock.patch.object(gate.promotion_gate, "_run_preflight", return_value=(0, [])),
                mock.patch.object(gate.promotion_gate, "_run", side_effect=[2, 0, 0]) as train_run,
                mock.patch.object(
                    gate,
                    "_load_payload",
                    side_effect=[
                        regular_clean,
                        frameselect_clean,
                    ],
                ),
            ):
                status = gate.main(
                    [
                        "--run-id",
                        "unit_retry_child_start_contended",
                        "--summary-json",
                        str(summary_path),
                        "--regular-out-json",
                        str(regular_path),
                        "--max-comparison-attempts",
                        "2",
                    ]
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["attempts"][0]["retry_reason"], "regular_start_environment_contended")
        self.assertEqual(summary["regular_accepted_attempt_index"], 2)
        self.assertEqual(summary["frameselect_accepted_attempt_index"], 1)
        self.assertEqual(summary["current_mode_label"], "frameselect")
        self.assertEqual(summary["current_attempt_index"], 1)
        self.assertEqual(train_run.call_count, 3)

    def test_main_clears_stale_artifact_fields_on_later_start_contamination(self) -> None:
        regular_contended = _payload(
            environment_status="contended",
            rows=[
                _row(2, total_s=0.0020, backward_s=0.0016, schema_bytes=1000, topology_bytes=400, noncoeff_bytes=500),
            ],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            regular_path = Path(tmpdir) / "regular.json"
            regular_path.with_name("regular.attempt1.json").write_text("{}", encoding="utf-8")
            with (
                mock.patch.object(gate.promotion_gate, "_run_preflight", return_value=(0, [])),
                mock.patch.object(gate.promotion_gate, "_run", side_effect=[1, 2]) as train_run,
                mock.patch.object(gate, "_load_payload", return_value=regular_contended),
            ):
                status = gate.main(
                    [
                        "--run-id",
                        "unit_clear_stale_start_contended",
                        "--summary-json",
                        str(summary_path),
                        "--regular-out-json",
                        str(regular_path),
                        "--allow-overwrite-artifacts",
                        "--max-comparison-attempts",
                        "2",
                    ]
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 2)
        self.assertEqual(summary["status"], "regular_start_environment_contended")
        self.assertEqual(summary["regular_train_status"], 2)
        self.assertEqual(summary["regular_artifact_missing_after_train_status"], 2)
        self.assertNotIn("regular_artifact_status", summary)
        self.assertNotIn("regular_benchmark_environment_status", summary)
        self.assertEqual(summary["attempts"][0]["retry_reason"], "regular_artifact_contaminated")
        self.assertEqual(summary["attempts"][1]["regular_status"], 2)
        self.assertEqual(train_run.call_count, 2)

    def test_main_retries_after_framebitmask_artifact_contamination(self) -> None:
        regular = _payload(
            environment_status="background",
            rows=[
                _row(16, total_s=0.0030, backward_s=0.0025, schema_bytes=67000, topology_bytes=42000, noncoeff_bytes=43000),
            ],
        )
        frameselect = _payload(
            environment_status="background",
            rows=[
                _row(16, total_s=0.0031, backward_s=0.0026, schema_bytes=74000, topology_bytes=49000, noncoeff_bytes=50000),
            ],
        )
        framebitmask_contended = _payload(
            environment_status="contended",
            rows=[
                _row(16, total_s=0.0030, backward_s=0.0025, schema_bytes=61760, topology_bytes=36624, noncoeff_bytes=36736),
            ],
        )
        framebitmask_clean = _payload(
            environment_status="background",
            rows=[
                _row(16, total_s=0.0030, backward_s=0.0025, schema_bytes=61760, topology_bytes=36624, noncoeff_bytes=36736),
            ],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            with (
                mock.patch.object(gate.promotion_gate, "_run_preflight", return_value=(0, [])),
                mock.patch.object(gate.promotion_gate, "_run", return_value=0) as train_run,
                mock.patch.object(
                    gate,
                    "_load_payload",
                    side_effect=[
                        regular,
                        frameselect,
                        framebitmask_contended,
                        framebitmask_clean,
                    ],
                ),
            ):
                status = gate.main(
                    [
                        "--run-id",
                        "unit_retry_framebitmask_contaminated",
                        "--summary-json",
                        str(summary_path),
                        "--max-comparison-attempts",
                        "2",
                        "--include-framebitmask",
                    ]
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(len(summary["attempts"]), 4)
        self.assertEqual(summary["attempts"][2]["retry_reason"], "framebitmask_artifact_contaminated")
        self.assertEqual(summary["regular_accepted_attempt_index"], 1)
        self.assertEqual(summary["frameselect_accepted_attempt_index"], 1)
        self.assertEqual(summary["framebitmask_accepted_attempt_index"], 2)
        self.assertEqual(summary["current_mode_label"], "framebitmask")
        self.assertEqual(summary["current_attempt_index"], 2)
        self.assertEqual(train_run.call_count, 4)
        self.assertEqual(summary["comparison"]["recommendation"], "framebitmask_candidate")

    def test_main_reuses_accepted_artifacts_and_runs_only_missing_framebitmask(self) -> None:
        regular = _payload(
            environment_status="background",
            rows=[
                _row(16, total_s=0.0030, backward_s=0.0025, schema_bytes=67000, topology_bytes=42000, noncoeff_bytes=43000),
            ],
        )
        frameselect = _payload(
            environment_status="background",
            rows=[
                _row(16, total_s=0.0031, backward_s=0.0026, schema_bytes=74000, topology_bytes=49000, noncoeff_bytes=50000),
            ],
        )
        framebitmask = _payload(
            environment_status="background",
            rows=[
                _row(16, total_s=0.0030, backward_s=0.0025, schema_bytes=61760, topology_bytes=36624, noncoeff_bytes=36736),
            ],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            accepted_regular = Path(tmpdir) / "accepted_regular.json"
            accepted_frameselect = Path(tmpdir) / "accepted_frameselect.json"
            accepted_regular.write_text("{}", encoding="utf-8")
            accepted_frameselect.write_text("{}", encoding="utf-8")
            with (
                mock.patch.object(gate.promotion_gate, "_run_preflight", return_value=(0, [])),
                mock.patch.object(gate.promotion_gate, "_run", return_value=0) as train_run,
                mock.patch.object(
                    gate,
                    "_load_payload",
                    side_effect=[
                        regular,
                        frameselect,
                        framebitmask,
                    ],
                ),
            ):
                status = gate.main(
                    [
                        "--run-id",
                        "unit_reuse_missing_framebitmask",
                        "--summary-json",
                        str(summary_path),
                        "--accepted-regular-json",
                        str(accepted_regular),
                        "--accepted-frameselect-json",
                        str(accepted_frameselect),
                        "--include-framebitmask",
                    ]
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["regular_accepted_artifact_source"], "input")
        self.assertEqual(summary["frameselect_accepted_artifact_source"], "input")
        self.assertEqual(summary["framebitmask_accepted_attempt_index"], 1)
        self.assertEqual(len(summary["attempts"]), 1)
        self.assertEqual(summary["attempts"][0]["mode_label"], "framebitmask")
        self.assertEqual(summary["current_mode_label"], "framebitmask")
        self.assertEqual(train_run.call_count, 1)
        self.assertEqual(summary["comparison"]["recommendation"], "framebitmask_candidate")

    def test_main_rejects_contended_accepted_artifact(self) -> None:
        regular_contended = _payload(
            environment_status="contended",
            rows=[
                _row(16, total_s=0.0030, backward_s=0.0025, schema_bytes=67000, topology_bytes=42000, noncoeff_bytes=43000),
            ],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            accepted_regular = Path(tmpdir) / "accepted_regular.json"
            accepted_regular.write_text("{}", encoding="utf-8")
            with (
                mock.patch.object(gate.promotion_gate, "_run_preflight") as preflight,
                mock.patch.object(gate.promotion_gate, "_run") as train_run,
                mock.patch.object(gate, "_load_payload", return_value=regular_contended),
            ):
                status = gate.main(
                    [
                        "--run-id",
                        "unit_reject_contended_accepted",
                        "--summary-json",
                        str(summary_path),
                        "--accepted-regular-json",
                        str(accepted_regular),
                    ]
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 2)
        self.assertEqual(summary["status"], "regular_accepted_artifact_not_clean")
        self.assertFalse(summary["regular_artifact_clean"])
        preflight.assert_not_called()
        train_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
