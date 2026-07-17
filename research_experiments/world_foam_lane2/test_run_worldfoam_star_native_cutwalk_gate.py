from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import run_worldfoam_star_native_cutwalk_gate as gate


def _accepted_worldfoam_payload(environment: str | None = "background") -> dict[str, object]:
    payload: dict[str, object] = {
        "status": "ok",
        "rows": [],
        "acceptance": {"all_rows_ok": True},
    }
    if environment is not None:
        payload["benchmark_environment"] = {"status": environment}
    return payload


class WorldFoamStarNativeCutwalkGateTests(unittest.TestCase):
    def test_blocking_process_summary_keeps_periodic_mps_exporters_with_high_cpu_rows(self) -> None:
        summary = gate._blocking_process_summary(
            {
                "blocking_cpu_threshold": 5.0,
                "blocking_processes": [
                    {
                        "pid": 1,
                        "pcpu": 25.0,
                        "command": "python dense_alpha_failure.py",
                        "block_reason": "high_cpu",
                    },
                    {
                        "pid": 2,
                        "pcpu": 0.0,
                        "command": "uv run python scripts/run_btc15m_overnight_shadow_monitor.py",
                        "block_reason": "periodic_mps_exporter",
                    },
                    {
                        "pid": 3,
                        "pcpu": 0.0,
                        "command": "python other blocked job",
                        "block_reason": "keyword:mps",
                    },
                ],
            }
        )

        self.assertEqual([row["pid"] for row in summary], [1, 2])
        self.assertEqual(summary[0]["block_reason"], "high_cpu")
        self.assertEqual(summary[1]["block_reason"], "periodic_mps_exporter")

    def test_dry_run_writes_guarded_worldfoam_and_star_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            status = gate.main(
                [
                    "--run-id",
                    "unit_native_cutwalk_gate",
                    "--summary-json",
                    str(summary_path),
                    "--wait-timeout-s",
                    "123",
                    "--wait-poll-s",
                    "7",
                    "--dry-run",
                ]
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertEqual(summary["summary_schema_version"], gate.SUMMARY_SCHEMA_VERSION)
        self.assertEqual(summary["status"], "dry_run")
        self.assertEqual(summary["frame_counts"], [2, 4, 8, 16])
        self.assertIn("unit_native_cutwalk_gate.worldfoam.json", summary["planned_worldfoam_artifact"])
        self.assertIsNone(summary["worldfoam_artifact"])
        self.assertIn("unit_native_cutwalk_gate.star_compare.json", summary["planned_star_compare_artifact"])
        self.assertIsNone(summary["star_compare_artifact"])
        self.assertIsNone(summary["star_compare_command"])
        self.assertIn("--benchmark-environment-check-only", summary["worldfoam_preflight_command"])
        self.assertIn("--wait-for-benchmark-environment-ok-timeout-s", summary["worldfoam_preflight_command"])
        self.assertIn("--experimental-native-owner-run-cutwalk-delta", summary["worldfoam_command"])
        self.assertIn("--experimental-selected-only-owner-run-delta-prep", summary["worldfoam_command"])
        self.assertIn("--require-benchmark-environment-ok", summary["worldfoam_command"])
        self.assertIn("--wait-for-benchmark-environment-ok-timeout-s", summary["worldfoam_command"])
        self.assertIn("123.0", summary["worldfoam_command"])
        self.assertIn("--post-run-benchmark-environment-settle-s", summary["worldfoam_command"])
        self.assertEqual(summary["post_run_benchmark_environment_settle_s"], 2.0)
        self.assertIn("--require-clean-worldfoam-artifact", summary["planned_star_compare_command"])
        self.assertIn("--require-benchmark-environment-ok", summary["planned_star_compare_command"])
        self.assertIn("--post-run-benchmark-environment-settle-s", summary["planned_star_compare_command"])
        self.assertIn("--star-target-size", summary["planned_star_compare_command"])
        self.assertIn("64", summary["planned_star_compare_command"])
        self.assertIn("--star-tube-count", summary["planned_star_compare_command"])
        self.assertIn("896", summary["planned_star_compare_command"])

    def test_dry_run_repeat_loaded_frames_propagates_to_both_benchmarks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            status = gate.main(
                [
                    "--run-id",
                    "unit_repeat_loaded_gate",
                    "--summary-json",
                    str(summary_path),
                    "--repeat-loaded-frames",
                    "--dry-run",
                ]
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertTrue(summary["repeat_loaded_frames"])
        self.assertIn("--repeat-loaded-frames", summary["worldfoam_command"])
        self.assertIn("--star-repeat-loaded-frames", summary["planned_star_compare_command"])

    def test_dry_run_custom_fixture_paths_and_real_frame_requirement_are_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            worldfoam_config = tmp_path / "worldfoam_64f.jsonc"
            star_video = tmp_path / "star_64f.mp4"
            summary_path = tmp_path / "summary.json"
            status = gate.main(
                [
                    "--run-id",
                    "unit_real_fixture_gate",
                    "--summary-json",
                    str(summary_path),
                    "--worldfoam-config",
                    str(worldfoam_config),
                    "--star-video-path",
                    str(star_video),
                    "--require-real-loaded-frames",
                    "--dry-run",
                ]
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertFalse(summary["repeat_loaded_frames"])
        self.assertTrue(summary["require_real_loaded_frames"])
        self.assertEqual(summary["worldfoam_config"], str(worldfoam_config))
        self.assertEqual(summary["star_video_path"], str(star_video))
        self.assertIn("--config", summary["worldfoam_preflight_command"])
        self.assertIn(str(worldfoam_config), summary["worldfoam_preflight_command"])
        self.assertIn("--config", summary["worldfoam_command"])
        self.assertIn(str(worldfoam_config), summary["worldfoam_command"])
        self.assertIn("--video-path", summary["planned_star_compare_command"])
        self.assertIn(str(star_video), summary["planned_star_compare_command"])

    def test_real_frame_requirement_rejects_repeat_loaded_mode(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            gate.parse_args(["--require-real-loaded-frames", "--repeat-loaded-frames"])

        self.assertEqual(ctx.exception.code, 2)

    def test_real_frame_requirement_requires_explicit_real_inputs(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            gate.parse_args(["--require-real-loaded-frames"])

        self.assertEqual(ctx.exception.code, 2)

    def test_real_frame_requirement_rejects_one_sided_real_inputs(self) -> None:
        cases = [
            ["--require-real-loaded-frames", "--worldfoam-config", "real32.jsonc"],
            ["--require-real-loaded-frames", "--star-video-path", "real32.mp4"],
        ]
        for argv in cases:
            with self.subTest(argv=argv), self.assertRaises(SystemExit) as ctx:
                gate.parse_args(argv)

            self.assertEqual(ctx.exception.code, 2)

    def test_frame_counts_reject_invalid_values_at_parse_time(self) -> None:
        cases = [
            ["--frame-counts", ""],
            ["--frame-counts", "2,nope,8"],
            ["--frame-counts", "0,2,4"],
            ["--frame-counts=-1,2,4"],
            ["--frame-counts", "2,4,4"],
        ]
        for argv in cases:
            with self.subTest(argv=argv), self.assertRaises(SystemExit) as ctx:
                gate.parse_args(argv)

            self.assertEqual(ctx.exception.code, 2)

    def test_worldfoam_preflight_exit_without_artifact_is_labeled_contended(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            preflight_environment = {
                "status": "contended",
                "blocking_processes": [{"pid": 123, "pcpu": 91.0, "command": "python torch job"}],
            }
            with (
                mock.patch.object(gate, "_run_json", return_value=(2, preflight_environment)),
                mock.patch.object(gate, "_run") as run_mock,
            ):
                status = gate.main(
                    [
                        "--run-id",
                        "unit_preflight_blocked",
                        "--summary-json",
                        str(summary_path),
                    ]
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 2)
        self.assertEqual(summary["summary_schema_version"], gate.SUMMARY_SCHEMA_VERSION)
        self.assertEqual(summary["status"], "worldfoam_preflight_failed_or_contended")
        self.assertEqual(summary["worldfoam_preflight_returncode"], 2)
        self.assertEqual(summary["worldfoam_preflight_benchmark_environment_status"], "contended")
        self.assertEqual(summary["worldfoam_preflight_benchmark_environment"]["blocking_processes"][0]["pid"], 123)
        self.assertEqual(summary["worldfoam_preflight_blocking_processes"][0]["pid"], 123)
        self.assertTrue(summary["worldfoam_preflight_blocking_processes"][0]["high_cpu"])
        self.assertIsNone(summary["worldfoam_artifact"])
        self.assertIsNone(summary["worldfoam_promotable_artifact"])
        self.assertIsNone(summary["worldfoam_latest_written_artifact"])
        self.assertIn("unit_preflight_blocked.worldfoam.json", summary["worldfoam_latest_attempt_artifact"])
        self.assertFalse(summary["worldfoam_attempts"][0]["artifact_written"])
        self.assertFalse(summary["worldfoam_attempts"][0]["promotable"])
        self.assertEqual(summary["worldfoam_attempts"][0]["preflight_blocking_processes"][0]["pcpu"], 91.0)
        self.assertIsNone(summary["worldfoam_returncode"])
        self.assertIsNone(summary["worldfoam_status"])
        self.assertIsNone(summary["star_compare_command"])
        self.assertNotIn("star_compare_returncode", summary)
        run_mock.assert_not_called()

    def test_preflight_only_writes_environment_without_launching_worldfoam(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            preflight_environment = {
                "status": "contended",
                "blocking_processes": [{"pid": 456, "pcpu": 42.0, "command": "python busy"}],
            }
            with (
                mock.patch.object(gate, "_run_json", return_value=(2, preflight_environment)),
                mock.patch.object(gate, "_run") as run_mock,
            ):
                status = gate.main(
                    [
                        "--run-id",
                        "unit_preflight_only",
                        "--summary-json",
                        str(summary_path),
                        "--preflight-only",
                    ]
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 2)
        self.assertEqual(summary["summary_schema_version"], gate.SUMMARY_SCHEMA_VERSION)
        self.assertEqual(summary["status"], "worldfoam_preflight_failed_or_contended")
        self.assertEqual(summary["worldfoam_preflight_benchmark_environment_status"], "contended")
        self.assertEqual(summary["worldfoam_preflight_benchmark_environment"]["blocking_processes"][0]["pid"], 456)
        self.assertEqual(summary["worldfoam_preflight_blocking_processes"][0]["command"], "python busy")
        self.assertNotIn("worldfoam_returncode", summary)
        self.assertIsNone(summary["worldfoam_artifact"])
        self.assertIsNone(summary["worldfoam_latest_attempt_artifact"])
        self.assertIsNone(summary["worldfoam_latest_written_artifact"])
        self.assertIsNone(summary["star_compare_command"])
        self.assertEqual(summary["worldfoam_attempts"], [])
        run_mock.assert_not_called()

    def test_preflight_only_accepts_ok_environment_without_background_processes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            with (
                mock.patch.object(gate, "_run_json", return_value=(0, {"status": "ok"})),
                mock.patch.object(gate, "_run") as run_mock,
            ):
                status = gate.main(
                    [
                        "--run-id",
                        "unit_preflight_only_ok",
                        "--summary-json",
                        str(summary_path),
                        "--preflight-only",
                    ]
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertEqual(summary["status"], "preflight_ok")
        self.assertEqual(summary["worldfoam_preflight_benchmark_environment_status"], "ok")
        self.assertEqual(summary["worldfoam_attempts"], [])
        run_mock.assert_not_called()

    def test_blocking_process_summary_prioritizes_high_cpu_blockers(self) -> None:
        summary = gate._blocking_process_summary(
            {
                "blocking_cpu_threshold": 5.0,
                "blocking_processes": [
                    {"pid": 1, "pcpu": 0.0, "command": "torch parent"},
                    {"pid": 2, "pcpu": 12.5, "command": "python hot child"},
                    {"pid": 3, "pcpu": 7.0, "command": "git add"},
                ],
            }
        )

        self.assertEqual([row["pid"] for row in summary], [2, 3])
        self.assertTrue(all(row["high_cpu"] for row in summary))

    def test_preflight_stability_requires_consecutive_clean_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            summary_path = tmp_path / "summary.json"
            preflight_results = [
                (0, {"status": "background"}),
                (
                    2,
                    {
                        "status": "contended",
                        "blocking_processes": [{"pid": 808, "pcpu": 88.0, "command": "python late hot"}],
                    },
                ),
            ]

            with (
                mock.patch.object(gate, "RESULTS_DIR", tmp_path),
                mock.patch.object(gate, "_run_json", side_effect=preflight_results),
                mock.patch.object(gate, "_run") as run_mock,
                mock.patch.object(gate.time, "sleep") as sleep_mock,
            ):
                status = gate.main(
                    [
                        "--run-id",
                        "unit_stable_preflight_late_contention",
                        "--summary-json",
                        str(summary_path),
                        "--preflight-stability-samples",
                        "2",
                        "--preflight-stability-interval-s",
                        "0.25",
                    ]
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 2)
        self.assertEqual(summary["status"], "worldfoam_preflight_failed_or_contended")
        self.assertEqual(summary["worldfoam_preflight_stability_samples_required"], 2)
        self.assertEqual(summary["worldfoam_preflight_stability_interval_s"], 0.25)
        self.assertEqual(
            [sample["benchmark_environment_status"] for sample in summary["worldfoam_preflight_samples"]],
            ["background", "contended"],
        )
        self.assertEqual(summary["worldfoam_preflight_blocking_processes"][0]["pid"], 808)
        self.assertEqual(len(summary["worldfoam_attempts"]), 1)
        self.assertEqual(len(summary["worldfoam_attempts"][0]["preflight_samples"]), 2)
        self.assertIsNone(summary["worldfoam_artifact"])
        self.assertIsNone(summary["star_compare_command"])
        run_mock.assert_not_called()
        sleep_mock.assert_called_once_with(0.25)

    def test_preflight_stability_accepts_ok_and_background_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            summary_path = tmp_path / "summary.json"
            preflight_results = [
                (0, {"status": "ok"}),
                (0, {"status": "background", "background_processes": [{"pid": 42}]}),
            ]

            def fake_run(cmd: list[str], *, dry_run: bool) -> int:
                out_path = Path(cmd[cmd.index("--out-json") + 1])
                if "compare_star_uvt_worldfoam_scale.py" in cmd[1]:
                    out_path.write_text(
                        json.dumps({"status": "ok", "benchmark_environment": {"status": "ok"}}),
                        encoding="utf-8",
                    )
                    return 0
                out_path.write_text(
                    json.dumps(_accepted_worldfoam_payload("ok")),
                    encoding="utf-8",
                )
                return 0

            with (
                mock.patch.object(gate, "RESULTS_DIR", tmp_path),
                mock.patch.object(gate, "_run_json", side_effect=preflight_results),
                mock.patch.object(gate, "_run", side_effect=fake_run),
                mock.patch.object(gate.time, "sleep") as sleep_mock,
            ):
                status = gate.main(
                    [
                        "--run-id",
                        "unit_stable_preflight_ok_background",
                        "--summary-json",
                        str(summary_path),
                        "--preflight-stability-samples",
                        "2",
                        "--preflight-stability-interval-s",
                        "0.25",
                    ]
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(
            [sample["benchmark_environment_status"] for sample in summary["worldfoam_attempts"][0]["preflight_samples"]],
            ["ok", "background"],
        )
        self.assertTrue(summary["worldfoam_attempts"][0]["promotable"])
        self.assertEqual(summary["worldfoam_attempts"][0]["benchmark_environment_status"], "ok")
        self.assertTrue(summary["star_compare_attempts"][0]["promotable"])
        self.assertEqual(summary["star_compare_attempts"][0]["benchmark_environment_status"], "ok")
        sleep_mock.assert_called_once_with(0.25)

    def test_worldfoam_contended_artifact_is_not_promotable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            summary_path = tmp_path / "summary.json"

            def fake_run(cmd: list[str], *, dry_run: bool) -> int:
                out_path = Path(cmd[cmd.index("--out-json") + 1])
                out_path.write_text(
                    json.dumps(
                        {
                            "status": "ok",
                            "benchmark_environment": {"status": "contended"},
                            "rows": [],
                            "acceptance": {"all_rows_ok": True},
                        }
                    ),
                    encoding="utf-8",
                )
                return 2

            with (
                mock.patch.object(gate, "RESULTS_DIR", tmp_path),
                mock.patch.object(gate, "_run_json", return_value=(0, {"status": "background"})),
                mock.patch.object(gate, "_run", side_effect=fake_run) as run_mock,
            ):
                status = gate.main(
                    [
                        "--run-id",
                        "unit_contended_artifact",
                        "--summary-json",
                        str(summary_path),
                    ]
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 2)
        self.assertEqual(summary["status"], "worldfoam_not_promotable")
        self.assertEqual(summary["worldfoam_returncode"], 2)
        self.assertEqual(summary["worldfoam_status"], "ok")
        self.assertEqual(summary["worldfoam_benchmark_environment_status"], "contended")
        self.assertIsNone(summary["worldfoam_artifact"])
        self.assertIsNone(summary["worldfoam_promotable_artifact"])
        self.assertIn("unit_contended_artifact.worldfoam.json", summary["worldfoam_latest_written_artifact"])
        self.assertIsNone(summary["star_compare_artifact"])
        self.assertIsNone(summary["star_compare_latest_attempt_artifact"])
        self.assertIsNone(summary["star_compare_latest_written_artifact"])
        self.assertTrue(summary["worldfoam_attempts"][0]["artifact_written"])
        self.assertFalse(summary["worldfoam_attempts"][0]["promotable"])
        self.assertIsNone(summary["star_compare_command"])
        run_mock.assert_called_once()

    def test_worldfoam_missing_environment_artifact_is_not_promotable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            summary_path = tmp_path / "summary.json"

            def fake_run(cmd: list[str], *, dry_run: bool) -> int:
                out_path = Path(cmd[cmd.index("--out-json") + 1])
                out_path.write_text(json.dumps(_accepted_worldfoam_payload(None)), encoding="utf-8")
                return 0

            with (
                mock.patch.object(gate, "RESULTS_DIR", tmp_path),
                mock.patch.object(gate, "_run_json", return_value=(0, {"status": "background"})),
                mock.patch.object(gate, "_run", side_effect=fake_run) as run_mock,
            ):
                status = gate.main(
                    [
                        "--run-id",
                        "unit_missing_worldfoam_environment",
                        "--summary-json",
                        str(summary_path),
                    ]
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 2)
        self.assertEqual(summary["status"], "worldfoam_not_promotable")
        self.assertEqual(summary["worldfoam_returncode"], 0)
        self.assertEqual(summary["worldfoam_status"], "ok")
        self.assertIsNone(summary["worldfoam_benchmark_environment_status"])
        self.assertIsNone(summary["worldfoam_artifact"])
        self.assertIsNone(summary["worldfoam_promotable_artifact"])
        self.assertTrue(summary["worldfoam_attempts"][0]["artifact_written"])
        self.assertFalse(summary["worldfoam_attempts"][0]["promotable"])
        self.assertIsNone(summary["worldfoam_attempts"][0]["benchmark_environment_status"])
        self.assertIsNone(summary["star_compare_command"])
        run_mock.assert_called_once()

    def test_worldfoam_missing_acceptance_artifact_is_not_promotable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            summary_path = tmp_path / "summary.json"

            def fake_run(cmd: list[str], *, dry_run: bool) -> int:
                out_path = Path(cmd[cmd.index("--out-json") + 1])
                out_path.write_text(
                    json.dumps({"status": "ok", "benchmark_environment": {"status": "background"}, "rows": []}),
                    encoding="utf-8",
                )
                return 0

            with (
                mock.patch.object(gate, "RESULTS_DIR", tmp_path),
                mock.patch.object(gate, "_run_json", return_value=(0, {"status": "background"})),
                mock.patch.object(gate, "_run", side_effect=fake_run) as run_mock,
            ):
                status = gate.main(
                    [
                        "--run-id",
                        "unit_missing_worldfoam_acceptance",
                        "--summary-json",
                        str(summary_path),
                    ]
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 2)
        self.assertEqual(summary["status"], "worldfoam_not_promotable")
        self.assertEqual(summary["worldfoam_returncode"], 0)
        self.assertEqual(summary["worldfoam_status"], "ok")
        self.assertEqual(summary["worldfoam_benchmark_environment_status"], "background")
        self.assertFalse(summary["worldfoam_acceptance_ok"])
        self.assertEqual(summary["worldfoam_acceptance_failures"], ["WorldFoam artifact acceptance is missing"])
        self.assertIsNone(summary["worldfoam_artifact"])
        self.assertIsNone(summary["worldfoam_promotable_artifact"])
        self.assertTrue(summary["worldfoam_attempts"][0]["artifact_written"])
        self.assertFalse(summary["worldfoam_attempts"][0]["promotable"])
        self.assertFalse(summary["worldfoam_attempts"][0]["acceptance_ok"])
        self.assertEqual(
            summary["worldfoam_attempts"][0]["acceptance_failures"],
            ["WorldFoam artifact acceptance is missing"],
        )
        self.assertIsNone(summary["star_compare_command"])
        run_mock.assert_called_once()

    def test_retries_contended_worldfoam_then_runs_star_with_clean_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            summary_path = tmp_path / "summary.json"
            calls: list[list[str]] = []

            def fake_run(cmd: list[str], *, dry_run: bool) -> int:
                calls.append(cmd)
                out_path = Path(cmd[cmd.index("--out-json") + 1])
                if "compare_star_uvt_worldfoam_scale.py" in cmd[1]:
                    out_path.write_text(
                        json.dumps({"status": "ok", "benchmark_environment": {"status": "background"}}),
                        encoding="utf-8",
                    )
                    return 0
                environment = "contended" if len(calls) == 1 else "background"
                out_path.write_text(
                    json.dumps(_accepted_worldfoam_payload(environment)),
                    encoding="utf-8",
                )
                return 2 if environment == "contended" else 0

            with (
                mock.patch.object(gate, "RESULTS_DIR", tmp_path),
                mock.patch.object(gate, "_run_json", return_value=(0, {"status": "background"})),
                mock.patch.object(gate, "_run", side_effect=fake_run),
            ):
                status = gate.main(
                    [
                        "--run-id",
                        "unit_retry_contended",
                        "--summary-json",
                        str(summary_path),
                        "--max-worldfoam-attempts",
                        "2",
                    ]
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertEqual(summary["summary_schema_version"], gate.SUMMARY_SCHEMA_VERSION)
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(len(summary["worldfoam_attempts"]), 2)
        self.assertEqual(summary["worldfoam_attempts"][0]["benchmark_environment_status"], "contended")
        self.assertEqual(summary["worldfoam_attempts"][1]["benchmark_environment_status"], "background")
        self.assertFalse(summary["worldfoam_attempts"][0]["promotable"])
        self.assertTrue(summary["worldfoam_attempts"][1]["promotable"])
        self.assertTrue(summary["worldfoam_attempts"][1]["artifact_written"])
        self.assertIn("unit_retry_contended.attempt2.worldfoam.json", summary["worldfoam_artifact"])
        self.assertEqual(summary["worldfoam_promotable_artifact"], summary["worldfoam_artifact"])
        self.assertTrue(
            any("unit_retry_contended.attempt2.worldfoam.json" in item for item in summary["star_compare_command"])
        )
        self.assertTrue(
            any(
                "unit_retry_contended.attempt2.worldfoam.json" in item
                for item in summary["planned_star_compare_command"]
            )
        )
        self.assertFalse(
            any(
                "unit_retry_contended.attempt1.worldfoam.json" in item
                for item in summary["planned_star_compare_command"]
            )
        )
        self.assertEqual(len(calls), 3)

    def test_verify_promotion_runs_after_successful_star_compare(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            summary_path = tmp_path / "summary.json"

            def fake_run_json(cmd: list[str], *, dry_run: bool) -> tuple[int, dict[str, object]]:
                if cmd[1] == str(gate.VERIFY_PROMOTION):
                    return 0, {"status": "ok", "failures": []}
                return 0, {"status": "background"}

            def fake_run(cmd: list[str], *, dry_run: bool) -> int:
                out_path = Path(cmd[cmd.index("--out-json") + 1])
                if "compare_star_uvt_worldfoam_scale.py" in cmd[1]:
                    out_path.write_text(
                        json.dumps({"status": "ok", "benchmark_environment": {"status": "background"}}),
                        encoding="utf-8",
                    )
                    return 0
                out_path.write_text(
                    json.dumps(_accepted_worldfoam_payload("background")),
                    encoding="utf-8",
                )
                return 0

            with (
                mock.patch.object(gate, "RESULTS_DIR", tmp_path),
                mock.patch.object(gate, "_run_json", side_effect=fake_run_json),
                mock.patch.object(gate, "_run", side_effect=fake_run),
            ):
                status = gate.main(
                    [
                        "--run-id",
                        "unit_verified_promotion",
                        "--summary-json",
                        str(summary_path),
                        "--verify-promotion",
                    ]
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["promotion_verifier_returncode"], 0)
        self.assertEqual(summary["promotion_verifier_status"], "ok")
        self.assertEqual(summary["promotion_verifier_failures"], [])
        self.assertIn(str(summary_path), summary["promotion_verifier_command"])

    def test_verify_promotion_failure_changes_final_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            summary_path = tmp_path / "summary.json"

            def fake_run_json(cmd: list[str], *, dry_run: bool) -> tuple[int, dict[str, object]]:
                if cmd[1] == str(gate.VERIFY_PROMOTION):
                    return 2, {"status": "failed", "failures": ["not clean"]}
                return 0, {"status": "background"}

            def fake_run(cmd: list[str], *, dry_run: bool) -> int:
                out_path = Path(cmd[cmd.index("--out-json") + 1])
                out_path.write_text(
                    json.dumps(_accepted_worldfoam_payload("background")),
                    encoding="utf-8",
                )
                return 0

            with (
                mock.patch.object(gate, "RESULTS_DIR", tmp_path),
                mock.patch.object(gate, "_run_json", side_effect=fake_run_json),
                mock.patch.object(gate, "_run", side_effect=fake_run),
            ):
                status = gate.main(
                    [
                        "--run-id",
                        "unit_verified_promotion_fails",
                        "--summary-json",
                        str(summary_path),
                        "--verify-promotion",
                    ]
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 1)
        self.assertEqual(summary["status"], "promotion_verification_failed")
        self.assertEqual(summary["promotion_verifier_returncode"], 2)
        self.assertEqual(summary["promotion_verifier_status"], "failed")
        self.assertEqual(summary["promotion_verifier_failures"], ["not clean"])

    def test_star_missing_environment_cannot_promote_even_when_star_status_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            summary_path = tmp_path / "summary.json"

            def fake_run(cmd: list[str], *, dry_run: bool) -> int:
                out_path = Path(cmd[cmd.index("--out-json") + 1])
                if "compare_star_uvt_worldfoam_scale.py" in cmd[1]:
                    out_path.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
                    return 0
                out_path.write_text(
                    json.dumps(_accepted_worldfoam_payload("background")),
                    encoding="utf-8",
                )
                return 0

            with (
                mock.patch.object(gate, "RESULTS_DIR", tmp_path),
                mock.patch.object(gate, "_run_json", return_value=(0, {"status": "background"})),
                mock.patch.object(gate, "_run", side_effect=fake_run),
            ):
                status = gate.main(
                    [
                        "--run-id",
                        "unit_missing_star_environment",
                        "--summary-json",
                        str(summary_path),
                        "--verify-promotion",
                    ]
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 1)
        self.assertEqual(summary["status"], "star_compare_failed")
        self.assertEqual(summary["star_compare_returncode"], 0)
        self.assertEqual(summary["star_compare_status"], "ok")
        self.assertIsNone(summary["star_compare_benchmark_environment_status"])
        self.assertNotIn("promotion_verifier_returncode", summary)
        self.assertIn("unit_missing_star_environment.worldfoam.json", summary["worldfoam_artifact"])
        self.assertIsNone(summary["star_compare_artifact"])
        self.assertIn("unit_missing_star_environment.star_compare.json", summary["star_compare_latest_attempt_artifact"])
        self.assertIn("unit_missing_star_environment.star_compare.json", summary["star_compare_latest_written_artifact"])
        self.assertIsNotNone(summary["star_compare_command"])

    def test_retries_contended_star_compare_without_rerunning_worldfoam(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            summary_path = tmp_path / "summary.json"
            calls: list[str] = []

            def fake_run(cmd: list[str], *, dry_run: bool) -> int:
                out_path = Path(cmd[cmd.index("--out-json") + 1])
                if "compare_star_uvt_worldfoam_scale.py" in cmd[1]:
                    calls.append("star")
                    environment = "contended" if calls.count("star") == 1 else "background"
                    out_path.write_text(
                        json.dumps(
                            {
                                "status": "failed" if environment == "contended" else "ok",
                                "failures": ["benchmark environment became contended during STAR run"]
                                if environment == "contended"
                                else [],
                                "benchmark_environment": {"status": environment},
                                "star": {"summary": {"status": "ok"}},
                                "worldfoam": {
                                    "artifact": str(tmp_path / "unit_star_retry.worldfoam.json"),
                                    "summary": {
                                        "status": "ok",
                                        "benchmark_environment_status": "background",
                                    },
                                },
                            }
                        ),
                        encoding="utf-8",
                    )
                    return 1 if environment == "contended" else 0
                calls.append("worldfoam")
                out_path.write_text(
                    json.dumps(_accepted_worldfoam_payload("background")),
                    encoding="utf-8",
                )
                return 0

            with (
                mock.patch.object(gate, "RESULTS_DIR", tmp_path),
                mock.patch.object(gate, "_run_json", return_value=(0, {"status": "background"})),
                mock.patch.object(gate, "_run", side_effect=fake_run),
            ):
                status = gate.main(
                    [
                        "--run-id",
                        "unit_star_retry",
                        "--summary-json",
                        str(summary_path),
                        "--max-star-attempts",
                        "2",
                    ]
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertEqual(calls, ["worldfoam", "star", "star"])
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(len(summary["worldfoam_attempts"]), 1)
        self.assertEqual(len(summary["star_compare_attempts"]), 2)
        self.assertFalse(summary["star_compare_attempts"][0]["promotable"])
        self.assertEqual(summary["star_compare_attempts"][0]["benchmark_environment_status"], "contended")
        self.assertTrue(summary["star_compare_attempts"][1]["promotable"])
        self.assertEqual(summary["star_compare_attempts"][1]["benchmark_environment_status"], "background")
        self.assertIn("unit_star_retry.star_attempt2.star_compare.json", summary["star_compare_artifact"])
        self.assertIn("unit_star_retry.star_attempt2.star_compare.json", summary["star_compare_latest_attempt_artifact"])
        self.assertIn("unit_star_retry.star_attempt2.star_compare.json", summary["star_compare_latest_written_artifact"])
        self.assertIn("unit_star_retry.star_attempt2.star_compare.json", summary["star_compare_command"][-1])

    def test_contended_artifact_then_preflight_failure_does_not_select_unwritten_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            summary_path = tmp_path / "summary.json"

            def fake_run(cmd: list[str], *, dry_run: bool) -> int:
                out_path = Path(cmd[cmd.index("--out-json") + 1])
                out_path.write_text(
                    json.dumps(_accepted_worldfoam_payload("contended")),
                    encoding="utf-8",
                )
                return 2

            preflight_results = [
                (0, {"status": "background"}),
                (
                    2,
                    {
                        "status": "contended",
                        "blocking_processes": [{"pid": 999, "pcpu": 44.0, "command": "python hot"}],
                    },
                ),
            ]

            with (
                mock.patch.object(gate, "RESULTS_DIR", tmp_path),
                mock.patch.object(gate, "_run_json", side_effect=preflight_results),
                mock.patch.object(gate, "_run", side_effect=fake_run) as run_mock,
            ):
                status = gate.main(
                    [
                        "--run-id",
                        "unit_contended_then_preflight_failed",
                        "--summary-json",
                        str(summary_path),
                        "--max-worldfoam-attempts",
                        "2",
                    ]
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 2)
        self.assertEqual(summary["status"], "worldfoam_preflight_failed_or_contended")
        self.assertIsNone(summary["worldfoam_artifact"])
        self.assertIsNone(summary["worldfoam_promotable_artifact"])
        self.assertIn("attempt1.worldfoam.json", summary["worldfoam_latest_written_artifact"])
        self.assertIn("attempt2.worldfoam.json", summary["worldfoam_latest_attempt_artifact"])
        self.assertEqual(len(summary["worldfoam_attempts"]), 2)
        self.assertTrue(summary["worldfoam_attempts"][0]["artifact_written"])
        self.assertFalse(summary["worldfoam_attempts"][0]["promotable"])
        self.assertFalse(summary["worldfoam_attempts"][1]["artifact_written"])
        self.assertFalse(summary["worldfoam_attempts"][1]["promotable"])
        self.assertEqual(summary["worldfoam_preflight_blocking_processes"][0]["pid"], 999)
        self.assertIsNone(summary["star_compare_artifact"])
        self.assertIsNone(summary["star_compare_latest_attempt_artifact"])
        self.assertIsNone(summary["star_compare_latest_written_artifact"])
        self.assertIsNone(summary["star_compare_command"])
        self.assertTrue(
            any(
                "unit_contended_then_preflight_failed.attempt1.worldfoam.json" in item
                for item in summary["planned_star_compare_command"]
            )
        )
        run_mock.assert_called_once()

    def test_all_retry_preflights_fail_without_selecting_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            summary_path = tmp_path / "summary.json"
            preflight_results = [
                (
                    2,
                    {
                        "status": "contended",
                        "blocking_processes": [{"pid": 101, "pcpu": 55.0, "command": "python hot one"}],
                    },
                ),
                (
                    2,
                    {
                        "status": "contended",
                        "blocking_processes": [{"pid": 202, "pcpu": 66.0, "command": "python hot two"}],
                    },
                ),
                (
                    2,
                    {
                        "status": "contended",
                        "blocking_processes": [{"pid": 303, "pcpu": 77.0, "command": "python hot three"}],
                    },
                ),
            ]

            with (
                mock.patch.object(gate, "RESULTS_DIR", tmp_path),
                mock.patch.object(gate, "_run_json", side_effect=preflight_results),
                mock.patch.object(gate, "_run") as run_mock,
            ):
                status = gate.main(
                    [
                        "--run-id",
                        "unit_all_preflights_fail",
                        "--summary-json",
                        str(summary_path),
                        "--max-worldfoam-attempts",
                        "3",
                    ]
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 2)
        self.assertEqual(summary["summary_schema_version"], gate.SUMMARY_SCHEMA_VERSION)
        self.assertEqual(summary["status"], "worldfoam_preflight_failed_or_contended")
        self.assertIsNone(summary["worldfoam_artifact"])
        self.assertIsNone(summary["worldfoam_promotable_artifact"])
        self.assertIsNone(summary["worldfoam_latest_written_artifact"])
        self.assertIn("unit_all_preflights_fail.attempt3.worldfoam.json", summary["worldfoam_latest_attempt_artifact"])
        self.assertEqual(len(summary["worldfoam_attempts"]), 3)
        self.assertEqual([attempt["attempt_index"] for attempt in summary["worldfoam_attempts"]], [1, 2, 3])
        self.assertFalse(any(attempt["artifact_written"] for attempt in summary["worldfoam_attempts"]))
        self.assertFalse(any(attempt["promotable"] for attempt in summary["worldfoam_attempts"]))
        self.assertEqual(summary["worldfoam_preflight_blocking_processes"][0]["pid"], 303)
        self.assertIsNone(summary["worldfoam_returncode"])
        self.assertIsNone(summary["worldfoam_status"])
        self.assertIsNone(summary["worldfoam_benchmark_environment_status"])
        self.assertIsNone(summary["star_compare_command"])
        self.assertNotIn("star_compare_returncode", summary)
        run_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
