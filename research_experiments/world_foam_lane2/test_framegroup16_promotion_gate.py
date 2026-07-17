from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import run_framegroup16_promotion_gate as promotion_gate


def _native_verify_result(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "status": "ok",
        **promotion_gate.REQUIRED_NATIVE_PACKED_VERIFY_VALUES,
    }
    result.update(overrides)
    for key, value in tuple(result.items()):
        if key.endswith(("_i32", "_i64")) and isinstance(value, list):
            result.setdefault(f"{key}_dtype", "int32" if key.endswith("_i32") else "int64")
            result.setdefault(f"{key}_device", "cpu")
            result.setdefault(f"{key}_shape", [len(value)])
            result.setdefault(f"{key}_contiguous", True)
    return result


class Framegroup16PromotionGateTests(unittest.TestCase):
    def test_load_verify_brief_extracts_row_medians(self) -> None:
        payload = {
            "status": "failed",
            "clean_speedscale_artifact": False,
            "promoted_path_not_regressed": False,
            "expected_payload_bools": {"experimental_rowdesc_launch_only_packed_delta": True},
            "storage_scale": 1.03,
            "topology_storage_scale": 1.08,
            "coeff_storage_scale": 1.0,
            "mps_resident_storage_scale": 1.02,
            "mps_resident_noncoeff_storage_scale": 1.04,
            "mps_resident_coeff_storage_scale": 1.0,
            "contamination": ["benchmark_environment status is 'contended'"],
            "failures": ["16f total median 8.0 ms is slow"],
            "rows": {
                "16": {
                    "total": {"median_ms": 8.0},
                    "backward": {"median_ms": 6.5},
                    "storage_bytes": 1000,
                    "topology_storage_bytes": 120,
                    "coeff_storage_bytes": 800,
                    "mps_resident_storage_bytes": 1012,
                    "mps_resident_noncoeff_storage_bytes": 212,
                    "mps_resident_coeff_storage_bytes": 800,
                    "heldout_psnr": 14.2,
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "verify.json"
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

            brief = promotion_gate._load_verify_brief(path)

        self.assertIsNotNone(brief)
        assert brief is not None
        self.assertEqual(brief["status"], "failed")
        self.assertEqual(
            brief["expected_payload_bools"],
            {"experimental_rowdesc_launch_only_packed_delta": True},
        )
        self.assertEqual(brief["storage_scale"], 1.03)
        self.assertEqual(brief["topology_storage_scale"], 1.08)
        self.assertEqual(brief["coeff_storage_scale"], 1.0)
        self.assertEqual(brief["mps_resident_storage_scale"], 1.02)
        self.assertEqual(brief["mps_resident_noncoeff_storage_scale"], 1.04)
        self.assertEqual(brief["mps_resident_coeff_storage_scale"], 1.0)
        self.assertEqual(brief["contamination"], ["benchmark_environment status is 'contended'"])
        self.assertEqual(brief["rows"]["16"]["total_median_ms"], 8.0)
        self.assertEqual(brief["rows"]["16"]["backward_median_ms"], 6.5)
        self.assertEqual(brief["rows"]["16"]["storage_bytes"], 1000)
        self.assertEqual(brief["rows"]["16"]["topology_storage_bytes"], 120)
        self.assertEqual(brief["rows"]["16"]["coeff_storage_bytes"], 800)
        self.assertEqual(brief["rows"]["16"]["mps_resident_storage_bytes"], 1012)
        self.assertEqual(brief["rows"]["16"]["mps_resident_noncoeff_storage_bytes"], 212)
        self.assertEqual(brief["rows"]["16"]["mps_resident_coeff_storage_bytes"], 800)
        self.assertEqual(brief["rows"]["16"]["heldout_psnr"], 14.2)

    def test_run_preflight_writes_live_summary_on_blocked_attempt(self) -> None:
        command = [
            sys.executable,
            "-c",
            "import json, sys; print(json.dumps({'status':'contended','blocking_processes':[{'pid':1,'pcpu':99.0,'command':'python busy.py'}]})); sys.exit(2)",
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            summary: dict[str, object] = {"run_id": "unit"}

            status, attempts = promotion_gate._run_preflight(
                command,
                dry_run=False,
                wait=False,
                timeout_s=0.0,
                interval_s=0.1,
                stable_checks=1,
                summary=summary,
                summary_path=summary_path,
            )
            written = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 2)
        self.assertEqual(len(attempts), 1)
        self.assertEqual(written["status"], "preflight_checked")
        self.assertEqual(written["preflight_status"], 2)
        self.assertEqual(written["preflight_attempts"][0]["status"], "contended")
        self.assertEqual(written["preflight_attempts"][0]["top_blocking_process"]["pcpu"], 99.0)

    def test_run_preflight_requires_consecutive_stable_successes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            sequence_path = tmpdir_path / "sequence.json"
            sequence_path.write_text(json.dumps(["background", "contended", "background", "background"]), encoding="utf-8")
            summary_path = tmpdir_path / "summary.json"
            command = [
                sys.executable,
                "-c",
                (
                    "import json, pathlib, sys; "
                    "p=pathlib.Path(sys.argv[1]); "
                    "seq=json.loads(p.read_text()); "
                    "status=seq.pop(0); "
                    "p.write_text(json.dumps(seq)); "
                    "payload={'status': status}; "
                    "code=0 if status == 'background' else 2; "
                    "payload['blocking_processes']=[] if code == 0 else "
                    "[{'pid': 7, 'pcpu': 77.0, 'command': 'python busy.py'}]; "
                    "print(json.dumps(payload)); "
                    "raise SystemExit(code)"
                ),
                str(sequence_path),
            ]
            summary: dict[str, object] = {"run_id": "unit_stable"}

            status, attempts = promotion_gate._run_preflight(
                command,
                dry_run=False,
                wait=True,
                timeout_s=5.0,
                interval_s=0.1,
                stable_checks=2,
                summary=summary,
                summary_path=summary_path,
            )
            written = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertEqual(len(attempts), 4)
        self.assertEqual([attempt["success_streak"] for attempt in attempts], [1, 0, 1, 2])
        self.assertEqual(written["status"], "preflight_checked")
        self.assertEqual(written["preflight_required_success_streak"], 2)
        self.assertEqual(written["preflight_current_success_streak"], 2)
        self.assertEqual(written["preflight_max_success_streak"], 2)
        self.assertEqual(written["preflight_attempts"][-1]["success_streak"], 2)

    def test_run_preflight_fails_when_stability_required_without_wait(self) -> None:
        command = [
            sys.executable,
            "-c",
            "import json; print(json.dumps({'status':'background','blocking_processes':[]}))",
        ]
        status, attempts = promotion_gate._run_preflight(
            command,
            dry_run=False,
            wait=False,
            timeout_s=0.0,
            interval_s=0.1,
            stable_checks=2,
        )

        self.assertEqual(status, 2)
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]["success_streak"], 1)

    def test_preflight_failure_summary_explains_missing_stable_streak(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            attempts: list[dict[str, object]] = [
                {"returncode": 0, "status": "background", "success_streak": 1},
            ]
            with (
                mock.patch.object(promotion_gate, "_run_preflight", return_value=(2, attempts)),
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "run_framegroup16_promotion_gate.py",
                        "--run-id",
                        "unit_stable_preflight_summary",
                        "--summary-json",
                        str(summary_path),
                        "--stable-preflight-checks",
                        "2",
                    ],
                ),
            ):
                status = promotion_gate.main()
            written = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 2)
        self.assertEqual(written["status"], "preflight_failed")
        self.assertEqual(written["preflight_failure_reason"], "stable_preflight_streak_not_reached")
        self.assertEqual(written["preflight_required_success_streak"], 2)
        self.assertEqual(written["preflight_current_success_streak"], 1)
        self.assertEqual(written["preflight_max_success_streak"], 1)

    def test_preflight_failure_reason_reports_never_clean(self) -> None:
        reason = promotion_gate._preflight_failure_reason(
            2,
            [{"returncode": 2, "status": "contended", "success_streak": 0}],
            required_successes=2,
        )

        self.assertEqual(reason, "benchmark_environment_never_clean")

    def test_preexisting_output_artifact_fails_before_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            out_json = tmpdir_path / "existing.json"
            out_json.write_text("{}\n", encoding="utf-8")
            summary_path = tmpdir_path / "summary.json"
            with (
                mock.patch.object(promotion_gate, "_run_preflight") as preflight,
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "run_framegroup16_promotion_gate.py",
                        "--run-id",
                        "unit_existing_artifact",
                        "--out-json",
                        str(out_json),
                        "--summary-json",
                        str(summary_path),
                    ],
                ),
            ):
                status = promotion_gate.main()
            written = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 2)
        self.assertEqual(written["status"], "config_failed")
        self.assertIn(str(out_json), written["preexisting_output_artifacts"])
        self.assertIn("pre-existing output artifacts", written["config_failures"][0])
        self.assertNotIn("preflight_status", written)
        preflight.assert_not_called()

    def test_preexisting_output_artifact_can_be_overwritten_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            out_json = tmpdir_path / "existing.json"
            out_json.write_text("{}\n", encoding="utf-8")
            summary_path = tmpdir_path / "summary.json"
            with mock.patch.object(
                sys,
                "argv",
                [
                    "run_framegroup16_promotion_gate.py",
                    "--run-id",
                    "unit_existing_artifact_allowed",
                    "--out-json",
                    str(out_json),
                    "--summary-json",
                    str(summary_path),
                    "--allow-overwrite-artifacts",
                    "--dry-run",
                ],
            ):
                status = promotion_gate.main()
            written = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertEqual(written["status"], "ok")
        self.assertTrue(written["allow_overwrite_artifacts"])
        self.assertIn(str(out_json), written["preexisting_output_artifacts"])

    def test_native_variant_flags_are_forwarded_in_stable_order(self) -> None:
        args = promotion_gate.build_parser().parse_args(
            [
                "--experimental-native-sorted-delta",
                "--experimental-native-emitted-pack-records",
            ]
        )

        flags = promotion_gate._enabled_native_variant_flags(args)

        self.assertEqual(
            flags,
            [
                "--experimental-native-sorted-delta",
                "--experimental-native-emitted-pack-records",
            ],
        )

    def test_train_variant_flags_include_packed_device_diagnostics(self) -> None:
        args = promotion_gate.build_parser().parse_args(
            [
                "--experimental-native-sorted-delta",
                "--experimental-minimal-packed-delta-device",
                "--experimental-kernel-order-packed-delta-device",
                "--experimental-smallrun16-packed-delta",
                "--experimental-launch-only-packed-delta",
                "--experimental-unchecked-launch-only-packed-delta",
                "--experimental-reduce32-launch-only-packed-delta",
                "--experimental-rowselect32-launch-only-packed-delta",
                "--experimental-rowdesc-launch-only-packed-delta",
                "--experimental-rowdesc32-launch-only-packed-delta",
                "--experimental-cpu-rebase-delta",
                "--experimental-native-emitted-pack-records",
            ]
        )

        flags = promotion_gate._enabled_train_variant_flags(args)

        self.assertEqual(
            flags,
            [
                "--experimental-native-sorted-delta",
                "--experimental-native-emitted-pack-records",
                "--experimental-minimal-packed-delta-device",
                "--experimental-kernel-order-packed-delta-device",
                "--experimental-smallrun16-packed-delta",
                "--experimental-launch-only-packed-delta",
                "--experimental-unchecked-launch-only-packed-delta",
                "--experimental-reduce32-launch-only-packed-delta",
                "--experimental-rowselect32-launch-only-packed-delta",
                "--experimental-rowdesc-launch-only-packed-delta",
                "--experimental-rowdesc32-launch-only-packed-delta",
                "--experimental-cpu-rebase-delta",
            ],
        )

    def test_default_promotion_requires_two_stable_preflights(self) -> None:
        args = promotion_gate.build_parser().parse_args([])

        self.assertEqual(args.stable_preflight_checks, 2)

    def test_sorted_native_emitted_pack_records_verifies_extension_before_launch(self) -> None:
        native_verify_result = _native_verify_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            with (
                mock.patch.object(promotion_gate, "_run_json_command", return_value=(0, native_verify_result)),
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "run_framegroup16_promotion_gate.py",
                        "--run-id",
                        "unit_native_extension_verify",
                        "--summary-json",
                        str(summary_path),
                        "--experimental-native-sorted-delta",
                        "--experimental-native-emitted-pack-records",
                        "--dry-run",
                    ],
                ),
            ):
                status = promotion_gate.main()
            written = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertEqual(written["status"], "ok")
        self.assertEqual(written["native_packed_extension_verify_status"], 0)
        self.assertEqual(written["native_packed_extension_verify_result"], native_verify_result)
        self.assertEqual(
            written["native_packed_extension_verify_result"]["changing_sorted_change_record_i32"],
            [2097153, 1049088],
        )
        self.assertEqual(
            written["native_packed_extension_verify_result"]["changing_cut_change_record_i32"],
            [2097153, 1049088],
        )
        self.assertTrue(
            written["native_packed_extension_verify_command"][-1].endswith("verify_native_packed_extension.py")
        )

    def test_cutprep_native_emitted_pack_records_verifies_extension_before_launch(self) -> None:
        native_verify_result = _native_verify_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            with (
                mock.patch.object(promotion_gate, "_run_json_command", return_value=(0, native_verify_result)),
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "run_framegroup16_promotion_gate.py",
                        "--run-id",
                        "unit_native_cutprep_extension_verify",
                        "--summary-json",
                        str(summary_path),
                        "--experimental-native-cut-prep-delta",
                        "--experimental-native-emitted-pack-records",
                        "--dry-run",
                    ],
                ),
            ):
                status = promotion_gate.main()
            written = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertEqual(written["status"], "ok")
        self.assertEqual(written["native_packed_extension_verify_status"], 0)
        self.assertEqual(written["native_packed_extension_verify_result"], native_verify_result)
        self.assertEqual(
            written["native_packed_extension_verify_result"]["changing_cut_change_record_i32"],
            [2097153, 1049088],
        )
        self.assertTrue(
            written["native_packed_extension_verify_command"][-1].endswith("verify_native_packed_extension.py")
        )

    def test_emitted_pack_records_verifies_extension_before_launch(self) -> None:
        native_verify_result = _native_verify_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            with (
                mock.patch.object(promotion_gate, "_run_json_command", return_value=(0, native_verify_result)),
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "run_framegroup16_promotion_gate.py",
                        "--run-id",
                        "unit_emitted_pack_extension_verify",
                        "--summary-json",
                        str(summary_path),
                        "--experimental-native-emitted-pack-records",
                        "--dry-run",
                    ],
                ),
            ):
                status = promotion_gate.main()
            written = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertEqual(written["status"], "ok")
        self.assertEqual(written["native_packed_extension_verify_status"], 0)
        self.assertEqual(written["native_packed_extension_verify_result"], native_verify_result)
        self.assertEqual(
            written["native_packed_extension_verify_result"]["changing_cut_change_record_i32"],
            [2097153, 1049088],
        )
        self.assertTrue(
            written["native_packed_extension_verify_command"][-1].endswith("verify_native_packed_extension.py")
        )

    def test_native_pack_records_verifies_extension_before_launch(self) -> None:
        native_verify_result = _native_verify_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            with (
                mock.patch.object(promotion_gate, "_run_json_command", return_value=(0, native_verify_result)),
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "run_framegroup16_promotion_gate.py",
                        "--run-id",
                        "unit_native_pack_extension_verify",
                        "--summary-json",
                        str(summary_path),
                        "--experimental-native-pack-records",
                        "--dry-run",
                    ],
                ),
            ):
                status = promotion_gate.main()
            written = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertEqual(written["status"], "ok")
        self.assertEqual(written["native_packed_extension_verify_status"], 0)
        self.assertEqual(written["native_packed_extension_verify_result"], native_verify_result)
        self.assertEqual(
            written["native_packed_extension_verify_result"]["changing_sorted_change_record_i32"],
            [2097153, 1049088],
        )
        self.assertTrue(
            written["native_packed_extension_verify_command"][-1].endswith("verify_native_packed_extension.py")
        )

    def test_native_extension_verify_failure_stops_before_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            with (
                mock.patch.object(promotion_gate, "_run_json_command", return_value=(9, {"status": "failed"})),
                mock.patch.object(promotion_gate, "_run_preflight") as preflight,
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "run_framegroup16_promotion_gate.py",
                        "--run-id",
                        "unit_native_extension_verify_fail",
                        "--summary-json",
                        str(summary_path),
                        "--experimental-native-sorted-delta",
                        "--experimental-native-emitted-pack-records",
                    ],
                ),
            ):
                status = promotion_gate.main()
            written = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 9)
        self.assertEqual(written["status"], "native_packed_extension_verify_failed")
        self.assertEqual(written["native_packed_extension_verify_result"], {"status": "failed"})
        preflight.assert_not_called()

    def test_native_pack_records_verify_failure_stops_before_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            with (
                mock.patch.object(promotion_gate, "_run_json_command", return_value=(9, {"status": "failed"})),
                mock.patch.object(promotion_gate, "_run_preflight") as preflight,
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "run_framegroup16_promotion_gate.py",
                        "--run-id",
                        "unit_native_pack_extension_verify_fail",
                        "--summary-json",
                        str(summary_path),
                        "--experimental-native-pack-records",
                    ],
                ),
            ):
                status = promotion_gate.main()
            written = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 9)
        self.assertEqual(written["status"], "native_packed_extension_verify_failed")
        self.assertEqual(written["native_packed_extension_verify_result"], {"status": "failed"})
        preflight.assert_not_called()

    def test_native_verify_zero_status_bad_payload_stops_before_preflight(self) -> None:
        native_verify_result = _native_verify_result(changing_sorted_change_record_i32=[0])
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            with (
                mock.patch.object(promotion_gate, "_run_json_command", return_value=(0, native_verify_result)),
                mock.patch.object(promotion_gate, "_run_preflight") as preflight,
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "run_framegroup16_promotion_gate.py",
                        "--run-id",
                        "unit_native_verify_bad_payload",
                        "--summary-json",
                        str(summary_path),
                        "--experimental-native-emitted-pack-records",
                    ],
                ),
            ):
                status = promotion_gate.main()
            written = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 2)
        self.assertEqual(written["status"], "native_packed_extension_verify_failed")
        self.assertIn("changing_sorted_change_record_i32", written["native_packed_extension_verify_failures"][0])
        preflight.assert_not_called()

    def test_native_verify_zero_status_wrong_variant_root_stops_before_preflight(self) -> None:
        native_verify_result = _native_verify_result(variant_root="/tmp/wrong_world_foam_variant")
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            with (
                mock.patch.object(promotion_gate, "_run_json_command", return_value=(0, native_verify_result)),
                mock.patch.object(promotion_gate, "_run_preflight") as preflight,
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "run_framegroup16_promotion_gate.py",
                        "--run-id",
                        "unit_native_verify_wrong_variant_root",
                        "--summary-json",
                        str(summary_path),
                        "--experimental-native-emitted-pack-records",
                    ],
                ),
            ):
                status = promotion_gate.main()
            written = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 2)
        self.assertEqual(written["status"], "native_packed_extension_verify_failed")
        self.assertIn("variant_root", written["native_packed_extension_verify_failures"][0])
        preflight.assert_not_called()

    def test_native_verify_zero_status_missing_semantic_guard_stops_before_preflight(self) -> None:
        native_verify_result = _native_verify_result(
            gate4_delta_replace_packed_from_cuts_rejects_nan_depth=False,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            with (
                mock.patch.object(promotion_gate, "_run_json_command", return_value=(0, native_verify_result)),
                mock.patch.object(promotion_gate, "_run_preflight") as preflight,
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "run_framegroup16_promotion_gate.py",
                        "--run-id",
                        "unit_native_verify_missing_semantic_guard",
                        "--summary-json",
                        str(summary_path),
                        "--experimental-native-emitted-pack-records",
                    ],
                ),
            ):
                status = promotion_gate.main()
            written = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 2)
        self.assertEqual(written["status"], "native_packed_extension_verify_failed")
        self.assertIn(
            "gate4_delta_replace_packed_from_cuts_rejects_nan_depth",
            written["native_packed_extension_verify_failures"][0],
        )
        preflight.assert_not_called()

    def test_native_verify_zero_status_bad_offsets_stops_before_preflight(self) -> None:
        native_verify_result = _native_verify_result(base_offsets_i32=[0, 1])
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            with (
                mock.patch.object(promotion_gate, "_run_json_command", return_value=(0, native_verify_result)),
                mock.patch.object(promotion_gate, "_run_preflight") as preflight,
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "run_framegroup16_promotion_gate.py",
                        "--run-id",
                        "unit_native_verify_bad_offsets",
                        "--summary-json",
                        str(summary_path),
                        "--experimental-native-emitted-pack-records",
                    ],
                ),
            ):
                status = promotion_gate.main()
            written = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 2)
        self.assertEqual(written["status"], "native_packed_extension_verify_failed")
        self.assertIn("base_offsets_i32", written["native_packed_extension_verify_failures"][0])
        preflight.assert_not_called()

    def test_native_verify_zero_status_bad_cut_array_payload_stops_before_preflight(self) -> None:
        native_verify_result = _native_verify_result(cut_array_cut_offsets_i64=[0, 2, 6])
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            with (
                mock.patch.object(promotion_gate, "_run_json_command", return_value=(0, native_verify_result)),
                mock.patch.object(promotion_gate, "_run_preflight") as preflight,
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "run_framegroup16_promotion_gate.py",
                        "--run-id",
                        "unit_native_verify_bad_cut_array_payload",
                        "--summary-json",
                        str(summary_path),
                        "--experimental-native-emitted-pack-records",
                    ],
                ),
            ):
                status = promotion_gate.main()
            written = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 2)
        self.assertEqual(written["status"], "native_packed_extension_verify_failed")
        self.assertIn("cut_array_cut_offsets_i64", written["native_packed_extension_verify_failures"][0])
        preflight.assert_not_called()

    def test_native_verify_zero_status_bad_dtype_stops_before_preflight(self) -> None:
        native_verify_result = _native_verify_result(base_record_i32_dtype="int64")
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            with (
                mock.patch.object(promotion_gate, "_run_json_command", return_value=(0, native_verify_result)),
                mock.patch.object(promotion_gate, "_run_preflight") as preflight,
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "run_framegroup16_promotion_gate.py",
                        "--run-id",
                        "unit_native_verify_bad_dtype",
                        "--summary-json",
                        str(summary_path),
                        "--experimental-native-emitted-pack-records",
                    ],
                ),
            ):
                status = promotion_gate.main()
            written = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 2)
        self.assertEqual(written["status"], "native_packed_extension_verify_failed")
        self.assertIn("base_record_i32_dtype", written["native_packed_extension_verify_failures"][0])
        preflight.assert_not_called()

    def test_native_verify_zero_status_bad_shape_stops_before_preflight(self) -> None:
        native_verify_result = _native_verify_result(base_record_i32_shape=[1, 2])
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            with (
                mock.patch.object(promotion_gate, "_run_json_command", return_value=(0, native_verify_result)),
                mock.patch.object(promotion_gate, "_run_preflight") as preflight,
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "run_framegroup16_promotion_gate.py",
                        "--run-id",
                        "unit_native_verify_bad_shape",
                        "--summary-json",
                        str(summary_path),
                        "--experimental-native-emitted-pack-records",
                    ],
                ),
            ):
                status = promotion_gate.main()
            written = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 2)
        self.assertEqual(written["status"], "native_packed_extension_verify_failed")
        self.assertIn("base_record_i32_shape", written["native_packed_extension_verify_failures"][0])
        preflight.assert_not_called()

    def test_native_verify_zero_status_bad_contiguity_stops_before_preflight(self) -> None:
        native_verify_result = _native_verify_result(base_record_i32_contiguous=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            with (
                mock.patch.object(promotion_gate, "_run_json_command", return_value=(0, native_verify_result)),
                mock.patch.object(promotion_gate, "_run_preflight") as preflight,
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "run_framegroup16_promotion_gate.py",
                        "--run-id",
                        "unit_native_verify_bad_contiguity",
                        "--summary-json",
                        str(summary_path),
                        "--experimental-native-emitted-pack-records",
                    ],
                ),
            ):
                status = promotion_gate.main()
            written = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 2)
        self.assertEqual(written["status"], "native_packed_extension_verify_failed")
        self.assertIn("base_record_i32_contiguous", written["native_packed_extension_verify_failures"][0])
        preflight.assert_not_called()

    def test_native_verify_zero_status_bad_device_stops_before_preflight(self) -> None:
        native_verify_result = _native_verify_result(base_record_i32_device="mps")
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            with (
                mock.patch.object(promotion_gate, "_run_json_command", return_value=(0, native_verify_result)),
                mock.patch.object(promotion_gate, "_run_preflight") as preflight,
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "run_framegroup16_promotion_gate.py",
                        "--run-id",
                        "unit_native_verify_bad_device",
                        "--summary-json",
                        str(summary_path),
                        "--experimental-native-emitted-pack-records",
                    ],
                ),
            ):
                status = promotion_gate.main()
            written = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 2)
        self.assertEqual(written["status"], "native_packed_extension_verify_failed")
        self.assertIn("base_record_i32_device", written["native_packed_extension_verify_failures"][0])
        preflight.assert_not_called()

    def test_native_verify_zero_status_without_json_stops_before_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            with (
                mock.patch.object(promotion_gate, "_run_json_command", return_value=(0, None)),
                mock.patch.object(promotion_gate, "_run_preflight") as preflight,
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "run_framegroup16_promotion_gate.py",
                        "--run-id",
                        "unit_native_verify_no_json",
                        "--summary-json",
                        str(summary_path),
                        "--experimental-native-emitted-pack-records",
                    ],
                ),
            ):
                status = promotion_gate.main()
            written = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 2)
        self.assertEqual(written["status"], "native_packed_extension_verify_failed")
        self.assertEqual(
            written["native_packed_extension_verify_failures"],
            ["native packed extension verifier did not return a JSON object"],
        )
        preflight.assert_not_called()

    def test_custom_frames_reject_default_reference_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            with mock.patch.object(
                sys,
                "argv",
                [
                    "run_framegroup16_promotion_gate.py",
                    "--run-id",
                    "unit_custom_default_reference",
                    "--frame-counts",
                    "64,128",
                    "--summary-json",
                    str(summary_path),
                    "--dry-run",
                ],
            ):
                status = promotion_gate.main()
            written = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 2)
        self.assertEqual(written["status"], "config_failed")
        self.assertIn("not covered by the default reference", written["config_failures"][0])
        self.assertNotIn("preflight_status", written)

    def test_reference_artifact_may_cover_more_frames_than_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            reference_path = tmpdir_path / "reference.json"
            reference_path.write_text(
                json.dumps({"rows": {"64": {}, "128": {}, "256": {}}}) + "\n",
                encoding="utf-8",
            )
            summary_path = tmpdir_path / "summary.json"
            with mock.patch.object(
                sys,
                "argv",
                [
                    "run_framegroup16_promotion_gate.py",
                    "--run-id",
                    "unit_custom_explicit_superset_reference",
                    "--frame-counts",
                    "64,128",
                    "--reference-artifact",
                    str(reference_path),
                    "--summary-json",
                    str(summary_path),
                    "--dry-run",
                ],
            ):
                status = promotion_gate.main()
            written = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertEqual(written["status"], "ok")
        self.assertEqual(written["reference_artifact_frames"], [64, 128, 256])
        self.assertIn("--reference-artifact", written["verify_command"])

    def test_reference_artifact_missing_requested_frame_fails_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            reference_path = tmpdir_path / "reference.json"
            reference_path.write_text(json.dumps({"rows": [{"frame_count": 64}]}) + "\n", encoding="utf-8")
            summary_path = tmpdir_path / "summary.json"
            with mock.patch.object(
                sys,
                "argv",
                [
                    "run_framegroup16_promotion_gate.py",
                    "--run-id",
                    "unit_custom_explicit_missing_reference",
                    "--frame-counts",
                    "64,128",
                    "--reference-artifact",
                    str(reference_path),
                    "--summary-json",
                    str(summary_path),
                    "--dry-run",
                ],
            ):
                status = promotion_gate.main()
            written = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 2)
        self.assertEqual(written["status"], "config_failed")
        self.assertIn("missing (128,)", written["config_failures"][0])
        self.assertNotIn("preflight_status", written)

    def test_custom_frames_can_intentionally_skip_reference_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            with mock.patch.object(
                sys,
                "argv",
                [
                    "run_framegroup16_promotion_gate.py",
                    "--run-id",
                    "unit_custom_no_reference",
                    "--frame-counts",
                    "64,128",
                    "--summary-json",
                    str(summary_path),
                    "--no-reference-artifact",
                    "--dry-run",
                ],
            ):
                status = promotion_gate.main()
            written = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertEqual(written["status"], "ok")
        self.assertIsNone(written["reference_artifact"])
        self.assertNotIn("--reference-artifact", written["verify_command"])

    def test_contaminated_verify_failure_retries_with_attempt_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            out_json = tmpdir_path / "promotion.json"
            partial_json = tmpdir_path / "promotion.partial.json"
            verify_json = tmpdir_path / "promotion.reference_verify.json"
            summary_path = tmpdir_path / "summary.json"
            with (
                mock.patch.object(
                    promotion_gate,
                    "_run_preflight",
                    return_value=(0, [{"returncode": 0, "status": "background", "success_streak": 2}]),
                ) as preflight,
                mock.patch.object(promotion_gate, "_run", side_effect=[0, 1, 0, 0]) as run_command,
                mock.patch.object(
                    promotion_gate,
                    "_load_verify_brief",
                    side_effect=[
                        {
                            "status": "failed",
                            "contamination": ["benchmark_environment status is 'contended'"],
                            "failures": ["16f total median was contaminated"],
                        },
                        {"status": "ok", "contamination": [], "failures": []},
                    ],
                ),
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "run_framegroup16_promotion_gate.py",
                        "--run-id",
                        "unit_retry_contaminated_verify",
                        "--out-json",
                        str(out_json),
                        "--partial-out-json",
                        str(partial_json),
                        "--verify-json",
                        str(verify_json),
                        "--summary-json",
                        str(summary_path),
                        "--max-promotion-attempts",
                        "2",
                        "--no-reference-artifact",
                    ],
                ),
            ):
                status = promotion_gate.main()
            written = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertEqual(written["status"], "ok")
        self.assertEqual(written["max_promotion_attempts"], 2)
        self.assertEqual(len(written["attempts"]), 2)
        self.assertEqual(written["attempts"][0]["retry_reason"], "verify_contamination")
        self.assertIn("promotion.attempt1.json", written["attempts"][0]["out_json"])
        self.assertIn("promotion.attempt1.partial.json", written["attempts"][0]["partial_out_json"])
        self.assertIn("promotion.attempt1.reference_verify.json", written["attempts"][0]["verify_json"])
        self.assertIn("promotion.attempt2.json", written["attempts"][1]["out_json"])
        self.assertEqual(preflight.call_count, 2)
        self.assertEqual(run_command.call_count, 4)

    def test_contaminated_structural_verify_failure_does_not_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            summary_path = tmpdir_path / "summary.json"
            with (
                mock.patch.object(
                    promotion_gate,
                    "_run_preflight",
                    return_value=(0, [{"returncode": 0, "status": "background", "success_streak": 2}]),
                ) as preflight,
                mock.patch.object(promotion_gate, "_run", side_effect=[0, 1]) as run_command,
                mock.patch.object(
                    promotion_gate,
                    "_load_verify_brief",
                    return_value={
                        "status": "failed",
                        "contamination": ["benchmark_environment status is 'contended'"],
                        "failures": [
                            "topology storage scale 2.501 exceeds 1.100",
                            "16f total median was contaminated",
                        ],
                    },
                ),
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "run_framegroup16_promotion_gate.py",
                        "--run-id",
                        "unit_no_retry_contaminated_structural_failure",
                        "--out-json",
                        str(tmpdir_path / "promotion.json"),
                        "--partial-out-json",
                        str(tmpdir_path / "promotion.partial.json"),
                        "--verify-json",
                        str(tmpdir_path / "promotion.reference_verify.json"),
                        "--summary-json",
                        str(summary_path),
                        "--max-promotion-attempts",
                        "2",
                        "--no-reference-artifact",
                    ],
                ),
            ):
                status = promotion_gate.main()
            written = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 1)
        self.assertEqual(written["status"], "verify_failed")
        self.assertEqual(len(written["attempts"]), 1)
        self.assertNotIn("retry_reason", written["attempts"][0])
        self.assertEqual(preflight.call_count, 1)
        self.assertEqual(run_command.call_count, 2)

    def test_clean_verify_failure_does_not_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            summary_path = tmpdir_path / "summary.json"
            with (
                mock.patch.object(
                    promotion_gate,
                    "_run_preflight",
                    return_value=(0, [{"returncode": 0, "status": "background", "success_streak": 2}]),
                ) as preflight,
                mock.patch.object(promotion_gate, "_run", side_effect=[0, 1]) as run_command,
                mock.patch.object(
                    promotion_gate,
                    "_load_verify_brief",
                    return_value={
                        "status": "failed",
                        "contamination": [],
                        "failures": ["16f total median is a clean regression"],
                    },
                ),
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "run_framegroup16_promotion_gate.py",
                        "--run-id",
                        "unit_no_retry_clean_verify_failure",
                        "--out-json",
                        str(tmpdir_path / "promotion.json"),
                        "--partial-out-json",
                        str(tmpdir_path / "promotion.partial.json"),
                        "--verify-json",
                        str(tmpdir_path / "promotion.reference_verify.json"),
                        "--summary-json",
                        str(summary_path),
                        "--max-promotion-attempts",
                        "2",
                        "--no-reference-artifact",
                    ],
                ),
            ):
                status = promotion_gate.main()
            written = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 1)
        self.assertEqual(written["status"], "verify_failed")
        self.assertEqual(len(written["attempts"]), 1)
        self.assertEqual(preflight.call_count, 1)
        self.assertEqual(run_command.call_count, 2)

    def test_minimal_and_kernel_order_packed_device_flags_are_mutually_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            with (
                mock.patch.object(promotion_gate, "_run_preflight") as preflight,
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "run_framegroup16_promotion_gate.py",
                        "--run-id",
                        "unit_mutually_exclusive_packed_device_flags",
                        "--summary-json",
                        str(summary_path),
                        "--experimental-minimal-packed-delta-device",
                        "--experimental-kernel-order-packed-delta-device",
                    ],
                ),
            ):
                status = promotion_gate.main()
            written = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 2)
        self.assertEqual(written["status"], "config_failed")
        self.assertIn("mutually exclusive", written["config_failures"][0])
        preflight.assert_not_called()

    def test_unchecked_launch_only_requires_launch_only_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            with (
                mock.patch.object(promotion_gate, "_run_preflight") as preflight,
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "run_framegroup16_promotion_gate.py",
                        "--run-id",
                        "unit_unchecked_launch_only_requires_checked_flag",
                        "--summary-json",
                        str(summary_path),
                        "--experimental-unchecked-launch-only-packed-delta",
                    ],
                ),
            ):
                status = promotion_gate.main()
            written = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 2)
        self.assertEqual(written["status"], "config_failed")
        self.assertIn("requires --experimental-launch-only-packed-delta", written["config_failures"][0])
        preflight.assert_not_called()

    def test_rowdesc_launch_only_requires_launch_only_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            with (
                mock.patch.object(promotion_gate, "_run_preflight") as preflight,
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "run_framegroup16_promotion_gate.py",
                        "--run-id",
                        "unit_rowdesc_launch_only_requires_checked_flag",
                        "--summary-json",
                        str(summary_path),
                        "--experimental-rowdesc-launch-only-packed-delta",
                    ],
                ),
            ):
                status = promotion_gate.main()
            written = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 2)
        self.assertEqual(written["status"], "config_failed")
        self.assertIn("requires --experimental-launch-only-packed-delta", written["config_failures"][0])
        preflight.assert_not_called()

    def test_reduce32_launch_only_requires_launch_only_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            with (
                mock.patch.object(promotion_gate, "_run_preflight") as preflight,
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "run_framegroup16_promotion_gate.py",
                        "--run-id",
                        "unit_reduce32_launch_only_requires_checked_flag",
                        "--summary-json",
                        str(summary_path),
                        "--experimental-reduce32-launch-only-packed-delta",
                    ],
                ),
            ):
                status = promotion_gate.main()
            written = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 2)
        self.assertEqual(written["status"], "config_failed")
        self.assertIn("requires --experimental-launch-only-packed-delta", written["config_failures"][0])
        preflight.assert_not_called()

    def test_reduce32_launch_only_rejects_rowdesc_combo(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            with (
                mock.patch.object(promotion_gate, "_run_preflight") as preflight,
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "run_framegroup16_promotion_gate.py",
                        "--run-id",
                        "unit_reduce32_launch_only_rejects_rowdesc_combo",
                        "--summary-json",
                        str(summary_path),
                        "--experimental-launch-only-packed-delta",
                        "--experimental-reduce32-launch-only-packed-delta",
                        "--experimental-rowdesc-launch-only-packed-delta",
                    ],
                ),
            ):
                status = promotion_gate.main()
            written = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 2)
        self.assertEqual(written["status"], "config_failed")
        self.assertTrue(
            any("cannot be combined" in failure for failure in written["config_failures"])
        )
        preflight.assert_not_called()

    def test_rowselect32_launch_only_requires_launch_only_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            with (
                mock.patch.object(promotion_gate, "_run_preflight") as preflight,
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "run_framegroup16_promotion_gate.py",
                        "--run-id",
                        "unit_rowselect32_launch_only_requires_checked_flag",
                        "--summary-json",
                        str(summary_path),
                        "--experimental-rowselect32-launch-only-packed-delta",
                    ],
                ),
            ):
                status = promotion_gate.main()
            written = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 2)
        self.assertEqual(written["status"], "config_failed")
        self.assertIn("requires --experimental-launch-only-packed-delta", written["config_failures"][0])
        preflight.assert_not_called()

    def test_rowselect32_launch_only_rejects_reduce32_combo(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            with (
                mock.patch.object(promotion_gate, "_run_preflight") as preflight,
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "run_framegroup16_promotion_gate.py",
                        "--run-id",
                        "unit_rowselect32_launch_only_rejects_reduce32_combo",
                        "--summary-json",
                        str(summary_path),
                        "--experimental-launch-only-packed-delta",
                        "--experimental-rowselect32-launch-only-packed-delta",
                        "--experimental-reduce32-launch-only-packed-delta",
                    ],
                ),
            ):
                status = promotion_gate.main()
            written = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 2)
        self.assertEqual(written["status"], "config_failed")
        self.assertTrue(any("cannot be combined" in failure for failure in written["config_failures"]))
        preflight.assert_not_called()

    def test_rowselect32_launch_only_rejects_rowdesc_combo(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            with (
                mock.patch.object(promotion_gate, "_run_preflight") as preflight,
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "run_framegroup16_promotion_gate.py",
                        "--run-id",
                        "unit_rowselect32_launch_only_rejects_rowdesc_combo",
                        "--summary-json",
                        str(summary_path),
                        "--experimental-launch-only-packed-delta",
                        "--experimental-rowselect32-launch-only-packed-delta",
                        "--experimental-rowdesc-launch-only-packed-delta",
                    ],
                ),
            ):
                status = promotion_gate.main()
            written = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 2)
        self.assertEqual(written["status"], "config_failed")
        self.assertTrue(any("cannot be combined" in failure for failure in written["config_failures"]))
        preflight.assert_not_called()

    def test_rowdesc32_launch_only_requires_rowdesc_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            with (
                mock.patch.object(promotion_gate, "_run_preflight") as preflight,
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "run_framegroup16_promotion_gate.py",
                        "--run-id",
                        "unit_rowdesc32_launch_only_requires_rowdesc_flag",
                        "--summary-json",
                        str(summary_path),
                        "--experimental-launch-only-packed-delta",
                        "--experimental-rowdesc32-launch-only-packed-delta",
                    ],
                ),
            ):
                status = promotion_gate.main()
            written = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 2)
        self.assertEqual(written["status"], "config_failed")
        self.assertIn("requires --experimental-rowdesc-launch-only-packed-delta", written["config_failures"][0])
        preflight.assert_not_called()

    def test_reduce32_launch_only_allows_unchecked_launch_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            with mock.patch.object(
                sys,
                "argv",
                [
                    "run_framegroup16_promotion_gate.py",
                    "--run-id",
                    "unit_reduce32_launch_only_allows_unchecked",
                    "--summary-json",
                    str(summary_path),
                    "--experimental-launch-only-packed-delta",
                    "--experimental-unchecked-launch-only-packed-delta",
                    "--experimental-reduce32-launch-only-packed-delta",
                    "--no-reference-artifact",
                    "--dry-run",
                ],
            ):
                status = promotion_gate.main()
            written = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertEqual(written["status"], "ok")
        self.assertIn("--experimental-launch-only-packed-delta", written["train_command"])
        self.assertIn("--experimental-unchecked-launch-only-packed-delta", written["train_command"])
        self.assertIn("--experimental-reduce32-launch-only-packed-delta", written["train_command"])
        self.assertIn("experimental_reduce32_launch_only_packed_delta=true", written["verify_command"])

    def test_rowselect32_launch_only_allows_unchecked_launch_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            with mock.patch.object(
                sys,
                "argv",
                [
                    "run_framegroup16_promotion_gate.py",
                    "--run-id",
                    "unit_rowselect32_launch_only_allows_unchecked",
                    "--summary-json",
                    str(summary_path),
                    "--experimental-launch-only-packed-delta",
                    "--experimental-unchecked-launch-only-packed-delta",
                    "--experimental-rowselect32-launch-only-packed-delta",
                    "--no-reference-artifact",
                    "--dry-run",
                ],
            ):
                status = promotion_gate.main()
            written = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertEqual(written["status"], "ok")
        self.assertIn("--experimental-launch-only-packed-delta", written["train_command"])
        self.assertIn("--experimental-unchecked-launch-only-packed-delta", written["train_command"])
        self.assertIn("--experimental-rowselect32-launch-only-packed-delta", written["train_command"])
        self.assertIn("experimental_rowselect32_launch_only_packed_delta=true", written["verify_command"])

    def test_rowdesc_launch_only_allows_unchecked_launch_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            with mock.patch.object(
                sys,
                "argv",
                [
                    "run_framegroup16_promotion_gate.py",
                    "--run-id",
                    "unit_rowdesc_launch_only_allows_unchecked",
                    "--summary-json",
                    str(summary_path),
                    "--experimental-launch-only-packed-delta",
                    "--experimental-unchecked-launch-only-packed-delta",
                    "--experimental-rowdesc-launch-only-packed-delta",
                    "--experimental-rowdesc32-launch-only-packed-delta",
                    "--no-reference-artifact",
                    "--dry-run",
                ],
            ):
                status = promotion_gate.main()
            written = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertEqual(written["status"], "ok")
        self.assertIn("--experimental-launch-only-packed-delta", written["train_command"])
        self.assertIn("--experimental-unchecked-launch-only-packed-delta", written["train_command"])
        self.assertIn("--experimental-rowdesc-launch-only-packed-delta", written["train_command"])
        self.assertIn("--experimental-rowdesc32-launch-only-packed-delta", written["train_command"])
        self.assertIn("--expect-payload-bool", written["verify_command"])
        self.assertIn("experimental_launch_only_packed_delta=true", written["verify_command"])
        self.assertIn("experimental_unchecked_launch_only_packed_delta=true", written["verify_command"])
        self.assertIn("experimental_rowdesc_launch_only_packed_delta=true", written["verify_command"])
        self.assertIn("experimental_rowdesc32_launch_only_packed_delta=true", written["verify_command"])


if __name__ == "__main__":
    unittest.main()
