from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import run_worldfoam_star_native_cutwalk_gate as gate
import verify_worldfoam_star_native_cutwalk_promotion as verifier


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _add_real_input_command_metadata(summary: dict[str, object], tmp_path: Path) -> None:
    worldfoam_config = str(tmp_path / "worldfoam_real32.jsonc")
    star_video_path = str(tmp_path / "star_real32.mp4")
    summary["worldfoam_config"] = worldfoam_config
    summary["star_video_path"] = star_video_path
    summary["worldfoam_preflight_command"] = [
        "python",
        "train_eval_owner_run_tape.py",
        "--benchmark-environment-check-only",
        "--config",
        worldfoam_config,
    ]
    summary["worldfoam_command"] = [
        "python",
        "train_eval_owner_run_tape.py",
        "--config",
        worldfoam_config,
    ]
    summary["planned_star_compare_command"] = [
        "python",
        "compare_star_uvt_worldfoam_scale.py",
        "--worldfoam-artifact",
        str(summary["worldfoam_artifact"]),
        "--video-path",
        star_video_path,
    ]
    summary["star_compare_command"] = [
        "python",
        "compare_star_uvt_worldfoam_scale.py",
        "--worldfoam-artifact",
        str(summary["worldfoam_artifact"]),
        "--video-path",
        star_video_path,
    ]


def _add_real_input_artifact_metadata(tmp_path: Path, summary: dict[str, object]) -> None:
    worldfoam_path = Path(str(summary["worldfoam_artifact"]))
    star_path = Path(str(summary["star_compare_artifact"]))
    worldfoam = json.loads(worldfoam_path.read_text(encoding="utf-8"))
    star = json.loads(star_path.read_text(encoding="utf-8"))
    worldfoam["config_path"] = summary["worldfoam_config"]
    star["star"]["video_path"] = summary["star_video_path"]
    _write_json(worldfoam_path, worldfoam)
    _write_json(star_path, star)


class VerifyWorldFoamStarNativeCutwalkPromotionTests(unittest.TestCase):
    def _write_valid_artifacts(self, tmp_path: Path) -> Path:
        worldfoam_path = tmp_path / "clean.worldfoam.json"
        star_path = tmp_path / "clean.star_compare.json"
        summary_path = tmp_path / "clean.promotion_summary.json"
        _write_json(
            worldfoam_path,
            {
                "status": "ok",
                "benchmark_environment": {"status": "background"},
                "tape_mode": gate.DEFAULT_TAPE_MODE,
                "endpoint_record_source": "slow-owner-run",
                "experimental_selected_only_owner_run_delta_prep": True,
                "experimental_native_owner_run_cutwalk_delta": True,
                "frame_counts": [2, 4, 8, 16],
                "repeat_loaded_frames": False,
                "rows": [
                    {
                        "frame_count": frame_count,
                        "loaded_frame_count": frame_count,
                        "repeat_loaded_frames": False,
                    }
                    for frame_count in (2, 4, 8, 16)
                ],
                "acceptance": {
                    "all_rows_ok": True,
                    "backward_sublinear_vs_frames": True,
                    "total_step_sublinear_vs_frames": True,
                },
            },
        )
        _write_json(
            star_path,
            {
                "status": "ok",
                "failures": [],
                "benchmark_environment": {"status": "background"},
                "star": {
                    "repeat_loaded_frames": False,
                    "rows": [
                        {
                            "frames": frame_count,
                            "requested_frames": frame_count,
                            "loaded_frame_count": frame_count,
                            "repeat_loaded_frames": False,
                            "repeat_loaded_frames_used": False,
                        }
                        for frame_count in (2, 4, 8, 16)
                    ],
                    "summary": {"status": "ok"},
                },
                "worldfoam": {
                    "artifact": str(worldfoam_path),
                    "summary": {
                        "status": "ok",
                        "benchmark_environment_status": "background",
                    },
                },
            },
        )
        _write_json(
            summary_path,
            {
                "summary_schema_version": gate.SUMMARY_SCHEMA_VERSION,
                "status": "ok",
                "worldfoam_artifact": str(worldfoam_path),
                "worldfoam_promotable_artifact": str(worldfoam_path),
                "worldfoam_latest_written_artifact": str(worldfoam_path),
                "worldfoam_returncode": 0,
                "worldfoam_status": "ok",
                "worldfoam_benchmark_environment_status": "background",
                "repeat_loaded_frames": False,
                "require_real_loaded_frames": False,
                "frame_counts": [2, 4, 8, 16],
                "worldfoam_attempts": [
                    {
                        "artifact": str(worldfoam_path),
                        "artifact_written": True,
                        "promotable": True,
                        "preflight_returncode": 0,
                        "preflight_benchmark_environment_status": "background",
                        "returncode": 0,
                        "status": "ok",
                        "benchmark_environment_status": "background",
                    }
                ],
                "star_compare_artifact": str(star_path),
                "star_compare_latest_attempt_artifact": str(star_path),
                "star_compare_latest_written_artifact": str(star_path),
                "planned_star_compare_command": [
                    "python",
                    "compare_star_uvt_worldfoam_scale.py",
                    "--worldfoam-artifact",
                    str(worldfoam_path),
                ],
                "star_compare_command": [
                    "python",
                    "compare_star_uvt_worldfoam_scale.py",
                    "--worldfoam-artifact",
                    str(worldfoam_path),
                ],
                "star_compare_returncode": 0,
                "star_compare_status": "ok",
                "star_compare_benchmark_environment_status": "background",
                "star_compare_attempts": [
                    {
                        "artifact": str(star_path),
                        "artifact_written": True,
                        "promotable": True,
                        "returncode": 0,
                        "status": "ok",
                        "benchmark_environment_status": "background",
                    }
                ],
            },
        )
        return summary_path

    def test_clean_promotion_summary_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = self._write_valid_artifacts(Path(tmpdir))
            report = verifier.verify_summary(summary_path)

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["failures"], [])

    def test_preflight_only_summary_fails_without_selected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            summary_path = tmp_path / "blocked.promotion_summary.json"
            _write_json(
                summary_path,
                {
                    "summary_schema_version": gate.SUMMARY_SCHEMA_VERSION,
                    "status": "worldfoam_preflight_failed_or_contended",
                    "worldfoam_artifact": None,
                    "worldfoam_promotable_artifact": None,
                    "worldfoam_attempts": [],
                    "star_compare_artifact": str(tmp_path / "missing.star.json"),
                    "star_compare_command": None,
                },
            )
            report = verifier.verify_summary(summary_path)

        self.assertEqual(report["status"], "failed")
        self.assertIn("promotion summary status is not ok: worldfoam_preflight_failed_or_contended", report["failures"])
        self.assertIn("worldfoam_artifact is not selected", report["failures"])
        self.assertIn("star_compare_command is not selected", report["failures"])
        self.assertIn("promotion summary has no STAR attempts", report["failures"])

    def test_summary_requires_selected_star_attempt_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            summary_path = self._write_valid_artifacts(tmp_path)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            wrong_star = str(tmp_path / "wrong.star_compare.json")
            summary["star_compare_latest_attempt_artifact"] = wrong_star
            summary["star_compare_latest_written_artifact"] = wrong_star
            summary["star_compare_attempts"] = [
                {
                    "artifact": wrong_star,
                    "artifact_written": True,
                    "promotable": True,
                    "returncode": 0,
                    "status": "ok",
                    "benchmark_environment_status": "background",
                }
            ]
            _write_json(summary_path, summary)
            report = verifier.verify_summary(summary_path)

        self.assertEqual(report["status"], "failed")
        self.assertIn("star_compare_latest_attempt_artifact does not match selected artifact", report["failures"])
        self.assertIn("star_compare_latest_written_artifact does not match selected artifact", report["failures"])
        self.assertIn("promotable STAR attempt does not match selected STAR artifact", report["failures"])

    def test_summary_requires_star_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            summary_path = self._write_valid_artifacts(tmp_path)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary.pop("star_compare_attempts")
            _write_json(summary_path, summary)
            report = verifier.verify_summary(summary_path)

        self.assertEqual(report["status"], "failed")
        self.assertIn("promotion summary has no STAR attempts", report["failures"])

    def test_contended_worldfoam_artifact_fails_even_if_summary_claims_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            summary_path = self._write_valid_artifacts(tmp_path)
            worldfoam_path = tmp_path / "clean.worldfoam.json"
            payload = json.loads(worldfoam_path.read_text(encoding="utf-8"))
            payload["benchmark_environment"] = {"status": "contended"}
            _write_json(worldfoam_path, payload)
            report = verifier.verify_summary(summary_path)

        self.assertEqual(report["status"], "failed")
        self.assertIn("WorldFoam artifact benchmark_environment is not clean: contended", report["failures"])

    def test_missing_worldfoam_acceptance_fails_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            summary_path = self._write_valid_artifacts(tmp_path)
            worldfoam_path = tmp_path / "clean.worldfoam.json"
            payload = json.loads(worldfoam_path.read_text(encoding="utf-8"))
            payload.pop("acceptance")
            _write_json(worldfoam_path, payload)
            report = verifier.verify_summary(summary_path)

        self.assertEqual(report["status"], "failed")
        self.assertIn("WorldFoam artifact acceptance is missing", report["failures"])

    def test_ok_environment_status_is_clean_for_promotion_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            summary_path = self._write_valid_artifacts(tmp_path)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            worldfoam_path = Path(str(summary["worldfoam_artifact"]))
            star_path = Path(str(summary["star_compare_artifact"]))
            worldfoam = json.loads(worldfoam_path.read_text(encoding="utf-8"))
            star = json.loads(star_path.read_text(encoding="utf-8"))
            worldfoam["benchmark_environment"] = {"status": "ok"}
            star["benchmark_environment"] = {"status": "ok"}
            star["worldfoam"]["summary"]["benchmark_environment_status"] = "ok"
            summary["worldfoam_benchmark_environment_status"] = "ok"
            summary["worldfoam_attempts"][0]["preflight_benchmark_environment_status"] = "ok"
            summary["worldfoam_attempts"][0]["benchmark_environment_status"] = "ok"
            summary["star_compare_benchmark_environment_status"] = "ok"
            summary["star_compare_attempts"][0]["benchmark_environment_status"] = "ok"
            _write_json(worldfoam_path, worldfoam)
            _write_json(star_path, star)
            _write_json(summary_path, summary)
            report = verifier.verify_summary(summary_path)

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["failures"], [])

    def test_unchecked_environment_status_is_not_clean_for_promotion_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            summary_path = self._write_valid_artifacts(tmp_path)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["worldfoam_benchmark_environment_status"] = "unchecked"
            summary["worldfoam_attempts"][0]["benchmark_environment_status"] = "unchecked"
            _write_json(summary_path, summary)
            report = verifier.verify_summary(summary_path)

        self.assertEqual(report["status"], "failed")
        self.assertIn("WorldFoam summary benchmark_environment_status is not clean", report["failures"])
        self.assertIn("promotable WorldFoam attempt benchmark environment is not clean", report["failures"])

    def test_real_loaded_frame_requirement_passes_with_explicit_real_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            summary_path = self._write_valid_artifacts(tmp_path)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["require_real_loaded_frames"] = True
            _add_real_input_command_metadata(summary, tmp_path)
            _add_real_input_artifact_metadata(tmp_path, summary)
            _write_json(summary_path, summary)
            report = verifier.verify_summary(summary_path)

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["failures"], [])

    def test_real_loaded_frame_requirement_rejects_missing_custom_input_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            summary_path = self._write_valid_artifacts(tmp_path)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["require_real_loaded_frames"] = True
            _write_json(summary_path, summary)
            report = verifier.verify_summary(summary_path)

        self.assertEqual(report["status"], "failed")
        self.assertIn("real-loaded-frame promotion must record worldfoam_config", report["failures"])
        self.assertIn("real-loaded-frame promotion must record star_video_path", report["failures"])

    def test_real_loaded_frame_requirement_rejects_mismatched_custom_input_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            summary_path = self._write_valid_artifacts(tmp_path)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["require_real_loaded_frames"] = True
            _add_real_input_command_metadata(summary, tmp_path)
            summary["worldfoam_preflight_command"] = ["python", "train_eval_owner_run_tape.py"]
            summary["worldfoam_command"] = ["python", "train_eval_owner_run_tape.py", "--config", "wrong.jsonc"]
            summary["planned_star_compare_command"] = [
                "python",
                "compare_star_uvt_worldfoam_scale.py",
                "--video-path",
                "wrong.mp4",
            ]
            summary["star_compare_command"] = ["python", "compare_star_uvt_worldfoam_scale.py"]
            _write_json(summary_path, summary)
            report = verifier.verify_summary(summary_path)

        self.assertEqual(report["status"], "failed")
        self.assertIn("worldfoam_preflight_command must pass --config matching worldfoam_config", report["failures"])
        self.assertIn("worldfoam_command must pass --config matching worldfoam_config", report["failures"])
        self.assertIn("planned_star_compare_command must pass --video-path matching star_video_path", report["failures"])
        self.assertIn("star_compare_command must pass --video-path matching star_video_path", report["failures"])

    def test_promotion_rejects_star_commands_pointing_at_wrong_worldfoam_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            summary_path = self._write_valid_artifacts(tmp_path)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            wrong_worldfoam = str(tmp_path / "wrong.worldfoam.json")
            summary["planned_star_compare_command"] = [
                "python",
                "compare_star_uvt_worldfoam_scale.py",
                "--worldfoam-artifact",
                wrong_worldfoam,
            ]
            summary["star_compare_command"] = [
                "python",
                "compare_star_uvt_worldfoam_scale.py",
                "--worldfoam-artifact",
                wrong_worldfoam,
            ]
            _write_json(summary_path, summary)
            report = verifier.verify_summary(summary_path)

        self.assertEqual(report["status"], "failed")
        self.assertIn(
            "planned_star_compare_command must consume the selected WorldFoam artifact",
            report["failures"],
        )
        self.assertIn(
            "star_compare_command must consume the selected WorldFoam artifact",
            report["failures"],
        )

    def test_real_loaded_frame_requirement_rejects_mismatched_artifact_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            summary_path = self._write_valid_artifacts(tmp_path)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["require_real_loaded_frames"] = True
            _add_real_input_command_metadata(summary, tmp_path)
            _add_real_input_artifact_metadata(tmp_path, summary)
            worldfoam_path = Path(str(summary["worldfoam_artifact"]))
            star_path = Path(str(summary["star_compare_artifact"]))
            worldfoam = json.loads(worldfoam_path.read_text(encoding="utf-8"))
            star = json.loads(star_path.read_text(encoding="utf-8"))
            worldfoam["config_path"] = "wrong.jsonc"
            star["star"]["video_path"] = "wrong.mp4"
            _write_json(worldfoam_path, worldfoam)
            _write_json(star_path, star)
            _write_json(summary_path, summary)
            report = verifier.verify_summary(summary_path)

        self.assertEqual(report["status"], "failed")
        self.assertIn("WorldFoam artifact config_path must match worldfoam_config", report["failures"])
        self.assertIn("STAR artifact video_path must match star_video_path", report["failures"])

    def test_real_loaded_frame_requirement_rejects_mismatched_frame_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            summary_path = self._write_valid_artifacts(tmp_path)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["require_real_loaded_frames"] = True
            summary["frame_counts"] = [2, 4, 8, 32]
            _add_real_input_command_metadata(summary, tmp_path)
            _add_real_input_artifact_metadata(tmp_path, summary)
            _write_json(summary_path, summary)
            report = verifier.verify_summary(summary_path)

        self.assertEqual(report["status"], "failed")
        self.assertIn(
            "WorldFoam artifact frame_counts [2, 4, 8, 16] do not match requested [2, 4, 8, 32]",
            report["failures"],
        )
        self.assertIn(
            "STAR artifact frame_counts [2, 4, 8, 16] do not match requested [2, 4, 8, 32]",
            report["failures"],
        )

    def test_real_loaded_frame_requirement_rejects_repeated_frame_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            summary_path = self._write_valid_artifacts(tmp_path)
            worldfoam_path = tmp_path / "clean.worldfoam.json"
            star_path = tmp_path / "clean.star_compare.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            worldfoam = json.loads(worldfoam_path.read_text(encoding="utf-8"))
            star = json.loads(star_path.read_text(encoding="utf-8"))
            summary["require_real_loaded_frames"] = True
            _add_real_input_command_metadata(summary, tmp_path)
            _add_real_input_artifact_metadata(tmp_path, summary)
            worldfoam["repeat_loaded_frames"] = True
            worldfoam["rows"][0]["frame_count"] = 32
            worldfoam["rows"][0]["loaded_frame_count"] = 16
            worldfoam["rows"][0]["repeat_loaded_frames"] = True
            star["star"]["repeat_loaded_frames"] = True
            star["star"]["rows"][0]["requested_frames"] = 32
            star["star"]["rows"][0]["loaded_frame_count"] = 16
            star["star"]["rows"][0]["repeat_loaded_frames_used"] = True
            _write_json(summary_path, summary)
            _write_json(worldfoam_path, worldfoam)
            _write_json(star_path, star)
            report = verifier.verify_summary(summary_path)

        self.assertEqual(report["status"], "failed")
        self.assertIn("WorldFoam artifact is marked as a repeated-loaded-frame smoke", report["failures"])
        self.assertIn(
            "WorldFoam row used too few loaded frames: frame_count=32 loaded_frame_count=16",
            report["failures"],
        )
        self.assertIn("WorldFoam row 32f used repeated loaded frames", report["failures"])
        self.assertIn("STAR artifact is marked as a repeated-loaded-frame smoke", report["failures"])
        self.assertIn("STAR row used too few loaded frames: requested=32 loaded_frame_count=16", report["failures"])
        self.assertIn("STAR row 32f used repeated loaded frames", report["failures"])


if __name__ == "__main__":
    unittest.main()
