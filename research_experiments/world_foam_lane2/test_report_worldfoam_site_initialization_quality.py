from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import report_worldfoam_site_initialization_quality as report


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _gate1_payload(
    *,
    initialization: str,
    train_psnr: float,
    heldout_psnr: float,
    train_l1: float = 0.2,
    heldout_l1: float = 0.2,
    render_size: int = 16,
) -> dict[str, object]:
    return {
        "benchmark": "world_foam_lane2_gate1_realray_per_sample_reference",
        "status": "ok",
        "config_path": "same_config.jsonc",
        "frame_count": 2,
        "render_size": render_size,
        "site_count": 9,
        "boundary_count": 36,
        "site_initialization": initialization,
        "train": {"target_psnr": train_psnr, "target_l1": train_l1, "target_mse": 0.05},
        "heldout": {"target_psnr": heldout_psnr, "target_l1": heldout_l1, "target_mse": 0.04},
    }


class WorldFoamSiteInitializationQualityTests(unittest.TestCase):
    def test_positive_candidate_requires_both_psnr_and_l1_improvement(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            legacy = _write_json(
                tmp_path / "legacy.json",
                _gate1_payload(initialization="legacy_sparse", train_psnr=11.0, heldout_psnr=12.0),
            )
            pixel_mean = _write_json(
                tmp_path / "pixel_mean.json",
                _gate1_payload(
                    initialization="legacy_pixel_mean",
                    train_psnr=13.0,
                    heldout_psnr=14.0,
                    train_l1=0.18,
                    heldout_l1=0.17,
                ),
            )
            train_only = _write_json(
                tmp_path / "train_only.json",
                _gate1_payload(
                    initialization="train_only",
                    train_psnr=12.0,
                    heldout_psnr=11.0,
                    train_l1=0.18,
                    heldout_l1=0.21,
                ),
            )

            payload = report.build_report(
                (legacy, pixel_mean, train_only),
                baseline_initialization="legacy_sparse",
            )

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["positive_candidate_count"], 1)
        self.assertEqual(payload["rejected_candidate_count"], 1)
        self.assertEqual(payload["best_by_heldout_psnr"], "legacy_pixel_mean")
        self.assertEqual(payload["next_mps_candidate"], "legacy_pixel_mean")
        by_mode = {row["site_initialization"]: row for row in payload["rows"]}
        self.assertTrue(by_mode["legacy_pixel_mean"]["positive_cpu_reference_candidate"])
        self.assertFalse(by_mode["train_only"]["positive_cpu_reference_candidate"])
        self.assertAlmostEqual(by_mode["legacy_pixel_mean"]["heldout_psnr_delta_vs_baseline"], 2.0)

    def test_fixture_mismatch_fails_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            legacy = _write_json(
                tmp_path / "legacy.json",
                _gate1_payload(initialization="legacy_sparse", train_psnr=11.0, heldout_psnr=12.0),
            )
            mismatch = _write_json(
                tmp_path / "mismatch.json",
                _gate1_payload(
                    initialization="legacy_pixel_mean",
                    train_psnr=13.0,
                    heldout_psnr=14.0,
                    render_size=32,
                ),
            )

            payload = report.build_report((legacy, mismatch), baseline_initialization="legacy_sparse")

        self.assertEqual(payload["status"], "failed")
        self.assertIn("fixture does not match baseline", payload["failures"][0])

    def test_missing_baseline_fails_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            candidate = _write_json(
                Path(tmpdir) / "candidate.json",
                _gate1_payload(initialization="legacy_pixel_mean", train_psnr=13.0, heldout_psnr=14.0),
            )

            payload = report.build_report((candidate,), baseline_initialization="legacy_sparse")

        self.assertEqual(payload["status"], "failed")
        self.assertIn("missing baseline initialization legacy_sparse", payload["failures"])

    def test_non_ok_gate1_artifact_fails_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_payload = _gate1_payload(
                initialization="legacy_sparse",
                train_psnr=11.0,
                heldout_psnr=12.0,
            )
            bad_payload["status"] = "failed"
            artifact = _write_json(Path(tmpdir) / "failed.json", bad_payload)

            payload = report.build_report((artifact,), baseline_initialization="legacy_sparse")

        self.assertEqual(payload["status"], "failed")
        self.assertIn("expected status=ok", payload["failures"][0])


if __name__ == "__main__":
    unittest.main()
