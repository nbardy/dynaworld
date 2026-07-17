from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import report_worldfoam_next_mps_candidate_readiness as report


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _quality_bridge(*, candidate: str = "legacy_pixel_mean", positive: bool = True) -> dict[str, object]:
    return {
        "status": "ok",
        "next_mps_candidate": candidate,
        "rows": [
            {
                "site_initialization": candidate,
                "positive_cpu_reference_candidate": positive,
                "train_psnr_delta_vs_baseline": 1.0,
                "heldout_psnr_delta_vs_baseline": 2.0,
            }
        ],
    }


def _topology(*, initialization: str = "legacy_pixel_mean", accepted: bool = True) -> dict[str, object]:
    return {
        "status": "ok",
        "site_initialization": initialization,
        "frame_counts": [2, 4],
        "candidate_count_scale_first_to_last": 0.99,
        "storage_scale_first_to_last": 0.98,
        "acceptance": {
            "all_rows_ok": True,
            "candidate_count_sublinear_vs_frame_count": accepted,
            "storage_sublinear_vs_frame_count": True,
        },
    }


class WorldFoamNextMpsCandidateReadinessTests(unittest.TestCase):
    def test_readiness_passes_for_matching_positive_quality_and_topology(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            quality = _write_json(tmp / "quality.json", _quality_bridge())
            topology = _write_json(tmp / "topology.json", _topology())

            payload = report.build_report(quality_bridge_path=quality, topology_artifact_path=topology)

        self.assertEqual(payload["status"], "ok", payload["failures"])
        self.assertEqual(payload["next_mps_candidate"], "legacy_pixel_mean")
        self.assertTrue(payload["ready_for_quiet_mps_quality_speed_run"])
        self.assertFalse(payload["quality_claim"])
        self.assertFalse(payload["speed_claim"])

    def test_readiness_fails_when_topology_initializer_does_not_match_quality_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            quality = _write_json(tmp / "quality.json", _quality_bridge(candidate="legacy_pixel_mean"))
            topology = _write_json(tmp / "topology.json", _topology(initialization="stratified_grid"))

            payload = report.build_report(quality_bridge_path=quality, topology_artifact_path=topology)

        self.assertEqual(payload["status"], "failed")
        self.assertIn("topology artifact site_initialization does not match", payload["failures"][0])

    def test_readiness_fails_when_quality_candidate_is_not_positive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            quality = _write_json(tmp / "quality.json", _quality_bridge(positive=False))
            topology = _write_json(tmp / "topology.json", _topology())

            payload = report.build_report(quality_bridge_path=quality, topology_artifact_path=topology)

        self.assertEqual(payload["status"], "failed")
        self.assertIn("is not positive_cpu_reference_candidate", payload["failures"][0])

    def test_readiness_fails_when_topology_acceptance_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            quality = _write_json(tmp / "quality.json", _quality_bridge())
            topology = _write_json(tmp / "topology.json", _topology(accepted=False))

            payload = report.build_report(quality_bridge_path=quality, topology_artifact_path=topology)

        self.assertEqual(payload["status"], "failed")
        self.assertIn(
            "topology acceptance failed: candidate_count_sublinear_vs_frame_count",
            payload["failures"],
        )


if __name__ == "__main__":
    unittest.main()
