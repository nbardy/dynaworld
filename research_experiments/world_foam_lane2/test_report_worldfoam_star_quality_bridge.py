from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import report_worldfoam_star_quality_bridge as bridge


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _worldfoam_payload(
    *,
    train_psnr: float = 12.25,
    quality_claim: bool = False,
    render_size: int = 64,
    site_count: int = 24,
    frame_counts: tuple[int, ...] = (2, 16),
) -> dict[str, object]:
    rows = []
    for index, frame_count in enumerate(frame_counts):
        rows.append(
            {
                "status": "ok",
                "frame_count": frame_count,
                "loaded_frame_count": frame_count,
                "repeat_loaded_frames": False,
                "final_train_psnr": train_psnr if index == len(frame_counts) - 1 else train_psnr - 0.5,
                "final_heldout_psnr": 12.85 if index == len(frame_counts) - 1 else 12.35,
            }
        )
    return {
        "status": "ok",
        "quality_claim": quality_claim,
        "render_size": render_size,
        "site_count": site_count,
        "tape_mode": "owner-run-delta-packed-factorized-framebitmask-recompute-fused-mse-nomid",
        "rows": rows,
    }


def _star_comparison_payload() -> dict[str, object]:
    return {
        "status": "ok",
        "comparison": {
            "total_median_ms_ratio_star_over_worldfoam_by_frame": {"2": 1.5, "16": 2.2},
            "backward_median_ms_ratio_star_over_worldfoam_by_frame": {"2": 1.1, "16": 1.8},
        },
    }


class WorldFoamStarQualityBridgeTests(unittest.TestCase):
    def test_speed_win_does_not_imply_star_quality_competitiveness(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            worldfoam_path = _write_json(tmp_path / "worldfoam.json", _worldfoam_payload())
            star_path = _write_json(tmp_path / "star.json", _star_comparison_payload())

            report = bridge.build_report(
                worldfoam_artifact=worldfoam_path,
                star_comparison_artifact=star_path,
                star_source_rgb_psnr=29.823,
                solid_source_rgb_psnr=21.36,
                quality_gap_tolerance=1.0,
            )

        self.assertEqual(report["status"], "ok")
        self.assertTrue(report["speed_bridge"]["speed_competitive_micro_gate"])
        self.assertFalse(report["quality_competitive_with_star_source"])
        self.assertFalse(report["quality_competitive_with_solid_same_source"])
        self.assertFalse(report["star_uvt_competitive_claim"])
        self.assertAlmostEqual(report["quality_gaps"]["train_psnr_gap_to_star_uvt_source"], 17.573)

    def test_high_quality_artifact_can_be_marked_competitive_when_speed_also_wins(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            worldfoam_path = _write_json(tmp_path / "worldfoam.json", _worldfoam_payload(train_psnr=29.2))
            star_path = _write_json(tmp_path / "star.json", _star_comparison_payload())

            report = bridge.build_report(
                worldfoam_artifact=worldfoam_path,
                star_comparison_artifact=star_path,
                star_source_rgb_psnr=29.823,
                solid_source_rgb_psnr=21.36,
                quality_gap_tolerance=1.0,
            )

        self.assertEqual(report["status"], "ok")
        self.assertTrue(report["quality_competitive_with_star_source"])
        self.assertTrue(report["quality_competitive_with_solid_same_source"])
        self.assertTrue(report["star_uvt_competitive_claim"])

    def test_worldfoam_quality_claim_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            worldfoam_path = _write_json(tmp_path / "worldfoam.json", _worldfoam_payload(quality_claim=True))
            star_path = _write_json(tmp_path / "star.json", _star_comparison_payload())

            report = bridge.build_report(
                worldfoam_artifact=worldfoam_path,
                star_comparison_artifact=star_path,
                star_source_rgb_psnr=29.823,
                solid_source_rgb_psnr=21.36,
                quality_gap_tolerance=1.0,
            )

        self.assertEqual(report["status"], "failed")
        self.assertIn("WorldFoam artifact unexpectedly claims quality parity", report["failures"])

    def test_extra_capacity_candidate_is_reported_without_overriding_primary_speed_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            worldfoam_path = _write_json(tmp_path / "worldfoam.json", _worldfoam_payload(train_psnr=12.25))
            capacity_path = _write_json(
                tmp_path / "capacity.json",
                _worldfoam_payload(train_psnr=10.0, render_size=96, site_count=48),
            )
            star_path = _write_json(tmp_path / "star.json", _star_comparison_payload())

            report = bridge.build_report(
                worldfoam_artifact=worldfoam_path,
                extra_worldfoam_artifacts=(capacity_path,),
                star_comparison_artifact=star_path,
                star_source_rgb_psnr=29.823,
                solid_source_rgb_psnr=21.36,
                quality_gap_tolerance=1.0,
            )

        self.assertEqual(report["status"], "ok")
        self.assertFalse(report["capacity_candidates_improve_train_psnr"])
        self.assertEqual(report["capacity_candidate_count"], 1)
        self.assertEqual(report["capacity_candidates"][0]["render_size"], 96)
        self.assertEqual(report["capacity_candidates"][0]["site_count"], 48)
        self.assertTrue(report["capacity_candidates"][0]["primary_frame_comparison"]["same_frame_set_as_primary"])
        self.assertFalse(report["capacity_candidates_improve_train_psnr_on_any_common_frame"])
        self.assertFalse(report["capacity_candidates_improve_train_psnr_on_all_common_frames"])
        self.assertEqual(report["best_worldfoam_quality_artifact"], str(worldfoam_path))
        self.assertTrue(report["speed_bridge"]["speed_competitive_micro_gate"])
        self.assertFalse(report["star_uvt_competitive_claim"])
        self.assertFalse(report["best_worldfoam_quality_needs_matched_speed_gate"])

    def test_high_quality_capacity_candidate_needs_its_own_matched_speed_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            worldfoam_path = _write_json(tmp_path / "worldfoam.json", _worldfoam_payload(train_psnr=12.25))
            capacity_path = _write_json(
                tmp_path / "capacity.json",
                _worldfoam_payload(train_psnr=29.2, render_size=96, site_count=48),
            )
            star_path = _write_json(tmp_path / "star.json", _star_comparison_payload())

            report = bridge.build_report(
                worldfoam_artifact=worldfoam_path,
                extra_worldfoam_artifacts=(capacity_path,),
                star_comparison_artifact=star_path,
                star_source_rgb_psnr=29.823,
                solid_source_rgb_psnr=21.36,
                quality_gap_tolerance=1.0,
            )

        self.assertEqual(report["status"], "ok")
        self.assertFalse(report["star_uvt_competitive_claim"])
        self.assertTrue(report["capacity_candidates_improve_train_psnr"])
        self.assertEqual(report["best_worldfoam_quality_artifact"], str(capacity_path))
        self.assertFalse(report["best_worldfoam_quality_is_primary_speed_artifact"])
        self.assertTrue(report["best_worldfoam_quality_competitive_with_star_source"])
        self.assertTrue(report["best_worldfoam_quality_needs_matched_speed_gate"])

    def test_capacity_candidate_missing_primary_frames_is_not_silently_full_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            worldfoam_path = _write_json(tmp_path / "worldfoam.json", _worldfoam_payload(train_psnr=12.25))
            capacity_path = _write_json(
                tmp_path / "capacity.json",
                _worldfoam_payload(train_psnr=13.0, render_size=96, site_count=48, frame_counts=(2,)),
            )
            star_path = _write_json(tmp_path / "star.json", _star_comparison_payload())

            report = bridge.build_report(
                worldfoam_artifact=worldfoam_path,
                extra_worldfoam_artifacts=(capacity_path,),
                star_comparison_artifact=star_path,
                star_source_rgb_psnr=29.823,
                solid_source_rgb_psnr=21.36,
                quality_gap_tolerance=1.0,
            )

        candidate_comparison = report["capacity_candidates"][0]["primary_frame_comparison"]
        self.assertEqual(report["status"], "ok")
        self.assertTrue(report["capacity_candidates_improve_train_psnr"])
        self.assertEqual(report["capacity_candidate_artifacts_missing_primary_frames"], [str(capacity_path)])
        self.assertFalse(candidate_comparison["same_frame_set_as_primary"])
        self.assertEqual(candidate_comparison["common_frame_counts_with_primary"], [2])
        self.assertEqual(candidate_comparison["missing_primary_frame_counts"], [16])
        self.assertTrue(candidate_comparison["improves_train_psnr_on_any_common_frame"])
        self.assertTrue(candidate_comparison["improves_train_psnr_on_all_common_frames"])
        self.assertFalse(
            report["best_worldfoam_quality_primary_frame_comparison"]["same_frame_set_as_primary"]
        )

    def test_missing_rows_fail_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            worldfoam_path = _write_json(tmp_path / "worldfoam.json", {"status": "ok", "rows": []})

            report = bridge.build_report(
                worldfoam_artifact=worldfoam_path,
                star_comparison_artifact=None,
                star_source_rgb_psnr=29.823,
                solid_source_rgb_psnr=21.36,
                quality_gap_tolerance=1.0,
            )

        self.assertEqual(report["status"], "failed")
        self.assertIn("WorldFoam artifact must contain at least one row", report["failures"][0])


if __name__ == "__main__":
    unittest.main()
