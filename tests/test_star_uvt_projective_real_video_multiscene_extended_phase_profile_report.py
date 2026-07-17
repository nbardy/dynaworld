from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research_experiments.star_uvt_feature_tubes.projective_real_video_multiscene_extended_phase_profile_report import (
    DEFAULT_OUT_DIR,
    EXPECTED_STRICT_TIMING_ERRORS,
    assert_extended_phase_profile_report,
    run_report,
    summarize,
    verify_extended_phase_profile_report,
)


FRAME_COUNTS = (4, 8, 16)


def _source_report() -> dict[str, object]:
    scenes = [
        {
            "scene_id": f"scene_{idx:03d}",
            "youtube_id": f"yt_{idx:03d}",
            "title": f"Scene {idx}",
            "motion_score": float(idx + 1),
            "source_video_exists": True,
        }
        for idx in range(5)
    ]
    ratios = {
        "scene_000": {4: 1.18, 8: 0.20, 16: 1.14},
        "scene_001": {4: 0.40, 8: 0.60, 16: 0.54},
        "scene_002": {4: 0.50, 8: 1.02, 16: 0.60},
        "scene_003": {4: 0.30, 8: 0.55, 16: 0.40},
        "scene_004": {4: 0.20, 8: 0.40, 16: 0.70},
    }
    measured_no_first = {
        "scene_000": {4: 118.0, 8: 40.0, 16: 456.0},
        "scene_001": {4: 100.0, 8: 160.0, 16: 401.0},
        "scene_002": {4: 80.0, 8: 204.0, 16: 300.0},
        "scene_003": {4: 60.0, 8: 110.0, 16: 200.0},
        "scene_004": {4: 40.0, 8: 120.0, 16: 120.0},
    }
    rows: list[dict[str, object]] = []
    for scene in scenes:
        scene_id = str(scene["scene_id"])
        for frames in FRAME_COUNTS:
            measured_ms = measured_no_first[scene_id][frames]
            cadence_ms = measured_ms / ratios[scene_id][frames]
            for policy, no_first, rebuilds, live_updates in (
                ("cadence", cadence_ms, 2, 2),
                ("measured", measured_ms, 1, 3),
            ):
                rows.append(
                    {
                        "scene_id": scene_id,
                        "youtube_id": scene["youtube_id"],
                        "title": scene["title"],
                        "frames": frames,
                        "policy": policy,
                        "motion_score": scene["motion_score"],
                        "no_first_step_ms": no_first,
                        "projective_interval_cache_rebuilds": rebuilds,
                        "projective_interval_cache_live_updates": live_updates,
                        "projective_interval_cache_staleness_checks": live_updates,
                        "projective_interval_cache_support_rebins": 0,
                        "projective_interval_cache_stale_refreshes": 0,
                        "projective_interval_cache_fallback_marks": 0,
                        "projective_interval_cache_visibility_stratifications": 0,
                        "tile_overflow_sum": 0,
                        "end_loss": 0.25 + frames * 0.001,
                    }
                )
    return {
        "status": "failed",
        "benchmark": "star_uvt_projective_real_video_multiscene_frame_scaling_matrix",
        "errors": list(EXPECTED_STRICT_TIMING_ERRORS),
        "scenes": scenes,
        "frame_counts": list(FRAME_COUNTS),
        "rows": rows,
        "summary": {
            "scene_count": len(scenes),
            "distinct_youtube_id_count": len(scenes),
            "row_count": len(rows),
            "frame_count_count": len(FRAME_COUNTS),
            "frame_growth_factor": 4.0,
        },
    }


def _phase_row(step_ms: float, *, render_frac: float = 0.60, backward_frac: float = 0.30) -> dict[str, float]:
    render = step_ms * render_frac
    backward = step_ms * backward_frac
    colorize = step_ms * 0.05
    optimizer = max(0.0, step_ms - render - backward - colorize)
    return {
        "step_ms": step_ms,
        "render_forward_ms": render,
        "backward_ms": backward,
        "colorize_loss_ms": colorize,
        "optimizer_ms": optimizer,
    }


def _write_case(case_dir: Path, scene_id: str, frames: int, policy: str, no_first_ms: float) -> None:
    rows = [_phase_row(no_first_ms * 3.0)]
    for _ in range(3):
        if policy == "measured":
            rows.append(_phase_row(no_first_ms, render_frac=0.70, backward_frac=0.20))
        else:
            rows.append(_phase_row(no_first_ms, render_frac=0.55, backward_frac=0.35))
    path = case_dir / f"{scene_id}_{frames}f_{policy}.json"
    path.write_text(json.dumps({"step_timings_ms": rows}, indent=2) + "\n", encoding="utf-8")


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    source = _source_report()
    source_path = tmp_path / "source.json"
    case_dir = tmp_path / "cases"
    case_dir.mkdir()
    source_path.write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    for row in source["rows"]:  # type: ignore[index]
        _write_case(
            case_dir,
            str(row["scene_id"]),
            int(row["frames"]),
            str(row["policy"]),
            float(row["no_first_step_ms"]),
        )
    return source_path, case_dir


def test_extended_phase_profile_accepts_saved_step_timing_diagnostic(tmp_path: Path) -> None:
    source_path, case_dir = _write_fixture(tmp_path)

    report = run_report(source_report=source_path, case_dir=case_dir)

    assert verify_extended_phase_profile_report(report) == []
    assert_extended_phase_profile_report(report)
    assert report["summary"]["no_first_miss_profile_count"] == 3
    assert report["summary"]["growth_endpoint_profile_count"] == 2
    assert report["summary"]["all_profile_step_no_first_matches_source"] is True


def test_extended_phase_profile_rejects_stale_summary(tmp_path: Path) -> None:
    source_path, case_dir = _write_fixture(tmp_path)
    report = run_report(source_report=source_path, case_dir=case_dir)
    report["phase_profiles"][0]["phase_ratios"]["render_forward_ms"] = 9.0

    errors = verify_extended_phase_profile_report(report)

    assert any("summary" in error and "mismatch" in error for error in errors)


def test_extended_phase_profile_rejects_source_case_timing_mismatch(tmp_path: Path) -> None:
    source_path, case_dir = _write_fixture(tmp_path)
    report = run_report(source_report=source_path, case_dir=case_dir)
    report["phase_profiles"][0]["source_case_no_first_abs_delta"] = 1.0
    report["summary"] = summarize(report)

    errors = verify_extended_phase_profile_report(report)

    assert any("case step no-first mean must match source row" in error for error in errors)


def test_extended_phase_profile_rejects_missing_growth_endpoint(tmp_path: Path) -> None:
    source_path, case_dir = _write_fixture(tmp_path)
    report = run_report(source_report=source_path, case_dir=case_dir)
    for row in report["phase_profiles"]:
        row["reasons"] = [reason for reason in row["reasons"] if reason != "growth_endpoint"]
    report["summary"] = summarize(report)

    errors = verify_extended_phase_profile_report(report)

    assert any("include frame-growth endpoint rows" in error for error in errors)


def test_extended_phase_profile_rejects_dirty_profile_pair(tmp_path: Path) -> None:
    source_path, case_dir = _write_fixture(tmp_path)
    report = run_report(source_report=source_path, case_dir=case_dir)
    report["phase_profiles"][0]["measured_support_rebins"] = 1
    report["summary"] = summarize(report)

    errors = verify_extended_phase_profile_report(report)

    assert any("phase profile pairs must remain cache/support clean" in error for error in errors)


def test_saved_extended_phase_profile_artifact_satisfies_contract() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(Path(summary_json).read_text(encoding="utf-8"))

    assert_extended_phase_profile_report(report)
