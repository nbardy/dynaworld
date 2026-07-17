from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_experiments.star_uvt_feature_tubes.projective_real_video_multiscene_extended_render_forward_shape_report import (
    DEFAULT_OUT_DIR,
    EXPECTED_STRICT_TIMING_ERRORS,
    assert_extended_render_forward_shape_report,
    run_report,
    summarize,
    verify_extended_render_forward_shape_report,
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
        "scene_000": {4: 1.2, 8: 0.6, 16: 0.7},
        "scene_001": {4: 0.5, 8: 0.6, 16: 0.7},
        "scene_002": {4: 0.4, 8: 1.1, 16: 0.8},
        "scene_003": {4: 0.5, 8: 0.6, 16: 0.7},
        "scene_004": {4: 0.4, 8: 0.5, 16: 0.6},
    }
    measured_no_first = {
        "scene_000": {4: 120.0, 8: 90.0, 16: 200.0},
        "scene_001": {4: 70.0, 8: 80.0, 16: 90.0},
        "scene_002": {4: 60.0, 8: 110.0, 16: 95.0},
        "scene_003": {4: 45.0, 8: 55.0, 16: 65.0},
        "scene_004": {4: 30.0, 8: 40.0, 16: 50.0},
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
                        "end_loss": 0.2 + frames * 0.001,
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


def _timing_row(step_ms: float, render_ms: float) -> dict[str, float]:
    return {
        "step_ms": step_ms,
        "render_forward_ms": render_ms,
        "backward_ms": step_ms * 0.25,
        "colorize_loss_ms": step_ms * 0.05,
        "optimizer_ms": max(0.0, step_ms * 0.70 - render_ms),
    }


def _no_first_rows(no_first_ms: float, policy: str) -> list[dict[str, float]]:
    if policy == "measured":
        return [
            _timing_row(no_first_ms * 2.0, no_first_ms * 1.7),
            _timing_row(no_first_ms * 0.5, no_first_ms * 0.3),
            _timing_row(no_first_ms * 0.5, no_first_ms * 0.2),
        ]
    return [
        _timing_row(no_first_ms * 0.8, no_first_ms * 0.4),
        _timing_row(no_first_ms * 1.1, no_first_ms * 0.8),
        _timing_row(no_first_ms * 1.1, no_first_ms * 0.8),
    ]


def _write_case(case_dir: Path, scene_id: str, frames: int, policy: str, no_first_ms: float) -> None:
    rows = [_timing_row(no_first_ms * 2.5, no_first_ms * 1.6)]
    rows.extend(_no_first_rows(no_first_ms, policy))
    path = case_dir / f"{scene_id}_{frames}f_{policy}.json"
    path.write_text(json.dumps({"step_timings_ms": rows, "chunk_traces": []}, indent=2) + "\n", encoding="utf-8")


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


def test_render_forward_shape_accepts_single_spike_misses(tmp_path: Path) -> None:
    source_path, case_dir = _write_fixture(tmp_path)

    report = run_report(source_report=source_path, case_dir=case_dir)

    assert verify_extended_render_forward_shape_report(report) == []
    assert_extended_render_forward_shape_report(report)
    assert report["summary"]["no_first_miss_pair_count"] == 2
    assert report["summary"]["all_no_first_misses_render_single_spike_driven"] is True
    assert report["summary"]["chunk_traces_present_pair_count"] == 0


def test_render_forward_shape_rejects_stale_summary(tmp_path: Path) -> None:
    source_path, case_dir = _write_fixture(tmp_path)
    report = run_report(source_report=source_path, case_dir=case_dir)
    report["pair_profiles"][0]["render_forward_step_ratios"][0] = 99.0

    errors = verify_extended_render_forward_shape_report(report)

    assert any("does not match render_forward_step_ratios" in error for error in errors)


def test_render_forward_shape_rejects_non_spike_miss(tmp_path: Path) -> None:
    source_path, case_dir = _write_fixture(tmp_path)
    report = run_report(source_report=source_path, case_dir=case_dir)
    miss = next(row for row in report["pair_profiles"] if row["no_first_timing_miss"])
    miss["render_forward_single_spike_drives_miss"] = False
    report["summary"] = summarize(report)

    errors = verify_extended_render_forward_shape_report(report)

    assert any("single-spike driven" in error for error in errors)


def test_render_forward_shape_rejects_present_chunk_traces(tmp_path: Path) -> None:
    source_path, case_dir = _write_fixture(tmp_path)
    report = run_report(source_report=source_path, case_dir=case_dir)
    report["pair_profiles"][0]["chunk_traces_present"] = True
    report["summary"] = summarize(report)

    errors = verify_extended_render_forward_shape_report(report)

    assert any("should not already contain chunk traces" in error for error in errors)


def test_render_forward_shape_rejects_dirty_pair(tmp_path: Path) -> None:
    source_path, case_dir = _write_fixture(tmp_path)
    report = run_report(source_report=source_path, case_dir=case_dir)
    report["pair_profiles"][0]["measured_support_rebins"] = 1
    report["summary"] = summarize(report)

    errors = verify_extended_render_forward_shape_report(report)

    assert any("pairs must remain cache/support clean" in error for error in errors)


def test_saved_render_forward_shape_artifact_satisfies_contract() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(Path(summary_json).read_text(encoding="utf-8"))

    assert_extended_render_forward_shape_report(report)
