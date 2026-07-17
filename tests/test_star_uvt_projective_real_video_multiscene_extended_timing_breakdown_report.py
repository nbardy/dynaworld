from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research_experiments.star_uvt_feature_tubes.projective_real_video_multiscene_extended_timing_breakdown_report import (
    DEFAULT_OUT_DIR,
    EXPECTED_STRICT_TIMING_ERRORS,
    _build_pair_breakdowns,
    _build_scene_breakdowns,
    assert_extended_timing_breakdown_report,
    summarize,
    verify_extended_timing_breakdown_report,
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
                        "mean_step_ms": no_first + 100.0,
                        "mean_render_forward_ms": no_first * 0.5,
                        "mean_backward_ms": no_first * 0.25,
                        "projective_interval_cache_rebuilds": rebuilds,
                        "projective_interval_cache_live_updates": live_updates,
                        "projective_interval_cache_staleness_checks": live_updates,
                        "projective_interval_cache_support_rebins": 0,
                        "projective_interval_cache_stale_refreshes": 0,
                        "projective_interval_cache_fallback_marks": 0,
                        "projective_interval_cache_visibility_stratifications": 0,
                        "projective_interval_cache_max_support_tail_alpha_bound": 0.0,
                        "projective_interval_cache_max_support_max_overshoot_px": 0.0,
                        "tile_overflow_sum": 0,
                        "max_tile_count": 22,
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


def _valid_report() -> dict[str, object]:
    source = _source_report()
    pairs = _build_pair_breakdowns(source)
    scenes = _build_scene_breakdowns(source, pairs)
    report: dict[str, object] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_real_video_multiscene_extended_timing_breakdown",
        "base_domain": "failed strict five-source real-video frame-scaling matrix",
        "theory_contract": (
            "This report is a pair-level timing breakdown. It does not prove a timing win; "
            "it checks timing misses against cache/support churn."
        ),
        "source_report": "extended5_frame_scaling.json",
        "source_status": "failed",
        "source_errors": list(EXPECTED_STRICT_TIMING_ERRORS),
        "source_summary": source["summary"],
        "pair_breakdowns": pairs,
        "scene_breakdowns": scenes,
    }
    report["summary"] = summarize(report)  # type: ignore[arg-type]
    return report


def test_extended_timing_breakdown_accepts_pair_level_timing_miss_diagnostic() -> None:
    report = _valid_report()

    assert verify_extended_timing_breakdown_report(report) == []
    assert_extended_timing_breakdown_report(report)
    assert report["summary"]["no_first_ratio_gt1_count"] == 3  # type: ignore[index]
    assert report["summary"]["growth_ratio_gt1_count"] == 1  # type: ignore[index]
    assert report["summary"]["all_failing_pairs_cache_clean"] is True  # type: ignore[index]


def test_extended_timing_breakdown_rejects_stale_summary() -> None:
    report = copy.deepcopy(_valid_report())
    report["pair_breakdowns"][0]["measured_vs_cadence_no_first_step_ms_ratio"] = 2.0  # type: ignore[index]

    errors = verify_extended_timing_breakdown_report(report)

    assert any("summary" in error and "mismatch" in error for error in errors)


def test_extended_timing_breakdown_rejects_support_churn_on_failing_pair() -> None:
    report = copy.deepcopy(_valid_report())
    for pair in report["pair_breakdowns"]:  # type: ignore[index]
        if pair["no_first_timing_miss"]:
            pair["measured_support_rebins"] = 1
            break
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_extended_timing_breakdown_report(report)

    assert any("failing timing pairs must remain cache/support clean" in error for error in errors)


def test_extended_timing_breakdown_rejects_missing_no_first_timing_miss() -> None:
    report = copy.deepcopy(_valid_report())
    for pair in report["pair_breakdowns"]:  # type: ignore[index]
        pair["measured_vs_cadence_no_first_step_ms_ratio"] = 0.5
        pair["no_first_timing_miss"] = False
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_extended_timing_breakdown_report(report)

    assert any("preserve at least one no-first timing miss" in error for error in errors)


def test_extended_timing_breakdown_rejects_missing_frame_growth_timing_miss() -> None:
    report = copy.deepcopy(_valid_report())
    for scene in report["scene_breakdowns"]:  # type: ignore[index]
        scene["measured_no_first_growth_vs_frame_growth_ratio"] = 0.75
        scene["growth_timing_miss"] = False
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_extended_timing_breakdown_report(report)

    assert any("preserve at least one frame-growth timing miss" in error for error in errors)


def test_saved_extended_timing_breakdown_artifact_satisfies_contract() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(Path(summary_json).read_text(encoding="utf-8"))

    assert_extended_timing_breakdown_report(report)
