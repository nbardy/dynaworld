from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research_experiments.star_uvt_feature_tubes.projective_real_video_frame_count_breadth_diagnostic_report import (
    DEFAULT_OUT_DIR,
    assert_frame_count_breadth_diagnostic_report,
    summarize,
    verify_frame_count_breadth_diagnostic_report,
)


def _source_summary() -> dict[str, object]:
    return {
        "scene_count": 3,
        "distinct_youtube_id_count": 3,
        "row_count": 24,
        "measured_row_count": 12,
        "frame_count_count": 4,
        "frame_growth_factor": 8.0,
        "all_source_videos_exist": True,
        "all_rows_pass": True,
        "all_rows_loss_decreased": True,
        "all_rows_no_overflow": True,
        "all_rows_fallback_free": True,
        "all_rows_visibility_stratification_free": True,
        "all_measured_loss_matches_cadence": True,
        "max_measured_vs_cadence_end_loss_abs_delta": 1.0e-8,
        "max_measured_vs_cadence_rebuild_ratio": 0.5,
        "max_measured_cache_rebuild_growth": 1.0,
        "max_measured_support_rebins": 0,
        "max_measured_stale_refreshes": 0,
        "max_measured_support_tail_alpha_bound": 0.0,
        "max_measured_support_overshoot_px": 0.0,
        "max_motion_score": 5.8,
        "max_tile_count": 22,
        "max_measured_vs_cadence_no_first_step_ms_ratio": 1.34,
        "max_measured_no_first_growth_vs_frame_growth_ratio": 0.23,
    }


def _valid_report() -> dict[str, object]:
    report: dict[str, object] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_real_video_frame_count_breadth_diagnostic",
        "base_domain": "failed strict source-distinct real-video four-frame-count matrix",
        "theory_contract": (
            "This report does not prove broad real-scene quality acceptance and does not prove a strict timing win. "
            "It accepts frame-count breadth."
        ),
        "source_report": "source.json",
        "source_status": "failed",
        "source_errors": ["multiscene frame-scaling measured no-first timing must beat cadence"],
        "source_summary": _source_summary(),
    }
    report["summary"] = summarize(report)  # type: ignore[arg-type]
    return report


def test_frame_count_breadth_diagnostic_accepts_valid_payload() -> None:
    report = _valid_report()

    assert verify_frame_count_breadth_diagnostic_report(report) == []
    assert_frame_count_breadth_diagnostic_report(report)
    assert report["summary"]["source_frame_count_count"] == 4  # type: ignore[index]
    assert report["summary"]["frame_count_breadth_accepted"] is True  # type: ignore[index]


def test_frame_count_breadth_diagnostic_rejects_low_frame_count() -> None:
    report = _valid_report()
    report["source_summary"]["frame_count_count"] = 3  # type: ignore[index]
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_frame_count_breadth_diagnostic_report(report)

    assert any("at least four frame counts" in error for error in errors)


def test_frame_count_breadth_diagnostic_rejects_non_timing_source_error() -> None:
    report = _valid_report()
    report["source_errors"] = ["tile overflow"]
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_frame_count_breadth_diagnostic_report(report)

    assert any("only expected strict timing failures" in error for error in errors)


def test_frame_count_breadth_diagnostic_rejects_support_churn() -> None:
    report = _valid_report()
    report["source_summary"]["max_measured_support_rebins"] = 1  # type: ignore[index]
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_frame_count_breadth_diagnostic_report(report)

    assert any("zero measured support rebins" in error for error in errors)


def test_frame_count_breadth_diagnostic_rejects_stale_summary() -> None:
    report = copy.deepcopy(_valid_report())
    report["source_summary"]["max_measured_vs_cadence_rebuild_ratio"] = 0.75  # type: ignore[index]

    errors = verify_frame_count_breadth_diagnostic_report(report)

    assert any("summary" in error and "mismatch" in error for error in errors)


def test_saved_frame_count_breadth_diagnostic_artifact_satisfies_contract() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(Path(summary_json).read_text(encoding="utf-8"))

    assert_frame_count_breadth_diagnostic_report(report)
