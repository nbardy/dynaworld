from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research_experiments.star_uvt_feature_tubes.projective_real_video_multiscene_extended_frame_scaling_diagnostic_report import (
    DEFAULT_OUT_DIR,
    EXPECTED_STRICT_TIMING_ERRORS,
    assert_extended_frame_scaling_diagnostic_report,
    summarize,
    verify_extended_frame_scaling_diagnostic_report,
)


def _source_summary() -> dict[str, object]:
    return {
        "scene_count": 5,
        "frame_count_count": 3,
        "row_count": 30,
        "measured_row_count": 15,
        "distinct_youtube_id_count": 5,
        "frame_growth_factor": 4.0,
        "all_source_videos_exist": True,
        "all_rows_pass": True,
        "all_rows_loss_decreased": True,
        "all_rows_no_overflow": True,
        "all_rows_fallback_free": True,
        "all_rows_visibility_stratification_free": True,
        "all_measured_loss_matches_cadence": True,
        "max_measured_vs_cadence_end_loss_abs_delta": 0.0,
        "max_measured_vs_cadence_no_first_step_ms_ratio": 1.18,
        "max_measured_vs_cadence_rebuild_ratio": 0.5,
        "max_measured_no_first_growth_vs_frame_growth_ratio": 1.001,
        "max_measured_cache_rebuild_growth": 1.0,
        "max_measured_support_rebins": 0,
        "max_measured_stale_refreshes": 0,
        "max_measured_support_tail_alpha_bound": 0.0,
        "max_measured_support_overshoot_px": 0.0,
        "max_motion_score": 7.02,
        "max_tile_count": 22,
    }


def _valid_report() -> dict[str, object]:
    report: dict[str, object] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_real_video_multiscene_extended_frame_scaling_diagnostic",
        "base_domain": "failed strict five-source real-video frame-scaling matrix",
        "theory_contract": (
            "This report does not prove broad real-scene quality acceptance and does not prove a timing win. "
            "It preserves the five-source frame-growth run whose failed strict timing verifier is expected."
        ),
        "source_report": "extended5_frame_scaling.json",
        "source_status": "failed",
        "source_errors": list(EXPECTED_STRICT_TIMING_ERRORS),
        "source_summary": _source_summary(),
    }
    report["summary"] = summarize(report)  # type: ignore[arg-type]
    return report


def test_extended_frame_scaling_diagnostic_accepts_expected_timing_failure() -> None:
    report = _valid_report()

    assert verify_extended_frame_scaling_diagnostic_report(report) == []
    assert_extended_frame_scaling_diagnostic_report(report)
    assert report["summary"]["no_first_timing_win"] is False  # type: ignore[index]


def test_extended_frame_scaling_diagnostic_rejects_non_timing_source_failure() -> None:
    report = _valid_report()
    report["source_errors"] = ["some correctness failure"]
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_extended_frame_scaling_diagnostic_report(report)

    assert any("expected strict timing failures" in error for error in errors)


def test_extended_frame_scaling_diagnostic_rejects_correctness_regression() -> None:
    report = _valid_report()
    report["source_summary"]["max_measured_support_rebins"] = 1  # type: ignore[index]
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_extended_frame_scaling_diagnostic_report(report)

    assert any("zero measured support rebins" in error for error in errors)


def test_extended_frame_scaling_diagnostic_rejects_stale_summary() -> None:
    report = copy.deepcopy(_valid_report())
    report["source_summary"]["max_measured_vs_cadence_rebuild_ratio"] = 0.75  # type: ignore[index]

    errors = verify_extended_frame_scaling_diagnostic_report(report)

    assert any("summary" in error and "mismatch" in error for error in errors)


def test_extended_frame_scaling_diagnostic_rejects_now_passing_timing() -> None:
    report = _valid_report()
    report["source_summary"]["max_measured_vs_cadence_no_first_step_ms_ratio"] = 0.95  # type: ignore[index]
    report["source_summary"]["max_measured_no_first_growth_vs_frame_growth_ratio"] = 0.90  # type: ignore[index]
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_extended_frame_scaling_diagnostic_report(report)

    assert any("use the strict frame-scaling verifier instead" in error for error in errors)


def test_saved_extended_frame_scaling_diagnostic_artifact_satisfies_contract() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(Path(summary_json).read_text(encoding="utf-8"))

    assert_extended_frame_scaling_diagnostic_report(report)
