from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research_experiments.star_uvt_feature_tubes.projective_exposure_rolling_quadrature_report import (
    DEFAULT_OUT_DIR,
    assert_exposure_rolling_quadrature_report,
    run_report,
    verify_exposure_rolling_quadrature_report,
)


def test_exposure_rolling_quadrature_report_accepts_reference_lowering() -> None:
    report = run_report(run_metal=False)

    assert verify_exposure_rolling_quadrature_report(report) == []
    assert_exposure_rolling_quadrature_report(report)
    assert report["finite_exposure"]["reference_lowered_max_abs_error"] <= 1.0e-6
    assert report["rolling_shutter"]["unique_to_row_sample_ratio"] < 1.0


def test_exposure_rolling_quadrature_report_rejects_bad_finite_error() -> None:
    report = copy.deepcopy(run_report(run_metal=False))
    report["finite_exposure"]["reference_lowered_max_abs_error"] = 1.0e-3
    report["summary"]["finite_reference_lowered_max_abs_error"] = 1.0e-3

    errors = verify_exposure_rolling_quadrature_report(report)

    assert any("finite reference_lowered_max_abs_error" in error for error in errors)


def test_exposure_rolling_quadrature_report_rejects_lost_rolling_reuse() -> None:
    report = copy.deepcopy(run_report(run_metal=False))
    report["rolling_shutter"]["lowered_unique_sample_count"] = report["rolling_shutter"]["total_row_sample_count"]
    report["rolling_shutter"]["unique_to_row_sample_ratio"] = 1.0
    report["summary"]["rolling_unique_to_row_sample_ratio"] = 1.0

    errors = verify_exposure_rolling_quadrature_report(report)

    assert any("deduplicate row quadrature samples" in error for error in errors)


def test_exposure_rolling_quadrature_report_rejects_bad_row_weights() -> None:
    report = copy.deepcopy(run_report(run_metal=False))
    report["rolling_shutter"]["row_weight_sums"][0] = 0.5

    errors = verify_exposure_rolling_quadrature_report(report)

    assert any("rolling row 0 weight sum" in error for error in errors)


def test_exposure_rolling_quadrature_report_rejects_missing_fallback_cells() -> None:
    report = copy.deepcopy(run_report(run_metal=False))
    report["finite_fallback"]["fallback"]["fallback_cells"] = 0

    errors = verify_exposure_rolling_quadrature_report(report)

    assert any("finite fallback must contain fallback cells" in error for error in errors)


def test_exposure_rolling_quadrature_report_rejects_stale_summary() -> None:
    report = copy.deepcopy(run_report(run_metal=False))
    report["summary"]["rolling_rowwise_batched_max_abs_error"] = 0.25

    errors = verify_exposure_rolling_quadrature_report(report)

    assert any("summary rolling_rowwise_batched_max_abs_error" in error for error in errors)


def test_exposure_rolling_quadrature_report_rejects_stale_complexity_ratio() -> None:
    report = copy.deepcopy(run_report(run_metal=False))
    report["finite_exposure"]["complexity"]["interval_to_dense_trace_sample_ratio"] = 0.75

    errors = verify_exposure_rolling_quadrature_report(report)

    assert any("finite interval_to_dense_trace_sample_ratio" in error for error in errors)


def test_exposure_rolling_quadrature_report_rejects_bad_fallback_sample_counts() -> None:
    report = copy.deepcopy(run_report(run_metal=False))
    report["finite_fallback"]["fallback"]["fallback_trace_samples"] = report["finite_fallback"]["fallback"][
        "total_trace_samples"
    ]

    errors = verify_exposure_rolling_quadrature_report(report)

    assert any("finite fallback fallback_trace_samples" in error for error in errors)


def test_exposure_rolling_quadrature_report_rejects_stale_metal_summary() -> None:
    report = copy.deepcopy(run_report(run_metal=False))
    report["finite_exposure"]["metal_max_abs_error"] = 1.0e-8

    errors = verify_exposure_rolling_quadrature_report(report)

    assert any("summary max_metal_abs_error" in error for error in errors)


def test_exposure_rolling_quadrature_report_rejects_fallback_complexity_mismatch() -> None:
    report = copy.deepcopy(run_report(run_metal=False))
    report["rolling_fallback"]["complexity"]["fallback_cells"] = 0

    errors = verify_exposure_rolling_quadrature_report(report)

    assert any("rolling fallback complexity fallback_cells" in error for error in errors)


def test_saved_exposure_rolling_quadrature_artifact_satisfies_contract() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(Path(summary_json).read_text(encoding="utf-8"))

    assert_exposure_rolling_quadrature_report(report)
