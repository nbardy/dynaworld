from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research_experiments.star_uvt_feature_tubes.projective_exposure_rolling_mixed_fallback_backward_report import (
    DEFAULT_OUT_DIR,
    assert_mixed_fallback_backward_report,
    run_report,
    verify_mixed_fallback_backward_report,
)


def test_mixed_fallback_backward_report_accepts_reference_only() -> None:
    report = run_report(run_metal=False)

    assert verify_mixed_fallback_backward_report(report) == []
    assert_mixed_fallback_backward_report(report)
    assert report["finite_mixed_fallback_backward"]["fallback"]["fallback_cells"] > 0
    assert report["rolling_mixed_fallback_backward"]["unique_to_row_sample_ratio"] < 1.0


def test_mixed_fallback_backward_report_accepts_mps_if_available() -> None:
    report = run_report(run_metal=True)

    assert_mixed_fallback_backward_report(report)
    if report["summary"]["mixed_backward_case_count"]:
        assert report["summary"]["max_mixed_grad_abs_error"] <= 2.0e-3
        assert report["summary"]["max_mixed_grad_rel_error"] <= 1.0e-2


def test_mixed_fallback_backward_report_rejects_missing_fallback() -> None:
    report = copy.deepcopy(run_report(run_metal=False))
    report["finite_mixed_fallback_backward"]["fallback"]["fallback_cells"] = 0

    errors = verify_mixed_fallback_backward_report(report)

    assert any("finite fallback_cells" in error for error in errors)


def test_mixed_fallback_backward_report_rejects_zero_reference_grad() -> None:
    report = copy.deepcopy(run_report(run_metal=False))
    report["rolling_mixed_fallback_backward"]["reference_grad_norms"]["coeffs"] = 0.0

    errors = verify_mixed_fallback_backward_report(report)

    assert any("rolling coeffs reference grad norm" in error for error in errors)


def test_mixed_fallback_backward_report_rejects_bad_mixed_compare() -> None:
    report = copy.deepcopy(run_report(run_metal=False))
    report["finite_mixed_fallback_backward"]["mixed_output_max_abs_error"] = 1.0e-3
    report["finite_mixed_fallback_backward"]["mixed_compare"] = {
        "max_abs_error": 3.0e-3,
        "max_rel_error": 2.0e-2,
        "coeffs": {"max_abs_error": 3.0e-3, "max_rel_error": 2.0e-2, "reference_norm": 1.0},
        "opacity": {"max_abs_error": 0.0, "max_rel_error": 0.0, "reference_norm": 1.0},
        "color": {"max_abs_error": 0.0, "max_rel_error": 0.0, "reference_norm": 1.0},
    }
    report["summary"]["finite_has_mixed_backward"] = True
    report["summary"]["mixed_backward_case_count"] = 1
    report["summary"]["max_mixed_grad_abs_error"] = 3.0e-3
    report["summary"]["max_mixed_grad_rel_error"] = 2.0e-2
    report["summary"]["max_mixed_output_abs_error"] = 1.0e-3

    errors = verify_mixed_fallback_backward_report(report)

    assert any("finite max_abs_error" in error for error in errors)
    assert any("finite max_rel_error" in error for error in errors)
    assert any("finite mixed output error" in error for error in errors)


def test_mixed_fallback_backward_report_rejects_stale_summary() -> None:
    report = copy.deepcopy(run_report(run_metal=False))
    report["summary"]["rolling_unique_to_row_sample_ratio"] = 0.25

    errors = verify_mixed_fallback_backward_report(report)

    assert any("summary rolling_unique_to_row_sample_ratio" in error for error in errors)


def test_saved_mixed_fallback_backward_artifact_satisfies_contract() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(Path(summary_json).read_text(encoding="utf-8"))

    assert_mixed_fallback_backward_report(report)
