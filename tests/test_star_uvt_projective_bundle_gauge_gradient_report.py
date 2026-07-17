from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research_experiments.star_uvt_feature_tubes.projective_bundle_gauge_gradient_report import (
    DEFAULT_OUT_DIR,
    assert_bundle_gauge_gradient_report,
    run_report,
    verify_bundle_gauge_gradient_report,
)


def test_bundle_gauge_gradient_report_accepts_matching_depth_and_log_depth_gradients() -> None:
    report = run_report(samples=1025)

    assert verify_bundle_gauge_gradient_report(report) == []
    assert_bundle_gauge_gradient_report(report)
    assert report["summary"]["max_gradient_rel_error"] < 2.0e-6
    assert report["summary"]["min_bad_no_jacobian_gradient_rel_error"] > 0.05


def test_bundle_gauge_gradient_report_rejects_high_gradient_error() -> None:
    report = run_report(samples=1025)
    report["summary"]["max_gradient_rel_error"] = 1.0e-3

    errors = verify_bundle_gauge_gradient_report(report)

    assert any("max_gradient_rel_error" in error for error in errors)


def test_bundle_gauge_gradient_report_rejects_high_value_error() -> None:
    report = run_report(samples=1025)
    report["summary"]["value_rel_error"] = 1.0e-3

    errors = verify_bundle_gauge_gradient_report(report)

    assert any("value_rel_error" in error for error in errors)


def test_bundle_gauge_gradient_report_rejects_missing_bad_gradient_control() -> None:
    report = run_report(samples=1025)
    report["summary"]["min_bad_no_jacobian_gradient_rel_error"] = 0.0

    errors = verify_bundle_gauge_gradient_report(report)

    assert any("missing-Jacobian failure" in error for error in errors)


def test_bundle_gauge_gradient_report_rejects_bad_finite_difference() -> None:
    report = run_report(samples=1025)
    report["summary"]["finite_difference_mean_x_rel_error"] = 1.0e-3

    errors = verify_bundle_gauge_gradient_report(report)

    assert any("finite_difference_mean_x_rel_error" in error for error in errors)


def test_bundle_gauge_gradient_report_rejects_missing_param_row() -> None:
    report = copy.deepcopy(run_report(samples=1025))
    rows = report["rows"]
    assert isinstance(rows, list)
    rows[0]["param"] = "wrong"

    errors = verify_bundle_gauge_gradient_report(report)

    assert any("rows must cover" in error for error in errors)


def test_bundle_gauge_gradient_report_rejects_nonfinite_row() -> None:
    report = copy.deepcopy(run_report(samples=1025))
    rows = report["rows"]
    assert isinstance(rows, list)
    rows[0]["rel_error"] = float("nan")

    errors = verify_bundle_gauge_gradient_report(report)

    assert any("row 0 rel_error must be finite" in error for error in errors)


def test_bundle_gauge_gradient_report_rejects_nonpositive_gradient_norm() -> None:
    report = copy.deepcopy(run_report(samples=1025))
    rows = report["rows"]
    assert isinstance(rows, list)
    rows[0]["depth_grad_norm"] = 0.0

    errors = verify_bundle_gauge_gradient_report(report)

    assert any("row 0 depth_grad_norm must be positive" in error for error in errors)


def test_bundle_gauge_gradient_report_rejects_inconsistent_finite_difference() -> None:
    report = copy.deepcopy(run_report(samples=1025))
    finite_difference = report["finite_difference"]
    assert isinstance(finite_difference, dict)
    finite_difference["abs_error"] = 1.0

    errors = verify_bundle_gauge_gradient_report(report)

    assert any("finite_difference abs_error mismatch" in error for error in errors)


def test_bundle_gauge_gradient_report_rejects_stale_summary() -> None:
    report = copy.deepcopy(run_report(samples=1025))
    rows = report["rows"]
    assert isinstance(rows, list)
    rows[0]["bad_no_jacobian_rel_error"] = 0.99

    errors = verify_bundle_gauge_gradient_report(report)

    assert any("summary mean_bad_no_jacobian_gradient_rel_error mismatch" in error for error in errors)


def test_saved_bundle_gauge_gradient_artifact_satisfies_contract() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(Path(summary_json).read_text(encoding="utf-8"))

    assert_bundle_gauge_gradient_report(report)
