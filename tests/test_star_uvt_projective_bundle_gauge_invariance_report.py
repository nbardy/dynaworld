from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research_experiments.star_uvt_feature_tubes.projective_bundle_gauge_invariance_report import (
    DEFAULT_OUT_DIR,
    assert_bundle_gauge_invariance_report,
    run_report,
    verify_bundle_gauge_invariance_report,
)


def test_bundle_gauge_invariance_report_accepts_revolving_camera_pushforward() -> None:
    report = run_report(samples=1025)

    assert verify_bundle_gauge_invariance_report(report) == []
    assert_bundle_gauge_invariance_report(report)
    assert report["summary"]["max_rel_error"] < 2.0e-6
    assert report["summary"]["min_bad_no_jacobian_rel_error"] > 0.05


def test_bundle_gauge_invariance_report_rejects_missing_jacobian_control() -> None:
    report = run_report(samples=1025)
    report["summary"]["min_bad_no_jacobian_rel_error"] = 0.0

    errors = verify_bundle_gauge_invariance_report(report)

    assert any("missing-Jacobian failure" in error for error in errors)


def test_bundle_gauge_invariance_report_rejects_lost_monotone_order_certificate() -> None:
    report = run_report(samples=1025)
    report["summary"]["monotone_log_order_preserved"] = False

    errors = verify_bundle_gauge_invariance_report(report)

    assert any("preserve depth order" in error for error in errors)


def test_bundle_gauge_invariance_report_rejects_orientation_reversing_gauge_as_valid() -> None:
    report = run_report(samples=1025)
    report["summary"]["orientation_reversing_neg_log_order_flipped"] = False

    errors = verify_bundle_gauge_invariance_report(report)

    assert any("orientation-reversing gauge" in error for error in errors)


def test_bundle_gauge_invariance_report_rejects_high_reparameterization_error() -> None:
    report = run_report(samples=1025)
    report["summary"]["max_rel_error"] = 1.0e-3

    errors = verify_bundle_gauge_invariance_report(report)

    assert any("max_rel_error" in error for error in errors)


def test_bundle_gauge_invariance_report_rejects_nonfinite_row() -> None:
    report = copy.deepcopy(run_report(samples=1025))
    rows = report["rows"]
    assert isinstance(rows, list)
    rows[0]["rel_error"] = float("nan")

    errors = verify_bundle_gauge_invariance_report(report)

    assert any("row 0 rel_error must be finite" in error for error in errors)


def test_bundle_gauge_invariance_report_rejects_inconsistent_row_error() -> None:
    report = copy.deepcopy(run_report(samples=1025))
    rows = report["rows"]
    assert isinstance(rows, list)
    rows[0]["abs_error"] = 1.0

    errors = verify_bundle_gauge_invariance_report(report)

    assert any("row 0 abs_error mismatch" in error for error in errors)


def test_bundle_gauge_invariance_report_rejects_stale_summary() -> None:
    report = copy.deepcopy(run_report(samples=1025))
    rows = report["rows"]
    assert isinstance(rows, list)
    rows[0]["bad_no_jacobian_rel_error"] = 0.99

    errors = verify_bundle_gauge_invariance_report(report)

    assert any("summary mean_bad_no_jacobian_rel_error mismatch" in error for error in errors)


def test_bundle_gauge_invariance_report_rejects_bad_order_derivative() -> None:
    report = copy.deepcopy(run_report(samples=1025))
    order = report["order"]
    assert isinstance(order, dict)
    order["log_min_derivative"] = -1.0

    errors = verify_bundle_gauge_invariance_report(report)

    assert any("log_min_derivative must be positive" in error for error in errors)


def test_saved_bundle_gauge_invariance_artifact_satisfies_contract() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(Path(summary_json).read_text(encoding="utf-8"))

    assert_bundle_gauge_invariance_report(report)
