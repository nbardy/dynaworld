from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research_experiments.star_uvt_feature_tubes.projective_camera_family_2d_gauge_report import (
    DEFAULT_OUT_DIR,
    assert_camera_family_2d_gauge_report,
    run_report,
    verify_camera_family_2d_gauge_report,
)


def test_camera_family_2d_gauge_report_accepts_q2_family() -> None:
    report = run_report(samples=1025)

    assert verify_camera_family_2d_gauge_report(report) == []
    assert_camera_family_2d_gauge_report(report)
    assert report["base_domain"] == "Q2 x Omega x T"
    assert report["summary"]["q_phase_min"] < 0.0 < report["summary"]["q_phase_max"]
    assert report["summary"]["q_height_min"] < 0.0 < report["summary"]["q_height_max"]
    assert report["summary"]["max_value_rel_error"] < 2.0e-6
    assert report["summary"]["max_primitive_gradient_rel_error"] < 2.0e-6
    assert report["summary"]["q_phase_gradient_rel_error"] < 2.0e-6
    assert report["summary"]["q_height_gradient_rel_error"] < 2.0e-6


def test_camera_family_2d_gauge_report_rejects_missing_contract() -> None:
    report = run_report(samples=1025)
    report["base_domain"] = "Q x Omega x T"
    report["theory_contract"] = "too vague"

    errors = verify_camera_family_2d_gauge_report(report)

    assert any("base_domain" in error for error in errors)
    assert any("Gamma(q_phase,q_height)" in error for error in errors)


def test_camera_family_2d_gauge_report_rejects_value_regression() -> None:
    report = run_report(samples=1025)
    report["summary"]["max_value_rel_error"] = 1.0e-3

    errors = verify_camera_family_2d_gauge_report(report)

    assert any("value gauge error" in error for error in errors)


def test_camera_family_2d_gauge_report_rejects_primitive_gradient_regression() -> None:
    report = run_report(samples=1025)
    report["summary"]["max_primitive_gradient_rel_error"] = 1.0e-3

    errors = verify_camera_family_2d_gauge_report(report)

    assert any("primitive gradient gauge error" in error for error in errors)


def test_camera_family_2d_gauge_report_rejects_camera_param_gradient_regression() -> None:
    report = run_report(samples=1025)
    report["summary"]["q_height_gradient_rel_error"] = 1.0e-3

    errors = verify_camera_family_2d_gauge_report(report)

    assert any("q_height gradient gauge error" in error for error in errors)


def test_camera_family_2d_gauge_report_rejects_bad_finite_difference() -> None:
    report = run_report(samples=1025)
    report["camera_gradients"][0]["finite_difference"] = report["camera_gradients"][0]["depth_grad"] + 1.0

    errors = verify_camera_family_2d_gauge_report(report)

    assert any("camera gradient row 0 finite_difference_rel_error mismatch" in error for error in errors)


def test_camera_family_2d_gauge_report_rejects_missing_jacobian_control_loss() -> None:
    report = run_report(samples=1025)
    report["summary"]["q_phase_bad_no_jacobian_rel_error"] = 0.0

    errors = verify_camera_family_2d_gauge_report(report)

    assert any("q_phase missing-Jacobian control" in error for error in errors)


def test_camera_family_2d_gauge_report_rejects_stale_summary_after_row_change() -> None:
    report = copy.deepcopy(run_report(samples=1025))
    report["rows"][0]["rel_error"] = 0.99

    errors = verify_camera_family_2d_gauge_report(report)

    assert any("summary max_value_rel_error mismatch" in error for error in errors)


def test_saved_camera_family_2d_gauge_artifact_satisfies_contract() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(Path(summary_json).read_text(encoding="utf-8"))

    assert_camera_family_2d_gauge_report(report)
