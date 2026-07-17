from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research_experiments.star_uvt_feature_tubes.projective_exposure_rolling_backward_report import (
    DEFAULT_OUT_DIR,
    assert_exposure_rolling_backward_report,
    run_report,
    verify_exposure_rolling_backward_report,
)


def test_exposure_rolling_backward_report_accepts_shared_adjoint_reference() -> None:
    report = run_report(run_metal=False)

    assert verify_exposure_rolling_backward_report(report) == []
    assert_exposure_rolling_backward_report(report)
    assert report["finite_exposure_backward"]["reference_grad_norms"]["coeffs"] > 0.0
    assert report["rolling_shutter_backward"]["unique_to_row_sample_ratio"] < 1.0


def test_exposure_rolling_backward_report_accepts_metal_if_available() -> None:
    report = run_report(run_metal=True)

    assert_exposure_rolling_backward_report(report)
    if report["summary"]["metal_backward_case_count"]:
        assert report["summary"]["max_metal_grad_abs_error"] <= 1.0e-3
        assert report["summary"]["max_metal_grad_rel_error"] <= 5.0e-3


def test_exposure_rolling_backward_report_rejects_zero_finite_adjoint() -> None:
    report = copy.deepcopy(run_report(run_metal=False))
    report["finite_exposure_backward"]["sample_adjoint_abs_sum"] = 0.0

    errors = verify_exposure_rolling_backward_report(report)

    assert any("finite sample adjoint" in error for error in errors)


def test_exposure_rolling_backward_report_rejects_lost_rolling_reuse() -> None:
    report = copy.deepcopy(run_report(run_metal=False))
    report["rolling_shutter_backward"]["lowered_unique_sample_count"] = report["rolling_shutter_backward"][
        "total_row_sample_count"
    ]
    report["rolling_shutter_backward"]["unique_to_row_sample_ratio"] = 1.0
    report["summary"]["rolling_unique_to_row_sample_ratio"] = 1.0

    errors = verify_exposure_rolling_backward_report(report)

    assert any("rolling backward must deduplicate" in error for error in errors)


def test_exposure_rolling_backward_report_rejects_bad_row_weight_sum() -> None:
    report = copy.deepcopy(run_report(run_metal=False))
    report["rolling_shutter_backward"]["row_weight_sums"][0] = 0.25

    errors = verify_exposure_rolling_backward_report(report)

    assert any("rolling row 0 weight sum" in error for error in errors)


def test_exposure_rolling_backward_report_rejects_bad_metal_compare() -> None:
    report = copy.deepcopy(run_report(run_metal=False))
    report["finite_exposure_backward"]["metal_compare"] = {
        "max_abs_error": 2.0e-3,
        "max_rel_error": 1.0e-2,
        "coeffs": {"max_abs_error": 2.0e-3, "max_rel_error": 1.0e-2, "reference_grad_norm": 1.0},
        "opacity": {"max_abs_error": 0.0, "max_rel_error": 0.0, "reference_grad_norm": 1.0},
        "color": {"max_abs_error": 0.0, "max_rel_error": 0.0, "reference_grad_norm": 1.0},
    }
    report["summary"]["finite_has_metal_backward"] = True
    report["summary"]["metal_backward_case_count"] = 1
    report["summary"]["max_metal_grad_abs_error"] = 2.0e-3
    report["summary"]["max_metal_grad_rel_error"] = 1.0e-2

    errors = verify_exposure_rolling_backward_report(report)

    assert any("finite max_abs_error" in error for error in errors)
    assert any("finite max_rel_error" in error for error in errors)


def test_exposure_rolling_backward_report_rejects_stale_summary() -> None:
    report = copy.deepcopy(run_report(run_metal=False))
    report["summary"]["rolling_unique_to_row_sample_ratio"] = 0.25

    errors = verify_exposure_rolling_backward_report(report)

    assert any("summary rolling_unique_to_row_sample_ratio" in error for error in errors)


def test_exposure_rolling_backward_report_rejects_stale_rolling_ratio() -> None:
    report = copy.deepcopy(run_report(run_metal=False))
    report["rolling_shutter_backward"]["unique_to_row_sample_ratio"] = 0.25
    report["summary"]["rolling_unique_to_row_sample_ratio"] = 0.25

    errors = verify_exposure_rolling_backward_report(report)

    assert any("rolling unique_to_row_sample_ratio" in error for error in errors)


def test_exposure_rolling_backward_report_rejects_stale_compare_aggregate() -> None:
    report = copy.deepcopy(run_report(run_metal=False))
    report["finite_exposure_backward"]["metal_compare"] = {
        "max_abs_error": 1.0e-6,
        "max_rel_error": 1.0e-6,
        "coeffs": {"max_abs_error": 5.0e-4, "max_rel_error": 1.0e-4, "reference_grad_norm": 1.0},
        "opacity": {"max_abs_error": 0.0, "max_rel_error": 0.0, "reference_grad_norm": 1.0},
        "color": {"max_abs_error": 0.0, "max_rel_error": 0.0, "reference_grad_norm": 1.0},
    }
    report["summary"] = {
        "finite_has_metal_backward": True,
        "rolling_has_metal_backward": False,
        "rolling_unique_to_row_sample_ratio": report["rolling_shutter_backward"]["unique_to_row_sample_ratio"],
        "max_metal_grad_abs_error": 1.0e-6,
        "max_metal_grad_rel_error": 1.0e-6,
        "metal_backward_case_count": 1,
    }

    errors = verify_exposure_rolling_backward_report(report)

    assert any("finite max_abs_error" in error for error in errors)


def test_exposure_rolling_backward_report_rejects_nonboolean_device_field() -> None:
    report = copy.deepcopy(run_report(run_metal=False))
    report["device"]["requested_metal"] = "no"

    errors = verify_exposure_rolling_backward_report(report)

    assert any("device requested_metal must be boolean" in error for error in errors)


def test_saved_exposure_rolling_backward_artifact_satisfies_contract() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(Path(summary_json).read_text(encoding="utf-8"))

    assert_exposure_rolling_backward_report(report)
