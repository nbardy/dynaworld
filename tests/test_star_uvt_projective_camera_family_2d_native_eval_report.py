from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from research_experiments.star_uvt_feature_tubes.projective_camera_family_2d_native_eval_report import (
    DEFAULT_OUT_DIR,
    assert_camera_family_2d_native_eval_report,
    run_report,
    summarize,
    verify_camera_family_2d_native_eval_report,
)


def _valid_report() -> dict[str, object]:
    rows = []
    for q_phase in (-0.3, -0.15, 0.0, 0.15, 0.3):
        for q_height in (-0.24, -0.12, 0.0, 0.12, 0.24):
            rows.append(
                {
                    "q_phase": q_phase,
                    "q_height": q_height,
                    "basis_abs_sum": 1.0 + abs(q_phase) + abs(q_height),
                    "output_abs_sum": 50.0 + q_phase + q_height,
                }
            )
    report: dict[str, object] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_camera_family_2d_native_eval",
        "base_domain": "Q2 x Omega x T native family trace eval/VJP",
        "theory_contract": "The Metal shader consumes Q2 family coefficients and q-basis values directly for pi_* Gamma^* trace evaluation and VJP. This is native family trace evaluation and shared-family VJP, not the full visibility/compositing renderer.",
        "family_metal_available": True,
        "family_backward_metal_available": True,
        "metal_ran": True,
        "q_axis_count": 5,
        "q_pair_count": 25,
        "trace_count": 2,
        "family_basis_count": 6,
        "time_sample_count": 4,
        "family_coeff_payload_bytes": 432,
        "q_basis_payload_bytes": 600,
        "materialized_coeff_payload_bytes": 1800,
        "native_eval_max_abs_error": 0.0,
        "native_eval_max_rel_error": 0.0,
        "native_grad_family_max_abs_error": 1.0e-6,
        "native_grad_family_max_rel_error": 1.0e-6,
        "native_grad_q_basis_max_abs_error": 1.0e-6,
        "native_grad_q_basis_max_rel_error": 1.0e-6,
        "metal_output_abs_sum": 100.0,
        "metal_grad_family_abs_sum": 80.0,
        "metal_grad_q_basis_abs_sum": 70.0,
        "rows": rows,
    }
    report["summary"] = summarize(report)
    return report


def test_camera_family_2d_native_eval_accepts_valid_payload() -> None:
    report = _valid_report()

    assert verify_camera_family_2d_native_eval_report(report) == []
    assert_camera_family_2d_native_eval_report(report)
    assert report["summary"]["family_coeff_to_materialized_coeff_payload_ratio"] == pytest.approx(0.24)
    assert report["summary"]["family_plus_q_basis_to_materialized_coeff_payload_ratio"] < 0.65


def test_camera_family_2d_native_eval_rejects_missing_contract() -> None:
    report = _valid_report()
    report["base_domain"] = "Q2 only"
    report["theory_contract"] = "too vague"

    errors = verify_camera_family_2d_native_eval_report(report)

    assert any("base_domain" in error for error in errors)
    assert any("pi_* Gamma^*" in error for error in errors)


def test_camera_family_2d_native_eval_rejects_payload_regression() -> None:
    report = _valid_report()
    report["family_coeff_payload_bytes"] = report["materialized_coeff_payload_bytes"]
    report["summary"] = summarize(report)

    errors = verify_camera_family_2d_native_eval_report(report)

    assert any("family/materialized coefficient payload ratio" in error for error in errors)


def test_camera_family_2d_native_eval_rejects_value_mismatch() -> None:
    report = _valid_report()
    report["native_eval_max_abs_error"] = 1.0e-3
    report["summary"] = summarize(report)

    errors = verify_camera_family_2d_native_eval_report(report)

    assert any("native_eval_max_abs_error" in error for error in errors)


def test_camera_family_2d_native_eval_rejects_gradient_mismatch() -> None:
    report = _valid_report()
    report["native_grad_q_basis_max_rel_error"] = 1.0e-3
    report["summary"] = summarize(report)

    errors = verify_camera_family_2d_native_eval_report(report)

    assert any("native_grad_q_basis_max_rel_error" in error for error in errors)


def test_camera_family_2d_native_eval_rejects_stale_summary_after_row_change() -> None:
    report = copy.deepcopy(_valid_report())
    report["rows"][0]["output_abs_sum"] = 0.0

    errors = verify_camera_family_2d_native_eval_report(report)

    assert any("output_abs_sum" in error for error in errors)


def test_camera_family_2d_native_eval_runs_metal_when_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is required for the native family eval smoke")

    report = run_report(q_axis_count=5)

    assert_camera_family_2d_native_eval_report(report)
    assert report["summary"]["q_pair_count"] == 25
    assert report["summary"]["family_coeff_to_materialized_coeff_payload_ratio"] < 0.30
    assert report["summary"]["metal_grad_family_abs_sum"] > 0.0


def test_saved_camera_family_2d_native_eval_artifact_satisfies_contract() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(Path(summary_json).read_text(encoding="utf-8"))

    assert_camera_family_2d_native_eval_report(report)
