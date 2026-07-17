from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from research_experiments.star_uvt_feature_tubes.projective_camera_family_2d_native_interval_backward_report import (
    DEFAULT_OUT_DIR,
    assert_camera_family_2d_native_interval_backward_report,
    run_report,
    summarize,
    verify_camera_family_2d_native_interval_backward_report,
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
                    "native_q_basis_grad_abs_sum": 4.0,
                }
            )
    report: dict[str, object] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_camera_family_2d_native_interval_backward",
        "base_domain": "Q2 x Omega x T native family interval backward",
        "theory_contract": "The Metal interval VJP consumes Q2 family coefficients and q-basis values directly for pi_* Gamma^* trace rendering. This is native backward/VJP over family traces with compiled visibility held fixed.",
        "interval_backward_available": True,
        "family_interval_backward_available": True,
        "metal_ran": True,
        "q_axis_count": 5,
        "q_pair_count": 25,
        "frames_per_q": 4,
        "batched_frames": 100,
        "image_size": 8,
        "trace_count": 2,
        "family_basis_count": 6,
        "family_forward_payload_bytes": 1160,
        "native_family_gradient_payload_bytes": 1112,
        "native_family_coeff_gradient_payload_bytes": 432,
        "native_q_basis_gradient_payload_bytes": 600,
        "materialized_gradient_payload_bytes": 3800,
        "materialized_trace_payload_bytes": 2600,
        "native_family_interval_backward_max_family_grad_abs_error": 5.0e-5,
        "native_family_interval_backward_max_family_grad_rel_error": 3.0e-6,
        "native_family_interval_backward_max_q_basis_grad_abs_error": 7.0e-6,
        "native_family_interval_backward_max_q_basis_grad_rel_error": 2.0e-6,
        "native_family_interval_backward_max_opacity_grad_abs_error": 8.0e-5,
        "native_family_interval_backward_max_opacity_grad_rel_error": 1.0e-6,
        "native_family_interval_backward_max_color_grad_abs_error": 9.0e-5,
        "native_family_interval_backward_max_color_grad_rel_error": 2.0e-6,
        "native_family_interval_backward_max_opacity_time_grad_abs_error": 5.0e-5,
        "native_family_interval_backward_max_opacity_time_grad_rel_error": 1.0e-6,
        "native_family_interval_backward_max_spatial_precision_grad_abs_error": 2.0e-4,
        "native_family_interval_backward_max_spatial_precision_grad_rel_error": 2.0e-6,
        "native_family_grad_abs_sum": 91.0,
        "native_q_basis_grad_abs_sum": 100.0,
        "native_opacity_grad_abs_sum": 206.0,
        "native_color_grad_abs_sum": 344.0,
        "rows": rows,
    }
    report["summary"] = summarize(report)
    return report


def test_camera_family_2d_native_interval_backward_accepts_valid_payload() -> None:
    report = _valid_report()

    assert verify_camera_family_2d_native_interval_backward_report(report) == []
    assert_camera_family_2d_native_interval_backward_report(report)
    assert report["summary"]["native_family_gradient_to_materialized_gradient_payload_ratio"] < 0.35


def test_camera_family_2d_native_interval_backward_rejects_missing_contract() -> None:
    report = _valid_report()
    report["base_domain"] = "Q2 only"
    report["theory_contract"] = "too vague"

    errors = verify_camera_family_2d_native_interval_backward_report(report)

    assert any("base_domain" in error for error in errors)
    assert any("native-backward" in error for error in errors)


def test_camera_family_2d_native_interval_backward_rejects_payload_regression() -> None:
    report = _valid_report()
    report["native_family_gradient_payload_bytes"] = report["materialized_gradient_payload_bytes"]
    report["summary"] = summarize(report)

    errors = verify_camera_family_2d_native_interval_backward_report(report)

    assert any("native family/materialized gradient payload ratio" in error for error in errors)


def test_camera_family_2d_native_interval_backward_rejects_gradient_mismatch() -> None:
    report = _valid_report()
    report["native_family_interval_backward_max_family_grad_rel_error"] = 1.0e-3
    report["summary"] = summarize(report)

    errors = verify_camera_family_2d_native_interval_backward_report(report)

    assert any("native_family_interval_backward_max_family_grad_rel_error" in error for error in errors)


def test_camera_family_2d_native_interval_backward_rejects_stale_summary_after_row_change() -> None:
    report = copy.deepcopy(_valid_report())
    report["rows"][0]["native_q_basis_grad_abs_sum"] = 0.0

    errors = verify_camera_family_2d_native_interval_backward_report(report)

    assert any("row native_q_basis_grad_abs_sum" in error for error in errors)


def test_camera_family_2d_native_interval_backward_runs_metal_when_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is required for the native family interval backward smoke")

    report = run_report(q_axis_count=5)

    assert_camera_family_2d_native_interval_backward_report(report)
    assert report["summary"]["q_pair_count"] == 25
    assert report["summary"]["native_family_gradient_to_materialized_gradient_payload_ratio"] < 0.35
    assert report["summary"]["native_family_grad_abs_sum"] > 0.0
    assert report["summary"]["native_q_basis_grad_abs_sum"] > 0.0


def test_saved_camera_family_2d_native_interval_backward_artifact_satisfies_contract() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(Path(summary_json).read_text(encoding="utf-8"))

    assert_camera_family_2d_native_interval_backward_report(report)
