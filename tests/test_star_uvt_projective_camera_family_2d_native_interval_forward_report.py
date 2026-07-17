from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from research_experiments.star_uvt_feature_tubes.projective_camera_family_2d_native_interval_forward_report import (
    DEFAULT_OUT_DIR,
    assert_camera_family_2d_native_interval_forward_report,
    run_report,
    summarize,
    verify_camera_family_2d_native_interval_forward_report,
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
                    "native_image_abs_sum": 40.0,
                }
            )
    report: dict[str, object] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_camera_family_2d_native_interval_forward",
        "base_domain": "Q2 x Omega x T native family interval forward",
        "theory_contract": "The Metal interval compositor consumes Q2 family coefficients and q-basis values directly for pi_* Gamma^* trace rendering. This is native forward rendering/compositing/visibility over family traces, not the full backward renderer VJP.",
        "interval_metal_available": True,
        "family_interval_metal_available": True,
        "metal_ran": True,
        "q_axis_count": 5,
        "q_pair_count": 25,
        "frames_per_q": 4,
        "batched_frames": 100,
        "image_size": 8,
        "trace_count": 2,
        "family_basis_count": 6,
        "family_coeff_payload_bytes": 432,
        "q_basis_payload_bytes": 600,
        "family_static_payload_bytes": 464,
        "family_forward_payload_bytes": 1160,
        "materialized_trace_payload_bytes": 2600,
        "native_family_forward_max_abs_error": 0.0,
        "native_family_forward_max_rel_error": 0.0,
        "materialized_image_abs_sum": 1000.0,
        "native_family_image_abs_sum": 1000.0,
        "rows": rows,
    }
    report["summary"] = summarize(report)
    return report


def test_camera_family_2d_native_interval_forward_accepts_valid_payload() -> None:
    report = _valid_report()

    assert verify_camera_family_2d_native_interval_forward_report(report) == []
    assert_camera_family_2d_native_interval_forward_report(report)
    assert report["summary"]["family_forward_to_materialized_trace_payload_ratio"] < 0.50


def test_camera_family_2d_native_interval_forward_rejects_missing_contract() -> None:
    report = _valid_report()
    report["base_domain"] = "Q2 only"
    report["theory_contract"] = "too vague"

    errors = verify_camera_family_2d_native_interval_forward_report(report)

    assert any("base_domain" in error for error in errors)
    assert any("native-forward" in error for error in errors)


def test_camera_family_2d_native_interval_forward_rejects_payload_regression() -> None:
    report = _valid_report()
    report["family_forward_payload_bytes"] = report["materialized_trace_payload_bytes"]
    report["summary"] = summarize(report)

    errors = verify_camera_family_2d_native_interval_forward_report(report)

    assert any("family forward/materialized trace payload ratio" in error for error in errors)


def test_camera_family_2d_native_interval_forward_rejects_image_mismatch() -> None:
    report = _valid_report()
    report["native_family_forward_max_abs_error"] = 1.0e-3
    report["summary"] = summarize(report)

    errors = verify_camera_family_2d_native_interval_forward_report(report)

    assert any("native_family_forward_max_abs_error" in error for error in errors)


def test_camera_family_2d_native_interval_forward_rejects_stale_summary_after_row_change() -> None:
    report = copy.deepcopy(_valid_report())
    report["rows"][0]["native_image_abs_sum"] = 0.0

    errors = verify_camera_family_2d_native_interval_forward_report(report)

    assert any("row native_image_abs_sum" in error for error in errors)


def test_camera_family_2d_native_interval_forward_runs_metal_when_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is required for the native family interval forward smoke")

    report = run_report(q_axis_count=5)

    assert_camera_family_2d_native_interval_forward_report(report)
    assert report["summary"]["q_pair_count"] == 25
    assert report["summary"]["family_forward_to_materialized_trace_payload_ratio"] < 0.50
    assert report["summary"]["native_family_image_abs_sum"] > 0.0


def test_saved_camera_family_2d_native_interval_forward_artifact_satisfies_contract() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(Path(summary_json).read_text(encoding="utf-8"))

    assert_camera_family_2d_native_interval_forward_report(report)
