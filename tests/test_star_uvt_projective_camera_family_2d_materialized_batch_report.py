from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from research_experiments.star_uvt_feature_tubes.projective_camera_family_2d_materialized_batch_report import (
    DEFAULT_OUT_DIR,
    assert_camera_family_2d_materialized_batch_report,
    run_report,
    summarize,
    verify_camera_family_2d_materialized_batch_report,
)


def _valid_report() -> dict[str, object]:
    rows = []
    for q_phase in (-0.3, -0.15, 0.0, 0.15, 0.3):
        for q_height in (-0.24, -0.12, 0.0, 0.12, 0.24):
            rows.append(
                {
                    "q_phase": q_phase,
                    "q_height": q_height,
                    "image_sum": 78.0 + q_phase,
                    "grad_coeff_abs_sum": 2.0 + abs(q_height),
                }
            )
    report: dict[str, object] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_camera_family_2d_materialized_batch",
        "base_domain": "Q2 x Omega x T materialized single-launch Metal batch",
        "theory_contract": "A Q2 camera-family trace grid is materialized into one Omega x T interval Metal atlas to test launch reuse for pi_* Gamma^* traces while leaving native family-coefficient evaluation open. This is a single-launch materialized baseline, not native Q2/Qn Metal evaluation.",
        "interval_metal_available": True,
        "interval_backward_metal_available": True,
        "metal_ran": True,
        "q_axis_count": 5,
        "q_pair_count": 25,
        "frames_per_q": 4,
        "batched_frames": 100,
        "image_size": 8,
        "trace_count": 2,
        "family_basis_count": 6,
        "family_payload_bytes": 464,
        "slice_trace_payload_bytes": 104,
        "materialized_trace_payload_bytes": 2600,
        "per_q_replay_trace_payload_bytes": 2600,
        "shared_family_gradient_payload_bytes": 432,
        "materialized_gradient_payload_bytes": 1800,
        "per_q_replay_gradient_payload_bytes": 1800,
        "max_batched_vs_slice_image_abs_error": 0.0,
        "max_batched_vs_slice_image_rel_error": 0.0,
        "max_batched_vs_slice_shared_grad_abs_error": 0.0,
        "max_batched_vs_slice_shared_grad_rel_error": 0.0,
        "batched_image_sum": 1900.0,
        "batched_grad_coeff_abs_sum": 91.0,
        "batched_shared_family_grad_abs_sum": 95.0,
        "rows": rows,
    }
    report["summary"] = summarize(report)
    return report


def test_camera_family_2d_materialized_batch_accepts_valid_payload() -> None:
    report = _valid_report()

    assert verify_camera_family_2d_materialized_batch_report(report) == []
    assert_camera_family_2d_materialized_batch_report(report)
    assert report["summary"]["forward_launch_ratio"] == pytest.approx(0.04)
    assert report["summary"]["materialized_to_replay_trace_payload_ratio"] == pytest.approx(1.0)
    assert report["summary"]["family_to_materialized_trace_payload_ratio"] < 0.35


def test_camera_family_2d_materialized_batch_rejects_missing_contract() -> None:
    report = _valid_report()
    report["base_domain"] = "Q2 only"
    report["theory_contract"] = "too vague"

    errors = verify_camera_family_2d_materialized_batch_report(report)

    assert any("base_domain" in error for error in errors)
    assert any("pi_* Gamma^*" in error for error in errors)


def test_camera_family_2d_materialized_batch_rejects_fake_native_compression() -> None:
    report = _valid_report()
    report["materialized_trace_payload_bytes"] = report["family_payload_bytes"]
    report["summary"] = summarize(report)

    errors = verify_camera_family_2d_materialized_batch_report(report)

    assert any("materialized/replay trace payload ratio" in error for error in errors)


def test_camera_family_2d_materialized_batch_rejects_launch_regression() -> None:
    report = _valid_report()
    report["q_axis_count"] = 3
    report["q_pair_count"] = 9
    report["batched_frames"] = 36
    report["rows"] = report["rows"][:9]
    report["summary"] = summarize(report)

    errors = verify_camera_family_2d_materialized_batch_report(report)

    assert any("q_axis_count" in error for error in errors)
    assert any("forward launch ratio" in error for error in errors)


def test_camera_family_2d_materialized_batch_rejects_mismatch_regression() -> None:
    report = _valid_report()
    report["max_batched_vs_slice_image_abs_error"] = 1.0e-3
    report["summary"] = summarize(report)

    errors = verify_camera_family_2d_materialized_batch_report(report)

    assert any("max_batched_vs_slice_image_abs_error" in error for error in errors)


def test_camera_family_2d_materialized_batch_rejects_stale_summary_after_row_change() -> None:
    report = copy.deepcopy(_valid_report())
    report["rows"][0]["grad_coeff_abs_sum"] = 0.0

    errors = verify_camera_family_2d_materialized_batch_report(report)

    assert any("grad_coeff_abs_sum" in error for error in errors)


def test_camera_family_2d_materialized_batch_runs_metal_when_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is required for the materialized batch smoke")

    report = run_report(q_axis_count=5)

    assert_camera_family_2d_materialized_batch_report(report)
    assert report["summary"]["q_pair_count"] == 25
    assert report["summary"]["forward_launch_ratio"] == pytest.approx(0.04)
    assert report["summary"]["materialized_to_replay_trace_payload_ratio"] == pytest.approx(1.0)


def test_saved_camera_family_2d_materialized_batch_artifact_satisfies_contract() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(Path(summary_json).read_text(encoding="utf-8"))

    assert_camera_family_2d_materialized_batch_report(report)
