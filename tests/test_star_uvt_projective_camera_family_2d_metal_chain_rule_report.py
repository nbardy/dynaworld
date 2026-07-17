from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from research_experiments.star_uvt_feature_tubes.projective_camera_family_2d_metal_chain_rule_report import (
    DEFAULT_OUT_DIR,
    assert_camera_family_2d_metal_chain_rule_report,
    run_report,
    summarize,
    verify_camera_family_2d_metal_chain_rule_report,
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
                    "objective": 4.0 + q_phase + q_height,
                    "slice_payload_bytes": 104,
                    "slice_grad_payload_bytes": 72,
                    "grad_coeff_abs_sum": 3.0,
                    "family_grad_abs_sum": 5.0,
                    "image_sum": 80.0,
                }
            )
    finite_differences = [
        {
            "trace": idx % 2,
            "coeff": idx,
            "basis": 0,
            "analytic_grad": 10.0 + idx,
            "finite_difference_grad": 10.0 + idx + 1.0e-4,
            "abs_error": 1.0e-4,
            "rel_error": 1.0e-5,
        }
        for idx in range(6)
    ]
    report: dict[str, object] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_camera_family_2d_metal_chain_rule",
        "base_domain": "Q2 x Omega x T shared backward from Omega x T Metal slices",
        "theory_contract": "Per-slice interval Metal VJPs accumulate through d coeff_slice / d family_coeff into one shared Q2 x Omega x T adjoint for pi_* Gamma^* traces. This is shared-family chain-rule accumulation over Metal slices, not native Q2 Metal evaluation.",
        "interval_metal_available": True,
        "interval_backward_metal_available": True,
        "metal_ran": True,
        "q_axis_count": 5,
        "q_pair_count": 25,
        "frames": 4,
        "image_size": 8,
        "trace_count": 2,
        "family_basis_count": 6,
        "family_payload_bytes": 464,
        "slice_payload_bytes": 104,
        "shared_family_gradient_payload_bytes": 432,
        "per_q_replay_gradient_payload_bytes": 1800,
        "shared_family_grad_abs_sum": 91.0,
        "finite_difference_eps": 0.01,
        "rows": rows,
        "finite_differences": finite_differences,
    }
    report["summary"] = summarize(report)
    return report


def test_camera_family_2d_metal_chain_rule_accepts_valid_payload() -> None:
    report = _valid_report()

    assert verify_camera_family_2d_metal_chain_rule_report(report) == []
    assert_camera_family_2d_metal_chain_rule_report(report)
    assert report["summary"]["shared_to_replay_gradient_payload_ratio"] < 0.30
    assert report["summary"]["max_finite_difference_rel_error"] < 1.0e-3


def test_camera_family_2d_metal_chain_rule_rejects_missing_contract() -> None:
    report = _valid_report()
    report["base_domain"] = "Q2 only"
    report["theory_contract"] = "too vague"

    errors = verify_camera_family_2d_metal_chain_rule_report(report)

    assert any("base_domain" in error for error in errors)
    assert any("pi_* Gamma^*" in error for error in errors)


def test_camera_family_2d_metal_chain_rule_rejects_gradient_payload_regression() -> None:
    report = _valid_report()
    report["shared_family_gradient_payload_bytes"] = report["per_q_replay_gradient_payload_bytes"]
    report["summary"] = summarize(report)

    errors = verify_camera_family_2d_metal_chain_rule_report(report)

    assert any("shared/replay gradient payload ratio" in error for error in errors)


def test_camera_family_2d_metal_chain_rule_rejects_bad_finite_difference() -> None:
    report = _valid_report()
    report["finite_differences"][0]["rel_error"] = 0.02
    report["summary"] = summarize(report)

    errors = verify_camera_family_2d_metal_chain_rule_report(report)

    assert any("rel_error too high" in error for error in errors)
    assert any("max finite-difference relative error" in error for error in errors)


def test_camera_family_2d_metal_chain_rule_rejects_stale_summary_after_row_change() -> None:
    report = copy.deepcopy(_valid_report())
    report["rows"][0]["grad_coeff_abs_sum"] = 0.0

    errors = verify_camera_family_2d_metal_chain_rule_report(report)

    assert any("grad_coeff_abs_sum" in error for error in errors)
    assert any("summary min_slice_grad_coeff_abs_sum mismatch" in error for error in errors)


def test_camera_family_2d_metal_chain_rule_runs_metal_when_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is required for the Metal chain-rule smoke")

    report = run_report(q_axis_count=5)

    assert_camera_family_2d_metal_chain_rule_report(report)
    assert report["summary"]["q_pair_count"] == 25
    assert report["summary"]["shared_family_grad_abs_sum"] > 0.0


def test_saved_camera_family_2d_metal_chain_rule_artifact_satisfies_contract() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(Path(summary_json).read_text(encoding="utf-8"))

    assert_camera_family_2d_metal_chain_rule_report(report)
