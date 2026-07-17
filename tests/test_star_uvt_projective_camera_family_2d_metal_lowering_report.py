from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from research_experiments.star_uvt_feature_tubes.projective_camera_family_2d_metal_lowering_report import (
    DEFAULT_OUT_DIR,
    _family_coeff_table,
    assert_camera_family_2d_metal_lowering_report,
    lower_q2_family_coeffs,
    run_report,
    verify_camera_family_2d_metal_lowering_report,
)


def _valid_report() -> dict[str, object]:
    rows = []
    for q_phase in (-0.3, -0.15, 0.0, 0.15, 0.3):
        for q_height in (-0.24, -0.12, 0.0, 0.12, 0.24):
            rows.append(
                {
                    "q_phase": q_phase,
                    "q_height": q_height,
                    "coeff_checksum": 20.0 + q_phase + q_height,
                    "slice_payload_bytes": 104,
                    "ordered_primitive_ids": [0, 1],
                    "image_sum": 78.0 + q_phase,
                    "image_max": 0.65,
                    "grad_coeff_abs_sum": 2.0,
                    "grad_opacity_abs_sum": 5.0,
                    "grad_color_abs_sum": 8.0,
                }
            )
    report: dict[str, object] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_camera_family_2d_metal_lowering",
        "base_domain": "Q2 x Omega x T -> Omega x T Metal slice",
        "theory_contract": "A Q2 camera-family trace chart is lowered to an Omega x T slice of pi_* Gamma^* world primitives for the existing interval Metal forward/backward path. This is a slice-lowering smoke, not native Q2 Metal evaluation.",
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
        "rows": rows,
    }
    from research_experiments.star_uvt_feature_tubes.projective_camera_family_2d_metal_lowering_report import (
        summarize,
    )

    report["summary"] = summarize(report)
    return report


def test_q2_lowering_changes_coefficients_but_preserves_shape() -> None:
    family_coeffs = _family_coeff_table()

    center = lower_q2_family_coeffs(family_coeffs, q_phase=0.0, q_height=0.0)
    corner = lower_q2_family_coeffs(family_coeffs, q_phase=0.3, q_height=-0.24)

    assert center.shape == (2, 9)
    assert corner.shape == (2, 9)
    assert not torch.allclose(center, corner)
    assert torch.all(corner[:, 0] > 2.0)
    assert torch.all(corner[:, 3] > 2.0)


def test_camera_family_2d_metal_lowering_accepts_valid_payload() -> None:
    report = _valid_report()

    assert verify_camera_family_2d_metal_lowering_report(report) == []
    assert_camera_family_2d_metal_lowering_report(report)
    assert report["summary"]["family_to_replay_payload_ratio"] < 0.35
    assert report["summary"]["peak_slice_to_replay_payload_ratio"] < 0.10


def test_camera_family_2d_metal_lowering_rejects_missing_contract() -> None:
    report = _valid_report()
    report["theory_contract"] = "too vague"
    report["base_domain"] = "Q2 only"

    errors = verify_camera_family_2d_metal_lowering_report(report)

    assert any("base_domain" in error for error in errors)
    assert any("pi_* Gamma^*" in error for error in errors)


def test_camera_family_2d_metal_lowering_rejects_no_metal() -> None:
    report = _valid_report()
    report["metal_ran"] = False

    errors = verify_camera_family_2d_metal_lowering_report(report)

    assert any("metal_ran" in error for error in errors)


def test_camera_family_2d_metal_lowering_rejects_payload_regression() -> None:
    report = _valid_report()
    report["family_payload_bytes"] = report["slice_payload_bytes"] * report["q_pair_count"]
    from research_experiments.star_uvt_feature_tubes.projective_camera_family_2d_metal_lowering_report import (
        summarize,
    )

    report["summary"] = summarize(report)

    errors = verify_camera_family_2d_metal_lowering_report(report)

    assert any("family-to-replay payload ratio" in error for error in errors)


def test_camera_family_2d_metal_lowering_rejects_stale_summary_after_row_change() -> None:
    report = copy.deepcopy(_valid_report())
    report["rows"][0]["grad_color_abs_sum"] = 0.0

    errors = verify_camera_family_2d_metal_lowering_report(report)

    assert any("grad_color_abs_sum" in error for error in errors)
    assert any("summary min_grad_color_abs_sum mismatch" in error for error in errors)


def test_camera_family_2d_metal_lowering_runs_metal_when_available() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is required for the Metal lowering smoke")

    report = run_report(q_axis_count=5)

    assert_camera_family_2d_metal_lowering_report(report)
    assert report["summary"]["q_pair_count"] == 25
    assert report["summary"]["min_grad_coeff_abs_sum"] > 0.0


def test_saved_camera_family_2d_metal_lowering_artifact_satisfies_contract() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(Path(summary_json).read_text(encoding="utf-8"))

    assert_camera_family_2d_metal_lowering_report(report)
