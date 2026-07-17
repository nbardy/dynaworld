from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research_experiments.star_uvt_feature_tubes.projective_shared_work_goal_audit import (
    DEFAULT_EXPOSURE_BACKWARD_REPORT,
    DEFAULT_EXPOSURE_MIXED_FALLBACK_BACKWARD_REPORT,
    DEFAULT_EXPOSURE_QUADRATURE_REPORT,
    DEFAULT_OUT_DIR,
    DEFAULT_ORBIT_REPORT,
    DEFAULT_TRAINED_REPORTS,
    assert_shared_work_goal_current_acceptance,
    assert_shared_work_goal_audit,
    run_report,
    summarize,
    verify_shared_work_goal_current_acceptance,
    verify_shared_work_goal_audit,
)


def _valid_report() -> dict[str, object]:
    orbit = {
        "path": "orbit.json",
        "underlying_errors": [],
        "frame_counts": [4, 8, 16, 32],
        "fixed_payload_growth": 1.0,
        "per_frame_payload_growth": 8.0,
        "final_payload_ratio": 0.125,
        "final_trace_ratio": 0.125,
        "final_segment_ratio": 0.125,
        "final_forward_ms_ratio": 0.15,
        "final_backward_ms_ratio": 0.27,
        "final_cpu_compile_ms_ratio": 0.12,
    }
    trained = [
        {
            "path": f"trained_{idx}.json",
            "underlying_errors": [],
            "frame_counts": [4, 8, 16],
            "size": size,
            "tube_count": tubes,
            "tile_capacity": 128,
            "shared_interval_entry_growth": shared_growth,
            "per_frame_interval_entry_growth": per_frame_growth,
            "final_interval_entry_ratio": entry_ratio,
            "final_trace_count_ratio": 0.1,
            "final_backward_ms_ratio": backward_ratio,
            "final_forward_ms_ratio": 0.3,
        }
        for idx, (size, tubes, shared_growth, per_frame_growth, entry_ratio, backward_ratio) in enumerate(
            [
                (32, 64, 1.46, 9.85, 0.148, 0.164),
                (64, 128, 1.43, 10.04, 0.143, 0.171),
                (96, 256, 1.38, 10.04, 0.138, 0.122),
            ]
        )
    ]
    exposure_quadrature = {
        "path": "exposure_quadrature.json",
        "underlying_errors": [],
        "finite_reference_lowered_max_abs_error": 0.0,
        "rolling_rowwise_batched_max_abs_error": 0.0,
        "rolling_unique_to_row_sample_ratio": 0.875,
        "finite_fallback_fraction": 0.5,
        "rolling_fallback_fraction": 0.5,
        "max_metal_abs_error": 5.96e-8,
        "metal_case_count": 4,
    }
    exposure_backward = {
        "path": "exposure_backward.json",
        "underlying_errors": [],
        "finite_has_metal_backward": True,
        "rolling_has_metal_backward": True,
        "rolling_unique_to_row_sample_ratio": 0.875,
        "max_metal_grad_abs_error": 1.43e-6,
        "max_metal_grad_rel_error": 6.38e-7,
        "metal_backward_case_count": 2,
    }
    exposure_mixed_fallback_backward = {
        "path": "exposure_mixed_fallback_backward.json",
        "underlying_errors": [],
        "finite_has_mixed_backward": True,
        "rolling_has_mixed_backward": True,
        "finite_fallback_fraction": 0.5,
        "rolling_fallback_fraction": 0.5,
        "rolling_unique_to_row_sample_ratio": 11.0 / 12.0,
        "max_mixed_output_abs_error": 5.96e-8,
        "max_mixed_grad_abs_error": 2.15e-6,
        "max_mixed_grad_rel_error": 7.41e-7,
        "mixed_backward_case_count": 2,
    }
    return {
        "status": "ok",
        "benchmark": "star_uvt_projective_shared_work_goal_audit",
        "theory_contract": (
            "Known camera-path traces should share projection/support/binning/payload "
            "and backward work so non-pixel world-side cost grows sublinearly with frame count, "
            "including finite-exposure and rolling-shutter evaluation/backward reuse with "
            "differentiable visibility fallback."
        ),
        "orbit": orbit,
        "trained": trained,
        "exposure_quadrature": exposure_quadrature,
        "exposure_backward": exposure_backward,
        "exposure_mixed_fallback_backward": exposure_mixed_fallback_backward,
        "summary": summarize(
            orbit,
            trained,
            exposure_quadrature,
            exposure_backward,
            exposure_mixed_fallback_backward,
        ),
    }


def test_shared_work_goal_audit_accepts_valid_payload() -> None:
    report = _valid_report()

    assert verify_shared_work_goal_audit(report) == []
    assert_shared_work_goal_audit(report)


def test_shared_work_current_acceptance_accepts_matching_payloads() -> None:
    report = _valid_report()

    assert verify_shared_work_goal_current_acceptance(report, current_report=copy.deepcopy(report)) == []
    assert_shared_work_goal_current_acceptance(report, current_report=copy.deepcopy(report))


def test_shared_work_current_acceptance_rejects_stale_but_valid_payload() -> None:
    current = _valid_report()
    saved = copy.deepcopy(current)
    saved["trained"][0]["final_backward_ms_ratio"] = 0.09
    saved["summary"] = summarize(
        saved["orbit"],
        saved["trained"],
        saved["exposure_quadrature"],
        saved["exposure_backward"],
        saved["exposure_mixed_fallback_backward"],
    )

    errors = verify_shared_work_goal_current_acceptance(saved, current_report=current)

    assert any("saved report differs from current inputs at trained[0].final_backward_ms_ratio" in error for error in errors)


def test_shared_work_goal_audit_rejects_orbit_payload_growth() -> None:
    report = copy.deepcopy(_valid_report())
    report["summary"]["orbit_fixed_payload_growth"] = 2.0

    errors = verify_shared_work_goal_audit(report)

    assert any("fixed payload growth" in error for error in errors)


def test_shared_work_goal_audit_rejects_orbit_payload_growth_ratio() -> None:
    report = copy.deepcopy(_valid_report())
    report["orbit"]["per_frame_payload_growth"] = 4.1
    report["summary"] = summarize(
        report["orbit"],
        report["trained"],
        report["exposure_quadrature"],
        report["exposure_backward"],
        report["exposure_mixed_fallback_backward"],
    )

    errors = verify_shared_work_goal_audit(report)

    assert any("payload growth ratio" in error for error in errors)


def test_shared_work_goal_audit_rejects_missing_underlying_verification() -> None:
    report = copy.deepcopy(_valid_report())
    report["trained"][0]["underlying_errors"] = ["bad timing"]

    errors = verify_shared_work_goal_audit(report)

    assert any("underlying verifier failed" in error for error in errors)


def test_shared_work_goal_audit_rejects_large_trained_entry_ratio() -> None:
    report = copy.deepcopy(_valid_report())
    report["summary"]["max_trained_final_interval_entry_ratio"] = 0.25

    errors = verify_shared_work_goal_audit(report)

    assert any("interval-entry ratios" in error for error in errors)


def test_shared_work_goal_audit_rejects_slow_trained_backward() -> None:
    report = copy.deepcopy(_valid_report())
    report["summary"]["max_trained_final_backward_ms_ratio"] = 0.30

    errors = verify_shared_work_goal_audit(report)

    assert any("backward ratios" in error for error in errors)


def test_shared_work_goal_audit_rejects_large_trained_growth_ratio() -> None:
    report = copy.deepcopy(_valid_report())
    for row in report["trained"]:
        row["shared_interval_entry_growth"] = 1.9
        row["per_frame_interval_entry_growth"] = 7.0
    report["summary"] = summarize(
        report["orbit"],
        report["trained"],
        report["exposure_quadrature"],
        report["exposure_backward"],
        report["exposure_mixed_fallback_backward"],
    )

    errors = verify_shared_work_goal_audit(report)

    assert any("shared/replay interval-entry growth ratio" in error for error in errors)


def test_shared_work_goal_audit_rejects_stale_summary_after_orbit_row_change() -> None:
    report = copy.deepcopy(_valid_report())
    report["orbit"]["final_payload_ratio"] = 0.24

    errors = verify_shared_work_goal_audit(report)

    assert any("summary orbit_final_payload_ratio mismatch" in error for error in errors)


def test_shared_work_goal_audit_rejects_nonmonotone_orbit_frame_counts() -> None:
    report = copy.deepcopy(_valid_report())
    report["orbit"]["frame_counts"] = [4, 16, 8, 32]
    report["summary"] = summarize(
        report["orbit"],
        report["trained"],
        report["exposure_quadrature"],
        report["exposure_backward"],
        report["exposure_mixed_fallback_backward"],
    )

    errors = verify_shared_work_goal_audit(report)

    assert any("orbit frame_counts must be strictly increasing" in error for error in errors)


def test_shared_work_goal_audit_rejects_large_trained_trace_ratio() -> None:
    report = copy.deepcopy(_valid_report())
    report["trained"][0]["final_trace_count_ratio"] = 0.25
    report["summary"] = summarize(
        report["orbit"],
        report["trained"],
        report["exposure_quadrature"],
        report["exposure_backward"],
        report["exposure_mixed_fallback_backward"],
    )

    errors = verify_shared_work_goal_audit(report)

    assert any("trace-count ratios" in error for error in errors)


def test_shared_work_goal_audit_rejects_slow_trained_forward() -> None:
    report = copy.deepcopy(_valid_report())
    report["trained"][1]["final_forward_ms_ratio"] = 0.80
    report["summary"] = summarize(
        report["orbit"],
        report["trained"],
        report["exposure_quadrature"],
        report["exposure_backward"],
        report["exposure_mixed_fallback_backward"],
    )

    errors = verify_shared_work_goal_audit(report)

    assert any("forward ratios" in error for error in errors)


def test_shared_work_goal_audit_rejects_missing_objective_contract() -> None:
    report = copy.deepcopy(_valid_report())
    report["theory_contract"] = "too vague"

    errors = verify_shared_work_goal_audit(report)

    assert any("theory_contract" in error for error in errors)


def test_shared_work_goal_audit_rejects_exposure_underlying_failure() -> None:
    report = copy.deepcopy(_valid_report())
    report["exposure_quadrature"]["underlying_errors"] = ["bad fallback summary"]

    errors = verify_shared_work_goal_audit(report)

    assert any("exposure_quadrature underlying verifier failed" in error for error in errors)


def test_shared_work_goal_audit_rejects_lost_exposure_rolling_reuse() -> None:
    report = copy.deepcopy(_valid_report())
    report["exposure_quadrature"]["rolling_unique_to_row_sample_ratio"] = 1.0
    report["summary"] = summarize(
        report["orbit"],
        report["trained"],
        report["exposure_quadrature"],
        report["exposure_backward"],
        report["exposure_mixed_fallback_backward"],
    )

    errors = verify_shared_work_goal_audit(report)

    assert any("exposure_quadrature rolling sample reuse ratio" in error for error in errors)


def test_shared_work_goal_audit_rejects_missing_backward_metal_case() -> None:
    report = copy.deepcopy(_valid_report())
    report["exposure_backward"]["rolling_has_metal_backward"] = False
    report["exposure_backward"]["metal_backward_case_count"] = 1
    report["summary"] = summarize(
        report["orbit"],
        report["trained"],
        report["exposure_quadrature"],
        report["exposure_backward"],
        report["exposure_mixed_fallback_backward"],
    )

    errors = verify_shared_work_goal_audit(report)

    assert any("exposure_backward must include rolling Metal backward" in error for error in errors)
    assert any("exposure_backward must verify both finite and rolling Metal backward cases" in error for error in errors)


def test_shared_work_goal_audit_rejects_mixed_fallback_underlying_failure() -> None:
    report = copy.deepcopy(_valid_report())
    report["exposure_mixed_fallback_backward"]["underlying_errors"] = ["detached fallback"]

    errors = verify_shared_work_goal_audit(report)

    assert any("exposure_mixed_fallback_backward underlying verifier failed" in error for error in errors)


def test_shared_work_goal_audit_rejects_missing_mixed_fallback_backward_case() -> None:
    report = copy.deepcopy(_valid_report())
    report["exposure_mixed_fallback_backward"]["finite_has_mixed_backward"] = False
    report["exposure_mixed_fallback_backward"]["mixed_backward_case_count"] = 1
    report["summary"] = summarize(
        report["orbit"],
        report["trained"],
        report["exposure_quadrature"],
        report["exposure_backward"],
        report["exposure_mixed_fallback_backward"],
    )

    errors = verify_shared_work_goal_audit(report)

    assert any("finite mixed fallback backward" in error for error in errors)
    assert any("both finite and rolling mixed fallback cases" in error for error in errors)


def test_shared_work_goal_audit_rejects_lost_mixed_fallback_rolling_reuse() -> None:
    report = copy.deepcopy(_valid_report())
    report["exposure_mixed_fallback_backward"]["rolling_unique_to_row_sample_ratio"] = 1.0
    report["summary"] = summarize(
        report["orbit"],
        report["trained"],
        report["exposure_quadrature"],
        report["exposure_backward"],
        report["exposure_mixed_fallback_backward"],
    )

    errors = verify_shared_work_goal_audit(report)

    assert any("exposure_mixed_fallback_backward rolling sample reuse ratio" in error for error in errors)


def test_shared_work_goal_audit_rejects_stale_mixed_fallback_summary() -> None:
    report = copy.deepcopy(_valid_report())
    report["exposure_mixed_fallback_backward"]["max_mixed_grad_abs_error"] = 3.0e-6

    errors = verify_shared_work_goal_audit(report)

    assert any("summary exposure_mixed_fallback_max_grad_abs_error mismatch" in error for error in errors)


def test_shared_work_goal_audit_rejects_stale_exposure_summary() -> None:
    report = copy.deepcopy(_valid_report())
    report["exposure_backward"]["max_metal_grad_abs_error"] = 2.0e-6

    errors = verify_shared_work_goal_audit(report)

    assert any("summary exposure_backward_max_metal_grad_abs_error mismatch" in error for error in errors)


def test_shared_work_goal_audit_reads_current_saved_artifacts() -> None:
    required = (
        DEFAULT_ORBIT_REPORT,
        *DEFAULT_TRAINED_REPORTS,
        DEFAULT_EXPOSURE_QUADRATURE_REPORT,
        DEFAULT_EXPOSURE_BACKWARD_REPORT,
        DEFAULT_EXPOSURE_MIXED_FALLBACK_BACKWARD_REPORT,
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        pytest.skip(f"missing optional audit inputs: {missing}")

    report = run_report()

    assert_shared_work_goal_audit(report)
    assert report["summary"]["orbit_final_payload_ratio"] < 0.25
    assert report["summary"]["orbit_payload_growth_ratio"] < 0.20
    assert report["summary"]["max_trained_final_interval_entry_ratio"] < 0.20
    assert report["summary"]["max_trained_final_trace_count_ratio"] < 0.20
    assert report["summary"]["trained_shared_to_replay_interval_growth_ratio"] < 0.25
    assert report["summary"]["exposure_forward_rolling_unique_to_row_sample_ratio"] < 1.0
    assert report["summary"]["exposure_backward_rolling_unique_to_row_sample_ratio"] < 1.0
    assert report["summary"]["exposure_mixed_fallback_rolling_unique_to_row_sample_ratio"] < 1.0


def test_saved_shared_work_goal_audit_artifact_satisfies_contract() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(Path(summary_json).read_text(encoding="utf-8"))

    assert_shared_work_goal_audit(report)


def test_saved_shared_work_goal_audit_artifact_matches_current_inputs() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    required = (
        summary_json,
        DEFAULT_ORBIT_REPORT,
        *DEFAULT_TRAINED_REPORTS,
        DEFAULT_EXPOSURE_QUADRATURE_REPORT,
        DEFAULT_EXPOSURE_BACKWARD_REPORT,
        DEFAULT_EXPOSURE_MIXED_FALLBACK_BACKWARD_REPORT,
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        pytest.skip(f"missing optional saved artifact inputs: {missing}")

    report = json.loads(Path(summary_json).read_text(encoding="utf-8"))

    assert_shared_work_goal_current_acceptance(report)
