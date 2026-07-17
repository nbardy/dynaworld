from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research_experiments.star_uvt_feature_tubes.projective_goal_final_completion_audit import (
    DEFAULT_OUT_DIR,
    EVIDENCE_ORDER,
    REQUIRED_THEORY_SUBFOLDERS,
    assert_projective_goal_final_completion_audit,
    assert_projective_goal_final_completion_current_acceptance,
    objective_rows,
    run_report,
    summarize,
    verify_projective_goal_final_completion_audit,
    verify_projective_goal_final_completion_current_acceptance,
)
from research_experiments.star_uvt_feature_tubes.projective_goal_progress_audit import (
    PROVEN_REQUIREMENT_IDS,
)


def _artifact(
    summary: dict[str, object],
    requirement_statuses: dict[str, str],
    benchmark: str,
) -> dict[str, object]:
    return {
        "path": f"{benchmark}.json",
        "benchmark": benchmark,
        "status": "ok",
        "verifier_errors": [],
        "current_input_errors": [],
        "summary": summary,
        "requirement_statuses": requirement_statuses,
    }


def _valid_report() -> dict[str, object]:
    progress_statuses = {row_id: "proved" for row_id in PROVEN_REQUIREMENT_IDS}
    progress_statuses["full_goal_completion"] = "open"
    gap_statuses = {
        "formal_goal_memory_and_audit": "proved",
        "sublinear_world_side_work_proxy": "proved",
        "broad_real_scene_quality_acceptance": "proved",
        "full_compiled_adjoint_trainer_replacement": "proved",
        "timing_acceptance_protocol": "proved",
        "full_goal_completion": "partial",
    }
    report: dict[str, object] = {
        "status": "complete",
        "benchmark": "star_uvt_projective_goal_final_completion_audit",
        "goal": "fast 2D rasters across time from 4D spacetime primitives",
        "meta_goal": "share projection/support/binning/visibility/backward work over time",
        "key_math": "UVT trace = pi_* Gamma^* world_primitive",
        "theory": "STAR UVT is one local gauge expression of a camera-ray bundle atlas",
        "evidence": {
            "theory_plan": _artifact(
                {
                    "goal_meta_key_math_exists": True,
                    "required_subfolder_count": 10,
                    "present_required_subfolder_count": 10,
                    "required_subfolders_present": True,
                    "missing_required_subfolders": [],
                    "extra_numbered_subfolders": [],
                    "at_most_ten_numbered_theory_subfolders": True,
                },
                {},
                "gauged_uvt_trace_atlas_theory_plan",
            ),
            "goal_progress": _artifact(
                {
                    "proved_requirement_count": 34,
                    "open_requirement_count": 1,
                    "failed_requirement_count": 0,
                    "is_goal_complete": False,
                },
                progress_statuses,
                "star_uvt_projective_goal_progress_audit",
            ),
            "completion_gap": _artifact(
                {
                    "proved_requirement_count": 5,
                    "partial_requirement_count": 1,
                    "open_gap_ids": ["full_goal_completion"],
                    "completion_ready": False,
                    "does_not_prove_completion": True,
                    "broad_quality_source_gap": 0,
                    "broad_media_source_gap": 0,
                    "broad_quality_frame_count_gap": 0,
                    "strict_timing_failure_gap": 0,
                    "timing_acceptance_gap": 0,
                    "compiled_trainer_source_gap": 0,
                    "compiled_trainer_replacement_gap": 0,
                    "shared_work_proxy": {
                        "passes_proxy_thresholds": True,
                        "orbit_payload_growth_ratio": 0.125,
                        "trained_shared_to_replay_interval_growth_ratio": 0.148,
                        "max_trained_final_backward_ms_ratio": 0.094,
                    },
                    "compiled_replacement": {
                        "final_compiled_adjoint_replacement_accepted": True,
                        "compiled_trainer_replacement_gap": 0,
                    },
                    "timing_protocol": {
                        "final_timing_protocol_accepted": True,
                        "timing_acceptance_gap": 0,
                    },
                },
                gap_statuses,
                "star_uvt_projective_goal_completion_gap",
            ),
        },
        "summary": {},
        "requirements": [],
    }
    report["summary"] = summarize(report)
    report["requirements"] = objective_rows(report["summary"])  # type: ignore[arg-type]
    return report


def test_final_completion_audit_accepts_valid_fixture() -> None:
    report = _valid_report()

    assert verify_projective_goal_final_completion_audit(report) == []
    assert_projective_goal_final_completion_audit(report)
    assert report["summary"]["final_goal_completion_accepted"] is True
    assert report["summary"]["completion_ready"] is True
    assert report["summary"]["does_not_prove_completion"] is False
    assert report["summary"]["open_objective_requirement_ids"] == []


def test_final_completion_current_acceptance_accepts_matching_fixture() -> None:
    report = _valid_report()

    assert verify_projective_goal_final_completion_current_acceptance(report, current_report=copy.deepcopy(report)) == []
    assert_projective_goal_final_completion_current_acceptance(report, current_report=copy.deepcopy(report))


def test_final_completion_current_acceptance_rejects_stale_but_valid_payload() -> None:
    saved = _valid_report()
    current = copy.deepcopy(saved)
    saved["evidence"]["completion_gap"]["summary"]["shared_work_proxy"]["orbit_payload_growth_ratio"] = 0.12
    saved["summary"] = summarize(saved)
    saved["requirements"] = objective_rows(saved["summary"])  # type: ignore[arg-type]

    assert verify_projective_goal_final_completion_audit(saved) == []
    errors = verify_projective_goal_final_completion_current_acceptance(saved, current_report=current)

    assert any("completion_gap.summary.shared_work_proxy.orbit_payload_growth_ratio" in error for error in errors)


def test_final_completion_audit_rejects_missing_memory_contract() -> None:
    report = copy.deepcopy(_valid_report())
    report["key_math"] = "not the right invariant"

    errors = verify_projective_goal_final_completion_audit(report)

    assert any("key_math" in error for error in errors)


def test_final_completion_audit_rejects_underlying_current_input_errors() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["goal_progress"]["current_input_errors"] = ["stale progress"]
    report["summary"] = summarize(report)
    report["requirements"] = objective_rows(report["summary"])  # type: ignore[arg-type]
    report["status"] = "in_progress"

    errors = verify_projective_goal_final_completion_audit(report)

    assert any("goal_progress current-input acceptance failed" in error for error in errors)
    assert any("final_goal_completion_accepted must be true" in error for error in errors)


def test_final_completion_audit_rejects_theory_folder_regression() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["theory_plan"]["summary"]["required_subfolders_present"] = False
    report["evidence"]["theory_plan"]["summary"]["missing_required_subfolders"] = [REQUIRED_THEORY_SUBFOLDERS[0]]
    report["summary"] = summarize(report)
    report["requirements"] = objective_rows(report["summary"])  # type: ignore[arg-type]
    report["status"] = "in_progress"

    errors = verify_projective_goal_final_completion_audit(report)

    assert any("final_goal_completion_accepted must be true" in error for error in errors)
    assert any("all required theory subfolders must be present" in error for error in errors)


def test_final_completion_audit_rejects_progress_math_regression() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["goal_progress"]["requirement_statuses"]["clean_fiber_derivatives"] = "open"
    report["summary"] = summarize(report)
    report["requirements"] = objective_rows(report["summary"])  # type: ignore[arg-type]
    report["status"] = "in_progress"

    errors = verify_projective_goal_final_completion_audit(report)

    assert any("final_goal_completion_accepted must be true" in error for error in errors)
    assert any("open_objective_requirement_ids must be empty" in error for error in errors)


def test_final_completion_audit_rejects_nonzero_concrete_gap() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["completion_gap"]["summary"]["compiled_trainer_replacement_gap"] = 1
    report["summary"] = summarize(report)
    report["requirements"] = objective_rows(report["summary"])  # type: ignore[arg-type]
    report["status"] = "in_progress"

    errors = verify_projective_goal_final_completion_audit(report)

    assert any("all concrete completion-gap counters must be zero" in error for error in errors)


def test_final_completion_audit_reads_current_saved_artifacts() -> None:
    required = (
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_goal_progress_audit/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_goal_completion_gap/summary.json"),
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        pytest.skip(f"missing optional final-audit inputs: {missing}")

    report = run_report()

    assert_projective_goal_final_completion_audit(report)
    assert report["summary"]["final_goal_completion_accepted"] is True


def test_saved_final_completion_audit_artifact_satisfies_contract() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(summary_json.read_text(encoding="utf-8"))

    assert_projective_goal_final_completion_audit(report)


def test_saved_final_completion_audit_artifact_matches_current_inputs() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(summary_json.read_text(encoding="utf-8"))

    assert_projective_goal_final_completion_current_acceptance(report)


def test_final_completion_audit_evidence_order_is_stable() -> None:
    assert EVIDENCE_ORDER == (
        "theory_plan",
        "goal_progress",
        "completion_gap",
    )
