from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research_experiments.star_uvt_feature_tubes.projective_goal_completion_promotion_audit import (
    DEFAULT_OUT_DIR,
    EVIDENCE_ORDER,
    assert_projective_goal_completion_promotion_audit,
    assert_projective_goal_completion_promotion_current_acceptance,
    completion_rows,
    run_report,
    summarize,
    verify_projective_goal_completion_promotion_audit,
    verify_projective_goal_completion_promotion_current_acceptance,
)


def _gap_requirement(requirement_id: str, status: str) -> dict[str, object]:
    return {
        "id": requirement_id,
        "status": status,
        "statement": f"{requirement_id} statement",
        "evidence": ["artifact"],
        "missing": [],
    }


def _gap_artifact() -> dict[str, object]:
    return {
        "path": "goal_completion_gap.json",
        "benchmark": "star_uvt_projective_goal_completion_gap",
        "status": "in_progress",
        "verifier_errors": [],
        "summary": {
            "all_underlying_verifiers_pass": True,
            "completion_ready": False,
            "does_not_prove_completion": True,
            "proved_requirement_count": 5,
            "partial_requirement_count": 1,
            "open_gap_ids": ["full_goal_completion"],
            "broad_quality_source_gap": 0,
            "broad_media_source_gap": 0,
            "broad_quality_frame_count_gap": 0,
            "strict_timing_failure_gap": 0,
            "timing_acceptance_gap": 0,
            "compiled_trainer_source_gap": 0,
            "compiled_trainer_replacement_gap": 0,
            "current_evidence": {
                "proved_requirement_count": 34,
                "open_requirement_count": 1,
                "is_goal_complete": False,
                "acceptance_does_not_prove_completion": True,
                "broad_quality_distinct_sources": 10,
                "broad_media_distinct_sources": 10,
                "real_video_frame_count_count": 4,
                "compiled_trainer_distinct_sources": 10,
            },
            "timing_protocol": {
                "final_timing_protocol_accepted": True,
                "timing_acceptance_gap": 0,
                "fresh_process_median_no_first_ratio": 0.56,
                "fresh_process_median_projective_total_ratio": 0.84,
                "strict_warm_state_failures_demoted_to_caveat": True,
            },
            "compiled_replacement": {
                "final_compiled_adjoint_replacement_accepted": True,
                "compiled_trainer_replacement_gap": 0,
                "source_contract_checks_pass": True,
                "all_cases_projective_interval_main_path": True,
                "all_cases_gradient_flags_present": True,
                "measured_cache_reuse_ok": True,
                "case_payload_count": 20,
                "does_not_prove_completion": True,
            },
            "shared_work_proxy": {
                "passes_proxy_thresholds": True,
                "orbit_payload_growth_ratio": 0.125,
                "trained_shared_to_replay_interval_growth_ratio": 0.148,
                "max_trained_final_backward_ms_ratio": 0.094,
            },
        },
        "requirements": [
            _gap_requirement("formal_goal_memory_and_audit", "proved"),
            _gap_requirement("sublinear_world_side_work_proxy", "proved"),
            _gap_requirement("broad_real_scene_quality_acceptance", "proved"),
            _gap_requirement("full_compiled_adjoint_trainer_replacement", "proved"),
            _gap_requirement("timing_acceptance_protocol", "proved"),
            _gap_requirement("full_goal_completion", "partial"),
        ],
    }


def _valid_report() -> dict[str, object]:
    report: dict[str, object] = {
        "status": "complete",
        "benchmark": "star_uvt_projective_goal_completion_promotion_audit",
        "goal": "fast 2D rasters across time from 4D spacetime primitives",
        "meta_goal": "share projection/support/binning/visibility/backward work over time",
        "key_math": "UVT trace = pi_* Gamma^* world_primitive",
        "theory": "STAR UVT is one local gauge expression of a camera-ray bundle atlas",
        "evidence": {"goal_completion_gap": _gap_artifact()},
        "summary": {},
        "requirements": [],
    }
    report["summary"] = summarize(report)
    report["requirements"] = completion_rows(report["summary"])  # type: ignore[arg-type]
    return report


def test_projective_goal_completion_promotion_audit_accepts_valid_fixture() -> None:
    report = _valid_report()

    assert verify_projective_goal_completion_promotion_audit(report) == []
    assert_projective_goal_completion_promotion_audit(report)
    assert report["summary"]["completion_ready"] is True
    assert report["summary"]["is_goal_complete"] is True
    assert report["summary"]["does_not_prove_completion"] is False
    assert report["summary"]["open_requirement_ids"] == []


def test_projective_goal_completion_promotion_current_acceptance_accepts_matching_payloads() -> None:
    report = _valid_report()

    assert (
        verify_projective_goal_completion_promotion_current_acceptance(
            report,
            current_report=copy.deepcopy(report),
        )
        == []
    )
    assert_projective_goal_completion_promotion_current_acceptance(report, current_report=copy.deepcopy(report))


def test_projective_goal_completion_promotion_current_acceptance_rejects_stale_payload() -> None:
    saved = _valid_report()
    current = copy.deepcopy(saved)
    saved["evidence"]["goal_completion_gap"]["summary"]["shared_work_proxy"][
        "orbit_payload_growth_ratio"
    ] = 0.12
    saved["summary"] = summarize(saved)
    saved["requirements"] = completion_rows(saved["summary"])  # type: ignore[arg-type]

    errors = verify_projective_goal_completion_promotion_current_acceptance(saved, current_report=current)

    assert any("orbit_payload_growth_ratio" in error for error in errors)


def test_projective_goal_completion_promotion_rejects_gap_verifier_error() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["goal_completion_gap"]["verifier_errors"] = ["stale gap"]
    report["summary"] = summarize(report)
    report["requirements"] = completion_rows(report["summary"])  # type: ignore[arg-type]

    errors = verify_projective_goal_completion_promotion_audit(report)

    assert any("goal completion gap evidence verifier failed" in error for error in errors)


def test_projective_goal_completion_promotion_rejects_nonzero_concrete_gap() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["goal_completion_gap"]["summary"]["compiled_trainer_replacement_gap"] = 1
    report["summary"] = summarize(report)
    report["requirements"] = completion_rows(report["summary"])  # type: ignore[arg-type]

    errors = verify_projective_goal_completion_promotion_audit(report)

    assert any("concrete gap compiled_trainer_replacement_gap must be zero" in error for error in errors)


def test_projective_goal_completion_promotion_rejects_extra_source_open_gap() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["goal_completion_gap"]["summary"]["open_gap_ids"] = [
        "full_goal_completion",
        "new_gap",
    ]
    report["summary"] = summarize(report)
    report["requirements"] = completion_rows(report["summary"])  # type: ignore[arg-type]

    errors = verify_projective_goal_completion_promotion_audit(report)

    assert any("source gap report must have only full_goal_completion open" in error for error in errors)


def test_projective_goal_completion_promotion_rejects_low_goal_progress_inventory() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["goal_completion_gap"]["summary"]["current_evidence"][
        "proved_requirement_count"
    ] = 33
    report["summary"] = summarize(report)
    report["requirements"] = completion_rows(report["summary"])  # type: ignore[arg-type]

    errors = verify_projective_goal_completion_promotion_audit(report)

    assert any("goal-progress evidence must prove at least 34 rows" in error for error in errors)


def test_projective_goal_completion_promotion_rejects_unscoped_source_noncompletion() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["goal_completion_gap"]["summary"]["does_not_prove_completion"] = False
    report["summary"] = summarize(report)
    report["requirements"] = completion_rows(report["summary"])  # type: ignore[arg-type]

    errors = verify_projective_goal_completion_promotion_audit(report)

    assert any("source non-completion flags must be scoped" in error for error in errors)


def test_projective_goal_completion_promotion_rejects_tampered_completion_flag() -> None:
    report = copy.deepcopy(_valid_report())
    report["summary"]["completion_ready"] = False

    errors = verify_projective_goal_completion_promotion_audit(report)

    assert any("completion_ready must be true" in error for error in errors)


def test_projective_goal_completion_promotion_reads_current_saved_gap_artifact() -> None:
    required = (Path("outputs/benchmarks/2026-05-25_star_uvt_projective_goal_completion_gap/summary.json"),)
    missing = [path for path in required if not path.exists()]
    if missing:
        pytest.skip(f"missing optional promotion-audit inputs: {missing}")

    report = run_report()

    assert_projective_goal_completion_promotion_audit(report)
    assert report["summary"]["is_goal_complete"] is True


def test_saved_projective_goal_completion_promotion_artifact_satisfies_contract() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(summary_json.read_text(encoding="utf-8"))

    assert_projective_goal_completion_promotion_audit(report)


def test_saved_projective_goal_completion_promotion_artifact_matches_current_inputs() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(summary_json.read_text(encoding="utf-8"))

    assert_projective_goal_completion_promotion_current_acceptance(report)


def test_projective_goal_completion_promotion_evidence_order_is_stable() -> None:
    assert EVIDENCE_ORDER == ("goal_completion_gap",)
