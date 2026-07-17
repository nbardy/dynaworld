from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research_experiments.star_uvt_feature_tubes.projective_goal_completion_gap_report import (
    DEFAULT_OUT_DIR,
    EVIDENCE_ORDER,
    assert_projective_goal_completion_gap_current_acceptance,
    assert_projective_goal_completion_gap_report,
    completion_rows,
    run_report,
    summarize,
    verify_projective_goal_completion_gap_current_acceptance,
    verify_projective_goal_completion_gap_report,
)


def _artifact(summary: dict[str, object], benchmark: str) -> dict[str, object]:
    return {
        "path": f"{benchmark}.json",
        "benchmark": benchmark,
        "status": "ok",
        "verifier_errors": [],
        "summary": summary,
    }


def _valid_report() -> dict[str, object]:
    report: dict[str, object] = {
        "status": "in_progress",
        "benchmark": "star_uvt_projective_goal_completion_gap",
        "goal": "fast 2D rasters across time from 4D spacetime primitives",
        "meta_goal": "share projection/support/binning/visibility/backward work over time",
        "key_math": "UVT trace = pi_* Gamma^* world_primitive",
        "theory": "STAR UVT is one local gauge expression of a camera-ray bundle atlas",
        "evidence": {
            "goal_progress": _artifact(
                {
                    "proved_requirement_count": 34,
                    "open_requirement_count": 1,
                    "is_goal_complete": False,
                },
                "star_uvt_projective_goal_progress_audit",
            ),
            "real_video_acceptance_envelope": _artifact(
                {
                    "functional_distinct_youtube_id_count": 5,
                    "quality_scene_count": 5,
                    "media_scene_count": 5,
                    "broad_media_distinct_youtube_id_count": 10,
                    "broad_frame_count_count": 4,
                    "frame_count_breadth_frame_count_count": 4,
                    "frame_scaling_frame_count_count": 3,
                    "min_quality_psnr_gain": 0.022,
                    "max_support_rebins": 0,
                    "max_stale_refreshes": 0,
                    "does_not_prove_completion": True,
                },
                "star_uvt_projective_real_video_acceptance_envelope",
            ),
            "real_video_broad10_trainer_matrix": _artifact(
                {
                    "distinct_youtube_id_count": 10,
                    "row_count": 20,
                    "max_measured_support_rebins": 0,
                    "max_measured_stale_refreshes": 0,
                },
                "star_uvt_projective_real_video_multiscene_trainer_matrix",
            ),
            "real_video_broad10_quality_tether": _artifact(
                {
                    "distinct_youtube_id_count": 10,
                    "pair_count": 10,
                    "all_measured_loss_curves_match_cadence": True,
                    "all_gradient_flags_present": True,
                    "all_measured_psnr_improves": True,
                },
                "star_uvt_projective_real_video_broad10_quality_tether",
            ),
            "real_video_broad10_media_tether": _artifact(
                {
                    "distinct_youtube_id_count": 10,
                    "pair_count": 10,
                    "all_contact_sheet_pixels_match_cadence": True,
                    "all_contact_sheet_hashes_match_cadence": True,
                    "all_gradient_flags_present": True,
                    "all_measured_psnr_improves": True,
                },
                "star_uvt_projective_real_video_multiscene_media_tether",
            ),
            "real_video_timing_variance_envelope": _artifact(
                {
                    "strict_failure_count": 2,
                    "strict_timing_win_claimed": False,
                    "fresh_process_timing_acceptance_status": "pass",
                    "fresh_process_median_no_first_ratio": 0.56,
                    "fresh_process_median_projective_total_ratio": 0.84,
                    "fresh_process_median_feature_state_update_ratio": 0.85,
                    "workload_explains_render_forward_miss_count": 0,
                    "does_not_prove_completion": True,
                },
                "star_uvt_projective_real_video_timing_variance_envelope",
            ),
            "real_video_timing_protocol_acceptance": _artifact(
                {
                    "final_timing_protocol_accepted": True,
                    "protocol_name": "fresh_process_median_with_warmup_discard",
                    "timing_acceptance_gap": 0,
                    "fresh_process_median_no_first_ratio": 0.56,
                    "fresh_process_median_projective_total_ratio": 0.84,
                    "fresh_process_median_feature_state_update_ratio": 0.85,
                    "strict_warm_state_failure_count": 2,
                    "strict_warm_state_failures_demoted_to_caveat": True,
                    "broad_real_context_passes": True,
                    "frame_count_breadth_passes": True,
                },
                "star_uvt_projective_real_video_timing_protocol_acceptance",
            ),
            "real_video_compiled_adjoint_replacement": _artifact(
                {
                    "final_compiled_adjoint_replacement_accepted": True,
                    "compiled_trainer_replacement_gap": 0,
                    "source_contract_checks_pass": True,
                    "broad_context_passes": True,
                    "clean_cache_and_support": True,
                    "all_cases_projective_interval_main_path": True,
                    "all_cases_gradient_flags_present": True,
                    "measured_cache_reuse_ok": True,
                    "case_payload_count": 20,
                    "broad10_trainer_distinct_youtube_id_count": 10,
                    "does_not_prove_completion": True,
                },
                "star_uvt_projective_real_video_compiled_adjoint_replacement",
            ),
            "shared_work": _artifact(
                {
                    "orbit_payload_growth_ratio": 0.125,
                    "trained_shared_to_replay_interval_growth_ratio": 0.148,
                    "max_trained_final_backward_ms_ratio": 0.094,
                },
                "star_uvt_projective_shared_work_goal_audit",
            ),
        },
        "summary": {},
        "requirements": [],
    }
    report["summary"] = summarize(report)
    report["requirements"] = completion_rows(report["summary"])  # type: ignore[arg-type]
    return report


def test_projective_goal_completion_gap_report_accepts_valid_fixture() -> None:
    report = _valid_report()

    assert verify_projective_goal_completion_gap_report(report) == []
    assert report["summary"]["completion_ready"] is False
    assert report["summary"]["does_not_prove_completion"] is True
    assert report["summary"]["open_gap_ids"] == [
        "full_goal_completion",
    ]
    assert report["summary"]["strict_timing_failure_gap"] == 0
    assert report["summary"]["timing_acceptance_gap"] == 0
    assert report["summary"]["compiled_trainer_replacement_gap"] == 0


def test_projective_goal_completion_gap_current_acceptance_accepts_matching_payloads() -> None:
    report = _valid_report()

    assert verify_projective_goal_completion_gap_current_acceptance(report, current_report=copy.deepcopy(report)) == []
    assert_projective_goal_completion_gap_current_acceptance(report, current_report=copy.deepcopy(report))


def test_projective_goal_completion_gap_current_acceptance_rejects_stale_but_valid_payload() -> None:
    saved = _valid_report()
    current = copy.deepcopy(saved)
    saved["evidence"]["shared_work"]["summary"]["orbit_payload_growth_ratio"] = 0.12
    saved["summary"] = summarize(saved)
    saved["requirements"] = completion_rows(saved["summary"])  # type: ignore[arg-type]

    assert verify_projective_goal_completion_gap_report(saved) == []
    errors = verify_projective_goal_completion_gap_current_acceptance(saved, current_report=current)

    assert any("evidence.shared_work.summary.orbit_payload_growth_ratio" in error for error in errors)


def test_projective_goal_completion_gap_report_requires_all_evidence_rows() -> None:
    report = copy.deepcopy(_valid_report())
    del report["evidence"]["shared_work"]

    errors = verify_projective_goal_completion_gap_report(report)

    assert any("evidence shared_work must be an object" in error for error in errors)


def test_projective_goal_completion_gap_report_rejects_broad10_trainer_regression() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["real_video_broad10_trainer_matrix"]["summary"]["distinct_youtube_id_count"] = 9
    report["summary"] = summarize(report)

    errors = verify_projective_goal_completion_gap_report(report)

    assert any("broad10 trainer evidence must cover at least 10 distinct sources" in error for error in errors)


def test_projective_goal_completion_gap_report_rejects_broad10_quality_regression() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["real_video_broad10_quality_tether"]["summary"]["distinct_youtube_id_count"] = 9
    report["summary"] = summarize(report)

    errors = verify_projective_goal_completion_gap_report(report)

    assert any("broad10 quality evidence must cover at least 10 distinct sources" in error for error in errors)


def test_projective_goal_completion_gap_report_rejects_broad10_media_regression() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["real_video_broad10_media_tether"]["summary"]["distinct_youtube_id_count"] = 9
    report["summary"] = summarize(report)

    errors = verify_projective_goal_completion_gap_report(report)

    assert any("broad10 media evidence must cover at least 10 distinct sources" in error for error in errors)


def test_projective_goal_completion_gap_report_rejects_premature_completion() -> None:
    report = copy.deepcopy(_valid_report())
    report["summary"]["completion_ready"] = True

    errors = verify_projective_goal_completion_gap_report(report)

    assert any("completion_ready must be false" in error for error in errors)


def test_projective_goal_completion_gap_report_rejects_unaccepted_compiled_replacement() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["real_video_compiled_adjoint_replacement"]["summary"][  # type: ignore[index]
        "final_compiled_adjoint_replacement_accepted"
    ] = False
    report["evidence"]["real_video_compiled_adjoint_replacement"]["summary"][  # type: ignore[index]
        "compiled_trainer_replacement_gap"
    ] = 1
    report["summary"] = summarize(report)
    report["requirements"] = completion_rows(report["summary"])  # type: ignore[arg-type]

    errors = verify_projective_goal_completion_gap_report(report)

    assert any("final compiled-adjoint replacement must be accepted" in error for error in errors)


def test_projective_goal_completion_gap_report_rejects_marking_full_goal_complete() -> None:
    report = copy.deepcopy(_valid_report())
    for row in report["requirements"]:
        if row["id"] == "full_goal_completion":
            row["status"] = "proved"

    errors = verify_projective_goal_completion_gap_report(report)

    assert any("full goal completion must remain partial" in error for error in errors)


def test_projective_goal_completion_gap_report_rejects_low_source_target() -> None:
    report = copy.deepcopy(_valid_report())
    report["summary"]["completion_targets"]["broad_quality_min_distinct_sources"] = 5

    errors = verify_projective_goal_completion_gap_report(report)

    assert any("broad quality source target must stay at least 10" in error for error in errors)


def test_projective_goal_completion_gap_report_rejects_low_media_source_target() -> None:
    report = copy.deepcopy(_valid_report())
    report["summary"]["completion_targets"]["broad_media_min_distinct_sources"] = 5

    errors = verify_projective_goal_completion_gap_report(report)

    assert any("broad media source target must stay at least 10" in error for error in errors)


def test_projective_goal_completion_gap_report_reads_current_saved_artifacts() -> None:
    required = (
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_goal_progress_audit/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_acceptance_envelope/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_trainer_matrix_broad10/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_broad10_quality_tether/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_broad10_media_tether/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_timing_variance_envelope/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_timing_protocol_acceptance/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_compiled_adjoint_replacement/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_shared_work_goal_audit/summary.json"),
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        pytest.skip(f"missing optional completion-gap inputs: {missing}")

    report = run_report()

    assert_projective_goal_completion_gap_report(report)
    assert report["summary"]["completion_ready"] is False


def test_saved_projective_goal_completion_gap_report_satisfies_contract() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(summary_json.read_text(encoding="utf-8"))

    assert_projective_goal_completion_gap_report(report)


def test_saved_projective_goal_completion_gap_report_matches_current_inputs() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(summary_json.read_text(encoding="utf-8"))

    assert_projective_goal_completion_gap_current_acceptance(report)


def test_projective_goal_completion_gap_report_evidence_order_is_stable() -> None:
    assert EVIDENCE_ORDER == (
        "goal_progress",
        "real_video_acceptance_envelope",
        "real_video_broad10_trainer_matrix",
        "real_video_broad10_quality_tether",
        "real_video_broad10_media_tether",
        "real_video_timing_variance_envelope",
        "real_video_timing_protocol_acceptance",
        "real_video_compiled_adjoint_replacement",
        "shared_work",
    )
