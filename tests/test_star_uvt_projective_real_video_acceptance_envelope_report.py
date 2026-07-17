from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research_experiments.star_uvt_feature_tubes.projective_real_video_acceptance_envelope_report import (
    DEFAULT_OUT_DIR,
    EVIDENCE_ORDER,
    assert_real_video_acceptance_envelope_report,
    summarize,
    verify_real_video_acceptance_envelope_report,
)


def _trainer_summary(scene_count: int) -> dict[str, object]:
    return {
        "scene_count": scene_count,
        "row_count": scene_count * 2,
        "measured_row_count": scene_count,
        "distinct_youtube_id_count": scene_count,
        "all_source_videos_exist": True,
        "all_rows_pass": True,
        "all_rows_loss_decreased": True,
        "all_rows_no_overflow": True,
        "all_rows_fallback_free": True,
        "all_rows_visibility_stratification_free": True,
        "all_measured_loss_matches_cadence": True,
        "max_measured_vs_cadence_end_loss_abs_delta": 0.0,
        "max_measured_vs_cadence_no_first_step_ms_ratio": 0.73,
        "max_measured_vs_cadence_rebuild_ratio": 0.5,
        "max_measured_support_rebins": 0,
        "max_measured_stale_refreshes": 0,
    }


def _frame_scaling_summary() -> dict[str, object]:
    summary = _trainer_summary(3)
    summary.update(
        {
            "frame_count_count": 3,
            "row_count": 18,
            "measured_row_count": 9,
            "frame_growth_factor": 4.0,
            "max_measured_no_first_growth_vs_frame_growth_ratio": 0.42,
        }
    )
    return summary


def _extended_frame_summary() -> dict[str, object]:
    return {
        "source_status": "failed",
        "source_scene_count": 5,
        "source_distinct_youtube_id_count": 5,
        "source_row_count": 30,
        "source_measured_row_count": 15,
        "source_frame_count_count": 3,
        "source_frame_growth_factor": 4.0,
        "strict_failure_count": 2,
        "strict_failed_only_expected_timing": True,
        "all_source_videos_exist": True,
        "all_rows_pass": True,
        "all_rows_loss_decreased": True,
        "all_rows_no_overflow": True,
        "all_rows_fallback_free": True,
        "all_rows_visibility_stratification_free": True,
        "all_measured_loss_matches_cadence": True,
        "max_measured_vs_cadence_end_loss_abs_delta": 0.0,
        "max_measured_vs_cadence_rebuild_ratio": 0.5,
        "max_measured_cache_rebuild_growth": 1.0,
        "max_measured_support_rebins": 0,
        "max_measured_stale_refreshes": 0,
        "max_measured_support_tail_alpha_bound": 0.0,
        "max_measured_support_overshoot_px": 0.0,
        "max_motion_score": 7.0,
        "max_tile_count": 22,
        "max_measured_vs_cadence_no_first_step_ms_ratio": 1.18,
        "max_measured_no_first_growth_vs_frame_growth_ratio": 1.001,
        "no_first_timing_win": False,
        "no_first_growth_sublinear": False,
        "max_no_first_ratio_overage": 0.18,
        "max_growth_ratio_overage": 0.001,
    }


def _frame_count_breadth_summary() -> dict[str, object]:
    return {
        "source_status": "failed",
        "source_scene_count": 3,
        "source_distinct_youtube_id_count": 3,
        "source_row_count": 24,
        "source_measured_row_count": 12,
        "source_frame_count_count": 4,
        "source_frame_growth_factor": 8.0,
        "strict_failure_count": 1,
        "strict_failed_only_expected_timing": True,
        "all_source_videos_exist": True,
        "all_rows_pass": True,
        "all_rows_loss_decreased": True,
        "all_rows_no_overflow": True,
        "all_rows_fallback_free": True,
        "all_rows_visibility_stratification_free": True,
        "all_measured_loss_matches_cadence": True,
        "max_measured_vs_cadence_end_loss_abs_delta": 1.0e-8,
        "max_measured_vs_cadence_rebuild_ratio": 0.5,
        "max_measured_cache_rebuild_growth": 1.0,
        "max_measured_support_rebins": 0,
        "max_measured_stale_refreshes": 0,
        "max_measured_support_tail_alpha_bound": 0.0,
        "max_measured_support_overshoot_px": 0.0,
        "max_motion_score": 5.8,
        "max_tile_count": 22,
        "max_measured_vs_cadence_no_first_step_ms_ratio": 1.34,
        "max_measured_no_first_growth_vs_frame_growth_ratio": 0.23,
        "no_first_timing_win": False,
        "no_first_growth_sublinear": True,
        "max_no_first_ratio_overage": 0.34,
        "max_growth_ratio_overage": 0.0,
        "frame_count_breadth_accepted": True,
    }


def _quality_summary(scene_count: int) -> dict[str, object]:
    return {
        "source_scene_count": scene_count,
        "source_distinct_youtube_id_count": scene_count,
        "scene_count": scene_count,
        "distinct_youtube_id_count": scene_count,
        "pair_count": scene_count,
        "all_case_files_exist": True,
        "all_rows_pass": True,
        "all_rows_error_free": True,
        "all_gradient_flags_present": True,
        "all_measured_loss_curves_match_cadence": True,
        "all_measured_rgb_loss_curves_match_cadence": True,
        "all_measured_end_psnr_matches_cadence": True,
        "all_measured_psnr_improves": True,
        "all_measured_loss_decreases": True,
        "max_abs_loss_curve_delta": 0.0,
        "max_abs_rgb_loss_curve_delta": 0.0,
        "max_end_loss_abs_delta": 0.0,
        "max_end_psnr_abs_delta": 0.0,
        "min_measured_psnr_gain": 0.04,
        "min_measured_loss_decrease": 0.001,
        "min_measured_end_psnr": 5.0,
        "max_measured_end_loss": 0.31,
    }


def _media_summary(scene_count: int) -> dict[str, object]:
    return {
        "scene_count": scene_count,
        "pair_count": scene_count,
        "case_row_count": scene_count * 2,
        "measured_row_count": scene_count,
        "cadence_row_count": scene_count,
        "distinct_youtube_id_count": scene_count,
        "all_source_videos_exist": True,
        "all_case_rows_pass": True,
        "all_contact_sheets_exist": True,
        "all_contact_sheet_pixels_match_cadence": True,
        "all_contact_sheet_hashes_match_cadence": True,
        "all_contact_sheet_layouts_valid": True,
        "all_contact_sheet_metrics_match_payload": True,
        "all_contact_sheet_rows_nontrivial": True,
        "all_loss_curves_match_cadence": True,
        "all_rgb_loss_curves_match_cadence": True,
        "all_final_full_rgb_losses_match_cadence": True,
        "all_final_full_rgb_psnr_matches_cadence": True,
        "all_gradient_flags_present": True,
        "all_measured_loss_decreases": True,
        "all_measured_psnr_improves": True,
        "all_measured_media_rendered": True,
        "all_cadence_media_rendered": True,
        "all_rows_no_overflow": True,
        "all_rows_fallback_free": True,
        "all_rows_visibility_stratification_free": True,
        "max_abs_contact_sheet_delta": 0,
        "max_mean_abs_contact_sheet_delta": 0.0,
        "max_contact_sheet_target_pred_mse_delta": 0.0,
        "max_contact_sheet_payload_loss_abs_delta": 0.001,
        "max_abs_loss_curve_delta": 0.0,
        "max_abs_rgb_loss_curve_delta": 0.0,
        "max_final_full_rgb_loss_abs_delta": 0.0,
        "max_final_full_rgb_psnr_abs_delta": 0.0,
        "min_measured_loss_decrease": 0.001,
        "min_measured_psnr_gain": 0.04,
        "min_contact_sheet_pixel_count": 1000,
        "min_contact_sheet_target_std": 0.14,
        "min_contact_sheet_pred_std": 0.07,
        "min_contact_sheet_target_pred_mse": 0.098,
        "max_measured_vs_cadence_no_first_step_ms_ratio": 0.99,
        "max_measured_vs_cadence_rebuild_ratio": 0.5,
    }


def _bq4_fresh_process_summary() -> dict[str, object]:
    return {
        "target_frames": 16,
        "requested_repeat_count": 3,
        "warmup_discard_repeats": 1,
        "acceptance_ratio_threshold": 1.0,
        "paired_ratio_count": 6,
        "all_rows_fresh_process": True,
        "all_expected_global_steps_traced": True,
        "all_projective_interval_timing_present": True,
        "all_rows_cache_support_clean": True,
        "all_target_summary": {
            "pair_count": 6,
            "median_no_first_ratio": 0.65,
            "median_projective_total_ratio": 0.84,
            "median_feature_state_update_ratio": 0.72,
            "max_no_first_ratio": 0.71,
            "max_projective_total_ratio": 2.25,
            "max_feature_state_update_ratio": 1.3,
            "no_first_bump_count": 0,
            "projective_total_bump_count": 1,
            "feature_state_update_bump_count": 2,
        },
        "timing_acceptance": {
            "status": "pass",
            "median_ratios_within_threshold": True,
            "post_warmup_pair_count": 4,
            "post_warmup_summary": {
                "pair_count": 4,
                "median_no_first_ratio": 0.56,
                "median_projective_total_ratio": 0.84,
                "median_feature_state_update_ratio": 0.85,
                "max_no_first_ratio": 0.71,
                "max_projective_total_ratio": 2.25,
                "max_feature_state_update_ratio": 1.3,
                "no_first_bump_count": 0,
                "projective_total_bump_count": 1,
                "feature_state_update_bump_count": 2,
            },
        },
    }


def _evidence(summary: dict[str, object]) -> dict[str, object]:
    return {
        "path": "artifact.json",
        "benchmark": "underlying",
        "status": "ok",
        "verifier_errors": [],
        "summary": summary,
    }


def _valid_report() -> dict[str, object]:
    evidence = {
        "trainer_matrix": _evidence(_trainer_summary(3)),
        "extended_trainer_matrix": _evidence(_trainer_summary(5)),
        "frame_scaling_matrix": _evidence(_frame_scaling_summary()),
        "extended_frame_scaling_diagnostic": _evidence(_extended_frame_summary()),
        "frame_count_breadth_diagnostic": _evidence(_frame_count_breadth_summary()),
        "quality_tether": _evidence(_quality_summary(3)),
        "extended_quality_tether": _evidence(_quality_summary(5)),
        "broad10_quality_tether": _evidence(_quality_summary(10)),
        "media_tether": _evidence(_media_summary(3)),
        "extended_media_tether": _evidence(_media_summary(5)),
        "broad10_media_tether": _evidence(_media_summary(10)),
        "bq4_fresh_process_timing": _evidence(_bq4_fresh_process_summary()),
    }
    report: dict[str, object] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_real_video_acceptance_envelope",
        "goal": "fast 2D rasters across time from 4D spacetime primitives",
        "meta_goal": "share projection/support/binning/visibility/backward work over time",
        "theory_contract": (
            "This acceptance envelope is a five-source real-video check with frame-count breadth, broad10 quality, and broad10 media tethering. It does not prove broad "
            "real-scene quality acceptance and does not prove full goal completion."
        ),
        "does_not_prove_completion": True,
        "evidence": evidence,
    }
    report["summary"] = summarize(report)  # type: ignore[arg-type]
    return report


def test_real_video_acceptance_envelope_accepts_valid_report() -> None:
    report = _valid_report()

    assert verify_real_video_acceptance_envelope_report(report) == []
    assert_real_video_acceptance_envelope_report(report)
    assert report["summary"]["strict_timing_win_claimed"] is False  # type: ignore[index]
    assert report["summary"]["fresh_process_median_timing_win_claimed"] is True  # type: ignore[index]
    assert report["summary"]["bq4_fresh_process_post_warmup_median_projective_total_ratio"] == 0.84  # type: ignore[index]
    assert report["summary"]["broad_quality_distinct_youtube_id_count"] == 10  # type: ignore[index]
    assert report["summary"]["broad_media_distinct_youtube_id_count"] == 10  # type: ignore[index]
    assert report["summary"]["broad_frame_count_count"] == 4  # type: ignore[index]


def test_real_video_acceptance_envelope_rejects_missing_scope_contract() -> None:
    report = _valid_report()
    report["theory_contract"] = "generic success"

    errors = verify_real_video_acceptance_envelope_report(report)

    assert any("non-completion scope" in error for error in errors)


def test_real_video_acceptance_envelope_rejects_underlying_verifier_error() -> None:
    report = _valid_report()
    report["evidence"]["quality_tether"]["verifier_errors"] = ["bad quality row"]  # type: ignore[index]

    errors = verify_real_video_acceptance_envelope_report(report)

    assert any("quality_tether verifier failed" in error for error in errors)


def test_real_video_acceptance_envelope_rejects_lost_five_source_coverage() -> None:
    report = _valid_report()
    report["evidence"]["extended_trainer_matrix"]["summary"]["scene_count"] = 4  # type: ignore[index]
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_real_video_acceptance_envelope_report(report)

    assert any("extended trainer matrix must cover at least 5 scenes" in error for error in errors)


def test_real_video_acceptance_envelope_rejects_timing_overclaim() -> None:
    report = _valid_report()
    report["evidence"]["extended_frame_scaling_diagnostic"]["summary"]["no_first_timing_win"] = True  # type: ignore[index]
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_real_video_acceptance_envelope_report(report)

    assert any("must not claim a no-first timing win" in error for error in errors)


def test_real_video_acceptance_envelope_rejects_fresh_process_median_regression() -> None:
    report = _valid_report()
    report["evidence"]["bq4_fresh_process_timing"]["summary"]["timing_acceptance"]["status"] = "fail"  # type: ignore[index]
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_real_video_acceptance_envelope_report(report)

    assert any("post-warmup median acceptance must pass" in error for error in errors)


def test_real_video_acceptance_envelope_rejects_fresh_process_no_first_bump() -> None:
    report = _valid_report()
    report["evidence"]["bq4_fresh_process_timing"]["summary"]["all_target_summary"]["no_first_bump_count"] = 1  # type: ignore[index]
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_real_video_acceptance_envelope_report(report)

    assert any("must have zero no-first bumps" in error for error in errors)


def test_real_video_acceptance_envelope_rejects_media_delta_regression() -> None:
    report = _valid_report()
    report["evidence"]["extended_media_tether"]["summary"]["max_abs_contact_sheet_delta"] = 1  # type: ignore[index]
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_real_video_acceptance_envelope_report(report)

    assert any("contact sheets must pixel-match cadence" in error for error in errors)


def test_real_video_acceptance_envelope_rejects_quality_delta_regression() -> None:
    report = _valid_report()
    report["evidence"]["extended_quality_tether"]["summary"]["max_abs_loss_curve_delta"] = 1.0e-4  # type: ignore[index]
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_real_video_acceptance_envelope_report(report)

    assert any("loss curves must match cadence" in error for error in errors)


def test_real_video_acceptance_envelope_rejects_lost_broad10_quality_coverage() -> None:
    report = _valid_report()
    report["evidence"]["broad10_quality_tether"]["summary"]["distinct_youtube_id_count"] = 9  # type: ignore[index]
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_real_video_acceptance_envelope_report(report)

    assert any("broad10 quality tether must cover at least 10 source-distinct videos" in error for error in errors)


def test_real_video_acceptance_envelope_rejects_lost_broad10_media_coverage() -> None:
    report = _valid_report()
    report["evidence"]["broad10_media_tether"]["summary"]["distinct_youtube_id_count"] = 9  # type: ignore[index]
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_real_video_acceptance_envelope_report(report)

    assert any("broad10 media tether must cover at least 10 source-distinct videos" in error for error in errors)


def test_real_video_acceptance_envelope_rejects_lost_frame_count_breadth() -> None:
    report = _valid_report()
    report["evidence"]["frame_count_breadth_diagnostic"]["summary"]["source_frame_count_count"] = 3  # type: ignore[index]
    report["evidence"]["frame_count_breadth_diagnostic"]["summary"]["frame_count_breadth_accepted"] = False  # type: ignore[index]
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_real_video_acceptance_envelope_report(report)

    assert any("frame-count breadth diagnostic must cover at least four frame counts" in error for error in errors)


def test_real_video_acceptance_envelope_rejects_stale_summary() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["extended_media_tether"]["summary"]["max_measured_vs_cadence_rebuild_ratio"] = 0.75  # type: ignore[index]

    errors = verify_real_video_acceptance_envelope_report(report)

    assert any("summary max_rebuild_ratio mismatch" in error for error in errors)


def test_real_video_acceptance_envelope_keeps_expected_evidence_keys() -> None:
    report = _valid_report()

    assert tuple(report["evidence"]) == EVIDENCE_ORDER


def test_saved_real_video_acceptance_envelope_artifact_satisfies_contract() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(Path(summary_json).read_text(encoding="utf-8"))

    assert_real_video_acceptance_envelope_report(report)
