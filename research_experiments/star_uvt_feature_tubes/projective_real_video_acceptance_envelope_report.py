from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from projective_real_video_multiscene_extended_frame_scaling_diagnostic_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_EXTENDED_FRAME_SCALING_DIAGNOSTIC_OUT_DIR,
    verify_extended_frame_scaling_diagnostic_report,
)
from projective_real_video_multiscene_extended_quality_tether_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_EXTENDED_QUALITY_TETHER_OUT_DIR,
    verify_real_video_multiscene_extended_quality_tether_report,
)
from projective_real_video_multiscene_frame_scaling_matrix import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_FRAME_SCALING_MATRIX_OUT_DIR,
    verify_real_video_multiscene_frame_scaling_matrix_report,
)
from projective_real_video_frame_count_breadth_diagnostic_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_FRAME_COUNT_BREADTH_DIAGNOSTIC_OUT_DIR,
    verify_frame_count_breadth_diagnostic_report,
)
from projective_real_video_multiscene_bq4_trace_fresh_process_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_BQ4_TRACE_FRESH_PROCESS_OUT_DIR,
    verify_bq4_trace_fresh_process_report,
)
from projective_real_video_multiscene_media_tether_report import (  # noqa: E402
    CONTACT_SHEET_PAYLOAD_LOSS_TOLERANCE,
    DEFAULT_OUT_DIR as DEFAULT_MEDIA_TETHER_OUT_DIR,
    MEDIA_SCALAR_LOSS_TOLERANCE,
    MEDIA_SCALAR_PSNR_TOLERANCE,
    verify_real_video_multiscene_media_tether_report,
)
from projective_real_video_multiscene_quality_tether_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_QUALITY_TETHER_OUT_DIR,
    verify_real_video_multiscene_quality_tether_report,
)
from projective_real_video_broad10_quality_tether_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_BROAD10_QUALITY_TETHER_OUT_DIR,
    verify_real_video_broad10_quality_tether_report,
)
from projective_real_video_multiscene_trainer_matrix import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_TRAINER_MATRIX_OUT_DIR,
    verify_real_video_multiscene_trainer_matrix_report,
)


DEFAULT_OUT_DIR = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-25_star_uvt_projective_real_video_acceptance_envelope"
)
DEFAULT_TRAINER_MATRIX_REPORT = DEFAULT_TRAINER_MATRIX_OUT_DIR / "summary.json"
DEFAULT_EXTENDED_TRAINER_MATRIX_REPORT = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-25_star_uvt_projective_real_video_multiscene_trainer_matrix_extended5"
    / "summary.json"
)
DEFAULT_FRAME_SCALING_MATRIX_REPORT = DEFAULT_FRAME_SCALING_MATRIX_OUT_DIR / "summary.json"
DEFAULT_EXTENDED_FRAME_SCALING_DIAGNOSTIC_REPORT = (
    DEFAULT_EXTENDED_FRAME_SCALING_DIAGNOSTIC_OUT_DIR / "summary.json"
)
DEFAULT_FRAME_COUNT_BREADTH_DIAGNOSTIC_REPORT = DEFAULT_FRAME_COUNT_BREADTH_DIAGNOSTIC_OUT_DIR / "summary.json"
DEFAULT_QUALITY_TETHER_REPORT = DEFAULT_QUALITY_TETHER_OUT_DIR / "summary.json"
DEFAULT_EXTENDED_QUALITY_TETHER_REPORT = DEFAULT_EXTENDED_QUALITY_TETHER_OUT_DIR / "summary.json"
DEFAULT_BROAD10_QUALITY_TETHER_REPORT = DEFAULT_BROAD10_QUALITY_TETHER_OUT_DIR / "summary.json"
DEFAULT_MEDIA_TETHER_REPORT = DEFAULT_MEDIA_TETHER_OUT_DIR / "summary.json"
DEFAULT_EXTENDED_MEDIA_TETHER_REPORT = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-25_star_uvt_projective_real_video_multiscene_extended_media_tether"
    / "summary.json"
)
DEFAULT_BROAD10_MEDIA_TETHER_REPORT = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-25_star_uvt_projective_real_video_broad10_media_tether"
    / "summary.json"
)
DEFAULT_BQ4_TRACE_FRESH_PROCESS_REPORT = DEFAULT_BQ4_TRACE_FRESH_PROCESS_OUT_DIR / "summary.json"
BROAD10_QUALITY_LOSS_CURVE_TOLERANCE = 2.0e-8

EVIDENCE_ORDER = (
    "trainer_matrix",
    "extended_trainer_matrix",
    "frame_scaling_matrix",
    "extended_frame_scaling_diagnostic",
    "frame_count_breadth_diagnostic",
    "quality_tether",
    "extended_quality_tether",
    "broad10_quality_tether",
    "media_tether",
    "extended_media_tether",
    "broad10_media_tether",
    "bq4_fresh_process_timing",
)

Verifier = Callable[[dict[str, Any]], list[str]]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact(path: Path, verifier: Verifier) -> dict[str, Any]:
    report = _load_json(path)
    return {
        "path": str(path),
        "benchmark": report.get("benchmark"),
        "status": report.get("status"),
        "verifier_errors": verifier(report),
        "summary": report.get("summary", {}),
    }


def _finite_float(value: Any, label: str, errors: list[str]) -> float:
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        errors.append(f"{label} must be finite, got {value!r}")
        return 0.0
    return float(value)


def _finite_int(value: Any, label: str, errors: list[str]) -> int:
    if not isinstance(value, int):
        errors.append(f"{label} must be an integer, got {value!r}")
        return 0
    return int(value)


def _summary(report: dict[str, Any], key: str) -> dict[str, Any]:
    return report["evidence"][key]["summary"]


def _bool(summary: dict[str, Any], key: str) -> bool:
    return summary.get(key) is True


def _max_float(summaries: list[dict[str, Any]], key: str) -> float:
    return max(float(summary[key]) for summary in summaries)


def _min_float(summaries: list[dict[str, Any]], key: str) -> float:
    return min(float(summary[key]) for summary in summaries)


def _max_int(summaries: list[dict[str, Any]], key: str) -> int:
    return max(int(summary[key]) for summary in summaries)


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    evidence = report["evidence"]
    trainer = _summary(report, "trainer_matrix")
    extended_trainer = _summary(report, "extended_trainer_matrix")
    frame_scaling = _summary(report, "frame_scaling_matrix")
    extended_frame = _summary(report, "extended_frame_scaling_diagnostic")
    frame_count_breadth = _summary(report, "frame_count_breadth_diagnostic")
    quality = _summary(report, "quality_tether")
    extended_quality = _summary(report, "extended_quality_tether")
    broad10_quality = _summary(report, "broad10_quality_tether")
    media = _summary(report, "media_tether")
    extended_media = _summary(report, "extended_media_tether")
    broad10_media = _summary(report, "broad10_media_tether")
    bq4_fresh_process = _summary(report, "bq4_fresh_process_timing")
    bq4_acceptance = bq4_fresh_process["timing_acceptance"]
    bq4_post_warmup = bq4_acceptance["post_warmup_summary"]
    functional_summaries = [trainer, extended_trainer, frame_scaling, extended_frame]
    quality_summaries = [quality, extended_quality, broad10_quality]
    media_summaries = [media, extended_media, broad10_media]
    all_underlying = all(
        isinstance(evidence.get(key), dict)
        and evidence[key].get("status") == "ok"
        and evidence[key].get("verifier_errors") == []
        and isinstance(evidence[key].get("summary"), dict)
        for key in EVIDENCE_ORDER
    )
    all_functional_rows_pass = (
        all(_bool(summary, "all_source_videos_exist") for summary in functional_summaries)
        and all(_bool(summary, "all_rows_pass") for summary in functional_summaries)
        and all(_bool(summary, "all_rows_no_overflow") for summary in functional_summaries)
        and all(_bool(summary, "all_rows_fallback_free") for summary in functional_summaries)
        and all(_bool(summary, "all_rows_visibility_stratification_free") for summary in functional_summaries)
        and all(_bool(summary, "all_measured_loss_matches_cadence") for summary in functional_summaries)
    )
    all_quality_tethers_match = (
        all(_bool(summary, "all_case_files_exist") for summary in quality_summaries)
        and all(_bool(summary, "all_rows_pass") for summary in quality_summaries)
        and all(_bool(summary, "all_rows_error_free") for summary in quality_summaries)
        and all(_bool(summary, "all_gradient_flags_present") for summary in quality_summaries)
        and all(_bool(summary, "all_measured_loss_curves_match_cadence") for summary in quality_summaries)
        and all(_bool(summary, "all_measured_rgb_loss_curves_match_cadence") for summary in quality_summaries)
        and all(_bool(summary, "all_measured_end_psnr_matches_cadence") for summary in quality_summaries)
        and all(_bool(summary, "all_measured_psnr_improves") for summary in quality_summaries)
    )
    all_media_tethers_match = (
        all(_bool(summary, "all_source_videos_exist") for summary in media_summaries)
        and all(_bool(summary, "all_case_rows_pass") for summary in media_summaries)
        and all(_bool(summary, "all_contact_sheets_exist") for summary in media_summaries)
        and all(_bool(summary, "all_contact_sheet_pixels_match_cadence") for summary in media_summaries)
        and all(_bool(summary, "all_contact_sheet_hashes_match_cadence") for summary in media_summaries)
        and all(_bool(summary, "all_contact_sheet_metrics_match_payload") for summary in media_summaries)
        and all(_bool(summary, "all_loss_curves_match_cadence") for summary in media_summaries)
        and all(_bool(summary, "all_final_full_rgb_losses_match_cadence") for summary in media_summaries)
        and all(_bool(summary, "all_gradient_flags_present") for summary in media_summaries)
        and all(_bool(summary, "all_measured_psnr_improves") for summary in media_summaries)
    )
    max_support_rebins = _max_int(functional_summaries, "max_measured_support_rebins")
    max_stale_refreshes = _max_int(functional_summaries, "max_measured_stale_refreshes")
    rebuild_summaries = functional_summaries + media_summaries
    max_rebuild_ratio = max(
        _max_float(functional_summaries, "max_measured_vs_cadence_rebuild_ratio"),
        _max_float(media_summaries, "max_measured_vs_cadence_rebuild_ratio"),
    )
    return {
        "underlying_report_count": len(EVIDENCE_ORDER),
        "all_underlying_verifiers_pass": all_underlying,
        "all_source_videos_exist": (
            all(_bool(summary, "all_source_videos_exist") for summary in functional_summaries)
            and all(_bool(summary, "all_source_videos_exist") for summary in media_summaries)
        ),
        "functional_scene_count": int(extended_trainer["scene_count"]),
        "functional_distinct_youtube_id_count": int(extended_trainer["distinct_youtube_id_count"]),
        "functional_row_count": int(extended_trainer["row_count"]),
        "all_functional_rows_pass": all_functional_rows_pass,
        "frame_scaling_scene_count": int(frame_scaling["scene_count"]),
        "frame_scaling_frame_count_count": int(frame_scaling["frame_count_count"]),
        "frame_scaling_frame_growth_factor": float(frame_scaling["frame_growth_factor"]),
        "frame_scaling_max_no_first_growth_vs_frame_growth_ratio": float(
            frame_scaling["max_measured_no_first_growth_vs_frame_growth_ratio"]
        ),
        "extended_frame_scaling_scene_count": int(extended_frame["source_scene_count"]),
        "extended_frame_scaling_distinct_youtube_id_count": int(
            extended_frame["source_distinct_youtube_id_count"]
        ),
        "extended_frame_scaling_expected_timing_failure_count": int(extended_frame["strict_failure_count"]),
        "extended_frame_scaling_failed_only_expected_timing": bool(
            extended_frame["strict_failed_only_expected_timing"]
        ),
        "extended_frame_scaling_no_first_timing_win": bool(extended_frame["no_first_timing_win"]),
        "extended_frame_scaling_no_first_growth_sublinear": bool(extended_frame["no_first_growth_sublinear"]),
        "max_extended_timing_growth_overage": float(extended_frame["max_growth_ratio_overage"]),
        "max_extended_no_first_ratio_overage": float(extended_frame["max_no_first_ratio_overage"]),
        "frame_count_breadth_scene_count": int(frame_count_breadth["source_scene_count"]),
        "frame_count_breadth_distinct_youtube_id_count": int(
            frame_count_breadth["source_distinct_youtube_id_count"]
        ),
        "frame_count_breadth_frame_count_count": int(frame_count_breadth["source_frame_count_count"]),
        "frame_count_breadth_strict_failure_count": int(frame_count_breadth["strict_failure_count"]),
        "frame_count_breadth_failed_only_expected_timing": bool(
            frame_count_breadth["strict_failed_only_expected_timing"]
        ),
        "broad_frame_count_count": max(
            int(frame_scaling["frame_count_count"]),
            int(extended_frame["source_frame_count_count"]),
            int(frame_count_breadth["source_frame_count_count"]),
        ),
        "quality_scene_count": int(extended_quality["scene_count"]),
        "broad10_quality_scene_count": int(broad10_quality["scene_count"]),
        "broad10_quality_distinct_youtube_id_count": int(broad10_quality["distinct_youtube_id_count"]),
        "broad_quality_distinct_youtube_id_count": max(
            int(extended_quality.get("source_distinct_youtube_id_count", extended_quality["scene_count"])),
            int(broad10_quality["distinct_youtube_id_count"]),
        ),
        "quality_pair_count": int(extended_quality["pair_count"]),
        "all_quality_tethers_match": all_quality_tethers_match,
        "max_quality_loss_curve_delta": _max_float(quality_summaries + media_summaries, "max_abs_loss_curve_delta"),
        "max_quality_rgb_loss_curve_delta": _max_float(
            quality_summaries + media_summaries,
            "max_abs_rgb_loss_curve_delta",
        ),
        "max_quality_end_psnr_delta": _max_float(quality_summaries, "max_end_psnr_abs_delta"),
        "min_quality_psnr_gain": min(
            _min_float(quality_summaries, "min_measured_psnr_gain"),
            _min_float(media_summaries, "min_measured_psnr_gain"),
        ),
        "media_scene_count": int(extended_media["scene_count"]),
        "broad10_media_scene_count": int(broad10_media["scene_count"]),
        "broad10_media_distinct_youtube_id_count": int(broad10_media["distinct_youtube_id_count"]),
        "broad_media_distinct_youtube_id_count": max(
            int(extended_media["distinct_youtube_id_count"]),
            int(broad10_media["distinct_youtube_id_count"]),
        ),
        "media_pair_count": int(extended_media["pair_count"]),
        "all_media_tethers_match": all_media_tethers_match,
        "max_media_contact_sheet_delta": _max_int(media_summaries, "max_abs_contact_sheet_delta"),
        "max_media_contact_sheet_payload_loss_delta": _max_float(
            media_summaries,
            "max_contact_sheet_payload_loss_abs_delta",
        ),
        "max_media_final_rgb_loss_delta": _max_float(media_summaries, "max_final_full_rgb_loss_abs_delta"),
        "min_media_contact_sheet_target_std": _min_float(media_summaries, "min_contact_sheet_target_std"),
        "min_media_contact_sheet_pred_std": _min_float(media_summaries, "min_contact_sheet_pred_std"),
        "bq4_fresh_process_pair_count": int(bq4_fresh_process["paired_ratio_count"]),
        "bq4_fresh_process_requested_repeat_count": int(bq4_fresh_process["requested_repeat_count"]),
        "bq4_fresh_process_warmup_discard_repeats": int(bq4_fresh_process["warmup_discard_repeats"]),
        "bq4_fresh_process_all_rows_fresh": bool(bq4_fresh_process["all_rows_fresh_process"]),
        "bq4_fresh_process_cache_support_clean": bool(bq4_fresh_process["all_rows_cache_support_clean"]),
        "bq4_fresh_process_timing_acceptance_status": str(bq4_acceptance["status"]),
        "bq4_fresh_process_post_warmup_pair_count": int(bq4_acceptance["post_warmup_pair_count"]),
        "bq4_fresh_process_post_warmup_median_no_first_ratio": float(
            bq4_post_warmup["median_no_first_ratio"]
        ),
        "bq4_fresh_process_post_warmup_median_projective_total_ratio": float(
            bq4_post_warmup["median_projective_total_ratio"]
        ),
        "bq4_fresh_process_post_warmup_median_feature_state_update_ratio": float(
            bq4_post_warmup["median_feature_state_update_ratio"]
        ),
        "bq4_fresh_process_no_first_bump_count": int(
            bq4_fresh_process["all_target_summary"]["no_first_bump_count"]
        ),
        "bq4_fresh_process_projective_total_bump_count": int(
            bq4_fresh_process["all_target_summary"]["projective_total_bump_count"]
        ),
        "bq4_fresh_process_feature_state_update_bump_count": int(
            bq4_fresh_process["all_target_summary"]["feature_state_update_bump_count"]
        ),
        "bq4_fresh_process_max_no_first_ratio": float(
            bq4_fresh_process["all_target_summary"]["max_no_first_ratio"]
        ),
        "bq4_fresh_process_max_projective_total_ratio": float(
            bq4_fresh_process["all_target_summary"]["max_projective_total_ratio"]
        ),
        "bq4_fresh_process_max_feature_state_update_ratio": float(
            bq4_fresh_process["all_target_summary"]["max_feature_state_update_ratio"]
        ),
        "max_support_rebins": max_support_rebins,
        "max_stale_refreshes": max_stale_refreshes,
        "all_support_churn_zero": max_support_rebins == 0 and max_stale_refreshes == 0,
        "max_rebuild_ratio": max_rebuild_ratio,
        "all_rebuild_ratios_at_most_half": max_rebuild_ratio <= 0.5,
        "max_no_first_ratio_any_checked_path": max(
            _max_float(rebuild_summaries, "max_measured_vs_cadence_no_first_step_ms_ratio"),
            float(extended_frame["max_measured_vs_cadence_no_first_step_ms_ratio"]),
        ),
        "strict_timing_win_claimed": False,
        "fresh_process_median_timing_win_claimed": bq4_acceptance["status"] == "pass",
        "does_not_prove_completion": report.get("does_not_prove_completion") is True,
    }


def _expect_true(summary: dict[str, Any], key: str, label: str, errors: list[str]) -> None:
    if summary.get(key) is not True:
        errors.append(f"{label} {key} must be true")


def _check_functional(summary: dict[str, Any], label: str, min_scenes: int, errors: list[str]) -> None:
    if _finite_int(summary.get("scene_count"), f"{label} scene_count", errors) < min_scenes:
        errors.append(f"{label} must cover at least {min_scenes} scenes")
    if _finite_int(summary.get("distinct_youtube_id_count"), f"{label} distinct_youtube_id_count", errors) < min_scenes:
        errors.append(f"{label} must cover at least {min_scenes} source-distinct videos")
    for key in (
        "all_source_videos_exist",
        "all_rows_pass",
        "all_rows_loss_decreased",
        "all_rows_no_overflow",
        "all_rows_fallback_free",
        "all_rows_visibility_stratification_free",
        "all_measured_loss_matches_cadence",
    ):
        _expect_true(summary, key, label, errors)
    if _finite_float(summary.get("max_measured_vs_cadence_end_loss_abs_delta"), f"{label} loss delta", errors) > 1.0e-5:
        errors.append(f"{label} measured/cadence loss delta must stay below 1e-5")
    if _finite_float(summary.get("max_measured_vs_cadence_rebuild_ratio"), f"{label} rebuild ratio", errors) > 0.5:
        errors.append(f"{label} rebuild ratio must stay at or below 0.5")
    if _finite_int(summary.get("max_measured_support_rebins"), f"{label} support rebins", errors) != 0:
        errors.append(f"{label} must have zero support rebins")
    if _finite_int(summary.get("max_measured_stale_refreshes"), f"{label} stale refreshes", errors) != 0:
        errors.append(f"{label} must have zero stale refreshes")


def _check_quality(summary: dict[str, Any], label: str, min_scenes: int, errors: list[str]) -> None:
    loss_curve_tolerance = BROAD10_QUALITY_LOSS_CURVE_TOLERANCE if "broad10" in label else 1.0e-8
    scene_key = "source_scene_count" if "source_scene_count" in summary else "scene_count"
    if _finite_int(summary.get(scene_key), f"{label} scene_count", errors) < min_scenes:
        errors.append(f"{label} must cover at least {min_scenes} scenes")
    for distinct_key in ("source_distinct_youtube_id_count", "distinct_youtube_id_count"):
        if (
            distinct_key in summary
            and _finite_int(summary.get(distinct_key), f"{label} {distinct_key}", errors) < min_scenes
        ):
            errors.append(f"{label} must cover at least {min_scenes} source-distinct videos")
    for key in (
        "all_case_files_exist",
        "all_rows_pass",
        "all_rows_error_free",
        "all_gradient_flags_present",
        "all_measured_loss_curves_match_cadence",
        "all_measured_rgb_loss_curves_match_cadence",
        "all_measured_end_psnr_matches_cadence",
        "all_measured_psnr_improves",
        "all_measured_loss_decreases",
    ):
        _expect_true(summary, key, label, errors)
    if _finite_float(summary.get("max_abs_loss_curve_delta"), f"{label} loss curve delta", errors) > loss_curve_tolerance:
        errors.append(f"{label} loss curves must match cadence")
    if _finite_float(summary.get("max_abs_rgb_loss_curve_delta"), f"{label} rgb loss curve delta", errors) > loss_curve_tolerance:
        errors.append(f"{label} rgb loss curves must match cadence")
    if _finite_float(summary.get("max_end_psnr_abs_delta"), f"{label} end psnr delta", errors) > 1.0e-8:
        errors.append(f"{label} end PSNR must match cadence")
    if _finite_float(summary.get("min_measured_psnr_gain"), f"{label} min psnr gain", errors) <= 0.0:
        errors.append(f"{label} measured PSNR must improve")


def _check_media(summary: dict[str, Any], label: str, min_scenes: int, errors: list[str]) -> None:
    if _finite_int(summary.get("scene_count"), f"{label} scene_count", errors) < min_scenes:
        errors.append(f"{label} must cover at least {min_scenes} scenes")
    if _finite_int(summary.get("distinct_youtube_id_count"), f"{label} distinct_youtube_id_count", errors) < min_scenes:
        errors.append(f"{label} must cover at least {min_scenes} source-distinct videos")
    for key in (
        "all_source_videos_exist",
        "all_case_rows_pass",
        "all_contact_sheets_exist",
        "all_contact_sheet_pixels_match_cadence",
        "all_contact_sheet_hashes_match_cadence",
        "all_contact_sheet_layouts_valid",
        "all_contact_sheet_metrics_match_payload",
        "all_contact_sheet_rows_nontrivial",
        "all_loss_curves_match_cadence",
        "all_rgb_loss_curves_match_cadence",
        "all_final_full_rgb_losses_match_cadence",
        "all_final_full_rgb_psnr_matches_cadence",
        "all_gradient_flags_present",
        "all_measured_loss_decreases",
        "all_measured_psnr_improves",
        "all_rows_no_overflow",
        "all_rows_fallback_free",
        "all_rows_visibility_stratification_free",
    ):
        _expect_true(summary, key, label, errors)
    if _finite_int(summary.get("max_abs_contact_sheet_delta"), f"{label} contact-sheet delta", errors) != 0:
        errors.append(f"{label} contact sheets must pixel-match cadence")
    if (
        _finite_float(summary.get("max_abs_loss_curve_delta"), f"{label} loss curve delta", errors)
        > MEDIA_SCALAR_LOSS_TOLERANCE
    ):
        errors.append(f"{label} loss curves must match cadence")
    if (
        _finite_float(summary.get("max_abs_rgb_loss_curve_delta"), f"{label} rgb loss curve delta", errors)
        > MEDIA_SCALAR_LOSS_TOLERANCE
    ):
        errors.append(f"{label} rgb loss curves must match cadence")
    if (
        _finite_float(summary.get("max_final_full_rgb_loss_abs_delta"), f"{label} final RGB loss delta", errors)
        > MEDIA_SCALAR_LOSS_TOLERANCE
    ):
        errors.append(f"{label} final RGB loss must match cadence")
    if (
        _finite_float(summary.get("max_final_full_rgb_psnr_abs_delta"), f"{label} final RGB PSNR delta", errors)
        > MEDIA_SCALAR_PSNR_TOLERANCE
    ):
        errors.append(f"{label} final RGB PSNR must match cadence")
    if (
        _finite_float(
            summary.get("max_contact_sheet_payload_loss_abs_delta"),
            f"{label} contact-sheet payload delta",
            errors,
        )
        > CONTACT_SHEET_PAYLOAD_LOSS_TOLERANCE
    ):
        errors.append(f"{label} contact-sheet payload loss must match final RGB loss")
    if _finite_float(summary.get("min_contact_sheet_target_std"), f"{label} target std", errors) <= 1.0e-6:
        errors.append(f"{label} target media must be nontrivial")
    if _finite_float(summary.get("min_contact_sheet_pred_std"), f"{label} pred std", errors) <= 1.0e-6:
        errors.append(f"{label} predicted media must be nontrivial")
    if _finite_float(summary.get("max_measured_vs_cadence_rebuild_ratio"), f"{label} rebuild ratio", errors) > 0.5:
        errors.append(f"{label} rebuild ratio must stay at or below 0.5")


def _check_frame_count_breadth(summary: dict[str, Any], errors: list[str]) -> None:
    label = "frame-count breadth diagnostic"
    if _finite_int(summary.get("source_scene_count"), f"{label} source scene count", errors) < 3:
        errors.append(f"{label} must cover at least three scenes")
    if _finite_int(summary.get("source_distinct_youtube_id_count"), f"{label} distinct sources", errors) < 3:
        errors.append(f"{label} must cover at least three source-distinct videos")
    if _finite_int(summary.get("source_frame_count_count"), f"{label} frame-count count", errors) < 4:
        errors.append(f"{label} must cover at least four frame counts")
    if summary.get("frame_count_breadth_accepted") is not True:
        errors.append(f"{label} must accept frame-count breadth")
    if summary.get("strict_failed_only_expected_timing") is not True:
        errors.append(f"{label} must fail only expected timing gates")
    for key in (
        "all_source_videos_exist",
        "all_rows_pass",
        "all_rows_loss_decreased",
        "all_rows_no_overflow",
        "all_rows_fallback_free",
        "all_rows_visibility_stratification_free",
        "all_measured_loss_matches_cadence",
        "no_first_growth_sublinear",
    ):
        _expect_true(summary, key, label, errors)
    if _finite_int(summary.get("max_measured_support_rebins"), f"{label} support rebins", errors) != 0:
        errors.append(f"{label} must have zero support rebins")
    if _finite_int(summary.get("max_measured_stale_refreshes"), f"{label} stale refreshes", errors) != 0:
        errors.append(f"{label} must have zero stale refreshes")
    if _finite_float(summary.get("max_measured_vs_cadence_rebuild_ratio"), f"{label} rebuild ratio", errors) > 0.5:
        errors.append(f"{label} rebuild ratio must stay at or below 0.5")
    if (
        _finite_float(summary.get("max_measured_no_first_growth_vs_frame_growth_ratio"), f"{label} timing growth", errors)
        >= 1.0
    ):
        errors.append(f"{label} measured timing growth must stay below frame growth")


def _check_bq4_fresh_process(summary: dict[str, Any], errors: list[str]) -> None:
    label = "Bq4 fresh-process timing"
    for key in (
        "all_rows_fresh_process",
        "all_expected_global_steps_traced",
        "all_projective_interval_timing_present",
        "all_rows_cache_support_clean",
    ):
        _expect_true(summary, key, label, errors)
    if _finite_int(summary.get("requested_repeat_count"), f"{label} requested repeats", errors) < 3:
        errors.append(f"{label} must have at least three fresh-process repeats")
    if _finite_int(summary.get("warmup_discard_repeats"), f"{label} warmup discard repeats", errors) < 1:
        errors.append(f"{label} must discard at least one warmup repeat")
    if _finite_int(summary.get("paired_ratio_count"), f"{label} paired ratio count", errors) < 6:
        errors.append(f"{label} must contain at least six paired ratios")
    target = summary.get("all_target_summary", {})
    if not isinstance(target, dict):
        errors.append(f"{label} all_target_summary must be an object")
        target = {}
    if _finite_int(target.get("no_first_bump_count"), f"{label} no-first bump count", errors) != 0:
        errors.append(f"{label} must have zero no-first bumps")
    acceptance = summary.get("timing_acceptance", {})
    if not isinstance(acceptance, dict):
        errors.append(f"{label} timing_acceptance must be an object")
        return
    if acceptance.get("status") != "pass":
        errors.append(f"{label} post-warmup median acceptance must pass")
    if acceptance.get("median_ratios_within_threshold") is not True:
        errors.append(f"{label} post-warmup medians must stay within threshold")
    if _finite_int(acceptance.get("post_warmup_pair_count"), f"{label} post-warmup pair count", errors) < 4:
        errors.append(f"{label} must keep at least four post-warmup pairs")
    post_warmup = acceptance.get("post_warmup_summary", {})
    if not isinstance(post_warmup, dict):
        errors.append(f"{label} post_warmup_summary must be an object")
        return
    for key in (
        "median_no_first_ratio",
        "median_projective_total_ratio",
        "median_feature_state_update_ratio",
    ):
        if _finite_float(post_warmup.get(key), f"{label} {key}", errors) > 1.0:
            errors.append(f"{label} {key} must stay at or below cadence")


def verify_real_video_acceptance_envelope_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "ok":
        errors.append(f"status must be ok, got {report.get('status')!r}")
    if report.get("benchmark") != "star_uvt_projective_real_video_acceptance_envelope":
        errors.append(f"unexpected benchmark {report.get('benchmark')!r}")
    theory_contract = report.get("theory_contract")
    if (
        not isinstance(theory_contract, str)
        or "acceptance envelope" not in theory_contract
        or "does not prove broad real-scene quality acceptance" not in theory_contract
        or "does not prove full goal completion" not in theory_contract
        or "five-source" not in theory_contract
        or "broad10 quality" not in theory_contract
        or "broad10 media" not in theory_contract
        or "frame-count breadth" not in theory_contract
    ):
        errors.append("theory_contract must preserve acceptance-envelope non-completion scope")
    if report.get("does_not_prove_completion") is not True:
        errors.append("does_not_prove_completion must remain true")

    evidence = report.get("evidence")
    if not isinstance(evidence, dict):
        errors.append("evidence must be an object")
        return errors
    for key in EVIDENCE_ORDER:
        row = evidence.get(key)
        if not isinstance(row, dict):
            errors.append(f"evidence {key} must be an object")
            continue
        if not isinstance(row.get("path"), str) or not row["path"]:
            errors.append(f"evidence {key} path must be nonempty")
        if row.get("status") != "ok":
            errors.append(f"evidence {key} status must be ok, got {row.get('status')!r}")
        if row.get("verifier_errors"):
            errors.append(f"evidence {key} verifier failed: {row.get('verifier_errors')}")
        if not isinstance(row.get("summary"), dict):
            errors.append(f"evidence {key} summary must be an object")
    if errors:
        return errors

    trainer = _summary(report, "trainer_matrix")
    extended_trainer = _summary(report, "extended_trainer_matrix")
    frame_scaling = _summary(report, "frame_scaling_matrix")
    extended_frame = _summary(report, "extended_frame_scaling_diagnostic")
    frame_count_breadth = _summary(report, "frame_count_breadth_diagnostic")
    quality = _summary(report, "quality_tether")
    extended_quality = _summary(report, "extended_quality_tether")
    broad10_quality = _summary(report, "broad10_quality_tether")
    media = _summary(report, "media_tether")
    extended_media = _summary(report, "extended_media_tether")
    broad10_media = _summary(report, "broad10_media_tether")
    bq4_fresh_process = _summary(report, "bq4_fresh_process_timing")

    _check_functional(trainer, "trainer matrix", 3, errors)
    _check_functional(extended_trainer, "extended trainer matrix", 5, errors)
    _check_functional(frame_scaling, "frame-scaling matrix", 3, errors)
    if _finite_int(frame_scaling.get("frame_count_count"), "frame-scaling frame_count_count", errors) < 3:
        errors.append("frame-scaling matrix must cover at least three frame counts")
    if _finite_float(frame_scaling.get("frame_growth_factor"), "frame-scaling growth factor", errors) < 4.0:
        errors.append("frame-scaling matrix must cover at least 4x frame growth")
    if (
        _finite_float(
            frame_scaling.get("max_measured_no_first_growth_vs_frame_growth_ratio"),
            "frame-scaling growth ratio",
            errors,
        )
        >= 1.0
    ):
        errors.append("frame-scaling matrix must keep measured timing growth below frame growth")

    if extended_frame.get("source_status") != "failed":
        errors.append("extended frame diagnostic must preserve failed strict timing source")
    if extended_frame.get("strict_failed_only_expected_timing") is not True:
        errors.append("extended frame diagnostic must fail only expected strict timing gates")
    if _finite_int(extended_frame.get("strict_failure_count"), "extended frame strict failure count", errors) != 2:
        errors.append("extended frame diagnostic must preserve the two expected timing failures")
    if extended_frame.get("no_first_timing_win") is not False:
        errors.append("extended frame diagnostic must not claim a no-first timing win")
    if extended_frame.get("no_first_growth_sublinear") is not False:
        errors.append("extended frame diagnostic must not claim sublinear no-first growth")
    if _finite_int(extended_frame.get("source_scene_count"), "extended frame source scene count", errors) < 5:
        errors.append("extended frame diagnostic must cover at least five scenes")
    if _finite_int(extended_frame.get("source_distinct_youtube_id_count"), "extended frame distinct source count", errors) < 5:
        errors.append("extended frame diagnostic must cover at least five source-distinct videos")
    for key in (
        "all_source_videos_exist",
        "all_rows_pass",
        "all_rows_no_overflow",
        "all_rows_fallback_free",
        "all_rows_visibility_stratification_free",
        "all_measured_loss_matches_cadence",
    ):
        _expect_true(extended_frame, key, "extended frame diagnostic", errors)
    if _finite_int(extended_frame.get("max_measured_support_rebins"), "extended frame support rebins", errors) != 0:
        errors.append("extended frame diagnostic must have zero support rebins")
    if _finite_int(extended_frame.get("max_measured_stale_refreshes"), "extended frame stale refreshes", errors) != 0:
        errors.append("extended frame diagnostic must have zero stale refreshes")
    _check_frame_count_breadth(frame_count_breadth, errors)

    _check_quality(quality, "quality tether", 3, errors)
    _check_quality(extended_quality, "extended quality tether", 5, errors)
    _check_quality(broad10_quality, "broad10 quality tether", 10, errors)
    _check_media(media, "media tether", 3, errors)
    _check_media(extended_media, "extended media tether", 5, errors)
    _check_media(broad10_media, "broad10 media tether", 10, errors)
    _check_bq4_fresh_process(bq4_fresh_process, errors)

    summary = report.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        return errors
    try:
        expected = summarize(report)
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"summary could not be recomputed: {exc}")
        return errors
    for key, expected_value in expected.items():
        actual = summary.get(key)
        if isinstance(expected_value, float):
            if not isinstance(actual, int | float) or abs(float(actual) - expected_value) > 1.0e-9:
                errors.append(f"summary {key} mismatch: expected {expected_value!r}, got {actual!r}")
        elif actual != expected_value:
            errors.append(f"summary {key} mismatch: expected {expected_value!r}, got {actual!r}")
    if summary.get("all_underlying_verifiers_pass") is not True:
        errors.append("summary all_underlying_verifiers_pass must be true")
    if summary.get("all_functional_rows_pass") is not True:
        errors.append("summary all_functional_rows_pass must be true")
    if summary.get("all_quality_tethers_match") is not True:
        errors.append("summary all_quality_tethers_match must be true")
    if summary.get("all_media_tethers_match") is not True:
        errors.append("summary all_media_tethers_match must be true")
    if summary.get("all_support_churn_zero") is not True:
        errors.append("summary all_support_churn_zero must be true")
    if summary.get("all_rebuild_ratios_at_most_half") is not True:
        errors.append("summary all_rebuild_ratios_at_most_half must be true")
    if summary.get("strict_timing_win_claimed") is not False:
        errors.append("summary strict_timing_win_claimed must remain false")
    if summary.get("fresh_process_median_timing_win_claimed") is not True:
        errors.append("summary fresh_process_median_timing_win_claimed must be true")
    if summary.get("does_not_prove_completion") is not True:
        errors.append("summary does_not_prove_completion must remain true")
    return errors


def assert_real_video_acceptance_envelope_report(report: dict[str, Any]) -> None:
    errors = verify_real_video_acceptance_envelope_report(report)
    if errors:
        raise AssertionError("real-video acceptance envelope failed:\n- " + "\n- ".join(errors))


def run_report(
    *,
    trainer_matrix_report: Path = DEFAULT_TRAINER_MATRIX_REPORT,
    extended_trainer_matrix_report: Path = DEFAULT_EXTENDED_TRAINER_MATRIX_REPORT,
    frame_scaling_matrix_report: Path = DEFAULT_FRAME_SCALING_MATRIX_REPORT,
    extended_frame_scaling_diagnostic_report: Path = DEFAULT_EXTENDED_FRAME_SCALING_DIAGNOSTIC_REPORT,
    frame_count_breadth_diagnostic_report: Path = DEFAULT_FRAME_COUNT_BREADTH_DIAGNOSTIC_REPORT,
    quality_tether_report: Path = DEFAULT_QUALITY_TETHER_REPORT,
    extended_quality_tether_report: Path = DEFAULT_EXTENDED_QUALITY_TETHER_REPORT,
    broad10_quality_tether_report: Path = DEFAULT_BROAD10_QUALITY_TETHER_REPORT,
    media_tether_report: Path = DEFAULT_MEDIA_TETHER_REPORT,
    extended_media_tether_report: Path = DEFAULT_EXTENDED_MEDIA_TETHER_REPORT,
    broad10_media_tether_report: Path = DEFAULT_BROAD10_MEDIA_TETHER_REPORT,
    bq4_trace_fresh_process_report: Path = DEFAULT_BQ4_TRACE_FRESH_PROCESS_REPORT,
) -> dict[str, Any]:
    evidence = {
        "trainer_matrix": _artifact(trainer_matrix_report, verify_real_video_multiscene_trainer_matrix_report),
        "extended_trainer_matrix": _artifact(
            extended_trainer_matrix_report,
            verify_real_video_multiscene_trainer_matrix_report,
        ),
        "frame_scaling_matrix": _artifact(
            frame_scaling_matrix_report,
            verify_real_video_multiscene_frame_scaling_matrix_report,
        ),
        "extended_frame_scaling_diagnostic": _artifact(
            extended_frame_scaling_diagnostic_report,
            verify_extended_frame_scaling_diagnostic_report,
        ),
        "frame_count_breadth_diagnostic": _artifact(
            frame_count_breadth_diagnostic_report,
            verify_frame_count_breadth_diagnostic_report,
        ),
        "quality_tether": _artifact(quality_tether_report, verify_real_video_multiscene_quality_tether_report),
        "extended_quality_tether": _artifact(
            extended_quality_tether_report,
            verify_real_video_multiscene_extended_quality_tether_report,
        ),
        "broad10_quality_tether": _artifact(
            broad10_quality_tether_report,
            verify_real_video_broad10_quality_tether_report,
        ),
        "media_tether": _artifact(media_tether_report, verify_real_video_multiscene_media_tether_report),
        "extended_media_tether": _artifact(
            extended_media_tether_report,
            verify_real_video_multiscene_media_tether_report,
        ),
        "broad10_media_tether": _artifact(
            broad10_media_tether_report,
            verify_real_video_multiscene_media_tether_report,
        ),
        "bq4_fresh_process_timing": _artifact(
            bq4_trace_fresh_process_report,
            verify_bq4_trace_fresh_process_report,
        ),
    }
    report: dict[str, Any] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_real_video_acceptance_envelope",
        "goal": "fast 2D rasters across time from 4D spacetime primitives",
        "meta_goal": "share projection/support/binning/visibility/backward work over time",
        "theory_contract": (
            "This real-video acceptance envelope consolidates source-distinct functional trainer rows, "
            "frame-scaling, frame-count breadth diagnostics, five-source timing diagnostics, cadence quality tethers, broad10 quality "
            "and broad10 media tethering, and actual media "
            "tethers plus a Bq4 fresh-process median timing gate. It does not prove broad real-scene quality acceptance and does not prove full goal "
            "completion; it is a bounded five-source acceptance envelope for the current guarded "
            "projective-interval STAR UVT route."
        ),
        "does_not_prove_completion": True,
        "evidence": evidence,
    }
    report["summary"] = summarize(report)
    errors = verify_real_video_acceptance_envelope_report(report)
    if errors:
        report["status"] = "failed"
        report["errors"] = errors
    return report


def write_report(report: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summary.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--trainer-matrix-report", type=Path, default=DEFAULT_TRAINER_MATRIX_REPORT)
    parser.add_argument(
        "--extended-trainer-matrix-report",
        type=Path,
        default=DEFAULT_EXTENDED_TRAINER_MATRIX_REPORT,
    )
    parser.add_argument("--frame-scaling-matrix-report", type=Path, default=DEFAULT_FRAME_SCALING_MATRIX_REPORT)
    parser.add_argument(
        "--extended-frame-scaling-diagnostic-report",
        type=Path,
        default=DEFAULT_EXTENDED_FRAME_SCALING_DIAGNOSTIC_REPORT,
    )
    parser.add_argument(
        "--frame-count-breadth-diagnostic-report",
        type=Path,
        default=DEFAULT_FRAME_COUNT_BREADTH_DIAGNOSTIC_REPORT,
    )
    parser.add_argument("--quality-tether-report", type=Path, default=DEFAULT_QUALITY_TETHER_REPORT)
    parser.add_argument("--extended-quality-tether-report", type=Path, default=DEFAULT_EXTENDED_QUALITY_TETHER_REPORT)
    parser.add_argument("--broad10-quality-tether-report", type=Path, default=DEFAULT_BROAD10_QUALITY_TETHER_REPORT)
    parser.add_argument("--media-tether-report", type=Path, default=DEFAULT_MEDIA_TETHER_REPORT)
    parser.add_argument("--extended-media-tether-report", type=Path, default=DEFAULT_EXTENDED_MEDIA_TETHER_REPORT)
    parser.add_argument("--broad10-media-tether-report", type=Path, default=DEFAULT_BROAD10_MEDIA_TETHER_REPORT)
    parser.add_argument("--bq4-trace-fresh-process-report", type=Path, default=DEFAULT_BQ4_TRACE_FRESH_PROCESS_REPORT)
    parser.add_argument("--verify-report", type=Path)
    args = parser.parse_args()

    if args.verify_report is not None:
        report = _load_json(args.verify_report)
        assert_real_video_acceptance_envelope_report(report)
        print(f"verified {args.verify_report}")
        return

    report = run_report(
        trainer_matrix_report=args.trainer_matrix_report,
        extended_trainer_matrix_report=args.extended_trainer_matrix_report,
        frame_scaling_matrix_report=args.frame_scaling_matrix_report,
        extended_frame_scaling_diagnostic_report=args.extended_frame_scaling_diagnostic_report,
        frame_count_breadth_diagnostic_report=args.frame_count_breadth_diagnostic_report,
        quality_tether_report=args.quality_tether_report,
        extended_quality_tether_report=args.extended_quality_tether_report,
        broad10_quality_tether_report=args.broad10_quality_tether_report,
        media_tether_report=args.media_tether_report,
        extended_media_tether_report=args.extended_media_tether_report,
        broad10_media_tether_report=args.broad10_media_tether_report,
        bq4_trace_fresh_process_report=args.bq4_trace_fresh_process_report,
    )
    if report.get("status") == "ok":
        assert_real_video_acceptance_envelope_report(report)
    path = write_report(report, args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
