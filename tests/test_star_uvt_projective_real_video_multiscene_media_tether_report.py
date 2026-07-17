from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research_experiments.star_uvt_feature_tubes.projective_real_video_multiscene_media_tether_report import (
    DEFAULT_OUT_DIR,
    MEDIA_SCALAR_LOSS_TOLERANCE,
    MEDIA_SCALAR_PSNR_TOLERANCE,
    assert_real_video_multiscene_media_tether_report,
    summarize,
    verify_real_video_multiscene_media_tether_report,
)


def _case_row(scene_id: str, policy: str) -> dict[str, object]:
    return {
        "scene_id": scene_id,
        "youtube_id": f"{scene_id}_yt",
        "title": scene_id,
        "frames": 8,
        "policy": policy,
        "elapsed_sec": 1.0,
        "pass": True,
        "steps": 4,
        "start_loss": 0.30,
        "end_loss": 0.20,
        "loss_decreased": True,
        "mean_step_ms": 10.0,
        "no_first_step_ms": 8.0 if policy == "measured" else 10.0,
        "mean_render_forward_ms": 3.0,
        "mean_backward_ms": 4.0,
        "projective_interval_cache_rebuilds": 2 if policy == "measured" else 4,
        "projective_interval_cache_live_updates": 3 if policy == "measured" else 1,
        "projective_interval_cache_staleness_checks": 3,
        "projective_interval_cache_stale_refreshes": 0,
        "projective_interval_cache_support_rebins": 0,
        "projective_interval_cache_visibility_stratifications": 0,
        "projective_interval_cache_fallback_marks": 0,
        "projective_interval_cache_alpha_renders": 1,
        "tile_overflow_sum": 0,
        "max_tile_count": 4,
        "case_json": f"cases/{scene_id}_{policy}.json",
        "contact_sheet": f"media/{scene_id}_{policy}.png",
        "contact_sheet_exists": True,
        "contact_sheet_bytes": 100,
        "contact_sheet_sha256": "samehash",
        "contact_sheet_mode": "linspace",
        "contact_sheet_frames": 8,
        "contact_sheet_layout_valid": True,
        "contact_sheet_layout_error": "",
        "contact_sheet_inferred_frame_count": 8,
        "contact_sheet_inferred_frame_height": 64,
        "contact_sheet_inferred_frame_width": 64,
        "contact_sheet_target_std": 0.20,
        "contact_sheet_pred_std": 0.07,
        "contact_sheet_target_mean": 0.50,
        "contact_sheet_pred_mean": 0.08,
        "contact_sheet_target_pred_mse": 0.201,
        "contact_sheet_target_pred_psnr": 6.96,
        "contact_sheet_payload_loss_abs_delta": 0.001,
        "media_render_ms": 12.0,
        "final_full_rgb_loss": 0.20,
        "final_full_rgb_psnr": 7.0,
        "losses": [0.30, 0.25, 0.22, 0.20],
        "rgb_losses": [0.30, 0.25, 0.22, 0.20],
        "start_psnr": 5.0,
        "end_psnr": 7.0,
        "gradient_flags": {
            "center_uv_grad_seen": True,
            "center_t_grad_seen": True,
            "velocity_uv_grad_seen": True,
            "raw_feature_grad_seen": True,
            "raw_opacity_grad_seen": True,
            "raw_precision_grad_seen": True,
            "colorizer_grad_seen": True,
        },
    }


def _pair_row(scene_id: str) -> dict[str, object]:
    return {
        "scene_id": scene_id,
        "frames": 8,
        "cadence_case_json": f"cases/{scene_id}_cadence.json",
        "measured_case_json": f"cases/{scene_id}_measured.json",
        "cadence_contact_sheet": f"media/{scene_id}_cadence.png",
        "measured_contact_sheet": f"media/{scene_id}_measured.png",
        "contact_sheet_files_exist": True,
        "contact_sheet_shape": [130, 526, 3],
        "contact_sheet_pixel_count": 205140,
        "max_abs_contact_sheet_delta": 0,
        "mean_abs_contact_sheet_delta": 0.0,
        "contact_sheet_sha256_match": True,
        "cadence_contact_sheet_sha256": "samehash",
        "measured_contact_sheet_sha256": "samehash",
        "cadence_contact_sheet_layout_valid": True,
        "measured_contact_sheet_layout_valid": True,
        "cadence_contact_sheet_target_pred_mse": 0.201,
        "measured_contact_sheet_target_pred_mse": 0.201,
        "contact_sheet_target_pred_mse_abs_delta": 0.0,
        "cadence_contact_sheet_payload_loss_abs_delta": 0.001,
        "measured_contact_sheet_payload_loss_abs_delta": 0.001,
        "max_contact_sheet_payload_loss_abs_delta": 0.001,
        "curve_length": 4,
        "max_abs_loss_curve_delta": 0.0,
        "max_abs_rgb_loss_curve_delta": 0.0,
        "final_full_rgb_loss_abs_delta": 0.0,
        "final_full_rgb_psnr_abs_delta": 0.0,
        "measured_loss_decrease": 0.10,
        "measured_psnr_gain": 2.0,
        "cadence_pass": True,
        "measured_pass": True,
        "cadence_media_render_ms": 12.0,
        "measured_media_render_ms": 12.0,
        "cadence_cache_rebuilds": 4,
        "measured_cache_rebuilds": 2,
        "cadence_no_first_step_ms": 10.0,
        "measured_no_first_step_ms": 8.0,
        "cadence_tile_overflow_sum": 0,
        "measured_tile_overflow_sum": 0,
        "cadence_fallback_marks": 0,
        "measured_fallback_marks": 0,
        "cadence_visibility_stratifications": 0,
        "measured_visibility_stratifications": 0,
        "missing_gradient_flags": [],
        "row_errors": [],
    }


def _valid_report() -> dict[str, object]:
    scene_ids = ("walk_seg_000", "bike_seg_000", "forest_seg_000")
    report: dict[str, object] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_real_video_multiscene_media_tether",
        "base_domain": "checked-in source-distinct real-video segments through actual contact-sheet media writer",
        "theory_contract": (
            "This report does not prove broad real-scene quality acceptance. It tethers the measured live-cache "
            "projective-interval path to the cadence full-rebuild reference through the actual contact-sheet media "
            "writer."
        ),
        "segments_manifest": "segments.jsonl",
        "frames": 8,
        "size": 64,
        "steps": 4,
        "refresh_every": 2,
        "tile_capacity": 128,
        "tube_count": 128,
        "support_guard_padding": 1.0,
        "support_guard_policy": "slack_budgeted",
        "support_guard_bisect_steps": 8,
        "support_stale_overshoot_epsilon": 0.0,
        "support_stale_tail_alpha_epsilon": 0.001,
        "contact_sheet_mode": "linspace",
        "required_gradient_flags": ["center_uv_grad_seen"],
        "scenes": [
            {
                "scene_id": scene_id,
                "youtube_id": f"{scene_id}_yt",
                "title": scene_id,
                "video_path": f"data/{scene_id}.mp4",
                "source_video_exists": True,
                "motion_score": 1.0,
                "scene_cut_count_in_source": 0,
            }
            for scene_id in scene_ids
        ],
        "case_rows": [
            _case_row(scene_id, policy)
            for scene_id in scene_ids
            for policy in ("cadence", "measured")
        ],
        "rows": [_pair_row(scene_id) for scene_id in scene_ids],
    }
    report["summary"] = summarize(report)  # type: ignore[arg-type]
    return report


def test_media_tether_accepts_valid_payload() -> None:
    report = _valid_report()

    assert verify_real_video_multiscene_media_tether_report(report) == []
    assert_real_video_multiscene_media_tether_report(report)
    assert report["summary"]["pair_count"] == 3  # type: ignore[index]


def test_media_tether_rejects_contact_sheet_delta() -> None:
    report = _valid_report()
    report["rows"][0]["max_abs_contact_sheet_delta"] = 1  # type: ignore[index]
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_real_video_multiscene_media_tether_report(report)

    assert any("pixel-match cadence" in error for error in errors)


def test_media_tether_rejects_missing_media() -> None:
    report = _valid_report()
    report["case_rows"][1]["contact_sheet_exists"] = False  # type: ignore[index]
    report["case_rows"][1]["contact_sheet_bytes"] = 0  # type: ignore[index]
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_real_video_multiscene_media_tether_report(report)

    assert any("contact sheet must exist" in error for error in errors)


def test_media_tether_rejects_invalid_contact_sheet_layout() -> None:
    report = _valid_report()
    report["case_rows"][0]["contact_sheet_layout_valid"] = False  # type: ignore[index]
    report["case_rows"][0]["contact_sheet_layout_error"] = "bad row geometry"  # type: ignore[index]
    report["rows"][0]["cadence_contact_sheet_layout_valid"] = False  # type: ignore[index]
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_real_video_multiscene_media_tether_report(report)

    assert any("contact sheet layout must be valid" in error for error in errors)


def test_media_tether_rejects_blank_contact_sheet_row() -> None:
    report = _valid_report()
    report["case_rows"][2]["contact_sheet_pred_std"] = 0.0  # type: ignore[index]
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_real_video_multiscene_media_tether_report(report)

    assert any("pred row must be nontrivial" in error for error in errors)


def test_media_tether_rejects_contact_sheet_payload_loss_delta() -> None:
    report = _valid_report()
    report["case_rows"][3]["contact_sheet_payload_loss_abs_delta"] = 0.01  # type: ignore[index]
    report["rows"][1]["max_contact_sheet_payload_loss_abs_delta"] = 0.01  # type: ignore[index]
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_real_video_multiscene_media_tether_report(report)

    assert any("contact-sheet pixel MSE must match payload final RGB loss" in error for error in errors)


def test_media_tether_rejects_final_rgb_delta() -> None:
    report = _valid_report()
    report["rows"][1]["final_full_rgb_loss_abs_delta"] = 1.0e-4  # type: ignore[index]
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_real_video_multiscene_media_tether_report(report)

    assert any("final media RGB loss must match cadence" in error for error in errors)


def test_media_tether_accepts_scalar_float_tick_with_pixel_exact_media() -> None:
    report = _valid_report()
    report["rows"][0]["max_abs_loss_curve_delta"] = MEDIA_SCALAR_LOSS_TOLERANCE  # type: ignore[index]
    report["rows"][0]["max_abs_rgb_loss_curve_delta"] = MEDIA_SCALAR_LOSS_TOLERANCE  # type: ignore[index]
    report["rows"][0]["final_full_rgb_loss_abs_delta"] = MEDIA_SCALAR_LOSS_TOLERANCE  # type: ignore[index]
    report["rows"][0]["final_full_rgb_psnr_abs_delta"] = MEDIA_SCALAR_PSNR_TOLERANCE  # type: ignore[index]
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    assert verify_real_video_multiscene_media_tether_report(report) == []
    assert report["summary"]["all_contact_sheet_pixels_match_cadence"] is True  # type: ignore[index]
    assert report["summary"]["all_final_full_rgb_losses_match_cadence"] is True  # type: ignore[index]


def test_media_tether_rejects_stale_summary() -> None:
    report = copy.deepcopy(_valid_report())
    report["rows"][2]["max_abs_loss_curve_delta"] = 1.0e-4  # type: ignore[index]

    errors = verify_real_video_multiscene_media_tether_report(report)

    assert any("summary" in error and "mismatch" in error for error in errors)


def test_saved_real_video_multiscene_media_tether_artifact_satisfies_contract() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(Path(summary_json).read_text(encoding="utf-8"))

    assert_real_video_multiscene_media_tether_report(report)
