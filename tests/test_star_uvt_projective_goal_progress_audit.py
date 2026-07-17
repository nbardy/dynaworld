from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research_experiments.star_uvt_feature_tubes.projective_goal_progress_audit import (
    DEFAULT_OUT_DIR,
    OPEN_REQUIREMENT_ID,
    PROVEN_REQUIREMENT_IDS,
    assert_projective_goal_progress_current_acceptance,
    assert_projective_goal_progress_audit,
    run_report,
    summarize,
    verify_projective_goal_progress_current_acceptance,
    verify_projective_goal_progress_audit,
)


def _requirement(requirement_id: str, status: str) -> dict[str, object]:
    row: dict[str, object] = {
        "id": requirement_id,
        "status": status,
        "statement": f"{requirement_id} statement",
        "evidence": ["artifact"],
    }
    if requirement_id == OPEN_REQUIREMENT_ID:
        row["gaps"] = [
            "final completion audit has not promoted the active all-night goal",
            "top-level goal remains deliberately in-progress until the user asks for completion",
        ]
    return row


def _valid_report() -> dict[str, object]:
    evidence = {
        "bundle_invariance": {
            "path": "bundle_invariance.json",
            "benchmark": "star_uvt_projective_bundle_gauge_invariance",
            "status": "ok",
            "verifier_errors": [],
            "summary": {
                "max_rel_error": 3.5e-13,
                "min_bad_no_jacobian_rel_error": 0.6,
            },
        },
        "bundle_gradient": {
            "path": "bundle_gradient.json",
            "benchmark": "star_uvt_projective_bundle_gauge_gradient",
            "status": "ok",
            "verifier_errors": [],
            "summary": {
                "max_gradient_rel_error": 2.3e-12,
                "finite_difference_mean_x_rel_error": 1.4e-10,
            },
        },
        "camera_family": {
            "path": "camera_family.json",
            "benchmark": "star_uvt_projective_camera_family_gauge",
            "status": "ok",
            "verifier_errors": [],
            "summary": {
                "max_value_rel_error": 3.5e-13,
                "max_primitive_gradient_rel_error": 2.3e-12,
                "q_gradient_rel_error": 2.0e-12,
                "q_finite_difference_rel_error": 1.5e-10,
            },
        },
        "camera_family_2d": {
            "path": "camera_family_2d.json",
            "benchmark": "star_uvt_projective_camera_family_2d_gauge",
            "status": "ok",
            "verifier_errors": [],
            "summary": {
                "max_value_rel_error": 8.4e-14,
                "max_primitive_gradient_rel_error": 2.3e-12,
                "q_phase_gradient_rel_error": 1.9e-11,
                "q_phase_finite_difference_rel_error": 2.4e-10,
                "q_height_gradient_rel_error": 1.2e-11,
                "q_height_finite_difference_rel_error": 3.3e-10,
            },
        },
        "camera_family_2d_metal_lowering": {
            "path": "camera_family_2d_metal_lowering.json",
            "benchmark": "star_uvt_projective_camera_family_2d_metal_lowering",
            "status": "ok",
            "verifier_errors": [],
            "summary": {
                "family_to_replay_payload_ratio": 0.178,
                "peak_slice_to_replay_payload_ratio": 0.04,
                "min_grad_coeff_abs_sum": 3.02,
            },
        },
        "camera_family_2d_metal_chain_rule": {
            "path": "camera_family_2d_metal_chain_rule.json",
            "benchmark": "star_uvt_projective_camera_family_2d_metal_chain_rule",
            "status": "ok",
            "verifier_errors": [],
            "summary": {
                "shared_to_replay_gradient_payload_ratio": 0.24,
                "max_finite_difference_rel_error": 4.9e-5,
                "shared_family_grad_abs_sum": 91.0,
            },
        },
        "camera_family_2d_materialized_batch": {
            "path": "camera_family_2d_materialized_batch.json",
            "benchmark": "star_uvt_projective_camera_family_2d_materialized_batch",
            "status": "ok",
            "verifier_errors": [],
            "summary": {
                "forward_launch_ratio": 0.04,
                "backward_launch_ratio": 0.04,
                "materialized_to_replay_trace_payload_ratio": 1.0,
                "family_to_materialized_trace_payload_ratio": 0.178,
                "max_batched_vs_slice_image_abs_error": 0.0,
                "max_batched_vs_slice_shared_grad_rel_error": 9.4e-8,
            },
        },
        "camera_family_2d_native_eval": {
            "path": "camera_family_2d_native_eval.json",
            "benchmark": "star_uvt_projective_camera_family_2d_native_eval",
            "status": "ok",
            "verifier_errors": [],
            "summary": {
                "family_coeff_to_materialized_coeff_payload_ratio": 0.24,
                "family_plus_q_basis_to_materialized_coeff_payload_ratio": 0.573,
                "native_eval_max_rel_error": 6.6e-8,
                "native_grad_family_max_rel_error": 5.8e-8,
                "native_grad_q_basis_max_rel_error": 2.6e-7,
            },
        },
        "camera_family_2d_native_interval_forward": {
            "path": "camera_family_2d_native_interval_forward.json",
            "benchmark": "star_uvt_projective_camera_family_2d_native_interval_forward",
            "status": "ok",
            "verifier_errors": [],
            "summary": {
                "family_forward_to_materialized_trace_payload_ratio": 0.446,
                "family_coeff_to_materialized_trace_payload_ratio": 0.166,
                "native_family_forward_max_rel_error": 0.0,
                "native_family_image_abs_sum": 1992.0,
            },
        },
        "camera_family_2d_native_interval_backward": {
            "path": "camera_family_2d_native_interval_backward.json",
            "benchmark": "star_uvt_projective_camera_family_2d_native_interval_backward",
            "status": "ok",
            "verifier_errors": [],
            "summary": {
                "native_family_gradient_to_materialized_gradient_payload_ratio": 0.293,
                "native_family_coeff_gradient_to_materialized_gradient_payload_ratio": 0.114,
                "native_family_interval_backward_max_family_grad_rel_error": 2.4e-6,
                "native_family_interval_backward_max_q_basis_grad_rel_error": 9.0e-7,
                "native_family_grad_abs_sum": 91.0,
                "native_q_basis_grad_abs_sum": 136.0,
            },
        },
        "camera_family_2d_tile_order_reuse": {
            "path": "camera_family_2d_tile_order_reuse.json",
            "benchmark": "star_uvt_projective_camera_family_2d_tile_order_reuse",
            "status": "ok",
            "verifier_errors": [],
            "summary": {
                "shared_to_materialized_tile_order_metadata_ratio": 0.117,
                "materialized_tile_order_metadata_growth": 25.0,
                "shared_tile_order_metadata_growth": 1.0,
                "expanded_topology_matches_materialized": True,
                "stable_union_depth_order": True,
                "min_union_depth_order_gap": 0.603,
            },
        },
        "camera_family_2d_tile_order_strata": {
            "path": "camera_family_2d_tile_order_strata.json",
            "benchmark": "star_uvt_projective_camera_family_2d_tile_order_strata",
            "status": "ok",
            "verifier_errors": [],
            "summary": {
                "shared_to_materialized_tile_order_metadata_ratio": 0.157,
                "materialized_tile_order_metadata_growth": 25.0,
                "shared_tile_order_metadata_growth": 2.0,
                "order_stratum_count": 2,
                "expanded_topology_matches_materialized": True,
                "all_strata_depth_order_stable": True,
                "min_stratum_union_depth_order_gap": 0.332,
            },
        },
        "camera_family_2d_active_set_strata": {
            "path": "camera_family_2d_active_set_strata.json",
            "benchmark": "star_uvt_projective_camera_family_2d_active_set_strata",
            "status": "ok",
            "verifier_errors": [],
            "summary": {
                "shared_to_materialized_tile_order_metadata_ratio": 0.197,
                "materialized_tile_order_metadata_growth": 25.0,
                "shared_tile_order_metadata_growth": 3.0,
                "active_set_stratum_count": 3,
                "expanded_topology_matches_materialized": True,
                "all_active_set_strata_depth_order_stable": True,
                "min_active_set_union_depth_order_gap": 0.263,
            },
        },
        "real_active_set_distribution": {
            "path": "real_active_set_distribution.json",
            "benchmark": "star_uvt_projective_real_active_set_distribution",
            "status": "ok",
            "verifier_errors": [],
            "summary": {
                "artifact_count": 3,
                "row_count": 9,
                "all_underlying_verifiers_pass": True,
                "all_source_videos_exist": True,
                "all_fallback_free": True,
                "max_active_set_group_to_dense_tile_pair_ratio": 0.041,
                "max_cells_per_active_set_group": 3,
                "max_cell_to_active_set_group_ratio": 1.32,
            },
        },
        "camera_family_shared_work": {
            "path": "camera_family_shared_work.json",
            "benchmark": "star_uvt_projective_camera_family_shared_work_scaling",
            "status": "ok",
            "verifier_errors": [],
            "summary": {
                "final_payload_ratio": 0.106,
                "final_chart_ratio": 0.0625,
                "family_payload_growth": 1.0,
                "per_q_replay_payload_growth": 16.0,
                "max_family_fit_uv_error_px": 0.306,
            },
        },
        "camera_family_2d_shared_work": {
            "path": "camera_family_2d_shared_work.json",
            "benchmark": "star_uvt_projective_camera_family_2d_shared_work_scaling",
            "status": "ok",
            "verifier_errors": [],
            "summary": {
                "final_payload_ratio": 0.0625,
                "final_chart_ratio": 0.015625,
                "family_payload_growth": 1.0,
                "per_q_replay_payload_growth": 64.0,
                "max_family_fit_uv_error_px": 0.111,
            },
        },
        "trainer_interval": {
            "path": "trainer_interval.json",
            "benchmark": "star_uvt_projective_interval_trainer_frame_scaling",
            "status": "ok",
            "verifier_errors": [],
            "summary": {
                "measured_all_pass": True,
                "measured_all_no_overflow": True,
                "all_measured_loss_matches_cadence": True,
                "max_measured_vs_cadence_end_loss_abs_delta": 2.0e-8,
                "measured_vs_cadence_no_first_step_ms_ratios": [0.84, 0.55, 0.74],
                "measured_vs_cadence_rebuild_ratios": [0.5, 0.5, 0.5],
            },
        },
        "trainer_real_video": {
            "path": "trainer_real_video.json",
            "benchmark": "star_uvt_projective_real_video_trainer_frame_scaling",
            "status": "ok",
            "verifier_errors": [],
            "summary": {
                "measured_all_pass": True,
                "measured_all_no_overflow": True,
                "all_measured_loss_matches_cadence": True,
                "max_measured_vs_cadence_end_loss_abs_delta": 0.0,
                "measured_vs_cadence_no_first_step_ms_ratios": [0.88, 0.35, 0.69],
                "measured_vs_cadence_rebuild_ratios": [0.5, 0.5, 0.5],
            },
        },
        "real_video_guarded_support_matrix": {
            "path": "real_video_guarded_support_matrix.json",
            "benchmark": "star_uvt_projective_real_video_guarded_support_matrix",
            "status": "ok",
            "verifier_errors": [],
            "summary": {
                "artifact_count": 5,
                "guarded_artifact_count": 4,
                "measured_row_count": 15,
                "guarded_measured_row_count": 12,
                "all_underlying_verifiers_pass": True,
                "all_guarded_support_verifiers_pass": True,
                "all_source_videos_exist": True,
                "min_guard_padding": 0.25,
                "max_guard_padding": 2.0,
                "all_guarded_loss_matches_cadence": True,
                "all_guarded_no_overflow": True,
                "all_guarded_fallback_free": True,
                "default_measured_support_rebins": 9,
                "guarded_measured_support_rebins": 0,
                "guarded_measured_stale_refreshes": 0,
                "guarded_measured_fallback_marks": 0,
                "max_guarded_measured_tail_alpha_bound": 0.0,
                "max_guarded_measured_overshoot_px": 0.0,
                "max_guarded_measured_no_first_ratio": 0.59,
                "max_guarded_measured_rebuild_ratio": 0.5,
                "max_guarded_measured_loss_delta": 0.0,
                "max_guarded_tile_count": 18,
                "max_guarded_effective_support_uv_padding": 10.0,
            },
        },
        "real_video_multiscene_trainer_matrix": {
            "path": "real_video_multiscene_trainer_matrix.json",
            "benchmark": "star_uvt_projective_real_video_multiscene_trainer_matrix",
            "status": "ok",
            "verifier_errors": [],
            "summary": {
                "scene_count": 3,
                "row_count": 6,
                "measured_row_count": 3,
                "distinct_youtube_id_count": 3,
                "all_source_videos_exist": True,
                "all_rows_pass": True,
                "all_rows_loss_decreased": True,
                "all_rows_no_overflow": True,
                "all_rows_fallback_free": True,
                "all_rows_visibility_stratification_free": True,
                "all_measured_loss_matches_cadence": True,
                "max_measured_vs_cadence_end_loss_abs_delta": 7.5e-9,
                "measured_vs_cadence_no_first_step_ms_ratios": [0.50, 0.51, 0.53],
                "measured_vs_cadence_rebuild_ratios": [0.5, 0.5, 0.5],
                "max_measured_vs_cadence_no_first_step_ms_ratio": 0.53,
                "max_measured_vs_cadence_rebuild_ratio": 0.5,
                "measured_support_rebins": [0, 0, 0],
                "measured_stale_refreshes": [0, 0, 0],
                "measured_live_updates": [3, 3, 3],
                "measured_staleness_checks": [3, 3, 3],
                "measured_cache_rebuilds": [1, 1, 1],
                "cadence_cache_rebuilds": [2, 2, 2],
                "max_measured_support_rebins": 0,
                "max_measured_stale_refreshes": 0,
                "max_measured_support_tail_alpha_bound": 0.0,
                "max_measured_support_overshoot_px": 0.0,
                "max_tile_count": 22,
                "min_motion_score": 0.57,
                "max_motion_score": 5.86,
            },
        },
        "real_video_multiscene_extended_trainer_matrix": {
            "path": "real_video_multiscene_trainer_matrix_extended5.json",
            "benchmark": "star_uvt_projective_real_video_multiscene_trainer_matrix",
            "status": "ok",
            "verifier_errors": [],
            "summary": {
                "scene_count": 5,
                "row_count": 10,
                "measured_row_count": 5,
                "distinct_youtube_id_count": 5,
                "all_source_videos_exist": True,
                "all_rows_pass": True,
                "all_rows_loss_decreased": True,
                "all_rows_no_overflow": True,
                "all_rows_fallback_free": True,
                "all_rows_visibility_stratification_free": True,
                "all_measured_loss_matches_cadence": True,
                "max_measured_vs_cadence_end_loss_abs_delta": 0.0,
                "measured_vs_cadence_no_first_step_ms_ratios": [0.36, 1.51, 0.20, 0.63, 0.55],
                "measured_vs_cadence_rebuild_ratios": [0.5, 0.5, 0.5, 0.5, 0.5],
                "max_measured_vs_cadence_no_first_step_ms_ratio": 1.51,
                "max_measured_vs_cadence_rebuild_ratio": 0.5,
                "measured_support_rebins": [0, 0, 0, 0, 0],
                "measured_stale_refreshes": [0, 0, 0, 0, 0],
                "measured_live_updates": [3, 3, 3, 3, 3],
                "measured_staleness_checks": [3, 3, 3, 3, 3],
                "measured_cache_rebuilds": [1, 1, 1, 1, 1],
                "cadence_cache_rebuilds": [2, 2, 2, 2, 2],
                "max_measured_support_rebins": 0,
                "max_measured_stale_refreshes": 0,
                "max_measured_support_tail_alpha_bound": 0.0,
                "max_measured_support_overshoot_px": 0.0,
                "max_tile_count": 22,
                "min_motion_score": 0.57,
                "max_motion_score": 7.02,
            },
        },
        "real_video_multiscene_frame_scaling_matrix": {
            "path": "real_video_multiscene_frame_scaling_matrix.json",
            "benchmark": "star_uvt_projective_real_video_multiscene_frame_scaling_matrix",
            "status": "ok",
            "verifier_errors": [],
            "summary": {
                "scene_count": 3,
                "frame_count_count": 3,
                "row_count": 18,
                "measured_row_count": 9,
                "distinct_youtube_id_count": 3,
                "frame_growth_factor": 4.0,
                "all_source_videos_exist": True,
                "all_rows_pass": True,
                "all_rows_loss_decreased": True,
                "all_rows_no_overflow": True,
                "all_rows_fallback_free": True,
                "all_rows_visibility_stratification_free": True,
                "all_measured_loss_matches_cadence": True,
                "max_measured_vs_cadence_end_loss_abs_delta": 0.0,
                "measured_vs_cadence_no_first_step_ms_ratios": [0.50, 0.60, 0.70],
                "max_measured_vs_cadence_no_first_step_ms_ratio": 0.70,
                "measured_vs_cadence_rebuild_ratios": [0.5] * 9,
                "max_measured_vs_cadence_rebuild_ratio": 0.5,
                "measured_no_first_growth_vs_frame_growth_ratios": [0.40, 0.41, 0.42],
                "max_measured_no_first_growth_vs_frame_growth_ratio": 0.42,
                "measured_cache_rebuild_growths": [1.0, 1.0, 1.0],
                "max_measured_cache_rebuild_growth": 1.0,
                "measured_support_rebins": [0] * 9,
                "measured_stale_refreshes": [0] * 9,
                "max_measured_support_rebins": 0,
                "max_measured_stale_refreshes": 0,
                "max_measured_support_tail_alpha_bound": 0.0,
                "max_measured_support_overshoot_px": 0.0,
                "max_tile_count": 22,
                "min_motion_score": 0.57,
                "max_motion_score": 5.86,
            },
        },
        "real_video_multiscene_extended_frame_scaling_diagnostic": {
            "path": "real_video_multiscene_extended_frame_scaling_diagnostic.json",
            "benchmark": "star_uvt_projective_real_video_multiscene_extended_frame_scaling_diagnostic",
            "status": "ok",
            "verifier_errors": [],
            "summary": {
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
                "max_motion_score": 7.02,
                "max_tile_count": 22,
                "max_measured_vs_cadence_no_first_step_ms_ratio": 1.18,
                "max_measured_no_first_growth_vs_frame_growth_ratio": 1.001,
                "no_first_timing_win": False,
                "no_first_growth_sublinear": False,
                "max_no_first_ratio_overage": 0.18,
                "max_growth_ratio_overage": 0.001,
            },
        },
        "real_video_multiscene_quality_tether": {
            "path": "real_video_multiscene_quality_tether.json",
            "benchmark": "star_uvt_projective_real_video_multiscene_quality_tether",
            "status": "ok",
            "verifier_errors": [],
            "summary": {
                "scene_count": 3,
                "frame_count_count": 3,
                "pair_count": 9,
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
                "min_measured_psnr_gain": 0.02,
                "min_measured_loss_decrease": 0.001,
                "min_measured_end_psnr": 4.7,
                "max_measured_end_loss": 0.34,
            },
        },
        "real_video_multiscene_extended_quality_tether": {
            "path": "real_video_multiscene_extended_quality_tether.json",
            "benchmark": "star_uvt_projective_real_video_multiscene_extended_quality_tether",
            "status": "ok",
            "verifier_errors": [],
            "summary": {
                "source_scene_count": 5,
                "source_distinct_youtube_id_count": 5,
                "scene_count": 5,
                "pair_count": 5,
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
                "min_measured_psnr_gain": 0.044,
                "min_measured_loss_decrease": 0.001,
                "min_measured_end_psnr": 5.2,
                "max_measured_end_loss": 0.31,
            },
        },
        "real_video_multiscene_media_tether": {
            "path": "real_video_multiscene_media_tether.json",
            "benchmark": "star_uvt_projective_real_video_multiscene_media_tether",
            "status": "ok",
            "verifier_errors": [],
            "summary": {
                "scene_count": 3,
                "pair_count": 3,
                "case_row_count": 6,
                "measured_row_count": 3,
                "cadence_row_count": 3,
                "distinct_youtube_id_count": 3,
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
                "min_contact_sheet_pixel_count": 205140,
                "min_contact_sheet_target_std": 0.14,
                "min_contact_sheet_pred_std": 0.07,
                "min_contact_sheet_target_pred_mse": 0.098,
                "max_measured_vs_cadence_no_first_step_ms_ratio": 0.49,
                "max_measured_vs_cadence_rebuild_ratio": 0.5,
            },
        },
        "real_video_multiscene_extended_media_tether": {
            "path": "real_video_multiscene_extended_media_tether.json",
            "benchmark": "star_uvt_projective_real_video_multiscene_media_tether",
            "status": "ok",
            "verifier_errors": [],
            "summary": {
                "scene_count": 5,
                "pair_count": 5,
                "case_row_count": 10,
                "measured_row_count": 5,
                "cadence_row_count": 5,
                "distinct_youtube_id_count": 5,
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
                "min_contact_sheet_pixel_count": 205140,
                "min_contact_sheet_target_std": 0.14,
                "min_contact_sheet_pred_std": 0.07,
                "min_contact_sheet_target_pred_mse": 0.098,
                "max_measured_vs_cadence_no_first_step_ms_ratio": 0.99,
                "max_measured_vs_cadence_rebuild_ratio": 0.5,
            },
        },
        "real_video_acceptance_envelope": {
            "path": "real_video_acceptance_envelope.json",
            "benchmark": "star_uvt_projective_real_video_acceptance_envelope",
            "status": "ok",
            "verifier_errors": [],
            "summary": {
                "underlying_report_count": 11,
                "all_underlying_verifiers_pass": True,
                "all_source_videos_exist": True,
                "functional_scene_count": 5,
                "functional_distinct_youtube_id_count": 5,
                "functional_row_count": 10,
                "all_functional_rows_pass": True,
                "frame_scaling_scene_count": 3,
                "frame_scaling_frame_count_count": 3,
                "frame_scaling_frame_growth_factor": 4.0,
                "frame_scaling_max_no_first_growth_vs_frame_growth_ratio": 0.42,
                "extended_frame_scaling_scene_count": 5,
                "extended_frame_scaling_distinct_youtube_id_count": 5,
                "extended_frame_scaling_expected_timing_failure_count": 2,
                "extended_frame_scaling_failed_only_expected_timing": True,
                "extended_frame_scaling_no_first_timing_win": False,
                "extended_frame_scaling_no_first_growth_sublinear": False,
                "max_extended_timing_growth_overage": 0.001,
                "max_extended_no_first_ratio_overage": 0.18,
                "quality_scene_count": 5,
                "broad10_quality_distinct_youtube_id_count": 10,
                "broad_quality_distinct_youtube_id_count": 10,
                "quality_pair_count": 5,
                "all_quality_tethers_match": True,
                "max_quality_loss_curve_delta": 0.0,
                "max_quality_rgb_loss_curve_delta": 0.0,
                "max_quality_end_psnr_delta": 0.0,
                "min_quality_psnr_gain": 0.02,
                "media_scene_count": 5,
                "broad10_media_distinct_youtube_id_count": 10,
                "broad_media_distinct_youtube_id_count": 10,
                "media_pair_count": 5,
                "all_media_tethers_match": True,
                "max_media_contact_sheet_delta": 0,
                "max_media_contact_sheet_payload_loss_delta": 0.001,
                "max_media_final_rgb_loss_delta": 0.0,
                "min_media_contact_sheet_target_std": 0.14,
                "min_media_contact_sheet_pred_std": 0.07,
                "max_support_rebins": 0,
                "max_stale_refreshes": 0,
                "all_support_churn_zero": True,
                "max_rebuild_ratio": 0.5,
                "all_rebuild_ratios_at_most_half": True,
                "max_no_first_ratio_any_checked_path": 1.51,
                "bq4_fresh_process_pair_count": 6,
                "bq4_fresh_process_requested_repeat_count": 3,
                "bq4_fresh_process_warmup_discard_repeats": 1,
                "bq4_fresh_process_all_rows_fresh": True,
                "bq4_fresh_process_cache_support_clean": True,
                "bq4_fresh_process_timing_acceptance_status": "pass",
                "bq4_fresh_process_post_warmup_pair_count": 4,
                "bq4_fresh_process_post_warmup_median_no_first_ratio": 0.56,
                "bq4_fresh_process_post_warmup_median_projective_total_ratio": 0.84,
                "bq4_fresh_process_post_warmup_median_feature_state_update_ratio": 0.85,
                "bq4_fresh_process_no_first_bump_count": 0,
                "bq4_fresh_process_projective_total_bump_count": 1,
                "bq4_fresh_process_feature_state_update_bump_count": 2,
                "bq4_fresh_process_max_no_first_ratio": 0.71,
                "bq4_fresh_process_max_projective_total_ratio": 2.25,
                "bq4_fresh_process_max_feature_state_update_ratio": 1.3,
                "strict_timing_win_claimed": False,
                "fresh_process_median_timing_win_claimed": True,
                "does_not_prove_completion": True,
            },
        },
        "real_video_timing_variance_envelope": {
            "path": "real_video_timing_variance_envelope.json",
            "benchmark": "star_uvt_projective_real_video_timing_variance_envelope",
            "status": "ok",
            "verifier_errors": [],
            "summary": {
                "underlying_report_count": 9,
                "all_underlying_verifiers_pass": True,
                "source_scene_count": 5,
                "source_distinct_youtube_id_count": 5,
                "source_row_count": 30,
                "strict_failure_count": 2,
                "strict_failed_only_expected_timing": True,
                "no_first_ratio_gt1_count": 3,
                "no_first_ratio_gt1_fraction": 0.2,
                "growth_ratio_gt1_count": 1,
                "growth_ratio_gt1_fraction": 0.2,
                "max_no_first_ratio_overage": 0.18,
                "max_growth_ratio_overage": 0.001,
                "all_timing_miss_pairs_cache_clean": True,
                "all_timing_miss_pairs_support_clean": True,
                "all_cache_support_clean": True,
                "max_loss_delta": 0.0,
                "max_rebuild_ratio": 0.5,
                "dominant_no_first_render_forward_count": 2,
                "dominant_no_first_colorize_count": 1,
                "workload_explains_render_forward_miss_count": 0,
                "all_no_first_misses_tile_stats_identical": True,
                "all_render_forward_misses_tile_stats_identical": True,
                "all_no_first_misses_single_spike_driven": True,
                "drop_spike_render_forward_ratio": 0.84,
                "bq4_traced_spike_reproduced": False,
                "bq4_rerun_max_no_first_ratio": 0.58,
                "bq4_repeat_no_first_spike_count": 0,
                "bq4_repeat_projective_bump_count": 0,
                "bq4_repeat_max_no_first_ratio": 0.45,
                "bq4_sequence_no_first_bump_count": 0,
                "bq4_sequence_projective_bump_count": 2,
                "bq4_policy_no_first_bump_count": 1,
                "bq4_policy_projective_bump_count": 3,
                "fresh_process_timing_acceptance_status": "pass",
                "fresh_process_post_warmup_pair_count": 4,
                "fresh_process_median_no_first_ratio": 0.56,
                "fresh_process_median_projective_total_ratio": 0.84,
                "fresh_process_median_feature_state_update_ratio": 0.85,
                "fresh_process_max_no_first_ratio": 0.71,
                "strict_timing_win_claimed": False,
                "does_not_prove_completion": True,
            },
        },
        "real_video_compiled_adjoint_replacement": {
            "path": "real_video_compiled_adjoint_replacement.json",
            "benchmark": "star_uvt_projective_real_video_compiled_adjoint_replacement",
            "status": "ok",
            "verifier_errors": [],
            "summary": {
                "status": "ok",
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
                "broad10_quality_distinct_youtube_id_count": 10,
                "broad10_media_distinct_youtube_id_count": 10,
                "does_not_prove_completion": True,
            },
        },
        "shared_work": {
            "path": "shared_work.json",
            "benchmark": "star_uvt_projective_shared_work_goal_audit",
            "status": "ok",
            "verifier_errors": [],
            "current_input_errors": [],
            "summary": {
                "orbit_payload_growth_ratio": 0.125,
                "orbit_final_payload_ratio": 0.0625,
                "orbit_final_trace_ratio": 0.0625,
                "max_trained_final_interval_entry_ratio": 0.148,
                "max_trained_final_trace_count_ratio": 0.1,
                "trained_shared_to_replay_interval_growth_ratio": 0.148,
                "max_trained_final_forward_ms_ratio": 0.266,
                "max_trained_final_backward_ms_ratio": 0.208,
                "exposure_forward_max_metal_abs_error": 5.96e-8,
                "exposure_backward_max_metal_grad_rel_error": 6.4e-7,
                "exposure_mixed_fallback_max_grad_rel_error": 7.5e-7,
            },
        },
    }
    requirements = [_requirement(requirement_id, "proved") for requirement_id in PROVEN_REQUIREMENT_IDS]
    requirements.append(_requirement(OPEN_REQUIREMENT_ID, "open"))
    return {
        "status": "in_progress",
        "benchmark": "star_uvt_projective_goal_progress_audit",
        "goal": "fast 2D rasters across time from 4D spacetime primitives",
        "meta_goal": "share projection/support/binning/visibility/backward work over time",
        "key_math": "UVT trace = pi_* Gamma^* world_primitive",
        "theory": "STAR UVT is one local gauge expression of a camera-ray bundle atlas",
        "evidence": evidence,
        "requirements": requirements,
        "summary": summarize(requirements, evidence),
    }


def test_projective_goal_progress_audit_accepts_valid_progress_payload() -> None:
    report = _valid_report()

    assert verify_projective_goal_progress_audit(report) == []
    assert_projective_goal_progress_audit(report)
    assert (
        report["summary"][
            "real_video_acceptance_envelope_bq4_fresh_process_post_warmup_median_projective_total_ratio"
        ]
        == report["summary"]["real_video_timing_variance_fresh_process_median_projective_total_ratio"]
    )
    assert (
        report["summary"][
            "real_video_acceptance_envelope_bq4_fresh_process_post_warmup_median_feature_state_update_ratio"
        ]
        == report["summary"]["real_video_timing_variance_fresh_process_median_feature_state_update_ratio"]
    )
    assert report["summary"]["real_video_acceptance_envelope_broad_media_distinct_youtube_id_count"] == 10


def test_projective_goal_progress_current_acceptance_accepts_matching_payloads() -> None:
    report = _valid_report()

    assert verify_projective_goal_progress_current_acceptance(report, current_report=copy.deepcopy(report)) == []
    assert_projective_goal_progress_current_acceptance(report, current_report=copy.deepcopy(report))


def test_projective_goal_progress_current_acceptance_rejects_stale_but_valid_payload() -> None:
    saved = _valid_report()
    current = copy.deepcopy(saved)
    saved["evidence"]["shared_work"]["summary"]["orbit_payload_growth_ratio"] = 0.12
    saved["summary"] = summarize(saved["requirements"], saved["evidence"])

    assert verify_projective_goal_progress_audit(saved) == []
    errors = verify_projective_goal_progress_current_acceptance(saved, current_report=current)

    assert any("evidence.shared_work.summary.orbit_payload_growth_ratio" in error for error in errors)


def test_projective_goal_progress_audit_rejects_premature_complete_status() -> None:
    report = copy.deepcopy(_valid_report())
    report["status"] = "ok"
    report["summary"]["is_goal_complete"] = True

    errors = verify_projective_goal_progress_audit(report)

    assert any("status must remain in_progress" in error for error in errors)
    assert any("is_goal_complete must be false" in error for error in errors)


def test_projective_goal_progress_audit_rejects_missing_memory_contract() -> None:
    report = copy.deepcopy(_valid_report())
    report["key_math"] = "some vague trace phrase"

    errors = verify_projective_goal_progress_audit(report)

    assert any("key_math" in error for error in errors)


def test_projective_goal_progress_audit_rejects_underlying_verifier_error() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["shared_work"]["verifier_errors"] = ["stale ratio"]

    errors = verify_projective_goal_progress_audit(report)

    assert any("shared_work verifier failed" in error for error in errors)


def test_projective_goal_progress_audit_rejects_shared_current_input_errors() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["shared_work"]["current_input_errors"] = ["stale shared-work aggregate"]

    errors = verify_projective_goal_progress_audit(report)

    assert any("shared_work current-input acceptance failed" in error for error in errors)


def test_projective_goal_progress_audit_rejects_lost_open_gap() -> None:
    report = copy.deepcopy(_valid_report())
    for row in report["requirements"]:
        if row["id"] == OPEN_REQUIREMENT_ID:
            row["status"] = "proved"
            row["gaps"] = []
    report["summary"] = summarize(report["requirements"], report["evidence"])

    errors = verify_projective_goal_progress_audit(report)

    assert any("full_goal_completion must remain open" in error for error in errors)


def test_projective_goal_progress_audit_rejects_stale_summary() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["shared_work"]["summary"]["orbit_payload_growth_ratio"] = 0.13

    errors = verify_projective_goal_progress_audit(report)

    assert any("summary orbit_payload_growth_ratio mismatch" in error for error in errors)


def test_projective_goal_progress_audit_rejects_regressed_growth_ratio() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["shared_work"]["summary"]["trained_shared_to_replay_interval_growth_ratio"] = 0.4
    report["summary"] = summarize(report["requirements"], report["evidence"])

    errors = verify_projective_goal_progress_audit(report)

    assert any("trained shared/replay interval growth ratio" in error for error in errors)


def test_projective_goal_progress_audit_rejects_camera_family_regression() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["camera_family"]["summary"]["q_gradient_rel_error"] = 1.0e-3
    report["summary"] = summarize(report["requirements"], report["evidence"])

    errors = verify_projective_goal_progress_audit(report)

    assert any("camera-family q gradient invariant" in error for error in errors)


def test_projective_goal_progress_audit_rejects_camera_family_2d_regression() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["camera_family_2d"]["summary"]["q_height_gradient_rel_error"] = 1.0e-3
    report["summary"] = summarize(report["requirements"], report["evidence"])

    errors = verify_projective_goal_progress_audit(report)

    assert any("2D camera-family q_height gradient invariant" in error for error in errors)


def test_projective_goal_progress_audit_rejects_camera_family_shared_work_regression() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["camera_family_shared_work"]["summary"]["final_payload_ratio"] = 0.5
    report["summary"] = summarize(report["requirements"], report["evidence"])

    errors = verify_projective_goal_progress_audit(report)

    assert any("camera-family shared payload ratio" in error for error in errors)


def test_projective_goal_progress_audit_rejects_camera_family_2d_shared_work_regression() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["camera_family_2d_shared_work"]["summary"]["per_q_replay_payload_growth"] = 8.0
    report["summary"] = summarize(report["requirements"], report["evidence"])

    errors = verify_projective_goal_progress_audit(report)

    assert any("2D camera-family per-q-pair replay payload growth" in error for error in errors)


def test_projective_goal_progress_audit_rejects_camera_family_2d_metal_lowering_regression() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["camera_family_2d_metal_lowering"]["summary"]["peak_slice_to_replay_payload_ratio"] = 0.5
    report["summary"] = summarize(report["requirements"], report["evidence"])

    errors = verify_projective_goal_progress_audit(report)

    assert any("2D camera-family Metal lowering peak slice/replay ratio" in error for error in errors)


def test_projective_goal_progress_audit_rejects_camera_family_2d_metal_chain_rule_regression() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["camera_family_2d_metal_chain_rule"]["summary"][
        "max_finite_difference_rel_error"
    ] = 1.0e-2
    report["summary"] = summarize(report["requirements"], report["evidence"])

    errors = verify_projective_goal_progress_audit(report)

    assert any("2D camera-family Metal chain-rule finite-difference error" in error for error in errors)


def test_projective_goal_progress_audit_rejects_camera_family_2d_materialized_batch_regression() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["camera_family_2d_materialized_batch"]["summary"][
        "materialized_to_replay_trace_payload_ratio"
    ] = 0.5
    report["summary"] = summarize(report["requirements"], report["evidence"])

    errors = verify_projective_goal_progress_audit(report)

    assert any("materialized/replay trace payload" in error for error in errors)


def test_projective_goal_progress_audit_rejects_camera_family_2d_native_eval_regression() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["camera_family_2d_native_eval"]["summary"]["native_grad_q_basis_max_rel_error"] = 1.0e-3
    report["summary"] = summarize(report["requirements"], report["evidence"])

    errors = verify_projective_goal_progress_audit(report)

    assert any("native eval q-basis-gradient rel error" in error for error in errors)


def test_projective_goal_progress_audit_rejects_camera_family_2d_native_interval_forward_regression() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["camera_family_2d_native_interval_forward"]["summary"][
        "family_forward_to_materialized_trace_payload_ratio"
    ] = 0.75
    report["summary"] = summarize(report["requirements"], report["evidence"])

    errors = verify_projective_goal_progress_audit(report)

    assert any("native interval forward payload ratio" in error for error in errors)


def test_projective_goal_progress_audit_rejects_camera_family_2d_native_interval_backward_regression() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["camera_family_2d_native_interval_backward"]["summary"][
        "native_family_interval_backward_max_q_basis_grad_rel_error"
    ] = 1.0e-3
    report["summary"] = summarize(report["requirements"], report["evidence"])

    errors = verify_projective_goal_progress_audit(report)

    assert any("native interval backward q-basis-gradient rel error" in error for error in errors)


def test_projective_goal_progress_audit_rejects_camera_family_2d_tile_order_reuse_regression() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["camera_family_2d_tile_order_reuse"]["summary"][
        "shared_to_materialized_tile_order_metadata_ratio"
    ] = 0.5
    report["summary"] = summarize(report["requirements"], report["evidence"])

    errors = verify_projective_goal_progress_audit(report)

    assert any("tile/order shared/materialized metadata ratio" in error for error in errors)


def test_projective_goal_progress_audit_rejects_camera_family_2d_tile_order_strata_regression() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["camera_family_2d_tile_order_strata"]["summary"][
        "shared_tile_order_metadata_growth"
    ] = 25.0
    report["summary"] = summarize(report["requirements"], report["evidence"])

    errors = verify_projective_goal_progress_audit(report)

    assert any("tile/order strata shared growth" in error for error in errors)


def test_projective_goal_progress_audit_rejects_camera_family_2d_active_set_strata_regression() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["camera_family_2d_active_set_strata"]["summary"][
        "shared_tile_order_metadata_growth"
    ] = 25.0
    report["summary"] = summarize(report["requirements"], report["evidence"])

    errors = verify_projective_goal_progress_audit(report)

    assert any("active-set strata shared growth" in error for error in errors)


def test_projective_goal_progress_audit_rejects_real_active_set_distribution_regression() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["real_active_set_distribution"]["summary"][
        "max_active_set_group_to_dense_tile_pair_ratio"
    ] = 0.25
    report["summary"] = summarize(report["requirements"], report["evidence"])

    errors = verify_projective_goal_progress_audit(report)

    assert any("real active-set distribution group/dense-tile-pair ratio" in error for error in errors)


def test_projective_goal_progress_audit_rejects_trainer_interval_regression() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["trainer_interval"]["summary"]["measured_vs_cadence_rebuild_ratios"] = [0.5, 1.0]
    report["summary"] = summarize(report["requirements"], report["evidence"])

    errors = verify_projective_goal_progress_audit(report)

    assert any("trainer interval measured rebuild ratios" in error for error in errors)


def test_projective_goal_progress_audit_rejects_real_video_trainer_regression() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["trainer_real_video"]["summary"]["measured_vs_cadence_no_first_step_ms_ratios"] = [
        0.8,
        1.0,
    ]
    report["summary"] = summarize(report["requirements"], report["evidence"])

    errors = verify_projective_goal_progress_audit(report)

    assert any("real-video trainer measured no-first-step timings" in error for error in errors)


def test_projective_goal_progress_audit_rejects_real_video_guarded_support_regression() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["real_video_guarded_support_matrix"]["summary"]["guarded_measured_support_rebins"] = 1
    report["summary"] = summarize(report["requirements"], report["evidence"])

    errors = verify_projective_goal_progress_audit(report)

    assert any("real-video guarded support matrix must eliminate guarded support rebins" in error for error in errors)


def test_projective_goal_progress_audit_rejects_real_video_multiscene_regression() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["real_video_multiscene_trainer_matrix"]["summary"][
        "max_measured_support_rebins"
    ] = 1
    report["summary"] = summarize(report["requirements"], report["evidence"])

    errors = verify_projective_goal_progress_audit(report)

    assert any("real-video multiscene trainer matrix must eliminate measured support rebins" in error for error in errors)


def test_projective_goal_progress_audit_rejects_extended_multiscene_regression() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["real_video_multiscene_extended_trainer_matrix"]["summary"]["scene_count"] = 4
    report["summary"] = summarize(report["requirements"], report["evidence"])

    errors = verify_projective_goal_progress_audit(report)

    assert any("extended real-video multiscene trainer matrix must cover at least five scenes" in error for error in errors)


def test_projective_goal_progress_audit_rejects_real_video_multiscene_frame_scaling_regression() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["real_video_multiscene_frame_scaling_matrix"]["summary"][
        "max_measured_no_first_growth_vs_frame_growth_ratio"
    ] = 1.0
    report["summary"] = summarize(report["requirements"], report["evidence"])

    errors = verify_projective_goal_progress_audit(report)

    assert any("real-video multiscene frame-scaling matrix timing growth" in error for error in errors)


def test_projective_goal_progress_audit_rejects_extended_frame_scaling_diagnostic_regression() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["real_video_multiscene_extended_frame_scaling_diagnostic"]["summary"][
        "strict_failed_only_expected_timing"
    ] = False
    report["summary"] = summarize(report["requirements"], report["evidence"])

    errors = verify_projective_goal_progress_audit(report)

    assert any("extended frame-scaling diagnostic must fail only the expected timing gates" in error for error in errors)


def test_projective_goal_progress_audit_rejects_real_video_quality_tether_regression() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["real_video_multiscene_quality_tether"]["summary"][
        "max_abs_loss_curve_delta"
    ] = 1.0e-4
    report["summary"] = summarize(report["requirements"], report["evidence"])

    errors = verify_projective_goal_progress_audit(report)

    assert any("real-video multiscene quality tether loss-curve delta" in error for error in errors)


def test_projective_goal_progress_audit_rejects_extended_real_video_quality_tether_regression() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["real_video_multiscene_extended_quality_tether"]["summary"][
        "max_abs_loss_curve_delta"
    ] = 1.0e-4
    report["summary"] = summarize(report["requirements"], report["evidence"])

    errors = verify_projective_goal_progress_audit(report)

    assert any("extended real-video quality tether loss-curve delta" in error for error in errors)


def test_projective_goal_progress_audit_rejects_real_video_media_tether_regression() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["real_video_multiscene_media_tether"]["summary"][
        "max_abs_contact_sheet_delta"
    ] = 1
    report["summary"] = summarize(report["requirements"], report["evidence"])

    errors = verify_projective_goal_progress_audit(report)

    assert any("real-video multiscene media tether contact-sheet delta" in error for error in errors)


def test_projective_goal_progress_audit_rejects_extended_real_video_media_tether_regression() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["real_video_multiscene_extended_media_tether"]["summary"][
        "max_abs_contact_sheet_delta"
    ] = 1
    report["summary"] = summarize(report["requirements"], report["evidence"])

    errors = verify_projective_goal_progress_audit(report)

    assert any("extended real-video media tether contact-sheet delta" in error for error in errors)


def test_projective_goal_progress_audit_rejects_real_video_acceptance_envelope_regression() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["real_video_acceptance_envelope"]["summary"]["does_not_prove_completion"] = False
    report["summary"] = summarize(report["requirements"], report["evidence"])

    errors = verify_projective_goal_progress_audit(report)

    assert any("real-video acceptance envelope must preserve non-completion scope" in error for error in errors)


def test_projective_goal_progress_audit_rejects_lost_broad_media_acceptance() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["real_video_acceptance_envelope"]["summary"][
        "broad_media_distinct_youtube_id_count"
    ] = 5
    report["summary"] = summarize(report["requirements"], report["evidence"])

    errors = verify_projective_goal_progress_audit(report)

    assert any("at least 10 broad media sources" in error for error in errors)


def test_projective_goal_progress_audit_rejects_real_video_acceptance_envelope_bq4_median_regression() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["real_video_acceptance_envelope"]["summary"][
        "bq4_fresh_process_timing_acceptance_status"
    ] = "fail"
    report["summary"] = summarize(report["requirements"], report["evidence"])

    errors = verify_projective_goal_progress_audit(report)

    assert any("Bq4 fresh-process median timing must pass" in error for error in errors)


def test_projective_goal_progress_audit_rejects_real_video_acceptance_envelope_bq4_no_first_bump() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["real_video_acceptance_envelope"]["summary"]["bq4_fresh_process_no_first_bump_count"] = 1
    report["summary"] = summarize(report["requirements"], report["evidence"])

    errors = verify_projective_goal_progress_audit(report)

    assert any("Bq4 fresh-process gate must have zero no-first bumps" in error for error in errors)


def test_projective_goal_progress_audit_rejects_real_video_timing_variance_envelope_regression() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["real_video_timing_variance_envelope"]["summary"][
        "fresh_process_timing_acceptance_status"
    ] = "fail"
    report["summary"] = summarize(report["requirements"], report["evidence"])

    errors = verify_projective_goal_progress_audit(report)

    assert any("timing variance envelope fresh-process acceptance" in error for error in errors)


def test_projective_goal_progress_audit_rejects_compiled_adjoint_replacement_regression() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["real_video_compiled_adjoint_replacement"]["summary"][
        "final_compiled_adjoint_replacement_accepted"
    ] = False
    report["summary"] = summarize(report["requirements"], report["evidence"])

    errors = verify_projective_goal_progress_audit(report)

    assert any("compiled-adjoint replacement must be accepted" in error for error in errors)


def test_projective_goal_progress_audit_rejects_bq4_fresh_process_cross_artifact_drift() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["real_video_timing_variance_envelope"]["summary"][
        "fresh_process_median_projective_total_ratio"
    ] = 0.92
    report["summary"] = summarize(report["requirements"], report["evidence"])

    errors = verify_projective_goal_progress_audit(report)

    assert any(
        "Bq4 fresh-process median projective-total ratio must match across artifacts" in error
        for error in errors
    )


def test_projective_goal_progress_audit_reads_current_saved_artifacts() -> None:
    required = (
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_bundle_gauge_invariance/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_bundle_gauge_gradient/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_gauge/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_gauge/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_metal_lowering/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_metal_chain_rule/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_materialized_batch/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_native_eval/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_native_interval_forward/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_native_interval_backward/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_tile_order_reuse/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_tile_order_strata/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_active_set_strata/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_real_active_set_distribution/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_shared_work_scaling/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_2d_shared_work_scaling/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_interval_trainer_frame_scaling/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_trainer_frame_scaling/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_guarded_support_matrix/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_trainer_matrix/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_trainer_matrix_extended5/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_frame_scaling_matrix/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_frame_scaling_diagnostic/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_quality_tether/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_quality_tether/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_media_tether/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_media_tether/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_acceptance_envelope/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_timing_variance_envelope/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_compiled_adjoint_replacement/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_shared_work_goal_audit/summary.json"),
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        pytest.skip(f"missing optional audit inputs: {missing}")

    report = run_report()

    assert_projective_goal_progress_audit(report)
    assert report["summary"]["is_goal_complete"] is False


def test_saved_projective_goal_progress_audit_artifact_satisfies_contract() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(Path(summary_json).read_text(encoding="utf-8"))

    assert_projective_goal_progress_audit(report)


def test_saved_projective_goal_progress_audit_artifact_matches_current_inputs() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(Path(summary_json).read_text(encoding="utf-8"))

    assert_projective_goal_progress_current_acceptance(report)
