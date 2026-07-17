from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research_experiments.star_uvt_feature_tubes.projective_real_video_timing_variance_envelope_report import (
    DEFAULT_OUT_DIR,
    EVIDENCE_ORDER,
    assert_real_video_timing_variance_envelope_report,
    summarize,
    verify_real_video_timing_variance_envelope_report,
)


def _timing_summary() -> dict[str, object]:
    return {
        "source_status": "failed",
        "source_scene_count": 5,
        "source_distinct_youtube_id_count": 5,
        "source_row_count": 30,
        "pair_count": 15,
        "scene_count": 5,
        "frame_count_count": 3,
        "frame_growth_factor": 4.0,
        "strict_failure_count": 2,
        "strict_failed_only_expected_timing": True,
        "no_first_ratio_gt1_count": 3,
        "no_first_ratio_gt1_fraction": 0.2,
        "growth_ratio_gt1_count": 1,
        "growth_ratio_gt1_fraction": 0.2,
        "max_measured_vs_cadence_no_first_step_ms_ratio": 1.18,
        "max_no_first_ratio_overage": 0.18,
        "max_measured_no_first_growth_vs_frame_growth_ratio": 1.001,
        "max_growth_ratio_overage": 0.001,
        "all_failing_pairs_cache_clean": True,
        "all_pair_support_clean": True,
        "all_pair_loss_matches_cadence": True,
        "all_pair_rebuild_ratio_below_cadence": True,
        "all_scene_rebuild_growth_flat": True,
        "max_end_loss_abs_delta": 0.0,
        "max_measured_vs_cadence_rebuild_ratio": 0.5,
        "max_measured_support_rebins": 0,
        "max_measured_stale_refreshes": 0,
        "max_measured_fallback_marks": 0,
        "max_measured_visibility_stratifications": 0,
        "max_measured_tile_overflow_sum": 0,
    }


def _phase_summary() -> dict[str, object]:
    return {
        "source_status": "failed",
        "source_scene_count": 5,
        "source_row_count": 30,
        "strict_failure_count": 2,
        "strict_failed_only_expected_timing": True,
        "phase_profile_count": 5,
        "profile_scene_count": 3,
        "no_first_miss_profile_count": 3,
        "all_profile_pairs_cache_support_clean": True,
        "all_profile_losses_match_cadence": True,
        "all_profile_step_no_first_matches_source": True,
        "max_profile_loss_delta": 0.0,
        "max_profile_rebuild_ratio": 0.5,
        "dominant_positive_phase_counts_for_no_first_misses": {
            "render_forward_ms": 2,
            "colorize_loss_ms": 1,
        },
        "max_render_forward_ratio": 1.35,
        "max_backward_ratio": 1.08,
    }


def _residual_summary() -> dict[str, object]:
    return {
        "source_status": "failed",
        "source_scene_count": 5,
        "source_row_count": 30,
        "strict_failure_count": 2,
        "strict_failed_only_expected_timing": True,
        "pair_count": 15,
        "no_first_miss_pair_count": 3,
        "render_forward_miss_pair_count": 3,
        "all_pairs_cache_support_clean": True,
        "all_pairs_losses_match_cadence": True,
        "all_profile_step_no_first_matches_source": True,
        "all_no_first_misses_tile_stats_identical": True,
        "all_render_forward_misses_tile_stats_identical": True,
        "all_policy_tile_stats_identical": True,
        "max_loss_delta": 0.0,
        "max_tile_stats_abs_delta": 0.0,
        "workload_explains_render_forward_miss_count": 0,
        "max_render_forward_ratio": 1.35,
    }


def _shape_summary() -> dict[str, object]:
    return {
        "source_status": "failed",
        "source_scene_count": 5,
        "source_row_count": 30,
        "strict_failure_count": 2,
        "strict_failed_only_expected_timing": True,
        "pair_count": 15,
        "no_first_miss_pair_count": 3,
        "render_forward_miss_pair_count": 3,
        "all_pairs_cache_support_clean": True,
        "all_pairs_losses_match_cadence": True,
        "all_profile_step_no_first_matches_source": True,
        "all_no_first_misses_render_single_spike_driven": True,
        "all_no_first_misses_step_single_spike_driven": True,
        "max_loss_delta": 0.0,
        "max_no_first_miss_render_forward_drop_spike_ratio": 0.84,
        "max_no_first_miss_render_forward_ratio_spread": 5.3,
        "max_no_first_miss_render_forward_spike_delta_ms": 728.0,
    }


def _trace_clean(**summary: object) -> dict[str, object]:
    base = {
        "all_expected_global_steps_traced": True,
        "all_projective_interval_timing_present": True,
        "all_rows_cache_support_clean": True,
        "chunk_profile_count": 4,
        "trace_profile_count": 4,
    }
    base.update(summary)
    return base


def _rerun_summary() -> dict[str, object]:
    return _trace_clean(
        source_shape_status="ok",
        trace_spec_count=2,
        source_shape_pair_count=15,
        traced_bq4_spike_reproduced=False,
        max_traced_measured_vs_cadence_no_first_step_ms_ratio=0.58,
        max_traced_measured_vs_cadence_projective_total_ratio=1.27,
        max_traced_measured_vs_cadence_feature_state_update_ratio=1.25,
    )


def _repeat_summary() -> dict[str, object]:
    return _trace_clean(
        source_shape_status="ok",
        repeat_count=3,
        requested_repeat_count=3,
        paired_repeat_count=3,
        no_first_spike_reproduced_count=0,
        projective_total_bump_count=0,
        feature_state_update_bump_count=0,
        max_no_first_ratio=0.45,
        max_projective_total_ratio=0.91,
        max_feature_state_update_ratio=0.79,
    )


def _sequence_summary() -> dict[str, object]:
    return _trace_clean(
        source_shape_status="ok",
        sequence_count=2,
        paired_16f_ratio_count=4,
        all_16f_summary={
            "pair_count": 4,
            "max_no_first_ratio": 0.46,
            "no_first_bump_count": 0,
            "max_projective_total_ratio": 1.84,
            "projective_total_bump_count": 2,
            "max_feature_state_update_ratio": 1.73,
            "feature_state_update_bump_count": 3,
        },
    )


def _policy_summary() -> dict[str, object]:
    return _trace_clean(
        source_shape_status="ok",
        policy_order_count=2,
        paired_ratio_count=4,
        all_target_summary={
            "pair_count": 4,
            "max_no_first_ratio": 1.78,
            "no_first_bump_count": 1,
            "max_projective_total_ratio": 1.72,
            "projective_total_bump_count": 3,
            "max_feature_state_update_ratio": 1.96,
            "feature_state_update_bump_count": 3,
        },
    )


def _fresh_summary() -> dict[str, object]:
    return _trace_clean(
        source_shape_status="ok",
        all_rows_fresh_process=True,
        requested_repeat_count=3,
        warmup_discard_repeats=1,
        policy_order_count=2,
        paired_ratio_count=6,
        all_target_summary={
            "pair_count": 6,
            "max_no_first_ratio": 0.71,
            "median_no_first_ratio": 0.65,
            "max_projective_total_ratio": 2.24,
            "median_projective_total_ratio": 0.84,
            "max_feature_state_update_ratio": 1.29,
            "median_feature_state_update_ratio": 0.71,
        },
        timing_acceptance={
            "status": "pass",
            "ratio_threshold": 1.0,
            "warmup_discard_repeats": 1,
            "requested_repeat_count": 3,
            "policy_order_count": 2,
            "expected_post_warmup_pair_count": 4,
            "post_warmup_pair_count": 4,
            "sufficient_repeats": True,
            "sufficient_pairs": True,
            "median_ratios_within_threshold": True,
            "post_warmup_summary": {
                "pair_count": 4,
                "median_no_first_ratio": 0.56,
                "median_projective_total_ratio": 0.84,
                "median_feature_state_update_ratio": 0.85,
                "max_no_first_ratio": 0.71,
                "max_projective_total_ratio": 2.24,
                "max_feature_state_update_ratio": 1.29,
            },
        },
    )


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
        "timing_breakdown": _evidence(_timing_summary()),
        "phase_profile": _evidence(_phase_summary()),
        "render_forward_residual": _evidence(_residual_summary()),
        "render_forward_shape": _evidence(_shape_summary()),
        "bq4_trace_rerun": _evidence(_rerun_summary()),
        "bq4_repeat_stability": _evidence(_repeat_summary()),
        "bq4_sequence_order": _evidence(_sequence_summary()),
        "bq4_policy_order": _evidence(_policy_summary()),
        "bq4_fresh_process": _evidence(_fresh_summary()),
    }
    report: dict[str, object] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_real_video_timing_variance_envelope",
        "goal": "fast 2D rasters across time from 4D spacetime primitives",
        "meta_goal": "share projection/support/binning/visibility/backward work over time",
        "theory_contract": (
            "This timing-variance envelope tracks MPS process variance. It does not prove a broad "
            "timing win and does not prove full goal completion."
        ),
        "does_not_prove_completion": True,
        "evidence": evidence,
    }
    report["summary"] = summarize(report)  # type: ignore[arg-type]
    return report


def test_real_video_timing_variance_envelope_accepts_valid_report() -> None:
    report = _valid_report()

    assert verify_real_video_timing_variance_envelope_report(report) == []
    assert_real_video_timing_variance_envelope_report(report)
    assert report["summary"]["strict_timing_win_claimed"] is False  # type: ignore[index]


def test_real_video_timing_variance_envelope_rejects_scope_overclaim() -> None:
    report = _valid_report()
    report["theory_contract"] = "all done"

    errors = verify_real_video_timing_variance_envelope_report(report)

    assert any("non-completion scope" in error for error in errors)


def test_real_video_timing_variance_envelope_rejects_underlying_error() -> None:
    report = _valid_report()
    report["evidence"]["bq4_fresh_process"]["verifier_errors"] = ["fresh process failed"]  # type: ignore[index]

    errors = verify_real_video_timing_variance_envelope_report(report)

    assert any("bq4_fresh_process verifier failed" in error for error in errors)


def test_real_video_timing_variance_envelope_rejects_dirty_cache_support() -> None:
    report = _valid_report()
    report["evidence"]["timing_breakdown"]["summary"]["all_pair_support_clean"] = False  # type: ignore[index]
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_real_video_timing_variance_envelope_report(report)

    assert any("all_pair_support_clean" in error for error in errors)


def test_real_video_timing_variance_envelope_rejects_workload_explanation() -> None:
    report = _valid_report()
    report["evidence"]["render_forward_residual"]["summary"]["workload_explains_render_forward_miss_count"] = 1  # type: ignore[index]
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_real_video_timing_variance_envelope_report(report)

    assert any("must not attribute misses to tile workload changes" in error for error in errors)


def test_real_video_timing_variance_envelope_rejects_failed_fresh_median() -> None:
    report = _valid_report()
    post = report["evidence"]["bq4_fresh_process"]["summary"]["timing_acceptance"]["post_warmup_summary"]  # type: ignore[index]
    post["median_projective_total_ratio"] = 1.2
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_real_video_timing_variance_envelope_report(report)

    assert any("fresh-process median_projective_total_ratio" in error for error in errors)


def test_real_video_timing_variance_envelope_rejects_stale_summary() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["bq4_repeat_stability"]["summary"]["max_no_first_ratio"] = 0.9  # type: ignore[index]

    errors = verify_real_video_timing_variance_envelope_report(report)

    assert any("summary bq4_repeat_max_no_first_ratio mismatch" in error for error in errors)


def test_real_video_timing_variance_envelope_keeps_expected_evidence_order() -> None:
    report = _valid_report()

    assert tuple(report["evidence"]) == EVIDENCE_ORDER


def test_saved_real_video_timing_variance_envelope_artifact_satisfies_contract() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(Path(summary_json).read_text(encoding="utf-8"))

    assert_real_video_timing_variance_envelope_report(report)
