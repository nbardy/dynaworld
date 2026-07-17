from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_experiments.star_uvt_feature_tubes.projective_real_video_multiscene_bq4_trace_fresh_process_report import (
    DEFAULT_ACCEPTANCE_RATIO_THRESHOLD,
    DEFAULT_OUT_DIR,
    DEFAULT_TARGET_FRAMES,
    assert_bq4_trace_fresh_process_report,
    summarize,
    verify_bq4_trace_fresh_process_report,
)


POLICY_ORDERS = (
    ("cadence_then_measured", ("cadence", "measured")),
    ("measured_then_cadence", ("measured", "cadence")),
)


def _chunk_profile(*, feature_state_ms: float, total: float) -> dict[str, object]:
    timing = {
        "feature_state_update_ms": feature_state_ms,
        "feature_render_ms": 2.0,
        "alpha_state_update_ms": 3.0,
        "alpha_render_ms": 4.0,
        "projective_interval_render_ms": total,
    }
    return {
        "frame_start": 0,
        "chunk_frames": DEFAULT_TARGET_FRAMES,
        "render_forward_ms": total + 1.0,
        "projective_interval_timing_ms": timing,
        "subtiming_sum_ms": feature_state_ms + 9.0,
        "subtiming_total_abs_delta_ms": abs(total - feature_state_ms - 9.0),
        "projective_total_to_chunk_render_ratio": total / (total + 1.0),
    }


def _profile(order_name: str, policy: str, repeat_index: int, policy_slot: int) -> dict[str, object]:
    feature_state_ms = 40.0 + repeat_index
    if policy == "measured":
        feature_state_ms *= 0.85
    total = feature_state_ms + 9.0
    cadence_no_first_ms = 200.0 + repeat_index
    return {
        "scene_id": "Bq4rmeIvJbs_seg_000",
        "frames": DEFAULT_TARGET_FRAMES,
        "policy": policy,
        "policy_order_name": order_name,
        "repeat_index": repeat_index,
        "policy_slot": policy_slot,
        "fresh_process": True,
        "case_json": f"case-{order_name}-{repeat_index}-{policy}.json",
        "expected_trace_global_step": 3,
        "chunk_trace_global_steps": [3],
        "matching_trace_count": 1,
        "chunk_profile_count": 1,
        "chunk_profiles": [_chunk_profile(feature_state_ms=feature_state_ms, total=total)],
        "projective_interval_cache_rebuilds": 2 if policy == "cadence" else 1,
        "projective_interval_cache_live_updates": 2 if policy == "cadence" else 3,
        "projective_interval_cache_support_rebins": 0,
        "projective_interval_cache_stale_refreshes": 0,
        "projective_interval_cache_fallback_marks": 0,
        "tile_overflow_sum": 0,
        "no_first_step_ms": cadence_no_first_ms if policy == "cadence" else cadence_no_first_ms * 0.75,
        "mean_render_forward_ms": total * 0.5,
    }


def _report() -> dict[str, object]:
    profiles = [
        _profile(order_name, policy, 0, slot)
        for order_name, policies in POLICY_ORDERS
        for slot, policy in enumerate(policies)
    ]
    report: dict[str, object] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_real_video_multiscene_bq4_trace_fresh_process",
        "base_domain": "Bq4 fresh-process 16f policy-order timing isolation",
        "theory_contract": "fresh-process policy-order timing test",
        "source_shape_status": "ok",
        "target_frames": DEFAULT_TARGET_FRAMES,
        "requested_repeat_count": 1,
        "warmup_discard_repeats": 0,
        "acceptance_ratio_threshold": DEFAULT_ACCEPTANCE_RATIO_THRESHOLD,
        "policy_orders": [{"name": name, "policies": list(policies)} for name, policies in POLICY_ORDERS],
        "rows": [{"scene_id": "Bq4rmeIvJbs_seg_000"} for _ in profiles],
        "trace_profiles": profiles,
    }
    report["summary"] = summarize(report)
    return report


def test_bq4_trace_fresh_process_verifier_accepts_fresh_policy_orders() -> None:
    report = _report()

    assert verify_bq4_trace_fresh_process_report(report) == []
    assert_bq4_trace_fresh_process_report(report)
    assert report["summary"]["paired_ratio_count"] == 2
    assert report["summary"]["all_rows_fresh_process"] is True
    assert report["summary"]["all_target_summary"]["median_no_first_ratio"] == 0.75
    assert report["summary"]["timing_acceptance"]["status"] == "pass"


def test_bq4_trace_fresh_process_summary_applies_warmup_discard_acceptance() -> None:
    first_repeat = [
        _profile(order_name, policy, 0, slot)
        for order_name, policies in POLICY_ORDERS
        for slot, policy in enumerate(policies)
    ]
    second_repeat = [
        _profile(order_name, policy, 1, slot)
        for order_name, policies in POLICY_ORDERS
        for slot, policy in enumerate(policies)
    ]
    for profile in first_repeat:
        if profile["policy"] == "measured":
            profile["no_first_step_ms"] = float(profile["no_first_step_ms"]) * 4.0
    report: dict[str, object] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_real_video_multiscene_bq4_trace_fresh_process",
        "base_domain": "Bq4 fresh-process 16f policy-order timing isolation",
        "theory_contract": "fresh-process policy-order timing test",
        "source_shape_status": "ok",
        "target_frames": DEFAULT_TARGET_FRAMES,
        "requested_repeat_count": 2,
        "warmup_discard_repeats": 1,
        "acceptance_ratio_threshold": DEFAULT_ACCEPTANCE_RATIO_THRESHOLD,
        "policy_orders": [{"name": name, "policies": list(policies)} for name, policies in POLICY_ORDERS],
        "rows": [{"scene_id": "Bq4rmeIvJbs_seg_000"} for _ in first_repeat + second_repeat],
        "trace_profiles": first_repeat + second_repeat,
    }

    report["summary"] = summarize(report)

    assert_bq4_trace_fresh_process_report(report)
    assert report["summary"]["all_target_summary"]["max_no_first_ratio"] > 1.0
    assert report["summary"]["timing_acceptance"]["post_warmup_pair_count"] == 2
    assert report["summary"]["timing_acceptance"]["post_warmup_summary"]["max_no_first_ratio"] == 0.75
    assert report["summary"]["timing_acceptance"]["status"] == "pass"


def test_bq4_trace_fresh_process_verifier_rejects_non_fresh_profile() -> None:
    report = _report()
    report["trace_profiles"][0]["fresh_process"] = False
    report["summary"] = summarize(report)

    errors = verify_bq4_trace_fresh_process_report(report)

    assert any("fresh_process" in error for error in errors)


def test_bq4_trace_fresh_process_verifier_rejects_missing_target_pair() -> None:
    report = _report()
    report["trace_profiles"] = [
        profile
        for profile in report["trace_profiles"]
        if not (
            profile["policy_order_name"] == "measured_then_cadence"
            and profile["policy"] == "cadence"
        )
    ]
    report["summary"] = summarize(report)

    errors = verify_bq4_trace_fresh_process_report(report)

    assert any("missing measured_then_cadence repeat 0 cadence target profile" in error for error in errors)
    assert any("expected 2 target paired ratios" in error for error in errors)


def test_bq4_trace_fresh_process_verifier_rejects_dirty_cache_support() -> None:
    report = _report()
    report["trace_profiles"][0]["projective_interval_cache_support_rebins"] = 1
    report["summary"] = summarize(report)

    errors = verify_bq4_trace_fresh_process_report(report)

    assert any("cache/support clean" in error for error in errors)


def test_saved_bq4_trace_fresh_process_artifact_satisfies_contract() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(Path(summary_json).read_text(encoding="utf-8"))

    assert_bq4_trace_fresh_process_report(report)
