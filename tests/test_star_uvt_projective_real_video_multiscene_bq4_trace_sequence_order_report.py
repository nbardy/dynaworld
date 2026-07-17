from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_experiments.star_uvt_feature_tubes.projective_real_video_multiscene_bq4_trace_sequence_order_report import (
    DEFAULT_OUT_DIR,
    assert_bq4_trace_sequence_order_report,
    summarize,
    verify_bq4_trace_sequence_order_report,
)


SEQUENCES = (
    ("mixed_4_to_16", (4, 16)),
    ("reverse_16_to_4", (16, 4)),
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
        "chunk_frames": 16,
        "render_forward_ms": total + 1.0,
        "projective_interval_timing_ms": timing,
        "subtiming_sum_ms": feature_state_ms + 9.0,
        "subtiming_total_abs_delta_ms": abs(total - feature_state_ms - 9.0),
        "projective_total_to_chunk_render_ratio": total / (total + 1.0),
    }


def _profile(sequence_name: str, frames: int, policy: str, repeat_index: int, frame_index: int) -> dict[str, object]:
    cadence_no_first_ms = 100.0 + frames + repeat_index
    measured_scale = 0.72 if frames == 16 else 0.85
    feature_state_ms = 20.0 + frames + repeat_index
    if policy == "measured":
        feature_state_ms *= 0.9
    total = feature_state_ms + 9.0
    return {
        "scene_id": "Bq4rmeIvJbs_seg_000",
        "frames": frames,
        "policy": policy,
        "sequence_name": sequence_name,
        "repeat_index": repeat_index,
        "sequence_frame_index": frame_index,
        "case_json": f"case-{sequence_name}-{repeat_index}-{frames}-{policy}.json",
        "expected_trace_global_step": 1 if frames == 4 else 3,
        "chunk_trace_global_steps": [1 if frames == 4 else 3],
        "matching_trace_count": 1,
        "chunk_profile_count": 1,
        "chunk_profiles": [_chunk_profile(feature_state_ms=feature_state_ms, total=total)],
        "projective_interval_cache_rebuilds": 2 if policy == "cadence" else 1,
        "projective_interval_cache_live_updates": 2 if policy == "cadence" else 3,
        "projective_interval_cache_support_rebins": 0,
        "projective_interval_cache_stale_refreshes": 0,
        "projective_interval_cache_fallback_marks": 0,
        "tile_overflow_sum": 0,
        "no_first_step_ms": cadence_no_first_ms if policy == "cadence" else cadence_no_first_ms * measured_scale,
        "mean_render_forward_ms": total * 0.5,
    }


def _report() -> dict[str, object]:
    profiles = [
        _profile(sequence_name, frames, policy, 0, frame_index)
        for sequence_name, frame_list in SEQUENCES
        for frame_index, frames in enumerate(frame_list)
        for policy in ("cadence", "measured")
    ]
    report: dict[str, object] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_real_video_multiscene_bq4_trace_sequence_order",
        "base_domain": "Bq4 traced mixed-frame sequence-order stability",
        "theory_contract": "sequence order timing test",
        "source_shape_status": "ok",
        "requested_repeat_count": 1,
        "sequences": [{"name": name, "frames": list(frames)} for name, frames in SEQUENCES],
        "rows": [{"scene_id": "Bq4rmeIvJbs_seg_000"} for _ in profiles],
        "trace_profiles": profiles,
    }
    report["summary"] = summarize(report)
    return report


def test_bq4_trace_sequence_order_verifier_accepts_mixed_sequences() -> None:
    report = _report()

    assert verify_bq4_trace_sequence_order_report(report) == []
    assert_bq4_trace_sequence_order_report(report)
    assert report["summary"]["paired_16f_ratio_count"] == 2
    assert report["summary"]["all_16f_summary"]["feature_state_update_bump_count"] == 0


def test_bq4_trace_sequence_order_verifier_rejects_stale_summary() -> None:
    report = _report()
    report["trace_profiles"][0]["chunk_profiles"][0]["projective_interval_timing_ms"][
        "feature_state_update_ms"
    ] = 99.0

    errors = verify_bq4_trace_sequence_order_report(report)

    assert any("summary" in error and "mismatch" in error for error in errors)


def test_bq4_trace_sequence_order_verifier_rejects_missing_16f_pair() -> None:
    report = _report()
    report["trace_profiles"] = [
        profile
        for profile in report["trace_profiles"]
        if not (
            profile["sequence_name"] == "reverse_16_to_4"
            and profile["frames"] == 16
            and profile["policy"] == "measured"
        )
    ]
    report["summary"] = summarize(report)

    errors = verify_bq4_trace_sequence_order_report(report)

    assert any("missing reverse_16_to_4 repeat 0 16f measured" in error for error in errors)
    assert any("expected 2 paired 16f ratios" in error for error in errors)


def test_bq4_trace_sequence_order_verifier_rejects_dirty_cache_support() -> None:
    report = _report()
    report["trace_profiles"][0]["projective_interval_cache_fallback_marks"] = 1
    report["summary"] = summarize(report)

    errors = verify_bq4_trace_sequence_order_report(report)

    assert any("cache/support clean" in error for error in errors)


def test_saved_bq4_trace_sequence_order_artifact_satisfies_contract() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(Path(summary_json).read_text(encoding="utf-8"))

    assert_bq4_trace_sequence_order_report(report)
