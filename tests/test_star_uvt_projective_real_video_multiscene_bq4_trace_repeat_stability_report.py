from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_experiments.star_uvt_feature_tubes.projective_real_video_multiscene_bq4_trace_repeat_stability_report import (
    DEFAULT_FRAMES,
    DEFAULT_OUT_DIR,
    assert_bq4_trace_repeat_stability_report,
    summarize,
    verify_bq4_trace_repeat_stability_report,
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
        "chunk_frames": DEFAULT_FRAMES,
        "render_forward_ms": total + 1.0,
        "projective_interval_timing_ms": timing,
        "subtiming_sum_ms": feature_state_ms + 9.0,
        "subtiming_total_abs_delta_ms": abs(total - feature_state_ms - 9.0),
        "projective_total_to_chunk_render_ratio": total / (total + 1.0),
    }


def _profile(repeat_index: int, policy: str) -> dict[str, object]:
    cadence_no_first_ms = 200.0 + repeat_index
    measured_scale = 0.8 + 0.05 * repeat_index
    feature_state_ms = 10.0 + repeat_index
    if policy == "measured":
        feature_state_ms *= 1.2
    total = feature_state_ms + 9.0
    return {
        "scene_id": "Bq4rmeIvJbs_seg_000",
        "frames": DEFAULT_FRAMES,
        "policy": policy,
        "repeat_index": repeat_index,
        "case_json": f"case-{repeat_index}-{policy}.json",
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
        "no_first_step_ms": cadence_no_first_ms if policy == "cadence" else cadence_no_first_ms * measured_scale,
        "mean_render_forward_ms": total * 0.5,
    }


def _report(repeats: int = 2) -> dict[str, object]:
    profiles = [
        _profile(repeat_index, policy)
        for repeat_index in range(repeats)
        for policy in ("cadence", "measured")
    ]
    report: dict[str, object] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_real_video_multiscene_bq4_trace_repeat_stability",
        "base_domain": "Bq4 16f traced repeat stability for live-update projective interval substeps",
        "theory_contract": "repeat stability for feature-state-update timing.",
        "source_shape_status": "ok",
        "frames": DEFAULT_FRAMES,
        "requested_repeat_count": repeats,
        "trace_spec": {"scene_id": "Bq4rmeIvJbs_seg_000", "frames": DEFAULT_FRAMES, "trace_global_step": 3},
        "rows": [{"scene_id": "Bq4rmeIvJbs_seg_000"} for _ in profiles],
        "trace_profiles": profiles,
    }
    report["summary"] = summarize(report)
    return report


def test_bq4_trace_repeat_stability_verifier_accepts_repeat_pairs() -> None:
    report = _report()

    assert verify_bq4_trace_repeat_stability_report(report) == []
    assert_bq4_trace_repeat_stability_report(report)
    assert report["summary"]["paired_repeat_count"] == 2
    assert report["summary"]["feature_state_update_bump_count"] == 2


def test_bq4_trace_repeat_stability_verifier_rejects_stale_summary() -> None:
    report = _report()
    report["trace_profiles"][0]["chunk_profiles"][0]["projective_interval_timing_ms"][
        "feature_state_update_ms"
    ] = 99.0

    errors = verify_bq4_trace_repeat_stability_report(report)

    assert any("summary" in error and "mismatch" in error for error in errors)


def test_bq4_trace_repeat_stability_verifier_rejects_missing_pair() -> None:
    report = _report()
    report["trace_profiles"] = report["trace_profiles"][:-1]
    report["summary"] = summarize(report)

    errors = verify_bq4_trace_repeat_stability_report(report)

    assert any("missing repeat 1 measured" in error for error in errors)
    assert any("every requested repeat" in error for error in errors)


def test_bq4_trace_repeat_stability_verifier_rejects_dirty_cache_support() -> None:
    report = _report()
    report["trace_profiles"][0]["projective_interval_cache_stale_refreshes"] = 1
    report["summary"] = summarize(report)

    errors = verify_bq4_trace_repeat_stability_report(report)

    assert any("cache/support clean" in error for error in errors)


def test_saved_bq4_trace_repeat_stability_artifact_satisfies_contract() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(Path(summary_json).read_text(encoding="utf-8"))

    assert_bq4_trace_repeat_stability_report(report)
