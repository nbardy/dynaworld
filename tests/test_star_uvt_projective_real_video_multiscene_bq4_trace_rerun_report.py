from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_experiments.star_uvt_feature_tubes.projective_real_video_multiscene_bq4_trace_rerun_report import (
    DEFAULT_OUT_DIR,
    assert_bq4_trace_rerun_report,
    summarize,
    verify_bq4_trace_rerun_report,
)


def _chunk_profile(total: float = 10.0) -> dict[str, object]:
    timing = {
        "feature_state_update_ms": 1.0,
        "feature_render_ms": 2.0,
        "alpha_state_update_ms": 3.0,
        "alpha_render_ms": 4.0,
        "projective_interval_render_ms": total,
    }
    return {
        "frame_start": 0,
        "chunk_frames": 4,
        "render_forward_ms": 11.0,
        "projective_interval_timing_ms": timing,
        "subtiming_sum_ms": 10.0,
        "subtiming_total_abs_delta_ms": abs(total - 10.0),
        "projective_total_to_chunk_render_ratio": total / 11.0,
    }


def _profile(frames: int, policy: str) -> dict[str, object]:
    cadence_no_first_ms = 100.0 if frames == 4 else 200.0
    no_first_ms = cadence_no_first_ms if policy == "cadence" else cadence_no_first_ms * 0.75
    return {
        "scene_id": "Bq4rmeIvJbs_seg_000",
        "frames": frames,
        "policy": policy,
        "case_json": f"case-{frames}-{policy}.json",
        "expected_trace_global_step": 1 if frames == 4 else 3,
        "chunk_trace_global_steps": [1 if frames == 4 else 3],
        "matching_trace_count": 1,
        "chunk_profile_count": 1,
        "chunk_profiles": [_chunk_profile()],
        "projective_interval_cache_rebuilds": 2 if policy == "cadence" else 1,
        "projective_interval_cache_live_updates": 2 if policy == "cadence" else 3,
        "projective_interval_cache_support_rebins": 0,
        "projective_interval_cache_stale_refreshes": 0,
        "projective_interval_cache_fallback_marks": 0,
        "tile_overflow_sum": 0,
        "no_first_step_ms": no_first_ms,
        "mean_render_forward_ms": no_first_ms * 0.5,
    }


def _report() -> dict[str, object]:
    profiles = [
        _profile(4, "cadence"),
        _profile(4, "measured"),
        _profile(16, "cadence"),
        _profile(16, "measured"),
    ]
    report: dict[str, object] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_real_video_multiscene_bq4_trace_rerun",
        "base_domain": "Bq4 spike-step traced rerun selected from saved render-forward shape artifact",
        "theory_contract": "Bq4 trace_global_steps substep attribution diagnostic.",
        "source_shape_status": "ok",
        "source_shape_summary": {"pair_count": 15},
        "trace_specs": [
            {"scene_id": "Bq4rmeIvJbs_seg_000", "frames": 4, "trace_global_step": 1},
            {"scene_id": "Bq4rmeIvJbs_seg_000", "frames": 16, "trace_global_step": 3},
        ],
        "rows": [{"scene_id": "Bq4rmeIvJbs_seg_000"} for _ in profiles],
        "trace_profiles": profiles,
    }
    report["summary"] = summarize(report)
    return report


def test_bq4_trace_rerun_verifier_accepts_substep_timing_report() -> None:
    report = _report()

    assert verify_bq4_trace_rerun_report(report) == []
    assert_bq4_trace_rerun_report(report)
    assert report["summary"]["all_expected_global_steps_traced"] is True
    assert report["summary"]["all_projective_interval_timing_present"] is True


def test_bq4_trace_rerun_verifier_rejects_stale_summary() -> None:
    report = _report()
    report["trace_profiles"][0]["chunk_profiles"][0]["projective_interval_timing_ms"]["feature_render_ms"] = 99.0

    errors = verify_bq4_trace_rerun_report(report)

    assert any("summary" in error and "mismatch" in error for error in errors)


def test_bq4_trace_rerun_verifier_rejects_missing_timing() -> None:
    report = _report()
    del report["trace_profiles"][0]["chunk_profiles"][0]["projective_interval_timing_ms"]["alpha_render_ms"]

    errors = verify_bq4_trace_rerun_report(report)

    assert any("alpha_render_ms" in error for error in errors)


def test_bq4_trace_rerun_verifier_rejects_dirty_cache_support() -> None:
    report = _report()
    report["trace_profiles"][0]["projective_interval_cache_support_rebins"] = 1
    report["summary"] = summarize(report)

    errors = verify_bq4_trace_rerun_report(report)

    assert any("cache/support clean" in error for error in errors)


def test_saved_bq4_trace_rerun_artifact_satisfies_contract() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(Path(summary_json).read_text(encoding="utf-8"))

    assert_bq4_trace_rerun_report(report)
