from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research_experiments.star_uvt_feature_tubes.projective_real_video_multiscene_frame_scaling_matrix import (
    DEFAULT_FRAME_COUNTS,
    DEFAULT_OUT_DIR,
    assert_real_video_multiscene_frame_scaling_matrix_report,
    summarize,
    verify_real_video_multiscene_frame_scaling_matrix_report,
)
from research_experiments.star_uvt_feature_tubes.projective_real_video_multiscene_trainer_matrix import (
    DEFAULT_SEGMENT_IDS,
    DEFAULT_SEGMENTS_MANIFEST,
    _load_segments,
)


def _row(scene_id: str, *, frames: int, policy: str, support_rebins: int = 0) -> dict[str, object]:
    frame_scale = frames / 4.0
    return {
        "scene_id": scene_id,
        "youtube_id": scene_id.replace("_seg_000", ""),
        "title": f"{scene_id} title",
        "video_path": f"data/{scene_id}.mp4",
        "source_video_exists": True,
        "motion_score": 1.25,
        "scene_cut_count_in_source": 0,
        "frames": frames,
        "policy": policy,
        "elapsed_sec": 1.0,
        "pass": True,
        "steps": 4,
        "start_loss": 0.31,
        "end_loss": 0.25 + frames * 1.0e-5,
        "loss_decreased": True,
        "no_first_step_ms": (72.0 if policy == "measured" else 100.0) * (frame_scale ** 0.5),
        "mean_render_forward_ms": 40.0 * frame_scale,
        "mean_backward_ms": 35.0 * frame_scale,
        "projective_interval_cache_rebuilds": 1 if policy == "measured" else 2,
        "projective_interval_cache_live_updates": 3 if policy == "measured" else 2,
        "projective_interval_cache_staleness_checks": 3 if policy == "measured" else 2,
        "projective_interval_cache_stale_refreshes": support_rebins,
        "projective_interval_cache_support_rebins": support_rebins,
        "projective_interval_cache_visibility_stratifications": 0,
        "projective_interval_cache_fallback_marks": 0,
        "projective_interval_cache_alpha_renders": 4,
        "projective_interval_cache_max_support_tail_alpha_bound": 0.0,
        "projective_interval_cache_max_support_max_overshoot_px": 0.0,
        "projective_interval_effective_support_uv_padding": 9.0,
        "tile_overflow_sum": 0,
        "max_tile_count": 18,
    }


def _valid_report() -> dict[str, object]:
    scene_ids = ["walk_seg_000", "bike_seg_000", "forest_seg_000"]
    scenes = [
        {
            "scene_id": scene_id,
            "youtube_id": scene_id.replace("_seg_000", ""),
            "title": f"{scene_id} title",
            "video_path": f"data/{scene_id}.mp4",
            "source_video_exists": True,
            "motion_score": 1.0 + idx,
            "scene_cut_count_in_source": idx,
        }
        for idx, scene_id in enumerate(scene_ids)
    ]
    rows: list[dict[str, object]] = []
    for scene_id in scene_ids:
        for frames in DEFAULT_FRAME_COUNTS:
            rows.append(_row(scene_id, frames=frames, policy="cadence"))
            rows.append(_row(scene_id, frames=frames, policy="measured"))
    report: dict[str, object] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_real_video_multiscene_frame_scaling_matrix",
        "base_domain": "checked-in source-distinct real-video segments over frame counts",
        "theory_contract": (
            "This report does not prove broad real-scene quality acceptance. It verifies that the guarded "
            "projective-interval trainer contract holds across source-distinct real-video frame growth, "
            "broadening frame-scaling evidence beyond a single high-motion clip."
        ),
        "segments_manifest": "data/youtube_scene_distinct/candidates/segments_manifest.jsonl",
        "frame_counts": list(DEFAULT_FRAME_COUNTS),
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
        "scenes": scenes,
        "rows": rows,
    }
    report["summary"] = summarize(report)  # type: ignore[arg-type]
    return report


def test_real_video_multiscene_frame_scaling_accepts_valid_payload() -> None:
    report = _valid_report()

    assert verify_real_video_multiscene_frame_scaling_matrix_report(report) == []
    assert_real_video_multiscene_frame_scaling_matrix_report(report)
    assert report["summary"]["scene_count"] == 3  # type: ignore[index]
    assert report["summary"]["frame_count_count"] == 3  # type: ignore[index]


def test_real_video_multiscene_frame_scaling_rejects_missing_frame_pair() -> None:
    report = _valid_report()
    report["rows"] = [
        row
        for row in report["rows"]  # type: ignore[index]
        if not (row["scene_id"] == "walk_seg_000" and row["frames"] == 8 and row["policy"] == "measured")
    ]

    errors = verify_real_video_multiscene_frame_scaling_matrix_report(report)

    assert any("one cadence and one measured row" in error for error in errors)


def test_real_video_multiscene_frame_scaling_rejects_duplicate_source() -> None:
    report = _valid_report()
    report["scenes"][1]["youtube_id"] = report["scenes"][0]["youtube_id"]  # type: ignore[index]
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_real_video_multiscene_frame_scaling_matrix_report(report)

    assert any("source-distinct" in error for error in errors)


def test_real_video_multiscene_frame_scaling_rejects_support_rebin() -> None:
    report = _valid_report()
    report["rows"][5]["projective_interval_cache_support_rebins"] = 1  # type: ignore[index]
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_real_video_multiscene_frame_scaling_matrix_report(report)

    assert any("measured support rebins must be 0 under guard" in error for error in errors)
    assert any("zero support rebins" in error for error in errors)


def test_real_video_multiscene_frame_scaling_rejects_frame_growth_regression() -> None:
    report = _valid_report()
    for row in report["rows"]:  # type: ignore[index]
        if row["policy"] == "measured" and row["frames"] == 16:
            row["no_first_step_ms"] = 400.0
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_real_video_multiscene_frame_scaling_matrix_report(report)

    assert any("no-first timing growth" in error for error in errors)


def test_real_video_multiscene_frame_scaling_rejects_stale_summary_after_payload_change() -> None:
    report = copy.deepcopy(_valid_report())
    report["rows"][0]["tile_overflow_sum"] = 1  # type: ignore[index]

    errors = verify_real_video_multiscene_frame_scaling_matrix_report(report)

    assert any("summary" in error and "mismatch" in error for error in errors)


def test_default_segment_ids_exist_for_frame_scaling_manifest() -> None:
    if not DEFAULT_SEGMENTS_MANIFEST.exists():
        pytest.skip(f"missing segment manifest: {DEFAULT_SEGMENTS_MANIFEST}")

    segments = _load_segments(DEFAULT_SEGMENTS_MANIFEST)

    assert all(segment_id in segments for segment_id in DEFAULT_SEGMENT_IDS)
    assert len({segments[segment_id]["youtube_id"] for segment_id in DEFAULT_SEGMENT_IDS}) == len(DEFAULT_SEGMENT_IDS)


def test_saved_real_video_multiscene_frame_scaling_artifact_satisfies_contract() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(Path(summary_json).read_text(encoding="utf-8"))

    assert_real_video_multiscene_frame_scaling_matrix_report(report)
