from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research_experiments.star_uvt_feature_tubes.projective_real_video_trainer_frame_scaling_benchmark import (
    DEFAULT_OUT_DIR,
    _base_config,
    assert_guarded_real_video_trainer_support_report,
    assert_real_video_trainer_frame_scaling_report,
    summarize,
    verify_guarded_real_video_trainer_support_report,
    verify_real_video_trainer_frame_scaling_report,
)


def _row(
    *,
    frames: int,
    policy: str,
    rebuilds: int,
    live_updates: int,
    staleness_checks: int,
    end_loss: float,
) -> dict[str, object]:
    return {
        "frames": frames,
        "policy": policy,
        "pass": True,
        "steps": 4,
        "start_loss": end_loss + 0.01,
        "end_loss": end_loss,
        "loss_decreased": True,
        "no_first_step_ms": 100.0,
        "mean_render_forward_ms": 60.0,
        "mean_backward_ms": 40.0,
        "projective_interval_cache_rebuilds": rebuilds,
        "projective_interval_cache_live_updates": live_updates,
        "projective_interval_cache_staleness_checks": staleness_checks,
        "projective_interval_cache_stale_refreshes": 0,
        "projective_interval_cache_support_rebins": 0,
        "projective_interval_cache_visibility_stratifications": 0,
        "projective_interval_cache_fallback_marks": 0,
        "tile_overflow_sum": 0,
        "max_tile_count": 8,
    }


def _valid_report() -> dict[str, object]:
    rows = [
        _row(frames=4, policy="cadence", rebuilds=2, live_updates=2, staleness_checks=2, end_loss=0.25),
        _row(frames=4, policy="measured", rebuilds=1, live_updates=3, staleness_checks=3, end_loss=0.25),
        _row(frames=16, policy="cadence", rebuilds=2, live_updates=2, staleness_checks=2, end_loss=0.28),
        _row(frames=16, policy="measured", rebuilds=1, live_updates=3, staleness_checks=3, end_loss=0.28),
    ]
    return {
        "status": "ok",
        "benchmark": "star_uvt_projective_real_video_trainer_frame_scaling",
        "source_video_exists": True,
        "frame_counts": [4, 16],
        "steps": 4,
        "tile_capacity": 128,
        "summary": summarize(rows),
        "rows": rows,
    }


def _valid_guarded_report() -> dict[str, object]:
    report = _valid_report()
    report["support_guard_padding"] = 1.0
    report["support_guard_policy"] = "slack_budgeted"
    report["support_stale_overshoot_epsilon"] = 0.0
    report["support_stale_tail_alpha_epsilon"] = 1.0e-3
    return report


def test_real_video_trainer_frame_scaling_verifier_accepts_cache_reuse_report() -> None:
    report = _valid_report()

    assert verify_real_video_trainer_frame_scaling_report(report) == []
    assert_real_video_trainer_frame_scaling_report(report)


def test_guarded_real_video_trainer_support_verifier_accepts_churn_free_report() -> None:
    report = _valid_guarded_report()

    assert verify_guarded_real_video_trainer_support_report(report) == []
    assert_guarded_real_video_trainer_support_report(report)


def test_real_video_trainer_frame_scaling_config_wires_support_knobs(tmp_path: Path) -> None:
    cfg = _base_config(
        frames=16,
        size=64,
        steps=4,
        policy="measured",
        refresh_every=2,
        tile_capacity=128,
        tube_count=128,
        support_guard_padding=2.0,
        support_guard_policy="slack_budgeted",
        support_guard_bisect_steps=5,
        support_stale_overshoot_epsilon=0.0,
        support_stale_tail_alpha_epsilon=1.0e-3,
        out_json=tmp_path / "out.json",
    )

    backend = cfg["feature_uvt"]["projective_interval"]

    assert backend["support_guard_padding"] == 2.0
    assert backend["support_guard_policy"] == "slack_budgeted"
    assert backend["support_guard_bisect_steps"] == 5
    assert backend["support_stale_overshoot_epsilon"] == 0.0
    assert backend["support_stale_tail_alpha_epsilon"] == 1.0e-3


def test_real_video_trainer_frame_scaling_verifier_rejects_loss_mismatch() -> None:
    report = copy.deepcopy(_valid_report())
    rows = report["rows"]
    assert isinstance(rows, list)
    rows[3]["end_loss"] = 0.30
    report["summary"] = summarize(rows)

    errors = verify_real_video_trainer_frame_scaling_report(report)

    assert any("end loss mismatch" in error for error in errors)


def test_real_video_trainer_frame_scaling_verifier_rejects_missing_rebuild_reduction() -> None:
    report = copy.deepcopy(_valid_report())
    rows = report["rows"]
    assert isinstance(rows, list)
    rows[1]["projective_interval_cache_rebuilds"] = 2
    report["summary"] = summarize(rows)

    errors = verify_real_video_trainer_frame_scaling_report(report)

    assert any("measured rebuilds must be lower" in error for error in errors)


def test_real_video_trainer_frame_scaling_verifier_rejects_missing_live_update() -> None:
    report = copy.deepcopy(_valid_report())
    rows = report["rows"]
    assert isinstance(rows, list)
    rows[1]["projective_interval_cache_live_updates"] = 0
    report["summary"] = summarize(rows)

    errors = verify_real_video_trainer_frame_scaling_report(report)

    assert any("live cache updates" in error for error in errors)


def test_real_video_trainer_frame_scaling_verifier_rejects_fallback_marks() -> None:
    report = copy.deepcopy(_valid_report())
    rows = report["rows"]
    assert isinstance(rows, list)
    rows[1]["projective_interval_cache_fallback_marks"] = 1
    report["summary"] = summarize(rows)

    errors = verify_real_video_trainer_frame_scaling_report(report)

    assert any("fallback marks must be zero" in error for error in errors)


def test_real_video_trainer_frame_scaling_verifier_rejects_stale_refresh_mismatch() -> None:
    report = copy.deepcopy(_valid_report())
    rows = report["rows"]
    assert isinstance(rows, list)
    rows[1]["projective_interval_cache_support_rebins"] = 1
    report["summary"] = summarize(rows)

    errors = verify_real_video_trainer_frame_scaling_report(report)

    assert any("support rebins must equal stale refreshes" in error for error in errors)


def test_real_video_trainer_frame_scaling_verifier_rejects_stale_summary() -> None:
    report = copy.deepcopy(_valid_report())
    summary = report["summary"]
    assert isinstance(summary, dict)
    summary["all_measured_loss_matches_cadence"] = False

    errors = verify_real_video_trainer_frame_scaling_report(report)

    assert any("summary" in error for error in errors)


def test_guarded_real_video_trainer_support_verifier_rejects_support_rebin() -> None:
    report = copy.deepcopy(_valid_guarded_report())
    rows = report["rows"]
    assert isinstance(rows, list)
    rows[1]["projective_interval_cache_support_rebins"] = 1
    report["summary"] = summarize(rows)

    errors = verify_guarded_real_video_trainer_support_report(report)

    assert any("support rebins must be 0" in error for error in errors)


def test_guarded_real_video_trainer_support_verifier_rejects_uncertified_tail() -> None:
    report = copy.deepcopy(_valid_guarded_report())
    rows = report["rows"]
    assert isinstance(rows, list)
    rows[1]["projective_interval_cache_max_support_tail_alpha_bound"] = 0.002

    errors = verify_guarded_real_video_trainer_support_report(report)

    assert any("support tail bound" in error for error in errors)


@pytest.mark.parametrize(
    "summary_json",
    [
        DEFAULT_OUT_DIR / "summary.json",
        DEFAULT_OUT_DIR.parent / "2026-05-25_star_uvt_projective_real_video_trainer_frame_scaling_guard025_tail001" / "summary.json",
        DEFAULT_OUT_DIR.parent / "2026-05-25_star_uvt_projective_real_video_trainer_frame_scaling_guard05_tail001" / "summary.json",
        DEFAULT_OUT_DIR.parent / "2026-05-25_star_uvt_projective_real_video_trainer_frame_scaling_guard10_tail001" / "summary.json",
        DEFAULT_OUT_DIR.parent / "2026-05-25_star_uvt_projective_real_video_trainer_frame_scaling_guard20_tail001" / "summary.json",
    ],
)
def test_saved_real_video_trainer_frame_scaling_artifact_satisfies_contract(summary_json: Path) -> None:
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(summary_json.read_text(encoding="utf-8"))

    assert_real_video_trainer_frame_scaling_report(report)


@pytest.mark.parametrize(
    "summary_json",
    [
        DEFAULT_OUT_DIR.parent / "2026-05-25_star_uvt_projective_real_video_trainer_frame_scaling_guard025_tail001" / "summary.json",
        DEFAULT_OUT_DIR.parent / "2026-05-25_star_uvt_projective_real_video_trainer_frame_scaling_guard05_tail001" / "summary.json",
        DEFAULT_OUT_DIR.parent / "2026-05-25_star_uvt_projective_real_video_trainer_frame_scaling_guard10_tail001" / "summary.json",
        DEFAULT_OUT_DIR.parent / "2026-05-25_star_uvt_projective_real_video_trainer_frame_scaling_guard20_tail001" / "summary.json",
    ],
)
def test_saved_guarded_real_video_trainer_support_artifacts_satisfy_contract(summary_json: Path) -> None:
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(summary_json.read_text(encoding="utf-8"))

    assert_guarded_real_video_trainer_support_report(report)
