from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research_experiments.star_uvt_feature_tubes.projective_interval_trainer_frame_scaling_benchmark import (
    DEFAULT_OUT_DIR,
    assert_interval_trainer_frame_scaling_report,
    summarize,
    verify_interval_trainer_frame_scaling_report,
)


def _row(
    *,
    frames: int,
    policy: str,
    rebuilds: int,
    live_updates: int,
    staleness_checks: int,
    stale_refreshes: int,
    support_rebins: int,
    start_loss: float,
    end_loss: float,
    no_first_step_ms: float,
) -> dict[str, object]:
    return {
        "frames": frames,
        "policy": policy,
        "pass": True,
        "steps": 4,
        "start_loss": start_loss,
        "end_loss": end_loss,
        "loss_decreased": True,
        "no_first_step_ms": no_first_step_ms,
        "mean_render_forward_ms": 50.0,
        "mean_backward_ms": 25.0,
        "projective_interval_cache_rebuilds": rebuilds,
        "projective_interval_cache_live_updates": live_updates,
        "projective_interval_cache_staleness_checks": staleness_checks,
        "projective_interval_cache_stale_refreshes": stale_refreshes,
        "projective_interval_cache_support_rebins": support_rebins,
        "projective_interval_cache_visibility_stratifications": 0,
        "projective_interval_cache_fallback_marks": 0,
        "tile_overflow_sum": 0,
        "max_tile_count": 4,
    }


def _valid_report() -> dict[str, object]:
    rows = [
        _row(
            frames=4,
            policy="cadence",
            rebuilds=2,
            live_updates=2,
            staleness_checks=2,
            stale_refreshes=0,
            support_rebins=0,
            start_loss=0.26,
            end_loss=0.25,
            no_first_step_ms=300.0,
        ),
        _row(
            frames=4,
            policy="measured",
            rebuilds=1,
            live_updates=3,
            staleness_checks=3,
            stale_refreshes=0,
            support_rebins=0,
            start_loss=0.26,
            end_loss=0.25,
            no_first_step_ms=100.0,
        ),
        _row(
            frames=8,
            policy="cadence",
            rebuilds=2,
            live_updates=2,
            staleness_checks=2,
            stale_refreshes=0,
            support_rebins=0,
            start_loss=0.28,
            end_loss=0.27,
            no_first_step_ms=200.0,
        ),
        _row(
            frames=8,
            policy="measured",
            rebuilds=1,
            live_updates=3,
            staleness_checks=3,
            stale_refreshes=1,
            support_rebins=1,
            start_loss=0.28,
            end_loss=0.27,
            no_first_step_ms=150.0,
        ),
    ]
    return {
        "status": "ok",
        "benchmark": "star_uvt_projective_interval_trainer_frame_scaling",
        "frame_counts": [4, 8],
        "steps": 4,
        "tile_capacity": 128,
        "summary": summarize(rows),
        "rows": rows,
    }


def _row_for(report: dict[str, object], *, frames: int, policy: str) -> dict[str, object]:
    rows = report["rows"]
    assert isinstance(rows, list)
    for raw_row in rows:
        assert isinstance(raw_row, dict)
        if raw_row["frames"] == frames and raw_row["policy"] == policy:
            return raw_row
    raise AssertionError(f"missing {policy} {frames}f row")


def _refresh_summary(report: dict[str, object]) -> None:
    rows = report["rows"]
    assert isinstance(rows, list)
    report["summary"] = summarize(rows)


def test_interval_trainer_frame_scaling_verifier_accepts_valid_payload() -> None:
    report = _valid_report()

    assert verify_interval_trainer_frame_scaling_report(report) == []
    assert_interval_trainer_frame_scaling_report(report)


def test_interval_trainer_frame_scaling_verifier_rejects_loss_mismatch() -> None:
    report = copy.deepcopy(_valid_report())
    _row_for(report, frames=8, policy="measured")["end_loss"] = 0.271
    _refresh_summary(report)

    errors = verify_interval_trainer_frame_scaling_report(report)

    assert any("end loss mismatch" in error for error in errors)


def test_interval_trainer_frame_scaling_verifier_rejects_missing_rebuild_reduction() -> None:
    report = copy.deepcopy(_valid_report())
    _row_for(report, frames=4, policy="measured")["projective_interval_cache_rebuilds"] = 2
    _refresh_summary(report)

    errors = verify_interval_trainer_frame_scaling_report(report)

    assert any("measured rebuilds must be lower" in error for error in errors)


def test_interval_trainer_frame_scaling_verifier_rejects_stale_refresh_mismatch() -> None:
    report = copy.deepcopy(_valid_report())
    _row_for(report, frames=8, policy="measured")["projective_interval_cache_support_rebins"] = 0
    _refresh_summary(report)

    errors = verify_interval_trainer_frame_scaling_report(report)

    assert any("support rebins must equal stale refreshes" in error for error in errors)


def test_interval_trainer_frame_scaling_verifier_rejects_lost_timing_win() -> None:
    report = copy.deepcopy(_valid_report())
    _row_for(report, frames=8, policy="measured")["no_first_step_ms"] = 250.0
    _refresh_summary(report)

    errors = verify_interval_trainer_frame_scaling_report(report)

    assert any("no-first-step timings must beat cadence" in error for error in errors)


def test_saved_interval_trainer_frame_scaling_artifact_satisfies_contract() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(summary_json.read_text(encoding="utf-8"))

    assert_interval_trainer_frame_scaling_report(report)
