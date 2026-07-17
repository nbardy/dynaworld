from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research_experiments.star_uvt_feature_tubes.projective_real_video_guarded_support_matrix_report import (
    DEFAULT_OUT_DIR,
    assert_real_video_guarded_support_matrix_report,
    run_report,
    summarize,
    verify_real_video_guarded_support_matrix_report,
)


def _artifact(label: str, *, is_guarded: bool, padding: float) -> dict[str, object]:
    return {
        "path": f"{label}.json",
        "label": label,
        "is_guarded": is_guarded,
        "status": "ok",
        "source_video": "data/high_motion.mp4",
        "source_video_exists": True,
        "frame_counts": [4, 8, 16],
        "steps": 4,
        "tile_capacity": 128,
        "support_guard_padding": padding,
        "support_guard_policy": "slack_budgeted" if is_guarded else "fixed",
        "support_stale_tail_alpha_epsilon": 0.001 if is_guarded else 0.0,
        "verifier_errors": [],
        "guarded_verifier_errors": [],
    }


def _row(label: str, *, is_guarded: bool, frames: int, support_rebins: int) -> dict[str, object]:
    return {
        "artifact_label": label,
        "is_guarded": is_guarded,
        "frames": frames,
        "pass": True,
        "loss_decreased": True,
        "end_loss": 0.25,
        "cadence_end_loss": 0.25,
        "end_loss_delta_vs_cadence": 0.0,
        "no_first_step_ms": 80.0,
        "cadence_no_first_step_ms": 100.0,
        "no_first_ratio_vs_cadence": 0.8,
        "mean_backward_ms": 40.0,
        "mean_render_forward_ms": 30.0,
        "cache_rebuilds": 1,
        "cadence_cache_rebuilds": 2,
        "cache_rebuild_ratio_vs_cadence": 0.5,
        "cache_live_updates": 3,
        "cache_staleness_checks": 3,
        "cache_stale_refreshes": 0 if is_guarded else support_rebins,
        "cache_support_rebins": support_rebins,
        "cache_visibility_stratifications": 0,
        "cache_fallback_marks": 0,
        "support_max_overshoot_px": 0.0,
        "support_tail_alpha_bound": 0.0,
        "effective_support_uv_padding": 8.0 + (0.25 if is_guarded else 0.0),
        "tile_overflow_sum": 0,
        "max_tile_count": 18,
    }


def _valid_report() -> dict[str, object]:
    artifacts = [_artifact("unguarded", is_guarded=False, padding=0.0)]
    artifacts.extend(_artifact(f"guard{idx}", is_guarded=True, padding=padding) for idx, padding in enumerate((0.25, 0.5, 1.0, 2.0), start=1))
    rows: list[dict[str, object]] = []
    for artifact in artifacts:
        label = str(artifact["label"])
        is_guarded = bool(artifact["is_guarded"])
        for frames in (4, 8, 16):
            rows.append(_row(label, is_guarded=is_guarded, frames=frames, support_rebins=0 if is_guarded else 3))
    report: dict[str, object] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_real_video_guarded_support_matrix",
        "base_domain": "checked-in high-motion real-video trainer guarded support matrix",
        "theory_contract": (
            "This report does not prove broad real-scene quality acceptance. It verifies that guarded support "
            "certificates on the real-video trainer route remove support rebins/stale refreshes while preserving "
            "the measured live-cache reuse contract."
        ),
        "artifacts": artifacts,
        "rows": rows,
    }
    report["summary"] = summarize(report)  # type: ignore[arg-type]
    return report


def test_real_video_guarded_support_matrix_accepts_valid_payload() -> None:
    report = _valid_report()

    assert verify_real_video_guarded_support_matrix_report(report) == []
    assert_real_video_guarded_support_matrix_report(report)
    assert report["summary"]["default_measured_support_rebins"] > 0  # type: ignore[index]
    assert report["summary"]["guarded_measured_support_rebins"] == 0  # type: ignore[index]


def test_real_video_guarded_support_matrix_rejects_missing_scope() -> None:
    report = _valid_report()
    report["theory_contract"] = "support is solved"

    errors = verify_real_video_guarded_support_matrix_report(report)

    assert any("guarded real-video support scope" in error for error in errors)


def test_real_video_guarded_support_matrix_rejects_guarded_verifier_error() -> None:
    report = _valid_report()
    report["artifacts"][1]["guarded_verifier_errors"] = ["support rebin"]  # type: ignore[index]
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_real_video_guarded_support_matrix_report(report)

    assert any("guarded input artifact" in error for error in errors)
    assert any("all guarded support artifact verifiers must pass" in error for error in errors)


def test_real_video_guarded_support_matrix_rejects_missing_unguarded_churn() -> None:
    report = _valid_report()
    for row in report["rows"]:  # type: ignore[index]
        row["cache_support_rebins"] = 0
        row["cache_stale_refreshes"] = 0
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_real_video_guarded_support_matrix_report(report)

    assert any("unguarded baseline must expose support rebins" in error for error in errors)


def test_real_video_guarded_support_matrix_rejects_guarded_support_rebin() -> None:
    report = _valid_report()
    report["rows"][3]["cache_support_rebins"] = 1  # type: ignore[index]
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_real_video_guarded_support_matrix_report(report)

    assert any("guarded measured support rebins must be zero" in error for error in errors)
    assert any("must eliminate measured support rebins" in error for error in errors)


def test_real_video_guarded_support_matrix_rejects_timing_regression() -> None:
    report = _valid_report()
    report["rows"][4]["no_first_ratio_vs_cadence"] = 1.25  # type: ignore[index]
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_real_video_guarded_support_matrix_report(report)

    assert any("measured no-first timing must beat cadence" in error for error in errors)


def test_real_video_guarded_support_matrix_rejects_stale_summary_after_payload_change() -> None:
    report = copy.deepcopy(_valid_report())
    report["rows"][0]["cache_support_rebins"] = 99  # type: ignore[index]

    errors = verify_real_video_guarded_support_matrix_report(report)

    assert any("summary" in error and "mismatch" in error for error in errors)


def test_real_video_guarded_support_matrix_run_report() -> None:
    report = run_report()

    assert_real_video_guarded_support_matrix_report(report)
    assert report["summary"]["guarded_artifact_count"] == 4
    assert report["summary"]["guarded_measured_support_rebins"] == 0


def test_saved_real_video_guarded_support_matrix_artifact_satisfies_contract() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(Path(summary_json).read_text(encoding="utf-8"))

    assert_real_video_guarded_support_matrix_report(report)
