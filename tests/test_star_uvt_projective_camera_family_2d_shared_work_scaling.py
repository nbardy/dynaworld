from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path

import pytest

from research_experiments.star_uvt_feature_tubes.projective_camera_family_2d_shared_work_scaling import (
    DEFAULT_OUT_DIR,
    assert_camera_family_2d_shared_work_scaling_report,
    run_report,
    summarize,
    verify_camera_family_2d_shared_work_scaling_report,
)


@lru_cache(maxsize=1)
def _cached_valid_report() -> dict[str, object]:
    return run_report(q_axis_counts=(1, 2, 4))


def _valid_report() -> dict[str, object]:
    return copy.deepcopy(_cached_valid_report())


def _row(report: dict[str, object], route: str, q_axis_count: int) -> dict[str, object]:
    rows = report["rows"]
    assert isinstance(rows, list)
    for row in rows:
        assert isinstance(row, dict)
        if row["route"] == route and row["q_axis_count"] == q_axis_count:
            return row
    raise AssertionError(f"missing {route} q_axis_count={q_axis_count}")


def test_camera_family_2d_shared_work_scaling_accepts_valid_report() -> None:
    report = _valid_report()

    assert verify_camera_family_2d_shared_work_scaling_report(report) == []
    assert_camera_family_2d_shared_work_scaling_report(report)
    assert report["summary"]["family_payload_growth"] == 1.0
    assert report["summary"]["final_payload_ratio"] < 0.30


def test_camera_family_2d_shared_work_scaling_rejects_missing_theory_contract() -> None:
    report = _valid_report()
    report["theory_contract"] = "too vague"

    errors = verify_camera_family_2d_shared_work_scaling_report(report)

    assert any("Q2 x Omega x T" in error for error in errors)


def test_camera_family_2d_shared_work_scaling_rejects_wrong_pair_count() -> None:
    report = _valid_report()
    _row(report, "family_chart", 4)["q_pair_count"] = 4
    report["summary"] = summarize(report["rows"])

    errors = verify_camera_family_2d_shared_work_scaling_report(report)

    assert any("q_pair_count must equal" in error for error in errors)


def test_camera_family_2d_shared_work_scaling_rejects_growing_family_payload() -> None:
    report = _valid_report()
    _row(report, "family_chart", 4)["payload_bytes"] = _row(report, "per_q_replay", 4)["payload_bytes"]
    report["summary"] = summarize(report["rows"])

    errors = verify_camera_family_2d_shared_work_scaling_report(report)

    assert any("payload_bytes mismatch" in error for error in errors)
    assert any("family payload growth" in error for error in errors)


def test_camera_family_2d_shared_work_scaling_rejects_lost_2d_replay_growth() -> None:
    report = _valid_report()
    for row in report["rows"]:
        if row["route"] == "per_q_replay":
            row["q_axis_count"] = 1
            row["q_pair_count"] = 1
    report["summary"] = summarize(report["rows"])

    errors = verify_camera_family_2d_shared_work_scaling_report(report)

    assert any("q_axis_counts" in error for error in errors)


def test_camera_family_2d_shared_work_scaling_rejects_high_family_residual() -> None:
    report = _valid_report()
    _row(report, "family_chart", 4)["max_fit_uv_error_px"] = 0.75
    report["summary"] = summarize(report["rows"])

    errors = verify_camera_family_2d_shared_work_scaling_report(report)

    assert any("family Q2xT fit residual" in error for error in errors)


def test_camera_family_2d_shared_work_scaling_rejects_stale_summary() -> None:
    report = copy.deepcopy(_valid_report())
    report["rows"][0]["max_fit_uv_error_px"] = 0.45

    errors = verify_camera_family_2d_shared_work_scaling_report(report)

    assert any("summary max_family_fit_uv_error_px mismatch" in error for error in errors)


def test_saved_camera_family_2d_shared_work_scaling_artifact_satisfies_contract() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(Path(summary_json).read_text(encoding="utf-8"))

    assert_camera_family_2d_shared_work_scaling_report(report)
