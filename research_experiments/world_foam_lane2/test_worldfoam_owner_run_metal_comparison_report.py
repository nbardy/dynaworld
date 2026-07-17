from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_experiments.world_foam_lane2.worldfoam_owner_run_metal_comparison_report import (
    assert_worldfoam_owner_run_metal_comparison_report,
    build_report,
    summarize,
    verify_worldfoam_owner_run_metal_comparison_report,
)


SAVED_REPORT = Path("outputs/benchmarks/2026-07-08_worldfoam_owner_run_metal_comparison_report/summary.json")


def _valid_report() -> dict[str, object]:
    return build_report()


def _row(report: dict[str, object], row_id: str) -> dict[str, object]:
    rows = report["rows"]
    assert isinstance(rows, list)
    for row in rows:
        assert isinstance(row, dict)
        if row["row_id"] == row_id:
            return row
    raise AssertionError(f"missing row {row_id}")


def _refresh_summary(report: dict[str, object]) -> None:
    rows = report["rows"]
    assert isinstance(rows, list)
    report["summary"] = summarize(rows)


def test_worldfoam_owner_run_metal_comparison_accepts_current_artifacts() -> None:
    report = _valid_report()

    assert verify_worldfoam_owner_run_metal_comparison_report(report) == []
    assert_worldfoam_owner_run_metal_comparison_report(report)


def test_worldfoam_owner_run_metal_comparison_rejects_stale_summary() -> None:
    report = _valid_report()
    summary = report["summary"]
    assert isinstance(summary, dict)
    summary["green_row_count"] = 0

    errors = verify_worldfoam_owner_run_metal_comparison_report(report)

    assert any("summary green_row_count mismatch" in error for error in errors)


def test_worldfoam_owner_run_metal_comparison_rejects_failed_metal_lane() -> None:
    report = _valid_report()
    metal = _row(report, "worldfoam_metal_capacity_lane")
    metal["status"] = "failed"
    _refresh_summary(report)

    errors = verify_worldfoam_owner_run_metal_comparison_report(report)

    assert any("worldfoam_metal_capacity_lane status must be ok" in error for error in errors)
    assert any("summary owner_run_metal_comparison_rows_ok must be true" in error for error in errors)


def test_worldfoam_owner_run_metal_comparison_rejects_missing_bridge() -> None:
    report = _valid_report()
    rows = report["rows"]
    assert isinstance(rows, list)
    report["rows"] = [
        row for row in rows if isinstance(row, dict) and row["row_id"] != "owner_run_contract_to_metal_bridge"
    ]
    _refresh_summary(report)

    errors = verify_worldfoam_owner_run_metal_comparison_report(report)

    assert any("missing row owner_run_contract_to_metal_bridge" in error for error in errors)


def test_worldfoam_owner_run_metal_comparison_rejects_bad_optical_grad() -> None:
    report = _valid_report()
    optical = _row(report, "optical_transfer_contract")
    optical["grad_error"] = 1.0
    _refresh_summary(report)

    errors = verify_worldfoam_owner_run_metal_comparison_report(report)

    assert any("optical_transfer_contract grad_error must be <= 1e-6" in error for error in errors)


def test_saved_worldfoam_owner_run_metal_comparison_satisfies_contract() -> None:
    if not SAVED_REPORT.exists():
        pytest.skip(f"missing optional saved artifact: {SAVED_REPORT}")

    report = json.loads(SAVED_REPORT.read_text(encoding="utf-8"))

    assert_worldfoam_owner_run_metal_comparison_report(report)
