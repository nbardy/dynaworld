from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_experiments.paper_runner_suite.paper_runner_table_report import (
    assert_paper_runner_table_report,
    build_report,
    summarize,
    verify_paper_runner_table_report,
)


SAVED_TABLE_REPORT = Path("outputs/benchmarks/2026-07-11_paper_runner_table_report/summary.json")


def _valid_report() -> dict[str, object]:
    return build_report()


def _refresh_summary(report: dict[str, object]) -> None:
    evidence_rows = report["evidence_rows"]
    representation_rows = report["representation_rows"]
    missing_rows = report["missing_rows"]
    assert isinstance(evidence_rows, list)
    assert isinstance(representation_rows, list)
    assert isinstance(missing_rows, list)
    report["summary"] = summarize(evidence_rows, representation_rows, missing_rows)


def test_paper_runner_table_report_accepts_current_artifacts() -> None:
    report = _valid_report()

    assert verify_paper_runner_table_report(report) == []
    assert_paper_runner_table_report(report)


def test_paper_runner_table_report_is_ready_with_real_multicam_quality_table() -> None:
    report = _valid_report()
    summary = report["summary"]
    assert isinstance(summary, dict)

    assert report["status"] == "ok"
    assert summary["paper_ready"] is True
    assert summary["missing_ids"] == []
    assert "world_tubes_real_video_media_rows" not in summary["missing_ids"]
    assert "worldfoam_owner_run_metal_comparison_rows" not in summary["missing_ids"]
    assert "paper_quality_benchmark_table" not in summary["missing_ids"]
    assert "coffee_martini_matched_3seed" not in summary["missing_ids"]
    assert summary["has_coffee_martini_matched_3seed"] is True
    representation_rows = report["representation_rows"]
    assert isinstance(representation_rows, list)
    world_tubes = next(row for row in representation_rows if isinstance(row, dict) and row["representation"] == "world_tubes_star_uvt")
    assert world_tubes["real_video_media"] == "ok"
    assert world_tubes["paper_quality_benchmark"] == "ok"
    assert world_tubes["coffee_martini_matched_3seed"] == "ok"
    assert world_tubes["paper_ready"] is True
    worldfoam = next(row for row in representation_rows if isinstance(row, dict) and row["representation"] == "worldfoam_powerfoam")
    assert worldfoam["owner_run_metal_comparison"] == "ok"
    assert worldfoam["paper_quality_benchmark"] == "ok"
    assert worldfoam["coffee_martini_matched_3seed"] == "ok"
    assert worldfoam["paper_ready"] is True
    baseline = next(row for row in representation_rows if isinstance(row, dict) and row["representation"] == "dynamic_3dgs_fast_mac")
    assert baseline["paper_quality_benchmark"] == "ok"
    assert baseline["coffee_martini_matched_3seed"] == "ok"
    assert baseline["paper_ready"] is True


def test_paper_runner_table_report_rejects_stale_summary() -> None:
    report = _valid_report()
    summary = report["summary"]
    assert isinstance(summary, dict)
    summary["green_evidence_row_count"] = 0

    errors = verify_paper_runner_table_report(report)

    assert any("summary green_evidence_row_count mismatch" in error for error in errors)


def test_paper_runner_table_report_rejects_missing_required_evidence() -> None:
    report = _valid_report()
    evidence_rows = report["evidence_rows"]
    assert isinstance(evidence_rows, list)
    report["evidence_rows"] = [
        row for row in evidence_rows if isinstance(row, dict) and row.get("evidence_id") != "worldfoam_optical_transfer_fixture"
    ]
    _refresh_summary(report)

    errors = verify_paper_runner_table_report(report)

    assert any("missing required evidence row worldfoam_optical_transfer_fixture" in error for error in errors)


def test_paper_runner_table_report_rejects_artificial_missing_rows() -> None:
    report = _valid_report()
    report["missing_rows"] = [
        {
            "missing_id": "paper_quality_benchmark_table",
            "representation": "all",
            "needed_for": "final paper ablation table across World Tubes, WorldFoam, and dynamic 3DGS",
        }
    ]
    _refresh_summary(report)

    errors = verify_paper_runner_table_report(report)

    assert any("report status must be incomplete while missing rows exist" in error for error in errors)


def test_paper_runner_table_report_rejects_fake_incomplete_status() -> None:
    report = _valid_report()
    report["status"] = "incomplete"
    summary = report["summary"]
    assert isinstance(summary, dict)
    summary["paper_ready"] = False

    errors = verify_paper_runner_table_report(report)

    assert any("report status must be ok when no missing rows remain" in error for error in errors)
    assert any("summary paper_ready mismatch" in error for error in errors)
    assert any("summary paper_ready must be true when no missing rows remain" in error for error in errors)


def test_saved_paper_runner_table_report_satisfies_contract() -> None:
    if not SAVED_TABLE_REPORT.exists():
        pytest.skip(f"missing optional saved artifact: {SAVED_TABLE_REPORT}")

    report = json.loads(SAVED_TABLE_REPORT.read_text(encoding="utf-8"))

    assert_paper_runner_table_report(report)
