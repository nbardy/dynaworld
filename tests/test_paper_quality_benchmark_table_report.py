from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_experiments.paper_runner_suite.paper_quality_benchmark_table_report import (
    REQUIRED_REPRESENTATIONS,
    assert_paper_quality_benchmark_table_report,
    build_report,
    summarize,
    verify_paper_quality_benchmark_table_report,
)


SAVED_QUALITY_REPORT = Path("outputs/benchmarks/2026-07-08_paper_quality_benchmark_table/summary.json")


def _valid_report() -> dict[str, object]:
    return build_report()


def _refresh_summary(report: dict[str, object]) -> None:
    rows = report["rows"]
    missing_rows = report["missing_rows"]
    assert isinstance(rows, list)
    assert isinstance(missing_rows, list)
    report["summary"] = summarize(rows, missing_rows)


def test_paper_quality_benchmark_table_accepts_current_artifacts() -> None:
    report = _valid_report()

    assert verify_paper_quality_benchmark_table_report(report) == []
    assert_paper_quality_benchmark_table_report(report)


def test_paper_quality_benchmark_table_is_ready_but_scoped() -> None:
    report = _valid_report()
    summary = report["summary"]
    assert isinstance(summary, dict)

    assert report["status"] == "ok"
    assert summary["paper_ready"] is True
    assert summary["benchmark_scope"] == "capacity_128_local_video_smoke"
    assert summary["missing_ids"] == []
    rows = report["rows"]
    assert isinstance(rows, list)
    assert {row["representation"] for row in rows if isinstance(row, dict)} == set(REQUIRED_REPRESENTATIONS)
    for row in rows:
        assert isinstance(row, dict)
        assert row["frame_count"] == 16
        assert row["render_size"] == 128
        assert row["media_psnr_mean"] > 0.0
        assert 0.0 <= row["media_l1_mean"] <= 1.0


def test_paper_quality_benchmark_table_rejects_stale_summary() -> None:
    report = _valid_report()
    summary = report["summary"]
    assert isinstance(summary, dict)
    summary["green_row_count"] = 0

    errors = verify_paper_quality_benchmark_table_report(report)

    assert any("summary green_row_count mismatch" in error for error in errors)


def test_paper_quality_benchmark_table_rejects_missing_representation() -> None:
    report = _valid_report()
    rows = report["rows"]
    assert isinstance(rows, list)
    report["rows"] = [
        row for row in rows if isinstance(row, dict) and row.get("representation") != "dynamic_3dgs_fast_mac"
    ]
    _refresh_summary(report)

    errors = verify_paper_quality_benchmark_table_report(report)

    assert any("missing representation row dynamic_3dgs_fast_mac" in error for error in errors)
    assert any("summary paper_ready must be true" in error for error in errors)


def test_paper_quality_benchmark_table_rejects_missing_media_artifact() -> None:
    report = _valid_report()
    rows = report["rows"]
    assert isinstance(rows, list)
    first = rows[0]
    assert isinstance(first, dict)
    first["side_by_side_video"] = "outputs/visual_comparisons/does_not_exist.mp4"
    _refresh_summary(report)

    errors = verify_paper_quality_benchmark_table_report(report)

    assert any("side_by_side_video must exist" in error for error in errors)


def test_paper_quality_benchmark_table_rejects_hidden_missing_rows() -> None:
    report = _valid_report()
    report["missing_rows"] = [
        {
            "missing_id": "dynamic_3dgs_fast_mac_paper_quality_row",
            "representation": "dynamic_3dgs_fast_mac",
            "needed_for": "matched paper-quality benchmark table",
        }
    ]
    _refresh_summary(report)

    errors = verify_paper_quality_benchmark_table_report(report)

    assert any("missing_rows must be empty" in error for error in errors)
    assert any("summary paper_ready must be true" in error for error in errors)


def test_saved_paper_quality_benchmark_table_report_satisfies_contract() -> None:
    if not SAVED_QUALITY_REPORT.exists():
        pytest.skip(f"missing optional saved artifact: {SAVED_QUALITY_REPORT}")

    report = json.loads(SAVED_QUALITY_REPORT.read_text(encoding="utf-8"))

    assert_paper_quality_benchmark_table_report(report)
