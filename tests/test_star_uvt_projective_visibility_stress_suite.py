from __future__ import annotations

from pathlib import Path

import pytest

from research_experiments.star_uvt_feature_tubes.projective_visibility_stress_suite import (
    assert_projective_visibility_stress_suite,
    run_report,
    summarize,
    verify_projective_visibility_stress_suite,
)


SAVED_VISIBILITY_STRESS_ARTIFACT = Path(
    "outputs/benchmarks/2026-07-08_star_uvt_projective_visibility_stress_suite/summary.json"
)


def _valid_report() -> dict[str, object]:
    return run_report(frames=4, image_size=8, tile_size=8)


def _row(report: dict[str, object], case_id: str) -> dict[str, object]:
    rows = report["rows"]
    assert isinstance(rows, list)
    for raw_row in rows:
        assert isinstance(raw_row, dict)
        if raw_row["case_id"] == case_id:
            return raw_row
    raise AssertionError(f"missing case {case_id}")


def _refresh_summary(report: dict[str, object]) -> None:
    rows = report["rows"]
    assert isinstance(rows, list)
    report["summary"] = summarize(rows)


def test_visibility_stress_accepts_valid_fixture_report() -> None:
    report = _valid_report()

    assert verify_projective_visibility_stress_suite(report) == []
    assert_projective_visibility_stress_suite(report)


def test_visibility_stress_fixture_has_clean_ambiguous_repaired_and_fallback_rows() -> None:
    report = _valid_report()

    clean = _row(report, "clean_orbit_ordered")
    raw = _row(report, "crossing_raw_interval")
    repaired = _row(report, "crossing_stratified")
    fallback = _row(report, "forced_fallback_ambiguous")

    assert clean["collapse"] is False
    assert clean["visibility_stale"] is False
    assert raw["visibility_stale"] is True
    assert raw["collapse"] is True
    assert repaired["visibility_stale"] is False
    assert repaired["collapse"] is False
    assert fallback["collapse"] is True
    assert fallback["fallback_sample_fraction"] == pytest.approx(1.0)


def test_visibility_stress_rejects_missing_collapse_boundary() -> None:
    report = _valid_report()
    rows = report["rows"]
    assert isinstance(rows, list)
    for row in rows:
        assert isinstance(row, dict)
        row["collapse"] = False
        row["collapse_reasons"] = []
        row["quality_error"] = 0.0
        row["fallback_sample_fraction"] = 0.0
        row["fallback_cell_fraction"] = 0.0
    _refresh_summary(report)

    errors = verify_projective_visibility_stress_suite(report)

    assert any("must expose a collapse boundary" in error for error in errors)
    assert any("must include at least one collapsed stress boundary" in error for error in errors)


def test_visibility_stress_rejects_unexplained_high_fallback() -> None:
    report = _valid_report()
    fallback = _row(report, "forced_fallback_ambiguous")
    fallback["collapse"] = True
    fallback["collapse_reasons"] = []
    _refresh_summary(report)

    errors = verify_projective_visibility_stress_suite(report)

    assert any("collapsed row must list collapse_reasons" in error for error in errors)
    assert any("high fallback_sample_fraction must explain collapse" in error for error in errors)


def test_visibility_stress_rejects_stale_summary() -> None:
    report = _valid_report()
    summary = report["summary"]
    assert isinstance(summary, dict)
    summary["collapsed_case_count"] = 0

    errors = verify_projective_visibility_stress_suite(report)

    assert any("summary collapsed_case_count mismatch" in error for error in errors)


def test_visibility_stress_rejects_lost_repair() -> None:
    report = _valid_report()
    repaired = _row(report, "crossing_stratified")
    repaired["visibility_stale"] = True
    _refresh_summary(report)

    errors = verify_projective_visibility_stress_suite(report)

    assert any("crossing_stratified must repair visibility_stale" in error for error in errors)
    assert any("summary must report has_repaired_crossing_case true" in error for error in errors)


def test_saved_visibility_stress_artifact_satisfies_contract() -> None:
    if not SAVED_VISIBILITY_STRESS_ARTIFACT.exists():
        pytest.skip(f"missing optional saved artifact: {SAVED_VISIBILITY_STRESS_ARTIFACT}")

    import json

    report = json.loads(SAVED_VISIBILITY_STRESS_ARTIFACT.read_text(encoding="utf-8"))

    assert_projective_visibility_stress_suite(report)
