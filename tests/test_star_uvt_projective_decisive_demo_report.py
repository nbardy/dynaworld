from __future__ import annotations

from pathlib import Path

import pytest

from research_experiments.star_uvt_feature_tubes.projective_decisive_demo_report import (
    assert_projective_decisive_demo_report,
    run_report,
    summarize,
    verify_projective_decisive_demo_report,
)


SAVED_DECISIVE_DEMO_ARTIFACT = Path(
    "outputs/benchmarks/2026-07-08_star_uvt_projective_decisive_demo_fixture/summary.json"
)
SAVED_WORLDTUBES_MEDIA_SOURCE = Path(
    "outputs/visual_comparisons/star_uvt_worldtubes_metal_128_16f_60step_2048tubes.json"
)


def _valid_report() -> dict[str, object]:
    return run_report(frames=4, image_size=8, tile_size=8)


def _row(report: dict[str, object], route: str) -> dict[str, object]:
    rows = report["rows"]
    assert isinstance(rows, list)
    for raw_row in rows:
        assert isinstance(raw_row, dict)
        if raw_row["route"] == route:
            return raw_row
    raise AssertionError(f"missing route {route}")


def _refresh_summary(report: dict[str, object]) -> None:
    rows = report["rows"]
    assert isinstance(rows, list)
    report["summary"] = summarize(rows)


def test_projective_decisive_demo_runner_satisfies_contract() -> None:
    report = _valid_report()

    assert verify_projective_decisive_demo_report(report) == []
    assert_projective_decisive_demo_report(report)


def test_projective_decisive_demo_verifier_rejects_stale_summary() -> None:
    report = _valid_report()
    summary = report["summary"]
    assert isinstance(summary, dict)
    summary["compiled_to_replay_interval_entry_ratio"] = 1.0

    errors = verify_projective_decisive_demo_report(report)

    assert any("summary compiled_to_replay_interval_entry_ratio mismatch" in error for error in errors)


def test_projective_decisive_demo_verifier_rejects_hidden_fallback() -> None:
    report = _valid_report()
    compiled = _row(report, "compiled_interval_atlas")
    compiled["fallback_sample_fraction"] = 0.25
    _refresh_summary(report)

    errors = verify_projective_decisive_demo_report(report)

    assert any("compiled_interval_atlas must be fallback-free" in error for error in errors)
    assert any("summary must report all_rows_fallback_free true" in error for error in errors)


def test_projective_decisive_demo_verifier_rejects_quality_regression() -> None:
    report = _valid_report()
    compiled = _row(report, "compiled_interval_atlas")
    compiled["max_image_abs_error_vs_reference"] = 0.125
    compiled["mean_image_abs_error_vs_reference"] = 0.01
    compiled["psnr_vs_reference"] = 12.0
    _refresh_summary(report)

    errors = verify_projective_decisive_demo_report(report)

    assert any("compiled_interval_atlas image error exceeds" in error for error in errors)
    assert any("compiled_interval_atlas psnr_vs_reference below" in error for error in errors)
    assert any("summary must report all_rows_quality_pass true" in error for error in errors)


def test_projective_decisive_demo_verifier_rejects_missing_replay_route() -> None:
    report = _valid_report()
    rows = report["rows"]
    assert isinstance(rows, list)
    report["rows"] = [row for row in rows if isinstance(row, dict) and row["route"] != "per_frame_replay"]
    _refresh_summary(report)

    errors = verify_projective_decisive_demo_report(report)

    assert any("rows must include per_frame_replay" in error for error in errors)


def test_projective_decisive_demo_verifier_rejects_missing_media_artifacts() -> None:
    report = _valid_report()
    report["mode"] = "real_video_media"
    report["requires_media_artifacts"] = True
    report["artifacts"] = {"contact_sheet": "contact.png"}
    _refresh_summary(report)

    errors = verify_projective_decisive_demo_report(report)

    assert any("media report missing artifact path: runtime_bars" in error for error in errors)
    assert any("media report missing artifact path: memory_bars" in error for error in errors)


def test_projective_decisive_demo_saved_real_video_media_mode(tmp_path: Path) -> None:
    if not SAVED_WORLDTUBES_MEDIA_SOURCE.exists():
        pytest.skip(f"missing saved media source: {SAVED_WORLDTUBES_MEDIA_SOURCE}")

    report = run_report(
        frames=4,
        image_size=8,
        tile_size=8,
        include_saved_real_video=True,
        media_artifact_dir=tmp_path,
        saved_real_video_summary=SAVED_WORLDTUBES_MEDIA_SOURCE,
    )

    assert report["mode"] == "real_video_media"
    assert report["requires_media_artifacts"] is True
    assert verify_projective_decisive_demo_report(report) == []
    assert_projective_decisive_demo_report(report)
    summary = report["summary"]
    assert isinstance(summary, dict)
    assert summary["has_real_video_media_rows"] is True
    assert summary["real_video_media_rows_ok"] is True
    media_row = _row(report, "real_video_media")
    assert media_row["final_psnr"] > 0.0
    assert media_row["artifact_count"] >= 5


def test_projective_decisive_demo_verifier_rejects_missing_media_file(tmp_path: Path) -> None:
    if not SAVED_WORLDTUBES_MEDIA_SOURCE.exists():
        pytest.skip(f"missing saved media source: {SAVED_WORLDTUBES_MEDIA_SOURCE}")
    report = run_report(
        frames=4,
        image_size=8,
        tile_size=8,
        include_saved_real_video=True,
        media_artifact_dir=tmp_path,
        saved_real_video_summary=SAVED_WORLDTUBES_MEDIA_SOURCE,
    )
    artifacts = report["artifacts"]
    assert isinstance(artifacts, dict)
    artifacts["runtime_bars"] = str(tmp_path / "missing_runtime.svg")
    media_row = _row(report, "real_video_media")
    media_artifacts = media_row["artifacts"]
    assert isinstance(media_artifacts, dict)
    media_artifacts["runtime_bars"] = str(tmp_path / "missing_runtime.svg")
    _refresh_summary(report)

    errors = verify_projective_decisive_demo_report(report)

    assert any("media report artifact does not exist: runtime_bars" in error for error in errors)
    assert any("real_video_media row artifact does not exist: runtime_bars" in error for error in errors)


def test_saved_projective_decisive_demo_artifact_satisfies_contract() -> None:
    if not SAVED_DECISIVE_DEMO_ARTIFACT.exists():
        pytest.skip(f"missing optional saved artifact: {SAVED_DECISIVE_DEMO_ARTIFACT}")

    import json

    report = json.loads(SAVED_DECISIVE_DEMO_ARTIFACT.read_text(encoding="utf-8"))

    assert_projective_decisive_demo_report(report)
