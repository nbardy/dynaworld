from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research_experiments.star_uvt_feature_tubes.projective_real_active_set_distribution_report import (
    DEFAULT_OUT_DIR,
    assert_real_active_set_distribution_report,
    run_report,
    summarize,
    verify_real_active_set_distribution_report,
)


def _valid_report() -> dict[str, object]:
    artifacts = [
        {
            "path": f"artifact_{idx}.json",
            "label": f"artifact_{idx}",
            "status": "ok",
            "source_video": "data/high_motion.mp4",
            "source_video_exists": True,
            "frame_counts": [4, 8, 16],
            "size": 32 * (idx + 1),
            "tube_count": 64 * (idx + 1),
            "tile_capacity": 128,
            "verifier_errors": [],
        }
        for idx in range(3)
    ]
    rows: list[dict[str, object]] = []
    for idx, artifact in enumerate(artifacts):
        for frame_index, frames in enumerate((4, 8, 16)):
            active_set_groups = (idx + 1) * (frame_index + 1) * 16
            dense_tile_pairs = active_set_groups * 40
            rows.append(
                {
                    "artifact_label": artifact["label"],
                    "frames": frames,
                    "size": artifact["size"],
                    "tube_count": artifact["tube_count"],
                    "tile_capacity": artifact["tile_capacity"],
                    "trace_count": artifact["tube_count"],
                    "cell_count": active_set_groups + frame_index * 4,
                    "tile_active_set_groups": active_set_groups,
                    "max_cells_per_active_set_group": min(3, frame_index + 1),
                    "dense_per_frame_tile_pairs": dense_tile_pairs,
                    "active_set_group_to_dense_tile_pair_ratio": active_set_groups / dense_tile_pairs,
                    "cell_to_active_set_group_ratio": (active_set_groups + frame_index * 4) / active_set_groups,
                    "interval_trace_entries": active_set_groups * 4,
                    "interval_to_dense_tile_pair_ratio": 0.1,
                    "fallback_cells": 0,
                    "fallback_fraction": 0.0,
                    "fallback_reasons": [],
                    "velocity_mean_px_per_frame": 0.3,
                    "velocity_max_px_per_frame": 0.8,
                }
            )
    report: dict[str, object] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_real_active_set_distribution",
        "base_domain": "checked-in high-motion real-video projective interval atlases",
        "theory_contract": (
            "This report does not prove broad real-scene quality acceptance. It verifies that checked-in "
            "high-motion real-video projective interval atlases expose bounded, fallback-free active-set "
            "topology distributions, so active-set splitting is measured on real compiled traces rather "
            "than only synthetic q-family strata."
        ),
        "artifact_count": len(artifacts),
        "final_frame_count": 16,
        "artifacts": artifacts,
        "rows": rows,
    }
    report["summary"] = summarize(report)  # type: ignore[arg-type]
    return report


def test_real_active_set_distribution_accepts_valid_payload() -> None:
    report = _valid_report()

    assert verify_real_active_set_distribution_report(report) == []
    assert_real_active_set_distribution_report(report)
    assert report["summary"]["max_active_set_group_to_dense_tile_pair_ratio"] < 0.05  # type: ignore[index]


def test_real_active_set_distribution_rejects_missing_scope() -> None:
    report = _valid_report()
    report["theory_contract"] = "active sets are solved"

    errors = verify_real_active_set_distribution_report(report)

    assert any("real active-set distribution scope" in error for error in errors)


def test_real_active_set_distribution_rejects_underlying_verifier_error() -> None:
    report = _valid_report()
    report["artifacts"][0]["verifier_errors"] = ["stale source report"]  # type: ignore[index]
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_real_active_set_distribution_report(report)

    assert any("verifier failed" in error for error in errors)
    assert any("all input artifact verifiers must pass" in error for error in errors)


def test_real_active_set_distribution_rejects_fallback_rows() -> None:
    report = _valid_report()
    report["rows"][0]["fallback_cells"] = 1  # type: ignore[index]
    report["rows"][0]["fallback_fraction"] = 0.1  # type: ignore[index]
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_real_active_set_distribution_report(report)

    assert any("fallback-free" in error for error in errors)


def test_real_active_set_distribution_rejects_materialized_ratio_regression() -> None:
    report = _valid_report()
    report["rows"][0]["active_set_group_to_dense_tile_pair_ratio"] = 0.20  # type: ignore[index]
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_real_active_set_distribution_report(report)

    assert any("active-set group/dense-tile-pair ratio" in error for error in errors)


def test_real_active_set_distribution_rejects_stale_summary_after_payload_change() -> None:
    report = copy.deepcopy(_valid_report())
    report["rows"][0]["tile_active_set_groups"] = 99  # type: ignore[index]

    errors = verify_real_active_set_distribution_report(report)

    assert any("summary" in error and "mismatch" in error for error in errors)


def test_real_active_set_distribution_run_report() -> None:
    report = run_report()

    assert_real_active_set_distribution_report(report)
    assert report["summary"]["artifact_count"] == 3
    assert report["summary"]["max_active_set_group_to_dense_tile_pair_ratio"] < 0.05


def test_saved_real_active_set_distribution_artifact_satisfies_contract() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(Path(summary_json).read_text(encoding="utf-8"))

    assert_real_active_set_distribution_report(report)
