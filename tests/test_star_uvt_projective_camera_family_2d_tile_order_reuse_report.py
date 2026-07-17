from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research_experiments.star_uvt_feature_tubes.projective_camera_family_2d_tile_order_reuse_report import (
    DEFAULT_OUT_DIR,
    assert_camera_family_2d_tile_order_reuse_report,
    run_report,
    summarize,
    verify_camera_family_2d_tile_order_reuse_report,
)


def _valid_report() -> dict[str, object]:
    report: dict[str, object] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_camera_family_2d_tile_order_reuse",
        "base_domain": "Q2 x Omega x T tile/order metadata reuse",
        "theory_contract": (
            "When q-family tile membership and depth order are stable, the compiler can store one "
            "local tile/order topology over Q2 x Omega x T plus q-index applicability, instead of "
            "materializing one tile/order cell per sampled q-pair. Depth intervals are stored as "
            "conservative family-union certificates."
        ),
        "q_axis_count": 5,
        "q_pair_count": 25,
        "frames_per_q": 4,
        "trace_count": 2,
        "materialized_cell_count": 25,
        "shared_topology_group_count": 1,
        "q_indices_covered": 25,
        "materialized_tile_order_metadata_bytes": 1300,
        "shared_tile_order_metadata_bytes": 152,
        "materialized_tile_order_metadata_growth": 25.0,
        "shared_tile_order_metadata_growth": 1.0,
        "expanded_topology_matches_materialized": True,
        "stable_union_depth_order": True,
        "min_union_depth_order_gap": 0.6033999919891357,
        "rows": [
            {
                "topology_group_index": 0,
                "q_count": 25,
                "q_indices": list(range(25)),
                "local_primitive_ids": [0, 1],
                "local_ordered_primitive_ids": [0, 1],
                "union_depth_intervals": [
                    [0.9785200357437134, 1.125920057296753],
                    [1.7293200492858887, 1.8175199031829834],
                ],
                "min_union_depth_order_gap": 0.6033999919891357,
            }
        ],
    }
    report["summary"] = summarize(report)
    return report


def test_camera_family_2d_tile_order_reuse_accepts_valid_payload() -> None:
    report = _valid_report()

    assert verify_camera_family_2d_tile_order_reuse_report(report) == []
    assert_camera_family_2d_tile_order_reuse_report(report)
    assert report["summary"]["shared_to_materialized_tile_order_metadata_ratio"] == pytest.approx(152 / 1300)


def test_camera_family_2d_tile_order_reuse_rejects_missing_contract() -> None:
    report = _valid_report()
    report["base_domain"] = "Q2"
    report["theory_contract"] = "too vague"

    errors = verify_camera_family_2d_tile_order_reuse_report(report)

    assert any("base_domain" in error for error in errors)
    assert any("tile/order reuse" in error or "tile/order topology" in error for error in errors)


def test_camera_family_2d_tile_order_reuse_rejects_materialized_metadata_scaling_regression() -> None:
    report = _valid_report()
    report["materialized_tile_order_metadata_growth"] = 4.0
    report["summary"] = summarize(report)

    errors = verify_camera_family_2d_tile_order_reuse_report(report)

    assert any("materialized tile/order metadata growth" in error for error in errors)


def test_camera_family_2d_tile_order_reuse_rejects_shared_metadata_ratio_regression() -> None:
    report = _valid_report()
    report["shared_tile_order_metadata_bytes"] = 400
    report["summary"] = summarize(report)

    errors = verify_camera_family_2d_tile_order_reuse_report(report)

    assert any("shared/materialized tile-order metadata ratio" in error for error in errors)


def test_camera_family_2d_tile_order_reuse_rejects_order_certificate_regression() -> None:
    report = _valid_report()
    report["stable_union_depth_order"] = False
    report["min_union_depth_order_gap"] = -0.01
    report["rows"][0]["min_union_depth_order_gap"] = -0.01  # type: ignore[index]
    report["summary"] = summarize(report)

    errors = verify_camera_family_2d_tile_order_reuse_report(report)

    assert any("depth certificate" in error or "depth order gap" in error for error in errors)


def test_camera_family_2d_tile_order_reuse_rejects_stale_summary_after_payload_change() -> None:
    report = copy.deepcopy(_valid_report())
    report["shared_tile_order_metadata_bytes"] = 200

    errors = verify_camera_family_2d_tile_order_reuse_report(report)

    assert any("summary shared_tile_order_metadata_bytes mismatch" in error for error in errors)


def test_camera_family_2d_tile_order_reuse_run_report() -> None:
    report = run_report(q_axis_count=5, frames=4)

    assert_camera_family_2d_tile_order_reuse_report(report)
    assert report["summary"]["shared_topology_group_count"] == 1
    assert report["summary"]["shared_to_materialized_tile_order_metadata_ratio"] < 0.20


def test_saved_camera_family_2d_tile_order_reuse_artifact_satisfies_contract() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(Path(summary_json).read_text(encoding="utf-8"))

    assert_camera_family_2d_tile_order_reuse_report(report)
