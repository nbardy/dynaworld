from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research_experiments.star_uvt_feature_tubes.projective_camera_family_2d_tile_order_strata_report import (
    DEFAULT_OUT_DIR,
    assert_camera_family_2d_tile_order_strata_report,
    run_report,
    summarize,
    verify_camera_family_2d_tile_order_strata_report,
)


def _valid_report() -> dict[str, object]:
    report: dict[str, object] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_camera_family_2d_tile_order_strata",
        "base_domain": "Q2 x Omega x T split-strata tile/order metadata reuse",
        "theory_contract": (
            "When q-family depth order changes across camera-family coordinates, the compiler should "
            "store a small set of tile/order topology strata with q-index applicability and per-stratum "
            "family-union depth certificates, rather than one materialized tile/order record per q-pair."
        ),
        "q_axis_count": 5,
        "q_pair_count": 25,
        "frames_per_q": 4,
        "trace_count": 2,
        "materialized_cell_count": 25,
        "shared_topology_group_count": 2,
        "q_indices_covered": 25,
        "order_stratum_count": 2,
        "materialized_tile_order_metadata_bytes": 1300,
        "shared_tile_order_metadata_bytes": 204,
        "materialized_tile_order_metadata_growth": 25.0,
        "shared_tile_order_metadata_growth": 2.0,
        "expanded_topology_matches_materialized": True,
        "all_strata_depth_order_stable": True,
        "min_stratum_union_depth_order_gap": 0.33200000002980246,
        "rows": [
            {
                "topology_group_index": 0,
                "q_count": 10,
                "q_indices": list(range(10)),
                "local_primitive_ids": [0, 1],
                "local_ordered_primitive_ids": [0, 1],
                "union_depth_intervals": [[0.7755, 0.9090], [1.2410, 1.3845]],
                "min_union_depth_order_gap": 0.33200000002980246,
            },
            {
                "topology_group_index": 1,
                "q_count": 15,
                "q_indices": list(range(10, 25)),
                "local_primitive_ids": [0, 1],
                "local_ordered_primitive_ids": [1, 0],
                "union_depth_intervals": [[1.2410, 1.3860], [0.7740, 0.9090]],
                "min_union_depth_order_gap": 0.33200000002980246,
            },
        ],
    }
    report["summary"] = summarize(report)
    return report


def test_camera_family_2d_tile_order_strata_accepts_valid_payload() -> None:
    report = _valid_report()

    assert verify_camera_family_2d_tile_order_strata_report(report) == []
    assert_camera_family_2d_tile_order_strata_report(report)
    assert report["summary"]["shared_topology_group_count"] == 2
    assert report["summary"]["shared_to_materialized_tile_order_metadata_ratio"] == pytest.approx(204 / 1300)


def test_camera_family_2d_tile_order_strata_rejects_missing_contract() -> None:
    report = _valid_report()
    report["base_domain"] = "Q2"
    report["theory_contract"] = "too vague"

    errors = verify_camera_family_2d_tile_order_strata_report(report)

    assert any("base_domain" in error for error in errors)
    assert any("split-strata" in error or "depth order changes" in error for error in errors)


def test_camera_family_2d_tile_order_strata_rejects_one_group_regression() -> None:
    report = _valid_report()
    report["shared_topology_group_count"] = 1
    report["order_stratum_count"] = 1
    report["rows"] = report["rows"][:1]
    report["summary"] = summarize(report)

    errors = verify_camera_family_2d_tile_order_strata_report(report)

    assert any("two topology groups" in error for error in errors)
    assert any("two order strata" in error for error in errors)


def test_camera_family_2d_tile_order_strata_rejects_q_pair_scaling_regression() -> None:
    report = _valid_report()
    report["shared_tile_order_metadata_growth"] = 25.0
    report["summary"] = summarize(report)

    errors = verify_camera_family_2d_tile_order_strata_report(report)

    assert any("shared metadata growth must track strata count" in error for error in errors)


def test_camera_family_2d_tile_order_strata_rejects_order_certificate_regression() -> None:
    report = _valid_report()
    report["all_strata_depth_order_stable"] = False
    report["min_stratum_union_depth_order_gap"] = -0.01
    report["rows"][0]["min_union_depth_order_gap"] = -0.01  # type: ignore[index]
    report["summary"] = summarize(report)

    errors = verify_camera_family_2d_tile_order_strata_report(report)

    assert any("depth order gap" in error or "depth order certificate" in error for error in errors)


def test_camera_family_2d_tile_order_strata_rejects_stale_summary_after_payload_change() -> None:
    report = copy.deepcopy(_valid_report())
    report["shared_tile_order_metadata_bytes"] = 300

    errors = verify_camera_family_2d_tile_order_strata_report(report)

    assert any("summary shared_tile_order_metadata_bytes mismatch" in error for error in errors)


def test_camera_family_2d_tile_order_strata_run_report() -> None:
    report = run_report(q_axis_count=5, frames=4)

    assert_camera_family_2d_tile_order_strata_report(report)
    assert report["summary"]["shared_topology_group_count"] == 2
    assert report["summary"]["shared_to_materialized_tile_order_metadata_ratio"] < 0.25


def test_saved_camera_family_2d_tile_order_strata_artifact_satisfies_contract() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(Path(summary_json).read_text(encoding="utf-8"))

    assert_camera_family_2d_tile_order_strata_report(report)
