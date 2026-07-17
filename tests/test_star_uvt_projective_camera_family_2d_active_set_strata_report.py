from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research_experiments.star_uvt_feature_tubes.projective_camera_family_2d_active_set_strata_report import (
    DEFAULT_OUT_DIR,
    assert_camera_family_2d_active_set_strata_report,
    run_report,
    summarize,
    verify_camera_family_2d_active_set_strata_report,
)


def _valid_report() -> dict[str, object]:
    report: dict[str, object] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_camera_family_2d_active_set_strata",
        "base_domain": "Q2 x Omega x T active-set split-strata metadata reuse",
        "theory_contract": (
            "When q-family support/culling changes the active primitive set across camera-family "
            "coordinates, the compiler should store a small set of active-set topology strata with "
            "q-index applicability and per-stratum family-union depth certificates, rather than one "
            "materialized tile/order record per q-pair."
        ),
        "q_axis_count": 5,
        "q_pair_count": 25,
        "frames_per_q": 4,
        "trace_count": 3,
        "materialized_cell_count": 25,
        "shared_topology_group_count": 3,
        "q_indices_covered": 25,
        "active_set_stratum_count": 3,
        "order_stratum_count": 3,
        "materialized_tile_order_metadata_bytes": 1300,
        "shared_tile_order_metadata_bytes": 256,
        "materialized_tile_order_metadata_growth": 25.0,
        "shared_tile_order_metadata_growth": 3.0,
        "expanded_topology_matches_materialized": True,
        "all_active_set_strata_depth_order_stable": True,
        "min_active_set_union_depth_order_gap": 0.2630399994850159,
        "rows": [
            {
                "topology_group_index": 0,
                "q_count": 10,
                "q_indices": list(range(10)),
                "local_primitive_ids": [0, 1],
                "local_ordered_primitive_ids": [0, 1],
                "union_depth_intervals": [[0.7027000001072883, 0.849240000128746], [1.1745999997854233, 1.3411399998068811]],
                "min_union_depth_order_gap": 0.3253599996566773,
            },
            {
                "topology_group_index": 1,
                "q_count": 9,
                "q_indices": [12, 13, 14, 17, 18, 19, 22, 23, 24],
                "local_primitive_ids": [0, 2],
                "local_ordered_primitive_ids": [2, 0],
                "union_depth_intervals": [[1.230759999871254, 1.41], [0.7561600000858307, 0.9154000002145768]],
                "min_union_depth_order_gap": 0.31535999965667716,
            },
            {
                "topology_group_index": 2,
                "q_count": 6,
                "q_indices": [10, 11, 15, 16, 20, 21],
                "local_primitive_ids": [1, 2],
                "local_ordered_primitive_ids": [2, 1],
                "union_depth_intervals": [[1.176519999742508, 1.3438399999141695], [0.7561600000858307, 0.9134800002574921]],
                "min_union_depth_order_gap": 0.2630399994850159,
            },
        ],
    }
    report["summary"] = summarize(report)
    return report


def test_camera_family_2d_active_set_strata_accepts_valid_payload() -> None:
    report = _valid_report()

    assert verify_camera_family_2d_active_set_strata_report(report) == []
    assert_camera_family_2d_active_set_strata_report(report)
    assert report["summary"]["shared_topology_group_count"] == 3  # type: ignore[index]
    assert report["summary"]["shared_to_materialized_tile_order_metadata_ratio"] == pytest.approx(256 / 1300)  # type: ignore[index]


def test_camera_family_2d_active_set_strata_rejects_missing_contract() -> None:
    report = _valid_report()
    report["base_domain"] = "Q2"
    report["theory_contract"] = "too vague"

    errors = verify_camera_family_2d_active_set_strata_report(report)

    assert any("base_domain" in error for error in errors)
    assert any("active-set" in error or "support/culling" in error for error in errors)


def test_camera_family_2d_active_set_strata_rejects_one_group_regression() -> None:
    report = _valid_report()
    report["shared_topology_group_count"] = 1
    report["active_set_stratum_count"] = 1
    report["order_stratum_count"] = 1
    report["rows"] = report["rows"][:1]  # type: ignore[index]
    report["summary"] = summarize(report)

    errors = verify_camera_family_2d_active_set_strata_report(report)

    assert any("three topology groups" in error for error in errors)
    assert any("three active-set strata" in error for error in errors)


def test_camera_family_2d_active_set_strata_rejects_q_pair_scaling_regression() -> None:
    report = _valid_report()
    report["shared_tile_order_metadata_growth"] = 25.0
    report["summary"] = summarize(report)

    errors = verify_camera_family_2d_active_set_strata_report(report)

    assert any("shared metadata growth must track active-set strata count" in error for error in errors)


def test_camera_family_2d_active_set_strata_rejects_order_certificate_regression() -> None:
    report = _valid_report()
    report["all_active_set_strata_depth_order_stable"] = False
    report["min_active_set_union_depth_order_gap"] = -0.01
    report["rows"][0]["min_union_depth_order_gap"] = -0.01  # type: ignore[index]
    report["summary"] = summarize(report)

    errors = verify_camera_family_2d_active_set_strata_report(report)

    assert any("depth order gap" in error or "depth order certificate" in error for error in errors)


def test_camera_family_2d_active_set_strata_rejects_stale_summary_after_payload_change() -> None:
    report = copy.deepcopy(_valid_report())
    report["shared_tile_order_metadata_bytes"] = 300

    errors = verify_camera_family_2d_active_set_strata_report(report)

    assert any("summary shared_tile_order_metadata_bytes mismatch" in error for error in errors)


def test_camera_family_2d_active_set_strata_run_report() -> None:
    report = run_report(q_axis_count=5, frames=4)

    assert_camera_family_2d_active_set_strata_report(report)
    assert report["summary"]["shared_topology_group_count"] == 3
    assert report["summary"]["shared_to_materialized_tile_order_metadata_ratio"] < 0.25


def test_saved_camera_family_2d_active_set_strata_artifact_satisfies_contract() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(Path(summary_json).read_text(encoding="utf-8"))

    assert_camera_family_2d_active_set_strata_report(report)
