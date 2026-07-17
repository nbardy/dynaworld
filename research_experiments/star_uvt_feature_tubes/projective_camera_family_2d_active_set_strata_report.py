from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
STAR_UVT_ROOT = ROOT / "third_party" / "fast-mac-gsplat" / "variants" / "star_uvt_v0"
for path in (ROOT, ROOT / "src" / "train", STAR_UVT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from torch_gsplat_bridge_star_uvt import ProjectiveTraceTileTimeCell  # noqa: E402

from research_experiments.star_uvt_feature_tubes.projective_camera_family_2d_metal_chain_rule_report import (  # noqa: E402
    _q_grid,
)
from research_experiments.star_uvt_feature_tubes.projective_camera_family_2d_tile_order_reuse_report import (  # noqa: E402
    _compress_tile_order_topology,
    _materialized_cell_metadata_bytes,
    _shared_group_metadata_bytes,
)


DEFAULT_OUT_DIR = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-25_star_uvt_projective_camera_family_2d_active_set_strata"
)

ACTIVE_TRACE_COUNT = 3


def _finite_float(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value))


def _active_set_and_order(q_phase: float, q_height: float) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if float(q_phase) < 0.0:
        return (0, 1), (0, 1)
    if float(q_height) < 0.0:
        return (1, 2), (2, 1)
    return (0, 2), (2, 0)


def _depth_interval_by_primitive(q_phase: float, q_height: float) -> dict[int, tuple[float, float]]:
    phase_shift = 0.018 * abs(float(q_phase))
    height_shift = 0.016 * abs(float(q_height))
    return {
        0: (0.70 + phase_shift + height_shift, 0.84 + phase_shift + height_shift),
        1: (1.18 - phase_shift + height_shift, 1.34 - phase_shift + height_shift),
        2: (0.76 + phase_shift - height_shift, 0.91 + phase_shift - height_shift),
        10: (1.24 - phase_shift - height_shift, 1.41 - phase_shift - height_shift),
    }


def _active_depth_intervals(
    active_set: tuple[int, ...],
    *,
    q_phase: float,
    q_height: float,
) -> tuple[tuple[float, float], ...]:
    intervals = _depth_interval_by_primitive(q_phase, q_height)
    if active_set == (0, 2):
        # In this stratum primitive 2 is in front and primitive 0 is the back layer.
        return intervals[10], intervals[2]
    return tuple(intervals[primitive_id] for primitive_id in active_set)


def _make_active_set_cells(
    *,
    q_pairs: list[tuple[float, float]],
    frames_per_q: int,
) -> list[ProjectiveTraceTileTimeCell]:
    cells: list[ProjectiveTraceTileTimeCell] = []
    for q_index, (q_phase, q_height) in enumerate(q_pairs):
        active_set, local_order = _active_set_and_order(q_phase, q_height)
        start = q_index * int(frames_per_q)
        stop = start + int(frames_per_q)
        base_id = q_index * ACTIVE_TRACE_COUNT
        cells.append(
            ProjectiveTraceTileTimeCell(
                tile_u=0,
                tile_v=0,
                start=start,
                stop=stop,
                primitive_ids=tuple(base_id + local_id for local_id in active_set),
                ordered_primitive_ids=tuple(base_id + local_id for local_id in local_order),
                depth_intervals=_active_depth_intervals(active_set, q_phase=q_phase, q_height=q_height),
                fallback=False,
                fallback_reasons=(),
            )
        )
    return cells


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    materialized_bytes = int(report["materialized_tile_order_metadata_bytes"])
    shared_bytes = int(report["shared_tile_order_metadata_bytes"])
    return {
        "q_axis_count": int(report["q_axis_count"]),
        "q_pair_count": int(report["q_pair_count"]),
        "trace_count": int(report["trace_count"]),
        "frames_per_q": int(report["frames_per_q"]),
        "materialized_cell_count": int(report["materialized_cell_count"]),
        "shared_topology_group_count": int(report["shared_topology_group_count"]),
        "q_indices_covered": int(report["q_indices_covered"]),
        "active_set_stratum_count": int(report["active_set_stratum_count"]),
        "order_stratum_count": int(report["order_stratum_count"]),
        "materialized_tile_order_metadata_bytes": materialized_bytes,
        "shared_tile_order_metadata_bytes": shared_bytes,
        "shared_to_materialized_tile_order_metadata_ratio": float(shared_bytes) / float(materialized_bytes),
        "materialized_tile_order_metadata_growth": float(report["materialized_tile_order_metadata_growth"]),
        "shared_tile_order_metadata_growth": float(report["shared_tile_order_metadata_growth"]),
        "expanded_topology_matches_materialized": bool(report["expanded_topology_matches_materialized"]),
        "all_active_set_strata_depth_order_stable": bool(report["all_active_set_strata_depth_order_stable"]),
        "min_active_set_union_depth_order_gap": float(report["min_active_set_union_depth_order_gap"]),
    }


def run_report(*, q_axis_count: int = 5, frames: int = 4) -> dict[str, Any]:
    q_pairs = _q_grid(int(q_axis_count))
    q_pair_count = len(q_pairs)
    cells = _make_active_set_cells(q_pairs=q_pairs, frames_per_q=int(frames))
    groups, expanded_matches = _compress_tile_order_topology(
        cells,
        q_pair_count=q_pair_count,
        frames_per_q=int(frames),
        trace_count=ACTIVE_TRACE_COUNT,
    )
    materialized_bytes = _materialized_cell_metadata_bytes(cells)
    shared_bytes = _shared_group_metadata_bytes(groups)
    q_indices_covered = len({q_index for group in groups for q_index in group.q_indices})
    active_sets = {tuple(group.signature[4]) for group in groups}
    orders = {tuple(group.signature[5]) for group in groups}
    min_gap = min((group.min_union_order_gap for group in groups), default=math.inf)
    report: dict[str, Any] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_camera_family_2d_active_set_strata",
        "base_domain": "Q2 x Omega x T active-set split-strata metadata reuse",
        "theory_contract": (
            "When q-family support/culling changes the active primitive set across camera-family "
            "coordinates, the compiler should store a small set of active-set topology strata with "
            "q-index applicability and per-stratum family-union depth certificates, rather than one "
            "materialized tile/order record per q-pair."
        ),
        "q_axis_count": int(q_axis_count),
        "q_pair_count": q_pair_count,
        "frames_per_q": int(frames),
        "trace_count": ACTIVE_TRACE_COUNT,
        "materialized_cell_count": len(cells),
        "shared_topology_group_count": len(groups),
        "q_indices_covered": q_indices_covered,
        "active_set_stratum_count": len(active_sets),
        "order_stratum_count": len(orders),
        "materialized_tile_order_metadata_bytes": materialized_bytes,
        "shared_tile_order_metadata_bytes": shared_bytes,
        "materialized_tile_order_metadata_growth": float(q_pair_count),
        "shared_tile_order_metadata_growth": float(len(groups)),
        "expanded_topology_matches_materialized": bool(expanded_matches),
        "all_active_set_strata_depth_order_stable": bool(min_gap > 0.0),
        "min_active_set_union_depth_order_gap": float(min_gap),
        "rows": [
            {
                "topology_group_index": group_index,
                "q_count": len(group.q_indices),
                "q_indices": list(group.q_indices),
                "local_primitive_ids": list(group.signature[4]),
                "local_ordered_primitive_ids": list(group.signature[5]),
                "union_depth_intervals": [list(interval) for interval in group.union_depth_intervals],
                "min_union_depth_order_gap": group.min_union_order_gap,
            }
            for group_index, group in enumerate(groups)
        ],
    }
    report["summary"] = summarize(report)
    errors = verify_camera_family_2d_active_set_strata_report(report)
    if errors:
        report["status"] = "failed"
        report["errors"] = errors
    return report


def _assert_summary_close(actual: Any, expected: Any, key: str, errors: list[str]) -> None:
    if isinstance(expected, float):
        if not _finite_float(actual) or abs(float(actual) - expected) > 1.0e-8:
            errors.append(f"summary {key} mismatch: expected {expected!r}, got {actual!r}")
    elif actual != expected:
        errors.append(f"summary {key} mismatch: expected {expected!r}, got {actual!r}")


def verify_camera_family_2d_active_set_strata_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "ok":
        errors.append(f"status must be ok, got {report.get('status')!r}")
    if report.get("benchmark") != "star_uvt_projective_camera_family_2d_active_set_strata":
        errors.append(f"unexpected benchmark {report.get('benchmark')!r}")
    if report.get("base_domain") != "Q2 x Omega x T active-set split-strata metadata reuse":
        errors.append(f"base_domain must name active-set split-strata Q2 reuse, got {report.get('base_domain')!r}")
    theory_contract = report.get("theory_contract")
    if (
        not isinstance(theory_contract, str)
        or "support/culling changes" not in theory_contract
        or "active-set topology strata" not in theory_contract
        or "family-union depth certificates" not in theory_contract
    ):
        errors.append("theory_contract must preserve the active-set split-strata reuse contract")

    q_axis_count = int(report.get("q_axis_count") or 0)
    q_pair_count = int(report.get("q_pair_count") or 0)
    if q_axis_count != 5:
        errors.append(f"q_axis_count must be 5 for the saved Q2 active-set strata guard, got {q_axis_count}")
    if q_pair_count != q_axis_count * q_axis_count:
        errors.append("q_pair_count must equal q_axis_count^2")
    if int(report.get("trace_count") or 0) != ACTIVE_TRACE_COUNT:
        errors.append(f"trace_count must stay {ACTIVE_TRACE_COUNT}")
    if int(report.get("materialized_cell_count") or 0) != q_pair_count:
        errors.append("materialized cell count must expose one sampled-q active-set cell per q-pair")
    if int(report.get("shared_topology_group_count") or 0) != 3:
        errors.append("active-set smoke must compress to exactly three topology groups")
    if int(report.get("active_set_stratum_count") or 0) != 3:
        errors.append("active-set smoke must expose three active-set strata")
    if int(report.get("order_stratum_count") or 0) != 3:
        errors.append("active-set smoke must expose three order strata")
    if int(report.get("q_indices_covered") or 0) != q_pair_count:
        errors.append("active-set encoding must cover every q-pair")
    if report.get("expanded_topology_matches_materialized") is not True:
        errors.append("expanded active-set strata topology must match materialized local records")
    if report.get("all_active_set_strata_depth_order_stable") is not True:
        errors.append("every active-set stratum must keep a stable union-depth order certificate")
    if _finite_float(report.get("min_active_set_union_depth_order_gap")) and float(report["min_active_set_union_depth_order_gap"]) <= 0.20:
        errors.append("active-set union depth order gap must stay comfortably positive")

    materialized_bytes = int(report.get("materialized_tile_order_metadata_bytes") or 0)
    shared_bytes = int(report.get("shared_tile_order_metadata_bytes") or 0)
    if materialized_bytes <= 0 or shared_bytes <= 0:
        errors.append("metadata byte counts must be positive")
    elif shared_bytes / materialized_bytes >= 0.25:
        errors.append("active-set shared/materialized metadata ratio must stay below 0.25")
    if float(report.get("materialized_tile_order_metadata_growth") or 0.0) < 25.0:
        errors.append("active-set materialized metadata growth must expose sampled-Q scaling")
    if float(report.get("shared_tile_order_metadata_growth") or math.inf) > 3.05:
        errors.append("active-set shared metadata growth must track active-set strata count, not q-pair count")

    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != int(report.get("shared_topology_group_count") or -1):
        errors.append("rows must contain one entry per active-set topology group")
    else:
        q_covered: set[int] = set()
        active_sets: set[tuple[int, ...]] = set()
        orders: set[tuple[int, ...]] = set()
        for row in rows:
            if not isinstance(row, dict):
                errors.append("active-set topology row must be an object")
                continue
            q_indices = row.get("q_indices")
            if not isinstance(q_indices, list) or len(q_indices) != int(row.get("q_count") or -1):
                errors.append("active-set topology row q_indices must match q_count")
            else:
                q_covered.update(int(q_index) for q_index in q_indices)
            local_ids = row.get("local_primitive_ids")
            if not isinstance(local_ids, list):
                errors.append("active-set topology row must include local_primitive_ids")
            else:
                active_sets.add(tuple(int(value) for value in local_ids))
            local_order = row.get("local_ordered_primitive_ids")
            if not isinstance(local_order, list):
                errors.append("active-set topology row must include local_ordered_primitive_ids")
            else:
                orders.add(tuple(int(value) for value in local_order))
            if _finite_float(row.get("min_union_depth_order_gap")) and float(row["min_union_depth_order_gap"]) <= 0.20:
                errors.append("active-set topology row union depth order gap must stay comfortably positive")
        if q_covered != set(range(q_pair_count)):
            errors.append("active-set topology rows must cover all q indices exactly once")
        if active_sets != {(0, 1), (0, 2), (1, 2)}:
            errors.append(f"active-set rows must contain all expected support strata, got {sorted(active_sets)}")
        if orders != {(0, 1), (2, 0), (2, 1)}:
            errors.append(f"active-set rows must contain all expected local orders, got {sorted(orders)}")

    summary = report.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        return errors
    try:
        expected = summarize(report)
    except (KeyError, ZeroDivisionError, TypeError, ValueError) as exc:
        errors.append(f"summary could not be recomputed: {exc}")
        return errors
    for key, expected_value in expected.items():
        _assert_summary_close(summary.get(key), expected_value, key, errors)
    return errors


def assert_camera_family_2d_active_set_strata_report(report: dict[str, Any]) -> None:
    errors = verify_camera_family_2d_active_set_strata_report(report)
    if errors:
        raise AssertionError("camera-family 2D active-set strata report failed:\n- " + "\n- ".join(errors))


def write_report(report: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summary.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--q-axis-count", type=int, default=5)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--verify-report", type=Path)
    args = parser.parse_args()

    if args.verify_report is not None:
        report = json.loads(args.verify_report.read_text(encoding="utf-8"))
        assert_camera_family_2d_active_set_strata_report(report)
        print(f"verified {args.verify_report}")
        return

    report = run_report(q_axis_count=args.q_axis_count, frames=args.frames)
    assert_camera_family_2d_active_set_strata_report(report)
    write_report(report, args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
