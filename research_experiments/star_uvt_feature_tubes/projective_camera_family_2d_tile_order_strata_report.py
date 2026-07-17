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
from research_experiments.star_uvt_feature_tubes.projective_camera_family_2d_metal_lowering_report import (  # noqa: E402
    TRACE_COUNT,
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
    / "2026-05-25_star_uvt_projective_camera_family_2d_tile_order_strata"
)


def _finite_float(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value))


def _stratified_depth_intervals(q_phase: float, q_height: float) -> tuple[tuple[float, float], ...]:
    q_height_shift = 0.025 * float(q_height)
    q_phase_shift = 0.010 * abs(float(q_phase))
    front = (0.78 + q_height_shift + q_phase_shift, 0.90 + q_height_shift + q_phase_shift)
    back = (1.25 + q_height_shift - q_phase_shift, 1.38 + q_height_shift - q_phase_shift)
    if float(q_phase) < 0.0:
        return front, back
    return back, front


def _stratified_order(q_phase: float) -> tuple[int, ...]:
    if float(q_phase) < 0.0:
        return (0, 1)
    return (1, 0)


def _make_stratified_cells(
    *,
    q_pairs: list[tuple[float, float]],
    frames_per_q: int,
) -> list[ProjectiveTraceTileTimeCell]:
    cells: list[ProjectiveTraceTileTimeCell] = []
    for q_index, (q_phase, q_height) in enumerate(q_pairs):
        start = q_index * int(frames_per_q)
        stop = start + int(frames_per_q)
        base_id = q_index * TRACE_COUNT
        local_order = _stratified_order(q_phase)
        cells.append(
            ProjectiveTraceTileTimeCell(
                tile_u=0,
                tile_v=0,
                start=start,
                stop=stop,
                primitive_ids=tuple(base_id + local_id for local_id in range(TRACE_COUNT)),
                ordered_primitive_ids=tuple(base_id + local_id for local_id in local_order),
                depth_intervals=_stratified_depth_intervals(q_phase, q_height),
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
        "order_stratum_count": int(report["order_stratum_count"]),
        "materialized_tile_order_metadata_bytes": materialized_bytes,
        "shared_tile_order_metadata_bytes": shared_bytes,
        "shared_to_materialized_tile_order_metadata_ratio": float(shared_bytes) / float(materialized_bytes),
        "materialized_tile_order_metadata_growth": float(report["materialized_tile_order_metadata_growth"]),
        "shared_tile_order_metadata_growth": float(report["shared_tile_order_metadata_growth"]),
        "expanded_topology_matches_materialized": bool(report["expanded_topology_matches_materialized"]),
        "all_strata_depth_order_stable": bool(report["all_strata_depth_order_stable"]),
        "min_stratum_union_depth_order_gap": float(report["min_stratum_union_depth_order_gap"]),
    }


def run_report(*, q_axis_count: int = 5, frames: int = 4) -> dict[str, Any]:
    q_pairs = _q_grid(int(q_axis_count))
    q_pair_count = len(q_pairs)
    cells = _make_stratified_cells(q_pairs=q_pairs, frames_per_q=int(frames))
    groups, expanded_matches = _compress_tile_order_topology(
        cells,
        q_pair_count=q_pair_count,
        frames_per_q=int(frames),
        trace_count=TRACE_COUNT,
    )
    materialized_bytes = _materialized_cell_metadata_bytes(cells)
    shared_bytes = _shared_group_metadata_bytes(groups)
    q_indices_covered = len({q_index for group in groups for q_index in group.q_indices})
    min_gap = min((group.min_union_order_gap for group in groups), default=math.inf)
    orders = {tuple(group.signature[5]) for group in groups}
    report: dict[str, Any] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_camera_family_2d_tile_order_strata",
        "base_domain": "Q2 x Omega x T split-strata tile/order metadata reuse",
        "theory_contract": (
            "When q-family depth order changes across camera-family coordinates, the compiler should "
            "store a small set of tile/order topology strata with q-index applicability and per-stratum "
            "family-union depth certificates, rather than one materialized tile/order record per q-pair."
        ),
        "q_axis_count": int(q_axis_count),
        "q_pair_count": q_pair_count,
        "frames_per_q": int(frames),
        "trace_count": TRACE_COUNT,
        "materialized_cell_count": len(cells),
        "shared_topology_group_count": len(groups),
        "q_indices_covered": q_indices_covered,
        "order_stratum_count": len(orders),
        "materialized_tile_order_metadata_bytes": materialized_bytes,
        "shared_tile_order_metadata_bytes": shared_bytes,
        "materialized_tile_order_metadata_growth": float(q_pair_count),
        "shared_tile_order_metadata_growth": float(len(groups)),
        "expanded_topology_matches_materialized": bool(expanded_matches),
        "all_strata_depth_order_stable": bool(min_gap > 0.0),
        "min_stratum_union_depth_order_gap": float(min_gap),
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
    errors = verify_camera_family_2d_tile_order_strata_report(report)
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


def verify_camera_family_2d_tile_order_strata_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "ok":
        errors.append(f"status must be ok, got {report.get('status')!r}")
    if report.get("benchmark") != "star_uvt_projective_camera_family_2d_tile_order_strata":
        errors.append(f"unexpected benchmark {report.get('benchmark')!r}")
    if report.get("base_domain") != "Q2 x Omega x T split-strata tile/order metadata reuse":
        errors.append(f"base_domain must name split-strata Q2 tile/order reuse, got {report.get('base_domain')!r}")
    theory_contract = report.get("theory_contract")
    if (
        not isinstance(theory_contract, str)
        or "depth order changes" not in theory_contract
        or "topology strata" not in theory_contract
        or "family-union depth certificates" not in theory_contract
    ):
        errors.append("theory_contract must preserve the split-strata tile/order reuse contract")

    q_axis_count = int(report.get("q_axis_count") or 0)
    q_pair_count = int(report.get("q_pair_count") or 0)
    if q_axis_count != 5:
        errors.append(f"q_axis_count must be 5 for the saved Q2 strata guard, got {q_axis_count}")
    if q_pair_count != q_axis_count * q_axis_count:
        errors.append("q_pair_count must equal q_axis_count^2")
    if int(report.get("trace_count") or 0) != TRACE_COUNT:
        errors.append(f"trace_count must stay {TRACE_COUNT}")
    if int(report.get("materialized_cell_count") or 0) != q_pair_count:
        errors.append("materialized cell count must expose one sampled-q tile/order cell per q-pair")
    if int(report.get("shared_topology_group_count") or 0) != 2:
        errors.append("split-strata smoke must compress to exactly two topology groups")
    if int(report.get("order_stratum_count") or 0) != 2:
        errors.append("split-strata smoke must expose two order strata")
    if int(report.get("q_indices_covered") or 0) != q_pair_count:
        errors.append("split-strata encoding must cover every q-pair")
    if report.get("expanded_topology_matches_materialized") is not True:
        errors.append("expanded split-strata topology must match materialized local tile/order records")
    if report.get("all_strata_depth_order_stable") is not True:
        errors.append("every split stratum must keep a stable union-depth order certificate")
    if _finite_float(report.get("min_stratum_union_depth_order_gap")) and float(report["min_stratum_union_depth_order_gap"]) <= 0.20:
        errors.append("split-strata union depth order gap must stay comfortably positive")

    materialized_bytes = int(report.get("materialized_tile_order_metadata_bytes") or 0)
    shared_bytes = int(report.get("shared_tile_order_metadata_bytes") or 0)
    if materialized_bytes <= 0 or shared_bytes <= 0:
        errors.append("metadata byte counts must be positive")
    elif shared_bytes / materialized_bytes >= 0.25:
        errors.append("split-strata shared/materialized tile-order metadata ratio must stay below 0.25")
    if float(report.get("materialized_tile_order_metadata_growth") or 0.0) < 25.0:
        errors.append("split-strata materialized tile/order metadata growth must expose sampled-Q scaling")
    if float(report.get("shared_tile_order_metadata_growth") or math.inf) > 2.05:
        errors.append("split-strata shared metadata growth must track strata count, not q-pair count")

    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != int(report.get("shared_topology_group_count") or -1):
        errors.append("rows must contain one entry per topology group")
    else:
        q_covered: set[int] = set()
        orders: set[tuple[int, ...]] = set()
        for row in rows:
            if not isinstance(row, dict):
                errors.append("topology row must be an object")
                continue
            q_indices = row.get("q_indices")
            if not isinstance(q_indices, list) or len(q_indices) != int(row.get("q_count") or -1):
                errors.append("topology row q_indices must match q_count")
            else:
                q_covered.update(int(q_index) for q_index in q_indices)
            local_order = row.get("local_ordered_primitive_ids")
            if not isinstance(local_order, list):
                errors.append("topology row must include local_ordered_primitive_ids")
            else:
                orders.add(tuple(int(value) for value in local_order))
            if _finite_float(row.get("min_union_depth_order_gap")) and float(row["min_union_depth_order_gap"]) <= 0.20:
                errors.append("topology row union depth order gap must stay comfortably positive")
        if q_covered != set(range(q_pair_count)):
            errors.append("topology rows must cover all q indices exactly once")
        if orders != {(0, 1), (1, 0)}:
            errors.append(f"split-strata rows must contain both local orders, got {sorted(orders)}")

    summary = report.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        return errors
    try:
        expected_summary = summarize(report)
        for key, expected_value in expected_summary.items():
            _assert_summary_close(summary.get(key), expected_value, key, errors)
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"summary could not be recomputed: {exc}")
    return errors


def assert_camera_family_2d_tile_order_strata_report(report: dict[str, Any]) -> None:
    errors = verify_camera_family_2d_tile_order_strata_report(report)
    if errors:
        raise AssertionError("2D camera-family tile/order strata report failed:\n- " + "\n- ".join(errors))


def write_report(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# STAR UVT Projective 2D Camera-Family Tile/Order Strata",
        "",
        "This report checks whether non-constant q-family depth order compresses to a small number of topology strata instead of one materialized cell per q-pair.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(report["summary"], indent=2, sort_keys=True),
        "```",
        "",
        "## Topology Strata",
        "",
        "| group | q count | local order | min union depth gap |",
        "| --- | --- | --- | --- |",
    ]
    for row in report["rows"]:
        lines.append(
            "| {topology_group_index} | {q_count} | {local_ordered_primitive_ids} | {min_union_depth_order_gap:.6g} |".format(
                **row
            )
        )
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--q-axis-count", type=int, default=5)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--verify-report", type=Path, default=None)
    args = parser.parse_args()

    if args.verify_report is not None:
        report = json.loads(args.verify_report.read_text(encoding="utf-8"))
        assert_camera_family_2d_tile_order_strata_report(report)
        print(f"verified {args.verify_report}")
        return

    report = run_report(q_axis_count=int(args.q_axis_count), frames=int(args.frames))
    assert_camera_family_2d_tile_order_strata_report(report)
    write_report(report, args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
