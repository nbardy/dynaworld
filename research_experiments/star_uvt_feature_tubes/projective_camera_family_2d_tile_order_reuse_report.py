from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
STAR_UVT_ROOT = ROOT / "third_party" / "fast-mac-gsplat" / "variants" / "star_uvt_v0"
for path in (ROOT, ROOT / "src" / "train", STAR_UVT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from torch_gsplat_bridge_star_uvt import ProjectiveTraceTileTimeCell  # noqa: E402

from research_experiments.star_uvt_feature_tubes.projective_camera_family_2d_materialized_batch_report import (  # noqa: E402
    _batched_atlas_from_family,
)
from research_experiments.star_uvt_feature_tubes.projective_camera_family_2d_metal_chain_rule_report import (  # noqa: E402
    _q_grid,
)
from research_experiments.star_uvt_feature_tubes.projective_camera_family_2d_metal_lowering_report import (  # noqa: E402
    TRACE_COUNT,
    _family_coeff_table,
)


DEFAULT_OUT_DIR = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-25_star_uvt_projective_camera_family_2d_tile_order_reuse"
)

I32_BYTES = 4
F32_BYTES = 4


@dataclass(frozen=True)
class _TopologyGroup:
    signature: tuple[Any, ...]
    q_indices: tuple[int, ...]
    union_depth_intervals: tuple[tuple[float, float], ...]
    min_union_order_gap: float


def _finite_float(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value))


def _localize_cell(
    cell: ProjectiveTraceTileTimeCell,
    *,
    q_index: int,
    frames_per_q: int,
    trace_count: int,
) -> tuple[Any, ...]:
    base_id = int(q_index) * int(trace_count)
    return (
        int(cell.tile_u),
        int(cell.tile_v),
        int(cell.start) - int(q_index) * int(frames_per_q),
        int(cell.stop) - int(q_index) * int(frames_per_q),
        tuple(int(pid) - base_id for pid in cell.primitive_ids),
        tuple(int(pid) - base_id for pid in cell.ordered_primitive_ids),
        bool(cell.fallback),
        tuple(cell.fallback_reasons),
    )


def _cell_q_index(cell: ProjectiveTraceTileTimeCell, *, frames_per_q: int) -> int:
    if int(cell.start) % int(frames_per_q) != 0 or int(cell.stop) - int(cell.start) != int(frames_per_q):
        raise ValueError("this reuse report expects one full local frame block per q sample")
    return int(cell.start) // int(frames_per_q)


def _materialized_cell_metadata_bytes(cells: list[ProjectiveTraceTileTimeCell]) -> int:
    total = 0
    for cell in cells:
        header_i32 = 5  # tile_u, tile_v, start, stop, fallback flag
        id_i32 = len(cell.primitive_ids) + len(cell.ordered_primitive_ids)
        interval_f32 = 2 * len(cell.depth_intervals)
        total += (header_i32 + id_i32) * I32_BYTES + interval_f32 * F32_BYTES
    return int(total)


def _shared_group_metadata_bytes(groups: list[_TopologyGroup]) -> int:
    total = 0
    for group in groups:
        primitive_ids = group.signature[4]
        ordered_ids = group.signature[5]
        header_i32 = 5  # tile_u, tile_v, local_start, local_stop, fallback flag
        id_i32 = len(primitive_ids) + len(ordered_ids)
        q_index_i32 = len(group.q_indices)
        union_interval_f32 = 2 * len(group.union_depth_intervals)
        total += (header_i32 + id_i32 + q_index_i32) * I32_BYTES + union_interval_f32 * F32_BYTES
    return int(total)


def _compress_tile_order_topology(
    cells: list[ProjectiveTraceTileTimeCell],
    *,
    q_pair_count: int,
    frames_per_q: int,
    trace_count: int,
) -> tuple[list[_TopologyGroup], bool]:
    buckets: dict[tuple[Any, ...], list[ProjectiveTraceTileTimeCell]] = {}
    for cell in cells:
        q_index = _cell_q_index(cell, frames_per_q=frames_per_q)
        if q_index < 0 or q_index >= int(q_pair_count):
            raise ValueError(f"q index {q_index} is outside q_pair_count={q_pair_count}")
        signature = _localize_cell(
            cell,
            q_index=q_index,
            frames_per_q=frames_per_q,
            trace_count=trace_count,
        )
        buckets.setdefault(signature, []).append(cell)

    groups: list[_TopologyGroup] = []
    expanded_matches = True
    for signature, group_cells in sorted(buckets.items(), key=lambda item: item[0]):
        q_indices = tuple(_cell_q_index(cell, frames_per_q=frames_per_q) for cell in group_cells)
        primitive_ids = tuple(int(pid) for pid in signature[4])
        ordered_ids = tuple(int(pid) for pid in signature[5])
        interval_by_primitive: dict[int, list[tuple[float, float]]] = {pid: [] for pid in primitive_ids}
        for cell in group_cells:
            q_index = _cell_q_index(cell, frames_per_q=frames_per_q)
            if _localize_cell(cell, q_index=q_index, frames_per_q=frames_per_q, trace_count=trace_count) != signature:
                expanded_matches = False
            for primitive_id, interval in zip(primitive_ids, cell.depth_intervals, strict=True):
                interval_by_primitive[int(primitive_id)].append((float(interval[0]), float(interval[1])))
        union_by_primitive = {
            primitive_id: (
                min(interval[0] for interval in intervals),
                max(interval[1] for interval in intervals),
            )
            for primitive_id, intervals in interval_by_primitive.items()
        }
        order_gaps = [
            union_by_primitive[int(back)][0] - union_by_primitive[int(front)][1]
            for front, back in zip(ordered_ids[:-1], ordered_ids[1:], strict=True)
        ]
        min_gap = min(order_gaps, default=math.inf)
        groups.append(
            _TopologyGroup(
                signature=signature,
                q_indices=tuple(sorted(q_indices)),
                union_depth_intervals=tuple(union_by_primitive[primitive_id] for primitive_id in primitive_ids),
                min_union_order_gap=float(min_gap),
            )
        )
    return groups, expanded_matches


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
        "materialized_tile_order_metadata_bytes": materialized_bytes,
        "shared_tile_order_metadata_bytes": shared_bytes,
        "shared_to_materialized_tile_order_metadata_ratio": float(shared_bytes) / float(materialized_bytes),
        "materialized_tile_order_metadata_growth": float(report["materialized_tile_order_metadata_growth"]),
        "shared_tile_order_metadata_growth": float(report["shared_tile_order_metadata_growth"]),
        "expanded_topology_matches_materialized": bool(report["expanded_topology_matches_materialized"]),
        "stable_union_depth_order": bool(report["stable_union_depth_order"]),
        "min_union_depth_order_gap": float(report["min_union_depth_order_gap"]),
    }


def run_report(*, q_axis_count: int = 5, frames: int = 4) -> dict[str, Any]:
    q_pairs = _q_grid(int(q_axis_count))
    q_pair_count = len(q_pairs)
    family_coeffs = _family_coeff_table(device="cpu")
    atlas = _batched_atlas_from_family(
        family_coeffs,
        q_pairs,
        frames_per_q=int(frames),
        device="cpu",
    )
    groups, expanded_matches = _compress_tile_order_topology(
        atlas.cells,
        q_pair_count=q_pair_count,
        frames_per_q=int(frames),
        trace_count=TRACE_COUNT,
    )
    materialized_bytes = _materialized_cell_metadata_bytes(atlas.cells)
    shared_bytes = _shared_group_metadata_bytes(groups)
    q_indices_covered = len({q_index for group in groups for q_index in group.q_indices})
    min_union_gap = min((group.min_union_order_gap for group in groups), default=math.inf)
    report: dict[str, Any] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_camera_family_2d_tile_order_reuse",
        "base_domain": "Q2 x Omega x T tile/order metadata reuse",
        "theory_contract": (
            "When q-family tile membership and depth order are stable, the compiler can store one "
            "local tile/order topology over Q2 x Omega x T plus q-index applicability, instead of "
            "materializing one tile/order cell per sampled q-pair. Depth intervals are stored as "
            "conservative family-union certificates."
        ),
        "q_axis_count": int(q_axis_count),
        "q_pair_count": q_pair_count,
        "frames_per_q": int(frames),
        "trace_count": TRACE_COUNT,
        "materialized_cell_count": len(atlas.cells),
        "shared_topology_group_count": len(groups),
        "q_indices_covered": q_indices_covered,
        "materialized_tile_order_metadata_bytes": materialized_bytes,
        "shared_tile_order_metadata_bytes": shared_bytes,
        "materialized_tile_order_metadata_growth": float(q_pair_count),
        "shared_tile_order_metadata_growth": float(len(groups)),
        "expanded_topology_matches_materialized": bool(expanded_matches),
        "stable_union_depth_order": bool(min_union_gap > 0.0),
        "min_union_depth_order_gap": float(min_union_gap),
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
    errors = verify_camera_family_2d_tile_order_reuse_report(report)
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


def verify_camera_family_2d_tile_order_reuse_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "ok":
        errors.append(f"status must be ok, got {report.get('status')!r}")
    if report.get("benchmark") != "star_uvt_projective_camera_family_2d_tile_order_reuse":
        errors.append(f"unexpected benchmark {report.get('benchmark')!r}")
    if report.get("base_domain") != "Q2 x Omega x T tile/order metadata reuse":
        errors.append(f"base_domain must name Q2 tile/order reuse, got {report.get('base_domain')!r}")
    theory_contract = report.get("theory_contract")
    if (
        not isinstance(theory_contract, str)
        or "tile/order topology" not in theory_contract
        or "q-index applicability" not in theory_contract
        or "family-union certificates" not in theory_contract
    ):
        errors.append("theory_contract must preserve the tile/order reuse and depth-certificate contract")

    q_axis_count = int(report.get("q_axis_count") or 0)
    q_pair_count = int(report.get("q_pair_count") or 0)
    if q_axis_count != 5:
        errors.append(f"q_axis_count must be 5 for the saved Q2 guard, got {q_axis_count}")
    if q_pair_count != q_axis_count * q_axis_count:
        errors.append("q_pair_count must equal q_axis_count^2")
    if int(report.get("trace_count") or 0) != TRACE_COUNT:
        errors.append(f"trace_count must stay {TRACE_COUNT}")
    if int(report.get("materialized_cell_count") or 0) != q_pair_count:
        errors.append("materialized cell count must expose one sampled-q tile/order cell per q-pair")
    if int(report.get("shared_topology_group_count") or 0) != 1:
        errors.append("shared topology group count must stay one for this stable-order smoke")
    if int(report.get("q_indices_covered") or 0) != q_pair_count:
        errors.append("shared topology encoding must cover every q-pair")
    if report.get("expanded_topology_matches_materialized") is not True:
        errors.append("expanded shared topology must match the materialized local tile/order records")
    if report.get("stable_union_depth_order") is not True:
        errors.append("conservative family-union depth certificate must keep the stored order stable")
    if _finite_float(report.get("min_union_depth_order_gap")) and float(report["min_union_depth_order_gap"]) <= 0.25:
        errors.append("family-union depth order gap must stay comfortably positive")

    materialized_bytes = int(report.get("materialized_tile_order_metadata_bytes") or 0)
    shared_bytes = int(report.get("shared_tile_order_metadata_bytes") or 0)
    if materialized_bytes <= 0 or shared_bytes <= 0:
        errors.append("metadata byte counts must be positive")
    elif shared_bytes / materialized_bytes >= 0.20:
        errors.append("shared/materialized tile-order metadata ratio must stay below 0.20")
    if float(report.get("materialized_tile_order_metadata_growth") or 0.0) < 25.0:
        errors.append("materialized tile/order metadata growth must expose sampled-Q scaling")
    if float(report.get("shared_tile_order_metadata_growth") or math.inf) > 1.05:
        errors.append("shared tile/order metadata growth must stay near constant")

    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != int(report.get("shared_topology_group_count") or -1):
        errors.append("rows must contain one entry per topology group")
    else:
        q_covered: set[int] = set()
        for row in rows:
            if not isinstance(row, dict):
                errors.append("topology row must be an object")
                continue
            q_indices = row.get("q_indices")
            if not isinstance(q_indices, list) or len(q_indices) != int(row.get("q_count") or -1):
                errors.append("topology row q_indices must match q_count")
            else:
                q_covered.update(int(q_index) for q_index in q_indices)
            if row.get("local_ordered_primitive_ids") != [0, 1]:
                errors.append("stable smoke should preserve local order [0, 1]")
            if _finite_float(row.get("min_union_depth_order_gap")) and float(row["min_union_depth_order_gap"]) <= 0.25:
                errors.append("topology row union depth order gap must stay comfortably positive")
        if len(q_covered) != q_pair_count:
            errors.append("topology rows must cover all q indices exactly once")

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


def assert_camera_family_2d_tile_order_reuse_report(report: dict[str, Any]) -> None:
    errors = verify_camera_family_2d_tile_order_reuse_report(report)
    if errors:
        raise AssertionError("2D camera-family tile/order reuse report failed:\n- " + "\n- ".join(errors))


def write_report(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# STAR UVT Projective 2D Camera-Family Tile/Order Reuse",
        "",
        "This report checks whether the sampled Q2 materialized tile/order cells can be represented by one shared topology record plus q-index applicability.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(report["summary"], indent=2, sort_keys=True),
        "```",
        "",
        "## Topology Groups",
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
        assert_camera_family_2d_tile_order_reuse_report(report)
        print(f"verified {args.verify_report}")
        return

    report = run_report(q_axis_count=int(args.q_axis_count), frames=int(args.frames))
    assert_camera_family_2d_tile_order_reuse_report(report)
    write_report(report, args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
