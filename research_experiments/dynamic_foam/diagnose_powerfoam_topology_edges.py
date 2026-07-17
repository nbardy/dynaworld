from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from config_utils import load_config_file
from checkpoint_utils import load_checkpoint_mapping, model_state_dict_from_checkpoint
from powerfoam_adjacency import build_csr_adjacency
from powerfoam_direct import POWERFOAM_SOFTPLUS_BETA
from powerfoam_metal_config import resolve_config
try:
    from .report_artifacts import relative_to_project as rel, write_report_json
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import relative_to_project as rel, write_report_json


def scalar(value: torch.Tensor | float | int) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def load_state_points(state: dict[str, torch.Tensor], cfg: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    points = torch.cat(
        [
            torch.tanh(state["raw_xy"]) * float(cfg["model"]["xy_extent"]),
            float(cfg["model"]["z_min"])
            + torch.sigmoid(state["raw_z"]) * (float(cfg["model"]["z_max"]) - float(cfg["model"]["z_min"])),
        ],
        dim=-1,
    ).to(device="cpu", dtype=torch.float32)
    radii = F.softplus(state["raw_radii"], beta=POWERFOAM_SOFTPLUS_BETA)
    radii = radii + float(cfg["model"]["radius_min"])
    return points, radii.to(device="cpu", dtype=torch.float32)


def csr_edges(rows: torch.Tensor, offsets: torch.Tensor) -> set[tuple[int, int]]:
    rows_cpu = rows.detach().cpu().to(dtype=torch.long)
    offsets_cpu = offsets.detach().cpu().to(dtype=torch.long)
    edges: set[tuple[int, int]] = set()
    for owner in range(int(offsets_cpu.numel()) - 1):
        for cursor in range(int(offsets_cpu[owner]), int(offsets_cpu[owner + 1])):
            neighbor = int(rows_cpu[cursor])
            if neighbor == owner:
                continue
            edges.add((min(owner, neighbor), max(owner, neighbor)))
    return edges


def graph_components(cell_count: int, edges: set[tuple[int, int]]) -> list[int]:
    neighbors = [[] for _ in range(int(cell_count))]
    for a, b in edges:
        neighbors[a].append(b)
        neighbors[b].append(a)
    seen = [False] * int(cell_count)
    sizes: list[int] = []
    for root in range(int(cell_count)):
        if seen[root]:
            continue
        seen[root] = True
        queue: deque[int] = deque([root])
        size = 0
        while queue:
            node = queue.popleft()
            size += 1
            for neighbor in neighbors[node]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    queue.append(neighbor)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def edge_margin_stats(
    edges: set[tuple[int, int]],
    dist_matrix: torch.Tensor,
    radii: torch.Tensor,
) -> dict[str, Any]:
    if not edges:
        return {"count": 0}
    a = torch.tensor([edge[0] for edge in edges], dtype=torch.long)
    b = torch.tensor([edge[1] for edge in edges], dtype=torch.long)
    dist = dist_matrix[a, b]
    radius_sum = radii[a] + radii[b]
    margin = radius_sum - dist
    return {
        "count": len(edges),
        "distance_mean": scalar(dist.mean()),
        "distance_p50": scalar(torch.quantile(dist, 0.5)),
        "distance_p90": scalar(torch.quantile(dist, 0.9)),
        "radius_sum_mean": scalar(radius_sum.mean()),
        "overlap_margin_mean": scalar(margin.mean()),
        "overlap_margin_p10": scalar(torch.quantile(margin, 0.1)),
        "overlap_margin_p50": scalar(torch.quantile(margin, 0.5)),
        "overlap_margin_p90": scalar(torch.quantile(margin, 0.9)),
        "non_overlapping_fraction": scalar((margin < 0.0).to(torch.float32).mean()),
    }


def frame_report(points: torch.Tensor, radii: torch.Tensor, cfg: dict[str, Any], frame_index: int) -> dict[str, Any]:
    point = points[int(frame_index)]
    radius = radii[int(frame_index)]
    cech_rows, cech_offsets = build_csr_adjacency(
        point,
        radius,
        neighbor_count=int(cfg["model"]["neighbor_count"]),
        mode="cech_aabb",
    )
    regular_rows, regular_offsets = build_csr_adjacency(
        point,
        radius,
        neighbor_count=int(cfg["model"]["neighbor_count"]),
        mode="regular_triangulation",
    )
    cech = csr_edges(cech_rows, cech_offsets)
    regular = csr_edges(regular_rows, regular_offsets)
    shared = cech & regular
    regular_missing = regular - cech
    cech_extra = cech - regular
    cell_count = int(point.shape[0])
    dist_matrix = torch.cdist(point, point)
    return {
        "frame_index": int(frame_index),
        "cell_count": cell_count,
        "cech_edge_count": len(cech),
        "regular_edge_count": len(regular),
        "shared_edge_count": len(shared),
        "regular_missing_from_cech_count": len(regular_missing),
        "cech_extra_count": len(cech_extra),
        "regular_edges_covered_by_cech_fraction": len(shared) / max(len(regular), 1),
        "cech_edges_used_by_regular_fraction": len(shared) / max(len(cech), 1),
        "cech_components": graph_components(cell_count, cech)[:16],
        "regular_components": graph_components(cell_count, regular)[:16],
        "regular_missing_from_cech_stats": edge_margin_stats(regular_missing, dist_matrix, radius),
        "cech_extra_stats": edge_margin_stats(cech_extra, dist_matrix, radius),
        "shared_edge_stats": edge_margin_stats(shared, dist_matrix, radius),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--frames", nargs="*", type=int, default=[0, 4, 8, 12])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cfg = resolve_config(load_config_file(args.config))
    checkpoint = args.checkpoint or (cfg["logging"]["output_dir"] / "checkpoint_best.pt")
    output = args.output or (cfg["logging"]["output_dir"] / "topology_edge_diagnostics.json")
    checkpoint_payload = load_checkpoint_mapping(checkpoint, map_location="cpu")
    state = model_state_dict_from_checkpoint(checkpoint_payload)
    points, radii = load_state_points(state, cfg)
    frame_indices = [int(frame) for frame in args.frames]
    frames = [frame_report(points, radii, cfg, frame) for frame in frame_indices]
    missing_total = sum(int(frame["regular_missing_from_cech_count"]) for frame in frames)
    regular_total = sum(int(frame["regular_edge_count"]) for frame in frames)
    report = {
        "config": rel(args.config),
        "checkpoint": rel(checkpoint),
        "output": rel(output),
        "frames": frames,
        "summary": {
            "frame_count": len(frames),
            "regular_edge_count": regular_total,
            "regular_missing_from_cech_count": missing_total,
            "regular_edges_covered_by_cech_fraction": (regular_total - missing_total) / max(regular_total, 1),
            "cech_graph_is_regular_superset": missing_total == 0,
        },
        "interpretation": (
            "This compares the fast sphere-overlap Cech/AABB graph against the SciPy regular-triangulation "
            "teacher on a frozen checkpoint. Missing regular edges mean the Cech graph cannot be treated as "
            "a conservative ray-walk graph for this state; false Cech extras are separate from missing faces."
        ),
    }
    write_report_json(output, report)
    print(json.dumps({"output": rel(output), "summary": report["summary"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
