from __future__ import annotations

import torch

from external_paths import ensure_third_party_path


def dense_overlap_mask(points_cpu: torch.Tensor, radii_cpu: torch.Tensor) -> torch.Tensor:
    dist_matrix = torch.cdist(points_cpu, points_cpu)
    overlap = dist_matrix <= (radii_cpu[:, None] + radii_cpu[None, :])
    overlap.fill_diagonal_(False)
    return overlap


def _ids_sorted_by_distance(ids: torch.Tensor, dist: torch.Tensor) -> list[int]:
    return [int(v) for v in sorted(ids.tolist(), key=lambda idx: (float(dist[idx]), int(idx)))]


def _regular_triangulation_adjacency(points: torch.Tensor, radii: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ensure_third_party_path("powerfoam-metal")
    from torch_powerfoam_metal import make_regular_triangulation_adjacency

    return make_regular_triangulation_adjacency(points, radii)


def build_csr_adjacency(
    points: torch.Tensor,
    radii: torch.Tensor,
    *,
    neighbor_count: int,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    points_cpu = points.detach().to(device="cpu", dtype=torch.float32)
    radii_cpu = radii.detach().to(device="cpu", dtype=torch.float32)
    cell_count = points_cpu.shape[0]
    k = min(max(int(neighbor_count), 0), max(cell_count - 1, 0))
    if cell_count == 0:
        return (
            torch.empty(0, device=points.device, dtype=torch.int32),
            torch.zeros(1, device=points.device, dtype=torch.int32),
        )

    dist_matrix = torch.cdist(points_cpu, points_cpu)
    dist_matrix.fill_diagonal_(float("inf"))
    if mode == "knn":
        if k == 0:
            return (
                torch.empty(0, device=points.device, dtype=torch.int32),
                torch.zeros(cell_count + 1, device=points.device, dtype=torch.int32),
            )
        rows_tensor = torch.topk(dist_matrix, k=k, dim=-1, largest=False).indices.reshape(-1)
        offsets_tensor = torch.arange(0, (cell_count + 1) * k, k, device=points.device, dtype=torch.int32)
        return (
            rows_tensor.to(device=points.device, dtype=torch.int32),
            offsets_tensor,
        )
    if mode == "cech_aabb":
        overlap = dense_overlap_mask(points_cpu, radii_cpu)
        rows: list[int] = []
        offsets = [0]
        for i in range(cell_count):
            ids = torch.nonzero(overlap[i], as_tuple=False).flatten()
            rows.extend(_ids_sorted_by_distance(ids, dist_matrix[i]))
            offsets.append(len(rows))
        return (
            torch.tensor(rows, device=points.device, dtype=torch.int32),
            torch.tensor(offsets, device=points.device, dtype=torch.int32),
        )
    if mode == "regular_triangulation":
        return _regular_triangulation_adjacency(points, radii)
    if mode != "overlap":
        raise ValueError(f"Unknown powerfoam adjacency mode {mode!r}")

    rows: list[int] = []
    offsets = [0]
    for i in range(cell_count):
        dist = dist_matrix[i]
        mask = dist <= (radii_cpu + radii_cpu[i])
        ids = torch.nonzero(mask, as_tuple=False).flatten()
        if ids.numel() == 0 and k > 0:
            ids = torch.topk(dist, k=k, largest=False).indices
        elif ids.numel() > k > 0:
            ids = ids[torch.topk(dist[ids], k=k, largest=False).indices]
        rows.extend(int(v) for v in ids.tolist())
        offsets.append(len(rows))
    return (
        torch.tensor(rows, device=points.device, dtype=torch.int32),
        torch.tensor(offsets, device=points.device, dtype=torch.int32),
    )


def csr_adjacency_stats(
    points: torch.Tensor,
    radii: torch.Tensor,
    rows: torch.Tensor,
    offsets: torch.Tensor,
    *,
    max_dense_cells: int = 4096,
) -> dict[str, float]:
    points_cpu = points.detach().to(device="cpu", dtype=torch.float32)
    radii_cpu = radii.detach().to(device="cpu", dtype=torch.float32)
    offsets_cpu = offsets.detach().to(device="cpu", dtype=torch.int64)
    rows_cpu = rows.detach().to(device="cpu", dtype=torch.int64)
    cell_count = int(points_cpu.shape[0])
    degrees = offsets_cpu[1:] - offsets_cpu[:-1]
    stats = {
        "adjacency_avg_degree": float(degrees.float().mean().item()) if cell_count > 0 else 0.0,
        "adjacency_max_degree": float(degrees.max().item()) if degrees.numel() > 0 else 0.0,
        "adjacency_edges": float(rows_cpu.numel()),
        "adjacency_required_overlap_edges": -1.0,
        "adjacency_missing_overlap_edges": -1.0,
    }
    if cell_count > int(max_dense_cells):
        return stats

    required = dense_overlap_mask(points_cpu, radii_cpu)
    present = torch.zeros_like(required)
    for cell in range(cell_count):
        start = int(offsets_cpu[cell])
        end = int(offsets_cpu[cell + 1])
        ids = rows_cpu[start:end]
        ids = ids[(ids >= 0) & (ids < cell_count) & (ids != cell)]
        if ids.numel() > 0:
            present[cell, ids] = True
    stats["adjacency_required_overlap_edges"] = float(required.sum().item())
    stats["adjacency_missing_overlap_edges"] = float((required & ~present).sum().item())
    return stats


__all__ = ["build_csr_adjacency", "csr_adjacency_stats", "dense_overlap_mask"]
