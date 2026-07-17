from __future__ import annotations

import torch


def _percentile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    sorted_values = values.to(torch.float32).sort().values
    index = int(round((float(q) / 100.0) * float(sorted_values.numel() - 1)))
    return float(sorted_values[index].item())


def _tile_load_stats(
    *,
    tile_counts: list[torch.Tensor],
    tile_overflow: list[torch.Tensor],
    tile_unstable: list[torch.Tensor],
    tile_capacity: int,
) -> dict[str, int | float | bool]:
    if tile_counts:
        counts = torch.cat([item.detach().cpu().to(torch.int64).flatten() for item in tile_counts], dim=0)
        overflow = torch.cat([item.detach().cpu().to(torch.int64).flatten() for item in tile_overflow], dim=0)
        unstable = torch.cat([item.detach().cpu().to(torch.int64).flatten() for item in tile_unstable], dim=0)
    else:
        counts = torch.zeros((0,), dtype=torch.int64)
        overflow = torch.zeros((0,), dtype=torch.int64)
        unstable = torch.zeros((0,), dtype=torch.int64)
    active = counts > 0
    active_counts = counts[active]
    overflow_excess = torch.clamp(counts - int(tile_capacity), min=0)
    overflow_tile_count = int((overflow > 0).sum().item())
    max_tile_count = int(counts.max().item()) if counts.numel() else 0
    return {
        "tile_count": int(counts.numel()),
        "active_tile_count": int(active.sum().item()),
        "active_tile_fraction": float(active.to(torch.float32).mean().item()) if counts.numel() else 0.0,
        "raw_tile_tube_refs": int(counts.sum().item()),
        "clipped_tile_tube_refs": int(torch.clamp(counts, max=int(tile_capacity)).sum().item()),
        "overflow_excess_tube_refs": int(overflow_excess.sum().item()),
        "overflow_tile_count": overflow_tile_count,
        "unstable_tile_count": int((unstable > 0).sum().item()),
        "mean_tile_count": float(counts.to(torch.float32).mean().item()) if counts.numel() else 0.0,
        "mean_active_tile_count": float(active_counts.to(torch.float32).mean().item())
        if active_counts.numel()
        else 0.0,
        "p50_tile_count": _percentile(counts, 50.0),
        "p95_tile_count": _percentile(counts, 95.0),
        "p99_tile_count": _percentile(counts, 99.0),
        "p95_active_tile_count": _percentile(active_counts, 95.0),
        "p99_active_tile_count": _percentile(active_counts, 99.0),
        "max_tile_count": max_tile_count,
        "tile_capacity": int(tile_capacity),
        "fixedbin_eligible": bool(overflow_tile_count == 0 and max_tile_count <= int(tile_capacity)),
    }


__all__ = ["_tile_load_stats"]
