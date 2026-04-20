from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class OverlapMetricConfig:
    tile_size: int = 16
    alpha_threshold: float = 1.0 / 255.0
    large_splat_tile_threshold: int = 64
    batch_size: int = 8192
    max_candidate_pairs_per_batch: int = 1_000_000


def _as_cpu_float(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu", dtype=torch.float32).contiguous()


def _as_cpu_int(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu", dtype=torch.int64).contiguous()


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _quantile(value: torch.Tensor, q: float) -> float:
    if value.numel() == 0:
        return 0.0
    return float(torch.quantile(value.float(), q).item())


def summarize_overlap_counts(
    per_splat_counts: torch.Tensor,
    per_tile_counts: torch.Tensor,
    *,
    large_splat_tile_threshold: int,
    tile_size: int,
) -> dict[str, float]:
    per_splat_counts = _as_cpu_int(per_splat_counts)
    per_tile_counts = _as_cpu_int(per_tile_counts)
    splat_count = int(per_splat_counts.numel())
    tile_count = int(per_tile_counts.numel())
    total_keys = int(per_splat_counts.sum().item())
    active_splat_count = int((per_splat_counts > 0).sum().item())
    nonempty_tile_count = int((per_tile_counts > 0).sum().item())
    large_splat_count = int((per_splat_counts >= int(large_splat_tile_threshold)).sum().item())
    active_counts = per_splat_counts[per_splat_counts > 0]
    nonempty_counts = per_tile_counts[per_tile_counts > 0]

    return {
        "tile_size": float(tile_size),
        "total_overlap_keys": float(total_keys),
        "duplication_factor": float(total_keys / splat_count) if splat_count else 0.0,
        "active_splats": float(active_splat_count),
        "zero_overlap_splats": float(splat_count - active_splat_count),
        "mean_tiles_per_splat": float(per_splat_counts.float().mean().item()) if splat_count else 0.0,
        "mean_tiles_per_active_splat": float(active_counts.float().mean().item()) if active_counts.numel() else 0.0,
        "p95_tiles_per_splat": _quantile(per_splat_counts, 0.95),
        "max_tiles_per_splat": float(per_splat_counts.max().item()) if splat_count else 0.0,
        "large_splat_count": float(large_splat_count),
        "large_splat_fraction": float(large_splat_count / splat_count) if splat_count else 0.0,
        "mean_splats_per_tile": float(per_tile_counts.float().mean().item()) if tile_count else 0.0,
        "mean_splats_per_nonempty_tile": float(nonempty_counts.float().mean().item())
        if nonempty_counts.numel()
        else 0.0,
        "p95_splats_per_tile": _quantile(per_tile_counts, 0.95),
        "max_splats_per_tile": float(per_tile_counts.max().item()) if tile_count else 0.0,
        "nonempty_tiles": float(nonempty_tile_count),
        "nonempty_tile_fraction": float(nonempty_tile_count / tile_count) if tile_count else 0.0,
    }


def _next_batch_end(
    start: int,
    total: int,
    bbox_counts: torch.Tensor,
    cfg: OverlapMetricConfig,
) -> int:
    end = min(total, start + max(int(cfg.batch_size), 1))
    while end > start + 1:
        max_bbox = int(bbox_counts[start:end].max().item()) if end > start else 0
        if max_bbox * (end - start) <= int(cfg.max_candidate_pairs_per_batch):
            break
        end = start + max(1, (end - start) // 2)
    return end


def _rect_tile_counts(
    tile_min_x: torch.Tensor,
    tile_max_x: torch.Tensor,
    tile_min_y: torch.Tensor,
    tile_max_y: torch.Tensor,
    active: torch.Tensor,
    *,
    tiles_x: int,
    tiles_y: int,
    cfg: OverlapMetricConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    tile_min_x = _as_cpu_int(tile_min_x).clamp(0, max(tiles_x - 1, 0))
    tile_max_x = _as_cpu_int(tile_max_x).clamp(0, max(tiles_x - 1, 0))
    tile_min_y = _as_cpu_int(tile_min_y).clamp(0, max(tiles_y - 1, 0))
    tile_max_y = _as_cpu_int(tile_max_y).clamp(0, max(tiles_y - 1, 0))
    active = active.detach().to(device="cpu", dtype=torch.bool).contiguous()
    widths = (tile_max_x - tile_min_x + 1).clamp_min(0)
    heights = (tile_max_y - tile_min_y + 1).clamp_min(0)
    bbox_counts = torch.where(active, widths * heights, torch.zeros_like(widths))
    per_splat_counts = bbox_counts.clone()
    per_tile_counts = torch.zeros((tiles_x * tiles_y,), dtype=torch.int64)

    start = 0
    while start < int(bbox_counts.numel()):
        end = _next_batch_end(start, int(bbox_counts.numel()), bbox_counts, cfg)
        local_count = bbox_counts[start:end]
        max_bbox = int(local_count.max().item()) if local_count.numel() else 0
        if max_bbox > 0:
            local_ids = torch.arange(max_bbox, dtype=torch.int64).view(1, -1)
            widths_b = widths[start:end].clamp_min(1).view(-1, 1)
            local_x = local_ids % widths_b
            local_y = torch.div(local_ids, widths_b, rounding_mode="floor")
            valid = local_ids < local_count.view(-1, 1)
            tile_x = tile_min_x[start:end].view(-1, 1) + local_x
            tile_y = tile_min_y[start:end].view(-1, 1) + local_y
            tile_ids = (tile_y * tiles_x + tile_x)[valid]
            if tile_ids.numel() > 0:
                per_tile_counts.index_add_(0, tile_ids, torch.ones_like(tile_ids, dtype=torch.int64))
        start = end
    return per_splat_counts, per_tile_counts


def custom_rect_overlap_stats(
    means2d: torch.Tensor,
    cov2d: torch.Tensor,
    opacities: torch.Tensor,
    image_size: tuple[int, int],
    *,
    tile_size: int,
    bound_scale: float,
    alpha_threshold: float,
    large_splat_tile_threshold: int = 64,
    batch_size: int = 8192,
    max_candidate_pairs_per_batch: int = 1_000_000,
) -> dict[str, float]:
    height, width = int(image_size[0]), int(image_size[1])
    cfg = OverlapMetricConfig(
        tile_size=int(tile_size),
        alpha_threshold=float(alpha_threshold),
        large_splat_tile_threshold=int(large_splat_tile_threshold),
        batch_size=int(batch_size),
        max_candidate_pairs_per_batch=int(max_candidate_pairs_per_batch),
    )
    means2d = _as_cpu_float(means2d)
    cov2d = _as_cpu_float(cov2d)
    opacities = _as_cpu_float(opacities).reshape(-1)
    sigma_x = torch.sqrt(torch.clamp(cov2d[:, 0, 0], min=1e-6))
    sigma_y = torch.sqrt(torch.clamp(cov2d[:, 1, 1], min=1e-6))
    if alpha_threshold > 0.0:
        opacity_factor = torch.sqrt(
            torch.clamp(2.0 * torch.log(torch.clamp(opacities, min=alpha_threshold) / alpha_threshold), min=0.0)
        )
        radius_factor = torch.minimum(torch.full_like(opacity_factor, float(bound_scale)), opacity_factor)
    else:
        radius_factor = torch.full_like(sigma_x, float(bound_scale))

    radius_x = radius_factor * sigma_x
    radius_y = radius_factor * sigma_y
    min_x = torch.floor(means2d[:, 0] - radius_x).clamp(0, max(width - 1, 0)).to(torch.int64)
    max_x = torch.ceil(means2d[:, 0] + radius_x).clamp(0, max(width - 1, 0)).to(torch.int64)
    min_y = torch.floor(means2d[:, 1] - radius_y).clamp(0, max(height - 1, 0)).to(torch.int64)
    max_y = torch.ceil(means2d[:, 1] + radius_y).clamp(0, max(height - 1, 0)).to(torch.int64)
    active = (max_x >= min_x) & (max_y >= min_y) & (radius_factor > 0)

    per_splat, per_tile = _rect_tile_counts(
        torch.div(min_x, tile_size, rounding_mode="floor"),
        torch.div(max_x, tile_size, rounding_mode="floor"),
        torch.div(min_y, tile_size, rounding_mode="floor"),
        torch.div(max_y, tile_size, rounding_mode="floor"),
        active,
        tiles_x=_ceil_div(width, tile_size),
        tiles_y=_ceil_div(height, tile_size),
        cfg=cfg,
    )
    return summarize_overlap_counts(
        per_splat,
        per_tile,
        large_splat_tile_threshold=int(large_splat_tile_threshold),
        tile_size=tile_size,
    )


def taichi_obb_overlap_stats(
    packed_gaussians: torch.Tensor,
    image_size: tuple[int, int],
    *,
    tile_size: int,
    alpha_threshold: float,
    large_splat_tile_threshold: int = 64,
    batch_size: int = 8192,
    max_candidate_pairs_per_batch: int = 1_000_000,
) -> dict[str, float]:
    height, width = int(image_size[0]), int(image_size[1])
    cfg = OverlapMetricConfig(
        tile_size=int(tile_size),
        alpha_threshold=float(alpha_threshold),
        large_splat_tile_threshold=int(large_splat_tile_threshold),
        batch_size=int(batch_size),
        max_candidate_pairs_per_batch=int(max_candidate_pairs_per_batch),
    )
    packed = _as_cpu_float(packed_gaussians)
    mean = packed[:, 0:2]
    axis1 = torch.nn.functional.normalize(packed[:, 2:4], dim=-1, eps=1e-12)
    sigma = torch.clamp(packed[:, 4:6], min=1e-8)
    alpha = packed[:, 6].clamp_min(0.0)
    active = alpha > float(alpha_threshold)
    scale_factor = torch.zeros_like(alpha)
    scale_factor[active] = torch.sqrt(torch.clamp(2.0 * torch.log(alpha[active] / float(alpha_threshold)), min=0.0))
    scale = sigma * scale_factor.unsqueeze(-1)
    axis2 = torch.stack([-axis1[:, 1], axis1[:, 0]], dim=-1)
    extent = torch.sqrt((axis1 * scale[:, 0:1]).square() + (axis2 * scale[:, 1:2]).square())
    min_bound = mean - extent
    max_bound = mean + extent

    tiles_x = _ceil_div(width, tile_size)
    tiles_y = _ceil_div(height, tile_size)
    tile_min_x = torch.floor(min_bound[:, 0] / tile_size).to(torch.int64).clamp(0, max(tiles_x - 1, 0))
    tile_max_x = torch.ceil(max_bound[:, 0] / tile_size).to(torch.int64).clamp(0, max(tiles_x, 1)) - 1
    tile_min_y = torch.floor(min_bound[:, 1] / tile_size).to(torch.int64).clamp(0, max(tiles_y - 1, 0))
    tile_max_y = torch.ceil(max_bound[:, 1] / tile_size).to(torch.int64).clamp(0, max(tiles_y, 1)) - 1
    tile_max_x = torch.maximum(tile_max_x, tile_min_x)
    tile_max_y = torch.maximum(tile_max_y, tile_min_y)
    active = active & torch.isfinite(scale).all(dim=-1) & (scale[:, 0] > 0.0) & (scale[:, 1] > 0.0)
    widths = (tile_max_x - tile_min_x + 1).clamp_min(0)
    heights = (tile_max_y - tile_min_y + 1).clamp_min(0)
    bbox_counts = torch.where(active, widths * heights, torch.zeros_like(widths))
    per_splat_counts = torch.zeros((packed.shape[0],), dtype=torch.int64)
    per_tile_counts = torch.zeros((tiles_x * tiles_y,), dtype=torch.int64)

    start = 0
    while start < packed.shape[0]:
        end = _next_batch_end(start, packed.shape[0], bbox_counts, cfg)
        local_count = bbox_counts[start:end]
        max_bbox = int(local_count.max().item()) if local_count.numel() else 0
        if max_bbox > 0:
            local_ids = torch.arange(max_bbox, dtype=torch.int64).view(1, -1)
            widths_b = widths[start:end].clamp_min(1).view(-1, 1)
            local_x = local_ids % widths_b
            local_y = torch.div(local_ids, widths_b, rounding_mode="floor")
            candidate_valid = local_ids < local_count.view(-1, 1)
            tile_x = tile_min_x[start:end].view(-1, 1) + local_x
            tile_y = tile_min_y[start:end].view(-1, 1) + local_y

            lower_x = tile_x.float() * tile_size - mean[start:end, 0:1]
            lower_y = tile_y.float() * tile_size - mean[start:end, 1:2]
            upper_x = lower_x + float(tile_size)
            upper_y = lower_y + float(tile_size)
            corners_x = torch.stack([lower_x, upper_x, upper_x, lower_x], dim=-1)
            corners_y = torch.stack([lower_y, lower_y, upper_y, upper_y], dim=-1)
            inv_x = axis1[start:end] / scale[start:end, 0:1].clamp_min(1e-8)
            inv_y = axis2[start:end] / scale[start:end, 1:2].clamp_min(1e-8)
            local_axis_x = corners_x * inv_x[:, 0:1, None] + corners_y * inv_x[:, 1:2, None]
            local_axis_y = corners_x * inv_y[:, 0:1, None] + corners_y * inv_y[:, 1:2, None]
            separated = (
                (local_axis_x.amin(dim=-1) > 1.0)
                | (local_axis_x.amax(dim=-1) < -1.0)
                | (local_axis_y.amin(dim=-1) > 1.0)
                | (local_axis_y.amax(dim=-1) < -1.0)
            )
            accepted = candidate_valid & (~separated)
            per_splat_counts[start:end] = accepted.sum(dim=1).to(torch.int64)
            tile_ids = (tile_y * tiles_x + tile_x)[accepted]
            if tile_ids.numel() > 0:
                per_tile_counts.index_add_(0, tile_ids, torch.ones_like(tile_ids, dtype=torch.int64))
        start = end

    return summarize_overlap_counts(
        per_splat_counts,
        per_tile_counts,
        large_splat_tile_threshold=int(large_splat_tile_threshold),
        tile_size=tile_size,
    )


def exact_conic_overlap_stats(
    means2d: torch.Tensor,
    conics: torch.Tensor,
    opacities: torch.Tensor,
    image_size: tuple[int, int],
    *,
    tile_size: int,
    alpha_threshold: float,
    large_splat_tile_threshold: int = 64,
    batch_size: int = 8192,
    max_candidate_pairs_per_batch: int = 1_000_000,
) -> dict[str, float]:
    height, width = int(image_size[0]), int(image_size[1])
    cfg = OverlapMetricConfig(
        tile_size=int(tile_size),
        alpha_threshold=float(alpha_threshold),
        large_splat_tile_threshold=int(large_splat_tile_threshold),
        batch_size=int(batch_size),
        max_candidate_pairs_per_batch=int(max_candidate_pairs_per_batch),
    )
    means2d = _as_cpu_float(means2d)
    conics = _as_cpu_float(conics)
    opacities = _as_cpu_float(opacities).reshape(-1).clamp_min(0.0)
    a = conics[:, 0]
    b = conics[:, 1]
    c = conics[:, 2]
    det = a * c - b * b
    active = (opacities > float(alpha_threshold)) & (a > 1e-8) & (c > 1e-8) & (det > 1e-12)
    tau = torch.zeros_like(opacities)
    tau[active] = -2.0 * torch.log(torch.clamp(float(alpha_threshold) / opacities[active], min=1e-12))
    cov_xx = torch.where(active, c / det.clamp_min(1e-12), torch.zeros_like(c))
    cov_yy = torch.where(active, a / det.clamp_min(1e-12), torch.zeros_like(a))
    extent_x = torch.sqrt(torch.clamp(tau * cov_xx, min=0.0))
    extent_y = torch.sqrt(torch.clamp(tau * cov_yy, min=0.0))

    min_x = torch.floor(means2d[:, 0] - extent_x).clamp(0, max(width - 1, 0)).to(torch.int64)
    max_x = torch.ceil(means2d[:, 0] + extent_x).clamp(0, max(width - 1, 0)).to(torch.int64)
    min_y = torch.floor(means2d[:, 1] - extent_y).clamp(0, max(height - 1, 0)).to(torch.int64)
    max_y = torch.ceil(means2d[:, 1] + extent_y).clamp(0, max(height - 1, 0)).to(torch.int64)
    active = active & (max_x >= min_x) & (max_y >= min_y) & (tau > 0.0)

    tiles_x = _ceil_div(width, tile_size)
    tiles_y = _ceil_div(height, tile_size)
    tile_min_x = torch.div(min_x, tile_size, rounding_mode="floor")
    tile_max_x = torch.div(max_x, tile_size, rounding_mode="floor")
    tile_min_y = torch.div(min_y, tile_size, rounding_mode="floor")
    tile_max_y = torch.div(max_y, tile_size, rounding_mode="floor")
    widths = (tile_max_x - tile_min_x + 1).clamp_min(0)
    heights = (tile_max_y - tile_min_y + 1).clamp_min(0)
    bbox_counts = torch.where(active, widths * heights, torch.zeros_like(widths))
    per_splat_counts = torch.zeros((means2d.shape[0],), dtype=torch.int64)
    per_tile_counts = torch.zeros((tiles_x * tiles_y,), dtype=torch.int64)

    start = 0
    while start < means2d.shape[0]:
        end = _next_batch_end(start, means2d.shape[0], bbox_counts, cfg)
        local_count = bbox_counts[start:end]
        max_bbox = int(local_count.max().item()) if local_count.numel() else 0
        if max_bbox > 0:
            local_ids = torch.arange(max_bbox, dtype=torch.int64).view(1, -1)
            widths_b = widths[start:end].clamp_min(1).view(-1, 1)
            local_x = local_ids % widths_b
            local_y = torch.div(local_ids, widths_b, rounding_mode="floor")
            candidate_valid = local_ids < local_count.view(-1, 1)
            tile_x = tile_min_x[start:end].view(-1, 1) + local_x
            tile_y = tile_min_y[start:end].view(-1, 1) + local_y

            rx0 = tile_x.float() * tile_size + 0.5
            ry0 = tile_y.float() * tile_size + 0.5
            rx1 = torch.minimum(
                torch.full_like(rx0, float(width) - 0.5),
                (tile_x.float() + 1.0) * tile_size - 0.5,
            )
            ry1 = torch.minimum(
                torch.full_like(ry0, float(height) - 0.5),
                (tile_y.float() + 1.0) * tile_size - 0.5,
            )
            mx = means2d[start:end, 0:1]
            my = means2d[start:end, 1:2]
            dx0 = rx0 - mx
            dx1 = rx1 - mx
            dy0 = ry0 - my
            dy1 = ry1 - my

            aa = a[start:end].view(-1, 1)
            bb = b[start:end].view(-1, 1)
            cc = c[start:end].view(-1, 1)

            qmin = torch.full_like(dx0, math.inf)
            contains_mean = (mx >= rx0) & (mx <= rx1) & (my >= ry0) & (my <= ry1)
            qmin = torch.where(contains_mean, torch.zeros_like(qmin), qmin)

            dy_star = torch.clamp(-(bb / cc.clamp_min(1e-8)) * dx0, min=dy0, max=dy1)
            q = aa * dx0.square() + 2.0 * bb * dx0 * dy_star + cc * dy_star.square()
            qmin = torch.minimum(qmin, q)
            dy_star = torch.clamp(-(bb / cc.clamp_min(1e-8)) * dx1, min=dy0, max=dy1)
            q = aa * dx1.square() + 2.0 * bb * dx1 * dy_star + cc * dy_star.square()
            qmin = torch.minimum(qmin, q)

            dx_star = torch.clamp(-(bb / aa.clamp_min(1e-8)) * dy0, min=dx0, max=dx1)
            q = aa * dx_star.square() + 2.0 * bb * dx_star * dy0 + cc * dy0.square()
            qmin = torch.minimum(qmin, q)
            dx_star = torch.clamp(-(bb / aa.clamp_min(1e-8)) * dy1, min=dx0, max=dx1)
            q = aa * dx_star.square() + 2.0 * bb * dx_star * dy1 + cc * dy1.square()
            qmin = torch.minimum(qmin, q)

            for corner_dx, corner_dy in ((dx0, dy0), (dx0, dy1), (dx1, dy0), (dx1, dy1)):
                q = aa * corner_dx.square() + 2.0 * bb * corner_dx * corner_dy + cc * corner_dy.square()
                qmin = torch.minimum(qmin, q)

            accepted = candidate_valid & (qmin <= tau[start:end].view(-1, 1))
            per_splat_counts[start:end] = accepted.sum(dim=1).to(torch.int64)
            tile_ids = (tile_y * tiles_x + tile_x)[accepted]
            if tile_ids.numel() > 0:
                per_tile_counts.index_add_(0, tile_ids, torch.ones_like(tile_ids, dtype=torch.int64))
        start = end

    return summarize_overlap_counts(
        per_splat_counts,
        per_tile_counts,
        large_splat_tile_threshold=int(large_splat_tile_threshold),
        tile_size=tile_size,
    )


def prefix_overlap_stats(prefix: str, stats: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in stats.items()}


def aggregate_stat_dicts(stat_dicts: list[dict[str, float]], prefix: str) -> dict[str, float]:
    if not stat_dicts:
        return {}
    keys = sorted({key for stats in stat_dicts for key in stats.keys()})
    out: dict[str, float] = {}
    for key in keys:
        values = [float(stats[key]) for stats in stat_dicts if key in stats and math.isfinite(float(stats[key]))]
        if not values:
            continue
        out[f"{prefix}_{key}_mean"] = sum(values) / len(values)
        out[f"{prefix}_{key}_max"] = max(values)
    return out


def selected_overlap_summary(stats: dict[str, float]) -> dict[str, float]:
    keys = (
        "total_overlap_keys",
        "duplication_factor",
        "p95_tiles_per_splat",
        "max_tiles_per_splat",
        "large_splat_count",
        "p95_splats_per_tile",
        "max_splats_per_tile",
    )
    return {key: float(stats[key]) for key in keys if key in stats}


__all__ = [
    "OverlapMetricConfig",
    "aggregate_stat_dicts",
    "custom_rect_overlap_stats",
    "exact_conic_overlap_stats",
    "prefix_overlap_stats",
    "selected_overlap_summary",
    "summarize_overlap_counts",
    "taichi_obb_overlap_stats",
]
