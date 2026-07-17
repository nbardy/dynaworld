from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


SUPPORT_BIRTH_SPLIT_TUBE_SELECTIONS = {"first", "lowest_opacity"}
SUPPORT_BIRTH_SPLIT_FEATURE_INIT_MODES = {"preserve", "target_group_mean"}
SUPPORT_BIRTH_SPLIT_TARGET_POINT_SOURCES = {
    "top_brightness",
    "uncovered_brightness",
    "low_alpha",
    "residual_uncovered_brightness",
    "footprint_residual_uncovered_brightness",
    "cap_slack_uncovered_brightness",
    "cap_slack_low_alpha",
    "cap_slack_residual_uncovered_brightness",
    "cap_slack_footprint_residual_uncovered_brightness",
}
SUPPORT_BIRTH_SPLIT_SHAPES = {"isotropic", "trajectory_ellipse"}
SUPPORT_BIRTH_SPLIT_CENTER_STRATEGIES = {"global_line", "farthest_xy"}
SUPPORT_BIRTH_SPLIT_TUBE_ALLOCATIONS = {"proportional", "uniform"}


def _visibility_proxy_target_points(
    target_rgb: torch.Tensor,
    *,
    target_top_fraction: float,
    max_points: int,
    grid_stride: int,
    frame_stride: int,
    device: torch.device,
) -> torch.Tensor:
    if not 0.0 < float(target_top_fraction) <= 1.0:
        raise ValueError("visibility_proxy.target_top_fraction must be in (0, 1]")
    if int(max_points) <= 0:
        raise ValueError("visibility_proxy.max_points must be positive")
    if int(grid_stride) <= 0 or int(frame_stride) <= 0:
        raise ValueError("visibility_proxy grid/frame strides must be positive")
    if target_rgb.dim() != 4 or int(target_rgb.shape[1]) != 3:
        raise ValueError(f"target_rgb must have shape [T,3,H,W], got {tuple(target_rgb.shape)}")
    frames, _channels, _height, _width = (int(item) for item in target_rgb.shape)
    score = target_rgb.detach().to(device="cpu", dtype=torch.float32).mean(dim=1)
    sampled = score[:: int(frame_stride), :: int(grid_stride), :: int(grid_stride)].contiguous()
    flat = sampled.flatten()
    if flat.numel() == 0:
        raise ValueError("visibility_proxy target sampling produced no candidates")
    keep = max(1, min(int(flat.numel()), int(math.ceil(float(flat.numel()) * float(target_top_fraction)))))
    if keep == int(flat.numel()):
        flat_ids = torch.arange(int(flat.numel()), dtype=torch.int64)
    else:
        flat_ids = torch.topk(flat, k=keep, largest=True, sorted=True).indices.to(torch.int64)
    if int(flat_ids.numel()) > int(max_points):
        select = torch.linspace(0, int(flat_ids.numel()) - 1, int(max_points)).round().to(torch.int64)
        flat_ids = flat_ids.index_select(0, select)
    sampled_h = int(sampled.shape[1])
    sampled_w = int(sampled.shape[2])
    sampled_frame = flat_ids // (sampled_h * sampled_w)
    remainder = flat_ids % (sampled_h * sampled_w)
    sampled_y = remainder // sampled_w
    sampled_x = remainder % sampled_w
    frame = sampled_frame * int(frame_stride)
    y = (sampled_y * int(grid_stride)).to(torch.float32) + 0.5
    x = (sampled_x * int(grid_stride)).to(torch.float32) + 0.5
    t = frame.to(torch.float32) - 0.5 * float(frames - 1)
    return torch.stack((x, y, t), dim=-1).contiguous().to(device=device)


def _support_birth_split_sample_grid(
    *,
    frames: int,
    height: int,
    width: int,
    frame_stride: int,
    grid_stride: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if int(grid_stride) <= 0 or int(frame_stride) <= 0:
        raise ValueError("support_birth_split grid/frame strides must be positive")
    frame_ids = torch.arange(0, int(frames), int(frame_stride), dtype=torch.int64, device=device)
    y_ids = torch.arange(0, int(height), int(grid_stride), dtype=torch.int64, device=device)
    x_ids = torch.arange(0, int(width), int(grid_stride), dtype=torch.int64, device=device)
    frame_grid, y_grid, x_grid = torch.meshgrid(frame_ids, y_ids, x_ids, indexing="ij")
    pixel_ids = (frame_grid * int(height) * int(width) + y_grid * int(width) + x_grid).flatten().to(torch.int32)
    return frame_ids, y_ids, x_ids, pixel_ids


def _support_birth_split_sampled_tile_load(
    tile_counts: torch.Tensor,
    *,
    frames: int,
    height: int,
    width: int,
    frame_stride: int,
    grid_stride: int,
    tile_x: int,
    tile_y: int,
    tile_t: int,
) -> torch.Tensor:
    if int(grid_stride) <= 0 or int(frame_stride) <= 0:
        raise ValueError("support_birth_split grid/frame strides must be positive")
    if int(tile_x) <= 0 or int(tile_y) <= 0 or int(tile_t) <= 0:
        raise ValueError("support_birth_split tile dimensions must be positive")
    if tile_counts.dim() != 1:
        raise ValueError(f"tile_counts must have shape [tile_count], got {tuple(tile_counts.shape)}")
    tiles_x = (int(width) + int(tile_x) - 1) // int(tile_x)
    tiles_y = (int(height) + int(tile_y) - 1) // int(tile_y)
    tiles_t = (int(frames) + int(tile_t) - 1) // int(tile_t)
    expected = tiles_x * tiles_y * tiles_t
    if int(tile_counts.numel()) != expected:
        raise ValueError(f"tile_counts must have {expected} entries, got {int(tile_counts.numel())}")
    frame_ids = torch.arange(0, int(frames), int(frame_stride), dtype=torch.int64)
    y_ids = torch.arange(0, int(height), int(grid_stride), dtype=torch.int64)
    x_ids = torch.arange(0, int(width), int(grid_stride), dtype=torch.int64)
    frame_grid, y_grid, x_grid = torch.meshgrid(frame_ids, y_ids, x_ids, indexing="ij")
    tile_frame = torch.div(frame_grid, int(tile_t), rounding_mode="floor").clamp_max(tiles_t - 1)
    tile_v = torch.div(y_grid, int(tile_y), rounding_mode="floor").clamp_max(tiles_y - 1)
    tile_u = torch.div(x_grid, int(tile_x), rounding_mode="floor").clamp_max(tiles_x - 1)
    tile_ids = (tile_frame * tiles_y * tiles_x + tile_v * tiles_x + tile_u).reshape(-1)
    loads = tile_counts.detach().to(device="cpu", dtype=torch.float32).index_select(0, tile_ids.cpu())
    return loads.reshape(int(frame_ids.numel()), int(y_ids.numel()), int(x_ids.numel())).contiguous()


def _support_birth_split_spatial_mean_pool(sampled: torch.Tensor, *, radius_samples: int) -> torch.Tensor:
    if int(radius_samples) <= 0:
        return sampled
    if sampled.dim() != 3:
        raise ValueError(f"sampled must have shape [T,H,W], got {tuple(sampled.shape)}")
    kernel = 2 * int(radius_samples) + 1
    values = sampled.unsqueeze(1)
    weights = torch.ones_like(values)
    total = F.avg_pool2d(values, kernel_size=kernel, stride=1, padding=int(radius_samples), count_include_pad=False)
    counts = F.avg_pool2d(weights, kernel_size=kernel, stride=1, padding=int(radius_samples), count_include_pad=False)
    return (total / counts.clamp_min(1.0e-8)).squeeze(1).contiguous()


def _support_birth_split_target_points(
    target_rgb: torch.Tensor,
    *,
    target_point_source: str,
    target_top_fraction: float,
    max_points: int,
    grid_stride: int,
    frame_stride: int,
    device: torch.device,
    sampled_alpha: torch.Tensor | None = None,
    sampled_residual: torch.Tensor | None = None,
    sampled_tile_load: torch.Tensor | None = None,
    tile_capacity: int | None = None,
    footprint_radius_px: float | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if str(target_point_source) not in SUPPORT_BIRTH_SPLIT_TARGET_POINT_SOURCES:
        expected = ", ".join(sorted(SUPPORT_BIRTH_SPLIT_TARGET_POINT_SOURCES))
        raise ValueError(f"support_birth_split.target_point_source must be one of: {expected}")
    if not 0.0 < float(target_top_fraction) <= 1.0:
        raise ValueError("support_birth_split.target_top_fraction must be in (0, 1]")
    if int(max_points) <= 0:
        raise ValueError("support_birth_split.max_points must be positive")
    if int(grid_stride) <= 0 or int(frame_stride) <= 0:
        raise ValueError("support_birth_split grid/frame strides must be positive")
    if target_rgb.dim() != 4 or int(target_rgb.shape[1]) != 3:
        raise ValueError(f"target_rgb must have shape [T,3,H,W], got {tuple(target_rgb.shape)}")
    frames, _channels, _height, _width = (int(item) for item in target_rgb.shape)
    brightness = target_rgb.detach().to(device="cpu", dtype=torch.float32).mean(dim=1)
    sampled_brightness = brightness[:: int(frame_stride), :: int(grid_stride), :: int(grid_stride)].contiguous()
    sampled_alpha_cpu: torch.Tensor | None = None
    needs_alpha = str(target_point_source) != "top_brightness"
    needs_residual = "residual" in str(target_point_source)
    needs_footprint = "footprint" in str(target_point_source)
    needs_tile_load = str(target_point_source).startswith("cap_slack_")
    if needs_alpha:
        if sampled_alpha is None:
            raise ValueError(f"support_birth_split.target_point_source={target_point_source} requires sampled alpha")
        sampled_alpha_cpu = sampled_alpha.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if sampled_alpha_cpu.shape != sampled_brightness.shape:
            raise ValueError(
                "sampled_alpha must match sampled target shape "
                f"{tuple(sampled_brightness.shape)}, got {tuple(sampled_alpha_cpu.shape)}"
            )
    sampled_residual_cpu: torch.Tensor | None = None
    if needs_residual:
        if sampled_residual is None:
            raise ValueError(
                f"support_birth_split.target_point_source={target_point_source} requires sampled residual"
            )
        sampled_residual_cpu = sampled_residual.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if sampled_residual_cpu.shape != sampled_brightness.shape:
            raise ValueError(
                "sampled_residual must match sampled target shape "
                f"{tuple(sampled_brightness.shape)}, got {tuple(sampled_residual_cpu.shape)}"
            )
    footprint_radius_samples = 0
    footprint_score: torch.Tensor | None = None
    if needs_footprint:
        if footprint_radius_px is None or float(footprint_radius_px) <= 0.0:
            raise ValueError("support_birth_split footprint target source requires positive footprint_radius_px")
        footprint_radius_samples = max(1, int(math.ceil(float(footprint_radius_px) / float(grid_stride))))
    sampled_tile_load_cpu: torch.Tensor | None = None
    tile_slack_score: torch.Tensor | None = None
    if needs_tile_load:
        if sampled_tile_load is None:
            raise ValueError(
                f"support_birth_split.target_point_source={target_point_source} requires sampled tile load"
            )
        if tile_capacity is None or int(tile_capacity) <= 0:
            raise ValueError("support_birth_split cap-slack target source requires positive tile_capacity")
        sampled_tile_load_cpu = sampled_tile_load.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if sampled_tile_load_cpu.shape != sampled_brightness.shape:
            raise ValueError(
                "sampled_tile_load must match sampled target shape "
                f"{tuple(sampled_brightness.shape)}, got {tuple(sampled_tile_load_cpu.shape)}"
            )
        slack = (float(tile_capacity) - sampled_tile_load_cpu).clamp_min(0.0)
        tile_slack_score = slack / max(float(tile_capacity), 1.0)
    if str(target_point_source) == "top_brightness":
        score = sampled_brightness
    elif str(target_point_source) == "uncovered_brightness":
        score = sampled_brightness * (1.0 - sampled_alpha_cpu.clamp(0.0, 1.0))
    elif str(target_point_source) == "low_alpha":
        score = 1.0 - sampled_alpha_cpu.clamp(0.0, 1.0)
    elif str(target_point_source) == "residual_uncovered_brightness":
        score = sampled_residual_cpu * sampled_brightness * (1.0 - sampled_alpha_cpu.clamp(0.0, 1.0))
    elif str(target_point_source) == "footprint_residual_uncovered_brightness":
        footprint_score = _support_birth_split_spatial_mean_pool(
            sampled_residual_cpu * sampled_brightness * (1.0 - sampled_alpha_cpu.clamp(0.0, 1.0)),
            radius_samples=footprint_radius_samples,
        )
        score = footprint_score
    elif str(target_point_source) == "cap_slack_uncovered_brightness":
        score = sampled_brightness * (1.0 - sampled_alpha_cpu.clamp(0.0, 1.0)) * tile_slack_score
    elif str(target_point_source) == "cap_slack_low_alpha":
        score = (1.0 - sampled_alpha_cpu.clamp(0.0, 1.0)) * tile_slack_score
    elif str(target_point_source) == "cap_slack_residual_uncovered_brightness":
        score = sampled_residual_cpu * sampled_brightness * (1.0 - sampled_alpha_cpu.clamp(0.0, 1.0)) * tile_slack_score
    elif str(target_point_source) == "cap_slack_footprint_residual_uncovered_brightness":
        footprint_score = _support_birth_split_spatial_mean_pool(
            sampled_residual_cpu * sampled_brightness * (1.0 - sampled_alpha_cpu.clamp(0.0, 1.0)),
            radius_samples=footprint_radius_samples,
        )
        score = footprint_score * tile_slack_score
    else:
        raise AssertionError("unreachable target point source")
    flat = score.flatten()
    if flat.numel() == 0:
        raise ValueError("support_birth_split target sampling produced no candidates")
    keep = max(1, min(int(flat.numel()), int(math.ceil(float(flat.numel()) * float(target_top_fraction)))))
    if keep == int(flat.numel()):
        flat_ids = torch.arange(int(flat.numel()), dtype=torch.int64)
    else:
        flat_ids = torch.topk(flat, k=keep, largest=True, sorted=True).indices.to(torch.int64)
    candidate_count = int(flat_ids.numel())
    if candidate_count > int(max_points):
        select = torch.linspace(0, candidate_count - 1, int(max_points)).round().to(torch.int64)
        flat_ids = flat_ids.index_select(0, select)
    sampled_h = int(sampled_brightness.shape[1])
    sampled_w = int(sampled_brightness.shape[2])
    sampled_frame = flat_ids // (sampled_h * sampled_w)
    remainder = flat_ids % (sampled_h * sampled_w)
    sampled_y = remainder // sampled_w
    sampled_x = remainder % sampled_w
    frame = sampled_frame * int(frame_stride)
    y = (sampled_y * int(grid_stride)).to(torch.float32) + 0.5
    x = (sampled_x * int(grid_stride)).to(torch.float32) + 0.5
    t = frame.to(torch.float32) - 0.5 * float(frames - 1)
    selected_score = flat.index_select(0, flat_ids)
    meta: dict[str, Any] = {
        "target_point_source": str(target_point_source),
        "candidate_count": candidate_count,
        "selected_count": int(flat_ids.numel()),
        "sampled_grid_shape": [int(item) for item in sampled_brightness.shape],
        "selected_score_mean": float(selected_score.mean().item()),
        "selected_score_min": float(selected_score.min().item()),
        "selected_score_max": float(selected_score.max().item()),
    }
    if needs_footprint:
        meta["footprint_radius_px"] = float(footprint_radius_px)
        meta["footprint_radius_samples"] = int(footprint_radius_samples)
    selected_footprint_score: torch.Tensor | None = None
    if footprint_score is not None:
        selected_footprint_score = footprint_score.flatten().index_select(0, flat_ids)
        meta["selected_footprint_score_mean"] = float(selected_footprint_score.mean().item())
        meta["selected_footprint_score_min"] = float(selected_footprint_score.min().item())
        meta["selected_footprint_score_max"] = float(selected_footprint_score.max().item())
    selected_brightness = sampled_brightness.flatten().index_select(0, flat_ids)
    meta["selected_brightness_mean"] = float(selected_brightness.mean().item())
    if sampled_alpha_cpu is not None:
        selected_alpha = sampled_alpha_cpu.flatten().index_select(0, flat_ids)
        meta["selected_alpha_mean"] = float(selected_alpha.mean().item())
        meta["selected_alpha_min"] = float(selected_alpha.min().item())
        meta["selected_alpha_max"] = float(selected_alpha.max().item())
    if sampled_residual_cpu is not None:
        selected_residual = sampled_residual_cpu.flatten().index_select(0, flat_ids)
        meta["selected_residual_mean"] = float(selected_residual.mean().item())
        meta["selected_residual_min"] = float(selected_residual.min().item())
        meta["selected_residual_max"] = float(selected_residual.max().item())
    if sampled_tile_load_cpu is not None and tile_slack_score is not None:
        selected_tile_load = sampled_tile_load_cpu.flatten().index_select(0, flat_ids)
        selected_tile_slack = tile_slack_score.flatten().index_select(0, flat_ids)
        meta["selected_tile_load_mean"] = float(selected_tile_load.mean().item())
        meta["selected_tile_load_min"] = float(selected_tile_load.min().item())
        meta["selected_tile_load_max"] = float(selected_tile_load.max().item())
        meta["selected_tile_slack_mean"] = float(selected_tile_slack.mean().item())
        meta["selected_tile_slack_min"] = float(selected_tile_slack.min().item())
        meta["selected_tile_slack_max"] = float(selected_tile_slack.max().item())
    points = torch.stack((x, y, t), dim=-1).contiguous().to(device=device)
    return points, meta


def _support_birth_split_sample_target_grid_features(
    target_grid: torch.Tensor,
    target_points: torch.Tensor,
    *,
    frames: int,
    height: int,
    width: int,
    mode: str,
) -> torch.Tensor:
    if target_grid.dim() != 4:
        raise ValueError(f"target_grid must have shape [T,F,H,W], got {tuple(target_grid.shape)}")
    if target_points.dim() != 2 or int(target_points.shape[1]) != 3:
        raise ValueError(f"target_points must have shape [N,3], got {tuple(target_points.shape)}")
    if int(frames) <= 0 or int(height) <= 0 or int(width) <= 0:
        raise ValueError("support_birth_split target-grid sampling dimensions must be positive")
    if str(mode) not in {"nearest", "trilinear"}:
        raise ValueError("support_birth_split feature init target-grid mode must be nearest or trilinear")
    if int(target_points.shape[0]) == 0:
        return target_grid.new_empty((0, int(target_grid.shape[1])))

    points = target_points.to(device=target_grid.device, dtype=target_grid.dtype)
    frame = points[:, 2] + 0.5 * float(int(frames) - 1)
    x_norm = (2.0 * points[:, 0] / float(width)) - 1.0
    y_norm = (2.0 * points[:, 1] / float(height)) - 1.0
    z_norm = (2.0 * (frame + 0.5) / float(frames)) - 1.0
    grid = torch.stack((x_norm, y_norm, z_norm), dim=-1).view(1, -1, 1, 1, 3)
    sampled = F.grid_sample(
        target_grid.permute(1, 0, 2, 3).unsqueeze(0),
        grid,
        mode="nearest" if str(mode) == "nearest" else "bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return sampled[0, :, :, 0, 0].transpose(0, 1).contiguous()


def _support_birth_split_target_pixel_ids_for_chunk(
    target_points: torch.Tensor,
    *,
    frames: int,
    height: int,
    width: int,
    frame_start: int,
    chunk_frames: int,
    device: torch.device,
) -> torch.Tensor:
    if target_points.dim() != 2 or int(target_points.shape[1]) != 3:
        raise ValueError(f"target_points must have shape [N,3], got {tuple(target_points.shape)}")
    if int(frames) <= 0 or int(height) <= 0 or int(width) <= 0:
        raise ValueError("support_birth_split target pixel dimensions must be positive")
    if int(frame_start) < 0 or int(chunk_frames) <= 0:
        raise ValueError("support_birth_split frame_start/chunk_frames must be valid")
    if int(target_points.shape[0]) == 0:
        return torch.empty((0,), dtype=torch.int32, device=device)
    points = target_points.detach().to(device="cpu", dtype=torch.float32)
    frame_ids = torch.round(points[:, 2] + 0.5 * float(int(frames) - 1)).to(torch.int64)
    frame_ids = frame_ids.clamp(0, int(frames) - 1)
    mask = (frame_ids >= int(frame_start)) & (frame_ids < int(frame_start) + int(chunk_frames))
    if not bool(mask.any().item()):
        return torch.empty((0,), dtype=torch.int32, device=device)
    selected = points[mask]
    local_frame_ids = frame_ids[mask] - int(frame_start)
    x_ids = torch.floor(selected[:, 0]).to(torch.int64).clamp(0, int(width) - 1)
    y_ids = torch.floor(selected[:, 1]).to(torch.int64).clamp(0, int(height) - 1)
    pixel_ids = local_frame_ids * int(height) * int(width) + y_ids * int(width) + x_ids
    return torch.unique(pixel_ids.to(torch.int32), sorted=True).to(device=device).contiguous()


def _support_birth_split_target_patch_pixel_ids_for_chunk(
    target_points: torch.Tensor,
    *,
    frames: int,
    height: int,
    width: int,
    frame_start: int,
    chunk_frames: int,
    patch_shape: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    if target_points.dim() != 2 or int(target_points.shape[1]) != 3:
        raise ValueError(f"target_points must have shape [N,3], got {tuple(target_points.shape)}")
    if int(frames) <= 0 or int(height) <= 0 or int(width) <= 0:
        raise ValueError("support_birth_split target patch dimensions must be positive")
    if int(frame_start) < 0 or int(chunk_frames) <= 0:
        raise ValueError("support_birth_split frame_start/chunk_frames must be valid")
    patch_h, patch_w = int(patch_shape[0]), int(patch_shape[1])
    if patch_h <= 0 or patch_w <= 0:
        raise ValueError("support_birth_split target_area_patch_shape must be positive")
    if patch_h > int(height) or patch_w > int(width):
        raise ValueError("support_birth_split target_area_patch_shape cannot exceed render size")
    if int(target_points.shape[0]) == 0:
        return torch.empty((0,), dtype=torch.int32, device=device), 0
    points = target_points.detach().to(device="cpu", dtype=torch.float32)
    frame_ids = torch.round(points[:, 2] + 0.5 * float(int(frames) - 1)).to(torch.int64)
    frame_ids = frame_ids.clamp(0, int(frames) - 1)
    mask = (frame_ids >= int(frame_start)) & (frame_ids < int(frame_start) + int(chunk_frames))
    if not bool(mask.any().item()):
        return torch.empty((0,), dtype=torch.int32, device=device), 0
    selected = points[mask]
    local_frame_ids = frame_ids[mask] - int(frame_start)
    center_x = torch.floor(selected[:, 0]).to(torch.int64).clamp(0, int(width) - 1)
    center_y = torch.floor(selected[:, 1]).to(torch.int64).clamp(0, int(height) - 1)
    start_x = (center_x - patch_w // 2).clamp(0, int(width) - patch_w)
    start_y = (center_y - patch_h // 2).clamp(0, int(height) - patch_h)
    y_offsets = torch.arange(patch_h, dtype=torch.int64)
    x_offsets = torch.arange(patch_w, dtype=torch.int64)
    y_ids = start_y[:, None, None] + y_offsets[None, :, None]
    x_ids = start_x[:, None, None] + x_offsets[None, None, :]
    pixel_ids = (
        local_frame_ids[:, None, None] * int(height) * int(width)
        + y_ids * int(width)
        + x_ids
    )
    return pixel_ids.reshape(-1).to(torch.int32).to(device=device).contiguous(), int(selected.shape[0])


def _visibility_proxy_loss(
    model: nn.Module,
    target_points: torch.Tensor,
    *,
    center_weight: float,
    support_weight: float,
    support_epsilon: float,
    max_alpha: float,
    scale_px: float,
    temperature: float,
    velocity_penalty: float,
) -> torch.Tensor:
    if float(center_weight) < 0.0:
        raise ValueError("visibility_proxy.center_weight must be non-negative")
    if float(support_weight) < 0.0:
        raise ValueError("visibility_proxy.support_weight must be non-negative")
    if float(center_weight) <= 0.0 and float(support_weight) <= 0.0:
        raise ValueError("visibility_proxy requires center_weight or support_weight to be positive")
    if float(support_epsilon) <= 0.0:
        raise ValueError("visibility_proxy.support_epsilon must be positive")
    if not 0.0 < float(max_alpha) < 1.0:
        raise ValueError("visibility_proxy max_alpha must be in (0, 1)")
    if float(scale_px) <= 0.0:
        raise ValueError("visibility_proxy.scale_px must be positive")
    if float(temperature) <= 0.0:
        raise ValueError("visibility_proxy.temperature must be positive")
    if target_points.dim() != 2 or int(target_points.shape[1]) != 3:
        raise ValueError(f"target_points must have shape [N,3], got {tuple(target_points.shape)}")
    points = target_points.to(device=model.center_uv.device, dtype=model.center_uv.dtype)
    dt = points[:, 2][:, None] - model.center_t[:, 0][None, :]
    projected = model.center_uv[None, :, :] + dt[:, :, None] * model.velocity_uv[None, :, :]
    dist = (points[:, None, :2] - projected).square().sum(dim=-1) / max(float(scale_px) ** 2, 1.0e-8)
    soft_nearest = -float(temperature) * torch.logsumexp(-dist / float(temperature), dim=1)
    loss = points.new_zeros(())
    if float(center_weight) > 0.0:
        loss = loss + float(center_weight) * soft_nearest.mean()
    if float(support_weight) > 0.0:
        _ma, q_uvt, _depth0, _depth_beta, opacity, _feature = model.tensors()
        delta = torch.cat((points[:, None, :2] - projected, dt[:, :, None]), dim=-1)
        qv = (
            q_uvt[None, :, 0] * delta[:, :, 0] * delta[:, :, 0]
            + 2.0 * q_uvt[None, :, 1] * delta[:, :, 0] * delta[:, :, 1]
            + 2.0 * q_uvt[None, :, 2] * delta[:, :, 0] * delta[:, :, 2]
            + q_uvt[None, :, 3] * delta[:, :, 1] * delta[:, :, 1]
            + 2.0 * q_uvt[None, :, 4] * delta[:, :, 1] * delta[:, :, 2]
            + q_uvt[None, :, 5] * delta[:, :, 2] * delta[:, :, 2]
        )
        alpha = opacity[None, :] * torch.exp(torch.clamp(-0.5 * qv, min=-80.0, max=0.0))
        alpha = torch.clamp(alpha, min=0.0, max=min(float(max_alpha), 1.0 - float(support_epsilon)))
        log_transmittance = torch.log1p(-alpha).sum(dim=1)
        coverage = -torch.expm1(log_transmittance)
        support_loss = -torch.log(torch.clamp(coverage, min=float(support_epsilon))).mean()
        loss = loss + float(support_weight) * support_loss
    if float(velocity_penalty) > 0.0:
        loss = loss + float(velocity_penalty) * model.velocity_uv.square().mean()
    return loss


def _logit_tensor(value: torch.Tensor) -> torch.Tensor:
    clamped = value.clamp(1.0e-5, 1.0 - 1.0e-5)
    return torch.log(clamped) - torch.log1p(-clamped)


def _inv_softplus_tensor(value: torch.Tensor) -> torch.Tensor:
    clamped = value.clamp_min(1.0e-8)
    return clamped + torch.log(-torch.expm1(-clamped))


def _fit_support_birth_split_line(target_points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if target_points.dim() != 2 or int(target_points.shape[1]) != 3:
        raise ValueError(f"target_points must have shape [N,3], got {tuple(target_points.shape)}")
    if int(target_points.shape[0]) <= 0:
        raise ValueError("support_birth_split target sampling produced no candidates")
    points = target_points.to(dtype=torch.float32)
    xy = points[:, :2]
    t = points[:, 2]
    mean_t = t.mean()
    mean_xy = xy.mean(dim=0)
    centered_t = t - mean_t
    denom = centered_t.square().sum()
    if float(denom.detach().cpu().item()) <= 1.0e-8:
        velocity = torch.zeros((2,), dtype=points.dtype, device=points.device)
    else:
        velocity = (centered_t[:, None] * (xy - mean_xy)).sum(dim=0) / denom
    center_uv_at_t0 = mean_xy - velocity * mean_t
    return center_uv_at_t0, velocity


def _support_birth_split_point_groups(
    target_points: torch.Tensor,
    *,
    center_strategy: str,
    center_count: int,
    reallocate_tubes: int,
) -> list[torch.Tensor]:
    groups, _indices = _support_birth_split_point_groups_with_indices(
        target_points,
        center_strategy=center_strategy,
        center_count=center_count,
        reallocate_tubes=reallocate_tubes,
    )
    return groups


def _support_birth_split_point_groups_with_indices(
    target_points: torch.Tensor,
    *,
    center_strategy: str,
    center_count: int,
    reallocate_tubes: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if str(center_strategy) not in SUPPORT_BIRTH_SPLIT_CENTER_STRATEGIES:
        expected = ", ".join(sorted(SUPPORT_BIRTH_SPLIT_CENTER_STRATEGIES))
        raise ValueError(f"support_birth_split.center_strategy must be one of: {expected}")
    if int(center_count) <= 0:
        raise ValueError("support_birth_split.center_count must be positive")
    if int(reallocate_tubes) <= 0:
        raise ValueError("support_birth_split.reallocate_tubes must be positive")
    if str(center_strategy) == "global_line" or int(center_count) == 1:
        indices = torch.arange(int(target_points.shape[0]), dtype=torch.int64, device=target_points.device)
        return [target_points], [indices]

    point_count = int(target_points.shape[0])
    group_count = min(int(center_count), int(reallocate_tubes), point_count)
    xy = target_points[:, :2].to(dtype=torch.float32)
    first_idx = torch.argmax(torch.linalg.vector_norm(xy - xy.mean(dim=0, keepdim=True), dim=1))
    selected = [int(first_idx.detach().cpu().item())]
    min_dist_sq = (xy - xy[selected[0]].view(1, 2)).square().sum(dim=1)
    while len(selected) < group_count:
        next_idx = torch.argmax(min_dist_sq)
        selected.append(int(next_idx.detach().cpu().item()))
        next_dist_sq = (xy - xy[selected[-1]].view(1, 2)).square().sum(dim=1)
        min_dist_sq = torch.minimum(min_dist_sq, next_dist_sq)
    centers = xy.index_select(0, torch.tensor(selected, dtype=torch.int64, device=target_points.device))
    assignments = torch.argmin((xy[:, None, :] - centers[None, :, :]).square().sum(dim=-1), dim=1)
    group_indices = [torch.where(assignments == idx)[0] for idx in range(group_count)]
    groups = [target_points.index_select(0, indices) for indices in group_indices]
    non_empty = [(group, indices) for group, indices in zip(groups, group_indices, strict=True) if int(group.shape[0]) > 0]
    return [group for group, _indices in non_empty], [indices for _group, indices in non_empty]


def _support_birth_split_tube_counts(
    point_groups: list[torch.Tensor],
    reallocate_tubes: int,
    *,
    tube_allocation: str = "proportional",
) -> list[int]:
    if not point_groups:
        raise ValueError("support_birth_split target grouping produced no groups")
    group_count = len(point_groups)
    if int(reallocate_tubes) < group_count:
        raise ValueError("support_birth_split.reallocate_tubes must be >= the active center count")
    if str(tube_allocation) not in SUPPORT_BIRTH_SPLIT_TUBE_ALLOCATIONS:
        expected = ", ".join(sorted(SUPPORT_BIRTH_SPLIT_TUBE_ALLOCATIONS))
        raise ValueError(f"support_birth_split.tube_allocation must be one of: {expected}")
    if str(tube_allocation) == "uniform":
        base = int(reallocate_tubes) // group_count
        remainder = int(reallocate_tubes) - base * group_count
        counts = torch.full((group_count,), base, dtype=torch.int64).clamp_min(1)
        if remainder > 0:
            sizes = torch.tensor([int(group.shape[0]) for group in point_groups], dtype=torch.float32)
            order = torch.argsort(sizes, descending=True)
            counts.index_add_(0, order[:remainder], torch.ones((remainder,), dtype=torch.int64))
        return [int(item) for item in counts.tolist()]
    sizes = torch.tensor([int(group.shape[0]) for group in point_groups], dtype=torch.float32)
    ideal = sizes / sizes.sum().clamp_min(1.0) * float(reallocate_tubes)
    counts = torch.floor(ideal).to(dtype=torch.int64).clamp_min(1)
    while int(counts.sum().item()) > int(reallocate_tubes):
        fractional = ideal - torch.floor(ideal)
        candidates = torch.where(counts > 1, fractional, torch.full_like(fractional, float("inf")))
        counts[int(torch.argmin(candidates).item())] -= 1
    while int(counts.sum().item()) < int(reallocate_tubes):
        fractional = ideal - torch.floor(ideal)
        counts[int(torch.argmax(fractional).item())] += 1
    return [int(item) for item in counts.tolist()]


def _support_birth_split_offsets(
    count: int,
    radius_px: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    support_shape: str = "isotropic",
    velocity_uv: torch.Tensor | None = None,
    radius_along_px: float | None = None,
    radius_across_px: float | None = None,
) -> torch.Tensor:
    if int(count) <= 0:
        raise ValueError("support_birth_split.reallocate_tubes must be positive")
    if float(radius_px) <= 0.0:
        raise ValueError("support_birth_split.support_radius_px must be positive")
    if str(support_shape) not in SUPPORT_BIRTH_SPLIT_SHAPES:
        expected = ", ".join(sorted(SUPPORT_BIRTH_SPLIT_SHAPES))
        raise ValueError(f"support_birth_split.support_shape must be one of: {expected}")
    tube_ids = torch.arange(int(count), dtype=dtype, device=device)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    angles = tube_ids * golden_angle
    unit_radii = torch.sqrt((tube_ids + 0.5) / float(count))
    if str(support_shape) == "isotropic":
        radii = float(radius_px) * unit_radii
        offsets = torch.stack((torch.cos(angles) * radii, torch.sin(angles) * radii), dim=-1)
    else:
        if velocity_uv is None:
            raise ValueError("support_birth_split.support_shape=trajectory_ellipse requires velocity_uv")
        if radius_along_px is None or radius_across_px is None:
            raise ValueError("support_birth_split.support_shape=trajectory_ellipse requires along/across radii")
        if float(radius_along_px) <= 0.0 or float(radius_across_px) <= 0.0:
            raise ValueError("support_birth_split support ellipse radii must be positive")
        velocity = velocity_uv.to(device=device, dtype=dtype).flatten()
        if int(velocity.numel()) != 2:
            raise ValueError(f"velocity_uv must have shape [2], got {tuple(velocity.shape)}")
        speed = torch.linalg.vector_norm(velocity)
        if float(speed.detach().cpu().item()) <= 1.0e-6:
            along = torch.tensor([1.0, 0.0], dtype=dtype, device=device)
        else:
            along = velocity / speed
        across = torch.stack((-along[1], along[0]))
        local_along = torch.cos(angles) * unit_radii * float(radius_along_px)
        local_across = torch.sin(angles) * unit_radii * float(radius_across_px)
        offsets = local_along[:, None] * along.view(1, 2) + local_across[:, None] * across.view(1, 2)
    offsets[0] = 0.0
    return offsets


def _support_birth_split_tube_ids(model: nn.Module, count: int, selection: str) -> torch.Tensor:
    tube_count = int(model.center_uv.shape[0])
    if int(count) <= 0:
        raise ValueError("support_birth_split.reallocate_tubes must be positive")
    if int(count) > tube_count:
        raise ValueError("support_birth_split.reallocate_tubes cannot exceed feature_uvt.tube_count")
    if selection == "first":
        return torch.arange(int(count), dtype=torch.int64, device=model.center_uv.device)
    if selection == "lowest_opacity":
        return torch.topk(model.raw_opacity.detach().flatten(), k=int(count), largest=False, sorted=True).indices
    expected = ", ".join(sorted(SUPPORT_BIRTH_SPLIT_TUBE_SELECTIONS))
    raise ValueError(f"support_birth_split.tube_selection must be one of: {expected}")


def _support_birth_split_repair_tile_overflow_ids(
    tile_counts: torch.Tensor,
    tile_tube_ids: torch.Tensor,
    selected_tube_ids: torch.Tensor,
    *,
    tile_capacity: int,
    max_drops: int,
    guard_refs: int = 0,
) -> dict[str, Any]:
    if int(tile_capacity) <= 0:
        raise ValueError("support_birth_split tile_capacity must be positive")
    if int(max_drops) < 0:
        raise ValueError("support_birth_split.tile_overflow_repair_max_drops must be non-negative")
    if int(guard_refs) < 0:
        raise ValueError("support_birth_split.tile_overflow_repair_guard_refs must be non-negative")
    target_capacity = max(0, int(tile_capacity) - int(guard_refs))
    if tile_counts.dim() != 1:
        raise ValueError(f"tile_counts must have shape [tile_count], got {tuple(tile_counts.shape)}")
    if tile_tube_ids.dim() != 1 or int(tile_tube_ids.numel()) != int(tile_counts.numel()) * int(tile_capacity):
        raise ValueError("tile_tube_ids must have shape [tile_count * tile_capacity]")
    selected = {int(item) for item in selected_tube_ids.detach().cpu().reshape(-1).tolist()}
    counts = tile_counts.detach().cpu().to(torch.int64).clone()
    ids = tile_tube_ids.detach().cpu().to(torch.int64).reshape(int(tile_counts.numel()), int(tile_capacity))
    repair_tiles = torch.where(counts > target_capacity)[0].to(torch.int64)
    tile_selected: dict[int, set[int]] = {}
    for tile_id_tensor in repair_tiles:
        tile_id = int(tile_id_tensor.item())
        stored_count = min(int(counts[tile_id].item()), int(tile_capacity))
        present = {
            int(item)
            for item in ids[tile_id, :stored_count].tolist()
            if int(item) in selected
        }
        if present:
            tile_selected[tile_id] = present
    dropped: list[int] = []
    while bool(torch.any(counts > target_capacity).item()) and len(dropped) < int(max_drops):
        scores: dict[int, int] = {}
        for tile_id, present in tile_selected.items():
            if int(counts[tile_id].item()) <= target_capacity:
                continue
            for tube_id in present:
                if tube_id not in dropped:
                    scores[tube_id] = scores.get(tube_id, 0) + 1
        if not scores:
            break
        drop_id = max(scores, key=lambda tube_id: (scores[tube_id], -tube_id))
        dropped.append(int(drop_id))
        for tile_id, present in tile_selected.items():
            if drop_id in present:
                counts[tile_id] -= 1
                present.remove(drop_id)
    remaining = torch.clamp(counts - int(tile_capacity), min=0)
    initial = torch.clamp(tile_counts.detach().cpu().to(torch.int64) - int(tile_capacity), min=0)
    return {
        "enabled": True,
        "requested_max_drops": int(max_drops),
        "requested_guard_refs": int(guard_refs),
        "target_capacity": int(target_capacity),
        "dropped_tube_ids": dropped,
        "drop_count": len(dropped),
        "initial_overflow_tile_count": int((initial > 0).sum().item()),
        "initial_overflow_excess_tube_refs": int(initial.sum().item()),
        "estimated_remaining_overflow_tile_count": int((remaining > 0).sum().item()),
        "estimated_remaining_overflow_excess_tube_refs": int(remaining.sum().item()),
    }


@torch.no_grad()
def _support_birth_split_set_tube_opacity(
    model: nn.Module,
    tube_ids: torch.Tensor,
    *,
    opacity: float,
) -> None:
    if not 0.0 < float(opacity) < 0.99:
        raise ValueError("support_birth_split repair opacity must be in (0, 0.99)")
    ids = tube_ids.to(device=model.raw_opacity.device, dtype=torch.int64).reshape(-1)
    if int(ids.numel()) == 0:
        return
    raw_opacity = _logit_tensor(
        torch.full(
            (int(ids.numel()),),
            float(opacity) / 0.99,
            dtype=model.raw_opacity.dtype,
            device=model.raw_opacity.device,
        )
    )
    model.raw_opacity.index_copy_(0, ids, raw_opacity)


@torch.no_grad()
def _apply_support_birth_split(
    model: nn.Module,
    target_points: torch.Tensor,
    *,
    reallocate_tubes: int,
    support_radius_px: float,
    support_shape: str,
    support_radius_along_px: float,
    support_radius_across_px: float,
    support_precision_radius_px: float,
    temporal_radius_frames: float,
    opacity: float,
    max_alpha: float,
    tube_selection: str,
    center_strategy: str = "global_line",
    center_count: int = 1,
    tube_allocation: str = "proportional",
    target_point_features: torch.Tensor | None = None,
    feature_init_mode: str = "preserve",
) -> dict[str, Any]:
    if float(temporal_radius_frames) <= 0.0:
        raise ValueError("support_birth_split.temporal_radius_frames must be positive")
    if str(support_shape) not in SUPPORT_BIRTH_SPLIT_SHAPES:
        expected = ", ".join(sorted(SUPPORT_BIRTH_SPLIT_SHAPES))
        raise ValueError(f"support_birth_split.support_shape must be one of: {expected}")
    if float(support_radius_along_px) <= 0.0:
        raise ValueError("support_birth_split.support_radius_along_px must be positive")
    if float(support_radius_across_px) <= 0.0:
        raise ValueError("support_birth_split.support_radius_across_px must be positive")
    if float(support_precision_radius_px) <= 0.0:
        raise ValueError("support_birth_split.support_precision_radius_px must be positive")
    if not 0.0 < float(opacity) < float(max_alpha):
        raise ValueError("support_birth_split.opacity must be in (0, feature_uvt.max_alpha)")
    if str(center_strategy) not in SUPPORT_BIRTH_SPLIT_CENTER_STRATEGIES:
        expected = ", ".join(sorted(SUPPORT_BIRTH_SPLIT_CENTER_STRATEGIES))
        raise ValueError(f"support_birth_split.center_strategy must be one of: {expected}")
    if int(center_count) <= 0:
        raise ValueError("support_birth_split.center_count must be positive")
    if str(tube_allocation) not in SUPPORT_BIRTH_SPLIT_TUBE_ALLOCATIONS:
        expected = ", ".join(sorted(SUPPORT_BIRTH_SPLIT_TUBE_ALLOCATIONS))
        raise ValueError(f"support_birth_split.tube_allocation must be one of: {expected}")
    if str(feature_init_mode) not in SUPPORT_BIRTH_SPLIT_FEATURE_INIT_MODES:
        expected = ", ".join(sorted(SUPPORT_BIRTH_SPLIT_FEATURE_INIT_MODES))
        raise ValueError(f"support_birth_split.feature_init_mode must be one of: {expected}")
    if str(feature_init_mode) != "preserve":
        if target_point_features is None:
            raise ValueError("support_birth_split.feature_init_mode requires target_point_features")
        if target_point_features.dim() != 2:
            raise ValueError(f"target_point_features must have shape [N,F], got {tuple(target_point_features.shape)}")
        if int(target_point_features.shape[0]) != int(target_points.shape[0]):
            raise ValueError("target_point_features must have one feature vector per target point")
        if int(target_point_features.shape[1]) != int(model.raw_feature.shape[1]):
            raise ValueError("target_point_features feature dimension must match model.raw_feature")
    target_points_device = target_points.to(device=model.center_uv.device, dtype=model.center_uv.dtype)
    point_groups, point_group_indices = _support_birth_split_point_groups_with_indices(
        target_points_device,
        center_strategy=str(center_strategy),
        center_count=int(center_count),
        reallocate_tubes=int(reallocate_tubes),
    )
    tube_counts = _support_birth_split_tube_counts(
        point_groups,
        int(reallocate_tubes),
        tube_allocation=str(tube_allocation),
    )
    selected_ids = _support_birth_split_tube_ids(model, int(reallocate_tubes), str(tube_selection))
    selected_before = (torch.sigmoid(model.raw_opacity.detach().flatten().index_select(0, selected_ids)) * 0.99).cpu()
    selected_feature_before = model.raw_feature.detach().index_select(0, selected_ids).cpu()
    center_chunks = []
    velocity_chunks = []
    feature_chunks = []
    center_state = []
    velocity_state = []
    target_point_features_device = None if target_point_features is None else target_point_features.to(
        device=model.raw_feature.device,
        dtype=model.raw_feature.dtype,
    )
    for group, group_indices, tube_count in zip(point_groups, point_group_indices, tube_counts, strict=True):
        center_uv_at_t0, velocity_uv = _fit_support_birth_split_line(group)
        offsets = _support_birth_split_offsets(
            int(tube_count),
            float(support_radius_px),
            device=model.center_uv.device,
            dtype=model.center_uv.dtype,
            support_shape=str(support_shape),
            velocity_uv=velocity_uv,
            radius_along_px=float(support_radius_along_px),
            radius_across_px=float(support_radius_across_px),
        )
        center_chunks.append(center_uv_at_t0.view(1, 2) + offsets)
        velocity_chunks.append(velocity_uv.view(1, 2).expand(int(tube_count), 2))
        if str(feature_init_mode) == "target_group_mean":
            group_features = target_point_features_device.index_select(
                0,
                group_indices.to(device=model.raw_feature.device, dtype=torch.int64),
            )
            feature_chunks.append(group_features.mean(dim=0, keepdim=True).expand(int(tube_count), -1))
        center_state.append([float(item) for item in center_uv_at_t0.detach().cpu().tolist()])
        velocity_state.append([float(item) for item in velocity_uv.detach().cpu().tolist()])
    center_uv_values = torch.cat(center_chunks, dim=0)
    velocity_uv_values = torch.cat(velocity_chunks, dim=0)
    fit_centers_tensor = torch.tensor(center_state, dtype=model.center_uv.dtype, device=model.center_uv.device)
    fit_velocities_tensor = torch.tensor(velocity_state, dtype=model.center_uv.dtype, device=model.center_uv.device)
    spatial_precision = 2.5 / max(float(support_precision_radius_px) ** 2, 1.0e-8)
    temporal_precision = 2.5 / max(float(temporal_radius_frames) ** 2, 1.0e-8)
    precision = torch.tensor(
        [spatial_precision, spatial_precision, temporal_precision],
        dtype=model.center_uv.dtype,
        device=model.center_uv.device,
    )
    raw_precision = _inv_softplus_tensor(precision - float(model.min_precision)).expand(int(reallocate_tubes), 3)
    raw_opacity = _logit_tensor(
        torch.full(
            (int(reallocate_tubes),),
            float(opacity) / 0.99,
            dtype=model.center_uv.dtype,
            device=model.center_uv.device,
        )
    )
    model.center_uv.index_copy_(0, selected_ids, center_uv_values)
    model.center_t.index_copy_(0, selected_ids, torch.zeros((int(reallocate_tubes), 1), device=model.center_t.device))
    model.velocity_uv.index_copy_(0, selected_ids, velocity_uv_values)
    model.raw_precision.index_copy_(0, selected_ids, raw_precision)
    if hasattr(model, "raw_spatial_correlation"):
        model.raw_spatial_correlation.index_fill_(0, selected_ids, 0.0)
    model.raw_opacity.index_copy_(0, selected_ids, raw_opacity)
    if feature_chunks:
        model.raw_feature.index_copy_(0, selected_ids, torch.cat(feature_chunks, dim=0).contiguous())
    selected_after = (torch.sigmoid(model.raw_opacity.detach().flatten().index_select(0, selected_ids)) * 0.99).cpu()
    selected_feature_after = model.raw_feature.detach().index_select(0, selected_ids).cpu()
    return {
        "enabled": True,
        "target_point_count": int(target_points.shape[0]),
        "reallocated_tubes": int(reallocate_tubes),
        "tube_selection": str(tube_selection),
        "selected_tube_ids": [int(item) for item in selected_ids.detach().cpu().tolist()],
        "support_radius_px": float(support_radius_px),
        "support_shape": str(support_shape),
        "support_radius_along_px": float(support_radius_along_px),
        "support_radius_across_px": float(support_radius_across_px),
        "support_precision_radius_px": float(support_precision_radius_px),
        "temporal_radius_frames": float(temporal_radius_frames),
        "requested_opacity": float(opacity),
        "center_strategy": str(center_strategy),
        "requested_center_count": int(center_count),
        "actual_center_count": len(point_groups),
        "tube_allocation": str(tube_allocation),
        "feature_init_mode": str(feature_init_mode),
        "feature_init_applied": bool(feature_chunks),
        "center_point_counts": [int(group.shape[0]) for group in point_groups],
        "center_tube_counts": tube_counts,
        "spatial_precision": float(spatial_precision),
        "temporal_precision": float(temporal_precision),
        "fit_center_uv_at_t0": [float(item) for item in fit_centers_tensor.detach().mean(dim=0).cpu().tolist()],
        "fit_velocity_uv": [float(item) for item in fit_velocities_tensor.detach().mean(dim=0).cpu().tolist()],
        "fit_centers_uv_at_t0": center_state,
        "fit_velocities_uv": velocity_state,
        "selected_opacity_mean_before": float(selected_before.mean().item()),
        "selected_opacity_mean_after": float(selected_after.mean().item()),
        "selected_feature_abs_mean_before": float(selected_feature_before.abs().mean().item()),
        "selected_feature_abs_mean_after": float(selected_feature_after.abs().mean().item()),
        "tube_count_preserved": int(model.center_uv.shape[0]) == int(model.tube_count),
    }
