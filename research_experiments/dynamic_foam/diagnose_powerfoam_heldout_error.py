from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from config_utils import load_config_file
from powerfoam_eval_render import render_powerfoam_samples
from powerfoam_metal_config import resolve_config
from powerfoam_objectives import composite_fixed_background, normals_from_ray_depth
from powerfoam_point_cloud import load_powerfoam_point_cloud_initialization
from powerfoam_raster_config import make_powerfoam_metal_raster_config as make_raster_config
from powerfoam_training_data import load_powerfoam_training_data
try:
    from .report_artifacts import relative_to_project as rel, write_report_json
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import relative_to_project as rel, write_report_json
from train_devices import resolve_torch_device
from powerfoam_metal_trainer import MetalPowerFoamVideo
from video_io import save_png


def scalar(value: torch.Tensor | float | int) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def quantiles(values: torch.Tensor, qs: tuple[float, ...]) -> dict[str, float]:
    flat = values.detach().flatten().to(dtype=torch.float32)
    return {f"p{int(q * 100):02d}": scalar(torch.quantile(flat, q)) for q in qs}


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float | None:
    if not bool(mask.any()):
        return None
    return scalar(values[mask].mean())


def masked_sum(values: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return 0.0
    return scalar(values[mask].sum())


def masked_quantiles(values: torch.Tensor, mask: torch.Tensor, qs: tuple[float, ...]) -> dict[str, float]:
    if not bool(mask.any()):
        return {}
    return quantiles(values[mask], qs)


def pearson_corr(values: torch.Tensor, residual: torch.Tensor, mask: torch.Tensor) -> float | None:
    mask = mask & torch.isfinite(values) & torch.isfinite(residual)
    if int(mask.sum().item()) < 2:
        return None
    x = values[mask].to(dtype=torch.float32)
    y = residual[mask].to(dtype=torch.float32)
    x = x - x.mean()
    y = y - y.mean()
    denom = x.square().sum().sqrt() * y.square().sum().sqrt()
    if float(denom.item()) <= 1.0e-12:
        return None
    return scalar((x * y).sum() / denom)


def sample_view_indices(view_names: list[str], *, frame_count: int, sample_count: int) -> torch.Tensor | None:
    if not view_names:
        return None
    expected = int(frame_count) * len(view_names)
    if sample_count != expected:
        raise ValueError(
            f"Expected {expected} samples for {len(view_names)} views x {frame_count} frames; got {sample_count}."
        )
    return torch.arange(len(view_names), dtype=torch.long).repeat_interleave(int(frame_count))


def label_for_sample(
    sample_index: int,
    *,
    frame_indices: torch.Tensor,
    view_names: list[str],
    frame_count: int,
) -> dict[str, Any]:
    label: dict[str, Any] = {
        "sample": int(sample_index),
        "frame": int(frame_indices.detach().cpu().to(dtype=torch.long)[sample_index].item()),
    }
    view_indices = sample_view_indices(view_names, frame_count=frame_count, sample_count=int(frame_indices.numel()))
    if view_indices is not None:
        view_index = int(view_indices[int(sample_index)].item())
        label["view_index"] = view_index
        label["view_name"] = view_names[view_index]
    return label


def grouped_metric_row(
    labels: dict[str, Any],
    mask: torch.Tensor,
    *,
    sample_l1: torch.Tensor,
    sample_mse: torch.Tensor,
    alpha: torch.Tensor,
    target_luma: torch.Tensor,
) -> dict[str, Any]:
    group_alpha = alpha[mask]
    group_mse = sample_mse[mask].mean()
    return {
        **labels,
        "sample_count": int(mask.sum().item()),
        "l1": scalar(sample_l1[mask].mean()),
        "mse": scalar(group_mse),
        "psnr": scalar(-10.0 * torch.log10(group_mse.clamp_min(1.0e-12))),
        "alpha_mean": scalar(group_alpha.mean()),
        "alpha_fraction_lt_0_05": scalar((group_alpha < 0.05).to(torch.float32).mean()),
        "alpha_fraction_gt_0_50": scalar((group_alpha > 0.5).to(torch.float32).mean()),
        "low_alpha_target_luma": masked_mean(target_luma[mask], group_alpha < 0.05),
    }


def pixel_bucket_summary(
    *,
    rendered: torch.Tensor,
    alpha: torch.Tensor,
    targets: torch.Tensor,
    normal_distance: torch.Tensor | None,
    residual_quantile: float,
) -> dict[str, Any]:
    rendered = rendered.detach().cpu().to(dtype=torch.float32)
    alpha = alpha.detach().cpu().to(dtype=torch.float32)
    targets = targets.detach().cpu().to(dtype=torch.float32)
    normal_distance = None if normal_distance is None else normal_distance.detach().cpu().to(dtype=torch.float32)
    pixel_l1 = (rendered - targets).abs().mean(dim=1)
    residual_threshold = scalar(torch.quantile(pixel_l1.flatten(), float(residual_quantile)))
    high_residual = pixel_l1 >= float(residual_threshold)
    alpha_buckets = {
        "alpha_low_lt_0_05": alpha < 0.05,
        "alpha_mid_0_05_to_0_50": (alpha >= 0.05) & (alpha < 0.5),
        "alpha_high_gte_0_50": alpha >= 0.5,
        "alpha_opaque_gte_0_90": alpha >= 0.9,
    }
    total_residual = float(pixel_l1.sum().clamp_min(1.0e-12))
    rows = []
    for alpha_name, alpha_mask in alpha_buckets.items():
        for residual_name, residual_mask in (("all", torch.ones_like(alpha_mask, dtype=torch.bool)), ("high_residual", high_residual)):
            mask = alpha_mask & residual_mask
            row = {
                "bucket": f"{alpha_name}/{residual_name}",
                "pixel_fraction": scalar(mask.to(torch.float32).mean()),
                "residual_l1_mean": masked_mean(pixel_l1, mask),
                "residual_l1_sum": masked_sum(pixel_l1, mask),
                "residual_share": masked_sum(pixel_l1, mask) / total_residual,
                "alpha_mean": masked_mean(alpha, mask),
            }
            if normal_distance is not None:
                row["normal_distance_mean"] = masked_mean(normal_distance, mask)
            rows.append(row)
    dominant = max(rows, key=lambda row: float(row["residual_share"]))
    dominant_high = max(
        [row for row in rows if row["bucket"].endswith("/high_residual")],
        key=lambda row: float(row["residual_share"]),
    )
    return {
        "residual_quantile": float(residual_quantile),
        "residual_threshold": residual_threshold,
        "total_residual_l1": total_residual,
        "dominant_bucket": dominant,
        "dominant_high_residual_bucket": dominant_high,
        "buckets": rows,
    }


def load_point_cloud_init(cfg: dict[str, Any], training_data: dict[str, Any]) -> Any | None:
    if cfg["model"]["init_point_cloud_path"] is None:
        return None
    point_cloud_coordinate_frame = str(cfg["model"]["init_point_cloud_coordinate_frame"])
    point_transform = None
    if point_cloud_coordinate_frame == "multicam_world":
        point_transform = training_data.get("world_to_model")
        if point_transform is None:
            raise ValueError("multicam_world point cloud init requires world_to_model metadata.")
    return load_powerfoam_point_cloud_initialization(
        path=cfg["model"]["init_point_cloud_path"],
        frame_count=int(training_data["frame_count"]),
        cell_count=int(cfg["model"]["cells"]),
        xy_extent=float(cfg["model"]["xy_extent"]),
        z_min=float(cfg["model"]["z_min"]),
        z_max=float(cfg["model"]["z_max"]),
        normalize_mode=str(cfg["model"]["init_point_cloud_normalize"]),
        coordinate_frame=point_cloud_coordinate_frame,
        point_transform=point_transform,
        visibility_filter=str(cfg["model"]["init_point_cloud_visibility_filter"]),
        min_visible_train_views=int(cfg["model"]["init_point_cloud_min_visible_train_views"]),
        visibility_train_K=training_data.get("point_cloud_visibility_train_K"),
        visibility_train_w2c=training_data.get("point_cloud_visibility_train_w2c"),
        visibility_train_lens_models=training_data.get("point_cloud_visibility_train_lens_models"),
        visibility_train_distortions=training_data.get("point_cloud_visibility_train_distortions"),
        visibility_render_size=int(cfg["render"]["render_size"]),
        sample_mode=str(cfg["model"]["init_point_cloud_sample_mode"]),
        duplicate_jitter=float(cfg["model"]["init_point_cloud_duplicate_jitter"]),
        seed=int(cfg["train"]["seed"]),
    )


def build_model(cfg: dict[str, Any], training_data: dict[str, Any], device: torch.device) -> MetalPowerFoamVideo:
    point_cloud_init = load_point_cloud_init(cfg, training_data)
    return MetalPowerFoamVideo(
        frame_count=int(training_data["frame_count"]),
        cell_count=int(cfg["model"]["cells"]),
        render_size=int(cfg["render"]["render_size"]),
        fov_degrees=float(cfg["render"]["fov_degrees"]),
        neighbor_count=int(cfg["model"]["neighbor_count"]),
        adjacency_mode=str(cfg["model"]["adjacency_mode"]),
        xy_extent=float(cfg["model"]["xy_extent"]),
        z_min=float(cfg["model"]["z_min"]),
        z_max=float(cfg["model"]["z_max"]),
        radius_init=float(cfg["model"]["radius_init"]),
        radius_min=float(cfg["model"]["radius_min"]),
        radius_scale=float(cfg["model"]["radius_scale"]),
        density_init=float(cfg["model"]["density_init"]),
        feature_mode=str(cfg["model"]["feature_mode"]),
        linear_coeff_init=float(cfg["model"]["linear_coeff_init"]),
        linear_coeff_scale=float(cfg["model"]["linear_coeff_scale"]),
        normal_init_jitter=float(cfg["model"]["normal_init_jitter"]),
        num_texel_sites=int(cfg["model"]["num_texel_sites"]),
        texel_site_scale=float(cfg["model"]["texel_site_scale"]),
        texel_height_scale=float(cfg["model"]["texel_height_scale"]),
        sv_dof=int(cfg["model"]["sv_dof"]),
        sv_axis_init=float(cfg["model"]["sv_axis_init"]),
        sv_axis_init_jitter=float(cfg["model"]["sv_axis_init_jitter"]),
        sv_rgb_init_jitter=float(cfg["model"]["sv_rgb_init_jitter"]),
        color_init_mode=str(cfg["model"]["color_init_mode"]),
        seed=int(cfg["train"]["seed"]),
        init_frames=training_data["init_frames"] if bool(cfg["model"]["init_from_video"]) else None,
        init_points=None if point_cloud_init is None else point_cloud_init.points,
        init_colors=None if point_cloud_init is None else point_cloud_init.colors,
        image_init_depth=None if cfg["model"]["image_init_depth"] is None else float(cfg["model"]["image_init_depth"]),
        image_init_jitter=float(cfg["model"]["image_init_jitter"]),
        raster_config=make_raster_config(cfg["render"]),
        use_raytrace=bool(cfg["render"]["use_raytrace"]),
    ).to(device)


@torch.no_grad()
def render_split(
    model: MetalPowerFoamVideo,
    cfg: dict[str, Any],
    frame_indices: torch.Tensor,
    rays: torch.Tensor | None,
    *,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rendered, alpha = render_powerfoam_samples(model, frame_indices, rays=rays, batch_size=batch_size)
    rendered = composite_fixed_background(rendered, alpha, cfg["render"])
    return rendered.clamp(0.0, 1.0), alpha.clamp(0.0, 1.0)


@torch.no_grad()
def render_split_with_normal_distance(
    model: MetalPowerFoamVideo,
    cfg: dict[str, Any],
    frame_indices: torch.Tensor,
    rays: torch.Tensor | None,
    *,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if model.feature_mode not in {"oriented_height_sv_texel_surface", "quaternion_height_sv_texel_surface"}:
        rendered, alpha = render_split(model, cfg, frame_indices, rays, batch_size=batch_size)
        return rendered, alpha, None, None
    renders = []
    alphas = []
    normal_distances = []
    rendered_normals = []
    device = next(model.parameters()).device
    frame_indices = frame_indices.to(device=device, dtype=torch.long)
    batch_rays_all = None if rays is None else rays.to(device=device, dtype=torch.float32)
    for start in range(0, int(frame_indices.numel()), int(batch_size)):
        end = min(start + int(batch_size), int(frame_indices.numel()))
        indices = frame_indices[start:end]
        batch_rays = None if batch_rays_all is None else batch_rays_all[start:end]
        if model.use_raytrace:
            rendered, alpha, normal_distance, rendered_normal = model(
                indices,
                rays=batch_rays,
                return_normal_distance=True,
                return_rendered_normal=True,
            )
            rendered_normals.append(rendered_normal.detach().cpu())
        else:
            rendered, alpha, normal_distance = model(indices, rays=batch_rays, return_normal_distance=True)
        renders.append(rendered.detach().cpu())
        alphas.append(alpha.detach().cpu())
        normal_distances.append(normal_distance.detach().cpu())
    rendered = torch.cat(renders, dim=0)
    alpha = torch.cat(alphas, dim=0)
    rendered = composite_fixed_background(rendered, alpha, cfg["render"])
    rendered_normal_out = torch.cat(rendered_normals, dim=0) if rendered_normals else None
    return rendered.clamp(0.0, 1.0), alpha.clamp(0.0, 1.0), torch.cat(normal_distances, dim=0), rendered_normal_out


def decoded_points_radii(model: MetalPowerFoamVideo) -> tuple[torch.Tensor, torch.Tensor]:
    points, radii, _densities, _features, _normals = model.decoded_parameters()
    return points.detach(), radii.detach()


@torch.no_grad()
def ray_support_maps(
    points: torch.Tensor,
    radii: torch.Tensor,
    rays: torch.Tensor,
    cfg: dict[str, Any],
    *,
    chunk_size: int,
) -> dict[str, torch.Tensor]:
    if rays.ndim == 3:
        rays_flat = rays.reshape(-1, 6)
        height, width = int(rays.shape[0]), int(rays.shape[1])
    elif rays.ndim == 4 and int(rays.shape[0]) == 1:
        rays_flat = rays[0].reshape(-1, 6)
        height, width = int(rays.shape[1]), int(rays.shape[2])
    else:
        raise ValueError(f"Expected rays [H,W,6] or [1,H,W,6], got {tuple(rays.shape)}.")
    device = points.device
    rays_flat = rays_flat.to(device=device, dtype=points.dtype)
    radius2 = radii.square().view(1, -1)
    hit_counts = []
    nearest_depths = []
    nearest_power_values = []
    near_plane = float(cfg["render"]["near_plane"])
    eps = float(cfg["render"]["eps"])
    for start in range(0, int(rays_flat.shape[0]), int(chunk_size)):
        chunk = rays_flat[start : start + int(chunk_size)]
        origins = chunk[:, :3]
        dirs = F.normalize(chunk[:, 3:], dim=-1, eps=eps)
        rel = points.view(1, -1, 3) - origins[:, None, :]
        t = (rel * dirs[:, None, :]).sum(dim=-1)
        t_clamped = t.clamp_min(near_plane)
        closest = rel - t_clamped[..., None] * dirs[:, None, :]
        power = closest.square().sum(dim=-1) - radius2
        sphere_front = t > near_plane
        hit = (power <= 0.0) & sphere_front
        hit_counts.append(hit.sum(dim=1).to(dtype=torch.float32).detach().cpu())
        nearest_power, nearest = power.detach().min(dim=1)
        nearest_power_values.append(nearest_power.detach().cpu())
        nearest_depths.append(t.gather(1, nearest[:, None]).squeeze(1).detach().cpu())
    return {
        "hit_count": torch.cat(hit_counts).reshape(height, width),
        "nearest_depth": torch.cat(nearest_depths).reshape(height, width),
        "nearest_power": torch.cat(nearest_power_values).reshape(height, width),
    }


def normalize_map(values: torch.Tensor, *, invert: bool = False) -> torch.Tensor:
    values = values.detach().cpu().to(dtype=torch.float32)
    finite = torch.isfinite(values)
    out = torch.zeros_like(values)
    if bool(finite.any()):
        src = values[finite]
        lo = torch.quantile(src, 0.02)
        hi = torch.quantile(src, 0.98)
        if float((hi - lo).abs()) < 1.0e-8:
            out[finite] = 0.0
        else:
            out[finite] = ((values[finite] - lo) / (hi - lo)).clamp(0.0, 1.0)
    if invert:
        out = 1.0 - out
    return out


def gray_rgb(values: torch.Tensor) -> torch.Tensor:
    image = values.detach().cpu().to(dtype=torch.float32).clamp(0.0, 1.0)
    return image.unsqueeze(0).repeat(3, 1, 1)


def normal_rgb(normals: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    unit = F.normalize(normals.detach().cpu().to(dtype=torch.float32), dim=-1, eps=1.0e-6)
    image = (unit * 0.5 + 0.5).clamp(0.0, 1.0)
    if mask is not None:
        image = torch.where(mask.detach().cpu().unsqueeze(-1), image, torch.zeros_like(image))
    return image.permute(2, 0, 1)


def make_residual_panel(
    *,
    target: torch.Tensor,
    rendered: torch.Tensor,
    alpha: torch.Tensor,
    normal_distance: torch.Tensor | None,
    support: dict[str, torch.Tensor],
) -> torch.Tensor:
    residual = (rendered - target).abs().mean(dim=0)
    residual_vis = normalize_map(residual)
    normal_vis = torch.zeros_like(alpha) if normal_distance is None else normalize_map(normal_distance)
    support_vis = normalize_map(torch.log1p(support["hit_count"]))
    nearest_power_vis = normalize_map(-support["nearest_power"])
    columns = [
        target.detach().cpu().to(dtype=torch.float32).clamp(0.0, 1.0),
        rendered.detach().cpu().to(dtype=torch.float32).clamp(0.0, 1.0),
        gray_rgb(alpha),
        gray_rgb(residual_vis),
        gray_rgb(normal_vis),
        gray_rgb(support_vis),
        gray_rgb(nearest_power_vis),
    ]
    return torch.cat(columns, dim=-1)


def structure_bucket_rows(
    *,
    residual: torch.Tensor,
    alpha: torch.Tensor,
    median_depth: torch.Tensor,
    normal_distance: torch.Tensor,
    rendered_normal_norm: torch.Tensor,
    normal_error: torch.Tensor,
    valid_normal: torch.Tensor,
) -> list[dict[str, Any]]:
    high_residual = residual >= torch.quantile(residual.flatten(), 0.8)
    masks = {
        "valid_normal": valid_normal,
        "valid_normal_high_residual": valid_normal & high_residual,
        "valid_normal_alpha_gte_0_5": valid_normal & (alpha >= 0.5),
        "valid_normal_alpha_gte_0_5_high_residual": valid_normal & (alpha >= 0.5) & high_residual,
        "valid_normal_alpha_gte_0_9": valid_normal & (alpha >= 0.9),
        "valid_normal_alpha_gte_0_9_high_residual": valid_normal & (alpha >= 0.9) & high_residual,
    }
    rows = []
    for name, mask in masks.items():
        rows.append(
            {
                "bucket": name,
                "pixel_fraction": scalar(mask.to(dtype=torch.float32).mean()),
                "residual_l1_mean": masked_mean(residual, mask),
                "alpha_mean": masked_mean(alpha, mask),
                "median_depth_mean": masked_mean(median_depth, mask),
                "normal_distance_mean": masked_mean(normal_distance, mask),
                "rendered_normal_norm_mean": masked_mean(rendered_normal_norm, mask),
                "rendered_vs_depth_normal_error_mean": masked_mean(normal_error, mask),
            }
        )
    return rows


def make_structure_panel(
    *,
    target: torch.Tensor,
    rendered: torch.Tensor,
    alpha: torch.Tensor,
    median_depth: torch.Tensor,
    normal_distance: torch.Tensor,
    rendered_normal: torch.Tensor,
    depth_normal: torch.Tensor,
    valid_normal: torch.Tensor,
    normal_error: torch.Tensor,
) -> torch.Tensor:
    residual = (rendered - target).abs().mean(dim=0)
    columns = [
        target.detach().cpu().to(dtype=torch.float32).clamp(0.0, 1.0),
        rendered.detach().cpu().to(dtype=torch.float32).clamp(0.0, 1.0),
        gray_rgb(alpha),
        gray_rgb(normalize_map(residual)),
        gray_rgb(normalize_map(median_depth)),
        gray_rgb(normalize_map(normal_distance)),
        normal_rgb(rendered_normal),
        normal_rgb(depth_normal, valid_normal),
        gray_rgb(torch.where(valid_normal, normalize_map(normal_error), torch.zeros_like(normal_error))),
    ]
    return torch.cat(columns, dim=-1)


@torch.no_grad()
def heldout_structure_diagnostics(
    *,
    model: MetalPowerFoamVideo,
    cfg: dict[str, Any],
    rendered: torch.Tensor,
    alpha: torch.Tensor,
    rendered_normal: torch.Tensor | None,
    targets: torch.Tensor,
    frame_indices: torch.Tensor,
    rays: torch.Tensor | None,
    view_names: list[str],
    frame_count: int,
    output: Path,
) -> dict[str, Any]:
    if rendered_normal is None:
        return {"skipped": "rendered normals are unavailable"}
    if rays is None:
        return {"skipped": "heldout rays are unavailable"}
    aux = model.height_sv_aux_batch(frame_indices, targets, rays)
    if aux is None:
        return {"skipped": "height+SV aux outputs are unavailable"}

    median_depth = aux.median_depth.detach().cpu().to(dtype=torch.float32)
    normal_distance = aux.normal_distance.detach().cpu().to(dtype=torch.float32)
    rays_cpu = rays.detach().cpu().to(dtype=torch.float32)
    depth_normal, valid_normal = normals_from_ray_depth(median_depth, rays_cpu)
    rendered_normal_cpu = rendered_normal.detach().cpu().to(dtype=torch.float32)
    rendered_normal_norm = torch.linalg.vector_norm(rendered_normal_cpu, dim=-1)
    rendered_unit = F.normalize(rendered_normal_cpu, dim=-1, eps=1.0e-6)
    normal_cos = (rendered_unit * depth_normal).sum(dim=-1).clamp(-1.0, 1.0)
    normal_error = torch.where(valid_normal, 1.0 - normal_cos, torch.zeros_like(normal_cos))

    rendered_cpu = rendered.detach().cpu().to(dtype=torch.float32)
    target_cpu = targets.detach().cpu().to(dtype=torch.float32)
    alpha_cpu = alpha.detach().cpu().to(dtype=torch.float32)
    residual = (rendered_cpu - target_cpu).abs().mean(dim=1)
    sample_l1 = residual.flatten(1).mean(dim=1)
    sample_index = int(torch.argmax(sample_l1).item())
    panel_path = output.with_name(output.stem + "_structure_panel.png")
    panel = make_structure_panel(
        target=target_cpu[sample_index],
        rendered=rendered_cpu[sample_index],
        alpha=alpha_cpu[sample_index],
        median_depth=median_depth[sample_index],
        normal_distance=normal_distance[sample_index],
        rendered_normal=rendered_normal_cpu[sample_index],
        depth_normal=depth_normal[sample_index],
        valid_normal=valid_normal[sample_index],
        normal_error=normal_error[sample_index],
    )
    save_png(panel_path, panel)

    valid_depth = median_depth > 0.0
    valid = valid_normal & torch.isfinite(normal_error)
    return {
        "panel": rel(panel_path),
        "panel_columns": [
            "gt",
            "render",
            "alpha",
            "residual_l1",
            "median_depth",
            "normal_distance",
            "rendered_normal",
            "depth_normal_from_median_depth",
            "rendered_vs_depth_normal_error",
        ],
        "selected_sample": {
            **label_for_sample(
                sample_index,
                frame_indices=frame_indices,
                view_names=view_names,
                frame_count=frame_count,
            ),
            "sample_l1": scalar(sample_l1[sample_index]),
            "alpha_mean": scalar(alpha_cpu[sample_index].mean()),
        },
        "valid_depth_fraction": scalar(valid_depth.to(dtype=torch.float32).mean()),
        "valid_depth_normal_fraction": scalar(valid.to(dtype=torch.float32).mean()),
        "median_depth_quantiles": masked_quantiles(median_depth, valid_depth, (0.1, 0.5, 0.9)),
        "normal_distance_quantiles": masked_quantiles(normal_distance, valid, (0.1, 0.5, 0.9)),
        "rendered_normal_norm_quantiles": masked_quantiles(rendered_normal_norm, valid, (0.1, 0.5, 0.9)),
        "rendered_vs_depth_normal_error_quantiles": masked_quantiles(normal_error, valid, (0.1, 0.5, 0.9)),
        "residual_correlations": {
            "alpha": pearson_corr(alpha_cpu, residual, torch.ones_like(valid, dtype=torch.bool)),
            "median_depth": pearson_corr(median_depth, residual, valid_depth),
            "normal_distance": pearson_corr(normal_distance, residual, valid),
            "rendered_normal_norm": pearson_corr(rendered_normal_norm, residual, valid),
            "rendered_vs_depth_normal_error": pearson_corr(normal_error, residual, valid),
        },
        "buckets": structure_bucket_rows(
            residual=residual,
            alpha=alpha_cpu,
            median_depth=median_depth,
            normal_distance=normal_distance,
            rendered_normal_norm=rendered_normal_norm,
            normal_error=normal_error,
            valid_normal=valid,
        ),
    }


def support_summary(
    *,
    support: dict[str, torch.Tensor],
    residual: torch.Tensor,
    alpha: torch.Tensor,
) -> dict[str, Any]:
    hit_count = support["hit_count"].to(dtype=torch.float32)
    nearest_depth = support["nearest_depth"].to(dtype=torch.float32)
    nearest_power = support["nearest_power"].to(dtype=torch.float32)
    support_hit = hit_count > 0
    high_residual = residual >= torch.quantile(residual.flatten(), 0.8)
    return {
        "support_hit_fraction": scalar(support_hit.to(torch.float32).mean()),
        "support_hit_count_mean": scalar(hit_count.mean()),
        "support_hit_count_quantiles": quantiles(hit_count, (0.5, 0.9, 0.99)),
        "nearest_depth_quantiles": quantiles(nearest_depth[torch.isfinite(nearest_depth)], (0.1, 0.5, 0.9))
        if bool(torch.isfinite(nearest_depth).any())
        else {},
        "nearest_power_quantiles": quantiles(nearest_power, (0.1, 0.5, 0.9)),
        "residual_mean_with_support": masked_mean(residual, support_hit),
        "residual_mean_without_support": masked_mean(residual, ~support_hit),
        "high_residual_support_hit_fraction": masked_mean(support_hit.to(torch.float32), high_residual),
        "alpha_mean_with_support": masked_mean(alpha, support_hit),
        "alpha_mean_without_support": masked_mean(alpha, ~support_hit),
    }


def split_summary(
    *,
    name: str,
    rendered: torch.Tensor,
    alpha: torch.Tensor,
    targets: torch.Tensor,
    frame_indices: torch.Tensor,
    view_names: list[str],
    frame_count: int,
) -> dict[str, Any]:
    rendered = rendered.detach().cpu().to(dtype=torch.float32)
    alpha = alpha.detach().cpu().to(dtype=torch.float32)
    targets = targets.detach().cpu().to(dtype=torch.float32)
    frame_indices_cpu = frame_indices.detach().cpu().to(dtype=torch.long)
    view_indices_cpu = sample_view_indices(
        view_names,
        frame_count=int(frame_count),
        sample_count=int(rendered.shape[0]),
    )
    abs_error = (rendered - targets).abs().mean(dim=1)
    sq_error = (rendered - targets).square().mean(dim=1)
    target_luma = targets.mean(dim=1)
    low_alpha = alpha < 0.05
    mid_alpha = (alpha >= 0.05) & (alpha < 0.5)
    high_alpha = alpha >= 0.5
    sample_l1 = abs_error.flatten(1).mean(dim=1)
    sample_mse = sq_error.flatten(1).mean(dim=1)
    sample_psnr = -10.0 * torch.log10(sample_mse.clamp_min(1.0e-12))
    worst_order = torch.argsort(sample_l1, descending=True)[: min(8, int(sample_l1.numel()))]
    frame_rows = []
    for frame in sorted(set(int(x) for x in frame_indices_cpu.tolist())):
        mask = frame_indices_cpu == frame
        frame_rows.append(
            grouped_metric_row(
                {"frame": frame},
                mask,
                sample_l1=sample_l1,
                sample_mse=sample_mse,
                alpha=alpha,
                target_luma=target_luma,
            )
        )
    view_rows = []
    if view_indices_cpu is not None:
        for view_index, view_name in enumerate(view_names):
            mask = view_indices_cpu == view_index
            view_rows.append(
                grouped_metric_row(
                    {"view_index": view_index, "view_name": view_name},
                    mask,
                    sample_l1=sample_l1,
                    sample_mse=sample_mse,
                    alpha=alpha,
                    target_luma=target_luma,
                )
            )
    worst_samples = []
    for index in worst_order:
        row = {
            "sample": int(index.item()),
            "frame": int(frame_indices_cpu[index].item()),
            "l1": scalar(sample_l1[index]),
            "mse": scalar(sample_mse[index]),
            "psnr": scalar(sample_psnr[index]),
            "alpha_mean": scalar(alpha[index].mean()),
        }
        if view_indices_cpu is not None:
            view_index = int(view_indices_cpu[index].item())
            row["view_index"] = view_index
            row["view_name"] = view_names[view_index]
        worst_samples.append(row)
    return {
        "name": name,
        "sample_count": int(rendered.shape[0]),
        "frame_count": len(frame_rows),
        "l1": scalar(sample_l1.mean()),
        "mse": scalar(sample_mse.mean()),
        "psnr": scalar(-10.0 * torch.log10(sample_mse.mean().clamp_min(1.0e-12))),
        "alpha_mean": scalar(alpha.mean()),
        "alpha_quantiles": quantiles(alpha, (0.05, 0.25, 0.5, 0.75, 0.95)),
        "alpha_fraction_lt_0_05": scalar((alpha < 0.05).to(torch.float32).mean()),
        "alpha_fraction_gt_0_50": scalar((alpha > 0.5).to(torch.float32).mean()),
        "alpha_fraction_gt_0_90": scalar((alpha > 0.9).to(torch.float32).mean()),
        "l1_low_alpha": masked_mean(abs_error, low_alpha),
        "l1_mid_alpha": masked_mean(abs_error, mid_alpha),
        "l1_high_alpha": masked_mean(abs_error, high_alpha),
        "target_luma_low_alpha": masked_mean(target_luma, low_alpha),
        "target_luma_mid_alpha": masked_mean(target_luma, mid_alpha),
        "target_luma_high_alpha": masked_mean(target_luma, high_alpha),
        "alpha_weighted_l1": scalar((abs_error * alpha).sum() / alpha.sum().clamp_min(1.0e-8)),
        "inverse_alpha_weighted_l1": scalar((abs_error * (1.0 - alpha)).sum() / (1.0 - alpha).sum().clamp_min(1.0e-8)),
        "worst_samples": worst_samples,
        "per_frame": frame_rows,
        "per_view": view_rows,
    }


def heldout_witness_diagnostics(
    *,
    model: MetalPowerFoamVideo,
    cfg: dict[str, Any],
    rendered: torch.Tensor,
    alpha: torch.Tensor,
    normal_distance: torch.Tensor | None,
    rendered_normal: torch.Tensor | None,
    targets: torch.Tensor,
    frame_indices: torch.Tensor,
    rays: torch.Tensor | None,
    view_names: list[str],
    frame_count: int,
    output: Path,
    residual_quantile: float,
    support_chunk_size: int,
) -> dict[str, Any]:
    if rays is None:
        return {"skipped": "heldout rays are unavailable"}
    pixel_l1 = (rendered.detach().cpu().to(dtype=torch.float32) - targets.detach().cpu().to(dtype=torch.float32)).abs().mean(dim=1)
    sample_l1 = pixel_l1.flatten(1).mean(dim=1)
    sample_index = int(torch.argmax(sample_l1).item())
    points, radii = decoded_points_radii(model)
    frame = int(frame_indices.detach().cpu().to(dtype=torch.long)[sample_index].item())
    support = ray_support_maps(
        points[frame],
        radii[frame],
        rays[sample_index],
        cfg,
        chunk_size=int(support_chunk_size),
    )
    panel_path = output.with_name(output.stem + "_panel.png")
    sample_normal_distance = None if normal_distance is None else normal_distance[sample_index]
    panel = make_residual_panel(
        target=targets[sample_index].detach().cpu().to(dtype=torch.float32),
        rendered=rendered[sample_index],
        alpha=alpha[sample_index],
        normal_distance=sample_normal_distance,
        support=support,
    )
    save_png(panel_path, panel)
    residual = pixel_l1[sample_index]
    return {
        "panel": rel(panel_path),
        "panel_columns": [
            "gt",
            "render",
            "alpha",
            "residual_l1",
            "normal_distance",
            "log_support_hit_count",
            "nearest_power_support",
        ],
        "selected_sample": {
            **label_for_sample(
                sample_index,
                frame_indices=frame_indices,
                view_names=view_names,
                frame_count=frame_count,
            ),
            "sample_l1": scalar(sample_l1[sample_index]),
            "alpha_mean": scalar(alpha[sample_index].mean()),
        },
        "pixel_buckets": pixel_bucket_summary(
            rendered=rendered,
            alpha=alpha,
            targets=targets,
            normal_distance=normal_distance,
            residual_quantile=float(residual_quantile),
        ),
        "selected_sample_support": support_summary(
            support=support,
            residual=residual,
            alpha=alpha[sample_index].detach().cpu().to(dtype=torch.float32),
        ),
        "structure": heldout_structure_diagnostics(
            model=model,
            cfg=cfg,
            rendered=rendered,
            alpha=alpha,
            rendered_normal=rendered_normal,
            targets=targets,
            frame_indices=frame_indices,
            rays=rays,
            view_names=view_names,
            frame_count=frame_count,
            output=output,
        ),
    }


def load_model_for_checkpoint(config_path: Path, checkpoint_path: Path, device: torch.device) -> tuple[dict[str, Any], dict[str, Any], MetalPowerFoamVideo]:
    cfg = resolve_config(load_config_file(config_path))
    training_data = load_powerfoam_training_data(cfg, device)
    model = build_model(cfg, training_data, device)
    checkpoint = load_checkpoint_mapping(checkpoint_path, map_location=device)
    model.load_state_dict(model_state_dict_from_checkpoint(checkpoint), strict=True)
    model.eval()
    return cfg, training_data, model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--residual-quantile", type=float, default=0.8)
    parser.add_argument("--support-chunk-size", type=int, default=2048)
    parser.add_argument("--skip-witness-panel", action="store_true")
    parser.add_argument("--heldout-only", action="store_true")
    args = parser.parse_args()
    cfg_preview = resolve_config(load_config_file(args.config))
    checkpoint = args.checkpoint or (cfg_preview["logging"]["output_dir"] / "checkpoint_best.pt")
    output = args.output or (cfg_preview["logging"]["output_dir"] / "heldout_error_diagnostics.json")
    device = resolve_torch_device(str(args.device), auto_cuda=False)
    cfg, training_data, model = load_model_for_checkpoint(args.config, checkpoint, device)
    checkpoint_payload = load_checkpoint_mapping(checkpoint, map_location="cpu")
    train_summary = None
    if not bool(args.heldout_only):
        train_render, train_alpha = render_split(
            model,
            cfg,
            training_data["sample_frame_indices"],
            training_data["sample_rays"],
            batch_size=int(args.batch_size),
        )
        train_summary = split_summary(
            name="train",
            rendered=train_render,
            alpha=train_alpha,
            targets=training_data["targets"],
            frame_indices=training_data["sample_frame_indices"],
            view_names=training_data["train_views"],
            frame_count=int(training_data["frame_count"]),
        )
    heldout_summary = None
    heldout_witness = None
    if training_data["heldout_targets"] is not None:
        heldout_render, heldout_alpha, heldout_normal_distance, heldout_rendered_normal = render_split_with_normal_distance(
            model,
            cfg,
            training_data["heldout_frame_indices"],
            training_data["heldout_rays"],
            batch_size=int(args.batch_size),
        )
        heldout_summary = split_summary(
            name="heldout",
            rendered=heldout_render,
            alpha=heldout_alpha,
            targets=training_data["heldout_targets"],
            frame_indices=training_data["heldout_frame_indices"],
            view_names=training_data["heldout_views"],
            frame_count=int(training_data["frame_count"]),
        )
        if not bool(args.skip_witness_panel):
            heldout_witness = heldout_witness_diagnostics(
                model=model,
                cfg=cfg,
                rendered=heldout_render,
                alpha=heldout_alpha,
                normal_distance=heldout_normal_distance,
                rendered_normal=heldout_rendered_normal,
                targets=training_data["heldout_targets"],
                frame_indices=training_data["heldout_frame_indices"],
                rays=training_data["heldout_rays"],
                view_names=training_data["heldout_views"],
                frame_count=int(training_data["frame_count"]),
                output=output,
                residual_quantile=float(args.residual_quantile),
                support_chunk_size=int(args.support_chunk_size),
            )
    report = {
        "config": rel(args.config),
        "checkpoint": rel(checkpoint),
        "checkpoint_step": int(checkpoint_payload.get("step", -1)),
        "output_dir": rel(cfg["logging"]["output_dir"]),
        "train_views": training_data["train_views"],
        "heldout_views": training_data["heldout_views"],
        "pose_source": training_data["pose_source"],
        "render_size": int(cfg["render"]["render_size"]),
        "cells": int(cfg["model"]["cells"]),
        "adjacency_mode": str(cfg["model"]["adjacency_mode"]),
        "feature_mode": str(cfg["model"]["feature_mode"]),
        "train": train_summary,
        "heldout": heldout_summary,
        "heldout_witness": heldout_witness,
    }
    if heldout_summary is not None and train_summary is not None:
        report["heldout_to_train"] = {
            "l1_ratio": heldout_summary["l1"] / max(train_summary["l1"], 1.0e-8),
            "alpha_mean_ratio": heldout_summary["alpha_mean"] / max(train_summary["alpha_mean"], 1.0e-8),
            "low_alpha_target_luma_ratio": (
                None
                if train_summary["target_luma_low_alpha"] in {None, 0.0}
                else heldout_summary["target_luma_low_alpha"] / max(train_summary["target_luma_low_alpha"], 1.0e-8)
            ),
        }
    write_report_json(output, report)
    print(json.dumps({"output": rel(output), "heldout": heldout_summary, "train": train_summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
