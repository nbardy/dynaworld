from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

try:
    from .report_artifacts import PROJECT_ROOT, ensure_train_path, parse_frame_indices
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import PROJECT_ROOT, ensure_train_path, parse_frame_indices

ROOT = PROJECT_ROOT
ensure_train_path()

from config_utils import load_config_file  # noqa: E402
from checkpoint_utils import load_checkpoint_mapping, model_state_dict_from_checkpoint  # noqa: E402
from powerfoam_adjacency import build_csr_adjacency  # noqa: E402
from powerfoam_metal_config import resolve_config  # noqa: E402
from powerfoam_raster_config import make_powerfoam_metal_raster_config as make_raster_config  # noqa: E402
from sequence_data import load_video_sequence  # noqa: E402
from powerfoam_metal_trainer import MetalPowerFoamVideo  # noqa: E402


def _as_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def _stats(values: torch.Tensor) -> dict[str, float]:
    v = values.detach().flatten().to(dtype=torch.float32, device="cpu")
    return {
        "mean": float(v.mean()),
        "p50": float(v.quantile(0.50)),
        "p90": float(v.quantile(0.90)),
        "p95": float(v.quantile(0.95)),
        "p99": float(v.quantile(0.99)),
        "max": float(v.max()),
    }


def _instantiate_model(cfg: dict[str, Any], checkpoint_path: Path, device: torch.device) -> tuple[MetalPowerFoamVideo, torch.Tensor]:
    sequence = load_video_sequence(
        cfg["data"]["video_path"],
        target_size=int(cfg["render"]["render_size"]),
        max_frames=int(cfg["data"]["max_frames"]),
        frame_source=str(cfg["data"]["frame_source"]),
    )
    targets = sequence.frames.to(device=device, dtype=torch.float32)
    model = MetalPowerFoamVideo(
        frame_count=targets.size(0),
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
        color_init_mode=str(cfg["model"]["color_init_mode"]),
        seed=int(cfg["train"]["seed"]),
        init_frames=targets.detach().cpu() if bool(cfg["model"]["init_from_video"]) else None,
        image_init_depth=None if cfg["model"]["image_init_depth"] is None else float(cfg["model"]["image_init_depth"]),
        image_init_jitter=float(cfg["model"]["image_init_jitter"]),
        raster_config=make_raster_config(cfg["render"]),
    ).to(device)
    checkpoint = load_checkpoint_mapping(checkpoint_path, map_location="cpu")
    model.load_state_dict(model_state_dict_from_checkpoint(checkpoint))
    model.eval()
    return model, targets


@torch.no_grad()
def _diagnose_frame(model: MetalPowerFoamVideo, cfg: dict[str, Any], frame_index: int) -> dict[str, Any]:
    device = next(model.parameters()).device
    points, radii, densities, _features, normals = model.decoded_parameters()
    if normals is None:
        raise RuntimeError("section diagnostics currently expect an oriented surface mode")

    point = points[frame_index]
    radius = radii[frame_index]
    density = densities[frame_index]
    normal = normals[frame_index]
    adjacency, offsets = build_csr_adjacency(
        point,
        radius,
        neighbor_count=int(cfg["model"]["neighbor_count"]),
        mode=str(cfg["model"]["adjacency_mode"]),
    )

    rays = model.rays.to(device=device, dtype=point.dtype)[0].reshape(-1, 6)
    origin = rays[:, :3]
    direction = rays[:, 3:]
    eps = float(cfg["render"]["eps"])
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(eps)
    pixel_count = direction.shape[0]

    camera_origin = origin[0]
    power = (point - camera_origin).square().sum(dim=-1) - radius.square()
    sorted_ids = torch.argsort(power.detach(), stable=True)

    transmittance = torch.ones(pixel_count, device=device, dtype=point.dtype)
    section_count = torch.zeros(pixel_count, device=device, dtype=torch.int32)
    significant_1e3 = torch.zeros(pixel_count, device=device, dtype=torch.int32)
    significant_1e4 = torch.zeros(pixel_count, device=device, dtype=torch.int32)
    alpha_sum = torch.zeros(pixel_count, device=device, dtype=point.dtype)
    max_segment_alpha = torch.zeros(pixel_count, device=device, dtype=point.dtype)
    total_length = torch.zeros(pixel_count, device=device, dtype=point.dtype)
    stopped_order = torch.full((pixel_count,), int(cfg["model"]["cells"]), device=device, dtype=torch.int32)
    stopped = torch.zeros(pixel_count, device=device, dtype=torch.bool)

    near_plane = float(cfg["render"]["near_plane"])
    alpha_threshold = float(cfg["render"]["alpha_threshold"])
    transmittance_threshold = float(cfg["render"]["transmittance_threshold"])
    max_alpha = float(cfg["render"]["max_alpha"])
    inf = torch.tensor(float("inf"), device=device, dtype=point.dtype)

    for order, cell_tensor in enumerate(sorted_ids):
        live = transmittance >= transmittance_threshold
        if not bool(live.any().detach().cpu()):
            break
        cell = int(cell_tensor.detach().cpu())
        center = point[cell]
        cell_radius = radius[cell]
        sigma = density[cell].clamp_min(0.0)
        if _as_float(sigma) <= 0.0:
            continue

        oc = origin - center
        qa = (direction * direction).sum(dim=-1)
        qb = 2.0 * (oc * direction).sum(dim=-1)
        qc = (oc * oc).sum(dim=-1) - cell_radius * cell_radius
        disc = qb * qb - 4.0 * qa * qc
        hit = (disc >= 0.0) & (qa > eps) & live
        root = disc.clamp_min(0.0).sqrt()
        t_near = (-qb - root) * 0.5 / qa.clamp_min(eps)
        t_far = (-qb + root) * 0.5 / qa.clamp_min(eps)
        hit &= t_far > near_plane
        t_near = torch.maximum(t_near, torch.full_like(t_near, near_plane))

        start = int(offsets[cell].detach().cpu())
        end = int(offsets[cell + 1].detach().cpu())
        if end > start:
            neighbors = adjacency[start:end].to(dtype=torch.long)
            neighbors = neighbors[(neighbors >= 0) & (neighbors < point.shape[0]) & (neighbors != cell)]
            if neighbors.numel() > 0:
                neighbor_points = point[neighbors]
                neighbor_radii = radius[neighbors]
                face_n = neighbor_points - center
                h = 0.5 * (
                    neighbor_points.square().sum(dim=-1)
                    - center.square().sum()
                    + cell_radius * cell_radius
                    - neighbor_radii * neighbor_radii
                )
                dp = direction @ face_n.T
                num = h[None, :] - origin @ face_n.T
                parallel = dp.abs() <= eps
                hit &= torch.where(parallel, num >= -eps, torch.ones_like(num, dtype=torch.bool)).all(dim=-1)
                t_face = num / torch.where(parallel, torch.ones_like(dp), dp)
                far_candidates = torch.where(dp > eps, t_face, inf).min(dim=-1).values
                near_candidates = torch.where(dp < -eps, t_face, -inf).max(dim=-1).values
                t_far = torch.minimum(t_far, far_candidates)
                t_near = torch.maximum(t_near, near_candidates)

        n = normal[cell]
        dp_surface = direction @ n
        surface_hit = dp_surface.abs() > eps
        safe_dp_surface = torch.where(surface_hit, dp_surface, torch.ones_like(dp_surface))
        t_surface = ((center - origin) @ n) / safe_dp_surface
        t_far = torch.where(dp_surface >= 0.0, torch.minimum(t_far, t_surface), t_far)
        t_near = torch.where(dp_surface < 0.0, torch.maximum(t_near, t_surface), t_near)
        hit &= surface_hit & (t_far > t_near)

        dt = t_far - t_near
        segment_alpha = (1.0 - torch.exp(-sigma * dt)).clamp(0.0, max_alpha)
        active = hit & (dt > 0.0) & (segment_alpha >= alpha_threshold) & live
        weight = transmittance * segment_alpha

        section_count += active.to(torch.int32)
        significant_1e3 += (active & (weight > 1.0e-3)).to(torch.int32)
        significant_1e4 += (active & (weight > 1.0e-4)).to(torch.int32)
        alpha_sum += torch.where(active, segment_alpha, torch.zeros_like(segment_alpha))
        max_segment_alpha = torch.maximum(max_segment_alpha, torch.where(active, segment_alpha, torch.zeros_like(segment_alpha)))
        total_length += torch.where(active, dt, torch.zeros_like(dt))
        transmittance = torch.where(active, transmittance * (1.0 - segment_alpha), transmittance)

        newly_stopped = (~stopped) & (transmittance < transmittance_threshold)
        stopped_order = torch.where(newly_stopped, torch.full_like(stopped_order, order + 1), stopped_order)
        stopped |= newly_stopped

    final_alpha = 1.0 - transmittance
    return {
        "frame": int(frame_index),
        "sections_per_pixel": _stats(section_count),
        "significant_weight_gt_1e-3_per_pixel": _stats(significant_1e3),
        "significant_weight_gt_1e-4_per_pixel": _stats(significant_1e4),
        "final_alpha": _stats(final_alpha),
        "segment_alpha_sum": _stats(alpha_sum),
        "max_segment_alpha": _stats(max_segment_alpha),
        "total_segment_length": _stats(total_length),
        "visited_orders_before_stop": _stats(stopped_order),
        "empty_pixel_fraction": _as_float((section_count == 0).float().mean()),
        "under_alpha_0.10_fraction": _as_float((final_alpha < 0.10).float().mean()),
        "under_alpha_0.50_fraction": _as_float((final_alpha < 0.50).float().mean()),
        "over_alpha_0.95_fraction": _as_float((final_alpha > 0.95).float().mean()),
        "over_alpha_0.99_fraction": _as_float((final_alpha > 0.99).float().mean()),
        "early_stop_fraction": _as_float((stopped_order < int(cfg["model"]["cells"])).float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frames", default="0,5,10,15")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = resolve_config(load_config_file(args.config))
    model, _targets = _instantiate_model(cfg, Path(args.checkpoint), device)
    frame_indices = parse_frame_indices(args.frames)
    per_frame = [_diagnose_frame(model, cfg, frame_index) for frame_index in frame_indices]
    print(json.dumps({"frames": per_frame}, indent=2))


if __name__ == "__main__":
    main()
