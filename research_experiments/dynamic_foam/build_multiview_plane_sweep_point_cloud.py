from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
TRAIN_SRC = ROOT / "src" / "train"
if str(TRAIN_SRC) not in sys.path:
    sys.path.insert(0, str(TRAIN_SRC))

from camera import CameraSpec, build_camera_rays
from config_utils import load_config_file
from multicam_video_data import cameras_from_K_w2c, load_multicam_video_bundle
from renderers.projection import project_points_camera
from train_powerfoam_metal import resolve_config


def project_points(camera: CameraSpec, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    world_to_camera = torch.linalg.inv(camera.camera_to_world.to(dtype=points.dtype, device=points.device))
    ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
    points_camera = (torch.cat([points, ones], dim=-1) @ world_to_camera.T)[:, :3]
    pixels, depths, _pixel_jacobian, front = project_points_camera(points_camera, camera, near_plane=1.0e-5)
    return pixels[:, 0], pixels[:, 1], depths, front


def sample_image(image: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    _, height, width = image.shape
    x = 2.0 * u / max(float(width - 1), 1.0) - 1.0
    y = 2.0 * v / max(float(height - 1), 1.0) - 1.0
    grid = torch.stack([x, y], dim=-1).view(1, 1, -1, 2)
    sampled = F.grid_sample(
        image.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled[0, :, 0, :].T.contiguous()


def local_mean_image(image: torch.Tensor, radius: int) -> torch.Tensor:
    radius = int(radius)
    if radius <= 0:
        return image
    padded = F.pad(image.unsqueeze(0), (radius, radius, radius, radius), mode="replicate")
    return F.avg_pool2d(padded, kernel_size=2 * radius + 1, stride=1)[0]


def patch_offsets(radius: int) -> torch.Tensor:
    radius = int(radius)
    if radius < 0:
        raise ValueError(f"patch radius must be non-negative, got {radius}.")
    offsets = [
        (float(dx), float(dy))
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
    ]
    return torch.tensor(offsets, dtype=torch.float32)


def sample_image_patch(image: torch.Tensor, u: torch.Tensor, v: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    patch_u = u[:, None] + offsets[None, :, 0].to(device=u.device, dtype=u.dtype)
    patch_v = v[:, None] + offsets[None, :, 1].to(device=v.device, dtype=v.dtype)
    sampled = sample_image(image, patch_u.reshape(-1), patch_v.reshape(-1))
    return sampled.reshape(int(u.numel()), int(offsets.shape[0]), int(image.shape[0]))


def patch_inside_mask(
    u: torch.Tensor,
    v: torch.Tensor,
    *,
    width: int,
    height: int,
    offsets: torch.Tensor,
) -> torch.Tensor:
    min_dx = float(offsets[:, 0].min().item())
    max_dx = float(offsets[:, 0].max().item())
    min_dy = float(offsets[:, 1].min().item())
    max_dy = float(offsets[:, 1].max().item())
    return (
        (u + min_dx >= 0.0)
        & (u + max_dx <= float(width - 1))
        & (v + min_dy >= 0.0)
        & (v + max_dy <= float(height - 1))
    )


def patch_errors(
    source_patch: torch.Tensor,
    target_patch: torch.Tensor,
    *,
    score_mode: str,
    min_patch_std: float,
) -> torch.Tensor:
    if score_mode == "center_l1":
        source_center = source_patch[:, source_patch.shape[1] // 2, :]
        target_center = target_patch[:, :, target_patch.shape[2] // 2, :]
        return (target_center - source_center[:, None, :]).abs().mean(dim=-1)
    if score_mode == "patch_l1":
        return (target_patch - source_patch[:, None, :, :]).abs().mean(dim=(-1, -2))
    if score_mode == "zncc":
        source_vec = source_patch.flatten(start_dim=1)
        target_vec = target_patch.flatten(start_dim=2)
        source_centered = source_vec - source_vec.mean(dim=1, keepdim=True)
        target_centered = target_vec - target_vec.mean(dim=2, keepdim=True)
        source_norm = source_centered.square().sum(dim=1).sqrt()
        target_norm = target_centered.square().sum(dim=2).sqrt()
        denom = source_norm[:, None] * target_norm
        corr = (target_centered * source_centered[:, None, :]).sum(dim=2) / denom.clamp_min(1.0e-6)
        error = (1.0 - corr.clamp(-1.0, 1.0)) * 0.5
        min_norm = float(min_patch_std) * float(source_vec.shape[1]) ** 0.5
        textured = (source_norm[:, None] >= min_norm) & (target_norm >= min_norm)
        return torch.where(textured, error, error.new_full(error.shape, 1.0e6))
    raise ValueError(f"Unsupported score mode {score_mode!r}.")


def pixel_sample_indices(height: int, width: int, stride: int) -> tuple[torch.Tensor, torch.Tensor]:
    ys = torch.arange(0, height, int(stride), dtype=torch.long)
    xs = torch.arange(0, width, int(stride), dtype=torch.long)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return yy.reshape(-1), xx.reshape(-1)


def best_depth_points_for_multiview_source(
    *,
    source_image: torch.Tensor,
    target_images: list[torch.Tensor],
    source_camera: CameraSpec,
    target_cameras: list[CameraSpec],
    depths: torch.Tensor,
    stride: int,
    chunk_size: int,
    min_support: int,
    max_error: float,
    support_error: float | None,
    score_mode: str,
    patch_radius: int,
    min_patch_std: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _, height, width = source_image.shape
    origins, directions = build_camera_rays(source_camera, height, width, device=source_image.device)
    ys, xs = pixel_sample_indices(height, width, stride)
    use_patch_score = score_mode in {"patch_l1", "zncc"}
    offsets = patch_offsets(patch_radius).to(device=source_image.device)
    if use_patch_score:
        source_patch_all = sample_image_patch(source_image, xs.to(torch.float32), ys.to(torch.float32), offsets)
        source_patch_valid_all = patch_inside_mask(
            xs.to(torch.float32),
            ys.to(torch.float32),
            width=width,
            height=height,
            offsets=offsets,
        )
        source_colors_all = source_patch_all[:, source_patch_all.shape[1] // 2, :].contiguous()
        score_target_images = target_images
    else:
        source_patch_all = None
        source_patch_valid_all = None
        source_score_image = local_mean_image(source_image, patch_radius) if score_mode == "mean_l1" else source_image
        source_colors_all = source_score_image[:, ys, xs].T.contiguous()
        score_target_images = [
            local_mean_image(target_image, patch_radius) if score_mode == "mean_l1" else target_image
            for target_image in target_images
        ]
    source_origins_all = origins[ys, xs].contiguous()
    source_dirs_all = directions[ys, xs].contiguous()

    best_points: list[torch.Tensor] = []
    best_colors: list[torch.Tensor] = []
    best_errors: list[torch.Tensor] = []
    best_support: list[torch.Tensor] = []
    min_support = max(1, min(int(min_support), len(target_images)))
    for start in range(0, int(xs.numel()), int(chunk_size)):
        end = min(start + int(chunk_size), int(xs.numel()))
        sample_count = end - start
        origins_chunk = source_origins_all[start:end]
        dirs_chunk = source_dirs_all[start:end]
        colors_chunk = source_colors_all[start:end]
        source_patch_chunk = None if source_patch_all is None else source_patch_all[start:end]
        source_patch_valid = None if source_patch_valid_all is None else source_patch_valid_all[start:end]
        candidates = origins_chunk[:, None, :] + dirs_chunk[:, None, :] * depths[None, :, None]
        flat = candidates.reshape(-1, 3)
        support = torch.zeros((sample_count, int(depths.numel())), dtype=torch.int16)
        error_sum = torch.zeros((sample_count, int(depths.numel())), dtype=torch.float32)
        for target_image, target_camera in zip(score_target_images, target_cameras):
            u, v, _z, front = project_points(target_camera, flat)
            if use_patch_score:
                assert source_patch_chunk is not None
                assert source_patch_valid is not None
                valid = front & patch_inside_mask(u, v, width=width, height=height, offsets=offsets)
                source_valid_grid = source_patch_valid[:, None].expand(sample_count, int(depths.numel()))
                valid = valid & source_valid_grid.reshape(-1)
                sampled = sample_image_patch(target_image, u, v, offsets).reshape(
                    sample_count,
                    int(depths.numel()),
                    int(offsets.shape[0]),
                    3,
                )
                errors = patch_errors(
                    source_patch_chunk,
                    sampled,
                    score_mode=score_mode,
                    min_patch_std=float(min_patch_std),
                )
            else:
                valid = front & (u >= 0.0) & (u <= float(width - 1)) & (v >= 0.0) & (v <= float(height - 1))
                sampled = sample_image(target_image, u, v).reshape(sample_count, int(depths.numel()), 3)
                errors = (sampled - colors_chunk[:, None, :]).abs().mean(dim=-1)
            valid_grid = valid.reshape(sample_count, int(depths.numel()))
            support_grid = valid_grid
            if support_error is not None:
                support_grid = support_grid & (errors <= float(support_error))
            support += support_grid.to(dtype=support.dtype)
            error_sum += torch.where(support_grid, errors, torch.zeros_like(errors))
        support_float = support.to(dtype=torch.float32)
        mean_error = error_sum / support_float.clamp_min(1.0)
        usable = support >= min_support
        mean_error = torch.where(usable, mean_error, mean_error.new_full(mean_error.shape, 1.0e6))
        best_error, best_index = mean_error.min(dim=1)
        point = candidates[torch.arange(sample_count), best_index]
        point_support = support[torch.arange(sample_count), best_index]
        finite = torch.isfinite(best_error) & (best_error < float(max_error))
        best_points.append(point[finite].detach().cpu())
        best_colors.append(colors_chunk[finite].detach().cpu())
        best_errors.append(best_error[finite].detach().cpu())
        best_support.append(point_support[finite].detach().cpu())
    return (
        torch.cat(best_points, dim=0),
        torch.cat(best_colors, dim=0),
        torch.cat(best_errors, dim=0),
        torch.cat(best_support, dim=0),
    )


def write_ascii_ply(path: Path, points: torch.Tensor, colors: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    colors_u8 = (colors.clamp(0.0, 1.0) * 255.0).round().to(torch.int64)
    with path.open("w", encoding="ascii") as fh:
        fh.write("ply\n")
        fh.write("format ascii 1.0\n")
        fh.write(f"element vertex {int(points.shape[0])}\n")
        fh.write("property float x\n")
        fh.write("property float y\n")
        fh.write("property float z\n")
        fh.write("property uchar red\n")
        fh.write("property uchar green\n")
        fh.write("property uchar blue\n")
        fh.write("end_header\n")
        for point, color in zip(points.tolist(), colors_u8.tolist()):
            fh.write(
                f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def build_plane_sweep_cloud(args: argparse.Namespace) -> dict[str, Any]:
    cfg = resolve_config(load_config_file(args.config))
    if str(cfg["data"]["frame_source"]) != "multicam_val":
        raise ValueError("Plane-sweep point cloud builder currently expects data.frame_source='multicam_val'.")
    device = torch.device("cpu")
    render_size = int(args.target_size if args.target_size is not None else cfg["render"]["render_size"])
    bundle = load_multicam_video_bundle(
        data_cfg=cfg["data"],
        camera_cfg=cfg["camera"],
        target_size=render_size,
        device=device,
    )
    if bundle.train_view_count < 2:
        raise ValueError("Plane-sweep point cloud builder requires at least two train cameras.")
    frame_index = int(args.frame_index)
    if frame_index < 0 or frame_index >= bundle.frame_count:
        raise IndexError(f"frame-index {frame_index} out of range for {bundle.frame_count} frames.")

    cameras = cameras_from_K_w2c(
        bundle.train_K,
        bundle.train_w2c,
        lens_models=bundle.train_lens_models,
        distortions=bundle.train_distortions,
    )
    depths = torch.linspace(float(args.depth_min), float(args.depth_max), int(args.depths), dtype=torch.float32)
    all_points = []
    all_colors = []
    all_errors = []
    all_support = []
    source_views = [int(view) for view in args.source_views] if args.source_views else list(range(bundle.train_view_count))
    for source_view in source_views:
        if source_view < 0 or source_view >= bundle.train_view_count:
            raise IndexError(f"source view {source_view} out of range for {bundle.train_view_count} train views.")
        target_views = [view for view in range(bundle.train_view_count) if view != source_view]
        points, colors, errors, support = best_depth_points_for_multiview_source(
            source_image=bundle.train_frames[source_view, frame_index].to(dtype=torch.float32),
            target_images=[bundle.train_frames[target_view, frame_index].to(dtype=torch.float32) for target_view in target_views],
            source_camera=cameras[source_view][frame_index],
            target_cameras=[cameras[target_view][frame_index] for target_view in target_views],
            depths=depths,
            stride=int(args.stride),
            chunk_size=int(args.chunk_size),
            min_support=int(args.min_support),
            max_error=float(args.max_error),
            support_error=None if args.support_error is None else float(args.support_error),
            score_mode=str(args.score_mode),
            patch_radius=int(args.patch_radius),
            min_patch_std=float(args.min_patch_std),
        )
        all_points.append(points)
        all_colors.append(colors)
        all_errors.append(errors)
        all_support.append(support)
    points = torch.cat(all_points, dim=0)
    colors = torch.cat(all_colors, dim=0)
    errors = torch.cat(all_errors, dim=0)
    support = torch.cat(all_support, dim=0)
    order = torch.argsort(errors, stable=True)
    if int(args.max_points) > 0:
        order = order[: int(args.max_points)]
    points = points.index_select(0, order).contiguous()
    colors = colors.index_select(0, order).contiguous()
    errors = errors.index_select(0, order).contiguous()
    support = support.index_select(0, order).contiguous()
    write_ascii_ply(Path(args.output), points, colors)
    summary = {
        "config": str(args.config),
        "output": str(args.output),
        "sample_id": str(bundle.metadata.get("sample_id")) if bundle.metadata else None,
        "train_cameras": list(bundle.train_camera_names),
        "frame_index": frame_index,
        "render_size": render_size,
        "depth_min": float(args.depth_min),
        "depth_max": float(args.depth_max),
        "depths": int(args.depths),
        "stride": int(args.stride),
        "source_views": source_views,
        "min_support": int(args.min_support),
        "max_error": float(args.max_error),
        "support_error": None if args.support_error is None else float(args.support_error),
        "score_mode": str(args.score_mode),
        "patch_radius": int(args.patch_radius),
        "min_patch_std": float(args.min_patch_std),
        "point_count": int(points.shape[0]),
        "mean_error": float(errors.mean().item()) if errors.numel() else None,
        "median_error": float(errors.median().item()) if errors.numel() else None,
        "p90_error": float(torch.quantile(errors, 0.9).item()) if errors.numel() else None,
        "support_mean": float(support.to(dtype=torch.float32).mean().item()) if support.numel() else None,
        "support_median": float(support.to(dtype=torch.float32).median().item()) if support.numel() else None,
        "support_p90": float(torch.quantile(support.to(dtype=torch.float32), 0.9).item()) if support.numel() else None,
        "lens_models": bundle.train_lens_models,
    }
    summary_path = Path(args.output).with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a train-camera-only DeepView plane-sweep PLY init.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-size", type=int, default=None)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--depth-min", type=float, default=1.0)
    parser.add_argument("--depth-max", type=float, default=3.25)
    parser.add_argument("--depths", type=int, default=96)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-points", type=int, default=8192)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--max-error", type=float, default=1.0)
    parser.add_argument("--support-error", type=float, default=None)
    parser.add_argument("--score-mode", choices=("center_l1", "mean_l1", "patch_l1", "zncc"), default="center_l1")
    parser.add_argument("--patch-radius", type=int, default=0)
    parser.add_argument("--min-patch-std", type=float, default=0.0)
    parser.add_argument("--source-views", nargs="+", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=1024)
    summary = build_plane_sweep_cloud(parser.parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
