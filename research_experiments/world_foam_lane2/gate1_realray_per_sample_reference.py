#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch


DYNAWORLD = Path(__file__).resolve().parents[2]
TRAIN_SRC = DYNAWORLD / "src" / "train"
DEFAULT_CONFIG = (
    DYNAWORLD
    / "src"
    / "train_configs"
    / "local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_32_smoke.jsonc"
)

if str(TRAIN_SRC) not in sys.path:
    sys.path.insert(0, str(TRAIN_SRC))

from config_utils import load_config_file  # noqa: E402
from powerfoam_training_data import load_powerfoam_training_data  # noqa: E402


EPS = 1.0e-8
SITE_INITIALIZATION_LEGACY_SPARSE = "legacy_sparse"
SITE_INITIALIZATION_LEGACY_PIXEL_MEAN = "legacy_pixel_mean"
SITE_INITIALIZATION_LEGACY_FRAME_PIXEL_MEAN = "legacy_frame_pixel_mean"
SITE_INITIALIZATION_LEGACY_FRAME_PATCH3_MEAN = "legacy_frame_patch3_mean"
SITE_INITIALIZATION_STRATIFIED_GRID = "stratified_grid"
SITE_INITIALIZATION_STRATIFIED_PIXEL_MEAN = "stratified_pixel_mean"
SITE_INITIALIZATION_CHOICES = (
    SITE_INITIALIZATION_LEGACY_SPARSE,
    SITE_INITIALIZATION_LEGACY_PIXEL_MEAN,
    SITE_INITIALIZATION_LEGACY_FRAME_PIXEL_MEAN,
    SITE_INITIALIZATION_LEGACY_FRAME_PATCH3_MEAN,
    SITE_INITIALIZATION_STRATIFIED_GRID,
    SITE_INITIALIZATION_STRATIFIED_PIXEL_MEAN,
)


@dataclass(frozen=True)
class Site4D:
    x: float
    y: float
    z: float
    t: float
    weight: float
    rgba: tuple[float, float, float, float]


@dataclass(frozen=True)
class Boundary4D:
    left: int
    right: int
    nx: float
    ny: float
    nz: float
    nt: float
    b: float


def _load_config(path: Path, *, max_frames: int | None, render_size: int | None) -> dict[str, Any]:
    cfg = load_config_file(path)
    cfg["data"] = dict(cfg["data"])
    cfg["render"] = dict(cfg["render"])
    manifest = Path(cfg["data"]["multicam_manifest"])
    if not manifest.is_absolute():
        cfg["data"]["multicam_manifest"] = str(DYNAWORLD / manifest)
    if max_frames is not None:
        cfg["data"]["max_frames"] = int(max_frames)
    if render_size is not None:
        cfg["render"]["render_size"] = int(render_size)
    return cfg


def _frame_time(frame_index: int, frame_count: int) -> float:
    if frame_count <= 1:
        return 0.0
    return float(frame_index) / float(frame_count - 1)


def make_boundaries_4d(sites: tuple[Site4D, ...]) -> tuple[Boundary4D, ...]:
    boundaries: list[Boundary4D] = []
    for left in range(len(sites)):
        a = sites[left]
        for right in range(left + 1, len(sites)):
            c = sites[right]
            nx = 2.0 * (c.x - a.x)
            ny = 2.0 * (c.y - a.y)
            nz = 2.0 * (c.z - a.z)
            nt = 2.0 * (c.t - a.t)
            b = (
                a.x * a.x
                + a.y * a.y
                + a.z * a.z
                + a.t * a.t
                - c.x * c.x
                - c.y * c.y
                - c.z * c.z
                - c.t * c.t
                - a.weight
                + c.weight
            )
            boundaries.append(Boundary4D(left=left, right=right, nx=nx, ny=ny, nz=nz, nt=nt, b=b))
    return tuple(boundaries)


def power_distance_4d(site: Site4D, *, x: float, y: float, z: float, t: float) -> float:
    return (x - site.x) ** 2 + (y - site.y) ** 2 + (z - site.z) ** 2 + (t - site.t) ** 2 - site.weight


def owner_at_4d(sites: tuple[Site4D, ...], *, x: float, y: float, z: float, t: float) -> int:
    return min(range(len(sites)), key=lambda idx: power_distance_4d(sites[idx], x=x, y=y, z=z, t=t))


def crossing_depth_4d(
    boundary: Boundary4D,
    *,
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    t: float,
    invalid_epsilon: float,
) -> float | None:
    ox, oy, oz = origin
    dx, dy, dz = direction
    denom = boundary.nx * dx + boundary.ny * dy + boundary.nz * dz
    if abs(denom) < invalid_epsilon:
        return None
    numer = -(boundary.nx * ox + boundary.ny * oy + boundary.nz * oz + boundary.nt * t + boundary.b)
    return numer / denom


def dedupe_sorted_depths(depths: list[float], *, epsilon: float = 1.0e-6) -> list[float]:
    depths.sort()
    unique: list[float] = []
    for depth in depths:
        if not unique or abs(depth - unique[-1]) > epsilon:
            unique.append(depth)
    return unique


def candidate_depths_4d(
    boundaries: Iterable[Boundary4D],
    *,
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    t: float,
    near: float,
    far: float,
    invalid_epsilon: float,
) -> tuple[list[float], int]:
    depths: list[float] = []
    invalid = 0
    for boundary in boundaries:
        depth = crossing_depth_4d(
            boundary,
            origin=origin,
            direction=direction,
            t=t,
            invalid_epsilon=invalid_epsilon,
        )
        if depth is None:
            invalid += 1
            continue
        if near <= depth <= far:
            depths.append(depth)
    return dedupe_sorted_depths(depths), invalid


def render_one_ray(
    *,
    sites: tuple[Site4D, ...],
    boundaries: tuple[Boundary4D, ...],
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    t: float,
    near: float,
    far: float,
    invalid_epsilon: float,
    transmittance_threshold: float,
) -> tuple[tuple[float, float, float], float, float, int, int]:
    depths, invalid = candidate_depths_4d(
        boundaries,
        origin=origin,
        direction=direction,
        t=t,
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
    )
    cuts = [near, *depths, far]
    rgb = [0.0, 0.0, 0.0]
    alpha_accum = 0.0
    depth_accum = 0.0
    transmittance = 1.0
    segment_count = 0
    ox, oy, oz = origin
    dx, dy, dz = direction
    for depth0, depth1 in zip(cuts, cuts[1:]):
        if depth1 - depth0 <= EPS:
            continue
        mid = 0.5 * (depth0 + depth1)
        x = ox + dx * mid
        y = oy + dy * mid
        z = oz + dz * mid
        site = sites[owner_at_4d(sites, x=x, y=y, z=z, t=t)]
        density = max(float(site.rgba[3]), 0.0)
        segment_alpha = 1.0 - math.exp(-density * (depth1 - depth0))
        contribution = transmittance * segment_alpha
        rgb[0] += contribution * float(site.rgba[0])
        rgb[1] += contribution * float(site.rgba[1])
        rgb[2] += contribution * float(site.rgba[2])
        alpha_accum += contribution
        depth_accum += contribution * mid
        transmittance *= 1.0 - segment_alpha
        segment_count += 1
        if transmittance <= transmittance_threshold:
            break
    expected_depth = depth_accum / max(alpha_accum, EPS) if alpha_accum > 0.0 else far
    return (rgb[0], rgb[1], rgb[2]), alpha_accum, expected_depth, segment_count, invalid


def initialize_sites_from_train_samples(
    *,
    targets: torch.Tensor,
    rays: torch.Tensor,
    frame_indices: torch.Tensor,
    frame_count: int,
    site_count: int,
    near: float,
    far: float,
    density: float,
    initialization: str = SITE_INITIALIZATION_LEGACY_SPARSE,
) -> tuple[Site4D, ...]:
    if targets.ndim != 4 or rays.ndim != 4:
        raise ValueError(f"Expected targets [B,3,H,W] and rays [B,H,W,6], got {tuple(targets.shape)} and {tuple(rays.shape)}.")
    if targets.shape[0] != rays.shape[0] or tuple(targets.shape[2:]) != tuple(rays.shape[1:3]):
        raise ValueError(f"Target/ray shape mismatch: {tuple(targets.shape)} vs {tuple(rays.shape)}.")
    if site_count <= 1:
        raise ValueError("site_count must be greater than one")
    if initialization not in SITE_INITIALIZATION_CHOICES:
        raise ValueError(f"initialization must be one of {SITE_INITIALIZATION_CHOICES}, got {initialization!r}")
    sample_count = int(targets.shape[0])
    height = int(targets.shape[2])
    width = int(targets.shape[3])
    grid_cols = int(math.ceil(math.sqrt(float(site_count))))
    grid_rows = int(math.ceil(float(site_count) / float(max(grid_cols, 1))))
    sites: list[Site4D] = []
    targets_cpu = targets.detach().cpu().to(dtype=torch.float32)
    rays_cpu = rays.detach().cpu().to(dtype=torch.float32)
    frame_indices_cpu = frame_indices.detach().cpu().to(dtype=torch.long)
    for site_id in range(site_count):
        sample_index = min(int((site_id + 0.5) * sample_count / float(site_count)), sample_count - 1)
        if initialization in {SITE_INITIALIZATION_STRATIFIED_GRID, SITE_INITIALIZATION_STRATIFIED_PIXEL_MEAN}:
            grid_x = site_id % grid_cols
            grid_y = (site_id // grid_cols) % grid_rows
            x = min(int((grid_x + 0.5) * width / float(grid_cols)), width - 1)
            y = min(int((grid_y + 0.5) * height / float(grid_rows)), height - 1)
        else:
            y = (site_id * 7 + site_id // max(width, 1)) % height
            x = (site_id * 11 + site_id // max(height, 1)) % width
        ray = rays_cpu[sample_index, y, x]
        depth_fraction = (float(site_id % max(site_count - 1, 1)) + 0.5) / float(max(site_count - 1, 1))
        depth = float(near) + (float(far) - float(near)) * min(depth_fraction, 1.0)
        origin = ray[:3]
        direction = ray[3:]
        point = origin + direction * depth
        frame_index = int(frame_indices_cpu[sample_index].item())
        if initialization == SITE_INITIALIZATION_LEGACY_FRAME_PATCH3_MEAN:
            same_frame = frame_indices_cpu == frame_index
            y0 = max(y - 1, 0)
            y1 = min(y + 2, height)
            x0 = max(x - 1, 0)
            x1 = min(x + 2, width)
            color = targets_cpu[same_frame, :, y0:y1, x0:x1].mean(dim=(0, 2, 3)).clamp(0.0, 1.0)
        elif initialization == SITE_INITIALIZATION_LEGACY_FRAME_PIXEL_MEAN:
            same_frame = frame_indices_cpu == frame_index
            color = targets_cpu[same_frame, :, y, x].mean(dim=0).clamp(0.0, 1.0)
        elif initialization in {SITE_INITIALIZATION_LEGACY_PIXEL_MEAN, SITE_INITIALIZATION_STRATIFIED_PIXEL_MEAN}:
            color = targets_cpu[:, :, y, x].mean(dim=0).clamp(0.0, 1.0)
        else:
            color = targets_cpu[sample_index, :, y, x].clamp(0.0, 1.0)
        sites.append(
            Site4D(
                x=float(point[0].item()),
                y=float(point[1].item()),
                z=float(point[2].item()),
                t=_frame_time(frame_index, frame_count),
                weight=0.0,
                rgba=(float(color[0].item()), float(color[1].item()), float(color[2].item()), float(density)),
            )
        )
    return tuple(sites)


def render_samples(
    *,
    sites: tuple[Site4D, ...],
    boundaries: tuple[Boundary4D, ...],
    rays: torch.Tensor,
    frame_indices: torch.Tensor,
    frame_count: int,
    near: float,
    far: float,
    invalid_epsilon: float,
    transmittance_threshold: float,
) -> dict[str, Any]:
    rays_cpu = rays.detach().cpu().to(dtype=torch.float32)
    frame_indices_cpu = frame_indices.detach().cpu().to(dtype=torch.long)
    sample_count, height, width, payload = rays_cpu.shape
    if payload != 6:
        raise ValueError(f"Expected rays payload dimension 6, got {payload}.")
    rgb = torch.empty((sample_count, 3, height, width), dtype=torch.float32)
    alpha = torch.empty((sample_count, height, width), dtype=torch.float32)
    depth = torch.empty((sample_count, height, width), dtype=torch.float32)
    total_segments = 0
    max_segments = 0
    invalid_denominators = 0
    started_at = time.perf_counter()
    for sample_index in range(sample_count):
        t = _frame_time(int(frame_indices_cpu[sample_index].item()), frame_count)
        for y in range(height):
            for x in range(width):
                ray = rays_cpu[sample_index, y, x]
                ray_rgb, ray_alpha, ray_depth, segment_count, invalid = render_one_ray(
                    sites=sites,
                    boundaries=boundaries,
                    origin=(float(ray[0].item()), float(ray[1].item()), float(ray[2].item())),
                    direction=(float(ray[3].item()), float(ray[4].item()), float(ray[5].item())),
                    t=t,
                    near=near,
                    far=far,
                    invalid_epsilon=invalid_epsilon,
                    transmittance_threshold=transmittance_threshold,
                )
                rgb[sample_index, 0, y, x] = ray_rgb[0]
                rgb[sample_index, 1, y, x] = ray_rgb[1]
                rgb[sample_index, 2, y, x] = ray_rgb[2]
                alpha[sample_index, y, x] = ray_alpha
                depth[sample_index, y, x] = ray_depth
                total_segments += segment_count
                max_segments = max(max_segments, segment_count)
                invalid_denominators += invalid
    elapsed_s = time.perf_counter() - started_at
    return {
        "rgb": rgb,
        "alpha": alpha,
        "depth": depth,
        "elapsed_s": elapsed_s,
        "total_segments": total_segments,
        "max_segments_per_ray": max_segments,
        "invalid_denominator_count": invalid_denominators,
    }


def metric_block(rendered: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    rendered = rendered.detach().cpu().to(dtype=torch.float32).clamp(0.0, 1.0)
    target = target.detach().cpu().to(dtype=torch.float32).clamp(0.0, 1.0)
    l1 = torch.mean(torch.abs(rendered - target)).item()
    mse = torch.mean((rendered - target).square()).item()
    psnr = -10.0 * math.log10(max(float(mse), 1.0e-12))
    return {"l1": float(l1), "mse": float(mse), "psnr": float(psnr)}


def write_ppm(path: Path, image_chw: torch.Tensor) -> None:
    image = image_chw.detach().cpu().to(dtype=torch.float32).clamp(0.0, 1.0).permute(1, 2, 0)
    height, width, _ = image.shape
    data = (image * 255.0).round().to(dtype=torch.uint8).numpy().tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        handle.write(data)


def summarize_split(
    *,
    split_name: str,
    render_result: dict[str, Any],
    targets: torch.Tensor,
    boundary_count: int,
) -> dict[str, Any]:
    rgb = render_result["rgb"]
    alpha = render_result["alpha"]
    depth = render_result["depth"]
    pixel_ray_count = int(rgb.shape[0] * rgb.shape[2] * rgb.shape[3])
    metrics = metric_block(rgb, targets)
    return {
        "split": split_name,
        "rgb_shape": list(rgb.shape),
        "alpha_shape": list(alpha.shape),
        "depth_shape": list(depth.shape),
        "pixel_ray_count": pixel_ray_count,
        "linear_boundary_scans": int(pixel_ray_count * boundary_count),
        "render_elapsed_s": float(render_result["elapsed_s"]),
        "total_segments": int(render_result["total_segments"]),
        "max_segments_per_ray": int(render_result["max_segments_per_ray"]),
        "invalid_denominator_count": int(render_result["invalid_denominator_count"]),
        "rgb_min": float(rgb.min().item()),
        "rgb_max": float(rgb.max().item()),
        "rgb_std": float(rgb.std().item()),
        "alpha_min": float(alpha.min().item()),
        "alpha_max": float(alpha.max().item()),
        "alpha_std": float(alpha.std().item()),
        "depth_min": float(depth.min().item()),
        "depth_max": float(depth.max().item()),
        "depth_std": float(depth.std().item()),
        "target_l1": metrics["l1"],
        "target_mse": metrics["mse"],
        "target_psnr": metrics["psnr"],
    }


def run_reference(
    *,
    config_path: Path,
    max_frames: int | None,
    render_size: int | None,
    site_count: int,
    near: float,
    far: float,
    density: float,
    site_initialization: str = SITE_INITIALIZATION_LEGACY_SPARSE,
    invalid_epsilon: float,
    transmittance_threshold: float,
    train_ppm_out: Path | None,
    heldout_ppm_out: Path | None,
) -> dict[str, Any]:
    cfg = _load_config(config_path, max_frames=max_frames, render_size=render_size)
    data = load_powerfoam_training_data(cfg, torch.device("cpu"))
    targets = data["targets"].detach().cpu().to(dtype=torch.float32)
    sample_rays = data["sample_rays"].detach().cpu().to(dtype=torch.float32)
    sample_frame_indices = data["sample_frame_indices"].detach().cpu().to(dtype=torch.long)
    heldout_targets = data["heldout_targets"]
    heldout_rays = data["heldout_rays"]
    heldout_frame_indices = data["heldout_frame_indices"]
    if heldout_targets is None or heldout_rays is None or heldout_frame_indices is None:
        raise ValueError("Gate 1 real-ray reference requires heldout targets, rays, and frame indices.")

    frame_count = int(data["frame_count"])
    sites = initialize_sites_from_train_samples(
        targets=targets,
        rays=sample_rays,
        frame_indices=sample_frame_indices,
        frame_count=frame_count,
        site_count=site_count,
        near=near,
        far=far,
        density=density,
        initialization=site_initialization,
    )
    boundaries = make_boundaries_4d(sites)
    train_render = render_samples(
        sites=sites,
        boundaries=boundaries,
        rays=sample_rays,
        frame_indices=sample_frame_indices,
        frame_count=frame_count,
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        transmittance_threshold=transmittance_threshold,
    )
    heldout_render = render_samples(
        sites=sites,
        boundaries=boundaries,
        rays=heldout_rays.detach().cpu().to(dtype=torch.float32),
        frame_indices=heldout_frame_indices.detach().cpu().to(dtype=torch.long),
        frame_count=frame_count,
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        transmittance_threshold=transmittance_threshold,
    )

    if train_ppm_out is not None:
        write_ppm(train_ppm_out, train_render["rgb"][0])
    if heldout_ppm_out is not None:
        write_ppm(heldout_ppm_out, heldout_render["rgb"][0])

    train_summary = summarize_split(
        split_name="train",
        render_result=train_render,
        targets=targets,
        boundary_count=len(boundaries),
    )
    heldout_summary = summarize_split(
        split_name="heldout",
        render_result=heldout_render,
        targets=heldout_targets.detach().cpu().to(dtype=torch.float32),
        boundary_count=len(boundaries),
    )
    acceptance = {
        "loaded_real_multicam_bundle": str(cfg["data"]["frame_source"]) == "multicam_val",
        "consumed_train_camera_rays": list(sample_rays.shape) == [targets.shape[0], targets.shape[2], targets.shape[3], 6],
        "consumed_heldout_camera_rays": list(heldout_rays.shape)
        == [heldout_targets.shape[0], heldout_targets.shape[2], heldout_targets.shape[3], 6],
        "train_output_shape_matches_targets": train_summary["rgb_shape"] == list(targets.shape),
        "heldout_output_shape_matches_targets": heldout_summary["rgb_shape"] == list(heldout_targets.shape),
        "all_outputs_finite": bool(
            torch.isfinite(train_render["rgb"]).all().item()
            and torch.isfinite(train_render["alpha"]).all().item()
            and torch.isfinite(train_render["depth"]).all().item()
            and torch.isfinite(heldout_render["rgb"]).all().item()
            and torch.isfinite(heldout_render["alpha"]).all().item()
            and torch.isfinite(heldout_render["depth"]).all().item()
        ),
        "alpha_in_unit_interval": bool(
            train_render["alpha"].min().item() >= -1.0e-6
            and train_render["alpha"].max().item() <= 1.0 + 1.0e-6
            and heldout_render["alpha"].min().item() >= -1.0e-6
            and heldout_render["alpha"].max().item() <= 1.0 + 1.0e-6
        ),
        "rgb_nonconstant": bool(train_render["rgb"].std().item() > 0.0 and heldout_render["rgb"].std().item() > 0.0),
        "target_metrics_are_finite": all(
            math.isfinite(float(value))
            for value in (
                train_summary["target_l1"],
                train_summary["target_mse"],
                train_summary["target_psnr"],
                heldout_summary["target_l1"],
                heldout_summary["target_mse"],
                heldout_summary["target_psnr"],
            )
        ),
        "uses_4d_power_boundaries": len(boundaries) == site_count * (site_count - 1) // 2,
    }
    return {
        "benchmark": "world_foam_lane2_gate1_realray_per_sample_reference",
        "status": "ok" if all(acceptance.values()) else "failed",
        "gate": "1B_realray_per_sample_cpu_reference",
        "device": "cpu",
        "config_path": str(config_path),
        "sample_id": data["source_label"],
        "train_views": list(data["train_views"]),
        "heldout_views": list(data["heldout_views"]),
        "pose_source": data["pose_source"],
        "frame_count": frame_count,
        "render_size": int(cfg["render"]["render_size"]),
        "site_count": site_count,
        "boundary_count": len(boundaries),
        "near": float(near),
        "far": float(far),
        "density": float(density),
        "transmittance_threshold": float(transmittance_threshold),
        "renderer_scope": "cpu_real_camera_ray_4d_power_cell_per_sample_reference",
        "gradient_scope": "none_forward_only_no_shared_backward",
        "sharing_scope": "none_linear_per_sample_baseline",
        "quality_claim": False,
        "world_foam_renderer_status": "cpu_per_sample_real_camera_ray_reference_no_sharing_no_metal_no_training",
        "site_initialization": site_initialization,
        "train": train_summary,
        "heldout": heldout_summary,
        "acceptance": acceptance,
        "proof_images": {
            "train_ppm": str(train_ppm_out) if train_ppm_out is not None else None,
            "heldout_ppm": str(heldout_ppm_out) if heldout_ppm_out is not None else None,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="World Foam Gate 1 real-camera-ray CPU per-sample reference.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--render-size", type=int)
    parser.add_argument("--site-count", type=int, default=12)
    parser.add_argument("--near", type=float, default=0.05)
    parser.add_argument("--far", type=float, default=3.25)
    parser.add_argument("--density", type=float, default=2.0)
    parser.add_argument(
        "--site-initialization",
        choices=SITE_INITIALIZATION_CHOICES,
        default=SITE_INITIALIZATION_LEGACY_SPARSE,
    )
    parser.add_argument("--invalid-epsilon", type=float, default=1.0e-7)
    parser.add_argument("--transmittance-threshold", type=float, default=1.0e-4)
    parser.add_argument("--train-ppm-out", type=Path)
    parser.add_argument("--heldout-ppm-out", type=Path)
    parser.add_argument("--out-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_reference(
        config_path=args.config,
        max_frames=args.max_frames,
        render_size=args.render_size,
        site_count=args.site_count,
        near=args.near,
        far=args.far,
        density=args.density,
        site_initialization=args.site_initialization,
        invalid_epsilon=args.invalid_epsilon,
        transmittance_threshold=args.transmittance_threshold,
        train_ppm_out=args.train_ppm_out,
        heldout_ppm_out=args.heldout_ppm_out,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    if payload["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
