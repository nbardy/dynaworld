from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

BENCHMARK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCHMARK_DIR.parents[1]
TRAIN_DIR = PROJECT_ROOT / "src" / "train"
VENDORED_TAICHI_SPLATTING_DIR = PROJECT_ROOT / "third_party" / "taichi-splatting"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))
if VENDORED_TAICHI_SPLATTING_DIR.exists() and str(VENDORED_TAICHI_SPLATTING_DIR) not in sys.path:
    sys.path.insert(0, str(VENDORED_TAICHI_SPLATTING_DIR))

from camera import CameraSpec
from config_utils import load_config_file
from memory_efficient_splat_rasterizer import MemoryEfficientGaussianRasterizer, RasterizeConfig
from raw_metal_mlx_bridge import RawMetalUnavailable, import_raw_metal, render_projected_torch
from renderers.common import project_gaussians_2d
from renderers.overlap_metrics import (
    custom_rect_overlap_stats,
    exact_conic_overlap_stats,
    prefix_overlap_stats,
    selected_overlap_summary,
    taichi_obb_overlap_stats,
)
from rendering import build_or_reuse_grid, render_gaussian_frame
from runtime_types import GaussianFrame
from vectorized_sparse_splat_rasterizer import SparseRasterConfig, VectorizedSparseGaussianRasterizer

DEFAULT_CONFIG: dict[str, Any] = {
    "device": "auto",
    "dtype": "float32",
    "renderers": ["custom_dense", "custom_tiled", "gsplat", "taichi", "vectorized_sparse"],
    "resolutions": [[64, 64]],
    "splat_counts": [128],
    "sets_per_case": 5,
    "warmup_iters": 2,
    "timed_iters": 5,
    "backward": True,
    "compare_outputs": True,
    "seed": 20260420,
    "save_images": {
        "enabled": True,
        "directory": "benchmark_outputs/splat_renderer_images",
        "set_index": 0,
        "largest_resolution_only": True,
        "largest_splat_count_only": True,
    },
    "random_splats": {
        "depth_range": [1.5, 4.0],
        "opacity_range": [0.08, 0.85],
        "scale_pixel_range": [1.0, 5.0],
        "xy_margin_fraction": 0.08,
        "focal_length_factor": 1.15,
    },
    "custom": {
        "mode": "tiled",
        "tile_size": 8,
        "bound_scale": 3.0,
        "alpha_threshold": 1.0 / 255.0,
    },
    "sparse": {
        "alpha_threshold": 1.0 / 255.0,
        "max_alpha": 0.99,
        "exact_ellipse_mask": True,
        "pixel_center_offset": 0.0,
        "use_checkpoint": False,
        "compile_core": False,
        "accumulation_dtype": "float32",
    },
    "taichi": {
        "variant": "auto",
        "sort_backend": "auto",
        "backward_variant": "pixel_reference",
        "metal_block_dim": 0,
        "use_depth16": False,
        "tile_size": 16,
        "alpha_threshold": 1.0 / 255.0,
        "clamp_max_alpha": 0.99,
        "saturate_threshold": 0.9999,
        "metal_compatible": True,
        "precision": {
            "compute_dtype": "input",
        },
    },
    "raw_metal": {
        "tile_size": 16,
        "chunk_size": 32,
        "alpha_threshold": 1.0 / 255.0,
        "transmittance_threshold": 1.0e-4,
    },
    "gsplat": {
        "tile_size": 16,
        "packed": True,
        "near_plane": 0.01,
        "far_plane": 1.0e10,
        "radius_clip": 0.0,
        "eps2d": 0.3,
        "rasterize_mode": "classic",
    },
    "overlap_stats": {
        "enabled": True,
        "variants": ["custom_rect", "taichi_obb", "exact_conic"],
        "print_per_case": True,
        "large_splat_tile_threshold": 64,
        "batch_size": 8192,
        "max_candidate_pairs_per_batch": 1_000_000,
    },
}


@dataclass(frozen=True)
class SplatCase:
    frame: GaussianFrame
    camera: CameraSpec
    height: int
    width: int
    splat_count: int
    case_seed: int
    set_index: int


@dataclass(frozen=True)
class ProjectedSplats:
    means2d: torch.Tensor
    conics: torch.Tensor
    colors: torch.Tensor
    opacities: torch.Tensor
    depths: torch.Tensor
    background: torch.Tensor


@dataclass(frozen=True)
class RendererSpec:
    name: str
    render: Callable[[SplatCase, dict[str, Any]], torch.Tensor]
    available: bool = True
    skip_reason: str = ""
    sync: Callable[[torch.device], None] | None = None


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def normalize_resolution(value: Any) -> tuple[int, int]:
    if isinstance(value, int):
        if value < 1:
            raise ValueError(f"Resolution must be positive, got {value}.")
        return value, value
    if isinstance(value, str):
        if "x" in value:
            left, right = value.lower().split("x", 1)
            return normalize_resolution([int(left), int(right)])
        return normalize_resolution(int(value))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        height, width = int(value[0]), int(value[1])
        if height < 1 or width < 1:
            raise ValueError(f"Resolution must be positive, got {height}x{width}.")
        return height, width
    raise ValueError(f"Expected resolution as int, 'HxW', or [height, width], got {value!r}.")


def parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_csv_resolutions(value: str) -> list[tuple[int, int]]:
    return [normalize_resolution(part.strip()) for part in value.split(",") if part.strip()]


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def sync_renderer(renderer: RendererSpec, device: torch.device) -> None:
    if renderer.sync is not None:
        renderer.sync(device)
    else:
        sync_device(device)


def dtype_from_name(name: str) -> torch.dtype:
    if name == "input":
        raise ValueError("'input' is a Taichi precision sentinel, not a torch dtype.")
    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"Unknown torch dtype: {name}") from exc


def optional_dtype_from_name(name: str, fallback: torch.dtype) -> torch.dtype:
    if name == "input":
        return fallback
    return dtype_from_name(name)


def make_camera(height: int, width: int, device: torch.device, dtype: torch.dtype, cfg: dict[str, Any]) -> CameraSpec:
    focal = float(max(height, width)) * float(cfg["focal_length_factor"])
    return CameraSpec(
        fx=focal,
        fy=focal,
        cx=(float(width) - 1.0) * 0.5,
        cy=(float(height) - 1.0) * 0.5,
        camera_to_world=torch.eye(4, device=device, dtype=dtype),
    )


def make_random_case(
    height: int,
    width: int,
    splat_count: int,
    set_index: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    cfg: dict[str, Any],
) -> SplatCase:
    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(seed)

    camera = make_camera(height, width, torch.device("cpu"), torch.float32, cfg)
    min_depth, max_depth = [float(v) for v in cfg["depth_range"]]
    min_opacity, max_opacity = [float(v) for v in cfg["opacity_range"]]
    min_scale_px, max_scale_px = [float(v) for v in cfg["scale_pixel_range"]]
    margin_fraction = float(cfg["xy_margin_fraction"])

    margin_x = float(width) * margin_fraction
    margin_y = float(height) * margin_fraction
    u = torch.rand(splat_count, generator=cpu_generator) * max(float(width) - 2.0 * margin_x, 1.0) + margin_x
    v = torch.rand(splat_count, generator=cpu_generator) * max(float(height) - 2.0 * margin_y, 1.0) + margin_y
    z = torch.rand(splat_count, generator=cpu_generator) * (max_depth - min_depth) + min_depth
    x = (u - float(camera.cx)) * z / float(camera.fx)
    y = (v - float(camera.cy)) * z / float(camera.fy)
    xyz = torch.stack([x, y, z], dim=-1)

    scale_px = torch.rand(splat_count, 3, generator=cpu_generator) * (max_scale_px - min_scale_px) + min_scale_px
    focal = 0.5 * (float(camera.fx) + float(camera.fy))
    scales = scale_px * z.unsqueeze(-1) / focal
    scales[:, 2] = scales[:, 2] * 0.5

    quats = F.normalize(torch.randn(splat_count, 4, generator=cpu_generator), dim=-1)
    opacities = torch.rand(splat_count, 1, generator=cpu_generator) * (max_opacity - min_opacity) + min_opacity
    rgbs = torch.rand(splat_count, 3, generator=cpu_generator)

    frame = GaussianFrame(
        xyz=xyz.to(device=device, dtype=dtype),
        scales=scales.to(device=device, dtype=dtype),
        quats=quats.to(device=device, dtype=dtype),
        opacities=opacities.to(device=device, dtype=dtype),
        rgbs=rgbs.to(device=device, dtype=dtype),
    )
    return SplatCase(
        frame=frame,
        camera=make_camera(height, width, device, dtype, cfg),
        height=height,
        width=width,
        splat_count=splat_count,
        case_seed=seed,
        set_index=set_index,
    )


def clone_case_for_grad(case: SplatCase, backward: bool) -> SplatCase:
    def clone_tensor(value: torch.Tensor) -> torch.Tensor:
        cloned = value.detach().clone()
        if backward:
            cloned.requires_grad_(True)
        return cloned

    return SplatCase(
        frame=GaussianFrame(
            xyz=clone_tensor(case.frame.xyz),
            scales=clone_tensor(case.frame.scales),
            quats=clone_tensor(case.frame.quats),
            opacities=clone_tensor(case.frame.opacities),
            rgbs=clone_tensor(case.frame.rgbs),
        ),
        camera=case.camera,
        height=case.height,
        width=case.width,
        splat_count=case.splat_count,
        case_seed=case.case_seed,
        set_index=case.set_index,
    )


def clear_frame_grads(frame: GaussianFrame) -> None:
    for tensor in (frame.xyz, frame.scales, frame.quats, frame.opacities, frame.rgbs):
        tensor.grad = None


def project_for_sparse(case: SplatCase, sparse_cfg: dict[str, Any]) -> ProjectedSplats:
    means2d, inv_cov2d, _cov2d, opacities, colors = project_gaussians_2d(
        case.frame.xyz,
        case.frame.scales,
        case.frame.quats,
        case.frame.opacities,
        case.frame.rgbs,
        case.camera.fx,
        case.camera.fy,
        case.camera.cx,
        case.camera.cy,
        camera_to_world=case.camera.camera_to_world,
    )
    conics = torch.stack([inv_cov2d[:, 0, 0], inv_cov2d[:, 0, 1], inv_cov2d[:, 1, 1]], dim=-1)
    depths = torch.arange(means2d.shape[0], device=means2d.device, dtype=means2d.dtype)
    background = torch.ones(colors.shape[-1], device=colors.device, dtype=colors.dtype)
    return ProjectedSplats(
        means2d=means2d,
        conics=conics,
        colors=colors,
        opacities=opacities.squeeze(-1),
        depths=depths,
        background=background,
    )


def render_custom_mode(
    case: SplatCase,
    config: dict[str, Any],
    mode: str,
    dense_grid_cache: dict[tuple[int, int, str], torch.Tensor] | None = None,
) -> torch.Tensor:
    custom_cfg = config["custom"]
    dense_grid = None
    if mode == "dense" and dense_grid_cache is not None:
        cache_key = (case.height, case.width, str(case.frame.xyz.device))
        dense_grid = dense_grid_cache.get(cache_key)
        dense_grid = build_or_reuse_grid(case.height, case.width, case.frame.xyz.device, dense_grid)
        dense_grid_cache[cache_key] = dense_grid
    return render_gaussian_frame(
        case.frame,
        camera=case.camera,
        height=case.height,
        width=case.width,
        mode=mode,
        dense_grid=dense_grid,
        tile_size=int(custom_cfg["tile_size"]),
        bound_scale=float(custom_cfg["bound_scale"]),
        alpha_threshold=float(custom_cfg["alpha_threshold"]),
    )


def sparse_config(config: dict[str, Any]) -> SparseRasterConfig:
    cfg = config["sparse"]
    return SparseRasterConfig(
        alpha_threshold=float(cfg["alpha_threshold"]),
        max_alpha=float(cfg["max_alpha"]),
        exact_ellipse_mask=bool(cfg["exact_ellipse_mask"]),
        pixel_center_offset=float(cfg["pixel_center_offset"]),
        use_checkpoint=bool(cfg["use_checkpoint"]),
        compile_core=bool(cfg["compile_core"]),
        accumulation_dtype=str(cfg["accumulation_dtype"]),
    )


def memory_config(config: dict[str, Any]) -> RasterizeConfig:
    cfg = config["sparse"]
    return RasterizeConfig(
        alpha_threshold=float(cfg["alpha_threshold"]),
        max_alpha=float(cfg["max_alpha"]),
        exact_ellipse_mask=bool(cfg["exact_ellipse_mask"]),
        pixel_center_offset=float(cfg["pixel_center_offset"]),
    )


def render_vectorized_sparse(
    case: SplatCase,
    config: dict[str, Any],
    rasterizer: VectorizedSparseGaussianRasterizer,
) -> torch.Tensor:
    projected = project_for_sparse(case, config["sparse"])
    image_hwc = rasterizer(
        projected.means2d,
        projected.conics,
        projected.colors,
        projected.opacities,
        projected.depths,
        (case.height, case.width),
        projected.background,
    )
    return image_hwc.permute(2, 0, 1).contiguous()


def render_memory_efficient_sparse(
    case: SplatCase,
    config: dict[str, Any],
    rasterizer: MemoryEfficientGaussianRasterizer,
) -> torch.Tensor:
    projected = project_for_sparse(case, config["sparse"])
    image_hwc = rasterizer(
        projected.means2d,
        projected.conics,
        projected.colors,
        projected.opacities,
        projected.depths,
        (case.height, case.width),
        projected.background,
    )
    return image_hwc.permute(2, 0, 1).contiguous()


def render_raw_metal(case: SplatCase, config: dict[str, Any]) -> torch.Tensor:
    projected = project_for_sparse(case, config["sparse"])
    return render_projected_torch(
        projected.means2d,
        projected.conics,
        projected.colors,
        projected.opacities,
        projected.depths,
        case.height,
        case.width,
        projected.background,
        config["raw_metal"],
        output_device=case.frame.rgbs.device,
    )


def project_for_taichi_axis(case: SplatCase) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    means2d, _inv_cov2d, cov2d, opacities, colors = project_gaussians_2d(
        case.frame.xyz,
        case.frame.scales,
        case.frame.quats,
        case.frame.opacities,
        case.frame.rgbs,
        case.camera.fx,
        case.camera.fy,
        case.camera.cx,
        case.camera.cy,
        camera_to_world=case.camera.camera_to_world,
    )
    cov_xx = cov2d[:, 0, 0]
    cov_xy = cov2d[:, 0, 1]
    cov_yy = cov2d[:, 1, 1]
    trace_half = 0.5 * (cov_xx + cov_yy)
    delta = torch.sqrt(torch.clamp((0.5 * (cov_xx - cov_yy)).square() + cov_xy.square(), min=1e-12))
    lambda_major = trace_half + delta
    lambda_minor = trace_half - delta
    axis = torch.stack([cov_xy, lambda_major - cov_xx], dim=-1)
    axis_norm = torch.linalg.norm(axis, dim=-1, keepdim=True)
    fallback_axis = torch.zeros_like(axis)
    fallback_axis[:, 0] = 1.0
    axis = torch.where(axis_norm > 1e-6, axis / axis_norm.clamp_min(1e-6), fallback_axis)
    sigma = torch.sqrt(torch.clamp(torch.stack([lambda_major, lambda_minor], dim=-1), min=1e-6))
    packed = torch.cat([means2d, axis, sigma, opacities], dim=-1).contiguous()
    depths = torch.arange(packed.shape[0], device=packed.device, dtype=packed.dtype).view(-1, 1)
    if packed.shape[0] > 1:
        depths = depths / float(packed.shape[0] - 1)
    return (
        packed,
        depths.contiguous(),
        colors.contiguous(),
        torch.ones(colors.shape[-1], device=colors.device, dtype=colors.dtype),
    )


def project_for_custom_rect_stats(case: SplatCase) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    means2d, _inv_cov2d, cov2d, opacities, _colors = project_gaussians_2d(
        case.frame.xyz,
        case.frame.scales,
        case.frame.quats,
        case.frame.opacities,
        case.frame.rgbs,
        case.camera.fx,
        case.camera.fy,
        case.camera.cx,
        case.camera.cy,
        camera_to_world=case.camera.camera_to_world,
    )
    return means2d, cov2d, opacities


@torch.no_grad()
def compute_case_overlap_stats(case: SplatCase, config: dict[str, Any]) -> dict[str, float]:
    overlap_cfg = config.get("overlap_stats", {})
    if not bool(overlap_cfg.get("enabled", False)):
        return {}

    variants = [str(value) for value in overlap_cfg.get("variants", [])]
    if not variants:
        return {}

    large_splat_threshold = int(overlap_cfg.get("large_splat_tile_threshold", 64))
    batch_size = int(overlap_cfg.get("batch_size", 8192))
    max_candidate_pairs = int(overlap_cfg.get("max_candidate_pairs_per_batch", 1_000_000))
    stats_by_variant: dict[str, dict[str, float]] = {}

    if "custom_rect" in variants:
        custom_cfg = config["custom"]
        means2d, cov2d, opacities = project_for_custom_rect_stats(case)
        stats_by_variant["custom_rect"] = custom_rect_overlap_stats(
            means2d,
            cov2d,
            opacities,
            (case.height, case.width),
            tile_size=int(custom_cfg["tile_size"]),
            bound_scale=float(custom_cfg["bound_scale"]),
            alpha_threshold=float(custom_cfg["alpha_threshold"]),
            large_splat_tile_threshold=large_splat_threshold,
            batch_size=batch_size,
            max_candidate_pairs_per_batch=max_candidate_pairs,
        )

    needs_sparse_projection = any(variant in variants for variant in {"exact_conic", "custom_rect"})
    projected = None
    if needs_sparse_projection:
        projected = project_for_sparse(case, config["sparse"])
        if "exact_conic" in variants:
            tile_size = int(overlap_cfg.get("tile_size", config["taichi"]["tile_size"]))
            alpha_threshold = float(overlap_cfg.get("alpha_threshold", config["taichi"]["alpha_threshold"]))
            stats_by_variant["exact_conic"] = exact_conic_overlap_stats(
                projected.means2d,
                projected.conics,
                projected.opacities,
                (case.height, case.width),
                tile_size=tile_size,
                alpha_threshold=alpha_threshold,
                large_splat_tile_threshold=large_splat_threshold,
                batch_size=batch_size,
                max_candidate_pairs_per_batch=max_candidate_pairs,
            )

    if "custom_rect" in stats_by_variant and projected is not None:
        custom_tile_size = int(config["custom"]["tile_size"])
        exact_tile_size = int(stats_by_variant.get("exact_conic", {}).get("tile_size", -1))
        if exact_tile_size != custom_tile_size:
            stats_by_variant["exact_conic_custom_tile"] = exact_conic_overlap_stats(
                projected.means2d,
                projected.conics,
                projected.opacities,
                (case.height, case.width),
                tile_size=custom_tile_size,
                alpha_threshold=float(config["custom"]["alpha_threshold"]),
                large_splat_tile_threshold=large_splat_threshold,
                batch_size=batch_size,
                max_candidate_pairs_per_batch=max_candidate_pairs,
            )

    if "taichi_obb" in variants:
        packed, _depths, _colors, _background = project_for_taichi_axis(case)
        taichi_cfg = config["taichi"]
        tile_size = int(overlap_cfg.get("tile_size", taichi_cfg["tile_size"]))
        alpha_threshold = float(overlap_cfg.get("alpha_threshold", taichi_cfg["alpha_threshold"]))
        stats_by_variant["taichi_obb"] = taichi_obb_overlap_stats(
            packed,
            (case.height, case.width),
            tile_size=tile_size,
            alpha_threshold=alpha_threshold,
            large_splat_tile_threshold=large_splat_threshold,
            batch_size=batch_size,
            max_candidate_pairs_per_batch=max_candidate_pairs,
        )

    row_stats: dict[str, float] = {}
    for variant, variant_stats in stats_by_variant.items():
        row_stats.update(prefix_overlap_stats(f"overlap_{variant}", variant_stats))

    if "exact_conic" in stats_by_variant and "taichi_obb" in stats_by_variant:
        exact_keys = stats_by_variant["exact_conic"]["total_overlap_keys"]
        obb_keys = stats_by_variant["taichi_obb"]["total_overlap_keys"]
        if obb_keys > 0:
            row_stats["overlap_exact_vs_taichi_obb_key_delta_pct"] = 100.0 * ((exact_keys / obb_keys) - 1.0)
            row_stats["overlap_taichi_obb_to_exact_key_ratio"] = obb_keys / max(exact_keys, 1.0)

    custom_exact_variant = "exact_conic_custom_tile" if "exact_conic_custom_tile" in stats_by_variant else "exact_conic"
    if custom_exact_variant in stats_by_variant and "custom_rect" in stats_by_variant:
        exact_keys = stats_by_variant[custom_exact_variant]["total_overlap_keys"]
        custom_keys = stats_by_variant["custom_rect"]["total_overlap_keys"]
        if custom_keys > 0:
            row_stats["overlap_exact_same_tile_vs_custom_rect_key_delta_pct"] = 100.0 * (
                (exact_keys / custom_keys) - 1.0
            )
            row_stats["overlap_custom_rect_to_exact_same_tile_key_ratio"] = custom_keys / max(exact_keys, 1.0)
            row_stats["overlap_custom_rect_exact_reference_tile_size"] = stats_by_variant[custom_exact_variant][
                "tile_size"
            ]

    return row_stats


def _overlap_variant_names(overlap_stats: dict[str, float]) -> list[str]:
    variants: set[str] = set()
    suffix = "_total_overlap_keys"
    for key in overlap_stats:
        if key.startswith("overlap_") and key.endswith(suffix):
            variants.add(key[len("overlap_") : -len(suffix)])
    return sorted(variants)


def print_overlap_case(row_stats: dict[str, float], height: int, width: int, splat_count: int, set_index: int) -> None:
    if not row_stats:
        return
    print(f"overlap_stats              {height:>4}x{width:<4} G={splat_count:<6} set={set_index:<2}")
    for variant in _overlap_variant_names(row_stats):
        prefix = f"overlap_{variant}_"
        summary = selected_overlap_summary(
            {key[len(prefix) :]: value for key, value in row_stats.items() if key.startswith(prefix)}
        )
        print(
            f"  {variant:<18} "
            f"K={summary.get('total_overlap_keys', math.nan):>10.0f} "
            f"K/G={summary.get('duplication_factor', math.nan):>7.3f} "
            f"splat_tiles p95/max={summary.get('p95_tiles_per_splat', math.nan):>5.1f}/"
            f"{summary.get('max_tiles_per_splat', math.nan):>5.0f} "
            f"tile_splats p95/max={summary.get('p95_splats_per_tile', math.nan):>5.1f}/"
            f"{summary.get('max_splats_per_tile', math.nan):>5.0f}"
        )
    if "overlap_exact_vs_taichi_obb_key_delta_pct" in row_stats:
        print(
            "  exact_conic vs taichi_obb "
            f"key_delta={row_stats['overlap_exact_vs_taichi_obb_key_delta_pct']:>7.2f}% "
            f"taichi/exact={row_stats['overlap_taichi_obb_to_exact_key_ratio']:>6.3f}x"
        )
    if "overlap_exact_same_tile_vs_custom_rect_key_delta_pct" in row_stats:
        print(
            "  exact_conic vs custom_rect "
            f"tile={row_stats['overlap_custom_rect_exact_reference_tile_size']:.0f} "
            f"key_delta={row_stats['overlap_exact_same_tile_vs_custom_rect_key_delta_pct']:>7.2f}% "
            f"custom/exact={row_stats['overlap_custom_rect_to_exact_same_tile_key_ratio']:>6.3f}x"
        )


def camera_viewmat_and_intrinsics(case: SplatCase) -> tuple[torch.Tensor, torch.Tensor]:
    device = case.frame.xyz.device
    dtype = case.frame.xyz.dtype
    viewmat = torch.linalg.inv(case.camera.camera_to_world.to(device=device, dtype=dtype)).unsqueeze(0).contiguous()
    intrinsics = torch.tensor(
        [
            [float(case.camera.fx), 0.0, float(case.camera.cx)],
            [0.0, float(case.camera.fy), float(case.camera.cy)],
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    ).unsqueeze(0)
    return viewmat, intrinsics


def build_gsplat_renderer(device: torch.device) -> RendererSpec:
    if device.type != "cuda":
        return RendererSpec(
            "gsplat",
            render=lambda _case, _config: torch.empty(0),
            available=False,
            skip_reason="gsplat rasterization is CUDA-only; no MPS/Metal backend",
        )

    try:
        try:
            from gsplat.rendering import rasterization as gsplat_rasterization
        except ImportError:
            from gsplat import rasterization as gsplat_rasterization
    except Exception as exc:  # pragma: no cover - depends on optional gsplat environment.
        return RendererSpec(
            "gsplat", render=lambda _case, _config: torch.empty(0), available=False, skip_reason=str(exc)
        )

    def render_gsplat(case: SplatCase, outer_config: dict[str, Any]) -> torch.Tensor:
        gsplat_cfg = outer_config["gsplat"]
        viewmat, intrinsics = camera_viewmat_and_intrinsics(case)
        background = torch.ones(
            (1, case.frame.rgbs.shape[-1]), device=case.frame.rgbs.device, dtype=case.frame.rgbs.dtype
        )
        rendered, _alphas, _meta = gsplat_rasterization(
            means=case.frame.xyz.contiguous(),
            quats=case.frame.quats.contiguous(),
            scales=case.frame.scales.contiguous(),
            opacities=case.frame.opacities.squeeze(-1).contiguous(),
            colors=case.frame.rgbs.contiguous(),
            viewmats=viewmat,
            Ks=intrinsics,
            width=case.width,
            height=case.height,
            near_plane=float(gsplat_cfg["near_plane"]),
            far_plane=float(gsplat_cfg["far_plane"]),
            radius_clip=float(gsplat_cfg["radius_clip"]),
            eps2d=float(gsplat_cfg["eps2d"]),
            packed=bool(gsplat_cfg["packed"]),
            tile_size=int(gsplat_cfg["tile_size"]),
            backgrounds=background,
            render_mode="RGB",
            rasterize_mode=str(gsplat_cfg["rasterize_mode"]),
        )
        return rendered[0].permute(2, 0, 1).contiguous()

    return RendererSpec("gsplat", render=render_gsplat)


def build_raw_metal_renderer(device: torch.device, config: dict[str, Any]) -> RendererSpec:
    renderer_name = "raw_metal_mlx"
    if bool(config["backward"]):
        return RendererSpec(
            renderer_name,
            render=lambda _case, _config: torch.empty(0),
            available=False,
            skip_reason="raw_metal uses the MLX benchmark bridge, which is forward-only in the Torch throughput harness",
        )
    if dtype_from_name(str(config["dtype"])) != torch.float32:
        return RendererSpec(
            renderer_name,
            render=lambda _case, _config: torch.empty(0),
            available=False,
            skip_reason="raw_metal benchmark bridge currently uses float32 inputs",
        )
    try:
        import_raw_metal()
    except RawMetalUnavailable as exc:
        return RendererSpec(
            renderer_name, render=lambda _case, _config: torch.empty(0), available=False, skip_reason=str(exc)
        )
    return RendererSpec(renderer_name, render=render_raw_metal)


def resolve_taichi_variant(device: torch.device, taichi_cfg: dict[str, Any]) -> str:
    variant = str(taichi_cfg.get("variant", taichi_cfg.get("kernel_variant", "auto")))
    if variant == "auto":
        return "cuda_simt" if device.type == "cuda" else "metal_reference"
    return variant


def taichi_renderer_name(
    device: torch.device,
    variant: str,
    compute_dtype: torch.dtype,
    use_depth16: bool = False,
    sort_backend: str = "auto",
) -> str:
    device_part = "metal" if device.type == "mps" else device.type
    dtype_part = "" if compute_dtype == torch.float32 else f"_{str(compute_dtype).replace('torch.', '')}"
    depth_part = "_depth16" if use_depth16 else ""
    if sort_backend == "taichi_field":
        return f"taichi_{device_part}_global_sort{dtype_part}{depth_part}"
    if sort_backend == "bucket_taichi":
        return f"taichi_{device_part}_bucket_sort{dtype_part}{depth_part}"
    if sort_backend == "ordered_taichi":
        return f"taichi_{device_part}_ordered{dtype_part}{depth_part}"
    variant_part = variant.replace("metal_", "").replace("_simt", "")
    return f"taichi_{device_part}_{variant_part}{dtype_part}{depth_part}"


def build_taichi_renderer(
    device: torch.device, config: dict[str, Any], override: dict[str, Any] | None = None
) -> RendererSpec:
    if device.type not in {"cuda", "mps", "cpu"}:
        return RendererSpec(
            "taichi",
            render=lambda _case, _config: torch.empty(0),
            available=False,
            skip_reason=f"unsupported device: {device}",
        )

    taichi_cfg = deep_merge(config["taichi"], override or {})
    variant = resolve_taichi_variant(device, taichi_cfg)
    precision_cfg = taichi_cfg.get("precision", {})
    input_dtype = dtype_from_name(str(config["dtype"]))
    compute_dtype = optional_dtype_from_name(str(precision_cfg.get("compute_dtype", "input")), input_dtype)
    use_depth16 = bool(taichi_cfg.get("use_depth16", False))
    sort_backend = str(taichi_cfg.get("sort_backend", "auto"))
    renderer_name = taichi_renderer_name(device, variant, compute_dtype, use_depth16, sort_backend)
    backward_variant = str(taichi_cfg.get("backward_variant", "pixel_reference"))
    if variant not in {"metal_reference", "cuda_simt"}:
        return RendererSpec(
            renderer_name,
            render=lambda _case, _config: torch.empty(0),
            available=False,
            skip_reason=f"taichi kernel variant {variant!r} is not implemented yet",
        )
    if backward_variant != "pixel_reference":
        return RendererSpec(
            renderer_name,
            render=lambda _case, _config: torch.empty(0),
            available=False,
            skip_reason=f"taichi backward variant {backward_variant!r} is not implemented yet",
        )
    if compute_dtype != torch.float32:
        return RendererSpec(
            renderer_name,
            render=lambda _case, _config: torch.empty(0),
            available=False,
            skip_reason=(
                "taichi-splatting Metal path currently requires float32 packed Gaussians/depths; "
                "low precision needs separate tile mapper/raster kernels."
            ),
        )
    metal_compatible = bool(taichi_cfg.get("metal_compatible", device.type != "cuda"))
    try:
        import taichi as ti
        from taichi_splatting.data_types import RasterConfig as TaichiRasterConfig
        from taichi_splatting.rasterizer import rasterize as taichi_rasterize
        from taichi_splatting.taichi_queue import TaichiQueue
    except Exception as exc:  # pragma: no cover - depends on optional taichi environment.
        return RendererSpec(
            "taichi", render=lambda _case, _config: torch.empty(0), available=False, skip_reason=str(exc)
        )

    arch = {"cuda": ti.cuda, "mps": ti.metal, "cpu": ti.cpu}[device.type]
    try:
        TaichiQueue.init(arch=arch, log_level=ti.ERROR)
    except Exception as exc:  # pragma: no cover - depends on optional CUDA runtime.
        return RendererSpec(
            "taichi", render=lambda _case, _config: torch.empty(0), available=False, skip_reason=str(exc)
        )

    raster_config = TaichiRasterConfig(
        tile_size=int(taichi_cfg["tile_size"]),
        alpha_threshold=float(taichi_cfg["alpha_threshold"]),
        clamp_max_alpha=float(taichi_cfg["clamp_max_alpha"]),
        saturate_threshold=float(taichi_cfg["saturate_threshold"]),
        metal_compatible=metal_compatible,
        kernel_variant=variant,
        sort_backend=sort_backend,
        backward_variant=backward_variant,
        metal_block_dim=int(taichi_cfg.get("metal_block_dim", 0)),
    )

    def render_taichi(case: SplatCase, outer_config: dict[str, Any]) -> torch.Tensor:
        packed, depths, colors, background = project_for_taichi_axis(case)
        if compute_dtype != packed.dtype:
            packed = packed.to(dtype=compute_dtype)
            depths = depths.to(dtype=compute_dtype)
            colors = colors.to(dtype=compute_dtype)
            background = background.to(dtype=compute_dtype)
        raster = taichi_rasterize(
            packed,
            depths,
            colors,
            image_size=(case.width, case.height),
            config=raster_config,
            use_depth16=use_depth16,
        )
        image_hwc = raster.image + (1.0 - raster.image_weight).unsqueeze(-1).clamp(0.0, 1.0) * background
        return image_hwc.permute(2, 0, 1).contiguous()

    def sync_taichi(_device: torch.device) -> None:
        sync_device(device)
        TaichiQueue.run_sync(ti.sync)
        sync_device(device)

    return RendererSpec(renderer_name, render=render_taichi, sync=sync_taichi)


def build_renderers(
    device: torch.device,
    config: dict[str, Any],
    requested_keys: list[str] | None = None,
) -> dict[str, RendererSpec]:
    requested = (
        set(requested_keys)
        if requested_keys is not None
        else {
            "custom",
            "custom_dense",
            "custom_tiled",
            "vectorized_sparse",
            "memory_efficient_sparse",
            "gsplat",
            "raw_metal",
            "taichi",
            "taichi_reference",
            "taichi_fast_forward",
            "taichi_metal_sort",
            "taichi_fp16",
            "taichi_per_splat_backward",
            "taichi_depth16",
            "taichi_ordered",
        }
    )
    dense_grid_cache: dict[tuple[int, int, str], torch.Tensor] = {}
    specs: dict[str, RendererSpec] = {}
    if "custom" in requested:
        specs["custom"] = RendererSpec(
            f"custom_{config['custom']['mode']}",
            lambda case, cfg: render_custom_mode(case, cfg, str(cfg["custom"]["mode"]), dense_grid_cache),
        )
    if "custom_dense" in requested:
        specs["custom_dense"] = RendererSpec(
            "custom_dense",
            lambda case, cfg: render_custom_mode(case, cfg, "dense", dense_grid_cache),
        )
    if "custom_tiled" in requested:
        specs["custom_tiled"] = RendererSpec(
            "custom_tiled",
            lambda case, cfg: render_custom_mode(case, cfg, "tiled", dense_grid_cache),
        )
    if "vectorized_sparse" in requested:
        vectorized_rasterizer = VectorizedSparseGaussianRasterizer(sparse_config(config)).to(device)
        specs["vectorized_sparse"] = RendererSpec(
            "vectorized_sparse",
            lambda case, cfg: render_vectorized_sparse(case, cfg, vectorized_rasterizer),
        )
    if "memory_efficient_sparse" in requested:
        memory_rasterizer = MemoryEfficientGaussianRasterizer(memory_config(config)).to(device)
        specs["memory_efficient_sparse"] = RendererSpec(
            "memory_efficient_sparse",
            lambda case, cfg: render_memory_efficient_sparse(case, cfg, memory_rasterizer),
        )
    if "gsplat" in requested:
        specs["gsplat"] = build_gsplat_renderer(device)
    if "raw_metal" in requested:
        specs["raw_metal"] = build_raw_metal_renderer(device, config)
    if "taichi" in requested:
        specs["taichi"] = build_taichi_renderer(device, config)
    if "taichi_reference" in requested:
        specs["taichi_reference"] = build_taichi_renderer(
            device,
            config,
            {"variant": "metal_reference", "backward_variant": "pixel_reference"},
        )
    if "taichi_fast_forward" in requested:
        specs["taichi_fast_forward"] = build_taichi_renderer(device, config, {"variant": "metal_fast_forward"})
    if "taichi_metal_sort" in requested:
        specs["taichi_metal_sort"] = build_taichi_renderer(
            device, config, {"variant": "metal_reference", "sort_backend": "bucket_taichi"}
        )
    if "taichi_global_sort" in requested:
        specs["taichi_global_sort"] = build_taichi_renderer(
            device, config, {"variant": "metal_reference", "sort_backend": "taichi_field"}
        )
    if "taichi_fp16" in requested:
        specs["taichi_fp16"] = build_taichi_renderer(
            device, config, {"variant": "metal_reference", "precision": {"compute_dtype": "float16"}}
        )
    if "taichi_per_splat_backward" in requested:
        specs["taichi_per_splat_backward"] = build_taichi_renderer(
            device,
            config,
            {"variant": "metal_reference", "backward_variant": "per_splat"},
        )
    if "taichi_depth16" in requested:
        specs["taichi_depth16"] = build_taichi_renderer(
            device, config, {"variant": "metal_reference", "use_depth16": True}
        )
    if "taichi_ordered" in requested:
        specs["taichi_ordered"] = build_taichi_renderer(
            device, config, {"variant": "metal_reference", "sort_backend": "ordered_taichi"}
        )
    return specs


def timed_renderer_run(
    renderer: RendererSpec,
    base_case: SplatCase,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    if not renderer.available:
        return {
            "status": "skipped",
            "skip_reason": renderer.skip_reason,
            "renderer": renderer.name,
        }

    backward = bool(config["backward"])
    warmup_iters = int(config["warmup_iters"])
    timed_iters = int(config["timed_iters"])
    if timed_iters < 1:
        raise ValueError("timed_iters must be at least 1.")

    last_output = None
    for _ in range(warmup_iters):
        case = clone_case_for_grad(base_case, backward)
        output = renderer.render(case, config)
        if backward:
            output.square().mean().backward()
            clear_frame_grads(case.frame)
        sync_renderer(renderer, device)
        last_output = output.detach()

    forward_times = []
    backward_times = []
    total_times = []
    losses = []
    for _ in range(timed_iters):
        case = clone_case_for_grad(base_case, backward)
        sync_renderer(renderer, device)
        start = time.perf_counter()
        output = renderer.render(case, config)
        sync_renderer(renderer, device)
        after_forward = time.perf_counter()
        loss_value = math.nan
        if backward:
            loss = output.square().mean()
            loss.backward()
            clear_frame_grads(case.frame)
        sync_renderer(renderer, device)
        end = time.perf_counter()
        if backward:
            loss_value = float(loss.detach().item())

        last_output = output.detach()
        forward_times.append(after_forward - start)
        backward_times.append(end - after_forward)
        total_times.append(end - start)
        if not math.isnan(loss_value):
            losses.append(loss_value)

    return {
        "status": "ok",
        "renderer": renderer.name,
        "forward_ms": 1000.0 * sum(forward_times) / len(forward_times),
        "backward_ms": 1000.0 * sum(backward_times) / len(backward_times) if backward else 0.0,
        "total_ms": 1000.0 * sum(total_times) / len(total_times),
        "forward_fps": 1000.0 / (1000.0 * sum(forward_times) / len(forward_times)),
        "loss": sum(losses) / len(losses) if losses else math.nan,
        "output_mean": float(last_output.mean().item()) if last_output is not None else math.nan,
        "output": last_output,
    }


def output_comparison(reference: torch.Tensor | None, output: torch.Tensor | None) -> dict[str, float]:
    if reference is None or output is None or reference.shape != output.shape:
        return {"max_abs_vs_custom": math.nan, "mean_abs_vs_custom": math.nan}
    diff = (output - reference).abs()
    return {
        "max_abs_vs_custom": float(diff.max().item()),
        "mean_abs_vs_custom": float(diff.mean().item()),
    }


def safe_filename_part(value: Any) -> str:
    text = str(value).strip().replace(" ", "_")
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in text)


def resolve_output_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def image_save_target(
    config: dict[str, Any], resolutions: list[tuple[int, int]], splat_counts: list[int]
) -> tuple[set[tuple[int, int]], set[int], int]:
    save_config = config.get("save_images", {})
    if bool(save_config.get("largest_resolution_only", True)):
        max_area = max(height * width for height, width in resolutions)
        target_resolutions = {(height, width) for height, width in resolutions if height * width == max_area}
    else:
        target_resolutions = set(resolutions)

    if bool(save_config.get("largest_splat_count_only", True)):
        target_splat_counts = {max(splat_counts)}
    else:
        target_splat_counts = set(splat_counts)

    return target_resolutions, target_splat_counts, int(save_config.get("set_index", 0))


def should_save_image(
    row: dict[str, Any],
    target_resolutions: set[tuple[int, int]],
    target_splat_counts: set[int],
    target_set_index: int,
) -> bool:
    return (
        row["status"] == "ok"
        and (int(row["height"]), int(row["width"])) in target_resolutions
        and int(row["splat_count"]) in target_splat_counts
        and int(row["set_index"]) == target_set_index
    )


def save_render_image(output: torch.Tensor, row: dict[str, Any], directory: Path) -> Path:
    from PIL import Image

    image = output.detach()
    if image.ndim != 3:
        raise ValueError(f"Expected renderer output as CHW image, got shape {tuple(image.shape)}.")
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    elif image.shape[0] > 3:
        image = image[:3]

    image_hwc = image.clamp(0.0, 1.0).nan_to_num(0.0).permute(1, 2, 0).mul(255.0).round().to(torch.uint8).cpu().numpy()
    directory.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{safe_filename_part(row['renderer'])}"
        f"__{int(row['height'])}x{int(row['width'])}"
        f"__G{int(row['splat_count'])}"
        f"__set{int(row['set_index'])}.png"
    )
    path = directory / filename
    Image.fromarray(image_hwc).save(path)
    return path


def write_outputs(rows: list[dict[str, Any]], jsonl_path: Path | None, csv_path: Path | None) -> None:
    serializable_rows = [{key: value for key, value in row.items() if key != "output"} for row in rows]
    if jsonl_path is not None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("w") as handle:
            for row in serializable_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in serializable_rows for key in row.keys()})
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(serializable_rows)


def print_row(row: dict[str, Any]) -> None:
    if row["status"] == "ok":
        forward_fps = 1000.0 / row["forward_ms"] if row["forward_ms"] > 0 else math.inf
        max_abs = row.get("max_abs_vs_custom", math.nan)
        print(
            f"{row['renderer']:<26} "
            f"{row['height']:>4}x{row['width']:<4} "
            f"G={row['splat_count']:<6} "
            f"set={row['set_index']:<2} "
            f"fwd={row['forward_ms']:>8.3f}ms "
            f"fps={forward_fps:>7.1f} "
            f"bwd={row['backward_ms']:>8.3f}ms "
            f"total={row['total_ms']:>8.3f}ms "
            f"max_abs={max_abs:>8.5f}"
        )
    elif row["status"] == "skipped":
        print(
            f"{row['renderer']:<26} "
            f"{row['height']:>4}x{row['width']:<4} "
            f"G={row['splat_count']:<6} "
            f"set={row['set_index']:<2} skipped: {row['skip_reason']}"
        )
    else:
        print(
            f"{row['renderer']:<26} "
            f"{row['height']:>4}x{row['width']:<4} "
            f"G={row['splat_count']:<6} "
            f"set={row['set_index']:<2} error: {row['error']}"
        )


def print_summary(rows: list[dict[str, Any]]) -> None:
    ok_rows = [row for row in rows if row["status"] == "ok"]
    key_fn = lambda row: (row["renderer"], row["height"], row["width"], row["splat_count"])
    if ok_rows:
        print("\nSummary mean total_ms by renderer/resolution/splats:")
        for key, group_iter in itertools.groupby(sorted(ok_rows, key=key_fn), key=key_fn):
            group = list(group_iter)
            renderer, height, width, splat_count = key
            mean_total = sum(row["total_ms"] for row in group) / len(group)
            mean_forward = sum(row["forward_ms"] for row in group) / len(group)
            mean_backward = sum(row["backward_ms"] for row in group) / len(group)
            forward_fps = 1000.0 / mean_forward if mean_forward > 0 else math.inf
            print(
                f"{renderer:<26} {height:>4}x{width:<4} G={splat_count:<6} "
                f"fwd={mean_forward:>8.3f}ms fps={forward_fps:>7.1f} "
                f"bwd={mean_backward:>8.3f}ms total={mean_total:>8.3f}ms"
            )

    skipped_rows = [row for row in rows if row["status"] == "skipped"]
    if skipped_rows:
        print("\nSummary skipped renderer cases:")
        skip_key_fn = lambda row: (
            row["renderer"],
            row["height"],
            row["width"],
            row["splat_count"],
            row["skip_reason"],
        )
        for key, group_iter in itertools.groupby(sorted(skipped_rows, key=skip_key_fn), key=skip_key_fn):
            group = list(group_iter)
            renderer, height, width, splat_count, reason = key
            print(f"{renderer:<26} {height:>4}x{width:<4} G={splat_count:<6} skipped={len(group):<3} reason={reason}")

    error_rows = [row for row in rows if row["status"] == "error"]
    if error_rows:
        print("\nSummary errored renderer cases:")
        for key, group_iter in itertools.groupby(sorted(error_rows, key=key_fn), key=key_fn):
            group = list(group_iter)
            renderer, height, width, splat_count = key
            print(f"{renderer:<26} {height:>4}x{width:<4} G={splat_count:<6} errors={len(group)}")

    print_overlap_summary(rows)

    saved_image_rows = [row for row in rows if row.get("saved_image_path")]
    if saved_image_rows:
        print("\nSaved render images:")
        for row in saved_image_rows:
            print(f"{row['renderer']:<26} {row['saved_image_path']}")


def print_overlap_summary(rows: list[dict[str, Any]]) -> None:
    case_rows: dict[tuple[int, int, int, int, int], dict[str, Any]] = {}
    for row in rows:
        key = (
            int(row["height"]),
            int(row["width"]),
            int(row["splat_count"]),
            int(row["set_index"]),
            int(row["seed"]),
        )
        if key not in case_rows and any(k.startswith("overlap_") for k in row):
            case_rows[key] = row

    if not case_rows:
        return

    overlap_rows = list(case_rows.values())
    variants = sorted(
        {
            key[len("overlap_") : -len("_total_overlap_keys")]
            for row in overlap_rows
            for key in row
            if key.startswith("overlap_") and key.endswith("_total_overlap_keys")
        }
    )
    if not variants:
        return

    group_key_fn = lambda row: (int(row["height"]), int(row["width"]), int(row["splat_count"]))
    print("\nSummary overlap-key pressure by resolution/splats:")
    for height, width, splat_count in sorted({group_key_fn(row) for row in overlap_rows}):
        group = [row for row in overlap_rows if group_key_fn(row) == (height, width, splat_count)]
        for variant in variants:
            total_key = f"overlap_{variant}_total_overlap_keys"
            dup_key = f"overlap_{variant}_duplication_factor"
            splat_p95_key = f"overlap_{variant}_p95_tiles_per_splat"
            splat_max_key = f"overlap_{variant}_max_tiles_per_splat"
            tile_p95_key = f"overlap_{variant}_p95_splats_per_tile"
            tile_max_key = f"overlap_{variant}_max_splats_per_tile"
            usable = [row for row in group if total_key in row]
            if not usable:
                continue
            mean_total = sum(float(row[total_key]) for row in usable) / len(usable)
            mean_dup = sum(float(row[dup_key]) for row in usable) / len(usable)
            mean_splat_p95 = sum(float(row[splat_p95_key]) for row in usable) / len(usable)
            max_splat = max(float(row[splat_max_key]) for row in usable)
            mean_tile_p95 = sum(float(row[tile_p95_key]) for row in usable) / len(usable)
            max_tile = max(float(row[tile_max_key]) for row in usable)
            print(
                f"{variant:<18} {height:>4}x{width:<4} G={splat_count:<6} "
                f"K={mean_total:>10.0f} K/G={mean_dup:>7.3f} "
                f"splat_tiles p95/max={mean_splat_p95:>5.1f}/{max_splat:>5.0f} "
                f"tile_splats p95/max={mean_tile_p95:>5.1f}/{max_tile:>5.0f}"
            )

        exact_key = "overlap_exact_conic_total_overlap_keys"
        taichi_key = "overlap_taichi_obb_total_overlap_keys"
        custom_key = "overlap_custom_rect_total_overlap_keys"
        custom_exact_key = (
            "overlap_exact_conic_custom_tile_total_overlap_keys"
            if any("overlap_exact_conic_custom_tile_total_overlap_keys" in row for row in group)
            else exact_key
        )
        if all(exact_key in row and taichi_key in row for row in group):
            exact_total = sum(float(row[exact_key]) for row in group)
            taichi_total = sum(float(row[taichi_key]) for row in group)
            if exact_total > 0:
                print(
                    f"{'taichi_obb/exact':<18} {height:>4}x{width:<4} G={splat_count:<6} "
                    f"ratio={taichi_total / exact_total:>7.3f}x"
                )
        if all(custom_exact_key in row and custom_key in row for row in group):
            exact_total = sum(float(row[custom_exact_key]) for row in group)
            custom_total = sum(float(row[custom_key]) for row in group)
            if exact_total > 0:
                tile_size_key = custom_exact_key.replace("total_overlap_keys", "tile_size")
                tile_size = group[0].get(tile_size_key, math.nan)
                print(
                    f"{'custom_rect/exact':<18} {height:>4}x{width:<4} G={splat_count:<6} "
                    f"tile={tile_size:>4.0f} ratio={custom_total / exact_total:>7.3f}x"
                )


def run_benchmark(
    config: dict[str, Any], jsonl_path: Path | None, csv_path: Path | None, fail_fast: bool
) -> list[dict[str, Any]]:
    device = pick_device(str(config["device"]))
    dtype = dtype_from_name(str(config["dtype"]))
    resolutions = [normalize_resolution(value) for value in config["resolutions"]]
    splat_counts = [int(value) for value in config["splat_counts"]]
    requested_renderers = [str(name) for name in config["renderers"]]
    renderers = build_renderers(device, config, requested_renderers)
    unknown_renderers = [name for name in requested_renderers if name not in renderers]
    if unknown_renderers:
        raise ValueError(f"Unknown renderer(s): {', '.join(unknown_renderers)}")

    print(f"device={device} dtype={dtype} backward={config['backward']}")
    print(f"resolutions={resolutions} splat_counts={splat_counts} sets_per_case={config['sets_per_case']}")
    print("")

    rows: list[dict[str, Any]] = []
    base_seed = int(config["seed"])
    save_config = config.get("save_images", {})
    save_images = bool(save_config.get("enabled", False))
    image_directory = resolve_output_path(save_config.get("directory", "benchmark_outputs/splat_renderer_images"))
    target_resolutions, target_splat_counts, target_set_index = image_save_target(config, resolutions, splat_counts)
    for height, width in resolutions:
        for splat_count in splat_counts:
            for set_index in range(int(config["sets_per_case"])):
                case_seed = base_seed + height * 1_000_003 + width * 10_007 + splat_count * 101 + set_index
                base_case = make_random_case(
                    height=height,
                    width=width,
                    splat_count=splat_count,
                    set_index=set_index,
                    seed=case_seed,
                    device=device,
                    dtype=dtype,
                    cfg=config["random_splats"],
                )
                overlap_metrics = compute_case_overlap_stats(base_case, config)
                if bool(config.get("overlap_stats", {}).get("print_per_case", False)):
                    print_overlap_case(overlap_metrics, height, width, splat_count, set_index)
                reference_output = None
                case_rows: list[dict[str, Any]] = []
                for renderer_key in requested_renderers:
                    renderer = renderers[renderer_key]
                    row_base = {
                        "renderer": renderer.name,
                        "renderer_key": renderer_key,
                        "height": height,
                        "width": width,
                        "splat_count": splat_count,
                        "set_index": set_index,
                        "seed": case_seed,
                        "device": str(device),
                        "dtype": str(dtype).replace("torch.", ""),
                        "backward": bool(config["backward"]),
                        **overlap_metrics,
                    }
                    try:
                        result = timed_renderer_run(renderer, base_case, config, device)
                        row = {**row_base, **{key: value for key, value in result.items() if key != "renderer"}}
                        row["renderer"] = result["renderer"]
                        if row["status"] == "ok":
                            if reference_output is None:
                                reference_output = result["output"]
                            comparison = (
                                output_comparison(reference_output, result["output"])
                                if config["compare_outputs"]
                                else {}
                            )
                            row.update(comparison)
                            if save_images and should_save_image(
                                row, target_resolutions, target_splat_counts, target_set_index
                            ):
                                saved_path = save_render_image(result["output"], row, image_directory)
                                row["saved_image_path"] = str(saved_path)
                    except Exception as exc:
                        if fail_fast:
                            raise
                        row = {**row_base, "status": "error", "error": repr(exc)}
                    case_rows.append(row)
                    print_row(row)
                rows.extend(case_rows)

    write_outputs(rows, jsonl_path=jsonl_path, csv_path=csv_path)
    print_summary(rows)
    return rows


def load_benchmark_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return deepcopy(DEFAULT_CONFIG)
    return deep_merge(DEFAULT_CONFIG, load_config_file(path))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Dynaworld differentiable splat renderers.")
    parser.add_argument("--config", type=Path, help="Optional JSONC benchmark config.")
    parser.add_argument("--device", type=str, help="Override config device, e.g. cpu, cuda, cuda:0, mps.")
    parser.add_argument("--dtype", type=str, choices=["float32", "float16", "bfloat16"], help="Override config dtype.")
    parser.add_argument("--renderers", type=str, help="Comma-separated renderer keys.")
    parser.add_argument("--resolutions", type=str, help="Comma-separated sizes, e.g. 64,128x96.")
    parser.add_argument("--splat-counts", type=str, help="Comma-separated splat counts.")
    parser.add_argument("--sets-per-case", type=int, help="Random splat sets per resolution/count combination.")
    parser.add_argument("--warmup-iters", type=int, help="Warmup iterations per renderer/set.")
    parser.add_argument("--timed-iters", type=int, help="Timed iterations per renderer/set.")
    parser.add_argument("--forward-only", action="store_true", help="Disable backward timing.")
    parser.add_argument("--jsonl", type=Path, help="Optional path for JSONL results.")
    parser.add_argument("--csv", type=Path, help="Optional path for CSV results.")
    parser.add_argument("--save-images", type=Path, help="Save selected render PNGs to this directory.")
    parser.add_argument("--no-save-images", action="store_true", help="Disable render PNG output.")
    parser.add_argument(
        "--overlap-variants",
        type=str,
        help="Comma-separated overlap stat variants: custom_rect,taichi_obb,exact_conic.",
    )
    parser.add_argument("--no-overlap-stats", action="store_true", help="Disable overlap-key pressure diagnostics.")
    parser.add_argument("--fail-fast", action="store_true", help="Raise immediately on renderer errors.")
    return parser.parse_args(argv)


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = deepcopy(config)
    if args.device is not None:
        cfg["device"] = args.device
    if args.dtype is not None:
        cfg["dtype"] = args.dtype
    if args.renderers is not None:
        cfg["renderers"] = [part.strip() for part in args.renderers.split(",") if part.strip()]
    if args.resolutions is not None:
        cfg["resolutions"] = parse_csv_resolutions(args.resolutions)
    if args.splat_counts is not None:
        cfg["splat_counts"] = parse_csv_ints(args.splat_counts)
    if args.sets_per_case is not None:
        cfg["sets_per_case"] = args.sets_per_case
    if args.warmup_iters is not None:
        cfg["warmup_iters"] = args.warmup_iters
    if args.timed_iters is not None:
        cfg["timed_iters"] = args.timed_iters
    if args.forward_only:
        cfg["backward"] = False
    if args.save_images is not None:
        cfg.setdefault("save_images", {})
        cfg["save_images"]["enabled"] = True
        cfg["save_images"]["directory"] = str(args.save_images)
    if args.no_save_images:
        cfg.setdefault("save_images", {})
        cfg["save_images"]["enabled"] = False
    if args.overlap_variants is not None:
        cfg.setdefault("overlap_stats", {})
        cfg["overlap_stats"]["enabled"] = True
        cfg["overlap_stats"]["variants"] = [part.strip() for part in args.overlap_variants.split(",") if part.strip()]
    if args.no_overlap_stats:
        cfg.setdefault("overlap_stats", {})
        cfg["overlap_stats"]["enabled"] = False
    return cfg


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = apply_cli_overrides(load_benchmark_config(args.config), args)
    run_benchmark(config, jsonl_path=args.jsonl, csv_path=args.csv, fail_fast=args.fail_fast)


if __name__ == "__main__":
    main()
