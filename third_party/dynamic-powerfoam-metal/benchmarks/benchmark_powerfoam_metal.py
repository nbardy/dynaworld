from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
DYNAWORLD = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_dynamic_powerfoam_metal import (
    FoamRasterConfig,
    rasterize_power_foam,
    rasterize_power_foam_linear,
    rasterize_power_foam_oriented_surface_linear,
    rasterize_power_foam_oriented_texel_surface,
    rasterize_power_foam_surface_linear,
)
from torch_dynamic_powerfoam_metal.random_scene import (
    make_adjacency,
    make_pinhole_rays,
    make_power_sorted_ids,
    make_random_foam,
)


def parse_int_list(raw: str) -> list[int]:
    return [int(v) for v in raw.split(",") if v.strip()]


def parse_resolutions(raw: str) -> list[tuple[int, int]]:
    out = []
    for part in raw.split(","):
        if not part.strip():
            continue
        w, h = part.lower().split("x", maxsplit=1)
        out.append((int(h), int(w)))
    return out


def sync() -> None:
    torch.mps.synchronize()


def measure(step: Callable[[], tuple[float, float]], *, warmup: int, iters: int) -> dict[str, float]:
    for _ in range(warmup):
        step()
    forward_ms = []
    backward_ms = []
    total_ms = []
    for _ in range(iters):
        f_ms, b_ms = step()
        forward_ms.append(f_ms)
        backward_ms.append(b_ms)
        total_ms.append(f_ms + b_ms)
    return {
        "forward_median_ms": float(statistics.median(forward_ms)),
        "backward_median_ms": float(statistics.median(backward_ms)),
        "total_median_ms": float(statistics.median(total_ms)),
        "forward_mean_ms": float(sum(forward_ms) / len(forward_ms)),
        "backward_mean_ms": float(sum(backward_ms) / len(backward_ms)),
        "total_mean_ms": float(sum(total_ms) / len(total_ms)),
    }


def make_gs_case(
    *,
    cells: int,
    feature_dim: int,
    height: int,
    width: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device).manual_seed(seed)
    means2d = torch.rand((1, cells, 2), device=device, dtype=torch.float32, generator=gen)
    means2d[..., 0] *= width
    means2d[..., 1] *= height
    sig = torch.rand((1, cells, 2), device=device, dtype=torch.float32, generator=gen) * 5.0 + 3.0
    conics = torch.stack(
        [
            1.0 / sig[..., 0].square().clamp_min(1.0e-8),
            torch.zeros((1, cells), device=device, dtype=torch.float32),
            1.0 / sig[..., 1].square().clamp_min(1.0e-8),
        ],
        dim=-1,
    ).contiguous()
    colors = torch.rand((1, cells, feature_dim), device=device, dtype=torch.float32, generator=gen)
    opacities = torch.rand((1, cells), device=device, dtype=torch.float32, generator=gen).mul_(0.7).add_(0.1)
    depths = torch.rand((1, cells), device=device, dtype=torch.float32, generator=gen)
    return means2d, conics, colors, opacities, depths


def import_gs_feature_renderer():
    gs_root = DYNAWORLD / "third_party" / "fast-mac-gsplat" / "variants" / "v5_features"
    if str(gs_root) not in sys.path:
        sys.path.insert(0, str(gs_root))
    from torch_gsplat_bridge_v5_features import RasterConfig, get_runtime_shader_config, rasterize_projected_gaussians

    return RasterConfig, get_runtime_shader_config, rasterize_projected_gaussians


def benchmark_foam(
    *,
    cells: int,
    feature_dim: int,
    height: int,
    width: int,
    device: torch.device,
    seed: int,
    adjacency_mode: str,
    neighbors: int,
    warmup: int,
    iters: int,
    backward: bool,
    linear_features: bool,
    surface_features: bool,
    oriented_surface_features: bool,
    texel_surface_features: bool,
) -> dict[str, float | int | str | bool | None]:
    points, radii, densities, features = make_random_foam(
        cell_count=cells,
        feature_dim=feature_dim,
        device=device,
        seed=seed,
    )
    texel_sites = torch.empty((cells, 0, 2), device=device, dtype=torch.float32)
    if texel_surface_features:
        gen = torch.Generator(device=device).manual_seed(seed + 1009)
        texel_count = 4
        texel_sites = torch.rand((cells, texel_count, 2), device=device, dtype=torch.float32, generator=gen) - 0.5
        features = torch.rand((cells, texel_count, feature_dim), device=device, dtype=torch.float32, generator=gen)
    elif linear_features or surface_features or oriented_surface_features:
        gen = torch.Generator(device=device).manual_seed(seed + 1009)
        coeffs = 0.1 * torch.randn((cells, feature_dim, 3), device=device, dtype=torch.float32, generator=gen)
        features = torch.cat([features.unsqueeze(-1), coeffs], dim=-1).contiguous()
    normals = torch.zeros((cells, 3), device=device, dtype=torch.float32)
    normals[:, 2] = -1.0
    if oriented_surface_features or texel_surface_features:
        gen = torch.Generator(device=device).manual_seed(seed + 2029)
        normals = normals + 0.05 * torch.randn((cells, 3), device=device, dtype=torch.float32, generator=gen)
        normals = torch.nn.functional.normalize(normals, dim=-1, eps=1.0e-6).contiguous()
    if backward:
        points.requires_grad_(True)
        radii.requires_grad_(True)
        densities.requires_grad_(True)
        features.requires_grad_(True)
        if texel_surface_features:
            texel_sites.requires_grad_(True)
            normals.requires_grad_(True)
        elif oriented_surface_features:
            normals.requires_grad_(True)
    adjacency, offsets = make_adjacency(points, radii, mode=adjacency_mode, neighbors=neighbors)
    rays = make_pinhole_rays(batch_size=1, height=height, width=width, device=device)
    sorted_ids = make_power_sorted_ids(points, radii, rays)
    config = FoamRasterConfig(alpha_threshold=0.0)

    def step() -> tuple[float, float]:
        if backward:
            t0 = time.perf_counter()
            if texel_surface_features:
                rasterize = rasterize_power_foam_oriented_texel_surface
            elif oriented_surface_features:
                rasterize = rasterize_power_foam_oriented_surface_linear
            elif surface_features:
                rasterize = rasterize_power_foam_surface_linear
            elif linear_features:
                rasterize = rasterize_power_foam_linear
            else:
                rasterize = rasterize_power_foam
            if texel_surface_features:
                out, alpha = rasterize(
                    points,
                    radii,
                    densities,
                    texel_sites,
                    features,
                    normals,
                    adjacency,
                    offsets,
                    rays,
                    config,
                    sorted_ids=sorted_ids,
                )
            elif oriented_surface_features:
                out, alpha = rasterize(
                    points,
                    radii,
                    densities,
                    features,
                    normals,
                    adjacency,
                    offsets,
                    rays,
                    config,
                    sorted_ids=sorted_ids,
                )
            else:
                out, alpha = rasterize(
                    points,
                    radii,
                    densities,
                    features,
                    adjacency,
                    offsets,
                    rays,
                    config,
                    sorted_ids=sorted_ids,
                )
            sync()
            t1 = time.perf_counter()
            (out.square().mean() + alpha.square().mean()).backward()
            sync()
            t2 = time.perf_counter()
            if texel_surface_features:
                grad_tensors = (points, radii, densities, features, normals, texel_sites)
            elif oriented_surface_features:
                grad_tensors = (points, radii, densities, features, normals)
            else:
                grad_tensors = (points, radii, densities, features)
            for tensor in grad_tensors:
                if tensor.grad is not None:
                    tensor.grad.zero_()
            return (t1 - t0) * 1000.0, (t2 - t1) * 1000.0
        with torch.no_grad():
            t0 = time.perf_counter()
            if texel_surface_features:
                rasterize = rasterize_power_foam_oriented_texel_surface
            elif oriented_surface_features:
                rasterize = rasterize_power_foam_oriented_surface_linear
            elif surface_features:
                rasterize = rasterize_power_foam_surface_linear
            elif linear_features:
                rasterize = rasterize_power_foam_linear
            else:
                rasterize = rasterize_power_foam
            if texel_surface_features:
                _features, _alpha = rasterize(
                    points,
                    radii,
                    densities,
                    texel_sites,
                    features,
                    normals,
                    adjacency,
                    offsets,
                    rays,
                    config,
                    sorted_ids=sorted_ids,
                )
            elif oriented_surface_features:
                _features, _alpha = rasterize(
                    points,
                    radii,
                    densities,
                    features,
                    normals,
                    adjacency,
                    offsets,
                    rays,
                    config,
                    sorted_ids=sorted_ids,
                )
            else:
                _features, _alpha = rasterize(
                    points,
                    radii,
                    densities,
                    features,
                    adjacency,
                    offsets,
                    rays,
                    config,
                    sorted_ids=sorted_ids,
                )
            sync()
            t1 = time.perf_counter()
        return (t1 - t0) * 1000.0, 0.0

    result = measure(step, warmup=warmup, iters=iters)
    avg_degree = float(adjacency.numel()) / float(cells) if cells else 0.0
    result.update(
        {
            "renderer": "dynamic_powerfoam_metal",
            "cells": cells,
            "feature_dim": feature_dim,
            "feature_mode": (
                "oriented_texel_surface"
                if texel_surface_features
                else (
                    "oriented_surface_linear"
                    if oriented_surface_features
                    else ("surface_linear" if surface_features else ("linear" if linear_features else "constant"))
                )
            ),
            "height": height,
            "width": width,
            "adjacency": adjacency_mode,
            "avg_degree": avg_degree,
            "backward_supported": True,
            "backward": bool(backward),
        }
    )
    return result


def benchmark_gs(
    *,
    cells: int,
    feature_dim: int,
    height: int,
    width: int,
    device: torch.device,
    seed: int,
    warmup: int,
    iters: int,
    backward: bool,
) -> dict[str, float | int | str | bool]:
    RasterConfig, get_runtime_shader_config, rasterize_projected_gaussians = import_gs_feature_renderer()
    means2d, conics, colors, opacities, depths = make_gs_case(
        cells=cells,
        feature_dim=feature_dim,
        height=height,
        width=width,
        device=device,
        seed=seed,
    )
    if backward:
        means2d.requires_grad_(True)
        conics.requires_grad_(True)
        colors.requires_grad_(True)
        opacities.requires_grad_(True)

    rt = get_runtime_shader_config()
    config = RasterConfig(height=height, width=width, tile_size=rt.tile_size, max_fast_pairs=rt.fast_cap)

    def step() -> tuple[float, float]:
        if backward:
            t0 = time.perf_counter()
            out, alpha = rasterize_projected_gaussians(means2d, conics, colors, opacities, depths, config)
            sync()
            t1 = time.perf_counter()
            (out.square().mean() + alpha.square().mean()).backward()
            sync()
            t2 = time.perf_counter()
            for tensor in (means2d, conics, colors, opacities):
                if tensor.grad is not None:
                    tensor.grad.zero_()
            return (t1 - t0) * 1000.0, (t2 - t1) * 1000.0
        with torch.no_grad():
            t0 = time.perf_counter()
            _out, _alpha = rasterize_projected_gaussians(means2d, conics, colors, opacities, depths, config)
            sync()
            t1 = time.perf_counter()
        return (t1 - t0) * 1000.0, 0.0

    result = measure(step, warmup=warmup, iters=iters)
    result.update(
        {
            "renderer": "gsplat_v5_features",
            "cells": cells,
            "feature_dim": feature_dim,
            "height": height,
            "width": width,
            "backward_supported": True,
            "backward": bool(backward),
            "runtime_tile_size": rt.tile_size,
            "runtime_fast_cap": rt.fast_cap,
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", default="256,1024", help="comma-separated cell/splat counts")
    parser.add_argument("--resolutions", default="128x128,256x256", help="comma-separated WxH values")
    parser.add_argument("--feature-dim", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--neighbors", type=int, default=32)
    parser.add_argument("--adjacency", choices=["overlap", "knn"], default="knn")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--compare-gs", action="store_true")
    parser.add_argument("--foam-backward", action="store_true")
    parser.add_argument("--foam-linear", action="store_true")
    parser.add_argument("--foam-surface", action="store_true")
    parser.add_argument("--foam-oriented-surface", action="store_true")
    parser.add_argument("--foam-texel-surface", action="store_true")
    parser.add_argument("--gs-backward", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    mode_flags = int(args.foam_linear) + int(args.foam_surface) + int(args.foam_oriented_surface) + int(args.foam_texel_surface)
    if mode_flags > 1:
        raise SystemExit(
            "--foam-linear, --foam-surface, --foam-oriented-surface, and --foam-texel-surface are mutually exclusive"
        )

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for the DynamicPowerFoam Metal benchmark")
    device = torch.device("mps")
    results = []
    for height, width in parse_resolutions(args.resolutions):
        for cells in parse_int_list(args.cells):
            foam = benchmark_foam(
                cells=cells,
                feature_dim=args.feature_dim,
                height=height,
                width=width,
                device=device,
                seed=args.seed,
                adjacency_mode=args.adjacency,
                neighbors=args.neighbors,
                warmup=args.warmup,
                iters=args.iters,
                backward=args.foam_backward,
                linear_features=args.foam_linear,
                surface_features=args.foam_surface,
                oriented_surface_features=args.foam_oriented_surface,
                texel_surface_features=args.foam_texel_surface,
            )
            results.append(foam)
            if args.compare_gs:
                results.append(
                    benchmark_gs(
                        cells=cells,
                        feature_dim=args.feature_dim,
                        height=height,
                        width=width,
                        device=device,
                        seed=args.seed,
                        warmup=args.warmup,
                        iters=args.iters,
                        backward=args.gs_backward,
                    )
                )

    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
        return
    for row in results:
        print(
            f"{row['renderer']:>18} {row['width']}x{row['height']} N={row['cells']} F={row['feature_dim']} "
            f"fwd_med={row['forward_median_ms']:.3f}ms bwd_med={row['backward_median_ms']:.3f}ms "
            f"total_med={row['total_median_ms']:.3f}ms"
        )
        if row["renderer"] == "dynamic_powerfoam_metal":
            print(
                f"{'':>18} adjacency={row['adjacency']} avg_degree={row['avg_degree']:.2f} "
                f"feature_mode={row['feature_mode']} backward_supported={row['backward_supported']}"
            )


if __name__ == "__main__":
    main()
