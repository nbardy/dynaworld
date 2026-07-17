from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

try:
    from .report_artifacts import (
        POWERFOAM_METAL_ROOT,
        PROJECT_ROOT,
        ensure_sys_path,
        ensure_train_path,
        load_report_json,
        write_report_json,
    )
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import (
        POWERFOAM_METAL_ROOT,
        PROJECT_ROOT,
        ensure_sys_path,
        ensure_train_path,
        load_report_json,
        write_report_json,
    )


ROOT = PROJECT_ROOT
ensure_sys_path(POWERFOAM_METAL_ROOT)
ensure_train_path()

from torch_powerfoam_metal import FoamRasterConfig, raytrace_power_foam_oriented_height_sv_texel_surface  # noqa: E402
from torch_powerfoam_metal.random_scene import make_adjacency, make_pinhole_rays, make_random_foam  # noqa: E402
from train_devices import sync_torch_device  # noqa: E402


DEFAULT_ARTIFACT = (
    ROOT
    / "outputs/benchmarks/"
    "powerfoam_metal_height_sv_texel_surface_raytrace_cech_aabb_4k_trainability_1024cells_2026-05-05.json"
)


def make_trainability_scene(
    *,
    cells: int,
    height: int,
    width: int,
    seed: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    points, radii, _densities, _features = make_random_foam(
        cell_count=cells,
        feature_dim=3,
        device=device,
        seed=seed,
        depth_min=1.8,
        depth_max=3.2,
    )
    radii = radii.mul(2.5).clamp_min(0.35)
    densities = torch.full((cells,), 0.35, dtype=torch.float32, device=device, requires_grad=True)
    gen = torch.Generator(device=device).manual_seed(seed + 1009)
    texel_sites = torch.rand((cells, 4, 2), device=device, dtype=torch.float32, generator=gen) - 0.5
    texel_heights = torch.zeros((cells, 4), device=device, dtype=torch.float32)
    texel_sv_axis = torch.zeros((cells, 4, 3, 3), device=device, dtype=torch.float32)
    texel_sv_axis[..., 2] = 1.0
    texel_sv_rgb = torch.zeros((cells, 4, 3, 3), device=device, dtype=torch.float32)
    normals = torch.zeros((cells, 3), device=device, dtype=torch.float32)
    normals[:, 2] = -1.0
    adjacency, offsets = make_adjacency(points, radii, mode="cech_aabb", neighbors=16)
    rays = make_pinhole_rays(batch_size=1, height=height, width=width, device=device, fov_degrees=55.0)
    return {
        "points": points,
        "radii": radii,
        "densities": densities,
        "texel_sites": texel_sites,
        "texel_heights": texel_heights,
        "texel_sv_axis": texel_sv_axis,
        "texel_sv_rgb": texel_sv_rgb,
        "normals": normals,
        "adjacency": adjacency,
        "offsets": offsets,
        "rays": rays,
    }


def trainability_loss(scene: dict[str, torch.Tensor], config: FoamRasterConfig) -> torch.Tensor:
    _features, alpha = raytrace_power_foam_oriented_height_sv_texel_surface(
        scene["points"],
        scene["radii"],
        scene["densities"],
        scene["texel_sites"],
        scene["texel_heights"],
        scene["texel_sv_axis"],
        scene["texel_sv_rgb"],
        scene["normals"],
        scene["adjacency"],
        scene["offsets"],
        scene["rays"],
        config,
    )
    return alpha.square().mean()


def generate_artifact(
    *,
    cells: int,
    height: int,
    width: int,
    seed: int,
    lr: float,
) -> dict[str, Any]:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for the PowerFoam Metal 4K trainability check.")
    device = torch.device("mps")
    scene = make_trainability_scene(cells=cells, height=height, width=width, seed=seed, device=device)
    config = FoamRasterConfig(alpha_threshold=0.0, transmittance_threshold=0.0, near_plane=0.0)
    optimizer = torch.optim.SGD([scene["densities"]], lr=lr)

    with torch.no_grad():
        before = trainability_loss(scene, config)
        sync_torch_device(device)
        loss_before = float(before.detach().cpu())

    density_before = scene["densities"].detach().clone()
    t0 = time.perf_counter()
    loss = trainability_loss(scene, config)
    sync_torch_device(device)
    t1 = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    sync_torch_device(device)
    t2 = time.perf_counter()
    grad = scene["densities"].grad.detach().clone()
    optimizer.step()
    with torch.no_grad():
        scene["densities"].clamp_(min=0.0)
    density_update = (scene["densities"].detach() - density_before).abs()
    with torch.no_grad():
        after = trainability_loss(scene, config)
        sync_torch_device(device)
        loss_after = float(after.detach().cpu())
    t3 = time.perf_counter()

    return {
        "renderer": "powerfoam_metal",
        "feature_mode": "oriented_height_sv_texel_surface",
        "adjacency": "cech_aabb",
        "raytrace": True,
        "backward": True,
        "optimizer": "sgd_density_only",
        "optimizer_step": True,
        "cells": int(cells),
        "width": int(width),
        "height": int(height),
        "seed": int(seed),
        "lr": float(lr),
        "loss_before": loss_before,
        "loss_after": loss_after,
        "loss_decreased": bool(loss_after < loss_before),
        "loss_ratio": float(loss_after / loss_before) if loss_before > 0.0 else None,
        "grad_abs_max": float(grad.abs().max().detach().cpu()),
        "grad_abs_mean": float(grad.abs().mean().detach().cpu()),
        "density_update_abs_max": float(density_update.max().detach().cpu()),
        "density_update_abs_mean": float(density_update.mean().detach().cpu()),
        "forward_ms": (t1 - t0) * 1000.0,
        "backward_ms": (t2 - t1) * 1000.0,
        "after_forward_ms": (t3 - t2) * 1000.0,
    }


def verify_artifact(path: Path) -> dict[str, Any]:
    data = load_report_json(path)
    checks = [
        ("renderer", data.get("renderer") == "powerfoam_metal"),
        ("feature_mode", data.get("feature_mode") == "oriented_height_sv_texel_surface"),
        ("adjacency", data.get("adjacency") == "cech_aabb"),
        ("raytrace", bool(data.get("raytrace"))),
        ("backward", bool(data.get("backward"))),
        ("optimizer_step", bool(data.get("optimizer_step"))),
        ("uhd_resolution", int(data.get("width", 0)) == 3840 and int(data.get("height", 0)) == 2160),
        ("minimum_cells", int(data.get("cells", 0)) >= 1024),
        ("finite_loss_before", float(data.get("loss_before", float("nan"))) > 0.0),
        ("loss_decreased", bool(data.get("loss_decreased"))),
        ("nonzero_gradient", float(data.get("grad_abs_max", 0.0)) > 0.0),
        ("nonzero_update", float(data.get("density_update_abs_max", 0.0)) > 0.0),
    ]
    return {
        "ok": all(passed for _name, passed in checks),
        "artifact": str(path.relative_to(ROOT)),
        "checks": [{"name": name, "passed": bool(passed)} for name, passed in checks],
        "metrics": data,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify or generate a 4K PowerFoam Metal trainability artifact.")
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--cells", type=int, default=1024)
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--seed", type=int, default=917)
    parser.add_argument("--lr", type=float, default=0.25)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    artifact = args.artifact
    if args.generate:
        data = generate_artifact(
            cells=int(args.cells),
            height=int(args.height),
            width=int(args.width),
            seed=int(args.seed),
            lr=float(args.lr),
        )
        write_report_json(artifact, data)
    report = verify_artifact(artifact)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ok"] and not args.allow_incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
