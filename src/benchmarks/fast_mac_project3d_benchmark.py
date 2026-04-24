from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch


BENCHMARK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCHMARK_DIR.parents[1]
SRC_TRAIN_DIR = PROJECT_ROOT / "src" / "train"
FAST_MAC_DIR = PROJECT_ROOT / "third_party" / "fast-mac-gsplat"
FAST_MAC_V5_DIR = FAST_MAC_DIR / "variants" / "v5"
FAST_MAC_V8_DIR = FAST_MAC_DIR / "variants" / "v8_project3d"

for path in (FAST_MAC_V8_DIR, FAST_MAC_V5_DIR, SRC_TRAIN_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from renderers.fast_mac import project_for_fast_mac_batch
from torch_gsplat_bridge_v5 import RasterConfig as RasterConfigV5
from torch_gsplat_bridge_v5 import rasterize_projected_gaussians as rasterize_v5


DEFAULT_CASES = "smoke:64:512:1,realistic_128_8192:128:8192:1"
DEFAULT_BG = (0.0, 0.0, 0.0)
NEAR_PLANE = 1e-4
RasterConfigV8 = None
rasterize_v8_project3d = None


@dataclass(frozen=True)
class Case:
    name: str
    size: int
    gaussians: int
    batch_size: int


@dataclass(frozen=True)
class Scene:
    means3d: torch.Tensor
    scales: torch.Tensor
    quats: torch.Tensor
    opacities: torch.Tensor
    colors: torch.Tensor
    fx: float | torch.Tensor
    fy: float | torch.Tensor
    cx: float | torch.Tensor
    cy: float | torch.Tensor
    camera_to_world: torch.Tensor


def sync_mps() -> None:
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def parse_cases(raw: str) -> list[Case]:
    cases: list[Case] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 4:
            raise ValueError("cases must use name:size:gaussians:batch, comma-separated")
        name, size, gaussians, batch = parts
        cases.append(Case(name=name, size=int(size), gaussians=int(gaussians), batch_size=int(batch)))
    return cases


def make_scene(case: Case, seed: int) -> Scene:
    device = torch.device("mps")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    b, g, size = case.batch_size, case.gaussians, case.size
    fx = 0.82 * float(size)
    fy = 0.80 * float(size)
    cx = 0.5 * float(size)
    cy = 0.5 * float(size)

    z = torch.empty((b, g), dtype=torch.float32).uniform_(2.0, 7.0, generator=generator)
    x = (torch.rand((b, g), dtype=torch.float32, generator=generator) - 0.5) * 0.95 * z
    y = (torch.rand((b, g), dtype=torch.float32, generator=generator) - 0.5) * 0.95 * z
    means_cpu = torch.stack([x, y, z], dim=-1)

    scales_cpu = torch.empty((b, g, 3), dtype=torch.float32).uniform_(0.012, 0.055, generator=generator)
    quats_cpu = torch.randn((b, g, 4), dtype=torch.float32, generator=generator)
    quats_cpu = quats_cpu / torch.clamp(quats_cpu.norm(dim=-1, keepdim=True), min=1e-6)
    opacities_cpu = torch.empty((b, g, 1), dtype=torch.float32).uniform_(0.04, 0.45, generator=generator)
    colors_cpu = torch.rand((b, g, 3), dtype=torch.float32, generator=generator)
    camera_to_world_cpu = torch.eye(4, dtype=torch.float32).view(1, 4, 4).expand(b, -1, -1).contiguous()

    return Scene(
        means3d=means_cpu.to(device).contiguous(),
        scales=scales_cpu.to(device).contiguous(),
        quats=quats_cpu.to(device).contiguous(),
        opacities=opacities_cpu.to(device).contiguous(),
        colors=colors_cpu.to(device).contiguous(),
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        camera_to_world=camera_to_world_cpu.to(device).contiguous(),
    )


def make_v5_config(case: Case) -> RasterConfigV5:
    return RasterConfigV5(
        height=case.size,
        width=case.size,
        background=DEFAULT_BG,
        batch_strategy="auto",
        enable_overflow_fallback=True,
    )


def make_v8_config(case: Case) -> RasterConfigV8:
    if RasterConfigV8 is None:
        raise RuntimeError("v8_project3d package has not been loaded")
    return RasterConfigV8(
        height=case.size,
        width=case.size,
        background=DEFAULT_BG,
        batch_strategy="auto",
        enable_overflow_fallback=True,
    )


def render_current_v5(scene: Scene, config: RasterConfigV5) -> torch.Tensor:
    means2d, conics, colors, opacities, depths = project_for_fast_mac_batch(
        scene.means3d,
        scene.scales,
        scene.quats,
        scene.opacities,
        scene.colors,
        scene.fx,
        scene.fy,
        scene.cx,
        scene.cy,
        camera_to_world=scene.camera_to_world,
        near_plane=NEAR_PLANE,
    )
    return rasterize_v5(means2d, conics, colors, opacities, depths, config)


def render_project3d_v8(scene: Scene, config: RasterConfigV8) -> torch.Tensor:
    if rasterize_v8_project3d is None:
        raise RuntimeError("v8_project3d package has not been loaded")
    return rasterize_v8_project3d(
        scene.means3d,
        scene.scales,
        scene.quats,
        scene.opacities,
        scene.colors,
        scene.fx,
        scene.fy,
        scene.cx,
        scene.cy,
        camera_to_world=scene.camera_to_world,
        near_plane=NEAR_PLANE,
        config=config,
    )


def measure_ms(fn, warmup: int, iters: int) -> tuple[float, float, float, torch.Tensor]:
    last = None
    with torch.no_grad():
        for _ in range(warmup):
            last = fn()
            sync_mps()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            last = fn()
            sync_mps()
            times.append((time.perf_counter() - t0) * 1000.0)
    assert last is not None
    return statistics.mean(times), min(times), max(times), last


def _scalar_leaf(value: float | torch.Tensor, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(value):
        value = float(value.detach().mean().cpu().item())
    return torch.tensor(float(value), device=device, dtype=torch.float32, requires_grad=True)


def make_train_scene(scene: Scene) -> Scene:
    device = scene.means3d.device
    return replace(
        scene,
        means3d=scene.means3d.detach().clone().requires_grad_(True),
        scales=scene.scales.detach().clone().requires_grad_(True),
        quats=scene.quats.detach().clone().requires_grad_(True),
        opacities=scene.opacities.detach().clone().requires_grad_(True),
        colors=scene.colors.detach().clone().requires_grad_(True),
        fx=_scalar_leaf(scene.fx, device),
        fy=_scalar_leaf(scene.fy, device),
        cx=_scalar_leaf(scene.cx, device),
        cy=_scalar_leaf(scene.cy, device),
        camera_to_world=scene.camera_to_world.detach().clone().requires_grad_(True),
    )


def measure_forward_backward_ms(render_fn, scene: Scene, config, warmup: int, iters: int) -> tuple[float, float, float, float]:
    last_loss = float("nan")
    for _ in range(warmup):
        train_scene = make_train_scene(scene)
        loss = render_fn(train_scene, config).sum()
        loss.backward()
        sync_mps()
        last_loss = float(loss.detach().cpu().item())

    times = []
    for _ in range(iters):
        train_scene = make_train_scene(scene)
        t0 = time.perf_counter()
        loss = render_fn(train_scene, config).sum()
        loss.backward()
        sync_mps()
        times.append((time.perf_counter() - t0) * 1000.0)
        last_loss = float(loss.detach().cpu().item())
    return statistics.mean(times), min(times), max(times), last_loss


def _scene_grads(scene: Scene) -> dict[str, torch.Tensor]:
    values = {
        "means3d": scene.means3d.grad,
        "scales": scene.scales.grad,
        "quats": scene.quats.grad,
        "opacities": scene.opacities.grad,
        "colors": scene.colors.grad,
        "camera_to_world": scene.camera_to_world.grad,
        "fx": scene.fx.grad if torch.is_tensor(scene.fx) else None,
        "fy": scene.fy.grad if torch.is_tensor(scene.fy) else None,
        "cx": scene.cx.grad if torch.is_tensor(scene.cx) else None,
        "cy": scene.cy.grad if torch.is_tensor(scene.cy) else None,
    }
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise RuntimeError(f"Missing gradients for: {', '.join(missing)}")
    return {name: value.detach().reshape(-1) for name, value in values.items()}


def full_grad_check(scene: Scene, v5_config: RasterConfigV5, v8_config: RasterConfigV8) -> tuple[float, float]:
    ref_scene = make_train_scene(scene)
    v8_scene = make_train_scene(scene)

    ref_loss = render_current_v5(ref_scene, v5_config).sum()
    ref_loss.backward()
    sync_mps()

    v8_loss = render_project3d_v8(v8_scene, v8_config).sum()
    v8_loss.backward()
    sync_mps()

    diffs = []
    for name, ref_grad in _scene_grads(ref_scene).items():
        v8_grad = _scene_grads(v8_scene)[name]
        diffs.append((ref_grad - v8_grad).abs())
    all_diffs = torch.cat(diffs, dim=0)
    return float(all_diffs.max().item()), float(all_diffs.mean().item())


def maybe_build_v8() -> None:
    subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=FAST_MAC_V8_DIR, check=True)


def load_v8() -> None:
    global RasterConfigV8, rasterize_v8_project3d
    from torch_gsplat_bridge_v8_project3d import RasterConfig as _RasterConfigV8
    from torch_gsplat_bridge_v8_project3d import rasterize_pinhole_gaussians as _rasterize_v8_project3d

    RasterConfigV8 = _RasterConfigV8
    rasterize_v8_project3d = _rasterize_v8_project3d


def ensure_runtime_ready() -> None:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available; this benchmark only runs on Apple Silicon/MPS.")
    if not hasattr(torch.ops.gsplat_metal_v8_project3d, "project_pinhole_forward"):
        raise RuntimeError(
            "v8_project3d extension is not loaded. Build it with:\n"
            f"  cd {FAST_MAC_V8_DIR}\n"
            "  python setup.py build_ext --inplace\n"
            "or rerun this benchmark with --build-v8."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare v5 Torch projection+raster vs v8 Metal projection+raster.")
    parser.add_argument("--cases", default=DEFAULT_CASES, help="Comma-separated name:size:gaussians:batch cases.")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--build-v8", action="store_true", help="Build the v8 extension before running.")
    parser.add_argument("--skip-grad-check", action="store_true", help="Skip full gradient parity on the first case.")
    args = parser.parse_args()

    if args.build_v8:
        maybe_build_v8()
    load_v8()
    ensure_runtime_ready()

    cases = parse_cases(args.cases)
    print(
        "case,size,gaussians,batch,phase,path,mean_ms,min_ms,max_ms,"
        "max_abs_err,mean_abs_err,grad_max_err,grad_mean_err"
    )
    for i, case in enumerate(cases):
        scene = make_scene(case, args.seed + i)
        v5_config = make_v5_config(case)
        v8_config = make_v8_config(case)

        v5_mean, v5_min, v5_max, img_v5 = measure_ms(lambda: render_current_v5(scene, v5_config), args.warmup, args.iters)
        v8_mean, v8_min, v8_max, img_v8 = measure_ms(lambda: render_project3d_v8(scene, v8_config), args.warmup, args.iters)

        err = (img_v5 - img_v8).abs()
        grad_max = float("nan")
        grad_mean = float("nan")
        if i == 0 and not args.skip_grad_check:
            grad_max, grad_mean = full_grad_check(scene, v5_config, v8_config)

        max_err = float(err.max().item())
        mean_err = float(err.mean().item())
        v5_train_mean, v5_train_min, v5_train_max, _v5_loss = measure_forward_backward_ms(
            render_current_v5,
            scene,
            v5_config,
            args.warmup,
            args.iters,
        )
        v8_train_mean, v8_train_min, v8_train_max, _v8_loss = measure_forward_backward_ms(
            render_project3d_v8,
            scene,
            v8_config,
            args.warmup,
            args.iters,
        )

        print(
            f"{case.name},{case.size},{case.gaussians},{case.batch_size},"
            f"forward_eval,v5_torch_project_plus_metal,{v5_mean:.4f},{v5_min:.4f},{v5_max:.4f},"
            f"{max_err:.6g},{mean_err:.6g},{grad_max:.6g},{grad_mean:.6g}"
        )
        print(
            f"{case.name},{case.size},{case.gaussians},{case.batch_size},"
            f"forward_eval,v8_metal_project_plus_metal,{v8_mean:.4f},{v8_min:.4f},{v8_max:.4f},"
            f"{max_err:.6g},{mean_err:.6g},{grad_max:.6g},{grad_mean:.6g}"
        )
        print(
            f"{case.name},{case.size},{case.gaussians},{case.batch_size},"
            f"forward_backward,v5_torch_project_plus_metal,{v5_train_mean:.4f},{v5_train_min:.4f},"
            f"{v5_train_max:.4f},{max_err:.6g},{mean_err:.6g},{grad_max:.6g},{grad_mean:.6g}"
        )
        print(
            f"{case.name},{case.size},{case.gaussians},{case.batch_size},"
            f"forward_backward,v8_metal_project_plus_metal,{v8_train_mean:.4f},{v8_train_min:.4f},"
            f"{v8_train_max:.4f},{max_err:.6g},{mean_err:.6g},{grad_max:.6g},{grad_mean:.6g}"
        )


if __name__ == "__main__":
    main()
