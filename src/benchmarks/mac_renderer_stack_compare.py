from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

BENCHMARK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCHMARK_DIR.parents[1]
FAST_MAC_DIR = PROJECT_ROOT / "third_party" / "fast-mac-gsplat"
FAST_MAC_V3_DIR = FAST_MAC_DIR / "variants" / "v3"
FAST_MAC_V5_DIR = FAST_MAC_DIR / "variants" / "v5"
TAICHI_SPLATTING_DIR = PROJECT_ROOT / "third_party" / "taichi-splatting"

for path in (FAST_MAC_V5_DIR, FAST_MAC_V3_DIR, FAST_MAC_DIR, TAICHI_SPLATTING_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from torch_gsplat_bridge_fast import RasterConfig as RasterConfigV2
from torch_gsplat_bridge_fast import rasterize_projected_gaussians as rasterize_v2
from torch_gsplat_bridge_v3 import RasterConfig as RasterConfigV3
from torch_gsplat_bridge_v3 import rasterize_projected_gaussians as rasterize_v3
from torch_gsplat_bridge_v5 import RasterConfig as RasterConfigV5
from torch_gsplat_bridge_v5 import rasterize_projected_gaussians as rasterize_v5


DEFAULT_BG = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class Case:
    name: str
    sigma_min: float
    sigma_max: float


@dataclass(frozen=True)
class BatchInputs:
    means: torch.Tensor
    conics: torch.Tensor
    colors: torch.Tensor
    opacities: torch.Tensor
    depths: torch.Tensor
    taichi_packed: torch.Tensor
    taichi_depths: torch.Tensor
    taichi_features: torch.Tensor


@dataclass(frozen=True)
class BenchRow:
    resolution: str
    gaussians: int
    batch_size: int
    case: str
    mode: str
    renderer: str
    mean_ms: float
    min_ms: float
    max_ms: float
    per_frame_ms: float
    speedup_vs_torch: float | None
    speedup_vs_taichi: float | None
    speedup_vs_fast: float | None
    max_abs_diff_vs_torch: float | None
    status: str = "ok"


def sync_mps() -> None:
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def sync_taichi() -> None:
    import taichi as ti
    from taichi_splatting.taichi_queue import TaichiQueue

    sync_mps()
    TaichiQueue.run_sync(ti.sync)
    sync_mps()


def make_inputs(
    height: int,
    width: int,
    gaussians: int,
    batch_size: int,
    sigma_min: float,
    sigma_max: float,
    seed: int,
) -> BatchInputs:
    device = torch.device("mps")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    means_cpu = torch.empty((batch_size, gaussians, 2), dtype=torch.float32)
    means_cpu[..., 0].uniform_(0, width - 1, generator=generator)
    means_cpu[..., 1].uniform_(0, height - 1, generator=generator)
    sigma_cpu = torch.empty((batch_size, gaussians), dtype=torch.float32).uniform_(
        sigma_min, sigma_max, generator=generator
    )
    inv_var_cpu = 1.0 / (sigma_cpu * sigma_cpu)
    conics_cpu = torch.stack([inv_var_cpu, torch.zeros_like(inv_var_cpu), inv_var_cpu], dim=-1)
    colors_cpu = torch.rand((batch_size, gaussians, 3), dtype=torch.float32, generator=generator)
    opacities_cpu = torch.empty((batch_size, gaussians), dtype=torch.float32).uniform_(
        0.08, 0.85, generator=generator
    )
    depths_cpu = torch.rand((batch_size, gaussians), dtype=torch.float32, generator=generator)

    means = means_cpu.to(device).contiguous()
    conics = conics_cpu.to(device).contiguous()
    colors = colors_cpu.to(device).contiguous()
    opacities = opacities_cpu.to(device).contiguous()
    depths = depths_cpu.to(device).contiguous()
    sigma = sigma_cpu.to(device).contiguous()

    axis = torch.zeros((batch_size, gaussians, 2), device=device, dtype=torch.float32)
    axis[..., 0] = 1.0
    sigma2 = torch.stack([sigma, sigma], dim=-1)
    taichi_packed = torch.cat([means, axis, sigma2, opacities.unsqueeze(-1)], dim=-1).contiguous()
    taichi_depths = depths.unsqueeze(-1).contiguous()
    taichi_features = colors.contiguous()
    return BatchInputs(means, conics, colors, opacities, depths, taichi_packed, taichi_depths, taichi_features)


def clone_fast_inputs(inputs: BatchInputs, *, backward: bool) -> tuple[torch.Tensor, ...]:
    tensors = [inputs.means, inputs.conics, inputs.colors, inputs.opacities, inputs.depths]
    clones: list[torch.Tensor] = []
    for i, tensor in enumerate(tensors):
        cloned = tensor.detach().clone().contiguous()
        cloned.requires_grad_(backward and i != 4)
        clones.append(cloned)
    return tuple(clones)


def clone_taichi_inputs(inputs: BatchInputs, *, backward: bool) -> tuple[torch.Tensor, ...]:
    packed = inputs.taichi_packed.detach().clone().contiguous()
    depths = inputs.taichi_depths.detach().clone().contiguous()
    features = inputs.taichi_features.detach().clone().contiguous()
    packed.requires_grad_(backward)
    features.requires_grad_(backward)
    return packed, depths, features


def clear_grads(tensors: tuple[torch.Tensor, ...]) -> None:
    for tensor in tensors:
        if tensor.grad is not None:
            tensor.grad.zero_()


def dense_torch_one(
    means2d: torch.Tensor,
    conics: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    depths: torch.Tensor,
    height: int,
    width: int,
    alpha_threshold: float,
    background: tuple[float, float, float],
) -> torch.Tensor:
    perm = torch.argsort(depths.detach(), dim=0, stable=True)
    means_s = means2d.index_select(0, perm)
    conics_s = conics.index_select(0, perm)
    colors_s = colors.index_select(0, perm)
    opacities_s = opacities.index_select(0, perm)

    ys = torch.arange(height, dtype=means2d.dtype, device=means2d.device) + 0.5
    xs = torch.arange(width, dtype=means2d.dtype, device=means2d.device) + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    out = torch.zeros((height, width, 3), dtype=means2d.dtype, device=means2d.device)
    transmittance = torch.ones((height, width), dtype=means2d.dtype, device=means2d.device)
    bg = torch.tensor(background, dtype=means2d.dtype, device=means2d.device)

    for i in range(means_s.shape[0]):
        dx = xx - means_s[i, 0]
        dy = yy - means_s[i, 1]
        q = conics_s[i]
        power = -0.5 * (q[0] * dx * dx + 2.0 * q[1] * dx * dy + q[2] * dy * dy)
        raw_alpha = opacities_s[i] * torch.exp(power)
        alpha = torch.clamp(raw_alpha, max=0.99)
        alpha = torch.where((power <= 0.0) & (alpha >= alpha_threshold), alpha, torch.zeros_like(alpha))
        weight = transmittance * alpha
        out = out + weight[..., None] * colors_s[i]
        transmittance = transmittance * (1.0 - alpha)

    return out + transmittance[..., None] * bg


def render_torch_reference(inputs: tuple[torch.Tensor, ...], cfg: RasterConfigV3) -> torch.Tensor:
    means, conics, colors, opacities, depths = inputs
    images = [
        dense_torch_one(
            means[b],
            conics[b],
            colors[b],
            opacities[b],
            depths[b],
            cfg.height,
            cfg.width,
            cfg.alpha_threshold,
            cfg.background,
        )
        for b in range(means.shape[0])
    ]
    return torch.stack(images, dim=0)


def render_v2_loop(inputs: tuple[torch.Tensor, ...], cfg: RasterConfigV2) -> torch.Tensor:
    means, conics, colors, opacities, depths = inputs
    images = [rasterize_v2(means[b], conics[b], colors[b], opacities[b], depths[b], cfg) for b in range(means.shape[0])]
    return torch.stack(images, dim=0)


def render_v3_loop(inputs: tuple[torch.Tensor, ...], cfg: RasterConfigV3) -> torch.Tensor:
    means, conics, colors, opacities, depths = inputs
    images = [rasterize_v3(means[b], conics[b], colors[b], opacities[b], depths[b], cfg) for b in range(means.shape[0])]
    return torch.stack(images, dim=0)


def render_v5_native(inputs: tuple[torch.Tensor, ...], cfg: RasterConfigV5) -> torch.Tensor:
    means, conics, colors, opacities, depths = inputs
    return rasterize_v5(means, conics, colors, opacities, depths, cfg)


def make_taichi_renderer(height: int, width: int, tile_size: int) -> Callable[[tuple[torch.Tensor, ...]], torch.Tensor]:
    import taichi as ti
    from taichi_splatting.data_types import RasterConfig as TaichiRasterConfig
    from taichi_splatting.rasterizer import rasterize, rasterize_batch
    from taichi_splatting.taichi_queue import TaichiQueue

    TaichiQueue.init(arch=ti.metal, log_level=ti.ERROR)
    raster_config = TaichiRasterConfig(
        tile_size=tile_size,
        alpha_threshold=1.0 / 255.0,
        clamp_max_alpha=0.99,
        saturate_threshold=0.9999,
        metal_compatible=True,
        kernel_variant="metal_reference",
        sort_backend="auto",
        backward_variant="pixel_reference",
    )
    background = torch.tensor(DEFAULT_BG, device="mps", dtype=torch.float32)

    def render(inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        packed, depths, features = inputs
        if packed.shape[0] == 1:
            raster = rasterize(
                packed[0],
                depths[0],
                features[0],
                image_size=(width, height),
                config=raster_config,
                use_depth16=False,
            )
            image = raster.image + (1.0 - raster.image_weight).unsqueeze(-1).clamp(0.0, 1.0) * background
            return image.unsqueeze(0)

        raster = rasterize_batch(
            packed,
            depths,
            features,
            image_size=(width, height),
            config=raster_config,
            use_depth16=False,
        )
        return raster.image + (1.0 - raster.image_weight).unsqueeze(-1).clamp(0.0, 1.0) * background

    return render


def time_renderer(
    renderer: str,
    render: Callable[[tuple[torch.Tensor, ...]], torch.Tensor],
    inputs: tuple[torch.Tensor, ...],
    sync: Callable[[], None],
    *,
    warmup: int,
    iters: int,
    backward: bool,
) -> tuple[float, float, float]:
    for _ in range(warmup):
        out = render(inputs)
        if backward:
            out.square().mean().backward()
            clear_grads(inputs)
    sync()

    elapsed_ms: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        out = render(inputs)
        if backward:
            out.square().mean().backward()
            clear_grads(inputs)
        sync()
        elapsed_ms.append((time.perf_counter() - start) * 1000.0)

    if not elapsed_ms:
        raise ValueError(f"{renderer}: iters must be positive")
    return sum(elapsed_ms) / len(elapsed_ms), min(elapsed_ms), max(elapsed_ms)


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach() - b.detach()).abs().max().item())


def format_speedup(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:7.2f}x"


def print_rows(rows: list[BenchRow]) -> None:
    for row in rows:
        diff = "n/a" if row.max_abs_diff_vs_torch is None else f"{row.max_abs_diff_vs_torch:.4g}"
        print(
            f"{row.renderer:<13} {row.mode:<16} "
            f"mean={row.mean_ms:>9.3f} ms "
            f"per_frame={row.per_frame_ms:>9.3f} ms "
            f"min={row.min_ms:>9.3f} "
            f"max={row.max_ms:>9.3f} "
            f"speedup_vs_torch={format_speedup(row.speedup_vs_torch)} "
            f"speedup_vs_taichi={format_speedup(row.speedup_vs_taichi)} "
            f"speedup_vs_fast={format_speedup(row.speedup_vs_fast)} "
            f"diff_vs_torch={diff}"
        )


def run_case(
    *,
    height: int,
    width: int,
    gaussians: int,
    batch_size: int,
    case: Case,
    seed: int,
    warmup: int,
    iters: int,
    backward: bool,
    include_torch: bool,
    torch_max_work_items: int,
    check_outputs: bool,
    taichi_render: Callable[[tuple[torch.Tensor, ...]], torch.Tensor],
    requested_renderers: set[str],
) -> list[BenchRow]:
    mode = "forward_backward" if backward else "forward"
    inputs = make_inputs(height, width, gaussians, batch_size, case.sigma_min, case.sigma_max, seed)
    cfg_v2 = RasterConfigV2(height=height, width=width, background=DEFAULT_BG)
    cfg_v3 = RasterConfigV3(height=height, width=width, background=DEFAULT_BG)
    cfg_v5 = RasterConfigV5(height=height, width=width, background=DEFAULT_BG, batch_strategy="flatten")
    resolution = f"{height}x{width}"
    work_items = batch_size * height * width * gaussians

    renderers: list[tuple[str, Callable[[tuple[torch.Tensor, ...]], torch.Tensor], tuple[torch.Tensor, ...], Callable[[], None]]]
    renderers = []
    want_torch = "all" in requested_renderers or "torch_direct" in requested_renderers or "torch" in requested_renderers
    if include_torch and want_torch and work_items <= torch_max_work_items:
        renderers.append(("torch_direct", lambda run_inputs: render_torch_reference(run_inputs, cfg_v3), clone_fast_inputs(inputs, backward=backward), sync_mps))
    elif include_torch and want_torch:
        print(
            f"torch_direct skipped for {resolution} B={batch_size} G={gaussians}: "
            f"work_items={work_items} exceeds --torch-max-work-items={torch_max_work_items}"
        )
    candidates = [
        ("taichi_native", taichi_render, clone_taichi_inputs(inputs, backward=backward), sync_taichi),
        ("metal_v2_loop", lambda run_inputs: render_v2_loop(run_inputs, cfg_v2), clone_fast_inputs(inputs, backward=backward), sync_mps),
        ("metal_v3_loop", lambda run_inputs: render_v3_loop(run_inputs, cfg_v3), clone_fast_inputs(inputs, backward=backward), sync_mps),
        ("metal_v5_native", lambda run_inputs: render_v5_native(run_inputs, cfg_v5), clone_fast_inputs(inputs, backward=backward), sync_mps),
    ]
    aliases = {
        "taichi": "taichi_native",
        "v2": "metal_v2_loop",
        "v3": "metal_v3_loop",
        "v5": "metal_v5_native",
    }
    requested_names = {aliases.get(name, name) for name in requested_renderers}
    renderers.extend(candidate for candidate in candidates if "all" in requested_names or candidate[0] in requested_names)

    reference: torch.Tensor | None = None
    diffs: dict[str, float | None] = {}
    if check_outputs and renderers and renderers[0][0] == "torch_direct":
        with torch.no_grad():
            reference = renderers[0][1](renderers[0][2])
            sync_mps()
            for name, render, run_inputs, sync in renderers:
                out = render(run_inputs)
                sync()
                diffs[name] = max_abs_diff(out, reference)
    else:
        for name, *_ in renderers:
            diffs[name] = None

    raw_rows: list[BenchRow] = []
    for name, render, run_inputs, sync in renderers:
        mean_ms, min_ms, max_ms = time_renderer(
            name,
            render,
            run_inputs,
            sync,
            warmup=warmup,
            iters=iters,
            backward=backward,
        )
        raw_rows.append(
            BenchRow(
                resolution=resolution,
                gaussians=gaussians,
                batch_size=batch_size,
                case=case.name,
                mode=mode,
                renderer=name,
                mean_ms=mean_ms,
                min_ms=min_ms,
                max_ms=max_ms,
                per_frame_ms=mean_ms / float(batch_size),
                speedup_vs_torch=None,
                speedup_vs_taichi=None,
                speedup_vs_fast=None,
                max_abs_diff_vs_torch=diffs.get(name),
            )
        )

    torch_ms = next((row.mean_ms for row in raw_rows if row.renderer == "torch_direct"), None)
    taichi_ms = next((row.mean_ms for row in raw_rows if row.renderer == "taichi_native"), None)
    fast_ms = min((row.mean_ms for row in raw_rows if row.renderer.startswith("metal_")), default=None)
    return [
        BenchRow(
            **{
                **row.__dict__,
                "speedup_vs_torch": (torch_ms / row.mean_ms if torch_ms is not None else None),
                "speedup_vs_taichi": (taichi_ms / row.mean_ms if taichi_ms is not None else None),
                "speedup_vs_fast": (fast_ms / row.mean_ms if fast_ms is not None else None),
            }
        )
        for row in raw_rows
    ]


def write_csv(path: Path, rows: list[BenchRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(BenchRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare direct Torch, Taichi/Metal, and fast-mac v2/v3 projected rasterizers.")
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--gaussians", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260421)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--include-torch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check-outputs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--torch-max-work-items", type=int, default=64_000_000)
    parser.add_argument("--taichi-tile-size", type=int, default=16)
    parser.add_argument(
        "--renderers",
        type=str,
        default="all",
        help="Comma-separated renderer list: all, torch/torch_direct, taichi, v2, v3, v5.",
    )
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available.")

    cases = [
        Case("sparse_sigma_1_5", 1.0, 5.0),
        Case("medium_sigma_3_8", 3.0, 8.0),
    ]

    print(
        f"resolution={args.height}x{args.width} gaussians={args.gaussians} "
        f"batch_size={args.batch_size} warmup={args.warmup} iters={args.iters} "
        f"backward={args.backward}"
    )
    taichi_render = make_taichi_renderer(args.height, args.width, args.taichi_tile_size)
    requested_renderers = {part.strip() for part in args.renderers.split(",") if part.strip()}
    all_rows: list[BenchRow] = []
    for i, case in enumerate(cases):
        print(f"\ncase={case.name} sigma=[{case.sigma_min}, {case.sigma_max}]")
        rows = run_case(
            height=args.height,
            width=args.width,
            gaussians=args.gaussians,
            batch_size=args.batch_size,
            case=case,
            seed=args.seed + i,
            warmup=args.warmup,
            iters=args.iters,
            backward=args.backward,
            include_torch=args.include_torch,
            torch_max_work_items=args.torch_max_work_items,
            check_outputs=args.check_outputs,
            taichi_render=taichi_render,
            requested_renderers=requested_renderers,
        )
        print_rows(rows)
        all_rows.extend(rows)

    if args.csv is not None:
        write_csv(args.csv, all_rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
