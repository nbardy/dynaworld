from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from benchmark_bootstrap import PROJECT_ROOT
from renderer_benchmark_cli import safe_filename_part
from src.train.postprocess_dof import depth_aware_defocus_blur as torch_depth_aware_defocus_blur
from src.train.train_artifacts import write_json
from src.train.train_devices import resolve_torch_device, sync_torch_device

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "depth_aware_dof_demo"
DEFAULT_FRAME_CANDIDATES = [
    PROJECT_ROOT / ".dust3r_probe_frames" / "frame_001.png",
    PROJECT_ROOT / ".dust3r_probe_frames" / "frame_002.png",
    PROJECT_ROOT / "third_party" / "dust3r" / "assets" / "demo.jpg",
    PROJECT_ROOT / "benchmark_outputs" / "taichi_scale_images" / "taichi_metal__512x512__G262144__set0.png",
]


@dataclass(frozen=True)
class DemoCase:
    path: Path
    rgb: torch.Tensor


@dataclass(frozen=True)
class BlurResult:
    depth: torch.Tensor
    coc: torch.Tensor
    blurred: torch.Tensor


def load_rgb(path: Path, size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    left = (size - image.width) // 2
    top = (size - image.height) // 2
    canvas.paste(image, (left, top))
    data = torch.from_numpy(np.asarray(canvas, dtype=np.uint8).copy())
    return data.permute(2, 0, 1).float().div(255.0).contiguous()


def load_cases(paths: list[Path], size: int, limit: int) -> list[DemoCase]:
    cases: list[DemoCase] = []
    for path in paths:
        if path.exists():
            cases.append(DemoCase(path=path, rgb=load_rgb(path, size=size)))
        if len(cases) >= limit:
            break
    if not cases:
        searched = "\n".join(str(path) for path in paths)
        raise FileNotFoundError(f"No demo frames found. Searched:\n{searched}")
    return cases


def gaussian_kernel1d(radius: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if radius <= 0.0:
        return torch.ones(1, device=device, dtype=dtype)
    sigma = max(radius * 0.5, 0.35)
    half_width = max(int(math.ceil(radius * 2.0)), 1)
    x = torch.arange(-half_width, half_width + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (x / sigma).square())
    return kernel / kernel.sum().clamp_min(1.0e-8)


def gaussian_blur(rgb: torch.Tensor, radius: float) -> torch.Tensor:
    if radius <= 0.0:
        return rgb
    channels = rgb.shape[1]
    kernel = gaussian_kernel1d(radius, rgb.device, rgb.dtype)
    pad = kernel.numel() // 2
    weight_x = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    weight_y = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    padded = F.pad(rgb, (pad, pad, 0, 0), mode="reflect")
    blurred = F.conv2d(padded, weight_x, groups=channels)
    padded = F.pad(blurred, (0, 0, pad, pad), mode="reflect")
    return F.conv2d(padded, weight_y, groups=channels)


def estimate_depth_heuristic(rgb: torch.Tensor) -> torch.Tensor:
    luminance = rgb[:, 0:1] * 0.299 + rgb[:, 1:2] * 0.587 + rgb[:, 2:3] * 0.114
    h = rgb.shape[-2]
    vertical = torch.linspace(0.0, 1.0, h, device=rgb.device, dtype=rgb.dtype).view(1, 1, h, 1)
    vertical = vertical.expand_as(luminance)
    local_contrast = (luminance - gaussian_blur(luminance, radius=3.0)).abs()
    depth = 0.45 * (1.0 - luminance) + 0.35 * vertical + 0.20 * (1.0 - local_contrast.clamp(0.0, 1.0))
    flat_min = depth.amin(dim=(-2, -1), keepdim=True)
    flat_max = depth.amax(dim=(-2, -1), keepdim=True)
    return (depth - flat_min) / (flat_max - flat_min).clamp_min(1.0e-6)


def depth_aware_defocus_blur(
    rgb_chw: torch.Tensor,
    focus_depth: torch.Tensor,
    aperture: float,
    radii: tuple[float, ...],
) -> BlurResult:
    rgb = rgb_chw.unsqueeze(0)
    depth = estimate_depth_heuristic(rgb)
    inv_depth = depth.clamp_min(1.0e-6).reciprocal()
    inv_focus = focus_depth.view(1, 1, 1, 1).clamp_min(1.0e-6).reciprocal()
    blur_strength = torch.as_tensor(float(aperture), device=rgb.device, dtype=rgb.dtype) * 0.1
    max_radius = int(math.ceil(radii[-1]))
    coc = (blur_strength * (inv_depth - inv_focus).abs()).clamp(0.0, float(max_radius))
    blurred = torch_depth_aware_defocus_blur(
        rgb,
        depth,
        inv_focus_depth=inv_focus.reshape(()),
        blur_strength=blur_strength,
        max_radius=max_radius,
        depth_edge_sigma=0.6,
        detach_depth=True,
    )
    return BlurResult(depth=depth.squeeze(0), coc=coc.squeeze(0), blurred=blurred.squeeze(0))


def run_one(rgb: torch.Tensor, device: torch.device, aperture: float, radii: tuple[float, ...], warmup: int, iters: int):
    rgb_device = rgb.to(device=device, dtype=torch.float32)
    focus_depth = torch.tensor(0.45, device=device, dtype=torch.float32, requires_grad=True)
    rgb_grad = rgb_device.detach().clone().requires_grad_(True)

    for _ in range(warmup):
        result = depth_aware_defocus_blur(rgb_device, focus_depth.detach(), aperture, radii)
        sync_torch_device(device)

    start = time.perf_counter()
    result = None
    for _ in range(iters):
        result = depth_aware_defocus_blur(rgb_device, focus_depth.detach(), aperture, radii)
    sync_torch_device(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / max(iters, 1)

    grad_result = depth_aware_defocus_blur(rgb_grad, focus_depth, aperture, radii)
    grad_loss = grad_result.blurred.square().mean() + 0.01 * grad_result.coc.mean()
    grad_loss.backward()
    sync_torch_device(device)

    if result is None:
        raise RuntimeError("No blur result was produced.")
    grad_stats = {
        "loss": float(grad_loss.detach().cpu()),
        "rgb_grad_mean_abs": float(rgb_grad.grad.detach().abs().mean().cpu()),
        "rgb_grad_max_abs": float(rgb_grad.grad.detach().abs().max().cpu()),
        "focus_depth_grad": float(focus_depth.grad.detach().cpu()),
    }
    return result, elapsed_ms, grad_stats


def to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
    array = tensor.permute(1, 2, 0).mul(255.0).round().byte().numpy()
    return Image.fromarray(array)


def make_panel(title: str, image: Image.Image) -> Image.Image:
    label_h = 24
    panel = Image.new("RGB", (image.width, image.height + label_h), (20, 20, 20))
    panel.paste(image, (0, label_h))
    draw = ImageDraw.Draw(panel)
    draw.text((6, 5), title, fill=(240, 240, 240))
    return panel


def save_comparison(
    output_path: Path,
    rgb: torch.Tensor,
    cpu_result: BlurResult,
    accel_result: BlurResult,
    accel_label: str,
) -> None:
    diff = (accel_result.blurred.detach().cpu() - cpu_result.blurred.detach().cpu()).abs().mul(20.0)
    panels = [
        make_panel("rgb", to_pil(rgb)),
        make_panel("depth", to_pil(cpu_result.depth)),
        make_panel("coc", to_pil(cpu_result.coc / cpu_result.coc.max().clamp_min(1.0e-6))),
        make_panel("cpu blurred", to_pil(cpu_result.blurred)),
        make_panel(f"{accel_label} blurred", to_pil(accel_result.blurred)),
        make_panel("abs diff x20", to_pil(diff)),
    ]
    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    strip = Image.new("RGB", (width, height), (0, 0, 0))
    x = 0
    for panel in panels:
        strip.paste(panel, (x, 0))
        x += panel.width
    output_path.parent.mkdir(parents=True, exist_ok=True)
    strip.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Depth-aware defocus blur demo using Torch CPU and PyTorch MPS.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--size", type=int, default=192)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--aperture", type=float, default=10.0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--frames", type=Path, nargs="*", default=DEFAULT_FRAME_CANDIDATES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    radii = (0.0, 1.5, 3.0, 5.0, 8.0)
    accel_device = resolve_torch_device("auto", auto_cuda=False)
    accel_label = "mps" if accel_device.type == "mps" else "cpu_fallback"
    cases = load_cases(args.frames, size=args.size, limit=args.limit)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "frame_count": len(cases),
                "size": args.size,
                "aperture": args.aperture,
                "radii": radii,
                "accelerated_path": accel_label,
                "torch_mps_available": bool(torch.backends.mps.is_available()),
            },
            indent=2,
        )
    )

    summary = []
    for index, case in enumerate(cases):
        cpu_result, cpu_ms, cpu_grad = run_one(
            case.rgb, torch.device("cpu"), args.aperture, radii, args.warmup, args.iters
        )
        accel_result, accel_ms, accel_grad = run_one(case.rgb, accel_device, args.aperture, radii, args.warmup, args.iters)
        accel_result_cpu = BlurResult(
            depth=accel_result.depth.detach().cpu(),
            coc=accel_result.coc.detach().cpu(),
            blurred=accel_result.blurred.detach().cpu(),
        )
        max_abs_diff = float((accel_result_cpu.blurred - cpu_result.blurred).abs().max())
        mean_abs_diff = float((accel_result_cpu.blurred - cpu_result.blurred).abs().mean())
        output_path = output_dir / f"{index:02d}_{safe_filename_part(case.path.stem, allow_dot=False)}_{accel_label}_comparison.png"
        save_comparison(output_path, case.rgb, cpu_result, accel_result_cpu, accel_label)

        row = {
            "frame": str(case.path.relative_to(PROJECT_ROOT) if case.path.is_relative_to(PROJECT_ROOT) else case.path),
            "output": str(output_path),
            "cpu_ms": cpu_ms,
            f"{accel_label}_ms": accel_ms,
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "cpu_grad": cpu_grad,
            f"{accel_label}_grad": accel_grad,
        }
        summary.append(row)
        print(json.dumps(row, indent=2))

    summary_path = output_dir / "summary.json"
    write_json(summary_path, summary)
    print(f"Wrote {len(summary)} comparisons and summary to {output_dir}")


if __name__ == "__main__":
    main()
