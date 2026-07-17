from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from colorize import FeatureToColor
try:
    from .report_artifacts import write_report_json
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import write_report_json
from research_project.trainer_harness.data import load_video_target
from train_devices import sync_torch_device
from torch_gsplat_bridge_star_uvt import UVTRenderConfig
from torch_gsplat_bridge_star_uvt.feature_rasterize import (
    direct_atomic_feature_backward,
    render_uvt_feature_tubes,
    render_uvt_feature_tubes_autograd,
    render_uvt_feature_tubes_autograd_frame_chunk,
)

from dense_feature_tube_prototype import FeatureScreenTimeTubeModel, FeatureTubeRenderConfig, colorize_and_compose
from star_uvt_colorizers import build_default_feature_colorizer


def _sync() -> None:
    sync_torch_device(torch.device("mps"))


def _make_uvt_config(config: FeatureTubeRenderConfig, *, tile_capacity: int) -> UVTRenderConfig:
    return UVTRenderConfig(
        height=config.height,
        width=config.width,
        frames=config.frames,
        tile_t=2,
        tile_capacity=tile_capacity,
        alpha_threshold=config.alpha_threshold,
        max_alpha=config.max_alpha,
    )


def _make_colorizer(feature_dim: int) -> FeatureToColor:
    return build_default_feature_colorizer(feature_dim=feature_dim, device=torch.device("mps"))


def _target_rgb(
    config: FeatureTubeRenderConfig,
    *,
    seed: int,
    video_path: Path | None,
    start_seconds: float | None,
    fps: float | None,
    duration_seconds: float | None,
    image_crop_mode: str,
) -> tuple[torch.Tensor, str]:
    if video_path is not None:
        target_thwc = load_video_target(
            video_path,
            target_size=config.height,
            max_frames=config.frames,
            device="mps",
            start_seconds=start_seconds,
            fps=fps,
            duration_seconds=duration_seconds,
            image_crop_mode=image_crop_mode,
        )
        return target_thwc.permute(0, 3, 1, 2).contiguous(), str(video_path)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.rand((config.frames, 3, config.height, config.width), generator=generator).to("mps"), "random"


def _grad_norms(model: FeatureScreenTimeTubeModel, colorizer: FeatureToColor) -> dict[str, float]:
    out: dict[str, float] = {}
    for prefix, module in (("model", model), ("colorizer", colorizer)):
        for name, param in module.named_parameters():
            if param.grad is not None:
                out[f"{prefix}.{name}"] = float(param.grad.detach().norm().cpu().item())
    return out


def run_autograd_parity(*, feature_dim: int, seed: int) -> dict[str, Any]:
    config = FeatureTubeRenderConfig(frames=2, height=8, width=8, feature_dim=feature_dim)
    uvt_config = _make_uvt_config(config, tile_capacity=128)
    model = FeatureScreenTimeTubeModel(3, config, seed=seed, device="mps")
    ma, q_uvt, depth0, depth_beta, opacity, feature = model.tensors()

    direct_result = render_uvt_feature_tubes(ma, q_uvt, depth0, depth_beta, opacity, feature, uvt_config)
    grad_feature = torch.linspace(
        -0.2,
        0.3,
        direct_result.feature_image.numel(),
        dtype=torch.float32,
        device="mps",
    ).view_as(direct_result.feature_image)
    grad_alpha = torch.linspace(
        0.1,
        -0.15,
        direct_result.alpha.numel(),
        dtype=torch.float32,
        device="mps",
    ).view_as(direct_result.alpha)

    manual = direct_atomic_feature_backward(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        feature,
        grad_feature,
        grad_alpha,
        uvt_config,
    )

    ma_ref = ma.detach().clone().requires_grad_(True)
    q_ref = q_uvt.detach().clone().requires_grad_(True)
    opacity_ref = opacity.detach().clone().requires_grad_(True)
    feature_ref = feature.detach().clone().requires_grad_(True)
    auto = render_uvt_feature_tubes_autograd(
        ma_ref,
        q_ref,
        depth0.detach(),
        depth_beta.detach(),
        opacity_ref,
        feature_ref,
        uvt_config,
    )
    loss = (auto.feature_image * grad_feature).sum() + (auto.alpha * grad_alpha).sum()
    loss.backward()
    _sync()

    expected = {
        "ma": manual[0],
        "q": manual[1],
        "opacity": manual[2],
        "feature": manual[3],
    }
    got = {
        "ma": ma_ref.grad,
        "q": q_ref.grad,
        "opacity": opacity_ref.grad,
        "feature": feature_ref.grad,
    }
    errors = {
        key: float((got[key].detach().cpu() - expected[key].detach().cpu()).abs().max().item())
        for key in expected
    }
    return {
        "feature_dim": feature_dim,
        "max_abs_errors": errors,
        "pass": max(errors.values()) <= 1.0e-6,
    }


def run_frame_chunk_parity(*, feature_dim: int, seed: int, chunk_size: int) -> dict[str, Any]:
    config = FeatureTubeRenderConfig(frames=4, height=8, width=8, feature_dim=feature_dim)
    uvt_config = _make_uvt_config(config, tile_capacity=128)
    model = FeatureScreenTimeTubeModel(5, config, seed=seed, device="mps")
    ma, q_uvt, depth0, depth_beta, opacity, feature = model.tensors()
    full = render_uvt_feature_tubes(ma, q_uvt, depth0.detach(), depth_beta.detach(), opacity, feature, uvt_config)
    grad_feature = torch.linspace(
        -0.3,
        0.2,
        full.feature_image.numel(),
        dtype=torch.float32,
        device="mps",
    ).view_as(full.feature_image)
    grad_alpha = torch.linspace(
        0.15,
        -0.1,
        full.alpha.numel(),
        dtype=torch.float32,
        device="mps",
    ).view_as(full.alpha)

    full_ma = ma.detach().clone().requires_grad_(True)
    full_q = q_uvt.detach().clone().requires_grad_(True)
    full_opacity = opacity.detach().clone().requires_grad_(True)
    full_feature = feature.detach().clone().requires_grad_(True)
    full_render = render_uvt_feature_tubes_autograd(
        full_ma,
        full_q,
        depth0.detach(),
        depth_beta.detach(),
        full_opacity,
        full_feature,
        uvt_config,
    )
    full_loss = (full_render.feature_image * grad_feature).sum() + (full_render.alpha * grad_alpha).sum()
    full_loss.backward()
    _sync()

    chunk_ma = ma.detach().clone().requires_grad_(True)
    chunk_q = q_uvt.detach().clone().requires_grad_(True)
    chunk_opacity = opacity.detach().clone().requires_grad_(True)
    chunk_feature = feature.detach().clone().requires_grad_(True)
    for frame_start in range(0, config.frames, chunk_size):
        chunk_frames = min(chunk_size, config.frames - frame_start)
        chunk_render = render_uvt_feature_tubes_autograd_frame_chunk(
            chunk_ma,
            chunk_q,
            depth0.detach(),
            depth_beta.detach(),
            chunk_opacity,
            chunk_feature,
            uvt_config,
            frame_start=frame_start,
            chunk_frames=chunk_frames,
        )
        feature_slice = grad_feature[frame_start : frame_start + chunk_frames]
        alpha_slice = grad_alpha[frame_start : frame_start + chunk_frames]
        chunk_loss = (chunk_render.feature_image * feature_slice).sum() + (chunk_render.alpha * alpha_slice).sum()
        chunk_loss.backward()
    _sync()

    expected = {
        "ma": full_ma.grad,
        "q": full_q.grad,
        "opacity": full_opacity.grad,
        "feature": full_feature.grad,
    }
    got = {
        "ma": chunk_ma.grad,
        "q": chunk_q.grad,
        "opacity": chunk_opacity.grad,
        "feature": chunk_feature.grad,
    }
    errors = {
        key: float((got[key].detach().cpu() - expected[key].detach().cpu()).abs().max().item())
        for key in expected
    }
    return {
        "feature_dim": feature_dim,
        "frames": config.frames,
        "chunk_size": chunk_size,
        "max_abs_errors": errors,
        "pass": max(errors.values()) <= 2.0e-6,
    }


def run_overfit(
    *,
    frames: int,
    size: int,
    tubes: int,
    feature_dim: int,
    steps: int,
    lr: float,
    seed: int,
    tile_capacity: int,
    video_path: Path | None,
    start_seconds: float | None,
    fps: float | None,
    duration_seconds: float | None,
    image_crop_mode: str,
) -> dict[str, Any]:
    config = FeatureTubeRenderConfig(frames=frames, height=size, width=size, feature_dim=feature_dim)
    uvt_config = _make_uvt_config(config, tile_capacity=tile_capacity)
    model = FeatureScreenTimeTubeModel(tubes, config, seed=seed, device="mps")
    colorizer = _make_colorizer(feature_dim)
    target_rgb, target_source = _target_rgb(
        config,
        seed=seed + 101,
        video_path=video_path,
        start_seconds=start_seconds,
        fps=fps,
        duration_seconds=duration_seconds,
        image_crop_mode=image_crop_mode,
    )
    optimizer = torch.optim.Adam([*model.parameters(), *colorizer.parameters()], lr=lr)

    losses: list[float] = []
    timings: list[dict[str, float]] = []
    final_grad_norms: dict[str, float] = {}
    final_overflow = 0
    final_unstable = 0

    for _step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        ma, q_uvt, depth0, depth_beta, opacity, feature = model.tensors()
        _sync()
        t0 = time.perf_counter()
        render = render_uvt_feature_tubes_autograd(ma, q_uvt, depth0.detach(), depth_beta.detach(), opacity, feature, uvt_config)
        _sync()
        t1 = time.perf_counter()
        rgb = colorize_and_compose(render.feature_image, render.alpha, colorizer)
        loss = torch.mean((rgb - target_rgb).square())
        _sync()
        t2 = time.perf_counter()
        loss.backward()
        _sync()
        t3 = time.perf_counter()
        optimizer.step()
        _sync()
        t4 = time.perf_counter()

        losses.append(float(loss.detach().cpu().item()))
        timings.append(
            {
                "render_forward_ms": (t1 - t0) * 1000.0,
                "colorize_loss_ms": (t2 - t1) * 1000.0,
                "backward_ms": (t3 - t2) * 1000.0,
                "optimizer_ms": (t4 - t3) * 1000.0,
                "step_ms": (t4 - t0) * 1000.0,
            }
        )
        final_grad_norms = _grad_norms(model, colorizer)

    with torch.no_grad():
        ma, q_uvt, depth0, depth_beta, opacity, feature = model.tensors()
        aux = render_uvt_feature_tubes(ma, q_uvt, depth0.detach(), depth_beta.detach(), opacity, feature, uvt_config)
        final_overflow = int(aux.tile_overflow.sum().cpu().item())
        final_unstable = int(aux.tile_unstable.sum().cpu().item())

    mean_timing = {
        key: sum(row[key] for row in timings) / max(len(timings), 1)
        for key in ("render_forward_ms", "colorize_loss_ms", "backward_ms", "optimizer_ms", "step_ms")
    }
    return {
        "target_source": target_source,
        "frames": frames,
        "size": size,
        "tubes": tubes,
        "feature_dim": feature_dim,
        "steps": steps,
        "lr": lr,
        "tile_capacity": tile_capacity,
        "start_loss": losses[0] if losses else None,
        "end_loss": losses[-1] if losses else None,
        "start_psnr": None if not losses else float(-10.0 * torch.log10(torch.tensor(max(losses[0], 1.0e-12))).item()),
        "end_psnr": None if not losses else float(-10.0 * torch.log10(torch.tensor(max(losses[-1], 1.0e-12))).item()),
        "loss_decreased": bool(losses and losses[-1] < losses[0]),
        "losses": losses,
        "mean_timing_ms": mean_timing,
        "last_timing_ms": timings[-1] if timings else None,
        "grad_norms": final_grad_norms,
        "raw_feature_grad_seen": final_grad_norms.get("model.raw_feature", 0.0) > 0.0,
        "center_uv_grad_seen": final_grad_norms.get("model.center_uv", 0.0) > 0.0,
        "velocity_uv_grad_seen": final_grad_norms.get("model.velocity_uv", 0.0) > 0.0,
        "raw_opacity_grad_seen": final_grad_norms.get("model.raw_opacity", 0.0) > 0.0,
        "colorizer_grad_seen": any(key.startswith("colorizer.") and value > 0.0 for key, value in final_grad_norms.items()),
        "tile_overflow_sum": final_overflow,
        "tile_unstable_sum": final_unstable,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--tubes", type=int, default=64)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2.0e-2)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--chunk-size", type=int, default=2)
    parser.add_argument("--tile-capacity", type=int, default=128)
    parser.add_argument("--video-path", type=Path, default=None)
    parser.add_argument("--start-seconds", type=float, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--duration-seconds", type=float, default=None)
    parser.add_argument("--image-crop-mode", default="center_square")
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("feature autograd overfit benchmark requires MPS")

    parity = run_autograd_parity(feature_dim=args.feature_dim, seed=args.seed)
    chunk_parity = run_frame_chunk_parity(
        feature_dim=args.feature_dim,
        seed=args.seed + 17,
        chunk_size=args.chunk_size,
    )
    overfit = run_overfit(
        frames=args.frames,
        size=args.size,
        tubes=args.tubes,
        feature_dim=args.feature_dim,
        steps=args.steps,
        lr=args.lr,
        seed=args.seed,
        tile_capacity=args.tile_capacity,
        video_path=args.video_path,
        start_seconds=args.start_seconds,
        fps=args.fps,
        duration_seconds=args.duration_seconds,
        image_crop_mode=args.image_crop_mode,
    )
    result = {
        "gate": "star_uvt_feature_autograd_overfit",
        "autograd_parity": parity,
        "frame_chunk_parity": chunk_parity,
        "overfit": overfit,
        "pass": bool(
            parity["pass"]
            and chunk_parity["pass"]
            and overfit["loss_decreased"]
            and overfit["raw_feature_grad_seen"]
            and overfit["center_uv_grad_seen"]
            and overfit["velocity_uv_grad_seen"]
            and overfit["raw_opacity_grad_seen"]
            and overfit["colorizer_grad_seen"]
            and overfit["tile_overflow_sum"] == 0
        ),
    }

    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.out_json is not None:
        write_report_json(args.out_json, result)
    print(payload)


if __name__ == "__main__":
    main()
