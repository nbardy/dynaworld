"""CPU-first support-changing visibility bridge for STAR UVT feature tubes.

Same-support alpha losses can only update pixels the current rasterizer already
touches. This prototype adds a tiny soft nearest-tube coverage proxy that sends
geometry gradients from target foreground pixels to projected tube centers even
when dense alpha has zero target hits.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    from .report_artifacts import write_report_json, write_report_text
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import write_report_json, write_report_text
from research_experiments.star_uvt_feature_tubes.dense_feature_tube_prototype import (
    FeatureScreenTimeTubeModel,
    FeatureTubeRenderConfig,
    _inv_softplus,
    _logit,
    _resolve_device,
    _sync_device,
    render_model_features,
)


@dataclass(frozen=True)
class BridgeConfig:
    frames: int = 6
    height: int = 28
    width: int = 28
    tubes: int = 8
    feature_dim: int = 32
    steps: int = 80
    lr: float = 0.15
    seed: int = 19
    target_radius: float = 2.35
    proxy_scale_px: float = 3.0
    proxy_temperature: float = 0.75


def _target_mask(config: FeatureTubeRenderConfig, *, radius: float, device: torch.device) -> Tensor:
    y = torch.arange(config.height, dtype=torch.float32, device=device) + 0.5
    x = torch.arange(config.width, dtype=torch.float32, device=device) + 0.5
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    masks: list[Tensor] = []
    for frame in range(config.frames):
        progress = 0.0 if config.frames <= 1 else float(frame) / float(config.frames - 1)
        cx = 0.68 * float(config.width) + 0.10 * float(config.width) * progress
        cy = 0.70 * float(config.height) - 0.06 * float(config.height) * progress
        masks.append((xx - cx).square() + (yy - cy).square() <= float(radius) ** 2)
    return torch.stack(masks, dim=0)


def _target_points(mask: Tensor, *, frames: int) -> Tensor:
    ids = mask.nonzero(as_tuple=False)
    if ids.numel() == 0:
        raise ValueError("target mask produced no foreground points")
    frame = ids[:, 0].to(torch.float32)
    y = ids[:, 1].to(torch.float32) + 0.5
    x = ids[:, 2].to(torch.float32) + 0.5
    t = frame - 0.5 * float(frames - 1)
    return torch.stack((x, y, t), dim=-1).contiguous()


def _make_miss_model(config: FeatureTubeRenderConfig, bridge: BridgeConfig, *, device: torch.device) -> FeatureScreenTimeTubeModel:
    model = FeatureScreenTimeTubeModel(
        bridge.tubes,
        config,
        seed=bridge.seed,
        device=device,
    )
    generator = torch.Generator(device="cpu").manual_seed(bridge.seed + 101)
    with torch.no_grad():
        jitter = torch.randn((bridge.tubes, 2), generator=generator, dtype=torch.float32) * 0.35
        miss_center = torch.tensor(
            [0.18 * float(config.width), 0.18 * float(config.height)],
            dtype=torch.float32,
        )
        model.center_uv.copy_((miss_center.view(1, 2) + jitter).to(device))
        model.center_t.zero_()
        model.velocity_uv.zero_()
        precision = torch.full((bridge.tubes, 3), 0.60, dtype=torch.float32, device=device)
        model.raw_precision.copy_(_inv_softplus(precision - model.min_precision))
        model.raw_opacity.copy_(_logit(torch.full((bridge.tubes,), 0.35 / 0.99, dtype=torch.float32, device=device)))
        model.raw_feature.zero_()
        model.depth0.copy_(torch.linspace(0.8, 1.2, bridge.tubes, dtype=torch.float32, device=device))
    return model


def _alpha_metrics(model: FeatureScreenTimeTubeModel, mask: Tensor) -> dict[str, float]:
    _sync_device(model.center_uv.device)
    started = time.perf_counter()
    _feature_image, alpha = render_model_features(model)
    _sync_device(model.center_uv.device)
    render_ms = (time.perf_counter() - started) * 1000.0
    target_alpha = alpha[mask]
    background_alpha = alpha[~mask]
    return {
        "render_ms": float(render_ms),
        "target_alpha_mean": float(target_alpha.mean().detach().cpu().item()),
        "target_alpha_gt_0_01": float((target_alpha > 0.01).to(torch.float32).mean().detach().cpu().item()),
        "target_alpha_gt_0_10": float((target_alpha > 0.10).to(torch.float32).mean().detach().cpu().item()),
        "background_alpha_mean": float(background_alpha.mean().detach().cpu().item()),
        "alpha_mean": float(alpha.mean().detach().cpu().item()),
    }


def _dense_alpha_loss(model: FeatureScreenTimeTubeModel, mask: Tensor) -> Tensor:
    _feature_image, alpha = render_model_features(model)
    target = mask.to(dtype=alpha.dtype, device=alpha.device)
    return F.mse_loss(alpha, target)


def _support_proxy_loss(
    model: FeatureScreenTimeTubeModel,
    target_points: Tensor,
    *,
    scale_px: float,
    temperature: float,
) -> Tensor:
    points_xy = target_points[:, :2]
    points_t = target_points[:, 2]
    dt = points_t[:, None] - model.center_t[:, 0][None, :]
    projected = model.center_uv[None, :, :] + dt[:, :, None] * model.velocity_uv[None, :, :]
    dist = (points_xy[:, None, :] - projected).square().sum(dim=-1) / max(float(scale_px) ** 2, 1.0e-6)
    soft_nearest = -float(temperature) * torch.logsumexp(-dist / float(temperature), dim=1)
    velocity_penalty = 0.0025 * model.velocity_uv.square().mean()
    return soft_nearest.mean() + velocity_penalty


def _grad_seen(model: FeatureScreenTimeTubeModel) -> dict[str, bool]:
    return {
        "center_uv": model.center_uv.grad is not None and bool((model.center_uv.grad.abs() > 0).any().detach().cpu()),
        "velocity_uv": model.velocity_uv.grad is not None and bool((model.velocity_uv.grad.abs() > 0).any().detach().cpu()),
        "raw_precision": model.raw_precision.grad is not None
        and bool((model.raw_precision.grad.abs() > 0).any().detach().cpu()),
        "raw_opacity": model.raw_opacity.grad is not None
        and bool((model.raw_opacity.grad.abs() > 0).any().detach().cpu()),
    }


def _train_same_support_alpha(
    model: FeatureScreenTimeTubeModel,
    mask: Tensor,
    *,
    steps: int,
    lr: float,
) -> dict[str, Any]:
    optimizer = torch.optim.Adam([model.center_uv, model.velocity_uv, model.raw_precision, model.raw_opacity], lr=lr)
    losses: list[float] = []
    grad_seen: dict[str, bool] | None = None
    step_ms: list[float] = []
    for _step in range(steps):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = _dense_alpha_loss(model, mask)
        loss.backward()
        if grad_seen is None:
            grad_seen = _grad_seen(model)
        optimizer.step()
        _sync_device(model.center_uv.device)
        losses.append(float(loss.detach().cpu().item()))
        step_ms.append((time.perf_counter() - started) * 1000.0)
    return {
        "losses": losses,
        "start_loss": losses[0],
        "end_loss": losses[-1],
        "loss_decreased": bool(losses[-1] < losses[0]),
        "first_step_grad_seen": grad_seen or {},
        "mean_step_ms": sum(step_ms) / max(len(step_ms), 1),
    }


def _train_support_proxy(
    model: FeatureScreenTimeTubeModel,
    mask: Tensor,
    target_points: Tensor,
    *,
    steps: int,
    lr: float,
    scale_px: float,
    temperature: float,
) -> dict[str, Any]:
    optimizer = torch.optim.Adam([model.center_uv, model.velocity_uv], lr=lr)
    proxy_losses: list[float] = []
    alpha_losses: list[float] = []
    grad_seen: dict[str, bool] | None = None
    step_ms: list[float] = []
    for _step in range(steps):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        proxy = _support_proxy_loss(model, target_points, scale_px=scale_px, temperature=temperature)
        proxy.backward()
        if grad_seen is None:
            grad_seen = _grad_seen(model)
        optimizer.step()
        _sync_device(model.center_uv.device)
        proxy_losses.append(float(proxy.detach().cpu().item()))
        with torch.no_grad():
            alpha_losses.append(float(_dense_alpha_loss(model, mask).detach().cpu().item()))
        step_ms.append((time.perf_counter() - started) * 1000.0)
    return {
        "proxy_losses": proxy_losses,
        "start_proxy_loss": proxy_losses[0],
        "end_proxy_loss": proxy_losses[-1],
        "proxy_loss_decreased": bool(proxy_losses[-1] < proxy_losses[0]),
        "alpha_losses": alpha_losses,
        "start_alpha_loss": alpha_losses[0],
        "end_alpha_loss": alpha_losses[-1],
        "alpha_loss_decreased": bool(alpha_losses[-1] < alpha_losses[0]),
        "first_step_grad_seen": grad_seen or {},
        "mean_step_ms": sum(step_ms) / max(len(step_ms), 1),
    }


def run_gate(bridge: BridgeConfig, *, device_name: str) -> dict[str, Any]:
    device = _resolve_device(device_name)
    render_config = FeatureTubeRenderConfig(
        frames=bridge.frames,
        height=bridge.height,
        width=bridge.width,
        feature_dim=bridge.feature_dim,
        alpha_threshold=1.0 / 255.0,
        max_alpha=0.99,
    )
    mask = _target_mask(render_config, radius=bridge.target_radius, device=device)
    points = _target_points(mask, frames=render_config.frames)

    initial_model = _make_miss_model(render_config, bridge, device=device)
    same_support_model = _make_miss_model(render_config, bridge, device=device)
    proxy_model = _make_miss_model(render_config, bridge, device=device)

    initial = _alpha_metrics(initial_model, mask)
    same_train = _train_same_support_alpha(
        same_support_model,
        mask,
        steps=bridge.steps,
        lr=bridge.lr,
    )
    same_final = _alpha_metrics(same_support_model, mask)
    proxy_train = _train_support_proxy(
        proxy_model,
        mask,
        points,
        steps=bridge.steps,
        lr=bridge.lr,
        scale_px=bridge.proxy_scale_px,
        temperature=bridge.proxy_temperature,
    )
    proxy_final = _alpha_metrics(proxy_model, mask)

    proxy_gain = proxy_final["target_alpha_gt_0_10"] - initial["target_alpha_gt_0_10"]
    same_gain = same_final["target_alpha_gt_0_10"] - initial["target_alpha_gt_0_10"]
    pass_gate = (
        initial["target_alpha_gt_0_10"] <= 0.01
        and proxy_gain >= 0.25
        and proxy_final["target_alpha_mean"] > initial["target_alpha_mean"] + 0.05
        and same_gain < 0.10
        and bool(proxy_train["first_step_grad_seen"].get("center_uv", False))
        and bool(proxy_train["first_step_grad_seen"].get("velocity_uv", False))
        and proxy_train["proxy_loss_decreased"]
    )

    return {
        "gate": "star_uvt_visibility_support_bridge_cpu_proxy",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pass": bool(pass_gate),
        "device": str(device),
        "config": {
            "frames": bridge.frames,
            "height": bridge.height,
            "width": bridge.width,
            "tubes": bridge.tubes,
            "feature_dim": bridge.feature_dim,
            "steps": bridge.steps,
            "lr": bridge.lr,
            "seed": bridge.seed,
            "target_radius": bridge.target_radius,
            "proxy_scale_px": bridge.proxy_scale_px,
            "proxy_temperature": bridge.proxy_temperature,
            "target_point_count": int(points.shape[0]),
        },
        "initial": initial,
        "same_support_alpha": {
            "train": same_train,
            "final": same_final,
            "target_coverage_gain_gt_0_10": float(same_gain),
        },
        "support_proxy": {
            "train": proxy_train,
            "final": proxy_final,
            "target_coverage_gain_gt_0_10": float(proxy_gain),
        },
        "interpretation": (
            "The proxy is a CPU parity gate for a support-changing objective. It proves geometry gradients "
            "exist before a Metal implementation; it is not a promoted trainer or visual-quality result."
        ),
    }


def _fmt(value: float) -> str:
    return f"{float(value):.4f}"


def write_markdown(payload: Mapping[str, Any], path: Path) -> None:
    initial = payload["initial"]
    same = payload["same_support_alpha"]
    proxy = payload["support_proxy"]
    lines = [
        "# STAR UVT Visibility Support Bridge Prototype",
        "",
        f"- generated: `{payload['generated_at']}`",
        f"- gate: `{payload['gate']}`",
        f"- pass: `{payload['pass']}`",
        f"- device: `{payload['device']}`",
        "",
        "## Purpose",
        "",
        "Same-support alpha/grid losses were rejected because they cannot create support where the current",
        "rasterizer has no useful visibility. This CPU gate tests the missing implementation detail:",
        "a soft target-pixel to projected-tube coverage proxy that sends geometry gradients before any",
        "new Metal shader work.",
        "",
        "## Results",
        "",
        "| path | target alpha mean | target alpha >0.10 | background alpha mean | train loss start -> end | mean step ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| initial | {_fmt(initial['target_alpha_mean'])} | "
            f"{_fmt(initial['target_alpha_gt_0_10'])} | {_fmt(initial['background_alpha_mean'])} | n/a | n/a |"
        ),
        (
            f"| same-support alpha | {_fmt(same['final']['target_alpha_mean'])} | "
            f"{_fmt(same['final']['target_alpha_gt_0_10'])} | {_fmt(same['final']['background_alpha_mean'])} | "
            f"{_fmt(same['train']['start_loss'])} -> {_fmt(same['train']['end_loss'])} | "
            f"{_fmt(same['train']['mean_step_ms'])} |"
        ),
        (
            f"| support proxy | {_fmt(proxy['final']['target_alpha_mean'])} | "
            f"{_fmt(proxy['final']['target_alpha_gt_0_10'])} | {_fmt(proxy['final']['background_alpha_mean'])} | "
            f"{_fmt(proxy['train']['start_proxy_loss'])} -> {_fmt(proxy['train']['end_proxy_loss'])} | "
            f"{_fmt(proxy['train']['mean_step_ms'])} |"
        ),
        "",
        "## Gradient Check",
        "",
        f"- same-support first-step grads: `{same['train']['first_step_grad_seen']}`",
        f"- support-proxy first-step grads: `{proxy['train']['first_step_grad_seen']}`",
        "",
        "## Decision",
        "",
    ]
    if payload["pass"]:
        lines.append(
            "Pass. The proxy creates center/velocity gradients and increases target alpha coverage from a zero-hit start."
        )
        lines.append(
            "This is enough to justify a first-class STAR UVT visibility bridge experiment; it is not enough to claim quality."
        )
    else:
        lines.append(
            "Fail. Do not port this proxy into the trainer until the CPU gate shows support-changing behavior."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_report_text(path, "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--out-json", type=Path, default=Path("outputs/benchmarks/2026-05-20_star_uvt_visibility_support_bridge_cpu_proxy.json"))
    parser.add_argument("--out-md", type=Path, default=Path("outputs/benchmarks/2026-05-20_star_uvt_visibility_support_bridge_cpu_proxy.md"))
    args = parser.parse_args()
    payload = run_gate(
        BridgeConfig(steps=int(args.steps), lr=float(args.lr), seed=int(args.seed)),
        device_name=str(args.device),
    )
    write_report_json(args.out_json, payload)
    write_markdown(payload, args.out_md)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
