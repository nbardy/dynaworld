"""CPU support birth/split gate for STAR UVT feature tubes.

The previous visibility proxy proved gradients can pull existing tubes toward
target support, but the first trainer port was too slow and barely changed
dense alpha. This prototype tests the next mechanism before shader work:
reallocate a fixed budget of currently useless tubes onto uncovered target
support, then let the existing dense-alpha gradient refine that support.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import torch
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
)
from research_experiments.star_uvt_feature_tubes.visibility_support_bridge_prototype import (
    BridgeConfig,
    _alpha_metrics,
    _make_miss_model,
    _target_mask,
    _target_points,
    _train_same_support_alpha,
    _train_support_proxy,
)


@dataclass(frozen=True)
class BirthSplitConfig:
    frames: int = 6
    height: int = 28
    width: int = 28
    tubes: int = 16
    birth_tubes: int = 8
    feature_dim: int = 32
    same_support_steps: int = 80
    proxy_steps: int = 80
    birth_refine_steps: int = 40
    lr: float = 0.15
    birth_refine_lr: float = 0.05
    seed: int = 19
    target_radius: float = 2.35
    proxy_scale_px: float = 3.0
    proxy_temperature: float = 0.75
    birth_spatial_precision: float = 0.45
    birth_temporal_precision: float = 0.035
    birth_opacity: float = 0.60


def _bridge_config(config: BirthSplitConfig, *, steps: int | None = None, lr: float | None = None) -> BridgeConfig:
    return BridgeConfig(
        frames=config.frames,
        height=config.height,
        width=config.width,
        tubes=config.tubes,
        feature_dim=config.feature_dim,
        steps=config.same_support_steps if steps is None else int(steps),
        lr=config.lr if lr is None else float(lr),
        seed=config.seed,
        target_radius=config.target_radius,
        proxy_scale_px=config.proxy_scale_px,
        proxy_temperature=config.proxy_temperature,
    )


def _target_centroids(mask: Tensor) -> Tensor:
    ids = mask.nonzero(as_tuple=False)
    if ids.numel() == 0:
        raise ValueError("target mask produced no foreground points")
    centroids: list[Tensor] = []
    for frame in range(int(mask.shape[0])):
        frame_ids = ids[ids[:, 0] == frame]
        if frame_ids.numel() == 0:
            raise ValueError(f"target mask has no foreground pixels on frame {frame}")
        x = frame_ids[:, 2].to(torch.float32).mean() + 0.5
        y = frame_ids[:, 1].to(torch.float32).mean() + 0.5
        centroids.append(torch.stack((x, y)))
    return torch.stack(centroids, dim=0)


def _fit_target_motion(centroids: Tensor) -> tuple[Tensor, Tensor]:
    frames = int(centroids.shape[0])
    if frames <= 1:
        return centroids[0], torch.zeros_like(centroids[0])
    times = torch.arange(frames, dtype=torch.float32, device=centroids.device) - 0.5 * float(frames - 1)
    velocity = (centroids[-1] - centroids[0]) / (times[-1] - times[0])
    mid_index = frames // 2
    center_at_t0 = centroids[mid_index] - velocity * times[mid_index]
    return center_at_t0, velocity


def _birth_offsets(count: int, *, radius: float, device: torch.device) -> Tensor:
    if count <= 0:
        raise ValueError("birth_tubes must be positive")
    offsets: list[tuple[float, float]] = []
    for index in range(count):
        if index == 0:
            ring = 0.0
        elif index < max(2, count // 2):
            ring = 0.55
        else:
            ring = 1.05
        angle = 2.0 * math.pi * float(index) / float(count)
        offsets.append((math.cos(angle) * radius * ring, math.sin(angle) * radius * ring))
    return torch.tensor(offsets, dtype=torch.float32, device=device)


def apply_support_birth_split(
    model: FeatureScreenTimeTubeModel,
    mask: Tensor,
    config: BirthSplitConfig,
) -> dict[str, Any]:
    """Reuse a fixed tube budget by moving a subset of dead tubes onto target support."""

    if config.birth_tubes <= 0:
        raise ValueError("birth_tubes must be positive")
    if config.birth_tubes > model.tube_count:
        raise ValueError("birth_tubes cannot exceed the fixed tube budget")
    device = model.center_uv.device
    centroids = _target_centroids(mask.to(device=device))
    center_at_t0, velocity = _fit_target_motion(centroids)
    offsets = _birth_offsets(config.birth_tubes, radius=config.target_radius, device=device)
    precision = torch.tensor(
        [config.birth_spatial_precision, config.birth_spatial_precision, config.birth_temporal_precision],
        dtype=torch.float32,
        device=device,
    )
    if bool((precision <= model.min_precision).any().detach().cpu()):
        raise ValueError("birth precisions must exceed model.min_precision")
    if not (0.0 < float(config.birth_opacity) < 0.99):
        raise ValueError("birth_opacity must be between 0 and 0.99")

    with torch.no_grad():
        sl = slice(0, config.birth_tubes)
        model.center_uv[sl].copy_(center_at_t0.view(1, 2) + offsets)
        model.center_t[sl].zero_()
        model.velocity_uv[sl].copy_(velocity.view(1, 2).expand(config.birth_tubes, 2))
        model.raw_precision[sl].copy_(
            _inv_softplus((precision - model.min_precision).view(1, 3).expand(config.birth_tubes, 3))
        )
        model.raw_opacity[sl].copy_(
            _logit(torch.full((config.birth_tubes,), float(config.birth_opacity) / 0.99, device=device))
        )

    return {
        "fixed_tube_budget": int(model.tube_count),
        "reallocated_tubes": int(config.birth_tubes),
        "center_at_t0": [float(v) for v in center_at_t0.detach().cpu().tolist()],
        "velocity_uv": [float(v) for v in velocity.detach().cpu().tolist()],
        "birth_spatial_precision": float(config.birth_spatial_precision),
        "birth_temporal_precision": float(config.birth_temporal_precision),
        "birth_opacity": float(config.birth_opacity),
    }


def run_gate(config: BirthSplitConfig, *, device_name: str) -> dict[str, Any]:
    device = _resolve_device(device_name)
    render_config = FeatureTubeRenderConfig(
        frames=config.frames,
        height=config.height,
        width=config.width,
        feature_dim=config.feature_dim,
        alpha_threshold=1.0 / 255.0,
        max_alpha=0.99,
    )
    mask = _target_mask(render_config, radius=config.target_radius, device=device)
    points = _target_points(mask, frames=render_config.frames)

    initial_model = _make_miss_model(render_config, _bridge_config(config), device=device)
    same_model = _make_miss_model(render_config, _bridge_config(config), device=device)
    proxy_model = _make_miss_model(render_config, _bridge_config(config), device=device)
    birth_model = _make_miss_model(render_config, _bridge_config(config), device=device)

    initial = _alpha_metrics(initial_model, mask)
    same_train = _train_same_support_alpha(
        same_model,
        mask,
        steps=config.same_support_steps,
        lr=config.lr,
    )
    same_final = _alpha_metrics(same_model, mask)
    proxy_train = _train_support_proxy(
        proxy_model,
        mask,
        points,
        steps=config.proxy_steps,
        lr=config.lr,
        scale_px=config.proxy_scale_px,
        temperature=config.proxy_temperature,
    )
    proxy_final = _alpha_metrics(proxy_model, mask)
    birth_split = apply_support_birth_split(birth_model, mask, config)
    birth_initial = _alpha_metrics(birth_model, mask)
    birth_refine_train = _train_same_support_alpha(
        birth_model,
        mask,
        steps=config.birth_refine_steps,
        lr=config.birth_refine_lr,
    )
    birth_final = _alpha_metrics(birth_model, mask)

    birth_gain = birth_initial["target_alpha_gt_0_10"] - initial["target_alpha_gt_0_10"]
    refined_background_improved = birth_final["background_alpha_mean"] < birth_initial["background_alpha_mean"]
    pass_gate = (
        initial["target_alpha_gt_0_10"] <= 0.01
        and same_final["target_alpha_gt_0_10"] < 0.10
        and birth_initial["target_alpha_gt_0_10"] >= 0.80
        and birth_initial["target_alpha_mean"] >= 0.50
        and birth_final["target_alpha_gt_0_10"] >= 0.80
        and refined_background_improved
        and birth_refine_train["loss_decreased"]
        and birth_split["fixed_tube_budget"] == config.tubes
    )

    return {
        "gate": "star_uvt_visibility_support_birth_split_cpu",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pass": bool(pass_gate),
        "device": str(device),
        "config": {
            "frames": config.frames,
            "height": config.height,
            "width": config.width,
            "tubes": config.tubes,
            "birth_tubes": config.birth_tubes,
            "feature_dim": config.feature_dim,
            "same_support_steps": config.same_support_steps,
            "proxy_steps": config.proxy_steps,
            "birth_refine_steps": config.birth_refine_steps,
            "lr": config.lr,
            "birth_refine_lr": config.birth_refine_lr,
            "seed": config.seed,
            "target_radius": config.target_radius,
            "proxy_scale_px": config.proxy_scale_px,
            "proxy_temperature": config.proxy_temperature,
            "birth_spatial_precision": config.birth_spatial_precision,
            "birth_temporal_precision": config.birth_temporal_precision,
            "birth_opacity": config.birth_opacity,
            "target_point_count": int(points.shape[0]),
        },
        "initial": initial,
        "same_support_alpha": {
            "train": same_train,
            "final": same_final,
            "target_coverage_gain_gt_0_10": float(same_final["target_alpha_gt_0_10"] - initial["target_alpha_gt_0_10"]),
        },
        "support_proxy": {
            "train": proxy_train,
            "final": proxy_final,
            "target_coverage_gain_gt_0_10": float(proxy_final["target_alpha_gt_0_10"] - initial["target_alpha_gt_0_10"]),
        },
        "birth_split": {
            "birth": birth_split,
            "initial": birth_initial,
            "refine_train": birth_refine_train,
            "final": birth_final,
            "target_coverage_gain_gt_0_10": float(birth_gain),
            "background_alpha_reduction": float(
                birth_initial["background_alpha_mean"] - birth_final["background_alpha_mean"]
            ),
        },
        "interpretation": (
            "Fixed-budget support birth/split is a CPU mechanism gate. It proves that reusing dead tubes for "
            "target support can immediately change dense visibility, then ordinary alpha gradients can refine "
            "background leakage. It is not yet a trainer or Metal implementation."
        ),
    }


def _fmt(value: float) -> str:
    return f"{float(value):.4f}"


def write_markdown(payload: Mapping[str, Any], path: Path) -> None:
    initial = payload["initial"]
    same = payload["same_support_alpha"]
    proxy = payload["support_proxy"]
    birth = payload["birth_split"]
    lines = [
        "# STAR UVT Visibility Support Birth/Split Prototype",
        "",
        f"- generated: `{payload['generated_at']}`",
        f"- gate: `{payload['gate']}`",
        f"- pass: `{payload['pass']}`",
        f"- device: `{payload['device']}`",
        "",
        "## Purpose",
        "",
        "The center-only and opacity/precision visibility proxies were useful plumbing gates but did not",
        "move dense support enough in the real trainer. This CPU gate tests the next mechanism: reuse a",
        "fixed tube budget by splitting/reallocating currently useless tubes onto uncovered target support,",
        "then let the existing alpha gradient refine background leakage.",
        "",
        "## Results",
        "",
        "| path | target alpha mean | target alpha >0.10 | background alpha mean | train loss start -> end | mean step ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| initial miss | {_fmt(initial['target_alpha_mean'])} | "
            f"{_fmt(initial['target_alpha_gt_0_10'])} | {_fmt(initial['background_alpha_mean'])} | n/a | n/a |"
        ),
        (
            f"| same-support alpha | {_fmt(same['final']['target_alpha_mean'])} | "
            f"{_fmt(same['final']['target_alpha_gt_0_10'])} | {_fmt(same['final']['background_alpha_mean'])} | "
            f"{_fmt(same['train']['start_loss'])} -> {_fmt(same['train']['end_loss'])} | "
            f"{_fmt(same['train']['mean_step_ms'])} |"
        ),
        (
            f"| center support proxy | {_fmt(proxy['final']['target_alpha_mean'])} | "
            f"{_fmt(proxy['final']['target_alpha_gt_0_10'])} | {_fmt(proxy['final']['background_alpha_mean'])} | "
            f"{_fmt(proxy['train']['start_proxy_loss'])} -> {_fmt(proxy['train']['end_proxy_loss'])} | "
            f"{_fmt(proxy['train']['mean_step_ms'])} |"
        ),
        (
            f"| birth/split initial | {_fmt(birth['initial']['target_alpha_mean'])} | "
            f"{_fmt(birth['initial']['target_alpha_gt_0_10'])} | {_fmt(birth['initial']['background_alpha_mean'])} | "
            "n/a | n/a |"
        ),
        (
            f"| birth/split refined | {_fmt(birth['final']['target_alpha_mean'])} | "
            f"{_fmt(birth['final']['target_alpha_gt_0_10'])} | {_fmt(birth['final']['background_alpha_mean'])} | "
            f"{_fmt(birth['refine_train']['start_loss'])} -> {_fmt(birth['refine_train']['end_loss'])} | "
            f"{_fmt(birth['refine_train']['mean_step_ms'])} |"
        ),
        "",
        "## Birth/Split Parameters",
        "",
        f"- fixed tube budget: `{birth['birth']['fixed_tube_budget']}`",
        f"- reallocated tubes: `{birth['birth']['reallocated_tubes']}`",
        f"- fitted center at t=0: `{birth['birth']['center_at_t0']}`",
        f"- fitted velocity uv: `{birth['birth']['velocity_uv']}`",
        f"- birth precision spatial/temporal: `{birth['birth']['birth_spatial_precision']}` / `{birth['birth']['birth_temporal_precision']}`",
        f"- birth opacity: `{birth['birth']['birth_opacity']}`",
        "",
        "## Decision",
        "",
    ]
    if payload["pass"]:
        lines.append(
            "Pass. Fixed-budget birth/split changes dense support immediately while same-support alpha cannot."
        )
        lines.append(
            "This is the next mechanism to port into the first-class STAR UVT trainer; it is not yet a Metal quality claim."
        )
    else:
        lines.append("Fail. Do not port birth/split until this CPU support gate passes.")
    write_report_text(path, "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--tubes", type=int, default=16)
    parser.add_argument("--birth-tubes", type=int, default=8)
    parser.add_argument("--same-support-steps", type=int, default=80)
    parser.add_argument("--proxy-steps", type=int, default=80)
    parser.add_argument("--birth-refine-steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--birth-refine-lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("outputs/benchmarks/2026-05-20_star_uvt_visibility_birth_split_cpu_gate.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("outputs/benchmarks/2026-05-20_star_uvt_visibility_birth_split_cpu_gate.md"),
    )
    args = parser.parse_args()
    payload = run_gate(
        BirthSplitConfig(
            tubes=int(args.tubes),
            birth_tubes=int(args.birth_tubes),
            same_support_steps=int(args.same_support_steps),
            proxy_steps=int(args.proxy_steps),
            birth_refine_steps=int(args.birth_refine_steps),
            lr=float(args.lr),
            birth_refine_lr=float(args.birth_refine_lr),
            seed=int(args.seed),
        ),
        device_name=str(args.device),
    )
    write_report_json(args.out_json, payload)
    write_markdown(payload, args.out_md)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
