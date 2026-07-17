from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from config_utils import load_config_file
from checkpoint_utils import load_checkpoint_mapping
from diagnose_powerfoam_heldout_error import build_model, load_model_for_checkpoint
from losses import ssim_per_image
from pipeline.diagnostics import reconstruction_eval_metrics
from powerfoam_eval_color import (
    apply_channel_affine,
    apply_rgb_matrix_affine,
    fit_channel_affine,
    fit_rgb_matrix_affine,
)
from powerfoam_eval_render import render_powerfoam_samples
from powerfoam_metal_config import resolve_config
from powerfoam_objectives import composite_powerfoam_background
from powerfoam_training_data import load_powerfoam_training_data
try:
    from .report_artifacts import relative_to_project as rel, write_report_json
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import relative_to_project as rel, write_report_json
from train_devices import resolve_torch_device


def color_ssim_loss(rendered: torch.Tensor, target: torch.Tensor, cfg: dict[str, Any]) -> torch.Tensor:
    window_size = min(int(cfg["losses"]["ssim_window_size"]), int(rendered.shape[-1]), int(rendered.shape[-2]))
    if window_size % 2 == 0:
        window_size -= 1
    window_size = max(window_size, 1)
    return 1.0 - ssim_per_image(
        rendered,
        target,
        window_size=window_size,
        c1=float(cfg["losses"]["ssim_c1"]),
        c2=float(cfg["losses"]["ssim_c2"]),
    ).mean()


def optimize_train_transform(
    *,
    initial: torch.Tensor,
    apply_fn: Any,
    train_render: torch.Tensor,
    train_target: torch.Tensor,
    cfg: dict[str, Any],
    ssim_weight: float,
    steps: int,
    lr: float,
) -> torch.Tensor:
    transform = initial.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([transform], lr=float(lr))
    best_transform = transform.detach().clone()
    best_loss = float("inf")
    for _step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        corrected = apply_fn(train_render, transform)
        loss = (corrected - train_target).square().mean() + float(ssim_weight) * color_ssim_loss(
            corrected,
            train_target,
            cfg,
        )
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        if loss_value < best_loss:
            best_loss = loss_value
            best_transform = transform.detach().clone()
    return best_transform


def weight_slug(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def metric_block(renders: torch.Tensor, targets: torch.Tensor, cfg: dict[str, Any], *, prefix: str) -> dict[str, float]:
    return reconstruction_eval_metrics(renders, targets, cfg, prefix=prefix)


@torch.no_grad()
def render_raw_split(
    model: Any,
    frame_indices: torch.Tensor,
    rays: torch.Tensor | None,
    *,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rendered, alpha = render_powerfoam_samples(model, frame_indices, rays=rays, batch_size=batch_size)
    return rendered.detach().cpu().to(dtype=torch.float32), alpha.detach().cpu().to(dtype=torch.float32)


def fit_constant_background(rendered: torch.Tensor, alpha: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    weight = (1.0 - alpha).unsqueeze(1)
    numerator = ((target - rendered) * weight).sum(dim=(0, 2, 3))
    denominator = weight.square().sum(dim=(0, 2, 3)).clamp_min(1.0e-8)
    return (numerator / denominator).clamp(0.0, 1.0)


def apply_constant_background(rendered: torch.Tensor, alpha: torch.Tensor, background: torch.Tensor) -> torch.Tensor:
    bg = background.to(dtype=rendered.dtype).view(1, 3, 1, 1)
    return composite_powerfoam_background(rendered, alpha, bg).clamp(0.0, 1.0)


def transform_row(
    *,
    name: str,
    transform: torch.Tensor,
    apply_fn: Any,
    train_render: torch.Tensor,
    train_target: torch.Tensor,
    heldout_render: torch.Tensor,
    heldout_target: torch.Tensor,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    train_corrected = apply_fn(train_render, transform)
    heldout_corrected = apply_fn(heldout_render, transform)
    return {
        "name": name,
        "transform": transform.detach().cpu().tolist(),
        "train": metric_block(train_corrected, train_target, cfg, prefix="train"),
        "heldout": metric_block(heldout_corrected, heldout_target, cfg, prefix="heldout"),
    }


def background_row(
    *,
    name: str,
    background: torch.Tensor,
    train_render_raw: torch.Tensor,
    train_alpha: torch.Tensor,
    train_target: torch.Tensor,
    heldout_render_raw: torch.Tensor,
    heldout_alpha: torch.Tensor,
    heldout_target: torch.Tensor,
    cfg: dict[str, Any],
    opt_steps: int,
    opt_lr: float,
    opt_ssim_weights: list[float],
) -> dict[str, Any]:
    train_render = apply_constant_background(train_render_raw, train_alpha, background)
    heldout_render = apply_constant_background(heldout_render_raw, heldout_alpha, background)
    transforms: list[tuple[str, torch.Tensor, Any]] = [
        (
            "train_fit_channel_affine",
            fit_channel_affine(train_render, train_target),
            apply_channel_affine,
        ),
        (
            "train_fit_rgb_matrix_affine",
            fit_rgb_matrix_affine(train_render, train_target),
            apply_rgb_matrix_affine,
        ),
        (
            "heldout_oracle_channel_affine",
            fit_channel_affine(heldout_render, heldout_target),
            apply_channel_affine,
        ),
        (
            "heldout_oracle_rgb_matrix_affine",
            fit_rgb_matrix_affine(heldout_render, heldout_target),
            apply_rgb_matrix_affine,
        ),
    ]
    if int(opt_steps) > 0:
        for transform_name, transform, apply_fn in transforms[:2]:
            for ssim_weight in opt_ssim_weights:
                transforms.append(
                    (
                        f"{transform_name}_mse_ssim_w{weight_slug(ssim_weight)}_adam",
                        optimize_train_transform(
                            initial=transform,
                            apply_fn=apply_fn,
                            train_render=train_render,
                            train_target=train_target,
                            cfg=cfg,
                            ssim_weight=float(ssim_weight),
                            steps=int(opt_steps),
                            lr=float(opt_lr),
                        ),
                        apply_fn,
                    )
                )
    return {
        "name": name,
        "background": background.detach().cpu().tolist(),
        "train": metric_block(train_render, train_target, cfg, prefix="train"),
        "heldout": metric_block(heldout_render, heldout_target, cfg, prefix="heldout"),
        "transforms": [
            transform_row(
                name=transform_name,
                transform=transform,
                apply_fn=apply_fn,
                train_render=train_render,
                train_target=train_target,
                heldout_render=heldout_render,
                heldout_target=heldout_target,
                cfg=cfg,
            )
            for transform_name, transform, apply_fn in transforms
        ],
    }


def build_report(
    config_path: Path,
    checkpoint_path: Path | None,
    *,
    batch_size: int,
    device_name: str,
    opt_steps: int,
    opt_lr: float,
    opt_ssim_weights: list[float],
    background_names: set[str] | None,
) -> dict[str, Any]:
    device = resolve_torch_device(device_name, auto_cuda=False)
    if checkpoint_path is None:
        cfg = resolve_config(load_config_file(config_path))
        training_data = load_powerfoam_training_data(cfg, device)
        model = build_model(cfg, training_data, device)
        model.eval()
        checkpoint_step = 0
        checkpoint_label = "init"
    else:
        cfg, training_data, model = load_model_for_checkpoint(config_path, checkpoint_path, device)
        checkpoint_step = int(load_checkpoint_mapping(checkpoint_path, map_location="cpu").get("step", -1))
        checkpoint_label = rel(checkpoint_path)
    train_render_raw, train_alpha = render_raw_split(
        model,
        training_data["sample_frame_indices"],
        training_data["sample_rays"],
        batch_size=batch_size,
    )
    if training_data["heldout_targets"] is None:
        raise ValueError("Color-affine diagnostic requires heldout targets.")
    heldout_render_raw, heldout_alpha = render_raw_split(
        model,
        training_data["heldout_frame_indices"],
        training_data["heldout_rays"],
        batch_size=batch_size,
    )
    train_target = training_data["targets"].detach().cpu().to(dtype=torch.float32)
    heldout_target = training_data["heldout_targets"].detach().cpu().to(dtype=torch.float32)
    backgrounds = [
        ("black", torch.zeros(3, dtype=torch.float32)),
        ("train_fit_constant_background", fit_constant_background(train_render_raw, train_alpha, train_target)),
        (
            "heldout_oracle_constant_background",
            fit_constant_background(heldout_render_raw, heldout_alpha, heldout_target),
        ),
    ]
    if background_names is not None:
        backgrounds = [(name, background) for name, background in backgrounds if name in background_names]
        if not backgrounds:
            raise ValueError(f"No requested backgrounds matched: {sorted(background_names)}")
    background_rows = [
        background_row(
            name=name,
            background=background,
            train_render_raw=train_render_raw,
            train_alpha=train_alpha,
            train_target=train_target,
            heldout_render_raw=heldout_render_raw,
            heldout_alpha=heldout_alpha,
            heldout_target=heldout_target,
            cfg=cfg,
            opt_steps=int(opt_steps),
            opt_lr=float(opt_lr),
            opt_ssim_weights=[float(weight) for weight in opt_ssim_weights],
        )
        for name, background in backgrounds
    ]
    black = background_rows[0]
    return {
        "schema_version": "powerfoam_color_affine_diagnostic_v1",
        "config": rel(config_path),
        "checkpoint": checkpoint_label,
        "checkpoint_step": checkpoint_step,
        "init_only": checkpoint_path is None,
        "opt_steps": int(opt_steps),
        "opt_lr": float(opt_lr),
        "opt_ssim_weights": [float(weight) for weight in opt_ssim_weights],
        "output_dir": rel(cfg["logging"]["output_dir"]),
        "baseline": {
            "train": black["train"],
            "heldout": black["heldout"],
        },
        "transforms": black["transforms"],
        "backgrounds": background_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bound PowerFoam heldout error from train-fit and oracle color affines.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--init-only", action="store_true", help="Diagnose the initialized model before any checkpointed training.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--opt-steps", type=int, default=0)
    parser.add_argument("--opt-lr", type=float, default=0.05)
    parser.add_argument("--opt-ssim-weights", type=float, nargs="+", default=[0.2, 1.0, 5.0])
    parser.add_argument(
        "--background-names",
        nargs="+",
        choices=["black", "train_fit_constant_background", "heldout_oracle_constant_background"],
        default=None,
    )
    args = parser.parse_args()
    cfg = resolve_config(load_config_file(args.config))
    checkpoint = None if bool(args.init_only) else (args.checkpoint or (cfg["logging"]["output_dir"] / "checkpoint_best.pt"))
    output = args.output or (
        cfg["logging"]["output_dir"]
        / ("color_affine_diagnostics_init.json" if bool(args.init_only) else "color_affine_diagnostics.json")
    )
    report = build_report(
        args.config,
        checkpoint,
        batch_size=int(args.batch_size),
        device_name=str(args.device),
        opt_steps=int(args.opt_steps),
        opt_lr=float(args.opt_lr),
        opt_ssim_weights=[float(weight) for weight in args.opt_ssim_weights],
        background_names=None if args.background_names is None else {str(name) for name in args.background_names},
    )
    write_report_json(output, report)
    print(json.dumps({"output": rel(output), "checkpoint_step": report["checkpoint_step"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
