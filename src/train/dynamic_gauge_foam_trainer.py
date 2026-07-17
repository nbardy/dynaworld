from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from tqdm import trange

from checkpoint_utils import atomic_torch_save
from config_utils import serialize_config_value
from dynamic_gauge_foam import (
    DynamicGaugeFoamVideo,
    build_knn_edges,
)
from dynamic_gauge_config import resolve_config
from dynamic_gauge_objectives import dynamic_gauge_training_loss
from dynamic_gauge_rendering import dynamic_gauge_render_kwargs, render_dynamic_gauge_sequence
from pipeline.diagnostics import reconstruction_l1_mse_metrics
from powerfoam_training import powerfoam_train_batch_indices
from sequence_data import load_video_sequence
from train_artifacts import write_resolved_config
from train_devices import resolve_torch_device
from train_logging import (
    log_wandb_run_payload,
    log_wandb_run_payload_lazy,
    mapped_metric_payload,
    should_log_image,
    should_log_scalar,
    should_log_video,
    wandb_run_lifecycle,
)
from train_optim import optimizer_backward_step
from wandb_media import (
    build_rgb_alpha_eval_media_payload,
    make_wandb_video,
)
from video_io import save_rgb_alpha_eval_media, video_fps_from_config

DYNAMIC_GAUGE_TRAIN_WANDB_KEYS = (
    ("loss", "Train/Loss"),
    ("l1", "Train/L1"),
    ("mse", "Train/MSE"),
    ("conn", "Train/Connection"),
    ("temp", "Train/Temporal"),
    ("elapsed_s", "Timing/ElapsedSeconds"),
)


def optimizer_param_groups(model: DynamicGaugeFoamVideo, cfg: dict[str, Any]) -> list[dict[str, object]]:
    lr = float(cfg["train"]["lr"])
    return [
        {"params": [model.p0], "lr": lr * float(cfg["train"]["center_lr_multiplier"]), "name": "centers"},
        {"params": [model.log_radius], "lr": lr * float(cfg["train"]["radius_lr_multiplier"]), "name": "radii"},
        {"params": [model.logit_opacity], "lr": lr * float(cfg["train"]["opacity_lr_multiplier"]), "name": "opacity"},
        {"params": [model.twist_ctrl], "lr": lr * float(cfg["train"]["twist_lr_multiplier"]), "name": "twist"},
        {"params": [model.atlas], "lr": lr * float(cfg["train"]["atlas_lr_multiplier"]), "name": "atlas"},
        {"params": list(model.color_mlp.parameters()), "lr": lr * float(cfg["train"]["color_lr_multiplier"]), "name": "color_mlp"},
    ]


def log_artifacts(
    model: DynamicGaugeFoamVideo,
    targets: torch.Tensor,
    cfg: dict[str, Any],
    step: int,
    output_dir: Path,
    wandb_run: Any | None,
) -> dict[str, float]:
    model.eval()
    renders, alphas, depths = render_dynamic_gauge_sequence(model, targets.size(0), cfg)
    metrics = {
        **reconstruction_l1_mse_metrics(renders, targets.cpu(), prefix="eval"),
        "eval_alpha_mean": float(alphas.mean().item()),
        "eval_alpha_max": float(alphas.max().item()),
    }
    metrics.update(model.state_metrics())
    save_rgb_alpha_eval_media(
        output_dir,
        step,
        renders,
        targets,
        alphas,
        fps=video_fps_from_config(cfg),
        save_videos=should_log_video(cfg, step),
    )

    def _wandb_payload() -> dict[str, Any]:
        fps = video_fps_from_config(cfg)
        payload: dict[str, Any] = mapped_metric_payload(
            metrics,
            (
                ("eval_l1", "Eval/L1"),
                ("eval_mse", "Eval/MSE"),
                ("eval_alpha_mean", "Eval/AlphaMean"),
                ("eval_alpha_max", "Eval/AlphaMax"),
                ("state_mean_center_delta", "State/MeanCenterDelta"),
                ("state_p95_center_delta", "State/P95CenterDelta"),
                ("state_max_center_delta", "State/MaxCenterDelta"),
                ("state_mean_radius", "State/MeanRadius"),
                ("state_mean_radius_delta", "State/MeanRadiusDelta"),
                ("state_mean_opacity", "State/MeanOpacity"),
                ("state_mean_atlas_delta", "State/MeanAtlasDelta"),
                ("state_mean_twist_abs", "State/MeanTwistAbs"),
            ),
        )
        payload.update(
            build_rgb_alpha_eval_media_payload(
                renders,
                targets,
                alphas,
                step=step,
                fps=fps,
                include_videos=should_log_video(cfg, step),
            )
        )
        if should_log_video(cfg, step):
            depth_vis = depths / depths[depths > 0.0].median().clamp_min(1.0e-6) if bool((depths > 0.0).any()) else depths
            payload["Depth_Video"] = make_wandb_video(depth_vis.clamp(0.0, 1.0).unsqueeze(1).repeat(1, 3, 1, 1), fps)
        return payload

    log_wandb_run_payload_lazy(wandb_run, _wandb_payload, step=step)
    model.train()
    return metrics


def run_training(config: dict[str, Any]) -> None:
    cfg = resolve_config(config)
    torch.manual_seed(int(cfg["train"]["seed"]))
    device = resolve_torch_device(str(cfg["train"]["device"]), auto_cuda=True)

    output_dir: Path = cfg["logging"]["output_dir"]
    write_resolved_config(output_dir, cfg)

    sequence = load_video_sequence(
        cfg["data"]["video_path"],
        target_size=int(cfg["render"]["render_size"]),
        max_frames=int(cfg["data"]["max_frames"]),
        frame_source=str(cfg["data"]["frame_source"]),
    )
    targets = sequence.frames.to(device=device, dtype=torch.float32)
    frame_times = sequence.frame_times.reshape(-1).to(device=device, dtype=torch.float32)
    cfg["video_fps"] = float(sequence.video_fps)
    with wandb_run_lifecycle(cfg) as wandb_run:
        model = DynamicGaugeFoamVideo(
            frame_times=frame_times,
            init_frames=sequence.frames,
            primitive_count=int(cfg["model"]["primitives"]),
            feature_dim=int(cfg["model"]["feature_dim"]),
            atlas_res=int(cfg["model"]["atlas_res"]),
            num_time_ctrl=int(cfg["model"]["num_time_ctrl"]),
            render_size=int(cfg["render"]["render_size"]),
            fov_degrees=float(cfg["render"]["fov_degrees"]),
            init_depth=float(cfg["model"]["init_depth"]),
            radius_scale=float(cfg["model"]["radius_scale"]),
            opacity_init=float(cfg["model"]["opacity_init"]),
            feature_noise=float(cfg["model"]["feature_noise"]),
            color_hidden_dim=int(cfg["model"]["color_hidden_dim"]),
            rgb_skip=bool(cfg["model"]["rgb_skip"]),
            seed=int(cfg["train"]["seed"]),
        ).to(device)
        optimizer = torch.optim.AdamW(optimizer_param_groups(model, cfg), lr=float(cfg["train"]["lr"]), weight_decay=1.0e-6)
        edge_index = build_knn_edges(model.p0.detach(), int(cfg["losses"]["knn_k"])).to(device=device, dtype=torch.long)

        print(
            {
                "arch": "dynamic_gauge_foam",
                "device": str(device),
                "video_path": str(cfg["data"]["video_path"]),
                "frames": int(targets.size(0)),
                "render_size": int(cfg["render"]["render_size"]),
                "primitives": int(cfg["model"]["primitives"]),
                "feature_dim": int(cfg["model"]["feature_dim"]),
                "atlas_res": int(cfg["model"]["atlas_res"]),
                "steps": int(cfg["train"]["steps"]),
            }
        )
        initial_metrics = log_artifacts(model, targets, cfg, 0, output_dir, wandb_run)
        print({"step": 0, **initial_metrics})

        start_time = time.perf_counter()
        progress = trange(1, int(cfg["train"]["steps"]) + 1, desc="dynamic_gauge_foam")
        for step in progress:
            frame_indices = powerfoam_train_batch_indices(targets.size(0), cfg, device=device)
            out = model(frame_indices, **dynamic_gauge_render_kwargs(cfg))
            target = targets[frame_indices]
            loss, loss_terms = dynamic_gauge_training_loss(model, out, target, frame_indices, edge_index, cfg)

            optimizer_backward_step(optimizer, loss, clip_grad_params=model.parameters(), max_grad_norm=1.0)

            progress.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}", l1=f"{float(loss_terms['l1'].detach().cpu()):.4f}")
            if should_log_scalar(cfg, step):
                elapsed = time.perf_counter() - start_time
                train_metrics = {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "l1": float(loss_terms["l1"].detach().cpu()),
                    "mse": float(loss_terms["mse"].detach().cpu()),
                    "conn": float(loss_terms["connection"].detach().cpu()),
                    "temp": float(loss_terms["temporal"].detach().cpu()),
                    "elapsed_s": elapsed,
                }
                print(train_metrics)
                log_wandb_run_payload(
                    wandb_run,
                    mapped_metric_payload(train_metrics, DYNAMIC_GAUGE_TRAIN_WANDB_KEYS),
                    step=step,
                )
            if should_log_image(cfg, step):
                metrics = log_artifacts(model, targets, cfg, step, output_dir, wandb_run)
                print({"step": step, **metrics})

        atomic_torch_save(
            {"model": model.state_dict(), "config": serialize_config_value(cfg)},
            output_dir / "checkpoint_final.pt",
        )


__all__ = ["run_training"]
