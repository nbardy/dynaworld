from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from tqdm import trange

from pipeline.diagnostics import reconstruction_l1_mse_metrics
from powerfoam_direct_config import resolve_config
from powerfoam_direct import DirectPowerFoamVideo, direct_powerfoam_render_options
from powerfoam_checkpoints import save_powerfoam_checkpoint
from powerfoam_eval_render import powerfoam_eval_batch_size, render_powerfoam_samples
from powerfoam_geometry import powerfoam_rays_from_camera, powerfoam_rays_from_camera_grid
from powerfoam_objectives import direct_powerfoam_loss, scheduled_loss_weights
from powerfoam_training import flatten_multiview_powerfoam_samples, powerfoam_train_batch_indices
from powerfoam_training_data import load_powerfoam_training_data
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
)
from video_io import save_rgb_alpha_eval_media, video_fps_from_config


DIRECT_POWERFOAM_DATA_KEYS = (
    "targets",
    "sample_frame_indices",
    "sample_rays",
    "heldout_targets",
    "heldout_frame_indices",
    "heldout_rays",
    "init_frames",
    "frame_count",
    "video_fps",
    "source_label",
    "train_views",
    "heldout_views",
    "pose_source",
)

DIRECT_POWERFOAM_TRAIN_WANDB_KEYS = (
    ("loss", "Train/Loss"),
    ("l1", "Train/L1"),
    ("mse", "Train/MSE"),
    ("ssim_loss", "Train/SSIMLoss"),
    ("normal_loss", "Train/NormalLoss"),
    ("contribution_loss", "Train/ContributionLoss"),
    ("interpenetration_loss", "Train/InterpenetrationLoss"),
    ("normal_weight", "Train/NormalWeight"),
    ("contribution_weight", "Train/ContributionWeight"),
    ("interpenetration_weight", "Train/InterpenetrationWeight"),
    ("elapsed_s", "Timing/ElapsedSeconds"),
)


def load_direct_powerfoam_training_data(cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    data = load_powerfoam_training_data(cfg, device)
    return {key: data[key] for key in DIRECT_POWERFOAM_DATA_KEYS}


def build_wandb_artifact_payload(
    renders: torch.Tensor,
    alphas: torch.Tensor,
    targets: torch.Tensor,
    cfg: dict[str, Any],
    step: int,
    metrics: dict[str, float],
) -> dict[str, Any]:
    fps = video_fps_from_config(cfg)
    payload: dict[str, Any] = mapped_metric_payload(
        metrics,
        (
            ("eval_l1", "Eval/L1"),
            ("eval_mse", "Eval/MSE"),
        ),
    )
    payload.update(
        mapped_metric_payload(
            metrics,
            (
                ("heldout_eval_l1", "Heldout/EvalL1"),
                ("heldout_eval_mse", "Heldout/EvalMSE"),
            ),
            require=False,
        )
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
    return payload


def log_artifacts(
    model: DirectPowerFoamVideo,
    targets: torch.Tensor,
    frame_indices: torch.Tensor,
    rays: torch.Tensor | None,
    cfg: dict[str, Any],
    step: int,
    output_dir: Path,
    wandb_run: Any | None,
    heldout_targets: torch.Tensor | None = None,
    heldout_frame_indices: torch.Tensor | None = None,
    heldout_rays: torch.Tensor | None = None,
) -> dict[str, float]:
    model.eval()
    renders, alphas = render_powerfoam_samples(
        model,
        frame_indices,
        batch_size=powerfoam_eval_batch_size(cfg),
        rays=rays,
    )
    metrics = reconstruction_l1_mse_metrics(renders, targets.cpu(), prefix="eval")
    heldout_renders = None
    heldout_alphas = None
    if heldout_targets is not None and heldout_frame_indices is not None:
        heldout_renders, heldout_alphas = render_powerfoam_samples(
            model,
            heldout_frame_indices,
            batch_size=powerfoam_eval_batch_size(cfg),
            rays=heldout_rays,
        )
        metrics.update(reconstruction_l1_mse_metrics(heldout_renders, heldout_targets.cpu(), prefix="heldout_eval"))

    save_rgb_alpha_eval_media(
        output_dir,
        step,
        renders,
        targets,
        alphas,
        fps=video_fps_from_config(cfg),
        save_videos=should_log_video(cfg, step),
        heldout_renders=heldout_renders,
        heldout_targets=heldout_targets,
        heldout_alphas=heldout_alphas,
    )
    log_wandb_run_payload_lazy(
        wandb_run,
        lambda: build_wandb_artifact_payload(renders, alphas, targets, cfg, step, metrics),
        step=step,
    )
    model.train()
    return metrics


def run_training(config: dict[str, Any]) -> None:
    cfg = resolve_config(config)
    torch.manual_seed(int(cfg["train"]["seed"]))
    device = resolve_torch_device(str(cfg["train"]["device"]), auto_cuda=False)
    output_dir: Path = cfg["logging"]["output_dir"]
    write_resolved_config(output_dir, cfg)

    training_data = load_direct_powerfoam_training_data(cfg, device)
    targets = training_data["targets"]
    sample_frame_indices = training_data["sample_frame_indices"]
    sample_rays = training_data["sample_rays"]
    heldout_targets = training_data["heldout_targets"]
    heldout_frame_indices = training_data["heldout_frame_indices"]
    heldout_rays = training_data["heldout_rays"]
    cfg["video_fps"] = float(training_data["video_fps"])
    with wandb_run_lifecycle(cfg) as wandb_run:
        model = DirectPowerFoamVideo(
            frame_count=int(training_data["frame_count"]),
            cell_count=int(cfg["model"]["cells"]),
            render_size=int(cfg["render"]["render_size"]),
            fov_degrees=float(cfg["render"]["fov_degrees"]),
            neighbor_count=int(cfg["model"]["neighbor_count"]),
            xy_extent=float(cfg["model"]["xy_extent"]),
            z_min=float(cfg["model"]["z_min"]),
            z_max=float(cfg["model"]["z_max"]),
            radius_init=float(cfg["model"]["radius_init"]),
            radius_min=float(cfg["model"]["radius_min"]),
            density_init=float(cfg["model"]["density_init"]),
            radius_scale=float(cfg["model"]["radius_scale"]),
            seed=int(cfg["train"]["seed"]),
            render_options=direct_powerfoam_render_options(cfg["render"]),
            init_frames=training_data["init_frames"] if bool(cfg["model"]["init_from_video"]) else None,
            image_init_depth=(
                None if cfg["model"]["image_init_depth"] is None else float(cfg["model"]["image_init_depth"])
            ),
            image_init_jitter=float(cfg["model"]["image_init_jitter"]),
            num_texel_sites=int(cfg["model"]["num_texel_sites"]),
            sv_dof=int(cfg["model"]["sv_dof"]),
            sv_axis_init=float(cfg["model"]["sv_axis_init"]),
            adjacency_mode=str(cfg["model"]["adjacency_mode"]),
        ).to(device)
        optimizer = torch.optim.Adam(
            model.optimizer_param_groups(float(cfg["train"]["lr"]))
            if bool(cfg["train"]["use_param_groups"])
            else model.parameters(),
            lr=float(cfg["train"]["lr"]),
        )

        print(
            {
                "arch": "powerfoam_direct",
                "device": str(device),
                "source": str(training_data["source_label"]),
                "frame_source": str(cfg["data"]["frame_source"]),
                "frames": int(training_data["frame_count"]),
                "samples": int(targets.size(0)),
                "train_views": training_data["train_views"],
                "heldout_views": training_data["heldout_views"],
                "pose_source": training_data["pose_source"],
                "render_size": int(cfg["render"]["render_size"]),
                "cells": int(cfg["model"]["cells"]),
                "neighbors": int(cfg["model"]["neighbor_count"]),
                "texel_sites": int(cfg["model"]["num_texel_sites"]),
                "sv_dof": int(cfg["model"]["sv_dof"]),
                "adjacency_mode": str(cfg["model"]["adjacency_mode"]),
                "steps": int(cfg["train"]["steps"]),
            }
        )
        initial_metrics = log_artifacts(
            model,
            targets,
            sample_frame_indices,
            sample_rays,
            cfg,
            0,
            output_dir,
            wandb_run,
            heldout_targets=heldout_targets,
            heldout_frame_indices=heldout_frame_indices,
            heldout_rays=heldout_rays,
        )
        print({"step": 0, **initial_metrics})

        start_time = time.perf_counter()
        progress = trange(1, int(cfg["train"]["steps"]) + 1, desc="powerfoam_direct")
        for step in progress:
            if int(cfg["model"]["rebuild_adjacency_every"]) > 0 and step % int(cfg["model"]["rebuild_adjacency_every"]) == 1:
                model.rebuild_adjacency()
            sample_indices = powerfoam_train_batch_indices(targets.size(0), cfg, device=device)
            frame_indices = sample_frame_indices[sample_indices]
            target = targets[sample_indices]
            ray_batch = None if sample_rays is None else sample_rays[sample_indices]
            render_result = model(frame_indices, target_rgb=target, rays=ray_batch)
            rendered = render_result.rendered
            loss_weights = scheduled_loss_weights(cfg["losses"], step, int(cfg["train"]["steps"]))
            loss, loss_terms = direct_powerfoam_loss(model, rendered, target, render_result, cfg["losses"], loss_weights)

            optimizer_backward_step(optimizer, loss)

            progress.set_postfix(
                loss=f"{float(loss.detach().cpu()):.4f}",
                l1=f"{float(loss_terms['l1'].detach().cpu()):.4f}",
            )
            if should_log_scalar(cfg, step):
                elapsed = time.perf_counter() - start_time
                train_metrics = {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "l1": float(loss_terms["l1"].detach().cpu()),
                    "mse": float(loss_terms["mse"].detach().cpu()),
                    "ssim_loss": float(loss_terms["ssim"].detach().cpu()),
                    "normal_loss": float(loss_terms["normal"].detach().cpu()),
                    "contribution_loss": float(loss_terms["contribution"].detach().cpu()),
                    "interpenetration_loss": float(loss_terms["interpenetration"].detach().cpu()),
                    "normal_weight": loss_weights["normal_weight"],
                    "contribution_weight": loss_weights["contribution_weight"],
                    "interpenetration_weight": loss_weights["interpenetration_weight"],
                    "elapsed_s": elapsed,
                }
                print(train_metrics)
                log_wandb_run_payload(
                    wandb_run,
                    mapped_metric_payload(train_metrics, DIRECT_POWERFOAM_TRAIN_WANDB_KEYS),
                    step=step,
                )
            if should_log_image(cfg, step):
                metrics = log_artifacts(
                    model,
                    targets,
                    sample_frame_indices,
                    sample_rays,
                    cfg,
                    step,
                    output_dir,
                    wandb_run,
                    heldout_targets=heldout_targets,
                    heldout_frame_indices=heldout_frame_indices,
                    heldout_rays=heldout_rays,
                )
                print({"step": step, **metrics})

        save_powerfoam_checkpoint(output_dir / "checkpoint_final.pt", model, cfg)


__all__ = ["run_training"]
