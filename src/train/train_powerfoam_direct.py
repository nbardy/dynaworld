from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import wandb
from torch.nn import functional as F
from tqdm import trange

from camera import CameraSpec, build_camera_rays
from checkpoint_utils import atomic_torch_save
from config_utils import apply_defaults, load_config_file, resolved_config, serialize_config_value
from losses import ssim_per_image
from multicam_video_data import cameras_from_K_w2c, heldout_cameras_from_K_w2c, load_multicam_video_bundle
from powerfoam_direct import DirectPowerFoamVideo, PowerFoamRenderOptions
from sequence_data import load_video_sequence
from train_logging import build_validation_video_payload, make_preview_image, make_wandb_video
from video_io import save_mp4, save_png


DATA_DEFAULTS = {
    "video_path": "test_data/test_video_small_128_4fps.mp4",
    "frame_source": "explicit_video",
    "max_frames": 16,
}
MODEL_DEFAULTS = {
    "cells": 96,
    "neighbor_count": 16,
    "init_from_video": False,
    "image_init_depth": None,
    "image_init_jitter": 0.0,
    "num_texel_sites": 8,
    "sv_dof": 8,
    "sv_axis_init": 8.0,
    "radius_scale": 0.75,
    "adjacency_mode": "cech_aabb",
    "rebuild_adjacency_every": 10,
    "xy_extent": 1.25,
    "z_min": 1.0,
    "z_max": 3.25,
    "radius_init": 0.45,
    "radius_min": 0.03,
    "density_init": 36.0,
}
RENDER_DEFAULTS = {
    "render_size": 128,
    "fov_degrees": 55.0,
    "near_plane": 0.05,
    "alpha_threshold": 0.0,
    "transmittance_threshold": 1.0e-4,
    "max_alpha": 0.99,
    "eps": 1.0e-6,
    "texel_temperature": 10.0,
    "background": [0.0, 0.0, 0.0],
}
TRAIN_DEFAULTS = {
    "steps": 250,
    "frames_per_step": 1,
    "lr": 0.03,
    "use_param_groups": True,
    "seed": 17,
    "device": "auto",
}
LOSS_DEFAULTS = {
    "l1_weight": 1.0,
    "mse_weight": 0.1,
    "rgb_mse_sum_weight": 0.0,
    "ssim_weight": 0.2,
    "ssim_window_size": 11,
    "ssim_c1": 0.0001,
    "ssim_c2": 0.0009,
    "normal_weight": 0.1,
    "normal_weight_final_multiplier": 0.1,
    "contribution_weight": 0.1,
    "contribution_weight_final_multiplier": 0.001,
    "interpenetration_weight": 1.0e-4,
    "interpenetration_weight_final_multiplier": 0.001,
    "radius_l2_weight": 1.0e-4,
    "density_l2_weight": 0.0,
}
LOGGING_DEFAULTS = {
    "log_every": 10,
    "image_log_every": 50,
    "video_log_every": 100,
    "always_log_last_step": True,
    "output_dir": "outputs/powerfoam_direct/local_mac_powerfoam_direct_128_smoke",
    "wandb_enabled": False,
    "wandb_project": "dynaworld",
    "wandb_run_name": "powerfoam-direct-128-smoke",
    "wandb_tags": ["powerfoam", "direct-fit", "128px"],
    "wandb_mode": None,
}


def resolve_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = resolved_config(config, ("data", "model", "render", "train", "losses", "logging"))
    cfg.setdefault("camera", {})
    apply_defaults(cfg["data"], DATA_DEFAULTS)
    apply_defaults(cfg["model"], MODEL_DEFAULTS)
    apply_defaults(cfg["render"], RENDER_DEFAULTS)
    apply_defaults(cfg["train"], TRAIN_DEFAULTS)
    apply_defaults(cfg["losses"], LOSS_DEFAULTS)
    apply_defaults(cfg["logging"], LOGGING_DEFAULTS)
    if cfg["data"]["video_path"] is not None:
        cfg["data"]["video_path"] = Path(cfg["data"]["video_path"])
    cfg["logging"]["output_dir"] = Path(cfg["logging"]["output_dir"])

    if int(cfg["model"]["cells"]) < 1:
        raise ValueError("model.cells must be positive")
    if int(cfg["model"]["num_texel_sites"]) < 1:
        raise ValueError("model.num_texel_sites must be positive")
    if int(cfg["model"]["sv_dof"]) < 1:
        raise ValueError("model.sv_dof must be positive")
    if int(cfg["model"]["neighbor_count"]) >= int(cfg["model"]["cells"]):
        cfg["model"]["neighbor_count"] = int(cfg["model"]["cells"]) - 1
    if str(cfg["model"]["adjacency_mode"]) not in {"overlap", "knn", "cech_aabb"}:
        raise ValueError("model.adjacency_mode must be 'overlap', 'knn', or 'cech_aabb'")
    if int(cfg["train"]["frames_per_step"]) < 1:
        raise ValueError("train.frames_per_step must be positive")
    if int(cfg["train"]["steps"]) < 1:
        raise ValueError("train.steps must be positive")
    background = cfg["render"]["background"]
    if len(background) != 3:
        raise ValueError("render.background must have exactly 3 values")
    return cfg


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(value)


def make_render_options(render_cfg: dict[str, Any]) -> PowerFoamRenderOptions:
    return PowerFoamRenderOptions(
        near_plane=float(render_cfg["near_plane"]),
        alpha_threshold=float(render_cfg["alpha_threshold"]),
        transmittance_threshold=float(render_cfg["transmittance_threshold"]),
        max_alpha=float(render_cfg["max_alpha"]),
        eps=float(render_cfg["eps"]),
        texel_temperature=float(render_cfg["texel_temperature"]),
        background=tuple(float(v) for v in render_cfg["background"]),
    )


def powerfoam_rays_from_camera(
    camera: CameraSpec,
    *,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    origins, directions = build_camera_rays(camera, height, width, device=device, dtype=dtype)
    return torch.cat([origins, directions], dim=-1).unsqueeze(0).contiguous()


def powerfoam_rays_from_camera_grid(
    cameras: tuple[tuple[CameraSpec, ...], ...],
    *,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if not cameras:
        raise ValueError("Expected at least one camera view.")
    per_view = []
    for view_cameras in cameras:
        if not view_cameras:
            raise ValueError("Expected at least one frame camera per view.")
        per_view.append(
            torch.cat(
                [
                    powerfoam_rays_from_camera(
                        camera,
                        height=height,
                        width=width,
                        device=device,
                        dtype=dtype,
                    )
                    for camera in view_cameras
                ],
                dim=0,
            )
        )
    return torch.stack(per_view, dim=0).contiguous()


def flatten_multiview_powerfoam_samples(
    frames: torch.Tensor,
    rays: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if frames.ndim != 5:
        raise ValueError(f"Expected multiview frames [V,T,C,H,W], got {tuple(frames.shape)}.")
    if rays.ndim != 5:
        raise ValueError(f"Expected multiview rays [V,T,H,W,6], got {tuple(rays.shape)}.")
    view_count, frame_count = int(frames.shape[0]), int(frames.shape[1])
    if tuple(rays.shape[:2]) != (view_count, frame_count):
        raise ValueError(f"Frame/ray view-time mismatch: {tuple(frames.shape[:2])} vs {tuple(rays.shape[:2])}.")
    targets = frames.reshape(view_count * frame_count, *frames.shape[2:]).contiguous()
    sample_frame_indices = torch.arange(frame_count, device=frames.device, dtype=torch.long).repeat(view_count)
    sample_rays = rays.reshape(view_count * frame_count, *rays.shape[2:]).contiguous()
    return targets, sample_frame_indices, sample_rays


def load_direct_powerfoam_training_data(cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    render_size = int(cfg["render"]["render_size"])
    frame_source = str(cfg["data"]["frame_source"])
    if frame_source == "multicam_val":
        bundle = load_multicam_video_bundle(
            data_cfg=cfg["data"],
            camera_cfg=cfg["camera"],
            target_size=render_size,
            device=device,
        )
        train_rays = powerfoam_rays_from_camera_grid(
            cameras_from_K_w2c(
                bundle.train_K,
                bundle.train_w2c,
                lens_models=bundle.train_lens_models,
                distortions=bundle.train_distortions,
            ),
            height=render_size,
            width=render_size,
            device=device,
        )
        targets, sample_frame_indices, sample_rays = flatten_multiview_powerfoam_samples(
            bundle.train_frames.to(device=device, dtype=torch.float32),
            train_rays,
        )
        heldout_targets = None
        heldout_frame_indices = None
        heldout_rays = None
        if bundle.heldout_frames is not None and bundle.heldout_K is not None and bundle.heldout_w2c is not None:
            heldout_rays_grid = powerfoam_rays_from_camera_grid(
                heldout_cameras_from_K_w2c(
                    bundle.heldout_K,
                    bundle.heldout_w2c,
                    lens_models=bundle.heldout_lens_models,
                    distortions=bundle.heldout_distortions,
                ),
                height=render_size,
                width=render_size,
                device=device,
            )
            heldout_targets, heldout_frame_indices, heldout_rays = flatten_multiview_powerfoam_samples(
                bundle.heldout_frames.to(device=device, dtype=torch.float32),
                heldout_rays_grid,
            )
        return {
            "targets": targets,
            "sample_frame_indices": sample_frame_indices,
            "sample_rays": sample_rays,
            "heldout_targets": heldout_targets,
            "heldout_frame_indices": heldout_frame_indices,
            "heldout_rays": heldout_rays,
            "init_frames": bundle.condition_sequence.frames.detach().cpu(),
            "frame_count": bundle.frame_count,
            "video_fps": float(bundle.condition_sequence.video_fps),
            "source_label": str(bundle.metadata.get("sample_id")) if bundle.metadata else "multicam_val",
            "train_views": bundle.train_camera_names,
            "heldout_views": bundle.heldout_camera_names or [],
            "pose_source": bundle.pose_source,
        }

    if cfg["data"]["video_path"] is None:
        raise ValueError("data.video_path is required unless data.frame_source is 'multicam_val'.")
    sequence = load_video_sequence(
        cfg["data"]["video_path"],
        target_size=render_size,
        max_frames=int(cfg["data"]["max_frames"]),
        frame_source=frame_source,
    )
    targets = sequence.frames.to(device=device, dtype=torch.float32)
    return {
        "targets": targets,
        "sample_frame_indices": torch.arange(targets.size(0), device=device, dtype=torch.long),
        "sample_rays": None,
        "heldout_targets": None,
        "heldout_frame_indices": None,
        "heldout_rays": None,
        "init_frames": targets.detach().cpu(),
        "frame_count": int(targets.size(0)),
        "video_fps": float(sequence.video_fps),
        "source_label": str(cfg["data"]["video_path"]),
        "train_views": [],
        "heldout_views": [],
        "pose_source": None,
    }


def should_log_video(cfg: dict[str, Any], step: int) -> bool:
    return step % int(cfg["logging"]["video_log_every"]) == 0 or (
        bool(cfg["logging"]["always_log_last_step"]) and step == int(cfg["train"]["steps"])
    )


def init_wandb_run(cfg: dict[str, Any]) -> Any | None:
    if not bool(cfg["logging"]["wandb_enabled"]):
        return None
    init_kwargs = {
        "project": cfg["logging"]["wandb_project"],
        "name": cfg["logging"]["wandb_run_name"],
        "tags": cfg["logging"]["wandb_tags"],
        "config": serialize_config_value(cfg),
    }
    if cfg["logging"]["wandb_mode"] is not None:
        init_kwargs["mode"] = str(cfg["logging"]["wandb_mode"])
    return wandb.init(**init_kwargs)


@torch.no_grad()
def render_all(
    model: DirectPowerFoamVideo,
    frame_indices: torch.Tensor,
    rays: torch.Tensor | None,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    renders = []
    alphas = []
    device = next(model.parameters()).device
    frame_indices = frame_indices.to(device=device, dtype=torch.long)
    ray_data = None if rays is None else rays.to(device=device, dtype=torch.float32)
    sample_count = int(frame_indices.numel())
    for start in range(0, sample_count, batch_size):
        indices = frame_indices[start : min(start + batch_size, sample_count)]
        ray_batch = None if ray_data is None else ray_data[start : min(start + batch_size, sample_count)]
        rendered, alpha, _, _ = model(indices, rays=ray_batch)
        renders.append(rendered.detach().cpu())
        alphas.append(alpha.detach().cpu())
    return torch.cat(renders, dim=0), torch.cat(alphas, dim=0)


def build_wandb_artifact_payload(
    renders: torch.Tensor,
    alphas: torch.Tensor,
    targets: torch.Tensor,
    cfg: dict[str, Any],
    step: int,
    metrics: dict[str, float],
) -> dict[str, Any]:
    fps = float(cfg.get("video_fps", 4.0))
    payload: dict[str, Any] = {
        "Eval/L1": metrics["eval_l1"],
        "Eval/MSE": metrics["eval_mse"],
        "Preview": make_preview_image(targets[0].cpu(), renders[0], caption=f"step {step}: GT | render"),
    }
    if "heldout_eval_l1" in metrics:
        payload["Heldout/EvalL1"] = metrics["heldout_eval_l1"]
        payload["Heldout/EvalMSE"] = metrics["heldout_eval_mse"]
    if should_log_video(cfg, step):
        payload.update(build_validation_video_payload(renders, targets.cpu(), fps))
        payload["GT_Video"] = make_wandb_video(targets.cpu(), fps)
        payload["Alpha_Video"] = make_wandb_video(alphas.unsqueeze(1).repeat(1, 3, 1, 1), fps)
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
    renders, alphas = render_all(
        model,
        frame_indices,
        rays,
        batch_size=max(1, int(cfg["train"]["frames_per_step"])),
    )
    metrics = {
        "eval_l1": F.l1_loss(renders, targets.cpu()).item(),
        "eval_mse": F.mse_loss(renders, targets.cpu()).item(),
    }
    heldout_renders = None
    heldout_alphas = None
    if heldout_targets is not None and heldout_frame_indices is not None:
        heldout_renders, heldout_alphas = render_all(
            model,
            heldout_frame_indices,
            heldout_rays,
            batch_size=max(1, int(cfg["train"]["frames_per_step"])),
        )
        metrics["heldout_eval_l1"] = F.l1_loss(heldout_renders, heldout_targets.cpu()).item()
        metrics["heldout_eval_mse"] = F.mse_loss(heldout_renders, heldout_targets.cpu()).item()

    preview = torch.cat([targets[0].cpu(), renders[0], alphas[0].unsqueeze(0).repeat(3, 1, 1)], dim=-1)
    save_png(output_dir / f"preview_step_{step:04d}.png", preview)
    if heldout_renders is not None and heldout_alphas is not None and heldout_targets is not None:
        heldout_preview = torch.cat(
            [heldout_targets.detach().cpu()[0], heldout_renders[0], heldout_alphas[0].unsqueeze(0).repeat(3, 1, 1)],
            dim=-1,
        )
        save_png(output_dir / f"heldout_preview_step_{step:04d}.png", heldout_preview)

    if should_log_video(cfg, step):
        side_by_side = torch.cat([targets.cpu(), renders], dim=-1)
        save_mp4(output_dir / f"render_step_{step:04d}.mp4", renders, fps=float(cfg.get("video_fps", 4.0)))
        save_mp4(output_dir / f"side_by_side_step_{step:04d}.mp4", side_by_side, fps=float(cfg.get("video_fps", 4.0)))
        if heldout_renders is not None and heldout_targets is not None:
            heldout_side_by_side = torch.cat([heldout_targets.detach().cpu(), heldout_renders], dim=-1)
            save_mp4(output_dir / f"heldout_render_step_{step:04d}.mp4", heldout_renders, fps=float(cfg.get("video_fps", 4.0)))
            save_mp4(output_dir / f"heldout_side_by_side_step_{step:04d}.mp4", heldout_side_by_side, fps=float(cfg.get("video_fps", 4.0)))
    if wandb_run is not None:
        wandb_run.log(build_wandb_artifact_payload(renders, alphas, targets, cfg, step, metrics), step=step)
    model.train()
    return metrics


def exp_scheduled_weight(initial: float, final_multiplier: float, step: int, total_steps: int) -> float:
    initial = float(initial)
    if initial <= 0.0:
        return initial
    final = initial * float(final_multiplier)
    if final <= 0.0:
        return final
    t = min(max(float(step) / max(float(total_steps), 1.0), 0.0), 1.0)
    return float(math.exp(math.log(initial) * (1.0 - t) + math.log(final) * t))


def scheduled_loss_weights(loss_cfg: dict[str, Any], step: int, total_steps: int) -> dict[str, float]:
    return {
        "l1_weight": float(loss_cfg["l1_weight"]),
        "mse_weight": float(loss_cfg["mse_weight"]),
        "rgb_mse_sum_weight": float(loss_cfg["rgb_mse_sum_weight"]),
        "ssim_weight": float(loss_cfg["ssim_weight"]),
        "normal_weight": exp_scheduled_weight(
            float(loss_cfg["normal_weight"]),
            float(loss_cfg["normal_weight_final_multiplier"]),
            step,
            total_steps,
        ),
        "contribution_weight": exp_scheduled_weight(
            float(loss_cfg["contribution_weight"]),
            float(loss_cfg["contribution_weight_final_multiplier"]),
            step,
            total_steps,
        ),
        "interpenetration_weight": exp_scheduled_weight(
            float(loss_cfg["interpenetration_weight"]),
            float(loss_cfg["interpenetration_weight_final_multiplier"]),
            step,
            total_steps,
        ),
        "radius_l2_weight": float(loss_cfg["radius_l2_weight"]),
        "density_l2_weight": float(loss_cfg["density_l2_weight"]),
    }


def compute_powerfoam_loss(
    model: DirectPowerFoamVideo,
    rendered: torch.Tensor,
    target: torch.Tensor,
    render_result: Any,
    loss_cfg: dict[str, Any],
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    l1 = F.l1_loss(rendered, target)
    mse = F.mse_loss(rendered, target)
    rgb_mse_sum = (rendered - target).square().sum(dim=1).mean()
    ssim_loss = 1.0 - ssim_per_image(
        rendered,
        target,
        window_size=int(loss_cfg["ssim_window_size"]),
        c1=float(loss_cfg["ssim_c1"]),
        c2=float(loss_cfg["ssim_c2"]),
    ).mean()
    normal_loss = render_result.normal_distance.mean()
    contribution_loss = render_result.contrib.sum(dim=1).mean()
    interpenetration_loss = model.interpenetration().sum(dim=1).mean()
    _, radii, densities, _ = model.decoded_parameters()
    radius_l2 = radii.square().mean()
    density_l2 = densities.square().mean()

    terms = {
        "l1": l1,
        "mse": mse,
        "rgb_mse_sum": rgb_mse_sum,
        "ssim": ssim_loss,
        "normal": normal_loss,
        "contribution": contribution_loss,
        "interpenetration": interpenetration_loss,
        "radius_l2": radius_l2,
        "density_l2": density_l2,
    }
    loss = (
        weights["l1_weight"] * l1
        + weights["mse_weight"] * mse
        + weights["rgb_mse_sum_weight"] * rgb_mse_sum
        + weights["ssim_weight"] * ssim_loss
        + weights["normal_weight"] * normal_loss
        + weights["contribution_weight"] * contribution_loss
        + weights["interpenetration_weight"] * interpenetration_loss
        + weights["radius_l2_weight"] * radius_l2
        + weights["density_l2_weight"] * density_l2
    )
    return loss, terms


def run_training(config: dict[str, Any]) -> None:
    cfg = resolve_config(config)
    torch.manual_seed(int(cfg["train"]["seed"]))
    device = resolve_device(str(cfg["train"]["device"]))
    output_dir: Path = cfg["logging"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(json.dumps(serialize_config_value(cfg), indent=2) + "\n")

    training_data = load_direct_powerfoam_training_data(cfg, device)
    targets = training_data["targets"]
    sample_frame_indices = training_data["sample_frame_indices"]
    sample_rays = training_data["sample_rays"]
    heldout_targets = training_data["heldout_targets"]
    heldout_frame_indices = training_data["heldout_frame_indices"]
    heldout_rays = training_data["heldout_rays"]
    cfg["video_fps"] = float(training_data["video_fps"])
    wandb_run = init_wandb_run(cfg)

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
        render_options=make_render_options(cfg["render"]),
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
        sample_indices = torch.randint(0, targets.size(0), (int(cfg["train"]["frames_per_step"]),), device=device)
        frame_indices = sample_frame_indices[sample_indices]
        target = targets[sample_indices]
        ray_batch = None if sample_rays is None else sample_rays[sample_indices]
        render_result = model(frame_indices, target_rgb=target, rays=ray_batch)
        rendered = render_result.rendered
        loss_weights = scheduled_loss_weights(cfg["losses"], step, int(cfg["train"]["steps"]))
        loss, loss_terms = compute_powerfoam_loss(model, rendered, target, render_result, cfg["losses"], loss_weights)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        progress.set_postfix(
            loss=f"{float(loss.detach().cpu()):.4f}",
            l1=f"{float(loss_terms['l1'].detach().cpu()):.4f}",
        )
        if step % int(cfg["logging"]["log_every"]) == 0:
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
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "Train/Loss": train_metrics["loss"],
                        "Train/L1": train_metrics["l1"],
                        "Train/MSE": train_metrics["mse"],
                        "Train/SSIMLoss": train_metrics["ssim_loss"],
                        "Train/NormalLoss": train_metrics["normal_loss"],
                        "Train/ContributionLoss": train_metrics["contribution_loss"],
                        "Train/InterpenetrationLoss": train_metrics["interpenetration_loss"],
                        "Train/NormalWeight": train_metrics["normal_weight"],
                        "Train/ContributionWeight": train_metrics["contribution_weight"],
                        "Train/InterpenetrationWeight": train_metrics["interpenetration_weight"],
                        "Timing/ElapsedSeconds": elapsed,
                    },
                    step=step,
                )
        if step % int(cfg["logging"]["image_log_every"]) == 0 or (
            bool(cfg["logging"]["always_log_last_step"]) and step == int(cfg["train"]["steps"])
        ):
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

    atomic_torch_save(
        {
            "model": model.state_dict(),
            "config": serialize_config_value(cfg),
        },
        output_dir / "checkpoint_final.pt",
    )
    if wandb_run is not None:
        wandb_run.finish()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: PYTHONPATH=src/train uv run python src/train/train_powerfoam_direct.py <config.jsonc>"
        )
    run_training(load_config_file(sys.argv[1]))


if __name__ == "__main__":
    main()
