from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import wandb
from torch.nn import functional as F
from tqdm import trange

from config_utils import apply_defaults, load_config_file, resolved_config, serialize_config_value
from dynamic_gauge_foam import (
    DynamicGaugeFoamVideo,
    atlas_total_variation,
    build_knn_edges,
    gauge_connection_loss,
    temporal_accel_loss,
)
from sequence_data import load_video_sequence
from train_logging import build_validation_video_payload, make_preview_image, make_wandb_video
from video_io import save_mp4, save_png


DATA_DEFAULTS = {
    "video_path": "test_data/test_video_small_128_4fps.mp4",
    "frame_source": "explicit_video",
    "max_frames": 16,
}
MODEL_DEFAULTS = {
    "primitives": 512,
    "feature_dim": 8,
    "atlas_res": 4,
    "num_time_ctrl": 8,
    "init_depth": 2.0,
    "radius_scale": 1.65,
    "opacity_init": 0.92,
    "feature_noise": 0.01,
    "color_hidden_dim": 64,
    "rgb_skip": True,
}
RENDER_DEFAULTS = {
    "render_size": 64,
    "fov_degrees": 55.0,
    "chunk_pixels": 1024,
    "max_hits": 8,
    "near": 0.05,
    "far": 100.0,
    "falloff": 2.5,
    "min_alpha": 1.0e-4,
    "background_feature": 0.0,
}
TRAIN_DEFAULTS = {
    "steps": 120,
    "frames_per_step": 1,
    "lr": 0.01,
    "center_lr_multiplier": 0.25,
    "radius_lr_multiplier": 0.1,
    "opacity_lr_multiplier": 0.1,
    "twist_lr_multiplier": 0.35,
    "atlas_lr_multiplier": 1.0,
    "color_lr_multiplier": 0.5,
    "seed": 17,
    "device": "auto",
}
LOSS_DEFAULTS = {
    "l1_weight": 1.0,
    "mse_weight": 0.1,
    "connection_weight": 0.01,
    "temporal_weight": 0.001,
    "opacity_weight": 1.0e-4,
    "radius_weight": 1.0e-4,
    "atlas_tv_weight": 1.0e-4,
    "knn_k": 8,
}
LOGGING_DEFAULTS = {
    "log_every": 30,
    "image_log_every": 60,
    "video_log_every": 120,
    "always_log_last_step": True,
    "output_dir": "outputs/dynamic_gauge_foam/local_mac_dynamic_gauge_foam_video_512_smoke",
    "wandb_enabled": False,
    "wandb_project": "dynaworld",
    "wandb_run_name": "dynamic-gauge-foam-video-512-smoke",
    "wandb_tags": ["dynamic-gauge-foam", "direct-fit", "video"],
    "wandb_mode": None,
}


def resolve_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = resolved_config(config, ("data", "model", "render", "train", "losses", "logging"))
    apply_defaults(cfg["data"], DATA_DEFAULTS)
    apply_defaults(cfg["model"], MODEL_DEFAULTS)
    apply_defaults(cfg["render"], RENDER_DEFAULTS)
    apply_defaults(cfg["train"], TRAIN_DEFAULTS)
    apply_defaults(cfg["losses"], LOSS_DEFAULTS)
    apply_defaults(cfg["logging"], LOGGING_DEFAULTS)
    cfg["data"]["video_path"] = Path(cfg["data"]["video_path"])
    cfg["logging"]["output_dir"] = Path(cfg["logging"]["output_dir"])
    if int(cfg["model"]["primitives"]) < 1:
        raise ValueError("model.primitives must be positive")
    if int(cfg["model"]["feature_dim"]) < 3:
        raise ValueError("model.feature_dim must be at least 3")
    if int(cfg["model"]["atlas_res"]) < 1:
        raise ValueError("model.atlas_res must be positive")
    if int(cfg["model"]["num_time_ctrl"]) < 1:
        raise ValueError("model.num_time_ctrl must be positive")
    if int(cfg["render"]["chunk_pixels"]) < 1:
        raise ValueError("render.chunk_pixels must be positive")
    if int(cfg["render"]["max_hits"]) < 1:
        raise ValueError("render.max_hits must be positive")
    if int(cfg["train"]["steps"]) < 1:
        raise ValueError("train.steps must be positive")
    if int(cfg["train"]["frames_per_step"]) < 1:
        raise ValueError("train.frames_per_step must be positive")
    return cfg


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(value)


def render_kwargs(cfg: dict[str, Any]) -> dict[str, float | int]:
    return {
        "chunk_pixels": int(cfg["render"]["chunk_pixels"]),
        "max_hits": int(cfg["render"]["max_hits"]),
        "near": float(cfg["render"]["near"]),
        "far": float(cfg["render"]["far"]),
        "falloff": float(cfg["render"]["falloff"]),
        "min_alpha": float(cfg["render"]["min_alpha"]),
        "background_feature": float(cfg["render"]["background_feature"]),
    }


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
def render_all(model: DynamicGaugeFoamVideo, frame_count: int, cfg: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    renders = []
    alphas = []
    depths = []
    for frame_index in range(frame_count):
        indices = torch.tensor([frame_index], device=device, dtype=torch.long)
        out = model(indices, **render_kwargs(cfg))
        renders.append(out.rgb.permute(0, 3, 1, 2).detach().cpu())
        alphas.append(out.alpha[..., 0].detach().cpu())
        depths.append(out.depth[..., 0].detach().cpu())
    return torch.cat(renders, dim=0), torch.cat(alphas, dim=0), torch.cat(depths, dim=0)


def log_artifacts(
    model: DynamicGaugeFoamVideo,
    targets: torch.Tensor,
    cfg: dict[str, Any],
    step: int,
    output_dir: Path,
    wandb_run: Any | None,
) -> dict[str, float]:
    model.eval()
    renders, alphas, depths = render_all(model, targets.size(0), cfg)
    metrics = {
        "eval_l1": F.l1_loss(renders, targets.cpu()).item(),
        "eval_mse": F.mse_loss(renders, targets.cpu()).item(),
        "eval_alpha_mean": float(alphas.mean().item()),
        "eval_alpha_max": float(alphas.max().item()),
    }
    metrics.update(model.state_metrics())
    preview = torch.cat([targets[0].cpu(), renders[0], alphas[0].unsqueeze(0).repeat(3, 1, 1)], dim=-1)
    save_png(output_dir / f"preview_step_{step:04d}.png", preview)
    if should_log_video(cfg, step):
        fps = float(cfg.get("video_fps", 4.0))
        save_mp4(output_dir / f"render_step_{step:04d}.mp4", renders, fps=fps)
        save_mp4(output_dir / f"side_by_side_step_{step:04d}.mp4", torch.cat([targets.cpu(), renders], dim=-1), fps=fps)
    if wandb_run is not None:
        fps = float(cfg.get("video_fps", 4.0))
        payload: dict[str, Any] = {
            "Eval/L1": metrics["eval_l1"],
            "Eval/MSE": metrics["eval_mse"],
            "Eval/AlphaMean": metrics["eval_alpha_mean"],
            "Eval/AlphaMax": metrics["eval_alpha_max"],
            "State/MeanCenterDelta": metrics["state_mean_center_delta"],
            "State/P95CenterDelta": metrics["state_p95_center_delta"],
            "State/MaxCenterDelta": metrics["state_max_center_delta"],
            "State/MeanRadius": metrics["state_mean_radius"],
            "State/MeanRadiusDelta": metrics["state_mean_radius_delta"],
            "State/MeanOpacity": metrics["state_mean_opacity"],
            "State/MeanAtlasDelta": metrics["state_mean_atlas_delta"],
            "State/MeanTwistAbs": metrics["state_mean_twist_abs"],
            "Preview": make_preview_image(targets[0].cpu(), renders[0], caption=f"step {step}: GT | render"),
        }
        if should_log_video(cfg, step):
            payload.update(build_validation_video_payload(renders, targets.cpu(), fps))
            payload["GT_Video"] = make_wandb_video(targets.cpu(), fps)
            payload["Alpha_Video"] = make_wandb_video(alphas.unsqueeze(1).repeat(1, 3, 1, 1), fps)
            depth_vis = depths / depths[depths > 0.0].median().clamp_min(1.0e-6) if bool((depths > 0.0).any()) else depths
            payload["Depth_Video"] = make_wandb_video(depth_vis.clamp(0.0, 1.0).unsqueeze(1).repeat(1, 3, 1, 1), fps)
        wandb_run.log(payload, step=step)
    model.train()
    return metrics


def run_training(config: dict[str, Any]) -> None:
    cfg = resolve_config(config)
    torch.manual_seed(int(cfg["train"]["seed"]))
    device = resolve_device(str(cfg["train"]["device"]))

    output_dir: Path = cfg["logging"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(json.dumps(serialize_config_value(cfg), indent=2) + "\n")

    sequence = load_video_sequence(
        cfg["data"]["video_path"],
        target_size=int(cfg["render"]["render_size"]),
        max_frames=int(cfg["data"]["max_frames"]),
        frame_source=str(cfg["data"]["frame_source"]),
    )
    targets = sequence.frames.to(device=device, dtype=torch.float32)
    frame_times = sequence.frame_times.reshape(-1).to(device=device, dtype=torch.float32)
    cfg["video_fps"] = float(sequence.video_fps)
    wandb_run = init_wandb_run(cfg)

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
        frame_indices = torch.randint(0, targets.size(0), (int(cfg["train"]["frames_per_step"]),), device=device)
        out = model(frame_indices, **render_kwargs(cfg))
        rendered = out.rgb.permute(0, 3, 1, 2)
        target = targets[frame_indices]
        l1 = F.l1_loss(rendered, target)
        mse = F.mse_loss(rendered, target)
        foam = model.evaluate_times(model.frame_times[frame_indices])
        dt = 1.0 / max(int(cfg["model"]["num_time_ctrl"]) - 1, 1)
        t_mid = model.frame_times[frame_indices]
        prev_foam = model.evaluate_times((t_mid - dt).clamp(0.0, 1.0))
        next_foam = model.evaluate_times((t_mid + dt).clamp(0.0, 1.0))
        loss_conn = gauge_connection_loss(foam.centers, foam.rotations, model.p0.detach(), edge_index)
        loss_temp = temporal_accel_loss(prev_foam.centers, foam.centers, next_foam.centers)
        loss_opacity = foam.opacities.mean()
        loss_radius = foam.radii.square().mean()
        loss_tv = atlas_total_variation(model.atlas)
        loss = (
            float(cfg["losses"]["l1_weight"]) * l1
            + float(cfg["losses"]["mse_weight"]) * mse
            + float(cfg["losses"]["connection_weight"]) * loss_conn
            + float(cfg["losses"]["temporal_weight"]) * loss_temp
            + float(cfg["losses"]["opacity_weight"]) * loss_opacity
            + float(cfg["losses"]["radius_weight"]) * loss_radius
            + float(cfg["losses"]["atlas_tv_weight"]) * loss_tv
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        progress.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}", l1=f"{float(l1.detach().cpu()):.4f}")
        if step % int(cfg["logging"]["log_every"]) == 0:
            elapsed = time.perf_counter() - start_time
            train_metrics = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "l1": float(l1.detach().cpu()),
                "mse": float(mse.detach().cpu()),
                "conn": float(loss_conn.detach().cpu()),
                "temp": float(loss_temp.detach().cpu()),
                "elapsed_s": elapsed,
            }
            print(train_metrics)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "Train/Loss": train_metrics["loss"],
                        "Train/L1": train_metrics["l1"],
                        "Train/MSE": train_metrics["mse"],
                        "Train/Connection": train_metrics["conn"],
                        "Train/Temporal": train_metrics["temp"],
                        "Timing/ElapsedSeconds": elapsed,
                    },
                    step=step,
                )
        if step % int(cfg["logging"]["image_log_every"]) == 0 or (
            bool(cfg["logging"]["always_log_last_step"]) and step == int(cfg["train"]["steps"])
        ):
            metrics = log_artifacts(model, targets, cfg, step, output_dir, wandb_run)
            print({"step": step, **metrics})

    torch.save({"model": model.state_dict(), "config": serialize_config_value(cfg)}, output_dir / "checkpoint_final.pt")
    if wandb_run is not None:
        wandb_run.finish()


def main() -> None:
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("Usage: PYTHONPATH=src/train uv run python src/train/train_dynamic_gauge_foam.py <config.jsonc>")
    run_training(load_config_file(sys.argv[1]))


if __name__ == "__main__":
    main()
