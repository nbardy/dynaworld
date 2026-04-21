import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import wandb
from config_utils import (
    apply_defaults,
    format_key_values,
    load_config_file,
    path_or_none,
    resolved_config,
    select_keys,
    serialize_config_value,
)
from debug_metrics import (
    dense_render_diagnostics,
    format_metric_summary,
    metric_config_from_logging,
    optimizer_diagnostics,
    print_metric_summary,
    render_aux_diagnostics,
)
from fast_attn import configure_fast_attn, fast_attn_context
from gs_models import DynamicTokenGS
from losses import reconstruction_loss_per_image, ssim_per_image
from renderers.common import build_pixel_grid
from rendering import pick_renderer_mode as resolve_renderer_mode
from rendering import render_gaussian_frame, render_gaussian_frames
from runtime_types import GaussianFrame, SequenceData
from sequence_data import load_camera_sequence, select_window_indices
from tqdm import tqdm
from train_logging import build_validation_video_payload, make_preview_image, make_wandb_video

PREBAKED_VIDEO_VARIANTS = {
    "small_32_2fps": {
        "sequence_dir": Path("test_data/dust3r_outputs/test_video_small_all_frames"),
        "model_size": 32,
    },
    "small_64_4fps": {
        "sequence_dir": Path("test_data/dust3r_outputs/test_video_small_64_4fps_all_frames"),
        "model_size": 64,
    },
    "small_128_4fps": {
        "sequence_dir": Path("test_data/dust3r_outputs/test_video_small_128_4fps_all_frames"),
        "model_size": 128,
    },
}

MODEL_OPTION_DEFAULTS = {
    "xy_extent": 1.5,
    "z_min": 0.5,
    "z_max": 2.5,
    "scale_init": 0.05,
    "scale_init_log_jitter": 0.0,
    "opacity_init": None,
    "token_init_std": 1.0,
    "head_hidden_dim": 64,
    "head_hidden_layers": 1,
    "head_output_init_std": None,
    "position_init_extent_coverage": 0.0,
    "rotation_init": "random",
}
MODEL_CONSTRUCTOR_KEYS = (
    "xy_extent",
    "z_min",
    "z_max",
    "scale_init",
    "scale_init_log_jitter",
    "opacity_init",
    "token_init_std",
    "head_hidden_dim",
    "head_hidden_layers",
    "head_output_init_std",
    "position_init_extent_coverage",
    "rotation_init",
)
GAUSSIAN_SUMMARY_KEYS = (
    "xy_extent",
    "z_min",
    "z_max",
    "scale_init",
    "scale_init_log_jitter",
    "opacity_init",
    "token_init_std",
    "head_output_init_std",
    "position_init_extent_coverage",
)
RENDER_SUMMARY_KEYS = (
    "renderer",
    "tile_size",
    "near_plane",
    "alpha_threshold",
)
RENDER_OPTION_DEFAULTS = {
    "taichi": None,
    "fast_mac": None,
    "near_plane": 0.05,
}
OPTIMIZER_DEFAULTS = {
    "type": "adam",
    "weight_decay": 0.0,
    "exclude_bias_norm": False,
    "lr_multipliers": {},
    "fused": None,
}
TRAIN_OPTION_DEFAULTS = {
    "clip_grad_norm": None,
}
LOGGING_OPTION_DEFAULTS = {
    "wandb_tags": None,
}
LOSS_OPTION_DEFAULTS = {
    "type": "standard_gs",
    "l1_weight": 0.8,
    "dssim_weight": 0.2,
    "mse_weight": 0.2,
    "ssim_window_size": 11,
    "ssim_c1": 0.0001,
    "ssim_c2": 0.0009,
}


def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def resolve_sequence_dir(data_cfg: dict[str, Any]) -> Path:
    sequence_dir = path_or_none(data_cfg["sequence_dir"])
    if sequence_dir is not None:
        return sequence_dir

    video_variant = data_cfg["video_variant"]
    if video_variant not in PREBAKED_VIDEO_VARIANTS:
        known = ", ".join(sorted(PREBAKED_VIDEO_VARIANTS))
        raise ValueError(f"Unknown prebaked video_variant={video_variant!r}. Expected one of: {known}.")
    return PREBAKED_VIDEO_VARIANTS[video_variant]["sequence_dir"]


def normalize_lr_schedule(train_cfg: dict[str, Any]) -> None:
    if "lr_schedule" not in train_cfg or train_cfg["lr_schedule"] is None:
        train_cfg["lr_schedule"] = {"type": "constant"}

    schedule_cfg = train_cfg["lr_schedule"]
    schedule_type = str(schedule_cfg["type"]).lower()
    schedule_cfg["type"] = schedule_type
    if schedule_type == "cosine":
        apply_defaults(schedule_cfg, {"final_lr_scale": 0.1})
    elif schedule_type not in {"constant", "none", "off"}:
        raise ValueError(f"Unknown lr_schedule.type={schedule_type!r}. Expected 'constant' or 'cosine'.")


def normalize_optimizer_config(train_cfg: dict[str, Any]) -> None:
    if "optimizer" not in train_cfg or train_cfg["optimizer"] is None:
        train_cfg["optimizer"] = {}
    apply_defaults(train_cfg["optimizer"], OPTIMIZER_DEFAULTS)
    train_cfg["optimizer"]["type"] = str(train_cfg["optimizer"]["type"]).lower()


def normalize_loss_config(cfg: dict[str, Any]) -> None:
    if "losses" not in cfg or cfg["losses"] is None:
        cfg["losses"] = {}
    apply_defaults(cfg["losses"], LOSS_OPTION_DEFAULTS)
    loss_cfg = cfg["losses"]
    loss_cfg["type"] = str(loss_cfg["type"]).lower()
    if loss_cfg["type"] not in {"standard_gs", "l1_mse", "l1", "mse"}:
        raise ValueError(f"Unknown losses.type={loss_cfg['type']!r}. Expected one of: standard_gs, l1_mse, l1, mse.")
    window_size = int(loss_cfg["ssim_window_size"])
    if window_size < 1 or window_size % 2 != 1:
        raise ValueError(f"losses.ssim_window_size must be a positive odd integer, got {window_size}.")
    loss_cfg["ssim_window_size"] = window_size


def normalize_render_config(cfg: dict[str, Any]) -> None:
    apply_defaults(cfg["render"], RENDER_OPTION_DEFAULTS)
    near_plane = float(cfg["render"]["near_plane"])
    if near_plane <= 0.0:
        raise ValueError(f"render.near_plane must be positive, got {near_plane}.")
    cfg["render"]["near_plane"] = near_plane


def resolve_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = resolved_config(config, ("data", "model", "render", "train", "logging"))
    cfg["data"]["sequence_dir"] = resolve_sequence_dir(cfg["data"])
    cfg["data"]["camera_json"] = path_or_none(cfg["data"]["camera_json"])
    if "position_init_raw_jitter" in cfg["model"]:
        if "position_init_extent_coverage" in cfg["model"]:
            raise ValueError(
                "Config cannot set both model.position_init_raw_jitter and model.position_init_extent_coverage."
            )
        cfg["model"]["position_init_extent_coverage"] = cfg["model"].pop("position_init_raw_jitter")
    apply_defaults(cfg["model"], MODEL_OPTION_DEFAULTS)
    normalize_render_config(cfg)
    apply_defaults(cfg["train"], TRAIN_OPTION_DEFAULTS)
    normalize_lr_schedule(cfg["train"])
    normalize_optimizer_config(cfg["train"])
    apply_defaults(cfg["logging"], LOGGING_OPTION_DEFAULTS)
    normalize_loss_config(cfg)
    video_variant = cfg["data"]["video_variant"]
    if video_variant in PREBAKED_VIDEO_VARIANTS:
        expected_size = PREBAKED_VIDEO_VARIANTS[video_variant]["model_size"]
        if cfg["model"]["size"] != expected_size:
            raise ValueError(
                f"Config video_variant={video_variant!r} expects model.size={expected_size}, "
                f"got {cfg['model']['size']}."
            )
    return cfg


def resolve_camera_json_path(sequence_dir: Path, camera_json: Path | None) -> Path:
    if camera_json is not None:
        return camera_json
    return sequence_dir / "per_frame_cameras.json"


def load_sequence_data(
    camera_json_path: Path,
    target_size: int,
    camera_image_size: int,
    max_frames: int,
    focal_mode: str,
    device,
):
    return load_camera_sequence(
        camera_json_path=camera_json_path,
        target_size=target_size,
        camera_image_size=camera_image_size,
        max_frames=max_frames,
        focal_mode=focal_mode,
        device=device,
    )


def pick_renderer_mode(config: dict[str, Any]) -> tuple[str, int]:
    model_cfg = config["model"]
    render_cfg = config["render"]
    effective_gaussians = model_cfg["tokens"] * model_cfg["gaussians_per_token"]
    renderer_mode = resolve_renderer_mode(
        renderer=render_cfg["renderer"],
        gaussian_count=effective_gaussians,
        height=model_cfg["size"],
        width=model_cfg["size"],
        auto_dense_limit=render_cfg["auto_dense_limit"],
    )
    return renderer_mode, effective_gaussians


def learning_rate_for_step(base_lr: float, train_cfg: dict[str, Any], step: int) -> float:
    schedule_cfg = train_cfg["lr_schedule"]
    schedule_type = schedule_cfg["type"]
    if schedule_type in {"constant", "none", "off"}:
        return base_lr

    total_steps = max(1, int(train_cfg["steps"]))
    final_scale = float(schedule_cfg["final_lr_scale"])
    if final_scale < 0:
        raise ValueError(f"lr_schedule.final_lr_scale must be non-negative, got {final_scale}.")
    progress = 1.0 if total_steps == 1 else float(step - 1) / float(total_steps - 1)
    cosine_scale = final_scale + 0.5 * (1.0 - final_scale) * (1.0 + math.cos(math.pi * progress))
    return base_lr * cosine_scale


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr * float(group["lr_scale"])


def _weight_decay_exempt_parameter(name: str, parameter: torch.nn.Parameter) -> bool:
    lowered = name.lower()
    return parameter.ndim <= 1 or name.endswith(".bias") or ".norm" in lowered


def build_optimizer_param_groups(
    model: torch.nn.Module,
    base_lr: float,
    optimizer_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    weight_decay = float(optimizer_cfg["weight_decay"])
    exclude_bias_norm = bool(optimizer_cfg["exclude_bias_norm"])
    lr_multipliers = optimizer_cfg["lr_multipliers"]
    groups_by_key: dict[tuple[float, float], dict[str, Any]] = {}

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lr_scale = 1.0
        for prefix, multiplier in lr_multipliers.items():
            if name == prefix or name.startswith(f"{prefix}."):
                lr_scale = float(multiplier)
                break
        group_weight_decay = (
            0.0 if exclude_bias_norm and _weight_decay_exempt_parameter(name, parameter) else weight_decay
        )
        key = (group_weight_decay, lr_scale)
        if key not in groups_by_key:
            groups_by_key[key] = {
                "params": [],
                "weight_decay": group_weight_decay,
                "lr": base_lr * lr_scale,
                "lr_scale": lr_scale,
            }
        groups_by_key[key]["params"].append(parameter)

    return list(groups_by_key.values())


def build_optimizer(model: torch.nn.Module, train_cfg: dict[str, Any], device: torch.device) -> torch.optim.Optimizer:
    optimizer_cfg = train_cfg["optimizer"]
    optimizer_type = optimizer_cfg["type"]
    if optimizer_type == "adam":
        optimizer_cls = torch.optim.Adam
    elif optimizer_type == "adamw":
        optimizer_cls = torch.optim.AdamW
    else:
        raise ValueError(f"Unknown optimizer.type={optimizer_type!r}. Expected 'adam' or 'adamw'.")

    base_lr = float(train_cfg["lr"])
    param_groups = build_optimizer_param_groups(model, base_lr=base_lr, optimizer_cfg=optimizer_cfg)
    fused_default = device.type in {"cuda", "mps"}
    use_fused = fused_default if optimizer_cfg["fused"] is None else bool(optimizer_cfg["fused"])
    kwargs: dict[str, Any] = {"lr": base_lr}
    if use_fused:
        kwargs["fused"] = True
    try:
        return optimizer_cls(param_groups, **kwargs)
    except (RuntimeError, TypeError) as exc:
        if "fused" not in kwargs:
            raise
        print(f"Fused {optimizer_type} unavailable on {device.type}: {exc}. Falling back to unfused optimizer.")
        kwargs.pop("fused")
        return optimizer_cls(param_groups, **kwargs)


def build_model_from_config(model_cfg: dict[str, Any]) -> DynamicTokenGS:
    return DynamicTokenGS(
        num_tokens=model_cfg["tokens"],
        gaussians_per_token=model_cfg["gaussians_per_token"],
        **select_keys(model_cfg, MODEL_CONSTRUCTOR_KEYS),
    )


def print_key_values(label: str, values: dict[str, Any]) -> None:
    print(f"{label}: {format_key_values(values)}")


def gaussian_sequence_nonfinite_counts(decoded) -> dict[str, int]:
    fields = {
        "xyz": decoded.xyz,
        "scales": decoded.scales,
        "quats": decoded.quats,
        "opacities": decoded.opacities,
        "rgbs": decoded.rgbs,
    }
    return {name: int((~torch.isfinite(value.detach())).sum().detach().cpu()) for name, value in fields.items()}


def raise_for_nonfinite_decoded(decoded, *, step: int, frame_indices: list[int]) -> None:
    counts = gaussian_sequence_nonfinite_counts(decoded)
    if not any(counts.values()):
        return
    summary = format_key_values({name: count for name, count in counts.items() if count})
    raise FloatingPointError(
        f"Non-finite decoded Gaussian tensor at step {step} for frame indices {frame_indices}: {summary}"
    )


def render_one_frame(renderer_mode, config, dense_grid, camera, frame: GaussianFrame, return_aux: bool = False):
    model_cfg = config["model"]
    render_cfg = config["render"]
    return render_gaussian_frame(
        frame,
        camera=camera,
        height=model_cfg["size"],
        width=model_cfg["size"],
        mode=renderer_mode,
        dense_grid=dense_grid,
        tile_size=render_cfg["tile_size"],
        bound_scale=render_cfg["bound_scale"],
        alpha_threshold=render_cfg["alpha_threshold"],
        near_plane=render_cfg["near_plane"],
        taichi_options=render_cfg["taichi"],
        fast_mac_options=render_cfg["fast_mac"],
        return_aux=return_aux,
    )


def render_frame_batch(renderer_mode, config, dense_grid, cameras, decoded, return_aux: bool = False):
    model_cfg = config["model"]
    render_cfg = config["render"]
    return render_gaussian_frames(
        decoded,
        cameras=cameras,
        height=model_cfg["size"],
        width=model_cfg["size"],
        mode=renderer_mode,
        dense_grid=dense_grid,
        tile_size=render_cfg["tile_size"],
        bound_scale=render_cfg["bound_scale"],
        alpha_threshold=render_cfg["alpha_threshold"],
        near_plane=render_cfg["near_plane"],
        taichi_options=render_cfg["taichi"],
        fast_mac_options=render_cfg["fast_mac"],
        return_aux=return_aux,
    )


@torch.no_grad()
def render_full_sequence(
    model,
    sequence_data: SequenceData,
    config: dict[str, Any],
    renderer_mode,
    dense_grid,
    amp_available,
    amp_dtype,
    device,
):
    was_training = model.training
    model.eval()
    rendered_frames = []
    train_cfg = config["train"]
    num_frames = sequence_data.frame_count

    for start in range(0, num_frames, train_cfg["eval_batch_size"]):
        end = min(start + train_cfg["eval_batch_size"], num_frames)
        batch_frames = sequence_data.frames[start:end]
        batch_times = sequence_data.frame_times[start:end]
        batch_cameras = sequence_data.cameras[start:end]

        autocast_context = torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_available else nullcontext()
        with fast_attn_context(device), autocast_context:
            decoded = model(batch_frames, camera=batch_cameras, frame_times=batch_times)
        raise_for_nonfinite_decoded(decoded, step=0, frame_indices=list(range(start, end)))

        rendered_frames.extend(render_frame_batch(renderer_mode, config, dense_grid, batch_cameras, decoded).cpu())

    if was_training:
        model.train()
    return torch.stack(rendered_frames, dim=0)


def run_training(config: dict[str, Any]):
    cfg = resolve_config(config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    render_cfg = cfg["render"]
    train_cfg = cfg["train"]
    logging_cfg = cfg["logging"]
    loss_cfg = cfg["losses"]
    metric_cfg = metric_config_from_logging(logging_cfg)
    device = pick_device()
    print(f"Using device: {device}")

    camera_json_path = resolve_camera_json_path(data_cfg["sequence_dir"], data_cfg["camera_json"])
    sequence_data = load_sequence_data(
        camera_json_path=camera_json_path,
        target_size=model_cfg["size"],
        camera_image_size=data_cfg["camera_image_size"],
        max_frames=data_cfg["max_frames"],
        focal_mode=data_cfg["camera_focal_mode"],
        device=device,
    )
    num_frames = sequence_data.frame_count
    frames_per_step = num_frames if train_cfg["frames_per_step"] <= 0 else train_cfg["frames_per_step"]
    print(f"Loaded {num_frames} frames from {camera_json_path}")
    intrinsics_summary = sequence_data.intrinsics_summary
    print(
        "Camera intrinsics: "
        f"focal_mode={intrinsics_summary['focal_mode']}, "
        f"median_fx={intrinsics_summary['resolved_fx_median'] * intrinsics_summary['training_scale']:.2f}, "
        f"median_fy={intrinsics_summary['resolved_fy_median'] * intrinsics_summary['training_scale']:.2f}"
    )

    wandb.init(
        project=logging_cfg["wandb_project"],
        name=logging_cfg["wandb_run_name"],
        tags=logging_cfg["wandb_tags"],
        config=serialize_config_value(cfg),
    )

    model = build_model_from_config(model_cfg).to(device)
    model.train()
    base_lr = float(train_cfg["lr"])
    optimizer = build_optimizer(model, train_cfg=train_cfg, device=device)

    dense_grid = build_pixel_grid(model_cfg["size"], model_cfg["size"], device)
    amp_available = bool(train_cfg["amp"] and torch.amp.autocast_mode.is_autocast_available(device.type))
    if train_cfg["amp"] and not amp_available:
        print(f"AMP requested but not available on device {device.type}; continuing in fp32.")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    attn_dtype = amp_dtype if amp_available else sequence_data.frames.dtype
    attn_backend = configure_fast_attn(device, attn_dtype)
    renderer_mode, effective_gaussians = pick_renderer_mode(cfg)

    print(
        "Starting DynamicTokenGS Training: "
        f"{num_frames} frames, {model_cfg['tokens']} latent tokens x {model_cfg['gaussians_per_token']} gaussians/token "
        f"= {effective_gaussians} explicit Gaussians with {renderer_mode} renderer..."
    )
    print_key_values("Gaussian head", select_keys(model_cfg, GAUSSIAN_SUMMARY_KEYS))
    print_key_values("Renderer", select_keys(render_cfg, RENDER_SUMMARY_KEYS))
    print(f"Attention backend: {attn_backend}")
    optimizer_cfg = train_cfg["optimizer"]
    print_key_values(
        "Optimizer",
        {
            "type": optimizer_cfg["type"],
            "weight_decay": optimizer_cfg["weight_decay"],
            "exclude_bias_norm": optimizer_cfg["exclude_bias_norm"],
            "clip_grad_norm": train_cfg["clip_grad_norm"],
        },
    )
    schedule_cfg = train_cfg["lr_schedule"]
    if schedule_cfg["type"] != "constant":
        print_key_values("LR schedule", schedule_cfg)
    print_key_values("Reconstruction loss", loss_cfg)
    if metric_cfg.enabled:
        print(
            "Debug metrics: "
            f"renderer={metric_cfg.renderer}, optimizer={metric_cfg.optimizer}, every={metric_cfg.every}, "
            f"print_summary={metric_cfg.print_summary}, wandb={metric_cfg.wandb}, fail_fast={metric_cfg.fail_fast}"
        )

    gt_video_logged = False
    pbar = tqdm(range(1, train_cfg["steps"] + 1))

    for step in pbar:
        current_lr = learning_rate_for_step(base_lr, train_cfg, step)
        set_optimizer_lr(optimizer, current_lr)
        optimizer.zero_grad(set_to_none=True)
        batch_indices = select_window_indices(num_frames, frames_per_step, device=device)
        batch_frames = sequence_data.frames[batch_indices]
        batch_times = sequence_data.frame_times[batch_indices]
        frame_indices = batch_indices.tolist()
        batch_cameras = [sequence_data.cameras[index] for index in frame_indices]

        autocast_context = torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_available else nullcontext()
        with fast_attn_context(device), autocast_context:
            decoded = model(batch_frames, camera=batch_cameras, frame_times=batch_times)
        raise_for_nonfinite_decoded(decoded, step=step, frame_indices=frame_indices)

        should_collect_metrics = metric_cfg.due(step)
        return_render_aux = metric_cfg.renderer and should_collect_metrics and renderer_mode == "dense"
        render_result = render_frame_batch(
            renderer_mode,
            cfg,
            dense_grid,
            batch_cameras,
            decoded,
            return_aux=return_render_aux,
        )
        if return_render_aux:
            renders, render_aux = render_result
        else:
            renders = render_result
            render_aux = None
        render_is_finite = bool(torch.isfinite(renders).all().detach().cpu())
        metric_payload = {}
        if render_aux is not None:
            metric_payload.update(render_aux_diagnostics(render_aux))
        if (metric_cfg.renderer and should_collect_metrics) or not render_is_finite:
            metric_payload.update(dense_render_diagnostics(cfg, dense_grid, batch_cameras, decoded, renders=renders))
        if not render_is_finite:
            if metric_cfg.print_summary:
                print_metric_summary(step, metric_payload, frame_indices=frame_indices, prefix="DebugMetrics failure")
            if metric_cfg.wandb and metric_payload:
                wandb.log(metric_payload, step=step)
            if not metric_cfg.enabled or metric_cfg.fail_fast:
                raise FloatingPointError(
                    f"Non-finite render at step {step} for frame indices {frame_indices}: "
                    f"{format_metric_summary(metric_payload)}"
                )

        losses = reconstruction_loss_per_image(renders, batch_frames, loss_cfg)
        loss = losses.mean()
        if not bool(torch.isfinite(loss).detach().cpu()):
            if "RenderDiag/PowerMax" not in metric_payload:
                metric_payload.update(
                    dense_render_diagnostics(cfg, dense_grid, batch_cameras, decoded, renders=renders)
                )
            if metric_cfg.print_summary:
                print_metric_summary(step, metric_payload, frame_indices=frame_indices, prefix="DebugMetrics failure")
            if metric_cfg.wandb and metric_payload:
                wandb.log(metric_payload, step=step)
            if not metric_cfg.enabled or metric_cfg.fail_fast:
                raise FloatingPointError(
                    f"Non-finite loss at step {step} for frame indices {frame_indices}: "
                    f"{format_metric_summary(metric_payload)}"
                )
        loss.backward()
        if metric_cfg.optimizer and (should_collect_metrics or metric_cfg.fail_fast):
            optimizer_metrics = optimizer_diagnostics(model, optimizer)
            has_nonfinite_optimizer_state = (
                optimizer_metrics["OptDiag/GradNonfiniteCount"] > 0
                or optimizer_metrics["OptDiag/ParamNonfiniteCount"] > 0
            )
            if should_collect_metrics or has_nonfinite_optimizer_state:
                metric_payload.update(optimizer_metrics)
            if has_nonfinite_optimizer_state:
                if metric_cfg.print_summary:
                    print_metric_summary(
                        step,
                        metric_payload,
                        frame_indices=frame_indices,
                        prefix="DebugMetrics failure",
                    )
                if metric_cfg.wandb and metric_payload:
                    wandb.log(metric_payload, step=step)
                if metric_cfg.fail_fast:
                    raise FloatingPointError(
                        f"Non-finite optimizer state at step {step} for frame indices {frame_indices}: "
                        f"{format_metric_summary(metric_payload)}"
                    )
        clip_grad_norm = train_cfg["clip_grad_norm"]
        if clip_grad_norm is not None:
            try:
                grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(clip_grad_norm),
                    error_if_nonfinite=True,
                )
            except RuntimeError as exc:
                raise FloatingPointError(
                    f"Non-finite gradient norm at step {step} for frame indices {frame_indices}."
                ) from exc
            if should_collect_metrics:
                metric_payload["OptDiag/GradNormBeforeClip"] = float(grad_norm_before_clip.detach().cpu())
        optimizer.step()

        pbar.set_description(f"Loss: {loss.item():.4f}")
        if metric_cfg.print_summary and metric_payload and should_collect_metrics:
            print_metric_summary(step, metric_payload, frame_indices=frame_indices)

        should_log_scalars = step % max(1, logging_cfg["log_every"]) == 0 or (
            logging_cfg["always_log_last_step"] and step == train_cfg["steps"]
        )
        should_log_images = step % max(1, logging_cfg["image_log_every"]) == 0 or (
            logging_cfg["always_log_last_step"] and step == train_cfg["steps"]
        )
        should_log_videos = step % max(1, logging_cfg["video_log_every"]) == 0 or (
            logging_cfg["always_log_last_step"] and step == train_cfg["steps"]
        )
        if should_log_scalars:
            payload = {
                "Loss": loss.item(),
                "LearningRate": current_lr,
                "FramesPerStep": int(batch_indices.numel()),
                "SequenceFrames": num_frames,
            }
            if should_log_images:
                payload["Render_GT_vs_Pred"] = make_preview_image(
                    batch_frames[0],
                    renders[0],
                    caption=f"Step {step}",
                )
            if should_log_videos:
                rendered_sequence = render_full_sequence(
                    model,
                    sequence_data,
                    cfg,
                    renderer_mode,
                    dense_grid,
                    amp_available,
                    amp_dtype,
                    device,
                )
                gt_sequence = sequence_data.frames.detach().cpu()
                eval_delta = rendered_sequence - gt_sequence
                eval_l1 = float(eval_delta.abs().mean().item())
                eval_mse = float(eval_delta.square().mean().item())
                eval_loss = float(reconstruction_loss_per_image(rendered_sequence, gt_sequence, loss_cfg).mean().item())
                eval_ssim = float(
                    ssim_per_image(
                        rendered_sequence,
                        gt_sequence,
                        window_size=loss_cfg["ssim_window_size"],
                        c1=loss_cfg["ssim_c1"],
                        c2=loss_cfg["ssim_c2"],
                    )
                    .mean()
                    .item()
                )
                payload["Eval/L1"] = eval_l1
                payload["Eval/MSE"] = eval_mse
                payload["Eval/SSIM"] = eval_ssim
                payload["Eval/DSSIM"] = (1.0 - eval_ssim) * 0.5
                payload["Eval/Loss"] = eval_loss
                payload["Eval/PSNR"] = -10.0 * math.log10(max(eval_mse, 1.0e-12))
                payload.update(
                    build_validation_video_payload(
                        rendered_sequence,
                        gt_sequence,
                        sequence_data.video_fps,
                    )
                )
                if not gt_video_logged:
                    payload["GT_Video"] = make_wandb_video(gt_sequence, sequence_data.video_fps)
                    gt_video_logged = True
            payload.update(metric_payload)
            wandb.log(payload, step=step)
        elif metric_cfg.wandb and metric_payload:
            wandb.log(metric_payload, step=step)

    print("DynamicTokenGS training complete. Check your Weights & Biases dashboard.")
    wandb.finish()


def main(config: dict[str, Any] | str | Path) -> None:
    if isinstance(config, (str, Path)):
        run_training(load_config_file(config))
    else:
        run_training(config)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: uv run python src/train/dynamicTokenGS.py src/train_configs/local_mac_overfit_prebaked_camera.jsonc"
        )
    main(sys.argv[1])
