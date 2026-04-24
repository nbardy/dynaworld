import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import wandb
from config_utils import apply_defaults, load_config_file, path_or_none, resolved_config, serialize_config_value
from dynamicTokenGS import (
    configure_fast_attn,
    fast_attn_context,
    pick_device,
    select_window_indices,
)
from gs_models import DynamicTokenGSImplicitCamera, DynamicTokenGSSeparatedImplicitCamera
from losses import reconstruction_loss_per_image, ssim_per_image
from renderers.common import build_pixel_grid
from rendering import pick_renderer_mode as resolve_renderer_mode
from rendering import render_gaussian_frame
from runtime_types import CameraState, GaussianFrame, SequenceData
from sequence_data import load_uncalibrated_sequence, resolve_frames_dir
from tqdm import tqdm
from train_logging import build_validation_video_payload, make_preview_image, make_wandb_video


LOSS_OPTION_DEFAULTS = {
    "type": "l1_mse",
    "l1_weight": 1.0,
    "mse_weight": 0.2,
    "dssim_weight": 0.2,
    "ssim_window_size": 11,
    "ssim_c1": 0.0001,
    "ssim_c2": 0.0009,
    "camera_motion_weight": 0.01,
    "camera_temporal_weight": 0.02,
    "camera_global_weight": 0.005,
}


def resolve_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = resolved_config(config, ("data", "model", "render", "train", "losses", "logging"))
    cfg["data"]["sequence_dir"] = Path(cfg["data"]["sequence_dir"])
    cfg["data"]["frames_dir"] = path_or_none(cfg["data"]["frames_dir"])
    cfg["data"]["video_path"] = path_or_none(cfg["data"]["video_path"])
    cfg["model"].setdefault("variant", "joint_attention")
    if cfg["model"]["variant"] not in {"joint_attention", "separated_camera"}:
        raise ValueError(
            "model.variant must be one of {'joint_attention', 'separated_camera'}, "
            f"got {cfg['model']['variant']!r}."
        )
    apply_defaults(cfg["losses"], LOSS_OPTION_DEFAULTS)
    cfg["losses"]["type"] = str(cfg["losses"]["type"]).lower()
    if cfg["losses"]["type"] not in {"standard_gs", "l1_mse", "l1", "mse"}:
        raise ValueError(
            f"Unknown losses.type={cfg['losses']['type']!r}. Expected one of: standard_gs, l1_mse, l1, mse."
        )
    window_size = int(cfg["losses"]["ssim_window_size"])
    if window_size < 1 or window_size % 2 != 1:
        raise ValueError(f"losses.ssim_window_size must be a positive odd integer, got {window_size}.")
    cfg["losses"]["ssim_window_size"] = window_size
    if "near_plane" not in cfg["render"]:
        cfg["render"]["near_plane"] = 1.0e-4
    if "fast_mac" not in cfg["render"]:
        cfg["render"]["fast_mac"] = None
    return cfg


def load_sequence_data(
    sequence_dir: Path,
    frames_dir: Path,
    video_path: Path | None,
    target_size: int,
    max_frames: int,
    frame_source: str,
    device,
):
    return load_uncalibrated_sequence(
        sequence_dir=sequence_dir,
        frames_dir=frames_dir,
        video_path=video_path,
        target_size=target_size,
        max_frames=max_frames,
        frame_source=frame_source,
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


def build_model_from_config(model_cfg: dict[str, Any]):
    model_cls = (
        DynamicTokenGSSeparatedImplicitCamera
        if model_cfg["variant"] == "separated_camera"
        else DynamicTokenGSImplicitCamera
    )
    return model_cls(
        num_tokens=model_cfg["tokens"],
        gaussians_per_token=model_cfg["gaussians_per_token"],
    )


def render_implicit_frame(renderer_mode, config, dense_grid, camera, frame: GaussianFrame):
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
        fast_mac_options=render_cfg["fast_mac"],
    )


def eval_metric_payload(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_cfg: dict[str, Any],
) -> dict[str, float]:
    prediction = prediction.float()
    target = target.float()
    delta = prediction - target
    l1 = delta.abs().flatten(1).mean()
    mse = delta.square().flatten(1).mean()
    ssim = ssim_per_image(
        prediction,
        target,
        window_size=loss_cfg["ssim_window_size"],
        c1=float(loss_cfg["ssim_c1"]),
        c2=float(loss_cfg["ssim_c2"]),
    ).mean()
    dssim = (1.0 - ssim) * 0.5
    recon_loss = reconstruction_loss_per_image(prediction, target, loss_cfg).mean()
    psnr = -10.0 * math.log10(max(float(mse.item()), 1.0e-12))
    return {
        "Eval/Loss": float(recon_loss.item()),
        "Eval/L1": float(l1.item()),
        "Eval/MSE": float(mse.item()),
        "Eval/SSIM": float(ssim.item()),
        "Eval/DSSIM": float(dssim.item()),
        "Eval/PSNR": psnr,
    }


@torch.no_grad()
def render_full_sequence(
    model, sequence_data: SequenceData, config, renderer_mode, dense_grid, amp_available, amp_dtype, device
):
    was_training = model.training
    model.eval()
    rendered_frames = []
    decoded_cameras = []
    train_cfg = config["train"]
    num_frames = sequence_data.frame_count
    global_camera_token = model.infer_global_camera_token(
        sequence_data.frames,
        frame_times=sequence_data.frame_times,
        batch_size=train_cfg["eval_batch_size"],
    )
    camera_states = []

    for start in range(0, num_frames, train_cfg["eval_batch_size"]):
        end = min(start + train_cfg["eval_batch_size"], num_frames)
        batch_frames = sequence_data.frames[start:end]
        batch_times = sequence_data.frame_times[start:end]

        autocast_context = torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_available else nullcontext()
        with fast_attn_context(device), autocast_context:
            decoded = model(
                batch_frames,
                frame_times=batch_times,
                global_camera_token=global_camera_token,
            )
        camera_states.append(decoded.camera_state)

        for local_index, camera in enumerate(decoded.cameras):
            decoded_cameras.append(camera)
            rendered_frames.append(
                render_implicit_frame(
                    renderer_mode,
                    config,
                    dense_grid,
                    camera,
                    decoded.frame(local_index),
                ).cpu()
            )

    if was_training:
        model.train()
    merged_camera_state = CameraState(
        fov_degrees=torch.stack([state.fov_degrees for state in camera_states]).mean(),
        radius=torch.stack([state.radius for state in camera_states]).mean(),
        global_residuals=torch.stack([state.global_residuals for state in camera_states]).mean(dim=0),
        rotation_delta=torch.cat([state.rotation_delta for state in camera_states], dim=0),
        translation_delta=torch.cat([state.translation_delta for state in camera_states], dim=0),
        path_residuals=torch.cat([state.path_residuals for state in camera_states], dim=0),
    )
    return torch.stack(rendered_frames, dim=0), decoded_cameras, merged_camera_state


def run_training(config: dict[str, Any]):
    cfg = resolve_config(config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    loss_cfg = cfg["losses"]
    logging_cfg = cfg["logging"]
    device = pick_device()
    print(f"Using device: {device}")

    frames_dir = resolve_frames_dir(data_cfg["sequence_dir"], data_cfg["frames_dir"])
    sequence_data = load_sequence_data(
        sequence_dir=data_cfg["sequence_dir"],
        frames_dir=frames_dir,
        video_path=data_cfg["video_path"],
        target_size=model_cfg["size"],
        max_frames=data_cfg["max_frames"],
        frame_source=data_cfg["frame_source"],
        device=device,
    )
    num_frames = sequence_data.frame_count
    frames_per_step = num_frames if train_cfg["frames_per_step"] <= 0 else train_cfg["frames_per_step"]
    print(
        f"Loaded {num_frames} frames from {sequence_data.source_path} "
        f"(source={sequence_data.frame_source}, source_total={sequence_data.all_frame_count})"
    )

    wandb.init(
        project=logging_cfg["wandb_project"],
        name=logging_cfg["wandb_run_name"],
        tags=logging_cfg.get("wandb_tags"),
        config=serialize_config_value(cfg),
    )

    model = build_model_from_config(model_cfg).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"], fused=device.type in {"cuda", "mps"})

    dense_grid = build_pixel_grid(model_cfg["size"], model_cfg["size"], device)
    amp_available = bool(train_cfg["amp"] and torch.amp.autocast_mode.is_autocast_available(device.type))
    if train_cfg["amp"] and not amp_available:
        print(f"AMP requested but not available on device {device.type}; continuing in fp32.")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    attn_dtype = amp_dtype if amp_available else sequence_data.frames.dtype
    attn_backend = configure_fast_attn(device, attn_dtype)
    renderer_mode, effective_gaussians = pick_renderer_mode(cfg)

    print(
        f"Starting {model.__class__.__name__} image-encoder baseline training "
        f"(variant={model_cfg['variant']}): "
        f"{num_frames} frames, {model.num_tokens} splat tokens ({model.total_tokens} attention tokens) x "
        f"{model_cfg['gaussians_per_token']} gaussians/token "
        f"= {effective_gaussians} explicit Gaussians with implicit cameras, no plucker conditioning, and {renderer_mode} renderer..."
    )
    print(f"Attention backend: {attn_backend}")

    gt_video_logged = False
    pbar = tqdm(range(1, train_cfg["steps"] + 1))

    for step in pbar:
        optimizer.zero_grad(set_to_none=True)
        batch_indices = select_window_indices(num_frames, frames_per_step, device=device)
        batch_frames = sequence_data.frames[batch_indices]
        batch_times = sequence_data.frame_times[batch_indices]

        autocast_context = torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_available else nullcontext()
        with fast_attn_context(device), autocast_context:
            decoded = model(batch_frames, frame_times=batch_times)
        camera_state = decoded.camera_state

        renders = []
        recon_losses = []
        for local_index, camera in enumerate(decoded.cameras):
            render = render_implicit_frame(
                renderer_mode,
                cfg,
                dense_grid,
                camera,
                decoded.frame(local_index),
            )
            target = batch_frames[local_index]
            renders.append(render)
            recon_losses.append(reconstruction_loss_per_image(render.unsqueeze(0), target.unsqueeze(0), loss_cfg)[0])

        recon_loss = torch.stack(recon_losses).mean()
        camera_motion_loss = (
            torch.cat(
                [
                    camera_state.rotation_delta,
                    camera_state.translation_delta / camera_state.radius.clamp_min(1e-6),
                ],
                dim=-1,
            )
            .pow(2)
            .mean()
        )
        camera_global_loss = camera_state.global_residuals.pow(2).mean()
        if batch_indices.numel() > 1:
            motion_features = camera_state.motion_features()
            camera_temporal_loss = (motion_features[1:] - motion_features[:-1]).pow(2).mean()
        else:
            camera_temporal_loss = recon_loss.new_tensor(0.0)
        loss = (
            recon_loss
            + loss_cfg["camera_motion_weight"] * camera_motion_loss
            + loss_cfg["camera_temporal_weight"] * camera_temporal_loss
            + loss_cfg["camera_global_weight"] * camera_global_loss
        )
        loss.backward()
        optimizer.step()

        mean_radius = camera_state.radius.item()
        mean_fov = camera_state.fov_degrees.item()
        mean_rot_deg = torch.rad2deg(torch.linalg.norm(camera_state.rotation_delta, dim=-1)).mean().item()
        mean_trans = torch.linalg.norm(camera_state.translation_delta, dim=-1).mean().item()
        pbar.set_description(
            f"Loss: {loss.item():.4f} fov: {mean_fov:.2f} r: {mean_radius:.2f} rot: {mean_rot_deg:.2f}"
        )

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
                "Loss/Reconstruction": recon_loss.item(),
                "Loss/CameraMotion": camera_motion_loss.item(),
                "Loss/CameraTemporal": camera_temporal_loss.item(),
                "Loss/CameraGlobal": camera_global_loss.item(),
                "FramesPerStep": int(batch_indices.numel()),
                "SequenceFrames": num_frames,
                "Camera/FOVDegrees": mean_fov,
                "Camera/Radius": mean_radius,
                "Camera/RotationDeltaMeanDegrees": mean_rot_deg,
                "Camera/TranslationDeltaMean": mean_trans,
            }
            if should_log_images:
                payload["Render_GT_vs_Pred"] = make_preview_image(
                    batch_frames[0],
                    renders[0],
                    caption=f"Step {step}",
                )
            if should_log_videos:
                rendered_sequence, _decoded_cameras, eval_camera_state = render_full_sequence(
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
                payload.update(
                    build_validation_video_payload(
                        rendered_sequence,
                        gt_sequence,
                        sequence_data.video_fps,
                    )
                )
                payload.update(eval_metric_payload(rendered_sequence, gt_sequence, loss_cfg))
                payload["Camera/EvalFOVDegrees"] = eval_camera_state.fov_degrees.item()
                payload["Camera/EvalRadius"] = eval_camera_state.radius.item()
                payload["Camera/EvalRotationDeltaMeanDegrees"] = (
                    torch.rad2deg(torch.linalg.norm(eval_camera_state.rotation_delta, dim=-1)).mean().item()
                )
                payload["Camera/EvalTranslationDeltaMean"] = (
                    torch.linalg.norm(eval_camera_state.translation_delta, dim=-1).mean().item()
                )
                if not gt_video_logged:
                    payload["GT_Video"] = make_wandb_video(gt_sequence, sequence_data.video_fps)
                    gt_video_logged = True
            wandb.log(payload, step=step)

    print("DynamicTokenGSImplicitCamera training complete. Check your Weights & Biases dashboard.")
    wandb.finish()


def main(config: dict[str, Any] | str | Path) -> None:
    if isinstance(config, (str, Path)):
        run_training(load_config_file(config))
    else:
        run_training(config)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: uv run python src/train/train_camera_implicit_dynamic.py "
            "src/train_configs/local_mac_overfit_image_implicit_camera.jsonc"
        )
    main(sys.argv[1])
