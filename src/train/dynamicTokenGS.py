import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import wandb
from config_utils import load_config_file, path_or_none, resolved_config, serialize_config_value
from fast_attn import configure_fast_attn, fast_attn_context
from gs_models import DynamicTokenGS
from renderers.common import build_pixel_grid
from rendering import pick_renderer_mode as resolve_renderer_mode
from rendering import render_gaussian_frame
from runtime_types import GaussianFrame, SequenceData
from sequence_data import load_camera_sequence, select_window_indices
from tqdm import tqdm
from train_logging import build_validation_video_payload, make_preview_image, make_wandb_video


def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def resolve_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = resolved_config(config, ("data", "model", "render", "train", "logging"))
    cfg["data"]["sequence_dir"] = Path(cfg["data"]["sequence_dir"])
    cfg["data"]["camera_json"] = path_or_none(cfg["data"]["camera_json"])
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


def render_one_frame(renderer_mode, config, dense_grid, camera, frame: GaussianFrame):
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
        with fast_attn_context(device):
            with autocast_context:
                decoded = model(batch_frames, camera=batch_cameras, frame_times=batch_times)

        for local_index, camera in enumerate(batch_cameras):
            rendered_frames.append(
                render_one_frame(
                    renderer_mode,
                    config,
                    dense_grid,
                    camera,
                    decoded.frame(local_index),
                ).cpu()
            )

    if was_training:
        model.train()
    return torch.stack(rendered_frames, dim=0)


def run_training(config: dict[str, Any]):
    cfg = resolve_config(config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    logging_cfg = cfg["logging"]
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
        tags=logging_cfg.get("wandb_tags"),
        config=serialize_config_value(cfg),
    )

    model = DynamicTokenGS(
        num_tokens=model_cfg["tokens"],
        gaussians_per_token=model_cfg["gaussians_per_token"],
    ).to(device)
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
        "Starting DynamicTokenGS Training: "
        f"{num_frames} frames, {model_cfg['tokens']} latent tokens x {model_cfg['gaussians_per_token']} gaussians/token "
        f"= {effective_gaussians} explicit Gaussians with {renderer_mode} renderer..."
    )
    print(f"Attention backend: {attn_backend}")

    gt_video_logged = False
    pbar = tqdm(range(1, train_cfg["steps"] + 1))

    for step in pbar:
        optimizer.zero_grad(set_to_none=True)
        batch_indices = select_window_indices(num_frames, frames_per_step, device=device)
        batch_frames = sequence_data.frames[batch_indices]
        batch_times = sequence_data.frame_times[batch_indices]
        batch_cameras = [sequence_data.cameras[index] for index in batch_indices.tolist()]

        autocast_context = torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_available else nullcontext()
        with fast_attn_context(device):
            with autocast_context:
                decoded = model(batch_frames, camera=batch_cameras, frame_times=batch_times)

        renders = []
        losses = []
        for local_index, camera in enumerate(batch_cameras):
            render = render_one_frame(
                renderer_mode,
                cfg,
                dense_grid,
                camera,
                decoded.frame(local_index),
            )
            target = batch_frames[local_index]
            renders.append(render)
            losses.append(F.l1_loss(render, target) + 0.2 * F.mse_loss(render, target))

        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()

        pbar.set_description(f"Loss: {loss.item():.4f}")

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
            wandb.log(payload, step=step)

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
