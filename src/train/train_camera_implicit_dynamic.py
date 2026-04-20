import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from dynamicTokenGS import (
    DEFAULT_SEQUENCE_DIR,
    configure_fast_attn,
    fast_attn_context,
    pick_device,
    pick_renderer_mode,
    select_window_indices,
)
from gs_models import DynamicTokenGSImplicitCamera
from rendering import render_gaussian_frame
from renderers.common import build_pixel_grid
from runtime_types import GaussianFrame
from sequence_data import load_uncalibrated_sequence, resolve_frames_dir
from train_logging import build_validation_video_payload, make_preview_image, make_wandb_video
from tqdm import tqdm


def build_arg_parser(default_renderer="auto"):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence-dir",
        type=Path,
        default=DEFAULT_SEQUENCE_DIR,
        help="Sequence directory containing extracted frames and optional summary.json metadata.",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="Optional explicit frames directory. Defaults to <sequence-dir>/frames.",
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=None,
        help="Optional explicit video path. Defaults to summary.json['video'] when using summary_video.",
    )
    parser.add_argument(
        "--frame-source",
        choices=("summary_video", "explicit_video", "summary_sampled", "all_frames"),
        default="summary_video",
        help="Use the source video from summary.json by default when it already matches the sampled DUSt3R clip.",
    )
    parser.add_argument("--steps", type=int, default=100, help="Training steps")
    parser.add_argument("--size", type=int, default=32, help="Render and training resolution")
    parser.add_argument("--max-frames", type=int, default=0, help="Use at most this many frames; 0 keeps all")
    parser.add_argument(
        "--frames-per-step",
        type=int,
        default=4,
        help="Train on a contiguous window of this many frames each step.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=4,
        help="How many frames to encode at once when rendering validation videos.",
    )
    parser.add_argument(
        "--tokens", type=int, default=128, help="Number of splat tokens. The model adds two camera tokens."
    )
    parser.add_argument(
        "--gaussians-per-token",
        type=int,
        default=4,
        help="Number of explicit Gaussians emitted by each splat token",
    )
    parser.add_argument(
        "--renderer",
        choices=("auto", "dense", "tiled"),
        default=default_renderer,
        help="Renderer backend. Auto uses dense for small workloads and tiled for larger ones.",
    )
    parser.add_argument(
        "--auto-dense-limit",
        type=int,
        default=400_000,
        help="Use dense when broadcasted_gaussians*H*W is below this",
    )
    parser.add_argument("--tile-size", type=int, default=8, help="Tile size for the tiled renderer")
    parser.add_argument("--bound-scale", type=float, default=3.0, help="Gaussian screen-space bound in sigmas")
    parser.add_argument(
        "--alpha-threshold",
        type=float,
        default=1.0 / 255.0,
        help="Opacity-aware tile culling threshold; set <=0 to disable opacity-aware shrinking",
    )
    parser.add_argument("--lr", type=float, default=0.005, help="Optimizer learning rate")
    parser.add_argument(
        "--camera-motion-weight",
        type=float,
        default=0.01,
        help="Penalty for large per-frame SE(3) residuals",
    )
    parser.add_argument(
        "--camera-temporal-weight",
        type=float,
        default=0.02,
        help="Penalty for changes in per-frame SE(3) residuals",
    )
    parser.add_argument(
        "--camera-global-weight",
        type=float,
        default=0.005,
        help="Penalty for sequence-global focal and radius drift",
    )
    parser.add_argument("--log-every", type=int, default=10, help="Log scalar loss to W&B every N steps")
    parser.add_argument("--image-log-every", type=int, default=50, help="Log preview image to W&B every N steps")
    parser.add_argument(
        "--video-log-every", type=int, default=50, help="Log GT and rendered videos to W&B every N steps"
    )
    parser.add_argument("--amp", action="store_true", help="Use autocast for the model forward pass")
    parser.add_argument("--wandb-project", type=str, default="dynamic-tokengs-overfit", help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, default="dynamic-implicit-camera-run", help="W&B run name")
    return parser


def parse_args(default_renderer="auto"):
    return build_arg_parser(default_renderer=default_renderer).parse_args()


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


def render_implicit_frame(renderer_mode, args, dense_grid, camera, xyz, scales, quats, opacities, rgbs):
    return render_gaussian_frame(
        GaussianFrame(xyz=xyz, scales=scales, quats=quats, opacities=opacities, rgbs=rgbs),
        camera=camera,
        height=args.size,
        width=args.size,
        mode=renderer_mode,
        dense_grid=dense_grid,
        tile_size=args.tile_size,
        bound_scale=args.bound_scale,
        alpha_threshold=args.alpha_threshold,
    )


@torch.no_grad()
def render_full_sequence(model, sequence_data, args, renderer_mode, dense_grid, amp_available, amp_dtype, device):
    was_training = model.training
    model.eval()
    rendered_frames = []
    decoded_cameras = []
    num_frames = sequence_data["frames"].shape[0]
    global_camera_token = model.infer_global_camera_token(
        sequence_data["frames"],
        frame_times=sequence_data["frame_times"],
        batch_size=args.eval_batch_size,
    )
    camera_states = []

    for start in range(0, num_frames, args.eval_batch_size):
        end = min(start + args.eval_batch_size, num_frames)
        batch_frames = sequence_data["frames"][start:end]
        batch_times = sequence_data["frame_times"][start:end]

        autocast_context = torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_available else nullcontext()
        with fast_attn_context(device), autocast_context:
            xyz, scales, quats, opacities, rgbs, cameras, camera_state = model(
                batch_frames,
                frame_times=batch_times,
                global_camera_token=global_camera_token,
            )
        camera_states.append(camera_state)

        for local_index, camera in enumerate(cameras):
            decoded_cameras.append(camera)
            rendered_frames.append(
                render_implicit_frame(
                    renderer_mode,
                    args,
                    dense_grid,
                    camera,
                    xyz[local_index],
                    scales[local_index],
                    quats[local_index],
                    opacities[local_index],
                    rgbs[local_index],
                ).cpu()
            )

    if was_training:
        model.train()
    merged_camera_state = {
        "fov_degrees": torch.stack([state["fov_degrees"] for state in camera_states]).mean(),
        "radius": torch.stack([state["radius"] for state in camera_states]).mean(),
        "rotation_delta": torch.cat([state["rotation_delta"] for state in camera_states], dim=0),
        "translation_delta": torch.cat([state["translation_delta"] for state in camera_states], dim=0),
    }
    return torch.stack(rendered_frames, dim=0), decoded_cameras, merged_camera_state


def run_training(args):
    device = pick_device()
    print(f"Using device: {device}")

    frames_dir = resolve_frames_dir(args.sequence_dir, args.frames_dir)
    sequence_data = load_sequence_data(
        sequence_dir=args.sequence_dir,
        frames_dir=frames_dir,
        video_path=args.video_path,
        target_size=args.size,
        max_frames=args.max_frames,
        frame_source=args.frame_source,
        device=device,
    )
    num_frames = sequence_data["frames"].shape[0]
    print(
        f"Loaded {num_frames} frames from {sequence_data['source_path']} "
        f"(source={sequence_data['frame_source']}, source_total={sequence_data['all_frame_count']})"
    )

    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    model = DynamicTokenGSImplicitCamera(num_tokens=args.tokens, gaussians_per_token=args.gaussians_per_token).to(
        device
    )
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, fused=device.type in {"cuda", "mps"})

    dense_grid = build_pixel_grid(args.size, args.size, device)
    amp_available = bool(args.amp and torch.amp.autocast_mode.is_autocast_available(device.type))
    if args.amp and not amp_available:
        print(f"AMP requested but not available on device {device.type}; continuing in fp32.")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    attn_dtype = amp_dtype if amp_available else sequence_data["frames"].dtype
    attn_backend = configure_fast_attn(device, attn_dtype)
    renderer_mode, effective_gaussians = pick_renderer_mode(args)

    print(
        "Starting DynamicTokenGSImplicitCamera image-encoder baseline training: "
        f"{num_frames} frames, 1 global camera token + 1 path token + {args.tokens} splat tokens x {args.gaussians_per_token} gaussians/token "
        f"= {effective_gaussians} explicit Gaussians with implicit cameras, no plucker conditioning, and {renderer_mode} renderer..."
    )
    print(f"Attention backend: {attn_backend}")

    gt_video_logged = False
    pbar = tqdm(range(1, args.steps + 1))

    for step in pbar:
        optimizer.zero_grad(set_to_none=True)
        batch_indices = select_window_indices(num_frames, args.frames_per_step, device=device)
        batch_frames = sequence_data["frames"][batch_indices]
        batch_times = sequence_data["frame_times"][batch_indices]

        autocast_context = torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_available else nullcontext()
        with fast_attn_context(device), autocast_context:
            xyz, scales, quats, opacities, rgbs, cameras, camera_state = model(batch_frames, frame_times=batch_times)

        renders = []
        recon_losses = []
        for local_index, camera in enumerate(cameras):
            render = render_implicit_frame(
                renderer_mode,
                args,
                dense_grid,
                camera,
                xyz[local_index],
                scales[local_index],
                quats[local_index],
                opacities[local_index],
                rgbs[local_index],
            )
            target = batch_frames[local_index]
            renders.append(render)
            recon_losses.append(F.l1_loss(render, target) + 0.2 * F.mse_loss(render, target))

        recon_loss = torch.stack(recon_losses).mean()
        camera_motion_loss = (
            torch.cat(
                [
                    camera_state["rotation_delta"],
                    camera_state["translation_delta"] / camera_state["radius"].clamp_min(1e-6),
                ],
                dim=-1,
            )
            .pow(2)
            .mean()
        )
        camera_global_loss = camera_state["global_residuals"].pow(2).mean()
        if batch_indices.numel() > 1:
            motion_features = torch.cat([camera_state["rotation_delta"], camera_state["translation_delta"]], dim=-1)
            camera_temporal_loss = (motion_features[1:] - motion_features[:-1]).pow(2).mean()
        else:
            camera_temporal_loss = recon_loss.new_tensor(0.0)
        loss = (
            recon_loss
            + args.camera_motion_weight * camera_motion_loss
            + args.camera_temporal_weight * camera_temporal_loss
            + args.camera_global_weight * camera_global_loss
        )
        loss.backward()
        optimizer.step()

        mean_radius = camera_state["radius"].item()
        mean_fov = camera_state["fov_degrees"].item()
        mean_rot_deg = torch.rad2deg(torch.linalg.norm(camera_state["rotation_delta"], dim=-1)).mean().item()
        mean_trans = torch.linalg.norm(camera_state["translation_delta"], dim=-1).mean().item()
        pbar.set_description(
            f"Loss: {loss.item():.4f} fov: {mean_fov:.2f} r: {mean_radius:.2f} rot: {mean_rot_deg:.2f}"
        )

        should_log_scalars = step % max(1, args.log_every) == 0 or step == args.steps
        should_log_images = step % max(1, args.image_log_every) == 0 or step == args.steps
        should_log_videos = step % max(1, args.video_log_every) == 0 or step == args.steps
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
                    args,
                    renderer_mode,
                    dense_grid,
                    amp_available,
                    amp_dtype,
                    device,
                )
                gt_sequence = sequence_data["frames"].detach().cpu()
                payload.update(
                    build_validation_video_payload(
                        rendered_sequence,
                        gt_sequence,
                        sequence_data["video_fps"],
                    )
                )
                payload["Camera/EvalFOVDegrees"] = eval_camera_state["fov_degrees"].item()
                payload["Camera/EvalRadius"] = eval_camera_state["radius"].item()
                payload["Camera/EvalRotationDeltaMeanDegrees"] = (
                    torch.rad2deg(torch.linalg.norm(eval_camera_state["rotation_delta"], dim=-1)).mean().item()
                )
                payload["Camera/EvalTranslationDeltaMean"] = (
                    torch.linalg.norm(eval_camera_state["translation_delta"], dim=-1).mean().item()
                )
                if not gt_video_logged:
                    payload["GT_Video"] = make_wandb_video(gt_sequence, sequence_data["video_fps"])
                    gt_video_logged = True
            wandb.log(payload, step=step)

    print("DynamicTokenGSImplicitCamera training complete. Check your Weights & Biases dashboard.")
    wandb.finish()


def main(default_renderer="auto", run_name="dynamic-implicit-camera-run"):
    args = parse_args(default_renderer=default_renderer)
    if args.wandb_run_name == "dynamic-implicit-camera-run":
        args.wandb_run_name = run_name
    run_training(args)


if __name__ == "__main__":
    main()
