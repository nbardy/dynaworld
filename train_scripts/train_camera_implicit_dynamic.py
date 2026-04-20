import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import wandb
from dynamicTokenGS import (
    DEFAULT_SEQUENCE_DIR,
    configure_fast_attn,
    fast_attn_context,
    infer_video_fps,
    make_wandb_video,
    pick_device,
    pick_renderer_mode,
    select_window_indices,
)
from gs_models import DynamicTokenGSImplicitCamera
from PIL import Image
from renderers.common import build_pixel_grid
from renderers.dense import render_pytorch_3dgs
from renderers.tiled import render_pytorch_3dgs_tiled
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
    parser.add_argument("--tokens", type=int, default=128, help="Number of splat tokens. The model adds two camera tokens.")
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
    parser.add_argument("--video-log-every", type=int, default=50, help="Log GT and rendered videos to W&B every N steps")
    parser.add_argument("--amp", action="store_true", help="Use autocast for the model forward pass")
    parser.add_argument("--wandb-project", type=str, default="dynamic-tokengs-overfit", help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, default="dynamic-implicit-camera-run", help="W&B run name")
    return parser


def parse_args(default_renderer="auto"):
    return build_arg_parser(default_renderer=default_renderer).parse_args()


def resolve_frames_dir(sequence_dir: Path, frames_dir: Path | None) -> Path:
    if frames_dir is not None:
        return frames_dir
    return sequence_dir / "frames"


def resolve_video_path(video_path: Path | None, metadata) -> Path | None:
    if video_path is not None:
        return video_path
    if metadata is None:
        return None
    value = metadata.get("video")
    if not value:
        return None
    return Path(value)


def load_sequence_metadata(sequence_dir: Path):
    summary_path = sequence_dir / "summary.json"
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text())


def build_frame_times(frame_paths, metadata):
    timestamps = None
    if metadata is not None:
        sampled_frames = metadata.get("frame_sampling", {}).get("sampled_frames", [])
        timestamp_by_path = {}
        for item in sampled_frames:
            path = item.get("path")
            if path is None:
                continue
            timestamp_by_path[str(Path(path).resolve())] = item.get("timestamp_seconds")
        timestamps = [timestamp_by_path.get(str(path.resolve())) for path in frame_paths]

    if timestamps is None or all(timestamp is None for timestamp in timestamps):
        values = np.arange(len(frame_paths), dtype=np.float32)
        return torch.from_numpy(values).unsqueeze(-1), 1.0

    times_np = np.asarray(
        [timestamp if timestamp is not None else index for index, timestamp in enumerate(timestamps)],
        dtype=np.float32,
    )
    video_fps = infer_video_fps([{"timestamp_seconds": value} for value in timestamps])
    return torch.from_numpy(times_np).unsqueeze(-1), video_fps


def build_uniform_frame_times(num_frames: int, fps: float):
    if num_frames < 1:
        raise ValueError("Need at least one frame to build timestamps.")
    safe_fps = float(fps) if fps and fps > 0 else 1.0
    values = np.arange(num_frames, dtype=np.float32) / safe_fps
    return torch.from_numpy(values).unsqueeze(-1), safe_fps


def normalize_frame_times(frame_times: torch.Tensor):
    frame_times_np = frame_times.numpy()
    if len(frame_times_np) > 1 and float(frame_times_np.max()) > float(frame_times_np.min()):
        frame_times_np = (frame_times_np - frame_times_np.min()) / (frame_times_np.max() - frame_times_np.min())
    else:
        frame_times_np = np.zeros_like(frame_times_np)
    return torch.from_numpy(frame_times_np)


def resolve_frame_paths(sequence_dir: Path, frames_dir: Path, metadata, frame_source: str):
    if frame_source == "summary_sampled" and metadata is not None:
        sampled_frames = metadata.get("frame_sampling", {}).get("sampled_frames", [])
        sampled_paths = [Path(item["path"]) for item in sampled_frames if item.get("path")]
        existing_paths = [path for path in sampled_paths if path.exists()]
        if len(existing_paths) >= 2:
            return existing_paths, "summary_sampled"
    return sorted(frames_dir.glob("*.png")), "all_frames"


def load_video_sequence(video_path: Path, target_size: int, max_frames: int):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    transform = T.Compose([T.Resize((target_size, target_size)), T.ToTensor()])
    frames = []

    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(transform(Image.fromarray(frame_rgb)))
        if max_frames > 0 and len(frames) >= max_frames:
            break

    capture.release()
    if len(frames) < 2:
        raise ValueError(f"Need at least 2 frames in video {video_path}")

    frame_times, video_fps = build_uniform_frame_times(len(frames), fps)
    return {
        "frames": torch.stack(frames, dim=0),
        "frame_times": normalize_frame_times(frame_times),
        "video_fps": video_fps,
        "frame_source": "summary_video",
        "source_path": video_path,
        "selected_frame_count": len(frames),
        "all_frame_count": total_frames if total_frames > 0 else len(frames),
    }


def load_frame_sequence(frames_dir: Path, metadata, target_size: int, max_frames: int, frame_source: str):
    all_frame_paths = sorted(frames_dir.glob("*.png"))
    frame_paths, resolved_frame_source = resolve_frame_paths(None, frames_dir, metadata, frame_source=frame_source)
    if max_frames > 0:
        frame_paths = frame_paths[:max_frames]
    if len(frame_paths) < 2:
        raise ValueError(f"Need at least 2 frames in {frames_dir}")
    transform = T.Compose([T.Resize((target_size, target_size)), T.ToTensor()])
    frames = [transform(Image.open(frame_path).convert("RGB")) for frame_path in frame_paths]
    frame_times, video_fps = build_frame_times(frame_paths, metadata)
    return {
        "frames": torch.stack(frames, dim=0),
        "frame_times": normalize_frame_times(frame_times),
        "frame_paths": frame_paths,
        "video_fps": video_fps,
        "frame_source": resolved_frame_source,
        "source_path": frames_dir,
        "selected_frame_count": len(frame_paths),
        "all_frame_count": len(all_frame_paths),
    }


def load_sequence_data(
    sequence_dir: Path,
    frames_dir: Path,
    video_path: Path | None,
    target_size: int,
    max_frames: int,
    frame_source: str,
    device,
):
    metadata = load_sequence_metadata(sequence_dir)
    if frame_source in {"summary_video", "explicit_video"}:
        resolved_video_path = resolve_video_path(video_path, metadata if frame_source == "summary_video" else None)
        if resolved_video_path is None or not resolved_video_path.exists():
            mode_name = "summary_video" if frame_source == "summary_video" else "explicit_video"
            raise FileNotFoundError(
                f"{mode_name} requested but no usable video path was found. "
                "Pass --video-path explicitly or use --frame-source summary_sampled."
            )
        expected_frames = metadata.get("frame_sampling", {}).get("total_frames") if metadata is not None else None
        video_sequence = load_video_sequence(resolved_video_path, target_size=target_size, max_frames=max_frames)
        actual_frames = video_sequence["all_frame_count"]
        if frame_source == "summary_video" and expected_frames is not None and max_frames == 0 and int(expected_frames) != int(actual_frames):
            raise ValueError(
                f"summary_video requested but video frame count {actual_frames} does not match "
                f"summary.json frame_sampling.total_frames {expected_frames}. "
                "Use --frame-source summary_sampled when the source video was not pre-downsampled."
            )
        video_sequence["frame_source"] = frame_source
        video_sequence["frames"] = video_sequence["frames"].to(device)
        video_sequence["frame_times"] = video_sequence["frame_times"].to(device=device, dtype=torch.float32)
        return video_sequence

    frame_sequence = load_frame_sequence(
        frames_dir=frames_dir,
        metadata=metadata,
        target_size=target_size,
        max_frames=max_frames,
        frame_source=frame_source,
    )
    frame_sequence["frames"] = frame_sequence["frames"].to(device)
    frame_sequence["frame_times"] = frame_sequence["frame_times"].to(device=device, dtype=torch.float32)
    return frame_sequence


def render_implicit_frame(renderer_mode, args, dense_grid, camera, xyz, scales, quats, opacities, rgbs):
    if renderer_mode == "dense":
        return render_pytorch_3dgs(
            xyz.float(),
            scales.float(),
            quats.float(),
            opacities.float(),
            rgbs.float(),
            args.size,
            args.size,
            camera.fx,
            camera.fy,
            camera.cx,
            camera.cy,
            grid=dense_grid,
            camera_to_world=camera.camera_to_world.float(),
        )
    return render_pytorch_3dgs_tiled(
        xyz.float(),
        scales.float(),
        quats.float(),
        opacities.float(),
        rgbs.float(),
        args.size,
        args.size,
        camera.fx,
        camera.fy,
        camera.cx,
        camera.cy,
        tile_size=args.tile_size,
        bound_scale=args.bound_scale,
        alpha_threshold=args.alpha_threshold,
        camera_to_world=camera.camera_to_world.float(),
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

        autocast_context = (
            torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_available else nullcontext()
        )
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

    model = DynamicTokenGSImplicitCamera(num_tokens=args.tokens, gaussians_per_token=args.gaussians_per_token).to(device)
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

        autocast_context = (
            torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_available else nullcontext()
        )
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
        camera_motion_loss = torch.cat(
            [camera_state["rotation_delta"], camera_state["translation_delta"] / camera_state["radius"].clamp_min(1e-6)],
            dim=-1,
        ).pow(2).mean()
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
                preview = torch.cat([batch_frames[0], renders[0].detach()], dim=2)
                payload["Render_GT_vs_Pred"] = wandb.Image(
                    T.ToPILImage()(preview.cpu().clamp(0, 1)),
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
                side_by_side = torch.cat([gt_sequence, rendered_sequence], dim=3)
                payload["Render_Video"] = make_wandb_video(rendered_sequence, sequence_data["video_fps"])
                payload["Render_GT_Video"] = make_wandb_video(side_by_side, sequence_data["video_fps"])
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
