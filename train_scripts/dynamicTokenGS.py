import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import wandb
from PIL import Image
from tqdm import tqdm

from fast_attn import configure_fast_attn, fast_attn_context
from camera import CameraSpec
from gs_models import DynamicTokenGS
from renderers.common import build_pixel_grid
from renderers.dense import render_pytorch_3dgs
from renderers.tiled import render_pytorch_3dgs_tiled


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEQUENCE_DIR = ROOT / "test_data" / "dust3r_outputs" / "test_video_small_all_frames"


def build_arg_parser(default_renderer="auto"):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence-dir",
        type=Path,
        default=DEFAULT_SEQUENCE_DIR,
        help="DUSt3R output directory containing per_frame_cameras.json and extracted frames.",
    )
    parser.add_argument(
        "--camera-json",
        type=Path,
        default=None,
        help="Optional explicit per-frame camera JSON path. Defaults to <sequence-dir>/per_frame_cameras.json.",
    )
    parser.add_argument("--steps", type=int, default=100, help="Training steps")
    parser.add_argument("--size", type=int, default=32, help="Render and training resolution")
    parser.add_argument(
        "--camera-image-size",
        type=int,
        default=224,
        help="Resolution that the DUSt3R intrinsics were estimated at.",
    )
    parser.add_argument(
        "--camera-focal-mode",
        choices=("per_frame", "median"),
        default="median",
        help="How to use DUSt3R focal estimates across the sequence.",
    )
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
    parser.add_argument("--tokens", type=int, default=128, help="Number of latent 3D tokens before broadcasting")
    parser.add_argument(
        "--gaussians-per-token",
        type=int,
        default=4,
        help="Number of explicit Gaussians emitted by each latent token",
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
    parser.add_argument("--log-every", type=int, default=10, help="Log scalar loss to W&B every N steps")
    parser.add_argument("--image-log-every", type=int, default=50, help="Log preview image to W&B every N steps")
    parser.add_argument("--video-log-every", type=int, default=50, help="Log GT and rendered videos to W&B every N steps")
    parser.add_argument("--amp", action="store_true", help="Use autocast for the model forward pass")
    parser.add_argument("--wandb-project", type=str, default="dynamic-tokengs-overfit", help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, default="dynamic-sequence-run", help="W&B run name")
    return parser


def parse_args(default_renderer="auto"):
    return build_arg_parser(default_renderer=default_renderer).parse_args()


def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def resolve_camera_json_path(sequence_dir: Path, camera_json: Path | None) -> Path:
    if camera_json is not None:
        return camera_json
    return sequence_dir / "per_frame_cameras.json"


def infer_video_fps(records):
    timestamps = [record.get("timestamp_seconds") for record in records]
    diffs = []
    for left, right in zip(timestamps[:-1], timestamps[1:]):
        if left is None or right is None:
            continue
        delta = float(right) - float(left)
        if delta > 0:
            diffs.append(delta)
    if not diffs:
        return 1.0
    return float(1.0 / np.median(np.asarray(diffs, dtype=np.float32)))


def summarize_sequence_intrinsics(intrinsics: np.ndarray) -> dict[str, float]:
    return {
        "fx_median": float(np.median(intrinsics[:, 0, 0])),
        "fy_median": float(np.median(intrinsics[:, 1, 1])),
        "cx_median": float(np.median(intrinsics[:, 0, 2])),
        "cy_median": float(np.median(intrinsics[:, 1, 2])),
    }


def resolve_sequence_intrinsics(
    intrinsics: np.ndarray,
    focal_mode: str,
) -> np.ndarray:
    resolved = intrinsics.copy()
    if focal_mode == "median":
        summary = summarize_sequence_intrinsics(intrinsics)
        resolved[:, 0, 0] = summary["fx_median"]
        resolved[:, 1, 1] = summary["fy_median"]
        return resolved
    if focal_mode == "per_frame":
        return resolved
    raise ValueError(f"Unsupported camera focal mode: {focal_mode}")


def load_sequence_data(
    camera_json_path: Path,
    target_size: int,
    camera_image_size: int,
    max_frames: int,
    focal_mode: str,
    device,
):
    records = json.loads(camera_json_path.read_text())
    if max_frames > 0:
        records = records[:max_frames]
    if len(records) < 2:
        raise ValueError(f"Need at least 2 frame-camera records in {camera_json_path}")

    transform = T.Compose([T.Resize((target_size, target_size)), T.ToTensor()])
    scale = float(target_size) / float(camera_image_size)
    base_pose = torch.tensor(records[0]["camera_to_world"], dtype=torch.float32)
    base_pose_inv = torch.linalg.inv(base_pose)
    raw_intrinsics = np.stack([np.asarray(record["intrinsics"], dtype=np.float32) for record in records], axis=0)
    intrinsics_per_frame = resolve_sequence_intrinsics(raw_intrinsics, focal_mode=focal_mode)
    intrinsics_summary = summarize_sequence_intrinsics(intrinsics_per_frame)

    frames = []
    cameras = []
    timestamps = []

    for record, intrinsics in zip(records, intrinsics_per_frame):
        frame_path = Path(record["frame_path"])
        if not frame_path.exists():
            raise FileNotFoundError(f"Missing frame referenced by camera JSON: {frame_path}")
        frames.append(transform(Image.open(frame_path).convert("RGB")))

        pose = torch.tensor(record["camera_to_world"], dtype=torch.float32)
        pose = base_pose_inv @ pose
        cameras.append(
            CameraSpec(
                fx=float(intrinsics[0, 0] * scale),
                fy=float(intrinsics[1, 1] * scale),
                cx=float(intrinsics[0, 2] * scale),
                cy=float(intrinsics[1, 2] * scale),
                camera_to_world=pose.to(device=device),
            )
        )

        timestamp = record.get("timestamp_seconds")
        timestamps.append(float(timestamp) if timestamp is not None else None)

    times_np = np.asarray(
        [timestamp if timestamp is not None else index for index, timestamp in enumerate(timestamps)],
        dtype=np.float32,
    )
    if len(times_np) > 1 and float(times_np.max()) > float(times_np.min()):
        times_np = (times_np - times_np.min()) / (times_np.max() - times_np.min())
    else:
        times_np = np.zeros_like(times_np)

    return {
        "frames": torch.stack(frames, dim=0).to(device),
        "cameras": cameras,
        "frame_times": torch.from_numpy(times_np).to(device=device).unsqueeze(-1),
        "records": records,
        "video_fps": infer_video_fps(records),
        "intrinsics_summary": {
            "focal_mode": focal_mode,
            "raw_fx_median": float(np.median(raw_intrinsics[:, 0, 0])),
            "raw_fy_median": float(np.median(raw_intrinsics[:, 1, 1])),
            "resolved_fx_median": intrinsics_summary["fx_median"],
            "resolved_fy_median": intrinsics_summary["fy_median"],
            "resolved_cx_median": intrinsics_summary["cx_median"],
            "resolved_cy_median": intrinsics_summary["cy_median"],
            "training_scale": scale,
        },
    }


def normalize_model_outputs(outputs):
    normalized = []
    for tensor in outputs:
        normalized.append(tensor.unsqueeze(0) if tensor.ndim == 2 else tensor)
    return tuple(normalized)


def pick_renderer_mode(args):
    effective_gaussians = args.tokens * args.gaussians_per_token
    if args.renderer == "auto":
        renderer_mode = "dense" if effective_gaussians * args.size * args.size <= args.auto_dense_limit else "tiled"
    else:
        renderer_mode = args.renderer
    return renderer_mode, effective_gaussians


def render_one_frame(renderer_mode, args, dense_grid, camera, xyz, scales, quats, opacities, rgbs):
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


def select_window_indices(num_frames, frames_per_step, device):
    window = min(frames_per_step, num_frames)
    if window >= num_frames:
        return torch.arange(num_frames, device=device)
    start = torch.randint(0, num_frames - window + 1, (1,), device=device).item()
    return torch.arange(start, start + window, device=device)


def make_wandb_video(frames, fps):
    video = (frames.detach().cpu().clamp(0, 1) * 255.0).to(torch.uint8).numpy()
    return wandb.Video(video, fps=max(1, int(round(fps))), format="mp4")


@torch.no_grad()
def render_full_sequence(model, sequence_data, args, renderer_mode, dense_grid, amp_available, amp_dtype, device):
    was_training = model.training
    model.eval()
    rendered_frames = []
    num_frames = sequence_data["frames"].shape[0]

    for start in range(0, num_frames, args.eval_batch_size):
        end = min(start + args.eval_batch_size, num_frames)
        batch_frames = sequence_data["frames"][start:end]
        batch_times = sequence_data["frame_times"][start:end]
        batch_cameras = sequence_data["cameras"][start:end]

        autocast_context = (
            torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_available else nullcontext()
        )
        with fast_attn_context(device):
            with autocast_context:
                outputs = model(batch_frames, camera=batch_cameras, frame_times=batch_times)
        xyz, scales, quats, opacities, rgbs = normalize_model_outputs(outputs)

        for local_index, camera in enumerate(batch_cameras):
            rendered_frames.append(
                render_one_frame(
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
    return torch.stack(rendered_frames, dim=0)


def run_training(args):
    device = pick_device()
    print(f"Using device: {device}")

    camera_json_path = resolve_camera_json_path(args.sequence_dir, args.camera_json)
    sequence_data = load_sequence_data(
        camera_json_path=camera_json_path,
        target_size=args.size,
        camera_image_size=args.camera_image_size,
        max_frames=args.max_frames,
        focal_mode=args.camera_focal_mode,
        device=device,
    )
    num_frames = sequence_data["frames"].shape[0]
    print(f"Loaded {num_frames} frames from {camera_json_path}")
    intrinsics_summary = sequence_data["intrinsics_summary"]
    print(
        "Camera intrinsics: "
        f"focal_mode={intrinsics_summary['focal_mode']}, "
        f"median_fx={intrinsics_summary['resolved_fx_median'] * intrinsics_summary['training_scale']:.2f}, "
        f"median_fy={intrinsics_summary['resolved_fy_median'] * intrinsics_summary['training_scale']:.2f}"
    )

    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    model = DynamicTokenGS(num_tokens=args.tokens, gaussians_per_token=args.gaussians_per_token).to(device)
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
        "Starting DynamicTokenGS Training: "
        f"{num_frames} frames, {args.tokens} latent tokens x {args.gaussians_per_token} gaussians/token "
        f"= {effective_gaussians} explicit Gaussians with {renderer_mode} renderer..."
    )
    print(f"Attention backend: {attn_backend}")

    gt_video_logged = False
    pbar = tqdm(range(1, args.steps + 1))

    for step in pbar:
        optimizer.zero_grad(set_to_none=True)
        batch_indices = select_window_indices(num_frames, args.frames_per_step, device=device)
        batch_frames = sequence_data["frames"][batch_indices]
        batch_times = sequence_data["frame_times"][batch_indices]
        batch_cameras = [sequence_data["cameras"][index] for index in batch_indices.tolist()]

        autocast_context = (
            torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_available else nullcontext()
        )
        with fast_attn_context(device):
            with autocast_context:
                outputs = model(batch_frames, camera=batch_cameras, frame_times=batch_times)
        xyz, scales, quats, opacities, rgbs = normalize_model_outputs(outputs)

        renders = []
        losses = []
        for local_index, camera in enumerate(batch_cameras):
            render = render_one_frame(
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
            losses.append(F.l1_loss(render, target) + 0.2 * F.mse_loss(render, target))

        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()

        pbar.set_description(f"Loss: {loss.item():.4f}")

        should_log_scalars = step % max(1, args.log_every) == 0 or step == args.steps
        should_log_images = step % max(1, args.image_log_every) == 0 or step == args.steps
        should_log_videos = step % max(1, args.video_log_every) == 0 or step == args.steps
        if should_log_scalars:
            payload = {
                "Loss": loss.item(),
                "FramesPerStep": int(batch_indices.numel()),
                "SequenceFrames": num_frames,
            }
            if should_log_images:
                preview = torch.cat([batch_frames[0], renders[0].detach()], dim=2)
                payload["Render_GT_vs_Pred"] = wandb.Image(
                    T.ToPILImage()(preview.cpu().clamp(0, 1)),
                    caption=f"Step {step}",
                )
            if should_log_videos:
                rendered_sequence = render_full_sequence(
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
                if not gt_video_logged:
                    payload["GT_Video"] = make_wandb_video(gt_sequence, sequence_data["video_fps"])
                    gt_video_logged = True
            wandb.log(payload, step=step)

    print("DynamicTokenGS training complete. Check your Weights & Biases dashboard.")
    wandb.finish()


def main(default_renderer="auto", run_name="dynamic-sequence-run"):
    args = parse_args(default_renderer=default_renderer)
    if args.wandb_run_name == "dynamic-sequence-run":
        args.wandb_run_name = run_name
    run_training(args)


if __name__ == "__main__":
    main()
