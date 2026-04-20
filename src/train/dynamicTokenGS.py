import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from fast_attn import configure_fast_attn, fast_attn_context
from gs_models import DynamicTokenGS
from renderers.common import build_pixel_grid
from rendering import pick_renderer_mode as resolve_renderer_mode
from rendering import render_gaussian_frame
from runtime_types import GaussianFrame
from sequence_data import load_camera_sequence, select_window_indices
from tqdm import tqdm
from train_logging import build_validation_video_payload, make_preview_image, make_wandb_video

ROOT = Path(__file__).resolve().parents[2]
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
    parser.add_argument(
        "--video-log-every", type=int, default=50, help="Log GT and rendered videos to W&B every N steps"
    )
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


def normalize_model_outputs(outputs):
    normalized = []
    for tensor in outputs:
        normalized.append(tensor.unsqueeze(0) if tensor.ndim == 2 else tensor)
    return tuple(normalized)


def pick_renderer_mode(args):
    effective_gaussians = args.tokens * args.gaussians_per_token
    renderer_mode = resolve_renderer_mode(
        renderer=args.renderer,
        gaussian_count=effective_gaussians,
        height=args.size,
        width=args.size,
        auto_dense_limit=args.auto_dense_limit,
    )
    return renderer_mode, effective_gaussians


def render_one_frame(renderer_mode, args, dense_grid, camera, xyz, scales, quats, opacities, rgbs):
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
    num_frames = sequence_data["frames"].shape[0]

    for start in range(0, num_frames, args.eval_batch_size):
        end = min(start + args.eval_batch_size, num_frames)
        batch_frames = sequence_data["frames"][start:end]
        batch_times = sequence_data["frame_times"][start:end]
        batch_cameras = sequence_data["cameras"][start:end]

        autocast_context = torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_available else nullcontext()
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

        autocast_context = torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_available else nullcontext()
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
                payload["Render_GT_vs_Pred"] = make_preview_image(
                    batch_frames[0],
                    renders[0],
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
                payload.update(
                    build_validation_video_payload(
                        rendered_sequence,
                        gt_sequence,
                        sequence_data["video_fps"],
                    )
                )
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
