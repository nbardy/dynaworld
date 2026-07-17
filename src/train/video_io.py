from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping

import torch


def tensor_to_uint8_hwc(image: torch.Tensor) -> Any:
    return (image.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0) * 255.0).to(torch.uint8).numpy()


def video_to_uint8_nhwc(frames: torch.Tensor) -> Any:
    return (frames.detach().cpu().clamp(0.0, 1.0) * 255.0).to(torch.uint8).permute(0, 2, 3, 1).numpy()


def video_fps_from_config(cfg: Mapping[str, Any], *, default: float = 4.0) -> float:
    return float(cfg.get("video_fps", default))


def save_png(path: Path, image: torch.Tensor) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tensor_to_uint8_hwc(image)).save(path)


def alpha_to_rgb_video(alpha_sequence: torch.Tensor) -> torch.Tensor:
    """Expand [T,H,W] or [T,1,H,W] alpha masks into RGB video tensors."""

    alpha = alpha_sequence.detach().cpu()
    if alpha.ndim == 3:
        alpha = alpha.unsqueeze(1)
    if alpha.ndim != 4 or alpha.shape[1] != 1:
        raise ValueError(f"Expected alpha video with shape [T,H,W] or [T,1,H,W], got {tuple(alpha.shape)}")
    return alpha.repeat(1, 3, 1, 1)


def rgb_alpha_preview(target: torch.Tensor, render: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    alpha_rgb = alpha_to_rgb_video(alpha.unsqueeze(0) if alpha.ndim in (2, 3) else alpha)[0]
    return torch.cat([target.detach().cpu(), render.detach().cpu(), alpha_rgb], dim=-1)


def save_rgb_alpha_preview(path: Path, target: torch.Tensor, render: torch.Tensor, alpha: torch.Tensor) -> None:
    save_png(path, rgb_alpha_preview(target, render, alpha))


def save_mp4(path: Path, frames: torch.Tensor, fps: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for H.264/avc1 MP4 artifact writing")

    path.parent.mkdir(parents=True, exist_ok=True)
    video = video_to_uint8_nhwc(frames)
    height, width = video.shape[1], video.shape[2]
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(max(1.0, float(fps))),
        "-i",
        "-",
        "-an",
        "-vf",
        "format=yuv420p,setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709",
        "-c:v",
        "libx264",
        "-profile:v",
        "baseline",
        "-level",
        "3.1",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-colorspace",
        "bt709",
        "-tag:v",
        "avc1",
        "-movflags",
        "+faststart",
        str(path),
    ]
    result = subprocess.run(cmd, input=video.tobytes(), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed while writing video: {path}")


def save_render_side_by_side_videos(
    output_dir: Path,
    step: int,
    renders: torch.Tensor,
    targets: torch.Tensor,
    *,
    fps: float,
    prefix: str = "",
) -> None:
    renders_cpu = renders.detach().cpu()
    targets_cpu = targets.detach().cpu()
    save_mp4(output_dir / f"{prefix}render_step_{step:04d}.mp4", renders_cpu, fps=fps)
    save_mp4(output_dir / f"{prefix}side_by_side_step_{step:04d}.mp4", torch.cat([targets_cpu, renders_cpu], dim=-1), fps=fps)


def save_rgb_alpha_eval_media(
    output_dir: Path,
    step: int,
    renders: torch.Tensor,
    targets: torch.Tensor,
    alphas: torch.Tensor,
    *,
    fps: float,
    save_videos: bool,
    heldout_renders: torch.Tensor | None = None,
    heldout_targets: torch.Tensor | None = None,
    heldout_alphas: torch.Tensor | None = None,
) -> None:
    save_rgb_alpha_preview(output_dir / f"preview_step_{step:04d}.png", targets[0], renders[0], alphas[0])
    if heldout_renders is not None and heldout_targets is not None and heldout_alphas is not None:
        save_rgb_alpha_preview(
            output_dir / f"heldout_preview_step_{step:04d}.png",
            heldout_targets[0],
            heldout_renders[0],
            heldout_alphas[0],
        )
    if save_videos:
        save_render_side_by_side_videos(output_dir, step, renders, targets, fps=fps)
        if heldout_renders is not None and heldout_targets is not None:
            save_render_side_by_side_videos(
                output_dir,
                step,
                heldout_renders,
                heldout_targets,
                fps=fps,
                prefix="heldout_",
            )
