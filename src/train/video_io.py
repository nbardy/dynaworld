from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch


def tensor_to_uint8_hwc(image: torch.Tensor) -> Any:
    return (image.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0) * 255.0).to(torch.uint8).numpy()


def video_to_uint8_nhwc(frames: torch.Tensor) -> Any:
    return (frames.detach().cpu().clamp(0.0, 1.0) * 255.0).to(torch.uint8).permute(0, 2, 3, 1).numpy()


def save_png(path: Path, image: torch.Tensor) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tensor_to_uint8_hwc(image)).save(path)


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
