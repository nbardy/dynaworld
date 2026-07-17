from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import wandb

from video_io import alpha_to_rgb_video


def make_wandb_video(sequence: torch.Tensor, fps: float) -> wandb.Video:
    video = (sequence.detach().cpu().clamp(0, 1) * 255.0).to(torch.uint8).numpy()
    return wandb.Video(video, fps=max(1, int(round(float(fps)))), format="mp4")


def make_preview_image(target: torch.Tensor, render: torch.Tensor, caption: str) -> wandb.Image:
    preview = torch.cat([target.detach().cpu(), render.detach().cpu()], dim=-1)
    image = (preview.clamp(0, 1).permute(1, 2, 0) * 255.0).to(torch.uint8).numpy()
    return wandb.Image(image, caption=caption)


def make_wandb_image(image: Any, caption: str) -> wandb.Image:
    return wandb.Image(image, caption=caption)


def make_step_preview_image(target: torch.Tensor, render: torch.Tensor, step: int) -> wandb.Image:
    return make_preview_image(target, render, caption=f"step {step}: GT | render")


def add_existing_wandb_media(
    payload: dict[str, Any],
    output_cfg: Mapping[str, Any],
    *,
    image_outputs: tuple[tuple[str, str], ...] = (),
    video_outputs: tuple[tuple[str, str], ...] = (),
) -> None:
    for config_key, payload_key in image_outputs:
        raw_path = output_cfg.get(config_key)
        if raw_path is None:
            continue
        path = Path(raw_path)
        if path.exists():
            payload[payload_key] = wandb.Image(str(path))
    for config_key, payload_key in video_outputs:
        raw_path = output_cfg.get(config_key)
        if raw_path is None:
            continue
        path = Path(raw_path)
        if path.exists():
            payload[payload_key] = wandb.Video(str(path), format="mp4")


def build_validation_video_payload(
    rendered_sequence: torch.Tensor,
    target_sequence: torch.Tensor,
    fps: float,
) -> dict[str, Any]:
    rendered = rendered_sequence.detach().cpu()
    target = target_sequence.detach().cpu()
    side_by_side = torch.cat([target, rendered], dim=-1)
    return {
        "Render_Video": make_wandb_video(rendered, fps),
        "Render_GT_Video": make_wandb_video(side_by_side, fps),
    }


def build_rgb_alpha_validation_video_payload(
    rendered_sequence: torch.Tensor,
    target_sequence: torch.Tensor,
    alpha_sequence: torch.Tensor,
    fps: float,
) -> dict[str, Any]:
    """Validation video payload for RGB reconstructions with alpha masks."""

    target = target_sequence.detach().cpu()
    payload = build_validation_video_payload(rendered_sequence, target, fps)
    payload["GT_Video"] = make_wandb_video(target, fps)
    payload["Alpha_Video"] = make_wandb_video(alpha_to_rgb_video(alpha_sequence), fps)
    return payload


def build_rgb_alpha_eval_media_payload(
    rendered_sequence: torch.Tensor,
    target_sequence: torch.Tensor,
    alpha_sequence: torch.Tensor,
    *,
    step: int,
    fps: float,
    include_videos: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "Preview": make_step_preview_image(target_sequence[0], rendered_sequence[0], step),
    }
    if include_videos:
        payload.update(build_rgb_alpha_validation_video_payload(rendered_sequence, target_sequence, alpha_sequence, fps))
    return payload


__all__ = [
    "add_existing_wandb_media",
    "alpha_to_rgb_video",
    "build_rgb_alpha_eval_media_payload",
    "build_rgb_alpha_validation_video_payload",
    "build_validation_video_payload",
    "make_preview_image",
    "make_step_preview_image",
    "make_wandb_image",
    "make_wandb_video",
]
