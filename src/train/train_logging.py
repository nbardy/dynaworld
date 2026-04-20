from __future__ import annotations

from typing import Any

import torch
import wandb


def make_wandb_video(sequence: torch.Tensor, fps: float) -> wandb.Video:
    video = (sequence.detach().cpu().clamp(0, 1) * 255.0).to(torch.uint8).numpy()
    return wandb.Video(video, fps=max(1, int(round(float(fps)))), format="mp4")


def make_preview_image(target: torch.Tensor, render: torch.Tensor, caption: str) -> wandb.Image:
    preview = torch.cat([target.detach().cpu(), render.detach().cpu()], dim=-1)
    image = (preview.clamp(0, 1).permute(1, 2, 0) * 255.0).to(torch.uint8).numpy()
    return wandb.Image(image, caption=caption)


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


__all__ = [
    "build_validation_video_payload",
    "make_preview_image",
    "make_wandb_video",
]
