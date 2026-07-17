from __future__ import annotations

import math

import torch


def flatten_multiview_powerfoam_samples(
    frames: torch.Tensor,
    rays: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if frames.ndim != 5:
        raise ValueError(f"Expected multiview frames [V,T,C,H,W], got {tuple(frames.shape)}.")
    if rays.ndim != 5:
        raise ValueError(f"Expected multiview rays [V,T,H,W,6], got {tuple(rays.shape)}.")
    view_count, frame_count = int(frames.shape[0]), int(frames.shape[1])
    if tuple(rays.shape[:2]) != (view_count, frame_count):
        raise ValueError(f"Frame/ray view-time mismatch: {tuple(frames.shape[:2])} vs {tuple(rays.shape[:2])}.")
    targets = frames.reshape(view_count * frame_count, *frames.shape[2:]).contiguous()
    sample_frame_indices = torch.arange(frame_count, device=frames.device, dtype=torch.long).repeat(view_count)
    sample_rays = rays.reshape(view_count * frame_count, *rays.shape[2:]).contiguous()
    return targets, sample_frame_indices, sample_rays


def powerfoam_train_batch_indices(
    sample_count: int,
    cfg: dict,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    return torch.randint(
        0,
        int(sample_count),
        (int(cfg["train"]["frames_per_step"]),),
        device=device,
    )


def exp_scheduled_weight(initial: float, final_multiplier: float, step: int, total_steps: int) -> float:
    initial = float(initial)
    if initial <= 0.0:
        return initial
    final = initial * float(final_multiplier)
    if final <= 0.0:
        return final
    t = min(max(float(step) / max(float(total_steps), 1.0), 0.0), 1.0)
    return float(math.exp(math.log(initial) * (1.0 - t) + math.log(final) * t))


__all__ = [
    "exp_scheduled_weight",
    "flatten_multiview_powerfoam_samples",
    "powerfoam_train_batch_indices",
]
