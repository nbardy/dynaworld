from __future__ import annotations

from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _lpips_alex_cpu():
    import lpips

    model = lpips.LPIPS(net="alex", verbose=False).eval().cpu()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.no_grad()
def video_lpips(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    batch_size: int = 8,
) -> float:
    """Mean AlexNet LPIPS for a video, evaluated in bounded CPU batches."""

    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError(
            f"LPIPS expects matching [T,C,H,W] or [T,H,W,C] videos, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if batch_size < 1:
        raise ValueError("LPIPS batch_size must be positive")
    if prediction.shape[-1] == 3:
        prediction = prediction.permute(0, 3, 1, 2)
        target = target.permute(0, 3, 1, 2)
    elif prediction.shape[1] != 3:
        raise ValueError("LPIPS videos must have three color channels")
    prediction = prediction.detach().float().cpu().clamp(0.0, 1.0).mul(2.0).sub(1.0)
    target = target.detach().float().cpu().clamp(0.0, 1.0).mul(2.0).sub(1.0)
    model = _lpips_alex_cpu()
    total = 0.0
    count = 0
    for start in range(0, prediction.shape[0], batch_size):
        values = model(prediction[start : start + batch_size], target[start : start + batch_size])
        total += float(values.sum().item())
        count += int(values.numel())
    if count == 0:
        raise ValueError("LPIPS video must contain at least one frame")
    return total / float(count)


__all__ = ["video_lpips"]
