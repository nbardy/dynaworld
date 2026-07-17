from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch


def adam_with_device_fused(
    params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
    *,
    lr: float,
    device: torch.device | str,
) -> torch.optim.Adam:
    """Build Adam with the trainer's shared fused-kernel policy."""
    device_type = torch.device(device).type
    return torch.optim.Adam(
        params,
        lr=float(lr),
        fused=device_type in {"cuda", "mps"},
    )


def optimizer_backward_step(
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
    *,
    clip_grad_params: Iterable[torch.nn.Parameter] | None = None,
    max_grad_norm: float | None = None,
) -> None:
    if max_grad_norm is not None and clip_grad_params is None:
        raise ValueError("clip_grad_params is required when max_grad_norm is set.")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(clip_grad_params, float(max_grad_norm))
    optimizer.step()


__all__ = ["adam_with_device_fused", "optimizer_backward_step"]
