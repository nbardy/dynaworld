from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn


def powerfoam_eval_batch_size(cfg: Mapping[str, Any]) -> int:
    return max(1, int(cfg["train"]["frames_per_step"]))


def rendered_alpha_from_powerfoam_output(output: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(output, "rendered") and hasattr(output, "alpha"):
        return output.rendered, output.alpha
    if isinstance(output, (tuple, list)) and len(output) >= 2:
        return output[0], output[1]
    raise TypeError("PowerFoam sample renderers must return rendered RGB and alpha.")


@torch.no_grad()
def render_powerfoam_samples(
    model: nn.Module,
    frame_indices: torch.Tensor,
    batch_size: int,
    rays: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    renders = []
    alphas = []
    device = next(model.parameters()).device
    frame_indices = frame_indices.to(device=device, dtype=torch.long)
    if rays is not None:
        rays = rays.to(device=device, dtype=torch.float32)
    for start in range(0, int(frame_indices.numel()), batch_size):
        end = min(start + batch_size, int(frame_indices.numel()))
        indices = frame_indices[start:end]
        batch_rays = None if rays is None else rays[start:end]
        output = model(indices, rays=batch_rays)
        rendered, alpha = rendered_alpha_from_powerfoam_output(output)
        renders.append(rendered.detach().cpu())
        alphas.append(alpha.detach().cpu())
    return torch.cat(renders, dim=0), torch.cat(alphas, dim=0)


__all__ = [
    "powerfoam_eval_batch_size",
    "render_powerfoam_samples",
    "rendered_alpha_from_powerfoam_output",
]
