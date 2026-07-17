from __future__ import annotations

from typing import Any

import torch

from dynamic_gauge_foam import DynamicGaugeFoamVideo


def dynamic_gauge_render_kwargs(cfg: dict[str, Any]) -> dict[str, float | int]:
    return {
        "chunk_pixels": int(cfg["render"]["chunk_pixels"]),
        "max_hits": int(cfg["render"]["max_hits"]),
        "near": float(cfg["render"]["near"]),
        "far": float(cfg["render"]["far"]),
        "falloff": float(cfg["render"]["falloff"]),
        "min_alpha": float(cfg["render"]["min_alpha"]),
        "background_feature": float(cfg["render"]["background_feature"]),
    }


@torch.no_grad()
def render_dynamic_gauge_sequence(
    model: DynamicGaugeFoamVideo,
    frame_count: int,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    renders = []
    alphas = []
    depths = []
    for frame_index in range(int(frame_count)):
        indices = torch.tensor([frame_index], device=device, dtype=torch.long)
        out = model(indices, **dynamic_gauge_render_kwargs(cfg))
        renders.append(out.rgb.permute(0, 3, 1, 2).detach().cpu())
        alphas.append(out.alpha[..., 0].detach().cpu())
        depths.append(out.depth[..., 0].detach().cpu())
    return torch.cat(renders, dim=0), torch.cat(alphas, dim=0), torch.cat(depths, dim=0)


__all__ = [
    "dynamic_gauge_render_kwargs",
    "render_dynamic_gauge_sequence",
]
