from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from dynamic_gauge_foam import (
    DynamicGaugeFoamVideo,
    GaugeFoamRenderOutput,
    atlas_total_variation,
    gauge_connection_loss,
    temporal_accel_loss,
)


def dynamic_gauge_training_loss(
    model: DynamicGaugeFoamVideo,
    render_output: GaugeFoamRenderOutput,
    target: torch.Tensor,
    frame_indices: torch.Tensor,
    edge_index: torch.Tensor,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    rendered = render_output.rgb.permute(0, 3, 1, 2)
    l1 = F.l1_loss(rendered, target)
    mse = F.mse_loss(rendered, target)

    frame_times = model.frame_times[frame_indices]
    foam = model.evaluate_times(frame_times)
    dt = 1.0 / max(int(cfg["model"]["num_time_ctrl"]) - 1, 1)
    prev_foam = model.evaluate_times((frame_times - dt).clamp(0.0, 1.0))
    next_foam = model.evaluate_times((frame_times + dt).clamp(0.0, 1.0))
    connection = gauge_connection_loss(foam.centers, foam.rotations, model.p0.detach(), edge_index)
    temporal = temporal_accel_loss(prev_foam.centers, foam.centers, next_foam.centers)
    opacity = foam.opacities.mean()
    radius = foam.radii.square().mean()
    atlas_tv = atlas_total_variation(model.atlas)

    loss = (
        float(cfg["losses"]["l1_weight"]) * l1
        + float(cfg["losses"]["mse_weight"]) * mse
        + float(cfg["losses"]["connection_weight"]) * connection
        + float(cfg["losses"]["temporal_weight"]) * temporal
        + float(cfg["losses"]["opacity_weight"]) * opacity
        + float(cfg["losses"]["radius_weight"]) * radius
        + float(cfg["losses"]["atlas_tv_weight"]) * atlas_tv
    )
    return loss, {
        "l1": l1,
        "mse": mse,
        "connection": connection,
        "temporal": temporal,
        "opacity": opacity,
        "radius": radius,
        "atlas_tv": atlas_tv,
    }


__all__ = ["dynamic_gauge_training_loss"]
