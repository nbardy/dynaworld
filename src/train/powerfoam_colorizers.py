from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from colorize import FeatureToColor


DYNAMIC_POWERFOAM_COLORIZE_DEFAULTS = {
    "hidden_dim": None,
    "activation": "sigmoid",
    "pre_norm": False,
    "weight_init": "kaiming",
    "weight_init_gain": 1.0,
    "view_condition": "none",
    "detach_view_condition": True,
    "init_rgb_identity": True,
}


def init_colorizer_rgb_identity(colorizer: FeatureToColor) -> None:
    if colorizer.feature_dim < 3:
        raise ValueError("RGB identity colorizer init requires at least 3 feature channels")
    if colorizer.hidden_dim is not None or colorizer.pre_norm is not None:
        raise ValueError("colorize.init_rgb_identity requires hidden_dim=null and pre_norm=false")
    if not isinstance(colorizer.net, nn.Conv2d):
        raise ValueError("colorize.init_rgb_identity requires a single Conv2d colorizer")
    with torch.no_grad():
        colorizer.net.weight.zero_()
        colorizer.net.bias.zero_()
        colorizer.net.weight[0, 0, 0, 0] = 1.0
        colorizer.net.weight[1, 1, 0, 0] = 1.0
        colorizer.net.weight[2, 2, 0, 0] = 1.0


def build_dynamic_powerfoam_colorizer(
    cfg: Mapping[str, Any],
    *,
    device: torch.device,
    feature_dynamic_mode: str,
) -> FeatureToColor | None:
    model_cfg = cfg["model"]
    if str(model_cfg["dynamic_mode"]) != feature_dynamic_mode:
        return None
    colorize_cfg = cfg["colorize"]
    colorizer = FeatureToColor(
        feature_dim=int(model_cfg["feature_dim"]),
        hidden_dim=None if colorize_cfg["hidden_dim"] is None else int(colorize_cfg["hidden_dim"]),
        activation=str(colorize_cfg["activation"]),
        pre_norm=bool(colorize_cfg["pre_norm"]),
        weight_init=str(colorize_cfg["weight_init"]),
        weight_init_gain=float(colorize_cfg["weight_init_gain"]),
        view_condition=str(colorize_cfg["view_condition"]),
        detach_view_condition=bool(colorize_cfg["detach_view_condition"]),
    )
    if bool(colorize_cfg["init_rgb_identity"]):
        init_colorizer_rgb_identity(colorizer)
    return colorizer.to(device)


__all__ = [
    "DYNAMIC_POWERFOAM_COLORIZE_DEFAULTS",
    "build_dynamic_powerfoam_colorizer",
    "init_colorizer_rgb_identity",
]
