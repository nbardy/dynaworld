from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from colorize import FeatureToColor


DEFAULT_FEATURE_COLORIZE_CFG: dict[str, object] = {
    "hidden_dim": None,
    "activation": "sigmoid",
    "pre_norm": True,
    "weight_init": "kaiming",
    "weight_init_gain": 4.0,
}


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def build_feature_colorizer(
    colorize_cfg: Mapping[str, Any],
    *,
    feature_dim: int,
    device: torch.device,
) -> FeatureToColor:
    return FeatureToColor(
        feature_dim=int(feature_dim),
        hidden_dim=_optional_int(colorize_cfg["hidden_dim"]),
        activation=str(colorize_cfg["activation"]),
        pre_norm=bool(colorize_cfg["pre_norm"]),
        weight_init=str(colorize_cfg["weight_init"]),
        weight_init_gain=float(colorize_cfg["weight_init_gain"]),
    ).to(device)


def build_default_feature_colorizer(
    *,
    feature_dim: int,
    device: torch.device | str,
) -> FeatureToColor:
    return build_feature_colorizer(
        DEFAULT_FEATURE_COLORIZE_CFG,
        feature_dim=feature_dim,
        device=torch.device(device),
    )


def set_module_trainable(module: torch.nn.Module, trainable: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(trainable)
    if trainable:
        module.train()
    else:
        module.eval()


__all__ = [
    "DEFAULT_FEATURE_COLORIZE_CFG",
    "build_default_feature_colorizer",
    "build_feature_colorizer",
    "set_module_trainable",
]
