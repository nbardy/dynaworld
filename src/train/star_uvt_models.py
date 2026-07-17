from __future__ import annotations

from typing import Any

import torch

from star_uvt_feature_tube_model import FeatureScreenTimeTubeModel


def build_feature_tube_model(
    cfg: dict[str, Any],
    feature_config: Any,
    *,
    device: torch.device,
    seed_section: str = "train",
) -> torch.nn.Module:
    return FeatureScreenTimeTubeModel(
        int(cfg["feature_uvt"]["tube_count"]),
        feature_config,
        seed=int(cfg[seed_section]["seed"]),
        device=device,
    )


__all__ = ["build_feature_tube_model"]
