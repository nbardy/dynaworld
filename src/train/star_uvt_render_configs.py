from __future__ import annotations

from typing import Any

from star_uvt_feature_tube_model import FeatureTubeRenderConfig
from star_uvt_runtime import ensure_star_uvt_on_path


def feature_tube_render_config_from_cfg(cfg: dict[str, Any]) -> Any:
    return FeatureTubeRenderConfig(
        frames=int(cfg["data"]["max_frames"]),
        height=int(cfg["data"]["target_size"]),
        width=int(cfg["data"]["target_size"]),
        feature_dim=int(cfg["feature_uvt"]["feature_dim"]),
        alpha_threshold=float(cfg["feature_uvt"]["alpha_threshold"]),
        max_alpha=float(cfg["feature_uvt"]["max_alpha"]),
    )


def uvt_render_config_from_cfg(cfg: dict[str, Any], feature_config: Any | None = None) -> Any:
    ensure_star_uvt_on_path()
    from torch_gsplat_bridge_star_uvt import UVTRenderConfig

    if feature_config is None:
        feature_config = feature_tube_render_config_from_cfg(cfg)
    return UVTRenderConfig(
        height=int(feature_config.height),
        width=int(feature_config.width),
        frames=int(feature_config.frames),
        tile_t=int(cfg["feature_uvt"]["tile_t"]),
        tile_capacity=int(cfg["feature_uvt"]["tile_capacity"]),
        alpha_threshold=float(feature_config.alpha_threshold),
        max_alpha=float(feature_config.max_alpha),
    )


def star_uvt_render_configs_from_cfg(cfg: dict[str, Any]) -> tuple[Any, Any]:
    feature_config = feature_tube_render_config_from_cfg(cfg)
    return feature_config, uvt_render_config_from_cfg(cfg, feature_config)


__all__ = [
    "feature_tube_render_config_from_cfg",
    "star_uvt_render_configs_from_cfg",
    "uvt_render_config_from_cfg",
]
