from __future__ import annotations

from typing import Any

from config_utils import require_config_keys, require_config_sections
from star_uvt_config_keys import (
    REQUIRED_STAR_UVT_COLORIZE_KEYS,
    REQUIRED_STAR_UVT_LOGGING_KEYS,
    REQUIRED_STAR_UVT_OUTPUT_CHECKPOINT_KEYS,
    require_star_uvt_colorize_config,
    require_star_uvt_logging_config,
    require_star_uvt_output_config,
)
from star_uvt_feature_targets import FEATURE_TARGET_GRID_ADAPTERS


REQUIRED_SECTIONS = ("data", "features", "feature_target", "feature_uvt", "probe", "colorize", "output", "logging")
REQUIRED_FEATURE_UVT_KEYS = ("feature_dim",)
REQUIRED_PROBE_KEYS = (
    "steps",
    "lr",
    "device",
    "seed",
    "target_rgb_adapter",
    "require_loss_decrease",
)
REQUIRED_OUTPUT_KEYS = REQUIRED_STAR_UVT_OUTPUT_CHECKPOINT_KEYS
REQUIRED_LOGGING_KEYS = REQUIRED_STAR_UVT_LOGGING_KEYS
REQUIRED_COLORIZE_KEYS = REQUIRED_STAR_UVT_COLORIZE_KEYS


def resolve_config(config: dict[str, Any]) -> dict[str, Any]:
    require_config_sections(config, REQUIRED_SECTIONS)
    require_config_keys("feature_uvt", config["feature_uvt"], REQUIRED_FEATURE_UVT_KEYS)
    require_config_keys("probe", config["probe"], REQUIRED_PROBE_KEYS)
    require_star_uvt_colorize_config(config)
    require_star_uvt_output_config(config, checkpoint=True)
    require_star_uvt_logging_config(config)
    if str(config["probe"]["target_rgb_adapter"]) not in FEATURE_TARGET_GRID_ADAPTERS:
        expected = ", ".join(sorted(FEATURE_TARGET_GRID_ADAPTERS))
        raise ValueError(f"probe.target_rgb_adapter must be one of: {expected}")
    if int(config["probe"]["steps"]) <= 0:
        raise ValueError("probe.steps must be positive")
    if float(config["probe"]["lr"]) <= 0.0:
        raise ValueError("probe.lr must be positive")
    if not bool(config["feature_target"].get("enabled", False)):
        raise ValueError("feature_target.enabled must be true for the RGB probe")
    if str(config["feature_target"].get("materialization")) != "target_grid":
        raise ValueError("feature_target.materialization must be target_grid for the RGB probe")
    return config


__all__ = [
    "REQUIRED_COLORIZE_KEYS",
    "REQUIRED_FEATURE_UVT_KEYS",
    "REQUIRED_LOGGING_KEYS",
    "REQUIRED_OUTPUT_KEYS",
    "REQUIRED_PROBE_KEYS",
    "REQUIRED_SECTIONS",
    "resolve_config",
]
