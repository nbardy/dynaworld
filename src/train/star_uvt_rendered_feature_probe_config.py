from __future__ import annotations

from typing import Any

from config_utils import require_config_keys, require_config_sections
from star_uvt_config_keys import (
    REQUIRED_STAR_UVT_COLORIZE_KEYS,
    REQUIRED_STAR_UVT_DATA_KEYS,
    REQUIRED_STAR_UVT_LOGGING_KEYS,
    REQUIRED_STAR_UVT_OUTPUT_CHECKPOINT_KEYS,
    require_star_uvt_colorize_config,
    require_star_uvt_data_config,
    require_star_uvt_logging_config,
    require_star_uvt_output_config,
)
from star_uvt_rendered_feature_probe_objective import (
    RENDERED_FEATURE_PROBE_GRID_ADAPTERS,
    RENDERED_FEATURE_PROBE_PIXEL_SOURCES,
)


REQUIRED_SECTIONS = ("data", "probe", "feature_uvt", "colorize", "output", "logging")
REQUIRED_DATA_KEYS = REQUIRED_STAR_UVT_DATA_KEYS
REQUIRED_PROBE_KEYS = (
    "steps",
    "lr",
    "device",
    "seed",
    "frame_chunk_size",
    "resume_checkpoint",
    "pixel_source",
    "sample_grid_shape",
    "sample_grid_adapter",
    "require_loss_decrease",
)
REQUIRED_FEATURE_UVT_KEYS = (
    "tube_count",
    "feature_dim",
    "tile_t",
    "tile_capacity",
    "alpha_threshold",
    "max_alpha",
)
REQUIRED_COLORIZE_KEYS = REQUIRED_STAR_UVT_COLORIZE_KEYS
REQUIRED_OUTPUT_KEYS = REQUIRED_STAR_UVT_OUTPUT_CHECKPOINT_KEYS
REQUIRED_LOGGING_KEYS = REQUIRED_STAR_UVT_LOGGING_KEYS


def resolve_config(config: dict[str, Any]) -> dict[str, Any]:
    require_config_sections(config, REQUIRED_SECTIONS)
    require_star_uvt_data_config(config)
    require_config_keys("probe", config["probe"], REQUIRED_PROBE_KEYS)
    require_config_keys("feature_uvt", config["feature_uvt"], REQUIRED_FEATURE_UVT_KEYS)
    require_star_uvt_colorize_config(config)
    require_star_uvt_output_config(config, checkpoint=True)
    require_star_uvt_logging_config(config)
    if int(config["probe"]["steps"]) <= 0:
        raise ValueError("probe.steps must be positive")
    if float(config["probe"]["lr"]) <= 0.0:
        raise ValueError("probe.lr must be positive")
    config["probe"].setdefault("train_star_model", False)
    config["probe"].setdefault("train_colorizer", True)
    config["probe"].setdefault("colorizer_init_checkpoint", None)
    if not bool(config["probe"]["train_star_model"]) and not bool(config["probe"]["train_colorizer"]):
        raise ValueError("at least one of probe.train_star_model or probe.train_colorizer must be true")
    if not bool(config["probe"]["train_colorizer"]) and config["probe"]["colorizer_init_checkpoint"] is None:
        raise ValueError("probe.colorizer_init_checkpoint is required when probe.train_colorizer=false")
    if config["probe"]["resume_checkpoint"] is None:
        raise ValueError("probe.resume_checkpoint is required")
    pixel_source = str(config["probe"]["pixel_source"])
    if pixel_source not in RENDERED_FEATURE_PROBE_PIXEL_SOURCES:
        expected = ", ".join(sorted(RENDERED_FEATURE_PROBE_PIXEL_SOURCES))
        raise ValueError(f"probe.pixel_source must be one of: {expected}")
    if str(config["probe"]["sample_grid_adapter"]) not in RENDERED_FEATURE_PROBE_GRID_ADAPTERS:
        expected = ", ".join(sorted(RENDERED_FEATURE_PROBE_GRID_ADAPTERS))
        raise ValueError(f"probe.sample_grid_adapter must be one of: {expected}")
    sample_grid_shape = config["probe"]["sample_grid_shape"]
    if (
        not isinstance(sample_grid_shape, list | tuple)
        or len(sample_grid_shape) != 3
        or any(int(item) <= 0 for item in sample_grid_shape)
    ):
        raise ValueError("probe.sample_grid_shape must be [frames, height, width]")
    if pixel_source == "stratified_grid":
        max_shape = (
            int(config["data"]["max_frames"]),
            int(config["data"]["target_size"]),
            int(config["data"]["target_size"]),
        )
        if any(int(requested) > int(limit) for requested, limit in zip(sample_grid_shape, max_shape, strict=True)):
            raise ValueError("stratified_grid sample_grid_shape cannot exceed [max_frames, target_size, target_size]")
    if int(config["probe"]["frame_chunk_size"]) <= 0:
        raise ValueError("probe.frame_chunk_size must be positive")
    if int(config["feature_uvt"]["feature_dim"]) <= 0:
        raise ValueError("feature_uvt.feature_dim must be positive")
    return config


__all__ = [
    "REQUIRED_COLORIZE_KEYS",
    "REQUIRED_DATA_KEYS",
    "REQUIRED_FEATURE_UVT_KEYS",
    "REQUIRED_LOGGING_KEYS",
    "REQUIRED_OUTPUT_KEYS",
    "REQUIRED_PROBE_KEYS",
    "REQUIRED_SECTIONS",
    "resolve_config",
]
