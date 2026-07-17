from __future__ import annotations

from typing import Any

from config_utils import require_config_keys, require_config_sections
from star_uvt_config_keys import (
    REQUIRED_STAR_UVT_DATA_KEYS,
    REQUIRED_STAR_UVT_LOGGING_KEYS,
    REQUIRED_STAR_UVT_OUTPUT_KEYS,
    require_star_uvt_data_config,
    require_star_uvt_logging_config,
    require_star_uvt_output_config,
)


REQUIRED_SECTIONS = ("data", "train", "uvt", "per_frame", "output", "logging")
REQUIRED_DATA_KEYS = REQUIRED_STAR_UVT_DATA_KEYS
REQUIRED_TRAIN_KEYS = ("steps", "lr", "device", "seed", "render_benchmark_repeats", "require_loss_decrease")
REQUIRED_UVT_KEYS = (
    "tube_count",
    "init_mode",
    "spatial_precision",
    "temporal_precision",
    "opacity",
    "sample_mode",
    "velocity_init",
    "velocity_search_radius",
    "velocity_patch_radius",
    "velocity_min_improvement_ratio",
    "final_lr",
    "final_lr_start_step",
    "coarse_target_size",
    "coarse_steps",
    "coarse_lr",
    "appearance_refine_steps",
    "appearance_lr",
    "temporal_split_step",
    "temporal_split_offset",
    "temporal_split_precision_scale",
    "temporal_split_opacity_scale",
    "temporal_split_depth_offset",
    "temporal_split_lr",
    "render_backend",
    "reduction_mode",
    "sample_emission_mode",
    "tile_t",
    "tile_capacity",
    "tile_load_reg_weight",
    "tile_load_target",
    "skip_uvt",
)
REQUIRED_PER_FRAME_KEYS = (
    "splats",
    "lr",
    "init_mode",
    "render_backend",
    "fast_max_pairs",
    "spatial_precision",
    "opacity",
    "sample_mode",
    "skip_per_frame",
)
REQUIRED_OUTPUT_KEYS = REQUIRED_STAR_UVT_OUTPUT_KEYS
REQUIRED_LOGGING_KEYS = REQUIRED_STAR_UVT_LOGGING_KEYS


def resolve_config(config: dict[str, Any]) -> dict[str, Any]:
    require_config_sections(config, REQUIRED_SECTIONS)
    require_star_uvt_data_config(config)
    require_config_keys("train", config["train"], REQUIRED_TRAIN_KEYS)
    require_config_keys("uvt", config["uvt"], REQUIRED_UVT_KEYS)
    require_config_keys("per_frame", config["per_frame"], REQUIRED_PER_FRAME_KEYS)
    require_star_uvt_output_config(config)
    require_star_uvt_logging_config(config)
    return config


__all__ = [
    "REQUIRED_DATA_KEYS",
    "REQUIRED_LOGGING_KEYS",
    "REQUIRED_OUTPUT_KEYS",
    "REQUIRED_PER_FRAME_KEYS",
    "REQUIRED_SECTIONS",
    "REQUIRED_TRAIN_KEYS",
    "REQUIRED_UVT_KEYS",
    "resolve_config",
]
