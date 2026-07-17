from __future__ import annotations

from typing import Any

from config_utils import require_config_keys


REQUIRED_STAR_UVT_DATA_KEYS = (
    "video_path",
    "start_seconds",
    "fps",
    "duration_seconds",
    "image_crop_mode",
    "target_size",
    "max_frames",
)
REQUIRED_STAR_UVT_COLORIZE_KEYS = (
    "hidden_dim",
    "activation",
    "pre_norm",
    "weight_init",
    "weight_init_gain",
)
REQUIRED_STAR_UVT_OUTPUT_KEYS = (
    "out_json",
    "contact_sheet",
    "contact_sheet_frames",
    "contact_sheet_mode",
    "side_by_side_video",
    "side_by_side_fps",
)
REQUIRED_STAR_UVT_OUTPUT_CHECKPOINT_KEYS = (
    "out_json",
    "checkpoint",
    "contact_sheet",
    "contact_sheet_frames",
    "contact_sheet_mode",
    "side_by_side_video",
    "side_by_side_fps",
)
REQUIRED_STAR_UVT_LOGGING_KEYS = (
    "wandb_enabled",
    "wandb_project",
    "wandb_run_name",
    "wandb_tags",
    "wandb_mode",
)


def require_star_uvt_data_config(config: dict[str, Any]) -> None:
    require_config_keys("data", config["data"], REQUIRED_STAR_UVT_DATA_KEYS)


def require_star_uvt_colorize_config(config: dict[str, Any]) -> None:
    require_config_keys("colorize", config["colorize"], REQUIRED_STAR_UVT_COLORIZE_KEYS)


def require_star_uvt_output_config(config: dict[str, Any], *, checkpoint: bool = False) -> None:
    keys = REQUIRED_STAR_UVT_OUTPUT_CHECKPOINT_KEYS if checkpoint else REQUIRED_STAR_UVT_OUTPUT_KEYS
    require_config_keys("output", config["output"], keys)


def require_star_uvt_logging_config(config: dict[str, Any]) -> None:
    require_config_keys("logging", config["logging"], REQUIRED_STAR_UVT_LOGGING_KEYS)


__all__ = [
    "REQUIRED_STAR_UVT_COLORIZE_KEYS",
    "REQUIRED_STAR_UVT_DATA_KEYS",
    "REQUIRED_STAR_UVT_LOGGING_KEYS",
    "REQUIRED_STAR_UVT_OUTPUT_CHECKPOINT_KEYS",
    "REQUIRED_STAR_UVT_OUTPUT_KEYS",
    "require_star_uvt_colorize_config",
    "require_star_uvt_data_config",
    "require_star_uvt_logging_config",
    "require_star_uvt_output_config",
]
