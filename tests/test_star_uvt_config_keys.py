from __future__ import annotations

import pytest

from config_utils import load_config_file
from star_uvt_config_keys import (
    REQUIRED_STAR_UVT_COLORIZE_KEYS,
    REQUIRED_STAR_UVT_LOGGING_KEYS,
    REQUIRED_STAR_UVT_OUTPUT_CHECKPOINT_KEYS,
    REQUIRED_STAR_UVT_OUTPUT_KEYS,
    require_star_uvt_colorize_config,
    require_star_uvt_logging_config,
    require_star_uvt_output_config,
)
from star_uvt_video_overfit_config import resolve_config as resolve_star_uvt_video_config


STAR_UVT_VIDEO_CONFIG = "src/train_configs/star_uvt_rgb_testvideo_64f_512_directatomic_8192t_20step_media.jsonc"


def test_star_uvt_output_checkpoint_keys_extend_base_output_keys() -> None:
    assert set(REQUIRED_STAR_UVT_OUTPUT_KEYS) < set(REQUIRED_STAR_UVT_OUTPUT_CHECKPOINT_KEYS)
    assert "checkpoint" in REQUIRED_STAR_UVT_OUTPUT_CHECKPOINT_KEYS


def test_star_uvt_output_config_requires_checkpoint_when_requested() -> None:
    cfg = {
        "output": {
            "out_json": None,
            "contact_sheet": None,
            "contact_sheet_frames": 1,
            "contact_sheet_mode": "first",
            "side_by_side_video": None,
            "side_by_side_fps": None,
        }
    }

    require_star_uvt_output_config(cfg)
    with pytest.raises(KeyError, match="checkpoint"):
        require_star_uvt_output_config(cfg, checkpoint=True)


def test_star_uvt_colorize_and_logging_helpers_report_missing_section_keys() -> None:
    colorize_cfg = {"colorize": {key: None for key in REQUIRED_STAR_UVT_COLORIZE_KEYS if key != "activation"}}
    logging_cfg = {"logging": {key: None for key in REQUIRED_STAR_UVT_LOGGING_KEYS if key != "wandb_tags"}}

    with pytest.raises(KeyError, match="activation"):
        require_star_uvt_colorize_config(colorize_cfg)
    with pytest.raises(KeyError, match="wandb_tags"):
        require_star_uvt_logging_config(logging_cfg)


def test_star_uvt_video_overfit_config_resolves_checked_in_config() -> None:
    cfg = load_config_file(STAR_UVT_VIDEO_CONFIG)

    resolved = resolve_star_uvt_video_config(cfg)

    assert resolved["arch"] == "star_uvt_video_overfit"
    assert resolved["uvt"]["render_backend"] == cfg["uvt"]["render_backend"]


def test_star_uvt_video_overfit_config_requires_per_frame_section_keys() -> None:
    cfg = load_config_file(STAR_UVT_VIDEO_CONFIG)
    del cfg["per_frame"]["fast_max_pairs"]

    with pytest.raises(KeyError, match="fast_max_pairs"):
        resolve_star_uvt_video_config(cfg)
