from __future__ import annotations

from pathlib import Path
from typing import Any

from config_utils import apply_defaults, resolved_config


DATA_DEFAULTS = {
    "video_path": "test_data/test_video_small_128_4fps.mp4",
    "frame_source": "explicit_video",
    "max_frames": 16,
}
MODEL_DEFAULTS = {
    "primitives": 512,
    "feature_dim": 8,
    "atlas_res": 4,
    "num_time_ctrl": 8,
    "init_depth": 2.0,
    "radius_scale": 1.65,
    "opacity_init": 0.92,
    "feature_noise": 0.01,
    "color_hidden_dim": 64,
    "rgb_skip": True,
}
RENDER_DEFAULTS = {
    "render_size": 64,
    "fov_degrees": 55.0,
    "chunk_pixels": 1024,
    "max_hits": 8,
    "near": 0.05,
    "far": 100.0,
    "falloff": 2.5,
    "min_alpha": 1.0e-4,
    "background_feature": 0.0,
}
TRAIN_DEFAULTS = {
    "steps": 120,
    "frames_per_step": 1,
    "lr": 0.01,
    "center_lr_multiplier": 0.25,
    "radius_lr_multiplier": 0.1,
    "opacity_lr_multiplier": 0.1,
    "twist_lr_multiplier": 0.35,
    "atlas_lr_multiplier": 1.0,
    "color_lr_multiplier": 0.5,
    "seed": 17,
    "device": "auto",
}
LOSS_DEFAULTS = {
    "l1_weight": 1.0,
    "mse_weight": 0.1,
    "connection_weight": 0.01,
    "temporal_weight": 0.001,
    "opacity_weight": 1.0e-4,
    "radius_weight": 1.0e-4,
    "atlas_tv_weight": 1.0e-4,
    "knn_k": 8,
}
LOGGING_DEFAULTS = {
    "log_every": 30,
    "image_log_every": 60,
    "video_log_every": 120,
    "always_log_last_step": True,
    "output_dir": "outputs/dynamic_gauge_foam/local_mac_dynamic_gauge_foam_video_512_smoke",
    "wandb_enabled": False,
    "wandb_project": "dynaworld",
    "wandb_run_name": "dynamic-gauge-foam-video-512-smoke",
    "wandb_tags": ["dynamic-gauge-foam", "direct-fit", "video"],
    "wandb_mode": None,
}


def resolve_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = resolved_config(config, ("data", "model", "render", "train", "losses", "logging"))
    apply_defaults(cfg["data"], DATA_DEFAULTS)
    apply_defaults(cfg["model"], MODEL_DEFAULTS)
    apply_defaults(cfg["render"], RENDER_DEFAULTS)
    apply_defaults(cfg["train"], TRAIN_DEFAULTS)
    apply_defaults(cfg["losses"], LOSS_DEFAULTS)
    apply_defaults(cfg["logging"], LOGGING_DEFAULTS)
    cfg["data"]["video_path"] = Path(cfg["data"]["video_path"])
    cfg["logging"]["output_dir"] = Path(cfg["logging"]["output_dir"])
    if int(cfg["model"]["primitives"]) < 1:
        raise ValueError("model.primitives must be positive")
    if int(cfg["model"]["feature_dim"]) < 3:
        raise ValueError("model.feature_dim must be at least 3")
    if int(cfg["model"]["atlas_res"]) < 1:
        raise ValueError("model.atlas_res must be positive")
    if int(cfg["model"]["num_time_ctrl"]) < 1:
        raise ValueError("model.num_time_ctrl must be positive")
    if int(cfg["render"]["chunk_pixels"]) < 1:
        raise ValueError("render.chunk_pixels must be positive")
    if int(cfg["render"]["max_hits"]) < 1:
        raise ValueError("render.max_hits must be positive")
    if int(cfg["train"]["steps"]) < 1:
        raise ValueError("train.steps must be positive")
    if int(cfg["train"]["frames_per_step"]) < 1:
        raise ValueError("train.frames_per_step must be positive")
    return cfg


__all__ = [
    "DATA_DEFAULTS",
    "LOGGING_DEFAULTS",
    "LOSS_DEFAULTS",
    "MODEL_DEFAULTS",
    "RENDER_DEFAULTS",
    "TRAIN_DEFAULTS",
    "resolve_config",
]
