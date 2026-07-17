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
    "cells": 96,
    "neighbor_count": 16,
    "init_from_video": False,
    "image_init_depth": None,
    "image_init_jitter": 0.0,
    "num_texel_sites": 8,
    "sv_dof": 8,
    "sv_axis_init": 8.0,
    "radius_scale": 0.75,
    "adjacency_mode": "cech_aabb",
    "rebuild_adjacency_every": 10,
    "xy_extent": 1.25,
    "z_min": 1.0,
    "z_max": 3.25,
    "radius_init": 0.45,
    "radius_min": 0.03,
    "density_init": 36.0,
}
RENDER_DEFAULTS = {
    "render_size": 128,
    "fov_degrees": 55.0,
    "near_plane": 0.05,
    "alpha_threshold": 0.0,
    "transmittance_threshold": 1.0e-4,
    "max_alpha": 0.99,
    "eps": 1.0e-6,
    "texel_temperature": 10.0,
    "background": [0.0, 0.0, 0.0],
}
TRAIN_DEFAULTS = {
    "steps": 250,
    "frames_per_step": 1,
    "lr": 0.03,
    "use_param_groups": True,
    "seed": 17,
    "device": "auto",
}
LOSS_DEFAULTS = {
    "l1_weight": 1.0,
    "mse_weight": 0.1,
    "rgb_mse_sum_weight": 0.0,
    "ssim_weight": 0.2,
    "ssim_window_size": 11,
    "ssim_c1": 0.0001,
    "ssim_c2": 0.0009,
    "normal_weight": 0.1,
    "normal_weight_start_step": 0,
    "normal_weight_final_multiplier": 0.1,
    "normal_map_weight": 0.0,
    "normal_map_weight_start_step": 0,
    "normal_map_weight_final_multiplier": 1.0,
    "contribution_weight": 0.1,
    "contribution_weight_start_step": 0,
    "contribution_weight_final_multiplier": 0.001,
    "interpenetration_weight": 1.0e-4,
    "interpenetration_weight_start_step": 0,
    "interpenetration_weight_final_multiplier": 0.001,
    "radius_l2_weight": 1.0e-4,
    "density_l2_weight": 0.0,
}
LOGGING_DEFAULTS = {
    "log_every": 10,
    "image_log_every": 50,
    "video_log_every": 100,
    "always_log_last_step": True,
    "output_dir": "outputs/powerfoam_direct/local_mac_powerfoam_direct_128_smoke",
    "wandb_enabled": False,
    "wandb_project": "dynaworld",
    "wandb_run_name": "powerfoam-direct-128-smoke",
    "wandb_tags": ["powerfoam", "direct-fit", "128px"],
    "wandb_mode": None,
}


def resolve_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = resolved_config(config, ("data", "model", "render", "train", "losses", "logging"))
    cfg.setdefault("camera", {})
    apply_defaults(cfg["data"], DATA_DEFAULTS)
    apply_defaults(cfg["model"], MODEL_DEFAULTS)
    apply_defaults(cfg["render"], RENDER_DEFAULTS)
    apply_defaults(cfg["train"], TRAIN_DEFAULTS)
    apply_defaults(cfg["losses"], LOSS_DEFAULTS)
    apply_defaults(cfg["logging"], LOGGING_DEFAULTS)
    if cfg["data"]["video_path"] is not None:
        cfg["data"]["video_path"] = Path(cfg["data"]["video_path"])
    cfg["logging"]["output_dir"] = Path(cfg["logging"]["output_dir"])

    if int(cfg["model"]["cells"]) < 1:
        raise ValueError("model.cells must be positive")
    if int(cfg["model"]["num_texel_sites"]) < 1:
        raise ValueError("model.num_texel_sites must be positive")
    if int(cfg["model"]["sv_dof"]) < 1:
        raise ValueError("model.sv_dof must be positive")
    if int(cfg["model"]["neighbor_count"]) >= int(cfg["model"]["cells"]):
        cfg["model"]["neighbor_count"] = int(cfg["model"]["cells"]) - 1
    if str(cfg["model"]["adjacency_mode"]) not in {"overlap", "knn", "cech_aabb"}:
        raise ValueError("model.adjacency_mode must be 'overlap', 'knn', or 'cech_aabb'")
    if int(cfg["train"]["frames_per_step"]) < 1:
        raise ValueError("train.frames_per_step must be positive")
    if int(cfg["train"]["steps"]) < 1:
        raise ValueError("train.steps must be positive")
    background = cfg["render"]["background"]
    if len(background) != 3:
        raise ValueError("render.background must have exactly 3 values")
    return cfg
