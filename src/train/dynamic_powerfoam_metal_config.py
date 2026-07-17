from __future__ import annotations

from pathlib import Path
from typing import Any

from config_utils import apply_defaults, resolved_config
from powerfoam_colorizers import DYNAMIC_POWERFOAM_COLORIZE_DEFAULTS


TOKEN_RBF_FEATURE_MODE = "token_rbf_features"

DATA_DEFAULTS = {
    "video_path": "test_data/test_video_small_128_4fps.mp4",
    "frame_source": "explicit_video",
    "max_frames": 16,
}
MODEL_DEFAULTS = {
    "cells": 1024,
    "neighbor_count": 16,
    "adjacency_mode": "knn",
    "dynamic_mode": "rbf",
    "time_basis_count": 8,
    "time_basis_sigma_scale": 0.75,
    "temporal_init_mode": "fit",
    "dynamic_centers": True,
    "dynamic_radii": True,
    "dynamic_densities": True,
    "dynamic_features": True,
    "dynamic_normals": False,
    "dynamic_texel_sites": False,
    "feature_dim": 3,
    "feature_init_noise": 0.01,
    "feature_rgb_init": "logit",
    "token_dim": 128,
    "token_hidden_dim": 128,
    "token_hidden_layers": 1,
    "token_init_std": 0.02,
    "token_output_init_std": 1.0e-4,
    "token_point_residual_scale": 0.08,
    "token_z_residual_scale": 0.08,
    "token_radius_residual_scale": 0.05,
    "token_density_residual_scale": 0.08,
    "token_feature_residual_scale": 0.25,
    "token_normal_residual_scale": 0.08,
    "token_texel_site_residual_scale": 0.08,
    "token_temporal_residual_scale": 0.2,
    "init_from_video": True,
    "video_init_mode": "fixed_camera",
    "color_init_mode": "image",
    "image_init_depth": 2.0,
    "image_init_jitter": 0.2,
    "static_dynamic_split": False,
    "dynamic_cells": None,
    "dynamic_cell_fraction": 0.125,
    "xy_extent": 1.25,
    "z_min": 1.0,
    "z_max": 3.25,
    "radius_init": 0.18,
    "radius_min": 0.03,
    "radius_scale": 0.72,
    "density_init": 16.0,
    "normal_init_jitter": 0.0,
    "num_texel_sites": 4,
    "texel_site_scale": 0.5,
}
CAMERA_DEFAULTS = {
    "enabled": False,
    "mode": "fixed_pinhole",
    "lens_model": "pinhole",
    "base_fov_degrees": None,
    "base_radius": 3.0,
    "token_dim": 32,
    "hidden_dim": 64,
    "time_basis_count": None,
    "time_basis_sigma_scale": None,
    "token_init_std": 0.02,
    "max_rotation_degrees": 8.0,
    "max_translation": None,
    "max_translation_ratio": 0.25,
    "base_position": None,
    "look_at": [0.0, 0.0, 0.0],
    "up": [0.0, 1.0, 0.0],
    "base_path_mode": "static",
    "path_parameterization": "pose_delta",
    "orbit_yaw_start_degrees": 0.0,
    "orbit_yaw_end_degrees": 0.0,
    "orbit_pitch_degrees": 0.0,
    "drone_integration_horizon": 1.0,
    "drone_damping": 0.98,
    "drone_max_linear_velocity_ratio": 0.35,
    "drone_max_linear_acceleration_ratio": 0.7,
    "drone_max_angular_velocity_degrees": 45.0,
    "drone_max_angular_acceleration_degrees": 90.0,
    "drone_gimbal_max_rotation_degrees": 5.0,
    "drone_body_frame_translation": True,
    "initial_zoom_steps": 0,
    "initial_zoom_translation": 0.0,
    "init_teacher_path": None,
    "init_teacher_steps": 0,
    "init_teacher_lr": 0.01,
    "init_teacher_rotation_weight": 1.0,
    "init_teacher_translation_weight": 1.0,
    "init_teacher_velocity_weight": 0.25,
    "init_teacher_normalize_to_first": True,
    "distortion": None,
}
RENDER_DEFAULTS = {
    "render_size": 64,
    "fov_degrees": 55.0,
    "near_plane": 0.05,
    "alpha_threshold": 0.0,
    "transmittance_threshold": 1.0e-4,
    "max_alpha": 0.99,
    "eps": 1.0e-6,
    "texel_temperature": 10.0,
    "train_background_mode": "none",
    "eval_background_mode": "none",
    "background": [0.0, 0.0, 0.0],
    "random_background_min": 0.0,
    "random_background_max": 1.0,
    "normalize_features_by_alpha": True,
}
TRAIN_DEFAULTS = {
    "steps": 120,
    "frames_per_step": 1,
    "lr": 0.02,
    "token_lr_multiplier": 1.0,
    "decoder_lr_multiplier": 1.0,
    "point_lr_multiplier": 0.1,
    "radius_lr_multiplier": 0.05,
    "density_lr_multiplier": 0.05,
    "feature_lr_multiplier": 1.0,
    "colorize_lr_multiplier": 1.0,
    "normal_lr_multiplier": 0.1,
    "texel_site_lr_multiplier": 0.25,
    "camera_lr_multiplier": 0.25,
    "temporal_lr_multiplier": 0.35,
    "static_only_steps": 0,
    "no_repaint_steps": 0,
    "camera_curriculum_enabled": False,
    "camera_curriculum_schedule": [],
    "seed": 17,
    "device": "mps",
}
LOSS_DEFAULTS = {
    "l1_weight": 1.0,
    "mse_weight": 0.1,
    "radius_l2_weight": 1.0e-4,
    "density_l2_weight": 0.0,
    "temporal_center_accel_weight": 1.0e-3,
    "temporal_radius_accel_weight": 1.0e-4,
    "temporal_density_accel_weight": 1.0e-5,
    "temporal_feature_accel_weight": 1.0e-4,
    "temporal_coeff_l2_weight": 1.0e-5,
    "camera_motion_weight": 0.0,
    "camera_temporal_weight": 0.0,
    "camera_global_weight": 0.0,
    "camera_velocity_weight": 0.0,
    "camera_acceleration_weight": 0.0,
    "camera_gimbal_weight": 0.0,
}
LOGGING_DEFAULTS = {
    "log_every": 10,
    "image_log_every": 25,
    "video_log_every": 50,
    "always_log_last_step": True,
    "output_dir": "outputs/dynamic_powerfoam_metal/local_mac_dynamic_powerfoam_metal_rbf_1024_smoke",
    "wandb_enabled": False,
    "wandb_project": "dynaworld",
    "wandb_run_name": "local-mac-dynamic-powerfoam-metal-rbf-1024-smoke",
    "wandb_tags": ["dynamic-powerfoam", "metal", "rbf", "video", "64px"],
    "wandb_mode": None,
}
COLORIZE_DEFAULTS = DYNAMIC_POWERFOAM_COLORIZE_DEFAULTS


def resolve_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = resolved_config(config, ("data", "model", "render", "train", "losses", "logging"))
    if "colorize" not in cfg:
        cfg["colorize"] = {}
    if "camera" not in cfg:
        cfg["camera"] = {}
    apply_defaults(cfg["data"], DATA_DEFAULTS)
    apply_defaults(cfg["model"], MODEL_DEFAULTS)
    apply_defaults(cfg["camera"], CAMERA_DEFAULTS)
    apply_defaults(cfg["render"], RENDER_DEFAULTS)
    apply_defaults(cfg["train"], TRAIN_DEFAULTS)
    apply_defaults(cfg["losses"], LOSS_DEFAULTS)
    apply_defaults(cfg["logging"], LOGGING_DEFAULTS)
    apply_defaults(cfg["colorize"], COLORIZE_DEFAULTS)
    cfg["data"]["video_path"] = Path(cfg["data"]["video_path"])
    cfg["logging"]["output_dir"] = Path(cfg["logging"]["output_dir"])
    if cfg["camera"]["init_teacher_path"] is not None:
        cfg["camera"]["init_teacher_path"] = Path(cfg["camera"]["init_teacher_path"])
    if int(cfg["model"]["cells"]) < 1:
        raise ValueError("model.cells must be positive")
    if int(cfg["model"]["neighbor_count"]) >= int(cfg["model"]["cells"]):
        cfg["model"]["neighbor_count"] = int(cfg["model"]["cells"]) - 1
    if str(cfg["model"]["adjacency_mode"]) not in {"knn", "overlap"}:
        raise ValueError("model.adjacency_mode must be 'knn' or 'overlap'")
    if str(cfg["model"]["dynamic_mode"]) not in {"per_frame_smooth", "rbf", TOKEN_RBF_FEATURE_MODE}:
        raise ValueError(f"model.dynamic_mode must be 'per_frame_smooth', 'rbf', or '{TOKEN_RBF_FEATURE_MODE}'")
    if str(cfg["model"]["temporal_init_mode"]) not in {"fit", "mean"}:
        raise ValueError("model.temporal_init_mode must be 'fit' or 'mean'")
    if str(cfg["model"]["video_init_mode"]) not in {"fixed_camera", "orbit_camera"}:
        raise ValueError("model.video_init_mode must be 'fixed_camera' or 'orbit_camera'")
    if int(cfg["model"]["time_basis_count"]) < 1:
        raise ValueError("model.time_basis_count must be positive")
    if float(cfg["model"]["time_basis_sigma_scale"]) <= 0.0:
        raise ValueError("model.time_basis_sigma_scale must be positive")
    if int(cfg["model"]["feature_dim"]) < 3:
        raise ValueError("model.feature_dim must be at least 3")
    if str(cfg["model"]["feature_rgb_init"]) not in {"logit", "rgb", "none"}:
        raise ValueError("model.feature_rgb_init must be 'logit', 'rgb', or 'none'")
    if int(cfg["model"]["token_dim"]) < 1:
        raise ValueError("model.token_dim must be positive")
    if int(cfg["model"]["token_hidden_dim"]) < 1:
        raise ValueError("model.token_hidden_dim must be positive")
    if int(cfg["model"]["token_hidden_layers"]) < 0:
        raise ValueError("model.token_hidden_layers must be non-negative")
    if str(cfg["model"]["color_init_mode"]) not in {"image", "random"}:
        raise ValueError("model.color_init_mode must be 'image' or 'random'")
    if int(cfg["model"]["num_texel_sites"]) < 1:
        raise ValueError("model.num_texel_sites must be positive")
    if float(cfg["model"]["texel_site_scale"]) <= 0.0:
        raise ValueError("model.texel_site_scale must be positive")
    if cfg["model"]["dynamic_cells"] is not None and int(cfg["model"]["dynamic_cells"]) < 1:
        raise ValueError("model.dynamic_cells must be positive when set")
    if not (0.0 < float(cfg["model"]["dynamic_cell_fraction"]) <= 1.0):
        raise ValueError("model.dynamic_cell_fraction must be in (0, 1]")
    cfg["camera"]["mode"] = str(cfg["camera"]["mode"]).lower()
    cfg["camera"]["enabled"] = bool(cfg["camera"]["enabled"]) or cfg["camera"]["mode"] in {
        "learned_implicit",
        "learned_pose",
        "implicit_camera",
    }
    if cfg["camera"]["enabled"] and cfg["camera"]["mode"] not in {
        "learned_implicit",
        "learned_pose",
        "implicit_camera",
    }:
        raise ValueError("camera.mode must be 'fixed_pinhole' or a learned implicit-camera mode")
    if not cfg["camera"]["enabled"] and cfg["camera"]["mode"] != "fixed_pinhole":
        raise ValueError("camera.enabled=false requires camera.mode='fixed_pinhole'")
    if str(cfg["camera"]["lens_model"]) not in {"pinhole", "radial_tangential", "opencv_fisheye"}:
        raise ValueError("camera.lens_model must be pinhole, radial_tangential, or opencv_fisheye")
    if float(cfg["camera"]["base_radius"]) <= 0.0:
        raise ValueError("camera.base_radius must be positive")
    if int(cfg["camera"]["token_dim"]) < 1:
        raise ValueError("camera.token_dim must be positive")
    if int(cfg["camera"]["hidden_dim"]) < 1:
        raise ValueError("camera.hidden_dim must be positive")
    if cfg["camera"]["max_translation"] is not None and float(cfg["camera"]["max_translation"]) < 0.0:
        raise ValueError("camera.max_translation must be non-negative")
    if float(cfg["camera"]["max_translation_ratio"]) < 0.0:
        raise ValueError("camera.max_translation_ratio must be non-negative")
    if str(cfg["camera"]["base_path_mode"]) not in {"static", "orbit_yaw"}:
        raise ValueError("camera.base_path_mode must be 'static' or 'orbit_yaw'")
    if str(cfg["camera"]["path_parameterization"]) not in {"pose_delta", "integrated_drone"}:
        raise ValueError("camera.path_parameterization must be 'pose_delta' or 'integrated_drone'")
    if float(cfg["camera"]["drone_integration_horizon"]) <= 0.0:
        raise ValueError("camera.drone_integration_horizon must be positive")
    if not (0.0 <= float(cfg["camera"]["drone_damping"]) <= 1.0):
        raise ValueError("camera.drone_damping must be in [0, 1]")
    for key in (
        "drone_max_linear_velocity_ratio",
        "drone_max_linear_acceleration_ratio",
        "drone_max_angular_velocity_degrees",
        "drone_max_angular_acceleration_degrees",
        "drone_gimbal_max_rotation_degrees",
        "initial_zoom_steps",
    ):
        if float(cfg["camera"][key]) < 0.0:
            raise ValueError(f"camera.{key} must be non-negative")
    if int(cfg["camera"]["init_teacher_steps"]) < 0:
        raise ValueError("camera.init_teacher_steps must be non-negative")
    if int(cfg["camera"]["init_teacher_steps"]) > 0 and cfg["camera"]["init_teacher_path"] is None:
        raise ValueError("camera.init_teacher_steps > 0 requires camera.init_teacher_path")
    if float(cfg["camera"]["init_teacher_lr"]) <= 0.0:
        raise ValueError("camera.init_teacher_lr must be positive")
    if str(cfg["model"]["video_init_mode"]) == "orbit_camera" and str(cfg["camera"]["base_path_mode"]) != "orbit_yaw":
        raise ValueError("model.video_init_mode='orbit_camera' requires camera.base_path_mode='orbit_yaw'")
    if str(cfg["model"]["video_init_mode"]) == "orbit_camera" and not bool(cfg["camera"]["enabled"]):
        raise ValueError("model.video_init_mode='orbit_camera' requires camera.enabled=true")
    background_modes = {"none", "black", "white", "fixed_rgb", "random_rgb"}
    cfg["render"]["train_background_mode"] = str(cfg["render"]["train_background_mode"]).lower()
    cfg["render"]["eval_background_mode"] = str(cfg["render"]["eval_background_mode"]).lower()
    if cfg["render"]["train_background_mode"] not in background_modes:
        raise ValueError(f"render.train_background_mode must be one of {sorted(background_modes)}")
    if cfg["render"]["eval_background_mode"] not in background_modes:
        raise ValueError(f"render.eval_background_mode must be one of {sorted(background_modes)}")
    if cfg["render"]["eval_background_mode"] == "random_rgb":
        raise ValueError("render.eval_background_mode='random_rgb' is intentionally unsupported for comparable eval")
    if len(cfg["render"]["background"]) != 3:
        raise ValueError("render.background must contain exactly 3 RGB values")
    if float(cfg["render"]["random_background_min"]) > float(cfg["render"]["random_background_max"]):
        raise ValueError("render.random_background_min must be <= render.random_background_max")
    if int(cfg["train"]["frames_per_step"]) < 1:
        raise ValueError("train.frames_per_step must be positive")
    if int(cfg["train"]["steps"]) < 1:
        raise ValueError("train.steps must be positive")
    if int(cfg["train"]["static_only_steps"]) < 0:
        raise ValueError("train.static_only_steps must be non-negative")
    if int(cfg["train"]["no_repaint_steps"]) < 0:
        raise ValueError("train.no_repaint_steps must be non-negative")
    if bool(cfg["train"]["camera_curriculum_enabled"]):
        schedule = cfg["train"]["camera_curriculum_schedule"]
        if not isinstance(schedule, list) or len(schedule) == 0:
            raise ValueError("train.camera_curriculum_schedule must be a non-empty list when enabled")
        normalized_schedule: list[list[int]] = []
        previous_step = -1
        for entry in schedule:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                raise ValueError("train.camera_curriculum_schedule entries must be [step, active_frames]")
            start_step = int(entry[0])
            active_frames = int(entry[1])
            if start_step < 0:
                raise ValueError("train.camera_curriculum_schedule steps must be non-negative")
            if active_frames < 1:
                raise ValueError("train.camera_curriculum_schedule active frame counts must be positive")
            if start_step <= previous_step:
                raise ValueError("train.camera_curriculum_schedule steps must be strictly increasing")
            normalized_schedule.append([start_step, active_frames])
            previous_step = start_step
        if normalized_schedule[0][0] != 0:
            raise ValueError("train.camera_curriculum_schedule must start at step 0")
        cfg["train"]["camera_curriculum_schedule"] = normalized_schedule
    if str(cfg["colorize"]["view_condition"]) != "none":
        raise ValueError("dynamic_powerfoam_metal colorizer currently supports colorize.view_condition='none' only")
    return cfg


__all__ = [
    "CAMERA_DEFAULTS",
    "COLORIZE_DEFAULTS",
    "DATA_DEFAULTS",
    "LOGGING_DEFAULTS",
    "LOSS_DEFAULTS",
    "MODEL_DEFAULTS",
    "RENDER_DEFAULTS",
    "TOKEN_RBF_FEATURE_MODE",
    "TRAIN_DEFAULTS",
    "resolve_config",
]
