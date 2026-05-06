from __future__ import annotations

import json
import math
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import wandb
from torch import nn
from torch.nn import functional as F
from tqdm import trange

from camera import CameraSpec, build_camera_rays
from checkpoint_utils import atomic_torch_save
from config_utils import apply_defaults, load_config_file, resolved_config, serialize_config_value
from losses import ssim_per_image
from multicam_video_data import cameras_from_K_w2c, heldout_cameras_from_K_w2c, load_multicam_video_bundle
from powerfoam_direct import (
    POWERFOAM_SOFTPLUS_BETA,
    camera_facing_quaternion,
    estimate_knn_radii,
    initialize_full_powerfoam_from_video,
    initialize_powerfoam_from_video,
    inverse_softplus,
    logit_clamped,
)
from renderers.projection import project_points_camera
from sequence_data import load_video_sequence
from train_logging import build_validation_video_payload, make_preview_image, make_wandb_video
from video_io import save_mp4, save_png

ROOT = Path(__file__).resolve().parents[2]
POWERFOAM_METAL_ROOT = ROOT / "third_party" / "powerfoam-metal"
if str(POWERFOAM_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(POWERFOAM_METAL_ROOT))

from torch_powerfoam_metal import (  # noqa: E402
    FoamRasterConfig,
    make_regular_triangulation_adjacency,
    quaternion_frames,
    raytrace_power_foam_oriented_height_sv_texel_surface,
    rasterize_power_foam,
    rasterize_power_foam_linear,
    rasterize_power_foam_oriented_height_sv_texel_surface,
    rasterize_power_foam_oriented_height_sv_texel_surface_aux,
    rasterize_power_foam_oriented_height_texel_surface,
    rasterize_power_foam_oriented_surface_linear,
    rasterize_power_foam_oriented_texel_surface,
    rasterize_power_foam_quaternion_height_sv_texel_surface,
    rasterize_power_foam_quaternion_height_sv_texel_surface_aux,
    rasterize_power_foam_quaternion_height_texel_surface,
    rasterize_power_foam_quaternion_texel_surface,
    rasterize_power_foam_surface_linear,
)


DATA_DEFAULTS = {
    "video_path": "test_data/test_video_small_128_4fps.mp4",
    "frame_source": "explicit_video",
    "max_frames": 16,
}
MODEL_DEFAULTS = {
    "cells": 64,
    "neighbor_count": 16,
    "adjacency_mode": "cech_aabb",
    "init_from_video": True,
    "color_init_mode": "image",
    "image_init_depth": 2.0,
    "image_init_jitter": 0.2,
    "xy_extent": 1.25,
    "z_min": 1.0,
    "z_max": 3.25,
    "radius_init": 0.18,
    "radius_min": 0.03,
    "radius_scale": 0.72,
    "density_init": 16.0,
    "feature_mode": "constant",
    "linear_coeff_init": 0.0,
    "linear_coeff_scale": 0.25,
    "normal_init_jitter": 0.0,
    "num_texel_sites": 4,
    "texel_site_scale": 0.5,
    "texel_height_scale": 0.25,
    "sv_dof": 3,
    "sv_axis_init": 1.0,
    "sv_axis_init_jitter": 0.02,
    "sv_rgb_init_jitter": 0.02,
    "resample_every": 0,
    "resample_target_cells": None,
    "resample_final_cells": None,
    "resample_from_step": 1,
    "resample_until_step": None,
    "resample_perturb_scale": 0.05,
    "init_point_cloud_path": None,
    "init_point_cloud_normalize": "none",
    "init_point_cloud_coordinate_frame": "model",
    "init_point_cloud_visibility_filter": "none",
    "init_point_cloud_min_visible_train_views": 1,
    "init_point_cloud_sample_mode": "random",
    "init_point_cloud_duplicate_jitter": 0.0,
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
    "use_tiled": False,
    "use_raytrace": False,
    "tiled_builder": "auto",
    "background": [0.0, 0.0, 0.0],
    "background_mode": "fixed",
    "eval_color_calibration": "none",
}
TRAIN_DEFAULTS = {
    "steps": 50,
    "frames_per_step": 1,
    "lr": 0.02,
    "lr_schedule": "constant",
    "point_lr_multiplier": 0.1,
    "radius_lr_multiplier": 0.05,
    "density_lr_multiplier": 0.05,
    "feature_lr_multiplier": 1.0,
    "normal_lr_multiplier": 0.1,
    "quaternion_lr_multiplier": 0.1,
    "texel_site_lr_multiplier": 0.25,
    "texel_height_lr_multiplier": 0.25,
    "texel_sv_axis_lr_multiplier": 0.1,
    "texel_sv_rgb_lr_multiplier": 1.0,
    "points_lr_init": None,
    "points_lr_final": None,
    "density_lr_init": None,
    "density_lr_final": None,
    "radii_lr_init": None,
    "radii_lr_final": None,
    "quaternions_lr_init": None,
    "quaternions_lr_final": None,
    "texel_sites_lr_init": None,
    "texel_sites_lr_final": None,
    "texel_sv_axis_lr_init": None,
    "texel_sv_axis_lr_final": None,
    "texel_sv_rgb_lr_init": None,
    "texel_sv_rgb_lr_final": None,
    "texel_height_lr_init": None,
    "texel_height_lr_final": None,
    "lr_warmup_steps": {},
    "seed": 17,
    "device": "mps",
}
LOSS_DEFAULTS = {
    "l1_weight": 1.0,
    "mse_weight": 0.1,
    "ssim_weight": 0.0,
    "radius_l2_weight": 1.0e-4,
    "density_l2_weight": 0.0,
    "normal_weight": 0.0,
    "normal_weight_start_step": 0,
    "normal_weight_final_multiplier": 0.1,
    "normal_map_weight": 0.0,
    "normal_map_weight_start_step": 0,
    "normal_map_weight_final_multiplier": 1.0,
    "normal_map_min_alpha": 0.05,
    "normal_map_teacher": "aux_median_depth",
    "contribution_weight": 0.0,
    "contribution_weight_start_step": 0,
    "contribution_weight_final_multiplier": 0.001,
    "interpenetration_weight": 0.0,
    "interpenetration_weight_start_step": 0,
    "interpenetration_weight_final_multiplier": 0.001,
    "ssim_window_size": 11,
    "ssim_c1": 0.0001,
    "ssim_c2": 0.0009,
}
LOGGING_DEFAULTS = {
    "log_every": 10,
    "image_log_every": 25,
    "video_log_every": 50,
    "always_log_last_step": True,
    "output_dir": "outputs/powerfoam_metal/local_mac_powerfoam_metal_video_64_smoke",
    "wandb_enabled": False,
    "wandb_project": "dynaworld",
    "wandb_run_name": "local-mac-powerfoam-metal-video-64-smoke",
    "wandb_tags": ["powerfoam", "metal", "direct-fit", "video", "64px"],
    "wandb_mode": None,
}
TEXEL_SURFACE_MODES = {
    "oriented_texel_surface",
    "quaternion_texel_surface",
    "oriented_height_texel_surface",
    "quaternion_height_texel_surface",
    "oriented_height_sv_texel_surface",
    "quaternion_height_sv_texel_surface",
}
HEIGHT_TEXEL_SURFACE_MODES = {
    "oriented_height_texel_surface",
    "quaternion_height_texel_surface",
    "oriented_height_sv_texel_surface",
    "quaternion_height_sv_texel_surface",
}
QUATERNION_TEXEL_SURFACE_MODES = {
    "quaternion_texel_surface",
    "quaternion_height_texel_surface",
    "quaternion_height_sv_texel_surface",
}
ORIENTED_TEXEL_SURFACE_MODES = {
    "oriented_texel_surface",
    "oriented_height_texel_surface",
    "oriented_height_sv_texel_surface",
}
SV_TEXEL_SURFACE_MODES = {
    "oriented_height_sv_texel_surface",
    "quaternion_height_sv_texel_surface",
}
LR_GROUP_SPECS: dict[str, tuple[str, str | None, int]] = {
    "points": ("point_lr_multiplier", "points", 0),
    "radii": ("radius_lr_multiplier", "radii", 1000),
    "density": ("density_lr_multiplier", "density", 1000),
    "features": ("feature_lr_multiplier", None, 0),
    "normals": ("normal_lr_multiplier", None, 0),
    "tangents": ("normal_lr_multiplier", None, 0),
    "quaternions": ("quaternion_lr_multiplier", "quaternions", 0),
    "texel_sites": ("texel_site_lr_multiplier", "texel_sites", 0),
    "texel_height": ("texel_height_lr_multiplier", "texel_height", 2000),
    "texel_sv_axis": ("texel_sv_axis_lr_multiplier", "texel_sv_axis", 0),
    "texel_sv_rgb": ("texel_sv_rgb_lr_multiplier", "texel_sv_rgb", 0),
}


def cosine_scheduled_lr(
    initial: float,
    final: float,
    step: int,
    total_steps: int,
    *,
    warmup_steps: int = 0,
) -> float:
    warmup = max(int(warmup_steps), 0)
    step_f = float(step)
    max_steps = max(int(total_steps), 1)
    if warmup and step < warmup:
        return float(initial) * step_f / float(warmup)
    if step > max_steps:
        return float(final)
    denom = max(float(max_steps - warmup), 1.0)
    progress = (step_f - float(warmup)) / denom
    lr_cos = float(final) + 0.5 * (float(initial) - float(final)) * (1.0 + math.cos(math.pi * progress))
    return float(lr_cos)


def powerfoam_group_initial_lr(train_cfg: dict[str, Any], group_name: str) -> float:
    multiplier_key, official_key, _warmup = LR_GROUP_SPECS[group_name]
    if official_key is not None:
        explicit = train_cfg[f"{official_key}_lr_init"]
        if explicit is not None:
            return float(explicit)
    return float(train_cfg["lr"]) * float(train_cfg[multiplier_key])


def powerfoam_group_final_lr(train_cfg: dict[str, Any], group_name: str, initial_lr: float) -> float:
    _multiplier_key, official_key, _warmup = LR_GROUP_SPECS[group_name]
    if official_key is None:
        return float(initial_lr)
    explicit = train_cfg[f"{official_key}_lr_final"]
    if explicit is None:
        return float(initial_lr)
    return float(explicit)


def powerfoam_group_lr_metadata(train_cfg: dict[str, Any], group_name: str) -> dict[str, float | int]:
    initial_lr = powerfoam_group_initial_lr(train_cfg, group_name)
    return {
        "lr": initial_lr,
        "initial_lr": initial_lr,
        "final_lr": powerfoam_group_final_lr(train_cfg, group_name, initial_lr),
        "warmup_steps": powerfoam_group_warmup_steps(train_cfg, group_name),
    }


def powerfoam_group_warmup_steps(train_cfg: dict[str, Any], group_name: str) -> int:
    _multiplier_key, _official_key, default_warmup_steps = LR_GROUP_SPECS[group_name]
    overrides = train_cfg["lr_warmup_steps"]
    if group_name in overrides:
        return int(overrides[group_name])
    return int(default_warmup_steps)


def update_powerfoam_learning_rates(
    optimizer: torch.optim.Optimizer,
    train_cfg: dict[str, Any],
    *,
    step: int,
    total_steps: int,
) -> dict[str, float]:
    if str(train_cfg["lr_schedule"]) == "cosine":
        for group in optimizer.param_groups:
            if "initial_lr" not in group or "final_lr" not in group:
                continue
            group["lr"] = cosine_scheduled_lr(
                float(group["initial_lr"]),
                float(group["final_lr"]),
                int(step),
                int(total_steps),
                warmup_steps=int(group.get("warmup_steps", 0)),
            )
    return {str(group.get("name", index)): float(group["lr"]) for index, group in enumerate(optimizer.param_groups)}


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
    if int(cfg["model"]["neighbor_count"]) >= int(cfg["model"]["cells"]):
        cfg["model"]["neighbor_count"] = int(cfg["model"]["cells"]) - 1
    if str(cfg["model"]["adjacency_mode"]) not in {"knn", "overlap", "cech_aabb", "regular_triangulation"}:
        raise ValueError("model.adjacency_mode must be 'knn', 'overlap', 'cech_aabb', or 'regular_triangulation'")
    if str(cfg["model"]["color_init_mode"]) not in {"image", "random"}:
        raise ValueError("model.color_init_mode must be 'image' or 'random'")
    if str(cfg["model"]["feature_mode"]) not in {
        "constant",
        "linear",
        "surface_linear",
        "oriented_surface_linear",
        "oriented_texel_surface",
        "quaternion_texel_surface",
        "oriented_height_texel_surface",
        "quaternion_height_texel_surface",
        "oriented_height_sv_texel_surface",
        "quaternion_height_sv_texel_surface",
    }:
        raise ValueError(
            "model.feature_mode must be 'constant', 'linear', 'surface_linear', "
            "'oriented_surface_linear', 'oriented_texel_surface', "
            "'quaternion_texel_surface', 'oriented_height_texel_surface', "
            "'quaternion_height_texel_surface', 'oriented_height_sv_texel_surface', or "
            "'quaternion_height_sv_texel_surface'"
        )
    if bool(cfg["render"]["use_raytrace"]) and str(cfg["model"]["feature_mode"]) not in {
        "oriented_height_sv_texel_surface",
        "quaternion_height_sv_texel_surface",
    }:
        raise ValueError("render.use_raytrace currently requires a height+SV PowerFoam feature mode")
    if int(cfg["model"]["num_texel_sites"]) < 1:
        raise ValueError("model.num_texel_sites must be positive")
    if float(cfg["model"]["texel_site_scale"]) <= 0.0:
        raise ValueError("model.texel_site_scale must be positive")
    if float(cfg["model"]["texel_height_scale"]) <= 0.0:
        raise ValueError("model.texel_height_scale must be positive")
    if int(cfg["model"]["sv_dof"]) < 1:
        raise ValueError("model.sv_dof must be positive")
    if float(cfg["model"]["sv_axis_init"]) <= 0.0:
        raise ValueError("model.sv_axis_init must be positive")
    if float(cfg["model"]["sv_axis_init_jitter"]) < 0.0:
        raise ValueError("model.sv_axis_init_jitter must be non-negative")
    if float(cfg["model"]["sv_rgb_init_jitter"]) < 0.0:
        raise ValueError("model.sv_rgb_init_jitter must be non-negative")
    if int(cfg["model"]["resample_every"]) < 0:
        raise ValueError("model.resample_every must be non-negative")
    if cfg["model"]["resample_target_cells"] is not None and int(cfg["model"]["resample_target_cells"]) < 1:
        raise ValueError("model.resample_target_cells must be positive or null")
    if cfg["model"]["resample_final_cells"] is not None and int(cfg["model"]["resample_final_cells"]) < 1:
        raise ValueError("model.resample_final_cells must be positive or null")
    if int(cfg["model"]["resample_from_step"]) < 1:
        raise ValueError("model.resample_from_step must be positive")
    if cfg["model"]["resample_until_step"] is not None and int(cfg["model"]["resample_until_step"]) <= int(
        cfg["model"]["resample_from_step"]
    ):
        raise ValueError("model.resample_until_step must be greater than model.resample_from_step")
    if float(cfg["model"]["resample_perturb_scale"]) < 0.0:
        raise ValueError("model.resample_perturb_scale must be non-negative")
    if cfg["model"]["init_point_cloud_path"] is not None:
        cfg["model"]["init_point_cloud_path"] = Path(cfg["model"]["init_point_cloud_path"])
    if str(cfg["model"]["init_point_cloud_normalize"]) not in {"none", "fit_box"}:
        raise ValueError("model.init_point_cloud_normalize must be 'none' or 'fit_box'")
    if str(cfg["model"]["init_point_cloud_coordinate_frame"]) not in {"model", "multicam_world"}:
        raise ValueError("model.init_point_cloud_coordinate_frame must be 'model' or 'multicam_world'")
    if str(cfg["model"]["init_point_cloud_visibility_filter"]) not in {"none", "train_visible"}:
        raise ValueError("model.init_point_cloud_visibility_filter must be 'none' or 'train_visible'")
    if int(cfg["model"]["init_point_cloud_min_visible_train_views"]) < 1:
        raise ValueError("model.init_point_cloud_min_visible_train_views must be positive")
    if str(cfg["model"]["init_point_cloud_sample_mode"]) not in {"random", "first"}:
        raise ValueError("model.init_point_cloud_sample_mode must be 'random' or 'first'")
    if float(cfg["model"]["init_point_cloud_duplicate_jitter"]) < 0.0:
        raise ValueError("model.init_point_cloud_duplicate_jitter must be non-negative")
    if float(cfg["model"]["linear_coeff_scale"]) < 0.0:
        raise ValueError("model.linear_coeff_scale must be non-negative")
    if float(cfg["model"]["normal_init_jitter"]) < 0.0:
        raise ValueError("model.normal_init_jitter must be non-negative")
    if int(cfg["train"]["frames_per_step"]) < 1:
        raise ValueError("train.frames_per_step must be positive")
    if int(cfg["train"]["steps"]) < 1:
        raise ValueError("train.steps must be positive")
    if str(cfg["train"]["lr_schedule"]) not in {"constant", "cosine"}:
        raise ValueError("train.lr_schedule must be 'constant' or 'cosine'")
    if not isinstance(cfg["train"]["lr_warmup_steps"], dict):
        raise ValueError("train.lr_warmup_steps must be an object mapping parameter-group names to step counts")
    cfg["train"]["lr_warmup_steps"] = {
        str(key): int(value) for key, value in cfg["train"]["lr_warmup_steps"].items()
    }
    for key, value in cfg["train"]["lr_warmup_steps"].items():
        if key not in LR_GROUP_SPECS:
            raise ValueError(f"train.lr_warmup_steps has unknown parameter group {key!r}")
        if int(value) < 0:
            raise ValueError(f"train.lr_warmup_steps.{key} must be non-negative")
    if float(cfg["losses"]["ssim_weight"]) < 0.0:
        raise ValueError("losses.ssim_weight must be non-negative")
    if float(cfg["losses"]["normal_weight"]) < 0.0:
        raise ValueError("losses.normal_weight must be non-negative")
    if int(cfg["losses"]["normal_weight_start_step"]) < 0:
        raise ValueError("losses.normal_weight_start_step must be non-negative")
    if float(cfg["losses"]["normal_weight_final_multiplier"]) < 0.0:
        raise ValueError("losses.normal_weight_final_multiplier must be non-negative")
    if float(cfg["losses"]["normal_map_weight"]) < 0.0:
        raise ValueError("losses.normal_map_weight must be non-negative")
    if int(cfg["losses"]["normal_map_weight_start_step"]) < 0:
        raise ValueError("losses.normal_map_weight_start_step must be non-negative")
    if float(cfg["losses"]["normal_map_weight_final_multiplier"]) < 0.0:
        raise ValueError("losses.normal_map_weight_final_multiplier must be non-negative")
    if float(cfg["losses"]["normal_map_min_alpha"]) < 0.0:
        raise ValueError("losses.normal_map_min_alpha must be non-negative")
    if str(cfg["losses"]["normal_map_teacher"]) not in {"aux_median_depth"}:
        raise ValueError("losses.normal_map_teacher must be 'aux_median_depth'")
    if float(cfg["losses"]["normal_map_weight"]) > 0.0:
        if not bool(cfg["render"]["use_raytrace"]):
            raise ValueError("losses.normal_map_weight requires render.use_raytrace=true")
        if str(cfg["model"]["feature_mode"]) not in {"oriented_height_sv_texel_surface", "quaternion_height_sv_texel_surface"}:
            raise ValueError("losses.normal_map_weight requires a height+SV PowerFoam feature mode")
    if float(cfg["losses"]["contribution_weight"]) < 0.0:
        raise ValueError("losses.contribution_weight must be non-negative")
    if int(cfg["losses"]["contribution_weight_start_step"]) < 0:
        raise ValueError("losses.contribution_weight_start_step must be non-negative")
    if float(cfg["losses"]["contribution_weight_final_multiplier"]) < 0.0:
        raise ValueError("losses.contribution_weight_final_multiplier must be non-negative")
    if float(cfg["losses"]["interpenetration_weight"]) < 0.0:
        raise ValueError("losses.interpenetration_weight must be non-negative")
    if int(cfg["losses"]["interpenetration_weight_start_step"]) < 0:
        raise ValueError("losses.interpenetration_weight_start_step must be non-negative")
    if float(cfg["losses"]["interpenetration_weight_final_multiplier"]) < 0.0:
        raise ValueError("losses.interpenetration_weight_final_multiplier must be non-negative")
    if str(cfg["render"]["background_mode"]) not in {"fixed", "random"}:
        raise ValueError("render.background_mode must be 'fixed' or 'random'")
    if str(cfg["render"]["eval_color_calibration"]) not in {
        "none",
        "train_fit_channel_affine",
        "train_fit_rgb_matrix_affine",
    }:
        raise ValueError(
            "render.eval_color_calibration must be one of: "
            "'none', 'train_fit_channel_affine', 'train_fit_rgb_matrix_affine'"
        )
    background = cfg["render"]["background"]
    if len(background) != 3:
        raise ValueError("render.background must contain exactly three RGB values")
    cfg["render"]["background"] = [float(value) for value in background]
    ssim_window_size = int(cfg["losses"]["ssim_window_size"])
    if ssim_window_size < 1 or ssim_window_size % 2 == 0:
        raise ValueError(f"losses.ssim_window_size must be a positive odd integer, got {ssim_window_size}.")
    cfg["losses"]["ssim_window_size"] = ssim_window_size
    for key in (
        "point_lr_multiplier",
        "radius_lr_multiplier",
        "density_lr_multiplier",
        "feature_lr_multiplier",
        "normal_lr_multiplier",
        "quaternion_lr_multiplier",
        "texel_site_lr_multiplier",
        "texel_height_lr_multiplier",
        "texel_sv_axis_lr_multiplier",
        "texel_sv_rgb_lr_multiplier",
    ):
        if float(cfg["train"][key]) < 0.0:
            raise ValueError(f"train.{key} must be non-negative")
    for _group_name, (_multiplier_key, official_key, _warmup_steps) in LR_GROUP_SPECS.items():
        if official_key is None:
            continue
        for suffix in ("init", "final"):
            key = f"{official_key}_lr_{suffix}"
            if cfg["train"][key] is not None and float(cfg["train"][key]) < 0.0:
                raise ValueError(f"train.{key} must be non-negative")
    return cfg


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(value)


def make_pinhole_rays(height: int, width: int, fov_degrees: float, device: torch.device) -> torch.Tensor:
    half_y = math.tan(math.radians(float(fov_degrees)) * 0.5)
    half_x = half_y * (float(width) / float(height))
    ys = torch.linspace(half_y, -half_y, height, device=device, dtype=torch.float32)
    xs = torch.linspace(-half_x, half_x, width, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    dirs = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
    dirs = F.normalize(dirs, dim=-1)
    origins = torch.zeros_like(dirs)
    return torch.cat([origins, dirs], dim=-1).unsqueeze(0).contiguous()


@dataclass(frozen=True)
class PowerFoamAuxBatch:
    contrib: torch.Tensor
    point_error: torch.Tensor
    visible_mask: torch.Tensor
    normal_distance: torch.Tensor
    normal_norm: torch.Tensor
    median_depth: torch.Tensor


@dataclass(frozen=True)
class PointCloudInitialization:
    points: torch.Tensor
    colors: torch.Tensor
    source_path: Path
    source_count: int
    sampled_count: int
    normalize_mode: str
    coordinate_frame: str
    visibility_filter: str
    sample_mode: str
    filtered_count: int


PLY_SCALAR_FORMATS = {
    "char": "b",
    "int8": "b",
    "uchar": "B",
    "uint8": "B",
    "short": "h",
    "int16": "h",
    "ushort": "H",
    "uint16": "H",
    "int": "i",
    "int32": "i",
    "uint": "I",
    "uint32": "I",
    "float": "f",
    "float32": "f",
    "double": "d",
    "float64": "d",
}


def resolve_point_cloud_path(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = (
        path / "input.ply",
        path / "point_cloud.ply",
        path / "points3D.txt",
        path / "points3D.bin",
        path / "sparse" / "0" / "points3D.txt",
        path / "sparse" / "0" / "points3D.bin",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No supported point cloud file found under {path}.")


def normalize_point_cloud_colors(colors: torch.Tensor | None, count: int) -> torch.Tensor:
    if colors is None:
        return torch.full((count, 3), 0.5, dtype=torch.float32)
    colors = colors.to(dtype=torch.float32)
    if colors.numel() > 0 and float(colors.max()) > 1.0:
        colors = colors / 255.0
    return colors.clamp(0.0, 1.0)


def load_ply_point_cloud(path: Path) -> tuple[torch.Tensor, torch.Tensor | None]:
    with path.open("rb") as fh:
        first = fh.readline().decode("ascii", errors="strict").strip()
        if first != "ply":
            raise ValueError(f"{path} is not a PLY file.")
        fmt = None
        vertex_count = None
        in_vertex = False
        vertex_properties: list[tuple[str, str]] = []
        while True:
            raw = fh.readline()
            if not raw:
                raise ValueError(f"{path} ended before PLY end_header.")
            line = raw.decode("ascii", errors="strict").strip()
            if line == "end_header":
                break
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and in_vertex:
                if parts[1] == "list":
                    raise ValueError(f"{path} has list properties on vertices; unsupported for PowerFoam init.")
                vertex_properties.append((parts[2], parts[1]))
        if fmt not in {"ascii", "binary_little_endian"}:
            raise ValueError(f"{path} PLY format {fmt!r} is unsupported.")
        if vertex_count is None:
            raise ValueError(f"{path} does not declare a vertex element.")
        prop_names = [name for name, _kind in vertex_properties]
        for required in ("x", "y", "z"):
            if required not in prop_names:
                raise ValueError(f"{path} vertex properties must include x/y/z.")
        xyz_rows = []
        rgb_rows = []
        has_rgb = all(name in prop_names for name in ("red", "green", "blue"))
        if fmt == "ascii":
            for _ in range(vertex_count):
                values = fh.readline().decode("ascii", errors="strict").split()
                if len(values) < len(vertex_properties):
                    raise ValueError(f"{path} has a short ASCII vertex row.")
                row = {name: float(values[index]) for index, (name, _kind) in enumerate(vertex_properties)}
                xyz_rows.append([row["x"], row["y"], row["z"]])
                if has_rgb:
                    rgb_rows.append([row["red"], row["green"], row["blue"]])
        else:
            try:
                row_struct = struct.Struct("<" + "".join(PLY_SCALAR_FORMATS[kind] for _name, kind in vertex_properties))
            except KeyError as exc:
                raise ValueError(f"{path} has unsupported PLY scalar type {exc.args[0]!r}.") from exc
            name_to_index = {name: index for index, (name, _kind) in enumerate(vertex_properties)}
            for _ in range(vertex_count):
                payload = fh.read(row_struct.size)
                if len(payload) != row_struct.size:
                    raise ValueError(f"{path} ended inside a binary vertex row.")
                values = row_struct.unpack(payload)
                xyz_rows.append([values[name_to_index["x"]], values[name_to_index["y"]], values[name_to_index["z"]]])
                if has_rgb:
                    rgb_rows.append(
                        [
                            values[name_to_index["red"]],
                            values[name_to_index["green"]],
                            values[name_to_index["blue"]],
                        ]
                    )
    points = torch.tensor(xyz_rows, dtype=torch.float32)
    colors = torch.tensor(rgb_rows, dtype=torch.float32) if has_rgb else None
    return points, colors


def load_colmap_points3d_txt(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    xyz_rows = []
    rgb_rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        xyz_rows.append([float(parts[1]), float(parts[2]), float(parts[3])])
        rgb_rows.append([float(parts[4]), float(parts[5]), float(parts[6])])
    if not xyz_rows:
        raise ValueError(f"{path} contains no COLMAP points3D rows.")
    return torch.tensor(xyz_rows, dtype=torch.float32), torch.tensor(rgb_rows, dtype=torch.float32)


def load_colmap_points3d_bin(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    xyz_rows = []
    rgb_rows = []
    with path.open("rb") as fh:
        count_payload = fh.read(8)
        if len(count_payload) != 8:
            raise ValueError(f"{path} is too short for a COLMAP points3D.bin file.")
        (point_count,) = struct.unpack("<Q", count_payload)
        fixed_struct = struct.Struct("<QdddBBBdQ")
        track_struct = struct.Struct("<ii")
        for _ in range(point_count):
            payload = fh.read(fixed_struct.size)
            if len(payload) != fixed_struct.size:
                raise ValueError(f"{path} ended inside a COLMAP point record.")
            values = fixed_struct.unpack(payload)
            _point_id, x, y, z, red, green, blue, _error, track_len = values
            xyz_rows.append([x, y, z])
            rgb_rows.append([float(red), float(green), float(blue)])
            skip = int(track_len) * track_struct.size
            if len(fh.read(skip)) != skip:
                raise ValueError(f"{path} ended inside a COLMAP point track.")
    if not xyz_rows:
        raise ValueError(f"{path} contains no COLMAP points.")
    return torch.tensor(xyz_rows, dtype=torch.float32), torch.tensor(rgb_rows, dtype=torch.float32)


def load_point_cloud_xyz_rgb(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    resolved = resolve_point_cloud_path(path)
    if resolved.suffix.lower() == ".ply":
        points, colors = load_ply_point_cloud(resolved)
    elif resolved.name == "points3D.txt":
        points, colors = load_colmap_points3d_txt(resolved)
    elif resolved.name == "points3D.bin":
        points, colors = load_colmap_points3d_bin(resolved)
    else:
        raise ValueError(f"Unsupported point cloud format for {resolved}.")
    finite = torch.isfinite(points).all(dim=-1)
    if colors is not None:
        finite = finite & torch.isfinite(colors).all(dim=-1)
    points = points[finite]
    if points.numel() == 0:
        raise ValueError(f"{resolved} has no finite points.")
    colors = normalize_point_cloud_colors(None if colors is None else colors[finite], int(points.shape[0]))
    return points, colors


def fit_point_cloud_to_powerfoam_box(
    points: torch.Tensor,
    *,
    xy_extent: float,
    z_min: float,
    z_max: float,
) -> torch.Tensor:
    center = points.median(dim=0).values
    centered = points - center
    q95 = torch.quantile(centered.abs(), 0.95, dim=0).clamp_min(1.0e-6)
    xy_scale = 0.85 * float(xy_extent) / torch.max(q95[:2])
    z_scale = 0.45 * (float(z_max) - float(z_min)) / q95[2]
    scale = torch.minimum(xy_scale, z_scale)
    out = centered * scale
    out[:, 2] += 0.5 * (float(z_min) + float(z_max))
    return out


def clamp_point_cloud_to_powerfoam_box(
    points: torch.Tensor,
    *,
    xy_extent: float,
    z_min: float,
    z_max: float,
) -> torch.Tensor:
    out = points.clone()
    out[:, :2] = out[:, :2].clamp(-0.999 * float(xy_extent), 0.999 * float(xy_extent))
    out[:, 2] = out[:, 2].clamp(float(z_min) + 1.0e-4, float(z_max) - 1.0e-4)
    return out


def point_cloud_box_mask(
    points: torch.Tensor,
    *,
    xy_extent: float,
    z_min: float,
    z_max: float,
) -> torch.Tensor:
    return (
        torch.isfinite(points).all(dim=-1)
        & (points[:, 0].abs() <= float(xy_extent))
        & (points[:, 1].abs() <= float(xy_extent))
        & (points[:, 2] >= float(z_min))
        & (points[:, 2] <= float(z_max))
    )


def filter_point_cloud_by_train_visibility(
    points: torch.Tensor,
    colors: torch.Tensor,
    *,
    train_K: torch.Tensor,
    train_w2c: torch.Tensor,
    train_lens_models: list[str] | None = None,
    train_distortions: torch.Tensor | None = None,
    render_size: int,
    min_visible_train_views: int,
    xy_extent: float,
    z_min: float,
    z_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if train_K.ndim != 3:
        raise ValueError(f"train_K must have shape [V,3,3], got {tuple(train_K.shape)}.")
    if train_w2c.ndim == 4:
        train_w2c = train_w2c[:, 0]
    if train_w2c.ndim != 3:
        raise ValueError(f"train_w2c must have shape [V,4,4] or [V,T,4,4], got {tuple(train_w2c.shape)}.")
    if int(train_K.shape[0]) != int(train_w2c.shape[0]):
        raise ValueError(f"train_K/train_w2c view count mismatch: {train_K.shape[0]} vs {train_w2c.shape[0]}.")

    train_K = train_K.detach().to(device=points.device, dtype=points.dtype)
    train_w2c = train_w2c.detach().to(device=points.device, dtype=points.dtype)
    train_distortions = None if train_distortions is None else train_distortions.to(device=points.device, dtype=points.dtype)
    points_h = torch.cat([points, torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)], dim=-1)
    visible_votes = torch.zeros(points.shape[0], dtype=torch.int64, device=points.device)
    width = height = int(render_size)
    for view in range(int(train_K.shape[0])):
        points_camera = (points_h @ train_w2c[view].T)[:, :3]
        if train_lens_models is None and train_distortions is None:
            z = points_camera[:, 2]
            u = train_K[view, 0, 0] * points_camera[:, 0] / z.clamp_min(1.0e-6) + train_K[view, 0, 2]
            v = train_K[view, 1, 1] * points_camera[:, 1] / z.clamp_min(1.0e-6) + train_K[view, 1, 2]
            front = z > 1.0e-5
        else:
            camera = CameraSpec(
                fx=train_K[view, 0, 0],
                fy=train_K[view, 1, 1],
                cx=train_K[view, 0, 2],
                cy=train_K[view, 1, 2],
                camera_to_world=torch.linalg.inv(train_w2c[view]),
                lens_model=("pinhole" if train_lens_models is None else train_lens_models[view]),  # type: ignore[arg-type]
                distortion=None if train_distortions is None else train_distortions[view],
            )
            pixels, _depths, _jacobian, front = project_points_camera(points_camera, camera, near_plane=1.0e-5)
            u = pixels[:, 0]
            v = pixels[:, 1]
        inside = front & (u >= 0.0) & (u < float(width)) & (v >= 0.0) & (v < float(height))
        visible_votes += inside.to(dtype=torch.int64)

    keep = point_cloud_box_mask(points, xy_extent=xy_extent, z_min=z_min, z_max=z_max) & (
        visible_votes >= int(min_visible_train_views)
    )
    if int(keep.sum().item()) == 0:
        raise ValueError("Point-cloud visibility filtering removed every point.")
    return points[keep].contiguous(), colors[keep].contiguous()


def load_powerfoam_point_cloud_initialization(
    *,
    path: Path,
    frame_count: int,
    cell_count: int,
    xy_extent: float,
    z_min: float,
    z_max: float,
    normalize_mode: str,
    coordinate_frame: str,
    point_transform: torch.Tensor | None = None,
    visibility_filter: str = "none",
    min_visible_train_views: int = 1,
    visibility_train_K: torch.Tensor | None = None,
    visibility_train_w2c: torch.Tensor | None = None,
    visibility_train_lens_models: list[str] | None = None,
    visibility_train_distortions: torch.Tensor | None = None,
    visibility_render_size: int | None = None,
    sample_mode: str = "random",
    duplicate_jitter: float = 0.0,
    seed: int,
) -> PointCloudInitialization:
    resolved = resolve_point_cloud_path(path)
    points, colors = load_point_cloud_xyz_rgb(resolved)
    source_count = int(points.shape[0])
    if point_transform is not None:
        if tuple(point_transform.shape) != (4, 4):
            raise ValueError(f"point_transform must have shape (4, 4), got {tuple(point_transform.shape)}.")
        transform = point_transform.detach().to(device=points.device, dtype=points.dtype)
        points_h = torch.cat([points, torch.ones((points.shape[0], 1), dtype=points.dtype)], dim=-1)
        points = (points_h @ transform.T)[:, :3].contiguous()
    if str(normalize_mode) == "fit_box":
        points = fit_point_cloud_to_powerfoam_box(points, xy_extent=xy_extent, z_min=z_min, z_max=z_max)
    elif str(normalize_mode) != "none":
        raise ValueError("normalize_mode must be 'none' or 'fit_box'")
    if str(visibility_filter) == "train_visible":
        if visibility_train_K is None or visibility_train_w2c is None or visibility_render_size is None:
            raise ValueError("train_visible point-cloud filtering requires train K/w2c camera metadata and render size.")
        points, colors = filter_point_cloud_by_train_visibility(
            points,
            colors,
            train_K=visibility_train_K,
            train_w2c=visibility_train_w2c,
            train_lens_models=visibility_train_lens_models,
            train_distortions=visibility_train_distortions,
            render_size=int(visibility_render_size),
            min_visible_train_views=int(min_visible_train_views),
            xy_extent=xy_extent,
            z_min=z_min,
            z_max=z_max,
        )
    elif str(visibility_filter) != "none":
        raise ValueError("visibility_filter must be 'none' or 'train_visible'")
    filtered_count = int(points.shape[0])
    points = clamp_point_cloud_to_powerfoam_box(points, xy_extent=xy_extent, z_min=z_min, z_max=z_max)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    if filtered_count >= int(cell_count):
        if str(sample_mode) == "random":
            sample = torch.randperm(filtered_count, generator=generator)[: int(cell_count)]
        elif str(sample_mode) == "first":
            sample = torch.arange(int(cell_count))
        else:
            raise ValueError("sample_mode must be 'random' or 'first'")
        duplicate_count = 0
    else:
        extra = torch.randint(filtered_count, (int(cell_count) - filtered_count,), generator=generator)
        sample = torch.cat([torch.arange(filtered_count), extra], dim=0)
        duplicate_count = int(extra.numel())
    sampled_points = points.index_select(0, sample).contiguous()
    sampled_colors = colors.index_select(0, sample).contiguous()
    if duplicate_count > 0 and float(duplicate_jitter) > 0.0:
        jitter = float(duplicate_jitter) * torch.randn(
            duplicate_count,
            3,
            generator=generator,
            dtype=sampled_points.dtype,
        )
        sampled_points[filtered_count:] = sampled_points[filtered_count:] + jitter.to(sampled_points.device)
        sampled_points = clamp_point_cloud_to_powerfoam_box(
            sampled_points,
            xy_extent=xy_extent,
            z_min=z_min,
            z_max=z_max,
        )
    return PointCloudInitialization(
        points=sampled_points.unsqueeze(0).repeat(int(frame_count), 1, 1),
        colors=sampled_colors.unsqueeze(0).repeat(int(frame_count), 1, 1),
        source_path=resolved,
        source_count=source_count,
        sampled_count=int(sample.numel()),
        normalize_mode=str(normalize_mode),
        coordinate_frame=str(coordinate_frame),
        visibility_filter=str(visibility_filter),
        sample_mode=str(sample_mode),
        filtered_count=filtered_count,
    )


def powerfoam_rays_from_camera(
    camera: CameraSpec,
    *,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    origins, directions = build_camera_rays(camera, height, width, device=device, dtype=dtype)
    return torch.cat([origins, directions], dim=-1).unsqueeze(0).contiguous()


def powerfoam_rays_from_camera_grid(
    cameras: tuple[tuple[CameraSpec, ...], ...],
    *,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if not cameras:
        raise ValueError("Expected at least one camera view.")
    per_view = []
    for view_cameras in cameras:
        if not view_cameras:
            raise ValueError("Expected at least one frame camera per view.")
        per_view.append(
            torch.cat(
                [
                    powerfoam_rays_from_camera(
                        camera,
                        height=height,
                        width=width,
                        device=device,
                        dtype=dtype,
                    )
                    for camera in view_cameras
                ],
                dim=0,
            )
        )
    return torch.stack(per_view, dim=0).contiguous()


def flatten_multiview_powerfoam_samples(
    frames: torch.Tensor,
    rays: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if frames.ndim != 5:
        raise ValueError(f"Expected multiview frames [V,T,C,H,W], got {tuple(frames.shape)}.")
    if rays.ndim != 5:
        raise ValueError(f"Expected multiview rays [V,T,H,W,6], got {tuple(rays.shape)}.")
    view_count, frame_count = int(frames.shape[0]), int(frames.shape[1])
    if tuple(rays.shape[:2]) != (view_count, frame_count):
        raise ValueError(f"Frame/ray view-time mismatch: {tuple(frames.shape[:2])} vs {tuple(rays.shape[:2])}.")
    targets = frames.reshape(view_count * frame_count, *frames.shape[2:]).contiguous()
    sample_frame_indices = torch.arange(frame_count, device=frames.device, dtype=torch.long).repeat(view_count)
    sample_rays = rays.reshape(view_count * frame_count, *rays.shape[2:]).contiguous()
    return targets, sample_frame_indices, sample_rays


def rays_for_sample_batch(
    rays: torch.Tensor | None,
    *,
    sample_index: int,
    batch_size: int,
) -> torch.Tensor | None:
    if rays is None:
        return None
    if rays.ndim != 4:
        raise ValueError(f"Expected rays [B,H,W,6], got {tuple(rays.shape)}.")
    if rays.shape[-1] != 6:
        raise ValueError(f"Expected ray payload dimension 6, got {rays.shape[-1]}.")
    if rays.shape[0] == 1:
        return rays
    if rays.shape[0] != batch_size:
        raise ValueError(f"Expected {batch_size} ray batches or one shared ray batch, got {rays.shape[0]}.")
    return rays[sample_index : sample_index + 1].contiguous()


def dense_overlap_mask(points_cpu: torch.Tensor, radii_cpu: torch.Tensor) -> torch.Tensor:
    dist_matrix = torch.cdist(points_cpu, points_cpu)
    overlap = dist_matrix <= (radii_cpu[:, None] + radii_cpu[None, :])
    overlap.fill_diagonal_(False)
    return overlap


def _ids_sorted_by_distance(ids: torch.Tensor, dist: torch.Tensor) -> list[int]:
    return [int(v) for v in sorted(ids.tolist(), key=lambda idx: (float(dist[idx]), int(idx)))]


def build_csr_adjacency(
    points: torch.Tensor,
    radii: torch.Tensor,
    *,
    neighbor_count: int,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    points_cpu = points.detach().to(device="cpu", dtype=torch.float32)
    radii_cpu = radii.detach().to(device="cpu", dtype=torch.float32)
    cell_count = points_cpu.shape[0]
    k = min(max(int(neighbor_count), 0), max(cell_count - 1, 0))
    if cell_count == 0:
        return (
            torch.empty(0, device=points.device, dtype=torch.int32),
            torch.zeros(1, device=points.device, dtype=torch.int32),
        )

    dist_matrix = torch.cdist(points_cpu, points_cpu)
    dist_matrix.fill_diagonal_(float("inf"))
    if mode == "knn":
        if k == 0:
            return (
                torch.empty(0, device=points.device, dtype=torch.int32),
                torch.zeros(cell_count + 1, device=points.device, dtype=torch.int32),
            )
        rows_tensor = torch.topk(dist_matrix, k=k, dim=-1, largest=False).indices.reshape(-1)
        offsets_tensor = torch.arange(0, (cell_count + 1) * k, k, device=points.device, dtype=torch.int32)
        return (
            rows_tensor.to(device=points.device, dtype=torch.int32),
            offsets_tensor,
        )
    if mode == "cech_aabb":
        overlap = dense_overlap_mask(points_cpu, radii_cpu)
        rows: list[int] = []
        offsets = [0]
        for i in range(cell_count):
            ids = torch.nonzero(overlap[i], as_tuple=False).flatten()
            rows.extend(_ids_sorted_by_distance(ids, dist_matrix[i]))
            offsets.append(len(rows))
        return (
            torch.tensor(rows, device=points.device, dtype=torch.int32),
            torch.tensor(offsets, device=points.device, dtype=torch.int32),
        )
    if mode == "regular_triangulation":
        return make_regular_triangulation_adjacency(points, radii)
    if mode != "overlap":
        raise ValueError(f"Unknown powerfoam adjacency mode {mode!r}")

    rows: list[int] = []
    offsets = [0]
    for i in range(cell_count):
        dist = dist_matrix[i]
        mask = dist <= (radii_cpu + radii_cpu[i])
        ids = torch.nonzero(mask, as_tuple=False).flatten()
        if ids.numel() == 0 and k > 0:
            ids = torch.topk(dist, k=k, largest=False).indices
        elif ids.numel() > k > 0:
            ids = ids[torch.topk(dist[ids], k=k, largest=False).indices]
        rows.extend(int(v) for v in ids.tolist())
        offsets.append(len(rows))
    return (
        torch.tensor(rows, device=points.device, dtype=torch.int32),
        torch.tensor(offsets, device=points.device, dtype=torch.int32),
    )


def csr_adjacency_stats(
    points: torch.Tensor,
    radii: torch.Tensor,
    rows: torch.Tensor,
    offsets: torch.Tensor,
    *,
    max_dense_cells: int = 4096,
) -> dict[str, float]:
    points_cpu = points.detach().to(device="cpu", dtype=torch.float32)
    radii_cpu = radii.detach().to(device="cpu", dtype=torch.float32)
    offsets_cpu = offsets.detach().to(device="cpu", dtype=torch.int64)
    rows_cpu = rows.detach().to(device="cpu", dtype=torch.int64)
    cell_count = int(points_cpu.shape[0])
    degrees = offsets_cpu[1:] - offsets_cpu[:-1]
    stats = {
        "adjacency_avg_degree": float(degrees.float().mean().item()) if cell_count > 0 else 0.0,
        "adjacency_max_degree": float(degrees.max().item()) if degrees.numel() > 0 else 0.0,
        "adjacency_edges": float(rows_cpu.numel()),
        "adjacency_required_overlap_edges": -1.0,
        "adjacency_missing_overlap_edges": -1.0,
    }
    if cell_count > int(max_dense_cells):
        return stats

    required = dense_overlap_mask(points_cpu, radii_cpu)
    present = torch.zeros_like(required)
    for cell in range(cell_count):
        start = int(offsets_cpu[cell])
        end = int(offsets_cpu[cell + 1])
        ids = rows_cpu[start:end]
        ids = ids[(ids >= 0) & (ids < cell_count) & (ids != cell)]
        if ids.numel() > 0:
            present[cell, ids] = True
    stats["adjacency_required_overlap_edges"] = float(required.sum().item())
    stats["adjacency_missing_overlap_edges"] = float((required & ~present).sum().item())
    return stats


def stable_tangent_from_normals(normals: torch.Tensor) -> torch.Tensor:
    z_axis = normals.new_tensor([0.0, 0.0, 1.0]).expand_as(normals)
    y_axis = normals.new_tensor([0.0, 1.0, 0.0]).expand_as(normals)
    helper = torch.where(normals[..., 2:3].abs() < 0.9, z_axis, y_axis)
    return F.normalize(torch.cross(helper, normals, dim=-1), dim=-1, eps=1.0e-6)


def orthonormal_surface_frame(normals: torch.Tensor, raw_tangents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tangents = raw_tangents - (raw_tangents * normals).sum(dim=-1, keepdim=True) * normals
    fallback = stable_tangent_from_normals(normals)
    tangent_norm = tangents.norm(dim=-1, keepdim=True)
    tangents = torch.where(tangent_norm > 1.0e-6, tangents / tangent_norm.clamp_min(1.0e-6), fallback)
    bitangents = F.normalize(torch.cross(normals, tangents, dim=-1), dim=-1, eps=1.0e-6)
    return tangents, bitangents


class MetalPowerFoamVideo(nn.Module):
    def __init__(
        self,
        *,
        frame_count: int,
        cell_count: int,
        render_size: int,
        fov_degrees: float,
        neighbor_count: int,
        adjacency_mode: str,
        xy_extent: float,
        z_min: float,
        z_max: float,
        radius_init: float,
        radius_min: float,
        radius_scale: float,
        density_init: float,
        feature_mode: str,
        linear_coeff_init: float,
        linear_coeff_scale: float,
        normal_init_jitter: float,
        num_texel_sites: int,
        texel_site_scale: float,
        texel_height_scale: float,
        sv_dof: int,
        sv_axis_init: float,
        sv_axis_init_jitter: float,
        sv_rgb_init_jitter: float,
        color_init_mode: str,
        seed: int,
        init_frames: torch.Tensor | None,
        init_points: torch.Tensor | None,
        init_colors: torch.Tensor | None,
        image_init_depth: float | None,
        image_init_jitter: float,
        raster_config: FoamRasterConfig,
        use_raytrace: bool = False,
    ) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        texel_sites_init = None
        texel_colors_init = None
        texel_heights_init = None
        texel_sv_axis_init = None
        texel_sv_rgb_init = None
        quaternions_init = None
        texel_surface_modes = TEXEL_SURFACE_MODES
        height_texel_surface_modes = HEIGHT_TEXEL_SURFACE_MODES
        quaternion_texel_surface_modes = QUATERNION_TEXEL_SURFACE_MODES
        oriented_texel_surface_modes = ORIENTED_TEXEL_SURFACE_MODES
        sv_texel_surface_modes = SV_TEXEL_SURFACE_MODES
        if init_points is not None:
            if init_points.shape != (frame_count, cell_count, 3):
                raise ValueError(
                    f"init_points must have shape {(frame_count, cell_count, 3)}, got {tuple(init_points.shape)}."
                )
            init_points = init_points.to(dtype=torch.float32)
            if init_colors is None:
                init_colors = torch.full((frame_count, cell_count, 3), 0.5, dtype=init_points.dtype)
            if init_colors.shape != (frame_count, cell_count, 3):
                raise ValueError(
                    f"init_colors must have shape {(frame_count, cell_count, 3)}, got {tuple(init_colors.shape)}."
                )
            init_colors = init_colors.to(dtype=init_points.dtype).clamp(0.0, 1.0)
            init_radii = estimate_knn_radii(init_points, radius_scale=radius_scale, radius_min=radius_min)
            quaternions_init = camera_facing_quaternion(frame_count, cell_count)
        elif init_frames is not None and str(feature_mode) in texel_surface_modes:
            init = initialize_full_powerfoam_from_video(
                init_frames,
                cell_count=cell_count,
                xy_extent=xy_extent,
                z_min=z_min,
                z_max=z_max,
                fov_degrees=fov_degrees,
                image_init_depth=image_init_depth,
                radius_min=radius_min,
                radius_scale=radius_scale,
                num_texel_sites=int(num_texel_sites),
                sv_dof=int(sv_dof) if str(feature_mode) in sv_texel_surface_modes else 1,
                sv_axis_init=float(sv_axis_init),
                image_init_jitter=image_init_jitter,
                generator=generator,
            )
            init_points = init.points
            init_radii = init.radii
            quaternions_init = init.quaternions
            texel_sites_init = init.texel_sites.clamp(-float(texel_site_scale) * 0.999, float(texel_site_scale) * 0.999)
            texel_sv_axis_init = init.texel_sv_axis
            texel_sv_rgb_init = init.texel_sv_rgb
            texel_colors_init = (init.texel_sv_rgb[..., 0, :] + 0.5).clamp(0.0, 1.0)
            texel_heights_init = init.texel_height
            init_colors = texel_colors_init.mean(dim=2)
        elif init_frames is not None:
            init_points, init_colors = initialize_powerfoam_from_video(
                init_frames,
                cell_count=cell_count,
                xy_extent=xy_extent,
                z_min=z_min,
                z_max=z_max,
                fov_degrees=fov_degrees,
                image_init_depth=image_init_depth,
                image_init_jitter=image_init_jitter,
                generator=generator,
            )
            init_radii = estimate_knn_radii(init_points, radius_scale=radius_scale, radius_min=radius_min)
        else:
            xy = (torch.rand(frame_count, cell_count, 2, generator=generator) * 2.0 - 1.0) * float(xy_extent)
            z = torch.rand(frame_count, cell_count, 1, generator=generator) * (float(z_max) - float(z_min)) + float(z_min)
            init_points = torch.cat([xy, z], dim=-1)
            init_colors = torch.rand(frame_count, cell_count, 3, generator=generator)
            init_radii = torch.full((frame_count, cell_count), max(float(radius_init), float(radius_min)))
            quaternions_init = camera_facing_quaternion(frame_count, cell_count)
        if str(color_init_mode) == "random":
            init_colors = torch.rand(frame_count, cell_count, 3, generator=generator, dtype=init_points.dtype)
            if str(feature_mode) in sv_texel_surface_modes:
                texel_sv_axis_init = float(sv_axis_init) * F.normalize(
                    torch.randn(
                        frame_count,
                        cell_count,
                        int(num_texel_sites),
                        int(sv_dof),
                        3,
                        generator=generator,
                        dtype=init_points.dtype,
                    ),
                    dim=-1,
                    eps=1.0e-6,
                )
                texel_sv_rgb_init = (
                    torch.rand(
                        frame_count,
                        cell_count,
                        int(num_texel_sites),
                        int(sv_dof),
                        3,
                        generator=generator,
                        dtype=init_points.dtype,
                    )
                    - 0.5
                )
            elif str(feature_mode) in texel_surface_modes:
                texel_colors_init = torch.rand(
                    frame_count,
                    cell_count,
                    int(num_texel_sites),
                    3,
                    generator=generator,
                    dtype=init_points.dtype,
                )

        self.raw_xy = nn.Parameter(torch.atanh((init_points[..., :2] / float(xy_extent)).clamp(-0.9999, 0.9999)))
        self.raw_z = nn.Parameter(logit_clamped((init_points[..., 2:] - float(z_min)) / (float(z_max) - float(z_min))))
        self.raw_radii = nn.Parameter(
            inverse_softplus(
                (init_radii - float(radius_min)).clamp_min(1.0e-4),
                beta=POWERFOAM_SOFTPLUS_BETA,
            )
        )
        init_density = torch.full((frame_count, cell_count), max(float(density_init), 1.0e-4))
        self.raw_densities = nn.Parameter(inverse_softplus(init_density, beta=POWERFOAM_SOFTPLUS_BETA))
        self.feature_mode = str(feature_mode)
        self.linear_coeff_scale = float(linear_coeff_scale)
        self.texel_site_scale = float(texel_site_scale)
        self.texel_height_scale = float(texel_height_scale)
        if self.feature_mode in texel_surface_modes:
            if texel_sites_init is None:
                site_cols = math.ceil(math.sqrt(float(num_texel_sites)))
                site_rows = math.ceil(float(num_texel_sites) / float(site_cols))
                xs = (torch.arange(site_cols, dtype=init_colors.dtype) + 0.5) / float(site_cols) - 0.5
                ys = (torch.arange(site_rows, dtype=init_colors.dtype) + 0.5) / float(site_rows) - 0.5
                yy, xx = torch.meshgrid(ys, xs, indexing="ij")
                grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)[: int(num_texel_sites)]
                texel_sites_init = grid.view(1, 1, int(num_texel_sites), 2).repeat(frame_count, cell_count, 1, 1)
                texel_colors_init = init_colors[:, :, None, :].repeat(1, 1, int(num_texel_sites), 1)
            if texel_heights_init is None:
                texel_heights_init = torch.zeros(
                    frame_count,
                    cell_count,
                    int(num_texel_sites),
                    dtype=init_colors.dtype,
                )
            self.raw_texel_sites = nn.Parameter(
                torch.atanh((texel_sites_init / self.texel_site_scale).clamp(-0.9999, 0.9999))
            )
            if self.feature_mode in sv_texel_surface_modes:
                if texel_sv_axis_init is None:
                    texel_sv_axis_init = float(sv_axis_init) * F.normalize(
                        torch.randn(
                            frame_count,
                            cell_count,
                            int(num_texel_sites),
                            int(sv_dof),
                            3,
                            generator=generator,
                            dtype=init_colors.dtype,
                        ),
                        dim=-1,
                        eps=1.0e-6,
                    )
                if texel_sv_rgb_init is None:
                    texel_sv_rgb_init = init_colors[:, :, None, None, :].repeat(
                        1,
                        1,
                        int(num_texel_sites),
                        int(sv_dof),
                        1,
                    ) - 0.5
                if float(sv_axis_init_jitter) != 0.0:
                    texel_sv_axis_init = texel_sv_axis_init + float(sv_axis_init_jitter) * torch.randn(
                        texel_sv_axis_init.shape,
                        generator=generator,
                        dtype=texel_sv_axis_init.dtype,
                    )
                if float(sv_rgb_init_jitter) != 0.0:
                    texel_sv_rgb_init = texel_sv_rgb_init + float(sv_rgb_init_jitter) * torch.randn(
                        texel_sv_rgb_init.shape,
                        generator=generator,
                        dtype=texel_sv_rgb_init.dtype,
                    )
                self.raw_texel_sv_axis = nn.Parameter(texel_sv_axis_init)
                self.raw_texel_sv_rgb = nn.Parameter(texel_sv_rgb_init)
                init_decoded_features = (texel_sv_rgb_init.mean(dim=3) + 0.5).clamp_min(0.0)
                self.raw_features = None
            else:
                init_decoded_features = texel_colors_init.clamp(0.0, 1.0)
                self.raw_features = nn.Parameter(logit_clamped(init_decoded_features))
                self.raw_texel_sv_axis = None
                self.raw_texel_sv_rgb = None
            if self.feature_mode in height_texel_surface_modes:
                self.raw_texel_heights = nn.Parameter(
                    torch.atanh((texel_heights_init / self.texel_height_scale).clamp(-0.9999, 0.9999))
                )
            else:
                self.raw_texel_heights = None
        elif self.feature_mode in {"linear", "surface_linear", "oriented_surface_linear"}:
            raw_features = torch.zeros(frame_count, cell_count, 3, 4, dtype=init_colors.dtype)
            raw_features[..., 0] = logit_clamped(init_colors.clamp(0.0, 1.0))
            if float(linear_coeff_init) != 0.0:
                raw_features[..., 1:] = float(linear_coeff_init) * torch.randn(
                    frame_count,
                    cell_count,
                    3,
                    3,
                    generator=generator,
                    dtype=init_colors.dtype,
                )
            init_decoded_features = torch.zeros_like(raw_features)
            init_decoded_features[..., 0] = init_colors.clamp(0.0, 1.0)
            self.raw_features = nn.Parameter(raw_features)
        else:
            init_decoded_features = init_colors.clamp(0.0, 1.0)
            self.raw_features = nn.Parameter(logit_clamped(init_decoded_features))
        if self.feature_mode not in texel_surface_modes:
            self.raw_texel_sites = None
            texel_sites_init = torch.zeros(frame_count, cell_count, int(num_texel_sites), 2, dtype=init_colors.dtype)
            texel_heights_init = torch.zeros(frame_count, cell_count, int(num_texel_sites), dtype=init_colors.dtype)
            self.raw_texel_heights = None
            self.raw_texel_sv_axis = None
            self.raw_texel_sv_rgb = None
        if self.feature_mode in quaternion_texel_surface_modes:
            if quaternions_init is None:
                quaternions_init = camera_facing_quaternion(frame_count, cell_count)
            self.raw_quaternions = nn.Parameter(quaternions_init)
            init_normals, init_tangents, _init_bitangents = quaternion_frames(quaternions_init, eps=1.0e-6)
            self.raw_normals = None
        else:
            self.raw_quaternions = None
        if self.feature_mode in {"oriented_surface_linear"} | oriented_texel_surface_modes:
            init_normals = torch.zeros(frame_count, cell_count, 3, dtype=init_colors.dtype)
            init_normals[..., 2] = -1.0
            if float(normal_init_jitter) != 0.0:
                init_normals = init_normals + float(normal_init_jitter) * torch.randn(
                    frame_count,
                    cell_count,
                    3,
                    generator=generator,
                    dtype=init_colors.dtype,
                )
            init_normals = F.normalize(init_normals, dim=-1, eps=1.0e-6)
            self.raw_normals = nn.Parameter(init_normals)
        else:
            if self.feature_mode not in quaternion_texel_surface_modes:
                init_normals = torch.zeros(frame_count, cell_count, 3, dtype=init_colors.dtype)
                self.raw_normals = None
        if self.feature_mode in oriented_texel_surface_modes:
            init_tangents = stable_tangent_from_normals(init_normals)
            self.raw_tangents = nn.Parameter(init_tangents)
        else:
            if self.feature_mode not in quaternion_texel_surface_modes:
                init_tangents = torch.zeros(frame_count, cell_count, 3, dtype=init_colors.dtype)
            self.raw_tangents = None
        self.register_buffer("initial_points", init_points.clone(), persistent=False)
        self.register_buffer("initial_radii", init_radii.clone(), persistent=False)
        self.register_buffer("initial_densities", init_density.clone(), persistent=False)
        self.register_buffer("initial_features", init_decoded_features.clone(), persistent=False)
        self.register_buffer("initial_normals", init_normals.clone(), persistent=False)
        self.register_buffer("initial_tangents", init_tangents.clone(), persistent=False)
        self.register_buffer("initial_texel_sites", texel_sites_init.clone(), persistent=False)
        self.register_buffer(
            "initial_texel_heights",
            (texel_heights_init * init_radii[:, :, None] * self.texel_height_scale).clone(),
            persistent=False,
        )
        self.register_buffer(
            "initial_texel_sv_axis",
            texel_sv_axis_init.clone()
            if texel_sv_axis_init is not None
            else torch.zeros(frame_count, cell_count, int(num_texel_sites), int(sv_dof), 3, dtype=init_colors.dtype),
            persistent=False,
        )
        self.register_buffer(
            "initial_texel_sv_rgb",
            texel_sv_rgb_init.clone()
            if texel_sv_rgb_init is not None
            else torch.zeros(frame_count, cell_count, int(num_texel_sites), int(sv_dof), 3, dtype=init_colors.dtype),
            persistent=False,
        )
        self.register_buffer(
            "initial_quaternions",
            quaternions_init.clone()
            if quaternions_init is not None
            else torch.zeros(frame_count, cell_count, 4, dtype=init_colors.dtype),
            persistent=False,
        )

        self.xy_extent = float(xy_extent)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.radius_min = float(radius_min)
        self.neighbor_count = int(neighbor_count)
        self.adjacency_mode = str(adjacency_mode)
        self.raster_config = raster_config
        self.use_raytrace = bool(use_raytrace)
        self.register_buffer("rays", make_pinhole_rays(render_size, render_size, fov_degrees, torch.device("cpu")), persistent=False)
        self.register_buffer("contrib_ema", torch.full((frame_count, cell_count), 1.0e-5), persistent=True)
        self.register_buffer("point_error_ema", torch.full((frame_count, cell_count), 1.0e-5), persistent=True)

    def decoded_parameters(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        xy = torch.tanh(self.raw_xy) * self.xy_extent
        z = self.z_min + torch.sigmoid(self.raw_z) * (self.z_max - self.z_min)
        points = torch.cat([xy, z], dim=-1)
        radii = F.softplus(self.raw_radii, beta=POWERFOAM_SOFTPLUS_BETA) + self.radius_min
        densities = F.softplus(self.raw_densities, beta=POWERFOAM_SOFTPLUS_BETA)
        if self.feature_mode in SV_TEXEL_SURFACE_MODES:
            if self.raw_texel_sv_rgb is None:
                raise RuntimeError("SV texel mode requires raw_texel_sv_rgb")
            features = (self.raw_texel_sv_rgb.mean(dim=3) + 0.5).clamp_min(0.0)
        elif self.feature_mode in TEXEL_SURFACE_MODES:
            features = torch.sigmoid(self.raw_features)
        elif self.feature_mode in {"linear", "surface_linear", "oriented_surface_linear"}:
            base = torch.sigmoid(self.raw_features[..., 0])
            coeffs = self.linear_coeff_scale * torch.tanh(self.raw_features[..., 1:])
            features = torch.cat([base[..., None], coeffs], dim=-1)
        else:
            features = torch.sigmoid(self.raw_features)
        normals = None
        if self.raw_normals is not None:
            normals = F.normalize(self.raw_normals, dim=-1, eps=1.0e-6)
        elif self.raw_quaternions is not None:
            normals, _tangents, _bitangents = quaternion_frames(self.raw_quaternions, eps=1.0e-6)
        return points, radii, densities, features, normals

    def decoded_surface_frame(self, normals: torch.Tensor | None) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if normals is None or self.raw_tangents is None:
            return None, None
        return orthonormal_surface_frame(normals, self.raw_tangents)

    def decoded_texel_sites(self) -> torch.Tensor | None:
        if self.raw_texel_sites is None:
            return None
        return self.texel_site_scale * torch.tanh(self.raw_texel_sites)

    def decoded_texel_heights(self, radii: torch.Tensor) -> torch.Tensor | None:
        if self.raw_texel_heights is None:
            return None
        return radii[..., None] * self.texel_height_scale * torch.tanh(self.raw_texel_heights)

    def decoded_texel_sv(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.raw_texel_sv_axis is None or self.raw_texel_sv_rgb is None:
            return None, None
        return self.raw_texel_sv_axis, self.raw_texel_sv_rgb

    def optimizer_param_groups(self, train_cfg: dict[str, Any]) -> list[dict[str, object]]:
        def group(params: list[nn.Parameter], name: str) -> dict[str, object]:
            return {"params": params, "name": name, **powerfoam_group_lr_metadata(train_cfg, name)}

        groups: list[dict[str, object]] = [
            group([self.raw_xy, self.raw_z], "points"),
            group([self.raw_radii], "radii"),
            group([self.raw_densities], "density"),
        ]
        if self.raw_features is not None:
            groups.append(group([self.raw_features], "features"))
        if self.raw_normals is not None:
            groups.append(group([self.raw_normals], "normals"))
        if self.raw_tangents is not None:
            groups.append(group([self.raw_tangents], "tangents"))
        if self.raw_quaternions is not None:
            groups.append(group([self.raw_quaternions], "quaternions"))
        if self.raw_texel_sites is not None:
            groups.append(group([self.raw_texel_sites], "texel_sites"))
        if self.raw_texel_heights is not None:
            groups.append(group([self.raw_texel_heights], "texel_height"))
        if self.raw_texel_sv_axis is not None:
            groups.append(group([self.raw_texel_sv_axis], "texel_sv_axis"))
        if self.raw_texel_sv_rgb is not None:
            groups.append(group([self.raw_texel_sv_rgb], "texel_sv_rgb"))
        return groups

    @torch.no_grad()
    def parameter_drift_metrics(self) -> dict[str, float]:
        points, radii, densities, features, normals = self.decoded_parameters()
        initial_points = self.initial_points.to(points.device)
        center_offset = points - initial_points
        center_delta = torch.linalg.vector_norm(center_offset, dim=-1)
        metrics = {
            "state_cell_count": float(points.shape[1]),
            "state_mean_center_delta": float(center_delta.mean().cpu()),
            "state_p95_center_delta": float(center_delta.flatten().quantile(0.95).cpu()),
            "state_max_center_delta": float(center_delta.max().cpu()),
            "state_mean_xy_delta": float(torch.linalg.vector_norm(center_offset[..., :2], dim=-1).mean().cpu()),
            "state_mean_z_delta": float(center_offset[..., 2].abs().mean().cpu()),
            "state_mean_radius_delta": float((radii - self.initial_radii.to(radii.device)).abs().mean().cpu()),
            "state_mean_density_delta": float((densities - self.initial_densities.to(densities.device)).abs().mean().cpu()),
            "state_mean_feature_delta": float((features - self.initial_features.to(features.device)).abs().mean().cpu()),
        }
        if normals is not None:
            initial_normals = self.initial_normals.to(normals.device)
            metrics["state_mean_normal_delta"] = float((normals - initial_normals).norm(dim=-1).mean().cpu())
            metrics["state_mean_normal_z"] = float(normals[..., 2].mean().cpu())
            tangents, _bitangents = self.decoded_surface_frame(normals)
            if tangents is not None:
                metrics["state_mean_tangent_delta"] = float(
                    (tangents - self.initial_tangents.to(tangents.device)).norm(dim=-1).mean().cpu()
                )
        if self.raw_quaternions is not None:
            metrics["state_mean_quaternion_delta"] = float(
                (self.raw_quaternions - self.initial_quaternions.to(self.raw_quaternions.device))
                .norm(dim=-1)
                .mean()
                .cpu()
            )
        texel_sites = self.decoded_texel_sites()
        if texel_sites is not None:
            metrics["state_mean_texel_site_delta"] = float(
                (texel_sites - self.initial_texel_sites.to(texel_sites.device)).norm(dim=-1).mean().cpu()
            )
        texel_heights = self.decoded_texel_heights(radii)
        if texel_heights is not None:
            metrics["state_mean_texel_height_delta"] = float(
                (texel_heights - self.initial_texel_heights.to(texel_heights.device)).abs().mean().cpu()
            )
        if self.raw_texel_sv_axis is not None:
            metrics["state_mean_texel_sv_axis_delta"] = float(
                (self.raw_texel_sv_axis - self.initial_texel_sv_axis.to(self.raw_texel_sv_axis.device))
                .norm(dim=-1)
                .mean()
                .cpu()
            )
        if self.raw_texel_sv_rgb is not None:
            metrics["state_mean_texel_sv_rgb_delta"] = float(
                (self.raw_texel_sv_rgb - self.initial_texel_sv_rgb.to(self.raw_texel_sv_rgb.device))
                .abs()
                .mean()
                .cpu()
            )
        return metrics

    def forward(
        self,
        frame_indices: torch.Tensor,
        rays: torch.Tensor | None = None,
        *,
        return_normal_distance: bool = False,
        return_rendered_normal: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        if next(self.parameters()).device.type != "mps":
            raise RuntimeError("MetalPowerFoamVideo requires an MPS device")
        points, radii, densities, features, normals = self.decoded_parameters()
        frame_indices = frame_indices.to(device=points.device, dtype=torch.long)
        default_rays = self.rays.to(device=points.device, dtype=points.dtype)
        provided_rays = None if rays is None else rays.to(device=points.device, dtype=points.dtype)
        renders = []
        alphas = []
        normal_distances = []
        rendered_normals = []
        for sample_index, frame_index in enumerate(frame_indices):
            point = points[frame_index]
            radius = radii[frame_index]
            sample_rays = rays_for_sample_batch(
                provided_rays,
                sample_index=sample_index,
                batch_size=int(frame_indices.numel()),
            )
            if sample_rays is None:
                sample_rays = default_rays
            adjacency, offsets = build_csr_adjacency(
                point,
                radius,
                neighbor_count=self.neighbor_count,
                mode=self.adjacency_mode,
            )
            normal_distance = None
            if self.feature_mode == "linear":
                out, alpha = rasterize_power_foam_linear(
                    point,
                    radius,
                    densities[frame_index],
                    features[frame_index],
                    adjacency,
                    offsets,
                    sample_rays,
                    self.raster_config,
                )
            elif self.feature_mode == "surface_linear":
                out, alpha = rasterize_power_foam_surface_linear(
                    point,
                    radius,
                    densities[frame_index],
                    features[frame_index],
                    adjacency,
                    offsets,
                    sample_rays,
                    self.raster_config,
                )
            elif self.feature_mode == "oriented_surface_linear":
                if normals is None:
                    raise RuntimeError("oriented_surface_linear requires decoded normals")
                out, alpha = rasterize_power_foam_oriented_surface_linear(
                    point,
                    radius,
                    densities[frame_index],
                    features[frame_index],
                    normals[frame_index],
                    adjacency,
                    offsets,
                    sample_rays,
                    self.raster_config,
                )
            elif self.feature_mode == "oriented_texel_surface":
                texel_sites = self.decoded_texel_sites()
                if normals is None or texel_sites is None:
                    raise RuntimeError("oriented_texel_surface requires decoded normals and texel sites")
                tangents, bitangents = self.decoded_surface_frame(normals)
                if tangents is None or bitangents is None:
                    raise RuntimeError("oriented_texel_surface requires decoded surface frame")
                out, alpha = rasterize_power_foam_oriented_texel_surface(
                    point,
                    radius,
                    densities[frame_index],
                    texel_sites[frame_index],
                    features[frame_index],
                    normals[frame_index],
                    adjacency,
                    offsets,
                    sample_rays,
                    self.raster_config,
                    tangents=tangents[frame_index],
                    bitangents=bitangents[frame_index],
                )
            elif self.feature_mode == "oriented_height_texel_surface":
                texel_sites = self.decoded_texel_sites()
                texel_heights = self.decoded_texel_heights(radii)
                if normals is None or texel_sites is None or texel_heights is None:
                    raise RuntimeError("oriented_height_texel_surface requires decoded normals, texel sites, and heights")
                tangents, bitangents = self.decoded_surface_frame(normals)
                if tangents is None or bitangents is None:
                    raise RuntimeError("oriented_height_texel_surface requires decoded surface frame")
                out, alpha = rasterize_power_foam_oriented_height_texel_surface(
                    point,
                    radius,
                    densities[frame_index],
                    texel_sites[frame_index],
                    texel_heights[frame_index],
                    features[frame_index],
                    normals[frame_index],
                    adjacency,
                    offsets,
                    sample_rays,
                    self.raster_config,
                    tangents=tangents[frame_index],
                    bitangents=bitangents[frame_index],
                )
            elif self.feature_mode == "oriented_height_sv_texel_surface":
                texel_sites = self.decoded_texel_sites()
                texel_heights = self.decoded_texel_heights(radii)
                texel_sv_axis, texel_sv_rgb = self.decoded_texel_sv()
                if normals is None or texel_sites is None or texel_heights is None:
                    raise RuntimeError("oriented_height_sv_texel_surface requires decoded normals, texel sites, and heights")
                if texel_sv_axis is None or texel_sv_rgb is None:
                    raise RuntimeError("oriented_height_sv_texel_surface requires decoded SV color parameters")
                tangents, bitangents = self.decoded_surface_frame(normals)
                if tangents is None or bitangents is None:
                    raise RuntimeError("oriented_height_sv_texel_surface requires decoded surface frame")
                if self.use_raytrace:
                    result = raytrace_power_foam_oriented_height_sv_texel_surface(
                        point,
                        radius,
                        densities[frame_index],
                        texel_sites[frame_index],
                        texel_heights[frame_index],
                        texel_sv_axis[frame_index],
                        texel_sv_rgb[frame_index],
                        normals[frame_index],
                        adjacency,
                        offsets,
                        sample_rays,
                        self.raster_config,
                        tangents=tangents[frame_index],
                        bitangents=bitangents[frame_index],
                        return_normal_distance=return_normal_distance,
                        return_normal=return_rendered_normal,
                    )
                else:
                    if return_rendered_normal:
                        raise RuntimeError("return_rendered_normal is currently implemented for raytrace height+SV")
                    result = rasterize_power_foam_oriented_height_sv_texel_surface(
                        point,
                        radius,
                        densities[frame_index],
                        texel_sites[frame_index],
                        texel_heights[frame_index],
                        texel_sv_axis[frame_index],
                        texel_sv_rgb[frame_index],
                        normals[frame_index],
                        adjacency,
                        offsets,
                        sample_rays,
                        self.raster_config,
                        tangents=tangents[frame_index],
                        bitangents=bitangents[frame_index],
                        return_normal_distance=return_normal_distance,
                    )
                if return_normal_distance:
                    out, alpha, normal_distance, *rest = result
                else:
                    out, alpha, *rest = result
                if return_rendered_normal:
                    rendered_normals.append(rest[0])
            elif self.feature_mode == "quaternion_texel_surface":
                texel_sites = self.decoded_texel_sites()
                if texel_sites is None or self.raw_quaternions is None:
                    raise RuntimeError("quaternion_texel_surface requires decoded texel sites and quaternions")
                out, alpha = rasterize_power_foam_quaternion_texel_surface(
                    point,
                    radius,
                    densities[frame_index],
                    texel_sites[frame_index],
                    features[frame_index],
                    self.raw_quaternions[frame_index],
                    adjacency,
                    offsets,
                    sample_rays,
                    self.raster_config,
                )
            elif self.feature_mode == "quaternion_height_texel_surface":
                texel_sites = self.decoded_texel_sites()
                texel_heights = self.decoded_texel_heights(radii)
                if texel_sites is None or texel_heights is None or self.raw_quaternions is None:
                    raise RuntimeError("quaternion_height_texel_surface requires decoded texel sites, heights, and quaternions")
                out, alpha = rasterize_power_foam_quaternion_height_texel_surface(
                    point,
                    radius,
                    densities[frame_index],
                    texel_sites[frame_index],
                    texel_heights[frame_index],
                    features[frame_index],
                    self.raw_quaternions[frame_index],
                    adjacency,
                    offsets,
                    sample_rays,
                    self.raster_config,
                )
            elif self.feature_mode == "quaternion_height_sv_texel_surface":
                texel_sites = self.decoded_texel_sites()
                texel_heights = self.decoded_texel_heights(radii)
                texel_sv_axis, texel_sv_rgb = self.decoded_texel_sv()
                if texel_sites is None or texel_heights is None or self.raw_quaternions is None:
                    raise RuntimeError("quaternion_height_sv_texel_surface requires decoded texel sites, heights, and quaternions")
                if texel_sv_axis is None or texel_sv_rgb is None:
                    raise RuntimeError("quaternion_height_sv_texel_surface requires decoded SV color parameters")
                if self.use_raytrace:
                    frame_normals, frame_tangents, frame_bitangents = quaternion_frames(
                        self.raw_quaternions[frame_index], eps=self.raster_config.eps
                    )
                    result = raytrace_power_foam_oriented_height_sv_texel_surface(
                        point,
                        radius,
                        densities[frame_index],
                        texel_sites[frame_index],
                        texel_heights[frame_index],
                        texel_sv_axis[frame_index],
                        texel_sv_rgb[frame_index],
                        frame_normals,
                        adjacency,
                        offsets,
                        sample_rays,
                        self.raster_config,
                        tangents=frame_tangents,
                        bitangents=frame_bitangents,
                        return_normal_distance=return_normal_distance,
                        return_normal=return_rendered_normal,
                    )
                else:
                    if return_rendered_normal:
                        raise RuntimeError("return_rendered_normal is currently implemented for raytrace height+SV")
                    result = rasterize_power_foam_quaternion_height_sv_texel_surface(
                        point,
                        radius,
                        densities[frame_index],
                        texel_sites[frame_index],
                        texel_heights[frame_index],
                        texel_sv_axis[frame_index],
                        texel_sv_rgb[frame_index],
                        self.raw_quaternions[frame_index],
                        adjacency,
                        offsets,
                        sample_rays,
                        self.raster_config,
                        return_normal_distance=return_normal_distance,
                    )
                if return_normal_distance:
                    out, alpha, normal_distance, *rest = result
                else:
                    out, alpha, *rest = result
                if return_rendered_normal:
                    rendered_normals.append(rest[0])
            else:
                out, alpha = rasterize_power_foam(
                    point,
                    radius,
                    densities[frame_index],
                    features[frame_index],
                    adjacency,
                    offsets,
                    sample_rays,
                    self.raster_config,
                )
            renders.append(out.permute(0, 3, 1, 2))
            alphas.append(alpha)
            if return_normal_distance:
                if normal_distance is None:
                    raise RuntimeError("normal_distance output requires a height+SV PowerFoam feature mode")
                normal_distances.append(normal_distance)
        result = (torch.cat(renders, dim=0), torch.cat(alphas, dim=0))
        if return_normal_distance:
            result = (*result, torch.cat(normal_distances, dim=0))
        if return_rendered_normal:
            if len(rendered_normals) != int(frame_indices.numel()):
                raise RuntimeError("rendered normal output requires raytrace height+SV PowerFoam feature mode")
            result = (*result, torch.cat(rendered_normals, dim=0))
        return result

    @torch.no_grad()
    def adjacency_diagnostics(self, frame_index: int = 0) -> dict[str, float]:
        points, radii, _densities, _features, _normals = self.decoded_parameters()
        frame = int(frame_index) % int(points.shape[0])
        adjacency, offsets = build_csr_adjacency(
            points[frame],
            radii[frame],
            neighbor_count=self.neighbor_count,
            mode=self.adjacency_mode,
        )
        return csr_adjacency_stats(points[frame], radii[frame], adjacency, offsets)

    def interpenetration_loss(self, frame_indices: torch.Tensor | None = None) -> torch.Tensor:
        points, radii, _densities, _features, _normals = self.decoded_parameters()
        if frame_indices is None:
            frame_ids = list(range(int(points.shape[0])))
        else:
            frame_ids = sorted({int(v) for v in frame_indices.detach().cpu().flatten().tolist()})
        if not frame_ids:
            return points.sum() * 0.0 + radii.sum() * 0.0

        losses = []
        for frame in frame_ids:
            point = points[frame]
            radius = radii[frame]
            adjacency, offsets = build_csr_adjacency(
                point,
                radius,
                neighbor_count=self.neighbor_count,
                mode=self.adjacency_mode,
            )
            if adjacency.numel() == 0:
                losses.append(point.sum() * 0.0 + radius.sum() * 0.0)
                continue
            degrees = (offsets[1:] - offsets[:-1]).to(dtype=torch.long)
            src = torch.repeat_interleave(torch.arange(point.shape[0], device=point.device), degrees)
            dst = adjacency.to(device=point.device, dtype=torch.long)
            distances = torch.linalg.vector_norm(point[src] - point[dst], dim=-1)
            overlap = (radius[src] + radius[dst] - distances).clamp_min(0.0)
            losses.append(overlap.square().sum())
        return torch.stack(losses).mean()

    @torch.no_grad()
    def height_sv_aux_batch(
        self,
        frame_indices: torch.Tensor,
        targets: torch.Tensor | None = None,
        rays: torch.Tensor | None = None,
    ) -> PowerFoamAuxBatch | None:
        if self.feature_mode not in {"oriented_height_sv_texel_surface", "quaternion_height_sv_texel_surface"}:
            return None
        points, radii, densities, _features, normals = self.decoded_parameters()
        texel_sites = self.decoded_texel_sites()
        texel_heights = self.decoded_texel_heights(radii)
        texel_sv_axis, texel_sv_rgb = self.decoded_texel_sv()
        if texel_sites is None or texel_heights is None or texel_sv_axis is None or texel_sv_rgb is None:
            return None
        if normals is None and self.raw_quaternions is None:
            return None

        frame_indices = frame_indices.to(device=points.device, dtype=torch.long)
        default_rays = self.rays.to(device=points.device, dtype=points.dtype)
        provided_rays = None if rays is None else rays.to(device=points.device, dtype=points.dtype)
        targets = None if targets is None else targets.to(device=points.device, dtype=points.dtype)
        contribs = []
        point_errors = []
        visible = []
        normal_distance = []
        normal_norm = []
        median_depth = []
        for sample_index, frame_index in enumerate(frame_indices):
            frame = int(frame_index.detach().cpu())
            point = points[frame]
            radius = radii[frame]
            sample_rays = rays_for_sample_batch(
                provided_rays,
                sample_index=sample_index,
                batch_size=int(frame_indices.numel()),
            )
            if sample_rays is None:
                sample_rays = default_rays
            adjacency, offsets = build_csr_adjacency(
                point,
                radius,
                neighbor_count=self.neighbor_count,
                mode=self.adjacency_mode,
            )
            target = None
            if targets is not None:
                target = targets[sample_index : sample_index + 1] if targets.shape[0] == int(frame_indices.numel()) else targets[frame : frame + 1]
            if self.feature_mode == "quaternion_height_sv_texel_surface":
                if self.raw_quaternions is None:
                    return None
                aux = rasterize_power_foam_quaternion_height_sv_texel_surface_aux(
                    point,
                    radius,
                    densities[frame],
                    texel_sites[frame],
                    texel_heights[frame],
                    texel_sv_axis[frame],
                    texel_sv_rgb[frame],
                    self.raw_quaternions[frame],
                    adjacency,
                    offsets,
                    sample_rays,
                    target,
                    self.raster_config,
                )
            else:
                if normals is None:
                    return None
                tangents, bitangents = self.decoded_surface_frame(normals)
                if tangents is None or bitangents is None:
                    return None
                aux = rasterize_power_foam_oriented_height_sv_texel_surface_aux(
                    point,
                    radius,
                    densities[frame],
                    texel_sites[frame],
                    texel_heights[frame],
                    texel_sv_axis[frame],
                    texel_sv_rgb[frame],
                    normals[frame],
                    adjacency,
                    offsets,
                    sample_rays,
                    target,
                    self.raster_config,
                    tangents=tangents[frame],
                    bitangents=bitangents[frame],
                )
            contribs.append(aux.contrib)
            point_errors.append(aux.point_error)
            visible.append(aux.visible_mask.to(dtype=points.dtype))
            normal_distance.append(aux.normal_distance)
            normal_norm.append(aux.normal.norm(dim=1))
            median_depth.append(aux.median_depth)

        return PowerFoamAuxBatch(
            contrib=torch.cat(contribs, dim=0),
            point_error=torch.cat(point_errors, dim=0),
            visible_mask=torch.cat(visible, dim=0),
            normal_distance=torch.cat(normal_distance, dim=0),
            normal_norm=torch.cat(normal_norm, dim=0),
            median_depth=torch.cat(median_depth, dim=0),
        )

    @torch.no_grad()
    def aux_metrics(
        self,
        frame_indices: torch.Tensor,
        targets: torch.Tensor,
        rays: torch.Tensor | None = None,
    ) -> dict[str, float]:
        aux = self.height_sv_aux_batch(frame_indices, targets, rays)
        if aux is None:
            return {}
        contrib = aux.contrib
        point_error = aux.point_error
        visible_tensor = aux.visible_mask.to(dtype=aux.normal_distance.dtype)
        normal_distance_tensor = aux.normal_distance
        normal_norm_tensor = aux.normal_norm
        median_depth_tensor = aux.median_depth
        ema_frames = frame_indices.to(device=aux.normal_distance.device, dtype=torch.long)
        for row, frame in enumerate(ema_frames):
            visible_alpha = 0.99 * visible_tensor[row] + (1.0 - visible_tensor[row])
            self.contrib_ema[frame] = visible_alpha * self.contrib_ema[frame] + (1.0 - visible_alpha) * contrib[row]
            self.point_error_ema[frame] = 0.99 * self.point_error_ema[frame] + 0.01 * point_error[row]
        valid_depth = median_depth_tensor >= 0.0
        metrics = {
            "aux_mean_contrib": float(contrib.mean().detach().cpu()),
            "aux_max_contrib": float(contrib.max().detach().cpu()),
            "aux_mean_point_error": float(point_error.mean().detach().cpu()),
            "aux_max_point_error": float(point_error.max().detach().cpu()),
            "aux_mean_contrib_ema": float(self.contrib_ema[ema_frames].mean().detach().cpu()),
            "aux_mean_point_error_ema": float(self.point_error_ema[ema_frames].mean().detach().cpu()),
            "aux_visible_fraction": float(visible_tensor.mean().detach().cpu()),
            "aux_mean_normal_distance": float(normal_distance_tensor.mean().detach().cpu()),
            "aux_mean_normal_norm": float(normal_norm_tensor.mean().detach().cpu()),
            "aux_median_depth_valid_fraction": float(valid_depth.to(dtype=aux.normal_distance.dtype).mean().detach().cpu()),
        }
        if bool(valid_depth.any().detach().cpu()):
            metrics["aux_mean_median_depth"] = float(median_depth_tensor[valid_depth].mean().detach().cpu())
        return metrics

    def _gather_cells(self, values: torch.Tensor, new_indices: torch.Tensor) -> torch.Tensor:
        frame_ids = torch.arange(new_indices.shape[0], device=values.device, dtype=torch.long)[:, None]
        indices = new_indices.to(device=values.device, dtype=torch.long)
        return values[frame_ids, indices]

    def _reindex_parameter_cells(
        self,
        param: nn.Parameter | None,
        new_indices: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        if param is None:
            return
        param.data.copy_(self._gather_cells(param.data, new_indices))
        state = optimizer.state.get(param)
        if state is None:
            return
        for value in state.values():
            if torch.is_tensor(value) and value.shape[:2] == param.shape[:2]:
                value.copy_(self._gather_cells(value, new_indices))

    def _replace_optimizer_parameter(
        self,
        old_param: nn.Parameter,
        new_param: nn.Parameter,
        new_indices: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> nn.Parameter:
        for group in optimizer.param_groups:
            group["params"] = [new_param if param is old_param else param for param in group["params"]]
        state = optimizer.state.pop(old_param, None)
        if state is not None:
            for key, value in list(state.items()):
                if torch.is_tensor(value) and value.shape[:2] == old_param.shape[:2]:
                    state[key] = self._gather_cells(value, new_indices).clone()
            optimizer.state[new_param] = state
        return new_param

    def _resize_parameter_cells(
        self,
        param: nn.Parameter | None,
        new_indices: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> nn.Parameter | None:
        if param is None:
            return None
        resized = self._gather_cells(param.data, new_indices).contiguous()
        if resized.shape == param.shape:
            param.data.copy_(resized)
            state = optimizer.state.get(param)
            if state is not None:
                for value in state.values():
                    if torch.is_tensor(value) and value.shape[:2] == param.shape[:2]:
                        value.copy_(self._gather_cells(value, new_indices))
            return param
        return self._replace_optimizer_parameter(param, nn.Parameter(resized), new_indices, optimizer)

    def _resize_buffer_cells(self, name: str, new_indices: torch.Tensor) -> None:
        value = getattr(self, name)
        if torch.is_tensor(value) and value.dim() >= 2 and value.shape[0] == new_indices.shape[0]:
            setattr(self, name, self._gather_cells(value, new_indices).contiguous())

    @torch.no_grad()
    def resample_from_ema(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        target_cells: int | None = None,
        perturb_scale: float = 0.05,
    ) -> dict[str, float]:
        frame_count, cell_count = self.contrib_ema.shape
        target_count = cell_count if target_cells is None else int(target_cells)
        if target_count < 1:
            raise ValueError("target_cells must be positive")
        if cell_count < 2:
            return {
                "resample_replaced": 0.0,
                "resample_valid_mean": float(cell_count),
                "resample_cell_count": float(cell_count),
            }

        new_rows: list[torch.Tensor] = []
        duplicate_count_rows: list[torch.Tensor] = []
        duplicate_mask_rows: list[torch.Tensor] = []
        total_replaced = 0
        total_pruned = 0
        total_invalid_pruned = 0
        points, radii, densities, features, normals = self.decoded_parameters()
        for frame in range(frame_count):
            contrib = self.contrib_ema[frame]
            point_error = self.point_error_ema[frame]
            finite_mask = (
                torch.isfinite(points[frame]).all(dim=-1)
                & torch.isfinite(radii[frame])
                & torch.isfinite(densities[frame])
                & torch.isfinite(features[frame].reshape(cell_count, -1)).all(dim=-1)
                & torch.isfinite(contrib)
                & torch.isfinite(point_error)
            )
            if normals is not None:
                finite_mask = finite_mask & torch.isfinite(normals[frame]).all(dim=-1)
            total_invalid_pruned += int(cell_count - int(finite_mask.sum().detach().cpu()))
            finite_indices = torch.nonzero(finite_mask, as_tuple=False).flatten()
            if finite_indices.numel() == 0:
                raise RuntimeError("PowerFoam resampling found no finite cells to keep.")
            finite_contrib = contrib[finite_indices]
            contrib_q = torch.quantile(finite_contrib, torch.tensor([0.1, 0.99], device=contrib.device), dim=0)
            threshold = torch.minimum(
                torch.tensor(1.0 / (float(cell_count) * 25.0), device=contrib.device, dtype=contrib.dtype),
                contrib_q[0],
            )
            valid_indices = torch.nonzero(finite_mask & (contrib > threshold), as_tuple=False).flatten()
            if valid_indices.numel() == 0:
                finite_scores = finite_contrib.nan_to_num(nan=-float("inf"), neginf=-float("inf"))
                valid_indices = finite_indices[torch.topk(finite_scores, k=1, largest=True).indices]
            if valid_indices.numel() >= target_count:
                order = torch.argsort(contrib[valid_indices], descending=True, stable=True)
                new_indices = valid_indices[order[:target_count]]
                total_pruned += cell_count - target_count
                duplicate_count = torch.ones(target_count, device=contrib.device, dtype=contrib.dtype)
                duplicate_mask = torch.zeros(target_count, device=contrib.device, dtype=torch.bool)
            else:
                num_samples = target_count - int(valid_indices.numel())
                total_replaced += num_samples
                point_error_q = torch.quantile(point_error, 0.99, dim=0)
                prob = point_error[valid_indices].clamp(min=0.0, max=float(point_error_q.detach().cpu()))
                if float(prob.sum().detach().cpu()) <= 0.0:
                    prob = torch.ones_like(prob)
                sampled_pos = torch.multinomial(
                    prob,
                    num_samples,
                    replacement=num_samples > int(valid_indices.numel()),
                )
                duplicate_count_valid = torch.ones(valid_indices.numel(), device=contrib.device, dtype=contrib.dtype)
                duplicate_count_valid.index_add_(0, sampled_pos, torch.ones_like(sampled_pos, dtype=contrib.dtype))
                new_indices = torch.cat([valid_indices, valid_indices[sampled_pos]], dim=0)
                duplicate_count = torch.cat([duplicate_count_valid, duplicate_count_valid[sampled_pos]], dim=0)
                duplicate_mask = duplicate_count > 1.0
            new_rows.append(new_indices)
            duplicate_count_rows.append(duplicate_count)
            duplicate_mask_rows.append(duplicate_mask)

        new_indices_tensor = torch.stack(new_rows, dim=0).to(device=self.raw_xy.device, dtype=torch.long)
        duplicate_count_tensor = torch.stack(duplicate_count_rows, dim=0).to(device=self.raw_xy.device, dtype=self.raw_xy.dtype)
        duplicate_mask_tensor = torch.stack(duplicate_mask_rows, dim=0).to(device=self.raw_xy.device)

        self.raw_xy = self._resize_parameter_cells(self.raw_xy, new_indices_tensor, optimizer)
        self.raw_z = self._resize_parameter_cells(self.raw_z, new_indices_tensor, optimizer)
        self.raw_radii = self._resize_parameter_cells(self.raw_radii, new_indices_tensor, optimizer)
        self.raw_densities = self._resize_parameter_cells(self.raw_densities, new_indices_tensor, optimizer)
        self.raw_features = self._resize_parameter_cells(self.raw_features, new_indices_tensor, optimizer)
        self.raw_normals = self._resize_parameter_cells(self.raw_normals, new_indices_tensor, optimizer)
        self.raw_tangents = self._resize_parameter_cells(self.raw_tangents, new_indices_tensor, optimizer)
        self.raw_quaternions = self._resize_parameter_cells(self.raw_quaternions, new_indices_tensor, optimizer)
        self.raw_texel_sites = self._resize_parameter_cells(self.raw_texel_sites, new_indices_tensor, optimizer)
        self.raw_texel_heights = self._resize_parameter_cells(self.raw_texel_heights, new_indices_tensor, optimizer)
        self.raw_texel_sv_axis = self._resize_parameter_cells(self.raw_texel_sv_axis, new_indices_tensor, optimizer)
        self.raw_texel_sv_rgb = self._resize_parameter_cells(self.raw_texel_sv_rgb, new_indices_tensor, optimizer)

        self.contrib_ema = (self._gather_cells(self.contrib_ema, new_indices_tensor) / duplicate_count_tensor).contiguous()
        self.point_error_ema = (self._gather_cells(self.point_error_ema, new_indices_tensor) / duplicate_count_tensor).contiguous()
        for name in (
            "initial_points",
            "initial_radii",
            "initial_densities",
            "initial_features",
            "initial_normals",
            "initial_tangents",
            "initial_texel_sites",
            "initial_texel_heights",
            "initial_texel_sv_axis",
            "initial_texel_sv_rgb",
            "initial_quaternions",
        ):
            self._resize_buffer_cells(name, new_indices_tensor)

        if float(perturb_scale) > 0.0 and bool(duplicate_mask_tensor.any().detach().cpu()):
            points, radii, _densities, _features, normals = self.decoded_parameters()
            if normals is None:
                normals = torch.zeros_like(points)
                normals[..., 2] = -1.0
            direction = torch.randn_like(points)
            direction = direction - (direction * normals).sum(dim=-1, keepdim=True) * normals
            direction = F.normalize(direction, dim=-1, eps=1.0e-6)
            perturbed = points + float(perturb_scale) * radii[..., None] * direction
            xy = perturbed[..., :2].clamp(-self.xy_extent * 0.9999, self.xy_extent * 0.9999)
            z = perturbed[..., 2:].clamp(self.z_min + 1.0e-4, self.z_max - 1.0e-4)
            encoded_xy = torch.atanh((xy / self.xy_extent).clamp(-0.9999, 0.9999))
            encoded_z = logit_clamped((z - self.z_min) / (self.z_max - self.z_min))
            mask = duplicate_mask_tensor[..., None]
            self.raw_xy.data.copy_(torch.where(mask, encoded_xy, self.raw_xy.data))
            self.raw_z.data.copy_(torch.where(mask, encoded_z, self.raw_z.data))

        return {
            "resample_replaced": float(total_replaced),
            "resample_pruned": float(total_pruned),
            "resample_invalid_pruned": float(total_invalid_pruned),
            "resample_valid_mean": float(target_count - (total_replaced / max(frame_count, 1))),
            "resample_cell_count": float(target_count),
        }


def make_raster_config(render_cfg: dict[str, Any]) -> FoamRasterConfig:
    return FoamRasterConfig(
        near_plane=float(render_cfg["near_plane"]),
        alpha_threshold=float(render_cfg["alpha_threshold"]),
        transmittance_threshold=float(render_cfg["transmittance_threshold"]),
        max_alpha=float(render_cfg["max_alpha"]),
        eps=float(render_cfg["eps"]),
        texel_temperature=float(render_cfg["texel_temperature"]),
        use_tiled=bool(render_cfg["use_tiled"]),
        tiled_builder=str(render_cfg["tiled_builder"]),
    )


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(serialize_config_value(payload), sort_keys=True) + "\n")


def should_log_video(cfg: dict[str, Any], step: int) -> bool:
    return step % int(cfg["logging"]["video_log_every"]) == 0 or (
        bool(cfg["logging"]["always_log_last_step"]) and step == int(cfg["train"]["steps"])
    )


def init_wandb_run(cfg: dict[str, Any]) -> Any | None:
    if not bool(cfg["logging"]["wandb_enabled"]):
        return None
    init_kwargs = {
        "project": cfg["logging"]["wandb_project"],
        "name": cfg["logging"]["wandb_run_name"],
        "tags": cfg["logging"]["wandb_tags"],
        "config": serialize_config_value(cfg),
    }
    if cfg["logging"]["wandb_mode"] is not None:
        init_kwargs["mode"] = str(cfg["logging"]["wandb_mode"])
    return wandb.init(**init_kwargs)


@torch.no_grad()
def render_samples(
    model: MetalPowerFoamVideo,
    frame_indices: torch.Tensor,
    batch_size: int,
    rays: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    renders = []
    alphas = []
    device = next(model.parameters()).device
    frame_indices = frame_indices.to(device=device, dtype=torch.long)
    if rays is not None:
        rays = rays.to(device=device, dtype=torch.float32)
    for start in range(0, int(frame_indices.numel()), batch_size):
        end = min(start + batch_size, int(frame_indices.numel()))
        indices = frame_indices[start:end]
        batch_rays = None if rays is None else rays[start:end]
        rendered, alpha = model(indices, rays=batch_rays)
        renders.append(rendered.detach().cpu())
        alphas.append(alpha.detach().cpu())
    return torch.cat(renders, dim=0), torch.cat(alphas, dim=0)


def reconstruction_eval_metrics(
    renders: torch.Tensor,
    targets: torch.Tensor,
    cfg: dict[str, Any],
    *,
    prefix: str,
) -> dict[str, float]:
    mse = F.mse_loss(renders, targets)
    psnr = -10.0 * torch.log10(mse.clamp_min(1.0e-12))
    window_size = min(int(cfg["losses"]["ssim_window_size"]), int(renders.shape[-1]), int(renders.shape[-2]))
    if window_size % 2 == 0:
        window_size -= 1
    window_size = max(window_size, 1)
    ssim = ssim_per_image(
        renders,
        targets,
        window_size=window_size,
        c1=float(cfg["losses"]["ssim_c1"]),
        c2=float(cfg["losses"]["ssim_c2"]),
    ).mean()
    return {
        f"{prefix}_l1": F.l1_loss(renders, targets).item(),
        f"{prefix}_mse": float(mse.item()),
        f"{prefix}_psnr": float(psnr.item()),
        f"{prefix}_ssim": float(ssim.item()),
    }


def powerfoam_ssim_loss(rendered: torch.Tensor, target: torch.Tensor, loss_cfg: dict[str, Any]) -> torch.Tensor:
    window_size = min(int(loss_cfg["ssim_window_size"]), int(rendered.shape[-1]), int(rendered.shape[-2]))
    if window_size % 2 == 0:
        window_size -= 1
    window_size = max(window_size, 1)
    return 1.0 - ssim_per_image(
        rendered,
        target,
        window_size=window_size,
        c1=float(loss_cfg["ssim_c1"]),
        c2=float(loss_cfg["ssim_c2"]),
    ).mean()


def flatten_rgb_pixels(images: torch.Tensor) -> torch.Tensor:
    return images.permute(0, 2, 3, 1).reshape(-1, images.shape[1]).to(dtype=torch.float32)


def add_bias_column(values: torch.Tensor) -> torch.Tensor:
    return torch.cat([values, torch.ones(values.shape[0], 1, dtype=values.dtype, device=values.device)], dim=1)


def fit_channel_affine(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    rows = []
    for channel in range(rendered.shape[1]):
        x = flatten_rgb_pixels(rendered[:, channel : channel + 1])
        y = flatten_rgb_pixels(target[:, channel : channel + 1])
        rows.append(torch.linalg.lstsq(add_bias_column(x), y).solution[:, 0])
    return torch.stack(rows, dim=0)


def apply_channel_affine(rendered: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    corrected = rendered.clone()
    for channel, row in enumerate(transform):
        corrected[:, channel] = rendered[:, channel] * row[0] + row[1]
    return corrected.clamp(0.0, 1.0)


def fit_rgb_matrix_affine(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.linalg.lstsq(add_bias_column(flatten_rgb_pixels(rendered)), flatten_rgb_pixels(target)).solution


def apply_rgb_matrix_affine(rendered: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    shape = rendered.shape
    corrected = add_bias_column(flatten_rgb_pixels(rendered)) @ transform
    return corrected.reshape(shape[0], shape[2], shape[3], shape[1]).permute(0, 3, 1, 2).clamp(0.0, 1.0)


def fit_eval_color_calibration(
    render_cfg: dict[str, Any],
    train_renders: torch.Tensor,
    train_targets: torch.Tensor,
) -> dict[str, Any] | None:
    mode = str(render_cfg["eval_color_calibration"])
    if mode == "none":
        return None
    if mode == "train_fit_channel_affine":
        transform = fit_channel_affine(train_renders, train_targets)
    elif mode == "train_fit_rgb_matrix_affine":
        transform = fit_rgb_matrix_affine(train_renders, train_targets)
    else:  # resolve_config validates this, but keep the helper total.
        raise ValueError(f"Unknown eval color calibration mode {mode!r}")
    return {"mode": mode, "transform": transform}


def apply_eval_color_calibration(rendered: torch.Tensor, calibration: dict[str, Any] | None) -> torch.Tensor:
    if calibration is None:
        return rendered
    mode = str(calibration["mode"])
    transform = calibration["transform"]
    if mode == "train_fit_channel_affine":
        return apply_channel_affine(rendered, transform)
    if mode == "train_fit_rgb_matrix_affine":
        return apply_rgb_matrix_affine(rendered, transform)
    raise ValueError(f"Unknown eval color calibration mode {mode!r}")


def frame_index_summary(frame_indices: torch.Tensor | None) -> dict[str, Any] | None:
    if frame_indices is None:
        return None
    values = [int(value) for value in frame_indices.detach().cpu().reshape(-1).tolist()]
    return {
        "count": len(values),
        "unique": sorted(set(values)),
    }


def serialize_eval_color_calibration(
    calibration: dict[str, Any] | None,
    *,
    step: int | None = None,
    train_frame_indices: torch.Tensor | None = None,
    heldout_frame_indices: torch.Tensor | None = None,
) -> dict[str, Any] | None:
    if calibration is None:
        return None
    transform = calibration["transform"]
    payload: dict[str, Any] = {
        "mode": str(calibration["mode"]),
        "transform": transform.detach().cpu().tolist(),
        "fit_scope": "train_render_to_train_target",
        "heldout_blind": True,
    }
    if step is not None:
        payload["step"] = int(step)
    if train_frame_indices is not None:
        payload["train_frame_indices"] = frame_index_summary(train_frame_indices)
    if heldout_frame_indices is not None:
        payload["heldout_frame_indices"] = frame_index_summary(heldout_frame_indices)
    return payload


def exp_scheduled_weight(initial: float, final_multiplier: float, step: int, total_steps: int) -> float:
    initial = float(initial)
    if initial <= 0.0:
        return initial
    final = initial * float(final_multiplier)
    if final <= 0.0:
        return final
    t = min(max(float(step) / max(float(total_steps), 1.0), 0.0), 1.0)
    return float(math.exp(math.log(initial) * (1.0 - t) + math.log(final) * t))


def scheduled_loss_weights(loss_cfg: dict[str, Any], step: int, total_steps: int) -> dict[str, float]:
    def aux_weight(name: str) -> float:
        if int(step) < int(loss_cfg[f"{name}_weight_start_step"]):
            return 0.0
        return exp_scheduled_weight(
            float(loss_cfg[f"{name}_weight"]),
            float(loss_cfg[f"{name}_weight_final_multiplier"]),
            int(step) - int(loss_cfg[f"{name}_weight_start_step"]),
            max(int(total_steps) - int(loss_cfg[f"{name}_weight_start_step"]), 1),
        )

    return {
        "l1_weight": float(loss_cfg["l1_weight"]),
        "mse_weight": float(loss_cfg["mse_weight"]),
        "ssim_weight": float(loss_cfg["ssim_weight"]),
        "radius_l2_weight": float(loss_cfg["radius_l2_weight"]),
        "density_l2_weight": float(loss_cfg["density_l2_weight"]),
        "normal_weight": aux_weight("normal"),
        "normal_map_weight": aux_weight("normal_map"),
        "contribution_weight": aux_weight("contribution"),
        "interpenetration_weight": aux_weight("interpenetration"),
    }


def powerfoam_contribution_loss(alpha: torch.Tensor) -> torch.Tensor:
    if alpha.ndim != 3:
        raise ValueError(f"alpha must have shape [B,H,W], got {tuple(alpha.shape)}.")
    return alpha.mean()


def powerfoam_normal_distance_loss(normal_distance: torch.Tensor) -> torch.Tensor:
    if normal_distance.ndim != 3:
        raise ValueError(f"normal_distance must have shape [B,H,W], got {tuple(normal_distance.shape)}.")
    return normal_distance.mean()


def expand_powerfoam_rays_to_batch(rays: torch.Tensor, batch_size: int) -> torch.Tensor:
    if rays.ndim != 4 or rays.shape[-1] != 6:
        raise ValueError(f"rays must have shape [B,H,W,6], got {tuple(rays.shape)}.")
    if rays.shape[0] == int(batch_size):
        return rays.contiguous()
    if rays.shape[0] == 1:
        return rays.expand(int(batch_size), -1, -1, -1).contiguous()
    raise ValueError(f"Expected {batch_size} ray batches or one shared ray batch, got {rays.shape[0]}.")


def normals_from_ray_depth(depth: torch.Tensor, rays: torch.Tensor, *, eps: float = 1.0e-6) -> tuple[torch.Tensor, torch.Tensor]:
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 3:
        raise ValueError(f"depth must have shape [B,H,W], got {tuple(depth.shape)}.")
    rays = expand_powerfoam_rays_to_batch(rays.to(device=depth.device, dtype=depth.dtype), int(depth.shape[0]))
    if tuple(rays.shape[:3]) != tuple(depth.shape):
        raise ValueError(f"ray/depth shape mismatch: {tuple(rays.shape[:3])} vs {tuple(depth.shape)}.")
    origins = rays[..., :3]
    raw_dirs = rays[..., 3:]
    dir_norm = torch.linalg.vector_norm(raw_dirs, dim=-1, keepdim=True)
    dirs = raw_dirs / dir_norm.clamp_min(float(eps))
    valid = (
        torch.isfinite(depth)
        & (depth > 0.0)
        & torch.isfinite(origins).all(dim=-1)
        & torch.isfinite(raw_dirs).all(dim=-1)
        & (dir_norm[..., 0] > float(eps))
    )
    points = origins + depth[..., None] * dirs
    normals = torch.zeros_like(points)
    mask = torch.zeros_like(depth, dtype=torch.bool)
    if depth.shape[1] < 3 or depth.shape[2] < 3:
        return normals, mask
    dx = points[:, 1:-1, 2:, :] - points[:, 1:-1, :-2, :]
    dy = points[:, 2:, 1:-1, :] - points[:, :-2, 1:-1, :]
    inner = torch.cross(dy, dx, dim=-1)
    inner_norm = torch.linalg.vector_norm(inner, dim=-1, keepdim=True)
    inner_normal = inner / inner_norm.clamp_min(float(eps))
    inner_dirs = dirs[:, 1:-1, 1:-1, :]
    inner_normal = torch.where(
        (inner_normal * inner_dirs).sum(dim=-1, keepdim=True) > 0.0,
        -inner_normal,
        inner_normal,
    )
    inner_mask = (
        valid[:, 1:-1, 1:-1]
        & valid[:, 1:-1, 2:]
        & valid[:, 1:-1, :-2]
        & valid[:, 2:, 1:-1]
        & valid[:, :-2, 1:-1]
        & (inner_norm[..., 0] > float(eps))
        & torch.isfinite(inner_normal).all(dim=-1)
    )
    normals[:, 1:-1, 1:-1, :] = torch.where(inner_mask[..., None], inner_normal, 0.0)
    mask[:, 1:-1, 1:-1] = inner_mask
    return normals, mask


def powerfoam_normal_map_loss(
    rendered_normal: torch.Tensor,
    target_normal: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if rendered_normal.ndim != 4 or rendered_normal.shape[-1] != 3:
        raise ValueError(f"rendered_normal must have shape [B,H,W,3], got {tuple(rendered_normal.shape)}.")
    if target_normal.shape != rendered_normal.shape:
        raise ValueError(f"target_normal shape mismatch: {tuple(target_normal.shape)} vs {tuple(rendered_normal.shape)}.")
    if valid_mask.shape != rendered_normal.shape[:3]:
        raise ValueError(f"valid_mask shape mismatch: {tuple(valid_mask.shape)} vs {tuple(rendered_normal.shape[:3])}.")
    mask = valid_mask.to(device=rendered_normal.device, dtype=rendered_normal.dtype)
    per_pixel = (rendered_normal - target_normal.to(device=rendered_normal.device, dtype=rendered_normal.dtype)).square().sum(dim=-1)
    return (per_pixel * mask).sum() / mask.sum().clamp_min(1.0)


def scheduled_resample_target_cells(
    model_cfg: dict[str, Any],
    *,
    initial_cells: int,
    current_cells: int,
    step: int,
    total_steps: int,
) -> int | None:
    if model_cfg["resample_target_cells"] is not None:
        return int(model_cfg["resample_target_cells"])
    if model_cfg["resample_final_cells"] is None:
        return None

    start = int(model_cfg["resample_from_step"])
    stop = int(model_cfg["resample_until_step"] or max(int(total_steps), start + 1))
    if int(step) < start or int(step) >= stop:
        return int(current_cells)
    if stop - start <= 1:
        return int(model_cfg["resample_final_cells"])

    final_cells = int(model_cfg["resample_final_cells"])
    if int(initial_cells) <= 0:
        raise ValueError("initial_cells must be positive")
    growth = (float(final_cells) / float(initial_cells)) ** (1.0 / float(stop - start - 1))
    return max(1, int(float(initial_cells) * (growth ** float(int(step) - start))))


def should_resample_powerfoam_step(cfg: dict[str, Any], step: int) -> bool:
    return (
        int(cfg["model"]["resample_every"]) > 0
        and int(step) < int(cfg["train"]["steps"])
        and int(step) % int(cfg["model"]["resample_every"]) == 0
    )


def fixed_background_tensor(rendered: torch.Tensor, render_cfg: dict[str, Any]) -> torch.Tensor:
    return torch.tensor(
        render_cfg["background"],
        device=rendered.device,
        dtype=rendered.dtype,
    ).view(1, 3, 1, 1)


def training_background_tensor(rendered: torch.Tensor, render_cfg: dict[str, Any]) -> torch.Tensor:
    if str(render_cfg["background_mode"]) == "random":
        return torch.rand((rendered.shape[0], 3, 1, 1), device=rendered.device, dtype=rendered.dtype)
    return fixed_background_tensor(rendered, render_cfg)


def composite_powerfoam_background(
    rendered: torch.Tensor,
    alpha: torch.Tensor,
    background: torch.Tensor,
) -> torch.Tensor:
    if alpha.ndim != 3:
        raise ValueError(f"alpha must have shape [B,H,W], got {tuple(alpha.shape)}.")
    if rendered.ndim != 4 or rendered.shape[1] != 3:
        raise ValueError(f"rendered must have shape [B,3,H,W], got {tuple(rendered.shape)}.")
    return rendered + (1.0 - alpha).unsqueeze(1) * background


def composite_fixed_background(rendered: torch.Tensor, alpha: torch.Tensor, render_cfg: dict[str, Any]) -> torch.Tensor:
    return composite_powerfoam_background(rendered, alpha, fixed_background_tensor(rendered, render_cfg))


def log_artifacts(
    model: MetalPowerFoamVideo,
    targets: torch.Tensor,
    cfg: dict[str, Any],
    step: int,
    output_dir: Path,
    wandb_run: Any | None,
    *,
    frame_indices: torch.Tensor | None = None,
    rays: torch.Tensor | None = None,
    heldout_targets: torch.Tensor | None = None,
    heldout_frame_indices: torch.Tensor | None = None,
    heldout_rays: torch.Tensor | None = None,
) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    if frame_indices is None:
        frame_indices = torch.arange(targets.size(0), device=device, dtype=torch.long)
    else:
        frame_indices = frame_indices.to(device=device, dtype=torch.long)
    renders, alphas = render_samples(
        model,
        frame_indices,
        batch_size=max(1, int(cfg["train"]["frames_per_step"])),
        rays=rays,
    )
    renders = composite_fixed_background(renders, alphas, cfg["render"])
    targets_cpu = targets.detach().cpu()
    calibration = fit_eval_color_calibration(cfg["render"], renders, targets_cpu)
    raw_renders = renders
    renders = apply_eval_color_calibration(raw_renders, calibration)
    metrics = reconstruction_eval_metrics(renders, targets_cpu, cfg, prefix="eval")
    if calibration is not None:
        metrics.update(reconstruction_eval_metrics(raw_renders, targets_cpu, cfg, prefix="uncalibrated_eval"))
    metrics.update(model.aux_metrics(frame_indices, targets, rays=rays))
    heldout_renders = None
    heldout_alphas = None
    if heldout_targets is not None and heldout_frame_indices is not None:
        heldout_renders, heldout_alphas = render_samples(
            model,
            heldout_frame_indices,
            batch_size=max(1, int(cfg["train"]["frames_per_step"])),
            rays=heldout_rays,
        )
        heldout_renders = composite_fixed_background(heldout_renders, heldout_alphas, cfg["render"])
        heldout_targets_cpu = heldout_targets.detach().cpu()
        raw_heldout_renders = heldout_renders
        heldout_renders = apply_eval_color_calibration(raw_heldout_renders, calibration)
        metrics.update(reconstruction_eval_metrics(heldout_renders, heldout_targets_cpu, cfg, prefix="heldout_eval"))
        if calibration is not None:
            metrics.update(
                reconstruction_eval_metrics(
                    raw_heldout_renders,
                    heldout_targets_cpu,
                    cfg,
                    prefix="uncalibrated_heldout_eval",
                )
            )
    metrics.update(model.parameter_drift_metrics())
    if calibration is not None:
        (output_dir / f"eval_color_calibration_step_{step:04d}.json").write_text(
            json.dumps(
                serialize_eval_color_calibration(
                    calibration,
                    step=step,
                    train_frame_indices=frame_indices,
                    heldout_frame_indices=heldout_frame_indices,
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    preview = torch.cat([targets_cpu[0], renders[0], alphas[0].unsqueeze(0).repeat(3, 1, 1)], dim=-1)
    save_png(output_dir / f"preview_step_{step:04d}.png", preview)
    if heldout_renders is not None and heldout_alphas is not None and heldout_targets is not None:
        heldout_preview = torch.cat(
            [
                heldout_targets.detach().cpu()[0],
                heldout_renders[0],
                heldout_alphas[0].unsqueeze(0).repeat(3, 1, 1),
            ],
            dim=-1,
        )
        save_png(output_dir / f"heldout_preview_step_{step:04d}.png", heldout_preview)
    if should_log_video(cfg, step):
        side_by_side = torch.cat([targets_cpu, renders], dim=-1)
        fps = float(cfg.get("video_fps", 4.0))
        save_mp4(output_dir / f"render_step_{step:04d}.mp4", renders, fps=fps)
        save_mp4(output_dir / f"side_by_side_step_{step:04d}.mp4", side_by_side, fps=fps)
        if heldout_renders is not None and heldout_targets is not None:
            heldout_side_by_side = torch.cat([heldout_targets.detach().cpu(), heldout_renders], dim=-1)
            save_mp4(output_dir / f"heldout_render_step_{step:04d}.mp4", heldout_renders, fps=fps)
            save_mp4(output_dir / f"heldout_side_by_side_step_{step:04d}.mp4", heldout_side_by_side, fps=fps)
    if wandb_run is not None:
        fps = float(cfg.get("video_fps", 4.0))
        payload: dict[str, Any] = {
            "Eval/L1": metrics["eval_l1"],
            "Eval/MSE": metrics["eval_mse"],
            "Eval/PSNR": metrics["eval_psnr"],
            "Eval/SSIM": metrics["eval_ssim"],
            "State/MeanCenterDelta": metrics["state_mean_center_delta"],
            "State/P95CenterDelta": metrics["state_p95_center_delta"],
            "State/MaxCenterDelta": metrics["state_max_center_delta"],
            "State/MeanXYDelta": metrics["state_mean_xy_delta"],
            "State/MeanZDelta": metrics["state_mean_z_delta"],
            "State/MeanRadiusDelta": metrics["state_mean_radius_delta"],
            "State/MeanDensityDelta": metrics["state_mean_density_delta"],
            "State/MeanFeatureDelta": metrics["state_mean_feature_delta"],
            "State/CellCount": metrics["state_cell_count"],
            "Preview": make_preview_image(targets_cpu[0], renders[0], caption=f"step {step}: GT | render"),
        }
        if "heldout_eval_l1" in metrics:
            payload["Heldout/EvalL1"] = metrics["heldout_eval_l1"]
            payload["Heldout/EvalMSE"] = metrics["heldout_eval_mse"]
            payload["Heldout/EvalPSNR"] = metrics["heldout_eval_psnr"]
            payload["Heldout/EvalSSIM"] = metrics["heldout_eval_ssim"]
        if should_log_video(cfg, step):
            payload.update(build_validation_video_payload(renders, targets_cpu, fps))
            payload["GT_Video"] = make_wandb_video(targets_cpu, fps)
            payload["Alpha_Video"] = make_wandb_video(alphas.unsqueeze(1).repeat(1, 3, 1, 1), fps)
        if "state_mean_normal_delta" in metrics:
            payload["State/MeanNormalDelta"] = metrics["state_mean_normal_delta"]
            payload["State/MeanNormalZ"] = metrics["state_mean_normal_z"]
        if "state_mean_texel_site_delta" in metrics:
            payload["State/MeanTexelSiteDelta"] = metrics["state_mean_texel_site_delta"]
        if "state_mean_texel_height_delta" in metrics:
            payload["State/MeanTexelHeightDelta"] = metrics["state_mean_texel_height_delta"]
        if "state_mean_texel_sv_axis_delta" in metrics:
            payload["State/MeanTexelSvAxisDelta"] = metrics["state_mean_texel_sv_axis_delta"]
        if "state_mean_texel_sv_rgb_delta" in metrics:
            payload["State/MeanTexelSvRgbDelta"] = metrics["state_mean_texel_sv_rgb_delta"]
        if "state_mean_quaternion_delta" in metrics:
            payload["State/MeanQuaternionDelta"] = metrics["state_mean_quaternion_delta"]
        for key in (
            "aux_mean_contrib",
            "aux_max_contrib",
            "aux_mean_point_error",
            "aux_max_point_error",
            "aux_mean_contrib_ema",
            "aux_mean_point_error_ema",
            "aux_visible_fraction",
            "aux_mean_normal_distance",
            "aux_mean_normal_norm",
            "aux_median_depth_valid_fraction",
            "aux_mean_median_depth",
        ):
            if key in metrics:
                payload[f"Aux/{key.removeprefix('aux_')}"] = metrics[key]
        wandb_run.log(payload, step=step)
    model.train()
    return metrics


def select_best_metric(metrics: dict[str, float]) -> tuple[str, float]:
    if "heldout_eval_psnr" in metrics:
        return "heldout_eval_psnr", float(metrics["heldout_eval_psnr"])
    return "eval_psnr", float(metrics["eval_psnr"])


def save_powerfoam_checkpoint(
    path: Path,
    model: MetalPowerFoamVideo,
    cfg: dict[str, Any],
    *,
    step: int,
    metrics: dict[str, float] | None = None,
    best_metric_name: str | None = None,
    best_metric_value: float | None = None,
) -> None:
    atomic_torch_save(
        {
            "model": model.state_dict(),
            "config": serialize_config_value(cfg),
            "step": int(step),
            "metrics": metrics or {},
            "best_metric_name": best_metric_name,
            "best_metric_value": best_metric_value,
        },
        path,
    )


def maybe_save_best_powerfoam_checkpoint(
    model: MetalPowerFoamVideo,
    cfg: dict[str, Any],
    output_dir: Path,
    *,
    step: int,
    metrics: dict[str, float],
    best_metric_value: float | None,
) -> float:
    metric_name, metric_value = select_best_metric(metrics)
    if best_metric_value is not None and metric_value <= best_metric_value:
        return best_metric_value
    save_powerfoam_checkpoint(
        output_dir / "checkpoint_best.pt",
        model,
        cfg,
        step=step,
        metrics=metrics,
        best_metric_name=metric_name,
        best_metric_value=metric_value,
    )
    summary = {
        "step": int(step),
        "best_metric_name": metric_name,
        "best_metric_value": metric_value,
        "metrics": metrics,
    }
    (output_dir / "best_metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return metric_value


def load_powerfoam_training_data(cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    render_size = int(cfg["render"]["render_size"])
    frame_source = str(cfg["data"]["frame_source"])
    if frame_source == "multicam_val":
        bundle = load_multicam_video_bundle(
            data_cfg=cfg["data"],
            camera_cfg=cfg["camera"],
            target_size=render_size,
            device=device,
        )
        train_cameras = cameras_from_K_w2c(
            bundle.train_K,
            bundle.train_w2c,
            lens_models=bundle.train_lens_models,
            distortions=bundle.train_distortions,
        )
        train_rays = powerfoam_rays_from_camera_grid(
            train_cameras,
            height=render_size,
            width=render_size,
            device=device,
        )
        targets, sample_frame_indices, sample_rays = flatten_multiview_powerfoam_samples(
            bundle.train_frames.to(device=device, dtype=torch.float32),
            train_rays,
        )

        heldout_targets = None
        heldout_frame_indices = None
        heldout_rays = None
        if bundle.heldout_frames is not None and bundle.heldout_K is not None and bundle.heldout_w2c is not None:
            heldout_camera_grid = heldout_cameras_from_K_w2c(
                bundle.heldout_K,
                bundle.heldout_w2c,
                lens_models=bundle.heldout_lens_models,
                distortions=bundle.heldout_distortions,
            )
            heldout_ray_grid = powerfoam_rays_from_camera_grid(
                heldout_camera_grid,
                height=render_size,
                width=render_size,
                device=device,
            )
            heldout_targets, heldout_frame_indices, heldout_rays = flatten_multiview_powerfoam_samples(
                bundle.heldout_frames.to(device=device, dtype=torch.float32),
                heldout_ray_grid,
            )

        return {
            "targets": targets,
            "sample_frame_indices": sample_frame_indices,
            "sample_rays": sample_rays,
            "heldout_targets": heldout_targets,
            "heldout_frame_indices": heldout_frame_indices,
            "heldout_rays": heldout_rays,
            "init_frames": bundle.condition_sequence.frames.detach().cpu(),
            "frame_count": bundle.frame_count,
            "video_fps": float(bundle.condition_sequence.video_fps),
            "source_label": str(bundle.metadata.get("sample_id")) if bundle.metadata else "multicam_val",
            "train_views": bundle.train_camera_names,
            "heldout_views": bundle.heldout_camera_names or [],
            "pose_source": bundle.pose_source,
            "world_to_model": None
            if bundle.anchor_c2w is None
            else torch.linalg.inv(bundle.anchor_c2w.detach().to(device="cpu", dtype=torch.float32)),
            "point_cloud_visibility_train_K": bundle.train_K.detach().to(device="cpu", dtype=torch.float32),
            "point_cloud_visibility_train_w2c": bundle.train_w2c.detach().to(device="cpu", dtype=torch.float32),
            "point_cloud_visibility_train_lens_models": bundle.train_lens_models,
            "point_cloud_visibility_train_distortions": None
            if bundle.train_distortions is None
            else bundle.train_distortions.detach().to(device="cpu", dtype=torch.float32),
        }

    if cfg["data"]["video_path"] is None:
        raise ValueError("data.video_path is required unless data.frame_source is 'multicam_val'.")
    sequence = load_video_sequence(
        cfg["data"]["video_path"],
        target_size=render_size,
        max_frames=int(cfg["data"]["max_frames"]),
        frame_source=frame_source,
    )
    targets = sequence.frames.to(device=device, dtype=torch.float32)
    return {
        "targets": targets,
        "sample_frame_indices": torch.arange(targets.size(0), device=device, dtype=torch.long),
        "sample_rays": None,
        "heldout_targets": None,
        "heldout_frame_indices": None,
        "heldout_rays": None,
        "init_frames": targets.detach().cpu(),
        "frame_count": int(targets.size(0)),
        "video_fps": float(sequence.video_fps),
        "source_label": str(cfg["data"]["video_path"]),
        "train_views": [],
        "heldout_views": [],
        "pose_source": None,
        "world_to_model": None,
        "point_cloud_visibility_train_K": None,
        "point_cloud_visibility_train_w2c": None,
        "point_cloud_visibility_train_lens_models": None,
        "point_cloud_visibility_train_distortions": None,
    }


def run_training(config: dict[str, Any]) -> None:
    cfg = resolve_config(config)
    torch.manual_seed(int(cfg["train"]["seed"]))
    device = resolve_device(str(cfg["train"]["device"]))
    if device.type != "mps" or not torch.backends.mps.is_available():
        raise RuntimeError("powerfoam_metal requires torch MPS")

    output_dir: Path = cfg["logging"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(json.dumps(serialize_config_value(cfg), indent=2) + "\n")

    training_data = load_powerfoam_training_data(cfg, device)
    targets = training_data["targets"]
    sample_frame_indices = training_data["sample_frame_indices"]
    sample_rays = training_data["sample_rays"]
    heldout_targets = training_data["heldout_targets"]
    heldout_frame_indices = training_data["heldout_frame_indices"]
    heldout_rays = training_data["heldout_rays"]
    cfg["video_fps"] = float(training_data["video_fps"])
    wandb_run = init_wandb_run(cfg)
    point_cloud_init = None
    if cfg["model"]["init_point_cloud_path"] is not None:
        point_cloud_coordinate_frame = str(cfg["model"]["init_point_cloud_coordinate_frame"])
        point_transform = None
        if point_cloud_coordinate_frame == "multicam_world":
            point_transform = training_data.get("world_to_model")
            if point_transform is None:
                raise ValueError(
                    "model.init_point_cloud_coordinate_frame='multicam_world' requires multicam camera metadata."
                )
        point_cloud_init = load_powerfoam_point_cloud_initialization(
            path=cfg["model"]["init_point_cloud_path"],
            frame_count=int(training_data["frame_count"]),
            cell_count=int(cfg["model"]["cells"]),
            xy_extent=float(cfg["model"]["xy_extent"]),
            z_min=float(cfg["model"]["z_min"]),
            z_max=float(cfg["model"]["z_max"]),
            normalize_mode=str(cfg["model"]["init_point_cloud_normalize"]),
            coordinate_frame=point_cloud_coordinate_frame,
            point_transform=point_transform,
            visibility_filter=str(cfg["model"]["init_point_cloud_visibility_filter"]),
            min_visible_train_views=int(cfg["model"]["init_point_cloud_min_visible_train_views"]),
            visibility_train_K=training_data.get("point_cloud_visibility_train_K"),
            visibility_train_w2c=training_data.get("point_cloud_visibility_train_w2c"),
            visibility_train_lens_models=training_data.get("point_cloud_visibility_train_lens_models"),
            visibility_train_distortions=training_data.get("point_cloud_visibility_train_distortions"),
            visibility_render_size=int(cfg["render"]["render_size"]),
            sample_mode=str(cfg["model"]["init_point_cloud_sample_mode"]),
            duplicate_jitter=float(cfg["model"]["init_point_cloud_duplicate_jitter"]),
            seed=int(cfg["train"]["seed"]),
        )

    model = MetalPowerFoamVideo(
        frame_count=int(training_data["frame_count"]),
        cell_count=int(cfg["model"]["cells"]),
        render_size=int(cfg["render"]["render_size"]),
        fov_degrees=float(cfg["render"]["fov_degrees"]),
        neighbor_count=int(cfg["model"]["neighbor_count"]),
        adjacency_mode=str(cfg["model"]["adjacency_mode"]),
        xy_extent=float(cfg["model"]["xy_extent"]),
        z_min=float(cfg["model"]["z_min"]),
        z_max=float(cfg["model"]["z_max"]),
        radius_init=float(cfg["model"]["radius_init"]),
        radius_min=float(cfg["model"]["radius_min"]),
        radius_scale=float(cfg["model"]["radius_scale"]),
        density_init=float(cfg["model"]["density_init"]),
        feature_mode=str(cfg["model"]["feature_mode"]),
        linear_coeff_init=float(cfg["model"]["linear_coeff_init"]),
        linear_coeff_scale=float(cfg["model"]["linear_coeff_scale"]),
        normal_init_jitter=float(cfg["model"]["normal_init_jitter"]),
        num_texel_sites=int(cfg["model"]["num_texel_sites"]),
        texel_site_scale=float(cfg["model"]["texel_site_scale"]),
        texel_height_scale=float(cfg["model"]["texel_height_scale"]),
        sv_dof=int(cfg["model"]["sv_dof"]),
        sv_axis_init=float(cfg["model"]["sv_axis_init"]),
        sv_axis_init_jitter=float(cfg["model"]["sv_axis_init_jitter"]),
        sv_rgb_init_jitter=float(cfg["model"]["sv_rgb_init_jitter"]),
        color_init_mode=str(cfg["model"]["color_init_mode"]),
        seed=int(cfg["train"]["seed"]),
        init_frames=training_data["init_frames"] if bool(cfg["model"]["init_from_video"]) else None,
        init_points=None if point_cloud_init is None else point_cloud_init.points,
        init_colors=None if point_cloud_init is None else point_cloud_init.colors,
        image_init_depth=None if cfg["model"]["image_init_depth"] is None else float(cfg["model"]["image_init_depth"]),
        image_init_jitter=float(cfg["model"]["image_init_jitter"]),
        raster_config=make_raster_config(cfg["render"]),
        use_raytrace=bool(cfg["render"]["use_raytrace"]),
    ).to(device)
    optimizer = torch.optim.Adam(model.optimizer_param_groups(cfg["train"]), lr=float(cfg["train"]["lr"]))
    initial_cell_count = int(model.contrib_ema.shape[1])
    adjacency_stats = model.adjacency_diagnostics()

    print(
        {
            "arch": "powerfoam_metal",
            "device": str(device),
            "source": str(training_data["source_label"]),
            "frame_source": str(cfg["data"]["frame_source"]),
            "frames": int(training_data["frame_count"]),
            "samples": int(targets.size(0)),
            "train_views": training_data["train_views"],
            "heldout_views": training_data["heldout_views"],
            "pose_source": training_data["pose_source"],
            "render_size": int(cfg["render"]["render_size"]),
            "cells": int(cfg["model"]["cells"]),
            "neighbors": int(cfg["model"]["neighbor_count"]),
            "adjacency_mode": str(cfg["model"]["adjacency_mode"]),
            "adjacency_avg_degree": adjacency_stats["adjacency_avg_degree"],
            "adjacency_max_degree": adjacency_stats["adjacency_max_degree"],
            "adjacency_required_overlap_edges": adjacency_stats["adjacency_required_overlap_edges"],
            "adjacency_missing_overlap_edges": adjacency_stats["adjacency_missing_overlap_edges"],
            "feature_mode": str(cfg["model"]["feature_mode"]),
            "render_backend": (
                "raytrace"
                if bool(cfg["render"]["use_raytrace"])
                else ("tiled" if bool(cfg["render"]["use_tiled"]) else "streaming")
            ),
            "color_init_mode": str(cfg["model"]["color_init_mode"]),
            "background_mode": str(cfg["render"]["background_mode"]),
            "lr_schedule": str(cfg["train"]["lr_schedule"]),
            "init_point_cloud": None if point_cloud_init is None else str(point_cloud_init.source_path),
            "init_point_cloud_source_count": None if point_cloud_init is None else point_cloud_init.source_count,
            "init_point_cloud_normalize": None if point_cloud_init is None else point_cloud_init.normalize_mode,
            "init_point_cloud_coordinate_frame": None
            if point_cloud_init is None
            else point_cloud_init.coordinate_frame,
            "init_point_cloud_visibility_filter": None
            if point_cloud_init is None
            else point_cloud_init.visibility_filter,
            "init_point_cloud_sample_mode": None if point_cloud_init is None else point_cloud_init.sample_mode,
            "init_point_cloud_filtered_count": None if point_cloud_init is None else point_cloud_init.filtered_count,
            "steps": int(cfg["train"]["steps"]),
        }
    )
    if wandb_run is not None:
        wandb_run.log({f"adjacency/{key}": value for key, value in adjacency_stats.items()}, step=0)
    initial_metrics = log_artifacts(
        model,
        targets,
        cfg,
        0,
        output_dir,
        wandb_run,
        frame_indices=sample_frame_indices,
        rays=sample_rays,
        heldout_targets=heldout_targets,
        heldout_frame_indices=heldout_frame_indices,
        heldout_rays=heldout_rays,
    )
    best_metric_value: float | None = maybe_save_best_powerfoam_checkpoint(
        model,
        cfg,
        output_dir,
        step=0,
        metrics=initial_metrics,
        best_metric_value=None,
    )
    append_jsonl(output_dir / "eval_metrics_history.jsonl", {"step": 0, "metrics": initial_metrics})
    last_artifact_step = 0
    last_artifact_metrics = dict(initial_metrics)
    print({"step": 0, **initial_metrics})

    start_time = time.perf_counter()
    progress = trange(1, int(cfg["train"]["steps"]) + 1, desc="powerfoam_metal")
    for step in progress:
        lr_by_group = update_powerfoam_learning_rates(
            optimizer,
            cfg["train"],
            step=step - 1,
            total_steps=int(cfg["train"]["steps"]),
        )
        sample_indices = torch.randint(0, targets.size(0), (int(cfg["train"]["frames_per_step"]),), device=device)
        frame_indices = sample_frame_indices[sample_indices]
        target = targets[sample_indices]
        batch_rays = None if sample_rays is None else sample_rays[sample_indices]
        loss_weights = scheduled_loss_weights(cfg["losses"], step - 1, int(cfg["train"]["steps"]))
        need_normal_distance = loss_weights["normal_weight"] > 0.0
        need_normal_map = loss_weights["normal_map_weight"] > 0.0
        if need_normal_distance or need_normal_map:
            render_result = model(
                frame_indices,
                rays=batch_rays,
                return_normal_distance=need_normal_distance,
                return_rendered_normal=need_normal_map,
            )
            rendered, alpha = render_result[:2]
            cursor = 2
            normal_distance = render_result[cursor] if need_normal_distance else None
            cursor += 1 if need_normal_distance else 0
            rendered_normal = render_result[cursor] if need_normal_map else None
        else:
            rendered, alpha = model(frame_indices, rays=batch_rays)
            normal_distance = None
            rendered_normal = None
        rendered = composite_powerfoam_background(
            rendered,
            alpha,
            training_background_tensor(rendered, cfg["render"]),
        )
        l1 = F.l1_loss(rendered, target)
        mse = F.mse_loss(rendered, target)
        ssim_loss = (
            powerfoam_ssim_loss(rendered, target, cfg["losses"])
            if loss_weights["ssim_weight"] > 0.0
            else rendered.new_zeros(())
        )
        _, radii, densities, _, _ = model.decoded_parameters()
        radius_l2 = radii.square().mean()
        density_l2 = densities.square().mean()
        normal_loss = (
            powerfoam_normal_distance_loss(normal_distance)
            if normal_distance is not None and loss_weights["normal_weight"] > 0.0
            else rendered.new_zeros(())
        )
        normal_map_valid_fraction = rendered.new_zeros(())
        if rendered_normal is not None and loss_weights["normal_map_weight"] > 0.0:
            aux = model.height_sv_aux_batch(frame_indices, target, batch_rays)
            if aux is None:
                raise RuntimeError("normal-map supervision requires a height+SV PowerFoam aux path")
            normal_rays = (
                model.rays.to(device=rendered.device, dtype=rendered.dtype)
                if batch_rays is None
                else batch_rays.to(device=rendered.device, dtype=rendered.dtype)
            )
            target_normal, normal_map_mask = normals_from_ray_depth(aux.median_depth, normal_rays)
            normal_map_mask = normal_map_mask & (alpha.detach() >= float(cfg["losses"]["normal_map_min_alpha"]))
            normal_map_loss = powerfoam_normal_map_loss(rendered_normal, target_normal.detach(), normal_map_mask)
            normal_map_valid_fraction = normal_map_mask.to(dtype=rendered.dtype).mean()
        else:
            normal_map_loss = rendered.new_zeros(())
        contribution_loss = (
            powerfoam_contribution_loss(alpha)
            if loss_weights["contribution_weight"] > 0.0
            else rendered.new_zeros(())
        )
        interpenetration_loss = (
            model.interpenetration_loss(frame_indices)
            if loss_weights["interpenetration_weight"] > 0.0
            else rendered.new_zeros(())
        )
        loss = (
            loss_weights["l1_weight"] * l1
            + loss_weights["mse_weight"] * mse
            + loss_weights["ssim_weight"] * ssim_loss
            + loss_weights["radius_l2_weight"] * radius_l2
            + loss_weights["density_l2_weight"] * density_l2
            + loss_weights["normal_weight"] * normal_loss
            + loss_weights["normal_map_weight"] * normal_map_loss
            + loss_weights["contribution_weight"] * contribution_loss
            + loss_weights["interpenetration_weight"] * interpenetration_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        progress.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}", l1=f"{float(l1.detach().cpu()):.4f}")
        if step % int(cfg["logging"]["log_every"]) == 0:
            elapsed = time.perf_counter() - start_time
            train_metrics = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "l1": float(l1.detach().cpu()),
                "mse": float(mse.detach().cpu()),
                "ssim_loss": float(ssim_loss.detach().cpu()),
                "normal_loss": float(normal_loss.detach().cpu()),
                "normal_weight": loss_weights["normal_weight"],
                "normal_map_loss": float(normal_map_loss.detach().cpu()),
                "normal_map_weight": loss_weights["normal_map_weight"],
                "normal_map_valid_fraction": float(normal_map_valid_fraction.detach().cpu()),
                "contribution_loss": float(contribution_loss.detach().cpu()),
                "contribution_weight": loss_weights["contribution_weight"],
                "interpenetration_loss": float(interpenetration_loss.detach().cpu()),
                "interpenetration_weight": loss_weights["interpenetration_weight"],
                "elapsed_s": elapsed,
            }
            for name, value in lr_by_group.items():
                train_metrics[f"lr_{name}"] = value
            append_jsonl(output_dir / "train_metrics_history.jsonl", train_metrics)
            print(train_metrics)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "Train/Loss": train_metrics["loss"],
                        "Train/L1": train_metrics["l1"],
                        "Train/MSE": train_metrics["mse"],
                        "Train/SSIMLoss": train_metrics["ssim_loss"],
                        "Train/NormalLoss": train_metrics["normal_loss"],
                        "Train/NormalWeight": train_metrics["normal_weight"],
                        "Train/NormalMapLoss": train_metrics["normal_map_loss"],
                        "Train/NormalMapWeight": train_metrics["normal_map_weight"],
                        "Train/NormalMapValidFraction": train_metrics["normal_map_valid_fraction"],
                        "Train/ContributionLoss": train_metrics["contribution_loss"],
                        "Train/ContributionWeight": train_metrics["contribution_weight"],
                        "Train/InterpenetrationLoss": train_metrics["interpenetration_loss"],
                        "Train/InterpenetrationWeight": train_metrics["interpenetration_weight"],
                        "Timing/ElapsedSeconds": elapsed,
                        **{f"LR/{name}": value for name, value in lr_by_group.items()},
                    },
                    step=step,
                )
        logged_artifacts = False
        if step % int(cfg["logging"]["image_log_every"]) == 0 or (
            bool(cfg["logging"]["always_log_last_step"]) and step == int(cfg["train"]["steps"])
        ):
            metrics = log_artifacts(
                model,
                targets,
                cfg,
                step,
                output_dir,
                wandb_run,
                frame_indices=sample_frame_indices,
                rays=sample_rays,
                heldout_targets=heldout_targets,
                heldout_frame_indices=heldout_frame_indices,
                heldout_rays=heldout_rays,
            )
            best_metric_value = maybe_save_best_powerfoam_checkpoint(
                model,
                cfg,
                output_dir,
                step=step,
                metrics=metrics,
                best_metric_value=best_metric_value,
            )
            append_jsonl(output_dir / "eval_metrics_history.jsonl", {"step": int(step), "metrics": metrics})
            last_artifact_step = int(step)
            last_artifact_metrics = dict(metrics)
            logged_artifacts = True
            print({"step": step, **metrics})
        if should_resample_powerfoam_step(cfg, step):
            if not logged_artifacts:
                model.aux_metrics(frame_indices, target, rays=batch_rays)
            target_cells = scheduled_resample_target_cells(
                cfg["model"],
                initial_cells=initial_cell_count,
                current_cells=int(model.contrib_ema.shape[1]),
                step=step,
                total_steps=int(cfg["train"]["steps"]),
            )
            resample_metrics = model.resample_from_ema(
                optimizer,
                target_cells=target_cells,
                perturb_scale=float(cfg["model"]["resample_perturb_scale"]),
            )
            print({"step": step, **resample_metrics})
            if wandb_run is not None:
                wandb_run.log(
                    {f"Resample/{key.removeprefix('resample_')}": value for key, value in resample_metrics.items()},
                    step=step,
                )

    final_step = int(cfg["train"]["steps"])
    save_powerfoam_checkpoint(
        output_dir / "checkpoint_final.pt",
        model,
        cfg,
        step=final_step,
        metrics=last_artifact_metrics if last_artifact_step == final_step else None,
    )
    if wandb_run is not None:
        wandb_run.finish()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: PYTHONPATH=src/train uv run python src/train/train_powerfoam_metal.py <config.jsonc>")
    run_training(load_config_file(sys.argv[1]))


if __name__ == "__main__":
    main()
