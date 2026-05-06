from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import trange

from colorize import FeatureToColor
from config_utils import apply_defaults, load_config_file, resolved_config, serialize_config_value
from powerfoam_direct import (
    POWERFOAM_SOFTPLUS_BETA,
    PowerFoamInitialization,
    camera_facing_quaternion,
    estimate_knn_radii,
    initialize_full_powerfoam_from_video,
    initialize_random_full_powerfoam,
    inverse_softplus,
    logit_clamped,
    make_image_init_uv,
)
from powerfoam_implicit_camera import PowerFoamImplicitCameraDecoder
from sequence_data import load_video_sequence
from train_logging import build_validation_video_payload, make_preview_image, make_wandb_video
from train_powerfoam_metal import (
    build_csr_adjacency,
    init_wandb_run,
    make_pinhole_rays,
    orthonormal_surface_frame,
    should_log_video,
    stable_tangent_from_normals,
)
from video_io import save_mp4, save_png

ROOT = Path(__file__).resolve().parents[2]
DYNAMIC_POWERFOAM_METAL_ROOT = ROOT / "third_party" / "dynamic-powerfoam-metal"
if str(DYNAMIC_POWERFOAM_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(DYNAMIC_POWERFOAM_METAL_ROOT))

from torch_dynamic_powerfoam_metal import (  # noqa: E402
    FoamRasterConfig,
    rasterize_power_foam_oriented_texel_surface,
)


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
    "orbit_yaw_start_degrees": 0.0,
    "orbit_yaw_end_degrees": 0.0,
    "orbit_pitch_degrees": 0.0,
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
COLORIZE_DEFAULTS = {
    "hidden_dim": None,
    "activation": "sigmoid",
    "pre_norm": False,
    "weight_init": "kaiming",
    "weight_init_gain": 1.0,
    "view_condition": "none",
    "detach_view_condition": True,
    "init_rgb_identity": True,
}


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
    if str(cfg["colorize"]["view_condition"]) != "none":
        raise ValueError("dynamic_powerfoam_metal colorizer currently supports colorize.view_condition='none' only")
    return cfg


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(value)


def make_raster_config(render_cfg: dict[str, Any]) -> FoamRasterConfig:
    return FoamRasterConfig(
        near_plane=float(render_cfg["near_plane"]),
        alpha_threshold=float(render_cfg["alpha_threshold"]),
        transmittance_threshold=float(render_cfg["transmittance_threshold"]),
        max_alpha=float(render_cfg["max_alpha"]),
        eps=float(render_cfg["eps"]),
        texel_temperature=float(render_cfg["texel_temperature"]),
    )


def build_camera_decoder(cfg: dict[str, Any], *, frame_count: int) -> PowerFoamImplicitCameraDecoder | None:
    camera_cfg = cfg["camera"]
    if not bool(camera_cfg["enabled"]):
        return None
    base_fov = cfg["render"]["fov_degrees"] if camera_cfg["base_fov_degrees"] is None else camera_cfg["base_fov_degrees"]
    basis_count = cfg["model"]["time_basis_count"] if camera_cfg["time_basis_count"] is None else camera_cfg["time_basis_count"]
    sigma_scale = (
        cfg["model"]["time_basis_sigma_scale"]
        if camera_cfg["time_basis_sigma_scale"] is None
        else camera_cfg["time_basis_sigma_scale"]
    )
    max_translation = (
        float(camera_cfg["base_radius"]) * float(camera_cfg["max_translation_ratio"])
        if camera_cfg["max_translation"] is None
        else float(camera_cfg["max_translation"])
    )
    return PowerFoamImplicitCameraDecoder(
        frame_count=int(frame_count),
        image_size=int(cfg["render"]["render_size"]),
        fov_degrees=float(base_fov),
        base_radius=float(camera_cfg["base_radius"]),
        token_dim=int(camera_cfg["token_dim"]),
        hidden_dim=int(camera_cfg["hidden_dim"]),
        time_basis_count=int(basis_count),
        time_basis_sigma_scale=float(sigma_scale),
        token_init_std=float(camera_cfg["token_init_std"]),
        max_rotation_degrees=float(camera_cfg["max_rotation_degrees"]),
        max_translation=max_translation,
        base_position=camera_cfg["base_position"],
        look_at=camera_cfg["look_at"],
        up=camera_cfg["up"],
        base_path_mode=str(camera_cfg["base_path_mode"]),
        orbit_yaw_start_degrees=float(camera_cfg["orbit_yaw_start_degrees"]),
        orbit_yaw_end_degrees=float(camera_cfg["orbit_yaw_end_degrees"]),
        orbit_pitch_degrees=float(camera_cfg["orbit_pitch_degrees"]),
        lens_model=str(camera_cfg["lens_model"]),  # type: ignore[arg-type]
        distortion=camera_cfg["distortion"],
    )


def camera_param_group(
    camera_decoder: PowerFoamImplicitCameraDecoder | None,
    train_cfg: dict[str, Any],
) -> dict[str, object] | None:
    if camera_decoder is None:
        return None
    return {
        "params": list(camera_decoder.parameters()),
        "lr": float(train_cfg["camera_lr_multiplier"]) * float(train_cfg["lr"]),
        "name": "implicit_camera",
    }


def camera_regularization(
    camera_decoder: PowerFoamImplicitCameraDecoder | None,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
    if camera_decoder is None:
        return None, {}
    terms = camera_decoder.regularization_terms()
    motion = terms["camera_rotation_l2"] + terms["camera_translation_l2"]
    loss = (
        float(loss_cfg["camera_motion_weight"]) * motion
        + float(loss_cfg["camera_temporal_weight"]) * terms["camera_temporal_l2"]
        + float(loss_cfg["camera_global_weight"]) * terms["camera_global_l2"]
    )
    return loss, terms


def compact_camera_metrics(camera_decoder: PowerFoamImplicitCameraDecoder | None) -> dict[str, float]:
    if camera_decoder is None:
        return {}
    state = camera_decoder.camera_state()
    c2w = camera_decoder.camera_to_world_matrices()
    base = camera_decoder.base_camera_to_world_matrices(device=c2w.device, dtype=c2w.dtype)
    origin_delta = torch.linalg.vector_norm(c2w[:, :3, 3] - base[:, :3, 3], dim=-1)
    forward_delta = torch.linalg.vector_norm(c2w[:, :3, 2] - base[:, :3, 2], dim=-1)
    return {
        "state_camera_fov_degrees": float(state.fov_degrees.detach().cpu()),
        "state_camera_radius": float(state.radius.detach().cpu()),
        "state_camera_rotation_delta_mean_degrees": float(
            torch.rad2deg(torch.linalg.norm(state.rotation_delta, dim=-1)).mean().detach().cpu()
        ),
        "state_camera_translation_delta_mean": float(
            torch.linalg.norm(state.translation_delta, dim=-1).mean().detach().cpu()
        ),
        "state_camera_origin_delta_mean": float(origin_delta.mean().detach().cpu()),
        "state_camera_forward_delta_mean": float(forward_delta.mean().detach().cpu()),
        "state_camera_global_residual_l2": float(state.global_residuals.square().mean().detach().cpu()),
    }


def decoded_powerfoam_rays(
    camera_decoder: PowerFoamImplicitCameraDecoder | None,
    fixed_rays: torch.Tensor,
    frame_indices: torch.Tensor,
    *,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if camera_decoder is None:
        return fixed_rays.to(device=device, dtype=dtype)
    origins, directions = camera_decoder.rays(
        height=height,
        width=width,
        frame_indices=frame_indices,
        dtype=dtype,
    )
    return torch.cat([origins, directions], dim=-1).to(device=device, dtype=dtype).contiguous()


def transform_powerfoam_frame_to_camera(
    points: torch.Tensor,
    normals: torch.Tensor,
    tangents: torch.Tensor,
    bitangents: torch.Tensor,
    camera_to_world: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    world_to_camera = torch.linalg.inv(camera_to_world.to(device=points.device, dtype=points.dtype))
    rotation = world_to_camera[:3, :3]
    translation = world_to_camera[:3, 3]
    points_camera = points @ rotation.T + translation
    normals_camera = F.normalize(normals @ rotation.T, dim=-1, eps=1.0e-6)
    tangents_camera = F.normalize(tangents @ rotation.T, dim=-1, eps=1.0e-6)
    bitangents_camera = F.normalize(bitangents @ rotation.T, dim=-1, eps=1.0e-6)
    return points_camera, normals_camera, tangents_camera, bitangents_camera


def transform_points_camera_to_world(points_camera: torch.Tensor, camera_to_world: torch.Tensor) -> torch.Tensor:
    rotation = camera_to_world[:, :3, :3]
    translation = camera_to_world[:, :3, 3]
    return torch.bmm(rotation, points_camera.unsqueeze(-1)).squeeze(-1) + translation


def initialize_powerfoam_normals(
    *,
    frame_count: int,
    cell_count: int,
    dtype: torch.dtype,
    normal_init_jitter: float,
    video_init_mode: str,
    camera_decoder: PowerFoamImplicitCameraDecoder | None,
    generator: torch.Generator,
) -> torch.Tensor:
    if str(video_init_mode) == "orbit_camera":
        if camera_decoder is None:
            raise ValueError("model.video_init_mode='orbit_camera' requires camera.enabled=true")
        source_frame = torch.remainder(torch.arange(int(cell_count), dtype=torch.long), int(frame_count))
        c2w = camera_decoder.base_camera_to_world_matrices(
            torch.arange(int(frame_count)),
            device=torch.device("cpu"),
            dtype=dtype,
        )
        camera_facing = torch.tensor([0.0, 0.0, -1.0], dtype=dtype)
        init_normals = torch.matmul(c2w[source_frame, :3, :3], camera_facing)
        init_normals = init_normals.unsqueeze(0).repeat(int(frame_count), 1, 1).contiguous()
    else:
        init_normals = torch.zeros(int(frame_count), int(cell_count), 3, dtype=dtype)
        init_normals[..., 2] = -1.0
    if float(normal_init_jitter) != 0.0:
        init_normals = init_normals + float(normal_init_jitter) * torch.randn(
            int(frame_count),
            int(cell_count),
            3,
            generator=generator,
            dtype=dtype,
        )
    return F.normalize(init_normals, dim=-1, eps=1.0e-6)


def initialize_full_powerfoam_from_orbit_video(
    init_frames: torch.Tensor,
    *,
    cell_count: int,
    xy_extent: float,
    z_min: float,
    z_max: float,
    fov_degrees: float,
    image_init_depth: float | None,
    radius_min: float,
    radius_scale: float,
    num_texel_sites: int,
    image_init_jitter: float,
    texel_site_scale: float,
    camera_decoder: PowerFoamImplicitCameraDecoder,
    generator: torch.Generator,
) -> PowerFoamInitialization:
    if init_frames.dim() != 4 or init_frames.size(1) != 3:
        raise ValueError("init_frames must be [T,3,H,W]")
    frame_count, _channels, height, width = init_frames.shape
    x01, y01, _rows, _cols = make_image_init_uv(
        cell_count,
        jitter_fraction=float(image_init_jitter),
        generator=generator,
    )
    source_frame = torch.remainder(torch.arange(cell_count, dtype=torch.long), int(frame_count))
    depth = float(camera_decoder.base_radius) if image_init_depth is None else float(image_init_depth)
    tan_half_fov = math.tan(math.radians(float(fov_degrees)) * 0.5)
    dirs = torch.stack(
        [
            (2.0 * x01 - 1.0) * tan_half_fov,
            -(2.0 * y01 - 1.0) * tan_half_fov * (float(height) / float(width)),
            torch.ones_like(x01),
        ],
        dim=-1,
    )
    dirs = F.normalize(dirs, dim=-1)
    points_camera = dirs * depth
    c2w = camera_decoder.base_camera_to_world_matrices(
        torch.arange(int(frame_count)),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    points = transform_points_camera_to_world(points_camera, c2w[source_frame])
    points = points.clone()
    points[:, :2] = points[:, :2].clamp(-0.95 * float(xy_extent), 0.95 * float(xy_extent))
    points[:, 2] = points[:, 2].clamp(float(z_min) + 1.0e-4, float(z_max) - 1.0e-4)
    points = points.unsqueeze(0).repeat(int(frame_count), 1, 1).contiguous()

    sample_grid = torch.stack([2.0 * x01 - 1.0, 2.0 * y01 - 1.0], dim=-1).view(1, cell_count, 1, 2)
    sampled_colors = F.grid_sample(
        init_frames.detach().cpu().float().clamp(0.0, 1.0),
        sample_grid.repeat(int(frame_count), 1, 1, 1),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).squeeze(-1).permute(0, 2, 1)
    cell_colors = sampled_colors[source_frame, torch.arange(cell_count)]

    radii = estimate_knn_radii(points, radius_scale=float(radius_scale), radius_min=float(radius_min))
    quaternions = camera_facing_quaternion(int(frame_count), int(cell_count))
    site_cols = math.ceil(math.sqrt(float(num_texel_sites)))
    site_rows = math.ceil(float(num_texel_sites) / float(site_cols))
    site_x = (torch.arange(site_cols, dtype=torch.float32) + 0.5) / float(site_cols) - 0.5
    site_y = (torch.arange(site_rows, dtype=torch.float32) + 0.5) / float(site_rows) - 0.5
    sy, sx = torch.meshgrid(site_y, site_x, indexing="ij")
    site_offsets = torch.stack([sx.reshape(-1), sy.reshape(-1)], dim=-1)[: int(num_texel_sites)]
    texel_sites = (0.5 * float(texel_site_scale) * site_offsets).view(1, 1, int(num_texel_sites), 2)
    texel_sites = texel_sites.repeat(int(frame_count), int(cell_count), 1, 1).contiguous()
    texel_sv_axis = torch.zeros(int(frame_count), int(cell_count), int(num_texel_sites), 1, 3, dtype=torch.float32)
    texel_sv_axis[..., 2] = 1.0
    texel_sv_rgb = (cell_colors.view(1, cell_count, 1, 1, 3).repeat(int(frame_count), 1, int(num_texel_sites), 1, 1) - 0.5)
    texel_height = torch.zeros(int(frame_count), int(cell_count), int(num_texel_sites), dtype=torch.float32)
    return PowerFoamInitialization(
        points=points,
        radii=radii,
        quaternions=quaternions,
        texel_sites=texel_sites,
        texel_sv_axis=texel_sv_axis,
        texel_sv_rgb=texel_sv_rgb.contiguous(),
        texel_height=texel_height,
    )


def make_gaussian_time_basis(frame_count: int, basis_count: int, sigma_scale: float) -> torch.Tensor:
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    times = torch.linspace(0.0, 1.0, frame_count, dtype=torch.float32)
    centers = torch.linspace(0.0, 1.0, basis_count, dtype=torch.float32)
    spacing = 1.0 / float(max(basis_count - 1, 1))
    sigma = max(spacing * float(sigma_scale), 1.0e-4)
    basis = torch.exp(-0.5 * ((times[:, None] - centers[None, :]) / sigma).square())
    return basis / basis.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)


def fit_temporal_basis(values: torch.Tensor, basis: torch.Tensor, *, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    base = values.mean(dim=0)
    coeff = torch.zeros((basis.shape[1], *values.shape[1:]), dtype=values.dtype)
    if mode == "fit" and values.shape[0] > 1:
        residual = (values - base).reshape(values.shape[0], -1)
        solution = torch.linalg.pinv(basis).to(residual.dtype) @ residual
        coeff = solution.reshape(basis.shape[1], *values.shape[1:]).contiguous()
    return base.contiguous(), coeff.contiguous()


def temporal_accel(values: torch.Tensor) -> torch.Tensor:
    if values.shape[0] < 3:
        return values.new_zeros(())
    return (values[2:] - 2.0 * values[1:-1] + values[:-2]).square().mean()


def atanh_clamped(values: torch.Tensor) -> torch.Tensor:
    return torch.atanh(values.clamp(-0.9999, 0.9999))


def temporal_motion_metrics(
    points: torch.Tensor,
    features: torch.Tensor,
    *,
    render_size: int,
    fov_degrees: float,
    camera_to_world: torch.Tensor | None = None,
) -> dict[str, float]:
    if points.shape[0] < 2:
        return {
            "state_mean_temporal_xy_delta": 0.0,
            "state_p95_temporal_xy_delta": 0.0,
            "state_mean_temporal_z_delta": 0.0,
            "state_mean_temporal_screen_delta_px": 0.0,
            "state_p95_temporal_screen_delta_px": 0.0,
            "state_temporal_screen_valid_fraction": 0.0,
            "state_mean_temporal_feature_abs_delta": 0.0,
        }
    dxy = torch.linalg.vector_norm(points[1:, :, :2] - points[:-1, :, :2], dim=-1)
    dz = (points[1:, :, 2] - points[:-1, :, 2]).abs()
    screen_points = points
    if camera_to_world is not None:
        world_to_camera = torch.linalg.inv(camera_to_world.to(device=points.device, dtype=points.dtype))
        screen_points = torch.bmm(points, world_to_camera[:, :3, :3].transpose(1, 2)) + world_to_camera[:, None, :3, 3]
    tan_half_fov = math.tan(math.radians(float(fov_degrees)) * 0.5)
    z = screen_points[..., 2]
    screen = torch.stack(
        [
            0.5 * (screen_points[..., 0] / (z.clamp_min(1.0e-6) * tan_half_fov) + 1.0) * float(int(render_size) - 1),
            0.5 * (-screen_points[..., 1] / (z.clamp_min(1.0e-6) * tan_half_fov) + 1.0) * float(int(render_size) - 1),
        ],
        dim=-1,
    )
    dscreen = torch.linalg.vector_norm(screen[1:] - screen[:-1], dim=-1)
    valid_screen = (z[1:] > 1.0e-4) & (z[:-1] > 1.0e-4)
    valid_dscreen = dscreen[valid_screen]
    if valid_dscreen.numel() == 0:
        mean_screen_delta = points.new_zeros(())
        p95_screen_delta = points.new_zeros(())
    else:
        mean_screen_delta = valid_dscreen.mean()
        p95_screen_delta = valid_dscreen.flatten().quantile(0.95)
    feature_delta = (features[1:] - features[:-1]).abs()
    return {
        "state_mean_temporal_xy_delta": float(dxy.mean().cpu()),
        "state_p95_temporal_xy_delta": float(dxy.flatten().quantile(0.95).cpu()),
        "state_mean_temporal_z_delta": float(dz.mean().cpu()),
        "state_mean_temporal_screen_delta_px": float(mean_screen_delta.cpu()),
        "state_p95_temporal_screen_delta_px": float(p95_screen_delta.cpu()),
        "state_temporal_screen_valid_fraction": float(valid_screen.float().mean().cpu()),
        "state_mean_temporal_feature_abs_delta": float(feature_delta.mean().cpu()),
    }


class DynamicMetalPowerFoamVideo(nn.Module):
    def __init__(
        self,
        *,
        frame_count: int,
        cell_count: int,
        render_size: int,
        fov_degrees: float,
        neighbor_count: int,
        adjacency_mode: str,
        dynamic_mode: str,
        time_basis_count: int,
        time_basis_sigma_scale: float,
        temporal_init_mode: str,
        dynamic_centers: bool,
        dynamic_radii: bool,
        dynamic_densities: bool,
        dynamic_features: bool,
        dynamic_normals: bool,
        dynamic_texel_sites: bool,
        xy_extent: float,
        z_min: float,
        z_max: float,
        radius_init: float,
        radius_min: float,
        radius_scale: float,
        density_init: float,
        normal_init_jitter: float,
        num_texel_sites: int,
        texel_site_scale: float,
        color_init_mode: str,
        video_init_mode: str,
        seed: int,
        init_frames: torch.Tensor | None,
        image_init_depth: float | None,
        image_init_jitter: float,
        raster_config: FoamRasterConfig,
        camera_decoder: PowerFoamImplicitCameraDecoder | None = None,
    ) -> None:
        super().__init__()
        if frame_count < 1:
            raise ValueError("frame_count must be positive")
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        if init_frames is None:
            init = initialize_random_full_powerfoam(
                frame_count=frame_count,
                cell_count=cell_count,
                xy_extent=xy_extent,
                z_min=z_min,
                z_max=z_max,
                radius_init=radius_init,
                radius_min=radius_min,
                num_texel_sites=int(num_texel_sites),
                sv_dof=1,
                sv_axis_init=1.0,
                generator=generator,
            )
            init_points = init.points
            init_radii = init.radii
            texel_sites_init = init.texel_sites.clamp(-float(texel_site_scale) * 0.999, float(texel_site_scale) * 0.999)
            texel_colors_init = (init.texel_sv_rgb[..., 0, :] + 0.5).clamp(0.0, 1.0)
        elif str(video_init_mode) == "orbit_camera":
            if camera_decoder is None:
                raise ValueError("model.video_init_mode='orbit_camera' requires camera.enabled=true")
            init = initialize_full_powerfoam_from_orbit_video(
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
                image_init_jitter=image_init_jitter,
                texel_site_scale=texel_site_scale,
                camera_decoder=camera_decoder,
                generator=generator,
            )
            init_points = init.points
            init_radii = init.radii
            texel_sites_init = init.texel_sites.clamp(-float(texel_site_scale) * 0.999, float(texel_site_scale) * 0.999)
            texel_colors_init = (init.texel_sv_rgb[..., 0, :] + 0.5).clamp(0.0, 1.0)
        else:
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
                sv_dof=1,
                sv_axis_init=1.0,
                image_init_jitter=image_init_jitter,
                generator=generator,
            )
            init_points = init.points
            init_radii = init.radii
            texel_sites_init = init.texel_sites.clamp(-float(texel_site_scale) * 0.999, float(texel_site_scale) * 0.999)
            texel_colors_init = (init.texel_sv_rgb[..., 0, :] + 0.5).clamp(0.0, 1.0)
        if str(color_init_mode) == "random":
            texel_colors_init = torch.rand(
                frame_count,
                cell_count,
                int(num_texel_sites),
                3,
                generator=generator,
                dtype=init_points.dtype,
            )

        init_density = torch.full((frame_count, cell_count), max(float(density_init), 1.0e-4))
        init_normals = initialize_powerfoam_normals(
            frame_count=frame_count,
            cell_count=cell_count,
            dtype=init_points.dtype,
            normal_init_jitter=normal_init_jitter,
            video_init_mode=video_init_mode,
            camera_decoder=camera_decoder,
            generator=generator,
        )
        init_tangents = stable_tangent_from_normals(init_normals)

        raw_xy = atanh_clamped(init_points[..., :2] / float(xy_extent))
        raw_z = logit_clamped((init_points[..., 2:] - float(z_min)) / (float(z_max) - float(z_min)))
        raw_radii = inverse_softplus((init_radii - float(radius_min)).clamp_min(1.0e-4), beta=POWERFOAM_SOFTPLUS_BETA)
        raw_densities = inverse_softplus(init_density, beta=POWERFOAM_SOFTPLUS_BETA)
        raw_features = logit_clamped(texel_colors_init.clamp(0.0, 1.0))
        raw_texel_sites = atanh_clamped(texel_sites_init / float(texel_site_scale))

        self.dynamic_mode = str(dynamic_mode)
        self.dynamic_centers = bool(dynamic_centers)
        self.dynamic_radii = bool(dynamic_radii)
        self.dynamic_densities = bool(dynamic_densities)
        self.dynamic_features = bool(dynamic_features)
        self.dynamic_normals = bool(dynamic_normals)
        self.dynamic_texel_sites = bool(dynamic_texel_sites)
        self.xy_extent = float(xy_extent)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.radius_min = float(radius_min)
        self.neighbor_count = int(neighbor_count)
        self.adjacency_mode = str(adjacency_mode)
        self.texel_site_scale = float(texel_site_scale)
        self.render_size = int(render_size)
        self.fov_degrees = float(fov_degrees)
        self.raster_config = raster_config
        self.camera_decoder = camera_decoder
        self.register_buffer("rays", make_pinhole_rays(render_size, render_size, fov_degrees, torch.device("cpu")), persistent=False)
        basis = make_gaussian_time_basis(frame_count, int(time_basis_count), float(time_basis_sigma_scale))
        self.register_buffer("frame_basis", basis, persistent=False)

        if self.dynamic_mode == "per_frame_smooth":
            self.raw_xy = nn.Parameter(raw_xy)
            self.raw_z = nn.Parameter(raw_z)
            self.raw_radii = nn.Parameter(raw_radii)
            self.raw_densities = nn.Parameter(raw_densities)
            self.raw_features = nn.Parameter(raw_features)
            self.raw_normals = nn.Parameter(init_normals)
            self.raw_tangents = nn.Parameter(init_tangents)
            self.raw_texel_sites = nn.Parameter(raw_texel_sites)
        else:
            self.raw_xy0, self.raw_xy_coeff = self._make_temporal_param(
                raw_xy, dynamic=self.dynamic_centers, temporal_init_mode=temporal_init_mode
            )
            self.raw_z0, self.raw_z_coeff = self._make_temporal_param(
                raw_z, dynamic=self.dynamic_centers, temporal_init_mode=temporal_init_mode
            )
            self.raw_radii0, self.raw_radii_coeff = self._make_temporal_param(
                raw_radii, dynamic=self.dynamic_radii, temporal_init_mode=temporal_init_mode
            )
            self.raw_densities0, self.raw_densities_coeff = self._make_temporal_param(
                raw_densities, dynamic=self.dynamic_densities, temporal_init_mode=temporal_init_mode
            )
            self.raw_features0, self.raw_features_coeff = self._make_temporal_param(
                raw_features, dynamic=self.dynamic_features, temporal_init_mode=temporal_init_mode
            )
            self.raw_normals0, self.raw_normals_coeff = self._make_temporal_param(
                init_normals, dynamic=self.dynamic_normals, temporal_init_mode=temporal_init_mode
            )
            self.raw_tangents0, self.raw_tangents_coeff = self._make_temporal_param(
                init_tangents, dynamic=self.dynamic_normals, temporal_init_mode=temporal_init_mode
            )
            self.raw_texel_sites0, self.raw_texel_sites_coeff = self._make_temporal_param(
                raw_texel_sites, dynamic=self.dynamic_texel_sites, temporal_init_mode=temporal_init_mode
            )

        self.register_buffer("initial_points", init_points.clone(), persistent=False)
        self.register_buffer("initial_radii", init_radii.clone(), persistent=False)
        self.register_buffer("initial_densities", init_density.clone(), persistent=False)
        self.register_buffer("initial_features", texel_colors_init.clone(), persistent=False)
        self.register_buffer("initial_normals", init_normals.clone(), persistent=False)
        self.register_buffer("initial_tangents", init_tangents.clone(), persistent=False)
        self.register_buffer("initial_texel_sites", texel_sites_init.clone(), persistent=False)

    def _make_temporal_param(
        self,
        values: torch.Tensor,
        *,
        dynamic: bool,
        temporal_init_mode: str,
    ) -> tuple[nn.Parameter, nn.Parameter | None]:
        base, coeff = fit_temporal_basis(values, self.frame_basis.cpu(), mode=str(temporal_init_mode))
        base_param = nn.Parameter(base)
        if not dynamic:
            return base_param, None
        return base_param, nn.Parameter(coeff)

    def _decode_temporal(self, base: torch.Tensor, coeff: torch.Tensor | None, frame_indices: torch.Tensor | None) -> torch.Tensor:
        if self.dynamic_mode == "per_frame_smooth":
            raise RuntimeError("_decode_temporal is only used by rbf mode")
        if frame_indices is None:
            basis = self.frame_basis.to(device=base.device, dtype=base.dtype)
        else:
            basis = self.frame_basis[frame_indices.to(device=self.frame_basis.device, dtype=torch.long)].to(
                device=base.device,
                dtype=base.dtype,
            )
        values = base[None].expand(basis.shape[0], *base.shape)
        if coeff is not None:
            values = values + torch.einsum("tk,k...->t...", basis, coeff)
        return values

    def raw_parameter_tensors(
        self, frame_indices: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.dynamic_mode == "per_frame_smooth":
            if frame_indices is None:
                return (
                    self.raw_xy,
                    self.raw_z,
                    self.raw_radii,
                    self.raw_densities,
                    self.raw_features,
                    self.raw_normals,
                    self.raw_tangents,
                    self.raw_texel_sites,
                )
            idx = frame_indices.to(device=self.raw_xy.device, dtype=torch.long)
            return (
                self.raw_xy[idx],
                self.raw_z[idx],
                self.raw_radii[idx],
                self.raw_densities[idx],
                self.raw_features[idx],
                self.raw_normals[idx],
                self.raw_tangents[idx],
                self.raw_texel_sites[idx],
            )
        return (
            self._decode_temporal(self.raw_xy0, self.raw_xy_coeff, frame_indices),
            self._decode_temporal(self.raw_z0, self.raw_z_coeff, frame_indices),
            self._decode_temporal(self.raw_radii0, self.raw_radii_coeff, frame_indices),
            self._decode_temporal(self.raw_densities0, self.raw_densities_coeff, frame_indices),
            self._decode_temporal(self.raw_features0, self.raw_features_coeff, frame_indices),
            self._decode_temporal(self.raw_normals0, self.raw_normals_coeff, frame_indices),
            self._decode_temporal(self.raw_tangents0, self.raw_tangents_coeff, frame_indices),
            self._decode_temporal(self.raw_texel_sites0, self.raw_texel_sites_coeff, frame_indices),
        )

    def decoded_parameters(
        self, frame_indices: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_xy, raw_z, raw_radii, raw_densities, raw_features, raw_normals, _raw_tangents, _raw_texel_sites = (
            self.raw_parameter_tensors(frame_indices)
        )
        xy = torch.tanh(raw_xy) * self.xy_extent
        z = self.z_min + torch.sigmoid(raw_z) * (self.z_max - self.z_min)
        points = torch.cat([xy, z], dim=-1)
        radii = F.softplus(raw_radii, beta=POWERFOAM_SOFTPLUS_BETA) + self.radius_min
        densities = F.softplus(raw_densities, beta=POWERFOAM_SOFTPLUS_BETA)
        features = torch.sigmoid(raw_features)
        normals = F.normalize(raw_normals, dim=-1, eps=1.0e-6)
        return points, radii, densities, features, normals

    def decoded_texel_sites(self, frame_indices: torch.Tensor | None = None) -> torch.Tensor:
        *_prefix, raw_texel_sites = self.raw_parameter_tensors(frame_indices)
        return self.texel_site_scale * torch.tanh(raw_texel_sites)

    def decoded_surface_frame(
        self,
        normals: torch.Tensor,
        frame_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _raw_xy, _raw_z, _raw_radii, _raw_densities, _raw_features, _raw_normals, raw_tangents, _raw_texel_sites = (
            self.raw_parameter_tensors(frame_indices)
        )
        return orthonormal_surface_frame(normals, raw_tangents)

    def optimizer_param_groups(self, train_cfg: dict[str, Any]) -> list[dict[str, object]]:
        lr = float(train_cfg["lr"])
        groups: list[dict[str, object]] = []

        def add(params: list[nn.Parameter | None], multiplier: str, name: str) -> None:
            clean = [param for param in params if param is not None]
            if clean:
                groups.append({"params": clean, "lr": float(train_cfg[multiplier]) * lr, "name": name})

        if self.dynamic_mode == "per_frame_smooth":
            add([self.raw_xy, self.raw_z], "point_lr_multiplier", "points")
            add([self.raw_radii], "radius_lr_multiplier", "radii")
            add([self.raw_densities], "density_lr_multiplier", "density")
            add([self.raw_features], "feature_lr_multiplier", "features")
            add([self.raw_normals, self.raw_tangents], "normal_lr_multiplier", "surface_frame")
            add([self.raw_texel_sites], "texel_site_lr_multiplier", "texel_sites")
            camera_group = camera_param_group(self.camera_decoder, train_cfg)
            if camera_group is not None:
                groups.append(camera_group)
            return groups

        add([self.raw_xy0, self.raw_z0], "point_lr_multiplier", "base_points")
        add([self.raw_radii0], "radius_lr_multiplier", "base_radii")
        add([self.raw_densities0], "density_lr_multiplier", "base_density")
        add([self.raw_features0], "feature_lr_multiplier", "base_features")
        add([self.raw_normals0, self.raw_tangents0], "normal_lr_multiplier", "base_surface_frame")
        add([self.raw_texel_sites0], "texel_site_lr_multiplier", "base_texel_sites")
        add([self.raw_xy_coeff, self.raw_z_coeff], "temporal_lr_multiplier", "temporal_points")
        add([self.raw_radii_coeff], "temporal_lr_multiplier", "temporal_radii")
        add([self.raw_densities_coeff], "temporal_lr_multiplier", "temporal_density")
        add([self.raw_features_coeff], "temporal_lr_multiplier", "temporal_features")
        add([self.raw_normals_coeff, self.raw_tangents_coeff], "temporal_lr_multiplier", "temporal_surface_frame")
        add([self.raw_texel_sites_coeff], "temporal_lr_multiplier", "temporal_texel_sites")
        camera_group = camera_param_group(self.camera_decoder, train_cfg)
        if camera_group is not None:
            groups.append(camera_group)
        return groups

    def temporal_regularization(self, loss_cfg: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        points, radii, densities, features, _normals = self.decoded_parameters()
        terms = {
            "temporal_center_accel": temporal_accel(points),
            "temporal_radius_accel": temporal_accel(radii),
            "temporal_density_accel": temporal_accel(densities),
            "temporal_feature_accel": temporal_accel(features),
            "temporal_coeff_l2": points.new_zeros(()),
        }
        if self.dynamic_mode == "rbf":
            coeffs = [
                self.raw_xy_coeff,
                self.raw_z_coeff,
                self.raw_radii_coeff,
                self.raw_densities_coeff,
                self.raw_features_coeff,
                self.raw_normals_coeff,
                self.raw_tangents_coeff,
                self.raw_texel_sites_coeff,
            ]
            coeff_terms = [c.square().mean() for c in coeffs if c is not None]
            if coeff_terms:
                terms["temporal_coeff_l2"] = torch.stack(coeff_terms).mean()
        loss = points.new_zeros(())
        for key, value in terms.items():
            loss = loss + float(loss_cfg[f"{key}_weight"]) * value
        camera_loss, camera_terms = camera_regularization(self.camera_decoder, loss_cfg)
        if camera_loss is not None:
            loss = loss + camera_loss
            terms.update(camera_terms)
        return loss, terms

    @torch.no_grad()
    def parameter_drift_metrics(self) -> dict[str, float]:
        points, radii, densities, features, normals = self.decoded_parameters()
        texel_sites = self.decoded_texel_sites()
        center_offset = points - self.initial_points.to(points.device)
        center_delta = torch.linalg.vector_norm(center_offset, dim=-1)
        metrics = {
            "state_mean_center_delta": float(center_delta.mean().cpu()),
            "state_p95_center_delta": float(center_delta.flatten().quantile(0.95).cpu()),
            "state_max_center_delta": float(center_delta.max().cpu()),
            "state_mean_xy_delta": float(torch.linalg.vector_norm(center_offset[..., :2], dim=-1).mean().cpu()),
            "state_mean_z_delta": float(center_offset[..., 2].abs().mean().cpu()),
            "state_mean_radius_delta": float((radii - self.initial_radii.to(radii.device)).abs().mean().cpu()),
            "state_mean_density_delta": float((densities - self.initial_densities.to(densities.device)).abs().mean().cpu()),
            "state_mean_feature_delta": float((features - self.initial_features.to(features.device)).abs().mean().cpu()),
            "state_mean_normal_delta": float((normals - self.initial_normals.to(normals.device)).norm(dim=-1).mean().cpu()),
            "state_mean_texel_site_delta": float(
                (texel_sites - self.initial_texel_sites.to(texel_sites.device)).norm(dim=-1).mean().cpu()
            ),
        }
        camera_to_world = None if self.camera_decoder is None else self.camera_decoder.camera_to_world_matrices()
        metrics.update(
            temporal_motion_metrics(
                points,
                features,
                render_size=self.render_size,
                fov_degrees=self.fov_degrees,
                camera_to_world=camera_to_world,
            )
        )
        if self.dynamic_mode == "rbf":
            coeff_abs = [
                c.abs().mean()
                for c in [
                    self.raw_xy_coeff,
                    self.raw_z_coeff,
                    self.raw_radii_coeff,
                    self.raw_densities_coeff,
                    self.raw_features_coeff,
                    self.raw_normals_coeff,
                    self.raw_tangents_coeff,
                    self.raw_texel_sites_coeff,
                ]
                if c is not None
            ]
            if coeff_abs:
                metrics["state_mean_temporal_coeff_abs"] = float(torch.stack(coeff_abs).mean().cpu())
        metrics.update(compact_camera_metrics(self.camera_decoder))
        return metrics

    def decoded_camera_rays(self, frame_indices: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        return decoded_powerfoam_rays(
            self.camera_decoder,
            self.rays,
            frame_indices.to(device=next(self.parameters()).device, dtype=torch.long),
            height=self.render_size,
            width=self.render_size,
            device=next(self.parameters()).device,
            dtype=dtype,
        )

    def forward(self, frame_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if next(self.parameters()).device.type != "mps":
            raise RuntimeError("DynamicMetalPowerFoamVideo requires an MPS device")
        frame_indices = frame_indices.to(device=next(self.parameters()).device, dtype=torch.long)
        points, radii, densities, features, normals = self.decoded_parameters(frame_indices)
        texel_sites = self.decoded_texel_sites(frame_indices)
        tangents, bitangents = self.decoded_surface_frame(normals, frame_indices)
        rays = self.rays.to(device=points.device, dtype=points.dtype)
        camera_to_world = None if self.camera_decoder is None else self.camera_decoder.camera_to_world_matrices(frame_indices)
        renders = []
        alphas = []
        for local_index in range(int(frame_indices.numel())):
            point = points[local_index]
            normal = normals[local_index]
            tangent = tangents[local_index]
            bitangent = bitangents[local_index]
            if camera_to_world is not None:
                point, normal, tangent, bitangent = transform_powerfoam_frame_to_camera(
                    point,
                    normal,
                    tangent,
                    bitangent,
                    camera_to_world[local_index],
                )
            radius = radii[local_index]
            adjacency, offsets = build_csr_adjacency(
                point,
                radius,
                neighbor_count=self.neighbor_count,
                mode=self.adjacency_mode,
            )
            out, alpha = rasterize_power_foam_oriented_texel_surface(
                point,
                radius,
                densities[local_index],
                texel_sites[local_index],
                features[local_index],
                normal,
                adjacency,
                offsets,
                rays,
                self.raster_config,
                tangents=tangent,
                bitangents=bitangent,
            )
            renders.append(out.permute(0, 3, 1, 2))
            alphas.append(alpha)
        return torch.cat(renders, dim=0), torch.cat(alphas, dim=0)


def make_texel_feature_init(
    texel_rgb: torch.Tensor,
    *,
    feature_dim: int,
    rgb_init: str,
    noise_std: float,
    generator: torch.Generator,
) -> torch.Tensor:
    features = float(noise_std) * torch.randn(
        (*texel_rgb.shape[:-1], int(feature_dim)),
        generator=generator,
        dtype=texel_rgb.dtype,
    )
    if rgb_init == "logit":
        features[..., :3] = logit_clamped(texel_rgb.clamp(0.0, 1.0))
    elif rgb_init == "rgb":
        features[..., :3] = texel_rgb.clamp(0.0, 1.0)
    return features.contiguous()


class TokenDynamicPowerFoamFeatures(nn.Module):
    """Token decoder for dynamic PowerFoam feature splatting.

    This is intentionally one token per cell for the first baseline: the token
    bank owns the canonical cell state plus RBF temporal coefficients, and the
    Metal rasterizer splats F-channel texel features before a colorizer maps
    features to RGB.
    """

    def __init__(
        self,
        *,
        frame_count: int,
        cell_count: int,
        render_size: int,
        fov_degrees: float,
        neighbor_count: int,
        adjacency_mode: str,
        time_basis_count: int,
        time_basis_sigma_scale: float,
        temporal_init_mode: str,
        dynamic_centers: bool,
        dynamic_radii: bool,
        dynamic_densities: bool,
        dynamic_features: bool,
        dynamic_normals: bool,
        dynamic_texel_sites: bool,
        feature_dim: int,
        feature_init_noise: float,
        feature_rgb_init: str,
        token_dim: int,
        token_hidden_dim: int,
        token_hidden_layers: int,
        token_init_std: float,
        token_output_init_std: float,
        token_point_residual_scale: float,
        token_z_residual_scale: float,
        token_radius_residual_scale: float,
        token_density_residual_scale: float,
        token_feature_residual_scale: float,
        token_normal_residual_scale: float,
        token_texel_site_residual_scale: float,
        token_temporal_residual_scale: float,
        static_dynamic_split: bool,
        dynamic_cells: int | None,
        dynamic_cell_fraction: float,
        xy_extent: float,
        z_min: float,
        z_max: float,
        radius_init: float,
        radius_min: float,
        radius_scale: float,
        density_init: float,
        normal_init_jitter: float,
        num_texel_sites: int,
        texel_site_scale: float,
        color_init_mode: str,
        video_init_mode: str,
        seed: int,
        init_frames: torch.Tensor | None,
        image_init_depth: float | None,
        image_init_jitter: float,
        raster_config: FoamRasterConfig,
        camera_decoder: PowerFoamImplicitCameraDecoder | None = None,
    ) -> None:
        super().__init__()
        if frame_count < 1:
            raise ValueError("frame_count must be positive")
        if int(feature_dim) < 3:
            raise ValueError("feature_dim must be at least 3")
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        if init_frames is None:
            init = initialize_random_full_powerfoam(
                frame_count=frame_count,
                cell_count=cell_count,
                xy_extent=xy_extent,
                z_min=z_min,
                z_max=z_max,
                radius_init=radius_init,
                radius_min=radius_min,
                num_texel_sites=int(num_texel_sites),
                sv_dof=1,
                sv_axis_init=1.0,
                generator=generator,
            )
            init_points = init.points
            init_radii = init.radii
            texel_sites_init = init.texel_sites.clamp(-float(texel_site_scale) * 0.999, float(texel_site_scale) * 0.999)
            texel_colors_init = (init.texel_sv_rgb[..., 0, :] + 0.5).clamp(0.0, 1.0)
        elif str(video_init_mode) == "orbit_camera":
            if camera_decoder is None:
                raise ValueError("model.video_init_mode='orbit_camera' requires camera.enabled=true")
            init = initialize_full_powerfoam_from_orbit_video(
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
                image_init_jitter=image_init_jitter,
                texel_site_scale=texel_site_scale,
                camera_decoder=camera_decoder,
                generator=generator,
            )
            init_points = init.points
            init_radii = init.radii
            texel_sites_init = init.texel_sites.clamp(-float(texel_site_scale) * 0.999, float(texel_site_scale) * 0.999)
            texel_colors_init = (init.texel_sv_rgb[..., 0, :] + 0.5).clamp(0.0, 1.0)
        else:
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
                sv_dof=1,
                sv_axis_init=1.0,
                image_init_jitter=image_init_jitter,
                generator=generator,
            )
            init_points = init.points
            init_radii = init.radii
            texel_sites_init = init.texel_sites.clamp(-float(texel_site_scale) * 0.999, float(texel_site_scale) * 0.999)
            texel_colors_init = (init.texel_sv_rgb[..., 0, :] + 0.5).clamp(0.0, 1.0)
        if str(color_init_mode) == "random":
            texel_colors_init = torch.rand(
                frame_count,
                cell_count,
                int(num_texel_sites),
                3,
                generator=generator,
                dtype=init_points.dtype,
            )

        init_density = torch.full((frame_count, cell_count), max(float(density_init), 1.0e-4))
        init_normals = initialize_powerfoam_normals(
            frame_count=frame_count,
            cell_count=cell_count,
            dtype=init_points.dtype,
            normal_init_jitter=normal_init_jitter,
            video_init_mode=video_init_mode,
            camera_decoder=camera_decoder,
            generator=generator,
        )
        init_tangents = stable_tangent_from_normals(init_normals)

        raw_xy = atanh_clamped(init_points[..., :2] / float(xy_extent))
        raw_z = logit_clamped((init_points[..., 2:] - float(z_min)) / (float(z_max) - float(z_min)))
        raw_radii = inverse_softplus((init_radii - float(radius_min)).clamp_min(1.0e-4), beta=POWERFOAM_SOFTPLUS_BETA)
        raw_densities = inverse_softplus(init_density, beta=POWERFOAM_SOFTPLUS_BETA)
        raw_features = make_texel_feature_init(
            texel_colors_init,
            feature_dim=int(feature_dim),
            rgb_init=str(feature_rgb_init),
            noise_std=float(feature_init_noise),
            generator=generator,
        )
        raw_texel_sites = atanh_clamped(texel_sites_init / float(texel_site_scale))

        self.dynamic_mode = TOKEN_RBF_FEATURE_MODE
        self.dynamic_centers = bool(dynamic_centers)
        self.dynamic_radii = bool(dynamic_radii)
        self.dynamic_densities = bool(dynamic_densities)
        self.dynamic_features = bool(dynamic_features)
        self.dynamic_normals = bool(dynamic_normals)
        self.dynamic_texel_sites = bool(dynamic_texel_sites)
        self.cell_count = int(cell_count)
        self.feature_dim = int(feature_dim)
        self.num_texel_sites = int(num_texel_sites)
        self.static_dynamic_split = bool(static_dynamic_split)
        if dynamic_cells is None:
            dynamic_cell_count = max(1, int(round(self.cell_count * float(dynamic_cell_fraction))))
        else:
            dynamic_cell_count = int(dynamic_cells)
        self.dynamic_cell_count = min(self.cell_count, max(1, dynamic_cell_count))
        self.static_cell_count = self.cell_count - self.dynamic_cell_count if self.static_dynamic_split else 0
        self.xy_extent = float(xy_extent)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.radius_min = float(radius_min)
        self.neighbor_count = int(neighbor_count)
        self.adjacency_mode = str(adjacency_mode)
        self.texel_site_scale = float(texel_site_scale)
        self.render_size = int(render_size)
        self.fov_degrees = float(fov_degrees)
        self.raster_config = raster_config
        self.camera_decoder = camera_decoder
        self.register_buffer("rays", make_pinhole_rays(render_size, render_size, fov_degrees, torch.device("cpu")), persistent=False)
        basis = make_gaussian_time_basis(frame_count, int(time_basis_count), float(time_basis_sigma_scale))
        self.register_buffer("frame_basis", basis, persistent=False)
        dynamic_mask = torch.zeros(self.cell_count, dtype=torch.float32)
        if self.static_dynamic_split:
            dynamic_mask[self.static_cell_count :] = 1.0
        else:
            dynamic_mask[:] = 1.0
        self.register_buffer("dynamic_cell_mask", dynamic_mask, persistent=False)
        self.temporal_geometry_runtime_scale = 1.0
        self.temporal_feature_runtime_scale = 1.0

        def fit_init(values: torch.Tensor, *, dynamic: bool) -> tuple[torch.Tensor, torch.Tensor]:
            base, coeff = fit_temporal_basis(values, basis, mode=str(temporal_init_mode))
            if not dynamic:
                coeff = torch.zeros_like(coeff)
            return base, coeff

        init_raw_xy0, init_raw_xy_coeff = fit_init(raw_xy, dynamic=self.dynamic_centers)
        init_raw_z0, init_raw_z_coeff = fit_init(raw_z, dynamic=self.dynamic_centers)
        init_raw_radii0, init_raw_radii_coeff = fit_init(raw_radii, dynamic=self.dynamic_radii)
        init_raw_densities0, init_raw_densities_coeff = fit_init(raw_densities, dynamic=self.dynamic_densities)
        init_raw_features0, init_raw_features_coeff = fit_init(raw_features, dynamic=self.dynamic_features)
        init_raw_normals0, init_raw_normals_coeff = fit_init(init_normals, dynamic=self.dynamic_normals)
        init_raw_tangents0, init_raw_tangents_coeff = fit_init(init_tangents, dynamic=self.dynamic_normals)
        init_raw_texel_sites0, init_raw_texel_sites_coeff = fit_init(raw_texel_sites, dynamic=self.dynamic_texel_sites)

        for name, value in {
            "init_raw_xy0": init_raw_xy0,
            "init_raw_xy_coeff": init_raw_xy_coeff,
            "init_raw_z0": init_raw_z0,
            "init_raw_z_coeff": init_raw_z_coeff,
            "init_raw_radii0": init_raw_radii0,
            "init_raw_radii_coeff": init_raw_radii_coeff,
            "init_raw_densities0": init_raw_densities0,
            "init_raw_densities_coeff": init_raw_densities_coeff,
            "init_raw_features0": init_raw_features0,
            "init_raw_features_coeff": init_raw_features_coeff,
            "init_raw_normals0": init_raw_normals0,
            "init_raw_normals_coeff": init_raw_normals_coeff,
            "init_raw_tangents0": init_raw_tangents0,
            "init_raw_tangents_coeff": init_raw_tangents_coeff,
            "init_raw_texel_sites0": init_raw_texel_sites0,
            "init_raw_texel_sites_coeff": init_raw_texel_sites_coeff,
        }.items():
            self.register_buffer(name, value.contiguous(), persistent=False)

        self.register_buffer("initial_points", init_points.clone(), persistent=False)
        self.register_buffer("initial_radii", init_radii.clone(), persistent=False)
        self.register_buffer("initial_densities", init_density.clone(), persistent=False)
        self.register_buffer("initial_features", raw_features.clone(), persistent=False)
        self.register_buffer("initial_normals", init_normals.clone(), persistent=False)
        self.register_buffer("initial_tangents", init_tangents.clone(), persistent=False)
        self.register_buffer("initial_texel_sites", texel_sites_init.clone(), persistent=False)

        basis_count = int(time_basis_count)
        temporal_scale = float(token_temporal_residual_scale)
        self._cell_chunks: list[tuple[str, tuple[int, ...], float]] = []

        def add_chunk(name: str, shape: tuple[int, ...], scale: float) -> None:
            self._cell_chunks.append((name, shape, float(scale)))

        add_chunk("raw_xy0", (2,), float(token_point_residual_scale))
        add_chunk("raw_z0", (1,), float(token_z_residual_scale))
        add_chunk("raw_radii0", (), float(token_radius_residual_scale))
        add_chunk("raw_densities0", (), float(token_density_residual_scale))
        add_chunk("raw_features0", (self.num_texel_sites, self.feature_dim), float(token_feature_residual_scale))
        add_chunk("raw_normals0", (3,), float(token_normal_residual_scale))
        add_chunk("raw_tangents0", (3,), float(token_normal_residual_scale))
        add_chunk("raw_texel_sites0", (self.num_texel_sites, 2), float(token_texel_site_residual_scale))
        if self.dynamic_centers:
            add_chunk("raw_xy_coeff", (basis_count, 2), float(token_point_residual_scale) * temporal_scale)
            add_chunk("raw_z_coeff", (basis_count, 1), float(token_z_residual_scale) * temporal_scale)
        if self.dynamic_radii:
            add_chunk("raw_radii_coeff", (basis_count,), float(token_radius_residual_scale) * temporal_scale)
        if self.dynamic_densities:
            add_chunk("raw_densities_coeff", (basis_count,), float(token_density_residual_scale) * temporal_scale)
        if self.dynamic_features:
            add_chunk(
                "raw_features_coeff",
                (basis_count, self.num_texel_sites, self.feature_dim),
                float(token_feature_residual_scale) * temporal_scale,
            )
        if self.dynamic_normals:
            add_chunk("raw_normals_coeff", (basis_count, 3), float(token_normal_residual_scale) * temporal_scale)
            add_chunk("raw_tangents_coeff", (basis_count, 3), float(token_normal_residual_scale) * temporal_scale)
        if self.dynamic_texel_sites:
            add_chunk(
                "raw_texel_sites_coeff",
                (basis_count, self.num_texel_sites, 2),
                float(token_texel_site_residual_scale) * temporal_scale,
            )

        self.total_decoder_dim = sum(math.prod(shape) if shape else 1 for _name, shape, _scale in self._cell_chunks)
        self.tokens = nn.Parameter(
            float(token_init_std) * torch.randn(self.cell_count, int(token_dim), generator=generator, dtype=init_points.dtype)
        )
        layers: list[nn.Module] = []
        in_dim = int(token_dim)
        for _ in range(int(token_hidden_layers)):
            layers.append(nn.Linear(in_dim, int(token_hidden_dim)))
            layers.append(nn.GELU())
            in_dim = int(token_hidden_dim)
        layers.append(nn.Linear(in_dim, self.total_decoder_dim))
        nn.init.normal_(layers[-1].weight, mean=0.0, std=float(token_output_init_std))
        nn.init.zeros_(layers[-1].bias)
        self.decoder = nn.Sequential(*layers)

    def _token_residuals(self) -> dict[str, torch.Tensor]:
        flat = self.decoder(self.tokens)
        residuals: dict[str, torch.Tensor] = {}
        offset = 0
        for name, shape, scale in self._cell_chunks:
            width = math.prod(shape) if shape else 1
            value = flat[:, offset : offset + width]
            offset += width
            if shape:
                value = value.reshape(self.cell_count, *shape)
            else:
                value = value.reshape(self.cell_count)
            value = value * scale
            if name.endswith("_coeff"):
                value = value.movedim(1, 0).contiguous()
            residuals[name] = value
        return residuals

    def _mask_temporal_coeff(self, name: str, value: torch.Tensor) -> torch.Tensor:
        if not name.endswith("_coeff"):
            return value
        mask = self.dynamic_cell_mask.to(device=value.device, dtype=value.dtype).view(1, self.cell_count, *([1] * (value.dim() - 2)))
        scale = self.temporal_feature_runtime_scale if name == "raw_features_coeff" else self.temporal_geometry_runtime_scale
        return value * mask * float(scale)

    def set_training_controls(self, *, temporal_geometry_scale: float, temporal_feature_scale: float) -> None:
        self.temporal_geometry_runtime_scale = float(temporal_geometry_scale)
        self.temporal_feature_runtime_scale = float(temporal_feature_scale)

    def _raw_state(self) -> dict[str, torch.Tensor]:
        residuals = self._token_residuals()
        names = (
            "raw_xy0",
            "raw_xy_coeff",
            "raw_z0",
            "raw_z_coeff",
            "raw_radii0",
            "raw_radii_coeff",
            "raw_densities0",
            "raw_densities_coeff",
            "raw_features0",
            "raw_features_coeff",
            "raw_normals0",
            "raw_normals_coeff",
            "raw_tangents0",
            "raw_tangents_coeff",
            "raw_texel_sites0",
            "raw_texel_sites_coeff",
        )
        state = {}
        for name in names:
            init = getattr(self, f"init_{name}")
            state[name] = self._mask_temporal_coeff(name, init + residuals.get(name, init.new_zeros(())))
        return state

    def _decode_temporal(self, base: torch.Tensor, coeff: torch.Tensor, frame_indices: torch.Tensor | None) -> torch.Tensor:
        if frame_indices is None:
            basis = self.frame_basis.to(device=base.device, dtype=base.dtype)
        else:
            basis = self.frame_basis[frame_indices.to(device=self.frame_basis.device, dtype=torch.long)].to(
                device=base.device,
                dtype=base.dtype,
            )
        return base[None].expand(basis.shape[0], *base.shape) + torch.einsum("tk,k...->t...", basis, coeff)

    def raw_parameter_tensors(
        self, frame_indices: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self._raw_state()
        return (
            self._decode_temporal(state["raw_xy0"], state["raw_xy_coeff"], frame_indices),
            self._decode_temporal(state["raw_z0"], state["raw_z_coeff"], frame_indices),
            self._decode_temporal(state["raw_radii0"], state["raw_radii_coeff"], frame_indices),
            self._decode_temporal(state["raw_densities0"], state["raw_densities_coeff"], frame_indices),
            self._decode_temporal(state["raw_features0"], state["raw_features_coeff"], frame_indices),
            self._decode_temporal(state["raw_normals0"], state["raw_normals_coeff"], frame_indices),
            self._decode_temporal(state["raw_tangents0"], state["raw_tangents_coeff"], frame_indices),
            self._decode_temporal(state["raw_texel_sites0"], state["raw_texel_sites_coeff"], frame_indices),
        )

    def decoded_parameters(
        self, frame_indices: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_xy, raw_z, raw_radii, raw_densities, raw_features, raw_normals, _raw_tangents, _raw_texel_sites = (
            self.raw_parameter_tensors(frame_indices)
        )
        xy = torch.tanh(raw_xy) * self.xy_extent
        z = self.z_min + torch.sigmoid(raw_z) * (self.z_max - self.z_min)
        points = torch.cat([xy, z], dim=-1)
        radii = F.softplus(raw_radii, beta=POWERFOAM_SOFTPLUS_BETA) + self.radius_min
        densities = F.softplus(raw_densities, beta=POWERFOAM_SOFTPLUS_BETA)
        normals = F.normalize(raw_normals, dim=-1, eps=1.0e-6)
        return points, radii, densities, raw_features, normals

    def decoded_texel_sites(self, frame_indices: torch.Tensor | None = None) -> torch.Tensor:
        *_prefix, raw_texel_sites = self.raw_parameter_tensors(frame_indices)
        return self.texel_site_scale * torch.tanh(raw_texel_sites)

    def decoded_surface_frame(
        self,
        normals: torch.Tensor,
        frame_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _raw_xy, _raw_z, _raw_radii, _raw_densities, _raw_features, _raw_normals, raw_tangents, _raw_texel_sites = (
            self.raw_parameter_tensors(frame_indices)
        )
        return orthonormal_surface_frame(normals, raw_tangents)

    def optimizer_param_groups(self, train_cfg: dict[str, Any]) -> list[dict[str, object]]:
        lr = float(train_cfg["lr"])
        groups: list[dict[str, object]] = [
            {"params": [self.tokens], "lr": float(train_cfg["token_lr_multiplier"]) * lr, "name": "tokens"},
            {
                "params": list(self.decoder.parameters()),
                "lr": float(train_cfg["decoder_lr_multiplier"]) * lr,
                "name": "decoder",
            },
        ]
        camera_group = camera_param_group(self.camera_decoder, train_cfg)
        if camera_group is not None:
            groups.append(camera_group)
        return groups

    def temporal_regularization(self, loss_cfg: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        points, radii, densities, features, _normals = self.decoded_parameters()
        state = self._raw_state()
        coeffs = [
            state["raw_xy_coeff"],
            state["raw_z_coeff"],
            state["raw_radii_coeff"],
            state["raw_densities_coeff"],
            state["raw_features_coeff"],
            state["raw_normals_coeff"],
            state["raw_tangents_coeff"],
            state["raw_texel_sites_coeff"],
        ]
        terms = {
            "temporal_center_accel": temporal_accel(points),
            "temporal_radius_accel": temporal_accel(radii),
            "temporal_density_accel": temporal_accel(densities),
            "temporal_feature_accel": temporal_accel(features),
            "temporal_coeff_l2": torch.stack([coeff.square().mean() for coeff in coeffs]).mean(),
        }
        loss = points.new_zeros(())
        for key, value in terms.items():
            loss = loss + float(loss_cfg[f"{key}_weight"]) * value
        camera_loss, camera_terms = camera_regularization(self.camera_decoder, loss_cfg)
        if camera_loss is not None:
            loss = loss + camera_loss
            terms.update(camera_terms)
        return loss, terms

    @torch.no_grad()
    def parameter_drift_metrics(self) -> dict[str, float]:
        points, radii, densities, features, normals = self.decoded_parameters()
        texel_sites = self.decoded_texel_sites()
        center_offset = points - self.initial_points.to(points.device)
        center_delta = torch.linalg.vector_norm(center_offset, dim=-1)
        metrics = {
            "state_mean_center_delta": float(center_delta.mean().cpu()),
            "state_p95_center_delta": float(center_delta.flatten().quantile(0.95).cpu()),
            "state_max_center_delta": float(center_delta.max().cpu()),
            "state_mean_xy_delta": float(torch.linalg.vector_norm(center_offset[..., :2], dim=-1).mean().cpu()),
            "state_mean_z_delta": float(center_offset[..., 2].abs().mean().cpu()),
            "state_mean_radius_delta": float((radii - self.initial_radii.to(radii.device)).abs().mean().cpu()),
            "state_mean_density_delta": float((densities - self.initial_densities.to(densities.device)).abs().mean().cpu()),
            "state_mean_feature_delta": float((features - self.initial_features.to(features.device)).abs().mean().cpu()),
            "state_mean_normal_delta": float((normals - self.initial_normals.to(normals.device)).norm(dim=-1).mean().cpu()),
            "state_mean_texel_site_delta": float(
                (texel_sites - self.initial_texel_sites.to(texel_sites.device)).norm(dim=-1).mean().cpu()
            ),
            "state_token_rms": float(self.tokens.detach().square().mean().sqrt().cpu()),
            "state_static_cell_count": float(self.static_cell_count),
            "state_dynamic_cell_count": float(self.dynamic_cell_count if self.static_dynamic_split else self.cell_count),
            "state_dynamic_cell_fraction": float(
                (self.dynamic_cell_count if self.static_dynamic_split else self.cell_count) / self.cell_count
            ),
        }
        camera_to_world = None if self.camera_decoder is None else self.camera_decoder.camera_to_world_matrices()
        metrics.update(
            temporal_motion_metrics(
                points,
                features,
                render_size=self.render_size,
                fov_degrees=self.fov_degrees,
                camera_to_world=camera_to_world,
            )
        )
        state = self._raw_state()
        coeff_abs = [
            state["raw_xy_coeff"].abs().mean(),
            state["raw_z_coeff"].abs().mean(),
            state["raw_radii_coeff"].abs().mean(),
            state["raw_densities_coeff"].abs().mean(),
            state["raw_features_coeff"].abs().mean(),
            state["raw_normals_coeff"].abs().mean(),
            state["raw_tangents_coeff"].abs().mean(),
            state["raw_texel_sites_coeff"].abs().mean(),
        ]
        metrics["state_mean_temporal_coeff_abs"] = float(torch.stack(coeff_abs).mean().cpu())
        metrics.update(compact_camera_metrics(self.camera_decoder))
        return metrics

    def decoded_camera_rays(self, frame_indices: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        return decoded_powerfoam_rays(
            self.camera_decoder,
            self.rays,
            frame_indices.to(device=next(self.parameters()).device, dtype=torch.long),
            height=self.render_size,
            width=self.render_size,
            device=next(self.parameters()).device,
            dtype=dtype,
        )

    def forward(self, frame_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if next(self.parameters()).device.type != "mps":
            raise RuntimeError("TokenDynamicPowerFoamFeatures requires an MPS device")
        frame_indices = frame_indices.to(device=next(self.parameters()).device, dtype=torch.long)
        points, radii, densities, features, normals = self.decoded_parameters(frame_indices)
        texel_sites = self.decoded_texel_sites(frame_indices)
        tangents, bitangents = self.decoded_surface_frame(normals, frame_indices)
        rays = self.rays.to(device=points.device, dtype=points.dtype)
        camera_to_world = None if self.camera_decoder is None else self.camera_decoder.camera_to_world_matrices(frame_indices)
        renders = []
        alphas = []
        for local_index in range(int(frame_indices.numel())):
            point = points[local_index]
            normal = normals[local_index]
            tangent = tangents[local_index]
            bitangent = bitangents[local_index]
            if camera_to_world is not None:
                point, normal, tangent, bitangent = transform_powerfoam_frame_to_camera(
                    point,
                    normal,
                    tangent,
                    bitangent,
                    camera_to_world[local_index],
                )
            radius = radii[local_index]
            adjacency, offsets = build_csr_adjacency(
                point,
                radius,
                neighbor_count=self.neighbor_count,
                mode=self.adjacency_mode,
            )
            out, alpha = rasterize_power_foam_oriented_texel_surface(
                point,
                radius,
                densities[local_index],
                texel_sites[local_index],
                features[local_index],
                normal,
                adjacency,
                offsets,
                rays,
                self.raster_config,
                tangents=tangent,
                bitangents=bitangent,
            )
            renders.append(out.permute(0, 3, 1, 2))
            alphas.append(alpha)
        return torch.cat(renders, dim=0), torch.cat(alphas, dim=0)


FoamModel = DynamicMetalPowerFoamVideo | TokenDynamicPowerFoamFeatures


def init_colorizer_rgb_identity(colorizer: FeatureToColor) -> None:
    if colorizer.feature_dim < 3:
        raise ValueError("RGB identity colorizer init requires at least 3 feature channels")
    if colorizer.hidden_dim is not None or colorizer.pre_norm is not None:
        raise ValueError("colorize.init_rgb_identity requires hidden_dim=null and pre_norm=false")
    if not isinstance(colorizer.net, nn.Conv2d):
        raise ValueError("colorize.init_rgb_identity requires a single Conv2d colorizer")
    with torch.no_grad():
        colorizer.net.weight.zero_()
        colorizer.net.bias.zero_()
        colorizer.net.weight[0, 0, 0, 0] = 1.0
        colorizer.net.weight[1, 1, 0, 0] = 1.0
        colorizer.net.weight[2, 2, 0, 0] = 1.0


def build_colorizer(cfg: dict[str, Any], device: torch.device) -> FeatureToColor | None:
    if str(cfg["model"]["dynamic_mode"]) != TOKEN_RBF_FEATURE_MODE:
        return None
    colorizer = FeatureToColor(
        feature_dim=int(cfg["model"]["feature_dim"]),
        hidden_dim=None if cfg["colorize"]["hidden_dim"] is None else int(cfg["colorize"]["hidden_dim"]),
        activation=str(cfg["colorize"]["activation"]),
        pre_norm=bool(cfg["colorize"]["pre_norm"]),
        weight_init=str(cfg["colorize"]["weight_init"]),
        weight_init_gain=float(cfg["colorize"]["weight_init_gain"]),
        view_condition=str(cfg["colorize"]["view_condition"]),
        detach_view_condition=bool(cfg["colorize"]["detach_view_condition"]),
    )
    if bool(cfg["colorize"]["init_rgb_identity"]):
        init_colorizer_rgb_identity(colorizer)
    return colorizer.to(device)


def sample_background(
    render_cfg: dict[str, Any],
    *,
    phase: str,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    mode = str(render_cfg["train_background_mode"] if phase == "train" else render_cfg["eval_background_mode"])
    if mode == "none":
        return None
    if mode == "black":
        rgb = torch.zeros(batch_size, 3, 1, 1, device=device, dtype=dtype)
    elif mode == "white":
        rgb = torch.ones(batch_size, 3, 1, 1, device=device, dtype=dtype)
    elif mode == "fixed_rgb":
        rgb = torch.tensor(render_cfg["background"], device=device, dtype=dtype).view(1, 3, 1, 1)
        rgb = rgb.expand(batch_size, -1, -1, -1).contiguous()
    elif mode == "random_rgb":
        low = float(render_cfg["random_background_min"])
        high = float(render_cfg["random_background_max"])
        rgb = low + (high - low) * torch.rand(batch_size, 3, 1, 1, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown background mode {mode!r}")
    return rgb


def render_features_to_rgb(
    features: torch.Tensor,
    alpha: torch.Tensor,
    colorizer: FeatureToColor | None,
    background: torch.Tensor | None,
    *,
    normalize_features_by_alpha: bool = True,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    color_features = features
    if normalize_features_by_alpha and (colorizer is not None or background is not None):
        color_features = features / alpha.unsqueeze(1).clamp_min(float(eps))
    rgb = color_features if colorizer is None else colorizer(color_features)
    if background is None:
        return rgb
    return alpha.unsqueeze(1).to(device=rgb.device, dtype=rgb.dtype) * rgb + (1.0 - alpha.unsqueeze(1)) * background


def apply_training_stage(model: FoamModel, cfg: dict[str, Any], step: int) -> dict[str, float]:
    static_only_steps = int(cfg["train"]["static_only_steps"])
    no_repaint_steps = int(cfg["train"]["no_repaint_steps"])
    static_only = static_only_steps > 0 and int(step) <= static_only_steps
    no_repaint = no_repaint_steps > 0 and int(step) <= no_repaint_steps
    controls = {
        "stage_temporal_geometry_scale": 0.0 if static_only else 1.0,
        "stage_temporal_feature_scale": 0.0 if no_repaint else 1.0,
    }
    if hasattr(model, "set_training_controls"):
        model.set_training_controls(
            temporal_geometry_scale=controls["stage_temporal_geometry_scale"],
            temporal_feature_scale=controls["stage_temporal_feature_scale"],
        )
    return controls


@torch.no_grad()
def render_all(
    model: FoamModel,
    frame_count: int,
    batch_size: int,
    cfg: dict[str, Any],
    colorizer: FeatureToColor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    renders = []
    features_out = []
    alphas = []
    device = next(model.parameters()).device
    for start in range(0, frame_count, batch_size):
        indices = torch.arange(start, min(start + batch_size, frame_count), device=device)
        features, alpha = model(indices)
        background = sample_background(
            cfg["render"],
            phase="eval",
            batch_size=int(features.shape[0]),
            device=features.device,
            dtype=features.dtype,
        )
        rendered = render_features_to_rgb(
            features,
            alpha,
            colorizer,
            background,
            normalize_features_by_alpha=bool(cfg["render"]["normalize_features_by_alpha"]),
            eps=float(cfg["render"]["eps"]),
        )
        renders.append(rendered.clamp(0.0, 1.0).detach().cpu())
        features_out.append(features.detach().cpu())
        alphas.append(alpha.detach().cpu())
    return torch.cat(renders, dim=0), torch.cat(features_out, dim=0), torch.cat(alphas, dim=0)


def log_artifacts(
    model: FoamModel,
    colorizer: FeatureToColor | None,
    targets: torch.Tensor,
    cfg: dict[str, Any],
    step: int,
    output_dir: Path,
    wandb_run: Any | None,
) -> dict[str, float]:
    model.eval()
    if colorizer is not None:
        colorizer.eval()
    renders, features, alphas = render_all(
        model,
        targets.size(0),
        batch_size=max(1, int(cfg["train"]["frames_per_step"])),
        cfg=cfg,
        colorizer=colorizer,
    )
    frame_metrics = per_frame_reconstruction_metrics(renders, targets.cpu())
    metrics = {
        "eval_l1": F.l1_loss(renders, targets.cpu()).item(),
        "eval_mse": F.mse_loss(renders, targets.cpu()).item(),
        "eval_frame_psnr_mean": frame_metrics["summary"]["frame_psnr_mean"],
        "eval_frame_psnr_min": frame_metrics["summary"]["frame_psnr_min"],
        "eval_frame_snr_mean": frame_metrics["summary"]["frame_snr_mean"],
        "eval_frame_snr_min": frame_metrics["summary"]["frame_snr_min"],
        "eval_alpha_mean": float(alphas.mean().cpu()),
        "eval_feature_mean": float(features.mean().cpu()),
        "eval_feature_std": float(features.std().cpu()),
    }
    metrics.update(temporal_alpha_metrics(alphas))
    metrics.update(model.parameter_drift_metrics())
    (output_dir / f"per_frame_metrics_step_{step:04d}.json").write_text(
        json.dumps(frame_metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    preview = torch.cat([targets[0].cpu(), renders[0], alphas[0].unsqueeze(0).repeat(3, 1, 1)], dim=-1)
    save_png(output_dir / f"preview_step_{step:04d}.png", preview)
    if should_log_video(cfg, step):
        side_by_side = torch.cat([targets.cpu(), renders], dim=-1)
        fps = float(cfg.get("video_fps", 4.0))
        save_mp4(output_dir / f"render_step_{step:04d}.mp4", renders, fps=fps)
        save_mp4(output_dir / f"side_by_side_step_{step:04d}.mp4", side_by_side, fps=fps)
    if wandb_run is not None:
        fps = float(cfg.get("video_fps", 4.0))
        payload: dict[str, Any] = {
            "Eval/L1": metrics["eval_l1"],
            "Eval/MSE": metrics["eval_mse"],
            "Eval/AlphaMean": metrics["eval_alpha_mean"],
            "Eval/FeatureMean": metrics["eval_feature_mean"],
            "Eval/FeatureStd": metrics["eval_feature_std"],
            "State/MeanCenterDelta": metrics["state_mean_center_delta"],
            "State/P95CenterDelta": metrics["state_p95_center_delta"],
            "State/MaxCenterDelta": metrics["state_max_center_delta"],
            "State/MeanXYDelta": metrics["state_mean_xy_delta"],
            "State/MeanZDelta": metrics["state_mean_z_delta"],
            "State/MeanRadiusDelta": metrics["state_mean_radius_delta"],
            "State/MeanDensityDelta": metrics["state_mean_density_delta"],
            "State/MeanFeatureDelta": metrics["state_mean_feature_delta"],
            "State/MeanNormalDelta": metrics["state_mean_normal_delta"],
            "State/MeanTexelSiteDelta": metrics["state_mean_texel_site_delta"],
            "State/MeanTemporalXYDelta": metrics["state_mean_temporal_xy_delta"],
            "State/P95TemporalXYDelta": metrics["state_p95_temporal_xy_delta"],
            "State/MeanTemporalZDelta": metrics["state_mean_temporal_z_delta"],
            "State/MeanTemporalScreenDeltaPx": metrics["state_mean_temporal_screen_delta_px"],
            "State/P95TemporalScreenDeltaPx": metrics["state_p95_temporal_screen_delta_px"],
            "State/TemporalScreenValidFraction": metrics["state_temporal_screen_valid_fraction"],
            "State/MeanTemporalFeatureAbsDelta": metrics["state_mean_temporal_feature_abs_delta"],
            "Preview": make_preview_image(targets[0].cpu(), renders[0], caption=f"step {step}: GT | render"),
        }
        if "state_mean_temporal_coeff_abs" in metrics:
            payload["State/MeanTemporalCoeffAbs"] = metrics["state_mean_temporal_coeff_abs"]
        camera_payload_keys = {
            "state_camera_fov_degrees": "Camera/FovDegrees",
            "state_camera_radius": "Camera/Radius",
            "state_camera_rotation_delta_mean_degrees": "Camera/RotationDeltaMeanDegrees",
            "state_camera_translation_delta_mean": "Camera/TranslationDeltaMean",
            "state_camera_origin_delta_mean": "Camera/OriginDeltaMean",
            "state_camera_forward_delta_mean": "Camera/ForwardDeltaMean",
            "state_camera_global_residual_l2": "Camera/GlobalResidualL2",
        }
        for metric_key, wandb_key in camera_payload_keys.items():
            if metric_key in metrics:
                payload[wandb_key] = metrics[metric_key]
        if should_log_video(cfg, step):
            payload.update(build_validation_video_payload(renders, targets.cpu(), fps))
            payload["GT_Video"] = make_wandb_video(targets.cpu(), fps)
            payload["Alpha_Video"] = make_wandb_video(alphas.unsqueeze(1).repeat(1, 3, 1, 1), fps)
        wandb_run.log(payload, step=step)
    model.train()
    if colorizer is not None:
        colorizer.train()
    return metrics


def per_frame_reconstruction_metrics(renders: torch.Tensor, targets: torch.Tensor) -> dict[str, Any]:
    diff = renders.float() - targets.float()
    mse = diff.square().flatten(1).mean(dim=1)
    l1 = diff.abs().flatten(1).mean(dim=1)
    signal = targets.float().square().flatten(1).mean(dim=1)
    psnr = -10.0 * torch.log10(mse.clamp_min(1.0e-12))
    snr = 10.0 * torch.log10((signal / mse.clamp_min(1.0e-12)).clamp_min(1.0e-12))
    rows = []
    for frame_index in range(int(renders.shape[0])):
        rows.append(
            {
                "frame_index": frame_index,
                "mse": float(mse[frame_index].cpu()),
                "l1": float(l1[frame_index].cpu()),
                "psnr": float(psnr[frame_index].cpu()),
                "snr": float(snr[frame_index].cpu()),
                "signal_power": float(signal[frame_index].cpu()),
            }
        )
    return {
        "summary": {
            "frame_psnr_mean": float(psnr.mean().cpu()),
            "frame_psnr_min": float(psnr.min().cpu()),
            "frame_snr_mean": float(snr.mean().cpu()),
            "frame_snr_min": float(snr.min().cpu()),
            "frame_l1_mean": float(l1.mean().cpu()),
            "frame_l1_max": float(l1.max().cpu()),
            "frame_mse_mean": float(mse.mean().cpu()),
            "frame_mse_max": float(mse.max().cpu()),
        },
        "per_frame": rows,
    }


def temporal_alpha_metrics(alphas: torch.Tensor) -> dict[str, float]:
    if alphas.shape[0] < 2:
        return {
            "eval_mean_temporal_alpha_delta": 0.0,
            "eval_max_temporal_alpha_delta": 0.0,
            "eval_mean_temporal_support_delta": 0.0,
        }
    delta = (alphas[1:] - alphas[:-1]).abs()
    support = alphas > 1.0e-4
    support_delta = (support[1:].float() - support[:-1].float()).abs()
    return {
        "eval_mean_temporal_alpha_delta": float(delta.mean().cpu()),
        "eval_max_temporal_alpha_delta": float(delta.max().cpu()),
        "eval_mean_temporal_support_delta": float(support_delta.mean().cpu()),
    }


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def dynamic_geometry_summary(
    cfg: dict[str, Any],
    history: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    eval_rows = [row for row in history if row.get("kind") == "eval"]
    train_rows = [row for row in history if row.get("kind") == "train"]
    initial = eval_rows[0] if eval_rows else {}
    final = eval_rows[-1] if eval_rows else {}
    best = min(eval_rows, key=lambda row: float(row.get("eval_l1", float("inf"))), default={})
    return {
        "schema_version": "dynamic_powerfoam_geometry_summary_v1",
        "status": "ok" if final else "missing_eval",
        "output_dir": str(output_dir),
        "config": {
            "video_path": str(cfg["data"]["video_path"]),
            "frames": int(cfg["data"]["max_frames"]),
            "render_size": int(cfg["render"]["render_size"]),
            "steps": int(cfg["train"]["steps"]),
            "cells": int(cfg["model"]["cells"]),
            "dynamic_mode": str(cfg["model"]["dynamic_mode"]),
            "dynamic_centers": bool(cfg["model"]["dynamic_centers"]),
            "dynamic_radii": bool(cfg["model"]["dynamic_radii"]),
            "dynamic_densities": bool(cfg["model"]["dynamic_densities"]),
            "dynamic_features": bool(cfg["model"]["dynamic_features"]),
            "dynamic_normals": bool(cfg["model"]["dynamic_normals"]),
            "dynamic_texel_sites": bool(cfg["model"]["dynamic_texel_sites"]),
            "static_dynamic_split": bool(cfg["model"]["static_dynamic_split"]),
            "dynamic_cells": cfg["model"]["dynamic_cells"],
            "dynamic_cell_fraction": float(cfg["model"]["dynamic_cell_fraction"]),
            "video_init_mode": str(cfg["model"]["video_init_mode"]),
            "camera_enabled": bool(cfg["camera"]["enabled"]),
            "camera_mode": str(cfg["camera"]["mode"]),
            "camera_base_path_mode": str(cfg["camera"]["base_path_mode"]),
            "static_only_steps": int(cfg["train"]["static_only_steps"]),
            "no_repaint_steps": int(cfg["train"]["no_repaint_steps"]),
        },
        "initial_eval": initial,
        "final_eval": final,
        "best_eval": best,
        "last_train": train_rows[-1] if train_rows else {},
        "motion_vs_repaint": {
            "state_mean_temporal_screen_delta_px": float(final.get("state_mean_temporal_screen_delta_px", 0.0)),
            "state_p95_temporal_screen_delta_px": float(final.get("state_p95_temporal_screen_delta_px", 0.0)),
            "state_temporal_screen_valid_fraction": float(final.get("state_temporal_screen_valid_fraction", 0.0)),
            "state_mean_temporal_feature_abs_delta": float(final.get("state_mean_temporal_feature_abs_delta", 0.0)),
            "eval_mean_temporal_alpha_delta": float(final.get("eval_mean_temporal_alpha_delta", 0.0)),
            "eval_mean_temporal_support_delta": float(final.get("eval_mean_temporal_support_delta", 0.0)),
            "state_camera_rotation_delta_mean_degrees": float(
                final.get("state_camera_rotation_delta_mean_degrees", 0.0)
            ),
            "state_camera_translation_delta_mean": float(final.get("state_camera_translation_delta_mean", 0.0)),
        },
        "artifacts": {
            "history": str(output_dir / "train_metrics_history.jsonl"),
            "checkpoint": str(output_dir / "checkpoint_final.pt"),
            "final_per_frame_metrics": str(output_dir / f"per_frame_metrics_step_{int(final.get('step', 0)):04d}.json")
            if final
            else None,
            "final_render": str(output_dir / f"render_step_{int(final.get('step', 0)):04d}.mp4") if final else None,
            "final_side_by_side": str(output_dir / f"side_by_side_step_{int(final.get('step', 0)):04d}.mp4") if final else None,
        },
    }


def run_training(config: dict[str, Any]) -> None:
    cfg = resolve_config(config)
    torch.manual_seed(int(cfg["train"]["seed"]))
    device = resolve_device(str(cfg["train"]["device"]))
    if device.type != "mps" or not torch.backends.mps.is_available():
        raise RuntimeError("dynamic_powerfoam_metal requires torch MPS")

    output_dir: Path = cfg["logging"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(json.dumps(serialize_config_value(cfg), indent=2) + "\n")
    history_path = output_dir / "train_metrics_history.jsonl"
    summary_path = output_dir / "dynamic_geometry_summary.json"
    for path in (history_path, summary_path):
        if path.exists():
            path.unlink()
    metrics_history: list[dict[str, Any]] = []

    sequence = load_video_sequence(
        cfg["data"]["video_path"],
        target_size=int(cfg["render"]["render_size"]),
        max_frames=int(cfg["data"]["max_frames"]),
        frame_source=str(cfg["data"]["frame_source"]),
    )
    targets = sequence.frames.to(device=device, dtype=torch.float32)
    cfg["video_fps"] = float(sequence.video_fps)
    wandb_run = init_wandb_run(cfg)
    camera_decoder = build_camera_decoder(cfg, frame_count=int(targets.size(0)))

    model_kwargs = {
        "frame_count": targets.size(0),
        "cell_count": int(cfg["model"]["cells"]),
        "render_size": int(cfg["render"]["render_size"]),
        "fov_degrees": float(cfg["render"]["fov_degrees"]),
        "neighbor_count": int(cfg["model"]["neighbor_count"]),
        "adjacency_mode": str(cfg["model"]["adjacency_mode"]),
        "time_basis_count": int(cfg["model"]["time_basis_count"]),
        "time_basis_sigma_scale": float(cfg["model"]["time_basis_sigma_scale"]),
        "temporal_init_mode": str(cfg["model"]["temporal_init_mode"]),
        "dynamic_centers": bool(cfg["model"]["dynamic_centers"]),
        "dynamic_radii": bool(cfg["model"]["dynamic_radii"]),
        "dynamic_densities": bool(cfg["model"]["dynamic_densities"]),
        "dynamic_features": bool(cfg["model"]["dynamic_features"]),
        "dynamic_normals": bool(cfg["model"]["dynamic_normals"]),
        "dynamic_texel_sites": bool(cfg["model"]["dynamic_texel_sites"]),
        "xy_extent": float(cfg["model"]["xy_extent"]),
        "z_min": float(cfg["model"]["z_min"]),
        "z_max": float(cfg["model"]["z_max"]),
        "radius_init": float(cfg["model"]["radius_init"]),
        "radius_min": float(cfg["model"]["radius_min"]),
        "radius_scale": float(cfg["model"]["radius_scale"]),
        "density_init": float(cfg["model"]["density_init"]),
        "normal_init_jitter": float(cfg["model"]["normal_init_jitter"]),
        "num_texel_sites": int(cfg["model"]["num_texel_sites"]),
        "texel_site_scale": float(cfg["model"]["texel_site_scale"]),
        "color_init_mode": str(cfg["model"]["color_init_mode"]),
        "video_init_mode": str(cfg["model"]["video_init_mode"]),
        "seed": int(cfg["train"]["seed"]),
        "init_frames": targets.detach().cpu() if bool(cfg["model"]["init_from_video"]) else None,
        "image_init_depth": None if cfg["model"]["image_init_depth"] is None else float(cfg["model"]["image_init_depth"]),
        "image_init_jitter": float(cfg["model"]["image_init_jitter"]),
        "raster_config": make_raster_config(cfg["render"]),
        "camera_decoder": camera_decoder,
    }
    if str(cfg["model"]["dynamic_mode"]) == TOKEN_RBF_FEATURE_MODE:
        model = TokenDynamicPowerFoamFeatures(
            **model_kwargs,
            feature_dim=int(cfg["model"]["feature_dim"]),
            feature_init_noise=float(cfg["model"]["feature_init_noise"]),
            feature_rgb_init=str(cfg["model"]["feature_rgb_init"]),
            token_dim=int(cfg["model"]["token_dim"]),
            token_hidden_dim=int(cfg["model"]["token_hidden_dim"]),
            token_hidden_layers=int(cfg["model"]["token_hidden_layers"]),
            token_init_std=float(cfg["model"]["token_init_std"]),
            token_output_init_std=float(cfg["model"]["token_output_init_std"]),
            token_point_residual_scale=float(cfg["model"]["token_point_residual_scale"]),
            token_z_residual_scale=float(cfg["model"]["token_z_residual_scale"]),
            token_radius_residual_scale=float(cfg["model"]["token_radius_residual_scale"]),
            token_density_residual_scale=float(cfg["model"]["token_density_residual_scale"]),
            token_feature_residual_scale=float(cfg["model"]["token_feature_residual_scale"]),
            token_normal_residual_scale=float(cfg["model"]["token_normal_residual_scale"]),
            token_texel_site_residual_scale=float(cfg["model"]["token_texel_site_residual_scale"]),
            token_temporal_residual_scale=float(cfg["model"]["token_temporal_residual_scale"]),
            static_dynamic_split=bool(cfg["model"]["static_dynamic_split"]),
            dynamic_cells=None if cfg["model"]["dynamic_cells"] is None else int(cfg["model"]["dynamic_cells"]),
            dynamic_cell_fraction=float(cfg["model"]["dynamic_cell_fraction"]),
        ).to(device)
    else:
        model = DynamicMetalPowerFoamVideo(
            **model_kwargs,
            dynamic_mode=str(cfg["model"]["dynamic_mode"]),
        ).to(device)
    colorizer = build_colorizer(cfg, device)
    param_groups = model.optimizer_param_groups(cfg["train"])
    if colorizer is not None:
        param_groups.append(
            {
                "params": list(colorizer.parameters()),
                "lr": float(cfg["train"]["colorize_lr_multiplier"]) * float(cfg["train"]["lr"]),
                "name": "colorize",
            }
        )
    optimizer = torch.optim.Adam(param_groups, lr=float(cfg["train"]["lr"]))

    print(
        {
            "arch": "dynamic_powerfoam_metal",
            "dynamic_mode": str(cfg["model"]["dynamic_mode"]),
            "device": str(device),
            "video_path": str(cfg["data"]["video_path"]),
            "frames": int(targets.size(0)),
            "render_size": int(cfg["render"]["render_size"]),
            "cells": int(cfg["model"]["cells"]),
            "feature_dim": int(cfg["model"]["feature_dim"]),
            "camera_mode": str(cfg["camera"]["mode"]),
            "camera_enabled": bool(cfg["camera"]["enabled"]),
            "train_background_mode": str(cfg["render"]["train_background_mode"]),
            "eval_background_mode": str(cfg["render"]["eval_background_mode"]),
            "neighbors": int(cfg["model"]["neighbor_count"]),
            "steps": int(cfg["train"]["steps"]),
        }
    )
    apply_training_stage(model, cfg, 0)
    initial_metrics = log_artifacts(model, colorizer, targets, cfg, 0, output_dir, wandb_run)
    initial_record = {"kind": "eval", "step": 0, **initial_metrics}
    metrics_history.append(initial_record)
    append_jsonl(history_path, initial_record)
    print({"step": 0, **initial_metrics})

    start_time = time.perf_counter()
    progress = trange(1, int(cfg["train"]["steps"]) + 1, desc=f"dynamic_powerfoam_{cfg['model']['dynamic_mode']}")
    for step in progress:
        stage_controls = apply_training_stage(model, cfg, step)
        frame_indices = torch.randint(0, targets.size(0), (int(cfg["train"]["frames_per_step"]),), device=device)
        target = targets[frame_indices]
        features, alpha = model(frame_indices)
        background = sample_background(
            cfg["render"],
            phase="train",
            batch_size=int(features.shape[0]),
            device=features.device,
            dtype=features.dtype,
        )
        rendered = render_features_to_rgb(
            features,
            alpha,
            colorizer,
            background,
            normalize_features_by_alpha=bool(cfg["render"]["normalize_features_by_alpha"]),
            eps=float(cfg["render"]["eps"]),
        )
        l1 = F.l1_loss(rendered, target)
        mse = F.mse_loss(rendered, target)
        _points, radii, densities, _features, _normals = model.decoded_parameters()
        temporal_loss, temporal_terms = model.temporal_regularization(cfg["losses"])
        loss = (
            float(cfg["losses"]["l1_weight"]) * l1
            + float(cfg["losses"]["mse_weight"]) * mse
            + float(cfg["losses"]["radius_l2_weight"]) * radii.square().mean()
            + float(cfg["losses"]["density_l2_weight"]) * densities.square().mean()
            + temporal_loss
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
                "temporal": float(temporal_loss.detach().cpu()),
                "elapsed_s": elapsed,
                **stage_controls,
            }
            for key in (
                "camera_rotation_l2",
                "camera_translation_l2",
                "camera_temporal_l2",
                "camera_global_l2",
            ):
                if key in temporal_terms:
                    train_metrics[key] = float(temporal_terms[key].detach().cpu())
            metrics_history.append({"kind": "train", **train_metrics})
            append_jsonl(history_path, {"kind": "train", **train_metrics})
            print(train_metrics)
            if wandb_run is not None:
                payload = {
                    "Train/Loss": train_metrics["loss"],
                    "Train/L1": train_metrics["l1"],
                    "Train/MSE": train_metrics["mse"],
                    "Train/TemporalLoss": train_metrics["temporal"],
                    "Train/AlphaMean": float(alpha.detach().mean().cpu()),
                    "Train/FeatureMean": float(features.detach().mean().cpu()),
                    "Train/FeatureStd": float(features.detach().std().cpu()),
                    "Train/TemporalCenterAccel": float(temporal_terms["temporal_center_accel"].detach().cpu()),
                    "Train/TemporalCoeffL2": float(temporal_terms["temporal_coeff_l2"].detach().cpu()),
                    "Timing/ElapsedSeconds": elapsed,
                    "Stage/TemporalGeometryScale": stage_controls["stage_temporal_geometry_scale"],
                    "Stage/TemporalFeatureScale": stage_controls["stage_temporal_feature_scale"],
                }
                if "camera_rotation_l2" in temporal_terms:
                    payload.update(
                        {
                            "Train/CameraRotationL2": train_metrics["camera_rotation_l2"],
                            "Train/CameraTranslationL2": train_metrics["camera_translation_l2"],
                            "Train/CameraTemporalL2": train_metrics["camera_temporal_l2"],
                            "Train/CameraGlobalL2": train_metrics["camera_global_l2"],
                        }
                    )
                wandb_run.log(payload, step=step)
        if step % int(cfg["logging"]["image_log_every"]) == 0 or (
            bool(cfg["logging"]["always_log_last_step"]) and step == int(cfg["train"]["steps"])
        ):
            metrics = log_artifacts(model, colorizer, targets, cfg, step, output_dir, wandb_run)
            eval_record = {"kind": "eval", "step": step, **metrics}
            metrics_history.append(eval_record)
            append_jsonl(history_path, eval_record)
            print({"step": step, **metrics})

    checkpoint: dict[str, Any] = {"model": model.state_dict(), "config": serialize_config_value(cfg)}
    if colorizer is not None:
        checkpoint["colorizer"] = colorizer.state_dict()
    torch.save(checkpoint, output_dir / "checkpoint_final.pt")
    summary_path.write_text(
        json.dumps(dynamic_geometry_summary(cfg, metrics_history, output_dir), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if wandb_run is not None:
        wandb_run.finish()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: PYTHONPATH=src/train uv run python src/train/train_dynamic_powerfoam_metal.py <config.jsonc>")
    run_training(load_config_file(sys.argv[1]))


if __name__ == "__main__":
    main()
