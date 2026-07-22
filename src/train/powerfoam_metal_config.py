from __future__ import annotations

from pathlib import Path
from typing import Any

from config_utils import apply_defaults, resolved_config
from paper_training_protocol import normalize_image_size, normalize_paper_stages


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
    "image_size": None,
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
PAPER_PROTOCOL_DEFAULTS = {
    "enabled": False,
    "same_time_count": 1,
    "local_time_count": 0,
    "local_time_radius": 0,
    "sampler_seed_offset": 7001,
    "stages": None,
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
    "eval_media_max_frames": None,
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


def resolve_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = resolved_config(config, ("data", "model", "render", "train", "losses", "logging"))
    cfg.setdefault("camera", {})
    cfg.setdefault("paper_protocol", {})
    apply_defaults(cfg["data"], DATA_DEFAULTS)
    apply_defaults(cfg["model"], MODEL_DEFAULTS)
    apply_defaults(cfg["render"], RENDER_DEFAULTS)
    apply_defaults(cfg["train"], TRAIN_DEFAULTS)
    apply_defaults(cfg["losses"], LOSS_DEFAULTS)
    apply_defaults(cfg["logging"], LOGGING_DEFAULTS)
    apply_defaults(cfg["paper_protocol"], PAPER_PROTOCOL_DEFAULTS)
    image_size = normalize_image_size(cfg["render"]["image_size"] or int(cfg["render"]["render_size"]))
    cfg["render"]["image_size"] = image_size.as_list()
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
    if cfg["logging"]["eval_media_max_frames"] is not None:
        cfg["logging"]["eval_media_max_frames"] = int(cfg["logging"]["eval_media_max_frames"])
        if cfg["logging"]["eval_media_max_frames"] < 1:
            raise ValueError("logging.eval_media_max_frames must be positive or null")
    if int(cfg["paper_protocol"]["same_time_count"]) < 1:
        raise ValueError("paper_protocol.same_time_count must be positive")
    if int(cfg["paper_protocol"]["local_time_count"]) < 0:
        raise ValueError("paper_protocol.local_time_count must be non-negative")
    if int(cfg["paper_protocol"]["local_time_radius"]) < 0:
        raise ValueError("paper_protocol.local_time_radius must be non-negative")
    if bool(cfg["paper_protocol"]["enabled"]):
        stages = normalize_paper_stages(
            cfg["paper_protocol"]["stages"],
            total_steps=int(cfg["train"]["steps"]),
            default_image_size=image_size,
            default_primitive_count=int(cfg["model"]["cells"]),
            default_frames_per_step=int(cfg["train"]["frames_per_step"]),
        )
        if stages[-1].image_size != image_size:
            raise ValueError("the final paper stage image size must match render.image_size")
        if stages[-1].primitive_count != int(cfg["model"]["cells"]):
            raise ValueError("the final paper stage primitive_count must match model.cells")
        if str(cfg["data"]["frame_source"]) != "multicam_val" and any(
            stage.image_size != image_size for stage in stages
        ):
            raise ValueError("progressive PowerFoam image stages currently require data.frame_source='multicam_val'")
        if int(cfg["model"]["resample_every"]) > 0:
            raise ValueError("paper stage capacity growth and model.resample_every cannot both be enabled")
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


__all__ = [
    "DATA_DEFAULTS",
    "HEIGHT_TEXEL_SURFACE_MODES",
    "LOGGING_DEFAULTS",
    "LOSS_DEFAULTS",
    "LR_GROUP_SPECS",
    "MODEL_DEFAULTS",
    "ORIENTED_TEXEL_SURFACE_MODES",
    "QUATERNION_TEXEL_SURFACE_MODES",
    "RENDER_DEFAULTS",
    "SV_TEXEL_SURFACE_MODES",
    "TEXEL_SURFACE_MODES",
    "TRAIN_DEFAULTS",
    "resolve_config",
]
