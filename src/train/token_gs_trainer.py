from __future__ import annotations

import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import torch
from clip_sampling import sample_clip_batch
from config_utils import apply_defaults, path_or_none, resolved_config
from fast_attn import (
    configure_fast_attn,
    fast_attn_context,
    pick_device,
)
from model_factories import build_colorizer, build_model_from_config
from objective.loss import objective_spec_from_loss_config
from objective.objective import RGBReconObjective
from objective.types import (
    BackgroundSample,
    CameraOwner,
    CameraRole,
    RasterizedView,
    RenderedView,
    RunPhase,
    TargetView,
    ViewRole,
)
from pipeline.diagnostics import (
    camera_state_payload,
    camera_state_summary_metrics,
    camera_temporal_payload,
    eval_metric_payload,
    temporal_similarity_payload,
)
from pipeline.losses import (
    build_bank_rate_loss as _build_bank_rate_loss_impl,
)
from pipeline.losses import (
    build_camera_loss as _build_camera_loss_impl,
)
from pipeline.losses import (
    temporal_recon_chunk_size as _temporal_recon_chunk_size_impl,
)
from pipeline.render import (
    colorize_view_dirs_for_features,
    gaussian_sequence_slice,
    render_clip_sequence,
    render_full_sequence as _render_full_sequence_impl,
)
from pipeline.validation_media import (
    single_cam_validation_video_payload,
    training_preview_payload,
)
from rendering import (
    build_or_reuse_grid,
    resize_images,
)
from render_dispatch import (
    decoded_token_count_from_model_config,
    pick_renderer_mode_from_config,
    token_summary_from_model_config,
)
from runtime_types import (
    CameraState,
    ClipBatch,
    GaussianSequence,
    RasterizedClip,
    RenderedClip,
    SequenceData,
    StepResult,
    build_step_result,
)
from sequence_data import (
    ManifestSequenceSampler,
    load_camera_sequence,
    load_manifest_sequences,
    load_uncalibrated_sequence,
    prepare_clip,
    resolve_frames_dir,
)
from temporal_sampling import normalize_frame_sampling_config, validate_frame_sampling_config
from train_devices import sync_torch_device
from train_logging import (
    finish_wandb_run,
    init_wandb_run,
    log_wandb_run_payload_lazy,
    scalar_payload as _scalar_payload_impl,
    should_log_image as _should_log_image,
    should_log_scalar as _should_log_scalar,
    should_log_video as _should_log_video,
)
from train_optim import adam_with_device_fused
from video_io import save_png, save_render_side_by_side_videos
from tqdm import tqdm

LOSS_OPTION_DEFAULTS = {
    "type": "l1_mse",
    "l1_weight": 1.0,
    "mse_weight": 0.2,
    "dssim_weight": 0.2,
    "dssim_backend": "torch",
    "ssim_window_size": 11,
    "ssim_c1": 0.0001,
    "ssim_c2": 0.0009,
    "camera_motion_weight": 0.0,
    "camera_temporal_weight": 0.0,
    "camera_global_weight": 0.0,
    "static_alpha_rate_weight": 0.0,
    "dynamic_alpha_rate_weight": 0.0,
    "dynamic_motion_rate_weight": 0.0,
    "dynamic_rotation_rate_weight": 0.0,
    "dynamic_alpha_time_rate_weight": 0.0,
    "vjepa_feature_weight": 0.0,
    "vjepa_feature_model_id": "vjepa2_1_vit_base_384",
    "vjepa_feature_dtype": "auto",
    "vjepa_feature_crop_size": None,
    "vjepa_feature_checkpoint_url": None,
    "vjepa_feature_checkpoint_key": None,
    "vjepa_feature_temporal_stride": 1,
    "vjepa_feature_normalize": True,
    "vjepa_feature_loss_type": "mse",
    "background": {
        "train_mode": "random_rgb",
        "eval_mode": "white",
        "fixed_rgb": (1.0, 1.0, 1.0),
    },
}


DATA_OPTION_DEFAULTS = {
    "manifest_path": None,
    "split": "train",
    "eval_manifest_path": None,
    "eval_split": "test",
    "eval_max_sequences": 1,
    "train_manifest_load_mode": "eager",
    "train_manifest_sample_mode": "random",
    "train_manifest_prefetch": 0,
    "frame_cache_dir": None,
    "image_crop_mode": "resize",
    "camera_json": None,
    "camera_image_size": 224,
    "camera_focal_mode": "median",
}


MODEL_OPTION_DEFAULTS = {
    "variant": "learned_time_orbit_path",
    "feature_dim": 3,
    "xy_extent": None,
    "z_min": None,
    "z_max": None,
    "scale_init": 0.05,
    "scale_init_log_jitter": 0.0,
    "opacity_init": None,
    "query_token_init_std": 0.02,
    "head_hidden_dim": 64,
    "head_hidden_layers": 1,
    "head_output_init_std": None,
    "position_init_extent_coverage": 0.0,
    "rotation_init": "random",
    "rgb_init": None,
    "rgb_init_min": 0.01,
    "rgb_init_max": 0.99,
    "video_encoder_backend": "local",
    "vjepa_model_id": None,
    "vjepa_feature_dim": None,
    "vjepa_freeze": True,
    "vjepa_attn_implementation": "sdpa",
    "vjepa_dtype": "auto",
    "vjepa_pretrained": True,
    "vjepa_crop_size": None,
    "vjepa_checkpoint_url": None,
    "video_feature_layers": None,
    "video_feature_channels": None,
    "video_feature_token_stride": 1,
    "video_feature_output_dtype": None,
    "camera_refine_with_decode_time": True,
    "time_fourier_bands": 8,
    "time_max_frequency": 128.0,
    "ray_condition_grid_size": 16,
    "static_tokens": None,
    "dynamic_tokens": None,
    "token_layout": None,
    "dynamic_time_basis_count": 8,
    "dynamic_time_max_frequency": 8.0,
    "dynamic_motion_extent": None,
    "dynamic_rotation_degrees": 10.0,
    "dynamic_alpha_logit_extent": 2.0,
    "dynamic_coeff_output_init_std": 1.0e-4,
    "free_frame_count": None,
    "free_time_interpolation": "nearest",
    "free_velocity_extent": 1.0,
    "free_velocity_init_std": 0.0,
    "free_opacity_slope_init_std": 0.0,
    "free_time_center": 0.5,
    "residual_output_init_std": 1.0e-3,
    "residual_xyz_raw_scale": 0.5,
    "residual_scale_log_scale": 0.5,
    "residual_rot_raw_scale": 0.5,
    "residual_opacity_logit_scale": 1.0,
    "residual_rgb_logit_scale": 1.0,
    "residual_head_input_norm": "rmsnorm",
}


CAMERA_OPTION_DEFAULTS = {
    "global_head": "legacy_orbit",
    "lens_model": "pinhole",
    "base_fov_degrees": 60.0,
    "base_radius": 3.0,
    "max_fov_delta_degrees": 15.0,
    "max_radius_scale": 1.5,
    "max_aspect_log_delta": 0.0,
    "max_principal_point_delta": 0.0,
    "distortion_max_abs": 0.0,
    "base_distortion": None,
    "max_rotation_degrees": 5.0,
    "max_translation_ratio": 0.2,
}


EXPORT_OPTION_DEFAULTS = {
    "enabled": False,
    "output_root": "outputs/browser_exports",
    "id": None,
    "sequence_index": 0,
    "window_start": 0,
}


def _resolve_amp_dtype(device: torch.device, dtype_name: str) -> torch.dtype:
    dtype_name = str(dtype_name).lower()
    if dtype_name == "auto":
        return torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    dtype_by_name = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype_name not in dtype_by_name:
        known = ", ".join(sorted(set(dtype_by_name) | {"auto"}))
        raise ValueError(f"Unknown train.amp_dtype={dtype_name!r}. Expected one of: {known}.")
    if device.type == "cuda" and dtype_by_name[dtype_name] == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError("train.amp_dtype='bfloat16' requested, but CUDA bf16 is not supported on this device.")
    return dtype_by_name[dtype_name]


TRAIN_TIMING_DEFAULTS = {
    "seed": 17,
    "profile_timing": False,
    "profile_timing_sync": True,
    "profile_timing_log_every": 1,
    "profile_backward_split": False,
    "render_size_schedule": None,
}


def normalize_render_size_schedule(value: Any) -> list[dict[str, int]]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError("train.render_size_schedule must be null or a list of {start_step, render_size} objects.")
    if not value:
        return []
    schedule: list[dict[str, int]] = []
    previous_step = -1
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError("train.render_size_schedule entries must be objects.")
        if "render_size" not in item:
            raise ValueError("train.render_size_schedule entries require render_size.")
        if "start_step" not in item:
            raise ValueError("train.render_size_schedule entries require start_step.")
        if isinstance(item["start_step"], bool) or isinstance(item["render_size"], bool):
            raise ValueError("train.render_size_schedule start_step and render_size must be integers, not booleans.")
        start_step = int(item["start_step"])
        render_size = int(item["render_size"])
        if start_step < 0:
            raise ValueError(f"train.render_size_schedule start_step must be >= 0, got {start_step}.")
        if render_size < 1:
            raise ValueError(f"train.render_size_schedule render_size must be >= 1, got {render_size}.")
        if index == 0 and start_step != 0:
            raise ValueError("train.render_size_schedule must start at step 0.")
        if start_step <= previous_step:
            raise ValueError("train.render_size_schedule start_step values must be strictly increasing.")
        schedule.append({"start_step": start_step, "render_size": render_size})
        previous_step = start_step
    return schedule


def resolve_config(config: dict[str, Any]) -> dict[str, Any]:
    if config is None:
        raise ValueError("A train config is required. Pass a JSONC path or config dict.")
    cfg = resolved_config(config, ("data", "model", "camera", "render", "train", "losses", "logging"))
    if "colorize" not in cfg:
        cfg["colorize"] = None
    if "feature_pca_log" not in cfg["logging"]:
        cfg["logging"]["feature_pca_log"] = False
    if "wandb_tags" not in cfg["logging"]:
        cfg["logging"]["wandb_tags"] = None
    if "log_initial_media" not in cfg["logging"]:
        cfg["logging"]["log_initial_media"] = True
    if "wandb_enabled" not in cfg["logging"]:
        cfg["logging"]["wandb_enabled"] = True
    if "wandb_mode" not in cfg["logging"]:
        cfg["logging"]["wandb_mode"] = None
    if "output_dir" not in cfg["logging"]:
        cfg["logging"]["output_dir"] = None
    cfg["logging"]["wandb_enabled"] = bool(cfg["logging"]["wandb_enabled"])
    cfg["logging"]["output_dir"] = path_or_none(cfg["logging"]["output_dir"])
    apply_defaults(cfg["data"], DATA_OPTION_DEFAULTS)
    cfg["data"]["sequence_dir"] = path_or_none(cfg["data"].get("sequence_dir"))
    cfg["data"]["frames_dir"] = path_or_none(cfg["data"]["frames_dir"])
    cfg["data"]["video_path"] = path_or_none(cfg["data"]["video_path"])
    cfg["data"]["manifest_path"] = path_or_none(cfg["data"]["manifest_path"])
    cfg["data"]["eval_manifest_path"] = path_or_none(cfg["data"]["eval_manifest_path"])
    cfg["data"]["frame_cache_dir"] = path_or_none(cfg["data"]["frame_cache_dir"])
    cfg["data"]["camera_json"] = path_or_none(cfg["data"]["camera_json"])
    cfg["data"]["split"] = str(cfg["data"]["split"])
    cfg["data"]["eval_split"] = str(cfg["data"]["eval_split"])
    cfg["data"]["eval_max_sequences"] = int(cfg["data"]["eval_max_sequences"])
    cfg["data"]["train_manifest_load_mode"] = str(cfg["data"]["train_manifest_load_mode"]).lower()
    if cfg["data"]["train_manifest_load_mode"] not in {"eager", "lazy"}:
        raise ValueError("data.train_manifest_load_mode must be one of: eager, lazy.")
    cfg["data"]["train_manifest_sample_mode"] = str(cfg["data"]["train_manifest_sample_mode"]).lower()
    if cfg["data"]["train_manifest_sample_mode"] not in {"random", "cycle"}:
        raise ValueError("data.train_manifest_sample_mode must be one of: random, cycle.")
    cfg["data"]["train_manifest_prefetch"] = int(cfg["data"]["train_manifest_prefetch"])
    if cfg["data"]["train_manifest_prefetch"] < 0:
        raise ValueError("data.train_manifest_prefetch must be >= 0.")
    cfg["data"]["image_crop_mode"] = str(cfg["data"]["image_crop_mode"]).lower()
    if cfg["data"]["image_crop_mode"] not in {"resize", "none", "center_square", "center_crop", "center"}:
        raise ValueError("data.image_crop_mode must be one of: resize, none, center_square, center_crop, center.")
    cfg["data"]["camera_image_size"] = int(cfg["data"]["camera_image_size"])
    cfg["data"]["camera_focal_mode"] = str(cfg["data"]["camera_focal_mode"]).lower()
    if cfg["data"]["manifest_path"] is None and cfg["data"]["sequence_dir"] is None:
        raise ValueError("config['data'] requires either sequence_dir or manifest_path.")
    if cfg["data"]["eval_max_sequences"] < 0:
        raise ValueError("config['data']['eval_max_sequences'] must be >= 0.")
    apply_defaults(cfg["model"], MODEL_OPTION_DEFAULTS)
    cfg["model"]["variant"] = str(cfg["model"]["variant"]).lower()
    cfg["model"]["feature_dim"] = int(cfg["model"]["feature_dim"])
    if cfg["model"]["feature_dim"] < 1:
        raise ValueError(f"model.feature_dim must be >= 1, got {cfg['model']['feature_dim']}.")
    cfg["model"]["video_encoder_backend"] = str(cfg["model"]["video_encoder_backend"]).lower()
    if cfg["model"]["video_encoder_backend"] not in {
        "local",
        "vjepa_hf",
        "vjepa_torchhub",
        "precomputed",
        "precomputed_ltx",
        "none",
    }:
        raise ValueError(
            f"Unknown model.video_encoder_backend={cfg['model']['video_encoder_backend']!r}. "
            "Expected one of: local, vjepa_hf, vjepa_torchhub, precomputed, precomputed_ltx, none."
        )
    if cfg["model"]["video_encoder_backend"] == "none" and cfg["model"]["variant"] not in {
        "free_splats",
        "free_linear_time_splats",
        "unconditioned_residual_free_bank",
        "unconditioned_tokens",
    }:
        raise ValueError("model.video_encoder_backend='none' is only valid for no-conditioning model variants.")
    if cfg["model"]["vjepa_feature_dim"] is not None:
        cfg["model"]["vjepa_feature_dim"] = int(cfg["model"]["vjepa_feature_dim"])
    if cfg["model"]["vjepa_crop_size"] is not None:
        cfg["model"]["vjepa_crop_size"] = int(cfg["model"]["vjepa_crop_size"])
    if cfg["model"]["vjepa_checkpoint_url"] is not None:
        cfg["model"]["vjepa_checkpoint_url"] = str(cfg["model"]["vjepa_checkpoint_url"])
    if cfg["model"]["free_frame_count"] is not None:
        cfg["model"]["free_frame_count"] = int(cfg["model"]["free_frame_count"])
    cfg["model"]["free_time_interpolation"] = str(cfg["model"]["free_time_interpolation"]).lower()
    cfg["model"]["free_velocity_extent"] = float(cfg["model"]["free_velocity_extent"])
    cfg["model"]["free_velocity_init_std"] = float(cfg["model"]["free_velocity_init_std"])
    cfg["model"]["free_opacity_slope_init_std"] = float(cfg["model"]["free_opacity_slope_init_std"])
    cfg["model"]["free_time_center"] = float(cfg["model"]["free_time_center"])
    cfg["model"]["residual_output_init_std"] = float(cfg["model"]["residual_output_init_std"])
    cfg["model"]["residual_xyz_raw_scale"] = float(cfg["model"]["residual_xyz_raw_scale"])
    cfg["model"]["residual_scale_log_scale"] = float(cfg["model"]["residual_scale_log_scale"])
    cfg["model"]["residual_rot_raw_scale"] = float(cfg["model"]["residual_rot_raw_scale"])
    cfg["model"]["residual_opacity_logit_scale"] = float(cfg["model"]["residual_opacity_logit_scale"])
    cfg["model"]["residual_rgb_logit_scale"] = float(cfg["model"]["residual_rgb_logit_scale"])
    cfg["model"]["residual_head_input_norm"] = str(cfg["model"]["residual_head_input_norm"]).lower()
    if cfg["model"]["video_feature_layers"] is not None:
        cfg["model"]["video_feature_layers"] = [str(name) for name in cfg["model"]["video_feature_layers"]]
    if cfg["model"]["video_feature_channels"] is not None:
        if not isinstance(cfg["model"]["video_feature_channels"], dict):
            raise ValueError("model.video_feature_channels must be a mapping of layer name to channel count.")
        cfg["model"]["video_feature_channels"] = {
            str(name): int(channels) for name, channels in cfg["model"]["video_feature_channels"].items()
        }
    cfg["model"]["video_feature_token_stride"] = int(cfg["model"]["video_feature_token_stride"])
    if cfg["model"]["video_feature_token_stride"] < 1:
        raise ValueError(
            "model.video_feature_token_stride must be >= 1, "
            f"got {cfg['model']['video_feature_token_stride']}."
        )
    if cfg["model"]["video_feature_output_dtype"] is not None:
        cfg["model"]["video_feature_output_dtype"] = str(cfg["model"]["video_feature_output_dtype"]).lower()
    cfg["model"]["camera_refine_with_decode_time"] = bool(cfg["model"]["camera_refine_with_decode_time"])
    if cfg["model"]["xy_extent"] is None:
        cfg["model"]["xy_extent"] = cfg["model"]["scene_extent"]
    if cfg["model"]["z_min"] is None:
        cfg["model"]["z_min"] = -cfg["model"]["scene_extent"]
    if cfg["model"]["z_max"] is None:
        cfg["model"]["z_max"] = cfg["model"]["scene_extent"]
    if cfg["model"]["token_layout"] is not None and not isinstance(cfg["model"]["token_layout"], dict):
        raise ValueError("model.token_layout must be an object or null.")
    has_token_layout = cfg["model"]["token_layout"] is not None
    has_static_dynamic_split = has_token_layout or (
        cfg["model"]["static_tokens"] is not None or cfg["model"]["dynamic_tokens"] is not None
    )
    cfg["model"]["use_static_dynamic_split"] = has_static_dynamic_split
    if has_static_dynamic_split:
        if cfg["model"]["variant"] in {
            "free_splats",
            "free_linear_time_splats",
            "residual_free_bank",
            "known_camera",
            "unconditioned_residual_free_bank",
        }:
            raise ValueError("static/dynamic splat split is not wired for this model.variant yet.")
        if cfg["model"]["variant"] == "token_to_pose_to_plucker":
            raise ValueError("static/dynamic splat split is not wired for token_to_pose_to_plucker yet.")
        total_tokens = int(cfg["model"]["tokens"])
        static_tokens = cfg["model"]["static_tokens"]
        dynamic_tokens = cfg["model"]["dynamic_tokens"]
        if has_token_layout:
            if (static_tokens is None) != (dynamic_tokens is None):
                raise ValueError("model.static_tokens and model.dynamic_tokens must be provided together with token_layout.")
            if static_tokens is not None:
                static_tokens = int(static_tokens)
                dynamic_tokens = int(dynamic_tokens)
                if static_tokens < 1 or dynamic_tokens < 1:
                    raise ValueError(
                        f"static/dynamic split requires positive static/dynamic tokens, "
                        f"got static_tokens={static_tokens}, dynamic_tokens={dynamic_tokens}."
                    )
                if static_tokens + dynamic_tokens > total_tokens:
                    raise ValueError(
                        f"token_layout static_tokens + dynamic_tokens cannot exceed model.tokens={total_tokens}, "
                        f"got {static_tokens} + {dynamic_tokens}."
                    )
                cfg["model"]["static_tokens"] = static_tokens
                cfg["model"]["dynamic_tokens"] = dynamic_tokens
        else:
            if static_tokens is None and dynamic_tokens is None:
                static_tokens = max(1, int(round(total_tokens * 0.75)))
                dynamic_tokens = total_tokens - static_tokens
            elif static_tokens is None:
                dynamic_tokens = int(dynamic_tokens)
                static_tokens = total_tokens - dynamic_tokens
            elif dynamic_tokens is None:
                static_tokens = int(static_tokens)
                dynamic_tokens = total_tokens - static_tokens
            else:
                static_tokens = int(static_tokens)
                dynamic_tokens = int(dynamic_tokens)
            if static_tokens < 1 or dynamic_tokens < 1:
                raise ValueError(
                    f"static/dynamic split requires positive static/dynamic tokens, "
                    f"got static_tokens={static_tokens}, dynamic_tokens={dynamic_tokens}."
                )
            if static_tokens + dynamic_tokens != total_tokens:
                raise ValueError(
                    f"static_tokens + dynamic_tokens must equal model.tokens={total_tokens}, "
                    f"got {static_tokens} + {dynamic_tokens}."
                )
            cfg["model"]["static_tokens"] = static_tokens
            cfg["model"]["dynamic_tokens"] = dynamic_tokens
    apply_defaults(cfg["camera"], CAMERA_OPTION_DEFAULTS)
    cfg["camera"]["global_head"] = str(cfg["camera"]["global_head"]).lower()
    cfg["camera"]["lens_model"] = str(cfg["camera"]["lens_model"]).lower()
    if cfg["camera"]["global_head"] not in {"legacy_orbit", "legacy_pinhole", "simple_pinhole", "central_lens"}:
        raise ValueError(
            f"Unknown camera.global_head={cfg['camera']['global_head']!r}. "
            "Expected legacy_orbit or central_lens."
        )
    if cfg["camera"]["lens_model"] not in {"pinhole", "radial_tangential", "opencv_fisheye"}:
        raise ValueError(
            f"Unknown camera.lens_model={cfg['camera']['lens_model']!r}. "
            "Expected pinhole, radial_tangential, or opencv_fisheye."
        )
    if cfg["camera"]["global_head"] in {"legacy_orbit", "legacy_pinhole", "simple_pinhole"}:
        if cfg["camera"]["lens_model"] != "pinhole":
            raise ValueError("camera.global_head='legacy_orbit' requires camera.lens_model='pinhole'.")
    apply_defaults(cfg["losses"], LOSS_OPTION_DEFAULTS)
    apply_defaults(cfg["losses"]["background"], LOSS_OPTION_DEFAULTS["background"])
    cfg["losses"]["background"]["train_mode"] = str(cfg["losses"]["background"]["train_mode"]).lower()
    cfg["losses"]["background"]["eval_mode"] = str(cfg["losses"]["background"]["eval_mode"]).lower()
    cfg["losses"]["type"] = str(cfg["losses"]["type"]).lower()
    if cfg["losses"]["type"] not in {"standard_gs", "l1_mse", "l1", "mse"}:
        raise ValueError(
            f"Unknown losses.type={cfg['losses']['type']!r}. Expected one of: standard_gs, l1_mse, l1, mse."
        )
    cfg["losses"]["dssim_backend"] = str(cfg["losses"]["dssim_backend"]).lower()
    if cfg["losses"]["dssim_backend"] not in {"torch", "metal"}:
        raise ValueError(
            f"Unknown losses.dssim_backend={cfg['losses']['dssim_backend']!r}. Expected one of: torch, metal."
        )
    window_size = int(cfg["losses"]["ssim_window_size"])
    if window_size < 1 or window_size % 2 != 1:
        raise ValueError(f"losses.ssim_window_size must be a positive odd integer, got {window_size}.")
    cfg["losses"]["ssim_window_size"] = window_size
    cfg["losses"]["vjepa_feature_weight"] = float(cfg["losses"]["vjepa_feature_weight"])
    cfg["losses"]["vjepa_feature_model_id"] = str(cfg["losses"]["vjepa_feature_model_id"])
    cfg["losses"]["vjepa_feature_dtype"] = str(cfg["losses"]["vjepa_feature_dtype"]).lower()
    if cfg["losses"]["vjepa_feature_crop_size"] is not None:
        cfg["losses"]["vjepa_feature_crop_size"] = int(cfg["losses"]["vjepa_feature_crop_size"])
    if cfg["losses"]["vjepa_feature_checkpoint_url"] is not None:
        cfg["losses"]["vjepa_feature_checkpoint_url"] = str(cfg["losses"]["vjepa_feature_checkpoint_url"])
    if cfg["losses"]["vjepa_feature_checkpoint_key"] is not None:
        cfg["losses"]["vjepa_feature_checkpoint_key"] = str(cfg["losses"]["vjepa_feature_checkpoint_key"])
    cfg["losses"]["vjepa_feature_temporal_stride"] = int(cfg["losses"]["vjepa_feature_temporal_stride"])
    if cfg["losses"]["vjepa_feature_temporal_stride"] < 1:
        raise ValueError("losses.vjepa_feature_temporal_stride must be >= 1.")
    cfg["losses"]["vjepa_feature_normalize"] = bool(cfg["losses"]["vjepa_feature_normalize"])
    cfg["losses"]["vjepa_feature_loss_type"] = str(cfg["losses"]["vjepa_feature_loss_type"]).lower()
    if cfg["losses"]["vjepa_feature_loss_type"] not in {"mse", "l1", "smooth_l1"}:
        raise ValueError("losses.vjepa_feature_loss_type must be one of: mse, l1, smooth_l1.")
    if "near_plane" not in cfg["render"]:
        cfg["render"]["near_plane"] = 1.0e-4
    if "camera_projection" not in cfg["render"]:
        cfg["render"]["camera_projection"] = "auto"
    cfg["render"]["camera_projection"] = str(cfg["render"]["camera_projection"]).lower()
    if cfg["render"]["camera_projection"] == "legacy":
        cfg["render"]["camera_projection"] = "legacy_pinhole"
    if cfg["render"]["camera_projection"] not in {"auto", "legacy_pinhole", "camera_model"}:
        raise ValueError(
            f"Unknown render.camera_projection={cfg['render']['camera_projection']!r}. "
            "Expected auto, legacy_pinhole, or camera_model."
        )
    if cfg["camera"]["lens_model"] != "pinhole" and cfg["render"]["camera_projection"] == "legacy_pinhole":
        raise ValueError("Non-pinhole camera.lens_model requires render.camera_projection='auto' or 'camera_model'.")
    if "fast_mac" not in cfg["render"]:
        cfg["render"]["fast_mac"] = None
    export_cfg = cfg.get("export", False)
    if isinstance(export_cfg, bool):
        export_cfg = {"enabled": export_cfg}
    elif isinstance(export_cfg, dict):
        export_cfg = dict(export_cfg)
    else:
        raise ValueError("config['export'] must be a boolean or object when provided.")
    apply_defaults(export_cfg, EXPORT_OPTION_DEFAULTS)
    export_cfg["enabled"] = bool(export_cfg["enabled"])
    export_cfg["output_root"] = path_or_none(export_cfg["output_root"])
    if export_cfg["output_root"] is None:
        raise ValueError("export.output_root cannot be null.")
    if export_cfg["id"] is not None:
        export_cfg["id"] = str(export_cfg["id"])
    export_cfg["sequence_index"] = int(export_cfg["sequence_index"])
    export_cfg["window_start"] = int(export_cfg["window_start"])
    if export_cfg["sequence_index"] < 0:
        raise ValueError("export.sequence_index must be >= 0.")
    if export_cfg["window_start"] < 0:
        raise ValueError("export.window_start must be >= 0.")
    cfg["export"] = export_cfg

    if "amp_dtype" not in cfg["train"]:
        cfg["train"]["amp_dtype"] = "auto"
    cfg["train"]["amp_dtype"] = str(cfg["train"]["amp_dtype"]).lower()
    apply_defaults(cfg["train"], TRAIN_TIMING_DEFAULTS)
    cfg["train"]["seed"] = int(cfg["train"]["seed"])
    cfg["train"]["profile_timing"] = bool(cfg["train"]["profile_timing"])
    cfg["train"]["profile_timing_sync"] = bool(cfg["train"]["profile_timing_sync"])
    cfg["train"]["profile_timing_log_every"] = int(cfg["train"]["profile_timing_log_every"])
    cfg["train"]["profile_backward_split"] = bool(cfg["train"]["profile_backward_split"])
    cfg["train"]["render_size_schedule"] = normalize_render_size_schedule(cfg["train"]["render_size_schedule"])
    if cfg["train"]["profile_timing_log_every"] < 1:
        raise ValueError(
            f"train.profile_timing_log_every must be >= 1, got {cfg['train']['profile_timing_log_every']}."
        )
    cfg["train"]["frame_sampling"] = normalize_frame_sampling_config(cfg["train"].get("frame_sampling"))
    validate_frame_sampling_config(cfg["train"]["frame_sampling"], int(cfg["model"]["train_frame_count"]))
    if cfg["train"]["recon_backward_strategy"] not in {"framewise", "microbatch", "batched"}:
        raise ValueError(
            f"Unsupported recon_backward_strategy={cfg['train']['recon_backward_strategy']!r}. "
            "Expected one of: framewise, microbatch, batched."
        )
    if int(cfg["train"]["temporal_microbatch_size"]) < 1:
        raise ValueError(f"temporal_microbatch_size must be >= 1, got {cfg['train']['temporal_microbatch_size']}.")
    if int(cfg["render"]["render_size"]) < 1:
        raise ValueError(f"render_size must be >= 1, got {cfg['render']['render_size']}.")
    return cfg


def save_token_gs_eval_media(
    output_dir: Path,
    step: int,
    renders: torch.Tensor,
    targets: torch.Tensor,
    *,
    fps: float,
    save_videos: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    preview = torch.cat([targets[0].detach().cpu(), renders[0].detach().cpu()], dim=-1)
    save_png(output_dir / f"preview_step_{step:04d}.png", preview)
    if save_videos:
        save_render_side_by_side_videos(output_dir, step, renders, targets, fps=fps)


class Trainer:
    @classmethod
    def resolve_config(cls, config: dict[str, Any]) -> dict[str, Any]:
        return resolve_config(config)

    def __init__(self, config: dict[str, Any]) -> None:
        self.cfg = self.resolve_config(config)
        self.data_cfg = self.cfg["data"]
        self.model_cfg = self.cfg["model"]
        self.render_cfg = self.cfg["render"]
        self.train_cfg = self.cfg["train"]
        self.loss_cfg = self.cfg["losses"]
        self.logging_cfg = self.cfg["logging"]
        self.export_cfg = self.cfg["export"]
        self.base_render_size = int(self.render_cfg["render_size"])
        self.render_size = self.base_render_size
        self.train_sequence_sampler: ManifestSequenceSampler | None = None
        self.lazy_train_entries: list[dict[str, Any]] = []

        torch.manual_seed(int(self.train_cfg["seed"]))
        self.device = pick_device()
        print(f"Using device: {self.device}")

        self.train_sequences = self.load_train_sequences()
        self.train_sequence_count = (
            self.train_sequence_sampler.sequence_count
            if self.train_sequence_sampler is not None
            else len(self.train_sequences)
        )
        self.sequence_data = self.train_sequences[0]
        self.num_frames = self.sequence_data.frame_count
        self.eval_sequences = self.load_eval_sequences()
        self.validate_train_sequences()

        if self.train_sequence_sampler is not None:
            train_frames = self.train_sequence_sampler.frame_counts(
                fallback_frame_count=int(self.model_cfg["train_frame_count"])
            )
        else:
            train_frames = [sequence.frame_count for sequence in self.train_sequences]
        eval_frames = [sequence.frame_count for sequence in self.eval_sequences]
        print(
            f"Loaded {self.train_sequence_count} train sequence(s), "
            f"frames min/median/max={min(train_frames)}/{sorted(train_frames)[len(train_frames) // 2]}/{max(train_frames)}"
        )
        if self.train_sequence_sampler is not None and self.train_sequence_sampler.is_lazy:
            print("Train manifest load mode: lazy (sampling one sequence from disk per step).")
            print(f"Train manifest sample mode: {self.data_cfg['train_manifest_sample_mode']}.")
        if self.eval_sequences:
            print(
                f"Loaded {len(self.eval_sequences)} eval sequence(s), "
                f"frames min/median/max={min(eval_frames)}/{sorted(eval_frames)[len(eval_frames) // 2]}/{max(eval_frames)}"
            )
        print(
            f"Primary train source: {self.sequence_data.source_path} "
            f"(source={self.sequence_data.frame_source}, source_total={self.sequence_data.all_frame_count})"
        )
        self.on_sequences_loaded()

        self.wandb_run = init_wandb_run(self.cfg)

        self.model = build_model_from_config(self.cfg).to(self.device)
        self.model.train()
        feature_dim = int(self.model_cfg["feature_dim"])
        colorizer = build_colorizer(self.cfg["colorize"], feature_dim=feature_dim)
        self.colorize = None if colorizer.module is None else colorizer.module.to(self.device)
        self.feature_pca_log = bool(self.logging_cfg["feature_pca_log"])
        if self.feature_pca_log and feature_dim == 3:
            raise ValueError("logging.feature_pca_log=true requires model.feature_dim != 3.")
        trainable_parameters = list(self.model.parameters())
        if self.colorize is not None:
            trainable_parameters = trainable_parameters + list(self.colorize.parameters())
        self.optimizer = adam_with_device_fused(
            trainable_parameters,
            lr=self.train_cfg["lr"],
            device=self.device,
        )

        self.dense_grid = build_or_reuse_grid(self.render_size, self.render_size, self.device)
        self._dense_grid_by_render_size = {int(self.render_size): self.dense_grid}
        self.amp_available = bool(
            self.train_cfg["amp"] and torch.amp.autocast_mode.is_autocast_available(self.device.type)
        )
        if self.train_cfg["amp"] and not self.amp_available:
            print(f"AMP requested but not available on device {self.device.type}; continuing in fp32.")
        self.amp_dtype = _resolve_amp_dtype(self.device, self.train_cfg["amp_dtype"])
        self.attn_dtype = self.amp_dtype if self.amp_available else self.sequence_data.frames.dtype
        self.attn_backend = configure_fast_attn(self.device, self.attn_dtype)
        self.renderer_mode, self.effective_gaussians = pick_renderer_mode_from_config(self.cfg)
        self.rgb_objective = RGBReconObjective(
            objective_spec_from_loss_config(self.loss_cfg),
            colorizer=self.colorize,
            rasterizer=self,
        )
        self.vjepa_feature_loss_weight = float(self.loss_cfg["vjepa_feature_weight"])
        self.vjepa_feature_loss = self.build_vjepa_feature_loss()
        self.gt_video_logged = False
        self.profile_timing_enabled = bool(self.train_cfg["profile_timing"])
        self._current_timing_terms: dict[str, float] = {}
        self.last_timing_terms: dict[str, float] = {}
        self.start_sequence_prefetch()

    def on_sequences_loaded(self) -> None:
        pass

    @contextmanager
    def model_eval_mode(self):
        was_training = self.model.training
        self.model.eval()
        try:
            yield
        finally:
            if was_training:
                self.model.train()

    def _dense_grid_for_render_size(self, render_size: int) -> torch.Tensor:
        size = int(render_size)
        if size not in self._dense_grid_by_render_size:
            self._dense_grid_by_render_size[size] = build_or_reuse_grid(size, size, self.device)
        return self._dense_grid_by_render_size[size]

    def _activate_render_size(self, render_size: int) -> None:
        size = int(render_size)
        self.cfg["render"]["render_size"] = size
        self.render_size = size
        self.dense_grid = self._dense_grid_for_render_size(size)
        self.renderer_mode, self.effective_gaussians = pick_renderer_mode_from_config(self.cfg)

    @contextmanager
    def temporary_render_size(self, render_size: int):
        previous_size = int(self.render_size)
        previous_grid = self.dense_grid
        previous_renderer_mode = self.renderer_mode
        previous_effective_gaussians = self.effective_gaussians
        self._activate_render_size(int(render_size))
        try:
            yield
        finally:
            self.cfg["render"]["render_size"] = previous_size
            self.render_size = previous_size
            self.dense_grid = previous_grid
            self.renderer_mode = previous_renderer_mode
            self.effective_gaussians = previous_effective_gaussians

    def render_size_for_step(self, step: int) -> int:
        schedule = self.train_cfg["render_size_schedule"]
        if not schedule:
            return self.base_render_size
        size = self.base_render_size
        for entry in schedule:
            if int(step) < int(entry["start_step"]):
                break
            size = int(entry["render_size"])
        return size

    def apply_render_size_schedule(self, step: int) -> None:
        size = self.render_size_for_step(step)
        if size == int(self.render_size):
            return
        previous = int(self.render_size)
        self._activate_render_size(size)
        print(
            "Render size schedule: "
            f"step {step} switched {previous}->{size} "
            f"({self.effective_gaussians} Gaussians, {self.renderer_mode} renderer)"
        )

    def render_size_schedule_summary(self) -> str | None:
        schedule = self.train_cfg["render_size_schedule"]
        if not schedule:
            return None
        return ", ".join(f"step {entry['start_step']} -> {entry['render_size']}" for entry in schedule)

    def build_vjepa_feature_loss(self):
        if self.vjepa_feature_loss_weight <= 0.0:
            return None
        from vjepa_feature_loss import TorchHubVJEPAFeatureLoss

        loss_module = TorchHubVJEPAFeatureLoss(
            model_id=self.loss_cfg["vjepa_feature_model_id"],
            crop_size=self.loss_cfg["vjepa_feature_crop_size"],
            dtype=self.loss_cfg["vjepa_feature_dtype"],
            checkpoint_url=self.loss_cfg["vjepa_feature_checkpoint_url"],
            checkpoint_key=self.loss_cfg["vjepa_feature_checkpoint_key"],
            temporal_stride=self.loss_cfg["vjepa_feature_temporal_stride"],
            normalize_features=self.loss_cfg["vjepa_feature_normalize"],
            loss_type=self.loss_cfg["vjepa_feature_loss_type"],
        ).to(self.device)
        loss_module.eval()
        print(
            "V-JEPA feature loss: "
            f"weight={self.vjepa_feature_loss_weight}, "
            f"model_id={self.loss_cfg['vjepa_feature_model_id']}, "
            f"crop_size={loss_module.crop_size}, "
            f"temporal_stride={loss_module.temporal_stride}"
        )
        return loss_module

    def view_dirs_for_features(self, features: torch.Tensor, cameras: tuple[Any, ...]) -> torch.Tensor | None:
        if self.colorize is None or self.colorize.view_condition == "none":
            return None
        return colorize_view_dirs_for_features(self.cfg, self.colorize, features, cameras)

    def make_target_view(
        self,
        *,
        view_id: str,
        frames: torch.Tensor,
        frame_indices: torch.Tensor,
        frame_times: torch.Tensor,
        cameras: tuple[Any, ...],
        role: ViewRole = "train",
        camera_role: CameraRole = "target",
        camera_owner: CameraOwner = "model",
        camera_name: str | None = None,
        metrics_prefix: str | None = None,
        log_media: bool = False,
    ) -> TargetView:
        return TargetView(
            view_id=view_id,
            role=role,
            camera_role=camera_role,
            camera_owner=camera_owner,
            frames=frames,
            frame_indices=frame_indices,
            frame_times=frame_times.reshape(-1),
            video_fps=self.sequence_data.video_fps,
            camera_name=camera_name,
            cameras=cameras,
            metrics_prefix=metrics_prefix,
            log_media=log_media,
        )

    def rasterize_decoded_clip(
        self,
        decoded: GaussianSequence,
        cameras: tuple[Any, ...],
    ) -> RasterizedClip:
        return render_clip_sequence(
            self.cfg,
            decoded,
            cameras,
            renderer_mode=self.renderer_mode,
            dense_grid=self.dense_grid,
        )

    def rasterize(self, decoded: GaussianSequence, target: TargetView) -> RasterizedView:
        if target.cameras is None:
            raise ValueError("TargetView.cameras is required for rasterization.")
        rasterized = self.rasterize_decoded_clip(decoded, target.cameras)
        return RasterizedView(
            view=target,
            features=rasterized.features,
            alpha=rasterized.alpha,
            cameras=target.cameras,
            view_dirs=self.view_dirs_for_features(rasterized.features, target.cameras),
        )

    def render_decoded_rgb_clip(
        self,
        decoded: GaussianSequence,
        *,
        frames: torch.Tensor,
        frame_indices: torch.Tensor,
        frame_times: torch.Tensor,
        phase: RunPhase,
        background: BackgroundSample | None = None,
        view_id: str = "decoded",
        role: str = "train",
    ) -> RenderedView:
        if decoded.cameras is None:
            raise ValueError("Video decode must include cameras.")
        target = self.make_target_view(
            view_id=view_id,
            frames=frames,
            frame_indices=frame_indices,
            frame_times=frame_times,
            cameras=decoded.cameras,
            role=role,
            camera_owner="model",
        )
        return self.rgb_objective.render_view(
            decoded,
            target,
            phase=phase,
            background=background,
        )

    def load_single_sequence_data(self) -> SequenceData:
        if self.data_cfg["sequence_dir"] is None:
            raise ValueError("config['data']['sequence_dir'] is required when manifest_path is not set.")
        if self.data_cfg["frame_source"] == "camera_json":
            camera_json_path = self.data_cfg["camera_json"] or (
                self.data_cfg["sequence_dir"] / "per_frame_cameras.json"
            )
            return load_camera_sequence(
                camera_json_path=camera_json_path,
                target_size=self.model_cfg["size"],
                camera_image_size=self.data_cfg["camera_image_size"],
                max_frames=self.data_cfg["max_frames"],
                focal_mode=self.data_cfg["camera_focal_mode"],
                image_crop_mode=self.data_cfg["image_crop_mode"],
                device=self.device,
            )
        if self.data_cfg["frame_source"] == "explicit_video" and self.data_cfg["video_path"] is None:
            raise ValueError("config['data']['video_path'] is required when frame_source='explicit_video'.")
        frames_dir = resolve_frames_dir(self.data_cfg["sequence_dir"], self.data_cfg["frames_dir"])
        return load_uncalibrated_sequence(
            sequence_dir=self.data_cfg["sequence_dir"],
            frames_dir=frames_dir,
            video_path=self.data_cfg["video_path"],
            target_size=self.model_cfg["size"],
            max_frames=self.data_cfg["max_frames"],
            frame_source=self.data_cfg["frame_source"],
            image_crop_mode=self.data_cfg["image_crop_mode"],
            device=self.device,
        )

    def load_train_sequences(self) -> list[SequenceData]:
        if self.data_cfg["manifest_path"] is None:
            return [self.load_single_sequence_data()]
        load_mode = self.data_cfg["train_manifest_load_mode"]
        self.train_sequence_sampler = ManifestSequenceSampler.from_manifest(
            self.data_cfg["manifest_path"],
            split=self.data_cfg["split"],
            data_cfg=self.data_cfg,
            model_cfg=self.model_cfg,
            device=self.device,
            load_mode=load_mode,
            sample_mode=self.data_cfg["train_manifest_sample_mode"] if load_mode == "lazy" else "random",
            prefetch_depth=int(self.data_cfg["train_manifest_prefetch"]),
        )
        self.lazy_train_entries = (
            self.train_sequence_sampler.entries if self.train_sequence_sampler.is_lazy else []
        )
        return self.train_sequence_sampler.sequences

    def load_eval_sequences(self) -> list[SequenceData]:
        eval_limit = int(self.data_cfg["eval_max_sequences"])
        if eval_limit == 0:
            return []
        eval_manifest_path = self.data_cfg["eval_manifest_path"] or self.data_cfg["manifest_path"]
        if eval_manifest_path is None:
            return [self.sequence_data] if hasattr(self, "sequence_data") else []
        return load_manifest_sequences(
            eval_manifest_path,
            split=self.data_cfg["eval_split"],
            data_cfg=self.data_cfg,
            model_cfg=self.model_cfg,
            device=self.device,
            limit=eval_limit,
        )

    def validate_train_sequences(self) -> None:
        if self.train_sequence_sampler is not None:
            minimum_required = int(self.model_cfg["train_frame_count"])
            label = "lazy train manifest entry" if self.train_sequence_sampler.is_lazy else "train sequence"
            self.train_sequence_sampler.validate_min_frame_count(minimum_required, label=label)
            return
        if not self.train_sequences:
            raise ValueError("No train sequences were loaded.")
        minimum_required = int(self.model_cfg["train_frame_count"])
        too_short = [
            sequence
            for sequence in self.train_sequences
            if sequence.frame_count < minimum_required
        ]
        if too_short:
            examples = ", ".join(str(sequence.source_path) for sequence in too_short[:3])
            raise ValueError(
                f"Need at least train_frame_count={minimum_required} frames in every train sequence; "
                f"{len(too_short)} sequence(s) were too short. Examples: {examples}"
            )

    def autocast_context(self):
        if self.amp_available:
            return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)
        return nullcontext()

    def reset_profile_timing(self) -> None:
        self._current_timing_terms = {}
        self.last_timing_terms = {}

    def _sync_for_profile_timing(self) -> None:
        if not self.profile_timing_enabled or not bool(self.train_cfg["profile_timing_sync"]):
            return
        sync_torch_device(self.device)

    @contextmanager
    def profile_section(self, name: str):
        if not self.profile_timing_enabled:
            yield
            return
        self._sync_for_profile_timing()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync_for_profile_timing()
            elapsed = time.perf_counter() - start
            self._current_timing_terms[name] = self._current_timing_terms.get(name, 0.0) + elapsed
            if name == "step_total":
                self.finish_profile_timing()

    def finish_profile_timing(self) -> None:
        if not self.profile_timing_enabled:
            return
        self.last_timing_terms = dict(sorted(self._current_timing_terms.items()))

    def profile_timing_payload(self) -> dict[str, float]:
        return {f"Timing/{key}_s": float(value) for key, value in self.last_timing_terms.items()}

    def should_print_profile_timing(self, step: int) -> bool:
        return self.profile_timing_enabled and step % int(self.train_cfg["profile_timing_log_every"]) == 0

    def profile_timing_message(self, step: int) -> str:
        if not self.last_timing_terms:
            return f"Timing step {step}: no measured sections"
        parts = " ".join(f"{key}={value:.4f}s" for key, value in self.last_timing_terms.items())
        return f"Timing step {step}: {parts}"

    @contextmanager
    def train_step_context(self):
        self.reset_profile_timing()
        with self.profile_section("step_total"):
            with self.profile_section("zero_grad"):
                self.optimizer.zero_grad(set_to_none=True)
            yield
        self.finish_profile_timing()

    def optimizer_step(self) -> None:
        with self.profile_section("optimizer_step"):
            self.optimizer.step()

    def start_sequence_prefetch(self) -> None:
        if self.train_sequence_sampler is None:
            return
        if self.train_sequence_sampler.start_prefetch():
            print(f"Train manifest prefetch: enabled, depth={self.train_sequence_sampler.prefetch_depth}.")

    def close_sequence_prefetch(self) -> None:
        if self.train_sequence_sampler is not None:
            self.train_sequence_sampler.close_prefetch()

    def sample_sequence(self) -> SequenceData:
        if self.train_sequence_sampler is not None:
            return self.train_sequence_sampler.sample()
        if len(self.train_sequences) == 1:
            return self.train_sequences[0]
        index = int(torch.randint(len(self.train_sequences), (1,)).item())
        return self.train_sequences[index]

    def sample_train_clip_batch(self) -> tuple[SequenceData, ClipBatch]:
        sequence_data = self.sample_sequence()
        clip = sample_clip_batch(
            sequence_data,
            train_frame_count=int(self.model_cfg["train_frame_count"]),
            frame_sampling=self.train_cfg["frame_sampling"],
            device=self.device,
        )
        return sequence_data, clip

    def sample_clip(self) -> tuple[SequenceData, torch.Tensor, torch.Tensor]:
        sequence_data, clip = self.sample_train_clip_batch()
        return sequence_data, clip.as_video_batch(), clip.as_time_batch(device=self.device)

    def initial_clip_indices(self) -> torch.Tensor:
        clip_length = int(self.model_cfg["train_frame_count"])
        return torch.arange(0, clip_length, device=self.device)

    def initial_clip_for_sequence(
        self,
        sequence_data: SequenceData,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        clip_indices = self.initial_clip_indices()
        clip_frames, clip_times = prepare_clip(sequence_data, clip_indices)
        return clip_indices, clip_frames, clip_times

    def export_window_indices(self, sequence_data: SequenceData) -> torch.Tensor:
        window = min(int(self.model_cfg["train_frame_count"]), int(sequence_data.frame_count))
        if window < 1:
            raise ValueError("Export window must contain at least one frame.")
        if window >= sequence_data.frame_count:
            return torch.arange(sequence_data.frame_count, device=self.device)
        start = min(int(self.export_cfg["window_start"]), sequence_data.frame_count - window)
        return torch.arange(start, start + window, device=self.device)

    def export_browser_bundle(self) -> Path | None:
        if not self.export_cfg["enabled"]:
            return None
        if not self.model_cfg["use_static_dynamic_split"]:
            raise ValueError("export=true currently requires model.static_tokens + model.dynamic_tokens.")
        sequence_index = int(self.export_cfg["sequence_index"])
        if self.train_sequence_sampler is not None:
            if sequence_index >= self.train_sequence_sampler.sequence_count:
                raise IndexError(
                    f"export.sequence_index={sequence_index} is out of range for "
                    f"{self.train_sequence_sampler.sequence_count} train manifest entries."
                )
            sequence_data = self.train_sequence_sampler.sequence_at(sequence_index)
        elif sequence_index >= len(self.train_sequences):
            raise IndexError(
                f"export.sequence_index={sequence_index} is out of range for "
                f"{len(self.train_sequences)} loaded train sequences."
            )
        else:
            sequence_data = self.train_sequences[sequence_index]
        from export_dynaworld_browser_bundle import export_browser_bundle_from_model, export_id_from_config

        clip_indices = self.export_window_indices(sequence_data)
        clip_frames, clip_times = prepare_clip(sequence_data, clip_indices)
        model_input = self.model_input_for_clip(sequence_data, clip_frames, clip_times)
        suffix = None
        if wandb.run is not None and getattr(wandb.run, "id", None):
            suffix = str(wandb.run.id)
        export_id = export_id_from_config(self.cfg, suffix=suffix)
        output_dir = self.export_cfg["output_root"] / export_id
        manifest_path = export_browser_bundle_from_model(
            model=self.model,
            resolved=self.cfg,
            sequence_data=sequence_data,
            clip_indices=clip_indices,
            clip_times=clip_times,
            model_input=model_input,
            output_dir=output_dir,
            config_path=None,
            state_dict_path=None,
            export_id=export_id,
        )
        print(f"Browser export written: {manifest_path}")
        return manifest_path

    def model_input_for_clip(
        self,
        sequence_data: SequenceData,
        clip_frames: torch.Tensor,
        clip_times: torch.Tensor,
    ) -> Any:
        del sequence_data, clip_times
        return clip_frames

    def forward_clip(self, model_input: Any, clip_times: torch.Tensor) -> GaussianSequence:
        with fast_attn_context(self.device), self.autocast_context():
            return self.model(model_input, decode_times=clip_times)

    def temporal_recon_chunk_size(self, frame_count: int) -> int:
        return _temporal_recon_chunk_size_impl(
            frame_count,
            recon_backward_strategy=self.train_cfg["recon_backward_strategy"],
            temporal_microbatch_size=int(self.train_cfg["temporal_microbatch_size"]),
        )

    def vjepa_loss_for_rendered(
        self,
        rendered: RenderedView,
        *,
        chunk_scale: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.vjepa_feature_loss is None:
            zero = rendered.rgb.new_zeros(())
            return zero, {}
        if rendered.target_rgb is None:
            raise ValueError("V-JEPA feature loss requires rendered.target_rgb.")
        raw = self.vjepa_feature_loss(rendered.rgb, rendered.target_rgb)
        weighted = raw * self.vjepa_feature_loss_weight * float(chunk_scale)
        return weighted, {
            "VJEPAFeature": raw.detach() * float(chunk_scale),
            "VJEPAFeatureWeighted": weighted.detach(),
        }

    def _requires_grad_tensors(self, values: list[Any]) -> list[torch.Tensor]:
        tensors: list[torch.Tensor] = []
        seen: set[int] = set()
        for value in values:
            if not torch.is_tensor(value) or not value.requires_grad:
                continue
            value_id = id(value)
            if value_id in seen:
                continue
            seen.add(value_id)
            tensors.append(value)
        return tensors

    def backward_boundary_tensors(
        self,
        chunk_sequence: GaussianSequence,
        cameras: tuple[Any, ...],
    ) -> list[torch.Tensor]:
        values: list[Any] = [
            chunk_sequence.xyz,
            chunk_sequence.scales,
            chunk_sequence.quats,
            chunk_sequence.opacities,
            chunk_sequence.rgbs,
        ]
        for camera in cameras:
            values.extend(
                [
                    getattr(camera, "fx", None),
                    getattr(camera, "fy", None),
                    getattr(camera, "cx", None),
                    getattr(camera, "cy", None),
                    getattr(camera, "camera_to_world", None),
                    getattr(camera, "distortion", None),
                ]
            )
        return self._requires_grad_tensors(values)

    def colorize_trainable_parameters(self) -> list[torch.Tensor]:
        if self.colorize is None:
            return []
        return [parameter for parameter in self.colorize.parameters() if parameter.requires_grad]

    @staticmethod
    def accumulate_manual_grads(parameters: list[torch.Tensor], gradients: tuple[torch.Tensor | None, ...]) -> None:
        for parameter, gradient in zip(parameters, gradients):
            if gradient is None:
                continue
            detached = gradient.detach()
            if parameter.grad is None:
                parameter.grad = detached.clone()
            else:
                parameter.grad = parameter.grad + detached

    def backward_recon_loss_split(
        self,
        chunk_recon_loss: torch.Tensor,
        regularizer_loss: torch.Tensor,
        *,
        boundary_tensors: list[torch.Tensor],
        is_last_chunk: bool,
    ) -> None:
        """Profile recon backward around the decoded Gaussian/camera boundary.

        This is an opt-in diagnostic path. It keeps optimizer gradients equivalent
        to the normal backward path while exposing whether time is spent below
        the decoder boundary (loss/colorize/raster backward) or above it (model
        backward from decoded Gaussian and camera tensors).
        """

        colorize_parameters = self.colorize_trainable_parameters()
        grad_inputs = boundary_tensors + colorize_parameters
        if grad_inputs:
            with self.profile_section("backward/raster_loss_to_boundary"):
                gradients = torch.autograd.grad(
                    chunk_recon_loss,
                    grad_inputs,
                    retain_graph=True,
                    allow_unused=True,
                )
            boundary_gradients = gradients[: len(boundary_tensors)]
            colorize_gradients = gradients[len(boundary_tensors) :]
            self.accumulate_manual_grads(colorize_parameters, colorize_gradients)
        else:
            boundary_gradients = ()

        usable_boundary = [
            (tensor, gradient)
            for tensor, gradient in zip(boundary_tensors, boundary_gradients)
            if gradient is not None
        ]
        if usable_boundary:
            retain_for_regularizers = bool(is_last_chunk and regularizer_loss.requires_grad)
            with self.profile_section("backward/model_from_boundary"):
                torch.autograd.backward(
                    [tensor for tensor, _gradient in usable_boundary],
                    [gradient for _tensor, gradient in usable_boundary],
                    retain_graph=(not is_last_chunk) or retain_for_regularizers,
                )

        if is_last_chunk and regularizer_loss.requires_grad:
            with self.profile_section("backward/regularizers"):
                regularizer_loss.backward()

    def recon_backward(
        self,
        clip_frames: torch.Tensor,
        decoded: GaussianSequence,
        regularizer_loss: torch.Tensor,
        keep_preview: bool,
        loss_scale: float | torch.Tensor = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, dict[str, torch.Tensor]]:
        recon_loss = clip_frames.new_tensor(0.0)
        loss_scale_tensor = torch.as_tensor(loss_scale, dtype=clip_frames.dtype, device=clip_frames.device)
        aux_loss_terms: dict[str, torch.Tensor] = {}
        preview_render = None
        preview_features = None
        if decoded.cameras is None:
            raise ValueError("Implicit-camera video decode must include cameras.")
        frame_count = len(decoded.cameras)
        chunk_size = self.temporal_recon_chunk_size(frame_count)

        with self.profile_section("background"):
            if self.rgb_objective.background_policy.spec.feature_train_mode == "none":
                train_background = self.rgb_objective.sample_background(
                    phase="train",
                    like=clip_frames,
                    frame_count=frame_count,
                )
            else:
                train_background = None

        for chunk_start in range(0, frame_count, chunk_size):
            chunk_end = min(chunk_start + chunk_size, frame_count)
            chunk_sequence = gaussian_sequence_slice(decoded, chunk_start, chunk_end)
            chunk_indices = torch.arange(chunk_start, chunk_end, device=clip_frames.device)
            chunk_times = chunk_indices.to(dtype=torch.float32) / float(max(frame_count - 1, 1))
            target = self.make_target_view(
                view_id="train_clip",
                frames=clip_frames[0, chunk_start:chunk_end],
                frame_indices=chunk_indices,
                frame_times=chunk_times,
                cameras=tuple(decoded.cameras[chunk_start:chunk_end]),
                role="train",
            )
            with self.profile_section("render_view_total"):
                rendered_chunk = self.rgb_objective.render_view(
                    chunk_sequence,
                    target,
                    phase="train",
                    background=train_background,
                )
            self.rgb_objective.require_alpha_for_feature_background(rendered_chunk)
            chunk_features = rendered_chunk.features
            if keep_preview and self.feature_pca_log and preview_features is None:
                preview_features = chunk_features[0].detach()
            chunk_renders = rendered_chunk.rgb
            if keep_preview and preview_render is None:
                preview_render = chunk_renders[0].detach()
            with self.profile_section("recon_loss"):
                chunk_losses = self.rgb_objective.reconstruction_loss_per_image(rendered_chunk)
            chunk_recon_loss = chunk_losses.sum() / frame_count
            with self.profile_section("vjepa_feature_loss"):
                vjepa_loss, vjepa_terms = self.vjepa_loss_for_rendered(
                    rendered_chunk,
                    chunk_scale=float(chunk_end - chunk_start) / float(frame_count),
                )
            chunk_recon_loss = chunk_recon_loss + vjepa_loss
            for key, value in vjepa_terms.items():
                aux_loss_terms[key] = aux_loss_terms.get(key, value.new_zeros(())) + value
            recon_loss = recon_loss + chunk_recon_loss.detach()
            is_last_chunk = chunk_end == frame_count
            scaled_chunk_recon_loss = chunk_recon_loss * loss_scale_tensor
            scaled_regularizer_loss = regularizer_loss * loss_scale_tensor
            backward_loss = scaled_chunk_recon_loss + (scaled_regularizer_loss if is_last_chunk else 0.0)
            with self.profile_section("backward"):
                if bool(self.train_cfg["profile_backward_split"]):
                    self.backward_recon_loss_split(
                        scaled_chunk_recon_loss,
                        scaled_regularizer_loss,
                        boundary_tensors=self.backward_boundary_tensors(chunk_sequence, target.cameras),
                        is_last_chunk=is_last_chunk,
                    )
                else:
                    backward_loss.backward(retain_graph=not is_last_chunk)

        return recon_loss, preview_render, preview_features, aux_loss_terms

    def initial_recon_step_result(
        self,
        sequence_data: SequenceData,
        clip_frames: torch.Tensor,
        clip_indices: torch.Tensor,
        clip_times: torch.Tensor,
        decoded: GaussianSequence,
        *,
        view_id: str,
        camera_state: CameraState | None,
        bank_rate_loss: torch.Tensor,
        bank_rate_terms: dict[str, torch.Tensor],
        camera_loss: torch.Tensor | None = None,
        camera_motion_loss: torch.Tensor | None = None,
        camera_temporal_loss: torch.Tensor | None = None,
        camera_global_loss: torch.Tensor | None = None,
    ) -> StepResult:
        rendered = self.render_decoded_rgb_clip(
            decoded,
            frames=clip_frames[0],
            frame_indices=clip_indices,
            frame_times=clip_times,
            phase="eval",
            view_id=view_id,
        )
        preview_features = rendered.features[0].detach() if self.feature_pca_log else None
        recon_loss = self.rgb_objective.reconstruction_loss(rendered)
        vjepa_loss, aux_loss_terms = self.vjepa_loss_for_rendered(rendered)
        recon_loss = recon_loss + vjepa_loss
        total_camera_loss = camera_loss if camera_loss is not None else clip_frames.new_zeros(())
        return build_step_result(
            sequence_data=sequence_data,
            clip_frames=clip_frames,
            preview_render=rendered.rgb[0].detach(),
            preview_features=preview_features,
            camera_state=camera_state,
            loss=recon_loss + total_camera_loss + bank_rate_loss,
            recon_loss=recon_loss,
            camera_motion_loss=camera_motion_loss,
            camera_temporal_loss=camera_temporal_loss,
            camera_global_loss=camera_global_loss,
            bank_rate_loss=bank_rate_loss,
            bank_rate_terms=bank_rate_terms,
            aux_loss_terms=aux_loss_terms,
        )

    @torch.no_grad()
    def initial_step_result(self) -> StepResult:
        with self.model_eval_mode():
            sequence_data = self.train_sequences[0]
            clip_indices, clip_frames, clip_times = self.initial_clip_for_sequence(sequence_data)
            model_input = self.model_input_for_clip(sequence_data, clip_frames, clip_times)
            decoded = self.forward_clip(model_input, clip_times)
            if decoded.camera_state is None:
                raise ValueError("Implicit-camera video decode must include camera_state.")

            camera_loss, camera_motion_loss, camera_temporal_loss, camera_global_loss = _build_camera_loss_impl(
                clip_times,
                decoded.camera_state,
                self.loss_cfg,
            )
            bank_rate_loss, bank_rate_terms = _build_bank_rate_loss_impl(decoded, self.loss_cfg)
            return self.initial_recon_step_result(
                sequence_data,
                clip_frames,
                clip_indices,
                clip_times,
                decoded,
                view_id="initial_train_view",
                camera_state=decoded.camera_state,
                bank_rate_loss=bank_rate_loss,
                bank_rate_terms=bank_rate_terms,
                camera_loss=camera_loss,
                camera_motion_loss=camera_motion_loss,
                camera_temporal_loss=camera_temporal_loss,
                camera_global_loss=camera_global_loss,
            )

    def step(self, keep_preview: bool = False) -> StepResult:
        with self.train_step_context():
            with self.profile_section("sample_clip"):
                sequence_data, clip = self.sample_train_clip_batch()
                clip_frames = clip.as_video_batch()
                clip_times = clip.as_time_batch(device=self.device)
            with self.profile_section("model_input"):
                model_input = self.model_input_for_clip(sequence_data, clip_frames, clip_times)
            with self.profile_section("forward_decode"):
                decoded = self.forward_clip(model_input, clip_times)
            if decoded.camera_state is None:
                raise ValueError("Implicit-camera video decode must include camera_state.")

            with self.profile_section("regularizers"):
                camera_loss, camera_motion_loss, camera_temporal_loss, camera_global_loss = _build_camera_loss_impl(
                    clip_times,
                    decoded.camera_state,
                    self.loss_cfg,
                )
                bank_rate_loss, bank_rate_terms = _build_bank_rate_loss_impl(decoded, self.loss_cfg)

            recon_loss, preview_render, preview_features, aux_loss_terms = self.recon_backward(
                clip_frames,
                decoded,
                camera_loss + bank_rate_loss,
                keep_preview,
            )

            self.optimizer_step()
        loss = recon_loss + camera_loss.detach() + bank_rate_loss.detach()
        return build_step_result(
            sequence_data=sequence_data,
            clip_frames=clip_frames,
            preview_render=preview_render,
            preview_features=preview_features,
            camera_state=decoded.camera_state,
            loss=loss,
            recon_loss=recon_loss,
            camera_motion_loss=camera_motion_loss,
            camera_temporal_loss=camera_temporal_loss,
            camera_global_loss=camera_global_loss,
            bank_rate_loss=bank_rate_loss,
            bank_rate_terms=bank_rate_terms,
            aux_loss_terms=aux_loss_terms,
        )

    def progress_message(self, result: StepResult) -> str:
        if result.camera_state is None:
            return f"Loss: {result.loss.item():.4f} recon: {result.recon_loss.item():.4f}"
        metrics = camera_state_summary_metrics(result.camera_state)
        return (
            f"Loss: {result.loss.item():.4f} "
            f"recon: {result.recon_loss.item():.4f} "
            f"fov: {metrics['fov_degrees']:.2f} "
            f"r: {metrics['radius']:.2f}"
        )

    def should_log_scalars(self, step: int) -> bool:
        return _should_log_scalar(self.cfg, step)

    def should_log_images(self, step: int) -> bool:
        return _should_log_image(self.cfg, step, log_step_zero=bool(self.logging_cfg["log_initial_media"]))

    def should_log_videos(self, step: int) -> bool:
        return _should_log_video(self.cfg, step, log_step_zero=bool(self.logging_cfg["log_initial_media"]))

    def log_gate_flags(self, step: int) -> tuple[bool, bool, bool]:
        return (
            self.should_log_scalars(step),
            self.should_log_images(step),
            self.should_log_videos(step),
        )

    def scalar_payload(self, result: StepResult) -> dict[str, Any]:
        payload = _scalar_payload_impl(
            self.cfg,
            result,
            train_sequence_count=self.train_sequence_count,
            eval_sequence_count=len(self.eval_sequences),
        )
        payload.update(self.profile_timing_payload())
        payload["RenderSize"] = int(getattr(result, "render_size", self.render_size))
        payload["Render/BaseSize"] = int(self.base_render_size)
        payload["Render/SizeScheduleEnabled"] = 1.0 if self.train_cfg["render_size_schedule"] else 0.0
        return payload

    def _eval_decode_clip(
        self,
        sequence_data: SequenceData,
        clip_indices: torch.Tensor,
        clip_frames: torch.Tensor,
        clip_times: torch.Tensor,
    ) -> GaussianSequence:
        del clip_indices
        return self.forward_clip(self.model_input_for_clip(sequence_data, clip_frames, clip_times), clip_times)

    def _eval_render_clip(
        self,
        decoded: GaussianSequence,
        *,
        clip_frames: torch.Tensor,
        clip_indices: torch.Tensor,
        clip_times: torch.Tensor,
        view_id: str,
    ) -> RenderedView:
        return self.render_decoded_rgb_clip(
            decoded,
            frames=clip_frames,
            frame_indices=clip_indices,
            frame_times=clip_times,
            phase="eval",
            view_id=view_id,
            role="debug",
        )

    @torch.no_grad()
    def render_full_sequence(
        self,
        sequence_data: SequenceData,
    ) -> RenderedClip:
        return _render_full_sequence_impl(
            self.cfg, self.model, sequence_data, self._eval_decode_clip, self._eval_render_clip
        )

    def validation_video_payload(self, step: int | None = None) -> dict[str, Any]:
        del step
        sequences = self.eval_sequences or [self.sequence_data]
        metric_payloads = []
        payload: dict[str, Any] = {
            "Eval/SequenceCount": len(sequences),
        }
        for sequence_index, sequence_data in enumerate(sequences):
            rendered = self.render_full_sequence(sequence_data)
            gt_sequence = resize_images(sequence_data.frames, self.render_size).detach().cpu()
            metrics = {
                **eval_metric_payload(rendered.rgb_sequence, gt_sequence, self.loss_cfg),
                **temporal_similarity_payload(rendered.rgb_sequence, gt_sequence, self.loss_cfg),
                **rendered.temporal_metrics,
            }
            if rendered.camera_state is not None:
                metrics.update(camera_state_payload(rendered.camera_state, key_prefix="Camera/Eval"))
                metrics.update(camera_temporal_payload(rendered.camera_state))
            metric_payloads.append(metrics)
            sequence_payload, self.gt_video_logged = single_cam_validation_video_payload(
                self.cfg,
                sequence_index=sequence_index,
                rendered_sequence=rendered.rgb_sequence,
                gt_sequence=gt_sequence,
                feature_sequence=rendered.feature_sequence,
                alpha_sequence=rendered.alpha_sequence,
                eval_payload={},
                gt_video_logged=self.gt_video_logged,
                fps=sequence_data.video_fps,
            )
            payload.update(sequence_payload)

        metric_keys = sorted({key for item in metric_payloads for key in item})
        for key in metric_keys:
            values = [item[key] for item in metric_payloads if key in item]
            payload[key] = sum(values) / len(values)
        return payload

    @torch.no_grad()
    def save_validation_media(self, step: int, *, save_videos: bool) -> None:
        output_dir = self.logging_cfg["output_dir"]
        if output_dir is None:
            return
        sequences = self.eval_sequences or [self.sequence_data]
        sequence_data = sequences[0]
        rendered = self.render_full_sequence(sequence_data)
        gt_sequence = resize_images(sequence_data.frames, self.render_size).detach().cpu()
        save_token_gs_eval_media(
            output_dir,
            step,
            rendered.rgb_sequence,
            gt_sequence,
            fps=sequence_data.video_fps,
            save_videos=save_videos,
        )

    def val_log(self, step: int, result: StepResult) -> None:
        should_log_scalars, should_log_images, should_log_videos = self.log_gate_flags(step)
        if not (should_log_scalars or should_log_images or should_log_videos):
            return
        if should_log_images or should_log_videos:
            self.save_validation_media(step, save_videos=should_log_videos)

        def _wandb_payload() -> dict[str, Any]:
            payload = self.scalar_payload(result)
            if should_log_images:
                payload.update(
                    training_preview_payload(self.cfg, result, step, feature_pca_log=self.feature_pca_log)
                )
            if should_log_videos:
                payload.update(self.validation_video_payload(step=step))
            return payload

        log_wandb_run_payload_lazy(self.wandb_run, _wandb_payload, step=step)

    def training_start_message(self) -> str:
        token_summary = token_summary_from_model_config(self.model_cfg)
        return (
            "Starting DynamicVideoTokenGSImplicitCamera Training: "
            f"{self.train_sequence_count} train sequence(s), train_frame_count={self.model_cfg['train_frame_count']}, "
            f"input_size={self.model_cfg['size']}, render_size={self.render_size}, "
            f"1 global camera token + 1 path token + {token_summary} x "
            f"{self.model_cfg['gaussians_per_token']} gaussians/token = "
            f"{self.effective_gaussians} explicit Gaussians with {self.renderer_mode} renderer..."
        )

    def training_camera_message(self) -> str:
        return (
            "Camera model: "
            f"global_head={self.cfg['camera']['global_head']}, "
            f"lens_model={self.cfg['camera']['lens_model']}"
        )

    def training_complete_message(self) -> str:
        if self.wandb_run is None:
            return "DynamicVideoTokenGSImplicitCamera training complete (W&B disabled)."
        return "DynamicVideoTokenGSImplicitCamera training complete. Check your Weights & Biases dashboard."

    def training_preamble_messages(self) -> tuple[str, ...]:
        return ()

    def after_training_complete(self) -> None:
        pass

    def should_export_browser_after_training(self) -> bool:
        return True

    def print_training_header(self) -> None:
        self.apply_render_size_schedule(0)
        print(self.training_start_message())
        schedule_summary = self.render_size_schedule_summary()
        if schedule_summary is not None:
            print(f"Render size schedule: {schedule_summary}; base/final config size={self.base_render_size}")
        print(f"Reconstruction backward strategy: {self.train_cfg['recon_backward_strategy']}")
        print(self.training_camera_message())
        print(
            "Video encoder: "
            f"backend={self.model_cfg['video_encoder_backend']}, "
            f"vjepa_model_id={self.model_cfg['vjepa_model_id']}"
        )
        print(
            f"Temporal reconstruction chunk size: {self.temporal_recon_chunk_size(self.model_cfg['train_frame_count'])}"
        )
        print(f"Attention backend: {self.attn_backend}")

    def run_training_loop(self) -> None:
        initial_result = self.initial_step_result()
        print(f"Step 0 initialization diagnostic: {self.progress_message(initial_result)}")
        self.val_log(0, initial_result)

        pbar = tqdm(range(1, self.train_cfg["steps"] + 1))
        try:
            for step in pbar:
                self.apply_render_size_schedule(step)
                keep_preview = self.should_log_images(step)
                result = self.step(keep_preview=keep_preview)
                pbar.set_description(self.progress_message(result))
                if self.should_print_profile_timing(step):
                    pbar.write(self.profile_timing_message(step))
                self.val_log(step, result)
            if self.should_export_browser_after_training():
                self.export_browser_bundle()
        finally:
            self.close_sequence_prefetch()
            finish_wandb_run(self.wandb_run)

    def run(self) -> None:
        for message in self.training_preamble_messages():
            print(message)
        self.print_training_header()
        self.run_training_loop()
        print(self.training_complete_message())
        self.after_training_complete()


class KnownCameraTrainer(Trainer):
    def validate_train_sequences(self) -> None:
        super().validate_train_sequences()
        missing = [sequence for sequence in self.train_sequences if sequence.cameras is None]
        if missing:
            raise ValueError(
                "Known-camera training requires cameras on every train sequence. "
                f"Missing camera metadata for {len(missing)} sequence(s)."
            )

    def sample_known_clip(self) -> tuple[SequenceData, torch.Tensor, torch.Tensor, tuple[Any, ...]]:
        sequence_data, clip = self.sample_train_clip_batch()
        if clip.cameras is None:
            raise ValueError("Known-camera sequence has no cameras.")
        return sequence_data, clip.as_video_batch(), clip.as_time_batch(device=self.device), clip.cameras

    def known_cameras_for_indices(
        self,
        sequence_data: SequenceData,
        clip_indices: torch.Tensor,
    ) -> tuple[Any, ...]:
        if sequence_data.cameras is None:
            raise ValueError("Known-camera sequence has no cameras.")
        return tuple(sequence_data.cameras[int(index)] for index in clip_indices.detach().cpu().tolist())

    def forward_known_clip(
        self,
        clip_frames: torch.Tensor,
        clip_times: torch.Tensor,
        clip_cameras: tuple[Any, ...],
    ) -> GaussianSequence:
        with fast_attn_context(self.device), self.autocast_context():
            return self.model(clip_frames, decode_times=clip_times, cameras=clip_cameras)

    def step(self, keep_preview: bool = False) -> StepResult:
        with self.train_step_context():
            with self.profile_section("sample_clip"):
                sequence_data, clip_frames, clip_times, clip_cameras = self.sample_known_clip()
            decoded = self.forward_known_clip(clip_frames, clip_times, clip_cameras)
            if decoded.cameras is None:
                raise ValueError("Known-camera video decode must include cameras.")

            bank_rate_loss, bank_rate_terms = _build_bank_rate_loss_impl(decoded, self.loss_cfg)
            recon_loss, preview_render, preview_features, aux_loss_terms = self.recon_backward(
                clip_frames,
                decoded,
                bank_rate_loss,
                keep_preview,
            )

            self.optimizer_step()
        loss = recon_loss + bank_rate_loss.detach()
        return build_step_result(
            sequence_data=sequence_data,
            clip_frames=clip_frames,
            preview_render=preview_render,
            preview_features=preview_features,
            camera_state=None,
            loss=loss,
            recon_loss=recon_loss,
            bank_rate_loss=bank_rate_loss,
            bank_rate_terms=bank_rate_terms,
            aux_loss_terms=aux_loss_terms,
        )

    @torch.no_grad()
    def initial_step_result(self) -> StepResult:
        with self.model_eval_mode():
            sequence_data = self.train_sequences[0]
            clip_indices, clip_frames, clip_times = self.initial_clip_for_sequence(sequence_data)
            clip_cameras = self.known_cameras_for_indices(sequence_data, clip_indices)
            decoded = self.forward_known_clip(clip_frames, clip_times, clip_cameras)
            if decoded.cameras is None:
                raise ValueError("Known-camera video decode must include cameras.")

            bank_rate_loss, bank_rate_terms = _build_bank_rate_loss_impl(decoded, self.loss_cfg)
            return self.initial_recon_step_result(
                sequence_data,
                clip_frames,
                clip_indices,
                clip_times,
                decoded,
                view_id="initial_known_view",
                camera_state=None,
                bank_rate_loss=bank_rate_loss,
                bank_rate_terms=bank_rate_terms,
            )

    def _eval_decode_clip(
        self,
        sequence_data: SequenceData,
        clip_indices: torch.Tensor,
        clip_frames: torch.Tensor,
        clip_times: torch.Tensor,
    ) -> GaussianSequence:
        clip_cameras = self.known_cameras_for_indices(sequence_data, clip_indices)
        return self.forward_known_clip(clip_frames, clip_times, clip_cameras)

    def training_start_message(self) -> str:
        return (
            "Starting DynamicVideoTokenGSKnownCamera Training: "
            f"{self.train_sequence_count} train sequence(s), train_frame_count={self.model_cfg['train_frame_count']}, "
            f"input_size={self.model_cfg['size']}, render_size={self.render_size}, "
            f"{token_summary_from_model_config(self.model_cfg)} x "
            f"{self.model_cfg['gaussians_per_token']} gaussians/token = "
            f"{self.effective_gaussians} explicit Gaussians with {self.renderer_mode} renderer..."
        )

    def training_camera_message(self) -> str:
        return "Camera model: known/precomputed"

    def training_complete_message(self) -> str:
        return "DynamicVideoTokenGSKnownCamera training complete. Check your Weights & Biases dashboard."

    def should_export_browser_after_training(self) -> bool:
        return False


def trainer_class_for_config(config: dict[str, Any]) -> type[Trainer]:
    variant = str(config.get("model", {}).get("variant", "learned_time_orbit_path")).lower()
    if variant == "known_camera":
        return KnownCameraTrainer
    return Trainer


def run_training(config: dict[str, Any]) -> None:
    trainer_class_for_config(config)(config).run()


__all__ = [
    "CAMERA_OPTION_DEFAULTS",
    "DATA_OPTION_DEFAULTS",
    "EXPORT_OPTION_DEFAULTS",
    "KnownCameraTrainer",
    "LOSS_OPTION_DEFAULTS",
    "MODEL_OPTION_DEFAULTS",
    "TRAIN_TIMING_DEFAULTS",
    "Trainer",
    "normalize_render_size_schedule",
    "resolve_config",
    "run_training",
    "trainer_class_for_config",
]
