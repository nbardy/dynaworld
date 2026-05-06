from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import wandb
from config_utils import apply_defaults, load_config_file, path_or_none, resolved_config, serialize_config_value
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
    RasterizedClip,
    RenderedClip,
    colorize_view_dirs_for_features,
    gaussian_sequence_slice,
    prepare_clip,
    render_clip_sequence,
    render_full_sequence as _render_full_sequence_impl,
)
from pipeline.validation_media import (
    render_preview_image as _render_preview_image_impl,
    scalar_payload as _scalar_payload_impl,
    single_cam_validation_video_payload,
)
from rendering import (
    build_or_reuse_grid,
    resize_images,
)
from rendering import pick_renderer_mode as resolve_renderer_mode
from runtime_types import CameraState, GaussianSequence, SequenceData, StepResult
from sequence_data import (
    load_camera_sequence,
    load_manifest_sequences,
    load_uncalibrated_sequence,
    resolve_frames_dir,
    select_window_indices,
)
from tqdm import tqdm

LOSS_OPTION_DEFAULTS = {
    "type": "l1_mse",
    "l1_weight": 1.0,
    "mse_weight": 0.2,
    "dssim_weight": 0.2,
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
    apply_defaults(cfg["data"], DATA_OPTION_DEFAULTS)
    cfg["data"]["sequence_dir"] = path_or_none(cfg["data"].get("sequence_dir"))
    cfg["data"]["frames_dir"] = path_or_none(cfg["data"]["frames_dir"])
    cfg["data"]["video_path"] = path_or_none(cfg["data"]["video_path"])
    cfg["data"]["manifest_path"] = path_or_none(cfg["data"]["manifest_path"])
    cfg["data"]["eval_manifest_path"] = path_or_none(cfg["data"]["eval_manifest_path"])
    cfg["data"]["camera_json"] = path_or_none(cfg["data"]["camera_json"])
    cfg["data"]["split"] = str(cfg["data"]["split"])
    cfg["data"]["eval_split"] = str(cfg["data"]["eval_split"])
    cfg["data"]["eval_max_sequences"] = int(cfg["data"]["eval_max_sequences"])
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
    has_static_dynamic_split = (
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
    window_size = int(cfg["losses"]["ssim_window_size"])
    if window_size < 1 or window_size % 2 != 1:
        raise ValueError(f"losses.ssim_window_size must be a positive odd integer, got {window_size}.")
    cfg["losses"]["ssim_window_size"] = window_size
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


def pick_renderer_mode_from_config(config: dict[str, Any]) -> tuple[str, int]:
    model_cfg = config["model"]
    render_cfg = config["render"]
    effective_gaussians = model_cfg["tokens"] * model_cfg["gaussians_per_token"]
    renderer_mode = resolve_renderer_mode(
        renderer=render_cfg["renderer"],
        gaussian_count=effective_gaussians,
        height=render_cfg["render_size"],
        width=render_cfg["render_size"],
        auto_dense_limit=render_cfg["auto_dense_limit"],
    )
    return renderer_mode, effective_gaussians


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
        self.recon_backward_strategy = self.train_cfg["recon_backward_strategy"]
        self.temporal_microbatch_size = int(self.train_cfg["temporal_microbatch_size"])
        self.render_size = int(self.render_cfg["render_size"])

        self.device = pick_device()
        print(f"Using device: {self.device}")

        self.train_sequences = self.load_train_sequences()
        self.sequence_data = self.train_sequences[0]
        self.num_frames = self.sequence_data.frame_count
        self.eval_sequences = self.load_eval_sequences()
        self.validate_train_sequences()

        train_frames = [sequence.frame_count for sequence in self.train_sequences]
        eval_frames = [sequence.frame_count for sequence in self.eval_sequences]
        print(
            f"Loaded {len(self.train_sequences)} train sequence(s), "
            f"frames min/median/max={min(train_frames)}/{sorted(train_frames)[len(train_frames) // 2]}/{max(train_frames)}"
        )
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

        wandb.init(
            project=self.logging_cfg["wandb_project"],
            name=self.logging_cfg["wandb_run_name"],
            tags=self.logging_cfg["wandb_tags"],
            config=serialize_config_value(self.cfg),
        )

        self.model = build_model_from_config(self.cfg).to(self.device)
        self.model.train()
        self.feature_dim = int(self.model_cfg["feature_dim"])
        colorizer = build_colorizer(self.cfg["colorize"], feature_dim=self.feature_dim)
        self.colorize = None if colorizer.module is None else colorizer.module.to(self.device)
        self.feature_pca_log = bool(self.logging_cfg["feature_pca_log"])
        if self.feature_pca_log and self.feature_dim == 3:
            raise ValueError("logging.feature_pca_log=true requires model.feature_dim != 3.")
        trainable_parameters = list(self.model.parameters())
        if self.colorize is not None:
            trainable_parameters = trainable_parameters + list(self.colorize.parameters())
        self.optimizer = torch.optim.Adam(
            trainable_parameters,
            lr=self.train_cfg["lr"],
            fused=self.device.type in {"cuda", "mps"},
        )

        self.dense_grid = build_or_reuse_grid(self.render_size, self.render_size, self.device)
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
        self.gt_video_logged = False

    def on_sequences_loaded(self) -> None:
        pass

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
            device=self.device,
        )

    def load_train_sequences(self) -> list[SequenceData]:
        if self.data_cfg["manifest_path"] is None:
            return [self.load_single_sequence_data()]
        return load_manifest_sequences(
            self.data_cfg["manifest_path"],
            split=self.data_cfg["split"],
            data_cfg=self.data_cfg,
            model_cfg=self.model_cfg,
            device=self.device,
        )

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

    def sample_sequence(self) -> SequenceData:
        if len(self.train_sequences) == 1:
            return self.train_sequences[0]
        index = int(torch.randint(len(self.train_sequences), (1,)).item())
        return self.train_sequences[index]

    def sample_clip(self) -> tuple[SequenceData, torch.Tensor, torch.Tensor]:
        sequence_data = self.sample_sequence()
        clip_indices = select_window_indices(
            sequence_data.frame_count,
            self.model_cfg["train_frame_count"],
            device=self.device,
        )
        clip_frames, clip_times = prepare_clip(sequence_data, clip_indices)
        return sequence_data, clip_frames, clip_times

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
        if sequence_index >= len(self.train_sequences):
            raise IndexError(
                f"export.sequence_index={sequence_index} is out of range for "
                f"{len(self.train_sequences)} loaded train sequences."
            )
        from export_dynaworld_browser_bundle import export_browser_bundle_from_model, export_id_from_config

        sequence_data = self.train_sequences[sequence_index]
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
            recon_backward_strategy=self.recon_backward_strategy,
            temporal_microbatch_size=self.temporal_microbatch_size,
        )

    def recon_backward(
        self,
        clip_frames: torch.Tensor,
        decoded: GaussianSequence,
        regularizer_loss: torch.Tensor,
        keep_preview: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        recon_loss = clip_frames.new_tensor(0.0)
        preview_render = None
        preview_features = None
        if decoded.cameras is None:
            raise ValueError("Implicit-camera video decode must include cameras.")
        frame_count = len(decoded.cameras)
        chunk_size = self.temporal_recon_chunk_size(frame_count)

        train_background = self.rgb_objective.sample_background(
            phase="train",
            like=clip_frames,
            frame_count=frame_count,
        )

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
            rendered_chunk = self.rgb_objective.render_view(
                chunk_sequence,
                target,
                phase="train",
                background=train_background,
            )
            if self.feature_dim != 3 and rendered_chunk.alpha is None:
                raise ValueError(
                    "F-channel training requires alpha-aware render output so random-background "
                    "composition is active. Got alpha=None; check renderer='fast_mac' and v5_features build."
                )
            chunk_features = rendered_chunk.features
            if keep_preview and self.feature_pca_log and preview_features is None:
                preview_features = chunk_features[0].detach()
            chunk_renders = rendered_chunk.rgb
            if keep_preview and preview_render is None:
                preview_render = chunk_renders[0].detach()
            chunk_losses = self.rgb_objective.reconstruction_loss_per_image(rendered_chunk)
            chunk_recon_loss = chunk_losses.sum() / frame_count
            recon_loss = recon_loss + chunk_recon_loss.detach()
            is_last_chunk = chunk_end == frame_count
            backward_loss = chunk_recon_loss + (regularizer_loss if is_last_chunk else 0.0)
            backward_loss.backward(retain_graph=not is_last_chunk)

        return recon_loss, preview_render, preview_features

    @torch.no_grad()
    def initial_step_result(self) -> StepResult:
        was_training = self.model.training
        self.model.eval()
        try:
            sequence_data = self.train_sequences[0]
            clip_length = int(self.model_cfg["train_frame_count"])
            clip_indices = torch.arange(0, clip_length, device=self.device)
            clip_frames, clip_times = prepare_clip(sequence_data, clip_indices)
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
            rendered = self.render_decoded_rgb_clip(
                decoded,
                frames=clip_frames[0],
                frame_indices=clip_indices,
                frame_times=clip_times,
                phase="eval",
                view_id="initial_train_view",
            )
            preview_features = rendered.features[0].detach() if self.feature_pca_log else None
            rendered_clip = rendered.rgb
            recon_loss = self.rgb_objective.reconstruction_loss(rendered)
            loss = recon_loss + camera_loss + bank_rate_loss
            return StepResult(
                source_path=sequence_data.source_path,
                sequence_frame_count=sequence_data.frame_count,
                clip_frames=clip_frames,
                preview_render=rendered_clip[0].detach(),
                preview_features=preview_features,
                camera_state=decoded.camera_state,
                loss=loss.detach(),
                recon_loss=recon_loss.detach(),
                camera_motion_loss=camera_motion_loss.detach(),
                camera_temporal_loss=camera_temporal_loss.detach(),
                camera_global_loss=camera_global_loss.detach(),
                bank_rate_loss=bank_rate_loss.detach(),
                bank_rate_terms={key: value.detach() for key, value in bank_rate_terms.items()},
            )
        finally:
            if was_training:
                self.model.train()

    def step(self, keep_preview: bool = False) -> StepResult:
        self.optimizer.zero_grad(set_to_none=True)
        sequence_data, clip_frames, clip_times = self.sample_clip()
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

        recon_loss, preview_render, preview_features = self.recon_backward(
            clip_frames,
            decoded,
            camera_loss + bank_rate_loss,
            keep_preview,
        )

        self.optimizer.step()
        loss = recon_loss + camera_loss.detach() + bank_rate_loss.detach()
        return StepResult(
            source_path=sequence_data.source_path,
            sequence_frame_count=sequence_data.frame_count,
            clip_frames=clip_frames,
            preview_render=preview_render,
            preview_features=preview_features,
            camera_state=decoded.camera_state,
            loss=loss,
            recon_loss=recon_loss,
            camera_motion_loss=camera_motion_loss,
            camera_temporal_loss=camera_temporal_loss,
            camera_global_loss=camera_global_loss,
            bank_rate_loss=bank_rate_loss.detach(),
            bank_rate_terms={key: value.detach() for key, value in bank_rate_terms.items()},
        )

    def camera_metrics(self, camera_state: CameraState) -> dict[str, float]:
        return {
            "fov_degrees": camera_state.fov_degrees.item(),
            "radius": camera_state.radius.item(),
            "rotation_delta_mean_degrees": (
                torch.rad2deg(torch.linalg.norm(camera_state.rotation_delta, dim=-1)).mean().item()
            ),
            "translation_delta_mean": torch.linalg.norm(camera_state.translation_delta, dim=-1).mean().item(),
        }

    def progress_message(self, result: StepResult) -> str:
        if result.camera_state is None:
            return f"Loss: {result.loss.item():.4f} recon: {result.recon_loss.item():.4f}"
        metrics = self.camera_metrics(result.camera_state)
        return (
            f"Loss: {result.loss.item():.4f} "
            f"recon: {result.recon_loss.item():.4f} "
            f"fov: {metrics['fov_degrees']:.2f} "
            f"r: {metrics['radius']:.2f}"
        )

    def should_log_scalars(self, step: int) -> bool:
        return step % max(1, self.logging_cfg["log_every"]) == 0 or (
            self.logging_cfg["always_log_last_step"] and step == self.train_cfg["steps"]
        )

    def should_log_images(self, step: int) -> bool:
        return step % max(1, self.logging_cfg["image_log_every"]) == 0 or (
            self.logging_cfg["always_log_last_step"] and step == self.train_cfg["steps"]
        )

    def should_log_videos(self, step: int) -> bool:
        return step % max(1, self.logging_cfg["video_log_every"]) == 0 or (
            self.logging_cfg["always_log_last_step"] and step == self.train_cfg["steps"]
        )

    def scalar_payload(self, result: StepResult) -> dict[str, Any]:
        return _scalar_payload_impl(
            self.cfg,
            result,
            train_sequence_count=len(self.train_sequences),
            eval_sequence_count=len(self.eval_sequences),
        )

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
                metrics.update(
                    {
                        "Camera/EvalFOVDegrees": rendered.camera_state.fov_degrees.item(),
                        "Camera/EvalRadius": rendered.camera_state.radius.item(),
                        "Camera/EvalRotationDeltaMeanDegrees": (
                            torch.rad2deg(torch.linalg.norm(rendered.camera_state.rotation_delta, dim=-1))
                            .mean()
                            .item()
                        ),
                        "Camera/EvalTranslationDeltaMean": (
                            torch.linalg.norm(rendered.camera_state.translation_delta, dim=-1).mean().item()
                        ),
                    }
                )
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

    def val_log(self, step: int, result: StepResult) -> None:
        should_log_scalars = self.should_log_scalars(step)
        should_log_images = self.should_log_images(step)
        should_log_videos = self.should_log_videos(step)
        if not (should_log_scalars or should_log_images or should_log_videos):
            return

        payload = self.scalar_payload(result)
        if should_log_images:
            payload["Render_GT_vs_Pred"] = _render_preview_image_impl(self.cfg, result, step)
            if self.feature_pca_log:
                if result.preview_features is None:
                    raise ValueError(
                        "feature_pca_log is on but preview_features was not retained for the training step."
                    )
                from feature_pca_viz import feature_pca_to_rgb
                pca_rgb = feature_pca_to_rgb(result.preview_features)
                pca_image = (
                    pca_rgb.detach().cpu().clamp(0, 1).permute(1, 2, 0) * 255.0
                ).to(torch.uint8).numpy()
                payload["media/feature_pca_image"] = wandb.Image(pca_image, caption=f"Step {step}")
        if should_log_videos:
            payload.update(self.validation_video_payload(step=step))
        wandb.log(payload, step=step)

    def run(self) -> None:
        token_summary = (
            f"{self.model_cfg['tokens']} 3DGS tokens"
            if not self.model_cfg["use_static_dynamic_split"]
            else (
                f"{self.model_cfg['static_tokens']} static + "
                f"{self.model_cfg['dynamic_tokens']} dynamic 3DGS tokens"
            )
        )
        print(
            "Starting DynamicVideoTokenGSImplicitCamera Training: "
            f"{len(self.train_sequences)} train sequence(s), train_frame_count={self.model_cfg['train_frame_count']}, "
            f"input_size={self.model_cfg['size']}, render_size={self.render_size}, "
            f"1 global camera token + 1 path token + {token_summary} x "
            f"{self.model_cfg['gaussians_per_token']} gaussians/token = "
            f"{self.effective_gaussians} explicit Gaussians with {self.renderer_mode} renderer..."
        )
        print(f"Reconstruction backward strategy: {self.recon_backward_strategy}")
        print(
            "Camera model: "
            f"global_head={self.cfg['camera']['global_head']}, "
            f"lens_model={self.cfg['camera']['lens_model']}"
        )
        print(
            "Video encoder: "
            f"backend={self.model_cfg['video_encoder_backend']}, "
            f"vjepa_model_id={self.model_cfg['vjepa_model_id']}"
        )
        print(
            f"Temporal reconstruction chunk size: {self.temporal_recon_chunk_size(self.model_cfg['train_frame_count'])}"
        )
        print(f"Attention backend: {self.attn_backend}")

        initial_result = self.initial_step_result()
        print(f"Step 0 initialization diagnostic: {self.progress_message(initial_result)}")
        self.val_log(0, initial_result)

        pbar = tqdm(range(1, self.train_cfg["steps"] + 1))
        try:
            for step in pbar:
                keep_preview = self.should_log_images(step)
                result = self.step(keep_preview=keep_preview)
                pbar.set_description(self.progress_message(result))
                self.val_log(step, result)
            self.export_browser_bundle()
        finally:
            wandb.finish()

        print("DynamicVideoTokenGSImplicitCamera training complete. Check your Weights & Biases dashboard.")


class KnownCameraTrainer(Trainer):
    def validate_train_sequences(self) -> None:
        super().validate_train_sequences()
        missing = [sequence for sequence in self.train_sequences if sequence.cameras is None]
        if missing:
            raise ValueError(
                "Known-camera training requires cameras on every train sequence. "
                f"Missing camera metadata for {len(missing)} sequence(s)."
            )

    def sample_clip(self) -> tuple[SequenceData, torch.Tensor, torch.Tensor, tuple[Any, ...]]:
        sequence_data = self.sample_sequence()
        clip_indices = select_window_indices(
            sequence_data.frame_count,
            self.model_cfg["train_frame_count"],
            device=self.device,
        )
        clip_frames, clip_times = prepare_clip(sequence_data, clip_indices)
        if sequence_data.cameras is None:
            raise ValueError("Known-camera sequence has no cameras.")
        clip_cameras = tuple(sequence_data.cameras[index] for index in clip_indices.detach().cpu().tolist())
        return sequence_data, clip_frames, clip_times, clip_cameras

    def forward_known_clip(
        self,
        clip_frames: torch.Tensor,
        clip_times: torch.Tensor,
        clip_cameras: tuple[Any, ...],
    ) -> GaussianSequence:
        with fast_attn_context(self.device), self.autocast_context():
            return self.model(clip_frames, decode_times=clip_times, cameras=clip_cameras)

    def step(self, keep_preview: bool = False) -> StepResult:
        self.optimizer.zero_grad(set_to_none=True)
        sequence_data, clip_frames, clip_times, clip_cameras = self.sample_clip()
        decoded = self.forward_known_clip(clip_frames, clip_times, clip_cameras)
        if decoded.cameras is None:
            raise ValueError("Known-camera video decode must include cameras.")

        zero = clip_frames.new_tensor(0.0)
        bank_rate_loss, bank_rate_terms = _build_bank_rate_loss_impl(decoded, self.loss_cfg)
        recon_loss, preview_render, preview_features = self.recon_backward(
            clip_frames,
            decoded,
            bank_rate_loss,
            keep_preview,
        )

        self.optimizer.step()
        loss = recon_loss + bank_rate_loss.detach()
        return StepResult(
            source_path=sequence_data.source_path,
            sequence_frame_count=sequence_data.frame_count,
            clip_frames=clip_frames,
            preview_render=preview_render,
            preview_features=preview_features,
            camera_state=None,
            loss=loss,
            recon_loss=recon_loss,
            camera_motion_loss=zero,
            camera_temporal_loss=zero,
            camera_global_loss=zero,
            bank_rate_loss=bank_rate_loss.detach(),
            bank_rate_terms={key: value.detach() for key, value in bank_rate_terms.items()},
        )

    @torch.no_grad()
    def initial_step_result(self) -> StepResult:
        was_training = self.model.training
        self.model.eval()
        try:
            sequence_data = self.train_sequences[0]
            if sequence_data.cameras is None:
                raise ValueError("Known-camera sequence has no cameras.")
            clip_length = int(self.model_cfg["train_frame_count"])
            clip_indices = torch.arange(0, clip_length, device=self.device)
            clip_frames, clip_times = prepare_clip(sequence_data, clip_indices)
            clip_cameras = tuple(sequence_data.cameras[index] for index in clip_indices.detach().cpu().tolist())
            decoded = self.forward_known_clip(clip_frames, clip_times, clip_cameras)
            if decoded.cameras is None:
                raise ValueError("Known-camera video decode must include cameras.")

            zero = clip_frames.new_tensor(0.0)
            bank_rate_loss, bank_rate_terms = _build_bank_rate_loss_impl(decoded, self.loss_cfg)
            rendered = self.render_decoded_rgb_clip(
                decoded,
                frames=clip_frames[0],
                frame_indices=clip_indices,
                frame_times=clip_times,
                phase="eval",
                view_id="initial_known_view",
            )
            preview_features = rendered.features[0].detach() if self.feature_pca_log else None
            rendered_clip = rendered.rgb
            recon_loss = self.rgb_objective.reconstruction_loss(rendered)
            loss = recon_loss + bank_rate_loss
            return StepResult(
                source_path=sequence_data.source_path,
                sequence_frame_count=sequence_data.frame_count,
                clip_frames=clip_frames,
                preview_render=rendered_clip[0].detach(),
                preview_features=preview_features,
                camera_state=None,
                loss=loss.detach(),
                recon_loss=recon_loss.detach(),
                camera_motion_loss=zero,
                camera_temporal_loss=zero,
                camera_global_loss=zero,
                bank_rate_loss=bank_rate_loss.detach(),
                bank_rate_terms={key: value.detach() for key, value in bank_rate_terms.items()},
            )
        finally:
            if was_training:
                self.model.train()

    def _eval_decode_clip(
        self,
        sequence_data: SequenceData,
        clip_indices: torch.Tensor,
        clip_frames: torch.Tensor,
        clip_times: torch.Tensor,
    ) -> GaussianSequence:
        if sequence_data.cameras is None:
            raise ValueError("Known-camera sequence has no cameras.")
        clip_cameras = tuple(sequence_data.cameras[i] for i in clip_indices.detach().cpu().tolist())
        return self.forward_known_clip(clip_frames, clip_times, clip_cameras)

    @torch.no_grad()
    def render_full_sequence(
        self,
        sequence_data: SequenceData,
    ) -> RenderedClip:
        return _render_full_sequence_impl(
            self.cfg, self.model, sequence_data, self._eval_decode_clip, self._eval_render_clip
        )

    def run(self) -> None:
        print(
            "Starting DynamicVideoTokenGSKnownCamera Training: "
            f"{len(self.train_sequences)} train sequence(s), train_frame_count={self.model_cfg['train_frame_count']}, "
            f"input_size={self.model_cfg['size']}, render_size={self.render_size}, "
            f"{self.model_cfg['tokens']} 3DGS tokens x {self.model_cfg['gaussians_per_token']} gaussians/token = "
            f"{self.effective_gaussians} explicit Gaussians with {self.renderer_mode} renderer..."
        )
        print(f"Reconstruction backward strategy: {self.recon_backward_strategy}")
        print("Camera model: known/precomputed")
        print(
            "Video encoder: "
            f"backend={self.model_cfg['video_encoder_backend']}, "
            f"vjepa_model_id={self.model_cfg['vjepa_model_id']}"
        )
        print(
            f"Temporal reconstruction chunk size: {self.temporal_recon_chunk_size(self.model_cfg['train_frame_count'])}"
        )
        print(f"Attention backend: {self.attn_backend}")

        initial_result = self.initial_step_result()
        print(f"Step 0 initialization diagnostic: {self.progress_message(initial_result)}")
        self.val_log(0, initial_result)

        pbar = tqdm(range(1, self.train_cfg["steps"] + 1))
        try:
            for step in pbar:
                keep_preview = self.should_log_images(step)
                result = self.step(keep_preview=keep_preview)
                pbar.set_description(self.progress_message(result))
                self.val_log(step, result)
        finally:
            wandb.finish()

        print("DynamicVideoTokenGSKnownCamera training complete. Check your Weights & Biases dashboard.")


def trainer_class_for_config(config: dict[str, Any]) -> type[Trainer]:
    variant = str(config.get("model", {}).get("variant", "learned_time_orbit_path")).lower()
    if variant == "known_camera":
        return KnownCameraTrainer
    return Trainer


def run_training(config: dict[str, Any]) -> None:
    trainer_class_for_config(config)(config).run()


def main(config: dict[str, Any] | str | Path) -> None:
    if isinstance(config, (str, Path)):
        run_training(load_config_file(config))
    else:
        run_training(config)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: uv run python src/train/train_video_token_implicit_dynamic.py "
            "src/train_configs/local_mac_overfit_video_token_full.jsonc"
        )
    main(sys.argv[1])
