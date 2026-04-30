from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from dataclasses import dataclass
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
from model_factories import build_colorizer, build_model_module
from objective import (
    BackgroundSample,
    RasterizedView,
    RenderedView,
    RGBReconObjective,
    RunPhase,
    TargetView,
    objective_spec_from_loss_config,
)
from pipeline.diagnostics import (
    camera_temporal_payload,
    decoded_temporal_payload,
    eval_metric_payload,
    fill_decoded_frame_buffers,
    init_decoded_frame_buffers,
    temporal_similarity_payload,
)
from pipeline.losses import (
    build_bank_rate_loss as _build_bank_rate_loss_impl,
)
from pipeline.losses import (
    build_camera_loss as _build_camera_loss_impl,
)
from pipeline.losses import (
    compute_camera_losses as _compute_camera_losses_impl,
)
from pipeline.losses import (
    temporal_recon_chunk_size as _temporal_recon_chunk_size_impl,
)
from pipeline.render import (
    FullSequenceDeps,
    colorize_view_dirs_for_features,
    gaussian_sequence_slice,
    render_clip_sequence,
    render_full_sequence as _render_full_sequence_impl,
    render_full_sequence_known_cameras as _render_full_sequence_known_cameras_impl,
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
from runtime_types import CameraState, GaussianSequence, SequenceData
from sequence_data import load_camera_sequence, load_uncalibrated_sequence, resolve_frames_dir, select_window_indices
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


@dataclass
class StepResult:
    source_path: Path | None
    sequence_frame_count: int
    clip_frames: torch.Tensor
    preview_render: torch.Tensor | None
    preview_features: torch.Tensor | None
    camera_state: CameraState | None
    loss: torch.Tensor
    recon_loss: torch.Tensor
    camera_motion_loss: torch.Tensor
    camera_temporal_loss: torch.Tensor
    camera_global_loss: torch.Tensor
    bank_rate_loss: torch.Tensor
    bank_rate_terms: dict[str, torch.Tensor]


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
        "free_gaussian_bank",
        "free_linear_splats",
        "free_linear_time_splats",
        "linear_free_splats",
        "unconditioned_residual_free_bank",
        "residual_free_bank_unconditioned_tokens",
        "unconditioned_tokens",
        "token_decoder_unconditioned",
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
            "free_gaussian_bank",
            "free_linear_splats",
            "free_linear_time_splats",
            "linear_free_splats",
            "residual_free_bank",
            "residual_free_bank_video_tokens",
            "residual_free_video",
            "known_camera",
            "known_camera_video_token",
            "unconditioned_residual_free_bank",
            "residual_free_bank_unconditioned_tokens",
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


def prepare_clip(sequence_data: SequenceData, clip_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    clip_frames = sequence_data.frames[clip_indices]
    time_denominator = max(sequence_data.frame_count - 1, 1)
    clip_times = (clip_indices.to(dtype=torch.float32) / float(time_denominator)).reshape(1, -1)
    return clip_frames.unsqueeze(0), clip_times


def load_manifest_entries(manifest_path: Path, split: str | None = None) -> list[dict[str, Any]]:
    entries = []
    with manifest_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            entry = json.loads(stripped)
            if not isinstance(entry, dict):
                raise ValueError(f"Expected object on {manifest_path}:{line_number}.")
            if split is None or str(entry.get("split", "train")) == split:
                entries.append(entry)
    if not entries:
        split_text = f" split={split!r}" if split is not None else ""
        raise ValueError(f"No manifest entries found in {manifest_path}{split_text}.")
    return entries


def load_manifest_sequence(
    entry: dict[str, Any],
    *,
    data_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    device: torch.device,
) -> SequenceData:
    sequence_dir = Path(entry["sequence_dir"])
    frames_dir = path_or_none(entry.get("frames_dir"))
    if frames_dir is None:
        frames_dir = resolve_frames_dir(sequence_dir, data_cfg["frames_dir"])
    video_path = path_or_none(entry.get("video_path"))
    frame_source = entry.get("frame_source", data_cfg["frame_source"])
    max_frames = int(entry.get("max_frames", data_cfg["max_frames"]))
    if frame_source == "camera_json":
        camera_json = path_or_none(entry.get("camera_json")) or data_cfg["camera_json"] or (
            sequence_dir / "per_frame_cameras.json"
        )
        return load_camera_sequence(
            camera_json_path=camera_json,
            target_size=model_cfg["size"],
            camera_image_size=int(entry.get("camera_image_size", data_cfg["camera_image_size"])),
            max_frames=max_frames,
            focal_mode=str(entry.get("camera_focal_mode", data_cfg["camera_focal_mode"])),
            device=device,
        )
    return load_uncalibrated_sequence(
        sequence_dir=sequence_dir,
        frames_dir=frames_dir,
        video_path=video_path,
        target_size=model_cfg["size"],
        max_frames=max_frames,
        frame_source=frame_source,
        device=device,
    )


def load_manifest_sequences(
    manifest_path: Path,
    *,
    split: str,
    data_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    device: torch.device,
    limit: int = 0,
) -> list[SequenceData]:
    entries = load_manifest_entries(manifest_path, split=split)
    if limit > 0:
        entries = entries[:limit]
    return [
        load_manifest_sequence(
            entry,
            data_cfg=data_cfg,
            model_cfg=model_cfg,
            device=device,
        )
        for entry in entries
    ]


def stack_complete_frame_list(frames: list[torch.Tensor | None], *, name: str) -> torch.Tensor:
    complete_frames = []
    missing_indices = []
    for index, frame in enumerate(frames):
        if frame is None:
            missing_indices.append(index)
        else:
            complete_frames.append(frame)
    if missing_indices:
        preview = ", ".join(str(index) for index in missing_indices[:8])
        suffix = ", ..." if len(missing_indices) > 8 else ""
        raise RuntimeError(f"{name} did not cover all frames; missing indices: {preview}{suffix}")
    if not complete_frames:
        raise RuntimeError(f"{name} did not produce any frames.")
    return torch.stack(complete_frames, dim=0)


def build_model_from_config(config: dict[str, Any]) -> torch.nn.Module:
    return build_model_module(config["model"], config["camera"])


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
        if self.recon_backward_strategy not in {"framewise", "microbatch", "batched"}:
            raise ValueError(
                f"Unsupported recon_backward_strategy={self.recon_backward_strategy!r}. "
                "Expected one of: framewise, microbatch, batched."
            )
        self.temporal_microbatch_size = int(self.train_cfg["temporal_microbatch_size"])
        if self.temporal_microbatch_size < 1:
            raise ValueError(f"temporal_microbatch_size must be >= 1, got {self.temporal_microbatch_size}.")
        self.render_size = int(self.render_cfg["render_size"])
        if self.render_size < 1:
            raise ValueError(f"render_size must be >= 1, got {self.render_size}.")

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
        self.colorize_view_condition = colorizer.view_condition
        self.colorize_detach_view_condition = colorizer.detach_view_condition
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
        self.amp_dtype = (
            torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
        )
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
        if self.colorize is None or self.colorize_view_condition == "none":
            return None
        return colorize_view_dirs_for_features(
            features,
            cameras,
            view_condition=self.colorize_view_condition,
            input_size=self.model_cfg["size"],
            render_size=self.render_size,
            detach=self.colorize_detach_view_condition,
        )

    def make_target_view(
        self,
        *,
        view_id: str,
        frames: torch.Tensor,
        frame_indices: torch.Tensor,
        frame_times: torch.Tensor,
        cameras: tuple[Any, ...],
        role: str = "train",
        camera_role: str = "target",
        camera_owner: str = "model",
        camera_name: str | None = None,
        metrics_prefix: str | None = None,
        log_media: bool = False,
    ) -> TargetView:
        return TargetView(
            view_id=view_id,
            role=role,  # type: ignore[arg-type]
            camera_role=camera_role,  # type: ignore[arg-type]
            camera_owner=camera_owner,  # type: ignore[arg-type]
            frames=frames,
            frame_indices=frame_indices,
            frame_times=frame_times.reshape(-1),
            video_fps=self.sequence_data.video_fps,
            camera_name=camera_name,
            cameras=cameras,
            metrics_prefix=metrics_prefix,
            log_media=log_media,
        )
        return self.colorize(features, view_dirs=view_dirs)

    def rasterize_decoded_clip(
        self,
        decoded: GaussianSequence,
        cameras: tuple[Any, ...],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return render_clip_sequence(
            decoded,
            cameras,
            renderer_mode=self.renderer_mode,
            render_cfg=self.render_cfg,
            input_size=self.model_cfg["size"],
            render_size=self.render_size,
            dense_grid=self.dense_grid,
        )

    def rasterize(self, decoded: GaussianSequence, target: TargetView) -> RasterizedView:
        if target.cameras is None:
            raise ValueError("TargetView.cameras is required for rasterization.")
        features, alpha = self.rasterize_decoded_clip(decoded, target.cameras)
        return RasterizedView(
            view=target,
            features=features,
            alpha=alpha,
            cameras=target.cameras,
            view_dirs=self.view_dirs_for_features(features, target.cameras),
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

    def compute_camera_losses(
        self,
        clip_times: torch.Tensor,
        camera_state: CameraState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _compute_camera_losses_impl(clip_times, camera_state)

    def build_camera_loss(
        self,
        clip_times: torch.Tensor,
        camera_state: CameraState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return _build_camera_loss_impl(clip_times, camera_state, self.loss_cfg)

    def build_bank_rate_loss(self, decoded: GaussianSequence) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return _build_bank_rate_loss_impl(decoded, self.loss_cfg)

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

            camera_loss, camera_motion_loss, camera_temporal_loss, camera_global_loss = self.build_camera_loss(
                clip_times,
                decoded.camera_state,
            )
            bank_rate_loss, bank_rate_terms = self.build_bank_rate_loss(decoded)
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

        camera_loss, camera_motion_loss, camera_temporal_loss, camera_global_loss = self.build_camera_loss(
            clip_times,
            decoded.camera_state,
        )
        bank_rate_loss, bank_rate_terms = self.build_bank_rate_loss(decoded)

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
            result,
            train_frame_count=int(self.model_cfg["train_frame_count"]),
            train_sequence_count=len(self.train_sequences),
            eval_sequence_count=len(self.eval_sequences),
            input_size=int(self.model_cfg["size"]),
            render_size=int(self.render_size),
        )

    def render_preview_image(self, result: StepResult, step: int) -> wandb.Image:
        return _render_preview_image_impl(result, render_size=self.render_size, step=step)

    def _full_sequence_deps(self) -> FullSequenceDeps:
        def decode_clip(
            sequence_data: SequenceData,
            clip_indices: torch.Tensor,
            clip_frames: torch.Tensor,
            clip_times: torch.Tensor,
        ) -> GaussianSequence:
            del clip_indices
            model_input = self.model_input_for_clip(sequence_data, clip_frames, clip_times)
            return self.forward_clip(model_input, clip_times)

        def render_clip(
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

        return FullSequenceDeps(
            decode_clip=decode_clip,
            render_clip=render_clip,
            prepare_clip=prepare_clip,
            init_decoded_frame_buffers=init_decoded_frame_buffers,
            fill_decoded_frame_buffers=fill_decoded_frame_buffers,
            decoded_temporal_payload=decoded_temporal_payload,
            stack_complete_frame_list=stack_complete_frame_list,
        )

    @torch.no_grad()
    def render_full_sequence(
        self,
        sequence_data: SequenceData,
    ) -> tuple[torch.Tensor, CameraState | None, dict[str, float], torch.Tensor | None, torch.Tensor | None]:
        clip = _render_full_sequence_impl(
            model=self.model,
            sequence_data=sequence_data,
            train_frame_count=int(self.model_cfg["train_frame_count"]),
            feature_pca_log=self.feature_pca_log,
            device=self.device,
            deps=self._full_sequence_deps(),
        )
        return (
            clip.rgb_sequence,
            clip.camera_state,
            clip.temporal_metrics,
            clip.feature_sequence,
            clip.alpha_sequence,
        )

    def validation_video_payload(self) -> dict[str, Any]:
        sequences = self.eval_sequences or [self.sequence_data]
        metric_payloads = []
        payload: dict[str, Any] = {
            "Eval/SequenceCount": len(sequences),
        }
        for sequence_index, sequence_data in enumerate(sequences):
            (
                rendered_sequence,
                eval_camera_state,
                decoded_metrics,
                feature_sequence,
                alpha_sequence,
            ) = self.render_full_sequence(sequence_data)
            gt_sequence = resize_images(sequence_data.frames, self.render_size).detach().cpu()
            metrics = {
                **eval_metric_payload(rendered_sequence, gt_sequence, self.loss_cfg),
                **temporal_similarity_payload(rendered_sequence, gt_sequence, self.loss_cfg),
                **decoded_metrics,
            }
            if eval_camera_state is not None:
                metrics.update(
                    {
                        "Camera/EvalFOVDegrees": eval_camera_state.fov_degrees.item(),
                        "Camera/EvalRadius": eval_camera_state.radius.item(),
                        "Camera/EvalRotationDeltaMeanDegrees": (
                            torch.rad2deg(torch.linalg.norm(eval_camera_state.rotation_delta, dim=-1)).mean().item()
                        ),
                        "Camera/EvalTranslationDeltaMean": (
                            torch.linalg.norm(eval_camera_state.translation_delta, dim=-1).mean().item()
                        ),
                    }
                )
                metrics.update(camera_temporal_payload(eval_camera_state))
            metric_payloads.append(metrics)
            sequence_payload, self.gt_video_logged = single_cam_validation_video_payload(
                sequence_index=sequence_index,
                rendered_sequence=rendered_sequence,
                gt_sequence=gt_sequence,
                feature_sequence=feature_sequence,
                alpha_sequence=alpha_sequence,
                eval_payload={},
                feature_pca_log=self.feature_pca_log,
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
            payload["Render_GT_vs_Pred"] = self.render_preview_image(result, step)
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
            payload.update(self.validation_video_payload())
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
            examples = ", ".join(str(sequence.source_path) for sequence in missing[:3])
            raise ValueError(f"Known-camera training requires cameras on every train sequence. Examples: {examples}")

    def sample_clip(self) -> tuple[SequenceData, torch.Tensor, torch.Tensor, tuple[Any, ...]]:
        sequence_data = self.sample_sequence()
        clip_indices = select_window_indices(
            sequence_data.frame_count,
            self.model_cfg["train_frame_count"],
            device=self.device,
        )
        clip_frames, clip_times = prepare_clip(sequence_data, clip_indices)
        if sequence_data.cameras is None:
            raise ValueError(f"Known-camera sequence has no cameras: {sequence_data.source_path}")
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
        bank_rate_loss, bank_rate_terms = self.build_bank_rate_loss(decoded)
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
                raise ValueError(f"Known-camera sequence has no cameras: {sequence_data.source_path}")
            clip_length = int(self.model_cfg["train_frame_count"])
            clip_indices = torch.arange(0, clip_length, device=self.device)
            clip_frames, clip_times = prepare_clip(sequence_data, clip_indices)
            clip_cameras = tuple(sequence_data.cameras[index] for index in clip_indices.detach().cpu().tolist())
            decoded = self.forward_known_clip(clip_frames, clip_times, clip_cameras)
            if decoded.cameras is None:
                raise ValueError("Known-camera video decode must include cameras.")

            zero = clip_frames.new_tensor(0.0)
            bank_rate_loss, bank_rate_terms = self.build_bank_rate_loss(decoded)
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

    def _full_sequence_deps(self) -> FullSequenceDeps:
        def decode_clip(
            sequence_data: SequenceData,
            clip_indices: torch.Tensor,
            clip_frames: torch.Tensor,
            clip_times: torch.Tensor,
        ) -> GaussianSequence:
            if sequence_data.cameras is None:
                raise ValueError(f"Known-camera sequence has no cameras: {sequence_data.source_path}")
            clip_cameras = tuple(
                sequence_data.cameras[index] for index in clip_indices.detach().cpu().tolist()
            )
            return self.forward_known_clip(clip_frames, clip_times, clip_cameras)

        def render_clip(
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

        return FullSequenceDeps(
            decode_clip=decode_clip,
            render_clip=render_clip,
            prepare_clip=prepare_clip,
            init_decoded_frame_buffers=init_decoded_frame_buffers,
            fill_decoded_frame_buffers=fill_decoded_frame_buffers,
            decoded_temporal_payload=decoded_temporal_payload,
            stack_complete_frame_list=stack_complete_frame_list,
        )

    @torch.no_grad()
    def render_full_sequence(
        self,
        sequence_data: SequenceData,
    ) -> tuple[torch.Tensor, CameraState | None, dict[str, float], torch.Tensor | None, torch.Tensor | None]:
        clip = _render_full_sequence_known_cameras_impl(
            model=self.model,
            sequence_data=sequence_data,
            train_frame_count=int(self.model_cfg["train_frame_count"]),
            feature_pca_log=self.feature_pca_log,
            device=self.device,
            deps=self._full_sequence_deps(),
        )
        return (
            clip.rgb_sequence,
            clip.camera_state,
            clip.temporal_metrics,
            clip.feature_sequence,
            clip.alpha_sequence,
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
    if variant in {"known_camera", "known_camera_video_token"}:
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
