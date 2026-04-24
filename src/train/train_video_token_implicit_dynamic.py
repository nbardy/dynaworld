from __future__ import annotations

import math
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import wandb
from config_utils import apply_defaults, load_config_file, path_or_none, resolved_config, serialize_config_value
from dynamicTokenGS import (
    configure_fast_attn,
    fast_attn_context,
    pick_device,
)
from gs_models import (
    DynamicVideoTokenGSImplicitCamera,
    DynamicVideoTokenGSImplicitCameraPoseToPlucker,
    DynamicVideoTokenGSImplicitCameraSinusoidalTime,
)
from losses import reconstruction_loss_per_image, ssim_per_image
from rendering import build_or_reuse_grid, camera_for_viewport, render_gaussian_frames, resize_images
from rendering import pick_renderer_mode as resolve_renderer_mode
from runtime_types import CameraState, GaussianSequence, SequenceData
from sequence_data import load_uncalibrated_sequence, resolve_frames_dir, select_window_indices
from tqdm import tqdm
from train_logging import build_validation_video_payload, make_preview_image, make_wandb_video

LOSS_OPTION_DEFAULTS = {
    "type": "l1_mse",
    "l1_weight": 1.0,
    "mse_weight": 0.2,
    "dssim_weight": 0.2,
    "ssim_window_size": 11,
    "ssim_c1": 0.0001,
    "ssim_c2": 0.0009,
    "camera_motion_weight": 0.01,
    "camera_temporal_weight": 0.02,
    "camera_global_weight": 0.005,
    "static_alpha_rate_weight": 0.0,
    "dynamic_alpha_rate_weight": 0.0,
    "dynamic_motion_rate_weight": 0.0,
    "dynamic_rotation_rate_weight": 0.0,
    "dynamic_alpha_time_rate_weight": 0.0,
}


MODEL_OPTION_DEFAULTS = {
    "variant": "learned_time_orbit_path",
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


@dataclass
class StepResult:
    clip_frames: torch.Tensor
    preview_render: torch.Tensor | None
    camera_state: CameraState
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
    cfg["data"]["sequence_dir"] = Path(cfg["data"]["sequence_dir"])
    cfg["data"]["frames_dir"] = path_or_none(cfg["data"]["frames_dir"])
    cfg["data"]["video_path"] = path_or_none(cfg["data"]["video_path"])
    apply_defaults(cfg["model"], MODEL_OPTION_DEFAULTS)
    cfg["model"]["variant"] = str(cfg["model"]["variant"]).lower()
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


def viewport_cameras(
    cameras: tuple[Any, ...],
    *,
    input_size: int,
    render_size: int,
) -> tuple[Any, ...]:
    return tuple(
        camera_for_viewport(
            camera,
            source_height=input_size,
            source_width=input_size,
            target_height=render_size,
            target_width=render_size,
        )
        for camera in cameras
    )


def gaussian_sequence_slice(sequence: GaussianSequence, start: int, end: int) -> GaussianSequence:
    cameras = None
    if sequence.cameras is not None:
        cameras = tuple(sequence.cameras[start:end])
    return GaussianSequence(
        xyz=sequence.xyz[start:end],
        scales=sequence.scales[start:end],
        quats=sequence.quats[start:end],
        opacities=sequence.opacities[start:end],
        rgbs=sequence.rgbs[start:end],
        cameras=cameras,
        camera_state=sequence.camera_state,
        auxiliary=sequence.auxiliary,
    )


def render_clip_sequence(
    sequence: GaussianSequence,
    cameras: tuple[Any, ...],
    *,
    renderer_mode: str,
    render_cfg: dict[str, Any],
    input_size: int,
    render_size: int,
    dense_grid: torch.Tensor,
) -> torch.Tensor:
    render_cameras = viewport_cameras(cameras, input_size=input_size, render_size=render_size)
    return render_gaussian_frames(
        sequence,
        render_cameras,
        height=render_size,
        width=render_size,
        mode=renderer_mode,
        dense_grid=dense_grid,
        tile_size=render_cfg["tile_size"],
        bound_scale=render_cfg["bound_scale"],
        alpha_threshold=render_cfg["alpha_threshold"],
        near_plane=render_cfg["near_plane"],
        fast_mac_options=render_cfg["fast_mac"],
        camera_projection=render_cfg["camera_projection"],
    )


def eval_metric_payload(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_cfg: dict[str, Any],
) -> dict[str, float]:
    prediction = prediction.float()
    target = target.float()
    delta = prediction - target
    l1 = delta.abs().flatten(1).mean()
    mse = delta.square().flatten(1).mean()
    ssim = ssim_per_image(
        prediction,
        target,
        window_size=loss_cfg["ssim_window_size"],
        c1=float(loss_cfg["ssim_c1"]),
        c2=float(loss_cfg["ssim_c2"]),
    ).mean()
    dssim = (1.0 - ssim) * 0.5
    recon_loss = reconstruction_loss_per_image(prediction, target, loss_cfg).mean()
    psnr = -10.0 * math.log10(max(float(mse.item()), 1.0e-12))
    return {
        "Eval/Loss": float(recon_loss.item()),
        "Eval/L1": float(l1.item()),
        "Eval/MSE": float(mse.item()),
        "Eval/SSIM": float(ssim.item()),
        "Eval/DSSIM": float(dssim.item()),
        "Eval/PSNR": psnr,
    }


def temporal_similarity_payload(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_cfg: dict[str, Any],
) -> dict[str, float]:
    if prediction.shape[0] < 2:
        return {}

    prediction = prediction.float()
    target = target.float()
    pred_adj_l1 = (prediction[1:] - prediction[:-1]).abs().flatten(1).mean().mean()
    gt_adj_l1 = (target[1:] - target[:-1]).abs().flatten(1).mean().mean()
    pred_to_first_l1 = (prediction[1:] - prediction[:1]).abs().flatten(1).mean().mean()
    gt_to_first_l1 = (target[1:] - target[:1]).abs().flatten(1).mean().mean()
    pred_adj_ssim = ssim_per_image(
        prediction[1:],
        prediction[:-1],
        window_size=loss_cfg["ssim_window_size"],
        c1=float(loss_cfg["ssim_c1"]),
        c2=float(loss_cfg["ssim_c2"]),
    ).mean()
    gt_adj_ssim = ssim_per_image(
        target[1:],
        target[:-1],
        window_size=loss_cfg["ssim_window_size"],
        c1=float(loss_cfg["ssim_c1"]),
        c2=float(loss_cfg["ssim_c2"]),
    ).mean()
    return {
        "Eval/TemporalPredAdjacentL1": float(pred_adj_l1.item()),
        "Eval/TemporalGTAdjacentL1": float(gt_adj_l1.item()),
        "Eval/TemporalAdjacentL1Ratio": float((pred_adj_l1 / gt_adj_l1.clamp_min(1.0e-8)).item()),
        "Eval/TemporalPredToFirstL1": float(pred_to_first_l1.item()),
        "Eval/TemporalGTToFirstL1": float(gt_to_first_l1.item()),
        "Eval/TemporalToFirstL1Ratio": float((pred_to_first_l1 / gt_to_first_l1.clamp_min(1.0e-8)).item()),
        "Eval/TemporalPredAdjacentSSIM": float(pred_adj_ssim.item()),
        "Eval/TemporalGTAdjacentSSIM": float(gt_adj_ssim.item()),
    }


@torch.no_grad()
def render_full_sequence(
    model: torch.nn.Module,
    sequence_data: SequenceData,
    config: dict[str, Any],
    renderer_mode: str,
    dense_grid: torch.Tensor,
    amp_available: bool,
    amp_dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, CameraState]:
    was_training = model.training
    model.eval()
    model_cfg = config["model"]
    render_cfg = config["render"]
    clip_length = model_cfg["train_frame_count"]
    num_frames = sequence_data.frame_count
    rendered_frames = [None] * num_frames
    camera_states = []

    for end in range(clip_length, num_frames + clip_length, clip_length):
        clip_end = min(end, num_frames)
        clip_start = max(0, clip_end - clip_length)
        clip_indices = torch.arange(clip_start, clip_end, device=device)
        clip_frames, clip_times = prepare_clip(sequence_data, clip_indices)

        autocast_context = torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_available else nullcontext()
        with fast_attn_context(device), autocast_context:
            decoded = model(clip_frames, decode_times=clip_times)
        if decoded.cameras is None:
            raise ValueError("Implicit-camera video decode must include cameras.")
        if decoded.camera_state is None:
            raise ValueError("Implicit-camera video decode must include camera_state.")
        camera_states.append(decoded.camera_state)
        rendered_clip = render_clip_sequence(
            decoded,
            decoded.cameras,
            renderer_mode=renderer_mode,
            render_cfg=render_cfg,
            input_size=model_cfg["size"],
            render_size=render_cfg["render_size"],
            dense_grid=dense_grid,
        ).cpu()

        for local_index, frame_index in enumerate(clip_indices.tolist()):
            if rendered_frames[frame_index] is not None:
                continue
            rendered_frames[frame_index] = rendered_clip[local_index]

    if was_training:
        model.train()
    merged_camera_state = CameraState(
        fov_degrees=torch.stack([state.fov_degrees for state in camera_states]).mean(),
        radius=torch.stack([state.radius for state in camera_states]).mean(),
        global_residuals=torch.stack([state.global_residuals for state in camera_states]).mean(dim=0),
        rotation_delta=torch.cat([state.rotation_delta for state in camera_states], dim=0),
        translation_delta=torch.cat([state.translation_delta for state in camera_states], dim=0),
        path_residuals=torch.cat([state.path_residuals for state in camera_states], dim=0),
    )
    return torch.stack(rendered_frames, dim=0), merged_camera_state


def build_model_from_config(config: dict[str, Any]) -> DynamicVideoTokenGSImplicitCamera:
    model_cfg = config["model"]
    camera_cfg = config["camera"]
    model_variant = str(model_cfg["variant"]).lower()
    if model_variant == "learned_time_orbit_path":
        model_cls = DynamicVideoTokenGSImplicitCamera
    elif model_variant == "sinusoidal_time_path_mlp":
        model_cls = DynamicVideoTokenGSImplicitCameraSinusoidalTime
    elif model_variant == "token_to_pose_to_plucker":
        model_cls = DynamicVideoTokenGSImplicitCameraPoseToPlucker
    else:
        raise ValueError(
            f"Unknown model.variant={model_variant!r}. "
            "Expected one of: learned_time_orbit_path, sinusoidal_time_path_mlp, token_to_pose_to_plucker."
        )
    model_kwargs = dict(
        clip_length=model_cfg["train_frame_count"],
        image_size=model_cfg["size"],
        num_tokens=model_cfg["tokens"],
        feat_dim=model_cfg["model_dim"],
        bottleneck_dim=model_cfg["bottleneck_dim"],
        num_heads=model_cfg["num_heads"],
        mlp_ratio=model_cfg["mlp_ratio"],
        gaussians_per_token=model_cfg["gaussians_per_token"],
        scene_extent=model_cfg["scene_extent"],
        xy_extent=model_cfg["xy_extent"],
        z_min=model_cfg["z_min"],
        z_max=model_cfg["z_max"],
        scale_init=model_cfg["scale_init"],
        scale_init_log_jitter=model_cfg["scale_init_log_jitter"],
        opacity_init=model_cfg["opacity_init"],
        query_token_init_std=model_cfg["query_token_init_std"],
        head_hidden_dim=model_cfg["head_hidden_dim"],
        head_hidden_layers=model_cfg["head_hidden_layers"],
        head_output_init_std=model_cfg["head_output_init_std"],
        position_init_extent_coverage=model_cfg["position_init_extent_coverage"],
        rotation_init=model_cfg["rotation_init"],
        tubelet_size=(
            model_cfg["tubelet_size_t"],
            model_cfg["patch_compression"],
            model_cfg["patch_compression"],
        ),
        encoder_self_attn_layers=model_cfg["encoder_self_attn_layers"],
        bottleneck_self_attn_layers=model_cfg["bottleneck_self_attn_layers"],
        cross_attn_layers=model_cfg["cross_attn_layers"],
        base_fov_degrees=camera_cfg["base_fov_degrees"],
        base_radius=camera_cfg["base_radius"],
        max_fov_delta_degrees=camera_cfg["max_fov_delta_degrees"],
        max_radius_scale=camera_cfg["max_radius_scale"],
        camera_global_head=camera_cfg["global_head"],
        lens_model=camera_cfg["lens_model"],
        max_aspect_log_delta=camera_cfg["max_aspect_log_delta"],
        max_principal_point_delta=camera_cfg["max_principal_point_delta"],
        distortion_max_abs=camera_cfg["distortion_max_abs"],
        base_distortion=camera_cfg["base_distortion"],
        max_rotation_degrees=camera_cfg["max_rotation_degrees"],
        max_translation_ratio=camera_cfg["max_translation_ratio"],
        static_tokens=model_cfg["static_tokens"],
        dynamic_tokens=model_cfg["dynamic_tokens"],
        dynamic_time_basis_count=model_cfg["dynamic_time_basis_count"],
        dynamic_time_max_frequency=model_cfg["dynamic_time_max_frequency"],
        dynamic_motion_extent=model_cfg["dynamic_motion_extent"],
        dynamic_rotation_degrees=model_cfg["dynamic_rotation_degrees"],
        dynamic_alpha_logit_extent=model_cfg["dynamic_alpha_logit_extent"],
        dynamic_coeff_output_init_std=model_cfg["dynamic_coeff_output_init_std"],
    )
    if model_variant in {"sinusoidal_time_path_mlp", "token_to_pose_to_plucker"}:
        model_kwargs["time_fourier_bands"] = model_cfg["time_fourier_bands"]
        model_kwargs["time_max_frequency"] = model_cfg["time_max_frequency"]
    if model_variant == "token_to_pose_to_plucker":
        model_kwargs["ray_condition_grid_size"] = model_cfg["ray_condition_grid_size"]
    return model_cls(**model_kwargs)


class Trainer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.cfg = resolve_config(config)
        self.data_cfg = self.cfg["data"]
        self.model_cfg = self.cfg["model"]
        self.render_cfg = self.cfg["render"]
        self.train_cfg = self.cfg["train"]
        self.loss_cfg = self.cfg["losses"]
        self.logging_cfg = self.cfg["logging"]
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

        self.sequence_data = self.load_sequence_data()
        self.num_frames = self.sequence_data.frame_count
        if self.num_frames < self.model_cfg["train_frame_count"]:
            raise ValueError(
                f"Need at least train_frame_count={self.model_cfg['train_frame_count']} frames, "
                f"got {self.num_frames} from {self.sequence_data.source_path}"
            )

        print(
            f"Loaded {self.num_frames} frames from {self.sequence_data.source_path} "
            f"(source={self.sequence_data.frame_source}, source_total={self.sequence_data.all_frame_count})"
        )

        wandb.init(
            project=self.logging_cfg["wandb_project"],
            name=self.logging_cfg["wandb_run_name"],
            tags=self.logging_cfg.get("wandb_tags"),
            config=serialize_config_value(self.cfg),
        )

        self.model = build_model_from_config(self.cfg).to(self.device)
        self.model.train()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
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
        self.gt_video_logged = False

    def load_sequence_data(self) -> SequenceData:
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

    def autocast_context(self):
        if self.amp_available:
            return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)
        return nullcontext()

    def sample_clip(self) -> tuple[torch.Tensor, torch.Tensor]:
        clip_indices = select_window_indices(self.num_frames, self.model_cfg["train_frame_count"], device=self.device)
        return prepare_clip(self.sequence_data, clip_indices)

    def forward_clip(self, clip_frames: torch.Tensor, clip_times: torch.Tensor) -> GaussianSequence:
        with fast_attn_context(self.device), self.autocast_context():
            return self.model(clip_frames, decode_times=clip_times)

    def compute_camera_losses(
        self,
        clip_times: torch.Tensor,
        camera_state: CameraState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        camera_motion_loss = (
            torch.cat(
                [
                    camera_state.rotation_delta,
                    camera_state.translation_delta / camera_state.radius.clamp_min(1e-6),
                ],
                dim=-1,
            )
            .pow(2)
            .mean()
        )

        if clip_times.shape[1] > 1:
            motion_features = camera_state.motion_features()
            camera_temporal_loss = (motion_features[1:] - motion_features[:-1]).pow(2).mean()
        else:
            camera_temporal_loss = camera_motion_loss.new_tensor(0.0)

        camera_global_loss = camera_state.global_residuals.pow(2).mean()
        return camera_motion_loss, camera_temporal_loss, camera_global_loss

    def build_camera_loss(
        self,
        clip_times: torch.Tensor,
        camera_state: CameraState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        camera_motion_loss, camera_temporal_loss, camera_global_loss = self.compute_camera_losses(
            clip_times,
            camera_state,
        )
        camera_loss = (
            self.loss_cfg["camera_motion_weight"] * camera_motion_loss
            + self.loss_cfg["camera_temporal_weight"] * camera_temporal_loss
            + self.loss_cfg["camera_global_weight"] * camera_global_loss
        )
        return camera_loss, camera_motion_loss, camera_temporal_loss, camera_global_loss

    def build_bank_rate_loss(self, decoded: GaussianSequence) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        zero = decoded.xyz.new_tensor(0.0)
        terms = {
            "static_alpha": zero,
            "dynamic_alpha": zero,
            "dynamic_motion": zero,
            "dynamic_rotation": zero,
            "dynamic_alpha_time": zero,
        }
        auxiliary = decoded.auxiliary
        required_keys = {
            "static_opacities",
            "dynamic_opacities",
            "dynamic_A_mu",
            "dynamic_A_rot",
            "dynamic_A_alpha",
        }
        if not required_keys.issubset(auxiliary.keys()):
            return zero, terms

        terms = {
            "static_alpha": auxiliary["static_opacities"].mean(),
            "dynamic_alpha": auxiliary["dynamic_opacities"].mean(),
            "dynamic_motion": auxiliary["dynamic_A_mu"].abs().mean(),
            "dynamic_rotation": auxiliary["dynamic_A_rot"].abs().mean(),
            "dynamic_alpha_time": auxiliary["dynamic_A_alpha"].abs().mean(),
        }
        rate_loss = (
            float(self.loss_cfg["static_alpha_rate_weight"]) * terms["static_alpha"]
            + float(self.loss_cfg["dynamic_alpha_rate_weight"]) * terms["dynamic_alpha"]
            + float(self.loss_cfg["dynamic_motion_rate_weight"]) * terms["dynamic_motion"]
            + float(self.loss_cfg["dynamic_rotation_rate_weight"]) * terms["dynamic_rotation"]
            + float(self.loss_cfg["dynamic_alpha_time_rate_weight"]) * terms["dynamic_alpha_time"]
        )
        return rate_loss, terms

    def temporal_recon_chunk_size(self, frame_count: int) -> int:
        if self.recon_backward_strategy == "batched":
            return frame_count
        if self.recon_backward_strategy == "framewise":
            return 1
        return min(self.temporal_microbatch_size, frame_count)

    def recon_backward(
        self,
        clip_frames: torch.Tensor,
        decoded: GaussianSequence,
        regularizer_loss: torch.Tensor,
        keep_preview: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        recon_loss = clip_frames.new_tensor(0.0)
        preview_render = None
        if decoded.cameras is None:
            raise ValueError("Implicit-camera video decode must include cameras.")
        frame_count = len(decoded.cameras)
        chunk_size = self.temporal_recon_chunk_size(frame_count)
        target_frames = resize_images(clip_frames[0], self.render_size)

        for chunk_start in range(0, frame_count, chunk_size):
            chunk_end = min(chunk_start + chunk_size, frame_count)
            chunk_sequence = gaussian_sequence_slice(decoded, chunk_start, chunk_end)
            chunk_renders = render_clip_sequence(
                chunk_sequence,
                tuple(decoded.cameras[chunk_start:chunk_end]),
                renderer_mode=self.renderer_mode,
                render_cfg=self.render_cfg,
                input_size=self.model_cfg["size"],
                render_size=self.render_size,
                dense_grid=self.dense_grid,
            )
            if keep_preview and preview_render is None:
                preview_render = chunk_renders[0].detach()
            target = target_frames[chunk_start:chunk_end]
            chunk_losses = reconstruction_loss_per_image(chunk_renders, target, self.loss_cfg)
            chunk_recon_loss = chunk_losses.sum() / frame_count
            recon_loss = recon_loss + chunk_recon_loss.detach()
            is_last_chunk = chunk_end == frame_count
            backward_loss = chunk_recon_loss + (regularizer_loss if is_last_chunk else 0.0)
            backward_loss.backward(retain_graph=not is_last_chunk)

        return recon_loss, preview_render

    def step(self, keep_preview: bool = False) -> StepResult:
        self.optimizer.zero_grad(set_to_none=True)
        clip_frames, clip_times = self.sample_clip()
        decoded = self.forward_clip(clip_frames, clip_times)
        if decoded.camera_state is None:
            raise ValueError("Implicit-camera video decode must include camera_state.")

        camera_loss, camera_motion_loss, camera_temporal_loss, camera_global_loss = self.build_camera_loss(
            clip_times,
            decoded.camera_state,
        )
        bank_rate_loss, bank_rate_terms = self.build_bank_rate_loss(decoded)

        recon_loss, preview_render = self.recon_backward(
            clip_frames,
            decoded,
            camera_loss + bank_rate_loss,
            keep_preview,
        )

        self.optimizer.step()
        loss = recon_loss + camera_loss.detach() + bank_rate_loss.detach()
        return StepResult(
            clip_frames=clip_frames,
            preview_render=preview_render,
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
        metrics = self.camera_metrics(result.camera_state)
        payload = {
            "Loss": result.loss.item(),
            "Loss/Reconstruction": result.recon_loss.item(),
            "Loss/CameraMotion": result.camera_motion_loss.item(),
            "Loss/CameraTemporal": result.camera_temporal_loss.item(),
            "Loss/CameraGlobal": result.camera_global_loss.item(),
            "Loss/BankRate": result.bank_rate_loss.item(),
            "TrainFrameCount": int(self.model_cfg["train_frame_count"]),
            "SequenceFrames": self.num_frames,
            "InputSize": int(self.model_cfg["size"]),
            "RenderSize": int(self.render_size),
            "Camera/FOVDegrees": metrics["fov_degrees"],
            "Camera/Radius": metrics["radius"],
            "Camera/RotationDeltaMeanDegrees": metrics["rotation_delta_mean_degrees"],
            "Camera/TranslationDeltaMean": metrics["translation_delta_mean"],
        }
        for key, value in result.bank_rate_terms.items():
            payload[f"BankRate/{key}"] = value.item()
        return payload

    def render_preview_image(self, result: StepResult, step: int) -> wandb.Image:
        if result.preview_render is None:
            raise ValueError("Preview render was requested for logging but was not retained during the training step.")
        target = resize_images(result.clip_frames[0, 0], self.render_size)
        return make_preview_image(target, result.preview_render, caption=f"Step {step}")

    def validation_video_payload(self) -> dict[str, Any]:
        rendered_sequence, eval_camera_state = render_full_sequence(
            self.model,
            self.sequence_data,
            self.cfg,
            self.renderer_mode,
            self.dense_grid,
            self.amp_available,
            self.amp_dtype,
            self.device,
        )
        gt_sequence = resize_images(self.sequence_data.frames, self.render_size).detach().cpu()
        payload = {
            **build_validation_video_payload(
                rendered_sequence,
                gt_sequence,
                self.sequence_data.video_fps,
            ),
            **eval_metric_payload(rendered_sequence, gt_sequence, self.loss_cfg),
            **temporal_similarity_payload(rendered_sequence, gt_sequence, self.loss_cfg),
            "Camera/EvalFOVDegrees": eval_camera_state.fov_degrees.item(),
            "Camera/EvalRadius": eval_camera_state.radius.item(),
            "Camera/EvalRotationDeltaMeanDegrees": (
                torch.rad2deg(torch.linalg.norm(eval_camera_state.rotation_delta, dim=-1)).mean().item()
            ),
            "Camera/EvalTranslationDeltaMean": (
                torch.linalg.norm(eval_camera_state.translation_delta, dim=-1).mean().item()
            ),
        }
        if not self.gt_video_logged:
            payload["GT_Video"] = make_wandb_video(gt_sequence, self.sequence_data.video_fps)
            self.gt_video_logged = True
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
            f"{self.num_frames} frames, train_frame_count={self.model_cfg['train_frame_count']}, "
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
            f"Temporal reconstruction chunk size: {self.temporal_recon_chunk_size(self.model_cfg['train_frame_count'])}"
        )
        print(f"Attention backend: {self.attn_backend}")

        pbar = tqdm(range(1, self.train_cfg["steps"] + 1))
        try:
            for step in pbar:
                keep_preview = self.should_log_images(step)
                result = self.step(keep_preview=keep_preview)
                pbar.set_description(self.progress_message(result))
                self.val_log(step, result)
        finally:
            wandb.finish()

        print("DynamicVideoTokenGSImplicitCamera training complete. Check your Weights & Biases dashboard.")


def run_training(config: dict[str, Any]) -> None:
    Trainer(config).run()


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
