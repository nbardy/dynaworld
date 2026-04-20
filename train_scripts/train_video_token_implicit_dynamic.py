from __future__ import annotations

import os
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import wandb
from dynamicTokenGS import (
    DEFAULT_SEQUENCE_DIR,
    configure_fast_attn,
    fast_attn_context,
    make_wandb_video,
    pick_device,
    select_window_indices,
)
from gs_models import DynamicVideoTokenGSImplicitCamera
from renderers.common import build_pixel_grid
from renderers.dense import render_pytorch_3dgs
from renderers.tiled import render_pytorch_3dgs_tiled
from tqdm import tqdm
from train_camera_implicit_dynamic import load_sequence_data, resolve_frames_dir

DEFAULT_CONFIG = {
    "data": {
        "sequence_dir": DEFAULT_SEQUENCE_DIR,
        "frames_dir": None,
        "video_path": None,
        "frame_source": "explicit_video",
        "max_frames": 0,
    },
    "model": {
        "size": 384,
        "train_frame_count": 16,
        "tokens": 8,
        "gaussians_per_token": 64,
        "model_dim": 128,
        "bottleneck_dim": 256,
        "num_heads": 8,
        "mlp_ratio": 4.0,
        "scene_extent": 1.0,
        "tubelet_size_t": 4,
        "patch_compression": 16,
        "encoder_self_attn_layers": 1,
        "bottleneck_self_attn_layers": 4,
        "cross_attn_layers": 1,
    },
    "camera": {
        "base_fov_degrees": 60.0,
        "base_radius": 3.0,
        "max_fov_delta_degrees": 15.0,
        "max_radius_scale": 1.5,
        "max_rotation_degrees": 5.0,
        "max_translation_ratio": 0.2,
    },
    "render": {
        "renderer": "dense",
        "auto_dense_limit": 400_000,
        "tile_size": 8,
        "bound_scale": 3.0,
        "alpha_threshold": 1.0 / 255.0,
    },
    "train": {
        "steps": 100,
        "lr": 0.005,
        "amp": False,
        "recon_backward_strategy": "framewise",
    },
    "losses": {
        "camera_motion_weight": 0.01,
        "camera_temporal_weight": 0.02,
        "camera_global_weight": 0.005,
    },
    "logging": {
        "log_every": 10,
        "image_log_every": 50,
        "video_log_every": 50,
        "always_log_last_step": True,
        "wandb_project": "dynamic-tokengs-overfit",
        "wandb_run_name": "dynamic-video-token-implicit-camera-run",
    },
}


@dataclass
class StepResult:
    clip_frames: torch.Tensor
    preview_render: torch.Tensor | None
    camera_state: dict[str, torch.Tensor]
    loss: torch.Tensor
    recon_loss: torch.Tensor
    camera_motion_loss: torch.Tensor
    camera_temporal_loss: torch.Tensor
    camera_global_loss: torch.Tensor


def build_default_config() -> dict[str, Any]:
    return deepcopy(DEFAULT_CONFIG)


def merge_config(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = build_default_config() if config is None else merge_config(build_default_config(), config)
    cfg["data"]["sequence_dir"] = Path(cfg["data"]["sequence_dir"])
    frames_dir = cfg["data"]["frames_dir"]
    cfg["data"]["frames_dir"] = None if frames_dir is None else Path(frames_dir)
    video_path = cfg["data"]["video_path"]
    cfg["data"]["video_path"] = None if video_path is None else Path(video_path)
    return cfg


def serialize_config_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: serialize_config_value(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_config_value(inner) for inner in value]
    return value


def config_from_env(env: dict[str, str] | None = None) -> dict[str, Any]:
    env = os.environ if env is None else env
    overrides: dict[str, Any] = {}

    def set_section_value(section: str, key: str, value: Any) -> None:
        overrides.setdefault(section, {})[key] = value

    def parse_bool(value: str) -> bool:
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"Could not parse boolean value from {value!r}")

    string_fields = {
        "data": {
            "sequence_dir": "SEQUENCE_DIR",
            "video_path": "VIDEO_PATH",
            "frame_source": "FRAME_SOURCE",
        },
        "render": {
            "renderer": "RENDERER",
        },
        "train": {
            "recon_backward_strategy": "TRAIN_BACKWARD_STRATEGY",
        },
        "logging": {
            "wandb_project": "WANDB_PROJECT",
            "wandb_run_name": "RUN_NAME",
        },
    }
    int_fields = {
        "data": {
            "max_frames": "MAX_FRAMES",
        },
        "model": {
            "size": "SIZE",
            "train_frame_count": "TRAIN_FRAME_COUNT",
            "tokens": "TOKENS",
            "gaussians_per_token": "GAUSSIANS_PER_TOKEN",
            "model_dim": "MODEL_DIM",
            "bottleneck_dim": "BOTTLENECK_DIM",
            "num_heads": "NUM_HEADS",
            "tubelet_size_t": "TUBELET_SIZE_T",
            "patch_compression": "PATCH_COMPRESSION",
            "encoder_self_attn_layers": "ENCODER_SELF_ATTN_LAYERS",
            "bottleneck_self_attn_layers": "BOTTLENECK_SELF_ATTN_LAYERS",
            "cross_attn_layers": "CROSS_ATTN_LAYERS",
        },
        "train": {
            "steps": "STEPS",
        },
        "logging": {
            "log_every": "LOG_EVERY",
            "image_log_every": "IMAGE_LOG_EVERY",
            "video_log_every": "VIDEO_LOG_EVERY",
        },
    }
    float_fields = {
        "model": {
            "mlp_ratio": "MLP_RATIO",
            "scene_extent": "SCENE_EXTENT",
        },
        "camera": {
            "base_fov_degrees": "BASE_FOV_DEGREES",
            "base_radius": "BASE_RADIUS",
            "max_fov_delta_degrees": "MAX_FOV_DELTA_DEGREES",
            "max_radius_scale": "MAX_RADIUS_SCALE",
            "max_rotation_degrees": "MAX_ROTATION_DEGREES",
            "max_translation_ratio": "MAX_TRANSLATION_RATIO",
        },
        "train": {
            "lr": "LR",
        },
        "losses": {
            "camera_motion_weight": "CAMERA_MOTION_WEIGHT",
            "camera_temporal_weight": "CAMERA_TEMPORAL_WEIGHT",
            "camera_global_weight": "CAMERA_GLOBAL_WEIGHT",
        },
        "render": {
            "bound_scale": "BOUND_SCALE",
            "alpha_threshold": "ALPHA_THRESHOLD",
        },
    }
    bool_fields = {
        "train": {
            "amp": "AMP",
        },
        "logging": {
            "always_log_last_step": "ALWAYS_LOG_LAST_STEP",
        },
    }

    for section, fields in string_fields.items():
        for key, env_name in fields.items():
            if env_name in env:
                set_section_value(section, key, env[env_name])

    for section, fields in int_fields.items():
        for key, env_name in fields.items():
            if env_name in env:
                set_section_value(section, key, int(env[env_name]))

    if "TRAIN_FRAME_COUNT" not in env and "FRAME_INPUT_COUNT" in env:
        set_section_value("model", "train_frame_count", int(env["FRAME_INPUT_COUNT"]))

    for section, fields in float_fields.items():
        for key, env_name in fields.items():
            if env_name in env:
                set_section_value(section, key, float(env[env_name]))

    for section, fields in bool_fields.items():
        for key, env_name in fields.items():
            if env_name in env:
                set_section_value(section, key, parse_bool(env[env_name]))

    return merge_config(build_default_config(), overrides)


def pick_renderer_mode_from_config(config: dict[str, Any]) -> tuple[str, int]:
    model_cfg = config["model"]
    render_cfg = config["render"]
    effective_gaussians = model_cfg["tokens"] * model_cfg["gaussians_per_token"]
    if render_cfg["renderer"] == "auto":
        renderer_mode = (
            "dense"
            if effective_gaussians * model_cfg["size"] * model_cfg["size"] <= render_cfg["auto_dense_limit"]
            else "tiled"
        )
    else:
        renderer_mode = render_cfg["renderer"]
    return renderer_mode, effective_gaussians


def normalize_clip_times(frame_times: torch.Tensor) -> torch.Tensor:
    frame_times = frame_times.to(dtype=torch.float32)
    minimum = frame_times.min()
    maximum = frame_times.max()
    if float(maximum - minimum) > 1e-6:
        return (frame_times - minimum) / (maximum - minimum)
    return torch.zeros_like(frame_times)


def prepare_clip(sequence_data: dict[str, Any], clip_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    clip_frames = sequence_data["frames"][clip_indices]
    clip_times = normalize_clip_times(sequence_data["frame_times"][clip_indices].reshape(-1)).unsqueeze(0)
    return clip_frames.unsqueeze(0), clip_times


def render_clip_frame(
    renderer_mode: str,
    render_cfg: dict[str, Any],
    image_size: int,
    dense_grid: torch.Tensor,
    camera: Any,
    xyz: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    rgbs: torch.Tensor,
) -> torch.Tensor:
    if renderer_mode == "dense":
        return render_pytorch_3dgs(
            xyz.float(),
            scales.float(),
            quats.float(),
            opacities.float(),
            rgbs.float(),
            image_size,
            image_size,
            camera.fx,
            camera.fy,
            camera.cx,
            camera.cy,
            grid=dense_grid,
            camera_to_world=camera.camera_to_world.float(),
        )
    return render_pytorch_3dgs_tiled(
        xyz.float(),
        scales.float(),
        quats.float(),
        opacities.float(),
        rgbs.float(),
        image_size,
        image_size,
        camera.fx,
        camera.fy,
        camera.cx,
        camera.cy,
        tile_size=render_cfg["tile_size"],
        bound_scale=render_cfg["bound_scale"],
        alpha_threshold=render_cfg["alpha_threshold"],
        camera_to_world=camera.camera_to_world.float(),
    )


@torch.no_grad()
def render_full_sequence(
    model: torch.nn.Module,
    sequence_data: dict[str, Any],
    config: dict[str, Any],
    renderer_mode: str,
    dense_grid: torch.Tensor,
    amp_available: bool,
    amp_dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    was_training = model.training
    model.eval()
    model_cfg = config["model"]
    clip_length = model_cfg["train_frame_count"]
    num_frames = sequence_data["frames"].shape[0]
    rendered_frames = [None] * num_frames
    camera_states = []

    for end in range(clip_length, num_frames + clip_length, clip_length):
        clip_end = min(end, num_frames)
        clip_start = max(0, clip_end - clip_length)
        clip_indices = torch.arange(clip_start, clip_end, device=device)
        clip_frames, clip_times = prepare_clip(sequence_data, clip_indices)

        autocast_context = torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_available else nullcontext()
        with fast_attn_context(device), autocast_context:
            xyz, scales, quats, opacities, rgbs, cameras, camera_state = model(clip_frames, decode_times=clip_times)
        camera_states.append(camera_state)

        for local_index, frame_index in enumerate(clip_indices.tolist()):
            if rendered_frames[frame_index] is not None:
                continue
            rendered_frames[frame_index] = render_clip_frame(
                renderer_mode,
                config["render"],
                model_cfg["size"],
                dense_grid,
                cameras[local_index],
                xyz[local_index],
                scales[local_index],
                quats[local_index],
                opacities[local_index],
                rgbs[local_index],
            ).cpu()

    if was_training:
        model.train()
    merged_camera_state = {
        "fov_degrees": torch.stack([state["fov_degrees"] for state in camera_states]).mean(),
        "radius": torch.stack([state["radius"] for state in camera_states]).mean(),
        "rotation_delta": torch.cat([state["rotation_delta"] for state in camera_states], dim=0),
        "translation_delta": torch.cat([state["translation_delta"] for state in camera_states], dim=0),
    }
    return torch.stack(rendered_frames, dim=0), merged_camera_state


def build_model_from_config(config: dict[str, Any]) -> DynamicVideoTokenGSImplicitCamera:
    model_cfg = config["model"]
    camera_cfg = config["camera"]
    return DynamicVideoTokenGSImplicitCamera(
        clip_length=model_cfg["train_frame_count"],
        image_size=model_cfg["size"],
        num_tokens=model_cfg["tokens"],
        feat_dim=model_cfg["model_dim"],
        bottleneck_dim=model_cfg["bottleneck_dim"],
        num_heads=model_cfg["num_heads"],
        mlp_ratio=model_cfg["mlp_ratio"],
        gaussians_per_token=model_cfg["gaussians_per_token"],
        scene_extent=model_cfg["scene_extent"],
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
        max_rotation_degrees=camera_cfg["max_rotation_degrees"],
        max_translation_ratio=camera_cfg["max_translation_ratio"],
    )


class Trainer:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.cfg = resolve_config(config)
        self.data_cfg = self.cfg["data"]
        self.model_cfg = self.cfg["model"]
        self.render_cfg = self.cfg["render"]
        self.train_cfg = self.cfg["train"]
        self.loss_cfg = self.cfg["losses"]
        self.logging_cfg = self.cfg["logging"]
        self.recon_backward_strategy = self.train_cfg["recon_backward_strategy"]
        if self.recon_backward_strategy not in {"framewise", "batched"}:
            raise ValueError(
                f"Unsupported recon_backward_strategy={self.recon_backward_strategy!r}. "
                "Expected one of: framewise, batched."
            )

        self.device = pick_device()
        print(f"Using device: {self.device}")

        self.sequence_data = self.load_sequence_data()
        self.num_frames = self.sequence_data["frames"].shape[0]
        if self.num_frames < self.model_cfg["train_frame_count"]:
            raise ValueError(
                f"Need at least train_frame_count={self.model_cfg['train_frame_count']} frames, "
                f"got {self.num_frames} from {self.sequence_data['source_path']}"
            )

        print(
            f"Loaded {self.num_frames} frames from {self.sequence_data['source_path']} "
            f"(source={self.sequence_data['frame_source']}, source_total={self.sequence_data['all_frame_count']})"
        )

        wandb.init(
            project=self.logging_cfg["wandb_project"],
            name=self.logging_cfg["wandb_run_name"],
            config=serialize_config_value(self.cfg),
        )

        self.model = build_model_from_config(self.cfg).to(self.device)
        self.model.train()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.train_cfg["lr"],
            fused=self.device.type in {"cuda", "mps"},
        )

        self.dense_grid = build_pixel_grid(self.model_cfg["size"], self.model_cfg["size"], self.device)
        self.amp_available = bool(
            self.train_cfg["amp"] and torch.amp.autocast_mode.is_autocast_available(self.device.type)
        )
        if self.train_cfg["amp"] and not self.amp_available:
            print(f"AMP requested but not available on device {self.device.type}; continuing in fp32.")
        self.amp_dtype = (
            torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
        )
        self.attn_dtype = self.amp_dtype if self.amp_available else self.sequence_data["frames"].dtype
        self.attn_backend = configure_fast_attn(self.device, self.attn_dtype)
        self.renderer_mode, self.effective_gaussians = pick_renderer_mode_from_config(self.cfg)
        self.gt_video_logged = False

    def load_sequence_data(self) -> dict[str, Any]:
        if self.data_cfg["frame_source"] == "explicit_video" and self.data_cfg["video_path"] is None:
            raise ValueError("config['data']['video_path'] is required when frame_source='explicit_video'.")
        frames_dir = resolve_frames_dir(self.data_cfg["sequence_dir"], self.data_cfg["frames_dir"])
        return load_sequence_data(
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

    def forward_clip(self, clip_frames: torch.Tensor, clip_times: torch.Tensor):
        with fast_attn_context(self.device), self.autocast_context():
            return self.model(clip_frames, decode_times=clip_times)

    def compute_camera_losses(
        self,
        clip_times: torch.Tensor,
        camera_state: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        camera_motion_loss = (
            torch.cat(
                [
                    camera_state["rotation_delta"],
                    camera_state["translation_delta"] / camera_state["radius"].clamp_min(1e-6),
                ],
                dim=-1,
            )
            .pow(2)
            .mean()
        )

        if clip_times.shape[1] > 1:
            motion_features = torch.cat([camera_state["rotation_delta"], camera_state["translation_delta"]], dim=-1)
            camera_temporal_loss = (motion_features[1:] - motion_features[:-1]).pow(2).mean()
        else:
            camera_temporal_loss = camera_motion_loss.new_tensor(0.0)

        camera_global_loss = camera_state["global_residuals"].pow(2).mean()
        return camera_motion_loss, camera_temporal_loss, camera_global_loss

    def build_camera_loss(
        self,
        clip_times: torch.Tensor,
        camera_state: dict[str, torch.Tensor],
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

    def batched_recon_backward(
        self,
        clip_frames: torch.Tensor,
        xyz: torch.Tensor,
        scales: torch.Tensor,
        quats: torch.Tensor,
        opacities: torch.Tensor,
        rgbs: torch.Tensor,
        cameras: list[Any],
        camera_loss: torch.Tensor,
        keep_preview: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        recon_losses = []
        preview_render = None
        for local_index, camera in enumerate(cameras):
            render = render_clip_frame(
                self.renderer_mode,
                self.render_cfg,
                self.model_cfg["size"],
                self.dense_grid,
                camera,
                xyz[local_index],
                scales[local_index],
                quats[local_index],
                opacities[local_index],
                rgbs[local_index],
            )
            if keep_preview and local_index == 0:
                preview_render = render.detach()
            target = clip_frames[0, local_index]
            recon_losses.append(F.l1_loss(render, target) + 0.2 * F.mse_loss(render, target))

        recon_loss = torch.stack(recon_losses).mean()
        (recon_loss + camera_loss).backward()
        return recon_loss.detach(), preview_render

    def framewise_recon_backward(
        self,
        clip_frames: torch.Tensor,
        xyz: torch.Tensor,
        scales: torch.Tensor,
        quats: torch.Tensor,
        opacities: torch.Tensor,
        rgbs: torch.Tensor,
        cameras: list[Any],
        camera_loss: torch.Tensor,
        keep_preview: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        recon_loss = clip_frames.new_tensor(0.0)
        preview_render = None
        frame_count = len(cameras)
        for local_index, camera in enumerate(cameras):
            render = render_clip_frame(
                self.renderer_mode,
                self.render_cfg,
                self.model_cfg["size"],
                self.dense_grid,
                camera,
                xyz[local_index],
                scales[local_index],
                quats[local_index],
                opacities[local_index],
                rgbs[local_index],
            )
            if keep_preview and local_index == 0:
                preview_render = render.detach()
            target = clip_frames[0, local_index]
            frame_recon_loss = F.l1_loss(render, target) + 0.2 * F.mse_loss(render, target)
            recon_loss = recon_loss + frame_recon_loss.detach() / frame_count
            frame_recon_loss.div(frame_count).backward(retain_graph=True)

        camera_loss.backward()
        return recon_loss, preview_render

    def step(self, keep_preview: bool = False) -> StepResult:
        self.optimizer.zero_grad(set_to_none=True)
        clip_frames, clip_times = self.sample_clip()
        xyz, scales, quats, opacities, rgbs, cameras, camera_state = self.forward_clip(clip_frames, clip_times)

        camera_loss, camera_motion_loss, camera_temporal_loss, camera_global_loss = self.build_camera_loss(
            clip_times,
            camera_state,
        )

        if self.recon_backward_strategy == "batched":
            recon_loss, preview_render = self.batched_recon_backward(
                clip_frames,
                xyz,
                scales,
                quats,
                opacities,
                rgbs,
                cameras,
                camera_loss,
                keep_preview,
            )
        else:
            recon_loss, preview_render = self.framewise_recon_backward(
                clip_frames,
                xyz,
                scales,
                quats,
                opacities,
                rgbs,
                cameras,
                camera_loss,
                keep_preview,
            )

        self.optimizer.step()
        loss = recon_loss + camera_loss.detach()
        return StepResult(
            clip_frames=clip_frames,
            preview_render=preview_render,
            camera_state=camera_state,
            loss=loss,
            recon_loss=recon_loss,
            camera_motion_loss=camera_motion_loss,
            camera_temporal_loss=camera_temporal_loss,
            camera_global_loss=camera_global_loss,
        )

    def camera_metrics(self, camera_state: dict[str, torch.Tensor]) -> dict[str, float]:
        return {
            "fov_degrees": camera_state["fov_degrees"].item(),
            "radius": camera_state["radius"].item(),
            "rotation_delta_mean_degrees": (
                torch.rad2deg(torch.linalg.norm(camera_state["rotation_delta"], dim=-1)).mean().item()
            ),
            "translation_delta_mean": torch.linalg.norm(camera_state["translation_delta"], dim=-1).mean().item(),
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
        return {
            "Loss": result.loss.item(),
            "Loss/Reconstruction": result.recon_loss.item(),
            "Loss/CameraMotion": result.camera_motion_loss.item(),
            "Loss/CameraTemporal": result.camera_temporal_loss.item(),
            "Loss/CameraGlobal": result.camera_global_loss.item(),
            "TrainFrameCount": int(self.model_cfg["train_frame_count"]),
            "SequenceFrames": self.num_frames,
            "Camera/FOVDegrees": metrics["fov_degrees"],
            "Camera/Radius": metrics["radius"],
            "Camera/RotationDeltaMeanDegrees": metrics["rotation_delta_mean_degrees"],
            "Camera/TranslationDeltaMean": metrics["translation_delta_mean"],
        }

    def render_preview_image(self, result: StepResult, step: int) -> wandb.Image:
        if result.preview_render is None:
            raise ValueError("Preview render was requested for logging but was not retained during the training step.")
        preview = torch.cat([result.clip_frames[0, 0], result.preview_render], dim=2)
        return wandb.Image(T.ToPILImage()(preview.cpu().clamp(0, 1)), caption=f"Step {step}")

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
        gt_sequence = self.sequence_data["frames"].detach().cpu()
        side_by_side = torch.cat([gt_sequence, rendered_sequence], dim=3)
        payload = {
            "Render_Video": make_wandb_video(rendered_sequence, self.sequence_data["video_fps"]),
            "Render_GT_Video": make_wandb_video(side_by_side, self.sequence_data["video_fps"]),
            "Camera/EvalFOVDegrees": eval_camera_state["fov_degrees"].item(),
            "Camera/EvalRadius": eval_camera_state["radius"].item(),
            "Camera/EvalRotationDeltaMeanDegrees": (
                torch.rad2deg(torch.linalg.norm(eval_camera_state["rotation_delta"], dim=-1)).mean().item()
            ),
            "Camera/EvalTranslationDeltaMean": (
                torch.linalg.norm(eval_camera_state["translation_delta"], dim=-1).mean().item()
            ),
        }
        if not self.gt_video_logged:
            payload["GT_Video"] = make_wandb_video(gt_sequence, self.sequence_data["video_fps"])
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
        print(
            "Starting DynamicVideoTokenGSImplicitCamera Training: "
            f"{self.num_frames} frames, train_frame_count={self.model_cfg['train_frame_count']}, "
            f"1 global camera token + 1 path token + {self.model_cfg['tokens']} 3DGS tokens x "
            f"{self.model_cfg['gaussians_per_token']} gaussians/token = "
            f"{self.effective_gaussians} explicit Gaussians with {self.renderer_mode} renderer..."
        )
        print(f"Reconstruction backward strategy: {self.recon_backward_strategy}")
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


def run_training(config: dict[str, Any] | None = None) -> None:
    Trainer(config).run()


def main(config: dict[str, Any] | None = None) -> None:
    run_training(config)


if __name__ == "__main__":
    raise SystemExit("Import this module and call main(config) with a config dict.")
