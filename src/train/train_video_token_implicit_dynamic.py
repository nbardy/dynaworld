from __future__ import annotations

import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import wandb
from config_utils import load_config_file, path_or_none, resolved_config, serialize_config_value
from dynamicTokenGS import (
    configure_fast_attn,
    fast_attn_context,
    pick_device,
)
from gs_models import DynamicVideoTokenGSImplicitCamera
from rendering import build_or_reuse_grid, camera_for_viewport, render_gaussian_frame, resize_images
from rendering import pick_renderer_mode as resolve_renderer_mode
from runtime_types import CameraState, GaussianFrame, GaussianSequence, SequenceData
from sequence_data import load_uncalibrated_sequence, resolve_frames_dir, select_window_indices
from tqdm import tqdm
from train_logging import build_validation_video_payload, make_preview_image, make_wandb_video


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


def resolve_config(config: dict[str, Any]) -> dict[str, Any]:
    if config is None:
        raise ValueError("A train config is required. Pass a JSONC path or config dict.")
    cfg = resolved_config(config, ("data", "model", "camera", "render", "train", "losses", "logging"))
    cfg["data"]["sequence_dir"] = Path(cfg["data"]["sequence_dir"])
    cfg["data"]["frames_dir"] = path_or_none(cfg["data"]["frames_dir"])
    cfg["data"]["video_path"] = path_or_none(cfg["data"]["video_path"])
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


def normalize_clip_times(frame_times: torch.Tensor) -> torch.Tensor:
    frame_times = frame_times.to(dtype=torch.float32)
    minimum = frame_times.min()
    maximum = frame_times.max()
    if float(maximum - minimum) > 1e-6:
        return (frame_times - minimum) / (maximum - minimum)
    return torch.zeros_like(frame_times)


def prepare_clip(sequence_data: SequenceData, clip_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    clip_frames = sequence_data.frames[clip_indices]
    clip_times = normalize_clip_times(sequence_data.frame_times[clip_indices].reshape(-1)).unsqueeze(0)
    return clip_frames.unsqueeze(0), clip_times


def render_clip_frame(
    renderer_mode: str,
    render_cfg: dict[str, Any],
    input_size: int,
    render_size: int,
    dense_grid: torch.Tensor,
    camera: Any,
    xyz: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    rgbs: torch.Tensor,
) -> torch.Tensor:
    render_camera = camera_for_viewport(
        camera,
        source_height=input_size,
        source_width=input_size,
        target_height=render_size,
        target_width=render_size,
    )
    return render_gaussian_frame(
        GaussianFrame(xyz=xyz, scales=scales, quats=quats, opacities=opacities, rgbs=rgbs),
        camera=render_camera,
        height=render_size,
        width=render_size,
        mode=renderer_mode,
        dense_grid=dense_grid,
        tile_size=render_cfg["tile_size"],
        bound_scale=render_cfg["bound_scale"],
        alpha_threshold=render_cfg["alpha_threshold"],
    )


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
        camera_states.append(decoded.camera_state)

        for local_index, frame_index in enumerate(clip_indices.tolist()):
            if rendered_frames[frame_index] is not None:
                continue
            camera = decoded.cameras[local_index]
            rendered_frames[frame_index] = render_clip_frame(
                renderer_mode,
                render_cfg,
                model_cfg["size"],
                render_cfg["render_size"],
                dense_grid,
                camera,
                decoded.xyz[local_index],
                decoded.scales[local_index],
                decoded.quats[local_index],
                decoded.opacities[local_index],
                decoded.rgbs[local_index],
            ).cpu()

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
        camera_loss: torch.Tensor,
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
            chunk_losses = []

            for local_index in range(chunk_start, chunk_end):
                camera = decoded.cameras[local_index]
                render = render_clip_frame(
                    self.renderer_mode,
                    self.render_cfg,
                    self.model_cfg["size"],
                    self.render_size,
                    self.dense_grid,
                    camera,
                    decoded.xyz[local_index],
                    decoded.scales[local_index],
                    decoded.quats[local_index],
                    decoded.opacities[local_index],
                    decoded.rgbs[local_index],
                )
                if keep_preview and preview_render is None:
                    preview_render = render.detach()
                target = target_frames[local_index]
                chunk_losses.append(F.l1_loss(render, target) + 0.2 * F.mse_loss(render, target))

            chunk_recon_loss = torch.stack(chunk_losses).sum() / frame_count
            recon_loss = recon_loss + chunk_recon_loss.detach()
            is_last_chunk = chunk_end == frame_count
            backward_loss = chunk_recon_loss + (camera_loss if is_last_chunk else 0.0)
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

        recon_loss, preview_render = self.recon_backward(
            clip_frames,
            decoded,
            camera_loss,
            keep_preview,
        )

        self.optimizer.step()
        loss = recon_loss + camera_loss.detach()
        return StepResult(
            clip_frames=clip_frames,
            preview_render=preview_render,
            camera_state=decoded.camera_state,
            loss=loss,
            recon_loss=recon_loss,
            camera_motion_loss=camera_motion_loss,
            camera_temporal_loss=camera_temporal_loss,
            camera_global_loss=camera_global_loss,
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
        return {
            "Loss": result.loss.item(),
            "Loss/Reconstruction": result.recon_loss.item(),
            "Loss/CameraMotion": result.camera_motion_loss.item(),
            "Loss/CameraTemporal": result.camera_temporal_loss.item(),
            "Loss/CameraGlobal": result.camera_global_loss.item(),
            "TrainFrameCount": int(self.model_cfg["train_frame_count"]),
            "SequenceFrames": self.num_frames,
            "InputSize": int(self.model_cfg["size"]),
            "RenderSize": int(self.render_size),
            "Camera/FOVDegrees": metrics["fov_degrees"],
            "Camera/Radius": metrics["radius"],
            "Camera/RotationDeltaMeanDegrees": metrics["rotation_delta_mean_degrees"],
            "Camera/TranslationDeltaMean": metrics["translation_delta_mean"],
        }

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
        print(
            "Starting DynamicVideoTokenGSImplicitCamera Training: "
            f"{self.num_frames} frames, train_frame_count={self.model_cfg['train_frame_count']}, "
            f"input_size={self.model_cfg['size']}, render_size={self.render_size}, "
            f"1 global camera token + 1 path token + {self.model_cfg['tokens']} 3DGS tokens x "
            f"{self.model_cfg['gaussians_per_token']} gaussians/token = "
            f"{self.effective_gaussians} explicit Gaussians with {self.renderer_mode} renderer..."
        )
        print(f"Reconstruction backward strategy: {self.recon_backward_strategy}")
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
