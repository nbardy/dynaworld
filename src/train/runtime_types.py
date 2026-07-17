from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch

try:
    from camera import CameraSpec, make_camera_like
except ImportError:  # pragma: no cover - supports package-style imports in tests.
    from .camera import CameraSpec, make_camera_like

Tensor = torch.Tensor

FrameSource = Literal[
    "camera_json",
    "summary_video",
    "explicit_video",
    "explicit_video_window",
    "summary_sampled",
    "all_frames",
]
RendererMode = Literal["auto", "dense", "tiled", "taichi", "fast_mac"]
ResolvedRendererMode = Literal["dense", "tiled", "taichi", "fast_mac"]
ReconstructionBackwardStrategy = Literal["batched", "microbatch", "framewise"]


def _move_camera(camera: CameraSpec, device: torch.device | str) -> CameraSpec:
    def move_value(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.to(device=device)
        return value

    return make_camera_like(
        camera,
        fx=move_value(camera.fx),
        fy=move_value(camera.fy),
        cx=move_value(camera.cx),
        cy=move_value(camera.cy),
        camera_to_world=camera.camera_to_world.to(device=device),
        distortion=move_value(camera.distortion),
    )


@dataclass(frozen=True)
class SequenceData:
    """A full training sequence after loading and resizing.

    frames: [T, 3, H, W], float, 0..1
    frame_times: [T, 1], normalized to 0..1 unless source timestamps are unavailable
    cameras: length T when cameras are known, None for implicit-camera baselines
    """

    frames: Tensor
    frame_times: Tensor
    video_fps: float
    frame_source: FrameSource
    image_crop_mode: str = "resize"
    frame_paths: tuple[Path, ...] = ()
    cameras: tuple[CameraSpec, ...] | None = None
    records: tuple[Mapping[str, Any], ...] = ()
    intrinsics_summary: Mapping[str, Any] = field(default_factory=dict)
    source_path: Path | None = None
    selected_frame_count: int | None = None
    all_frame_count: int | None = None

    @property
    def frame_count(self) -> int:
        return int(self.frames.shape[0])

    @property
    def image_size(self) -> int:
        return int(self.frames.shape[-1])

    def to(self, device: torch.device | str) -> "SequenceData":
        cameras = None
        if self.cameras is not None:
            cameras = tuple(_move_camera(camera, device) for camera in self.cameras)
        return SequenceData(
            frames=self.frames.to(device=device),
            frame_times=self.frame_times.to(device=device),
            video_fps=self.video_fps,
            frame_source=self.frame_source,
            image_crop_mode=self.image_crop_mode,
            frame_paths=self.frame_paths,
            cameras=cameras,
            records=self.records,
            intrinsics_summary=self.intrinsics_summary,
            source_path=self.source_path,
            selected_frame_count=self.selected_frame_count,
            all_frame_count=self.all_frame_count,
        )


@dataclass(frozen=True)
class ClipBatch:
    """A sampled training window.

    frames: [K, 3, H, W]
    frame_times: [K, 1]
    frame_indices: [K]
    cameras: length K only for known-camera training
    """

    frames: Tensor
    frame_times: Tensor
    frame_indices: Tensor
    video_fps: float
    cameras: tuple[CameraSpec, ...] | None = None

    @property
    def frame_count(self) -> int:
        return int(self.frames.shape[0])

    def as_video_batch(self) -> Tensor:
        """Return [1, K, 3, H, W] for video-token models."""
        return self.frames.unsqueeze(0)

    def as_time_batch(self, device: torch.device | str | None = None) -> Tensor:
        """Return [1, K] normalized times for video-token models."""
        times = self.frame_times.to(dtype=torch.float32)
        if device is not None:
            times = times.to(device=device)
        return times.reshape(1, -1)


@dataclass(frozen=True)
class CameraState:
    """Predicted camera diagnostics and regularization payload.

    fov_degrees: scalar tensor
    radius: scalar tensor
    global_residuals: raw output vector from the configured global camera head
    rotation_delta: [T, 3]
    translation_delta: [T, 3]
    path_residuals: [T, 6] when available
    """

    fov_degrees: Tensor
    radius: Tensor
    global_residuals: Tensor
    rotation_delta: Tensor
    translation_delta: Tensor
    path_residuals: Tensor | None = None

    def motion_features(self) -> Tensor:
        """Return [T, 6] for motion/temporal camera regularizers."""
        return torch.cat([self.rotation_delta, self.translation_delta], dim=-1)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Tensor]) -> "CameraState":
        return cls(
            fov_degrees=values["fov_degrees"],
            radius=values["radius"],
            global_residuals=values["global_residuals"],
            rotation_delta=values["rotation_delta"],
            translation_delta=values["translation_delta"],
            path_residuals=values.get("path_residuals"),
        )


@dataclass(frozen=True)
class GaussianFrame:
    """One renderable gaussian frame.

    xyz: [G, 3]
    scales: [G, 3]
    quats: [G, 4], normalized
    opacities: [G, 1]
    rgbs: [G, F]: F-channel splat features. F=3 means RGB-3 (legacy
        path); F>3 requires a downstream colorize MLP to map to RGB at
        the loss boundary. Field name retained for cascade compatibility.
    """

    xyz: Tensor
    scales: Tensor
    quats: Tensor
    opacities: Tensor
    rgbs: Tensor

    @property
    def gaussian_count(self) -> int:
        return int(self.xyz.shape[0])

    def float(self) -> "GaussianFrame":
        return GaussianFrame(
            xyz=self.xyz.float(),
            scales=self.scales.float(),
            quats=self.quats.float(),
            opacities=self.opacities.float(),
            rgbs=self.rgbs.float(),
        )


@dataclass(frozen=True)
class GaussianSequence:
    """Decoded model output for K frames.

    Tensor shapes are [K, G, C].
    rgbs is [K, G, F]: F-channel splat features. F=3 means RGB-3 (legacy
    path); F>3 requires a downstream colorize MLP to map to RGB at the loss
    boundary. Field name retained for cascade compatibility.
    cameras is present for implicit-camera outputs and known-camera render payloads.
    camera_state is present only for implicit-camera models.
    """

    xyz: Tensor
    scales: Tensor
    quats: Tensor
    opacities: Tensor
    rgbs: Tensor
    cameras: tuple[CameraSpec, ...] | None = None
    camera_state: CameraState | None = None
    auxiliary: Mapping[str, Any] = field(default_factory=dict)

    @property
    def frame_count(self) -> int:
        return int(self.xyz.shape[0])

    @property
    def gaussian_count(self) -> int:
        return int(self.xyz.shape[1])

    def frame(self, index: int) -> GaussianFrame:
        return GaussianFrame(
            xyz=self.xyz[index],
            scales=self.scales[index],
            quats=self.quats[index],
            opacities=self.opacities[index],
            rgbs=self.rgbs[index],
        )


@dataclass(frozen=True)
class RasterizedClip:
    """Trainer-side raster output before RGB objective composition.

    Features are [T, F, H, W]. Alpha is [T, H, W] when the active renderer can
    expose it, otherwise None for legacy renderers.
    """

    features: Tensor
    alpha: Tensor | None


@dataclass(frozen=True)
class RenderedClip:
    """Full-sequence validation render bundle stitched from chunked eval.

    This is distinct from `objective.types.RenderedView`, which represents one
    target view through rasterization, colorization, background composition, and
    loss. `RenderedClip` is the runtime payload returned by validation sequence
    rendering.
    """

    rgb_sequence: Tensor
    camera_state: CameraState | None
    temporal_metrics: dict[str, float]
    feature_sequence: Tensor | None
    alpha_sequence: Tensor | None


@dataclass
class StepResult:
    source_path: Path | None
    sequence_frame_count: int
    clip_frames: Tensor
    preview_render: Tensor | None
    preview_features: Tensor | None
    camera_state: CameraState | None
    loss: Tensor
    recon_loss: Tensor
    camera_motion_loss: Tensor
    camera_temporal_loss: Tensor
    camera_global_loss: Tensor
    bank_rate_loss: Tensor
    bank_rate_terms: dict[str, Tensor]
    aux_loss_terms: dict[str, Tensor] = field(default_factory=dict)


def _detached_or_zero(value: Tensor | None, zero: Tensor) -> Tensor:
    if value is None:
        return zero
    return value.detach()


def _detached_terms(terms: Mapping[str, Tensor] | None) -> dict[str, Tensor]:
    if terms is None:
        return {}
    return {key: value.detach() for key, value in terms.items()}


def build_step_result(
    *,
    sequence_data: SequenceData,
    clip_frames: Tensor,
    preview_render: Tensor | None,
    preview_features: Tensor | None,
    camera_state: CameraState | None,
    loss: Tensor,
    recon_loss: Tensor,
    bank_rate_loss: Tensor,
    bank_rate_terms: Mapping[str, Tensor] | None = None,
    camera_motion_loss: Tensor | None = None,
    camera_temporal_loss: Tensor | None = None,
    camera_global_loss: Tensor | None = None,
    aux_loss_terms: Mapping[str, Tensor] | None = None,
) -> StepResult:
    zero = clip_frames.new_zeros(())
    return StepResult(
        source_path=sequence_data.source_path,
        sequence_frame_count=sequence_data.frame_count,
        clip_frames=clip_frames,
        preview_render=preview_render,
        preview_features=preview_features,
        camera_state=camera_state,
        loss=loss.detach(),
        recon_loss=recon_loss.detach(),
        camera_motion_loss=_detached_or_zero(camera_motion_loss, zero),
        camera_temporal_loss=_detached_or_zero(camera_temporal_loss, zero),
        camera_global_loss=_detached_or_zero(camera_global_loss, zero),
        bank_rate_loss=bank_rate_loss.detach(),
        bank_rate_terms=_detached_terms(bank_rate_terms),
        aux_loss_terms=_detached_terms(aux_loss_terms),
    )
