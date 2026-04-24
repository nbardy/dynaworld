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
    rgbs: [G, 3], 0..1
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
class StepLosses:
    total: Tensor
    reconstruction: Tensor
    camera_motion: Tensor | None = None
    camera_temporal: Tensor | None = None
    camera_global: Tensor | None = None

    def scalar_payload(self) -> dict[str, float]:
        payload = {
            "total": float(self.total.detach().item()),
            "reconstruction": float(self.reconstruction.detach().item()),
        }
        if self.camera_motion is not None:
            payload["camera_motion"] = float(self.camera_motion.detach().item())
        if self.camera_temporal is not None:
            payload["camera_temporal"] = float(self.camera_temporal.detach().item())
        if self.camera_global is not None:
            payload["camera_global"] = float(self.camera_global.detach().item())
        return payload


@dataclass(frozen=True)
class TrainStepResult:
    batch: ClipBatch
    decoded: GaussianSequence
    losses: StepLosses
    preview_render: Tensor | None = None
