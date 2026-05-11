from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

import torch

try:
    from camera import CameraSpec
except ImportError:  # pragma: no cover - package-style imports in tests.
    from ..camera import CameraSpec


RunPhase = Literal["train", "eval", "preview", "export"]
ViewRole = Literal["source", "train", "heldout", "debug"]
CameraRole = Literal["condition", "anchor", "target", "heldout", "debug"]
CameraOwner = Literal["model", "target", "external_rig", "none"]
BackgroundMode = Literal["white", "black", "fixed_rgb", "random_rgb", "none"]
BackgroundSampleScope = Literal["step", "view", "frame", "pixel"]
ReconstructionLossKind = Literal["mse", "l1", "l1_mse", "standard_gs"]
DSSIMBackend = Literal["torch", "metal"]


@dataclass(frozen=True)
class TargetView:
    view_id: str
    role: ViewRole
    camera_role: CameraRole
    camera_owner: CameraOwner
    frames: torch.Tensor
    frame_indices: torch.Tensor
    frame_times: torch.Tensor  # [K, 1] or [K]
    video_fps: float
    camera_name: str | None = None
    cameras: tuple[CameraSpec, ...] | None = None
    loss_weight: float = 1.0
    metrics_prefix: str | None = None
    log_media: bool = False

    @property
    def frame_count(self) -> int:
        return int(self.frames.shape[0])


@dataclass(frozen=True)
class RasterizedView:
    view: TargetView
    features: torch.Tensor
    alpha: torch.Tensor | None
    cameras: tuple[CameraSpec, ...] | None = None
    view_dirs: torch.Tensor | None = None

    @property
    def frame_count(self) -> int:
        return int(self.features.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    @property
    def render_size(self) -> int:
        return int(self.features.shape[-1])


@dataclass(frozen=True)
class ColorizedView:
    splat_rgb: torch.Tensor  # [K, 3, H, W]
    logits: torch.Tensor | None = None
    view_dirs: torch.Tensor | None = None


@dataclass(frozen=True)
class BackgroundSpec:
    train_mode: BackgroundMode = "random_rgb"
    eval_mode: BackgroundMode = "white"
    preview_mode: BackgroundMode | None = None
    export_mode: BackgroundMode | None = None
    fixed_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)
    sample_scope: BackgroundSampleScope = "step"
    apply_when_alpha_missing: bool = False


@dataclass(frozen=True)
class BackgroundSample:
    rgb: torch.Tensor | None  # Broadcastable to [K, 3, H, W], or None for no bg.
    mode: BackgroundMode
    phase: RunPhase
    step: int | None = None


@dataclass(frozen=True)
class ReconstructionLossSpec:
    kind: ReconstructionLossKind = "standard_gs"
    l1_weight: float = 0.8
    dssim_weight: float = 0.2
    mse_weight: float = 0.0
    ssim_window_size: int = 11
    ssim_c1: float = 0.01**2
    ssim_c2: float = 0.03**2
    dssim_backend: DSSIMBackend = "torch"


@dataclass(frozen=True)
class ObjectiveSpec:
    version: str
    reconstruction: ReconstructionLossSpec = field(default_factory=ReconstructionLossSpec)
    background: BackgroundSpec = field(default_factory=BackgroundSpec)
    retain_train_artifacts: bool = False


@dataclass(frozen=True)
class RenderedView:
    view: TargetView
    rgb: torch.Tensor
    target_rgb: torch.Tensor | None
    rasterized: RasterizedView
    colorized: ColorizedView | None
    background: BackgroundSample
    phase: RunPhase
    metrics_prefix: str | None = None

    @property
    def features(self) -> torch.Tensor:
        return self.rasterized.features

    @property
    def alpha(self) -> torch.Tensor | None:
        return self.rasterized.alpha

    @property
    def splat_rgb(self) -> torch.Tensor | None:
        return None if self.colorized is None else self.colorized.splat_rgb

    def detach_cpu(self) -> "RenderedView":
        view = replace(
            self.view,
            frames=self.view.frames.detach().cpu(),
            frame_indices=self.view.frame_indices.detach().cpu(),
            frame_times=self.view.frame_times.detach().cpu(),
        )
        return replace(
            self,
            view=view,
            rgb=self.rgb.detach().cpu(),
            target_rgb=None if self.target_rgb is None else self.target_rgb.detach().cpu(),
            rasterized=replace(
                self.rasterized,
                view=view,
                features=self.rasterized.features.detach().cpu(),
                alpha=None if self.rasterized.alpha is None else self.rasterized.alpha.detach().cpu(),
                view_dirs=None if self.rasterized.view_dirs is None else self.rasterized.view_dirs.detach().cpu(),
            ),
            colorized=None
            if self.colorized is None
            else replace(
                self.colorized,
                splat_rgb=self.colorized.splat_rgb.detach().cpu(),
                logits=None if self.colorized.logits is None else self.colorized.logits.detach().cpu(),
                view_dirs=None if self.colorized.view_dirs is None else self.colorized.view_dirs.detach().cpu(),
            ),
            background=replace(
                self.background,
                rgb=None if self.background.rgb is None else self.background.rgb.detach().cpu(),
            ),
        )


@dataclass(frozen=True)
class ViewLoss:
    view_id: str
    role: ViewRole
    total: torch.Tensor
    per_image: torch.Tensor  # [K]
    weight: float
