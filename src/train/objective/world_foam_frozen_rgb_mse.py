from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch


PROMOTED_FRAMEGROUP16_TAPE_MODE = "endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse"
FROZEN_SITE_RGBA_GRADIENT_SCOPE = (
    "frozen_geometry_endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse_"
    "rgb_only_site_rgba_autograd"
)

FRAMEGROUP16_TAPE_KEYS = (
    "coeff_f16",
    "frame_t_f32",
    "base_offsets_i32",
    "base_record_i16",
    "track_change_offsets_i32",
    "track_chunk_change_offsets_i16",
    "change_frame_i32",
    "change_offsets_i32",
    "change_record_i16",
)


@dataclass(frozen=True)
class WorldFoamFrozenRGBMSEScope:
    tape_mode: str = PROMOTED_FRAMEGROUP16_TAPE_MODE
    gradient_scope: str = FROZEN_SITE_RGBA_GRADIENT_SCOPE
    full_trainer_claim: bool = False
    full_geometry_gradient_claim: bool = False
    quality_claim: bool = False
    renderer_backend_claim: bool = False
    supports_rgb_mse_only: bool = True
    supports_background_composition: bool = False
    supports_colorizer: bool = False
    supports_vjepa_feature_loss: bool = False


@dataclass(frozen=True)
class WorldFoamTargetLayout:
    view_count: int
    frame_count: int
    height: int
    width: int

    def __post_init__(self) -> None:
        for name in ("view_count", "frame_count", "height", "width"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")

    @classmethod
    def from_track_major(cls, *, track_count: int, frame_count: int) -> "WorldFoamTargetLayout":
        if int(track_count) <= 0:
            raise ValueError("track_count must be positive")
        return cls(view_count=1, frame_count=int(frame_count), height=int(track_count), width=1)

    @property
    def track_count(self) -> int:
        return int(self.view_count) * int(self.height) * int(self.width)


FusedFramegroup16LossFn = Callable[..., torch.Tensor]


def validate_world_foam_frozen_rgb_mse_scope(
    *,
    loss_kind: str,
    feature_dim: int,
    vjepa_feature_weight: float = 0.0,
    uses_colorizer: bool = False,
    uses_background_composition: bool = False,
) -> None:
    if str(loss_kind).lower() != "mse":
        raise ValueError("WorldFoam frozen fused-MSE path currently supports only losses.type='mse'.")
    if int(feature_dim) != 3:
        raise ValueError("WorldFoam frozen fused-MSE path currently supports only RGB feature_dim=3.")
    if float(vjepa_feature_weight) != 0.0:
        raise ValueError("WorldFoam frozen fused-MSE path does not support V-JEPA feature loss.")
    if uses_colorizer:
        raise ValueError("WorldFoam frozen fused-MSE path does not support a colorizer.")
    if uses_background_composition:
        raise ValueError("WorldFoam frozen fused-MSE path does not support background composition.")


def target_rgb_to_track_major(target_rgb: torch.Tensor, layout: WorldFoamTargetLayout) -> torch.Tensor:
    """Convert trainer target RGB layouts into [track, frame, 3] for the fused kernel."""
    if target_rgb.dim() == 3:
        expected = (layout.track_count, layout.frame_count, 3)
        if tuple(target_rgb.shape) != expected:
            raise ValueError(f"track-major target_rgb must have shape {expected}, got {tuple(target_rgb.shape)}")
        return target_rgb.contiguous()
    if target_rgb.dim() == 4:
        expected = (layout.view_count * layout.frame_count, 3, layout.height, layout.width)
        if tuple(target_rgb.shape) != expected:
            raise ValueError(f"image target_rgb must have shape {expected}, got {tuple(target_rgb.shape)}")
        return (
            target_rgb.reshape(layout.view_count, layout.frame_count, 3, layout.height, layout.width)
            .permute(0, 3, 4, 1, 2)
            .reshape(layout.track_count, layout.frame_count, 3)
            .contiguous()
        )
    if target_rgb.dim() == 5:
        expected = (layout.view_count, layout.frame_count, 3, layout.height, layout.width)
        if tuple(target_rgb.shape) != expected:
            raise ValueError(f"view-major target_rgb must have shape {expected}, got {tuple(target_rgb.shape)}")
        return (
            target_rgb.permute(0, 3, 4, 1, 2)
            .reshape(layout.track_count, layout.frame_count, 3)
            .contiguous()
        )
    raise ValueError(
        "target_rgb must be [track, frame, 3], [view*frame, 3, height, width], "
        "or [view, frame, 3, height, width]."
    )


class WorldFoamFrozenRGBMSEObjective:
    """Narrow adapter for the promoted frozen-geometry WorldFoam fused-MSE kernel."""

    scope = WorldFoamFrozenRGBMSEScope()

    def __init__(
        self,
        *,
        tape: Mapping[str, torch.Tensor],
        config: Any,
        boundary_count: int,
        layout: WorldFoamTargetLayout,
        fused_loss_fn: FusedFramegroup16LossFn,
    ) -> None:
        missing = [key for key in FRAMEGROUP16_TAPE_KEYS if key not in tape]
        if missing:
            raise ValueError(f"WorldFoam framegroup16 tape is missing keys: {missing}")
        if int(boundary_count) <= 0:
            raise ValueError("boundary_count must be positive")
        self.tape = tape
        self.config = config
        self.boundary_count = int(boundary_count)
        self.layout = layout
        self.fused_loss_fn = fused_loss_fn

    def target_track_major(self, target_rgb: torch.Tensor) -> torch.Tensor:
        return target_rgb_to_track_major(target_rgb, self.layout)

    def loss(self, *, site_rgba: torch.Tensor, target_rgb: torch.Tensor) -> torch.Tensor:
        target_track = self.target_track_major(target_rgb).to(device=site_rgba.device, dtype=torch.float32)
        return self.fused_loss_fn(
            coeff_f16=self.tape["coeff_f16"],
            frame_t_f32=self.tape["frame_t_f32"],
            base_offsets_i32=self.tape["base_offsets_i32"],
            base_record_i16=self.tape["base_record_i16"],
            track_change_offsets_i32=self.tape["track_change_offsets_i32"],
            track_chunk_change_offsets_i16=self.tape["track_chunk_change_offsets_i16"],
            change_frame_i32=self.tape["change_frame_i32"],
            change_offsets_i32=self.tape["change_offsets_i32"],
            change_record_i16=self.tape["change_record_i16"],
            site_rgba_f32=site_rgba,
            target_rgb_f32=target_track,
            config=self.config,
            track_count=self.layout.track_count,
            frame_count=self.layout.frame_count,
            boundary_count=self.boundary_count,
        )


def promoted_framegroup16_loss_fn() -> FusedFramegroup16LossFn:
    try:
        from torch_world_foam_lane2_fused_slab import (
            endpoint_record_delta_replace_coeff16_i16x3_framegroup16_mse_autograd,
        )
    except ImportError as exc:  # pragma: no cover - depends on optional local Metal variant path.
        raise RuntimeError(
            "Promoted WorldFoam framegroup16 fused-MSE op is unavailable. "
            "Build the world_foam_lane2_fused_slab_v0 variant and put it on PYTHONPATH."
        ) from exc
    return endpoint_record_delta_replace_coeff16_i16x3_framegroup16_mse_autograd


__all__ = [
    "FRAMEGROUP16_TAPE_KEYS",
    "FROZEN_SITE_RGBA_GRADIENT_SCOPE",
    "PROMOTED_FRAMEGROUP16_TAPE_MODE",
    "WorldFoamFrozenRGBMSEObjective",
    "WorldFoamFrozenRGBMSEScope",
    "WorldFoamTargetLayout",
    "promoted_framegroup16_loss_fn",
    "target_rgb_to_track_major",
    "validate_world_foam_frozen_rgb_mse_scope",
]
