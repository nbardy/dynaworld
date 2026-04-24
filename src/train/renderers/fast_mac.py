from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from renderers.common import MIN_RENDER_DEPTH, project_gaussians_2d, project_gaussians_2d_batch
from renderers.projection import project_gaussians_2d_camera, project_gaussians_2d_camera_batch

FAST_MAC_V5_DIR = Path(__file__).resolve().parents[3] / "third_party" / "fast-mac-gsplat" / "variants" / "v5"


@dataclass(frozen=True)
class FastMacRendererConfig:
    tile_size: int = 16
    max_fast_pairs: int = 2048
    alpha_threshold: float = 1.0 / 255.0
    transmittance_threshold: float = 1.0e-4
    background: tuple[float, float, float] = (1.0, 1.0, 1.0)
    enable_overflow_fallback: bool = True
    batch_strategy: str = "flatten"
    batch_launch_limit_tiles: int = 262144
    batch_launch_limit_gaussians: int = 262144

    @classmethod
    def from_mapping(
        cls,
        values: dict[str, Any] | None,
        *,
        fallback_tile_size: int,
        fallback_alpha_threshold: float,
    ) -> "FastMacRendererConfig":
        values = values or {}
        background = values.get("background", cls.background)
        if len(background) != 3:
            raise ValueError(f"fast_mac.background must contain three values, got {background!r}.")
        return cls(
            tile_size=int(values.get("tile_size", fallback_tile_size)),
            max_fast_pairs=int(values.get("max_fast_pairs", cls.max_fast_pairs)),
            alpha_threshold=float(values.get("alpha_threshold", fallback_alpha_threshold)),
            transmittance_threshold=float(values.get("transmittance_threshold", cls.transmittance_threshold)),
            background=tuple(float(value) for value in background),
            enable_overflow_fallback=bool(values.get("enable_overflow_fallback", cls.enable_overflow_fallback)),
            batch_strategy=str(values.get("batch_strategy", cls.batch_strategy)),
            batch_launch_limit_tiles=int(values.get("batch_launch_limit_tiles", cls.batch_launch_limit_tiles)),
            batch_launch_limit_gaussians=int(
                values.get("batch_launch_limit_gaussians", cls.batch_launch_limit_gaussians)
            ),
        )


def _ensure_fast_mac_v5_on_path() -> None:
    if not FAST_MAC_V5_DIR.exists():
        raise RuntimeError(f"fast-mac-gsplat v5 directory not found: {FAST_MAC_V5_DIR}")
    if str(FAST_MAC_V5_DIR) not in sys.path:
        sys.path.insert(0, str(FAST_MAC_V5_DIR))


def _make_v5_config(config: FastMacRendererConfig, height: int, width: int):
    _ensure_fast_mac_v5_on_path()
    from torch_gsplat_bridge_v5 import RasterConfig

    return RasterConfig(
        height=height,
        width=width,
        tile_size=config.tile_size,
        max_fast_pairs=config.max_fast_pairs,
        alpha_threshold=config.alpha_threshold,
        transmittance_threshold=config.transmittance_threshold,
        background=config.background,
        enable_overflow_fallback=config.enable_overflow_fallback,
        batch_strategy=config.batch_strategy,
        batch_launch_limit_tiles=config.batch_launch_limit_tiles,
        batch_launch_limit_gaussians=config.batch_launch_limit_gaussians,
    )


def _conics_from_inv_cov(inv_cov2d: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            inv_cov2d[..., 0, 0],
            0.5 * (inv_cov2d[..., 0, 1] + inv_cov2d[..., 1, 0]),
            inv_cov2d[..., 1, 1],
        ],
        dim=-1,
    ).contiguous()


def _rank_depths(
    gaussian_count: int, *, batch_size: int | None, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    depths = torch.arange(gaussian_count, device=device, dtype=dtype)
    if gaussian_count > 1:
        depths = depths / float(gaussian_count - 1)
    if batch_size is None:
        return depths.contiguous()
    return depths.view(1, -1).expand(batch_size, -1).contiguous()


def project_for_fast_mac(
    means3d: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    rgbs: torch.Tensor,
    fx,
    fy,
    cx,
    cy,
    *,
    camera=None,
    projection_mode: str = "legacy_pinhole",
    camera_to_world: torch.Tensor | None = None,
    near_plane: float = MIN_RENDER_DEPTH,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if projection_mode == "camera_model":
        if camera is None:
            raise ValueError("camera_model projection requires a CameraSpec.")
        means2d, inv_cov2d, _cov2d, opacities, colors = project_gaussians_2d_camera(
            means3d,
            scales,
            quats,
            opacities,
            rgbs,
            camera,
            near_plane=near_plane,
        )
    elif projection_mode == "legacy_pinhole":
        means2d, inv_cov2d, _cov2d, opacities, colors = project_gaussians_2d(
            means3d,
            scales,
            quats,
            opacities,
            rgbs,
            fx,
            fy,
            cx,
            cy,
            camera_to_world=camera_to_world,
            near_plane=near_plane,
        )
    else:
        raise ValueError(f"Unknown projection_mode: {projection_mode}")
    depths = _rank_depths(means2d.shape[0], batch_size=None, device=means2d.device, dtype=means2d.dtype)
    return (
        means2d.contiguous(),
        _conics_from_inv_cov(inv_cov2d),
        colors.contiguous(),
        opacities.squeeze(-1).contiguous(),
        depths,
    )


def project_for_fast_mac_batch(
    means3d: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    rgbs: torch.Tensor,
    fx,
    fy,
    cx,
    cy,
    *,
    cameras=None,
    projection_mode: str = "legacy_pinhole",
    camera_to_world: torch.Tensor | None = None,
    near_plane: float = MIN_RENDER_DEPTH,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if projection_mode == "camera_model":
        if cameras is None:
            raise ValueError("camera_model batch projection requires CameraSpec values.")
        means2d, inv_cov2d, _cov2d, opacities, colors = project_gaussians_2d_camera_batch(
            means3d,
            scales,
            quats,
            opacities,
            rgbs,
            cameras,
            near_plane=near_plane,
        )
    elif projection_mode == "legacy_pinhole":
        means2d, inv_cov2d, _cov2d, opacities, colors = project_gaussians_2d_batch(
            means3d,
            scales,
            quats,
            opacities,
            rgbs,
            fx,
            fy,
            cx,
            cy,
            camera_to_world=camera_to_world,
            near_plane=near_plane,
        )
    else:
        raise ValueError(f"Unknown projection_mode: {projection_mode}")
    depths = _rank_depths(
        means2d.shape[1],
        batch_size=means2d.shape[0],
        device=means2d.device,
        dtype=means2d.dtype,
    )
    return (
        means2d.contiguous(),
        _conics_from_inv_cov(inv_cov2d),
        colors.contiguous(),
        opacities.squeeze(-1).contiguous(),
        depths,
    )


def render_fast_mac_3dgs(
    means3d: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    rgbs: torch.Tensor,
    height: int,
    width: int,
    fx,
    fy,
    cx,
    cy,
    *,
    camera=None,
    projection_mode: str = "legacy_pinhole",
    camera_to_world: torch.Tensor | None = None,
    near_plane: float = MIN_RENDER_DEPTH,
    config: FastMacRendererConfig,
) -> torch.Tensor:
    _ensure_fast_mac_v5_on_path()
    from torch_gsplat_bridge_v5 import rasterize_projected_gaussians

    means2d, conics, colors, projected_opacities, depths = project_for_fast_mac(
        means3d.float(),
        scales.float(),
        quats.float(),
        opacities.float(),
        rgbs.float(),
        fx,
        fy,
        cx,
        cy,
        camera=camera,
        projection_mode=projection_mode,
        camera_to_world=camera_to_world.float() if camera_to_world is not None else None,
        near_plane=near_plane,
    )
    image_hwc = rasterize_projected_gaussians(
        means2d,
        conics,
        colors,
        projected_opacities,
        depths,
        _make_v5_config(config, height, width),
    )
    return image_hwc.clamp(0.0, 1.0).permute(2, 0, 1).contiguous()


def render_fast_mac_3dgs_batch(
    means3d: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    rgbs: torch.Tensor,
    height: int,
    width: int,
    fx,
    fy,
    cx,
    cy,
    *,
    cameras=None,
    projection_mode: str = "legacy_pinhole",
    camera_to_world: torch.Tensor | None = None,
    near_plane: float = MIN_RENDER_DEPTH,
    config: FastMacRendererConfig,
) -> torch.Tensor:
    _ensure_fast_mac_v5_on_path()
    from torch_gsplat_bridge_v5 import rasterize_projected_gaussians

    means2d, conics, colors, projected_opacities, depths = project_for_fast_mac_batch(
        means3d.float(),
        scales.float(),
        quats.float(),
        opacities.float(),
        rgbs.float(),
        fx,
        fy,
        cx,
        cy,
        cameras=cameras,
        projection_mode=projection_mode,
        camera_to_world=camera_to_world.float() if camera_to_world is not None else None,
        near_plane=near_plane,
    )
    image_bhwc = rasterize_projected_gaussians(
        means2d,
        conics,
        colors,
        projected_opacities,
        depths,
        _make_v5_config(config, height, width),
    )
    return image_bhwc.clamp(0.0, 1.0).permute(0, 3, 1, 2).contiguous()


__all__ = [
    "FastMacRendererConfig",
    "project_for_fast_mac",
    "project_for_fast_mac_batch",
    "render_fast_mac_3dgs",
    "render_fast_mac_3dgs_batch",
]
