from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from renderers.common import MIN_RENDER_DEPTH, project_gaussians_2d, project_gaussians_2d_batch


VENDORED_TAICHI_SPLATTING_DIR = Path(__file__).resolve().parents[3] / "third_party" / "taichi-splatting"


@dataclass(frozen=True)
class TaichiRendererConfig:
    tile_size: int = 16
    alpha_threshold: float = 1.0 / 255.0
    clamp_max_alpha: float = 0.99
    saturate_threshold: float = 0.9999
    kernel_variant: str = "auto"
    sort_backend: str = "auto"
    backward_variant: str = "pixel_reference"
    metal_block_dim: int = 0
    metal_compatible: bool | None = None
    use_depth16: bool = False

    @classmethod
    def from_mapping(
        cls,
        values: dict[str, Any] | None,
        *,
        fallback_tile_size: int,
        fallback_alpha_threshold: float,
    ) -> "TaichiRendererConfig":
        values = values or {}
        return cls(
            tile_size=int(values.get("tile_size", fallback_tile_size)),
            alpha_threshold=float(values.get("alpha_threshold", fallback_alpha_threshold)),
            clamp_max_alpha=float(values.get("clamp_max_alpha", cls.clamp_max_alpha)),
            saturate_threshold=float(values.get("saturate_threshold", cls.saturate_threshold)),
            kernel_variant=str(values.get("kernel_variant", values.get("variant", cls.kernel_variant))),
            sort_backend=str(values.get("sort_backend", cls.sort_backend)),
            backward_variant=str(values.get("backward_variant", cls.backward_variant)),
            metal_block_dim=int(values.get("metal_block_dim", cls.metal_block_dim)),
            metal_compatible=(
                None if values.get("metal_compatible") is None else bool(values.get("metal_compatible"))
            ),
            use_depth16=bool(values.get("use_depth16", cls.use_depth16)),
        )


def _ensure_vendored_taichi_splatting_on_path() -> None:
    if VENDORED_TAICHI_SPLATTING_DIR.exists() and str(VENDORED_TAICHI_SPLATTING_DIR) not in sys.path:
        sys.path.insert(0, str(VENDORED_TAICHI_SPLATTING_DIR))


def _resolve_kernel_variant(device: torch.device, variant: str) -> str:
    if variant != "auto":
        return variant
    return "cuda_simt" if device.type == "cuda" else "metal_reference"


@lru_cache(maxsize=8)
def _build_raster_config(
    device_type: str,
    tile_size: int,
    alpha_threshold: float,
    clamp_max_alpha: float,
    saturate_threshold: float,
    kernel_variant: str,
    sort_backend: str,
    backward_variant: str,
    metal_block_dim: int,
    metal_compatible: bool | None,
):
    _ensure_vendored_taichi_splatting_on_path()

    import taichi as ti
    from taichi_splatting.data_types import RasterConfig
    from taichi_splatting.taichi_queue import TaichiQueue

    arch_by_device = {"cuda": ti.cuda, "mps": ti.metal, "cpu": ti.cpu}
    if device_type not in arch_by_device:
        raise RuntimeError(f"Unsupported Taichi renderer device type: {device_type}")

    TaichiQueue.init(arch=arch_by_device[device_type], log_level=ti.ERROR)
    resolved_variant = _resolve_kernel_variant(torch.device(device_type), kernel_variant)
    if resolved_variant not in {"metal_reference", "cuda_simt"}:
        raise RuntimeError(f"Taichi renderer kernel_variant={resolved_variant!r} is not implemented for training.")
    if backward_variant != "pixel_reference":
        raise RuntimeError(f"Taichi renderer backward_variant={backward_variant!r} is not implemented for training.")

    return RasterConfig(
        tile_size=tile_size,
        alpha_threshold=alpha_threshold,
        clamp_max_alpha=clamp_max_alpha,
        saturate_threshold=saturate_threshold,
        metal_compatible=bool(device_type != "cuda") if metal_compatible is None else metal_compatible,
        kernel_variant=resolved_variant,
        sort_backend=sort_backend,
        backward_variant=backward_variant,
        metal_block_dim=metal_block_dim,
    )


def project_for_taichi_axis(
    means3d: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    rgbs: torch.Tensor,
    fx,
    fy,
    cx,
    cy,
    camera_to_world: torch.Tensor | None = None,
    near_plane: float = MIN_RENDER_DEPTH,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    means2d, _inv_cov2d, cov2d, opacities, colors = project_gaussians_2d(
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

    cov_xx = cov2d[:, 0, 0]
    cov_xy = cov2d[:, 0, 1]
    cov_yy = cov2d[:, 1, 1]
    trace_half = 0.5 * (cov_xx + cov_yy)
    delta = torch.sqrt(torch.clamp((0.5 * (cov_xx - cov_yy)).square() + cov_xy.square(), min=1.0e-12))
    lambda_major = trace_half + delta
    lambda_minor = trace_half - delta

    axis = torch.stack([cov_xy, lambda_major - cov_xx], dim=-1)
    axis_norm = torch.linalg.norm(axis, dim=-1, keepdim=True)
    fallback_axis = torch.zeros_like(axis)
    fallback_axis[:, 0] = 1.0
    axis = torch.where(axis_norm > 1.0e-6, axis / axis_norm.clamp_min(1.0e-6), fallback_axis)

    sigma = torch.sqrt(torch.clamp(torch.stack([lambda_major, lambda_minor], dim=-1), min=1.0e-6))
    packed = torch.cat([means2d, axis, sigma, opacities], dim=-1).contiguous()

    # project_gaussians_2d already sorts front-to-back. Taichi's tile mapper only
    # needs a nonnegative scalar preserving that order, so use rank-depth here.
    depths = torch.arange(packed.shape[0], device=packed.device, dtype=packed.dtype).view(-1, 1)
    if packed.shape[0] > 1:
        depths = depths / float(packed.shape[0] - 1)

    background = torch.ones(colors.shape[-1], device=colors.device, dtype=colors.dtype)
    return packed, depths.contiguous(), colors.contiguous(), background


def project_for_taichi_axis_batch(
    means3d: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    rgbs: torch.Tensor,
    fx,
    fy,
    cx,
    cy,
    camera_to_world: torch.Tensor | None = None,
    near_plane: float = MIN_RENDER_DEPTH,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    means2d, _inv_cov2d, cov2d, opacities, colors = project_gaussians_2d_batch(
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

    cov_xx = cov2d[..., 0, 0]
    cov_xy = cov2d[..., 0, 1]
    cov_yy = cov2d[..., 1, 1]
    trace_half = 0.5 * (cov_xx + cov_yy)
    delta = torch.sqrt(torch.clamp((0.5 * (cov_xx - cov_yy)).square() + cov_xy.square(), min=1.0e-12))
    lambda_major = trace_half + delta
    lambda_minor = trace_half - delta

    axis = torch.stack([cov_xy, lambda_major - cov_xx], dim=-1)
    axis_norm = torch.linalg.norm(axis, dim=-1, keepdim=True)
    fallback_axis = torch.zeros_like(axis)
    fallback_axis[..., 0] = 1.0
    axis = torch.where(axis_norm > 1.0e-6, axis / axis_norm.clamp_min(1.0e-6), fallback_axis)

    sigma = torch.sqrt(torch.clamp(torch.stack([lambda_major, lambda_minor], dim=-1), min=1.0e-6))
    packed = torch.cat([means2d, axis, sigma, opacities], dim=-1).contiguous()

    depths = torch.arange(packed.shape[1], device=packed.device, dtype=packed.dtype).view(1, -1, 1)
    depths = depths.expand(packed.shape[0], -1, -1).contiguous()
    if packed.shape[1] > 1:
        depths = depths / float(packed.shape[1] - 1)

    background = torch.ones(colors.shape[-1], device=colors.device, dtype=colors.dtype)
    return packed, depths.contiguous(), colors.contiguous(), background


def render_taichi_3dgs(
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
    camera_to_world: torch.Tensor | None = None,
    near_plane: float = MIN_RENDER_DEPTH,
    config: TaichiRendererConfig,
) -> torch.Tensor:
    _ensure_vendored_taichi_splatting_on_path()

    from taichi_splatting.rasterizer import rasterize

    packed, depths, colors, background = project_for_taichi_axis(
        means3d.float(),
        scales.float(),
        quats.float(),
        opacities.float(),
        rgbs.float(),
        fx,
        fy,
        cx,
        cy,
        camera_to_world=camera_to_world.float() if camera_to_world is not None else None,
        near_plane=near_plane,
    )

    raster_config = _build_raster_config(
        packed.device.type,
        config.tile_size,
        config.alpha_threshold,
        config.clamp_max_alpha,
        config.saturate_threshold,
        config.kernel_variant,
        config.sort_backend,
        config.backward_variant,
        config.metal_block_dim,
        config.metal_compatible,
    )
    raster = rasterize(
        packed,
        depths,
        colors,
        image_size=(width, height),
        config=raster_config,
        use_depth16=config.use_depth16,
    )
    image_hwc = raster.image + (1.0 - raster.image_weight).unsqueeze(-1).clamp(0.0, 1.0) * background
    return image_hwc.clamp(0.0, 1.0).permute(2, 0, 1).contiguous()


def render_taichi_3dgs_batch(
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
    camera_to_world: torch.Tensor | None = None,
    near_plane: float = MIN_RENDER_DEPTH,
    config: TaichiRendererConfig,
) -> torch.Tensor:
    _ensure_vendored_taichi_splatting_on_path()

    from taichi_splatting.rasterizer import rasterize_batch

    packed, depths, colors, background = project_for_taichi_axis_batch(
        means3d.float(),
        scales.float(),
        quats.float(),
        opacities.float(),
        rgbs.float(),
        fx,
        fy,
        cx,
        cy,
        camera_to_world=camera_to_world.float() if camera_to_world is not None else None,
        near_plane=near_plane,
    )

    raster_config = _build_raster_config(
        packed.device.type,
        config.tile_size,
        config.alpha_threshold,
        config.clamp_max_alpha,
        config.saturate_threshold,
        config.kernel_variant,
        config.sort_backend,
        config.backward_variant,
        config.metal_block_dim,
        config.metal_compatible,
    )
    raster = rasterize_batch(
        packed,
        depths,
        colors,
        image_size=(width, height),
        config=raster_config,
        use_depth16=config.use_depth16,
    )
    image_bhwc = raster.image + (1.0 - raster.image_weight).unsqueeze(-1).clamp(0.0, 1.0) * background
    return image_bhwc.clamp(0.0, 1.0).permute(0, 3, 1, 2).contiguous()


__all__ = [
    "TaichiRendererConfig",
    "project_for_taichi_axis",
    "project_for_taichi_axis_batch",
    "render_taichi_3dgs",
    "render_taichi_3dgs_batch",
]
