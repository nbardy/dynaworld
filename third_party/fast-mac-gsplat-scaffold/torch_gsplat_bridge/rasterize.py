from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor

try:
    from . import _C  # noqa: F401
except Exception:
    _C = None


@dataclass(frozen=True)
class RasterConfig:
    height: int
    width: int
    tile_size: int = 16
    chunk_size: int = 32
    alpha_threshold: float = 1.0 / 255.0
    transmittance_threshold: float = 1e-4
    background: Tuple[float, float, float] = (0.0, 0.0, 0.0)


class _RasterizeProjectedGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means2d: Tensor,
        conics: Tensor,
        colors: Tensor,
        opacities: Tensor,
        depths: Tensor,
        meta_i32: Tensor,
        meta_f32: Tensor,
    ) -> Tensor:
        if not hasattr(torch.ops, "gsplat_metal"):
            raise RuntimeError("gsplat_metal custom ops not found. Build the extension first.")
        out, aux = torch.ops.gsplat_metal.forward(means2d, conics, colors, opacities, depths, meta_i32, meta_f32)
        ctx.save_for_backward(means2d, conics, colors, opacities, depths, meta_i32, meta_f32, aux)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        means2d, conics, colors, opacities, depths, meta_i32, meta_f32, aux = ctx.saved_tensors
        g_means2d, g_conics, g_colors, g_opacities, g_depths = torch.ops.gsplat_metal.backward(
            grad_out.contiguous(),
            means2d,
            conics,
            colors,
            opacities,
            depths,
            meta_i32,
            meta_f32,
            aux,
        )
        return g_means2d, g_conics, g_colors, g_opacities, g_depths, None, None


def _make_meta(config: RasterConfig, device: torch.device):
    tiles_y = (config.height + config.tile_size - 1) // config.tile_size
    tiles_x = (config.width + config.tile_size - 1) // config.tile_size
    meta_i32 = torch.tensor(
        [
            config.height,
            config.width,
            tiles_y,
            tiles_x,
            config.tile_size,
            0,  # patched per-call with G
            tiles_y * tiles_x,
            config.chunk_size,
        ],
        device=device,
        dtype=torch.int32,
    )
    meta_f32 = torch.tensor(
        [
            config.alpha_threshold,
            config.transmittance_threshold,
            config.background[0],
            config.background[1],
            config.background[2],
            1e-8,
        ],
        device=device,
        dtype=torch.float32,
    )
    return meta_i32, meta_f32


def rasterize_projected_gaussians(
    means2d: Tensor,
    conics: Tensor,
    colors: Tensor,
    opacities: Tensor,
    depths: Tensor,
    config: RasterConfig,
) -> Tensor:
    if means2d.ndim != 2 or means2d.shape[-1] != 2:
        raise ValueError("means2d must have shape [G,2]")
    if conics.ndim != 2 or conics.shape[-1] != 3:
        raise ValueError("conics must have shape [G,3]")
    if colors.ndim != 2 or colors.shape[-1] != 3:
        raise ValueError("colors must have shape [G,3]")
    if opacities.ndim != 1:
        raise ValueError("opacities must have shape [G]")
    if depths.ndim != 1:
        raise ValueError("depths must have shape [G]")

    G = means2d.shape[0]
    device = means2d.device
    meta_i32, meta_f32 = _make_meta(config, device)
    meta_i32 = meta_i32.clone()
    meta_i32[5] = G

    return _RasterizeProjectedGaussians.apply(
        means2d.contiguous(),
        conics.contiguous(),
        colors.contiguous(),
        opacities.contiguous(),
        depths.contiguous(),
        meta_i32,
        meta_f32,
    )


class ProjectedGaussianRasterizer(torch.nn.Module):
    def __init__(self, config: RasterConfig):
        super().__init__()
        self.config = config

    def forward(self, means2d: Tensor, conics: Tensor, colors: Tensor, opacities: Tensor, depths: Tensor) -> Tensor:
        return rasterize_projected_gaussians(means2d, conics, colors, opacities, depths, self.config)
