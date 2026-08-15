"""Variance-certified STAR / retained-fiber hybrid rendering.

The fast branch uses the existing mean-ordered STAR Beer--Lambert renderer.
For each tile-time cell, a detached compiler kernel proves that every pair of
potentially overlapping conditional-depth confidence bands has one fixed
order.  Cells without such a certificate are rendered by the depth-resolved
retained-fiber Metal VJP instead.

The decision is intentionally nondifferentiable, just like tile membership or
visibility-order compilation.  Gradients flow through whichever physical
branch rendered the pixel.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .retained_fiber_metal import (
    RetainedFiberMetal,
    RetainedFiberTileCertificate,
    render_retained_fiber_metal,
)


@dataclass(frozen=True)
class HybridRetainedFiberRender:
    rgb: Tensor
    certificate: RetainedFiberTileCertificate

    @property
    def fallback_mask(self) -> Tensor:
        return self.certificate.fallback_mask


_CERTIFIER = RetainedFiberMetal()


def render_variance_certified_hybrid_metal(
    *,
    fast_rgb: Tensor,
    ma: Tensor,
    q_uvt: Tensor,
    depth0: Tensor,
    depth_beta: Tensor,
    depth_variance: Tensor,
    optical_thickness: Tensor,
    color: Tensor,
    times: Tensor,
    height: int,
    width: int,
    tile_x: int,
    tile_y: int,
    tile_t: int,
    alpha_threshold: float,
    max_alpha: float = 1.0,
    depth_samples: int = 48,
    sigma_extent: float = 6.0,
    certificate_sigma: float = 6.0,
    required_gap: float = 0.0,
    depth_fit_error: Tensor | None = None,
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> HybridRetainedFiberRender:
    """Render certified tiles fast and ambiguous tiles with retained depth."""

    frame_count = int(times.numel())
    expected_rgb = (frame_count, int(height), int(width), 3)
    if tuple(fast_rgb.shape) != expected_rgb:
        raise ValueError(f"fast_rgb must have shape {expected_rgb}")
    if (
        fast_rgb.device.type != "mps"
        or fast_rgb.dtype != torch.float32
        or not fast_rgb.is_contiguous()
    ):
        raise ValueError("fast_rgb must be contiguous MPS float32")
    if certificate_sigma < sigma_extent:
        raise ValueError(
            "certificate_sigma must be at least sigma_extent so the fast "
            "order covers every depth sample retained by the fallback"
        )
    if float(max_alpha) != 1.0:
        raise ValueError(
            "variance-certified hybrid Beer rendering requires max_alpha=1 "
            "so the fast branch and retained optical transfer have identical "
            "single-atom transmittance"
        )

    with torch.no_grad():
        certificate = _CERTIFIER.certify_tiles(
            ma.detach().contiguous(),
            q_uvt.detach().contiguous(),
            depth0.detach().contiguous(),
            depth_beta.detach().contiguous(),
            depth_variance.detach().contiguous(),
            optical_thickness.detach().contiguous(),
            frames=frame_count,
            height=int(height),
            width=int(width),
            tile_x=int(tile_x),
            tile_y=int(tile_y),
            tile_t=int(tile_t),
            alpha_threshold=float(alpha_threshold),
            sigma_multiplier=float(certificate_sigma),
            required_gap=float(required_gap),
            depth_fit_error=(
                None
                if depth_fit_error is None
                else depth_fit_error.detach().contiguous()
            ),
        )
    retained_rgb = render_retained_fiber_metal(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        depth_variance,
        optical_thickness,
        color,
        times,
        height=int(height),
        width=int(width),
        depth_samples=int(depth_samples),
        sigma_extent=float(sigma_extent),
        background=background,
        fallback_mask=certificate.fallback_mask,
        alpha_threshold=float(alpha_threshold),
    )
    rgb = torch.where(
        certificate.fallback_mask[..., None].bool(),
        retained_rgb,
        fast_rgb,
    )
    return HybridRetainedFiberRender(rgb=rgb, certificate=certificate)


__all__ = [
    "HybridRetainedFiberRender",
    "render_variance_certified_hybrid_metal",
]
