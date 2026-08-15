"""Depth-retaining optical transfer for projected SPD(4) atoms.

Unlike peak-splat compositing, this reference keeps each atom's conditional
Gaussian depth profile and integrates the *combined* extinction/emission field
along the camera fiber.  Differently colored overlapping atoms therefore do
not require an arbitrary mean-depth ordering.

The quadrature bounds are a deliberately piecewise compiler decision:
``mu +/- sigma_extent * sqrt(variance)``.  They are detached from autograd;
the integrand and all source parameters remain differentiable inside the
certified bound.  With six-sigma bounds the omitted Gaussian mass is below
two parts per billion per tail.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _packed_precision_matrix(q_uvt: Tensor) -> Tensor:
    """Expand STAR's ``[uu,uv,ut,vv,vt,tt]`` precision storage."""

    row0 = torch.stack((q_uvt[:, 0], q_uvt[:, 1], q_uvt[:, 2]), dim=1)
    row1 = torch.stack((q_uvt[:, 1], q_uvt[:, 3], q_uvt[:, 4]), dim=1)
    row2 = torch.stack((q_uvt[:, 2], q_uvt[:, 4], q_uvt[:, 5]), dim=1)
    return torch.stack((row0, row1, row2), dim=1)


def _validate(
    ma: Tensor,
    q_uvt: Tensor,
    depth0: Tensor,
    depth_beta: Tensor,
    depth_variance: Tensor,
    optical_thickness: Tensor,
    color: Tensor,
    times: Tensor,
    *,
    height: int,
    width: int,
    depth_samples: int,
    sigma_extent: float,
) -> None:
    atom_count = int(ma.shape[0])
    if ma.shape != (atom_count, 3):
        raise ValueError("ma must have shape [N,3]")
    if q_uvt.shape != (atom_count, 6):
        raise ValueError("q_uvt must have shape [N,6]")
    if depth0.shape != (atom_count,):
        raise ValueError("depth0 must have shape [N]")
    if depth_beta.shape != (atom_count, 3):
        raise ValueError("depth_beta must have shape [N,3]")
    if depth_variance.shape != (atom_count,):
        raise ValueError("depth_variance must have shape [N]")
    if optical_thickness.shape != (atom_count,):
        raise ValueError("optical_thickness must have shape [N]")
    if color.shape != (atom_count, 3):
        raise ValueError("color must have shape [N,3]")
    if times.ndim != 1 or times.numel() == 0:
        raise ValueError("times must have shape [F] with F > 0")
    tensors = (
        ma,
        q_uvt,
        depth0,
        depth_beta,
        depth_variance,
        optical_thickness,
        color,
        times,
    )
    if any(value.device != ma.device for value in tensors):
        raise ValueError("all retained-fiber inputs must share one device")
    if any(value.dtype not in (torch.float32, torch.float64) for value in tensors):
        raise ValueError("retained-fiber inputs must be float32 or float64")
    if any(value.dtype != ma.dtype for value in tensors):
        raise ValueError("all retained-fiber inputs must share one dtype")
    if not all(bool(torch.isfinite(value).all().detach()) for value in tensors):
        raise ValueError("retained-fiber inputs must be finite")
    if bool(torch.any(depth_variance <= 0.0).detach()):
        raise ValueError("depth_variance must be strictly positive")
    if bool(torch.any(optical_thickness < 0.0).detach()):
        raise ValueError("optical_thickness must be nonnegative")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if depth_samples <= 0:
        raise ValueError("depth_samples must be positive")
    if not math.isfinite(sigma_extent) or sigma_extent <= 0.0:
        raise ValueError("sigma_extent must be finite and positive")


def render_retained_fiber_reference(
    ma: Tensor,
    q_uvt: Tensor,
    depth0: Tensor,
    depth_beta: Tensor,
    depth_variance: Tensor,
    optical_thickness: Tensor,
    color: Tensor,
    times: Tensor,
    *,
    height: int,
    width: int,
    depth_samples: int = 64,
    sigma_extent: float = 6.0,
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Tensor:
    """Render ``[F,H,W,3]`` with midpoint integration along retained depth."""

    _validate(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        depth_variance,
        optical_thickness,
        color,
        times,
        height=height,
        width=width,
        depth_samples=depth_samples,
        sigma_extent=sigma_extent,
    )
    q_matrix = _packed_precision_matrix(q_uvt)
    bg = torch.tensor(background, dtype=ma.dtype, device=ma.device)
    normalizer = torch.rsqrt(
        2.0
        * torch.as_tensor(math.pi, dtype=ma.dtype, device=ma.device)
        * depth_variance
    )
    std = torch.sqrt(depth_variance)
    frames: list[Tensor] = []
    for time in times:
        rows: list[Tensor] = []
        for y in range(height):
            pixels: list[Tensor] = []
            for x in range(width):
                coordinate = torch.stack(
                    (
                        torch.as_tensor(x + 0.5, dtype=ma.dtype, device=ma.device),
                        torch.as_tensor(y + 0.5, dtype=ma.dtype, device=ma.device),
                        time,
                    )
                )
                delta = coordinate.unsqueeze(0) - ma
                qv = torch.einsum("ni,nij,nj->n", delta, q_matrix, delta)
                fiber_tau = optical_thickness * torch.exp(-0.5 * qv)
                means = depth0 + torch.sum(delta * depth_beta, dim=1)
                # Bounds are compiled control flow, not a differentiable
                # substitute for the Gaussian tails.
                z_min = torch.min(means - float(sigma_extent) * std).detach()
                z_max = torch.max(means + float(sigma_extent) * std).detach()
                dz = (z_max - z_min) / float(depth_samples)
                transmittance = torch.ones((), dtype=ma.dtype, device=ma.device)
                accum = torch.zeros((3,), dtype=ma.dtype, device=ma.device)
                for sample_index in range(depth_samples):
                    z = z_min + (float(sample_index) + 0.5) * dz
                    centered = z - means
                    profile = normalizer * torch.exp(
                        -0.5 * centered.square() / depth_variance
                    )
                    density = fiber_tau * profile
                    total_density = density.sum()
                    beta = torch.exp(-total_density * dz)
                    alpha = 1.0 - beta
                    source = torch.sum(density[:, None] * color, dim=0) / total_density.clamp_min(
                        torch.finfo(ma.dtype).tiny
                    )
                    accum = accum + transmittance * alpha * source
                    transmittance = transmittance * beta
                pixels.append(accum + transmittance * bg)
            rows.append(torch.stack(pixels))
        frames.append(torch.stack(rows))
    return torch.stack(frames)


__all__ = ["render_retained_fiber_reference"]
