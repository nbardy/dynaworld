"""Dense retained-depth oracle for falsifying mean-sorted UVT rendering."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .compiler import FiberTrace
from .model import _require_float64


def _uvt_matrix(uvt: Tensor) -> Tensor:
    _require_float64("uvt", uvt)
    if uvt.shape == (3,):
        return uvt[None, :]
    if uvt.ndim != 2 or uvt.shape[-1] != 3:
        raise ValueError("uvt must have shape [3] or [P,3]")
    return uvt


def marginal_quadratic(trace: FiberTrace, uvt: Tensor) -> Tensor:
    query = _uvt_matrix(uvt)
    delta = query[:, None, :] - trace.ma[None, :, :]
    return torch.einsum("pni,nij,pnj->pn", delta, trace.q_uvt_dense, delta)


def analytic_fiber_optical_depth(trace: FiberTrace, uvt: Tensor) -> Tensor:
    """Analytic infinite-fiber integral, shape ``[P,N]``."""

    return trace.fiber_integrated_amplitude[None, :] * torch.exp(
        -0.5 * marginal_quadratic(trace, uvt)
    )


def retained_fiber_density(trace: FiberTrace, uvt: Tensor, depth: Tensor) -> Tensor:
    """Evaluate every joint atom on a dense depth grid.

    Returns ``[P,S,N]`` for ``P`` UVT queries, ``S`` depth samples and ``N``
    atoms.  This retains the conditional depth profiles instead of replacing
    them by a mean-depth order.
    """

    query = _uvt_matrix(uvt)
    _require_float64("depth", depth)
    if depth.ndim == 1:
        depth_grid = depth[None, :].expand(query.shape[0], -1)
    elif depth.ndim == 2 and depth.shape[0] == query.shape[0]:
        depth_grid = depth
    else:
        raise ValueError("depth must have shape [S] or [P,S]")

    delta = query[:, None, :] - trace.ma[None, :, :]
    marginal = torch.einsum(
        "pni,nij,pnj->pn",
        delta,
        trace.q_uvt_dense,
        delta,
    )
    conditional_mean = trace.depth0[None, :] + (
        delta * trace.depth_beta[None, :, :]
    ).sum(dim=-1)
    conditional = (
        depth_grid[:, :, None] - conditional_mean[:, None, :]
    ).square() / trace.depth_variance[None, None, :]
    return trace.peak_density_amplitude[None, None, :] * torch.exp(
        -0.5 * (marginal[:, None, :] + conditional)
    )


@dataclass(frozen=True)
class DenseRetainedFiberRender:
    rgb: Tensor
    transmittance: Tensor
    optical_depth: Tensor
    step_optical_depth: Tensor


def dense_retained_fiber_render(
    trace: FiberTrace,
    uvt: Tensor,
    depth_edges: Tensor,
) -> DenseRetainedFiberRender:
    """Midpoint volume-render the retained joint density.

    The reference intentionally favors clarity and convergence under depth-grid
    refinement over speed.  It handles overlapping thick atoms without a
    discrete primitive order and is therefore an oracle for visibility cases
    rejected by the confidence-band certificate.
    """

    query = _uvt_matrix(uvt)
    _require_float64("depth_edges", depth_edges)
    if depth_edges.ndim == 1:
        edges = depth_edges[None, :].expand(query.shape[0], -1)
    elif depth_edges.ndim == 2 and depth_edges.shape[0] == query.shape[0]:
        edges = depth_edges
    else:
        raise ValueError("depth_edges must have shape [S+1] or [P,S+1]")
    if edges.shape[-1] < 2 or not ((edges[:, 1:] - edges[:, :-1]) > 0.0).all():
        raise ValueError("depth_edges must be strictly increasing")

    widths = (edges[:, 1:] - edges[:, :-1]) * trace.gauge.fiber_measure_scale
    midpoint = 0.5 * (edges[:, 1:] + edges[:, :-1])
    density = retained_fiber_density(trace, query, midpoint)
    total_density = density.sum(dim=-1)
    step_optical_depth = total_density * widths
    weighted_color = torch.einsum("psn,nc->psc", density, trace.color)
    mixture_color = torch.where(
        total_density[:, :, None] > 0.0,
        weighted_color / total_density[:, :, None].clamp_min(
            torch.finfo(torch.float64).tiny
        ),
        torch.zeros_like(weighted_color),
    )

    transmittance = torch.ones(
        (query.shape[0],), dtype=torch.float64, device=query.device
    )
    rgb = torch.zeros(
        (query.shape[0], trace.color.shape[-1]),
        dtype=torch.float64,
        device=query.device,
    )
    for step in range(step_optical_depth.shape[1]):
        beta = torch.exp(-step_optical_depth[:, step])
        absorbed = -torch.expm1(-step_optical_depth[:, step])
        rgb = rgb + transmittance[:, None] * absorbed[:, None] * mixture_color[:, step]
        transmittance = transmittance * beta
    return DenseRetainedFiberRender(
        rgb=rgb,
        transmittance=transmittance,
        optical_depth=step_optical_depth.sum(dim=-1),
        step_optical_depth=step_optical_depth,
    )
