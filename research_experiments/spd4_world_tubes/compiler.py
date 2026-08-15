"""Exact affine compilation of SPD(4) world atoms to STAR UVT traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from .model import AffineRayGauge, AmplitudeConvention, WorldAtomBatch, _require_float64


STAROpacityMapping = Literal["peak_preserving", "thin_fiber_optical_depth"]


def pack_symmetric_3x3(matrix: Tensor) -> Tensor:
    """Pack ``[uu, uv, ut, vv, vt, tt]`` for the existing UVT ABI."""

    if matrix.ndim != 3 or matrix.shape[-2:] != (3, 3):
        raise ValueError("matrix must have shape [N,3,3]")
    return torch.stack(
        (
            matrix[:, 0, 0],
            matrix[:, 0, 1],
            matrix[:, 0, 2],
            matrix[:, 1, 1],
            matrix[:, 1, 2],
            matrix[:, 2, 2],
        ),
        dim=-1,
    )


def unpack_symmetric_3x3(packed: Tensor) -> Tensor:
    if packed.ndim != 2 or packed.shape[-1] != 6:
        raise ValueError("packed must have shape [N,6]")
    q00, q01, q02, q11, q12, q22 = packed.unbind(dim=-1)
    return torch.stack(
        (
            torch.stack((q00, q01, q02), dim=-1),
            torch.stack((q01, q11, q12), dim=-1),
            torch.stack((q02, q12, q22), dim=-1),
        ),
        dim=-2,
    )


@dataclass(frozen=True)
class UVTTubesAdapter:
    """The six fields accepted by the existing ``uvt_tubes`` renderer.

    ``opacity_mapping`` is reference metadata and is not part of the legacy
    six-tensor tuple.
    """

    ma: Tensor
    q_uvt: Tensor
    depth0: Tensor
    depth_beta: Tensor
    opacity: Tensor
    color: Tensor
    opacity_mapping: STAROpacityMapping

    def as_tuple(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        return (
            self.ma,
            self.q_uvt,
            self.depth0,
            self.depth_beta,
            self.opacity,
            self.color,
        )

    def as_legacy_float32(self) -> "UVTTubesAdapter":
        """Explicit lowering for the production ABI (reference stays float64)."""

        return UVTTubesAdapter(
            *(value.to(torch.float32).contiguous() for value in self.as_tuple()),
            opacity_mapping=self.opacity_mapping,
        )


@dataclass(frozen=True)
class FiberTrace:
    """Lossless affine trace plus the legacy renderer projection."""

    ma: Tensor
    q_uvt_dense: Tensor
    q_uvt: Tensor
    depth0: Tensor
    depth_beta: Tensor
    depth_variance: Tensor
    peak_to_fiber_scale: Tensor
    peak_density_amplitude: Tensor
    fiber_integrated_amplitude: Tensor
    color: Tensor
    pushed_mean_uvdt: Tensor
    pushed_covariance_uvdt: Tensor
    gauge: AffineRayGauge
    source_amplitude_convention: AmplitudeConvention

    def to_uvt_tubes(
        self,
        *,
        opacity_mapping: STAROpacityMapping = "peak_preserving",
    ) -> UVTTubesAdapter:
        """Lower geometry exactly and opacity under an explicit STAR convention.

        ``peak_preserving`` emits the maximum joint density along the depth
        fiber and matches STAR's factorized ``opacity * exp(-q/2)`` splat
        convention. ``thin_fiber_optical_depth`` emits the integrated optical
        depth coefficient; STAR then approximates ``1-exp(-tau)`` by ``tau``.
        """

        if opacity_mapping == "peak_preserving":
            opacity = self.peak_density_amplitude
        elif opacity_mapping == "thin_fiber_optical_depth":
            opacity = self.fiber_integrated_amplitude
        else:
            raise ValueError(f"unsupported STAR opacity mapping {opacity_mapping!r}")

        return UVTTubesAdapter(
            ma=self.ma,
            q_uvt=self.q_uvt,
            depth0=self.depth0,
            depth_beta=self.depth_beta,
            opacity=opacity,
            color=self.color,
            opacity_mapping=opacity_mapping,
        )


def pushforward_world_atoms(atoms: WorldAtomBatch, gauge: AffineRayGauge) -> FiberTrace:
    """Push an SPD(4) atom through an affine gauge without approximation.

    Coordinates are partitioned as ``a=(u,v,t)`` and fiber coordinate
    ``d=depth``.  The returned marginal and conditional law satisfy

    ``q_joint = q_marginal(a) + (d - E[d|a])**2 / Var[d|a]``.
    """

    matrix = gauge.gauge_from_world
    pushed_mean = atoms.mean_xyzt @ matrix.T + gauge.gauge_offset
    pushed_covariance = (
        matrix[None, :, :]
        @ atoms.covariance_xyzt
        @ matrix.T[None, :, :]
    )
    pushed_covariance = 0.5 * (
        pushed_covariance + pushed_covariance.transpose(-1, -2)
    )

    a_index = torch.tensor((0, 1, 3), dtype=torch.long, device=pushed_mean.device)
    ma = pushed_mean.index_select(-1, a_index)
    marginal_covariance = (
        pushed_covariance.index_select(-2, a_index).index_select(-1, a_index)
    )
    marginal_cholesky = torch.linalg.cholesky(marginal_covariance)
    marginal_precision = torch.cholesky_inverse(marginal_cholesky)
    covariance_a_depth = pushed_covariance.index_select(-2, a_index)[:, :, 2]
    depth_beta = torch.cholesky_solve(
        covariance_a_depth[:, :, None],
        marginal_cholesky,
    )[:, :, 0]
    depth_variance = pushed_covariance[:, 2, 2] - (
        covariance_a_depth * depth_beta
    ).sum(dim=-1)
    if not (depth_variance > 0.0).all():
        raise ValueError("affine pushforward produced nonpositive conditional depth variance")

    peak_to_fiber_scale = (
        gauge.fiber_measure_scale
        * torch.sqrt(2.0 * torch.pi * depth_variance)
    )
    if atoms.amplitude_convention == "peak_density":
        peak_density_amplitude = atoms.amplitude
        fiber_integrated_amplitude = atoms.amplitude * peak_to_fiber_scale
    else:
        fiber_integrated_amplitude = atoms.amplitude
        peak_density_amplitude = atoms.amplitude / peak_to_fiber_scale

    return FiberTrace(
        ma=ma,
        q_uvt_dense=marginal_precision,
        q_uvt=pack_symmetric_3x3(marginal_precision),
        depth0=pushed_mean[:, 2],
        depth_beta=depth_beta,
        depth_variance=depth_variance,
        peak_to_fiber_scale=peak_to_fiber_scale,
        peak_density_amplitude=peak_density_amplitude,
        fiber_integrated_amplitude=fiber_integrated_amplitude,
        color=atoms.color,
        pushed_mean_uvdt=pushed_mean,
        pushed_covariance_uvdt=pushed_covariance,
        gauge=gauge,
        source_amplitude_convention=atoms.amplitude_convention,
    )


def affine_box_extrema(
    intercept: Tensor,
    slope: Tensor,
    box_lower: Tensor,
    box_upper: Tensor,
) -> tuple[Tensor, Tensor]:
    """Exact extrema of an affine function on an axis-aligned box."""

    if slope.shape[-1] != 3 or intercept.shape != slope.shape[:-1]:
        raise ValueError("intercept/slope must have shapes [...] and [...,3]")
    _require_float64("box_lower", box_lower, (3,))
    _require_float64("box_upper", box_upper, (3,))
    if not (box_lower <= box_upper).all():
        raise ValueError("box_lower must not exceed box_upper")
    minimum_point = torch.where(slope >= 0.0, box_lower, box_upper)
    maximum_point = torch.where(slope >= 0.0, box_upper, box_lower)
    return (
        intercept + (slope * minimum_point).sum(dim=-1),
        intercept + (slope * maximum_point).sum(dim=-1),
    )


@dataclass(frozen=True)
class ConfidenceBandOrderCertificate:
    """Exact local affine confidence-band ordering over one UVT box."""

    minimum_band_gap: Tensor
    certified_before: Tensor
    ambiguous: Tensor
    proposed_order: Tensor
    proposed_order_certified: Tensor
    box_lower: Tensor
    box_upper: Tensor
    sigma_multiplier: float


def certify_confidence_band_order(
    trace: FiberTrace,
    box_lower: Tensor,
    box_upper: Tensor,
    *,
    sigma_multiplier: float = 3.0,
    fit_error: Tensor | None = None,
    proposed_order: Tensor | None = None,
    minimum_gap: float = 0.0,
) -> ConfidenceBandOrderCertificate:
    """Certify front-to-back order using exact affine box extrema.

    ``minimum_band_gap[i,j]`` is the exact minimum, over the whole box, of
    ``lower_band_j - upper_band_i``.  A positive value certifies ``i`` before
    ``j`` everywhere in the box.
    """

    count = trace.ma.shape[0]
    _require_float64("box_lower", box_lower, (3,))
    _require_float64("box_upper", box_upper, (3,))
    if sigma_multiplier < 0.0:
        raise ValueError("sigma_multiplier must be nonnegative")
    if fit_error is None:
        fit_error = torch.zeros_like(trace.depth_variance)
    _require_float64("fit_error", fit_error, (count,))
    if not (fit_error >= 0.0).all():
        raise ValueError("fit_error must be nonnegative")

    mean_intercept = trace.depth0 - (trace.depth_beta * trace.ma).sum(dim=-1)
    pair_intercept = mean_intercept[None, :] - mean_intercept[:, None]
    pair_slope = trace.depth_beta[None, :, :] - trace.depth_beta[:, None, :]
    radius = sigma_multiplier * torch.sqrt(trace.depth_variance) + fit_error
    pair_intercept = pair_intercept - radius[:, None] - radius[None, :]
    minimum_band_gap, _ = affine_box_extrema(
        pair_intercept,
        pair_slope,
        box_lower,
        box_upper,
    )
    eye = torch.eye(count, dtype=torch.bool, device=trace.ma.device)
    certified_before = (minimum_band_gap > minimum_gap) & ~eye
    ambiguous = ~(certified_before | certified_before.T) & ~eye

    if proposed_order is None:
        center = 0.5 * (box_lower + box_upper)
        center_depth = mean_intercept + (trace.depth_beta * center).sum(dim=-1)
        proposed_order = torch.argsort(center_depth, stable=True)
    if proposed_order.dtype != torch.long or proposed_order.shape != (count,):
        raise ValueError("proposed_order must be int64 with shape [N]")
    if proposed_order.device != trace.ma.device:
        raise ValueError("proposed_order must be on the trace device")
    if not torch.equal(
        torch.sort(proposed_order).values,
        torch.arange(count, dtype=torch.long, device=trace.ma.device),
    ):
        raise ValueError("proposed_order must be a permutation of range(N)")
    if count <= 1:
        proposed_order_certified = torch.ones(
            (), dtype=torch.bool, device=trace.ma.device
        )
    else:
        adjacent_front = proposed_order[:-1]
        adjacent_back = proposed_order[1:]
        proposed_order_certified = certified_before[
            adjacent_front, adjacent_back
        ].all()

    return ConfidenceBandOrderCertificate(
        minimum_band_gap=minimum_band_gap,
        certified_before=certified_before,
        ambiguous=ambiguous,
        proposed_order=proposed_order,
        proposed_order_certified=proposed_order_certified,
        box_lower=box_lower,
        box_upper=box_upper,
        sigma_multiplier=sigma_multiplier,
    )
