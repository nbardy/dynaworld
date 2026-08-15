"""Native SPD(4) world atoms and affine ray gauges.

This module is deliberately float64-only reference code.  It is a source-side
oracle for the existing float32 STAR ``uvt_tubes`` ABI, not a production
renderer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor


AmplitudeConvention = Literal["peak_density", "fiber_integrated"]


def _require_float64(name: str, value: Tensor, shape: tuple[int, ...] | None = None) -> None:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != torch.float64:
        raise ValueError(f"{name} must be float64, got {value.dtype}")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class BlockCholeskySPD4:
    """Lossless conditional block chart for an SPD(4) covariance.

    With ``C = spatial_cholesky @ spatial_cholesky.T``,
    ``c = exp(2 * log_temporal_scale)`` and ``v = space_time_tilt``,

    ``Sigma = [[C + c vv.T, cv], [cv.T, c]]``.

    The chart is lossless: every strict SPD(4) matrix has exactly one such
    representation when the spatial factor is the positive-diagonal Cholesky
    factor.
    """

    spatial_cholesky: Tensor
    space_time_tilt: Tensor
    log_temporal_scale: Tensor

    def covariance(self) -> Tensor:
        return covariance_from_block_cholesky(
            self.spatial_cholesky,
            self.space_time_tilt,
            self.log_temporal_scale,
        )


def covariance_from_block_cholesky(
    spatial_cholesky: Tensor,
    space_time_tilt: Tensor,
    log_temporal_scale: Tensor,
) -> Tensor:
    """Construct every strict SPD(4) covariance without dropping a degree."""

    if spatial_cholesky.ndim != 3 or spatial_cholesky.shape[-2:] != (3, 3):
        raise ValueError("spatial_cholesky must have shape [N,3,3]")
    count = spatial_cholesky.shape[0]
    _require_float64("spatial_cholesky", spatial_cholesky)
    _require_float64("space_time_tilt", space_time_tilt, (count, 3))
    _require_float64("log_temporal_scale", log_temporal_scale, (count,))
    if not torch.allclose(
        spatial_cholesky,
        torch.tril(spatial_cholesky),
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("spatial_cholesky must be lower triangular")
    if not (torch.diagonal(spatial_cholesky, dim1=-2, dim2=-1) > 0.0).all():
        raise ValueError("spatial_cholesky must have a positive diagonal")

    spatial_conditional = spatial_cholesky @ spatial_cholesky.transpose(-1, -2)
    temporal_variance = torch.exp(2.0 * log_temporal_scale)
    cross = temporal_variance[:, None] * space_time_tilt
    spatial_joint = spatial_conditional + (
        temporal_variance[:, None, None]
        * space_time_tilt[:, :, None]
        * space_time_tilt[:, None, :]
    )
    top = torch.cat((spatial_joint, cross[:, :, None]), dim=-1)
    bottom = torch.cat((cross, temporal_variance[:, None]), dim=-1)[:, None, :]
    return torch.cat((top, bottom), dim=-2)


def block_cholesky_from_covariance(covariance_xyzt: Tensor) -> BlockCholeskySPD4:
    """Invert :func:`covariance_from_block_cholesky` by a Schur complement."""

    if covariance_xyzt.ndim != 3 or covariance_xyzt.shape[-2:] != (4, 4):
        raise ValueError("covariance_xyzt must have shape [N,4,4]")
    _require_float64("covariance_xyzt", covariance_xyzt)
    if not torch.allclose(
        covariance_xyzt,
        covariance_xyzt.transpose(-1, -2),
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise ValueError("covariance_xyzt must be symmetric")
    if not (torch.linalg.cholesky_ex(covariance_xyzt).info == 0).all():
        raise ValueError("covariance_xyzt must be strictly positive definite")

    temporal_variance = covariance_xyzt[:, 3, 3]
    tilt = covariance_xyzt[:, :3, 3] / temporal_variance[:, None]
    conditional_spatial = covariance_xyzt[:, :3, :3] - (
        temporal_variance[:, None, None] * tilt[:, :, None] * tilt[:, None, :]
    )
    return BlockCholeskySPD4(
        spatial_cholesky=torch.linalg.cholesky(conditional_spatial),
        space_time_tilt=tilt,
        log_temporal_scale=0.5 * torch.log(temporal_variance),
    )


@dataclass(frozen=True)
class WorldAtomBatch:
    """A batch of finite-mass Gaussian atoms native to world ``(x,y,z,t)``."""

    mean_xyzt: Tensor
    covariance_xyzt: Tensor
    amplitude: Tensor
    color: Tensor
    amplitude_convention: AmplitudeConvention = "peak_density"

    def __post_init__(self) -> None:
        if self.mean_xyzt.ndim != 2 or self.mean_xyzt.shape[-1] != 4:
            raise ValueError("mean_xyzt must have shape [N,4]")
        count = self.mean_xyzt.shape[0]
        _require_float64("mean_xyzt", self.mean_xyzt)
        _require_float64("covariance_xyzt", self.covariance_xyzt, (count, 4, 4))
        _require_float64("amplitude", self.amplitude, (count,))
        _require_float64("color", self.color, (count, 3))
        if self.amplitude_convention not in {"peak_density", "fiber_integrated"}:
            raise ValueError(
                "amplitude_convention must be 'peak_density' or 'fiber_integrated'"
            )
        if not torch.allclose(
            self.covariance_xyzt,
            self.covariance_xyzt.transpose(-1, -2),
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            raise ValueError("covariance_xyzt must be symmetric")
        if not (torch.linalg.cholesky_ex(self.covariance_xyzt).info == 0).all():
            raise ValueError("covariance_xyzt must be strictly positive definite")
        if not (self.amplitude >= 0.0).all():
            raise ValueError("amplitude must be nonnegative")

    @classmethod
    def from_block_cholesky(
        cls,
        *,
        mean_xyzt: Tensor,
        spatial_cholesky: Tensor,
        space_time_tilt: Tensor,
        log_temporal_scale: Tensor,
        amplitude: Tensor,
        color: Tensor,
        amplitude_convention: AmplitudeConvention = "peak_density",
    ) -> "WorldAtomBatch":
        return cls(
            mean_xyzt=mean_xyzt,
            covariance_xyzt=covariance_from_block_cholesky(
                spatial_cholesky,
                space_time_tilt,
                log_temporal_scale,
            ),
            amplitude=amplitude,
            color=color,
            amplitude_convention=amplitude_convention,
        )


@dataclass(frozen=True)
class AffineRayGauge:
    """An invertible affine world-to-``(u,v,depth,t)`` camera gauge.

    ``gauge = world @ gauge_from_world.T + gauge_offset``.
    ``fiber_measure_scale`` is ``ds_world / ddepth`` and makes line integrals
    invariant to a mere rescaling of the depth coordinate.
    """

    gauge_from_world: Tensor
    gauge_offset: Tensor
    fiber_measure_scale: Tensor

    def __post_init__(self) -> None:
        _require_float64("gauge_from_world", self.gauge_from_world, (4, 4))
        _require_float64("gauge_offset", self.gauge_offset, (4,))
        _require_float64("fiber_measure_scale", self.fiber_measure_scale, ())
        if torch.abs(torch.linalg.det(self.gauge_from_world)) <= 1.0e-14:
            raise ValueError("gauge_from_world must be invertible")
        if self.fiber_measure_scale <= 0.0:
            raise ValueError("fiber_measure_scale must be positive")

    @property
    def world_from_gauge(self) -> Tensor:
        return torch.linalg.inv(self.gauge_from_world)

    @property
    def world_origin(self) -> Tensor:
        return -(self.world_from_gauge @ self.gauge_offset)

    def to_gauge(self, world_xyzt: Tensor) -> Tensor:
        if world_xyzt.shape[-1] != 4:
            raise ValueError("world_xyzt must end in four coordinates")
        return world_xyzt @ self.gauge_from_world.T + self.gauge_offset

    def to_world(self, gauge_uvdt: Tensor) -> Tensor:
        if gauge_uvdt.shape[-1] != 4:
            raise ValueError("gauge_uvdt must end in four coordinates")
        return (gauge_uvdt - self.gauge_offset) @ self.world_from_gauge.T

    def world_from_uvt_depth(self, uvt: Tensor, depth: Tensor) -> Tensor:
        """Evaluate the exact affine ray bundle at ``(u,v,t,depth)``."""

        if uvt.shape[-1] != 3 or depth.shape != uvt.shape[:-1]:
            raise ValueError("uvt must have shape [...,3] and depth shape [...]")
        gauge = torch.stack(
            (uvt[..., 0], uvt[..., 1], depth, uvt[..., 2]),
            dim=-1,
        )
        return self.to_world(gauge)

    @classmethod
    def from_ray_bundle(
        cls,
        *,
        world_origin: Tensor,
        world_uvt_basis: Tensor,
        world_depth_direction: Tensor,
    ) -> "AffineRayGauge":
        """Build the gauge from ``Z = b + A (u,v,t) + d s`` exactly."""

        _require_float64("world_origin", world_origin, (4,))
        _require_float64("world_uvt_basis", world_uvt_basis, (4, 3))
        _require_float64("world_depth_direction", world_depth_direction, (4,))
        if torch.abs(world_depth_direction[3]) > 1.0e-12:
            raise ValueError(
                "a camera depth fiber must remain at fixed physical world time"
            )
        world_from_gauge = torch.stack(
            (
                world_uvt_basis[:, 0],
                world_uvt_basis[:, 1],
                world_depth_direction,
                world_uvt_basis[:, 2],
            ),
            dim=-1,
        )
        gauge_from_world = torch.linalg.inv(world_from_gauge)
        spatial_measure = torch.linalg.vector_norm(world_depth_direction[:3])
        return cls(
            gauge_from_world=gauge_from_world,
            gauge_offset=-(gauge_from_world @ world_origin),
            fiber_measure_scale=spatial_measure,
        )
