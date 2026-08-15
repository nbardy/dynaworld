"""Capacity-bounded shared depth flow for the connection reference.

The flow is deliberately a small global Chebyshev field rather than a table
indexed by requested frame, pixel, or ray.  It can therefore explain coherent
motion without silently storing the transfer that the connection is supposed
to simplify.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class FlowDomain:
    """Closed time/depth domain used to normalize the polynomial field."""

    t_min: float
    t_max: float
    z_min: float
    z_max: float

    def validate(self) -> None:
        if not self.t_max > self.t_min:
            raise ValueError("flow time domain must have positive extent")
        if not self.z_max > self.z_min:
            raise ValueError("flow depth domain must have positive extent")


@dataclass(frozen=True)
class FlowCapacityReport:
    """Explicit capacity receipt for a shared flow instance."""

    temporal_degree: int
    depth_degree: int
    coefficient_count: int
    parameter_bytes: int
    reference_temporal_dof: int
    capacity_ratio: float
    within_reference_capacity: bool
    requested_frame_indexed_state_count: int = 0
    ray_indexed_state_count: int = 0


@dataclass(frozen=True)
class FlowEvaluation:
    """Flow value and analytic coordinate derivatives."""

    value: torch.Tensor
    d_dt: torch.Tensor
    d_dz: torch.Tensor
    local_orientation_rate: torch.Tensor


def _chebyshev_basis_and_derivative(
    x: torch.Tensor,
    degree: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``T_0..T_degree`` and derivatives with respect to ``x``."""

    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 0:
        raise ValueError("Chebyshev degree must be a nonnegative integer")
    basis = [torch.ones_like(x)]
    derivative = [torch.zeros_like(x)]
    if degree >= 1:
        basis.append(x)
        derivative.append(torch.ones_like(x))
    for _ in range(2, degree + 1):
        basis.append(2.0 * x * basis[-1] - basis[-2])
        derivative.append(
            2.0 * basis[-2]
            + 2.0 * x * derivative[-1]
            - derivative[-2]
        )
    return torch.stack(basis, dim=-1), torch.stack(derivative, dim=-1)


def _normalize_coordinate(
    value: torch.Tensor,
    lower: float,
    upper: float,
) -> tuple[torch.Tensor, float]:
    scale = 2.0 / (upper - lower)
    return (value - lower) * scale - 1.0, scale


def evaluate_chebyshev_flow_coefficients(
    coefficients: torch.Tensor,
    *,
    domain: FlowDomain,
    maximum_speed: float,
    t: torch.Tensor,
    z: torch.Tensor,
) -> FlowEvaluation:
    """Pure tensor evaluation used by end-to-end forward-mode transforms.

    The public module validates domains before calling this function.  This
    lower-level map deliberately contains no tensor-to-Python branch, so cuts,
    query depths, time, and coefficients can participate in one ``jvp``.
    """

    if coefficients.ndim != 2 or coefficients.dtype not in {
        torch.float32,
        torch.float64,
    }:
        raise ValueError("flow coefficients must be a float matrix")
    if t.dtype != coefficients.dtype or z.dtype != coefficients.dtype:
        raise TypeError("flow coordinates and coefficients must share a dtype")
    if t.device != coefficients.device or z.device != coefficients.device:
        raise ValueError("flow coordinates and coefficients must share a device")
    t, z = torch.broadcast_tensors(t, z)
    t_normalized, t_scale = _normalize_coordinate(
        t,
        domain.t_min,
        domain.t_max,
    )
    z_normalized, z_scale = _normalize_coordinate(
        z,
        domain.z_min,
        domain.z_max,
    )
    t_basis, t_derivative = _chebyshev_basis_and_derivative(
        t_normalized,
        coefficients.shape[0] - 1,
    )
    z_basis, z_derivative = _chebyshev_basis_and_derivative(
        z_normalized,
        coefficients.shape[1] - 1,
    )
    series = torch.einsum("...i,ij,...j->...", t_basis, coefficients, z_basis)
    dseries_dt = t_scale * torch.einsum(
        "...i,ij,...j->...",
        t_derivative,
        coefficients,
        z_basis,
    )
    dseries_dz = z_scale * torch.einsum(
        "...i,ij,...j->...",
        t_basis,
        coefficients,
        z_derivative,
    )
    bounded = torch.tanh(series)
    slope = maximum_speed * (1.0 - bounded.square())
    return FlowEvaluation(
        value=maximum_speed * bounded,
        d_dt=slope * dseries_dt,
        d_dz=slope * dseries_dz,
        local_orientation_rate=slope * dseries_dz,
    )


class SharedChebyshevFlow(nn.Module):
    """One low-dimensional flow shared over a complete ray-time chart.

    ``w(t,z)`` is a bounded scalar depth velocity.  The coefficient tensor has
    no requested-frame or ray axis.  Its shape and byte count are therefore
    fixed before a sampling protocol chooses how many frames or rays to query.
    """

    def __init__(
        self,
        *,
        domain: FlowDomain,
        temporal_degree: int,
        depth_degree: int,
        maximum_speed: float,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        domain.validate()
        if (
            isinstance(temporal_degree, bool)
            or not isinstance(temporal_degree, int)
            or temporal_degree < 0
            or isinstance(depth_degree, bool)
            or not isinstance(depth_degree, int)
            or depth_degree < 0
        ):
            raise ValueError("flow degrees must be nonnegative integers")
        if not maximum_speed > 0.0:
            raise ValueError("maximum_speed must be positive")
        if dtype not in {torch.float32, torch.float64}:
            raise TypeError("shared flow supports float32 or float64")
        self.domain = domain
        self.temporal_degree = temporal_degree
        self.depth_degree = depth_degree
        self.maximum_speed = float(maximum_speed)
        self.coefficients = nn.Parameter(
            torch.zeros(
                (temporal_degree + 1, depth_degree + 1),
                dtype=dtype,
                device=device,
            )
        )

    @property
    def coefficient_count(self) -> int:
        return int(self.coefficients.numel())

    def capacity_report(
        self,
        *,
        reference_temporal_dof: int,
    ) -> FlowCapacityReport:
        if (
            isinstance(reference_temporal_dof, bool)
            or not isinstance(reference_temporal_dof, int)
            or reference_temporal_dof < 1
        ):
            raise ValueError("reference_temporal_dof must be a positive integer")
        ratio = self.coefficient_count / float(reference_temporal_dof)
        return FlowCapacityReport(
            temporal_degree=self.temporal_degree,
            depth_degree=self.depth_degree,
            coefficient_count=self.coefficient_count,
            parameter_bytes=(
                self.coefficient_count * self.coefficients.element_size()
            ),
            reference_temporal_dof=reference_temporal_dof,
            capacity_ratio=ratio,
            within_reference_capacity=(
                self.coefficient_count <= reference_temporal_dof
            ),
        )

    def evaluate(
        self,
        t: torch.Tensor,
        z: torch.Tensor,
    ) -> FlowEvaluation:
        if not isinstance(t, torch.Tensor) or not isinstance(z, torch.Tensor):
            raise TypeError("flow coordinates must be tensors")
        if t.device != self.coefficients.device or z.device != t.device:
            raise ValueError("flow coordinates and coefficients must share a device")
        if t.dtype != self.coefficients.dtype or z.dtype != t.dtype:
            raise TypeError("flow coordinates and coefficients must share a dtype")
        t, z = torch.broadcast_tensors(t, z)
        if not bool(torch.all(torch.isfinite(t))) or not bool(
            torch.all(torch.isfinite(z))
        ):
            raise ValueError("flow coordinates must be finite")
        if not bool(
            torch.all(
                (t >= self.domain.t_min)
                & (t <= self.domain.t_max)
                & (z >= self.domain.z_min)
                & (z <= self.domain.z_max)
            )
        ):
            raise ValueError("flow query lies outside its declared domain")
        t_normalized, t_scale = _normalize_coordinate(
            t,
            self.domain.t_min,
            self.domain.t_max,
        )
        z_normalized, z_scale = _normalize_coordinate(
            z,
            self.domain.z_min,
            self.domain.z_max,
        )
        t_basis, t_derivative = _chebyshev_basis_and_derivative(
            t_normalized,
            self.temporal_degree,
        )
        z_basis, z_derivative = _chebyshev_basis_and_derivative(
            z_normalized,
            self.depth_degree,
        )
        series = torch.einsum(
            "...i,ij,...j->...",
            t_basis,
            self.coefficients,
            z_basis,
        )
        dseries_dt = t_scale * torch.einsum(
            "...i,ij,...j->...",
            t_derivative,
            self.coefficients,
            z_basis,
        )
        dseries_dz = z_scale * torch.einsum(
            "...i,ij,...j->...",
            t_basis,
            self.coefficients,
            z_derivative,
        )
        bounded = torch.tanh(series)
        slope = self.maximum_speed * (1.0 - bounded.square())
        return FlowEvaluation(
            value=self.maximum_speed * bounded,
            d_dt=slope * dseries_dt,
            d_dz=slope * dseries_dz,
            local_orientation_rate=slope * dseries_dz,
        )

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.evaluate(t, z).value

    def minimum_euler_orientation_margin(
        self,
        t: torch.Tensor,
        z: torch.Tensor,
        *,
        delta_t: float,
    ) -> torch.Tensor:
        """Diagnostic ``min(1 + delta_t * partial_z w)`` on supplied probes."""

        if not delta_t > 0.0:
            raise ValueError("delta_t must be positive")
        return torch.amin(1.0 + delta_t * self.evaluate(t, z).d_dz)


__all__ = [
    "FlowCapacityReport",
    "FlowDomain",
    "FlowEvaluation",
    "SharedChebyshevFlow",
    "evaluate_chebyshev_flow_coefficients",
]
