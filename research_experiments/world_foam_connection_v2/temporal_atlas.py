"""Equal-family temporal atlases for the three connection ablations.

This module is intentionally renderer-independent.  It provides one adaptive
piecewise-linear family for physical transfer ``U``, unrestricted affine-group
transfer ``U_tilde``, and signed tangent source ``K_F``.  Its certificate is a
dense discrete probe certificate, not a continuous approximation proof.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import torch


class AtlasKind(str, Enum):
    PHYSICAL_U = "physical_u"
    GROUP_U_TILDE = "group_u_tilde"
    SIGNED_K_F = "signed_k_f"


@dataclass(frozen=True)
class AtlasCertificate:
    kind: AtlasKind
    node_count: int
    probe_count: int
    maximum_primal_error: float
    maximum_tangent_error: float | None
    primal_tolerance: float
    tangent_tolerance: float | None
    payload_bytes: int
    probe_grid_only: bool = True
    verified: bool = False


@dataclass(frozen=True)
class LinearTemporalAtlas:
    """One immutable ragged temporal chart with four values per node."""

    kind: AtlasKind
    knots: torch.Tensor
    values: torch.Tensor

    def validate(self) -> None:
        if not isinstance(self.kind, AtlasKind):
            raise TypeError("atlas kind must be AtlasKind")
        if not isinstance(self.knots, torch.Tensor) or not isinstance(
            self.values,
            torch.Tensor,
        ):
            raise TypeError("atlas knots and values must be tensors")
        if self.knots.ndim != 1 or self.knots.numel() < 2:
            raise ValueError("atlas requires at least two scalar knots")
        if self.values.shape != (self.knots.numel(), 4):
            raise ValueError("atlas values must have shape [J,4]")
        if self.values.device != self.knots.device:
            raise ValueError("atlas knots and values must share a device")
        if self.values.dtype != self.knots.dtype:
            raise TypeError("atlas knots and values must share a dtype")
        if self.knots.dtype not in {torch.float32, torch.float64}:
            raise TypeError("atlas supports float32 or float64")
        if not bool(torch.all(torch.isfinite(self.knots))):
            raise ValueError("atlas knots must be finite")
        if not bool(torch.all(torch.isfinite(self.values))):
            raise ValueError("atlas values must be finite")
        if not bool(torch.all(self.knots[1:] > self.knots[:-1])):
            raise ValueError("atlas knots must be strictly increasing")

    @property
    def node_count(self) -> int:
        return int(self.knots.numel())

    @property
    def payload_bytes(self) -> int:
        return int(
            self.knots.numel() * self.knots.element_size()
            + self.values.numel() * self.values.element_size()
        )

    def _segments(
        self,
        query_times: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.validate()
        if not isinstance(query_times, torch.Tensor):
            raise TypeError("atlas query times must be a tensor")
        if query_times.device != self.knots.device:
            raise ValueError("atlas queries must share the atlas device")
        if query_times.dtype != self.knots.dtype:
            raise TypeError("atlas queries must share the atlas dtype")
        if not bool(torch.all(torch.isfinite(query_times))):
            raise ValueError("atlas query times must be finite")
        if not bool(
            torch.all(
                (query_times >= self.knots[0])
                & (query_times <= self.knots[-1])
            )
        ):
            raise ValueError("atlas query lies outside its closed interval")
        flat = query_times.reshape(-1)
        right = torch.searchsorted(self.knots, flat, right=True)
        left = torch.clamp(right - 1, 0, self.node_count - 2)
        right = left + 1
        width = self.knots[right] - self.knots[left]
        alpha = (flat - self.knots[left]) / width
        return left, right, alpha

    def evaluate(self, query_times: torch.Tensor) -> torch.Tensor:
        left, right, alpha = self._segments(query_times)
        flat = (
            (1.0 - alpha[:, None]) * self.values[left]
            + alpha[:, None] * self.values[right]
        )
        return flat.reshape(query_times.shape + (4,))

    def derivative(self, query_times: torch.Tensor) -> torch.Tensor:
        left, right, _ = self._segments(query_times)
        slopes = (self.values[right] - self.values[left]) / (
            self.knots[right] - self.knots[left]
        )[:, None]
        return slopes.reshape(query_times.shape + (4,))

    def integral_from_start(self, query_times: torch.Tensor) -> torch.Tensor:
        """Integrate the piecewise-linear four-vector exactly in time."""

        left, right, alpha = self._segments(query_times)
        widths = self.knots[1:] - self.knots[:-1]
        segment_integrals = 0.5 * widths[:, None] * (
            self.values[:-1] + self.values[1:]
        )
        cumulative = torch.cat(
            (
                torch.zeros_like(segment_integrals[:1]),
                torch.cumsum(segment_integrals, dim=0),
            ),
            dim=0,
        )
        local_width = query_times.reshape(-1) - self.knots[left]
        local_end = (
            (1.0 - alpha[:, None]) * self.values[left]
            + alpha[:, None] * self.values[right]
        )
        local = 0.5 * local_width[:, None] * (
            self.values[left] + local_end
        )
        result = cumulative[left] + local
        return result.reshape(query_times.shape + (4,))


def _kappa_over_one_minus_exp_negative(kappa: torch.Tensor) -> torch.Tensor:
    """Stable ``kappa / (1-exp(-kappa))`` with the analytic identity limit."""

    threshold = torch.finfo(kappa.dtype).eps ** 0.25
    safe = torch.where(
        torch.abs(kappa) > threshold,
        kappa,
        torch.ones_like(kappa),
    )
    exact = safe / (-torch.expm1(-safe))
    square = kappa.square()
    series = 1.0 + 0.5 * kappa + square / 12.0 - square.square() / 720.0
    return torch.where(torch.abs(kappa) > threshold, exact, series)


def _one_minus_exp_negative_over_kappa(kappa: torch.Tensor) -> torch.Tensor:
    """Stable ``(1-exp(-kappa)) / kappa`` with derivatives at zero."""

    threshold = torch.finfo(kappa.dtype).eps ** 0.2
    safe = torch.where(
        torch.abs(kappa) > threshold,
        kappa,
        torch.ones_like(kappa),
    )
    exact = -torch.expm1(-safe) / safe
    square = kappa.square()
    series = (
        1.0
        - 0.5 * kappa
        + square / 6.0
        - square * kappa / 24.0
        + square.square() / 120.0
    )
    return torch.where(torch.abs(kappa) > threshold, exact, series)


def transfer_to_unrestricted_log_chart(transfer: torch.Tensor) -> torch.Tensor:
    """Encode ``(beta,m)`` for any finite affine transfer with ``beta>0``."""

    if not isinstance(transfer, torch.Tensor) or transfer.shape[-1] != 4:
        raise ValueError("transfer must be a tensor with trailing shape 4")
    beta = transfer[..., 0]
    moment = transfer[..., 1:]
    if not bool(torch.all(torch.isfinite(transfer))):
        raise ValueError("transfer must be finite")
    if not bool(torch.all(beta > 0.0)):
        raise ValueError("affine-group transfer requires beta>0")
    kappa = -torch.log(beta)
    ratio = _kappa_over_one_minus_exp_negative(kappa)
    return torch.cat((kappa[..., None], ratio[..., None] * moment), dim=-1)


def unrestricted_log_chart_to_transfer(chart: torch.Tensor) -> torch.Tensor:
    if not isinstance(chart, torch.Tensor) or chart.shape[-1] != 4:
        raise ValueError("chart must be a tensor with trailing shape 4")
    if not bool(torch.all(torch.isfinite(chart))):
        raise ValueError("chart must be finite")
    kappa = chart[..., 0]
    vector = chart[..., 1:]
    beta = torch.exp(-kappa)
    scale = _one_minus_exp_negative_over_kappa(kappa)
    moment = scale[..., None] * vector
    return torch.cat((beta[..., None], moment), dim=-1)


def transfer_to_physical_chart(
    transfer: torch.Tensor,
    *,
    tolerance: float = 1.0e-9,
) -> torch.Tensor:
    chart = transfer_to_unrestricted_log_chart(transfer)
    beta = transfer[..., 0]
    moment = transfer[..., 1:]
    if not bool(torch.all(beta <= 1.0 + tolerance)):
        raise ValueError("physical transfer requires beta<=1")
    if not bool(torch.all(moment >= -tolerance)):
        raise ValueError("physical transfer requires nonnegative moment")
    if not bool(torch.all(moment <= (1.0 - beta)[..., None] + tolerance)):
        raise ValueError("physical transfer exceeds its bounded-radiance cone")
    kappa = chart[..., :1]
    vector = chart[..., 1:]
    if not bool(torch.all(vector >= -tolerance)) or not bool(
        torch.all(vector <= kappa + tolerance)
    ):
        raise ValueError("physical log chart left the affine Lie cone")
    return chart


def physical_chart_to_transfer(
    chart: torch.Tensor,
    *,
    tolerance: float = 1.0e-9,
) -> torch.Tensor:
    kappa = chart[..., :1]
    vector = chart[..., 1:]
    if not bool(torch.all(kappa >= -tolerance)):
        raise ValueError("physical chart requires kappa>=0")
    if not bool(torch.all(vector >= -tolerance)) or not bool(
        torch.all(vector <= kappa + tolerance)
    ):
        raise ValueError("physical chart vector must lie in [0,kappa]")
    return unrestricted_log_chart_to_transfer(chart)


def compile_probe_certified_linear_atlas(
    *,
    kind: AtlasKind,
    probe_times: torch.Tensor,
    value_evaluator: Callable[[torch.Tensor], torch.Tensor],
    primal_tolerance: float,
    tangent_evaluator: Callable[[torch.Tensor], torch.Tensor] | None = None,
    tangent_tolerance: float | None = None,
    maximum_nodes: int | None = None,
) -> tuple[LinearTemporalAtlas, AtlasCertificate]:
    """Greedily certify one atlas on a declared finite probe grid.

    The evaluator is called exactly once on the complete probe tensor.  This
    prevents adaptive selection from changing the reference function between
    iterations, but it does not turn the finite grid into a continuous proof.
    """

    if not isinstance(kind, AtlasKind):
        raise TypeError("kind must be AtlasKind")
    if not isinstance(probe_times, torch.Tensor) or probe_times.ndim != 1:
        raise ValueError("probe_times must be a rank-one tensor")
    if probe_times.numel() < 3:
        raise ValueError("atlas certification requires at least three probes")
    if not bool(torch.all(probe_times[1:] > probe_times[:-1])):
        raise ValueError("probe times must be strictly increasing")
    if not primal_tolerance > 0.0:
        raise ValueError("primal_tolerance must be positive")
    if (tangent_evaluator is None) != (tangent_tolerance is None):
        raise ValueError("tangent evaluator and tolerance must be supplied together")
    if tangent_tolerance is not None and not tangent_tolerance > 0.0:
        raise ValueError("tangent_tolerance must be positive")
    probe_count = int(probe_times.numel())
    if maximum_nodes is None:
        maximum_nodes = probe_count
    if (
        isinstance(maximum_nodes, bool)
        or not isinstance(maximum_nodes, int)
        or maximum_nodes < 2
        or maximum_nodes > probe_count
    ):
        raise ValueError("maximum_nodes must lie in [2, probe_count]")

    reference_values = value_evaluator(probe_times)
    if reference_values.shape != (probe_count, 4):
        raise ValueError("value evaluator must return [probe_count,4]")
    if not bool(torch.all(torch.isfinite(reference_values))):
        raise ValueError("atlas reference values must be finite")
    reference_tangents = (
        None if tangent_evaluator is None else tangent_evaluator(probe_times)
    )
    if reference_tangents is not None and (
        reference_tangents.shape != (probe_count, 4)
        or not bool(torch.all(torch.isfinite(reference_tangents)))
    ):
        raise ValueError("tangent evaluator must return finite [probe_count,4]")

    selected = torch.zeros(
        (probe_count,),
        dtype=torch.bool,
        device=probe_times.device,
    )
    selected[0] = True
    selected[-1] = True
    maximum_primal_error = float("inf")
    maximum_tangent_error: float | None = None

    while True:
        indices = torch.nonzero(selected, as_tuple=False).flatten()
        atlas = LinearTemporalAtlas(
            kind=kind,
            knots=probe_times[indices],
            values=reference_values[indices],
        )
        atlas.validate()
        approximation = atlas.evaluate(probe_times)
        per_probe_primal = torch.linalg.vector_norm(
            approximation - reference_values,
            dim=-1,
        ) / torch.clamp(
            torch.linalg.vector_norm(reference_values, dim=-1),
            min=1.0,
        )
        combined = per_probe_primal / primal_tolerance
        maximum_primal_error = float(torch.amax(per_probe_primal).detach().cpu())
        maximum_tangent_error = None
        if reference_tangents is not None:
            tangent_approximation = atlas.derivative(probe_times)
            per_probe_tangent = torch.linalg.vector_norm(
                tangent_approximation - reference_tangents,
                dim=-1,
            ) / torch.clamp(
                torch.linalg.vector_norm(reference_tangents, dim=-1),
                min=1.0,
            )
            combined = torch.maximum(
                combined,
                per_probe_tangent / tangent_tolerance,
            )
            maximum_tangent_error = float(
                torch.amax(per_probe_tangent).detach().cpu()
            )
        if bool(torch.all(combined <= 1.0)) or atlas.node_count >= maximum_nodes:
            verified = bool(torch.all(combined <= 1.0))
            return atlas, AtlasCertificate(
                kind=kind,
                node_count=atlas.node_count,
                probe_count=probe_count,
                maximum_primal_error=maximum_primal_error,
                maximum_tangent_error=maximum_tangent_error,
                primal_tolerance=primal_tolerance,
                tangent_tolerance=tangent_tolerance,
                payload_bytes=atlas.payload_bytes,
                verified=verified,
            )
        combined = torch.where(
            selected,
            torch.full_like(combined, -1.0),
            combined,
        )
        selected[int(torch.argmax(combined))] = True


__all__ = [
    "AtlasCertificate",
    "AtlasKind",
    "LinearTemporalAtlas",
    "compile_probe_certified_linear_atlas",
    "physical_chart_to_transfer",
    "transfer_to_physical_chart",
    "transfer_to_unrestricted_log_chart",
    "unrestricted_log_chart_to_transfer",
]
