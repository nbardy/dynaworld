"""Small differentiable fitting harness for connection representations.

This is a reference optimizer, not the production renderer trainer.  It fits
the same fixed temporal-node family for direct physical ``U``, unrestricted
``U_tilde``, or signed ``K_F`` and always scores the reconstructed physical
transfer/radiance.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .temporal_atlas import (
    AtlasKind,
    transfer_to_physical_chart,
    transfer_to_unrestricted_log_chart,
    unrestricted_log_chart_to_transfer,
)


@dataclass(frozen=True)
class AtlasFitConfig:
    steps: int = 300
    learning_rate: float = 1.0e-2
    tangent_weight: float = 0.0
    cone_weight: float = 10.0
    gradient_clip_norm: float | None = 10.0

    def validate(self) -> None:
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError("fit steps must be a positive integer")
        if not self.learning_rate > 0.0:
            raise ValueError("fit learning_rate must be positive")
        if self.tangent_weight < 0.0 or self.cone_weight < 0.0:
            raise ValueError("fit weights must be nonnegative")
        if self.gradient_clip_norm is not None and not self.gradient_clip_norm > 0.0:
            raise ValueError("gradient_clip_norm must be positive when supplied")


@dataclass(frozen=True)
class AtlasFitReport:
    kind: AtlasKind
    step_count: int
    parameter_count: int
    parameter_bytes: int
    initial_loss: float
    final_loss: float
    final_transfer_loss: float
    final_tangent_loss: float
    final_cone_penalty: float
    finite: bool
    loss_decreased: bool


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    return value + torch.log(-torch.expm1(-value))


def _logit(value: torch.Tensor) -> torch.Tensor:
    return torch.log(value) - torch.log1p(-value)


def _compose_raw(front: torch.Tensor, rear: torch.Tensor) -> torch.Tensor:
    if front.shape[-1] != 4 or rear.shape[-1] != 4:
        raise ValueError("raw affine transfers require trailing shape 4")
    front, rear = torch.broadcast_tensors(front, rear)
    beta = front[..., :1] * rear[..., :1]
    moment = front[..., 1:] + front[..., :1] * rear[..., 1:]
    return torch.cat((beta, moment), dim=-1)


def _inverse_raw(transfer: torch.Tensor) -> torch.Tensor:
    if transfer.shape[-1] != 4:
        raise ValueError("raw affine transfer requires trailing shape 4")
    beta = transfer[..., :1]
    inverse_beta = beta.reciprocal()
    return torch.cat((inverse_beta, -inverse_beta * transfer[..., 1:]), dim=-1)


def _interpolate(
    knots: torch.Tensor,
    values: torch.Tensor,
    query_times: torch.Tensor,
) -> torch.Tensor:
    if not bool(
        torch.all((query_times >= knots[0]) & (query_times <= knots[-1]))
    ):
        raise ValueError("atlas query lies outside its closed interval")
    flat = query_times.reshape(-1)
    right = torch.searchsorted(knots, flat, right=True)
    left = torch.clamp(right - 1, 0, knots.numel() - 2)
    right = left + 1
    alpha = (flat - knots[left]) / (knots[right] - knots[left])
    result = (
        (1.0 - alpha[:, None]) * values[left]
        + alpha[:, None] * values[right]
    )
    return result.reshape(query_times.shape + (4,))


def _integrate(
    knots: torch.Tensor,
    values: torch.Tensor,
    query_times: torch.Tensor,
) -> torch.Tensor:
    if not bool(
        torch.all((query_times >= knots[0]) & (query_times <= knots[-1]))
    ):
        raise ValueError("atlas query lies outside its closed interval")
    flat = query_times.reshape(-1)
    right = torch.searchsorted(knots, flat, right=True)
    left = torch.clamp(right - 1, 0, knots.numel() - 2)
    right = left + 1
    widths = knots[1:] - knots[:-1]
    segments = 0.5 * widths[:, None] * (values[:-1] + values[1:])
    cumulative = torch.cat(
        (torch.zeros_like(segments[:1]), torch.cumsum(segments, dim=0)),
        dim=0,
    )
    local_width = flat - knots[left]
    alpha = local_width / (knots[right] - knots[left])
    local_end = (
        (1.0 - alpha[:, None]) * values[left]
        + alpha[:, None] * values[right]
    )
    local = 0.5 * local_width[:, None] * (values[left] + local_end)
    return (cumulative[left] + local).reshape(query_times.shape + (4,))


class TrainableConnectionAtlas(nn.Module):
    """Trainable fixed-node realization of A0, A1, or A2."""

    def __init__(
        self,
        *,
        kind: AtlasKind,
        knots: torch.Tensor,
        initial_values: torch.Tensor,
        base_group_transfer: torch.Tensor | None = None,
        minimum_optical_depth: float = 1.0e-8,
        minimum_group_beta: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if not isinstance(kind, AtlasKind):
            raise TypeError("kind must be AtlasKind")
        if knots.ndim != 1 or knots.numel() < 2:
            raise ValueError("trainable atlas requires at least two knots")
        if initial_values.shape != (knots.numel(), 4):
            raise ValueError("initial atlas values must have shape [J,4]")
        if knots.device != initial_values.device or knots.dtype != initial_values.dtype:
            raise ValueError("atlas initialization must share device and dtype")
        if not bool(torch.all(knots[1:] > knots[:-1])):
            raise ValueError("atlas knots must be strictly increasing")
        if not minimum_optical_depth > 0.0:
            raise ValueError("minimum_optical_depth must be positive")
        if not minimum_group_beta > 0.0:
            raise ValueError("minimum_group_beta must be positive")
        self.kind = kind
        self.minimum_optical_depth = float(minimum_optical_depth)
        self.minimum_group_beta = float(minimum_group_beta)
        self.register_buffer("knots", knots.detach().clone())
        self.raw_kappa: nn.Parameter | None = None
        self.raw_fraction: nn.Parameter | None = None
        self.chart_values: nn.Parameter | None = None
        self.tangent_values: nn.Parameter | None = None
        self.base_group_chart: nn.Parameter | None = None

        if kind is AtlasKind.PHYSICAL_U:
            chart = transfer_to_physical_chart(initial_values)
            kappa = torch.clamp(
                chart[:, :1],
                min=self.minimum_optical_depth,
            )
            fraction = torch.clamp(
                chart[:, 1:] / kappa,
                min=1.0e-6,
                max=1.0 - 1.0e-6,
            )
            self.raw_kappa = nn.Parameter(
                _inverse_softplus(
                    torch.clamp(
                        kappa - self.minimum_optical_depth,
                        min=self.minimum_optical_depth,
                    )
                )
            )
            self.raw_fraction = nn.Parameter(_logit(fraction))
        elif kind is AtlasKind.GROUP_U_TILDE:
            self.chart_values = nn.Parameter(
                transfer_to_unrestricted_log_chart(initial_values)
            )
        else:
            if base_group_transfer is None or base_group_transfer.shape != (4,):
                raise ValueError("K_F atlas requires one base group transfer [4]")
            if base_group_transfer.device != knots.device or base_group_transfer.dtype != knots.dtype:
                raise ValueError("K_F base transfer must share atlas device/dtype")
            self.tangent_values = nn.Parameter(initial_values.detach().clone())
            self.base_group_chart = nn.Parameter(
                transfer_to_unrestricted_log_chart(base_group_transfer)
            )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def parameter_bytes(self) -> int:
        return sum(
            parameter.numel() * parameter.element_size()
            for parameter in self.parameters()
        )

    def node_values(self) -> torch.Tensor:
        if self.kind is AtlasKind.PHYSICAL_U:
            kappa = self.minimum_optical_depth + torch.nn.functional.softplus(
                self.raw_kappa
            )
            vector = kappa * torch.sigmoid(self.raw_fraction)
            return torch.cat((kappa, vector), dim=-1)
        if self.kind is AtlasKind.GROUP_U_TILDE:
            return self.chart_values
        return self.tangent_values

    def represented_transfer(self, query_times: torch.Tensor) -> torch.Tensor:
        """Return direct ``U`` or group-completion ``U_tilde`` before endpoints."""

        if self.kind is AtlasKind.PHYSICAL_U:
            return unrestricted_log_chart_to_transfer(
                _interpolate(self.knots, self.node_values(), query_times)
            )
        if self.kind is AtlasKind.GROUP_U_TILDE:
            return unrestricted_log_chart_to_transfer(
                _interpolate(self.knots, self.node_values(), query_times)
            )
        base = unrestricted_log_chart_to_transfer(self.base_group_chart)
        return base + _integrate(self.knots, self.node_values(), query_times)

    def physical_transfer(
        self,
        query_times: torch.Tensor,
        *,
        near_endpoint_transport: torch.Tensor | None = None,
        far_endpoint_transport: torch.Tensor | None = None,
    ) -> torch.Tensor:
        represented = self.represented_transfer(query_times)
        if self.kind is AtlasKind.PHYSICAL_U:
            if near_endpoint_transport is not None or far_endpoint_transport is not None:
                raise ValueError("direct U does not consume endpoint transports")
            return represented
        if near_endpoint_transport is None or far_endpoint_transport is None:
            raise ValueError("U_tilde/K_F reconstruction requires both endpoint transports")
        if near_endpoint_transport.shape != represented.shape or far_endpoint_transport.shape != represented.shape:
            raise ValueError("endpoint transports must match represented transfer shape")
        group_betas = torch.cat(
            (
                represented[..., :1],
                near_endpoint_transport[..., :1],
                far_endpoint_transport[..., :1],
            ),
            dim=-1,
        )
        if not bool(torch.all(torch.isfinite(group_betas))):
            raise ValueError("group-completion reconstruction became nonfinite")
        if not bool(torch.all(group_betas > self.minimum_group_beta)):
            raise ValueError(
                "group-completion reconstruction crossed beta<=minimum_group_beta"
            )
        return _compose_raw(
            _compose_raw(_inverse_raw(near_endpoint_transport), represented),
            far_endpoint_transport,
        )


def physical_cone_penalty(transfer: torch.Tensor) -> torch.Tensor:
    if transfer.shape[-1] != 4:
        raise ValueError("physical cone penalty requires trailing shape 4")
    beta = transfer[..., :1]
    moment = transfer[..., 1:]
    return (
        torch.relu(-beta).square().mean()
        + torch.relu(beta - 1.0).square().mean()
        + torch.relu(-moment).square().mean()
        + torch.relu(moment - (1.0 - beta)).square().mean()
    )


def render_rear_radiance(
    transfer: torch.Tensor,
    rear_radiance: torch.Tensor,
) -> torch.Tensor:
    if transfer.shape[-1] != 4 or rear_radiance.shape[-1] != 3:
        raise ValueError("render expects transfer [...,4] and radiance [...,3]")
    return transfer[..., 1:] + transfer[..., :1] * rear_radiance


def fit_connection_atlas(
    model: TrainableConnectionAtlas,
    *,
    query_times: torch.Tensor,
    target_physical_transfer: torch.Tensor,
    config: AtlasFitConfig,
    near_endpoint_transport: torch.Tensor | None = None,
    far_endpoint_transport: torch.Tensor | None = None,
) -> AtlasFitReport:
    """Fit one representation without retaining a step-sized autograd history."""

    config.validate()
    if target_physical_transfer.shape != query_times.shape + (4,):
        raise ValueError("target transfer shape must equal query shape plus 4")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    initial_loss: float | None = None

    for _ in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        predicted = model.physical_transfer(
            query_times,
            near_endpoint_transport=near_endpoint_transport,
            far_endpoint_transport=far_endpoint_transport,
        )
        transfer_loss = torch.mean((predicted - target_physical_transfer).square())
        tangent_loss = torch.zeros_like(transfer_loss)
        if config.tangent_weight > 0.0:
            dt = query_times[1:] - query_times[:-1]
            if query_times.ndim != 1 or query_times.numel() < 2 or not bool(torch.all(dt > 0.0)):
                raise ValueError("tangent fitting requires increasing rank-one query times")
            predicted_tangent = (predicted[1:] - predicted[:-1]) / dt[:, None]
            target_tangent = (
                target_physical_transfer[1:] - target_physical_transfer[:-1]
            ) / dt[:, None]
            tangent_loss = torch.mean((predicted_tangent - target_tangent).square())
        cone_penalty = physical_cone_penalty(predicted)
        loss = (
            transfer_loss
            + config.tangent_weight * tangent_loss
            + config.cone_weight * cone_penalty
        )
        if initial_loss is None:
            initial_loss = float(loss.detach().cpu())
        loss.backward()
        if config.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.gradient_clip_norm,
            )
        optimizer.step()

    with torch.no_grad():
        predicted = model.physical_transfer(
            query_times,
            near_endpoint_transport=near_endpoint_transport,
            far_endpoint_transport=far_endpoint_transport,
        )
        final_transfer = torch.mean(
            (predicted - target_physical_transfer).square()
        )
        final_tangent = torch.zeros_like(final_transfer)
        if config.tangent_weight > 0.0:
            dt = query_times[1:] - query_times[:-1]
            final_tangent = torch.mean(
                (
                    (predicted[1:] - predicted[:-1]) / dt[:, None]
                    - (
                        target_physical_transfer[1:]
                        - target_physical_transfer[:-1]
                    )
                    / dt[:, None]
                ).square()
            )
        final_cone = physical_cone_penalty(predicted)
        final_total = (
            final_transfer
            + config.tangent_weight * final_tangent
            + config.cone_weight * final_cone
        )
    final_loss = float(final_total.detach().cpu())
    return AtlasFitReport(
        kind=model.kind,
        step_count=config.steps,
        parameter_count=model.parameter_count,
        parameter_bytes=model.parameter_bytes,
        initial_loss=initial_loss,
        final_loss=final_loss,
        final_transfer_loss=float(final_transfer.detach().cpu()),
        final_tangent_loss=float(final_tangent.detach().cpu()),
        final_cone_penalty=float(final_cone.detach().cpu()),
        finite=bool(torch.isfinite(final_total)),
        loss_decreased=final_loss < initial_loss,
    )


__all__ = [
    "AtlasFitConfig",
    "AtlasFitReport",
    "TrainableConnectionAtlas",
    "fit_connection_atlas",
    "physical_cone_penalty",
    "render_rear_radiance",
]
