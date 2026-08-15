"""Provenance-sealed construction of endpoint transport histories.

The low-level theorem kernel accepts an explicit history so that it remains a
pure tensor map.  These builders derive every endpoint generator from the same
ray, cut velocity, and sampled flow that enter that kernel, and return an
honest retained-state receipt.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .affine import AffineGenerator, scale_generator
from .connection import (
    EndpointTransportHistory,
    P0FlowSamples,
    P0Ray,
    P0RayRate,
)


@dataclass(frozen=True)
class EndpointHistoryReceipt:
    step_count: int
    retained_scalar_count: int
    retained_bytes: int
    derived_from_connection_inputs: bool
    piecewise_constant_time_rule: str
    uses_requested_frame_table: bool


def derive_endpoint_generators(
    ray: P0Ray,
    rate: P0RayRate,
    flow: P0FlowSamples,
) -> tuple[AffineGenerator, AffineGenerator]:
    """Derive ``B=(z_dot-w)A_z`` at the near and far clips."""

    ray.validate()
    rate.validate_for(ray)
    if flow.bulk_value.ndim != 2 or flow.bulk_value.shape[1] < 1:
        raise ValueError("endpoint flow must have a positive quadrature order")
    expected_bulk = (ray.run_count, int(flow.bulk_value.shape[1]))
    if flow.bulk_value.shape != expected_bulk or flow.bulk_d_dz.shape != expected_bulk:
        raise ValueError("endpoint flow bulk tensors must have shape [R,Q]")
    if flow.cell_left_value.shape != (ray.run_count,) or flow.cell_right_value.shape != (ray.run_count,):
        raise ValueError("endpoint flow traces must have shape [R]")
    for value in (
        flow.bulk_value,
        flow.bulk_d_dz,
        flow.cell_left_value,
        flow.cell_right_value,
    ):
        if value.dtype != ray.cuts.dtype or value.device != ray.cuts.device:
            raise ValueError("endpoint flow tensors must share ray dtype/device")
        if not bool(torch.all(torch.isfinite(value))):
            raise ValueError("endpoint flow tensors must be finite")
    near = scale_generator(
        rate.cut_velocity[0] - flow.cell_left_value[0],
        AffineGenerator(
            scalar=-ray.extinction[0],
            source=ray.emission_density[0],
        ),
    )
    far = scale_generator(
        rate.cut_velocity[-1] - flow.cell_right_value[-1],
        AffineGenerator(
            scalar=-ray.extinction[-1],
            source=ray.emission_density[-1],
        ),
    )
    return near, far


def derive_constant_endpoint_history(
    ray: P0Ray,
    rate: P0RayRate,
    flow: P0FlowSamples,
    *,
    duration: torch.Tensor,
) -> tuple[EndpointTransportHistory, EndpointHistoryReceipt]:
    """Build one exact constant-generator time step from current inputs."""

    if duration.ndim != 0 or duration.dtype != ray.cuts.dtype or duration.device != ray.cuts.device:
        raise ValueError("endpoint duration must be a scalar on the ray dtype/device")
    if not bool(torch.isfinite(duration)) or not bool(duration >= 0.0):
        raise ValueError("endpoint duration must be finite and nonnegative")
    near, far = derive_endpoint_generators(ray, rate, flow)
    history = EndpointTransportHistory(
        durations=duration.reshape(1),
        near_scalar=near.scalar.reshape(1),
        near_source=near.source.reshape(1, 3),
        far_scalar=far.scalar.reshape(1),
        far_source=far.source.reshape(1, 3),
    )
    retained = sum(
        tensor.numel()
        for tensor in (
            history.durations,
            history.near_scalar,
            history.near_source,
            history.far_scalar,
            history.far_source,
        )
    )
    return history, EndpointHistoryReceipt(
        step_count=1,
        retained_scalar_count=retained,
        retained_bytes=retained * duration.element_size(),
        derived_from_connection_inputs=True,
        piecewise_constant_time_rule="one exact constant endpoint-generator step",
        uses_requested_frame_table=False,
    )


def derive_piecewise_constant_endpoint_history(
    *,
    interval_durations: torch.Tensor,
    rays: tuple[P0Ray, ...] | list[P0Ray],
    rates: tuple[P0RayRate, ...] | list[P0RayRate],
    flows: tuple[P0FlowSamples, ...] | list[P0FlowSamples],
) -> tuple[EndpointTransportHistory, EndpointHistoryReceipt]:
    """Build a charged left-sampled P0 time history from exact snapshots.

    This is an oracle integration rule, not a compact temporal representation.
    Supplying one snapshot per requested frame is reported as such and cannot
    be used to claim memory-light scaling.
    """

    step_count = int(interval_durations.numel())
    if interval_durations.ndim != 1 or step_count < 1:
        raise ValueError("endpoint intervals must be a nonempty vector")
    if len(rays) != step_count or len(rates) != step_count or len(flows) != step_count:
        raise ValueError("endpoint histories need one ray/rate/flow per interval")
    if not bool(torch.all(torch.isfinite(interval_durations))) or not bool(
        torch.all(interval_durations > 0.0)
    ):
        raise ValueError("endpoint interval durations must be finite and positive")
    near = []
    far = []
    for ray, rate, flow in zip(rays, rates, flows, strict=True):
        if ray.cuts.dtype != interval_durations.dtype or ray.cuts.device != interval_durations.device:
            raise ValueError("all endpoint snapshots must share interval dtype/device")
        near_generator, far_generator = derive_endpoint_generators(ray, rate, flow)
        near.append(near_generator)
        far.append(far_generator)
    history = EndpointTransportHistory(
        durations=interval_durations,
        near_scalar=torch.stack([item.scalar for item in near]),
        near_source=torch.stack([item.source for item in near]),
        far_scalar=torch.stack([item.scalar for item in far]),
        far_source=torch.stack([item.source for item in far]),
    )
    retained = sum(
        tensor.numel()
        for tensor in (
            history.durations,
            history.near_scalar,
            history.near_source,
            history.far_scalar,
            history.far_source,
        )
    )
    return history, EndpointHistoryReceipt(
        step_count=step_count,
        retained_scalar_count=retained,
        retained_bytes=retained * interval_durations.element_size(),
        derived_from_connection_inputs=True,
        piecewise_constant_time_rule="left-sampled P0 endpoint-generator scan",
        uses_requested_frame_table=step_count > 1,
    )


__all__ = [
    "EndpointHistoryReceipt",
    "derive_constant_endpoint_history",
    "derive_endpoint_generators",
    "derive_piecewise_constant_endpoint_history",
]
