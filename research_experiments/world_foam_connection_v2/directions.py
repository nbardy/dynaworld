"""Explicit forward-mode selected directions for the connection reference.

Callers supply every tensor tangent directly.  This module accepts no model
callback and stores no graph or cache: each requested direction performs one
fresh ``torch.func.jvp`` through :func:`evaluate_connection_core`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .connection import (
    ConnectionCoreEvaluation,
    EndpointTransportHistory,
    P0FlowSamples,
    P0Ray,
    P0RayRate,
    evaluate_connection_core,
    gauss_legendre_layout,
)


@dataclass(frozen=True)
class ConnectionTensorDirection:
    """One selected direction matching all differentiable core inputs."""

    cuts: torch.Tensor
    extinction: torch.Tensor
    emission_density: torch.Tensor
    cut_velocity: torch.Tensor
    extinction_time: torch.Tensor
    emission_density_time: torch.Tensor
    flow_bulk_value: torch.Tensor
    flow_bulk_d_dz: torch.Tensor
    flow_cell_left_value: torch.Tensor
    flow_cell_right_value: torch.Tensor
    endpoint_durations: torch.Tensor
    endpoint_near_scalar: torch.Tensor
    endpoint_near_source: torch.Tensor
    endpoint_far_scalar: torch.Tensor
    endpoint_far_source: torch.Tensor


@dataclass(frozen=True)
class ConnectionObservables:
    """Named four-vector outputs used for primal/tangent certification."""

    direct_transfer: torch.Tensor
    direct_time_derivative: torch.Tensor
    endpoint_flux: torch.Tensor
    bulk_curvature: torch.Tensor
    singular_curvature: torch.Tensor
    total_curvature: torch.Tensor
    predicted_time_derivative: torch.Tensor
    theorem_residual: torch.Tensor
    near_endpoint_transport: torch.Tensor
    far_endpoint_transport: torch.Tensor
    flow_corrected_transfer: torch.Tensor
    transported_curvature_source: torch.Tensor
    algebraically_transported_covariant_derivative: torch.Tensor
    covariant_residual: torch.Tensor
    reconstructed_transfer: torch.Tensor
    reconstruction_residual: torch.Tensor


@dataclass(frozen=True)
class SelectedDirectionEvaluation:
    """Primal observables and their JVP for one explicit input direction."""

    primal: ConnectionObservables
    tangent: ConnectionObservables


def zero_connection_direction(
    ray: P0Ray,
    rate: P0RayRate,
    flow: P0FlowSamples,
    endpoint_history: EndpointTransportHistory,
) -> ConnectionTensorDirection:
    """Create an all-zero direction with exactly matching tensor metadata."""

    return ConnectionTensorDirection(
        cuts=torch.zeros_like(ray.cuts),
        extinction=torch.zeros_like(ray.extinction),
        emission_density=torch.zeros_like(ray.emission_density),
        cut_velocity=torch.zeros_like(rate.cut_velocity),
        extinction_time=torch.zeros_like(rate.extinction_time),
        emission_density_time=torch.zeros_like(rate.emission_density_time),
        flow_bulk_value=torch.zeros_like(flow.bulk_value),
        flow_bulk_d_dz=torch.zeros_like(flow.bulk_d_dz),
        flow_cell_left_value=torch.zeros_like(flow.cell_left_value),
        flow_cell_right_value=torch.zeros_like(flow.cell_right_value),
        endpoint_durations=torch.zeros_like(endpoint_history.durations),
        endpoint_near_scalar=torch.zeros_like(endpoint_history.near_scalar),
        endpoint_near_source=torch.zeros_like(endpoint_history.near_source),
        endpoint_far_scalar=torch.zeros_like(endpoint_history.far_scalar),
        endpoint_far_source=torch.zeros_like(endpoint_history.far_source),
    )


def evaluate_selected_direction(
    ray: P0Ray,
    rate: P0RayRate,
    flow: P0FlowSamples,
    endpoint_history: EndpointTransportHistory,
    direction: ConnectionTensorDirection,
    *,
    quadrature_order: int,
) -> SelectedDirectionEvaluation:
    """Evaluate a concrete forward-mode direction through the whole theorem.

    The direction includes rates when those rates themselves depend on model
    parameters.  This is a selected parameter JVP, not the physical time
    derivative, which is independently available in the returned observables.
    """

    ray.validate()
    rate.validate_for(ray)
    endpoint_history.validate_for(ray)
    quadrature = gauss_legendre_layout(ray.cuts, quadrature_order)
    flow.validate_for(ray, quadrature)
    primals = _flatten_inputs(ray, rate, flow, endpoint_history)
    tangents = _flatten_direction(direction)
    _validate_direction(primals, tangents)

    def tensor_kernel(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        rebuilt_ray, rebuilt_rate, rebuilt_flow, rebuilt_history = (
            _unflatten_inputs(tensors)
        )
        core = evaluate_connection_core(
            rebuilt_ray,
            rebuilt_rate,
            rebuilt_flow,
            rebuilt_history,
            quadrature_order=quadrature_order,
        )
        return _flatten_observables(core)

    primal_values, tangent_values = torch.func.jvp(
        tensor_kernel,
        primals,
        tangents,
    )
    return SelectedDirectionEvaluation(
        primal=_unflatten_observables(primal_values),
        tangent=_unflatten_observables(tangent_values),
    )


def evaluate_selected_directions(
    ray: P0Ray,
    rate: P0RayRate,
    flow: P0FlowSamples,
    endpoint_history: EndpointTransportHistory,
    directions: tuple[ConnectionTensorDirection, ...]
    | list[ConnectionTensorDirection],
    *,
    quadrature_order: int,
) -> tuple[SelectedDirectionEvaluation, ...]:
    """Evaluate several directions independently, with no retained JVP cache."""

    return tuple(
        evaluate_selected_direction(
            ray,
            rate,
            flow,
            endpoint_history,
            direction,
            quadrature_order=quadrature_order,
        )
        for direction in directions
    )


def _flatten_inputs(
    ray: P0Ray,
    rate: P0RayRate,
    flow: P0FlowSamples,
    history: EndpointTransportHistory,
) -> tuple[torch.Tensor, ...]:
    return (
        ray.cuts,
        ray.extinction,
        ray.emission_density,
        rate.cut_velocity,
        rate.extinction_time,
        rate.emission_density_time,
        flow.bulk_value,
        flow.bulk_d_dz,
        flow.cell_left_value,
        flow.cell_right_value,
        history.durations,
        history.near_scalar,
        history.near_source,
        history.far_scalar,
        history.far_source,
    )


def _flatten_direction(
    direction: ConnectionTensorDirection,
) -> tuple[torch.Tensor, ...]:
    return (
        direction.cuts,
        direction.extinction,
        direction.emission_density,
        direction.cut_velocity,
        direction.extinction_time,
        direction.emission_density_time,
        direction.flow_bulk_value,
        direction.flow_bulk_d_dz,
        direction.flow_cell_left_value,
        direction.flow_cell_right_value,
        direction.endpoint_durations,
        direction.endpoint_near_scalar,
        direction.endpoint_near_source,
        direction.endpoint_far_scalar,
        direction.endpoint_far_source,
    )


def _unflatten_inputs(
    tensors: tuple[torch.Tensor, ...],
) -> tuple[P0Ray, P0RayRate, P0FlowSamples, EndpointTransportHistory]:
    (
        cuts,
        extinction,
        emission_density,
        cut_velocity,
        extinction_time,
        emission_density_time,
        flow_bulk_value,
        flow_bulk_d_dz,
        flow_cell_left_value,
        flow_cell_right_value,
        endpoint_durations,
        endpoint_near_scalar,
        endpoint_near_source,
        endpoint_far_scalar,
        endpoint_far_source,
    ) = tensors
    return (
        P0Ray(cuts, extinction, emission_density),
        P0RayRate(cut_velocity, extinction_time, emission_density_time),
        P0FlowSamples(
            flow_bulk_value,
            flow_bulk_d_dz,
            flow_cell_left_value,
            flow_cell_right_value,
        ),
        EndpointTransportHistory(
            endpoint_durations,
            endpoint_near_scalar,
            endpoint_near_source,
            endpoint_far_scalar,
            endpoint_far_source,
        ),
    )


def _flatten_observables(
    core: ConnectionCoreEvaluation,
) -> tuple[torch.Tensor, ...]:
    return (
        core.ordered.total.as_vector(),
        core.direct_time_derivative.as_vector(),
        core.endpoint_kinematics.flux.as_vector(),
        core.curvature.bulk_total.as_vector(),
        core.curvature.singular_total.as_vector(),
        core.curvature.total.as_vector(),
        core.predicted_time_derivative.as_vector(),
        core.theorem_residual.as_vector(),
        core.endpoint_transports.near.as_vector(),
        core.endpoint_transports.far.as_vector(),
        core.flow_corrected_transfer.as_vector(),
        core.transported_curvature_source.as_vector(),
        core.algebraically_transported_covariant_derivative.as_vector(),
        core.covariant_residual.as_vector(),
        core.reconstructed_transfer.as_vector(),
        core.reconstruction_residual.as_vector(),
    )


def _unflatten_observables(
    values: tuple[torch.Tensor, ...],
) -> ConnectionObservables:
    return ConnectionObservables(*values)


def _validate_direction(
    primals: tuple[torch.Tensor, ...],
    tangents: tuple[torch.Tensor, ...],
) -> None:
    if len(primals) != len(tangents):
        raise ValueError("direction has the wrong number of tensors")
    for index, (primal, tangent) in enumerate(zip(primals, tangents, strict=True)):
        if not isinstance(tangent, torch.Tensor):
            raise TypeError(f"direction tensor {index} is not a tensor")
        if tangent.shape != primal.shape:
            raise ValueError(f"direction tensor {index} has the wrong shape")
        if tangent.dtype != primal.dtype:
            raise TypeError(f"direction tensor {index} has the wrong dtype")
        if tangent.device != primal.device:
            raise ValueError(f"direction tensor {index} is on the wrong device")
        if not bool(torch.isfinite(tangent).all()):
            raise ValueError(f"direction tensor {index} must be finite")


__all__ = [
    "ConnectionObservables",
    "ConnectionTensorDirection",
    "SelectedDirectionEvaluation",
    "evaluate_selected_direction",
    "evaluate_selected_directions",
    "zero_connection_direction",
]
