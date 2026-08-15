"""End-to-end scalar-flow sampling and selected geometry/flow directions.

The low-level theorem ABI intentionally consumes explicit flow samples.  This
module supplies the missing chain rule: Gauss--Legendre nodes move with cuts,
the shared Chebyshev field is resampled at those nodes, and one ``jvp`` covers
geometry, material, time, flow coefficients, and endpoint-history tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .connection import (
    EndpointTransportHistory,
    P0FlowSamples,
    P0Ray,
    P0RayRate,
    evaluate_connection_core,
    gauss_legendre_layout,
)
from .shared_flow import (
    SharedChebyshevFlow,
    evaluate_chebyshev_flow_coefficients,
)


@dataclass(frozen=True)
class SharedFlowConnectionDirection:
    cuts: torch.Tensor
    extinction: torch.Tensor
    emission_density: torch.Tensor
    cut_velocity: torch.Tensor
    extinction_time: torch.Tensor
    emission_density_time: torch.Tensor
    time: torch.Tensor
    flow_coefficients: torch.Tensor
    endpoint_durations: torch.Tensor
    endpoint_near_scalar: torch.Tensor
    endpoint_near_source: torch.Tensor
    endpoint_far_scalar: torch.Tensor
    endpoint_far_source: torch.Tensor


@dataclass(frozen=True)
class SharedFlowConnectionObservables:
    direct_transfer: torch.Tensor
    direct_time_derivative: torch.Tensor
    endpoint_flux: torch.Tensor
    curvature: torch.Tensor
    flow_corrected_transfer: torch.Tensor
    transported_curvature_source: torch.Tensor
    theorem_residual: torch.Tensor
    covariant_residual: torch.Tensor
    reconstructed_transfer: torch.Tensor


@dataclass(frozen=True)
class SharedFlowSelectedDirectionEvaluation:
    primal: SharedFlowConnectionObservables
    tangent: SharedFlowConnectionObservables


def sample_shared_chebyshev_flow(
    ray: P0Ray,
    *,
    time: torch.Tensor,
    coefficients: torch.Tensor,
    flow: SharedChebyshevFlow,
    quadrature_order: int,
) -> P0FlowSamples:
    """Sample one continuous shared flow at all bulk and one-sided sites."""

    quadrature = gauss_legendre_layout(ray.cuts, quadrature_order)
    bulk_time = time.expand_as(quadrature.depth_nodes)
    bulk = evaluate_chebyshev_flow_coefficients(
        coefficients,
        domain=flow.domain,
        maximum_speed=flow.maximum_speed,
        t=bulk_time,
        z=quadrature.depth_nodes,
    )
    left_depth = ray.cuts[:-1]
    right_depth = ray.cuts[1:]
    left = evaluate_chebyshev_flow_coefficients(
        coefficients,
        domain=flow.domain,
        maximum_speed=flow.maximum_speed,
        t=time.expand_as(left_depth),
        z=left_depth,
    )
    right = evaluate_chebyshev_flow_coefficients(
        coefficients,
        domain=flow.domain,
        maximum_speed=flow.maximum_speed,
        t=time.expand_as(right_depth),
        z=right_depth,
    )
    return P0FlowSamples(
        bulk_value=bulk.value,
        bulk_d_dz=bulk.d_dz,
        cell_left_value=left.value,
        cell_right_value=right.value,
    )


def evaluate_shared_flow_connection_core(
    ray: P0Ray,
    rate: P0RayRate,
    *,
    time: torch.Tensor,
    flow: SharedChebyshevFlow,
    coefficients: torch.Tensor,
    endpoint_history: EndpointTransportHistory,
    quadrature_order: int,
):
    samples = sample_shared_chebyshev_flow(
        ray,
        time=time,
        coefficients=coefficients,
        flow=flow,
        quadrature_order=quadrature_order,
    )
    return evaluate_connection_core(
        ray,
        rate,
        samples,
        endpoint_history,
        quadrature_order=quadrature_order,
    )


def zero_shared_flow_connection_direction(
    ray: P0Ray,
    rate: P0RayRate,
    *,
    time: torch.Tensor,
    flow: SharedChebyshevFlow,
    endpoint_history: EndpointTransportHistory,
) -> SharedFlowConnectionDirection:
    return SharedFlowConnectionDirection(
        cuts=torch.zeros_like(ray.cuts),
        extinction=torch.zeros_like(ray.extinction),
        emission_density=torch.zeros_like(ray.emission_density),
        cut_velocity=torch.zeros_like(rate.cut_velocity),
        extinction_time=torch.zeros_like(rate.extinction_time),
        emission_density_time=torch.zeros_like(rate.emission_density_time),
        time=torch.zeros_like(time),
        flow_coefficients=torch.zeros_like(flow.coefficients),
        endpoint_durations=torch.zeros_like(endpoint_history.durations),
        endpoint_near_scalar=torch.zeros_like(endpoint_history.near_scalar),
        endpoint_near_source=torch.zeros_like(endpoint_history.near_source),
        endpoint_far_scalar=torch.zeros_like(endpoint_history.far_scalar),
        endpoint_far_source=torch.zeros_like(endpoint_history.far_source),
    )


def evaluate_shared_flow_selected_direction(
    ray: P0Ray,
    rate: P0RayRate,
    *,
    time: torch.Tensor,
    flow: SharedChebyshevFlow,
    endpoint_history: EndpointTransportHistory,
    direction: SharedFlowConnectionDirection,
    quadrature_order: int,
) -> SharedFlowSelectedDirectionEvaluation:
    """JVP through cut-dependent sampling and the full scalar-flow core."""

    ray.validate()
    rate.validate_for(ray)
    endpoint_history.validate_for(ray)
    if time.ndim != 0 or time.dtype != ray.cuts.dtype or time.device != ray.cuts.device:
        raise ValueError("shared-flow connection time must be scalar on ray dtype/device")
    if not bool(
        (time >= flow.domain.t_min)
        & (time <= flow.domain.t_max)
        & torch.all(ray.cuts >= flow.domain.z_min)
        & torch.all(ray.cuts <= flow.domain.z_max)
    ):
        raise ValueError("shared-flow connection query lies outside the flow domain")

    primals = (
        ray.cuts,
        ray.extinction,
        ray.emission_density,
        rate.cut_velocity,
        rate.extinction_time,
        rate.emission_density_time,
        time,
        flow.coefficients,
        endpoint_history.durations,
        endpoint_history.near_scalar,
        endpoint_history.near_source,
        endpoint_history.far_scalar,
        endpoint_history.far_source,
    )
    tangents = (
        direction.cuts,
        direction.extinction,
        direction.emission_density,
        direction.cut_velocity,
        direction.extinction_time,
        direction.emission_density_time,
        direction.time,
        direction.flow_coefficients,
        direction.endpoint_durations,
        direction.endpoint_near_scalar,
        direction.endpoint_near_source,
        direction.endpoint_far_scalar,
        direction.endpoint_far_source,
    )
    for index, (primal, tangent) in enumerate(zip(primals, tangents, strict=True)):
        if tangent.shape != primal.shape:
            raise ValueError(f"shared-flow direction tensor {index} has wrong shape")
        if tangent.dtype != primal.dtype or tangent.device != primal.device:
            raise ValueError(f"shared-flow direction tensor {index} has wrong metadata")
        if not bool(torch.all(torch.isfinite(tangent))):
            raise ValueError(f"shared-flow direction tensor {index} must be finite")

    def tensor_kernel(*values: torch.Tensor) -> tuple[torch.Tensor, ...]:
        (
            cuts,
            extinction,
            emission,
            cut_velocity,
            extinction_time,
            emission_time,
            query_time,
            coefficients,
            durations,
            near_scalar,
            near_source,
            far_scalar,
            far_source,
        ) = values
        core = evaluate_shared_flow_connection_core(
            P0Ray(cuts, extinction, emission),
            P0RayRate(cut_velocity, extinction_time, emission_time),
            time=query_time,
            flow=flow,
            coefficients=coefficients,
            endpoint_history=EndpointTransportHistory(
                durations,
                near_scalar,
                near_source,
                far_scalar,
                far_source,
            ),
            quadrature_order=quadrature_order,
        )
        return (
            core.ordered.total.as_vector(),
            core.direct_time_derivative.as_vector(),
            core.endpoint_kinematics.flux.as_vector(),
            core.curvature.total.as_vector(),
            core.flow_corrected_transfer.as_vector(),
            core.transported_curvature_source.as_vector(),
            core.theorem_residual.as_vector(),
            core.covariant_residual.as_vector(),
            core.reconstructed_transfer.as_vector(),
        )

    primal, tangent = torch.func.jvp(tensor_kernel, primals, tangents)
    return SharedFlowSelectedDirectionEvaluation(
        primal=SharedFlowConnectionObservables(*primal),
        tangent=SharedFlowConnectionObservables(*tangent),
    )


__all__ = [
    "SharedFlowConnectionDirection",
    "SharedFlowConnectionObservables",
    "SharedFlowSelectedDirectionEvaluation",
    "evaluate_shared_flow_connection_core",
    "evaluate_shared_flow_selected_direction",
    "sample_shared_chebyshev_flow",
    "zero_shared_flow_connection_direction",
]
