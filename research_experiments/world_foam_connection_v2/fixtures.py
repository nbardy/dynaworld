"""Analytic fixtures for the constrained ray-fiber optical connection.

The fixtures deliberately expose every quantity consumed by the theorem
kernel.  They do not call a renderer, retrieve a target, or hide a learned
answer in a per-frame/per-ray table.  All dynamic fixtures are valid only on
their documented stable P0 chart; topology events belong to a separate
compiler layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .affine import (
    AffineGenerator,
    AffineTransferTangent,
    generator_exponential,
    generator_sandwich,
)
from .connection import (
    EndpointTransportHistory,
    FlowAdmissibilityDeclaration,
    P0FlowSamples,
    P0Ray,
    P0RayRate,
    SensorDepthLiftInput,
    gauss_legendre_layout,
)
from .endpoint_history import derive_constant_endpoint_history


DEFAULT_QUADRATURE_ORDER = 4


@dataclass(frozen=True)
class ConnectionFixture:
    """One fully explicit stable-chart snapshot."""

    name: str
    description: str
    time: torch.Tensor
    ray: P0Ray
    rate: P0RayRate
    flow: P0FlowSamples
    endpoint_history: EndpointTransportHistory
    flow_declaration: FlowAdmissibilityDeclaration
    quadrature_order: int
    expected_flow_admissible: bool


@dataclass(frozen=True)
class SmoothCurvatureCancellationFixture:
    """Nonzero pointwise curvature whose ordered depth integral cancels."""

    depth: torch.Tensor
    generator: AffineGenerator
    generator_time: AffineGenerator
    flow: torch.Tensor
    flow_depth_derivative: torch.Tensor
    curvature: AffineGenerator
    transported_curvature_integral: AffineTransferTangent


def _time_tensor(
    time: float | torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> torch.Tensor:
    if isinstance(time, torch.Tensor):
        if time.ndim != 0:
            raise ValueError("fixture time must be scalar")
        if time.dtype not in {torch.float32, torch.float64}:
            raise TypeError("fixture time must use float32 or float64")
        if dtype != time.dtype:
            raise TypeError("explicit fixture dtype must match tensor time")
        if device is not None and torch.device(device) != time.device:
            raise ValueError("explicit fixture device must match tensor time")
        return time
    return torch.tensor(float(time), dtype=dtype, device=device)


def _vector(reference: torch.Tensor, values: tuple[float, ...]) -> torch.Tensor:
    return reference.new_tensor(values)


def _matrix3(
    reference: torch.Tensor,
    rows: tuple[tuple[float, float, float], ...],
) -> torch.Tensor:
    return reference.new_tensor(rows)


def _zero_rate(ray: P0Ray) -> P0RayRate:
    return P0RayRate(
        cut_velocity=torch.zeros_like(ray.cuts),
        extinction_time=torch.zeros_like(ray.extinction),
        emission_density_time=torch.zeros_like(ray.emission_density),
    )


def _empty_history(reference: torch.Tensor) -> EndpointTransportHistory:
    return EndpointTransportHistory(
        durations=reference.new_empty((0,)),
        near_scalar=reference.new_empty((0,)),
        near_source=reference.new_empty((0, 3)),
        far_scalar=reference.new_empty((0,)),
        far_source=reference.new_empty((0, 3)),
    )


def _constant_flow(
    ray: P0Ray,
    *,
    quadrature_order: int,
    per_run_value: torch.Tensor,
    per_run_depth_derivative: torch.Tensor | None = None,
    left_traces: torch.Tensor | None = None,
    right_traces: torch.Tensor | None = None,
) -> P0FlowSamples:
    if per_run_value.shape != (ray.run_count,):
        raise ValueError("per-run flow value must have shape [R]")
    derivative = (
        torch.zeros_like(per_run_value)
        if per_run_depth_derivative is None
        else per_run_depth_derivative
    )
    if derivative.shape != per_run_value.shape:
        raise ValueError("per-run flow derivative must have shape [R]")
    return P0FlowSamples(
        bulk_value=per_run_value[:, None].expand(-1, quadrature_order),
        bulk_d_dz=derivative[:, None].expand(-1, quadrature_order),
        cell_left_value=(
            per_run_value if left_traces is None else left_traces
        ),
        cell_right_value=(
            per_run_value if right_traces is None else right_traces
        ),
    )


def _declaration(
    *,
    run_count: int,
    maximum_abs_speed: float,
    maximum_abs_depth_gradient: float = 1.0,
    continuity_tolerance: float = 1.0e-12,
) -> FlowAdmissibilityDeclaration:
    return FlowAdmissibilityDeclaration(
        temporal_dof=1,
        source_motion_temporal_dof=max(1, 2 * run_count),
        retained_bytes=8,
        shared_scene_camera_model=True,
        source_motion_retained_bytes=max(8, 16 * run_count),
        maximum_abs_speed=maximum_abs_speed,
        maximum_abs_depth_gradient=maximum_abs_depth_gradient,
        continuity_tolerance=continuity_tolerance,
        orientation_horizon=0.5,
    )


def front_red_back_blue(
    time: float | torch.Tensor = 0.0,
    *,
    quadrature_order: int = DEFAULT_QUADRATURE_ORDER,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> ConnectionFixture:
    """Noncommuting order sentinel: red is in front of blue."""

    t = _time_tensor(time, dtype=dtype, device=device)
    ray = P0Ray(
        cuts=_vector(t, (0.0, 1.0, 2.5)),
        extinction=_vector(t, (1.0, 0.8)),
        emission_density=_matrix3(t, ((1.0, 0.0, 0.0), (0.0, 0.0, 0.8))),
    )
    flow = _constant_flow(
        ray,
        quadrature_order=quadrature_order,
        per_run_value=torch.zeros_like(ray.extinction),
    )
    return ConnectionFixture(
        name="front_red_back_blue",
        description="Static front-red/back-blue ordering sentinel.",
        time=t,
        ray=ray,
        rate=_zero_rate(ray),
        flow=flow,
        endpoint_history=_empty_history(t),
        flow_declaration=_declaration(run_count=ray.run_count, maximum_abs_speed=1.0),
        quadrature_order=quadrature_order,
        expected_flow_admissible=True,
    )


def moving_noncommuting_boundary(
    time: float | torch.Tensor = 0.0,
    *,
    quadrature_order: int = DEFAULT_QUADRATURE_ORDER,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> ConnectionFixture:
    """Pinned theorem fixture ``r(t)=1+t`` with ``w=0``."""

    t = _time_tensor(time, dtype=dtype, device=device)
    if not bool((t >= 0.0) & (t < 2.0)):
        raise ValueError("moving-boundary fixture requires 0 <= t < 2")
    ray = P0Ray(
        cuts=torch.stack((t.new_tensor(0.0), 1.0 + t, t.new_tensor(3.0))),
        extinction=_vector(t, (1.0, 2.0)),
        emission_density=_matrix3(t, ((1.0, 0.0, 0.0), (0.0, 0.0, 2.0))),
    )
    rate = P0RayRate(
        cut_velocity=_vector(t, (0.0, 1.0, 0.0)),
        extinction_time=torch.zeros_like(ray.extinction),
        emission_density_time=torch.zeros_like(ray.emission_density),
    )
    flow = _constant_flow(
        ray,
        quadrature_order=quadrature_order,
        per_run_value=torch.zeros_like(ray.extinction),
    )
    return ConnectionFixture(
        name="moving_noncommuting_boundary",
        description="Pinned noncommuting jump with exact -r_dot[A] atom.",
        time=t,
        ray=ray,
        rate=rate,
        flow=flow,
        endpoint_history=_empty_history(t),
        flow_declaration=_declaration(run_count=ray.run_count, maximum_abs_speed=1.0),
        quadrature_order=quadrature_order,
        expected_flow_admissible=True,
    )


def advected_colored_slabs_moving_clips(
    time: float | torch.Tensor = 0.0,
    *,
    quadrature_order: int = DEFAULT_QUADRATURE_ORDER,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> ConnectionFixture:
    """Every cut follows one flow, so the complete physical transfer is flat."""

    t = _time_tensor(time, dtype=dtype, device=device)
    speed = t.new_tensor(0.4)
    shift = speed * t
    ray = P0Ray(
        cuts=_vector(t, (0.0, 1.0, 2.7)) + shift,
        extinction=_vector(t, (0.7, 1.2)),
        emission_density=_matrix3(t, ((0.56, 0.07, 0.0), (0.0, 0.24, 1.08))),
    )
    rate = P0RayRate(
        cut_velocity=speed.expand(ray.run_count + 1),
        extinction_time=torch.zeros_like(ray.extinction),
        emission_density_time=torch.zeros_like(ray.emission_density),
    )
    flow = _constant_flow(
        ray,
        quadrature_order=quadrature_order,
        per_run_value=speed.expand(ray.run_count),
    )
    return ConnectionFixture(
        name="advected_colored_slabs_moving_clips",
        description="Differently colored slabs and both clips advect together.",
        time=t,
        ray=ray,
        rate=rate,
        flow=flow,
        endpoint_history=_empty_history(t),
        flow_declaration=_declaration(run_count=ray.run_count, maximum_abs_speed=0.5),
        quadrature_order=quadrature_order,
        expected_flow_admissible=True,
    )


def boundary_flow_mismatch(
    time: float | torch.Tensor = 0.0,
    *,
    quadrature_order: int = DEFAULT_QUADRATURE_ORDER,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> ConnectionFixture:
    """Moving interface with ``r_dot != w`` and fixed clips."""

    t = _time_tensor(time, dtype=dtype, device=device)
    if not bool(t >= 0.0):
        raise ValueError("boundary mismatch fixture requires t >= 0")
    interface_speed = t.new_tensor(0.6)
    flow_speed = t.new_tensor(0.2)
    cut = 1.0 + interface_speed * t
    if not bool((cut > 0.0) & (cut < 3.0)):
        raise ValueError("boundary mismatch left its stable chart")
    ray = P0Ray(
        cuts=torch.stack((t.new_tensor(0.0), cut, t.new_tensor(3.0))),
        extinction=_vector(t, (0.8, 1.3)),
        emission_density=_matrix3(t, ((0.64, 0.08, 0.0), (0.0, 0.26, 1.04))),
    )
    rate = P0RayRate(
        cut_velocity=torch.stack((t.new_tensor(0.0), interface_speed, t.new_tensor(0.0))),
        extinction_time=torch.zeros_like(ray.extinction),
        emission_density_time=torch.zeros_like(ray.emission_density),
    )
    flow = _constant_flow(
        ray,
        quadrature_order=quadrature_order,
        per_run_value=flow_speed.expand(ray.run_count),
    )
    endpoint_history, _ = derive_constant_endpoint_history(
        ray,
        rate,
        flow,
        duration=t,
    )
    return ConnectionFixture(
        name="boundary_flow_mismatch",
        description="Tests the full (w-r_dot)[A] atom plus fixed-clip flux.",
        time=t,
        ray=ray,
        rate=rate,
        flow=flow,
        endpoint_history=endpoint_history,
        flow_declaration=_declaration(run_count=ray.run_count, maximum_abs_speed=0.3),
        quadrature_order=quadrature_order,
        expected_flow_admissible=True,
    )


def material_evolution(
    time: float | torch.Tensor = 0.0,
    *,
    quadrature_order: int = DEFAULT_QUADRATURE_ORDER,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> ConnectionFixture:
    """One fixed run with nonzero bulk and zero singular curvature."""

    t = _time_tensor(time, dtype=dtype, device=device)
    extinction = 1.0 + 0.2 * t
    emission = torch.stack((0.55 + 0.04 * t, 0.2 - 0.02 * t, 0.1 + 0.01 * t))
    ray = P0Ray(
        cuts=_vector(t, (0.0, 2.0)),
        extinction=extinction.reshape(1),
        emission_density=emission.reshape(1, 3),
    )
    rate = P0RayRate(
        cut_velocity=torch.zeros_like(ray.cuts),
        extinction_time=t.new_tensor((0.2,)),
        emission_density_time=t.new_tensor(((0.04, -0.02, 0.01),)),
    )
    flow = _constant_flow(
        ray,
        quadrature_order=quadrature_order,
        per_run_value=torch.zeros_like(ray.extinction),
    )
    return ConnectionFixture(
        name="material_evolution",
        description="Eulerian material evolution isolated in bulk curvature.",
        time=t,
        ray=ray,
        rate=rate,
        flow=flow,
        endpoint_history=_empty_history(t),
        flow_declaration=_declaration(run_count=ray.run_count, maximum_abs_speed=1.0),
        quadrature_order=quadrature_order,
        expected_flow_admissible=True,
    )


def discontinuous_flow(
    time: float | torch.Tensor = 0.0,
    *,
    quadrature_order: int = DEFAULT_QUADRATURE_ORDER,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> ConnectionFixture:
    """Static word with discontinuous ``w``; the general ``[wA]`` is required."""

    t = _time_tensor(time, dtype=dtype, device=device)
    ray = P0Ray(
        cuts=_vector(t, (0.0, 1.2, 3.0)),
        extinction=_vector(t, (0.9, 1.4)),
        emission_density=_matrix3(t, ((0.72, 0.0, 0.09), (0.0, 1.12, 0.28))),
    )
    run_flow = _vector(t, (0.15, -0.1))
    flow = _constant_flow(
        ray,
        quadrature_order=quadrature_order,
        per_run_value=run_flow,
    )
    endpoint_history, _ = derive_constant_endpoint_history(
        ray,
        _zero_rate(ray),
        flow,
        duration=t,
    )
    return ConnectionFixture(
        name="discontinuous_flow",
        description="Intentional continuity-policy failure with exact BV theorem identity.",
        time=t,
        ray=ray,
        rate=_zero_rate(ray),
        flow=flow,
        endpoint_history=endpoint_history,
        flow_declaration=_declaration(
            run_count=ray.run_count,
            maximum_abs_speed=0.2,
            continuity_tolerance=1.0e-12,
        ),
        quadrature_order=quadrature_order,
        expected_flow_admissible=False,
    )


def flat_translation_fixed_clips(
    time: float | torch.Tensor = 0.0,
    *,
    quadrature_order: int = DEFAULT_QUADRATURE_ORDER,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> ConnectionFixture:
    """Flat translating medium whose physical ``U`` changes only by clipping.

    This is the decisive compression fixture: the direct physical transfer is
    temporally nonlinear, while endpoint transport makes ``U_tilde`` constant
    and ``K_F`` zero on the complete stable chart.
    """

    t = _time_tensor(time, dtype=dtype, device=device)
    if not bool(t >= 0.0):
        raise ValueError("flat-translation fixture requires t >= 0")
    speed = t.new_tensor(0.35)
    first = 0.8 + speed * t
    second = 1.9 + speed * t
    if not bool((first > 0.0) & (first < second) & (second < 3.0)):
        raise ValueError("flat-translation fixture left its stable chart")
    ray = P0Ray(
        cuts=torch.stack((t.new_tensor(0.0), first, second, t.new_tensor(3.0))),
        extinction=_vector(t, (0.6, 1.1, 0.9)),
        emission_density=_matrix3(
            t,
            ((0.54, 0.06, 0.0), (0.11, 0.88, 0.22), (0.0, 0.09, 0.81)),
        ),
    )
    rate = P0RayRate(
        cut_velocity=torch.stack((t.new_tensor(0.0), speed, speed, t.new_tensor(0.0))),
        extinction_time=torch.zeros_like(ray.extinction),
        emission_density_time=torch.zeros_like(ray.emission_density),
    )
    flow = _constant_flow(
        ray,
        quadrature_order=quadrature_order,
        per_run_value=speed.expand(ray.run_count),
    )
    endpoint_history, _ = derive_constant_endpoint_history(
        ray,
        rate,
        flow,
        duration=t,
    )
    return ConnectionFixture(
        name="flat_translation_fixed_clips",
        description="Zero curvature with nonzero endpoint flux and changing direct U.",
        time=t,
        ray=ray,
        rate=rate,
        flow=flow,
        endpoint_history=endpoint_history,
        flow_declaration=_declaration(run_count=ray.run_count, maximum_abs_speed=0.4),
        quadrature_order=quadrature_order,
        expected_flow_admissible=True,
    )


def cosine_depth_curvature_cancellation(
    *,
    sample_count: int = 65,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> SmoothCurvatureCancellationFixture:
    """Nonzero cosine-depth curvature with zero transported total variation.

    At fixed time, ``A_z=(1+epsilon*t*cos(2*pi*z)) X0`` and ``w=0``.
    Thus ``F=epsilon*cos(2*pi*z) X0`` is nonzero pointwise, while every
    generator is collinear and the transported integral vanishes over
    ``[0,1]``.  This prevents the oracle from confusing constant total
    transfer with a flat connection.
    """

    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 3:
        raise ValueError("cosine fixture needs at least three samples")
    cuts = torch.tensor((0.0, 1.0), dtype=dtype, device=device)
    quadrature = gauss_legendre_layout(cuts, sample_count)
    depth = quadrature.depth_nodes[0]
    weights = quadrature.depth_weights[0]
    base_scalar = depth.new_tensor(-0.8)
    base_source = depth.new_tensor((0.48, 0.16, 0.08))
    epsilon = depth.new_tensor(0.25)
    time = depth.new_tensor(0.4)
    cosine = torch.cos(2.0 * torch.pi * depth)
    scale = 1.0 + epsilon * time * cosine
    generator = AffineGenerator(
        scalar=scale * base_scalar,
        source=scale.unsqueeze(-1) * base_source,
    )
    generator_time = AffineGenerator(
        scalar=epsilon * cosine * base_scalar,
        source=(epsilon * cosine).unsqueeze(-1) * base_source,
    )
    flow = torch.zeros_like(depth)
    flow_depth = torch.zeros_like(depth)
    curvature = generator_time
    cumulative_scale = (
        depth
        + epsilon
        * time
        * torch.sin(2.0 * torch.pi * depth)
        / (2.0 * torch.pi)
    )
    total_scale = depth.new_tensor(1.0)
    base = AffineGenerator(base_scalar, base_source)
    prefix = generator_exponential(base, cumulative_scale)
    suffix = generator_exponential(base, total_scale - cumulative_scale)
    integrand = generator_sandwich(prefix, curvature, suffix)
    transported = AffineTransferTangent(
        beta=torch.sum(weights * integrand.beta),
        moment=torch.sum(weights.unsqueeze(-1) * integrand.moment, dim=0),
    )
    return SmoothCurvatureCancellationFixture(
        depth=depth,
        generator=generator,
        generator_time=generator_time,
        flow=flow,
        flow_depth_derivative=flow_depth,
        curvature=curvature,
        transported_curvature_integral=transported,
    )


def sideways_pinhole_lift(
    *,
    depths: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> SensorDepthLiftInput:
    """Central pinhole track with ``V=e_x``: scalar depth flow cannot fit it."""

    if depths is None:
        depths = torch.tensor((0.5, 1.0, 2.0, 4.0), dtype=dtype, device=device)
    if depths.ndim != 1 or not bool(torch.all(depths > 0.0)):
        raise ValueError("sideways-pinhole depths must be a positive vector")
    zeros = torch.zeros_like(depths)
    ones = torch.ones_like(depths)
    return SensorDepthLiftInput(
        gamma_u=torch.stack((depths, zeros, zeros), dim=-1),
        gamma_v=torch.stack((zeros, depths, zeros), dim=-1),
        gamma_z=torch.stack((zeros, zeros, ones), dim=-1),
        gamma_t=torch.zeros((depths.numel(), 3), dtype=depths.dtype, device=depths.device),
        world_velocity=torch.stack((ones, zeros, zeros), dim=-1),
        supplied_depth_flow=zeros,
    )


FixtureBuilder = Callable[..., ConnectionFixture]

FIXTURE_BUILDERS: dict[str, FixtureBuilder] = {
    "front_red_back_blue": front_red_back_blue,
    "moving_noncommuting_boundary": moving_noncommuting_boundary,
    "advected_colored_slabs_moving_clips": advected_colored_slabs_moving_clips,
    "boundary_flow_mismatch": boundary_flow_mismatch,
    "material_evolution": material_evolution,
    "discontinuous_flow": discontinuous_flow,
    "flat_translation_fixed_clips": flat_translation_fixed_clips,
}


def build_fixture(
    name: str,
    time: float | torch.Tensor = 0.0,
    *,
    quadrature_order: int = DEFAULT_QUADRATURE_ORDER,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> ConnectionFixture:
    """Build a named canonical fixture without dynamic import or dispatch magic."""

    try:
        builder = FIXTURE_BUILDERS[name]
    except KeyError as error:
        choices = ", ".join(sorted(FIXTURE_BUILDERS))
        raise ValueError(f"unknown connection fixture {name!r}; choose {choices}") from error
    return builder(
        time,
        quadrature_order=quadrature_order,
        dtype=dtype,
        device=device,
    )


__all__ = [
    "ConnectionFixture",
    "DEFAULT_QUADRATURE_ORDER",
    "FIXTURE_BUILDERS",
    "SmoothCurvatureCancellationFixture",
    "advected_colored_slabs_moving_clips",
    "boundary_flow_mismatch",
    "build_fixture",
    "cosine_depth_curvature_cancellation",
    "discontinuous_flow",
    "flat_translation_fixed_clips",
    "front_red_back_blue",
    "material_evolution",
    "moving_noncommuting_boundary",
    "sideways_pinhole_lift",
]
