"""Reference constrained-Lagrangian connection for a stable P0 ray word.

This module implements the corrected repository-order identity

``dU/dt = U B_far - B_near U + integral prefix F suffix dz``

including general BV interface atoms ``[w A] - r_dot [A]``.  Flow values are
explicit tensors sampled by the caller; no function callback, cache, model,
or production renderer is hidden inside the reference calculation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .affine import (
    AffineGenerator,
    AffineGroupReport,
    AffineTransfer,
    AffineTransferTangent,
    PhysicalConeReport,
    add_generators,
    add_tangents,
    affine_group_report,
    compose,
    compose_jets,
    generator_exponential,
    generator_sandwich,
    identity_transfer,
    inverse,
    left_generator_action,
    physical_cone_report,
    right_generator_action,
    scale_generator,
    segment_time_derivative,
    subtract_generators,
    subtract_tangents,
    tangent_sandwich,
    zero_tangent,
)


@dataclass(frozen=True)
class P0Ray:
    """One exact stable ordered word.

    Shapes are ``cuts [R+1]``, coordinate extinction ``extinction [R]``, and
    emitted-density ``emission_density [R,3]``.  The caller must already have
    included the physical ray-speed Jacobian in the last two tensors.
    """

    cuts: torch.Tensor
    extinction: torch.Tensor
    emission_density: torch.Tensor

    @property
    def run_count(self) -> int:
        return int(self.extinction.numel())

    @property
    def lengths(self) -> torch.Tensor:
        return self.cuts[1:] - self.cuts[:-1]

    @property
    def generators(self) -> AffineGenerator:
        return AffineGenerator(
            scalar=-self.extinction,
            source=self.emission_density,
        )

    def validate(self) -> None:
        _require_vector("ray.cuts", self.cuts)
        _require_vector("ray.extinction", self.extinction)
        _require_matrix3("ray.emission_density", self.emission_density)
        if self.run_count < 1:
            raise ValueError("a P0 ray requires at least one run")
        if self.cuts.shape != (self.run_count + 1,):
            raise ValueError("ray cuts must have shape [R+1]")
        if self.emission_density.shape != (self.run_count, 3):
            raise ValueError("ray emission_density must have shape [R,3]")
        _require_compatible(
            self.cuts,
            self.extinction,
            self.emission_density,
            context="P0 ray",
        )
        _require_finite(self.cuts, self.extinction, self.emission_density)
        if not bool(torch.all(self.cuts[1:] > self.cuts[:-1])):
            raise ValueError("ray cuts must be strictly increasing")
        if not bool(torch.all(self.extinction >= 0.0)):
            raise ValueError("coordinate extinction must be nonnegative")


@dataclass(frozen=True)
class P0RayRate:
    """Eulerian material rates and moving-cut velocities for :class:`P0Ray`.

    Shapes are ``cut_velocity [R+1]``, ``extinction_time [R]``, and
    ``emission_density_time [R,3]``.
    """

    cut_velocity: torch.Tensor
    extinction_time: torch.Tensor
    emission_density_time: torch.Tensor

    @property
    def generator_rate(self) -> AffineGenerator:
        return AffineGenerator(
            scalar=-self.extinction_time,
            source=self.emission_density_time,
        )

    def validate_for(self, ray: P0Ray) -> None:
        _require_vector("rate.cut_velocity", self.cut_velocity)
        _require_vector("rate.extinction_time", self.extinction_time)
        _require_matrix3(
            "rate.emission_density_time",
            self.emission_density_time,
        )
        if self.cut_velocity.shape != (ray.run_count + 1,):
            raise ValueError("cut_velocity must have shape [R+1]")
        if self.extinction_time.shape != (ray.run_count,):
            raise ValueError("extinction_time must have shape [R]")
        if self.emission_density_time.shape != (ray.run_count, 3):
            raise ValueError("emission_density_time must have shape [R,3]")
        _require_compatible(
            ray.cuts,
            self.cut_velocity,
            self.extinction_time,
            self.emission_density_time,
            context="P0 ray rates",
        )
        _require_finite(
            self.cut_velocity,
            self.extinction_time,
            self.emission_density_time,
        )


@dataclass(frozen=True)
class P0Quadrature:
    """Gauss--Legendre nodes and physical-depth weights.

    Shapes are ``local_nodes/local_weights [Q]`` and
    ``depth_nodes/depth_weights [R,Q]``.
    """

    local_nodes: torch.Tensor
    local_weights: torch.Tensor
    depth_nodes: torch.Tensor
    depth_weights: torch.Tensor

    @property
    def order(self) -> int:
        return int(self.local_nodes.numel())


@dataclass(frozen=True)
class P0FlowSamples:
    """Explicit sampled scalar depth flow, with no hidden flow callback.

    ``bulk_value`` and ``bulk_d_dz`` have shape ``[R,Q]`` at the nodes from
    :func:`gauss_legendre_layout`.  ``cell_left_value`` and
    ``cell_right_value`` have shape ``[R]`` and retain both one-sided traces,
    allowing the general discontinuous-flow BV atom.
    """

    bulk_value: torch.Tensor
    bulk_d_dz: torch.Tensor
    cell_left_value: torch.Tensor
    cell_right_value: torch.Tensor

    @property
    def quadrature_order(self) -> int:
        return int(self.bulk_value.shape[1])

    def validate_for(self, ray: P0Ray, quadrature: P0Quadrature) -> None:
        _require_matrix("flow.bulk_value", self.bulk_value)
        _require_matrix("flow.bulk_d_dz", self.bulk_d_dz)
        _require_vector("flow.cell_left_value", self.cell_left_value)
        _require_vector("flow.cell_right_value", self.cell_right_value)
        expected_bulk = (ray.run_count, quadrature.order)
        if self.bulk_value.shape != expected_bulk:
            raise ValueError("flow.bulk_value must have shape [R,Q]")
        if self.bulk_d_dz.shape != expected_bulk:
            raise ValueError("flow.bulk_d_dz must have shape [R,Q]")
        if self.cell_left_value.shape != (ray.run_count,):
            raise ValueError("flow.cell_left_value must have shape [R]")
        if self.cell_right_value.shape != (ray.run_count,):
            raise ValueError("flow.cell_right_value must have shape [R]")
        _require_compatible(
            ray.cuts,
            self.bulk_value,
            self.bulk_d_dz,
            self.cell_left_value,
            self.cell_right_value,
            context="P0 flow samples",
        )
        _require_finite(
            self.bulk_value,
            self.bulk_d_dz,
            self.cell_left_value,
            self.cell_right_value,
        )


@dataclass(frozen=True)
class EndpointTransportHistory:
    """Piecewise-constant endpoint generators integrated from ``t0``.

    All scalar arrays have shape ``[K]`` and sources ``[K,3]``.  Each step
    performs ``H <- H exp(duration * B)`` as required by ``dot H = H B``.
    Empty ``K=0`` histories produce identity endpoint transports.
    """

    durations: torch.Tensor
    near_scalar: torch.Tensor
    near_source: torch.Tensor
    far_scalar: torch.Tensor
    far_source: torch.Tensor

    @property
    def step_count(self) -> int:
        return int(self.durations.numel())

    def validate_for(self, ray: P0Ray) -> None:
        _require_vector("endpoint_history.durations", self.durations)
        _require_vector("endpoint_history.near_scalar", self.near_scalar)
        _require_matrix3("endpoint_history.near_source", self.near_source)
        _require_vector("endpoint_history.far_scalar", self.far_scalar)
        _require_matrix3("endpoint_history.far_source", self.far_source)
        if self.near_scalar.shape != (self.step_count,):
            raise ValueError("near_scalar must have shape [K]")
        if self.near_source.shape != (self.step_count, 3):
            raise ValueError("near_source must have shape [K,3]")
        if self.far_scalar.shape != (self.step_count,):
            raise ValueError("far_scalar must have shape [K]")
        if self.far_source.shape != (self.step_count, 3):
            raise ValueError("far_source must have shape [K,3]")
        _require_compatible(
            ray.cuts,
            self.durations,
            self.near_scalar,
            self.near_source,
            self.far_scalar,
            self.far_source,
            context="endpoint history",
        )
        _require_finite(
            self.durations,
            self.near_scalar,
            self.near_source,
            self.far_scalar,
            self.far_source,
        )


@dataclass(frozen=True)
class FlowAdmissibilityDeclaration:
    """Auditable static receipt for the origin and capacity of ``w``."""

    temporal_dof: int
    source_motion_temporal_dof: int
    retained_bytes: int
    shared_scene_camera_model: bool
    source_motion_retained_bytes: int | None = None
    uses_per_ray_answer_table: bool = False
    uses_per_frame_answer_table: bool = False
    uses_target_or_transfer_conditioning: bool = False
    maximum_abs_speed: float = float("inf")
    maximum_abs_depth_gradient: float = float("inf")
    continuity_tolerance: float = 1.0e-9
    orientation_horizon: float = 1.0

    def validate(self) -> None:
        if self.temporal_dof < 0 or self.source_motion_temporal_dof < 0:
            raise ValueError("flow temporal degrees of freedom must be nonnegative")
        if self.retained_bytes < 0:
            raise ValueError("flow retained_bytes must be nonnegative")
        if (
            self.source_motion_retained_bytes is not None
            and self.source_motion_retained_bytes < 0
        ):
            raise ValueError(
                "flow source_motion_retained_bytes must be nonnegative"
            )
        if self.maximum_abs_speed <= 0.0:
            raise ValueError("maximum_abs_speed must be positive")
        if self.maximum_abs_depth_gradient <= 0.0:
            raise ValueError("maximum_abs_depth_gradient must be positive")
        if self.continuity_tolerance < 0.0:
            raise ValueError("continuity_tolerance must be nonnegative")
        if self.orientation_horizon <= 0.0:
            raise ValueError("orientation_horizon must be positive")


@dataclass(frozen=True)
class FlowAdmissibilityReport:
    maximum_observed_abs_speed: torch.Tensor
    maximum_observed_abs_depth_gradient: torch.Tensor
    maximum_internal_trace_jump: torch.Tensor
    minimum_euler_orientation_margin: torch.Tensor
    ode_jacobian_lower_bound: torch.Tensor
    declared_retained_bytes: int
    source_motion_retained_bytes: int | None
    temporal_capacity_ratio: float
    retained_byte_capacity_ratio: float | None
    capacity_passed: bool
    provenance_passed: bool
    bounds_passed: torch.Tensor
    continuity_passed: torch.Tensor
    euler_orientation_passed: torch.Tensor
    passed: torch.Tensor
    probe_grid_only: bool = True
    continuous_bound_certified: bool = False


@dataclass(frozen=True)
class OrderedP0Transport:
    """Exact P0 segment, prefix, suffix, and total transports."""

    segments: AffineTransfer
    prefixes: AffineTransfer
    suffixes: AffineTransfer
    total: AffineTransfer


@dataclass(frozen=True)
class CurvatureBreakdown:
    """Bulk quadrature and exact BV singular contributions."""

    bulk_per_run: AffineTransferTangent
    singular_per_interface: AffineTransferTangent
    bulk_total: AffineTransferTangent
    singular_total: AffineTransferTangent
    total: AffineTransferTangent


@dataclass(frozen=True)
class EndpointKinematics:
    near_generator: AffineGenerator
    far_generator: AffineGenerator
    flux: AffineTransferTangent


@dataclass(frozen=True)
class EndpointTransports:
    near: AffineTransfer
    far: AffineTransfer


@dataclass(frozen=True)
class ConnectionCoreEvaluation:
    """Entire differentiable theorem calculation, excluding pass/fail policy."""

    quadrature: P0Quadrature
    ordered: OrderedP0Transport
    direct_time_derivative: AffineTransferTangent
    curvature: CurvatureBreakdown
    endpoint_kinematics: EndpointKinematics
    predicted_time_derivative: AffineTransferTangent
    theorem_residual: AffineTransferTangent
    endpoint_transports: EndpointTransports
    flow_corrected_transfer: AffineTransfer
    transported_curvature_source: AffineTransferTangent
    algebraically_transported_covariant_derivative: AffineTransferTangent
    covariant_residual: AffineTransferTangent
    reconstructed_transfer: AffineTransfer
    reconstruction_residual: AffineTransferTangent


@dataclass(frozen=True)
class ConnectionEvaluation:
    core: ConnectionCoreEvaluation
    direct_physical_cone: PhysicalConeReport
    flow_corrected_group: AffineGroupReport
    reconstructed_physical_cone: PhysicalConeReport
    near_endpoint_group: AffineGroupReport
    far_endpoint_group: AffineGroupReport
    flow_admissibility: FlowAdmissibilityReport | None


@dataclass(frozen=True)
class SensorDepthLiftInput:
    """Full sensor-depth Jacobian columns and a proposed scalar axial flow.

    Vector fields have shape ``[...,3]`` and ``supplied_depth_flow`` has the
    shared leading shape ``[...]``.
    """

    gamma_u: torch.Tensor
    gamma_v: torch.Tensor
    gamma_z: torch.Tensor
    gamma_t: torch.Tensor
    world_velocity: torch.Tensor
    supplied_depth_flow: torch.Tensor


@dataclass(frozen=True)
class SensorDepthLiftReport:
    full_lift: torch.Tensor
    full_lift_residual_norm: torch.Tensor
    jacobian_determinant: torch.Tensor
    jacobian_condition_number: torch.Tensor
    best_axial_depth_flow: torch.Tensor
    best_axial_residual_norm: torch.Tensor
    supplied_axial_residual_norm: torch.Tensor
    jacobian_nonsingular: torch.Tensor
    full_lift_passed: torch.Tensor
    supplied_scalar_flow_exact: torch.Tensor


def gauss_legendre_layout(cuts: torch.Tensor, order: int) -> P0Quadrature:
    """Return a differentiable per-run Gauss--Legendre layout.

    The order may be any positive integer.  Golub--Welsch nodes are generated
    afresh on the input dtype/device; there is intentionally no global cache.
    """

    _require_vector("cuts", cuts)
    if isinstance(order, bool) or not isinstance(order, int) or order < 1:
        raise ValueError("quadrature order must be a positive integer")
    index = torch.arange(1, order, dtype=cuts.dtype, device=cuts.device)
    off_diagonal = index / torch.sqrt(4.0 * index.square() - 1.0)
    jacobi = torch.zeros((order, order), dtype=cuts.dtype, device=cuts.device)
    if order > 1:
        jacobi = jacobi + torch.diag(off_diagonal, diagonal=1)
        jacobi = jacobi + torch.diag(off_diagonal, diagonal=-1)
    local_nodes, eigenvectors = torch.linalg.eigh(jacobi)
    local_weights = 2.0 * eigenvectors[0].square()
    centers = 0.5 * (cuts[1:] + cuts[:-1])
    half_widths = 0.5 * (cuts[1:] - cuts[:-1])
    return P0Quadrature(
        local_nodes=local_nodes,
        local_weights=local_weights,
        depth_nodes=(
            centers.unsqueeze(-1)
            + half_widths.unsqueeze(-1) * local_nodes.unsqueeze(0)
        ),
        depth_weights=(
            half_widths.unsqueeze(-1) * local_weights.unsqueeze(0)
        ),
    )


def ordered_p0_transport(ray: P0Ray) -> OrderedP0Transport:
    """Compute exact segments and every boundary prefix/suffix."""

    generators = ray.generators
    segments = generator_exponential(generators, ray.lengths)
    identity = identity_transfer(ray.cuts)

    prefix_values = [identity]
    running = identity
    for run_index in range(ray.run_count):
        running = compose(running, _select_transfer(segments, run_index))
        prefix_values.append(running)

    suffix_values: list[AffineTransfer] = [identity] * (ray.run_count + 1)
    running = identity
    suffix_values[ray.run_count] = identity
    for run_index in range(ray.run_count - 1, -1, -1):
        running = compose(_select_transfer(segments, run_index), running)
        suffix_values[run_index] = running

    prefixes = _stack_transfers(prefix_values)
    suffixes = _stack_transfers(suffix_values)
    return OrderedP0Transport(
        segments=segments,
        prefixes=prefixes,
        suffixes=suffixes,
        total=_select_transfer(prefixes, ray.run_count),
    )


def direct_p0_time_derivative(
    ray: P0Ray,
    rate: P0RayRate,
) -> AffineTransferTangent:
    """Differentiate the exact moving P0 word without autograd."""

    lengths = ray.lengths
    length_rates = rate.cut_velocity[1:] - rate.cut_velocity[:-1]
    segments = generator_exponential(ray.generators, lengths)
    segment_rates = segment_time_derivative(
        ray.generators,
        rate.generator_rate,
        lengths,
        length_rates,
    )
    running = identity_transfer(ray.cuts)
    running_rate = zero_tangent(ray.cuts)
    for run_index in range(ray.run_count):
        running, running_rate = compose_jets(
            running,
            running_rate,
            _select_transfer(segments, run_index),
            _select_tangent(segment_rates, run_index),
        )
    return running_rate


def scan_endpoint_transports(
    ray: P0Ray,
    history: EndpointTransportHistory,
) -> EndpointTransports:
    """Integrate both right-acting endpoint ODEs by exact P0 time steps."""

    near = identity_transfer(ray.cuts)
    far = identity_transfer(ray.cuts)
    near_generators = AffineGenerator(history.near_scalar, history.near_source)
    far_generators = AffineGenerator(history.far_scalar, history.far_source)
    near_steps = generator_exponential(near_generators, history.durations)
    far_steps = generator_exponential(far_generators, history.durations)
    for step_index in range(history.step_count):
        near = compose(near, _select_transfer(near_steps, step_index))
        far = compose(far, _select_transfer(far_steps, step_index))
    return EndpointTransports(near=near, far=far)


def integrate_bulk_curvature(
    ray: P0Ray,
    rate: P0RayRate,
    flow: P0FlowSamples,
    quadrature: P0Quadrature,
    ordered: OrderedP0Transport,
) -> tuple[AffineTransferTangent, AffineTransferTangent]:
    """Gauss--Legendre integration of ``prefix F_bulk suffix``.

    On each P0 run, ``A_z`` is coordinate-depth constant, hence
    ``F_bulk = d_t A_z + (d_z w) A_z``.  ``w`` itself enters the exact BV
    interfaces and endpoints below; only its depth derivative enters here.
    """

    bulk_generator = add_generators(
        AffineGenerator(
            scalar=rate.generator_rate.scalar.unsqueeze(-1),
            source=rate.generator_rate.source.unsqueeze(-2),
        ),
        AffineGenerator(
            scalar=(
                flow.bulk_d_dz
                * ray.generators.scalar.unsqueeze(-1)
            ),
            source=(
                flow.bulk_d_dz.unsqueeze(-1)
                * ray.generators.source.unsqueeze(-2)
            ),
        ),
    )

    per_run: list[AffineTransferTangent] = []
    for run_index in range(ray.run_count):
        generator = _select_generator(ray.generators, run_index)
        node_depths = quadrature.depth_nodes[run_index]
        left_lengths = node_depths - ray.cuts[run_index]
        right_lengths = ray.cuts[run_index + 1] - node_depths
        prefix = compose(
            _select_transfer(ordered.prefixes, run_index),
            generator_exponential(generator, left_lengths),
        )
        suffix = compose(
            generator_exponential(generator, right_lengths),
            _select_transfer(ordered.suffixes, run_index + 1),
        )
        integrand = generator_sandwich(
            prefix,
            _select_generator(bulk_generator, run_index),
            suffix,
        )
        weights = quadrature.depth_weights[run_index]
        per_run.append(
            AffineTransferTangent(
                beta=torch.sum(weights * integrand.beta),
                moment=torch.sum(
                    weights.unsqueeze(-1) * integrand.moment,
                    dim=0,
                ),
            )
        )
    stacked = _stack_tangents(per_run)
    return stacked, AffineTransferTangent(
        beta=torch.sum(stacked.beta),
        moment=torch.sum(stacked.moment, dim=0),
    )


def integrate_singular_curvature(
    ray: P0Ray,
    rate: P0RayRate,
    flow: P0FlowSamples,
    ordered: OrderedP0Transport,
) -> tuple[AffineTransferTangent, AffineTransferTangent]:
    """Integrate exact general BV atoms at all internal moving interfaces.

    At interface ``r`` this uses

    ``[w A] - r_dot [A] = w_plus A_plus - w_minus A_minus
                               - r_dot (A_plus - A_minus)``.

    It intentionally does not replace the two one-sided flow traces by one
    value, so discontinuous-flow diagnostics exercise the full theorem.
    """

    contributions: list[AffineTransferTangent] = []
    generators = ray.generators
    for boundary_index in range(1, ray.run_count):
        left_index = boundary_index - 1
        right_index = boundary_index
        left_generator = _select_generator(generators, left_index)
        right_generator = _select_generator(generators, right_index)
        cut_velocity = rate.cut_velocity[boundary_index]
        singular_generator = subtract_generators(
            subtract_generators(
                scale_generator(
                    flow.cell_left_value[right_index],
                    right_generator,
                ),
                scale_generator(
                    flow.cell_right_value[left_index],
                    left_generator,
                ),
            ),
            scale_generator(
                cut_velocity,
                subtract_generators(right_generator, left_generator),
            ),
        )
        contributions.append(
            generator_sandwich(
                _select_transfer(ordered.prefixes, boundary_index),
                singular_generator,
                _select_transfer(ordered.suffixes, boundary_index),
            )
        )

    if contributions:
        stacked = _stack_tangents(contributions)
        total = AffineTransferTangent(
            beta=torch.sum(stacked.beta),
            moment=torch.sum(stacked.moment, dim=0),
        )
        return stacked, total
    empty = AffineTransferTangent(
        beta=torch.empty((0,), dtype=ray.cuts.dtype, device=ray.cuts.device),
        moment=torch.empty((0, 3), dtype=ray.cuts.dtype, device=ray.cuts.device),
    )
    return empty, zero_tangent(ray.cuts)


def endpoint_kinematics(
    ray: P0Ray,
    rate: P0RayRate,
    flow: P0FlowSamples,
    total: AffineTransfer,
) -> EndpointKinematics:
    """Compute ``B=A_t+z_dot A_z=(z_dot-w)A_z`` and endpoint flux."""

    near = scale_generator(
        rate.cut_velocity[0] - flow.cell_left_value[0],
        _select_generator(ray.generators, 0),
    )
    far = scale_generator(
        rate.cut_velocity[-1] - flow.cell_right_value[-1],
        _select_generator(ray.generators, ray.run_count - 1),
    )
    flux = subtract_tangents(
        right_generator_action(total, far),
        left_generator_action(near, total),
    )
    return EndpointKinematics(
        near_generator=near,
        far_generator=far,
        flux=flux,
    )


def evaluate_connection_core(
    ray: P0Ray,
    rate: P0RayRate,
    flow: P0FlowSamples,
    endpoint_history: EndpointTransportHistory,
    *,
    quadrature_order: int,
) -> ConnectionCoreEvaluation:
    """Evaluate every differentiable connection identity without policy checks.

    Call :func:`evaluate_connection` for shape/value validation and reports.
    This lower-level entry point is public so selected-direction transforms can
    avoid differentiating through discrete pass/fail and condition-number
    diagnostics.
    """

    quadrature = gauss_legendre_layout(ray.cuts, quadrature_order)
    ordered = ordered_p0_transport(ray)
    direct_derivative = direct_p0_time_derivative(ray, rate)
    bulk_per_run, bulk_total = integrate_bulk_curvature(
        ray,
        rate,
        flow,
        quadrature,
        ordered,
    )
    singular_per_interface, singular_total = integrate_singular_curvature(
        ray,
        rate,
        flow,
        ordered,
    )
    curvature_total = add_tangents(bulk_total, singular_total)
    curvature = CurvatureBreakdown(
        bulk_per_run=bulk_per_run,
        singular_per_interface=singular_per_interface,
        bulk_total=bulk_total,
        singular_total=singular_total,
        total=curvature_total,
    )
    endpoints = endpoint_kinematics(ray, rate, flow, ordered.total)
    predicted_derivative = add_tangents(endpoints.flux, curvature_total)
    theorem_residual = subtract_tangents(
        direct_derivative,
        predicted_derivative,
    )

    endpoint_transports = scan_endpoint_transports(ray, endpoint_history)
    far_inverse = inverse(endpoint_transports.far)
    flow_corrected = compose(
        compose(endpoint_transports.near, ordered.total),
        far_inverse,
    )
    transported_curvature = tangent_sandwich(
        endpoint_transports.near,
        curvature_total,
        far_inverse,
    )

    direct_covariant = add_tangents(
        subtract_tangents(
            direct_derivative,
            right_generator_action(
                ordered.total,
                endpoints.far_generator,
            ),
        ),
        left_generator_action(
            endpoints.near_generator,
            ordered.total,
        ),
    )
    transported_covariant = tangent_sandwich(
        endpoint_transports.near,
        direct_covariant,
        far_inverse,
    )
    covariant_residual = subtract_tangents(
        transported_covariant,
        transported_curvature,
    )

    reconstructed = compose(
        compose(inverse(endpoint_transports.near), flow_corrected),
        endpoint_transports.far,
    )
    reconstruction_residual = _transfer_difference(
        reconstructed,
        ordered.total,
    )
    return ConnectionCoreEvaluation(
        quadrature=quadrature,
        ordered=ordered,
        direct_time_derivative=direct_derivative,
        curvature=curvature,
        endpoint_kinematics=endpoints,
        predicted_time_derivative=predicted_derivative,
        theorem_residual=theorem_residual,
        endpoint_transports=endpoint_transports,
        flow_corrected_transfer=flow_corrected,
        transported_curvature_source=transported_curvature,
        algebraically_transported_covariant_derivative=transported_covariant,
        covariant_residual=covariant_residual,
        reconstructed_transfer=reconstructed,
        reconstruction_residual=reconstruction_residual,
    )


def evaluate_connection(
    ray: P0Ray,
    rate: P0RayRate,
    flow: P0FlowSamples,
    endpoint_history: EndpointTransportHistory,
    *,
    quadrature_order: int,
    flow_declaration: FlowAdmissibilityDeclaration | None = None,
    cone_tolerance: float = 1.0e-9,
    minimum_group_beta: float = 0.0,
) -> ConnectionEvaluation:
    """Validated, fail-closed reference evaluation of the latest algorithm."""

    ray.validate()
    rate.validate_for(ray)
    endpoint_history.validate_for(ray)
    quadrature = gauss_legendre_layout(ray.cuts, quadrature_order)
    flow.validate_for(ray, quadrature)
    if cone_tolerance < 0.0:
        raise ValueError("cone_tolerance must be nonnegative")
    if minimum_group_beta < 0.0:
        raise ValueError("minimum_group_beta must be nonnegative")
    core = evaluate_connection_core(
        ray,
        rate,
        flow,
        endpoint_history,
        quadrature_order=quadrature_order,
    )
    flow_report = (
        diagnose_flow_admissibility(flow, flow_declaration)
        if flow_declaration is not None
        else None
    )
    return ConnectionEvaluation(
        core=core,
        direct_physical_cone=physical_cone_report(
            core.ordered.total,
            tolerance=cone_tolerance,
            minimum_beta=minimum_group_beta,
        ),
        flow_corrected_group=affine_group_report(
            core.flow_corrected_transfer,
            minimum_beta=minimum_group_beta,
        ),
        reconstructed_physical_cone=physical_cone_report(
            core.reconstructed_transfer,
            tolerance=cone_tolerance,
            minimum_beta=minimum_group_beta,
        ),
        near_endpoint_group=affine_group_report(
            core.endpoint_transports.near,
            minimum_beta=minimum_group_beta,
        ),
        far_endpoint_group=affine_group_report(
            core.endpoint_transports.far,
            minimum_beta=minimum_group_beta,
        ),
        flow_admissibility=flow_report,
    )


def diagnose_flow_admissibility(
    flow: P0FlowSamples,
    declaration: FlowAdmissibilityDeclaration,
) -> FlowAdmissibilityReport:
    """Audit boundedness, continuity, local orientation, capacity, provenance."""

    declaration.validate()
    observed_values = torch.cat(
        (
            flow.bulk_value.reshape(-1),
            flow.cell_left_value,
            flow.cell_right_value,
        )
    )
    max_speed = torch.amax(torch.abs(observed_values))
    max_gradient = torch.amax(torch.abs(flow.bulk_d_dz))
    if flow.cell_left_value.numel() > 1:
        trace_jump = torch.amax(
            torch.abs(
                flow.cell_left_value[1:] - flow.cell_right_value[:-1]
            )
        )
    else:
        trace_jump = torch.zeros_like(max_speed)
    horizon = declaration.orientation_horizon
    euler_margin = torch.amin(1.0 + horizon * flow.bulk_d_dz)
    ode_lower_bound = torch.exp(-horizon * max_gradient)
    bounds_passed = (
        (max_speed <= declaration.maximum_abs_speed)
        & (max_gradient <= declaration.maximum_abs_depth_gradient)
    )
    continuity_passed = trace_jump <= declaration.continuity_tolerance
    euler_passed = euler_margin > 0.0
    temporal_capacity_passed = (
        declaration.temporal_dof <= declaration.source_motion_temporal_dof
    )
    if declaration.source_motion_temporal_dof == 0:
        temporal_capacity_ratio = (
            0.0 if declaration.temporal_dof == 0 else float("inf")
        )
    else:
        temporal_capacity_ratio = (
            declaration.temporal_dof
            / declaration.source_motion_temporal_dof
        )
    if declaration.source_motion_retained_bytes is None:
        retained_byte_capacity_ratio = None
        retained_byte_capacity_passed = True
    elif declaration.source_motion_retained_bytes == 0:
        retained_byte_capacity_ratio = (
            0.0 if declaration.retained_bytes == 0 else float("inf")
        )
        retained_byte_capacity_passed = declaration.retained_bytes == 0
    else:
        retained_byte_capacity_ratio = (
            declaration.retained_bytes
            / declaration.source_motion_retained_bytes
        )
        retained_byte_capacity_passed = (
            declaration.retained_bytes
            <= declaration.source_motion_retained_bytes
        )
    capacity_passed = (
        temporal_capacity_passed and retained_byte_capacity_passed
    )
    provenance_passed = (
        declaration.shared_scene_camera_model
        and not declaration.uses_per_ray_answer_table
        and not declaration.uses_per_frame_answer_table
        and not declaration.uses_target_or_transfer_conditioning
    )
    passed = (
        bounds_passed
        & continuity_passed
        & euler_passed
        & capacity_passed
        & provenance_passed
    )
    return FlowAdmissibilityReport(
        maximum_observed_abs_speed=max_speed,
        maximum_observed_abs_depth_gradient=max_gradient,
        maximum_internal_trace_jump=trace_jump,
        minimum_euler_orientation_margin=euler_margin,
        ode_jacobian_lower_bound=ode_lower_bound,
        declared_retained_bytes=declaration.retained_bytes,
        source_motion_retained_bytes=(
            declaration.source_motion_retained_bytes
        ),
        temporal_capacity_ratio=temporal_capacity_ratio,
        retained_byte_capacity_ratio=retained_byte_capacity_ratio,
        capacity_passed=capacity_passed,
        provenance_passed=provenance_passed,
        bounds_passed=bounds_passed,
        continuity_passed=continuity_passed,
        euler_orientation_passed=euler_passed,
        passed=passed,
        probe_grid_only=True,
        continuous_bound_certified=False,
    )


def diagnose_sensor_depth_lift(
    inputs: SensorDepthLiftInput,
    *,
    singular_value_tolerance: float = 1.0e-10,
    residual_tolerance: float = 1.0e-8,
) -> SensorDepthLiftReport:
    """Compare the exact full sensor-depth lift with a scalar axial flow.

    The Jacobian columns are ``[Gamma_u, Gamma_v, Gamma_z]`` and the target is
    ``V-Gamma_t``.  A thresholded SVD gives the exact inverse on regular
    samples and a diagnostic pseudoinverse on singular samples, which remain
    explicitly failed by ``jacobian_nonsingular``.
    """

    _validate_lift_inputs(inputs)
    if singular_value_tolerance <= 0.0:
        raise ValueError("singular_value_tolerance must be positive")
    if residual_tolerance < 0.0:
        raise ValueError("residual_tolerance must be nonnegative")
    jacobian = torch.stack(
        (inputs.gamma_u, inputs.gamma_v, inputs.gamma_z),
        dim=-1,
    )
    target = inputs.world_velocity - inputs.gamma_t
    left, singular_values, right_h = torch.linalg.svd(
        jacobian,
        full_matrices=False,
    )
    safe_inverse = torch.where(
        singular_values > singular_value_tolerance,
        torch.reciprocal(singular_values),
        torch.zeros_like(singular_values),
    )
    left_coordinates = torch.matmul(
        left.transpose(-2, -1),
        target.unsqueeze(-1),
    ).squeeze(-1)
    full_lift = torch.matmul(
        right_h.transpose(-2, -1),
        (safe_inverse * left_coordinates).unsqueeze(-1),
    ).squeeze(-1)
    full_reconstruction = torch.matmul(
        jacobian,
        full_lift.unsqueeze(-1),
    ).squeeze(-1)
    full_residual = torch.linalg.vector_norm(
        full_reconstruction - target,
        dim=-1,
    )
    determinant = torch.linalg.det(jacobian)
    minimum_singular = singular_values[..., -1]
    maximum_singular = singular_values[..., 0]
    condition = maximum_singular / torch.clamp_min(
        minimum_singular,
        singular_value_tolerance,
    )

    axial_denominator = torch.sum(inputs.gamma_z.square(), dim=-1)
    safe_axial_denominator = torch.clamp_min(
        axial_denominator,
        singular_value_tolerance,
    )
    best_axial = torch.sum(target * inputs.gamma_z, dim=-1) / safe_axial_denominator
    best_axial_residual = torch.linalg.vector_norm(
        target - best_axial.unsqueeze(-1) * inputs.gamma_z,
        dim=-1,
    )
    supplied_axial_residual = torch.linalg.vector_norm(
        target
        - inputs.supplied_depth_flow.unsqueeze(-1) * inputs.gamma_z,
        dim=-1,
    )
    nonsingular = minimum_singular > singular_value_tolerance
    return SensorDepthLiftReport(
        full_lift=full_lift,
        full_lift_residual_norm=full_residual,
        jacobian_determinant=determinant,
        jacobian_condition_number=condition,
        best_axial_depth_flow=best_axial,
        best_axial_residual_norm=best_axial_residual,
        supplied_axial_residual_norm=supplied_axial_residual,
        jacobian_nonsingular=nonsingular,
        full_lift_passed=nonsingular & (full_residual <= residual_tolerance),
        supplied_scalar_flow_exact=(
            axial_denominator > singular_value_tolerance
        )
        & (supplied_axial_residual <= residual_tolerance),
    )


def _select_transfer(transfer: AffineTransfer, index: int) -> AffineTransfer:
    return AffineTransfer(
        beta=transfer.beta[index],
        moment=transfer.moment[index],
    )


def _select_tangent(
    tangent: AffineTransferTangent,
    index: int,
) -> AffineTransferTangent:
    return AffineTransferTangent(
        beta=tangent.beta[index],
        moment=tangent.moment[index],
    )


def _select_generator(
    generator: AffineGenerator,
    index: int,
) -> AffineGenerator:
    return AffineGenerator(
        scalar=generator.scalar[index],
        source=generator.source[index],
    )


def _stack_transfers(values: list[AffineTransfer]) -> AffineTransfer:
    return AffineTransfer(
        beta=torch.stack([value.beta for value in values], dim=0),
        moment=torch.stack([value.moment for value in values], dim=0),
    )


def _stack_tangents(
    values: list[AffineTransferTangent],
) -> AffineTransferTangent:
    return AffineTransferTangent(
        beta=torch.stack([value.beta for value in values], dim=0),
        moment=torch.stack([value.moment for value in values], dim=0),
    )


def _transfer_difference(
    left: AffineTransfer,
    right: AffineTransfer,
) -> AffineTransferTangent:
    return AffineTransferTangent(
        beta=left.beta - right.beta,
        moment=left.moment - right.moment,
    )


def _validate_lift_inputs(inputs: SensorDepthLiftInput) -> None:
    vectors = (
        inputs.gamma_u,
        inputs.gamma_v,
        inputs.gamma_z,
        inputs.gamma_t,
        inputs.world_velocity,
    )
    for name, vector in zip(
        (
            "gamma_u",
            "gamma_v",
            "gamma_z",
            "gamma_t",
            "world_velocity",
        ),
        vectors,
        strict=True,
    ):
        _require_leading_vector3(name, vector)
    _require_float_tensor("supplied_depth_flow", inputs.supplied_depth_flow)
    if any(vector.shape != vectors[0].shape for vector in vectors[1:]):
        raise ValueError("all sensor-depth lift vectors must share shape [...,3]")
    if inputs.supplied_depth_flow.shape != vectors[0].shape[:-1]:
        raise ValueError("supplied_depth_flow must have the vectors' leading shape")
    _require_compatible(
        *vectors,
        inputs.supplied_depth_flow,
        context="sensor-depth lift",
    )
    _require_finite(*vectors, inputs.supplied_depth_flow)


def _require_vector(name: str, tensor: torch.Tensor) -> None:
    _require_float_tensor(name, tensor)
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be rank one")


def _require_matrix(name: str, tensor: torch.Tensor) -> None:
    _require_float_tensor(name, tensor)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be rank two")


def _require_matrix3(name: str, tensor: torch.Tensor) -> None:
    _require_matrix(name, tensor)
    if tensor.shape[-1] != 3:
        raise ValueError(f"{name} must have final dimension 3")


def _require_leading_vector3(name: str, tensor: torch.Tensor) -> None:
    _require_float_tensor(name, tensor)
    if tensor.ndim < 1 or tensor.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [...,3]")


def _require_float_tensor(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if tensor.dtype not in {torch.float32, torch.float64}:
        raise TypeError(f"{name} must use float32 or float64")


def _require_compatible(*tensors: torch.Tensor, context: str) -> None:
    reference = tensors[0]
    for tensor in tensors[1:]:
        if tensor.dtype != reference.dtype:
            raise TypeError(f"all {context} tensors must share a dtype")
        if tensor.device != reference.device:
            raise ValueError(f"all {context} tensors must share a device")


def _require_finite(*tensors: torch.Tensor) -> None:
    if any(not bool(torch.isfinite(tensor).all()) for tensor in tensors):
        raise ValueError("connection inputs must be finite")


__all__ = [
    "ConnectionCoreEvaluation",
    "ConnectionEvaluation",
    "CurvatureBreakdown",
    "EndpointKinematics",
    "EndpointTransportHistory",
    "EndpointTransports",
    "FlowAdmissibilityDeclaration",
    "FlowAdmissibilityReport",
    "OrderedP0Transport",
    "P0FlowSamples",
    "P0Quadrature",
    "P0Ray",
    "P0RayRate",
    "SensorDepthLiftInput",
    "SensorDepthLiftReport",
    "diagnose_flow_admissibility",
    "diagnose_sensor_depth_lift",
    "direct_p0_time_derivative",
    "endpoint_kinematics",
    "evaluate_connection",
    "evaluate_connection_core",
    "gauss_legendre_layout",
    "integrate_bulk_curvature",
    "integrate_singular_curvature",
    "ordered_p0_transport",
    "scan_endpoint_transports",
]
