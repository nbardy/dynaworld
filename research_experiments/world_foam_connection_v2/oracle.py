"""Deterministic float64 oracle for the new WorldFoam connection branch.

This is the executable decision gate for the mathematics, not a production
renderer.  It evaluates independent moving-word and curvature paths, selected
parameter directions, closed-loop orientation, scalar-flow realizability, and
an equal-family ``U``/``U_tilde``/``K_F`` compression comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from .affine import AffineGenerator
from .connection import (
    EndpointTransportHistory,
    P0FlowSamples,
    P0Ray,
    P0RayRate,
    diagnose_sensor_depth_lift,
    evaluate_connection,
    evaluate_connection_core,
)
from .directions import (
    ConnectionTensorDirection,
    evaluate_selected_direction,
    zero_connection_direction,
)
from .fixtures import (
    FIXTURE_BUILDERS,
    ConnectionFixture,
    build_fixture,
    cosine_depth_curvature_cancellation,
    flat_translation_fixed_clips,
    sideways_pinhole_lift,
)
from .holonomy import positive_rectangle_holonomy
from .representation_benchmark import (
    RepresentationCertificate,
    RepresentationProbeSeries,
    compile_equal_family_representation,
)
from .shared_flow import FlowDomain, SharedChebyshevFlow
from .shared_flow_connection import (
    evaluate_shared_flow_connection_core,
    evaluate_shared_flow_selected_direction,
    zero_shared_flow_connection_direction,
)
from .temporal_atlas import AtlasKind


IDENTITY_TOLERANCE = 1.0e-9
SELECTED_DIRECTION_TOLERANCE = 1.0e-6


@dataclass(frozen=True)
class FixtureOracleResult:
    name: str
    theorem_residual: float
    covariant_residual: float
    reconstruction_residual: float
    direct_time_derivative_norm: float
    endpoint_flux_norm: float
    bulk_curvature_norm: float
    singular_curvature_norm: float
    specific_check: str
    specific_check_error: float
    direct_physical_cone_passed: bool
    reconstructed_physical_cone_passed: bool
    flow_admissibility_matched_expectation: bool
    passed: bool


@dataclass(frozen=True)
class SelectedDirectionOracleResult:
    checked_observable_count: int
    maximum_normalized_central_difference_error: float
    tolerance: float
    finite: bool
    passed: bool


@dataclass(frozen=True)
class CorrectedDerivativeOracleResult:
    time: float
    finite_difference_step: float
    maximum_absolute_error: float
    corrected_derivative_norm: float
    transported_curvature_norm: float
    passed: bool


@dataclass(frozen=True)
class SharedFlowDirectionOracleResult:
    checked_observable_count: int
    maximum_normalized_central_difference_error: float
    covers_cut_resampling: bool
    covers_flow_coefficients: bool
    covers_time_coordinate: bool
    passed: bool


@dataclass(frozen=True)
class HolonomyOracleResult:
    orientation: str
    commutator_norm: float
    maximum_small_rectangle_error: float
    passed: bool


@dataclass(frozen=True)
class LiftOracleResult:
    maximum_full_lift_residual: float
    minimum_scalar_axial_residual: float
    full_lift_passed: bool
    scalar_flow_rejected: bool
    passed: bool


@dataclass(frozen=True)
class PromotionDecision:
    required_payload_improvement: float
    required_ordered_work_improvement: float
    required_measured_time_improvement: float
    k_f_payload_improvement_vs_best_direct: float
    k_f_work_improvement_vs_best_direct: float
    measured_time_improvement: float | None
    promote_native_runtime: bool
    reason: str


@dataclass(frozen=True)
class ConnectionOracleReport:
    schema_version: int
    algorithm: str
    fixture_results: tuple[FixtureOracleResult, ...]
    cosine_pointwise_curvature_maximum: float
    cosine_transported_integral_maximum: float
    cosine_cancellation_passed: bool
    independently_recomputed_corrected_derivative: CorrectedDerivativeOracleResult
    selected_direction: SelectedDirectionOracleResult
    endpoint_history_direction: SelectedDirectionOracleResult
    shared_flow_direction: SharedFlowDirectionOracleResult
    holonomy: HolonomyOracleResult
    lift: LiftOracleResult
    representation_certificates: tuple[RepresentationCertificate, ...]
    promotion: PromotionDecision
    all_reference_correctness_gates_passed: bool
    runtime_promotion_closed: bool
    limitations: tuple[str, ...]


def _maximum_absolute(vector: torch.Tensor) -> float:
    return float(torch.amax(torch.abs(vector)).detach().cpu())


def _norm(vector: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(vector).detach().cpu())


def _front_red_analytic_error(fixture: ConnectionFixture, total: torch.Tensor) -> float:
    lengths = fixture.ray.lengths
    beta_front = torch.exp(-fixture.ray.extinction[0] * lengths[0])
    beta_back = torch.exp(-fixture.ray.extinction[1] * lengths[1])
    color_front = fixture.ray.emission_density[0] / fixture.ray.extinction[0]
    color_back = fixture.ray.emission_density[1] / fixture.ray.extinction[1]
    expected = torch.cat(
        (
            (beta_front * beta_back).reshape(1),
            (1.0 - beta_front) * color_front
            + beta_front * (1.0 - beta_back) * color_back,
        )
    )
    return _maximum_absolute(total - expected)


def _fixture_specific_check(
    fixture: ConnectionFixture,
    evaluation: object,
) -> tuple[str, float, bool]:
    core = evaluation.core
    name = fixture.name
    if name == "front_red_back_blue":
        error = _front_red_analytic_error(
            fixture,
            core.ordered.total.as_vector(),
        )
        return "independent front-over-back affine formula", error, error <= IDENTITY_TOLERANCE
    if name == "moving_noncommuting_boundary":
        error = _maximum_absolute(
            core.direct_time_derivative.as_vector()
            - core.curvature.singular_total.as_vector()
        )
        nonzero = _norm(core.curvature.singular_total.as_vector()) > 1.0e-6
        return "pinned -r_dot[A] noncommuting boundary derivative", error, error <= IDENTITY_TOLERANCE and nonzero
    if name == "advected_colored_slabs_moving_clips":
        error = _maximum_absolute(core.direct_time_derivative.as_vector())
        return "all cuts follow w so direct U is constant", error, error <= IDENTITY_TOLERANCE
    if name == "boundary_flow_mismatch":
        value = _norm(core.curvature.singular_total.as_vector())
        return "r_dot!=w produces a nonzero singular atom", 0.0 if value > 1.0e-6 else value, value > 1.0e-6
    if name == "material_evolution":
        bulk = _norm(core.curvature.bulk_total.as_vector())
        singular = _maximum_absolute(core.curvature.singular_total.as_vector())
        return "material rate is bulk-only", singular, bulk > 1.0e-6 and singular <= IDENTITY_TOLERANCE
    if name == "discontinuous_flow":
        value = _norm(core.curvature.singular_total.as_vector())
        return "discontinuous w requires [wA]", 0.0 if value > 1.0e-6 else value, value > 1.0e-6
    if name == "flat_translation_fixed_clips":
        curvature = _maximum_absolute(core.curvature.total.as_vector())
        endpoint = _norm(core.endpoint_kinematics.flux.as_vector())
        return (
            "flat connection with nonzero fixed-clip endpoint flux",
            curvature,
            curvature <= IDENTITY_TOLERANCE and endpoint > 1.0e-6,
        )
    raise ValueError(f"missing specific check for fixture {name!r}")


def evaluate_fixture(fixture: ConnectionFixture) -> FixtureOracleResult:
    evaluation = evaluate_connection(
        fixture.ray,
        fixture.rate,
        fixture.flow,
        fixture.endpoint_history,
        quadrature_order=fixture.quadrature_order,
        flow_declaration=fixture.flow_declaration,
    )
    core = evaluation.core
    theorem = _maximum_absolute(core.theorem_residual.as_vector())
    covariant = _maximum_absolute(core.covariant_residual.as_vector())
    reconstruction = _maximum_absolute(core.reconstruction_residual.as_vector())
    label, specific_error, specific_passed = _fixture_specific_check(
        fixture,
        evaluation,
    )
    flow_match = bool(
        evaluation.flow_admissibility is not None
        and bool(evaluation.flow_admissibility.passed)
        == fixture.expected_flow_admissible
    )
    passed = (
        theorem <= IDENTITY_TOLERANCE
        and covariant <= IDENTITY_TOLERANCE
        and reconstruction <= IDENTITY_TOLERANCE
        and bool(evaluation.direct_physical_cone.passed)
        and bool(evaluation.reconstructed_physical_cone.passed)
        and flow_match
        and specific_passed
    )
    return FixtureOracleResult(
        name=fixture.name,
        theorem_residual=theorem,
        covariant_residual=covariant,
        reconstruction_residual=reconstruction,
        direct_time_derivative_norm=_norm(core.direct_time_derivative.as_vector()),
        endpoint_flux_norm=_norm(core.endpoint_kinematics.flux.as_vector()),
        bulk_curvature_norm=_norm(core.curvature.bulk_total.as_vector()),
        singular_curvature_norm=_norm(core.curvature.singular_total.as_vector()),
        specific_check=label,
        specific_check_error=specific_error,
        direct_physical_cone_passed=bool(evaluation.direct_physical_cone.passed),
        reconstructed_physical_cone_passed=bool(
            evaluation.reconstructed_physical_cone.passed
        ),
        flow_admissibility_matched_expectation=flow_match,
        passed=bool(passed),
    )


def _perturb_fixture(
    fixture: ConnectionFixture,
    direction: ConnectionTensorDirection,
    scale: float,
) -> tuple[P0Ray, P0RayRate, P0FlowSamples, EndpointTransportHistory]:
    def shifted(value: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
        return value + scale * tangent

    return (
        P0Ray(
            shifted(fixture.ray.cuts, direction.cuts),
            shifted(fixture.ray.extinction, direction.extinction),
            shifted(fixture.ray.emission_density, direction.emission_density),
        ),
        P0RayRate(
            shifted(fixture.rate.cut_velocity, direction.cut_velocity),
            shifted(fixture.rate.extinction_time, direction.extinction_time),
            shifted(
                fixture.rate.emission_density_time,
                direction.emission_density_time,
            ),
        ),
        P0FlowSamples(
            shifted(fixture.flow.bulk_value, direction.flow_bulk_value),
            shifted(fixture.flow.bulk_d_dz, direction.flow_bulk_d_dz),
            shifted(
                fixture.flow.cell_left_value,
                direction.flow_cell_left_value,
            ),
            shifted(
                fixture.flow.cell_right_value,
                direction.flow_cell_right_value,
            ),
        ),
        EndpointTransportHistory(
            shifted(
                fixture.endpoint_history.durations,
                direction.endpoint_durations,
            ),
            shifted(
                fixture.endpoint_history.near_scalar,
                direction.endpoint_near_scalar,
            ),
            shifted(
                fixture.endpoint_history.near_source,
                direction.endpoint_near_source,
            ),
            shifted(
                fixture.endpoint_history.far_scalar,
                direction.endpoint_far_scalar,
            ),
            shifted(
                fixture.endpoint_history.far_source,
                direction.endpoint_far_source,
            ),
        ),
    )


def _observable_vectors(core: object) -> tuple[torch.Tensor, ...]:
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


def _evaluate_explicit_direction(
    fixture: ConnectionFixture,
    direction: ConnectionTensorDirection,
) -> SelectedDirectionOracleResult:
    selected = evaluate_selected_direction(
        fixture.ray,
        fixture.rate,
        fixture.flow,
        fixture.endpoint_history,
        direction,
        quadrature_order=fixture.quadrature_order,
    )
    epsilon = 1.0e-6
    plus = evaluate_connection_core(
        *_perturb_fixture(fixture, direction, epsilon),
        quadrature_order=fixture.quadrature_order,
    )
    minus = evaluate_connection_core(
        *_perturb_fixture(fixture, direction, -epsilon),
        quadrature_order=fixture.quadrature_order,
    )
    finite_differences = tuple(
        (positive - negative) / (2.0 * epsilon)
        for positive, negative in zip(
            _observable_vectors(plus),
            _observable_vectors(minus),
            strict=True,
        )
    )
    selected_tangents = tuple(
        getattr(selected.tangent, name)
        for name in selected.tangent.__dataclass_fields__
    )
    errors = []
    finite = True
    for tangent, finite_difference in zip(
        selected_tangents,
        finite_differences,
        strict=True,
    ):
        finite = finite and bool(torch.isfinite(tangent).all())
        errors.append(
            torch.linalg.vector_norm(tangent - finite_difference)
            / torch.clamp(
                torch.linalg.vector_norm(finite_difference),
                min=1.0,
            )
        )
    maximum = float(torch.amax(torch.stack(errors)).detach().cpu())
    return SelectedDirectionOracleResult(
        checked_observable_count=len(errors),
        maximum_normalized_central_difference_error=maximum,
        tolerance=SELECTED_DIRECTION_TOLERANCE,
        finite=finite,
        passed=finite and maximum <= SELECTED_DIRECTION_TOLERANCE,
    )


def evaluate_selected_direction_oracle() -> SelectedDirectionOracleResult:
    fixture = build_fixture("material_evolution", 0.2)
    direction = replace(
        zero_connection_direction(
            fixture.ray,
            fixture.rate,
            fixture.flow,
            fixture.endpoint_history,
        ),
        extinction=fixture.ray.extinction.new_tensor((0.17,)),
        emission_density=fixture.ray.emission_density.new_tensor(
            ((0.08, -0.03, 0.02),)
        ),
        extinction_time=fixture.rate.extinction_time.new_tensor((0.025,)),
        emission_density_time=fixture.rate.emission_density_time.new_tensor(
            ((0.01, -0.005, 0.002),)
        ),
        flow_bulk_value=torch.full_like(fixture.flow.bulk_value, 0.03),
        flow_bulk_d_dz=torch.full_like(fixture.flow.bulk_d_dz, -0.02),
        flow_cell_left_value=torch.full_like(
            fixture.flow.cell_left_value,
            0.03,
        ),
        flow_cell_right_value=torch.full_like(
            fixture.flow.cell_right_value,
            0.03,
        ),
    )
    return _evaluate_explicit_direction(fixture, direction)


def evaluate_endpoint_history_direction_oracle() -> SelectedDirectionOracleResult:
    fixture = build_fixture("flat_translation_fixed_clips", 0.2)
    direction = replace(
        zero_connection_direction(
            fixture.ray,
            fixture.rate,
            fixture.flow,
            fixture.endpoint_history,
        ),
        endpoint_durations=fixture.endpoint_history.durations.new_tensor((0.03,)),
        endpoint_near_scalar=fixture.endpoint_history.near_scalar.new_tensor((0.04,)),
        endpoint_near_source=fixture.endpoint_history.near_source.new_tensor(
            ((0.02, -0.01, 0.03),)
        ),
        endpoint_far_scalar=fixture.endpoint_history.far_scalar.new_tensor((-0.02,)),
        endpoint_far_source=fixture.endpoint_history.far_source.new_tensor(
            ((-0.01, 0.025, 0.015),)
        ),
    )
    return _evaluate_explicit_direction(fixture, direction)


def evaluate_corrected_derivative_oracle() -> CorrectedDerivativeOracleResult:
    """Differentiate a freshly reconstructed ``H_a U H_b^-1`` in time.

    This does not reuse the algebraic covariant-derivative expression inside
    the core and therefore catches endpoint-history scan sign/order mistakes.
    """

    time = 0.2
    epsilon = 1.0e-6
    values = []
    for query_time in (time - epsilon, time, time + epsilon):
        fixture = flat_translation_fixed_clips(query_time)
        values.append(
            evaluate_connection_core(
                fixture.ray,
                fixture.rate,
                fixture.flow,
                fixture.endpoint_history,
                quadrature_order=fixture.quadrature_order,
            )
        )
    finite_difference = (
        values[2].flow_corrected_transfer.as_vector()
        - values[0].flow_corrected_transfer.as_vector()
    ) / (2.0 * epsilon)
    expected = values[1].transported_curvature_source.as_vector()
    error = _maximum_absolute(finite_difference - expected)
    return CorrectedDerivativeOracleResult(
        time=time,
        finite_difference_step=epsilon,
        maximum_absolute_error=error,
        corrected_derivative_norm=_norm(finite_difference),
        transported_curvature_norm=_norm(expected),
        passed=error <= SELECTED_DIRECTION_TOLERANCE,
    )


def evaluate_shared_flow_direction_oracle() -> SharedFlowDirectionOracleResult:
    fixture = build_fixture("front_red_back_blue", 0.0)
    flow = SharedChebyshevFlow(
        domain=FlowDomain(t_min=0.0, t_max=1.0, z_min=0.0, z_max=3.0),
        temporal_degree=1,
        depth_degree=2,
        maximum_speed=0.5,
        dtype=torch.float64,
    )
    with torch.no_grad():
        flow.coefficients.copy_(
            flow.coefficients.new_tensor(
                ((0.02, -0.03, 0.01), (0.04, 0.02, -0.015))
            )
        )
    time = torch.tensor(0.3, dtype=torch.float64)
    direction = replace(
        zero_shared_flow_connection_direction(
            fixture.ray,
            fixture.rate,
            time=time,
            flow=flow,
            endpoint_history=fixture.endpoint_history,
        ),
        cuts=fixture.ray.cuts.new_tensor((0.0, 0.04, 0.0)),
        extinction=fixture.ray.extinction.new_tensor((0.03, -0.02)),
        emission_density=fixture.ray.emission_density.new_tensor(
            ((0.01, 0.005, 0.0), (0.0, -0.004, 0.008))
        ),
        time=time.new_tensor(0.07),
        flow_coefficients=flow.coefficients.new_tensor(
            ((0.01, -0.02, 0.005), (-0.015, 0.01, 0.02))
        ),
    )
    selected = evaluate_shared_flow_selected_direction(
        fixture.ray,
        fixture.rate,
        time=time,
        flow=flow,
        endpoint_history=fixture.endpoint_history,
        direction=direction,
        quadrature_order=fixture.quadrature_order,
    )
    epsilon = 1.0e-6

    def evaluate_offset(scale: float):
        ray = P0Ray(
            fixture.ray.cuts + scale * direction.cuts,
            fixture.ray.extinction + scale * direction.extinction,
            fixture.ray.emission_density + scale * direction.emission_density,
        )
        rate = P0RayRate(
            fixture.rate.cut_velocity + scale * direction.cut_velocity,
            fixture.rate.extinction_time + scale * direction.extinction_time,
            fixture.rate.emission_density_time
            + scale * direction.emission_density_time,
        )
        history = EndpointTransportHistory(
            fixture.endpoint_history.durations
            + scale * direction.endpoint_durations,
            fixture.endpoint_history.near_scalar
            + scale * direction.endpoint_near_scalar,
            fixture.endpoint_history.near_source
            + scale * direction.endpoint_near_source,
            fixture.endpoint_history.far_scalar
            + scale * direction.endpoint_far_scalar,
            fixture.endpoint_history.far_source
            + scale * direction.endpoint_far_source,
        )
        return evaluate_shared_flow_connection_core(
            ray,
            rate,
            time=time + scale * direction.time,
            flow=flow,
            coefficients=(
                flow.coefficients + scale * direction.flow_coefficients
            ),
            endpoint_history=history,
            quadrature_order=fixture.quadrature_order,
        )

    plus = evaluate_offset(epsilon)
    minus = evaluate_offset(-epsilon)
    finite_differences = tuple(
        (positive - negative) / (2.0 * epsilon)
        for positive, negative in zip(
            (
                plus.ordered.total.as_vector(),
                plus.direct_time_derivative.as_vector(),
                plus.endpoint_kinematics.flux.as_vector(),
                plus.curvature.total.as_vector(),
                plus.flow_corrected_transfer.as_vector(),
                plus.transported_curvature_source.as_vector(),
                plus.theorem_residual.as_vector(),
                plus.covariant_residual.as_vector(),
                plus.reconstructed_transfer.as_vector(),
            ),
            (
                minus.ordered.total.as_vector(),
                minus.direct_time_derivative.as_vector(),
                minus.endpoint_kinematics.flux.as_vector(),
                minus.curvature.total.as_vector(),
                minus.flow_corrected_transfer.as_vector(),
                minus.transported_curvature_source.as_vector(),
                minus.theorem_residual.as_vector(),
                minus.covariant_residual.as_vector(),
                minus.reconstructed_transfer.as_vector(),
            ),
            strict=True,
        )
    )
    tangents = tuple(
        getattr(selected.tangent, name)
        for name in selected.tangent.__dataclass_fields__
    )
    errors = tuple(
        torch.linalg.vector_norm(tangent - finite_difference)
        / torch.clamp(torch.linalg.vector_norm(finite_difference), min=1.0)
        for tangent, finite_difference in zip(
            tangents,
            finite_differences,
            strict=True,
        )
    )
    maximum = float(torch.amax(torch.stack(errors)).detach().cpu())
    return SharedFlowDirectionOracleResult(
        checked_observable_count=len(errors),
        maximum_normalized_central_difference_error=maximum,
        covers_cut_resampling=True,
        covers_flow_coefficients=True,
        covers_time_coordinate=True,
        passed=maximum <= SELECTED_DIRECTION_TOLERANCE,
    )


def evaluate_holonomy_oracle() -> HolonomyOracleResult:
    reference = torch.tensor(0.0, dtype=torch.float64)
    depth = AffineGenerator(
        scalar=reference.new_tensor(-1.0),
        source=reference.new_tensor((1.0, 0.0, 0.0)),
    )
    temporal = AffineGenerator(
        scalar=reference.new_tensor(-2.0),
        source=reference.new_tensor((0.0, 0.0, 2.0)),
    )
    report = positive_rectangle_holonomy(
        depth_generator=depth,
        time_generator=temporal,
        depth_extent=reference.new_tensor(1.0e-5),
        time_extent=reference.new_tensor(1.0e-5),
    )
    error = _maximum_absolute(report.curvature_error.as_vector())
    commutator = _norm(report.predicted_curvature.as_vector())
    return HolonomyOracleResult(
        orientation=report.orientation,
        commutator_norm=commutator,
        maximum_small_rectangle_error=error,
        passed=commutator > 1.0e-6 and error <= 1.0e-4,
    )


def evaluate_lift_oracle() -> LiftOracleResult:
    report = diagnose_sensor_depth_lift(sideways_pinhole_lift())
    full_residual = _maximum_absolute(report.full_lift_residual_norm)
    scalar_residual = float(
        torch.amin(report.supplied_axial_residual_norm).detach().cpu()
    )
    full_passed = bool(torch.all(report.full_lift_passed))
    scalar_rejected = bool(torch.all(~report.supplied_scalar_flow_exact))
    return LiftOracleResult(
        maximum_full_lift_residual=full_residual,
        minimum_scalar_axial_residual=scalar_residual,
        full_lift_passed=full_passed,
        scalar_flow_rejected=scalar_rejected,
        passed=full_passed and scalar_rejected and scalar_residual > 0.5,
    )


def build_flat_translation_probe_series(
    *,
    probe_count: int,
    end_time: float = 0.5,
) -> tuple[RepresentationProbeSeries, int, int]:
    if isinstance(probe_count, bool) or not isinstance(probe_count, int) or probe_count < 3:
        raise ValueError("probe_count must be an integer >=3")
    times = torch.linspace(0.0, end_time, probe_count, dtype=torch.float64)
    physical = []
    near = []
    far = []
    corrected = []
    curvature = []
    retained_flow_bytes: int | None = None
    run_count: int | None = None
    for time in times:
        fixture = flat_translation_fixed_clips(
            time,
            dtype=times.dtype,
            device=times.device,
        )
        core = evaluate_connection_core(
            fixture.ray,
            fixture.rate,
            fixture.flow,
            fixture.endpoint_history,
            quadrature_order=fixture.quadrature_order,
        )
        physical.append(core.ordered.total.as_vector())
        near.append(core.endpoint_transports.near.as_vector())
        far.append(core.endpoint_transports.far.as_vector())
        corrected.append(core.flow_corrected_transfer.as_vector())
        curvature.append(core.transported_curvature_source.as_vector())
        retained_flow_bytes = fixture.flow_declaration.retained_bytes
        run_count = fixture.ray.run_count
    if retained_flow_bytes is None or run_count is None:
        raise RuntimeError("flat-translation probe construction produced no rows")
    return (
        RepresentationProbeSeries(
            times=times,
            physical_transfer=torch.stack(physical),
            near_endpoint_transport=torch.stack(near),
            far_endpoint_transport=torch.stack(far),
            flow_corrected_transfer=torch.stack(corrected),
            transported_curvature_source=torch.stack(curvature),
        ),
        run_count,
        retained_flow_bytes,
    )


def _promotion_decision(
    certificates: tuple[RepresentationCertificate, ...],
) -> PromotionDecision:
    by_variant = {
        certificate.variant: certificate for certificate in certificates
    }
    direct = (
        by_variant["A0_direct_U"],
        by_variant["A0c_direct_U_capacity_matched_flow"],
        by_variant["A1_group_U_tilde"],
    )
    curvature = by_variant["A2_signed_K_F"]
    best_direct_bytes = min(item.total_retained_bytes for item in direct)
    best_direct_work = min(item.compile_ordered_word_work for item in direct)
    payload_improvement = best_direct_bytes / max(
        curvature.total_retained_bytes,
        1,
    )
    work_improvement = best_direct_work / max(
        curvature.compile_ordered_word_work,
        1,
    )
    return PromotionDecision(
        required_payload_improvement=2.0,
        required_ordered_work_improvement=2.0,
        required_measured_time_improvement=0.20,
        k_f_payload_improvement_vs_best_direct=payload_improvement,
        k_f_work_improvement_vs_best_direct=work_improvement,
        measured_time_improvement=None,
        promote_native_runtime=False,
        reason=(
            "Runtime promotion remains closed: this oracle has no measured "
            "request-time result, and K_F must beat both direct ABIs after "
            "charging flow, endpoints, base state, reconstruction, and gradients."
        ),
    )


def run_reference_oracle(
    *,
    probe_count: int = 65,
    maximum_atlas_nodes: int | None = None,
    primal_tolerance: float = 1.0e-7,
    secant_tolerance: float = 1.0e-6,
) -> ConnectionOracleReport:
    """Run the complete source-only decision oracle on a safe CPU host."""

    fixture_times = {
        "front_red_back_blue": 0.0,
        "moving_noncommuting_boundary": 0.0,
        "advected_colored_slabs_moving_clips": 0.2,
        "boundary_flow_mismatch": 0.2,
        "material_evolution": 0.2,
        "discontinuous_flow": 0.0,
        "flat_translation_fixed_clips": 0.2,
    }
    if maximum_atlas_nodes is None:
        maximum_atlas_nodes = probe_count
    fixture_results = tuple(
        evaluate_fixture(build_fixture(name, fixture_times[name]))
        for name in FIXTURE_BUILDERS
    )
    cosine = cosine_depth_curvature_cancellation()
    cosine_pointwise = max(
        _maximum_absolute(cosine.curvature.scalar),
        _maximum_absolute(cosine.curvature.source),
    )
    cosine_integral = _maximum_absolute(
        cosine.transported_curvature_integral.as_vector()
    )
    corrected_derivative = evaluate_corrected_derivative_oracle()
    selected_direction = evaluate_selected_direction_oracle()
    endpoint_history_direction = evaluate_endpoint_history_direction_oracle()
    shared_flow_direction = evaluate_shared_flow_direction_oracle()
    holonomy = evaluate_holonomy_oracle()
    lift = evaluate_lift_oracle()
    series, run_count, flow_bytes = build_flat_translation_probe_series(
        probe_count=probe_count
    )
    variants = (
        (AtlasKind.PHYSICAL_U, "A0_direct_U", 0),
        (
            AtlasKind.PHYSICAL_U,
            "A0c_direct_U_capacity_matched_flow",
            flow_bytes,
        ),
        (AtlasKind.GROUP_U_TILDE, "A1_group_U_tilde", flow_bytes),
        (AtlasKind.SIGNED_K_F, "A2_signed_K_F", flow_bytes),
    )
    certificates = tuple(
        compile_equal_family_representation(
            series,
            kind=kind,
            variant=variant,
            primal_tolerance=primal_tolerance,
            secant_tolerance=secant_tolerance,
            run_count=run_count,
            shared_flow_payload_bytes=retained_bytes,
            maximum_nodes=maximum_atlas_nodes,
        ).certificate
        for kind, variant, retained_bytes in variants
    )
    correctness = (
        all(result.passed for result in fixture_results)
        and cosine_pointwise > 1.0e-6
        and cosine_integral <= IDENTITY_TOLERANCE
        and corrected_derivative.passed
        and selected_direction.passed
        and endpoint_history_direction.passed
        and shared_flow_direction.passed
        and holonomy.passed
        and lift.passed
        and all(
            certificate.probe_primal_secant_verified
            for certificate in certificates
        )
    )
    return ConnectionOracleReport(
        schema_version=1,
        algorithm=(
            "stratified P0 ray-fiber optical connection with constrained "
            "Lagrangian flow and U/U_tilde/K_F ABIs"
        ),
        fixture_results=fixture_results,
        cosine_pointwise_curvature_maximum=cosine_pointwise,
        cosine_transported_integral_maximum=cosine_integral,
        cosine_cancellation_passed=(
            cosine_pointwise > 1.0e-6
            and cosine_integral <= IDENTITY_TOLERANCE
        ),
        independently_recomputed_corrected_derivative=corrected_derivative,
        selected_direction=selected_direction,
        endpoint_history_direction=endpoint_history_direction,
        shared_flow_direction=shared_flow_direction,
        holonomy=holonomy,
        lift=lift,
        representation_certificates=certificates,
        promotion=_promotion_decision(certificates),
        all_reference_correctness_gates_passed=correctness,
        runtime_promotion_closed=True,
        limitations=(
            "Stable P0 owner words only; event discovery and topology changes are external.",
            "Atlas certificates cover a declared finite probe grid, not the continuous interval.",
            "Secant checks are temporal diagnostics; selected parameter JVPs are checked separately.",
            "Endpoint histories in the generic core remain caller-supplied "
            "and must be provenance-sealed by an integration layer.",
            "No native shader, performance result, training result, or paper claim is produced by this oracle.",
        ),
    )


__all__ = [
    "ConnectionOracleReport",
    "FixtureOracleResult",
    "HolonomyOracleResult",
    "LiftOracleResult",
    "PromotionDecision",
    "SelectedDirectionOracleResult",
    "SharedFlowDirectionOracleResult",
    "build_flat_translation_probe_series",
    "evaluate_fixture",
    "evaluate_corrected_derivative_oracle",
    "evaluate_endpoint_history_direction_oracle",
    "evaluate_holonomy_oracle",
    "evaluate_lift_oracle",
    "evaluate_shared_flow_direction_oracle",
    "evaluate_selected_direction_oracle",
    "run_reference_oracle",
]
