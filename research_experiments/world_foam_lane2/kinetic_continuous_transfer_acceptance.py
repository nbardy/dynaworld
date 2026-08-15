"""Continuous CPU rank selection for kinetic P0 transfer charts.

This module closes the sampled-error gap as far as the present mathematics
allows.  It reuses the outward-rounded interval arithmetic, dual-number
operations, and affine-Lie decoder from
``continuous_lie_jet_certificate``.  The legacy exact-world evaluator itself
is not reusable: it assumes Möbius cuts from fixed 4D planes, whereas a
general kinetic face has ``z(t)=-B(t)/A(t)`` with quadratic ``A`` and ``B``.
The exact branch below therefore evaluates those kinetic polynomials directly.

For each certified owner chart, a frame-independent policy tries a fixed rank
schedule.  Interval bisection encloses continuously over the chart's supported
float64 interval:

* exact ordered P0 transfer minus compact affine-Lie transfer; and
* every material-Jacobian entry for referenced densities and RGB values.

The Jacobian enclosure certifies all material JVPs with a declared direction
``L1`` bound and all accumulated material VJPs with a declared total output-
cotangent ``L1`` bound.  Rank selection never observes requested render times
and retains no sample-sized tape.  Geometry/ray/weight/event derivatives and
runtime floating-point roundoff are not certified.  Irrational topology-seam
isolator neighborhoods remain excluded by the outer multi-chart program.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from fractions import Fraction

import torch
from continuous_lie_jet_certificate import (
    _Arithmetic,
    _decode_lie_chart,
    _Dual,
    _dual_unary_at_exact_zero,
    _float_down,
    _float_up,
    _Interval,
    _is_exact_zero,
    _maximum_absolute,
    _NeedsSplitError,
)
from kinetic_chart_transfer_bridge import (
    BoundKineticOwnerProgram,
    KineticChartP0Geometry,
    KineticChartP0Transfer,
    KineticOwnerProgramLike,
    bind_kinetic_owner_program,
    compile_bound_kinetic_chart_p0_geometry,
    refresh_kinetic_chart_p0_transfer,
)
from kinetic_multichart_transfer_program import (
    KineticMultiChartP0Program,
    KineticMultiChartP0Transfer,
    assemble_bound_kinetic_multichart_p0_program,
    refresh_kinetic_multichart_p0_transfer,
)
from kinetic_power_word_compiler import (
    AffineKineticPowerSites,
    KineticRayPowerDifference,
    kinetic_pair_ray_power_difference,
)
from transfer_lie_chart import transfer_lie_encode

DTYPE = torch.float64


class KineticContinuousCertificateError(ValueError):
    """A bounded continuous kinetic proof could not be completed."""


@dataclass(frozen=True)
class KineticContinuousTransferPolicy:
    """Frame-independent rank schedule, tolerances, and proof budget."""

    node_count_schedule: tuple[int, ...]
    transfer_tolerance: float
    material_jacobian_entry_tolerance: float
    material_jvp_direction_l1_bound: float
    material_jvp_tolerance: float
    material_vjp_cotangent_l1_bound: float
    material_vjp_tolerance: float
    max_split_depth: int = 10
    max_leaves_per_rank: int = 4096
    arithmetic_fraction_bits: int = 96
    max_material_dual_dimension: int = 256


@dataclass(frozen=True)
class KineticContinuousTransferCertificate:
    """Continuous primal and material-action bounds for one fixed rank."""

    passed: bool
    chart_id: int
    node_count: int
    transfer_error_upper_bound: float
    material_jacobian_entry_error_upper_bound: float
    material_jvp_error_upper_bound: float
    material_vjp_error_upper_bound: float
    transfer_tolerance: float
    material_jacobian_entry_tolerance: float
    material_jvp_tolerance: float
    material_vjp_tolerance: float
    parameter_labels: tuple[str, ...]
    leaf_count: int
    deepest_split: int
    arithmetic_fraction_bits: int
    minimum_cut_denominator_absolute_lower_bound: float | None
    minimum_fiber_speed_lower_bound: float
    minimum_coordinate_segment_length_lower_bound: float
    compiled_lie_cone_certified: bool
    source_content_digest: str
    transfer_snapshot_digest: str
    continuous_supported_interval_coverage: bool = True
    full_algebraic_owner_boundary_coverage: bool = False
    exact_kinetic_word_replay_used: bool = True
    requested_frame_sampling_used: bool = False
    material_jacobian_certified: bool = True
    material_jvp_action_certified: bool = True
    material_vjp_action_certified: bool = True
    geometry_jacobian_certified: bool = False
    event_time_jacobian_certified: bool = False
    runtime_floating_point_roundoff_certified: bool = False
    certified_sample_weight_semantics: str = "real_arithmetic_second_form_barycentric"
    runtime_dense_fallback_certified: bool = False


@dataclass(frozen=True)
class KineticChartRankAttempt:
    """One deterministic candidate rank and its proof outcome."""

    node_count: int
    passed: bool
    certificate: KineticContinuousTransferCertificate | None
    failure_reason: str | None


@dataclass(frozen=True)
class KineticChartContinuousSelection:
    """Rank-selection trace for one owner chart."""

    chart_id: int
    attempts: tuple[KineticChartRankAttempt, ...]
    selected_node_count: int | None
    passed: bool


@dataclass(frozen=True)
class KineticContinuousTransferSelection:
    """All-chart selection result and accepted material snapshot."""

    passed: bool
    policy: KineticContinuousTransferPolicy
    charts: tuple[KineticChartContinuousSelection, ...]
    program: KineticMultiChartP0Program | None
    transfer: KineticMultiChartP0Transfer | None
    source_content_digest: str
    maximum_transfer_error_upper_bound: float | None
    maximum_material_jacobian_entry_error_upper_bound: float | None
    maximum_material_jvp_error_upper_bound: float | None
    maximum_material_vjp_error_upper_bound: float | None
    failure_reasons: tuple[str, ...]
    total_certificate_leaves: int
    deepest_certificate_split: int
    rank_selection_used_requested_samples: bool = False
    retained_validation_sample_bytes: int = 0
    temporal_subdivision_used: bool = False
    full_algebraic_owner_boundary_coverage: bool = False
    geometry_jacobian_certified: bool = False
    event_time_jacobian_certified: bool = False


@dataclass(frozen=True)
class _MaterialLayout:
    referenced_sites: tuple[int, ...]
    labels: tuple[str, ...]
    density_index: dict[int, int]
    color_index: dict[tuple[int, int], int]

    @property
    def size(self) -> int:
        return len(self.labels)


@dataclass(frozen=True)
class _CompiledMaterialLinearization:
    denominator_coefficients: tuple[Fraction, ...]
    numerator_coefficients: tuple[tuple[Fraction, ...], ...]
    numerator_coefficient_tangents: tuple[tuple[tuple[_Interval, ...], ...], ...]


@dataclass(frozen=True)
class _KineticLeafBounds:
    transfer_upper: Fraction
    material_jacobian_upper: Fraction
    minimum_denominator: Fraction | None
    minimum_speed: Fraction
    minimum_coordinate_length: Fraction
    minimum_compiled_cone_margin: Fraction


def select_continuously_certified_kinetic_transfer(
    owner_program: KineticOwnerProgramLike,
    sites: AffineKineticPowerSites,
    ray_coefficients: torch.Tensor,
    site_density: torch.Tensor,
    site_color: torch.Tensor,
    *,
    policy: KineticContinuousTransferPolicy,
) -> KineticContinuousTransferSelection:
    """Choose one certified rank per owner chart without requested samples."""

    _validate_policy(policy)
    binding = bind_kinetic_owner_program(owner_program, sites, ray_coefficients)
    selected_geometries: list[KineticChartP0Geometry] = []
    chart_results = []
    failures = []
    certificates = []
    for chart_id in range(len(binding.program.charts)):
        attempts = []
        selected = None
        for node_count in policy.node_count_schedule:
            geometry = compile_bound_kinetic_chart_p0_geometry(
                binding,
                chart_id=chart_id,
                node_count=node_count,
            )
            transfer = refresh_kinetic_chart_p0_transfer(
                geometry,
                site_density,
                site_color,
            )
            try:
                certificate = certify_kinetic_chart_transfer(
                    binding,
                    transfer,
                    policy=policy,
                )
            except KineticContinuousCertificateError as error:
                attempts.append(
                    KineticChartRankAttempt(
                        node_count=node_count,
                        passed=False,
                        certificate=None,
                        failure_reason=str(error),
                    )
                )
                continue
            attempts.append(
                KineticChartRankAttempt(
                    node_count=node_count,
                    passed=certificate.passed,
                    certificate=certificate,
                    failure_reason=None if certificate.passed else "continuous_tolerance_exceeded",
                )
            )
            if certificate.passed:
                selected = geometry
                certificates.append(certificate)
                break
        if selected is None:
            failures.append(f"chart[{chart_id}]: no candidate rank passed continuous certification")
        else:
            selected_geometries.append(selected)
        chart_results.append(
            KineticChartContinuousSelection(
                chart_id=chart_id,
                attempts=tuple(attempts),
                selected_node_count=None if selected is None else selected.node_count,
                passed=selected is not None,
            )
        )

    selected_program = None
    selected_transfer = None
    if not failures:
        selected_program = assemble_bound_kinetic_multichart_p0_program(
            binding,
            tuple(selected_geometries),
        )
        selected_transfer = refresh_kinetic_multichart_p0_transfer(
            selected_program,
            site_density,
            site_color,
        )
    return KineticContinuousTransferSelection(
        passed=not failures,
        policy=policy,
        charts=tuple(chart_results),
        program=selected_program,
        transfer=selected_transfer,
        source_content_digest=binding.source_content_digest,
        maximum_transfer_error_upper_bound=max(
            (certificate.transfer_error_upper_bound for certificate in certificates),
            default=None,
        ),
        maximum_material_jacobian_entry_error_upper_bound=max(
            (certificate.material_jacobian_entry_error_upper_bound for certificate in certificates),
            default=None,
        ),
        maximum_material_jvp_error_upper_bound=max(
            (certificate.material_jvp_error_upper_bound for certificate in certificates),
            default=None,
        ),
        maximum_material_vjp_error_upper_bound=max(
            (certificate.material_vjp_error_upper_bound for certificate in certificates),
            default=None,
        ),
        failure_reasons=tuple(failures),
        total_certificate_leaves=sum(certificate.leaf_count for certificate in certificates),
        deepest_certificate_split=max(
            (certificate.deepest_split for certificate in certificates),
            default=0,
        ),
        full_algebraic_owner_boundary_coverage=(
            not failures and all(certificate.full_algebraic_owner_boundary_coverage for certificate in certificates)
        ),
    )


def certify_kinetic_chart_transfer(
    binding: BoundKineticOwnerProgram,
    transfer: KineticChartP0Transfer,
    *,
    policy: KineticContinuousTransferPolicy,
) -> KineticContinuousTransferCertificate:
    """Certify one fixed-rank kinetic chart over its supported interval."""

    _validate_policy(policy)
    binding.assert_current()
    geometry = transfer.geometry
    if geometry.binding_digest != binding.source_content_digest:
        raise ValueError("kinetic transfer geometry has stale source provenance")
    if geometry.chart_id >= len(binding.program.charts):
        raise ValueError("kinetic transfer chart id is outside its bound owner program")
    if transfer.site_density.device.type != "cpu" or transfer.site_color.device.type != "cpu":
        raise ValueError("continuous kinetic certification is CPU-only")
    layout = _material_layout(geometry.owner_word)
    if layout.size > policy.max_material_dual_dimension:
        raise KineticContinuousCertificateError(
            "material dual dimension exceeds the preallocation limit: "
            f"required={layout.size}, limit={policy.max_material_dual_dimension}"
        )
    arithmetic = _Arithmetic(policy.arithmetic_fraction_bits)
    linearization = _compile_material_linearization(arithmetic, transfer, layout)
    differences = tuple(
        kinetic_pair_ray_power_difference(
            binding.sites,
            binding.ray_coefficients,
            left,
            right,
        )
        for left, right in zip(geometry.owner_word, geometry.owner_word[1:], strict=False)
    )
    t_min = Fraction.from_float(geometry.schedule.t_min)
    t_max = Fraction.from_float(geometry.schedule.t_max)
    queue: list[tuple[Fraction, Fraction, int]] = [(t_min, t_max, 0)]
    leaves = []
    deepest = 0
    while queue:
        lo, hi, depth = queue.pop()
        deepest = max(deepest, depth)
        try:
            leaf = _evaluate_kinetic_leaf(
                arithmetic,
                binding,
                transfer,
                layout,
                linearization,
                differences,
                lo,
                hi,
            )
        except _NeedsSplitError as error:
            if depth >= policy.max_split_depth or lo == hi:
                raise KineticContinuousCertificateError(
                    "kinetic interval precondition remains unproved at maximum split depth "
                    f"on [{float(lo):.17g},{float(hi):.17g}]: {error}"
                ) from error
            leaf = None
        should_split = leaf is None or (
            (
                _fraction_exceeds_float(leaf.transfer_upper, policy.transfer_tolerance)
                or _fraction_exceeds_float(
                    leaf.material_jacobian_upper,
                    _effective_material_jacobian_tolerance(policy),
                )
            )
            and depth < policy.max_split_depth
        )
        if should_split:
            midpoint = (lo + hi) / 2
            if not lo < midpoint < hi:
                raise KineticContinuousCertificateError("kinetic interval can no longer be bisected")
            if len(leaves) + len(queue) + 2 > policy.max_leaves_per_rank:
                raise KineticContinuousCertificateError("kinetic continuous certificate exceeded max_leaves_per_rank")
            queue.append((midpoint, hi, depth + 1))
            queue.append((lo, midpoint, depth + 1))
        else:
            if leaf is None:
                raise ArithmeticError("kinetic certificate accepted an absent leaf")
            leaves.append(leaf)

    transfer_upper = max(leaf.transfer_upper for leaf in leaves)
    jacobian_upper = max(leaf.material_jacobian_upper for leaf in leaves)
    transfer_upper_float = _float_up(transfer_upper)
    jacobian_upper_float = _float_up(jacobian_upper)
    jvp_upper = _multiply_float_up(
        jacobian_upper_float,
        policy.material_jvp_direction_l1_bound,
    )
    vjp_upper = _multiply_float_up(
        jacobian_upper_float,
        policy.material_vjp_cotangent_l1_bound,
    )
    minimum_denominators = [leaf.minimum_denominator for leaf in leaves if leaf.minimum_denominator is not None]
    minimum_cone = min(leaf.minimum_compiled_cone_margin for leaf in leaves)
    passed = (
        transfer_upper_float <= policy.transfer_tolerance
        and jacobian_upper_float <= policy.material_jacobian_entry_tolerance
        and jvp_upper <= policy.material_jvp_tolerance
        and vjp_upper <= policy.material_vjp_tolerance
        and minimum_cone >= 0
    )
    return KineticContinuousTransferCertificate(
        passed=passed,
        chart_id=geometry.chart_id,
        node_count=geometry.node_count,
        transfer_error_upper_bound=transfer_upper_float,
        material_jacobian_entry_error_upper_bound=jacobian_upper_float,
        material_jvp_error_upper_bound=jvp_upper,
        material_vjp_error_upper_bound=vjp_upper,
        transfer_tolerance=policy.transfer_tolerance,
        material_jacobian_entry_tolerance=policy.material_jacobian_entry_tolerance,
        material_jvp_tolerance=policy.material_jvp_tolerance,
        material_vjp_tolerance=policy.material_vjp_tolerance,
        parameter_labels=layout.labels,
        leaf_count=len(leaves),
        deepest_split=deepest,
        arithmetic_fraction_bits=arithmetic.bits,
        minimum_cut_denominator_absolute_lower_bound=(
            None if not minimum_denominators else _float_down(min(minimum_denominators))
        ),
        minimum_fiber_speed_lower_bound=_float_down(min(leaf.minimum_speed for leaf in leaves)),
        minimum_coordinate_segment_length_lower_bound=_float_down(
            min(leaf.minimum_coordinate_length for leaf in leaves)
        ),
        compiled_lie_cone_certified=minimum_cone >= 0,
        source_content_digest=binding.source_content_digest,
        transfer_snapshot_digest=_transfer_snapshot_digest(transfer),
        full_algebraic_owner_boundary_coverage=geometry.full_algebraic_boundary_coverage,
    )


def _evaluate_kinetic_leaf(
    arithmetic: _Arithmetic,
    binding: BoundKineticOwnerProgram,
    transfer: KineticChartP0Transfer,
    layout: _MaterialLayout,
    linearization: _CompiledMaterialLinearization,
    differences: tuple[KineticRayPowerDifference, ...],
    lo: Fraction,
    hi: Fraction,
) -> _KineticLeafBounds:
    time_interval = _Interval(arithmetic.down(lo), arithmetic.up(hi))
    time = arithmetic.dual_time(time_interval, layout.size)
    exact, exact_margins = _exact_kinetic_transfer_jet(
        arithmetic,
        binding,
        transfer,
        layout,
        differences,
        time,
    )
    compiled_chart = _compiled_material_chart_jet(
        arithmetic,
        linearization,
        time,
    )
    compiled = _decode_lie_chart(arithmetic, compiled_chart)
    compiled_kappa = compiled_chart[0].value
    if _is_exact_zero(compiled_kappa):
        if any(not _is_exact_zero(component.value) for component in compiled_chart[1:]):
            raise _NeedsSplitError("compiled zero-kappa chart has nonzero color velocity")
    elif compiled_kappa.lo <= 0:
        raise _NeedsSplitError("compiled kinetic kappa has no positive lower bound")
    midpoint = (lo + hi) / 2
    radius = (hi - lo) / 2
    midpoint_time = arithmetic.dual_time(_Interval(midpoint, midpoint), layout.size)
    exact_midpoint, _ = _exact_kinetic_transfer_jet(
        arithmetic,
        binding,
        transfer,
        layout,
        differences,
        midpoint_time,
    )
    compiled_midpoint_chart = _compiled_material_chart_jet(
        arithmetic,
        linearization,
        midpoint_time,
    )
    compiled_midpoint = _decode_lie_chart(arithmetic, compiled_midpoint_chart)
    cone_quantities = (
        compiled_chart[0],
        *compiled_chart[1:],
        *(arithmetic.dual_sub(compiled_chart[0], component) for component in compiled_chart[1:]),
    )
    midpoint_cone_quantities = (
        compiled_midpoint_chart[0],
        *compiled_midpoint_chart[1:],
    )
    midpoint_cone_quantities = (
        midpoint_cone_quantities[0],
        *midpoint_cone_quantities[1:],
        *(arithmetic.dual_sub(midpoint_cone_quantities[0], component) for component in midpoint_cone_quantities[1:]),
    )
    cone_margins = tuple(
        max(
            quantity.value.lo,
            midpoint_quantity.value.lo - radius * _maximum_absolute(quantity.time_tangent),
        )
        for quantity, midpoint_quantity in zip(
            cone_quantities,
            midpoint_cone_quantities,
            strict=True,
        )
    )
    transfer_upper = Fraction(0)
    jacobian_upper = Fraction(0)
    for exact_component, compiled_component, exact_mid, compiled_mid in zip(
        exact,
        compiled,
        exact_midpoint,
        compiled_midpoint,
        strict=True,
    ):
        difference = arithmetic.dual_sub(exact_component, compiled_component)
        midpoint_difference = arithmetic.dual_sub(exact_mid, compiled_mid)
        transfer_upper = max(
            transfer_upper,
            min(
                _maximum_absolute(difference.value),
                _maximum_absolute(midpoint_difference.value) + radius * _maximum_absolute(difference.time_tangent),
            ),
        )
        for parameter, tangent in enumerate(difference.tangent):
            jacobian_upper = max(
                jacobian_upper,
                min(
                    _maximum_absolute(tangent),
                    _maximum_absolute(midpoint_difference.tangent[parameter])
                    + radius * _maximum_absolute(difference.mixed_time_tangent[parameter]),
                ),
            )
    return _KineticLeafBounds(
        transfer_upper=transfer_upper,
        material_jacobian_upper=jacobian_upper,
        minimum_denominator=exact_margins[0],
        minimum_speed=exact_margins[1],
        minimum_coordinate_length=exact_margins[2],
        minimum_compiled_cone_margin=min(cone_margins),
    )


def _exact_kinetic_transfer_jet(
    arithmetic: _Arithmetic,
    binding: BoundKineticOwnerProgram,
    transfer: KineticChartP0Transfer,
    layout: _MaterialLayout,
    differences: tuple[KineticRayPowerDifference, ...],
    time: _Dual,
) -> tuple[tuple[_Dual, ...], tuple[Fraction | None, Fraction, Fraction]]:
    density, color = _material_duals(arithmetic, transfer, layout)
    size = layout.size
    ray = binding.ray_coefficients
    direction = [
        arithmetic.dual_add(
            arithmetic.dual_constant(arithmetic.point(float(ray[6 + axis].item())), size),
            arithmetic.dual_mul(
                time,
                arithmetic.dual_constant(arithmetic.point(float(ray[9 + axis].item())), size),
            ),
        )
        for axis in range(3)
    ]
    speed_squared = arithmetic.dual_constant(arithmetic.zero, size)
    for component in direction:
        speed_squared = arithmetic.dual_add(
            speed_squared,
            arithmetic.dual_mul(component, component),
        )
    speed = arithmetic.dual_sqrt(speed_squared)
    cuts = [
        arithmetic.dual_constant(arithmetic.point(binding.program.near), size),
        *[
            arithmetic.dual_div(
                arithmetic.dual_neg(_polynomial_dual(arithmetic, difference.depth_intercept.coefficients, time)),
                _polynomial_dual(arithmetic, difference.depth_slope.coefficients, time),
            )
            for difference in differences
        ],
        arithmetic.dual_constant(arithmetic.point(binding.program.far), size),
    ]
    denominator_margins = []
    for difference in differences:
        denominator = _polynomial_dual(
            arithmetic,
            difference.depth_slope.coefficients,
            time,
        ).value
        if denominator.lo <= 0 <= denominator.hi:
            raise _NeedsSplitError("kinetic cut denominator interval contains zero")
        denominator_margins.append(min(abs(denominator.lo), abs(denominator.hi)))

    prefix_beta = arithmetic.dual_constant(arithmetic.one, size)
    moment = [arithmetic.dual_constant(arithmetic.zero, size) for _ in range(3)]
    minimum_coordinate = None
    for run_id, owner in enumerate(transfer.geometry.owner_word):
        coordinate_length = arithmetic.dual_sub(cuts[run_id + 1], cuts[run_id])
        if coordinate_length.value.lo < 0:
            raise _NeedsSplitError("kinetic owner segment interval becomes negative")
        minimum_coordinate = (
            coordinate_length.value.lo
            if minimum_coordinate is None
            else min(minimum_coordinate, coordinate_length.value.lo)
        )
        optical_depth = arithmetic.dual_mul(
            density[owner],
            arithmetic.dual_mul(speed, coordinate_length),
        )
        if optical_depth.value.lo < 0:
            raise _NeedsSplitError("kinetic optical-depth interval becomes negative")
        beta = arithmetic.dual_exp(arithmetic.dual_neg(optical_depth))
        alpha = arithmetic.dual_sub(
            arithmetic.dual_constant(arithmetic.one, size),
            beta,
        )
        for channel in range(3):
            moment[channel] = arithmetic.dual_add(
                moment[channel],
                arithmetic.dual_mul(
                    arithmetic.dual_mul(prefix_beta, alpha),
                    color[owner][channel],
                ),
            )
        prefix_beta = arithmetic.dual_mul(prefix_beta, beta)
    if minimum_coordinate is None:
        raise ValueError("kinetic owner chart must contain at least one run")
    return (
        (prefix_beta, *moment),
        (
            min(denominator_margins) if denominator_margins else None,
            speed.value.lo,
            minimum_coordinate,
        ),
    )


def _compile_material_linearization(
    arithmetic: _Arithmetic,
    transfer: KineticChartP0Transfer,
    layout: _MaterialLayout,
) -> _CompiledMaterialLinearization:
    node_chart_duals = tuple(
        _node_material_chart_dual(
            arithmetic,
            transfer,
            layout,
            transfer.geometry.node_physical_lengths[node_id],
        )
        for node_id in range(transfer.geometry.node_count)
    )
    node_chart = transfer_lie_encode(transfer.node_transfers)
    cardinals = []
    for node_id in range(transfer.geometry.node_count):
        polynomial = (Fraction(1),)
        for other_id in range(transfer.geometry.node_count):
            if other_id != node_id:
                node = Fraction.from_float(float(transfer.geometry.schedule.node_times[other_id].item()))
                polynomial = _multiply_polynomials(polynomial, (-node, Fraction(1)))
        weight = Fraction.from_float(float(transfer.geometry.schedule.barycentric_weights[node_id].item()))
        cardinals.append(tuple(weight * coefficient for coefficient in polynomial))
    denominator = tuple(
        sum((cardinal[degree] for cardinal in cardinals), Fraction(0)) for degree in range(transfer.geometry.node_count)
    )
    numerator_coefficients = []
    numerator_tangents = []
    for component in range(4):
        component_coefficients = []
        component_tangents = []
        for degree in range(transfer.geometry.node_count):
            component_coefficients.append(
                sum(
                    (
                        cardinal[degree] * Fraction.from_float(float(node_chart[node_id, component].item()))
                        for node_id, cardinal in enumerate(cardinals)
                    ),
                    Fraction(0),
                )
            )
            tangent_row = []
            for parameter in range(layout.size):
                total = arithmetic.zero
                for node_id, cardinal in enumerate(cardinals):
                    total = arithmetic.add(
                        total,
                        arithmetic.mul(
                            arithmetic.point(cardinal[degree]),
                            node_chart_duals[node_id][component].tangent[parameter],
                        ),
                    )
                tangent_row.append(total)
            component_tangents.append(tuple(tangent_row))
        numerator_coefficients.append(tuple(component_coefficients))
        numerator_tangents.append(tuple(component_tangents))
    return _CompiledMaterialLinearization(
        denominator_coefficients=denominator,
        numerator_coefficients=tuple(numerator_coefficients),
        numerator_coefficient_tangents=tuple(numerator_tangents),
    )


def _node_material_chart_dual(
    arithmetic: _Arithmetic,
    transfer: KineticChartP0Transfer,
    layout: _MaterialLayout,
    physical_lengths: torch.Tensor,
) -> tuple[_Dual, ...]:
    density, color = _material_duals(arithmetic, transfer, layout)
    size = layout.size
    prefix_beta = arithmetic.dual_constant(arithmetic.one, size)
    kappa = arithmetic.dual_constant(arithmetic.zero, size)
    moment = [arithmetic.dual_constant(arithmetic.zero, size) for _ in range(3)]
    for owner, length in zip(
        transfer.geometry.owner_word,
        physical_lengths.tolist(),
        strict=True,
    ):
        optical_depth = arithmetic.dual_mul(
            density[owner],
            arithmetic.dual_constant(arithmetic.point(float(length)), size),
        )
        beta = arithmetic.dual_exp(arithmetic.dual_neg(optical_depth))
        alpha = arithmetic.dual_sub(
            arithmetic.dual_constant(arithmetic.one, size),
            beta,
        )
        for channel in range(3):
            moment[channel] = arithmetic.dual_add(
                moment[channel],
                arithmetic.dual_mul(
                    arithmetic.dual_mul(prefix_beta, alpha),
                    color[owner][channel],
                ),
            )
        prefix_beta = arithmetic.dual_mul(prefix_beta, beta)
        kappa = arithmetic.dual_add(kappa, optical_depth)
    if _is_exact_zero(kappa.value):
        inverse_phi = _dual_unary_at_exact_zero(
            arithmetic,
            kappa,
            value=Fraction(1),
            first_derivative=Fraction(1, 2),
            second_derivative=Fraction(1, 6),
        )
    elif kappa.value.lo <= 0:
        raise KineticContinuousCertificateError("node Lie derivative lacks a positive total optical-depth margin")
    else:
        inverse_phi = arithmetic.dual_div(
            kappa,
            arithmetic.dual_sub(
                arithmetic.dual_constant(arithmetic.one, size),
                prefix_beta,
            ),
        )
    return (kappa, *(arithmetic.dual_mul(inverse_phi, component) for component in moment))


def _compiled_material_chart_jet(
    arithmetic: _Arithmetic,
    linearization: _CompiledMaterialLinearization,
    time: _Dual,
) -> tuple[_Dual, ...]:
    """Evaluate the actual second-form interpolant after clearing poles.

    For stored barycentric weights ``w_j`` and nodes ``x_j``, multiplying
    numerator and denominator by ``prod_k(t-x_k)`` gives a nonsingular
    rational expression that also equals the exact one-hot node branch.
    This is the real-arithmetic compact evaluator; a runtime dense fallback
    caused by floating-point conditioning remains outside the certificate.
    """

    size = len(linearization.numerator_coefficient_tangents[0][0])
    denominator = _dual_polynomial_with_tangents(
        arithmetic,
        linearization.denominator_coefficients,
        None,
        time,
        size=size,
    )
    if denominator.value.lo <= 0 <= denominator.value.hi:
        raise _NeedsSplitError("cleared barycentric denominator interval contains zero")

    result = []
    for component in range(4):
        numerator = _dual_polynomial_with_tangents(
            arithmetic,
            linearization.numerator_coefficients[component],
            linearization.numerator_coefficient_tangents[component],
            time,
            size=size,
        )
        result.append(arithmetic.dual_div(numerator, denominator))
    return tuple(result)


def _dual_polynomial_with_tangents(
    arithmetic: _Arithmetic,
    coefficients: tuple[Fraction, ...],
    coefficient_tangents: tuple[tuple[_Interval, ...], ...] | None,
    time: _Dual,
    *,
    size: int,
) -> _Dual:
    result = arithmetic.dual_constant(arithmetic.zero, size)
    for degree in reversed(range(len(coefficients))):
        tangent = (arithmetic.zero,) * size if coefficient_tangents is None else coefficient_tangents[degree]
        coefficient = _Dual(
            arithmetic.point(coefficients[degree]),
            tangent,
            arithmetic.zero,
            (arithmetic.zero,) * size,
        )
        result = arithmetic.dual_add(
            arithmetic.dual_mul(result, time),
            coefficient,
        )
    return result


def _multiply_polynomials(
    left: tuple[Fraction, ...],
    right: tuple[Fraction, ...],
) -> tuple[Fraction, ...]:
    result = [Fraction(0)] * (len(left) + len(right) - 1)
    for left_degree, left_value in enumerate(left):
        for right_degree, right_value in enumerate(right):
            result[left_degree + right_degree] += left_value * right_value
    return tuple(result)


def _polynomial_dual(
    arithmetic: _Arithmetic,
    coefficients: tuple[Fraction, ...],
    time: _Dual,
) -> _Dual:
    result = arithmetic.dual_constant(arithmetic.zero, len(time.tangent))
    for coefficient in reversed(coefficients):
        result = arithmetic.dual_add(
            arithmetic.dual_mul(result, time),
            arithmetic.dual_constant(arithmetic.point(coefficient), len(time.tangent)),
        )
    return result


def _material_duals(
    arithmetic: _Arithmetic,
    transfer: KineticChartP0Transfer,
    layout: _MaterialLayout,
) -> tuple[list[_Dual], list[list[_Dual]]]:
    density = []
    color = []
    for site_id in range(transfer.geometry.site_count):
        density_value = arithmetic.point(float(transfer.site_density[site_id].item()))
        density.append(
            arithmetic.dual_variable(
                density_value,
                layout.size,
                layout.density_index[site_id],
            )
            if site_id in layout.density_index
            else arithmetic.dual_constant(density_value, layout.size)
        )
        row = []
        for channel in range(3):
            color_value = arithmetic.point(float(transfer.site_color[site_id, channel].item()))
            index = layout.color_index.get((site_id, channel))
            row.append(
                arithmetic.dual_variable(color_value, layout.size, index)
                if index is not None
                else arithmetic.dual_constant(color_value, layout.size)
            )
        color.append(row)
    return density, color


def _material_layout(owner_word: tuple[int, ...]) -> _MaterialLayout:
    referenced = tuple(sorted(set(owner_word)))
    labels = []
    density_index = {}
    color_index = {}
    for site_id in referenced:
        density_index[site_id] = len(labels)
        labels.append(f"site_density[{site_id}]")
        for channel in range(3):
            color_index[(site_id, channel)] = len(labels)
            labels.append(f"site_color[{site_id},{channel}]")
    return _MaterialLayout(
        referenced_sites=referenced,
        labels=tuple(labels),
        density_index=density_index,
        color_index=color_index,
    )


def _effective_material_jacobian_tolerance(
    policy: KineticContinuousTransferPolicy,
) -> float:
    return min(
        policy.material_jacobian_entry_tolerance,
        policy.material_jvp_tolerance / policy.material_jvp_direction_l1_bound,
        policy.material_vjp_tolerance / policy.material_vjp_cotangent_l1_bound,
    )


def _validate_policy(policy: KineticContinuousTransferPolicy) -> None:
    if (
        not policy.node_count_schedule
        or any(
            isinstance(node_count, bool) or not isinstance(node_count, int) or node_count < 2
            for node_count in policy.node_count_schedule
        )
        or tuple(sorted(set(policy.node_count_schedule))) != policy.node_count_schedule
    ):
        raise ValueError("node_count_schedule must be strictly increasing unique integers >=2")
    nonnegative = (
        policy.transfer_tolerance,
        policy.material_jacobian_entry_tolerance,
        policy.material_jvp_tolerance,
        policy.material_vjp_tolerance,
    )
    positive = (
        policy.material_jvp_direction_l1_bound,
        policy.material_vjp_cotangent_l1_bound,
    )
    if any(not math.isfinite(value) or value < 0 for value in nonnegative):
        raise ValueError("continuous kinetic tolerances must be finite and nonnegative")
    if any(not math.isfinite(value) or value <= 0 for value in positive):
        raise ValueError("continuous kinetic action norm bounds must be finite and positive")
    if policy.max_split_depth < 0 or policy.max_leaves_per_rank < 1:
        raise ValueError("continuous kinetic split budgets are invalid")
    if policy.arithmetic_fraction_bits < 64 or policy.max_material_dual_dimension < 1:
        raise ValueError("continuous kinetic arithmetic/dual budgets are invalid")


def _fraction_exceeds_float(value: Fraction, threshold: float) -> bool:
    return value > Fraction.from_float(threshold)


def _multiply_float_up(left: float, right: float) -> float:
    return _float_up(Fraction.from_float(left) * Fraction.from_float(right))


def _transfer_snapshot_digest(transfer: KineticChartP0Transfer) -> str:
    digest = hashlib.sha256()
    for tensor in (
        transfer.site_density,
        transfer.site_color,
        transfer.node_transfers,
        transfer.geometry.node_physical_lengths,
    ):
        value = tensor.detach().to(device="cpu").contiguous()
        digest.update(repr((tuple(value.shape), str(value.dtype))).encode("utf-8"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


__all__ = [
    "KineticChartContinuousSelection",
    "KineticChartRankAttempt",
    "KineticContinuousCertificateError",
    "KineticContinuousTransferCertificate",
    "KineticContinuousTransferPolicy",
    "KineticContinuousTransferSelection",
    "certify_kinetic_chart_transfer",
    "select_continuously_certified_kinetic_transfer",
]
