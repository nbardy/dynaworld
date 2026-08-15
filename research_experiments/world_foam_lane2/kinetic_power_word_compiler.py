"""Exact CPU frontend for affine kinetic 3D weighted power sites.

The site state is parameterized once,

``p_i(t) = p_i0 + t v_i`` and ``w_i(t) = sum_k w_ik t^k, k <= 2``.

For the existing affine ray track

``x(t,z) = o0 + t o1 + z (d0 + t d1)``,

the pairwise power-distance difference is

``power_i - power_j = A_ij(t) z + B_ij(t)``.

Both ``A`` and ``B`` have degree at most two.  Two adjacent cuts concur when
``B_ij A_jk - B_jk A_ij = 0``, a polynomial of degree at most four.  This
module derives that polynomial exactly from the binary64 inputs and routes it
through exact square-free/Sturm isolation.  Cross-product roots sharing a cut
denominator and full-fiber ties fail closed; they are not topology seams.

At one requested time, every site's power along the ray is a common quadratic
in depth plus a site-specific line.  The common term is discarded and the
exact sparse lower-envelope implementation is reused.  No per-frame site
table is stored.

Unlike a fixed pair of sites in R4, whose spatial slice-face normal is fixed,
``p_j(t)-p_i(t)`` may change direction.  The frontend therefore represents
rotating spatial power faces rather than only translating fixed-orientation
slices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import torch
from power_topology_event_predicates import (
    CertifiedEventRoot,
    RationalPolynomial,
    UnsupportedTopologyDegeneracyError,
)
from rational_polynomial_roots import (
    isolate_rational_polynomial_roots,
    multiply_rational_polynomials,
    rational_polynomial_gcd,
)
from sparse_power_word_compiler import (
    SparsePowerRayWord,
    discover_sparse_line_envelope_word,
)


@dataclass(frozen=True)
class KineticSiteStorageReport:
    """Frame-independent parameter storage for one kinetic site table."""

    requested_frame_count: int
    site_count: int
    weight_coefficient_count: int
    parameter_scalar_count: int
    scalar_bytes: int
    parameter_bytes: int
    stored_frame_state_bytes: int = 0
    frame_dependent_parameter_bytes: int = 0


@dataclass(frozen=True)
class AffineKineticPowerSites:
    """CPU compiler parameters for affine positions and degree-<=2 weights."""

    positions0: torch.Tensor
    velocities: torch.Tensor
    weight_coefficients: torch.Tensor

    def __post_init__(self) -> None:
        positions0 = _finite_f64_cpu_tensor(self.positions0, name="positions0")
        velocities = _finite_f64_cpu_tensor(self.velocities, name="velocities")
        weights = _finite_f64_cpu_tensor(
            self.weight_coefficients,
            name="weight_coefficients",
        )
        if positions0.ndim != 2 or positions0.shape[1] != 3 or int(positions0.shape[0]) < 1:
            raise ValueError("positions0 must have shape [S,3] with S >= 1")
        if velocities.shape != positions0.shape:
            raise ValueError("velocities must have the same [S,3] shape as positions0")
        if weights.ndim != 2 or weights.shape[0] != positions0.shape[0] or not 1 <= int(weights.shape[1]) <= 3:
            raise ValueError("weight_coefficients must have shape [S,C] with 1 <= C <= 3")
        object.__setattr__(self, "positions0", positions0)
        object.__setattr__(self, "velocities", velocities)
        object.__setattr__(self, "weight_coefficients", weights)

    @property
    def site_count(self) -> int:
        return int(self.positions0.shape[0])

    @property
    def parameter_scalar_count(self) -> int:
        return int(self.positions0.numel() + self.velocities.numel() + self.weight_coefficients.numel())

    @property
    def parameter_bytes(self) -> int:
        return self.parameter_scalar_count * self.positions0.element_size()

    def storage_report(self, *, requested_frame_count: int) -> KineticSiteStorageReport:
        if (
            isinstance(requested_frame_count, bool)
            or not isinstance(requested_frame_count, int)
            or requested_frame_count < 1
        ):
            raise ValueError("requested_frame_count must be a positive integer")
        return KineticSiteStorageReport(
            requested_frame_count=int(requested_frame_count),
            site_count=self.site_count,
            weight_coefficient_count=int(self.weight_coefficients.shape[1]),
            parameter_scalar_count=self.parameter_scalar_count,
            scalar_bytes=self.positions0.element_size(),
            parameter_bytes=self.parameter_bytes,
        )


def affine_kinetic_sites_from_identity_spd4_sites(
    sites: torch.Tensor,
) -> AffineKineticPowerSites:
    """Embed the existing identity-SPD(4) site model into kinetic 3D.

    For ``q_i=(a_i,s_i)`` and power weight ``w_i``, evaluation on the physical
    slice ``X=(x,t)`` is

    ``||x-a_i||^2 + t^2 - 2 s_i t + s_i^2 - w_i``.

    Dropping the common ``t^2`` term gives a stationary 3D site with kinetic
    weight ``(w_i-s_i^2) + 2 s_i t``.  The representation identity is exact
    over the reals. This executable float64 table performs one rounding when
    ``w_i-s_i^2`` is not itself binary64-representable, so bit-exact predicate
    parity still requires retaining rational transformed coefficients. It does
    not claim a general shared-SPD(4) whitening implementation.
    """

    normalized = _finite_f64_cpu_tensor(sites, name="identity_spd4_sites")
    if normalized.ndim != 2 or normalized.shape[1] != 5 or normalized.shape[0] < 1:
        raise ValueError("identity_spd4_sites must have shape [S,5] with S >= 1")
    slice_time = normalized[:, 3]
    power_weight = normalized[:, 4]
    return AffineKineticPowerSites(
        positions0=normalized[:, :3],
        velocities=torch.zeros_like(normalized[:, :3]),
        weight_coefficients=torch.stack(
            (power_weight - slice_time.square(), 2.0 * slice_time),
            dim=1,
        ),
    )


@dataclass(frozen=True)
class KineticRayPowerDifference:
    """Exact ``power(left)-power(right) = A(t) z + B(t)`` coefficients."""

    left_site_id: int
    right_site_id: int
    depth_slope: RationalPolynomial
    depth_intercept: RationalPolynomial

    def evaluate(
        self,
        *,
        time: Fraction | float | int,
        depth: Fraction | float | int,
    ) -> Fraction:
        time_q = _as_fraction(time, name="time")
        depth_q = _as_fraction(depth, name="depth")
        return self.depth_slope.evaluate(time_q) * depth_q + self.depth_intercept.evaluate(time_q)


@dataclass(frozen=True)
class KineticAdjacentCutConcurrence:
    """Exact degree-<=4 predicate for adjacent pair-cut concurrence."""

    site_ids: tuple[int, int, int]
    polynomial: RationalPolynomial
    first_difference: KineticRayPowerDifference
    second_difference: KineticRayPowerDifference
    derivation: str = "B_ij(t) A_jk(t) - B_jk(t) A_ij(t)"
    degree_bound: int = 4
    root_isolation_implemented: bool = True


@dataclass(frozen=True)
class KineticAdjacentCutIsolation:
    """Filtered finite-cut concurrence roots on one exact time interval."""

    concurrence: KineticAdjacentCutConcurrence
    t_min: Fraction
    t_max: Fraction
    roots: tuple[CertifiedEventRoot, ...]
    exact_rational_arithmetic: bool = True
    denominator_roots_filtered: bool = True
    full_fiber_ties_rejected: bool = True
    requested_frame_sampling_used: bool = False


@dataclass(frozen=True)
class KineticPairEventPredicates:
    """Exact candidate events for one generically finite kinetic face cut."""

    difference: KineticRayPowerDifference
    denominator: RationalPolynomial
    near_crossing: RationalPolynomial
    far_crossing: RationalPolynomial
    near: Fraction
    far: Fraction
    degree_bound: int = 2
    active_owner_filter_required: bool = True


@dataclass(frozen=True)
class KineticPairEventIsolation:
    """Guarded denominator and near/far roots for one pair candidate."""

    predicates: KineticPairEventPredicates
    t_min: Fraction
    t_max: Fraction
    denominator_roots: tuple[CertifiedEventRoot, ...]
    near_crossing_roots: tuple[CertifiedEventRoot, ...]
    far_crossing_roots: tuple[CertifiedEventRoot, ...]
    exact_rational_arithmetic: bool = True
    active_owner_filter_applied: bool = False
    requested_frame_sampling_used: bool = False


def kinetic_pair_ray_power_difference(
    sites: AffineKineticPowerSites,
    ray_coefficients: torch.Tensor,
    left_site_id: int,
    right_site_id: int,
) -> KineticRayPowerDifference:
    """Derive exact quadratic-or-lower ``A_ij(t)`` and ``B_ij(t)``.

    With ``q=p_j-p_i``, the coefficients are

    ``A = 2 q dot d`` and
    ``B = 2 q dot o + ||p_i||^2 - ||p_j||^2 - w_i + w_j``.
    """

    ray = _validate_ray(ray_coefficients)
    _validate_pair_site_ids(sites, left_site_id, right_site_id)
    p0 = _fraction_rows(sites.positions0)
    velocity = _fraction_rows(sites.velocities)
    weights = _fraction_rows(sites.weight_coefficients)
    ray_q = tuple(Fraction.from_float(float(value)) for value in ray.tolist())
    origin0, origin1 = ray_q[:3], ray_q[3:6]
    direction0, direction1 = ray_q[6:9], ray_q[9:12]

    left0, right0 = p0[left_site_id], p0[right_site_id]
    left_velocity, right_velocity = velocity[left_site_id], velocity[right_site_id]
    separation0 = _subtract_vectors(right0, left0)
    separation1 = _subtract_vectors(right_velocity, left_velocity)

    depth_slope = RationalPolynomial(
        (
            2 * _dot(separation0, direction0),
            2 * (_dot(separation0, direction1) + _dot(separation1, direction0)),
            2 * _dot(separation1, direction1),
        )
    )
    left_weight = _padded_quadratic(weights[left_site_id])
    right_weight = _padded_quadratic(weights[right_site_id])
    depth_intercept = RationalPolynomial(
        (
            2 * _dot(separation0, origin0)
            + _dot(left0, left0)
            - _dot(right0, right0)
            - left_weight[0]
            + right_weight[0],
            2 * (_dot(separation0, origin1) + _dot(separation1, origin0))
            + 2 * (_dot(left0, left_velocity) - _dot(right0, right_velocity))
            - left_weight[1]
            + right_weight[1],
            2 * _dot(separation1, origin1)
            + _dot(left_velocity, left_velocity)
            - _dot(right_velocity, right_velocity)
            - left_weight[2]
            + right_weight[2],
        )
    )
    if depth_slope.degree > 2 or depth_intercept.degree > 2:
        raise ArithmeticError("affine kinetic pair coefficients exceeded the quadratic bound")
    return KineticRayPowerDifference(
        left_site_id=left_site_id,
        right_site_id=right_site_id,
        depth_slope=depth_slope,
        depth_intercept=depth_intercept,
    )


def kinetic_adjacent_cut_concurrence(
    sites: AffineKineticPowerSites,
    ray_coefficients: torch.Tensor,
    first_site_id: int,
    middle_site_id: int,
    last_site_id: int,
) -> KineticAdjacentCutConcurrence:
    """Return the exact quartic-or-lower adjacent-cut event polynomial.

    Root isolation is deliberately outside this bounded frontend.
    """

    if len({first_site_id, middle_site_id, last_site_id}) != 3:
        raise UnsupportedTopologyDegeneracyError("adjacent cut concurrence requires three distinct sites")
    first = kinetic_pair_ray_power_difference(
        sites,
        ray_coefficients,
        first_site_id,
        middle_site_id,
    )
    second = kinetic_pair_ray_power_difference(
        sites,
        ray_coefficients,
        middle_site_id,
        last_site_id,
    )
    polynomial = _subtract_polynomials(
        _multiply_polynomials(first.depth_intercept, second.depth_slope),
        _multiply_polynomials(second.depth_intercept, first.depth_slope),
    )
    if polynomial.identically_zero:
        raise UnsupportedTopologyDegeneracyError(
            "adjacent cuts are persistently concurrent; no isolated event polynomial exists"
        )
    if polynomial.degree > 4:
        raise ArithmeticError("affine kinetic concurrence exceeded the quartic bound")
    return KineticAdjacentCutConcurrence(
        site_ids=(first_site_id, middle_site_id, last_site_id),
        polynomial=polynomial,
        first_difference=first,
        second_difference=second,
    )


def kinetic_pair_event_predicates(
    difference: KineticRayPowerDifference,
    *,
    near: Fraction | float | int,
    far: Fraction | float | int,
) -> KineticPairEventPredicates:
    """Construct degree-<=2 denominator and fixed-depth crossing predicates."""

    near_q = _as_fraction(near, name="near")
    far_q = _as_fraction(far, name="far")
    if far_q <= near_q:
        raise ValueError("kinetic pair events require near < far")
    near_crossing = _add_scaled_polynomial(
        difference.depth_intercept,
        difference.depth_slope,
        scale=near_q,
    )
    far_crossing = _add_scaled_polynomial(
        difference.depth_intercept,
        difference.depth_slope,
        scale=far_q,
    )
    if (
        max(
            difference.depth_slope.degree,
            near_crossing.degree,
            far_crossing.degree,
        )
        > 2
    ):
        raise ArithmeticError("affine kinetic pair event exceeded the quadratic bound")
    return KineticPairEventPredicates(
        difference=difference,
        denominator=difference.depth_slope,
        near_crossing=near_crossing,
        far_crossing=far_crossing,
        near=near_q,
        far=far_q,
    )


def isolate_kinetic_pair_events(
    predicates: KineticPairEventPredicates,
    *,
    t_min: Fraction | float | int,
    t_max: Fraction | float | int,
    max_interval_width: Fraction = Fraction(1, 1 << 40),
    max_bisection_depth: int = 192,
) -> KineticPairEventIsolation:
    """Isolate all low-degree pair candidates, rejecting persistent ties.

    These roots are a superset until a kinetic lower-envelope compiler checks
    that the pair is adjacent and active on the corresponding one-sided chart.
    """

    lo = _as_fraction(t_min, name="t_min")
    hi = _as_fraction(t_max, name="t_max")
    if hi <= lo:
        raise ValueError("kinetic pair event isolation requires t_min < t_max")
    _reject_kinetic_full_fiber_ties(
        predicates.difference,
        t_min=lo,
        t_max=hi,
        max_interval_width=max_interval_width,
        max_bisection_depth=max_bisection_depth,
    )
    for name, polynomial in (
        ("cut denominator", predicates.denominator),
        ("near crossing", predicates.near_crossing),
        ("far crossing", predicates.far_crossing),
    ):
        if polynomial.identically_zero:
            raise UnsupportedTopologyDegeneracyError(
                f"kinetic pair has a persistent {name}; no isolated event set exists"
            )

    def roots(polynomial: RationalPolynomial) -> tuple[CertifiedEventRoot, ...]:
        return isolate_rational_polynomial_roots(
            polynomial,
            t_min=lo,
            t_max=hi,
            max_interval_width=max_interval_width,
            max_bisection_depth=max_bisection_depth,
        ).roots

    return KineticPairEventIsolation(
        predicates=predicates,
        t_min=lo,
        t_max=hi,
        denominator_roots=roots(predicates.denominator),
        near_crossing_roots=roots(predicates.near_crossing),
        far_crossing_roots=roots(predicates.far_crossing),
    )


def isolate_kinetic_adjacent_cut_concurrence(
    concurrence: KineticAdjacentCutConcurrence,
    *,
    t_min: Fraction | float | int,
    t_max: Fraction | float | int,
    max_interval_width: Fraction = Fraction(1, 1 << 40),
    max_bisection_depth: int = 192,
) -> KineticAdjacentCutIsolation:
    """Isolate genuine finite-cut concurrence roots and fail on degeneracy.

    The cross-product polynomial alone is only a candidate predicate: it also
    vanishes when one or both cut denominators vanish.  This wrapper first
    rejects full-fiber pair ties, then uses exact polynomial GCDs to reject any
    candidate event sharing a real root with ``A_ij A_jk`` on the requested
    interval.  Only the remaining roots can safely become topology seams.
    """

    lo = _as_fraction(t_min, name="t_min")
    hi = _as_fraction(t_max, name="t_max")
    if hi <= lo:
        raise ValueError("kinetic event isolation requires t_min < t_max")
    for difference in (concurrence.first_difference, concurrence.second_difference):
        _reject_kinetic_full_fiber_ties(
            difference,
            t_min=lo,
            t_max=hi,
            max_interval_width=max_interval_width,
            max_bisection_depth=max_bisection_depth,
        )
    denominator_product = multiply_rational_polynomials(
        concurrence.first_difference.depth_slope,
        concurrence.second_difference.depth_slope,
    )
    shared_denominator_roots = rational_polynomial_gcd(
        concurrence.polynomial,
        denominator_product,
    )
    if _has_real_root_in_interval(
        shared_denominator_roots,
        t_min=lo,
        t_max=hi,
        max_interval_width=max_interval_width,
        max_bisection_depth=max_bisection_depth,
    ):
        raise UnsupportedTopologyDegeneracyError("candidate concurrence coincides with a zero cut denominator")
    isolation = isolate_rational_polynomial_roots(
        concurrence.polynomial,
        t_min=lo,
        t_max=hi,
        max_interval_width=max_interval_width,
        max_bisection_depth=max_bisection_depth,
    )
    return KineticAdjacentCutIsolation(
        concurrence=concurrence,
        t_min=lo,
        t_max=hi,
        roots=isolation.roots,
    )


def discover_kinetic_power_word_at_time(
    sites: AffineKineticPowerSites,
    ray_coefficients: torch.Tensor,
    *,
    time: Fraction | float | int,
    near: Fraction | float | int,
    far: Fraction | float | int,
) -> SparsePowerRayWord:
    """Evaluate kinetic parameters on demand and discover one exact owner word."""

    ray = _validate_ray(ray_coefficients)
    time_q = _as_fraction(time, name="time")
    p0 = _fraction_rows(sites.positions0)
    velocity = _fraction_rows(sites.velocities)
    weights = _fraction_rows(sites.weight_coefficients)
    ray_q = tuple(Fraction.from_float(float(value)) for value in ray.tolist())
    origin = tuple(ray_q[axis] + time_q * ray_q[3 + axis] for axis in range(3))
    direction = tuple(ray_q[6 + axis] + time_q * ray_q[9 + axis] for axis in range(3))
    if _dot(direction, direction) == 0:
        raise ValueError("ray direction must be nonzero at the requested time")

    slopes: list[Fraction] = []
    intercepts: list[Fraction] = []
    for site_id in range(sites.site_count):
        position = tuple(p0[site_id][axis] + time_q * velocity[site_id][axis] for axis in range(3))
        delta = _subtract_vectors(origin, position)
        weight = _evaluate_coefficients(weights[site_id], time_q)
        slopes.append(2 * _dot(direction, delta))
        intercepts.append(_dot(delta, delta) - weight)
    return discover_sparse_line_envelope_word(
        slopes,
        intercepts,
        near=near,
        far=far,
    )


def _finite_f64_cpu_tensor(value: torch.Tensor, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float64).detach().cpu().clone()
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must be finite")
    return tensor


def _validate_ray(ray_coefficients: torch.Tensor) -> torch.Tensor:
    ray = _finite_f64_cpu_tensor(ray_coefficients, name="ray_coefficients")
    if ray.shape != (12,):
        raise ValueError("ray_coefficients must be a finite vector with 12 entries")
    return ray


def _validate_pair_site_ids(
    sites: AffineKineticPowerSites,
    left_site_id: int,
    right_site_id: int,
) -> None:
    if not 0 <= left_site_id < sites.site_count or not 0 <= right_site_id < sites.site_count:
        raise ValueError("pair site ids must lie inside the kinetic site table")
    if left_site_id == right_site_id:
        raise UnsupportedTopologyDegeneracyError("a kinetic power face requires two sites")


def _fraction_rows(tensor: torch.Tensor) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(tuple(Fraction.from_float(float(value)) for value in row) for row in tensor.tolist())


def _as_fraction(value: Fraction | float | int, *, name: str) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if not isinstance(value, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite rational, integer, or float")
    return Fraction.from_float(value)


def _padded_quadratic(coefficients: tuple[Fraction, ...]) -> tuple[Fraction, Fraction, Fraction]:
    return tuple(coefficients[index] if index < len(coefficients) else Fraction(0) for index in range(3))


def _evaluate_coefficients(coefficients: tuple[Fraction, ...], time: Fraction) -> Fraction:
    result = Fraction(0)
    for coefficient in reversed(coefficients):
        result = result * time + coefficient
    return result


def _subtract_vectors(
    left: tuple[Fraction, ...],
    right: tuple[Fraction, ...],
) -> tuple[Fraction, ...]:
    return tuple(a - b for a, b in zip(left, right, strict=True))


def _dot(left: tuple[Fraction, ...], right: tuple[Fraction, ...]) -> Fraction:
    return sum((a * b for a, b in zip(left, right, strict=True)), Fraction(0))


def _multiply_polynomials(
    left: RationalPolynomial,
    right: RationalPolynomial,
) -> RationalPolynomial:
    coefficients = [Fraction(0)] * (left.degree + right.degree + 1)
    for left_index, left_value in enumerate(left.coefficients):
        for right_index, right_value in enumerate(right.coefficients):
            coefficients[left_index + right_index] += left_value * right_value
    return RationalPolynomial(tuple(coefficients))


def _subtract_polynomials(
    left: RationalPolynomial,
    right: RationalPolynomial,
) -> RationalPolynomial:
    size = max(len(left.coefficients), len(right.coefficients))
    return RationalPolynomial(
        tuple(
            (left.coefficients[index] if index < len(left.coefficients) else Fraction(0))
            - (right.coefficients[index] if index < len(right.coefficients) else Fraction(0))
            for index in range(size)
        )
    )


def _add_scaled_polynomial(
    base: RationalPolynomial,
    direction: RationalPolynomial,
    *,
    scale: Fraction,
) -> RationalPolynomial:
    size = max(len(base.coefficients), len(direction.coefficients))
    return RationalPolynomial(
        tuple(
            (base.coefficients[index] if index < len(base.coefficients) else Fraction(0))
            + scale * (direction.coefficients[index] if index < len(direction.coefficients) else Fraction(0))
            for index in range(size)
        )
    )


def _reject_kinetic_full_fiber_ties(
    difference: KineticRayPowerDifference,
    *,
    t_min: Fraction,
    t_max: Fraction,
    max_interval_width: Fraction,
    max_bisection_depth: int,
) -> None:
    slope = difference.depth_slope
    intercept = difference.depth_intercept
    if slope.identically_zero and intercept.identically_zero:
        raise UnsupportedTopologyDegeneracyError(
            f"sites {difference.left_site_id}/{difference.right_site_id} tie along the full fiber"
        )
    common = rational_polynomial_gcd(slope, intercept)
    if _has_real_root_in_interval(
        common,
        t_min=t_min,
        t_max=t_max,
        max_interval_width=max_interval_width,
        max_bisection_depth=max_bisection_depth,
    ):
        raise UnsupportedTopologyDegeneracyError(
            f"sites {difference.left_site_id}/{difference.right_site_id} have a full-fiber tie in the interval"
        )


def _has_real_root_in_interval(
    polynomial: RationalPolynomial,
    *,
    t_min: Fraction,
    t_max: Fraction,
    max_interval_width: Fraction,
    max_bisection_depth: int,
) -> bool:
    if polynomial.identically_zero:
        return True
    if polynomial.degree == 0:
        return False
    return bool(
        isolate_rational_polynomial_roots(
            polynomial,
            t_min=t_min,
            t_max=t_max,
            max_interval_width=max_interval_width,
            max_bisection_depth=max_bisection_depth,
        ).roots
    )


__all__ = [
    "AffineKineticPowerSites",
    "KineticAdjacentCutConcurrence",
    "KineticAdjacentCutIsolation",
    "KineticPairEventIsolation",
    "KineticPairEventPredicates",
    "KineticRayPowerDifference",
    "KineticSiteStorageReport",
    "discover_kinetic_power_word_at_time",
    "affine_kinetic_sites_from_identity_spd4_sites",
    "kinetic_adjacent_cut_concurrence",
    "kinetic_pair_ray_power_difference",
    "isolate_kinetic_adjacent_cut_concurrence",
    "isolate_kinetic_pair_events",
    "kinetic_pair_event_predicates",
]
