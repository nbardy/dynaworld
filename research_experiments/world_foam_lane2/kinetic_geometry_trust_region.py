"""Exact fail-closed trust radius for one event-free kinetic owner chart.

This module certifies a deliberately narrow statement.  Let every kinetic
site and the affine ray follow the exact rational parameter path

``theta(e) = theta_0 + e * delta_theta,  0 <= e <= r``.

If the active direct-kinetic compiler has exactly one owner chart on the
closed time interval and no active or endpoint event, this module can prove
that the *same owner word* remains the unique lower envelope for every
``(t,e)`` in that rectangle.  The proof does not sample time.  It constructs
the exact bivariate pair-power polynomials, reduces word validity to strict
polynomial inequalities, and bounds every update-dependent coefficient with
exact rational Bernstein hulls in time.

For a word ``o_0,...,o_{R-1}``, internal cuts are

``z_q = -B_{o_{q-1},o_q} / A_{o_{q-1},o_q}``,

where ``power_i-power_j = A_ij z + B_ij``.  The certificate proves:

* the affine ray never collapses;
* every active cut denominator has fixed nonzero sign;
* ``near < z_1 < ... < z_{R-1} < far``; and
* each run owner is strictly below every non-adjacent competitor at both run
  endpoints (the defining adjacent equality is the only allowed tie).

Because every pair difference is affine in depth, endpoint dominance implies
dominance throughout the run.  These conditions therefore prove the exact
word, not merely node-local margins.

The proof boundary matters.  A program containing an active event gets a zero
radius here.  In general an arbitrarily small geometry update moves an event
root, so the old numeric chart endpoint is immediately stale.  A useful
multi-chart theorem would instead prove simple-root persistence and event
order, then re-isolate/refit every endpoint.  This module does not do that and
does not differentiate event times, chart endpoints, ranks, or compiler
choices.  It also certifies the exact rational path through the supplied
binary64 base and direction tensors; an optimizer using differently rounded
endpoints must certify the actual rounded candidate before reuse.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from math import comb
from typing import Any

import torch
from kinetic_active_owner_chart_compiler import ActiveKineticOwnerChartProgram
from kinetic_power_word_compiler import AffineKineticPowerSites
from power_topology_event_predicates import RationalPolynomial


@dataclass(frozen=True)
class KineticGeometryUpdateDirection:
    """One exact-rational direction represented by finite binary64 tensors."""

    positions0: torch.Tensor
    velocities: torch.Tensor
    weight_coefficients: torch.Tensor
    ray_coefficients: torch.Tensor


@dataclass(frozen=True)
class StrictPredicateCertificate:
    """One strict ``P(t,e)>0`` proof and its coefficient perturbation bound."""

    kind: str
    site_ids: tuple[int, ...]
    time_degree: int
    step_degree: int
    base_lower_bound: Fraction
    step_coefficient_absolute_bounds: tuple[Fraction, ...]
    time_leaf_count: int
    derivation: str

    def perturbation_bound(self, radius: Fraction) -> Fraction:
        return sum(
            (bound * radius**degree for degree, bound in enumerate(self.step_coefficient_absolute_bounds, 1)),
            Fraction(0),
        )

    def accepts(self, radius: Fraction) -> bool:
        return self.perturbation_bound(radius) < self.base_lower_bound


@dataclass(frozen=True)
class KineticGeometryTrustRegionCertificate:
    """Sound result for the event-free exact-rational directional path."""

    passed: bool
    certified_step_radius: Fraction
    requested_step_radius: Fraction
    owner_word: tuple[int, ...]
    predicate_certificates: tuple[StrictPredicateCertificate, ...]
    reason: str
    limiting_predicate_kind: str | None
    limiting_site_ids: tuple[int, ...]
    continuous_time_proof: bool
    exact_rational_arithmetic: bool = True
    active_event_endpoints_reused: bool = False
    event_root_reisolation_performed: bool = False
    event_time_derivatives_included: bool = False
    chart_endpoint_derivatives_included: bool = False
    compiler_choice_derivatives_included: bool = False
    theorem_scope: str = "event_free_single_chart_exact_directional_path"

    @property
    def recompile_required(self) -> bool:
        return not self.requested_radius_certified

    @property
    def requested_radius_certified(self) -> bool:
        return self.passed and self.certified_step_radius == self.requested_step_radius


@dataclass(frozen=True)
class _PositivePredicate:
    kind: str
    site_ids: tuple[int, ...]
    polynomial: _BivariatePolynomial
    derivation: str


@dataclass(frozen=True)
class _BivariatePolynomial:
    """Sparse exact polynomial ``sum c_ab t^a e^b``."""

    terms: tuple[tuple[int, int, Fraction], ...]

    @classmethod
    def from_terms(
        cls,
        terms: dict[tuple[int, int], Fraction | int],
    ) -> _BivariatePolynomial:
        normalized = tuple(
            (time_degree, step_degree, Fraction(value))
            for (time_degree, step_degree), value in sorted(terms.items())
            if value != 0
        )
        if any(time_degree < 0 or step_degree < 0 for time_degree, step_degree, _ in normalized):
            raise ValueError("polynomial degrees must be nonnegative")
        return cls(normalized)

    @classmethod
    def constant(cls, value: Fraction | int) -> _BivariatePolynomial:
        return cls.from_terms({(0, 0): Fraction(value)})

    @property
    def time_degree(self) -> int:
        return max((time_degree for time_degree, _, _ in self.terms), default=0)

    @property
    def step_degree(self) -> int:
        return max((step_degree for _, step_degree, _ in self.terms), default=0)

    @property
    def identically_zero(self) -> bool:
        return not self.terms

    def derivative_time(self) -> _BivariatePolynomial:
        return _BivariatePolynomial.from_terms(
            {
                (time_degree - 1, step_degree): time_degree * value
                for time_degree, step_degree, value in self.terms
                if time_degree > 0
            }
        )

    def at_step(self, step: Fraction) -> RationalPolynomial:
        coefficients = [Fraction(0)] * (self.time_degree + 1)
        for time_degree, step_degree, value in self.terms:
            coefficients[time_degree] += value * step**step_degree
        return RationalPolynomial(tuple(coefficients))

    def __add__(self, other: _BivariatePolynomial) -> _BivariatePolynomial:
        values = self._as_dict()
        for time_degree, step_degree, value in other.terms:
            key = (time_degree, step_degree)
            values[key] = values.get(key, Fraction(0)) + value
        return _BivariatePolynomial.from_terms(values)

    def __neg__(self) -> _BivariatePolynomial:
        return _BivariatePolynomial.from_terms(
            {(time_degree, step_degree): -value for time_degree, step_degree, value in self.terms}
        )

    def __sub__(self, other: _BivariatePolynomial) -> _BivariatePolynomial:
        return self + (-other)

    def __mul__(self, other: _BivariatePolynomial) -> _BivariatePolynomial:
        values: dict[tuple[int, int], Fraction] = {}
        for left_time, left_step, left_value in self.terms:
            for right_time, right_step, right_value in other.terms:
                key = (left_time + right_time, left_step + right_step)
                values[key] = values.get(key, Fraction(0)) + left_value * right_value
        return _BivariatePolynomial.from_terms(values)

    def scale(self, value: Fraction | int) -> _BivariatePolynomial:
        scale = Fraction(value)
        return _BivariatePolynomial.from_terms(
            {(time_degree, step_degree): scale * coefficient for time_degree, step_degree, coefficient in self.terms}
        )

    def evaluate(self, time: Fraction, step: Fraction) -> Fraction:
        return sum(
            (value * time**time_degree * step**step_degree for time_degree, step_degree, value in self.terms),
            Fraction(0),
        )

    def time_power_coefficients(self, *, step_degree: int) -> tuple[Fraction, ...]:
        degree = max(
            (time_degree for time_degree, eta_degree, _ in self.terms if eta_degree == step_degree),
            default=0,
        )
        coefficients = [Fraction(0)] * (degree + 1)
        for time_degree, eta_degree, value in self.terms:
            if eta_degree == step_degree:
                coefficients[time_degree] += value
        return tuple(coefficients)

    def _as_dict(self) -> dict[tuple[int, int], Fraction]:
        return {(time_degree, step_degree): value for time_degree, step_degree, value in self.terms}


@dataclass(frozen=True)
class _PairDifference:
    slope: _BivariatePolynomial
    intercept: _BivariatePolynomial


@dataclass(frozen=True)
class _BasePositiveProof:
    lower_bound: Fraction
    leaves: tuple[tuple[Fraction, Fraction], ...]


@dataclass(frozen=True)
class _ExactGeometryDirection:
    positions0: tuple[tuple[Fraction, ...], ...]
    velocities: tuple[tuple[Fraction, ...], ...]
    weight_coefficients: tuple[tuple[Fraction, ...], ...]
    ray_coefficients: tuple[Fraction, ...]


class _BaseProofError(Exception):
    pass


def make_kinetic_geometry_update_direction(
    sites: AffineKineticPowerSites,
    ray_coefficients: torch.Tensor,
    *,
    delta_positions0: torch.Tensor | None = None,
    delta_velocities: torch.Tensor | None = None,
    delta_weight_coefficients: torch.Tensor | None = None,
    delta_ray_coefficients: torch.Tensor | None = None,
) -> KineticGeometryUpdateDirection:
    """Normalize a finite binary64 direction against one base geometry."""

    if not isinstance(sites, AffineKineticPowerSites):
        raise TypeError("sites must be AffineKineticPowerSites")
    ray = _finite_f64_cpu(ray_coefficients, name="ray_coefficients")
    if tuple(ray.shape) != (12,):
        raise ValueError("ray_coefficients must have shape [12]")
    return KineticGeometryUpdateDirection(
        positions0=_direction_tensor(delta_positions0, sites.positions0, name="delta_positions0"),
        velocities=_direction_tensor(delta_velocities, sites.velocities, name="delta_velocities"),
        weight_coefficients=_direction_tensor(
            delta_weight_coefficients,
            sites.weight_coefficients,
            name="delta_weight_coefficients",
        ),
        ray_coefficients=_direction_tensor(delta_ray_coefficients, ray, name="delta_ray_coefficients"),
    )


def certify_event_free_binary64_geometry_candidate(
    sites: AffineKineticPowerSites,
    ray_coefficients: torch.Tensor,
    program: ActiveKineticOwnerChartProgram,
    candidate_sites: AffineKineticPowerSites,
    candidate_ray_coefficients: torch.Tensor,
    *,
    maximum_time_subdivision_depth: int = 20,
    radius_search_iterations: int = 48,
) -> KineticGeometryTrustRegionCertificate:
    """Certify the exact segment between two stored binary64 geometries.

    The exact rational difference is formed *after* reading both rounded
    endpoints, avoiding an unsound assumption that a binary64 subtraction
    tensor stores their exact real difference.  The candidate itself is safe
    only when ``result.requested_radius_certified`` is true.  A smaller
    positive radius is a mathematical line-search suggestion; its newly
    rounded endpoint must be passed through this function again.
    """

    if not isinstance(sites, AffineKineticPowerSites) or not isinstance(candidate_sites, AffineKineticPowerSites):
        raise TypeError("sites and candidate_sites must be AffineKineticPowerSites")
    if not isinstance(program, ActiveKineticOwnerChartProgram):
        raise TypeError("program must be ActiveKineticOwnerChartProgram")
    if maximum_time_subdivision_depth < 0:
        raise ValueError("maximum_time_subdivision_depth must be nonnegative")
    if radius_search_iterations < 0:
        raise ValueError("radius_search_iterations must be nonnegative")
    if sites.positions0.shape != candidate_sites.positions0.shape:
        raise ValueError("candidate positions0 shape changed")
    if sites.velocities.shape != candidate_sites.velocities.shape:
        raise ValueError("candidate velocities shape changed")
    if sites.weight_coefficients.shape != candidate_sites.weight_coefficients.shape:
        raise ValueError("candidate weight_coefficients shape changed")
    ray = _finite_f64_cpu(ray_coefficients, name="ray_coefficients")
    candidate_ray = _finite_f64_cpu(candidate_ray_coefficients, name="candidate_ray_coefficients")
    if tuple(ray.shape) != (12,) or tuple(candidate_ray.shape) != (12,):
        raise ValueError("base and candidate rays must have shape [12]")
    exact_direction = _ExactGeometryDirection(
        positions0=_subtract_fraction_rows(
            _fraction_rows(candidate_sites.positions0),
            _fraction_rows(sites.positions0),
        ),
        velocities=_subtract_fraction_rows(
            _fraction_rows(candidate_sites.velocities),
            _fraction_rows(sites.velocities),
        ),
        weight_coefficients=_subtract_fraction_rows(
            _fraction_rows(candidate_sites.weight_coefficients),
            _fraction_rows(sites.weight_coefficients),
        ),
        ray_coefficients=tuple(
            candidate - base
            for candidate, base in zip(
                _fraction_vector(candidate_ray),
                _fraction_vector(ray),
                strict=True,
            )
        ),
    )
    return _certify_event_free_kinetic_geometry_trust_region(
        sites,
        ray,
        program,
        exact_direction,
        requested=Fraction(1),
        maximum_time_subdivision_depth=maximum_time_subdivision_depth,
        radius_search_iterations=radius_search_iterations,
    )


def certify_event_free_kinetic_geometry_trust_region(
    sites: AffineKineticPowerSites,
    ray_coefficients: torch.Tensor,
    program: ActiveKineticOwnerChartProgram,
    direction: KineticGeometryUpdateDirection,
    *,
    requested_step_radius: Fraction | float | int = Fraction(1),
    maximum_time_subdivision_depth: int = 20,
    radius_search_iterations: int = 48,
) -> KineticGeometryTrustRegionCertificate:
    """Certify a nonzero exact-rational radius for one event-free chart.

    Invalid API inputs raise ``ValueError``.  Unsupported or unsafe geometry
    returns a zero-radius fail-closed result.  A passing result proves the
    complete closed time interval, not only compiler nodes.
    """

    requested = _as_positive_fraction(requested_step_radius, name="requested_step_radius")
    if maximum_time_subdivision_depth < 0:
        raise ValueError("maximum_time_subdivision_depth must be nonnegative")
    if radius_search_iterations < 0:
        raise ValueError("radius_search_iterations must be nonnegative")
    if not isinstance(program, ActiveKineticOwnerChartProgram):
        raise TypeError("program must be ActiveKineticOwnerChartProgram")
    if not isinstance(sites, AffineKineticPowerSites):
        raise TypeError("sites must be AffineKineticPowerSites")
    ray = _finite_f64_cpu(ray_coefficients, name="ray_coefficients")
    if tuple(ray.shape) != (12,):
        raise ValueError("ray_coefficients must have shape [12]")
    _validate_direction(direction, sites, ray)

    return _certify_event_free_kinetic_geometry_trust_region(
        sites,
        ray,
        program,
        _exact_direction(direction),
        requested=requested,
        maximum_time_subdivision_depth=maximum_time_subdivision_depth,
        radius_search_iterations=radius_search_iterations,
    )


def _certify_event_free_kinetic_geometry_trust_region(
    sites: AffineKineticPowerSites,
    ray: torch.Tensor,
    program: ActiveKineticOwnerChartProgram,
    direction: _ExactGeometryDirection,
    *,
    requested: Fraction,
    maximum_time_subdivision_depth: int,
    radius_search_iterations: int,
) -> KineticGeometryTrustRegionCertificate:

    empty_word: tuple[int, ...] = ()
    if not program.passed or not program.continuous_time_coverage or not program.owner_identity_certified:
        return _zero_result(
            requested,
            empty_word,
            reason="base_program_not_continuously_certified",
        )
    if len(program.charts) != 1 or program.active_event_guards:
        word = program.charts[0].owner_word if len(program.charts) == 1 else empty_word
        return _zero_result(
            requested,
            word,
            reason="active_or_multichart_program_requires_event_root_reisolation",
        )
    if program.endpoint_event_guards:
        return _zero_result(
            requested,
            program.charts[0].owner_word,
            reason="endpoint_event_requires_a_one_sided_structural_policy",
        )
    if program.work.site_count != sites.site_count:
        return _zero_result(
            requested,
            program.charts[0].owner_word,
            reason="base_program_site_count_mismatch",
        )

    word = program.charts[0].owner_word
    if not word or len(set(word)) != len(word) or min(word) < 0 or max(word) >= sites.site_count:
        return _zero_result(requested, word, reason="base_program_owner_word_is_malformed")

    certificates: list[StrictPredicateCertificate] = []
    try:
        predicates = _word_positive_predicates(
            sites,
            ray,
            direction,
            owner_word=word,
            near=program.near,
            far=program.far,
            time_anchor=(program.t_min + program.t_max) / 2,
        )
        for predicate in predicates:
            certificates.append(
                _certify_positive_predicate(
                    predicate,
                    time_min=program.t_min,
                    time_max=program.t_max,
                    maximum_subdivision_depth=maximum_time_subdivision_depth,
                )
            )
    except _BaseProofError as failure:
        failed_kind, _, failed_ids = str(failure).partition(":")
        site_ids = tuple(int(value) for value in failed_ids.split(",") if value)
        return KineticGeometryTrustRegionCertificate(
            passed=False,
            certified_step_radius=Fraction(0),
            requested_step_radius=requested,
            owner_word=word,
            predicate_certificates=tuple(certificates),
            reason="base_strict_continuous_owner_word_not_certified",
            limiting_predicate_kind=failed_kind,
            limiting_site_ids=site_ids,
            continuous_time_proof=False,
        )

    certified = _largest_bounded_radius(
        tuple(certificates),
        requested=requested,
        search_iterations=radius_search_iterations,
    )
    if certified <= 0:
        raise ArithmeticError("strict base proof failed to produce a positive directional radius")
    limiting = min(
        certificates,
        key=lambda certificate: certificate.base_lower_bound - certificate.perturbation_bound(certified),
    )
    return KineticGeometryTrustRegionCertificate(
        passed=True,
        certified_step_radius=certified,
        requested_step_radius=requested,
        owner_word=word,
        predicate_certificates=tuple(certificates),
        reason="requested_radius_certified" if certified == requested else "strict_coefficient_bound_limited_radius",
        limiting_predicate_kind=limiting.kind,
        limiting_site_ids=limiting.site_ids,
        continuous_time_proof=True,
    )


def _word_positive_predicates(
    sites: AffineKineticPowerSites,
    ray: torch.Tensor,
    direction: _ExactGeometryDirection,
    *,
    owner_word: tuple[int, ...],
    near: Fraction,
    far: Fraction,
    time_anchor: Fraction,
) -> tuple[_PositivePredicate, ...]:
    positions, weights, origin, ray_direction = _geometry_polynomials(sites, ray, direction)
    cache: dict[tuple[int, int], _PairDifference] = {}

    def difference(left: int, right: int) -> _PairDifference:
        key = (left, right)
        cached = cache.get(key)
        if cached is not None:
            return cached
        separation = tuple(positions[right][axis] - positions[left][axis] for axis in range(3))
        normal = tuple(component.scale(2) for component in separation)
        slope = _dot_polynomials(normal, ray_direction)
        intercept = (
            _dot_polynomials(normal, origin)
            + _dot_polynomials(positions[left], positions[left])
            - _dot_polynomials(positions[right], positions[right])
            - weights[left]
            + weights[right]
        )
        result = _PairDifference(slope=slope, intercept=intercept)
        cache[key] = result
        return result

    predicates: list[_PositivePredicate] = []
    ray_speed_squared = _dot_polynomials(ray_direction, ray_direction)
    predicates.append(
        _PositivePredicate(
            kind="ray_speed_squared",
            site_ids=(),
            polynomial=ray_speed_squared,
            derivation="||d(t,e)||^2 > 0",
        )
    )

    cut_differences = [difference(left, right) for left, right in zip(owner_word, owner_word[1:], strict=False)]
    cut_signs: list[int] = []
    for cut_id, (owners, cut) in enumerate(
        zip(zip(owner_word, owner_word[1:], strict=False), cut_differences, strict=True)
    ):
        value = cut.slope.evaluate(time_anchor, Fraction(0))
        if value == 0:
            raise _BaseProofError(f"active_cut_denominator:{owners[0]},{owners[1]}")
        sign = 1 if value > 0 else -1
        cut_signs.append(sign)
        predicates.append(
            _PositivePredicate(
                kind="active_cut_denominator",
                site_ids=owners,
                polynomial=cut.slope.scale(sign),
                derivation=f"sign(A_cut_{cut_id}) is fixed and nonzero",
            )
        )

    if cut_differences:
        first = cut_differences[0]
        first_near = first.intercept + first.slope.scale(near)
        predicates.append(
            _PositivePredicate(
                kind="first_cut_above_near",
                site_ids=owner_word[:2],
                polynomial=first_near.scale(-cut_signs[0]),
                derivation="z_1-near=-(B_01+near*A_01)/A_01 > 0",
            )
        )
        last = cut_differences[-1]
        last_far = last.intercept + last.slope.scale(far)
        predicates.append(
            _PositivePredicate(
                kind="last_cut_below_far",
                site_ids=owner_word[-2:],
                polynomial=last_far.scale(cut_signs[-1]),
                derivation="far-z_last=(B_last+far*A_last)/A_last > 0",
            )
        )
        for cut_id, (left, right) in enumerate(zip(cut_differences, cut_differences[1:], strict=False)):
            numerator = left.intercept * right.slope - right.intercept * left.slope
            predicates.append(
                _PositivePredicate(
                    kind="ordered_adjacent_cuts",
                    site_ids=owner_word[cut_id : cut_id + 3],
                    polynomial=numerator.scale(cut_signs[cut_id] * cut_signs[cut_id + 1]),
                    derivation="z_{q+1}-z_q=(B_q*A_{q+1}-B_{q+1}*A_q)/(A_q*A_{q+1}) > 0",
                )
            )

    for run_id, owner in enumerate(owner_word):
        for side in ("left", "right"):
            if side == "left" and run_id == 0:
                boundary_kind = "near_owner_gap"
                boundary_depth = near
                cut_id = None
                allowed_tie = None
            elif side == "right" and run_id == len(owner_word) - 1:
                boundary_kind = "far_owner_gap"
                boundary_depth = far
                cut_id = None
                allowed_tie = None
            else:
                cut_id = run_id - 1 if side == "left" else run_id
                boundary_kind = "internal_cut_owner_gap"
                boundary_depth = None
                allowed_tie = owner_word[run_id - 1] if side == "left" else owner_word[run_id + 1]
            for competitor in range(sites.site_count):
                if competitor in (owner, allowed_tie):
                    continue
                owner_gap = difference(owner, competitor)
                if cut_id is None:
                    if boundary_depth is None:
                        raise ArithmeticError("a fixed owner-gap boundary lost its depth")
                    positive = (owner_gap.intercept + owner_gap.slope.scale(boundary_depth)).scale(-1)
                    derivation = "-(power_owner-power_competitor) at fixed ray endpoint > 0"
                else:
                    cut = cut_differences[cut_id]
                    cross = owner_gap.intercept * cut.slope - owner_gap.slope * cut.intercept
                    positive = cross.scale(-cut_signs[cut_id])
                    derivation = "-(B_ok*A_cut-A_ok*B_cut)/sign(A_cut) > 0"
                predicates.append(
                    _PositivePredicate(
                        kind=boundary_kind,
                        site_ids=(owner, competitor),
                        polynomial=positive,
                        derivation=derivation,
                    )
                )
    return tuple(predicates)


def _geometry_polynomials(
    sites: AffineKineticPowerSites,
    ray: torch.Tensor,
    direction: _ExactGeometryDirection,
) -> tuple[
    tuple[tuple[_BivariatePolynomial, ...], ...],
    tuple[_BivariatePolynomial, ...],
    tuple[_BivariatePolynomial, ...],
    tuple[_BivariatePolynomial, ...],
]:
    base_positions = _fraction_rows(sites.positions0)
    delta_positions = direction.positions0
    base_velocities = _fraction_rows(sites.velocities)
    delta_velocities = direction.velocities
    base_weights = _fraction_rows(sites.weight_coefficients)
    delta_weights = direction.weight_coefficients
    base_ray = _fraction_vector(ray)
    delta_ray = direction.ray_coefficients

    positions = tuple(
        tuple(
            _BivariatePolynomial.from_terms(
                {
                    (0, 0): base_positions[site][axis],
                    (1, 0): base_velocities[site][axis],
                    (0, 1): delta_positions[site][axis],
                    (1, 1): delta_velocities[site][axis],
                }
            )
            for axis in range(3)
        )
        for site in range(sites.site_count)
    )
    weights = tuple(
        _BivariatePolynomial.from_terms(
            {
                **{(degree, 0): value for degree, value in enumerate(base_weights[site])},
                **{(degree, 1): value for degree, value in enumerate(delta_weights[site])},
            }
        )
        for site in range(sites.site_count)
    )

    def ray_vector(offset: int) -> tuple[_BivariatePolynomial, ...]:
        return tuple(
            _BivariatePolynomial.from_terms(
                {
                    (0, 0): base_ray[offset + axis],
                    (1, 0): base_ray[offset + 3 + axis],
                    (0, 1): delta_ray[offset + axis],
                    (1, 1): delta_ray[offset + 3 + axis],
                }
            )
            for axis in range(3)
        )

    return positions, weights, ray_vector(0), ray_vector(6)


def _certify_positive_predicate(
    predicate: _PositivePredicate,
    *,
    time_min: Fraction,
    time_max: Fraction,
    maximum_subdivision_depth: int,
) -> StrictPredicateCertificate:
    base_coefficients = predicate.polynomial.time_power_coefficients(step_degree=0)
    try:
        base = _prove_univariate_positive(
            base_coefficients,
            lower=time_min,
            upper=time_max,
            maximum_depth=maximum_subdivision_depth,
        )
    except _BaseProofError as error:
        ids = ",".join(str(site) for site in predicate.site_ids)
        raise _BaseProofError(f"{predicate.kind}:{ids}") from error
    step_bounds = []
    for step_degree in range(1, predicate.polynomial.step_degree + 1):
        coefficients = predicate.polynomial.time_power_coefficients(step_degree=step_degree)
        step_bounds.append(
            max(
                (
                    max(abs(value) for value in _power_to_bernstein(coefficients, lower, upper))
                    for lower, upper in base.leaves
                ),
                default=Fraction(0),
            )
        )
    return StrictPredicateCertificate(
        kind=predicate.kind,
        site_ids=predicate.site_ids,
        time_degree=predicate.polynomial.time_degree,
        step_degree=predicate.polynomial.step_degree,
        base_lower_bound=base.lower_bound,
        step_coefficient_absolute_bounds=tuple(step_bounds),
        time_leaf_count=len(base.leaves),
        derivation=predicate.derivation,
    )


def _prove_univariate_positive(
    coefficients: tuple[Fraction, ...],
    *,
    lower: Fraction,
    upper: Fraction,
    maximum_depth: int,
) -> _BasePositiveProof:
    pending = [(lower, upper, 0)]
    leaves: list[tuple[Fraction, Fraction]] = []
    lower_bounds: list[Fraction] = []
    while pending:
        cell_lower, cell_upper, depth = pending.pop()
        bernstein = _power_to_bernstein(coefficients, cell_lower, cell_upper)
        bound = min(bernstein)
        if bound > 0:
            leaves.append((cell_lower, cell_upper))
            lower_bounds.append(bound)
            continue
        if max(bernstein) <= 0:
            raise _BaseProofError("the base predicate is nonpositive on a certified time cell")
        if depth >= maximum_depth:
            raise _BaseProofError("strict positivity was not Bernstein-certified")
        midpoint = (cell_lower + cell_upper) / 2
        pending.append((midpoint, cell_upper, depth + 1))
        pending.append((cell_lower, midpoint, depth + 1))
    if not lower_bounds:
        raise _BaseProofError("empty positivity proof")
    return _BasePositiveProof(lower_bound=min(lower_bounds), leaves=tuple(leaves))


def _power_to_bernstein(
    coefficients: tuple[Fraction, ...],
    lower: Fraction,
    upper: Fraction,
) -> tuple[Fraction, ...]:
    """Exact power-to-Bernstein conversion on one rational interval."""

    degree = max(len(coefficients) - 1, 0)
    width = upper - lower
    shifted = [Fraction(0)] * (degree + 1)
    for source_degree, coefficient in enumerate(coefficients):
        for local_degree in range(source_degree + 1):
            shifted[local_degree] += (
                coefficient
                * comb(source_degree, local_degree)
                * lower ** (source_degree - local_degree)
                * width**local_degree
            )
    if degree == 0:
        return (shifted[0],)
    return tuple(
        sum(
            (
                shifted[power_degree] * Fraction(comb(bernstein_degree, power_degree), comb(degree, power_degree))
                for power_degree in range(bernstein_degree + 1)
            ),
            Fraction(0),
        )
        for bernstein_degree in range(degree + 1)
    )


def _largest_bounded_radius(
    certificates: tuple[StrictPredicateCertificate, ...],
    *,
    requested: Fraction,
    search_iterations: int,
) -> Fraction:
    def accepts(radius: Fraction) -> bool:
        return all(certificate.accepts(radius) for certificate in certificates)

    if accepts(requested):
        return requested
    guaranteed = requested
    for certificate in certificates:
        total = sum(certificate.step_coefficient_absolute_bounds, Fraction(0))
        if total > 0:
            guaranteed = min(guaranteed, Fraction(1), certificate.base_lower_bound / (2 * total))
    if guaranteed <= 0 or not accepts(guaranteed):
        raise ArithmeticError("the exact positive-base perturbation bound lost its guaranteed radius")
    lower = guaranteed
    upper = requested
    for _ in range(search_iterations):
        middle = (lower + upper) / 2
        if accepts(middle):
            lower = middle
        else:
            upper = middle
    return lower


def _dot_polynomials(
    left: tuple[_BivariatePolynomial, ...],
    right: tuple[_BivariatePolynomial, ...],
) -> _BivariatePolynomial:
    return sum((a * b for a, b in zip(left, right, strict=True)), _BivariatePolynomial.constant(0))


def _direction_tensor(
    value: torch.Tensor | None,
    reference: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    if value is None:
        return torch.zeros_like(reference)
    tensor = _finite_f64_cpu(value, name=name)
    if tensor.shape != reference.shape:
        raise ValueError(f"{name} must have shape {tuple(reference.shape)}")
    return tensor


def _validate_direction(
    direction: KineticGeometryUpdateDirection,
    sites: AffineKineticPowerSites,
    ray: torch.Tensor,
) -> None:
    if not isinstance(direction, KineticGeometryUpdateDirection):
        raise TypeError("direction must be KineticGeometryUpdateDirection")
    expected = (
        ("positions0", sites.positions0),
        ("velocities", sites.velocities),
        ("weight_coefficients", sites.weight_coefficients),
        ("ray_coefficients", ray),
    )
    for name, reference in expected:
        tensor = _finite_f64_cpu(getattr(direction, name), name=f"direction.{name}")
        if tensor.shape != reference.shape:
            raise ValueError(f"direction.{name} must have shape {tuple(reference.shape)}")


def _exact_direction(direction: KineticGeometryUpdateDirection) -> _ExactGeometryDirection:
    return _ExactGeometryDirection(
        positions0=_fraction_rows(_finite_f64_cpu(direction.positions0, name="direction.positions0")),
        velocities=_fraction_rows(_finite_f64_cpu(direction.velocities, name="direction.velocities")),
        weight_coefficients=_fraction_rows(
            _finite_f64_cpu(direction.weight_coefficients, name="direction.weight_coefficients")
        ),
        ray_coefficients=_fraction_vector(
            _finite_f64_cpu(direction.ray_coefficients, name="direction.ray_coefficients")
        ),
    )


def _zero_result(
    requested: Fraction,
    owner_word: tuple[int, ...],
    *,
    reason: str,
) -> KineticGeometryTrustRegionCertificate:
    return KineticGeometryTrustRegionCertificate(
        passed=False,
        certified_step_radius=Fraction(0),
        requested_step_radius=requested,
        owner_word=owner_word,
        predicate_certificates=(),
        reason=reason,
        limiting_predicate_kind=None,
        limiting_site_ids=(),
        continuous_time_proof=False,
    )


def _finite_f64_cpu(value: Any, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float64, device="cpu").detach().clone()
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return tensor.contiguous()


def _fraction_rows(tensor: torch.Tensor) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(tuple(Fraction.from_float(float(value)) for value in row) for row in tensor.tolist())


def _fraction_vector(tensor: torch.Tensor) -> tuple[Fraction, ...]:
    return tuple(Fraction.from_float(float(value)) for value in tensor.tolist())


def _subtract_fraction_rows(
    left: tuple[tuple[Fraction, ...], ...],
    right: tuple[tuple[Fraction, ...], ...],
) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(a - b for a, b in zip(left_row, right_row, strict=True))
        for left_row, right_row in zip(left, right, strict=True)
    )


def _as_positive_fraction(value: Fraction | float | int, *, name: str) -> Fraction:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a rational, integer, or float")
    if isinstance(value, Fraction):
        result = value
    elif isinstance(value, int):
        result = Fraction(value)
    elif isinstance(value, float) and math.isfinite(value):
        result = Fraction.from_float(value)
    else:
        raise ValueError(f"{name} must be a finite rational, integer, or float")
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


# Shared exact-geometry primitives.  The multichart moving-root certificate
# uses the same rational directional geometry as this event-free certificate;
# exposing aliases here prevents a second subtly different polynomial algebra.
RationalBivariatePolynomial = _BivariatePolynomial
ExactKineticGeometryDirection = _ExactGeometryDirection
build_exact_kinetic_geometry_polynomials = _geometry_polynomials
dot_exact_bivariate_polynomials = _dot_polynomials
exact_power_to_bernstein = _power_to_bernstein
finite_f64_cpu_tensor = _finite_f64_cpu
fraction_rows_from_binary64 = _fraction_rows
fraction_vector_from_binary64 = _fraction_vector
subtract_fraction_rows = _subtract_fraction_rows


__all__ = [
    "ExactKineticGeometryDirection",
    "KineticGeometryTrustRegionCertificate",
    "KineticGeometryUpdateDirection",
    "RationalBivariatePolynomial",
    "StrictPredicateCertificate",
    "build_exact_kinetic_geometry_polynomials",
    "certify_event_free_binary64_geometry_candidate",
    "certify_event_free_kinetic_geometry_trust_region",
    "dot_exact_bivariate_polynomials",
    "exact_power_to_bernstein",
    "finite_f64_cpu_tensor",
    "fraction_rows_from_binary64",
    "fraction_vector_from_binary64",
    "make_kinetic_geometry_update_direction",
    "subtract_fraction_rows",
]
