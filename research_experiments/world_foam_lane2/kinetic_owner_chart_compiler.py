"""Exact continuous owner charts for one affine kinetic WorldFoam ray.

This is a CPU proof compiler, not the production sparse kinetic data
structure.  It exhaustively enumerates the arrangement events at which the
lower envelope of the site power lines can change on a finite ray segment:

* a pair cut reaches ``near`` or ``far``;
* three finite pair cuts concur; or
* two site lines tie along the complete depth fiber.

The last case is rejected.  Pair-cut denominator roots are retained only as
analytic-chart guards; they are not asserted to be topology events.  Every
candidate algebraic root retains its exact polynomial and a certified rational
isolating interval.  Overlapping isolators belonging to distinct roots are
refined until disjoint or compilation fails closed.

Between consecutive guards, an exact rational witness is used to discover the
fixed-time lower-envelope word.  That word is checked against every site at
both endpoints and an interior point of every positive-length depth run.
Completeness of the exhaustive event arrangement then proves that the owner
word is constant on the open time cell.  Candidate guards whose left and right
words agree are removed from the final chart partition, while their proof
records remain attached to the merged chart.

Charts use right-continuous half-open dispatch: ``[a,b)`` except for the final
chart, which includes the domain upper endpoint.  At an ordinary zero-length
birth/death or triple event this may override fixed-time site-id tie-breaking
on a zero-measure stratum.  It is forward-equivalent for finite P0 material;
event-time geometry derivatives are not certified.  Full-fiber ties and
simultaneous active events remain fail-closed.

The compiler has no requested-frame input and stores no per-frame geometry.
Its exhaustive ``O(S^3)`` triple enumeration is a correctness oracle and proof
bridge, not yet the output-sensitive production event finder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import combinations

import torch
from kinetic_power_word_compiler import (
    AffineKineticPowerSites,
    KineticRayPowerDifference,
    discover_kinetic_power_word_at_time,
    kinetic_pair_ray_power_difference,
)
from power_topology_event_predicates import CertifiedEventRoot, RationalPolynomial
from rational_polynomial_roots import (
    isolate_rational_polynomial_roots,
    multiply_rational_polynomials,
    rational_polynomial_gcd,
)
from sparse_power_word_compiler import SparsePowerRayWord


@dataclass(frozen=True)
class KineticEventSource:
    """One exact arrangement predicate contributing an event root."""

    kind: str
    site_ids: tuple[int, ...]
    polynomial: RationalPolynomial
    derivation: str
    analytic_guard_only: bool = False


@dataclass(frozen=True)
class KineticAlgebraicEventGuard:
    """One distinct real event with exact polynomial guard semantics."""

    guard_id: int
    lower_bound: Fraction
    upper_bound: Fraction
    exact: bool
    sources: tuple[KineticEventSource, ...]
    source_multiplicities: tuple[int, ...]
    left_owner_word: tuple[int, ...] = ()
    right_owner_word: tuple[int, ...] = ()
    active_owner_change: bool = False
    algebraic_identity_certified: bool = True
    distinct_neighbor_roots_certified: bool = True
    dispatch_rule: str = (
        "the increasing-time chart owns the seam; its topology may override "
        "site-id tie-breaking only on the zero-measure event stratum"
    )

    @property
    def canonical_polynomial(self) -> RationalPolynomial:
        return self.sources[0].polynomial

    @property
    def simultaneous_source_count(self) -> int:
        return len(self.sources)


@dataclass(frozen=True)
class KineticTimeBoundary:
    """A rational domain endpoint or an exact algebraic event guard."""

    kind: str
    rational_value: Fraction | None
    event_guard: KineticAlgebraicEventGuard | None


@dataclass(frozen=True)
class ExactOwnerWitnessCertificate:
    """Exact all-site lower-envelope checks at one rational time witness."""

    time: Fraction
    owners: tuple[int, ...]
    transition_depths: tuple[Fraction, ...]
    run_count: int
    all_site_endpoint_checks: int
    all_site_interior_checks: int
    adjacent_boundary_equalities_checked: int
    minimum_strict_interior_margin: Fraction
    exact_fraction_arithmetic: bool = True
    all_site_owner_identity_passed: bool = True


@dataclass(frozen=True)
class CertifiedKineticOwnerChart:
    """One right-continuous chart with a constant open-interval owner word."""

    chart_id: int
    left_boundary: KineticTimeBoundary
    right_boundary: KineticTimeBoundary
    representative_word: SparsePowerRayWord
    owner_word: tuple[int, ...]
    witness_certificates: tuple[ExactOwnerWitnessCertificate, ...]
    filtered_inactive_guards: tuple[KineticAlgebraicEventGuard, ...]
    left_closed: bool
    right_closed: bool
    owner_word_constant_on_open_chart: bool = True
    all_site_witness_checks_passed: bool = True
    arrangement_event_completeness_used: bool = True

    @property
    def interval_notation(self) -> str:
        return "[left,right]" if self.right_closed else "[left,right)"


@dataclass(frozen=True)
class KineticChartDegeneracy:
    """A structural ambiguity deliberately preserved instead of sampled."""

    kind: str
    message: str
    site_ids: tuple[int, ...] = ()
    lower_bound: Fraction | None = None
    upper_bound: Fraction | None = None
    polynomial: RationalPolynomial | None = None


@dataclass(frozen=True)
class KineticOwnerChartCompleteness:
    """Evidence and the precise proof boundary of the exhaustive compiler."""

    site_count: int
    pair_count: int
    triple_count: int
    candidate_predicate_count: int
    isolated_candidate_root_count: int
    distinct_event_guard_count: int
    active_event_guard_count: int
    inactive_event_guard_count: int
    all_pairs_checked_for_full_fiber_ties: bool
    all_pair_boundary_predicates_isolated: bool
    all_finite_triple_predicates_isolated: bool
    continuous_ray_direction_certified_nonzero: bool
    requested_frame_sampling_used: bool = False
    proof_scope: str = (
        "exact for one affine ray, affine kinetic 3D sites with degree<=2 weights, "
        "finite near/far, and nondegenerate P0 lower-envelope topology"
    )
    proof_boundary: str = (
        "exhaustive O(S^3) correctness oracle; no event-time geometry derivative, "
        "no projective/non-affine ray, no persistent/simultaneous degeneracy policy, "
        "and no output-sensitive production event discovery"
    )


@dataclass(frozen=True)
class KineticOwnerChartProgram:
    """Exact continuous chart result, or a fail-closed degeneracy report."""

    passed: bool
    t_min: Fraction
    t_max: Fraction
    near: Fraction
    far: Fraction
    charts: tuple[CertifiedKineticOwnerChart, ...]
    active_event_guards: tuple[KineticAlgebraicEventGuard, ...]
    inactive_event_guards: tuple[KineticAlgebraicEventGuard, ...]
    endpoint_event_guards: tuple[KineticAlgebraicEventGuard, ...]
    unresolved_degeneracies: tuple[KineticChartDegeneracy, ...]
    completeness: KineticOwnerChartCompleteness
    seam_policy_id: str = "right_continuous_half_open_v1"
    continuous_time_coverage: bool = False
    owner_identity_certified: bool = False
    requested_frame_sampling_used: bool = False


@dataclass(frozen=True)
class _RawEventRoot:
    source: KineticEventSource
    root: CertifiedEventRoot


@dataclass(frozen=True)
class _RootGroup:
    members: tuple[_RawEventRoot, ...]
    lower_bound: Fraction
    upper_bound: Fraction


class _FailClosedError(Exception):
    def __init__(self, degeneracy: KineticChartDegeneracy) -> None:
        super().__init__(degeneracy.message)
        self.degeneracy = degeneracy


def compile_exact_kinetic_owner_charts(
    sites: AffineKineticPowerSites,
    ray_coefficients: torch.Tensor,
    *,
    t_min: Fraction | float | int,
    t_max: Fraction | float | int,
    near: Fraction | float | int,
    far: Fraction | float | int,
    max_root_interval_width: Fraction = Fraction(1, 1 << 48),
    max_bisection_depth: int = 192,
    max_root_refinements: int = 32,
) -> KineticOwnerChartProgram:
    """Compile exact right-continuous owner charts for one affine ray track.

    Invalid API inputs raise :class:`ValueError`.  Mathematical degeneracies
    produce ``passed=False`` with one explicit unresolved record and no partial
    chart coverage.
    """

    lo = _as_fraction(t_min, name="t_min")
    hi = _as_fraction(t_max, name="t_max")
    near_q = _as_fraction(near, name="near")
    far_q = _as_fraction(far, name="far")
    if hi <= lo:
        raise ValueError("kinetic owner charts require t_min < t_max")
    if far_q <= near_q:
        raise ValueError("kinetic owner charts require near < far")
    if max_root_interval_width <= 0:
        raise ValueError("max_root_interval_width must be positive")
    if max_bisection_depth < 1 or max_root_refinements < 1:
        raise ValueError("root isolation budgets must be positive")
    ray = torch.as_tensor(ray_coefficients, dtype=torch.float64).detach().cpu().clone()
    if ray.shape != (12,) or not bool(torch.isfinite(ray).all().item()):
        raise ValueError("ray_coefficients must be a finite vector with 12 entries")

    pair_count = sites.site_count * (sites.site_count - 1) // 2
    triple_count = sites.site_count * (sites.site_count - 1) * (sites.site_count - 2) // 6
    empty_completeness = KineticOwnerChartCompleteness(
        site_count=sites.site_count,
        pair_count=pair_count,
        triple_count=triple_count,
        candidate_predicate_count=0,
        isolated_candidate_root_count=0,
        distinct_event_guard_count=0,
        active_event_guard_count=0,
        inactive_event_guard_count=0,
        all_pairs_checked_for_full_fiber_ties=False,
        all_pair_boundary_predicates_isolated=False,
        all_finite_triple_predicates_isolated=False,
        continuous_ray_direction_certified_nonzero=False,
    )
    try:
        _certify_continuous_ray_direction(
            ray,
            t_min=lo,
            t_max=hi,
            max_interval_width=max_root_interval_width,
            max_bisection_depth=max_bisection_depth,
        )
        sources, raw_roots, pair_differences = _enumerate_candidate_roots(
            sites,
            ray,
            t_min=lo,
            t_max=hi,
            near=near_q,
            far=far_q,
            max_interval_width=max_root_interval_width,
            max_bisection_depth=max_bisection_depth,
        )
        groups = _separate_and_group_roots(
            raw_roots,
            t_min=lo,
            t_max=hi,
            max_interval_width=max_root_interval_width,
            max_bisection_depth=max_bisection_depth,
            max_refinements=max_root_refinements,
        )
        guards = tuple(_guard_from_group(index, group) for index, group in enumerate(groups))
        endpoint_guards = tuple(guard for guard in guards if guard.exact and guard.lower_bound in (lo, hi))
        for guard in endpoint_guards:
            if guard.simultaneous_source_count > 1:
                raise _FailClosedError(  # noqa: TRY301 - converted to a result below
                    KineticChartDegeneracy(
                        kind="ambiguous_simultaneous_endpoint_event",
                        message=(
                            "multiple exact arrangement predicates share a domain endpoint; "
                            "one-sided active-event classification is insufficient"
                        ),
                        site_ids=_union_site_ids(guard.sources),
                        lower_bound=guard.lower_bound,
                        upper_bound=guard.upper_bound,
                        polynomial=guard.canonical_polynomial,
                    )
                )
        interior_guards = tuple(guard for guard in guards if guard not in endpoint_guards)
        witnesses = _open_cell_witnesses(lo, hi, interior_guards)
        words_and_certificates = tuple(
            _discover_and_certify_witness(
                sites,
                ray,
                pair_differences,
                time=time,
                near=near_q,
                far=far_q,
            )
            for time in witnesses
        )
        classified_guards = []
        for index, guard in enumerate(interior_guards):
            left_word = _owner_tuple(words_and_certificates[index][0])
            right_word = _owner_tuple(words_and_certificates[index + 1][0])
            active = left_word != right_word
            classified = replace(
                guard,
                left_owner_word=left_word,
                right_owner_word=right_word,
                active_owner_change=active,
            )
            if active and classified.simultaneous_source_count > 1:
                raise _FailClosedError(  # noqa: TRY301 - converted to a result below
                    KineticChartDegeneracy(
                        kind="ambiguous_simultaneous_active_event",
                        message=(
                            "multiple exact predicates share one active topology seam; "
                            "a simultaneous-event stratum policy is required"
                        ),
                        site_ids=_union_site_ids(classified.sources),
                        lower_bound=classified.lower_bound,
                        upper_bound=classified.upper_bound,
                        polynomial=classified.canonical_polynomial,
                    )
                )
            if active and classified.sources[0].analytic_guard_only:
                raise _FailClosedError(  # noqa: TRY301 - converted to a result below
                    KineticChartDegeneracy(
                        kind="unclassified_denominator_owner_change",
                        message=(
                            "a pair-denominator analytic guard changed the owner word without "
                            "a boundary or finite-concurrence event"
                        ),
                        site_ids=classified.sources[0].site_ids,
                        lower_bound=classified.lower_bound,
                        upper_bound=classified.upper_bound,
                        polynomial=classified.canonical_polynomial,
                    )
                )
            classified_guards.append(classified)
        classified_tuple = tuple(classified_guards)
        active_guards = tuple(guard for guard in classified_tuple if guard.active_owner_change)
        inactive_guards = tuple(guard for guard in classified_tuple if not guard.active_owner_change)
        charts = _merge_open_cells_into_charts(
            t_min=lo,
            t_max=hi,
            guards=classified_tuple,
            words_and_certificates=words_and_certificates,
        )
        completeness = KineticOwnerChartCompleteness(
            site_count=sites.site_count,
            pair_count=pair_count,
            triple_count=triple_count,
            candidate_predicate_count=len(sources),
            isolated_candidate_root_count=len(raw_roots),
            distinct_event_guard_count=len(guards),
            active_event_guard_count=len(active_guards),
            inactive_event_guard_count=len(inactive_guards),
            all_pairs_checked_for_full_fiber_ties=True,
            all_pair_boundary_predicates_isolated=True,
            all_finite_triple_predicates_isolated=True,
            continuous_ray_direction_certified_nonzero=True,
        )
        return KineticOwnerChartProgram(
            passed=True,
            t_min=lo,
            t_max=hi,
            near=near_q,
            far=far_q,
            charts=charts,
            active_event_guards=active_guards,
            inactive_event_guards=inactive_guards,
            endpoint_event_guards=endpoint_guards,
            unresolved_degeneracies=(),
            completeness=completeness,
            continuous_time_coverage=True,
            owner_identity_certified=True,
        )
    except _FailClosedError as failure:
        return KineticOwnerChartProgram(
            passed=False,
            t_min=lo,
            t_max=hi,
            near=near_q,
            far=far_q,
            charts=(),
            active_event_guards=(),
            inactive_event_guards=(),
            endpoint_event_guards=(),
            unresolved_degeneracies=(failure.degeneracy,),
            completeness=empty_completeness,
        )


def _enumerate_candidate_roots(
    sites: AffineKineticPowerSites,
    ray: torch.Tensor,
    *,
    t_min: Fraction,
    t_max: Fraction,
    near: Fraction,
    far: Fraction,
    max_interval_width: Fraction,
    max_bisection_depth: int,
) -> tuple[
    tuple[KineticEventSource, ...],
    tuple[_RawEventRoot, ...],
    dict[tuple[int, int], KineticRayPowerDifference],
]:
    differences: dict[tuple[int, int], KineticRayPowerDifference] = {}
    sources: list[KineticEventSource] = []
    raw_roots: list[_RawEventRoot] = []
    for left, right in combinations(range(sites.site_count), 2):
        difference = kinetic_pair_ray_power_difference(sites, ray, left, right)
        differences[(left, right)] = difference
        _reject_full_fiber_tie(
            difference,
            t_min=t_min,
            t_max=t_max,
            max_interval_width=max_interval_width,
            max_bisection_depth=max_bisection_depth,
        )
        predicates = (
            KineticEventSource(
                kind="pair_denominator",
                site_ids=(left, right),
                polynomial=difference.depth_slope,
                derivation="A_ij(t)=0; analytic cut chart guard only",
                analytic_guard_only=True,
            ),
            KineticEventSource(
                kind="pair_near_crossing",
                site_ids=(left, right),
                polynomial=_add_scaled(
                    difference.depth_intercept,
                    difference.depth_slope,
                    near,
                ),
                derivation="B_ij(t)+near*A_ij(t)=0",
            ),
            KineticEventSource(
                kind="pair_far_crossing",
                site_ids=(left, right),
                polynomial=_add_scaled(
                    difference.depth_intercept,
                    difference.depth_slope,
                    far,
                ),
                derivation="B_ij(t)+far*A_ij(t)=0",
            ),
        )
        for source in predicates:
            if source.polynomial.identically_zero:
                if source.analytic_guard_only:
                    continue
                raise _FailClosedError(
                    KineticChartDegeneracy(
                        kind="persistent_boundary_tie",
                        message=(f"sites {left}/{right} tie persistently at a finite depth boundary"),
                        site_ids=(left, right),
                        lower_bound=t_min,
                        upper_bound=t_max,
                        polynomial=source.polynomial,
                    )
                )
            sources.append(source)
            raw_roots.extend(
                _isolate_source(
                    source,
                    t_min=t_min,
                    t_max=t_max,
                    max_interval_width=max_interval_width,
                    max_bisection_depth=max_bisection_depth,
                )
            )

    for first, middle, last in combinations(range(sites.site_count), 3):
        first_difference = differences[(first, middle)]
        second_difference = differences[(middle, last)]
        polynomial = _subtract(
            multiply_rational_polynomials(
                first_difference.depth_intercept,
                second_difference.depth_slope,
            ),
            multiply_rational_polynomials(
                second_difference.depth_intercept,
                first_difference.depth_slope,
            ),
        )
        if polynomial.identically_zero:
            if first_difference.depth_slope.identically_zero and second_difference.depth_slope.identically_zero:
                # Three lines are generically parallel in depth; pair boundary
                # predicates already contain every possible ranking change.
                continue
            raise _FailClosedError(
                KineticChartDegeneracy(
                    kind="persistent_triple_concurrence",
                    message=(f"sites {first}/{middle}/{last} have persistently coincident finite cuts"),
                    site_ids=(first, middle, last),
                    lower_bound=t_min,
                    upper_bound=t_max,
                    polynomial=polynomial,
                )
            )
        source = KineticEventSource(
            kind="triple_concurrence",
            site_ids=(first, middle, last),
            polynomial=polynomial,
            derivation="B_ij(t)A_jk(t)-B_jk(t)A_ij(t)=0",
        )
        sources.append(source)
        denominator_product = multiply_rational_polynomials(
            first_difference.depth_slope,
            second_difference.depth_slope,
        )
        for raw in _isolate_source(
            source,
            t_min=t_min,
            t_max=t_max,
            max_interval_width=max_interval_width,
            max_bisection_depth=max_bisection_depth,
        ):
            if _root_is_also_root_of(
                raw,
                denominator_product,
                max_interval_width=max_interval_width,
                max_bisection_depth=max_bisection_depth,
            ):
                # This is a concurrence at infinity. Denominator sources retain
                # the analytic guard, but it is not a finite topology event.
                continue
            raw_roots.append(raw)
    return tuple(sources), tuple(raw_roots), differences


def _certify_continuous_ray_direction(
    ray: torch.Tensor,
    *,
    t_min: Fraction,
    t_max: Fraction,
    max_interval_width: Fraction,
    max_bisection_depth: int,
) -> None:
    values = tuple(Fraction.from_float(float(value)) for value in ray.tolist())
    components = tuple(RationalPolynomial((values[6 + axis], values[9 + axis])) for axis in range(3))
    nonzero = tuple(poly for poly in components if not poly.identically_zero)
    if not nonzero:
        raise _FailClosedError(
            KineticChartDegeneracy(
                kind="persistent_zero_ray_direction",
                message="the affine ray direction is zero throughout the time domain",
                lower_bound=t_min,
                upper_bound=t_max,
            )
        )
    common = nonzero[0]
    for polynomial in nonzero[1:]:
        common = rational_polynomial_gcd(common, polynomial)
    roots = _isolate_polynomial(
        common,
        t_min=t_min,
        t_max=t_max,
        max_interval_width=max_interval_width,
        max_bisection_depth=max_bisection_depth,
    )
    if roots:
        root = roots[0]
        raise _FailClosedError(
            KineticChartDegeneracy(
                kind="zero_ray_direction",
                message=(
                    "all three affine ray-direction components vanish at a certified time; "
                    "continuous ray validity is unresolved"
                ),
                lower_bound=root.lower_bound,
                upper_bound=root.upper_bound,
                polynomial=common,
            )
        )


def _reject_full_fiber_tie(
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
        raise _FailClosedError(
            KineticChartDegeneracy(
                kind="persistent_full_fiber_tie",
                message=(
                    f"sites {difference.left_site_id}/{difference.right_site_id} "
                    "tie along the full depth fiber for the complete time domain"
                ),
                site_ids=(difference.left_site_id, difference.right_site_id),
                lower_bound=t_min,
                upper_bound=t_max,
            )
        )
    common = rational_polynomial_gcd(slope, intercept)
    roots = _isolate_polynomial(
        common,
        t_min=t_min,
        t_max=t_max,
        max_interval_width=max_interval_width,
        max_bisection_depth=max_bisection_depth,
    )
    if roots:
        root = roots[0]
        raise _FailClosedError(
            KineticChartDegeneracy(
                kind="full_fiber_tie",
                message=(
                    f"sites {difference.left_site_id}/{difference.right_site_id} "
                    "tie along the full depth fiber at a certified event time"
                ),
                site_ids=(difference.left_site_id, difference.right_site_id),
                lower_bound=root.lower_bound,
                upper_bound=root.upper_bound,
                polynomial=common,
            )
        )


def _isolate_source(
    source: KineticEventSource,
    *,
    t_min: Fraction,
    t_max: Fraction,
    max_interval_width: Fraction,
    max_bisection_depth: int,
) -> tuple[_RawEventRoot, ...]:
    return tuple(
        _RawEventRoot(source=source, root=root)
        for root in _isolate_polynomial(
            source.polynomial,
            t_min=t_min,
            t_max=t_max,
            max_interval_width=max_interval_width,
            max_bisection_depth=max_bisection_depth,
        )
    )


def _isolate_polynomial(
    polynomial: RationalPolynomial,
    *,
    t_min: Fraction,
    t_max: Fraction,
    max_interval_width: Fraction,
    max_bisection_depth: int,
) -> tuple[CertifiedEventRoot, ...]:
    if polynomial.identically_zero or polynomial.degree == 0:
        return ()
    return isolate_rational_polynomial_roots(
        polynomial,
        t_min=t_min,
        t_max=t_max,
        max_interval_width=max_interval_width,
        max_bisection_depth=max_bisection_depth,
    ).roots


def _root_is_also_root_of(
    raw: _RawEventRoot,
    polynomial: RationalPolynomial,
    *,
    max_interval_width: Fraction,
    max_bisection_depth: int,
) -> bool:
    if polynomial.identically_zero:
        return True
    common = rational_polynomial_gcd(raw.source.polynomial, polynomial)
    if common.degree == 0:
        return False
    if raw.root.exact:
        return common.evaluate(raw.root.lower_bound) == 0
    return bool(
        _isolate_polynomial(
            common,
            t_min=raw.root.lower_bound,
            t_max=raw.root.upper_bound,
            max_interval_width=min(max_interval_width, raw.root.width / 4),
            max_bisection_depth=max_bisection_depth,
        )
    )


def _separate_and_group_roots(
    roots: tuple[_RawEventRoot, ...],
    *,
    t_min: Fraction,
    t_max: Fraction,
    max_interval_width: Fraction,
    max_bisection_depth: int,
    max_refinements: int,
) -> tuple[_RootGroup, ...]:
    working = list(roots)
    for _ in range(max_refinements + 1):
        groups = _group_equal_roots(
            working,
            max_interval_width=max_interval_width,
            max_bisection_depth=max_bisection_depth,
        )
        problematic: set[int] = set()
        ordered = sorted(enumerate(groups), key=lambda item: item[1].lower_bound)
        for ordered_index, (group_index, group) in enumerate(ordered):
            endpoint_value = group.lower_bound if group.lower_bound == group.upper_bound else None
            if endpoint_value not in (t_min, t_max) and (group.lower_bound <= t_min or group.upper_bound >= t_max):
                problematic.add(group_index)
            if ordered_index:
                previous_index, previous = ordered[ordered_index - 1]
                if previous.upper_bound >= group.lower_bound:
                    problematic.update((previous_index, group_index))
        if not problematic:
            return tuple(group for _, group in ordered)
        member_ids = {
            id(member) for group_index in problematic for member in groups[group_index].members if not member.root.exact
        }
        if not member_ids:
            raise _FailClosedError(
                KineticChartDegeneracy(
                    kind="ambiguous_overlapping_algebraic_roots",
                    message="distinct exact event guards could not be ordered",
                    lower_bound=t_min,
                    upper_bound=t_max,
                )
            )
        refined = []
        for raw in working:
            if id(raw) not in member_ids:
                refined.append(raw)
                continue
            target_width = min(max_interval_width, raw.root.width / 4)
            refined.append(
                _refine_raw_root(
                    raw,
                    t_min=t_min,
                    t_max=t_max,
                    max_interval_width=target_width,
                    max_bisection_depth=max_bisection_depth,
                )
            )
        working = refined
    raise _FailClosedError(
        KineticChartDegeneracy(
            kind="ambiguous_overlapping_algebraic_roots",
            message=("distinct overlapping root isolators did not separate within the exact refinement budget"),
            lower_bound=t_min,
            upper_bound=t_max,
        )
    )


def _group_equal_roots(
    roots: list[_RawEventRoot],
    *,
    max_interval_width: Fraction,
    max_bisection_depth: int,
) -> list[_RootGroup]:
    parent = list(range(len(roots)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(roots)):
        for right in range(left + 1, len(roots)):
            if _raw_roots_are_equal(
                roots[left],
                roots[right],
                max_interval_width=max_interval_width,
                max_bisection_depth=max_bisection_depth,
            ):
                union(left, right)
    buckets: dict[int, list[_RawEventRoot]] = {}
    for index, raw in enumerate(roots):
        buckets.setdefault(find(index), []).append(raw)
    groups = []
    for members in buckets.values():
        lower = max(member.root.lower_bound for member in members)
        upper = min(member.root.upper_bound for member in members)
        if lower > upper:
            raise ArithmeticError("equal algebraic roots have disjoint certified intervals")
        groups.append(_RootGroup(tuple(members), lower, upper))
    return groups


def _raw_roots_are_equal(
    left: _RawEventRoot,
    right: _RawEventRoot,
    *,
    max_interval_width: Fraction,
    max_bisection_depth: int,
) -> bool:
    lo = max(left.root.lower_bound, right.root.lower_bound)
    hi = min(left.root.upper_bound, right.root.upper_bound)
    if lo > hi:
        return False
    common = rational_polynomial_gcd(left.source.polynomial, right.source.polynomial)
    if common.degree == 0:
        return False
    if lo == hi:
        return common.evaluate(lo) == 0
    return bool(
        _isolate_polynomial(
            common,
            t_min=lo,
            t_max=hi,
            max_interval_width=min(max_interval_width, (hi - lo) / 4),
            max_bisection_depth=max_bisection_depth,
        )
    )


def _refine_raw_root(
    raw: _RawEventRoot,
    *,
    t_min: Fraction,
    t_max: Fraction,
    max_interval_width: Fraction,
    max_bisection_depth: int,
) -> _RawEventRoot:
    candidates = _isolate_source(
        raw.source,
        t_min=t_min,
        t_max=t_max,
        max_interval_width=max_interval_width,
        max_bisection_depth=max_bisection_depth,
    )
    matches = tuple(
        candidate
        for candidate in candidates
        if _raw_roots_are_equal(
            raw,
            candidate,
            max_interval_width=max_interval_width,
            max_bisection_depth=max_bisection_depth,
        )
    )
    if len(matches) != 1:
        raise _FailClosedError(
            KineticChartDegeneracy(
                kind="root_refinement_identity_failure",
                message="exact root refinement could not preserve one algebraic root identity",
                site_ids=raw.source.site_ids,
                lower_bound=raw.root.lower_bound,
                upper_bound=raw.root.upper_bound,
                polynomial=raw.source.polynomial,
            )
        )
    return matches[0]


def _guard_from_group(index: int, group: _RootGroup) -> KineticAlgebraicEventGuard:
    ordered = tuple(
        sorted(
            group.members,
            key=lambda raw: (raw.source.kind, raw.source.site_ids),
        )
    )
    exact = group.lower_bound == group.upper_bound and all(
        raw.source.polynomial.evaluate(group.lower_bound) == 0 for raw in ordered
    )
    return KineticAlgebraicEventGuard(
        guard_id=index,
        lower_bound=group.lower_bound,
        upper_bound=group.upper_bound,
        exact=exact,
        sources=tuple(raw.source for raw in ordered),
        source_multiplicities=tuple(raw.root.multiplicity for raw in ordered),
    )


def _open_cell_witnesses(
    t_min: Fraction,
    t_max: Fraction,
    guards: tuple[KineticAlgebraicEventGuard, ...],
) -> tuple[Fraction, ...]:
    witnesses = []
    left = t_min
    for guard in guards:
        if guard.lower_bound <= left:
            raise _FailClosedError(
                KineticChartDegeneracy(
                    kind="missing_rational_witness_gap",
                    message="an algebraic guard leaves no certified rational witness on its left",
                    lower_bound=left,
                    upper_bound=guard.upper_bound,
                    polynomial=guard.canonical_polynomial,
                )
            )
        witnesses.append((left + guard.lower_bound) / 2)
        left = guard.upper_bound
    if left >= t_max:
        raise _FailClosedError(
            KineticChartDegeneracy(
                kind="missing_rational_witness_gap",
                message="the final algebraic guard leaves no certified rational witness on its right",
                lower_bound=left,
                upper_bound=t_max,
            )
        )
    witnesses.append((left + t_max) / 2)
    return tuple(witnesses)


def _discover_and_certify_witness(
    sites: AffineKineticPowerSites,
    ray: torch.Tensor,
    pair_differences: dict[tuple[int, int], KineticRayPowerDifference],
    *,
    time: Fraction,
    near: Fraction,
    far: Fraction,
) -> tuple[SparsePowerRayWord, ExactOwnerWitnessCertificate]:
    result = discover_kinetic_power_word_at_time(
        sites,
        ray,
        time=time,
        near=near,
        far=far,
    )
    owners = _owner_tuple(result)
    cuts = (near, *result.transition_depths, far)
    if len(cuts) != len(owners) + 1 or any(left >= right for left, right in zip(cuts, cuts[1:], strict=False)):
        raise ArithmeticError("fixed-time lower envelope emitted a nonpositive run")
    endpoint_checks = 0
    interior_checks = 0
    strict_margins: list[Fraction] = []
    for run_index, owner in enumerate(owners):
        left_depth, right_depth = cuts[run_index], cuts[run_index + 1]
        interior_depth = (left_depth + right_depth) / 2
        for competitor in range(sites.site_count):
            if competitor == owner:
                continue
            for depth in (left_depth, right_depth):
                difference = _evaluate_oriented_difference(
                    pair_differences,
                    owner,
                    competitor,
                    time=time,
                    depth=depth,
                )
                endpoint_checks += 1
                if difference > 0:
                    raise ArithmeticError("fixed-time owner fails an exact all-site depth-endpoint check")
            interior_difference = _evaluate_oriented_difference(
                pair_differences,
                owner,
                competitor,
                time=time,
                depth=interior_depth,
            )
            interior_checks += 1
            if interior_difference >= 0:
                raise ArithmeticError("fixed-time owner is not the unique exact interior minimizer")
            strict_margins.append(-interior_difference)
    for boundary_index, depth in enumerate(result.transition_depths):
        difference = _evaluate_oriented_difference(
            pair_differences,
            owners[boundary_index],
            owners[boundary_index + 1],
            time=time,
            depth=depth,
        )
        if difference != 0:
            raise ArithmeticError("an active adjacent transition is not an exact pair equality")
    return result, ExactOwnerWitnessCertificate(
        time=time,
        owners=owners,
        transition_depths=result.transition_depths,
        run_count=len(owners),
        all_site_endpoint_checks=endpoint_checks,
        all_site_interior_checks=interior_checks,
        adjacent_boundary_equalities_checked=len(result.transition_depths),
        minimum_strict_interior_margin=min(strict_margins, default=Fraction(0)),
    )


def _merge_open_cells_into_charts(
    *,
    t_min: Fraction,
    t_max: Fraction,
    guards: tuple[KineticAlgebraicEventGuard, ...],
    words_and_certificates: tuple[tuple[SparsePowerRayWord, ExactOwnerWitnessCertificate], ...],
) -> tuple[CertifiedKineticOwnerChart, ...]:
    active_indices = [index for index, guard in enumerate(guards) if guard.active_owner_change]
    cell_starts = [0, *[index + 1 for index in active_indices]]
    cell_stops = [*[index + 1 for index in active_indices], len(words_and_certificates)]
    charts = []
    for chart_id, (start, stop) in enumerate(zip(cell_starts, cell_stops, strict=True)):
        entries = words_and_certificates[start:stop]
        owners = {_owner_tuple(word) for word, _ in entries}
        if len(owners) != 1:
            raise ArithmeticError("inactive candidate filtering merged different owner words")
        left_guard = guards[start - 1] if start else None
        right_guard = guards[stop - 1] if stop - 1 < len(guards) else None
        left_boundary = KineticTimeBoundary(
            kind="event" if left_guard is not None else "domain_start",
            rational_value=None if left_guard is not None else t_min,
            event_guard=left_guard,
        )
        right_boundary = KineticTimeBoundary(
            kind="event" if right_guard is not None else "domain_end",
            rational_value=None if right_guard is not None else t_max,
            event_guard=right_guard,
        )
        inactive = tuple(guard for guard in guards[start : max(start, stop - 1)] if not guard.active_owner_change)
        charts.append(
            CertifiedKineticOwnerChart(
                chart_id=chart_id,
                left_boundary=left_boundary,
                right_boundary=right_boundary,
                representative_word=entries[0][0],
                owner_word=next(iter(owners)),
                witness_certificates=tuple(certificate for _, certificate in entries),
                filtered_inactive_guards=inactive,
                left_closed=True,
                right_closed=chart_id == len(cell_starts) - 1,
            )
        )
    return tuple(charts)


def _evaluate_oriented_difference(
    pair_differences: dict[tuple[int, int], KineticRayPowerDifference],
    left: int,
    right: int,
    *,
    time: Fraction,
    depth: Fraction,
) -> Fraction:
    if left < right:
        return pair_differences[(left, right)].evaluate(time=time, depth=depth)
    return -pair_differences[(right, left)].evaluate(time=time, depth=depth)


def _owner_tuple(word: SparsePowerRayWord) -> tuple[int, ...]:
    return tuple(int(owner) for owner in word.word.owners.tolist())


def _union_site_ids(sources: tuple[KineticEventSource, ...]) -> tuple[int, ...]:
    return tuple(sorted({site_id for source in sources for site_id in source.site_ids}))


def _add_scaled(
    base: RationalPolynomial,
    direction: RationalPolynomial,
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


def _subtract(
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


def _as_fraction(value: Fraction | float | int, *, name: str) -> Fraction:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite rational, integer, or float")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if not isinstance(value, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite rational, integer, or float")
    return Fraction.from_float(value)


__all__ = [
    "CertifiedKineticOwnerChart",
    "ExactOwnerWitnessCertificate",
    "KineticAlgebraicEventGuard",
    "KineticChartDegeneracy",
    "KineticEventSource",
    "KineticOwnerChartCompleteness",
    "KineticOwnerChartProgram",
    "KineticTimeBoundary",
    "compile_exact_kinetic_owner_charts",
]
