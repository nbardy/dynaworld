"""Exact restricted multichart root persistence for kinetic WorldFoam.

This CPU reference certifies one exact directional homotopy between two
stored binary64 geometries,

``theta(eta) = theta_base + eta * (theta_candidate-theta_base)``,
``(t,eta) in [t_min,t_max] x [0,1]``.

The result is deliberately fail closed.  It supports only separated singleton
simple roots.  Repeated roots, algebraically shared roots, endpoint roots,
persistent-zero predicates, and ambiguous semantic classifications require a
full recompilation.

The important completeness step is that the registry is reconstructed from
*every base owner word*, including predicates that have no root at the base
geometry.  Retaining only the root records emitted by the existing compiler
would be unsound: a rootless predicate can acquire a pair of roots during an
update.  For every registry polynomial this module proves, with exact tensor-
product Bernstein bounds, either:

* one monotone root graph in its assigned rational time neighborhood; or
* a strict fixed sign on every complementary time strip.

The three predicate classes are kept separate:

* ``topology_event_candidate``: near/far owner contacts and active-cut
  competitor concurrences;
* ``analytic_guard``: pair cut denominators, with the corresponding intercept
  certified nonzero on every denominator-root tube to reject full-fiber ties;
* ``nonroot_validity_guard``: currently the ray noncollapse predicate
  ``||d(t,eta)||^2 > 0``.

An algebraic topology-candidate root is counted as a semantic event only when
the exact left and right positive-length owner words differ.  Analytic roots
never increment the semantic event count.  This module performs no requested-
frame sampling and stores no frame-indexed state.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

import torch
from kinetic_active_owner_chart_compiler import ActiveKineticOwnerChartProgram
from kinetic_geometry_trust_region import (
    ExactKineticGeometryDirection,
    RationalBivariatePolynomial,
    build_exact_kinetic_geometry_polynomials,
    dot_exact_bivariate_polynomials,
    exact_power_to_bernstein,
    finite_f64_cpu_tensor,
    fraction_rows_from_binary64,
    fraction_vector_from_binary64,
    subtract_fraction_rows,
)
from kinetic_power_word_compiler import (
    AffineKineticPowerSites,
    discover_kinetic_power_word_at_time,
)
from power_topology_event_predicates import CertifiedEventRoot, RationalPolynomial
from rational_polynomial_roots import (
    isolate_rational_polynomial_roots,
    rational_polynomial_gcd,
)

PredicateClass = Literal[
    "topology_event_candidate",
    "analytic_guard",
    "nonroot_validity_guard",
]


@dataclass(frozen=True)
class DirectionalPredicateSource:
    """One complete-registry predicate with explicit semantics and provenance."""

    source_id: int
    predicate_class: PredicateClass
    kind: str
    site_ids: tuple[int, ...]
    polynomial: RationalBivariatePolynomial
    witness_owner_words: tuple[tuple[int, ...], ...]
    derivation: str
    representation_chart_split_required: bool = False
    nonzero_at_root_companion: RationalBivariatePolynomial | None = None

    @property
    def base_polynomial(self) -> RationalPolynomial:
        return self.polynomial.at_step(Fraction(0))


@dataclass(frozen=True)
class DirectionalRootContinuation:
    """One singleton simple root continued through a rational root tube."""

    root_id: int
    source: DirectionalPredicateSource
    base_root: CertifiedEventRoot
    neighborhood_lower: Fraction
    neighborhood_upper: Fraction
    candidate_root: CertifiedEventRoot
    derivative_sign: int
    base_left_owner_word: tuple[int, ...]
    base_right_owner_word: tuple[int, ...]
    candidate_left_owner_word: tuple[int, ...]
    candidate_right_owner_word: tuple[int, ...]
    semantic_owner_change: bool
    representation_chart_split_required: bool
    root_tube_certified: bool = True
    algebraically_shared: bool = False


@dataclass(frozen=True)
class KineticSimpleRootReisolationCertificate:
    """Sound endpoint repair result for one exact binary64 update segment."""

    passed: bool
    reason: str
    predicate_sources: tuple[DirectionalPredicateSource, ...]
    root_continuations: tuple[DirectionalRootContinuation, ...]
    base_owner_words: tuple[tuple[int, ...], ...]
    candidate_owner_words: tuple[tuple[int, ...], ...]
    topology_source_count: int
    analytic_source_count: int
    nonroot_guard_count: int
    semantic_event_count: int
    representation_chart_split_count: int
    exact_rational_arithmetic: bool = True
    continuous_homotopy_proof: bool = False
    root_complements_certified: bool = False
    semantic_reclassification_performed: bool = False
    requested_frame_sampling_used: bool = False
    requested_step_radius: Fraction = Fraction(1)
    certified_step_radius: Fraction = Fraction(0)
    theorem_scope: str = "separated_singleton_simple_roots_exact_binary64_directional_segment"

    @property
    def requested_radius_certified(self) -> bool:
        return self.passed and self.certified_step_radius == self.requested_step_radius

    @property
    def full_recompile_required(self) -> bool:
        return not self.requested_radius_certified


@dataclass(frozen=True)
class _PairDifference:
    slope: RationalBivariatePolynomial
    intercept: RationalBivariatePolynomial


@dataclass(frozen=True)
class _RawRoot:
    source: DirectionalPredicateSource
    root: CertifiedEventRoot


@dataclass(frozen=True)
class _RootTube:
    raw: _RawRoot
    lower: Fraction
    upper: Fraction
    derivative_sign: int = 0


class _CertificationFailure(Exception):  # noqa: N818 - internal fail-closed control flow
    pass


def certify_multichart_simple_root_binary64_candidate(  # noqa: TRY301
    sites: AffineKineticPowerSites,
    ray_coefficients: torch.Tensor,
    program: ActiveKineticOwnerChartProgram,
    candidate_sites: AffineKineticPowerSites,
    candidate_ray_coefficients: torch.Tensor,
    *,
    max_root_interval_width: Fraction = Fraction(1, 1 << 40),
    max_root_bisection_depth: int = 192,
    max_root_refinements: int = 48,
    max_bernstein_subdivision_depth: int = 14,
) -> KineticSimpleRootReisolationCertificate:
    """Certify and re-isolate separated simple roots at a rounded candidate.

    API/schema errors raise.  Unsupported mathematics returns ``passed=False``
    and requires the existing full exact compiler.  The candidate is reusable
    only when ``requested_radius_certified`` is true.
    """

    if not isinstance(sites, AffineKineticPowerSites) or not isinstance(candidate_sites, AffineKineticPowerSites):
        raise TypeError("sites and candidate_sites must be AffineKineticPowerSites")
    if not isinstance(program, ActiveKineticOwnerChartProgram):
        raise TypeError("program must be ActiveKineticOwnerChartProgram")
    if candidate_sites.positions0.shape != sites.positions0.shape:
        raise ValueError("candidate positions0 shape changed")
    if candidate_sites.velocities.shape != sites.velocities.shape:
        raise ValueError("candidate velocities shape changed")
    if candidate_sites.weight_coefficients.shape != sites.weight_coefficients.shape:
        raise ValueError("candidate weight_coefficients shape changed")
    if max_root_interval_width <= 0:
        raise ValueError("max_root_interval_width must be positive")
    if min(max_root_bisection_depth, max_root_refinements, max_bernstein_subdivision_depth) < 1:
        raise ValueError("root and Bernstein budgets must be positive")
    ray = finite_f64_cpu_tensor(ray_coefficients, name="ray_coefficients")
    candidate_ray = finite_f64_cpu_tensor(candidate_ray_coefficients, name="candidate_ray_coefficients")
    if tuple(ray.shape) != (12,) or tuple(candidate_ray.shape) != (12,):
        raise ValueError("base and candidate rays must have shape [12]")

    sources: tuple[DirectionalPredicateSource, ...] = ()
    try:
        _validate_base_program(sites, ray, program)
        direction = _exact_endpoint_direction(sites, ray, candidate_sites, candidate_ray)
        geometry = build_exact_kinetic_geometry_polynomials(sites, ray, direction)
        sources = _build_complete_registry(sites, program, geometry)
        _validate_program_root_source_provenance(program, sources)
        root_sources = tuple(source for source in sources if source.predicate_class != "nonroot_validity_guard")
        nonroot_sources = tuple(source for source in sources if source.predicate_class == "nonroot_validity_guard")
        raw_roots = _isolate_base_roots(
            root_sources,
            t_min=program.t_min,
            t_max=program.t_max,
            max_interval_width=max_root_interval_width,
            max_bisection_depth=max_root_bisection_depth,
        )
        ordered = _separate_and_reject_shared_roots(
            raw_roots,
            t_min=program.t_min,
            t_max=program.t_max,
            max_interval_width=max_root_interval_width,
            max_bisection_depth=max_root_bisection_depth,
            max_refinements=max_root_refinements,
        )
        tubes = _make_disjoint_root_tubes(ordered, t_min=program.t_min, t_max=program.t_max)
        for source in nonroot_sources:
            if source.kind != "ray_speed_squared":
                raise _CertificationFailure("unsupported_nonroot_validity_guard")  # noqa: TRY301
            _require_box_sign(
                source.polynomial,
                time_lower=program.t_min,
                time_upper=program.t_max,
                step_lower=Fraction(0),
                step_upper=Fraction(1),
                expected_sign=1,
                max_subdivision_depth=max_bernstein_subdivision_depth,
                failure="nonroot_guard_not_uniformly_positive",
            )
        certified_tubes = _certify_root_tubes_and_complements(
            root_sources,
            tubes,
            t_min=program.t_min,
            t_max=program.t_max,
            max_subdivision_depth=max_bernstein_subdivision_depth,
        )

        candidate_roots = tuple(
            _reisolate_candidate_root(
                tube,
                max_interval_width=max_root_interval_width,
                max_bisection_depth=max_root_bisection_depth,
            )
            for tube in certified_tubes
        )
        base_cell_words = _owner_words_between_roots(
            sites,
            ray,
            roots=tuple(tube.raw.root for tube in certified_tubes),
            t_min=program.t_min,
            t_max=program.t_max,
            near=program.near,
            far=program.far,
        )
        candidate_cell_words = _owner_words_between_roots(
            candidate_sites,
            candidate_ray,
            roots=candidate_roots,
            t_min=program.t_min,
            t_max=program.t_max,
            near=program.near,
            far=program.far,
        )
        if _compress_words(base_cell_words) != tuple(chart.owner_word for chart in program.charts):
            raise _CertificationFailure("complete_registry_disagrees_with_base_program")  # noqa: TRY301

        continuations: list[DirectionalRootContinuation] = []
        for root_id, (tube, candidate_root) in enumerate(zip(certified_tubes, candidate_roots, strict=True)):
            source = tube.raw.source
            base_left, base_right = base_cell_words[root_id : root_id + 2]
            candidate_left, candidate_right = candidate_cell_words[root_id : root_id + 2]
            base_change = base_left != base_right
            candidate_change = candidate_left != candidate_right
            if source.predicate_class == "analytic_guard" and (base_change or candidate_change):
                raise _CertificationFailure("analytic_guard_changed_owner_word")  # noqa: TRY301
            if source.predicate_class == "topology_event_candidate":
                if base_change != candidate_change:
                    raise _CertificationFailure("topology_root_semantic_activity_changed")  # noqa: TRY301
                if (base_left, base_right) != (candidate_left, candidate_right):
                    raise _CertificationFailure("topology_root_left_right_words_changed")  # noqa: TRY301
            chart_split = (
                source.predicate_class == "analytic_guard"
                and source.representation_chart_split_required
                and any(
                    _pair_is_adjacent(source.site_ids, word)
                    for word in (base_left, base_right, candidate_left, candidate_right)
                )
            )
            continuations.append(
                DirectionalRootContinuation(
                    root_id=root_id,
                    source=source,
                    base_root=tube.raw.root,
                    neighborhood_lower=tube.lower,
                    neighborhood_upper=tube.upper,
                    candidate_root=candidate_root,
                    derivative_sign=tube.derivative_sign,
                    base_left_owner_word=base_left,
                    base_right_owner_word=base_right,
                    candidate_left_owner_word=candidate_left,
                    candidate_right_owner_word=candidate_right,
                    semantic_owner_change=base_change,
                    representation_chart_split_required=chart_split,
                )
            )

        semantic = tuple(
            item
            for item in continuations
            if item.source.predicate_class == "topology_event_candidate" and item.semantic_owner_change
        )
        if len(semantic) != len(program.active_event_guards):
            raise _CertificationFailure("semantic_event_count_disagrees_with_base_program")  # noqa: TRY301
        if _compress_words(candidate_cell_words) != tuple(chart.owner_word for chart in program.charts):
            raise _CertificationFailure("candidate_chart_word_sequence_changed")  # noqa: TRY301
        return KineticSimpleRootReisolationCertificate(
            passed=True,
            reason="requested_binary64_candidate_root_reisolation_certified",
            predicate_sources=sources,
            root_continuations=tuple(continuations),
            base_owner_words=_compress_words(base_cell_words),
            candidate_owner_words=_compress_words(candidate_cell_words),
            topology_source_count=sum(source.predicate_class == "topology_event_candidate" for source in sources),
            analytic_source_count=sum(source.predicate_class == "analytic_guard" for source in sources),
            nonroot_guard_count=len(nonroot_sources),
            semantic_event_count=len(semantic),
            representation_chart_split_count=sum(item.representation_chart_split_required for item in continuations),
            continuous_homotopy_proof=True,
            root_complements_certified=True,
            semantic_reclassification_performed=True,
            certified_step_radius=Fraction(1),
        )
    except _CertificationFailure as failure:
        return _failure_result(str(failure), sources)


def _validate_base_program(
    sites: AffineKineticPowerSites,
    ray: torch.Tensor,
    program: ActiveKineticOwnerChartProgram,
) -> None:
    if not program.passed or not program.continuous_time_coverage or not program.owner_identity_certified:
        raise _CertificationFailure("base_program_not_continuously_certified")
    if len(program.charts) < 2 or not program.active_event_guards:
        raise _CertificationFailure("base_program_is_not_multichart")
    if program.endpoint_event_guards:
        raise _CertificationFailure("endpoint_root_requires_full_recompile")
    if program.work.site_count != sites.site_count:
        raise _CertificationFailure("base_program_site_count_mismatch")
    for guard in (*program.active_event_guards, *program.inactive_event_guards):
        if guard.simultaneous_source_count != 1 or guard.source_multiplicities != (1,):
            raise _CertificationFailure("shared_repeated_or_ambiguous_base_root")
    for chart in program.charts:
        for witness in chart.witness_certificates:
            observed = _owner_word_at(
                sites,
                ray,
                time=witness.time,
                near=program.near,
                far=program.far,
            )
            if observed != witness.owners:
                raise _CertificationFailure("base_geometry_does_not_match_program_provenance")


def _build_complete_registry(
    sites: AffineKineticPowerSites,
    program: ActiveKineticOwnerChartProgram,
    geometry: tuple[
        tuple[tuple[RationalBivariatePolynomial, ...], ...],
        tuple[RationalBivariatePolynomial, ...],
        tuple[RationalBivariatePolynomial, ...],
        tuple[RationalBivariatePolynomial, ...],
    ],
) -> tuple[DirectionalPredicateSource, ...]:
    positions, weights, origin, direction = geometry
    cache: dict[tuple[int, int], _PairDifference] = {}

    def difference(pair: tuple[int, int]) -> _PairDifference:
        cached = cache.get(pair)
        if cached is not None:
            return cached
        left, right = pair
        separation = tuple(positions[right][axis] - positions[left][axis] for axis in range(3))
        normal = tuple(component.scale(2) for component in separation)
        result = _PairDifference(
            slope=dot_exact_bivariate_polynomials(normal, direction),
            intercept=(
                dot_exact_bivariate_polynomials(normal, origin)
                + dot_exact_bivariate_polynomials(positions[left], positions[left])
                - dot_exact_bivariate_polynomials(positions[right], positions[right])
                - weights[left]
                + weights[right]
            ),
        )
        cache[pair] = result
        return result

    entries: dict[
        tuple[PredicateClass, str, tuple[int, ...], tuple[tuple[int, int, Fraction], ...]],
        tuple[
            RationalBivariatePolynomial,
            set[tuple[int, ...]],
            str,
            bool,
            RationalBivariatePolynomial | None,
        ],
    ] = {}

    def register(
        predicate_class: PredicateClass,
        kind: str,
        site_ids: tuple[int, ...],
        polynomial: RationalBivariatePolynomial,
        owner_word: tuple[int, ...],
        derivation: str,
        *,
        chart_split: bool = False,
        companion: RationalBivariatePolynomial | None = None,
    ) -> None:
        key = (predicate_class, kind, site_ids, polynomial.terms)
        current = entries.get(key)
        if current is None:
            entries[key] = (polynomial, {owner_word}, derivation, chart_split, companion)
        else:
            if current[4] != companion:
                raise ArithmeticError("canonical predicate companion changed across owner words")
            current[1].add(owner_word)
            entries[key] = (current[0], current[1], current[2], current[3] or chart_split, current[4])

    active_pairs: set[tuple[int, int]] = set()
    owner_pairs: set[tuple[int, int]] = set()
    owner_words = tuple(dict.fromkeys(chart.owner_word for chart in program.charts))
    for word in owner_words:
        if not word:
            raise _CertificationFailure("empty_base_owner_word")
        for kind, owner, depth in (
            ("pair_near", word[0], program.near),
            ("pair_far", word[-1], program.far),
        ):
            for competitor in range(sites.site_count):
                if competitor == owner:
                    continue
                pair = tuple(sorted((owner, competitor)))
                delta = difference(pair)
                register(
                    "topology_event_candidate",
                    kind,
                    pair,
                    delta.intercept + delta.slope.scale(depth),
                    word,
                    f"B_ij+{kind.removeprefix('pair_')}*A_ij=0",
                )
        for left, right in zip(word, word[1:], strict=False):
            pair = tuple(sorted((left, right)))
            active_pairs.add(pair)
            for competitor in range(sites.site_count):
                if competitor in pair:
                    continue
                triple = tuple(sorted((*pair, competitor)))
                first = difference((triple[0], triple[1]))
                second = difference((triple[1], triple[2]))
                register(
                    "topology_event_candidate",
                    "active_cut_competitor",
                    triple,
                    first.intercept * second.slope - second.intercept * first.slope,
                    word,
                    "B_ab*A_bc-B_bc*A_ab=0",
                )
        for owner in word:
            for competitor in range(sites.site_count):
                if owner != competitor:
                    owner_pairs.add(tuple(sorted((owner, competitor))))

    for pair in sorted(owner_pairs):
        delta = difference(pair)
        witness = tuple(word for word in owner_words if any(site in pair for site in word))
        for word in witness:
            register(
                "analytic_guard",
                "pair_cut_denominator_guard",
                pair,
                delta.slope,
                word,
                "A_ij(t,eta)=0; cut representation guard, not a topology event",
                chart_split=pair in active_pairs,
                companion=delta.intercept,
            )

    register(
        "nonroot_validity_guard",
        "ray_speed_squared",
        (),
        dot_exact_bivariate_polynomials(direction, direction),
        (),
        "||d(t,eta)||^2>0 on the complete update strip",
    )
    result = []
    for source_id, (key, value) in enumerate(sorted(entries.items(), key=lambda item: item[0][:3])):
        predicate_class, kind, site_ids, _ = key
        polynomial, words, derivation, chart_split, companion = value
        if polynomial.identically_zero:
            raise _CertificationFailure(f"persistent_zero_predicate:{kind}:{site_ids}")
        result.append(
            DirectionalPredicateSource(
                source_id=source_id,
                predicate_class=predicate_class,
                kind=kind,
                site_ids=site_ids,
                polynomial=polynomial,
                witness_owner_words=tuple(sorted(words)),
                derivation=derivation,
                representation_chart_split_required=chart_split,
                nonzero_at_root_companion=companion,
            )
        )
    return tuple(result)


def _validate_program_root_source_provenance(
    program: ActiveKineticOwnerChartProgram,
    sources: tuple[DirectionalPredicateSource, ...],
) -> None:
    kind_map = {
        "pair_near": "pair_near",
        "pair_far": "pair_far",
        "active_cut_competitor": "active_cut_competitor",
        "active_cut_denominator": "pair_cut_denominator_guard",
    }
    for guard in (*program.active_event_guards, *program.inactive_event_guards):
        program_source = guard.sources[0]
        expected_kind = kind_map.get(program_source.kind)
        if expected_kind is None:
            raise _CertificationFailure(f"unsupported_program_root_source:{program_source.kind}")
        matches = tuple(
            source
            for source in sources
            if source.kind == expected_kind
            and source.site_ids == program_source.site_ids
            and _polynomials_are_associates(source.base_polynomial, program_source.polynomial)
        )
        if len(matches) != 1:
            raise _CertificationFailure("base_program_root_source_provenance_not_reconstructed")


def _polynomials_are_associates(left: RationalPolynomial, right: RationalPolynomial) -> bool:
    if left.identically_zero or right.identically_zero or left.degree != right.degree:
        return False
    return rational_polynomial_gcd(left, right).degree == left.degree


def _isolate_base_roots(
    sources: tuple[DirectionalPredicateSource, ...],
    *,
    t_min: Fraction,
    t_max: Fraction,
    max_interval_width: Fraction,
    max_bisection_depth: int,
) -> tuple[_RawRoot, ...]:
    roots = []
    for source in sources:
        polynomial = source.base_polynomial
        if polynomial.identically_zero:
            raise _CertificationFailure(f"persistent_zero_predicate:{source.kind}:{source.site_ids}")
        if polynomial.degree == 0:
            continue
        isolation = isolate_rational_polynomial_roots(
            polynomial,
            t_min=t_min,
            t_max=t_max,
            max_interval_width=max_interval_width,
            max_bisection_depth=max_bisection_depth,
        )
        for root in isolation.roots:
            if root.multiplicity != 1:
                raise _CertificationFailure(f"repeated_or_grazing_root:{source.kind}:{source.site_ids}")
            roots.append(_RawRoot(source, root))
    if not roots:
        raise _CertificationFailure("registry_has_no_multichart_roots")
    return tuple(roots)


def _separate_and_reject_shared_roots(
    roots: tuple[_RawRoot, ...],
    *,
    t_min: Fraction,
    t_max: Fraction,
    max_interval_width: Fraction,
    max_bisection_depth: int,
    max_refinements: int,
) -> tuple[_RawRoot, ...]:
    current = list(roots)
    for _ in range(max_refinements + 1):
        current.sort(key=lambda raw: (raw.root.lower_bound, raw.root.upper_bound, raw.source.source_id))
        refine_ids: set[int] = set()
        for left, right in zip(current, current[1:], strict=False):
            if left.root.upper_bound < right.root.lower_bound:
                continue
            if _same_algebraic_root(left, right, max_interval_width, max_bisection_depth):
                raise _CertificationFailure(
                    f"shared_or_simultaneous_algebraic_root:{left.source.kind}:{right.source.kind}"
                )
            if not left.root.exact:
                refine_ids.add(id(left))
            if not right.root.exact:
                refine_ids.add(id(right))
            if left.root.exact and right.root.exact:
                raise _CertificationFailure("distinct_exact_roots_are_not_ordered")
        for raw in current:
            if raw.root.lower_bound <= t_min or raw.root.upper_bound >= t_max:
                if raw.root.exact and raw.root.lower_bound in (t_min, t_max):
                    raise _CertificationFailure("endpoint_root_requires_full_recompile")
                if not raw.root.exact:
                    refine_ids.add(id(raw))
        if not refine_ids:
            return tuple(current)
        current = [
            _refine_root(
                raw,
                t_min=t_min,
                t_max=t_max,
                max_interval_width=min(
                    max_interval_width,
                    raw.root.width / 4 if raw.root.width else max_interval_width,
                ),
                max_bisection_depth=max_bisection_depth,
            )
            if id(raw) in refine_ids
            else raw
            for raw in current
        ]
    raise _CertificationFailure("algebraic_root_separation_budget_exhausted")


def _same_algebraic_root(
    left: _RawRoot,
    right: _RawRoot,
    max_interval_width: Fraction,
    max_bisection_depth: int,
) -> bool:
    lower = max(left.root.lower_bound, right.root.lower_bound)
    upper = min(left.root.upper_bound, right.root.upper_bound)
    if lower > upper:
        return False
    common = rational_polynomial_gcd(left.source.base_polynomial, right.source.base_polynomial)
    if common.degree == 0:
        return False
    if lower == upper:
        return common.evaluate(lower) == 0
    return bool(
        isolate_rational_polynomial_roots(
            common,
            t_min=lower,
            t_max=upper,
            max_interval_width=min(max_interval_width, (upper - lower) / 4),
            max_bisection_depth=max_bisection_depth,
        ).roots
    )


def _refine_root(
    raw: _RawRoot,
    *,
    t_min: Fraction,
    t_max: Fraction,
    max_interval_width: Fraction,
    max_bisection_depth: int,
) -> _RawRoot:
    roots = isolate_rational_polynomial_roots(
        raw.source.base_polynomial,
        t_min=t_min,
        t_max=t_max,
        max_interval_width=max_interval_width,
        max_bisection_depth=max_bisection_depth,
    ).roots
    matches = tuple(
        root
        for root in roots
        if max(root.lower_bound, raw.root.lower_bound) <= min(root.upper_bound, raw.root.upper_bound)
    )
    if len(matches) != 1:
        raise _CertificationFailure("root_identity_lost_during_refinement")
    return _RawRoot(raw.source, matches[0])


def _make_disjoint_root_tubes(
    roots: tuple[_RawRoot, ...],
    *,
    t_min: Fraction,
    t_max: Fraction,
) -> tuple[_RootTube, ...]:
    tubes = []
    for index, raw in enumerate(roots):
        previous_upper = t_min if index == 0 else roots[index - 1].root.upper_bound
        next_lower = t_max if index + 1 == len(roots) else roots[index + 1].root.lower_bound
        left_gap = raw.root.lower_bound - previous_upper
        right_gap = next_lower - raw.root.upper_bound
        if left_gap <= 0 or right_gap <= 0:
            raise _CertificationFailure("root_neighborhood_has_no_rational_separation_margin")
        tubes.append(
            _RootTube(
                raw=raw,
                lower=raw.root.lower_bound - left_gap / 3,
                upper=raw.root.upper_bound + right_gap / 3,
            )
        )
    if any(left.upper >= right.lower for left, right in zip(tubes, tubes[1:], strict=False)):
        raise ArithmeticError("constructed root tubes are not disjoint")
    return tuple(tubes)


def _certify_root_tubes_and_complements(
    sources: tuple[DirectionalPredicateSource, ...],
    tubes: tuple[_RootTube, ...],
    *,
    t_min: Fraction,
    t_max: Fraction,
    max_subdivision_depth: int,
) -> tuple[_RootTube, ...]:
    certified = []
    for tube in tubes:
        source = tube.raw.source
        derivative = source.polynomial.derivative_time()
        sign = _nonzero_reference_sign(
            derivative,
            time_lower=tube.lower,
            time_upper=tube.upper,
        )
        _require_box_sign(
            derivative,
            time_lower=tube.lower,
            time_upper=tube.upper,
            step_lower=Fraction(0),
            step_upper=Fraction(1),
            expected_sign=sign,
            max_subdivision_depth=max_subdivision_depth,
            failure=f"root_derivative_not_uniformly_nonzero:{source.kind}:{source.site_ids}",
        )
        left_sign = _sign(source.base_polynomial.evaluate(tube.lower))
        right_sign = _sign(source.base_polynomial.evaluate(tube.upper))
        if not left_sign or left_sign != -right_sign:
            raise _CertificationFailure("simple_root_tube_does_not_have_opposite_boundary_signs")
        for boundary, expected in ((tube.lower, left_sign), (tube.upper, right_sign)):
            _require_box_sign(
                source.polynomial,
                time_lower=boundary,
                time_upper=boundary,
                step_lower=Fraction(0),
                step_upper=Fraction(1),
                expected_sign=expected,
                max_subdivision_depth=max_subdivision_depth,
                failure=f"root_left_or_right_boundary_sign_not_preserved:{source.kind}:{source.site_ids}",
            )
        if source.nonzero_at_root_companion is not None:
            companion_sign = _nonzero_reference_sign(
                source.nonzero_at_root_companion,
                time_lower=tube.lower,
                time_upper=tube.upper,
            )
            _require_box_sign(
                source.nonzero_at_root_companion,
                time_lower=tube.lower,
                time_upper=tube.upper,
                step_lower=Fraction(0),
                step_upper=Fraction(1),
                expected_sign=companion_sign,
                max_subdivision_depth=max_subdivision_depth,
                failure=f"full_fiber_tie_on_denominator_root:{source.site_ids}",
            )
        certified.append(_RootTube(tube.raw, tube.lower, tube.upper, derivative_sign=sign))

    tubes_by_source = {
        source.source_id: tuple(tube for tube in certified if tube.raw.source.source_id == source.source_id)
        for source in sources
    }
    for source in sources:
        source_tubes = tubes_by_source[source.source_id]
        cursor = t_min
        for tube in source_tubes:
            if cursor < tube.lower:
                _certify_root_free_strip(
                    source,
                    lower=cursor,
                    upper=tube.lower,
                    max_subdivision_depth=max_subdivision_depth,
                )
            cursor = tube.upper
        if cursor < t_max:
            _certify_root_free_strip(
                source,
                lower=cursor,
                upper=t_max,
                max_subdivision_depth=max_subdivision_depth,
            )
    return tuple(certified)


def _certify_root_free_strip(
    source: DirectionalPredicateSource,
    *,
    lower: Fraction,
    upper: Fraction,
    max_subdivision_depth: int,
) -> None:
    expected = _sign(source.polynomial.evaluate((lower + upper) / 2, Fraction(0)))
    if not expected:
        raise _CertificationFailure("base_complement_witness_is_a_root")
    _require_box_sign(
        source.polynomial,
        time_lower=lower,
        time_upper=upper,
        step_lower=Fraction(0),
        step_upper=Fraction(1),
        expected_sign=expected,
        max_subdivision_depth=max_subdivision_depth,
        failure=f"new_or_uncontrolled_root_in_complement:{source.kind}:{source.site_ids}",
    )


def _reisolate_candidate_root(
    tube: _RootTube,
    *,
    max_interval_width: Fraction,
    max_bisection_depth: int,
) -> CertifiedEventRoot:
    polynomial = tube.raw.source.polynomial.at_step(Fraction(1))
    if polynomial.identically_zero or polynomial.degree == 0:
        raise _CertificationFailure("continued_predicate_lost_isolated_root_semantics")
    roots = isolate_rational_polynomial_roots(
        polynomial,
        t_min=tube.lower,
        t_max=tube.upper,
        max_interval_width=max_interval_width,
        max_bisection_depth=max_bisection_depth,
    ).roots
    if len(roots) != 1 or roots[0].multiplicity != 1:
        raise _CertificationFailure("candidate_root_reisolation_is_not_single_simple")
    return roots[0]


def _owner_words_between_roots(
    sites: AffineKineticPowerSites,
    ray: torch.Tensor,
    *,
    roots: tuple[CertifiedEventRoot, ...],
    t_min: Fraction,
    t_max: Fraction,
    near: Fraction,
    far: Fraction,
) -> tuple[tuple[int, ...], ...]:
    witnesses = []
    cursor = t_min
    for root in roots:
        if cursor >= root.lower_bound:
            raise _CertificationFailure("root_intervals_do_not_leave_a_rational_witness_cell")
        witnesses.append((cursor + root.lower_bound) / 2)
        cursor = root.upper_bound
    if cursor >= t_max:
        raise _CertificationFailure("last_root_does_not_leave_a_rational_witness_cell")
    witnesses.append((cursor + t_max) / 2)
    return tuple(_owner_word_at(sites, ray, time=time, near=near, far=far) for time in witnesses)


def _owner_word_at(
    sites: AffineKineticPowerSites,
    ray: torch.Tensor,
    *,
    time: Fraction,
    near: Fraction,
    far: Fraction,
) -> tuple[int, ...]:
    result = discover_kinetic_power_word_at_time(
        sites,
        ray,
        time=time,
        near=near,
        far=far,
    )
    return tuple(int(owner) for owner in result.word.owners.tolist())


def _compress_words(words: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    result = []
    for word in words:
        if not result or result[-1] != word:
            result.append(word)
    return tuple(result)


def _pair_is_adjacent(pair: tuple[int, ...], word: tuple[int, ...]) -> bool:
    if len(pair) != 2:
        return False
    canonical = tuple(sorted(pair))
    return any(tuple(sorted(adjacent)) == canonical for adjacent in zip(word, word[1:], strict=False))


def _exact_endpoint_direction(
    sites: AffineKineticPowerSites,
    ray: torch.Tensor,
    candidate_sites: AffineKineticPowerSites,
    candidate_ray: torch.Tensor,
) -> ExactKineticGeometryDirection:
    return ExactKineticGeometryDirection(
        positions0=subtract_fraction_rows(
            fraction_rows_from_binary64(candidate_sites.positions0),
            fraction_rows_from_binary64(sites.positions0),
        ),
        velocities=subtract_fraction_rows(
            fraction_rows_from_binary64(candidate_sites.velocities),
            fraction_rows_from_binary64(sites.velocities),
        ),
        weight_coefficients=subtract_fraction_rows(
            fraction_rows_from_binary64(candidate_sites.weight_coefficients),
            fraction_rows_from_binary64(sites.weight_coefficients),
        ),
        ray_coefficients=tuple(
            candidate - base
            for candidate, base in zip(
                fraction_vector_from_binary64(candidate_ray),
                fraction_vector_from_binary64(ray),
                strict=True,
            )
        ),
    )


def _require_box_sign(
    polynomial: RationalBivariatePolynomial,
    *,
    time_lower: Fraction,
    time_upper: Fraction,
    step_lower: Fraction,
    step_upper: Fraction,
    expected_sign: int,
    max_subdivision_depth: int,
    failure: str,
) -> None:
    if expected_sign not in (-1, 1):
        raise ValueError("expected_sign must be -1 or 1")
    pending = [(time_lower, time_upper, step_lower, step_upper, 0)]
    while pending:
        t0, t1, e0, e1, depth = pending.pop()
        lower, upper = _tensor_bernstein_bounds(polynomial, t0=t0, t1=t1, e0=e0, e1=e1)
        if (expected_sign > 0 and lower > 0) or (expected_sign < 0 and upper < 0):
            continue
        if (expected_sign > 0 and upper <= 0) or (expected_sign < 0 and lower >= 0):
            raise _CertificationFailure(failure)
        if depth >= max_subdivision_depth:
            raise _CertificationFailure(failure)
        if t0 < t1 and (e0 == e1 or depth % 2 == 0):
            middle = (t0 + t1) / 2
            pending.append((middle, t1, e0, e1, depth + 1))
            pending.append((t0, middle, e0, e1, depth + 1))
        elif e0 < e1:
            middle = (e0 + e1) / 2
            pending.append((t0, t1, middle, e1, depth + 1))
            pending.append((t0, t1, e0, middle, depth + 1))
        else:
            raise _CertificationFailure(failure)


def _tensor_bernstein_bounds(
    polynomial: RationalBivariatePolynomial,
    *,
    t0: Fraction,
    t1: Fraction,
    e0: Fraction,
    e1: Fraction,
) -> tuple[Fraction, Fraction]:
    time_degree = polynomial.time_degree
    step_degree = polynomial.step_degree
    power = [[Fraction(0)] * (step_degree + 1) for _ in range(time_degree + 1)]
    for time_index, step_index, value in polynomial.terms:
        power[time_index][step_index] += value
    time_bernstein_by_step = [
        exact_power_to_bernstein(
            tuple(power[time_index][step_index] for time_index in range(time_degree + 1)),
            lower=t0,
            upper=t1,
        )
        for step_index in range(step_degree + 1)
    ]
    coefficients = []
    for time_index in range(time_degree + 1):
        step_power = tuple(time_bernstein_by_step[step_index][time_index] for step_index in range(step_degree + 1))
        coefficients.extend(exact_power_to_bernstein(step_power, lower=e0, upper=e1))
    if not coefficients:
        return Fraction(0), Fraction(0)
    return min(coefficients), max(coefficients)


def _nonzero_reference_sign(
    polynomial: RationalBivariatePolynomial,
    *,
    time_lower: Fraction,
    time_upper: Fraction,
) -> int:
    for time in (time_lower, (time_lower + time_upper) / 2, time_upper):
        sign = _sign(polynomial.evaluate(time, Fraction(0)))
        if sign:
            return sign
    raise _CertificationFailure("simple_root_derivative_has_no_nonzero_reference_sign")


def _failure_result(
    reason: str,
    sources: tuple[DirectionalPredicateSource, ...],
) -> KineticSimpleRootReisolationCertificate:
    return KineticSimpleRootReisolationCertificate(
        passed=False,
        reason=reason,
        predicate_sources=sources,
        root_continuations=(),
        base_owner_words=(),
        candidate_owner_words=(),
        topology_source_count=sum(source.predicate_class == "topology_event_candidate" for source in sources),
        analytic_source_count=sum(source.predicate_class == "analytic_guard" for source in sources),
        nonroot_guard_count=sum(source.predicate_class == "nonroot_validity_guard" for source in sources),
        semantic_event_count=0,
        representation_chart_split_count=0,
    )


def _sign(value: Fraction) -> int:
    return (value > 0) - (value < 0)


__all__ = [
    "DirectionalPredicateSource",
    "DirectionalRootContinuation",
    "KineticSimpleRootReisolationCertificate",
    "RationalBivariatePolynomial",
    "certify_multichart_simple_root_binary64_candidate",
]
