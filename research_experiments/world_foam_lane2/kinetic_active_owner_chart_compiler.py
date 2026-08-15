"""Exact active-boundary owner charts for one affine kinetic power ray.

This CPU compiler replaces exhaustive all-triple enumeration with a closure
over certificates of the *currently observed* lower-envelope word.  For a
word with ``R`` positive-length runs and ``S`` sites, one certificate round
constructs exactly

``(2 R + 1) (S - 1)``

candidate attempts:

* the near and far owners against every other site;
* every active cut against every other site;
* one denominator guard per active cut; and
* every active owner against every other site for full-fiber classification.

Constant and duplicate predicates are removed, and sources are cached by
unique witnessed owner word.  Candidate construction is therefore
``O(U S R_max)`` across ``U`` unique source words.  The current monotone
closure additionally performs ``W`` root-complement word discoveries and
all-site certifications at ``O(W (S log S + S R_max))``.  Exact low-degree
root isolation has additional rational bit complexity, and exact root ordering
costs ``O(B M log M)`` for ``M`` isolated roots and ``B`` refinement rounds.
Final chart count ``C`` alone is not a sufficient work bound and, without a
certified kinetic-neighbor graph, the closure can approach the exhaustive
universe in the worst case.  This module does not claim otherwise.

The compiler uses a monotone certificate closure.  Every newly observed word
adds its active predicates, each predicate is isolated over the complete time
domain once, and the enlarged exact root arrangement is resampled.  Closure
is reached when every root-complement cell adds no new predicate.  The
first-contact theorem for affine-in-depth competitor gaps then proves that no
owner change is hidden inside a final cell.

Full-fiber ties, persistent active endpoint/triple strata, ray collapse, and
simultaneous active physical events fail closed.  Ordinary inactive/grazing
roots are retained as proof records but merged out of the final half-open,
right-continuous chart partition.  No requested-frame count participates in
compilation or storage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

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
class ActiveKineticCertificateSource:
    """One exact predicate emitted by a current-word certificate."""

    kind: str
    site_ids: tuple[int, ...]
    polynomial: RationalPolynomial
    derivation: str
    analytic_guard_only: bool = False


@dataclass(frozen=True)
class ActiveKineticEventGuard:
    """One exact algebraic event bucket shared by all matching sources."""

    guard_id: int
    lower_bound: Fraction
    upper_bound: Fraction
    exact: bool
    sources: tuple[ActiveKineticCertificateSource, ...]
    source_multiplicities: tuple[int, ...]
    left_owner_word: tuple[int, ...] = ()
    right_owner_word: tuple[int, ...] = ()
    active_owner_change: bool = False
    distinct_neighbor_roots_certified: bool = True
    dispatch_rule: str = (
        "the increasing-time chart owns the seam; zero-length event runs are "
        "transfer identities, while unsupported positive-length ties fail closed"
    )

    @property
    def canonical_polynomial(self) -> RationalPolynomial:
        return self.sources[0].polynomial

    @property
    def simultaneous_source_count(self) -> int:
        return len(self.sources)


@dataclass(frozen=True)
class ActiveKineticTimeBoundary:
    """A rational domain endpoint or exact algebraic event guard."""

    kind: str
    rational_value: Fraction | None
    event_guard: ActiveKineticEventGuard | None


@dataclass(frozen=True)
class ActiveOwnerWitnessCertificate:
    """Exact all-site evidence for one root-complement witness."""

    time: Fraction
    owners: tuple[int, ...]
    transition_depths: tuple[Fraction, ...]
    run_count: int
    endpoint_owner_checks: int
    active_boundary_competitor_checks: int
    active_boundary_equalities_checked: int
    interior_owner_checks: int
    candidate_attempt_count: int
    candidate_attempt_bound: int
    exact_fraction_arithmetic: bool = True
    all_site_owner_identity_passed: bool = True
    active_boundary_certificate_complete: bool = True


@dataclass(frozen=True)
class ActiveCertifiedKineticOwnerChart:
    """One constant-word right-continuous chart."""

    chart_id: int
    left_boundary: ActiveKineticTimeBoundary
    right_boundary: ActiveKineticTimeBoundary
    representative_word: SparsePowerRayWord
    owner_word: tuple[int, ...]
    witness_certificates: tuple[ActiveOwnerWitnessCertificate, ...]
    filtered_inactive_guards: tuple[ActiveKineticEventGuard, ...]
    left_closed: bool
    right_closed: bool
    owner_word_constant_on_open_chart: bool = True
    all_site_witness_checks_passed: bool = True
    active_boundary_event_completeness_used: bool = True

    @property
    def interval_notation(self) -> str:
        return "[left,right]" if self.right_closed else "[left,right)"


@dataclass(frozen=True)
class ActiveKineticCompilerDegeneracy:
    """A condition deliberately surfaced rather than sampled through."""

    kind: str
    message: str
    site_ids: tuple[int, ...] = ()
    lower_bound: Fraction | None = None
    upper_bound: Fraction | None = None
    polynomial: RationalPolynomial | None = None


@dataclass(frozen=True)
class ActiveKineticCompilerWork:
    """Auditable structural work; rational bit complexity is stated separately."""

    site_count: int
    certificate_round_count: int
    root_complement_witness_count: int
    witness_word_discovery_count: int
    candidate_source_attempt_count: int
    unique_source_word_count: int
    unique_candidate_source_count: int
    root_isolation_call_count: int
    isolated_raw_root_count: int
    distinct_event_guard_count: int
    pair_difference_request_count: int
    unique_pair_difference_count: int
    all_site_witness_check_count: int
    algebraic_root_refinement_count: int
    max_run_count: int
    sum_site_run_products: int
    per_witness_candidate_bound_verified: bool
    exhaustive_triple_enumeration_used: bool = False
    requested_frame_sampling_used: bool = False
    structural_complexity: str = (
        "O(U*S*R_max) predicate construction across U unique witnessed owner "
        "words; current closure additionally spends O(W*(S log S + S*R_max)) "
        "on W cumulative root-complement discoveries/certificates, excluding "
        "exact rational root bit complexity"
    )
    root_ordering_complexity: str = (
        "O(B*M log M) interval ordering/refinement for M cached roots and B "
        "separation rounds; no pairwise all-root clustering"
    )
    limitation: str = (
        "without a certified kinetic regular/Delaunay or conservative neighbor "
        "supergraph, the all-competitor S factor and worst-case global predicate "
        "universe remain"
    )


@dataclass(frozen=True)
class ActiveKineticOwnerChartProgram:
    """Active-boundary chart result or one fail-closed degeneracy."""

    passed: bool
    t_min: Fraction
    t_max: Fraction
    near: Fraction
    far: Fraction
    charts: tuple[ActiveCertifiedKineticOwnerChart, ...]
    active_event_guards: tuple[ActiveKineticEventGuard, ...]
    inactive_event_guards: tuple[ActiveKineticEventGuard, ...]
    endpoint_event_guards: tuple[ActiveKineticEventGuard, ...]
    root_complement_witnesses: tuple[ActiveOwnerWitnessCertificate, ...]
    unresolved_degeneracies: tuple[ActiveKineticCompilerDegeneracy, ...]
    work: ActiveKineticCompilerWork
    seam_policy_id: str = "right_continuous_half_open_v1"
    continuous_time_coverage: bool = False
    owner_identity_certified: bool = False
    requested_frame_sampling_used: bool = False


@dataclass(frozen=True)
class _RawRoot:
    source: ActiveKineticCertificateSource
    root: CertifiedEventRoot


@dataclass(frozen=True)
class _RootGroup:
    members: tuple[_RawRoot, ...]
    lower_bound: Fraction
    upper_bound: Fraction


@dataclass(frozen=True)
class _CellWitness:
    left_event_index: int | None
    right_event_index: int | None
    result: SparsePowerRayWord
    certificate: ActiveOwnerWitnessCertificate


@dataclass
class _MutableWork:
    certificate_round_count: int = 0
    witness_word_discovery_count: int = 0
    candidate_source_attempt_count: int = 0
    unique_source_word_count: int = 0
    root_isolation_call_count: int = 0
    isolated_raw_root_count: int = 0
    pair_difference_request_count: int = 0
    all_site_witness_check_count: int = 0
    algebraic_root_refinement_count: int = 0
    max_run_count: int = 0
    sum_site_run_products: int = 0
    per_witness_candidate_bound_verified: bool = True


class _FailClosedError(Exception):
    def __init__(self, degeneracy: ActiveKineticCompilerDegeneracy) -> None:
        super().__init__(degeneracy.message)
        self.degeneracy = degeneracy


class _PairDifferenceCache:
    def __init__(
        self,
        sites: AffineKineticPowerSites,
        ray: torch.Tensor,
        work: _MutableWork,
    ) -> None:
        self.sites = sites
        self.ray = ray
        self.work = work
        self.values: dict[tuple[int, int], KineticRayPowerDifference] = {}

    def canonical(self, left: int, right: int) -> KineticRayPowerDifference:
        if left == right:
            raise ValueError("a pair difference requires distinct sites")
        self.work.pair_difference_request_count += 1
        pair = tuple(sorted((left, right)))
        if pair not in self.values:
            self.values[pair] = kinetic_pair_ray_power_difference(
                self.sites,
                self.ray,
                pair[0],
                pair[1],
            )
        return self.values[pair]

    def oriented(
        self,
        left: int,
        right: int,
    ) -> tuple[RationalPolynomial, RationalPolynomial]:
        difference = self.canonical(left, right)
        if left < right:
            return difference.depth_slope, difference.depth_intercept
        return _negate(difference.depth_slope), _negate(difference.depth_intercept)


def compile_active_kinetic_owner_charts(
    sites: AffineKineticPowerSites,
    ray_coefficients: torch.Tensor,
    *,
    t_min: Fraction | float | int,
    t_max: Fraction | float | int,
    near: Fraction | float | int,
    far: Fraction | float | int,
    max_root_interval_width: Fraction = Fraction(1, 1 << 48),
    max_bisection_depth: int = 192,
    max_root_refinements: int = 48,
    max_certificate_rounds: int = 128,
) -> ActiveKineticOwnerChartProgram:
    """Compile exact charts by closing active-boundary certificates.

    Invalid API values raise :class:`ValueError`. Unsupported mathematical
    strata return ``passed=False`` with no partial chart coverage.
    """

    lo = _as_fraction(t_min, name="t_min")
    hi = _as_fraction(t_max, name="t_max")
    near_q = _as_fraction(near, name="near")
    far_q = _as_fraction(far, name="far")
    width = _as_fraction(max_root_interval_width, name="max_root_interval_width")
    if hi <= lo:
        raise ValueError("active kinetic owner charts require t_min < t_max")
    if far_q <= near_q:
        raise ValueError("active kinetic owner charts require near < far")
    if width <= 0:
        raise ValueError("max_root_interval_width must be positive")
    if min(max_bisection_depth, max_root_refinements, max_certificate_rounds) < 1:
        raise ValueError("compiler root and closure budgets must be positive")
    ray = torch.as_tensor(ray_coefficients, dtype=torch.float64).detach().cpu().clone()
    if ray.shape != (12,) or not bool(torch.isfinite(ray).all().item()):
        raise ValueError("ray_coefficients must be a finite vector with 12 entries")

    mutable = _MutableWork()
    cache = _PairDifferenceCache(sites, ray, mutable)
    source_roots: dict[ActiveKineticCertificateSource, tuple[_RawRoot, ...]] = {}
    sources_by_owner_word: dict[
        tuple[int, ...],
        tuple[ActiveKineticCertificateSource, ...],
    ] = {}
    final_groups: tuple[_RootGroup, ...] = ()
    final_cells: tuple[_CellWitness, ...] = ()
    try:
        _certify_ray_direction(
            ray,
            t_min=lo,
            t_max=hi,
            max_interval_width=width,
            max_bisection_depth=max_bisection_depth,
        )
        for round_index in range(max_certificate_rounds):
            mutable.certificate_round_count = round_index + 1
            final_groups = _order_and_group_roots(
                tuple(raw for roots in source_roots.values() for raw in roots),
                t_min=lo,
                t_max=hi,
                max_interval_width=width,
                max_bisection_depth=max_bisection_depth,
                max_refinements=max_root_refinements,
                work=mutable,
            )
            final_cells = _sample_and_certify_cells(
                sites,
                ray,
                cache,
                groups=final_groups,
                t_min=lo,
                t_max=hi,
                near=near_q,
                far=far_q,
                work=mutable,
            )
            unseen: list[ActiveKineticCertificateSource] = []
            for cell in final_cells:
                owner_word = cell.certificate.owners
                sources = sources_by_owner_word.get(owner_word)
                if sources is None:
                    sources = _active_sources_for_word(
                        sites,
                        cache,
                        cell.result,
                        near=near_q,
                        far=far_q,
                        work=mutable,
                    )
                    sources_by_owner_word[owner_word] = sources
                    mutable.unique_source_word_count += 1
                unseen.extend(source for source in sources if source not in source_roots)
            unique_unseen = tuple(dict.fromkeys(unseen))
            if not unique_unseen:
                break
            for source in unique_unseen:
                roots = _isolate_source(
                    source,
                    t_min=lo,
                    t_max=hi,
                    max_interval_width=width,
                    max_bisection_depth=max_bisection_depth,
                )
                mutable.root_isolation_call_count += int(source.polynomial.degree > 0)
                mutable.isolated_raw_root_count += len(roots)
                source_roots[source] = roots
        else:
            raise _FailClosedError(  # noqa: TRY301 - converted to a result below
                ActiveKineticCompilerDegeneracy(
                    kind="certificate_closure_budget_exhausted",
                    message="active-boundary predicate closure did not stabilize",
                    lower_bound=lo,
                    upper_bound=hi,
                )
            )

        guards = tuple(_guard_from_group(index, group) for index, group in enumerate(final_groups))
        classified, endpoint = _classify_guards(
            guards,
            final_cells,
            t_min=lo,
            t_max=hi,
        )
        # A full-fiber tie is material-ambiguous even when the open cells on
        # both sides have the same owner word.  At the isolated time the
        # fixed-time site-id tie-break can select a different positive-length
        # owner and therefore a different optical transfer.  Check inactive
        # and domain-endpoint guards too; left/right word equality is not a
        # valid reason to merge this stratum.
        for guard in (*classified, *endpoint):
            full_fiber = tuple(source for source in guard.sources if source.kind == "active_owner_full_fiber")
            if full_fiber:
                raise _FailClosedError(  # noqa: TRY301 - converted to a result below
                    ActiveKineticCompilerDegeneracy(
                        kind="full_fiber_tie",
                        message="an active positive-length owner tie needs an explicit material seam policy",
                        site_ids=tuple(sorted({site for source in full_fiber for site in source.site_ids})),
                        lower_bound=guard.lower_bound,
                        upper_bound=guard.upper_bound,
                        polynomial=full_fiber[0].polynomial,
                    )
                )
            if not guard.active_owner_change:
                continue
            physical = tuple(source for source in guard.sources if not source.analytic_guard_only)
            if len(physical) > 1:
                raise _FailClosedError(  # noqa: TRY301 - converted to a result below
                    ActiveKineticCompilerDegeneracy(
                        kind="ambiguous_simultaneous_active_event",
                        message="multiple physical predicates share one owner-changing algebraic seam",
                        site_ids=tuple(sorted({site for source in physical for site in source.site_ids})),
                        lower_bound=guard.lower_bound,
                        upper_bound=guard.upper_bound,
                        polynomial=guard.canonical_polynomial,
                    )
                )
            if not physical:
                raise _FailClosedError(  # noqa: TRY301 - converted to a result below
                    ActiveKineticCompilerDegeneracy(
                        kind="analytic_guard_changed_owner_word",
                        message="an analytic denominator guard changed the owner word without a physical event",
                        site_ids=tuple(sorted({site for source in guard.sources for site in source.site_ids})),
                        lower_bound=guard.lower_bound,
                        upper_bound=guard.upper_bound,
                        polynomial=guard.canonical_polynomial,
                    )
                )

        active = tuple(guard for guard in classified if guard.active_owner_change)
        inactive = tuple(guard for guard in classified if not guard.active_owner_change)
        charts = _merge_cells_into_charts(
            t_min=lo,
            t_max=hi,
            guards=classified,
            cells=final_cells,
        )
        work = _freeze_work(
            mutable,
            sites=sites,
            source_roots=source_roots,
            groups=final_groups,
            cells=final_cells,
            cache=cache,
        )
        return ActiveKineticOwnerChartProgram(
            passed=True,
            t_min=lo,
            t_max=hi,
            near=near_q,
            far=far_q,
            charts=charts,
            active_event_guards=active,
            inactive_event_guards=inactive,
            endpoint_event_guards=endpoint,
            root_complement_witnesses=tuple(cell.certificate for cell in final_cells),
            unresolved_degeneracies=(),
            work=work,
            continuous_time_coverage=True,
            owner_identity_certified=True,
        )
    except _FailClosedError as failure:
        return ActiveKineticOwnerChartProgram(
            passed=False,
            t_min=lo,
            t_max=hi,
            near=near_q,
            far=far_q,
            charts=(),
            active_event_guards=(),
            inactive_event_guards=(),
            endpoint_event_guards=(),
            root_complement_witnesses=(),
            unresolved_degeneracies=(failure.degeneracy,),
            work=_freeze_work(
                mutable,
                sites=sites,
                source_roots=source_roots,
                groups=final_groups,
                cells=final_cells,
                cache=cache,
            ),
        )


def _active_sources_for_word(
    sites: AffineKineticPowerSites,
    cache: _PairDifferenceCache,
    result: SparsePowerRayWord,
    *,
    near: Fraction,
    far: Fraction,
    work: _MutableWork,
) -> tuple[ActiveKineticCertificateSource, ...]:
    owners = _owners(result)
    run_count = len(owners)
    site_count = sites.site_count
    expected_attempts = (2 * run_count + 1) * (site_count - 1)
    attempts = 0
    sources: dict[ActiveKineticCertificateSource, None] = {}

    for kind, owner, depth in (
        ("pair_near", owners[0], near),
        ("pair_far", owners[-1], far),
    ):
        for competitor in range(site_count):
            if competitor == owner:
                continue
            attempts += 1
            pair = tuple(sorted((owner, competitor)))
            slope, intercept = cache.oriented(pair[0], pair[1])
            polynomial = _add_scaled(intercept, slope, depth)
            _register_source(
                sources,
                ActiveKineticCertificateSource(
                    kind=kind,
                    site_ids=pair,
                    polynomial=polynomial,
                    derivation=f"B_ij(t)+{kind.removeprefix('pair_')}*A_ij(t)=0",
                ),
                persistent_kind="persistent_endpoint_owner_tie",
            )

    for left_owner, right_owner in zip(owners, owners[1:], strict=False):
        attempts += 1
        pair = tuple(sorted((left_owner, right_owner)))
        slope, _ = cache.oriented(pair[0], pair[1])
        if slope.identically_zero:
            raise _FailClosedError(
                ActiveKineticCompilerDegeneracy(
                    kind="active_cut_has_persistent_zero_denominator",
                    message="a positive-length active cut has an identically zero denominator",
                    site_ids=pair,
                    polynomial=slope,
                )
            )
        _register_source(
            sources,
            ActiveKineticCertificateSource(
                kind="active_cut_denominator",
                site_ids=pair,
                polynomial=slope,
                derivation="A_ij(t)=0; analytic active-cut guard only",
                analytic_guard_only=True,
            ),
        )
        for competitor in range(site_count):
            if competitor in (left_owner, right_owner):
                continue
            attempts += 1
            triple = tuple(sorted((left_owner, right_owner, competitor)))
            polynomial = _canonical_triple_polynomial(cache, triple)
            _register_source(
                sources,
                ActiveKineticCertificateSource(
                    kind="active_cut_competitor",
                    site_ids=triple,
                    polynomial=polynomial,
                    derivation=("H_{k|ij}=B_ik*A_ij-A_ik*B_ij=0, canonicalized to B_ab*A_bc-B_bc*A_ab"),
                ),
                persistent_kind="persistent_active_cut_concurrence",
            )

    full_pairs: set[tuple[int, int]] = set()
    for owner in owners:
        for competitor in range(site_count):
            if competitor == owner:
                continue
            attempts += 1
            pair = tuple(sorted((owner, competitor)))
            if pair in full_pairs:
                continue
            full_pairs.add(pair)
            slope, intercept = cache.oriented(pair[0], pair[1])
            if slope.identically_zero and intercept.identically_zero:
                raise _FailClosedError(
                    ActiveKineticCompilerDegeneracy(
                        kind="persistent_full_fiber_tie",
                        message="an active owner and competitor are identical on the complete fiber",
                        site_ids=pair,
                    )
                )
            common = rational_polynomial_gcd(slope, intercept)
            if common.degree == 0:
                continue
            _register_source(
                sources,
                ActiveKineticCertificateSource(
                    kind="active_owner_full_fiber",
                    site_ids=pair,
                    polynomial=common,
                    derivation="gcd(A_ij(t),B_ij(t))=0",
                ),
            )

    if attempts != expected_attempts:
        raise ArithmeticError("active certificate attempt count violated its exact O(SR) formula")
    work.candidate_source_attempt_count += attempts
    work.per_witness_candidate_bound_verified &= attempts == expected_attempts
    return tuple(sources)


def _register_source(
    sources: dict[ActiveKineticCertificateSource, None],
    source: ActiveKineticCertificateSource,
    *,
    persistent_kind: str | None = None,
) -> None:
    if source.polynomial.identically_zero:
        if persistent_kind is None:
            return
        raise _FailClosedError(
            ActiveKineticCompilerDegeneracy(
                kind=persistent_kind,
                message=f"active certificate {source.kind} is identically zero",
                site_ids=source.site_ids,
                polynomial=source.polynomial,
            )
        )
    if source.polynomial.degree > 0:
        sources[source] = None


def _canonical_triple_polynomial(
    cache: _PairDifferenceCache,
    triple: tuple[int, int, int],
) -> RationalPolynomial:
    first, middle, last = triple
    first_slope, first_intercept = cache.oriented(first, middle)
    second_slope, second_intercept = cache.oriented(middle, last)
    return _subtract(
        multiply_rational_polynomials(first_intercept, second_slope),
        multiply_rational_polynomials(second_intercept, first_slope),
    )


def _sample_and_certify_cells(
    sites: AffineKineticPowerSites,
    ray: torch.Tensor,
    cache: _PairDifferenceCache,
    *,
    groups: tuple[_RootGroup, ...],
    t_min: Fraction,
    t_max: Fraction,
    near: Fraction,
    far: Fraction,
    work: _MutableWork,
) -> tuple[_CellWitness, ...]:
    gaps: list[tuple[int | None, int | None, Fraction, Fraction]] = []
    cursor = t_min
    left_event: int | None = None
    for index, group in enumerate(groups):
        if group.upper_bound < t_min or group.lower_bound > t_max:
            continue
        if cursor < group.lower_bound:
            gaps.append((left_event, index, cursor, group.lower_bound))
        cursor = max(cursor, group.upper_bound)
        left_event = index
    if cursor < t_max:
        gaps.append((left_event, None, cursor, t_max))
    if not gaps:
        raise _FailClosedError(
            ActiveKineticCompilerDegeneracy(
                kind="no_rational_root_complement_witness",
                message="event isolators leave no rational open-cell witness",
                lower_bound=t_min,
                upper_bound=t_max,
            )
        )
    cells = []
    for left_index, right_index, lower, upper in gaps:
        time = (lower + upper) / 2
        result = discover_kinetic_power_word_at_time(
            sites,
            ray,
            time=time,
            near=near,
            far=far,
        )
        work.witness_word_discovery_count += 1
        certificate = _certify_witness(
            sites,
            cache,
            result,
            time=time,
            near=near,
            far=far,
            work=work,
        )
        cells.append(_CellWitness(left_index, right_index, result, certificate))
    return tuple(cells)


def _certify_witness(
    sites: AffineKineticPowerSites,
    cache: _PairDifferenceCache,
    result: SparsePowerRayWord,
    *,
    time: Fraction,
    near: Fraction,
    far: Fraction,
    work: _MutableWork,
) -> ActiveOwnerWitnessCertificate:
    owners = _owners(result)
    cuts = (near, *result.transition_depths, far)
    if len(cuts) != len(owners) + 1 or any(left >= right for left, right in zip(cuts, cuts[1:], strict=False)):
        raise ArithmeticError("fixed-time word contains a nonpositive run")
    endpoint_checks = 0
    boundary_checks = 0
    interior_checks = 0
    for endpoint_owner, depth in ((owners[0], near), (owners[-1], far)):
        for competitor in range(sites.site_count):
            if competitor == endpoint_owner:
                continue
            value = _evaluate_difference(cache, endpoint_owner, competitor, time=time, depth=depth)
            endpoint_checks += 1
            if value > 0:
                raise ArithmeticError("fixed-time endpoint owner fails an exact all-site check")
    for boundary_index, depth in enumerate(result.transition_depths):
        left_owner = owners[boundary_index]
        right_owner = owners[boundary_index + 1]
        equality = _evaluate_difference(cache, left_owner, right_owner, time=time, depth=depth)
        if equality != 0:
            raise ArithmeticError("active cut is not an exact adjacent-owner equality")
        slope, _ = cache.oriented(left_owner, right_owner)
        if slope.evaluate(time) == 0:
            raise ArithmeticError("fixed-time active cut has a zero denominator")
        for competitor in range(sites.site_count):
            if competitor in (left_owner, right_owner):
                continue
            value = _evaluate_difference(cache, left_owner, competitor, time=time, depth=depth)
            boundary_checks += 1
            if value > 0:
                raise ArithmeticError("active cut is undercut by an exact competitor")
    for run_index, owner in enumerate(owners):
        depth = (cuts[run_index] + cuts[run_index + 1]) / 2
        for competitor in range(sites.site_count):
            if competitor == owner:
                continue
            value = _evaluate_difference(cache, owner, competitor, time=time, depth=depth)
            interior_checks += 1
            if value == 0:
                slope, intercept = cache.oriented(owner, competitor)
                kind = (
                    "full_fiber_tie"
                    if slope.evaluate(time) == 0 and intercept.evaluate(time) == 0
                    else "nonunique_positive_length_owner_witness"
                )
                raise _FailClosedError(
                    ActiveKineticCompilerDegeneracy(
                        kind=kind,
                        message="a rational chart witness has a nonunique positive-length owner",
                        site_ids=tuple(sorted((owner, competitor))),
                        lower_bound=time,
                        upper_bound=time,
                    )
                )
            if value > 0:
                raise ArithmeticError("positive-length owner is not the unique interior minimum")
    all_checks = endpoint_checks + boundary_checks + interior_checks + len(result.transition_depths)
    work.all_site_witness_check_count += all_checks
    work.max_run_count = max(work.max_run_count, len(owners))
    work.sum_site_run_products += sites.site_count * len(owners)
    expected_attempts = (2 * len(owners) + 1) * (sites.site_count - 1)
    return ActiveOwnerWitnessCertificate(
        time=time,
        owners=owners,
        transition_depths=result.transition_depths,
        run_count=len(owners),
        endpoint_owner_checks=endpoint_checks,
        active_boundary_competitor_checks=boundary_checks,
        active_boundary_equalities_checked=len(result.transition_depths),
        interior_owner_checks=interior_checks,
        candidate_attempt_count=expected_attempts,
        candidate_attempt_bound=expected_attempts,
    )


def _isolate_source(
    source: ActiveKineticCertificateSource,
    *,
    t_min: Fraction,
    t_max: Fraction,
    max_interval_width: Fraction,
    max_bisection_depth: int,
) -> tuple[_RawRoot, ...]:
    if source.polynomial.identically_zero or source.polynomial.degree == 0:
        return ()
    return tuple(
        _RawRoot(source, root)
        for root in isolate_rational_polynomial_roots(
            source.polynomial,
            t_min=t_min,
            t_max=t_max,
            max_interval_width=max_interval_width,
            max_bisection_depth=max_bisection_depth,
        ).roots
    )


def _order_and_group_roots(
    roots: tuple[_RawRoot, ...],
    *,
    t_min: Fraction,
    t_max: Fraction,
    max_interval_width: Fraction,
    max_bisection_depth: int,
    max_refinements: int,
    work: _MutableWork,
) -> tuple[_RootGroup, ...]:
    groups = [_RootGroup((root,), root.root.lower_bound, root.root.upper_bound) for root in roots]
    for _ in range(max_refinements + 1):
        groups.sort(key=lambda group: (group.lower_bound, group.upper_bound))
        merged: list[_RootGroup] = []
        refinable_ids: set[int] = set()
        for current in groups:
            if not merged or merged[-1].upper_bound < current.lower_bound:
                merged.append(current)
                continue
            previous = merged[-1]
            if _groups_are_same_root(
                previous,
                current,
                max_interval_width=max_interval_width,
                max_bisection_depth=max_bisection_depth,
            ):
                merged[-1] = _merge_groups(previous, current)
                continue
            refinable_ids.update(
                id(raw) for group in (previous, current) for raw in group.members if not raw.root.exact
            )
            merged.append(current)
        if refinable_ids:
            refined = []
            for group in merged:
                members = tuple(
                    _refine_raw_root(
                        raw,
                        t_min=t_min,
                        t_max=t_max,
                        max_interval_width=min(max_interval_width, raw.root.width / 4),
                        max_bisection_depth=max_bisection_depth,
                    )
                    if id(raw) in refinable_ids
                    else raw
                    for raw in group.members
                )
                refined.append(_group_from_members(members))
            groups = refined
            work.algebraic_root_refinement_count += len(refinable_ids)
            continue
        for previous, current in zip(merged, merged[1:], strict=False):
            if previous.upper_bound >= current.lower_bound:
                raise _FailClosedError(
                    ActiveKineticCompilerDegeneracy(
                        kind="ambiguous_overlapping_algebraic_roots",
                        message="distinct exact roots could not be ordered",
                        lower_bound=previous.lower_bound,
                        upper_bound=current.upper_bound,
                    )
                )
        if any(group.lower_bound < t_min or group.upper_bound > t_max for group in merged):
            raise ArithmeticError("event root left the requested time domain")
        return tuple(merged)
    raise _FailClosedError(
        ActiveKineticCompilerDegeneracy(
            kind="algebraic_root_separation_budget_exhausted",
            message="distinct event roots did not separate within the refinement budget",
            lower_bound=t_min,
            upper_bound=t_max,
        )
    )


def _groups_are_same_root(
    left: _RootGroup,
    right: _RootGroup,
    *,
    max_interval_width: Fraction,
    max_bisection_depth: int,
) -> bool:
    return _raw_roots_are_same(
        left.members[0],
        right.members[0],
        max_interval_width=max_interval_width,
        max_bisection_depth=max_bisection_depth,
    )


def _raw_roots_are_same(
    left: _RawRoot,
    right: _RawRoot,
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
    roots = isolate_rational_polynomial_roots(
        common,
        t_min=lo,
        t_max=hi,
        max_interval_width=min(max_interval_width, (hi - lo) / 4),
        max_bisection_depth=max_bisection_depth,
    ).roots
    return bool(roots)


def _merge_groups(left: _RootGroup, right: _RootGroup) -> _RootGroup:
    members = tuple(dict.fromkeys((*left.members, *right.members)))
    return _group_from_members(members)


def _group_from_members(members: tuple[_RawRoot, ...]) -> _RootGroup:
    lower = max(member.root.lower_bound for member in members)
    upper = min(member.root.upper_bound for member in members)
    if lower > upper:
        raise ArithmeticError("equal algebraic roots have disjoint isolating intervals")
    return _RootGroup(members, lower, upper)


def _refine_raw_root(
    raw: _RawRoot,
    *,
    t_min: Fraction,
    t_max: Fraction,
    max_interval_width: Fraction,
    max_bisection_depth: int,
) -> _RawRoot:
    if raw.root.exact:
        return raw
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
        if _raw_roots_are_same(
            raw,
            candidate,
            max_interval_width=max_interval_width,
            max_bisection_depth=max_bisection_depth,
        )
    )
    if len(matches) != 1:
        raise _FailClosedError(
            ActiveKineticCompilerDegeneracy(
                kind="algebraic_root_identity_lost_during_refinement",
                message="root refinement did not preserve exactly one source root",
                site_ids=raw.source.site_ids,
                lower_bound=raw.root.lower_bound,
                upper_bound=raw.root.upper_bound,
                polynomial=raw.source.polynomial,
            )
        )
    return matches[0]


def _guard_from_group(index: int, group: _RootGroup) -> ActiveKineticEventGuard:
    ordered = tuple(
        sorted(
            group.members,
            key=lambda raw: (raw.source.kind, raw.source.site_ids),
        )
    )
    exact = group.lower_bound == group.upper_bound and all(
        raw.source.polynomial.evaluate(group.lower_bound) == 0 for raw in ordered
    )
    return ActiveKineticEventGuard(
        guard_id=index,
        lower_bound=group.lower_bound,
        upper_bound=group.upper_bound,
        exact=exact,
        sources=tuple(raw.source for raw in ordered),
        source_multiplicities=tuple(raw.root.multiplicity for raw in ordered),
    )


def _classify_guards(
    guards: tuple[ActiveKineticEventGuard, ...],
    cells: tuple[_CellWitness, ...],
    *,
    t_min: Fraction,
    t_max: Fraction,
) -> tuple[tuple[ActiveKineticEventGuard, ...], tuple[ActiveKineticEventGuard, ...]]:
    left_cells = {cell.right_event_index: cell for cell in cells if cell.right_event_index is not None}
    right_cells = {cell.left_event_index: cell for cell in cells if cell.left_event_index is not None}
    classified = []
    endpoints = []
    for index, guard in enumerate(guards):
        left = left_cells.get(index)
        right = right_cells.get(index)
        if left is None or right is None:
            if guard.exact and guard.lower_bound in (t_min, t_max):
                endpoints.append(guard)
            continue
        left_word = left.certificate.owners
        right_word = right.certificate.owners
        classified.append(
            ActiveKineticEventGuard(
                guard_id=guard.guard_id,
                lower_bound=guard.lower_bound,
                upper_bound=guard.upper_bound,
                exact=guard.exact,
                sources=guard.sources,
                source_multiplicities=guard.source_multiplicities,
                left_owner_word=left_word,
                right_owner_word=right_word,
                active_owner_change=left_word != right_word,
            )
        )
    return tuple(classified), tuple(endpoints)


def _merge_cells_into_charts(
    *,
    t_min: Fraction,
    t_max: Fraction,
    guards: tuple[ActiveKineticEventGuard, ...],
    cells: tuple[_CellWitness, ...],
) -> tuple[ActiveCertifiedKineticOwnerChart, ...]:
    guard_by_id = {guard.guard_id: guard for guard in guards}
    charts: list[ActiveCertifiedKineticOwnerChart] = []
    current: list[_CellWitness] = []
    for cell in cells:
        if current and current[-1].certificate.owners != cell.certificate.owners:
            charts.append(_chart_from_cells(len(charts), current, guard_by_id, t_min=t_min, t_max=t_max))
            current = []
        current.append(cell)
    if current:
        charts.append(_chart_from_cells(len(charts), current, guard_by_id, t_min=t_min, t_max=t_max))
    for index, chart in enumerate(charts):
        charts[index] = ActiveCertifiedKineticOwnerChart(
            chart_id=chart.chart_id,
            left_boundary=chart.left_boundary,
            right_boundary=chart.right_boundary,
            representative_word=chart.representative_word,
            owner_word=chart.owner_word,
            witness_certificates=chart.witness_certificates,
            filtered_inactive_guards=chart.filtered_inactive_guards,
            left_closed=True,
            right_closed=index == len(charts) - 1,
        )
    return tuple(charts)


def _chart_from_cells(
    chart_id: int,
    cells: list[_CellWitness],
    guard_by_id: dict[int, ActiveKineticEventGuard],
    *,
    t_min: Fraction,
    t_max: Fraction,
) -> ActiveCertifiedKineticOwnerChart:
    first, last = cells[0], cells[-1]
    left_guard = None if first.left_event_index is None else guard_by_id.get(first.left_event_index)
    right_guard = None if last.right_event_index is None else guard_by_id.get(last.right_event_index)
    inactive_ids = {
        index
        for cell in cells
        for index in (cell.left_event_index, cell.right_event_index)
        if index is not None and index in guard_by_id and not guard_by_id[index].active_owner_change
    }
    return ActiveCertifiedKineticOwnerChart(
        chart_id=chart_id,
        left_boundary=ActiveKineticTimeBoundary(
            kind="event" if left_guard is not None else "domain_start",
            rational_value=None if left_guard is not None else t_min,
            event_guard=left_guard,
        ),
        right_boundary=ActiveKineticTimeBoundary(
            kind="event" if right_guard is not None else "domain_end",
            rational_value=None if right_guard is not None else t_max,
            event_guard=right_guard,
        ),
        representative_word=first.result,
        owner_word=first.certificate.owners,
        witness_certificates=tuple(cell.certificate for cell in cells),
        filtered_inactive_guards=tuple(guard_by_id[index] for index in sorted(inactive_ids)),
        left_closed=True,
        right_closed=False,
    )


def _certify_ray_direction(
    ray: torch.Tensor,
    *,
    t_min: Fraction,
    t_max: Fraction,
    max_interval_width: Fraction,
    max_bisection_depth: int,
) -> None:
    values = tuple(Fraction.from_float(float(value)) for value in ray.tolist())
    components = tuple(RationalPolynomial((values[6 + axis], values[9 + axis])) for axis in range(3))
    nonzero = tuple(component for component in components if not component.identically_zero)
    if not nonzero:
        raise _FailClosedError(
            ActiveKineticCompilerDegeneracy(
                kind="persistent_zero_ray_direction",
                message="the affine ray direction is zero throughout the time domain",
                lower_bound=t_min,
                upper_bound=t_max,
            )
        )
    common = nonzero[0]
    for component in nonzero[1:]:
        common = rational_polynomial_gcd(common, component)
    if common.degree == 0:
        return
    roots = isolate_rational_polynomial_roots(
        common,
        t_min=t_min,
        t_max=t_max,
        max_interval_width=max_interval_width,
        max_bisection_depth=max_bisection_depth,
    ).roots
    if roots:
        root = roots[0]
        raise _FailClosedError(
            ActiveKineticCompilerDegeneracy(
                kind="zero_ray_direction",
                message="all affine ray-direction components vanish at a certified time",
                lower_bound=root.lower_bound,
                upper_bound=root.upper_bound,
                polynomial=common,
            )
        )


def _evaluate_difference(
    cache: _PairDifferenceCache,
    left: int,
    right: int,
    *,
    time: Fraction,
    depth: Fraction,
) -> Fraction:
    slope, intercept = cache.oriented(left, right)
    return slope.evaluate(time) * depth + intercept.evaluate(time)


def _freeze_work(
    mutable: _MutableWork,
    *,
    sites: AffineKineticPowerSites,
    source_roots: dict[ActiveKineticCertificateSource, tuple[_RawRoot, ...]],
    groups: tuple[_RootGroup, ...],
    cells: tuple[_CellWitness, ...],
    cache: _PairDifferenceCache,
) -> ActiveKineticCompilerWork:
    return ActiveKineticCompilerWork(
        site_count=sites.site_count,
        certificate_round_count=mutable.certificate_round_count,
        root_complement_witness_count=len(cells),
        witness_word_discovery_count=mutable.witness_word_discovery_count,
        candidate_source_attempt_count=mutable.candidate_source_attempt_count,
        unique_source_word_count=mutable.unique_source_word_count,
        unique_candidate_source_count=len(source_roots),
        root_isolation_call_count=mutable.root_isolation_call_count,
        isolated_raw_root_count=mutable.isolated_raw_root_count,
        distinct_event_guard_count=len(groups),
        pair_difference_request_count=mutable.pair_difference_request_count,
        unique_pair_difference_count=len(cache.values),
        all_site_witness_check_count=mutable.all_site_witness_check_count,
        algebraic_root_refinement_count=mutable.algebraic_root_refinement_count,
        max_run_count=mutable.max_run_count,
        sum_site_run_products=mutable.sum_site_run_products,
        per_witness_candidate_bound_verified=mutable.per_witness_candidate_bound_verified,
    )


def _owners(result: SparsePowerRayWord) -> tuple[int, ...]:
    return tuple(int(owner) for owner in result.word.owners.tolist())


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


def _negate(polynomial: RationalPolynomial) -> RationalPolynomial:
    return RationalPolynomial(tuple(-coefficient for coefficient in polynomial.coefficients))


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
    "ActiveCertifiedKineticOwnerChart",
    "ActiveKineticCertificateSource",
    "ActiveKineticCompilerDegeneracy",
    "ActiveKineticCompilerWork",
    "ActiveKineticEventGuard",
    "ActiveKineticOwnerChartProgram",
    "ActiveKineticTimeBoundary",
    "ActiveOwnerWitnessCertificate",
    "compile_active_kinetic_owner_charts",
]
