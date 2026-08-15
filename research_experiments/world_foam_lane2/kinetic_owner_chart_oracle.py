"""Independent exhaustive oracle for small kinetic power-ray worlds.

This module is deliberately *not* a continuous owner-chart compiler.  It is a
small-world falsification oracle for one.  It constructs the unfiltered
algebraic candidate set

* ``A_ij(t) = 0`` (pair-cut denominator),
* ``B_ij(t) + near A_ij(t) = 0``,
* ``B_ij(t) + far A_ij(t) = 0``, and
* ``B_ij A_jk - B_jk A_ij = 0`` for every unordered triple,

isolates every real root exactly, and evaluates an independent brute-force
owner word at a rational point in every root-separated component.  It never
uses adjacency or active-owner filtering.  Consequently its candidate set is
intentionally conservative: inactive concurrence roots, roots hidden by a
third site, and denominator-zero cross-product artifacts remain visible.

The production compiler may discard those roots, but it must reproduce the
oracle's owner-word sequence.  This separation is important: using the future
compiler's active filtering inside its oracle would make the check circular.

All arithmetic after binary64 input normalization is exact
``fractions.Fraction`` arithmetic.  The oracle is intended for adversarial
small worlds, not production-scale compilation.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from fractions import Fraction

import torch
from kinetic_power_word_compiler import (
    AffineKineticPowerSites,
    kinetic_pair_event_predicates,
    kinetic_pair_ray_power_difference,
)
from power_topology_event_predicates import CertifiedEventRoot, RationalPolynomial
from rational_polynomial_roots import (
    multiply_rational_polynomials,
)


class KineticOwnerOracleDegeneracyError(ValueError):
    """The requested interval contains geometry the oracle cannot order."""


class PersistentKineticOwnerTieError(KineticOwnerOracleDegeneracyError):
    """Two sites have identical ray-restricted power for the whole interval."""


@dataclass(frozen=True)
class OracleOwnerWord:
    """Exact positive-length owner runs at one rational time."""

    owners: tuple[int, ...]
    transition_depths: tuple[Fraction, ...]
    method: str = "independent_exact_all_pair_cut_sweep_v1"


@dataclass(frozen=True)
class RawKineticEventPredicate:
    """One unfiltered event predicate and all its distinct real roots."""

    kind: str
    site_ids: tuple[int, ...]
    polynomial: RationalPolynomial
    roots: tuple[CertifiedEventRoot, ...]
    persistent: bool


@dataclass(frozen=True)
class OracleEventSource:
    """A raw predicate that vanishes at one globally isolated event."""

    predicate_index: int
    kind: str
    site_ids: tuple[int, ...]
    multiplicity: int


@dataclass(frozen=True)
class OracleIntervalSample:
    """One exact rational sample in a component of the candidate complement."""

    interval_index: int
    left_event_index: int | None
    right_event_index: int | None
    guaranteed_lower_bound: Fraction
    guaranteed_upper_bound: Fraction
    sample_time: Fraction
    word: OracleOwnerWord


@dataclass(frozen=True)
class OracleKineticEvent:
    """One distinct algebraic candidate root with independently observed sides."""

    event_index: int
    root: CertifiedEventRoot
    sources: tuple[OracleEventSource, ...]
    left_sample_index: int | None
    right_sample_index: int | None
    left_word: OracleOwnerWord | None
    right_word: OracleOwnerWord | None
    changes_owner_word: bool | None
    exact_seam_word: OracleOwnerWord | None
    exact_seam_error: str | None
    source_root_relation: str
    distinct_from_neighbor_roots_certified: bool = True
    exact_seam_evaluated_separately: bool = True

    @property
    def simultaneous(self) -> bool:
        return len(self.sources) > 1

    @property
    def repeated(self) -> bool:
        return any(source.multiplicity > 1 for source in self.sources)


@dataclass(frozen=True)
class KineticOwnerChartOracleReport:
    """Exhaustive raw candidates and exact sample words for one small world."""

    t_min: Fraction
    t_max: Fraction
    near: Fraction
    far: Fraction
    site_count: int
    predicates: tuple[RawKineticEventPredicate, ...]
    events: tuple[OracleKineticEvent, ...]
    interval_samples: tuple[OracleIntervalSample, ...]
    persistent_predicate_indices: tuple[int, ...]
    active_owner_filter_used: bool = False
    requested_frame_sampling_used: bool = False
    all_pair_and_triple_candidates_enumerated: bool = True
    distinct_root_isolation_method: str = "global_product_square_free_sturm_v1"

    @property
    def owner_word_sequence(self) -> tuple[tuple[int, ...], ...]:
        return tuple(sample.word.owners for sample in self.interval_samples)

    @property
    def active_event_indices(self) -> tuple[int, ...]:
        return tuple(event.event_index for event in self.events if event.changes_owner_word is True)


def build_kinetic_owner_chart_oracle(
    sites: AffineKineticPowerSites,
    ray_coefficients: torch.Tensor,
    *,
    t_min: Fraction | float | int,
    t_max: Fraction | float | int,
    near: Fraction | float | int,
    far: Fraction | float | int,
    max_interval_width: Fraction = Fraction(1, 1 << 36),
    max_bisection_depth: int = 256,
) -> KineticOwnerChartOracleReport:
    """Build the exhaustive, deliberately unfiltered small-world oracle.

    A future compiler is complete only if its charts reproduce every returned
    ``owner_word_sequence`` component.  It may use fewer seams than this
    oracle because raw algebraic candidates can be inactive.
    """

    lo = _as_fraction(t_min, name="t_min")
    hi = _as_fraction(t_max, name="t_max")
    near_q = _as_fraction(near, name="near")
    far_q = _as_fraction(far, name="far")
    width = _as_fraction(max_interval_width, name="max_interval_width")
    if hi <= lo:
        raise ValueError("oracle requires t_min < t_max")
    if far_q <= near_q:
        raise ValueError("oracle requires near < far")
    if width <= 0:
        raise ValueError("max_interval_width must be positive")
    ray = _validate_ray(ray_coefficients)
    _reject_direction_zero_in_interval(ray, t_min=lo, t_max=hi)

    raw: list[tuple[str, tuple[int, ...], RationalPolynomial]] = []
    pair_differences = {}
    for left, right in itertools.combinations(range(sites.site_count), 2):
        difference = kinetic_pair_ray_power_difference(sites, ray, left, right)
        if difference.depth_slope.identically_zero and difference.depth_intercept.identically_zero:
            raise PersistentKineticOwnerTieError(f"sites {left} and {right} have a persistent full-fiber power tie")
        pair_differences[(left, right)] = difference
        pair = kinetic_pair_event_predicates(
            difference,
            near=near_q,
            far=far_q,
        )
        raw.extend(
            (
                ("pair_denominator", (left, right), pair.denominator),
                ("pair_near", (left, right), pair.near_crossing),
                ("pair_far", (left, right), pair.far_crossing),
            )
        )

    for first, middle, last in itertools.combinations(range(sites.site_count), 3):
        first_difference = pair_differences[(first, middle)]
        second_difference = pair_differences[(middle, last)]
        concurrence = _subtract_polynomials(
            multiply_rational_polynomials(
                first_difference.depth_intercept,
                second_difference.depth_slope,
            ),
            multiply_rational_polynomials(
                second_difference.depth_intercept,
                first_difference.depth_slope,
            ),
        )
        raw.append(("triple_concurrence", (first, middle, last), concurrence))

    predicates = tuple(
        _isolate_raw_predicate(
            kind,
            site_ids,
            polynomial,
            t_min=lo,
            t_max=hi,
            max_interval_width=width,
            max_bisection_depth=max_bisection_depth,
        )
        for kind, site_ids, polynomial in raw
    )
    global_roots = _isolate_global_candidate_roots(
        predicates,
        t_min=lo,
        t_max=hi,
        max_interval_width=width,
        max_bisection_depth=max_bisection_depth,
    )
    samples = _sample_root_complement(
        sites,
        ray,
        roots=global_roots,
        t_min=lo,
        t_max=hi,
        near=near_q,
        far=far_q,
    )
    events = _describe_events(
        sites,
        ray,
        predicates=predicates,
        roots=global_roots,
        samples=samples,
        t_min=lo,
        t_max=hi,
        near=near_q,
        far=far_q,
        max_interval_width=width,
        max_bisection_depth=max_bisection_depth,
    )
    return KineticOwnerChartOracleReport(
        t_min=lo,
        t_max=hi,
        near=near_q,
        far=far_q,
        site_count=sites.site_count,
        predicates=predicates,
        events=events,
        interval_samples=samples,
        persistent_predicate_indices=tuple(index for index, predicate in enumerate(predicates) if predicate.persistent),
    )


def brute_force_owner_word_at_rational_time(
    sites: AffineKineticPowerSites,
    ray_coefficients: torch.Tensor,
    *,
    time: Fraction | float | int,
    near: Fraction | float | int,
    far: Fraction | float | int,
) -> OracleOwnerWord:
    """Discover a word by sweeping every exact pair cut, not by hull code."""

    ray = _validate_ray(ray_coefficients)
    time_q = _as_fraction(time, name="time")
    near_q = _as_fraction(near, name="near")
    far_q = _as_fraction(far, name="far")
    if far_q <= near_q:
        raise ValueError("word discovery requires near < far")
    positions0 = _fraction_rows(sites.positions0)
    velocities = _fraction_rows(sites.velocities)
    weights = _fraction_rows(sites.weight_coefficients)
    ray_q = tuple(Fraction.from_float(float(value)) for value in ray.tolist())
    origin = tuple(ray_q[axis] + time_q * ray_q[3 + axis] for axis in range(3))
    direction = tuple(ray_q[6 + axis] + time_q * ray_q[9 + axis] for axis in range(3))
    if _dot(direction, direction) == 0:
        raise KineticOwnerOracleDegeneracyError(f"ray direction is zero at time {time_q}")

    lines: list[tuple[Fraction, Fraction]] = []
    for site_id in range(sites.site_count):
        position = tuple(positions0[site_id][axis] + time_q * velocities[site_id][axis] for axis in range(3))
        delta = tuple(origin[axis] - position[axis] for axis in range(3))
        weight = _evaluate_polynomial_coefficients(weights[site_id], time_q)
        lines.append((2 * _dot(direction, delta), _dot(delta, delta) - weight))

    depth_seams = {near_q, far_q}
    for left, right in itertools.combinations(range(sites.site_count), 2):
        slope = lines[left][0] - lines[right][0]
        intercept = lines[left][1] - lines[right][1]
        if slope == 0:
            if intercept == 0:
                raise KineticOwnerOracleDegeneracyError(
                    f"sites {left} and {right} tie on the full fiber at time {time_q}"
                )
            continue
        cut = -intercept / slope
        if near_q < cut < far_q:
            depth_seams.add(cut)

    ordered_depths = sorted(depth_seams)
    owners: list[int] = []
    transitions: list[Fraction] = []
    for left_depth, right_depth in zip(ordered_depths, ordered_depths[1:], strict=False):
        depth = (left_depth + right_depth) / 2
        costs = tuple(slope * depth + intercept for slope, intercept in lines)
        minimum = min(costs)
        tied = tuple(site_id for site_id, cost in enumerate(costs) if cost == minimum)
        if len(tied) != 1:
            raise KineticOwnerOracleDegeneracyError(
                f"owner is tied between sites {tied} at time {time_q}, depth {depth}"
            )
        owner = tied[0]
        if not owners or owner != owners[-1]:
            if owners:
                transitions.append(left_depth)
            owners.append(owner)
    if not owners:
        raise ArithmeticError("positive near/far interval produced no owner run")
    return OracleOwnerWord(tuple(owners), tuple(transitions))


def _isolate_raw_predicate(
    kind: str,
    site_ids: tuple[int, ...],
    polynomial: RationalPolynomial,
    *,
    t_min: Fraction,
    t_max: Fraction,
    max_interval_width: Fraction,
    max_bisection_depth: int,
) -> RawKineticEventPredicate:
    persistent = polynomial.identically_zero
    roots = (
        ()
        if persistent
        else _isolate_exact_roots(
            polynomial,
            t_min=t_min,
            t_max=t_max,
            max_interval_width=max_interval_width,
            max_bisection_depth=max_bisection_depth,
        )
    )
    return RawKineticEventPredicate(kind, site_ids, polynomial, roots, persistent)


def _isolate_global_candidate_roots(
    predicates: tuple[RawKineticEventPredicate, ...],
    *,
    t_min: Fraction,
    t_max: Fraction,
    max_interval_width: Fraction,
    max_bisection_depth: int,
) -> tuple[CertifiedEventRoot, ...]:
    product = RationalPolynomial((Fraction(1),))
    nonconstant_count = 0
    for predicate in predicates:
        if predicate.persistent or predicate.polynomial.degree == 0:
            continue
        product = multiply_rational_polynomials(product, predicate.polynomial)
        nonconstant_count += 1
    if nonconstant_count == 0:
        return ()
    return _isolate_exact_roots(
        product,
        t_min=t_min,
        t_max=t_max,
        max_interval_width=max_interval_width,
        max_bisection_depth=max_bisection_depth,
    )


def _sample_root_complement(
    sites: AffineKineticPowerSites,
    ray: torch.Tensor,
    *,
    roots: tuple[CertifiedEventRoot, ...],
    t_min: Fraction,
    t_max: Fraction,
    near: Fraction,
    far: Fraction,
) -> tuple[OracleIntervalSample, ...]:
    gaps: list[tuple[int | None, int | None, Fraction, Fraction]] = []
    cursor = t_min
    left_event: int | None = None
    for event_index, root in enumerate(roots):
        if cursor < root.lower_bound:
            gaps.append((left_event, event_index, cursor, root.lower_bound))
        cursor = root.upper_bound
        left_event = event_index
    if cursor < t_max:
        gaps.append((left_event, None, cursor, t_max))
    samples = []
    for interval_index, (left_index, right_index, lower, upper) in enumerate(gaps):
        sample_time = (lower + upper) / 2
        samples.append(
            OracleIntervalSample(
                interval_index=interval_index,
                left_event_index=left_index,
                right_event_index=right_index,
                guaranteed_lower_bound=lower,
                guaranteed_upper_bound=upper,
                sample_time=sample_time,
                word=brute_force_owner_word_at_rational_time(
                    sites,
                    ray,
                    time=sample_time,
                    near=near,
                    far=far,
                ),
            )
        )
    return tuple(samples)


def _describe_events(
    sites: AffineKineticPowerSites,
    ray: torch.Tensor,
    *,
    predicates: tuple[RawKineticEventPredicate, ...],
    roots: tuple[CertifiedEventRoot, ...],
    samples: tuple[OracleIntervalSample, ...],
    t_min: Fraction,
    t_max: Fraction,
    near: Fraction,
    far: Fraction,
    max_interval_width: Fraction,
    max_bisection_depth: int,
) -> tuple[OracleKineticEvent, ...]:
    sample_by_left = {sample.left_event_index: sample for sample in samples if sample.left_event_index is not None}
    sample_by_right = {sample.right_event_index: sample for sample in samples if sample.right_event_index is not None}
    events = []
    for event_index, root in enumerate(roots):
        sources = []
        for predicate_index, predicate in enumerate(predicates):
            multiplicity = _predicate_root_multiplicity_in_global_interval(
                predicate,
                root,
                t_min=t_min,
                t_max=t_max,
                max_interval_width=max_interval_width,
                max_bisection_depth=max_bisection_depth,
            )
            if multiplicity is not None:
                sources.append(
                    OracleEventSource(
                        predicate_index=predicate_index,
                        kind=predicate.kind,
                        site_ids=predicate.site_ids,
                        multiplicity=multiplicity,
                    )
                )
        if not sources:
            raise ArithmeticError("global candidate root has no source predicate")
        if sum(source.multiplicity for source in sources) != root.multiplicity:
            raise ArithmeticError("global candidate multiplicity disagrees with source predicates")
        source_root_relation = _certify_source_root_relation(
            predicates,
            tuple(sources),
            root,
        )
        left_sample = sample_by_right.get(event_index)
        right_sample = sample_by_left.get(event_index)
        left_word = None if left_sample is None else left_sample.word
        right_word = None if right_sample is None else right_sample.word
        change = None if left_word is None or right_word is None else left_word.owners != right_word.owners
        seam_word = None
        seam_error = None
        if root.exact:
            try:
                seam_word = brute_force_owner_word_at_rational_time(
                    sites,
                    ray,
                    time=root.lower_bound,
                    near=near,
                    far=far,
                )
            except KineticOwnerOracleDegeneracyError as error:
                seam_error = str(error)
        events.append(
            OracleKineticEvent(
                event_index=event_index,
                root=root,
                sources=tuple(sources),
                left_sample_index=None if left_sample is None else left_sample.interval_index,
                right_sample_index=None if right_sample is None else right_sample.interval_index,
                left_word=left_word,
                right_word=right_word,
                changes_owner_word=change,
                exact_seam_word=seam_word,
                exact_seam_error=seam_error,
                source_root_relation=source_root_relation,
            )
        )
    return tuple(events)


def _predicate_root_multiplicity_in_global_interval(
    predicate: RawKineticEventPredicate,
    root: CertifiedEventRoot,
    *,
    t_min: Fraction,
    t_max: Fraction,
    max_interval_width: Fraction,
    max_bisection_depth: int,
) -> int | None:
    del t_min, t_max  # retained in the signature to make the closed-domain contract explicit
    if predicate.persistent or predicate.polynomial.degree == 0:
        return None
    if root.exact:
        if predicate.polynomial.evaluate(root.lower_bound) != 0:
            return None
        return _rational_root_multiplicity(predicate.polynomial, root.lower_bound)
    local_width = min(max_interval_width, root.width / 4)
    local = _isolate_exact_roots(
        predicate.polynomial,
        t_min=root.lower_bound,
        t_max=root.upper_bound,
        max_interval_width=local_width,
        max_bisection_depth=max_bisection_depth,
    )
    if not local:
        return None
    if len(local) != 1:
        raise ArithmeticError("one global isolating interval contains multiple source roots")
    return local[0].multiplicity


def _rational_root_multiplicity(
    polynomial: RationalPolynomial,
    root: Fraction,
) -> int:
    coefficients = list(polynomial.coefficients)
    multiplicity = 0
    while len(coefficients) > 1:
        quotient = [Fraction(0)] * (len(coefficients) - 1)
        quotient[-1] = coefficients[-1]
        for index in range(len(coefficients) - 2, 0, -1):
            quotient[index - 1] = coefficients[index] + root * quotient[index]
        remainder = coefficients[0] + root * quotient[0]
        if remainder != 0:
            break
        multiplicity += 1
        coefficients = quotient
    if multiplicity < 1:
        raise ArithmeticError("requested rational value is not a polynomial root")
    return multiplicity


def _certify_source_root_relation(
    predicates: tuple[RawKineticEventPredicate, ...],
    sources: tuple[OracleEventSource, ...],
    root: CertifiedEventRoot,
) -> str:
    """Distinguish one algebraic event from accidentally overlapping boxes."""

    if len(sources) == 1:
        return "single_source"
    for left, right in itertools.combinations(sources, 2):
        gcd = _polynomial_gcd(
            predicates[left.predicate_index].polynomial,
            predicates[right.predicate_index].polynomial,
        )
        if gcd.degree == 0:
            raise ArithmeticError("two sources were merged without a shared algebraic root")
        if root.exact:
            contains = gcd.evaluate(root.lower_bound) == 0
        else:
            contains = (
                _closed_root_count(
                    gcd,
                    lo=root.lower_bound,
                    hi=root.upper_bound,
                )
                == 1
            )
        if not contains:
            raise ArithmeticError("simultaneous source predicates are not GCD-equivalent at the event")
    return "same_algebraic_root_gcd_certified"


def _reject_direction_zero_in_interval(
    ray: torch.Tensor,
    *,
    t_min: Fraction,
    t_max: Fraction,
) -> None:
    values = tuple(Fraction.from_float(float(value)) for value in ray.tolist())
    direction0 = values[6:9]
    direction1 = values[9:12]
    moving_axes = [axis for axis, value in enumerate(direction1) if value != 0]
    if not moving_axes:
        if all(value == 0 for value in direction0):
            raise KineticOwnerOracleDegeneracyError("ray direction is identically zero on the requested interval")
        return
    candidate = -direction0[moving_axes[0]] / direction1[moving_axes[0]]
    if all(direction0[axis] + candidate * direction1[axis] == 0 for axis in range(3)) and t_min <= candidate <= t_max:
        raise KineticOwnerOracleDegeneracyError(
            f"ray direction becomes zero at time {candidate}; pair/triple events are incomplete"
        )


def _isolate_exact_roots(
    polynomial: RationalPolynomial,
    *,
    t_min: Fraction,
    t_max: Fraction,
    max_interval_width: Fraction,
    max_bisection_depth: int,
) -> tuple[CertifiedEventRoot, ...]:
    """Independent exact Sturm isolator used only by the oracle.

    The production root helper deliberately remains outside this oracle's
    trusted core.  In particular, every Sturm remainder here is normalized by
    a *positive* scalar.  Arbitrarily making each remainder monic can flip its
    sign and invalidate Sturm variation counts for positive-definite factors.
    """

    if polynomial.identically_zero:
        raise ValueError("an identically zero polynomial has no finite isolated root set")
    if polynomial.degree == 0:
        return ()
    factors = _square_free_decomposition(polynomial)
    repeated = _polynomial_gcd(polynomial, _polynomial_derivative(polynomial))
    square_free = _positive_normalize(_divide_exact(polynomial, repeated))
    roots = _isolate_square_free_roots(
        square_free,
        lo=t_min,
        hi=t_max,
        max_interval_width=max_interval_width,
        max_bisection_depth=max_bisection_depth,
    )
    result = []
    for root in sorted(roots, key=lambda item: (item.lower_bound, item.upper_bound)):
        memberships = []
        for factor, multiplicity in factors:
            if root.exact:
                contains = factor.evaluate(root.lower_bound) == 0
            else:
                contains = (
                    _closed_root_count(
                        factor,
                        lo=root.lower_bound,
                        hi=root.upper_bound,
                    )
                    == 1
                )
            if contains:
                memberships.append(multiplicity)
        if len(memberships) != 1:
            raise ArithmeticError("oracle root does not belong to exactly one square-free factor")
        result.append(
            CertifiedEventRoot(
                lower_bound=root.lower_bound,
                upper_bound=root.upper_bound,
                exact=root.exact,
                multiplicity=memberships[0],
                sturm_root_count=1,
                polynomial_sign_at_lower=root.polynomial_sign_at_lower,
                polynomial_sign_at_upper=root.polynomial_sign_at_upper,
            )
        )
    if len(result) != _closed_root_count(square_free, lo=t_min, hi=t_max):
        raise ArithmeticError("oracle root isolation is incomplete")
    if any(left.upper_bound >= right.lower_bound for left, right in zip(result, result[1:], strict=False)):
        raise ArithmeticError("oracle root isolating intervals overlap")
    return tuple(result)


def _isolate_square_free_roots(
    polynomial: RationalPolynomial,
    *,
    lo: Fraction,
    hi: Fraction,
    max_interval_width: Fraction,
    max_bisection_depth: int,
) -> tuple[CertifiedEventRoot, ...]:
    if polynomial.degree == 0:
        return ()
    for endpoint in (lo, hi):
        if polynomial.evaluate(endpoint) == 0:
            quotient = _divide_exact(
                polynomial,
                RationalPolynomial((-endpoint, Fraction(1))),
            )
            return (
                _exact_root(polynomial, endpoint),
                *_isolate_away_from_removed_exact_root(
                    quotient,
                    lo=lo,
                    hi=hi,
                    removed_root=endpoint,
                    max_interval_width=max_interval_width,
                    max_bisection_depth=max_bisection_depth,
                ),
            )
    sequence = _sturm_sequence(polynomial)
    root_count = _sturm_root_count(sequence, lo, hi)
    queue = [(lo, hi, 0, root_count)]
    isolated = []
    while queue:
        left, right, depth, count = queue.pop()
        if count == 0:
            continue
        if count == 1 and right - left <= max_interval_width:
            isolated.append(
                CertifiedEventRoot(
                    lower_bound=left,
                    upper_bound=right,
                    exact=False,
                    multiplicity=1,
                    sturm_root_count=1,
                    polynomial_sign_at_lower=_sign(polynomial.evaluate(left)),
                    polynomial_sign_at_upper=_sign(polynomial.evaluate(right)),
                )
            )
            continue
        if depth >= max_bisection_depth:
            raise ValueError("oracle event roots exceeded the exact bisection budget")
        midpoint = (left + right) / 2
        if polynomial.evaluate(midpoint) == 0:
            quotient = _divide_exact(
                polynomial,
                RationalPolynomial((-midpoint, Fraction(1))),
            )
            return (
                _exact_root(polynomial, midpoint),
                *_isolate_away_from_removed_exact_root(
                    quotient,
                    lo=lo,
                    hi=hi,
                    removed_root=midpoint,
                    max_interval_width=max_interval_width,
                    max_bisection_depth=max_bisection_depth,
                ),
            )
        left_count = _sturm_root_count(sequence, left, midpoint)
        right_count = _sturm_root_count(sequence, midpoint, right)
        if left_count + right_count != count:
            raise ArithmeticError("oracle Sturm root accounting changed during bisection")
        queue.append((midpoint, right, depth + 1, right_count))
        queue.append((left, midpoint, depth + 1, left_count))
    return tuple(isolated)


def _isolate_away_from_removed_exact_root(
    quotient: RationalPolynomial,
    *,
    lo: Fraction,
    hi: Fraction,
    removed_root: Fraction,
    max_interval_width: Fraction,
    max_bisection_depth: int,
) -> tuple[CertifiedEventRoot, ...]:
    """Keep neighboring isolators strictly away from a removed exact root."""

    if quotient.degree == 0:
        return ()
    roots = []
    if lo < removed_root:
        separator = _root_free_separator(
            quotient,
            fixed=removed_root,
            outer=lo,
            max_bisection_depth=max_bisection_depth,
        )
        roots.extend(
            _isolate_square_free_roots(
                quotient,
                lo=lo,
                hi=separator,
                max_interval_width=max_interval_width,
                max_bisection_depth=max_bisection_depth,
            )
        )
    if removed_root < hi:
        separator = _root_free_separator(
            quotient,
            fixed=removed_root,
            outer=hi,
            max_bisection_depth=max_bisection_depth,
        )
        roots.extend(
            _isolate_square_free_roots(
                quotient,
                lo=separator,
                hi=hi,
                max_interval_width=max_interval_width,
                max_bisection_depth=max_bisection_depth,
            )
        )
    return tuple(roots)


def _root_free_separator(
    polynomial: RationalPolynomial,
    *,
    fixed: Fraction,
    outer: Fraction,
    max_bisection_depth: int,
) -> Fraction:
    """Find a rational point between ``fixed`` and the nearest other root."""

    sequence = _sturm_sequence(polynomial)
    moving = outer
    for _ in range(max_bisection_depth):
        candidate = (fixed + moving) / 2
        if polynomial.evaluate(candidate) == 0:
            moving = candidate
            continue
        left, right = sorted((fixed, candidate))
        if _sturm_root_count(sequence, left, right) == 0:
            return candidate
        moving = candidate
    raise ValueError("could not separate a neighboring root from an exact oracle event")


def _square_free_decomposition(
    polynomial: RationalPolynomial,
) -> tuple[tuple[RationalPolynomial, int], ...]:
    normalized = _monic(polynomial)
    repeated = _polynomial_gcd(normalized, _polynomial_derivative(normalized))
    remaining = _divide_exact(normalized, repeated)
    multiplicity = 1
    factors = []
    while not _is_one(remaining):
        overlap = _polynomial_gcd(remaining, repeated)
        factor = _divide_exact(remaining, overlap)
        if not _is_one(factor):
            factors.append((_monic(factor), multiplicity))
        remaining = overlap
        repeated = _divide_exact(repeated, overlap)
        multiplicity += 1
    if not _is_one(repeated) or not factors:
        raise ArithmeticError("oracle square-free decomposition did not terminate")
    return tuple(factors)


def _closed_root_count(
    polynomial: RationalPolynomial,
    *,
    lo: Fraction,
    hi: Fraction,
) -> int:
    endpoint_count = int(polynomial.evaluate(lo) == 0) + int(polynomial.evaluate(hi) == 0)
    interior = polynomial
    if interior.evaluate(lo) == 0:
        interior = _divide_exact(interior, RationalPolynomial((-lo, Fraction(1))))
    if interior.evaluate(hi) == 0:
        interior = _divide_exact(interior, RationalPolynomial((-hi, Fraction(1))))
    if interior.degree:
        endpoint_count += _sturm_root_count(_sturm_sequence(interior), lo, hi)
    return endpoint_count


def _sturm_sequence(polynomial: RationalPolynomial) -> tuple[RationalPolynomial, ...]:
    sequence = [
        _positive_normalize(polynomial),
        _positive_normalize(_polynomial_derivative(polynomial)),
    ]
    while not sequence[-1].identically_zero:
        _, remainder = _polynomial_divmod(sequence[-2], sequence[-1])
        if remainder.identically_zero:
            break
        sequence.append(_positive_normalize(_negate(remainder)))
    return tuple(sequence)


def _sturm_root_count(
    sequence: tuple[RationalPolynomial, ...],
    lo: Fraction,
    hi: Fraction,
) -> int:
    if sequence[0].evaluate(lo) == 0 or sequence[0].evaluate(hi) == 0:
        raise ArithmeticError("oracle Sturm interval bounds must not be roots")
    return _sign_variations(poly.evaluate(lo) for poly in sequence) - _sign_variations(
        poly.evaluate(hi) for poly in sequence
    )


def _polynomial_gcd(
    left: RationalPolynomial,
    right: RationalPolynomial,
) -> RationalPolynomial:
    a, b = left, right
    while not b.identically_zero:
        _, remainder = _polynomial_divmod(a, b)
        a, b = b, remainder
    return _monic(a)


def _polynomial_divmod(
    numerator: RationalPolynomial,
    denominator: RationalPolynomial,
) -> tuple[RationalPolynomial, RationalPolynomial]:
    if denominator.identically_zero:
        raise ZeroDivisionError("cannot divide by the zero polynomial")
    if numerator.degree < denominator.degree:
        return RationalPolynomial((Fraction(0),)), numerator
    remainder = list(numerator.coefficients)
    quotient = [Fraction(0)] * (numerator.degree - denominator.degree + 1)
    while len(remainder) - 1 >= denominator.degree and any(remainder):
        shift = len(remainder) - 1 - denominator.degree
        scale = remainder[-1] / denominator.coefficients[-1]
        quotient[shift] = scale
        for index, coefficient in enumerate(denominator.coefficients):
            remainder[index + shift] -= scale * coefficient
        while len(remainder) > 1 and remainder[-1] == 0:
            remainder.pop()
    return RationalPolynomial(tuple(quotient)), RationalPolynomial(tuple(remainder))


def _divide_exact(
    numerator: RationalPolynomial,
    denominator: RationalPolynomial,
) -> RationalPolynomial:
    quotient, remainder = _polynomial_divmod(numerator, denominator)
    if not remainder.identically_zero:
        raise ArithmeticError("oracle polynomial division expected an exact quotient")
    return quotient


def _polynomial_derivative(polynomial: RationalPolynomial) -> RationalPolynomial:
    if polynomial.degree == 0:
        return RationalPolynomial((Fraction(0),))
    return RationalPolynomial(
        tuple(index * coefficient for index, coefficient in enumerate(polynomial.coefficients[1:], start=1))
    )


def _monic(polynomial: RationalPolynomial) -> RationalPolynomial:
    if polynomial.identically_zero:
        return polynomial
    leading = polynomial.coefficients[-1]
    return RationalPolynomial(tuple(coefficient / leading for coefficient in polynomial.coefficients))


def _positive_normalize(polynomial: RationalPolynomial) -> RationalPolynomial:
    if polynomial.identically_zero:
        return polynomial
    scale = abs(polynomial.coefficients[-1])
    return RationalPolynomial(tuple(coefficient / scale for coefficient in polynomial.coefficients))


def _negate(polynomial: RationalPolynomial) -> RationalPolynomial:
    return RationalPolynomial(tuple(-coefficient for coefficient in polynomial.coefficients))


def _is_one(polynomial: RationalPolynomial) -> bool:
    return polynomial.degree == 0 and polynomial.coefficients[0] == 1


def _exact_root(polynomial: RationalPolynomial, root: Fraction) -> CertifiedEventRoot:
    if polynomial.evaluate(root) != 0:
        raise ArithmeticError("purported oracle root does not satisfy its polynomial")
    return CertifiedEventRoot(root, root, True, 1, 1, 0, 0)


def _sign_variations(values) -> int:
    signs = [_sign(value) for value in values]
    nonzero = [value for value in signs if value]
    return sum(left != right for left, right in zip(nonzero, nonzero[1:], strict=False))


def _sign(value: Fraction) -> int:
    return (value > 0) - (value < 0)


def _subtract_polynomials(
    left: RationalPolynomial,
    right: RationalPolynomial,
) -> RationalPolynomial:
    degree = max(left.degree, right.degree)
    return RationalPolynomial(
        tuple(
            (left.coefficients[index] if index <= left.degree else Fraction(0))
            - (right.coefficients[index] if index <= right.degree else Fraction(0))
            for index in range(degree + 1)
        )
    )


def _validate_ray(ray_coefficients: torch.Tensor) -> torch.Tensor:
    ray = torch.as_tensor(ray_coefficients, dtype=torch.float64).detach().cpu().clone()
    if ray.shape != (12,) or not bool(torch.isfinite(ray).all().item()):
        raise ValueError("ray_coefficients must be a finite vector with 12 entries")
    return ray


def _fraction_rows(tensor: torch.Tensor) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(tuple(Fraction.from_float(float(value)) for value in row) for row in tensor.tolist())


def _evaluate_polynomial_coefficients(
    coefficients: tuple[Fraction, ...],
    time: Fraction,
) -> Fraction:
    value = Fraction(0)
    for coefficient in reversed(coefficients):
        value = value * time + coefficient
    return value


def _dot(left: tuple[Fraction, ...], right: tuple[Fraction, ...]) -> Fraction:
    return sum((a * b for a, b in zip(left, right, strict=True)), Fraction(0))


def _as_fraction(value: Fraction | float | int, *, name: str) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if not isinstance(value, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite rational, integer, or float")
    return Fraction.from_float(value)
