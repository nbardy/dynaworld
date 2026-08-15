"""Exact topology-event predicates for affine-ray P0 power-cell words.

This module is a small, CPU-only compiler primitive.  It does not discover a
complete kinetic power diagram and it does not sample requested frame times.
Instead it exposes the exact low-degree predicates needed to put chart seams
at topology events rather than hoping that recursive midpoint bisection lands
on them.

For a pair of 4D power sites ``i`` and ``j`` and an affine ray track

``x(t,z) = o0 + t o1 + z (d0 + t d1)``,

the power-distance difference has the form

``Delta_ij(t,z) = A_ij(t) z + B_ij(t)``,

where both ``A`` and ``B`` are affine in time.  Consequently:

* a face crossing a fixed near/far depth is a linear predicate; and
* concurrence of adjacent cuts ``ij`` and ``jk`` is the quadratic predicate
  ``B_ij A_jk - B_jk A_ij``.

All coefficients are :class:`fractions.Fraction` values constructed from the
exact binary64 input values.  Irrational quadratic roots are represented by
certified rational isolating intervals.  The polynomial remains part of the
result: an isolating interval alone is not an exact real-valued chart seam.
A continuous-real compiler must retain the polynomial guard (or an equivalent
algebraic-root object) when dispatching across an irrational event.

Zero-depth-run birth/death events do not need a new optical-transfer payload.
For finite P0 density, their segment transfer is the affine-transfer identity
``(beta,m)=(1,0)``.  A right-continuous half-open time policy makes the chart
choice deterministic.  Full-fiber ties are different: they can exchange a
positive-length colored region at one time, so this module rejects them until
the caller supplies an explicit material/tie policy.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction

import torch


class UnsupportedTopologyDegeneracyError(ValueError):
    """An event cannot be represented by an ordinary zero-run chart seam."""


@dataclass(frozen=True)
class RationalPolynomial:
    """A non-empty rational polynomial with coefficients in ascending order."""

    coefficients: tuple[Fraction, ...]

    def __post_init__(self) -> None:
        if not self.coefficients:
            raise ValueError("a rational polynomial needs at least one coefficient")
        trimmed = list(self.coefficients)
        while len(trimmed) > 1 and trimmed[-1] == 0:
            trimmed.pop()
        object.__setattr__(self, "coefficients", tuple(trimmed))

    @property
    def degree(self) -> int:
        return len(self.coefficients) - 1

    @property
    def identically_zero(self) -> bool:
        return self.degree == 0 and self.coefficients[0] == 0

    def evaluate(self, value: Fraction | float | int) -> Fraction:
        point = _as_fraction(value, name="polynomial evaluation point")
        result = Fraction(0)
        for coefficient in reversed(self.coefficients):
            result = result * point + coefficient
        return result


@dataclass(frozen=True)
class AffineRayPowerDifference:
    """Exact ``power(left)-power(right) = A(t) z + B(t)`` coefficients."""

    left_site_id: int
    right_site_id: int
    depth_slope: RationalPolynomial
    depth_intercept: RationalPolynomial

    def at_depth(self, depth: Fraction | float | int) -> RationalPolynomial:
        depth_q = _as_fraction(depth, name="depth")
        return RationalPolynomial(
            tuple(
                _coefficient(self.depth_intercept, index)
                + depth_q * _coefficient(self.depth_slope, index)
                for index in range(2)
            )
        )


@dataclass(frozen=True)
class TopologyEventPredicate:
    """One exact event polynomial and the face denominators it assumes."""

    kind: str
    polynomial: RationalPolynomial
    site_ids: tuple[int, ...]
    pair_differences: tuple[AffineRayPowerDifference, ...]
    fixed_depth: Fraction | None
    derivation: str


@dataclass(frozen=True)
class CertifiedEventRoot:
    """One real root, either an exact rational point or a rational isolating interval.

    ``exact=False`` does not by itself prove irrationality; a general Sturm
    isolator may conservatively retain an undiscovered rational root inside a
    certified interval.
    """

    lower_bound: Fraction
    upper_bound: Fraction
    exact: bool
    multiplicity: int
    sturm_root_count: int
    polynomial_sign_at_lower: int
    polynomial_sign_at_upper: int

    @property
    def width(self) -> Fraction:
        return self.upper_bound - self.lower_bound


@dataclass(frozen=True)
class RightContinuousSeamPolicy:
    """Time-domain convention compatible with the existing atlas dispatcher."""

    policy_id: str
    interval_rule: str
    exact_event_rule: str
    persistent_tie_rule: str
    supports_full_fiber_ties: bool


@dataclass(frozen=True)
class ZeroRunDeletionEquivalence:
    """Conditions under which a topology event needs no transfer event atom."""

    segment_transfer_at_zero_length: tuple[Fraction, tuple[Fraction, Fraction, Fraction]]
    insertion_or_deletion_preserves_ordered_product: bool
    requires_finite_density_and_color: bool
    supports_delta_measure_opacity: bool
    forward_value_equivalent: bool
    classical_geometry_derivative_at_event_certified: bool
    explanation: str


@dataclass(frozen=True)
class TopologyEventIsolation:
    """Certified roots of one predicate over one exact binary64-real interval."""

    predicate: TopologyEventPredicate
    t_min: Fraction
    t_max: Fraction
    roots: tuple[CertifiedEventRoot, ...]
    exact_rational_arithmetic: bool = True
    requested_frame_sampling_used: bool = False
    continuous_real_dispatch_requires_polynomial_guard_for_irrational_roots: bool = True
    seam_policy_id: str = "right_continuous_half_open_v1"


RIGHT_CONTINUOUS_SEAM_POLICY = RightContinuousSeamPolicy(
    policy_id="right_continuous_half_open_v1",
    interval_rule="[t_k,t_{k+1}) for every non-final chart; the final upper endpoint is included",
    exact_event_rule=(
        "a shared seam time is owned by the increasing-time chart, matching the existing "
        "half-open atlas dispatch"
    ),
    persistent_tie_rule="unsupported here; a caller must provide an explicit material/tie rule",
    supports_full_fiber_ties=False,
)


ZERO_RUN_DELETION_EQUIVALENCE = ZeroRunDeletionEquivalence(
    segment_transfer_at_zero_length=(
        Fraction(1),
        (Fraction(0), Fraction(0), Fraction(0)),
    ),
    insertion_or_deletion_preserves_ordered_product=True,
    requires_finite_density_and_color=True,
    supports_delta_measure_opacity=False,
    forward_value_equivalent=True,
    classical_geometry_derivative_at_event_certified=False,
    explanation=(
        "For finite P0 density, beta=exp(-sigma*0)=1 and m=(1-beta)c=0, so a "
        "zero-length run is the ordered affine-transfer identity. The forward seam is exact, "
        "but topology-changing geometry derivatives can be one-sided or undefined."
    ),
)


def pairwise_ray_power_difference(
    sites: torch.Tensor,
    ray_coefficients: torch.Tensor,
    left_site_id: int,
    right_site_id: int,
) -> AffineRayPowerDifference:
    """Derive exact affine ``A(t)`` and ``B(t)`` for one power-site pair."""

    sites_f64 = _validate_sites(sites)
    ray_f64 = _validate_ray(ray_coefficients)
    site_count = int(sites_f64.shape[0])
    if not 0 <= left_site_id < site_count or not 0 <= right_site_id < site_count:
        raise ValueError("pair site ids must lie inside the site table")
    if left_site_id == right_site_id:
        raise UnsupportedTopologyDegeneracyError("a power face requires two distinct sites")

    left = tuple(Fraction.from_float(float(value)) for value in sites_f64[left_site_id].tolist())
    right = tuple(Fraction.from_float(float(value)) for value in sites_f64[right_site_id].tolist())
    ray = tuple(Fraction.from_float(float(value)) for value in ray_f64.tolist())
    normal = tuple(2 * (right[axis] - left[axis]) for axis in range(3))
    time_normal = 2 * (right[3] - left[3])
    bias = sum(
        (left[axis] * left[axis] - right[axis] * right[axis] for axis in range(4)),
        Fraction(0),
    )
    bias = bias - left[4] + right[4]
    depth_slope = RationalPolynomial(
        (
            _dot(normal, ray[6:9]),
            _dot(normal, ray[9:12]),
        )
    )
    depth_intercept = RationalPolynomial(
        (
            _dot(normal, ray[:3]) + bias,
            _dot(normal, ray[3:6]) + time_normal,
        )
    )
    return AffineRayPowerDifference(
        left_site_id=left_site_id,
        right_site_id=right_site_id,
        depth_slope=_pad_affine(depth_slope),
        depth_intercept=_pad_affine(depth_intercept),
    )


def fixed_depth_crossing_predicate(
    sites: torch.Tensor,
    ray_coefficients: torch.Tensor,
    left_site_id: int,
    right_site_id: int,
    *,
    depth: Fraction | float | int,
    boundary_name: str,
) -> TopologyEventPredicate:
    """Return the exact linear predicate for a face crossing near or far."""

    if not boundary_name:
        raise ValueError("boundary_name must be non-empty")
    difference = pairwise_ray_power_difference(
        sites,
        ray_coefficients,
        left_site_id,
        right_site_id,
    )
    depth_q = _as_fraction(depth, name="fixed depth")
    return TopologyEventPredicate(
        kind=f"fixed_depth_crossing:{boundary_name}",
        polynomial=difference.at_depth(depth_q),
        site_ids=(left_site_id, right_site_id),
        pair_differences=(difference,),
        fixed_depth=depth_q,
        derivation="Delta_ij(t,z_fixed) = A_ij(t) z_fixed + B_ij(t)",
    )


def triple_concurrence_predicate(
    sites: torch.Tensor,
    ray_coefficients: torch.Tensor,
    first_site_id: int,
    middle_site_id: int,
    last_site_id: int,
) -> TopologyEventPredicate:
    """Return the quadratic predicate for cuts ``ij`` and ``jk`` to coincide."""

    if len({first_site_id, middle_site_id, last_site_id}) != 3:
        raise UnsupportedTopologyDegeneracyError("triple concurrence requires three distinct sites")
    first = pairwise_ray_power_difference(
        sites,
        ray_coefficients,
        first_site_id,
        middle_site_id,
    )
    second = pairwise_ray_power_difference(
        sites,
        ray_coefficients,
        middle_site_id,
        last_site_id,
    )
    if first.depth_slope.identically_zero or second.depth_slope.identically_zero:
        raise UnsupportedTopologyDegeneracyError(
            "triple concurrence requires two generically finite pairwise cuts"
        )
    coefficients = _subtract_polynomials(
        _multiply_polynomials(first.depth_intercept, second.depth_slope),
        _multiply_polynomials(second.depth_intercept, first.depth_slope),
    )
    if coefficients.identically_zero:
        raise UnsupportedTopologyDegeneracyError(
            "adjacent cut depths are persistently coincident; no isolated triple event exists"
        )
    return TopologyEventPredicate(
        kind="triple_concurrence",
        polynomial=coefficients,
        site_ids=(first_site_id, middle_site_id, last_site_id),
        pair_differences=(first, second),
        fixed_depth=None,
        derivation="B_ij(t) A_jk(t) - B_jk(t) A_ij(t)",
    )


def isolate_topology_event_roots(
    predicate: TopologyEventPredicate,
    *,
    t_min: Fraction | float | int,
    t_max: Fraction | float | int,
    max_interval_width: Fraction | float = Fraction(1, 1 << 40),
    max_bisection_depth: int = 160,
) -> TopologyEventIsolation:
    """Isolate every real predicate root in the requested time interval.

    Rational roots are returned exactly.  A simple irrational quadratic root
    is returned in a rational interval with exact Sturm count one.  The caller
    must retain the polynomial as a dispatch guard; rounding an irrational
    root to either interval endpoint would leave a real-time coverage gap.
    """

    lo = _as_fraction(t_min, name="t_min")
    hi = _as_fraction(t_max, name="t_max")
    width_limit = _as_fraction(max_interval_width, name="max_interval_width")
    if hi <= lo:
        raise ValueError("event isolation requires t_min < t_max")
    if width_limit <= 0:
        raise ValueError("max_interval_width must be positive")
    if max_bisection_depth < 1:
        raise ValueError("max_bisection_depth must be positive")
    for difference in predicate.pair_differences:
        _reject_full_fiber_ties(difference, lo=lo, hi=hi)

    polynomial = predicate.polynomial
    if polynomial.identically_zero:
        raise UnsupportedTopologyDegeneracyError("an identically zero event predicate is unsupported")
    if polynomial.degree > 2:
        raise UnsupportedTopologyDegeneracyError("only degree <= 2 event predicates are supported")
    roots = _isolate_polynomial_roots(
        polynomial,
        lo=lo,
        hi=hi,
        max_interval_width=width_limit,
        max_bisection_depth=max_bisection_depth,
    )
    for root in roots:
        if root.exact:
            _reject_zero_cut_denominator_at_root(predicate, root.lower_bound)
    return TopologyEventIsolation(
        predicate=predicate,
        t_min=lo,
        t_max=hi,
        roots=roots,
    )


def right_continuous_chart_index(
    time: Fraction | float | int,
    chart_intervals: Sequence[tuple[Fraction | float | int, Fraction | float | int]],
) -> int:
    """Select one chart under ``[lo,hi)`` semantics, including the final end."""

    if not chart_intervals:
        raise ValueError("right-continuous dispatch requires at least one chart")
    point = _as_fraction(time, name="time")
    intervals = tuple(
        (_as_fraction(lo, name="chart lower bound"), _as_fraction(hi, name="chart upper bound"))
        for lo, hi in chart_intervals
    )
    for index, (lo, hi) in enumerate(intervals):
        if hi <= lo:
            raise ValueError("every chart interval must have positive width")
        if index and intervals[index - 1][1] != lo:
            raise ValueError("chart intervals must be ordered and exactly contiguous")
    for index, (lo, hi) in enumerate(intervals):
        is_last = index == len(intervals) - 1
        if lo <= point and (point < hi or (is_last and point <= hi)):
            return index
    raise ValueError("time lies outside the chart partition")


def _isolate_polynomial_roots(
    polynomial: RationalPolynomial,
    *,
    lo: Fraction,
    hi: Fraction,
    max_interval_width: Fraction,
    max_bisection_depth: int,
) -> tuple[CertifiedEventRoot, ...]:
    if polynomial.degree == 0:
        return ()
    if polynomial.degree == 1:
        root = -polynomial.coefficients[0] / polynomial.coefficients[1]
        return (_exact_root(polynomial, root, multiplicity=1),) if lo <= root <= hi else ()

    c, b, a = polynomial.coefficients
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return ()
    if discriminant == 0:
        root = -b / (2 * a)
        return (_exact_root(polynomial, root, multiplicity=2),) if lo <= root <= hi else ()
    square_root = _rational_square_root(discriminant)
    if square_root is not None:
        values = sorted({(-b - square_root) / (2 * a), (-b + square_root) / (2 * a)})
        return tuple(
            _exact_root(polynomial, root, multiplicity=1)
            for root in values
            if lo <= root <= hi
        )
    return _isolate_irrational_quadratic_roots(
        polynomial,
        lo=lo,
        hi=hi,
        max_interval_width=max_interval_width,
        max_bisection_depth=max_bisection_depth,
    )


def _isolate_irrational_quadratic_roots(
    polynomial: RationalPolynomial,
    *,
    lo: Fraction,
    hi: Fraction,
    max_interval_width: Fraction,
    max_bisection_depth: int,
) -> tuple[CertifiedEventRoot, ...]:
    sturm = _quadratic_sturm_sequence(polynomial)
    initial_count = _sturm_root_count(sturm, lo, hi)
    queue: list[tuple[Fraction, Fraction, int, int]] = [(lo, hi, 0, initial_count)]
    isolated: list[CertifiedEventRoot] = []
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
            raise UnsupportedTopologyDegeneracyError(
                "irrational event root could not be isolated within the bisection budget"
            )
        midpoint = (left + right) / 2
        if polynomial.evaluate(midpoint) == 0:
            raise ArithmeticError("a nonsquare-discriminant quadratic unexpectedly had a rational root")
        left_count = _sturm_root_count(sturm, left, midpoint)
        right_count = _sturm_root_count(sturm, midpoint, right)
        if left_count + right_count != count:
            raise ArithmeticError("Sturm root accounting changed during rational bisection")
        queue.append((midpoint, right, depth + 1, right_count))
        queue.append((left, midpoint, depth + 1, left_count))
    return tuple(sorted(isolated, key=lambda root: (root.lower_bound, root.upper_bound)))


def _quadratic_sturm_sequence(
    polynomial: RationalPolynomial,
) -> tuple[RationalPolynomial, RationalPolynomial, RationalPolynomial]:
    c, b, a = polynomial.coefficients
    discriminant = b * b - 4 * a * c
    return (
        polynomial,
        RationalPolynomial((b, 2 * a)),
        RationalPolynomial((discriminant / (4 * a),)),
    )


def _sturm_root_count(
    sturm: Sequence[RationalPolynomial],
    lo: Fraction,
    hi: Fraction,
) -> int:
    if sturm[0].evaluate(lo) == 0 or sturm[0].evaluate(hi) == 0:
        raise ArithmeticError("irrational-root Sturm bounds must not be polynomial roots")
    return _sign_variations(polynomial.evaluate(lo) for polynomial in sturm) - _sign_variations(
        polynomial.evaluate(hi) for polynomial in sturm
    )


def _sign_variations(values: Iterable[Fraction]) -> int:
    signs = [_sign(value) for value in values]
    signs = [value for value in signs if value]
    return sum(left != right for left, right in zip(signs, signs[1:], strict=False))


def _exact_root(
    polynomial: RationalPolynomial,
    root: Fraction,
    *,
    multiplicity: int,
) -> CertifiedEventRoot:
    if polynomial.evaluate(root) != 0:
        raise ArithmeticError("purported exact rational root does not satisfy its polynomial")
    return CertifiedEventRoot(
        lower_bound=root,
        upper_bound=root,
        exact=True,
        multiplicity=multiplicity,
        sturm_root_count=1,
        polynomial_sign_at_lower=0,
        polynomial_sign_at_upper=0,
    )


def _reject_full_fiber_ties(
    difference: AffineRayPowerDifference,
    *,
    lo: Fraction,
    hi: Fraction,
) -> None:
    slope = difference.depth_slope
    intercept = difference.depth_intercept
    if slope.identically_zero and intercept.identically_zero:
        raise UnsupportedTopologyDegeneracyError(
            f"sites {difference.left_site_id}/{difference.right_site_id} tie along the full fiber for all times"
        )
    candidates: set[Fraction] = set()
    for polynomial in (slope, intercept):
        if polynomial.degree == 1:
            candidates.add(-polynomial.coefficients[0] / polynomial.coefficients[1])
    for time in candidates:
        if lo <= time <= hi and slope.evaluate(time) == 0 and intercept.evaluate(time) == 0:
            raise UnsupportedTopologyDegeneracyError(
                f"sites {difference.left_site_id}/{difference.right_site_id} have a full-fiber tie at "
                f"t={float(time):.17g}"
            )


def _reject_zero_cut_denominator_at_root(
    predicate: TopologyEventPredicate,
    root: Fraction,
) -> None:
    for difference in predicate.pair_differences:
        if difference.depth_slope.evaluate(root) == 0:
            raise UnsupportedTopologyDegeneracyError(
                "event root coincides with a zero pairwise cut denominator"
            )


def _multiply_polynomials(
    left: RationalPolynomial,
    right: RationalPolynomial,
) -> RationalPolynomial:
    coefficients = [Fraction(0)] * (left.degree + right.degree + 1)
    for left_id, left_value in enumerate(left.coefficients):
        for right_id, right_value in enumerate(right.coefficients):
            coefficients[left_id + right_id] += left_value * right_value
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


def _rational_square_root(value: Fraction) -> Fraction | None:
    numerator_root = math.isqrt(value.numerator)
    denominator_root = math.isqrt(value.denominator)
    if numerator_root * numerator_root != value.numerator:
        return None
    if denominator_root * denominator_root != value.denominator:
        return None
    return Fraction(numerator_root, denominator_root)


def _validate_sites(sites: torch.Tensor) -> torch.Tensor:
    sites_f64 = torch.as_tensor(sites, dtype=torch.float64).detach().cpu()
    if sites_f64.ndim != 2 or sites_f64.shape[1] != 5 or int(sites_f64.shape[0]) < 2:
        raise ValueError("sites must have shape [S,5] with S >= 2")
    if not bool(torch.isfinite(sites_f64).all().item()):
        raise ValueError("sites must be finite")
    return sites_f64


def _validate_ray(ray_coefficients: torch.Tensor) -> torch.Tensor:
    ray_f64 = torch.as_tensor(ray_coefficients, dtype=torch.float64).detach().cpu()
    if ray_f64.shape != (12,) or not bool(torch.isfinite(ray_f64).all().item()):
        raise ValueError("ray_coefficients must be a finite vector with 12 entries")
    return ray_f64


def _pad_affine(polynomial: RationalPolynomial) -> RationalPolynomial:
    return RationalPolynomial(
        (
            polynomial.coefficients[0],
            polynomial.coefficients[1] if polynomial.degree else Fraction(0),
        )
    )


def _coefficient(polynomial: RationalPolynomial, index: int) -> Fraction:
    return polynomial.coefficients[index] if index < len(polynomial.coefficients) else Fraction(0)


def _as_fraction(value: Fraction | float | int, *, name: str) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return Fraction.from_float(value)


def _dot(left: Sequence[Fraction], right: Sequence[Fraction]) -> Fraction:
    return sum((a * b for a, b in zip(left, right, strict=True)), Fraction(0))


def _sign(value: Fraction) -> int:
    return (value > 0) - (value < 0)


__all__ = [
    "AffineRayPowerDifference",
    "CertifiedEventRoot",
    "RIGHT_CONTINUOUS_SEAM_POLICY",
    "RationalPolynomial",
    "RightContinuousSeamPolicy",
    "TopologyEventIsolation",
    "TopologyEventPredicate",
    "UnsupportedTopologyDegeneracyError",
    "ZERO_RUN_DELETION_EQUIVALENCE",
    "ZeroRunDeletionEquivalence",
    "fixed_depth_crossing_predicate",
    "isolate_topology_event_roots",
    "pairwise_ray_power_difference",
    "right_continuous_chart_index",
    "triple_concurrence_predicate",
]
