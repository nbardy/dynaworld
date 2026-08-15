"""Exact rational-polynomial root isolation for bounded compiler predicates.

The kinetic WorldFoam frontend produces event polynomials of degree at most
four.  Closed-form quartic formulas are a poor compiler interface: they are
numerically awkward and do not directly provide certified chart ownership.
This module instead uses exact arithmetic over :class:`fractions.Fraction`,
square-free decomposition, Sturm root counts, and rational bisection.

The result retains the polynomial plus a disjoint rational isolating interval
for every real root.  A rational root is marked exact when it is encountered
at an interval endpoint or rational bisection point.  A rational root that is
not encountered may conservatively remain represented by a certified
interval; callers must never infer irrationality merely from ``exact=False``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction

from power_topology_event_predicates import CertifiedEventRoot, RationalPolynomial


@dataclass(frozen=True)
class RationalPolynomialRootIsolation:
    """All distinct real roots in one closed rational interval."""

    polynomial: RationalPolynomial
    t_min: Fraction
    t_max: Fraction
    roots: tuple[CertifiedEventRoot, ...]
    square_free_factor_degrees: tuple[tuple[int, int], ...]
    exact_rational_arithmetic: bool = True
    requested_frame_sampling_used: bool = False
    method: str = "square_free_sturm_rational_bisection_v1"


def isolate_rational_polynomial_roots(
    polynomial: RationalPolynomial,
    *,
    t_min: Fraction | int,
    t_max: Fraction | int,
    max_interval_width: Fraction = Fraction(1, 1 << 40),
    max_bisection_depth: int = 192,
) -> RationalPolynomialRootIsolation:
    """Isolate every distinct real root of ``polynomial`` in ``[t_min,t_max]``.

    The implementation is intended for the low-degree exact predicates used
    by the compiler, but the algorithm itself is not specialized to quartics.
    Multiplicity is recovered from an exact square-free decomposition.
    """

    lo = _require_fraction(t_min, name="t_min")
    hi = _require_fraction(t_max, name="t_max")
    width_limit = _require_fraction(max_interval_width, name="max_interval_width")
    if hi <= lo:
        raise ValueError("root isolation requires t_min < t_max")
    if width_limit <= 0:
        raise ValueError("max_interval_width must be positive")
    if isinstance(max_bisection_depth, bool) or not isinstance(max_bisection_depth, int):
        raise TypeError("max_bisection_depth must be an integer")
    if max_bisection_depth < 1:
        raise ValueError("max_bisection_depth must be positive")
    if polynomial.identically_zero:
        raise ValueError("an identically zero polynomial has no isolated root set")
    if polynomial.degree == 0:
        return RationalPolynomialRootIsolation(
            polynomial=polynomial,
            t_min=lo,
            t_max=hi,
            roots=(),
            square_free_factor_degrees=(),
        )

    factors = _square_free_decomposition(polynomial)
    square_free_part = _monic(_divide_exact(polynomial, _polynomial_gcd(polynomial, _derivative(polynomial))))
    distinct_roots = _isolate_square_free_roots(
        square_free_part,
        lo=lo,
        hi=hi,
        max_interval_width=width_limit,
        max_bisection_depth=max_bisection_depth,
    )
    roots = tuple(
        _with_root_multiplicity(root, factors)
        for root in sorted(
            distinct_roots,
            key=lambda item: (item.lower_bound, item.upper_bound),
        )
    )
    _assert_disjoint_complete_isolation(square_free_part, roots, lo=lo, hi=hi)
    return RationalPolynomialRootIsolation(
        polynomial=polynomial,
        t_min=lo,
        t_max=hi,
        roots=roots,
        square_free_factor_degrees=tuple((factor.degree, multiplicity) for factor, multiplicity in factors),
    )


def rational_polynomial_gcd(
    left: RationalPolynomial,
    right: RationalPolynomial,
) -> RationalPolynomial:
    """Return the monic greatest common divisor over the rationals."""

    return _polynomial_gcd(left, right)


def multiply_rational_polynomials(
    left: RationalPolynomial,
    right: RationalPolynomial,
) -> RationalPolynomial:
    """Multiply two exact rational polynomials."""

    coefficients = [Fraction(0)] * (left.degree + right.degree + 1)
    for left_index, left_value in enumerate(left.coefficients):
        for right_index, right_value in enumerate(right.coefficients):
            coefficients[left_index + right_index] += left_value * right_value
    return RationalPolynomial(tuple(coefficients))


def _isolate_square_free_roots(
    polynomial: RationalPolynomial,
    *,
    lo: Fraction,
    hi: Fraction,
    max_interval_width: Fraction,
    max_bisection_depth: int,
) -> tuple[CertifiedEventRoot, ...]:
    """Isolate a square-free polynomial, restarting after exact rational hits."""

    if polynomial.degree == 0:
        return ()
    for endpoint in (lo, hi):
        if polynomial.evaluate(endpoint) == 0:
            quotient = _divide_exact(
                polynomial,
                RationalPolynomial((-endpoint, Fraction(1))),
            )
            others = _isolate_square_free_roots(
                quotient,
                lo=lo,
                hi=hi,
                max_interval_width=max_interval_width,
                max_bisection_depth=max_bisection_depth,
            )
            return (
                _exact_root(polynomial, endpoint),
                *others,
            )

    sturm = _sturm_sequence(polynomial)
    root_count = _sturm_root_count(sturm, lo, hi)
    queue: list[tuple[Fraction, Fraction, int, int]] = [(lo, hi, 0, root_count)]
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
            raise ValueError("event roots could not be separated within the exact bisection budget")
        midpoint = (left + right) / 2
        if polynomial.evaluate(midpoint) == 0:
            quotient = _divide_exact(
                polynomial,
                RationalPolynomial((-midpoint, Fraction(1))),
            )
            others = _isolate_square_free_roots(
                quotient,
                lo=lo,
                hi=hi,
                max_interval_width=max_interval_width,
                max_bisection_depth=max_bisection_depth,
            )
            return (
                _exact_root(polynomial, midpoint),
                *others,
            )
        left_count = _sturm_root_count(sturm, left, midpoint)
        right_count = _sturm_root_count(sturm, midpoint, right)
        if left_count + right_count != count:
            raise ArithmeticError("Sturm root accounting changed during exact bisection")
        queue.append((midpoint, right, depth + 1, right_count))
        queue.append((left, midpoint, depth + 1, left_count))
    return tuple(isolated)


def _with_root_multiplicity(
    root: CertifiedEventRoot,
    factors: tuple[tuple[RationalPolynomial, int], ...],
) -> CertifiedEventRoot:
    matches = []
    for factor, multiplicity in factors:
        if root.exact:
            contains = factor.evaluate(root.lower_bound) == 0
        else:
            contains = (
                _sturm_root_count(
                    _sturm_sequence(factor),
                    root.lower_bound,
                    root.upper_bound,
                )
                == 1
            )
        if contains:
            matches.append(multiplicity)
    if len(matches) != 1:
        raise ArithmeticError("isolated root does not belong to exactly one square-free factor")
    return CertifiedEventRoot(
        lower_bound=root.lower_bound,
        upper_bound=root.upper_bound,
        exact=root.exact,
        multiplicity=matches[0],
        sturm_root_count=1,
        polynomial_sign_at_lower=root.polynomial_sign_at_lower,
        polynomial_sign_at_upper=root.polynomial_sign_at_upper,
    )


def _square_free_decomposition(
    polynomial: RationalPolynomial,
) -> tuple[tuple[RationalPolynomial, int], ...]:
    normalized = _monic(polynomial)
    repeated = _polynomial_gcd(normalized, _derivative(normalized))
    remaining = _divide_exact(normalized, repeated)
    multiplicity = 1
    factors: list[tuple[RationalPolynomial, int]] = []
    while not _is_one(remaining):
        overlap = _polynomial_gcd(remaining, repeated)
        factor = _divide_exact(remaining, overlap)
        if not _is_one(factor):
            factors.append((_monic(factor), multiplicity))
        remaining = overlap
        repeated = _divide_exact(repeated, overlap)
        multiplicity += 1
    if not _is_one(repeated):
        raise ArithmeticError("square-free decomposition did not terminate in characteristic zero")
    if sum(factor.degree for factor, _ in factors) < 1:
        raise ArithmeticError("nonconstant polynomial produced no square-free factors")
    return tuple(factors)


def _assert_disjoint_complete_isolation(
    polynomial: RationalPolynomial,
    roots: tuple[CertifiedEventRoot, ...],
    *,
    lo: Fraction,
    hi: Fraction,
) -> None:
    for previous, current in zip(roots, roots[1:], strict=False):
        if previous.upper_bound >= current.lower_bound:
            raise ArithmeticError("certified root intervals overlap or are unordered")
    expected = _closed_interval_root_count(polynomial, lo=lo, hi=hi)
    if len(roots) != expected or any(root.lower_bound < lo or root.upper_bound > hi for root in roots):
        raise ArithmeticError("certified root intervals do not cover the exact root count")


def _closed_interval_root_count(
    polynomial: RationalPolynomial,
    *,
    lo: Fraction,
    hi: Fraction,
) -> int:
    count = int(polynomial.evaluate(lo) == 0) + int(polynomial.evaluate(hi) == 0)
    interior = polynomial
    if polynomial.evaluate(lo) == 0:
        interior = _divide_exact(interior, RationalPolynomial((-lo, Fraction(1))))
    if interior.evaluate(hi) == 0:
        interior = _divide_exact(interior, RationalPolynomial((-hi, Fraction(1))))
    if interior.degree:
        count += _sturm_root_count(_sturm_sequence(interior), lo, hi)
    return count


def _sturm_sequence(polynomial: RationalPolynomial) -> tuple[RationalPolynomial, ...]:
    if polynomial.degree < 1:
        raise ValueError("a Sturm sequence requires a nonconstant polynomial")
    # A Sturm-chain member may be rescaled by a *positive* constant only.
    # Making every member monic is incorrect when its leading coefficient is
    # negative: that flips the member's sign and can corrupt the variation
    # count.  In particular, the chain for x^2 + 1 used to turn the required
    # negative constant remainder positive and report a spurious root.
    normalized = _positive_normalized(polynomial)
    sequence = [normalized, _positive_normalized(_derivative(normalized))]
    while not sequence[-1].identically_zero:
        _, remainder = _polynomial_divmod(sequence[-2], sequence[-1])
        if remainder.identically_zero:
            break
        sequence.append(_positive_normalized(_negate(remainder)))
    return tuple(sequence)


def _sturm_root_count(
    sequence: Sequence[RationalPolynomial],
    lo: Fraction,
    hi: Fraction,
) -> int:
    if hi <= lo:
        raise ValueError("Sturm root count requires an increasing interval")
    if sequence[0].evaluate(lo) == 0 or sequence[0].evaluate(hi) == 0:
        raise ArithmeticError("Sturm interval bounds must not be roots")
    return _sign_variations(poly.evaluate(lo) for poly in sequence) - _sign_variations(
        poly.evaluate(hi) for poly in sequence
    )


def _polynomial_gcd(
    left: RationalPolynomial,
    right: RationalPolynomial,
) -> RationalPolynomial:
    a = left
    b = right
    while not b.identically_zero:
        _, remainder = _polynomial_divmod(a, b)
        a, b = b, remainder
    return _monic(a)


def _divide_exact(
    numerator: RationalPolynomial,
    denominator: RationalPolynomial,
) -> RationalPolynomial:
    quotient, remainder = _polynomial_divmod(numerator, denominator)
    if not remainder.identically_zero:
        raise ArithmeticError("polynomial division expected an exact quotient")
    return quotient


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
    denominator_lead = denominator.coefficients[-1]
    while len(remainder) - 1 >= denominator.degree and any(remainder):
        shift = len(remainder) - 1 - denominator.degree
        scale = remainder[-1] / denominator_lead
        quotient[shift] = scale
        for index, coefficient in enumerate(denominator.coefficients):
            remainder[index + shift] -= scale * coefficient
        while len(remainder) > 1 and remainder[-1] == 0:
            remainder.pop()
    return RationalPolynomial(tuple(quotient)), RationalPolynomial(tuple(remainder))


def _derivative(polynomial: RationalPolynomial) -> RationalPolynomial:
    if polynomial.degree == 0:
        return RationalPolynomial((Fraction(0),))
    return RationalPolynomial(
        tuple(index * coefficient for index, coefficient in enumerate(polynomial.coefficients[1:], start=1))
    )


def _monic(polynomial: RationalPolynomial) -> RationalPolynomial:
    if polynomial.identically_zero:
        return polynomial
    lead = polynomial.coefficients[-1]
    return RationalPolynomial(tuple(coefficient / lead for coefficient in polynomial.coefficients))


def _positive_normalized(polynomial: RationalPolynomial) -> RationalPolynomial:
    """Normalize magnitude without changing a Sturm member's sign."""
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
        raise ArithmeticError("purported exact root does not satisfy its polynomial")
    return CertifiedEventRoot(
        lower_bound=root,
        upper_bound=root,
        exact=True,
        multiplicity=1,
        sturm_root_count=1,
        polynomial_sign_at_lower=0,
        polynomial_sign_at_upper=0,
    )


def _require_fraction(value: Fraction | int, *, name: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (Fraction, int)):
        raise TypeError(f"{name} must be an exact Fraction or integer")
    return Fraction(value)


def _sign_variations(values: Iterable[Fraction]) -> int:
    signs = [_sign(value) for value in values]
    nonzero = [value for value in signs if value]
    return sum(left != right for left, right in zip(nonzero, nonzero[1:], strict=False))


def _sign(value: Fraction) -> int:
    return (value > 0) - (value < 0)


__all__ = [
    "RationalPolynomialRootIsolation",
    "isolate_rational_polynomial_roots",
    "multiply_rational_polynomials",
    "rational_polynomial_gcd",
]
