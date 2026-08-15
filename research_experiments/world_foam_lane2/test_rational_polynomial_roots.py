from __future__ import annotations

from fractions import Fraction

import pytest
from power_topology_event_predicates import RationalPolynomial
from rational_polynomial_roots import (
    isolate_rational_polynomial_roots,
    multiply_rational_polynomials,
    rational_polynomial_gcd,
)


def _linear(root: Fraction | int) -> RationalPolynomial:
    value = Fraction(root)
    return RationalPolynomial((-value, Fraction(1)))


def _power(polynomial: RationalPolynomial, exponent: int) -> RationalPolynomial:
    result = RationalPolynomial((Fraction(1),))
    for _ in range(exponent):
        result = multiply_rational_polynomials(result, polynomial)
    return result


def test_quartic_irrational_roots_are_disjoint_and_complete() -> None:
    polynomial = multiply_rational_polynomials(
        RationalPolynomial((Fraction(-2), Fraction(0), Fraction(1))),
        RationalPolynomial((Fraction(-3), Fraction(0), Fraction(1))),
    )

    result = isolate_rational_polynomial_roots(
        polynomial,
        t_min=-2,
        t_max=2,
        max_interval_width=Fraction(1, 1 << 32),
    )

    assert len(result.roots) == 4
    assert all(not root.exact for root in result.roots)
    assert all(root.multiplicity == 1 for root in result.roots)
    assert all(root.width <= Fraction(1, 1 << 32) for root in result.roots)
    assert all(
        left.upper_bound < right.lower_bound for left, right in zip(result.roots, result.roots[1:], strict=False)
    )
    expected = (-(3**0.5), -(2**0.5), 2**0.5, 3**0.5)
    for root, value in zip(result.roots, expected, strict=True):
        assert float(root.lower_bound) < value < float(root.upper_bound)


def test_repeated_irrational_quartic_roots_recover_multiplicity() -> None:
    quadratic = RationalPolynomial((Fraction(-2), Fraction(0), Fraction(1)))

    result = isolate_rational_polynomial_roots(
        multiply_rational_polynomials(quadratic, quadratic),
        t_min=-2,
        t_max=2,
        max_interval_width=Fraction(1, 1 << 28),
    )

    assert len(result.roots) == 2
    assert tuple(root.multiplicity for root in result.roots) == (2, 2)
    assert result.square_free_factor_degrees == ((2, 2),)


def test_exact_endpoint_and_midpoint_roots_keep_true_multiplicity() -> None:
    polynomial = multiply_rational_polynomials(
        _power(_linear(0), 3),
        _linear(2),
    )

    result = isolate_rational_polynomial_roots(
        polynomial,
        t_min=0,
        t_max=2,
    )

    assert tuple((root.lower_bound, root.upper_bound, root.exact, root.multiplicity) for root in result.roots) == (
        (Fraction(0), Fraction(0), True, 3),
        (Fraction(2), Fraction(2), True, 1),
    )


def test_exact_interior_rational_root_is_discovered_by_bisection() -> None:
    polynomial = multiply_rational_polynomials(
        _linear(0),
        RationalPolynomial((Fraction(-2), Fraction(0), Fraction(1))),
    )

    result = isolate_rational_polynomial_roots(
        polynomial,
        t_min=-2,
        t_max=2,
        max_interval_width=Fraction(1, 1 << 24),
    )

    exact = tuple(root for root in result.roots if root.exact)
    assert len(exact) == 1
    assert exact[0].lower_bound == 0
    assert exact[0].multiplicity == 1
    assert len(result.roots) == 3


def test_rational_polynomial_gcd_is_monic_and_exact() -> None:
    shared = _power(_linear(1), 2)
    left = multiply_rational_polynomials(shared, _linear(-3))
    right = multiply_rational_polynomials(shared, _linear(4))

    assert rational_polynomial_gcd(left, right) == shared


def test_constant_and_identically_zero_inputs_have_distinct_semantics() -> None:
    constant = isolate_rational_polynomial_roots(
        RationalPolynomial((Fraction(7),)),
        t_min=-1,
        t_max=1,
    )
    assert constant.roots == ()

    with pytest.raises(ValueError, match="identically zero"):
        isolate_rational_polynomial_roots(
            RationalPolynomial((Fraction(0),)),
            t_min=-1,
            t_max=1,
        )


def test_rootless_quadratic_does_not_create_a_spurious_sturm_root() -> None:
    result = isolate_rational_polynomial_roots(
        RationalPolynomial((Fraction(1), Fraction(0), Fraction(1))),
        t_min=-1,
        t_max=1,
    )

    assert result.roots == ()


def test_exact_bisection_budget_fails_closed() -> None:
    polynomial = multiply_rational_polynomials(
        RationalPolynomial((Fraction(-2), Fraction(0), Fraction(1))),
        RationalPolynomial((Fraction(-3), Fraction(0), Fraction(1))),
    )

    with pytest.raises(ValueError, match="bisection budget"):
        isolate_rational_polynomial_roots(
            polynomial,
            t_min=-2,
            t_max=2,
            max_interval_width=Fraction(1, 1 << 80),
            max_bisection_depth=4,
        )
