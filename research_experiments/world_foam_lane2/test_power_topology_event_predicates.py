from __future__ import annotations

import inspect
import math
import unittest
from fractions import Fraction

import torch
from power_topology_event_predicates import (
    RIGHT_CONTINUOUS_SEAM_POLICY,
    ZERO_RUN_DELETION_EQUIVALENCE,
    UnsupportedTopologyDegeneracyError,
    fixed_depth_crossing_predicate,
    isolate_topology_event_roots,
    pairwise_ray_power_difference,
    right_continuous_chart_index,
    triple_concurrence_predicate,
)

DTYPE = torch.float64


class PowerTopologyEventPredicateTest(unittest.TestCase):
    def setUp(self) -> None:
        # d(t)=(t,1,0).  The 0/1 and 1/2 cuts are z=-2/t and z=-t,
        # respectively, so the middle run dies at t=sqrt(2).
        self.sites = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0, 0.0, 2.25],
                [0.5, 0.5, 0.0, 0.5, 2.75],
            ],
            dtype=DTYPE,
        )
        self.ray = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            dtype=DTYPE,
        )

    def test_exact_pairwise_affine_coefficients(self) -> None:
        first = pairwise_ray_power_difference(self.sites, self.ray, 0, 1)
        second = pairwise_ray_power_difference(self.sites, self.ray, 1, 2)

        self.assertEqual(first.depth_slope.coefficients, (Fraction(0), Fraction(1)))
        self.assertEqual(first.depth_intercept.coefficients, (Fraction(2),))
        self.assertEqual(second.depth_slope.coefficients, (Fraction(1),))
        self.assertEqual(second.depth_intercept.coefficients, (Fraction(0), Fraction(1)))

    def test_fixed_depth_crossing_has_exact_rational_root(self) -> None:
        predicate = fixed_depth_crossing_predicate(
            self.sites,
            self.ray,
            0,
            1,
            depth=-1,
            boundary_name="near",
        )
        report = isolate_topology_event_roots(predicate, t_min=1, t_max=3)

        self.assertEqual(predicate.polynomial.coefficients, (Fraction(2), Fraction(-1)))
        self.assertEqual(len(report.roots), 1)
        self.assertTrue(report.roots[0].exact)
        self.assertEqual(report.roots[0].lower_bound, Fraction(2))
        self.assertEqual(report.roots[0].upper_bound, Fraction(2))
        self.assertFalse(report.requested_frame_sampling_used)

    def test_irrational_triple_event_gets_certified_rational_isolator(self) -> None:
        predicate = triple_concurrence_predicate(self.sites, self.ray, 0, 1, 2)
        report = isolate_topology_event_roots(
            predicate,
            t_min=1,
            t_max=2,
            max_interval_width=Fraction(1, 1 << 48),
        )

        self.assertEqual(predicate.polynomial.coefficients, (Fraction(2), Fraction(0), Fraction(-1)))
        self.assertEqual(len(report.roots), 1)
        root = report.roots[0]
        self.assertFalse(root.exact)
        self.assertEqual(root.sturm_root_count, 1)
        self.assertLessEqual(root.width, Fraction(1, 1 << 48))
        self.assertLess(root.lower_bound * root.lower_bound, 2)
        self.assertGreater(root.upper_bound * root.upper_bound, 2)
        self.assertLess(float(root.lower_bound), math.sqrt(2.0))
        self.assertGreater(float(root.upper_bound), math.sqrt(2.0))
        self.assertNotEqual(root.polynomial_sign_at_lower, root.polynomial_sign_at_upper)
        self.assertTrue(
            report.continuous_real_dispatch_requires_polynomial_guard_for_irrational_roots
        )

    def test_both_irrational_roots_are_isolated_when_interval_contains_them(self) -> None:
        predicate = triple_concurrence_predicate(self.sites, self.ray, 0, 1, 2)
        report = isolate_topology_event_roots(
            predicate,
            t_min=-2,
            t_max=2,
            max_interval_width=Fraction(1, 1 << 32),
        )

        self.assertEqual(len(report.roots), 2)
        self.assertLess(float(report.roots[0].upper_bound), 0.0)
        self.assertGreater(float(report.roots[1].lower_bound), 0.0)
        self.assertTrue(all(root.sturm_root_count == 1 for root in report.roots))

    def test_full_fiber_tie_fails_closed(self) -> None:
        sites = torch.tensor(
            [
                [0.0, 0.0, 0.0, -0.5, 0.0],
                [0.0, 0.0, 0.0, 0.5, 0.0],
            ],
            dtype=DTYPE,
        )
        ray = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            dtype=DTYPE,
        )
        predicate = fixed_depth_crossing_predicate(
            sites,
            ray,
            0,
            1,
            depth=0,
            boundary_name="probe",
        )

        with self.assertRaisesRegex(UnsupportedTopologyDegeneracyError, "full-fiber tie"):
            isolate_topology_event_roots(predicate, t_min=-1, t_max=1)

    def test_persistent_cut_concurrence_fails_closed(self) -> None:
        sites = torch.tensor(
            [
                [-1.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=DTYPE,
        )
        ray = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=DTYPE,
        )

        with self.assertRaisesRegex(UnsupportedTopologyDegeneracyError, "persistently coincident"):
            triple_concurrence_predicate(sites, ray, 0, 1, 2)

    def test_right_continuous_policy_assigns_shared_seam_to_later_chart(self) -> None:
        intervals = ((-1, 0), (0, 1))

        self.assertEqual(right_continuous_chart_index(Fraction(-1), intervals), 0)
        self.assertEqual(right_continuous_chart_index(Fraction(0), intervals), 1)
        self.assertEqual(right_continuous_chart_index(Fraction(1), intervals), 1)
        self.assertEqual(RIGHT_CONTINUOUS_SEAM_POLICY.policy_id, "right_continuous_half_open_v1")
        self.assertFalse(RIGHT_CONTINUOUS_SEAM_POLICY.supports_full_fiber_ties)

    def test_zero_run_is_exact_forward_identity_but_not_a_gradient_claim(self) -> None:
        fact = ZERO_RUN_DELETION_EQUIVALENCE

        self.assertEqual(
            fact.segment_transfer_at_zero_length,
            (Fraction(1), (Fraction(0), Fraction(0), Fraction(0))),
        )
        self.assertTrue(fact.insertion_or_deletion_preserves_ordered_product)
        self.assertTrue(fact.forward_value_equivalent)
        self.assertFalse(fact.supports_delta_measure_opacity)
        self.assertFalse(fact.classical_geometry_derivative_at_event_certified)

    def test_event_compiler_contract_has_no_frame_sampling_parameter(self) -> None:
        for function in (
            fixed_depth_crossing_predicate,
            triple_concurrence_predicate,
            isolate_topology_event_roots,
        ):
            parameters = inspect.signature(function).parameters
            self.assertNotIn("frame_count", parameters)
            self.assertNotIn("sample_count", parameters)


if __name__ == "__main__":
    unittest.main()
