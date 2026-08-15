from __future__ import annotations

import inspect
import unittest
from fractions import Fraction

import torch
from kinetic_power_word_compiler import (
    AffineKineticPowerSites,
    affine_kinetic_sites_from_identity_spd4_sites,
    discover_kinetic_power_word_at_time,
    isolate_kinetic_adjacent_cut_concurrence,
    isolate_kinetic_pair_events,
    kinetic_adjacent_cut_concurrence,
    kinetic_pair_event_predicates,
    kinetic_pair_ray_power_difference,
)
from power_topology_event_predicates import UnsupportedTopologyDegeneracyError
from sparse_power_word_compiler import discover_sparse_power_word_at_time

DTYPE = torch.float64


class KineticPowerWordCompilerTest(unittest.TestCase):
    def test_pair_ray_coefficients_are_exact_quadratics(self) -> None:
        # p0=(t,0,0), w0=t^2; p1=(0,1+t,0), w1=2t+3t^2.
        sites = AffineKineticPowerSites(
            positions0=torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=DTYPE),
            velocities=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=DTYPE),
            weight_coefficients=torch.tensor(
                [[0.0, 0.0, 1.0], [0.0, 2.0, 3.0]],
                dtype=DTYPE,
            ),
        )
        # o=(1,t,0), d=(1,1-t,0).
        ray = torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, -1.0, 0.0],
            dtype=DTYPE,
        )

        difference = kinetic_pair_ray_power_difference(sites, ray, 0, 1)

        self.assertEqual(
            difference.depth_slope.coefficients,
            (Fraction(2), Fraction(-2), Fraction(-2)),
        )
        self.assertEqual(
            difference.depth_intercept.coefficients,
            (Fraction(-1), Fraction(0), Fraction(4)),
        )
        self.assertLessEqual(difference.depth_slope.degree, 2)
        self.assertLessEqual(difference.depth_intercept.degree, 2)
        self.assertEqual(
            difference.evaluate(time=Fraction(1, 2), depth=Fraction(3, 2)),
            Fraction(3, 4),
        )

    def test_adjacent_cut_concurrence_reaches_but_does_not_exceed_degree_four(self) -> None:
        sites = AffineKineticPowerSites(
            positions0=torch.tensor(
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, -1.0, 1.0]],
                dtype=DTYPE,
            ),
            velocities=torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 1.0, 1.0]],
                dtype=DTYPE,
            ),
            weight_coefficients=torch.tensor(
                [[0.0, 0.0, 1.0], [0.0, 2.0, 3.0], [1.0, -1.0, 2.0]],
                dtype=DTYPE,
            ),
        )
        ray = torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, -1.0, 1.0],
            dtype=DTYPE,
        )

        concurrence = kinetic_adjacent_cut_concurrence(sites, ray, 0, 1, 2)

        self.assertEqual(concurrence.polynomial.degree, 4)
        self.assertEqual(
            concurrence.polynomial.coefficients,
            (
                Fraction(0),
                Fraction(2),
                Fraction(-8),
                Fraction(10),
                Fraction(-4),
            ),
        )
        self.assertEqual(concurrence.degree_bound, 4)
        self.assertTrue(concurrence.root_isolation_implemented)

        isolated = isolate_kinetic_adjacent_cut_concurrence(
            concurrence,
            # This fixture also has a full-fiber pair tie at t=1/2.  Restrict
            # this positive case to the finite-cut event at t=0; the separate
            # negative test below exercises fail-closed denominator handling.
            t_min=Fraction(-1, 4),
            t_max=Fraction(1, 4),
            max_interval_width=Fraction(1, 1 << 28),
        )
        self.assertEqual(len(isolated.roots), 1)
        self.assertTrue(isolated.denominator_roots_filtered)
        for root in isolated.roots:
            midpoint = (root.lower_bound + root.upper_bound) / 2
            self.assertNotEqual(
                concurrence.first_difference.depth_slope.evaluate(midpoint),
                0,
            )
            self.assertNotEqual(
                concurrence.second_difference.depth_slope.evaluate(midpoint),
                0,
            )

    def test_cross_product_root_at_zero_denominators_fails_closed(self) -> None:
        # Red-team fixture: A01=A12=2t while
        # C=B01*A12-B12*A01=2t(2t^2+7).  C(0)=0 is not a finite-cut
        # concurrence because both cuts are at infinity.
        sites = AffineKineticPowerSites(
            positions0=torch.tensor(
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 3.0, 0.0]],
                dtype=DTYPE,
            ),
            velocities=torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                dtype=DTYPE,
            ),
            weight_coefficients=torch.zeros((3, 1), dtype=DTYPE),
        )
        ray = self._static_ray(
            origin=(0.0, 0.0, 0.0),
            direction=(1.0, 0.0, 0.0),
        )
        concurrence = kinetic_adjacent_cut_concurrence(sites, ray, 0, 1, 2)

        self.assertEqual(concurrence.polynomial.evaluate(0), 0)
        self.assertEqual(concurrence.first_difference.depth_slope.evaluate(0), 0)
        self.assertEqual(concurrence.second_difference.depth_slope.evaluate(0), 0)
        with self.assertRaisesRegex(
            UnsupportedTopologyDegeneracyError,
            "zero cut denominator",
        ):
            isolate_kinetic_adjacent_cut_concurrence(
                concurrence,
                t_min=-1,
                t_max=1,
            )

    def test_pair_denominator_and_near_far_candidates_are_exactly_isolated(self) -> None:
        # The finite cut is z=(1+t)/2. It crosses near=1/2 at t=0 and
        # far=1 at t=1; its denominator zero/full-site tie is at t=-1,
        # outside the requested interval.
        sites = AffineKineticPowerSites(
            positions0=torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                dtype=DTYPE,
            ),
            velocities=torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                dtype=DTYPE,
            ),
            weight_coefficients=torch.zeros((2, 1), dtype=DTYPE),
        )
        ray = self._static_ray(
            origin=(0.0, 0.0, 0.0),
            direction=(1.0, 0.0, 0.0),
        )
        difference = kinetic_pair_ray_power_difference(sites, ray, 0, 1)
        predicates = kinetic_pair_event_predicates(
            difference,
            near=Fraction(1, 2),
            far=1,
        )

        isolated = isolate_kinetic_pair_events(
            predicates,
            t_min=0,
            t_max=3,
        )

        self.assertEqual(isolated.denominator_roots, ())
        self.assertEqual(
            tuple(root.lower_bound for root in isolated.near_crossing_roots),
            (Fraction(0),),
        )
        self.assertEqual(len(isolated.far_crossing_roots), 1)
        far_root = isolated.far_crossing_roots[0]
        self.assertLessEqual(far_root.lower_bound, Fraction(1))
        self.assertGreaterEqual(far_root.upper_bound, Fraction(1))
        self.assertFalse(isolated.active_owner_filter_applied)

    def test_random_fixed_time_words_match_brute_force_power_argmin(self) -> None:
        generator = torch.Generator().manual_seed(29)
        sites = AffineKineticPowerSites(
            positions0=torch.randn((12, 3), generator=generator, dtype=DTYPE),
            velocities=0.2 * torch.randn((12, 3), generator=generator, dtype=DTYPE),
            weight_coefficients=0.15 * torch.randn((12, 3), generator=generator, dtype=DTYPE),
        )
        ray = torch.tensor(
            [0.1, -0.2, 0.3, 0.02, 0.01, -0.03, 0.1, 0.05, 1.0, 0.01, -0.02, 0.03],
            dtype=DTYPE,
        )

        for time in (-0.75, 0.2, 1.1):
            result = discover_kinetic_power_word_at_time(
                sites,
                ray,
                time=time,
                near=-2.0,
                far=2.0,
            )
            self._assert_word_matches_brute_force(
                sites,
                ray,
                time=time,
                near=-2.0,
                far=2.0,
                owners=result.word.owners.tolist(),
                transitions=result.transition_depths,
            )

    def test_rotating_face_words_match_brute_and_exceed_fixed_r4_pair_family(self) -> None:
        # q(t)=p1(t)-p0(t)=(2,2t,0), so the spatial face normal is
        # n(t)=2q(t)=(4,4t,0).  n(0) and n(1) are not parallel.  A fixed pair
        # of R4 sites always has constant sliced spatial normal 2(q_xyz), so no
        # such pair can represent this two-time face family.
        sites = AffineKineticPowerSites(
            positions0=torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=DTYPE),
            velocities=torch.tensor([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]], dtype=DTYPE),
            weight_coefficients=torch.zeros((2, 1), dtype=DTYPE),
        )
        ray_x = self._static_ray(origin=(0.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
        ray_y = self._static_ray(origin=(0.0, 0.0, 0.0), direction=(0.0, 1.0, 0.0))
        normal_x = kinetic_pair_ray_power_difference(sites, ray_x, 0, 1).depth_slope
        normal_y = kinetic_pair_ray_power_difference(sites, ray_y, 0, 1).depth_slope

        self.assertEqual(normal_x.coefficients, (Fraction(4),))
        self.assertEqual(normal_y.coefficients, (Fraction(0), Fraction(4)))
        normal_at_zero = (normal_x.evaluate(0), normal_y.evaluate(0))
        normal_at_one = (normal_x.evaluate(1), normal_y.evaluate(1))
        self.assertNotEqual(
            normal_at_zero[0] * normal_at_one[1],
            normal_at_zero[1] * normal_at_one[0],
        )

        ray = self._static_ray(origin=(0.0, 1.0, 0.0), direction=(1.0, 0.25, 0.0))
        expected_cuts = {
            -1: Fraction(4, 3),
            0: Fraction(0),
            1: Fraction(-4, 5),
        }
        for time, expected_cut in expected_cuts.items():
            result = discover_kinetic_power_word_at_time(
                sites,
                ray,
                time=time,
                near=-2,
                far=2,
            )
            self.assertEqual(result.word.owners.tolist(), [0, 1])
            self.assertEqual(result.transition_depths, (expected_cut,))
            self._assert_word_matches_brute_force(
                sites,
                ray,
                time=float(time),
                near=-2.0,
                far=2.0,
                owners=result.word.owners.tolist(),
                transitions=result.transition_depths,
            )

    def test_parameter_bytes_do_not_scale_with_requested_frame_count(self) -> None:
        sites = AffineKineticPowerSites(
            positions0=torch.zeros((5, 3), dtype=DTYPE),
            velocities=torch.ones((5, 3), dtype=DTYPE),
            weight_coefficients=torch.zeros((5, 3), dtype=DTYPE),
        )

        one_frame = sites.storage_report(requested_frame_count=1)
        million_frames = sites.storage_report(requested_frame_count=1_000_000)

        self.assertEqual(one_frame.parameter_scalar_count, 45)
        self.assertEqual(one_frame.parameter_bytes, 45 * 8)
        self.assertEqual(one_frame.parameter_bytes, million_frames.parameter_bytes)
        self.assertEqual(one_frame.stored_frame_state_bytes, 0)
        self.assertEqual(million_frames.stored_frame_state_bytes, 0)
        self.assertEqual(million_frames.frame_dependent_parameter_bytes, 0)

    def test_identity_spd4_world_maps_to_stationary_kinetic_special_case(self) -> None:
        generator = torch.Generator().manual_seed(43)
        sites4d = torch.randn((9, 5), generator=generator, dtype=DTYPE)
        kinetic = affine_kinetic_sites_from_identity_spd4_sites(sites4d)
        ray = torch.tensor(
            [0.2, -0.1, 0.3, 0.04, -0.02, 0.01, 0.1, 0.2, 1.0, 0.03, -0.01, 0.02],
            dtype=DTYPE,
        )

        self.assertTrue(bool(torch.equal(kinetic.positions0, sites4d[:, :3])))
        self.assertEqual(int(torch.count_nonzero(kinetic.velocities).item()), 0)
        for time in (-0.7, 0.0, 0.9):
            fixed4d = discover_sparse_power_word_at_time(
                sites4d,
                ray,
                time=time,
                near=-1.5,
                far=1.5,
            )
            kinetic3d = discover_kinetic_power_word_at_time(
                kinetic,
                ray,
                time=time,
                near=-1.5,
                far=1.5,
            )
            self.assertTrue(bool(torch.equal(fixed4d.word.owners, kinetic3d.word.owners)))
            self.assertEqual(len(fixed4d.transition_depths), len(kinetic3d.transition_depths))
            torch.testing.assert_close(
                torch.tensor([float(value) for value in fixed4d.transition_depths]),
                torch.tensor([float(value) for value in kinetic3d.transition_depths]),
                rtol=1.0e-12,
                atol=1.0e-12,
            )

    def test_fixed_time_discovery_has_no_frame_count_argument(self) -> None:
        parameters = inspect.signature(discover_kinetic_power_word_at_time).parameters
        self.assertNotIn("frame_count", parameters)
        self.assertNotIn("sample_count", parameters)

    def _assert_word_matches_brute_force(
        self,
        sites: AffineKineticPowerSites,
        ray: torch.Tensor,
        *,
        time: float,
        near: float,
        far: float,
        owners: list[int],
        transitions: tuple[Fraction, ...],
    ) -> None:
        cuts = [near, *[float(value) for value in transitions], far]
        positions = sites.positions0 + time * sites.velocities
        powers = torch.tensor(
            [time**degree for degree in range(int(sites.weight_coefficients.shape[1]))],
            dtype=DTYPE,
        )
        weights = sites.weight_coefficients @ powers
        origin = ray[:3] + time * ray[3:6]
        direction = ray[6:9] + time * ray[9:12]
        for run_id, owner in enumerate(owners):
            for fraction in (0.2, 0.5, 0.8):
                depth = cuts[run_id] + fraction * (cuts[run_id + 1] - cuts[run_id])
                point = origin + depth * direction
                distance = (positions - point).square().sum(dim=1) - weights
                self.assertEqual(owner, int(torch.argmin(distance).item()))

    @staticmethod
    def _static_ray(
        *,
        origin: tuple[float, float, float],
        direction: tuple[float, float, float],
    ) -> torch.Tensor:
        return torch.tensor([*origin, 0.0, 0.0, 0.0, *direction, 0.0, 0.0, 0.0], dtype=DTYPE)


if __name__ == "__main__":
    unittest.main()
