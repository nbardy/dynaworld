from __future__ import annotations

import inspect
import unittest

import torch
from compiled_transfer_adjoint import power_boundary_parameters
from continuous_owner_identity_certificate import certify_fixed_word_owner_identity
from sparse_power_word_compiler import (
    compile_certified_sparse_owner_program,
    discover_sparse_power_word_at_time,
    discover_sparse_power_words_at_time,
)

DTYPE = torch.float64


class SparsePowerWordCompilerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sites = torch.tensor(
            [
                [0.0, 0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
            ],
            dtype=DTYPE,
        )
        self.ray = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            dtype=DTYPE,
        )

    def test_three_sites_emit_only_two_adjacent_active_faces(self) -> None:
        result = discover_sparse_power_word_at_time(
            self.sites,
            self.ray,
            time=0.0,
            near=-1.5,
            far=1.5,
        )

        self.assertEqual(result.word.owners.tolist(), [0, 1, 2])
        self.assertEqual(result.boundary_site_pairs.tolist(), [[0, 1], [1, 2]])
        self.assertEqual([float(value) for value in result.transition_depths], [-0.5, 0.5])
        self.assertEqual(result.active_boundary_count, 2)
        self.assertLess(result.active_boundary_count, 3)

    def test_discovered_sparse_word_passes_continuous_all_site_certificate(self) -> None:
        result = discover_sparse_power_word_at_time(
            self.sites,
            self.ray,
            time=0.0,
            near=-1.5,
            far=1.5,
        )
        boundary = power_boundary_parameters(self.sites, result.boundary_site_pairs)
        certificate = certify_fixed_word_owner_identity(
            sites=self.sites,
            boundary=boundary,
            ray_coefficients=self.ray.unsqueeze(0),
            words=(result.word,),
            t_min=-1.0,
            t_max=1.0,
            near=-1.5,
            far=1.5,
            ownership_tolerance=1.0e-12,
        )

        self.assertTrue(certificate.passed)
        self.assertTrue(certificate.owner_identity_certified)

    def test_dominated_equal_slope_site_is_removed_deterministically(self) -> None:
        sites = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, -4.0],
            ],
            dtype=DTYPE,
        )
        result = discover_sparse_power_word_at_time(
            sites,
            self.ray,
            time=0.0,
            near=-1.0,
            far=1.0,
        )

        self.assertEqual(result.word.owners.tolist(), [0])
        self.assertEqual(tuple(result.boundary_site_pairs.shape), (0, 2))

    def test_random_depth_witnesses_match_brute_power_argmin(self) -> None:
        generator = torch.Generator().manual_seed(17)
        sites = torch.cat(
            (
                torch.randn((12, 4), generator=generator, dtype=DTYPE),
                0.2 * torch.randn((12, 1), generator=generator, dtype=DTYPE),
            ),
            dim=1,
        )
        ray = torch.tensor(
            [0.1, -0.2, 0.3, 0.02, 0.01, -0.03, 0.1, 0.05, 1.0, 0.01, -0.02, 0.03],
            dtype=DTYPE,
        )
        time = 0.27
        result = discover_sparse_power_word_at_time(
            sites,
            ray,
            time=time,
            near=-2.0,
            far=2.0,
        )
        cuts = [-2.0, *[float(value) for value in result.transition_depths], 2.0]
        o = ray[:3] + time * ray[3:6]
        d = ray[6:9] + time * ray[9:12]
        for run_id, owner in enumerate(result.word.owners.tolist()):
            for fraction in (0.2, 0.5, 0.8):
                depth = cuts[run_id] + fraction * (cuts[run_id + 1] - cuts[run_id])
                point = o + depth * d
                distance = (
                    (sites[:, :3] - point).square().sum(dim=1)
                    + (sites[:, 3] - time).square()
                    - sites[:, 4]
                )
                self.assertEqual(owner, int(torch.argmin(distance).item()))

    def test_multi_track_program_deduplicates_active_pairs(self) -> None:
        program = discover_sparse_power_words_at_time(
            self.sites,
            torch.stack((self.ray, self.ray)),
            time=0.0,
            near=-1.5,
            far=1.5,
        )

        self.assertEqual(program.track_count, 2)
        self.assertEqual(program.boundary_site_pairs.tolist(), [[0, 1], [1, 2]])
        self.assertEqual([word.owners.tolist() for word in program.words], [[0, 1, 2], [0, 1, 2]])
        self.assertEqual(program.candidate_line_count, 6)
        self.assertEqual(program.active_run_count, 6)

    def test_discovery_contract_has_no_requested_frame_count(self) -> None:
        parameters = inspect.signature(discover_sparse_power_words_at_time).parameters
        self.assertNotIn("frame_count", parameters)
        self.assertNotIn("sample_count", parameters)

    def test_static_three_site_word_compiles_one_continuous_sparse_chart(self) -> None:
        result = compile_certified_sparse_owner_program(
            self.sites,
            self.ray.unsqueeze(0),
            t_min=-1.0,
            t_max=1.0,
            near=-1.5,
            far=1.5,
            ownership_tolerance=1.0e-12,
        )

        self.assertTrue(result.passed)
        self.assertTrue(result.continuous_time_coverage)
        self.assertTrue(result.owner_identity_certified)
        self.assertFalse(result.frame_sampling_used)
        self.assertEqual(result.leaf_count, 1)
        self.assertEqual(result.unresolved_intervals, ())
        self.assertEqual(len(result.charts), 1)
        self.assertEqual(result.charts[0].program.boundary_site_pairs.tolist(), [[0, 1], [1, 2]])
        self.assertEqual(result.active_boundary_rows, 2)

    def test_global_temporal_owner_swap_fails_closed_without_an_event_seam_policy(self) -> None:
        sites = torch.tensor(
            [
                [0.0, 0.0, 0.0, -0.5, 0.0],
                [0.0, 0.0, 0.0, 0.5, 0.0],
            ],
            dtype=DTYPE,
        )
        result = compile_certified_sparse_owner_program(
            sites,
            self.ray.unsqueeze(0),
            t_min=-1.0,
            t_max=1.0,
            near=-1.0,
            far=1.0,
            ownership_tolerance=1.0e-12,
        )

        self.assertFalse(result.passed)
        self.assertFalse(result.continuous_time_coverage)
        self.assertFalse(result.owner_identity_certified)
        self.assertEqual([(chart.t_min, chart.t_max) for chart in result.charts], [(-1.0, 0.0), (0.0, 1.0)])
        self.assertEqual(
            [chart.program.words[0].owners.tolist() for chart in result.charts],
            [[0], [1]],
        )
        self.assertEqual(result.active_boundary_rows, 0)
        self.assertEqual(result.deepest_split, 1)
        self.assertEqual(len(result.unresolved_intervals), 1)
        self.assertEqual(
            (result.unresolved_intervals[0].t_min, result.unresolved_intervals[0].t_max),
            (0.0, 0.0),
        )
        self.assertIn("event seam policy is required", result.unresolved_intervals[0].reason)

    def test_unsplittable_owner_swap_is_an_explicit_unresolved_interval(self) -> None:
        sites = torch.tensor(
            [
                [0.0, 0.0, 0.0, -0.5, 0.0],
                [0.0, 0.0, 0.0, 0.5, 0.0],
            ],
            dtype=DTYPE,
        )
        result = compile_certified_sparse_owner_program(
            sites,
            self.ray.unsqueeze(0),
            t_min=-1.0,
            t_max=1.0,
            near=-1.0,
            far=1.0,
            ownership_tolerance=1.0e-12,
            max_split_depth=0,
        )

        self.assertFalse(result.passed)
        self.assertFalse(result.continuous_time_coverage)
        self.assertFalse(result.owner_identity_certified)
        self.assertEqual(result.charts, ())
        self.assertEqual(len(result.unresolved_intervals), 1)
        self.assertIn("owner identity remains unproved", result.unresolved_intervals[0].reason)

    def test_continuous_compiler_contract_has_no_requested_frame_count(self) -> None:
        parameters = inspect.signature(compile_certified_sparse_owner_program).parameters
        self.assertNotIn("frame_count", parameters)
        self.assertNotIn("sample_count", parameters)

    def test_blind_dyadic_splitting_fails_closed_on_an_irrational_triple_event(self) -> None:
        sites = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0, 0.0, 2.25],
                [0.5, 0.5, 0.0, 0.5, 2.75],
            ],
            dtype=DTYPE,
        )
        ray = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            dtype=DTYPE,
        )
        result = compile_certified_sparse_owner_program(
            sites,
            ray.unsqueeze(0),
            t_min=1.0,
            t_max=2.0,
            near=-3.0,
            far=0.0,
            ownership_tolerance=1.0e-12,
            max_split_depth=8,
        )

        self.assertFalse(result.passed)
        self.assertTrue(result.unresolved_intervals)
        event_time = 2.0**0.5
        self.assertTrue(
            any(
                interval.t_min <= event_time <= interval.t_max
                for interval in result.unresolved_intervals
            )
        )


if __name__ == "__main__":
    unittest.main()
