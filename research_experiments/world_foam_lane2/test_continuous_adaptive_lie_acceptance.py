from __future__ import annotations

import unittest

import torch
from compiled_lie_world_adjoint import (
    AdaptiveCompiledLieWorldAtlas,
    AdaptiveLieWorldCompilePolicy,
    compile_lie_world_atlas,
)
from compiled_transfer_adjoint import (
    make_stable_cell_word,
    power_boundary_parameters,
)
from continuous_adaptive_lie_acceptance import (
    ContinuousAdaptiveLieCertificationPolicy,
    certify_prepared_adaptive_lie_world,
    power_boundary_to_site_maximum_column_l1_norm,
)
from staged_compiled_lie_adjoint import refresh_staged_lie_world_snapshot

DTYPE = torch.float64


def _wrap_single_chart(chart) -> AdaptiveCompiledLieWorldAtlas:
    return AdaptiveCompiledLieWorldAtlas(
        charts=(chart,),
        selections=(),
        policy=AdaptiveLieWorldCompilePolicy(),
        supplied_word_ordering_check=chart.supplied_word_ordering_check,
    )


def _moving_power_snapshot(*, node_count: int = 4):
    sites = torch.tensor(
        [
            [0.0, 0.0, 0.2, -0.05, 0.01],
            [0.03, 0.0, 0.6, 0.05, -0.02],
        ],
        dtype=DTYPE,
    )
    boundary_pairs = torch.tensor([[0, 1]], dtype=torch.int64)
    boundary = power_boundary_parameters(sites, boundary_pairs)
    rays = torch.tensor(
        [[0.01, 0.0, 0.0, 0.005, 0.0, 0.005, 0.0, 0.0, 1.0, 0.005, 0.0, 0.02]],
        dtype=DTYPE,
    )
    density = torch.tensor([0.5, 0.7], dtype=DTYPE)
    color = torch.tensor([[0.2, 0.4, 0.8], [0.8, 0.3, 0.2]], dtype=DTYPE)
    words = (make_stable_cell_word([0, 1], [-1, 0], [0, -2]),)
    chart = compile_lie_world_atlas(
        boundary=boundary,
        ray_coefficients=rays,
        words=words,
        site_density=density,
        site_color=color,
        t_min=-0.5,
        t_max=0.5,
        near=0.1,
        far=0.9,
        node_count=node_count,
    )
    snapshot = refresh_staged_lie_world_snapshot(
        _wrap_single_chart(chart),
        assume_fixed_topology=True,
        boundary=boundary,
        ray_coefficients=rays,
        site_density=density,
        site_color=color,
    )
    return snapshot, sites, boundary_pairs


def _cancelled_hidden_pole_snapshot():
    boundary = torch.tensor([[0.0, 0.0, 1.0, 0.0, -1.0]], dtype=DTYPE)
    safe_rays = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
        dtype=DTYPE,
    )
    bad_rays = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 1.0, 0.0, 0.0, -4.0]],
        dtype=DTYPE,
    )
    density = torch.tensor([0.4, 0.7], dtype=DTYPE)
    color = torch.tensor([[0.2, 0.4, 0.8], [0.8, 0.3, 0.1]], dtype=DTYPE)
    words = (make_stable_cell_word([0, 1], [-1, 0], [0, -2]),)
    chart = compile_lie_world_atlas(
        boundary=boundary,
        ray_coefficients=safe_rays,
        words=words,
        site_density=density,
        site_color=color,
        t_min=-0.5,
        t_max=0.5,
        near=0.1,
        far=2.0,
        node_count=4,
    )
    return refresh_staged_lie_world_snapshot(
        _wrap_single_chart(chart),
        assume_fixed_topology=True,
        boundary=boundary,
        ray_coefficients=bad_rays,
        site_density=density,
        site_color=color,
    )


def _constant_two_chart_snapshot():
    boundary = torch.empty((0, 5), dtype=DTYPE)
    rays = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
        dtype=DTYPE,
    )
    density = torch.tensor([0.7], dtype=DTYPE)
    color = torch.tensor([[0.2, 0.4, 0.8]], dtype=DTYPE)
    words = (make_stable_cell_word([0], [-1], [-2]),)
    charts = tuple(
        compile_lie_world_atlas(
            boundary=boundary,
            ray_coefficients=rays,
            words=words,
            site_density=density,
            site_color=color,
            t_min=t_min,
            t_max=t_max,
            near=0.1,
            far=2.0,
            node_count=2,
        )
        for t_min, t_max in ((-1.0, 0.0), (0.0, 1.0))
    )
    atlas = AdaptiveCompiledLieWorldAtlas(
        charts=charts,
        selections=(),
        policy=AdaptiveLieWorldCompilePolicy(),
        supplied_word_ordering_check=charts[0].supplied_word_ordering_check,
    )
    return refresh_staged_lie_world_snapshot(
        atlas,
        assume_fixed_topology=True,
        boundary=boundary,
        ray_coefficients=rays,
        site_density=density,
        site_color=color,
    )


def _third_cell_undercut_snapshot():
    sites = torch.tensor(
        [
            [0.0, 0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ],
        dtype=DTYPE,
    )
    pairs = torch.tensor([[0, 1], [0, 2], [1, 2]], dtype=torch.int64)
    boundary = power_boundary_parameters(sites, pairs)
    rays = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
        dtype=DTYPE,
    )
    density = torch.tensor([0.5, 0.7, 0.6], dtype=DTYPE)
    color = torch.tensor(
        [[0.2, 0.4, 0.8], [0.3, 0.8, 0.2], [0.8, 0.3, 0.2]],
        dtype=DTYPE,
    )
    # Boundary 1 separates sites 0 and 2 and is strictly ordered, but site 1
    # owns the omitted middle interval.
    words = (make_stable_cell_word([0, 2], [-1, 1], [1, -2]),)
    chart = compile_lie_world_atlas(
        boundary=boundary,
        ray_coefficients=rays,
        words=words,
        site_density=density,
        site_color=color,
        t_min=0.0,
        t_max=1.0,
        near=-1.5,
        far=1.5,
        node_count=2,
    )
    snapshot = refresh_staged_lie_world_snapshot(
        _wrap_single_chart(chart),
        assume_fixed_topology=True,
        boundary=boundary,
        ray_coefficients=rays,
        site_density=density,
        site_color=color,
    )
    return snapshot, sites, pairs


class ContinuousAdaptiveLieAcceptanceTest(unittest.TestCase):
    def test_track_local_sparse_mode_is_explicit_and_reaches_chart_certificate(self) -> None:
        snapshot, _, _ = _moving_power_snapshot()
        report = certify_prepared_adaptive_lie_world(
            snapshot,
            policy=ContinuousAdaptiveLieCertificationPolicy(
                transfer_tolerance=0.01,
                world_jacobian_tolerance=0.2,
                max_split_depth=2,
                max_leaves_per_chart=16,
                max_interval_jet_work_units_per_chart=10_000,
                arithmetic_fraction_bits=64,
                certificate_mode="track_local_sparse",
            ),
        )

        self.assertTrue(report.passed)
        self.assertEqual(report.policy.certificate_mode, "track_local_sparse")
        certificate = report.charts[0].certificate
        self.assertIsNotNone(certificate)
        self.assertEqual(certificate.certification_mode, "track_local_sparse")
        self.assertEqual(certificate.maximum_dual_dimension, 29)
        self.assertEqual(certificate.global_parameter_count, 29)
        self.assertFalse(certificate.parameter_labels_complete)

    def test_moving_power_world_and_induced_site_geometry_jet_are_certified(self) -> None:
        snapshot, sites, pairs = _moving_power_snapshot()
        report = certify_prepared_adaptive_lie_world(
            snapshot,
            policy=ContinuousAdaptiveLieCertificationPolicy(
                transfer_tolerance=0.01,
                world_jacobian_tolerance=0.2,
                site_geometry_jacobian_tolerance=0.25,
                max_split_depth=2,
                max_leaves_per_chart=16,
                max_interval_jet_work_units_per_chart=10_000,
                arithmetic_fraction_bits=64,
            ),
            sites=sites,
            boundary_pairs=pairs,
        )

        self.assertTrue(report.passed)
        self.assertEqual(len(report.charts), 1)
        self.assertTrue(report.charts[0].passed)
        self.assertLess(report.maximum_transfer_error_upper_bound, 0.01)
        self.assertLess(report.maximum_world_jacobian_error_upper_bound, 0.2)
        self.assertLess(report.maximum_site_geometry_jacobian_error_upper_bound, 0.25)
        self.assertEqual(report.boundary_to_site_maximum_column_l1_norm, 3.2)
        self.assertEqual(report.total_certificate_leaves, 2)
        self.assertGreater(report.minimum_cut_denominator_absolute_lower_bound, 0.79)
        self.assertGreaterEqual(report.minimum_fiber_speed_lower_bound, 0.99)
        self.assertGreater(report.minimum_coordinate_segment_length_lower_bound, 0.2)
        self.assertGreater(report.minimum_physical_segment_length_lower_bound, 0.2)
        self.assertTrue(report.continuous_time_coverage)
        self.assertTrue(report.atlas_world_provenance_certified)
        self.assertTrue(report.optimizer_site_geometry_covered)
        self.assertTrue(report.optimizer_site_geometry_accepted)
        self.assertFalse(report.continuous_acceptance_used_sampling)
        self.assertTrue(report.atlas_selection_was_not_reperformed)
        self.assertFalse(report.owner_identity_certified)
        self.assertFalse(report.runtime_floating_point_roundoff_certified)

    def test_optional_owner_identity_certificate_closes_third_cell_scope(self) -> None:
        snapshot, sites, pairs = _moving_power_snapshot()
        report = certify_prepared_adaptive_lie_world(
            snapshot,
            policy=ContinuousAdaptiveLieCertificationPolicy(
                transfer_tolerance=0.01,
                world_jacobian_tolerance=0.2,
                site_geometry_jacobian_tolerance=0.25,
                max_split_depth=2,
                max_leaves_per_chart=16,
                max_interval_jet_work_units_per_chart=10_000,
                arithmetic_fraction_bits=64,
                owner_identity_tolerance=1.0e-9,
                owner_max_split_depth=12,
                owner_max_leaves_per_chart=4096,
                owner_max_work_units_per_chart=100_000,
            ),
            sites=sites,
            boundary_pairs=pairs,
        )

        self.assertTrue(report.passed)
        self.assertTrue(report.owner_identity_certified)
        self.assertTrue(report.charts[0].owner_identity_certificate.passed)
        self.assertGreater(report.total_owner_certificate_leaves, 0)
        self.assertLessEqual(report.maximum_owner_difference_upper_bound, 1.0e-9)

    def test_owner_identity_gate_rejects_pairwise_word_with_third_cell_undercut(self) -> None:
        snapshot, sites, pairs = _third_cell_undercut_snapshot()
        report = certify_prepared_adaptive_lie_world(
            snapshot,
            policy=ContinuousAdaptiveLieCertificationPolicy(
                transfer_tolerance=1.0e-10,
                world_jacobian_tolerance=1.0e-10,
                site_geometry_jacobian_tolerance=1.0e-8,
                max_split_depth=0,
                max_leaves_per_chart=1,
                max_interval_jet_work_units_per_chart=100_000,
                arithmetic_fraction_bits=64,
                owner_identity_tolerance=1.0e-12,
            ),
            sites=sites,
            boundary_pairs=pairs,
        )

        self.assertFalse(report.passed)
        self.assertFalse(report.owner_identity_certified)
        self.assertFalse(report.continuous_time_coverage)
        self.assertTrue(
            any("third-cell undercut witness" in reason for reason in report.failure_reasons)
        )

    def test_every_selected_chart_contributes_to_aggregate_bounds_and_margins(self) -> None:
        report = certify_prepared_adaptive_lie_world(
            _constant_two_chart_snapshot(),
            policy=ContinuousAdaptiveLieCertificationPolicy(
                transfer_tolerance=1.0e-12,
                world_jacobian_tolerance=1.0e-12,
                max_split_depth=0,
                max_leaves_per_chart=1,
                max_interval_jet_work_units_per_chart=10_000,
                arithmetic_fraction_bits=64,
            ),
        )

        self.assertTrue(report.passed)
        self.assertEqual(len(report.charts), 2)
        self.assertTrue(all(chart.passed for chart in report.charts))
        self.assertEqual(report.total_certificate_leaves, 2)
        self.assertLess(report.maximum_transfer_error_upper_bound, 1.0e-14)
        self.assertLess(report.maximum_world_jacobian_error_upper_bound, 1.0e-14)
        self.assertGreater(report.minimum_fiber_speed_lower_bound, 0.99)
        self.assertGreater(report.minimum_coordinate_segment_length_lower_bound, 1.89)
        self.assertGreater(report.minimum_physical_segment_length_lower_bound, 1.89)
        self.assertTrue(report.continuous_time_coverage)

    def test_power_boundary_to_site_norm_matches_full_autograd_jacobian(self) -> None:
        sites = torch.tensor(
            [
                [-0.4, 0.2, 0.3, -0.1, 0.2],
                [0.6, -0.5, 0.7, 0.9, -0.3],
                [0.1, 0.8, -0.2, 0.4, 0.5],
            ],
            dtype=DTYPE,
        )
        pairs = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64)

        def flattened_boundaries(flat_sites: torch.Tensor) -> torch.Tensor:
            return power_boundary_parameters(flat_sites.reshape_as(sites), pairs).reshape(-1)

        jacobian = torch.autograd.functional.jacobian(flattened_boundaries, sites.reshape(-1))
        reference = float(jacobian.abs().sum(dim=0).max().item())
        observed = power_boundary_to_site_maximum_column_l1_norm(sites, pairs)
        self.assertGreaterEqual(observed, reference)
        self.assertLessEqual(observed, torch.nextafter(torch.tensor(reference), torch.tensor(float("inf"))).item())

    def test_world_or_atlas_mutation_fails_closed_before_certification(self) -> None:
        snapshot, _, _ = _moving_power_snapshot()
        snapshot.site_density[0].add_(0.01)
        with self.assertRaisesRegex(ValueError, "world tensors changed"):
            certify_prepared_adaptive_lie_world(
                snapshot,
                policy=ContinuousAdaptiveLieCertificationPolicy(1.0, 1.0),
            )

        snapshot, _, _ = _moving_power_snapshot()
        snapshot.atlas.charts[0].transfer_atlas.coefficients[0, 0, 0].add_(0.01)
        with self.assertRaisesRegex(ValueError, "no longer matches its bound world snapshot"):
            certify_prepared_adaptive_lie_world(
                snapshot,
                policy=ContinuousAdaptiveLieCertificationPolicy(1.0, 1.0),
            )

    def test_inadequate_rank_fails_at_explicit_certificate_limit(self) -> None:
        snapshot, _, _ = _moving_power_snapshot(node_count=2)
        report = certify_prepared_adaptive_lie_world(
            snapshot,
            policy=ContinuousAdaptiveLieCertificationPolicy(
                transfer_tolerance=1.0e-6,
                world_jacobian_tolerance=1.0e-5,
                max_split_depth=0,
                max_leaves_per_chart=1,
                max_interval_jet_work_units_per_chart=10_000,
                arithmetic_fraction_bits=64,
            ),
        )

        self.assertFalse(report.passed)
        self.assertTrue(report.continuous_time_coverage)
        self.assertTrue(report.rank_or_certificate_limit)
        self.assertTrue(any("continuous_transfer_tolerance_exceeded" in reason for reason in report.failure_reasons))
        self.assertTrue(
            any("continuous_world_jacobian_tolerance_exceeded" in reason for reason in report.failure_reasons)
        )

    def test_hidden_cancelled_pole_is_rejected_without_sampling(self) -> None:
        report = certify_prepared_adaptive_lie_world(
            _cancelled_hidden_pole_snapshot(),
            policy=ContinuousAdaptiveLieCertificationPolicy(
                transfer_tolerance=1.0,
                world_jacobian_tolerance=1.0,
                max_split_depth=1,
                max_leaves_per_chart=4,
                max_interval_jet_work_units_per_chart=10_000,
                arithmetic_fraction_bits=64,
            ),
        )

        self.assertFalse(report.passed)
        self.assertFalse(report.continuous_time_coverage)
        self.assertTrue(report.rank_or_certificate_limit)
        self.assertIsNone(report.charts[0].certificate)
        self.assertTrue(any("precondition" in reason for reason in report.failure_reasons))
        self.assertFalse(report.continuous_acceptance_used_sampling)

    def test_work_budget_rejects_before_interval_jet_construction(self) -> None:
        snapshot, _, _ = _moving_power_snapshot()
        report = certify_prepared_adaptive_lie_world(
            snapshot,
            policy=ContinuousAdaptiveLieCertificationPolicy(
                transfer_tolerance=1.0,
                world_jacobian_tolerance=1.0,
                max_interval_jet_work_units_per_chart=1,
                arithmetic_fraction_bits=64,
            ),
        )

        self.assertFalse(report.passed)
        self.assertFalse(report.continuous_time_coverage)
        self.assertTrue(report.rank_or_certificate_limit)
        self.assertIsNone(report.charts[0].certificate)
        self.assertGreater(report.maximum_estimated_interval_jet_work_units, 1)
        self.assertIn("work_budget_exceeded", report.failure_reasons[0])


if __name__ == "__main__":
    unittest.main()
