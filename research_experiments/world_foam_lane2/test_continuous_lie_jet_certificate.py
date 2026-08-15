from __future__ import annotations

import unittest
from fractions import Fraction

import torch
from compiled_lie_world_adjoint import DTYPE, compile_lie_world_atlas
from compiled_transfer_adjoint import direct_word_transfer, make_stable_cell_word
from continuous_lie_jet_certificate import (
    ContinuousCertificateError,
    _Arithmetic,
    _Dual,
    _dual_unary_at_exact_zero,
    certify_fixed_topology_lie_jet,
    certify_fixed_topology_lie_jet_track_local,
)
from transfer_lie_chart import (
    chebyshev_basis,
    transfer_lie_decode,
    transfer_lie_encode,
)


def _constant_fixture() -> dict[str, object]:
    return {
        "boundary": torch.empty((0, 5), dtype=DTYPE),
        "ray_coefficients": torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
            dtype=DTYPE,
        ),
        "words": (make_stable_cell_word([0], [-1], [-2]),),
        "site_density": torch.tensor([0.7], dtype=DTYPE),
        "site_color": torch.tensor([[0.2, 0.4, 0.8]], dtype=DTYPE),
        "t_min": -1.0,
        "t_max": 1.0,
        "near": 0.1,
        "far": 2.0,
    }


def _moving_mobius_fixture() -> dict[str, object]:
    return {
        "boundary": torch.tensor([[0.1, 0.0, 1.0, 0.1, -1.0]], dtype=DTYPE),
        "ray_coefficients": torch.tensor(
            [[0.02, 0.0, 0.0, 0.01, 0.0, 0.02, 0.0, 0.0, 1.0, 0.01, 0.0, 0.05]],
            dtype=DTYPE,
        ),
        "words": (make_stable_cell_word([0, 1], [-1, 0], [0, -2]),),
        "site_density": torch.tensor([0.4, 0.7], dtype=DTYPE),
        "site_color": torch.tensor([[0.2, 0.4, 0.8], [0.8, 0.3, 0.1]], dtype=DTYPE),
        "t_min": -0.5,
        "t_max": 0.5,
        "near": 0.1,
        "far": 2.0,
    }


def _hard_dormant_fixture() -> dict[str, object]:
    return {
        "boundary": torch.tensor([[0.0, 0.0, 1.0, -0.9, -1.0]], dtype=DTYPE),
        "ray_coefficients": torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
            dtype=DTYPE,
        ),
        "words": (make_stable_cell_word([0, 1], [-1, 0], [0, -2]),),
        "site_density": torch.tensor([50.0, 0.0], dtype=DTYPE),
        "site_color": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=DTYPE),
        "t_min": -1.0,
        "t_max": 1.0,
        "near": 0.05,
        "far": 2.0,
    }


def _two_track_moving_fixture() -> dict[str, object]:
    fixture = _moving_mobius_fixture()
    fixture["ray_coefficients"] = torch.cat(
        (
            fixture["ray_coefficients"],
            fixture["ray_coefficients"]
            + torch.tensor(
                [[0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                dtype=DTYPE,
            ),
        ),
        dim=0,
    )
    fixture["words"] = (fixture["words"][0], fixture["words"][0])
    return fixture


def _cancelled_pole_fixture() -> dict[str, object]:
    fixture = _moving_mobius_fixture()
    fixture["boundary"] = torch.tensor([[0.0, 0.0, 1.0, 0.0, -1.0]], dtype=DTYPE)
    fixture["ray_coefficients"] = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
        dtype=DTYPE,
    )
    return fixture


def _compile(fixture: dict[str, object], *, node_count: int):
    return compile_lie_world_atlas(
        boundary=fixture["boundary"],
        ray_coefficients=fixture["ray_coefficients"],
        words=fixture["words"],
        site_density=fixture["site_density"],
        site_color=fixture["site_color"],
        t_min=fixture["t_min"],
        t_max=fixture["t_max"],
        near=fixture["near"],
        far=fixture["far"],
        node_count=node_count,
    )


def _certificate(fixture: dict[str, object], atlas, **overrides: object):
    arguments: dict[str, object] = {
        "atlas": atlas,
        "boundary": fixture["boundary"],
        "ray_coefficients": fixture["ray_coefficients"],
        "site_density": fixture["site_density"],
        "site_color": fixture["site_color"],
        "transfer_tolerance": 0.1,
        "world_jacobian_tolerance": 0.5,
        "max_split_depth": 0,
        "arithmetic_fraction_bits": 80,
    }
    arguments.update(overrides)
    return certify_fixed_topology_lie_jet(**arguments)


class ContinuousLieJetCertificateTest(unittest.TestCase):
    def test_outward_sqrt_and_exp_primitives_enclose_reference_values(self) -> None:
        arithmetic = _Arithmetic(96)
        square_root = arithmetic.sqrt(arithmetic.point(Fraction(2)))
        self.assertLessEqual(square_root.lo * square_root.lo, 2)
        self.assertGreaterEqual(square_root.hi * square_root.hi, 2)

        argument = arithmetic.point(Fraction(7, 13))
        positive = arithmetic.exp(argument)
        negative = arithmetic.exp(arithmetic.neg(argument))
        product = arithmetic.mul(positive, negative)
        self.assertLessEqual(product.lo, 1)
        self.assertGreaterEqual(product.hi, 1)
        self.assertLess(float(positive.hi - positive.lo), 1.0e-24)

    def test_mixed_time_world_hyperdual_rules_enclose_analytic_derivatives(self) -> None:
        arithmetic = _Arithmetic(96)
        x = arithmetic.dual_variable(arithmetic.point(Fraction(2, 3)), 1, 0)
        time = arithmetic.dual_time(arithmetic.point(Fraction(1, 5)), 1)
        exponential = arithmetic.dual_exp(arithmetic.dual_mul(x, time))
        reference_exp = arithmetic.exp(arithmetic.point(Fraction(2, 15)))
        reference_world = arithmetic.mul(arithmetic.point(Fraction(1, 5)), reference_exp)
        reference_time = arithmetic.mul(arithmetic.point(Fraction(2, 3)), reference_exp)
        reference_mixed = arithmetic.mul(arithmetic.point(Fraction(17, 15)), reference_exp)
        for observed, reference in (
            (exponential.tangent[0], reference_world),
            (exponential.time_tangent, reference_time),
            (exponential.mixed_time_tangent[0], reference_mixed),
        ):
            self.assertLessEqual(observed.lo, reference.lo)
            self.assertGreaterEqual(observed.hi, reference.hi)

        root = arithmetic.dual_sqrt(arithmetic.dual_add(x, time))
        reference_root = arithmetic.sqrt(arithmetic.point(Fraction(13, 15)))
        reference_first = arithmetic.reciprocal(arithmetic.mul(arithmetic.point(2), reference_root))
        reference_mixed_root = arithmetic.neg(
            arithmetic.reciprocal(
                arithmetic.mul(
                    arithmetic.point(4),
                    arithmetic.mul(arithmetic.square(reference_root), reference_root),
                )
            )
        )
        for observed, reference in (
            (root.tangent[0], reference_first),
            (root.time_tangent, reference_first),
            (root.mixed_time_tangent[0], reference_mixed_root),
        ):
            self.assertLessEqual(observed.lo, reference.lo)
            self.assertGreaterEqual(observed.hi, reference.hi)

    def test_exact_zero_lie_scalar_branches_use_the_removable_taylor_jet(self) -> None:
        arithmetic = _Arithmetic(96)
        zero = arithmetic.zero
        one = arithmetic.one
        argument = _Dual(zero, (one,), one, (zero,))

        inverse_phi = _dual_unary_at_exact_zero(
            arithmetic,
            argument,
            value=Fraction(1),
            first_derivative=Fraction(1, 2),
            second_derivative=Fraction(1, 6),
        )
        phi = _dual_unary_at_exact_zero(
            arithmetic,
            argument,
            value=Fraction(1),
            first_derivative=Fraction(-1, 2),
            second_derivative=Fraction(1, 3),
        )
        self.assertEqual(inverse_phi.value, arithmetic.one)
        self.assertEqual(inverse_phi.tangent[0], arithmetic.point(Fraction(1, 2)))
        self.assertEqual(inverse_phi.time_tangent, arithmetic.point(Fraction(1, 2)))
        self.assertEqual(inverse_phi.mixed_time_tangent[0], arithmetic.point(Fraction(1, 6)))
        self.assertEqual(phi.value, arithmetic.one)
        self.assertEqual(phi.tangent[0], arithmetic.point(Fraction(-1, 2)))
        self.assertEqual(phi.mixed_time_tangent[0], arithmetic.point(Fraction(1, 3)))

    def test_constant_word_has_tight_continuous_primal_and_full_world_jet_certificate(self) -> None:
        fixture = _constant_fixture()
        report = _certificate(
            fixture,
            _compile(fixture, node_count=2),
            transfer_tolerance=1.0e-12,
            world_jacobian_tolerance=1.0e-12,
        )
        self.assertTrue(report.passed)
        self.assertLess(report.transfer_error_upper_bound, 1.0e-14)
        self.assertLess(report.world_jacobian_error_upper_bound, 1.0e-14)
        self.assertEqual(report.leaf_count, 1)
        self.assertIsNone(report.minimum_cut_denominator_absolute_lower_bound)
        self.assertGreater(report.minimum_fiber_speed_lower_bound, 0.99)
        self.assertGreater(report.minimum_coordinate_segment_length_lower_bound, 1.89)
        self.assertGreater(report.minimum_physical_segment_length_lower_bound, 1.89)
        self.assertTrue(report.continuous_time_coverage)
        self.assertFalse(report.owner_identity_certified)
        self.assertFalse(report.atlas_snapshot_provenance_certified)
        self.assertFalse(report.runtime_floating_point_roundoff_certified)
        self.assertEqual(len(report.parameter_labels), 16)

    def test_exactly_transparent_word_certifies_identity_and_density_tangent(self) -> None:
        fixture = _constant_fixture()
        fixture["site_density"] = torch.zeros((1,), dtype=DTYPE)
        report = _certificate(
            fixture,
            _compile(fixture, node_count=2),
            transfer_tolerance=1.0e-12,
            world_jacobian_tolerance=1.0e-12,
        )
        self.assertTrue(report.passed)
        self.assertEqual(report.minimum_exact_total_optical_depth_lower_bound, 0.0)
        self.assertEqual(report.minimum_compiled_kappa_lower_bound, 0.0)
        self.assertTrue(report.compiled_lie_cone_certified)
        self.assertLess(report.transfer_error_upper_bound, 1.0e-14)
        self.assertLess(report.world_jacobian_error_upper_bound, 1.0e-14)

        sparse = certify_fixed_topology_lie_jet_track_local(
            _compile(fixture, node_count=2),
            boundary=fixture["boundary"],
            ray_coefficients=fixture["ray_coefficients"],
            site_density=fixture["site_density"],
            site_color=fixture["site_color"],
            transfer_tolerance=1.0e-12,
            world_jacobian_tolerance=1.0e-12,
            max_split_depth=0,
            arithmetic_fraction_bits=80,
        )
        self.assertTrue(sparse.passed)
        self.assertEqual(sparse.certification_mode, "track_local_sparse")
        self.assertEqual(sparse.maximum_dual_dimension, 16)

    def test_track_local_sparse_bounds_match_dense_oracle_on_small_shared_world(self) -> None:
        fixture = _two_track_moving_fixture()
        atlas = _compile(fixture, node_count=4)
        arguments = {
            "atlas": atlas,
            "boundary": fixture["boundary"],
            "ray_coefficients": fixture["ray_coefficients"],
            "site_density": fixture["site_density"],
            "site_color": fixture["site_color"],
            "transfer_tolerance": 0.1,
            "world_jacobian_tolerance": 0.5,
            "max_split_depth": 0,
            "arithmetic_fraction_bits": 80,
        }
        dense = certify_fixed_topology_lie_jet(**arguments)
        sparse = certify_fixed_topology_lie_jet_track_local(**arguments)

        self.assertEqual(sparse.passed, dense.passed)
        self.assertEqual(sparse.transfer_error_upper_bound, dense.transfer_error_upper_bound)
        self.assertEqual(
            sparse.world_jacobian_error_upper_bound,
            dense.world_jacobian_error_upper_bound,
        )
        self.assertEqual(
            sparse.world_jacobian_error_upper_bound_by_block,
            dense.world_jacobian_error_upper_bound_by_block,
        )
        self.assertEqual(
            sparse.minimum_cut_denominator_absolute_lower_bound,
            dense.minimum_cut_denominator_absolute_lower_bound,
        )
        self.assertEqual(sparse.minimum_fiber_speed_lower_bound, dense.minimum_fiber_speed_lower_bound)
        self.assertEqual(sparse.maximum_dual_dimension, 29)
        self.assertEqual(sparse.global_parameter_count, 45)
        self.assertEqual(sparse.total_seeded_parameter_occurrences, 58)
        self.assertEqual(sparse.parameter_labels, ())
        self.assertFalse(sparse.parameter_labels_complete)
        self.assertEqual(len(sparse.parameter_scope_digest), 64)

    def test_track_local_dual_dimension_ignores_unreferenced_global_resources(self) -> None:
        fixture = _constant_fixture()
        fixture["boundary"] = torch.zeros((64, 5), dtype=DTYPE)
        fixture["boundary"][:, 2] = 1.0
        fixture["boundary"][:, 4] = -torch.arange(3, 67, dtype=DTYPE)
        fixture["site_density"] = torch.full((101,), 0.7, dtype=DTYPE)
        fixture["site_color"] = torch.full((101, 3), 0.4, dtype=DTYPE)
        report = certify_fixed_topology_lie_jet_track_local(
            _compile(fixture, node_count=2),
            boundary=fixture["boundary"],
            ray_coefficients=fixture["ray_coefficients"],
            site_density=fixture["site_density"],
            site_color=fixture["site_color"],
            transfer_tolerance=1.0e-12,
            world_jacobian_tolerance=1.0e-12,
            max_split_depth=0,
            arithmetic_fraction_bits=64,
        )
        self.assertTrue(report.passed)
        self.assertEqual(report.maximum_dual_dimension, 16)
        self.assertEqual(report.total_seeded_parameter_occurrences, 16)
        self.assertEqual(report.global_parameter_count, 736)

    def test_track_local_path_rejects_unsupported_global_vjp_reduction(self) -> None:
        fixture = _constant_fixture()
        with self.assertRaisesRegex(ContinuousCertificateError, "summed/global VJP"):
            certify_fixed_topology_lie_jet_track_local(
                _compile(fixture, node_count=2),
                boundary=fixture["boundary"],
                ray_coefficients=fixture["ray_coefficients"],
                site_density=fixture["site_density"],
                site_color=fixture["site_color"],
                transfer_tolerance=1.0,
                world_jacobian_tolerance=1.0,
                shared_parameter_reduction="summed_vjp",
            )

    def test_track_local_path_fails_before_quadratic_state_exceeds_dimension_cap(self) -> None:
        fixture = _constant_fixture()
        atlas = _compile(fixture, node_count=2)
        with self.assertRaisesRegex(ContinuousCertificateError, "before interval-jet construction"):
            certify_fixed_topology_lie_jet_track_local(
                atlas,
                boundary=fixture["boundary"],
                ray_coefficients=fixture["ray_coefficients"],
                site_density=fixture["site_density"],
                site_color=fixture["site_color"],
                transfer_tolerance=1.0,
                world_jacobian_tolerance=1.0,
                max_local_dual_dimension=15,
            )
        with self.assertRaisesRegex(ContinuousCertificateError, "hidden full-snapshot casts"):
            certify_fixed_topology_lie_jet_track_local(
                atlas,
                boundary=fixture["boundary"],
                ray_coefficients=fixture["ray_coefficients"],
                site_density=fixture["site_density"],
                site_color=fixture["site_color"].float(),
                transfer_tolerance=1.0,
                world_jacobian_tolerance=1.0,
            )

    def test_moving_mobius_certificate_bounds_independent_dense_autograd_error(self) -> None:
        fixture = _moving_mobius_fixture()
        atlas = _compile(fixture, node_count=4)
        report = _certificate(fixture, atlas)
        self.assertTrue(report.passed)
        self.assertGreater(report.minimum_cut_denominator_absolute_lower_bound, 0.9)
        self.assertGreater(report.minimum_coordinate_segment_length_lower_bound, 0.8)
        self.assertGreater(report.minimum_physical_segment_length_lower_bound, 0.79)
        self.assertIn(
            "mobius_depth_coefficient",
            report.world_jacobian_error_upper_bound_by_block,
        )

        times = torch.linspace(-0.5, 0.5, 7, dtype=DTYPE)
        inputs = tuple(
            fixture[name].clone().requires_grad_(True)
            for name in ("boundary", "ray_coefficients", "site_density", "site_color")
        )

        def error_function(boundary, rays, density, color):
            exact = direct_word_transfer(
                boundary=boundary,
                ray_coefficients=rays,
                words=fixture["words"],
                site_density=density,
                site_color=color,
                times=times,
                near=fixture["near"],
                far=fixture["far"],
            )
            node_transfer = direct_word_transfer(
                boundary=boundary,
                ray_coefficients=rays,
                words=fixture["words"],
                site_density=density,
                site_color=color,
                times=atlas.transfer_atlas.node_times,
                near=fixture["near"],
                far=fixture["far"],
            )
            node_chart = transfer_lie_encode(node_transfer)
            fresh_coefficients = torch.einsum(
                "kn,pnc->pkc",
                atlas.transfer_atlas.fit_matrix,
                node_chart,
            )
            # Stored primal plus the compiler's real-arithmetic node Jacobian.
            coefficients = (
                atlas.transfer_atlas.coefficients
                + fresh_coefficients
                - fresh_coefficients.detach()
            )
            basis = chebyshev_basis(
                times,
                t_min=fixture["t_min"],
                t_max=fixture["t_max"],
                rank=atlas.node_count,
            )
            compiled = transfer_lie_decode(torch.einsum("fk,pkc->pfc", basis, coefficients))
            return exact - compiled

        primal_error = float(error_function(*inputs).detach().abs().max().item())
        jacobians = torch.autograd.functional.jacobian(error_function, inputs)
        jacobian_error = max(float(jacobian.detach().abs().max().item()) for jacobian in jacobians)
        self.assertLessEqual(primal_error, report.transfer_error_upper_bound + 1.0e-14)
        self.assertLessEqual(jacobian_error, report.world_jacobian_error_upper_bound + 1.0e-13)

    def test_point_witness_proves_forward_rank_does_not_certify_world_tangent_rank(self) -> None:
        fixture = _hard_dormant_fixture()
        report = _certificate(
            fixture,
            _compile(fixture, node_count=2),
            transfer_tolerance=100.0,
            world_jacobian_tolerance=10000.0,
            arithmetic_fraction_bits=72,
        )
        self.assertLess(report.transfer_point_witness_lower_bound, 1.0e-12)
        self.assertGreater(report.world_jacobian_point_witness_lower_bound, 0.1)

    def test_hidden_mobius_pole_fails_closed_instead_of_sampling_through_it(self) -> None:
        fixture = _cancelled_pole_fixture()
        atlas = _compile(fixture, node_count=4)
        adversarial_rays = fixture["ray_coefficients"].clone()
        # A=B=C=D up to the same (1-4t) factor, so sampled cut depth is the
        # apparently benign constant one wherever it is defined.  Every
        # compiler node is finite, but both the Möbius denominator and fiber
        # speed vanish at t=0.25 inside the chart.
        adversarial_rays[0, 5] = 4.0
        adversarial_rays[0, 11] = -4.0
        sampled_nodes = direct_word_transfer(
            boundary=fixture["boundary"],
            ray_coefficients=adversarial_rays,
            words=fixture["words"],
            site_density=fixture["site_density"],
            site_color=fixture["site_color"],
            times=atlas.transfer_atlas.node_times,
            near=fixture["near"],
            far=fixture["far"],
        )
        self.assertTrue(bool(torch.isfinite(sampled_nodes).all().item()))
        with self.assertRaises(ContinuousCertificateError):
            certify_fixed_topology_lie_jet(
                atlas,
                boundary=fixture["boundary"],
                ray_coefficients=adversarial_rays,
                site_density=fixture["site_density"],
                site_color=fixture["site_color"],
                transfer_tolerance=1.0,
                world_jacobian_tolerance=1.0,
                max_split_depth=1,
                arithmetic_fraction_bits=72,
            )


if __name__ == "__main__":
    unittest.main()
