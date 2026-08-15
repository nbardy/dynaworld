from __future__ import annotations

import inspect
from fractions import Fraction

import torch
from kinetic_active_owner_chart_compiler import (
    ActiveKineticOwnerChartProgram,
    compile_active_kinetic_owner_charts,
)
from kinetic_continuous_transfer_acceptance import (
    KineticContinuousTransferPolicy,
    select_continuously_certified_kinetic_transfer,
)
from kinetic_power_word_compiler import AffineKineticPowerSites

DTYPE = torch.float64


def _static_x_ray() -> torch.Tensor:
    return torch.tensor(
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        dtype=DTYPE,
    )


def _sites_from_ray_lines(
    slopes: list[tuple[int | Fraction, int | Fraction]],
    intercepts: list[tuple[int | Fraction, int | Fraction, int | Fraction]],
) -> AffineKineticPowerSites:
    positions = []
    velocities = []
    weights = []
    for (slope0, slope1), (bias0, bias1, bias2) in zip(
        slopes,
        intercepts,
        strict=True,
    ):
        position = -Fraction(slope0) / 2
        velocity = -Fraction(slope1) / 2
        positions.append((position, Fraction(0), Fraction(0)))
        velocities.append((velocity, Fraction(0), Fraction(0)))
        weights.append(
            (
                position * position - Fraction(bias0),
                2 * position * velocity - Fraction(bias1),
                velocity * velocity - Fraction(bias2),
            )
        )
    return AffineKineticPowerSites(
        positions0=torch.tensor([[float(value) for value in row] for row in positions], dtype=DTYPE),
        velocities=torch.tensor([[float(value) for value in row] for row in velocities], dtype=DTYPE),
        weight_coefficients=torch.tensor([[float(value) for value in row] for row in weights], dtype=DTYPE),
    )


def _policy(
    ranks: tuple[int, ...],
    *,
    tolerance: float,
    max_split_depth: int = 8,
) -> KineticContinuousTransferPolicy:
    return KineticContinuousTransferPolicy(
        node_count_schedule=ranks,
        transfer_tolerance=tolerance,
        material_jacobian_entry_tolerance=tolerance,
        material_jvp_direction_l1_bound=1.0,
        material_jvp_tolerance=tolerance,
        material_vjp_cotangent_l1_bound=1.0,
        material_vjp_tolerance=tolerance,
        max_split_depth=max_split_depth,
        max_leaves_per_rank=64,
        arithmetic_fraction_bits=96,
        max_material_dual_dimension=32,
    )


def test_constant_kinetic_chart_has_continuous_primal_jvp_and_vjp_certificate() -> None:
    sites = AffineKineticPowerSites(
        positions0=torch.zeros((1, 3), dtype=DTYPE),
        velocities=torch.zeros((1, 3), dtype=DTYPE),
        weight_coefficients=torch.zeros((1, 1), dtype=DTYPE),
    )
    ray = _static_x_ray()
    owner_program = compile_active_kinetic_owner_charts(
        sites,
        ray,
        t_min=-1,
        t_max=1,
        near=0,
        far=1,
    )
    selection = select_continuously_certified_kinetic_transfer(
        owner_program,
        sites,
        ray,
        torch.tensor([0.4], dtype=DTYPE),
        torch.tensor([[0.8, 0.3, 0.15]], dtype=DTYPE),
        policy=_policy((2, 4), tolerance=2.0e-11, max_split_depth=4),
    )

    assert selection.passed
    assert selection.program is not None and selection.transfer is not None
    assert isinstance(selection.program.binding.program, ActiveKineticOwnerChartProgram)
    assert selection.program.binding.compiler_provenance == "active_kinetic_owner_chart_compiler_v1"
    assert not selection.program.binding.program.work.exhaustive_triple_enumeration_used
    assert selection.charts[0].selected_node_count == 2
    certificate = selection.charts[0].attempts[-1].certificate
    assert certificate is not None and certificate.passed
    assert certificate.continuous_supported_interval_coverage
    assert certificate.material_jacobian_certified
    assert certificate.material_jvp_action_certified
    assert certificate.material_vjp_action_certified
    assert certificate.certified_sample_weight_semantics == "real_arithmetic_second_form_barycentric"
    assert not certificate.runtime_dense_fallback_certified
    assert not certificate.runtime_floating_point_roundoff_certified
    assert certificate.transfer_error_upper_bound <= certificate.transfer_tolerance
    assert certificate.material_jvp_error_upper_bound <= certificate.material_jvp_tolerance
    assert certificate.material_vjp_error_upper_bound <= certificate.material_vjp_tolerance
    assert selection.rank_selection_used_requested_samples is False
    assert selection.retained_validation_sample_bytes == 0


def test_low_rank_adversarial_moving_color_boundary_fails_closed() -> None:
    # The owner word stays (0,1), but differently colored/dense runs exchange
    # length. Its affine-Lie material Jacobian is not rank-two in time.
    sites = _sites_from_ray_lines(
        slopes=[(0, 0), (-2, 0)],
        intercepts=[(0, 0, 0), (1, -1, 0)],
    )
    ray = _static_x_ray()
    owner_program = compile_active_kinetic_owner_charts(
        sites,
        ray,
        t_min=Fraction(-9, 10),
        t_max=Fraction(9, 10),
        near=0,
        far=1,
    )
    selection = select_continuously_certified_kinetic_transfer(
        owner_program,
        sites,
        ray,
        torch.tensor([0.2, 1.1], dtype=DTYPE),
        torch.tensor([[0.95, 0.08, 0.03], [0.03, 0.25, 0.98]], dtype=DTYPE),
        policy=_policy((2,), tolerance=1.0e-4, max_split_depth=3),
    )

    assert not selection.passed
    assert selection.program is None and selection.transfer is None
    assert selection.failure_reasons == ("chart[0]: no candidate rank passed continuous certification",)
    attempt = selection.charts[0].attempts[0]
    assert not attempt.passed
    assert attempt.certificate is not None
    assert (
        attempt.certificate.transfer_error_upper_bound > 1.0e-4
        or attempt.certificate.material_jacobian_entry_error_upper_bound > 1.0e-4
    )


def test_rank_selection_api_has_no_requested_time_or_frame_input() -> None:
    parameters = inspect.signature(select_continuously_certified_kinetic_transfer).parameters
    assert not {
        "times",
        "sample_times",
        "frame_count",
        "requested_frame_count",
        "sample_count",
    }.intersection(parameters)
