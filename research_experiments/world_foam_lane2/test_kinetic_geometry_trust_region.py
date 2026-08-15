from __future__ import annotations

from fractions import Fraction

import torch
from kinetic_active_owner_chart_compiler import compile_active_kinetic_owner_charts
from kinetic_geometry_trust_region import (
    certify_event_free_binary64_geometry_candidate,
    certify_event_free_kinetic_geometry_trust_region,
    make_kinetic_geometry_update_direction,
)
from kinetic_power_word_compiler import (
    AffineKineticPowerSites,
    discover_kinetic_power_word_at_time,
)

DTYPE = torch.float64


def _static_x_ray() -> torch.Tensor:
    return torch.tensor(
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        dtype=DTYPE,
    )


def _sites_from_static_ray_lines(
    slopes: list[int | Fraction],
    intercepts: list[int | Fraction],
) -> AffineKineticPowerSites:
    positions = []
    weights = []
    for slope, intercept in zip(slopes, intercepts, strict=True):
        position = -Fraction(slope) / 2
        positions.append((position, Fraction(0), Fraction(0)))
        weights.append((position * position - Fraction(intercept),))
    return AffineKineticPowerSites(
        positions0=torch.tensor(
            [[float(value) for value in row] for row in positions],
            dtype=DTYPE,
        ),
        velocities=torch.zeros((len(slopes), 3), dtype=DTYPE),
        weight_coefficients=torch.tensor(
            [[float(value) for value in row] for row in weights],
            dtype=DTYPE,
        ),
    )


def _compile(sites: AffineKineticPowerSites, ray: torch.Tensor | None = None):
    return compile_active_kinetic_owner_charts(
        sites,
        _static_x_ray() if ray is None else ray,
        t_min=0,
        t_max=1,
        near=0,
        far=1,
    )


def _word_at(
    sites: AffineKineticPowerSites,
    ray: torch.Tensor,
    *,
    time: Fraction = Fraction(1, 2),
) -> tuple[int, ...]:
    result = discover_kinetic_power_word_at_time(
        sites,
        ray,
        time=time,
        near=0,
        far=1,
    )
    return tuple(int(owner) for owner in result.word.owners.tolist())


def _updated_sites(
    sites: AffineKineticPowerSites,
    direction,
    step: float,
) -> AffineKineticPowerSites:
    return AffineKineticPowerSites(
        positions0=sites.positions0 + step * direction.positions0,
        velocities=sites.velocities + step * direction.velocities,
        weight_coefficients=sites.weight_coefficients + step * direction.weight_coefficients,
    )


def test_zero_direction_certifies_the_requested_continuous_radius() -> None:
    sites = _sites_from_static_ray_lines([0, -2], [0, 1])
    ray = _static_x_ray()
    program = _compile(sites)
    assert program.passed
    assert tuple(chart.owner_word for chart in program.charts) == ((0, 1),)

    certificate = certify_event_free_kinetic_geometry_trust_region(
        sites,
        ray,
        program,
        make_kinetic_geometry_update_direction(sites, ray),
        requested_step_radius=10,
    )

    assert certificate.passed
    assert certificate.certified_step_radius == 10
    assert certificate.continuous_time_proof
    assert certificate.active_event_endpoints_reused is False
    assert certificate.event_time_derivatives_included is False
    assert certificate.predicate_certificates
    assert all(item.accepts(Fraction(10)) for item in certificate.predicate_certificates)


def test_near_optimizer_root_limits_radius_and_directional_replay() -> None:
    epsilon = Fraction(1, 1024)
    sites = _sites_from_static_ray_lines([0, -2, 0], [0, 1, epsilon])
    ray = _static_x_ray()
    program = _compile(sites)
    assert tuple(chart.owner_word for chart in program.charts) == ((0, 1),)
    delta_weights = torch.zeros_like(sites.weight_coefficients)
    delta_weights[2, 0] = 1.0
    direction = make_kinetic_geometry_update_direction(
        sites,
        ray,
        delta_weight_coefficients=delta_weights,
    )

    certificate = certify_event_free_kinetic_geometry_trust_region(
        sites,
        ray,
        program,
        direction,
        requested_step_radius=1,
    )

    assert certificate.passed
    assert 0 < certificate.certified_step_radius < epsilon
    inside = float(certificate.certified_step_radius / 2)
    assert _word_at(_updated_sites(sites, direction, inside), ray) == (0, 1)
    assert _word_at(_updated_sites(sites, direction, float(2 * epsilon)), ray) != (0, 1)
    assert certificate.limiting_predicate_kind in {
        "near_owner_gap",
        "far_owner_gap",
        "internal_cut_owner_gap",
    }


def test_denominator_collapse_is_strictly_outside_certified_radius() -> None:
    sites = _sites_from_static_ray_lines([0, -2], [0, 1])
    ray = _static_x_ray()
    program = _compile(sites)
    delta_positions = torch.zeros_like(sites.positions0)
    delta_positions[1, 0] = -1.0
    direction = make_kinetic_geometry_update_direction(
        sites,
        ray,
        delta_positions0=delta_positions,
    )

    certificate = certify_event_free_kinetic_geometry_trust_region(
        sites,
        ray,
        program,
        direction,
        requested_step_radius=2,
    )

    assert certificate.passed
    assert 0 < certificate.certified_step_radius < 1
    assert _word_at(_updated_sites(sites, direction, 0.5), ray) == (0, 1)
    assert _word_at(_updated_sites(sites, direction, 1.5), ray) != (0, 1)
    denominator_certificates = [
        item for item in certificate.predicate_certificates if item.kind == "active_cut_denominator"
    ]
    assert len(denominator_certificates) == 1
    assert denominator_certificates[0].accepts(certificate.certified_step_radius)
    assert not denominator_certificates[0].accepts(Fraction(1))


def test_inactive_competitor_is_part_of_the_continuous_certificate() -> None:
    sites = _sites_from_static_ray_lines([0, -2, -1], [0, 1, 4])
    ray = _static_x_ray()
    program = _compile(sites)
    assert tuple(chart.owner_word for chart in program.charts) == ((0, 1),)
    delta_weights = torch.zeros_like(sites.weight_coefficients)
    delta_weights[2, 0] = 4.0
    direction = make_kinetic_geometry_update_direction(
        sites,
        ray,
        delta_weight_coefficients=delta_weights,
    )

    certificate = certify_event_free_kinetic_geometry_trust_region(
        sites,
        ray,
        program,
        direction,
        requested_step_radius=2,
    )

    assert certificate.passed
    assert 0 < certificate.certified_step_radius < 1
    assert any(2 in item.site_ids for item in certificate.predicate_certificates)
    assert _word_at(_updated_sites(sites, direction, 0.5), ray) == (0, 1)
    assert 2 in _word_at(_updated_sites(sites, direction, 1.5), ray)


def test_multichart_and_simultaneous_event_programs_fail_closed() -> None:
    moving = AffineKineticPowerSites(
        positions0=torch.tensor([[0, 0, 0], [1, 0, 0]], dtype=DTYPE),
        velocities=torch.zeros((2, 3), dtype=DTYPE),
        weight_coefficients=torch.tensor([[0, 0], [0, 1]], dtype=DTYPE),
    )
    ray = _static_x_ray()
    multichart = compile_active_kinetic_owner_charts(
        moving,
        ray,
        t_min=-2,
        t_max=2,
        near=0,
        far=1,
    )
    assert multichart.passed
    assert len(multichart.charts) > 1
    rejected = certify_event_free_kinetic_geometry_trust_region(
        moving,
        ray,
        multichart,
        make_kinetic_geometry_update_direction(moving, ray),
    )
    assert not rejected.passed
    assert rejected.certified_step_radius == 0
    assert rejected.reason == "active_or_multichart_program_requires_event_root_reisolation"

    simultaneous = AffineKineticPowerSites(
        positions0=torch.tensor([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=DTYPE),
        velocities=torch.zeros((3, 3), dtype=DTYPE),
        weight_coefficients=torch.tensor([[0, 0], [1, 1], [4, -1]], dtype=DTYPE),
    )
    failed_program = compile_active_kinetic_owner_charts(
        simultaneous,
        ray,
        t_min=-1,
        t_max=1,
        near=0,
        far=1,
    )
    assert not failed_program.passed
    assert failed_program.unresolved_degeneracies[0].kind == "ambiguous_simultaneous_active_event"
    failed = certify_event_free_kinetic_geometry_trust_region(
        simultaneous,
        ray,
        failed_program,
        make_kinetic_geometry_update_direction(simultaneous, ray),
    )
    assert not failed.passed
    assert failed.reason == "base_program_not_continuously_certified"


def test_interior_grazing_tie_is_not_mistaken_for_node_margin_evidence() -> None:
    # L0=0 and L1=z+t^2 on z in [0,1].  The word is (0,) on both
    # sides, but the inactive competitor ties at (t,z)=(0,0).  Sparse node
    # checks away from zero could look safe; the closed-domain strict proof
    # must fail because the continuous predicate has an exact interior root.
    sites = AffineKineticPowerSites(
        positions0=torch.tensor([[0, 0, 0], [-0.5, 0, 0]], dtype=DTYPE),
        velocities=torch.zeros((2, 3), dtype=DTYPE),
        weight_coefficients=torch.tensor([[0, 0, 0], [0.25, 0, -1]], dtype=DTYPE),
    )
    ray = _static_x_ray()
    program = compile_active_kinetic_owner_charts(
        sites,
        ray,
        t_min=-1,
        t_max=1,
        near=0,
        far=1,
    )
    assert program.passed
    assert tuple(chart.owner_word for chart in program.charts) == ((0,),)
    assert program.inactive_event_guards

    certificate = certify_event_free_kinetic_geometry_trust_region(
        sites,
        ray,
        program,
        make_kinetic_geometry_update_direction(sites, ray),
    )

    assert not certificate.passed
    assert certificate.certified_step_radius == 0
    assert certificate.reason == "base_strict_continuous_owner_word_not_certified"
    assert certificate.limiting_predicate_kind == "near_owner_gap"


def test_requested_radius_monotonicity_and_binary64_candidate_adapter() -> None:
    sites = _sites_from_static_ray_lines([0, -2, 0], [0, 1, Fraction(1, 8)])
    ray = _static_x_ray()
    program = _compile(sites)
    candidate_weights = sites.weight_coefficients.clone()
    candidate_weights[2, 0] += 1.0 / 16.0
    candidate = AffineKineticPowerSites(
        positions0=sites.positions0,
        velocities=sites.velocities,
        weight_coefficients=candidate_weights,
    )
    delta_weights = torch.zeros_like(sites.weight_coefficients)
    delta_weights[2, 0] = 1.0 / 16.0
    direction = make_kinetic_geometry_update_direction(
        sites,
        ray,
        delta_weight_coefficients=delta_weights,
    )

    full = certify_event_free_kinetic_geometry_trust_region(
        sites,
        ray,
        program,
        direction,
        requested_step_radius=1,
    )
    half = certify_event_free_kinetic_geometry_trust_region(
        sites,
        ray,
        program,
        direction,
        requested_step_radius=Fraction(1, 2),
    )
    endpoint = certify_event_free_binary64_geometry_candidate(
        sites,
        ray,
        program,
        candidate,
        ray,
    )

    assert full.passed and full.certified_step_radius == 1
    assert half.passed and half.certified_step_radius == Fraction(1, 2)
    assert endpoint.requested_radius_certified
    assert _word_at(candidate, ray) == (0, 1)
    assert all(item.accepts(Fraction(1)) for item in full.predicate_certificates)

    unsafe_weights = sites.weight_coefficients.clone()
    unsafe_weights[2, 0] += 1.0 / 4.0
    unsafe_candidate = AffineKineticPowerSites(
        positions0=sites.positions0,
        velocities=sites.velocities,
        weight_coefficients=unsafe_weights,
    )
    unsafe = certify_event_free_binary64_geometry_candidate(
        sites,
        ray,
        program,
        unsafe_candidate,
        ray,
    )
    assert unsafe.passed
    assert not unsafe.requested_radius_certified
    assert unsafe.recompile_required
    assert _word_at(unsafe_candidate, ray) != (0, 1)


def test_ray_collapse_direction_limits_the_radius() -> None:
    sites = _sites_from_static_ray_lines([0], [0])
    ray = _static_x_ray()
    program = _compile(sites)
    delta_ray = torch.zeros(12, dtype=DTYPE)
    delta_ray[6] = -1.0
    direction = make_kinetic_geometry_update_direction(
        sites,
        ray,
        delta_ray_coefficients=delta_ray,
    )

    certificate = certify_event_free_kinetic_geometry_trust_region(
        sites,
        ray,
        program,
        direction,
        requested_step_radius=2,
    )

    assert certificate.passed
    assert 0 < certificate.certified_step_radius < 1
    assert certificate.limiting_predicate_kind == "ray_speed_squared"
