from __future__ import annotations

import inspect
from fractions import Fraction

import torch
from kinetic_owner_chart_compiler import compile_exact_kinetic_owner_charts
from kinetic_owner_chart_oracle import build_kinetic_owner_chart_oracle
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
    """Realize ``m_i(t)z+b_i(t)`` exactly on the static x-axis ray."""

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
        positions0=torch.tensor(
            [[float(value) for value in row] for row in positions],
            dtype=DTYPE,
        ),
        velocities=torch.tensor(
            [[float(value) for value in row] for row in velocities],
            dtype=DTYPE,
        ),
        weight_coefficients=torch.tensor(
            [[float(value) for value in row] for row in weights],
            dtype=DTYPE,
        ),
    )


def test_pair_boundary_events_compile_three_right_continuous_charts() -> None:
    # Static slopes 0,-2 and intercepts 0,1-t make the cut
    # z=(1-t)/2. It enters at far=1 when t=-1 and exits at near=0
    # when t=1 without a full-fiber tie.
    sites = _sites_from_ray_lines(
        slopes=[(0, 0), (-2, 0)],
        intercepts=[(0, 0, 0), (1, -1, 0)],
    )

    program = compile_exact_kinetic_owner_charts(
        sites,
        _static_x_ray(),
        t_min=-2,
        t_max=2,
        near=0,
        far=1,
    )

    assert program.passed
    assert tuple(chart.owner_word for chart in program.charts) == ((0,), (0, 1), (1,))
    assert tuple(guard.lower_bound for guard in program.active_event_guards) == (
        Fraction(-1),
        Fraction(1),
    )
    assert all(guard.exact for guard in program.active_event_guards)
    assert [chart.interval_notation for chart in program.charts] == [
        "[left,right)",
        "[left,right)",
        "[left,right]",
    ]
    assert all(chart.left_closed for chart in program.charts)
    assert not program.completeness.requested_frame_sampling_used
    assert program.continuous_time_coverage


def test_active_triple_event_births_a_middle_owner_run() -> None:
    # L0=0, L1=-2z+1, L2=-4z+(2-t).  At t=0 all three meet at
    # z=1/2. Site 1 has a positive run only for t<0.
    sites = _sites_from_ray_lines(
        slopes=[(0, 0), (-2, 0), (-4, 0)],
        intercepts=[(0, 0, 0), (1, 0, 0), (2, -1, 0)],
    )

    program = compile_exact_kinetic_owner_charts(
        sites,
        _static_x_ray(),
        t_min=Fraction(-1, 2),
        t_max=Fraction(1, 2),
        near=0,
        far=1,
    )

    assert program.passed
    assert tuple(chart.owner_word for chart in program.charts) == ((0, 1, 2), (0, 2))
    assert len(program.active_event_guards) == 1
    guard = program.active_event_guards[0]
    assert guard.exact and guard.lower_bound == 0
    assert tuple(source.kind for source in guard.sources) == ("triple_concurrence",)
    assert "increasing-time chart owns" in guard.dispatch_rule
    assert all(
        certificate.all_site_owner_identity_passed
        for chart in program.charts
        for certificate in chart.witness_certificates
    )


def test_irrational_boundary_event_retains_polynomial_and_isolating_interval() -> None:
    # L0=0 and L1=-2z+(1-2t^2). The near crossing is t=1/sqrt(2).
    sites = _sites_from_ray_lines(
        slopes=[(0, 0), (-2, 0)],
        intercepts=[(0, 0, 0), (1, 0, -2)],
    )

    program = compile_exact_kinetic_owner_charts(
        sites,
        _static_x_ray(),
        t_min=0,
        t_max=1,
        near=0,
        far=2,
    )

    assert program.passed
    assert tuple(chart.owner_word for chart in program.charts) == ((0, 1), (1,))
    assert len(program.active_event_guards) == 1
    guard = program.active_event_guards[0]
    expected = 2**-0.5
    assert not guard.exact
    assert float(guard.lower_bound) < expected < float(guard.upper_bound)
    assert guard.canonical_polynomial.coefficients == (
        Fraction(-1),
        Fraction(0),
        Fraction(2),
    )
    assert guard.distinct_neighbor_roots_certified


def test_denominator_root_is_filtered_as_an_analytic_guard_only() -> None:
    # L0=0; L1=2(t+1)z-100. Site 1 owns the complete finite segment
    # while the pair denominator crosses zero at t=-1.
    sites = _sites_from_ray_lines(
        slopes=[(0, 0), (2, 2)],
        intercepts=[(0, 0, 0), (-100, 0, 0)],
    )

    program = compile_exact_kinetic_owner_charts(
        sites,
        _static_x_ray(),
        t_min=-2,
        t_max=0,
        near=0,
        far=1,
    )

    assert program.passed
    assert tuple(chart.owner_word for chart in program.charts) == ((1,),)
    assert program.active_event_guards == ()
    assert len(program.inactive_event_guards) == 1
    guard = program.inactive_event_guards[0]
    assert guard.lower_bound == guard.upper_bound == -1
    assert guard.sources[0].kind == "pair_denominator"
    assert guard.sources[0].analytic_guard_only
    assert program.charts[0].filtered_inactive_guards == (guard,)
    assert len(program.charts[0].witness_certificates) == 2


def test_compiler_owner_cells_match_independent_exhaustive_oracle() -> None:
    sites = _sites_from_ray_lines(
        slopes=[(1, 0), (0, 0), (-1, 0), (0, 0)],
        intercepts=[(0, 0, 0), (0, -1, 0), (0, 0, 0), (5, 0, 0)],
    )
    compiler = compile_exact_kinetic_owner_charts(
        sites,
        _static_x_ray(),
        t_min=Fraction(-1, 2),
        t_max=Fraction(1, 2),
        near=-1,
        far=1,
    )
    oracle = build_kinetic_owner_chart_oracle(
        sites,
        _static_x_ray(),
        t_min=Fraction(-1, 2),
        t_max=Fraction(1, 2),
        near=-1,
        far=1,
    )

    assert compiler.passed
    compiler_cells = tuple(
        certificate.owners for chart in compiler.charts for certificate in chart.witness_certificates
    )
    # The oracle retains inactive raw events, so compare its root-complement
    # sequence rather than expecting identical chart counts.
    assert compiler_cells == oracle.owner_word_sequence
    assert tuple(chart.owner_word for chart in compiler.charts) == ((0, 2), (0, 1, 2))


def test_close_distinct_roots_are_ordered_not_merged_as_one_seam() -> None:
    epsilon = Fraction(1, 1024)
    sites = _sites_from_ray_lines(
        slopes=[(0, 0), (1, 0), (2, 0)],
        intercepts=[(0, 0, 0), (0, 1, 0), (-epsilon, 1, 0)],
    )

    program = compile_exact_kinetic_owner_charts(
        sites,
        _static_x_ray(),
        t_min=Fraction(-1, 8),
        t_max=Fraction(1, 8),
        near=0,
        far=1,
        # Deliberately coarser than the separation of the three roots.
        max_root_interval_width=Fraction(1, 64),
    )

    assert program.passed
    all_guards = sorted(
        (*program.active_event_guards, *program.inactive_event_guards),
        key=lambda guard: guard.lower_bound,
    )
    assert len(all_guards) == 3
    assert all(left.upper_bound < right.lower_bound for left, right in zip(all_guards, all_guards[1:], strict=False))
    assert all(guard.simultaneous_source_count == 1 for guard in all_guards)


def test_full_fiber_tie_fails_closed_without_partial_charts() -> None:
    sites = _sites_from_ray_lines(
        slopes=[(0, 0), (0, 2)],
        # At t=0 both A and B vanish, producing a complete-fiber tie.
        intercepts=[(0, 0, 0), (0, 1, 0)],
    )

    program = compile_exact_kinetic_owner_charts(
        sites,
        _static_x_ray(),
        t_min=-1,
        t_max=1,
        near=-1,
        far=1,
    )

    assert not program.passed
    assert program.charts == ()
    assert len(program.unresolved_degeneracies) == 1
    assert program.unresolved_degeneracies[0].kind == "full_fiber_tie"
    assert not program.continuous_time_coverage


def test_continuous_ray_direction_zero_is_an_unresolved_degeneracy() -> None:
    sites = _sites_from_ray_lines(
        slopes=[(0, 0), (1, 0)],
        intercepts=[(0, 0, 0), (0, 0, 0)],
    )
    ray = torch.tensor(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        dtype=DTYPE,
    )

    program = compile_exact_kinetic_owner_charts(
        sites,
        ray,
        t_min=-1,
        t_max=1,
        near=-1,
        far=1,
    )

    assert not program.passed
    assert program.unresolved_degeneracies[0].kind == "zero_ray_direction"
    assert program.unresolved_degeneracies[0].lower_bound == 0
    assert program.unresolved_degeneracies[0].upper_bound == 0


def test_simultaneous_active_event_fails_closed() -> None:
    # All three lines meet near=0 at t=0, so three pair-near predicates
    # and the triple predicate share an owner-changing seam.
    sites = _sites_from_ray_lines(
        slopes=[(0, 0), (-2, 0), (-4, 0)],
        intercepts=[(0, 0, 0), (0, -1, 0), (0, 1, 0)],
    )

    program = compile_exact_kinetic_owner_charts(
        sites,
        _static_x_ray(),
        t_min=-1,
        t_max=1,
        near=0,
        far=1,
    )

    assert not program.passed
    assert program.charts == ()
    assert program.unresolved_degeneracies[0].kind == "ambiguous_simultaneous_active_event"


def test_persistent_triple_concurrence_is_not_sampled_past() -> None:
    sites = _sites_from_ray_lines(
        slopes=[(0, 0), (1, 0), (2, 0)],
        intercepts=[(0, 0, 0), (0, 0, 1), (0, 0, 2)],
    )

    program = compile_exact_kinetic_owner_charts(
        sites,
        _static_x_ray(),
        t_min=-1,
        t_max=1,
        near=0,
        far=1,
    )

    assert not program.passed
    assert program.unresolved_degeneracies[0].kind == "persistent_triple_concurrence"


def test_api_and_report_have_no_requested_frame_count() -> None:
    parameters = inspect.signature(compile_exact_kinetic_owner_charts).parameters
    assert "frame_count" not in parameters
    assert "requested_frame_count" not in parameters
    assert "sample_count" not in parameters

    sites = _sites_from_ray_lines(
        slopes=[(0, 0), (-1, 0)],
        intercepts=[(0, 0, 0), (0, 0, 0)],
    )
    program = compile_exact_kinetic_owner_charts(
        sites,
        _static_x_ray(),
        t_min=-1,
        t_max=1,
        near=-1,
        far=1,
    )
    assert program.passed
    assert program.requested_frame_sampling_used is False
