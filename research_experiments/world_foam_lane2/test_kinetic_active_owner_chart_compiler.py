from __future__ import annotations

import inspect
import math
import random
from fractions import Fraction

import torch
from kinetic_active_owner_chart_compiler import compile_active_kinetic_owner_charts
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
    """Realize exact dyadic ``s_i(t) z+b_i(t)`` lines on the x ray."""

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


def _compress_words(words: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    result: list[tuple[int, ...]] = []
    for word in words:
        if not result or result[-1] != word:
            result.append(word)
    return tuple(result)


def test_pair_events_emit_three_right_continuous_active_charts() -> None:
    sites = _sites_from_ray_lines(
        slopes=[(0, 0), (-2, 0)],
        intercepts=[(0, 0, 0), (1, -1, 0)],
    )

    program = compile_active_kinetic_owner_charts(
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
    assert [chart.interval_notation for chart in program.charts] == [
        "[left,right)",
        "[left,right)",
        "[left,right]",
    ]
    assert program.continuous_time_coverage
    assert program.owner_identity_certified


def test_inactive_site_birth_at_active_vertex_is_not_missed() -> None:
    # L0=z, L1=-z, L2=-t. Site 2 is absent for t<0 and enters through
    # the existing 0/1 cut at t=0; endpoint-only monitoring would miss it.
    sites = _sites_from_ray_lines(
        slopes=[(1, 0), (-1, 0), (0, 0)],
        intercepts=[(0, 0, 0), (0, 0, 0), (0, -1, 0)],
    )

    program = compile_active_kinetic_owner_charts(
        sites,
        _static_x_ray(),
        t_min=Fraction(-1, 2),
        t_max=Fraction(1, 2),
        near=-1,
        far=1,
    )

    assert program.passed
    assert tuple(chart.owner_word for chart in program.charts) == ((0, 1), (0, 2, 1))
    assert len(program.active_event_guards) == 1
    assert tuple(source.kind for source in program.active_event_guards[0].sources) == ("active_cut_competitor",)
    assert program.active_event_guards[0].lower_bound == 0


def test_grazing_endpoint_root_is_proved_inactive_and_merged() -> None:
    # L0=0, L1=z+t^2 on z in [0,1]. The near predicate has a double
    # root, but site 1 never owns a positive-length interval.
    sites = _sites_from_ray_lines(
        slopes=[(0, 0), (1, 0)],
        intercepts=[(0, 0, 0), (0, 0, 1)],
    )

    program = compile_active_kinetic_owner_charts(
        sites,
        _static_x_ray(),
        t_min=-1,
        t_max=1,
        near=0,
        far=1,
    )

    assert program.passed
    assert tuple(chart.owner_word for chart in program.charts) == ((0,),)
    assert program.active_event_guards == ()
    assert len(program.inactive_event_guards) == 1
    guard = program.inactive_event_guards[0]
    assert guard.lower_bound == guard.upper_bound == 0
    assert guard.source_multiplicities == (2,)
    assert tuple(source.kind for source in guard.sources) == ("pair_near",)
    assert program.charts[0].filtered_inactive_guards == (guard,)


def test_irrational_event_retains_exact_polynomial_and_isolating_interval() -> None:
    sites = _sites_from_ray_lines(
        slopes=[(0, 0), (-2, 0)],
        intercepts=[(0, 0, 0), (1, 0, -2)],
    )

    program = compile_active_kinetic_owner_charts(
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


def test_active_cut_denominator_is_a_separate_quadratic_guard() -> None:
    # L0=0, L1=2(t+1)z-1. The pair is active after its far crossing at
    # t=-1/2. Its A root at t=-1 is retained only as an inactive analytic
    # guard and is never multiplied into the physical event polynomial.
    sites = _sites_from_ray_lines(
        slopes=[(0, 0), (2, 2)],
        intercepts=[(0, 0, 0), (-1, 0, 0)],
    )

    program = compile_active_kinetic_owner_charts(
        sites,
        _static_x_ray(),
        t_min=-2,
        t_max=1,
        near=0,
        far=1,
    )

    assert program.passed
    assert tuple(chart.owner_word for chart in program.charts) == ((1,), (1, 0))
    denominator = tuple(
        guard
        for guard in program.inactive_event_guards
        if any(source.kind == "active_cut_denominator" for source in guard.sources)
    )
    assert len(denominator) == 1
    assert denominator[0].lower_bound <= -1 <= denominator[0].upper_bound
    source = denominator[0].sources[0]
    assert source.analytic_guard_only
    assert source.polynomial.degree <= 2
    assert all(
        source.polynomial.degree <= 4
        for guard in (*program.active_event_guards, *program.inactive_event_guards)
        for source in guard.sources
        if source.kind == "active_cut_competitor"
    )


def test_close_distinct_roots_are_refined_not_merged() -> None:
    epsilon = Fraction(1, 1024)
    sites = _sites_from_ray_lines(
        slopes=[(0, 0), (1, 0), (2, 0)],
        intercepts=[(0, 0, 0), (0, 1, 0), (-epsilon, 1, 0)],
    )

    program = compile_active_kinetic_owner_charts(
        sites,
        _static_x_ray(),
        t_min=Fraction(-1, 8),
        t_max=Fraction(1, 8),
        near=0,
        far=1,
        max_root_interval_width=Fraction(1, 64),
    )

    assert program.passed
    guards = tuple(
        sorted(
            (*program.active_event_guards, *program.inactive_event_guards),
            key=lambda guard: guard.lower_bound,
        )
    )
    assert len(guards) == 3
    assert all(left.upper_bound < right.lower_bound for left, right in zip(guards, guards[1:], strict=False))
    assert program.work.algebraic_root_refinement_count > 0


def test_full_fiber_and_ray_collapse_fail_closed() -> None:
    full_fiber_sites = _sites_from_ray_lines(
        slopes=[(0, 0), (0, 2)],
        intercepts=[(0, 0, 0), (0, 1, 0)],
    )
    full_fiber = compile_active_kinetic_owner_charts(
        full_fiber_sites,
        _static_x_ray(),
        t_min=-1,
        t_max=1,
        near=-1,
        far=1,
    )
    assert not full_fiber.passed
    assert full_fiber.charts == ()
    assert full_fiber.unresolved_degeneracies[0].kind == "full_fiber_tie"

    # L0=(t-1/4)^2 and L1=0 have the same unique open-cell owner on both
    # sides, but tie over the complete positive-length fiber at t=1/4.  The
    # fixed-time site-id rule would otherwise switch material only at that
    # isolated sample, so this inactive-looking guard must not be merged.
    inactive_word_full_fiber = compile_active_kinetic_owner_charts(
        _sites_from_ray_lines(
            slopes=[(0, 0), (0, 0)],
            intercepts=[(Fraction(1, 16), Fraction(-1, 2), 1), (0, 0, 0)],
        ),
        _static_x_ray(),
        t_min=0,
        t_max=1,
        near=-1,
        far=1,
    )
    assert not inactive_word_full_fiber.passed
    assert inactive_word_full_fiber.charts == ()
    degeneracy = inactive_word_full_fiber.unresolved_degeneracies[0]
    assert degeneracy.kind == "full_fiber_tie"
    assert degeneracy.lower_bound == degeneracy.upper_bound == Fraction(1, 4)

    zero_ray = torch.tensor(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        dtype=DTYPE,
    )
    collapsed = compile_active_kinetic_owner_charts(
        _sites_from_ray_lines(
            slopes=[(0, 0), (1, 0)],
            intercepts=[(0, 0, 0), (0, 0, 0)],
        ),
        zero_ray,
        t_min=-1,
        t_max=1,
        near=-1,
        far=1,
    )
    assert not collapsed.passed
    assert collapsed.unresolved_degeneracies[0].kind == "zero_ray_direction"
    assert collapsed.unresolved_degeneracies[0].lower_bound == 0


def test_persistent_active_concurrence_and_simultaneous_change_fail_closed() -> None:
    persistent = compile_active_kinetic_owner_charts(
        _sites_from_ray_lines(
            slopes=[(1, 0), (0, 0), (-1, 0)],
            intercepts=[(0, 0, 0), (0, 0, 0), (0, 0, 0)],
        ),
        _static_x_ray(),
        t_min=-1,
        t_max=1,
        near=-1,
        far=1,
    )
    assert not persistent.passed
    assert persistent.unresolved_degeneracies[0].kind == "persistent_active_cut_concurrence"

    simultaneous = compile_active_kinetic_owner_charts(
        _sites_from_ray_lines(
            slopes=[(0, 0), (-2, 0), (-4, 0)],
            intercepts=[(0, 0, 0), (0, -1, 0), (0, 1, 0)],
        ),
        _static_x_ray(),
        t_min=-1,
        t_max=1,
        near=0,
        far=1,
    )
    assert not simultaneous.passed
    assert simultaneous.unresolved_degeneracies[0].kind == "ambiguous_simultaneous_active_event"


def test_dominated_world_has_linear_candidate_attempts_not_all_triples() -> None:
    site_count = 24
    sites = _sites_from_ray_lines(
        slopes=[(0, 0)] * site_count,
        intercepts=[
            (0, 0, 0),
            *[(10 + site_id, 1, 0) for site_id in range(1, site_count)],
        ],
    )

    program = compile_active_kinetic_owner_charts(
        sites,
        _static_x_ray(),
        t_min=0,
        t_max=1,
        near=-1,
        far=1,
    )

    assert program.passed
    assert tuple(chart.owner_word for chart in program.charts) == ((0,),)
    assert program.work.certificate_round_count == 2
    # Sources depend only on the owner word, so the closure round reuses them.
    assert program.work.candidate_source_attempt_count == 3 * (site_count - 1)
    assert program.work.unique_source_word_count == 1
    assert program.work.unique_candidate_source_count == 3 * (site_count - 1)
    assert program.work.unique_pair_difference_count == site_count - 1
    assert program.work.unique_candidate_source_count < math.comb(site_count, 3)
    assert program.work.per_witness_candidate_bound_verified
    assert not program.work.exhaustive_triple_enumeration_used
    assert "O(U*S*R_max)" in program.work.structural_complexity
    assert "neighbor" in program.work.limitation


def test_inactive_root_cells_are_separate_from_unique_word_source_work() -> None:
    site_count = 8
    tangencies = [Fraction(index, site_count) for index in range(1, site_count)]
    sites = _sites_from_ray_lines(
        slopes=[(0, 0), *[(1, 0) for _ in tangencies]],
        intercepts=[
            (0, 0, 0),
            *((time * time, -2 * time, 1) for time in tangencies),
        ],
    )

    program = compile_active_kinetic_owner_charts(
        sites,
        _static_x_ray(),
        t_min=0,
        t_max=1,
        near=0,
        far=1,
    )

    assert program.passed
    assert tuple(chart.owner_word for chart in program.charts) == ((0,),)
    assert len(program.inactive_event_guards) == site_count - 1
    assert program.work.unique_source_word_count == 1
    assert program.work.candidate_source_attempt_count == 3 * (site_count - 1)
    assert program.work.root_complement_witness_count == site_count
    assert program.work.witness_word_discovery_count == site_count + 1
    assert "W cumulative root-complement" in program.work.structural_complexity


def test_randomized_words_match_exhaustive_compiler_and_independent_oracle() -> None:
    for seed in (0, 1, 2):
        generator = random.Random(seed)
        slopes = [(generator.randint(-3, 3), generator.randint(-1, 1)) for _ in range(3)]
        intercepts = [
            (
                generator.randint(-4, 4),
                generator.randint(-2, 2),
                generator.randint(-1, 1),
            )
            for _ in range(3)
        ]
        sites = _sites_from_ray_lines(slopes, intercepts)
        arguments = dict(
            t_min=Fraction(-7, 8),
            t_max=Fraction(9, 8),
            near=Fraction(-3, 2),
            far=Fraction(5, 4),
        )
        width = Fraction(1, 1 << 24)

        active = compile_active_kinetic_owner_charts(
            sites,
            _static_x_ray(),
            max_root_interval_width=width,
            **arguments,
        )
        exhaustive = compile_exact_kinetic_owner_charts(
            sites,
            _static_x_ray(),
            max_root_interval_width=width,
            **arguments,
        )
        oracle = build_kinetic_owner_chart_oracle(
            sites,
            _static_x_ray(),
            max_interval_width=width,
            **arguments,
        )

        assert active.passed and exhaustive.passed
        active_words = tuple(chart.owner_word for chart in active.charts)
        exhaustive_words = tuple(chart.owner_word for chart in exhaustive.charts)
        oracle_words = _compress_words(oracle.owner_word_sequence)
        assert active_words == exhaustive_words == oracle_words


def test_api_and_work_report_have_no_requested_frame_axis() -> None:
    parameters = inspect.signature(compile_active_kinetic_owner_charts).parameters
    assert "frame_count" not in parameters
    assert "requested_frame_count" not in parameters
    assert "sample_count" not in parameters

    program = compile_active_kinetic_owner_charts(
        _sites_from_ray_lines(
            slopes=[(0, 0), (-1, 0)],
            intercepts=[(0, 0, 0), (0, 0, 0)],
        ),
        _static_x_ray(),
        t_min=-1,
        t_max=1,
        near=-1,
        far=1,
    )
    assert program.passed
    assert not program.requested_frame_sampling_used
    assert not program.work.requested_frame_sampling_used
