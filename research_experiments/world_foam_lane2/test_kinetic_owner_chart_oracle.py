from __future__ import annotations

import random
from fractions import Fraction

import pytest
import torch
from kinetic_owner_chart_oracle import (
    KineticOwnerOracleDegeneracyError,
    PersistentKineticOwnerTieError,
    brute_force_owner_word_at_rational_time,
    build_kinetic_owner_chart_oracle,
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


def _sites_from_ray_lines(
    slopes: list[tuple[int | Fraction, int | Fraction]],
    intercepts: list[tuple[int | Fraction, int | Fraction, int | Fraction]],
) -> AffineKineticPowerSites:
    """Realize ``m_i(t) z + b_i(t)`` on the static x-axis ray exactly."""

    if len(slopes) != len(intercepts):
        raise ValueError("one slope and intercept polynomial are required per site")
    positions = []
    velocities = []
    weights = []
    for (slope0, slope1), (bias0, bias1, bias2) in zip(
        slopes,
        intercepts,
        strict=True,
    ):
        p0 = -Fraction(slope0) / 2
        velocity = -Fraction(slope1) / 2
        # ||p(t)||^2 - w(t) = b(t).
        weights.append(
            (
                p0 * p0 - Fraction(bias0),
                2 * p0 * velocity - Fraction(bias1),
                velocity * velocity - Fraction(bias2),
            )
        )
        positions.append((p0, Fraction(0), Fraction(0)))
        velocities.append((velocity, Fraction(0), Fraction(0)))
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


def _event_at(report, time: Fraction | int):
    point = Fraction(time)
    matches = [event for event in report.events if event.root.lower_bound <= point <= event.root.upper_bound]
    assert len(matches) == 1
    return matches[0]


def test_rotating_boundary_enters_and_leaves_the_depth_window() -> None:
    # p1(t)-p0(t)=(2,2t,0), so the spatial face normal rotates.  On this ray
    # its cut is z=-4t/(4+t), entering far at -4/5 and leaving near at 4/3.
    sites = AffineKineticPowerSites(
        positions0=torch.tensor([[-1, 0, 0], [1, 0, 0]], dtype=DTYPE),
        velocities=torch.tensor([[0, -1, 0], [0, 1, 0]], dtype=DTYPE),
        weight_coefficients=torch.zeros((2, 1), dtype=DTYPE),
    )
    ray = torch.tensor(
        [0, 1, 0, 0, 0, 0, 1, Fraction(1, 4), 0, 0, 0, 0],
        dtype=DTYPE,
    )

    report = build_kinetic_owner_chart_oracle(
        sites,
        ray,
        t_min=-1,
        t_max=2,
        near=-1,
        far=1,
    )

    expected_roots = (Fraction(-4, 5), Fraction(4, 3))
    assert all(
        event.root.lower_bound <= expected <= event.root.upper_bound
        for event, expected in zip(report.events, expected_roots, strict=True)
    )
    assert report.owner_word_sequence == ((0,), (0, 1), (1,))
    assert all(event.changes_owner_word for event in report.events)
    assert {source.kind for event in report.events for source in event.sources} == {"pair_near", "pair_far"}
    assert report.active_owner_filter_used is False


def test_third_site_undercut_preserves_inactive_pair_candidates() -> None:
    # L0=z+t and L1=-z-t exchange within the depth window, but L2=-10
    # undercuts both everywhere.  The raw roots must remain in the oracle even
    # though no owner chart needs them.
    sites = _sites_from_ray_lines(
        slopes=[(1, 0), (-1, 0), (0, 0)],
        intercepts=[(0, 1, 0), (0, -1, 0), (-10, 0, 0)],
    )

    report = build_kinetic_owner_chart_oracle(
        sites,
        _static_x_ray(),
        t_min=-2,
        t_max=2,
        near=-1,
        far=1,
    )

    assert len(report.events) == 2
    assert all(
        event.root.lower_bound <= expected <= event.root.upper_bound
        for event, expected in zip(report.events, (-1, 1), strict=True)
    )
    assert report.owner_word_sequence == ((2,), (2,), (2,))
    assert all(event.changes_owner_word is False for event in report.events)
    assert all(any(source.site_ids == (0, 1) for source in event.sources) for event in report.events)


def test_inactive_concurrence_is_visible_beneath_fourth_site() -> None:
    # L0=z, L1=0, L2=-z+t concur at t=0. L3=-10 owns the whole
    # ray, so the concurrence is intentionally inactive.
    sites = _sites_from_ray_lines(
        slopes=[(1, 0), (0, 0), (-1, 0), (0, 0)],
        intercepts=[(0, 0, 0), (0, 0, 0), (0, 1, 0), (-10, 0, 0)],
    )

    report = build_kinetic_owner_chart_oracle(
        sites,
        _static_x_ray(),
        t_min=-1,
        t_max=1,
        near=-1,
        far=1,
    )
    event = _event_at(report, 0)

    assert len(report.predicates) == 3 * 6 + 4
    assert sum(predicate.kind == "pair_denominator" for predicate in report.predicates) == 6
    assert sum(predicate.kind == "pair_near" for predicate in report.predicates) == 6
    assert sum(predicate.kind == "pair_far" for predicate in report.predicates) == 6
    assert sum(predicate.kind == "triple_concurrence" for predicate in report.predicates) == 4
    assert any(source.kind == "triple_concurrence" and source.site_ids == (0, 1, 2) for source in event.sources)
    assert event.changes_owner_word is False
    assert all(sample.word.owners == (3,) for sample in report.interval_samples)


def test_zero_denominator_cross_product_root_is_an_inactive_raw_candidate() -> None:
    # A01=A12=2t and the triple cross product also vanishes at t=0.
    # All finite-depth power values are distinct there; the raw candidate is a
    # denominator-zero artifact, not an owner event.
    sites = AffineKineticPowerSites(
        positions0=torch.tensor([[0, 0, 0], [0, 1, 0], [0, 3, 0]], dtype=DTYPE),
        velocities=torch.tensor([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=DTYPE),
        weight_coefficients=torch.zeros((3, 1), dtype=DTYPE),
    )

    report = build_kinetic_owner_chart_oracle(
        sites,
        _static_x_ray(),
        t_min=Fraction(-1, 4),
        t_max=Fraction(1, 4),
        near=-1,
        far=1,
    )
    event = _event_at(report, 0)

    assert {source.kind for source in event.sources} >= {
        "pair_denominator",
        "triple_concurrence",
    }
    assert event.changes_owner_word is False
    assert event.left_word is not None and event.left_word.owners == (0,)
    assert event.right_word is not None and event.right_word.owners == (0,)
    assert event.exact_seam_error is None


def test_simultaneous_repeated_roots_are_merged_without_active_filtering() -> None:
    # L0=0, L1=z+t^2, L2=2z+2t^2.  Three near predicates have a
    # double root at t=0; the triple concurrence is persistently zero.
    sites = _sites_from_ray_lines(
        slopes=[(0, 0), (1, 0), (2, 0)],
        intercepts=[(0, 0, 0), (0, 0, 1), (0, 0, 2)],
    )

    report = build_kinetic_owner_chart_oracle(
        sites,
        _static_x_ray(),
        t_min=-1,
        t_max=1,
        near=0,
        far=1,
    )
    event = _event_at(report, 0)

    assert len(report.events) == 1
    assert event.simultaneous
    assert event.repeated
    assert event.root.multiplicity == 6
    assert len(event.sources) == 3
    assert event.source_root_relation == "same_algebraic_root_gcd_certified"
    assert {source.multiplicity for source in event.sources} == {2}
    assert all(source.kind == "pair_near" for source in event.sources)
    assert len(report.persistent_predicate_indices) == 1
    assert report.predicates[report.persistent_predicate_indices[0]].kind == "triple_concurrence"
    assert report.owner_word_sequence == ((0,), (0,))


def test_close_distinct_roots_are_refined_not_merged_as_simultaneous() -> None:
    # Raw single-predicate isolators may stop at width 1/64, much wider than
    # the 1/1024 event spacing here.  The global product isolation must still
    # separate the roots -eps, 0, +eps rather than clustering overlapping
    # boxes and losing the two tiny intervening chart components.
    epsilon = Fraction(1, 1024)
    sites = _sites_from_ray_lines(
        slopes=[(0, 0), (1, 0), (2, 0)],
        intercepts=[(0, 0, 0), (0, 1, 0), (-epsilon, 1, 0)],
    )

    report = build_kinetic_owner_chart_oracle(
        sites,
        _static_x_ray(),
        t_min=Fraction(-1, 8),
        t_max=Fraction(1, 8),
        near=0,
        far=1,
        max_interval_width=Fraction(1, 64),
    )

    expected = (-epsilon, Fraction(0), epsilon)
    assert len(report.events) == 3
    assert all(
        event.root.lower_bound <= value <= event.root.upper_bound
        for event, value in zip(report.events, expected, strict=True)
    )
    assert all(
        left.root.upper_bound < right.root.lower_bound
        for left, right in zip(report.events, report.events[1:], strict=False)
    )
    assert len(report.interval_samples) == 4
    assert report.distinct_root_isolation_method == "global_product_square_free_sturm_v1"
    assert report.events[0].source_root_relation == "single_source"
    assert report.events[1].source_root_relation == "single_source"
    assert report.events[2].source_root_relation == "single_source"


def test_zero_length_middle_run_birth_has_exact_deleted_seam_word() -> None:
    # L0=z, L1=-t, L2=-z.  Site 1 owns no positive-length interval for
    # t<0, is a zero-length run at t=0, and owns a positive interval for t>0.
    sites = _sites_from_ray_lines(
        slopes=[(1, 0), (0, 0), (-1, 0)],
        intercepts=[(0, 0, 0), (0, -1, 0), (0, 0, 0)],
    )

    report = build_kinetic_owner_chart_oracle(
        sites,
        _static_x_ray(),
        t_min=Fraction(-1, 2),
        t_max=Fraction(1, 2),
        near=-1,
        far=1,
    )
    event = _event_at(report, 0)

    assert report.owner_word_sequence == ((0, 2), (0, 1, 2))
    assert event.changes_owner_word is True
    assert event.exact_seam_word is not None
    assert event.exact_seam_word.owners == (0, 2)
    assert event.exact_seam_word.transition_depths == (Fraction(0),)
    assert any(source.kind == "triple_concurrence" for source in event.sources)


def test_persistent_full_fiber_tie_fails_before_sampling() -> None:
    sites = _sites_from_ray_lines(
        slopes=[(0, 0), (0, 0)],
        intercepts=[(0, 0, 0), (0, 0, 0)],
    )

    with pytest.raises(PersistentKineticOwnerTieError, match="persistent full-fiber"):
        build_kinetic_owner_chart_oracle(
            sites,
            _static_x_ray(),
            t_min=-1,
            t_max=1,
            near=-1,
            far=1,
        )


def test_ray_direction_zero_is_rejected_as_a_missing_event_family() -> None:
    sites = _sites_from_ray_lines(
        slopes=[(0, 0), (1, 0)],
        intercepts=[(0, 0, 0), (0, 0, 0)],
    )
    ray = torch.tensor(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        dtype=DTYPE,
    )

    with pytest.raises(KineticOwnerOracleDegeneracyError, match="pair/triple events are incomplete"):
        build_kinetic_owner_chart_oracle(
            sites,
            ray,
            t_min=-1,
            t_max=1,
            near=-1,
            far=1,
        )


def test_independent_brute_sweep_matches_existing_exact_hull_on_small_worlds() -> None:
    generator = random.Random(43)
    ray = _static_x_ray()
    times = (Fraction(-3, 4), Fraction(-1, 4), Fraction(1, 4), Fraction(3, 4))
    for _ in range(12):
        slopes = [(generator.randint(-4, 4), generator.randint(-2, 2)) for _ in range(4)]
        intercepts = [
            (
                generator.randint(-4, 4),
                generator.randint(-2, 2),
                generator.randint(-1, 1),
            )
            for _ in range(4)
        ]
        sites = _sites_from_ray_lines(slopes, intercepts)
        for time in times:
            try:
                brute = brute_force_owner_word_at_rational_time(
                    sites,
                    ray,
                    time=time,
                    near=-2,
                    far=2,
                )
            except KineticOwnerOracleDegeneracyError:
                # Random integer tables can create an exact full-fiber tie at
                # one chosen sample. That is a valid fail-closed oracle result.
                continue
            hull = discover_kinetic_power_word_at_time(
                sites,
                ray,
                time=time,
                near=-2,
                far=2,
            )
            assert brute.owners == tuple(int(owner) for owner in hull.word.owners.tolist())
            assert brute.transition_depths == hull.transition_depths
