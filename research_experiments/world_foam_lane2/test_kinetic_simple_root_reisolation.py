from __future__ import annotations

import inspect
import random
from fractions import Fraction

import torch
from kinetic_active_owner_chart_compiler import compile_active_kinetic_owner_charts
from kinetic_power_word_compiler import AffineKineticPowerSites
from kinetic_simple_root_reisolation import certify_multichart_simple_root_binary64_candidate

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
    for (slope0, slope1), (bias0, bias1, bias2) in zip(slopes, intercepts, strict=True):
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
        positions0=torch.tensor(positions, dtype=DTYPE),
        velocities=torch.tensor(velocities, dtype=DTYPE),
        weight_coefficients=torch.tensor(weights, dtype=DTYPE),
    )


def _compile(
    sites: AffineKineticPowerSites,
    *,
    t_min: Fraction | int = -2,
    t_max: Fraction | int = 2,
    near: Fraction | int = 0,
    far: Fraction | int = 1,
):
    return compile_active_kinetic_owner_charts(
        sites,
        _static_x_ray(),
        t_min=t_min,
        t_max=t_max,
        near=near,
        far=far,
    )


def _rational_two_event_sites() -> AffineKineticPowerSites:
    # L0=0; L1=-2z+1-t.  The far and near events are t=-1,+1.
    return _sites_from_ray_lines(
        [(0, 0), (-2, 0)],
        [(0, 0, 0), (1, -1, 0)],
    )


def test_rational_roots_move_and_reisolate_with_semantics_preserved() -> None:
    sites = _rational_two_event_sites()
    ray = _static_x_ray()
    program = _compile(sites)
    candidate_weights = sites.weight_coefficients.clone()
    candidate_weights[1, 0] += 0.25
    candidate = AffineKineticPowerSites(
        positions0=sites.positions0,
        velocities=sites.velocities,
        weight_coefficients=candidate_weights,
    )

    certificate = certify_multichart_simple_root_binary64_candidate(
        sites,
        ray,
        program,
        candidate,
        ray,
    )

    assert certificate.requested_radius_certified
    assert certificate.continuous_homotopy_proof
    assert certificate.root_complements_certified
    assert certificate.semantic_reclassification_performed
    assert certificate.base_owner_words == certificate.candidate_owner_words == ((0,), (0, 1), (1,))
    assert certificate.semantic_event_count == 2
    assert tuple(item.source.kind for item in certificate.root_continuations) == ("pair_far", "pair_near")
    assert all(item.semantic_owner_change for item in certificate.root_continuations)
    candidate_midpoints = tuple(
        (item.candidate_root.lower_bound + item.candidate_root.upper_bound) / 2
        for item in certificate.root_continuations
    )
    assert abs(float(candidate_midpoints[0] - Fraction(-5, 4))) < 1.0e-10
    assert abs(float(candidate_midpoints[1] - Fraction(3, 4))) < 1.0e-10
    assert {source.predicate_class for source in certificate.predicate_sources} == {
        "topology_event_candidate",
        "analytic_guard",
        "nonroot_validity_guard",
    }


def test_irrational_root_keeps_polynomial_and_certified_interval() -> None:
    # L1=-2z+1-2t^2 has one active near event at sqrt(1/2).
    sites = _sites_from_ray_lines(
        [(0, 0), (-2, 0)],
        [(0, 0, 0), (1, 0, -2)],
    )
    ray = _static_x_ray()
    program = _compile(sites, t_min=0, t_max=1, far=2)
    candidate_weights = sites.weight_coefficients.clone()
    candidate_weights[1, 0] += 0.125
    candidate = AffineKineticPowerSites(sites.positions0, sites.velocities, candidate_weights)

    certificate = certify_multichart_simple_root_binary64_candidate(
        sites,
        ray,
        program,
        candidate,
        ray,
    )

    assert certificate.passed
    assert len(certificate.root_continuations) == 1
    root = certificate.root_continuations[0]
    assert not root.base_root.exact
    expected = (Fraction(7, 16)) ** Fraction(1, 2)
    assert float(root.candidate_root.lower_bound) < float(expected) < float(root.candidate_root.upper_bound)
    assert root.source.base_polynomial.coefficients == (Fraction(-1), Fraction(0), Fraction(2))


def test_denominator_root_is_reisolated_but_not_counted_as_topology() -> None:
    # L1=2(t+1)z-1.  A=0 at t=-1 is a cut-at-infinity guard;
    # the physical far crossing is t=-1/2.
    sites = _sites_from_ray_lines(
        [(0, 0), (2, 2)],
        [(0, 0, 0), (-1, 0, 0)],
    )
    ray = _static_x_ray()
    program = _compile(sites, t_max=1)

    certificate = certify_multichart_simple_root_binary64_candidate(
        sites,
        ray,
        program,
        sites,
        ray,
    )

    assert certificate.passed
    analytic = tuple(item for item in certificate.root_continuations if item.source.predicate_class == "analytic_guard")
    assert len(analytic) == 1
    assert analytic[0].source.kind == "pair_cut_denominator_guard"
    assert not analytic[0].semantic_owner_change
    assert certificate.semantic_event_count == 1
    # The pair is not an active cut on either side of t=-1, so this analytic
    # root needs no physical or representation seam after reclassification.
    assert certificate.representation_chart_split_count == 0
    assert not analytic[0].representation_chart_split_required


def test_interior_third_site_event_moves_and_remains_semantically_active() -> None:
    # L0=z, L1=-z, L2=-t.  Site 2 enters through the active 0/1 cut.
    sites = _sites_from_ray_lines(
        [(1, 0), (-1, 0), (0, 0)],
        [(0, 0, 0), (0, 0, 0), (0, -1, 0)],
    )
    ray = _static_x_ray()
    program = _compile(
        sites,
        t_min=Fraction(-1, 2),
        t_max=Fraction(1, 2),
        near=-1,
        far=1,
    )
    candidate_weights = sites.weight_coefficients.clone()
    candidate_weights[2, 0] += 0.125
    candidate = AffineKineticPowerSites(sites.positions0, sites.velocities, candidate_weights)

    certificate = certify_multichart_simple_root_binary64_candidate(
        sites,
        ray,
        program,
        candidate,
        ray,
    )

    assert certificate.passed
    assert certificate.candidate_owner_words == ((0, 1), (0, 2, 1))
    assert len(certificate.root_continuations) == 1
    event = certificate.root_continuations[0]
    assert event.source.kind == "active_cut_competitor"
    assert event.source.site_ids == (0, 1, 2)
    assert event.semantic_owner_change
    midpoint = (event.candidate_root.lower_bound + event.candidate_root.upper_bound) / 2
    assert abs(float(midpoint + Fraction(1, 8))) < 1.0e-10


def test_new_root_born_in_a_base_complement_fails_closed() -> None:
    # Site 2 is rootless and dominated at eta=0.  The candidate lowers it
    # enough to insert through the active 0/1 cut, so a certificate that kept
    # only old root records would be unsound.
    sites = _sites_from_ray_lines(
        [(0, 0), (-2, 0), (1, 0)],
        [(0, 0, 0), (1, -1, 0), (4, 0, 1)],
    )
    ray = _static_x_ray()
    program = _compile(sites)
    candidate_weights = sites.weight_coefficients.clone()
    candidate_weights[2, 0] += 5
    candidate = AffineKineticPowerSites(sites.positions0, sites.velocities, candidate_weights)

    certificate = certify_multichart_simple_root_binary64_candidate(
        sites,
        ray,
        program,
        candidate,
        ray,
    )

    assert not certificate.passed
    assert certificate.full_recompile_required
    assert certificate.reason.startswith("new_or_uncontrolled_root_in_complement:active_cut_competitor")


def test_repeated_and_shared_roots_fail_closed() -> None:
    ray = _static_x_ray()
    repeated_sites = _sites_from_ray_lines(
        [(0, 0), (-2, 0), (1, 0)],
        [(0, 0, 0), (1, -1, 0), (0, 0, 1)],
    )
    repeated_program = _compile(repeated_sites)
    repeated = certify_multichart_simple_root_binary64_candidate(
        repeated_sites,
        ray,
        repeated_program,
        repeated_sites,
        ray,
    )
    assert not repeated.passed
    assert repeated.reason == "shared_repeated_or_ambiguous_base_root"

    # Both inactive pair-near predicates have the simple root t=3/2.  They
    # are algebraically shared but do not alter the currently dominant site 1.
    shared_sites = _sites_from_ray_lines(
        [(0, 0), (-2, 0), (1, 0), (2, 0)],
        [
            (0, 0, 0),
            (1, -1, 0),
            (Fraction(3, 2), -1, 0),
            (Fraction(3, 2), -1, 0),
        ],
    )
    shared_program = _compile(shared_sites)
    assert shared_program.passed
    assert any(guard.simultaneous_source_count == 2 for guard in shared_program.inactive_event_guards)
    shared = certify_multichart_simple_root_binary64_candidate(
        shared_sites,
        ray,
        shared_program,
        shared_sites,
        ray,
    )
    assert not shared.passed
    assert shared.reason == "shared_repeated_or_ambiguous_base_root"


def test_nearly_colliding_distinct_simple_roots_keep_disjoint_tubes() -> None:
    epsilon = Fraction(1, 1024)
    # L1=z-t crosses near=0 at t=0 and far=epsilon at t=epsilon.
    sites = _sites_from_ray_lines(
        [(0, 0), (1, 0)],
        [(0, 0, 0), (0, -1, 0)],
    )
    ray = _static_x_ray()
    program = _compile(
        sites,
        t_min=Fraction(-1, 8),
        t_max=Fraction(1, 8),
        near=0,
        far=epsilon,
    )

    certificate = certify_multichart_simple_root_binary64_candidate(
        sites,
        ray,
        program,
        sites,
        ray,
    )

    assert certificate.passed
    assert len(certificate.root_continuations) == 2
    left, right = certificate.root_continuations
    assert left.neighborhood_upper < right.neighborhood_lower
    assert left.source.kind == "pair_near"
    assert right.source.kind == "pair_far"


def test_ray_collapse_and_endpoint_event_fail_closed_by_typed_guard() -> None:
    sites = _rational_two_event_sites()
    ray = _static_x_ray()
    program = _compile(sites)
    collapsing_ray = ray.clone()
    collapsing_ray[6] = -1

    collapsed = certify_multichart_simple_root_binary64_candidate(
        sites,
        ray,
        program,
        sites,
        collapsing_ray,
    )
    assert not collapsed.passed
    assert collapsed.reason == "nonroot_guard_not_uniformly_positive"

    endpoint_program = _compile(sites, t_min=-1)
    assert endpoint_program.endpoint_event_guards
    endpoint = certify_multichart_simple_root_binary64_candidate(
        sites,
        ray,
        endpoint_program,
        sites,
        ray,
    )
    assert not endpoint.passed
    assert endpoint.reason == "endpoint_root_requires_full_recompile"


def test_step_inside_root_tubes_passes_and_outside_fails() -> None:
    sites = _rational_two_event_sites()
    ray = _static_x_ray()
    program = _compile(sites)

    def candidate(delta: float) -> AffineKineticPowerSites:
        weights = sites.weight_coefficients.clone()
        weights[1, 0] += delta
        return AffineKineticPowerSites(sites.positions0, sites.velocities, weights)

    inside = certify_multichart_simple_root_binary64_candidate(
        sites,
        ray,
        program,
        candidate(0.25),
        ray,
    )
    outside = certify_multichart_simple_root_binary64_candidate(
        sites,
        ray,
        program,
        candidate(1.5),
        ray,
    )

    assert inside.requested_radius_certified
    assert not outside.passed
    assert outside.reason.startswith("root_left_or_right_boundary_sign_not_preserved")


def test_api_and_certificate_have_no_requested_frame_axis() -> None:
    parameters = inspect.signature(certify_multichart_simple_root_binary64_candidate).parameters
    assert "frame_count" not in parameters
    assert "requested_frame_count" not in parameters
    assert "sample_count" not in parameters

    sites = _rational_two_event_sites()
    ray = _static_x_ray()
    certificate = certify_multichart_simple_root_binary64_candidate(
        sites,
        ray,
        _compile(sites),
        sites,
        ray,
    )
    assert certificate.passed
    assert not certificate.requested_frame_sampling_used


def test_fixed_seed_accepted_candidates_match_fresh_exact_compilation() -> None:
    rng = random.Random(20260803)
    ray = _static_x_ray()
    fixtures = (
        (
            _rational_two_event_sites(),
            1,
            dict(t_min=Fraction(-2), t_max=Fraction(2), near=Fraction(0), far=Fraction(1)),
        ),
        (
            _sites_from_ray_lines(
                [(1, 0), (-1, 0), (0, 0)],
                [(0, 0, 0), (0, 0, 0), (0, -1, 0)],
            ),
            2,
            dict(
                t_min=Fraction(-1, 2),
                t_max=Fraction(1, 2),
                near=Fraction(-1),
                far=Fraction(1),
            ),
        ),
    )
    accepted = 0
    rejected = 0
    case_count = 0
    for sites, perturbed_site, domain in fixtures:
        base = compile_active_kinetic_owner_charts(sites, ray, **domain)
        assert base.passed
        # Five small dyadic candidates plus one intentionally larger dyadic
        # candidate per fixture keep this deterministic and bounded while
        # exercising both acceptance and fail-closed fallback.
        constant_deltas = [Fraction(rng.randint(-40, 40), 256) for _ in range(5)]
        constant_deltas.append(Fraction(rng.choice((-1, 1)) * rng.randint(320, 448), 256))
        rng.shuffle(constant_deltas)
        for constant_delta in constant_deltas:
            linear_delta = Fraction(rng.randint(-4, 4), 256)
            candidate_weights = sites.weight_coefficients.clone()
            candidate_weights[perturbed_site, 0] += float(constant_delta)
            candidate_weights[perturbed_site, 1] += float(linear_delta)
            candidate = AffineKineticPowerSites(
                positions0=sites.positions0,
                velocities=sites.velocities,
                weight_coefficients=candidate_weights,
            )
            certificate = certify_multichart_simple_root_binary64_candidate(
                sites,
                ray,
                base,
                candidate,
                ray,
            )
            case_count += 1
            if not certificate.passed:
                rejected += 1
                continue
            accepted += 1
            recompiled = compile_active_kinetic_owner_charts(candidate, ray, **domain)
            assert recompiled.passed
            assert tuple(chart.owner_word for chart in recompiled.charts) == certificate.candidate_owner_words
            assert len(recompiled.active_event_guards) == certificate.semantic_event_count

    assert case_count == 12
    assert accepted > 0
    assert rejected > 0
