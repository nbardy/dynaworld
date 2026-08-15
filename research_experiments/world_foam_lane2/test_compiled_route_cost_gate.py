from __future__ import annotations

import pytest
from compiled_route_cost_gate import (
    CompiledRouteChartCost,
    choose_worldfoam_replay_route,
)


def test_current_two_run_twenty_node_fixture_is_not_a_dense_weight_speed_claim() -> None:
    # Fixed interval proportions from the saved F=1024 death-curve fixture:
    # ranks 16/2/2 over 256/256/512 samples, with two active runs per chart.
    charts = (
        CompiledRouteChartCost(256, 16, 2),
        CompiledRouteChartCost(256, 2, 2),
        CompiledRouteChartCost(512, 2, 2),
    )
    decision = choose_worldfoam_replay_route(charts, track_count=1)

    assert decision.exact_interactions == 6144
    assert decision.compiled_world_interactions == 80
    assert decision.compiled_sample_interactions == 5632
    assert decision.dense_fit_interactions == 264
    assert decision.sample_weight_interactions == 68608
    assert decision.sample_weight_linear_interactions == 0
    assert decision.sample_weight_dense_fallback_interactions == 0
    assert decision.sample_weight_dense_fallback_rows == 0
    assert decision.route == "exact"
    assert decision.break_even_density_multiplier is None
    assert "no temporal break-even" in decision.reason


def test_high_run_scene_has_a_real_compiled_break_even() -> None:
    charts = (CompiledRouteChartCost(sample_count=4, node_count=8, active_run_count=2048),)
    below = choose_worldfoam_replay_route(charts, track_count=64, weight_evaluation="linear_barycentric")
    assert below.route == "exact"
    assert below.break_even_density_multiplier == 2
    assert below.break_even_total_sample_count == 8

    above = choose_worldfoam_replay_route(
        (CompiledRouteChartCost(sample_count=32, node_count=8, active_run_count=2048),),
        track_count=64,
        weight_evaluation="linear_barycentric",
    )
    assert above.route == "compiled"
    assert above.compiled_to_exact_ratio < 1.0


def test_linear_weight_route_counts_row_local_dense_fallbacks() -> None:
    no_fallback = choose_worldfoam_replay_route(
        (
            CompiledRouteChartCost(
                sample_count=32,
                node_count=8,
                active_run_count=280,
            ),
        ),
        track_count=64,
        weight_evaluation="linear_barycentric",
    )
    with_fallback = choose_worldfoam_replay_route(
        (
            CompiledRouteChartCost(
                sample_count=32,
                node_count=8,
                active_run_count=280,
                sample_weight_dense_fallback_row_count=32,
            ),
        ),
        track_count=64,
        weight_evaluation="linear_barycentric",
    )

    assert no_fallback.route == "compiled"
    assert no_fallback.sample_weight_linear_interactions == 32 * 8
    assert no_fallback.sample_weight_dense_fallback_interactions == 0
    assert with_fallback.route == "exact"
    assert with_fallback.sample_weight_linear_interactions == 32 * 8
    assert with_fallback.sample_weight_dense_fallback_interactions == 32 * 8 * 8
    assert with_fallback.sample_weight_dense_fallback_rows == 32
    assert (
        with_fallback.compiled_interactions - no_fallback.compiled_interactions
        == with_fallback.sample_weight_dense_fallback_interactions
    )


def test_margin_can_hold_a_near_parity_compiled_route_back() -> None:
    charts = (CompiledRouteChartCost(sample_count=64, node_count=4, active_run_count=16),)
    nominal = choose_worldfoam_replay_route(
        charts,
        track_count=8,
        weight_evaluation="linear_barycentric",
    )
    guarded = choose_worldfoam_replay_route(
        charts,
        track_count=8,
        weight_evaluation="linear_barycentric",
        minimum_compiled_margin=0.2,
    )
    assert nominal.route == "compiled"
    assert guarded.route == "exact"


@pytest.mark.parametrize(
    ("charts", "kwargs", "message"),
    (
        ((), {"track_count": 1}, "nonempty"),
        ((CompiledRouteChartCost(1, 2, 1),), {"track_count": 0}, "positive"),
        ((CompiledRouteChartCost(0, 2, 1),), {"track_count": 1}, "positive samples"),
        ((CompiledRouteChartCost(1, 1, 1),), {"track_count": 1}, "rank"),
        (
            (CompiledRouteChartCost(1, 2, 1, sample_weight_dense_fallback_row_count=2),),
            {"track_count": 1},
            "fallback",
        ),
        ((CompiledRouteChartCost(1, 2, 1),), {"track_count": 2}, "one active run"),
        (
            (CompiledRouteChartCost(1, 2, 1),),
            {"track_count": 1, "minimum_compiled_margin": 1.0},
            "margin",
        ),
    ),
)
def test_invalid_cost_contracts_fail_closed(
    charts: tuple[CompiledRouteChartCost, ...],
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        choose_worldfoam_replay_route(charts, **kwargs)
