"""Deterministic exact-versus-compiled WorldFoam routing cost model.

The compiled Lie atlas removes the expensive ``F x R`` world replay, but it is
not universally cheaper.  A low-run word or an over-ranked chart can cost less
to replay exactly.  This source-only gate makes that negative case explicit so
benchmark code cannot equate "sublinear world work" with "lower total work".

The units are scalar interaction proxies for the current CPU implementation:

* exact P0 forward plus constant-state reverse: ``3 F_c W_c``;
* compiled node forward plus reverse: ``2 J_c W_c``;
* per-track sample evaluation/reduction: ``P F_c J_c``;
* dense coefficient fitting: ``P J_c^2``;
* verified barycentric sample-to-node weights: ``F_c J_c``; and
* exceptional row-local dense fallback: ``N_fb,c J_c^2``.

``W_c`` is the total active run count across all tracks in chart ``c`` and
``N_fb,c`` counts time rows, not track-time rows: one interpolation row is
shared across the chart's tracks.  The retained ``dense_fit`` mode charges
``F_c J_c^2`` as an oracle/control.  Constants are intentionally conservative
and must be replaced by measured native timings before a publication speed
claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class CompiledRouteChartCost:
    """Work dimensions for one half-open topology/transfer chart."""

    sample_count: int
    node_count: int
    active_run_count: int
    sample_weight_dense_fallback_row_count: int = 0

    def validate(self, *, track_count: int) -> None:
        if self.sample_count < 1 or self.node_count < 2 or self.active_run_count < track_count:
            raise ValueError(
                "each chart needs positive samples, rank >= 2, and at least one active run per track"
            )
        if not 0 <= self.sample_weight_dense_fallback_row_count <= self.sample_count:
            raise ValueError(
                "sample_weight_dense_fallback_row_count must be between zero and sample_count"
            )


@dataclass(frozen=True)
class CompiledRouteDecision:
    """Auditable route choice under one declared interaction model."""

    route: Literal["exact", "compiled"]
    reason: str
    exact_interactions: int
    compiled_interactions: int
    compiled_world_interactions: int
    compiled_sample_interactions: int
    dense_fit_interactions: int
    sample_weight_interactions: int
    sample_weight_linear_interactions: int
    sample_weight_dense_fallback_interactions: int
    sample_weight_dense_fallback_rows: int
    current_total_sample_count: int
    break_even_density_multiplier: int | None
    break_even_total_sample_count: int | None
    weight_evaluation: Literal["dense_fit", "linear_barycentric"]

    @property
    def compiled_to_exact_ratio(self) -> float:
        return self.compiled_interactions / self.exact_interactions


def choose_worldfoam_replay_route(
    charts: tuple[CompiledRouteChartCost, ...],
    *,
    track_count: int,
    weight_evaluation: Literal["dense_fit", "linear_barycentric"] = "dense_fit",
    minimum_compiled_margin: float = 0.0,
) -> CompiledRouteDecision:
    """Choose exact replay unless compiled work clears the requested margin.

    ``minimum_compiled_margin=0.1`` requires the compiled proxy to be at least
    ten percent below exact.  Break-even scales the supplied chart sample
    distribution uniformly while keeping ranks and active words fixed.
    """

    if track_count < 1:
        raise ValueError("track_count must be positive")
    if not charts:
        raise ValueError("charts must be nonempty")
    if weight_evaluation not in {"dense_fit", "linear_barycentric"}:
        raise ValueError("weight_evaluation must be dense_fit or linear_barycentric")
    if not math.isfinite(minimum_compiled_margin) or not 0.0 <= minimum_compiled_margin < 1.0:
        raise ValueError("minimum_compiled_margin must be finite and in [0,1)")
    for chart in charts:
        chart.validate(track_count=track_count)

    exact = 3 * sum(chart.sample_count * chart.active_run_count for chart in charts)
    compiled_world = 2 * sum(chart.node_count * chart.active_run_count for chart in charts)
    compiled_sample = track_count * sum(
        chart.sample_count * chart.node_count for chart in charts
    )
    dense_fit = track_count * sum(chart.node_count * chart.node_count for chart in charts)
    if weight_evaluation == "dense_fit":
        sample_weight_linear = 0
        sample_weight_dense_fallback = 0
        sample_weights = sum(
            chart.sample_count * chart.node_count * chart.node_count for chart in charts
        )
    else:
        sample_weight_linear = sum(
            chart.sample_count * chart.node_count for chart in charts
        )
        sample_weight_dense_fallback = sum(
            chart.sample_weight_dense_fallback_row_count
            * chart.node_count
            * chart.node_count
            for chart in charts
        )
        sample_weights = sample_weight_linear + sample_weight_dense_fallback
    compiled = compiled_world + compiled_sample + dense_fit + sample_weights
    threshold = exact * (1.0 - minimum_compiled_margin)
    route: Literal["exact", "compiled"] = "compiled" if compiled < threshold else "exact"

    exact_slope = exact
    compiled_slope = compiled_sample + sample_weights
    compiled_constant = compiled_world + dense_fit
    if exact_slope <= compiled_slope:
        multiplier = None
        break_even_samples = None
        reason = (
            "compiled per-sample slope is not below exact replay; rank/run structure has no temporal break-even"
        )
    else:
        # Need C + n*S < n*E*(1-margin). Strict inequality adds one.
        effective_gain = exact_slope * (1.0 - minimum_compiled_margin) - compiled_slope
        if effective_gain <= 0.0:
            multiplier = None
            break_even_samples = None
            reason = "the requested safety margin removes the compiled temporal break-even"
        else:
            multiplier = math.floor(compiled_constant / effective_gain) + 1
            break_even_samples = multiplier * sum(chart.sample_count for chart in charts)
            reason = (
                "compiled proxy clears the declared margin"
                if route == "compiled"
                else "current temporal density is below the compiled break-even"
            )

    return CompiledRouteDecision(
        route=route,
        reason=reason,
        exact_interactions=exact,
        compiled_interactions=compiled,
        compiled_world_interactions=compiled_world,
        compiled_sample_interactions=compiled_sample,
        dense_fit_interactions=dense_fit,
        sample_weight_interactions=sample_weights,
        sample_weight_linear_interactions=sample_weight_linear,
        sample_weight_dense_fallback_interactions=sample_weight_dense_fallback,
        sample_weight_dense_fallback_rows=(
            sum(chart.sample_weight_dense_fallback_row_count for chart in charts)
            if weight_evaluation == "linear_barycentric"
            else 0
        ),
        current_total_sample_count=sum(chart.sample_count for chart in charts),
        break_even_density_multiplier=multiplier,
        break_even_total_sample_count=break_even_samples,
        weight_evaluation=weight_evaluation,
    )


__all__ = [
    "CompiledRouteChartCost",
    "CompiledRouteDecision",
    "choose_worldfoam_replay_route",
]
