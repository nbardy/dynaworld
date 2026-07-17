# Gauged UVT Support-Event Slack Telemetry

## Context

The current projective interval cache has three cap-safe support-guard policies:
`budgeted`, `local_budgeted`, and `trace_budgeted`. The latest docs/artifacts say
they avoid cap128 overflow but still churn support metadata under ordinary tube
motion. Fixed guard2 at cap256 removes churn, so the missing ingredient is not
the existence of chart support margin; it is knowing which traces/cells actually
need margin under the optimizer update.

## Change

Added signed support-boundary slack to
`projective_trace_cell_atlas_support_margin_report(...)`:

```text
min_boundary_slack_px
mean_boundary_slack_px
p05_boundary_slack_px
```

Positive slack means the live footprint still has room inside the compiled
tile support. Negative slack means the live support has crossed the nearest
support-event boundary; its magnitude matches the existing overshoot measure
for simple boundary misses.

The feature trainer cache now records:

```text
projective_interval_cache_last_support_min_slack_px
projective_interval_cache_min_support_min_slack_px
```

alongside the existing missing-tile and max-overshoot counters.

## Why This Matters

The next adaptive guard policy can now be phrased as a measurable inequality:

```text
predicted_optimizer_displacement_i < boundary_slack_i,C
```

If the inequality holds, the live update can reuse the cell safely. If it fails,
we need extra guard, split/refit, local fallback, or a rebin. This is the
bridge from "budgeted slots" to support-event-root-aware guards.

## Tests

Updated the subpixel-overshoot refresh test and support-margin correctness test
to assert signed slack. Added a cache telemetry unit test so the trainer report
surface cannot silently drop the new metric.

Focused verification:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_correctness.py \
  tests/test_star_uvt_trainer_interval_gated.py \
  tests/test_star_uvt_projective_uvt_producer.py \
  tests/test_star_uvt_render_configs.py -q

78 passed in 11.25s
```

Full projective plus interval-gated trainer gate:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_trace.py \
  tests/test_star_uvt_projective_orbit_windows.py \
  tests/test_star_uvt_projective_visibility.py \
  tests/test_star_uvt_projective_binning.py \
  tests/test_star_uvt_projective_correctness.py \
  tests/test_star_uvt_projective_uvt_producer.py \
  tests/test_star_uvt_render_configs.py \
  tests/test_star_uvt_trainer_interval_gated.py -q

114 passed in 17.59s
```
