# Gauged UVT Combined Orbit Cache Stress

## Context

The previous orbit gates were intentionally factored:

- support-tail alpha debounce accepts tiny missing Gaussian tails and rejects
  center/core loss;
- visibility refresh remains independent from support debounce;
- projective orbit depth roots split one cell into stable order strata;
- `slack_budgeted` spends crowded guard slots on traces nearest to a support
  boundary instead of primitive id order.

The next falsification step was to put those mechanisms into one synthetic
orbit compiler decision.

## Regression Added

Added:

```text
tests/test_star_uvt_trainer_interval_gated.py::test_projective_interval_orbit_tail_visibility_and_slack_guard_share_one_refresh
```

Fixture shape:

```text
frames = 4
times = tan(theta/2), theta in [-3deg, -1deg, 1deg, 3deg]
base traces = 20
far guard extras = 12
near guard extras = 12
tile_capacity = 32
uv_padding = 4.0
support_uv_padding = 8.5
```

Traces 0 and 1 form the orbit visibility crossing:

```text
z_0(t) = 2 + 4t
z_1(t) = 2 - 4t
root: t = 0
```

Trace 0 is also shifted by `+0.10` in the live coefficient table after compile,
creating a small support-boundary miss that is below the tail-alpha epsilon.

## Evidence

Focused single-test gate:

```text
1 passed in 6.17s
```

Full focused projective plus interval-gated trainer suite:

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

124 passed in 9.87s
```

Observed contract inside the test:

```text
support_tail_alpha_bound_before in (1e-4, 3e-4)
support_boundary_overshoot_px in (0.04, 0.08)
visibility_before.order_mismatch_samples = 4
visibility_stratified = True
fallback_marked = False
visibility_after.order_mismatch_samples = 0
max per-cell active count = 32
```

Guard allocation comparison:

```text
trace_budgeted:
    tile0 keeps lower-id far extras
    tile0 excludes near extras

slack_budgeted:
    tile0 keeps near-boundary extras
    tile0 excludes far extras
```

## Interpretation

This is the first single synthetic gate where the compiler does all of these at
once:

1. accepts a support miss by a render-relevant tail-alpha bound;
2. still refreshes because visibility is stale;
3. turns the orbit depth root into two stable order strata;
4. resolves guard-slot pressure using support-event geometry rather than id
   order.

The remaining weakness is scope, not algebra: this is still a synthetic
many-trace orbit fixture, not a broader scene/cache-policy artifact. The next
promotion step should run the same combined checks on measured trainer cache
rows or a richer generated orbit scene.
