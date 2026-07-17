# Gauged UVT Slack-Budgeted Guard Policy

## Context

The interval atlas cache already had cap-safe guard policies:

```text
fixed -> budgeted -> local_budgeted -> trace_budgeted
```

`trace_budgeted` proved that crowded guarded tiles can keep base-active traces
and spend remaining capacity on extra guarded traces without overflowing.
However, it spent those extra slots by primitive id order. That is stable, but
not mathematical: it does not know which trace is closest to crossing the tile
support event.

## Change

Added:

```text
projective_interval.support_guard_policy = "slack_budgeted"
```

When a guarded tile overflows, the policy:

1. Keeps all base-active trace ids for the tile.
2. Computes a per-trace support-event distance for extra guarded ids:

```text
d_i,C = min_sample dist_inf(base_support_i(sample), tile C)
```

3. Spends remaining tile slots on smaller `d_i,C` first, with primitive id as a
deterministic tie-break.

This lives in both the production-facing backend
`src/train/star_uvt_projective_interval_backend.py` and the trainer-state
refresh path
`third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/trainer_harness/tile_metal_autograd.py`.

## Test Contract

The new unit fixture creates one crowded tile with:

- base traces already active in the tile,
- a lower-id extra group farther from the boundary,
- a higher-id extra group closer to the boundary.

`trace_budgeted` selects the lower-id farther group. `slack_budgeted` selects
the higher-id nearer group and remains zero-overflow. This catches regressions
where the policy silently falls back to id-order allocation.

## Meaning

This does not yet prove the final adaptive guard solution. It is the first
support-event-aware allocator. The next gate should combine it with the
bounded subpixel debounce artifacts and validate on harder motion/visibility
stress scenes.

## Tests

Targeted producer/config gate:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_uvt_producer.py \
  tests/test_star_uvt_render_configs.py -q

28 passed in 17.45s
```

Full focused projective plus interval-gated trainer gate:

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

115 passed in 16.82s
```

Follow-up image-level support-debounce stress added two more trainer interval
tests. The full focused gate now passes:

```text
117 passed in 18.20s
```
