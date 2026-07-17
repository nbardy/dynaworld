# Projective Interval Measured Cache Policy

## Context

The active UVT trace-atlas goal is to share projection/support/binning/
visibility/backward work over time instead of recompiling world-side metadata
per frame or per training step. The compatible projective interval trainer
route already had live differentiable atlas updates and a refresh oracle, but
the cache lifetime was still controlled by the fixed `refresh_every` cadence.
That made measured atlas validity a repair mechanism, not the cache policy.

## Change

Added `feature_uvt.projective_interval.refresh_policy` with two values:

```text
cadence   old behavior; rebuild the full compatible atlas when refresh_every expires
measured  build once, then reuse compiled cells and repair by measured staleness
```

The trainer now decides rebuilds through
`_projective_interval_cache_should_rebuild(...)`. In measured mode, cached
steps rebuild live differentiable tensors from the current UVT model, then call
`ProjectiveCellIntervalTrainerState.refresh(force=False)` before rendering.
That oracle checks support coverage, visibility/order, fallback marking, and
complexity budget. If metadata is stale, it rebins/stratifies/marks fallback
without replacing the live trace/color/opacity tensors.

## Evidence

Focused checks:

```text
PYTHONPATH=src/train uv run python -m py_compile \
  src/train/star_uvt_projective_interval_backend.py \
  src/train/star_uvt_feature_overfit_trainer.py \
  tests/test_star_uvt_projective_uvt_producer.py \
  tests/test_star_uvt_render_configs.py

PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_render_configs.py -q
  # 6 passed in 3.13s

PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_uvt_producer.py -q
  # 16 passed in 15.13s
```

Focused projective plus interval-gated bundle:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_trace.py \
  tests/test_star_uvt_projective_orbit_windows.py \
  tests/test_star_uvt_projective_visibility.py \
  tests/test_star_uvt_projective_binning.py \
  tests/test_star_uvt_projective_correctness.py \
  tests/test_star_uvt_projective_uvt_producer.py \
  tests/test_star_uvt_render_configs.py \
  tests/test_star_uvt_config_keys.py \
  tests/test_star_uvt_trainer_interval_gated.py -q
  # 113 passed in 10.45s
```

The real MPS trainer smoke now uses `refresh_policy="measured"` with
`refresh_every=2` for 3 steps and verifies:

```text
projective_interval_cache_rebuilds == 1
projective_interval_cache_live_updates == 2
projective_interval_cache_alpha_renders == 3
projective_interval_cache_staleness_checks == 2
```

That means step 2 skips the cadence rebuild that would have happened under
`refresh_policy="cadence"`.

A separate MPS render-helper gate now shifts a cached trace across a spatial
tile boundary under `refresh_policy="measured"` and verifies:

```text
projective_interval_cache_rebuilds == 1
projective_interval_cache_live_updates == 1
projective_interval_cache_staleness_checks == 1
projective_interval_cache_stale_refreshes == 1
projective_interval_cache_support_rebins == 1
```

That covers actual stale support repair in the cached render path, not only the
policy decision.

A four-step optimizer-style MPS gate now parameterizes the trace center, renders
through the measured cache, applies SGD to move the center across the same
support boundary, and verifies:

```text
projective_interval_cache_rebuilds == 1
projective_interval_cache_live_updates == 3
projective_interval_cache_alpha_renders == 4
projective_interval_cache_staleness_checks == 3
projective_interval_cache_stale_refreshes == 1
projective_interval_cache_support_rebins == 1
```

That is the first sustained update-loop evidence for measured reuse. It is
still controlled: the next proof should be a real trainer timing/quality smoke
under ordinary optimizer dynamics.

A real synthetic `run_training` A/B gate now uses the same sequence, seed,
four steps, and `refresh_every=2` for both policies. It verifies:

```text
cadence:  rebuilds=2, live_updates=2, staleness_checks=2
measured: rebuilds=1, live_updates=3, staleness_checks=3
```

and asserts that measured `losses` and `end_loss` match cadence within `1e-5`.
This proves measured cache reuse is behavior-preserving in the real trainer
route for a small ordinary optimizer run.

## Current Model

The cache should be understood as:

```text
static-ish metadata: cells, support, active sets, order/fallback/budget state
live tensors:        trace coefficients, opacity, temporal opacity, color
```

Measured mode is the first production-facing approximation of the desired
sensor-time compiler: keep the expensive atlas object alive and validate it by
geometry/visibility certificates, rather than refreshing because a counter
expired.

## Remaining Risks

- This does not yet prove scale-relevant trainer timing/quality; the real
  trainer A/B is synthetic and four steps.
- The compatible producer still rejects anisotropic spatial precision and
  pixel-varying depth slopes.
- The measured trainer smoke currently observes no stale repair events; stale
  support repair is covered in focused render-helper and optimizer-style paths.
- Frame chunking and analytic/sparse VJP target modes are still outside the
  projective interval route.

## Next Test

Run a longer saved measured-vs-cadence timing/quality artifact on the intended
STAR UVT feature config, then compare rebuild/live-update/staleness counters,
step timing, and final quality.
