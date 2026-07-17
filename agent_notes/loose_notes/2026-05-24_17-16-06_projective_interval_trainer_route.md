# Projective Interval Producer Routed Through STAR UVT Trainer

Date: 2026-05-24

## Context

The active Gauged UVT Trace Atlas goal is:

```text
goal       fast 2D rasters across time from 4D spacetime primitives
meta-goal  share projection/support/binning/visibility/backward work over time
key math   UVT trace = pi_* Gamma^* world_primitive
theory     STAR UVT is one local gauge expression of a camera-ray bundle atlas
```

The previous implementation had a production-facing compatible UVT-tube
producer in `src/train/star_uvt_projective_interval_backend.py`, but the real
STAR UVT feature trainer still called `require_projective_interval_atlas_producer`
with `producer_available=False`, so `feature_uvt.projective_interval.enabled`
was always rejected.

## What Changed

`src/train/star_uvt_feature_overfit_trainer.py` now has a narrow production
route for the compatible producer:

- it accepts the projective interval backend when `feature_dim == 3`
- it locks spatial precision to the backend `sigma_px`
- it keeps temporal precision, velocity, center, opacity, and feature gradients
  live
- it renders the feature image through
  `make_projective_cell_interval_trainer_state_from_uvt_tubes(...)`
- it renders a second white-color interval atlas to recover total alpha for the
  existing alpha-background/colorizer objective
- it requires full-frame training chunks and autograd feature-target VJP mode

The spatial precision lock is intentional, not a hidden fallback: the current
producer is exact only for isotropic screen fibers matching `sigma_px`. The
gradient hook zeros only the spatial precision columns, leaving temporal
precision gradients alive so the residual temporal opacity envelope remains
trainable.

## Tests

Focused checks passed:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_uvt_producer.py -q
```

Result:

```text
10 passed in 15.64s
```

Broader focused gate passed:

```bash
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
```

Result:

```text
111 passed in 19.81s
```

`py_compile` passed for the edited trainer/test files, and `git diff --check`
reported no whitespace errors on those paths.

## Current Limitation

This is not the full camera-bundle endpoint. It is the first real trainer
route for a compatible local gauge expression:

- feature image is RGB-width only (`feature_dim=3`)
- measured staleness refresh now has an explicit `refresh_policy="measured"`
  trainer cache mode, but still needs longer-run stability evidence
- frame chunking is not supported yet
- analytic/sparse target-grid VJP modes are not wired to the projective state
- anisotropic footprints and pixel-varying depth still require a richer trace
  representation

## Next Move

Prove measured staleness refresh on longer trainer runs. A focused render-helper
gate now covers actual stale support rebin during measured reuse, but sustained
optimizer-run behavior remains open. Then extend the trace payload for
anisotropic footprints, pixel-varying depth, and WorldFoam/instance cells rather
than multiplying flags around the compatible screen-tube path.

## Update: Fixed Cadence Cache

The route now has a first cache path. `make_projective_cell_interval_live_atlas_from_uvt_tubes(...)`
updates differentiable trace/color/opacity tensors from current UVT model
tensors while reusing compiled cells from a reference atlas. The trainer uses
that when `feature_uvt.projective_interval.refresh_every > 1`: rebuild on
cadence, otherwise reuse cells and replace live tensors. A 2-step MPS
`run_training` test verifies one full rebuild, one live update, two alpha
white-trace renders, and gradient flow through the second step.

## Update: Measured Cache Policy

`feature_uvt.projective_interval.refresh_policy` now accepts `cadence` and
`measured`. `cadence` preserves the old `refresh_every` full-atlas rebuild
behavior; `measured` rebuilds the compatible atlas only for the first cached
render and then reuses compiled cells with fresh live tensors. Each live update
still calls the trainer-state refresh oracle before render, so coverage,
visibility/order, fallback, and complexity staleness can repair metadata before
Metal rendering.

The MPS trainer smoke now uses `refresh_policy="measured"` with
`refresh_every=2` for 3 steps and verifies one rebuild, two live updates, three
white-trace alpha renders, and two staleness checks. That proves measured reuse
can skip the cadence rebuild path in the real trainer route. The focused
projective plus interval-gated bundle now passes:

```text
111 passed in 19.81s
```

## Update: Measured Staleness Refresh

Cached live updates now call `ProjectiveCellIntervalTrainerState.refresh(force=False)`
before rendering. That checks support coverage, visibility/order, fallback
marking, and complexity budget; stale metadata is repaired by rebinning or
visibility stratification while preserving the live differentiable tensor
payload. The trainer row now reports staleness checks, stale refreshes, support
rebins, visibility stratifications, and fallback marks. A CPU test moves a live
trace across a cached tile boundary, observes stale coverage, and verifies
`state.refresh()` repairs it.
