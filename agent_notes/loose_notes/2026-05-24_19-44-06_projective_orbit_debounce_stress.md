# Projective Orbit Debounce Stress

## Context

Goal memory for this work stays:

```text
goal       fast 2D rasters across time from 4D spacetime primitives
meta-goal  share projection/support/binning/visibility/backward over time
key math   UVT trace = pi_* Gamma^* world_primitive
theory     STAR UVT is one local gauge expression of a camera-ray bundle atlas
```

The previous cap128 trace-budgeted artifact showed support churn was mostly
subpixel support-boundary noise, but the first render-error debounce checks were
axis-aligned. That left an uncomfortable question: does the tolerance still
make sense for a revolving-camera/projective chart?

## Result

Added `test_projective_interval_orbit_support_debounce_has_bounded_tail_error`
in `tests/test_star_uvt_trainer_interval_gated.py`.

The fixture creates a tiny yaw-window rational/projective trace, compiles it
through `split_projective_trace_windows(...)` and
`projective_trace_windows_to_cell_trace_atlas(...)`, then applies a live
coefficient update that pushes padded support about `0.056px` across a tile
boundary. Strict refresh rebins; tolerant refresh with
`support_stale_overshoot_epsilon=0.075` reuses the stale cell.

Strict-rebinned versus tolerant-reused reference render:

```text
max RGB error  < 1e-4
mean RGB error < 1e-6
missing tile pairs = 4
```

This supports the local-chart version of the debounce: a certified support
padding plus subpixel overshoot only drops a Gaussian tail even when the trace
comes from projective/revolving-camera math.

## What It Does Not Prove

This is not a full-orbit theorem. It is a local chart stress. A long orbit can
still require chart changes at denominator margins, support events, visibility
strata, near-camera nonlinearities, and disocclusions.

The correct theory remains:

```text
full orbit = atlas of camera-ray bundle charts
local chart = rational/projective trace with support and visibility certificates
debounce = bounded tolerance on certified support coverage only
fallback/refinement = required when support/visibility certificates fail
```

## Verification

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_trainer_interval_gated.py::test_projective_interval_orbit_support_debounce_has_bounded_tail_error -q

1 passed in 3.12s
```

Focused STAR UVT projective/interval suite:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_visibility.py \
  tests/test_star_uvt_projective_correctness.py \
  tests/test_star_uvt_visibility_support_bridge.py \
  tests/test_star_uvt_render_configs.py \
  tests/test_star_uvt_trainer_interval_gated.py \
  tests/test_star_uvt_projective_trace.py \
  tests/test_star_uvt_projective_uvt_producer.py \
  tests/test_star_uvt_projective_orbit_windows.py \
  tests/test_star_uvt_projective_binning.py -q

121 passed in 26.58s
```
