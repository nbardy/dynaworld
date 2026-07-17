# Gauged UVT Orbit Support/Visibility Regression

## Context

The current Gauged UVT Trace Atlas goal is to compile 4D spacetime primitives
through a known camera program into reusable sensor-time traces. The recent
question was whether rich projective/gauged math can replace ad hoc "charts"
and fallbacks for revolving cameras.

The working answer is sharper now:

- projective/rational gauges should carry smooth orbit support motion;
- visibility is still a separate certificate over the trace arrangement;
- support debounce must not become accidental visibility debounce.

## Current Model

For a camera orbit, a primitive trace is locally represented by projective
coefficients and lowered into direct cell-polynomial atlas rows. A small live
coefficient update can move support across a tile boundary by a subpixel amount.
If the compiled support padding is a true footprint bound, that can be safely
debounced under a visual/error budget.

However, front-to-back order is not a support property. If live depth order no
longer matches the compiled depth intervals, the atlas must refresh, stratify,
or fall back even when support drift is below the debounce threshold.

## Regression Added

Added:

```text
tests/test_star_uvt_trainer_interval_gated.py::test_projective_interval_orbit_support_debounce_still_repairs_visibility_order
```

The fixture builds two tiny yaw-window projective traces with the same orbit
support shape but different live depth. It then applies a `0.10px` live
coefficient update and intentionally stale cell depth intervals.

Observed contract:

```text
support_stale_overshoot_epsilon = 0.10
support_only_refresh.rebinned = False
max_boundary_overshoot_px ~= 0.083
visibility_before.order_mismatch_samples = 4
visibility_enabled_refresh.rebinned = True
visibility_after.order_mismatch_samples = 0
refreshed_order = (1, 0)
```

This proves the local orbit gauge can absorb bounded support drift while the
visibility certificate still forces repair.

## Gate

Focused projective plus interval-gated trainer suite:

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

120 passed in 24.07s
```

## Decision Implication

Do not phrase the method as "throw away charts/fallbacks." The better formalism
is:

```text
gauge domains carry smooth camera-program geometry;
event certificates govern support and visibility;
fallback is reserved for uncertified event regions.
```

Next falsification target: repeat the support/visibility debounce on broader
multi-trace orbit scenes where support drift, order flips, and tile-capacity
pressure occur together.
