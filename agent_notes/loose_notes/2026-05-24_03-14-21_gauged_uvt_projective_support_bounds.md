# Gauged UVT Projective Support Bounds

Date: 2026-05-24 03:14:21

## Context

The previous Gauged UVT gate proved that projective orbit traces can be split
into accepted local chart windows whose count follows trace complexity rather
than frame density. The next compiler gate was support bounds:

```text
For every accepted rational/projective chart window, sampled u/v/depth traces
stay inside compiled tile-time support bounds.
```

This is still compiler-side work. No Metal shader or renderer hot path was
edited in this pass.

## What Changed

Extended:

```text
third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/projective_trace.py
```

with:

```text
ProjectiveTraceSupportBounds
bound_projective_trace_window(window)
bound_projective_trace_windows(windows)
```

The helper computes continuous polynomial chart extrema over normalized window
time for:

```text
[u(t), v(t), h_z(t)]
```

and inflates:

```text
u/v bounds by residual_max_uv
depth bounds by residual_max_depth
```

It refuses unresolved windows by default, so denominator-boundary windows cannot
silently generate ordinary support bounds.

Updated exports in:

```text
third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/__init__.py
```

Extended:

```text
tests/test_star_uvt_projective_orbit_windows.py
```

with tests proving:

- accepted orbit windows bound all sampled rational UV/depth values
- unresolved denominator-boundary windows raise instead of returning default
  bounds

## Tests

Focused projective trace gate:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_trace.py \
  tests/test_star_uvt_projective_orbit_windows.py -q
```

Result:

```text
16 passed in 1.11s
```

## Current Model

The support-bound helper is the first real bridge from chart validity to
tile-time binning:

```text
accepted projective chart window
    -> polynomial center/depth bound
    -> residual-inflated UV/depth support sidecar
```

This is not full footprint support yet. It bounds the trace center/depth, not
the projected Gaussian covariance or foam-cell extent. The next support step
must add footprint radius/covariance inflation before renderer bin integration.

## Next Gate

Add denominator/depth sidecars for visibility:

```text
denominator_min_abs
depth_uncertainty
depth_monotonicity
chart_gauge_id
```

Then add a synthetic visibility-strata test with two crossing traces before
attempting rational/projective renderer integration.
