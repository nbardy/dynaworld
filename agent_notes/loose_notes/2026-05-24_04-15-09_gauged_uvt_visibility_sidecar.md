# Gauged UVT Visibility Sidecar

Date: 2026-05-24 04:15:09

## Context

The prior gate added projective support bounds for accepted chart windows. The
next planned gate was denominator/depth sidecars for visibility plus a synthetic
crossing-strata test.

This pass remains compiler-side. No renderer hot path or Metal shader was
edited.

## What Changed

Extended:

```text
third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/projective_trace.py
```

with:

```text
ProjectiveTraceVisibilitySidecar
ProjectiveTraceDepthOrder
make_projective_trace_visibility_sidecar(window)
make_projective_trace_visibility_sidecars(windows)
compare_projective_trace_depth_order(sidecar_a, sidecar_b)
```

The sidecar records:

```text
chart_gauge_id
time_min/time_max
depth_coeffs
depth_min/depth_max
depth_slope_min/depth_slope_max
depth_monotonic_sign
depth_uncertainty
denominator_min_abs
denominator_has_root
```

The depth-order comparator assumes smaller depth is "before" in the current
depth gauge. It returns:

```text
a_before_b
b_before_a
crosses
ambiguous
```

where `crosses` means the fitted depth-difference polynomial has a root inside
the normalized chart interval.

Updated package exports in:

```text
third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/__init__.py
```

Added:

```text
tests/test_star_uvt_projective_visibility.py
```

The tests cover:

- monotone depth metadata
- stable front/back order
- crossing depth traces marked as ambiguous visibility strata

## Tests

Focused projective suite:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_trace.py \
  tests/test_star_uvt_projective_orbit_windows.py \
  tests/test_star_uvt_projective_visibility.py -q
```

Result:

```text
19 passed in 1.15s
```

## Current Model

The visibility compiler now has the minimum local metadata to distinguish:

```text
stable order
crossing stratum
ambiguous order requiring either swap-bound acceptance or fallback
```

It still does not know if an ambiguous crossing is visually important. That
requires opacity/color sidecars and the visible-swap bound:

```text
|Delta I_ij| <= alpha_i alpha_j |c_i - c_j|
```

## Next Gate

Add visible-swap cost bounds for ambiguous crossing pairs. After that, wire
accepted projective/rational windows into a compiler-side binning prototype.
