# Gauged UVT Tile-Time Binning

Date: 2026-05-24 06:15:18

## Context

The prior gate added visible-swap bounds for ambiguous crossing pairs. The next
planned step was wiring accepted rational/projective windows into a
compiler-side binning prototype.

This pass remains CPU/Torch compiler-side. It does not edit Metal kernels or
the renderer hot path.

## What Changed

Extended:

```text
third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/projective_trace.py
```

with:

```text
ProjectiveTraceTileTimeRecord
bin_projective_trace_support_bounds(bounds, image_width, image_height, tile_size)
```

The binning helper consumes accepted `ProjectiveTraceSupportBounds` and emits
compressed records:

```text
primitive_id
window_index
start/stop
tile_u_min/tile_u_max
tile_v_min/tile_v_max
depth_min/depth_max
fallback
fallback_reason
```

It skips offscreen supports, preserves custom primitive ids, and carries an
optional fallback mask/reason from visibility classification.

Updated exports in:

```text
third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/__init__.py
```

Added:

```text
tests/test_star_uvt_projective_binning.py
```

The tests prove:

- visible supports produce compressed tile-rectangle/time-window records
- offscreen supports are skipped
- custom primitive ids are preserved
- fallback masks are carried into records

## Tests

Focused projective suite:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_trace.py \
  tests/test_star_uvt_projective_orbit_windows.py \
  tests/test_star_uvt_projective_visibility.py \
  tests/test_star_uvt_projective_binning.py -q
```

Result:

```text
24 passed in 1.15s
```

## Current Model

The projective compiler now has these local pieces:

```text
trace fit -> windows -> support bounds -> visibility/swap masks -> tile-time records
```

This is the first concrete shape of a Sensor-Time Trace Atlas index, but it is
still a list of compressed records. It does not yet assemble per-tile active
sets, order graphs, or fallback masks.

## Next Gate

Add a compiler-side tile-time atlas assembly test:

```text
tile-time records + depth order + swap fallback flags
    -> per-tile active primitive lists / order metadata / fallback mask
```

Keep it synthetic and CPU/Torch. Do not touch renderer integration until this
atlas assembly contract is clear.
