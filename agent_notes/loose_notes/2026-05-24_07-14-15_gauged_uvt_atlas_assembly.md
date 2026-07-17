# Gauged UVT Atlas Assembly

Date: 2026-05-24 07:14:15

## Context

The prior gate added compiler-side tile-time binning records. The next planned
gate was assembling those compressed records into actual tile-time atlas cells
with active sets, order metadata, and fallback masks.

This pass remains CPU-side. It does not touch Metal kernels or renderer hot
paths.

## What Changed

Extended:

```text
third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/projective_trace.py
```

with:

```text
ProjectiveTraceTileTimeCell
assemble_projective_trace_tile_time_atlas(records)
```

The assembler expands compressed tile-rectangle records into concrete tile-time
cells keyed by:

```text
tile_u, tile_v, start, stop
```

Each cell stores:

```text
primitive_ids
ordered_primitive_ids
depth_intervals
fallback
fallback_reasons
```

Order is currently a simple midpoint-depth sort. That is appropriate for this
compiler prototype because prior gates already mark ambiguous crossing pairs
with fallback flags.

Updated exports in:

```text
third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/__init__.py
```

Extended:

```text
tests/test_star_uvt_projective_binning.py
```

with tests proving:

- active primitive ids are grouped per tile-time cell
- depth order metadata is sorted front to back
- fallback reasons are preserved
- tile rectangles expand into multiple cells

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
26 passed in 1.22s
```

## Current Model

The local compiler stack now has:

```text
projective trace eval
chart fit / split
support bounds
visibility sidecars
visible-swap fallback
tile-time compressed records
tile-time atlas cells
```

This is enough structure to start the next correctness gate: compare atlas
evaluation behavior against dense per-frame/per-sample projective reference on
synthetic orbit scenes.

## Next Gate

Add synthetic orbit correctness tests against dense per-frame/per-ray reference.
Keep them compiler-side first. Only after correctness is crisp should this feed
a renderer hot path.
