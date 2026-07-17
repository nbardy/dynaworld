# Projective Atlas Metal Forward Gate

## Context

Continuation of the Gauged UVT Trace Atlas goal. The previous gate gave split
affine projective chart windows a native interval-gated q-UVT forward/backward
surface. That was useful, but it still forced the richer camera-gauge math
through affine lowering. The next theory-aligned gate was a native forward path
for nonlinear/projective atlas cells.

## What Changed

Added a packed tile-time atlas forward renderer:

```text
pack_projective_trace_tile_time_bins(...)
torch.ops.star_uvt_v0.render_projective_trace_tiles(...)
render_projective_trace_tile_time_atlas_metal(...)
```

The CPU compiler packs `ProjectiveTraceTileTimeCell` records into dense
tile-time buffers:

```text
tile_counts
tile_primitive_ids
tile_active_start
tile_active_stop
tile_overflow
```

The per-entry active intervals are important: a coarse tile-time group may
straddle multiple split chart windows, so a primitive id alone would leak a
chart outside its validity domain. The Metal kernel now checks the packed
interval at the sample frame before considering a candidate.

The shader evaluates the homogeneous projective chart directly:

```text
h_u(t), h_v(t), h_z(t)
u(t) = h_u(t) / h_z(t)
v(t) = h_v(t) / h_z(t)
depth(t) = h_z(t)
```

Then it composites an isotropic screen Gaussian by per-sample projective depth.
This is a forward renderer only. It does not yet differentiate projective
coefficients, support bounds, or visibility decisions.

## Tests

Rebuilt the extension:

```bash
( cd third_party/fast-mac-gsplat/variants/star_uvt_v0
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

Focused new tests:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_correctness.py::test_projective_tile_time_bins_preserve_split_window_intervals \
  tests/test_star_uvt_projective_correctness.py::test_projective_quadratic_atlas_cells_render_in_metal_if_available -q
```

Result:

```text
2 passed
```

Full focused projective suite:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_trace.py \
  tests/test_star_uvt_projective_orbit_windows.py \
  tests/test_star_uvt_projective_visibility.py \
  tests/test_star_uvt_projective_binning.py \
  tests/test_star_uvt_projective_correctness.py -q
```

Result:

```text
38 passed
```

Existing q-UVT smoke:

```bash
PYTHONPATH=src/train uv run python \
  third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/benchmarks/uvt_pair_benchmark.py \
  --scenes single_static
```

Result highlights:

```text
max_rgb_error = 5.960464477539063e-08
overflow_tile_count = 0
pair_ratio = 0.5
```

## Current Model

This is now a real atlas forward path:

```text
projective/rational chart cells
    -> packed tile-time active sets
    -> native Metal per-sample projective evaluation
    -> depth-ordered compositing
```

It moves the implementation closer to the theory commitment that STAR UVT is
one local chart of a camera-ray bundle atlas, not the whole representation.

## Open Next Gates

1. Add projective atlas-cell VJP/gradient coverage for color/opacity first,
   then homogeneous coefficients.
2. Integrate the interval-gated q-UVT forward/backward path into a real trainer
   route.
3. Add a microbenchmark for frame-count scaling of packed projective atlas
   cells versus per-frame dense projection/binning.
4. Bridge WorldFoam cell-camera intersections through the same packed
   tile-time active-set contract.
