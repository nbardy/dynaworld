# Gauged UVT atlas reference renderer

## Context

After the dense-reference atlas coverage/order gate passed, the next gap was
pixel-level evaluation. A compiler can have the right cells and still render
wrong if opacity evaluation, depth order, primitive-id mapping, or tile/sample
lookup are inconsistent.

## Work Changed

Added a CPU/Torch oracle:

```text
render_projective_trace_tile_time_atlas_reference(...)
```

Location:

```text
third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/projective_trace.py
```

It consumes `ProjectiveTraceTileTimeCell` records, maps atlas primitive ids back
to primitive indices, evaluates dense projective centers, then composites
front-to-back inside each tile/time cell with a simple isotropic screen-space
Gaussian opacity model.

It is exported from:

```text
third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/__init__.py
```

## New Test

Extended:

```text
tests/test_star_uvt_projective_correctness.py
```

with:

```text
test_projective_atlas_reference_renderer_matches_dense_per_frame_compositing
```

The test compiles two stable-depth projective primitives into a tile-time atlas,
renders through the atlas reference helper, renders a dense per-frame reference
with the same opacity law, and checks pixel equality.

## Evidence

Focused suite:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_trace.py \
  tests/test_star_uvt_projective_orbit_windows.py \
  tests/test_star_uvt_projective_visibility.py \
  tests/test_star_uvt_projective_binning.py \
  tests/test_star_uvt_projective_correctness.py -q
```

Result:

```text
29 passed in 0.62s
```

## Current Model

The projective compiler path now has both structural and pixel-level CPU
reference tests:

```text
trace eval -> chart windows -> bounds -> binning -> atlas cells -> CPU atlas render
```

This is still not Metal integration. It is the oracle that should make Metal
integration less slippery.

## Next Gate

Port this evaluation contract into a guarded Metal/hot renderer path, or first
wire the atlas candidate/order data through the existing STAR UVT `q_uvt`
renderer mode if that is the smaller bridge. Keep the CPU reference as the
acceptance target.
