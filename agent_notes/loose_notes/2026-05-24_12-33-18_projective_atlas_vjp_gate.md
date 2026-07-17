# Projective Atlas Direct VJP Gate

## Context

The previous step added a native Metal forward renderer for packed
projective/rational atlas cells. That closed the "can nonlinear camera-gauge
cells render without affine q-UVT lowering?" gate, but it left the clean
derivative condition open.

## What Changed

Added a native direct VJP for packed projective atlas cells:

```text
torch.ops.star_uvt_v0.direct_projective_trace_backward(...)
direct_backward_projective_trace_tile_time_atlas_metal(...)
ProjectiveTraceAtlasGrad
```

The backward kernel replays the compiled tile-time cell candidates per sample,
sorts by per-sample projective depth, and differentiates the local footprint:

```text
u(t) = h_u(t) / h_z(t)
v(t) = h_v(t) / h_z(t)
alpha = opacity * exp(-0.5 * ||pixel - (u(t), v(t))||^2 / sigma^2)
```

It accumulates direct atomic gradients for:

```text
grad_coeffs   [N, 9]  for h_u, h_v, h_z quadratic coefficients
grad_opacity  [N]
grad_color    [N, 3]
```

Visibility order, tile membership, and support intervals are treated as compiled
constants. This matches the current q-UVT direct-backward convention and is the
right first gate before differentiating compiler decisions.

## Tests

Rebuilt the extension:

```bash
( cd third_party/fast-mac-gsplat/variants/star_uvt_v0
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

New VJP parity gate:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_correctness.py::test_projective_quadratic_atlas_cell_backward_matches_torch_autograd_if_available -q
```

Result:

```text
1 passed
```

Focused forward/backward projective atlas gates:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_correctness.py::test_projective_tile_time_bins_preserve_split_window_intervals \
  tests/test_star_uvt_projective_correctness.py::test_projective_quadratic_atlas_cells_render_in_metal_if_available \
  tests/test_star_uvt_projective_correctness.py::test_projective_quadratic_atlas_cell_backward_matches_torch_autograd_if_available -q
```

Result:

```text
3 passed
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
39 passed
```

Existing q-UVT smoke still passes:

```text
max_rgb_error = 5.960464477539063e-08
overflow_tile_count = 0
pair_ratio = 0.5
```

## Current Model

The projective atlas now has the first forward/backward local calculus:

```text
packed tile-time cells
    -> Metal projective forward
    -> Metal direct VJP for opacity/color/homogeneous coefficients
```

This is a meaningful clean-derivative step, but not yet the full objective:
there is still no real trainer route, no projective atlas trainability smoke,
and no scaling benchmark showing sublinear world-side growth on an orbit.

## Next Gates

1. Projective atlas-cell trainability smoke: render target, apply VJP, update
   color and/or projective coefficients, verify loss drops.
2. Integrate projective/q-UVT trace atlas forward/backward into a real trainer
   path.
3. Add frame-count scaling microbenchmarks for packed projective atlas cells.
4. Start the WorldFoam bridge by packing foam cell-camera intersections into
   the same tile-time cell contract.
