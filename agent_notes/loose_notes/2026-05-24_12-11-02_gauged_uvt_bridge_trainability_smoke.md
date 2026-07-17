# Gauged UVT Bridge Trainability Smoke

Date: 2026-05-24 12:11:02 +0700

## Context

The active objective includes clean derivatives and backward-pass reuse across
time. The previous gate added native interval-gated q-UVT forward rendering and
direct atomic VJP. This pass turns that into a minimal optimization smoke at the
projective bridge level.

## Change

Added bridge-level VJP wrapper:

```text
direct_backward_projective_trace_uvt_bridge_metal_gated(...)
ProjectiveTraceUVTBridgeGrad
```

It wraps:

```text
direct_atomic_backward_gated(...)
```

and returns gradients for the interval-gated q-UVT bridge tensors. The current
direct VJP surface provides geometry/color/opacity gradients from the q-UVT
kernel and zero placeholders for depth fields, matching the existing direct
backward contract.

Added a trainability smoke:

```text
test_projective_interval_gated_bridge_one_step_color_training_smoke_if_available
```

The test:

1. Builds split projective chart windows.
2. Lowers them into q-UVT tubes with active intervals.
3. Renders a target through native gated Metal.
4. Renders a current bridge with different colors.
5. Computes image MSE gradient.
6. Calls `direct_backward_projective_trace_uvt_bridge_metal_gated`.
7. Takes one color update.
8. Verifies loss decreases.

This is still not full trainer integration, but it proves the interval-gated
bridge can drive a real optimization step without chart-domain leakage.

## Verification

Single training-smoke gate:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_correctness.py::test_projective_interval_gated_bridge_one_step_color_training_smoke_if_available -q
```

Result:

```text
1 passed in 2.63s
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
36 passed in 4.42s
```

Renderer smoke:

```bash
PYTHONPATH=src/train uv run python \
  third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/benchmarks/uvt_pair_benchmark.py \
  --scenes single_static
```

Key result:

```text
max_rgb_error = 5.960464477539063e-08
mean_rgb_error = 1.1123878485008731e-09
overflow_tile_count = 0
unstable_tile_fraction = 0.0
```

## Next Gate

The active objective remains open. The next aligned gates are:

```text
nonlinear/projective atlas-cell Metal evaluator
```

or:

```text
real trainer integration for interval-gated q-UVT trace segments
```

The method still needs measured sublinear non-pixel world-side scaling on a
useful multi-frame orbit, rolling-shutter, or finite-exposure workload before
the objective can be considered complete.
