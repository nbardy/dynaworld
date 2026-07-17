# Gauged UVT Native Interval Gate

Date: 2026-05-24 11:58:44 +0700

## Context

The active Gauged UVT Trace Atlas goal needs 4D spacetime primitives compiled
through a camera program into reusable UVT traces, with projection/support/
binning/visibility/backward work shared across time. The previous bridge proved
split affine chart segments could avoid leakage by partitioning the frame axis
outside the shader and calling the existing q-UVT Metal renderer per span.

That was correct but still not the native interval-gate contract we want.

## Change

Added a native q-UVT interval-gated Metal render path:

```text
torch.ops.star_uvt_v0.render_gated(...)
render_uvt_tubes_gated(...)
```

The new path passes per-tube int32 sample-domain intervals:

```text
[active_start, active_stop)
```

into the Metal renderer.

Metal kernels:

```text
bin_screen_tubes_to_uvt_tiles_gated
render_uvt_tiles_gated
```

The gated binning kernel clamps each tube's tile-time support to its active
frame interval. The gated render kernel also checks the interval per sample, so
split chart segments do not leak inside a multi-frame tile.

The projective bridge:

```text
render_projective_trace_uvt_bridge_metal_gated(...)
```

now calls the native gated renderer instead of doing external span partitioning.

## Files

```text
third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/metal/star_uvt_kernels.metal
third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/metal/star_uvt_metal.mm
third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/bindings.cpp
third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/shared/common.h
third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/rasterize.py
third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/projective_trace.py
third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/__init__.py
tests/test_star_uvt_projective_correctness.py
```

Docs refreshed:

```text
research_notes/gauged_uvt_trace_atlas/README.md
research_notes/gauged_uvt_trace_atlas/00_WHAT_IS_THIS_GOAL.md
research_notes/gauged_uvt_trace_atlas/GOAL_META_KEY_MATH.md
research_notes/gauged_uvt_trace_atlas/09_metal_acceptance_plan/README.md
research_notes/gauged_uvt_trace_atlas/clean_thread_handoff/README.md
```

## Verification

Rebuilt the STAR UVT extension:

```bash
( cd third_party/fast-mac-gsplat/variants/star_uvt_v0
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld \
    python setup.py build_ext --inplace )
```

Focused native-gate tests:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_correctness.py::test_q_uvt_native_interval_gates_match_cpu_reference_if_available \
  tests/test_star_uvt_projective_correctness.py::test_projective_split_q_uvt_bridge_interval_gates_reach_metal_if_available -q
```

Result:

```text
2 passed in 4.50s
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
34 passed in 2.97s
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

## Current Model

Degree-1 projective chart windows now have a real Metal path:

```text
projective chart windows
  -> q-UVT tubes
  -> native interval-gated q-UVT Metal render
```

This is aligned with the goal because chart domains are now a shader-side
rendering primitive rather than an external loop over spans.

## Next Gate

The next implementation gate is one of:

```text
first nonlinear/projective atlas-cell Metal evaluator
```

or:

```text
backward coverage for interval-gated q-UVT segments
```

The full active objective remains open until forward/backward behavior and
sublinear non-pixel world-side scaling are proven on a useful multi-frame orbit,
rolling-shutter, or finite-exposure workload.
