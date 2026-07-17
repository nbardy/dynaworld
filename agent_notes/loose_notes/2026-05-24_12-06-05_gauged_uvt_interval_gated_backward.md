# Gauged UVT Interval-Gated Backward

Date: 2026-05-24 12:06:05 +0700

## Context

The active objective asks for clean derivatives and backward-pass reuse across
time, not only forward rendering. The previous gate added native shader-side
intervals for q-UVT forward rendering:

```text
render_gated / render_uvt_tubes_gated
```

This pass adds the matching direct VJP surface for split affine chart segments.

## Change

Added:

```text
torch.ops.star_uvt_v0.direct_atomic_backward_gated(...)
direct_atomic_backward_gated(...)
```

The op takes the same per-tube int32 intervals as the gated forward path:

```text
[active_start, active_stop)
```

and uses them in both places that matter:

```text
bin_screen_tubes_to_uvt_tiles_gated
direct_atomic_backward_gated
```

The binning step clamps each tube to its active frame interval. The VJP kernel
also skips inactive tubes per sample, including in the unstable per-sample
depth-order path.

## Files

```text
third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/metal/star_uvt_kernels.metal
third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/metal/star_uvt_metal.mm
third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/bindings.cpp
third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/shared/common.h
third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/rasterize.py
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

Focused backward test:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_correctness.py::test_q_uvt_native_interval_gated_backward_matches_single_tube_references_if_available -q
```

Result:

```text
1 passed in 3.97s
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
35 passed in 7.86s
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

## Meaning

Degree-1 projective chart windows now have a native q-UVT Metal forward and
direct-backward surface:

```text
projective chart windows
  -> q-UVT tubes with active intervals
  -> render_gated
  -> direct_atomic_backward_gated
```

This advances the clean-derivative condition for the camera-path compiler: chart
domains are now respected by both rendering and VJP accumulation.

## Next Gate

The active objective is still not complete. The next aligned gate is one of:

```text
nonlinear/projective atlas-cell Metal evaluator
```

or:

```text
end-to-end training smoke using interval-gated q-UVT trace segments
```

The full objective still needs measured sublinear non-pixel world-side scaling
and a useful multi-frame orbit, rolling-shutter, or finite-exposure workload.
