# Gauged UVT q-UVT bridge Metal parity

## Context

The affine projective chart to q-UVT bridge passed against the CPU atlas
reference. The next question was whether those lowered tensors can actually use
the existing STAR UVT Metal renderer, rather than only the Python CPU
brute-force renderer.

## Work Changed

Added a guarded MPS test:

```text
test_projective_affine_q_uvt_bridge_matches_metal_renderer_if_available
```

Location:

```text
tests/test_star_uvt_projective_correctness.py
```

The test skips if MPS or the `star_uvt_v0.render` op is unavailable. On this
machine, it did not skip. It lowers two affine projective chart windows through
`projective_trace_windows_to_uvt_tubes(...)`, renders the result with
`render_uvt_tubes(...)` on MPS, and compares to `brute_force_render_uvt_tubes`.

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
31 passed in 0.69s
```

No skip marker appeared in the compact pytest output, so the guarded Metal test
executed.

## Current Model

For accepted degree-1 projective charts, the path is now:

```text
projective camera-gauge chart
-> q-UVT bridge tensors
-> existing STAR UVT Metal renderer
-> CPU q-UVT reference parity
```

This is the first true hot-path bridge for the Gauged UVT idea. It does not yet
handle nonlinear/rational projective charts as a native Metal object.

## Next Gate

Choose one of two next branches:

1. Add a nonlinear/projective atlas-cell Metal evaluator that consumes atlas
   cells and evaluates rational/projective centers directly.
2. Add explicit interval gating for split affine q-UVT chart segments, then
   test whether a revolving camera can be covered by a small number of q-UVT
   chart tubes without leaking outside each window.
