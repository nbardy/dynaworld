# Gauged UVT dense-reference correctness gate

## Context

The previous Gauged UVT work had reached the compiler-side atlas assembly gate:
projective rational traces can be split into accepted chart windows, bounded in
UV/depth, annotated with visibility sidecars and visible-swap costs, binned into
compressed tile-time records, and expanded into tile-time atlas cells.

The stale next goal in the handoff docs was to prove those cells agree with a
dense per-frame reference instead of only proving their internal structure.

## Work Changed

Added:

```text
tests/test_star_uvt_projective_correctness.py
```

The test file covers two contracts:

```text
test_projective_atlas_covers_dense_orbit_projection_reference
test_projective_atlas_depth_order_matches_dense_stable_reference
```

The first contract builds a synthetic yaw-orbit projective trace, compiles it
through window splitting, support bounds, tile-time binning, and atlas assembly,
then checks every valid dense per-frame projected sample lands in a tile-time
cell containing the expected primitive id.

The second contract builds two stable-depth traces, compiles them into an
atlas, and checks the atlas cell front-to-back order matches dense per-frame
depth sorting for every time sample.

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
28 passed in 0.98s
```

## Current Model

The compiler path now has a tested chain:

```text
projective trace eval
-> local chart windows
-> support bounds
-> visibility / visible-swap metadata
-> tile-time binning
-> atlas assembly
-> dense-reference coverage and stable-depth order checks
```

This does not yet prove renderer quality. It proves the compiled atlas is not
obviously losing dense per-frame projective samples or stable order metadata on
the synthetic orbit case.

## Next Gate

Build a minimal atlas evaluator / renderer-quality comparison against dense
per-frame reference rendering. That should come before a full Metal hot-path
renderer rewrite, because it will expose whether the atlas evaluation contract
is missing opacity, footprint, or ordering details.
