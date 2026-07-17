# Gauged UVT Goal Memory Refresh

Date: 2026-05-24 11:47:36 +0700

## Context

The user explicitly asked to keep the goal, meta-goals, key math, and theory in
memory for the Gauged UVT Trace Atlas / STAR UVT / WorldFoam camera-path
compiler thread.

This note records the durable memory refresh after the span-gated Metal bridge
state was already present.

## Memory Pins

The compact memory file is:

```text
research_notes/gauged_uvt_trace_atlas/GOAL_META_KEY_MATH.md
```

It now starts with a `Memory Contract` that keeps these anchors together:

```text
goal       fast 2D rasters across time from 4D spacetime primitives
meta-goal  share projection/support/binning/visibility/backward work over time
key math   UVT trace = pi_* Gamma^* world_primitive
theory     STAR UVT is one local gauge chart of a camera-ray bundle atlas
```

The first-read theory contract is:

```text
research_notes/gauged_uvt_trace_atlas/00_WHAT_IS_THIS_GOAL.md
```

The folder index is:

```text
research_notes/gauged_uvt_trace_atlas/README.md
```

Both were patched so they no longer understate the implementation as only a
Gate A/B projective evaluator. They now mention support bounds, visibility
sidecars, visible-swap bounds, tile-time records/cells, CPU atlas reference
rendering, the q-UVT bridge, explicit interval gates, and the span-gated Metal
wrapper.

## Current Verified State

Focused projective tests pass:

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
33 passed in 1.94s
```

## Next Gate

Do not mark the active objective complete yet. The next useful implementation
gate remains either:

```text
native shader-side interval gate buffers for q-UVT tubes
```

or:

```text
the first nonlinear/projective atlas-cell Metal evaluator
```

The objective is not done until a real renderer path shows clean
forward/backward behavior and useful sublinear non-pixel world-side scaling on
a multi-frame orbit, rolling-shutter, or finite-exposure workload.
