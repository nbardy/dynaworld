# Revolving Camera Interval Backward

## Context

The active objective explicitly asks for clean derivatives and shared backward
work across time, not just a fast forward renderer. The previous orbit work
proved interval compression and zero fallback on a synthetic revolving camera
fixture. This pass adds a backward gate through the same interval atlas path.

Pinned memory:

```text
goal       fast 2D rasters across time from 4D spacetime primitives
meta-goal  share projection/support/binning/visibility/backward over time
key math   UVT trace = pi_* Gamma^* world_primitive
theory     STAR UVT is one local gauge expression of a camera-ray bundle atlas
```

## Test Added

Added to `tests/test_star_uvt_projective_orbit_windows.py`:

```text
test_revolving_camera_interval_backward_reaches_orbit_uvt_trace_params_if_available
```

The test uses the elevated look-at orbit fixture with `frames = 4` and
`frames_per_segment = 2`. It marks the projected chart tensors differentiable,
lowers them into the interval atlas, renders through the Metal autograd bridge,
and backprops an asymmetric image loss.

## Protected Gradient Contract

The reusable atlas topology stays static, but gradients must reach:

```text
ma
opacity
color
q_uu, q_uv, q_vv
q_ut, q_vt, q_tt
```

The key new assertion is nonzero gradient on `q_uv`, the rotated screen-fiber
cross term induced by the revolving camera.

## Verification

```text
focused backward: 1 passed in 3.37s
orbit file: 12 passed in 18.07s
py_compile: passed
broad projective/interval suite: 164 passed in 28.66s
```

## Implication

This moves the orbit lane from forward amortization to differentiable
amortization. The interval atlas can share support/binning/visibility topology
while preserving gradients into the orbit-derived trace parameters, including
the full SPD UV fiber metric.
