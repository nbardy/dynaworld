# Revolving Camera Fiber Metric Gate

## Context

The current active goal is not just "make STAR UVT pass local tests." It is:

```text
goal       fast 2D rasters across time from 4D spacetime primitives
meta-goal  share projection/support/binning/visibility/backward over time
key math   UVT trace = pi_* Gamma^* world_primitive
theory     STAR UVT is one local gauge expression of a camera-ray bundle atlas
```

After adding trainable rotated UV precision, the next missing evidence was
whether a revolving camera actually exercises that fiber metric in the
world-to-screen compiler.

## Implementation

Added an orbit-facing test to
`tests/test_star_uvt_projective_orbit_windows.py`:

```text
test_revolving_camera_segments_carry_rotated_spd_uv_fiber_metric
```

The test builds a synthetic elevated look-at orbit over 16 frames, with two
anisotropic world tubes and `frames_per_segment=4`. It calls the existing
variable-camera segment compiler:

```text
project_piecewise_camera_time_segments(...)
```

and compares it against `frames_per_segment=1`.

The asserted contract:

```text
charted temporal chunks = 4
per-frame temporal chunks = 16
charted segment count < per-frame segment count
q_uu > 0, q_vv > 0
det(Q_uv) = q_uu q_vv - q_uv^2 > 0
max |q_uv| > 1e-3
first tube q_uv changes sign across the orbit
CPU UVT render is finite and nonzero
```

This is the concrete bridge between the fiber-bundle theory and current code:
the orbit does not require one segment per frame, and each local chart carries
the pulled-back rotated UV metric.

## Verification

Focused orbit metric:

```text
1 passed in 4.45s
```

Full orbit file:

```text
7 passed in 2.18s
```

Syntax/import gate:

```text
py_compile passed for the orbit test and variable-camera harness modules.
```

Broad projective/interval suite:

```text
158 passed in 14.19s
```

## Implication

This does not finish the orbit story. It proves the current compiler carries a
per-chart SPD screen-fiber metric under a revolving camera. The next gate
should quantify image error, fallback fraction, and interval-to-dense ratio
for chart sizes `1, 2, 4, 8, 16`, using the per-frame route as the reference.
