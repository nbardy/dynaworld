# Revolving Camera Interval Atlas Sweep

## Context

The active goal is to compile spacetime primitives through a camera program so
projection/support/binning/visibility/backward can be shared across time. The
previous orbit chart-size sweep only compared CPU UVT renders. This pass moves
the same orbit family through the interval atlas object and the Metal interval
renderer.

Pinned memory:

```text
goal       fast 2D rasters across time from 4D spacetime primitives
meta-goal  share projection/support/binning/visibility/backward over time
key math   UVT trace = pi_* Gamma^* world_primitive
theory     STAR UVT is one local gauge expression of a camera-ray bundle atlas
```

## Tests Added

Added to `tests/test_star_uvt_projective_orbit_windows.py`:

```text
test_revolving_camera_interval_atlas_sweep_reports_compression_and_zero_fallback
test_revolving_camera_interval_metal_matches_reference_if_available
```

The first lowers orbit chart segments through:

```text
uvt_tubes_to_projective_trace_cell_atlas(...)
```

with anisotropic spatial precision enabled, auto support padding from alpha,
support-event rebinning, and visibility stratification.

The second runs the same orbit family through:

```text
render_projective_trace_cell_interval_atlas_metal(...)
```

when MPS and the native op are available.

## Protected Contract

For `frames = 8` and `frames_per_segment = 1, 2, 4, 8`:

```text
trace counts = [16, 8, 4, 2]
fallback fractions = [0.0, 0.0, 0.0, 0.0]
interval ratios decrease monotonically
first interval ratio = 1.0
last interval ratio < 0.35
```

The atlas reference stays close to the charted UVT render:

```text
mean_abs < 3e-5
max_abs  < 0.02
```

The sampled rows before locking thresholds were:

```text
frames_per_segment  trace_count  fallback  interval_ratio
1                   16           0.0       1.000
2                   8            0.0       0.635
4                   4            0.0       0.392
8                   2            0.0       0.310
```

## Verification

```text
focused interval tests: 2 passed in 10.42s
orbit file: 11 passed in 9.96s
py_compile: passed
broad projective/interval suite: 163 passed in 33.16s
```

## Implication

This is a stronger sublinear-growth proxy than the image-only sweep. It proves
the reusable tile-time atlas itself compresses orbit traces while preserving
zero fallback on the fixture, and the Metal interval forward can consume the
orbit-derived anisotropic traces.
