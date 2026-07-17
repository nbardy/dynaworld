# Revolving Camera Backward Frame Densification

## Context

After adding the forward work-unit gate for the revolving-camera lane, the
remaining active-goal pressure was the backward phrase:

```text
share compute and memory bandwidth and backwards passes maximally across time
```

The existing orbit backward test proved a single 4-frame orbit route could
backprop through the interval Metal autograd bridge into chart centers, opacity,
color, and all `q_uvt` terms. It did not yet check what happens when frame
samples are densified while the chart parameter set remains fixed.

## Test Added

File:

```text
tests/test_star_uvt_projective_orbit_windows.py
```

New helper:

```text
_render_differentiable_orbit_interval_metal(...)
```

New test:

```text
test_revolving_camera_interval_backward_keeps_fixed_chart_params_when_frames_grow_if_available
```

The helper also simplified the existing orbit interval backward test so both
tests use the same differentiable Metal setup.

## Setup

```text
frames = 4, 8
temporal charts per tube = 2
tube_count = 2
frames_per_segment = frames / 2
```

Both frame counts compile to:

```text
charted segments = 4
atlas traces     = 4
```

Then the test renders through:

```text
render_projective_cell_interval_atlas_metal_backward(...)
```

and backprops an asymmetric time-weighted image loss.

## Protected Gradient Contract

For both frame counts, the test requires nonzero gradients for:

```text
ma
opacity
color
q_uu, q_uv, q_vv
q_uv
q_ut, q_vt, q_tt
```

This protects the derivative topology: increasing output samples does not force
the orbit route to allocate new per-frame chart parameters for gradients to
land on.

## Observed Dry Run

Before writing the test, the local probe printed:

```text
frames  segments  traces  sum|grad_q_uvt|  sum|grad_q_uv|  sum|grad_color|
4       4         4       18.8213          2.4203          6.4030
8       4         4       54.7883          10.4217         17.1237
```

The gradient magnitudes differ because there are more image samples and a
time-weighted loss, but the parameter count stays fixed.

## Verification

```text
focused backward gates:
    2 passed in 4.28s

full orbit file:
    14 passed in 30.73s

py_compile:
    passed

broad projective/interval suite:
    166 passed in 49.64s
```

## Current Interpretation

Supported:

```text
STAR UVT can hold a fixed local gauge atlas for a smooth orbit while frame
samples densify, and the interval Metal VJP still reaches the same trace
parameters.
```

Not yet proved:

```text
end-to-end backward wall time is sublinear in frame count on real scenes.
```

Next falsification step:

```text
run the same fixed-chart backward probe as a measured artifact with 4/8/16/32
frames, record interval entries, fallback fraction, forward time, backward
time, and compare against a per-frame projection/sort baseline.
```
