# Revolving Camera Frame-Growth Work Units

## Context

The active goal is fast 2D rasters across time from 4D spacetime primitives,
with shared projection/support/binning/visibility/backward work over time. The
recent orbit lane already had:

- a variable-camera fiber-metric gate,
- a chart-size image-error sweep,
- interval atlas compression and Metal forward parity,
- an interval Metal backward gate into orbit-derived `q_uvt` trace parameters.

The missing certificate was the frame-growth condition itself for the newer
revolving-camera UVT route.

## Current Model

STAR UVT should not claim that materialized video has sublinear pixel cost.
Writing `F * H * W` samples is still linear in frames. The useful claim is:

```text
world-side projection/support/binning/visibility metadata grows with trace
complexity, not directly with requested frame count
```

For a smooth revolving camera, a fixed number of orbit charts can cover a
larger number of output frame slices. Densifying frame samples should increase
dense per-frame work much faster than interval atlas work.

## Test Added

File:

```text
tests/test_star_uvt_projective_orbit_windows.py
```

Test:

```text
test_revolving_camera_fixed_chart_count_keeps_world_side_work_sublinear_with_frame_growth
```

Setup:

```text
frames = 8, 16, 32
tube_count = 2
fixed temporal charts per tube = 4
frames_per_segment = frames / 4
image = 32 x 32
tile_size = 8
```

Reference comparison:

```text
per-frame route: frames_per_segment = 1
charted route:   frames_per_segment = frames / 4
```

## Observed Work Units

```text
frames  per_frame_segments  charted_segments  atlas_traces  interval_entries  dense_samples  interval_ratio
8       16                  8                 8             99                156            0.6346
16      32                  8                 8             135               366            0.3689
32      64                  8                 8             156               820            0.1902
```

Fallback fraction stayed `0.0` for all rows.

## Interpretation

Observed fact:

```text
per-frame segment count grows 4x from 8 to 32 frames.
dense trace samples grow more than 5x.
compiled chart count and atlas trace count remain fixed.
interval entries grow less than 2x.
```

Inference:

```text
the orbit compiler is sharing world-side trace construction over frame growth
on this smooth revolving-camera fixture.
```

This is exactly the distinction the theory needs:

```text
pixel work remains O(FHW)
world-side compiled trace work follows chart/event complexity
```

## Falsification

This gate would fail if:

- chart segmentation silently fell back to one chart per frame,
- interval atlas lowering duplicated traces per sampled frame,
- visibility/support event splitting exploded with frame count,
- fallback appeared on the smooth orbit fixture.

The current thresholds protect those failure modes by checking fixed segment
counts, fixed atlas trace counts, sub-2x interval-entry growth over 4x frames,
monotone interval-ratio decrease, and zero fallback.

## Verification

```text
focused frame-growth gate:
    1 passed in 12.13s

full orbit file:
    13 passed in 33.10s

py_compile:
    passed

broad projective/interval suite:
    165 passed in 48.65s
```

## Next Implication

The next stronger version should move from this synthetic orbit to extracted
high-motion real-view trace geometry and report:

```text
frame count
chart count
support-event cell count
visibility-stratum split count
fallback fraction
image error versus per-frame projection
Metal forward/backward time
```

The mathematical target remains:

```text
UVT trace = pi_* Gamma^* world_primitive
```

with STAR UVT as one local gauge expression of the camera-ray bundle atlas.
