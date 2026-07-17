# Revolving Camera Chart-Size Sweep

## Context

The active goal asks for fast rasters across time from spacetime primitives,
with shared projection/support/binning/visibility/backward work as frame count
grows. The previous orbit gate proved that a revolving camera path carries a
rotated SPD UV fiber metric. The missing next piece was a measured share-vs-
error row.

Pinned memory:

```text
goal       fast 2D rasters across time from 4D spacetime primitives
meta-goal  share projection/support/binning/visibility/backward over time
key math   UVT trace = pi_* Gamma^* world_primitive
theory     STAR UVT is one local gauge expression of a camera-ray bundle atlas
```

## Test Added

Added:

```text
tests/test_star_uvt_projective_orbit_windows.py::test_revolving_camera_chart_size_sweep_quantifies_error_vs_framewise_reference
```

The fixture uses the elevated look-at orbit with two anisotropic world tubes.
It renders a one-segment-per-frame reference and compares charted routes with
`frames_per_segment = 2, 4, 8`.

The executable contract is:

```text
segment ratios = [1.0, 0.5, 0.25, 0.125]
shared routes mean_abs < 0.009
shared routes mse      < 0.0011
shared routes max_abs  < 0.40
```

This is intentionally a measurement gate, not a final paper metric. It catches
regressions where the orbit compiler loses its temporal sharing or where
charted approximation error blows up against the framewise route.

## Observed Sweep

The sampled values before codifying thresholds were approximately:

```text
frames_per_segment  segment_ratio  max_abs   mean_abs  mse
1                   1.000          0.0000    0.0000    0.000000
2                   0.500          0.2852    0.0085    0.000986
4                   0.250          0.3700    0.0073    0.000945
8                   0.125          0.3845    0.0062    0.000802
```

## Verification

```text
focused sweep: 1 passed in 10.70s
orbit file: 8 passed in 14.11s
py_compile: passed
broad projective/interval suite: 160 passed in 26.11s
```

## Implication

This moves the revolving-camera lane from a pure representational claim to a
measured amortization claim: fewer camera-projection chart segments across time
with bounded image error on the synthetic orbit. The next useful benchmark is
the same sweep through the Metal interval/fallback path, reporting fallback
fraction and interval-to-dense ratio.
