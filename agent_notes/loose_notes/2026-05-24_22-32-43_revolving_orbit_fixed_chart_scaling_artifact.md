# Revolving Orbit Fixed-Chart Scaling Artifact

## Context

The active goal asks for fast rasterization across time from 4D spacetime
primitives, with shared projection/support/binning/visibility/backward work and
sublinear growth in the cost of frames where possible.

The preceding tests proved:

- fixed orbit charts keep chart/trace counts constant as frame samples grow,
- interval Metal VJP still reaches the fixed orbit chart parameters.

This session added a measured artifact script to make those claims visible in a
repeatable report.

## Added Script

```text
research_experiments/star_uvt_feature_tubes/projective_orbit_fixed_chart_scaling_benchmark.py
```

The script compares:

```text
fixed_chart:
    four temporal orbit charts per tube

per_frame:
    one chart per frame
```

It records:

```text
segment count
trace count
cell count
interval trace entries
dense trace samples
interval/dense ratio
fallback fraction
atlas tensor payload bytes
interval Metal forward time
interval Metal direct backward time
fixed-chart autograd topology gradients into q_uv and temporal q_uvt
```

The script now performs a discarded Metal prewarm before recording rows so the
first measured fixed-chart row does not absorb first-kernel compilation noise.

## Artifact

```text
outputs/benchmarks/2026-05-24_star_uvt_revolving_orbit_fixed_chart_scaling/summary.json
outputs/benchmarks/2026-05-24_star_uvt_revolving_orbit_fixed_chart_scaling/summary.md
```

Command:

```bash
PYTHONPATH=src/train uv run python \
  research_experiments/star_uvt_feature_tubes/projective_orbit_fixed_chart_scaling_benchmark.py \
  --frame-counts 4,8,16,32 \
  --iterations 5 \
  --warmup 3 \
  --out-dir outputs/benchmarks/2026-05-24_star_uvt_revolving_orbit_fixed_chart_scaling
```

## Key Rows

Fixed chart:

```text
frames          4      8       16      32
segments        8      8       8       8
traces          8      8       8       8
payload bytes   608    608     608     608
interval ratio  1.000  0.635   0.369   0.190
fallback        0      0       0       0
```

Per-frame:

```text
frames          4      8       16      32
segments        8      16      32      64
traces          8      16      32      64
payload bytes   608    1216    2432    4864
```

After adding compile-phase timing, the saved artifact reports:

```text
route        project_ms 4->32    atlas_build_ms 4->32    cpu_compile_ms 4->32
fixed_chart 8.06 -> 4.38       19.73 -> 32.25        27.80 -> 36.64
per_frame   3.98 -> 35.09      17.95 -> 261.14       21.93 -> 296.22
```

At 32 frames:

```text
fixed/per_frame compile ratio  = 0.124
fixed/per_frame trace ratio    = 0.125
fixed/per_frame payload ratio  = 0.125
fixed/per_frame forward ratio  = 0.153
fixed/per_frame backward ratio = 0.267
```

## Interpretation

Strong evidence:

```text
The fixed-chart orbit atlas avoids per-frame trace and payload growth.
The interval entries grow with support/event complexity rather than frame count.
The fixed-chart route keeps zero fallback on this synthetic orbit.
The fixed-chart autograd topology reaches q_uv and temporal q_uvt at all frame counts.
The fixed-chart route measures the expected compile-side amortization on this fixture.
```

Weaker evidence:

```text
The small MPS timing probe shows gentler growth than per-frame charting, but it
is synthetic, tiny, and not a real-scene training benchmark.
```

Next falsification:

```text
Move the same report shape to extracted high-motion real-view trace geometry
and compare against a per-frame projection/sort baseline, not just per-frame
chart lowering into the same interval renderer.
```

## Verification

```text
py_compile projective_orbit_fixed_chart_scaling_benchmark.py:
    passed

focused orbit frame-growth/backward gates:
    2 passed in 18.19s

full orbit file after final script/docs patch:
    14 passed in 43.52s

py_compile after compile-timing patch:
    passed

focused orbit frame-growth/backward gates after compile-timing patch:
    2 passed in 46.48s
```
