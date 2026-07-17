# Revolving Orbit Compile Timing

## Context

The previous fixed-chart artifact measured structural work units plus Metal
forward/backward timing, but the active theory is specifically about amortizing
world-side work:

```text
projection/support/binning/visibility/backward over time
```

So I extended the benchmark to time the two compile-side phases that are
closest to that claim:

```text
project_ms       = project world tubes through the orbit camera into UVT charts
atlas_build_ms   = lower projected charts into the projective interval atlas
cpu_compile_ms   = project_ms + atlas_build_ms
```

## Code

```text
research_experiments/star_uvt_feature_tubes/projective_orbit_fixed_chart_scaling_benchmark.py
```

New fields per row:

```text
project_ms
atlas_build_ms
cpu_compile_ms
mps_atlas_build_ms
```

New summary fields:

```text
fixed_chart_project_ms_growth
fixed_chart_atlas_build_ms_growth
fixed_chart_cpu_compile_ms_growth
per_frame_project_ms_growth
per_frame_atlas_build_ms_growth
per_frame_cpu_compile_ms_growth
last_fixed_vs_per_frame_cpu_compile_ms_ratio
```

## Artifact

The saved artifact was regenerated:

```text
outputs/benchmarks/2026-05-24_star_uvt_revolving_orbit_fixed_chart_scaling/summary.md
outputs/benchmarks/2026-05-24_star_uvt_revolving_orbit_fixed_chart_scaling/summary.json
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

## Compile Timing Result

```text
route        project_ms 4->32    atlas_build_ms 4->32    cpu_compile_ms 4->32
fixed_chart 8.06 -> 4.38       19.73 -> 32.25        27.80 -> 36.64
per_frame   3.98 -> 35.09      17.95 -> 261.14       21.93 -> 296.22
```

Growth:

```text
fixed_chart compile growth = 1.32x
per_frame compile growth   = 13.51x
```

At `32` frames:

```text
fixed/per_frame compile ratio = 0.124
fixed/per_frame trace ratio   = 0.125
fixed/per_frame payload ratio = 0.125
```

This directly supports the theory's world-side work claim on the synthetic
orbit fixture: fixed-chart compile cost tracks chart/event complexity, while
per-frame compile cost tracks frame count.

## Caveats

The timing is still a small synthetic local MPS/CPU diagnostic. It is not a
real-scene training wall-time proof. The strong result is the structural
work-unit and compile-side gap; the next step is to port the same report shape
to extracted high-motion real-view traces.

## Verification

```text
py_compile projective_orbit_fixed_chart_scaling_benchmark.py:
    passed

focused orbit frame-growth/backward gates:
    2 passed in 46.48s

full orbit file after notes patch:
    14 passed in 74.06s
```
