# Projective Interval Trainer Frame Scaling

## Context

The pinned STAR UVT / gauged trace-atlas goal is to compile world primitives
through a camera program into reusable sensor-time traces, then share
projection/support/binning/visibility/backward work as frame count grows. The
previous strongest evidence was synthetic revolving-orbit fixed-chart evidence:
chart/trace/payload counts stay fixed while dense samples grow.

This pass added a smaller but more production-adjacent artifact: run the actual
`star_uvt_feature_overfit_trainer.run_training(...)` projective-interval route
while monkeypatching the video loader with synthetic frame tensors. That keeps
trainer config normalization, projective interval atlas production, Metal
forward/backward, cache refresh policy, loss, optimizer, and emitted trainer
metrics in the loop without waiting on video I/O.

## Current Model

Measured refresh is the right trainer-side cache policy shape when the atlas is
compatible: rebuild once, then refresh live tensor/cache metadata only when
support/order/fallback/budget checks say it is stale. Cadence still proves the
reference behavior by rebuilding on schedule.

The durable claim from this artifact is not timing. It is:

```text
the real trainer route can reuse compiled sensor-time cache metadata while
matching cadence loss and staying inside tile capacity over frame counts
```

## Evidence

New script:

```text
research_experiments/star_uvt_feature_tubes/projective_interval_trainer_frame_scaling_benchmark.py
```

Artifact:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_trainer_frame_scaling/summary.md
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_trainer_frame_scaling/summary.json
```

For `frames = 4, 8, 16`, `steps = 4`, `size = 16`, `tube_count = 4`:

```text
cadence_cache_rebuilds  = 2, 2, 2
measured_cache_rebuilds = 1, 1, 1
measured/cadence rebuild ratio = 0.5, 0.5, 0.5
max measured-vs-cadence end-loss delta = 0.0
tile_overflow_sum = 0 for every row
max_tile_count = 4 for every row
```

The measured rows also execute live updates and staleness checks before render;
the `8f` row records one stale refresh/support rebin, which is useful because
it proves the measured policy is not merely skipping checks.

## Caveats

The timing columns are smoke diagnostics only. The first MPS/trainer case can
carry cold-start cost, and this artifact is intentionally tiny. Do not cite it
as the wall-time scaling proof.

This is still source-view `feature_dim=3`, not the future F32 target-grid path.
It uses synthetic target tensors, not high-motion extracted world traces.

## Verification

```text
PYTHONPATH=src/train uv run python -m py_compile \
  research_experiments/star_uvt_feature_tubes/projective_interval_trainer_frame_scaling_benchmark.py

PYTHONPATH=src/train uv run python \
  research_experiments/star_uvt_feature_tubes/projective_interval_trainer_frame_scaling_benchmark.py \
  --frame-counts 4,8,16 \
  --steps 4 \
  --size 16 \
  --out-dir outputs/benchmarks/2026-05-24_star_uvt_projective_interval_trainer_frame_scaling
```

## Decision Implication

The memory contract should now say:

```text
goal       fast 2D rasters across time from 4D spacetime primitives
meta-goal share projection/support/binning/visibility/backward over time
key math  UVT trace = pi_* Gamma^* world_primitive
theory    STAR UVT is one local gauge expression of a camera-ray bundle atlas
evidence  real trainer route can reuse projective interval cache with matching loss
```

Next useful gates: run the same trainer evidence against extracted
high-motion/world-trace geometry; then graduate from cache-count proof to
quality and robust warm-timed wall-clock proof.
