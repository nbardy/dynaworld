# 2026-05-25 00:09:11 - Synthetic Trainer Frame-Scaling Verifier

## Context

Continuation of the Gauged UVT Trace Atlas goal. Several saved artifacts now
have executable report contracts: cache policy, tail-alpha image error,
anisotropic tail bounds, revolving fixed charts, real-video trainer
frame-scaling, and trained high-motion trace scaling. The older synthetic
production-trainer frame-scaling artifact still lacked a verifier:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_trainer_frame_scaling/summary.json
```

This artifact matters because it runs the actual `run_training` path with a
small generated target, so it isolates trainer/cache behavior without
real-video decoding or high-motion support churn.

## Current Model

The synthetic trainer report is the production-route smoke for this invariant:

```text
same target + same optimizer + same trace parameters
cadence rebuild policy and measured live-cache policy must produce the same loss
measured policy should do fewer full atlas rebuilds
measured live updates and staleness checks must run before render
tile overflow/fallback/visibility stratification should stay zero
```

This is still not the final paper claim. It is a controlled gate that the real
trainer route can execute cache reuse without changing loss.

## What Changed

Added to:

```text
research_experiments/star_uvt_feature_tubes/projective_interval_trainer_frame_scaling_benchmark.py
```

New exports:

```text
verify_interval_trainer_frame_scaling_report(report) -> list[str]
assert_interval_trainer_frame_scaling_report(report)
--verify-report <summary.json>
```

Added focused tests:

```text
tests/test_star_uvt_projective_interval_trainer_frame_scaling_benchmark.py
```

## Contract

The verifier checks:

```text
topology:
    frame_counts strictly increase
    exactly one cadence and one measured row per frame count

trainer:
    status = ok
    rows pass
    loss decreases
    measured/cadence end loss matches per frame count
    tile_overflow_sum = 0
    max_tile_count <= tile_capacity

cache:
    measured rebuilds < cadence rebuilds
    measured live updates > cadence live updates
    measured staleness checks cover measured live updates
    support_rebins = stale_refreshes
    visibility_stratifications = 0
    fallback_marks = 0

synthetic timing smoke:
    measured no-first-step timings beat cadence for every frame count
```

Mutation tests reject:

- measured/cadence loss drift
- missing measured rebuild reduction
- support rebin/stale-refresh mismatch
- lost synthetic no-first-step timing win

## Verification

Focused synthetic verifier:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_interval_trainer_frame_scaling_benchmark.py -q
```

Result:

```text
6 passed in 11.58s
```

Saved synthetic artifact:

```text
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_interval_trainer_frame_scaling_benchmark.py \
  --verify-report outputs/benchmarks/2026-05-24_star_uvt_projective_interval_trainer_frame_scaling/summary.json
```

Result:

```text
verified outputs/benchmarks/2026-05-24_star_uvt_projective_interval_trainer_frame_scaling/summary.json
```

Trainer verifier suite:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_interval_trainer_frame_scaling_benchmark.py \
  tests/test_star_uvt_projective_real_video_trainer_frame_scaling_benchmark.py -q
```

Result:

```text
19 passed in 29.34s
```

Real-video base, guard1, and guard2 frame-scaling artifacts also verified
through their existing `--verify-report` mode.

## Decision Implication

The production trainer cache-reuse chain is now executable at two levels:

```text
synthetic generated target:
    controlled trainer/cache smoke

real high-motion video:
    source-video smoke with support-churn evidence and guarded tail-budget reruns
```

The next higher-value gap is not another synthetic trainer verifier. It is a
real high-motion / revolving-camera or WorldFoam scene where the same trace
atlas object proves quality-preserving reuse under richer visibility.
