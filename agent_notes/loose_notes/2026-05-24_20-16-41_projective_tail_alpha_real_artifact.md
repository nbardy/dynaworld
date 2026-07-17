# Projective Tail-Alpha Real Artifact

## Context

Goal memory:

```text
goal       fast 2D rasters across time from 4D spacetime primitives
meta-goal  share projection/support/binning/visibility/backward over time
key math   UVT trace = pi_* Gamma^* world_primitive
theory     STAR UVT is one local gauge expression of a camera-ray bundle atlas
```

The previous support debounce evidence had two layers:

1. Pixel overshoot tolerance: `support_stale_overshoot_epsilon=0.5` removed
   measured support rebins on the cap128 smoke.
2. Tail-alpha certificate: tests proved small boundary tails can be debounced
   while center/core loss still rebins.

The missing link was a real cache-policy artifact where the measured run skips
rebins because the omitted-alpha certificate is below budget, not because a
pixel-distance escape hatch is enabled.

## Artifact

Command:

```text
PYTHONPATH=src/train uv run python \
  research_experiments/star_uvt_feature_tubes/projective_interval_cache_policy_benchmark.py \
  --steps 8 \
  --refresh-every 2 \
  --support-guard-padding 2 \
  --support-guard-policy slack_budgeted \
  --support-stale-tail-alpha-epsilon 0.001 \
  --tile-capacity 128 \
  --timeout-sec 240 \
  --out-dir outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_slack_tail001_cap128
```

Summary:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_slack_tail001_cap128/summary.md
```

## Result

Both rows pass with identical final loss:

```text
cadence end_loss  = 0.08477679640054703
measured end_loss = 0.08477679640054703
```

Measured cache behavior:

```text
atlas rebuilds        = 1
live updates          = 7
staleness checks      = 7
stale refreshes       = 0
support rebins        = 0
tile overflow sum     = 0
max tile count        = 70
no-first-step         = 1810.1 ms
```

Certificate:

```text
support_stale_tail_alpha_epsilon         = 0.001
support_stale_overshoot_epsilon          = 0.0
max_support_tail_alpha_bound             = 0.00032070223950928124
max_support_max_overshoot_px             = 0.49323272705078125
last_support_missing_tile_pairs          = 275
```

## Interpretation

The old `0.5px` debounce result was suspicious because pixel distance is not a
rendering error. This artifact makes the same cache behavior defensible in the
local isotropic trace setting: the missing support tiles are tail-only under
the compiled support radius, and their maximum omitted alpha is about
`3.2e-4`, below the explicit `1e-3` budget.

This is still not a universal theorem. It depends on:

- isotropic screen-sigma trace metadata being representative,
- support padding being a true footprint radius rather than a center marker,
- visibility staleness remaining independently checked,
- color/residual error tolerating this alpha budget.

## Decision Implication

For the current STAR UVT projective interval route, prefer the math-backed
support reuse certificate over raw pixel stale-overshoot. The next useful gate
is not another cap128 smoke with the same scene; it is broader-scene image/error
validation and anisotropic/pixel-depth trace metadata so the same certificate
can survive less toy-like footprints.
