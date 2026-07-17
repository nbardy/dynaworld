# Projective Tail-Alpha Telemetry

## Context

The tail-alpha certificate made support reuse mathematically better, but it
would be hard to trust in real cache-policy runs if the bound stayed hidden
inside `refresh_projective_cell_interval_atlas_if_stale(...)`.

Goal memory remains:

```text
goal       fast 2D rasters across time from 4D spacetime primitives
meta-goal  share projection/support/binning/visibility/backward over time
key math   UVT trace = pi_* Gamma^* world_primitive
theory     STAR UVT is one local gauge expression of a camera-ray bundle atlas
```

## Change

Plumbed the certificate into trainer/benchmark artifacts:

```text
projective_interval_cache_last_support_tail_alpha_bound
projective_interval_cache_max_support_tail_alpha_bound
```

The cache records the last refresh's bound and the maximum observed bound over
staleness checks. The benchmark table now carries both fields beside support
overshoot metrics.

## Why It Matters

The cache-policy question becomes auditable:

```text
Did we avoid a support rebin?
If yes, what omitted-alpha bound justified reuse?
```

That is closer to the final renderer contract than a naked pixel overshoot.
It also gives the next real artifact a scalar to sweep against quality/runtime.

## Verification

Targeted telemetry test:

```text
1 passed in 2.88s
```

Focused projective/interval suite:

```text
126 passed in 16.39s
```

Benchmark dry run:

```text
PYTHONPATH=src/train uv run python \
  research_experiments/star_uvt_feature_tubes/projective_interval_cache_policy_benchmark.py \
  --steps 1 \
  --support-guard-policy slack_budgeted \
  --support-stale-tail-alpha-epsilon 0.0003 \
  --dry-run \
  --out-dir /tmp/star_uvt_tail_alpha_benchmark_dryrun2
```

The generated summary contains both tail-alpha columns.
