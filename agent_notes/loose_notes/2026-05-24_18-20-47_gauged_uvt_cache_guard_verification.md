# 2026-05-24 18:20 +07 - Gauged UVT cache guard verification

## Context

Heartbeat continuation for Gauged UVT Trace Atlas. The previous verified state
had the compatible projective interval producer routed through the real STAR
UVT feature trainer for the `feature_dim=3` exact route. The next stated gap was
cache ownership and staleness/recompile cadence.

## Current Model

The tree is ahead of that gap. The feature trainer now has a concrete cache
policy:

- `projective_interval.refresh_policy="cadence"` performs fixed full-atlas
  rebuilds by `refresh_every`.
- `projective_interval.refresh_policy="measured"` keeps compiled cell metadata
  alive and updates only differentiable live tensors, then calls the
  `ProjectiveCellIntervalTrainerState` refresh oracle before rendering.
- Cache telemetry is reported on the trainer row: rebuilds, live updates,
  alpha renders, staleness checks, stale refreshes, support rebins, visibility
  stratifications, and fallback marks.
- `projective_interval.support_guard_padding` widens compiled support cells
  while the correctness/staleness check still uses base `uv_padding`.

The important math split is:

```text
coverage check support = uv_padding
compiled chart support = uv_padding + support_guard_padding
```

This is a real gauge-domain margin: correctness remains certified against the
base support, while the compiled atlas has room for ordinary optimizer motion.

## Verification

Targeted route/cache/config gate:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_uvt_producer.py \
  tests/test_star_uvt_render_configs.py -q
```

Result:

```text
23 passed in 30.08s
```

Focused projective plus interval-gated trainer gate:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_trace.py \
  tests/test_star_uvt_projective_orbit_windows.py \
  tests/test_star_uvt_projective_visibility.py \
  tests/test_star_uvt_projective_binning.py \
  tests/test_star_uvt_projective_correctness.py \
  tests/test_star_uvt_projective_uvt_producer.py \
  tests/test_star_uvt_render_configs.py \
  tests/test_star_uvt_trainer_interval_gated.py -q
```

Result:

```text
109 passed in 26.38s
```

`--collect-only` confirms this exact focused set currently collects 109 tests.
The durable docs had stale 113/114-pass counts; corrected them to the verified
109-pass count.

## Artifact Check

Confirmed both cache-policy benchmark artifact folders exist:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step/
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_cap256/
```

The guard artifact reports:

```text
cadence:  rebuilds=4, live_updates=4, stale_refreshes=0, support_rebins=0
measured: rebuilds=1, live_updates=7, stale_refreshes=0, support_rebins=0
```

Both rows have matching final loss at the displayed precision. This supports
the current claim: measured cache plus guard padding gives reuse without the
every-step support rebin churn seen in the earlier no-guard artifact.

## Next

The next real gate is budget-aware/adaptive guard selection rather than another
global flag. Guard `2` with tile capacity `256` works in the saved artifact,
while smaller capacity settings can overflow; production needs to choose guard
and capacity from atlas budget reports, fallback fraction, and tile overflow
risk.
