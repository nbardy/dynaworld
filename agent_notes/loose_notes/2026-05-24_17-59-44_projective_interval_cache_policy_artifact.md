# Projective Interval Cache Policy Artifact

## Context

The projective interval cache policy needed a saved timing/quality artifact,
not only pytest evidence. The compatible trainer route is still narrow:
`feature_dim=3`, full-frame, projective interval enabled, no F32 target-grid
payload. That is intentional; this artifact measures cache policy behavior
before the richer anisotropic/pixel-depth trace payload exists.

## Artifact

Runner:

```text
research_experiments/star_uvt_feature_tubes/projective_interval_cache_policy_benchmark.py
```

Command:

```text
PYTHONPATH=src/train uv run python \
  research_experiments/star_uvt_feature_tubes/projective_interval_cache_policy_benchmark.py \
  --steps 8 \
  --refresh-every 2 \
  --out-dir outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step
```

Outputs:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step/summary.md
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step/summary.json
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step/rows.csv
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step/cases/{cadence,measured}.json
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step/logs/{cadence,measured}.log
```

## Result

```text
cadence:
    rebuilds              4
    live_updates          4
    staleness_checks      4
    stale_refreshes       4
    support_rebins        4
    end_loss              0.0847767964
    no_first_step_ms      3473.2648

measured:
    rebuilds              1
    live_updates          7
    staleness_checks      7
    stale_refreshes       7
    support_rebins        7
    end_loss              0.0847767964
    no_first_step_ms      2137.2323
```

Comparison:

```text
rebuild_delta_measured_minus_cadence    -3
live_update_delta_measured_minus_cadence +3
end_loss_delta_measured_minus_cadence    0.0
no_first_step_ms_delta                   -1336.0325
```

## Interpretation

Measured policy is useful even in the current narrow route: it avoids three
full compatible-atlas rebuilds over eight steps, keeps the loss identical, and
improves no-first-step mean time. But the support refresh oracle still rebins
on every live update. That means the next cache optimization is not "turn on
measured"; it is reducing metadata invalidation under normal tube motion.

Likely next forks:

- increase or learn support padding/margins so small optimizer motion stays in
  the compiled tile-time cells
- store conservative velocity/support envelopes rather than exact per-step
  bounds
- split only support-event cells that actually changed instead of rebinning the
  whole compatible atlas metadata
- move toward richer gauges/anisotropic traces so the compiled object follows
  motion more naturally

## Verification

The runner compile and dry-run passed. The saved artifact command exited
successfully and wrote all expected outputs.
