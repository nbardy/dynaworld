# 2026-05-24 23:46:07 - Interval Cache-Policy Report Verifier

## Context

Continuation of the Gauged UVT Trace Atlas overnight goal. The previous gate
verified local tail-alpha image error and anisotropic omitted-support bounds.
The remaining gap was one level higher: the saved projective interval
cache-policy artifacts claimed that `measured` atlas reuse preserves the
cadence loss curve while reducing full rebuilds, but that benchmark report did
not yet have an executable verifier.

Relevant file:

```text
research_experiments/star_uvt_feature_tubes/projective_interval_cache_policy_benchmark.py
```

Saved aggregate artifacts checked:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_slack_budgeted_cap128_tail00035_aggregate/summary.json
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_slack_budgeted_cap128_tail00045_aggregate/summary.json
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_slack_budgeted_cap128_tail0006_aggregate/summary.json
outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_slack_budgeted_cap128_tail001_aggregate/summary.json
```

## Current Model

The cache-policy artifact is the operational bridge between the math
certificate and the claimed speed path:

```text
local tail certificate -> support rebin / stale reuse decision
cache-policy benchmark -> fewer world-side atlas rebuilds at same loss
```

So the report verifier should not merely check `status = ok`; it should check
the actual contract:

```text
support_guard_policy = slack_budgeted
support_stale_overshoot_epsilon = 0
support_stale_tail_alpha_epsilon > 0
rows = {cadence, measured}
cadence and measured both pass
tile_overflow_sum = 0
visibility_stratifications = fallback_marks = 0
end_loss_measured = end_loss_cadence
measured_rebuilds < cadence_rebuilds
measured_live_updates > cadence_live_updates
last_tail_bound <= epsilon
```

For reports with measured support rebins:

```text
max_tail_bound > epsilon
```

For reports without measured support rebins:

```text
max_tail_bound <= epsilon
```

That distinction matters because the report stores both instantaneous/final
accepted tail bound and the maximum bound observed before repair.

## What Changed

Added:

```text
verify_projective_interval_cache_policy_report(payload) -> list[str]
assert_projective_interval_cache_policy_report(payload)
--verify-report <summary.json>
```

Added tests:

```text
tests/test_star_uvt_projective_interval_cache_policy_benchmark.py
```

The tests cover:

- synthetic valid report acceptance
- measured/cadence loss drift rejection
- uncertified tail reuse rejection
- missing rebuild-win rejection
- all four saved aggregate artifacts
- monotone epsilon bracket: looser tail budgets produce non-increasing support
  rebin counts and increasing max observed tail bounds

## Verification

Focused pytest:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_interval_cache_policy_benchmark.py -q
```

Result:

```text
9 passed in 0.14s
```

CLI verification:

```text
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_interval_cache_policy_benchmark.py \
  --verify-report <each aggregate summary.json>
```

All four saved aggregate artifacts verified.

## Decision Implication

The cache policy is now part of the certificate chain. Future claims should
route through this order:

```text
1. local tail-alpha / anisotropic support certificate
2. image-error negative controls
3. cache-policy saved-report verifier
4. Metal/runtime parity and timing gates
```

This still does not prove arbitrary revolving-camera correctness. It proves the
current scalar/projective support-reuse path has a checked amortization report
for the compatible 8f/64px full-frame route.

## Open Questions

- Add the same saved-report verifier pattern to any future revolving-camera or
  q-UVT higher-motion cache-policy artifact.
- Decide whether the report verifier should eventually accept non-tail
  policies, or whether the benchmark should keep this strict contract as the
  paper-facing certified path.
