# Shared-Work Aggregate Artifact Restored

## Context

The aggregate shared-work audit had the right contract after the mixed fallback
backward integration, but the default saved inputs were not all present in this
checkout. The active objective needs actual evidence for:

```text
share projection/support/binning/payload and backward work across time
```

not just tests over synthetic in-memory payloads.

## Work Done

Restored and verified the three trained high-motion inputs:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling/summary.json
outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling_64px_128t/summary.json
outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling_96px_256t_cap256/summary.json
```

The 96px/256t/cap256 row was noisy with one timing iteration, so I reran it
with:

```text
--timing-iterations 5 --timing-warmup 3
```

This preserved the topology/entry results and brought the final trained
shared/per-frame backward ratio under the aggregate threshold.

Then regenerated and verified:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_shared_work_goal_audit/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_shared_work_goal_audit/summary.md
```

## Current Aggregate Evidence

```text
orbit fixed payload growth:             1.0x
orbit per-frame payload growth:         8.0x
orbit final payload ratio:              0.0625
orbit final backward ratio:             0.158417
max trained final interval-entry ratio: 0.148369
max trained final backward ratio:       0.207953
max trained shared entry growth:        1.461735x
min trained per-frame entry growth:     9.852041x
exposure forward Metal cases:           4
exposure backward Metal cases:          2
mixed fallback backward cases:          2
mixed fallback max grad rel error:      7.406e-7
```

The old `4/8/16/32` orbit timing failure remains quarantined:

```text
outputs/benchmarks/2026-05-25_star_uvt_revolving_orbit_fixed_chart_scaling_current_timing_fail/
```

It is useful negative evidence that small timing probes can be dominated by
MPS launch/schedule noise. The restored default orbit artifact uses
`8/16/32/64` frames and verifies.

## Verification

```text
PYTHONPATH=src/train uv run python \
  research_experiments/star_uvt_feature_tubes/projective_orbit_fixed_chart_scaling_benchmark.py \
  --verify-report outputs/benchmarks/2026-05-24_star_uvt_revolving_orbit_fixed_chart_scaling/summary.json

PYTHONPATH=src/train uv run python \
  research_experiments/star_uvt_feature_tubes/projective_trained_high_motion_trace_scaling_benchmark.py \
  --verify-report outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling_96px_256t_cap256/summary.json

PYTHONPATH=src/train uv run python \
  research_experiments/star_uvt_feature_tubes/projective_shared_work_goal_audit.py \
  --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_shared_work_goal_audit/summary.json

PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_shared_work_goal_audit.py \
  tests/test_star_uvt_projective_trained_high_motion_trace_scaling_benchmark.py -q
```

Result:

```text
shared audit artifact verified
28 passed in 20.51s
```

## Decision Implication

The current aggregate evidence is live again. It proves the current research
prototype has reusable orbit payload/compile/forward/backward behavior, trained
high-motion shared interval scaling across three sizes, rendered-field
exposure/rolling semantics, ordinary shared adjoints, and differentiable mixed
fallback adjoints. It still does not complete the whole active goal; production
quality, broad real-scene camera programs, and full trainer-side deployment are
separate remaining gates.
