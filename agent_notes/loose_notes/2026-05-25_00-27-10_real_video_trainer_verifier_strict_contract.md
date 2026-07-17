# Real-Video Trainer Verifier Strict Contract

## Context

The Gauged UVT heartbeat goal is still active: compile 4D spacetime primitives
through a known camera program into reusable sensor-time traces, with clean
derivatives and maximal compute/memory/backward reuse across frames. The
immediate code path is the STAR UVT projective interval trainer benchmark. Its
saved real-video artifacts already showed measured cache reuse versus cadence
rebuilds, and guard025/guard05/guard1/guard2 eliminated support churn on the
checked-in high-motion clip.

The weak point was verification. The saved-artifact tests listed guard025, but
the broad real-video verifier still trusted too much of the report shape and
summary. That meant a stale or hand-edited `summary.json` could preserve the
headline even if row-level evidence drifted.

## Current Model

The projective measured-cache claim should be treated as a contract, not a
pretty table:

```text
For each real-video frame count:
    cadence and measured rows both pass
    loss decreases
    measured end loss matches cadence end loss within 1e-5
    measured rebuilds < cadence rebuilds
    measured live updates > cadence live updates
    measured staleness checks cover live updates
    support_rebins == stale_refreshes
    fallback marks == visibility stratifications == 0
    tile overflow == 0 and max tile count is inside capacity
    positive no-first-step, forward, and backward timings exist
```

The report summary is now derived evidence. It must match `summarize(rows)` for
the important keys, and `all_measured_loss_matches_cadence` must be true.

## What Changed

File:
    `research_experiments/star_uvt_feature_tubes/projective_real_video_trainer_frame_scaling_benchmark.py`

The real-video verifier now checks:

- strictly increasing `frame_counts` with at least two entries
- positive top-level `steps` and `tile_capacity`
- exactly one cadence row and one measured row per frame count
- row pass status, loss decrease, step count, zero overflow, bounded max tile count
- zero fallback marks and zero visibility stratification marks
- finite positive timings for no-first-step, forward, and backward
- measured/cadence end-loss delta below `1e-5`
- measured rebuild reduction and live update/staleness coverage
- support-rebin/stale-refresh consistency
- recomputed summary equality for cache, pass, overflow, loss-match, rebuild-ratio, and timing-ratio keys

File:
    `tests/test_star_uvt_projective_real_video_trainer_frame_scaling_benchmark.py`

The fixture now builds summaries through `summarize(rows)`. Mutation tests were
added for fallback marks, support-refresh mismatch, and stale summary fields.

## Verification

Focused real-video verifier:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_real_video_trainer_frame_scaling_benchmark.py -q

20 passed in 21.70s
```

Broad CLI verification passed for all five saved real-video artifacts:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_real_video_trainer_frame_scaling/summary.json
outputs/benchmarks/2026-05-24_star_uvt_projective_real_video_trainer_frame_scaling_guard025_tail001/summary.json
outputs/benchmarks/2026-05-24_star_uvt_projective_real_video_trainer_frame_scaling_guard05_tail001/summary.json
outputs/benchmarks/2026-05-24_star_uvt_projective_real_video_trainer_frame_scaling_guard1_tail001/summary.json
outputs/benchmarks/2026-05-24_star_uvt_projective_real_video_trainer_frame_scaling_guard2_tail001/summary.json
```

Strict guarded-support CLI verification passed for all four guarded artifacts:

```text
guard025, guard05, guard1, guard2
```

Paired synthetic plus real-video verifier suite:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_interval_trainer_frame_scaling_benchmark.py \
  tests/test_star_uvt_projective_real_video_trainer_frame_scaling_benchmark.py -q

26 passed in 10.59s
```

## Decision Implications

The guard025 claim is now safer to cite: it is the smallest certified no-churn
guard on this real-video fixture under a verifier that checks row evidence and
summary consistency. This still does not prove arbitrary revolving-camera
generalization or visual quality. It does support the narrower Metal acceptance
lane: live projective interval cache reuse can be verified on a real video with
zero overflow, zero fallback, exact cadence loss, reduced rebuilds, and
guarded support stability.

Next useful tests:

1. Mirror the same strict row/summary contract onto any future revolving-camera
   or screen-fiber benchmark before trusting saved summaries.
2. Add a GPU/Metal version only after the CPU/MPS verifier rejects stale
   reports strongly enough.
3. Keep guard size as a policy knob, not a monotone quality assumption; guard2
   clears churn but can slow measured 16f versus cadence.
