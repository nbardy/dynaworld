# 2026-05-25 00:14:23 - Guard025 Real-Video Support Verifier Coverage

## Context

Continuation of the Gauged UVT Trace Atlas goal. The real-video trainer
frame-scaling benchmark already had two verifier layers:

```text
verify_real_video_trainer_frame_scaling_report(...)
verify_guarded_real_video_trainer_support_report(...)
```

The saved artifact set includes a quarter-pixel guarded run:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_real_video_trainer_frame_scaling_guard025_tail001/summary.json
```

but the focused saved-artifact tests covered base, guard05, guard1, and guard2,
not guard025.

## What Changed

Updated:

```text
tests/test_star_uvt_projective_real_video_trainer_frame_scaling_benchmark.py
```

The guard025 artifact is now included in:

```text
test_saved_real_video_trainer_frame_scaling_artifact_satisfies_contract
test_saved_guarded_real_video_trainer_support_artifacts_satisfy_contract
```

This means guard025 is checked by both the broad real-video frame-scaling
contract and the strict guarded-support contract.

## Verification

Strict guarded-support CLI:

```text
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_real_video_trainer_frame_scaling_benchmark.py \
  --verify-report outputs/benchmarks/2026-05-24_star_uvt_projective_real_video_trainer_frame_scaling_guard025_tail001/summary.json \
  --verify-guarded-support
```

Result:

```text
verified outputs/benchmarks/2026-05-24_star_uvt_projective_real_video_trainer_frame_scaling_guard025_tail001/summary.json
```

Focused real-video verifier tests:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_real_video_trainer_frame_scaling_benchmark.py -q
```

Result:

```text
17 passed in 16.46s
```

## Decision Implication

The guarded support-churn evidence now covers more of the guard-size bracket:

```text
guard025, guard05, guard1, guard2
```

All are routed through the same saved-artifact verifier. This does not prove an
optimal guard policy, but it prevents a quiet regression where the smallest
guarded run drops out of the executable handoff.
