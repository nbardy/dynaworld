# Broad10 Quality Acceptance Gap Update

## Context

The active Gauged UVT goal is still not complete. The current completion-gap
artifact said trainer source coverage had reached 10 sources, but broad
real-scene quality/media acceptance was still stuck at five-source evidence.
There was already a broad10 quality tether script, but no saved artifact.

## What Changed

Generated and verified:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_broad10_quality_tether/summary.json
```

The broad10 quality tether reads saved cadence/measured case payloads from:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_trainer_matrix_broad10/
```

It checks 10 source-distinct quality pairs, matching measured live-cache
projective-interval training against the cadence full-rebuild reference.

## Numerical Note

One broad10 case has loss/RGB-loss curve deltas of
`1.4901161193847656e-08`, while end loss and end PSNR deltas are `0.0`.
That is float32-tick scale, so the broad10 tether uses an explicit
`2.0e-08` loss-curve tolerance. This is deliberately still tight enough to
reject ordinary drift; the focused test rejects `1.0e-4`.

## Propagated Artifacts

Regenerated:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_acceptance_envelope/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_goal_progress_audit/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_goal_completion_gap/summary.json
```

The acceptance envelope now has 10 underlying evidence rows and records
`broad_quality_distinct_youtube_id_count=10`. The completion gap now records:

```text
broad_quality_source_gap=0
broad_media_source_gap=5
broad_quality_frame_count_gap=1
strict_timing_failure_gap=2
compiled_trainer_source_gap=0
```

## Verification

Direct artifact verifiers:

```text
broad10 quality tether: verified
acceptance envelope: verified
goal completion gap with --verify-current-inputs: verified
```

Focused tests:

```text
tests/test_star_uvt_projective_real_video_broad10_quality_tether_report.py
tests/test_star_uvt_projective_real_video_acceptance_envelope_report.py
tests/test_star_uvt_projective_goal_completion_gap_report.py
34 passed in 3.97s
```

Top-level current-input chain:

```text
tests/test_star_uvt_projective_goal_progress_audit.py
tests/test_star_uvt_projective_real_video_acceptance_envelope_report.py
tests/test_star_uvt_projective_goal_completion_gap_report.py
70 passed in 3.25s
```

## Current Model

This closes the quality source-count gap, not the active goal. The remaining
open evidence gaps are:

- broad media acceptance beyond five distinct videos
- at least one fourth real-video frame-count point
- strict timing acceptance, or an explicit replacement protocol based on
  fresh-process medians
- broad full compiled-adjoint trainer replacement evidence

## Decision Implication

The next concrete acceptance move is broad10 media or a fourth frame-count
point. The next timing move is a final timing protocol decision: either make
the strict warm-state gate pass or promote the fresh-process median protocol
as the explicit acceptance contract.
