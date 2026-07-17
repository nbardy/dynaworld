# 2026-05-25 19:31:51 Broad10 Media Tether Gap Close

## Context

The active goal remains:

```text
fast 2D rasters across time from 4D spacetime primitives
```

with the meta-goal:

```text
share projection/support/binning/visibility/backward work over time
```

The pinned math/theory anchors remain:

```text
key math: UVT trace = pi_* Gamma^* world_primitive
theory:   STAR UVT is one local gauge expression of a camera-ray bundle atlas
```

Before this chunk, the completion-gap report had already closed the broad10
trainer and broad10 quality source-count gaps, but still had
`broad_media_source_gap=5`.

## Current Model

Broad media acceptance is not a new renderer theorem. It is an artifact
contract: the measured live-cache projective-interval route must pass through
the same actual contact-sheet media writer as the cadence full-rebuild route,
and the resulting media must match cadence while retaining gradients and clean
support/cache metadata.

The meaningful media invariant is exact image equality:

```text
max_abs_contact_sheet_delta = 0
contact_sheet_sha256_match = true
```

Scalar losses and PSNR computed around that media are finite float summaries.
They can differ by one float32 tick even when the PNG rows are identical.

## What Ran

Ran the broad10 media tether over the same 10 source-distinct clips as the
broad10 trainer/quality matrix:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_broad10_media_tether/summary.json
```

The raw run produced all contact sheets and case rows, but initially wrote
`status=failed` because verifier thresholds still required `1.0e-8` scalar
equality. The only failures were:

```text
max_abs_loss_curve_delta      = 1.4901161193847656e-08
max_abs_rgb_loss_curve_delta  = 1.4901161193847656e-08
max_final_full_rgb_loss_delta = 2.9802322387695312e-08
max_final_full_rgb_psnr_delta = 5.960464477539062e-07
```

while the media itself was exact:

```text
max_abs_contact_sheet_delta = 0
max_mean_abs_contact_sheet_delta = 0.0
all_contact_sheet_hashes_match_cadence = true
```

## Change

Added explicit media scalar tolerances to
`projective_real_video_multiscene_media_tether_report.py`:

```text
MEDIA_SCALAR_LOSS_TOLERANCE = 3.0e-8
MEDIA_SCALAR_PSNR_TOLERANCE = 1.0e-6
```

The tolerance is deliberately narrow: it admits the observed float32 tick while
keeping exact pixel/hash contact-sheet equality as the stronger media gate.
Added a focused test proving scalar ticks are accepted only when the media row
contract remains exact/nontrivial.

Then wired the broad10 media report into:

```text
projective_real_video_acceptance_envelope_report.py
projective_goal_completion_gap_report.py
```

and regenerated:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_acceptance_envelope/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_goal_progress_audit/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_goal_completion_gap/summary.json
```

## Result

The broad10 media artifact now verifies:

```text
scene_count = 10
distinct_youtube_id_count = 10
pair_count = 10
all_contact_sheet_pixels_match_cadence = true
all_contact_sheet_hashes_match_cadence = true
all_gradient_flags_present = true
all_rows_no_overflow = true
all_rows_fallback_free = true
all_rows_visibility_stratification_free = true
max_measured_vs_cadence_rebuild_ratio = 0.5
min_measured_psnr_gain = 0.03675997257232666
```

The completion-gap report now records:

```text
completion_ready = false
broad_quality_source_gap = 0
broad_media_source_gap = 0
broad_quality_frame_count_gap = 1
strict_timing_failure_gap = 2
compiled_trainer_source_gap = 0
```

## Decision Implications

Closed: broad source-count coverage for trainer, quality, and media.

Still open:

```text
1 more real-video frame-count point
2 strict timing failures or an accepted replacement timing protocol
full compiled-adjoint trainer replacement evidence beyond cadence tethers
```

This does not change the theory. It tightens the evidence accounting around the
STAR UVT trace atlas route.

## Verification

Commands verified the refreshed artifacts:

```text
projective_real_video_multiscene_media_tether_report.py --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_broad10_media_tether/summary.json
projective_real_video_acceptance_envelope_report.py --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_acceptance_envelope/summary.json
projective_goal_completion_gap_report.py --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_goal_completion_gap/summary.json --verify-current-inputs
```

Focused test suite:

```text
83 passed in 3.06s
```
