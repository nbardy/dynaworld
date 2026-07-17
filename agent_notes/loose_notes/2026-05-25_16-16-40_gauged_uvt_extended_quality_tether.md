# Gauged UVT Extended Quality Tether

## Context

The goal-progress audit already had a five-source real-video extended trainer
matrix, but that row was only a functional trainer-contract broadening row:
cadence/measured loss agreement, rebuild count reduction, and zero support
churn/fallback/overflow. It did not make the quality tether explicit for all
five sources.

## Current Model

The right proof shape is:

```text
live-cache trainer path == cadence full-rebuild trainer path
```

not only at the final scalar end loss, but at the saved loss/RGB-loss curves,
end PSNR, and gradient-flow flags. This does not prove broad real-scene quality
acceptance, but it makes the five-source functional matrix harder to
misinterpret as merely a bookkeeping artifact.

## Work Done

Added:

```text
research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_extended_quality_tether_report.py
tests/test_star_uvt_projective_real_video_multiscene_extended_quality_tether_report.py
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_quality_tether/summary.json
```

Then wired that report into the top-level goal-progress audit as:

```text
real_video_multiscene_extended_quality_tether
```

The top-level audit now proves 29 requirement rows and still keeps
`full_goal_completion` open.

## Evidence

Extended quality-tether artifact:

```text
scene_count: 5
pair_count: 5
source_distinct_youtube_id_count: 5
max_abs_loss_curve_delta: 0.0
max_abs_rgb_loss_curve_delta: 0.0
max_end_psnr_abs_delta: 0.0
min_measured_psnr_gain: 0.04466235637664795
min_measured_loss_decrease: 0.0013181120157241821
all_gradient_flags_present: true
```

Regenerated top-level audit:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_goal_progress_audit/summary.json
proved_requirement_count: 29
open_requirement_count: 1
is_goal_complete: false
```

Focused verification:

```text
extended-quality + goal-progress tests: 42 passed in 8.94s
multiscene/tether/guarded/audit pack: 82 passed in 5.51s
saved extended-quality report verified by CLI
saved goal-progress report verified with --verify-current-inputs
```

## Open Gap

This strengthens the five-source matrix by tethering it to cadence quality
curves, but it still is not a broad real-scene quality benchmark and still is
not a full compiled-adjoint trainer replacement.
