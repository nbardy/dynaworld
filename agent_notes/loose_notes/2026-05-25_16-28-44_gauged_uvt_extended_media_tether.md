# Gauged UVT Extended Media Tether

## Context

The five-source extended trainer matrix had scalar/curve quality tether
evidence, but only the three-source matrix had actual contact-sheet media
evidence. The next natural rung was to run the real media writer on all five
extended real-video sources and tether live-cache output to cadence at the
pixel artifact level.

## Work Done

Ran:

```text
research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_media_tether_report.py
```

with the five extended segment ids:

```text
Bq4rmeIvJbs_seg_000
Iagm3K8QtFw_seg_000
KUDJ8HDFVQo_seg_000
C8kTRrtE3KU_seg_000
kcfs1-ryKWE_seg_000
```

Artifact:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_media_tether/summary.json
```

Then wired that artifact into the top-level goal-progress audit as:

```text
real_video_multiscene_extended_media_tether
```

## Evidence

Extended media-tether summary:

```text
scene_count: 5
pair_count: 5
distinct_youtube_id_count: 5
max_abs_contact_sheet_delta: 0
max_contact_sheet_payload_loss_abs_delta: 0.001525666420389149
max_abs_loss_curve_delta: 0.0
max_final_full_rgb_loss_abs_delta: 0.0
min_contact_sheet_target_std: 0.14441643529730494
min_contact_sheet_pred_std: 0.07178262974117959
min_measured_psnr_gain: 0.04466235637664795
max_measured_vs_cadence_rebuild_ratio: 0.5
max_measured_vs_cadence_no_first_step_ms_ratio: 1.2065694734694634
```

The no-first ratio above 1 means this is media/quality evidence, not timing-win
evidence.

Regenerated top-level audit:

```text
proved_requirement_count: 30
open_requirement_count: 1
is_goal_complete: false
```

Verification:

```text
saved extended media tether verifies by CLI
saved goal-progress audit verifies with --verify-current-inputs
focused media + goal-progress tests: 46 passed in 8.02s
focused multiscene/tether/guarded/audit pack: 83 passed in 8.57s
```

## Open Gap

This removes the mismatch where only the three-source set had real media
artifacts. It still does not prove broad real-scene quality acceptance or a
full compiled-adjoint trainer replacement.
