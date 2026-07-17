# Gauged UVT Extended Frame-Scaling Diagnostic

## Context

The active goal remains Gauged UVT Trace Atlas: compile 4D spacetime primitives
through a known camera program into reusable sensor-time traces so projection,
support, binning, visibility metadata, and backward replay grow sublinearly with
frame count where the camera-path structure allows it.

After the five-source extended functional and media/quality tethers, I ran the
strict five-source real-video frame-scaling matrix over:

```text
Bq4rmeIvJbs_seg_000
Iagm3K8QtFw_seg_000
KUDJ8HDFVQo_seg_000
C8kTRrtE3KU_seg_000
kcfs1-ryKWE_seg_000
```

The strict artifact is:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_frame_scaling_matrix_extended5/summary.json
```

It failed, but only on the intended timing gates.

## Observed Facts

The strict five-source source report has:

```text
scene_count = 5
distinct_youtube_id_count = 5
row_count = 30
measured_row_count = 15
frame_count_count = 3
frame_growth_factor = 4.0
all_source_videos_exist = true
all_rows_pass = true
all_rows_loss_decreased = true
all_rows_no_overflow = true
all_rows_fallback_free = true
all_rows_visibility_stratification_free = true
all_measured_loss_matches_cadence = true
max_measured_vs_cadence_end_loss_abs_delta = 0.0
max_measured_vs_cadence_rebuild_ratio = 0.5
max_measured_cache_rebuild_growth = 1.0
max_measured_support_rebins = 0
max_measured_stale_refreshes = 0
max_measured_support_tail_alpha_bound = 0.0
max_measured_support_overshoot_px = 0.0
max_motion_score = 7.018424034118652
max_tile_count = 22
```

The strict timing failures are:

```text
max_measured_vs_cadence_no_first_step_ms_ratio = 1.188933546093892
max_measured_no_first_growth_vs_frame_growth_ratio = 1.0009153415685994
```

So the source report failed exactly:

```text
multiscene frame-scaling measured no-first timing must beat cadence
multiscene frame-scaling no-first timing growth must stay below frame growth
```

## Diagnostic Artifact

I added:

```text
research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_extended_frame_scaling_diagnostic_report.py
tests/test_star_uvt_projective_real_video_multiscene_extended_frame_scaling_diagnostic_report.py
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_frame_scaling_diagnostic/summary.json
```

This verifier intentionally requires the source report to remain `failed` and
requires the failures to be exactly the two strict timing errors above. It then
locks the correctness/cache/support invariants: source videos exist,
rows pass and decrease loss, cadence loss matches, rebuild ratio is below
cadence, rebuild count does not grow, support rebins/stale refreshes stay zero,
support tail/overshoot stay zero, and fallback/overflow/visibility
stratification stay absent.

This is not a broad real-scene quality acceptance row and not a timing-win row.
It is a caveat row: the harder five-source frame-growth set keeps the math and
cache behavior stable while exposing the current timing limit.

## Top-Level Audit

I wired the diagnostic into:

```text
research_experiments/star_uvt_feature_tubes/projective_goal_progress_audit.py
tests/test_star_uvt_projective_goal_progress_audit.py
outputs/benchmarks/2026-05-25_star_uvt_projective_goal_progress_audit/summary.json
```

The regenerated top-level summary now says:

```text
proved_requirement_count = 31
open_requirement_count = 1
failed_requirement_count = 0
is_goal_complete = false
real_video_multiscene_extended_frame_scaling_scene_count = 5
real_video_multiscene_extended_frame_scaling_row_count = 30
real_video_multiscene_extended_frame_scaling_max_no_first_ratio = 1.188933546093892
real_video_multiscene_extended_frame_scaling_max_growth_ratio = 1.0009153415685994
real_video_multiscene_extended_frame_scaling_max_rebuild_ratio = 0.5
real_video_multiscene_extended_frame_scaling_max_support_rebins = 0
real_video_multiscene_extended_frame_scaling_expected_timing_failures = 2
```

## Current Model

The compiler thesis is not falsified by the five-source frame-growth run:

```text
projection/support/binning/cache-rebuild side:
    still stable

quality/cadence side:
    still tethered

support/fallback side:
    still clean

timing side:
    not yet a uniform five-source win
```

The useful distinction is:

```text
strict source matrix:
    asks whether the five-source frame-growth run is already a timing win

diagnostic report:
    asks whether its non-timing invariants are still good enough to preserve as evidence
```

## Commands

Generation and verification:

```bash
.venv/bin/python research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_extended_frame_scaling_diagnostic_report.py
.venv/bin/python research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_extended_frame_scaling_diagnostic_report.py --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_frame_scaling_diagnostic/summary.json
.venv/bin/python research_experiments/star_uvt_feature_tubes/projective_goal_progress_audit.py
.venv/bin/python research_experiments/star_uvt_feature_tubes/projective_goal_progress_audit.py --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_goal_progress_audit/summary.json --verify-current-inputs
```

Focused tests:

```bash
.venv/bin/python -m py_compile research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_extended_frame_scaling_diagnostic_report.py research_experiments/star_uvt_feature_tubes/projective_goal_progress_audit.py
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 uv run --with pytest python -m pytest tests/test_star_uvt_projective_real_video_multiscene_extended_frame_scaling_diagnostic_report.py tests/test_star_uvt_projective_goal_progress_audit.py -q
```

Result:

```text
44 passed in 6.48s
```

Broader touched verifier pack:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_real_video_multiscene_trainer_matrix.py \
  tests/test_star_uvt_projective_real_video_multiscene_frame_scaling_matrix.py \
  tests/test_star_uvt_projective_real_video_multiscene_extended_frame_scaling_diagnostic_report.py \
  tests/test_star_uvt_projective_real_video_multiscene_extended_quality_tether_report.py \
  tests/test_star_uvt_projective_real_video_multiscene_media_tether_report.py \
  tests/test_star_uvt_projective_real_video_multiscene_quality_tether_report.py \
  tests/test_star_uvt_projective_real_video_guarded_support_matrix_report.py \
  tests/test_star_uvt_projective_goal_progress_audit.py -q
```

Result:

```text
90 passed in 7.23s
```

## Decision Implications

Do not weaken the three-source strict frame-scaling verifier. It is still the
clean small timing-win matrix.

Do not promote the five-source strict frame-scaling source report as passing.
The correct durable artifact is the diagnostic report, which preserves the
negative timing information.

Next timing work should inspect why the five-source no-first ratio crosses
1.0 while rebuild/support metrics remain clean. Good branches:

```text
1. per-scene outlier dominates no-first timing
2. MPS launch noise / short-run variance is enough to cross the gate
3. active-list/tile shape is clean but per-sample shading/compositing dominates
4. the current measured path amortizes rebuilds but not enough live evaluation
```

Cheap falsification tests:

```text
repeat the five-source strict matrix with repeat timing
emit per-scene timing ratios and max-tile/active-set correlations
compare an 8/16/32 frame variant to reduce 4-frame startup noise
profile measured path phases on the worst source
```
