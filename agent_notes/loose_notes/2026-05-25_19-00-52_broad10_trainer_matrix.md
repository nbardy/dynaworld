# Broad10 Real-Video Trainer Matrix

## Context

The completion-gap report made the remaining target explicit:

```text
goal       fast 2D rasters across time from 4D spacetime primitives
meta-goal  share projection/support/binning/visibility/backward work over time
key math   UVT trace = pi_* Gamma^* world_primitive
theory     STAR UVT is one local gauge expression of a camera-ray bundle atlas
```

Before this run, the completion gap still showed:

```text
compiled_trainer_source_gap = 5
```

because the strongest trainer evidence covered five distinct source videos.

## Run

I ran the existing guarded projective-interval trainer matrix on 10 distinct
checked-in real-video segments:

```text
PYTHONPATH=src/train uv run python \
  research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_trainer_matrix.py \
  --out-dir outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_trainer_matrix_broad10 \
  --segment-id Bq4rmeIvJbs_seg_000 \
  --segment-id Iagm3K8QtFw_seg_000 \
  --segment-id KUDJ8HDFVQo_seg_000 \
  --segment-id C8kTRrtE3KU_seg_000 \
  --segment-id kcfs1-ryKWE_seg_000 \
  --segment-id iWWUSiERE2I_seg_000 \
  --segment-id WCX_dKC-6Ak_seg_000 \
  --segment-id mFuSjk7jv_M_seg_000 \
  --segment-id BCmnefMeCSA_seg_000 \
  --segment-id I2E_Th5Mocg_seg_000 \
  --frames 8 --size 64 --steps 4 --refresh-every 2 \
  --tile-capacity 128 --tube-count 128 \
  --support-guard-padding 1.0 \
  --support-guard-policy slack_budgeted \
  --support-guard-bisect-steps 8 \
  --support-stale-overshoot-epsilon 0.0 \
  --support-stale-tail-alpha-epsilon 0.001
```

Saved artifact:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_trainer_matrix_broad10/summary.json
```

## Result

The broad10 matrix passes the existing trainer-matrix verifier:

```text
scene_count = 10
distinct_youtube_id_count = 10
row_count = 20
measured_row_count = 10
all_source_videos_exist = true
all_rows_pass = true
all_measured_loss_matches_cadence = true
max_measured_vs_cadence_rebuild_ratio = 0.5
max_measured_support_rebins = 0
max_measured_stale_refreshes = 0
min_motion_score = 0.5781455039978027
max_motion_score = 7.018424034118652
```

The timing caveat remains:

```text
max_measured_vs_cadence_no_first_step_ms_ratio = 1.9762875807881346
```

So this is broad trainer-correctness/source-coverage evidence, not a timing win
and not quality/media acceptance.

## Completion-Gap Update

The completion-gap report now consumes this broad10 trainer matrix and verifies
against current inputs:

```text
PYTHONPATH=src/train uv run python \
  research_experiments/star_uvt_feature_tubes/projective_goal_completion_gap_report.py \
  --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_goal_completion_gap/summary.json \
  --verify-current-inputs
```

Current gaps after regeneration:

```text
compiled_trainer_source_gap = 0
broad_quality_source_gap = 5
broad_quality_frame_count_gap = 1
strict_timing_failure_gap = 2
completion_ready = false
```

The full goal stays active. The next meaningful pushes are broad quality/media
acceptance over 10 sources, a fourth frame-count point, and timing protocol
resolution.

## Verification

```text
PYTHONPATH=src/train uv run python \
  research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_trainer_matrix.py \
  --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_trainer_matrix_broad10/summary.json

PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_goal_completion_gap_report.py -q
```

Result:

```text
12 passed in 2.44s
```
