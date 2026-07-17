# Broad10 Quality Tether

## Context

The active thread memory remains:

```text
goal       fast 2D rasters across time from 4D spacetime primitives
meta-goal  share projection/support/binning/visibility/backward work over time
key math   UVT trace = pi_* Gamma^* world_primitive
theory     STAR UVT is one local gauge expression of a camera-ray bundle atlas
```

The broad10 trainer matrix closed the compiled-trainer source-count gap, but it
did not by itself prove quality tethering beyond five sources. The next
target was to use the saved broad10 cadence/measured trainer cases as a
quality-only tether.

## Artifact

Added:

```text
research_experiments/star_uvt_feature_tubes/projective_real_video_broad10_quality_tether_report.py
tests/test_star_uvt_projective_real_video_broad10_quality_tether_report.py
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_broad10_quality_tether/summary.json
```

The report reads:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_trainer_matrix_broad10/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_trainer_matrix_broad10/cases/*.json
```

## Result

Saved summary:

```text
scene_count = 10
distinct_youtube_id_count = 10
pair_count = 10
all_rows_pass = true
all_measured_loss_curves_match_cadence = true
all_gradient_flags_present = true
all_measured_psnr_improves = true
max_abs_loss_curve_delta = 1.4901161193847656e-08
max_abs_rgb_loss_curve_delta = 1.4901161193847656e-08
min_measured_psnr_gain = 0.03675997257232666
```

One broad10 case has a `1.49e-8` curve delta, so the broad10 tether uses an
explicit `2.0e-8` float32-tick tolerance. This is still tight enough to catch
real drift while not treating one MPS/JSON float tick as a quality failure.

## Envelope Updates

The real-video acceptance envelope now includes broad10 quality tethering:

```text
underlying_report_count = 10
broad10_quality_distinct_youtube_id_count = 10
broad_quality_distinct_youtube_id_count = 10
does_not_prove_completion = true
```

The completion-gap report now says:

```text
broad_quality_source_gap = 0
broad_media_source_gap = 5
broad_quality_frame_count_gap = 1
strict_timing_failure_gap = 2
compiled_trainer_source_gap = 0
completion_ready = false
```

So quality source coverage moved; media, frame-count breadth, timing, and full
goal completion remain open.

## Verification

Commands run:

```text
PYTHONPATH=src/train uv run python \
  research_experiments/star_uvt_feature_tubes/projective_real_video_broad10_quality_tether_report.py

PYTHONPATH=src/train uv run python \
  research_experiments/star_uvt_feature_tubes/projective_real_video_broad10_quality_tether_report.py \
  --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_broad10_quality_tether/summary.json

PYTHONPATH=src/train uv run python \
  research_experiments/star_uvt_feature_tubes/projective_real_video_acceptance_envelope_report.py

PYTHONPATH=src/train uv run python \
  research_experiments/star_uvt_feature_tubes/projective_goal_progress_audit.py \
  --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_goal_progress_audit/summary.json \
  --verify-current-inputs

PYTHONPATH=src/train uv run python \
  research_experiments/star_uvt_feature_tubes/projective_goal_completion_gap_report.py \
  --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_goal_completion_gap/summary.json \
  --verify-current-inputs

PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_real_video_broad10_quality_tether_report.py \
  tests/test_star_uvt_projective_real_video_acceptance_envelope_report.py \
  tests/test_star_uvt_projective_goal_completion_gap_report.py \
  tests/test_star_uvt_projective_goal_progress_audit.py -q
```

Focused result:

```text
77 passed in 3.83s
```
