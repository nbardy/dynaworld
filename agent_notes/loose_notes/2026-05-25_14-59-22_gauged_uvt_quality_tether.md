# Gauged UVT Quality Tether

## Context

The source-distinct frame-scaling matrix proved that guarded projective-interval
training keeps support churn at zero and grows non-pixel work sublinearly across
three checked-in videos and `4,8,16` frames. Its remaining weakness was that the
quality claim was mostly an end-loss scalar. I added a report that reads the
saved per-case trainer payloads and compares the measured live-cache route
against the cadence full-rebuild route more tightly.

## Added

```text
research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_quality_tether_report.py
tests/test_star_uvt_projective_real_video_multiscene_quality_tether_report.py
```

Saved artifact:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_quality_tether/summary.json
```

## What It Proves

The report uses the saved cases under:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_frame_scaling_matrix/cases/
```

It verifies:

```text
pair_count:                      9
scene_count:                     3
frame_count_count:               3
max_abs_loss_curve_delta:        0.0
max_abs_rgb_loss_curve_delta:    0.0
max_end_loss_abs_delta:          0.0
max_end_psnr_abs_delta:          0.0
min_measured_psnr_gain:          0.02227306365966797
min_measured_end_psnr:           4.748017191886902
all_gradient_flags_present:      true
```

This is a stronger tether than the previous end-loss equality: the measured
live-cache path has the same loss curve, RGB-loss curve, end loss, and end PSNR
as the cadence full-rebuild path for every saved source/frame pair.

## Audit Integration

`projective_goal_progress_audit.py` now has a top-level
`real_video_multiscene_quality_tether` requirement. The regenerated audit reports:

```text
proved_requirement_count: 26
open_requirement_count:   1
is_goal_complete:         false
```

The open completion row is still correct. This is not broad real-scene quality
acceptance; it is a cadence tether over a small saved source-distinct matrix.

## Verification

Commands run:

```text
.venv/bin/python research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_quality_tether_report.py --out-dir outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_quality_tether
.venv/bin/python research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_quality_tether_report.py --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_quality_tether/summary.json
.venv/bin/python research_experiments/star_uvt_feature_tubes/projective_goal_progress_audit.py --out-dir outputs/benchmarks/2026-05-25_star_uvt_projective_goal_progress_audit
.venv/bin/python research_experiments/star_uvt_feature_tubes/projective_goal_progress_audit.py --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_goal_progress_audit/summary.json --verify-current-inputs
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 uv run --with pytest python -m pytest tests/test_star_uvt_projective_real_video_multiscene_quality_tether_report.py tests/test_star_uvt_projective_real_video_multiscene_frame_scaling_matrix.py tests/test_star_uvt_projective_real_video_multiscene_trainer_matrix.py tests/test_star_uvt_projective_real_video_guarded_support_matrix_report.py tests/test_star_uvt_projective_goal_progress_audit.py -q
```

Focused suite:

```text
64 passed in 1.70s
```

## Next Implication

The next real acceptance step should not be another scalar cache metric. It
should broaden quality: more scenes, larger resolution, longer training, or a
dense/reference image comparison that checks actual rendered frames, while
keeping the same support/rebuild/fallback/timing fields.
