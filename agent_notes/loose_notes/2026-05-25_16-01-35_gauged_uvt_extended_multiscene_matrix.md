# Gauged UVT Extended Multiscene Matrix

## Context

The top-level Gauged UVT progress audit still keeps full completion open because
the current evidence is focused: three-source real-video trainer, frame-scaling,
quality, and media-tether matrices plus local Q/Q2 Metal and fallback probes.
To move toward broader real-scene acceptance without overclaiming, this pass ran
an extended real-video trainer matrix using existing guarded projective-interval
report plumbing.

## Artifact

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_trainer_matrix_extended5/summary.json
```

Segments:

```text
Bq4rmeIvJbs_seg_000
Iagm3K8QtFw_seg_000
KUDJ8HDFVQo_seg_000
C8kTRrtE3KU_seg_000
kcfs1-ryKWE_seg_000
```

This adds higher-motion bike/FPV clips beyond the original three-scene default.

## Result

The extended matrix verified with the existing
`verify_real_video_multiscene_trainer_matrix_report` contract:

```text
scene_count = 5
row_count = 10
distinct_youtube_id_count = 5
max_motion_score = 7.018424034118652
max_measured_vs_cadence_end_loss_abs_delta = 0.0
max_measured_vs_cadence_rebuild_ratio = 0.5
max_measured_support_rebins = 0
max_measured_stale_refreshes = 0
tile overflow / fallback / visibility stratification = 0
```

One timing row is noisy:

```text
max_measured_vs_cadence_no_first_step_ms_ratio = 1.50811535915855
```

So the correct claim is functional broadening, not uniform timing speedup.
The compile/cache mechanism still lowers rebuild count and preserves exact
cadence loss on five sources, but this artifact is not a timing promotion.

## Audit Update

`projective_goal_progress_audit.py` now includes this artifact as
`real_video_multiscene_extended_trainer_matrix`, with a new proved requirement.
The top audit now reports:

```text
proved_requirement_count = 28
open_requirement_count = 1
is_goal_complete = false
```

The remaining open item is unchanged in kind: broad real-scene quality and full
trainer replacement remain unproven beyond focused/extended probes.

## Tests

Passed:

```bash
.venv/bin/python research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_trainer_matrix.py --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_trainer_matrix_extended5/summary.json
.venv/bin/python -m py_compile research_experiments/star_uvt_feature_tubes/projective_goal_progress_audit.py
.venv/bin/python research_experiments/star_uvt_feature_tubes/projective_goal_progress_audit.py --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_goal_progress_audit/summary.json --verify-current-inputs
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 uv run --with pytest python -m pytest tests/test_star_uvt_projective_goal_progress_audit.py tests/test_star_uvt_projective_real_video_multiscene_trainer_matrix.py -q
```

The focused pytest command passed `43 passed in 23.81s`.
