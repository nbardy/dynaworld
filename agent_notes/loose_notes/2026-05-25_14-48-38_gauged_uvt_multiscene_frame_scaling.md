# Gauged UVT Multiscene Frame-Scaling Matrix

## Context

The previous source-distinct real-video matrix proved the guarded
projective-interval trainer contract across three checked-in video segments, but
only at one frame count. The remaining sublinear-growth question was whether the
same source-distinct contract survives as frame count grows.

## What Changed

Added:

```text
research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_frame_scaling_matrix.py
tests/test_star_uvt_projective_real_video_multiscene_frame_scaling_matrix.py
```

The report runs three source-distinct checked-in segments over `4,8,16` frames,
with both cadence and measured cache policies. It uses the guarded
projective-interval trainer route, not a synthetic substitute.

Saved artifact:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_frame_scaling_matrix/summary.json
```

## Evidence

The artifact verifies:

```text
scene_count:                                  3
frame_count_count:                            3
row_count:                                    18
frame_growth_factor:                          4.0
max measured/cadence no-first ratio:          0.6901796551165242
max measured/cadence rebuild ratio:           0.5
max measured no-first/frame-growth ratio:     0.4376975869236762
max measured cache rebuild growth:            1.0
max measured support rebins/stale refreshes:  0 / 0
max measured support tail alpha bound:        0.0
```

Per-scene measured rebuild count stayed flat from 4 to 16 frames. Cadence losses
matched within `2.98e-8`, and every row stayed free of overflow, fallback marks,
and visibility stratification.

## Audit Integration

`projective_goal_progress_audit.py` now includes
`real_video_multiscene_frame_scaling_matrix` as a top-level proved requirement.
The regenerated goal-progress artifact verifies against current inputs and now
reports:

```text
proved_requirement_count: 25
open_requirement_count:   1
is_goal_complete:         false
```

The remaining open row is still intentional: broad real-scene quality acceptance
and a full compiled-adjoint trainer replacement are not proven by these focused
small matrices.

## Verification

Commands run:

```text
.venv/bin/python research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_frame_scaling_matrix.py --out-dir outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_frame_scaling_matrix
.venv/bin/python research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_frame_scaling_matrix.py --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_frame_scaling_matrix/summary.json
.venv/bin/python research_experiments/star_uvt_feature_tubes/projective_goal_progress_audit.py --out-dir outputs/benchmarks/2026-05-25_star_uvt_projective_goal_progress_audit
.venv/bin/python research_experiments/star_uvt_feature_tubes/projective_goal_progress_audit.py --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_goal_progress_audit/summary.json --verify-current-inputs
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 uv run --with pytest python -m pytest tests/test_star_uvt_projective_real_video_multiscene_frame_scaling_matrix.py tests/test_star_uvt_projective_real_video_multiscene_trainer_matrix.py tests/test_star_uvt_projective_real_video_guarded_support_matrix_report.py tests/test_star_uvt_projective_goal_progress_audit.py -q
```

Focused tests passed:

```text
57 passed in 1.51s
```

## Decision Implication

This closes the narrow "source-distinct frame growth" gap for the guarded
projective-interval trainer smoke. Do not inflate it into a paper-level result:
the evidence is still small, low-resolution, and focused on cache/rebuild/support
contracts rather than broad image quality.

Next useful step: broaden real-scene acceptance with more scenes/resolution or a
quality-tethered dense-reference comparison while preserving the same strict
support/rebuild/fallback fields.
