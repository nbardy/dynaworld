# Gauged UVT Multiscene Trainer Matrix

## Context

Continuation of the Gauged UVT Trace Atlas goal: compile 4D spacetime
primitives through a known camera program into reusable sensor-time traces so
projection/support/binning/visibility/backward work can be shared across time.

The previous real-video trainer evidence covered a single high-motion clip and
a guarded-support ladder on that clip. That was useful but too narrow for the
remaining broad-scene/trainer gap, so this pass added a small source-distinct
real-video trainer matrix without claiming broad acceptance.

## What Changed

Added:

```text
research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_trainer_matrix.py
tests/test_star_uvt_projective_real_video_multiscene_trainer_matrix.py
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_trainer_matrix/summary.json
```

The report runs the actual `star_uvt_feature_overfit_trainer.run_training`
route on three checked-in source-distinct video segments:

```text
Bq4rmeIvJbs_seg_000
Iagm3K8QtFw_seg_000
KUDJ8HDFVQo_seg_000
```

Each scene gets a cadence row and a measured live-cache row with guarded
projective-interval support:

```text
frames = 8
size = 64
steps = 4
tube_count = 128
tile_capacity = 128
support_guard_padding = 1.0
support_guard_policy = slack_budgeted
support_stale_tail_alpha_epsilon = 0.001
```

The saved artifact verifies:

```text
scene_count = 3
row_count = 6
distinct_youtube_id_count = 3
all_rows_pass = true
all_measured_loss_matches_cadence = true
max_measured_vs_cadence_no_first_step_ms_ratio = 0.549583769671522
max_measured_vs_cadence_rebuild_ratio = 0.5
max_measured_support_rebins = 0
max_measured_stale_refreshes = 0
max_measured_support_tail_alpha_bound = 0.0
max_tile_count = 22
```

I removed an order-dependent `motion_score_growth` summary key because the
scene list is not a sorted motion sweep. I also tightened the rebuild-ratio
verifier to reject missing/non-finite values explicitly rather than relying on
truthy fallback behavior.

## Audit Promotion

The top-level goal-progress audit now imports the multiscene verifier and adds
the proved requirement:

```text
real_video_multiscene_trainer_matrix
```

The regenerated audit is:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_goal_progress_audit/summary.json
```

It now proves `24` focused progress rows and still keeps:

```text
full_goal_completion = open
is_goal_complete = false
```

The open gap wording now says the current proof includes focused artifacts,
checked-in high-motion probes, and a small source-distinct real-video matrix,
not broad real-scene quality acceptance.

## Verification

Commands run:

```text
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 uv run --with pytest python -m pytest tests/test_star_uvt_projective_real_video_multiscene_trainer_matrix.py -q
# 8 passed in 1.14s

.venv/bin/python research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_trainer_matrix.py --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_trainer_matrix/summary.json
# verified

.venv/bin/python research_experiments/star_uvt_feature_tubes/projective_goal_progress_audit.py --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_goal_progress_audit/summary.json --verify-current-inputs
# verified against current inputs

PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 uv run --with pytest python -m pytest tests/test_star_uvt_projective_real_video_multiscene_trainer_matrix.py tests/test_star_uvt_projective_goal_progress_audit.py -q
# 39 passed in 1.00s

PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 uv run --with pytest python -m pytest tests/test_star_uvt_projective_real_video_multiscene_trainer_matrix.py tests/test_star_uvt_projective_real_video_guarded_support_matrix_report.py tests/test_star_uvt_projective_goal_progress_audit.py -q
# 48 passed in 1.41s
```

## Interpretation

This is a useful widening step, not the end. It shows the guarded
projective-interval trainer mechanics survive three distinct real-video
sources while preserving cadence loss and cutting rebuilds. It does not prove
general image quality, long training, high resolution, or a full compiled
adjoint trainer replacement.

Next useful falsification step: keep the same source-distinct matrix shape but
increase duration/resolution/training steps and add image-error or quality
thresholds, rather than only trainer-mechanics checks.
