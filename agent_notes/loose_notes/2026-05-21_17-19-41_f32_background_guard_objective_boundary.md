# F32 Background Guard Objective Boundary

Date: 2026-05-21

## Goal

Continue trainer modularization by auditing the F32/colorizer/random-background
path after the mixed trainer bridge. The main question was whether the mixed
same-view plus heldout-view route inherits the shared feature-splatting
composition contract or accidentally reintroduces a background/colorizer cheat.

## Findings

- Base token-GS reconstruction, multicam train-view reconstruction,
  multicam heldout reconstruction, and the mixed trainer all render through
  `RGBReconObjective.render_view(...)`.
- That means F=32 feature maps are colorized before RGB reconstruction, and RGB
  background is composed after the colorizer with alpha.
- The weak spot was not composition itself. The weak spot was duplicated trainer
  guards of the form `feature_dim != 3 and rendered.alpha is None`.

## What changed

- Added `RGBReconObjective.require_alpha_for_feature_background(...)`.
- Replaced the duplicated F32 alpha/background checks in:
  - `train_video_token_implicit_dynamic.py`
  - `train_multicam_precomputed_feature_implicit_dynamic.py`
  - `train_multicam_relative_pose_implicit_dynamic.py`
- Added a focused objective test that proves the guard raises when F32 training
  has an RGB background but no raster alpha.

This keeps behavior the same but moves the invariant to the objective boundary:
feature-splat training with an RGB background still requires alpha-aware render
output.

## Validation

- Compile check passed for:
  `src/train/objective/objective.py`,
  `src/train/train_video_token_implicit_dynamic.py`,
  `src/train/train_multicam_precomputed_feature_implicit_dynamic.py`,
  `src/train/train_multicam_relative_pose_implicit_dynamic.py`,
  `tests/test_rgb_recon_objective.py`.
- Focused pytest:
  `PYTHONPATH=src/train:. rtk uv run --with pytest python -m pytest tests/test_rgb_recon_objective.py tests/test_objective_background_and_composition.py tests/test_star_uvt_background_cheat_diagnostic.py tests/test_temporal_sampling.py tests/test_mixed_same_heldout_trainer.py tests/test_mixed_data_scheduler.py tests/test_multicam_video_data.py tests/test_pipeline_helpers.py tests/test_sequence_data_single_frame.py tests/test_pipeline_diagnostics.py tests/test_train_logging.py tests/test_config_factory_helpers.py -q`
  passed with `76 passed in 1.76s`.
- Cheap single-cam F32 smoke passed:
  `wandb/offline-run-20260521_171908-pgv52pgm`.
- Checked-in mixed same-heldout smoke passed again:
  `wandb/offline-run-20260521_171924-mkj9af97`.

## Remaining

- This proves the mixed route inherits the shared composition/guard mechanics;
  it does not prove convergence or visual quality.
- Next modularization target should be the duplicate multicam train/heldout
  reconstruction loop shape, or a longer W&B-enabled mixed media run if the
  priority shifts from code organization to training evidence.
