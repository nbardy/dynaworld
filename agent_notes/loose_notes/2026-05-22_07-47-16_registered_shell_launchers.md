# Registered Shell Launchers

## Goal

Continue the trainer-interface cleanup by removing live shell-script bypasses
around `src/train/train.py` where the config arch already resolves through
`trainer_registry`.

## What changed

- Rerouted older local Token-GS/precomputed launch scripts through
  `PYTHONPATH=src/train uv run python src/train/train.py <config>`.
- Touched launchers:
  - `src/train_scripts/train_static_dynamic_vjepa_features_ablation.sh`
  - `src/train_scripts/train_video_temporal_ablation_suite.sh`
  - `src/train_scripts/train_local_mac_30_clip_baseline.sh`
  - `src/train_scripts/train_local_mac_30_clip_vjepa2_256_baseline.sh`
  - `src/train_scripts/train_single_video_pretrain_100_64f.sh`
  - `src/train_scripts/train_single_video_pretrain_all_youtube_64f_512.sh`
  - `src/train_scripts/train_compare_vjepa2_fpc16_256_16f_single_overfit.sh`
  - `src/train_scripts/train_fast_overfit_star_uvt_and_dynamic_gsplat.sh`
  - `research_experiments/dynamic_foam/run_powerfoam_external_blockers.py`
- The STAR compact visual and native full-cell feature-overfit configs both
  declare `arch=star_uvt_feature_overfit`, which already routes through
  `train_star_uvt_feature_overfit.run_training`; those two calls now go through
  `train.py` as well.

## Validation

- `bash -n` passed for all touched launchers.
- `rg` found no remaining direct `train_video_token_implicit_dynamic.py`,
  `train_precomputed_feature_implicit_dynamic.py`, multicam, mixed, or
  relative-pose trainer launches under `src/train_scripts`.
- A fake-runner `trainer_registry.run_config(...)` smoke covered 31 touched
  config paths and confirmed they dispatch to registered modules without
  launching training.
- A focused STAR route audit confirmed the compact visual and native full-cell
  configs dispatch to `train_star_uvt_feature_overfit.run_training` through
  `arch=star_uvt_feature_overfit`.
- The Dynamic Foam external-blocker PowerFoam Metal command now points at
  `src/train/train.py` while preserving `PYTHONPATH=src/train:third_party/powerfoam-metal`.
- `PYTHONPATH=src/train uv run --with pytest python -m pytest
  tests/test_trainer_registry.py -q` passed: 9 tests.
- Targeted `git diff --check` passed for the tracked touched shell files.

## Current state

The ordinary shell launch layer now follows the same config-driven entrypoint
for Token-GS, precomputed-feature, STAR UVT, and dynamic-gsplat registered
trainer configs. Remaining direct training script calls should be treated as
exceptions that need an explicit registry/external-launcher reason.
