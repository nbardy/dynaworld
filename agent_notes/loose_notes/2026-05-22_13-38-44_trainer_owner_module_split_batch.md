# Trainer Owner-Module Split Batch

## Context

The trainer modularization goal is converging on a mechanical pattern:

- owner module contains warm-path implementation and public runner
- historical `train_*.py` file remains a thin CLI/backcompat wrapper
- `trainer_registry.py` routes train.py dispatch to the owner module
- docs record smoke evidence instead of only saying the split happened

This batch used subagents for disjoint file sets while the main thread handled
registry, docs, and runtime validation.

## Completed Splits

### STAR UVT rendered-feature RGB probe

- Owner: `src/train/star_uvt_rendered_feature_rgb_probe_trainer.py`
- Wrapper: `src/train/train_star_uvt_rendered_feature_rgb_probe.py`
- Registry arch: `star_uvt_rendered_feature_rgb_probe`
- Runner: `run_probe`
- Wrapper re-exports: `main`, `run_probe`

Validation:

- `py_compile` passed for the owner, wrapper, registry, and registry tests.
- `tests/test_star_uvt_feature_rgb_probe.py`,
  `tests/test_star_uvt_checkpoints.py`, and `tests/test_trainer_registry.py`
  passed: `23 passed`.
- `tests/test_train_cli.py` passed: `8 passed`.
- Wrapper identity smoke passed.
- One-step MPS runtime smoke through `src/train/train.py` passed with:
  4 frames, 64px, `sample_grid_shape=[4,16,16]`, 8192 tubes retained for
  checkpoint compatibility, W&B disabled, no media/checkpoint writes, and row
  output under `/tmp`.

### STAR UVT feature overfit

- Owner: `src/train/star_uvt_feature_overfit_trainer.py`
- Wrapper: `src/train/train_star_uvt_feature_overfit.py`
- Registry arch: `star_uvt_feature_overfit`
- Runner: `run_training`
- Wrapper re-exports: `main`, `run_training`, `_render_rgb_probe_chunks`,
  `_assert_requirements`

Validation:

- Worker `py_compile` and import identity smoke passed.
- Main-thread `py_compile` passed for owner, wrapper, registry, and registry
  tests.
- Main-thread wrapper/registry identity smoke passed.
- Focused STAR UVT tests passed:
  `tests/test_trainer_registry.py`,
  `tests/test_star_uvt_feature_target_adapter.py`,
  `tests/test_star_uvt_sparse_grid.py`, and `tests/test_star_uvt_outputs.py`
  reported `54 passed`.
- One-step MPS runtime smoke through `src/train/train.py` passed with:
  8 frames, 64px, 512 tubes, direct-atomic rendering, W&B disabled, no
  media/checkpoint writes, and row output under `/tmp`. It saw gradient flow
  through STAR parameters and the colorizer and reported zero tile overflow.
  Row `pass=false` is expected for a one-step smoke with loss-decrease disabled.

### PowerFoam Metal

- Owner: `src/train/powerfoam_metal_trainer.py`
- Wrapper: `src/train/train_powerfoam_metal.py`
- Registry arch: `powerfoam_metal`
- Runner: `run_training`
- Wrapper re-exports: `main`, `MetalPowerFoamVideo`, `run_training`

Validation:

- Worker `py_compile`, import identity smoke, focused pytest gate, and
  `git diff --check` passed.
- Main-thread `py_compile` passed for owner, wrapper, registry, registry tests,
  and PowerFoam direct tests.
- Main-thread wrapper/registry identity smoke passed.
- Main-thread focused PowerFoam/registry gate passed:
  `tests/test_powerfoam_direct.py`, `tests/test_multicam_video_data.py`, and
  `tests/test_trainer_registry.py` reported `66 passed, 1 skipped`.
- One-step MPS runtime smoke through `src/train/train.py` passed with a `/tmp`
  output dir, final checkpoint write, and eval L1 improving from `0.07646` at
  step 0 to `0.07628` at step 1.

### Dynamic PowerFoam Metal

- Owner: `src/train/dynamic_powerfoam_metal_trainer.py`
- Wrapper: `src/train/train_dynamic_powerfoam_metal.py`
- Registry arch: `dynamic_powerfoam_metal`
- Runner: `run_training`
- Wrapper re-exports: `main`, `DynamicMetalPowerFoamVideo`,
  `TokenDynamicPowerFoamFeatures`, `run_training`

Validation:

- Worker `py_compile`, import identity smoke, and focused pytest passed:
  `34 passed`.
- Main-thread `py_compile` passed for owner, wrapper, registry, registry tests,
  PowerFoam dispatch tests, and Dynamic PowerFoam tests.
- Main-thread wrapper/registry identity smoke passed.
- Main-thread focused Dynamic/registry tests passed:
  `tests/test_dynamic_powerfoam_metal.py`,
  `tests/test_powerfoam_direct.py::test_powerfoam_direct_config_dispatches_to_trainer`,
  and `tests/test_trainer_registry.py` reported `47 passed`.
- One-step MPS runtime smoke through `src/train/train.py` passed for the RBF
  branch with 4 frames, 64 cells, W&B disabled, and `/tmp` output.
- One-step MPS runtime smoke through `src/train/train.py` passed for the
  token/F32 branch with 4 frames, 64 cells, W&B disabled, `/tmp` output,
  `train_background_mode=random_rgb`, and `eval_background_mode=fixed_rgb`.

## Notes

No `key_learnings.md` update was needed. The useful lesson here is not a new
shader/training trick; it is confirmation that the owner/wrapper split can be
mechanical when reusable helpers have already been extracted.
