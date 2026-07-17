# STAR UVT Checkpoint Helper Cleanup

## Context

Continuation of the trainer modularization goal after splitting STAR UVT
feature targets, sparse target-grid helpers, RGB render chunks, and colorizer
construction. The next repeated contract was checkpoint handling:

- `train_star_uvt_feature_overfit.py` owned STAR training checkpoint
  save/load, optimizer LR helpers, and resume metadata.
- `train_star_uvt_rendered_feature_rgb_probe.py` separately loaded the same
  STAR model state from those checkpoints and manually extracted row metadata.
- `train_star_uvt_feature_rgb_probe.py` saved target-grid RGB probe
  checkpoints that `train_star_uvt_feature_overfit.py` later loaded through the
  `feature_target.rgb_probe_checkpoint` path.
- Tests for checkpoint behavior imported private helpers from the large overfit
  trainer even though the behavior is a reusable checkpoint contract.

## Changes

- Added `src/train/star_uvt_checkpoints.py`.
  - `save_star_training_checkpoint(...)`
  - `load_star_training_checkpoint(...)`
  - `load_star_model_from_training_checkpoint(...)`
  - `save_feature_rgb_probe_checkpoint(...)`
  - `load_feature_rgb_probe_checkpoint(...)`
  - `optimizer_lrs(...)`
  - `set_optimizer_lr(...)`
- Rewired `train_star_uvt_feature_overfit.py` to import checkpoint helpers
  instead of defining training checkpoint helpers or RGB probe checkpoint
  loading locally.
- Rewired `train_star_uvt_feature_rgb_probe.py` to save target-grid RGB probe
  checkpoints through the shared helper.
- Rewired `train_star_uvt_rendered_feature_rgb_probe.py` to load frozen or
  trainable STAR model state through the shared checkpoint helper.
- Updated checkpoint roundtrip tests to import the training checkpoint contract
  from `star_uvt_checkpoints.py`.
- Added `tests/test_star_uvt_checkpoints.py` for the rendered-feature probe
  contract and target-grid RGB probe checkpoint contract: model-only load,
  freeze behavior, row metadata extraction, colorizer load, and payload
  metadata.
- Updated `CODE_ORGANIZATION.md` and `TODO/trainer_landscape_unification.md`.

## Validation

```bash
rtk env PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/star_uvt_checkpoints.py \
  src/train/train_star_uvt_feature_overfit.py \
  src/train/train_star_uvt_feature_rgb_probe.py \
  src/train/train_star_uvt_rendered_feature_rgb_probe.py \
  tests/test_star_uvt_checkpoints.py \
  tests/test_star_uvt_feature_target_adapter.py
```

Passed.

```bash
rtk env PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_checkpoints.py \
  tests/test_star_uvt_colorizers.py \
  tests/test_star_uvt_feature_target_adapter.py \
  tests/test_star_uvt_feature_rgb_probe.py -q
```

Passed: `47 passed in 1.73s`.

## Remaining Work

The overfit trainer is still large. The next useful cleanup should not be a
generic framework; it should split another real shared contract. Candidate
boundaries:

- Visual-support diagnostics, if those helpers stay used across more than one
  STAR probe/trainer.
- Config normalization tables, only if repeated required/default sections keep
  growing across STAR scripts.
