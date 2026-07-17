# STAR Rendered Probe Checkpoint Helper

## Goal

Continue the trainer modularization pass by removing one more STAR
trainer-local payload/write boundary.

## Change

- Added `star_uvt_checkpoints.save_rendered_feature_rgb_probe_checkpoint(...)`.
- Updated `train_star_uvt_rendered_feature_rgb_probe.py` to call that helper
  instead of importing `checkpoint_utils.atomic_torch_save(...)` and assembling
  the checkpoint payload inline.
- Added a focused checkpoint test that verifies the rendered-probe payload:
  serialized config, resume metadata, colorizer-init metadata, optimizer state,
  optional model state, sparse-sample loss, and full loss.
- Updated `CODE_ORGANIZATION.md` and `TODO/trainer_landscape_unification.md`
  to record that rendered-feature RGB probe checkpoint saves now live in
  `star_uvt_checkpoints.py`.

## Why This Boundary

`star_uvt_checkpoints.py` already owned STAR training checkpoints, target-grid
RGB probe checkpoints, feature-to-RGB probe loading, optimizer LR helpers, and
rendered-probe model-only resume metadata. The rendered-feature RGB probe was
the remaining STAR probe that still wrote a checkpoint payload directly from
the trainer. Moving it keeps the trainer focused on the sparse-render objective
and row/media orchestration.

## Validation

- `rtk .venv/bin/python -m py_compile src/train/star_uvt_checkpoints.py src/train/train_star_uvt_rendered_feature_rgb_probe.py tests/test_star_uvt_checkpoints.py`
- `PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_star_uvt_checkpoints.py -q`
