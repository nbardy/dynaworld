# Checkpoint Mapping Follow-Up

## Goal

Continue the checkpoint read-side cleanup after adding shared checkpoint shape
helpers.

## Changes

- Routed `MulticamRelativePoseImplicitTrainer._load_checkpoint_if_configured`
  through `checkpoint_utils.load_checkpoint_mapping(...)` instead of direct
  `torch.load(...)` plus local dict validation.
- Routed `star_uvt_common.load_colorizer_init_checkpoint(...)` through the same
  mapping helper while keeping the required `colorizer` key and colorizer state
  validation local.
- Updated the organization docs to state that the checkpoint helper now covers
  STAR UVT checkpoint loads, STAR UVT colorizer-init loads, relative-pose
  checkpoint resume, and camera-scene diagnostics.

## Validation

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:. .venv/bin/python -m py_compile ...`
  passed for the touched modules/tests.
- `PYTHONPATH=src/train:. uv run --with pytest python -m pytest tests/test_star_uvt_common.py tests/test_star_uvt_feature_target_adapter.py tests/test_multicam_relative_pose_trainer.py -q`
  passed: `53 passed`.

## Notes

This is still intentionally narrow. `checkpoint_utils` owns file/payload shape
checks; trainer-specific required keys, optional colorizer/camera-rig payloads,
and compatibility missing-key policy stay with the owning trainer/helper.
