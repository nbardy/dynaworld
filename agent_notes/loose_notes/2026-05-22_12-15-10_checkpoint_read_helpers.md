# Checkpoint Read Helpers

## Goal

Continue the trainer/interface cleanup by removing repeated checkpoint payload
shape checks from live code without turning checkpoint schemas into a broad
framework.

## Changes

- Added read-side helpers to `src/train/checkpoint_utils.py`:
  - `load_torch_checkpoint(...)`
  - `load_checkpoint_mapping(...)`
  - `model_state_dict_from_checkpoint(...)`
- Routed STAR UVT checkpoint mapping loads through
  `load_checkpoint_mapping(...)` while keeping STAR-specific required-key,
  colorizer, model, and optimizer semantics local to
  `src/train/star_uvt_checkpoints.py`.
- Routed `src/train/visualize_camera_scene_diagnostic.py` through
  `load_torch_checkpoint(...)` and `model_state_dict_from_checkpoint(...)`
  instead of carrying a local `_checkpoint_state(...)` helper and repeated raw
  `torch.load(...)` calls.
- Removed stale `json` imports from `src/train/multicam_video_data.py` and
  `src/train/dynamic_powerfoam_camera.py` left after the JSON/JSONL loader
  cleanup.
- Added `tests/test_checkpoint_utils.py` for the shared mapping validation and
  wrapped/raw model-state contract.

## Validation

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:. .venv/bin/python -m py_compile ...`
  passed for the changed train modules and new test.
- `PYTHONPATH=src/train:. uv run --with pytest python -m pytest tests/test_checkpoint_utils.py tests/test_star_uvt_checkpoints.py -q`
  passed: `5 passed`.
- `PYTHONPATH=src/train:. uv run --with pytest python -m pytest tests/test_config_and_dataset_io.py tests/test_multicam_video_data.py tests/test_dynamic_powerfoam_metal.py -q`
  passed: `53 passed`.
- `PYTHONPATH=src/train:. .venv/bin/python src/train/visualize_camera_scene_diagnostic.py --help`
  passed.

## Notes

This deliberately does not centralize checkpoint schemas. The useful shared
boundary is only the file/payload shape: loading a torch checkpoint, requiring a
mapping when the caller expects one, and accepting both wrapped `model` state
dicts and raw state dicts for diagnostics. Trainer-specific checkpoint keys
stay with the trainer/helper that owns the model.
