# Train JSON I/O Helper

## Goal

Continue the modularization goal in active training-path code by removing
repeated low-level `json.loads(path.read_text(...))` calls from data/camera
loaders without moving domain validation out of the owning modules.

## Change

- Added `src/train/json_io.py` with `load_json(path, encoding="utf-8")`.
- Routed train-local data/camera JSON reads through it:
  - `sequence_data.load_sequence_metadata(...)`
  - `sequence_data.load_camera_sequence(...)`
  - `multicam_video_data.deepview_models_by_name(...)`
  - `multicam_video_data.load_camxtime_camera_data(...)`
  - AIST camera settings and ViVo calibration loading in `multicam_video_data.py`
  - `dynamic_powerfoam_camera.load_teacher_camera_to_world(...)`
- Kept domain contracts local: manifest JSONL line-number errors,
  sequence/camera record shapes, multicam calibration semantics, and teacher
  camera shape validation still live in their existing modules.
- Added a focused `load_json(...)` test to `tests/test_config_and_dataset_io.py`.

## Validation

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:. .venv/bin/python -m py_compile \
  src/train/json_io.py src/train/sequence_data.py src/train/multicam_video_data.py \
  src/train/dynamic_powerfoam_camera.py tests/test_config_and_dataset_io.py

PYTHONPATH=src/train:. uv run --with pytest python -m pytest \
  tests/test_config_and_dataset_io.py \
  tests/test_sequence_data_single_frame.py \
  tests/test_multicam_video_data.py \
  tests/test_dynamic_powerfoam_metal.py -q
```

Results:

- `py_compile` passed.
- Focused data/camera test set passed: `57 passed in 6.42s`.
- Follow-up scan for `json.loads(...read_text...)` in the touched modules now
  leaves only the local JSONL manifest parser in `sequence_data.py` and
  `json_io.load_json(...)` itself.

## Handoff

This is a data-I/O boundary cleanup. It does not alter dataset semantics or
prove any trainer convergence; the broader goal still needs benchmark evidence
for the unified trainer paths.
