# Dynamic PowerFoam Initialization Boundary

## What Changed

- Added `src/train/dynamic_powerfoam_initialization.py` for Dynamic PowerFoam
  initialization geometry:
  - `transform_powerfoam_frame_to_camera(...)`
  - `transform_points_camera_to_world(...)`
  - `initialize_powerfoam_normals(...)`
  - `initialize_full_powerfoam_from_orbit_video(...)`
- `src/train/train_dynamic_powerfoam_metal.py` imports those helpers instead of
  defining the orbit-camera initialization block inline.
- The trainer still imports the helper names at module scope, preserving the old
  compatibility surface for callers that reached through
  `train_dynamic_powerfoam_metal`.

## Why This Slice

This was pure initialization and coordinate-transform logic shared by both
Dynamic PowerFoam model variants. It did not depend on optimizer state, W&B,
logging, scheduler policy, or the train loop. Moving it out keeps the trainer
focused on model parameterization plus training/eval orchestration, and puts
orbit-camera initialization next to the other light Dynamic PowerFoam helper
modules.

## Validation

```bash
PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with pytest python -m pytest tests/test_dynamic_powerfoam_metal.py -q
```

Result:

```text
33 passed in 4.97s
```

```bash
PYTHONPATH=src/train:third_party/powerfoam-metal .venv/bin/python -m py_compile \
  src/train/dynamic_powerfoam_initialization.py \
  src/train/dynamic_powerfoam_camera.py \
  src/train/dynamic_powerfoam_temporal.py \
  src/train/train_dynamic_powerfoam_metal.py \
  src/train/visualize_camera_scene_diagnostic.py \
  tests/test_dynamic_powerfoam_metal.py
```

Result: passed.

## Remaining Cleanup

- `train_dynamic_powerfoam_metal.py` is now much more model/training focused,
  but still large because it owns two model classes and the full trainer loop.
- Further extraction should be live-dependency driven. The next reasonable
  candidates are small shared result/logging helpers or parameter-group helpers
  if another Dynamic PowerFoam surface needs them directly.
