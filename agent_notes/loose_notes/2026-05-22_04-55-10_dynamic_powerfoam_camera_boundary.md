# Dynamic PowerFoam Camera Boundary

## What Changed

- Added `src/train/dynamic_powerfoam_camera.py` for Dynamic PowerFoam camera
  helpers:
  - `build_camera_decoder(...)`
  - `camera_param_group(...)`
  - `camera_regularization(...)`
  - `compact_camera_metrics(...)`
  - `load_teacher_camera_to_world(...)`
  - `camera_teacher_alignment_loss(...)`
  - `prefit_camera_decoder_from_teacher(...)`
  - `decoded_powerfoam_rays(...)`
- `src/train/train_dynamic_powerfoam_metal.py` imports those helpers instead of
  defining the camera block inline. The old trainer-module names remain
  available through imports, preserving compatibility for existing scripts.
- `tests/test_dynamic_powerfoam_metal.py` now imports camera regularization and
  teacher-prefit helpers from `dynamic_powerfoam_camera.py` directly.
- `src/train/visualize_camera_scene_diagnostic.py` now imports
  `build_camera_decoder(...)` from the helper module instead of reaching through
  the full dynamic trainer namespace.

## Why This Slice

The camera block was not train-loop policy. It was construction, optimizer
grouping, regularization, teacher-pose loading, and ray assembly around the
implicit camera decoder. Keeping it in the trainer made diagnostics and tests
import a large train loop for helper behavior. The new module follows the same
compatibility pattern as the config, colorizer, raster-config, geometry, and
temporal splits.

## Validation

```bash
PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with pytest python -m pytest tests/test_dynamic_powerfoam_metal.py -q
```

Result:

```text
33 passed in 5.04s
```

```bash
PYTHONPATH=src/train:third_party/powerfoam-metal .venv/bin/python -m py_compile \
  src/train/dynamic_powerfoam_camera.py \
  src/train/dynamic_powerfoam_temporal.py \
  src/train/train_dynamic_powerfoam_metal.py \
  src/train/visualize_camera_scene_diagnostic.py \
  tests/test_dynamic_powerfoam_metal.py
```

Result: passed.

## Remaining Cleanup

- `train_dynamic_powerfoam_metal.py` still owns the dynamic model classes and
  train/eval loop. That is fine for now; further extraction should target
  small helper clusters only when diagnostics or tests need them directly.
- The next plausible helper slice is dynamic PowerFoam init geometry
  (`initialize_powerfoam_normals`, orbit-video initialization, camera/world
  transforms), but that should be split only if it reduces a live dependency or
  unlocks a focused test.
