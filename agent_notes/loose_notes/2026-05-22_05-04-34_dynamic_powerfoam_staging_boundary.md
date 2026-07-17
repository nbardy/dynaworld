# Dynamic PowerFoam Staging Boundary

## What Changed

- Added `src/train/dynamic_powerfoam_staging.py` for Dynamic PowerFoam training
  stage controls:
  - `camera_curriculum_active_frames(...)`
  - `apply_training_stage(...)`
- `src/train/train_dynamic_powerfoam_metal.py` imports the helpers instead of
  defining them inline. The old trainer-module names remain available through
  compatibility imports.
- `tests/test_dynamic_powerfoam_metal.py` imports the staging helpers directly,
  so stage-control tests no longer treat the full trainer as the helper
  namespace.

## Why This Slice

Static-only warmup, no-repaint warmup, and camera active-frame curriculum are
small control policies around a model object. They are not rasterizer code,
W&B policy, checkpoint logic, or model parameterization. Moving them into a
small staging module keeps the trainer loop readable and gives the tested stage
contract an explicit owner.

## Validation

```bash
PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with pytest python -m pytest tests/test_dynamic_powerfoam_metal.py -q
```

Result:

```text
33 passed in 5.31s
```

```bash
PYTHONPATH=src/train:third_party/powerfoam-metal .venv/bin/python -m py_compile \
  src/train/dynamic_powerfoam_staging.py \
  src/train/dynamic_powerfoam_rendering.py \
  src/train/dynamic_powerfoam_initialization.py \
  src/train/dynamic_powerfoam_camera.py \
  src/train/dynamic_powerfoam_temporal.py \
  src/train/train_dynamic_powerfoam_metal.py \
  tests/test_dynamic_powerfoam_metal.py
```

Result: passed.

## Next

- The remaining top-level helpers in `train_dynamic_powerfoam_metal.py` are now
  mostly train-lane artifact/report policy: `log_artifacts(...)` and
  `dynamic_geometry_summary(...)`. Leave those local unless a second script or
  trainer needs the exact same schema.
- Further cleanup should probably switch back to broader live-file scans for
  trainer-as-helper imports instead of continuing to shave this one trainer.
