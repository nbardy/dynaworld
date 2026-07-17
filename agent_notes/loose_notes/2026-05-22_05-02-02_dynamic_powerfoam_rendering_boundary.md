# Dynamic PowerFoam Rendering Boundary

## What Changed

- Added `src/train/dynamic_powerfoam_rendering.py` for Dynamic PowerFoam
  render/composition helpers:
  - `sample_background(...)`
  - `render_features_to_rgb(...)`
  - `render_all(...)`
  - `per_frame_reconstruction_metrics(...)`
  - `temporal_alpha_metrics(...)`
- `src/train/train_dynamic_powerfoam_metal.py` imports these helpers instead of
  defining them inline. This preserves the old trainer-module compatibility
  names while keeping pure render math out of the train-loop file.
- `tests/test_dynamic_powerfoam_metal.py` imports
  `sample_background(...)` and `render_features_to_rgb(...)` from the helper
  module directly.

## Why This Slice

Dynamic PowerFoam rendering is not identical to the token-GS objective
composition path. It handles premultiplied raster outputs and can normalize
features by alpha before colorization, then blend RGB after colorization. That
contract is still shared within Dynamic PowerFoam train/eval and tests, so it
belongs in a small explicit helper module rather than in the full trainer.

The trainer keeps W&B/media artifact assembly local for now because payload
names, preview files, MP4 outputs, and camera/state metric selection are
specific to this train lane.

## Validation

```bash
PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with pytest python -m pytest tests/test_dynamic_powerfoam_metal.py -q
```

Result:

```text
33 passed in 5.41s
```

```bash
PYTHONPATH=src/train:third_party/powerfoam-metal .venv/bin/python -m py_compile \
  src/train/dynamic_powerfoam_rendering.py \
  src/train/dynamic_powerfoam_initialization.py \
  src/train/dynamic_powerfoam_camera.py \
  src/train/dynamic_powerfoam_temporal.py \
  src/train/train_dynamic_powerfoam_metal.py \
  tests/test_dynamic_powerfoam_metal.py
```

Result: passed.

## Next

- `apply_training_stage(...)` and `camera_curriculum_active_frames(...)` are
  also pure enough to split, but they mutate model/camera controls and are
  currently only used by this trainer. Extract only if another diagnostic or
  trainer path needs the same staging contract.
- Keep artifact logging local unless a second PowerFoam-family trainer needs
  the exact same W&B payload and file layout.
