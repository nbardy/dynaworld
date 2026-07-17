# Dynamic PowerFoam Feature Init Boundary

## Goal

Continue the trainer modularization pass by moving one pure Dynamic PowerFoam
initialization helper out of the trainer file and into the initialization module
that already owns related geometry and video-init helpers.

## Change

- Moved `make_texel_feature_init(...)` from
  `src/train/train_dynamic_powerfoam_metal.py` to
  `src/train/dynamic_powerfoam_initialization.py`.
- `TokenDynamicPowerFoamFeatures` still calls the same helper with the same
  arguments; only the owner module changed.
- Updated `CODE_ORGANIZATION.md` and `TODO/trainer_landscape_unification.md`
  to record that token-feature texel initialization now belongs to
  `dynamic_powerfoam_initialization.py`.

## Why This Boundary

`make_texel_feature_init(...)` is pure initialization logic: allocate feature
channels, optionally seed RGB channels from color or logits, add feature noise,
and return contiguous feature tensors. Keeping that in the trainer made the
trainer file act as a helper namespace. Moving it beside the orbit-video and
normal initialization helpers keeps initialization policy in one module without
touching model parameterization or training behavior.

## Validation Plan

- Compile the dynamic PowerFoam trainer, initialization module, and focused
  dynamic PowerFoam tests.
- Run the focused Dynamic PowerFoam pytest gate.
- Search for the helper to confirm its definition moved and the trainer imports
  it from `dynamic_powerfoam_initialization.py`.
- Run whitespace and diff checks on touched files.

## Validation Results

- `rtk .venv/bin/python -m py_compile src/train/train_dynamic_powerfoam_metal.py src/train/dynamic_powerfoam_initialization.py tests/test_dynamic_powerfoam_metal.py` passed.
- `PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with pytest python -m pytest tests/test_dynamic_powerfoam_metal.py -q` passed: `33 passed`.
- `rtk rg -n "def make_texel_feature_init|make_texel_feature_init\\(" ...` shows the helper defined in `dynamic_powerfoam_initialization.py` and called from `train_dynamic_powerfoam_metal.py`.
- Touched-file trailing-whitespace scan passed.
- `rtk git diff --check -- src/train/train_dynamic_powerfoam_metal.py src/train/dynamic_powerfoam_initialization.py CODE_ORGANIZATION.md TODO/trainer_landscape_unification.md` passed for tracked touched files.
