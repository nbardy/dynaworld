# Dynamic PowerFoam Colorizer Wrapper Removal

## Goal

Continue the trainer modularization pass by removing a pass-through helper from
`train_dynamic_powerfoam_metal.py` after `powerfoam_colorizers.py` became the
owning module for Dynamic PowerFoam colorizer construction.

## Change

- Removed local `build_colorizer(cfg, device)` from
  `src/train/train_dynamic_powerfoam_metal.py`.
- The train loop now calls
  `powerfoam_colorizers.build_dynamic_powerfoam_colorizer(...)` directly with
  `feature_dynamic_mode=TOKEN_RBF_FEATURE_MODE`.
- Updated `CODE_ORGANIZATION.md` and `TODO/trainer_landscape_unification.md`
  to record that the compatibility wrapper is gone.

## Why This Boundary

The wrapper no longer owned any behavior. The defaults, RGB identity init, mode
gate, and `FeatureToColor` constructor already live in `powerfoam_colorizers.py`.
Removing the pass-through keeps the trainer loop explicit while reducing
trainer-as-helper surface.

## Validation Plan

- Compile the dynamic PowerFoam trainer and colorizer module.
- Run focused Dynamic PowerFoam tests.
- Search for remaining local `build_colorizer(...)` definitions in the dynamic
  trainer.
- Run whitespace and diff checks on touched files.

## Validation Results

- `rtk .venv/bin/python -m py_compile src/train/train_dynamic_powerfoam_metal.py src/train/powerfoam_colorizers.py tests/test_dynamic_powerfoam_metal.py` passed.
- `PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with pytest python -m pytest tests/test_dynamic_powerfoam_metal.py -q` passed: `33 passed`.
- `rtk rg -n "def build_colorizer|build_colorizer\\(|build_dynamic_powerfoam_colorizer" ...` shows no local `def build_colorizer(...)` in the dynamic trainer; the train loop calls `build_dynamic_powerfoam_colorizer(...)` directly.
- Touched-file trailing-whitespace scan passed.
- `rtk git diff --check -- src/train/train_dynamic_powerfoam_metal.py CODE_ORGANIZATION.md TODO/trainer_landscape_unification.md` passed for tracked touched files.
