# Dynamic PowerFoam test helper import cleanup

## Context

`tests/test_dynamic_powerfoam_metal.py` still imported Dynamic PowerFoam
defaults, config resolution, and raster config construction through
`train_dynamic_powerfoam_metal.py`.

The owning modules already exist:

- `dynamic_powerfoam_metal_config.py` owns defaults and `resolve_config(...)`.
- `powerfoam_raster_config.py` owns Dynamic PowerFoam Metal raster config
  construction.

The model classes still live in the trainer file, so this cleanup does not try
to move them.

## Change

- Tests now import `LOSS_DEFAULTS` and `resolve_config(...)` from
  `dynamic_powerfoam_metal_config.py`.
- Tests now import `make_dynamic_powerfoam_metal_raster_config` from
  `powerfoam_raster_config.py` under the existing `make_raster_config` local
  name.
- Tests still import `DynamicMetalPowerFoamVideo` and
  `TokenDynamicPowerFoamFeatures` from `train_dynamic_powerfoam_metal.py`
  because those are structural model classes, not helper functions.

## Validation

- Focused Dynamic PowerFoam pytest gate passed.
