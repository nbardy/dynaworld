# Dynamic PowerFoam visualizer helper import cleanup

## Context

`src/train/visualize_camera_scene_diagnostic.py` still imported
`train_dynamic_powerfoam_metal` as a namespace when decoding Dynamic PowerFoam
checkpoints. The diagnostic genuinely needs the model classes, but it did not
need to reach through the trainer for pure config/raster helpers.

## Change

- `TOKEN_RBF_FEATURE_MODE` now comes from `dynamic_powerfoam_metal_config.py`.
- Dynamic PowerFoam raster config construction now comes from
  `powerfoam_raster_config.py`.
- The dynamic trainer import is limited to `DynamicMetalPowerFoamVideo` and
  `TokenDynamicPowerFoamFeatures`, which are still structural model classes in
  the trainer module.

## Validation

- `py_compile` covered `visualize_camera_scene_diagnostic.py` and the owner
  modules.
- `visualize_camera_scene_diagnostic.py --help` was checked.
