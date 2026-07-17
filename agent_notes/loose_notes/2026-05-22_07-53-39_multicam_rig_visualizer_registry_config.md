# Multicam Rig Visualizer Registry Config Boundary

## Goal

Remove one more concrete trainer-as-helper import from a live utility script.
The multicam rig visualizer needed the multicam config resolver, not the
trainer class.

## Change

- Updated `src/dataset_scripts/visualize_multicam_rig.py` to call
  `trainer_registry.resolve_config_for_arch(load_config_file(config_path),
  config_path)`.
- Removed the direct
  `from train_multicam_precomputed_feature_implicit_dynamic import
  MulticamPrecomputedFeatureImplicitTrainer` import from that script.
- Left all camera-rig construction and HTML/JSON visualization logic local to
  the script.

## Validation

- `rtk .venv/bin/python -m py_compile src/dataset_scripts/visualize_multicam_rig.py`
  passed.
- `rtk sh -lc 'PYTHONPATH=src/train .venv/bin/python
  src/dataset_scripts/visualize_multicam_rig.py --help'` passed.
- A focused registry smoke resolved
  `local_mac_multicam_deepview_3cam_train2_test1_rgb_pyramid_static_dynamic_smoke_32_2f_64splats.jsonc`
  through `train_multicam_precomputed_feature_implicit_dynamic`.
- `rtk git diff --check -- src/dataset_scripts/visualize_multicam_rig.py`
  passed.

## Remaining Direct Imports

The live scan now leaves structural trainer inheritance imports, structural
tests, and two Dynamic Foam diagnostics that need `MetalPowerFoamVideo` itself:

- `src/train/train_precomputed_feature_implicit_dynamic.py` inherits the base
  Token-GS trainer.
- `src/train/train_multicam_precomputed_feature_implicit_dynamic.py`,
  `src/train/train_multicam_relative_pose_implicit_dynamic.py`, and
  `src/train/train_mixed_same_heldout_implicit_dynamic.py` preserve the current
  trainer inheritance chain.
- `tests/test_temporal_sampling.py`, `tests/test_multicam_relative_pose_trainer.py`,
  `tests/test_mixed_same_heldout_trainer.py`, `tests/test_dynamic_powerfoam_metal.py`,
  and `tests/test_powerfoam_direct.py` still use concrete classes/constants for
  focused structural coverage.
- `research_experiments/dynamic_foam/diagnose_powerfoam_sections.py` and
  `diagnose_powerfoam_heldout_error.py` still import `MetalPowerFoamVideo`.
  That is not a config/registry helper lookup; extracting it would require a
  separate model-class boundary.
