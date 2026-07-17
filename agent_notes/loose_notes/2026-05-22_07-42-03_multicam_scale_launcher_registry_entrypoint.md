# Multicam Scale Launcher Registry Entrypoint

## Context

The scale/pretrain shell launchers had already been cleaned up to use
`trainer_registry.resolve_config_for_arch(...)`, `run_config_dict(...)`, and
`train_artifacts.write_json(...)` inside embedded Python snippets. The multicam
scale launcher still used a concrete trainer script for actual sample/sweep
runs:

```bash
TRAINER="src/train/train_multicam_precomputed_feature_implicit_dynamic.py"
```

That bypassed the shared `src/train/train.py` registry dispatch even though
`arch=multicam_precomputed_feature_implicit_camera` is registered.

## Change

- Changed `src/train_scripts/train_scale_static_dynamic_vjepa_multicam.sh` to
  launch `src/train/train.py`.
- The patched per-record config still resolves to
  `train_multicam_precomputed_feature_implicit_dynamic.run_training` through
  `trainer_registry`.

## Validation

- Confirmed the base config resolves through the registry:
  `arch=multicam_precomputed_feature_implicit_camera`,
  module `train_multicam_precomputed_feature_implicit_dynamic`, runner
  `run_training`, trainer class `MulticamPrecomputedFeatureImplicitTrainer`.
- `bash -n src/train_scripts/train_scale_static_dynamic_vjepa_multicam.sh`
  passed.
- `bash src/train_scripts/train_scale_static_dynamic_vjepa_multicam.sh check`
  passed, exercising patched config generation and arch resolution.
- A direct registry-dispatch smoke stubbed `_entry_runner` and called
  `trainer_registry.run_config(...)`; it routed the multicam config to the
  expected module without starting training.
- `tests/test_trainer_registry.py -q` passed: `9 passed`.
- Targeted `git diff --check` passed.

## Current State

The multicam scale launcher now follows the shared train entrypoint. This
reduces one concrete-trainer launch bypass without changing trainer semantics
or forcing Gauge/WorldFoam external research launchers into `src/train/train.py`.
