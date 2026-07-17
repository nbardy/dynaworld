# Train Metric Helper Unification

Date: 2026-05-21 16:29

## Goal Context

The active goal is still broad trainer modularization: establish simple shared
patterns and interfaces across train runs, organize reusable code in clear
submodules, and clean duplicate trainer logic without changing behavior.

The prior slice put shared log cadence and identical W&B run init into
`src/train/train_logging.py`. The next documented duplication target in
`CODE_ORGANIZATION.md` and `TODO/trainer_landscape_unification.md` was eval
metric payloads.

## Slice Implemented

Added shared reconstruction metric helpers to
`src/train/pipeline/diagnostics.py`:

- `reconstruction_l1_mse_metrics(prediction, target, prefix=...)`
- `reconstruction_eval_metrics(prediction, target, cfg, prefix=...)`

The helpers keep the existing snake_case result JSON key contract:
`eval_l1`, `eval_mse`, `heldout_eval_l1`, `heldout_eval_mse`,
`eval_psnr`, `eval_ssim`, etc. They do not emit W&B display keys; W&B payload
assembly remains local to each trainer.

Updated these eval paths:

- `src/train/train_powerfoam_direct.py`
- `src/train/train_powerfoam_metal.py`
- `src/train/train_dynamic_powerfoam_metal.py`
- `src/train/train_dynamic_gauge_foam.py`

`train_powerfoam_metal.py` no longer owns a local `reconstruction_eval_metrics`
copy; it imports the shared helper. Direct PowerFoam, Dynamic PowerFoam, and
Dynamic Gauge Foam use the shared L1/MSE helper where the old keys and values
match exactly.

## Validation

Passed:

```bash
PYTHONPATH=src/train:. rtk .venv/bin/python -m py_compile \
  src/train/pipeline/diagnostics.py \
  src/train/train_powerfoam_direct.py \
  src/train/train_powerfoam_metal.py \
  src/train/train_dynamic_powerfoam_metal.py \
  src/train/train_dynamic_gauge_foam.py \
  tests/test_pipeline_diagnostics.py

PYTHONPATH=src/train:. rtk uv run --with pytest python -m pytest \
  tests/test_pipeline_diagnostics.py \
  tests/test_train_logging.py \
  tests/test_pipeline_helpers.py \
  tests/test_config_factory_helpers.py -q
```

Result: `28 passed in 1.26s`.

Also passed `git diff --check` and a trailing-whitespace scan on touched files.

## Next Refactor Sequence

After the metric slice, I verified that two older plan items are already true in
the current tree:

- `objective.compose_rgb` plus `RGBReconObjective.render_view` own RGB
  composition, and both single-cam token-GS and multicam feature trainers call
  that objective boundary.
- `pipeline.validation_media` owns single-cam and multicam validation media
  payload helpers, including alpha and feature-PCA diagnostic grids.

Verification:

```bash
PYTHONPATH=src/train:. rtk uv run --with pytest python -m pytest \
  tests/test_objective_background_and_composition.py \
  tests/test_rgb_recon_objective.py \
  tests/test_pipeline_helpers.py -q
```

Result: `16 passed in 1.21s`.

Updated `CODE_ORGANIZATION.md` and `TODO/trainer_landscape_unification.md` so
future agents do not re-implement these already-present interfaces.

Current next refactor sequence:

1. Render-dispatch convergence: older wrappers should return or adapt to the
   alpha-aware shape.
2. Mixed same-view plus heldout-view scheduler bridge, keeping the two losses
   and data semantics separate.
3. Entrypoint cleanup only after active configs are checked.
4. Keep trainer loops separate; share helpers and typed payloads, not a base
   trainer framework.
