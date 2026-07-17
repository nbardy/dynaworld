# Train Logging Cadence Unification

Date: 2026-05-21 16:18

## Goal Context

The active goal is to modularize the training code by establishing simple shared
patterns and interfaces across train runs, with reuse and organization favored
over another large trainer framework.

The existing organization docs already point in the right direction:

- `CODE_ORGANIZATION.md` says to prefer small shared helpers and explicit
  contracts over a giant abstract trainer.
- `TODO/trainer_landscape_unification.md` identifies repeated log-cadence checks
  as a P2 cleanup after the larger render/objective helpers.

## Slice Implemented

Added shared log-cadence helpers to `src/train/train_logging.py`:

- `should_log_step(...)`
- `should_log_from_config(...)`
- `should_log_scalar(...)`
- `should_log_image(...)`
- `should_log_video(...)`

Also centralized the identical W&B run initialization block as
`init_wandb_run(cfg)`. This deliberately covers only the shared
`logging.wandb_*` config contract; trainer-specific payload names, media paths,
checkpoint metadata, and artifact choices stay local to each trainer.

Updated these trainer families to use the shared helper instead of local
`step % every == 0 or last_step` logic:

- `src/train/train_video_token_implicit_dynamic.py`
- `src/train/train_powerfoam_direct.py`
- `src/train/train_powerfoam_metal.py`
- `src/train/train_dynamic_powerfoam_metal.py`
- `src/train/train_dynamic_gauge_foam.py`

Updated these scripts to use the shared W&B initializer instead of duplicate
local `_wandb_run`/`init_wandb_run` functions:

- `src/train/train_powerfoam_direct.py`
- `src/train/train_powerfoam_metal.py`
- `src/train/train_dynamic_powerfoam_metal.py`
- `src/train/train_dynamic_gauge_foam.py`
- `src/train/train_star_uvt_video_overfit.py`
- `src/train/train_star_uvt_feature_overfit.py`
- `src/train/train_star_uvt_feature_rgb_probe.py`
- `src/train/train_star_uvt_rendered_feature_rgb_probe.py`

Preserved token-GS `logging.log_initial_media` behavior by passing
`log_step_zero=false` for image/video gates when initial media is disabled.

## Why This Slice

This is small but cross-cutting. It creates one shared policy used by token-GS,
PowerFoam, Dynamic PowerFoam, and Gauge Foam without forcing their train loops
or optimizer semantics into a fake common class. It matches the project rule:
reduce real behavior forks while keeping different trainer families readable.

## Validation

Passed:

```bash
PYTHONPATH=src/train:. rtk .venv/bin/python -m py_compile \
  src/train/train_logging.py \
  src/train/train_video_token_implicit_dynamic.py \
  src/train/train_powerfoam_direct.py \
  src/train/train_powerfoam_metal.py \
  src/train/train_dynamic_powerfoam_metal.py \
  src/train/train_dynamic_gauge_foam.py \
  src/train/train_star_uvt_feature_overfit.py \
  src/train/train_star_uvt_feature_rgb_probe.py \
  src/train/train_star_uvt_video_overfit.py \
  src/train/train_star_uvt_rendered_feature_rgb_probe.py \
  tests/test_train_logging.py

PYTHONPATH=src/train:. rtk uv run --with pytest python -m pytest \
  tests/test_train_logging.py \
  tests/test_pipeline_helpers.py \
  tests/test_config_factory_helpers.py -q
```

Result after the W&B helper tests: `26 passed in 1.59s`.

## Next Refactor Sequence

1. Move remaining metric payload duplication into `pipeline.diagnostics` or a
   small sibling helper, then migrate procedural baselines where the math is
   identical.
2. Split generic W&B media construction from trainer-specific media naming only
   where more than one trainer already needs the exact same behavior.
3. Keep PowerFoam/WorldFoam train-loop semantics separate; share utilities, not
   a base trainer.
4. After this goal's small slices accumulate, update `CODE_ORGANIZATION.md` with
   the current completed and remaining boundaries.
