# Dynamic Gauge Config Boundary

## Goal

Continue trainer modularization by moving Dynamic Gauge Foam defaults and config
normalization out of the trainer loop file, matching the config-module pattern
already used by Direct PowerFoam, PowerFoam Metal, and Dynamic PowerFoam Metal.

## Change

- Added `src/train/dynamic_gauge_config.py`.
- Moved `DATA_DEFAULTS`, `MODEL_DEFAULTS`, `RENDER_DEFAULTS`,
  `TRAIN_DEFAULTS`, `LOSS_DEFAULTS`, `LOGGING_DEFAULTS`, and
  `resolve_config(...)` from `train_dynamic_gauge_foam.py` into the new config
  module.
- Updated `train_dynamic_gauge_foam.py` to import `resolve_config(...)` from
  `dynamic_gauge_config.py`.
- Kept optimizer groups, losses, artifact/media payloads, checkpointing, and
  the run loop in the trainer.
- Updated `CODE_ORGANIZATION.md` and `TODO/trainer_landscape_unification.md`.

## Why This Boundary

The Gauge trainer was still carrying defaults and validation inline after the
PowerFoam-family trainers had been split into dedicated config modules. This
made `train_dynamic_gauge_foam.py` act as both launcher/trainer and config
schema owner. The new module keeps config ownership explicit without changing
the public trainer entrypoint or registry route.

## Validation Plan

- Compile the Dynamic Gauge trainer, config module, rendering helper, and
  focused test.
- Run focused Dynamic Gauge tests.
- Smoke `dynamic_gauge_config.resolve_config(...)` directly.
- Smoke `trainer_registry.resolve_config_for_arch(...)` for a checked-in Gauge
  config to ensure the trainer re-export path still works.
- Run whitespace and diff checks on touched files.

## Validation Results

- `rtk .venv/bin/python -m py_compile src/train/train_dynamic_gauge_foam.py src/train/dynamic_gauge_config.py src/train/dynamic_gauge_rendering.py tests/test_dynamic_gauge_foam.py` passed.
- `PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_dynamic_gauge_foam.py -q` passed: `1 passed`.
- Direct config smoke passed for
  `dynamic_gauge_config.resolve_config(load_config_file("src/train_configs/local_mac_dynamic_gauge_foam_video_1024_smoke.jsonc"))`.
- Registry config smoke passed for
  `trainer_registry.resolve_config_for_arch(...)` on the same checked-in Gauge
  config.
- `rtk rg -n "DATA_DEFAULTS|MODEL_DEFAULTS|RENDER_DEFAULTS|TRAIN_DEFAULTS|LOSS_DEFAULTS|LOGGING_DEFAULTS|def resolve_config|from dynamic_gauge_config import resolve_config|apply_defaults|resolved_config" ...` shows defaults and normalization only in `dynamic_gauge_config.py`, with the trainer importing `resolve_config(...)`.
- Touched-file trailing-whitespace scan passed.
- `rtk git diff --check -- src/train/train_dynamic_gauge_foam.py CODE_ORGANIZATION.md TODO/trainer_landscape_unification.md` passed for tracked touched files.
