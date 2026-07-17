# W&B Run Payload Boundary

## Context

Continued the trainer modularization goal by removing the remaining direct
`wandb_run.log(...)` calls in `src/train`. The earlier cleanup had centralized
global `wandb.log(...)`, W&B initialization, finish guards, and default W&B
environment setup, but PowerFoam-family trainers still submitted explicit run
payloads directly.

## Changes

- Added `train_logging.log_wandb_run_payload(run, payload, step=None)`.
- Routed explicit run-object logging through that helper in:
  - `train_dynamic_gauge_foam.py`
  - `train_dynamic_powerfoam_metal.py`
  - `train_powerfoam_direct.py`
  - `train_powerfoam_metal.py`
  - `powerfoam_eval_artifacts.py`
- Kept payload construction and cadence local. Those schemas differ by trainer
  family and should not be flattened into one generic metric helper.
- Added tests for explicit-run forwarding, payload copying, step forwarding,
  and disabled-run no-op behavior.
- Updated `CODE_ORGANIZATION.md` and
  `TODO/trainer_landscape_unification.md`.

## Validation

Commands run:

```bash
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/train_logging.py \
  src/train/train_dynamic_gauge_foam.py \
  src/train/train_dynamic_powerfoam_metal.py \
  src/train/train_powerfoam_direct.py \
  src/train/train_powerfoam_metal.py \
  src/train/powerfoam_eval_artifacts.py

PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_train_logging.py tests/test_train_cli.py -q

PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_dynamic_gauge_foam.py \
  tests/test_dynamic_powerfoam_metal.py::test_dynamic_powerfoam_geometry_summary_verifier_contract \
  tests/test_powerfoam_direct.py::test_atomic_torch_save_preserves_existing_checkpoint_on_failure \
  -q

rg -n "wandb_run\\.log\\(" src/train
```

Results:

- `py_compile` passed for all touched Python files.
- `tests/test_train_logging.py tests/test_train_cli.py`: `20 passed`.
- Focused PowerFoam-family checks: `3 passed`.
- `rg -n "wandb_run\\.log\\(" src/train` returned no matches.

## Remaining

This removes another live duplicated boundary, but the broader modularization
goal remains active. Next useful slices are still live-file driven: longer
W&B-enabled runtime evidence, canonical STAR/dynamic alpha-background
ablations, and careful compatibility alias deletion only after import/config
scans prove no active use.
