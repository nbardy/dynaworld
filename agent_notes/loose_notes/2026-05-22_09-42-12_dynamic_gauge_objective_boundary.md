# Dynamic Gauge Objective Boundary

## Goal

Continue trainer modularization by moving Dynamic Gauge Foam loss assembly out
of the trainer loop while preserving sampling, optimizer stepping, scalar names,
artifact logging, and checkpointing.

## Change

- Added `src/train/dynamic_gauge_objectives.py`.
- Moved the Dynamic Gauge training loss assembly into
  `dynamic_gauge_training_loss(...)`.
- The helper returns the weighted total loss plus named tensor terms:
  `l1`, `mse`, `connection`, `temporal`, `opacity`, `radius`, and `atlas_tv`.
- Updated `src/train/train_dynamic_gauge_foam.py` to call the helper and keep
  its existing logging names.
- Updated `CODE_ORGANIZATION.md` and `TODO/trainer_landscape_unification.md`.

## Why This Boundary

The Gauge trainer had become both the run loop and the objective owner. The
loss block is pure objective math over model output, targets, frame indices,
the KNN edge index, and normalized config. Moving it into an objective module
matches the existing `powerfoam_objectives.py` pattern while keeping
trainer-specific scheduling, W&B names, media payloads, and checkpoints local.

## Validation Plan

- Compile the Dynamic Gauge trainer, config, rendering, objective helper, and
  focused test.
- Run the focused Dynamic Gauge pytest.
- Run a tiny objective smoke that calls `dynamic_gauge_training_loss(...)` and
  backpropagates through the model.
- Search to confirm the trainer no longer imports objective primitives directly.
- Run whitespace and diff checks on touched files.

## Validation Results

- `rtk .venv/bin/python -m py_compile src/train/train_dynamic_gauge_foam.py src/train/dynamic_gauge_config.py src/train/dynamic_gauge_rendering.py src/train/dynamic_gauge_objectives.py tests/test_dynamic_gauge_foam.py` passed.
- `PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_dynamic_gauge_foam.py -q` passed: `1 passed`.
- The tiny objective smoke passed: it built a 2-frame Dynamic Gauge model, rendered, called `dynamic_gauge_training_loss(...)`, checked finite terms, and backpropagated into `model.p0`.
- Symbol scan showed the trainer only imports/calls `dynamic_gauge_training_loss(...)`; objective primitives now live behind `src/train/dynamic_gauge_objectives.py`.
- Touched-file whitespace scan passed.
- `rtk git diff --check -- src/train/train_dynamic_gauge_foam.py CODE_ORGANIZATION.md TODO/trainer_landscape_unification.md` passed.
