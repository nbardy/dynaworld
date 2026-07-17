# PowerFoam Constant And Metric Routes

## Context

Several Dynamic Foam diagnostics still imported pure helpers through
`train_powerfoam_metal.py`:

- `POWERFOAM_SOFTPLUS_BETA`, which is defined in `powerfoam_direct.py`
- `reconstruction_eval_metrics(...)`, which is defined in
  `pipeline.diagnostics`

The full Metal trainer still re-exports those names, but diagnostics do not
need to depend on the trainer for constants or metric helpers.

## Changes

- Routed raytrace/topology diagnostics to import `POWERFOAM_SOFTPLUS_BETA` from
  `powerfoam_direct.py`.
- Routed color-affine and camera-perturbation diagnostics to import
  `reconstruction_eval_metrics(...)` from `pipeline.diagnostics`.
- Kept `train_powerfoam_metal.py` compatibility re-exports untouched.

## Validation

- `PYTHONPATH=src/train:. uv run python -m py_compile` passed for the touched
  diagnostics.
- Search found no Dynamic Foam diagnostic importing `POWERFOAM_SOFTPLUS_BETA`
  or `reconstruction_eval_metrics(...)` from `train_powerfoam_metal.py`.
- Import smoke passed: trainer re-export values still match the owning modules.
- Focused PowerFoam tests passed:
  `test_powerfoam_metal_camera_rays_include_camera_pose` and
  `test_powerfoam_metal_lr_schedule_uses_official_cosine_shape_and_warmups`.

The usual parent `pyproject.toml` warning appeared during `uv run`; commands
still exited 0.

## State

This is a pure import-routing cleanup. It does not change diagnostic math,
PowerFoam trainer behavior, kernels, or benchmark results.
