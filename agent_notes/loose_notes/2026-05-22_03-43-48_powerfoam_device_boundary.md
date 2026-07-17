# PowerFoam Device Boundary Follow-Up

## Context

After the PowerFoam config, geometry, and adjacency helpers moved out of
`train_powerfoam_metal.py`, a few Dynamic Foam diagnostics still imported
`resolve_device` from the full Metal trainer. The trainer itself had already
moved to `train_devices.resolve_torch_device(...)`, so those diagnostics were
still coupled to the wrong module for a pure device-selection policy.

## Changes

- Restored `train_powerfoam_metal.resolve_device(...)` as a compatibility
  wrapper around `train_devices.resolve_torch_device(..., auto_cuda=False)`.
- Routed Dynamic Foam diagnostics that only needed device resolution to import
  and call `resolve_torch_device(..., auto_cuda=False)` directly:
  - `diagnose_powerfoam_color_affine.py`
  - `diagnose_powerfoam_heldout_error.py`
  - `probe_powerfoam_camera_perturbations.py`

## Validation

- `PYTHONPATH=src/train:. uv run python -m py_compile` passed for the touched
  trainer and diagnostics.
- Import smoke passed:
  `train_powerfoam_metal.resolve_device("cpu") ==
  train_devices.resolve_torch_device("cpu", auto_cuda=False)`.
- Search confirmed no Dynamic Foam diagnostic imports `resolve_device` from
  `train_powerfoam_metal.py`.
- Focused PowerFoam LR schedule test passed.

The usual parent `pyproject.toml` warning appeared during `uv run`; commands
still exited 0.

## State

This is a device-boundary cleanup only. It does not change PowerFoam training,
MPS selection behavior, diagnostics math, or benchmark results.
