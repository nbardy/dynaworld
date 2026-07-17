# PowerFoam Metal resolve_device alias deletion

## Context

Earlier cleanup routed Dynamic Foam diagnostics and PowerFoam-family trainers
through `train_devices.resolve_torch_device(...)`, but left
`train_powerfoam_metal.resolve_device(...)` as a compatibility wrapper.

## Evidence

A live import scan found no repo callers of the wrapper. The remaining
`resolve_device(...)` call sites are Gauge-local and STAR runtime wrappers,
not imports from `train_powerfoam_metal.py`.

## Change

- Removed `train_powerfoam_metal.resolve_device(...)`.
- Updated the cleanup docs to record that this compatibility alias is gone.

## Validation

- `py_compile` covered `src/train/train_powerfoam_metal.py`.
- Focused PowerFoam-family pytest gate passed after the deletion.
