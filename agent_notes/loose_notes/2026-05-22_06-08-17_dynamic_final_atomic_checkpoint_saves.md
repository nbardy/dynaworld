# Dynamic Final Atomic Checkpoint Saves

## Goal

Continue the trainer modularization pass by moving the remaining simple dynamic
trainer final checkpoint writes onto the shared atomic checkpoint persistence
helper.

## What Changed

- `train_dynamic_powerfoam_metal.py` now imports
  `checkpoint_utils.atomic_torch_save(...)` and uses it for
  `checkpoint_final.pt`.
- `train_dynamic_gauge_foam.py` now imports the same helper and uses it for
  `checkpoint_final.pt`.
- The checkpoint payload dictionaries are unchanged and remain local to each
  trainer. Only the file persistence primitive changed.

## Why This Boundary

The repo already had `checkpoint_utils.atomic_torch_save(...)`, and PowerFoam
Direct, PowerFoam Metal, and STAR UVT checkpoint paths now use it. Dynamic
PowerFoam Metal and Dynamic Gauge Foam had the same simple final-checkpoint
file write shape, so keeping them on direct `torch.save(...)` was unnecessary
duplication.

## Validation

- `py_compile` passed for:
  - `src/train/train_dynamic_powerfoam_metal.py`
  - `src/train/train_dynamic_gauge_foam.py`
- Focused pytest passed:

```text
tests/test_dynamic_gauge_foam.py
tests/test_dynamic_powerfoam_metal.py::test_dynamic_powerfoam_geometry_summary_verifier_contract
tests/test_powerfoam_direct.py::test_atomic_torch_save_preserves_existing_checkpoint_on_failure
3 passed
```

- Search confirms those two trainer files no longer call direct `torch.save(...)`
  for final checkpoints.

## Remaining Work

- Do not force every checkpoint-like binary or cache path into this helper.
  Video feature caches and sequence-data caches have their own cache semantics.
- If more final trainer checkpoints appear, use `atomic_torch_save(...)` unless
  there is a concrete reason the write must be non-atomic.
