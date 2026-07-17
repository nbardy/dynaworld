# STAR UVT Atomic Checkpoint Saves

## Goal

Continue the trainer modularization pass by routing STAR UVT checkpoint writes
through the shared atomic checkpoint helper instead of local `torch.save(...)`
blocks.

## What Changed

- `star_uvt_checkpoints.py` now imports and uses
  `checkpoint_utils.atomic_torch_save(...)` for:
  - feature RGB probe checkpoints
  - STAR UVT feature-overfit training checkpoints
- `train_star_uvt_rendered_feature_rgb_probe.py` now uses the same atomic
  checkpoint helper for its rendered-feature RGB probe checkpoint.
- Removed local `path.parent.mkdir(...)` plus direct `torch.save(...)` from
  those STAR checkpoint save paths.

## Why This Boundary

`checkpoint_utils.atomic_torch_save(...)` already owns the safe checkpoint write
contract and is covered by a failure-preservation test. STAR UVT had its own
payload schemas, which should stay local, but parent-directory creation and
temporary-file replace behavior should not fork.

## Validation

- `py_compile` passed for:
  - `src/train/star_uvt_checkpoints.py`
  - `src/train/train_star_uvt_rendered_feature_rgb_probe.py`
  - focused STAR UVT checkpoint tests
- Focused pytest passed:

```text
tests/test_star_uvt_checkpoints.py
tests/test_star_uvt_feature_target_adapter.py::test_training_checkpoint_roundtrips_model_colorizer_optimizer
tests/test_star_uvt_feature_target_adapter.py::test_training_checkpoint_can_skip_colorizer_for_probe_init
tests/test_star_uvt_feature_rgb_probe.py
12 passed
```

- Search over the routed STAR files no longer finds local direct checkpoint
  `torch.save(...)` or parent mkdir blocks in those save helpers.

## Remaining Work

- Dynamic PowerFoam and Dynamic Gauge final checkpoint writes still have local
  `torch.save(...)` calls. They are good future candidates if their payloads
  are simple final checkpoints and the output directory already exists.
- Keep checkpoint payload schemas local to each trainer family. The shared
  boundary is atomic file persistence, not a universal checkpoint format.
