# PowerFoam Train Batch Indices Helper

## Context

The live duplicate scan found the same train-loop random index draw in four
PowerFoam-family trainers:

```python
torch.randint(0, ..., (int(cfg["train"]["frames_per_step"]),), device=device)
```

The sampled index upper bound differs by trainer:

- Direct PowerFoam samples rows in flattened train targets.
- PowerFoam Metal samples rows in flattened train targets.
- Dynamic PowerFoam Metal samples within the current staged active camera/frame
  count.
- Dynamic Gauge Foam samples frame ids directly.

The random draw itself is shared; the mapping from indices to targets/rays/model
inputs stays trainer-local.

## Changes

- `src/train/powerfoam_training.py`
  - Added `powerfoam_train_batch_indices(sample_count, cfg, device=...)`.
- `src/train/train_powerfoam_direct.py`
  - Uses the helper for train sample row selection.
- `src/train/train_powerfoam_metal.py`
  - Uses the helper for train sample row selection.
- `src/train/train_dynamic_powerfoam_metal.py`
  - Uses the helper with the staged active-frame count.
- `src/train/train_dynamic_gauge_foam.py`
  - Uses the helper for direct frame-index sampling.
- `tests/test_powerfoam_training.py`
  - Covers frames-per-step shape, dtype, and bounds.

## Validation

- `rtk .venv/bin/python -m py_compile src/train/powerfoam_training.py tests/test_powerfoam_training.py src/train/train_powerfoam_direct.py src/train/train_powerfoam_metal.py src/train/train_dynamic_powerfoam_metal.py src/train/train_dynamic_gauge_foam.py`
- `rtk sh -lc 'PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with pytest python -m pytest tests/test_powerfoam_training.py tests/test_powerfoam_direct.py tests/test_dynamic_gauge_foam.py tests/test_dynamic_powerfoam_metal.py -q'`
  - `79 passed, 1 skipped`
- Duplicate scan confirmed the old explicit `torch.randint(... frames_per_step
  ...)` train-batch draw no longer appears in the updated trainer paths.

## Notes

This keeps train-loop target/ray/stage/colorizer assembly local. It only
centralizes the random index draw policy.
