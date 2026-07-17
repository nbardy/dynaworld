# PowerFoam Eval Batch-Size Helper

## Context

The live duplicate scan found the same eval-render batch-size policy repeated in
PowerFoam artifact paths:

```python
max(1, int(cfg["train"]["frames_per_step"]))
```

This is distinct from train-time random frame sampling. It controls how many
frames the eval/artifact render loop sends through each render call.

## Changes

- `src/train/powerfoam_eval_render.py`
  - Added `powerfoam_eval_batch_size(cfg)`.
  - It preserves the existing minimum-one-frame clamp.
- `src/train/train_powerfoam_direct.py`
  - Uses the helper for train/eval and heldout artifact renders.
- `src/train/powerfoam_eval_artifacts.py`
  - Uses the helper for shared PowerFoam train/eval and heldout artifact
    renders.
- `src/train/train_dynamic_powerfoam_metal.py`
  - Uses the helper for full eval artifact renders.
- `tests/test_powerfoam_eval_render.py`
  - Covers string coercion and the minimum-one clamp.

## Validation

- `rtk .venv/bin/python -m py_compile src/train/powerfoam_eval_render.py tests/test_powerfoam_eval_render.py src/train/train_powerfoam_direct.py src/train/powerfoam_eval_artifacts.py src/train/train_dynamic_powerfoam_metal.py`
- `rtk sh -lc 'PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with pytest python -m pytest tests/test_powerfoam_eval_render.py tests/test_powerfoam_direct.py tests/test_dynamic_powerfoam_metal.py -q'`
  - `78 passed, 1 skipped`
- Duplicate scan confirmed no remaining artifact/eval render calls with
  open-coded `max(1, int(cfg["train"]["frames_per_step"]))` in the updated
  paths.

## Notes

This only centralizes eval/artifact render batching. It does not change
train-loop frame sampling or optimizer behavior.
