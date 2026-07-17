# PowerFoam Direct Checkpoint Boundary

## Goal

Continue the trainer modularization pass by removing a remaining Direct
PowerFoam train-loop checkpoint write.

## Change

- Extended `powerfoam_checkpoints.save_powerfoam_checkpoint(...)` so `step` is
  optional.
- With `step=None`, the helper writes the Direct historical minimal payload:
  `model` plus serialized `config`.
- With `step` present, the helper keeps the existing Metal-style payload:
  `model`, serialized `config`, `step`, `metrics`, `best_metric_name`, and
  `best_metric_value`.
- Updated `train_powerfoam_direct.py` to call the shared helper instead of
  importing `checkpoint_utils.atomic_torch_save(...)` and
  `serialize_config_value(...)` directly.
- Added `tests/test_powerfoam_checkpoints.py` to cover both payload shapes.

## Validation

- `rtk .venv/bin/python -m py_compile src/train/powerfoam_checkpoints.py src/train/train_powerfoam_direct.py tests/test_powerfoam_checkpoints.py`
- `PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_powerfoam_checkpoints.py tests/test_powerfoam_training.py -q`
