# PowerFoam loss schedule helper reuse

## Context

The trainer-cleanup pass found one live duplicate left in the PowerFoam
family: `train_powerfoam_direct.py` still carried its own
`scheduled_loss_weights(...)`, while `powerfoam_objectives.py` already owned
the Metal-style auxiliary schedule.

Direct could not be routed blindly because its loss dictionary includes
`rgb_mse_sum_weight`, while Metal includes normal-map and auxiliary
`*_weight_start_step` gates.

## Change

- `powerfoam_objectives.scheduled_loss_weights(...)` now preserves an optional
  `rgb_mse_sum_weight` key when a caller's loss config has it.
- `powerfoam_direct_config.LOSS_DEFAULTS` now carries the explicit auxiliary
  start-step keys and zero normal-map defaults needed by the shared schedule.
- `train_powerfoam_direct.py` imports `scheduled_loss_weights` from
  `powerfoam_objectives` and no longer owns a local schedule wrapper.
- The Direct schedule test now asserts the Direct-only RGB-MSE-sum key survives
  the shared schedule and that normal-map defaults remain zero.

## Validation

- `rtk .venv/bin/python -m py_compile src/train/powerfoam_objectives.py src/train/powerfoam_direct_config.py src/train/train_powerfoam_direct.py tests/test_powerfoam_direct.py`
- `rtk sh -lc 'PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with pytest python -m pytest tests/test_powerfoam_direct.py -q'`
  - `44 passed, 1 skipped`
- `rtk sh -lc 'PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with pytest python -m pytest tests/test_powerfoam_training.py tests/test_powerfoam_direct.py tests/test_dynamic_gauge_foam.py tests/test_dynamic_powerfoam_metal.py -q'`
  - `79 passed, 1 skipped`

## Follow-up

Keep loss computation local. Direct's RGB-MSE-sum term and Metal's normal-map
path are real trainer-family differences; only the schedule shape is shared.
