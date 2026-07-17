# PowerFoam Training Primitives

## Context

The helper-routing pass found two exact primitive duplicates across
`train_powerfoam_direct.py` and `train_powerfoam_metal.py`:

- `flatten_multiview_powerfoam_samples(...)`
- `exp_scheduled_weight(...)`

The full `scheduled_loss_weights(...)` functions are not identical and should
stay local: PowerFoam Direct and PowerFoam Metal expose different loss keys and
start-step behavior. The useful boundary is the shared primitive underneath
those trainer-specific contracts.

## Change

- Added `src/train/powerfoam_training.py`.
- Moved multiview frame/ray flattening into that helper.
- Moved exponential scheduled-weight interpolation into that helper.
- Updated PowerFoam Direct and PowerFoam Metal to import these primitives.
- Kept old import compatibility because the trainers still re-export imported
  names at module scope.

This reduces duplicated trainer logic without flattening distinct loss
contracts into a misleading common API.

## Validation

```bash
PYTHONPATH=src/train:. uv run python -m py_compile \
  src/train/powerfoam_training.py \
  src/train/train_powerfoam_direct.py \
  src/train/train_powerfoam_metal.py
```

Passed. `uv` printed the known parent `pyproject.toml` warning.

```bash
PYTHONPATH=src/train:. uv run --with pytest python -m pytest \
  tests/test_powerfoam_direct.py::test_powerfoam_direct_shared_state_accepts_posed_multiview_rays \
  tests/test_powerfoam_direct.py::test_powerfoam_metal_multiview_flatten_shares_frame_indices_across_views \
  tests/test_powerfoam_direct.py::test_powerfoam_regularizer_weights_use_official_exp_decay_shape \
  tests/test_powerfoam_direct.py::test_powerfoam_metal_contribution_loss_uses_differentiable_alpha_sum \
  -q
```

Passed: `4 passed in 2.26s`.

## State

The active modularization goal remains open. This was a narrow cross-trainer
dedupe: shared tensor reshaping and schedule math moved out, while Direct and
Metal train loops, renderers, and loss dictionaries remain explicit.
