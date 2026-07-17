# PowerFoam Objective Boundary

## Context

`train_powerfoam_metal.py` still owned several pure objective/composition
helpers that were not trainer-loop logic:

- Metal SSIM loss wrapping
- Metal loss-weight scheduling
- alpha contribution and normal-distance losses
- depth/ray to normal-map targets
- normal-map loss
- fixed/random background tensors
- alpha/background compositing

Those helpers were imported by tests and Dynamic Foam diagnostics through the
full Metal trainer. The goal of this slice was to move the objective math into a
light helper module while keeping the trainer's old public import surface.

## Change

- Added `src/train/powerfoam_objectives.py`.
- Moved trainer-independent objective helpers into that module.
- `train_powerfoam_metal.py` now imports/re-exports those helpers for
  compatibility.
- `diagnose_powerfoam_color_affine.py` now imports
  `composite_powerfoam_background(...)` from `powerfoam_objectives.py`.
- `diagnose_powerfoam_heldout_error.py` now imports
  `composite_fixed_background(...)` and `normals_from_ray_depth(...)` from
  `powerfoam_objectives.py`.

This does not collapse Direct and Metal loss contracts. PowerFoam Direct keeps
its own `scheduled_loss_weights(...)`; `powerfoam_objectives.scheduled_loss_weights`
is the Metal objective schedule because that schedule has Metal-specific keys
and start-step behavior.

## Validation

```bash
PYTHONPATH=src/train:. uv run python -m py_compile \
  src/train/powerfoam_objectives.py \
  src/train/train_powerfoam_metal.py \
  research_experiments/dynamic_foam/diagnose_powerfoam_color_affine.py \
  research_experiments/dynamic_foam/diagnose_powerfoam_heldout_error.py
```

Passed. `uv` printed the known parent `pyproject.toml` warning.

```bash
PYTHONPATH=src/train:. uv run python - <<'PY'
from powerfoam_objectives import (
    composite_fixed_background as helper_composite_fixed,
    composite_powerfoam_background as helper_composite,
    normals_from_ray_depth as helper_normals,
    powerfoam_ssim_loss as helper_ssim,
    scheduled_loss_weights as helper_schedule,
)
from train_powerfoam_metal import (
    composite_fixed_background as trainer_composite_fixed,
    composite_powerfoam_background as trainer_composite,
    normals_from_ray_depth as trainer_normals,
    powerfoam_ssim_loss as trainer_ssim,
    scheduled_loss_weights as trainer_schedule,
)
assert trainer_composite_fixed is helper_composite_fixed
assert trainer_composite is helper_composite
assert trainer_normals is helper_normals
assert trainer_ssim is helper_ssim
assert trainer_schedule is helper_schedule
print("powerfoam_objectives_exports_ok")
PY
```

Passed with `powerfoam_objectives_exports_ok`.

```bash
PYTHONPATH=src/train:. uv run --with pytest python -m pytest \
  tests/test_powerfoam_direct.py::test_powerfoam_metal_ssim_loss_is_zero_for_identical_images \
  tests/test_powerfoam_direct.py::test_powerfoam_metal_background_compositing_uses_alpha \
  tests/test_powerfoam_direct.py::test_powerfoam_metal_contribution_loss_uses_differentiable_alpha_sum \
  tests/test_powerfoam_direct.py::test_powerfoam_normals_from_ray_depth_orients_against_rays \
  tests/test_powerfoam_direct.py::test_powerfoam_normal_map_loss_masks_invalid_pixels \
  tests/test_powerfoam_direct.py::test_powerfoam_metal_normal_distance_loss_backprops_through_tiled_primitive \
  -q
```

Passed: `6 passed in 3.46s`.

## State

The active modularization goal remains open. This slice removes another
trainer-as-helper edge and leaves the trainer focused more tightly on model
construction, rendering, evaluation, checkpointing, and the train loop.
