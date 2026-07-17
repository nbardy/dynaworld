# PowerFoam Eval Color Boundary

## Context

`train_powerfoam_metal.py` still owned pure eval color-calibration helpers:
pixel flattening, bias-column construction, channel-affine fit/apply,
RGB-matrix affine fit/apply, calibration fit/apply, frame-index summaries, and
calibration provenance serialization. The color-affine diagnostic carried a
second copy of the same affine math.

## Change

- Added `src/train/powerfoam_eval_color.py`.
- Moved eval color-calibration helpers into that module.
- `train_powerfoam_metal.py` imports/re-exports the public calibration fit,
  apply, and serialize helpers for compatibility.
- `tests/test_powerfoam_eval_color_calibration.py` now imports the helper module
  directly.
- `diagnose_powerfoam_color_affine.py` now reuses
  `fit_channel_affine(...)`, `apply_channel_affine(...)`,
  `fit_rgb_matrix_affine(...)`, and `apply_rgb_matrix_affine(...)` from the
  shared helper instead of carrying local duplicates.

This keeps eval calibration reusable and lets the trainer focus on when to fit
and log calibration, not how affine color correction is computed.

## Validation

```bash
PYTHONPATH=src/train:. uv run python -m py_compile \
  src/train/powerfoam_eval_color.py \
  src/train/train_powerfoam_metal.py \
  research_experiments/dynamic_foam/diagnose_powerfoam_color_affine.py \
  tests/test_powerfoam_eval_color_calibration.py
```

Passed. `uv` printed the known parent `pyproject.toml` warning.

```bash
PYTHONPATH=src/train:. uv run python - <<'PY'
from powerfoam_eval_color import (
    apply_eval_color_calibration as helper_apply,
    fit_eval_color_calibration as helper_fit,
    serialize_eval_color_calibration as helper_serialize,
)
from train_powerfoam_metal import (
    apply_eval_color_calibration as trainer_apply,
    fit_eval_color_calibration as trainer_fit,
    serialize_eval_color_calibration as trainer_serialize,
)
assert trainer_apply is helper_apply
assert trainer_fit is helper_fit
assert trainer_serialize is helper_serialize
print("powerfoam_eval_color_exports_ok")
PY
```

Passed with `powerfoam_eval_color_exports_ok`.

```bash
PYTHONPATH=src/train:. uv run --with pytest python -m pytest \
  tests/test_powerfoam_eval_color_calibration.py -q
```

Passed: `3 passed in 1.05s`.

The first diagnostic import smoke used only `PYTHONPATH=src/train:.` and failed
because `diagnose_powerfoam_color_affine.py` imports its sibling
`diagnose_powerfoam_heldout_error.py` by bare module name. Rerunning with the
diagnostic directory on `PYTHONPATH` matched the script's import shape:

```bash
PYTHONPATH=src/train:research_experiments/dynamic_foam:. uv run python - <<'PY'
import importlib
mod = importlib.import_module('diagnose_powerfoam_color_affine')
assert hasattr(mod, 'fit_channel_affine')
assert hasattr(mod, 'fit_rgb_matrix_affine')
print('diagnose_powerfoam_color_affine_import_ok')
PY
```

Passed with `diagnose_powerfoam_color_affine_import_ok`.

## State

The active modularization goal remains open. This slice removes another pure
helper block from PowerFoam Metal and also eliminates duplicated affine
calibration math in the diagnostic script.
