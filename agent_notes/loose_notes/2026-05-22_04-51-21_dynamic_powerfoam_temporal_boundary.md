# Dynamic PowerFoam Temporal Boundary

## What Changed

- Added `src/train/dynamic_powerfoam_temporal.py` as the light module for pure
  Dynamic PowerFoam temporal math:
  - `make_gaussian_time_basis(...)`
  - `fit_temporal_basis(...)`
  - `temporal_accel(...)`
  - `atanh_clamped(...)`
  - `temporal_motion_metrics(...)`
- `src/train/train_dynamic_powerfoam_metal.py` now imports those helpers instead
  of defining them inline. The imported names preserve the old trainer-module
  compatibility surface for existing callers.
- `tests/test_dynamic_powerfoam_metal.py` now imports
  `temporal_motion_metrics(...)` from the helper module directly, so the test no
  longer treats the full trainer as the temporal-metrics namespace.

## Why This Slice

The dynamic PowerFoam trainer still has experiment-specific model classes and
train-loop logic that should stay local. The temporal helpers are different:
they are deterministic tensor transforms used by both dynamic model variants
and tested independently of the rasterizer. Moving them out reduces trainer-file
surface without creating a new base class or changing the runtime contract.

## Validation

```bash
PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with pytest python -m pytest tests/test_dynamic_powerfoam_metal.py -q
```

Result:

```text
33 passed in 8.94s
```

```bash
PYTHONPATH=src/train:third_party/powerfoam-metal .venv/bin/python -m py_compile \
  src/train/dynamic_powerfoam_temporal.py \
  src/train/train_dynamic_powerfoam_metal.py \
  tests/test_dynamic_powerfoam_metal.py
```

Result: passed.

## Next

- A similar camera-helper extraction is plausible, but it is riskier because
  teacher camera prefit writes artifacts and uses the implicit camera decoder.
  Do it only as a focused module with compatibility imports and the existing
  teacher-prefit tests.
- Keep deleting trainer-as-helper imports only when a live `rg` shows a clear
  helper boundary; do not chase stale historical file names from older audits.
