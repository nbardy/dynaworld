# PowerFoam Point-Cloud Boundary

## Context

`train_powerfoam_metal.py` still owned a large point-cloud helper surface:
PLY/COLMAP parsing, color normalization, fit/clamp to the PowerFoam box,
train-view visibility filtering, duplicate backfill, and the
`PointCloudInitialization` dataclass. That code was used by the trainer, tests,
and Dynamic Foam diagnostics, making the full Metal trainer a helper module for
point-cloud preparation.

## Change

- Added `src/train/powerfoam_point_cloud.py`.
- Moved point-cloud initialization ownership into that module:
  - `PointCloudInitialization`
  - `resolve_point_cloud_path(...)`
  - PLY and COLMAP point loaders
  - `load_point_cloud_xyz_rgb(...)`
  - fit/clamp/model-box helpers
  - train-visibility filtering
  - `load_powerfoam_point_cloud_initialization(...)`
- `train_powerfoam_metal.py` imports and re-exports the public point-cloud
  helpers for compatibility.
- `diagnose_powerfoam_heldout_error.py` and
  `prepare_ex4dgs_anchor_point_cloud.py` now import point-cloud helpers from
  `powerfoam_point_cloud.py` instead of the full Metal trainer.

This keeps point-cloud preparation explicit and reusable without cloning trainer
logic or changing the Metal training path.

## Validation

```bash
PYTHONPATH=src/train:. uv run python -m py_compile \
  src/train/powerfoam_point_cloud.py \
  src/train/train_powerfoam_metal.py \
  research_experiments/dynamic_foam/diagnose_powerfoam_heldout_error.py \
  research_experiments/dynamic_foam/prepare_ex4dgs_anchor_point_cloud.py
```

Passed. `uv` printed the known parent `pyproject.toml` warning.

```bash
PYTHONPATH=src/train:. uv run python - <<'PY'
from powerfoam_point_cloud import load_point_cloud_xyz_rgb as helper_load_xyz, load_powerfoam_point_cloud_initialization as helper_load_init
from train_powerfoam_metal import load_point_cloud_xyz_rgb as trainer_load_xyz, load_powerfoam_point_cloud_initialization as trainer_load_init
assert trainer_load_xyz is helper_load_xyz
assert trainer_load_init is helper_load_init
print("powerfoam_point_cloud_exports_ok")
PY
```

Passed with `powerfoam_point_cloud_exports_ok`.

```bash
PYTHONPATH=src/train:. uv run --with pytest python -m pytest \
  tests/test_powerfoam_direct.py::test_powerfoam_metal_point_cloud_init_loads_ply_static_geometry \
  tests/test_powerfoam_direct.py::test_powerfoam_point_cloud_init_applies_world_to_model_transform \
  tests/test_powerfoam_direct.py::test_powerfoam_point_cloud_init_can_keep_ply_order_for_ranked_points \
  tests/test_powerfoam_direct.py::test_powerfoam_point_cloud_init_filters_train_visible_points \
  tests/test_powerfoam_direct.py::test_powerfoam_point_cloud_init_jitters_duplicate_backfill \
  -q
```

Passed: `5 passed in 1.90s`.

## State

The active modularization goal remains open. This removes another heavy
trainer-as-helper edge while preserving old imports. The next useful cleanup is
to keep scanning for remaining reusable helpers in trainer files and reroute
diagnostics/tests only where the helper boundary is actually shared or
trainer-independent.
