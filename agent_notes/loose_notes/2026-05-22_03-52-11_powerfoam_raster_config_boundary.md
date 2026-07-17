# PowerFoam Raster Config Boundary

## Context

The trainer unification pass had already moved PowerFoam config normalization,
geometry helpers, adjacency helpers, device resolution, constants, and metrics
out of `train_powerfoam_metal.py`. A remaining small helper boundary was still
trainer-local: `make_raster_config(...)`. Dynamic Foam diagnostics imported
the full Metal trainer just to construct a `FoamRasterConfig`, and Dynamic
PowerFoam Metal carried a near-duplicate constructor for its own extension type.

## Change

- Added `src/train/powerfoam_raster_config.py`.
- Centralized shared raster-config kwargs for:
  - PowerFoam Metal, including `use_tiled` and `tiled_builder`.
  - Dynamic PowerFoam Metal, without tiled fields because that extension config
    type does not expose them.
- Kept compatibility aliases:
  - `train_powerfoam_metal.make_raster_config`
  - `train_dynamic_powerfoam_metal.make_raster_config`
- Rerouted Dynamic Foam diagnostics that only needed raster config construction
  to import `make_powerfoam_metal_raster_config(...)` from the helper module.

This is deliberately not a new trainer framework. It is one more light helper
that removes a heavy import edge and keeps the extension-specific constructor
rules in one place.

## Validation

```bash
PYTHONPATH=src/train:. uv run python -m py_compile \
  src/train/powerfoam_raster_config.py \
  src/train/train_powerfoam_metal.py \
  src/train/train_dynamic_powerfoam_metal.py \
  research_experiments/dynamic_foam/diagnose_powerfoam_sections.py \
  research_experiments/dynamic_foam/diagnose_powerfoam_heldout_error.py \
  research_experiments/dynamic_foam/verify_powerfoam_raytrace_start_support.py \
  research_experiments/dynamic_foam/verify_powerfoam_raytrace_real_view_alpha.py
```

Passed. `uv` printed the known parent `pyproject.toml` warning.

```bash
PYTHONPATH=src/train:. uv run python - <<'PY'
from powerfoam_raster_config import (
    make_dynamic_powerfoam_metal_raster_config,
    make_powerfoam_metal_raster_config,
)
from train_dynamic_powerfoam_metal import make_raster_config as dynamic_make, RENDER_DEFAULTS as DYNAMIC_RENDER_DEFAULTS
from train_powerfoam_metal import make_raster_config as metal_make, RENDER_DEFAULTS as METAL_RENDER_DEFAULTS

assert metal_make is make_powerfoam_metal_raster_config
assert dynamic_make is make_dynamic_powerfoam_metal_raster_config
metal_cfg = metal_make(dict(METAL_RENDER_DEFAULTS))
dynamic_cfg = dynamic_make(dict(DYNAMIC_RENDER_DEFAULTS))
print("powerfoam_raster_config_ok", type(metal_cfg).__name__, type(dynamic_cfg).__name__)
PY
```

Passed with `powerfoam_raster_config_ok FoamRasterConfig FoamRasterConfig`.

```bash
PYTHONPATH=src/train:. uv run --with pytest python -m pytest \
  tests/test_powerfoam_direct.py::test_powerfoam_metal_camera_rays_include_camera_pose \
  tests/test_powerfoam_direct.py::test_powerfoam_metal_lr_schedule_uses_official_cosine_shape_and_warmups \
  tests/test_dynamic_powerfoam_metal.py::test_dynamic_powerfoam_colorizer_builder_gates_on_feature_mode \
  -q
```

Passed: `3 passed in 2.55s`.

## State

The active cleanup goal remains open. The useful next work is more live-file
helper routing and deletion of stale compatibility shims only after `rg` proves
callers are gone. The current pass preserved trainer compatibility aliases
because tests and scripts still import them.
