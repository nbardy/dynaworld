# Dynamic PowerFoam Config Boundary

## Context

The trainer unification pass had already moved the Dynamic PowerFoam colorizer,
PowerFoam geometry, adjacency, device, artifact, and raster config helpers out
of the trainer loops. `train_dynamic_powerfoam_metal.py` still carried a large
defaults plus `resolve_config(...)` block inline. That made the file both the
training loop and the config contract, unlike `train_powerfoam_metal.py`, which
now delegates its config contract to `powerfoam_metal_config.py`.

## Change

- Added `src/train/dynamic_powerfoam_metal_config.py`.
- Moved Dynamic PowerFoam Metal config ownership into that module:
  - section defaults
  - `TOKEN_RBF_FEATURE_MODE`
  - `COLORIZE_DEFAULTS`
  - camera/render/train validation
  - `resolve_config(...)`
- `train_dynamic_powerfoam_metal.py` now imports/re-exports those names for
  compatibility with existing tests/scripts.
- The trainer no longer owns the large config-normalization block inline.

This follows the same light-helper pattern as `powerfoam_metal_config.py`: keep
experiment-specific validation close to the config contract, but keep it out of
the trainer loop file.

## Validation

```bash
PYTHONPATH=src/train:. uv run python -m py_compile \
  src/train/dynamic_powerfoam_metal_config.py \
  src/train/train_dynamic_powerfoam_metal.py \
  tests/test_dynamic_powerfoam_metal.py
```

Passed. `uv` printed the known parent `pyproject.toml` warning.

```bash
PYTHONPATH=src/train:. uv run python - <<'PY'
from dynamic_powerfoam_metal_config import (
    LOSS_DEFAULTS as config_loss_defaults,
    TOKEN_RBF_FEATURE_MODE as config_token_mode,
    resolve_config as config_resolve,
)
from train_dynamic_powerfoam_metal import (
    LOSS_DEFAULTS as trainer_loss_defaults,
    TOKEN_RBF_FEATURE_MODE as trainer_token_mode,
    resolve_config as trainer_resolve,
)

assert trainer_resolve is config_resolve
assert trainer_loss_defaults is config_loss_defaults
assert trainer_token_mode == config_token_mode
cfg = config_resolve({"data": {}, "model": {}, "camera": {}, "render": {}, "train": {}, "losses": {}, "logging": {}})
assert cfg["model"]["dynamic_mode"] == "rbf"
assert cfg["camera"]["mode"] == "fixed_pinhole"
print("dynamic_powerfoam_config_ok", cfg["model"]["dynamic_mode"], cfg["render"]["render_size"])
PY
```

Passed with `dynamic_powerfoam_config_ok rbf 64`.

```bash
PYTHONPATH=src/train:. uv run --with pytest python -m pytest \
  tests/test_dynamic_powerfoam_metal.py::test_resolve_config_requires_orbit_camera_init_to_use_orbit_base_path \
  tests/test_dynamic_powerfoam_metal.py::test_resolve_config_accepts_integrated_drone_camera \
  tests/test_dynamic_powerfoam_metal.py::test_dynamic_powerfoam_colorizer_builder_gates_on_feature_mode \
  -q
```

Passed: `3 passed in 2.01s`.

```bash
PYTHONPATH=src/train:. uv run --with pytest python -m pytest \
  tests/test_dynamic_powerfoam_metal.py -q
```

Passed: `33 passed in 7.37s`.

## State

The active cleanup goal remains open. This removes another large trainer-owned
helper surface without changing public imports. Future cleanup can now reason
about Dynamic PowerFoam config via `dynamic_powerfoam_metal_config.py`, while
training/math code stays in `train_dynamic_powerfoam_metal.py`.
