# Train Third-Party Bootstrap Cleanup

## Context

The trainer unification pass had already moved device resolution and benchmark
bootstrap logic into shared helpers, but several train-local entrypoints still
rebuilt Dynaworld or third-party roots and mutated `sys.path` directly. The
PowerFoam Metal entrypoints also referenced `sys.path` without importing
`sys`, which made their import-time bootstrap fragile.

## Changes

- Added `src/train/external_paths.py` with shared `PROJECT_ROOT`,
  `THIRD_PARTY_ROOT`, `third_party_path(...)`, `ensure_sys_path(...)`,
  `ensure_third_party_path(...)`, and `ensure_module_path(...)`.
- Routed `train_powerfoam_metal.py`, `train_dynamic_powerfoam_metal.py`, and
  `run_dust3r_video.py` through `ensure_third_party_path(...)` for vendored
  path setup.
- Routed `star_uvt_runtime.py` and `renderers/taichi.py` through the same
  project/third-party path helpers while keeping their runtime import policies
  local.
- Replaced the duplicated v12a compiled-bridge path/origin guard in
  `objective/metal_dssim.py` and `objective/v12a_fused_l1.py` with
  `ensure_module_path(...)`.

## Validation

- `PYTHONPATH=src/train:. uv run python -m py_compile` passed for the helper
  and all routed train/runtime/objective files.
- A small `external_paths` smoke confirmed the project root, third-party path
  resolution, and `require_exists=True` skip behavior.
- Import smoke passed for `star_uvt_runtime`, `renderers.taichi`,
  `objective.metal_dssim`, and `objective.v12a_fused_l1`.
- Import smoke also passed for `train_powerfoam_metal` and
  `train_dynamic_powerfoam_metal`, confirming their vendored Metal package paths
  are inserted before the compiled bridge imports.

The usual parent `pyproject.toml` warning appeared during `uv run`; commands
still exited 0.

## State

This is a plumbing cleanup only. It does not rerun PowerFoam, DUSt3R, STAR UVT,
or Taichi rendering. Heavy third-party imports remain local to their owning
modules, and renderer-specific variant choices are intentionally not hidden
behind the generic helper.
