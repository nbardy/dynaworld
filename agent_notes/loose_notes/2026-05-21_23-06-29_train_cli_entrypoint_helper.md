# Train CLI Entrypoint Helper

## Context

The active cleanup goal is to make trainer boundaries reusable without turning
the codebase into a base-class maze. After the registry and optimizer slices,
the remaining obvious duplicate was the script `main(...)` boundary: many
trainers repeated `len(sys.argv) != 2`, a usage string, and
`load_config_file(sys.argv[1])`; the Token-GS family also repeated
`main(config_or_path)` path-vs-dict dispatch.

## Change

- Added `src/train/train_cli.py`.
- Added `run_config_arg(runner, usage=..., argv=None)` for one-config CLI
  entrypoints.
- Added `run_config_or_path(config, runner)` for public `main(config_or_path)`
  functions that are called directly by tests/scripts.
- Updated active `src/train/train_*.py` modules across Token-GS, precomputed
  feature, multicam, mixed same-heldout, PowerFoam, and STAR UVT train/probe
  paths to use the helper.
- Kept `src/train/train.py` custom because it dispatches from the config path
  through `trainer_registry.run_config(...)`, not by passing a loaded config to
  one fixed runner.
- Added `tests/test_train_cli.py` for dict passthrough, path loading, argv
  loading, and usage errors.

## Validation

```bash
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/train_cli.py \
  src/train/train_video_token_implicit_dynamic.py \
  src/train/train_precomputed_feature_implicit_dynamic.py \
  src/train/train_multicam_precomputed_feature_implicit_dynamic.py \
  src/train/train_multicam_relative_pose_implicit_dynamic.py \
  src/train/train_mixed_same_heldout_implicit_dynamic.py \
  src/train/train_powerfoam_direct.py \
  src/train/train_dynamic_gauge_foam.py \
  src/train/train_dynamic_powerfoam_metal.py \
  src/train/train_powerfoam_metal.py \
  src/train/train_star_uvt_video_overfit.py \
  src/train/train_star_uvt_feature_overfit.py \
  src/train/train_star_uvt_feature_rgb_probe.py \
  src/train/train_star_uvt_rendered_feature_rgb_probe.py \
  tests/test_train_cli.py
```

Passed.

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_train_cli.py \
  tests/test_trainer_registry.py \
  tests/test_config_factory_helpers.py::test_train_router_accepts_star_uvt_video_overfit_config \
  -q
```

Result: `8 passed in 3.61s`.

`git diff --check` passed for the touched train CLI/helper files.

## Remaining

This is a narrow entrypoint cleanup. It does not prove convergence, renderer
quality, or broader trainer unification. The next cleanup choice should be
either curating the many untracked configs/scripts into canonical versus scratch
sets, or removing stale historical rows from the landscape doc that reference
files no longer present under current `src/train/`.
