# STAR UVT Runtime Helper Cleanup

## Goal

Continue modularizing the training code by extracting generic STAR UVT runtime
helpers out of trainer scripts without changing training behavior.

## Change

- Added `src/train/star_uvt_runtime.py` as the shared owner for:
  - `DYNAWORLD_ROOT`
  - `STAR_UVT_ROOT`
  - `ensure_star_uvt_on_path(...)`
  - `resolve_device(...)`
  - `sync_device(...)`
  - `psnr_from_loss(...)`
- Rewired:
  - `src/train/train_star_uvt_video_overfit.py`
  - `src/train/train_star_uvt_feature_overfit.py`
  - `src/train/train_star_uvt_feature_rgb_probe.py`
  - `src/train/train_star_uvt_rendered_feature_rgb_probe.py`
- Preserved the RGB STAR path shape by calling
  `ensure_star_uvt_on_path(include_dynaworld_root=False)` there, while the
  feature-side scripts keep the previous behavior of inserting both the
  Dynaworld root and the STAR UVT checkout.
- Added `tests/test_star_uvt_runtime.py` to pin path insertion order, CPU device
  resolution, and PSNR conversion.

## Validation

```bash
rtk env PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/star_uvt_runtime.py \
  src/train/train_star_uvt_video_overfit.py \
  src/train/train_star_uvt_feature_overfit.py \
  src/train/train_star_uvt_feature_rgb_probe.py \
  src/train/train_star_uvt_rendered_feature_rgb_probe.py
```

Passed.

```bash
rtk env PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_runtime.py \
  tests/test_star_uvt_feature_rgb_probe.py \
  tests/test_star_uvt_feature_target_adapter.py::test_rgb_probe_config_requires_target_grid_materialization \
  tests/test_config_factory_helpers.py::test_train_router_accepts_star_uvt_video_overfit_config \
  -q
```

Result: `13 passed in 1.48s`.

```bash
rtk env PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_feature_target_adapter.py -q
```

Result: `35 passed in 1.70s`.

```bash
rtk git diff --check
```

Passed.

## Remaining

The STAR UVT probe scripts still import feature/model helpers from
`train_star_uvt_feature_overfit.py`. That is a larger split than this runtime
cleanup. A future aligned slice would move reusable feature-target loading,
training-sequence loading, and sparse-grid helpers into one or more non-trainer
modules.
