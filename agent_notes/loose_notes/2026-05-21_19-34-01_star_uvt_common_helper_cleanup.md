# STAR UVT Common Helper Cleanup

## Goal

Continue modularizing the STAR UVT training/probe code by extracting reusable
non-policy helpers out of `train_star_uvt_feature_overfit.py`.

## Change

Added `src/train/star_uvt_common.py` as the shared owner for:

- `load_training_sequence(...)`
- `load_colorizer_init_checkpoint(...)`
- `grad_norms(...)`
- `target_grid_slice_for_render_chunk(...)`

Rewired:

- `src/train/train_star_uvt_feature_overfit.py`
- `src/train/train_star_uvt_feature_rgb_probe.py`
- `src/train/train_star_uvt_rendered_feature_rgb_probe.py`

The feature-overfit trainer still imports these helpers with the existing
underscore aliases, so current tests and internal call sites keep working. The
probe scripts no longer import sequence loading, colorizer checkpoint loading,
gradient norms, or target-grid chunk slicing from a trainer module.

## Validation

```bash
rtk env PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/star_uvt_common.py \
  src/train/train_star_uvt_feature_overfit.py \
  src/train/train_star_uvt_feature_rgb_probe.py \
  src/train/train_star_uvt_rendered_feature_rgb_probe.py \
  tests/test_star_uvt_common.py
```

Passed.

```bash
rtk env PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_common.py \
  tests/test_star_uvt_feature_rgb_probe.py \
  tests/test_star_uvt_feature_target_adapter.py \
  -q
```

Result: `46 passed in 1.78s`.

```bash
rtk git diff --check
```

Passed.

## Remaining

The STAR UVT probe scripts still import two feature-specific helpers from
`train_star_uvt_feature_overfit.py`:

- `_load_cached_feature_target`
- `_sparse_target_grid_pixel_ids`

The rendered probe also imports `_render_rgb_chunks` for media rendering. Those
depend on a larger feature-target/sparse-grid/render-helper cluster and should
be moved as a deliberate next slice, not mixed into this stable common-helper
cleanup.
