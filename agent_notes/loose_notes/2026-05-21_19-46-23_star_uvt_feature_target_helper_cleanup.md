# STAR UVT Feature Target Helper Cleanup

Date: 2026-05-21 19:46:23 Asia/Ho_Chi_Minh

## Goal

Continue the trainer modularization pass by finishing the STAR UVT
feature-target helper extraction without changing training behavior.

## Changes

- Confirmed `src/train/star_uvt_feature_targets.py` now owns
  `FeatureTargetTensor`, cached feature-target loading, target-grid chunking,
  RGB grid adapters, render-to-target-grid adapters, channel adaptation,
  normalization, and streaming channel stats.
- Rewired `src/train/train_star_uvt_feature_rgb_probe.py` so it imports
  `_load_cached_feature_target`, `downsample_rgb_to_grid`, and
  `upsample_grid_rgb` from `star_uvt_feature_targets.py` instead of importing
  cached-target loading through `train_star_uvt_feature_overfit.py` and carrying
  duplicate interpolation code.
- Rewired `tests/test_star_uvt_feature_target_adapter.py` so the feature-target
  behavior tests import feature-target helpers from the new owner module. The
  same test file still imports sparse visual/support/checkpoint helpers from the
  trainer because those remain trainer-specific mechanics.
- Updated `CODE_ORGANIZATION.md` to record the new boundary.

## Validation

Compile:

```bash
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/star_uvt_feature_targets.py \
  src/train/train_star_uvt_feature_overfit.py \
  src/train/train_star_uvt_feature_rgb_probe.py \
  src/train/train_star_uvt_rendered_feature_rgb_probe.py
```

Focused tests:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_feature_target_adapter.py \
  tests/test_star_uvt_feature_rgb_probe.py \
  -q
```

Result: `43 passed`.

## Remaining Modularization

The rendered-feature RGB probe still imports `_render_rgb_chunks` and
`_sparse_target_grid_pixel_ids` from `train_star_uvt_feature_overfit.py`. That
is the next clean slice if we keep shrinking trainer-as-library coupling: move
STAR feature sparse-grid/render-probe mechanics into a small helper module only
after checking that the helper will be used by more than one script.
