# STAR UVT Sparse Render Helper Cleanup

Date: 2026-05-21 19:54:08 Asia/Ho_Chi_Minh

## Goal

Continue the trainer modularization pass by removing the rendered-feature RGB
probe's dependency on `train_star_uvt_feature_overfit.py` helper internals.

## Changes

- Added `src/train/star_uvt_feature_rendering.py` as the shared owner for STAR
  UVT feature alpha-background composition and chunked RGB media rendering.
  `train_star_uvt_feature_overfit.py` and
  `train_star_uvt_rendered_feature_rgb_probe.py` now import
  `_compose_alpha_background_rgb` / `_render_rgb_chunks` from that module.
- Added `src/train/star_uvt_sparse_grid.py` as the shared owner for sparse
  target-grid interpolation plans, sparse target-grid forward projection,
  sparse target-grid VJP packing, and target-grid pixel-id selection.
- Rewired `train_star_uvt_feature_overfit.py` to consume sparse-grid helpers
  from the new module while keeping trainer-specific sparse visual losses,
  support birth/split, checkpoints, and schedules local.
- Rewired `train_star_uvt_rendered_feature_rgb_probe.py` so it no longer imports
  from the feature-overfit trainer. The rendered-probe media path now calls the
  shared `_render_rgb_chunks(...)` signature with explicit fixed-black eval
  alpha-background arguments.
- Added `tests/test_star_uvt_sparse_grid.py` to protect the moved sparse-grid
  math against dense trilinear forward and backward behavior.

## Validation

Compile:

```bash
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/star_uvt_feature_rendering.py \
  src/train/star_uvt_sparse_grid.py \
  src/train/train_star_uvt_feature_overfit.py \
  src/train/train_star_uvt_rendered_feature_rgb_probe.py \
  tests/test_star_uvt_sparse_grid.py
```

Focused tests:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_sparse_grid.py \
  tests/test_star_uvt_feature_target_adapter.py \
  tests/test_star_uvt_feature_rgb_probe.py \
  -q
```

Result: `45 passed`.

Alpha-background diagnostic:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_background_cheat_diagnostic.py \
  -q
```

Result: `2 passed`.

## Remaining Modularization

The rendered-feature RGB probe is no longer coupled to the overfit trainer.
The next useful STAR UVT cleanup should stay narrow: either move sparse visual
loss local-gradient helpers if another consumer appears, or stop here and shift
back to the broader trainer unification roadmap. Avoid extracting support
birth/split or checkpoint mechanics unless there is a real second consumer.
