# Train Torch Load Boundary

## Goal

Finish the train-local read-side checkpoint cleanup by routing the remaining
binary torch-load sites through the shared checkpoint helper without changing
their payload schemas.

## Changes

- Extended `checkpoint_utils.load_torch_checkpoint(...)` with an optional
  `weights_only` argument.
- Routed frame-cache reads in `sequence_data.py` through
  `load_torch_checkpoint(..., weights_only=True)`.
- Routed precomputed V-JEPA feature-cache reads in `video_feature_cache.py`
  through the same helper.
- Routed browser-bundle state-dict export reads in
  `export_dynaworld_browser_bundle.py` through the same helper.
- Added a focused `weights_only=True` smoke to `tests/test_checkpoint_utils.py`.

The callers still own their domain contracts: frame-cache keys and stale-cache
handling, feature-cache payload validation, and browser state-dict shape checks.

## Validation

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:. .venv/bin/python -m py_compile ...`
  passed for changed modules and relevant tests.
- `PYTHONPATH=src/train:. uv run --with pytest python -m pytest tests/test_checkpoint_utils.py tests/test_sequence_data_single_frame.py tests/test_video_feature_cache.py -q`
  passed: `10 passed`.
- `PYTHONPATH=src/train:. .venv/bin/python src/train/export_dynaworld_browser_bundle.py --help`
  passed.
- `rg "torch\\.load\\(" src/train` now finds only `src/train/checkpoint_utils.py`.

## Notes

This closes the direct `torch.load` cleanup for the train tree. It does not mean
all experiment scripts should be routed immediately; many one-off diagnostics
still own their checkpoint shape or remote execution contract. Future cleanup
should keep targeting reusable train/benchmark surfaces first.
