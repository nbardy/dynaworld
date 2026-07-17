# Prepare Clip Re-Export Removal

## Goal

Delete a small compatibility surface now that active callers import clip
preparation from the data module directly.

## Evidence Before Editing

`rg` found no real code imports of `pipeline.render.prepare_clip`; the only
active import was the compatibility test in `tests/test_pipeline_helpers.py`.
Older loose notes and docs still mentioned the re-export as a temporary bridge.

## Changed

- `src/train/pipeline/render.py` now imports `sequence_data.prepare_clip` as an
  internal `_prepare_clip` helper for full-sequence rendering.
- Removed `prepare_clip` from `pipeline.render.__all__`.
- Removed the compatibility-only test that asserted
  `pipeline.render.prepare_clip is sequence_data.prepare_clip`.
- Updated `CODE_ORGANIZATION.md` and `TODO/trainer_landscape_unification.md` to
  record that the re-export is gone.

## Validation

Validation was run after the re-export removal with the broader
W&B-media/logging/helper test set:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_wandb_media.py tests/test_train_logging.py tests/test_pipeline_helpers.py -q
```

Passed: `15 passed in 3.85s`.

## Remaining

The larger render cleanup is still open: active code uses
`render_gaussian_frames_rasterized(...)`, while
`render_gaussian_frames_alpha_aware(...)` remains as a tuple compatibility API
for direct legacy callers/tests.
