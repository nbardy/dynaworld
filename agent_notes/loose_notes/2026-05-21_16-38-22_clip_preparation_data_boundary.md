# Clip Preparation Data Boundary

Date: 2026-05-21 16:38

## Goal Context

The active goal is broad trainer modularization: establish shared interfaces
across train runs, organize reusable code into clear submodules, and remove
duplicate trainer logic without changing behavior.

The remaining architectural gap is still the mixed same-view plus heldout-view
scheduler bridge. Before building that, clip slicing should live with the data
contracts rather than the render pipeline.

## Slice Implemented

Moved the legacy trainer clip tensor adapter into `src/train/sequence_data.py`:

- `make_clip(sequence, frame_indices) -> ClipBatch` remains the typed data
  contract.
- `prepare_clip(sequence, frame_indices) -> (clip_frames, clip_times)` now wraps
  `make_clip` for older trainer call sites that still expect
  `[1, K, 3, H, W]` frames and `[1, K]` times.

Updated active callers to import `prepare_clip` from `sequence_data`:

- `src/train/train_video_token_implicit_dynamic.py`
- `src/train/train_multicam_precomputed_feature_implicit_dynamic.py`
- `src/train/train_multicam_relative_pose_implicit_dynamic.py`
- `src/train/export_dynaworld_browser_bundle.py`
- `tests/test_pipeline_helpers.py`

`pipeline.render.prepare_clip` remains as a compatibility re-export so older
external imports do not break immediately, but render orchestration no longer
owns the implementation.

## Validation

Passed:

```bash
PYTHONPATH=src/train:. rtk .venv/bin/python -m py_compile \
  src/train/sequence_data.py \
  src/train/pipeline/render.py \
  src/train/train_video_token_implicit_dynamic.py \
  src/train/train_multicam_precomputed_feature_implicit_dynamic.py \
  src/train/train_multicam_relative_pose_implicit_dynamic.py \
  src/train/export_dynaworld_browser_bundle.py \
  tests/test_pipeline_helpers.py \
  tests/test_temporal_sampling.py \
  tests/test_sequence_data_single_frame.py

PYTHONPATH=src/train:. rtk uv run --with pytest python -m pytest \
  tests/test_pipeline_helpers.py \
  tests/test_temporal_sampling.py \
  tests/test_sequence_data_single_frame.py \
  tests/test_pipeline_diagnostics.py \
  tests/test_train_logging.py \
  tests/test_config_factory_helpers.py -q
```

Result: `44 passed in 1.06s`.

## Next Refactor Sequence

Follow-up slice in the same lane:

- Added `src/train/clip_sampling.py` with `sample_clip_batch(...)`, the shared
  frame-sampling plus `ClipBatch` construction helper.
- Added `ClipBatch.as_time_batch(...)`.
- Updated token-GS, known-camera, multicam, and camera-swap sample paths to use
  `sample_clip_batch(...)`, then adapt to legacy tuple returns at the trainer
  boundary.
- Added focused test coverage for camera slicing through the typed clip batch.

Validation after this follow-up:

```bash
PYTHONPATH=src/train:. rtk uv run --with pytest python -m pytest \
  tests/test_temporal_sampling.py \
  tests/test_pipeline_helpers.py \
  tests/test_sequence_data_single_frame.py \
  tests/test_pipeline_diagnostics.py \
  tests/test_train_logging.py \
  tests/test_config_factory_helpers.py -q
```

Result: `45 passed in 1.03s`.

1. Introduce a small sampler/bridge object for mixed same-view plus heldout-view
   batches, preserving separate loss names.
2. Gradually move trainer `sample_clip` methods to return `ClipBatch` directly
   instead of the legacy tuple adapter.
3. Remove the `pipeline.render.prepare_clip` compatibility re-export after all
   local and script callers import from `sequence_data`.
