# Scalar Payload Logging Boundary

## Goal

Continue the trainer-code modularization pass without broad rewrites. The
specific cleanup was to move scalar W&B payload construction out of
`pipeline.validation_media`, because that module now owns validation images,
videos, alpha masks, feature-PCA clips, and composite media grids.

## Changed

- Moved `_camera_state_metrics(...)` and `scalar_payload(...)` into
  `src/train/train_logging.py`.
- Updated `src/train/train_video_token_implicit_dynamic.py` to import
  `scalar_payload` from `train_logging`.
- Removed `scalar_payload` from `pipeline.validation_media.__all__`; the module
  is now media-only.
- Added `tests/test_train_logging.py::test_scalar_payload_builds_step_result_metrics`
  to cover base loss scalars, sequence counts, camera metrics, bank-rate terms,
  and aux loss terms.
- Updated `CODE_ORGANIZATION.md` and `TODO/trainer_landscape_unification.md`.

## Validation

```bash
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/train_logging.py \
  src/train/pipeline/validation_media.py \
  src/train/train_video_token_implicit_dynamic.py \
  tests/test_train_logging.py
```

Passed.

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_train_logging.py tests/test_pipeline_helpers.py -q
```

Passed: `15 passed in 4.05s`.

## Current Modularization State

- `train_logging.py` owns shared W&B setup, cadence checks, STAR row-output
  flattening/media attachment, and now base `StepResult` scalar payloads.
- `pipeline.validation_media` owns only media payloads and media diagnostics.
- `runtime_types` owns shared clip/result payload dataclasses and
  `build_step_result(...)`.
- `objective` owns RGB/feature background composition and alpha-safety guards.
- `sequence_data.ManifestSequenceSampler`, `clip_sampling.sample_clip_batch`,
  and `mixed_data_scheduler` own the reusable data/sampling boundaries.
- `render_dispatch.py` and `rendering.render_gaussian_frames_rasterized(...)`
  own the main render dispatch and typed raster payload boundaries.
- STAR UVT has shared runtime/checkpoint/colorizer/render-config/model helpers,
  but its scripts still have several experiment-specific wrappers because the
  kernel/gate surface is changing quickly.

## Remaining Cleanup Candidates

- Finish render-wrapper convergence: old tuple wrapper
  `render_gaussian_frames_alpha_aware(...)` should stay until all call sites
  are on `RasterizedClip`.
- Keep `pipeline.render.prepare_clip` as a compatibility re-export until `rg`
  proves no external scripts import it.
- Do not delete the dense STAR UVT prototype shim yet; it is still a gate runner
  and compatibility import path for older benchmark scripts.
- Audit legacy trainers before deleting: `train_camera_implict_dynamic.py`
  (typo in name), `train_image_encoder_implicit_camera_baseline.py`, the LTX
  subclass, and `dynamicTokenGS.py` are likely delete/archive candidates only if
  their configs are confirmed inactive.
- The next high-value code unification is not another base class. It is either
  render-dispatch convergence, a longer mixed same-view/heldout benchmark, or
  cleaning the remaining STAR UVT feature/probe wrapper imports once the selected
  kernel path stops moving.
