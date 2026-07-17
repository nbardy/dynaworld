# Runtime Render Payload Ownership

Date: 2026-05-21 16:34

## Goal Context

The active goal is still trainer modularization: establish shared interfaces,
organize reusable code into clear submodules, and reduce duplicated trainer
logic without changing behavior.

The previous slices moved shared log cadence/W&B init to `train_logging.py` and
eval metric helpers to `pipeline.diagnostics`.

## Slice Implemented

Moved clip-level render payload dataclasses out of `pipeline.render` and into
`runtime_types.py`:

- `RasterizedClip`: features plus optional alpha from render dispatch.
- `RenderedClip`: stitched full-sequence validation RGB, optional feature/alpha
  sequences, camera state, and decoded-temporal metrics.

`pipeline.render` now owns orchestration functions only and imports those
payload contracts from `runtime_types.py`. The token-GS trainer imports the
payload types from `runtime_types.py` as well.

This aligns the tree with `CODE_ORGANIZATION.md`: runtime payloads live in
`runtime_types.py`; rendering modules own render mechanics, not shared data
contracts.

## Current-State Correction

While checking the planned render-dispatch cleanup, I verified that the older
procedural files referenced in `TODO/trainer_landscape_unification.md` are no
longer present under `src/train/`:

- `dynamicTokenGS.py`
- `train_camera_implicit_dynamic.py`
- `train_camera_implict_dynamic.py`
- `train_image_encoder_implicit_camera_baseline.py`

Active LTX/WAN-VACE feature configs dispatch through `src/train/train.py` to
`train_precomputed_feature_implicit_dynamic`, so the old shim-deletion plan is
historical context rather than live work.

## Validation

Passed:

```bash
PYTHONPATH=src/train:. rtk .venv/bin/python -m py_compile \
  src/train/runtime_types.py \
  src/train/pipeline/render.py \
  src/train/train_video_token_implicit_dynamic.py \
  tests/test_pipeline_helpers.py

PYTHONPATH=src/train:. rtk uv run --with pytest python -m pytest \
  tests/test_pipeline_helpers.py \
  tests/test_pipeline_diagnostics.py \
  tests/test_train_logging.py \
  tests/test_config_factory_helpers.py -q
```

Result: `29 passed in 1.30s`.

Also passed `git diff --check` and trailing-whitespace scans.

## Next Refactor Sequence

1. Mixed same-view plus heldout-view scheduler bridge remains the next real
   architectural gap.
2. Render-dispatch work should now focus on live callers of
   `pipeline.render.render_clip_sequence` and `rendering.render_gaussian_*`,
   not on deleted procedural trainer wrappers.
3. Entrypoint cleanup should be based on `src/train/train.py` and active
   configs, not stale file names from older notes.
