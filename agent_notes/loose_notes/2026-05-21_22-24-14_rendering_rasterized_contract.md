# Rendering Rasterized Contract Cleanup

## Goal

Continue P4 render-dispatch convergence by moving the typed alpha-aware render
payload down into the renderer wrapper layer. Before this slice, the typed
`RasterizedClip` contract started in `pipeline.render`; `rendering.py` still
had a tensor-only batch wrapper and a separate tuple-returning alpha-aware
wrapper with duplicated fast-mac batch dispatch.

## What Changed

- Added `rendering.render_gaussian_frames_rasterized(...)`.
- The new wrapper returns `runtime_types.RasterizedClip(features, alpha)`.
- `rendering.render_gaussian_frames_alpha_aware(...)` is now a compatibility
  tuple wrapper around the typed function.
- Factored the repeated fast-mac batch dispatch into `_render_fast_mac_frames`.
- Factored repeated camera-to-world batch stacking into `_camera_to_world_batch`
  and reused it in dense, Taichi, and fast-mac batch calls.
- Updated `pipeline.render.render_clip_sequence(...)` to call
  `render_gaussian_frames_rasterized(...)` directly.
- Added `tests/test_rendering_contracts.py` for the typed dense render payload
  and tuple compatibility contract.

This preserves existing behavior: tensor-only callers still use
`render_gaussian_frames(...)`, tuple callers still use
`render_gaussian_frames_alpha_aware(...)`, and trainer paths now enter through
the typed `RasterizedClip` wrapper.

## Validation

- `py_compile` passed for:
  - `src/train/rendering.py`
  - `src/train/pipeline/render.py`
  - `tests/test_rendering_contracts.py`
- Focused tests passed:
  - `tests/test_rendering_contracts.py`
  - `tests/test_pipeline_helpers.py`
  - `tests/test_render_dispatch.py`
  - result: `14 passed in 1.37s`
- Runtime smoke passed:
  - `PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python src/train/train.py src/train_configs/local_mac_single_video_pretrain_100_local_encoder_64f_tiny_smoke.jsonc`
  - device: MPS
  - selected dense renderer for the tiny smoke config
  - completed the 1-step train/validation path
  - offline W&B run:
    `wandb/offline-run-20260521_222403-htb07ceu`

## Remaining Gaps

- Low-level renderer implementations still do not all expose alpha. The typed
  boundary represents optional alpha, not universal alpha support.
- `render_gaussian_frame(...)` is still tensor-only for legacy single-frame and
  research-experiment callers.
- A full `RenderConfig` dataclass is still not introduced; this slice kept the
  existing JSONC config mapping in place and only moved the payload boundary.
