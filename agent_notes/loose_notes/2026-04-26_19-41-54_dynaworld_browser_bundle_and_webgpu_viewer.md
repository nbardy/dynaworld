# Dynaworld browser bundle + WebGPU viewer

## What landed

- Added `src/train/export_dynaworld_browser_bundle.py`.
- Added a standalone browser viewer under `web/dynaworld_result_viewer/`.
- The current export/viewer path is intentionally scoped to the static/dynamic
  split decoder (`model.static_tokens` + `model.dynamic_tokens`).

## Export contract

- The exporter writes:
  - `manifest.json`
  - `refined_queries.f32`
  - `static_query_tokens.f32`
  - `dynamic_query_tokens.f32`
  - `static_interleaved.f32`
  - `dynamic_base_interleaved.f32`
  - `dynamic_A_mu.f32`
  - `dynamic_A_rot.f32`
  - `dynamic_A_alpha.f32`
  - `time_proj_*`
  - `head_time_proj_*`
- The viewer currently consumes the split static gaussians and decoded dynamic
  bank, not the raw token-space Gaussian heads.
- `time_proj` / `head_time_proj` are exported for future browser-side token
  decode work, but the current split viewer does not use them.

## Why this shape

- The parent `gsplats_browser` viewer renderer already wants baked per-frame
  Gaussian payloads.
- For the current split Dynaworld decoder, the dynamic residual bank is the
  exact narrow state needed to bake time-varying splats in browser.
- This avoided trying to rerun the full video encoder / cross-attention stack in
  JS on day one while still preserving the refined token arrays in the export.

## Verification

- `uv run python -m py_compile src/train/export_dynaworld_browser_bundle.py`
- `uv run python src/train/export_dynaworld_browser_bundle.py --help`
- `node --check web/dynaworld_result_viewer/app.js`
- `node --check web/dynaworld_result_viewer/decoder.js`
- `node --check web/dynaworld_result_viewer/staticGaussianWebGpu.js`
- `git diff --check`
- Real export smoke:
  - `uv run python src/train/export_dynaworld_browser_bundle.py src/train_configs/local_mac_ablate_time_static_dynamic_96_32_crossattn4_rgb_uniform_strong_video_implicit_128_fast_mac_8192splats.jsonc --output-dir /tmp/dynaworld_browser_bundle_smoke`
  - this wrote a real bundle for `test_data/test_video_small_128_4fps.mp4`
  - counts in the smoke manifest were `6144 static + 2048 dynamic = 8192 total`

## Bug hit and fixed

- First smoke failed in `_bounding_box(...)` because `static_interleaved` was on
  CPU while `dynamic_A_mu` was still on MPS.
- Fixed by explicitly moving all bound-computation tensors to CPU float32 before
  combining them.

## What is still narrow

- No checkpoint saving was added to the trainer; the exporter accepts an
  optional external `--state-dict` path and otherwise exports random-init
  state.
- No non-split decoder support yet.
- No browser-side camera-head decode yet.
- I did not run the viewer inside an actual browser in this pass; the browser
  side is syntax-checked and the export side is end-to-end smoke-tested.
