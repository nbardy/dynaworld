# Token/Head Browser Export Config

## Context

We initially had a browser bundle exporter/viewer that wrote refined tokens plus decoded Gaussian payloads (`static_interleaved`, dynamic base bank, and dynamic motion/rotation/alpha coefficient arrays). Nicholas clarified that the durable result artifact should stay compressed: save the trained result's refined token arrays and the decoder head MLP weights only, not decoded splats.

## Changes

- Added a top-level training config export switch. `resolve_config` now accepts `"export": true` or an object with `enabled`, `output_root`, `id`, `sequence_index`, and `window_start`.
- `Trainer.run()` now calls `export_browser_bundle()` after the training loop and before `wandb.finish()`. The default destination is `outputs/browser_exports/<timestamp>_<wandb_run_name>[-wandb_id]/`.
- `src/train/export_dynaworld_browser_bundle.py` now writes `dynaworld_token_head_bundle/v2` bundles:
  - refined query tokens
  - static query tokens
  - dynamic query tokens
  - static Gaussian head MLP tensors
  - dynamic base/motion/rotation/alpha head MLP tensors
  - `time_proj` and `head_time_proj` tensors
- The v2 bundle intentionally does not write decoded Gaussian arrays. The browser JS decodes static/dynamic splats from tokens plus MLPs, then bakes a per-time interleaved frame for the WebGPU renderer.
- Enabled `"export": true` in `src/train_configs/local_mac_ablate_time_static_dynamic_96_32_crossattn4_rgb_uniform_strong_video_implicit_128_fast_mac_8192splats.jsonc`.

## Verification

- Python syntax: `uv run python -m py_compile src/train/export_dynaworld_browser_bundle.py src/train/train_video_token_implicit_dynamic.py`.
- JS syntax: `node --check` on `decoder.js`, `headless.js`, `run_headless_bundle.js`, `app.js`, and `staticGaussianWebGpu.js`.
- Diff hygiene: `git diff --check`.
- Export smoke:

```bash
uv run python src/train/export_dynaworld_browser_bundle.py \
  src/train_configs/local_mac_ablate_time_static_dynamic_96_32_crossattn4_rgb_uniform_strong_video_implicit_128_fast_mac_8192splats.jsonc \
  --output-dir /tmp/dynaworld_token_head_bundle_smoke_after_patch
```

Manifest check:

- `version`: `dynaworld_token_head_bundle/v2`
- `bundle_contract`: `refined_tokens_plus_decoder_heads`
- `counts`: 6144 static, 2048 dynamic, 8192 total splats
- forbidden decoded files absent: `static_interleaved.f32`, `dynamic_base_interleaved.f32`, `dynamic_A_mu.f32`, `dynamic_A_rot.f32`, `dynamic_A_alpha.f32`
- tensor payload size: 2,093,312 bytes across 63 tensor entries

Headless WebGPU smoke:

```bash
bun web/dynaworld_result_viewer/run_headless_bundle.js \
  --bundle-dir /tmp/dynaworld_token_head_bundle_smoke_after_patch \
  --out /tmp/dynaworld_token_head_bundle_smoke_after_patch/headless_render_v2_t025.png \
  --time 0.25 \
  --width 960 \
  --height 720
```

Result: Chromium/WebGPU loaded the v2 bundle, decoded tokens plus heads, rendered 8192 splats, and wrote a valid 960x720 PNG.

## Caveat

The standalone exporter without `--state-dict` still exports random-init weights. A real trained result comes from either passing a trained state dict to the CLI or letting the trainer hit the new `"export": true` path after training.
