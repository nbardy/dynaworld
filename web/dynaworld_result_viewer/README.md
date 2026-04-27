# Dynaworld Browser Viewer

This is a small standalone browser viewer for Dynaworld static/dynamic split
exports.

What it does:

- loads an exported bundle directory from `src/train/export_dynaworld_browser_bundle.py`
- reads saved refined token arrays plus Gaussian decoder head MLP weights
- decodes static and dynamic splats in browser, then bakes dynamics for the selected time `t`
- renders the baked frame with a local WebGPU gaussian splat renderer

What it does not do yet:

- rerun the full video encoder / cross-attention stack in browser
- consume non-split Dynaworld decoders
- consume raw checkpoints directly from the page
- save decoded Gaussian arrays in the export bundle

## Export

Example:

```bash
uv run python src/train/export_dynaworld_browser_bundle.py \
  src/train_configs/local_mac_ablate_time_static_dynamic_96_32_crossattn4_rgb_uniform_strong_video_implicit_128_fast_mac_8192splats.jsonc \
  --state-dict /path/to/model_state_dict.pt \
  --output-dir /tmp/dynaworld_browser_bundle
```

Notes:

- `--state-dict` is optional; without it the exporter writes the random-init model state.
- The exporter currently requires `model.static_tokens` and `model.dynamic_tokens`.
- The bundle contract is `dynaworld_token_head_bundle/v2`: refined static/dynamic query tokens plus static/dynamic Gaussian head MLP tensors. It intentionally does not save decoded static splats or decoded dynamic banks.
- Training configs can set `"export": true` to write the same compressed bundle after training, under `outputs/browser_exports/<id>/` by default.

## Run

Serve the viewer on localhost:

```bash
cd web/dynaworld_result_viewer
python3 -m http.server 8000
```

Then open `http://localhost:8000`, choose the exported bundle directory, and use
the time slider or autoplay toggle.

## Headless PNG render

The current machine does not expose WebGPU directly inside the Bun runtime
itself, so the headless path uses Bun as the launcher and headless Chromium as
the actual WebGPU runtime.

Example:

```bash
bun web/dynaworld_result_viewer/run_headless_bundle.js \
  --bundle-dir /tmp/dynaworld_browser_bundle_smoke \
  --out /tmp/dynaworld_browser_bundle_smoke/render.png \
  --time 0.25 \
  --width 1280 \
  --height 720
```

Optional camera override:

```bash
bun web/dynaworld_result_viewer/run_headless_bundle.js \
  --bundle-dir /tmp/dynaworld_browser_bundle_smoke \
  --camera /path/to/camera.json \
  --out /tmp/render.png
```

Notes:

- `--camera` should point to a JSON camera spec matching the WebGPU renderer
  contract (`width`, `height`, `fx`, `fy`, `cx`, `cy`, and `w2c` or `c2w`).
- Without `--camera`, the headless page uses the bundle bounds plus a default
  orbit camera.
