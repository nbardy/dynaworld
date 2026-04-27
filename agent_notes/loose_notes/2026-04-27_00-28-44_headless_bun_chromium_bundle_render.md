# Headless Bun + Chromium bundle render

## Ask

- Follow up on the browser bundle viewer with a non-interactive image render
  path.
- The desired shape was "use bun and headless render via webgpu" for the
  Dynaworld exported bundle.

## What we checked first

- `bun --version` worked (`1.3.10`).
- But direct Bun runtime globals do **not** expose the browser WebGPU surface:
  `navigator.gpu = false`, `OffscreenCanvas = undefined`, `document = undefined`.
- Conclusion: Bun can be the launcher, but not the actual WebGPU runtime for
  this path on this machine.

## Implementation shape

- Kept the real rendering in headless Chromium with `--enable-unsafe-webgpu`.
- Used Bun only to run the launcher script.
- Reused the Dynaworld browser bundle decoder and static WebGPU renderer rather
  than introducing a separate CPU or native Bun render path.

## Files added / changed

- `web/dynaworld_result_viewer/headless.html`
- `web/dynaworld_result_viewer/headless.js`
- `web/dynaworld_result_viewer/run_headless_bundle.js`
- `web/dynaworld_result_viewer/decoder.js`
  - added URL-based bundle loading (`loadBundleFromBaseUrl(...)`)
- `web/dynaworld_result_viewer/staticGaussianWebGpu.js`
  - `renderFrameFromCamera(...)` now returns the rendered splat count
- `web/dynaworld_result_viewer/README.md`
  - documented the headless PNG flow

## Launcher contract

- `bun web/dynaworld_result_viewer/run_headless_bundle.js --bundle-dir <bundle> --out <png> [--time ...] [--width ...] [--height ...] [--camera ...]`
- The launcher starts a tiny local HTTP server, serves the viewer assets and the
  bundle directory under `/__bundle`, optionally serves `/__camera.json`,
  launches Puppeteer/Chromium, waits for `window.__headlessRender.ready`, and
  screenshots the canvas.

## Verification

- Syntax:
  - `node --check web/dynaworld_result_viewer/headless.js`
  - `node --check web/dynaworld_result_viewer/run_headless_bundle.js`
  - `git diff --check`
- Real render:
  - `bun web/dynaworld_result_viewer/run_headless_bundle.js --bundle-dir /tmp/dynaworld_browser_bundle_smoke --out /tmp/dynaworld_browser_bundle_smoke/headless_render_t025.png --time 0.25 --width 960 --height 720`
  - Page log reported `Rendered 8192 splats.`
  - Output file verified as:
    - `/tmp/dynaworld_browser_bundle_smoke/headless_render_t025.png`
    - `PNG image data, 960 x 720, 8-bit/color RGB`

## Small bug / cleanup

- First successful run still logged `Rendered undefined splats` because the
  WebGPU renderer method did not return a count. Fixed by returning
  `rendererState.data.count` from `renderFrameFromCamera(...)`.
- Chromium also requested `/favicon.ico`; the launcher server now returns `204`
  for that path to keep the console clean.

## Important limit

- This still uses the current split-viewer contract, not full raw token-space
  browser decode through Gaussian head MLPs.
- In practice the headless path loads the exported bundle, rebuilds the dynamic
  frame splats in JS, rasterizes them with the same browser WebGPU path, and
  writes a PNG.
