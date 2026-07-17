# Browser WebGPU Trainer SPA

Date: 2026-07-07 16:51 KST.

## Goal

The user asked whether the fast local Metal shader work should move toward
WGPU/WebGPU so we can test training in the browser, ideally as a SPA that
preloads a standard dynamic NeRF dataset, trains in-browser, and renders the
current result live.

## Context Read

- Read project startup docs (`PROJECT_INDEX.md`, `README.md` progress,
  `TODO/README.md`, `EXPERIMENTS.md`, `BASELINES.md`,
  `research_notes/data_contract.md`, `CODE_ORGANIZATION.md`,
  `agent_notes/key_learnings.md`) and the parent `RTK.md` compatibility shim.
- The freshest local shader evidence was the 2026-07-07 three-lane visual
  compare. The 64px, 128px medium, and 128px capacity tiers are green for
  dynamic WorldFoam/PowerFoam Metal, WorldTubes/STAR UVT Metal, and base
  dynamic 3DGS fast-mac Metal.
- The browser side already had `web/dynaworld_result_viewer/`, which renders
  exported split-token bundles with WebGPU but does not train.

## Implementation

Added a separate standalone SPA:

- `web/dynaworld_browser_trainer/index.html`
- `web/dynaworld_browser_trainer/styles.css`
- `web/dynaworld_browser_trainer/app.js`
- `web/dynaworld_browser_trainer/dataset.js`
- `web/dynaworld_browser_trainer/trainerWebGpu.js`
- `web/dynaworld_browser_trainer/README.md`

The app:

- Serves from repo root so it can fetch the local Neural3D
  `coffee_martini` preview at
  `data/multicam_val/clip_sets/multicam_val_v1_128_4fps_16f/previews/neural3d_coffee_martini_cam00_to_cam10.mp4`.
- Falls back to a deterministic synthetic D-NeRF-style mini fixture if the
  local preview is unavailable.
- Preloads target frames into a float storage buffer.
- Initializes 96 dynamic Gaussian/tube splats.
- Runs a WGSL compute SGD kernel over stochastic RGB samples.
- Uses ping-pong parameter buffers and live WebGPU rendering.
- Shows target and current WebGPU result side by side with step/loss/GPU stats.

Boundary: this is a browser-first source-view training prototype. It ports the
shape of the practical STAR UVT/WorldTubes path (dynamic splat parameters,
source-view reconstruction, compute-shader optimization, live renderer) but it
does not yet implement the Metal tile/depth/alpha compositor, native VJP parity,
PowerFoam ray walking, camera-family trace atlas, or heldout-camera training.

## Verification

Syntax:

```bash
node --check web/dynaworld_browser_trainer/app.js
node --check web/dynaworld_browser_trainer/dataset.js
node --check web/dynaworld_browser_trainer/trainerWebGpu.js
```

Browser smoke:

```bash
python3 -m http.server 8080
bash /Users/nicholasbardy/.codex/skills/playwright/scripts/playwright_cli.sh open \
  http://127.0.0.1:8080/web/dynaworld_browser_trainer/
```

Results:

- The page loaded `Neural3D coffee_martini preview`.
- WebGPU initialized on the Apple adapter.
- Status reached `Ready.` and the render pane stayed around 60 fps.
- Single-step smoke advanced to step `1` with finite loss `0.05983`.
- Short running smoke reached about step `1892` with displayed loss `0.01933`.
- Final console check reported zero errors and zero warnings.
- Screenshot saved to `output/playwright/dynaworld_browser_trainer_smoke.png`.

One false alarm: reading the WebGPU canvas through `drawImage` returned black
pixels, but the actual Playwright screenshot showed a nonblank blurred
reconstruction. Treat screenshot/browser view as the truth surface for this
canvas path.

## Next

- Add real renderer parity: depth buckets/tiles, alpha/transmittance, and
  source/heldout camera data.
- Decide whether the next browser artifact should consume a trained Metal
  checkpoint/export, or train a small browser-native scene from frames first.
- Do not add this to `BASELINES.md`; it is not a benchmark row.
