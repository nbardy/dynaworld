# Browser comparison grid and camera controls

## Request

Clean up the browser trainer so the first useful screen is a three-camera,
two-row comparison: GT above WebGPU result. Dragging any of the six cells should
move the corresponding result camera.

## Design decisions

- The first screen is a fixed three-column matrix. The columns remain the
  canonical comparison set: two training cameras and the heldout camera.
- The GT row is a fixed calibrated reference. Dragging a GT cell controls the
  render-only camera below it; the GT image itself cannot follow an arbitrary
  camera because the dataset has no image for that novel pose.
- Each column owns an independent orbit state initialized exactly from that
  column's calibrated OpenCV camera.
- Camera interactions are direct rather than gated behind a mode selector:
  left drag orbits, Shift/other-button drag pans, wheel dollies, double-click
  resets one column, and the toolbar reset restores all columns.
- A moved result gains an `orbit` label so it cannot be mistaken for a
  pixel-aligned calibrated evaluation view.
- Training and validation still bind the canonical camera buffer. Rendering
  uses a separate camera buffer with three appended private slots. Interactive
  camera changes therefore cannot alter train sampling, losses, gradients, or
  heldout metrics.
- The three WebGPU panels remain one 4:1 canvas. Three transparent interaction
  cells divide it into camera-owned regions without adding extra canvas or
  raster submissions.

## Page structure

- Moved Start, Step, and Reset into a compact header.
- Removed the redundant large single-target canvas and the buried Render Camera
  and Result Camera selectors.
- Placed the GT/result matrix before controls, stats, and charts.
- Reflowed controls into a responsive settings grid.
- Preserved all three columns at narrow viewport widths, with ellipsis and
  smaller labels rather than stacking cameras vertically.

## Renderer changes

- `trainerWebGpu3d.js` now reserves `MAX_RENDER_VIEWS` appended render-camera
  records rather than one.
- `writePreviewCameras()` packs and uploads up to three camera records in one
  `queue.writeBuffer` call.
- `trainingWorker.js` forwards `previewCameras` while retaining calibrated
  `viewIndices` as the fallback.
- `benchmarkResourcePlan.js` accounts for the two additional 80-byte camera
  records. The 30K exact allocation expectations increased by 160 bytes.

## Verification

- `npm test` in `web/dynaworld_browser_trainer`: 147/147 passed.
- `node --check web/dynaworld_browser_trainer/app.js`: passed.
- DOM contract check: 131 unique IDs and all 125 literal app ID references
  resolved.
- `git diff --check`: passed.
- The isolated server was restarted on port 8080 and returned 200 with
  `no-store`, COOP, COEP, and same-origin CORP headers.
- Automated live visual inspection was not available in this run because the
  in-app browser refused local-page control under its URL policy. Do not infer
  screenshot-level desktop/mobile acceptance from the source and unit gates;
  visually inspect the live page after reload.

## Files in this change

- `web/dynaworld_browser_trainer/index.html`
- `web/dynaworld_browser_trainer/styles.css`
- `web/dynaworld_browser_trainer/app.js`
- `web/dynaworld_browser_trainer/trainerWebGpu3d.js`
- `web/dynaworld_browser_trainer/trainingWorker.js`
- `web/dynaworld_browser_trainer/benchmarkResourcePlan.js`
- `web/dynaworld_browser_trainer/tests/benchmarkResourcePlan.test.mjs`
- `web/dynaworld_browser_trainer/tests/workerProtocol.test.mjs`
- `web/dynaworld_browser_trainer/README.md`

