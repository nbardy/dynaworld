# Browser Orbit Camera And Native 384x288 Preset

## Goal

Add an interactive result camera without creating a second renderer or allowing
preview state to leak into calibrated training/validation. Promote 384x288 from
a synthetic kernel scale test to a real Coffee Martini browser preset and
measure its live cost.

## Decisions

- Reuse the existing WebGPU projection, depth sort, splat raster, temporal
  evaluation, and display-filter shaders.
- Keep canonical camera bytes immutable. Render bind groups use a separate
  camera-buffer copy with one extra preview slot; training and validation keep
  the canonical buffer.
- Generate the free camera in the dataset's OpenCV convention: +X right, +Y
  down, +Z forward. Initialize the pivot on the selected camera's principal ray
  at median positive seed depth.
- Apply browser geometry normalization only while packing the preview camera,
  scaling W2C translation exactly as calibrated cameras are scaled.
- Label the free camera `unscored`. It is a geometry inspection view, not a
  calibrated validation sample.
- Use the canonical multicamera export adapter for the high-resolution preset.
  This preserves the 17-train/1-heldout split, v2 LLFF-to-OpenCV calibration,
  frame indices, anchor frame, and seed provenance.

## Implementation

- `orbitCamera.js` contains dependency-free rigid camera math and immutable
  orbit/pan/dolly state transforms.
- Pointer input remains attached to the DOM canvas after it transfers rendering
  to `OffscreenCanvas`: left drag orbits; Shift-left, middle, or right drag
  pans; wheel dollies; double-click/reset restores the selected camera.
- `trainerWebGpu3d.js` appends one render-only camera slot and writes it through
  `queue.writeBuffer`. The same render shader reads that slot.
- `trainingWorker.js` accepts `previewCamera` as ordinary render state. This is
  independent of the continuously submitted optimization queue.
- The SPA now offers native `96x72` and `384x288` dataset presets. Switching
  resolution performs a full reload and fresh trainer allocation. The sampled
  ray control is disabled at 384 because it still binds the complete target
  tensor.
- The 384 bundle contains 18 atlases at 6144x288, each packing 16 synchronized
  384x288 frames. It uses the same 4,096 external/unverified Ex4DGS points as
  the 96 preset; this work does not change initialization provenance.

## Memory

Default 384x288 configuration: 4,096 active initial splats, 8,192 capacity,
packed-FP16 checkpoints, fast tiled full-frame backend.

| Storage | Measured or exact size |
| --- | ---: |
| WebGPU buffers reported by live trainer | 97.6 MiB |
| Shared RGBA8 target frame bank | 121.5 MiB |
| Shared FP32 per-camera temporal backgrounds | 30.375 MiB |
| One transient decoded RGBA8 atlas | 6.75 MiB |
| Largest GPU binding (packed checkpoints) | 54 MiB |

The old 486 MiB all-target GPU binding does not return: one camera/time target
is paged into a reusable GPU buffer. `SharedArrayBuffer` prevents worker copies
of the host target bank. The 384 configuration remained below the local Apple
adapter's 128 MiB per-binding limit and initialized successfully.

## Matched Live Measurement

Apple browser, same app controls, 4,096 active / 8,192 capacity, packed-FP16,
fast tiled full-frame objective, 15 Hz free-camera preview, full metrics enabled,
and zero tile overflow:

| Raster | Completed steps/s | GPU buffers |
| --- | ---: | ---: |
| 96x72 | about 309 | 11.3 MiB |
| 384x288 | about 130 | 97.6 MiB |

The raster has 16x more pixels but was about 2.4x slower in this live smoke.
Projection, sorting, optimizer work, and scheduling are not proportional to
pixel count. This is a local smoke, not a stable benchmark row; contention,
occupancy, active splats, and validation cadence can move it. The exporter was
finished before the timed interval, so it was not competing with these runs.

## Verification

- Pure camera tests preserve the exact selected camera at initialization,
  maintain a finite proper rotation under orbit/pan/dolly, and prove projection
  invariance under geometry scaling.
- Worker source-contract tests verify the render-only buffer and preview-camera
  plumbing.
- Browser artifact tests verify both checked-in bundles, all camera roles,
  frame indices, pose source, seed count, and atlas dimensions.
- Live browser checks confirmed a frozen-time canvas hash changes after a mouse
  drag, optimization continues while free orbit is selected, 384 initializes,
  loss stays finite, and tile overflow remains zero.
- Desktop and 390x844 layouts showed no horizontal overflow or result-label /
  canvas overlap. The 384 canvas rendered nonblank at the correct 4:3 aspect.

## Remaining Work

- Add calibrated camera-path interpolation only if it has a clear inspection
  need; do not report interpolated frames as heldout metrics.
- Run longer quality/capacity comparisons at both resolutions before adding a
  384 result to `BASELINES.md`.
- The native 384 mode still uses provenance-unverified external seeds. Rerun
  known-pose train-only triangulation under the v2 camera convention before
  claiming a leakage-free baseline.
- The sampled-ray control needs its own target paging conversion if it is ever
  expected to run at 384; this is lower priority than the tiled lane.
