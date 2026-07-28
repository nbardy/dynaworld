# Browser Multicam Demo Rebase

## Scope

Rebased `web/dynaworld_browser_trainer/` from a source-view crop prototype onto
the canonical calibrated Coffee Martini contract. This remains a browser demo,
not a new model/renderer lane and not a Python trainer integration.

## Consolidation Decision

- Read and reused `multicam_video_data.py`, `multicam_val_data.py`, the typed
  paper protocol, `research_notes/data_contract.md`, and the renderer taxonomy.
- Removed the first hard-coded standalone Neural3D exporter.
- Added dataset-export mode to the existing
  `src/train/export_dynaworld_browser_bundle.py`.
- The exporter calls `load_multicam_video_bundle`; it does not duplicate LLFF
  camera conversion, split validation, frame sampling, or anchor-relative pose
  semantics.
- Kept `trainerWebGpu3d.js` isolated under `web/`; no browser abstractions enter
  the Python trainer hierarchy.

## Dataset And Initialization

- Manifest/sample: full 300-frame Coffee Martini paper row.
- Train cameras: `cam04`, `cam09`.
- Heldout validation-only camera: `cam06`.
- Browser samples: native indices `0,43,85,128,171,214,256,299`, `96x72`.
- Initialization: 768 visible XYZRGB points from the existing Ex4DGS SfM
  `input.ply`, transformed into the `cam04` OpenCV anchor frame. No browser
  COLMAP run and no heldout target pixel initialization.
- Initial radius/opacity were revised from sparse `0.24/-1.35` to
  `0.55/-0.60`; initial train/heldout loss improved from about
  `0.2461/0.2558` to `0.182629/0.192498`, while coverage rose from roughly 6%
  to 33%.

## Failures That Changed The Implementation

1. Browser MP4 seeking silently returned the same decoded frame for every
   requested time under the local serving path. The UI looked valid but motion
   sampling was exactly zero. Offline canonical decoding showed thousands of
   moving pixels, and a browser canvas signature proved frame 0 equaled frame
   7. The exporter now writes one tiny exact-frame PNG atlas per camera; browser
   signatures differ and motion sampling finds 5,635 train pixels.
2. WGSL compilation rejected `target` and `common` as reserved names, then
   rejected writable vector swizzles. Renamed/reconstructed values explicitly.
3. Apple WebGPU rejected 11 compute-stage storage buffers against its limit of
   8. Motion/static sample indices are now packed into one buffer; redundant
   GPU metrics/background buffers were removed; CPU validation supplies visible
   metrics.

## Browser Evidence

- Apple WebGPU adapter initializes with no app errors after `multicam67`.
- Three synchronized camera tiles loop over time; the camera selector switches
  the main target and live render to train or heldout views.
- Browser frame-0/frame-7 target signatures differ.
- Both simplified modes execute finite Adam updates.
- World Tubes-style: step 132 reached train loss `0.173302`, train PSNR
  `7.6 dB`, heldout loss `0.185356`, heldout PSNR `7.3 dB`, heldout coverage
  `33.6%` from the revised initialization.
- Earlier sparse-init traces: World Tubes-style step 117 reached
  `0.241910/0.252093` train/heldout loss; dynamic-splats-style step 119 reached
  `0.241840/0.252024`.
- Reported SSIM is an honest global-luma proxy and remains validation-only. It
  is not windowed SSIM/D-SSIM and is not a training loss.

## Verification

- `node --check` passes for `dataset.js`, `app.js`, and
  `trainerWebGpu3d.js`.
- `PYTHONPATH=src/train uv run --with pytest python -m pytest
  tests/test_browser_multicam_export_adapter.py
  tests/test_paper_training_protocol.py -q` passes `11` tests.
- Canonical bundle export completed and produced JSON plus three PNG atlases.
- Repo-root HTTP server returns `200` for the SPA, bundle, atlases, and source
  videos.

## Boundary And Next Gate

Do not add a baseline row or paper claim. The demo is still fixed-order,
all-pairs, isotropic splatting without depth sorting, densification, or true
windowed SSIM. Further browser quality work is admissible only after tiled
image-space backward, depth-aware composition, and a matched train/heldout
ablation. The canonical Python paper protocol remains the research truth
surface.
