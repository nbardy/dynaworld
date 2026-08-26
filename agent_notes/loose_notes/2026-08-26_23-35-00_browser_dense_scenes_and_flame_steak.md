# Browser dense scenes and Flame Steak

## Goal

Move the scene selector into the visible header, stop presenting Cook Spinach
and Cut Roasted Beef as if their source captures had only two cameras, and add
a substantially more dynamic scene without creating browser-only calibration
or split semantics.

## Data changes

- Cook Spinach source inventory: 21 calibrated cameras. New browser split uses
  20 train cameras and holds out `cam16`.
- Cut Roasted Beef source inventory: 20 calibrated cameras. New browser split
  uses 19 train cameras and holds out `cam16`.
- The old 272-point Cook/Cut initializers remain valid verified subsets because
  `cam14` and `cam18` are both still train cameras. They are sparse initializers,
  not statements about the training-camera count.
- Downloaded the official 1,199,567,884-byte `flame_steak.zip` release and
  extracted its 21-camera, 300-frame capture locally.
- Added a Flame Steak `train20_holdout1` canonical manifest with `cam16` held
  out and `cam14` as anchor/condition camera.
- Ran known-pose SIFT/pycolmap over all 20 train cameras at source frame 0 and
  1024 px. It found 45,091 keypoints, 46 requested/verified pairs, 6,519 raw
  points, and 6,487 accepted points. Mean accepted reprojection error was
  0.584 px; median was 0.378 px. The browser retains a farthest-point 4,096
  subset. The report declares `train_only_verified=true` and the corrected
  `neural_3d_llff_opencv_relative_pinhole_v2` pose source.
- Transcoded compact 384x288 H.264 full-rate streams for every camera. Each
  scene's 20-21 streams occupy only about 1.4-1.5 MiB at CRF 28.
- The 384 fallback atlases are much larger: about 41-42 MiB per dense scene.
  Static deployment is therefore atlas dominated, not MP4 dominated.

## Packaging lesson

The canonical exporter takes roughly four minutes per 20-camera/16-frame
bundle on this host because it exact-seeks the full-resolution source videos.
Exporting 96 and 384 separately needlessly repeats those seeks. Added
`src/train/downsample_dynaworld_browser_bundle.py` to derive a 96 bundle from a
384 bundle in about one second. It resizes every frame independently, avoiding
Lanczos bleed across horizontal atlas seams, and preserves normalized camera
intrinsics, poses, splits, timeline metadata, stream URLs, and seed provenance.

One transcode launch attempted a four-process zsh job throttle, but the
noninteractive runner launched all 21 ffmpeg children. This temporarily
saturated CPU and slowed the concurrent Cook export. No performance result was
recorded during that contention, and all later source exports/SfM jobs ran
sequentially.

## UI and runtime

- Scene selector now sits in the header and wraps cleanly at 390 px.
- Selector labels show Coffee 17/1, Cook 20/1, Cut 19/1, and Flame 20/1.
- A real scene switch exposed a null optional GPU-buffer disposal bug:
  `buffer.destroy?.()` dereferenced `null` before optional-call handling.
  Disposal now uses `buffer?.destroy?.()` (and the same guard for array items).
- Desktop and 390x844 visual checks showed all six Flame panels, compact metrics,
  and controls without overlap. Flame -> Cook -> Flame reload testing produced
  no new console errors after the disposal fix.

## Dataset research decision

`research_notes/browser_dynamic_multicam_dataset_shortlist.md` ranks:

1. PanopticSports / Dynamic 3D Gaussians for rapid sports motion and small fast
   objects, but only behind a canonical adapter and explicit CMU/prepared-data
   license handling.
2. Neural 3D Video Flame Steak/Flame Salmon for immediate compatible flame and
   reflection stress; Flame Steak is shipped now.
3. The UCSD Deep 3D Mask Volume dataset: 96 scenes, 10 cameras, 120 FPS, MIT,
   requiring an HDF5/video adapter.
4. Charge as a synthetic cinematic diagnostic lane.
5. Kubric-4D rejected for this local/static deployment because even its tiny
   official subset is about 48 GiB.

## Verification

- Browser Node suite: 195/195 passing.
- Focused Python bundle/catalog/downsample suite: 23/23 passing.
- `git diff --check`: clean.
- Real isolated browser: four selector entries; Flame metadata reports 20 train,
  one heldout, 300 frames at 30 FPS; desktop and mobile layouts inspected.
