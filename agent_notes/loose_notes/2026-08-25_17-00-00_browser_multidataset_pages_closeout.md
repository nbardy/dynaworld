# Browser multidataset and Pages closeout

## Goal

Finish the browser trainer deployment lane without creating a parallel data
contract. Add a few useful scenes, expose honest load progress, document the
static-host behavior, and keep the Python paper-trainer hierarchy untouched.

## What changed

- Added Cook Spinach and Cut Roasted Beef through their canonical
  `train2_holdout1` manifests (`cam14`/`cam18` train, `cam16` held out).
- Built one-frame known-pose SIFT/pycolmap seeds at 1024 px. Each scene produced
  272 accepted train-only points from one verified camera pair. Mean reprojection
  error was 0.322 px for Cook Spinach and 0.338 px for Cut Roasted Beef.
- Transcoded only the selected three cameras per scene to 384x288 H.264. The six
  complete ten-second streams total about 1.5 MiB.
- Exported 96x72 and 384x288 browser bundles with 16-frame temporal pages and
  all 300 source timestamps. The exporter correctly rejected an attempted
  4,096-point export from a 272-point seed; the final bundles report 272 honest
  seed points and let the browser trainer populate its configured topology.
- Added a dataset selector and load indicator. PNG atlas transfers report bytes
  when available; MP4 temporal pages report camera/frame decode progress.
- Clarified that 384x288 is a 4x-linear stage, not 4K. A 3840x2880 RGBA32F target
  is already 168.75 MiB, before checkpoint and raster scratch storage.

## Verification

- Browser Node suite: 194/194 passing.
- Bundle assertions: canonical train/heldout cameras, `v2` poses, verified
  train-only seed provenance, 300-frame temporal stream.
- Real isolated-browser load: all three 96x72 selectors resolved their correct
  scene/split/cameras; Cut Roasted Beef also loaded directly at 384x288. The
  automation browser had no WebGPU adapter, so shader execution remained under
  the existing unit/headless gates rather than being claimed from that session.
- Source commit `157e7c1` was pushed. Repository Actions rejected the deploy job
  before creating steps, so the exact browser tree was published manually with
  a force-with-lease as Pages commit `6f92743`; GitHub Pages reported `built`
  and the public selector plus Cook Spinach bundle contract were fetched back.

## Dataset candidates

`flame_salmon_1` is visually attractive but its source release is a four-part
archive totaling roughly 4.7 GiB. It was intentionally not pulled into this
closeout. `flame_steak.zip` is about 1.2 GiB and is a more practical next flame
scene once a canonical split, train-only seed, and deployment-size budget are
chosen.
