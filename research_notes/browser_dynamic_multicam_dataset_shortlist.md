# Browser Dynamic Multicam Dataset Shortlist

Date: 2026-08-26

## Decision

Use the existing Neural 3D Video adapter for immediate browser breadth, and add
`flame_steak` as the first visibly higher-motion scene. It has the same static
calibrated rig, synchronized 300-frame timeline, LLFF camera payload, and source
license as the existing browser scenes. This keeps the SPA a thin consumer of
the canonical multicamera contract.

Do not imply that the previous Cook Spinach and Cut Roasted Beef packages were
two-camera captures. Those were deliberately small `train2_holdout1` deployment
packages. Their source captures contain 21 and 20 calibrated streams. The new
browser packages use every source camera except the heldout camera.

## Ranked Candidates

### 1. PanopticSports via Dynamic 3D Gaussians

- Scenes: Basketball, Boxes, Football, Juggle, Softball, and Tennis.
- Why it matters: rapid articulated human motion plus small objects with large
  inter-frame displacement. This is a stronger motion benchmark than tabletop
  cooking.
- Camera protocol: the Dynamic 3D Gaussians benchmark uses a cleaned calibrated
  subset of the CMU Panoptic capture and is the most directly comparable public
  dynamic-GS baseline.
- Integration cost: a new canonical manifest/pose/image adapter, foreground
  mask semantics, and a license gate. The official Dynamic 3D Gaussians code
  links a prepared `data.zip`, but also states that its data-preparation code is
  not released. CMU Panoptic data is research-oriented rather than an asset we
  should republish casually on GitHub Pages.
- Decision: highest-priority research adapter; not a silent static-site bundle.

Official sources:

- https://github.com/JonathonLuiten/Dynamic3DGaussians
- https://dynamic3dgaussians.github.io/
- https://domedb.perception.cs.cmu.edu/

### 2. Neural 3D Video Flame Steak / Flame Salmon

- Why it matters: flames, reflections, smoke-like structure, and fast nonrigid
  appearance changes stress the current harmonic-trajectory 3DGS model more
  than hand/arm motion alone.
- Camera protocol: static synchronized multicamera videos; the paper describes
  10 seconds at 30 FPS and over-1K source resolution.
- Integration cost: already supported. `flame_steak.zip` is about 1.2 GiB;
  `flame_salmon_1` is a four-part archive totaling about 4.7 GiB.
- Decision: ship Flame Steak now. Keep Flame Salmon as an opt-in ingest until
  its source and deployed payload budgets are justified.

Official source: https://neural-3d-video.github.io/

### 3. Deep 3D Mask Volume Dataset

- Why it matters: 96 outdoor human-interaction and visual-effects scenes from a
  10-camera, 120 FPS static rig. It explicitly includes crossing people and
  dynamic vegetation, useful for disocclusion and temporal-stability tests.
- Format: raw videos or processed HDF5 plus LLFF camera poses.
- License: MIT according to the official project page.
- Integration cost: a canonical HDF5/video adapter, exact synchronized split
  definitions, and browser-oriented scene curation. Camera baselines target
  binocular extrapolation, so it complements rather than replaces dense rigs.
- Decision: second practical adapter after PanopticSports.

Official source: https://cseweb.ucsd.edu/~viscomp/projects/ICCV21Deep/

### 4. Charge

- Why it matters: high-fidelity synthetic film imagery with rich motion,
  lighting, depth, normals, segmentation, and optical flow; it defines dense,
  sparse, and monocular protocols.
- License: derived from Blender Open Movie assets under CC BY 4.0; verify the
  downloadable dataset's exact redistribution terms during adapter work.
- Integration cost: new image/camera/modality adapter and a bounded scene
  selection. Synthetic cinematic data is valuable for stress and correctness,
  but should remain separately reported from real captures.
- Decision: useful synthetic diagnostic lane, not the default real benchmark.

Official source: https://charge-benchmark.github.io/

### 5. Kubric-4D

- Why it matters: 16 synchronized cameras and complex multi-object dynamics,
  with depth, flow, normals, and instance labels.
- Cost: even the official tiny 20-scene subset is about 48 GiB; the validation
  archive is about 239 GiB. This is not proportionate for the current browser
  demo or this workstation's free-disk budget.
- Decision: generate a tiny license-compatible scene locally if a synthetic
  oracle becomes necessary; do not ingest the published archive now.

Official source: https://gcd.cs.columbia.edu/

## Adapter Acceptance Contract

A new selector option is ready only when all of the following are true:

1. Synchronized timestamps, intrinsics, world-to-camera transforms, and
   train/heldout camera identities are represented by the canonical multicam
   manifest rather than browser-only semantics.
2. Heldout pixels cannot enter initialization, optimization, densification, or
   regularization.
3. Initialization provenance names every contributing camera and coordinate
   frame.
4. The browser bundle has bounded 96x72 fallback atlases, compact full-rate
   streams, loading progress, and a measured deployed payload.
5. The scene is visually inspected at source, train, heldout, and perturbed
   orbit views before it is described as a working benchmark.
6. Redistribution terms permit the deployed assets. Otherwise the selector
   must fetch from an authorized source or remain a local adapter.
