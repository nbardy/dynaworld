# WorldFoam G4 train/heldout mapped-cache bridge

## Scope

Completed only the public-target data boundary needed before a G4 worker can
consume real Neural3D observations. No MP4 was decoded, no dataset payload was
mapped, and no native/MPS work ran.

## What changed

- The target binding and bounded converter now accept exactly `train` or
  `heldout`; every other split label fails before decode/output creation.
- The Neural3D adapter validates both checked manifest camera lists, rejects
  overlap or unsorted/duplicate views, and selects only the requested split.
  The anchor remains a train camera even while a heldout camera cache is built.
- Train and heldout camera tensors remain split-specific, while their
  camera-generation digest seals the common rig, pose convention, source
  identities, resolution, and physical frame-time grid.
- `verify_train_heldout_target_dataset_pair` rehashes two distinct mapped
  caches, rejects view leakage or grid drift, and returns one path-free pair
  receipt suitable for a future public worker.
- The existing pixel-time layout is retained. It supports bounded chunked
  selected-pixel requests, including a full-frame-equivalent coordinate set,
  without making eager whole-video decode part of training.
- `neural3d_mapped_rgb8_offline_preflight` exposes PyAV cache construction as
  an explicit offline dependency. On the current `.venv` it reports
  `pyav_not_installed`; there is no fallback to an unsealed decoder.

## Verification

- Focused binding/converter/Neural3D tests: `48 passed`.
- Existing mapped selected-pixel target-source tests: `2 passed`.
- Python compilation of all three bridge modules passed.
- G4 planning no longer reports the train-only binding or missing-heldout-
  adapter blockers. Remaining blockers are the procedural production target,
  missing public worker/capability receipt, and stale native extension.

## Deliberately not claimed

This closes the split-safe cache preparation and identity boundary. It does
not build the three-scene caches, implement the public row worker, replace the
procedural trainer target source, rebuild native code, or measure G4 quality.
