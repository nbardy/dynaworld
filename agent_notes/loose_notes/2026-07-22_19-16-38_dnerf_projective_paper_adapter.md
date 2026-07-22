# D-NeRF matched-trajectory projective paper adapter

## Why this lane exists

The World Tubes paper closure requires one controlled D-NeRF sequence. D-NeRF
is not synchronized multicamera video: its official train and test splits are
separate posed trajectories. For `bouncingballs`, all 20 official test times
occur exactly in the 150-frame training timeline, so the honest controlled
contract pairs one observed train pose with one held-out test pose at each
official timestamp.

## Implemented contract

- Added a checked-in 20-time manifest for `bouncingballs` using train indices
  `0,7,15,23,31,39,47,54,62,70,78,86,94,101,109,117,125,133,141,149`
  and test indices `0..19`.
- Added RGBA loading with an explicit black-background policy, Blender/OpenGL
  to OpenCV camera conversion, and per-frame relative poses.
- Added the 600-step progressive 512 protocol, one-row control matrix, and a
  two-step runtime-smoke protocol.
- Routed both the STAR/dynamic-3DGS comparison and WorldFoam through
  `camera.rig_init=dnerf`.
- Routed World Tubes through `projective_first_order + legacy_pinhole` for this
  moving-camera dataset. Neural3D static-camera rows retain
  `static_view + dataset_lens`.
- Generalized Metal trace statistics to cover moving-camera projected traces,
  so the fail-closed evidence schema still receives active trace/pair/fallback
  diagnostics.

## Important failure caught by runtime smoke

A time-varying `bundle.train_w2c` does not by itself make STAR use moving
cameras. The comparison runner's default `uvt_camera_sequence_mode=static_view`
silently selects one pose for an entire trace. The first smoke therefore ran
but would have been methodologically invalid. The corrected runner selects the
projective moving-camera compiler explicitly for D-NeRF.

## Verification

- Focused Python gate: `26 passed`.
- Real bundle load: train/heldout `(1,20,3,48,64)`, moving train poses, distinct
  held-out poses, official times `[0,1]` at the endpoints.
- Corrected three-lane runtime smoke:
  `outputs/benchmarks/2026-07-22_dnerf_projective_gauged_smoke_v2/`
  completed two optimizer steps, backward passes, held-out evaluation, LPIPS,
  cost accounting, and Metal diagnostics for all three representations.
- Smoke World Tubes metadata records
  `camera_sequence_mode=projective_first_order` and
  `pose_source=dnerf_matched_time_blender_to_opencv_relative_pinhole`.

The smoke is mechanical evidence only. The checked-in 600-step seed-17 matrix
remains the publication row to run from a clean commit.
