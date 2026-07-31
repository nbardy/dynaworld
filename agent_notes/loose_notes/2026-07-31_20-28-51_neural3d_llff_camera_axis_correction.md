# Neural3D LLFF Camera Axis Correction

Date: 2026-07-31

## Trigger

The browser reconstruction looked substantially sharper from `cam09` than
from train camera `cam04` or heldout `cam06`. Matrix orthogonality, focal
scaling, camera indexing, and the browser WGSL row-major upload all looked
internally consistent, so the audit moved to image-grounded epipolar checks.

## Root Cause

`neural_3d_camera_from_poses_bounds` treated the raw first three columns in an
LLFF `poses_bounds.npy` row as the post-load NeRF basis
`[right, up, backwards]`. LLFF documents the stored basis as
`[down, right, backwards]`. Applying only a sign flip therefore produced the
wrong camera frame.

The correct raw-storage conversion is:

```text
R_opencv = R_stored[:, [1, 0, 2]]
R_opencv[:, 2] *= -1

[down, right, backwards] -> [right, down, forwards]
```

Translation remains the world-space camera center and is not axis-flipped.

## Evidence

SIFT correspondences were extracted from native 2704x2028 Coffee Martini
frame 159. For `cam04 -> cam05`, the old calibration had `90.54 px` median
symmetric epipolar error. The corrected conversion had `0.54 px` median,
`2.14 px` p90, and `89.3%` of ratio-test matches within `2 px`.

An all-camera half-resolution check over adjacent camera-name pairs found a
median pair error of `59.85 px` under the old conversion and `0.48 px` under
the corrected one. Sixteen of seventeen corrected pairs were below `1 px`;
the `cam10/cam11` name-adjacent pair has little overlap and only 75 matches, so
it is not a useful neighboring-view gate.

This is direct image evidence for the matrix convention and also supports the
existing sorted-video-name to pose-row mapping.

## Changes And Impact

- The canonical loader now emits pose source
  `neural_3d_llff_opencv_relative_pinhole_v2`.
- A synthetic raw-LLFF-axis regression test protects the reorder.
- The checked-in browser train17/holdout1 bundle was regenerated through the
  canonical exporter. Its 18 PNG atlases remained byte-identical; camera
  matrices, anchor-space seed coordinates, provenance, and pose identity
  changed.
- The 4,096 external Ex4DGS seeds are still explicitly unverified for heldout
  provenance. They are loaded as world-space points and transformed once into
  the corrected `cam04` OpenCV anchor frame.
- The 815-point known-pose pycolmap artifact from 2026-07-28 used the old
  camera conversion. It is failure evidence, not a reusable calibrated seed.
- Every earlier Neural3D quality row using
  `neural_3d_llff_relative_pinhole` needs a corrected rerun before comparison.
  Old metrics remain historical diagnostics only.

The browser shader's camera packing and projection were not the bug: it packs
the 4x4 row-major world-to-camera matrix followed by normalized intrinsics,
and WGSL dots each stored row with the homogeneous world point. Horizontal
coordinates use height-normalized screen units consistently with the 4:3
raster.
