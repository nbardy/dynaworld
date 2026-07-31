# Browser camera contract and heldout growth audit

## Question

The browser preview showed cam04 improving while heldout cam06 appeared to
become increasingly smeared. Audit whether the browser reads ground-truth
calibration with the correct convention, numeric types, dimensions, and world
scale, then separate camera failure from training-schedule failure.

## Camera chain

- Source calibration is Neural3D `poses_bounds.npy`, sorted with the camera
  videos. Raw LLFF c2w columns are `[down, right, backwards]`.
- Canonical loading converts to OpenCV `[right, down, forwards]`, then casts to
  float32. The browser model world is the cam04 OpenCV camera frame.
- Each view transform is `inverse(c2w_view) @ c2w_cam04`. Cam04 is therefore
  identity and cam06 is a true heldout view, never an optimizer camera.
- JSON stores a row-major 4x4 transform plus four normalized pinhole values.
  JavaScript converts cameras and seeds to `Float32Array`; WGSL consumes five
  `vec4<f32>` records. Targets stay compact RGBA8 and decode to f32 for loss.
- Browser normalization multiplies seed XYZ and camera translation by one
  shared scale, `1 / median_positive_seed_depth`. This changes conditioning,
  not projection. Native LLFF units have no physical-unit metadata and must not
  be described as meters.

## Numeric checks

- Raster: 96x72, 18 cameras, 17 train and one heldout, 16 times, 4,096 seeds.
- Normalized intrinsics: `[0.5402198235, 0.7202930980, 0.5, 0.5]`.
- Pixel intrinsics: `fx=fy=51.861103`, `cx=48`, `cy=36`.
- Anchor max identity error: `1.19e-7`.
- Rotation determinant range: approximately `0.9999998..1.0000002`.
- Rotation orthogonality error: below `3e-7`.
- Median positive anchor depth: `14.0877` native units; browser scale is
  approximately `0.07098`.
- Applying the same scale to points and translations changed reprojection by at
  most `1.72e-5` pixel in the audit.
- Independent SIFT epipolar checks involving cam06 had median errors around
  `0.19..0.36` pixel.

These checks rule out an active transpose, axis-sign, focal normalization, or
point/camera scale mismatch. They do not prove the external Ex4DGS seed cloud's
original provenance, lens distortion, physical units, or sub-frame camera
synchronization.

## Reproduction

A new worker loaded from the corrected v2 bundle behaved well:

| Step | Train PSNR / SSIM | Heldout PSNR / SSIM | Heldout coverage |
| ---: | --- | --- | ---: |
| 0 | 6.8 / 0.056 | 6.9 / 0.051 | 16.7% |
| 3,008 | 17.9 / 0.835 | 23.0 / 0.840 | 89.3% |
| 6,400 | 23.4 / 0.896 | 26.5 / 0.896 | 94.8% |
| 12,288 | 25.0 / 0.916 | 27.0 / 0.909 | 95.9% |

The user-visible bad run was therefore consistent with a tab/worker created
before the camera-axis fix. Workers retain their loaded matrix copy; source
updates do not mutate an already-running optimizer. A full page reload is
required; Reset reuses the dataset already held by the page.

## Genuine long-run risk

The previous UI default reserved 30,000 splats but started with 4,096. Growth
adds 16 children every 100 steps beginning at step 600, so 30K fills only near
step 162,400. The geometry learning-rate schedule has decayed about 100x by
then. Split selection uses screen-gradient, alpha, and velocity proxies rather
than residual/depth/multiview evidence, and there is no pruning or geometric
consistency regularizer. This can keep perturbing geometry while appearance
fits training views, producing a real train/heldout divergence even with valid
cameras.

The stable default is now 8,192, which fills around step 26,100 while geometry
learning remains useful. Larger capacities remain explicit scaling/topology
experiments.

## Changes and verification

- Dataset loading now requires the corrected v2 pose source and validates
  finite proper-rigid transforms, positive intrinsics, shared anchor frame, and
  anchor identity.
- Preview draw, CPU preview error, validation metrics, and validation parameter
  deltas now use the active splat prefix instead of reserved capacity.
- Full validation cadence is 8,192 steps and the UI visibly labels
  `LLFF/OpenCV v2`.
- The default maximum is 8,192; 16K/24K/30K remain selectable experiments.
- `node --test web/dynaworld_browser_trainer/tests/*.test.mjs`: 142 passed.
- Canonical loader/export/relative-pose pytest gate: 41 passed.
- Fresh live WebGPU smoke: 12,288 steps, about 787 steps/s during the observed
  interval, finite metrics, zero tile overflow, and coherent heldout quality.
