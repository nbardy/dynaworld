# Browser Raster Filter And Aspect Audit

Date: 2026-07-29

## Question

Why do we cap Gaussian aspect ratio, what antialiasing is actually implemented,
and should Softmax Splatting be added?

## Findings

- The default tiled trainer bounds 3D scale standard-deviation ratio to 6:1.
  Initialization is bounded to 3:1 and the sampled fallback to 4:1.
- The cap is an optimizer and work bound, not a roundness regularizer. A 6:1
  scale ratio still means 36:1 covariance conditioning. Larger projected
  needles touch more tiles and make scale/rotation optimization less stable.
- A matched 12:1 browser run slightly improved train fit but slightly reduced
  heldout quality and increased tile work, so the measured default remains 6:1.
- The raster's screen filter is `sigma = 0.3 px`, implemented as
  `(0.3 / height)^2 I` in normalized screen covariance. That means 0.09 pixel
  squared variance, not the original 3DGS rasterizer's `+0.3` pixel squared
  covariance dilation.
- Pixels are still point samples. The filter is a conservative EWA-style
  footprint floor, not exact pixel integration and not full Mip-Splatting.
- Mip-Splatting adds determinant-based opacity compensation to its 2D filter
  and differentiates that coefficient in backward. It also adds a separate 3D
  frequency constraint. A forward-only port would be mathematically wrong.
- Analytic-Splatting approximates the Gaussian integral over each pixel area,
  directly addressing the point-sampling problem.
- Softmax Splatting solves collisions in optical-flow-based 2D forward warps.
  It does not antialias Gaussian footprints and cannot replace depth-sorted
  source-over compositing without changing the visibility model.

## Changes

- Shared one exported `FILTER_SIGMA_PIXELS` constant across CPU projection,
  sampled WGSL, tiled WGSL, and display WGSL.
- Added rationale comments at local-PCA initialization, sampled/tiled aspect
  guards, screen filtering, topology-fill stop, and source-over compositing.
- Added tests pinning the filter as a 0.3-pixel sigma and the sampled/tiled
  aspect constants.
- Expanded the browser README with the exact filter equation, limits, paper
  lineage, and the requirements for a correct Mip-style ablation.

## Decision

Do not add Softmax Splatting to the primary rasterizer. If antialiasing becomes
the next measured lane, implement the complete 2D Mip filter with its backward
and evaluate it across resolutions. Keep optical-flow/Softmax Splatting as a
separate temporal initialization or auxiliary-loss ablation.
