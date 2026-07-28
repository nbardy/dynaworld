# Browser anisotropic capacity live ablation

## What changed

- Replaced the active 16-float isotropic primitive with a 24-float world-space
  anisotropic Gaussian: three log-scales, normalized quaternion, calibrated
  projected covariance/conic, and matching WGSL train/render plus CPU validation.
- Added analytic conic VJP through perspective depth, log-scales, and quaternion
  normalization. Added finite-difference coverage for center, scales, and all
  quaternion channels.
- Preserved the portable shared-memory alpha tape through 768 splats and added a
  storage-backed path above 768. Re-exported the canonical Coffee Martini bundle
  with 1,536 distinct train-visible SfM seeds; the UI now exposes 96-1,536.
- Cached CPU conic projections once per camera-frame. Before this, anisotropic
  validation took about 20 seconds because it rebuilt matrices for every grid
  pixel; afterward heldout PSNR/SSIM reached the UI every 2,048 steps while
  training remained uninterrupted.

## Live observations

- Isotropic/hybrid 768 fast-path check: about 793 completed steps/s.
- Storage-tape 1,536 check: about 374 steps/s; visibly denser but slightly worse
  at matched wall time, so 768 remains the default.
- Unbounded anisotropic scales produced broad streaks by roughly 9k steps.
  A 4:1 scale aspect guard and a final `0.10 * lrPosition` scale LR removed the
  worst failure while allowing aspect ratio to leave 1:1 early enough to inspect.
- Initial-opacity `-1.5` and `0.75 * geometryScale` max-scale ablations starved
  coverage and reached only about 11.0-11.9 dB heldout near 10k steps.
- The retained compromise uses opacity logit `-0.60`, nearest-neighbor radius
  `0.75x`, max world scale `1.0 * geometryScale`, and the 4:1 aspect guard. One
  live run reached 12.9 dB heldout at step 10,280 and reported about 841 steps/s.

## Remaining limitation

The result is still a blurry point cloud rather than a clean reconstruction.
Anisotropy and extra capacity fix two false constraints, but the active renderer
still uses one global anchor-camera order for every view/time and has no tiled
camera-depth sort, densification, relocation, or pruning. Those are now the
highest-value structural fixes; another LR sweep cannot repair visibility order.
