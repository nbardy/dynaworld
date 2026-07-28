# Browser Trajectory-3DGS Plateau Audit

Date: 2026-07-29

## Verdict

The active SPA backend is **trajectory-gated dynamic 3DGS**. It is not native
4D Gaussian Splatting, Spacetime Gaussian Splatting, Dynamic 3D Gaussians, or
World Tubes.

For normalized time `t`, each primitive uses

```text
center(t) = center0 + velocity * (2t - 1) + harmonic * sin(2 pi t)
alpha(t) = sigmoid(opacity) * mix(shared_gaussian_gate(t), 1, static_mix)
```

Its anisotropic spatial covariance, rotation, scale, and RGB are constant over
time. All splats share one temporal width. The harmonic UI mode only enables
the sine term above; it does not invoke the repo's STAR/UVT or World Tubes
research code.

## What The Plateau Is Saying

The old loaded run at step 988,608 reached:

| Metric | Train | Heldout |
| --- | ---: | ---: |
| PSNR | 19.7 dB | 15.3 dB |
| SSIM | 0.689 | 0.265 |

It was not quietly converged. Between 8,192-step validation snapshots its RMS
parameter deltas remained approximately `3.7e-3` center, `1.2e-2` motion,
`2.9e-2` scale, `1.4e-2` rotation, `2.7e-2` color, and `6.2e-2` opacity.
Constant learning rates were keeping it in a noisy late orbit.

A fresh instrumented run exposed stronger structural limits:

- initialization had `0/2048` dynamic splats, median static mix `0.92`, and
  `94.5%` mean support at the temporal endpoints;
- under the full-image objective, median static mix rose above `0.98`, endpoint
  support rose above `98%`, and no ordinary splat became dynamic;
- the former 3:1 shape cap clipped 91% of active splats at initialization;
- after nominal growth, roughly 1,200-1,300 of 4,096 allocated slots remained
  below the raster threshold;
- an opt-in 25% dynamic reserve preserved 235 dynamic splats at about 10.9k
  steps, but slightly regressed the matched early PSNR (`15.3/14.6 dB` versus
  `15.5/15.0 dB` train/heldout). The runtime branch was removed.

These are single-scene diagnostic smokes, not baseline claims. They explain
failure modes; they do not establish a quality win.

## Main Bottlenecks

1. **Effective topology is much smaller than allocated topology.** The 4,096
   slot count includes hidden and raster-dead slots. Split/recycle uses
   opacity/gradient proxies, not the spatial loss map, so it cannot reliably
   put births on missing structure or motion.
2. **The representation collapses static.** Full images are dominated by the
   stable kitchen. A shared broad temporal gate plus trainable static mix makes
   "be persistent" the easiest solution.
3. **Appearance is under-parameterized.** One constant RGB must explain every
   view and time. There is no spherical-harmonic/view feature, exposure model,
   or temporal appearance.
4. **Motion and shape are under-parameterized.** There is one affine-plus-sine
   center path, but no time-varying rotation/scale and no per-splat temporal
   width.
5. **Initialization is not yet a clean baseline.** The default 4,096-point
   bundle is external/unverified. The train-only known-pose pycolmap cloud is
   verified but sparse.

## Paper-Space Comparison

- Original 3DGS uses view-dependent appearance and gradient-driven
  densification/pruning rather than treating allocated slots as effective
  support: <https://github.com/graphdeco-inria/gaussian-splatting>.
- Fudan 4DGS represents anisotropic 4D primitives with spatial-temporal
  rotation and a native temporal marginal:
  <https://fudan-zvg.github.io/4d-gaussian-splatting/>.
- SpacetimeGS adds per-splat temporal opacity, cubic position, time-varying
  rotation, richer appearance, and training-error/depth-guided births:
  <https://openaccess.thecvf.com/content/CVPR2024/html/Li_Spacetime_Gaussian_Feature_Splatting_for_Real-Time_Dynamic_View_Synthesis_CVPR_2024_paper.html>.
- Dynamic 3D Gaussians stores persistent identities with per-time position and
  rotation and adds local rigidity/isometry priors:
  <https://dynamic3dgaussians.github.io/>.
- FastGS uses multi-view loss maps to score useful densification and pruning:
  <https://fastgs.github.io/>.

World Tubes is a different compiler/raster architecture: UVT traces, conditional
depth/order support, and a compiled shared adjoint. It could improve temporal
reuse and ordering, but it does not by itself supply the missing appearance,
motion, or allocation capacity above.

## Implemented In This Pass

- renamed the visible model to harmonic/linear trajectory 3DGS;
- added default-on, toggleable geometry/appearance LR decay over 120k steps;
- changed Adam from `beta2=0.99, epsilon=1e-6` to `0.999, 1e-8`;
- extended density statistics from `0.995` to `0.999` decay;
- changed the default from 4,096 initialized slots to 2,048 initialized plus
  fixed-capacity growth to 4,096;
- extended cheap recycling every 512 steps through step 120k;
- raised the still-bounded tiled anisotropy cap from 3:1 to 6:1;
- added asynchronous diagnostics for dynamic/persistent counts, static-mix
  quantiles, endpoint temporal support, anisotropy saturation, raster-dead
  slots, and current LR multipliers;
- tested and removed a dynamic-reserve heuristic after it failed the early
  quality A/B.

The Apple WebGPU parity harness still passes with `8.2e-8` maximum RGB error,
`1.1e-8` objective error, and seven active gradient families. The focused
browser suite passes 67 tests.

## Highest-Value Next Experiments

1. Add an independent-per-time 3DGS oracle on the same bundle. If it cannot
   overfit train views, fix raster/topology/appearance before adding 4D motion.
2. Feed a downsampled full-image loss map into residual/depth-guided birth and
   pruning. This is the missing density signal; an opacity quota is not a
   substitute.
3. Add a bounded view-dependent appearance control (small SH or learned
   view/color basis) and measure train/heldout gaps.
4. Implement per-splat temporal width plus richer center/rotation trajectories
   as a separate, explicitly named SpacetimeGS-style backend.
5. Compare verified train-only known-pose initialization against the external
   4,096-point seed under matched settings.
6. Record long matched runs in `BASELINES.md`; do not promote the short smokes
   above.

Do not build browser SfM next. The bundle already carries calibrated cameras,
and duplicating calibration/split semantics would violate the browser adapter
boundary. Improve train-only point support through
`src/train/export_dynaworld_browser_bundle.py` and the existing known-pose
pycolmap route first.
