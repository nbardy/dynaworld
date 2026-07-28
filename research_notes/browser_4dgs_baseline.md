# Native 4D Gaussian Splatting Browser Baseline Contract

Date: 2026-07-23

## Decision

Fudan ZVG's native 4D Gaussian Splatting is a relevant external baseline for
Coffee Martini and should be tracked as a reference implementation. It is not
equivalent to the browser trainer's harmonic trajectory option and is not yet a
SPA backend.

Primary sources:

- Project and papers: <https://fudan-zvg.github.io/4d-gaussian-splatting/>
- Official implementation: <https://github.com/fudan-zvg/4d-gaussian-splatting>
- Extended paper: <https://arxiv.org/abs/2412.20720>

## Representation Contract

Native 4DGS represents each primitive with a four-dimensional mean and a full
space-time covariance parameterized by 4D scale and a pair of quaternions. At a
requested time it conditions that 4D Gaussian into a 3D Gaussian, multiplies by
the temporal marginal, projects the conditional covariance, evaluates temporal
and view-dependent appearance, depth-sorts visible splats, and alpha composites
them. It interleaves optimization with spatial-temporal densification and
pruning; temporal gradients participate in the split decision.

The active browser model instead stores one 3D covariance plus linear and
optional sinusoidal center trajectories, a scalar temporal gate, and constant
RGB. It now has camera/time-correct ordering and fixed-capacity split/recycle,
but does not have a native 4D covariance, space-time rotation, conditional
Gaussian derivation, 4D appearance basis, or dynamic primitive count.

## Baseline Gate

A publishable comparison requires one adapter around the official implementation
using the canonical Coffee Martini loader/split semantics:

- identical train cameras, heldout camera, sampled times, image resolution, and
  image color convention;
- train-only initialization and explicit provenance for any COLMAP/SfM points;
- PSNR, windowed SSIM, LPIPS, L1, wall time, peak memory, primitive count, and
  rendered FPS;
- checkpoint selection without heldout optimization;
- a checked-in result artifact and appended `BASELINES.md` row.

Until that run exists, cite the paper as an external reference, not as a repo
baseline result. Do not copy its headline numbers into `BASELINES.md`: its
published protocol, resolution, training duration, and initialization are not
matched to the browser demo or the current paper runner.

## Browser Backend Roadmap

The eventual browser backend should be named `Native 4DGS` only after it has:

1. trainable 4D mean, scale, and 4D rotation;
2. conditional 3D mean/covariance and temporal marginal in WGSL;
3. camera/time depth ordering and matching analytic backward;
4. GPU-resident spatial-temporal density control;
5. the shared SPA worker, preview, validation, and export contracts;
6. a matched quality/performance ablation against calibrated dynamic 3DGS.

The current `Harmonic trajectory splats` mode remains a useful compact motion
basis ablation and must not be renamed to 4DGS.
