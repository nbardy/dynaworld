# STAR UVT Camera-Path Compiler Review

Date: 2026-05-23

## Context

User asked whether the proposed 4D Gaussian sensor-time footprint / camera-path
compiler idea should change the STAR UVT direction.

Local evidence reviewed:

- `PROJECT_INDEX.md`, `TODO/README.md`, `EXPERIMENTS.md`
- `research_notes/framing_the_problem/framing_3.md`
- `research_notes/training_contract_v1.md`
- `src/train/star_uvt_feature_tube_model.py`
- `src/train/star_uvt_visibility_support.py`
- `third_party/fast-mac-gsplat/variants/star_uvt_v0/README.md`
- `third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/trainer_harness/world_tube.py`
- `third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/metal/star_uvt_kernels.metal`

## Current Model

STAR UVT already contains half of the proposed idea:

```text
ma:         [N, 3]  // u, v, t
q_uvt:      [N, 6]  // quadratic precision in sensor-time coordinates
depth0:     [N]
depth_beta: [N, 3]  // affine depth model over u,v,t
opacity:    [N]
color/feat: [N, C]
```

The renderer bins projected screen-time tubes into tile-time cells, sorts by
tile-center depth, detects unstable tile order, and falls back to per-sample
depth sorting when order can flip within a tile. That is already the
visibility-aware part of the camera-path compiler story in miniature.

What STAR UVT does not yet have is the proposed exact ray/4D-Gaussian depth
marginalization. Current world-tube projection is a local screen-space chart:
world-space tube -> pinhole/moving-camera linearization -> one quadratic UVT
footprint plus affine depth. This is cheaper and matches the current renderer
contract, but it is only a local affine approximation of projection, not an
analytic line integral through a 4D Gaussian.

## Implication

The proposal should not be framed as a replacement for STAR UVT. It suggests a
next STAR-compatible branch:

```text
World/ray Gaussian primitive
  -> compile to one or more STAR UVT chart records
  -> reuse the existing tile-time renderer, unstable-order fallback, and stats
```

In other words, STAR UVT becomes the charted sensor-time backend for a richer
world/ray primitive, not the thing to discard.

## Why This Matters

The current feature-tube quality blocker is coverage / visibility /
composition, not raw shader plumbing:

- dense RGB remains near `5.6-6.0` PSNR for many feature/visual routes;
- forced alpha and target-background oracle rows show recoverable content;
- alpha coverage stalls around the low 40% range;
- support birth/split improves coverage modestly but does not solve visual
  quality.

The camera-path compiler idea attacks a different bottleneck: repeated
projection/binning for known path or many shutter/rolling samples. It may
improve the renderer side and deepen the world-space semantics, but it will
not automatically fix the current STAR feature coverage failure.

## Recommended Change

Add a narrow research lane, not a wholesale pivot:

1. Keep `direct_atomic + index_add` as the practical STAR source-view path.
2. Keep `K=8/r64/o0.4` birth/split as the current support primitive default.
3. Start a path-compiled STAR UVT prototype that emits the existing
   `(ma, q_uvt, depth0, depth_beta, opacity, feature)` contract from a
   world/ray Gaussian chart compiler.
4. Measure whether the new compiler reduces projection/binning work or
   unstable/fallback regions before changing training objectives.

## First Cheap Falsification Tests

1. Synthetic line-integral reference:
   - build a tiny 4D Gaussian scene and compare analytic ray marginalization
     against high-sample depth integration.
   - output `L(u,v,tau)`, conditional depth mean, and depth variance.

2. STAR compatibility projection:
   - moment-match or Hessian-fit the analytic footprint around a chart center
     into STAR's existing `ma/q_uvt/depth0/depth_beta` tensors.
   - compare rendered alpha/RGB against dense sampled reference.

3. Chart split diagnostic:
   - sweep FOV, camera rotation speed, near depth, and rolling-shutter time.
   - report chart error and split count.

4. Visibility diagnostic:
   - reuse STAR's `tile_unstable` notion, but add depth-variance overlap from
     the analytic conditional depth.
   - report unstable/fallback tile fraction and quality loss.

5. Memory/amortization diagnostic:
   - compare ordinary per-frame STAR binning vs compiled interval tile-time
     entries.
   - report pair ratio, cache bytes, build time, and amortization frame count.

## Open Questions

- Can analytic conditional depth variance be compressed into STAR's current
  affine `depth_beta` model, or does the renderer need a `depth_sigma` tensor?
- Does a single chart per primitive/path interval match the current STAR
  quality envelope, or does chart splitting erase the memory win?
- For training, can the compiler be refreshed cheaply enough, or should it stay
  inference-first until geometry stabilizes?
- Does this help the feature-tube coverage failure, or only the world-space
  path-rendering story?

## Decision

Treat this as a new `path_compiled_star_uvt` research lane. Its first gate is a
synthetic correctness and memory/amortization report. Do not let it supersede
the current STAR support/visibility bridge or source-view direct-atomic path
until it shows quality-parity and cache payoff on known-path multi-sample
rendering.
