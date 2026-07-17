# World Tubes Visibility Gauge Update And WorldFoam Paper Split

Date: 2026-07-05

## Context

The user brought in an external response that sharpened the current paper
direction. The main correction was that the World Tubes paper should not blur
two different claims:

```text
World Tubes:
    baseline-compatible camera-path compiler for dynamic Gaussian splats

WorldFoam:
    lifted opacity/transmittance model over camera ray fibers
```

The response also pushed on a visibility issue: if we depth-marginalize a
primitive down to `(u,v,t)` too early, we lose the lifted depth/order
information needed to reproduce baseline Gaussian compositing. The fix is to
compile two related objects:

```text
Trace atlas:
    alpha_i(u,v,t), c_i(u,v,t), support_i

Visibility gauge atlas:
    lifted depth intervals/densities, support-overlap graph, order certificates
```

WorldFoam then becomes the natural second paper: do not force it into the
World Tubes compatibility paper. It uses the same camera-bundle language but
changes the visibility semantics to Beer-Lambert transmittance over
`(u,v,t,z)`.

## Current Working Model

The split now looks like:

```text
Paper 1: World Tubes in Gauged Camera Space
    - Dynamic Gaussian splat semantics.
    - Pullback through camera-ray bundle.
    - Schur/fiber marginalization for UVT footprints.
    - Separate lifted visibility gauge atlas for depth/order.
    - Compiled interval VJP and frame-count scaling.
    - Main baseline: per-frame replay of the same representation.

Paper 2: WorldFoam in Gauged Camera Space
    - Bounded world cell complex.
    - Pull cells into lifted camera foam sigma(u,v,t,z).
    - Render by transmittance prefix, not primitive sorting.
    - Cech/AABB, witnessed power complex, and gauge connection diagnostics.
    - Current evidence: Metal speed microgates and trainable smokes.
    - Current gap: real quality and official parity are not solved.
```

## World Tubes Changes Made

Patched:

```text
research_notes/gauged_uvt_trace_atlas/paper/WORLD_TUBES_PAPER_DRAFT.md
research_notes/gauged_uvt_trace_atlas/paper/WORLD_TUBES_EXPERIMENT_PLAN.md
```

Key edits:

- sharpened the sublinear claim: no information-theoretic sublinear output-pixel
  bound; the claim is sublinear growth of dominant world-side bottlenecks and
  observed sublinear training-time growth in tested regimes.
- added the lifted **visibility gauge atlas**:

```text
O_Gamma = { C_l, G_l, Delta_l, Pi_l, R_l }
```

- separated footprint support from visibility/order:

```text
footprint/support: alpha_i(u,v,t)
visibility/order: lifted depth intervals or depth densities before z is thrown away
```

- added pairwise support-overlap order certificates:

```text
Delta_ij^-(C) = z_i^-(C) - z_j^+(C)
Delta_ij^+(C) = z_i^+(C) - z_j^-(C)

Delta_ij^+ < 0  => i definitely in front of j
Delta_ij^- > 0  => j definitely in front of i
otherwise       => split, commute if visually negligible, or fallback
```

- clarified that pair complexity is local support-overlap graph complexity,
  not global `N^2`.
- added a baseline-compatible theorem shape:

```text
If every support-overlapping pair is certified ordered, certified commutable
below epsilon, or rendered by fallback, compiled compositing matches baseline
Gaussian compositing up to trace and commutation error.
```

- added a short WorldFoam-as-lifted-transmittance bridge, but explicitly kept
  WorldFoam as a second paper rather than the core World Tubes claim.
- added related-work anchors for Sort-free Gaussian Splatting and Gaussian
  Blending as evidence that scalar alpha sorting/blending is an active
  limitation in the literature.

## WorldFoam Paper Packet Added

Created:

```text
research_notes/worldfoam_paper/README.md
research_notes/worldfoam_paper/WORLD_FOAM_PAPER_DRAFT.md
research_notes/worldfoam_paper/WORLD_FOAM_EXPERIMENT_PLAN.md
```

Also updated:

```text
research_notes/README.md
```

### WorldFoam paper thesis

WorldFoam represents world matter as bounded cells/foam regions. A camera
program pulls those cells into a lifted ray-fiber opacity field:

```text
sigma_l(y,z) = sum_j 1_{Gamma_l(y,z) in F_j}
                    sigma_j(Gamma_l(y,z))
                    J_l(y,z)
```

Rendering uses:

```text
tau_l(y,z) = integral_{z_front}^{z} sigma_l(y,s) ds
T_l(y,z)   = exp(-tau_l(y,z))
I(y)       = integral T_l(y,z) sigma_l(y,z) c_l(y,z) dz
```

This dissolves discrete sorting into cumulative optical depth. It is a cleaner
visibility story for translucent crossings, finite exposure, rolling shutter,
and fast camera programs, but it is not baseline-compatible Gaussian splatting
unless run in a discrete/quadrature compatibility mode.

### WorldFoam evidence boundary

Positive current evidence:

```text
Gate4/native-cutwalk 2/4/8/16f:
    WorldFoam total    3.008 / 3.014 / 3.323 / 4.095 ms
    WorldFoam backward 2.739 / 2.517 / 2.561 / 3.796 ms
    scale 1.361x total / 1.386x backward over 8x frame increase

matched STAR:
    total    5.003 / 5.943 / 8.092 / 9.794 ms
    backward 2.629 / 3.411 / 5.083 / 6.768 ms
```

Repeated-fixture 32f speed smoke:

```text
WorldFoam total    2.829 / 3.248 / 4.414 / 4.643 / 6.371 ms
WorldFoam backward 2.557 / 2.965 / 4.054 / 4.254 / 6.001 ms
scale 2.252x total / 2.347x backward over requested 2 -> 32f
```

Important caveat: that 32f row repeats 16 loaded frames.

Quality gap:

```text
best current WorldFoam train PSNR:   about 12.248
best current WorldFoam heldout PSNR: about 12.857
gap to solid same-source baseline:   about 9.112 dB
gap to STAR UVT source route:        about 17.575 dB
```

So the WorldFoam paper is not ready as a full quality/SOTA paper. It can be:

```text
theory + prototype + speed/scale paper
```

or it waits until:

```text
synthetic exactness + same-representation replay + gradient correctness +
visibility-stress advantage + public quality floor + official parity
```

clear.

## What This Changes In Approach

Do:

- keep World Tubes as the near-term arXiv push.
- add visibility gauge atlas language to World Tubes so it does not pretend
  depth marginalization alone solves compositing.
- run same-representation baselines first: per-frame replay versus compiled
  route.
- start WorldFoam as a second paper with its own experiment gates.
- use a crossing translucent slab/sheet synthetic as the bridge experiment
  between the two papers.

Do not:

- claim WorldFoam quality parity from speed microgates.
- claim World Tubes solves physical volumetric transmittance.
- claim sublinear total work in output pixels.
- let topology math become decorative; Cech/witness/holonomy must predict
  heldout-free validation or be cut.

## Next Work

Immediate World Tubes:

1. Implement/record a synthetic visibility-gauge test where support-overlap
   pairs are certified ordered, commutable, or fallback.
2. Add one small WorldFoam transmittance teaser scene to the World Tubes paper
   only as motivation for the second paper.
3. Convert the current manuscript into LaTeX once the baseline tables are
   stable.

Immediate WorldFoam:

1. Build the synthetic exact suite: constant-density sphere, crossing slabs,
   crossing Gaussian sheets, thin occluder, semi-transparent cloud.
2. Add per-frame WorldFoam replay baseline so compiler speed is isolated.
3. Add gradient correctness for prefix/suffix transmittance.
4. Rerun clean real `2/4/8/16/32f` loaded-frame sweeps without repeated frames.
5. Decide after synthetic and replay gates whether this is a prototype paper
   now or a full rendering paper later.
