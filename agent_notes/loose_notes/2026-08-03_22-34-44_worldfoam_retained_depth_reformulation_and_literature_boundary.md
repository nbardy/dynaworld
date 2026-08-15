# WorldFoam retained-depth reformulation and literature boundary

Date: 2026-08-03

## Context

The user asked why World Tubes can eliminate depth with a Gaussian Schur
complement while WorldFoam cannot, how static foams are fast, whether dynamic
foam literature already contains the needed formulation, and what exact prompt
should be handed to a strong mathematician.

The repository already contained the long canonical handoff:

```text
research_notes/worldfoam_paper/WORLD_FOAM_DYNAMIC_DEPTH_ORDER_MATHEMATICIAN_PROMPT.md
```

This pass updated that file instead of creating a competing prompt. No MPS,
Metal, CUDA, extension build, dataset decode, or training workload ran.

## Observed facts

### Static foam literature

- [Radiant Foam](https://arxiv.org/abs/2502.01157) uses a static 3D Voronoi
  partition. Each point has one owner, rays walk neighbor-to-neighbor through
  convex cells, and piecewise-constant cell transfer is integrated exactly over
  each segment.
- [Power Foam](https://arxiv.org/abs/2604.24994) adds bounded power cells,
  conservative sphere-overlap/Čech candidate adjacency, oriented interfaces,
  and a rasterization route.
- [Semantic Foam](https://arxiv.org/abs/2604.26262) adds cell semantics, not
  physical scene time.

These papers make a static ray efficient by removing overlapping-primitive
search, reusing local adjacency, integrating one cell segment analytically, and
terminating opaque rays early. They do not prove a constant total ray cost. A
ray crossing `R` cells still costs at least `Omega(R)`, and a spatial partition
does not bound line-stabbing depth.

### Adjacent but incomplete literature

- Exact kinetic data structures and kinetic regular triangulations maintain
  moving weighted-point topology with polynomial certificates and event queues:
  [CGAL exact KDS paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC3001684/),
  [CGAL kinetic 3D regular triangulation](https://doc.cgal.org/4.11/Kinetic_data_structures/classCGAL_1_1Kinetic_1_1Regular__triangulation__3.html).
- [STBVH](https://www.embree.org/papers/2017-HPG-msmblur.pdf) shares a spatial-
  temporal acceleration structure across motion segments and uses temporal
  splits for difficult motion.
- [K-Planes](https://arxiv.org/abs/2301.10241),
  [D-NeRF](https://arxiv.org/abs/2011.13961), and
  [DynMF](https://arxiv.org/abs/2312.00112) compress persistent dynamic-world
  state with factorized fields, canonical deformation, or shared trajectory
  bases.

The literature search found no published neural-rendering method that combines
all of:

```text
moving power/Voronoi cells
+ exact/certified continuous ray-order events
+ retained ordered emission-absorption
+ frame-density-independent compiled word work
+ a reusable sparse cross-time world adjoint.
```

This is a dated search result, not a proof of absence. Novelty claims still need
an updated formal literature audit before submission.

## Current model

### Why Gaussian Schur closure does not transfer

A pulled-back Gaussian stays Gaussian when ray depth is marginalized. The
conditional covariance is a Schur complement, so World Tubes can replace the
continuous fiber with a compact sensor-time trace without changing the
Gaussian-family semantics.

WorldFoam's intended observable depends on the front-to-back order of
differently colored media. Let one ray/time owner word be

```text
W_p(t) = ((i_1, L_1), ..., (i_R, L_R)).
```

For P0 density/color, one segment has exact affine transfer

```text
beta_r = exp(-rho_i L_r)
m_r    = (1 - beta_r) c_i
T_r    = (beta_r, m_r).
```

Composition is

```text
(beta_1,m_1) star (beta_2,m_2)
  = (beta_1 beta_2, m_1 + beta_1 m_2).
```

It is associative but generally noncommutative. Therefore attenuation alone
can collapse through summed optical depth, but differently colored emission
cannot be made order-blind without an error or material restriction.

### The correct Schur analogue

The foam analogue is a systems/mathematical factorization, not another
marginal:

```text
kinetic geometry
-> event-free owner-word charts
-> exact ordered transfer at adaptive J nodes
-> certified temporal transfer and sparse derivative actions
-> streamed residual-to-node reduction
-> one word/world VJP per active compiled block.
```

At one fixed time, an arbitrary P0 ordered word collapses exactly to four
transfer scalars `(beta, RGB m)`. The difficult object is the function of time
and parameters:

```text
t -> T_p(t),
t -> D_theta T_p(t) v,
t -> D_theta T_p(t)^T lambda,
```

including the event set where the owner word changes.

### Honest complexity target

For fixed physical interval, world, camera program, and tolerance:

```text
heavy structure/word/world-VJP
  = O(Topology(S,E,Q) + sum_(p,c) J_(p,c) R_(p,c))

sample evaluation/residual reduction
  = O(sum_(p,c) F_(p,c) J_(p,c)
      + N_fallback J_max^2)
    + Omega(PF).
```

The `Omega(PF)` target/output work is unavoidable. The claim is that expensive
cell-word traversal, topology work, world differentiation, and reverse
interaction storage are independent of requested frame density. Peak state
should depend on persistent world state, the largest live spatial/native
bundle, and bounded target/sample blocks, not an `F x R` tape.

## Three actual mathematical gaps

1. **Kinetic event complexity and maintenance.** Prove a complete event
   predicate set, exact degeneracy semantics, useful output-sensitive bounds,
   and affected-chart repair for moving sites/weights and camera rays.
2. **Primal-plus-operator temporal rank.** Certify total transfer and required
   sparse JVP/VJP actions, not only RGB. Rank should depend on event distance,
   motion, optical depth, material contrast, and tolerance rather than sample
   density.
3. **Training reuse.** Derive practical predicate-margin trust regions and
   local repair. If most optimizer steps invalidate most charts, inference may
   remain memory-light while end-to-end geometry training does not.

These are the questions for the mathematician. A negative theorem that proves
the need for bounded algebraic degree, shared motion bases, separated events,
bounded line depth, or a restricted material family is useful.

## Current repository boundary

Observed CPU/source evidence already includes exact ordered transfer,
constant-state word VJP, direct affine kinetic 3D sites with low-degree event
polynomials, exact root isolation, exhaustive and active-owner chart compilers,
multi-chart dispatch, stable-stratum sparse geometry/material VJPs, actual-rank
native-shaped lowering, block-major fake-native accumulation, and an event-free
directional trust certificate.

The older topology-token claim was stale. The current legacy material path now
uses an explicit LRU policy with separate limits for cached entries, cached
tensor bytes, and cached-plus-one-live-token bytes. It preflights before
allocation and checks actual token bytes. It does not intrinsically retain one
token per spatial block.

A new source-only step target-frame cache is independently green: it decodes
each `(view,frame)` at most once across bundles, owns bounded CPU float32 frame
clones, fails before decode when its explicit resident-byte budget would be
exceeded, and clears on close. Focused cache/provider tests reported `10
passed`. It is not yet integrated into a verified production native trainer.

The newer lazy native material-step coordinator is not complete evidence at
this snapshot. Inspection found unresolved helper references and incomplete
`BaseException`/generator/lock cleanup in
`research_experiments/world_foam_lane2/kinetic_lazy_native_material_step.py`.
The native binary has not been rebuilt or allocator-measured. Dense-F replay,
bounded compiled-program reuse across optimizer steps, the full geometry VJP
and recompilation policy, unified-runner integration, and public training remain
open.

## Backtracks

- **Invalidated:** “Find a foam Schur complement that removes depth.” It would
  erase the noncommuting color-order phenomenon.
- **Invalidated:** “4D automatically gives cross-frame reuse.” A 4D world can
  still be sliced and replayed independently at every time.
- **Invalidated:** “Constant transition cost means constant ray cost.” The
  total still depends on crossed cells `R`.
- **Invalidated:** “32 GB is mathematically required.” The source contract can
  be bounded; 32 GB was a safety recommendation for older eager paths, not an
  intrinsic requirement.
- **Weakened:** fixed shared-SPD(4) power cells as a general dynamic world. In a
  fixed gauge they form a restricted common-translation/fixed-normal family.
  Direct kinetic 3D residual sites are the current general frontend candidate.
- **Unresolved:** whether local event repair is worth its complexity. Full
  recompilation may remain the correct production choice if affected regions
  are usually large.

## Falsification tests

1. Fix world/camera/interval/tolerance and sweep only `F`. Kill the compiled
   claim if event count, `J`, word forward count, word VJP count, or reverse
   interaction peak grows with `F`.
2. Compare primal-only rank selection with sparse material and geometry tangent
   probes. Kill forward-only certification if tangent error remains large.
3. Perturb geometry just inside/outside certified predicate margins and compare
   local repair against a fresh exact compile. Kill local repair on any missed
   event or owner-word mismatch.
4. Measure invalidated chart fraction across real optimizer steps. If most
   steps invalidate most charts, keep exact recompilation and narrow the
   amortized-training claim.
5. On a quiet approved runtime, measure allocator peak and command-buffer
   in-flight retention for `F={16,64,256}`. Source byte ledgers alone do not
   prove native memory.

## Decision implication

No new material family or global 4D mesher should be invented now. The strongest
next mathematical pass is the output-sensitive moving-root/event-maintenance
problem already selected by the canonical prompt. The systems path should in
parallel finish the bounded dense-F data/program lifecycle and native runtime
measurement. WorldFoam remains a second-paper lane and must not delay the
World Tubes submission experiments.
