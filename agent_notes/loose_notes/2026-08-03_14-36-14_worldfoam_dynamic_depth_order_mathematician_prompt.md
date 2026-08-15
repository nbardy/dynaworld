# WorldFoam dynamic depth-order mathematician prompt

Date: 2026-08-03

## Request

Reformulate the retained-depth WorldFoam problem for a high-end mathematician:
explain why static foams are fast, identify the missing dynamic formulation,
preserve the ordered-depth physics that a Gaussian Schur marginal would erase,
and create a paste-ready research prompt.

## Main conclusion

At one camera time, an arbitrarily long P0 cell word composes exactly to the
four-scalar affine transfer `(beta, m_rgb)`. The hard object is therefore not a
smaller pointwise ray state. It is the time-dependent total transfer together
with sparse world-parameter JVP/VJP actions and the exact topology interval on
which the ordered word remains valid. The implemented adjoint is a
frozen-program derivative; a full derivative must additionally address event
times, chart/topology dispatch, adaptive nodes/rank, and interpolation weights.

A red-team pass exposed a representation-level gate that precedes compiler
optimization: slicing a fixed shared-metric 4D power diagram gives candidate
3D faces with constant spatial normals and time-varying offsets. It is not an
arbitrary kinetic 3D power diagram. The prompt therefore asks first whether
that restricted motion family is adequate, refinable at a reasonable rate, or
must be replaced by explicitly moving 3D sites/weights.

The leading formulation is:

```text
kinetic lower-envelope event compiler
+ exact ordered transfer at adaptive J nodes
+ affine-transfer Lie atlas
+ separate primal and tangent certification
+ streamed residual-to-node reduction
+ one node-level sparse world VJP
```

This is the WorldFoam analogue of World Tubes' architectural closure, not a
second Gaussian Schur complement.

## Literature interpretation

Primary-source review found the existing neural foam family to be static in
physical scene time. Radiant Foam gets speed from non-overlapping Voronoi
ownership, neighbor-to-neighbor traversal, and exact constant-cell segment
integration. Power Foam adds bounded power cells, conservative sphere-overlap
adjacency, and tile-friendly rasterization. Their constant-time language means
amortized cost per cell transition, not constant total ray cost.

The closest missing pieces live separately in kinetic Delaunay/regular
triangulations, spacetime acceleration structures, 4D simplex meshes,
differentiable tetrahedral rendering, and dynamic canonical/basis models. None
of the checked primary sources supplied a retained-depth neural foam with an
event-certified camera program and a shared cross-frame adjoint.

## Deliverable

Created:

```text
research_notes/worldfoam_paper/WORLD_FOAM_DYNAMIC_DEPTH_ORDER_MATHEMATICIAN_PROMPT.md
```

The prompt contains the formal 4D power-world/ray model, exact lower-envelope
reduction, ordered transfer semigroup, resource contract, current implemented
boundary, lower bounds, nine ranked research branches, a prerequisite
representation kill gate plus twelve theorem requests, primary-source map,
required response schema, GPU contract, counterexample suite, and stop rules.
It also separates arithmetic exactness on supplied binary64 values from
robustness to geometric/calibration perturbations. The WorldFoam and research
indexes were updated.

## Execution boundary

No MPS, Metal, CUDA, trainer, native build, or publication-scale experiment was
run. This was a source/document/literature pass only. The primary World Tubes
paper lane remains the submission priority; this prompt scopes the separate
WorldFoam mathematical research lane.
