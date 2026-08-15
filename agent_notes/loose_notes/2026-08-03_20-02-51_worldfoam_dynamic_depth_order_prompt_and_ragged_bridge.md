# WorldFoam Dynamic Depth-Order Prompt and Ragged Native Bridge

Date: 2026-08-03

## Context

The user asked for a precise reformulation of the retained-depth WorldFoam
problem suitable for a very strong mathematician. The motivating correction
was:

```text
World Tubes can Schur-marginalize Gaussian depth because Gaussians are closed
under marginalization. WorldFoam cannot remove depth order without removing
the differently colored overlap phenomenon it is intended to model.
```

The host has suffered resource incidents. This session therefore ran only
source inspection, CPU tests, documentation, and small fake-native lifecycle
fixtures. It did not build or execute Metal/MPS/CUDA, decode a public dataset,
or launch training.

## Current Model

Static neural foams are fast for sparse geometric reasons, not because they
integrate depth out:

1. the cells form a nonoverlapping ownership partition;
2. a ray walks from one neighboring owner to the next instead of testing every
   primitive at every depth;
3. a P0 cell segment integrates exactly as
   `alpha = 1 - exp(-density * physical_length)`;
4. bounded cells enable culling and early transmittance termination.

If a ray crosses `R` cells, whole-ray traversal remains `Theta(R)` under the
usual bounded/amortized neighbor-cost assumption. Power Foam's claimed
amortized constant work is per cell transition, not for an arbitrary complete
ray.

For dynamics, the useful analogue of a Schur complement is therefore not a
depth marginal. It is a compiled ordered program:

```text
kinetic owner charts
    -> ordered CSR owner words + physical node lengths
    -> exact affine-transfer products at J compiler nodes
    -> bounded temporal interpolation/residual reduction
    -> one shared sparse world VJP
```

For fixed world, camera program, physical interval, and tolerance, the desired
world-side work is

```text
Theta(topology/event compilation + sum_(p,c) J_(p,c) R_(p,c)),
```

while requested-sample work remains `Theta(sum_(p,c) F_(p,c) J_(p,c))` plus
the unavoidable `Omega(PF)` output/target stream. The claim is sublinear or
invariant expensive world work as frame density increases, not sublinear image
output.

Confidence: high for the fixed-program algebra and CPU/source lifecycle;
unresolved for production event maintenance and native performance.

## Durable Mathematician Handoff

The paste-ready brief is:

```text
research_notes/worldfoam_paper/WORLD_FOAM_DYNAMIC_DEPTH_ORDER_MATHEMATICIAN_PROMPT.md
```

Its first pass now requests exactly one result: prove, restrict, or refute a
multichart simple-root persistence and re-isolation theorem under one exact
directional update

```text
theta(eta) = theta_0 + eta * delta_theta, eta in [0,r].
```

The prompt distinguishes three predicate classes:

1. topology-event candidates;
2. root-bearing analytic/representation guards, such as a pair denominator;
3. non-root validity, positivity, and ray-noncollapse guards.

It also distinguishes polynomial-root continuation from semantic event
persistence. Every continued root must be reclassified from exact
co-minimality/activity and certified left/right owner words. An analytic guard
root may force a cut-chart refit without changing the semantic owner word or
incrementing topology-event count `E`.

The required first-pass attachments were reduced to the event-sufficiency
note, active/reference compilers, exact polynomial-root code, the directional
trust certificate, and focused tests. The broader paper/trainer/Metal archive
is explicitly optional.

## CPU/Source Integration Closed In This Pass

The following proof-oriented components now exist:

- `kinetic_native_equal_rank_lowering.py`: bounded actual-`J` batching for real
  `(track,chart)` rows, without a global temporal refinement or `J_max`
  padding;
- `kinetic_native_equal_rank_runtime_adapter.py`: cold-provenance and warm
  identity/layout/version lifecycle against an injected fake-native CPU
  backend;
- `paper_kinetic_ragged_sample_plan.py`: exact right-continuous joining of
  arbitrary view/frame/pixel observations to true-rank native blocks;
- `paper_kinetic_union_local_bar_assembly.py`: one caller-owned
  `[S_union,4]` request bar joining heterogeneous compact native blocks, with
  no per-request global `[S,4]` bar;
- `paper_ragged_material_bar_coordinator.py`: one global loss denominator,
  exact view/block coverage, repeated-ID `index_add`, and one optimizer
  authorization; and
- `kinetic_geometry_trust_region.py`: exact-rational reuse certificate for one
  strict event-free single chart along one directional homotopy.

The union-local assembler checks missing, duplicate, out-of-order, and foreign
native contributions. It proves assembly/provenance/count coverage only. The
sample reducer and native VJP remain responsible for numerical derivative
correctness.

## Memory Accounting

For `N` observations in an actual-rank `J` sample launch:

```text
native-shaped payload                 4 N J + 16 N + O(1) bytes
CPU/source wrapper with provenance    4 N J + 24 N + O(1) bytes
coexisting CPU/device row wrappers   >=4 N J + 28 N + O(1) bytes
```

For one spatial union:

```text
persistent source/map tensors         8 S_union + 8 sum_b S_b bytes
caller-owned request material bar    16 S_union bytes
per-request global material bar       0 bytes
```

These are logical tensor-payload formulas. Python heap, allocator reservation,
transfer temporaries, command buffers, and device peak are unmeasured. They do
not imply an intrinsic 32-GB requirement.

## Backtracks and Restrictions

- Fixed shared-SPD(4) sites are not the general dynamic foam. In a fixed gauge
  they yield one common translation, fixed relative site positions, affine
  relative weights, and constant candidate-face normals.
- Direct affine kinetic 3D residual sites with quadratic weights remain the
  selected general frontend because their ray/event predicates have bounded
  degree without frame-indexed parameters.
- The current event-free trust certificate does not justify reuse for an
  eventful or multichart program. Numeric event roots generically move under a
  nonzero geometry update.
- Continuing a scalar root does not prove that the root remains an active
  topology event. Activity/co-minimality can change along the root graph.
- The fake-native adapter is not a Metal runtime result, and logical byte
  accounting is not allocator evidence.
- The new modules are proof/source scaffolds, not a reason to fork another
  renderer or to put this lane ahead of the World Tubes submission queue.

## Literature Boundary

Primary starting points used in the prompt:

- Radiant Foam: static Voronoi ownership, adjacency walking, exact cell
  segments;
- Power Foam: bounded power cells, conservative sphere-overlap/Cech candidate
  adjacency, ray tracing plus rasterization;
- kinetic Delaunay/Voronoi work: certificate/event maintenance and warning
  that topology-event counts can be large;
- canonical deformation, higher-dimensional slicing, and spacetime
  acceleration: neighboring dynamic formulations, not a complete dynamic
  neural-foam plus shared-adjoint solution.

The current search did not find a published neural foam combining dynamic cell
topology, certified ray-order events, a continuous camera program, and one
cross-time sparse adjoint. This is a search result to verify before a novelty
claim, not proof that no such paper exists.

## Falsification Tests

The next mathematical work should stop if it cannot handle these exact small
cases without a full global recompile:

1. one rational moving root;
2. one irrational moving root;
3. nearly colliding simple roots;
4. two predicates sharing one algebraic root;
5. a root that persists algebraically but becomes semantically inactive;
6. a new root born in a previously certified complement;
7. a repeated/grazing root;
8. a denominator guard root that is not an owner event;
9. ray collapse or an endpoint event; and
10. a step just inside and just outside the certified radius.

If root continuation cannot beat exact recompilation on realistic active-event
fixtures, retain the existing exact compiler and spend no more architecture
budget on local maintenance.

## Verification

CPU/source combined gate:

```text
152 passed, 11 source-verifier subtests passed
```

This covered all `test_kinetic*.py` tests, the ragged sample/union/coordinator/
staging tests, and the native source verifier. The focused union-local/sample/
coordinator gate was `15 passed`. Ruff passed for the new modules and tests.

## Remaining Work

1. Obtain the multichart simple-root persistence/reclassification/re-isolation
   theorem or a decisive counterexample.
2. Generate and serialize dataset-bound per-view/per-spatial-block programs.
3. Bind the existing native ABI to the ragged plan and union-local coordinator.
4. Rebuild only in an approved quiet window; run bounded forward/VJP parity
   and allocator measurements before any performance claim.
5. Add streamed evaluator/checkpoint integration and a distinct unified-runner
   lane.
6. Extend from the current unbounded power partition plus global near/far to
   bounded PowerFoam sphere/vacuum events only if the simpler route survives.

