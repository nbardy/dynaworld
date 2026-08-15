# WorldFoam production kinetic compiler and bounded native time state

Date: 2026-08-03

## Context

The user restated the actual systems goal: WorldFoam should share expensive
rasterization and backward work across physical time in the same sense that
World Tubes does. Requested frame density may increase cheap camera/sample and
target work, but it must not create a per-frame world traversal, per-frame word
tape, or per-frame world reverse. The host has suffered resource incidents, so
this work was restricted to source inspection, CPU fixtures, and bounded test
suites. No MPS, Metal runtime, CUDA, dataset decode, or publication training was
launched.

The central mathematical correction remains:

```text
World Tubes: Gaussian depth can be marginalized analytically.
WorldFoam: differently colored ordered transfer cannot be marginalized without
           deleting the depth-order phenomenon being modeled.
```

Therefore the analogue of the World Tubes Schur complement is not another
depth integral. It is a compiler:

```text
kinetic world + camera program
-> exact topology/order charts
-> ordered transfer at a bounded number of chart nodes
-> streamed sample residuals reduced to node cotangents
-> one sparse world reverse per node/chart.
```

## Formal target

Fix a physical interval, world, affine camera program, error tolerance, and
compiled structural choices. Let:

- `F` be requested temporal sample density;
- `S` be site count;
- `R_c` be the ordered run count on chart `c`;
- `J_c` be the accepted transfer rank/node count on chart `c`;
- `B_p` and `K` be resident spatial and temporal block sizes.

The intended complexity split is:

```text
unavoidable sample/output work       Omega(P F)
current block-first weight work       O(N_B F J)
streamed residual-to-node reduction   O(P F J)
compiled world reverse                O(sum_c J_c R_c)
reverse structural state              O(sum_c J_c), plus world gradients
resident sample state                 O(B_p K), not O(P F)
```

The theorem is always about varying `F` while fixing the physical interval and
event/rank complexity. It is not a claim that total image production is
sublinear in output size.

## Geometry model selected

The fixed shared-SPD(4) world remains a useful exact special case, but its
fixed-gauge time slices are one common translation of fixed anisotropic 3D
sites with affine relative weights and constant candidate-face normals. That
is too restrictive for several independently rotating/deforming boundaries.

The selected general CPU frontend is direct affine kinetic 3D power geometry:

\[
p_i(t)=p_{i,0}+t v_i,
\qquad
w_i(t)=w_{i,0}+w_{i,1}t+w_{i,2}t^2.
\]

For an affine ray `x=o(t)+z d(t)`, pair differences have the form

\[
D_i-D_j=A_{ij}(t)z+B_{ij}(t),
\]

with degree at most two. Adjacent-cut concurrence is

\[
B_{ij}A_{jk}-B_{jk}A_{ij}=0,
\]

with degree at most four. This gives exact bounded-degree predicates without a
frame axis in persistent world parameters.

This theorem currently covers an unbounded Euclidean power partition clipped
by fixed global near/far. Full bounded PowerFoam requires additional moving
sphere-entry/exit, radius-positivity, vacuum-gap, culling, and adjacency
events. Those are not implicitly covered.

## Implemented active compiler

The exhaustive reference and independent oracle remain the small-world truth
routes. The new active compiler is:

- `research_experiments/world_foam_lane2/kinetic_active_owner_chart_compiler.py`
- `research_experiments/world_foam_lane2/test_kinetic_active_owner_chart_compiler.py`

It starts from witnessed endpoint/right-continuous owner words, constructs
predicates from endpoint owners and active cuts against all competitors, groups
exact algebraic roots, discovers missing root-complement words monotonically,
and validates charts with all-site certificates. Predicate sources are cached
by unique owner word.

An early description called this `O(SR)`. That was false. The honest current
accounting is:

\[
O(U S R_{\max})
\]

for predicate construction over `U` unique witnessed words, plus

\[
O\!\left(W(S\log S+S R_{\max})\right)
\]

for `W` cumulative root-complement discoveries and all-site word
certifications. Final chart count alone is not the work bound. An
`O(delta R)` neighbor-only route would require a separately certified kinetic
regular/Delaunay or conservative supergraph, including its construction and
maintenance cost.

### Degeneracy correction

The first active version failed closed on owner-changing full-fiber ties but
could miss an inactive tie such as:

```text
L0(t) = (t - 1/4)^2
L1(t) = 0.
```

At the isolated equality, the selected owner word may not change, yet the
full-fiber equality still destroys a strict structural certificate. The
compiler now fails closed on `active_owner_full_fiber` in inactive classified
events and endpoint guards as well. A regression protects this behavior.

Persistent, simultaneous, and full-fiber events are still fail-closed rather
than supported with a perturbation convention. That is honest P0 behavior.

## Multi-chart transfer and certification

The following CPU path is now connected:

- `kinetic_chart_transfer_bridge.py`
- `kinetic_multichart_transfer_program.py`
- `kinetic_continuous_transfer_acceptance.py`

It seals compiler/program content digests, binds every active chart to exact
ordered P0 transfer at fixed `J_c` nodes, dispatches dyadic/binary samples
right-continuously, and fails closed inside irrational algebraic isolator
neighborhoods. It does not allocate a dense sample-by-chart table.

The residual reducer streams samples into one ragged tuple of node
cotangents:

```text
grad_chart_node_transfers[c] has shape [J_c, 4]
```

so reverse state is `O(sum J_c)`, not `O(F)`.

The continuous acceptance gate certifies the actual cleared second-form
barycentric runtime evaluator. It outward-rounds:

- primal transfer error;
- the complete referenced-material Jacobian;
- declared-norm material JVP error; and
- accumulated material VJP error.

It deliberately does not certify runtime float roundoff, dense runtime
fallback, irrational seam neighborhoods, geometry/event Jacobians, or
compiler-choice derivatives. On the adversarial moving-overlap fixture,
candidate ranks 2, 4, and 6 failed the declared budgets. That is a useful
rank-death result, not a promotion.

## Frozen-program world reverse

The stable-stratum reverse lives in:

- `kinetic_stable_stratum_vjp.py`
- `kinetic_multichart_stable_stratum_vjp.py`

It differentiates the sealed compiled objective with fixed:

```text
owner charts
chart endpoints
sample dispatch
node times
rank
interpolation weights
word topology
```

Within strict owner, order, denominator, positive-length, and ray-speed
margins, it accumulates gradients for:

- initial site positions;
- site velocities;
- quadratic weight coefficients;
- affine ray origins and directions;
- P0 density; and
- P0 RGB.

It includes the implicit cut derivative and the physical ray-speed factor
`||d(t)||`. Its expensive reverse is `O(sum J_c R_c)` and retains no
frame-indexed world tape.

### Numerical correction

The first forward used `1 - exp(-tau)` while another layer used
`-expm1(-tau)`. At tiny optical depth this creates a real forward/reverse
semantic mismatch. Forward and color VJP now share the stored stable alpha,
and a `tau=1e-18` regression protects the boundary.

### Derivative boundary

This is not the derivative of recompiling after changing geometry. In a
three-chart rank-2 fixture, the frozen-program position gradient was
`0.042315686`, while finite-differencing through recompilation gave
`0.041359246`, a `2.31%` difference. The correct paper language is
"frozen-program" or "stop-structural-choice" VJP.

A broader training claim requires one of:

1. a certified trust radius that keeps the structural program valid;
2. recertification/recompile after accepted geometry steps;
3. event/shape derivatives for the physical integrated objective; or
4. a declared surrogate/straight-through policy with empirical validation.

## Native time-state correction

The native source lifecycle previously allowed redundant global/chart sample
time ownership. It now keeps sample times only for the live temporal block:

```text
prepared token               owns no [F] or [F_c] times
sample launch                receives one CPU-float64 [K] block
post-launch                  releases that block after synchronization
```

The affected seam includes:

- the fused-slab native `ops.py`;
- `native_track_adapter.py`;
- `native_piecewise_topology_adapter.py`;
- `material_training_step.py`; and
- `host_memory_contract.py`.

The host memory report now counts native sample times once as `8K` bytes and
does not double-count them inside the cheap temporal tensor bucket.

This removes avoidable `O(F)` resident time clones. It does not remove the
unavoidable streamed targets/output or the current `O(N_B FJ)` construction of
barycentric weights. A global `FJ` cache could replace repeated block-first
weight construction, but that is a deliberate `O(FJ)` memory/order trade, not
free work sharing.

## Branches considered and decisions

### Another Schur complement

Rejected. Order-blind elimination is not exact for differently colored
segments and would erase WorldFoam's central phenomenon.

### Static fixed-SPD(4) as the general world

Rejected as the default general-motion representation. Retained as an exact
special case, oracle, and potential bulk-motion gauge component.

### Exhaustive all-triple compiler

Retained as a correctness oracle only. It is not the production complexity
claim.

### Active-owner closure

Kept. It avoids eager all-triple enumeration and is differential-tested, but
its current `U/W` complexity must be stated honestly. A neighbor/event-queue
refinement is research, not assumed.

### Compact transfer atlas

Kept conditionally. Route per chart. Short words/high required rank may be
cheaper under exact streamed replay. Rank must cover both primal and required
derivative actions.

### Full differentiation through compiler decisions

Deferred. The stable-stratum VJP is useful now, but a total derivative claim is
blocked on trust regions, event semantics, and schedule/rank differentiation.

### Global `FJ` weight cache

Deferred as an optional execution trade. The block-first path is bounded in
memory; the cache may reduce repeated construction only when measured bytes and
ordering justify it.

## Falsification and promotion tests

Cheap CPU tests that must remain green:

1. active compiler versus exhaustive compiler and independent oracle;
2. inactive full-fiber tie fails closed;
3. rational and irrational event grouping/right-continuous dispatch;
4. multi-chart sampled forward parity;
5. continuous primal and material-action certificate acceptance/failure;
6. finite differences for stable-stratum site, velocity, weight, ray, density,
   and RGB gradients;
7. tiny-optical-depth `expm1` parity;
8. `F` variation changes sample reduction only, not world reverse interactions
   or stored structural state;
9. native prepared tokens own zero global/chart time-clone bytes; and
10. `B_p`/`K` changes preserve loss and gradients.

Native/runtime promotion, only in an approved quiet window, additionally needs:

- rebuild/import/source parity;
- exact multi-chart kinetic lowering;
- measured allocator peak and bytes moved, not logical tensor volume alone;
- realistic high-run `F/R/J` death curves against exact replay;
- dataset-bound initialization, ragged sampler, evaluator, checkpoint, and
  artifact integration; and
- frozen-checkpoint and then geometry-training evidence under a declared
  recertification policy.

Kill or narrow the broad systems claim if realistic charts mostly choose exact
fallback, accepted `J` grows with requested sample density rather than physical
complexity/tolerance, topology invalidates after nearly every useful geometry
step, or native world reverse interactions/bytes scale linearly with `F` after
subtracting target/output work.

## Current decision

No new Gaussian-like formula is presently required. The necessary structural
formulation now exists at CPU/reference scope:

```text
active kinetic owner charts
+ exact ordered affine transfer
+ certified compact temporal evaluation
+ streamed residual-to-node reduction
+ frozen-program sparse world reverse.
```

The next mathematical work is narrower: sharpen active closure/maintenance,
add bounded-cell event predicates, and derive a structural trust-region or
event-derivative contract. The next engineering work is native multi-chart
lowering and trainer/evaluator integration. Publication-scale runs should wait
for those gates and an approved machine.

## Verification completed

The final CPU-only scopes passed after Ruff formatting:

```text
kinetic compiler/dispatch/certificate/VJP suite     75 passed
native adapter/material/host-memory/source suite    54 passed, 11 subtests
Ruff check                                           passed
Ruff format check                                    30 files formatted/clean
git diff --check                                     passed
```

No MPS/Metal runtime, native build, CUDA, dataset decode, or training workload
was run.

## Open questions

1. Can a certified neighbor supergraph reduce all-site closure without losing
   exactness under kinetic updates?
2. What is the best output-sensitive bound for cumulative root-complement
   discoveries `W`?
3. Which bounded PowerFoam sphere/vacuum events can share the existing
   polynomial isolator?
4. What practical trust-region bound follows from current predicate margins?
5. Can continuous geometry-action error be certified sparsely without a dense
   global dual tensor?
6. Where is the measured exact-versus-compiled break-even on realistic
   high-depth public scenes?
