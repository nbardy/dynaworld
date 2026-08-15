# WorldFoam multichart simple-root re-isolation

Date: 2026-08-03

Status: CPU/reference theorem implementation. This proves whole-direction
acceptance and exact endpoint root re-isolation on a restricted generic
stratum. It does **not** yet implement output-sensitive native chart-payload
repair, warm optimizer integration, or derivatives through event times.

## Context

The event-free geometry trust certificate proves one owner word over an exact
directional update, but correctly returns radius zero for a multichart program:
old numeric event endpoints generally move after any geometry update. The next
question was whether separated simple roots can be continued and re-isolated
without treating every optimizer candidate as unrelated geometry.

Files inspected before this implementation:

- `kinetic_power_word_compiler.py` and its tests;
- `kinetic_active_owner_chart_compiler.py` and its tests;
- `rational_polynomial_roots.py` and its tests;
- `kinetic_geometry_trust_region.py` and its tests; and
- the exact event-sufficiency red-team note dated `2026-08-03_16-35-33`.

Implemented:

- `research_experiments/world_foam_lane2/kinetic_simple_root_reisolation.py`
- `research_experiments/world_foam_lane2/test_kinetic_simple_root_reisolation.py`

The existing event-free trust module now exposes its exact bivariate
polynomial, directional-geometry, and Bernstein-conversion primitives. The new
module reuses them rather than maintaining a second algebra.

## Current decision

Current belief: a useful restricted theorem is true.

Confidence: high for the stated exact certificate, medium for whether it will
beat complete recompilation after production packing and allocator costs.

The accepted stratum is intentionally narrow:

1. one exact segment
   `theta(eta)=theta_0+eta*(theta_1-theta_0)`, `eta in [0,1]`;
2. both endpoints are interpreted as exact values of stored binary64 tensors;
3. the base active-owner program is continuously certified and multichart;
4. every root in the reconstructed registry is interior, simple, singleton,
   and separated from every other root;
5. no predicate is persistently zero;
6. ray speed stays strictly positive on the complete `(t,eta)` rectangle;
7. every denominator root has an intercept companion that stays nonzero in
   its root tube, excluding a full-fiber tie; and
8. exact endpoint owner-word classification agrees before and after the
   update.

Repeated, shared, simultaneous, persistent-zero, endpoint, ray-collapse, and
semantically ambiguous cases fail closed to the full exact compiler.

## The provenance correction

The existing `ActiveKineticOwnerChartProgram` retains roots and their sources,
but it does not retain the candidate polynomials that were rootless at the base
geometry. Reusing only emitted root records is unsound.

Minimal counterexample:

```text
P(t,eta) = t^2 + epsilon - 2 epsilon eta.
```

At `eta=0`, `P` has no real root. For `eta>1/2`, it has two. No amount of
continuing the empty base root list can discover them.

The implementation therefore reconstructs the complete active-boundary
registry from every distinct base owner word, including rootless sources.
This is the load-bearing correction.

## Canonical predicate registry

For every distinct certified base word, the implementation reconstructs:

### Class I: topology-event candidates

- near owner versus every site: `N_ij=B_ij+near*A_ij`;
- far owner versus every site: `F_ij=B_ij+far*A_ij`; and
- every active cut versus every third-site competitor:
  `H_k|ij=B_ik*A_ij-A_ik*B_ij`, represented by the canonical sorted-triple
  polynomial.

These have bidegree at most `(2,2)` for pairs and `(4,4)` for triples under the
linear geometry homotopy. Their algebraic roots are only candidates. A root is
counted in semantic event count `E` exactly when the certified left/right
positive-length owner words differ.

### Class II: analytic/representation guards

- every relevant pair denominator `A_ij`.

At an `A_ij` root, the corresponding `B_ij` is certified strictly nonzero on
the complete root tube. Thus the event is a cut-at-infinity guard rather than
a full-fiber tie. It never increments `E`. A representation split is reported
only when the pair is actually adjacent in a certified left/right word; a
denominator root for an inactive pair is retained as proof evidence but does
not force a chart.

### Class III: non-root validity guards

- `Q_d(t,eta)=||d(t,eta)||^2>0` on the whole update rectangle.

This is a uniform positivity certificate, not an event root list.

## Restricted persistence theorem

Let `P_1,...,P_K` be the complete Class-I/II registry. At `eta=0`, isolate all
distinct roots exactly with rational Sturm isolation. Suppose each root is
simple and belongs to exactly one source. Assign disjoint rational tubes

```text
I_m x [0,1],  I_m=[a_m,b_m]
```

with strict separation `b_m<a_{m+1}`. Certify:

```text
sign(dP_m/dt) is fixed and nonzero on I_m x [0,1],
sign(P_m(a_m,eta)) = -sign(P_m(b_m,eta)) for every eta,
P_k has a fixed nonzero sign outside its assigned tubes,
Q_d > 0 everywhere,
B_ij != 0 on every A_ij-root tube.
```

Then:

1. the intermediate value theorem gives at least one root of `P_m` in every
   tube for each `eta`;
2. fixed nonzero `dP_m/dt` gives at most one, hence exactly one;
3. fixed signs on complementary strips exclude root birth or escape;
4. disjoint fixed tubes preserve root order and exclude collisions;
5. the full-fiber companion check excludes `A_ij=B_ij=0`;
6. the active-boundary first-contact theorem says an owner word cannot change
   away from a Class-I root; and
7. because no second source can vanish in a root tube, activity/co-minimality
   cannot silently change there. Exact left/right word replay at `eta=1`
   confirms the semantic classification and fails closed on disagreement.

Therefore the same ordered sequence of semantic owner words persists along
the accepted update. Every endpoint root can be re-isolated exactly inside its
old rational tube. This is a root-graph persistence theorem, not a derivative
theorem.

## What the theorem does not prove

- It does not accept a norm ball around `theta_0`.
- It does not handle repeated, grazing, shared, or simultaneous roots.
- It does not handle roots on the time-domain endpoints.
- It does not prove bounded PowerFoam sphere/vacuum event completeness.
- It does not differentiate event times, chart boundaries, selected ranks,
  or compiler decisions.
- It does not update native CSR words, node lengths, interpolation schedules,
  or source maps in place.
- It does not prove that local maintenance is faster than full compilation.

Stable-stratum geometry gradients remain the only certified geometry-gradient
claim. Accepted root movement only authorizes structural reuse/re-isolation;
it adds no event-time term to the rendering objective.

## Exact work and storage bound

Let:

```text
U = number of distinct base owner words,
S = site count,
R = maximum active word depth,
K = number of unique reconstructed root-bearing sources,
M = number of distinct isolated base roots,
D = Bernstein subdivision-depth cap.
```

Registry construction uses `O(U*S*R)` predicate attempts and `O(K)` retained
source polynomials. Base exact root isolation costs `K` degree-at-most-four
Sturm calls plus rational bit complexity. Overlap separation uses at most the
configured `B` refinement rounds, with sorting `O(B*M log M)` and exact GCD
checks only for overlapping neighbors.

The current reference checks `O(K+M)` source strips/tubes. In the conservative
worst case, recursive tensor-Bernstein certification visits `O(2^D)` leaves
per check; degrees are bounded by four in each variable. Candidate endpoint
re-isolation uses `M` bounded low-degree Sturm calls. Semantic reclassification
uses `M+1` exact fixed-time lower-envelope discoveries at `O(S log S)` work and
`O(S)` scratch each. The continuous all-site guarantee comes from the
separately reconstructed active-boundary registry and strip proof, not from
those endpoint witnesses.

Storage is `O(K+M+U*R)` exact host objects and has no requested-frame axis.

This is **not yet output-sensitive local payload repair**. It certifies a whole
direction, then re-isolates the endpoint roots and owner words. The source
registry and proof are rebuilt on each current API call. A warm implementation
would need sealed source provenance, affected-source incidence, and bounded
chart-payload patching before claiming a lower update cost than recompilation.

## Behavioral falsification results

The focused exact tests cover:

1. rational roots moving from `(-1,+1)` to `(-5/4,+3/4)`;
2. one irrational quadratic root retained as polynomial plus rational interval;
3. a moving active triple/third-site insertion event;
4. two simple roots separated by only `1/1024`;
5. a denominator-only root that is not counted as topology;
6. a root born from a previously rootless active-cut competitor polynomial;
7. repeated/grazing rejection;
8. two inactive predicates sharing one exact algebraic root;
9. ray collapse inside the update segment;
10. an endpoint event; and
11. a step inside versus outside the fixed root tubes; and
12. twelve fixed-seed dyadic perturbations across pair and interior-triple
    fixtures. Every accepted candidate is freshly recompiled and must match
    the certificate's semantic chart-word sequence and active-event count;
    larger candidates also exercise fail-closed rejection.

The root-birth fixture is the decisive soundness test. If it ever passes, the
local-maintenance route is invalid because the complete registry or complement
proof has been weakened.

## Branches and backtracks

### Branch A: root-record-only maintenance

Status: invalidated.

Reason: a rootless base predicate can acquire roots. Old event records are not
a complete update certificate.

### Branch B: generic simple-root continuation with complete sources

Status: supported on the restricted exact directional stratum.

Could be wrong if the active-boundary first-contact registry omits an event
source. The cheap falsifier is differential comparison against the exhaustive
compiler on small random accepted updates.

### Branch C: local repair is automatically faster

Status: unresolved.

The current reference reconstructs the whole `O(U*S*R)` registry and performs
global exact strip proofs. Full recompilation remains the simpler and safer
fallback. Local maintenance should be promoted only after a sealed-provenance
prototype measures fewer predicate/root operations on realistic updates.

## Decision implication

Keep this certificate as the exact gate for the generic simple-root stratum.
Do not broaden it to simultaneous or repeated events yet. The next singular
engineering slice, if pursued, is an affected-source incidence index plus a
candidate chart-payload patcher that consumes this certificate. Kill local
maintenance if accepted realistic updates do not reduce exact predicate/root
work relative to `compile_active_kinetic_owner_charts`; in that case retain
the full compiler and use the certificate only as a mathematical audit.
