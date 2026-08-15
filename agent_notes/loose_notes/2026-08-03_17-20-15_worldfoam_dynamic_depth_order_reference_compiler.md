# WorldFoam dynamic depth-order reference compiler and handoff hardening

Date: 2026-08-03

## Question

The motivating distinction was:

- World Tubes can eliminate Gaussian depth analytically with a Schur
  complement because the Gaussian family is closed under marginalization.
- WorldFoam must preserve ordered depth transfer; an order-blind marginal would
  erase the differently colored overlap phenomenon.

The task was to reformulate the remaining WorldFoam problem, audit nearby
literature, prepare a rigorous mathematician handoff, and advance the exact
CPU proof route without another large MPS/training workload.

## Decision

No second Schur complement is the leading answer. The structural object is a
kinetic lower envelope of ray-restricted power lines; the transfer object is
the ordered affine emission--absorption semigroup. The production target is:

```text
continuous kinetic owner charts independent of requested F
-> exact ordered P0 transfer at adaptive J nodes
-> compact Lie sample evaluation/reduction
-> one node-level prefix-only material/world reverse
```

Writing/comparing `P*F` colors remains `Omega(PF)`. The claim is only that
topology, word traversal, world VJP work, and reverse interaction state depend
on physical events/runs/rank rather than requested temporal sampling density.

Static Radiant/Power Foam is fast because it walks sparse cell adjacency,
integrates one crossed segment exactly, culls spatially, and terminates early.
It does not provide a dynamic topology compiler or shared cross-time adjoint.
Kinetic regular triangulations and spacetime BVHs are useful precedents, not a
solution to ordered neural-foam transfer and training.

## Event theorem

For affine kinetic sites/rays, each pair gap is

```text
Delta_ij(t,z) = A_ij(t) z + B_ij(t),
deg A_ij, deg B_ij <= 2.
```

On finite `[near,far]`, the universal small-world seam set is:

1. all pair equalities at `near` and `far`;
2. all finite triple-concurrence roots, degree at most four;
3. time-domain endpoints; and
4. separate ray-collapse roots `||d(t)||^2=0`.

An isolated full-fiber tie `A=B=0` is already a common near/far root. A root of
`A` with `B!=0` is only an analytic rational-cut guard; it cannot change the
owner word locally on a compact depth interval.

The key completeness fact is that a competitor-owner difference is affine in
depth. It cannot first become negative strictly inside an owner segment while
remaining nonnegative at both segment endpoints. Its first contact is therefore
at near/far, at an existing active cut, or on the full fiber.

The production reduction checks current endpoint owners against all sites and
every active cut against every competitor. For cut `i-j` and competitor `k`,

```text
H_(k|ij) = B_ik A_ij - A_ik B_ij,
H_(k|ij) / A_ij <= 0,
equivalently H_(k|ij) A_ij <= 0.
```

This is `O(SR)` per active chart, not fully output-sensitive. Replacing all
sites by `delta` neighbors requires a separately certified kinetic regular/
Delaunay or conservative Cech-style supergraph and its maintenance cost.

## CPU implementation completed

New exact reference/oracle files:

- `research_experiments/world_foam_lane2/kinetic_owner_chart_compiler.py`
- `research_experiments/world_foam_lane2/test_kinetic_owner_chart_compiler.py`
- `research_experiments/world_foam_lane2/kinetic_owner_chart_oracle.py`
- `research_experiments/world_foam_lane2/test_kinetic_owner_chart_oracle.py`

The reference compiler exhaustively enumerates all pair near/far and finite
triple candidates, keeps denominator roots as analytic-only guards, retains
polynomials plus rational isolators, separates close roots, GCD-groups equal
algebraic roots, evaluates exact one-sided owner words, checks every owner
against every site, merges inactive guards, and emits right-continuous
half-open charts. Full-fiber ties, ray collapse, persistent unsupported strata,
and simultaneous active events fail closed with no partial coverage.

The independent oracle intentionally avoids the production hull/filtering. It
uses all pair cuts for fixed-time words and a separate global-product Sturm
isolator for small adversarial worlds. It covers inactive concurrence,
denominator artifacts, close/simultaneous roots, zero-length birth/death,
full-fiber ties, ray collapse, and random fixed-time parity.

The oracle exposed a production root bug: making every negated Sturm remainder
monic may multiply it by a negative scalar and corrupt sign variations.
`rational_polynomial_roots.py` now normalizes Sturm members only by positive
scalars and has a rootless `x^2+1` regression.

New CPU transfer bridge:

- `research_experiments/world_foam_lane2/kinetic_chart_transfer_bridge.py`
- `research_experiments/world_foam_lane2/test_kinetic_chart_transfer_bridge.py`

For one safe chart it discovers exact rational words/cuts at fixed `J` nodes,
computes float64 physical lengths and P0 Beer transfer, reuses the compact
barycentric Lie schedule for blocked sample-to-node reduction, and applies one
prefix-only density/color VJP over node words. Structural and reverse state are
independent of requested `F`.

The bridge deliberately omits kinetic geometry/ray/weight/event-time VJPs,
multi-chart seam dispatch, native lowering, and coverage inside irrational
root-isolator neighborhoods. It recompiles once to bind a chart because the
owner program does not yet carry an immutable source digest.

## Handoff red-team corrections

The mathematician prompt and theorem ledger were corrected before sharing:

- the exponential family rules out a universal fixed linear/polynomial atlas,
  not every finite-dimensional nonlinear representation;
- a global chart refinement is semantic only; storage must be ragged over
  `(track,local_chart)` and must never materialize `P*C`;
- persistent topology/binding/token state is a `sum_b` term, while only live
  staging/scratch belongs under `max_b`;
- the current material trainer constructs sample weights per spatial block,
  so its term is `N_B*F*J`; `F*J` needs a future validated global cache;
- the executable kinetic frontend is Euclidean `A=I`, affine positions, and
  degree-at-most-two weights: nine structural plus four P0 material scalars;
- a finite trust radius needs derivative bounds over the entire parameter
  segment/ball, not only at the current point;
- current kinetic completeness covers an unbounded power partition clipped by
  global near/far, not Power Foam's moving controlling spheres;
- the rotating-face lower bound is on active fixed-normal pieces/chart
  switches in its stated approximation class, not arbitrary site count;
- physical, frozen-program, and adaptive compiled-algorithm derivatives are
  distinct; and
- cited planar kinetic-Delaunay complexity is cautionary prior art, not a bound
  for the present weighted 3D per-ray problem.

The handoff also discloses that the kinetic/root/compiler/oracle files are
dirty or untracked. External reproducibility needs a deliberate commit or
content-hash manifest.

## Validation

Host-safe CPU-only gate:

```text
83 passed
```

It covers rational roots, fixed-time kinetic words, continuous reference
charts, the independent oracle, affine-transfer Lie math, compact schedules,
and the kinetic material bridge. Ruff lint passed; the two pre-existing root
files were mechanically formatted. No MPS, Metal build, CUDA, W&B, dataset
decode, or training workload ran.

## Remaining production work

1. Implement the proved `O(SR)` active-boundary sweep and compare it against
   the exhaustive oracle on randomized small worlds.
2. Add an immutable source/program digest so the transfer bridge does not
   repeat `O(S^3)` binding compilation.
3. Implement sparse kinetic position/velocity/weight/ray VJPs inside certified
   regular strata.
4. Add explicit persistent/full-fiber/simultaneous seam policy or exact replay
   fallback.
5. Lower ragged multi-chart programs to the bounded native lifecycle.
6. Add moving sphere/vacuum events only if bounded Power-Foam parity is still a
   required representation claim.
7. Rebuild and measure native allocator/bandwidth behavior only in an approved
   quiet window; publication-scale training remains off this host.

This remains a WorldFoam second-paper lane and must not displace the primary
World Tubes public experiments and paper packaging.
