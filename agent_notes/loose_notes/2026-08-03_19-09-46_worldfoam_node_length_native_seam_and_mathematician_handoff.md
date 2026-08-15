# WorldFoam node-length native seam and mathematician handoff

Date: 2026-08-03

## Question

The user asked for a precise reformulation of dynamic WorldFoam: why static
foams are fast, why Gaussian Schur marginalization does not transfer, what
literature is relevant, and how to direct a strong mathematician toward a
memory-light retained-depth solution. The target remains World-Tubes-shaped
systems scaling: expensive world traversal and reverse work must be shared over
requested time density, while cheap camera/sample interpolation may remain
linear and RGB output has its unavoidable `Omega(PF)` cost.

## Selected formulation

Do not eliminate depth order. Compile it.

For every certified `(track, chart)` and compiler node, retain:

```text
ordered owner word o[0:R]
positive physical segment lengths length[J,R]
```

Then compute exact P0 ordered transfer at the `J` nodes and encode the result in
the affine-transfer Lie chart `(kappa,v)`. Samples use only row-local temporal
weights. Reverse mode first reduces all sample residuals to node Lie bars,
scans each ordered word once per node to produce material and physical-length
bars, and maps the length bars through one frozen-stratum geometry VJP.

This is analogous to the role of the World Tubes Schur closure at the systems
boundary, but it is not a Schur complement. It removes the within-cell depth
coordinate only after preserving the full noncommutative owner word and its
physical lengths. Consequently it preserves changing colored overlap order.

For direct affine kinetic sites and affine rays, the general face cut is
quadratic-over-quadratic in time, not the legacy static-site Mobius form. The
honest route is therefore to compile certified physical lengths outside the
sample loop rather than pretending the old boundary ABI applies.

## Landed CPU/source seam

- `kinetic_native_topology_lowering.py` lowers a certified single-ray kinetic
  chart to compact CSR owners and positive `[J,R]` node lengths. Its structural
  bytes and provenance are independent of requested frame count.
- `kinetic_native_precompiled_length_oracle.py` is an independent CPU
  affine-Lie forward/VJP oracle.
- `kinetic_native_precompiled_length_adapter.py` binds the exact source-only
  native forward/VJP ABI, persistent `[track,J,compact-site,R]` configuration,
  compact/global material scatter, and bounded `[J,R]` length bars.
- `kinetic_stable_stratum_vjp.py` maps node physical-length bars to positions,
  velocities, quadratic weights, and affine-ray bars once per node under an
  explicit continuous-topology provenance contract.
- The native source now includes a row-ragged Lie sample reducer. It consumes
  `[row,J,4]` charts, selected row ids `[N]`, row-local `[N,J]` weights, and
  `[N,3]` targets. It does not build a row-by-global-time table; the loss-only
  path allocates no prediction tensor.
- `src/train/paper_ragged_track_staging.py` groups arbitrary paper observations
  by view, preserves original batch positions and one global loss denominator,
  and stages targets one selected frame at a time without a view/time Cartesian
  product.
- Native sample ledgers were audited to avoid retaining expected/consumed
  block collections proportional to `F/K`; they retain constant-size cursor and
  count state while generating deterministic ranges lazily.

The CPU oracle originally computed `kappa=-log(product beta)`. Red-team tests
found that this rounded tiny optical depth to zero and failed once the beta
product underflowed. It now matches Metal and computes
`kappa=sum_r density_r*length_r` directly. Parametrized tests cover optical
depths from `1e-18` to `1e4` in forward and VJP.

## Verification

- all `test_kinetic*.py`: `87 passed`;
- native source verifier pytest: `25 passed`, plus `11` verifier subtests;
- focused lowering/oracle/adapter/geometry/ragged gate: `27 passed`;
- source inventory: `123` schemas, `123` implementations, `110` initialized
  kernels, zero verifier failures;
- Ruff format/check and diff checks: clean.

No extension rebuild, MPS/Metal execution, CUDA job, dataset decode, or training
run occurred. All native results remain explicitly `source_only/
runtime_unverified`.

## Remaining production work

1. Rebuild the extension in a resource-approved window and run tightly bounded
   Metal forward/VJP/ragged parity, including tiny/high optical depth.
2. Replace the single-ray reference lowerer with an output-sensitive batched
   per-pixel compiler. Bucket or stream equal `J`; do not pad to global
   `J_max` or form a global common time refinement.
3. Add the outer coordinator that accumulates all view-local compact material
   bars under the global denominator and performs exactly one optimizer update.
4. Add dataset-bound initialization, session lifecycle, checkpoint/evaluation,
   streamed media, allocator accounting, and the distinct
   `worldfoam_native4d` runner lane.
5. Derive topology-maintenance trust regions/local recertification for geometry
   optimizer steps. The frozen-program VJP is not the derivative through a
   changing compiler.
6. Extend the currently unbounded near/far-clipped power partition to bounded
   Power-Foam sphere/vacuum events only if the second-paper route justifies it.
7. Run fixed-duration `F` scaling and exact-replay parity before any public
   quality claim.

## Research handoff

`research_notes/worldfoam_paper/WORLD_FOAM_DYNAMIC_DEPTH_ORDER_MATHEMATICIAN_PROMPT.md`
is the paste-ready external prompt. It now treats the numeric node-length seam
as landed and directs theory toward the remaining problems: a sharper
output-sensitive kinetic compiler, structural trust regions, event/physical
derivatives, bounded-cell events, lower bounds, and exact-replay kill rules.

This remains a WorldFoam second-paper/future lane. It must not displace the
publication experiments and packaging needed to finish the primary World Tubes
paper.
