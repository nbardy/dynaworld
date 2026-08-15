# WorldFoam Paper Packet

Status: second-paper lane, opened 2026-07-05.

This folder separates the **WorldFoam** paper from the **World Tubes in Gauged
Camera Space** paper.

The split is deliberate:

- **World Tubes** is the baseline-compatible camera-path compiler for dynamic
  Gaussian primitives. It preserves Gaussian-splat semantics, compiles
  support/visibility/adjoints through a known camera program, and compares most
  directly against per-frame replay of the same representation.
- **WorldFoam** is the lifted opacity/transmittance paper. It treats matter as
  bounded world cells or foam regions, pulls those cells through the camera-ray
  bundle, and renders by cumulative optical depth along ray fibers. It is a
  cleaner visibility model, but it is less mature as a quality/SOTA claim.

Use this folder when the question is:

```text
Can we replace discrete splat sorting with a camera-gauged ray-fiber opacity
field whose dominant cell/intersection work is reused across time?
```

Use `research_notes/gauged_uvt_trace_atlas/paper/` when the question is:

```text
Can dynamic Gaussian splats be compiled into reusable world-tube traces while
remaining compatible with baseline Gaussian-splat alpha compositing?
```

## Files

- `WORLDFOAM_HANDOFF.md` - historical 2026-07-05 clean-thread handoff. Its
  chronology remains useful, but its current-status/next-work sections are
  superseded by the 2026-08-03 theorem ledger and memory-light TODO.
- `WORLD_FOAM_PAPER_DRAFT.md` - arXiv-style draft skeleton for the second
  paper, now subtitled **Gauge-Invariant Ordered Ray Transfer for Moving
  Cameras**. "Ray holonomy" remains useful informal intuition, while the draft
  reserves holonomy for closed cell-complex loops.
- `WORLD_FOAM_ICLR_MAIN_DRAFT.md` - concise anonymous venue-facing manuscript.
  Its accepted numerical statements are limited to the verified CPU
  foundation evidence until G4 public quality and G6 native memory exist.
- `generated/foundation_v1/` - deterministic fail-closed Paper-B foundation
  bundle. It currently contains `14` accepted foundation rows and explicit
  `not measured` placeholders for G4 and G6; `complete=false` is intentional.
- `WORLD_FOAM_EXPERIMENT_PLAN.md` - baselines, ablations, charts, acceptance
  gates, and honest claim boundaries.
- `WORLD_FOAM_MATH_APPENDIX.md` - polished math appendix scaffold: optical
  measures, ray-fiber pullback, visibility monoid, product integral, cell-path
  rasterization, replay theorem, VJP, commutator, and validation gates.
- `WORLD_FOAM_DYNAMIC_DEPTH_ORDER_MATHEMATICIAN_PROMPT.md` - paste-ready
  external-mathematician handoff for the remaining dynamic-foam closure. It now
  opens with the canonical reformulation: preserve noncommuting depth order,
  quotient each stable word into exact affine transfer, compile its continuous
  event/rank structure, and reduce streamed residuals before one shared world
  VJP. A literature-boundary table separates static foam speed, kinetic
  geometry, spacetime acceleration, dynamic-field compression, and associative
  compositing from the still-missing combined theorem. It then
  treats the active-owner CPU compiler, multi-chart transfer, continuous
  material-action certificate, frozen-program sparse VJP, bounded actual-rank
  lowering, fake-native lifecycle adapter, ragged paper sampler, union-local
  bar assembly, block-major material step, global-denominator coordinator, and
  narrow event-free directional trust certificate as landed at CPU/source
  scope. A restricted multichart singleton-simple-root certificate is landed
  too; the first work order now asks an external mathematician to audit or
  falsify it and derive output-sensitive affected-chart repair rather than
  invent another depth marginal.
- `WORLD_FOAM_MEMORY_LIGHT_THEOREM_LEDGER_2026-08-03.md` - theorem,
  counterexample, complexity, and derivative ledger for the memory-light path.
  It records the honest `U/W` active-closure bound and separates a
  frozen-program VJP from differentiation through recompilation.
- `WORLD_FOAM_STRATIFIED_LAGRANGIAN_CONNECTION_AUDIT_2026-08-05.md` - audited
  derivation of the latest fiber/connection/curvature/cosheaf/jet/monodromy
  scientist proposal. It repairs the transfer-order convention, proves the
  constrained-flow subinterval-reuse theorem, records the total-transfer and
  scalar-flow counterexamples, corrects the P0 boundary measure and seam
  theorem, and specifies the three-way `U`/`U_tilde`/`K_F` kill gate. Curvature
  is a theorem and candidate diagnostic, not a promoted runtime.
- `WORLD_FOAM_MEASURE_CONNECTION_SYNTHESIS_2026-08-05.md` - canonical unified
  math spine for the translated optical-depth measure, four-scalar affine
  quotient, and constrained Lagrangian temporal connection. It adds the exact
  measure/connection implication chain, full sensor-depth lift, separates the
  physical `U`, group-completion `U_tilde`, and signed-tangent `K_F` ABIs, and
  fixes the oracle, work accounting, claim ladder, and promotion stop rules.
- `WORLD_FOAM_UNION_LOCAL_FUSED_GEOMETRY_V2_DESIGN_2026-08-05.md` - exact
  compact-to-union-to-global cotangent factorization, complete bridge/map byte
  formulas, three-index-space ABI contract, fail-atomic transaction design,
  source implementation map, and falsification gates for the first fused
  memory-v2 candidate.
- `WORLD_FOAM_NATIVE_MEMORY_SOURCE_AUDIT_2026-08-03.md` - exact `F/K/B_p/R/J/W/S`
  tensor accounting for the equal-rank/ragged native path, the new
  material-only no-`[J,W]` VJP ABI, and the source/runtime audit that exposed
  the step-lifecycle requirement. A separate CPU/fake-native block-major gate
  now closes that lifecycle at integration-proof scope; rebuilt-native launch
  frequency and allocator telemetry remain open.
- `WORLD_FOAM_OPTICAL_TRANSFER_PAPER_PLAN.md` - current best plan after the
  reformulation pass: promote visibility monoid, optical-transfer event
  elements, commutator theorem, monoid VJP, and event closure; keep Magnus and
  boundary flux behind tests.
- `GAUSSIAN_FINITE_ELEMENT_WORLD_FOAM.md` - complete native 4D
  finite-element representation proposal: cell choices, P0/P1/P2 degrees of
  freedom, exact log-quadratic ray integrals and VJPs, continuity and
  approximation bounds, camera-gauge compatibility, and the mandatory
  positive-Bernstein counterbaseline.
- `SELF_NORMALIZED_CONVEX_ATOM_AND_RAY_TRANSFER_AUDIT.md` - critical audit of
  the second external dump. Separates existing ordered-transfer WorldFoam math
  from the distinct self-normalized strongly-convex atom, proves its slice,
  ridge, ray-support, and tangency properties, and records minimizer, overlap,
  event, terminology, and prior-art failures.
- `PAPER_METHOD_CLASSIFICATION_AND_METAL_GATES.md` - publication decision and
  implementation ladder. Classifies what belongs to World Tubes, WorldFoam, a
  material extension, or an incubating new primitive, then specifies the
  controlled P0/P1/P2/log/convex Metal matrix and paper acceptance gates.
- `../../research_experiments/world_foam_lane2/finite_element_material_transfer.py`
  and neighboring `.metal`, wrapper, tests, and gate runner - implemented
  M0--M5 fixed-segment foundation. The schema-rich tiny parity artifact is
  `../../artifacts/foundation_gates/worldfoam_material_m0_m5_cpu_metal_20260727.json`;
  it is numerical evidence, not trained quality or renderer speed.
- `experiment_designs/cell_path_optical_transfer_fixture.md` - code-level plan
  for the first falsifiable fixture: exact optical-transfer monoid scan,
  same-representation cell-path replay, finite-difference VJP, commutator
  probe, artifact schema, and pytest command.
- `scientist_notes/` - triaged external-model/scientist dumps. Use this for
  incoming scratchpads so good ideas become searchable notes instead of chat
  sediment. The current reformulation triage is
  `scientist_notes/2026-07-05_optical_transfer_reformulation_intake.md`. The
  moving-camera ray-holonomy intake and paper-split decision is
  `scientist_notes/2026-07-26_gauge_invariant_ray_holonomy_intake_and_paper_split.md`;
  it preserves the self-normalized convex-potential atom proposal,
  discriminant-certified fiber compiler, Duhamel VJP, complexity claims, and
  the boundary between STAR UVT Paper A and retained-transfer Paper B.
- `proofs/depth_fiber_operator_ordering.md` - cleaned theorem scaffold for the
  depth-fiber/operator-ordering story: gauge-invariant traces, non-commutation,
  lifted opacity, transmittance prefixes, same-representation replay, and
  fixed-topology VJP boundaries.

Related cross-track note:

```text
../gauged_uvt_trace_atlas/DEPTH_FIBER_CROSS_TRACK_NOTE.md
```

## Expansion Map

The folder is intentionally small but ready to expand:

```text
scientist_notes/      incoming external-model dumps and triage
proofs/               gauge invariance, replay equivalence, error bounds, scaling
experiment_designs/   active experiment specs before code
figures/              figure and table plans
paper/                eventual LaTeX/arXiv manuscript sources
```

Create optional subfolders only when they get their first real file. `proofs/`
now exists because the depth-fiber/operator-ordering theorem scaffold is a real
paper object, and `experiment_designs/` now exists because the cell-path
optical-transfer fixture has a concrete code/test plan.

## Current Claim Boundary

WorldFoam currently has real local evidence:

```text
Metal Gate4/native-cutwalk speed gates
sublinear-ish frame-count timing in focused microgates
trainable one-step real32 loader smoke
bounded-cell / Cech-AABB / raytrace implementation artifacts
```

It does **not** yet have the full evidence needed for a strong quality paper:

```text
public dynamic-scene benchmark quality parity
official CUDA/Warp parity fixture acceptance
DeepView-quality acceptance above the current low-PSNR/low-SSIM rows
clean evidence that foam transmittance beats STAR/GS on real RGB quality
```

So the right near-term paper shape is either:

```text
theory + prototype + speed/scale paper
```

or:

```text
full rendering paper after the quality/parity gates clear
```

Do not blend the stronger World Tubes compiled-adjoint evidence into WorldFoam
as if it proves WorldFoam quality.

The current CPU/source systems claim is narrower but substantial: expensive
ordered-word forward/reverse work can be independent of requested frame density
at fixed physical interval, camera program, world, charts, rank, and tolerance.
The remaining sample/output stream is necessarily `O(PF)`. The active kinetic
compiler, exact multi-chart dispatch, continuous material-action certificate,
frozen-program position/velocity/weight/ray/material VJP, bounded actual-rank
native-shaped lowering, ragged observation/gradient coordination, a
block-major one-forward/one-material-VJP paper step, and one event-free
directional trust certificate are green on CPU/source paths. A separate exact
CPU reference now certifies whole-direction persistence and endpoint
re-isolation for separated singleton simple roots, including rooted/rootless
predicate reconstruction and semantic reclassification. The native runtime,
bounded PowerFoam sphere/vacuum events, warm/output-sensitive affected-chart
repair, total event/compiler derivatives, allocator measurements, and public
training evidence are not. Prepared native tokens no longer retain
global or chart-local frame-time arrays; only the live `K`-sample block is
staged. The CPU/fake-native integration proves the required
spatial-bundle-outer lifecycle: `K=1/4` agrees with a direct-autograd oracle and
`F=5/41` leaves compiled node-forward/material-VJP counts and retained runtime
bytes fixed. The material path allocates no `[J,W]` bar, and sequential spatial
bundles peak at the largest bundle rather than their sum. Native Metal build,
launch-frequency telemetry, and allocator peaks are still unverified.
