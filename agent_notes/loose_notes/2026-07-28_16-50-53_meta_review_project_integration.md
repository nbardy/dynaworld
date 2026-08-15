# Meta-review reconciliation: what changes, what does not, and how to integrate it

## Context

This note reconciles:

- `research_notes/meta_review_jul_28th.md`;
- the current World Tubes / STAR UVT code and paper;
- the native SPD(4), retained-fiber, and WorldFoam lanes;
- the frozen identical-world causal experiment;
- the broader DynaWorld world-token and novel-view training plan.

The repository is already materially ahead of the state assumed by parts of the
review. In particular, it already contains:

- a four-layer paper decomposition into world, camera compiler, evaluator, and
  compiled adjoint;
- projective interval trace atlases;
- support, order, UV-visibility, and sensor-time event machinery;
- interval Metal forward and direct VJP paths;
- orbit, exposure, rolling-shutter, visibility, and frame-density fixtures;
- a native full-SPD(4) source/compiler oracle and production route;
- a frozen-world replay-versus-compiled executor;
- a separate WorldFoam retained-depth/material lane.

Therefore the review does **not** justify a representation pivot. Its main value
is to sharpen the scientific claim and expose the remaining gaps between an
implemented interval renderer and a defensible camera-program compiler paper.

## Executive decision

Current belief:

> Keep World Tubes / STAR as the primary paper. Define the contribution as
> camera-program compilation plus a shared world adjoint, not as a new Gaussian
> primitive. Treat native SPD(4) as a parity/conditioning axis, WorldFoam as a
> separate retained-transfer stress backend, and pentatopes or swept-volume
> atoms as conditional reference/representation lanes.

Confidence: high.

What changes:

1. The next paper gate becomes stricter and more causal.
2. Camera-pushforward complexity, structural invalidation, and event-boundary
   derivatives become named quantities rather than vague limitations.
3. The current fixed-topology direct VJP must be described precisely; it is not
   a full derivative through event movement.
4. The compiled camera atlas must be treated as a derived camera-specific
   cache, never as the exported world asset.
5. Optional richer atoms and cellular backends move behind measured compiler
   failure gates.

What does not change:

1. Do not abandon STAR for WorldFoam.
2. Do not promote SPD(4) itself as the novelty.
3. Do not begin production pentatope meshing.
4. Do not add a new umbrella formalism or rename the existing gauge/atlas work.
5. Do not let this renderer paper displace the mixed novel-view sampler and
   world-token training contract.

## Do not copy the memo's proposed abstract verbatim

The memo's final abstract describes a stronger future system than the repository
currently implements. In particular, it says that stable patches may store a
certified low-rank composite transfer and that the shared reverse exposes
visibility-boundary terms. The current paper has:

- certified projective trace-list/interval patches;
- a bounded static-affine retained-fiber/hybrid extension;
- a fixed-topology, piecewise-smooth compiled VJP;
- no general Type-II composite-transfer compiler;
- no event-boundary gradient estimator.

Adopting the proposed abstract verbatim would turn useful future directions into
false present-tense claims. Keep the **World Tubes in Gauged Camera Space**
identity and rewrite only around the claim that is implemented:

```text
canonical spacetime world + bounded continuous camera-chart program
    -> event-stratified sensor-time trace atlas
    -> many output samples
    -> fixed-topology shared world adjoint.
```

The broader event-stratified transfer-atlas formulation can appear in
discussion/future work, with explicit promotion gates. It should not be a
submission blocker or current method claim.

## The best insights in the review

### 1. The real object is the compilation boundary, not the Gaussian

Observed fact:

- Full-rank tube geometry with unrestricted `C in SPD(3)`, velocity `v`, and
  positive temporal variance is bijective with `SPD(4)`.
- The historical 14-scalar `legacy_tube` implementation does not expose that
  unrestricted source family; the explicit 18-scalar `full_spd4` lane is the
  strict source used for the equivalence/conditioning ablation.
- Native 4D Gaussian representations predate this project.
- The current paper already says that its strict SPD(4) source uses standard
  full-covariance mathematics and locates the contribution in lowering,
  certification, reuse, and adjoints.

Implication:

The strongest claim is:

```text
(camera-independent spacetime world, continuous camera program)
    -> reusable event-stratified sensor-time object
    -> many output samples
    -> one shared world VJP
```

This is stronger than "we splat a Gaussian in UVT" because it identifies the
repeated computation being removed and gives a causal same-world experiment.

Paper action:

- Keep the existing four-layer decomposition.
- Remove or subordinate any residual language suggesting that unrestricted
  tubes are a novel representation family.
- Describe tube coordinates as a physically legible chart of SPD(4), possibly
  with better optimization conditioning, not additional expressivity.

Falsification:

If frozen identical-world replay and compiled execution do not produce a
meaningful forward, reverse, or interaction-memory advantage, then the central
paper contribution fails even if World Tubes reconstruct better than another
representation.

### 2. Frame count is the wrong independent variable unless the camera program is held fixed

The review's most important complexity correction is:

```text
heavy work should depend on continuous camera-pushforward complexity,
not requested temporal sampling density T.
```

Increasing `T` while also changing the camera path, duration, shutter, or chart
fit is not evidence of frame-density invariance. Conversely, constant trace
count over sampled orbit fixtures is encouraging but not enough if the
compiler secretly consumes `T` poses or uses per-sample fitting work.

The quantities that should be held nearly invariant when only sample density
changes are:

- camera chart count;
- structural atlas record count;
- continuous primitive-chart references;
- support/order/event count;
- coefficient count;
- interaction memory.

The unavoidable term is output materialization:

```text
total work = heavy compiler/world work + N_output * evaluator/write cost.
```

Code action:

- Make the camera-program descriptor an explicit input identity.
- Vary evaluation samples without changing that descriptor.
- Record structural counts before output sampling.
- Fit runtime as `t(T) = t_heavy + N(T) * c_eval`.

Falsification:

If chart count, coefficient count, or continuous references scale with `T`,
then frame count has been hidden in compilation rather than eliminated.

### 3. Fixed-topology VJP and event-boundary differentiation are different claims

Current code and paper behavior:

- interval Metal differentiates trace coefficients, opacity, precision, and
  appearance;
- tile membership, visibility order, event topology, and fallback choices are
  compiled constants during the VJP;
- the paper already admits that event-boundary derivatives are not included.

This is a valid piecewise-smooth adjoint. It is not the derivative of the full
rendering map when a parameter moves a silhouette, support boundary, order
crossing, chart split, or fallback boundary.

This distinction should become a first-class evidence axis:

```text
pathwise coefficient VJP
    +
boundary/event term
    =
full finite-pixel derivative, when the event model is valid.
```

Near a scalar event `h(y, theta) = 0`, a boundary contribution has the generic
shape

```text
integral_{h=0} jump(integrand) *
    (-partial_theta h / ||grad_y h||) dS.
```

Assumptions:

- the event surface is regular: `||grad_y h|| > 0`;
- the two one-sided render semantics are defined;
- higher-order root collisions and topology flips are excluded or separately
  handled.

Near tangencies or root coalescence these assumptions fail. The correct
response is local subdivision/fallback, not pretending the boundary formula is
stable.

Decision:

- Do not block the initial paper on a general event-boundary estimator if the
  claim is explicitly piecewise-smooth and frozen-topology.
- Add one synthetic boundary-gradient experiment before making any broader
  geometry-gradient claim.

Cheap test:

- two overlapping colored sheets with a controlled depth crossing;
- one grazing support boundary;
- compare fixed-topology VJP, high-sample finite differences, and a prototype
  boundary correction;
- report error as a function of signed event distance.

### 4. Training-time usefulness depends on structural refresh, not just a shared backward

The review correctly separates:

```text
structural compilation:
    charts, active references, event topology, order, dependencies

numeric refresh:
    coefficients under a fixed structure
```

The current frozen-world experiment is the right causal renderer test, but it
does not establish that the atlas remains reusable while the world is being
optimized. If most steps rebuild most cells, a shared VJP may still lose
end-to-end.

For an event predicate `h_e`, a sufficient local stability condition is:

```text
|h_e| >= gamma_e
|Delta h_e| <= L_e ||Delta theta||
||Delta theta|| < gamma_e / L_e.
```

The exact Lipschitz bound can be conservative; even empirical signed margins
are useful initially.

Required training diagnostics:

- coefficient-only refresh fraction;
- locally invalidated record fraction;
- globally rebuilt fraction and frequency;
- rebuild wall time;
- minimum and quantiles of predicate margins;
- stale-certificate violations found by replay checks.

Decision:

- First prove frozen-world causal speed/correctness.
- Then add structural-refresh instrumentation.
- If training invalidation is too high, narrow the first paper to compiled
  inference/evaluation rather than hiding the failure.

### 5. Trace-list patches are only one possible evaluator regime

The current projective atlas is principally a trace-list/interval design. The
review's Type-II "composite transfer patch" is a genuinely new option:

```text
many local contributors
    -> compile their composed emission/transmittance
    -> evaluate a low-rank local transfer basis.
```

This could cap evaluation cost when line depth or overlap is large, but it also
introduces approximation rank, certification, positivity, and a more difficult
adjoint. It should not be implemented speculatively.

Trigger:

- p95 trace-list size or line-stabbing depth dominates runtime/payload;
- event stratification alone does not restore compactness;
- a low-rank transfer fit achieves the renderer-parity threshold at materially
  lower evaluator work.

Cheap test:

- fit one Chebyshev or Bernstein transfer patch on a dense translucent
  synthetic scene;
- compare coefficient count, forward error, VJP error, and evaluator time
  against the direct trace list;
- reject if rank grows approximately with contributor count.

### 6. "Atlas" must mean local camera charts, not one global fitted footprint

The review is right that a global UVT domain is not the same thing as a single
low-degree footprint chart. Wide orbits can cross:

- projection-denominator zeros;
- panorama or lens seams;
- field-of-view exits/re-entries;
- support and order events;
- high-curvature regions.

The current event/window machinery already addresses much of this in bounded
fixtures. The remaining gap is a production-quality continuous camera
descriptor and measured chart growth under general pose/intrinsics programs.

Next supported camera family should include:

- an `SE(3)` spline or equivalent bounded local chart;
- spline intrinsics;
- explicit exposure interval;
- explicit rolling-shutter time map.

Do not require a 360-degree orbit in one chart. Require chart count to depend on
angular extent/conditioning and remain invariant to requested sample density.

### 7. Camera-pushforward complexity is the right cross-representation metric

Primitive count and learned bytes are insufficient. Two worlds with equal bytes
can induce radically different camera-path workloads.

Useful measured complexity vector:

```text
K_Gamma =
(
  primitive-chart references,
  support events,
  order/traversal events,
  chart count,
  local trace depth,
  approximation rank,
  unresolved/fallback measure
).
```

This is not yet a single theorem-friendly scalar, but it is already a better
experimental object than primitive count.

Use it to compare:

- restricted tubes vs native SPD(4);
- World Tubes vs a cellular backend;
- linear tubes vs a future curved-motion atom;
- the same world under different camera programs.

### 8. Preserve the gauged order repair, but separate it from retained transfer

The recent gauge/order work contains two related mechanisms that must not be
collapsed into one vague "holonomy" claim.

The first is part of the primary World Tubes compiler:

```text
moving camera program
    -> bounded camera-ray gauge charts
    -> conditional depth fields
    -> certified support/order events
    -> stratified projective trace atlas.
```

This is the mechanism that repairs mean-depth order crossings caused by camera
motion on the tested bounded chart segments. It is already represented in the
synthetic theorem table by the raw crossing failure and zero-error stratified
repair. Removing it would remove central method code and invalidate the gauged
camera-space paper framing.

The second mechanism is **World Tubes + Ordered Ray Transfer**. It addresses a
narrower failure: thick differently colored density profiles can overlap along
one ray so that no ordering of representative depths reproduces their
front-to-back emission--absorption integral. The retained-fiber implementation
evaluates the path-ordered transfer product on those ambiguous tiles. This is
parallel transport along an open ray; "holonomy" is geometric inspiration, not
the executable or paper name.

Current evidence:

- the 16-atom fixture routes `10/64` tiles to retained transfer and matches the
  all-retained oracle;
- the 199-atom dense fixture routes `64/64` tiles to retained transfer, so it is
  a negative selectivity result rather than a hybrid speed result;
- the implemented route is bounded static-affine retained depth, not the
  meta-review's general Type-II composite-transfer atlas.

Decision:

- retain gauge charts, conditional depth, and event-stratified order repair in
  the primary method;
- retain ordered transfer as the explicit WT-OT0--3 extension/ablation;
- do not put WT-OT into the 21-row selected-time matrix until the dense
  certificate becomes selective;
- do not imply that general WorldFoam or low-rank composite-transfer patches
  have been absorbed into World Tubes.

Falsification:

If retained transfer does not improve the colored-overlap stress case, or the
hybrid remains all-fallback and cannot beat the all-retained oracle, demote the
extension without weakening the primary event-stratified trace-atlas result.

## What the review has already been integrated into

| Review recommendation | Current project status | Remaining gap |
| --- | --- | --- |
| Four layers: world/compiler/evaluator/adjoint | Present in the World Tubes paper | Keep terminology consistent in abstract, figures, and runner schema |
| SPD(4) equivalence / no primitive novelty | Present in paper and native-SPD(4) lane | Add exact tube-to-SPD parity render/gradient row if not already artifact-complete |
| Event-stratified local atlas | Substantial projective event/window implementation exists | General continuous camera descriptor and public variable-camera stress |
| Same-world causal ablation | Dedicated frozen-world executor exists | Runtime artifact is still pending |
| Shared direct VJP | Native interval direct VJP exists and trainer route is wired | Explicit phase timing, interaction memory, deterministic reduction, boundary term |
| Frame-density invariance | Synthetic orbit tests and scaling artifacts exist | Continuous-program identity and full causal `T` sweep |
| WorldFoam as parallel backend | Already the lane taxonomy | Keep it gated on retained-transfer evidence |
| Pentatope/reference backend | Not implemented | Defer unless a small oracle is needed for event/transport certification |
| Curved swept-volume atom | Not implemented | Defer behind measured linear-tube failure |
| Structural vs numeric refresh | Not established | Instrument invalidation and rebuild cost during training |
| Composite-transfer patch | Not established | Conditional experiment only after trace-depth failure |
| Event-boundary derivatives | Explicitly absent | Add one bounded synthetic experiment and precise limitation |

## How this interacts with the broader DynaWorld plan

The renderer/compiler paper and the learned world-token project are related but
must not be conflated.

The world-token contract remains:

```text
W0 = E(O)
S_tau = G(W0, tau)
Ihat = R_fixed(S_tau, camera)
```

World Tubes can implement or accelerate `R_fixed`, but:

- `W0` or the generated spacetime world is the camera-independent asset;
- the compiled atlas `A(W0, Gamma)` is a camera-specific derived cache;
- a new camera program must be compiled from the world;
- the encoder must never export only the camera-specific atlas;
- no target-camera-conditioned learned branch may be smuggled into the world
  generator.

This gives a clean deployment split:

```text
export:
    world asset W0 + world/generator version

runtime cache:
    compile(W0, camera program Gamma) -> atlas A

render:
    evaluate(A, requested sensor-time samples)
```

The review therefore improves the export-purity story: it makes explicit that
camera compilation is allowed renderer-side conditioning, while the canonical
world remains self-sufficient and camera-independent.

It does **not** solve the missing mixed same-view + heldout-novel-view sampler.
That sampler remains the bridge required to learn useful worlds. Renderer
efficiency cannot compensate for a training distribution that never identifies
novel-camera behavior.

## Revised priority order

### P0: run the frozen identical-world fixed-interval sweep

Use the existing lane-isolated executor on a clean, adequately provisioned
host. The executor now trains and saves once, then evaluates
`F={4,8,16,32,64,128,full}` from the same checkpoint. Each row uses ordered
integer samples spanning the same full physical interval rather than a growing
prefix.

Must establish:

- checkpoint hash identity;
- exact selected-frame/time-grid identity;
- identical cameras, targets, alpha semantics, precision, and support cutoff;
- image/loss parity;
- nonzero world-gradient coverage and parity;
- compile, forward, backward, and payload accounting;
- fallback fraction.
- non-unit selected-time full-atlas versus chunk-slice forward/VJP parity;
- warmed/repeated performance timing separate from single-shot correctness
  timing.

Do not cite the source implementation as a result.

Future instrumentation can add:

- chart count;
- continuous references;
- event count;
- coefficient count;
- evaluator writes;
- interaction memory;
- fitted heavy intercept and output slope.

The fixed-interval sweep precedes representation breadth because it directly
tests the paper's core mechanism. Each `F` compiles its own atlas from the same
world/program; the claim is not that one identical atlas object is reused
across densities.

### P1: variable-camera stress and chart-complexity boundary

Sweep one axis at a time:

- orbit extent;
- angular velocity;
- translation;
- FOV/near-plane conditioning;
- exposure;
- rolling-shutter slope.

Report collapse rather than hiding it. A collapse can mean:

- fallback dominates;
- chart width approaches per-frame width;
- runtime ratio reaches or exceeds replay;
- parity fails without fallback.

### P2: training-time structural refresh instrumentation

Measure coefficient refresh, local invalidation, global rebuilds, and margin
quantiles over real optimizer steps.

Decision branch:

- low invalidation: keep the training-time compilation claim;
- high invalidation but strong frozen-world/inference reuse: narrow to an
  inference/evaluation compiler paper;
- high invalidation and weak inference break-even: kill the main claim.

### P3: one event-boundary gradient fixture

This should be a bounded scientific honesty gate, not a general differentiable
visibility project.

### P4: only then consider richer evaluator/backend work

- composite-transfer patches if trace depth is the measured bottleneck;
- WorldFoam camera-bundle compilation if retained transfer is the measured
  semantic bottleneck;
- a tiny pentatope backend if exact local transport/event certification needs
  an independent oracle;
- swept-volume atoms only if curved motion requires at least roughly three
  linear tubes at matched quality/bytes.

## Acceptance thresholds: adopt the structure, calibrate the numbers

The memo proposes useful gate categories:

- heavy-work invariance;
- end-to-end speed;
- output slope;
- interaction memory;
- deterministic backward;
- same-world image/gradient parity;
- geometry/event-gradient accuracy;
- candidate/rank compactness;
- stable compiled coverage;
- angular-motion robustness;
- training-time invalidation;
- representation/backend keep-kill gates.

The categories should be integrated into the evidence schema now.

The exact numeric cutoffs in the memo should **not** all be canonized without a
pilot. Values such as `2x` speedup, `1.7x` training speedup, `98%` stable
coverage, or a `20%` invalidation ceiling are research targets, not derived
constants. Preserve them as aspirational kill targets, then freeze protocol
thresholds after:

1. one clean pilot;
2. measurement-noise quantification;
3. comparison against the actual same-world replay cost;
4. confirmation that the threshold cannot be gamed by smaller worlds, fewer
   outputs, lower quality, or hidden fallback.

Parity thresholds are less negotiable and can be strict immediately.

## Branches and backtracks

### Branch A: the current projective atlas already proves the paper

Status: weakened.

Why it might be true:

- substantial correctness, orbit, event, forward, and VJP machinery exists;
- frame-density tests show structural compression;
- a paper draft and evidence schema already exist.

Why it is not yet enough:

- the frozen identical-world run has no accepted runtime artifact;
- current general moving-camera SPD(4) compilation is first-order/one-chart;
- structural refresh under training is unmeasured;
- event topology is detached during VJP;
- broad matrix rows are representation/context evidence, not causal compiler
  evidence.

### Branch B: native SPD(4) should replace legacy tubes as the headline

Status: invalidated as a novelty claim; supported as an implementation axis.

Evidence:

- the parameterizations are bijective when the tube spatial covariance is
  unrestricted;
- native 4D Gaussian prior art is explicit;
- current bounded quality differences can arise from initialization,
  parameter count, conditioning, amplitude semantics, or code paths.

Use:

- exact parity and conditioning ablation;
- strict source semantics for depth variance and retained transfer;
- not a new primitive contribution.

### Branch C: WorldFoam should replace STAR because it preserves ray order

Status: unresolved as a future backend; rejected as the immediate mainline.

Why it might win:

- retained optical transfer has cleaner semantics for overlapping colored
  density;
- cells can expose exact traversal events.

Why it might lose:

- point partition complexity does not bound line-stabbing depth;
- topology flips and site gradients are difficult;
- the current broad quality/system case is weaker;
- it risks displacing the compiler question with a representation problem.

Keep gate:

```text
compiled camera-time event word materially smaller than summed per-frame
traversals, correct site gradients away from flips, and real end-to-end benefit.
```

### Branch D: a richer curved atom is the missing breakthrough

Status: deferred.

The Bernstein swept-volume atom is mathematically coherent, but it combines
known ingredients and may simply trade world primitive count for higher chart
rank/event count. It earns implementation only after a curved-motion benchmark
shows a material matched-byte win over mixtures of linear tubes.

### Branch E: the compiler paper is orthogonal to DynaWorld's product goal

Status: partly true, but not a reason to stop.

The compiler does not learn novel views and does not solve ambiguous world
inference. It can nevertheless become the fixed renderer/export runtime for the
world-token model. The project should protect both lanes:

- evidence closure for the compiler paper;
- mixed omitted-observation training for the learned world model.

Neither should be presented as solving the other.

## Concrete document/code integration map

Do not edit all of these at once. Apply changes only as the corresponding
evidence lands.

1. `research_notes/gauged_uvt_trace_atlas/paper/WORLD_TUBES_PAPER_DRAFT.md`
   - preserve the current four-layer framing;
   - add camera-pushforward complexity and structural-refresh limitations;
   - keep event-boundary derivatives explicitly outside the current VJP claim;
   - make the frozen same-world experiment the primary causal table.

2. `research_experiments/paper_runner_suite/run_frozen_world_replay_compiled.py`
   - extend from one accepted row to a fixed-camera-program density sweep;
   - add phase-separated timing and structural counts;
   - add interaction-memory rather than only total payload.

3. Projective trace/compiler code
   - expose a continuous camera-program identity independent of requested
     samples;
   - report chart/event/reference/coefficient counts before evaluation;
   - later expose predicate margins and dependency lists for local refresh.

4. `TODO/README.md`, `EXPERIMENTS.md`, and `BASELINES.md`
   - keep frozen identical-world execution first;
   - distinguish causal compiler evidence from representation/context rows;
   - add variable-camera and structural-refresh gates only after P0 runs.

5. `research_notes/training_contract_v1.md`
   - eventually clarify that a camera-specific compiled atlas is a derived
     renderer cache, not `W0`;
   - keep query camera out of learned world generation.

6. WorldFoam documents
   - retain separate paper identity;
   - use the same camera-pushforward complexity metrics;
   - do not merge its transfer semantics into STAR compatibility claims.

## Final recommendation

The meta-review should be treated as a **paper-claim and acceptance-criteria
upgrade**, not an architecture reset.

The immediate sequence is:

```text
frozen same-world causal run
    -> fixed-program T-density sweep
    -> variable-camera collapse map
    -> training structural-refresh audit
    -> bounded event-gradient fixture
    -> only then optional composite/cellular/curved-atom work
```

The best possible outcome is a cleaner paper whose contribution survives
representation prior art:

> World Tubes compiles a camera-independent spacetime world and a continuous
> camera program into reusable, event-stratified sensor-time traces, then
> reduces many image residuals through a shared adjoint back to the same world.

The result should be killed or narrowed if that mechanism does not survive the
identical-world, fixed-program, variable-camera, and structural-invalidation
tests. That is the review's most valuable change: it turns an attractive
formalism into a falsifiable systems claim.

## Evidence-contract audit after the meta-review

The four-layer reading exposed a separate problem in the existing three-lane
table: the reports named similar protocols, but did not prove that the three
lanes consumed the same ordered samples, decoded targets and cameras, evaluator,
runtime, binaries, or W&B artifacts. A source audit found:

- World Tubes and dynamic 3DGS averaged per-view PSNR, whereas WorldFoam
  converted one global MSE to PSNR. The current heldout split has one view, so
  this does not explain its heldout numbers, but it falsifies the claim that all
  reported metrics came from one mechanically identical evaluator.
- No report bound the bytes of every consumed raw input or the final decoded
  RGB/camera bundle.
- Hardware and actual loaded native binaries were incomplete, so timing reuse
  was not fail-closed.
- The seed-17 WorldFoam ID named as a clean/offline run resolves to an older
  online/non-clean W&B artifact.
- Matrix reuse trusted too much parent metadata and did not reopen all child
  execution and artifact identities.

Evidence schema v2 now binds the canonical ordered schedule; raw inputs;
decoded RGB, intrinsics, poses, lens/distortion, names, times, and anchor
transform; one canonical full-set evaluator; hardware/runtime and route-native
binaries; checkpoints/configs/histories/media; initializer bytes; and actual
finalized W&B files. Matrix reuse revalidates those identities. The frozen
replay-versus-compiled payload fields were also renamed to logical
tensor-element volume because they double-count shared replay values and omit
topology, packed bins, and transient working memory.

This changes evidence status, not the method:

```text
method/code:
  near closure, pending behavior gates

selected-time schema-v1 rows:
  historical numerical diagnostics

selected-time schema-v2 accepted ledger:
  minimum Coffee submission subset: 0 / 7
  full breadth target: 0 / 21

publication runtime queue:
  frozen identical-world fixed-interval sweep
  -> bounded variable-camera closure/death curve
  -> seven schema-v2 Coffee progressive/control rows
  -> regenerated tables/figures/venue PDF

post-minimum breadth:
  remaining 14 selected-time rows
```

The old three progressive rows cannot be migrated by relabelling because the
missing identities were never recorded. They must be rerun. This is the
practical consequence of the meta-review's compiler/evaluator separation:
paper evidence must identify the world and observations, compiler execution,
evaluator, and adjoint/schedule independently, or a numerical table cannot
support a causal systems claim.

## Safe implementation closeout

The source-only paper slice was completed without launching Torch, MPS,
training, dataset decoding, or other memory-heavy work on the incident host:

- the frozen identical-world runner now compares non-unit full-atlas slices
  against the exact same parent atlas, requires forward/loss/world-VJP parity,
  and records alternating warmed repeated timings with raw samples;
- the bounded variable-camera closure/death runner now holds one world,
  physical interval, and sample count fixed while increasing camera motion,
  and verifies against an exact rational live-depth-order oracle;
- the submission artifact generator fails closed on a canonical completed
  matrix summary, exact retained schema-v2 run summaries, recomputed frozen
  timing statistics, and verifier-accepted variable-camera and theorem
  reports;
- incomplete components export labelled placeholders rather than partial
  numbers.

The current generated ledger has ten unresolved evidence records: one
canonical matrix summary, seven public run summaries, one frozen-world report,
and one variable-camera report. These are nine runtime jobs because the
canonical matrix summary is emitted after all seven matrix jobs validate.
The focused pure-Python gate is `23 passed`; static compilation and incomplete
bundle generation are green. Runtime evidence remains intentionally absent
until a clean, adequately provisioned host passes the incident-calibrated
preflight.
