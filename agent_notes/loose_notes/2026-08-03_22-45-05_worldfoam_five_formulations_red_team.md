# WorldFoam Five-Formulation Red Team

## Context

The user supplied five proposed mathematical formulations and asked whether
they improve the current WorldFoam route, what work is active now, and whether
the repository has already derived new formulations of its own.

This review compares the attachment against the current source and theorem
documents, especially:

- `research_notes/worldfoam_paper/WORLD_FOAM_DYNAMIC_DEPTH_ORDER_MATHEMATICIAN_PROMPT.md`;
- `research_notes/worldfoam_paper/WORLD_FOAM_MEMORY_LIGHT_THEOREM_LEDGER_2026-08-03.md`;
- `research_experiments/world_foam_lane2/kinetic_owner_chart_compiler.py`;
- `research_experiments/world_foam_lane2/kinetic_active_owner_chart_compiler.py`;
- `research_experiments/world_foam_lane2/kinetic_geometry_trust_region.py`;
- `research_experiments/world_foam_lane2/kinetic_stable_stratum_vjp.py`; and
- the current bounded lazy/native coordinator work.

No literature-novelty claim is made here. “New” means derived or newly
formalized in this repository, not proven absent from prior mathematics.

## Current Model

The current route is already:

```text
direct kinetic 3D power sites
  -> exact/certified ray-time lower-envelope charts
  -> ordered P0 affine transfer at adaptive temporal nodes
  -> streamed residual-to-node reduction
  -> one material-word VJP per compiled spatial block today
  -> one full frozen-program word/geometry VJP after the open integration gate
```

The required asymptotic separation is:

```text
heavy geometry + ordered-word + world-VJP work
    O(Topology(S,E,Q) + sum_(p,c) J_pc R_pc)

sample/output/target work
    O(sum_(p,c) F_pc J_pc) + Omega(PF)
```

The first term should not grow when only the requested frame grid becomes
denser. The second cannot disappear because targets and output pixels must be
read and written.

## Evidence Classification

### Observed in source

- Exact fixed-time lower-envelope discovery exists.
- Exact algebraic near/far/triple guard isolation exists in the exhaustive
  correctness compiler.
- A unique-word active-boundary closure exists as the production-oriented
  structural candidate.
- Full-fiber ties and unsupported degeneracies fail closed.
- P0 ordered transfer and its constant-state prefix-only reverse exist.
- Stable-stratum physical-length geometry VJP exists at CPU/source scope.
- Simple-root re-isolation and a narrow event-free optimizer trust region
  exist.
- CPU/fake-native block-major execution demonstrates node reduction followed
  by one word VJP, independent of temporal chunk count.

### Inference

These pieces support the mathematical architecture, but they do not yet prove
that the production trainer, native runtime, optimizer lifecycle, and measured
allocator peak satisfy it end to end.

### Still unproved

- output-sensitive kinetic event maintenance with a useful bound;
- a production certificate for both primal transfer and the needed sparse
  derivative actions;
- useful multi-chart optimizer reuse/local repair under real training;
- native-runtime frame-density scaling and allocator/command-buffer peak;
- publication-scale quality and performance.

## Candidate-by-Candidate Decision

| Candidate | Relationship to current route | Value | Decision |
| --- | --- | --- | --- |
| Parametric tropical ray-time complex | Formalizes the lower-envelope object already compiled in source | Strong theorem/exposition; no new production complexity bound | Keep as geometry language, avoid “tropical” branding and full arrangement materialization |
| Optical-depth translated-measure monoid | Lifts the current four-scalar affine monoid to an order-explicit measure | Adds useful tangent-measure and error/truncation certificates | Promote selectively into theorem/certificate work |
| Constructible transfer sheaf / exit-path automaton | Renames current exact charts, event records, seam rules, and re-isolation problem | Clarifies regularity and seams; assumptions hide the hard repair problem | Appendix/exposition only; do not make it an implementation lane |
| Persistent affine-monoid circuit | Same hierarchical-product branch already listed in the mathematician prompt | Useful only for sparse leaf edits or selected JVP/VJP queries | Optional fast path after the primary trainer; not a replacement for prefix reverse |
| Similarity-gauge residual foam | Strengthens the existing canonical-deformation/shared-motion branch | Could exactly remove coherent similarity motion and reduce events/rank | Worth a small gated experiment later; must preserve polynomial/certified camera complexity |

## 1. Parametric Lower-Envelope Complex

### What is correct

For affine site trajectories, quadratic weights, and affine camera rays,
site-independent quadratic-in-depth terms cancel. Ownership is the lower
envelope of lines in depth whose coefficients have degree at most two in time.
Under the stated genericity exclusions, endpoint crossings and active triple
concurrences characterize changes of the vertical owner word.

The repository already embodies this formulation. The exhaustive compiler
stores exact polynomials and rational isolating intervals, distinguishes
denominator guards from semantic events, uses right-continuous seams, and
fails closed on full-fiber and simultaneous degeneracies.

### What it does not buy

The full semialgebraic arrangement is potentially enormous. Calling it a
“tropical complex” changes neither the event count nor the construction cost.
The hard problem remains discovering only active events and repairing them
output-sensitively.

### Decision

Use the lower-envelope-complex theorem in the paper if it shortens the proof.
Do not build the full complex, and do not rename the method around tropical
geometry.

## 2. Optical-Depth Translated Measure

### Genuine addition

For ordered optical depths `tau_r`, cumulative depths `K_r`, and piecewise
color profile `c#(u)`, define

```text
kappa = sum_r tau_r
dnu(u) = c#(u) du on [0,kappa].
```

Concatenation is

```text
(kappa_A, nu_A) odot (kappa_B, nu_B)
  = (kappa_A + kappa_B,
     nu_A + shift(kappa_A)_# nu_B).
```

The existing affine transfer is the Laplace image

```text
L(kappa,nu) = (exp(-kappa), integral exp(-u) dnu(u)).
```

This is a clean explanation of where order lives: in translations of color
mass along cumulative optical depth. It also yields the repository's current
prefix-only VJP as the adjoint of this representation.

### Why it is not a new executor

The full measure is larger than the four-scalar transfer. Building it still
requires the exact owner word. At one time, the native P0 executor already
uses the minimal four-scalar homomorphic image. Replacing that executor with a
measure object would increase state, not reduce it.

### Useful consequences

1. A weighted-total-variation certificate can control primal transfer error.
2. The tangent measure exposes moving boundaries as Dirac masses, explaining
   why low primal rank does not imply low tangent rank.
3. An opacity tail beyond `U` has a per-channel forward bound `C exp(-U)` for
   colors bounded by `C`.

Before promotion, define the norm for vector-valued measures precisely and
extend any opacity-tail gate to the requested tangent/VJP and loss bounds.
Opaque-tail truncation is unsafe for training if it certifies only RGB while
discarding a significant geometry/material gradient.

### Decision

This is the best proposal. Add it as a theorem/certificate layer and first
build a CPU oracle against the existing P0 prefix implementation. Do not
replace the execution ABI.

## 3. Constructible Transfer Sheaf

### What it clarifies

The time interval is stratified into open topology-stable charts and exact
event points. Transfer is analytic within a regular chart, continuous but
generally not differentiable at a zero-length birth/death, and genuinely
ambiguous at a positive-length differently materialized full-fiber tie.

This is useful paper language. It agrees with current half-open charts,
identity seams, fail-closed tie policy, event-time derivatives, and
simple-root re-isolation.

### Hidden tautology

The proposed root-isotopy theorem assumes that roots remain simple, no roots
appear in complements, root order is preserved, every event retains its
classification, and chart witness words remain valid. Those are almost the
entire difficult certification problem. The theorem then proves that the
program is structurally the same, which is correct but not a repair algorithm.

### Decision

Use the regularity/gluing statements in an appendix. Keep implementation
vocabulary concrete: chart, event record, transition, conflict, and repair.

## 4. Persistent Monoid Circuit

### Exact benefit

A balanced product tree gives `O(J k log R)` repair for `k` changed leaves and
selected sparse JVP/VJP queries. A blocked tree trades cache bytes against
repair cost.

### Why it is not the primary backward

Normal training updates many or all site materials and geometry. One changed
site can own runs across many track-chart incidences, so “k changed sites” is
not “k changed leaves.” When the touched fraction is order one, a tree is no
better asymptotically and can be worse than the existing `Theta(JR)`
constant-state prefix reverse. It also adds `O(JR/b)` retained summaries.

### Decision

Keep only for coordinate-sparse updates, frozen-material geometry updates,
local topology edits, or selected diagnostics. It must earn itself against the
flat prefix implementation on touched-incidence fraction and bytes.

## 5. Similarity-Gauge Residual Foam

### Exact useful theorem

An isotropic Euclidean power diagram is preserved by common similarities
`x = s R y + b` when positions and weights transform as

```text
p_i = s R a_i + b
omega_i = s^2 w_i.
```

For a coordinate gauge, extinction transforms as `rho_x = rho_y / s`, so
optical depth is invariant. A world and camera sharing the same similarity
motion can become static in canonical coordinates, yielding no topology events
and rank one for a constant material program.

### Main danger

A time-dependent rotation can turn the current affine/polynomial camera and
site functions into trigonometric or rational functions. That can destroy the
degree-two/degree-four event theorem and make certification harder than the
events it removes. Physical expansion must also not be confused with a mere
coordinate rescaling of density.

### Decision

This is a potentially useful representation preconditioner and a clean bridge
to the project's camera-gauge mathematics. Test it only with a gate that
measures the combined structural cost:

```text
event count + chart count + certified rank + transformed-camera predicate cost.
```

Keep the gauge only if that total decreases. Do not block the P0 native trainer
on it.

## Did This Repository Derive New Formulations?

Yes, in the limited and honest sense of repository-derived constructions:

1. It rejected fixed shared-SPD(4) slicing as a general dynamic foam and
   selected direct kinetic 3D sites with affine trajectories and quadratic
   weights.
2. It derived the pulled-back degree bounds and exact lower-envelope event
   predicates for that frontend.
3. It implemented exhaustive and active-certificate continuous owner-chart
   compilers with exact root isolators and fail-closed semantics.
4. It derived and implemented the four-scalar ordered affine-transfer closure
   and constant-state prefix-only word VJP.
5. It derived the frame-density factorization: evaluate words at adaptive
   nodes, stream all residuals into node cotangents, then execute one world VJP
   per compiled block as the target architecture. The currently integrated
   coordinator proves this invocation count for the material-word VJP; the
   full geometry/world bridge remains separate.
6. It separated frozen-program, physical stable-stratum, and compiled-algorithm
   derivatives; implemented the stable-stratum physical-length geometry VJP;
   and added narrow trust-region/root-re-isolation machinery.

These are real mathematical and systems formulations. Their publication
novelty has not been established merely by deriving them here.

The attachment is mostly a second formalization of branches already present
in the canonical mathematician prompt:

```text
1 -> B1 kinetic lower envelope
2 -> B4 affine/product-integral structure, with a genuinely useful measure lift
3 -> B3/B7 exact charts and structural repair
4 -> B5 hierarchical products
5 -> B9 canonical deformation/shared motion
```

## What Is Active Right Now

The active work is implementation plumbing required to make the proven CPU
factorization honest in a production-style step:

1. A bounded replayable dense-frame observation source, avoiding an `O(PF)`
   resident Python observation tuple.
2. Separation of structural compile identity from sampled target observations,
   so a program can be reused across steps without recompiling because targets
   changed.
3. Cold full-content validation versus cheap warm identity/version validation,
   avoiding repeated `O(VF)` hashing during bundle iteration.
4. A byte-bounded LRU of immutable CPU compiled programs/lowerings, retaining
   neither targets nor device runtimes.
5. Coordinator lifecycle hardening: lock release, generator closure, exception
   safety, device fences, provenance checks, and explicit memory accounting.
6. An honest target policy. Caching every decoded target frame once is
   `O(F * image_bytes)` in bundle-major execution; the memory-minimal default
   must stream one or a bounded number of frames even if that repeats decode.

This work is not yet the production unified WorldFoam trainer. Full geometry
VJP integration, raw-parameter chain rules, optimizer/checkpoint/eval wiring,
native rebuild/parity, and measured quiet-host scaling still remain.

## Cheap Falsification Tests

### Measure formulation

Build a CPU oracle from existing CSR owner words and `[J,R]` lengths. Compare:

- total `(beta,m)`;
- every density, color, and length VJP;
- zero-density and zero-width limits;
- high-opacity forward and tangent truncation bounds; and
- left/right birth/death limits.

Any mismatch invalidates the proposed measure formulas before they touch a
shader.

### Similarity gauge

Use a synthetic coherent rigid/similarity motion with known canonical world.
Compare ungauged versus gauged:

- exact rendered transfer and VJP parity;
- event and chart counts;
- adaptive rank;
- transformed camera predicate degree/conditioning; and
- compiler/runtime bytes.

Reject the gauge if total certified cost does not decrease.

### Persistent circuit

Sweep touched run-incidence fraction. Compare flat prefix versus blocked tree
for wall work and retained bytes. Promote only below an empirically and
analytically stated sparsity threshold.

## Decision

Do not pivot the implementation around all five proposals.

Promote the optical-depth translated-measure view into a CPU oracle and paper
certificate layer. Retain the lower-envelope and stratified-chart language as
proof organization. Park persistent circuits. Gate similarity factoring as a
later event/rank preconditioner. Continue the bounded compiled-program and
production-step work because that is the current shortest path from the math
to the claimed memory-light trainer.

## Independent Scientist Feedback

A subsequent scientist review agreed with the ranking above:

- the repository work is stronger as project engineering and implementation;
- the translated optical-depth measure is the one genuinely strong new
  mathematical formulation;
- the lower-envelope, constructible-chart, product-tree, and similarity-gauge
  proposals mostly formalize or extend branches already in the canonical
  handoff; and
- the best paper combines the existing architecture with the measure theorem,
  without materializing the measure in the runtime.

The review also identified one wording overreach: the integrated block-major
path currently demonstrates one **material-word** VJP per active native block.
The stable-stratum geometry and native node-length-bar pieces exist, but they
are not yet one rebuilt, dataset-bound, end-to-end native geometry-training
step. The paper draft and mathematician prompt were tightened accordingly.

The scientist's recommended next mathematical question is not a sixth
formulation. It is whether separated simple-root continuation supports sound,
output-sensitive affected-chart/payload repair during optimizer updates, or
whether full structural recompilation should remain the safe production rule.

The follow-up audit answered that question for the current implementation:
fresh structural recompilation remains mandatory after every geometry or ray
update. A previously rootless predicate can acquire roots, so existing event
records are not a sufficient repair set. The simple-root routine rebuilds the
whole rooted/rootless registry and certifies a restricted continuation, but it
returns no patched program and does not rebuild charts, ranks, payloads, or
native dispatch. Material-only updates remain the safe reuse case.

Two more source-level prerequisites also closed without native execution: a
bounded frame-free compiled CPU artifact store plus a replayable dense-
observation source, and a separate fenced reduction from one native
equal-rank `[J,W_b]` length bar to frozen-stratum site/trajectory/weight/ray
bars. They remain separate from the legacy coordinator and production trainer;
that integration boundary is now explicit rather than hidden by a broad
"geometry VJP complete" statement.

## Safe Closeout Under Host Saturation

The translated-measure formulation now has a compact proof-only CPU oracle.
Its six focused tests cover semidirect associativity, the Laplace homomorphism,
the two-segment commutator, distributional tangents against autograd/finite
differences, P0 VJP parity, and one-sided zero-width/zero-density limits. It
does not alter runtime state or claim literature novelty.

The corrected block-scoped geometry reducer passed `12` focused CPU tests and
now carries `(view_index, track_id)` ray keys, checks every analytic helper's
derivative scope/provenance, and reports all-site validation plus dense site
accumulation separately from `O(JR)` word work.

Two final integration candidates were being reduced rather than expanded:

- dense cached material replay with one-frame target decoding had a green
  five-test pre-dedup gate; the current deduplicated file is statically clean
  but its post-dedup runtime rerun was externally terminated;
- a full material-plus-length VJP assembler had a green larger draft, then was
  rewritten around the existing outer coordinator; the reduced current file
  did not receive its final runtime gate.

Static review of that reduced geometry assembler found that its caller-
supplied node bar and sample counts are not sealed by the executor that
performed the reductions. A nonempty provenance string is not a coverage
proof. The production fix is to extend the already-tested block lifecycle
executor with mutually exclusive material-only and full-geometry finalizers,
then reuse that executor from dense cached replay. The free-standing assembler
must remain unpromoted (and should be removed once its useful aggregation code
has migrated); adding another coordinator would move away from the cleanup and
unification goal.

The executor source was subsequently extended along that exact seam. One
session now binds either `material_only` or `full_geometry`, rejects mixing,
ties both modes to the same world token and accumulated node-bar identity, and
accounts the bounded `[J,W_b]` length output only in full mode. Source tests
were added for both material parity and `F=5/41` full-mode invariance. No test
was run because host load remained near `90`; this is source progress, not a
green gate. Dense replay and fenced geometry reduction are still to be joined
to the new executor mode.

At that point the host load exceeded `100` (observed approximately `117`), so
all subagents and repeat tests were stopped. No MPS, Metal, CUDA, extension
build, dataset decode, or training workload ran. Both reduced integration files
must remain unaccepted until one quiet-host CPU run passes; native parity and
allocator evidence remain later approved-host gates.
