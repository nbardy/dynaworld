# Framing 3 — Bitter-Lesson Predictive Quotient (the default baseline)

Third framing of the novel-view problem. Where framing 1 used information
theory and framing 2 used the self-sufficiency contract with dedicated
mechanisms per constraint, framing 3 asks a stricter question: *which of
those mechanisms are earning their cost, and which are only supervising
internals that the deployed query behavior does not identify?*

The bitter-lesson answer collapses most of them.

This is the current default baseline. Framings 1 and 2 are preserved as
derivation history and audit vocabulary, not replaced.

---

## One-liner (north star)

> **The simplest model whose held-out predictive objective identifies a
> behavioral world quotient on supported queries, then uses
> rate/minimality to choose a useful representative.**

One predictive signal. One sampler. One encoder. One exported world
asset. The invariants you care about either change predictive behavior
on the query support or they are outside this contract. Rate/minimality
is load-bearing because the predictive loss defines an equivalence class,
not a unique latent.

---

## The world-view

Architecture selection is the wrong level of the problem. At the right
level:

1. **Identify invariants.** What properties must W have at deployment?
2. **Identify unavoidable physics.** What does the world force you to
   respect regardless of data?
3. **Identify the inference contract.** What shape does the deployed
   system have?
4. **Identify the supported predictive query set.** What held-out
   observations make the deployed behavior observable?
5. **Select a representative.** What rate/minimality pressure prevents
   useless behavior-equivalent encodings?
6. **Drop everything else.** Every mechanism whose target is already
   handled by predictive behavior, rate/minimality, unavoidable physics,
   or the export contract is decorative.

The job is not to build a bigger model or a smarter architecture. The job
is to identify the minimum commitment that closes the contract and let
scale do the rest.

---

## The inference contract (load-bearing; physics + product)

```
S_tau = G(W, tau)          # time-conditioned scene / splats
image = R_fixed(S_tau, c)  # fixed rasterizer at query camera
```

Three deploy-time inputs:
- `W` : world tokens (the exported asset)
- `c in SE(3)` : camera query
- `tau in R` : time query

One deploy-time output: `image`.
No other state is read. No per-frame cache, no encoder residual, no side
channel. Query camera enters only at `R_fixed`. Query time may enter
`G(W, tau)`. There is no learned camera-conditioned pre-render branch.

The semantically complete object can be stochastic, `p(W | O)`. For v1,
a deterministic `W0 = E(O)` is allowed and should be read as a point
estimate or posterior summary. Full stochastic `W` is optional and lives
behind EH-3, not in the baseline.

This is not a loss; this is the architecture. The loss lives on top.

---

## The training contract (predictive loss + representative pressure)

### Sampler

Let `V` be a clip (possibly from a single monocular camera). A training
sample is a triple `(O, H, Q)`:

- `O subset V` : observation set the encoder sees
- `H subset V` with `H not_subset O` : held-out target frames (ground truth)
- `Q = { (c_h, tau_h) : h in H }` : query poses / times for the held-out
  frames

`(O, H, Q)` is drawn from a variable-budget distribution `D_var` whose
support covers:

- monocular sparse (`|O|` small; `H` elsewhere in the clip)
- monocular rich (long `O`, short `H`)
- monocular + crop-as-extrinsic (`O` crops; `H` uncropped or differently
  cropped, with shifted principal point)
- multi-view rich (when available; some cameras in `O`, others in `H`)
- synthetic multi-view (Kubric-style; optional, EH-4 below)

### Forward pass

```
W0      = E(O)                              // encoder -> world asset
S_h     = G(W0, tau_h)                      // time query -> splats at tau_h
Ihat_h  = R_fixed(S_h, c_h)                 // camera query -> rendered image
```

Observation camera/time may enter once at encoder ingestion, e.g. as
Pluecker-ray, pose, intrinsics, and time embeddings on observation
tokens. After ingestion, learned world-side computation reads `W0` and
query time only. Query camera is consumed only by the fixed rasterizer.

### Loss

```
L_pred(theta) =
  E_{(O,H,Q) ~ D_var} [ sum_{h in H} ell(Ihat_h, I_h) ]

ell(Ihat, I) = lambda_L1 * ||Ihat - I||_1
             + lambda_LPIPS * LPIPS(Ihat, I)
```

Add a rate/minimality pressure on `W0` when it is not already enforced
architecturally:

```
L_v1 = L_pred + beta * Rate(W0)
```

`Rate(W0)` is deliberately implementation-agnostic: it can be a KL term,
MDL/code-length estimate, hard token/channel bottleneck, quantized-rate
penalty, entropy model, or export-size constraint. Its job is not to
encode a semantic invariant. Its job is to choose a compact, useful
representative from a predictive equivalence class.

That is the entire pretraining contract. No decorative auxiliary loss,
no agreement term, no adversarial head, no JEPA term, no diffusion term.

---

## Support-qualified vocabulary

Use this vocabulary instead of claims about latent purity.

### Query support

```
S_train := supp(D_var^Q)          # train query support
S_dep   := deployment query region
Delta_S := S_dep \ S_train        # extrapolation region
```

Anything proved on `S_train` is identified by the training contract.
Anything claimed on `Delta_S` is extrapolation.

### Predictive map

For a query `q = (c, tau)`:

```
Pi_S(W) := { p_theta(. | W, q) : q in S }
```

For a deterministic renderer, `Pi_S(W)` is just the family of rendered
images over supported queries. This is the clean object the baseline
identifies; raw token identity is not.

### Support-relative predictive equivalence

```
W equiv_S W'  iff  Pi_S(W) = Pi_S(W')
```

This is the right formal replacement for "`W_s` and `W_r` agree." Do not
write `W_s = W_r` as a target. Write `E(O_s) equiv_S E(O_r)` on the
shared held-out query support.

### Gauge equivalence

```
W ~_g W'  =>  W equiv_S_phys W'
```

`~_g` covers permutation, reparameterization, canonical-frame choice, and
other benign internal relabelings. The identified object is a quotient
modulo gauge, not a globally canonical latent.

### Minimal predictive representation

`W` is predictively sufficient on `S` when:

```
p(x | W, q) = p(x | scene, q)    for all q in S
```

`W*` is minimal predictive when:

```
W* in argmin R(W; O)
     subject to W predictively sufficient on S
```

`R` is a rate surrogate: `I(W;O)`, KL-to-prior, code length,
quantized-rate, hard export cap, or whatever implementation actually
uses. This separates sufficiency from held-out prediction, minimality
from rate control, and minimal predictive posterior semantics from
generic VAE compression.

### Predictive null bits

A latent component `z` is null on `S` if:

```
Pi_S(W, z) = Pi_S(W)
```

Photometric prediction alone does not remove null bits. Rate/minimality
is what makes useless observation-specific residue, camera hashes, or
ignored cancellation remnants costly.

### Observation-side vs query-side camera/time

Use two names:

- `(c_o, tau_o)` : observation-side camera/time, allowed once at encoder
  ingestion.
- `(c_q, tau_q)` : query-side camera/time, the deployment interventions.

Contract:

```
W    = E(O; c_o, tau_o)
S    = G(W, tau_q)
Ihat = R_fixed(S, c_q)
```

Correct rule: **query camera enters only at `R_fixed`; observation camera
enters once at ingestion and nowhere after.**

### Export purity

**Export purity:** the deployed render call reads only `(W, c_q, tau_q)`
plus fixed model/rasterizer weights. If any encoder feature, source pose,
cache, teacher state, or training-side tensor is required, export purity
failed.

---

## Predictive quotient on supported queries

Two world assets are equivalent on a support when they induce the same
predictive map:

```
W equiv_S W'  iff  Pi_S(W) = Pi_S(W')
```

This is behavioral equivalence on supported queries. It is not latent
purity, token equality, a no-cancellation theorem, or a proof that `W`
contains only scene state. The predictive loss identifies an equivalence
class under `equiv_S_train`; rate/minimality chooses among representatives.

**Corrected claim.** For any target property whose only observable
consequence is predictive behavior inside `S_train`, minimizing `L_pred`
can enforce only that behavior modulo `equiv_S_train`. Anything outside
`S_train`, or invisible to the renderer on `S_train`, is not identified by
the loss. Any desired preference among behavior-equivalent worlds must
come from export purity, architecture, or rate/minimality pressure.

**Subsumption, patched.**

```
P is subsumed by L_v1 on S
  iff every rate-minimal minimizer of L_v1 satisfies P on S.
```

The `rate-minimal` qualifier blocks null bits and cancellation residue
from being falsely ruled out. The `on S` qualifier blocks supported
behavior from being mislabeled global identifiability.

**Sketch.** `L_pred` penalizes render error at held-out
`q_h = (c_h, tau_h) in Q`, and `D_var` is supposed to cover the
deployment region being claimed. A minimizer must match target behavior
on `S_train`. But if two assets render identically on every supported
query, the loss cannot prefer one because it is "purer" or has cleaner
internal tokens. The only principled selector left is minimal commitment:
lower rate, smaller export, simpler posterior, or a harder bottleneck.

### What this removes as permanent training machinery

| Target | Traditional mechanism | Framing_3 treatment |
|--------|-----------------------|---------------------|
| Camera leakage prevention (F1) | Adversarial head on `W` | If leakage changes supported-query renders, `L_pred` penalizes it. If it does not, the predictive quotient cannot see it; rate/minimality makes wasted camera storage costly. Keep leakage probes diagnostic. |
| JEPA agreement (framing_2 C4) | Student-teacher distillation | `W_s = E(O_s)` and `W_r = E(O_r)` agree when `W_s equiv_S W_r` on shared supported queries, modulo gauge/permutation. Token equality is not required and is usually the wrong target. |
| 4D consistency (framing_2 C2) | Cross-view / cross-time loss | If `D_var` includes same-time/different-camera and same-camera/different-time held-out pairs, inconsistency is visible to `L_pred`. Outside that support, no claim. |
| World-agreement collapse (F7) | EMA teacher, diversity term | No agreement loss means no agreement-collapse objective. Constant `W` is rejected only insofar as it cannot predict held-out targets under `D_var`; rate alone cannot rescue a bad predictor. |
| Scene / camera / time factorization | Dedicated factor losses | Factorization is not identifiable unless it changes supported-query prediction or export rate. The baseline commits only to the tensor boundary: query time to `G`, query camera to `R_fixed`. |
| Cheating splats (F2) | Crop-perturb aux loss | Crop-as-extrinsic belongs in `D_var`. A representation that renders correctly under supported crop/query perturbations is behaviorally acceptable, even if its internals are inelegant. |
| Rolling noise / denoising regularizers | Variance schedules on visible frames | Omission is the supervision. Noise on visible frames with visible targets is autoencoding unless it improves held-out prediction or rate. |

All of the above remain available as **diagnostics** (e.g., fit an
adversarial probe post-hoc and check whether it succeeds above chance);
they do not ship as permanent gradient sources.

---

## What the quotient does NOT enforce (load-bearing constraints)

`L_pred` cannot enforce the architecture, the support assumptions, or
the representative selector. These constraints stay explicit.

### R0 — Rate/minimality selects the representative

The predictive quotient gives behavior classes, not clean latents.
Without rate/minimality, `W` can carry arbitrary supported-query
behavior plus unused junk, memorized camera hints, or fragile
cancellation. The baseline must impose minimal commitment through some
combination of export size, token/channel count, quantization,
entropy/MDL, KL, or hard capacity limits.

**Implementation check.** Plot rate-distortion and export size against
held-out prediction. A larger `W` must buy prediction quality,
calibration, or deploy utility; otherwise it is capacity leak.

### C1 — Self-sufficient decode

Deploy-time `R_fixed(G(W, tau), c)` reads nothing beyond `(W, c, tau)`. No
per-frame cache, no encoder residual, no side channel.

**Implementation check.** Trace every tensor used by the deployed render
call. Each must live in `W`, in the query, or in the rasterizer's fixed
state.

### C3 / C9 (merged) — Generative pressure from omitted observations

For every training step, target is not input. The encoder never sees the
target; `L_pred` is computed on `H not_subset O` by construction of
`D_var`.

**Implementation check.** For each training step, verify exact target
observations are not in the encoder input. Same source time is allowed only
for a genuinely different query (different camera, crop-as-extrinsic, changed
intrinsics) whose target pixels were not already visible to the encoder. If
the target observation itself overlaps the input, the regime collapses to
autoencoding.

### C5 — Rasterizer is the output boundary

No parallel decode paths emitting images. Splats commit at the
rasterizer; nothing else. Query camera `c` enters here and only here.

### C6 — Query boundary is split: time to `G`, camera to `R_fixed`

Learned scene generation has signature `G(W, tau)`. The fixed renderer
has signature `R_fixed(S_tau, c)`. No learned module receives query
camera before the rasterizer. No side conditioning, no per-frame
features, no auxiliary inputs.

### C7 — No prebake dependence at inference

GT pose is training-only. Predicted camera or user-supplied camera at
inference.

### C2 (demoted) — 4D consistency

Handled by `L_pred` when `D_var` covers the relevant `(c, tau)`
cross-pairs. No longer needs its own loss term. Still needs a support
check.

### C4 (demoted) — Joint world-token target

Replaced by predictive agreement under `~_Q`. `W_s` and `W_r` do not
need token equality; they need matching predictive behavior on the
supported query set modulo gauge/permutation. No dedicated agreement
loss in the baseline.

### C8 (conditional) — Coarse-to-fine stays world-space

Only applies if a densifier is added. Baseline does not add one.

---

## Unavoidable physics priors (architectural commitments, not priors-as-loss)

These are architecture-level, not loss-level:

1. **Differentiable splat rasterizer.** Perspective projection is
   physics. Splats commit at this boundary.
2. **Set-permutation-invariant splats.** World is a set of 3D
   primitives; permutation is gauge.
3. **Time monotone forward.** If streaming, causality is physics.
4. **`SE(3)` camera.** Measurement structure, not a parameterization
   choice.
5. **Observation pose/time consumed once at encoder ingestion.** The
   canonical pattern: for each observation `(I_t, C_t, tau_t)`, add a
   Pluecker-ray embedding (direction + moment, computed per patch
   from intrinsics/extrinsics) and a time embedding to the patch
   tokens. After the ingestion block, no encoder-side layer reads
   observation camera or time except through world-side tokens. Query
   time may enter `G(W, tau)`. Query camera enters only the fixed
   renderer `R_fixed(S_tau, c)`. No learned camera-conditioned
   pre-render branch. This is how C1/C6 are enforced architecturally:
   by tensor-flow contract, not by loss. Verified by the export-purity
   tripwire below.

These are not auxiliary losses. They are in the model shape. They age
with scale.

---

## Evaluation: how to judge any proposed mechanism

For every proposed architectural change, auxiliary loss, or mechanism,
run the checklist:

1. **Does it encode unavoidable physics or the inference contract?**
   If yes -> earned. Stop.
2. **Does it select a lower-rate representative without changing the
   predictive target?** If yes -> it belongs in rate/minimality, not as
   a new semantic loss.
3. **Is its target only supported-query behavior already covered by
   `L_pred` and `D_var`?** If yes -> decorative. Drop or demote to
   diagnostic.
4. **Is it a measurement tool that carries no gradient?** If yes ->
   free. Keep as diagnostic.
5. **Is it an optimization accelerator with a decay schedule that
   retires it by end of pretraining?** If yes and a concrete
   convergence measurement justifies it -> admit as escape hatch.
6. **Is it none of the above?** -> Does not ship.

If a proposal fails these gates and is still being defended, the defender
is anchoring on the mechanism rather than the invariant.

---

## Escape hatches (admitted only by measurement)

Not in the baseline. Admitted only when a specific observable signal
demands them. Labeled with trigger, admission rule, and retirement
condition.

### EH-1. Geometry Forcing / VGGT alignment

**What.** Auxiliary loss aligning encoder intermediate features with a
frozen geometric foundation model (e.g., VGGT). Angular alignment +
scale alignment in feature space.

**Paper.** Wu et al. 2025, "Geometry Forcing: Marrying Video Diffusion
and 3D Representation for Consistent World Modeling" (arXiv 2507.07982).

**Trigger.** Geometry is visibly unstable after `N` training steps
while `L_pred` plateaus — i.e., the photometric loss is satisfied by
textures that do not correspond to physical geometry. Signal: render
quality at `H` is high but depth/normals are incoherent.

**Admission rule.** Decay schedule
`lambda_geom(t) = lambda_0 * exp(-t / tau_geom)` with
`lambda_geom(T) approx 0` by end of pretraining. Never permanent.

**Retirement condition.** Deployed model must train without this term.

### EH-2. Diffusion-as-loss (SJC / DiffRep)

**What.** Frozen image/video diffusion teacher scores held-out renders
via Jacobian pullback from pixel space into splat/token parameter space.
Rigorous pullback (Score Jacobian Chaining), not naive SDS.

**Trigger.** `L_pred` saturates at a level where L1 + LPIPS can no longer
distinguish plausible from implausible renders on the sparsest budgets
in `D_var`. Averaging losses are hedging; need off-manifold
supervision.

**Admission rule.** Post-training only, not pretraining. Weighted as a
prior, not a GT replacement. Held-out-camera renders only.

**Retirement condition.** None. Kept if it helps at post-training.
Excluded from pretraining regardless.

### EH-3. Residual world-token stochastic prior

**What.** Full stochastic `W` is semantically right for ambiguous
observations, but optional for v1. Baseline exports deterministic
`W0`, interpreted as a point estimate / posterior summary. EH-3 adds
`p_theta(W_hi | W0, O_partial)` over residual world detail; `W_hi` is
a sampled refinement and noise goes over missing world detail only.

**Trigger.** Ambiguity in `O_partial` is provably multimodal — two
qualitatively different worlds both explain `H` equally well, and the
deterministic `W0` picks a mean that matches neither.

**Admission rule.** Residual-only. Inference defaults to the mean/MAP
summary of `W_hi` conditional on `W0, O_partial`, preserving the
deterministic deploy contract unless stochastic export is explicitly
enabled.

**Retirement condition.** None if multimodality is real. But demonstrate
multimodality before adopting.

### EH-4. Synthetic / multi-view observation budgets

**What.** Add synthetic multi-view clips (Kubric, etc.) and real
multi-view clips (when available) as additional samples in `D_var`.
Optional 3D flow or depth labels become *rich-budget auxiliary targets*,
not required losses.

**Trigger.** None required. Strictly additive curriculum. No trigger,
no admission friction.

**Admission rule.** Just another point in the observation-budget
distribution. No architectural or loss change.

**Retirement condition.** None.

### Notes on the escape hatches

EH-4 is free. EH-1 is cheap and time-bounded. EH-2 and EH-3 are real
cost; add them only with measurement in hand. Stacking more than one
simultaneously is a slop smell — if the baseline needs multiple
escape hatches, the baseline is wrong and the failure mode is
architectural, not optimization.

---

## Diagnostic tripwires (measurement-only, no gradient)

Three tripwires to log from day one. They carry no gradient; they are
evaluation tools that tell you whether the baseline is actually
satisfying the contract it claims to. Every one of them should be
green before you ship, and any that turns red is strictly more
informative than the training loss curve.

### T1. Camera-leakage probe

**What.** Fit a small adversarial head post-hoc on frozen `W` (or
on a canonical ordered readout `Gamma(W)` if `W` is a slot set) to
predict camera yaw / pitch / crop center / focal. Fit with `W`
frozen, no gradient flowing back.

**What it measures.** Whether camera information is recoverable
from `W` above chance. Framing_3 does not claim latent purity:
unobservable leakage is outside the predictive quotient. But recovery
at high accuracy is a symptom that the encoder is routing camera into
world-token capacity, which is at least a rate/waste issue and possibly
a sign that `D_var` is not covering enough of the camera manifold.

**Interpretation.** High probe accuracy -> broaden `D_var` camera
coverage (crop, multi-view, synthetic-multi-view) before considering
an adversarial loss (framing_2 territory).

### T2. World-dependence test

**What.** At eval time, render from `W` under each of:
- full `W`,
- 30% tokens randomly dropped,
- shuffled child-index embeddings,
- zeroed time-basis coefficients,
- `W` from a different clip (fully out-of-distribution world).

**What it measures.** Whether `W` is structurally load-bearing or
whether the decoder is hiding information in a few degenerate
slots / child indices / time coefficients. If dropping 30% of tokens
barely changes the render, capacity is unused; if zeroing time
coefficients barely changes the render, the decoder is not using the
dynamic pathway; if a different clip's `W` renders plausibly, the
decoder has memorized a prior rather than consuming evidence.

**Interpretation.** Each sub-test pinpoints a different pathology.
Run them as a matrix, not a single number.

### T3. Export-purity test

**What.** At eval, force the render call to receive only `(W, c, tau)`.
Explicitly disable or null every training-time tensor that is not one
of these three (e.g., encoder intermediate features, per-frame caches,
teacher-network state, ingestion-side embeddings). Render.

**What it measures.** Whether C1 (self-sufficient decode) actually
holds. If any pathway silently depends on something outside `(W, c, tau)`,
performance will drop when that pathway is nulled. The drop quantifies
the C1 violation.

**Interpretation.** Any performance delta greater than noise is a
contract violation that must be fixed architecturally before anything
else ships.

### Notes on tripwires

- All three are measurement-only and free to run at any eval interval.
- None of them should ever enter the training graph (that's mistake
  #17 + #18: measurement promoted to loss).
- If you add any escape hatch (EH-1 through EH-4), re-run all three
  after adding and confirm nothing regressed.
- T3 is the hard C1 check. If T3 fails, no other number matters.

---

## Relationship to framing 1 and framing 2

| | Framing 1 | Framing 2 | Framing 3 |
|-|-----------|-----------|-----------|
| View | Information-theoretic | Contract + mechanisms | Bitter-lesson predictive quotient |
| Primary use | Deriving loss bounds | Auditing architectures | Default baseline |
| Commitments | Gauge-invariant `W`, MI bounds | 7 C-constraints + dedicated losses | export contract, supported-query prediction, rate/minimality |
| Mechanism count | N/A | 3+ (agreement, cross-view, photometric) | 1 predictive signal + representative pressure |
| Derives losses | yes | partial | from predictive quotient + rate |
| Audits arch | partial | yes | via evaluation checklist |

Framings 1 and 2 remain useful:

- **Framing 1** when you need to derive a *new* loss term or argue about
  what a proposed loss bounds. Mutual-information language cuts through
  ambiguity about what a term supervises.
- **Framing 2** when you need to audit whether an implementation
  silently reads frame-local state, or whether a densifier leaks. The
  contract vocabulary catches violations that the predictive quotient
  alone glosses over.
- **Framing 3** is the default for proposing anything new. Start here.
  Drop to 1 or 2 only when framing 3 can't discriminate.

---

## Paper grounding (verified)

Real papers this framing draws on:

- **Geometry Forcing** (Wu et al. 2025, arXiv 2507.07982). Real. EH-1.
- **Score Jacobian Chaining / Diffusing Differentiable Representations**
  (Wang, Du, Mordatch 2022; Savani et al. 2024). Real. EH-2 math.
- **Self-Forcing / Rolling Forcing / Diffusion Forcing.** Real.
  Relevant only if the project later moves to AR over rolling state.
  Not used in the baseline.
- **V-JEPA 2.** Real. Its masked-prediction objective is a
  decoding-expansion proof-of-concept; useful as a feature-alignment
  target (EH-1 variant), not as an architecture to adopt wholesale.
- **DGS-LRM** (Lin et al. 2025, arXiv 2506.09997). Real. Trains on
  synthetic multi-view + 3D flow labels to produce per-pixel
  deformable splats. Out of framing_3's data assumption; its Kubric
  pipeline is usable as EH-4.

Citations from exploration-style LLM outputs that could not be
verified (Lyra 2.0, PERSIST, Relax Forcing, Causal Forcing, ReconPhys,
V-JEPA 2.1) are excluded until verified. See mistake #19 in
`../meta_philosophy/how_to_think_about_architecture.md`.

---

## Slop-prevention checklist

Before writing a new loss term, a new head, a new stage, or a new
regularizer, answer in order:

- [ ] Does it encode unavoidable physics or the inference contract?
- [ ] Is its target already covered by supported-query prediction?
- [ ] Is it really rate/minimality rather than a new semantic loss?
- [ ] Is it a diagnostic carrying no gradient?
- [ ] Is it a decaying-to-zero accelerator justified by a measurement?
- [ ] Are all cited papers verified?
- [ ] If it ships, what's its retirement condition?

If the answer is "no / yes / no / no / no / none," it does not ship.

---

## What this framing gives up

- **Optimization comfort.** The held-out predictive loss on variable
  budgets is harder to optimize than a menu of auxiliary losses. We
  accept curriculum-on-budget (data-distribution schedule) but not
  auxiliary-loss-stacks as the stabilizer.
- **Implicit intermediate checkpoints.** An agreement loss or an
  adversarial probe gives you a per-step number to watch. The
  predictive-quotient baseline only gives you held-out render quality
  plus rate-distortion behavior. You can
  still fit probes as diagnostics, but they are not training signals.
- **Credit assignment within `W`.** The baseline does not tell you
  which part of `W` carries scene vs. camera vs. time. If you need
  that answer (you probably don't), it lives in framing 1 or 2.

What the framing gains is that the deployed model has exactly one
predictive contract to satisfy and one representative pressure to keep
`W` useful. No decorative auxiliary has to be carried into production.
No escape hatch without a named trigger and retirement condition.

---

## Canonical one-paragraph description

DynaWorld's baseline is a single encoder `E` producing an exported world
asset `W0 = E(O)` from a variable-budget observation set `O`, a
time-conditioned scene generator `S_tau = G(W0, tau)`, and a fixed
differentiable rasterizer `R_fixed(S_tau, c)` at a queried camera `c`.
Training uses held-out predictive supervision (L1 + LPIPS initially) on
omitted observations `H not_subset O` sampled from `D_var`, plus an
explicit rate/minimality pressure on `W0` when capacity is not enforced
architecturally. The predictive loss identifies worlds only up to
behavioral equivalence on supported queries; it does not promise latent
purity, token equality, or absence of cancellation. Auxiliary mechanisms
(VGGT alignment, diffusion-as-loss, stochastic residual prior) exist only
as measurement-triggered escape hatches with retirement conditions.

That paragraph should fit on one slide. If it does not, framing has
drifted.
