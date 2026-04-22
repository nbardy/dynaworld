# Framing 2 — Self-Sufficient World Tokens and the Generative Reconstruction Contract

Second framing of the novel-view problem. Where framing 1 used information
theory (channels, mutual information, gauge symmetry), this framing uses
a contract-based view: what must a world-token representation *mean* for
the task to be well-posed?

Crystallizes vocabulary and design contract from a thinking session. The
structure is deliberately glossary-like: terms first, philosophy derived
from terms, constraints as the architecture-facing contract.

Honor the phrasing. These are not my words paraphrased — they are the
user's working concepts surfaced and named.

---

## Terms

### Self-Sufficient World Tokens

A representation W such that, given only W and a query `(C_q, t_q)`, the
renderer produces the correct image. No per-clip hidden state, no
encoder-side persistence, no additional latent carried outside W
participates in the decode.

```
render(W, C_q, t_q)  →  image
```

If the decode pipeline reads any information not contained in W or the
query, W is not self-sufficient.

### Decoding Expansion (the VAE-like view)

The model compresses input V into W, and decoding reproduces V's content
under different camera / time queries drawn from the *encoder-covered*
distribution. Can only render what was in V's encoder path.

Decoding expansion is what a VAE-over-video does. It is not a world model.
It compresses information; it does not produce new information.

### Generative Reconstruction

The model produces W from input V, and decoding reproduces content that
was **not** in V's encoder path. The tokens hallucinate what would have
been there for cameras, times, or regions the input never covered.

Generative reconstruction is the contract a world model must satisfy. The
test: can the model reconstruct held-out chunks, cameras, or times that
were excluded from the encoder input? If yes, it is generative. If only
encoder-path reconstruction works, it is decoding expansion.

### Frame-Local State

Any decode-time information tied to a specific input frame that is not
contained in W. The enemy of self-sufficiency.

If any rendered frame's quality depends on frame-specific hidden state
(residual from the encoder, frame-specific attention caches, frame-indexed
adapters, per-frame offsets), then W is not the full representation; W is
a *part* of the representation and the rest is frame-local.

Frame-local state is a silent contract violation.

### 4D Consistency

The joint property that W is simultaneously:

- **Camera-invariant**: the same W, queried at different `C_q` at the same
  t, produces geometrically consistent renders.
- **Time-identity-invariant**: the same W, queried at different t, reflects
  consistent scene identity (objects do not appear / disappear; persistence
  is preserved). Time-dependence lives in the decoder, not in W.

These are two symmetries of the same underlying scene, not two independent
properties. Splats are 4D because they are decoded with a time-conditioned
decoder from the same W; the "4D" refers to the decoder's query space, not
W's internal structure.

### Persistent 3D Objects

Splats (or scene primitives) that exist in a canonical 3D frame and are
consistent across all rendering cameras. "Persistent" is spatial (across
cameras); "3D" is structural (not tied to any one pixel grid).

Good splats *are* persistent 3D objects. This is the physics, not a
modeling choice. A model that produces non-persistent splats (different
geometry per camera) is wrong about the world.

### Shared World-Token Target

The property that different encoder paths of the same scene — different
chunks, different crops, different masked subsets — must converge on the
same W.

JEPA-style: if two chunks of the same clip are independently encoded, the
resulting W's should agree on the shared scene content. This is the
supervision hook for self-sufficiency.

### Observation Set `O`

The subset of frames, crops, and poses given to the encoder for one
training example. Distinct from the full clip on disk; the encoder only
sees `O`, not the ground-truth frames outside it.

### Observation Budget

How much evidence `O` contains. One frame, one short chunk, many chunks,
or many crops are all different budgets over the same underlying scene.
Budget is a training-time knob; at inference it is whatever the user
provides.

### Held-Out Target Set `H`

Frames that are supervised during training but were **not** shown to the
encoder. `H` is the teacher signal for self-sufficiency: if `W` inferred
from `O` can render `H` correctly under held-out `(C, t)` queries, the
encoder has done conditional world completion rather than autoencoding.

### Rich-Context / Sparse-Context World Tokens (`W_r`, `W_s`)

Two world-token solutions of the same scene inferred at different
observation budgets. `W_r = E(O_r)` with a rich budget; `W_s = E(O_s)`
with a sparse budget. Same underlying world, different evidence — the
training contract demands they agree on behavior.

### Variable-Budget Regime

A single-stage training distribution in which the observation budget is
randomized per step from sparse to rich. Collapses chunk-swap,
crop-as-extrinsic, JEPA-masking, and held-out-camera losses into one
recipe: sample `(O_s, O_r, H)` from the same clip, export `W_s` and
`W_r`, and drive both to explain `H` under held-out `(C, t)`.

The variable-budget regime is the cleanest single-stage instance of the
Generative Reconstruction Regime below. It is what removes "two-stage
multi-clip pretrain + single-clip finetune" as a necessity: a single
encoder trained on randomized budgets already learns the partial →
world mapping.

### Conditional World Completion

The training regime where the encoder sees partial observations and the
exported `W` must explain omitted real observations at arbitrary
`(C, t)`. Deterministic: one input, one world. This is the target
regime for the MVP.

Distinct from stochastic world modeling: conditional completion predicts
a single best world given the evidence; stochastic modeling predicts a
distribution.

### Stochastic World Modeling

An explicit distribution `p_θ(W | O_partial)` over world tokens, trained
via diffusion / flow / AR. Necessary only when ambiguity in `O_partial`
justifies multimodal completions — i.e., when one-best-world loses too
much detail. For the MVP this is deferred; conditional completion comes
first.

### Render-Agreement

The equality check used to enforce `W_s` and `W_r` agree, measured at
the render output rather than at the token level:

```
Render(Decode(W_s, τ_h), c_h)  ≈  Render(Decode(W_r, τ_h), c_h)
```

for held-out `(c_h, τ_h)` drawn from `H`. This is the correct
replacement for raw token-L2 agreement when `W` is set-permutation-
invariant. It also covers framing_1's C-equivariance constraint: any
pair of worlds that renders identically under every `(c, t)` are
behaviorally equivalent, which is what we actually care about.

Token-level agreement is only valid when tokens are explicitly
canonicalized (matched slots, sorted readout, learned canonicalizer).

### Generative Reconstruction Regime

The training regime that enforces generative reconstruction (vs decoding
expansion). At minimum, it must include loss terms computed on content
that was **not** in the encoder's input path. Options:

- Chunk swap: encode chunk A, reconstruct chunk B.
- Held-out camera: render at a perturbed camera, score with a teacher or
  consistency loss.
- Rolling noise: perturb or mask rolling temporal windows and require the
  rest to reconstruct.
- JEPA masking: mask a portion of input, require the remaining encoded
  part to predict the masked target.

Without at least one such term, the model cannot be distinguished from a
VAE at training time.

### Rolling Noise Regularizer

A training-time perturbation applied to a rolling temporal window in the
clip, forcing the model to be robust to incomplete or corrupted temporal
information.

Conceptually between chunk swap (discrete) and denoising diffusion
(continuous). The rolling nature means the model sees variable windows of
corrupted input at variable positions in the clip.

---

## Philosophy on Modal Design

### 1. Self-sufficiency is the primary contract

The value of a representation is whether it can render anything downstream
with no additional state. If the rendering pipeline needs side-channel
info beyond W, W is not a world representation — it is a partial latent
that only works in concert with other machinery.

If the model cannot commit to a self-sufficient W, it cannot claim to have
learned the world. It has only learned to decode a specific encoder path.

### 2. Self-sufficiency implies generativity

If W must render any `(C_q, t_q)` but the input V only covered some
cameras and times, then W must generate unseen content. This is not
compression; it is generative completion.

Consequence: the model is **not** a VAE over V. It is a generative mapping
`V → W` where W covers more than V ever could. The training objective
must reflect this. A purely reconstructive loss on V's encoder path
cannot produce self-sufficient W.

### 3. Persistent 3D is physics, not a modeling choice

A rendered scene from multiple cameras is one scene. Splats are persistent
3D objects because the world is one persistent 3D object. The model must
match this, not impose its own factorization.

The test: same W, rendered from K different cameras, must produce
geometrically consistent K views. A model that produces K inconsistent
views is not failing a loss — it is wrong about physics.

### 4. 4D consistency is one contract, not two

Camera invariance and time-identity invariance are two symmetries of the
same scene. Enforcing one without the other is asymmetric for no physical
reason.

A model with strong camera invariance and weak time identity is as broken
as one with strong time identity and weak camera invariance. Neither is
closer to a world model than the other.

### 5. Frame-local state is leakage

Every piece of decode state must be either in W, in the query `(C_q, t_q)`,
or in the renderer itself. Nothing else.

If a specific frame's reconstruction uses frame-specific hidden info —
residual from the encoder, a frame-indexed adapter, a per-frame correction
head — the model has silently relaxed self-sufficiency and will not
generalize to novel views / times at inference.

Frame-local state is the silent version of camera leakage. The difference
is that camera leakage is detectable by adversarial probe; frame-local
state may escape that probe while still breaking the contract.

### 6. Generative reconstruction requires training on what wasn't encoded

The only way to distinguish a VAE-like model from a generative one is
whether it is trained to reconstruct inputs that were **not** in its
encoder path.

Chunk swap is the cleanest enforcement. Held-out cameras via diffusion-as-
loss is the second cleanest. Rolling-noise or JEPA-masking regimes force
the same property via different mechanics. Without at least one such term,
self-sufficiency cannot be verified at training time.

### 7. Multi-clip pretraining is a legitimate curriculum

A curriculum where the model first learns "world" from many aligned clips
(multi-view by accident — different clips of similar content), then is
fine-tuned to produce that world from a single clip, may be more stable
than trying to learn generativity directly from single-clip data.

Two-stage here is acceptable because stage 2 is *fine-tuning a known-good
mapping*, not *patching a broken representation*. This is different from
the two-stage leakage trap (where stage 1 bakes in a failure).

### 8. Shared world-token target is a non-adversarial probe

If we enforce that two independent encoder paths of the same scene
produce the same W, we have a non-adversarial supervision signal for
self-sufficiency. This is JEPA's contribution adapted to our setting.

Not the same as camera-leakage prevention. Different chunks of the same
clip share scene content but differ in camera, time, and coverage.
Agreement on W across them means W has abstracted past those differences
to the invariant content — which is exactly what "world" means.

### 9. Observation omission is the supervision move, not noise

Noise on visible frames with visible targets still lets the model copy
its encoder path — denoise what you saw, predict what you saw. The
generative pressure comes from *which observations are missing from the
encoder*, not from how corrupted the visible ones are.

Noise is an augmentation. Omission is the contract. Rolling-noise,
dropout, and masking regularizers only earn their cost when the target
set includes observations the encoder did not see.

### 10. Deterministic conditional completion before stochastic generation

Three regimes, in order:

1. **Autoencoding reconstruction.** Encoder and decoder see the same
   observations. Underconstrained on the axes we care about.
2. **Conditional world completion.** Encoder sees partial; decoder
   must explain held-out real observations at arbitrary `(C, t)`.
   Deterministic, one-best-world.
3. **Stochastic world modeling.** Explicit distribution over `W` given
   partial evidence. Necessary only when ambiguity justifies multimodal
   outputs.

The MVP target is (2), not (3). Jump to (3) only after (2) saturates
and the residual ambiguity is demonstrably multimodal. Do not reach for
diffusion over splats because "reconstruction is underconstrained" —
that's a category error. Reconstruction is underconstrained; conditional
completion is what fixes it.

### 11. Render-agreement beats token-agreement

Student-teacher distillation between `W_s` and `W_r` is the
non-adversarial probe for self-sufficiency. The distance metric matters.

World tokens are a set; raw token L2 is the wrong distance under
permutations. Agreement at the render output — same `(c, τ)`, same
rendered image, up to tolerance — is both permutation-safe and directly
tied to the behavior we care about. Use token agreement only where
tokens are explicitly canonicalized.

---

## Constraints (architecture-facing)

These are the conditions the architecture must satisfy. They are distinct
from F1–F7 (failure modes) — constraints are design requirements,
failures are what happens when constraints are violated.

### C1. Self-Sufficient Decode

W alone (with query `C_q, t_q`) must produce correct renders. Architecture
must have no frame-local decode state.

**Implementation check**: trace every tensor used in `render(...)`.
Everything must be either inside `W`, inside the query, or inside the
renderer. Nothing else.

### C2. 4D Consistency

Rendering the same W from `(C_a, t_i)` and `(C_b, t_i)` must be camera-
consistent. Rendering the same W at `(C_q, t_i)` and `(C_q, t_j)` must
reflect time-consistent scene identity (object persistence, no
hallucination of scene cuts).

**Implementation check**: a consistency loss term must exist in the
training objective, covering both axes.

### C3. Generative Reconstruction Test

The architecture must be trainable to reconstruct inputs **not** in its
encoder path. If it can only reconstruct encoder-path inputs, it is a
VAE, not a world model.

**Implementation check**: at least one loss term in training must compare
decoded outputs against GT content that was never fed to the encoder.
Chunk swap, held-out camera, JEPA masking, or rolling noise all qualify.

### C4. Joint World-Token Target

Different encodings of the same scene (different chunks, crops, masked
subsets) must produce W that agree on the shared content. Architecture
must support a loss that enforces this agreement.

**Implementation check**: two encoder passes on overlapping content must
produce W's that can be compared via a concrete loss (MSE, KL, cosine on
shared subspace, etc.).

### C5. Rasterizer Is the Output Boundary

Splats commit at the rasterizer. No other decode path exists. This is the
only place W is allowed to be "made explicit."

**Implementation check**: no parallel decode paths that emit images
bypassing the rasterizer. No "auxiliary head" that renders via a
different mechanism.

### C6. Query Boundary Is SE(3) × R

The only decode-time inputs besides W are a camera (SE(3)) and a time
(R, or a discrete frame index). No other conditioning at decode.

**Implementation check**: the decode function signature is
`decode(W, C, t)` with no other arguments.

### C7. No Prebake Dependence at Inference

If the model needs ground-truth camera/pose to render, it is a
pose-conditioned decoder, not a world model. Predicted camera is fine
(emergent); required ground-truth camera is not.

**Implementation check**: at inference, disable the prebake pipeline
entirely. If the model still renders, the contract is met.

### C8. Coarse-to-Fine Stays World-Space

If a densification stage exists (`W_0 → W_1`), both levels must be
persistent world assets. The densifier may read optional training-only
evidence, but the deployed render call must consume `W_1` alone under
`(C_q, t_q)`. Any densification path that produces frame-local rescue
features consumed by the renderer at deploy time violates the contract.

Valid: `W_1 = Densify(W_0, optional training-only evidence)`.
Invalid: `Render(W_1, per-frame-feature_f, C_q, t_q)` where
`per-frame-feature_f` is not re-derivable from `W_1` and the query.

This is C1 applied recursively to every stage of the pipeline. A
coarse world + a finer world is meaningful. A coarse world + a
frame-local rescue net is not.

**Implementation check**: for every stage that emits or refines world
state, trace the render call at deploy time. Everything consumed must
live inside the current stage's world tokens or inside the query.

### C9. Generative Pressure Comes From Omitted Observations

At least one training loss term must be computed on observations the
encoder did not see. Pure denoising / reconstruction of visible content
does not satisfy this. The variable-budget regime is the cleanest
satisfier; chunk-swap, JEPA masking, and held-out-camera losses are
valid specializations.

**Implementation check**: for each loss term, identify which tensors the
encoder received as input and which it is being scored against. At least
one term must have input-set and target-set disjoint (or target ⊄ input).

---

## Open Questions Added By This Framing

These do not belong in the constraints above — they are unresolved tensions
the framing surfaces.

### Q1. Is self-sufficiency compatible with finite-capacity W?

There is always scene complexity exceeding W's capacity. How does the
model degrade gracefully at capacity?

Candidate answer: tokens represent a posterior over possible scenes, not
a deterministic scene. Detail scales with scene coverage in input; unseen
content is drawn from a prior conditioned on what was seen. This makes
the model a conditional generative model, not a deterministic encoder.

### Q2. Does generative reconstruction require an explicit generative objective?

**Partially resolved — see philosophy bullet 10 and C9.**

The variable-budget regime with held-out-observation loss + render-agreement
gives implicit generativity without an explicit diffusion objective.
That is the MVP target (regime 2, conditional world completion).

Explicit stochastic generation (regime 3) is deferred until conditional
completion is shown to saturate on ambiguity. If needed, the cleanest
placement is a residual world prior `p(W_hi | W_0, O_partial)` — noise
goes over missing world detail, not over the whole scene from scratch
and not over rendered pixels.

### Q3. Multi-clip pretraining + single-clip fine-tune vs. single-stage

Is two-stage cheaper and more stable? Or does it add complexity without
proportional gain?

Key distinction: two-stage is safe when stage 2 is fine-tuning a known-
good mapping. Two-stage is a trap when stage 2 inherits stage 1's
failures. For generativity, stage 1 learns from many aligned clips
(where self-sufficiency is cheap to enforce) and stage 2 generalizes to
single-clip inputs (the deployed case). This is arguably safe.

### Q4. Low-res dense + high-res upsample two-stage

**Promoted to constraint C8** (Coarse-to-Fine Stays World-Space).

Two-stage is legitimate iff both stages export persistent world assets.
The invalid form — stage 1 coarse world + stage 2 frame-local rescue —
is a silent C1 violation. The valid form — `W_0 → W_1` where `W_1` is
the deploy-time render target — is a scaling curriculum, not a
contract break. See C8 for the implementation check.

### Q5. Is "no frame-local state" achievable in practice?

Modern architectures use KV caches, attention, and residual connections
that implicitly carry frame-local state. Strict no-frame-local-state may
require architectural discipline that fights the transformer's default
behavior.

Pragmatic answer: aim for bounded frame-local state (it must be
re-derivable from W at decode time, not a stashed secret). Policy: any
per-frame state must round-trip through the encoder → W → decoder path.
If it short-circuits past W, it is the forbidden kind.

---

## Relationship to Framing 1

Framing 1 used information theory: W is gauge-invariant under SE(3) × R;
leakage is mutual information between W and the group element; losses
bound various mutual-information quantities.

Framing 2 uses contracts and vocabulary: W is self-sufficient; generative
reconstruction is the contract; frame-local state is the enemy.

These are the same underlying idea expressed in different languages.
Information-theoretically, self-sufficiency is I(image | W, C_q, t_q) =
I(image | scene_true, C_q, t_q) — W is a sufficient statistic for the
conditional rendering distribution. Generative reconstruction is W
containing strictly more information than its encoder input V (when V is
a restricted view of the scene) — this is only possible if W incorporates
a learned prior, which is the generative prior.

Framing 1 is probably better for deriving losses. Framing 2 is probably
better for naming contract violations during implementation review.

---

## What This Framing Unlocks

If these terms and constraints are accepted, the implementation check
becomes concrete rather than philosophical:

1. Trace the decode path. Is it `(W, C_q, t_q) → render`? If not, C1 is
   violated.
2. Enumerate training losses. Is at least one computed on content not in
   the encoder path? If not, C3/C9 is violated — you have a VAE.
3. Is there a loss comparing W across different encoder paths of the
   same scene? If not, C4 is violated — you have no probe for
   self-sufficiency. Prefer render-agreement over token-L2.
4. At inference, does the prebake pipeline still run? If yes and the
   model depends on it, C7 is violated.
5. For every refinement stage (`W_0 → W_1`, densifier, upsampler), does
   the deployed render read only the final world tokens plus
   `(C_q, t_q)`? If the refined stage depends on per-frame features at
   render time, C8 is violated.

Each constraint has a concrete check. That's the value of the contract
framing over the information-theoretic one: implementation review becomes
possible.
