# World Tokens, Splat Tokens, and Observed-Modality Tokens

Philosophy note on why "tokenizer" means something different for text,
images, videos, worlds, and splats.

This document is meant to prevent a specific regression:

```text
"Train a splat tokenizer first, then train a predictor into that token space."
```

That phrase sounds natural by analogy to text tokenizers, image VAEs, and video
latent tokenizers. For DynaWorld it is dangerous unless the tokenizer is trained
under a world-predictive contract. We do not have ground-truth splat strings on
disk. We only have observations of light. A splat tokenizer trained as a
same-view autoencoder can learn a perfect-looking but useless camera-specific
billboard code. A predictor trained to imitate that code only distills the
failure.

The short version:

```text
Text/image/video tokens are tokens of observed data.
World tokens are tokens of a predictive hidden cause.
Splat tokens are typed renderer-facing commitments derived from world tokens.
```

The hard part is not naming the tokens. The hard part is making the training
contract rule out degenerate tokens that satisfy the source-view loss but fail
under novel cameras.

---

## 1. Core Thesis

Observed-modality tokenizers compress things we actually observe.

```text
text tokenizer:   string on disk -> token sequence
image tokenizer:  image on disk  -> latent grid/code sequence
video tokenizer:  video on disk  -> latent spacetime tokens
```

DynaWorld wants to tokenize a thing we do not observe directly:

```text
world tokenizer:  partial video observations -> hidden world asset W
splat tokenizer:  partial video observations -> renderable 3D/4D splat asset
```

There is no dataset field:

```text
ground_truth_world_tokens
ground_truth_splat_tokens
ground_truth_dynamic_gaussians
```

The only scalable supervision is:

```text
render predicted asset under a query camera/time
compare rendered light to real video frames
```

Therefore, world/splat tokens are not data tokens in the text-tokenizer sense.
They are latent predictive variables. Their validity is behavioral:

```text
Can this exported asset predict omitted observations under supported queries?
```

not:

```text
Does this token sequence match a canonical label?
```

This is why the default baseline should be single-stage held-out prediction
unless a two-stage tokenizer can be shown to produce a stronger non-degenerate
teacher than the end-to-end model.

---

## 2. The Degeneracy We Care About

The primary failure is not "bad splats" in the abstract. It is this:

```text
The model emits splats that render the source camera well
but do not represent persistent 3D/4D scene structure.
```

Common variants:

- Source-view billboards: flat or near-flat sheets aligned to the training
  camera that reproduce pixels from that view.
- Camera-conditioned splats: the learned decoder changes the splat set when
  the query camera changes, so splats are a view-specific image generator in
  disguise.
- Camera leakage into world tokens: the exported asset stores the source camera
  or crop geometry in spare capacity.
- Cancellation solutions: world tokens store camera residue and another branch
  cancels it during source-view reconstruction.
- Frame-local rescue: render quality depends on encoder features, target-frame
  caches, per-frame residuals, or hidden state outside the exported asset.
- Dynamic billboard tracks: a time-conditioned sequence of source-view sheets
  that follows the input video but never becomes a queryable world.

The killer property of these failures is that they can be excellent
autoencoders. A source-view photometric loss can reward them.

A representation is non-degenerate only relative to a query support:

```text
Q_train = cameras/times/crops/views sampled by the training contract
```

If `Q_train` contains only the exact source camera views that the encoder saw,
then source-view billboards are not a bug. They are a valid optimum. The
training signal did not ask for anything else.

The batch-level red-team question should be:

```text
Could a source-view billboard pass this training step?
```

If yes, the step is weak for non-degeneracy. If no, the step pressures
persistent world structure.

---

## 3. Token Types: Definitions and Contracts

### 3.1 Observation Tokens

Observation tokens are input-side evidence.

```text
obs_token_i = PatchEmbed(I_i)
            + RayEmbed(camera_i, intrinsics_i, patch_xy)
            + TimeEmbed(t_i)
            + optional mask/budget/type embeddings
```

They are allowed to contain camera, crop, and time because they describe how
the observation was measured. They do not need to be camera-invariant.

They are not exported.

They are not available at deployed render time unless they have been compressed
into the exported world asset.

If deployed rendering reads observation tokens, the model is not exporting a
world. It is exporting a partial latent plus a hidden side channel.

### 3.2 World Tokens

World tokens are the exported predictive asset:

```text
W = E(O)
```

`W` is not necessarily "pure scene" in a philosophical sense. Framing 3 is more
precise: `W` is a representative of a predictive equivalence class on the
supported query set. The contract is:

```text
S_tau = G(W, tau_q)
I_hat = R_fixed(S_tau, camera_q)
```

At deployed render time, the system reads:

```text
W
query time tau_q
query camera camera_q
fixed generator weights
fixed rasterizer implementation
```

It does not read:

```text
source images
target images
encoder intermediate features
teacher network state
per-frame caches
ground-truth source poses except through W
```

World tokens are allowed to be abstract. They do not have to be one Gaussian
each. They may encode canonical slots, object-ish slots, dense latent grids,
global context, style, occluded content, dynamic potential, or uncertainty
summary. Their required property is self-sufficient predictive use.

### 3.3 Splat Tokens

Splat tokens are renderer-facing typed commitments.

They may be:

```text
S_tau = G(W, tau_q)
```

or, in a simpler design:

```text
S_tau = time_eval(W, tau_q)
```

where `W` already stores splat-like parameters and time bases.

Splat tokens are closer to physics than world tokens. They should eventually
commit to fields such as:

```text
xyz
rotation
scale/covariance
opacity
color / SH / features
time basis or time-conditioned residual
```

But "splat token" does not automatically mean "real 3D object." A splat token
can be degenerate if the training contract permits it.

Good splat tokens are:

- persistent enough to render from multiple supported cameras,
- time-queryable through `G(W, tau)` or an explicit time basis,
- set-permutation invariant up to gauge,
- self-sufficient at export,
- consumed only by the fixed rasterizer at the camera boundary.

Bad splat tokens are:

- generated by a learned module that sees query camera before rasterization,
- tied to source-frame image grids,
- dependent on frame-local caches,
- only evaluated on source cameras,
- unrestricted in rate/capacity relative to prediction gain.

### 3.4 Memory Tokens

Memory tokens are temporal or streaming state.

They are valid only if they obey one of these contracts:

```text
offline export:
    memory is absorbed into W before deployment

streaming export:
    memory_t is part of the exported state and render(asset_t, camera, tau)
    reads only asset_t/query/fixed weights

re-derivable cache:
    memory can be deterministically recomputed from W and the query
```

They are invalid if they are hidden frame-local rescue:

```text
render(W, memory_from_encoder_frame_i, camera_q, tau_q)
```

Memory tokens are useful for long clips and rolling context, but they are a
leakage risk. They can silently hold the real representation while `W` becomes
a thin shell.

### 3.5 Camera Tokens

Observation-side camera tokens are allowed. They may represent estimated
source camera poses or camera uncertainty.

Query-side camera tokens are dangerous if fed into learned scene generation.
The clean contract is:

```text
observation camera/time: allowed at encoder ingestion
query time: allowed in G(W, tau)
query camera: only allowed in R_fixed
```

Why the asymmetry?

Time changes the scene state. The learned generator must know which time state
to produce. Camera does not change the world. It changes the measurement. The
fixed renderer should own that transformation.

If a learned decoder sees query camera, it can learn:

```text
camera -> view-specific splats/image
```

instead of:

```text
world -> persistent splats -> physical camera projection
```

There are possible future exceptions, but they should be admitted as explicit
escape hatches, not the default.

### 3.6 Query Tokens

The output side should provide queries, not answers.

Allowed query:

```text
q = (camera_q, tau_q, maybe intrinsics_q)
```

Forbidden query leakage:

```text
target image patches
target-frame encoder features
target optical flow computed from GT
target-view depth from a teacher as a permanent render-time input
```

At training time, target images are used only for loss. They are not
conditioning data.

---

## 4. Why Text/Image/Video Tokenizers Are Different

### 4.1 Text Tokenizers

Text tokenizers are supervised by the existence of text itself.

There is an observed object:

```text
"the cat sat"
```

The tokenizer maps it to discrete IDs:

```text
[464, 3797, 3332]
```

The IDs are not hidden causes of the text. They are a code for the observed
sequence. A language model can then learn:

```text
p(next_token | previous_tokens)
```

The tokenizer can be judged directly by compression, reversibility, vocabulary
coverage, and downstream language modeling performance.

There is no analogy to "degenerate source-view billboard text tokens" because
the tokenized object is the observed object.

### 4.2 Image Tokenizers

Image tokenizers usually compress an observed image:

```text
z = Enc(image)
image_hat = Dec(z)
```

Even VQ-VAE or latent diffusion tokenizers have direct reconstruction targets:

```text
image on disk
```

They can be blurry, lossy, or semantically awkward, but they are not being
asked to infer an unobserved 3D cause. Their target exists in the dataset.

An image tokenizer can be trained as a same-image autoencoder because the
deployment object is also an image or image latent. The code only needs to
decode the same modality.

### 4.3 Video Tokenizers

Video tokenizers compress observed spacetime:

```text
z = Enc(video_clip)
video_hat = Dec(z)
```

They face temporal compression and motion representation problems, but their
training target is still observed frames.

A video tokenizer can learn excellent view-specific 2D dynamics and still be
useful for video generation. That is not enough for DynaWorld because
DynaWorld's primary use case is editing the camera path inside a renderable
world.

A good video token is not automatically a good world token.

### 4.4 World/Splat Tokenizers

A world/splat tokenizer would try:

```text
W = Enc(video_observations)
render(W, camera_q, tau_q) -> image_q
```

But the object being tokenized is not on disk. It is an inferred cause behind
the observations. Therefore the tokenizer must be trained by prediction under
interventions:

```text
different time
different crop/intrinsics
different camera when available
different held-out observation budget
```

A same-view reconstruction tokenizer does not become a world tokenizer merely
because the decoder contains a rasterizer. The rasterizer can render degenerate
geometry too.

---

## 5. The Tokenizer Fallacy

The fallacy:

```text
1. Train a splat tokenizer.
2. Freeze it.
3. Train a model to predict splat tokens.
4. Profit.
```

This imports the text/image/video pattern but skips the load-bearing fact that
text/image/video tokenizers have observed ground-truth objects.

For splats, stage 1 is already the hard world-learning problem:

```text
video observations -> non-degenerate world/splat asset
```

If stage 1 is trained with same-view reconstruction only, it can choose:

```text
source-view billboard W_b
```

Then stage 2 learns:

```text
O_sparse -> W_b
```

The predictor may become excellent at predicting a useless asset. The two-stage
pipeline did not solve non-degeneracy. It made the degeneracy stable and
possibly harder to see.

The correct question is not:

```text
Should we train a tokenizer?
```

It is:

```text
What training contract makes the tokenizer's tokens worth predicting?
```

If the answer is the same held-out predictive contract as the full model, then
the tokenizer is not a simpler prerequisite. It is an alternative optimization
schedule for the same problem.

---

## 6. The Single-Stage Default

The default architecture should be:

```text
O = observation set
H = held-out target set, H not_subset O
Q = {(camera_h, tau_h) for h in H}

W       = E(O)
S_h     = G(W, tau_h)
I_hat_h = R_fixed(S_h, camera_h)

L_pred = sum_h ell(I_hat_h, I_h)
L      = L_pred + beta * Rate(W)
```

This single-stage form has the cleanest pressure:

- The encoder cannot copy exact target observations because they are omitted.
- Query camera only enters the fixed rasterizer, limiting camera-conditioned
  image-generation bypasses.
- Query time enters the scene generator because the scene state changes over
  time.
- Rate/minimality discourages useless source-camera residue and unused junk.
- Held-out prediction defines behavior on supported queries.

This is not guaranteed to solve arbitrary novel views from monocular data. It
only identifies behavior on the support of `D_var`. But it is honest: every
claim is tied to a supported query distribution.

### 6.1 What This Means for Source Videos

For a single moving-camera clip:

```text
V = {(I_i, camera_i, tau_i)}
```

sample:

```text
O = frames/crops visible to the encoder
H = frames/crops omitted from encoder input
Q = cameras/times/intrinsics for H
```

Moving camera gives real parallax over time. This is useful. It is still not
same-time multi-view supervision for dynamic objects. The model must infer
dynamics and novel geometry jointly.

Useful budgets:

- sparse frames -> nearby omitted frames,
- rich frames -> sparse omitted targets,
- early frames -> later held-out frames,
- odd frames -> even held-out frames,
- cropped/ray-shifted observations -> different crops or uncropped targets,
- multi-view/synthetic targets when available.

The supported claim is:

```text
W can predict omitted observations along the observed moving-camera/time support
and any crop/camera perturbations included in D_var.
```

The unsupported claim is:

```text
W is correct from arbitrary cameras far outside Q_train.
```

Do not claim the second unless the training support or post-training signal
actually reaches it.

---

## 7. When Two-Stage Training Is Legitimate

Two-stage training is not forbidden. It is legitimate when it improves
optimization or amortization without weakening the world contract.

### 7.1 Rich-Budget Teacher, Sparse-Budget Student

A valid two-stage version:

```text
Stage 1:
    W_rich = E_rich(O_rich)
    train W_rich by held-out prediction + rate

Stage 2:
    W_sparse = E_sparse(O_sparse)
    train W_sparse by held-out prediction
    optionally train render-agreement with W_rich on shared query support
```

Important: the teacher is not ground truth. It is a richer-budget predictive
asset. It is useful only if it passes non-degeneracy diagnostics better than
the sparse model.

Do not use raw token L2 as the primary imitation loss unless tokens are
canonicalized. World/splat slots are permutation- and gauge-ambiguous.

Prefer behavioral targets:

```text
render_agreement(W_sparse, W_rich, Q_shared)
photometric(W_sparse, H_real)
rate(W_sparse)
```

### 7.2 Per-Scene Optimization, Then Amortization

Another valid two-stage version:

```text
Stage 1:
    optimize a per-scene W directly against many held-out renders/crops/views

Stage 2:
    train E(O) to initialize or predict W
```

This resembles NeRF/3DGS amortization. It can produce better targets than a
feed-forward model early in research, but the per-scene optimized asset must
itself be non-degenerate. Same-view per-scene 3DGS fits can also cheat.

Per-scene optimization should include:

- held-out frames not used as direct optimization input,
- crop/intrinsics shifts,
- multi-view if available,
- rate/export-size constraints,
- export-purity checks.

### 7.3 Frozen Decoder / Learned Encoder

A third version:

```text
train W -> G -> R as a strong decoder/asset format
then train E(O) into that format
```

This is only useful if the decoder/asset format encodes a real inductive bias:
for example, bounded splat parameterization, canonical scene scale, time-basis
regularity, or a renderer-compatible export layout.

If the decoder is flexible enough to render arbitrary source-view sheets, it
does not solve non-degeneracy.

### 7.4 What Makes Two-Stage Invalid

Two-stage is invalid or suspect when:

- stage 1 is same-view autoencoding,
- stage 1 tokens are not export-pure,
- stage 1 requires hidden per-frame state at render time,
- stage 2 uses raw token MSE on non-canonical slot sets,
- the teacher's novel-view renders are never checked,
- the tokenizer objective would be the first thing we ablate away,
- the two-stage split exists mainly because "tokenizers work for LLMs."

---

## 8. Unified World Tokens vs Separate Splat Tokens

This is not a binary choice.

The clean default is:

```text
one exported world asset W
typed splat readout S_tau = G(W, tau)
fixed rasterizer R_fixed(S_tau, camera)
```

That gives us unified world tokens at the semantic/export boundary and splat
tokens at the renderer boundary.

### 8.1 Why Not Only Unified World Tokens?

If `W` stays fully abstract until an image decoder, the model can bypass 3D:

```text
W + camera + time -> image
```

This is a video generator, not a renderable world asset.

We need a physical commitment:

```text
W -> splats -> rasterizer
```

The splat boundary forces perspective projection, visibility/opacity, spatial
extent, and set structure into the computation. That is useful unavoidable
structure.

### 8.2 Why Not Only Splat Tokens?

If every world token is immediately a Gaussian, we may over-constrain the
representation too early.

Some information is not naturally one Gaussian:

- global scene scale,
- camera-origin gauge,
- long-range appearance consistency,
- dynamic identity,
- occluded completion,
- uncertainty over unseen regions,
- shared material/lighting priors,
- object-level persistence.

A world token layer can hold abstract predictive state. A splat readout can
commit that state to renderer-facing primitives at time `tau`.

### 8.3 Recommended Split

Use:

```text
W_slots: abstract exported world tokens
S_slots_tau: time-conditioned splat tokens
GaussianParams_tau: typed parameters consumed by renderer
```

with:

```text
W_slots = E(O)
S_slots_tau = G_slots(W_slots, tau)
GaussianParams_tau = Heads(S_slots_tau)
image = R_fixed(GaussianParams_tau, camera)
```

The key restriction:

```text
G_slots and Heads do not see query camera.
```

This lets the model be expressive while keeping camera projection at the fixed
physics boundary.

### 8.4 When Direct Splat Tokens Are Fine

Direct splat tokens are fine for baselines and small experiments:

```text
W is already a set of Gaussian base params + time coefficients
S_tau = evaluate_time_basis(W, tau)
```

This is simple and export-friendly. It may be the right first implementation.

The abstract-world-token layer becomes useful when direct splats lack capacity
or when we need a richer internal representation for long clips, occlusion, or
dynamic completion.

### 8.5 The Wrong Split

The wrong split is:

```text
world tokens + query camera -> learned decoder -> splats/image
```

This invites view-conditioned splat generation. It can still work under strong
multi-view constraints, but it weakens the cleanest anti-degeneracy boundary.

If we ever allow camera into learned generation, the doc should explicitly say:

- why fixed rasterization is insufficient,
- what failure mode the camera-conditioned branch solves,
- why it does not reintroduce billboard splats,
- what diagnostic will catch leakage,
- when the branch retires.

---

## 9. Time Conditioning

Time is not just another token type. It is a query axis that changes the scene.

The clean signature:

```text
S_tau = G(W, tau_q)
image = R_fixed(S_tau, camera_q)
```

Recommended time inputs:

```text
tau_abs = frame_index / (num_frames - 1)
tau_rel = tau_q - center_time(O)
time_embed = MLP(FourierFeatures(tau_abs, tau_rel))
```

Rules:

- Use absolute clip time, not local window `0..1` time, unless local time is
  explicitly part of a hierarchical representation.
- Observation time enters encoder ingestion.
- Query time enters `G`.
- Query camera stays out of `G`.
- If streaming, memory updates must be export-pure or part of the asset state.

Failure modes:

- Local-window time makes the same frame mean different things across batches.
- No explicit time makes the model collapse dynamics into appearance.
- Time only after query attention can make tokens ignore motion.
- Per-frame caches can become the true dynamic representation.

Time conditioning should be tested by:

- zeroing time basis,
- rendering same `W` at multiple `tau`,
- measuring adjacent-frame motion vs first-frame motion,
- wrong-time swaps,
- held-out temporal interpolation/extrapolation.

---

## 10. Rate, Minimality, and Capacity

Held-out prediction identifies behavior on supported queries. It does not by
itself remove all junk.

If two assets render the same on `Q_train`:

```text
Pi_Q(W1) = Pi_Q(W2)
```

then photometric prediction cannot prefer the cleaner one. We need a
representative selector:

```text
L = L_pred + beta * Rate(W)
```

Rate/minimality can mean:

- hard token count,
- channel bottleneck,
- export-size cap,
- quantization,
- entropy model,
- KL to prior,
- MDL/code-length estimate,
- pruning pressure.

Rate pressure is not primarily for compression aesthetics. It fights
degeneracy by making useless source-camera residue expensive.

But rate cannot create missing supervision. A tiny billboard can also be low
rate. Rate only helps after `D_var` makes billboard behavior predictive-bad on
held-out queries.

The useful question:

```text
Does increasing W capacity buy held-out prediction quality on supported novel
queries, or only source-view quality?
```

If capacity improves source-view reconstruction but not held-out/crop/novel
queries, it may be storing degeneracy.

---

## 11. Post-Training GAN/Diffusion on Novel Views

Post-training adversarial or diffusion losses can help, but they do not replace
the predictive world contract.

Potential GAN setup:

```text
real:    real frames from source-camera distribution
fake:    renders from W under held-out/perturbed/novel cameras
update:  push novel renders to look like real video frames
```

Potential diffusion-as-loss setup:

```text
render W at held-out/perturbed camera
score the rendered image/clip with frozen image/video diffusion teacher
backprop score through rasterizer into W/G/E or post-training adapter
```

What this can fix:

- off-manifold visual artifacts,
- floaters/holes that photometric L1 tolerates,
- novel-view texture implausibility,
- dynamic realism under weak GT.

What it cannot guarantee:

- correct hidden geometry,
- same-time multi-view consistency,
- object permanence,
- camera-independent splat identity.

A GAN can make a novel view plausible by hallucination. That may be useful for
final pixels, but it is not the same as making splat tokens non-degenerate.

Therefore the recommended placement is:

```text
pretraining:
    held-out predictive loss + rate/minimality

post-training:
    adversarial/diffusion prior on held-out or novel renders
```

If adversarial training is used on a tokenizer, the tokenizer still must be
export-pure and trained on omitted observations. The GAN should be a pressure
on novel-render realism, not the only reason the tokens are world-like.

---

## 12. Diagnostics for Non-Degenerate Tokens

Diagnostics must be measurement-only unless admitted as escape hatches.

### 12.1 Export-Purity Test

Render with only:

```text
W, camera_q, tau_q, fixed weights
```

Null:

```text
encoder features
source frames
per-frame caches
teacher state
training-only tensors
```

Any quality drop beyond noise means `W` was not the full asset.

### 12.2 Camera-Leakage Probe

Freeze `W`. Train a small probe to predict:

```text
source yaw/pitch/roll
translation
crop center
focal length
frame index if inappropriate
```

High accuracy does not prove failure by itself, because framing 3 does not
promise latent purity. But it indicates rate waste and weak support. First
response should usually be:

```text
broaden D_var
increase crop/camera support
tighten rate
```

not immediately:

```text
add adversarial loss
```

### 12.3 World-Dependence Matrix

At eval:

- drop 30% of tokens,
- shuffle token order or child-index embeddings,
- zero time coefficients,
- swap in `W` from a different clip,
- scramble source-camera metadata at encoder ingestion,
- render under camera perturbations.

Interpretation:

- token drop no effect -> unused capacity or hidden side channel,
- zero time no effect -> dynamic pathway unused,
- wrong-world still plausible -> decoder prior ignoring evidence,
- camera perturbation collapses -> source-view degeneracy.

### 12.4 Billboard Stress Test

Construct or fit an explicit billboard baseline:

```text
flat textured sheet at source depth
or camera-facing dynamic sheet sequence
```

Compare it to the model on:

- source-view reconstruction,
- held-out frames,
- crop/ray-shift targets,
- camera perturbations,
- multi-view/synthetic targets if available.

If the model does not beat a billboard on held-out query support, it is not
learning the thing we care about.

### 12.5 Rate-Distortion by Query Type

Plot quality vs capacity separately:

```text
source camera
nearby held-out camera/time
crop-shift
far time
synthetic/multi-view
novel perturbation
```

Good sign:

```text
capacity improves held-out query quality
```

Bad sign:

```text
capacity improves only source-view reconstruction
```

---

## 13. Architecture Workflow for Researchers

When defining a new architecture, ask in this order.

### 13.1 What Is the Deployed Function?

Write the deploy signature first:

```text
asset = encode(video_observations)
render(asset, camera_q, tau_q) -> image
```

Then expand:

```text
W = E(O)
S_tau = G(W, tau_q)
image = R_fixed(S_tau, camera_q)
```

If the proposed architecture cannot fit this signature, decide whether the
signature is wrong or the proposal is a different product.

### 13.2 What Is Omitted?

For every loss, write:

```text
encoder_input_set = ...
loss_target_set = ...
```

If target is inside input, the loss is autoencoding. Autoencoding may be useful
as a warmup, but it does not earn the world-model claim.

### 13.3 Where Can the Model Cheat?

List concrete cheating solutions:

- source-view billboard,
- camera-conditioned splat generator,
- frame-local cache,
- source-camera hash in W,
- time ignored,
- wrong-world decoder prior,
- raw token agreement collapse,
- capacity bypass.

For each, ask:

```text
Which term in the training contract makes this solution high-loss?
```

If no term does, the architecture does not rule it out.

### 13.4 What Is the Strongest Correct Prior?

Good priors:

- fixed perspective rasterizer,
- splats as 3D primitives,
- set permutation invariance,
- SE(3) cameras at renderer boundary,
- time-conditioned scene state,
- export purity,
- observation camera/ray embeddings at ingestion.

Risky priors:

- object/background split,
- static/dynamic hard partition too early,
- low-rank motion as a universal assumption,
- learned camera-conditioned pre-render decoder,
- token equality across non-canonical slot sets.

### 13.5 What Is the Query Support?

State:

```text
Q_train = support of cameras/times/crops/views sampled in training
Q_deploy = desired deployment support
Delta = Q_deploy - Q_train
```

Everything in `Delta` is extrapolation unless a post-training prior or external
data supports it.

### 13.6 What Is the Retirement Condition for Complexity?

Any auxiliary mechanism should have a retirement condition:

- geometry teacher decays to zero,
- adversarial head stays diagnostic unless leakage persists,
- GAN post-training remains a final-pixel prior,
- stochastic residual prior only ships if deterministic W averages incompatible
  futures,
- tokenizer stage only ships if it improves held-out prediction or optimization
  without weakening export purity.

No retirement condition usually means the mechanism is compensating for a weak
contract.

---

## 14. Recommended Current Position

Current belief:

```text
Train a unified single-stage predictive world model first.
Use one exported W and a typed time-conditioned splat readout.
Do not start with a separate splat tokenizer unless optimization forces it.
```

Concrete default:

```text
Observation tokens:
    image patch + observation ray/camera + observation time

World tokens:
    exported W slots

Splat tokens:
    S_tau = G(W, tau_q)
    Gaussian params = Heads(S_tau)

Render:
    image = R_fixed(Gaussian params, camera_q)

Loss:
    held-out photometric/LPIPS/DSSIM on H not_subset O
    + rate/minimality on W

Diagnostics:
    export purity
    camera leakage probe
    wrong-world/token-drop/time-zero matrix
    billboard baseline
```

Two-stage becomes attractive only after one of these is true:

- single-stage optimization is unstable but the contract is sound,
- rich-budget teacher clearly produces better non-degenerate `W`,
- per-scene optimization gives stronger export-pure assets,
- we need a reusable latent format for many downstream predictors,
- freezing `W -> G -> R` improves training speed without freezing a failure.

Even then, the two-stage target should be behavioral:

```text
render agreement + held-out real-frame loss
```

not naive:

```text
token MSE to non-canonical splat slots
```

---

## 15. Open Questions

### Q1. How abstract should W be before splat readout?

Direct splat `W` is simple and export-friendly. Abstract `W` plus splat readout
may be more expressive. The right answer depends on whether direct splats fail
because of representational limits or because of training support.

Cheap test:

```text
same D_var, same renderer, same splat budget
compare direct time-basis splats vs abstract W -> S_tau readout
on held-out query matrix
```

### Q2. How much camera perturbation can monocular source videos support?

Crop/ray-shift gives weak local camera support. Moving camera gives parallax
over time. Neither fully solves arbitrary same-time novel view for dynamic
objects.

Cheap test:

```text
train with increasing crop/camera perturbation support
measure where held-out quality collapses vs billboard baseline
```

### Q3. Is the main bottleneck data support or representation?

If even multi-view/synthetic budgets produce degenerate splats, representation
or optimization is broken. If multi-view fixes it but monocular does not, the
issue is support.

Cheap test:

```text
run same model on synthetic multi-view/camera-rich clips
compare degeneracy diagnostics
```

### Q4. Does rate pressure actually remove camera residue?

Rate may select smaller billboards, not better worlds, unless `D_var` makes
billboards predictive-bad.

Cheap test:

```text
sweep token count / channels / quantization
plot source-view vs held-out/crop/perturbed quality
fit camera-leakage probe at each rate
```

### Q5. Where should post-training adversarial pressure enter?

Possibilities:

- only final pixels,
- update `G` and splat heads,
- update encoder too,
- train a refinement adapter,
- train only on held-out-camera renders.

Cheap test:

```text
freeze E vs unfreeze E during adversarial/diffusion post-training
monitor export-purity and wrong-world tests
```

---

## 16. One-Sentence Rule

Do not ask whether we need "world tokens or splat tokens" as if names solve the
problem.

Ask:

```text
What exported state, when queried only by time and rendered only by a fixed
camera rasterizer, predicts observations the encoder did not see while making
source-view billboard solutions fail?
```

Whatever token structure satisfies that contract is the right one for now.

