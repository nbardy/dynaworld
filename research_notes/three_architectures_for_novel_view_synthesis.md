# Three Architectures for Novel-View Synthesis Without Multi-View Data

Long-form design doc. Fixes the training recipe and asks: given those
assumptions, what is the right model shape to force novel-view synthesis,
prevent camera/time leakage, and keep the world tokens view-agnostic?

Three candidate architectures, each committing to a different
leakage-prevention mechanism. Diagrams, forward-pass detail, training
contracts, debate, and iteration notes per candidate.

This doc treats architecture as a downstream variable of the training
recipe, not the other way around. The recipe is fixed; the architecture is
the question.

---

## 1. Fixed Assumptions (the recipe is decided)

Treat these as constraints the architectures must fit, not design choices
open for debate.

**Training objectives (pretraining):**

1. **Crop-based pseudo multi-view.** High-res GT frames are cropped
   off-center and the crop is treated as a camera extrinsic (ray-shift, not
   perspective warp). Forces the model to render correctly when the
   principal point is not centered.
2. **Chunk-swap prediction.** Two chunks of the same video. The model
   encodes chunk 1, then is asked to produce chunk 2's GT frames given
   chunk 2's camera and time conditioning. Forces world tokens to generalize
   across the camera/time span of the clip.
3. **Standard photometric loss** on training-camera renders (L1 + LPIPS).

**Post-training:**

4. **Teacher-model post-training on novel-view renders.** Either adversarial
   (GAN) or diffusion-as-loss (SDS/VSD). Both play the same role: score
   novel-view plausibility where no GT frame exists. Diffusion-as-loss is
   likely the cleaner default; GAN is acceptable if simpler to integrate.

**Data:**

- Self-supervised, single-camera video.
- Posed or implicit-camera (the architecture must tolerate either; prebake
  is an input signal, not a training label).
- No multi-view paired data.

**Output boundary:**

- The model emits splat parameters consumed by a fixed differentiable
  rasterizer. Splats commit at the rasterizer boundary. Cameras commit at
  the rasterizer boundary. Upstream, both are latent.

---

## 2. What "Force Novel-View Synthesis" Means

Novel-view synthesis works iff the world tokens the model emits are
**view-invariant**: the same tokens render plausibly from multiple cameras,
including cameras the training reconstruction loss never saw.

The three mechanisms that can enforce view-invariance given our data:

- **Information bottleneck.** The main trunk literally does not see camera
  information; it cannot leak what it does not possess.
- **Adversarial recovery.** A discriminator tries to recover camera from
  world tokens; the encoder gets a gradient penalty if it succeeds.
- **Multi-camera consistency pressure.** During training, the same world
  tokens are rendered from K cameras simultaneously. The only way to
  satisfy K reconstruction losses is for the tokens to be view-agnostic.
- **Augmentation invariance.** Crop and camera perturbation during training
  force the world tokens to be invariant to those perturbations.

The three architectures below each commit to a different primary mechanism
and compose the others as secondary pressure.

---

## 3. How to Embed Video into View-Agnostic Tokens

Question shared by all three architectures: given a video, how does the
encoder produce tokens that are not tied to the encoding camera?

Options, roughly in order of prescriptive commitment:

| Option | Description | Commitment |
|--------|-------------|-----------|
| V1 | Generic causal/bidirectional transformer over patch tokens. View-invariance emerges from loss pressure. | Low |
| V2 | V1 + view-invariance auxiliary loss (random crop / camera perturb at encode time, require token similarity). | Medium |
| V3 | V1 + contrastive loss across overlapping chunks of the same clip (same-scene different-time should agree on a canonical subspace). | Medium |
| V4 | Pretrained video backbone (V-JEPA, video diffusion features) with view-invariance priors from large-scale pretraining. | High (external model dep) |
| V5 | Explicit two-stream: world encoder + pose encoder. Architectural split as the invariance mechanism. | High |

Architecture A uses V1 + V2.
Architecture B uses V5.
Architecture C uses V1 + training-time multi-camera pressure (no separate
invariance loss; the invariance IS the reconstruction loss at K cameras).

---

## 4. Leakage Prevention — Enumerated

The problem: world tokens may silently carry camera or time information.
Five mechanisms to prevent or detect this:

1. **Information bottleneck.** Deny the encoder access to the thing you
   don't want in the tokens. For camera: never feed camera to the encoder.
   For time: scramble frame order or mask positional encoding into a
   separate pathway.
2. **Dimensional bottleneck.** Camera / time tokens have high enough
   capacity to carry camera / time info; world tokens do not (or are
   penalized for it). Asymmetric capacity.
3. **Adversarial recovery.** A small discriminator tries to predict C / t
   from world tokens; encoder gets a gradient-reversal penalty.
4. **Multi-condition consistency.** Render the same world tokens under K
   different `(C, t)` conditions. If tokens carry condition info, they
   render inconsistently. Forces invariance.
5. **Augmentation invariance.** Perturb `(C, t)` during training; require
   tokens to be invariant. Implicit via reconstruction loss at perturbed
   conditions.

Architecture A relies on (1), (2), and (4).
Architecture B relies on (3), with (4) as backup.
Architecture C relies on (4) and (5) — no architectural commitment.

---

## 5. Architecture A — "Blind World Encoder" (Information Bottleneck)

Primary mechanism: **the encoder never sees camera, so it cannot leak
camera.** Camera only exists as a decoder query at the rasterizer boundary.

### 5.1 Forward Pass Diagram

```
   ┌───────────────────────────────────────────────────────────┐
   │                                                            │
   │   Video frames (F, H, W, 3)                                │
   │         │                                                  │
   │         ▼                                                  │
   │   Patch tokenize (ViViT-style, spatiotemporal patches)     │
   │         │                                                  │
   │         ▼                                                  │
   │   ┌───────────────────────────────────┐                   │
   │   │ Blind World Encoder               │                   │
   │   │ (transformer, NO camera input,    │                   │
   │   │  NO time positional encoding at   │                   │
   │   │  the token level — ordering only) │                   │
   │   └───────────────────────────────────┘                   │
   │         │                                                  │
   │         ├──► world_tokens  (N, D_world)                    │
   │         ├──► camera_tokens (F, D_cam)      ← small         │
   │         └──► time_tokens   (F, D_time)     ← small         │
   │                                                            │
   └───────────────────────────────────────────────────────────┘

   DECODE QUERY at (C_q, t_q):

   ┌───────────────────────────────────────────────────────────┐
   │   C_q (SE(3))              t_q (continuous)                │
   │         │                        │                         │
   │         ▼                        ▼                         │
   │   Query embedding (small MLP each, concat)                 │
   │         │                                                  │
   │         ▼                                                  │
   │   Cross-attention → world_tokens                           │
   │         │                                                  │
   │         ▼                                                  │
   │   Splat head → splat_set (G, 14)  [xyz, rot, scale, rgb, α] │
   │         │                                                  │
   │         ▼                                                  │
   │   Rasterizer(splat_set, C_q) → rendered_image               │
   └───────────────────────────────────────────────────────────┘
```

### 5.2 The Key Commitment

- `world_tokens` have a large capacity (e.g., N=128, D=512).
- `camera_tokens` and `time_tokens` have a small capacity (e.g.,
  D_cam=32, D_time=16). Small enough to carry C and t, too small to carry
  scene.
- The main encoder never receives camera as input. Ordering of input tokens
  gives it the sequence structure; any inference it makes about the
  training camera must happen through motion parallax and content cues.

### 5.3 Training Contract

Per training step:

1. Run encoder on video → `(world_tokens, camera_tokens, time_tokens)`.
2. For each frame f, decode at `(C_pred_f, t_pred_f)` where `C_pred_f` is
   derived from `camera_tokens[f]` and `t_pred_f` from `time_tokens[f]`.
3. Render. Photometric L1 + LPIPS vs GT frame f.
4. **Crop augmentation:** randomly crop the GT frame and shift the query
   camera by the corresponding principal-point delta. World tokens must
   handle this without error.
5. **Chunk swap:** encode chunk 1 → world_tokens_1. Predict chunk 2 GT by
   decoding at chunk 2's `(C_pred, t_pred)` from world_tokens_1.
   Reconstruction loss against chunk 2 GT. World tokens that leak camera
   or time from chunk 1 fail this loss.
6. **Camera-perturb (novel view).** Sample δ. Decode at `(C_pred_f + δ,
   t_pred_f)`. No GT exists. Score with frozen video-diffusion
   (diffusion-as-loss) or send to GAN post-training stage.
7. **Cross-query consistency.** Decode at `(C_pred_f + δ1, t_pred_f)` and
   `(C_pred_f + δ2, t_pred_f)`. Splat overlap must be geometrically
   consistent (epipolar check or feature-consistency).

### 5.4 When Camera and Time Enter

- **Never in the main trunk.** Encoder is "blind" to both.
- **At the decoder query level.** `C_q` and `t_q` are provided by the user
  (or predicted by the per-frame head) and cross-attend into world tokens.
- **Never back-projected into world tokens.** The decoder query pulls
  information out; nothing flows back up.

### 5.5 Leakage Prevention (F1, F2)

- F1 (camera leakage): world_tokens do not see camera as input. The
  encoder can only implicitly infer camera from video motion. The
  dimensional bottleneck on `camera_tokens` gives an explicit place for
  camera info to live, which reduces (but does not eliminate) the
  incentive to hide camera in world_tokens.
- F2 (cheating splats): chunk swap + crop + camera perturbation force
  world_tokens to produce correct renders under many `(C, t)` conditions.
  A splat set that only works at training cameras fails these losses.

### 5.6 Failure Modes

- The encoder can still infer camera from motion parallax and bake it into
  world_tokens, since we have no direct way to prevent inference from
  content. Dimensional bottleneck reduces but does not remove this
  pressure.
- Time leakage through frame ordering: if the encoder is causal, the
  ordering itself carries time. Partially mitigated by randomized ordering
  on chunk swap.
- Small `camera_tokens` / `time_tokens` may be insufficient capacity for
  complex camera motion; may need to scale up carefully.

### 5.7 Where It Sits in Mechanism Space

| Mechanism | Used? |
|-----------|-------|
| Information bottleneck | ★ Primary |
| Dimensional bottleneck | ★ Primary |
| Adversarial recovery | no (but addable) |
| Multi-condition consistency | ✓ Secondary |
| Augmentation invariance | ✓ Secondary |

### 5.8 Iteration Ideas

- Add a small adversarial head trying to recover C from world_tokens. GRL
  into encoder. Optional but likely helpful.
- Mask frame-index positional encoding on a subset of training steps to
  reduce time leakage via ordering.
- Cap `camera_tokens` / `time_tokens` capacity adversarially (e.g., via
  quantization or noise injection) to force information to route through
  intended channels.

---

## 6. Architecture B — "Two-Stream + Adversarial Separation"

Primary mechanism: **explicit architectural split** between a world encoder
and a pose/time encoder, with an adversarial head actively punishing camera
information in world tokens.

### 6.1 Forward Pass Diagram

```
   ┌──────────────────────────────────────────────────────────────────┐
   │                                                                   │
   │   Video frames                                                    │
   │         │                                                         │
   │         ▼                                                         │
   │   Patch tokenize                                                  │
   │         │                                                         │
   │         ├─────────────┐                                           │
   │         │             │                                           │
   │         ▼             ▼                                           │
   │   ┌──────────┐   ┌──────────┐                                    │
   │   │  World   │   │   Pose   │                                    │
   │   │ Encoder  │   │ Encoder  │                                    │
   │   └──────────┘   └──────────┘                                    │
   │         │             │                                           │
   │         ▼             ▼                                           │
   │   world_tokens   camera_tokens (F, D_cam)                         │
   │      (N, D)      time_tokens   (F, D_time)                        │
   │         │                                                         │
   │         │                                                         │
   │         ▼                                                         │
   │   ┌─────────────────────┐                                        │
   │   │ Adversarial Head    │   (small classifier)                   │
   │   │ recovers C from     │                                         │
   │   │ world_tokens        │                                         │
   │   └─────────────────────┘                                        │
   │         │                                                         │
   │         ▼                                                         │
   │   L_adv ─── GRL ───► penalizes World Encoder                     │
   │                                                                   │
   └──────────────────────────────────────────────────────────────────┘

   DECODE (C_q, t_q):

   world_tokens × C_q × t_q → cross-attn → splat head → rasterizer
```

### 6.2 The Key Commitment

- Parallel encoders, shared input, disjoint outputs.
- Adversarial head is the active pressure: it is trained to recover camera
  from world tokens; the world encoder is trained via gradient reversal to
  defeat it.
- Pose encoder carries camera and time explicitly.

### 6.3 Training Contract

Per step:

1. Both encoders run on video.
2. Photometric reconstruction using `(world_tokens, camera_tokens,
   time_tokens)` through the decoder.
3. Adversarial head takes `world_tokens` (possibly time-aggregated),
   predicts camera rotation / translation class bins.
4. `L_adv` trains the adversarial head to succeed; GRL into world encoder
   trains it to fail.
5. Crop + chunk-swap + perturbed-camera losses as in Architecture A.

### 6.4 When Camera and Time Enter

- **In the pose encoder branch** (not the world encoder).
- **At the decoder query level,** same as A.
- Explicit architectural split is the commitment.

### 6.5 Leakage Prevention

- F1 (camera): adversarial head provides an active gradient penalty
  whenever world tokens contain camera-recoverable info. Strongest direct
  pressure of the three architectures.
- F2 (cheating splats): same chunk-swap / crop / perturb regime as A.
- Time leakage: adversarial head can be extended to recover time from
  world tokens too.

### 6.6 Failure Modes

- **Adversarial training is finicky.** GRL hyperparameter needs tuning,
  and adversarial collapse is a known failure mode (the adversarial head
  gets too good or too weak and the game destabilizes).
- **Pose encoder may duplicate content.** Without constraints, both
  encoders might learn overlapping representations, defeating the split.
  Likely needs an orthogonality loss or information-bottleneck on the
  pose encoder.
- **Compute overhead:** two encoders, one adversarial head. ~1.5–2x
  training cost of A.
- **The adversarial signal is only as good as the head's task
  formulation.** If the head predicts coarse camera bins, it may miss
  fine-grained leakage.

### 6.7 Where It Sits in Mechanism Space

| Mechanism | Used? |
|-----------|-------|
| Information bottleneck | ✓ Secondary (via stream split) |
| Dimensional bottleneck | optional |
| Adversarial recovery | ★ Primary |
| Multi-condition consistency | ✓ Secondary |
| Augmentation invariance | ✓ Secondary |

### 6.8 Iteration Ideas

- Train adversarial head on a slower schedule to keep the game balanced
  (e.g., K adversary steps per G main step).
- Use a mutual-information upper bound (CLUB, MINE) instead of direct
  classification for finer-grained leakage detection.
- Add a parallel time-adversarial head targeting time recovery.

---

## 7. Architecture C — "Multi-Camera Consistency by Design" (Simplest)

Primary mechanism: **training-time pressure alone.** Same world tokens are
rendered from many cameras per training step. No architectural split. The
invariance IS the loss.

### 7.1 Forward Pass Diagram

```
   ┌──────────────────────────────────────────────────────────────────┐
   │                                                                   │
   │   Video frames                                                    │
   │         │                                                         │
   │         ▼                                                         │
   │   Patch tokenize                                                  │
   │         │                                                         │
   │         ▼                                                         │
   │   ┌────────────────────────────────────┐                         │
   │   │ Monolithic Transformer             │                         │
   │   │ (single encoder, emits world+      │                         │
   │   │  camera+time as disjoint output    │                         │
   │   │  heads — factorization is a        │                         │
   │   │  learned output structure, not     │                         │
   │   │  an architectural split)           │                         │
   │   └────────────────────────────────────┘                         │
   │         │                                                         │
   │         ├──► world_tokens                                         │
   │         ├──► camera_tokens (F, D)                                 │
   │         └──► time_tokens   (F, D)                                 │
   │                                                                   │
   └──────────────────────────────────────────────────────────────────┘

   TRAINING STEP — render from K cameras simultaneously:

     ┌─ render at C_pred_f             → L_photometric (vs GT frame)
     │                                                   │
   world_tokens ─┼─ render at C_pred_f + δ1    → L_diffusion  (no GT)
                 │                                                   │
                 ├─ render at C_pred_f + δ2    → L_diffusion  (no GT)
                 │                                                   │
                 └─ render at C_pred_f + δ3    → L_diffusion  (no GT)
                                                    │
                               pairwise L_consistency across overlap

   DECODE at inference: user supplies C_edit, render once.
```

### 7.2 The Key Commitment

- No architectural prescription for camera/time separation. The monolithic
  encoder emits `(world_tokens, camera_tokens, time_tokens)` as three
  output heads, but all three live in the same representation space
  upstream.
- Invariance comes entirely from training pressure.
- Most bitter-lesson-shaped of the three.

### 7.3 Training Contract

Per step, K cameras are rendered:

1. Encoder runs on video → `(world_tokens, camera_tokens, time_tokens)`.
2. Compute K query poses: `C_pred_f` (GT direction) plus K-1 perturbed
   poses `C_pred_f + δ_k` where δ_k is sampled from a curriculum
   distribution (small early, larger later).
3. Render world_tokens under each of the K poses.
4. Loss:
   - `C_pred_f` → photometric L1 + LPIPS vs GT.
   - `C_pred_f + δ_k` (no GT) → diffusion-as-loss (frozen video/image
     diffusion score through renderer).
   - Pairwise consistency: overlapping image regions across rendered
     cameras must agree (diffusion-score on cross-view warping, or direct
     epipolar projection loss).
5. Crop + chunk-swap as usual.

### 7.4 When Camera and Time Enter

- Camera and time are **output tokens** of the single encoder, not inputs.
- The decoder renders via the rasterizer, which takes an explicit pose —
  but the pose can be either predicted or user-supplied at inference.
- Architecturally no commitment to where the factorization happens. The
  training loss forces it.

### 7.5 Leakage Prevention

- F1 (camera): if `world_tokens` carried camera info, renders at K
  different cameras would produce K different world interpretations,
  violating consistency. The only stable equilibrium is view-invariant
  world tokens. Invariance is the fixed point of the loss.
- F2 (cheating splats): same mechanism — a cheating splat set works for
  one camera and fails the others.

### 7.6 Failure Modes

- **K matters enormously.** Small K = weak pressure = weak invariance.
  Large K = compute cost scales linearly.
- **δ distribution matters.** If all perturbations are small, the
  invariance only holds locally around the training trajectory. Large
  perturbations push novel-view but may be too hard to converge.
  Curriculum from small to large is probably necessary.
- **No explicit check** that world tokens are actually camera-free. The
  invariance is implicit; if the model finds a degenerate solution where
  world tokens carry camera that somehow produces consistent renders
  anyway (e.g., the camera token also carries a correction that cancels
  out), the loss doesn't catch it.
- **Compute cost.** K renders per step plus K pairwise consistency
  losses. Roughly K-1 extra forward passes through the renderer + teacher.

### 7.7 Where It Sits in Mechanism Space

| Mechanism | Used? |
|-----------|-------|
| Information bottleneck | no |
| Dimensional bottleneck | no |
| Adversarial recovery | no |
| Multi-condition consistency | ★ Primary |
| Augmentation invariance | ★ Primary (perturbation IS the loss) |

### 7.8 Iteration Ideas

- Stage K over training: K=2 early (just GT + one perturbed), K=4-8 later.
- Curriculum on δ: start with 5° perturbations, grow to 45°+ as training
  progresses.
- Add adversarial recovery head as a diagnostic (not as a loss) to
  measure how much camera info actually remains in world_tokens.
- Optional: add an information-bottleneck regularizer on the encoder
  output to prevent the degenerate solution where camera info is carried
  but cancelled.

---

## 8. Head-to-Head Debate

### 8.1 Against F1–F6

| Failure mode | Architecture A | Architecture B | Architecture C |
|--------------|:-:|:-:|:-:|
| F1 camera leakage into tokens | Partial (bottleneck) | Strongest (adversarial) | Emergent (consistency) |
| F2 cheating splats | ✓ | ✓ | ✓ |
| F3 traj-geom ambiguity | N/A at MVP | N/A at MVP | N/A at MVP |
| F4 long-horizon drift | Out of scope | Out of scope | Out of scope |
| F5 latent cheating | Low risk (bottleneck) | Low risk (adversarial) | Medium risk (needs IB regularizer) |
| F6 low-rank motion | Orthogonal | Orthogonal | Orthogonal |

### 8.2 Axes of Comparison

**Simplicity (bitter-lesson alignment).**
C > A > B.
C is one encoder, one forward pass shape, training does all the work.
A adds architectural commitments (info bottleneck, dimensional
asymmetry). B is the most prescriptive.

**Compute cost at training time.**
C > A ≈ B at K cameras.
A and B are ~1x a standard encoder-decoder. C is K× in the render/loss
path (though not K× in the encoder pass). Most of the cost in C is at the
rasterizer and teacher, not the transformer.

**Robustness of the leakage-prevention mechanism.**
B > C > A.
B's adversarial signal is the most direct and can be diagnostic even when
it fails. C's consistency pressure is strong but has the "canceled
leakage" degenerate solution. A's bottleneck is the weakest in isolation
(the encoder can still infer camera from content).

**Training stability.**
A ≈ C > B.
B's adversarial training is the known failure mode. A and C are
stable-by-default.

**Scaling behavior.**
C > A > B.
C has one loss family that composes with scale: more data, bigger
transformer, larger K, larger δ. A scales similarly but has a
dimensional bottleneck hyperparameter that needs retuning. B's
adversarial head is a second model that must scale with the world
encoder; historically these games get harder at scale.

**Recoverability from failure.**
C > A > B.
If C's pressure isn't enough, bolt on an adversarial diagnostic head (A's
or B's mechanism) as a targeted fix. If A's bottleneck is too tight,
expand it. If B's adversarial game destabilizes, the whole thing can
collapse and the fix is unclear.

### 8.3 Composability

The three architectures are not mutually exclusive. C is the weakest
commitment; A and B are additive on top of C:

- **C + A**: multi-camera consistency pressure + information bottleneck /
  dimensional asymmetry. No adversarial head.
- **C + B**: multi-camera consistency + adversarial recovery on
  world_tokens.
- **C + A + B**: everything. Belt and suspenders. Expensive.

The natural path is to start with C, diagnose with an adversarial head (so
borrow B's diagnostic without making it load-bearing), and only escalate to
A or B if C's pressure is insufficient.

---

## 9. When to Pick Each

- **Pick A if** you want clear architectural semantics. The blind encoder
  makes it easy to explain and audit what the model can and can't see.
  Good for a paper; good if you want to commit to specific invariants.
- **Pick B if** leakage is known to be severe and you've tried C first.
  The adversarial signal is the most direct attack on F1. Accept the
  training instability as the cost of directness.
- **Pick C if** you want the minimal architectural commitment and trust
  the training-time pressure to do the work. This is the
  bitter-lesson-shaped default and the right first bet.

---

## 10. Diagrammed Composition — What Actually Lives Inside the Encoder

All three share the encoder shape question: how are world / camera / time
output tokens produced? The difference is whether the encoder has access
to conditioning information.

```
     Video patch tokens (T, N_patches, D)
              │
              ▼
     ┌────────────────────────────────────┐
     │ Transformer trunk                  │  ← A: no camera/time seen
     │                                    │  ← B: separate trunk from
     │                                    │     pose encoder
     │                                    │  ← C: monolithic, one trunk,
     │                                    │     output heads split
     └────────────────────────────────────┘
              │
              ▼
     ┌────────┴────────┬─────────┬─────────┐
     │                 │         │         │
     ▼                 ▼         ▼         ▼
  world_tokens   camera_tokens  time_tokens   (optional aux)
  (large D)      (small D)      (small D)
```

Where the conditioning enters:

| Stage | Architecture A | Architecture B | Architecture C |
|-------|----------------|----------------|----------------|
| Encoder input | No camera/time | World encoder: no; Pose encoder: yes (implicit via video motion) | No (monolithic sees only video) |
| Encoder output | world, camera, time heads | world (main), camera+time (parallel stream) | world, camera, time heads |
| Decoder | query `(C_q, t_q)` cross-attends into world | query `(C_q, t_q)` cross-attends into world, optionally reads camera/time tokens | same as A |
| Rasterizer | query `C_q` as SE(3) | query `C_q` as SE(3) | query `C_q` as SE(3) |

---

## 11. Open Questions (applicable to all three)

- **What δ distribution for camera perturbation?** Uniform small angle,
  truncated Gaussian, curriculum, learned? The right answer probably
  depends on target edit range at inference.
- **How to parameterize chunk swap?** Random chunk-pair sampling from the
  same clip? Fixed stride? Overlapping vs disjoint chunks?
- **Where should time_tokens commit?** If we want time-invariant world
  tokens, time needs its own pathway. For dynamic scenes, time is
  time-conditioned at the splat level (the splats change over time).
  Cleanest story: world_tokens are time-invariant identity; splat
  decoding is time-conditioned via time query. Splats are the
  time-conditioned projection of time-invariant world tokens.
- **How to handle unposed inference** (implicit-camera mode)? All three
  architectures currently emit `camera_tokens` as a per-frame prediction,
  which can serve as the pose at inference. Relies on the prediction
  being plausible; no GT pose is needed.
- **What does the diffusion-as-loss teacher actually score?** A single
  rendered frame? A short clip? With or without camera conditioning in
  the teacher? The teacher model's data distribution matters — if it's a
  text-to-video model, it may reward textures that violate 3D
  consistency.

---

## 12. Recommended First Experiment

Gun-to-head: **start with Architecture C.** Monolithic encoder, K=2
cameras per step (GT + one perturbed), small δ (≤10°), photometric +
diffusion-as-loss + a simple cross-view consistency penalty. Skip the
adversarial head and the bottleneck.

If after a reasonable training budget the world tokens fail a diagnostic
(adversarial head trained post-hoc recovers camera from them at > chance),
add A's dimensional bottleneck on `camera_tokens` / `time_tokens`. If that
still fails, add B's adversarial head as a real loss, accepting the
training-instability cost.

Do not start with B. Adversarial training is a tax you take when you know
you need it, not a default.

---

## 13. Revisit Triggers

Update this doc when:

- Any of the three architectures is actually implemented and produces a
  result that contradicts the analysis here.
- A fourth mechanism for leakage prevention emerges (e.g., contrastive
  across videos, explicit gauge fixing, latent-space mixing augmentation).
- The training recipe in Section 1 changes. This doc assumes the recipe
  is fixed; if it changes, the architecture comparison needs to be
  redone.
- A new teacher-model option (diffusion, GAN, contrastive) changes the
  post-training cost calculus.
