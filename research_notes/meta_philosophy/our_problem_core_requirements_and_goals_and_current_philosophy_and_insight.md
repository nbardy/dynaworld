# Our Problem — Core Requirements, Goals, Current Philosophy, and Insight

Companion doc for architecture design prompts. Describes the DynaWorld project's goal, the training-data signal we actually have, the inference-time behavior we require, the unavoidable structure we must respect, and the current meta-philosophy.

**Important caveat for any model reading this:** the *requirements, goals, data contract, unavoidable structure, and philosophy* in this doc are load-bearing. The *specific ideas surfaced during exploration* (scene/camera token split, chunk-swap, crop-as-extrinsic, GAN post-training, etc.) are artifacts of one conversation and should NOT be treated as fixed. Treat them as prior art to argue with, not as constraints to preserve. If your analysis concludes a different factorization is better, say so.

---

## 1. Project goal (load-bearing)

DynaWorld is a **video ↔ splats** modality shift for Hollywood-quality filmmaking.

- **Encode**: a video clip (possibly from a single camera) becomes a compact **dynamic 3D Gaussian splat scene**.
- **Edit**: the user moves the camera inside that scene (primary use case) or runs classical FX on the splat representation (secondary).
- **Decode**: the edited scene re-renders as a video.

Video diffusion is complementary, not competing: it is the generative layer; DynaWorld is the exploratory layer on top of it.

MVP: import video → train splat → edit camera path → bake render (with diffusion refinement). The MVP is explicitly camera-path editing, not scene generation.

Phase II is interactive manipulation (physics handles, agents). No planned Phase III for text-to-world.

## 2. Training-data contract (load-bearing)

- **Primary data**: self-supervised, single-camera, posed video. Posed either via DUSt3R-style prebake or via an implicit-camera branch that predicts camera from the video itself.
- **Not available**: large-scale multi-view paired data. We explicitly do not want to require it, because robust web-scale multi-view data does not exist.
- **Supervision signal**: pixel-space reconstruction against GT frames after differentiable rendering. Optional depth/flow/mask aux losses. No direct 3D labels.

### What this signal does supervise

- temporal coherence across frames inside a clip
- reconstruction quality from the *training* camera trajectory
- scale / depth implicitly via parallax over a moving camera
- (with pose prebake) absolute camera path

### What this signal does NOT directly supervise

- **novel-view correctness**: no GT pixel exists for unseen cameras at any time
- **time-invariant scene identity**: no paired "same scene at different times from a different camera"
- **cross-clip scene identity**: no cross-video correspondence
- **decomposition into camera vs. scene vs. time**: the signal is the composition, not the factors

The axes in the second list are the ones architecture must supply, because data cannot.

## 3. Inference-time requirements (load-bearing)

The deployed model must:

- accept a video clip (single camera, plausibly unposed in the wild)
- emit a splat representation the user can render from arbitrary novel cameras
- preserve dynamic content (humans, water, foliage, events) across novel camera paths
- run on long clips, not just 16-frame windows — ideally streaming / rolling
- support editing the camera path independent of the video content that was encoded

### Distribution shift axes at inference

- **camera axis**: train cameras are the clip's trajectory; inference cameras are arbitrary
- **time axis**: possibly long-horizon, beyond training clip length
- **scene axis**: unseen content at inference; backbone must generalize

## 4. Unavoidable structure (load-bearing)

These are facts, not priors to be earned:

- 3D space is 3D. Geometry respects perspective projection.
- Time runs forward. Dynamics are causal during streaming, smoothable offline.
- Cameras project via extrinsics + intrinsics.
- Splats are set-permutation-invariant.
- The single-camera-at-instant-t signal carries no direct multi-view info at that t.

Priors that match these are "free" (convolutions, attention, Gaussian splats, extrinsics-as-inputs, time-causal memory).

## 5. Current meta-philosophy (load-bearing)

### North star

> **Simplest architecture + objective whose supervision mechanism reaches the invariants your data cannot directly constrain.**

Supervision can live in an architectural seam, a training objective, an augmentation regime, a post-training reward, or as an emergent property of scale plus the right loss. Factorization is one family among several. A clean objective on an unfactored model often beats a factored architecture with a trivial objective. Do not pre-commit to factorization.

See `architecture_design_north_star.md` in this folder for the full principles. Short form:

1. Pick the coarsest structure that matches unavoidable structure, but only commit to the structure the mechanism actually needs.
2. Put the supervision mechanism on axes the self-supervised signal cannot reach (camera, time-invariant identity) — whichever mechanism family (objective, augmentation, seam, post-training, emergent) supplies the invariant with the least committed structure.
3. "Simplest" is hypothesis-space size, not component count. Count ruled-out solution families, not boxes.
4. Match prior strength to data regime. We are weak-data on the novel-view axis, so strong correct priors there are earned — but a well-designed objective can often carry the weight of architectural priors.
5. Inference-time distribution shift is part of the target, not a surprise at deploy.

### Core beliefs (from the project README)

- World models are video models; a strong video backbone already carries geometry, motion, and lighting structure.
- DynaWorld is modality shift, not world generation.
- Video ↔ video is the training contract. Splats sit in the middle.
- Static and dynamic are the same problem.
- Modalities don't require pretraining. A lightweight splat head on a frozen video backbone is cheap adapter training.
- Supervision stays in pixel space.
- Memory goes to dynamic scene state, not luxury parameters.

## 6. Known failure modes we are designing against

Named explicitly so the architecture can be asked "what rules this out":

- **F1 — Camera leakage into scene tokens.** Even with a camera-token output head, image tokens may implicitly carry pose, so swapping the camera token at render time fails to produce correct novel views.
- **F2 — Cheating splats.** Splats overfit to look good only from the encoded camera and fail under novel angles because no loss pressured them otherwise.
- **F3 — Trajectory-geometry ambiguity (implicit-camera mode).** Small mis-predicted camera pose makes the model warp geometry to match pixels; geometry and camera "gaslight" each other into collapse within a few frames.
- **F4 — Long-horizon drift.** Recurrent scene memory accumulates error; small per-step mistakes compound over hundreds of frames.
- **F5 — Latent cheating.** Scene memory or backbone features absorb geometry that should live in the explicit splats, leaving splats as a thin rendering shell rather than a real 3D asset.
- **F6 — Low-rank motion assumption breaks.** Water, smoke, crowds, and topology changes are not naturally low-rank; compressions that assume they are collapse quality.

## 7. Ideas surfaced during exploration (NON-load-bearing — do not anchor on these)

**Read this section as prior art to argue with, not as targets to justify.** A proposed solution does not have to adopt any of these. It does have to say whether and why.

**Explicit warning:** the scene/camera/time token split below is the *current leading candidate inside one conversation*, not the frame. We strongly suspect it may be a local minimum — it leans entirely on the architectural-seam family and assumes factorization is the supervision mechanism. Objective-family solutions (AR with principled masking, diffusion with invariance-inducing conditioning, forcing-family training, contrastive or self-distillation losses on unfactored models) may be genuinely cleaner for this problem, because they can induce the same invariants at scale without explicit architectural commitments. Treat the ideas below as the ones we've already thought of; the good idea is probably the one we haven't.

Ideas indexed by supervision-mechanism family:

**Architectural-seam family:**
- **Explicit scene / camera / time factorization.** Split the token output into (time-invariant scene tokens) + (camera token) + (splats as time-conditioned projection). Lets you write a chunk-agreement MSE/KL loss on scene tokens across chunks. Scene ≠ splats: scene tokens are time-invariant identity; splats are instantaneous state. *Risk: same leakage problem moves up one layer unless an asymmetry is added.*
- **Persistent scene tokens + ephemeral chunk tokens + causal memory update** as the long-horizon contract.
- **Anchor-actor graph tokens** as an alternative to strict static/dynamic splits.

**Augmentation family:**
- **Same-video chunk swap.** Encode two chunks of the same video; use chunk-1's scene tokens with chunk-2's camera + time; train against chunk-2's GT frames. Can be applied on a factored or unfactored model.
- **Crop / virtual reprojection as pseudo multi-view.** Crop a corner of high-res frames and treat the crop as a camera extrinsic shift (ray-shift, not perspective warp).

**Objective family:**
- **Forcing-family AR training** (self-forcing / rolling forcing / diffusion forcing) to close the AR train/inference gap. *Novel views are a special case of this gap* — if the rollout includes camera perturbations, forcing supervises the novel-view distribution directly, not just temporal drift.
- **Diffusion objective with camera conditioning**: denoise splats (or video latents) from noise, conditioned on a camera path; the right conditioning pattern may force the representation to be camera-swappable without explicit factorization.
- **Diffusion-as-loss (score distillation)**: a frozen pretrained video diffusion model scores rendered outputs via SDS / VSD / DiffRep-style Jacobian pullback, backpropping off-manifold supervision through the renderer into splats or tokens. Can supervise novel views where no GT frame exists. See the "Diffusion as loss vs diffusion as conditioning" entry in `potential_directions_index.md`.
- **Contrastive / self-distillation**: two views of the same data through the same encoder should produce agreeing representations on the invariant axis.

**Post-training family:**
- **GAN / reward-style post-training on novel-view renders.** Render from encoded and novel cameras; train a discriminator to tell them apart; push the model to fool it. Compatible with any pretraining mechanism above.

**Readout decisions (partially orthogonal):**
- **Local-time spline readout per chunk**, not a global-T MLP, for long-horizon stability.
- **Oscillator residual branch** for periodic ambient motion (water, foliage).

The preferred response is not "pick one from each list." The preferred response is "name the mechanism family that supplies the needed invariants cheapest, then design within that family — and if none of the ideas above fit, propose one that isn't on the list."

Ideas explicitly NOT yet in scope (say so if you want to add one):

- Text-to-world generation.
- Multi-view finetuning as a hard requirement.
- Full foundation-model rewrite of the video backbone.

## 8. Anti-patterns observed in prior rounds

Architectural responses that miss the point tend to:

- Enumerate temporal-evolution decoder variants (answers F6 but not F1/F2/F3).
- Answer unposed streaming SLAM when the bottleneck is novel-view generalization (wrong question).
- Propose "persistent scene memory" without naming the asymmetry that prevents camera from leaking into it (re-introduces F1 at a new layer).
- Stack weak regularizers (masking + crop + chunk swap + GAN) to compensate for a factorization unwilling to be committed to.
- Map deep-learning modules to classical-CV names (ICP, BA, loop closure) as mnemonics masquerading as proofs.

If a proposal pattern-matches to any of these, audit it explicitly against F1–F6 before defending it.

## 9. What a good response looks like

- Answers the five-axis alignment check before proposing anything.
- Identifies which of F1–F6 each branch rules out and which it does not.
- Spans at least three supervision-mechanism families across branches (objective, augmentation, architectural-seam, post-training, emergent). A response that only proposes variants of the scene/camera/time split has failed the divergence requirement.
- Names the concrete writable loss / objective / augmentation pipeline for each branch, not just "we condition on camera".
- Explicitly addresses: is the supervision mechanism in the architecture, the objective, the data pipeline, the post-training step, or emergent from scale — and *why that family* for our data regime.
- States whether forcing-family AR and diffusion-family objectives are orthogonal to novel-view generalization, load-bearing for it, or a cleaner substitute for factorization.
- Pays the cost of saying "I don't know" when the alignment axes don't uniquely pick a winner.
- Is willing to reject the scene/camera/time split as the frame if a different (or no) factorization is cleaner. The job is not to justify our current idea.
