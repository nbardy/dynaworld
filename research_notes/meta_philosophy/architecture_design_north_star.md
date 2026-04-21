# Architecture Design — North Star

## One-liner (north star)

> **Simplest architecture + objective whose supervision mechanism reaches the invariants your data cannot directly constrain.**

The supervision mechanism can be an architectural seam with a loss on it, a training objective that implicitly forces the invariant, a data-augmentation regime that synthesizes the missing signal, or an emergent property of scale plus the right loss. Factorization is one family of solutions, not the only one. Often the cleanest answer is an objective on an unfactored model, not a factored architecture with a simple loss.

Use this as the default filter when proposing or rejecting architectural changes. If a proposed mechanism supervises an invariant your data already reaches through standard reconstruction loss, it is avoidable. If it reaches an invariant the data cannot reach — novel view, cross-chunk identity, time-invariant scene structure — it is earning its complexity regardless of whether it lives in the architecture or in the objective.

## Principles

### 1. Pick the coarsest factorization that matches unavoidable structure

Unavoidable structure is physics and measurement: 3D space is 3D, time runs
forward, cameras project, sets are permutation-invariant. Factorizations that
match unavoidable structure age well (convolutions, attention as set
equivariance, splats as 3D primitives, extrinsics as explicit inputs).

Avoidable structure is a guess about what matters ("foreground vs background",
"objects vs texture", "motion vs appearance"). These are often wrong and they
do not age well. Let data handle them.

Rule: keep priors that describe facts. Drop priors that describe hunches.

### 2. Put your supervision mechanism on the axes your data cannot reach

Self-structured video supervision gives you temporal coherence, reconstruction
loss, optical flow, multi-frame consistency. It does not give you novel-view
signal, cross-scene identity, or explicit camera factorization.

The axes your self-supervised signal cannot reach are exactly the axes that
need *some* supervision mechanism beyond standard reconstruction — because the
axes the data already covers do not need extra structure.

Supervision-mechanism families to consider, in rough order of increasing
architectural commitment:

- **Objective-family**: AR with principled masking, diffusion with conditioning
  that makes the invariant emerge, forcing-family training (self-forcing,
  rolling forcing, diffusion forcing), contrastive losses between views of the
  same data, self-distillation.
- **Augmentation-family**: synthesize pseudo-supervision for the missing axis
  (crop-as-extrinsic for camera, chunk-swap for time, same-video cross-view
  pairs).
- **Architectural seams**: split the representation so a loss can be written
  directly on the invariant (scene vs. camera vs. time tokens with agreement
  losses).
- **Post-training / reward**: refine a model whose pretraining already has
  some prior on the invariant (GAN on novel views, RLHF-style rewards).

These compose. They are also partially substitutable. The right answer is
whichever mechanism supervises the needed invariant with the least committed
structure, given data scale and budget. A clean objective on an unfactored
model often beats a complex factored model with a trivial objective.

For this project the weak axes are:
- camera / viewpoint (no multi-view data)
- time-invariant scene identity (no cross-clip paired data)

Do not pre-commit to the architectural-seam family. Start by asking: *what
invariant do I need and which mechanism family supplies it cheapest?*

### 3. "Simplest" is measured in degrees of freedom, not boxes

Simplicity is the size of the hypothesis space that still contains the target
function, not the number of components in a diagram.

A bigger model with a well-chosen prior often has a smaller effective
hypothesis space than a smaller model with no prior. An explicit scene/camera
split looks like more structure, but it rules out the entire family of
camera-leakage solutions; an implicit single-encoder arch looks simpler but
keeps those solutions live and then requires data or post-hoc refinement to
suppress them.

Count the ruled-out solutions, not the boxes on the whiteboard.

### 4. Bitter-lesson caveat

Sutton's bitter lesson is "do not hand-engineer features that scaling compute
can learn." It is **not** "no priors ever."

- Priors that describe unavoidable structure (convolutions, attention,
  3D geometry, causality in time) compose with scale and age well.
- Priors that guess what matters (feature engineering, hand-crafted object
  models, hard-coded motion models) compete with scale and age badly.

When in doubt, ask whether scale would plausibly discover the prior from data.
If yes, do not bake it in. If no, it is structural and you should.

### 5. Match prior strength to data

- Lots of data + weak priors: foundation-model approach works.
- Little data + strong, correct priors: classical / domain-specific works.
- Little data + weak priors: does not work.
- Lots of data + strong priors: wasteful if prior is right, disaster if prior
  is wrong.

We are in the "little data on the weak axis" regime for camera / novel view.
So strong, correct priors on that axis are earned; weak priors are not an
option.

### 6. Inference-time behavior includes distribution shift

"Align inductive biases to inference-time behavior" is only half right. The
biases have to align to the inference-time *distribution*, including the shift
from training. Novel views are a distribution shift. If training data does not
cover the inference distribution, the architecture must do one of:

- bias toward generalization along the shifted axis,
- augment the data to cover the shift (synthetic multi-view, chunk swap,
  crop-as-extrinsic),
- factor the shifted axis out explicitly so the rest of the model does not
  have to generalize over it.

Doing all three at once is fine, but at least one must be load-bearing.

## Application Checklist

When considering a new architectural change, ask in order:

1. What is the invariant I want this change to enforce?
2. Does my self-supervised signal already supervise that invariant somewhere?
   - If yes, the change is probably avoidable structure — drop it.
   - If no, this is a candidate seam.
3. Does the proposed factorization match an unavoidable structural fact, or a
   guess about what matters?
   - Unavoidable: keep.
   - Guess: let scale handle it instead.
4. How many solution families does this rule out?
   - If it rules out a whole leakage / failure mode family, the added
     structure is earning its place.
   - If it only changes surface area, it is not.
5. Where will I put the loss that supervises this seam, and does my data
   actually give me that signal?
   - If there is no loss you can write against this seam, the seam is
     decorative.

## Anti-Patterns

- Adding a component because it is intellectually tidy, not because a loss
  lives on it.
- Factorizing along axes the data already covers ("motion vs appearance" when
  reconstruction already sees both).
- Stacking weak regularizers (masking + chunk swap + crop + GAN) to
  compensate for a factorization you are unwilling to commit to.
- Confusing "fewer boxes" with "simpler" when the smaller model has a larger
  hypothesis space.
- Bolting post-training refinement onto an implicit arch to recover an
  invariant the arch could have expressed cheaply at a seam.
- **Pre-committing to factorization as the supervision mechanism.** If an
  objective-family or augmentation-family solution supplies the same
  invariant with less committed structure, it is probably the cleaner answer.
  The scene/camera/time split is one candidate, not the frame.
- Anchoring on the current leading candidate because it is concrete. Concrete
  is not the same as correct. The next idea is allowed to be cleaner than
  this one.

## Origin

Distilled from a working session on novel-view training strategies
(`agent_notes/loose_notes/2026-04-21_00-46-28_novel_view_training_strategies.md`)
and the chunk-agreement question
(`agent_notes/loose_notes/2026-04-21_03-03-11_token_agreement_across_chunks.md`),
then sharpened through a back-and-forth on whether the scene/camera token
split earns its complexity.

Keep this file tight. If future insights refine the principles, edit in place
rather than appending a dated section. The goal is a living north star, not a
changelog.
