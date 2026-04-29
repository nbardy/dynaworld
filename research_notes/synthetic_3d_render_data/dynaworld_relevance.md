# How Synthetic 3D Render Data Fits the DynaWorld Contract

Read `../../DATASET_V1.md` and the top-level `../../README.md` first. This
file builds on those.

## The contract, restated

DynaWorld's training rule:

> `Video <=> Video` is the only training data for world models that scales.
> Render splats and compare directly against ground-truth video. No fake 3D
> labels. No synthetic ground truth.

The strict reading: any pixel that the model produces a loss against must
come from a real captured video. Synthetic frames are not allowed as GT.

This is correct for the world-model thesis: if we want the model to learn
the distribution of real video, we must use real video as the supervisory
signal.

## Where synthetic still helps

The contract bans synthetic *as GT*. It does not ban synthetic for
diagnostics, probes, or pretraining pressure that doesn't write the loss.
The four legitimate uses:

### 1. Novel-camera consistency probes (unit tests)

**Problem**: real video can't tell you "is this novel-camera output
correct?" — it can only say "does it look plausible?" Real multi-camera
data (AIST, DeepView, Neural 3D, ViVo) gives you GT for held-out cameras at
the same time, but only for those specific scenes.

**Synthetic's role**: render the same animated scene from cam A and cam B
in v2. Train on `video_A → world tokens → render at B's pose`. The
synthetic GT at cam B is **not the production loss** — it's a unit test
that says "for this controlled scene, did the model actually learn to swap
cameras?"

This is fine because:
- The synthetic clip never enters the production training distribution.
- It's used as a structured architectural assertion, not as scaling data.
- Failure on the probe means the architecture is broken; success doesn't
  prove generalization (real GT does that).

### 2. Camera-leakage stress test

The hardest failure mode in `video → world tokens → splats` is **camera
leakage**: the model hides camera-specific information inside the video
tokens, so swapping the camera token doesn't actually change anything.

Real video can't isolate this. You can't take a real handheld clip and ask
"what would this look like from 30° rotated?" without already having that
view (which only multi-camera datasets provide, for their specific scenes).

Synthetic can. Render the *exact same scene at the exact same time from N
cameras*. Then:
- Encode video from cam A, decode at cam A → should match cam A GT.
- Encode video from cam A, decode at cam B → should match cam B GT.
- Mix cam A's video tokens with cam B's camera token → should produce
  cam B's view.

If the model fails the third test, the camera information is leaking into
the video tokens. Synthetic is the only way to construct this test
exhaustively.

### 3. Pretraining priors before paired-camera finetune

The current paired-camera datasets (AIST, DeepView, Neural 3D, ViVo) total
a few thousand sequences. That's a finetune set, not a pretraining set.

Synthetic can scale: BlenderProc + Blender Foundation scenes + Polyhaven
HDRI randomization could emit ~100K+ clips with paired multi-camera views.

**The legitimate use**: pretrain the world-token + splat-decoder on
synthetic paired-camera data to learn the *form* of the camera-swap
behavior, then finetune on real multi-camera data so the *distribution*
matches.

This is structurally similar to how vision models pretrain on ImageNet
(synthetic-feeling, distribution-shifted) and finetune on the real target
domain. The pretraining isn't claiming to be the deployment distribution.

**Risk to manage**: synthetic pretraining could bias the model toward the
synthetic camera-noise distribution (perfect calibration, no rolling
shutter, no real lens artifacts). Mitigation: domain randomization over
camera-noise during synthetic pretrain.

### 4. Bullet-time / Nova-side feature work

This is *not* world-model research. Nova's iOS app has separate features
(deblur, stabilization, bullet-time generation) that have application-level
goals: ship a usable feature in the app.

Application training has different rules from research training. Nova can
train freely on synthetic data because the goal is "look good on the
phone," not "discover the structure of the real world."

The synthetic 3D pipelines we own (v2 Blender) are most directly useful
here. DynaWorld benefits indirectly: a working bullet-time feature in Nova
is a customer of DynaWorld's eventual world-model output, and the
synthetic pipelines used to develop it can be reused for DynaWorld probes.

## Where synthetic violates the contract

These are **banned**:

- **Synthetic clips in the production training mix as if they were real.**
  Mixing 50% real + 50% synthetic and training on both with the same loss
  teaches the model that synthetic is part of the distribution. It isn't.

- **Synthetic GT for the primary loss.** The loss must compare model output
  against real captured pixels. Comparing against rendered pixels embeds
  the renderer's biases into the model.

- **Synthetic 3D labels.** The whole point of `video <=> video` is to avoid
  3D labels. Don't generate synthetic depth / normals / poses and train
  the world tokens to predict them. The render is the supervision.

- **Synthetic as evaluation set.** A model that scores well on synthetic
  evaluation has only learned to do synthetic well. Real multi-camera GT
  (AIST / DeepView / Neural3D / ViVo) is the evaluation; synthetic is at
  most a regression smoke test.

## Concrete proposal: DynaWorld synthetic supplement track

Conditional on agreement that the four uses above are legitimate:

### Step 1 — fix v2 minor issues
- Replace `find_target_object` silent fallback with typed `TargetSpec`.
- Emit camera metadata in DynaWorld's `CameraSpec` schema (see
  `../../data/CAMERA_CONTRACT.md`).
- Add a config preset that renders N cameras of the same scene at the same
  time window.

### Step 2 — populate Blender Foundation stubs
- Extract `spring.zip`.
- Download Sprite Fright, Charge, Cosmos Laundromat into respective stubs.
- ~10 production-quality scenes total.

### Step 3 — wire BlenderProc on top
- BlenderProc randomizes camera, lighting, materials.
- Each scene → ~1000 clips with synchronized multi-camera views.
- Total: ~10K synthetic multi-camera clips.

### Step 4 — define probe and pretraining splits
- **Probe split**: 100 clips. Used as unit-test for novel-camera consistency
  + camera-leakage. Not in any training mix.
- **Pretraining split**: 9.9K clips. Used only for the pretraining stage,
  before paired-camera finetune on real data. Loss decay schedule should
  zero out the synthetic loss by end of pretraining.

### Step 5 — instrument as smoke tests
- Synthetic camera-leakage test runs as part of DynaWorld's CI.
- Failure on the probe blocks merge; success is a necessary-but-not-
  sufficient condition.

## Open questions

These are spelled out further in `open_questions.md`:

- What's the right domain-randomization budget for synthetic pretraining?
  (Too little → renderer artifacts in the prior. Too much → the prior
  doesn't transfer.)
- Can we use the synthetic-vs-real distribution shift as an *additional*
  loss signal — train an adversarial discriminator that tries to tell them
  apart, push the world tokens toward real-only?
- Does the existing v2 Cycles render look match BEDLAM-Unreal closely
  enough to use both as a single synthetic source, or are they distinct
  distributions?
- For bullet-time validation specifically: should we render synthetic
  clips that mimic the AIST camera rig exactly (9 fixed cams in known
  positions), so the synthetic and real evals are directly comparable?

## Summary

Synthetic 3D render data is not a substitute for real `video <=> video`
training. It's a complementary tool for **probes, stress tests, and
structured pretraining**. The contract is preserved as long as the loss of
record stays on real captured pixels, and synthetic stays in the
diagnostic / pretraining role.

The pipelines for this already exist (v2 Blender + BlenderProc + Blender
Foundation scenes). The data sources already exist (Tier 1–4 in
`scene_sources.md`). The integration is half a day to a week of work, not
a new project.
