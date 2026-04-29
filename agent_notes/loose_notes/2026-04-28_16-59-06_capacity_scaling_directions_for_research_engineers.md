# Capacity Scaling Directions for Research Engineers

Date: 2026-04-28

## Context

The strongest current same-source recipe is not "V-JEPA alone." It is the
interaction of:

- precomputed V-JEPA 2.1 ViT-B/384 tokens
- static/dynamic token split, currently 96 static / 32 dynamic
- 4 cross-attention layers
- strong RGB/uniform/token/head init
- 8192 decoded splats
- camera-clamped control, so improvement is less likely to be pure camera compensation

That recipe has shown much better same-source metrics than the recent plain
unconditioned and local-token baselines. The important caveat is that these are
still mostly single-clip or tiny-data overfit results. Future capacity work
should be judged by held-out-camera and held-out-sample performance, not by
source-view PSNR alone.

## Core Thesis

Increase capacity in the world-token and splat-decoder path, not in the camera
escape hatch.

The model should learn a representation:

```text
video / feature tokens x -> world tokens z -> time-conditioned splats G_t
camera c_t + G_t -> rendered image I_hat[t, c]
```

The capacity we want is in `z -> G_t`: more geometry, appearance, temporal
basis, and cross-view consistency. Capacity in the learned camera path can
improve source reconstruction while weakening the intended 3D contract.

## Capacity Levers

### 1. Splat Count

Current useful setting: `8192` splats.

Recommended ladder:

```text
8192 -> 16384 -> 32768 -> 65536
```

Only scale one rung at a time. Record render memory, step time, and whether
camera-heldout metrics improve. More splats can memorize the training view
without improving novel camera synthesis.

Primary failure mode:

```text
train PSNR increases
camera-heldout PSNR stays flat or drops
decoded splats become a view-dependent billboard cloud
```

### 2. Token Count and Static/Dynamic Split

Treat static/dynamic as a capacity split, not a semantic classifier.

Recommended ladder:

```text
128 tokens: 96 static / 32 dynamic
192 tokens: 144 static / 48 dynamic
256 tokens: 192 static / 64 dynamic
```

Do not first jump to a huge dynamic bank. Static capacity should carry stable
scene support; dynamic capacity should explain residual time variation. If
dynamic capacity is too large too early, the model can use time as a memorized
index.

Track:

- static-token contribution to heldout-camera quality
- dynamic-token contribution to temporal residual quality
- decoded XYZ motion magnitude
- dynamic opacity and scale distributions

### 3. Cross-Attention Depth and Width

Current useful setting: 4 cross-attention layers.

Recommended ladder:

```text
layers: 4 -> 6 -> 8
model_dim: 128 -> 192 -> 256
decoder bottleneck: 256 -> 384 -> 512
```

Hold token count and splat count fixed while testing attention depth/width.
Otherwise it will be impossible to tell whether gains came from more tokens,
more splats, or better feature-to-world fusion.

Expected domains:

```text
V-JEPA feature tokens: high-level semantic/motion features, not pixel-aligned RGB
world tokens: compact latent scene state
decoded splats: explicit raster substrate with position, scale, opacity, color
```

The cross-attention module should be big enough to map semantic video evidence
into scene state, but not so large that it memorizes clip identity before the
benchmark can measure generalization.

### 4. Dynamic Basis Capacity

Current dynamic path should be scaled after static geometry is stable.

Recommended ladder:

```text
motion basis: 8 -> 12 -> 16
dynamic max frequency: keep fixed first
motion extent: keep clamped first
rotation extent: keep clamped first
```

Only increase dynamic motion ranges after metrics show underfitting of true
motion. If increasing motion range improves source PSNR but hurts heldout-camera
PSNR, it is probably camera/view compensation or temporal memorization.

### 5. Decoder and Head Capacity

The strong init is part of the current useful recipe. Keep it on while scaling.

Recommended ladder:

```text
head hidden dim: 64 -> 128 -> 192
MLP hidden layers: 1 -> 2 -> 3
init mode: strong RGB/uniform/token/head init fixed across comparisons
```

Do not compare weak init at one capacity against strong init at another
capacity. Init and capacity interact strongly; the benchmark should isolate
them.

### 6. Feature Memory and Conditioning

V-JEPA static/dynamic is currently the useful treatment. Other video features
are still worth testing, but only after the feature-cache contract is source
faithful.

Before comparing LTX/Wan/other feature backends, verify:

- same source frames
- same resize/crop
- same temporal window
- feature metadata includes layout, frame indices, and source-camera identity
- render and loss resolution match the intended benchmark resolution

## Recommended Ablation Ladder

Run capacity changes in this order:

1. Same-source sanity: reproduce the clamped V-JEPA static/dynamic baseline.
2. Multicam single-sample: train two cameras, validate on the third.
3. Multicam overlap heldout: use a heldout camera within the train camera arc.
4. Multicam outside heldout: use a harder heldout camera outside or near the edge.
5. Ten-sample multicam: train across 10 three-camera clips.
6. Mixed data: add 20 single-camera YouTube clips with balanced sampling.
7. Capacity sweep: scale tokens, splats, and MLP width only after the benchmark is stable.

## What Counts as a Real Capacity Win

A capacity change is useful if it improves at least one of:

- camera-heldout PSNR/SSIM at fixed train-camera quality
- sample-heldout reconstruction from video conditioning
- foreground/high-pass quality
- temporal consistency without exploding camera motion
- equal or better quality at lower export size

It is not enough for source-view PSNR to improve. In this project, source-view
overfit is cheap. The research question is whether video-conditioned world
tokens can decode to splats that survive new views and new clips.

## Guardrails

Always keep these controls in the matrix:

- unconditioned tokens with the same static/dynamic split and strong init
- V-JEPA static/dynamic with the same init
- local encoder static/dynamic
- free splats or known-camera 3DGS-style upper reference when available
- camera-clamped and camera-free variants when the camera path changes

The unconditioned run is the most important guardrail. If a larger conditioned
model cannot beat the unconditioned model on heldout-camera or sample-heldout
metrics, it is not using video conditioning in the way we need.
