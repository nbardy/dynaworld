# Thread reflection: small baseline to current 128px Taichi overfit

## Context

This note reconstructs the current thread after several intertwined changes:

- data/video bakes went from the original tiny prebaked clip to 64px/128px 4fps variants
- the 128px config hit NaNs and white/gray renders
- renderer diagnostics identified camera/depth and projection failures
- the trainer moved from dense PyTorch rendering to Taichi/Metal rendering
- capacity was increased from 512 to 8192 explicit Gaussians
- loss, optimizer, init, and config defaults were changed

The user asked to stop and reflect on what originally worked, what changed, what
broke, and where the baseline now stands.

This is a chronology note, not a proof that the current model is correct.

## Short answer

The originally working baseline was simple:

```text
data:        32px, 23 frames, 2fps, old DUSt3R camera bake
model:       128 tokens x 4 splats/token = 512 splats
heads:       one-hidden-layer MLPs with normal PyTorch init
positions:   x,y = tanh(raw) * 1.5
             z   = sigmoid(raw) * 2.0 + 0.5
scale:       exp(raw) * 0.05
opacity/rgb: sigmoid(raw)
renderer:    dense PyTorch renderer, frame loop
loss:        L1 + 0.2 * MSE
optimizer:   Adam, lr 0.005
batch:       all 23 frames per step
```

The important part: it had no custom "careful" head-output init. The learned
tokens started with `torch.randn(...)*1.0`, the heads used default Linear init,
and the nonlinear bounds were simple. It was sloppy, but it gave the 512 splats
some natural diversity.

The first real break after scaling was not "tanh is bad." The first real break
was that the new 128px/4fps DUSt3R camera solve had a different camera/scene
scale. Late cameras moved far enough that the old fixed z slab `[0.5, 2.5]`
put many splats behind or nearly on the camera plane. That caused renderer
NaNs before optimization. Widening z to `[0.5, 5.0]` and guarding projection
made the run stable.

The later quality plateau is a different issue. After we made init more
"controlled" for stability, we likely over-constrained diversity:

```text
token_init_std=0.01
head_output_init_std=0.002
rotation_init=identity
opacity_init fixed
AdamW weight_decay=0.05
```

That made 8192 splats behave less like 8192 independent primitives and more
like repeated sibling templates. The latest patch backs away from that by using
decoded-space uniform xyz bias, larger token/head randomness, random rotations,
and Adam without weight decay for the local overfit probe.

## Known committed stable baseline

The stable committed baseline I can identify is:

```text
be87e96 Record dynamic train matrix
```

At that commit:

File:

```text
src/train_configs/local_mac_overfit_prebaked_camera.jsonc
```

Relevant config:

```jsonc
{
  "data": {
    "sequence_dir": "test_data/dust3r_outputs/test_video_small_all_frames",
    "camera_image_size": 224,
    "camera_focal_mode": "median"
  },
  "model": {
    "size": 32,
    "tokens": 128,
    "gaussians_per_token": 4
  },
  "render": {
    "renderer": "dense",
    "tile_size": 8
  },
  "train": {
    "steps": 1000,
    "lr": 0.005,
    "frames_per_step": 0,
    "eval_batch_size": 4
  }
}
```

`frames_per_step=0` meant all 23 frames each step.

The model code at that commit:

```python
xyz = torch.cat(
    [
        torch.tanh(xyz_raw[..., :2]) * 1.5,
        torch.sigmoid(xyz_raw[..., 2:]) * 2.0 + 0.5,
    ],
    dim=-1,
)
scales = torch.exp(scale_raw) * 0.05
quats = normalize(rot_raw)
opacities = sigmoid(opacity_raw)
rgbs = sigmoid(rgb_raw)
```

There were five separate MLP heads:

```text
xyz_head      feat_dim -> hidden 64 -> 3 per splat
scale_head    feat_dim -> hidden 64 -> 3 per splat
rot_head      feat_dim -> hidden 64 -> 4 per splat
opacity_head  feat_dim -> hidden 64 -> 1 per splat
rgb_head      feat_dim -> hidden 64 -> 3 per splat
```

Each MLP used default PyTorch `Linear` initialization. Tokens used:

```python
self.tokens = nn.Parameter(torch.randn(1, num_tokens, feat_dim))
```

So the old baseline had high token variance and non-tiny default head weights.
It did not try to initialize every field near a specific semantic mean.

## Why the old baseline could work despite being crude

### Small data regime

The old task was much easier:

```text
23 frames
32x32 render/loss
512 splats
old camera path
all frames every step
```

At 32px there are only 1024 pixels. A 512-splat model is already high capacity
relative to the target. Also, all frames were present every optimizer step, so
there was no stochastic window hiding late-frame failures for many steps.

### Initial diversity was accidental but useful

Default token/head init means the decoded splats were not uniform or clean, but
they were diverse. The nonlinear bounds constrained the result enough to stay in
a plausible scene volume, while the raw MLP noise gave different tokens and
sibling slots different starting positions, colors, opacities, scales, and
rotations.

This matters because the current architecture has no classic 3DGS density
control:

```text
no splitting
no cloning
no pruning
no opacity reset
no adaptive spawn near high-gradient regions
```

If initial splats are too correlated, nothing later creates new independent
primitives. The optimizer can move them, but it cannot spawn fresh primitives
where detail is missing.

### The fixed z range matched the old camera path well enough

The old head's z support was:

```text
z in [0.5, 2.5]
```

That was not mathematically principled. It happened to be compatible enough with
the old camera solve. The old baseline did not prove that `[0.5, 2.5]` is a
stable general training contract.

## What changed, chronologically

### 1. New data variants

We added config routing for video variants:

```text
small_32_2fps   -> old 23-frame bake
small_64_4fps   -> new 46-frame bake
small_128_4fps  -> new 46-frame bake
```

There was an intermediate mistake where higher-FPS videos were generated from an
already downsampled 2fps clip. That was corrected: the 64px/128px 4fps bakes
were regenerated from the original source video in one ffmpeg pass.

Important backtrack:

The 128px/4fps run is not merely "the old working video but larger." It is a new
camera-distribution test because DUSt3R solved a different trajectory/scale/FoV
for the newly baked frames.

### 2. Batched dense rendering and diagnostics

The dense renderer was moved toward batched frame rendering. This helped
throughput but also changed how we observed performance.

Diagnostics were added for:

```text
camera z ranges
front/behind splat counts
nonfinite render/power/alpha counts
tile overlap estimates
optimizer/gradient metrics
full-sequence eval scalars
```

This was a good change. It converted "gray/white output" into concrete failure
signals.

### 3. 128px raw config hit NaNs

The 128px/4fps raw config failed before optimizer dynamics could explain it.
Renderer diagnostics showed:

```text
many near/behind-camera splats
huge projection powers
nonfinite alpha prevalues
negative or tiny determinants
NaNs in rendered image
```

The concrete mechanism was roughly:

```text
late camera window + old z slab -> many splats at z <= near plane
projection math uses 1/z and 1/z^2
exp(power) can overflow or become nonfinite
opacity masking too late can produce 0 * inf = NaN
```

So the failure was not initially "LR too high" or "too many splats." It was a
camera/scene-scale mismatch plus renderer robustness issue.

### 4. Finite-depth guard

The projection code now treats non-front splats safely:

```text
MIN_RENDER_DEPTH = 1e-4
front_mask = z > MIN_RENDER_DEPTH
z_safe = z if front else 1
x_project,y_project = x,y if front else 0
opacity = opacity * front_mask
```

This prevents the renderer from using invalid depth values in projection. It is
a numerical guard, not a modeling solution. It stops bad windows from producing
NaNs, but if most splats are behind the camera, the model still cannot render
the frame well.

### 5. Wider depth range

The Gaussian head became configurable:

```text
old: z = sigmoid(raw_z) * 2.0 + 0.5   -> [0.5, 2.5]
new: z = sigmoid(raw_z) * (z_max-z_min) + z_min
```

For the 128px wide-depth configs:

```text
z_min = 0.5
z_max = 5.0
```

This was a first-order fix. Late-window diagnostics improved because many more
splats were in front of the late cameras. But this is still not true camera
normalization. It is a hand-tuned larger canonical volume.

### 6. Taichi renderer

We added a Taichi/Metal renderer path:

```text
renderer: "taichi"
```

It projects our 3D Gaussians with the shared Torch projection code, converts the
2D covariance to Taichi's packed axis/sigma representation, rasterizes with the
vendored Taichi fork, and blends against a white background.

Later the Taichi fork gained native batch rendering:

```text
rasterize_batch([B,G,7], [B,G,1], [B,G,C]) -> [B,H,W,C]
```

This made 128px training much faster and removed the worst dense renderer
bottleneck. The user observed that it converged faster and looked better even at
512 splats. Likely reasons:

- faster iterations, less dense-render overhead
- different renderer numerics/saturation behavior
- stable wide-depth geometry by that point

This improvement should not be confused with the model architecture being
solved. It mostly made experiments practical.

### 7. Frames per step and LR schedule

For 128px experiments we moved away from all frames per step:

```text
frames_per_step = 4, then 8
```

LR also changed:

```text
old: lr 0.005 constant
new: lr 0.001 cosine -> 0.0001
```

This makes the new loss curves not directly comparable to the old baseline.
The old baseline saw every frame every step at higher LR on a much smaller
task.

### 8. Capacity increase to 8192 splats

The active high-capacity config uses:

```text
128 tokens x 64 splats/token = 8192 splats
```

This sounds like many primitives, but they are not independent 3DGS splats in
the classical optimizer sense. Each group of 64 sibling splats shares one
refined token latent. The heads emit per-sibling values, but sibling outputs are
correlated by the same input vector and same head function.

So increasing `gaussians_per_token` gives more output slots, but it does not
automatically give the same freedom as 8192 separately optimized parameter
records plus densification.

### 9. Standard GS loss

The loss changed from:

```text
L1 + 0.2 * MSE
```

to configurable loss, defaulting to:

```text
0.8 * L1 + 0.2 * D-SSIM
```

This is closer to standard 3DGS. It also changes scalar magnitudes and local
texture incentives, so loss values before/after this change are not directly
comparable.

### 10. Small controlled init, then backtrack

After stability work, we tried a more controlled 8192-splat init:

```text
scale_init=0.02
scale_init_log_jitter=0.7
opacity_init=0.1
token_init_std=0.01
head_output_init_std=0.002
position_init_raw_jitter=0.9
rotation_init=identity
AdamW weight_decay=0.05
```

The intent was reasonable:

- avoid massive alpha saturation from 8192 splats
- avoid huge initial scales
- keep depth in a wider valid range
- use small TokenGS-style learned token/head init

But the later backtrack is important:

```text
position_init_raw_jitter was a final bias with shape [gaussians_per_token, 3]
```

For 64 splats/token, this created 64 sibling anchors reused across all 128
tokens. With tiny token/head noise, the initial 8192 splats were closer to 64
repeated templates than to 8192 meaningfully distinct primitives. AdamW weight
decay likely reinforced that clustering by pulling variation back toward shared
biases.

Latest patch changed this to:

```text
position_init_extent_coverage=0.9
token_init_std=0.3
head_output_init_std=0.06
rotation_init=random
optimizer=Adam, weight_decay=0
```

and initializes xyz bias in decoded space:

```text
xy_target ~ Uniform(-coverage, coverage)
raw_xy    = atanh(xy_target)

z_target  ~ Uniform(margin, 1-margin)
raw_z     = logit(z_target)
```

This makes the mean distribution nearly uniform in the bounded output space,
instead of relying on uniform raw logits and the nonlinearities' density.

## What actually broke

There were multiple different "breaks"; conflating them caused confusion.

### Break A: 128px NaNs

Status:
    fixed enough for training stability

Cause:
    New 128px/4fps DUSt3R camera trajectory did not fit the old fixed Gaussian
    z support. Late windows put many splats near/behind the camera. Projection
    math produced nonfinite values.

Fixes:
    finite-depth projection guard; configurable/wider z range.

Remaining risk:
    still no true scene/camera normalization. A future DUSt3R solve can define
    another arbitrary scale.

### Break B: gray/white renders

Status:
    partly fixed, but symptoms may recur for different reasons

Causes seen:
    - NaNs / invalid projection in raw 128px
    - late windows with no contributing splats
    - opacity/background behavior when most splats do not contribute
    - possible overly clustered init in later 8192 experiments

Fixes:
    renderer guard, depth range, Taichi renderer, more diagnostics, less
    clustered init.

Remaining risk:
    rendering a white/gray image can also be a local optimum if color/opacity
    remain too correlated or if the loss sees a large sky/background region.

### Break C: "more splats did not make it crisp"

Status:
    unresolved

Likely causes:
    - fixed-slot TokenGS output is not classic densified 3DGS
    - sibling splats per token are correlated
    - no split/clone/prune/opacity reset
    - small controlled init reduced diversity
    - sampled-window training loss is noisy
    - LR/optimizer/head-specific gradients may be imbalanced
    - 128px/4fps clip has more frames and harder camera path than the old tiny
      baseline

Fixes attempted:
    added eval metrics; changed loss; adjusted init away from clustered setup.

Not yet proven:
    whether current bounded-random init improves visual quality.

## Current state

The current active 128px local config is:

```text
src/train_configs/local_mac_overfit_prebaked_camera_128_4fps_wide_depth_taichi_8192splats.jsonc
```

Key properties:

```text
data:        small_128_4fps, 46 frames
model:       128 tokens x 64 splats/token = 8192 splats
render:      Taichi/Metal native batch
z range:     [0.5, 5.0]
scale init:  0.02 * exp(U[-0.7, 0.7])
opacity:     around 0.1
token init:  std 0.3
head init:   final output weight std 0.06
xyz init:    decoded-space bounded uniform coverage 0.9
rotation:    random
optimizer:   Adam, no weight decay
lr:          0.001 cosine to 0.0001
batch:       8 frames/step
loss:        0.8 L1 + 0.2 D-SSIM
metrics:     renderer/optimizer debug off by default, eval SSIM on video logs
```

Recent synthetic init stats after the bounded-random patch:

```text
xyz std by axis: [0.7975, 0.7979, 1.1628]
scale median:    0.0207
scale p05/p95:   0.0106 / 0.0386
opacity median:  0.0999
opacity p05/p95: 0.0937 / 0.1066
rgb median:      0.5000
rgb p05/p95:     0.4813 / 0.5188
rgb std:         0.0113
```

This does not prove training quality. It only proves we are no longer starting
from almost identical positions/colors/opacities.

## Backtracks and revised beliefs

### Backtrack 1: "z max is a far plane"

Old framing:
    Maybe larger `z_max` means far-plane clipping.

Revised model:
    `z_min/z_max` are output support for where the model may place Gaussian
    means in the current coordinate system. They are not renderer far planes.
    A far sky/background cannot be represented simply by setting z to infinity;
    distant background appearance is usually better represented by explicit
    background/sky modeling, view-dependent color, or normalized scene scale.

### Backtrack 2: "more splats means crisp"

Old framing:
    8192 splats at 128px should be enough because it is close to half the pixel
    count.

Revised model:
    8192 emitted splats from correlated token heads are not equivalent to 8192
    adaptively densified 3DGS primitives. Spatial allocation and independence
    matter as much as count.

### Backtrack 3: "small init is safer"

Old framing:
    Small TokenGS-style token/head init may stabilize 8192-splat training.

Revised model:
    Small init can stabilize, but in this architecture it can also collapse
    diversity because final-head biases are per sibling slot and shared across
    tokens. For local overfit, diversity is part of the capacity.

### Backtrack 4: "renderer speed and quality changes are separable"

Old framing:
    Taichi only changes speed.

Revised model:
    Taichi primarily changes speed, but renderer numerics, saturation, sorting,
    and background blending can also affect optimization. Any quality comparison
    across dense vs Taichi should be treated as renderer-plus-optimizer behavior,
    not pure speed.

## Current working theory

Confidence:
    medium

Current belief:
    The original tiny baseline worked because the data/camera scale was easy and
    the crude default init gave enough diversity. The 128px run first broke due
    to camera-scale/depth mismatch. After stabilizing that, quality plateaued
    because the fixed-token architecture lacks density control and because our
    "safer" init/optimizer choices reduced diversity.

What this predicts:

1. The latest bounded-random init should make the first few videos less gray and
   more varied than the small-init 8192 run.
2. If quality still plateaus, increasing `gaussians_per_token` alone will not
   help much; more independent tokens or better density allocation should help
   more.
3. If full-sequence eval improves but sampled training loss looks noisy, the
   sampled-window scalar was misleading.
4. If Eval/SSIM does not improve and visuals stay blurry, the issue is probably
   model parameterization/density allocation/loss weighting rather than renderer
   stability.

## Cheap falsification tests

### Test 1: old-simple init variant at 128px

Make a config that keeps current Taichi/wide-depth/data but restores old-style
init as much as possible:

```text
token_init_std=1.0
head_output_init_std=null
position_init_extent_coverage=0.0
rotation_init=random
opacity_init=null
scale_init=0.05
scale_init_log_jitter=0.0
optimizer=Adam
loss=l1_mse or standard_gs as a controlled variable
```

Interpretation:

- If this gets crisper faster, the controlled small init was indeed hurting.
- If it explodes or saturates, the old style does not survive 8192/128px without
  more careful opacity/scale handling.

### Test 2: current bounded init vs small init, fixed renderer/loss

Run two 300-500 step probes with:

```text
same data
same Taichi renderer
same 8192 splats
same loss
same LR/schedule
only init differs
```

Compare:

```text
Eval/L1
Eval/SSIM
visual dog detail
opacity histogram
scale histogram
projected-radius histogram
never-visible/contributing splat fraction
```

Interpretation:

- If bounded init wins, keep the backtrack.
- If small init wins, diversity was not the bottleneck and we should revisit LR,
  loss, or renderer differences.

### Test 3: token independence ablation

Compare equal splat count-ish variants:

```text
128 tokens x 64 splats/token
256 tokens x 32 splats/token
512 tokens x 16 splats/token
```

Interpretation:

- If more tokens win at similar splat count, sibling correlation is a bottleneck.
- If no variant wins, problem is elsewhere.

### Test 4: full-batch small clip sanity

Run the old 32px/2fps baseline after all code changes:

```bash
./src/train_scripts/train_full_dynamic_with_camera_prebake_all_frames.sh \
  src/train_configs/local_mac_overfit_prebaked_camera.jsonc
```

Interpretation:

- If it still converges, core code path did not destroy the original baseline.
- If it regresses, bisect loss/renderer/config normalization changes before
  blaming 128px model capacity.

## Practical next-step recommendation

For the next run, do not add another architecture knob yet. Run the current
bounded-random 8192 config and compare to the last small-init 8192 run using
full-sequence eval metrics and video, not only sampled `Loss`.

If it improves:

```text
keep Taichi + wide-depth + bounded-random init
then test token independence or head-specific LR
```

If it does not improve:

```text
run old-simple init variant
turn optimizer diagnostics on for 100-300 steps
inspect gradient/parameter histograms by xyz/scale/opacity/rgb heads
```

Avoid immediately adding learnable tanh slope. It might be useful, but it mixes
three issues:

```text
scene scale
activation saturation
initial distribution
```

The cleaner path is first to isolate whether the problem is initialization
diversity, token correlation, or optimizer/loss dynamics.
