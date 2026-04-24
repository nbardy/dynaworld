# Video-token implicit bad baseline diagnosis

## Context

After fixing the time-conditioning path, the W&B video still looked much worse
than the known-camera and earlier implicit-camera overfit baselines. The render
moved over time, but it was blurry/washed out and did not reconstruct the dog
well.

## What was different from the working baselines

### Gaussian init

The video-token implicit model was still using the old zero-init Gaussian head:

- all 8192 splats at exactly `xyz=0`
- scale exactly `0.05`
- opacity exactly `0.5`
- RGB exactly `0.5`
- query token std about `0.02`

The working 8192-splat known-camera baseline used the later bounded-random
Gaussian init:

- z spread roughly `0.7..4.8`
- scale roughly `0.009..0.046`
- opacity around `0.1`
- token std about `0.3`
- random per-split position/scale/rotation variation

That was a real apples-to-oranges config/code mismatch. Starting 8192 identical
opaque-ish gray splats at the origin is a bad 128px baseline.

### Camera range

The implicit camera does start as a look-at-origin orbit camera:

- base camera position is `(0, 0, -radius)`
- zero-init global camera head gives `FOV=60 deg`, `radius=3`
- zero-init path head gives no per-frame rotation/translation delta

The DUSt3R known-camera trajectory for the 128px/4fps clip is much larger than
the implicit camera config allowed:

- DUSt3R camera distance from first frame: max about `2.59`
- DUSt3R forward-vector angle from the initial +z view: max about `49.45 deg`
- implicit config allowed only `max_rotation_degrees=5` per axis and
  `max_translation_ratio=0.2`, i.e. `0.6` world units per translation axis

So the implicit camera was not even in the same motion range as the camera-input
baseline.

### Baseline contract

The earlier "decent implicit camera" run was the image-encoder implicit-camera
baseline at 32px. That model gets per-frame image features directly. The new
run is a video-token implicit model at 128px that encodes a 16-frame clip into
tubelet tokens and decodes requested times from that shared clip memory. It is
not a direct continuation of the 32px per-frame implicit baseline.

## Changes made

- Replaced the video-token implicit `TimeConditionedGaussianHeads` zero-init
  path with the shared `GaussianParameterHeads` used by the known-camera model.
- Added model config normalization for the Gaussian init knobs.
- Updated the 128px/4fps video-token implicit config to match the 8192-splat
  known-camera init:
  - `tokens=128`
  - `gaussians_per_token=64`
  - `xy_extent=1.5`
  - `z_min=0.5`, `z_max=5.0`
  - `scale_init=0.02`
  - `scale_init_log_jitter=0.7`
  - `opacity_init=0.1`
  - `query_token_init_std=0.3`
  - `head_output_init_std=0.06`
  - `position_init_extent_coverage=0.9`
  - `rotation_init=random`

After the patch, the video-token implicit init numerically matched the
known-camera 8192 init closely:

- video implicit: token std `0.299`, z `0.71..4.74`, opacity mean `0.100`
- known camera: token std `0.301`, z `0.72..4.76`, opacity mean `0.100`

## Probes

### Bounded init, original camera range

- W&B `etcghanx`: https://wandb.ai/nbardy/dynaworld/runs/etcghanx
- 200 steps, fixed init, original camera range.
- Full eval:
  - `Eval/Loss 0.14872`
  - `Eval/L1 0.09697`
  - `Eval/SSIM 0.28856`
  - `Eval/PSNR 16.64`
  - `TemporalAdjacentL1Ratio 0.02318`
  - `TemporalToFirstL1Ratio 0.17632`
  - camera eval: FOV `58.51`, radius `2.84`, rot `2.46 deg`, trans `0.465`

The init fix made the first sampled loss drop quickly (`~0.42` to `~0.18` in
the first ~16 steps), but full-video eval stayed in the same rough band as the
bad run.

### Bounded init, wide camera range

- W&B `op7l2ypp`: https://wandb.ai/nbardy/dynaworld/runs/op7l2ypp
- 200 steps, fixed init, `max_rotation_degrees=60`,
  `max_translation_ratio=1.0`.
- Full eval:
  - `Eval/Loss 0.15014`
  - `Eval/L1 0.10032`
  - `Eval/SSIM 0.30114`
  - `Eval/PSNR 16.60`
  - `TemporalAdjacentL1Ratio 0.03159`
  - `TemporalToFirstL1Ratio 0.40228`
  - camera eval: FOV `58.73`, radius `2.87`, rot `3.87 deg`, trans `1.206`

Widening the camera range let the model use more camera translation and improved
accumulated temporal motion, but it did not improve reconstruction.

## Current interpretation

The zero-init Gaussian head was a real mistake and is patched. The camera range
was also too small relative to the DUSt3R trajectory, but simply widening it is
not enough.

The remaining gap appears architectural/training-contract level: the
known-camera baseline gets explicit cameras and Plucker conditioning, while the
video-token implicit baseline must infer both camera and geometry from
photometric reconstruction. The earlier 32px image-implicit baseline is also not
the same task because it gets per-frame image features directly. The next fair
baseline should either be image-encoder implicit at 128px with the same init, or
a video-token model with a stronger camera/temporal contract.
