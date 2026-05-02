# Training Ideas for Novel Synthesis

This note records the current camera-token swap training idea after the
train2/heldout1 multicam setup.

## Current Claim

The right task is not only:

```text
video_1 -> world tokens -> render video_1
```

It is:

```text
encode(video_a) -> world_a, camera_a
encode(video_b) -> world_b, camera_b

render(world_a, camera_a) -> video_a self reconstruction
render(world_b, camera_b) -> video_b self reconstruction
render(world_a, camera_b) -> video_b cross-view reconstruction
render(world_b, camera_a) -> video_a cross-view reconstruction
```

For calibrated multicam data, `video_a` and `video_b` are synchronized clips of
the same scene from different cameras. The cross-view losses are the important
ones: they force the video tokens/world tokens from one camera to be renderable
from the other camera token.

## Why This Makes Sense

Single-camera implicit-camera training has a gauge freedom: the camera and the
splat world can co-move, and the rendered source view can still match. That was
acceptable when the only target was source-view reconstruction.

Novel-view synthesis needs a query camera. At inference we want:

```text
encode(source_video) -> world_source, camera_source
query_camera_token = requested new view
render(world_source, query_camera_token) -> novel view
```

So training must prove that camera tokens are actually swappable. The paired
multicam objective gives direct supervision:

```text
world from camera 1 + camera token from camera 2 -> GT camera 2 frames
```

and symmetrically:

```text
world from camera 2 + camera token from camera 1 -> GT camera 1 frames
```

This is stronger than rendering both cameras from a fixed external rig, because
it tests whether the model's own camera-token representation can be used as a
query, not only whether the rasterizer accepts known extrinsics.

## Required Token Contract

The camera token cannot be an arbitrary high-capacity latent taken from the
target video. If it is, the target camera token can leak target appearance or
motion and the model can cheat.

Camera tokens should be one of:

1. An explicit relative camera parameterization:

```text
camera_token = {
  relative_SE3_to_anchor,
  intrinsics,
  optional small distortion params
}
```

2. A learned token with a hard camera decoder:

```text
camera_token -> SE3 + intrinsics
```

with low capacity, bounded deltas, and diagnostics showing it cannot carry
target RGB content.

3. A calibrated-pose token during paired-data finetuning, with optional learned
SE3 residuals:

```text
camera_token = calibrated_relative_pose + learned_delta
```

For inference, the camera token must be controllable. "Ask for a new camera
angle" should mean constructing or sampling a query camera token, not hoping an
opaque latent moves in a semantic direction.

## Training Loss

For a synchronized pair `(A, B)`:

```text
W_A, C_A = encoder(video_A)
W_B, C_B = encoder(video_B)

L_self =
    recon(render(W_A, C_A), video_A)
  + recon(render(W_B, C_B), video_B)

L_cross =
    recon(render(W_A, C_B), video_B)
  + recon(render(W_B, C_A), video_A)

L_world_consistency =
    distance(project_world(W_A), project_world(W_B))

L_camera =
    optional pose loss if calibrated poses are available
  + bounded camera residual regularization

L_total = L_self + lambda_cross * L_cross + lambda_world * L_world_consistency + L_camera
```

The cross-view terms should be the selector. A model that has low source-view
loss but poor `W_A + C_B -> video_B` has not learned novel-view synthesis.

## Pair Sampling

Training should not only use cross-camera swaps. Same-camera reconstruction is
still useful because it keeps the encoder/render path grounded and prevents the
model from seeing only off-diagonal query pairs.

For two train cameras, the canonical train items are:

```text
W_A + C_A -> GT_A    self
W_A + C_B -> GT_B    cross
W_B + C_B -> GT_B    self
W_B + C_A -> GT_A    cross
```

For more than two cameras, use all `source != target` cross pairs plus all self
pairs, or sample a shuffled subset per step with an explicit self-pair
probability. The utility surface for this contract starts in:

```text
src/train/camera_swap_sampling.py
```

## Heldout Evaluation

With train cameras `A` and `B` plus heldout camera `H`:

```text
W_A, C_A = encoder(video_A)
W_B, C_B = encoder(video_B)

train:
  render(W_A, C_A), render(W_A, C_B)
  render(W_B, C_B), render(W_B, C_A)

heldout:
  render(W_A, C_H) -> GT video_H
  render(W_B, C_H) -> GT video_H
```

`C_H` can come from calibrated heldout pose for evaluation. It should not be
learned from heldout RGB if the goal is a true novel-camera test.

## Cheap Falsification Tests

1. **Wrong-camera swap:** render `W_A` with a deliberately wrong camera token.
   The output should move to the wrong view and the loss should worsen.
2. **Wrong-world swap:** render `W_A` with `C_B` against an unrelated scene's
   `video_B`. Loss should worsen sharply.
3. **Camera-token bottleneck probe:** train a small decoder from camera token to
   RGB frame. It should fail; if it succeeds, the camera token is carrying
   target appearance.
4. **Heldout selector:** choose checkpoints by heldout `W_A + C_H` PSNR/SSIM,
   not source-view reconstruction.

## Immediate Implementation Implication

The current multicam trainer already supports:

```text
condition video -> world tokens
external train camera rig -> losses on two train cameras
external heldout camera -> eval
```

The next model change for implicit camera learning is to make camera tokens
first-class query tokens:

```text
encoder(video) -> world_tokens, estimated_camera_token
renderer(world_tokens, query_camera_token) -> RGB/video
```

Then train with the symmetric camera-token swap objective above.

## TODO: Single-Image Or Video Conditioning

It would be useful for the same world-token / query-camera renderer to accept
either a single image or a video clip as conditioning input. This would expand
the usable training data beyond synchronized videos while keeping the novel-view
contract the same:

```text
encode(image_or_video) -> world_tokens, source_camera_token
render(world_tokens, query_camera_token, query_time?) -> RGB/video
```

This is an architecture task, not just a sampler flag. The encoder needs a
shared output contract across `T=1` images and `T>1` videos, and the decoder
needs a clean time interface so static image conditioning can still answer
single-frame novel-view queries without pretending it observed motion.

Implementation questions to resolve before coding:

1. Should image conditioning be represented as a length-1 video, or should the
   encoder have explicit image/video branches that merge into the same
   world-token space?
2. How should dynamic tokens behave when the source is a single image: disabled,
   regularized toward static, or predicted with uncertainty?
3. How do camera-token swap losses mix across image-only, video-only, and
   synchronized multicam-video batches without letting the easiest source-view
   objective dominate?
4. What is the inference API for `query_time` when the conditioning source is a
   single image?
