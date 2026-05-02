# Multiple Camera, Pre-Training Notes

This note records the current mixed image/video and multi-camera pretraining
concern. It is a follow-up architecture topic, not the current speed task.

## Problem

The naive camera-token swap story is incomplete:

```text
encode(video_A) -> W_A, C_A
encode(video_B) -> W_B, C_B
render(W_A, C_B) -> video_B
```

The issue is frame of reference. If `video_A` and `video_B` are encoded
independently, each encoder pass can choose its own gauge. There is no guarantee
that `C_B` is expressed in the same coordinate frame as `W_A`, so `W_A + C_B`
is not necessarily a meaningful render query.

## Better Contract

Use the source observation as the local frame and predict a relative camera
change into the target observation:

```text
encode(source_image_or_video) -> W_source
relpose(source_image_or_video, target_image_or_video) -> Delta_source_to_target
render(W_source, Delta_source_to_target, query_time?) -> target_rgb_or_video
```

For calibrated multicam data:

```text
Delta_GT = inverse(camera_source) @ camera_target
L_pose = pose_loss(Delta_pred, Delta_GT)
L_render = recon(render(W_source, Delta_pred), target)
```

For uncalibrated paired data, the relative-pose head is weaker but still useful:

```text
Delta_12 = relpose(obs_1, obs_2)
Delta_21 = relpose(obs_2, obs_1)

render(W_1, Delta_12) -> obs_2
render(W_2, Delta_21) -> obs_1
cycle: Delta_12 * Delta_21 ~= identity
```

## Mixed Image/Video Training

The useful long-term shape is one renderer/API that can condition on either a
single image or a video:

```text
encode(image_or_video) -> world_tokens
query = explicit camera delta, calibrated relative pose, or predicted relpose
render(world_tokens, query, query_time?) -> image_or_video target
```

Single images expand the training data and can train static novel-view behavior.
Videos add temporal and dynamic-token pressure. Both should land in the same
source-anchored world-token contract.

## Architecture Questions

1. Should single images be represented as length-1 videos, or should image/video
   encoders be separate branches that merge into a shared world-token space?
2. What happens to dynamic tokens when the source is a single image: disable
   them, regularize them toward static, or predict uncertainty?
3. Is the relative-pose head pairwise-only, or should it also consume global
   dataset camera metadata when calibrated poses exist?
4. How do we balance self reconstruction, cross-rendering, pose supervision, and
   cycle constraints so the easiest source-view loss does not dominate?
5. What is the inference API for `query_time` when conditioning on a single
   image?

## Implementation Follow-Up

Do not wire this into the current speed baseline yet. The next implementation
step is a design pass for a source-anchored world-token API and relative-camera
head:

```text
encoder(source) -> W_source
relative_camera_head(source, target or requested_delta) -> Delta_query
renderer(W_source, Delta_query, query_time?) -> target
```

Only after that contract is clear should the camera-swap sampler be connected to
the trainer.
