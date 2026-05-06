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

The first implementation step is now wired as an oracle baseline. The trainer
can load both train-camera videos, decode a source camera into `W_source`, and
render self/cross targets with calibrated source-relative cameras:

```text
train.camera_swap_mode = "oracle_relative"
```

The config entrypoint is:

```text
src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_static_dynamic_96_32_precomputed_vjepa2_1_vitb_384_128_16f_8192splats_oracle_relative_camera.jsonc
```

The next implementation step is also available behind:

```text
train.camera_swap_mode = "learned_residual"
```

That mode adds a tiny cross-attention relative-pose head over projected
source/target V-JEPA memories, predicts a bounded SE(3) residual around the
calibrated delta, and keeps the world decoder source-only. Heldout validation
still uses calibrated/query deltas rather than heldout RGB features.
The trainer logs residual identity and cycle losses for the learned relpose
path.

First 250-step learned-residual baseline:

```text
wandb/offline-run-20260503_014531-woco72am
TrainView0 PSNR/SSIM = 14.9334 / 0.1966
TrainView1 PSNR/SSIM = 15.5946 / 0.2731
Heldout camera_0040 PSNR/SSIM = 14.0453 / 0.1714
```

This is the best measured Tier-2a heldout row as of 2026-05-03, narrowly ahead
of the 80-step free dynamic 3DGS row (`13.2940` heldout PSNR). Treat it as a
baseline candidate rather than a settled result: it still needs leakage probes,
seed checks, and a longer run.

The intended API shape remains:

```text
encoder(source) -> W_source
relative_camera_head(source, target or requested_delta) -> Delta_query
renderer(W_source, Delta_query, query_time?) -> target
```

For the learned version, keep the leak guard: target/reference features may
condition only the relative-pose head, never `W_source`.
