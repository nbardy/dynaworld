# Mixed Multiview and YouTube Training Plan

Date: 2026-04-28

## Target

Migrate from single-camera overfit to a mixed training setup:

```text
10 clips with 3 synchronized camera views each
20 YouTube single-camera train clips
optional 5 YouTube single-camera validation clips
```

For each three-camera clip, train on two cameras and evaluate on the third:

```text
train: camera_a, camera_b
camera_heldout validation: camera_c
```

This creates a direct test of whether the learned world tokens and decoded
splats are view-consistent.

## Sample Schema

Use an explicit sample manifest. Do not infer benchmark membership from folders
or filenames.

Suggested record:

```json
{
  "sample_id": "deepview_scene_0007",
  "modality": "multicam",
  "split": "train",
  "frame_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
  "condition_camera_id": "camera_0001",
  "train_camera_ids": ["camera_0001", "camera_0015"],
  "heldout_camera_ids": ["camera_0013"],
  "camera_metadata": "path/to/cameras.json",
  "feature_cache": {
    "backend": "vjepa2_vitb_384",
    "keys": {
      "camera_0001": "sample/camera_0001/window_0000.pt",
      "camera_0015": "sample/camera_0015/window_0000.pt",
      "camera_0013": "sample/camera_0013/window_0000.pt"
    }
  }
}
```

Single-camera YouTube records should use the same structure with an empty
`heldout_camera_ids` list:

```json
{
  "sample_id": "youtube_clip_0012",
  "modality": "singlecam",
  "split": "train",
  "condition_camera_id": "camera_0000",
  "train_camera_ids": ["camera_0000"],
  "heldout_camera_ids": []
}
```

## Training Objective

For multicam samples:

```text
features = encode(video[condition_camera])
world_tokens = model(features)
splats_t = decode(world_tokens, t)

loss_train =
  render_loss(splats_t, camera_a, image_a)
  + render_loss(splats_t, camera_b, image_b)
  + regularizers

validation_only =
  render_loss(splats_t, camera_c, image_c)
```

For YouTube single-camera samples:

```text
features = encode(video[source_camera])
world_tokens = model(features)
splats_t = decode(world_tokens, t)

loss_train =
  render_loss(splats_t, source_camera, image_source)
  + temporal_regularizers
  + representation_regularizers
```

The heldout camera should initially be validation-only. Once the benchmark is
stable, add cross-validation folds that rotate which camera is held out.

## Mixed Batch Sampler

Start with balanced modality sampling:

```python
if rand() < 0.5:
    batch = sample_multicam_clip_uniformly()
else:
    batch = sample_youtube_clip_uniformly()
```

Within a multicam batch:

```python
sample_id = uniform(multicam_train_samples)
condition_camera = uniform(train_camera_ids)
loss_cameras = train_camera_ids
eval_cameras = heldout_camera_ids
```

Do not let the 20 YouTube clips dominate simply because there are more of them.
The multicam clips provide the 3D pressure; the YouTube clips provide broader
video-conditioning pressure.

## Loss Balancing

Initial loss:

```text
L =
  L_rgb_train
  + lambda_ssim * L_ssim_train
  + lambda_temporal * L_temporal
  + lambda_camera * L_camera_regularization
  + lambda_splat * L_splat_regularization
```

Validation reports:

```text
camera_heldout/L1
camera_heldout/SSIM
camera_heldout/PSNR
sample_heldout/source_L1
sample_heldout/source_SSIM
sample_heldout/source_PSNR
camera_adjustment_norm
decoded_xyz_motion
```

Do not put heldout-camera loss into training until the validation protocol is
stable. The heldout camera is the measurement instrument.

## Phased Plan

### Phase 1: One Multicam Sample Smoke

Use the existing train-two / heldout-one path.

Recommended heldout choices:

- overlap heldout first, because it tests interpolation instead of extreme extrapolation
- near-outside heldout second, because it tests harder view generalization

Run W&B-disabled local smokes first if online logging stalls.

### Phase 2: Ten Multicam Samples

Build a manifest with 10 three-camera samples.

For every sample, record:

- camera IDs
- camera angles or relative pose bins
- train cameras
- heldout camera
- frame window
- feature cache keys

Run the same model cells on all 10 samples before scaling capacity.

### Phase 3: Sample-Heldout Multicam

Hold out complete multicam samples.

This prevents the model from winning by memorizing scene-specific priors across
camera views. Report both source-camera and heldout-camera metrics for these
samples when possible.

### Phase 4: Add YouTube Single-Camera Mix

Add 20 single-camera YouTube clips with balanced sampling.

Expected effect:

- better robustness of video conditioning
- broader appearance and motion priors
- possible drop in camera-heldout quality if YouTube is overweighted

If camera-heldout quality drops, reduce YouTube sampling probability before
changing the architecture.

### Phase 5: Scale Capacity

Only after phases 1-4 are stable, run the capacity ladder:

```text
tokens: 128 -> 192 -> 256
splats: 8192 -> 16384 -> 32768
cross-attn layers: 4 -> 6 -> 8
MLP hidden: 64 -> 128
```

Change one dimension at a time. Keep the benchmark split frozen.

## Expected Signals

If the system is learning the intended representation:

- train-camera PSNR improves first
- overlap camera-heldout improves next
- outside camera-heldout improves more slowly
- sample-heldout source reconstruction improves when the feature prior helps
- YouTube mix improves sample robustness but may not immediately improve multicam heldout

If the system is not learning the intended representation:

- train-camera PSNR improves while heldout-camera metrics stay flat
- learned camera adjustments grow with quality
- dynamic splats absorb static geometry
- YouTube mix improves source reconstruction but weakens multiview consistency
- heldout-camera error is concentrated in foreground/motion regions

## Immediate Next Implementation Direction

Create a checked-in benchmark manifest and config family:

```text
multicam_10sample_train2_holdout1_128_16f_static_dynamic_vjepa.jsonc
multicam_10sample_train2_holdout1_youtube20_128_16f_static_dynamic_vjepa.jsonc
multicam_10sample_train2_holdout1_youtube20_256_16f_static_dynamic_vjepa.jsonc
```

Start with:

```text
128px
16 frames
8192 splats
96 static / 32 dynamic tokens
V-JEPA 2.1 ViT-B/384 cached features
strong RGB/uniform/token/head init
camera-clamped primary comparison
```

The first model comparison should be:

```text
unconditioned static/dynamic strong-init
local static/dynamic strong-init
V-JEPA static/dynamic strong-init
V-JEPA no-split strong-init
```

Only then spend compute on larger capacity.
