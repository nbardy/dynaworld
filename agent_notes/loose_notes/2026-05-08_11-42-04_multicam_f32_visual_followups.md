# Multicam F32 Visual Followups: Edge Blur And Relative-Pose Strength

## Context

After committing the alpha-threshold A/B (`ecb5588`), visual inspection raised
two concerns:

- all camera rows look center-sharp but edge/ring blurry, especially around the
  outer quarter of each frame
- cameras may be initialized at the same start position, and the model may not
  be learning a meaningful relative-camera offset

This note records the current code/config read and the measured offset strength
for the promoted `1/128` F32 checkpoint.

## Current Camera And Projection Contract

Promoted config:

```text
src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_vjepa_full_relpose_features_F32_256_16f_8192splats_goodset_train0006_0014_holdout0005_alphaab_alpha1_128.jsonc
```

Key settings:

- train: `camera_0006`, `camera_0014`
- heldout: `camera_0005`
- anchor/source for eval render: `camera_0006`
- `render.camera_projection = "legacy_pinhole"`
- `camera.lens_model = "pinhole"`
- `camera.rig_learn_global_se3 = false`
- `camera.rig_learn_per_camera_se3 = false`
- `train.heldout_eval_camera_mode = "predicted_relpose"`
- `train.relpose_feature_frame_mode = "first_frame"`
- `train.relpose_max_rotation_degrees = 45.0`
- `train.relpose_max_translation_ratio = 1.0`

The loader did recover DeepView fisheye metadata:

```text
pose_source deepview_models_relative_opencv_fisheye
train_lens_models ['opencv_fisheye', 'opencv_fisheye']
heldout_lens_models ['opencv_fisheye']
```

But the active render path still behaves as pinhole for this trainer:

- `LearnableCameraRig._make_camera(...)` constructs `CameraSpec(lens_model="pinhole")`.
- `source_relative_cameras_from_K_w2c(...)` supports `target_lens_model` and
  `target_distortion`, but `source_relative_cameras_for_pair(...)` calls it
  without passing the bundle's DeepView lens metadata.
- The promoted config explicitly sets `render.camera_projection="legacy_pinhole"`.

So the edge/ring blur is very plausibly a projection-model mismatch: DeepView
frames carry fisheye distortion, but this multicam F32 relpose path renders
source-relative cameras as pinhole. A config flip alone may not be enough,
because the `CameraSpec` objects given to the renderer are already pinhole.

## Relative-Pose Head Behavior

The no-VGGT full relpose trainer does learn a camera transform, but not through
the `camera_rig` offsets in this config:

- the frozen `camera_rig` metrics are exactly zero
- the learned component is `RelativePoseCrossAttentionHead`
- `RelativePoseCrossAttentionHead.output[-1]` is zero-initialized, so before
  training every query predicts identity: same position/orientation as the
  source camera
- heldout validation uses the predicted relpose head when
  `heldout_eval_camera_mode="predicted_relpose"`

This means "all cameras initialize at the same start position" is true for the
full relpose head by construction. It is a conservative zero-bias start, not a
camera-rig collapse.

## Offset-Strength Measurement

Measured from:

```text
outputs/multicam_relative_pose/full_relpose_features_F32_256_alphaab_alpha1_128_goodset_train0006_0014_holdout0005/checkpoint_final.pt
```

Diagnostic command was a disabled-W&B trainer load with the checkpoint injected
as `train.checkpoint_load_path`, then `full_relative_pose_for_pair(...)` for
source `camera_0006` to self, train target `camera_0014`, and heldout
`camera_0005`.

Results:

| Pair | Predicted offset | Calibrated/oracle offset | Error |
| --- | ---: | ---: | ---: |
| `camera_0006 -> camera_0006` | `1.443 deg`, `0.1687` translation | `0.000 deg`, `0.0000` | `1.443 deg`, `0.1687` |
| `camera_0006 -> camera_0014` | `1.589 deg`, `0.1292` translation | `33.186 deg`, `0.3094` | `32.307 deg`, `0.4136` |
| `camera_0006 -> camera_0005` | `0.622 deg`, `0.0311` translation | `18.631 deg`, `0.1851` | `18.479 deg`, `0.1849` |

The calibrated/oracle source-relative transforms are static across the 16-frame
clip (`oracle_frame_span_rot=0`, `oracle_frame_span_t=0`), so this is not a
temporal-pose artifact.

Read: the promoted `1/128` run did not learn the intended full camera offset.
Its heldout PSNR is real for the logged render, but we should not interpret it
as evidence that the relpose head recovered the physical heldout camera. The
model is mostly rendering from near-source camera pose and getting useful
numbers because this is a close-overlap trio plus learned splats/features.

## Followups

1. Add a relpose diagnostic payload to validation:
   - per target/source pair predicted rotation degrees
   - predicted translation norm
   - oracle rotation/translation norm
   - geodesic rotation error
   - translation error
   This should be logged next to heldout PSNR so a good heldout metric cannot
   hide an identity-pose solution.

2. Run a fisheye-preserving smoke before changing the stable trainer:
   - pass `train_lens_models` / `train_distortions` and
     `heldout_lens_models` / `heldout_distortions` into the `CameraSpec`
     constructors used by `LearnableCameraRig` and
     `source_relative_cameras_for_pair`
   - set `render.camera_projection="auto"` or `"camera_model"`
   - run a 1-step F32 smoke and a short 250-step A/B only after the camera specs
     actually arrive at the renderer as `opencv_fisheye`

3. Revisit relpose supervision:
   - current matrix L2 pose loss is not forcing the head to leave identity
   - use explicit geodesic rotation + scaled translation terms, or pretrain the
     relpose head on train-pair calibrated transforms before joint splat fitting
   - report heldout quality together with pose error, not alone

4. Consider initializing train-pair relpose from calibrated offsets or adding a
   stronger supervised warmup. The current zero-init is stable but may be too
   sticky for full-pose prediction.
