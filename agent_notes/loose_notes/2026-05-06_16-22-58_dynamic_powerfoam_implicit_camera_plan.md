# Dynamic PowerFoam Implicit Camera Plan

## Trigger

Research Engineer 4 ownership for this pass is docs/notes/config only. The goal
is to prepare the next dynamic PowerFoam run artifact without touching Python
trainer/model code, because other workers currently own the implementation
surface.

## Baseline To Preserve

The current matched fixed-ray full-clip baseline is:

```text
src/train_configs/local_mac_token_dynamic_powerfoam_features_F32_1024_8sites_youtube_hlaZbH_center_crop_8fps_512_56f_120step.jsonc
```

It trains all dynamic F32 feature-foam channels on the 7s/56f center-cropped
high-motion YouTube clip. The run produced valid MP4s and a nice-looking moving
render relative to earlier coarse/frozen controls, but it is the wrong
factorization baseline for camera-motion clips: fixed rays force camera motion,
object motion, and representation motion into the same world-state dynamics.
That can look plausible while learning the wrong explanation.

Use it as a visual/optimization baseline, not as evidence that the model has
separated camera motion from scene dynamics.

## Config Variant Prepared

New tentative variant:

```text
src/train_configs/local_mac_token_dynamic_powerfoam_features_F32_1024_8sites_youtube_hlaZbH_center_crop_8fps_512_56f_120step_implicit_camera.jsonc
```

This keeps the same:

- video path: `data/youtube_curated_spans/high_motion_smokes/hlaZbH_OFBU_seg_003_center_crop_8fps_full.mp4`
- 56 frames at 512px render size
- 1024 cells, 8 texel sites, F32 features
- `token_rbf_features`
- all dynamic switches enabled: centers, radii, densities, features, normals,
  texel sites
- 120 steps and the existing PowerFoam/colorize LR shape

The only intended new axis is learned implicit camera support. Because the
dynamic PowerFoam schema may not be landed yet, the config uses explicit
tentative keys in two places:

- `model.camera_conditioning`
- top-level `camera`

Both are marked with:

```text
schema_status: tentative_until_dynamic_powerfoam_camera_schema_lands
```

Main implementation agent should either wire these keys directly or rename them
during schema normalization. The intent is clear: a video-conditioned learned
camera head should predict per-frame pose deltas, feed full rays into
PowerFoam, and keep small motion/global/temporal priors so the model cannot
explain everything as unconstrained camera drift.

## First Acceptance Read

Do not rank this only by train loss. The comparison should ask whether learned
camera motion reduces pressure on dynamic geometry/material and improves the
factorization.

Minimum logged comparison against the fixed-ray full-clip baseline:

- render and side-by-side MP4 validity
- mean/min PSNR and L1/MSE
- camera rotation/translation/FOV deltas over time
- PowerFoam center/radius/density/feature/normal/texel-site deltas
- temporal screen motion from foam centers
- whether visible camera-motion frames stop requiring large world-state drift

If the learned-camera run improves visual plausibility but simply moves all
motion into camera deltas, it is still not accepted. If it keeps camera motion
bounded while reducing unnecessary foam deformation on camera-motion portions
of the clip, it becomes the next baseline to test on a held-out or synthetic
camera-motion control.
