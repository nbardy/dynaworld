# PowerFoam Orbit180 Static-Dynamic Static-Warm Run

## Context

We wanted to test the more physical camera hypothesis: instead of fixed pinhole rays or a tiny residual camera, initialize the foam in world/object space under a 180 degree orbit camera path, keep most cells static, allow only a small dynamic cell bank, and prevent early repaint so the model has pressure to explain the clip through support/camera/geometry.

This followed the question: can PowerFoam handle a static/dynamic split like TokenGS, with mostly static scene cells and a smaller dynamic bank, while learning camera motion instead of just moving/repainting points under fixed rays?

## Code Changes

Implemented:

- `PowerFoamImplicitCameraDecoder.base_path_mode="orbit_yaw"` with a per-frame orbit base camera path.
- `base_camera_to_world_matrices(...)` so metrics/init can compare against a per-frame base instead of one static base matrix.
- `initialize_full_powerfoam_from_orbit_video(...)` to sample pixels across frames, backproject through the orbit base camera, and initialize a repeated world-space foam support.
- Static/dynamic cell masking in `TokenDynamicPowerFoamFeatures`: 896 static cells, 128 dynamic cells for the high-res run.
- Runtime stage controls:
  - `static_only_steps=60`: temporal geometry disabled through step 60.
  - `no_repaint_steps=90`: temporal feature repaint disabled through step 90.
- Orbit-source camera-facing normal initialization. The first implementation kept all normals in fixed-camera `[0,0,-1]`; this was wrong for a 180 degree orbit. The corrected code initializes each cell normal from its source camera and then repeats it across frames.
- Camera-aware temporal screen-motion metrics. The old metric projected world points as if they were already fixed-camera points and produced impossible values such as 278k px. The corrected metric transforms through the active camera path and reports a valid-depth fraction.

## Config

```text
src/train_configs/local_mac_token_dynamic_powerfoam_features_F32_1024_128dyn_orbit180_staticwarm_youtube_hlaZbH_center_crop_8fps_512_56f_180step.jsonc
```

Key settings:

```text
video: data/youtube_curated_spans/high_motion_smokes/hlaZbH_OFBU_seg_003_center_crop_8fps_full.mp4
frames: 56
fps: 8
render_size: 512
cells: 1024
feature_dim: 32
num_texel_sites: 8
camera.base_path_mode: orbit_yaw
camera.orbit_yaw_start_degrees: 0
camera.orbit_yaw_end_degrees: 180
model.video_init_mode: orbit_camera
model.static_dynamic_split: true
model.dynamic_cells: 128
train.static_only_steps: 60
train.no_repaint_steps: 90
```

Output:

```text
outputs/dynamic_powerfoam_metal/local_mac_token_dynamic_powerfoam_features_F32_1024_128dyn_orbit180_staticwarm_youtube_hlaZbH_center_crop_8fps_512_56f_180step
```

W&B:

```text
wandb/run-20260506_175522-jkrbsjj6
run id: jkrbsjj6
```

## Commands

Full run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train WANDB_SILENT=true .venv/bin/python \
  src/train/train_dynamic_powerfoam_metal.py \
  src/train_configs/local_mac_token_dynamic_powerfoam_features_F32_1024_128dyn_orbit180_staticwarm_youtube_hlaZbH_center_crop_8fps_512_56f_180step.jsonc
```

Fast smoke after fixes:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train WANDB_MODE=disabled .venv/bin/python \
  src/train/train_dynamic_powerfoam_metal.py \
  /tmp/dynaworld_powerfoam_orbit_static_dynamic_smoke.jsonc
```

The smoke reduced to 3 frames, 64px, 64 cells, 8 dynamic cells, 1 step. It completed on MPS and exercised orbit init, static/dynamic masking, stage controls, eval logging, and MP4 artifact writing.

## Metrics

Corrected full run:

```text
step 0:
  mean/min PSNR 8.0975 / 6.8215
  L1/MSE          0.31020 / 0.15679
  alpha mean      0.96738
  screen motion   5.27px mean / 14.00px p95, valid fraction ~1.0

step 60, best:
  mean/min PSNR 11.0654 / 6.8484
  L1/MSE          0.23630 / 0.08626
  alpha mean      0.95537
  screen motion   17.88px mean / 55.09px p95, valid fraction ~1.0
  feature temporal delta 0.0 because repaint was still disabled
  camera residual motion 18.53deg rotation / 0.368 translation

step 120:
  mean/min PSNR 10.2515 / 6.6380
  L1/MSE          0.24735 / 0.09981
  alpha mean      0.90897
  screen motion   25.04px mean / 70.25px p95
  feature temporal delta 0.00197

step 180, final:
  mean/min PSNR 10.3100 / 6.6002
  L1/MSE          0.26030 / 0.10353
  alpha mean      0.99576
  screen motion   28.34px mean / 82.41px p95, valid fraction ~1.0
  feature temporal delta 0.00312
  camera residual motion 22.15deg rotation / 0.447 translation
```

Comparison against earlier same-clip rows:

```text
fixed-camera all-enabled baseline:    mean PSNR 18.0796, L1 0.08914
residual-camera branch:               mean PSNR 18.0871, L1 0.08928
object-centric learned-camera run:    mean PSNR 11.2901, L1 0.22518
orbit/static-dynamic corrected best:  mean PSNR 11.0654, L1 0.23630
orbit/static-dynamic corrected final: mean PSNR 10.3100, L1 0.26030
```

## Media Sanity

Final MP4s:

```text
outputs/dynamic_powerfoam_metal/local_mac_token_dynamic_powerfoam_features_F32_1024_128dyn_orbit180_staticwarm_youtube_hlaZbH_center_crop_8fps_512_56f_180step/render_step_0180.mp4
outputs/dynamic_powerfoam_metal/local_mac_token_dynamic_powerfoam_features_F32_1024_128dyn_orbit180_staticwarm_youtube_hlaZbH_center_crop_8fps_512_56f_180step/side_by_side_step_0180.mp4
```

ffprobe:

```text
render: 512x512, 56 frames, 8 fps, 7.0s
side-by-side: 1024x512, 56 frames, 8 fps, 7.0s
```

Pixel sanity:

```text
render first frame: std [2.24, 1.09, 1.26], sample unique 28
render mid frame:   std [28.28, 27.39, 50.49], sample unique 3221
render last frame:  std [32.20, 38.81, 45.46], sample unique 2147

side-by-side first frame: std [58.98, 61.47, 60.51], sample unique 2379
side-by-side mid frame:   std [61.83, 61.92, 71.98], sample unique 4711
side-by-side last frame:  std [48.94, 49.52, 53.61], sample unique 4134
```

The MP4s are not the old all-green writer/player failure. The render-only first frame is nearly flat/white, which is a model/artifact-quality issue, not a codec-green issue.

## Interpretation

Observed fact:

- The orbit/static-dynamic path runs end to end on MPS, logs W&B media, writes MP4s, and produces non-green videos.
- It learns camera residuals and activates temporal geometry after the warmup.
- The static/no-repaint stage does what it says: feature temporal delta is exactly 0.0 at the step-60 eval.

Current belief:

- The more physical orbit/world-space initialization did not rescue the learned-camera support problem. It starts much lower than the fixed/residual image-plane gauge, improves to only ~11 dB, and regresses after dynamic geometry/repaint are unlocked.
- The residual-camera branch around the strong fixed-pinhole/image-plane support remains the usable high-res branch for this clip.
- The orbit path may be too hard because initial support is clamped into a small world box after backprojection and each cell stores one static world orientation/source sample even though the video is not a calibrated 180 degree object scan.

Branches:

Hypothesis A:
    The orbit initialization is geometrically mismatched to the real video camera motion.
Why:
    The YouTube clip is high-motion but not a known calibrated 180 degree orbit around a static object. We imposed a 180 degree orbit, then asked the residual path to repair it.
Cheap test:
    Sweep orbit span 15/30/60/90/180 degrees and compare step-0 and step-60 metrics.

Hypothesis B:
    The world-space support box is too aggressive.
Why:
    Backprojected points are clamped to `xy_extent=1.6`, `z=-1.6..1.6`; this can collapse distant/background structure into a compact object shell.
Cheap test:
    Keep orbit init but sweep `xy_extent` and `z_min/z_max`, measuring alpha/support and step-0 PSNR.

Hypothesis C:
    Static/dynamic split is not the problem; orbit support is.
Why:
    Static/dynamic masking and staged feature freeze function mechanically, but quality is poor before the dynamic bank has much chance to matter.
Cheap test:
    Copy `static_dynamic_split=true`, `dynamic_cells=128`, `static_only_steps=60`, and `no_repaint_steps=90` onto the residual-camera/image-plane config.

Hypothesis D:
    The normal fix made the run more physically correct but not better.
Why:
    Correcting normals removed a real convention bug, but the corrected run underperformed the pre-fix diagnostic. That suggests the issue is not only local surface orientation.
Cheap test:
    Compare fixed normals vs source-camera normals on the 3-frame smoke and a 40-step 512 run, but do not treat the pre-fix path as physically valid.

## Decision

Do not route next high-res work through full 180 degree orbit/world-space init unless we first have a calibrated or synthetic orbit clip. The next practical branch should be:

```text
residual camera + image-plane init + static_dynamic_split + static/no-repaint warmup
```

That tests the TokenGS-like capacity split without throwing away the known-good fixed-pinhole/image-plane support.

## Verification

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/powerfoam_implicit_camera.py \
  src/train/train_dynamic_powerfoam_metal.py \
  tests/test_powerfoam_implicit_camera.py \
  tests/test_dynamic_powerfoam_metal.py

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train uv run --with pytest python -m pytest \
  -p no:cacheprovider \
  tests/test_powerfoam_implicit_camera.py \
  tests/test_dynamic_powerfoam_metal.py \
  -q -rs
```

Result:

```text
29 passed in 5.75s
```

Runtime smoke:

```text
/tmp/dynaworld_powerfoam_orbit_static_dynamic_smoke.jsonc
step 0 mean PSNR 13.0392, L1 0.14653
step 1 mean PSNR 12.7845, L1 0.15066
screen motion 16.37px mean / 38.47px p95, valid fraction 1.0
```
