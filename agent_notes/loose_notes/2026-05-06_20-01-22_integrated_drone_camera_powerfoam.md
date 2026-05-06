# Integrated Drone Camera PowerFoam

## Context

The user wanted the learned camera path to overfit from renderer loss while
optionally using a camera-following model only as initialization supervision.
The direct request was to implement the three-phase version at once:

1. optional teacher/path initialization for the camera decoder
2. renderer-only overfit with a static/no-repaint warmup
3. full dynamic feature PowerFoam after warmup

This was implemented in the existing dynamic PowerFoam Metal trainer instead of
adding another script.

## Code Changes

- `src/train/powerfoam_implicit_camera.py`
  - Added `camera.path_parameterization = "integrated_drone"`.
  - Kept `"pose_delta"` as the default for compatibility.
  - Added start pose, initial velocity, acceleration, angular acceleration, and
    optional gimbal heads.
  - Integrated dynamics across the full frame horizon, then indexed requested
    frames so random train batches remain consistent with the whole path.
  - Zero-init still reproduces the configured base camera exactly.

- `src/train/train_dynamic_powerfoam_metal.py`
  - Added config defaults/validation for the drone dynamics knobs.
  - Added optional one-time camera teacher prefit via `camera.init_teacher_path`
    and `camera.init_teacher_steps`.
  - The teacher prefit writes `camera_teacher_init_metrics.json` and is not used
    as a main training loss.
  - Added camera velocity/acceleration/gimbal regularization terms and W&B keys.

- New config:
  - `src/train_configs/local_mac_token_dynamic_powerfoam_features_F32_1024_128dyn_drone_camera_staticwarm_youtube_hlaZbH_center_crop_8fps_512_56f_180step.jsonc`
  - Uses the high-motion 512px/56f clip, F32 feature foam, 1024 cells, 128
    dynamic cells, `static_only_steps=60`, and `no_repaint_steps=90`.
  - W&B is enabled in the checked-in config.

## Dynamics Model

The new head predicts bounded dynamics:

```text
camera_token -> T0, v0, omega0
time_basis_tokens -> a_t, alpha_t, gimbal_t

v_{t+1} = damping * v_t + dt * a_t
p_{t+1} = p_t + dt * (R_t v_{t+1})     # body-frame translation by default

omega_{t+1} = damping * omega_t + dt * alpha_t
R_{t+1} = R_t ExpSO3(dt * omega_{t+1})
R_render_t = R_t ExpSO3(gimbal_t)
```

`drone_integration_horizon` controls the normalized integration horizon. Larger
values make the same bounded velocity/acceleration heads create bigger camera
swings.

## Validation

Commands:

```bash
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/powerfoam_implicit_camera.py \
  src/train/train_dynamic_powerfoam_metal.py \
  tests/test_powerfoam_implicit_camera.py \
  tests/test_dynamic_powerfoam_metal.py

PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_powerfoam_implicit_camera.py \
  tests/test_dynamic_powerfoam_metal.py -q
```

Result:

```text
35 passed in 8.04s
```

Runtime smoke:

```bash
PYTHONPATH=src/train WANDB_MODE=offline uv run python \
  src/train/train_dynamic_powerfoam_metal.py /tmp/dynaworld_drone_camera_smoke.jsonc
```

The smoke patched the new config down to 3 frames, 32px, 16 cells, 4 dynamic
cells, and 1 step. It exercised MPS, the Metal PowerFoam op, feature colorizer,
static/dynamic mask, validation logging, MP4 writing, and integrated drone
camera state.

Smoke metrics:

```text
step 0: camera rotation 0 deg, translation 0
step 1: camera rotation 39.799 deg, translation 0.369
step 1: screen motion 59.598 px mean / 202.590 px p95
```

The final smoke MP4 was not a flat-green artifact by ffmpeg signal stats:

```text
render_step_0001 first-frame luma range: YMIN=26, YAVG=150.028, YMAX=219
```

## Full 512px Run

Command:

```bash
PYTHONPATH=src/train uv run python \
  src/train/train_dynamic_powerfoam_metal.py \
  src/train_configs/local_mac_token_dynamic_powerfoam_features_F32_1024_128dyn_drone_camera_staticwarm_youtube_hlaZbH_center_crop_8fps_512_56f_180step.jsonc
```

W&B:

```text
run id: gccogv4l
url: https://wandb.ai/nbardy/dynaworld/runs/gccogv4l
```

Output:

```text
outputs/dynamic_powerfoam_metal/local_mac_token_dynamic_powerfoam_features_F32_1024_128dyn_drone_camera_staticwarm_youtube_hlaZbH_center_crop_8fps_512_56f_180step
```

Key metrics:

```text
step 0:
    mean PSNR 11.1152, min PSNR 8.3591, L1 0.24520
    camera motion 0 deg / 0 translation

step 60:
    mean PSNR 11.0469, min PSNR 7.3369, L1 0.23674
    camera rotation 20.9648 deg, translation 0.4598
    feature temporal delta 0.0

step 120 best:
    mean PSNR 12.1951, min PSNR 6.6769, L1 0.21466
    camera rotation 21.3350 deg, translation 0.7903
    screen motion 136.04 px mean / 166.85 px p95
    feature temporal delta 0.00378

step 180 final:
    mean PSNR 7.8857, min PSNR 3.2175, L1 0.34708
    camera rotation 82.8515 deg, translation 0.2533
    screen motion 1908.92 px mean / 1756.73 px p95
    feature temporal delta 0.00417
```

The full run is a negative quality result as configured. It proves the drone
head can learn materially larger camera motion than the residual-camera branch,
but after the full dynamic/repaint release it over-rotates and loses support
coverage (`eval_alpha_mean` falls to `0.337` by step 180). The best eval is
step 120, still far behind the residual fixed-gauge branch (`~18.09` mean PSNR,
`~0.089` L1).

Final MP4 first-frame signal stats were non-flat/non-green:

```text
render_step_0180 first-frame luma range: YMIN=10, YAVG=125.263, YMAX=186
```

The checked-in config was updated after the run from `video_log_every=180` to
`video_log_every=60`, because the best eval happened at step 120 and this run
only preserved final MP4s. Future runs should keep videos at 60/120/180.

## Camera Path Measurement

Measured from `checkpoint_final.pt` by reloading the camera decoder and
extracting `camera_to_world_matrices()` for all 56 frames.

Drone final checkpoint:

```text
adjacent rotation:
    mean 15.7227 deg/frame
    median 16.7528 deg/frame
    p95 23.2528 deg/frame
    max 24.4180 deg/frame

adjacent translation:
    mean 0.05790 units/frame
    median 0.06460 units/frame
    p95 0.08971 units/frame
    max 0.09013 units/frame

aggregate path:
    summed adjacent rotation 864.75 deg
    translation path length 3.1843 units
    path length / base radius 1.0614
    first-to-last rotation 140.7904 deg
    first-to-last translation 0.5293 units
    max displacement from first 0.5293 units
    max displacement / base radius 0.1764

relative to base camera:
    mean rotation from base 82.8454 deg
    max rotation from base 179.1425 deg
    mean translation from base 0.2533 units
    max translation from base 0.4747 units
    max forward-axis change from first 76.3989 deg

camera center bbox:
    min [-0.2793, -0.3379, -0.4721]
    max [ 0.2156,  0.1330,  0.0539]
    size [0.4949, 0.4709, 0.5260]
```

Residual-camera comparator final checkpoint:

```text
adjacent rotation mean 0.01220 deg/frame, max 0.05595
adjacent translation mean 0.00216 units/frame, max 0.00665
summed adjacent rotation 0.6707 deg
translation path length 0.1187 units
first-to-last rotation 0.1187 deg
first-to-last translation 0.00369 units
mean rotation from base 0.1330 deg
mean translation from base 0.01163 units
```

Interpretation: the drone head did not merely move more than residual camera; it
exploded into a near-spinning camera path by the final checkpoint. The step-120
best checkpoint was not saved, so detailed adjacent-frame path metrics are only
available for final; the logged step-120 summary still says `21.335deg` mean
rotation-from-base and `0.7903` mean translation-from-base.

## Current Belief

The implementation path is viable, but the ambitious config is too loose. The
full run demonstrates the intended degree of camera motion, but not useful
reconstruction. The next branch should keep the integrated-drone head and clamp
the horizon/caps/LR rather than returning to hardcoded `orbit_yaw`.

## Falsification Tests

1. Run the new 512px/56f/180-step config as-is.
   - Status: done, weakened. Best step 120 reached only `12.1951` mean PSNR /
     `0.21466` L1, then final collapsed to `7.8857` mean PSNR / `0.34708` L1.

2. Run a camera-clamped sibling.
   - Keep `path_parameterization="integrated_drone"` but reduce horizon/caps
     by 4x.
   - If quality improves while motion remains nonzero, the new head is too
     aggressive rather than structurally wrong.

3. Add a teacher-init JSON from a camera-following model.
   - Use `camera.init_teacher_path` and `init_teacher_steps`.
   - Keep the main training loop renderer-only.
   - If teacher init helps early PSNR without locking the final path, keep it as
     the Phase 0 bootstrap.

## Progressive Camera-Curriculum Follow-Up

Motivation: the final integrated-drone run proved the head can move the camera,
but it released the whole 56-frame path at once and collapsed into high-energy
rotation. The next implementation should make the optimizer solve an easier
path problem first: frame 0 only, then a short prefix, then the next chunk.

Code changes made:

- `PowerFoamImplicitCameraDecoder` now supports `initial_zoom_steps` and
  `initial_zoom_translation`. This is a deterministic smoothstep dolly along
  local camera +z, so zero learned dynamics can begin with a gentle zoom rather
  than a perfectly static camera.
- The camera decoder now has `set_active_frame_count()`. During curriculum,
  suffix camera deltas are detached so camera loss and eval calls do not leak
  gradients into future frames before their stage opens.
- `train_dynamic_powerfoam_metal.py` now supports
  `train.camera_curriculum_enabled` plus
  `train.camera_curriculum_schedule: [[step, active_frames], ...]`.
  Training samples frames only from `[0, active_frames)`, and W&B/history log
  `Stage/CameraActiveFrames` / `Camera/ActiveFrames`.
- New config:
  `src/train_configs/local_mac_token_dynamic_powerfoam_features_F32_1024_128dyn_drone_camera_progressive_clamped_youtube_hlaZbH_center_crop_8fps_512_56f_180step.jsonc`
  with schedule `1 -> 3 -> 7 -> 11 -> 16 -> 24 -> 36 -> 48 -> 56`, a
  `0.12` local-z zoom over 8 frames, lower camera LR, and tighter drone caps.

Validation:

```text
/Users/nicholasbardy/git/gsplats_browser/dynaworld/.venv/bin/python -m py_compile \
  src/train/powerfoam_implicit_camera.py \
  src/train/train_dynamic_powerfoam_metal.py \
  tests/test_dynamic_powerfoam_metal.py \
  tests/test_powerfoam_implicit_camera.py

PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_powerfoam_implicit_camera.py \
  tests/test_dynamic_powerfoam_metal.py -q
# 39 passed in 9.79s
```

Trainer smoke:

```text
PYTHONPATH=src/train WANDB_MODE=disabled .venv/bin/python \
  src/train/train_dynamic_powerfoam_metal.py /tmp/dynaworld_progressive_camera_smoke.jsonc
```

The smoke used a patched 4-frame/64px/1-step copy of the progressive config.
It exercised config resolution, frame-prefix sampling, MPS forward/backward,
validation render, MP4 save, and summary writing. Step 0 logged
`state_camera_active_frames=1.0` and the expected zoom-only camera displacement
(`state_camera_translation_delta_mean=0.06` for the 4-frame subset). Step 1
completed with `eval_l1=0.04352`, mean PSNR `23.145`, and active frames still
`1.0`.

First-frame render sanity on the smoke MP4:

```text
render_step_0001.mp4 signalstats:
YMIN=14
YAVG=143.062
YMAX=216
SATAVG=11.7725
```

Interpretation: the implementation path is wired and no longer produces a flat
green smoke artifact. It still needs the full 512px/56f/180-step run to learn
whether the progressive schedule actually prevents the late camera spin seen in
the earlier integrated-drone run.

## Full Progressive-Clamped Run

Run:

```text
PYTHONPATH=src/train .venv/bin/python \
  src/train/train_dynamic_powerfoam_metal.py \
  src/train_configs/local_mac_token_dynamic_powerfoam_features_F32_1024_128dyn_drone_camera_progressive_clamped_youtube_hlaZbH_center_crop_8fps_512_56f_180step.jsonc
```

W&B:

```text
https://wandb.ai/nbardy/dynaworld/runs/c4s320dh
```

Output:

```text
outputs/dynamic_powerfoam_metal/local_mac_token_dynamic_powerfoam_features_F32_1024_128dyn_drone_camera_progressive_clamped_youtube_hlaZbH_center_crop_8fps_512_56f_180step
```

Eval summary:

```text
step 0:   mean PSNR 11.0943, min PSNR 8.4796, L1 0.24636, active camera frames 1
step 60:  mean PSNR  9.9114, min PSNR 5.4861, L1 0.26594, active camera frames 16
step 120: mean PSNR 11.3746, min PSNR 6.0958, L1 0.21840, active camera frames 36
step 180: mean PSNR 13.0775, min PSNR 6.2103, L1 0.16383, active camera frames 56
```

Final artifacts:

```text
outputs/dynamic_powerfoam_metal/local_mac_token_dynamic_powerfoam_features_F32_1024_128dyn_drone_camera_progressive_clamped_youtube_hlaZbH_center_crop_8fps_512_56f_180step/render_step_0180.mp4
outputs/dynamic_powerfoam_metal/local_mac_token_dynamic_powerfoam_features_F32_1024_128dyn_drone_camera_progressive_clamped_youtube_hlaZbH_center_crop_8fps_512_56f_180step/side_by_side_step_0180.mp4
outputs/dynamic_powerfoam_metal/local_mac_token_dynamic_powerfoam_features_F32_1024_128dyn_drone_camera_progressive_clamped_youtube_hlaZbH_center_crop_8fps_512_56f_180step/checkpoint_final.pt
outputs/dynamic_powerfoam_metal/local_mac_token_dynamic_powerfoam_features_F32_1024_128dyn_drone_camera_progressive_clamped_youtube_hlaZbH_center_crop_8fps_512_56f_180step/dynamic_geometry_summary.json
```

Final first-frame render sanity:

```text
render_step_0180.mp4 signalstats:
YMIN=7
YAVG=158.83
YMAX=230
SATAVG=12.5381
```

Detailed camera path at final checkpoint:

```text
adjacent rotation mean 0.6579 deg/frame, median 0.7215, p95 1.3031, max 1.3551
adjacent translation mean 0.01830, median 0.01987, p95 0.02616, max 0.03231
summed adjacent rotation 36.18 deg
translation path length 1.0065 units
path length / base radius 0.3355
first-to-last rotation 30.3079 deg
first-to-last translation 0.9749 units
mean rotation from base 10.7215 deg
max rotation from base 31.4526 deg
mean translation from base 0.4525 units
max translation from base 0.9690 units
max forward-axis change from first 22.7208 deg
camera center bbox size [0.0722, 0.4334, 0.8703]
```

Interpretation: the progressive/clamped schedule fixed the pathological camera
spin. The old integrated-drone final had `82.85deg` mean rotation from base and
`864.75deg` summed adjacent rotation; this run ended at `10.72deg` mean
rotation and `36.18deg` summed adjacent rotation. It also improved final
quality over the old drone final (`13.08` PSNR / `0.1638` L1 vs `7.89` /
`0.3471`), but it still does not approach the residual fixed-gauge PowerFoam
branch around `18.08` PSNR / `0.0893` L1.

The camera learned a smooth modest path, not the full apparent 180-degree scene
motion. That may be correct for this single-view overfit if the foam/appearance
can absorb too much, but it means unsupervised camera path learning still needs
a stronger gauge or bootstrap before it can recover large camera arcs from this
clip.
