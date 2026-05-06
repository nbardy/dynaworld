# YouTube High-Motion Dynamic PowerFoam Benchmark

## Context

The user asked to stop leaning on unit tests alone and run a real training fit
on a bigger local YouTube clip with high camera motion, while preserving the
benchmark/reflection trail for future agents.

Relevant existing result before this pass:

- `outputs/dynamic_powerfoam_metal/motion_vs_repaint_comparison_1024_16f_40step_20260506.json`
- Same 16f/64px/1024-cell explicit-video smoke.
- Geometry-only beat fixed-geometry color-only on mean/min SNR and PSNR, but
  color-only had lower L1.

## Clip Selection

The local `data/youtube_motion/` seed was not materialized. I ranked the local
YouTube assets instead.

Added utility:

```text
research_experiments/dynamic_foam/rank_video_motion.py
```

It scores a cheap proxy: mean grayscale absolute difference between adjacent
sampled frames after square downsample. This is not a camera-pose estimator; it
is a fast high-motion screen-space proxy.

Ranking artifacts:

```text
outputs/dynamic_powerfoam_metal/youtube_motion_rank_segments_20260506.json
outputs/dynamic_powerfoam_metal/youtube_motion_rank_curated_raw_20260506.json
```

Top pure segment:

```text
data/youtube_scene_distinct/segments/C-OAFv5uGOw_seg_000.mp4
source 640x360
mean_absdiff 0.1953573823
```

Chosen larger-source clip:

```text
data/youtube_curated_spans/raw/hlaZbH_OFBU_seg_003_s00131000_e00138000.mp4
source 1280x640
mean_absdiff 0.1572056860
```

Important trainer caveat: `load_video_sequence(...)` reads the first
consecutive frames from a video. It does not sample at 4fps. To make the
training clip match the ranking window, I materialized a sampled 4fps/16-frame
video:

```text
data/youtube_curated_spans/high_motion_smokes/hlaZbH_OFBU_seg_003_4fps_16f.mp4
1280x640, 4fps, 16 frames, 4.0s
```

## Benchmark Matrix

Added configs:

```text
src/train_configs/local_mac_dynamic_powerfoam_metal_rbf_geometry_only_youtube_hlaZbH_128_16f_40step.jsonc
src/train_configs/local_mac_dynamic_powerfoam_metal_rbf_color_only_fixed_geometry_youtube_hlaZbH_128_16f_40step.jsonc
```

Shared setup:

```text
video: data/youtube_curated_spans/high_motion_smokes/hlaZbH_OFBU_seg_003_4fps_16f.mp4
frames: 16
render_size: 128
cells: 1024
steps: 40
device: mps
dynamic_mode: rbf
```

Geometry-only branch:

```text
dynamic_centers true
dynamic_radii true
dynamic_features false
```

Color-only branch:

```text
dynamic_centers false
dynamic_radii false
dynamic_features true
```

## Results

Comparison artifact:

```text
outputs/dynamic_powerfoam_metal/motion_vs_repaint_comparison_youtube_hlaZbH_128_16f_40step_20260506.json
ok true
winner_by_mean_snr geometry_only
winner_by_min_snr geometry_only
```

Geometry-only final:

```text
eval_l1 0.1701234430
eval_mse 0.0539068319
mean/min PSNR 13.2116289139 / 8.0830202103
mean/min SNR  8.3076114655 / -2.1480443478
mean/p95 temporal screen delta 1.0672498941 / 2.5281271935 px
alpha/support delta 0.0064221271 / 0.0053751627
feature temporal delta 0.0
```

Color-only fixed-geometry final:

```text
eval_l1 0.1841649860
eval_mse 0.0610489808
mean/min PSNR 12.7163810730 / 7.5698680878
mean/min SNR  7.8123636246 / -2.6611959934
mean/p95 temporal screen delta 0.0 / 0.0 px
alpha/support delta 0.0 / 0.0
feature temporal delta 0.0049524177
```

Geometry minus color-only:

```text
mean PSNR +0.4952478409
min PSNR  +0.5131521225
mean SNR  +0.4952478409
min SNR   +0.5131516457
L1        -0.0140415430
MSE       -0.0071421489
```

Unlike the earlier 64px test clip, geometry-only won all tracked quality
scalars here, including L1. The worst frame remains genuinely hard: frame 8 has
negative SNR in both branches, but geometry-only is less bad.

## Current Model

Current belief: on a higher-motion monocular YouTube clip, a constrained
geometry-only dynamic foam is not merely repainting. It changes screen geometry,
alpha, and support with features frozen, and it beats a fixed-geometry repaint
control on mean and worst-frame SNR/PSNR.

Confidence: medium for the mechanics claim, low-to-medium for general quality.

Evidence:

- geometry branch has nonzero screen/alpha/support temporal motion and zero
  temporal feature drift
- color branch has zero screen/alpha/support motion and nonzero feature drift
- same clip, seed, steps, cells, and render size
- comparison verifier passed

Could be wrong if:

- frame-difference ranking selected object/cut motion rather than camera motion
- 40 steps is too short and color-only catches up at longer schedules
- the 128px resize makes the task mostly appearance statistics rather than
  true geometry
- monocular train-fit quality does not transfer to heldout/interpolation frames

## Branches And Falsification Tests

Hypothesis:
    Geometry motion helps because the clip has parallax/support changes that a
    fixed alpha/support field cannot absorb cleanly.
Cheap test:
    Add frame-to-frame residual flow or homography-compensated residual metrics;
    geometry should improve residual after dominant camera motion is removed.

Hypothesis:
    The result is mostly an initialization/schedule artifact.
Cheap test:
    Repeat 80 or 120 steps with the same two configs and compare whether the
    color-only branch closes the SNR and L1 gap.

Hypothesis:
    The chosen clip includes a scene cut or exposure shock, not camera motion.
Cheap test:
    Inspect frame 8 in the side-by-side video and run a simple scene-cut score
    or feature-track survival metric over the sampled 16 frames.

Hypothesis:
    Dynamic feature foam should improve both branches by using F16/F32 features
    plus a colorizer, but it may simply repaint more effectively.
Cheap test:
    Run a token F32 feature-foam config on this same sampled clip and require
    the summary to include temporal screen motion, feature drift, PCA video, and
    per-frame SNR.

## Artifacts

Training outputs:

```text
outputs/dynamic_powerfoam_metal/local_mac_dynamic_powerfoam_metal_rbf_geometry_only_youtube_hlaZbH_128_16f_40step
outputs/dynamic_powerfoam_metal/local_mac_dynamic_powerfoam_metal_rbf_color_only_fixed_geometry_youtube_hlaZbH_128_16f_40step
```

Verification:

```text
research_experiments/dynamic_foam/verify_dynamic_powerfoam_geometry_run.py ... --require-geometry-motion --require-alpha-support-motion --require-appearance-freeze-control
ok true

research_experiments/dynamic_foam/compare_dynamic_powerfoam_motion_vs_repaint.py ...
ok true

py_compile research_experiments/dynamic_foam/rank_video_motion.py
passed
```

## 512px Center-Crop 8fps Follow-Up

The user asked to run 512x512 too, increase FPS, and make sure the video is
center-cropped instead of square-warped.

Derived clip:

```text
data/youtube_curated_spans/high_motion_smokes/hlaZbH_OFBU_seg_003_center_crop_8fps_16f.mp4
source raw: data/youtube_curated_spans/raw/hlaZbH_OFBU_seg_003_s00131000_e00138000.mp4
derived: 640x640, 8fps, 16 frames, 2.0s
ffmpeg crop: center crop 1280x640 -> 640x640, then fps=8
```

Configs:

```text
src/train_configs/local_mac_dynamic_powerfoam_metal_rbf_geometry_only_youtube_hlaZbH_center_crop_8fps_512_16f_40step.jsonc
src/train_configs/local_mac_dynamic_powerfoam_metal_rbf_color_only_fixed_geometry_youtube_hlaZbH_center_crop_8fps_512_16f_40step.jsonc
```

Comparison:

```text
outputs/dynamic_powerfoam_metal/motion_vs_repaint_comparison_youtube_hlaZbH_center_crop_8fps_512_16f_40step_20260506.json
ok true
winner_by_mean_snr geometry_only
winner_by_min_snr geometry_only
```

512 geometry-only:

```text
train loop elapsed_s 7.6606
full process wall time 24.95s
eval_l1 0.1157351658
eval_mse 0.0322180167
mean/min PSNR 15.2546272278 / 10.2760200500
mean/min SNR  10.5223588943 / 1.8168407679
mean/p95 temporal screen delta 4.2966279984 / 10.3185281754 px
alpha/support delta 0.0170116480 / 0.0151596069
feature temporal delta 0.0
```

512 fixed-geometry color-only:

```text
train loop elapsed_s 7.6853
full process wall time 28.39s
eval_l1 0.1360260099
eval_mse 0.0447325297
mean/min PSNR 14.1906528473 / 8.2194223404
mean/min SNR  9.4583845139 / -0.2397560477
mean/p95 temporal screen delta 0.0 / 0.0 px
alpha/support delta 0.0 / 0.0
feature temporal delta 0.0042034537
```

Geometry minus color-only:

```text
mean PSNR +1.0639743805
min PSNR  +2.0565977097
mean SNR  +1.0639743805
min SNR   +2.0565968156
L1        -0.0202908441
MSE       -0.0125145130
```

Performance note: with fixed 1024 cells, the 512 training loop itself is still
small (`~7.7s` for 40 one-frame steps). The full process wall time is much
larger because initial/final 512 eval and MP4/PNG artifact writes dominate. A
clean render-size scaling benchmark should separate train-step timing from
eval/media logging and should run matched 128/256/512 center-crop inputs.

W&B status correction: these runs were originally local-only because the
configs had `wandb_enabled: false` and the launch used `WANDB_MODE=disabled`.
The high-motion 128px and 512px configs have since been flipped to
`wandb_enabled: true`, and the repo guide now says benchmark/training runs
should keep W&B on. Current local CLI status reported no API key, so the
existing 512 MP4s are still local artifacts until the runs are rerun after
`wandb login` or `WANDB_API_KEY` is configured.

Green MP4 playback correction: the 512 MP4s looked solid green in the user's
local player even though ffmpeg/OpenCV decoded balanced RGB frames and the
preview PNGs were correct. The cause was the OpenCV `mp4v` writer path, which
produced MPEG-4 Part 2 MP4s that are not reliable in the local player/QuickTime
path. `train_powerfoam_metal.save_mp4` now writes ffmpeg/libx264 H.264
(`codec=h264`, `tag=avc1`, `pix_fmt=yuv420p`) when ffmpeg is available, with a
small regression test in `tests/test_powerfoam_direct.py`. Existing 512 render
and side-by-side MP4s for geometry-only and color-only were transcoded in place
to H.264. This was a media export/playback issue, not a loss-time raster issue:
train loss and val logging both call `model(indices)` and the same
`render_features_to_rgb(...)` path.

## Next Work

- Add a first-class sampled-video or `sample_fps` data contract so configs do
  not depend on a manually materialized sampled MP4.
- Add a first-class center-crop option to the video loader so we do not rely on
  derived MP4s to avoid aspect-ratio warping.
- Add timing fields that separate forward/backward/optimizer from eval and
  media writes; current wall time is not the same as rasterizer speed.
- Run the same clip with token F32 feature foam and demand per-frame metrics,
  PCA/feature diagnostics, and motion-vs-repaint fields.
- Add a camera-motion proxy: feature tracks, homography residual, or COLMAP-lite
  survival, so "high camera motion" is not just screen absdiff.
- Only append to `BASELINES.md` after the clip/protocol is stable enough to be
  a repeatable benchmark, not a one-off probe.
