# STAR UVT Critique

Date: 2026-05-17

## What Changed

The STAR UVT research harness now supports the 64-frame single-video overfit
contract we actually need:

- explicit video-window loading with `--start-seconds`, `--fps`, and
  `--duration-seconds`;
- crop parity via `--image-crop-mode center_square`;
- all-frame SSIM metrics:
  - `final_ssim_mean`;
  - `final_ssim_min`;
  - `final_ssim_max`;
  - `final_dssim_mean`;
  - `final_ssim_per_frame`;
- contact sheets that can sample across the full sequence via
  `--contact-sheet-mode linspace --contact-sheet-frames N`.

Files touched:

```text
third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/trainer_harness/data.py
third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/benchmarks/video_fit_comparison.py
star_uvt/may_17th_review.md
star_uvt/may_17th_plan.md
star_uvt/critique.md
star_uvt/plan.md
```

## What Ran

All runs were local on MPS. No remote or paid jobs.

Forest/manifest-first probe:

```text
video: data/youtube_scene_distinct/raw/KUDJ8HDFVQo.mp4
start_seconds: 2.0
fps: 23.976023976023978
frames: 64
crop: center_square
```

Curated high-motion probe:

```text
video: data/youtube_curated_spans/raw/hlaZbH_OFBU_seg_003_s00131000_e00138000.mp4
start_seconds: 2.0
fps: 29.97002997002997
frames: 64
crop: center_square
```

The high-motion raw clip is in the 300-clip 64f manifest at multiple starts:

```text
start 1.0, 2.0, 3.0, 4.0
source fps 30000/1001
source duration about 7.007s
source size 1280x640
```

## Result Table

| result | clip | size | tubes | steps | mode | wall s | PSNR | SSIM mean | SSIM min | render median ms | tile load |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `may17_smoke_128_64f_directatomic_5step.json` | forest | 128 | 1024 | 5 | direct | 5.56 | 10.000 | 0.0548 | 0.0378 | 17.31 | n/a |
| `may17_128_64f_2048t_directatomic_50step.json` | forest | 128 | 2048 | 50 | direct | 10.61 | 25.102 | 0.4211 | 0.4137 | 18.48 | n/a |
| `may17_128_64f_8192t_cap256_directatomic_50step.json` | forest | 128 | 8192 | 50 | direct | 11.75 | 26.135 | 0.4221 | 0.4135 | 19.29 | n/a |
| `may17_128_64f_8192t_cap256_directatomic_200step.json` | forest | 128 | 8192 | 200 | direct | 49.09 | 27.591 | 0.5518 | 0.5348 | 45.62 | n/a |
| `may17_128_64f_16384t_cap256_directatomic_100step.json` | forest | 128 | 16384 | 100 | direct | 48.47 | 28.002 | 0.5883 | 0.5686 | 83.35 | n/a |
| `may17_256_64f_8192t_cap256_directatomic_50step.json` | forest | 256 | 8192 | 50 | direct | 34.26 | 23.869 | 0.3357 | 0.3283 | 64.45 | n/a |
| `may17_highmotion_hlaZbH_start2_128_64f_8192t_cap256_directatomic_100step.json` | high motion | 128 | 8192 | 100 | direct | 37.74 | 26.873 | 0.7893 | 0.6856 | 59.42 | n/a |
| `may17_highmotion_hlaZbH_start2_256_64f_8192t_cap256_directatomic_50step.json` | high motion | 256 | 8192 | 50 | direct | 28.92 | 23.228 | 0.7134 | 0.6179 | 70.38 | 855.1 |
| `may17_highmotion_hlaZbH_start2_256_64f_16384t_cap256_directatomic_50step.json` | high motion | 256 | 16384 | 50 | direct | 31.03 | 24.723 | 0.7463 | 0.6453 | 79.53 | 409.1 |
| `may17_highmotion_hlaZbH_start2_256_64f_16384t_cap256_directatomic_200step.json` | high motion | 256 | 16384 | 200 | direct | 136.20 | 27.724 | 0.8162 | 0.7300 | 96.31 | 886.6 |
| `may17_highmotion_hlaZbH_start2_256_64f_16384t_cap256_directatomic_200step_app50.json` | high motion | 256 | 16384 | 200+50 app | direct | 149.69 | 27.764 | 0.8173 | 0.7308 | 84.59 | 886.3 |
| `may17_highmotion_hlaZbH_start2_256_64f_8192to16384_temporalsplit100_directatomic_200step.json` | high motion | 256 | 16384 | 200 | split100 | 130.47 | 27.625 | 0.8201 | 0.7165 | 122.08 | 1728.4 |
| `may17_highmotion_hlaZbH_start2_256_64f_16384t_cap256_directatomic_tileload001target300_50step.json` | high motion | 256 | 16384 | 50 | reg .001 | 23.58 | 24.719 | 0.7387 | 0.6447 | 54.98 | 383.7 |
| `may17_highmotion_hlaZbH_start2_256_64f_16384t_cap256_directatomic_tileload003target300_50step.json` | high motion | 256 | 16384 | 50 | reg .003 | 24.49 | 24.696 | 0.7190 | 0.6417 | 57.03 | 358.7 |
| `may17_highmotion_hlaZbH_start2_256_64f_32768t_cap256_directatomic_50step.json` | high motion | 256 | 32768 | 50 | direct | 23.81 | 26.648 | 0.7845 | 0.6951 | 54.11 | 226.9 |
| `may17_highmotion_hlaZbH_start2_256_64f_32768t_cap256_directatomic_200step.json` | high motion | 256 | 32768 | 200 | direct | 111.52 | 29.823 | 0.8572 | 0.7788 | 99.05 | 298.0 |
| `may17_highmotion_hlaZbH_start2_512_64f_32768t_cap256_directatomic_10step.json` | high motion | 512 | 32768 | 10 | direct | 7.33 | 11.044 | 0.2431 | 0.1454 | 67.89 | 165.9 |
| `may17_highmotion_hlaZbH_start2_512_64f_32768t_cap256_directatomic_50step.json` | high motion | 512 | 32768 | 50 | direct | 74.17 | 24.445 | 0.7475 | 0.5961 | 253.05 | 789.7 |
| `may17_highmotion_hlaZbH_start2_512_64f_32768t_cap256_directatomic_200step.json` | high motion | 512 | 32768 | 200 | direct | 452.80 | 27.878 | 0.8410 | 0.7553 | 475.58 | 907.1 |
| `may17_highmotion_hlaZbH_start2_512_64f_32768t_cap256_directatomic_tileload001target300_50step.json` | high motion | 512 | 32768 | 50 | reg .001 | 88.78 | 24.342 | 0.7129 | 0.4814 | 317.95 | 586.3 |
| `may17_highmotion_hlaZbH_start2_512_64f_32768t_cap256_directatomic_multires256c200_50fine.json` | high motion | 512 | 32768 | 200 coarse + 50 fine | multires | 188.93 | 29.135 | 0.8606 | 0.7794 | 292.73 | 1470.5 |
| `may17_highmotion_hlaZbH_start2_512_64f_32768t_cap256_directatomic_multires256c100_50fine.json` | high motion | 512 | 32768 | 100 coarse + 50 fine | multires | 151.30 | 28.551 | 0.8517 | 0.7655 | 386.28 | 918.6 |
| `may17_highmotion_hlaZbH_start2_512_64f_32768t_cap256_directatomic_multires256c200_25fine.json` | high motion | 512 | 32768 | 200 coarse + 25 fine | multires | 190.31 | 28.864 | 0.8561 | 0.7734 | 412.29 | 1211.7 |
| `may17_highmotion_hlaZbH_start2_512_64f_32768t_cap256_directatomic_multires256c100_25fine.json` | high motion | 512 | 32768 | 100 coarse + 25 fine | multires | 121.58 | 28.098 | 0.8442 | 0.7551 | 324.73 | 849.9 |
| `may17_highmotion_hlaZbH_start2_256_64f_8192t_cap256_tilepair_keysortscan_20step.json` | high motion | 256 | 8192 | 20 | deterministic | 46.62 | 17.598 | 0.6148 | 0.5462 | 42.41 | 463.0 |
| `may17_highmotion_hlaZbH_start2_256_64f_8192t_cap256_tilepair_suffix_keyseg_20step.json` | high motion | 256 | 8192 | 20 | det suffix/seg | 18.20 | 17.599 | 0.6148 | 0.5446 | 23.82 | 463.0 |
| `may17_highmotion_hlaZbH_start2_256_64f_8192t_cap256_tilepair_suffix_keyseg_50step.json` | high motion | 256 | 8192 | 50 | det suffix/seg | 216.53 | 23.228 | 0.7134 | 0.6179 | 38.83 | 855.1 |
| `may17_highmotion_hlaZbH_start2_256_64f_8192t_cap256_tilepair_suffix_keyseg_tileload001target300_50step.json` | high motion | 256 | 8192 | 50 | det suffix/seg reg | 170.98 | 23.108 | 0.6432 | 0.4636 | 28.16 | 581.5 |

## Visual Critique

The high-motion linspace sheets are the most important artifacts:

```text
star_uvt/results/may17_highmotion_hlaZbH_start2_256_64f_8192t_cap256_directatomic_50step.png
star_uvt/results/may17_highmotion_hlaZbH_start2_256_64f_16384t_cap256_directatomic_50step.png
star_uvt/results/may17_highmotion_hlaZbH_start2_256_64f_16384t_cap256_directatomic_200step.png
star_uvt/results/may17_highmotion_hlaZbH_start2_256_64f_16384t_cap256_directatomic_200step_app50.png
star_uvt/results/may17_highmotion_hlaZbH_start2_256_64f_32768t_cap256_directatomic_200step.png
star_uvt/results/may17_highmotion_hlaZbH_start2_512_64f_32768t_cap256_directatomic_200step.png
star_uvt/results/may17_highmotion_hlaZbH_start2_512_64f_32768t_cap256_directatomic_multires256c200_50fine.png
star_uvt/results/may17_highmotion_hlaZbH_start2_512_64f_32768t_cap256_directatomic_multires256c200_50fine.mp4
```

The model tracks:

- global camera path;
- coarse color;
- horizon/road/tree regions;
- dark interior versus bright exterior transition;
- all 64 frames without collapsing.

The model does not yet recover:

- sharp dirt/road texture;
- tree edge detail;
- thin high-contrast interior structure;
- frame-local high-frequency detail.

The 200-step 16k high-motion run is materially better than the 50-step rows, but
still looks like a smoothed/low-pass reconstruction. The +50 appearance-only
refine row moved PSNR by only +0.040 and SSIM mean by only +0.0011, with no
meaningful visual sharpening in the linspace sheet. That makes appearance-only
refinement a weak next lever; support, tube subdivision, or geometry/init are
more likely blockers.

The 32768-tube 256px row is the first actual scale-up win:

```text
PSNR 29.823
SSIM mean 0.8572
SSIM min 0.7788
wall 111.52s
```

It is visibly sharper than the 16k rows and recovers more road/tree structure,
but it is still not a high-frequency reconstruction. The 512px/32768/200 row is
feasible and passes the min-SSIM gate, but it takes 452.80s and still looks soft
in the predicted strip.

The multi-resolution row is the new 512px result to beat:

```text
coarse: 256px, 32768 tubes, 200 steps
fine: 512px, same tubes promoted by spatial scaling, 50 steps
wall 188.93s
PSNR 29.135
SSIM mean/min 0.8606 / 0.7794
render median 292.73ms
```

It beats the full 512px/200 run on speed and quality. The contact sheet remains
soft, but the schedule is clearly better than starting at 512px from step 0.

The follow-up schedule bracket says the default 512px choice should stay
`256c200 -> 512f50` when quality matters. `256c100 -> 512f50` is the faster
acceptable row at 151.30s with SSIM mean/min 0.8517 / 0.7655. Cutting the fine
stage to 25 steps was not a useful trade: `256c200 -> 512f25` barely saved wall
time and lost quality, while `256c100 -> 512f25` is fast but falls below the
0.85 mean-SSIM target.

## Speed Critique

The good news:

- STAR UVT is running on 64 frames.
- Direct atomic backward is not the old 36s-per-step V-JEPA-loss failure mode.
- 128px rows can run in tens of seconds.
- 256px/64f/16k/200 steps finished in about 136s.
- 256px/64f/32768/200 steps finished in 111.52s and is now the best quality row.
- 512px multi-resolution 256c200 -> 512f50 finished in 188.93s and beats the
  512px-from-scratch 200-step row.
- 512px multi-resolution 256c100 -> 512f50 finished in 151.30s and is the
  speed-biased row that still clears the 0.85 mean-SSIM gate.

The bad news:

- the "fast as fuck" target is only true relative to the V-JEPA-loss trainer,
  not relative to a high-quality 512px overfit;
- render medians rise hard with capacity and resolution;
- 256px/64f/32768 has median render about 99ms after optimization;
- 512px/64f/32768/200 takes 452.80s and has median render about 476ms;
- 512px multi-resolution still has median render about 293ms after promotion,
  so high-res forward/backward remains the next speed target;
- reducing the fine stage to 25 steps did not buy enough wall-clock savings to
  justify the quality drop;
- the current harness still uses direct atomics, so fast results are exploration
  rows, not deterministic promotion rows.

## Determinism Critique

This work did not solve deterministic compact backward. It intentionally used:

```text
sample_emission_mode=direct_atomic
reduction_mode=index_add
```

That is the fast exploration branch. The previous STAR state review still holds:

```text
fast sparse forward: yes
fast deterministic compact training backward: no
```

For this single-video overfit goal, direct atomic is acceptable for discovery.
For promotion, deterministic compact backward remains the blocker.

The fresh high-motion comparison row:

```text
256px, 64f, 8192 tubes, 20 steps
sample_emission_mode=tile_pair
reduction_mode=key_sort_scan_metal
wall: 46.62s
PSNR: 17.598
SSIM mean/min: 0.6148 / 0.5462
```

The suffix/segmented-key variant is a better 20-step deterministic probe:

```text
256px, 64f, 8192 tubes, 20 steps
sample_emission_mode=tile_pair_suffix
reduction_mode=key_sort_segmented_metal
wall: 18.20s
PSNR: 17.599
SSIM mean/min: 0.6148 / 0.5446
```

That is an important direction, but it does not solve the blocker. At 50 steps
the same suffix/segmented row balloons to 216.53s as tile load grows to 855.1.
Adding tile-load regularization reduces the proxy to 581.5 but still takes
170.98s and damages SSIM mean/min to 0.6432 / 0.4636. The deterministic path
needs load control or a different compact backward, not just the existing
segmented reducer.

## Representation Critique

The current primitive and init still behave as a low-pass fit at 64f:

- increasing tubes helps;
- increasing steps helps;
- higher resolution exposes blur;
- the initial broad support is useful for convergence but may encourage smooth
  support growth;
- a narrow-support forest probe (`spatial_precision=1.0`) was worse, so simply
  starting narrower is not enough.
- appearance-only refinement barely helps after the 16k/200 row, so color and
  opacity are not the main remaining bottleneck.
- 32768 tubes, which is 512 tubes/frame for a 64-frame clip, is the first strong
  capacity scaling result.
- 512px needs either a faster high-res backward path or a multi-resolution
  schedule; simply doing the 256px recipe at 512px is feasible but too slow.
- the first multi-resolution schedule works and should become the default 512px
  overfit strategy until the high-res backward path improves.

The quality ceiling probably needs one or more of:

- more tubes per frame;
- temporal split / tube subdivision after coarse alignment;
- support regularization that shrinks after convergence;
- appearance-only refinement after geometry/tube motion settles;
- smarter velocity/motion init;
- a fused loss/backward path that can afford higher capacity.

## Best Current Row

For the requested high-motion 64-frame overfit gate, the current best row is:

```text
star_uvt/results/may17_highmotion_hlaZbH_start2_256_64f_16384t_cap256_directatomic_200step.json
star_uvt/results/may17_highmotion_hlaZbH_start2_256_64f_16384t_cap256_directatomic_200step_app50.json
star_uvt/results/may17_highmotion_hlaZbH_start2_256_64f_32768t_cap256_directatomic_200step.json
star_uvt/results/may17_highmotion_hlaZbH_start2_512_64f_32768t_cap256_directatomic_multires256c200_50fine.json
star_uvt/results/may17_highmotion_hlaZbH_start2_512_64f_32768t_cap256_directatomic_multires256c100_50fine.json
```

Metrics:

```text
best 256px wall: 111.52s
best 256px PSNR: 29.823
best 256px SSIM mean/min: 0.8572 / 0.7788
best 512px wall: 188.93s
best 512px PSNR: 29.135
best 512px SSIM mean/min: 0.8606 / 0.7794
fast acceptable 512px wall: 151.30s
fast acceptable 512px SSIM mean/min: 0.8517 / 0.7655
```

This is a working STAR UVT 64f high-motion overfit and the first row that
clears the numeric overfit gate. It is still visually soft, so the next work is
not another appearance-only pass; it is high-res speed, better support geometry,
and deterministic compact backward.

## Main-Trainer Bridge Result

The STAR UVT screen-space harness is still separate from the production
precomputed-feature Dynaworld trainer, but the useful schedule idea now exists
in the main trainer as `train.render_size_schedule`.

Run:

```text
screen: dynaworld_overfit_multires_256to512_20260517_112425
config: src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_gpt256_8192splats_overfit1_400step_multires_256to512.jsonc
log: outputs/run_logs/dynaworld_overfit_multires_256to512_20260517_112425.log
wandb: wandb/offline-run-20260517_112427-acv8pinq
schedule: 256px steps 0-299, 512px steps 300-400
```

Outcome:

```text
completed 400/400
wall: 16:36
final train loss/recon: 0.0781 / 0.0777
final eval PSNR: 24.766
final eval SSIM: 0.5316
final eval L1/MSE: 0.0436 / 0.00334
```

Timing medians from W&B history:

```text
256px rows, n=28:
  step_total 1.676s
  backward 0.858s
  rasterize 0.153s
  vjepa_feature_loss 0.00077s

512px rows, n=10:
  step_total 3.187s
  backward 1.825s
  rasterize 0.295s
  vjepa_feature_loss 0.00083s
```

The final media verifies that the schedule actually promoted resolution:

```text
early preview images: 512x256 combined GT/pred strip
late preview images: 1024x512 combined GT/pred strip
final Render_Video_390: 512x512, 63 frames
```

Interpretation: the main trainer schedule is wired and much faster than the
old V-JEPA-loss lane, but it is not a quality breakthrough. The loss falls and
the 512px fine stage is stable, but final eval SSIM is still low. This supports
the current thesis: turning off differentiable V-JEPA loss fixes wall clock;
the remaining quality problem is representation/conditioning/camera/splat
capacity, not renderer forward cost alone.

## First-Class STAR Trainer Update

The STAR UVT harness is now launchable through the normal Dynaworld trainer
router:

```text
arch: star_uvt_video_overfit
trainer: src/train/train_star_uvt_video_overfit.py
router: src/train/train.py
```

Direct-atomic 256px high-motion row:

```text
config:
src/train_configs/star_uvt_highmotion_hlaZbH_64f_256_directatomic_200step.jsonc
result:
star_uvt/results/may17_firstclass_highmotion_hlaZbH_start2_256_64f_32768t_cap256_directatomic_200step.json
contact:
star_uvt/results/may17_firstclass_highmotion_hlaZbH_start2_256_64f_32768t_cap256_directatomic_200step.png
video:
star_uvt/results/may17_firstclass_highmotion_hlaZbH_start2_256_64f_32768t_cap256_directatomic_200step.mp4
wandb:
https://wandb.ai/nbardy/dynaworld/runs/jba7kztn
```

Metrics:

```text
steps: 200
tubes: 32768
final loss: 0.0010415
PSNR: 29.823
SSIM mean/min: 0.8572 / 0.7788
UVT wall: 183.99s
render median: 151.16ms
```

Quality reproduced the previous best 256px row almost exactly. Runtime was
slower than the previous saved `111.52s` row, so do not use this run alone for a
new speed claim. It does prove the first-class route preserves the STAR UVT
quality path while adding config/W&B/media integration.

Deterministic compact comparison through the same first-class route:

```text
config:
src/train_configs/star_uvt_highmotion_hlaZbH_64f_256_tilepair_suffix_keyseg_20step_profile.jsonc
result:
star_uvt/results/may17_firstclass_highmotion_hlaZbH_start2_256_64f_8192t_cap256_tilepair_suffix_keyseg_20step.json
contact:
star_uvt/results/may17_firstclass_highmotion_hlaZbH_start2_256_64f_8192t_cap256_tilepair_suffix_keyseg_20step.png
video:
star_uvt/results/may17_firstclass_highmotion_hlaZbH_start2_256_64f_8192t_cap256_tilepair_suffix_keyseg_20step.mp4
wandb:
https://wandb.ai/nbardy/dynaworld/runs/641gxm9l
```

Metrics:

```text
steps: 20
tubes: 8192
final loss: 0.017382
PSNR: 17.599
SSIM mean/min: 0.6148 / 0.5446
UVT wall: 39.54s
render median: 45.03ms
```

Quality matched the previous compact row, but timing regressed from the saved
`18.20s` row. Deterministic compact backward is therefore still not promoted.
The practical STAR path remains `direct_atomic + index_add`; compact backward
needs a focused load-growth/backward implementation pass before scaling.

The stopped Gaussian 300-clip multires run adds one more warning: it reached the
512px switch at step `2400`, then produced NaN total/camera losses around step
`2429`. That lane should not be treated as a stable scale-up baseline until the
512 promotion/camera stability problem is fixed.
