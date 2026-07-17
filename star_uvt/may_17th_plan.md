# STAR UVT May 17 Plan

Date: 2026-05-17
Workspace: `/Users/nicholasbardy/git/gsplats_browser/dynaworld`

## Objective

Make STAR UVT run as the fast 64-frame source-view overfit lane and produce a
clear speed/accuracy read:

```text
single video window -> 64 frames -> center-square crop -> STAR UVT overfit
metrics: loss, PSNR, SSIM mean/min/per-frame, wall clock, render timing
```

The goal is not a heldout claim. It is an overfit/convergence/speed gate.

## Phase 0: Scope And Safety

Rules:

- keep work local only;
- do not launch Modal/SkyPilot/RunPod/Vast or paid jobs;
- preserve unrelated dirty-tree changes;
- keep edits scoped to STAR docs and the STAR research harness unless evidence
  demands a renderer-kernel change;
- write raw progress under `star_uvt/` and use existing benchmark result folders
  for JSON/media artifacts.

Current branch is already a STAR/PRT fork branch:

```text
codex/satar-prt-compact-backward
```

Do not create a second fork unless branch state forces it.

## Phase 1: Measurement Patch

Patch `star_uvt_v0` harness:

```text
third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/trainer_harness/data.py
third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/benchmarks/video_fit_comparison.py
```

Required additions:

- load video windows by `start_seconds`, `fps`, `frame_count`;
- pass `image_crop_mode=center_square`;
- compute SSIM over all frames;
- report:
  - `final_ssim_mean`;
  - `final_ssim_min`;
  - `final_ssim_per_frame`;
  - `final_dssim_mean`.

Implementation notes:

- use Dynaworld's existing `load_video_window_sequence`;
- implement SSIM locally with grouped `conv2d`, Gaussian window, and no new
  dependency;
- keep default behavior unchanged when no start/fps arguments are provided;
- keep MSE/PSNR fields intact so old result readers do not break.

Verification:

```bash
uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python -m py_compile \
  third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/trainer_harness/data.py \
  third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/benchmarks/video_fit_comparison.py
```

Then build the extension:

```bash
( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/star_uvt_v0
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

## Phase 2: Smoke Gate

Use the real first 64-frame YouTube window, but at small resolution first:

```text
video: data/youtube_scene_distinct/raw/KUDJ8HDFVQo.mp4
start_seconds: 2.0
fps: 23.976023976023978
frames: 64
crop: center_square
```

Smoke command shape:

```bash
uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python \
  third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/benchmarks/video_fit_comparison.py \
  data/youtube_scene_distinct/raw/KUDJ8HDFVQo.mp4 \
  --target-size 128 --max-frames 64 --start-seconds 2.0 --fps 23.976023976023978 \
  --image-crop-mode center_square \
  --tube-count 1024 --steps 5 --lr 0.12 --device mps \
  --uvt-init-mode video_samples --uvt-sample-mode stratified \
  --uvt-spatial-precision 0.125 --uvt-temporal-precision 2.0 --uvt-opacity 0.7 \
  --uvt-render-backend metal_tile --uvt-tile-t 1 --uvt-tile-capacity 128 \
  --uvt-sample-emission-mode direct_atomic --uvt-reduction-mode index_add \
  --skip-per-frame \
  --out-json star_uvt/results/may17_smoke_128_64f_directatomic_5step.json \
  --contact-sheet star_uvt/results/may17_smoke_128_64f_directatomic_5step.png
```

Pass condition:

- command finishes;
- frames field is `64`;
- loss decreases;
- SSIM metrics are present and finite;
- no Metal overflow crash.

## Phase 3: First Real 64-frame Matrix

Run direct-atomic first. Candidate matrix:

```text
128px, 64f, 1024 tubes, 50 steps
128px, 64f, 2048 tubes, 100-200 steps
256px, 64f, 4096 tubes, 50 steps
256px, 64f, 8192 tubes, 100-200 steps
512px, 64f, 8192 tubes, 50 steps
512px, 64f, 16384 tubes, 50-200 steps if speed holds
```

Use `tile_load_reg=0.003,target=60` as the first speed/quality bracket for
256/512 if support grows. Also test unregularized if speed is already acceptable.

Main command template:

```bash
uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python \
  third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/benchmarks/video_fit_comparison.py \
  data/youtube_scene_distinct/raw/KUDJ8HDFVQo.mp4 \
  --target-size ${SIZE} --max-frames 64 --start-seconds 2.0 --fps 23.976023976023978 \
  --image-crop-mode center_square \
  --tube-count ${TUBES} --steps ${STEPS} --lr 0.12 --device mps \
  --uvt-init-mode video_samples --uvt-sample-mode stratified \
  --uvt-spatial-precision 0.125 --uvt-temporal-precision 2.0 --uvt-opacity 0.7 \
  --uvt-render-backend metal_tile --uvt-tile-t 1 --uvt-tile-capacity 128 \
  --uvt-sample-emission-mode direct_atomic --uvt-reduction-mode index_add \
  --render-benchmark-repeats 10 --skip-per-frame \
  --out-json star_uvt/results/${RUN}.json \
  --contact-sheet star_uvt/results/${RUN}.png
```

Rank rows by:

```text
final_ssim_mean
final_ssim_min
final_psnr
wall_clock_ms / steps
render_benchmark_ms.median
loss_ratio
```

## Phase 4: If Direct Atomic Is Fast But Quality Is Poor

Try in this order:

1. Increase tubes at same resolution.
2. Use `uvt-sample-mode temporal_quarters` if motion spans are undercovered.
3. Use temporal split at mid-run:

```text
--uvt-temporal-split-step 100
--uvt-temporal-split-offset 0.5
--uvt-temporal-split-precision-scale 2.0
--uvt-temporal-split-opacity-scale 1.0
```

4. Add appearance refine:

```text
--uvt-appearance-refine-steps 50
--uvt-appearance-lr 0.04
```

5. Use block-match gated velocity init only if visual artifacts suggest motion
   underfit:

```text
--uvt-velocity-init block_match_gated
--uvt-velocity-min-improvement-ratio 0.9
```

Avoid jumping to PRT until the screen-space source-view gate is understood.

## Phase 5: If Direct Atomic Is Fast And Accurate

Run one deterministic comparison at a smaller/representative budget:

```text
sample_emission_mode=tile_pair
reduction_mode=key_sort_scan_metal
```

Purpose:

- not to replace direct atomic for the speed loop;
- only to quantify how much deterministic promotion still costs on the new
  64-frame window.

If deterministic is still 3-5x slower, document it and continue direct-atomic
exploration for overfit quality.

## Phase 6: First Work-block Critique

After the first extended local work block, write:

```text
star_uvt/critique.md
star_uvt/plan.md
```

Critique requirements:

- what ran;
- exact commands or result JSONs;
- speed table;
- PSNR/SSIM table;
- visual read from contact sheets/media;
- whether direct atomic is acceptable for overfit exploration;
- whether deterministic compact backward remains the blocker;
- whether next work should be kernel work, init/capacity work, or trainer
  integration work.

## Phase 7: Main-trainer Scale Bridge

STAR UVT is not yet a production renderer mode, but the strongest practical
finding from the STAR harness is already portable:

```text
coarse 256px optimization -> 512px finishing
```

Implemented bridge:

```text
train.render_size_schedule = [
  {"start_step": 0, "render_size": 256},
  {"start_step": N, "render_size": 512}
]
```

This changes the active render size, dense grid, and renderer-mode selection at
the configured step while leaving the model, cached V-JEPA conditioning, static
/ dynamic split, register tokens, and implicit camera path unchanged.

Verification gate:

```text
1-step precomputed-feature smoke:
step 0 render_size 64 -> step 1 render_size 128
cache hit
render/rasterize 0.1374s
step_total 4.8071s
```

Completed overfit bridge run:

```text
screen: dynaworld_overfit_multires_256to512_20260517_112425
wandb: wandb/offline-run-20260517_112427-acv8pinq
wall: 16:36 for 400 steps
final eval PSNR/SSIM: 24.766 / 0.5316
256px median step_total: 1.676s
512px median step_total: 3.187s
```

The schedule is mechanically verified by media dimensions: early preview strips
are 512x256, late preview strips are 1024x512, and the final render video is
512x512.

Next main-trainer run:

```text
src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_gpt256_8192splats_overfit1_400step_multires_256to512.jsonc
```

Only promote to the 300-clip 3k schedule after the overfit row shows that the
coarse 256px stage does not damage final 512px SSIM.

This overfit row did not clear that quality condition. Keep the mechanism, but
compare against the existing 512-only overfit before launching the 300-clip 3k
schedule.

## Stop Conditions

Stop and reassess if any of these happen:

- extension build fails;
- 64-frame smoke does not reduce loss;
- direct atomic crashes or produces NaNs;
- 256px/64f is already slower than the old token trainer recon-only path;
- SSIM is flat while PSNR rises, indicating blurry/color-only overfit;
- tile-load regularization speeds up but visibly prevents detail.

## Best Guess Initial Target

The best first real result is likely:

```text
256px, 64f, 4096-8192 tubes, 100-200 steps,
direct_atomic + index_add, tile_t=1, cap128,
video_samples + stratified init,
optional tile_load_reg=0.003,target=60.
```

The 512px/64f/16384-tube target is the user-shaped scale target, but it should be
attempted after a 256px row proves the harness and metric path.
