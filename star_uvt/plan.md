# STAR UVT Next Plan

Date: 2026-05-17

## Current Position

STAR UVT is running locally on 64-frame source-view overfit. The fast branch is:

```text
direct_atomic + index_add
```

The best high-motion run so far:

```text
256px, 64f, 32768 tubes, 200 steps
wall: 111.52s
PSNR: 29.823
SSIM mean/min: 0.8572 / 0.7788
```

This clears the numeric 256px overfit gate. The visual sheet is still soft, but
the lane is now alive in the intended sense: high-motion, natural-fps, 64 frames,
and under two minutes locally at 256px. The next plan is to carry this win to
512px and remove the deterministic backward blocker.

New 512px result:

```text
512px target, 32768 tubes
coarse 256px/200 steps -> promote to 512px -> fine 50 steps
wall: 188.93s
PSNR: 29.135
SSIM mean/min: 0.8606 / 0.7794
side-by-side MP4:
star_uvt/results/may17_highmotion_hlaZbH_start2_512_64f_32768t_cap256_directatomic_multires256c200_50fine.mp4
```

This beats 512px-from-scratch 200 steps both in speed and quality.

## Priority 1: High-motion 256px Quality Sweep

Use the same clip and start for comparability:

```text
data/youtube_curated_spans/raw/hlaZbH_OFBU_seg_003_s00131000_e00138000.mp4
start_seconds=2.0
fps=29.97002997002997
frames=64
crop=center_square
```

Run these rows:

### A. Appearance refine after 16k/200: done

Goal: see whether color/opacity-only refinement sharpens without moving support.

```text
256px, 64f, 16384 tubes, 200 main steps + 50 appearance steps
direct_atomic
```

Expected:

- small wall increase;
- PSNR/SSIM should improve if blur is mostly color/opacity;
- if visual texture stays blurred, geometry/support is the issue.

Result:

```text
star_uvt/results/may17_highmotion_hlaZbH_start2_256_64f_16384t_cap256_directatomic_200step_app50.json
wall: 149.69s
PSNR: 27.764
SSIM mean/min: 0.8173 / 0.7308
```

The improvement over the non-refined 16k/200 row was only +0.040 PSNR and
+0.0011 SSIM mean. Treat this as a weak negative for more appearance-only
refinement until support/tube layout improves.

### B. Temporal split: done

Goal: let broad tubes find coarse alignment, then split support in time.

```text
256px, 64f, 8192 -> 16384 tubes
split at step 100
total 200 steps
```

Expected:

- if temporal underfitting is the issue, later frames improve and SSIM min rises;
- if spatial support is the issue, it remains blurry.

Result:

```text
star_uvt/results/may17_highmotion_hlaZbH_start2_256_64f_8192to16384_temporalsplit100_directatomic_200step.json
wall: 130.47s
PSNR: 27.625
SSIM mean/min: 0.8201 / 0.7165
render median: 122.08ms
```

Mean SSIM rose slightly against 16k/200, but min SSIM and PSNR fell and render
cost increased. Do not promote this split shape.

### C. Support regularization bracket: done

Goal: reduce support bloat and render/backward cost while preserving quality.

Rows:

```text
tile_load_reg=0.001,target=300
tile_load_reg=0.003,target=300
tile_load_reg=0.003,target=200
```

Use 50-step probes first. Promote only if SSIM does not collapse.

Result:

```text
256px, 16k, reg .001,target=300, 50 steps:
wall 23.58s, PSNR 24.719, SSIM mean/min 0.7387 / 0.6447, render 54.98ms

256px, 16k, reg .003,target=300, 50 steps:
wall 24.49s, PSNR 24.696, SSIM mean/min 0.7190 / 0.6417, render 57.03ms
```

The weak reg can be useful as a speed knob, but it did not improve quality. The
stronger reg damages quality. At 512px, reg .001,target=300 was also negative:
88.78s for 50 steps with SSIM mean/min 0.7129 / 0.4814.

### D. 32768 tubes: promoted

Goal: test the user's 256 tubes/frame intuition at 256px and then one scale
above it.

```text
32768 tubes = 512 tubes/frame
256px, 64f, 50 steps
```

Stop if wall time or render median becomes silly. If it improves visual texture
strongly, run 100/200 steps.

Result:

```text
256px, 64f, 32768 tubes, 50 steps:
wall 23.81s, PSNR 26.648, SSIM mean/min 0.7845 / 0.6951

256px, 64f, 32768 tubes, 200 steps:
wall 111.52s, PSNR 29.823, SSIM mean/min 0.8572 / 0.7788
```

This is the current best lane. It validates scale-up: 32768 tubes equals 512
tubes/frame for 64 frames, and it is faster and better than the 16k/200 row.

## Priority 2: 512px Reality Check

Initial reality check is done:

```text
512px, 64f, 32768 tubes, 10 steps:
wall 7.33s, render median 67.89ms

512px, 64f, 32768 tubes, 50 steps:
wall 74.17s, PSNR 24.445, SSIM mean/min 0.7475 / 0.5961

512px, 64f, 32768 tubes, 200 steps:
wall 452.80s, PSNR 27.878, SSIM mean/min 0.8410 / 0.7553
```

The 512 lane is feasible but not fast. A blind 512 sweep is the wrong move until
we either add a multi-resolution schedule or improve the high-res backward path.

The multi-resolution schedule is now implemented in
`video_fit_comparison.py`:

```text
--uvt-coarse-target-size 256
--uvt-coarse-steps 200
```

It promotes learned UVT tubes by scaling center/velocity and spatial precision
before the 512px fine stage. The first promoted run took 188.93s versus 452.80s
for full 512px/200, and quality improved from SSIM 0.8410 / 0.7553 to
0.8606 / 0.7794.

Follow-up bracket:

```text
256c100 -> 512f50:
wall 151.30s, PSNR 28.551, SSIM mean/min 0.8517 / 0.7655

256c200 -> 512f25:
wall 190.31s, PSNR 28.864, SSIM mean/min 0.8561 / 0.7734

256c100 -> 512f25:
wall 121.58s, PSNR 28.098, SSIM mean/min 0.8442 / 0.7551
```

Decision:

```text
quality default: 256c200 -> 512f50
speed-biased acceptable row: 256c100 -> 512f50
do not use 25-step fine as the default; it gives up quality without enough
wall-clock savings.
```

The same idea is now wired into the main precomputed-feature Dynaworld trainer
as `train.render_size_schedule`. New configs:

```text
src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_gpt256_8192splats_overfit1_400step_multires_256to512.jsonc
src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_gpt256_8192splats_300clips_3kstep_multires_256to512.jsonc
```

The overfit config uses 256px through step 299 and 512px from step 300. The
300-clip config uses 256px through step 2399 and 512px from step 2400.

## Priority 3: Deterministic Comparison

After the direct-atomic high-motion quality target improves, run one small
deterministic row:

```text
128px or 256px
64f
8192 or 16384 tubes
tile_pair + key_sort_scan_metal
20-50 steps
```

Purpose:

- quantify deterministic tax on the new 64f clip;
- do not block quality exploration on it.

Result:

```text
256px, 64f, 8192 tubes, 20 steps
tile_pair + key_sort_scan_metal
wall 46.62s, PSNR 17.598, SSIM mean/min 0.6148 / 0.5462
```

This confirms the deterministic compact backward blocker. Direct atomic is the
right exploration branch; deterministic promotion still needs code work.

Follow-up deterministic probes:

```text
tile_pair_suffix + key_sort_segmented_metal, 20 steps:
wall 18.20s, PSNR 17.599, SSIM mean/min 0.6148 / 0.5446

tile_pair_suffix + key_sort_segmented_metal, 50 steps:
wall 216.53s, PSNR 23.228, SSIM mean/min 0.7134 / 0.6179

same plus tile_load_reg .001,target=300, 50 steps:
wall 170.98s, PSNR 23.108, SSIM mean/min 0.6432 / 0.4636
```

The segmented suffix path is a better short deterministic row than plain
`tile_pair + key_sort_scan_metal`, but it still blows up as learned support
grows. Tile-load regularization lowers render cost but does not fix training
wall time or quality. Deterministic compact work should target load growth and
gradient accumulation together.

## Next Execution

1. Use the new side-by-side MP4 export for all promoted 512px rows. Contact
   sheets are still useful, but the MP4 is the real motion artifact.
2. Tune the main-trainer render schedule locally before launching the 300-clip
   3k run. The first multires row beat the 512-only overfit on speed and final
   eval quality, so the next useful row is an earlier 512px switch, not another
   512-only baseline.
3. Work on deterministic compact backward with the narrowed target:
   `tile_pair_suffix + key_sort_segmented_metal` at 256px/64f/8192. Keep the
   20-step win, stop the 50-step load blow-up.
4. Try a high-capacity 256px quality row only after code changes: 32768 tubes
   already wins, but the residual blur likely needs better support geometry or
   motion init, not more appearance-only refinement.

## Priority 4: Save Better Visual Artifacts

The linspace contact sheet is good. Side-by-side MP4 export is now implemented
behind `--side-by-side-video`; use it for promoted 512px rows so motion quality
is inspectable:

```text
top or left: GT
bottom or right: STAR UVT render
fps: source fps
```

The first 512px promoted row wrote:

```text
star_uvt/results/may17_highmotion_hlaZbH_start2_512_64f_32768t_cap256_directatomic_multires256c200_50fine.mp4
```

## Decision Rules

Promote a row as the current overfit target if:

```text
SSIM mean >= 0.85
SSIM min >= 0.75
PSNR improves over 27.724
wall <= 3 minutes for 256px/64f
visual sheet shows road/tree/interior texture recovering, not just color blobs
```

If no direct-atomic row reaches that, the next code work should be:

```text
support/temporal split schedule first,
then fused loss/backward or deterministic compact backward work.
```

The direct-atomic 32768 row now reaches the numeric gate. Do not claim 512px
solved until it reaches similar quality in under a few minutes.

## Main-Trainer Bridge Status

The overfit multires config ran to completion:

```text
config: src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_gpt256_8192splats_overfit1_400step_multires_256to512.jsonc
wandb: wandb/offline-run-20260517_112427-acv8pinq
wall: 16:36 for 400 steps
final eval PSNR/SSIM: 24.766 / 0.5316
```

Speed result:

```text
256px median step_total: 1.676s
512px median step_total: 3.187s
```

Comparison against the existing 512-only overfit:

```text
512-only 400-step row:
log: outputs/run_logs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_gpt256_8192splats_overfit1_400step_run_20260516_235439.log
wandb: wandb/offline-run-20260516_235441-0x2l83ko
wall: 49:15
final eval PSNR/SSIM: 24.374 / 0.4721
median step_total/backward/raster: 5.496s / 3.361s / 0.443s

256warm300 -> 512fine100 row:
log: outputs/run_logs/dynaworld_overfit_multires_256to512_20260517_112425.log
wandb: wandb/offline-run-20260517_112427-acv8pinq
wall: 16:36
final eval PSNR/SSIM: 24.766 / 0.5316
median step_total/backward/raster: 2.467s / 1.356s / 0.161s
final 512px median step_total/backward/raster: 3.138s / 1.827s / 0.292s
```

Decision:

```text
Keep the schedule mechanism.
The 300-step 256px warmup is a mechanical and relative-quality win over
512-only, but absolute SSIM is still weak. Tune where the 512px switch happens
before promoting the 300-clip 3k config.
```

Next concrete run:

```text
config:
src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_gpt256_8192splats_overfit1_400step_multires_256c200_512f200.jsonc
screen:
dynaworld_overfit_multires_256c200_512f200_20260517_141005
log:
outputs/run_logs/dynaworld_overfit_multires_256c200_512f200_20260517_141005.log

This keeps 400 total steps but switches to 512px at step 200. Promote only if
it improves final eval SSIM or media detail enough to justify the extra 512px
time. Do not launch the 300-clip 3k multires config until this row finishes.
```

Result:

```text
wall: 16:59
wandb: wandb/offline-run-20260517_141007-uvp7ifwr
final train loss/recon: 0.1088 / 0.1087
final eval PSNR/SSIM: 23.291 / 0.3304
final eval L1/MSE: 0.05287 / 0.004687
all-step median step_total/backward/raster: 2.514s / 1.488s / 0.255s
512px median step_total/backward/raster: 2.815s / 1.675s / 0.299s
```

Decision:

```text
Do not promote 256c200 -> 512f200. It is a quality regression despite similar
wall time. The 256c300 -> 512f100 row stays the main-trainer overfit default.
For 300-clip 3k, keep the analogous 80/20 schedule: 256 through step 2399,
512 from step 2400.
```

Dataset-scale run launched:

```text
config: src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_gpt256_8192splats_300clips_3kstep_multires_256to512.jsonc
screen: dynaworld_300clips_3k_multires_256to512_20260517_143003
log: outputs/run_logs/dynaworld_300clips_3k_multires_256to512_20260517_143003.log
wandb: wandb/offline-run-20260517_143005-s230kzhu
cache: 300/300 before launch
early timing: step 30 total/backward/raster 1.812s / 0.971s / 0.163s;
step 40 total/backward/raster 1.974s / 1.047s / 0.187s
latest early status: around step 57, still cache-hot in the 256px stage
```

Next gate: inspect step-100 media/timing. Keep this run as the only active MPS
training job.

Live follow-up:

```text
latest checked active run progress: about 340/3000, still 256px stage
first image: wandb/offline-run-20260517_143005-s230kzhu/files/media/images/Render_GT_vs_Pred_225_b63ffec9e91424e51f34.png
image size: 512x256 side-by-side
visual: prediction is still early broad color fields; wait for video gate before quality call
timing through step310 median step_total/backward/sample_clip/raster:
2.703s / 1.371s / 0.515s / 0.173s
last-5 median step_total/backward/sample_clip/raster:
3.846s / 2.112s / 0.521s / 0.221s
```

Decision:

```text
Do not divert into STAR/Metal renderer work for this active run. Raster is still
small. The immediately actionable speed issue is repeated 64-frame video-window
decode/loading under lazy 300-clip cycling.
```

Implemented next-run speed patch:

```text
src/train/sequence_data.py:
  opt-in uint8 decoded-frame cache for explicit_video_window records

src/train/train_video_token_implicit_dynamic.py:
  resolved data.frame_cache_dir

300-clip 3k config:
  data.frame_cache_dir =
  data/frame_cache/single_video_pretrain_300_youtube_64f_512center_nativefps

test:
  PYTHONPATH=src/train uv run --with pytest pytest tests/test_sequence_data_single_frame.py -q
  5 passed
```

The active screen was launched before this patch, so keep it alive for quality
evidence. Use the frame cache on the next launch/restart only.

Restart decision after first video gate:

```text
old run:
dynaworld_300clips_3k_multires_256to512_20260517_143003
reached about 512/3000, then stopped intentionally

step475 media:
wandb/offline-run-20260517_143005-s230kzhu/files/media/images/Render_GT_vs_Pred_475_73e213b1f4add4243b36.png
wandb/offline-run-20260517_143005-s230kzhu/files/media/videos/Render_Video_475_774263a0b0d535d9417b.mp4
wandb/offline-run-20260517_143005-s230kzhu/files/media/videos/Render_GT_Video_475_c8639a7cffe437904a15.mp4
outputs/run_logs/300clip_multires_step475_render_contact.jpg
outputs/run_logs/300clip_multires_step475_sidebyside_contact.jpg

visual read:
temporally coherent but mostly blurred low-frequency color fields; not worth
preserving an uncheckpointed stale-config run.
```

Frame cache:

```text
path: data/frame_cache/single_video_pretrain_300_youtube_64f_512center_nativefps
files: 300
size: 14G
prewarm wall: 253.9s
```

New active run:

```text
screen:
dynaworld_300clips_3k_multires_256to512_framecache_noprofile_20260517_150143
log:
outputs/run_logs/dynaworld_300clips_3k_multires_256to512_framecache_noprofile_20260517_150143.log
wandb:
wandb/offline-run-20260517_150153-r8fwjqhb

config changes now active:
data.frame_cache_dir set to prewarmed cache
train.profile_timing=false
train.profile_timing_sync=false
wandb name includes framecache-noprofile

early status:
about 107/3000, no traceback, no feature-cache misses, no Timing rows by design
observed early tqdm rate roughly 2.5s/step
```

Next gates:

```text
step250 image
step500 video
step2400 render-size switch to 512px
```

May 17 speed/debug reset:

```text
active run progress at reset: about 1371/3000, no traceback, no feature-cache misses
effective batch size: 1 manifest video window per optimizer step
frames per step: 64
temporal recon microbatch: 4 chunks x 16 frames
observed live throughput near reset: about 0.34 samples/s, about 22 frames/s
```

The active trainer process does not use `torch.utils.data.DataLoader`, workers,
or async prefetch. Lazy manifest sampling loads the RGB frame window and cached
V-JEPA features synchronously on the step path. Current profiling sections split
`sample_clip`, `forward_decode`, `render/rasterize`, `recon_loss`, aggregate
`backward`, and `optimizer_step`; they do not yet separate model backward from
raster backward.

Next-run fixes staged after this reset:

```text
src/train/train_video_token_implicit_dynamic.py:
  logging.wandb_mode support
  data.train_manifest_prefetch support with a bounded CPU sequence prefetch queue

300-clip 3k config:
  logging.wandb_mode = online
  data.train_manifest_prefetch = 2
```

The current W&B run is offline because it was launched with `WANDB_MODE=offline`.
Do not repeat that for future launches. Do not stop the live run only for W&B or
prefetch unless the next quality gate says the run is not worth finishing; this
trainer still has no useful checkpoint/resume path.

May 17 first-class STAR UVT plan update:

```text
done:
  src/train/train.py now routes arch=star_uvt_video_overfit
  src/train/train_star_uvt_video_overfit.py wraps the STAR UVT video-fit harness
  configs added for direct_atomic 32768/200 and tile_pair_suffix+keyseg 8192/20
  online W&B, output JSON, contact sheet, and side-by-side MP4 work
  tiny 64px/4f smoke passed
  targeted pytest passed: 29 passed
```

First-class direct-atomic overfit result:

```text
config:
src/train_configs/star_uvt_highmotion_hlaZbH_64f_256_directatomic_200step.jsonc
result:
star_uvt/results/may17_firstclass_highmotion_hlaZbH_start2_256_64f_32768t_cap256_directatomic_200step.json
video:
star_uvt/results/may17_firstclass_highmotion_hlaZbH_start2_256_64f_32768t_cap256_directatomic_200step.mp4
wandb:
https://wandb.ai/nbardy/dynaworld/runs/jba7kztn
metrics:
PSNR 29.823, SSIM mean/min 0.8572/0.7788, final loss 0.0010415
```

First-class compact-backward probe:

```text
config:
src/train_configs/star_uvt_highmotion_hlaZbH_64f_256_tilepair_suffix_keyseg_20step_profile.jsonc
result:
star_uvt/results/may17_firstclass_highmotion_hlaZbH_start2_256_64f_8192t_cap256_tilepair_suffix_keyseg_20step.json
video:
star_uvt/results/may17_firstclass_highmotion_hlaZbH_start2_256_64f_8192t_cap256_tilepair_suffix_keyseg_20step.mp4
wandb:
https://wandb.ai/nbardy/dynaworld/runs/641gxm9l
metrics:
PSNR 17.599, SSIM mean/min 0.6148/0.5446, final loss 0.017382
```

Immediate next work:

```text
1. keep direct_atomic/index_add as the STAR UVT overfit lane for quality/speed
2. build the next 512px schedule as first-class STAR config, not Gaussian trainer
3. profile or rerun direct_atomic timing in a cooled/quiet state before claiming a new speed number
4. focus deterministic compact work on load growth/backward; current keyseg suffix path is not promoted
5. separately fix Gaussian 512 promotion NaNs before using the 300-clip trainer as a scale baseline
```
