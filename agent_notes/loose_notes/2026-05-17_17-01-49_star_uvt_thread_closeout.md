# STAR UVT Thread Closeout

Date: 2026-05-17
Workspace: `/Users/nicholasbardy/git/gsplats_browser/dynaworld`

## Goals From The Thread

Primary goal:

```text
Get STAR UVT running as the real 64-frame overfit path, because the point of
STAR UVT is to make multi-frame training much closer to single-frame cost.
```

Specific goals and questions raised:

```text
1. Stop confusing the active Gaussian/token trainer with STAR UVT.
2. Explain what the 2048/8192 splat counts meant.
3. Turn off differentiable V-JEPA prediction-side feature loss for speed.
4. Clarify V-JEPA feature caching versus backward-through-V-JEPA loss.
5. Check whether data loading was synchronous and whether prefetch was missing.
6. List/understand trainer landscape and whether STAR UVT trainers already exist.
7. Use a curated high-motion YouTube clip for a 64-frame overfit gate.
8. Get a first-class STAR UVT path instead of only loose benchmark scripts.
9. Compare direct atomic STAR UVT against deterministic compact backward.
10. Track the Gaussian 300-clip run, but do not treat it as STAR UVT proof.
11. Record W&B links, artifact paths, tests, and remaining work.
```

## What Was Done

First-class STAR UVT trainer route:

```text
src/train/train.py:
  added arch=star_uvt_video_overfit

src/train/train_star_uvt_video_overfit.py:
  new config-driven wrapper around the STAR UVT video-fit harness
  supports online W&B
  writes result JSON
  exports contact sheet
  exports side-by-side MP4
  logs final metrics/media to W&B
```

Configs added:

```text
src/train_configs/star_uvt_highmotion_hlaZbH_64f_256_directatomic_200step.jsonc
src/train_configs/star_uvt_highmotion_hlaZbH_64f_256_tilepair_suffix_keyseg_20step_profile.jsonc
```

Docs updated:

```text
BASELINES.md
TODO/README.md
star_uvt/critique.md
star_uvt/plan.md
star_uvt/may_17th_workstreams.md
star_uvt/may_17th_training_lane_note.md
agent_notes/loose_notes/2026-05-16_12-21-11_300_youtube_static_dynamic_register_64f.md
agent_notes/key_learnings.md
```

Validation:

```text
tiny 64px/4f STAR UVT wrapper smoke: passed
PYTHONPATH=src/train python3 -m py_compile ...: passed
PYTHONPATH=src/train uv run --with pytest pytest \
  tests/test_config_factory_helpers.py \
  tests/test_sequence_data_single_frame.py \
  tests/test_temporal_sampling.py -q
result: 29 passed
git diff --check: passed
```

## Results

### Direct Atomic STAR UVT

This is the current practical STAR UVT lane.

```text
config:
src/train_configs/star_uvt_highmotion_hlaZbH_64f_256_directatomic_200step.jsonc

video:
data/youtube_curated_spans/raw/hlaZbH_OFBU_seg_003_s00131000_e00138000.mp4

clip:
start 2.0s, natural fps 29.97002997002997, 64 frames, center_square crop

STAR settings:
32768 tubes, 256px render, direct_atomic + index_add, tile_t=1, tile_capacity=256

W&B:
https://wandb.ai/nbardy/dynaworld/runs/jba7kztn

result JSON:
star_uvt/results/may17_firstclass_highmotion_hlaZbH_start2_256_64f_32768t_cap256_directatomic_200step.json

side-by-side video:
star_uvt/results/may17_firstclass_highmotion_hlaZbH_start2_256_64f_32768t_cap256_directatomic_200step.mp4

contact sheet:
star_uvt/results/may17_firstclass_highmotion_hlaZbH_start2_256_64f_32768t_cap256_directatomic_200step.png
```

Metrics:

```text
steps: 200
UVT wall: 183.99s
final loss: 0.0010415
PSNR: 29.823
SSIM mean/min: 0.8572 / 0.7788
render median: 151.16ms
```

Interpretation:

```text
The new first-class route reproduces the prior best quality row almost exactly.
The wall time was slower than the previously saved 111.52s harness row, so do
not claim a new speed best from this run. Claim: first-class route works and
preserves STAR UVT quality/logging/media.
```

### Deterministic Compact Probe

This is the current blocker lane.

```text
config:
src/train_configs/star_uvt_highmotion_hlaZbH_64f_256_tilepair_suffix_keyseg_20step_profile.jsonc

STAR settings:
8192 tubes, 256px render, tile_pair_suffix + key_sort_segmented_metal

W&B:
https://wandb.ai/nbardy/dynaworld/runs/641gxm9l

result JSON:
star_uvt/results/may17_firstclass_highmotion_hlaZbH_start2_256_64f_8192t_cap256_tilepair_suffix_keyseg_20step.json

side-by-side video:
star_uvt/results/may17_firstclass_highmotion_hlaZbH_start2_256_64f_8192t_cap256_tilepair_suffix_keyseg_20step.mp4

contact sheet:
star_uvt/results/may17_firstclass_highmotion_hlaZbH_start2_256_64f_8192t_cap256_tilepair_suffix_keyseg_20step.png
```

Metrics:

```text
steps: 20
UVT wall: 39.54s
final loss: 0.017382
PSNR: 17.599
SSIM mean/min: 0.6148 / 0.5446
render median: 45.03ms
```

Interpretation:

```text
Quality matches the older compact row, but runtime is worse than the saved
18.20s row. Deterministic compact backward remains unpromoted. The next compact
work should target load growth/backward behavior, not another scale run.
```

### Gaussian 300-Clip Multires Run

This run was not STAR UVT. It was the modular precomputed-feature
GaussianSequence trainer:

```text
arch=precomputed_feature_implicit_camera
src/train/train_precomputed_feature_implicit_dynamic.py
VideoTokenImplicitTrainer
fast_mac renderer
8192 active Gaussians per frame
64 frames per step
300 clips
render schedule 256px until step 2400, then 512px
```

Run:

```text
screen:
dynaworld_300clips_3k_multires_256to512_framecache_noprofile_20260517_150143

log:
outputs/run_logs/dynaworld_300clips_3k_multires_256to512_framecache_noprofile_20260517_150143.log

W&B:
wandb/offline-run-20260517_150153-r8fwjqhb
```

Outcome:

```text
reached step 2400 render-size switch
after 512px promotion, speed slowed to about 4-8s/step
NaNs appeared around step 2429:
  Loss: nan recon: 0.6157 fov: nan r: nan
process was stopped
```

Interpretation:

```text
This is a Gaussian-trainer 512px promotion stability warning, not a STAR UVT
result. Do not use it as evidence against STAR UVT. Do not relaunch this exact
lane without a NaN/camera-stability fix or checkpoint/resume plan.
```

## Answers Captured

```text
V-JEPA cached conditioning features are not the same as differentiable V-JEPA
prediction-side loss. Cached conditioning can be cheap; prediction-side
V-JEPA-loss backward was the old 30s+ step killer.

The active 300-clip trainer had V-JEPA feature loss disabled, but it still used
cached V-JEPA conditioning features.

The static/dynamic split is a Gaussian token prior. It does not directly map to
STAR UVT. STAR UVT should use tube-family/init/support policies instead:
static-ish tubes, dynamic-ish tubes, and detail/error tubes.

For STAR UVT, 32768 tubes over 64 frames is about 512 tubes/frame as a capacity
intuition. It is not "2048 splats per token"; it is tube count in the UVT
harness.

The previous active trainer had no real DataLoader path. A small lazy-manifest
prefetch path has been staged for the Gaussian trainer, but STAR UVT currently
uses the harness loader for one clip.
```

## Remaining Work

Highest priority STAR UVT work:

```text
1. Add a first-class 512px STAR UVT multi-resolution config:
   256px coarse direct_atomic -> 512px fine direct_atomic.

2. Add step-level timing hooks inside the STAR UVT trainer path:
   data/load
   render forward
   loss
   backward
   optimizer
   media/export

3. Rerun direct_atomic timing in a quiet/cool local state before making a new
   speed claim; current first-class quality is confirmed, speed best is not.

4. Keep deterministic compact work focused on making
   tile_pair_suffix/key_sort_segmented_metal viable across more than short
   probes. The specific blocker is load-growth/backward speed.

5. Design STAR-native tube-family split instead of carrying over
   static/dynamic/register token layout from the Gaussian trainer.
```

Gaussian trainer follow-up:

```text
1. Fix 512px promotion NaNs before treating 300-clip multires as a baseline.
2. Use `train.profile_backward_split=true` only on short diagnostic runs.
3. Verify `data.train_manifest_prefetch=2` actually hides sample/data time.
4. Do not re-enable differentiable V-JEPA feature loss until the recon-only path
   is stable and the question explicitly requires perceptual feature loss.
```

Benchmark/data follow-up:

```text
1. Keep same-view reconstruction, heldout-view reconstruction, and source/camera
   disjoint benchmark claims separate.
2. Promote any new meaningful result into `BASELINES.md`.
3. Keep direct-atomic dynamic splats and STAR UVT frame-amortized tubes as
   separate measurements; direct atomic alone reduces scratch/backward pain but
   does not create STAR's time-tube scaling.
```

## Current Handoff State

```text
No Dynaworld training process is active.
First-class STAR UVT route works.
Direct atomic is the practical 64f overfit path.
Deterministic compact backward remains the main STAR blocker.
Gaussian 512px promotion is unstable after step 2400 and should not be
continued blindly.
```
