# May 17 Training Lane Note

Date: 2026-05-17
Workspace: `/Users/nicholasbardy/git/gsplats_browser/dynaworld`

## Current Run Goal

The active local lane is the 300-clip single-video pretrain run, not a heldout or
novel-view claim. It is a source-view reconstruction/convergence gate for the
current static/dynamic token trainer:

```text
300 explicit video windows
64 frames per window
center-square crop
cached V-JEPA conditioning features
recon-only loss, V-JEPA prediction-side feature loss disabled
8192 active Gaussians per rendered frame
256px optimization through step 2399, then 512px finishing from step 2400
```

The run should answer whether the cache-hot recon-only path can make useful
progress across many natural 64-frame windows before spending time on slower
V-JEPA-loss or heldout-camera variants.

The immediately relevant config is:

```text
src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_gpt256_8192splats_300clips_3kstep_multires_256to512.jsonc
```

The latest note in `star_uvt/plan.md` records the current active screen/log/W&B
surface:

```text
screen: dynaworld_300clips_3k_multires_256to512_framecache_noprofile_20260517_150143
log: outputs/run_logs/dynaworld_300clips_3k_multires_256to512_framecache_noprofile_20260517_150143.log
	wandb: wandb/offline-run-20260517_150153-r8fwjqhb
```

This live process was launched with `WANDB_MODE=offline`, so it cannot become an
online W&B run without restarting. Do not repeat that launch choice for the next
run.

## Batch And Step Semantics

This config does not use a multi-sample batch size. Treat one optimizer step as
one sampled manifest record:

```text
effective sample batch: 1 video window per step
frames per sample: 64
target render frames per step: 64
	recon backward strategy: microbatch
	temporal microbatch size: 16 frames
	gradient accumulation: none in the config
	train_manifest_sample_mode: cycle
	train_manifest_prefetch: 2 for the next launch
	train steps: 3000
```

The capacity arithmetic is:

```text
24 static decoded tokens + 8 dynamic decoded tokens = 32 active decoded tokens
32 active decoded tokens * 256 gaussians_per_token = 8192 active Gaussians
```

The extra non-camera query tokens are world/register/detail-register tokens and
should not be counted as emitted Gaussian tokens.

## W&B Policy

Use online W&B for training-lane runs by default.

The active run is offline only because it was launched with `WANDB_MODE=offline`.
That should not be used for the next launch. The trainer now supports
`logging.wandb_mode`, and the 300-clip multires config sets it to `online`.

Do not stop or restart the active run only to change W&B mode unless we decide
the run is not worth finishing. If it finishes offline, sync it afterward and
record the resulting W&B URL.

## Speed And Debug Next Steps

The active trainer previously had no `DataLoader`, worker pool, or prefetch path.
It loaded the sampled manifest record synchronously inside each optimizer step.
The next launch uses a small lazy-manifest prefetch queue so CPU video/frame-cache
loading can overlap with the current GPU step. Device transfer stays on the main
thread to avoid MPS context issues.

Current rough throughput on the live offline run was about `0.34 samples/s` and
`22 frames/s` near step 1371, with profiling disabled. Historical profiled rows
showed that the main trainer's coarse `backward` section dominates; the current
profile sections do not yet split model backward from raster backward.

Do not divert into STAR/Metal renderer work for this active run unless fresh
timings show raster dominates. Existing profiled recon-only rows had raster well
below total backward time.

Current speed/debug gates:

```text
step 250: inspect image artifact
step 500: inspect video artifact
step 2400: verify render-size switch to 512px
step 3000: compare final video/contact sheet against the 400-step overfit gate
```

If speed regresses:

1. Check whether prefetch is enabled and whether sample loading is still on the
   critical path.
2. Confirm `train.profile_timing=false` and `profile_timing_sync=false` are still
   intentional for the non-profile run.
3. Use a short separate profile probe only if needed; do not mutate the active
   run config midstream.
4. Keep V-JEPA feature loss disabled for this lane unless the explicit question
   becomes quality under the slower differentiable feature-loss path.
5. For a real speed run, compare compressed video backends against frame-cache
   reads. The current OpenCV+PIL path is slower than cv2-native crop/resize and
   ffmpeg-pipe decode in the one-row benchmark.

If quality is still blurry at the first video gate, the next conservative
debugging moves are to inspect per-window examples and capacity/layout choices,
not to claim a heldout result or rewrite trainer behavior during the live run.

## May 17 Q&A Reset: Trainer, Data, And STAR UVT

Latest checked active run:

```text
screen:
dynaworld_300clips_3k_multires_256to512_framecache_noprofile_20260517_150143
log:
outputs/run_logs/dynaworld_300clips_3k_multires_256to512_framecache_noprofile_20260517_150143.log
status:
about 2089/3000, still before 512px switch at step 2400, cache-hit only
```

Important correction:

```text
sample_clip is the dataloader-ish cost, not whole-step cost.
```

For this trainer, `sample_clip` includes selecting the manifest record, loading
the cached RGB frame window and cached V-JEPA conditioning features, moving the
sequence to the training device, selecting the 64-frame clip, and preparing
`clip_frames` / `clip_times`. It does not include model forward, rasterization,
loss, backward, or optimizer step.

The earlier `0.28s` number was sample/data loading in a profiled row. It was not
`0.28s/sample` for the full training step. The full profiled 256px-ish step was
closer to 1.5-2.2s in the 400-step overfit multires run, and the active 300-clip
run has recently been around 2-3s/step by tqdm. At 64 frames/step, that is
roughly the observed 16-30 frames/s range depending on the window measured.

The active run is not STAR UVT training. It is routed through:

```text
src/train/train.py
arch=precomputed_feature_implicit_camera
-> src/train/train_precomputed_feature_implicit_dynamic.py
-> PrecomputedFeatureImplicitTrainer
-> VideoTokenImplicitTrainer in src/train/train_video_token_implicit_dynamic.py
```

It decodes a standard `GaussianSequence`:

```text
xyz/scales/quats/opacities/rgbs with shape [T, G, C]
cameras per frame from the implicit camera head
```

and renders with the modular renderer path:

```text
render.renderer = fast_mac
fast_mac.rgb_variant = v6_refined
fast_mac.feature_variant = v5_features
max_fast_pairs = 2048
batch_strategy = flatten
active_policy = off
stop_count_mode = adaptive
```

STAR UVT is separate code today:

```text
third_party/fast-mac-gsplat/variants/star_uvt_v0/
third_party/fast-mac-gsplat/variants/star_uvt_prt_v0/
```

The useful STAR UVT harness model emits screen/time tube parameters, not a
normal per-frame 3D Gaussian sequence:

```text
ScreenTimeTubeModel:
center_uv
center_t
velocity_uv
raw_precision / q_uvt
raw_opacity
raw_color
depth0
```

Projective STAR UVT / PRT emits projected-rational world-tube parameters with a
different camera/projection contract again. STAR UVT is not currently a drop-in
`renderer_mode` for the modular trainer. The modular trainer is forkable, but
the clean integration is a new trainer/model contract or adapter, not only a
renderer switch:

```text
Option A: add a STAR UVT trainer arch that reuses data loading, W&B, metrics,
crop parity, precomputed features, and media logging, but swaps model output and
render/loss path to UVT tubes.

Option B: add a UVT adapter that maps current token outputs to tube parameters
instead of GaussianSequence, then routes to the STAR UVT renderer.

Option C: keep the existing GaussianSequence model and only swap the renderer.
This does not capture the STAR UVT scaling thesis because the output is still
per-frame/per-Gaussian, not compact time-tube support.
```

Current trainer inventory from `src/train/train.py`:

```text
tokengs -> train_video_token_implicit_dynamic
tokengs_video_implicit_camera -> train_video_token_implicit_dynamic
tokengs_video_known_camera -> train_video_token_implicit_dynamic
precomputed_feature_implicit_camera -> train_precomputed_feature_implicit_dynamic
ltx_feature_implicit_camera -> train_precomputed_feature_implicit_dynamic
wan_vace_feature_implicit_camera -> train_precomputed_feature_implicit_dynamic
powerfoam_direct -> train_powerfoam_direct
powerfoam_metal -> train_powerfoam_metal
dynamic_powerfoam_metal -> train_dynamic_powerfoam_metal
dynamic_gauge_foam -> train_dynamic_gauge_foam
multicam_precomputed_feature_implicit_camera -> train_multicam_precomputed_feature_implicit_dynamic
multicam_relative_pose_implicit_camera -> train_multicam_relative_pose_implicit_dynamic
```

STAR UVT/PRT harnesses are not registered in that router.

Data loading should be driven toward zero critical-path time for this workload.
The staged next-run code adds:

```text
data.train_manifest_prefetch
single-worker bounded CPU prefetch of lazy manifest sequences
main-thread MPS transfer to avoid device-context surprises
```

This is still not a full `torch.utils.data.DataLoader` path. It is the smallest
safe overlap patch for the current lazy-manifest trainer. A more complete fix is
to make a real dataset/dataloader/pinned-prefetch equivalent where possible, or
switch back to compressed-video streaming if benchmarking proves frame-cache
reads are slower than correct video decode. Either way the target is that
sample/data time is hidden under the previous GPU step.

The backward split question is now partially instrumented for the next profile
probe:

```text
train.profile_backward_split = true
```

When enabled with `train.profile_timing=true`, reconstruction backward is split
around the decoded Gaussian/camera boundary:

```text
backward/raster_loss_to_boundary
backward/model_from_boundary
backward/regularizers
```

This is a diagnostic mode, not the default training path. It should be used on a
short profile probe, not enabled in the current live run.

Validation after adding the profiling option:

```text
PYTHONPATH=src/train python3 -m py_compile \
  src/train/train_video_token_implicit_dynamic.py \
  src/train/train_precomputed_feature_implicit_dynamic.py \
  src/train/sequence_data.py

PYTHONPATH=src/train uv run --with pytest pytest \
  tests/test_temporal_sampling.py \
  tests/test_config_factory_helpers.py \
  tests/test_sequence_data_single_frame.py -q

result: 28 passed
```

Useful pending questions and current answers:

```text
Should we keep optimizing this active trainer or move back to STAR UVT?
Answer: use this active run for dataset/training signal only. It is not proving
the STAR UVT time-scaling thesis. The STAR thesis needs a STAR UVT/PRT trainer
arch or fork that reuses the current data/W&B/media upgrades.

Is the current run's speed evidence enough?
Answer: no. It has profiling disabled and no backward split. It is enough to say
sample/data was nonzero historically and the current step rate is not near the
desired "64 frames near single-frame cost" target.

Is sample/data loading the main bottleneck?
Answer: not proven for the current active run because profiling is off. In prior
profile rows it was real and annoying, but aggregate backward was larger.
Sample time should still be hidden with prefetch because there is no reason for
the GPU step to wait on CPU/frame-cache loading.

Do cached V-JEPA features mean V-JEPA has zero cost?
Answer: cached conditioning features remove V-JEPA input-feature extraction from
the step. They do not remove a differentiable V-JEPA prediction-side feature
loss. This current config has `vjepa_feature_weight=0.0`, so the expensive
prediction-side V-JEPA loss is off.

Does STAR UVT already have the speed isolation result we want?
Answer: yes for harness-level evidence, especially the direct_atomic/index_add
branch. No for full production trainer integration. The full-train script still
uses GaussianSequence + fast_mac, so it does not inherit STAR UVT's compact
time-tube scaling.

What is the implementation target?
Answer: fork/add a STAR UVT trainer arch that reuses the modern data/config/W&B
shell and emits UVT/PRT tube parameters directly. Then benchmark one high-motion
64-frame overfit with online W&B, media, SSIM/PSNR, sample/data timing,
forward_decode, render forward, raster/loss backward, model backward, and
optimizer step.
```

## May 17 Gaussian Run Stop Verdict

The 300-clip Gaussian-sequence multires run is no longer active.

```text
screen:
dynaworld_300clips_3k_multires_256to512_framecache_noprofile_20260517_150143
log:
outputs/run_logs/dynaworld_300clips_3k_multires_256to512_framecache_noprofile_20260517_150143.log
W&B:
wandb/offline-run-20260517_150153-r8fwjqhb
```

It reached the intended render-size switch:

```text
Render size schedule: step 2400 switched 256->512 (8192 Gaussians, fast_mac renderer)
```

After the 512px switch, tqdm step time rose from roughly `2-3s/step` to roughly
`4-8s/step`, and the loss became numerically unsafe around step `2429`:

```text
Loss: nan recon: 0.6157 fov: nan r: nan
```

The process was stopped rather than left to spend MPS time propagating NaNs.
Interpretation: this run is useful evidence about the current Gaussian trainer,
but it is not a STAR UVT proof. The 512px promotion needs a stability fix before
being treated as a scalable many-window baseline. The immediate STAR work should
move to a first-class `star_uvt_video_overfit` route instead of further tuning
this Gaussian-sequence lane.

## May 17 First-Class STAR UVT Runs

New trainer route:

```text
arch:
star_uvt_video_overfit
trainer:
src/train/train_star_uvt_video_overfit.py
router:
src/train/train.py
```

Validation:

```text
tiny 64px/4f smoke: passed
py_compile: passed
targeted pytest: 29 passed
```

Direct-atomic high-motion 64f overfit:

```text
config:
src/train_configs/star_uvt_highmotion_hlaZbH_64f_256_directatomic_200step.jsonc
W&B:
https://wandb.ai/nbardy/dynaworld/runs/jba7kztn
result JSON:
star_uvt/results/may17_firstclass_highmotion_hlaZbH_start2_256_64f_32768t_cap256_directatomic_200step.json
side-by-side video:
star_uvt/results/may17_firstclass_highmotion_hlaZbH_start2_256_64f_32768t_cap256_directatomic_200step.mp4
metrics:
PSNR 29.823, SSIM mean/min 0.8572/0.7788, final loss 0.0010415
```

Compact deterministic probe:

```text
config:
src/train_configs/star_uvt_highmotion_hlaZbH_64f_256_tilepair_suffix_keyseg_20step_profile.jsonc
W&B:
https://wandb.ai/nbardy/dynaworld/runs/641gxm9l
result JSON:
star_uvt/results/may17_firstclass_highmotion_hlaZbH_start2_256_64f_8192t_cap256_tilepair_suffix_keyseg_20step.json
side-by-side video:
star_uvt/results/may17_firstclass_highmotion_hlaZbH_start2_256_64f_8192t_cap256_tilepair_suffix_keyseg_20step.mp4
metrics:
PSNR 17.599, SSIM mean/min 0.6148/0.5446, final loss 0.017382
```

Interpretation: first-class STAR UVT launch/logging/media is working. The
direct-atomic row is still the practical path. The deterministic compact row
reproduces quality but remains too slow to promote.
