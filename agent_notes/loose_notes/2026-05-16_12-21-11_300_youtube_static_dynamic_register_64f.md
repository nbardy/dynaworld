# 300 YouTube 64f Static/Dynamic Register Lane

User goal: cook overnight on a fast 300-clip Dynaworld run using 64-frame natural-FPS YouTube windows, keep 512 center-crop/V-JEPA contract, add dynamic/static split, register tokens, and implicit camera learning. Also decide whether STAR UVT/projective STAR should be promoted now.

## Artifacts Added

- `src/dataset_configs/single_video_pretrain_300_youtube_64f_512_manifest.jsonc`
- `src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_vjepa_loss.jsonc`
- `src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_384render_static_dynamic_register_vjepa_loss_speed.jsonc`
- `src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_256render_static_dynamic_register_vjepa_loss_speed.jsonc`
- `src/train_scripts/train_single_video_pretrain_300_64f.sh`

The train configs use:

- `arch=precomputed_feature_implicit_camera`
- 64 source frames at natural FPS
- 512 center-square source crop for the loaded video tensor
- V-JEPA conditioning crop 256 and V-JEPA feature-loss crop 256 for local MPS memory
- `camera_refine_with_decode_time=true`
- token layout:
  - `tokens=40`
  - `world_tokens=4`
  - `register_tokens=2`
  - static full decoded capacity 24
  - dynamic full decoded capacity 8
  - active decoded tokens 32
  - `gaussians_per_token=16`, so 512 active Gaussians

## Manifest

Built with:

```bash
./src/train_scripts/train_single_video_pretrain_300_64f.sh build
./src/train_scripts/train_single_video_pretrain_300_64f.sh audit
./src/train_scripts/train_single_video_pretrain_300_64f.sh load-check
```

Result:

- `data/single_video_pretrain/dynaworld_single_video_pretrain_300_youtube_64f_512_v0/train_manifest.jsonl`
- 300 train windows
- source counts:
  - `youtube_scene_distinct_raw_64f_512`: 140
  - `youtube_curated_spans_raw_64f_512`: 132
  - `youtube_scene_distinct_segments_64f_512`: 28
- bad record count 0
- missing video count 0
- FPS range 23.976 to 30.0
- window duration range 2.133s to 2.669s
- load-check loaded 8 real windows as 64 frames at 512, `center_square`

Sidecar data stats for the full all-YouTube pool: 119 source files, 103 manifest videos, about 7.22s average source duration over all files, 7.70s over manifest videos, 615 raw possible 64f windows and 588 selected train windows after heldout/too-short filtering.

## Probe Results

Command:

```bash
PROBE_STEPS=1 ./src/train_scripts/train_single_video_pretrain_300_64f.sh bench
```

All probes used W&B disabled because they were mechanical speed checks. They exercised the real lazy manifest, precomputed V-JEPA cache path, V-JEPA feature loss, static/dynamic token layout, register tokens, implicit camera path, and fast-mac renderer.

Results:

- 256 render: 58.36s/step, log `outputs/run_logs/local_mac_single_video_pretrain_300_youtube_64f_256render_static_dynamic_register_vjepa_loss_speed_probe1step_20260516_121712.log`
- 384 render: 52.70s/step, log `outputs/run_logs/local_mac_single_video_pretrain_300_youtube_64f_384render_static_dynamic_register_vjepa_loss_speed_probe1step_20260516_121840.log`
- 512 render: 52.68s/step, log `outputs/run_logs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_vjepa_loss_probe1step_20260516_121947.log`

The first 256 probe had one V-JEPA feature-cache miss for the first 300-manifest sample, then subsequent probes hit the same cache. The tqdm step time still shows that output resolution was not the dominant knob: 512 was effectively tied with 384 and faster than the 256 first probe.

Projected true 512 300-step time at cached-step speed is about 4.4h before any first-epoch cache misses, W&B overhead, or variance.

## Overnight Run

Direct detached `nohup` launches were unreliable for process supervision: one direct module launch emitted only `uv` parent-project warnings, and an inline W&B launch reached W&B init but left no live process. A foreground one-step `WANDB_MODE=offline` run completed successfully at 55.96s/step, proving the config and W&B-offline path were sound.

The long run was then launched in a detached screen session:

```bash
screen -dmS dynaworld_300_64f_512_20260516_122709 zsh -lc "cd /Users/nicholasbardy/git/gsplats_browser/dynaworld && WANDB_MODE=offline TRAIN_CONFIG=src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_vjepa_loss.jsonc ./src/train_scripts/train_single_video_pretrain_300_64f.sh run >> outputs/run_logs/dynaworld_300_64f_512_20260516_122709_screen.log 2>&1"
```

Live process at handoff:

- screen session: `dynaworld_300_64f_512_20260516_122709`
- Python PID at check: `52188`
- primary run log: `outputs/run_logs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_vjepa_loss_run_20260516_122710.log`
- screen log: `outputs/run_logs/dynaworld_300_64f_512_20260516_122709_screen.log`
- W&B offline run: `wandb/offline-run-20260516_122712-ltgc3mts`
- progress at check: step 2/300 running
- cached first step: 56.57s/it
- first-epoch cache misses pushed ETA to about 5.6h at step 2

Added `./src/train_scripts/train_single_video_pretrain_300_64f.sh status` to parse the active run log instead of manually reading tqdm carriage returns. At the next check it reported:

- screen session still detached/alive
- last step: 3/300
- last metrics: loss 0.5172, recon 0.5172, fov 59.88, radius 2.99
- last rate: 79.05s/step, ETA `6:31:16`
- cache hits logged: 4
- cache misses logged: 3
- feature cache coverage: 3/300 files

Decision: let the true-512 run continue. Do not start competing MPS probes while it is live. The early overhead is mostly first-epoch V-JEPA feature-cache baking, not 512 raster resolution.

## STAR UVT Decision

Do not promote STAR UVT/projective STAR into the overnight mainline yet. The current projective-rational STAR path is still an implementation plan requiring new projection sequence objects, a public render call, and projection/dense/Metal/backward/timing gates. Existing `projective_first_order` is a first-order approximation in the isolated harness, not the planned production PRT compiler.

Immediate path: benchmark current `fast_mac` first. Keep STAR/projective STAR as a side benchmark until it has same-source production baseline, same resolution/frame split, held-out-camera metrics, production-scale backward timing/memory, and visual proof.

## Follow-up Cache Fix

The first detached true-512 run was stopped after it showed the real blocker: lazy V-JEPA feature-cache baking during training. It reached only a few steps before cache misses pushed the live rate from about 56s/step to 175s/step, and a process sample showed the trainer waiting on MPS-to-CPU copy from the V-JEPA feature extractor path. The bottleneck was not `fast_mac` raster resolution.

Added `prebake` and `cache-status` actions to `src/train_scripts/train_single_video_pretrain_300_64f.sh`.

Smoke:

```bash
PREBAKE_LIMIT=12 ./src/train_scripts/train_single_video_pretrain_300_64f.sh prebake
```

Result: skipped 9 existing records, baked 3 missing records, ended with 12/300 feature cache files in 40.45s.

Full pre-bake:

```bash
screen -dmS dynaworld_300_64f_512_prebake_20260516_124632 bash -lc "cd /Users/nicholasbardy/git/gsplats_browser/dynaworld && ./src/train_scripts/train_single_video_pretrain_300_64f.sh prebake > 'outputs/run_logs/dynaworld_300_64f_512_prebake_20260516_124632_screen.log' 2>&1"
```

Status command:

```bash
./src/train_scripts/train_single_video_pretrain_300_64f.sh cache-status
```

At 12:55 local time the cache was 45/300 files and the pre-bake screen was still alive. Do not restart training until cache coverage is 300/300.

Added a `REQUIRE_FULL_CACHE=1` guard to `run`; it refused to start at 60/300 cache files, as intended.

Added `wait-cache-run` and launched:

```bash
screen -dmS dynaworld_300_64f_512_wait_cache_run_20260516_125814 bash -lc "cd /Users/nicholasbardy/git/gsplats_browser/dynaworld && WANDB_MODE=offline CACHE_POLL_SECONDS=60 ./src/train_scripts/train_single_video_pretrain_300_64f.sh wait-cache-run > 'outputs/run_logs/dynaworld_300_64f_512_wait_cache_run_20260516_125814_screen.log' 2>&1"
```

It polls cache coverage every 60s and will automatically launch `WANDB_MODE=offline REQUIRE_FULL_CACHE=1 ./src/train_scripts/train_single_video_pretrain_300_64f.sh run` when the feature cache reaches 300/300. At launch time it saw 71/300 cache files. A 30-minute thread heartbeat is also active, but should not create a duplicate while the wait-cache-run screen is alive.

Replaced the waiter with a clearer version that launches a dedicated `dynaworld_300_64f_512_train_*` screen instead of running training inside the wait-cache-run screen. The stale older waiter process was killed, leaving only:

```bash
dynaworld_300_64f_512_wait_cache_run_20260516_131229
```

Pre-bake completed successfully:

- final cache coverage: 300/300
- baked this pass: 288
- skipped existing: 12
- elapsed: 2701.54s

The waiter launched the guarded cache-hot baseline:

- screen: `dynaworld_300_64f_512_train_20260516_133140`
- run log: `outputs/run_logs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_vjepa_loss_run_20260516_133141.log`
- W&B offline: `wandb/offline-run-20260516_133144-v856t3eg`

Early cache-hot evidence:

- step 7/300 at status check
- cache hits logged: 11
- cache misses logged: 0
- rate: about 45.0s/step, ETA about 3h40m
- first profile timing at step 5:
  - `step_total=44.8957s`
  - `backward=42.5630s`
  - `forward_decode=0.5245s`
  - `vjepa_feature_loss=0.6547s`
  - `sample_clip=0.7602s`
  - `render/rasterize=0.1349s`

Interpretation: after cache-hot launch, current bottleneck is backward, not RGB raster forward. Do not promote STAR/projective STAR based on this run; current evidence points away from forward renderer as the main limiter. The stride-8 config may still be useful if V-JEPA-loss backward is hidden inside the aggregate `backward` timing, but the baseline should reach at least first media unless it fails or slows badly.

Step-10 confirmation:

- status: step 11/300
- cache hits logged: 15
- cache misses logged: 0
- rate: 44.83s/step, ETA about 3h36m
- timing step 10:
  - `step_total=44.4908s`
  - `backward=42.7810s`
  - `forward_decode=0.4184s`
  - `vjepa_feature_loss=0.6362s`
  - `sample_clip=0.2896s`
  - `render/rasterize=0.1094s`

Decision at step 10: keep baseline running to first media at step 50. The speed is good enough for overnight, and the bottleneck still does not justify interrupting for STAR/Metal. Run the stride-8 comparison only if the baseline fails, slows badly, or first media suggests the loss tradeoff needs A/B evidence.

Step-50 media gate:

- status after inspection: step 53/300
- cache hits logged: 57
- cache misses logged: 0
- rate: 45.70s/step, ETA about 3h08m
- timing step 50:
  - `step_total=46.3601s`
  - `backward=43.8854s`
  - `forward_decode=0.7146s`
  - `vjepa_feature_loss=0.7046s`
  - `sample_clip=0.5104s`
  - `render/rasterize=0.2437s`
- media:
  - `wandb/offline-run-20260516_133144-v856t3eg/files/media/images/Render_GT_vs_Pred_40_9a6eee2206ed0cf59ee2.png`

Visual read: valid image artifact. GT left is a landscape frame; prediction right is still coarse blocky color/shape fields at step 50. That is early/expected enough to keep running, but the next useful quality gate is step 100 video rather than a speed ablation. Continue baseline unless it stalls/fails.

Step-100 media gate:

- status at inspection: step 106/300
- cache hits logged: 111
- cache misses logged: 0
- rate: 44.97s/step, ETA about 2h25m
- timing step 105:
  - `step_total=44.0172s`
  - `backward=41.8632s`
  - `forward_decode=0.6363s`
  - `vjepa_feature_loss=0.6403s`
  - `sample_clip=0.4106s`
  - `render/rasterize=0.1948s`
- media:
  - `wandb/offline-run-20260516_133144-v856t3eg/files/media/images/Render_GT_vs_Pred_90_d86e5f8da53317cb331b.png`
  - `wandb/offline-run-20260516_133144-v856t3eg/files/media/videos/GT_Video_90_7f2676d2642a25ebd134.mp4`
  - `wandb/offline-run-20260516_133144-v856t3eg/files/media/videos/Render_GT_Video_90_7c8f316be962c7be13d2.mp4`
  - `wandb/offline-run-20260516_133144-v856t3eg/files/media/videos/Render_Video_90_3a17e9c967516c69dfd2.mp4`
- video metadata: 63 frames, 24fps; GT/render videos are 512x512, side-by-side video is 1024x512.
- contact sheet generated for inspection: `outputs/run_logs/render_gt_video_step100_contact_sheet.jpg`

Visual read: step-100 media is valid and temporally coherent enough to continue, but predictions remain very blocky/coarse with weak scene detail. The current issue is quality/capacity/backward-efficiency, not forward raster speed. Keep the baseline running to completion. Do not launch stride-8 while this run is alive; stride-8 is still only a follow-up if final quality/throughput tradeoff requires it.

The baseline and comparison configs now have `train.profile_timing=true`, synced timing, and `profile_timing_log_every=5` so the cache-hot run can identify whether the remaining cost sits in model decode, RGB render/backward, or V-JEPA feature loss.

Prepared comparison configs:

- `src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_vjepa_loss_stride8_speed.jsonc`: keeps 512 render and 512 active Gaussians but changes V-JEPA feature-loss temporal stride from 4 to 8.
- `src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_vjepa_loss_detail1_stability.jsonc`: keeps 512 render and stride 4 but sets `active_detail_level=1`, reducing active decoded tokens from 32 to 26 and active Gaussians from 512 to 416.
- `src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_vjepa_loss_2048splats_capacity.jsonc`: keeps the same 40-token static/dynamic/register layout and V-JEPA stride 4, but raises `gaussians_per_token` from 16 to 64. Resolve check reports 32 active decoded tokens and 2048 active Gaussians. Use this as the first follow-up if the final 512-splat run stays blocky/coarse.

The current V-JEPA contract is 64 frames sampled at source/native FPS, 512 center-square loaded target/render frames, and 256 crop inside V-JEPA conditioning/loss. V-JEPA receives frames only; FPS is implicit in which frames are sampled, not passed as metadata.

Static STAR promotion gate update:

- Patched `third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/benchmarks/deterministic_compact_promotion_gate.py` so repeatability parsing catches the existing `final_state_0_1.max_abs` evidence in multicam repeatability JSONs.
- The gate now also checks that a supplied quality JSON's `uvt_reduction_mode` and `uvt_sample_emission_mode` match the selected policy. This prevents accidentally using repeatability evidence from a different compact backward path.
- Passing gate:

```bash
uv run python variants/star_uvt_v0/research_project/benchmarks/deterministic_compact_promotion_gate.py --require-deterministic --require-compact --require-promotion-contract --quality-json variants/star_uvt_v0/research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_tilepair_zero_prune_repeatability_600steps.json --max-repeatability-delta 0.0 --out-json variants/star_uvt_v0/research_project/benchmarks/results/deterministic_compact_promotion_gate_zero_prune_repeatability_20260516.json
```

Result: `pass=true`, `max_repeatability_delta=0.0`, `reduction_mode=key_sort_scan_metal`, `sample_emission_mode=tile_pair`.

Negative control:

```bash
uv run python variants/star_uvt_v0/research_project/benchmarks/deterministic_compact_promotion_gate.py --require-deterministic --require-compact --require-promotion-contract --quality-json variants/star_uvt_v0/research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_tilepair_reduced_repeatability_20steps.json --max-repeatability-delta 0.0
```

Result: rejected because the quality JSON is `index_add + tile_pair_reduced`, not the promoted `key_sort_scan_metal + tile_pair` policy.

Step-150 image gate:

- status at inspection: step 151/300
- cache hits logged: 156
- cache misses logged: 0
- rate: 44.99s/step, ETA about 1h52m
- timing step 150:
  - `step_total=45.4266s`
  - `backward=43.0175s`
  - `forward_decode=0.7043s`
  - `vjepa_feature_loss=0.6656s`
  - `sample_clip=0.5500s`
  - `render/rasterize=0.1700s`
- media:
  - `wandb/offline-run-20260516_133144-v856t3eg/files/media/images/Render_GT_vs_Pred_140_d7e722822699298b92fe.png`

Visual read: prediction is still very coarse/blocky. The GT is a surfer/water clip; the prediction has large flat color regions and a few object-like blobs but weak spatial detail. Continue the baseline to the step-200 video gate, but current evidence makes the 2048-splat capacity config a stronger next comparison than stride8 or STAR renderer promotion if the final run remains this coarse.

Step-200 video gate:

- status at inspection: step 201/300
- later live status after the step-200 gate: step 207/300
- cache hits logged at step-207 status: 213
- cache misses logged: 0
- step-201 displayed rate included W&B media-encoding overhead: 51.63s/step, ETA about 1h25m
- later step-207 displayed rate recovered to 42.53s/step, ETA about 1h06m
- timing step 200:
  - `step_total=45.5989s`
  - `backward=43.3017s`
  - `forward_decode=0.5289s`
  - `vjepa_feature_loss=0.6777s`
  - `sample_clip=0.5810s`
  - `render/rasterize=0.1882s`
- timing step 205:
  - `step_total=47.2799s`
  - `backward=44.1313s`
  - `forward_decode=0.8988s`
  - `vjepa_feature_loss=0.7760s`
  - `sample_clip=0.8228s`
  - `render/rasterize=0.3365s`
- media:
  - `wandb/offline-run-20260516_133144-v856t3eg/files/media/images/Render_GT_vs_Pred_190_ca211d95c181a6998ab6.png`
  - `wandb/offline-run-20260516_133144-v856t3eg/files/media/videos/Render_GT_Video_190_2080f21c2160678b5a50.mp4`
  - `wandb/offline-run-20260516_133144-v856t3eg/files/media/videos/Render_Video_190_c671dee9dd979d47ef39.mp4`
- generated inspection contact sheets:
  - `outputs/run_logs/render_gt_video_step200_contact_sheet.jpg`
  - `outputs/run_logs/render_video_step200_contact_sheet.jpg`
- video metadata: `Render_GT_Video_190` is 1024x512, 24fps, 2.625s, 63 frames; `Render_Video_190` is 512x512, 24fps, 2.625s, 63 frames.

Visual read: step-200 video is temporally coherent, but the prediction is still blocky and sparse compared with the GT forest/scene content. This keeps pointing to capacity/quality/backward behavior as the main issue. The baseline should finish, then the first queued local comparison should be the 2048-splat capacity config. Do not launch stride8 or detail1 while the baseline is still alive.

Overnight queue update:

- waiter screen: `dynaworld_capacity_after_baseline_20260516_160827`
- waiter script: `outputs/run_logs/dynaworld_capacity_after_baseline_20260516_160827_waiter.sh`
- behavior: polls baseline status every 120s; launches `src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_vjepa_loss_2048splats_capacity.jsonc` only after the baseline reaches its configured total step count; exits without launching if the baseline screen dies early.
- status script update: `train_single_video_pretrain_300_64f.sh status` now searches for the run log by the selected `TRAIN_CONFIG` basename, so capacity-run status will not accidentally read the baseline log.

Natural-FPS timing fix for follow-up runs:

- `src/train/pipeline/render.py::prepare_clip` now uses `SequenceData.frame_times[clip_indices]` instead of reconstructing normalized times from local frame indices.
- Why: the manifest already samples 64-frame windows at source/native FPS and stores normalized timestamp metadata. The precomputed-feature path should preserve that timing signal into model input/decode times rather than treating every clip as only uniformly indexed positions.
- Validation: `uv run --with pytest pytest tests/test_pipeline_helpers.py tests/test_sequence_data_single_frame.py tests/test_vjepa_feature_loss.py` passed, 11 tests.

Step-225 live status:

- status: step 226/300
- cache hits logged: 232
- cache misses logged: 0
- rate: 36.40s/step, ETA about 44m53s
- timing step 225:
  - `step_total=36.1451s`
  - `backward=34.1189s`
  - `forward_decode=0.5502s`
  - `vjepa_feature_loss=0.6014s`
  - `sample_clip=0.5160s`
  - `render/rasterize=0.1251s`

The bottleneck remains aggregate backward, not raster forward. STAR/projective STAR should stay parked for this specific baseline unless a more granular backward profile shows fast-mac raster backward dominates the aggregate `.backward()` bucket.

Timing comparison helper:

- Added `timing-summary` action to `src/train_scripts/train_single_video_pretrain_300_64f.sh`.
- It selects the latest run log by `TRAIN_CONFIG` basename, then reports median/mean/min/max/last timing terms and the latest fraction of step total.
- Baseline timing summary at last profile step 230:
  - timing records: 46
  - median `step_total=44.7993s`
  - median `backward=42.5664s`
  - median `render/rasterize=0.1710s`
  - median `vjepa_feature_loss=0.6556s`
  - latest `step_total=38.4170s`
  - latest fractions: `backward=0.9405`, `render/rasterize=0.0053`, `vjepa_feature_loss=0.0168`, `forward_decode=0.0151`
- Command:

```bash
TRAIN_CONFIG=src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_vjepa_loss.jsonc ./src/train_scripts/train_single_video_pretrain_300_64f.sh timing-summary
```

Use the same action on the 2048-splat capacity config after it has logged several timing records.

Register/static-dynamic wiring audit:

- `DynamicVideoTokenGSImplicitCamera.decoded_static_query_tokens` and `decoded_dynamic_query_tokens` use the token layout gather helpers when `model.token_layout` is set.
- That means world/register/detail-register tokens remain in the query bank for attention, but are skipped for Gaussian decode.
- Existing tests in `tests/test_config_factory_helpers.py` cover this directly:
  - `test_token_layout_keeps_world_register_queries_but_decodes_active_core_only`
  - `test_token_layout_active_detail_level_adds_decoded_detail_tokens`
- Expanded local validation:

```bash
uv run --with pytest pytest tests/test_pipeline_helpers.py tests/test_sequence_data_single_frame.py tests/test_vjepa_feature_loss.py tests/test_config_factory_helpers.py
```

Result: 24 passed.

Live status after this audit:

- baseline: step 233/300
- cache hits logged: 239
- cache misses logged: 0
- last displayed rate: 37.34s/step, ETA about 41m41s
- latest media is still the step-200 gate; no step-250 media yet.

Prepared V-JEPA backward ablation:

- Config: `src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_profile.jsonc`
- Purpose: isolate hidden V-JEPA-gradient cost from the aggregate backward bucket.
- It keeps the same 300-window manifest, 512 render size, 512 active Gaussians, static/dynamic/register token layout, precomputed V-JEPA conditioning, and implicit-camera path, but sets `losses.vjepa_feature_weight=0.0`.
- Resolve check:
  - active decoded tokens: 32
  - approx decoded Gaussians: 512
  - V-JEPA crop metadata remains 256 for conditioning/cache compatibility
  - camera refine with decode time remains true
- Do not launch this while the baseline or 2048-capacity run is alive. Use it after the capacity comparison if timing still leaves uncertainty about whether the V-JEPA feature loss is causing the hidden backward cost.

Live status after preparing the ablation:

- baseline: step 236/300
- cache hits logged: 242
- cache misses logged: 0
- last displayed rate: 36.39s/step, ETA about 38m49s
- timing step 235:
  - `step_total=36.0291s`
  - `backward=34.3945s`
  - `forward_decode=0.3600s`
  - `vjepa_feature_loss=0.6073s`
  - `sample_clip=0.3090s`
  - `render/rasterize=0.1279s`

Step-250 status and step-240 image gate:

- baseline status: step 250/300
- cache hits logged: 256
- cache misses logged: 0
- last displayed rate: 37.82s/step, ETA about 31m30s
- timing step 250:
  - `step_total=38.5483s`
  - `backward=36.6743s`
  - `forward_decode=0.4189s`
  - `vjepa_feature_loss=0.5992s`
  - `sample_clip=0.4918s`
  - `render/rasterize=0.1191s`
- media:
  - `wandb/offline-run-20260516_133144-v856t3eg/files/media/images/Render_GT_vs_Pred_240_94b2deaf31cd13dc2189.png`

Visual read: GT is a basketball-arena huddle frame. Prediction is still mostly white background plus sparse, blocky color blobs. This is a strong capacity/representation failure signal for the 512-splat baseline, not a forward-renderer speed issue. Keep the baseline to final and let the queued 2048-splat capacity run start afterward.

Final 512-splat baseline:

- run log: `outputs/run_logs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_vjepa_loss_run_20260516_133141.log`
- W&B offline run: `wandb/offline-run-20260516_133144-v856t3eg`
- status: completed 300/300
- cache: 306 visible feature-cache hits, 0 misses, 300/300 feature files
- final displayed rate: 42.81s/step, elapsed 12843s
- final timing step 300:
  - `step_total=38.6522s`
  - `backward=36.6975s`
  - `forward_decode=0.4260s`
  - `vjepa_feature_loss=0.6095s`
  - `sample_clip=0.5101s`
  - `render/rasterize=0.1438s`
  - `render_view_total=0.1441s`
- full baseline timing summary:
  - timing records: 60
  - median `step_total=44.3755s`
  - mean `step_total=42.5192s`
  - median `backward=42.1598s`
  - mean `backward=40.3130s`
  - median `render/rasterize=0.1547s`
  - median `vjepa_feature_loss=0.6503s`
  - latest fractions at step 300: `backward=0.9494`, `forward_decode=0.0110`, `render/rasterize=0.0037`, `vjepa_feature_loss=0.0158`
- final media:
  - image: `wandb/offline-run-20260516_133144-v856t3eg/files/media/images/Render_GT_vs_Pred_290_3916f64d10a73b94e3e1.png`
  - side-by-side video: `wandb/offline-run-20260516_133144-v856t3eg/files/media/videos/Render_GT_Video_290_43454003d5ffbed5d8ab.mp4`
  - render video: `wandb/offline-run-20260516_133144-v856t3eg/files/media/videos/Render_Video_290_de03aa102642e47dd04b.mp4`
  - contact sheets:
    - `outputs/run_logs/render_gt_video_step300_contact_sheet.jpg`
    - `outputs/run_logs/render_video_step300_contact_sheet.jpg`
- visual read: final baseline remains mostly white background plus sparse blocky blobs. GT final image is a biking/street scene; the prediction failed to reconstruct scene detail. The final video is temporally coherent but under-capacity/blocky. The baseline is mechanically valid but not good enough.

Status script fix:

- `src/train_scripts/train_single_video_pretrain_300_64f.sh status` now selects run logs by the active `TRAIN_CONFIG` basename and includes `all_dynaworld_300_64f_512_screen_sessions`.
- This prevents the baseline status command from accidentally reporting the active `2048` capacity screen as a baseline training screen.

2048-splat capacity run:

- config: `src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_vjepa_loss_2048splats_capacity.jsonc`
- screen: `35643.dynaworld_300_64f_512_2048_train_20260516_170603`
- run log: `outputs/run_logs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_vjepa_loss_2048splats_capacity_run_20260516_170604.log`
- W&B offline run: `wandb/offline-run-20260516_170606-doer3v9t`
- early status at step 11:
  - cache hits logged: 15
  - cache misses logged: 0
  - rate: 37.57s/step, ETA about 3h00m59s
  - last loss: 0.3922, recon: 0.3921
- timing step 10:
  - `step_total=36.8690s`
  - `backward=35.3700s`
  - `forward_decode=0.2999s`
  - `vjepa_feature_loss=0.6090s`
  - `sample_clip=0.2914s`
  - `render/rasterize=0.0802s`
  - `render_view_total=0.0804s`
- early timing summary:
  - timing records: 2
  - mean/median `step_total=37.7039s`
  - mean/median `backward=36.0225s`
  - mean/median `render/rasterize=0.0955s`
  - mean/median `vjepa_feature_loss=0.6095s`
  - latest fractions: `backward=0.9593`, `forward_decode=0.0081`, `render/rasterize=0.0022`, `vjepa_feature_loss=0.0165`

Early interpretation: 2048 active Gaussians is not materially slower than the 512-splat baseline on this fast_mac path. The run is still dominated by aggregate backward, not raster forward. Let it continue to the first image/video gate before deciding whether capacity improves the severe blockiness. Do not launch recon-only, stride8, or lower-resolution variants while this screen is alive.

2048-splat step-53 checkpoint and first image:

- status: step 53/300
- cache hits logged: 57
- cache misses logged: 0
- rate: 39.03s/step, ETA about 2h40m41s
- latest loss: 0.2200, recon: 0.2171
- timing step 50:
  - `step_total=38.4715s`
  - `backward=36.9448s`
  - `forward_decode=0.2933s`
  - `vjepa_feature_loss=0.6070s`
  - `sample_clip=0.2590s`
  - `render/rasterize=0.1424s`
  - `render_view_total=0.1426s`
- timing summary through step 50:
  - timing records: 10
  - median `step_total=37.3486s`
  - mean `step_total=37.8693s`
  - median `backward=35.5040s`
  - mean `backward=36.1257s`
  - median `render/rasterize=0.1511s`
  - median `vjepa_feature_loss=0.6221s`
  - latest fractions: `backward=0.9603`, `forward_decode=0.0076`, `render/rasterize=0.0037`, `vjepa_feature_loss=0.0158`
- first capacity image:
  - `wandb/offline-run-20260516_170606-doer3v9t/files/media/images/Render_GT_vs_Pred_40_1c30acb365601476520a.png`
- apples-to-apples baseline image:
  - `wandb/offline-run-20260516_133144-v856t3eg/files/media/images/Render_GT_vs_Pred_40_9a6eee2206ed0cf59ee2.png`

Visual read: the 2048-splat step-40 prediction is slightly more filled and has less blank-white area than the 512-splat baseline at the same scene/step, but it is still dominated by large low-frequency rectangular blocks. This is an improvement over severe blanking, not yet a good reconstruction. Keep the 2048 run alive to the step-100 video gate before committing to capacity as the main lane. The next checkpoint should include both an image around step 90 and validation video around step 100.

2048-splat step-100 media gate:

- status: step 100/300
- cache hits logged: 105
- cache misses logged: 0
- latest displayed rate: 46.29s/step, ETA about 2h34m18s
- latest loss: 0.3041, recon: 0.3039
- timing step 100:
  - `step_total=41.2131s`
  - `backward=39.4542s`
  - `forward_decode=0.3944s`
  - `vjepa_feature_loss=0.6278s`
  - `sample_clip=0.3092s`
  - `render/rasterize=0.1874s`
  - `render_view_total=0.1878s`
- timing summary through step 100:
  - timing records: 20
  - median `step_total=37.3486s`
  - mean `step_total=37.5316s`
  - median `backward=35.5040s`
  - mean `backward=35.8345s`
  - median `render/rasterize=0.1573s`
  - median `vjepa_feature_loss=0.6217s`
  - latest fractions: `backward=0.9573`, `forward_decode=0.0096`, `render/rasterize=0.0045`, `vjepa_feature_loss=0.0152`
- media:
  - image: `wandb/offline-run-20260516_170606-doer3v9t/files/media/images/Render_GT_vs_Pred_90_741401e775dbabc399a4.png`
  - side-by-side video: `wandb/offline-run-20260516_170606-doer3v9t/files/media/videos/Render_GT_Video_90_f18188015536bdcf1ff6.mp4`
  - render video: `wandb/offline-run-20260516_170606-doer3v9t/files/media/videos/Render_Video_90_5d98cc53191f0e3d92ab.mp4`
  - contact sheets:
    - `outputs/run_logs/capacity_2048_render_gt_video_step100_contact_sheet.jpg`
    - `outputs/run_logs/capacity_2048_render_video_step100_contact_sheet.jpg`
- video metadata:
  - side-by-side: 1024x512, 24fps, 63 frames, 2.625s
  - render-only: 512x512, 24fps, 63 frames, 2.625s

Visual read: step-90/100 2048-splat media is temporally coherent and less blank than the 512-splat baseline, but still collapses the scene into broad smooth rectangular color fields. The image shows the city/arcade GT reduced to a blurry green-gray block; the video contact sheet shows the forest GT reduced to large soft blocks. This says "more splats per token helped fill, but not enough independent scene tokens/detail." The current run should still finish to 300 for a fair final comparison, but the next queued run should increase active decoded token count, not only gaussians per token.

Queued token-capacity follow-up:

- config: `src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_vjepa_loss_8192splats_token_capacity.jsonc`
- resolve result:
  - active decoded tokens: 128
  - static full capacity: 96
  - dynamic full capacity: 32
  - total non-camera query tokens: 136
  - gaussians per token: 64
  - approx decoded Gaussians: 8192
  - render size: 512
  - model size: 512
  - train frames: 64
  - V-JEPA crop and feature-loss crop: 256
  - V-JEPA feature temporal stride: 4
- cache status: 300/300 feature files, 1.0 coverage, so no feature rebake expected.
- waiter script: `outputs/run_logs/dynaworld_8192_after_2048_20260516_181323_waiter.sh`
- waiter screen: `56189.dynaworld_8192_after_2048_20260516_181323`
- waiter log: `outputs/run_logs/dynaworld_8192_after_2048_20260516_181323_waiter.log`
- current waiter evidence: it saw the 2048 run at 107/300 with one live training screen and is polling every 120s.

Next decision: do not stop the 2048 run early. Let it finish its 300-step comparison, then the waiter launches the 8192-splat token-capacity run. If 8192 OOMs or slows badly, the next fallback is a smaller token-capacity run, not lower render resolution, because render forward remains tiny. If 8192 is also blocky, run the prepared recon-only profile to isolate hidden V-JEPA/backward attribution before touching STAR/Metal renderer work.

2048-splat mid-run step-155 / image-140 checkpoint:

- status: step 155/300
- cache hits logged: 160
- cache misses logged: 0
- latest displayed rate: 39.62s/step, ETA about 1h35m45s
- latest loss: 0.1891, recon: 0.1889
- timing step 155:
  - `step_total=38.7033s`
  - `backward=37.0677s`
  - `forward_decode=0.3294s`
  - `vjepa_feature_loss=0.6234s`
  - `sample_clip=0.2750s`
  - `render/rasterize=0.1780s`
  - `render_view_total=0.1782s`
- timing summary through step 155:
  - timing records: 31
  - median `step_total=38.4663s`
  - mean `step_total=38.2933s`
  - median `backward=36.2277s`
  - mean `backward=36.5012s`
  - median `render/rasterize=0.1672s`
  - median `vjepa_feature_loss=0.6235s`
  - latest fractions: `backward=0.9577`, `forward_decode=0.0085`, `render/rasterize=0.0046`, `vjepa_feature_loss=0.0161`
- media:
  - 2048 image: `wandb/offline-run-20260516_170606-doer3v9t/files/media/images/Render_GT_vs_Pred_140_a257f4c4cc11d1616d41.png`
  - 512 baseline comparison: `wandb/offline-run-20260516_133144-v856t3eg/files/media/images/Render_GT_vs_Pred_140_d7e722822699298b92fe.png`

Visual read: the 2048 step-140 surfer/wave image is much less blank than the 512 baseline at the same logged image step, but still lacks scene detail. The 512 baseline had mostly white area with isolated colored blobs; 2048 fills the water/sky area better and is more color-aligned, but it still collapses the surfer and wave texture into broad blocky fields. This strengthens the current queue decision: 2048 is a useful fast capacity baseline, but the next run needs more decoded token banks. Keep waiting for the 2048 final media and then let the 8192-token-capacity waiter launch.

2048-splat step-205 / image-video-190 checkpoint:

- status: step 205/300
- cache hits logged: 211
- cache misses logged: 0
- latest displayed rate: 39.47s/step, ETA about 1h02m29s
- latest loss: 0.2396, recon: 0.2395
- timing step 205:
  - `step_total=39.8719s`
  - `backward=37.7262s`
  - `forward_decode=0.5597s`
  - `vjepa_feature_loss=0.6563s`
  - `sample_clip=0.4663s`
  - `render/rasterize=0.2044s`
  - `render_view_total=0.2048s`
- timing summary through step 205:
  - timing records: 41
  - median `step_total=38.5201s`
  - mean `step_total=38.4719s`
  - median `backward=36.2277s`
  - mean `backward=36.5093s`
  - median `render/rasterize=0.1818s`
  - median `vjepa_feature_loss=0.6273s`
  - latest fractions: `backward=0.9462`, `forward_decode=0.0140`, `render/rasterize=0.0051`, `vjepa_feature_loss=0.0165`
- media:
  - image: `wandb/offline-run-20260516_170606-doer3v9t/files/media/images/Render_GT_vs_Pred_190_ed49e517f80df99631f4.png`
  - side-by-side video: `wandb/offline-run-20260516_170606-doer3v9t/files/media/videos/Render_GT_Video_190_104f477d106c014a75ba.mp4`
  - render video: `wandb/offline-run-20260516_170606-doer3v9t/files/media/videos/Render_Video_190_1f903136f0ddc7b797cd.mp4`
  - contact sheets:
    - `outputs/run_logs/capacity_2048_render_gt_video_step200_contact_sheet.jpg`
    - `outputs/run_logs/capacity_2048_render_video_step200_contact_sheet.jpg`
- video metadata:
  - side-by-side: 1024x512, 24fps, 63 frames, 2.625s
  - render-only: 512x512, 24fps, 63 frames, 2.625s

Visual read: the step-190 still and video are more filled and more stable than the 512-splat baseline, but the output is still a smooth block field. The forest video has coherent motion/color regions but no tree texture or usable structure. This confirms the current decision: 2048 is a valid fast/fill baseline, not the target quality config. Do not change renderer lanes yet because `render/rasterize` remains around 0.5% of step time. Let the 2048 run finish and let the queued 8192-token-capacity run test whether independent decoded token count fixes the blockiness.

Final 2048-splat capacity run and 8192 launch:

- 2048 status: completed 300/300
- 2048 screen: gone after completion
- 2048 cache hits logged: 306
- 2048 cache misses logged: 0
- 2048 elapsed: 11565s
- 2048 final displayed rate: 38.55s/step
- 2048 final loss: 0.2260, recon: 0.2258
- 2048 timing step 300:
  - `step_total=38.8082s`
  - `backward=36.7480s`
  - `forward_decode=0.4787s`
  - `vjepa_feature_loss=0.6147s`
  - `sample_clip=0.5074s`
  - `render/rasterize=0.2109s`
  - `render_view_total=0.2113s`
- 2048 timing summary:
  - timing records: 60
  - median `step_total=38.4716s`
  - mean `step_total=38.2840s`
  - median `backward=36.2320s`
  - mean `backward=36.3524s`
  - median `render/rasterize=0.1865s`
  - median `vjepa_feature_loss=0.6241s`
  - latest fractions: `backward=0.9469`, `forward_decode=0.0123`, `render/rasterize=0.0054`, `vjepa_feature_loss=0.0158`
- 2048 final media:
  - image: `wandb/offline-run-20260516_170606-doer3v9t/files/media/images/Render_GT_vs_Pred_290_55861dd3cd95b8d2d683.png`
  - side-by-side video: `wandb/offline-run-20260516_170606-doer3v9t/files/media/videos/Render_GT_Video_290_2109e5fac496500fa964.mp4`
  - render video: `wandb/offline-run-20260516_170606-doer3v9t/files/media/videos/Render_Video_290_d97d4527769fc5c8b8e3.mp4`
  - contact sheets:
    - `outputs/run_logs/capacity_2048_render_gt_video_step300_contact_sheet.jpg`
    - `outputs/run_logs/capacity_2048_render_video_step300_contact_sheet.jpg`
- video metadata:
  - side-by-side: 1024x512, 24fps, 63 frames, 2.625s
  - render-only: 512x512, 24fps, 63 frames, 2.625s

Final 2048 visual read: much better fill than the 512-splat baseline, but still blocky. The final biking/street still shows rough color/region alignment but no handlebar/body/tree/road detail. The final forest video remains temporally coherent but smooth and block-structured, not scene-textured. 2048 is the current fast local baseline, not a quality win.

8192 token-capacity launch:

- launched by waiter after 2048 reached 300/300
- screen: `93181.dynaworld_300_64f_512_8192_train_20260516_202029`
- config: `src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_vjepa_loss_8192splats_token_capacity.jsonc`
- run log: `outputs/run_logs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_vjepa_loss_8192splats_token_capacity_run_20260516_202029.log`
- W&B offline run: `wandb/offline-run-20260516_202031-e5jd34ut`
- start contract: 128 active decoded 3DGS tokens inside 136 total non-camera query tokens x 64 gaussians/token = 8192 explicit Gaussians, same 300 clips, 64f, 512 render, center-square crop, precomputed V-JEPA features, V-JEPA feature loss, static/dynamic/register layout, and implicit camera.
- early status at step 10:
  - cache hits logged: 14
  - cache misses logged: 0
  - rate: 37.44s/step, ETA about 3h00m57s
  - latest loss: 0.5451, recon: 0.5451
- early timing summary through step 10:
  - timing records: 2
  - median `step_total=37.1685s`
  - mean `step_total=37.1685s`
  - median `backward=35.1021s`
  - median `render/rasterize=0.1470s`
  - median `vjepa_feature_loss=0.6244s`
  - latest fractions: `backward=0.9460`, `forward_decode=0.0114`, `render/rasterize=0.0043`, `vjepa_feature_loss=0.0171`

Early 8192 interpretation: 4x more active decoded tokens than 2048 does not slow the current fast_mac path. It is still backward-bound, not raster-forward-bound. Let it reach the first image gate before judging quality; if it remains blocky, the next likely question is whether the loss/model backward path is limiting detail rather than renderer forward.

8192 token-capacity step-52 / image-40 checkpoint:

- status: step 52/300
- cache hits logged: 56
- cache misses logged: 0
- latest displayed rate: 36.87s/step, ETA about 2h32m24s
- latest loss: 0.2432, recon: 0.2427
- timing step 50:
  - `step_total=35.7104s`
  - `backward=34.0827s`
  - `forward_decode=0.3233s`
  - `vjepa_feature_loss=0.5920s`
  - `sample_clip=0.3162s`
  - `render/rasterize=0.1730s`
  - `render_view_total=0.1732s`
- timing summary through step 50:
  - timing records: 10
  - median `step_total=36.0889s`
  - mean `step_total=36.5489s`
  - median `backward=34.4369s`
  - mean `backward=34.6992s`
  - median `render/rasterize=0.1735s`
  - median `vjepa_feature_loss=0.6086s`
  - latest fractions: `backward=0.9544`, `forward_decode=0.0091`, `render/rasterize=0.0048`, `vjepa_feature_loss=0.0166`
- media:
  - 8192 image: `wandb/offline-run-20260516_202031-e5jd34ut/files/media/images/Render_GT_vs_Pred_40_60171b3e59560eb1a548.png`
  - 2048 same-checkpoint comparison: `wandb/offline-run-20260516_170606-doer3v9t/files/media/images/Render_GT_vs_Pred_40_1c30acb365601476520a.png`

Visual read: 8192 is not a clear quality jump over 2048 at the first image gate. It is stable and filled, but the predicted half is still a broad rectangular color field with no extra rock/tree/detail structure. The important positive is speed: 8192 active decoded tokens are not slower than 2048 on this setup. Keep 8192 running to the step-100 video gate; if video remains equally blocky, the next useful test is the prepared recon-only profile to isolate whether V-JEPA/loss backward is constraining detail before doing STAR/Metal renderer work.

8192 token-capacity step-109 / image-video-90 checkpoint:

- status: step 109/300
- cache hits logged: 114
- cache misses logged: 0
- latest displayed rate: 38.67s/step, ETA about 2h03m06s
- latest loss: 0.2009, recon: 0.2005
- timing step 100:
  - `step_total=37.5353s`
  - `backward=35.1737s`
  - `forward_decode=0.6949s`
  - `vjepa_feature_loss=0.6358s`
  - `sample_clip=0.5006s`
  - `render/rasterize=0.2670s`
  - `render_view_total=0.2675s`
- timing summary through step 100:
  - timing records: 20
  - median `step_total=36.7015s`
  - mean `step_total=37.0731s`
  - median `backward=34.8517s`
  - mean `backward=35.2713s`
  - median `render/rasterize=0.1830s`
  - median `vjepa_feature_loss=0.6084s`
  - latest fractions: `backward=0.9371`, `forward_decode=0.0185`, `render/rasterize=0.0071`, `vjepa_feature_loss=0.0169`
- media:
  - image: `wandb/offline-run-20260516_202031-e5jd34ut/files/media/images/Render_GT_vs_Pred_90_268b54df2d2151410aa8.png`
  - side-by-side video: `wandb/offline-run-20260516_202031-e5jd34ut/files/media/videos/Render_GT_Video_90_653980b1fefd33eab9f0.mp4`
  - render video: `wandb/offline-run-20260516_202031-e5jd34ut/files/media/videos/Render_Video_90_38652720570c9c423113.mp4`
  - contact sheets:
    - `outputs/run_logs/capacity_8192_render_gt_video_step100_contact_sheet.jpg`
    - `outputs/run_logs/capacity_8192_render_video_step100_contact_sheet.jpg`
- video metadata:
  - side-by-side: 1024x512, 24fps, 63 frames, 2.625s
  - render-only: 512x512, 24fps, 63 frames, 2.625s

Visual read: 8192 remains fast and stable but still blocky at the video gate. The step-90 arcade/city still is smoother and less blank than the 512 baseline, but it does not recover pedestrians, cars, ceiling panels, or floor detail. The video contact sheets are coherent but nearly all low-frequency color fields. 8192 is not yet the quality breakthrough; it should still finish for a fair final comparison because it is not slower than 2048.

Queued recon-only attribution after 8192:

- resolve/cache checked for `src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_profile.jsonc`
- resolve result:
  - active decoded tokens: 32
  - static full capacity: 24
  - dynamic full capacity: 8
  - total non-camera query tokens: 40
  - gaussians per token: 16
  - approx decoded Gaussians: 512
  - render size: 512
  - train frames: 64
  - V-JEPA crop and feature-loss crop metadata: 256
- cache status: 300/300 feature files, 1.0 coverage.
- waiter script: `outputs/run_logs/dynaworld_recon_only_after_8192_20260516_212735_waiter.sh`
- waiter screen: `8732.dynaworld_recon_only_after_8192_20260516_212735`
- waiter log: `outputs/run_logs/dynaworld_recon_only_after_8192_20260516_212735_waiter.log`
- current waiter evidence: it saw the 8192 run at 109/300 with one live training screen and is polling every 120s.

Next decision: let 8192 finish. If 8192 final is still blocky, the recon-only profile should launch automatically after 8192 reaches 300/300. Use that to compare backward cost and quality behavior without V-JEPA feature loss. Only revisit STAR/Metal renderer work if the evidence shifts toward renderer backward/forward being a real fraction of wall time; all current 512/2048/8192 timings keep raster forward below 1% of step time.

8192 token-capacity partial stop / recon-only pivot:

- status: the 8192 training screen `93181.dynaworld_300_64f_512_8192_train_20260516_202029` disappeared before completion.
- final parsed 8192 progress: step 155/300.
- cache hits logged: 160.
- cache misses logged: 0.
- final 8192 log: `outputs/run_logs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_vjepa_loss_8192splats_token_capacity_run_20260516_202029.log`
- final 8192 W&B dir: `wandb/offline-run-20260516_202031-e5jd34ut`
- final 8192 timing step 155:
  - `step_total=50.9317s`
  - `backward=48.5587s`
  - `forward_decode=0.5091s`
  - `vjepa_feature_loss=0.6557s`
  - `sample_clip=0.6293s`
  - `render/rasterize=0.2855s`
  - `render_view_total=0.2858s`
- timing summary through step 155:
  - timing records: 31
  - median `step_total=37.6275s`
  - median `backward=35.7636s`
  - median `render/rasterize=0.2011s`
  - median `vjepa_feature_loss=0.6119s`
- observed exit evidence: no Python traceback in the run log or screen log; the final line is a Python `resource_tracker` warning about one leaked semaphore object during shutdown.
- media still available:
  - image: `wandb/offline-run-20260516_202031-e5jd34ut/files/media/images/Render_GT_vs_Pred_140_03afe4eede06454f5428.png`
  - image: `wandb/offline-run-20260516_202031-e5jd34ut/files/media/images/Render_GT_vs_Pred_90_268b54df2d2151410aa8.png`
  - image: `wandb/offline-run-20260516_202031-e5jd34ut/files/media/images/Render_GT_vs_Pred_40_60171b3e59560eb1a548.png`
  - side-by-side video: `wandb/offline-run-20260516_202031-e5jd34ut/files/media/videos/Render_GT_Video_90_653980b1fefd33eab9f0.mp4`
  - render video: `wandb/offline-run-20260516_202031-e5jd34ut/files/media/videos/Render_Video_90_38652720570c9c423113.mp4`

Visual read at the last available 8192 image gate: the step-140 surfer/wave sample is smoother and less noisy than early 512, but still lacks the surfer and wave texture. It remains a low-frequency/blocky reconstruction, not a quality breakthrough. Because this trainer has no usable checkpoint/resume artifact for this run, restarting 8192 would restart from step 0. I treated the 155-step stop plus the step-90/140 visual verdict as enough evidence to move to the queued attribution lane instead of burning another full 8192 local pass.

Recon-only attribution launched after partial 8192 stop:

- config: `src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_profile.jsonc`
- screen: `27502.dynaworld_300_64f_512_recon_only_after_partial8192_20260516_220256`
- screen log: `outputs/run_logs/dynaworld_recon_only_after_partial8192_20260516_220256_screen.log`
- run log: `outputs/run_logs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_profile_run_20260516_220256.log`
- W&B dir: `wandb/offline-run-20260516_220258-dtd5gbcl`
- cache status: 300/300 feature files, 0 cache misses so far.
- initial parsed progress after launch: step 24/300.
- initial displayed rate after launch: 2.53s/step, ETA about 11m37s.
- initial loss after launch: `0.4315`, recon: `0.4307`.
- timing summary through step 20:
  - timing records: 4
  - median `step_total=2.6576s`
  - mean `step_total=2.7194s`
  - median `backward=1.5871s`
  - median `render/rasterize=0.0835s`
  - median `vjepa_feature_loss=0.0008s`
  - latest fractions: `backward=0.5575`, `forward_decode=0.1618`, `render/rasterize=0.0325`, `vjepa_feature_loss=0.0003`

Early attribution read: recon-only is roughly 14x faster than the V-JEPA-loss configs at the same 64-frame/512-render shell (`~2.7s` median step total versus `~37-44s`). Renderer forward is still not the problem. The expensive term in the V-JEPA-loss lanes is the autograd path surrounding feature-loss/backward, not rasterization. Let recon-only finish, then inspect its final media before deciding whether the next useful local config is a lower-cost perceptual loss, a detached/low-stride V-JEPA loss, or a higher-capacity recon-only variant.

Recon-only step-100 video gate:

- status at inspection: step 112/300, then step 137/300 shortly after.
- cache misses: 0.
- timing summary through step 110:
  - timing records: 22
  - median `step_total=2.5558s`
  - mean `step_total=2.6318s`
  - median `backward=1.4667s`
  - median `render/rasterize=0.1120s`
  - median `vjepa_feature_loss=0.0008s`
- media:
  - image: `wandb/offline-run-20260516_220258-dtd5gbcl/files/media/images/Render_GT_vs_Pred_90_bd2d522c5640c0ee6165.png`
  - side-by-side video: `wandb/offline-run-20260516_220258-dtd5gbcl/files/media/videos/Render_GT_Video_90_2fcc1be991263995c61f.mp4`
  - render video: `wandb/offline-run-20260516_220258-dtd5gbcl/files/media/videos/Render_Video_90_a643f7d132357cac358e.mp4`
  - contact sheets:
    - `outputs/run_logs/recon_only_Render_GT_Video_90_2fcc1be991263995c61f_step100_contact_sheet.jpg`
    - `outputs/run_logs/recon_only_Render_Video_90_a643f7d132357cac358e_step100_contact_sheet.jpg`
- video metadata:
  - side-by-side: 1024x512, 24fps, 63 frames, 2.625s
  - render-only: 512x512, 24fps, 63 frames, 2.625s

Visual read: recon-only is dramatically faster, but it does not solve the blocky output. The step-100 forest video still collapses to coherent low-frequency rectangular color fields. Removing the V-JEPA feature-loss backward path fixes wall-clock cost, but not the missing high-frequency scene detail by itself.

Prepared next local capacity attribution:

- new config: `src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_8192splats_token_capacity.jsonc`
- purpose: keep the 8192 active-Gaussian token-capacity layout from the stopped 8192 V-JEPA-loss run, but set `losses.vjepa_feature_weight=0.0` to test capacity without the expensive V-JEPA-loss backward path.
- resolve result:
  - active decoded tokens: 128
  - approx decoded Gaussians: 8192
  - render size: 512
  - train frames: 64
  - implicit camera decode-time refinement: true
- cache status: 300/300 feature files, 1.0 coverage.
- waiter script: `outputs/run_logs/dynaworld_recon8192_after_recon512_20260516_220951_waiter.sh`
- waiter screen: `32961.dynaworld_recon8192_after_recon512_20260516_220951`
- waiter log: `outputs/run_logs/dynaworld_recon8192_after_recon512_20260516_220951_waiter.log`

Next decision: let the 512-Gaussian recon-only run finish. The waiter should then launch the 8192-Gaussian recon-only token-capacity run. If 8192 recon-only remains blocky, the failure is less likely to be V-JEPA feature loss or raw Gaussian count alone; the next useful direction is changing the representation/objective, not STAR/Metal renderer work.

512-Gaussian recon-only final:

- status: completed 300/300.
- cache hits logged: 306.
- cache misses logged: 0.
- final log: `outputs/run_logs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_profile_run_20260516_220256.log`
- W&B dir: `wandb/offline-run-20260516_220258-dtd5gbcl`
- final displayed rate: 3.62s/step, elapsed 1086s.
- timing summary:
  - timing records: 60
  - median `step_total=3.0846s`
  - mean `step_total=3.4532s`
  - median `backward=1.6751s`
  - median `forward_decode=0.5215s`
  - median `render/rasterize=0.1288s`
  - median `vjepa_feature_loss=0.0009s`
- final media:
  - image: `wandb/offline-run-20260516_220258-dtd5gbcl/files/media/images/Render_GT_vs_Pred_290_2ab57ecae6233ccc3da8.png`
  - side-by-side video: `wandb/offline-run-20260516_220258-dtd5gbcl/files/media/videos/Render_GT_Video_290_732928ed3f39b529f128.mp4`
  - render video: `wandb/offline-run-20260516_220258-dtd5gbcl/files/media/videos/Render_Video_290_98ad377b8c7fdf873536.mp4`
  - contact sheets:
    - `outputs/run_logs/Render_GT_Video_290_732928ed3f39b529f128_contact_sheet.jpg`
    - `outputs/run_logs/Render_Video_290_98ad377b8c7fdf873536_contact_sheet.jpg`

Visual read: the final 512-Gaussian recon-only forest video is worse than the early smooth blocky output: it collapses into sparse white/background-dominated blocks plus a few colored shapes. It is fast, but not useful quality.

8192-Gaussian recon-only token-capacity final:

- status: completed 300/300.
- cache hits logged: 306.
- cache misses logged: 0.
- final log: `outputs/run_logs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_8192splats_token_capacity_run_20260516_222130.log`
- W&B dir: `wandb/offline-run-20260516_222132-vrt6hnb4`
- final displayed rate: 3.81s/step, elapsed 1143s.
- final loss: `0.2331`, recon: `0.2330`.
- timing summary:
  - timing records: 60
  - median `step_total=3.2835s`
  - mean `step_total=3.4598s`
  - median `backward=1.8172s`
  - median `forward_decode=0.4521s`
  - median `render/rasterize=0.2755s`
  - median `vjepa_feature_loss=0.0009s`
- final media:
  - image: `wandb/offline-run-20260516_222132-vrt6hnb4/files/media/images/Render_GT_vs_Pred_290_1d17a4494cc43d1dc088.png`
  - side-by-side video: `wandb/offline-run-20260516_222132-vrt6hnb4/files/media/videos/Render_GT_Video_290_b38dbfdf1648f12e2b3e.mp4`
  - render video: `wandb/offline-run-20260516_222132-vrt6hnb4/files/media/videos/Render_Video_290_4994fa5e973e306ca9bb.mp4`
  - contact sheets:
    - `outputs/run_logs/Render_GT_Video_290_b38dbfdf1648f12e2b3e_recon8192_final_contact_sheet.jpg`
    - `outputs/run_logs/Render_Video_290_4994fa5e973e306ca9bb_recon8192_final_contact_sheet.jpg`

Final attribution:

- Runtime: with V-JEPA feature loss disabled, 512-Gaussian and 8192-Gaussian runs are both around 3-3.3s median step total. The 36-38s median step total belongs to the V-JEPA feature-loss path, not to cached feature conditioning, decode, or renderer forward.
- Quality: 8192 recon-only improves over the collapsed 512 recon-only final, but it still produces broad low-frequency rectangular fields and does not recover forest texture or scene detail. Raw Gaussian/token capacity alone is not enough.
- Renderer: final recon-only 8192 raster median is `0.2755s`, under 10% of step total; renderer forward is still not the main runtime blocker.
- Strongest next direction: keep V-JEPA features cached for conditioning, but do not backprop through a full frozen V-JEPA encoder every step/frame chunk. Test cheaper detail objectives or representation changes before more STAR/Metal renderer work.

User-requested next key runs:

- 400-step single-item overfit. Purpose: prove the representation/objective can actually memorize one 64-frame 512-center-crop clip before spending time on dataset-scale runs. Preferred scale-up axis: test `gaussians_per_token=256` because it is a relatively cheap capacity increase compared with increasing query token count. With the current 32 active decoded-token layout this would be `32 * 256 = 8192` active Gaussians per rendered frame. With the 128 active decoded-token layout this would be `128 * 256 = 32768` active Gaussians per rendered frame.
- 300-item dataset for 3000 steps. Purpose: check whether the full all-YouTube 300-window lane learns beyond the 300-step one-pass regime once the overfit gate is positive. Use the same 64-frame, 512-center-crop data contract and keep V-JEPA conditioning cache-hot. Avoid full differentiable V-JEPA-loss backward by default unless testing a deliberately cheaper/strided objective.

Capacity accounting reminder:

- 512 render pixels per frame: `512 * 512 = 262144`.
- 64-frame clip pixels: `64 * 512 * 512 = 16777216`.
- 2048 active Gaussians means `2048` per rendered frame, not per token. At 512px this is `2048 / 262144 = 0.0078125` Gaussians per pixel, or about one Gaussian per 128 pixels.
- 8192 active Gaussians is `0.03125` Gaussians per pixel, or about one Gaussian per 32 pixels.
- 32768 active Gaussians is `0.125` Gaussians per pixel, or about one Gaussian per 8 pixels.
- These Gaussians are reused across all 64 frames with static/dynamic time behavior, so a 64-frame 8192-Gaussian clip renders `8192 * 64 = 524288` Gaussian-frame evaluations before culling/tile pruning.

No-V-JEPA-loss current lane config update:

- The launcher default now points at `src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_gpt256_8192splats_overfit1_400step.jsonc`, not the old V-JEPA-loss config.
- Added one-record overfit manifest: `data/single_video_pretrain/dynaworld_single_video_pretrain_300_youtube_64f_512_v0/train_manifest_overfit_first.jsonl`.
- Added `src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_gpt256_8192splats_overfit1_400step.jsonc`.
  - Purpose: first gate for whether the representation/objective can memorize one natural-FPS 64-frame, 512-center-crop clip.
  - V-JEPA usage: cached conditioning only; `losses.vjepa_feature_weight=0.0`.
  - Capacity: 24 static + 8 dynamic active decoded tokens, `gaussians_per_token=256`, `32 * 256 = 8192` active Gaussians/frame.
- Added `src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_gpt256_8192splats_300clips_3kstep.jsonc`.
  - Purpose: full 300-window, 3000-step follow-up after the one-item overfit gate is positive.
  - V-JEPA usage: cached conditioning only; `losses.vjepa_feature_weight=0.0`.
  - Capacity: same 32 active decoded tokens and `gaussians_per_token=256`, so 8192 active Gaussians/frame.
- Bench list in `src/train_scripts/train_single_video_pretrain_300_64f.sh` now compares no-loss variants by default: 512-splat recon-only, 128-token/64-gpt 8192-splat recon-only, and 32-token/256-gpt 8192-splat recon-only.
- Interpretation: old `*_vjepa_loss*.jsonc` configs are historical ablations. Current local training should use cached V-JEPA conditioning plus RGB/DSSIM/MSE reconstruction, not differentiable frozen-V-JEPA feature loss, until we deliberately design a cheaper feature objective.

400-step one-record overfit launch:

- Probe: `TRAIN_CONFIG=src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_gpt256_8192splats_overfit1_400step.jsonc PROBE_STEPS=1 ./src/train_scripts/train_single_video_pretrain_300_64f.sh probe`.
  - Result: passed on MPS, 1 train sequence, cache hits, `32 active decoded tokens x 256 gaussians/token = 8192 explicit Gaussians`, no V-JEPA feature-loss module, ~3.92s/step.
  - Probe log: `outputs/run_logs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_gpt256_8192splats_overfit1_400step_probe1step_20260516_235414.log`.
- Active run: `97833.dynaworld_gpt256_overfit1_400_20260516_235439`.
  - Screen log: `outputs/run_logs/dynaworld_gpt256_overfit1_400_20260516_235439_screen.log`.
  - Trainer log: `outputs/run_logs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_gpt256_8192splats_overfit1_400step_run_20260516_235439.log`.
  - W&B offline dir: `wandb/offline-run-20260516_235441-0x2l83ko`.
  - Early status: step 7/400, 0 cache misses, last loss/recon `0.5899`, last logged rate `6.22s/step`, timing step 5 `step_total=4.1542s`, `backward=2.6559s`, `render/rasterize=0.1668s`, `vjepa_feature_loss=0.0010s`.

Runtime correction after user asked whether this should be a ~2-minute UVT STAR run:

- Current one-record overfit is not a UVT STAR trainer run. `train_video_token_implicit_dynamic` resolves renderers through `rendering.pick_renderer_mode`, whose supported explicit modes are `dense`, `tiled`, `taichi`, and `fast_mac`; this config resolves to `fast_mac`.
- Live status check at step 21/400:
  - elapsed: 146s already, so the run is not a 2-minute finish.
  - latest displayed ETA: ~52m30s, volatile but directionally far above 2 minutes.
  - cache hits logged: 25; cache misses: 0.
  - loss/recon: `0.2863` / `0.2850`.
  - timing records through step 20:
    - median `step_total=6.4308s`
    - median `backward=4.4001s`
    - median `forward_decode=0.7648s`
    - median `render/rasterize=0.3236s`
    - median `vjepa_feature_loss=0.0021s`
- Interpretation: turning off V-JEPA loss worked, but 64f/512/8192-splat token-GS training is still dominated by model/reconstruction backward. Renderer forward is not the primary cost, and UVT STAR speed expectations do not apply to this trainer until a STAR renderer/trainer path is actually wired into the same overfit lane.
- The run was stopped after confirming this mismatch, around step 190/400. It produced media through step 140 in `wandb/offline-run-20260516_235441-0x2l83ko`.

May 17 correction/update: the 512-only overfit artifact is complete and
comparable despite the earlier stopped-run note above. Parsed W&B history for
`wandb/offline-run-20260516_235441-0x2l83ko` and the trainer log show:

- config: `src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_gpt256_8192splats_overfit1_400step.jsonc`
- log: `outputs/run_logs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_gpt256_8192splats_overfit1_400step_run_20260516_235439.log`
- wall: `49:15`
- final train loss/recon: `0.0907 / 0.0904`
- final eval PSNR/SSIM: `24.374 / 0.4721`
- final eval L1/MSE: `0.04589 / 0.003653`
- timing medians over W&B profile rows:
  - `step_total=5.4956s`
  - `backward=3.3612s`
  - `forward_decode=0.7558s`
  - `render/rasterize=0.4428s`
  - `vjepa_feature_loss=0.0011s`

Main-trainer render-size schedule comparison:

- implemented opt-in `train.render_size_schedule` in `src/train/train_video_token_implicit_dynamic.py`;
- completed `src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_gpt256_8192splats_overfit1_400step_multires_256to512.jsonc`;
- W&B: `wandb/offline-run-20260517_112427-acv8pinq`;
- log: `outputs/run_logs/dynaworld_overfit_multires_256to512_20260517_112425.log`;
- schedule: 256px through step 299, 512px from step 300;
- wall: `16:36`;
- final train loss/recon: `0.0781 / 0.0777`;
- final eval PSNR/SSIM: `24.766 / 0.5316`;
- final eval L1/MSE: `0.04359 / 0.003338`;
- timing medians:
  - all profile rows: `step_total=2.4667s`, `backward=1.3558s`, `render/rasterize=0.1611s`;
  - 256px rows: `step_total=1.7302s`, `backward=0.8653s`, `render/rasterize=0.1539s`;
  - final 512px rows: `step_total=3.1376s`, `backward=1.8265s`, `render/rasterize=0.2920s`.

Interpretation: the STAR-style coarse-to-fine render-size schedule is a real
main-trainer throughput win and also beat the 512-only overfit on final eval
PSNR/SSIM. It is still not a solved quality lane because absolute SSIM is weak,
so do not launch the 300-clip 3k multires config yet.

New local schedule probe launched:

- config: `src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_gpt256_8192splats_overfit1_400step_multires_256c200_512f200.jsonc`
- screen: `dynaworld_overfit_multires_256c200_512f200_20260517_141005`
- log: `outputs/run_logs/dynaworld_overfit_multires_256c200_512f200_20260517_141005.log`
- W&B: `wandb/offline-run-20260517_141007-uvp7ifwr`
- schedule: 256px through step 199, 512px from step 200;
- purpose: test whether the 300-step warmup stayed low-res too long. Promote
  only if final eval/media improve enough to justify more 512px training time.

New local schedule probe result:

- status: completed 400/400, no traceback.
- wall: `16:59`.
- final train loss/recon: `0.1088 / 0.1087`.
- final eval PSNR/SSIM: `23.291 / 0.3304`.
- final eval L1/MSE: `0.05287 / 0.004687`.
- final media:
  - image: `wandb/offline-run-20260517_141007-uvp7ifwr/files/media/images/Render_GT_vs_Pred_390_3bde87bfe40e788751f5.png`
  - render video: `wandb/offline-run-20260517_141007-uvp7ifwr/files/media/videos/Render_Video_390_257c2270b9b01f99385d.mp4`
  - side-by-side video: `wandb/offline-run-20260517_141007-uvp7ifwr/files/media/videos/Render_GT_Video_390_37ca5b51c50e65eaa7ea.mp4`
- media dimensions verified:
  - final image strip `1024x512`;
  - render video `512x512`, 63 frames, 24fps, 2.625s;
  - side-by-side video `1024x512`, 63 frames, 24fps, 2.625s.
- timing medians:
  - all profile rows: `step_total=2.5135s`, `backward=1.4881s`, `render/rasterize=0.2553s`;
  - 256px rows: `step_total=1.6331s`, `backward=0.8361s`, `render/rasterize=0.1534s`;
  - 512px rows: `step_total=2.8146s`, `backward=1.6748s`, `render/rasterize=0.2992s`.

Decision: do not promote `256c200 -> 512f200`. It is similar speed to
`256c300 -> 512f100` but much worse quality (`23.291/0.3304` vs
`24.766/0.5316` PSNR/SSIM). The 80/20 coarse-to-fine schedule remains the
main-trainer default: for 400 steps use 300/100, and for the prepared 300-clip
3k config keep 2400/600.

300-clip 3k dataset-scale launch:

- config: `src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_gpt256_8192splats_300clips_3kstep_multires_256to512.jsonc`
- screen: `dynaworld_300clips_3k_multires_256to512_20260517_143003`
- log: `outputs/run_logs/dynaworld_300clips_3k_multires_256to512_20260517_143003.log`
- W&B: `wandb/offline-run-20260517_143005-s230kzhu`
- data: 300 train sequences, cache coverage verified 300/300 before launch.
- schedule: 256px through step 2399, 512px from step 2400.
- capacity: `32 active decoded tokens * 256 gaussians/token = 8192` active Gaussians/frame.
- V-JEPA: cached conditioning only, `losses.vjepa_feature_weight=0.0`.
- early evidence at step ~57: screen alive, 0 visible cache misses, displayed tqdm rate around `2.8s/step` after some sequence variance, ETA around `2h19m` while still in the 256px stage.
- timing step 30: `step_total=1.8118s`, `backward=0.9713s`, `forward_decode=0.3349s`, `render/rasterize=0.1625s`, `vjepa_feature_loss=0.0025s`.
- timing step 40: `step_total=1.9737s`, `backward=1.0466s`, `forward_decode=0.3819s`, `render/rasterize=0.1866s`, `vjepa_feature_loss=0.0008s`.

Next gate: inspect timing/media around step 100, then step 300/600 if the run is
still healthy. Do not start another MPS training job while this screen is alive.

May 17 live follow-up:

- active screen is still `dynaworld_300clips_3k_multires_256to512_20260517_143003`.
- latest checked progress: about step `340/3000`, still in the 256px stage.
- no traceback and no visible feature-cache misses.
- first image artifact:
  `wandb/offline-run-20260517_143005-s230kzhu/files/media/images/Render_GT_vs_Pred_225_b63ffec9e91424e51f34.png`.
- first image dimensions: `512x256`, meaning GT/pred at the current 256px render size.
- visual read: very early and still broad/color-field only on the prediction side; not enough
  to judge the 300-clip run, but it confirms media logging is working.
- timing through step 310:
  - all timing rows median `step_total=2.703s`, `backward=1.371s`,
    `forward_decode=0.608s`, `sample_clip=0.515s`,
    `render/rasterize=0.173s`, `vjepa_feature_loss=0.0009s`;
  - last-5 timing rows median `step_total=3.846s`, `backward=2.112s`,
    `sample_clip=0.521s`, `render/rasterize=0.221s`.

Interpretation: renderer work is not the main blocker in the active 300-clip
lane. V-JEPA loss remains effectively off. The next speed target is repeated
per-step frame-window decoding/loading, especially because `cycle` repeats the
same 300 windows for roughly 10 epochs over a 3000-step run.

Code follow-up landed while the run continued:

- added an opt-in decoded-frame cache for `explicit_video_window` sequences in
  `src/train/sequence_data.py`;
- added `data.frame_cache_dir` config resolution in
  `src/train/train_video_token_implicit_dynamic.py`;
- enabled it in the 300-clip 3k config at
  `data/frame_cache/single_video_pretrain_300_youtube_64f_512center_nativefps`;
- focused test passed:
  `PYTHONPATH=src/train uv run --with pytest pytest tests/test_sequence_data_single_frame.py -q`
  -> `5 passed`.

Important caveat: the currently running screen was launched before this code and
config change, so it will not use the frame cache. A subsequent run with the
same config will populate/read cached uint8 decoded windows and should reduce
the repeated `sample_clip` cost after the first pass through the 300 clips.

May 17 restart decision:

- prewarmed decoded-frame cache:
  `data/frame_cache/single_video_pretrain_300_youtube_64f_512center_nativefps`;
- cache contents: `300` `.pt` files, `14G`;
- prewarm command loaded all manifest records through `load_manifest_sequence`
  on CPU; total time `253.9s`.

Old active run reached the first video gate before restart:

- stopped run: `dynaworld_300clips_3k_multires_256to512_20260517_143003`;
- W&B: `wandb/offline-run-20260517_143005-s230kzhu`;
- final inspected progress before stop: about `512/3000`;
- step-475 media:
  - image:
    `wandb/offline-run-20260517_143005-s230kzhu/files/media/images/Render_GT_vs_Pred_475_73e213b1f4add4243b36.png`
  - render video:
    `wandb/offline-run-20260517_143005-s230kzhu/files/media/videos/Render_Video_475_774263a0b0d535d9417b.mp4`
  - side-by-side video:
    `wandb/offline-run-20260517_143005-s230kzhu/files/media/videos/Render_GT_Video_475_c8639a7cffe437904a15.mp4`
  - contact sheets:
    `outputs/run_logs/300clip_multires_step475_render_contact.jpg`,
    `outputs/run_logs/300clip_multires_step475_sidebyside_contact.jpg`;
- visual read: GT was a forest/flyover-style clip; prediction was temporally
  coherent but mostly low-frequency blurred color bands. Not a quality win.
- timing through step 490:
  - all timing rows median `step_total=3.355s`, `backward=1.588s`,
    `forward_decode=0.689s`, `sample_clip=0.569s`,
    `render/rasterize=0.182s`, `vjepa_feature_loss=0.0011s`;
  - last-10 timing rows median `step_total=4.399s`, `backward=2.292s`,
    `sample_clip=0.660s`, `render/rasterize=0.230s`.

Decision: stop the old run and relaunch. Reason: it used the stale config with
sync-heavy profiling and no decoded-frame cache, had no useful checkpoint/resume
artifact, and the first video gate did not show quality worth preserving.

Replacement run:

- screen:
  `dynaworld_300clips_3k_multires_256to512_framecache_noprofile_20260517_150143`
- log:
  `outputs/run_logs/dynaworld_300clips_3k_multires_256to512_framecache_noprofile_20260517_150143.log`
- W&B:
  `wandb/offline-run-20260517_150153-r8fwjqhb`
- config:
  `src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_gpt256_8192splats_300clips_3kstep_multires_256to512.jsonc`
- config changes now active for this run:
  - `data.frame_cache_dir` points to the prewarmed decoded-frame cache;
  - `train.profile_timing=false`;
  - `train.profile_timing_sync=false`;
  - W&B run name includes `framecache-noprofile`.
- early status: about step `107/3000`, no traceback, no feature-cache misses,
  no `Timing step` rows by design.
- observed early tqdm rate: roughly `2.5s/step`; still monitor around step
  `250` image and step `500` video.

Next gates: step `250` image, step `500` video, then the step `2400` switch to
512px. Keep this as the only active MPS training job.

May 17 speed/debug reset:

- Active run parsed around `1371/3000`, no traceback, no feature-cache misses.
- Effective batch size is `1` manifest video window per optimizer step.
- Each step trains `64` frames; recon backward is microbatched as `4 x 16`
  frames, with no gradient accumulation.
- Live tqdm near the reset implied roughly `0.34 samples/s` and `22 frames/s`.
- The active trainer has no `torch.utils.data.DataLoader`, worker pool, or async
  prefetch; lazy manifest sampling loads the RGB window and feature cache on the
  step path.
- Next-run code/config changes were staged after the active process started:
  `logging.wandb_mode` support, `data.train_manifest_prefetch` support, config
  `wandb_mode=online`, and config `train_manifest_prefetch=2`.
- Current W&B is offline only because the process was launched with
  `WANDB_MODE=offline`; do not repeat that launch env for future training runs.
  Sync the offline run after finish if it remains worth keeping.

May 17 consolidated Q&A:

- Latest checked active run was about `2089/3000`, still before the 512px
  render switch at step `2400`, with feature-cache hits and no traceback.
- `sample_clip` is the dataloader-like timing section: manifest selection,
  cached RGB/window load, cached V-JEPA conditioning feature load, device move,
  64-frame clip selection, and `clip_frames` / `clip_times` prep. It is not the
  full sample step and does not include model forward, raster, loss, backward,
  or optimizer.
- The active 300-clip trainer is not STAR UVT. It routes through
  `src/train/train.py` with `arch=precomputed_feature_implicit_camera`, then
  `train_precomputed_feature_implicit_dynamic.py`, then
  `VideoTokenImplicitTrainer`. It emits standard per-frame `GaussianSequence`
  tensors and renders with `fast_mac` (`rgb_variant=v6_refined`,
  `feature_variant=v5_features`).
- STAR UVT and STAR UVT PRT remain separate research harnesses under
  `third_party/fast-mac-gsplat/variants/star_uvt_v0/` and
  `third_party/fast-mac-gsplat/variants/star_uvt_prt_v0/`. They emit screen/time
  or projective-rational tube parameters, not the same `GaussianSequence`
  contract.
- The modular trainer can be forked to keep data/W&B/media/config upgrades, but
  a real STAR integration should be a new arch/trainer or output adapter. A
  renderer-only swap would not test the STAR thesis because the output would
  still be per-frame GaussianSequence rather than compact UVT tubes.
- Next profile probe now has an opt-in `train.profile_backward_split` flag. With
  `train.profile_timing=true`, it times `backward/raster_loss_to_boundary`,
  `backward/model_from_boundary`, and `backward/regularizers`. Use it only on a
  short diagnostic run.
- Validation after the profiling patch:
  `PYTHONPATH=src/train python3 -m py_compile src/train/train_video_token_implicit_dynamic.py src/train/train_precomputed_feature_implicit_dynamic.py src/train/sequence_data.py`
  and
  `PYTHONPATH=src/train uv run --with pytest pytest tests/test_temporal_sampling.py tests/test_config_factory_helpers.py tests/test_sequence_data_single_frame.py -q`
  passed (`28 passed`).

May 17 Gaussian multires stop verdict:

- The replacement 300-clip run
  `dynaworld_300clips_3k_multires_256to512_framecache_noprofile_20260517_150143`
  reached the scheduled 512px promotion at step `2400`.
- The run slowed after promotion: recent tqdm rows moved from roughly
  `2-3s/step` before promotion into roughly `4-8s/step` after promotion.
- Total loss and camera terms became NaN around step `2429`:
  `Loss: nan recon: 0.6157 fov: nan r: nan`.
- I stopped the process rather than continue spending local MPS time on NaN
  updates. This is a Gaussian-sequence trainer stability warning, not a STAR
  UVT speed or quality result.
- Immediate next direction: stop treating this lane as the STAR proof and wire
  the STAR UVT overfit harness into a first-class `src/train/train.py` arch so
  high-motion 64-frame overfits can run with the modern config/W&B/media shell.

May 17 first-class STAR UVT follow-through:

- Added `arch=star_uvt_video_overfit` to `src/train/train.py`.
- Added `src/train/train_star_uvt_video_overfit.py`, a thin first-class wrapper
  around the existing STAR UVT video-fit harness with config-driven launch,
  online W&B, final metric logging, JSON output, contact sheets, and
  side-by-side MP4 export.
- Added configs:
  - `src/train_configs/star_uvt_highmotion_hlaZbH_64f_256_directatomic_200step.jsonc`
  - `src/train_configs/star_uvt_highmotion_hlaZbH_64f_256_tilepair_suffix_keyseg_20step_profile.jsonc`
- Validation:
  - tiny 64px/4f smoke through the new wrapper passed;
  - `PYTHONPATH=src/train python3 -m py_compile ...` passed;
  - `PYTHONPATH=src/train uv run --with pytest pytest tests/test_config_factory_helpers.py tests/test_sequence_data_single_frame.py tests/test_temporal_sampling.py -q` passed (`29 passed`).
- First-class direct-atomic 32768-tube/200-step result:
  - W&B: `https://wandb.ai/nbardy/dynaworld/runs/jba7kztn`
  - JSON: `star_uvt/results/may17_firstclass_highmotion_hlaZbH_start2_256_64f_32768t_cap256_directatomic_200step.json`
  - MP4: `star_uvt/results/may17_firstclass_highmotion_hlaZbH_start2_256_64f_32768t_cap256_directatomic_200step.mp4`
  - PSNR `29.823`, SSIM mean/min `0.8572/0.7788`, final loss `0.0010415`.
- First-class compact deterministic 8192-tube/20-step result:
  - W&B: `https://wandb.ai/nbardy/dynaworld/runs/641gxm9l`
  - JSON: `star_uvt/results/may17_firstclass_highmotion_hlaZbH_start2_256_64f_8192t_cap256_tilepair_suffix_keyseg_20step.json`
  - MP4: `star_uvt/results/may17_firstclass_highmotion_hlaZbH_start2_256_64f_8192t_cap256_tilepair_suffix_keyseg_20step.mp4`
  - PSNR `17.599`, SSIM mean/min `0.6148/0.5446`, final loss `0.017382`.
- Interpretation: first-class STAR UVT launch/logging/media is working.
  Direct-atomic/index_add remains the practical path. The deterministic compact
  path reproduces quality but remains too slow to promote and still needs a
  focused load-growth/backward pass.
