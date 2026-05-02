# V-JEPA Performance Logs

Append-only notes for the V-JEPA multicam baseline speed audit. Add a new dated entry when new measurements or conclusions change the bottleneck picture.

## 2026-05-02 - DeepView train2/test1 V-JEPA baseline timing

Goal: make the V-JEPA multicam baseline fast enough to be the default before scaling to the 1k-sample dataset, and avoid blaming the rasterizer without timing the actual path.

Config:

- `src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_static_dynamic_96_32_precomputed_vjepa2_1_vitb_384_128_16f_8192splats.jsonc`
- Architecture: `multicam_precomputed_feature_implicit_camera`
- Encoder backend: `precomputed`
- Static/dynamic split: `static_tokens=96`, `dynamic_tokens=32`
- Renderer: `fast_mac`
- Output representation: RGB splatting, not feature splatting. This config does not set `model.feature_dim`, so the model default is `feature_dim=3`; `src/train/renderers/fast_mac.py` dispatches `F == 3` to `v5`. The F32 configs are the feature-splatting path because they set `model.feature_dim: 32`, which dispatches to `v5_features`.

V-JEPA input and projection:

- Training clip loaded at 16 frames and 128 px render/input resolution for this config.
- The V-JEPA extractor resizes/normalizes frames to 384 px before the encoder, producing an encoder input shaped like `[B, C, T, 384, 384]`.
- Cache hit inspected on disk:
  - `data/feature_cache/multicam_deepview_static_dynamic_vjepa2_1_vitb_384/5bcd6903a42607cd116ef474.pt`
  - `vjepa_tokens`: shape `(1, 4608, 768)`, dtype `float16`, about 6.75 MiB
- The precomputed feature adapter keeps the 4608-token sequence length, applies `LayerNorm(768)`, then projects each token to `model_dim=128`.
- Projected conditioning shape during the benchmark: `(1, 4608, 128)`, dtype `float32`, about 2.25 MiB.

Command:

```bash
PYTHONPATH=src/train uv run python research_experiments/vjepa_performance/benchmark_multicam_vjepa.py --steps 3 --warmup 1
```

Measured on MPS with cache hits:

| Section | Mean |
| --- | ---: |
| trainer init total | 34.013866 s |
| adapter project features | 0.127501 s |
| sample clip | 0.006445 s |
| feature load / memory hit | 0.000322 s |
| model forward decode | 1.549352 s |
| render both train views | 0.175473 s |
| RGB reconstruction losses | 0.019328 s |
| backward | 0.455757 s |
| optimizer step | 0.012173 s |
| measured step total | 2.431634 s |

Current bottleneck read:

- Cached V-JEPA loading is not the runtime bottleneck after the first cache build; it is effectively free in the timed step loop.
- Rendering both views is much smaller than decode/backward, consistent with the rasterizer not being the limiting component at this scale.
- The dominant cost is `model_forward_decode`, likely because the static/dynamic decoder repeatedly attends over all 4608 projected V-JEPA memory tokens across the 16 decode times.

Next timing passes:

- Benchmark the F32 feature-splatting V-JEPA config to see whether feature rendering/colorization materially changes the profile.
- Add a focused decode breakdown or config-safe token reduction knob before changing architecture.
- Test V-JEPA memory-token reduction before cross-attention, since keeping all 4608 tokens is the current obvious attention multiplier.

## 2026-05-02 - F32 feature-splatting timing check

Question: is the V-JEPA multicam path using feature splatting, and if so does it make the step slow?

Answer:

- The ready DeepView 128 px V-JEPA baseline above is not feature splatting.
- The F32 configs are feature splatting. `local_mac_ultimate_features_F32_vjepa_multicam_256px_8192splats_alpha_lr3e4_camclamp.jsonc` sets `model.feature_dim: 32`, so `fast_mac` dispatches to `v5_features`.

Command:

```bash
PYTHONPATH=src/train uv run python research_experiments/vjepa_performance/benchmark_multicam_vjepa.py \
  --config src/train_configs/local_mac_ultimate_features_F32_vjepa_multicam_256px_8192splats_alpha_lr3e4_camclamp.jsonc \
  --steps 3 --warmup 1
```

Measured on MPS with cache hits:

| Section | Mean |
| --- | ---: |
| trainer init total | 27.154020 s |
| adapter project features | 0.007857 s |
| sample clip | 0.001651 s |
| feature load / memory hit | 0.000172 s |
| model forward decode | 0.500818 s |
| bank and rig losses | 0.004989 s |
| render both train views | 0.138377 s |
| recon losses | 0.031438 s |
| backward | 0.635486 s |
| optimizer step | 0.002515 s |
| measured step total | 1.485260 s |

Read:

- Feature splatting at 256 px is not the dominant slowdown in this measurement.
- Render time for two 256 px feature views is still only about 0.14 s/step, while decode plus backward is about 1.14 s/step.
- The F32 config is faster overall than the first RGB 128 px timing. The architectural knobs are mostly matched (`96/32` tokens, `cross_attn_layers=4`, `bottleneck_self_attn_layers=2`), so rerun the RGB timing before treating the absolute delta as a model truth. MPS warmup and graph compilation variance may have polluted the first short run.

## 2026-05-02 - Bottleneck probes and fast-camera baseline config

Implemented benchmark knobs:

- `model.video_feature_token_stride`: optional decimation of precomputed feature tokens before adapter projection/cross-attention. Default is `1`, so existing configs are unchanged.
- `model.camera_refine_with_decode_time`: optional static/dynamic camera-path speed knob. Default is `true`. When set `false`, the static/dynamic decoder reuses fixed V-JEPA-refined camera/path tokens and lets `_decode_camera_single_time` apply the existing per-frame `head_time_proj` before the camera head. This removes the 16 per-frame full-query cross-attention calls from the static/dynamic camera loop.

Failed/neutral probes:

- Mean-pooling precomputed tokens before projection reduced token count but was slower on MPS because the reduction op dominated; replaced with simple stride decimation.
- `video_feature_token_stride=4` reduced F32 projected tokens from `(1, 4608, 128)` to `(1, 1152, 128)`, but short MPS runs were slower/noisier than the default. Do not promote token stride as the speed default without a quality/speed sweep.
- Batched per-frame query refinement looked exact on paper but made the measured F32 path slower, especially backward, so it was reverted from the default path.

Controlled RGB baseline timing after longer warmup:

Command:

```bash
PYTHONPATH=src/train uv run python research_experiments/vjepa_performance/benchmark_multicam_vjepa.py \
  --config src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_static_dynamic_96_32_precomputed_vjepa2_1_vitb_384_128_16f_8192splats.jsonc \
  --steps 5 --warmup 3
```

| Section | Mean | Median |
| --- | ---: | ---: |
| model forward decode | 0.756111 s | 0.663648 s |
| render both train views | 0.067692 s | 0.060881 s |
| backward | 0.197003 s | 0.178685 s |
| measured step total | 1.120608 s | 0.991253 s |

Fast-camera RGB timing:

Command:

```bash
PYTHONPATH=src/train uv run python research_experiments/vjepa_performance/benchmark_multicam_vjepa.py \
  --config src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_static_dynamic_96_32_precomputed_vjepa2_1_vitb_384_128_16f_8192splats.jsonc \
  --steps 5 --warmup 3 --no-camera-refine-with-decode-time
```

| Section | Mean | Median |
| --- | ---: | ---: |
| model forward decode | 0.119905 s | 0.120707 s |
| render both train views | 0.047596 s | 0.045585 s |
| backward | 0.157968 s | 0.158292 s |
| measured step total | 0.395623 s | 0.392858 s |

Checked-in fast-camera config:

- `src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_static_dynamic_96_32_precomputed_vjepa2_1_vitb_384_128_16f_8192splats_fast_camera.jsonc`
- Validation command confirmed config resolution and factory kwargs both set `camera_refine_with_decode_time=False`.
- One-step benchmark smoke against the checked config path succeeded with a cache hit. With no warmup, the single measured step was 0.881818 s total and `model_forward_decode=0.215944 s`; use the warmed 5-step run above for speed comparison.

Current conclusion:

- Cached V-JEPA is not the step-time bottleneck.
- The rasterizer is not the bottleneck at this 128 px / 8192 splat RGB setting.
- The main default bottleneck is repeated time-conditioned cross-attention in model decode.
- The fast-camera config is the first credible lightning-fast baseline candidate: about 0.40 s/step measured on the same cached RGB setup. It still needs a real train/eval quality gate before being promoted in `BASELINES.md`, because it changes how camera tokens receive time-conditioned V-JEPA attention.

## 2026-05-02 - Small bf16 V-JEPA conditioning config

User concern: the adapter projected cached V-JEPA to `(1, 4608, 128)` fp32, which is too large for the intended fast baseline. Also verify whether the trainer was already bf16.

Findings:

- The existing configs were not globally bf16. `train.amp=false` kept model weights and adapter projection in fp32.
- On MPS, the trainer's previous `amp_dtype=auto` path selected fp16 when AMP was enabled. Added explicit `train.amp_dtype`, including `bf16`.
- The fast-mac renderer already promotes decoded Gaussian parameters/colors to `.float()` before projection/rasterization, so bf16/fp16 model compute does not imply bf16/fp16 raster inputs.

Implemented:

- `train.amp_dtype` config key, default `auto`.
- `model.video_feature_output_dtype`, default `null`, for explicitly casting projected precomputed feature memory.
- Camera composition dtype fix for bf16 autocast in `axis_angle_to_matrix` / `compose_camera_with_se3_delta`.
- Small conditioning config:
  - `src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_jepa_cond_small_bf16_fast_camera.jsonc`
  - `model_dim=64`
  - `num_heads=4`
  - `video_feature_token_stride=9`
  - `video_feature_output_dtype="bf16"`
  - `train.amp=true`
  - `train.amp_dtype="bf16"`
  - `camera_refine_with_decode_time=false`

Shape smoke:

- Cache input stayed `vjepa_tokens: (1, 4608, 768) float16`, 6.75 MiB.
- Projected conditioning became `(1, 512, 64) bfloat16`, 0.06 MiB.

Warm timing command:

```bash
PYTHONPATH=src/train uv run python research_experiments/vjepa_performance/benchmark_multicam_vjepa.py \
  --config src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_jepa_cond_small_bf16_fast_camera.jsonc \
  --steps 5 --warmup 3
```

| Section | Mean | Median |
| --- | ---: | ---: |
| model forward decode | 0.389069 s | 0.416915 s |
| render both train views | 0.072366 s | 0.072942 s |
| backward | 0.308054 s | 0.307237 s |
| measured step total | 0.880288 s | 0.872482 s |

Read:

- The small bf16 conditioning config achieves the target tensor shape/dtype and drastically cuts conditioning memory.
- It is not faster than the fp32 fast-camera path on this MPS run. The previous fast-camera RGB config measured about 0.396 s/step mean; small bf16 measured about 0.880 s/step mean.
- MPS bf16/autocast and smaller attention are not automatically faster here. Treat this as a memory/ablation config, not the current speed winner.

## 2026-05-02 - Current V-JEPA vs no-V-JEPA multicam speed

Question: all the direct free-splat / TokenGS numbers look fast, so why is the
V-JEPA multicam baseline still slow, and can we get a no-V-JEPA baseline close
to the same speed under the same train-2/test-1 camera contract?

Implementation added:

- `src/train/train_precomputed_feature_implicit_dynamic.py` now allows
  `model.video_encoder_backend="none"` for unconditioned controls. In that mode
  it skips `VideoFeatureCache` construction and returns `clip_frames` as model
  input.
- `research_experiments/vjepa_performance/benchmark_multicam_vjepa.py` now
  handles no-feature configs and can write `--output-json`.
- Added config:
  `src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_unconditioned_tokens_128_16f_8192splats_fast_camera.jsonc`.

All measurements below are local MPS, 128px, 16 decoded frames, 8192 splats,
two train cameras per step, DeepView `camera_0001,camera_0015 -> camera_0040`.
Each step renders/losses 32 target frames (`16f x 2 views`).

Commands:

```bash
PYTHONPATH=src/train WANDB_MODE=disabled uv run python \
  research_experiments/vjepa_performance/benchmark_multicam_vjepa.py \
  --config src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_static_dynamic_96_32_precomputed_vjepa2_1_vitb_384_128_16f_8192splats.jsonc \
  --steps 5 --warmup 3 \
  --output-json outputs/benchmarks/multicam_vjepa_default_128px_16f_8192splats_2026-05-02.json

PYTHONPATH=src/train WANDB_MODE=disabled uv run python \
  research_experiments/vjepa_performance/benchmark_multicam_vjepa.py \
  --config src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_static_dynamic_96_32_precomputed_vjepa2_1_vitb_384_128_16f_8192splats_fast_camera.jsonc \
  --steps 5 --warmup 3 \
  --output-json outputs/benchmarks/multicam_vjepa_fast_camera_128px_16f_8192splats_2026-05-02.json

PYTHONPATH=src/train WANDB_MODE=disabled uv run python \
  research_experiments/vjepa_performance/benchmark_multicam_vjepa.py \
  --config src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_unconditioned_tokens_128_16f_8192splats_fast_camera.jsonc \
  --steps 5 --warmup 3 \
  --output-json outputs/benchmarks/multicam_no_vjepa_unconditioned_tokens_128px_16f_8192splats_2026-05-02.json
```

Verification:

- `PYTHONPATH=src/train uv run python -m py_compile src/train/train_precomputed_feature_implicit_dynamic.py research_experiments/vjepa_performance/benchmark_multicam_vjepa.py`
- One-step no-VJEPA multicam smoke passed after the `video_encoder_backend="none"`
  validation cleanup and wrote
  `/tmp/multicam_no_vjepa_unconditioned_tokens_smoke_after_cache_validation_2026-05-02.json`.

Current step timing:

| Config | Mean step | Steps/s | Target frames/s | Decode | Render both views | Backward |
|---|---:|---:|---:|---:|---:|---:|
| Default V-JEPA | 2.535 s | 0.39 | 12.62 | 1.548 s | 0.189 s | 0.535 s |
| V-JEPA fast-camera | 0.697 s | 1.44 | 45.93 | 0.274 s | 0.080 s | 0.223 s |
| No-V-JEPA unconditioned tokens | 0.555 s | 1.80 | 57.66 | 0.202 s | 0.068 s | 0.189 s |

Why the default V-JEPA config is slow:

- It is not cache I/O. The timed feature load/memory-hit path is about
  `0.0004 s`.
- It is not primarily raster. Rendering both train views is about `0.19 s` in
  the slow default V-JEPA path.
- The main cost is model decode plus backward through that decode. The default
  V-JEPA path keeps `(1, 4608, 768) fp16` cached tokens, projects them to
  `(1, 4608, 128) fp32`, then repeatedly applies time-conditioned cross-attention
  for camera/path refinement across the 16 decode times. That puts most wall
  time in `model_forward_decode` (`~1.55 s`) and `backward` (`~0.53 s`).

Current no-V-JEPA baseline:

- The no-feature unconditioned TokenGS multicam config is the fast baseline to
  use when we want the same two-camera train/heldout-camera data contract without
  paying the V-JEPA memory-attention cost.
- It is still slower than same-source single-view TokenGS frame-throughput
  numbers because each multicam step renders two training cameras. It reaches
  `~58 target frames/s` on the exact multicam contract, which is close to the
  direct/token-only speed regime and much faster than the original V-JEPA
  baseline.
- It is a speed baseline, not yet a quality baseline. It needs an actual
  train/eval run before it can be added to `BASELINES.md` as a heldout-camera
  quality result.
