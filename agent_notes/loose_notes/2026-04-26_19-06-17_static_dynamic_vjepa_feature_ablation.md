# Static/Dynamic Split + V-JEPA Feature Ablation

## Context

After the temporal ablation suite, the 96 static / 32 dynamic token split was
the clear best local 16-frame 128px diagnostic. User asked to make a new one
that combines that approach with video features.

The implementation path was a new checked-in JSONC config plus launcher:

```bash
src/train_configs/local_mac_ablate_time_static_dynamic_96_32_crossattn4_precomputed_vjepa2_1_vitb_384_rgb_uniform_strong_video_implicit_128_fast_mac_8192splats.jsonc
src/train_scripts/train_static_dynamic_vjepa_features_ablation.sh
```

This uses the existing `train_precomputed_feature_implicit_dynamic.py` trainer
and V-JEPA 2.1 torchhub ViT-B/384 feature extractor.

## What Changed

- Added the new static/dynamic + precomputed V-JEPA config.
- Added a launcher that runs the precomputed-feature trainer with
  `PYTHONPATH=src/train`.
- Added `timm` and `einops` to project deps because torchhub V-JEPA requires
  them.
- Extended `collect_video_temporal_ablation_stats.py` to show
  `video_encoder_backend`, feature extractor, and feature model in the table.

## Caching Behavior

`PrecomputedFeatureImplicitTrainer.on_sequences_loaded()` builds
`VideoFeatureCache`, prebakes train+eval sequences, infers feature channels,
then releases the feature extractor before the normal training loop.

The successful cache file is:

```text
data/feature_cache/ablate_time_static_dynamic_vjepa2_1_vitb_384/b6ba09206f179d4c2cc29d52.pt
```

It is about 6.8 MB. With `force_rebake: false`, reruns with the same config and
source video should hit this file rather than recomputing V-JEPA.

Important correction: this feature-cache path bakes the whole loaded
`SequenceData`, not the 16-frame sampled training window. The first attempt used
`data.max_frames: 0`, so the V-JEPA bake tried all 46 loaded frames. Sampling the
process showed it inside MPS scaled dot product attention with about 17.6 GB
current and 25.5 GB peak physical footprint. That was not a good local-fast
baseline. The config now uses `data.max_frames: 16` and a distinct cache key:

```text
ablate-time-static-dynamic-96-32-vjepa2-1-vitb-384-small128-max16-v1
```

## Run

Command:

```bash
PYTHONPATH=src/train uv run python src/train/train_precomputed_feature_implicit_dynamic.py \
  src/train_configs/local_mac_ablate_time_static_dynamic_96_32_crossattn4_precomputed_vjepa2_1_vitb_384_rgb_uniform_strong_video_implicit_128_fast_mac_8192splats.jsonc
```

W&B:

```text
https://wandb.ai/nbardy/dynaworld/runs/oaor6um2
```

The run trained 250 steps with W&B `_runtime ~= 810s`. In this trainer,
`on_sequences_loaded()` runs before `wandb.init(...)`, so `_runtime` excludes
the V-JEPA feature prebake. It mostly measures model construction after W&B
init, step-0 validation, 250 train steps, validation media work, and final logs.

## Result

Final summary:

| run | eval loss | L1 | SSIM | PSNR | pred adj / GT adj | decoded XYZ adj | camera adj rot | dynamic motion | runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| static/dynamic local encoder | 0.1195 | 0.0779 | 0.4287 | 18.42 | 0.3408 | 0.0455 | 0.0159 deg | 0.0844 | 488s |
| static/dynamic + V-JEPA features | 0.0881 | 0.0615 | 0.6109 | 20.29 | 0.6322 | 0.0945 | 0.1309 deg | 0.0813 | 810s |

The feature version strongly improved same-source reconstruction and temporal
motion in this tiny diagnostic. It also moved camera more than the local encoder
version, so novel-view or held-out-clip validation is still needed before
calling this generally better.

Runtime note: the cached feature tensor is `vjepa_tokens: [1, 4608, 768]` in
float16. The trainable adapter projects those tokens to model width, then the
query decoder runs 4 cross-attention layers over the large memory. In the
static/dynamic path, it also refines camera queries once per decoded frame, so
the cached-feature path is slower even though the frozen V-JEPA network is not
in the optimizer loop.

## Verification

Passed:

```bash
PYTHONPATH=src/train uv run python -m py_compile \
  src/train/train_precomputed_feature_implicit_dynamic.py \
  src/train/train_video_token_implicit_dynamic.py \
  src/train_scripts/collect_video_temporal_ablation_stats.py
bash -n src/train_scripts/train_static_dynamic_vjepa_features_ablation.sh \
  src/train_scripts/train_video_temporal_ablation_suite.sh
git diff --check -- pyproject.toml uv.lock \
  src/train_configs/local_mac_ablate_time_static_dynamic_96_32_crossattn4_precomputed_vjepa2_1_vitb_384_rgb_uniform_strong_video_implicit_128_fast_mac_8192splats.jsonc \
  src/train_scripts/train_static_dynamic_vjepa_features_ablation.sh \
  src/train_scripts/collect_video_temporal_ablation_stats.py \
  src/train/train_video_token_implicit_dynamic.py
```

## Next

- Rerun from cache once to measure pure train-loop runtime without first-bake
  cost. Done in the longer-run follow-up: `mybv736f` hit the feature cache.
- Add clip-aware precomputed feature caching if we want `data.max_frames: 0`
  while training 16-frame windows.
- Try the same static/dynamic + features config on a scene-distinct clip set
  and on multi-camera validation once camera adapters are unified.

## Follow-Up: Longer Cached Run

Added:

```text
src/train_configs/local_mac_ablate_time_static_dynamic_96_32_crossattn4_precomputed_vjepa2_1_vitb_384_rgb_uniform_strong_video_implicit_128_fast_mac_8192splats_1000step.jsonc
```

The launcher now accepts:

```bash
./src/train_scripts/train_static_dynamic_vjepa_features_ablation.sh 250
./src/train_scripts/train_static_dynamic_vjepa_features_ablation.sh 1000
```

The 1000-step run was started from the existing feature cache and interrupted
after the step-500 video checkpoint because it had already answered the
"longer" question and the machine load was variable. W&B run:

```text
https://wandb.ai/nbardy/dynaworld/runs/mybv736f
```

Summary from the latest synced scalar checkpoint:

| run | eval loss | L1 | SSIM | PSNR | pred adj / GT adj | decoded XYZ adj | camera adj rot | runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 250-step V-JEPA split | 0.0881 | 0.0615 | 0.6109 | 20.29 | 0.6322 | 0.0945 | 0.1309 deg | 810s |
| cached longer V-JEPA split | 0.0547 | 0.0413 | 0.7836 | 23.69 | 0.8009 | 0.1305 | 0.1827 deg | 2424s |

Longer training helped a lot on the source-view fit. It also increased decoded
motion and camera adjacent rotation, so this remains a strong overfit baseline
and a hypothesis to validate, not proof of novel-view quality.

Infra note: cache-hit runs were still instantiating the V-JEPA extractor before
checking the `.pt`. `VideoFeatureCache` was patched to build extractors lazily
only on cache miss, so future cache-hit starts should skip the V-JEPA model-load
overhead as well as the feature forward.
