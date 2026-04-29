# Training Queue: Video Model Bootstrap

Date: 2026-04-24

Status: planning only. Do not treat this file as evidence that any run below
has been started.

## Gates Before Any Run

1. Fix feature-frame loading so LTX/Wan/V-JEPA feature bakes can read source
   frames at feature resolution instead of resized training tensors.
2. Add feature-bundle metadata to the cache: native layout, frame indices,
   feature resolution, source resolution, timestep/layer, and model-conditioning
   fields.
3. Decide whether hook captures are one-step only or keyed by layer plus
   timestep/step.
4. Run non-training smoke checks for config load, extractor construction, cache
   hit/miss, and feature shape inference.

## Tier 0: Controls And Sanity Checks

These are not meant to win; they tell us whether the training loop and feature
adapter are behaving.

| Run | Config / surface | Purpose | Success signal |
| --- | --- | --- | --- |
| Tiny local video-token smoke | `src/train_configs/local_mac_tiny_30_video_token_smoke.jsonc` | Confirm 30-clip data path, renderer, W&B logging, and eval loop still work. | No crashes; eval video nonblank; loss moves. |
| 128px implicit-camera baseline | `src/train_configs/local_mac_overfit_video_token_implicit_camera_128_4fps_fast_mac_8192splats.jsonc` | Baseline for all feature-prior comparisons. | Same or better trajectory than prior local baseline. |
| RGB-pyramid precomputed control | `features.extractor="rgb_pyramid"` with precomputed trainer | Test whether generic cached features already explain gains. | If this matches V-JEPA/LTX/Wan, the video prior claim is weak. |
| Wrong-world swap diagnostic | cached features from sample A with targets from sample B | Verify decoder depends on cached features. | Loss/video degrades sharply. |

## Tier 1: Existing Frozen Encoder Baselines

These isolate V-JEPA as a discriminative video encoder, separate from diffusion
hidden states.

| Run | Config / surface | Purpose | Success signal |
| --- | --- | --- | --- |
| V-JEPA 2.1 Base online | `src/train_configs/local_mac_overfit_video_token_implicit_camera_vjepa2_1_torchhub_vitb_384.jsonc` | Measure online frozen encoder path. | Beats local encoder at equal decoder/render budget or gives better early convergence. |
| V-JEPA 2.1 Base precomputed | `src/train_configs/local_mac_overfit_precomputed_vjepa2_1_torchhub_vitb_384.jsonc` | Same feature semantics without repeated encoder cost. | Same quality trend as online with lower train-loop overhead. |
| Fast-start V-JEPA fpc16/256 A/B | `src/train_scripts/train_compare_vjepa2_fpc16_256_16f_single_overfit.sh` | First single-video side-by-side: local non-pretrained 16f encoder vs frozen HF V-JEPA 2 ViT-L fpc16/256. | V-JEPA reaches lower full-video eval loss or better W&B video faster than local encoder. |
| V-JEPA 2.1 Base 64f/384 A/B | `local_mac_compare_*_64f_implicit_camera_128_fast_mac_8192splats.jsonc` | Cleaner base-model comparison after the fast-start result. | Confirms whether V-JEPA helps when using the 2.1 384px base checkpoint. |
| V-JEPA layer/clip ablation | new configs from precomputed V-JEPA base | Compare 16/32/64 frame feature bakes and crop sizes. | Find cheapest feature payload that still improves held-out render quality. |

## Tier 2: Diffusion Hidden-State Feature Priors

These test whether diffusion/editing hidden states are better input-side priors
than V-JEPA encoder tokens.

| Run | Config / surface | Purpose | Success signal |
| --- | --- | --- | --- |
| LTX one-step hidden-state baseline | `src/train_configs/local_mac_overfit_ltx_feature_implicit_camera_128_4fps_fast_mac_8192splats.jsonc` | Current lean diffusion-prior route. | Better early held-out loss/video than V-JEPA or RGB control at comparable decoder settings. |
| LTX layer sweep | same config, layers one-at-a-time: `4`, `12`, `20`, then pairs | Identify which LTX block is useful. | One layer/pair dominates per cached byte. |
| LTX timestep sweep | same config, `timestep` variants | Check whether low/mid/high noise features differ materially. | Clear best timestep band; no ambiguous cache semantics. |
| Wan-VACE known-video baseline | `src/train_configs/local_mac_overfit_wan_vace_feature_implicit_camera_128_4fps_fast_mac_8192splats.jsonc` | Small official VACE editing-prior route. | Quality gain justifies bake cost over LTX/V-JEPA. |
| Wan-VACE layer sweep | Wan blocks one-at-a-time: `2`, `6`, `10`, `14`, then pairs | Find useful VACE hidden-state surface. | Later/mid blocks show stronger world-token signal than early blocks. |
| Wan-VACE conditioning sweep | known mask, masked-hole mask, first/last-frame mask | Test whether VACE editing/control semantics matter. | Mask/reference variants change quality predictably, not just randomly. |

## Tier 3: Architecture Comparison Runs

These compare model-side choices with the same data, renderer, and feature
source.

| Run | Config / surface | Purpose | Success signal |
| --- | --- | --- | --- |
| Joint implicit-camera vs separated-camera | existing `joint_attention` / `separated_camera` variants where supported | Test whether camera estimation should share token memory with scene features. | One variant consistently improves camera stability and render metrics. |
| Pose-to-Plucker baseline | `src/train_configs/local_mac_overfit_video_token_implicit_camera_128_4fps_fast_mac_8192splats_pose_to_plucker.jsonc` | Check if explicit ray conditioning helps the same renderer/model scale. | Better camera/reconstruction tradeoff than legacy orbit head. |
| Sinusoidal time-path baseline | `src/train_configs/local_mac_overfit_video_token_implicit_camera_128_4fps_fast_mac_8192splats_sinusoidal_time.jsonc` | Test whether the time path, not feature source, is the bottleneck. | Better temporal stability without higher static reconstruction loss. |
| Static/dynamic splat-bank variant | current static/dynamic fork config when finalized | Separate background geometry from dynamic content. | Dynamic regions improve without degrading static background. |

## Tier 4: Dataset Scale-Up

These should wait until Tier 0/1/2 establish which feature source is worth
paying for.

| Run | Config / surface | Purpose | Success signal |
| --- | --- | --- | --- |
| Local 30-clip baseline | `src/train_scripts/train_local_mac_30_clip_baseline.sh` | Scene-diverse train/eval sanity once data split is accepted. | Eval examples nonblank; train/test gap measurable. |
| Local 30-clip V-JEPA precomputed | new config from V-JEPA precomputed base plus 30-clip manifest | Check whether V-JEPA scales past single-video overfit. | Better eval trend than local encoder. |
| Local 30-clip LTX precomputed | new config from LTX base plus 30-clip manifest | Check whether diffusion prior survives scene diversity. | Beats RGB and V-JEPA controls on held-out clips. |
| Local 30-clip Wan-VACE precomputed | new config from Wan base plus 30-clip manifest | Decide if VACE bake cost is justified. | Strongest eval quality, or retire for cost. |
| Scene-distinct curated spans | `src/dataset_configs/youtube_curated_spans_64_4fps_16f.jsonc` after data validation | Test generalization on less duplicated scenes. | Scene-held-out eval improves over local 30-clip split. |

## Tier 5: Per-Gaussian Feature Lift

These wait for the feature-bundle refactor, because projection requires reliable
layout metadata.

| Run | Surface | Purpose | Success signal |
| --- | --- | --- | --- |
| LTX token-memory vs per-Gaussian lift | same cached LTX tensors, two adapters | Test whether attaching features to splats beats global token memory. | Per-Gaussian lift improves resolution/generalization or converges faster. |
| V-JEPA grid lift | V-JEPA patch/tubelet grid plus camera projection sampler | Test discriminative feature lift without diffusion cost. | Better wrong-world sensitivity and lower held-out loss. |
| Wan-VACE grid/control lift | Wan hidden-state layout plus known/masked variants | Test whether editing-control features attach cleanly to 3D assets. | Mask/control variants produce interpretable per-Gaussian feature changes. |

## Run Ordering

Recommended order once the non-training refactors land:

1. RGB-pyramid precomputed control.
2. Fast-start single-video 16f local vs V-JEPA fpc16/256 A/B.
3. Current local implicit-camera 128px baseline.
4. V-JEPA precomputed base.
5. LTX one-step base.
6. Wan-VACE one-step known-video base.
7. Layer sweeps only for whichever feature source beats RGB/local.
8. 30-clip scale-up for the winner plus V-JEPA control.
9. Per-Gaussian lift after feature-bundle metadata is real.

## Logging Requirements

- Every real run logs W&B videos at a useful cadence, not only scalar loss.
- Every run records config path, git commit, renderer mode, effective splat
  count, feature cache key, and feature source model id.
- Compare by full-video eval metrics and videos, not final sampled train loss.
- Keep cache bake time and cache byte size in the run note; feature quality per
  cached byte matters.
