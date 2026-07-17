# STAR UVT feature tube investigation

## Context

User asked for a feature-splatting / UVT feature world-tubes pass without touching
core trainer files. Scope was read-only investigation plus isolated notes or
prototypes, no long GPU jobs, and no baseline claims beyond `BASELINES.md`.

## What is current today

- Same-view precomputed V-JEPA F32 scale config:
  `src/train_configs/local_mac_scale_static_dynamic_vjepa_1k_video_pretrain_F32_256_16f_8192splats.jsonc`.
  It resolves to `arch=precomputed_feature_implicit_camera`, V-JEPA torchhub
  `vjepa2_1_vit_base_384`, `vjepa_feature_dim=768`, `vjepa_crop_size=384`,
  model `feature_dim=32`, `frames=16`, `image_size=256`, fast-mac
  `feature_variant=v5_features`, and `alpha_threshold=1/128`.
- Heldout-view / multicam precomputed V-JEPA F32 scale config:
  `src/train_configs/local_mac_scale_static_dynamic_vjepa_multicam_train2_holdout1_F32_256_16f_8192splats.jsonc`.
  It resolves to `arch=multicam_precomputed_feature_implicit_camera`, model
  `feature_dim=32`, fast-mac `feature_variant=v5_features`, and
  `alpha_threshold=1/255`. This is a separate loader/trainer family from the
  same-view manifest route.
- Multires heldout-view F32 config verified:
  `src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_vjepa_full_relpose_features_F32_multires64_128_256_512_tokenbudget_world4_fast_16f_8192splats_goodset_train0006_0014_holdout0005_alpha1_128_relpose_outputinit012.jsonc`.
  It resolves to `arch=multicam_relative_pose_implicit_camera`, model
  `feature_dim=32`, fast-mac `feature_variant=v5_features`,
  `alpha_threshold=1/128`, `multires_render_sizes=[64,128,256,512]`,
  probabilities `[0.25,0.45,0.25,0.05]`, and token detail levels `[0,0,1,2]`.
- The 300-clip 64f 256-to-512 config
  `src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_gpt256_8192splats_300clips_3kstep_multires_256to512.jsonc`
  is V-JEPA-conditioned RGB Gaussian training, not F32 feature splatting. It
  resolves to `feature_dim=3`, no colorizer, base `render_size=512`, schedule
  `256 -> 512` at step 2400, and RGB `rgb_variant=v6_refined`.

## Feature-splatting path

- `src/train/train.py` routes `precomputed_feature_implicit_camera`,
  `multicam_precomputed_feature_implicit_camera`, and
  `multicam_relative_pose_implicit_camera` separately.
- `src/train/train_precomputed_feature_implicit_dynamic.py` builds
  `VideoFeatureCache`, prebakes or loads V-JEPA tokens, and feeds cached tokens
  as model input. That is cache-side V-JEPA conditioning; it is distinct from a
  differentiable prediction-side V-JEPA feature loss.
- `src/train/renderers/fast_mac.py` dispatches `rgbs.shape[-1] == 3` to RGB
  variants and `F != 3` to `_rasterize_features_projected`.
- `src/train/rendering.py` returns `(features, alpha)` for fast-mac F-channel
  splatting. `src/train/objective/objective.py` requires a colorizer for
  `F != 3`, colorizes `[K,F,H,W]` through `FeatureToColor`, then alpha-composes
  RGB against the sampled background before reconstruction loss.

## Fastest feature shader evidence

The configs still point at `v5_features`. The fastest documented opt-in path is
`v11_features_gradcache_zero_bg_hostmeta_fixedbin`, and both the v5 and v11
Python bridge modules import in the current environment. Prior local notes
describe v11 as the best opt-in row, but not a safe default: it keeps the
grad-cache F32 backward path and removes one host allocation sync, while the
fixedbin ID buffer is no-overflow only and costs bounded memory. The bottleneck
is still backward and feature/loss surfaces, not simply forward projection.

## STAR UVT feature-tube conclusion

STAR UVT is RGB-only right now. `train_star_uvt_video_overfit.py` delegates to
`star_uvt_v0` where `ScreenTimeTubeModel.raw_color` is `[N,3]`,
`WorldTubeBatch.color` validates `[N,3]`, `UVTRenderConfig.background` is RGB,
and the Metal kernels are full of `float3`, `grad_rgb`, `grad_color`, and
`color_base = id * 3` assumptions. F32 feature tubes are not a config flip.

Minimal useful route:

1. Prove the feature-image contract with a dense prototype: UVT tubes carry
   `[N,F]` feature vectors, render `[T,F,H,W]` plus alpha, run existing
   `FeatureToColor`, compose RGB, and optimize RGB reconstruction.
2. If the tiny dense overfit learns, fork a feature-specific STAR variant
   rather than mutating `star_uvt_v0`. Add `feature_dim`, feature background,
   `[N,F]` feature tensors, `[T,H,W,F]` output and gradient tensors, and reducers
   with `grad_feature_samples` width `F`.
3. Start with direct atomic / `index_add` for throughput probes, because that is
   the current fast STAR exploration valve. Do not use it as deterministic
   promotion evidence.
4. Keep colorization outside the renderer initially. Existing F32 Gaussian
   evidence already points at image-space colorize/loss handoff rather than
   tile-loop colorize fusion.

## Prototype added

Added `research_experiments/star_uvt_feature_tubes/dense_feature_tube_prototype.py`.
It is intentionally not wired into `src/train/train.py`; it is a tiny CPU-first
gradient contract for feature-valued tubes and `FeatureToColor`.
