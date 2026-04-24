# LTX Feature Cache Implicit Trainer

- Request: try an LTX-based architecture that keeps the useful frozen-video
  feature extraction path, but routes it through the existing implicit-camera
  style rather than replacing the camera branch with explicit camera inputs.
- Existing state: `train_video_token_implicit_dynamic.py` already had a
  multi-sequence video-token implicit trainer and an in-flight V-JEPA backend
  branch. The worktree was dirty before this pass, including that trainer and
  `dynamic_video_token_gs_implicit_camera.py`, so this pass layered changes on
  top rather than reverting anything.
- Implementation choice: use cached LTX native layer outputs as query-decoder
  memory tokens. This does not upsample LTX maps to a single image grid; each
  cached tensor stays at native token/grid shape and is projected to the
  model's `model_dim` before cross-attention. That matches the existing
  TokenGS implicit-camera contract better than grafting the pasted
  per-Gaussian UV sampler directly into the current model.
- Added `PrecomputedVideoFeatureAdapter` to the video implicit model. It
  accepts a mapping of layer names to tensors with shapes like `[B,N,C]`,
  `[B,C,H,W]`, or `[B,C,T,H,W]`, flattens native maps into tokens, adds
  trainable per-layer embeddings, and feeds the existing query decoder.
- Added trainer hooks so subclasses can provide a model input other than RGB
  clips while still using the same render/loss/camera regularization path.
- Added `video_feature_cache.py`: computes a per-sample cache key from the
  source sample fingerprint plus feature extraction semantics, including a
  human-controlled `features.sample_cache_key` for deliberate busting when
  layers/timesteps/preprocessing change. It stores `torch.save` payloads under
  `features.cache_dir`, loads hits, and bakes misses.
- Added two extractors: `ltx` is a Diffusers `LTXConditionPipeline` hook-based
  extractor with config-owned layer paths; `rgb_pyramid` is a deterministic
  local smoke-test extractor for validating cache/model plumbing without
  downloading LTX.
- Added `train_ltx_feature_implicit_dynamic.py`, which defaults the model to
  `video_encoder_backend=precomputed_ltx`, prebakes loaded train/eval samples
  before building the trainable adapter, infers `video_feature_channels` from
  the first successful cache bake when config leaves them null, then trains
  through the inherited implicit-camera loop.
- Added the first LTX config and wrapper:
  `src/train_configs/local_mac_overfit_ltx_feature_implicit_camera_128_4fps_fast_mac_8192splats.jsonc`
  and `src/train_scripts/train_ltx_feature_implicit_camera_128_4fps_fast_mac.sh`.
- Known caveat: the checked-in LTX layer paths are plausible Diffusers
  transformer block names, but they still need a real LTX model import to
  confirm against the installed Diffusers version. The extractor prints
  candidate module paths if a configured layer path misses.
- Follow-up merge: generalized the LTX-specific trainer into
  `train_precomputed_feature_implicit_dynamic.py`. The same trainer now accepts
  `features.extractor = "ltx" | "vjepa_hf" | "vjepa_torchhub" | "rgb_pyramid"`
  and always feeds cached tensors through the model's `precomputed` feature
  adapter. The old LTX-named entrypoint remains as a thin compatibility wrapper.
- Added a V-JEPA 2.1 torchhub cached-feature config:
  `src/train_configs/local_mac_overfit_precomputed_vjepa2_1_torchhub_vitb_384.jsonc`.
  It bakes raw `vjepa_tokens` once per sample under
  `data/feature_cache/vjepa2_1_torchhub_vitb_384` and uses
  `sample_cache_key = "vjepa2-1-vitb-384-tokens-v1"` for deliberate cache busts.
