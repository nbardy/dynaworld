# Video Model Bootstrap Architectures

Date: 2026-04-24

Scope: two frozen video-model bootstrap routes that sit under the patched
framing 3 contract. They are observation-side priors for building `W0`, not new
deploy-time inputs. The exported renderer still has the same contract:

```text
S_tau = G(W0, tau)
image = R_fixed(S_tau, c)
```

The cache/trainer boundary is intentionally shared with the existing
precomputed V-JEPA path:

```text
source video
  -> frozen video model feature bake
  -> VideoFeatureCache
  -> PrecomputedVideoFeatureAdapter
  -> DynamicTokenGSImplicitCamera
  -> fixed renderer loss
```

## Architecture A: LTX Conditioned Feature Cache

Use LTX-Video as a frozen diffusion transformer over a source video condition.
The input clip is passed through the Diffusers LTX condition pipeline, selected
transformer block outputs are hooked, and those hidden states are cached once
per sample.

Current repo surface:

```text
features.extractor = "ltx"
model.video_encoder_backend = "precomputed_ltx"
trainer = src/train/train_precomputed_feature_implicit_dynamic.py
config = src/train_configs/local_mac_overfit_ltx_feature_implicit_camera_128_4fps_fast_mac_8192splats.jsonc
```

Data flow:

```text
frames
  -> LTXConditionPipeline(video=frames, prompt, timestep, guidance_scale)
  -> hook transformer_blocks.{4,12,20}
  -> cache {layer_name: hidden_state}
  -> trainable layernorm/linear adapter
  -> query token cross-attention
  -> Gaussian heads
```

What this buys:

- Diffusion hidden states carry motion, appearance, and denoising-prior
  structure that a pure encoder may not expose.
- The timestep/guidance/layer choice gives a cheap axis for feature probing:
  early blocks may act more local, middle blocks may be motion/shape biased, and
  late blocks may be prompt/denoise biased.
- It is the fastest diffusion-prior experiment because LTX is the leanest of
  the two diffusion/editing routes and the repo already has this path.

What it does not buy yet:

- It does not attach features persistently to individual Gaussians. The current
  adapter treats cached hidden states as a memory token set.
- It does not make LTX part of deployment. LTX is only used while baking
  features for training/input encoding.
- It does not change the fixed-rasterizer boundary or the held-out image loss.

## Architecture B: Wan-VACE Editing-Control Feature Cache

Use Wan2.1-VACE-1.3B as the smallest official VACE editing backbone. The input
video is passed as VACE conditioning, a black mask marks every pixel/frame as
known, selected Wan transformer blocks are hooked, and those hidden states are
cached.

New repo surface:

```text
features.extractor = "wan_vace"
model.video_encoder_backend = "precomputed"
trainer = src/train/train_precomputed_feature_implicit_dynamic.py
config = src/train_configs/local_mac_overfit_wan_vace_feature_implicit_camera_128_4fps_fast_mac_8192splats.jsonc
```

Data flow:

```text
frames
  -> WanVACEPipeline(video=frames, mask=black, prompt, conditioning_scale)
  -> hook transformer.blocks.{2,6,10,14}
  -> cache {layer_name: hidden_state}
  -> trainable layernorm/linear adapter
  -> query token cross-attention
  -> Gaussian heads
```

Mask convention:

```text
black mask = known / conditioned region
white mask = generation region
```

For first-pass feature extraction, the mask is black everywhere because the goal
is to read the source-video editing representation, not to ask VACE to fill a
hole.

What this buys:

- Wan-VACE is trained for video editing/control, so its hidden states may encode
  object persistence, reference consistency, and mask-aware structure more
  directly than V-JEPA or plain LTX.
- Mask/reference/video conditioning semantics become explicit cache-key fields.
  That lets us later probe known-everywhere, masked-hole, first-last-frame, or
  reference-image variants without mixing caches.
- It gives an "official VACE" baseline below the 14B tier.

What it costs:

- It is much heavier than V-JEPA and likely heavier than LTX on local MPS.
- Hooks are version-sensitive until Diffusers exposes hidden-state returns.
- Wan's useful signal may be entangled with prompt, mask, conditioning scale,
  scheduler shift, and denoising timestep. Those all need ablations.

## How These Differ From What We Already Did

Existing local encoder:

```text
frames -> small repo-native encoder -> train end-to-end with splat renderer
```

This is the cleanest baseline. It learns from the held-out render loss only, but
has no pretrained video prior.

Existing V-JEPA online path:

```text
frames -> frozen V-JEPA encoder inside model forward -> splat decoder
```

This uses V-JEPA encoder output directly. It is semantically a video encoder:
one forward pass, no denoising loop, no prompt, no mask, no editing control.

Existing V-JEPA precomputed path:

```text
frames -> frozen V-JEPA encoder -> cached tokens -> splat decoder
```

This removes repeated encoder cost from training but keeps the same V-JEPA
feature semantics.

Current LTX path:

```text
frames -> frozen LTX diffusion hidden states -> cached tokens -> splat decoder
```

This is already present as a precomputed-feature variant. Its difference from
V-JEPA is upstream: diffusion transformer states instead of encoder tokens.
After cache bake, both are just feature mappings consumed by the same adapter.

New Wan-VACE path:

```text
frames + known mask -> frozen Wan-VACE editing hidden states -> cached tokens -> splat decoder
```

This adds editing-control semantics to the feature source. It is not merely a
larger V-JEPA; the mask, reference, conditioning scale, prompt, scheduler, and
VACE layer structure are part of the representation being tested.

Still separate future architecture: per-Gaussian lifted features.

```text
video feature grid -> project/sample through cameras -> persistent per-Gaussian feature vectors
```

That idea is different from the current cached-token adapter. It would make the
video-model prior more resolution-independent and more tightly attached to the
3D asset. The code added here does not implement that lift; it preserves the
shared token-memory path so LTX/Wan/V-JEPA can be compared before adding a new
Gaussian-side representation.

## First Falsification Tests

1. Cache bake import smoke:

```text
build_feature_extractor({"extractor": "wan_vace", ...})
```

This only proves the local Diffusers install exposes the needed classes.

2. Config normalization smoke:

```text
PrecomputedFeatureImplicitTrainer.resolve_config(wan_config)
```

This proves Wan-specific fields survive the JSONC schema boundary and enter the
feature fingerprint.

3. RGB-pyramid control:

```text
features.extractor = "rgb_pyramid"
```

If RGB cache improves as much as Wan/LTX, the video-model prior may not be doing
the work.

4. Layer ablation:

```text
Wan blocks: 2 vs 6 vs 10 vs 14
LTX blocks: 4 vs 12 vs 20
```

Hold renderer, model size, and data split fixed. The useful question is not
"does the model train?" but which layer reduces held-out render loss per cached
byte.

5. Wrong-world swap:

```text
cache(sample A) + target/render sample B
```

If the decoder ignores cached features, wrong-world swap will not hurt enough.

## Sources Checked

- Diffusers `pipeline_ltx_condition.py`: LTX condition pipeline accepts source
  video conditioning and exposes transformer modules suitable for hooks.
- Diffusers `pipeline_wan_vace.py`: Wan-VACE accepts `video`, `mask`,
  `reference_images`, `conditioning_scale`, and rounds frame counts to `4n+1`.
- Diffusers `transformer_wan.py`: Wan transformer blocks live under `blocks`.
