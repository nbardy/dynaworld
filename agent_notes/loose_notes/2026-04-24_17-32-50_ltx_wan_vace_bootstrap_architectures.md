# 2026-04-24 17:32:50 - LTX and Wan-VACE bootstrap architectures

## Context

The question was how to add two more video-model bootstrap approaches and how
they differ from the V-JEPA paths already added. Two subagents split the read:
one checked the existing LTX/precomputed-feature route, the other checked the
Wan-VACE route.

## Current Model

The lowest-risk design is not a new trainer per video backbone. The repo already
has the useful shared boundary:

```text
frozen feature extractor -> VideoFeatureCache -> PrecomputedVideoFeatureAdapter -> DynamicTokenGSImplicitCamera
```

LTX is already in that boundary. Wan-VACE should be added as a new
`features.extractor` so it can be compared against V-JEPA and LTX without moving
the splat decoder or the fixed renderer contract.

## Branches

Hypothesis:
    Diffusion hidden states help because denoising/editing pretraining encodes
    object and motion priors that V-JEPA encoder tokens do not.
What would make it false:
    RGB-pyramid cached features or shuffled diffusion features match the same
    held-out render improvements.
Cheap test:
    Run the same config with `rgb_pyramid`, V-JEPA, LTX, and Wan-VACE while
    holding renderer/model/data fixed.

Hypothesis:
    Wan-VACE helps specifically because editing-control training makes mask and
    reference semantics useful for world construction.
What would make it false:
    Known-everywhere Wan features do no better than LTX at equal cache budget,
    or performance disappears when prompt/mask fields are neutralized.
Cheap test:
    Known-everywhere mask first, then masked-hole and first-last-frame variants
    with cache keys that include mask semantics.

Hypothesis:
    The per-Gaussian feature-lift idea is a separate architecture, not a config
    change to the current cached-token adapter.
What would make it false:
    Query cross-attention over cached tokens already gives stable
    resolution-independent features and wrong-world swap is strongly sensitive.
Cheap test:
    Compare token-memory adapter vs projected per-Gaussian feature attachment
    on the same cached LTX/Wan tensors.

## Decisions

- Keep LTX as the existing `features.extractor="ltx"` path.
- Add Wan-VACE as `features.extractor="wan_vace"`.
- Cache Wan-specific semantics in the feature fingerprint: mask mode,
  conditioning scale, scheduler flow shift, second guidance scale, module root,
  max sequence length, and VAE dtype.
- Use a known-everywhere black mask for the first Wan-VACE feature bake.
- Do not add a permanent diffusion loss or any deploy-time side channel.

## Open Questions

- Does the local Diffusers install include `WanVACEPipeline` yet, or do we need
  a GitHub install like the V-JEPA/LTX work?
- Does Wan-VACE 1.3B fit on the available local hardware even for one-step
  feature baking?
- Are hidden states under `blocks.N` the best Wan layer surface, or should we
  hook VACE-specific submodules once a real bake shows their shapes?
- Does a per-Gaussian lift beat the current token-memory adapter after the
  basic feature-backbone comparison is done?
