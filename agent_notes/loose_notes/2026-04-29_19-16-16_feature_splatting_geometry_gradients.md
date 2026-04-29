# Feature Splatting Geometry-Gradient Check

## Trigger

Feature-splatting runs showed a centered cluster of splats and a persistent
background/border region. RGB F=3 runs used to move/expand splats more readily.

## Checks

- Compared current feature path against RGB F=3 path in:
  - `src/train/gs_models/dynamic_video_token_gs_implicit_camera.py`
  - `src/train/renderers/fast_mac.py`
  - `src/train/train_video_token_implicit_dynamic.py`
  - `third_party/fast-mac-gsplat/variants/v5_features/`
- Ran a one-off v5 vs v5_features F=3 forward/backward parity check with the
  same random output gradient. Forward max abs was `0.0`; gradients for means,
  conics, colors, and opacities matched up to float noise.
- Ran one-step gradient probes on the unconditioned token configs using the same
  seed/clip.

## Finding

The obvious issue is not a gross rasterizer backward break. It is the F32
colorizer initialization and feature-space background scale.

Default F32 before the config change:

- raster feature std: `0.01739`
- RGB output std: `0.03084`
- model grad sums: `xyz=0.00730`, `scale=0.00016`, `opacity=0.00077`

RGB F=3 baseline under the same probe:

- RGB/raster std: `0.09060`
- model grad sums: `xyz=0.07545`, `scale=0.00388`, `opacity=0.01796`

The F32 default therefore started with geometry gradients roughly one to two
orders weaker. This matches the visual symptom: colorizer/features can learn
first, while xyz/scale/opacity have little pressure to move into the background.

Turning on feature-channel LayerNorm before the 1x1 colorizer fixes the weak
gradient scale in the probe. With `pre_norm=true`, Kaiming init, gain `2.0`:

- RGB output std: `0.11992`
- model grad sums: `xyz=0.21711`, `scale=0.00559`, `opacity=0.01699`

## Change

Updated `src/train_configs/local_mac_unconditioned_tokens_features_F32.jsonc` to
use:

```jsonc
"pre_norm": true,
"weight_init": "kaiming",
"weight_init_gain": 2.0
```

This leaves the renderer and geometry code unchanged and just makes the default
F32 experiment start with an actually informative colorizer.

## Caveat

The feature-space background remains `0.0`. That is still the right simple
feature-space background for now, but it means geometry gradients depend on
feature/background contrast after the colorizer Jacobian. If future runs still
show slow background fill, the next targeted knob is a feature initialization
scale or an RGB-seeded first-three-feature initialization, not a renderer rewrite.
