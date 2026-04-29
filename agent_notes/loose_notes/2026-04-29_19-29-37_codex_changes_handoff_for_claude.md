# Codex Changes Handoff for Claude: F32 Colorize Config vs Alpha Hypothesis

Date: 2026-04-29

Purpose: document exactly what Codex changed around the F32 feature-splatting
config, so Claude/other agents can decide whether it conflicts with the
alpha-aware-background diagnosis.

## TL;DR

- The LayerNorm implementation was already present in `src/train/colorize.py`.
- Codex's later diagnosis change only enabled that existing colorizer path in
  the default F32 config:

  ```jsonc
  "pre_norm": true,
  "weight_init": "kaiming",
  "weight_init_gain": 2.0
  ```

- This does **not** implement alpha-aware RGB-space background compositing.
- This does **not** add an alpha output to `v5_features`.
- This does **not** change `feature_background`; it remains `0.0`.
- This does **not** change F=3/v5 baseline behavior.

The config change is best understood as an init/gradient-conditioning tweak, not
the root fix for the centered-splat/background problem.

## Exact Relevant Config Change

File:

```text
src/train_configs/local_mac_unconditioned_tokens_features_F32.jsonc
```

Current `colorize` block:

```jsonc
"colorize": {
  "hidden_dim": null,
  "activation": "sigmoid",
  "pre_norm": true,
  "weight_init": "kaiming",
  "weight_init_gain": 2.0,
  "view_condition": "none",
  "detach_view_condition": true
}
```

Prior default behavior was effectively:

```jsonc
"pre_norm": false,
"weight_init": "kaiming",
"weight_init_gain": 1.0
```

or those keys absent, since `FeatureToColor` defaults to those values.

## Why Codex Changed It

Codex ran a one-step gradient probe on the same seed/clip comparing F=3 and F32.
The old F32 default had much smaller geometry gradients than the F=3 baseline:

```text
F3 baseline:
  xyz=0.07545, scale=0.00388, opacity=0.01796

F32 before enabling pre_norm/gain:
  xyz=0.00730, scale=0.00016, opacity=0.00077

F32 after pre_norm=true, kaiming, gain=2:
  xyz=0.21711, scale=0.00559, opacity=0.01699
```

This suggested the old F32 colorizer was attenuating gradients where splats
already touched pixels. Enabling per-pixel feature LayerNorm made the colorizer
input scale sane and restored non-tiny initial geometry gradients.

## Why This Does Not Resolve Claude's Alpha/Background Diagnosis

Claude's diagnosis is structural and still stands:

```text
F=3 path:
  rgb = sum_i T_i * alpha_i * rgb_i + T_final * white

F32 path today:
  features = sum_i T_i * alpha_i * feat_i + T_final * feature_background
  rgb = colorize(features)
```

Alpha compositing in the rasterizer was not removed. The problem is that the
background feature vector is routed through `colorize`. With `pre_norm=true`, a
constant background feature vector maps to approximately zero normalized input,
then `sigmoid(bias) ~= 0.5`. Without LayerNorm it still maps to one fixed color.
Either way, empty pixels no longer preserve the old white-background loss.

So the Codex config change can improve gradient scale inside/near covered
regions, but it cannot restore the old "missed pixels are white and costly"
incentive. If all splats miss a region, there is still no direct geometry
gradient from that pixel except through Gaussian tails and thresholds.

## Does This Conflict With Alpha-Aware Composition?

No direct conflict.

The alpha-aware fix should work with or without `pre_norm=true`:

```python
features, alpha = rasterize(...)
splat_rgb = colorize(features)
final_rgb = alpha * splat_rgb + (1 - alpha) * white
```

However, for a controlled experiment, do not let this config change blur the
comparison:

- To test the structural alpha hypothesis alone, run one alpha-aware experiment
  with the old colorize defaults (`pre_norm=false`, gain `1.0`) and one with the
  current normalized colorizer.
- If only one can be run, prefer current normalized colorizer as the better F32
  training recipe, but label the run clearly.

## Rollback Surface

If Claude wants to revert Codex's config-only tweak, the rollback is local to the
F32 config:

```jsonc
"colorize": {
  "hidden_dim": null,
  "activation": "sigmoid",
  "view_condition": "none",
  "detach_view_condition": true
}
```

or explicitly:

```jsonc
"pre_norm": false,
"weight_init": "kaiming",
"weight_init_gain": 1.0
```

No renderer files need to be reverted for this specific change.

## Other Relevant Codex/Agent Changes Nearby

Separate from the config tweak above:

- `src/train/colorize.py` has configurable `pre_norm`, `weight_init`,
  `weight_init_gain`, and `view_condition`.
- `src/train/train_video_token_implicit_dynamic.py` has view-direction plumbing
  for `colorize.view_condition` modes:
  - `none`
  - `camera_center_ray`
  - `pixel_ray`
- `src/train/renderers/fast_mac.py` has separate RGB background and
  feature-space background handling:
  - F=3 uses `background` and original v5.
  - F!=3 uses `feature_background` and v5_features.
- `tests/test_fast_mac_feature_background.py` checks that F!=3 does not inherit
  RGB white background as `[1] * F`.

These are orthogonal to the alpha-aware composition fix, except that alpha-aware
composition will need to decide whether the background is handled in:

- feature space before colorize, current behavior, or
- RGB space after colorize, proposed fix.

## Recommended Next Decision

Do not spend more time tuning `pre_norm`/gain as the primary fix. Keep the
normalized colorizer as a useful training recipe unless a controlled comparison
needs the old default.

The high-value next experiment is still alpha-aware RGB-space composition,
either via:

1. marker-channel alpha extraction for a fast hypothesis test, or
2. clean `v5_features` alpha output for the long-term implementation.
