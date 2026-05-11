# Alpha BG Bleed + Features

Date: 2026-05-08 16:25

## Context

The current feature-kernel thread raised a real concern: if the F32 feature
rasterizer uses `feature_background = 0.0`, could the colorize MLP learn that
zero/low-norm features mean background and absorb loss that should move splats
outward? The user remembered an older line of work where this was fixed by
exposing the alpha mask, logging alpha-mask videos, and applying random
background outside the feature MLP.

This note records the recovered stack so future shader forks do not regress it.

## Current Belief

The old fix was not feature-space randomization. It was alpha-aware RGB-space
composition after colorization:

```text
splat_rgb = colorize(rendered_features)
final_rgb = alpha * splat_rgb + (1 - alpha) * rgb_background
```

With this contract, a fully empty pixel (`alpha = 0`) does not train the MLP at
all because `d final_rgb / d splat_rgb = alpha`. The MLP may numerically see a
zero feature tensor, but the reconstruction gradient on that pixel is zeroed
before it reaches the colorizer. The geometry/opacity path still receives
`dL/dalpha`, so non-background GT pixels push splats outward.

This means the earlier answer "random RGB does not prevent the color MLP from
recognizing zero features" was incomplete. More precise: random RGB does not
hide the zero feature value from the MLP forward pass, but alpha-aware
composition prevents fully transparent pixels from training the MLP on that
zero feature value. Low-alpha edge pixels still pass a small colorizer gradient
proportional to alpha.

## Recovered Commit Stack

### 1. Pre-refactor trainer implementation

Commit:

```text
9b68192bee3aeffae484e0f78968541d98f3c1c6
Checkpoint dynaworld before trainer cleanup refactor
```

Important files:

- `src/train/train_video_token_implicit_dynamic.py`
- `src/train/renderers/fast_mac.py`
- `src/train/colorize.py`
- `tests/test_fast_mac_feature_background.py`
- `TODO/alpha_mask_white_background_cheating.md`
- `agent_notes/loose_notes/2026-04-29_22-00-00_feature_splatting_alpha_aware_composition_session.md`

The old trainer sampled one random RGB background per training step and used it
for all chunks:

```python
random_bg = torch.rand(3, device=clip_frames.device, dtype=clip_frames.dtype).view(1, 3, 1, 1)
...
splat_rgb = self.colorize_features(chunk_features, tuple(decoded.cameras[chunk_start:chunk_end]))
alpha_expanded = chunk_alpha.unsqueeze(1)
chunk_renders = alpha_expanded * splat_rgb + (1.0 - alpha_expanded) * random_bg
```

The note from 2026-04-29 states the root cause clearly: before alpha-aware
composition, feature-space background pixels fed a constant feature vector into
the MLP, and the MLP bias learned the missed-pixel color instead of forcing
splats to cover those pixels.

### 2. Rasterizer alpha output

Submodule commit:

```text
235150c5041463b8a094fc9e638bf69276862ba3
Add v5 feature rasterizer variant
```

Important files:

- `third_party/fast-mac-gsplat/variants/v5_features/torch_gsplat_bridge_v5_features/rasterize.py`
- `third_party/fast-mac-gsplat/variants/v5_features/csrc/metal/gsplat_v5_features_kernels.metal`
- `third_party/fast-mac-gsplat/variants/v5_features/tests/alpha_output_check.py`

Implemented behavior:

- F!=3 returns `(features, alpha)`.
- Feature-channel background defaults to `(0.0,)`.
- Alpha is `1 - T_final`.
- Backward accepts `grad_alpha` and routes that signal into means/conics/
  opacities, so alpha-aware loss can move geometry.
- The implementation was motivated by the synthetic-channel equivalence:
  alpha is the same as rendering an extra channel with splat value `1.0` and
  background `0.0`, but without learning gradients for that synthetic color.

### 3. Shared objective refactor

Commit:

```text
37f5f0be13efa70835b47a8cbf31f32f559d5ac7
Land objective/ refactor scaffold and delete dead trainer lanes
```

Important files:

- `src/train/objective/background.py`
- `src/train/objective/objective.py`
- `src/train/objective/types.py`
- `tests/test_objective_background_and_composition.py`
- `tests/test_rgb_recon_objective.py`

This generalized the old trainer-local fix into `RGBReconObjective`. Current
`compose_rgb` still has the same contract:

```python
alpha = rasterized.alpha.unsqueeze(1).to(device=splat_rgb.device, dtype=splat_rgb.dtype)
bg = background.rgb.to(device=splat_rgb.device, dtype=splat_rgb.dtype)
return alpha * splat_rgb + (1.0 - alpha) * bg
```

Current tests verify the critical gradient relationship:

```python
composed.sum().backward()
assert splat_rgb.grad == alpha.unsqueeze(1).expand_as(splat_rgb)
```

So for `alpha=0`, the colorizer side gets no reconstruction gradient.

## Current Code State

The recovered fix is present in the current shared trainer stack:

- `src/train/objective/objective.py` composes alpha after colorize.
- `src/train/objective/background.py` defaults train background to
  `random_rgb` and eval background to `white`.
- `src/train/train_video_token_implicit_dynamic.py` samples one background for
  the train step and asserts that F-channel training must return alpha.
- `src/train/train_multicam_precomputed_feature_implicit_dynamic.py` and
  `src/train/train_multicam_relative_pose_implicit_dynamic.py` also assert
  alpha exists for F-channel training.
- `src/train/pipeline/validation_media.py` logs alpha-mask, feature-PCA,
  composite videos, and multicam grids.

Therefore `v6_refined_features_f32_zero_bg` and the new
`v9_features_gradcache_zero_bg` are not inherently alpha-cheating regressions
as long as the objective path still receives alpha and composes random RGB after
colorize.

## What Zero BG Means Here

`render.fast_mac.feature_background = 0.0` is only the feature-rasterizer fill
value. It should not be confused with the loss/composition background.

The zero-background shader fork skips the final feature-background tail add
only when the feature background is exactly zero. It still writes alpha and it
does not change the RGB random-background objective. This is compatible with
the old anti-cheat fix.

## Remaining Risks

### R1. Low-alpha edge bleed

The old fix makes the colorizer gradient proportional to alpha, not exactly
zero unless alpha is zero. Very low-alpha edge pixels can still train the MLP a
little on weak/blurred feature mixtures. This is probably part of the edge-ring
blur pathology, but it is not the catastrophic original MLP-bias shortcut.

Cheap test:

```text
Log alpha quantiles and colorizer grad contribution by alpha bucket:
  alpha in [0, .01), [.01, .05), [.05, .2), [.2, 1]
```

Supported if low-alpha buckets carry significant colorizer grad or feature
reconstruction error while geometry stays sparse.

### R2. Fixed eval white is safe for eval, not train

The old TODO `TODO/alpha_mask_white_background_cheating.md` found that white
background composition can become a different cheat: bright sky/cloud pixels
can be explained by `(1 - alpha) * white`. That is why train mode must be
random RGB. Eval can stay white for comparability, but alpha masks must be
diagnosed separately.

Cheap test:

```text
alpha_to_gt_background_correlation =
  corr(1 - alpha, distance(GT, eval_background) < threshold)
```

High correlation means the background composite is explaining visible scene
content.

### R3. Feature-space normalization from PowerFoam is separate

PowerFoam later added:

```python
color_features = features / alpha.unsqueeze(1).clamp_min(eps)
rgb = colorizer(color_features)
final = alpha * rgb + (1 - alpha) * background
```

That unpremultiplies before colorization. It is useful when features are
premultiplied and alpha is meaningful per cell, but it was not the original
F32-v5_features fix. Porting it to this trainer is an ablation, not a required
restoration.

## Decision Implications

1. Do not block zero-bg shader benchmarking on feature-space randomization.
   The restored key contract is alpha-aware RGB composition after colorize.
2. Before promoting any faster feature shader, run trainer-path fixed-render
   parity against stable `v6_refined_features`: feature, alpha, RGB, loss, and
   gradients must match.
3. The next objective-path improvement should be diagnostic first:
   add alpha-to-background-color correlation and/or colorizer-gradient-by-alpha
   bucket logging.
4. If edge blur persists, test PowerFoam-style `normalize_features_by_alpha` as
   a named ablation, not as an assumed missing fix.

## Open Questions

- Does the current multicam F32 run show low-alpha correlation with sky/edge
  brightness, or is the blur mostly camera/raster resolution?
- Does `pre_norm=true` in `FeatureToColor` amplify low-alpha feature mixtures
  by normalizing their channel variance?
- Would `normalize_features_by_alpha` improve edge sharpness or destabilize
  colorizer inputs where alpha is tiny?
- Should validation media include a "background-only" diagnostic column:
  `(1 - alpha) * bg`, next to GT, render, alpha, and feature PCA?

