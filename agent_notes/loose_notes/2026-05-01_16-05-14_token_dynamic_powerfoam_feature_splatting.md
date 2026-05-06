# Token Dynamic PowerFoam Feature Splatting

## What Changed

- Added `token_rbf_features` to `src/train/train_dynamic_powerfoam_metal.py`.
- The new path uses one learned token per PowerFoam cell. A small MLP decodes raw canonical cell state and Gaussian RBF temporal coefficients for:
  - centers / depth
  - radii
  - densities
  - F-channel texel features
  - normals / material tangents
  - texel sites
- The Metal renderer still does the rasterization/backward pass through `rasterize_power_foam_oriented_texel_surface`.
- Added `FeatureToColor` after rasterization for token feature mode.
- Added alpha-normalization before colorization: `features / alpha.clamp_min(eps)`.
- Added random RGB train backgrounds and fixed eval backgrounds to the dynamic PowerFoam trainer. The checked-in token feature config uses `train_background_mode=random_rgb` to prevent transparent-pixel opacity cheats.

## Config

New config:

```text
src/train_configs/local_mac_token_dynamic_powerfoam_features_F16_1024_smoke.jsonc
```

This is a same-source renderer-development diagnostic on `test_data/test_video_small_128_4fps.mp4`, not a held-out-camera benchmark.

## Validation

Commands run:

```bash
PYTHONPATH=src/train /Users/nicholasbardy/git/gsplats_browser/dynaworld/.venv/bin/python -m py_compile \
  src/train/train_dynamic_powerfoam_metal.py \
  tests/test_dynamic_powerfoam_metal.py \
  tests/test_powerfoam_direct.py

PYTHONPATH=src/train /Users/nicholasbardy/git/gsplats_browser/dynaworld/.venv/bin/python - <<'PY'
from tests.test_dynamic_powerfoam_metal import (
    test_dynamic_powerfoam_rbf_decode_has_temporal_grads,
    test_dynamic_powerfoam_per_frame_smooth_decode_has_temporal_loss,
    test_token_dynamic_powerfoam_features_decode_has_token_grads,
    test_feature_colorizer_identity_and_background_composition,
    test_random_background_sampler_bounds_and_shape,
)
for fn in [
    test_dynamic_powerfoam_rbf_decode_has_temporal_grads,
    test_dynamic_powerfoam_per_frame_smooth_decode_has_temporal_loss,
    test_token_dynamic_powerfoam_features_decode_has_token_grads,
    test_feature_colorizer_identity_and_background_composition,
    test_random_background_sampler_bounds_and_shape,
]:
    fn()
PY

PYTHONPATH=src/train WANDB_MODE=offline /Users/nicholasbardy/git/gsplats_browser/dynaworld/.venv/bin/python \
  src/train/train_dynamic_powerfoam_metal.py \
  src/train_configs/local_mac_token_dynamic_powerfoam_features_F16_1024_smoke.jsonc
```

`pytest` was not available in the repo venv (`No module named pytest`), so the focused test functions were invoked directly.

## Result

The 120-step MPS run completed and wrote:

```text
outputs/dynamic_powerfoam_metal/local_mac_token_dynamic_powerfoam_features_F16_1024_smoke/
```

Final metrics:

- wall: 7.29 s train loop
- eval L1: 0.03436
- eval MSE: 0.00381
- eval alpha mean: 0.99047
- eval feature std: 0.56778

Artifacts:

- `preview_step_0000.png`
- `preview_step_0060.png`
- `preview_step_0120.png`
- `render_step_0000.mp4`
- `render_step_0120.mp4`
- `side_by_side_step_0000.mp4`
- `side_by_side_step_0120.mp4`
- `checkpoint_final.pt`

## Notes

- The new token feature baseline lands between the naive shared-RBF dynamic PowerFoam and the direct oriented texel-surface PowerFoam diagnostics at this tiny 64px same-source setting.
- The alpha mean rises near 1.0 under random background, which is the intended anti-cheat pressure: empty pixels must pay reconstruction loss against random colors.
- This is still a diagnostic baseline, not a full official PowerFoam reproduction and not a held-out-camera result.

## Follow-Up Audit

After the first F16 run, an edge-case issue was fixed in `render_features_to_rgb`: RGB-direct foam outputs are premultiplied by alpha, so enabling a background for direct RGB mode must unpremultiply before compositing. The F16 token-feature path was already colorizing normalized features, but the helper contract is now correct for both paths.

Additional checks run:

```bash
PYTHONPATH=src/train /Users/nicholasbardy/git/gsplats_browser/dynaworld/.venv/bin/python -m py_compile \
  src/train/train_dynamic_powerfoam_metal.py \
  tests/test_dynamic_powerfoam_metal.py

PYTHONPATH=src/train /Users/nicholasbardy/git/gsplats_browser/dynaworld/.venv/bin/python - <<'PY'
from tests.test_dynamic_powerfoam_metal import (
    test_dynamic_powerfoam_rbf_decode_has_temporal_grads,
    test_dynamic_powerfoam_per_frame_smooth_decode_has_temporal_loss,
    test_token_dynamic_powerfoam_features_decode_has_token_grads,
    test_feature_colorizer_identity_and_background_composition,
    test_rgb_direct_background_composition_uses_unpremultiplied_color,
    test_random_background_sampler_bounds_and_shape,
    test_token_dynamic_powerfoam_features_mps_raster_backward_smoke,
)
for fn in [
    test_dynamic_powerfoam_rbf_decode_has_temporal_grads,
    test_dynamic_powerfoam_per_frame_smooth_decode_has_temporal_loss,
    test_token_dynamic_powerfoam_features_decode_has_token_grads,
    test_feature_colorizer_identity_and_background_composition,
    test_rgb_direct_background_composition_uses_unpremultiplied_color,
    test_random_background_sampler_bounds_and_shape,
    test_token_dynamic_powerfoam_features_mps_raster_backward_smoke,
]:
    fn()
PY
```

The new MPS smoke backpropagates through the actual Metal rasterizer, token decoder, and colorizer with random background. It passed.

## F32 Run

Added:

```text
src/train_configs/local_mac_token_dynamic_powerfoam_features_F32_1024_smoke.jsonc
```

Local offline F32 run:

- wall: 6.37 s train loop
- eval L1: 0.03465
- eval MSE: 0.00382
- eval alpha mean: 0.99216

Online W&B F32 run:

```text
https://wandb.ai/nbardy/dynaworld/runs/0v67kicc
```

- wall: 6.27 s train loop
- eval L1: 0.03427
- eval MSE: 0.00381
- eval alpha mean: 0.99226

The online run used the F32 config with only logging/output-dir overridden at launch so videos and alpha were uploaded to W&B.

Final canonical entrypoint check:

```bash
PYTHONPATH=src/train WANDB_MODE=offline /Users/nicholasbardy/git/gsplats_browser/dynaworld/.venv/bin/python \
  src/train/train.py \
  src/train_configs/local_mac_token_dynamic_powerfoam_features_F32_1024_smoke.jsonc
```

This completed through the public `train.py` dispatcher with eval L1 `0.03436`, eval MSE `0.00380`, and eval alpha mean `0.99151`.

## Motion-Honesty Audit

The user suspected that the dynamic PowerFoam feature run was mostly a fixed
grid learning color over time, not particles moving through the video. That
was correct.

I added decoded-state motion metrics to `train_dynamic_powerfoam_metal.py`:

- mean / p95 temporal XY delta
- mean temporal Z delta
- mean / p95 projected screen delta in pixels
- mean temporal feature absolute delta

After adding those metrics, the standard checked-in F32 config was re-run
offline through the public dispatcher:

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train.py \
  src/train_configs/local_mac_token_dynamic_powerfoam_features_F32_1024_smoke.jsonc
```

Final step 120:

- eval L1: `0.03446`
- eval MSE: `0.00381`
- eval alpha mean: `0.99204`
- mean temporal XY delta: `0.00114`
- p95 temporal XY delta: `0.00354`
- mean temporal screen delta: `0.039 px/frame`
- p95 temporal screen delta: `0.109 px/frame`
- mean temporal feature abs delta: `0.02370`

So the standard feature-foam fit is effectively a fixed image-grid/lattice
with time-varying features. The center parameters do drift from their initial
values, but the measured frame-to-frame projected motion is far below one
pixel. The low L1 is not evidence of learned object motion.

I then added a deliberately harsher motion probe:

```text
src/train_configs/local_mac_token_dynamic_powerfoam_features_F32_1024_motion_probe.jsonc
```

This disables dynamic densities and dynamic features, while keeping dynamic
centers/radii/normals/texel-sites enabled. The point is not to win quality; it
is to test whether the representation can be forced to explain the video by
moving cells instead of recoloring them.

Offline motion-probe result:

- eval L1: `0.08194`
- eval MSE: `0.01680`
- eval alpha mean: `0.99841`
- mean temporal XY delta: `0.14355`
- p95 temporal XY delta: `0.44820`
- mean temporal screen delta: `4.27 px/frame`
- p95 temporal screen delta: `13.22 px/frame`
- mean temporal feature abs delta: `0.0`

Online W&B motion-probe run:

```text
https://wandb.ai/nbardy/dynaworld/runs/xk5hwatb
```

Final online metrics:

- eval L1: `0.08901`
- eval MSE: `0.01899`
- eval alpha mean: `0.98611`
- mean temporal XY delta: `0.10556`
- p95 temporal XY delta: `0.31514`
- mean temporal screen delta: `2.92 px/frame`
- p95 temporal screen delta: `8.46 px/frame`
- mean temporal feature abs delta: `0.0`

Conclusion: the current token dynamic PowerFoam feature path is implemented and
trainable, but the default same-source RGB objective is not motion-identifying.
When time-varying features/density are available, it learns appearance changes
on a nearly fixed grid. When those appearance channels are frozen, the cells do
move, but the reconstruction gets much worse.

The next rigorous version needs a motion-identifying constraint or curriculum:

- optical-flow or track loss on projected centers / rendered alpha support
- held-out camera or multiview evaluation so a fixed image lattice cannot win
- staged training: geometry/motion first, then unfreeze dynamic features
- less UV-grid-like initialization, e.g. blue-noise/random/flow-seeded centers
- reporting guard: do not call a same-source run "dynamic" unless temporal
  screen motion is above a threshold and feature/density deltas are not doing
  all the work

## 512px Same-Instance Ratio Probe

The next idea was to keep the same number of instances but raise render/loss
resolution to 512px, reducing the instance-to-pixel ratio:

- 1024 cells at 512x512 = 1 cell per 256 pixels
- 4 texel sites per cell = 4096 texel sites = 1 texel site per 64 pixels

Added:

```text
src/train_configs/local_mac_token_dynamic_powerfoam_features_F32_1024_512px_smoke.jsonc
```

Important caveat: this keeps the same source path,
`test_data/test_video_small_128_4fps.mp4`. The trainer resizes targets to
`render.render_size`, so this is a 128px source upsampled to 512px. It is a
useful cell/pixel-ratio stress test, but not a true higher-detail data run.

One-step MPS probe succeeded:

- step 0 eval L1: `0.06673`
- one train step L1: `0.06345`
- one train step wall: `1.78 s`

Full offline 120-step run:

- train-loop wall: `38.90 s`
- eval L1: `0.04635`
- eval MSE: `0.00613`
- eval alpha mean: `0.99155`
- mean temporal screen delta: `0.349 px/frame`
- p95 temporal screen delta: `0.766 px/frame`
- mean temporal feature abs delta: `0.02850`

Online W&B run:

```text
https://wandb.ai/nbardy/dynaworld/runs/4fpwpb5p
```

Final online metrics:

- train-loop wall: `45.69 s`
- eval L1: `0.04645`
- eval MSE: `0.00612`
- eval alpha mean: `0.99176`
- mean temporal screen delta: `0.285 px/frame`
- p95 temporal screen delta: `0.651 px/frame`
- mean temporal feature abs delta: `0.02848`

Conclusion: raising render/loss resolution from 64 to 512 with the same 1024
cells does not fix the fixed-grid repaint mode. It reduces the instance/pixel
ratio, but the model still uses dynamic features substantially and the
projected motion is tiny relative to image width. The 512px mean temporal
screen delta is about `0.00056` of image width, close to the 64px run's
normalized `0.00061`; in normalized screen terms it did not become more
dynamic.

## Field-Specific Optimizer Audit

The user's hunch about texel-site LR being a useful signal exposed a real
implementation issue in the token dynamic path. The official PowerFoam config
uses very different optimizer groups: points, densities, radii, quaternions,
texel sites, spherical-view axes/RGB, and texel heights all train at separate
rates. Our direct dynamic module already had separate groups, but
`TokenDynamicPowerFoamFeatures.optimizer_param_groups()` returned only:

- `tokens`
- one monolithic `decoder`

That meant config knobs such as `point_lr_multiplier`,
`feature_lr_multiplier`, `temporal_lr_multiplier`, and
`texel_site_lr_multiplier` were not actually controlling the decoded field
heads in the token path.

I split the token decoder into:

- shared `decoder_trunk`
- one head per decoded raw field chunk

`optimizer_param_groups()` now emits per-field groups such as
`decoder_raw_xy0`, `decoder_raw_features0`, `decoder_raw_texel_sites0`,
`decoder_raw_features_coeff`, and `decoder_raw_texel_sites_coeff`. The LR for
each group is the product of the base train LR, `decoder_lr_multiplier`, and
the relevant field/temporal multipliers. For example:

```text
raw_texel_sites_coeff LR =
  train.lr * decoder_lr_multiplier * temporal_lr_multiplier * texel_site_lr_multiplier
```

Added a regression check:

```text
test_token_dynamic_powerfoam_features_optimizer_groups_use_field_lrs
```

Focused tests run after the change:

```bash
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/train_dynamic_powerfoam_metal.py \
  tests/test_dynamic_powerfoam_metal.py \
  tests/test_powerfoam_direct.py

PYTHONPATH=src/train .venv/bin/python - <<'PY'
from tests.test_dynamic_powerfoam_metal import (
    test_dynamic_powerfoam_rbf_decode_has_temporal_grads,
    test_dynamic_powerfoam_per_frame_smooth_decode_has_temporal_loss,
    test_token_dynamic_powerfoam_features_decode_has_token_grads,
    test_token_dynamic_powerfoam_features_optimizer_groups_use_field_lrs,
    test_feature_colorizer_identity_and_background_composition,
    test_rgb_direct_background_composition_uses_unpremultiplied_color,
    test_random_background_sampler_bounds_and_shape,
    test_temporal_motion_metrics_report_screen_motion,
    test_token_dynamic_powerfoam_features_mps_raster_backward_smoke,
)

for fn in [
    test_dynamic_powerfoam_rbf_decode_has_temporal_grads,
    test_dynamic_powerfoam_per_frame_smooth_decode_has_temporal_loss,
    test_token_dynamic_powerfoam_features_decode_has_token_grads,
    test_token_dynamic_powerfoam_features_optimizer_groups_use_field_lrs,
    test_feature_colorizer_identity_and_background_composition,
    test_rgb_direct_background_composition_uses_unpremultiplied_color,
    test_random_background_sampler_bounds_and_shape,
    test_temporal_motion_metrics_report_screen_motion,
    test_token_dynamic_powerfoam_features_mps_raster_backward_smoke,
]:
    fn()
PY
```

`pytest` is still not installed in this venv, so focused test functions were
called directly.

Post-fix standard F32 rerun:

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train.py \
  src/train_configs/local_mac_token_dynamic_powerfoam_features_F32_1024_smoke.jsonc
```

Final step 120:

- train-loop wall: `7.10 s`
- eval L1: `0.03462`
- eval MSE: `0.00381`
- eval alpha mean: `0.99198`
- mean center delta: `0.03356`
- mean XY delta from init: `0.00396`
- mean texel-site delta: `0.13833`
- mean temporal screen delta: `0.011 px/frame`
- p95 temporal screen delta: `0.021 px/frame`
- mean temporal feature abs delta: `0.02478`

So the optimizer bug was real, but simply honoring the existing LR multipliers
does not fix the fixed-grid repaint behavior. The standard F32 setup still
finds a low-loss solution mostly through time-varying features and densities.

## Texel-Site LR Probe

Added:

```text
src/train_configs/local_mac_token_dynamic_powerfoam_features_F32_1024_texel_site_lr_probe.jsonc
```

This probe deliberately turns off dynamic features and dynamic densities,
keeps centers/radii/normals/texel-sites dynamic, reduces feature LR, and makes
texel sites much faster:

- `dynamic_features=false`
- `dynamic_densities=false`
- `token_feature_residual_scale=0.03`
- `token_texel_site_residual_scale=0.4`
- `token_temporal_residual_scale=0.8`
- `point_lr_multiplier=0.2`
- `feature_lr_multiplier=0.02`
- `texel_site_lr_multiplier=2.0`
- `temporal_lr_multiplier=0.5`

Offline 120-step result:

- train-loop wall: `8.08 s`
- eval L1: `0.06457`
- eval MSE: `0.01157`
- eval alpha mean: `0.99214`
- mean center delta: `0.04158`
- mean XY delta from init: `0.03363`
- max center delta: `0.43084`
- mean texel-site delta: `0.74895`
- mean temporal screen delta: `0.286 px/frame`
- p95 temporal screen delta: `0.823 px/frame`
- mean temporal feature abs delta: `0.0`

Online W&B run:

```text
https://wandb.ai/nbardy/dynaworld/runs/0o32ezia
```

Final online metrics:

- train-loop wall: `6.53 s`
- eval L1: `0.06437`
- eval MSE: `0.01140`
- eval alpha mean: `0.99241`
- mean center delta: `0.04611`
- p95 center delta: `0.12490`
- max center delta: `0.84113`
- mean XY delta from init: `0.04037`
- mean Z delta from init: `0.01615`
- mean texel-site delta: `0.72582`
- mean temporal XY delta: `0.01232`
- p95 temporal XY delta: `0.03403`
- mean temporal screen delta: `0.379 px/frame`
- p95 temporal screen delta: `1.045 px/frame`
- mean temporal feature abs delta: `0.0`
- mean temporal coeff abs: `0.15506`

Conclusion: the LR hint was good. Once field-specific LR actually applies,
raising texel-site LR while freezing dynamic appearance makes texel sites and
centers move substantially more. But reconstruction gets much worse than the
standard F32 repainting run (`0.06437` L1 vs `0.03427` to `0.03462` L1), so
texel-site LR alone is not enough. The next useful curriculum is probably:

- train geometry/texel motion with dynamic feature/density channels frozen or
  very weak
- unfreeze feature dynamics only after projected motion has crossed a minimum
  threshold
- keep logging temporal screen motion and temporal feature/density deltas as
  anti-cheat diagnostics

## Rollback: Split-Head Token Decoder

2026-05-02 13:29:12 +07: the split-head token decoder / field-specific
optimizer patch was rolled back after review. It did expose that the token path
was not honoring field LR multipliers, but the measured result did not justify
the added implementation complexity:

- the standard F32 post-fix audit still had only `0.011 px/frame` mean temporal
  screen motion
- the texel-site LR probe moved more, but degraded to `0.06437` eval L1
- the core failure mode remained: RGB same-source training still prefers
  repainting features over physically meaningful motion

Active code is back to the lean token decoder:

- one `nn.Sequential` decoder MLP
- one `tokens` optimizer group
- one `decoder` optimizer group

The texel-site LR probe config is retained only as a historical run artifact.
Current token-rbf code no longer treats its per-field LR multipliers as active.
