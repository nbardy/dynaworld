# Init Diagnostics And RGB Uniform Bias Handoff

## Context

The thread was about random-init health for DynaWorld overfit configs. The user
wanted measurements and opt-in code updates without changing baseline configs.
The specific symptom was weak color diversity and low same-split inter-token
variance at initialization.

Relevant files touched in this pass:

- `src/train/init_diagnostics.py`
- `src/train/probe_init_diagnostics.py`
- `src/train/gs_models/blocks.py`
- known-camera and video-token model/config plumbing for new RGB init knobs
- `tests/test_init_diagnostics.py`
- `src/train_configs/local_mac_ablate_init_rgb_uniform_strong_video_implicit_128_fast_mac_8192splats.jsonc`

## Current Model

The low inter-token variance is not one bug. It is two separable effects:

1. `position_init_extent_coverage` initializes per-split output biases. That
   gives broad within-token XYZ coverage immediately, but those biases are
   shared by every token, so same-split cross-token variance stays low unless
   token-dependent output weights are strong enough.
2. RGB previously had no per-split bias init. With output bias near zero and a
   sigmoid decode, colors start near 0.5 and only move by the small token/head
   product. This measured as RGB range around 0.45-0.55.

The practical fix is to keep RGB in RGB/sigmoid space, not switch to HSV/LAB,
and add opt-in per-split RGB bias initialization plus an opt-in stronger
token/head scale for inter-token spread.

Exact 0 or 1 RGB values are not representable by finite logits. The new
`rgb_init_min=0.0`, `rgb_init_max=1.0` path samples a continuous target and
uses a finite logit clamp only for numerical safety. In practice the 8192-splat
probe reached about 0.0008 to 0.99996.

## Measurements

Existing 16-frame local video-token implicit-camera baseline, seed 0:

```text
RGB min/max/std: 0.4465 / 0.5505 / 0.0113
RGB entropy01: 0.2317
RGB within-token range mean: 0.0528
RGB same-split cross-token std mean: 0.0105
XYZ same-split cross-token std mean: 0.0407
XYZ cross/within ratio: 0.0462
```

Same config with only `rgb_init="uniform", rgb_init_min=0.0, rgb_init_max=1.0`:

```text
RGB min/max/std: 0.0011 / 0.9999 / 0.2975
RGB entropy01: 0.9847
RGB within-token range mean: 0.9723
RGB same-split cross-token std mean: 0.0068
XYZ unchanged
```

Same config with only `query_token_init_std=0.8`, `head_output_init_std=0.12`:

```text
RGB min/max/std: 0.2255 / 0.7696 / 0.0631
RGB entropy01: 0.5592
RGB within-token range mean: 0.2871
RGB same-split cross-token std mean: 0.0591
XYZ same-split cross-token std mean: 0.2286
XYZ cross/within ratio: 0.2548
```

Combined ablation:

```text
RGB min/max/std: 0.0008 / 0.99996 / 0.2986
RGB entropy01: 0.9915
RGB within-token range mean: 0.9716
RGB same-split cross-token std mean: 0.0386
XYZ same-split cross-token std mean: 0.2286
XYZ cross/within ratio: 0.2548
```

## Code Changes

- Added optional `rgb_init="uniform"` to `GaussianParameterHeads`.
- Threaded `rgb_init`, `rgb_init_min`, and `rgb_init_max` through known-camera
  TokenGS and video-token implicit/known-camera models.
- Left defaults as `None`/`0.01`/`0.99`, so existing baseline configs do not
  change behavior unless they opt in.
- Extended the init probe to support current video-token arch names:
  `tokengs_video_implicit_camera` and `tokengs_video_known_camera`.
- Added an actionable failure for precomputed feature configs whose feature
  channels are still null; they need prebaked/inferred channels before model
  construction can be probed.
- Added RGB min/max/entropy and RGB spread metrics to the compact summary.
- Added a separate ablation config rather than editing baselines.

## Tests And Probes

Passing:

```text
PYTHONPATH=src/train uv run python -m py_compile \
  src/train/gs_models/blocks.py \
  src/train/gs_models/dynamic_token_gs.py \
  src/train/gs_models/dynamic_video_token_gs_implicit_camera.py \
  src/train/dynamicTokenGS.py \
  src/train/train_video_token_implicit_dynamic.py \
  src/train/init_diagnostics.py \
  src/train/probe_init_diagnostics.py

uv run --with pytest pytest -q tests/test_init_diagnostics.py tests/test_postprocess_dof.py
# 4 passed

PYTHONPATH=src/train uv run python src/train/probe_init_diagnostics.py \
  src/train_configs/local_mac_ablate_init_rgb_uniform_strong_video_implicit_128_fast_mac_8192splats.jsonc --seed 0
```

`uv run pytest` without `--with pytest` failed because pytest is not installed
in the project environment.

## Next

Run the new ablation config against the same 250-step overfit target and compare
early render behavior against the unchanged baseline:

```text
PYTHONPATH=src/train uv run python src/train/train_video_token_implicit_dynamic.py \
  src/train_configs/local_mac_ablate_init_rgb_uniform_strong_video_implicit_128_fast_mac_8192splats.jsonc
```

Falsification criteria:

- If early colors look noisy but PSNR/SSIM improve faster, keep the RGB uniform
  path and sweep token/head scale down.
- If color noise destabilizes geometry or opacity, try `rgb_init_min=0.01`,
  `rgb_init_max=0.99` or reduce `head_output_init_std` to 0.09.
- If inter-token geometry remains too correlated after training starts, add a
  direct token-correlation metric over decoded splats after each validation
  interval, not just at init.
