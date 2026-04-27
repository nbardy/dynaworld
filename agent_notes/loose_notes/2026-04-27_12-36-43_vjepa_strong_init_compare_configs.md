# V-JEPA Strong-Init Compare Configs

## Context

After diagnosing the recent V-JEPA matrix, Nicholas asked whether the strong
RGB/uniform/token/head init can be added to the V-JEPA run and whether that is
configurable.

## Answer

It is already configurable through model config keys:

- `query_token_init_std`
- `head_output_init_std`
- `rgb_init`
- `rgb_init_min`
- `rgb_init_max`
- plus the existing `scale_init_log_jitter`, `position_init_extent_coverage`,
  `rotation_init`, and `opacity_init`

The established strong-init values from the local and static/dynamic ablations
are:

```text
scale_init_log_jitter       0.7
opacity_init                0.1
query_token_init_std        0.8
head_output_init_std        0.12
position_init_extent_coverage 0.9
rotation_init               random
rgb_init                    uniform
rgb_init_min                0.0
rgb_init_max                1.0
```

## Changes

Added two sibling configs so the comparison is not one-sided:

- `src/train_configs/local_mac_compare_local_video_encoder_strong_init_16f_implicit_camera_128_fast_mac_8192splats.jsonc`
- `src/train_configs/local_mac_compare_vjepa2_vitl_fpc16_256_frozen_strong_init_16f_implicit_camera_128_fast_mac_8192splats.jsonc`

Updated `src/train_scripts/train_compare_vjepa2_fpc16_256_16f_single_overfit.sh`
with modes:

```bash
./src/train_scripts/train_compare_vjepa2_fpc16_256_16f_single_overfit.sh local-strong
./src/train_scripts/train_compare_vjepa2_fpc16_256_16f_single_overfit.sh vjepa-strong
./src/train_scripts/train_compare_vjepa2_fpc16_256_16f_single_overfit.sh strong
./src/train_scripts/train_compare_vjepa2_fpc16_256_16f_single_overfit.sh strong-matrix
```

`strong-matrix` runs local strong-init, V-JEPA strong-init, and the existing
unconditioned-token strong-init control.

## Verification

- `bash -n src/train_scripts/train_compare_vjepa2_fpc16_256_16f_single_overfit.sh`
- loaded both new JSONC configs through `resolve_config`
- confirmed both normalize to `query_token_init_std=0.8`,
  `head_output_init_std=0.12`, and `rgb_init=uniform 0.0..1.0`
- `git diff --check`

## Metric Plan

We do have a reason to use these values: local strong-init already improved
detail and SSIM in earlier ablations. But there is no measured V-JEPA
strong-init result yet, so the next evidence should be paired:

```bash
./src/train_scripts/train_compare_vjepa2_fpc16_256_16f_single_overfit.sh strong-matrix
```

If V-JEPA strong-init still loses to local/unconditioned, then the weak recent
V-JEPA result was not just initialization. If it catches up, rerun the
static/dynamic V-JEPA path and then add camera-clamp controls.
