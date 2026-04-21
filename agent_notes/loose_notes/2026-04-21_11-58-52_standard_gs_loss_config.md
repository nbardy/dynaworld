# Standard GS Loss Config

## Context

The prebaked-camera trainer still used the old inline reconstruction objective:

```text
L1 + 0.2 * MSE
```

The user asked to make the standard 3DGS reconstruction objective configurable and use it by default:

```text
0.8 * L1 + 0.2 * D-SSIM
```

## Changes

- Added `src/train/losses.py` with differentiable image reconstruction losses:
  - `standard_gs`
  - `l1_mse`
  - `l1`
  - `mse`
- Implemented local SSIM/D-SSIM over `[B, C, H, W]` tensors using an odd local averaging window.
- Added `losses` config normalization to `src/train/dynamicTokenGS.py`.
- Default normalized loss is now:

```jsonc
{
  "type": "standard_gs",
  "l1_weight": 0.8,
  "dssim_weight": 0.2,
  "mse_weight": 0.2,
  "ssim_window_size": 11,
  "ssim_c1": 0.0001,
  "ssim_c2": 0.0009
}
```

- Updated the current 128px Taichi 8192-splat config to include an explicit `losses` section using `standard_gs`.
- Evaluation now logs `Eval/Loss` using the configured reconstruction loss. It still logs `Eval/L1`, `Eval/MSE`, and `Eval/PSNR`; for `standard_gs` it also logs `Eval/DSSIM`.

## Validation

- `py_compile` passed for `src/train/losses.py` and `src/train/dynamicTokenGS.py`.
- A tensor-level standard-GS loss check produced finite per-image losses and finite gradients.
- `git diff --check` passed for touched files.

## Caveat

No full train smoke was run during this change because the machine was on very low battery and under heavy load. The loss itself compiled and backpropagated on synthetic tensors.
