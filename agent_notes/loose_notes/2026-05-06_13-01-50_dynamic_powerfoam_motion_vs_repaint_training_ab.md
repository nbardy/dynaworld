# Dynamic PowerFoam Motion Vs Repaint Training A/B

Date: 2026-05-06 13:01 +07

## Why

The useful evidence for dynamic foam is not just unit tests. We need a fit on a
dynamic video with per-frame quality metrics and an explicit control separating
geometry motion from per-cell color repaint.

## What Changed

- `src/train/train_dynamic_powerfoam_metal.py` now logs per-frame reconstruction
  metrics at each eval:
  - `eval_frame_psnr_mean`
  - `eval_frame_psnr_min`
  - `eval_frame_snr_mean`
  - `eval_frame_snr_min`
  - `per_frame_metrics_step_XXXX.json`
- Added fixed-geometry color-only config:
  `src/train_configs/local_mac_dynamic_powerfoam_metal_rbf_color_only_fixed_geometry_video_1024_16f_40step_smoke.jsonc`
- Added comparator:
  `research_experiments/dynamic_foam/compare_dynamic_powerfoam_motion_vs_repaint.py`

## Runs

Geometry-only dynamic branch:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:third_party/dynamic-powerfoam-metal \
  WANDB_MODE=disabled .venv/bin/python -u src/train/train_dynamic_powerfoam_metal.py \
  src/train_configs/local_mac_dynamic_powerfoam_metal_rbf_geometry_only_video_1024_16f_40step_smoke.jsonc
```

Fixed-geometry color-only branch:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:third_party/dynamic-powerfoam-metal \
  WANDB_MODE=disabled .venv/bin/python -u src/train/train_dynamic_powerfoam_metal.py \
  src/train_configs/local_mac_dynamic_powerfoam_metal_rbf_color_only_fixed_geometry_video_1024_16f_40step_smoke.jsonc
```

Comparison:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/dynamic_foam/compare_dynamic_powerfoam_motion_vs_repaint.py \
  --geometry-summary outputs/dynamic_powerfoam_metal/local_mac_dynamic_powerfoam_metal_rbf_geometry_only_video_1024_16f_40step_smoke \
  --color-only-summary outputs/dynamic_powerfoam_metal/local_mac_dynamic_powerfoam_metal_rbf_color_only_fixed_geometry_video_1024_16f_40step_smoke \
  --output outputs/dynamic_powerfoam_metal/motion_vs_repaint_comparison_1024_16f_40step_20260506.json
```

## Results

Saved comparison:

```text
outputs/dynamic_powerfoam_metal/motion_vs_repaint_comparison_1024_16f_40step_20260506.json
```

Geometry-only, dynamic features frozen:

```text
mean PSNR/SNR 19.5667 / 14.8959
min  PSNR/SNR 18.2504 / 13.5512
eval L1/MSE   0.06514 / 0.01122
screen motion mean/p95 px 0.4446 / 1.0526
alpha/support delta 0.00957 / 0.00793
temporal feature delta 0.0
worst frame 4
```

Color-only, fixed geometry:

```text
mean PSNR/SNR 19.3554 / 14.6847
min  PSNR/SNR 17.3692 / 12.6700
eval L1/MSE   0.06108 / 0.01212
screen motion mean/p95 px 0.0 / 0.0
alpha/support delta 0.0 / 0.0
temporal feature delta 0.00427
worst frame 4
```

Geometry minus color-only:

```text
mean PSNR +0.2112 dB
min PSNR  +0.8812 dB
mean SNR  +0.2112 dB
min SNR   +0.8812 dB
L1        +0.00407 (worse)
MSE       -0.000896 (better)
```

## Interpretation

For this small explicit-video fit, geometry motion is not just a test artifact:
it improves mean and worst-frame PSNR/SNR versus a fixed-geometry repaint
control. The color-only branch gets lower average L1, so we should not claim a
universal quality win from one scalar. The more useful claim is: geometry motion
improves squared-error / SNR robustness and the worst frame while changing
alpha/support; the repaint control changes features only and leaves geometry and
support fixed.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  src/train/train_dynamic_powerfoam_metal.py \
  research_experiments/dynamic_foam/compare_dynamic_powerfoam_motion_vs_repaint.py
```

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/dynamic_foam/verify_dynamic_powerfoam_geometry_run.py \
  outputs/dynamic_powerfoam_metal/local_mac_dynamic_powerfoam_metal_rbf_geometry_only_video_1024_16f_40step_smoke \
  --require-geometry-motion --require-alpha-support-motion --require-appearance-freeze-control
```

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:third_party/dynamic-powerfoam-metal \
  uv run --with pytest python -m pytest -p no:cacheprovider \
  tests/test_dynamic_powerfoam_metal.py -q -rs
```

Result: `15 passed`.
