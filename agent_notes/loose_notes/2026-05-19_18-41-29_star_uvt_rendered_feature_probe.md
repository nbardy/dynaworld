# STAR UVT rendered-feature RGB probe

## Context

After the full-resolution autograd RGB-aux probe-init bridge from sparse 1500
failed, the next open question was whether the failure came from copying a
target-grid-trained RGB probe into the wrong full-res rendered-feature
distribution.

## What changed

Added a narrow diagnostic trainer:

`src/train/train_star_uvt_rendered_feature_rgb_probe.py`

It freezes a STAR UVT checkpoint, renders sparse full-res pixels from the actual
rendered feature-image distribution, and trains only a hidden64
`FeatureToColor` probe against RGB at those sampled pixels. It is routed through
`src/train/train.py` as `arch=star_uvt_rendered_feature_rgb_probe`.

The first config is:

`src/train_configs/star_uvt_rendered_feature_rgb_probe_from1500_sparsepixels_hidden64_lr01_100step_media.jsonc`

It loads:

`outputs/checkpoints/2026-05-19_star_uvt_feature_targetgrid_sparseforward_batchedvjp_lr001_resume50_from1450_lr005sparse_1500step.pt`

and samples the same target-grid source lattice as the sparse-forward target
path: `65,536` pixels per step, `0.390625%` dense.

## Run

Command:

```bash
PYTHONPATH=src/train .venv/bin/python src/train/train.py \
  src/train_configs/star_uvt_rendered_feature_rgb_probe_from1500_sparsepixels_hidden64_lr01_100step_media.jsonc
```

W&B offline:

`wandb/offline-run-20260519_184129-gr9ngyl7`

## Result

- pass: `true`
- sparse sample loss: `0.168261 -> 0.099014`
- sparse sample PSNR: `7.740 -> 10.043`
- final full-video loss: `0.245710`
- final full-video PSNR: `6.096`
- mean step/render/backward: `241.44 / 110.22 / 31.68 ms`
- last step/render/backward: `245.49 / 103.67 / 29.92 ms`
- dense media render: `1571.96 ms`

Artifacts:

- `outputs/benchmarks/2026-05-19_star_uvt_rendered_feature_rgb_probe_from1500_sparsepixels_hidden64_lr01_100step_media.json`
- `outputs/benchmarks/2026-05-19_star_uvt_rendered_feature_rgb_probe_from1500_sparsepixels.md`
- `outputs/checkpoints/2026-05-19_star_uvt_rendered_feature_rgb_probe_from1500_sparsepixels_hidden64_lr01_100step.pt`
- `outputs/media/2026-05-19_star_uvt_rendered_feature_rgb_probe_from1500_sparsepixels_hidden64_lr01_100step_contact.jpg`
- `outputs/media/2026-05-19_star_uvt_rendered_feature_rgb_probe_from1500_sparsepixels_hidden64_lr01_100step_sbs.mp4`

## Interpretation

This diagnostic answers the immediate distribution-mismatch question. Training
the probe on rendered feature pixels is fast and the sampled loss falls smoothly,
but dense media still becomes high-frequency sparse streaks and full-video PSNR
is only `6.096`. That is only slightly above the failed dense autograd
probe-init bridge (`5.851` RGB PSNR), so simply adapting the colorizer to the
rendered feature-image distribution is not enough.

The current sparse 1500 feature/alpha field does not expose a clean full-res
visual basis to a per-pixel colorizer under this sparse sampled RGB supervision.
The next useful gate is either a denser/stratified sparse-pixel visual probe to
rule out sampling bias or a native sparse visual/probe VJP that can move STAR
features directly without paying dense autograd cost.

## Validation

- `PYTHONPATH=src/train .venv/bin/python -m py_compile src/train/train_star_uvt_rendered_feature_rgb_probe.py src/train/train.py`
- `PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_star_uvt_feature_rgb_probe.py -q`
  - `5 passed`
