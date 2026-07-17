# STAR UVT Visibility-Proxy Support Gate

## Context

The previous trainer gate proved that `visibility_proxy` is first-class in the
STAR UVT feature trainer, but it did not prove dense visual quality. The missing
question was whether the proxy actually changes alpha/support.

## What changed

Added a stronger follow-up config:

`src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_visibility_proxy10x_from1500_lr001_20step_media.jsonc`

It uses the same selected sparse step-1500 checkpoint and target-grid/probe
setup as the 5-step gate, but raises `visibility_proxy.loss_weight` from
`0.001` to `0.01` and runs 20 steps.

## Runs

The 10x/20-step command was:

```bash
PYTHONPATH=src/train:. rtk .venv/bin/python \
  src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_visibility_proxy10x_from1500_lr001_20step_media.jsonc
```

The shell command was piped through `tee`, which masked the nonzero trainer
assertion exit. The trainer output itself is authoritative: `_assert_requirements`
failed on loss decrease.

The dense support diagnostic compared:

- `start1500`
- `visibility5`
- `visibility10x20`

with raw opacity biases `-2,-1,0,1,2,3,4`.

## Artifacts

- `outputs/benchmarks/2026-05-20_star_uvt_visibility_proxy_support_gate.md`
- `outputs/benchmarks/2026-05-20_star_uvt_visibility_proxy_dense_support_diagnostic.md`
- `outputs/benchmarks/2026-05-20_star_uvt_visibility_proxy_dense_support_diagnostic.json`
- `outputs/benchmarks/2026-05-20_star_uvt_feature_targetgrid_visibility_proxy10x_from1500_lr001_20step_media.json`
- `outputs/checkpoints/2026-05-20_star_uvt_feature_targetgrid_visibility_proxy10x_from1500_lr001_20step.pt`
- `outputs/run_logs/star_uvt_visibility_proxy10x_from1500_lr001_20step_20260520_040916.log`

## Result

The 10x/20-step trainer gate fails:

- weighted loss `0.834100 -> 0.844115`
- feature target loss `0.625418 -> 0.626559`
- RGB-probe loss `0.006269 -> 0.006492`
- visibility proxy loss `-4.20957 -> -4.21215`
- final dense RGB PSNR `5.644`
- mean step/backward/proxy `431.6/297.9/224.9ms`

The dense support diagnostic says the proxy improves color/content under forced
alpha, but not support:

- normal dense PSNR: `5.438 -> 5.640 -> 5.644`
- forced-alpha PSNR: `11.722 -> 14.552 -> 14.554`
- target-background oracle: `20.140 -> 25.834 -> 25.812`
- alpha mean: `0.1607 -> 0.1586 -> 0.1591`
- alpha `>0.1`: `0.411 -> 0.405 -> 0.406`
- best raw-opacity PSNR: `5.568 -> 5.809 -> 5.812`

## Read

This is real negative progress: the current proxy is not a quality bridge. It
sends gradients and improves the content field when alpha is forced, but it
does not improve alpha support. Cranking its weight hurts the feature/probe
objective before support recovers.

Next STAR UVT support work needs either an explicit opacity/support term, a
support-density parameterization, or a denser visibility objective that changes
coverage directly. The center-only target-point proxy should not be scaled to
300 videos.
