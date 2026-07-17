# STAR UVT Birth/Split Uncovered Gate

Date: 2026-05-20 04:50 +07

## Why

The top-brightness birth/split trainer gate proved the fixed-budget tube reallocation path is real, but the dense support diagnostic said alpha `>0.1` coverage was still only `0.411`. The obvious stale next step in the docs was to sample uncovered/low-alpha target pixels instead of simply bright target pixels.

## What Changed

- Added `support_birth_split.target_point_source` with values:
  - `top_brightness`
  - `uncovered_brightness`
  - `low_alpha`
- Kept old configs compatible by defaulting the new field to `top_brightness`.
- Added a sampled alpha pre-pass for non-top-brightness target selection using sparse F1 alpha rendering over the sampled frame/grid lattice.
- Added a 512px/64f uncovered-birth config:
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit32_uncovered_from1500_lr001_5step_media.jsonc`.
- Added unit coverage proving `uncovered_brightness` prefers a dimmer low-alpha pixel over a brighter already-covered pixel.

## Trainer Gate

Command:

```bash
PYTHONPATH=src/train rtk uv run python src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit32_uncovered_from1500_lr001_5step_media.jsonc
```

Result:

- W&B offline run: `d731jv15`.
- JSON: `outputs/benchmarks/2026-05-20_star_uvt_feature_targetgrid_birthsplit32_uncovered_from1500_lr001_5step_media.json`.
- Report: `outputs/benchmarks/2026-05-20_star_uvt_birthsplit32_uncovered_trainer_gate.md`.
- Checkpoint: `outputs/checkpoints/2026-05-20_star_uvt_feature_targetgrid_birthsplit32_uncovered_from1500_lr001_5step.pt`.
- Pass: `true`.
- Alpha pre-sample: `85.5ms`.
- Target candidates/selected: `5243/2048`.
- Selected target alpha mean/min/max: `0.0209/0.0000/0.1717`.
- Reallocated tubes: `32/8192`.
- Selected opacity: `0.3418 -> 0.8000`.
- Zero overflow: max/p95/cap `100/71/128`.
- Mean step/backward/render: `187.4/61.8/65.3ms`.
- Last step: `140.4ms`.
- Weighted loss: `0.900186 -> 0.899545`.
- Feature target loss: `0.634780 -> 0.634690`.
- RGB-probe loss: `0.006635 -> 0.006621`.
- Full RGB PSNR: `5.713`.

## Dense Support Diagnostic

Command wrote:

- `outputs/benchmarks/2026-05-20_star_uvt_birthsplit32_uncovered_dense_support_diagnostic.json`
- `outputs/benchmarks/2026-05-20_star_uvt_birthsplit32_uncovered_dense_support_diagnostic.md`

Rows:

| row | normal PSNR | forced-alpha PSNR | target-bg oracle PSNR | alpha >0.1 | alpha >0.5 |
|---|---:|---:|---:|---:|---:|
| start1500 | 5.438 | 11.722 | 20.140 | 0.411 | 0.100 |
| center5 | 5.640 | 14.552 | 25.834 | 0.405 | 0.099 |
| support5 | 5.643 | 14.553 | 25.820 | 0.406 | 0.099 |
| top-birth32 | 5.708 | 14.606 | 25.234 | 0.411 | 0.117 |
| uncovered-birth32 | 5.713 | 14.579 | 25.319 | 0.411 | 0.119 |

## Read

Uncovered sampling is mechanically correct and slightly improves normal dense RGB over top-brightness birth/split. It does not materially solve coverage: alpha `>0.1` remains pinned at `0.411`, forced-alpha PSNR is a little lower than top-brightness, and target-background oracle is still below center/support rows.

This is real progress because the project now has a fixed-budget support-changing trainer primitive plus a sharper negative result: choosing uncovered target points alone is not enough. The next experiment should sweep birth count/radius and compare `uncovered_brightness` vs `low_alpha`, gating on dense alpha coverage and tile overflow before any longer run.
