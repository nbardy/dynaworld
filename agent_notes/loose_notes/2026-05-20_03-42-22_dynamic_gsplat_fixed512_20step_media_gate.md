# Dynamic Gsplat Fixed-512 20-Step Media Gate

Date: 2026-05-20 03:42 +07

## Context

The active STAR UVT goal audit still had a comparator gap: the matched dynamic
gsplat evidence was a 5-step smoke with `WANDB_MODE=disabled`, no media, and a
single timing row. To close that gap without spending a full 400-step run, I
added a bounded fixed-512 comparator config and ran it with offline W&B media.

## Config Added

`src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_gpt256_8192splats_overfit1_20step_matched_media.jsonc`

It keeps the same matched scale as the existing 400-step fixed-512 config:

- `64f`
- `512px` input/render
- cached V-JEPA conditioning
- differentiable V-JEPA feature loss disabled
- `32` active decoded tokens x `256` splats/token = `8192` active Gaussians
- temporal microbatch chunk size `16`

Only the run length/logging changes: `20` steps, `log_every=5`,
`always_log_last_step=true`, and offline W&B media.

## Commands

```bash
TRAIN_CONFIG=src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_gpt256_8192splats_overfit1_20step_matched_media.jsonc \
rtk ./src/train_scripts/train_single_video_pretrain_300_64f.sh resolve
```

```bash
PYTHONPATH=src/train rtk .venv/bin/python src/train/train.py \
  src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_gpt256_8192splats_overfit1_20step_matched_media.jsonc
```

## Results

- offline W&B: `wandb/offline-run-20260520_033857-90ynjlmp`
- run log:
  `outputs/run_logs/dynamic_gsplat_fixed512_64f_8192_20step_matched_media_20260520_033856.log`
- summary JSON:
  `outputs/benchmarks/2026-05-20_dynamic_gsplat_fixed512_20step_matched_media.json`
- report:
  `outputs/benchmarks/2026-05-20_dynamic_gsplat_fixed512_20step_matched_media.md`

Quality:

- train loss `0.601325 -> 0.492486`
- final eval PSNR `5.587`
- final eval SSIM `0.165`
- final eval L1 `0.469`
- media is a smeared blob, not a useful dense reconstruction

Timing over logged steps `5/10/15/20`:

- mean timed step `2.940s`
- mean backward `1.926s`
- mean forward decode `0.330s`
- mean sample/load `0.292s`
- mean rasterize `0.141s`

## Interpretation

This is stronger than the earlier 5-step smoke because it has media and eval
metrics. It also shows the earlier `8.019s` smoke row was too weak as a final
timing claim. The better 20-step evidence still does not make dynamic gsplat a
keeper: it is slower than the STAR UVT compact visual helper (`930.6ms`) and
lower quality (`5.587` eval PSNR versus compact STAR UVT dense `6.023`, both far
below the RGB STAR bracket `12.444`).

The remaining main STAR UVT blocker is still dense visibility/support quality,
not data loading, rasterization, or switching back to dynamic gsplats.
