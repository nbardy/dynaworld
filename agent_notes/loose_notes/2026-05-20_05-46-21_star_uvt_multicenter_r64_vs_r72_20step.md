# STAR UVT Multicenter r64 vs r72 20-Step Gate

Date: 2026-05-20

## Goal

Close the matched comparison requested after the `K=8/r64/o0.4` selected
support gate. The question was whether the slightly wider `r72/o0.4` row should
replace `r64/o0.4` before any longer continuation or dataset-scale work.

## Commands

Trainer:

```bash
PYTHONPATH=src/train:. STAR_UVT_TILE_CAPACITY=128 rtk .venv/bin/python \
  src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit32_multicenter_k8_r72_o04_from1500_lr001_20step_media.jsonc
```

Dense support comparison:

```bash
PYTHONPATH=src/train:. STAR_UVT_TILE_CAPACITY=128 rtk .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/dense_alpha_failure_diagnostic.py \
  --case start1500=src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr001_resume50_from1450_lr005sparse_media.jsonc \
  --case topbirth32=src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit32_from1500_lr001_5step_media.jsonc \
  --case uncovered32=src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit32_uncovered_from1500_lr001_5step_media.jsonc \
  --case multicenter_k8_r64_o04_20step=src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit32_multicenter_k8_r64_o04_from1500_lr001_20step_media.jsonc \
  --case multicenter_k8_r72_o04_20step=src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit32_multicenter_k8_r72_o04_from1500_lr001_20step_media.jsonc \
  --out-json outputs/benchmarks/2026-05-20_star_uvt_birthsplit_multicenter_k8_r64_vs_r72_o04_20step_dense_support.json \
  --out-md outputs/benchmarks/2026-05-20_star_uvt_birthsplit_multicenter_k8_r64_vs_r72_o04_20step_dense_support.md \
  --date 2026-05-20
```

## Artifacts

- Config:
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit32_multicenter_k8_r72_o04_from1500_lr001_20step_media.jsonc`
- Trainer JSON:
  `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_multicenter_k8_r72_o04_20step_media.json`
- Dense comparison:
  `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_multicenter_k8_r64_vs_r72_o04_20step_dense_support.md`
- Checkpoint:
  `outputs/checkpoints/2026-05-20_star_uvt_feature_targetgrid_birthsplit32_multicenter_k8_r72_o04_from1500_lr001_20step.pt`
- Media:
  `outputs/media/2026-05-20_star_uvt_birthsplit_multicenter_k8_r72_o04_20step_contact.png`
  and
  `outputs/media/2026-05-20_star_uvt_birthsplit_multicenter_k8_r72_o04_20step_rgb_probe_contact.png`
- W&B offline run:
  `wandb/offline-run-20260520_054621-s74sprse`

## Results

| row | pass | loss | feature loss | probe PSNR | full RGB PSNR | mean step/backward | last step/backward | dense alpha >0.1 | dense alpha >0.5 | forced-alpha | oracle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `K8/r64/o0.4` | yes | `0.903197 -> 0.897231` | `0.631571 -> 0.631083` | `21.681 -> 21.769` | `5.794` | `157.5/59.3ms` | `147.3/54.3ms` | `0.431158` | `0.138506` | `14.631` | `24.851` |
| `K8/r72/o0.4` | yes | `0.910099 -> 0.903088` | `0.633414 -> 0.632829` | `21.601 -> 21.703` | `5.820` | `157.9/61.1ms` | `140.3/53.0ms` | `0.432454` | `0.146591` | `14.635` | `24.668` |

Both rows are zero-overflow at cap `128` with max/p95/cap `101/71/128`.

## Read

`r72/o0.4` is not the new default. It buys tiny dense coverage and normal RGB
PSNR improvements, but it starts and ends with worse weighted loss, feature
loss, and probe PSNR, and it gives back target-background oracle (`24.851 ->
24.668`). The balanced support row remains `K=8/r64/o0.4`.

This is real progress on support placement: multi-center birth/split is the
first fixed-budget support move that holds a measurable alpha coverage gain
after a trainer run. It is not yet visual-quality progress. The next useful
experiment is a longer `K=8/r64/o0.4` continuation with the current objective,
or a new visibility/support/model bridge. It is still too early to scale this
branch to the 300-video dataset.

## Follow-Up

Sync the r72 decision into the routing docs and rerun final py_compile, focused
pytest, diff-check, line-count, whitespace, and lingering-process validation.
