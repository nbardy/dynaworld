# STAR UVT Multi-Center K8 R64 O0.4 20-Step Media Gate

## Question

The K8 radius/opacity sweep selected `r64/o0.4` as the balanced support row:
it preserved most of the multi-center coverage gain while recovering oracle.
This gate asks whether that row holds under a short media run instead of only a
5-step diagnostic.

## Config

Checked-in config:

`src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit32_multicenter_k8_r64_o04_from1500_lr001_20step_media.jsonc`

Support settings:

- target source: `uncovered_brightness`
- center strategy/count: `farthest_xy`, `K=8`
- reallocated tubes: `32/8192`
- support radius: `64px`
- born opacity: `0.4`
- tile capacity: `128`
- resume checkpoint: sparse-forward batched VJP step 1500

Command:

```bash
PYTHONPATH=src/train:. STAR_UVT_TILE_CAPACITY=128 rtk .venv/bin/python \
  src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit32_multicenter_k8_r64_o04_from1500_lr001_20step_media.jsonc
```

Dense-support diagnostic:

```bash
PYTHONPATH=src/train:. STAR_UVT_TILE_CAPACITY=128 rtk .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/dense_alpha_failure_diagnostic.py \
  --case start1500=src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr001_resume50_from1450_lr005sparse_media.jsonc \
  --case topbirth32=src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit32_from1500_lr001_5step_media.jsonc \
  --case uncovered32=src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit32_uncovered_from1500_lr001_5step_media.jsonc \
  --case multicenter_k8_r64_o04_20step=src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit32_multicenter_k8_r64_o04_from1500_lr001_20step_media.jsonc \
  --out-json outputs/benchmarks/2026-05-20_star_uvt_birthsplit_multicenter_k8_r64_o04_20step_dense_support.json \
  --out-md outputs/benchmarks/2026-05-20_star_uvt_birthsplit_multicenter_k8_r64_o04_20step_dense_support.md \
  --date 2026-05-20
```

## Artifacts

- JSON: `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_multicenter_k8_r64_o04_20step_media.json`
- Dense support JSON: `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_multicenter_k8_r64_o04_20step_dense_support.json`
- Dense support MD: `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_multicenter_k8_r64_o04_20step_dense_support.md`
- Checkpoint: `outputs/checkpoints/2026-05-20_star_uvt_feature_targetgrid_birthsplit32_multicenter_k8_r64_o04_from1500_lr001_20step.pt`
- Contact sheet: `outputs/media/2026-05-20_star_uvt_birthsplit_multicenter_k8_r64_o04_20step_contact.png`
- Probe contact sheet: `outputs/media/2026-05-20_star_uvt_birthsplit_multicenter_k8_r64_o04_20step_rgb_probe_contact.png`
- Side-by-side video: `outputs/media/2026-05-20_star_uvt_birthsplit_multicenter_k8_r64_o04_20step_side_by_side.mp4`
- Probe side-by-side video: `outputs/media/2026-05-20_star_uvt_birthsplit_multicenter_k8_r64_o04_20step_rgb_probe_side_by_side.mp4`
- W&B offline run: `wandb/offline-run-20260520_053952-bvnwpudz`

## Trainer Result

The trainer passed:

- weighted loss: `0.903197 -> 0.897231`
- feature target loss: `0.631571 -> 0.631083`
- RGB probe loss: `0.006791 -> 0.006654`
- RGB probe PSNR: `21.681 -> 21.769`
- final full RGB PSNR: `5.794`
- tile overflow/unstable sums: `0/0`
- tile max/p95/cap: `101/71/128`
- mean step/backward: `157.5ms` / `59.3ms`
- last step/backward: `147.3ms` / `54.3ms`

## Dense Support Result

Dense support after 20 steps:

- normal PSNR: `5.794`
- forced-alpha PSNR: `14.631`
- target-background oracle PSNR: `24.851`
- alpha `>0.1`: `0.431158`
- alpha `>0.5`: `0.138506`
- alpha mean: `0.186588`

Baseline comparisons in the same diagnostic:

- `start1500`: alpha `>0.1` `0.411094`, oracle `20.140`, forced-alpha `11.722`
- `topbirth32`: alpha `>0.1` `0.410507`, oracle `25.234`, forced-alpha `14.606`
- `uncovered32`: alpha `>0.1` `0.411279`, oracle `25.319`, forced-alpha `14.579`

## Read

This is a positive support gate, not a final visual-quality gate. Multi-center
K8 keeps the support gain after 20 steps: alpha `>0.1` remains `0.431`, above
the old `0.411` band, with zero overflow and good speed. It also improves the
5-step selected row's oracle slightly (`24.805 -> 24.851`) and improves probe
PSNR.

The contact sheet still shows the same qualitative failure mode: the visual
render is sparse/black and not competitive with RGB STAR. Forced-alpha and
oracle remain far above normal PSNR, so visibility/coverage is still the main
gap, even though this gate moved coverage in the right direction.

Next:

- run a 50-step continuation from this 20-step checkpoint only if we want to see
  whether support/feature losses keep improving
- run a matched `K=8/r72/o0.4` 20-step media gate if we want slightly more
  alpha coverage at some oracle cost
- do not scale this to 300 clips yet; STAR UVT still needs a visual-quality
  bridge after the support primitive
