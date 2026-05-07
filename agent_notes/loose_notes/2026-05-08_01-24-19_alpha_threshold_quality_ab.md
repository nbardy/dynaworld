# Alpha-Threshold Quality A/B

Date: 2026-05-08

## Context

The fast-mac feature-kernel catalog left `alpha_threshold` as the most direct
semantic speed lever: it reduces binned support and actual feature contributors,
but could remove useful splats. The goal here was the smallest decisive
heldout-quality A/B before doing more kernel work.

Base setup stayed fixed:

- Config family:
  `src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_vjepa_full_relpose_features_F32_256_16f_8192splats_goodset_train0006_0014_holdout0005_alphaab_alpha1_*.jsonc`
- Base model: 256px F32 feature splatting, 8192 splats, 16 frames, 250 steps.
- Renderer: stable `fast_mac` `v5_features`, not a new kernel fork.
- Split: train `camera_0006` and `camera_0014`, heldout `camera_0005`,
  anchor/condition `camera_0006`.
- Fixed across runs: model shape, V-JEPA feature cache, `transmittance_threshold`,
  random RGB train background, white eval background, colorize setup, schedule,
  validation media.
- Changed across runs: only top-level `render.alpha_threshold`,
  nested `render.fast_mac.alpha_threshold`, W&B run labels, and checkpoint paths.

## Runs

Launch command shape from the dynaworld root:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train uv run python \
  src/train/train_multicam_relative_pose_implicit_dynamic.py \
  src/train_configs/<alpha-ab-config>.jsonc
```

W&B was online. The script printed `date: invalid argument 's' for -I` around
the wrapper timestamps because macOS `date` does not accept GNU `-Is`; this did
not affect training.

| Threshold | Config suffix | W&B | Runtime | Heldout PSNR / SSIM / L1 | Train PSNRs | Checkpoint |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `1/255` | `alphaab_alpha1_255` | [`j9fkocvj`](https://wandb.ai/nbardy/dynaworld/runs/j9fkocvj) | 26.01 min | 12.7536 / 0.1716 / 0.1729 | 19.3927 / 19.5976 | `outputs/multicam_relative_pose/full_relpose_features_F32_256_alphaab_alpha1_255_goodset_train0006_0014_holdout0005/checkpoint_final.pt` |
| `1/128` | `alphaab_alpha1_128` | [`hru1yv0t`](https://wandb.ai/nbardy/dynaworld/runs/hru1yv0t) | 18.37 min | 13.6248 / 0.1922 / 0.1561 | 19.2271 / 19.7479 | `outputs/multicam_relative_pose/full_relpose_features_F32_256_alphaab_alpha1_128_goodset_train0006_0014_holdout0005/checkpoint_final.pt` |
| `1/96` | `alphaab_alpha1_96` | [`dsq6u3wq`](https://wandb.ai/nbardy/dynaworld/runs/dsq6u3wq) | 16.60 min | 13.2942 / 0.1838 / 0.1599 | 20.1208 / 20.6967 | `outputs/multicam_relative_pose/full_relpose_features_F32_256_alphaab_alpha1_96_goodset_train0006_0014_holdout0005/checkpoint_final.pt` |
| `1/64` | `alphaab_alpha1_64` | [`obclxj4w`](https://wandb.ai/nbardy/dynaworld/runs/obclxj4w) | 14.27 min | 12.7667 / 0.1806 / 0.1712 | 18.0380 / 18.3682 | `outputs/multicam_relative_pose/full_relpose_features_F32_256_alphaab_alpha1_64_goodset_train0006_0014_holdout0005/checkpoint_final.pt` |

Speedup vs default W&B runtime:

- `1/128`: 29.4%
- `1/96`: 36.2%
- `1/64`: 45.1%

Best heldout was `1/128`; best runtime was `1/64`.

## Media Evidence

Each run uploaded final `Multicam_Feature_GT_Render_ByCamera_Grid_Video` and
`Multicam_GT_Splat_Alpha_Feature_Grid_Video` videos. Local final media paths:

- Default `1/255`:
  `wandb/run-20260508_000333-j9fkocvj/files/media/videos/Multicam_Feature_GT_Render_ByCamera_Grid_Video_240_2e87aac4589dbdf45fd3.mp4`
- `1/128`:
  `wandb/run-20260508_003038-hru1yv0t/files/media/videos/Multicam_Feature_GT_Render_ByCamera_Grid_Video_240_d580a70baf9ce1e7a9e3.mp4`
- `1/96`:
  `wandb/run-20260508_005001-dsq6u3wq/files/media/videos/Multicam_Feature_GT_Render_ByCamera_Grid_Video_240_212ac84ee0ddcec998fc.mp4`
- `1/64`:
  `wandb/run-20260508_010728-obclxj4w/files/media/videos/Multicam_Feature_GT_Render_ByCamera_Grid_Video_240_43e44c9c64de4cb3e4eb.mp4`

Qualitative read from first-frame extracts:

- All variants remain blurry and under-detailed; this A/B does not solve F32
  texture quality.
- `1/128` and `1/96` preserve the broad camera-row layout and do not show
  obvious support holes in the final grids.
- `1/64` keeps broad coverage but is visibly more smeared and source-view
  fidelity drops, matching its lower source PSNR.

## Decision

Promote `alpha_threshold = 1/128` for the current stable 256px goodset F32
`v5_features` setup. It is both faster and better on heldout metrics than the
default control. `1/96` is a valid speed-biased follow-up if runtime matters
more than the best heldout scalar. `1/64` should stay a speed stress point, not
a default, because its source-view quality drops and heldout PSNR is basically
default despite the runtime win.

This result does not promote any experimental kernel fork. It only validates a
stable-renderer support threshold for this trainer/config family. The next
kernel work should still be judged by heldout media and metrics after parity
checks, not by raster timing alone.
