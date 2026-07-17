# STAR UVT Feature1/Probe40 Balance Resume50

## Context

We had a usable 64f/512px STAR target-grid V-JEPA frozen-probe continuation
chain, but the most recent rows were oscillating:

- 1000->1100 `feature=0.5/probe=40` improved probe PSNR but drifted feature
  loss.
- 1100->1200 recover schedule improved feature loss but gave back probe PSNR.
- 1200->1250 `feature=0.75/probe=40` restored probe PSNR but raised feature
  loss.

This run tested whether a short constant `feature=1.0/probe=40.0` continuation
from the 1250-step checkpoint can improve both signals.

## Run

- Config:
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume50_from1250_checkpoint_media.jsonc`
- Source checkpoint:
  `outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature075_lr005_1250step_after_resume.pt`
- Result JSON:
  `outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume50_from1250_checkpoint_media.json`
- Output checkpoint:
  `outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_1300step_after_resume.pt`
- Media:
  `outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume50_from1250_contact.jpg`
  and
  `outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume50_from1250_sbs.mp4`
- Offline W&B:
  `wandb/offline-run-20260519_085636-t9elx75j`

Command:

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume50_from1250_checkpoint_media.jsonc
```

## Result

- Pass: `true`
- Global steps: `1250 -> 1300`
- Resume loaded / optimizer loaded / source local steps:
  `true / true / 50`
- Objective: `feature_target.loss_weight=1.0`,
  `rgb_probe_loss_weight=40.0`
- Feature loss: `0.6388027304783463 -> 0.6321917325258255`
- RGB-probe loss: `0.006408382269000867 -> 0.0063642389977758285`
- RGB-probe PSNR: `21.93251609802246 -> 21.96253538131714`
- Mean step timing:
  `1284.96ms/step`, `538.26ms` render, `12.97ms` feature target,
  `18.03ms` RGB probe, `677.64ms` backward
- Tile overflow: `0`

## Interpretation

This is the first current both-improving local balance row in the 64f/512px
frozen-probe continuation chain. It improves V-JEPA target-grid alignment and
probe PSNR at the same time, after the previous rows alternated between visual
gain and feature recovery.

It does not close the same-grid oracle (`23.401` probe PSNR). The next gate
should either extend this `feature=1/probe=40` balance row or move the remaining
work into native VJP/dataset-scale machinery. It is no longer useful to keep
doing simple recovery oscillations unless a new schedule has a specific
hypothesis.

## Validation

- `py_compile` passed for `train_star_uvt_feature_overfit.py`, both STAR UVT
  report scripts, and `tests/test_star_uvt_feature_target_adapter.py`.
- `PYTHONPATH=src/train uv run --with pytest python -m pytest
  tests/test_star_uvt_feature_target_adapter.py -q` passed: `10 passed`.
- Report invariants passed: comparison JSON has `26` rows, the
  `feature1_lr005_resume50_from1250` comparison row is `pass`, bridge audit flag
  is `true`, global steps are `1250 -> 1300`, and tile overflow is `0`.
- Touched-file trailing whitespace/newline scan passed.
- `git diff --check` passed.
- No active `train_star_uvt_feature_overfit.py` process remained.
