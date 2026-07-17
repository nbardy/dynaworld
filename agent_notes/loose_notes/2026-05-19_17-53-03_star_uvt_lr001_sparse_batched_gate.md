# STAR UVT lr001 Sparse-Forward Batched Gate

Date: 2026-05-19 17:53 Asia/Ho_Chi_Minh

## Why this run

The docs refresh left a concrete quality question: effective `lr=0.001` from the
1300-step checkpoint improved frozen-probe PSNR on the dense target-grid path,
but that row was slow (`1463.8ms/step`). The selected speed path was the lr005
sparse-forward batched VJP helper (`399.9ms/step`) but it ended with lower probe
PSNR and better feature loss. This run tests whether the safer lr001 quality
continuation transfers onto the fast sparse-forward batched VJP implementation.

## Command

```bash
PYTHONPATH=src/train .venv/bin/python src/train/train.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr001_resume100_from1300_sparseforward_batchedvjp_checkpoint_media.jsonc
```

Offline W&B:

```text
wandb/offline-run-20260519_175011-mzycmxlf
```

## Result

The run passed:

- loss `0.886537 -> 0.880940`
- feature loss `0.632124 -> 0.630549`
- frozen-probe loss `0.006360 -> 0.006260`
- frozen-probe PSNR `21.965 -> 22.034`
- mean step/backward/render `372.31/158.88/119.86ms`
- no-first step/backward/render `361.84/158.66/118.94ms`
- last-20 step/backward/render `538.79/229.60/178.40ms`
- last step spiked to `1562.01ms` step, `615.23ms` backward, `579.28ms` render
- zero overflow, max/p95/cap tile count `63/43/128`
- loaded optimizer LR `[0.005]`, effective optimizer LR `[0.001]`

Media exists and the MP4 probes as `1024x512`, `64` frames, `10.666667s`.
The contact sheet is valid but still blurry.

## Comparison

Against dense lr001, sparse-forward batched lr001 reaches the same quality
endpoint (`22.034` probe PSNR, `0.630549` feature loss) while cutting mean step
`1463.76 -> 372.31ms`, backward `778.41 -> 158.88ms`, and render
`571.75 -> 119.86ms`.

Against sparse-forward batched lr005, lr001 improves probe PSNR
(`22.034` vs `21.979`) but loses feature alignment (`0.630549` vs `0.627122`)
and has worse late timing because of the final-step spike.

## Next read

The speed path now covers both objective continuations. The unresolved gate is
visual quality, not dense target VJP. Next useful work is checkpoint selection,
measured recovery scheduling, or a stronger feature-to-RGB/objective bridge that
closes the same-grid `23.401` oracle; native VJP or real fixedbin only matters
if it beats the batched speed surface.
