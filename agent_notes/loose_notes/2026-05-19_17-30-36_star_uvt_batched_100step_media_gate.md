# STAR UVT Batched 100-Step Media Gate

## Why

The 5-step repeat gate proved that sparse-forward plus batched target/probe VJP
is fast, but it did not prove the path remains useful over a real overfit
window or through the launch helper. This gate reruns the selected path for 100
steps from the same 1300-step checkpoint and writes media/checkpoint artifacts.

## Change

Added:

```text
src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume100_from1300_sparseforward_batchedvjp_checkpoint_media.jsonc
```

Updated:

```text
src/train_scripts/train_fast_overfit_star_uvt_and_dynamic_gsplat.sh
```

`star-feature-512-fast` now launches the batched V-JEPA target route. The older
RGB-target speed row remains available as `star-feature-512-rgbfast`.

## Command

```bash
./src/train_scripts/train_fast_overfit_star_uvt_and_dynamic_gsplat.sh star-feature-512-fast
```

## Result

```text
pass: true
offline W&B run: wandb/offline-run-20260519_172902-wq71rfuf
image_vjp_mode: analytic_sparse_grid_forward_batched
render_mode: feature_direct_gradcache_reduce_vec4
steps: 100
global steps: 1300 -> 1400
tile overflow: 0
max tile: 68
p95 tile: 45
loss: 0.886537 -> 0.880744
feature target loss: 0.632124 -> 0.627122
RGB-probe loss: 0.006360 -> 0.006341
RGB-probe PSNR: 21.965 -> 21.979
```

Timing:

```text
mean step/backward/render: 399.884 / 176.920 / 125.154ms
no-first step/backward/render: 392.289 / 176.404 / 123.754ms
last-20 step/backward/render: 262.927 / 109.391 / 93.992ms
last step/backward/render: 187.782 / 74.409 / 73.838ms
```

Compared with the previous same-checkpoint 100-step target-grid row
(`1690.225ms` step, `909.575ms` backward, `616.709ms` render), the batched
sparse-forward path preserves the same loss/probe movement and is roughly
`4.23x` faster on mean step, `5.14x` faster on mean backward, and `4.93x`
faster on mean render.

## Artifacts

```text
outputs/benchmarks/2026-05-19_star_uvt_sparseforward_batchedvjp_64f512_from1300_100step_media.md
outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparseforward_batchedvjp_64f512_from1300_100step_media.json
outputs/checkpoints/2026-05-19_star_uvt_feature_targetgrid_sparseforward_batchedvjp_64f512_1400step.pt
outputs/media/2026-05-19_star_uvt_feature_targetgrid_sparseforward_batchedvjp_64f512_from1300_100step_contact.jpg
outputs/media/2026-05-19_star_uvt_feature_targetgrid_sparseforward_batchedvjp_64f512_from1300_100step_sbs.mp4
```

The contact sheet exists, opens as `4110x1026` RGB, and is not blank. The MP4
is `1024x512`, 64 frames, 10.667 seconds. The media still shows a blurry
frozen-probe reconstruction. Treat this as a speed/path validation, not a
quality promotion.

## Next

The next STAR UVT work should focus on feature-target visual quality or on a
real native fixedbin/tile-slot/target-probe VJP kernel that beats the batched
repeat/100-step surface. Dense VJP packing and dense-forward variants are no
longer the right target.
