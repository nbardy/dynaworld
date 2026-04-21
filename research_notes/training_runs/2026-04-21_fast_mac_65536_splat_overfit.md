# Fast-Mac 65k-Splat 128px/4fps Overfit Run

## Status

This is the first successful local 128px/4fps DynaWorld overfit run with `512` gaussians per token using the fixed fast-mac v5 renderer.

The run is useful as a new high-capacity local baseline, not as final quality evidence.

## Config

```text
src/train_configs/local_mac_overfit_prebaked_camera_128_4fps_wide_depth_fast_mac_65536splats.jsonc
```

Main settings:

```text
video_variant: small_128_4fps
frames: 46
render size: 128 x 128
tokens: 128
gaussians/token: 512
total gaussians: 65,536
renderer: fast_mac
frames_per_step: 8
steps: 1000
base_lr: 0.0005
schedule: cosine to 0.1x
loss: 0.8 L1 + 0.2 DSSIM
```

Initialization:

```text
xy_extent: 1.5
z_min: 0.5
z_max: 5.0
scale_init: 0.02
scale_init_log_jitter: 0.7
opacity_init: 0.1
token_init_std: 0.3
head_output_init_std: 0.06
position_init_extent_coverage: 0.9
```

## Command

```bash
./src/train_scripts/train_full_dynamic_with_camera_prebake_all_frames.sh \
  src/train_configs/local_mac_overfit_prebaked_camera_128_4fps_wide_depth_fast_mac_65536splats.jsonc
```

## W&B

```text
https://wandb.ai/nbardy/dynaworld/runs/3piy8cww
```

## Metrics

Final full-video eval summary:

```text
Eval/Loss  0.10858
Eval/L1    0.07233
Eval/MSE   0.01278
Eval/SSIM  0.49289
Eval/DSSIM 0.25355
Eval/PSNR  18.93466
```

Training completed:

```text
steps: 1000/1000
wall time: about 4:02
throughput: about 4.13 it/s overall
final sampled train loss: 0.1495
final lr: 0.00005
```

The final sampled train loss is not the best quality summary because each step trains on a frame window and the last sampled window can be harder. Prefer the full-video eval metrics and W&B videos.

## Interpretation

The fixed fast-mac v5 renderer makes 65k splats locally practical for this DynaWorld baseline. This run was stable: no NaNs, no OOM, and successful video/eval logging.

Quality looked meaningfully better than the broken fast-mac run and acceptable for this early overfit stage, but this run does not isolate whether the improvement comes from splat count, renderer correctness, or the current bounded-random initialization. The next clean comparison is fixed-v5 `8192` splats with the same current settings, then longer `65k` training if the visual trajectory still improves.

## Comparison Rules

When using this as a baseline:

- Compare against runs made after the v5 saturated-backward fix.
- Do not compare quality against the pre-fix fast-mac 8192 run; that run had corrupted backward behavior.
- Compare full-video eval and W&B videos, not only the final sampled train loss.
- Keep renderer, init, data, loss, and schedule fixed when testing splat count.
