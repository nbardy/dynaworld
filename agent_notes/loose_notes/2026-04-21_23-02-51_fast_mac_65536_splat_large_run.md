# Fast-Mac 65k-Splat Large Local Run

## Context

The user asked to return to the larger 128px/4fps DynaWorld prebaked-camera run and push splat count now that the fast-mac v5 renderer is fixed and fast enough. The target was 512 gaussians per token with the existing 128-token baseline:

```text
128 tokens * 512 gaussians/token = 65,536 splats
```

The intent was not to add new training complexity, only to test whether the fixed renderer makes the higher-capacity version mechanically realistic.

## Config Added

Added:

```text
src/train_configs/local_mac_overfit_prebaked_camera_128_4fps_wide_depth_fast_mac_65536splats.jsonc
```

This clones the fast-mac 8192-splat 128px/4fps config and changes only the splat count / run naming:

- `model.gaussians_per_token = 512`
- `run_name = local-mac-overfit-prebaked-camera-128-4fps-wide-depth-fast-mac-65536splats-bounded-random-init`
- tags include `65536splats`

Important held-constant settings:

- `video_variant = small_128_4fps`
- `model.tokens = 128`
- `renderer.mode = fast_mac`
- `train.frames_per_step = 8`
- `optimizer.lr = 0.0005`
- cosine schedule to `0.1x`
- `loss.type = standard_gs`
- renderer diagnostics off

## Command

```bash
./src/train_scripts/train_full_dynamic_with_camera_prebake_all_frames.sh \
  src/train_configs/local_mac_overfit_prebaked_camera_128_4fps_wide_depth_fast_mac_65536splats.jsonc
```

## Result

The run completed locally on MPS with no NaNs and no OOM.

W&B:

```text
https://wandb.ai/nbardy/dynaworld/runs/3piy8cww
```

Runtime:

- 1000/1000 steps
- about 4:02 wall-clock after Taichi-free fast-mac path
- final reported throughput about `4.13 it/s`
- warm training generally around `4.0-4.6 it/s`

Final W&B summary:

```text
Eval/Loss  0.10858
Eval/L1    0.07233
Eval/MSE   0.01278
Eval/SSIM  0.49289
Eval/DSSIM 0.25355
Eval/PSNR  18.93466
Loss       0.1495
LR         0.00005
```

The final sampled train loss is a hard-window scalar and should not be over-read. The run history declined substantially, and the full-video eval metrics are a cleaner comparison point.

## Interpretation

The fixed fast-mac v5 renderer makes 65k splats at 128px/4fps practical on the local Mac. This is now mechanically viable as an overfit baseline: no immediate numerical failure, no memory blow-up, and a useful eval signal.

This does not prove the model architecture is solved. If visual detail is still poor, the next comparison should be controlled:

- rerun fixed v5 at 8192 splats using the same current settings
- compare W&B videos/eval against the new 65k run
- only then tune LR/opacity/scale or train longer

Old 8192 fast-mac results before the v5 saturated-backward fix are not a fair baseline for quality.
