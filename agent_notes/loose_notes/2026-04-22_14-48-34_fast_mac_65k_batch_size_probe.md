# Fast-Mac 65k Batch Size Probe

## Context

The user asked whether the 65,536-splat 128px/4fps overfit run could use more
frames per training batch now that the fast-mac renderer has native batch
support, and asked which rasterizer the current path uses.

The active Dynaworld `fast_mac` renderer path is v5:

```text
src/train/renderers/fast_mac.py
third_party/fast-mac-gsplat/variants/v5/torch_gsplat_bridge_v5
torch.ops.gsplat_metal_v5
```

`render_gaussian_frames(..., mode="fast_mac")` calls
`render_fast_mac_3dgs_batch(...)`, which projects `[B,G,...]` tensors and calls
`torch_gsplat_bridge_v5.rasterize_projected_gaussians(...)`. The current 65k
config uses:

```text
fast_mac.batch_strategy = flatten
batch_launch_limit_tiles = 262144
batch_launch_limit_gaussians = 262144
```

So this was testing v5 native batch rendering, not a Python loop over frames.

## Probes

Base config:

```text
src/train_configs/local_mac_overfit_prebaked_camera_128_4fps_wide_depth_fast_mac_65536splats.jsonc
```

The initial probes used in-memory config overrides rather than adding another
JSONC file:

- `frames_per_step=4`, `steps=100`, W&B offline, no video/eval
- `frames_per_step=16`, `steps=100`, W&B offline
- `frames_per_step=23`, `steps=25`, W&B offline
- `frames_per_step=46`, `steps=10`, W&B offline

Results:

```text
B=4:  completed, about 7.90 it/s overall, warm loop roughly 7.5-9.4 it/s
B=16: completed, around 2.09 it/s including final eval/video, warm loop around 2.3 it/s
B=23: completed, around 1.63 it/s
B=46: completed, around 0.83 it/s
```

Per trained frame, larger batches were modestly better:

```text
B=4 speed probe:          7.9 it/s * 4  ~= 31.6 frame-losses/s overall
B=8 previous full run: 4.13 it/s * 8  ~= 33.0 frame-losses/s overall
B=16 probe:             2.3 it/s * 16 ~= 36.8 frame-losses/s warm
B=23 probe:             1.63 it/s *23 ~= 37.5 frame-losses/s
B=46 probe:             0.83 it/s *46 ~= 38.2 frame-losses/s
```

The user's memory of `~15 it/s` was real but came from other regimes, not this
65k configuration: older small runs hit `~15.9 it/s`, and the pre-fix fast-mac
8192-splat run hit `~17.1 it/s` while training badly. The 65k path is materially
heavier.

Interpretation: the larger native batch path fits and gives a small frame
throughput gain, but it is not a huge scaling win. Full-sequence `B=46` cuts
optimizer update rate too hard for the current overfit loop unless the goal is
specifically lower gradient noise.

## Equal-Frame-Budget Run

Ran a medium online W&B run:

```text
W&B: https://wandb.ai/nbardy/dynaworld/runs/yl8usu29
run_name: local-mac-overfit-prebaked-camera-128-4fps-wide-depth-fast-mac-65536splats-b23-equal-frame-budget-350step
frames_per_step: 23
steps: 350
```

This gives about the same number of sampled frame losses as the earlier
`B=8, steps=1000` run:

```text
B=8 * 1000 = 8000 frame losses
B=23 * 350 = 8050 frame losses
```

Final summary:

```text
Eval/Loss  0.11280
Eval/L1    0.07511
Eval/MSE   0.01349
Eval/SSIM  0.47283
Eval/DSSIM 0.26359
Eval/PSNR  18.70023
Loss       0.09400
runtime    about 3:30
throughput about 1.66 it/s
```

Prior `B=8, steps=1000` 65k run for comparison:

```text
W&B: https://wandb.ai/nbardy/dynaworld/runs/3piy8cww
Eval/Loss  0.10858
Eval/L1    0.07233
Eval/MSE   0.01278
Eval/SSIM  0.49289
Eval/PSNR  18.93466
runtime    about 4:02
throughput about 4.13 it/s
```

The equal-frame-budget `B=23` run did not beat the older `B=8` run on full-video
eval metrics. The final sampled-window loss was lower, but that scalar is not
comparable because sampled windows differ and larger batches average over more
frames.

## LR Thought

For bigger batches, a modest LR bump is plausible because gradient variance is
lower. But the equal-frame-budget test also reduced optimizer updates from 1000
to 350 while keeping a cosine schedule that still decayed all the way to `5e-5`.
That means any next LR test should separate two knobs:

- larger batch with same `1000` updates, which is more compute and more frame
  exposure
- equal frame budget with a slower decay or higher base LR, e.g. `1e-3` for
  `B=16/B=23`

Do not conclude from `B=23,350` alone that larger batches are worse; it mostly
shows that larger batches plus fewer update steps plus the same full cosine
decay did not win.

## Takeaway

More frames per batch is mechanically viable with the v5 fast-mac path. It is
worth using `B=16` or `B=23` when we want lower-variance gradients or more clip
coverage per step, but the first equal-frame-budget test does not show a quality
win over `B=8`. For the next quality run, changing optimization schedule, total
update count, or model/head parameterization is likely higher leverage than
only increasing `frames_per_step`.
