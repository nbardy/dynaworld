# Three Renderer Tiny Baseline Control

## Context

The user called out that we were adding renderer/model/init complexity before
reconfirming the tiny stable baseline. The immediate question was whether the
simple baseline is alive, whether it is the Torch rasterizer, and whether the
same exact baseline can be tried across all three rasterizers.

## Baseline Under Test

Base config:

```text
src/train_configs/local_mac_overfit_prebaked_camera.jsonc
```

Important properties:

- `small_32_2fps`
- 23 prebaked DUSt3R frames
- `model_size=32`
- 128 tokens x 4 gaussians/token = 512 splats
- old all-frames training behavior (`frames_per_step=0`)
- old tiny-baseline loss: `l1_mse` with `l1_weight=1.0`, `mse_weight=0.2`
- `near_plane=0.0001`

The controlled harness ran 100 optimizer steps with W&B disabled, fixed seed
`20260421`, and changed only the renderer-specific config needed to call each
backend:

- `dense`: PyTorch dense rasterizer, tile size 8
- `taichi`: Taichi Metal rasterizer, tile size 16
- `fast_mac`: fast Mac Metal rasterizer, tile size 16

## Results

```text
dense    step=  1 loss=0.232605 render_mean=0.7098
dense    step= 10 loss=0.118925 render_mean=0.5343
dense    step= 25 loss=0.089385 render_mean=0.5519
dense    step= 50 loss=0.076461 render_mean=0.5630
dense    step=100 loss=0.065573 render_mean=0.5458
RESULT dense    train_loss=0.065573 eval_loss=0.065807 eval_l1=0.063866 eval_mse=0.009707 eval_ssim=0.535537 render_mean=0.5608 white=0.0000 seconds=85.16 it_s=1.17

taichi   step=  1 loss=0.235207 render_mean=0.7105
taichi   step= 10 loss=0.118555 render_mean=0.5559
taichi   step= 25 loss=0.105950 render_mean=0.5110
taichi   step= 50 loss=0.078382 render_mean=0.5603
taichi   step=100 loss=0.066641 render_mean=0.5604
RESULT taichi   train_loss=0.066641 eval_loss=0.066723 eval_l1=0.064770 eval_mse=0.009763 eval_ssim=0.508788 render_mean=0.5424 white=0.0000 seconds=45.28 it_s=2.21

fast_mac step=  1 loss=0.235207 render_mean=0.7105
fast_mac step= 10 loss=0.117463 render_mean=0.5367
fast_mac step= 25 loss=0.227202 render_mean=0.5190
fast_mac step= 50 loss=0.408617 render_mean=0.8261
fast_mac step=100 loss=0.174565 render_mean=0.6525
RESULT fast_mac train_loss=0.174565 eval_loss=0.176579 eval_l1=0.166798 eval_mse=0.048908 eval_ssim=0.221364 render_mean=0.6535 white=0.0498 seconds=18.43 it_s=5.43
```

## Interpretation

The simple baseline is alive. It is the PyTorch `dense` renderer path and it
still rapidly overfits the tiny 32px/23-frame/512-splat setup.

Taichi also reproduces the tiny baseline closely enough for this 100-step
control. It is about 1.9x faster than dense in this run and reaches essentially
the same eval loss.

fast-mac is much faster but is not a drop-in training-equivalent renderer yet.
It starts from the same loss, improves initially, then diverges into a higher
loss regime with brighter/partly white output. That means fast-mac should not
be made the default trainer renderer until tiny-baseline train parity is fixed.

This corrects the previous mistake: renderer throughput and renderer training
parity are separate claims. We should not use fast-mac speed as justification
for changing the default baseline until it passes the small baseline control.

## Likely Next Check

The highest-leverage next check is not more model complexity. It is renderer
semantic parity on the tiny baseline:

- compare dense/Taichi/fast-mac forward images and gradients from the same
  decoded splats at step 0
- check pixel-center convention (`integer pixel grid` vs `x + 0.5`)
- check alpha/background/depth ordering semantics
- only after these match should we retry fast-mac training
