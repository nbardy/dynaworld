# Fast-mac white-render diagnosis

## Context

The active default script was switched to:

```text
src/train_configs/local_mac_overfit_prebaked_camera_128_4fps_wide_depth_fast_mac_8192splats.jsonc
```

It runs much faster than Taichi, about 17 it/s in the user's run, but the loss
stayed around `0.418` and W&B validation samples went all white. The first
guess was that fast-mac v5 might be returning background incorrectly.

## Renderer parity checks

I compared fast-mac v5 against Taichi on the same decoded 8192-splat batch:

```text
B=8, H=W=128, G=8192
target mean ~= 0.545
fast mean   ~= 0.873
taichi mean ~= 0.873
fast vs taichi L1 ~= 6.1e-8
fast vs taichi Linf ~= 3.8e-6
```

The projected v5 profile showed:

```text
tiles=512
total_pairs=25660
mean_pairs_per_tile=50.1
max_pairs_per_tile=209
overflow_tile_count=0
```

So the fast-mac forward path was not dropping everything and was not overflowing
tiles.

I also compared gradients through decoded tensors for `B=2, G=8192` using the
same reconstruction loss:

```text
loss fast   ~= 0.35040656
loss taichi ~= 0.35040635
xyz grad mean matched around 4.43e-5
scale grad mean matched around 1.07e-4
opacity grad mean matched around 4.95e-5
rgb grad mean matched around 4.45e-6
p99 relative grad diff was around 1e-4
```

This made a fast-mac renderer math bug unlikely.

## Actual failure

The failure is optimizer/projection stability.

With the active bounded-random init and `lr=1e-3`, seed-dependent runs produced
huge finite xyz-head gradients:

```text
grad_max around 1e24..1e26 in gaussian_heads.xyz_head.2.weight
clip_grad_norm returned inf on some steps
```

Some trajectories then produced nonfinite decoded tensors. Once decoded
`scales`/`opacities` become NaN, v5 can still return a finite all-white image
because invalid splats do not contribute and the white background remains. That
made W&B show a finite high loss instead of a crash.

The mathematical source is splats crossing close to the camera plane. Projection
uses terms like:

```text
fx / z
fx * x / z^2
```

The old guard used `z > 1e-4`, which is finite but too small for training. At
128px with focal around `379`, `fx / 1e-4` is millions, and the Jacobian terms
can make gradients enormous before any renderer-specific logic matters.

## Fixes made

1. Added `render.near_plane`, defaulting to `0.05` during config normalization.
2. Threaded `near_plane` through dense, tiled, Taichi, and fast-mac projections.
3. Set the active 8192 fast-mac and Taichi configs to `near_plane=0.05`.
4. Reduced the active 8192 fast-mac/Taichi learning rate from `1e-3` to `5e-4`.
5. Added trainer fail-fast checks for nonfinite decoded Gaussian tensors.
6. Changed gradient clipping to `error_if_nonfinite=True` so an infinite grad
   norm cannot silently zero/poison an update.
7. Printed the renderer summary including `near_plane`.

## Validation

Compile check passed:

```text
uv run python -m py_compile \
  src/train/renderers/common.py \
  src/train/renderers/dense.py \
  src/train/renderers/tiled.py \
  src/train/renderers/taichi.py \
  src/train/renderers/fast_mac.py \
  src/train/rendering.py \
  src/train/debug_metrics.py \
  src/train/dynamicTokenGS.py
```

A real `run_training` smoke with W&B disabled completed for 3 steps and printed:

```text
Renderer: renderer=fast_mac, tile_size=16, near_plane=0.05, alpha_threshold=0.00392156862745098
```

Two 1000-step local fast-mac trajectories with `lr=5e-4` and `near_plane=0.05`
did not collapse:

```text
seed 123 eval: loss=0.1288, l1=0.0822, ssim=0.3702, mean=0.542, white=0.000
seed 456 eval: loss=0.1320, l1=0.0859, ssim=0.3679, mean=0.544, white=0.000
```

This is stable and no longer white, but still not crisp. Remaining quality work
is likely model/density/scale dynamics, not fast-mac forward correctness.
