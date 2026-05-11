# Fast-Mac v12a: Fused No-Norm Colorize + Alpha Compose + L1 Gradient

Date: 2026-05-10

Scope:

- New isolated variant path:
  `third_party/fast-mac-gsplat/variants/v12a_fused_colorize_l1_no_norm/`.
- Do not change stable `v5`, `v6`, `v9`, `v10`, or `v11` files.
- Target only the no-hidden, no-pre-norm, sigmoid colorizer case:
  `Conv2d(F, 3, 1x1) + sigmoid`, alpha RGB composition, and RGB L1.
- This is a kernel-design and prototype lane, not a trainer-promotion claim.

## Current Bottleneck Evidence

The earlier feature-kernel lane already moved the main F32 rasterizer backward
cost down from the original feature path. The current best copied shader family
is `v11_features_gradcache_zero_bg_hostmeta_fixedbin`; it wins only small
bridge/binning deltas over v9/v10, which means the next large target is outside
the rasterizer itself.

Direct rasterizer rows, local MPS, `B=16,G=8192,F=32`, `512x512`,
`GSP_CHUNK=64,GSP_FAST_CAP=2048`, `case=medium_sigma_3_8`:

| Variant | Forward ms | Backward ms | Total ms | Artifact |
| --- | ---: | ---: | ---: | --- |
| `v9_features_gradcache_zero_bg` | 89.8770 | 278.0232 | 367.9002 | `benchmark_outputs/fast_mac_feature_kernels/2026-05-08_v9_v10_v11_512_B16_G8192_F32.jsonl` |
| `v10_features_gradcache_zero_bg_hostmeta` | 88.3717 | 277.7613 | 366.1330 | same |
| `v11_features_gradcache_zero_bg_hostmeta_fixedbin` | 86.2557 | 277.7933 | 364.0490 | same |

For the same `256x256` shape, v11 was also best but only by about 1% total:
`270.4714ms` vs v9 `273.1887ms`.

Trainer-phase evidence shows that when the full F32 RGB reconstruction path is
included, colorize/loss is now a visible warm path:

| Probe | Key read | Artifact |
| --- | --- | --- |
| Full v11 512 step, 20 iterations | total mean `2947.6ms`, raster forward `382.2ms`, loss forward `268.5ms`, autograd backward `1944.9ms` | `2026-05-09_fullstep_v11_features_512_512_warm1_iters20.json` |
| v11 512 breakdown, default colorizer/loss | loss forward `1113.5ms`, loss/colorize backward probe `1668.0ms`, raster backward probe `211.3ms` | `2026-05-09_backward_breakdown_v11_features_512_512.json` |
| v11 512, no-pre-norm | loss forward `434.7ms`, loss/colorize backward probe `464.4ms`, raster backward probe `230.9ms` | `2026-05-09_loss_colorize_breakdown_v11_512_no_prenorm.json` |
| v11 512, L1 no-pre-norm | loss forward `103.3ms`, loss/colorize backward probe `67.7ms`, raster backward probe `190.8ms` | `2026-05-10_loss_colorize_breakdown_v11_512_l1_no_prenorm.json` |

Interpretation:

- v11 raster-only improvement is incremental; it is not a new speed class.
- LayerNorm/pre-norm is still extremely expensive at 512, but v12a explicitly
  scopes it out instead of trying to fuse LayerNorm.
- In the no-norm L1 case, the remaining PyTorch colorize/loss path is smaller
  but still a dense multi-tensor pass over a 1 GiB feature image at 32 targets.
  That is enough to justify a fused producer if it is simple and correct.

## Exact 512 / 32-Image / F32 Tensor Accounting

Definitions:

- `N = 32` image targets.
- `H = W = 512`.
- `F = 32` feature channels.
- `C = 3` RGB channels.
- dtype is `float32`, 4 bytes/value.
- Raster extension layout is `[N, H, W, F]`; trainer objective layout is
  `[N, F, H, W]`. The byte counts are identical.

Pixel counts:

```text
pixels_per_image = 512 * 512 = 262,144
pixels_total = 32 * 512 * 512 = 8,388,608
feature_values = pixels_total * 32 = 268,435,456
rgb_values = pixels_total * 3 = 25,165,824
```

Dense tensors:

| Tensor | Shape | Values | Bytes | MiB |
| --- | ---: | ---: | ---: | ---: |
| Raster features | `[32,512,512,32]` | 268,435,456 | 1,073,741,824 | 1024 |
| Raster alpha | `[32,512,512]` | 8,388,608 | 33,554,432 | 32 |
| Colorizer logits | `[32,3,512,512]` | 25,165,824 | 100,663,296 | 96 |
| Colorizer RGB | `[32,3,512,512]` | 25,165,824 | 100,663,296 | 96 |
| Composed RGB | `[32,3,512,512]` | 25,165,824 | 100,663,296 | 96 |
| Target RGB | `[32,3,512,512]` | 25,165,824 | 100,663,296 | 96 |
| Background RGB if materialized | `[32,3,512,512]` | 25,165,824 | 100,663,296 | 96 |
| Grad features | `[32,512,512,32]` | 268,435,456 | 1,073,741,824 | 1024 |
| Grad alpha | `[32,512,512]` | 8,388,608 | 33,554,432 | 32 |
| Colorizer weight | `[3,32]` | 96 | 384 | 0.000366 |
| Colorizer bias | `[3]` | 3 | 12 | tiny |

Current no-norm L1 PyTorch lower-bound dense memory traffic after rasterization:

```text
read features for conv                    ~= 1024 MiB
write/read logits or sigmoid RGB          ~= 192-384 MiB, implementation-dependent
read alpha + RGB + background, write comp ~= 320 MiB
read comp + target for L1                 ~= 192 MiB
backward read saved dense tensors         ~= 1200+ MiB
write grad_features + grad_alpha          ~= 1056 MiB
```

This is a conservative lower bound because it ignores MPS convolution internals,
temporary deltas, autograd graph bookkeeping, layout conversions, and pre-norm
intermediates. The no-norm L1 case is still plausibly a 3-5 GiB dense-traffic
subgraph around a 1 GiB feature image.

## Design Goals

Primary v12a goal:

- Produce the exact gradients needed by the existing raster backward without
  materializing colorizer logits, splat RGB, composed RGB, and L1 delta tensors.

Strict scope:

- No LayerNorm.
- No hidden colorizer layer.
- No view-direction conditioning.
- Sigmoid activation only for the first useful kernel.
- L1 reconstruction only.
- Fixed/materialized RGB background first; fixed RGB broadcast can be a later
  specialization.
- Keep v11 raster math untouched unless a future full-fusion fork is explicitly
  created.

Target outputs for the smallest useful producer:

```text
loss_per_image: [N]
grad_features: [N,H,W,F]
grad_alpha: [N,H,W]
grad_weight: [3,F]
grad_bias: [3]
```

Inputs:

```text
features: [N,H,W,F] float32, v11 raster output
alpha: [N,H,W] float32
target_rgb: [N,3,H,W] float32
background_rgb: [N,3,H,W] float32 or later fixed [3]
weight: [3,F] float32
bias: [3] float32
```

Autograd replacement objective:

```text
logits[n,c,y,x] = bias[c] + sum_f weight[c,f] * features[n,y,x,f]
splat_rgb = sigmoid(logits)
rgb = alpha[n,y,x] * splat_rgb + (1 - alpha[n,y,x]) * background_rgb[n,c,y,x]
loss_per_image[n] = mean_cyx(abs(rgb - target_rgb))
loss_total = mean_n(loss_per_image[n])
```

## Key Metal Kernel Pseudocode

This is the smallest v12a producer kernel. One thread owns one pixel. It reads
the F32 feature vector once, computes RGB/logits in registers, writes dense
feature and alpha gradients, and atomically reduces the tiny colorizer parameter
gradients.

```c
kernel fused_no_norm_l1_grad(
    features[N,H,W,F],
    alpha[N,H,W],
    target[N,3,H,W],
    background[N,3,H,W],
    weight[3,F],
    bias[3],
    loss_per_image[N],
    grad_features[N,H,W,F],
    grad_alpha[N,H,W],
    atomic grad_weight[3,F],
    atomic grad_bias[3]) {

  pix = global_thread_id;
  n, y, x = decode(pix);
  inv_per_image = 1.0 / (3 * H * W);
  inv_total = 1.0 / (N * 3 * H * W);

  // F=32 target. Store in thread registers if the compiler permits.
  float feat[32];
  for f in 0..F-1:
      feat[f] = features[pix, f];

  float a = alpha[pix];
  float g_feature[32] = {0};
  float g_alpha = 0;
  float loss_sum = 0;

  for c in 0..2:
      logit = bias[c];
      for f in 0..F-1:
          logit += weight[c,f] * feat[f];
      splat = sigmoid(logit);
      bg = background[n,c,y,x];
      pred = a * splat + (1 - a) * bg;
      diff = pred - target[n,c,y,x];
      sign = diff > 0 ? 1 : (diff < 0 ? -1 : 0);

      loss_sum += abs(diff);

      // Gradients for loss_total, not only loss_per_image.
      g_pred = sign * inv_total;
      g_alpha += g_pred * (splat - bg);
      g_logit = g_pred * a * splat * (1 - splat);

      atomic_add(grad_bias[c], g_logit);
      for f in 0..F-1:
          atomic_add(grad_weight[c,f], g_logit * feat[f]);
          g_feature[f] += g_logit * weight[c,f];

  for f in 0..F-1:
      grad_features[pix,f] = g_feature[f];
  grad_alpha[pix] = g_alpha;
  atomic_add(loss_per_image[n], loss_sum * inv_per_image);
}
```

Future full-fusion kernel shape:

```c
// In the existing forward raster loop, accumulate feature[32] and alpha in
// registers, then run the colorize/L1 math above. Instead of writing dense
// features and grad_features, immediately enter the reverse compositing pass
// with per-pixel grad_feature[32] and grad_alpha.
//
// This would remove:
// - feature image write from raster forward
// - feature image read by colorize
// - dense grad_features write by producer
// - dense grad_features read by raster backward
//
// It is a bigger kernel because the current v11 backward needs sorted IDs,
// tile stop counts, and reverse transmittance state. v12a should prove the
// colorize/L1 gradient producer first.
```

## Expected Saved Intermediates

Small producer prototype:

- Still saved by v11 raster autograd:
  - sorted/possibly fixed-bin IDs
  - tile counts/offsets
  - tile stop counts
  - sorted projected Gaussian inputs
  - raster `features` and `alpha` outputs
- No longer needed from PyTorch colorize/loss if the producer is integrated:
  - colorizer logits
  - sigmoid RGB
  - composed RGB
  - dense L1 delta or sign tensor
  - per-pixel LayerNorm state, because v12a does not support pre-norm
- Newly produced:
  - `grad_features [N,H,W,F]`
  - `grad_alpha [N,H,W]`
  - `grad_weight [3,F]`
  - `grad_bias [3]`
  - `loss_per_image [N]`

Full-fusion later:

- Keep v11 binning/sorted IDs/tile stop counts unless the backward schedule
  changes.
- Do not save or write dense `features` or `grad_features`.
- The per-pixel F32 feature vector and F32 feature-gradient vector live only in
  registers/thread memory during the fused forward/reverse pass.

## Autograd Boundary

Prototype boundary:

- Expose a local op under the v12a package, for example
  `fused_no_norm_l1_grad(...)`.
- The op is not a trainer-facing `torch.autograd.Function` at first. It is a
  gradient producer used by tests and benchmarks.
- Correct integration would use the op inside a custom objective autograd
  function:
  1. call v11/v12 raster forward to get `features, alpha`;
  2. call fused producer to get loss and upstream gradients;
  3. call raster backward with `grad_features, grad_alpha`;
  4. return gradients for Gaussian parameters and colorizer parameters.

Reason for this boundary:

- It gives a real kernel benchmark without touching the stable trainer or
  renderer dispatch.
- It keeps the mathematical contract narrow and easy to compare against
  PyTorch.
- It avoids mixing a new loss mode into `src/train/objective/` before the
  kernel proves useful.

## Memory Traffic Estimate

For `N=32,H=W=512,F=32`, the small producer lower-bound traffic is:

```text
read features      1024 MiB
read alpha           32 MiB
read target          96 MiB
read background      96 MiB
write grad_features 1024 MiB
write grad_alpha     32 MiB
loss/weight/bias atomics are tiny by byte count, though not free by latency
------------------------------------------------------------
producer dense IO  2304 MiB lower bound
```

Compared with the current PyTorch no-norm L1 path, v12a producer should save
the materialized RGB/logit/compose/delta graph and several MPS/autograd launches.
It does not yet save the 1 GiB raster feature write/read or 1 GiB grad-feature
write/read. A full raster+loss fusion is the path that could remove those.

Expected outcome:

- Small prototype: lower loss/colorize forward+backward wall time, modest total
  step improvement.
- Full fusion: only worth attempting if the producer is correct and launch/
  atomic overhead does not erase the dense-memory savings.

## Correctness Tests

Minimum local tests:

1. Synthetic MPS parity against PyTorch for `N=1,H=4,W=5,F=32`:
   - `loss_per_image`
   - `grad_features`
   - `grad_alpha`
   - `grad_weight`
   - `grad_bias`
2. Repeat with `N=2,H=16,W=16,F=32` and nontrivial alpha/background.
3. Exact-zero L1 sign case:
   - set prediction equal to target for some pixels;
   - gradient must be zero at those pixels to match PyTorch `abs`.
4. Batch scaling:
   - returned parameter/input gradients must match
     `reconstruction_loss_per_image(...).mean().backward()`, not only per-image
     losses.
5. Layout and dtype checks:
   - require MPS float32 contiguous tensors;
   - reject `F != 32` until generalized.
6. Raster boundary test after integration:
   - compare one v11 raster+PyTorch colorize/L1 backward against
     v11 raster+v12a producer gradients fed into raster backward.
7. Trainer fixed-render parity after integration:
   - fixed scene, fixed colorizer weights, fixed target/background;
   - compare loss and gradients for colorizer and Gaussian sequence tensors.

Tolerances:

- loss max diff <= `1e-5` on small tests;
- gradient max diff <= `2e-4` on MPS due to atomic order;
- colorizer parameter gradients may need looser `5e-4` at larger images.

## Benchmark Plan

Prototype-only benchmarks:

1. `N=4, H=W=128, F=32`, compare PyTorch no-norm L1 forward+backward against
   fused producer, warmup 3, iters 20.
2. `N=16, H=W=512, F=32`, compare the same row to the available v11 512
   trainer-breakdown scale.
3. `N=32, H=W=512, F=32`, run only if memory stays stable; this is the target
   byte-accounting row.

Metrics:

- producer wall time;
- PyTorch colorize+compose+L1 wall time;
- peak sampled memory if using a memory sampler;
- parity max/mean diffs for all gradients.

Integration benchmarks, only after prototype parity:

1. v11 raster + PyTorch no-norm L1 objective.
2. v11 raster + v12a fused producer + existing raster backward.
3. Same fixed-render trainer config as the v11 artifacts, with no-pre-norm L1.

Do not claim a training win from the producer microbenchmark alone.

## Failure Modes

- Atomics for `grad_weight` can serialize hard at large `N*H*W`. If this is the
  bottleneck, use a two-pass reduction: per-threadgroup partials
  `[num_groups,3,F]`, then a tiny reduction kernel.
- Writing dense `grad_features` can dominate runtime. If producer timing is only
  mildly better than PyTorch, the next useful step is full raster+loss fusion,
  not more producer polish.
- MPS `atomic_float` order will cause small nondeterministic differences. Parity
  thresholds must allow float-order noise but still catch sign/scale errors.
- L1 at exactly zero must match PyTorch's zero subgradient.
- Background broadcasting bugs are easy: first implementation should require a
  fully materialized `[N,3,H,W]` background tensor.
- Shape/layout confusion can silently transpose RGB/feature dimensions. Keep the
  op NHWF for features and NCHW for RGB tensors, matching the raster extension
  and trainer target conventions explicitly.
- Sigmoid saturation can make gradients tiny; parity tests should include
  moderate logits and saturated logits.
- This kernel is intentionally no-norm. Accidentally using it with LayerNorm,
  hidden MLP, or view conditioning would be mathematically wrong, not a small
  approximation.

## Current Implementation Decision

Implement v12a as an isolated copy of v11 plus a local fused-gradient producer
or, if the Metal hook cannot be completed cleanly in the current pass, a
scaffold that documents the missing C++/Metal hook names and keeps all code
inside the v12a variant. Do not wire this into `src/train/renderers/fast_mac.py`
or any checked-in trainer config until parity and benchmark artifacts exist.

## 2026-05-10 Benchmark Result

The v12a producer was implemented and benchmarked. It is favorable as a
pixel-loss gradient producer, but it is not yet trainer-integrated.

Artifacts:

- `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_v12a_fused_colorize_l1_N16_256_F32_convfair.json`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_v12a_fused_colorize_l1_N32_512_F32_convfair.json`

Results:

- `N=16,H=W=256,F=32`: fused median `8.79ms`; current trainer-style
  `permute -> Conv2d -> sigmoid -> compose -> L1 backward` median `52.05ms`;
  speedup `5.92x`.
- `N=32,H=W=512,F=32`: fused median `61.49ms`; trainer-style PyTorch path
  median `568.69ms`; speedup `9.25x`.

Parity on the 512 row stayed within float32 noise:

- loss abs `5.96e-08`
- `grad_features` max abs `3.33e-15`
- `grad_alpha` max abs `2.13e-14`
- `grad_weight` max abs `3.26e-08`
- `grad_bias` max abs `1.92e-09`

This proves the image-space no-norm L1 producer is worth integrating as a
separate fast path. It does not prove full trainer speedup until the returned
`grad_features` is fed into the existing raster backward from the training
objective.

## 2026-05-10 Trainer Benchmark Integration

The v12a producer now has a narrow opt-in PyTorch autograd wrapper at
`src/train/objective/v12a_fused_l1.py` and a benchmark flag in
`src/benchmarks/trainer_phase_benchmark.py`:

```bash
--v12a-fused-l1
```

This path computes no-norm sigmoid 1x1 colorize + alpha/background compose + L1
mean loss in the v12a Metal producer and returns precomputed gradients for:

- raster features
- raster alpha
- colorizer weight
- colorizer bias

It intentionally rejects LayerNorm/pre-norm, hidden MLPs, view conditioning,
non-sigmoid colorize, and non-MPS tensors.

Small MPS parity against PyTorch no-norm L1 matched:

- loss diff `0.0`
- feature grad max diff `1.46e-11`
- alpha grad max diff `1.75e-10`
- weight grad max diff `5.82e-10`
- bias grad max diff `4.37e-10`

### 128px 20-Step Result

Config base:
`src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_vjepa_full_relpose_features_F32_256_16f_8192splats_goodset_train0006_0014_holdout0005_alpha1_128_relpose_pairdelta012.jsonc`

Temporary benchmark configs:

- `/tmp/dynaworld_v12d_128/no_norm_l1.json`
- `/tmp/dynaworld_v12d_128/layer_norm_l1.json`

Common overrides:

- `render.render_size=128`
- `render.fast_mac.feature_variant=v11_features_gradcache_zero_bg_hostmeta_fixedbin`
- `losses.type=l1`
- W&B and media logging disabled
- `iters=20`, `warmup=1`

Full train-step medians:

- Torch no-norm: `346.16ms`
- Torch LayerNorm/pre-norm: `390.14ms`
- v12a fused no-norm: `334.34ms`

Fixed-render medians:

- Torch no-norm: `174.73ms`
- Torch LayerNorm/pre-norm: `265.58ms`
- v12a fused no-norm: `188.87ms`

Artifacts:

- `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_v12d_128_fullstep_torch_no_norm_l1_iters20.json`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_v12d_128_fullstep_torch_layer_norm_l1_iters20.json`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_v12d_128_fullstep_fused_no_norm_l1_iters20.json`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_v12d_128_fixedrender_torch_no_norm_l1_iters20.json`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_v12d_128_fixedrender_torch_layer_norm_l1_iters20.json`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_v12d_128_fixedrender_fused_no_norm_l1_iters20.json`

Interpretation:

- At 128px, v12a fused is clearly better than LayerNorm/pre-norm.
- At 128px fixed-render, v12a fused is slower than the simple Torch no-norm
  path. Producer overhead is not amortized enough.
- The earlier large `N32/512` producer win remains real, but promotion needs a
  256/512 fixed-render or trainer A/B, not just the 128px result.

## 2026-05-10 Metal DSSIM Integration

The same experimental v12a extension now contains an image-space DSSIM
forward+gradient op:

```text
dssim_forward_grad(prediction[N,C,H,W], target[N,C,H,W], window_size, c1, c2)
  -> loss_per_image[N], grad_prediction[N,C,H,W]
```

It matches the current PyTorch DSSIM contract: reflect padding, local
`avg_pool2d` means, and `0.5 * (1 - mean(ssim_map))`. It is not fused into the
rasterizer. The intended boundary is:

```text
prediction,target -> Metal DSSIM + grad_rgb -> colorize/alpha/raster VJP
```

Standalone benchmark summary:

- `N=16,128px`: Metal `7.60ms`, Torch `19.52ms`, `2.57x`
- `N=16,256px`: Metal `18.67ms`, Torch `64.80ms`, `3.47x`
- `N=16,512px`: Metal `70.45ms`, Torch `234.66ms`, `3.33x`

The shared RGB objective has an opt-in config toggle:

```json
"losses": {
  "type": "standard_gs",
  "dssim_backend": "metal"
}
```

Default remains `"torch"`. This was integrated in `src/train/objective/loss.py`
via `src/train/objective/metal_dssim.py`, not by forking a trainer.

128px trainer-phase benchmark on the current 3-camera F32 relpose config,
`colorize.pre_norm=false`, `standard_gs`, v11 feature rasterizer:

- fixed-render Torch DSSIM median total `269.87ms`
- fixed-render Metal DSSIM median total `229.01ms`
- full-step Torch DSSIM median total `482.32ms`
- full-step Metal DSSIM median total `434.64ms`

Artifacts:

- `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_metal_dssim_integration_fixedrender_torch_128_iters20.json`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_metal_dssim_integration_fixedrender_metal_128_iters20.json`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_metal_dssim_integration_fullstep_torch_128_iters20.json`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_metal_dssim_integration_fullstep_metal_128_iters20.json`

Caveat: the objective wrapper uses the Metal mean DSSIM gradient and expands the
mean value across the per-image loss vector. That is correct for the current
uniform `sum()/frame_count` and `.mean()` call sites, but it is not a true
per-image DSSIM breakdown for arbitrary per-image weighting.
