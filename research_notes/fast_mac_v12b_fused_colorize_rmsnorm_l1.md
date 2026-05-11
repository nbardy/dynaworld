# Fast-Mac v12b: Fused RMSNorm Colorize Alpha-Compose L1 Backward

Date: 2026-05-10

Scope:

- Design target: `third_party/fast-mac-gsplat/variants/v12b_fused_colorize_rmsnorm_l1/`.
- Preserve stable variants. This note does not change `v5`, `v5_features`, `v9`, `v10`, or `v11`.
- Narrow first implementation target: F-channel feature splatting with `F <= GSP_FEATURE_CAP`, no hidden colorizer layer, no view-conditioning, RGB L1 reconstruction, and alpha-aware background composition.
- Non-targets for v12b: SSIM/DSSIM fusion, GELU hidden colorizer fusion, view-ray conditioning, browser export format changes, and trainer dispatch wiring.

## Context

Current F32 feature-splatting runs are split across four expensive stages:

1. Fast-mac rasterizer emits rasterized features `X` and accumulated alpha `A`.
2. `FeatureToColor` optionally applies per-pixel `LayerNorm` over feature channels.
3. A 1x1 colorizer maps features to RGB, usually followed by sigmoid.
4. The objective composes with RGB background using alpha and computes reconstruction loss.

The existing `v5_features` alpha contract returns `(features, alpha)`, where `features` is `[B,H,W,F]` at the raw variant boundary and `alpha` is `[B,H,W]`. The trainer wrapper commonly permutes features to `[K,F,H,W]` for PyTorch colorization. The `v10`/`v11` lineage then optimizes feature-kernel metadata and binning while preserving the same external feature and alpha contract.

The v12b idea is not another binning tweak. It is a different backward boundary: compute the RGB loss derivative with respect to rasterized features and alpha in a single Metal path, and later push that derivative directly into the tile backward so `grad_features` and `grad_alpha` do not need to be materialized.

## Shapes And Units

Use:

- `B`: rendered frames/images in the launch.
- `H, W`: render height and width.
- `N = B * H * W`: rendered pixels.
- `F`: rasterized feature channels, expected `32` for current V-JEPA feature splatting.
- `C = 3`: RGB channels.
- `G`: Gaussian count per batch/image.
- `P`: Gaussian-pixel pairs visited by tile backward.

Tensor layout for the proposed Metal-side fused color/loss path:

```text
features_x     [B,H,W,F] float32 contiguous, bytes = 4*N*F
alpha_a        [B,H,W]   float32 contiguous, bytes = 4*N
target_rgb     [B,H,W,3] float32 contiguous or packed from [B,3,H,W], bytes = 12*N
bg_rgb         broadcast [1|B,H|1,W|1,3], worst pixel scope bytes = 12*N
rms_gamma      [F]       float32, bytes = 4*F
color_w        [3,F]     float32, bytes = 12*F
color_b        [3]       float32, bytes = 12
loss_per_image [B]       float32, bytes = 4*B
grad_w         [3,F]     float32, bytes = 12*F
grad_b         [3]       float32, bytes = 12
grad_gamma     [F]       float32, bytes = 4*F
```

Concrete memory sizes:

```text
B=16, H=W=256, F=32:
  N                 = 1,048,576 pixels
  features_x        = 128 MiB
  alpha_a           =   4 MiB
  target_rgb        =  12 MiB
  pixel-scope bg    =  12 MiB
  rgb/logits each   =  12 MiB
  grad_features     = 128 MiB
  grad_alpha        =   4 MiB

B=16, H=W=512, F=32:
  N                 = 4,194,304 pixels
  features_x        = 512 MiB
  alpha_a           =  16 MiB
  target_rgb        =  48 MiB
  pixel-scope bg    =  48 MiB
  rgb/logits each   =  48 MiB
  grad_features     = 512 MiB
  grad_alpha        =  16 MiB
```

The v12b near-term pixel-loss kernel still needs `features_x` because RMSNorm backward depends on the feature vector. It can avoid materializing normalized features, logits, RGB, composed RGB, and native PyTorch autograd intermediates. The later tile-backward integration can also avoid writing and rereading `grad_features` and `grad_alpha`.

## Why LayerNorm Pre-Norm Is Costly

`FeatureToColor(pre_norm=True)` uses `nn.LayerNorm(self.feature_dim)` per pixel. The implementation has to normalize over the channel dimension, so `[B,F,H,W]` is permuted to `[B,H,W,F]`, normalized, then permuted back before the 1x1 `Conv2d`.

The cost has three parts:

1. Layout churn: the colorizer operates on PyTorch `Conv2d` layout, while the rasterizer naturally emits `[B,H,W,F]`. The pre-norm path crosses that boundary with permutes and likely contiguous materialization at kernel boundaries.
2. LayerNorm statistics: per pixel, it computes both `mean(x)` and `variance(x) = mean((x - mean)^2)`. Backward needs reductions over `grad_y` and `grad_y * (x - mean)`, plus affine `gamma` and `beta` gradients.
3. Autograd state: native LayerNorm/Conv/Sigmoid/Compose/L1 creates a graph with several pixel-sized tensors or saved intermediates. At F32, the feature-sized tensors dominate memory traffic.

For `F=32`, LayerNorm needs at least:

```text
forward per pixel:
  read x[32]
  reduce sum(x) and sum(x^2) or sum((x - mean)^2)
  write y[32]

backward per pixel:
  read x[32], grad_y[32], gamma[32]
  reduce sum(grad_y * gamma)
  reduce sum(grad_y * gamma * normalized_x)
  write grad_x[32]
```

This is not bad in isolation. It becomes costly because the normalized feature tensor is only an intermediate for a tiny `3 x F` colorizer. The current pipeline pays full feature-image bandwidth for a subgraph that could fit in one pixel thread's registers.

## Why RMSNorm Changes The Tradeoff

RMSNorm removes mean subtraction:

```text
m     = eps + (1/F) * sum_i x_i^2
r     = rsqrt(m)
y_i   = gamma_i * x_i * r
```

Compared with LayerNorm:

- no per-pixel mean;
- no beta term in the minimal design;
- one reduction instead of mean plus centered variance;
- backward needs one dot product instead of two reductions;
- saved state can be just `r` per pixel, or no extra state if `r` is recomputed from `features_x`.

RMSNorm is not numerically equivalent to LayerNorm. It preserves feature-vector direction and normalizes scale, but it does not remove a common per-pixel feature offset. That is a quality risk, not a bug. It is the price paid for cheaper math and easier fusion.

## Forward Math

For one pixel, feature vector `x in R^F`, alpha `a`, target `t in R^3`, and background `bkg in R^3`:

```text
m       = eps + (1/F) * sum_i x_i^2
r       = rsqrt(m)
y_i     = gamma_i * x_i * r
z_c     = color_b_c + sum_i color_w[c,i] * y_i
s_c     = sigmoid(z_c)            # or identity for a debug mode
p_c     = a * s_c + (1 - a) * bkg_c
ell     = reduction_scale * sum_c abs(p_c - t_c)
```

For trainer parity with `objective/loss.py` L1, `reduction_scale = 1 / (N * 3)` for a scalar mean over all pixels and RGB channels. If the trainer applies a per-view loss weight outside this kernel, leave `reduction_scale` unweighted and multiply the returned scalar at the existing objective boundary.

## Backward Formulas

Use `sign0(u)` as `-1` for `u < 0`, `1` for `u > 0`, and `0` for `u == 0`. For each RGB channel:

```text
d_c       = p_c - t_c
g_p_c     = reduction_scale * sign0(d_c)
g_s_c     = g_p_c * a
g_alpha   = sum_c g_p_c * (s_c - bkg_c)
```

If sigmoid is enabled:

```text
g_z_c = g_s_c * s_c * (1 - s_c)
```

If identity activation is enabled:

```text
g_z_c = g_s_c
```

Colorizer parameter gradients:

```text
grad_color_b[c]   += g_z_c
grad_color_w[c,i] += g_z_c * y_i
```

Gradient into RMS-normalized vector:

```text
v_i = sum_c g_z_c * color_w[c,i]
u_i = v_i * gamma_i
```

RMS gamma gradient:

```text
grad_gamma[i] += v_i * x_i * r
```

RMSNorm feature gradient:

```text
dot = sum_j u_j * x_j
grad_x[i] = r * u_i - (x_i * r^3 / F) * dot
```

Checks:

- `grad_x` has shape `[F]` per pixel and becomes the rasterizer backward's `grad_features[pix,:]`.
- `g_alpha` has shape scalar per pixel and becomes the rasterizer backward's `grad_alpha[pix]`.
- With `gamma_i = 1`, RMSNorm still has the second term; a no-norm variant is not recovered by setting gamma to one. No-norm must be a separate mode with `y_i = x_i` and `grad_x = v_i`.

## Saved State Options

Option A, recompute `r` in backward:

```text
saved tensors:
  features_x [N,F]
  alpha_a    [N]
  target_rgb [N,3] or target handle
  bg_rgb     broadcast or sampled tensor
  color_w, color_b, gamma

extra saved per-pixel state:
  none
```

Backward reads `x[0:F]`, computes `sum(x^2)`, then continues. This costs one F-channel read and one reduction per pixel. Since the tile backward already needs per-pixel `grad_x[0:F]`, the compute is acceptable for F32 and keeps memory lower.

Option B, save `r` per pixel:

```text
extra saved per-pixel state:
  inv_rms [N] float32, bytes = 4*N
```

This saves one reduction in backward but adds a forward write and backward read. At `B=16,H=W=256`, that is 4 MiB. At 512, 16 MiB. This may be worth benchmarking because F32 reductions are not free, but it is not the first scaffold.

Option C, save normalized `y`:

```text
extra saved per-pixel state:
  normed_y [N,F] float32, bytes = 4*N*F
```

Reject for v12b. It recreates the large intermediate that fusion is meant to remove.

## Proposed Autograd Boundary

There are two implementation levels.

### Level 1: Pixel-Loss Kernel Plus Existing Raster Backward

Forward:

```text
features_x, alpha_a, raster_state = v11_raster_forward(...)
loss, maybe_rgb = v12b_fused_pixel_l1_forward(features_x, alpha_a, target_rgb, bg_rgb, color_params)
```

Backward:

```text
grad_x, grad_alpha, grad_color_w, grad_color_b, grad_gamma =
    v12b_fused_pixel_l1_backward(features_x, alpha_a, target_rgb, bg_rgb, color_params)

grad_gaussians = v11_raster_backward_saved(grad_x, grad_alpha, raster_state)
```

This is the safest prototype. It fuses colorize, alpha-compose, and L1 backward into one pixel kernel while leaving tile traversal and Gaussian gradients untouched.

### Level 2: Inline Pixel-Loss Math In Tile Backward

The v11 tile backward currently loads `grad_features[pix,:]` into a per-thread cache and reads `grad_alpha[pix]`. Replace that load with:

```text
compute_fused_colorize_l1_pixel_grad(
    x = features_x[pix,:],
    a = alpha[pix],
    target = target_rgb[pix,:],
    bg = bg_rgb[pix,:],
    color_params,
    out grad_cache[0:F],
    out alpha_grad
)
```

Then run the existing reverse Gaussian accumulation using `grad_cache` and `alpha_grad`.

This removes:

- write of `grad_features [N,F]`;
- read of `grad_features [N,F]` by tile backward;
- write/read of `grad_alpha [N]`;
- separate pixel-backward launch if fused directly.

It adds:

- reads of `features_x [N,F]`, `alpha [N]`, target/background, and colorizer parameters inside tile backward;
- global atomics or a separate reduction for `grad_color_w`, `grad_color_b`, and `grad_gamma`.

Level 2 is the desired fast path but has more risk because colorizer parameter gradients become reductions over all pixels and tile scheduling can revisit pixels only once in the current backward layout. The first Metal implementation should keep Level 1 until the formulas and parity are locked.

## Metal Kernel Pseudocode

### Level 1 Pixel Backward

One thread handles one pixel.

```text
kernel fused_rmsnorm_colorize_l1_backward(
    features_x[N,F],
    alpha[N],
    target_rgb[N,3],
    bg_rgb[...],
    gamma[F],
    color_w[3,F],
    color_b[3],
    meta,
    grad_x[N,F],
    grad_alpha[N],
    atomic grad_w[3,F],
    atomic grad_b[3],
    atomic grad_gamma[F],
    loss_per_image[B])
{
    pix = global_id
    if pix >= N: return

    // Load x into thread registers. F <= GSP_FEATURE_CAP.
    sum_sq = 0
    for f in 0..F-1:
        x[f] = features_x[pix, f]
        sum_sq += x[f] * x[f]

    r = rsqrt(eps + sum_sq / F)

    for f in 0..F-1:
        y[f] = gamma[f] * x[f] * r

    for c in 0..2:
        z[c] = color_b[c]
        for f in 0..F-1:
            z[c] += color_w[c,f] * y[f]
        s[c] = sigmoid(z[c])
        bg[c] = load_bg(pix, c)
        p[c] = alpha[pix] * s[c] + (1 - alpha[pix]) * bg[c]
        delta[c] = p[c] - target_rgb[pix,c]
        gp[c] = reduction_scale * sign0(delta[c])
        gz[c] = gp[c] * alpha[pix] * s[c] * (1 - s[c])
        local_loss += reduction_scale * abs(delta[c])

    galpha = sum_c gp[c] * (s[c] - bg[c])
    grad_alpha[pix] = galpha

    for c in 0..2:
        atomic_add(grad_b[c], gz[c])

    dot = 0
    for f in 0..F-1:
        v = 0
        for c in 0..2:
            v += gz[c] * color_w[c,f]
            atomic_add(grad_w[c,f], gz[c] * y[f])
        atomic_add(grad_gamma[f], v * x[f] * r)
        u[f] = v * gamma[f]
        dot += u[f] * x[f]

    r3_over_f = r * r * r / F
    for f in 0..F-1:
        grad_x[pix,f] = r * u[f] - x[f] * r3_over_f * dot

    atomic_add(loss_per_image[batch_of_pix], local_loss)
}
```

The first version can use atomics for `grad_w`, `grad_b`, and `grad_gamma` because the parameter tensors are tiny (`3*F + 3 + F = 4F + 3`, only 131 floats when `F=32`). If atomics become noisy or slow, use a two-pass block reduction.

### Level 2 Tile Backward Inline Hook

Inside the existing backward pixel thread:

```text
thread float grad_cache[GSP_FEATURE_CAP];
float alpha_grad = 0.0f;

if (pixel_active) {
    compute_fused_colorize_l1_pixel_grad(
        features_x,
        alpha,
        target_rgb,
        bg_rgb,
        gamma,
        color_w,
        color_b,
        pix,
        meta,
        grad_cache,
        alpha_grad);
} else {
    for f in 0..F-1:
        grad_cache[f] = 0.0f;
}

// Existing v11 reverse compositing consumes grad_cache and alpha_grad.
for g in reverse_tile_gaussians:
    ...
    combined = dot(grad_cache, feature_contribution_terms) + alpha_grad * alpha_terms
    ...
```

The key rule is to keep the existing alpha tuple math as the authority. The fused code only replaces how the image-space upstream gradients are produced.

## Memory Traffic Model

Approximate current PyTorch color/loss backward for `F=32`, ignoring cache reuse and native kernel overhead:

```text
forward color path:
  read features_x            4*N*F
  write normed features      4*N*F
  read normed features       4*N*F
  write logits               12*N
  write sigmoid rgb          12*N
  read alpha/rgb/bg          ~28*N
  write composed rgb         12*N

backward color path:
  read composed/target       24*N
  write/read grad rgb        12*N to 24*N
  sigmoid/conv/LN saved reads roughly O(4*N*F) again
  write grad_features        4*N*F
  write grad_alpha           4*N

raster backward:
  read grad_features         4*N*F
  read grad_alpha            4*N
```

Level 1 fused pixel backward:

```text
read features_x              4*N*F
read alpha/target/bg         4*N + 12*N + bg
read tiny params             hot cache
write grad_features          4*N*F
write grad_alpha             4*N
atomic tiny params           O(N*F) atomics or reduced later
read grad_features/alpha     raster backward still pays 4*N*F + 4*N
```

Level 2 inline tile backward:

```text
read features_x              4*N*F
read alpha/target/bg         4*N + 12*N + bg
no grad_features write/read
no grad_alpha write/read
existing Gaussian-gradient writes unchanged
```

At `B=16,H=W=256,F=32`, avoiding only `grad_features` write plus read saves about `256 MiB` of memory traffic per backward. At 512 it saves about `1 GiB`. That is before counting removed PyTorch intermediate traffic.

## Autograd Surface

Prototype Python-facing API:

```python
loss, aux = fused_rmsnorm_colorize_alpha_l1_loss(
    features_bhwf,      # [B,H,W,F]
    alpha_bhw,          # [B,H,W]
    target_bhw3,        # [B,H,W,3]
    background_bhw3,    # broadcastable to [B,H,W,3]
    color_weight_3f,    # [3,F]
    color_bias_3,       # [3]
    rms_gamma_f,        # [F]
    eps=1e-6,
    activation="sigmoid",
    reduction="mean",
)
```

Near-term implementation can be a pure PyTorch reference plus a custom-op stub. A later C++/Metal op should return:

```text
forward:
  loss scalar or [B] per-image loss
  optional rgb preview [B,H,W,3] only when requested

backward:
  grad_features [B,H,W,F] for Level 1
  grad_alpha [B,H,W] for Level 1
  grad_color_weight [3,F]
  grad_color_bias [3]
  grad_rms_gamma [F]
```

Do not replace the existing generic renderer API until v12b proves parity and a measurable local speedup. The first branch can live as an opt-in benchmark script under its variant directory.

## Test Plan

Minimum local tests inside the owned variant:

1. Formula parity:
   - Compare pure PyTorch reference against an explicit manual implementation for `B=2,H=5,W=7,F in {3,8,32}`.
   - Assert loss, `grad_features`, `grad_alpha`, `grad_w`, `grad_b`, and `grad_gamma` match within float32 tolerance.

2. Layer restriction checks:
   - Reject hidden colorizer layer.
   - Reject view-conditioning.
   - Reject `F > GSP_FEATURE_CAP`.
   - Reject background tensors that cannot broadcast to `[B,H,W,3]`.

3. L1 kink behavior:
   - Construct pixels where prediction equals target.
   - Assert sign-zero convention yields zero image-space gradient for exactly matched RGB channels.

4. Alpha composition:
   - `alpha=0` should make `grad_features` zero and `grad_alpha = sum_c g_p_c * (s_c - bg_c)`.
   - `alpha=1` should remove background from prediction and route all RGB loss through colorizer.

5. RMSNorm/no-norm contrast:
   - Verify no-norm mode equals `linear(x)` gradients.
   - Verify RMSNorm mode does not pretend to match no-norm at `gamma=1`.

6. Existing rasterizer contract after wiring:
   - Run the inherited feature/alpha checks from v11.
   - Add one fused Level 1 test that calls `rasterize -> fused loss -> backward` and confirms finite gradients on means2d, conics, colors, opacities, color weights, bias, and gamma.

When the Metal kernel exists:

```bash
( cd third_party/fast-mac-gsplat/variants/v12b_fused_colorize_rmsnorm_l1
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
PYTHONPATH=third_party/fast-mac-gsplat/variants/v12b_fused_colorize_rmsnorm_l1 \
  .venv/bin/python third_party/fast-mac-gsplat/variants/v12b_fused_colorize_rmsnorm_l1/tests/fused_colorize_l1_reference_check.py
```

## Benchmark Plan

Keep the first benchmark matrix bounded and sequential to avoid sloppy shader crashes:

```text
Devices:
  local MPS only first

Cases:
  B=1,  H=W=128, F=32, G=1024, warmup=3, iters=10
  B=16, H=W=256, F=32, G=8192, warmup=3, iters=10
  B=16, H=W=512, F=32, G=8192, warmup=2, iters=5

Variants:
  baseline v11 raster + PyTorch LayerNorm colorizer + compose + L1
  v11 raster + PyTorch RMSNorm reference + compose + L1
  v12b Level 1 fused pixel-loss backward + v11 raster backward
  v12b Level 2 inline tile backward, only after Level 1 parity

Metrics:
  total forward+backward wall time
  raster forward
  color/loss forward
  color/loss backward
  raster backward
  max absolute drift in loss and all gradients
  allocated/current MPS memory if available
```

Acceptance gates:

- Level 1 must match PyTorch RMSNorm reference gradients at `F=3,8,32`.
- Fused RMSNorm quality must be compared to no-norm and LayerNorm in held-out-camera metrics before treating it as a model improvement.
- A timing win is only meaningful if measured against the same target/background/reduction and includes colorizer parameter gradients.

## Quality Risks Versus No-Norm

RMSNorm improves scale stability but changes the representation contract:

- It removes feature-vector magnitude as a direct colorizer signal. If alpha or density already encodes confidence, this may be fine. If feature magnitude carries useful shading or visibility cues, quality can drop.
- It does not subtract the channel mean. LayerNorm can suppress a per-pixel DC offset; RMSNorm cannot. If the F32 collapse mode is partly a common feature bias, RMSNorm may be weaker than LayerNorm.
- It has no beta in the minimal design. A learned colorizer bias still exists after the 1x1, but there is no per-feature post-norm shift.
- It can saturate sigmoid differently. Unit RMS does not mean unit variance after learned `gamma`; orthogonal gain may need retuning.
- It is harder to compare against the existing browser/runtime path. Exact LayerNorm could not be folded into a fixed 1x1 matrix; RMSNorm also remains input-dependent, but its runtime implementation is simpler.
- L1-only fusion is narrower than `standard_gs`; if the training config relies on DSSIM, the fused loss is an ablation, not a drop-in replacement.

Therefore v12b should be framed as a performance and stability experiment, not as an automatic replacement for no-norm or LayerNorm. The selector remains held-out-camera quality plus measured end-to-end training speed.

## Implementation Notes For v12b Scaffold

The smallest useful scaffold should:

- copy the current best standalone feature variant lineage (`v11_features_gradcache_zero_bg_hostmeta_fixedbin`) into a new opt-in directory;
- rename the Python package, custom op namespace, and Metal source so it can build without colliding with v11;
- add a pure PyTorch RMSNorm+1x1+alpha-compose+L1 reference module with explicit shape checks;
- add a reference-gradient test that exercises the exact formulas above;
- document that the Metal fused backward is not implemented yet.

This keeps stable variants intact and gives the next kernel pass a precise parity target.
