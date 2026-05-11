# Fast-Mac v12c: Fused Raster Color Loss Backward

Date: 2026-05-10

Scope: Dynaworld fast-mac feature splatting on Apple Metal/MPS. This note
designs an opt-in `v12c_fused_raster_color_loss_backward` variant whose
backward path avoids materializing the dense `grad_features[B,H,W,F]` image.
The forward path may still produce rasterized features and accumulated alpha for
diagnostics, color videos, and the current trainer contract. The intended
backward path computes the RGB reconstruction gradient at each pixel, runs the
feature-to-color and alpha-composition VJP locally, and immediately feeds that
local pixel gradient into the existing reverse contributor loop.

## Claim Boundary

The design target is not a replacement for stable fast-mac variants yet.

- Keep stable variants untouched.
- Start from the v11 feature fork shape: F-channel native rasterization,
  accumulated alpha output, host metadata split, grad-cache backward, zero
  feature-background tail skip, and fixed-cap no-overflow binning.
- Add a fused backward surface that is opt-in and narrow at first.
- The first feasible kernel should support `FeatureToColor(hidden_dim=None,
  activation="sigmoid", pre_norm=False, view_condition="none")` plus mean MSE
  reconstruction and RGB background composition.
- Broader support for L1, L1+MSE, LayerNorm, view conditioning, hidden GELU
  colorizers, and DSSIM should be added only after the narrow MSE path has
  parity and timing proof.

Current confidence: medium for the math/topology, low for speed until measured.
The risk is not that the VJP is exotic; it is that fusing the VJP into the
per-tile raster backward increases register pressure, atomics, or barrier cost
enough to erase the saved `grad_features` traffic.

## Current Unfused Path

For feature splatting, the current trainer does roughly:

```text
splatted_features, alpha = raster_forward(gaussians)       # [B,F,H,W], [B,H,W]
splat_rgb = FeatureToColor(splatted_features, view_dirs)   # [B,3,H,W]
pred_rgb = alpha * splat_rgb + (1 - alpha) * bg_rgb
loss = reconstruction_loss(pred_rgb, target_rgb)
loss.backward()
```

PyTorch autograd computes:

```text
grad_features = d loss / d splatted_features   # dense [B,H,W,F]
grad_alpha = d loss / d alpha                  # dense [B,H,W]
```

Then the Metal raster backward reads `grad_features[pix, :]` and `grad_alpha[pix]`
while walking the same pixel contributors in reverse depth order. The v9-v11
grad-cache path reduces repeated `grad_features[pix, f]` device reads by loading
one pixel's F-vector into thread-local memory. It still requires PyTorch to
materialize and store the full dense gradient image before raster backward.

The v12c idea removes that dense gradient image from the main train backward:

```text
splatted_features, alpha, tile_state = raster_forward_state(gaussians)
fused_backward(
    splatted_features,
    alpha,
    target_rgb,
    bg_rgb,
    colorizer_params,
    loss_params,
    tile_state,
    gaussians,
) -> gaussian_grads, colorizer_grads
```

Inside `fused_backward`, each pixel thread computes its own local
`dL/dfeature[pix, :]` and `dL/dalpha[pix]`, keeps those values in registers or a
thread-local cache, then immediately runs the contributor-loop VJP. There is no
global `grad_features` tensor on the critical path.

## Forward Contract

The forward kernel remains normal feature rasterization:

```text
Input splats per batch:
    means2d      [B*G, 2] float32
    conics       [B*G, 3] float32, packed (a,b,c)
    features     [B*G, F] float32
    opacities    [B*G] float32
    depths       [B,G] float32, used by Python depth sort before flattening

Raster metadata:
    meta_i32     MPS int32
    meta_f32     MPS float32
    meta_host_*  CPU copies used by bridge allocation/validation

Binning/tile state:
    tile_counts      [B*tiles_y*tiles_x] int32
    tile_offsets     [B*tiles_y*tiles_x + 1] int32
    binned_ids       fixed [tile_count * max_fast_pairs] int32 in v11/v12c
    tile_stop_counts [tile_count] int32 from forward-state kernel

Forward outputs:
    splat_features   [B,H,W,F] float32, BHWF in the extension
    alpha            [B,H,W] float32, alpha = 1 - T_final
```

The forward output is still useful. It preserves:

- existing diagnostic videos and feature PCA logging
- existing colorize probes
- parity checks against v5/F=3 and v5_features/F32 references
- a fallback route for unsupported fused losses

The fused design only changes the train backward owner.

## Reconstruction Math

For one pixel `p`, let:

```text
z      = splat_features[p] in R^F
a      = accumulated alpha[p]
b      = RGB background[p] in R^3
y      = target RGB[p] in R^3
c(z)   = colorizer output in R^3
r      = composed prediction = a * c(z) + (1 - a) * b
L      = reconstruction loss
```

Composition VJP:

```text
g_r = dL/dr
g_c = a * g_r
g_a = dot(g_r, c(z) - b)
```

For the first prototype, colorizer is one 1x1 conv plus sigmoid:

```text
u_c = bias_c + sum_f W[c,f] * z_f
c_c = sigmoid(u_c)
```

VJP:

```text
g_u_c = g_c_c * c_c * (1 - c_c)
g_z_f = sum_c W[c,f] * g_u_c
g_W[c,f] += g_u_c * z_f
g_bias[c] += g_u_c
```

Mean MSE over `[B,3,H,W]`:

```text
L = mean((r - y)^2)
g_r_c = 2 * (r_c - y_c) / (B * H * W * 3)
```

For weighted view losses, multiply `g_r` by the scalar view weight before the
colorizer/compose VJP.

For L1:

```text
g_r_c = sign(r_c - y_c) / (B * H * W * 3)
```

At zero residual, PyTorch's `abs` subgradient is zero. The kernel must match
that for parity.

For `l1_mse`, add the weighted L1 and MSE gradients. This is easy to fuse after
MSE parity.

## Raster Backward Math

For sorted contributors `i = 0..N-1` at a pixel:

```text
alpha_i = clamp(opacity_i * exp(power_i), max_alpha)
w_i     = T_i * alpha_i
T_{i+1} = T_i * (1 - alpha_i)
z       = sum_i w_i * feature_i + T_N * feature_bg
a_out   = 1 - T_N
```

Current fast-mac backward reconstructs `T_final`, then walks contributors in
reverse. It carries a transmittance adjoint `gT` initialized from the background
tail:

```text
gT = dot(g_z, feature_bg)
```

For each contributor in reverse:

```text
T_prev = T_cur / max(1 - alpha_i, eps)
dot_gc = dot(g_z, feature_i) + g_a_out
g_alpha_i = T_prev * (dot_gc - gT)
g_feature_i += T_prev * alpha_i * g_z
gT = alpha_i * dot_gc + (1 - alpha_i) * gT
T_cur = T_prev
```

Then alpha's local derivatives give:

```text
power = -0.5 * (a*dx^2 + 2*b*dx*dy + c*dy^2)
raw_alpha = opacity * exp(power)
alpha = min(max_alpha, raw_alpha)

if raw_alpha < max_alpha:
    g_raw = g_alpha_i
else:
    g_raw = 0

g_power = g_raw * raw_alpha
g_opacity += g_raw * raw_alpha / max(opacity, eps)
g_conic_a += g_power * (-0.5) * dx^2
g_conic_b += g_power * (-1.0) * dx * dy
g_conic_c += g_power * (-0.5) * dy^2
g_mean_x  += -g_power * (-(a*dx + b*dy))
g_mean_y  += -g_power * (-(b*dx + c*dy))
```

v12c keeps this contributor-loop math and replaces only the source of `g_z` and
`g_a_out`: they are computed locally from colorize/compose/loss instead of read
from dense image-gradient buffers.

## Key Metal Pseudocode

### Forward State Kernel

Forward stays close to v11:

```metal
kernel tile_fast_forward_state(
    tile_counts, tile_offsets, binned_ids,
    means2d, conics, features, opacities,
    meta_i32, meta_f32,
    out_features, out_alpha, out_stop_counts)
{
    tile = threadgroup_position_in_grid;
    tid = thread_position_in_threadgroup;
    pix = pixel_index_for(tile, tid);

    if invalid_pixel:
        return;

    zero_or_background_initialize(out_features[pix]);

    T = 1.0;
    tile_stop = 0;
    for chunk in binned contributors front_to_back:
        load means/conics/opacities to threadgroup memory;
        if all pixels saturated:
            break;
        for contributor in chunk:
            alpha = eval_alpha(pixel_center, gaussian);
            if alpha below threshold:
                continue;
            out_features[pix, f] += T * alpha * feature[g, f];
            T *= (1 - alpha);
            tile_stop = max(tile_stop, contributor_index + 1);
            if T <= transmittance_threshold:
                break;

    if feature background is nonzero:
        out_features[pix, f] += T * feature_bg[f];
    out_alpha[pix] = 1 - T;
    out_stop_counts[tile] = threadgroup_max(tile_stop);
}
```

### Pixel Gradient Microkernel

This can be a standalone debug kernel first, but the production fused path
should inline it into raster backward:

```metal
inline PixelGrad pixel_grad_linear_sigmoid_mse(
    pix,
    out_features, out_alpha,
    target_rgb, background_rgb,
    color_w, color_b,
    loss_params,
    meta)
{
    float z[GSP_GRAD_CACHE_CAP];
    for f in 0..F-1:
        z[f] = out_features[pix, f];

    a = out_alpha[pix];
    for c in 0..2:
        logit[c] = color_b[c];
        for f in 0..F-1:
            logit[c] += color_w[c, f] * z[f];
        splat_rgb[c] = sigmoid(logit[c]);
        pred[c] = a * splat_rgb[c] + (1 - a) * background_rgb[pix, c];
        g_pred[c] = loss_scale * 2 * (pred[c] - target_rgb[pix, c]) / (B*H*W*3);
        g_splat_rgb[c] = a * g_pred[c];
        g_alpha += g_pred[c] * (splat_rgb[c] - background_rgb[pix, c]);
        g_logit[c] = g_splat_rgb[c] * splat_rgb[c] * (1 - splat_rgb[c]);

    for f in 0..F-1:
        g_feature[f] = sum_c color_w[c, f] * g_logit[c];

    // Optional trainable colorizer parameter gradients.
    for c in 0..2:
        atomic_add(g_color_b[c], g_logit[c]);
        for f in 0..F-1:
            atomic_add(g_color_w[c, f], g_logit[c] * z[f]);

    return {g_feature, g_alpha};
}
```

### Fused Raster Backward Kernel

The fused kernel is the existing tile backward with local pixel-gradient
generation inserted before the reverse contributor walk:

```metal
kernel tile_fast_backward_fused_linear_sigmoid_mse(
    out_features, out_alpha,
    target_rgb, background_rgb,
    color_w, color_b, loss_params,
    tile_counts, tile_offsets, binned_ids, tile_stop_counts,
    means2d, conics, splat_features, opacities,
    meta_i32, meta_f32,
    g_means2d, g_conics, g_splat_features, g_opacities,
    g_color_w, g_color_b)
{
    tile = threadgroup_position_in_grid;
    tid = thread_position_in_threadgroup;
    pix = pixel_index_for(tile, tid);

    load fixed binned ids up to tile_stop_count;
    recompute T_final and per-pixel end_i exactly as current v11 backward;

    PixelGrad pg = pixel_grad_linear_sigmoid_mse(...);

    float g_feature[GSP_GRAD_CACHE_CAP];
    copy pg.g_feature into thread local cache;
    float g_alpha_out = pg.g_alpha;

    gT = dot(g_feature, feature_bg);
    T_cur = T_final;

    for contributor chunks reverse:
        load means/conics/opacities to threadgroup;
        for contributor reverse within chunk:
            if pixel valid and contributor before end_i and alpha active:
                T_prev = T_cur / max(1 - alpha, eps);
                dot_gc = dot(g_feature, splat_features[g]) + g_alpha_out;
                g_alpha_i = T_prev * (dot_gc - gT);
                g_splat_features[g, f] += T_prev * alpha * g_feature[f];
                compute mean/conic/opacity gradients from g_alpha_i;
                gT = alpha * dot_gc + (1 - alpha) * gT;
                T_cur = T_prev;

            reduce per-thread scalar gradients across the threadgroup;
            atomic_add gaussian grads;
            reduce feature grads per channel before global atomics;
}
```

The central invariant: every value that would have been loaded from
`grad_features[pix, :]` in v11 is instead produced by `pixel_grad_*` and held in
thread-local storage.

## Exact Buffer Plan

### Inputs Kept From v11

```text
means2d           [BG,2]       MPS float32
conics            [BG,3]       MPS float32
splat_features    [BG,F]       MPS float32, per-Gaussian feature/color tensor
opacities         [BG]         MPS float32
meta_i32          [15]         MPS int32
meta_f32          [4+Fcap]     MPS float32
meta_host_i32     [15]         CPU int32
meta_host_f32     [4+Fcap]     CPU float32
tile_counts       [T]          MPS int32
tile_offsets      [T+1]        MPS int32
binned_ids        [T*cap]      MPS int32 in fixedbin mode
tile_stop_counts  [T]          MPS int32
```

### New Fused Backward Inputs

```text
out_features      [B,H,W,F]    MPS float32, raster forward output, BHWF
out_alpha         [B,H,W]      MPS float32
target_rgb        [B,H,W,3]    MPS float32, BHWC contiguous
background_rgb    [B,H,W,3]    MPS float32, BHWC contiguous
color_weight      [3,F]        MPS float32, first prototype only
color_bias        [3]          MPS float32, first prototype only
loss_params       [N]          MPS float32
```

For the first prototype:

```text
loss_params[0] = scalar loss multiplier
```

The kernel derives the mean-MSE normalizer from metadata:

```text
normalizer = 2 * loss_params[0] / (B * H * W * 3)
```

### Outputs

```text
g_means2d         [BG,2]       MPS float32
g_conics          [BG,3]       MPS float32
g_splat_features  [BG,F]       MPS float32
g_opacities       [BG]         MPS float32
g_color_weight    [3,F]        MPS float32
g_color_bias      [3]          MPS float32
```

### Explicitly Avoided In Main Path

```text
grad_features     [B,H,W,F]    not allocated
grad_alpha        [B,H,W]      not allocated
grad_rgb          [B,H,W,3]    not allocated
```

Debug kernels may materialize these buffers only for parity tests.

## Colorizer Support Ladder

1. `hidden_dim=None`, `activation="sigmoid"`, `pre_norm=False`,
   `view_condition="none"`:
   - Straight line VJP.
   - Small parameter gradient buffers `[3,F]` and `[3]`.
   - First implementation target.

2. `activation="identity"`:
   - Easier than sigmoid; remove sigmoid derivative.
   - Useful for debugging because it is linear.

3. L1 and L1+MSE reconstruction:
   - Easy pixel-loss VJP.
   - Needs exact zero-residual subgradient parity.

4. `view_condition != "none"`:
   - Input to colorizer is `[z, view_dir]`.
   - If `detach_view_condition=True`, do not produce view-dir gradients.
   - If false, fused kernel must write view-dir gradients or decline fusion.
   - `pixel_ray` requires a `[B,H,W,3]` view-dir buffer in BHWC layout.

5. `pre_norm=True`:
   - Requires per-pixel LayerNorm VJP over F features.
   - Need compute mean, variance, inv_std, gamma/beta VJP.
   - Adds reductions over F and parameter atomics for gamma/beta.

6. `hidden_dim != None`:
   - Requires first-layer activations or recompute.
   - Need GELU VJP and two parameter-gradient reductions.
   - Register pressure grows with hidden size; likely only feasible for small
     hidden dimensions unless split into pixel-gradient and raster-backward
     passes.

7. DSSIM:
   - Not a local per-pixel loss. See below.

## Why SSIM/DSSIM Complicates Fusion

`standard_gs` uses:

```text
loss = l1_weight * L1(pred, target) + dssim_weight * DSSIM(pred, target)
DSSIM = 0.5 * (1 - SSIM)
```

The current SSIM implementation computes local window means using reflect
padding and average pooling:

```text
mu_x       = avg_pool(pred)
mu_y       = avg_pool(target)
sigma_x2  = avg_pool(pred^2) - mu_x^2
sigma_y2  = avg_pool(target^2) - mu_y^2
sigma_xy  = avg_pool(pred*target) - mu_x*mu_y
ssim_map  = ((2*mu_xy+c1)*(2*sigma_xy+c2)) /
            ((mu_x^2+mu_y^2+c1)*(sigma_x2+sigma_y2+c2))
```

Therefore `dL/dpred[p]` depends on every SSIM window that includes pixel `p`,
not only on pixel `p` itself. A single raster tile thread cannot compute the
correct DSSIM pixel gradient from only that pixel's target, prediction, alpha,
and colorized feature. Correct DSSIM fusion needs at least one of these:

1. Precompute `grad_rgb = dDSSIM/dpred` in a separate image-space kernel, then
   feed that RGB gradient into fused colorize/compose/raster backward.
   This avoids `grad_features[B,H,W,F]` but still materializes
   `grad_rgb[B,H,W,3]`.
2. Fuse SSIM local-stat computation and backward as a separate tiled image
   pipeline, with halo handling and reflect padding, then call raster backward.
   This is larger than the current v12c target.
3. Decline fusion for `standard_gs` until MSE/L1 paths prove useful.

Recommended boundary: v12c should initially support `mse`, then `l1_mse`, and
should explicitly fall back for `standard_gs` or require a precomputed
`grad_rgb` bridge. Do not approximate DSSIM inside the raster tile; that would
silently change the objective.

## Relation To FasterGS

FasterGS-style acceleration themes are relevant, but this design is not simply
"use FasterGS":

- FasterGS emphasizes reducing redundant splat work, pruning low-value
  contributors, and improving raster scheduling.
- Existing fast-mac F32 work already showed that contributor counts and
  backward scheduling matter: alpha threshold changes binned candidates, while
  transmittance threshold alone often leaves backward paying tile max depth.
- v12c targets a different but adjacent cost: the boundary between rasterizer
  and neural color/loss code. It removes the dense image-gradient handoff from
  PyTorch to Metal and turns the raster backward into the owner of the local
  reconstruction VJP.
- This is most FasterGS-like in spirit where the system stops doing work for
  values that are only intermediates. `grad_features` is such an intermediate:
  useful for modular autograd, but not semantically needed after the raster
  contributor loop consumes it.

If FasterGS-style pruning is later added, v12c still benefits: fewer
contributors means the fused backward has less reverse-loop work. But pruning
does not remove the `grad_features` materialization by itself.

## Correctness Tests

### Pixel VJP Tests

Use tiny CPU/MPS tensors and compare against PyTorch autograd for:

- MSE, linear sigmoid colorizer, background RGB, random alpha.
- `alpha=0`, `alpha=1`, `background == splat_rgb`, and zero residual.
- F values: `1`, `3`, `8`, `32`.
- Bias-only colorizer and zero colorizer weights.
- Nonzero loss scalar/view weight.

Expected checks:

```text
max_abs(g_feature_fused - g_feature_ref) <= 1e-5 for F<=32
max_abs(g_alpha_fused - g_alpha_ref) <= 1e-5
max_abs(g_W_fused - g_W_ref) <= 1e-5 or relaxed for MPS atomic order
max_abs(g_b_fused - g_b_ref) <= 1e-5 or relaxed for MPS atomic order
```

### Raster Fused Backward Tests

Compare fused output gradients against the unfused path:

```text
out_features, alpha = raster_forward_state(...)
rgb = sigmoid(conv1x1(out_features))
pred = alpha * rgb + (1-alpha) * bg
loss = mse(pred, target)
loss.backward()
```

Then compare:

- `means2d.grad`
- `conics.grad`
- `splat_features.grad`
- `opacities.grad`
- `color_weight.grad`
- `color_bias.grad`

Cases:

- Single Gaussian, one pixel, analytic hand check.
- Tiny `H=5,W=6,G=4,F=3`.
- F32 path with `H=16,W=16,G=24,F=32`.
- Batched path `B=2`.
- Saturated alpha clamp case, checking raw-alpha clamp gate.
- Zero background and nonzero RGB background.
- `inputs_sorted_by_depth=True` and default sorted path, if exposed.
- Active-tile path later; first prototype can be fast-tile only.
- Overflow path should raise in v12c fixedbin prototype.

### Trainer Contract Tests

Before trainer wiring:

- Existing `feature_contract_check.py` copied to v12c should still pass for
  normal unfused rasterization.
- A new prototype parity script should build random MPS tensors, run the
  fused MSE backward op, and compare against PyTorch autograd.
- F=3 v5 parity remains a smoke gate for inherited forward behavior.

After trainer wiring:

- 1-step F=32 smoke with offline W&B, because it exercises validation,
  colorize, alpha logging, and feature PCA.
- Fixed-render trainer parity at 256px against v11 or the current stable
  feature variant, with sequence-grad drift tolerance recorded.

## Benchmark Gates

Do not promote v12c on a single tiny direct benchmark. Use staged gates:

1. Compile/import:
   - build extension from the v12c directory using the dynaworld project.
   - import package and list custom ops.

2. Direct synthetic parity:
   - `B=1,H=16,W=16,G=32,F=8`, MSE fused vs unfused.
   - `B=2,H=32,W=32,G=128,F=32`, MSE fused vs unfused.

3. Direct timing:
   - Compare unfused `forward + colorize + compose + loss.backward()` versus
     `forward_state + fused_backward` on the same tensors.
   - Target rows:
     - `128px B16 G8192 F32`
     - `256px B16 G8192 F32`
     - `512px B16 G8192 F32`
   - Record forward, pixel/color/loss, raster backward, total, peak sampled
     MPS memory if available.

4. Trainer fixed-render timing:
   - Same fixed decoded Gaussian sequence and target frames.
   - Compare total backward, raster/colorize/loss split, and full step time.
   - Save JSON under `benchmark_outputs/fast_mac_feature_kernels/`.

5. Quality gate:
   - Only after timing wins, run the current 256px DeepView F32 goodset config
     with heldout camera metrics.
   - W&B must stay enabled for real benchmark runs.
   - Use heldout PSNR/SSIM/L1 and media grids as selector; source-view PSNR
     alone is not enough.

Promotion threshold proposal:

```text
parity: max gradient drift <= 1e-4 on F32 MPS direct tests
timing: >= 10% train backward or full step win on 256px/B16/G8192/F32
quality: no heldout metric regression outside normal seed noise
```

If the fused kernel is within noise or slower, leave the note and stop.

## Failure Modes

- Register pressure: local `g_feature[F]`, logits, RGB gradients, and raster
  VJP state may reduce occupancy enough to lose.
- Colorizer parameter atomics: every pixel contributes to `[3,F]` and `[3]`.
  For F32 this is only 96+3 parameters, but global atomics from every pixel can
  still serialize. A two-stage reduction may be needed.
- Objective drift: approximating DSSIM, L1 zero subgradient, sigmoid clamp, or
  LayerNorm eps would silently change training.
- Unsupported colorizer configs: hidden layers, view dirs, or LayerNorm must
  fail loudly until implemented.
- Background semantics: feature background, RGB background, and alpha
  composition are separate. The fused kernel must use RGB background in compose
  and feature background only for raster transmittance tail math.
- Alpha double counting: `grad_alpha` from RGB composition must be added once
  to `dot(g_z, feature_i)` in the raster backward formula. Adding it both in
  pixel VJP and contributor loop would over-scale opacity/geometry grads.
- Layout mistakes: extension forward is BHWF; trainer colorize uses BFHW.
  Fused buffers should standardize on BHWC/BHWF inside Metal and convert at
  Python boundaries.
- Active/overflow path mismatch: first prototype should restrict to fast
  fixedbin no-overflow. Active tiles and overflow segments can be added later.
- Stop-count mismatch: fused backward must use the same `tile_stop_counts` and
  per-pixel `end_i` recomputation as v11. Per-pixel early exit cannot create
  nonuniform barrier control.
- Sorting/unsorting: if the Python wrapper sorts by depth, fused gradients must
  be unsorted exactly like the inherited autograd wrapper.
- Colorizer optimizer state: if fused backward bypasses PyTorch autograd for
  colorizer parameters, the wrapper must still return or assign gradients in a
  form optimizers see.

## Implementation Plan For The Owned Prototype

1. Copy v11 into
   `third_party/fast-mac-gsplat/variants/v12c_fused_raster_color_loss_backward`.
2. Rename package, op namespace, and Metal kernel prefix so it can coexist with
   v11.
3. Preserve inherited raster APIs and tests.
4. Add an explicit prototype API:

```python
fused_linear_sigmoid_mse_backward(
    means2d,
    conics,
    splat_features,
    opacities,
    depths,
    target_rgb,
    color_weight,
    color_bias,
    config,
    background_rgb=None,
    loss_scale=1.0,
) -> FusedLinearSigmoidMSEBackwardResult
```

5. The prototype should:
   - run inherited forward-state rasterization
   - call one new fast-path Metal fused backward op
   - return forward features/alpha plus splat and colorizer gradients
   - raise for overflow, active tiles, F > `GSP_GRAD_CACHE_CAP`, non-MPS, or
     unsupported shapes
6. Add a small parity script under the variant directory that compares the
   prototype against PyTorch autograd for tiny MPS cases.

This is intentionally a scaffold, not trainer integration. Trainer dispatch
should wait until direct parity and timing justify touching `src/train`.

## 2026-05-10 Prototype Benchmark Result

The narrow v12c prototype was implemented:

- MPS only
- fixedbin/no-overflow fast tiles only
- `F <= 32`
- linear 1x1 sigmoid colorizer
- mean MSE
- no active-tile fused path
- no overflow fused path
- no LayerNorm/RMSNorm
- no hidden colorizer or view conditioning
- no L1/DSSIM fusion

Correctness is good. The parity script
`third_party/fast-mac-gsplat/variants/v12c_fused_raster_color_loss_backward/tests/fused_linear_sigmoid_mse_check.py`
passes F3/F8/F32 tiny MPS cases; worst observed drift was around `1e-9`.

Timing is not favorable yet:

- `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_v12c_fused_linear_sigmoid_mse_B4_G2048_F32_128.json`
  - fused median `33.52ms`
  - unfused trainer-style median `35.87ms`
  - speedup `1.07x`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_v12c_fused_linear_sigmoid_mse_B16_G8192_F32_256.json`
  - fused median `653.64ms`
  - unfused trainer-style median `430.33ms`
  - speedup `0.66x`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_v12c_fused_linear_sigmoid_mse_B16_G8192_F32_256_freeze_colorizer.json`
  - fused median `652.77ms`
  - unfused trainer-style median `436.56ms`
  - speedup `0.67x`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_v12c_fused_linear_sigmoid_mse_B16_G8192_F32_512.json`
  - fused median `893.01ms`
  - unfused trainer-style median `539.58ms`
  - speedup `0.60x`

Freezing colorizer parameter gradients did not recover the win, so the first
hypothesis that `[3,F]` global atomics were the main issue was wrong. The deeper
problem is that the prototype keeps the full forward feature image, then runs
scalar per-pixel colorize/compose/loss math inside the tile backward. PyTorch's
image-space `Conv2d` path is highly optimized enough that avoiding the
`grad_features` image does not pay for the slower scalar fused math.

The v12c lesson is not "full fusion is impossible"; it is that this first
layout is the wrong fusion boundary. A better next attempt should either:

1. keep colorize/loss as a fast image-space kernel and fuse only the
   `grad_features -> raster backward` consumption path, or
2. make the raster backward consume a precomputed RGB/pixel gradient image
   (`dL/dpred`) and do only the small RGB-to-feature VJP locally, or
3. implement a dedicated image-space Metal colorize/loss kernel with vectorized
   reductions, then feed its outputs into the raster VJP.

Do not promote v12c as implemented. Keep it as a correctness scaffold and a
negative benchmark that explains the next kernel boundary.
