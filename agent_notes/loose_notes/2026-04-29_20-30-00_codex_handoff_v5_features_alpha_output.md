# Codex Handoff: v5_features → expose accumulated alpha as a second output

Date: 2026-04-29
Audience: Codex (or another agent better suited to Metal kernel work)

## TL;DR

Extend the **v5_features** rasterizer fork to emit `accumulated_alpha = 1 - T_final` as a **second output** alongside `out_features`. The forward kernel already tracks `T_final` per pixel; we just need to write it to a separate output buffer and thread the corresponding gradient through backward.

The math is already there — this is plumbing, not new geometry. But **gradient correctness is critical**: the dynaworld trainer is going to use this alpha mask in the loss path, and the gradient from `dL/dalpha` must propagate correctly back to **opacities, means2d, and conics** so splats can learn to move into and grow over empty regions. That's the whole point of the change: restore the F=3-style geometric supervision that the F=32 path lost.

This builds on Codex's earlier v5_features fork (handoff at
`agent_notes/loose_notes/2026-04-29_17-30-00_codex_handoff_v5_features_rasterizer.md`).
The F-channel generalization works correctly. We're now adding one scalar output channel for alpha.

## Why we need this

Read `2026-04-29_19-30-00_feature_splatting_init_handoff.md` and the chat thread it summarizes for the full diagnosis. Short version:

In F=3 (the original RGB path), the renderer composites `pixel = sum_i T_i · α_i · color_i + T_final · white`. Pixels with no splat coverage render as white, the loss `|white - GT|` is large in non-white regions, and the gradient pulls splats outward to cover those regions via Gaussian-tail expansion.

In F=32 (the current feature splatting path), the renderer composites in feature space, and a downstream MLP maps features to RGB. For background pixels (`T_final = 1`), the MLP sees a constant feature input and learns to absorb the supervision via its bias — splats don't get the outward gradient signal. The MLP becomes a "background colorizer" and splats stay clustered at the GT centroid.

The fix is to bypass the MLP for background pixels and composite white in RGB space *after* the MLP:

```
splat_rgb = colorize(rendered_features)
final_rgb = alpha_mask · splat_rgb + (1 - alpha_mask) · white
loss      = L1(final_rgb, GT) + DSSIM(...)
```

Where `alpha_mask = 1 - T_final` is what we need from the rasterizer. With this composition, the colorize MLP's gradient on background pixels is structurally zero (multiplied by `alpha_mask = 0`), so it can no longer absorb supervision that should reach the splats.

## Currently active fork (for reference)

- Variant directory: `third_party/fast-mac-gsplat/variants/v5_features/`
- Python bridge: `third_party/fast-mac-gsplat/variants/v5_features/torch_gsplat_bridge_v5_features/rasterize.py`
- Metal kernels: `third_party/fast-mac-gsplat/variants/v5_features/csrc/metal/gsplat_v5_features_kernels.metal`
- Bindings: `third_party/fast-mac-gsplat/variants/v5_features/csrc/bindings.cpp`
- Custom-op namespace: `torch.ops.gsplat_metal_v5_features.*`
- Dynaworld wrapper: `src/train/renderers/fast_mac.py`
- Build (Python 3.11):
  ```bash
  ( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/v5_features
    uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
  ```

**Don't fork again.** Edit `v5_features` in place. We've already crossed the bit-for-bit-parity-with-v5 bar (max_abs=0 at F=3); we just need to extend the same fork.

## DO NOT derive new gradient math — flag-don't-fix policy

**This is the most important constraint in this doc.** The existing v5_features kernel math is verified (Codex's earlier parity test passed at F=3 vs v5 with `max_abs=0`, gradient correctness at F={3, 8, 32} with `max_abs ≤ 1e-9`). We are not changing any of that math. We are *piping one extra synthetic channel* through the existing math.

If during this work you notice anything that looks mathematically wrong in the *existing* (already-shipped) kernel code — backward formulas, alpha blending, conic projection, anything — **do not fix it**. Write a brief note in a new file:

```
agent_notes/loose_notes/<datetime>_v5_features_math_audit_findings.md
```

Describe what you saw, what you think is wrong, and stop. We will triage it separately. Mixing math fixes with this plumbing change makes regressions hard to bisect, and the F=3/F=8/F=32 parity numbers tell us the existing math is at least empirically correct.

The only "math" you should be writing in this PR is the *concrete value* `out_alpha = 1 - T_final` in the forward pass, which is already tracked as a local variable. That's it.

## What needs to change

### 1. Mathematical equivalence trick — this is THE implementation strategy

`accumulated_alpha = 1 - T_final` is *mathematically identical* to a feature channel where every splat has feature value `1.0` and the background is `0.0`. Specifically:

```
out_alpha = sum_i T_i · α_i_at_pixel · 1.0 + T_final · 0
          = sum_i T_i · α_i_at_pixel
          = 1 - T_final
```

This is NOT just a derivation aid. **It is your implementation plan.** The alpha output's forward and backward have **the same structure** as a feature channel. The existing kernel code already knows how to:

- composite a per-channel feature value through alpha-blending (forward)
- accumulate per-pixel feature gradients into per-splat opacity, means2d, conics gradients (backward)

You will reuse that code by treating alpha as a "synthetic feature channel with value 1.0 and background 0.0." You will **not** derive parallel gradient formulas for alpha — they are the same formulas the kernel already uses, with the per-splat color value hardcoded to 1.0 and the color gradient write skipped.

Two acceptable implementation approaches, in order of preference:

#### Approach A — kernel-internal synthetic channel (recommended)

Inside the kernel, allocate one extra channel internally on top of the user's `feature_dim`. The kernel internally treats the workload as `feature_dim + 1` channels where the (F+1)th channel is synthesized with `c_i = 1.0` for every splat and `bg = 0.0`. Forward writes that channel to a separate `out_alpha` buffer. Backward reads `grad_alpha` and routes it through the same per-splat-channel-gradient code that already exists, with two adjustments:

1. The synthetic channel's `c_i` is always 1.0 — no need to read from the colors buffer for that channel.
2. The synthetic channel's color gradient is *not written back* (the 1.0s are not learnable).

The geometry gradients (means2d, conics, opacities) accumulate over **all channels including the synthetic one**, which gives the alpha-stream contribution to the geometric supervision automatically.

Pros: cleanest API, reuses the most code, no derivation of new math.
Cons: a few extra ifs in the loop body.

#### Approach B — Python-wrapper-level synthetic channel (fallback)

If Approach A is hard to get right in Metal, fall back to a pure Python wrapper change that does the equivalence trick at the Python layer:

```python
# In rasterize.py's rasterize_projected_gaussians:
# Append a 1.0-marker channel to colors and a 0.0 channel to feature_background.
ones_marker = torch.ones(*colors.shape[:-1], 1, device=colors.device, dtype=colors.dtype)
ones_marker.requires_grad_(False)  # not a trainable input
colors_extended = torch.cat([colors, ones_marker], dim=-1)
bg_extended = list(config.background) + [0.0]
config_extended = replace(config, background=tuple(bg_extended))

# Run the EXISTING kernel unmodified, with feature_dim = F + 1.
out_extended = existing_rasterize(means2d, conics, colors_extended, opacities, depths, config_extended)

# Slice off the alpha channel.
features = out_extended[..., :-1]
alpha = out_extended[..., -1]
return features, alpha
```

Pros: zero Metal kernel changes, zero math risk, ~30 lines of Python.
Cons: ~3% extra compute and memory per raster call (one extra channel out of F+1). Negligible at F=32.

**Approach B is acceptable as the final implementation.** If Approach A fights you for more than a couple of hours, just ship Approach B. The user cares about correctness and "splats spread, no clustering" much more than they care about a 3% raster speedup.

The audit + tests below apply to both approaches.

### 2. Forward kernel — write alpha output

The forward kernel (`gsplat_v5_features_kernels.metal:590` and adjacent compositing loops) already tracks `T_final` as a local variable per pixel. Add:

- A new device output buffer `out_alpha[batch, H, W]` (or `[H, W]` for the single-image path) — float32.
- After the compositing loop completes for a pixel, write `out_alpha[pix] = 1.0f - T_final` (i.e., `accum_alpha`).

For the early-out path (where compositing breaks because `T_final < threshold`), `1 - T_final` should reflect the actual accumulated alpha at the break point. The existing kernel structure already handles this naturally — just ensure the value written is the post-loop `T_final`.

For pixels with no splat coverage at all (the early-empty-tile branch), write `out_alpha[pix] = 0.0f` (matches `1 - T_final = 1 - 1 = 0`).

### 3. Backward kernel — accept grad_alpha, propagate to geometry

The new backward signature accepts both `grad_features [B, H, W, F]` and `grad_alpha [B, H, W]` from the upstream. It must produce gradients on:

- `means2d` — through the per-pixel α_i computation (which depends on the splat's 2D Gaussian profile)
- `conics` — same chain (conic determines the spatial Gaussian shape)
- `colors` — only from `grad_features` (alpha doesn't depend on colors)
- `opacities` — from BOTH `grad_features` and `grad_alpha`

**This is the critical correctness requirement.** The whole point of the alpha output is to drive geometric supervision back to opacity, position, and scale. If `grad_alpha` only flows to opacity but not means2d/conics, the splats can change opacity but not move or stretch — and we'll see the same clustering symptom we're trying to fix.

Concretely, for each splat `i`:

```
∂L/∂α_i               = ∂L/∂α_i(from features) + ∂L/∂α_i(from alpha)
∂L/∂means2d_i         = ∂L/∂means2d_i(from features) + ∂L/∂means2d_i(from alpha)
∂L/∂conics_i          = ∂L/∂conics_i(from features) + ∂L/∂conics_i(from alpha)
∂L/∂color_i[f]        = ∂L/∂color_i[f](from features only)    ← unchanged
```

The "from alpha" contributions have the same structure as a feature-channel contribution with `c_i = 1` (and no color gradient written back). If using the synthesizer trick from §1, this falls out automatically.

### 4. Python bridge — return alpha as second output

`torch_gsplat_bridge_v5_features.rasterize.rasterize_projected_gaussians` currently returns `out_features`. New signature:

```python
def rasterize_projected_gaussians(
    means2d: Tensor,
    conics: Tensor,
    colors: Tensor,
    opacities: Tensor,
    depths: Tensor,
    config: RasterConfig,
) -> tuple[Tensor, Tensor]:                 # (out_features, accumulated_alpha)
    ...
```

Return shapes:
- Single-image path: `(out_features [H, W, F], alpha [H, W])`
- Batched path:      `(out_features [B, H, W, F], alpha [B, H, W])`

The `_RasterizeProjectedGaussiansV5Features` autograd Function's `forward` returns the tuple; `backward(ctx, grad_features, grad_alpha)` accepts both. PyTorch handles tuple outputs natively.

### 5. RasterConfig — no changes required

`feature_background` continues to be the configurable background for the feature channels. The alpha output's "background" is hardcoded to 0 (since `1 - T_final` for empty pixels is 0 by the math). Don't expose it as a config knob — it's a structural property.

### 6. Test for gradient correctness — non-negotiable

This is the test that prevents the "splats can change opacity but not move or stretch" silent failure mode. Before declaring done, in `v5_features/tests/`:

#### Test A: forward alpha shape and values

For a tiny fixture (`H=8, W=8, G=4, F=3`) with hand-placed splats:
- One pixel covered by no splats → `alpha = 0.0` exactly
- One pixel covered by one splat with α=0.5 → `alpha ≈ 0.5`
- One pixel covered by two splats both α=0.5 → `alpha ≈ 0.75` (1 - 0.5·0.5)

#### Test B: backward gradient on geometry from alpha

The most important test. Set up a tiny fixture, define a loss that depends *only* on `alpha`, and verify gradients flow correctly to all geometry parameters.

Use a CPU pure-PyTorch reference for ground truth. Pseudocode:

```python
def reference_alpha(means2d, conics, opacities, H, W):
    # Pure PyTorch front-to-back alpha-blend, scalar-only (no colors).
    # Returns alpha [H, W] = 1 - T_final per pixel.
    ...

# Build random splats with requires_grad=True on means2d, conics, opacities, colors
features_kernel, alpha_kernel = rasterize_projected_gaussians(
    means2d, conics, colors, opacities, depths, config
)
loss_alpha = alpha_kernel.sum()  # or any scalar function of alpha only
loss_alpha.backward()

# Re-run the reference, compute its gradients
alpha_ref = reference_alpha(means2d_ref, conics_ref, opacities_ref, H, W)
loss_ref = alpha_ref.sum()
loss_ref.backward()

# Compare per-tensor max abs diff
assert (means2d.grad - means2d_ref.grad).abs().max() < 1e-4
assert (conics.grad - conics_ref.grad).abs().max() < 1e-4
assert (opacities.grad - opacities_ref.grad).abs().max() < 1e-4
assert colors.grad.abs().max() < 1e-8     # ALPHA HAS NO COLOR GRADIENT
```

The first three asserts catch the "alpha-only changes opacity, not means2d/conics" silent bug. The fourth asserts that alpha gradient does NOT contaminate the color gradient stream.

#### Test C: forward parity vs synthetic feature channel

A sanity-check that confirms the equivalence trick: if you append a synthetic feature channel `colors_with_marker = cat([colors, ones], dim=-1)` and set `bg_with_marker = cat([bg, [0]], dim=-1)`, then the F+1 raster's last channel should equal the new `alpha` output to `max_abs ≤ 1e-6`. This validates that the backward is consistent with the forward path it logically duplicates.

#### Test D: combined backward (features + alpha)

When loss depends on both features and alpha, the gradients should compose linearly (no double-counting, no missing contributions). Construct a loss `L = features.sum() + alpha.sum()`, run the kernel backward, and compare against running each separately and summing the gradients.

#### Test E: F=3 v5 parity check

The previous parity check verified `v5_features (F=3 features) == v5 (F=3 colors)`. With the alpha output added, also verify:
- `v5_features (F=3, alpha)` forward at `colors=ones, feature_background=0` reproduces `1 - T_final` from v5's internal state at `bg=(1,1,1)`. (Alpha is mathematically equivalent regardless of which kernel computes it.)

### 7. Throughput note

Adding alpha output should be a small overhead (one more scalar per pixel in forward, one more scalar accumulator in backward). Mention the new throughput numbers in your handback alongside the existing F=3/F=8/F=32 measurements.

## Math audit — why the synthetic-channel trick produces correct geometry gradients

Read this section once for confidence. You don't need to *do* anything in this section — the existing kernel does the math.

The existing forward kernel composites:
```
out_features[pix, f] = sum_i T_i · α_i_at_pixel · color_i[f] + T_final · feature_background[f]
```
The existing backward kernel computes, for each splat i:
```
∂L/∂α_i           = sum_f (color_i[f] - downstream_blend[f]) · T_pre_i · ∂L/∂out_features[pix, f]
∂L/∂color_i[f]    = T_pre_i · α_i_at_pixel · ∂L/∂out_features[pix, f]
∂L/∂means2d_i     = (∂α_i_at_pixel/∂means2d_i) · sum_f (color_i[f] - downstream_blend[f]) · ∂L/∂out_features[pix, f]
∂L/∂conics_i      = (∂α_i_at_pixel/∂conics_i ) · sum_f (color_i[f] - downstream_blend[f]) · ∂L/∂out_features[pix, f]
```

(That `sum_f` is the key — the kernel already sums over all feature channels.)

If we add the (F+1)th synthetic channel with `color_i[F+1] = 1.0` and `feature_background[F+1] = 0`:

```
out_features[pix, F+1] = sum_i T_i · α_i_at_pixel · 1.0 + T_final · 0
                      = 1 - T_final
                      = the alpha output we want.
```

The backward then automatically gives:
```
∂L/∂α_i           += (1.0 - downstream_blend_alpha) · T_pre_i · ∂L/∂out_features[pix, F+1]
∂L/∂means2d_i     += (∂α/∂means2d_i)  · (1.0 - downstream_blend_alpha) · ∂L/∂out_features[pix, F+1]
∂L/∂conics_i      += (∂α/∂conics_i )  · (1.0 - downstream_blend_alpha) · ∂L/∂out_features[pix, F+1]
∂L/∂color_i[F+1]   = T_pre_i · α_i_at_pixel · ∂L/∂out_features[pix, F+1]   ← we DISCARD this
```

The means2d, conics, and opacities gradients **automatically pick up the alpha contribution** because the existing kernel sums over all channels. This is exactly what we need: a strong gradient signal pushing splats to grow opacity, expand scale, and migrate position to fill missed pixels.

**The only thing to actively skip is the color-grad write for the synthetic channel** (the 1.0s are not learnable). For Approach A, this means: in backward, don't `atomic_add` to `g_colors[i*F + F]` for the synthetic channel. For Approach B, this is automatic because we marked the synthetic ones as `requires_grad=False`.

If anything in the audit above doesn't match what the existing kernel does, that's a candidate "pre-existing math issue" — flag in the audit findings file and stop.

## Step-by-step implementation guide

This is the explicit list of every file and function to change. **All changes are confined to `third_party/fast-mac-gsplat/variants/v5_features/`** — do not modify v5 or anything in `src/train/` (the dynaworld trainer plumbing happens after you hand back).

### Files to edit

| File | What changes |
|---|---|
| `csrc/metal/gsplat_v5_features_kernels.metal` | 4 of 7 Metal kernels need alpha output added (forward eval / forward state / overflow forward / forward backward / overflow backward) |
| `csrc/bindings.cpp` | Update 4 dispatch function signatures + 4 schema strings in `TORCH_LIBRARY` |
| `csrc/metal/gsplat_metal.mm` | Update the `metal_render_fast_forward_eval`, `metal_render_fast_forward_state`, `metal_render_overflow_forward` (and corresponding backward) implementations to allocate + bind the new alpha output buffer |
| `torch_gsplat_bridge_v5_features/rasterize.py` | Update `_RasterizeProjectedGaussiansV5Features.forward/backward` to handle the tuple return + grad_alpha input |

### Kernels to update (Metal)

These are the 7 kernels in `gsplat_v5_features_kernels.metal`. Lines are from current HEAD; **verify when you start**.

| Kernel | Line | Forward/Backward | Needs alpha? |
|---|---:|---|---|
| `v5_features_count_tiles` | 324 | (binning) | **No** — pre-raster step |
| `v5_features_emit_binned_ids` | 370 | (binning) | **No** — pre-raster step |
| `v5_features_tile_fast_forward_eval` | 410 | Forward (inference, no grad) | **Yes** — write `1 - T_final` to alpha output |
| `v5_features_tile_fast_forward_state` | 476 | Forward (training, with grad) | **Yes** — write `1 - T_final` to alpha output |
| `v5_features_tile_fast_backward_saved` | 550 | Backward (main path) | **Yes** — read `grad_alpha` and accumulate into geometry grads |
| `v5_features_tile_overflow_forward` | 679 | Forward (overflow tiles, slow path) | **Yes** — same as fast forward |
| `v5_features_tile_overflow_backward` | 738 | Backward (overflow tiles) | **Yes** — same as fast backward |

### Forward kernels — what to add (pseudocode)

Each forward kernel currently has a per-pixel compositing loop that already tracks `T` (transmittance). At the end of the loop, add **one new output buffer write**:

```c
// Inside the compositing loop body — UNCHANGED:
float T = 1.0f;
for (each splat in this tile, in depth order) {
    if (!valid || T <= mf.transmittance_threshold) break;
    float alpha = α_i_at_pixel(splat, pixel);
    for (uint f = 0; f < feature_dim; f++) {
        accum[f] += T * alpha * feature_i[f];
    }
    T *= (1.0f - alpha);
}

// EXISTING: write features with background tail:
add_background_tail(out_features, pix, T, mi, mf);

// NEW: write alpha output (1 - T_final). This MUST be added in:
//   - v5_features_tile_fast_forward_eval
//   - v5_features_tile_fast_forward_state
//   - v5_features_tile_overflow_forward
out_alpha[pix] = 1.0f - T;

// Edge case: empty-tile early-out branch (where the kernel returns before
// the compositing loop because no splats reach this tile). In that branch,
// also write:
out_alpha[pix] = 0.0f;     // T_final = 1, so 1 - T_final = 0
```

The empty-tile branch is the `if (alive_total == 0u)` (or similar) check near the top of each forward kernel where it writes the background-only output and returns early. Same pattern: write `0.0f` to `out_alpha[pix]` there.

### Backward kernels — what to add (NO new math)

Each backward kernel currently accepts `grad_out` (or `grad_tiles` for overflow), processes splats in reverse depth order, and accumulates gradients on means2d, conics, colors, opacities.

**Use the synthetic-channel trick described in the audit section above.** Concretely (Approach A, kernel-internal):

1. Treat the workload as `feature_dim + 1` channels internally where the (F+1)th channel has `c_i = 1.0` for every splat.
2. Read `grad_alpha[pix]` as the gradient signal for that synthetic channel.
3. Let the existing per-channel backward code (the loop over `f` from 0 to `F`) run for the synthetic channel just like for any other. The means2d / conics / opacity gradients sum across channels as they already do — you don't write any new gradient formulas.
4. **Skip the color-grad write for the synthetic channel.** The 1.0s are not learnable. Don't allocate space for it; or write to a discard buffer; or branch on `f == F` to skip the atomic add.

**Do not derive new gradient formulas for means2d, conics, or opacities.** The existing math handles them. Your job is to (a) supply the correct upstream `grad_alpha` value for the synthetic channel and (b) suppress the color-grad write for that channel.

If you find that the existing kernel structure fights you on either of those — e.g., the channel count is hardcoded somewhere awkward, or the color-grad write is hard to skip per-channel — that's a signal Approach B (Python-wrapper-level) is the right answer for this PR. Switch to Approach B and ship.

The pseudocode below is **for understanding only** — it shows what the existing kernel does (renamed to make the alpha contribution explicit), not new code you need to write:

```c
// NEW input parameters to the kernel signature:
//   const device float* grad_alpha [[buffer(N)]],  // [B, H, W] gradient on alpha output
// (grad_alpha is treated like a scalar feature channel)

// Inside backward, when iterating splats in REVERSE depth order:
for (each splat from far to near) {
    // EXISTING: compute alpha_i_at_pixel and the conic/Gaussian contribution

    // EXISTING: grad on color from features
    for (uint f = 0; f < feature_dim; f++) {
        atomic_add(g_colors[i*F + f], T_pre_i · alpha_i_at_pixel · grad_features[pix*F + f]);
    }

    // EXISTING: grad on opacity from features
    float feature_contrib_to_opacity = sum_f (
        feature_i[f] * (T_pre_i · grad_features[pix*F + f])
        - downstream_blend_features[f] · grad_features[pix*F + f]
    );

    // NEW: grad on opacity from alpha (synthetic feature value c=1, no color grad)
    float alpha_contrib_to_opacity = (
        1.0 * (T_pre_i · grad_alpha[pix])
        - downstream_blend_alpha · grad_alpha[pix]
    );

    // Accumulate combined opacity grad:
    atomic_add(g_opacities[i], feature_contrib_to_opacity + alpha_contrib_to_opacity);

    // EXISTING: grad on means2d, conics — these depend on alpha_i_at_pixel,
    // which is shared between feature and alpha gradient streams.
    // The combined grad upstream signal is:
    //   combined_dL_dalpha_i = sum_f (feature_i[f] · grad_features[pix*F + f])
    //                        + 1.0 · grad_alpha[pix]
    //   plus the downstream-blend correction terms (existing math)
    // Use this combined signal in the existing means2d / conics gradient
    // formulas. DO NOT compute means2d/conics grads twice — use one combined
    // upstream grad and run the existing chain once.

    // IMPORTANT: do NOT write any color gradient from alpha. The synthetic
    // c=1 has no learnable parameters; alpha-stream contributes ONLY to
    // means2d, conics, and opacities.
}
```

The "downstream blend" tracking already exists in the kernel for the feature stream; you'll need a parallel scalar accumulator for the alpha stream's downstream blend. Or, equivalently, treat alpha as the (F+1)th channel of a synthesized features array internally (the equivalence trick), run the existing per-channel backward, and skip the color-grad write for that channel.

**The key invariant**: the splat's per-pixel contribution to alpha is exactly the same `alpha_i_at_pixel` value that determines its contribution to features. The means2d and conics gradients depend on `alpha_i_at_pixel` in the same way for both streams. So the gradient on means2d/conics from alpha follows the **same chain** as from features, just with the upstream signal being `grad_alpha[pix]` instead of `sum_f(feature_i[f] * grad_features[pix, f])`.

### bindings.cpp — schema changes

Update the 4 schema strings in `TORCH_LIBRARY(gsplat_metal_v5_features, m)`:

```cpp
// BEFORE (current):
m.def("render_fast_forward_eval(Tensor means2d, Tensor conics, Tensor colors, Tensor opacities, Tensor meta_i32, Tensor meta_f32, Tensor tile_counts, Tensor tile_offsets, Tensor binned_ids) -> Tensor");
m.def("render_fast_forward_state(Tensor means2d, Tensor conics, Tensor colors, Tensor opacities, Tensor meta_i32, Tensor meta_f32, Tensor(a!) binned_ids, Tensor tile_counts, Tensor tile_offsets) -> (Tensor, Tensor)");
m.def("render_fast_backward_saved(Tensor grad_out, Tensor means2d, Tensor conics, Tensor colors, Tensor opacities, Tensor meta_i32, Tensor meta_f32, Tensor tile_counts, Tensor tile_offsets, Tensor binned_ids, Tensor tile_stop_counts) -> (Tensor, Tensor, Tensor, Tensor)");
m.def("render_overflow_forward(Tensor means2d, Tensor conics, Tensor colors, Tensor opacities, Tensor meta_i32, Tensor meta_f32, Tensor overflow_tile_ids, Tensor overflow_tile_offsets, Tensor overflow_sorted_ids) -> Tensor");
m.def("render_overflow_backward(Tensor grad_tiles, Tensor means2d, Tensor conics, Tensor colors, Tensor opacities, Tensor meta_i32, Tensor meta_f32, Tensor overflow_tile_ids, Tensor overflow_tile_offsets, Tensor overflow_sorted_ids) -> (Tensor, Tensor, Tensor, Tensor)");

// AFTER (new):
m.def("render_fast_forward_eval(Tensor means2d, Tensor conics, Tensor colors, Tensor opacities, Tensor meta_i32, Tensor meta_f32, Tensor tile_counts, Tensor tile_offsets, Tensor binned_ids) -> (Tensor, Tensor)");                                              // returns (out_features, out_alpha)
m.def("render_fast_forward_state(Tensor means2d, Tensor conics, Tensor colors, Tensor opacities, Tensor meta_i32, Tensor meta_f32, Tensor(a!) binned_ids, Tensor tile_counts, Tensor tile_offsets) -> (Tensor, Tensor, Tensor)");                                  // returns (out_features, out_alpha, tile_stop_counts)
m.def("render_fast_backward_saved(Tensor grad_features, Tensor grad_alpha, Tensor means2d, Tensor conics, Tensor colors, Tensor opacities, Tensor meta_i32, Tensor meta_f32, Tensor tile_counts, Tensor tile_offsets, Tensor binned_ids, Tensor tile_stop_counts) -> (Tensor, Tensor, Tensor, Tensor)");
m.def("render_overflow_forward(Tensor means2d, Tensor conics, Tensor colors, Tensor opacities, Tensor meta_i32, Tensor meta_f32, Tensor overflow_tile_ids, Tensor overflow_tile_offsets, Tensor overflow_sorted_ids) -> (Tensor, Tensor)");                          // returns (out_features, out_alpha)
m.def("render_overflow_backward(Tensor grad_features_tiles, Tensor grad_alpha_tiles, Tensor means2d, Tensor conics, Tensor colors, Tensor opacities, Tensor meta_i32, Tensor meta_f32, Tensor overflow_tile_ids, Tensor overflow_tile_offsets, Tensor overflow_sorted_ids) -> (Tensor, Tensor, Tensor, Tensor)");
```

The dispatch function C++ signatures also change to match (return type goes from `torch::Tensor` to `std::tuple<torch::Tensor, torch::Tensor>` for forwards; backwards now take `grad_alpha` as a new first/second positional arg).

### Output dtype / contiguity

- `out_alpha`: `torch::kFloat32`, contiguous, allocated with the same layout as the existing feature output (per-image batched: `[B, H, W]` row-major).
- `grad_alpha`: same shape and dtype.
- `grad_alpha` for the overflow path is per-tile: `[overflow_tile_count, tile_size, tile_size]` — same as how `grad_tiles` is gathered for the existing overflow backward.

### Python bridge — what to update in `rasterize.py`

The existing `_RasterizeProjectedGaussiansV5Features` autograd Function (line ~362):

```python
# BEFORE — forward returns Tensor:
out = torch.ops.gsplat_metal_v5_features.render_fast_forward_state(...)
# `out` is the features tensor

# AFTER — forward returns (features, alpha):
out_features, out_alpha = torch.ops.gsplat_metal_v5_features.render_fast_forward_state(...)
# Save out_alpha to ctx if needed for backward (it's not strictly needed
# for the gradient computation since we have access to the saved tensors,
# but the autograd Function must declare both as outputs).
return out_features, out_alpha

# BEFORE — backward signature:
def backward(ctx, grad_out):
    ...
    return g_means_b, g_conics_b, g_colors_b, g_opacities_b, g_depths_b, None, None, None, None

# AFTER — backward signature accepts both:
def backward(ctx, grad_features, grad_alpha):
    ...
    # Pass BOTH grads into the new render_fast_backward_saved op:
    g_means_flat, g_conics_flat, g_colors_flat, g_opacities_flat = (
        torch.ops.gsplat_metal_v5_features.render_fast_backward_saved(
            grad_features, grad_alpha, means_flat, conics_flat, colors_flat,
            opacities_flat, meta_i32, meta_f32, tile_counts, tile_offsets,
            binned_ids, tile_stop_counts,
        )
    )
    # Same for overflow backward.
    # Return same shape as before.
    return g_means_b, g_conics_b, g_colors_b, g_opacities_b, g_depths_b, None, None, None, None
```

The top-level `rasterize_projected_gaussians` function (the public API) currently returns a single tensor. Change it to return `(features, alpha)`. The chunked-eval path (`_rasterize_chunk_eval`, line ~540) also needs updating to return the alpha tensor concatenated across chunks.

### CPU PyTorch reference for Test B (full code, not pseudocode)

Save this in `tests/reference_alpha.py`:

```python
import torch


def reference_features_and_alpha(
    means2d: torch.Tensor,        # [G, 2] in pixels
    conics: torch.Tensor,         # [G, 3] = (a, b, c) for inverse covariance [[a, b], [b, c]]
    colors: torch.Tensor,         # [G, F]
    opacities: torch.Tensor,      # [G]
    depths: torch.Tensor,         # [G] (used for sort order)
    height: int,
    width: int,
    feature_background: torch.Tensor,  # [F] — broadcast value, all zeros for our tests
    alpha_threshold: float = 1.0 / 255.0,
    transmittance_threshold: float = 1.0e-4,
):
    """Pure-PyTorch front-to-back alpha-blend. Reference for v5_features test.

    Returns (features [H, W, F], alpha [H, W]).
    Both are differentiable wrt means2d, conics, colors, opacities (not depths).
    """
    G = means2d.shape[0]
    F = colors.shape[-1]
    device = means2d.device
    dtype = means2d.dtype

    # Sort splats by depth (near to far)
    order = torch.argsort(depths)
    means2d_s   = means2d[order]
    conics_s    = conics[order]
    colors_s    = colors[order]
    opacities_s = opacities[order]

    # Build pixel grid
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    pixels = torch.stack([xx, yy], dim=-1)         # [H, W, 2]

    out_features = torch.zeros(height, width, F, device=device, dtype=dtype)
    out_alpha    = torch.zeros(height, width,    device=device, dtype=dtype)
    T            = torch.ones (height, width,    device=device, dtype=dtype)

    for i in range(G):
        if (T < transmittance_threshold).all():
            break
        d  = pixels - means2d_s[i]                  # [H, W, 2]
        a, b, c = conics_s[i, 0], conics_s[i, 1], conics_s[i, 2]
        power = -0.5 * (a * d[..., 0] ** 2 + 2.0 * b * d[..., 0] * d[..., 1] + c * d[..., 1] ** 2)
        gauss = torch.exp(power.clamp(max=0.0))     # [H, W]
        alpha_pixel = (opacities_s[i] * gauss).clamp(0.0, 0.999)
        alpha_mask  = alpha_pixel >= alpha_threshold
        alpha_pixel = alpha_pixel * alpha_mask.to(dtype)

        contrib = T * alpha_pixel
        out_features = out_features + contrib.unsqueeze(-1) * colors_s[i]
        out_alpha    = out_alpha    + contrib
        T = T * (1.0 - alpha_pixel)

    # Apply feature background tail (alpha background is 0, so no tail addition)
    out_features = out_features + T.unsqueeze(-1) * feature_background

    return out_features, out_alpha
```

This reference is intentionally simple (no tiling, no batching, no overflow path) — it's the *mathematical specification* for the kernel, not a fast implementation. Use it as ground truth for Test B.

### Test B — full code (not pseudocode)

```python
import torch
from torch_gsplat_bridge_v5_features import RasterConfig, rasterize_projected_gaussians
from tests.reference_alpha import reference_features_and_alpha


def test_alpha_only_loss_propagates_to_geometry():
    torch.manual_seed(0)
    G = 4
    F = 3
    H, W = 8, 8
    device = torch.device("mps")  # or "cpu" if MPS unavailable

    # Two random sets of identical inputs (one for kernel, one for reference).
    def make_inputs():
        means2d   = (torch.rand(G, 2, device=device) * H).requires_grad_(True)
        conics    = torch.zeros(G, 3, device=device)
        conics[:, 0] = 1.0
        conics[:, 2] = 1.0
        conics    = conics.requires_grad_(True)
        colors    = torch.randn(G, F, device=device).requires_grad_(True)
        opacities = torch.sigmoid(torch.randn(G, device=device)).requires_grad_(True)
        depths    = torch.linspace(0.0, 1.0, G, device=device)
        return means2d, conics, colors, opacities, depths

    # Kernel path
    m1, c1, col1, o1, d1 = make_inputs()
    config = RasterConfig(
        height=H, width=W, tile_size=8, max_fast_pairs=2048,
        alpha_threshold=1.0 / 255.0, transmittance_threshold=1.0e-4,
        background=(0.0,) * F, enable_overflow_fallback=True,
        inputs_sorted_by_depth=True, batch_strategy="serial",
        batch_launch_limit_tiles=262144, batch_launch_limit_gaussians=262144,
    )
    features_k, alpha_k = rasterize_projected_gaussians(m1, c1, col1, o1, d1, config)
    loss_k = alpha_k.sum()
    loss_k.backward()

    # Reference path (re-create inputs with the same seed pattern by reusing tensors)
    torch.manual_seed(0)  # IMPORTANT: re-seed before make_inputs() to get the same values
    m2, c2, col2, o2, d2 = make_inputs()
    feat_bg = torch.zeros(F, device=device)
    features_r, alpha_r = reference_features_and_alpha(m2, c2, col2, o2, d2, H, W, feat_bg)
    loss_r = alpha_r.sum()
    loss_r.backward()

    # Assertions
    assert (m1.grad   - m2.grad  ).abs().max().item() < 1.0e-4, "means2d alpha-only grad mismatch"
    assert (c1.grad   - c2.grad  ).abs().max().item() < 1.0e-4, "conics alpha-only grad mismatch"
    assert (o1.grad   - o2.grad  ).abs().max().item() < 1.0e-4, "opacities alpha-only grad mismatch"
    # Crucial: alpha doesn't depend on colors. col1.grad must be near zero.
    assert col1.grad.abs().max().item() < 1.0e-6, "alpha-only loss should NOT produce color grad"
    print("Test B passed.")


if __name__ == "__main__":
    test_alpha_only_loss_propagates_to_geometry()
```

If any assert fires, the kernel has a real bug — do not call this done. The most likely failure: the kernel propagates alpha grad to opacity but skips means2d/conics. Both must work.

### Test C — equivalence trick parity (full code)

```python
def test_alpha_matches_synthetic_feature_channel():
    """Equivalence: append a feature channel of all 1s, set its bg to 0,
    and the F+1th rendered channel should equal the new alpha output."""
    torch.manual_seed(0)
    G, F, H, W = 4, 3, 8, 8
    device = torch.device("mps")

    means2d   = (torch.rand(G, 2, device=device) * H)
    conics    = torch.zeros(G, 3, device=device); conics[:, 0] = 1.0; conics[:, 2] = 1.0
    colors    = torch.randn(G, F, device=device)
    opacities = torch.sigmoid(torch.randn(G, device=device))
    depths    = torch.linspace(0.0, 1.0, G, device=device)

    # Path 1: new alpha output
    config_F = RasterConfig(
        height=H, width=W, tile_size=8, max_fast_pairs=2048,
        alpha_threshold=1.0 / 255.0, transmittance_threshold=1.0e-4,
        background=(0.0,) * F, enable_overflow_fallback=True,
        inputs_sorted_by_depth=True, batch_strategy="serial",
        batch_launch_limit_tiles=262144, batch_launch_limit_gaussians=262144,
    )
    features, alpha = rasterize_projected_gaussians(
        means2d, conics, colors, opacities, depths, config_F,
    )

    # Path 2: synthetic (F+1)-channel raster with a marker channel
    colors_marker = torch.cat([colors, torch.ones(G, 1, device=device)], dim=-1)
    config_F1 = RasterConfig(
        height=H, width=W, tile_size=8, max_fast_pairs=2048,
        alpha_threshold=1.0 / 255.0, transmittance_threshold=1.0e-4,
        background=(0.0,) * (F + 1), enable_overflow_fallback=True,
        inputs_sorted_by_depth=True, batch_strategy="serial",
        batch_launch_limit_tiles=262144, batch_launch_limit_gaussians=262144,
    )
    features_marker, _alpha_unused = rasterize_projected_gaussians(
        means2d, conics, colors_marker, opacities, depths, config_F1,
    )

    # The last channel of the F+1 raster should equal the new alpha output
    diff = (features_marker[..., -1] - alpha).abs().max().item()
    assert diff < 1.0e-6, f"alpha-vs-marker-channel parity max abs diff = {diff:g}"
    print("Test C passed.")
```

### Test D — combined (features + alpha) backward correctness

```python
def test_combined_backward_linear():
    """L = features.sum() + alpha.sum(). Combined backward = sum of separate backwards."""
    torch.manual_seed(0)
    G, F, H, W = 4, 3, 8, 8
    device = torch.device("mps")

    def make_inputs():
        m = (torch.rand(G, 2, device=device) * H).requires_grad_(True)
        c = torch.zeros(G, 3, device=device)
        c[:, 0] = 1.0; c[:, 2] = 1.0
        c = c.requires_grad_(True)
        col = torch.randn(G, F, device=device).requires_grad_(True)
        o = torch.sigmoid(torch.randn(G, device=device)).requires_grad_(True)
        d = torch.linspace(0.0, 1.0, G, device=device)
        return m, c, col, o, d

    config = RasterConfig(
        height=H, width=W, tile_size=8, max_fast_pairs=2048,
        alpha_threshold=1.0 / 255.0, transmittance_threshold=1.0e-4,
        background=(0.0,) * F, enable_overflow_fallback=True,
        inputs_sorted_by_depth=True, batch_strategy="serial",
        batch_launch_limit_tiles=262144, batch_launch_limit_gaussians=262144,
    )

    # Combined: L = features.sum() + alpha.sum()
    m, c, col, o, d = make_inputs()
    feat, a = rasterize_projected_gaussians(m, c, col, o, d, config)
    (feat.sum() + a.sum()).backward()
    g_combined = (m.grad.clone(), c.grad.clone(), col.grad.clone(), o.grad.clone())

    # Separate: feat-only and alpha-only, sum their grads
    torch.manual_seed(0)
    m_f, c_f, col_f, o_f, d_f = make_inputs()
    feat_f, _ = rasterize_projected_gaussians(m_f, c_f, col_f, o_f, d_f, config)
    feat_f.sum().backward()
    g_feat = (m_f.grad, c_f.grad, col_f.grad, o_f.grad)

    torch.manual_seed(0)
    m_a, c_a, col_a, o_a, d_a = make_inputs()
    _feat_unused, a_a = rasterize_projected_gaussians(m_a, c_a, col_a, o_a, d_a, config)
    a_a.sum().backward()
    g_alpha = (m_a.grad, c_a.grad, col_a.grad, o_a.grad)

    for combined, feat, alpha, name in zip(g_combined, g_feat, g_alpha, ["m", "c", "col", "o"]):
        diff = (combined - (feat + alpha)).abs().max().item()
        assert diff < 1.0e-5, f"{name}: combined != separate sum, diff={diff:g}"
    print("Test D passed.")
```

### Edge cases to handle explicitly

1. **Empty tile** (no splats reach the tile): `out_alpha[pix] = 0.0f` for all pixels in that tile.
2. **Saturated tile** (compositing breaks early because T < threshold): `out_alpha[pix]` should reflect the actual `1 - T_final` at the break point, *not* clamped to 1. For example, if T drops to 1e-5 after 50 splats, `out_alpha = 1 - 1e-5 = 0.99999`. That's correct.
3. **Numerical stability**: `1 - T_final` should always be in `[0, 1]` because `T_final ∈ [0, 1]`. No clamping needed if the rest of the math is right; if you find it slipping out of range due to fp32 noise, clamp to `[0, 1]` after writing.
4. **inputs_sorted_by_depth=True**: alpha output ordering convention should match features. The Python bridge re-permutes feature gradients in backward when `inputs_sorted_by_depth=False`; alpha doesn't have a per-splat gradient (it's a per-pixel scalar), so no re-permutation is needed for alpha itself, but the *gradient propagation through opacities* still benefits from the same depth-ordering. Just follow the existing pattern.
5. **Overflow path**: alpha output is gathered+scattered the same way as features in the overflow path. The `_zero_tile_images_` call in backward (line ~488) zeros tiles for the fast path that overflowed; do the same for `grad_alpha` (zero out overflow tiles before passing to `render_fast_backward_saved`).

## API surface (target)

```python
from torch_gsplat_bridge_v5_features import RasterConfig, rasterize_projected_gaussians

# Forward
features, alpha = rasterize_projected_gaussians(
    means2d,    # [B, G, 2] or [G, 2]
    conics,     # [B, G, 3] or [G, 3]
    colors,     # [B, G, F] or [G, F]
    opacities,  # [B, G] or [G]
    depths,     # [B, G] or [G]
    config,
)
# features: [B, H, W, F] or [H, W, F]
# alpha:    [B, H, W]    or [H, W]    -- in [0, 1], 0 = no splat coverage, 1 = fully opaque

# Backward
loss = some_function(features, alpha)
loss.backward()
# means2d.grad   gets contributions from BOTH features AND alpha (geometric supervision)
# conics.grad    gets contributions from BOTH features AND alpha (geometric supervision)
# opacities.grad gets contributions from BOTH features AND alpha
# colors.grad    gets contributions from features ONLY (alpha doesn't depend on colors)
```

## Out-of-scope

- Don't touch v5. Leave it for parity comparisons.
- Don't change projection math.
- Don't add a learnable alpha background. The alpha background is structurally 0; expose only the feature background as before.
- Don't change v5_features's existing feature-only API surface in a breaking way for any callers other than the dynaworld wrapper. (You can do whatever's cleanest internally; just make sure `rasterize_projected_gaussians` still callable. The dynaworld wrapper will be updated to consume the new tuple return.)

## Hand back when

- All five tests pass on Apple Silicon MPS:
  - Test A: forward alpha shape + values
  - Test B: backward gradient on means2d, conics, opacities from alpha only (with zero color grad contamination)
  - Test C: equivalence trick parity (alpha == ones-feature-channel)
  - Test D: combined features + alpha backward correctness
  - Test E: F=3 v5 parity (alpha output reproduces 1 - T_final from v5)
- A short note appended to this doc with: forward+backward throughput at F ∈ {3, 8, 32}, the test results, and a one-line description of which implementation path you chose (synthesizer trick vs inline scalar accumulator).

Once this lands, the dynaworld trainer plumbing is out-of-scope for this PR. Don't touch `src/train/`. The integration (renderer wrapper return signature, alpha-aware composition in the loss path, W&B video logging) happens in a separate dynaworld-side change after you hand back.

## Handback: 2026-04-29 20:08 +07

Implementation path: Approach A, kernel-internal synthetic alpha channel. The
Metal backward feeds `grad_alpha` into the same combined channel contribution
used by real features and still writes color gradients only for real feature
channels.

Changed files:

- `third_party/fast-mac-gsplat/variants/v5_features/csrc/metal/gsplat_v5_features_kernels.metal`
- `third_party/fast-mac-gsplat/variants/v5_features/csrc/metal/gsplat_metal.mm`
- `third_party/fast-mac-gsplat/variants/v5_features/csrc/bindings.cpp`
- `third_party/fast-mac-gsplat/variants/v5_features/csrc/shared/common.h`
- `third_party/fast-mac-gsplat/variants/v5_features/torch_gsplat_bridge_v5_features/rasterize.py`
- `third_party/fast-mac-gsplat/variants/v5_features/tests/reference_alpha.py`
- `third_party/fast-mac-gsplat/variants/v5_features/tests/alpha_output_check.py`
- existing v5_features tests/benchmark docs updated for tuple return

Validation:

```text
uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace
uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python -m py_compile torch_gsplat_bridge_v5_features/rasterize.py benchmarks/benchmark_mps.py tests/feature_contract_check.py tests/reference_check.py tests/reference_alpha.py tests/alpha_output_check.py
uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python tests/alpha_output_check.py
  Test A passed.
  Test B passed.
  Test C passed.
  Test D passed.
  Test E passed.
uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python tests/feature_contract_check.py
  shape contract: ok
  F=3 v5 parity max_abs=0
  F=3 feature grad max_abs=3.7252903e-09
  F=8 feature grad max_abs=1.3969839e-09
  F=32 feature grad max_abs=6.9849193e-10
  F=32 no-NaN smoke: ok
uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python tests/reference_check.py
  B=1/B=2/saturated/presorted/overflow checks passed
```

Throughput command shape:

```text
uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python benchmarks/benchmark_mps.py \
  --height 256 --width 256 --gaussians 2048 --case medium_sigma_3_8 \
  --feature-dim F --backward --alpha-loss --warmup 3 --iters 10 \
  --batch-strategy auto --json
```

Measured forward+backward throughput with alpha loss included:

```text
F=3:  mean=18.006 ms, median=15.537 ms, forward=9.729 ms, backward=8.277 ms
F=8:  mean=20.458 ms, median=21.827 ms, forward=9.076 ms, backward=11.382 ms
F=32: mean=44.056 ms, median=32.020 ms, forward=19.880 ms, backward=24.176 ms
```

Note: `tests/reference_alpha.py` uses the existing v5/v5_features pixel-center
convention (`x + 0.5`, `y + 0.5`) so the reference matches the already-verified
kernel coordinate contract.
