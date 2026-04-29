# Colorize MLP Gray-Collapse: Diagnosis + Three Hypotheses

Date: 2026-04-29
Context: First F=32 feature-splatting run (`unconditioned-tokens-features-F32`). PCA-of-features video shows rich spatial structure across all 8 frames at step 0; the actual rendered RGB image at step 0 is uniform gray-teal. The MLP is collapsing the rich feature signal into a degenerate flat color.

## Measured at init (seeds {0, 1, 2}, fast-iter F32 config)

Run via the new `probe_colorize_init.py` (extends `init_diagnostics.py`):

```
PYTHONPATH=src/train uv run python src/train/probe_colorize_init.py \
  src/train_configs/local_mac_unconditioned_tokens_features_F32.jsonc \
  --seeds 0,1,2
```

| Metric | Value | What it means |
|---|---:|---|
| `PerPixelChroma/Mean` | 0.072 ± 0.004 | std across (R,G,B) per pixel; ~0 means every pixel is gray |
| `SpatialStdMean` | 0.018 ± 0.001 | std across pixels per channel; ~0 means image is spatially flat |
| `Range` (max - min) | 0.189 ± 0.021 | RGB values span only 19% of [0, 1] |
| `R/G/B Mean` | 0.36, 0.45, 0.48 | the teal-gray bias |
| `Logit/AbsGt0p5Frac` | 0.19 ± 0.14 | only 19% of pre-sigmoid values exceed \|0.5\| |
| **`Logit/AbsGt2Frac`** | **0.00** | **zero pre-sigmoid values exceed \|2\| — sigmoid stuck in linear band** |

The decisive number is `Logit/AbsGt2Frac = 0`. Sigmoid's nonlinearity engages around |z| ≳ 2 (where output is in [0.12, 0.88]). At init **no pre-sigmoid value gets there**, so output is `sigmoid(small) ≈ 0.5 + 0.25·z` — a tiny linear ramp around 0.5. That is gray + small variation. The teal bias comes from the random Conv2d bias init (each of the 3 output channels gets a slightly different bias, but all small).

## Math sketch (why pre-sigmoid is too small)

`FeatureToColor` (current default) is `nn.Conv2d(F=32, 3, kernel_size=1, bias=True)` with PyTorch's default Kaiming uniform init:

- `weight ~ U(-bound, bound)` with `bound = sqrt(1/fan_in) = sqrt(1/32) ≈ 0.177`
- `Var(W) ≈ 0.0104`
- `bias ~ U(-0.177, 0.177)` (also small)

The pre-sigmoid linear output for one pixel is `z_c = sum_f W_cf · x_f + b_c`. With features `x ~ N(0, σ_x)`:

```
Var(z_c) ≈ F · Var(W) · σ_x²  +  Var(b)
        ≈ 32 · 0.0104 · σ_x²  +  0.0104
        ≈ 0.333 · σ_x²        +  0.0104
```

The rasterized features in this run have small magnitude (`head_output_init_std=0.12`, then opacity-blended with `opacity_init=0.1` per splat). Per-pixel feature std is roughly `σ_x ≈ 0.2` empirically. So:

```
std(z_c) ≈ sqrt(0.333 · 0.04 + 0.0104) ≈ 0.154
```

Pre-sigmoid is in roughly [-0.5, 0.5] (3σ). `sigmoid([-0.5, 0.5]) = [0.378, 0.622]`. **That is the 0.244 RGB range we measured (matches `Range = 0.19` after attenuation by alpha-blending).**

To make sigmoid engage its non-linear region, we need `std(z_c) ≳ 1.5` — about **10× larger** than current.

## Three hypotheses + experiments

Each hypothesis names what to change, why it should help mathematically, and the metric the probe will produce that confirms or falsifies it.

### Hypothesis 1: Increase the colorize weight std (10× gain)

**Change:** Initialize `Conv2d.weight` with `std ≈ 5/sqrt(F)` instead of Kaiming default `1/sqrt(F)`. Concretely, after `nn.Conv2d` construction, multiply the weight tensor by 5.0.

**Math:** `std(z_c)` scales linearly with weight std, so `std(z_c) ≈ 0.154 · 5 = 0.77`. Pre-sigmoid range becomes ~[-2.3, 2.3], `sigmoid` covers [0.09, 0.91]. **Should hit `AbsGt2Frac > 0.3`.**

**Knob:** add a `colorize.weight_init_gain: float` config key (default 1.0 = current behavior). Multiply `Conv2d.weight.data` by gain after default init.

**Predicted probe output:**
- `Logit/AbsGt2Frac` rises from 0.00 → ~0.3
- `PerPixelChroma/Mean` rises from 0.07 → ~0.20+
- `SpatialStdMean` rises from 0.018 → ~0.15
- `Range` rises from 0.19 → ~0.7

**Risk:** if too aggressive, sigmoid saturates → vanishing gradients early in training. Sweet spot is probably gain ∈ [3, 7].

### Hypothesis 2: Bias-only color spread (bias = logit(uniform(0.05, 0.95)))

**Change:** Init the 3 output biases by sampling 3 colors uniform in [0.05, 0.95] and storing their logits as biases. So at init, before any feature contribution, each pixel renders a saturated random color.

**Math:** `bias_c = logit(u_c)` where `u_c ~ U(0.05, 0.95)`. With small weight, `z_c ≈ bias_c + small_noise`, so `sigmoid(z_c) ≈ u_c + tiny`. Each output channel reads as a different saturated value → not gray.

This **does not** add per-pixel diversity (every pixel gets the same color), but it breaks the teal-gray symmetry and gives a non-degenerate starting point for the bias to learn from.

**Knob:** add a `colorize.bias_init: "uniform_logit" | "default"` config key. When `uniform_logit`, sample biases as above.

**Predicted probe output:**
- `R/G/B Mean` becomes spread across [0.05, 0.95] (no longer ~0.4 each)
- `Range` rises from 0.19 → ~0.5+ (because the channels alone differ)
- `PerPixelChroma/Mean` also rises (R≠G≠B per pixel)
- `SpatialStdMean` stays low (~0.02) — bias alone doesn't add per-pixel variance

**Falsified if:** `SpatialStdMean` doesn't rise. Then the gray-collapse isn't only "bias means too close" — it's also the weight magnitude story (H1).

### Hypothesis 3: Identity-passthrough init (3 features ≡ RGB at init)

**Change:** Initialize the conv weight as a sparse identity-like matrix that picks 3 specific feature channels as RGB. Concretely:

```python
W[c, c, 0, 0] = 4.0  for c in {0, 1, 2}     # picks features 0, 1, 2 as R, G, B
W[c, f, 0, 0] = 0.0  otherwise
b[c] = -2.0           # so sigmoid(4·feature - 2) covers [0, 1] as feature ∈ [0, 1]
```

**Math:** at init, `z_c = 4·x_c - 2`. If splat features 0..2 happen to vary across pixels (which the rasterized PCA shows they do), `z_c` lands in [-2, 2] easily, and `sigmoid` covers [0.12, 0.88] per pixel. Maps the F=32 case back to the F=3 baseline behavior at init, then training learns to incorporate the other 29 channels.

Subtler: the splat-level feature head output (`gaussian_heads.rgb_head`) for the first 3 channels is now effectively the RGB head. So if we want the F=3 baseline parity, we'd also want to set `rgb_head.bias[:3]` to logit(0.5) and ensure the feature MLP routes diverse values through channels 0..2.

**Knob:** add a `colorize.init: "passthrough3" | "default"` key. When `passthrough3`, set the weight as above.

**Predicted probe output:**
- `Range` rises sharply (channels 0..2 dominate; their per-splat variance dominates the output)
- `SpatialStdMean` rises to ~0.1+
- `Logit/AbsGt2Frac` rises to ~0.1 (less than H1, because gain of 4 is smaller than H1's 5x of all 32 channels superimposing)
- Behaviorally, the run starts close to F=3 parity in PSNR.

**Falsified if:** SpatialStdMean stays low. Then the rasterized feature channels themselves are too low-variance — gray-collapse is upstream of the colorize MLP, and we need to bump `head_output_init_std` or `opacity_init` for features.

## Recommended order to try

1. **H1 first** — the math is unambiguous and the change is one line (`conv.weight.data.mul_(gain)`). Run probe, see if `AbsGt2Frac` rises and `PerPixelChroma/Mean` doubles. ~30 s per try.
2. **H1 + H2 combined** — a saturated random-color start + properly-engaged sigmoid. Highest expected PSNR jump from a single init change.
3. **H3** — only if H1 doesn't lift `SpatialStdMean` enough. H3 binds the colorize output to the specific feature channels which is more invasive.

## Out-of-scope but worth noting

- **Pre-norm**: putting a `LayerNorm` before the conv would re-scale per-pixel features to unit std, making `z_c` magnitude much larger regardless of feature head init. But this changes the model graph; keep this as a fallback if H1–H3 don't work.
- **Hidden-layer colorize MLP** (`colorize.hidden_dim: 64`): a GELU non-linearity in the middle could break the linear-projection-near-zero failure mode by introducing per-pixel non-linearity. Cheap to test via the existing config knob; should run probe after.
- **Splat-level feature init** is upstream and orthogonal: increasing `head_output_init_std` on the feature head also increases σ_x, which propagates into `std(z_c)`. But by the math above we'd need σ_x ≈ 2.7 to hit `std(z_c) = 1.5` at the current weight scale, which is huge — better to fix the colorize side.

## Probe usage cheat-sheet

```bash
# Default config (gray-collapse baseline)
PYTHONPATH=src/train uv run python src/train/probe_colorize_init.py \
  src/train_configs/local_mac_unconditioned_tokens_features_F32.jsonc \
  --seeds 0,1,2

# After implementing H1: edit colorize.py to support `weight_init_gain` and add it to the F32 config.
# Re-run the probe and compare. Specifically watch:
#   - Logit/AbsGt2Frac : 0.00 -> ?
#   - PerPixelChroma/Mean : 0.07 -> ?
#   - SpatialStdMean : 0.018 -> ?
#   - Range : 0.19 -> ?
```

The CLI prints `mean ± std` across seeds for the eight key metrics so per-iteration noise is visible.
