# 8192-Splat Loss Plateau Diagnostics

## Context

The 128px/4fps wide-depth Taichi run with `128 x 64 = 8192` splats completed
quickly and stably:

- `frames_per_step = 8`
- cosine LR `0.001 -> 0.0001`
- final sampled-window loss around `0.096`

This is much faster than the dense baseline and more stable than the raw 128px
run, but the loss did not improve dramatically over the 512-splat Taichi run.

## Interpretation

The 8192 splats are not 8192 independent optimized scene primitives in the
classic 3DGS sense. They are produced by:

- 128 learned tokens
- each token emits 64 sibling splats
- all 64 siblings from a token share the same refined token latent
- no adaptive densification/pruning/opacity reset
- no separate per-parameter optimizers for means/scales/opacity/color

So the current bottleneck may be token/query capacity, head parameterization,
opacity/scale dynamics, or loss/eval noise, not raw splat count.

The final training loss is also a sampled-window loss, not a fixed full-sequence
validation loss. Step-to-step comparison can be misleading unless we add a
stable eval scalar.

## Code Change

Enabled optimizer-only diagnostics for the 8192-splat config while keeping
renderer diagnostics off:

```text
with_metrics.renderer = false
with_metrics.optimizer = true
```

Added per-module/per-head gradient and parameter L2 metrics in
`src/train/debug_metrics.py`, including:

- `OptDiag/GradL2ByGroup/gaussian_heads_xyz_head`
- `OptDiag/GradL2ByGroup/gaussian_heads_scale_head`
- `OptDiag/GradL2ByGroup/gaussian_heads_opacity_head`
- `OptDiag/GradL2ByGroup/gaussian_heads_rgb_head`
- `OptDiag/GradL2ByGroup/tokens`
- encoder, ray projection, token block, and time projection groups

This should help decide whether we need a lower/higher LR globally or different
learning rates by head.

Also added stable full-sequence eval scalars whenever the trainer already
renders a validation video:

- `Eval/L1`
- `Eval/MSE`
- `Eval/Loss`
- `Eval/PSNR`

These should be used to compare 512 vs 8192 splats instead of relying only on
the sampled training-window `Loss`.

## Validation

- `uv run python -m py_compile src/train/debug_metrics.py src/train/dynamicTokenGS.py`
- Config parse confirmed optimizer metrics enabled and renderer metrics off.
- One-step smoke with optimizer diagnostics completed with finite loss and
  finite optimizer diagnostics.
- `uv run python -m py_compile src/train/dynamicTokenGS.py src/train/debug_metrics.py`
  passed after adding eval scalars.
- `git diff --check` passed.

## Likely Next Experiments

1. Add fixed full-sequence eval loss/PSNR, because sampled training-window loss
   is too noisy for comparing 512 vs 8192 splats.
2. Compare `128 x 64` against more independent tokens, e.g. `256 x 32`, because
   sibling splats from the same token are correlated.
3. Inspect per-head gradient norms. If xyz/scale/opacity/rgb are imbalanced,
   add parameter groups or head-specific LR multipliers.
4. Consider scale/opacity initialization or activation changes before adding
   more splats.
5. Longer-term: classic 3DGS quality depends heavily on adaptive density
   control: densify/split/clone/prune and opacity reset. This baseline has none
   of that; it only predicts a fixed set.
