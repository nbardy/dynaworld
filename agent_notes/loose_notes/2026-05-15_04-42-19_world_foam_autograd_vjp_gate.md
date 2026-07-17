# World Foam autograd VJP gate

## Context

The raw VJP smoke now covered both RGB-only and nonzero alpha/depth adjoint
seeds, but train/eval uses the `fused_slab_affine_num32_den16_autograd`
wrapper. That wrapper has one important dispatch detail: `direct_atomic_rgb_only`
uses the raw RGB-only kernel only when alpha/depth output gradients are absent;
if alpha/depth gradients exist, it falls through to the general grad-only path.

That behavior should be proven by the smoke artifact, not just inferred from
Python code.

## Change

The fused slab smoke now runs an autograd-level gradient check whenever
`--include-vjp` is set:

- build the same deterministic VJP seed used by the raw VJP check
- run `fused_slab_affine_num32_den16_autograd` for `reduce`, `direct_atomic`,
  `direct_atomic_grad_only`, `direct_atomic_rgb_only`, and `direct_atomic_track`
- form a scalar loss from `rgb`, `alpha`, and `depth` outputs using the seed
- compare each `site_rgba.grad` against autograd reduce
- compare autograd reduce against the raw reducer VJP

The aggregate verifier now requires `autograd_vjp_diagnostics` for its smoke
artifacts.

## Evidence

Regenerated the RGB and RGBA/depth smoke artifacts:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_fused_slab_affine_realray_mps.py \
  --frame-counts 2 \
  --render-size 16 \
  --site-count 12 \
  --layout per-track \
  --candidate-order slab-mid-depth \
  --time-slabs 1 \
  --include-vjp \
  --vjp-seed-mode rgb \
  --vjp-reduce-chunk-size 16 \
  --timing-iters 1 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_direct_atomic_track_smoke_2f_render16_pertrack.json
```

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_fused_slab_affine_realray_mps.py \
  --frame-counts 2 \
  --render-size 16 \
  --site-count 12 \
  --layout per-track \
  --candidate-order slab-mid-depth \
  --time-slabs 1 \
  --include-vjp \
  --vjp-seed-mode rgba-depth \
  --vjp-reduce-chunk-size 16 \
  --timing-iters 1 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_vjp_rgba_depth_smoke_2f_render16_pertrack.json
```

RGBA/depth autograd results:

```text
status: ok
max autograd mode rel delta vs autograd reduce: 1.16e-6
autograd reduce rel delta vs raw reduce: 0
raw RGB-only rel delta vs reducer: 2.82e-2, expected divergence
autograd RGB-only rel delta vs autograd reduce: 1.16e-6, expected wrapper fallback
```

Aggregate verifier:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_fused_slab_mixed_scaling.py \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_scaling_verifier_with_autograd_vjp_smoke.json
```

Result:

```text
status: ok
failures: []
```

## Takeaway

The current best `direct_atomic_grad_only` path is now checked at both raw VJP
and autograd-wrapper levels. The RGB-only raw kernel remains intentionally
RGB-only, but the autograd wrapper is safe under nonzero alpha/depth adjoints
because it falls back to a general path and matches reduce within tolerance.
