# World Foam RGBA/depth VJP gate

## Context

The fused mixed VJP smoke had only been seeding `grad_rgb = 1` with zero
alpha/depth adjoints. That matches the current RGB reconstruction training
path, but it left a correctness gap for the exposed `alpha` and `depth`
outputs.

## Change

Added `--vjp-seed-mode` to the fused slab smoke:

- `rgb`: existing training-path seed (`grad_rgb` nonzero, alpha/depth zero)
- `rgba-depth`: deterministic nonzero RGB, alpha, and depth adjoints

The non-RGB seed keeps direct-atomic, grad-only, and track variants matched
against the reducer, while treating the RGB-only shortcut as expected to
diverge. This proves the seed actually exercises the alpha/depth terms instead
of silently replaying the old RGB-only check.

## Evidence

Ran:

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

Result:

```text
status: ok
direct_atomic grad rel delta vs reducer: 6.68e-7
direct_atomic_grad_only grad rel delta vs reducer: 9.80e-7
direct_atomic_track grad rel delta vs reducer: 4.45e-7
direct_atomic_rgb_only grad rel delta vs reducer: 2.82e-2, expected divergence
```

Then reran the aggregate verifier with the new smoke artifact included:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_fused_slab_mixed_scaling.py \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_scaling_verifier_with_rgba_depth_smoke.json
```

Result:

```text
status: ok
failures: []
```

## Takeaway

The current best `direct_atomic_grad_only` path is now covered for both the RGB
training-path seed and a nonzero alpha/depth adjoint seed. The RGB-only shortcut
should remain labeled as RGB-only; it is not a valid general RGBA/depth VJP.
