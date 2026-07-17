# World Foam depth-order probe

## Context

We were looking for another STAR-UVT-style cleanup in the World Foam fused slab
path after the owner-update VJP shortcut failed to beat the current
`direct_atomic_grad_only` winner. The next candidate shortcut was to avoid
per-sample insertion sorting by relying on the existing per-track candidate
append order.

The core question was whether the CPU-side `slab-mid-depth` candidate ordering
is already monotone enough in real per-frame camera depth that a Metal shader
could safely do ordered append instead of sorting.

## Probe

Added CPU diagnostics to the fused slab smoke/train-eval bundle builders:

- per-sample adjacent depth inversion counts
- samples with at least one adjacent inversion
- maximum adjacent inversions per sample
- maximum depth drop
- `ordered_append_safe`

Then ran the forward-only render32, 2/4/8/16-frame probe:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_fused_slab_affine_realray_mps.py \
  --frame-counts 2,4,8,16 \
  --render-size 32 \
  --site-count 12 \
  --layout per-track \
  --candidate-order slab-mid-depth \
  --time-slabs 1 \
  --timing-iters 1 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_depth_order_probe_render32_pertrack_2_4_8_16.json
```

## Result

The probe passed as an artifact check, but it decisively rejected ordered append:

```text
ordered_append_safe: false
adjacent_inversions: 803444 / 2251403 adjacent pairs
adjacent_inversion_rate: 0.35686369788083255
samples_with_adjacent_inversions: 61386 / 61440
max_adjacent_inversions_per_sample: 30
max_depth_drop: 3.173972027687824
```

Per-frame adjacent inversions:

```text
2f:  62378 / 149718, 4096 / 4096 samples inverted
4f:  115903 / 297863, 8192 / 8192 samples inverted
8f:  214115 / 597255, 16380 / 16384 samples inverted
16f: 411048 / 1206567, 32718 / 32768 samples inverted
```

## Takeaway

Do not implement the ordered-append shader off the current `slab-mid-depth`
ordering. The track/slab ordering is not depth ordered from the per-frame camera
view, so the shader still needs sorting or a different candidate/tape layout.

This reinforces the main scaling read:

- `direct_atomic_grad_only` remains the best measured current path.
- Local shortcuts are not enough to make World Foam as clean as STAR UVT.
- The remaining gap is structural: frame-local candidate depth evaluation,
  sorting/blending, and VJP replay still happen too much per frame.
