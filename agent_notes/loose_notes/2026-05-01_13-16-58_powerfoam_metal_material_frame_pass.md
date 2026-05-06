# PowerFoam Metal Material-Frame Pass

## Context

The previous Metal `oriented_texel_surface` path learned a surface normal and
2D texel sites, but sampled texels with `local_coord.xy`. That made the detail
layer front-facing: tilted surfaces could move the clipping plane without
rotating the texel coordinate frame. This was one of the most direct
paper-math mismatches in the fast path.

## Edit

Changed the oriented-texel Metal layout from:

```text
[S * (C + 2)] + normal3
```

to:

```text
[S * (C + 2)] + normal3 + tangent3 + bitangent3
```

The shader now evaluates texel coordinates as:

```text
local = (x_hit - center) / radius
u = dot(local, tangent)
v = dot(local, bitangent)
```

and propagates gradients into tangent/bitangent as well as texel sites and
texel features. The trainer now owns a raw tangent parameter for
`oriented_texel_surface`; it projects that tangent orthogonal to the learned
normal and derives `bitangent = normalize(cross(normal, tangent))`.

Files touched:

- `third_party/powerfoam-metal/torch_powerfoam_metal/rasterize.py`
- `third_party/powerfoam-metal/csrc/metal/powerfoam_streaming_kernels.metal`
- `third_party/powerfoam-metal/csrc/metal/powerfoam_metal.mm`
- `src/train/train_powerfoam_metal.py`
- `third_party/powerfoam-metal/tests/linear_texture_check.py`
- `third_party/powerfoam-metal/benchmarks/benchmark_powerfoam_metal.py`
- `third_party/powerfoam-metal/README.md`
- `research_notes/foam_papers/powerfoam_reproduction_audit.md`
- `research_notes/foam_papers/powerfoam_mathematical_aspects_deep_dive.md`

## Tests

Rebuilt the extension:

```bash
( cd third_party/powerfoam-metal && rtk uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

Parity checks passed:

```bash
rtk uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python third_party/powerfoam-metal/tests/backward_check.py
rtk uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python third_party/powerfoam-metal/tests/linear_texture_check.py
```

Largest oriented-texel parity errors:

```text
features max error: 8.568e-07
alpha max error:    1.431e-06
points grad max:    1.194e-07
radii grad max:     1.788e-07
density grad max:   3.353e-08
texel_sites grad:   4.191e-09
texel_features grad:1.118e-08
normals grad:       2.346e-08
```

Added an explicit rolled-frame subcase to `linear_texture_check.py`, where
tangent and bitangent are independent differentiable inputs rather than only
derived from the normal. That also passed:

```text
explicit-frame features max error: 8.494e-07
explicit-frame alpha max error:    1.431e-06
explicit-frame points grad max:    1.322e-07
explicit-frame radii grad max:     1.788e-07
explicit-frame density grad max:   3.539e-08
explicit-frame texel_sites grad:   4.540e-09
explicit-frame texel_features grad:1.118e-08
explicit-frame normals grad:       2.200e-08
explicit-frame tangents grad:      1.506e-09
explicit-frame bitangents grad:    1.271e-09
```

Trainer smoke passed:

```bash
PYTHONPATH=src/train WANDB_MODE=offline rtk uv run python -c "..."
```

Smoke settings: oriented texel surface, 64 cells, 16 neighbors, 2 frames,
32px render, 1 step, W&B disabled. Step 0 `eval_l1=0.05739`; step 1
`eval_l1=0.05581`. The smoke logged nonzero center, radius, normal, tangent,
feature, and texel-site movement.

Benchmark after the frame change:

```bash
rtk uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python third_party/powerfoam-metal/benchmarks/benchmark_powerfoam_metal.py --cells 256,1024 --resolutions 128x128 --neighbors 32 --adjacency knn --warmup 3 --iters 8 --foam-backward --foam-texel-surface --compare-gs --gs-backward --json
```

Results:

```text
PowerFoam frame-texel 256 cells:  fwd median 5.195 ms, bwd median 4.700 ms, total median 10.197 ms
GS v5_features      256 cells:    fwd median 3.965 ms, bwd median 1.383 ms, total median  5.316 ms
PowerFoam frame-texel 1024 cells: fwd median 4.968 ms, bwd median 6.023 ms, total median 10.893 ms
GS v5_features      1024 cells:   fwd median 5.253 ms, bwd median 5.351 ms, total median 10.622 ms
```

## Reflection

This was not a speed optimization. It made the fast path more faithful by
fixing the frame semantics of the detail layer, and it added trainable
in-plane orientation. The price is a larger feature layout and extra dot
products/atomics in backward.

The result is still not full PowerFoam. It uses an explicit normal/raw-tangent
parameterization, not the official quaternion. It still lacks detail height,
spherical-Voronoi color, Cech/AABB adjacency, densification/resampling, static
multi-view training, and tiled replay.

## Next Falsification Tests

1. Tilted-plane synthetic: render one cell with known tangent/bitangent and
   verify that translating a texel site along local `u` moves color along the
   material tangent, not screen x.
2. Frame-roll fit: use one tilted plane with asymmetric texel colors and verify
   the raw tangent learns in-plane roll.
3. Height port: add per-site height after the frame test, because height must
   use the same local coordinates.
4. Re-run the benchmark with more iterations; the GS comparison in this run was
   noisy at 1024 cells and should not be used as a baseline-table result.

## Follow-up: Random Color Init For TokenGS Fairness

Added `model.color_init_mode` to `src/train/train_powerfoam_metal.py`:

```text
image   = keep the old image-sampled color/detail init
random  = keep geometry/video placement but initialize per-cell/per-texel colors randomly
```

Also added:

```text
src/train_configs/local_mac_powerfoam_metal_oriented_texel_surface_random_color_video_1024_smoke.jsonc
```

Reason: the image-color init is too favorable for judging a future
TokenGS-to-PowerFoam decoder. We want to see whether the foam state learns
appearance instead of starting near the answer.

Smoke:

```text
64 cells, 2 frames, 32px, 1 step, random color
step 0 eval_l1 = 0.24465
step 1 eval_l1 = 0.24219
```

Online W&B run:

```text
https://wandb.ai/nbardy/dynaworld/runs/rj8pjc05
run name: powerfoam-metal-material-frame-random-color-1024-120step
step 0 eval_l1 = 0.25368
step 120 eval_l1 = 0.14490
step 120 eval_mse = 0.04130
```

Comparison to the image-color init run:

```text
image-color init run:  https://wandb.ai/nbardy/dynaworld/runs/ecysgsk8
image step 0 eval_l1 = 0.04711
image step 120 eval_l1 = 0.01984
random step 0 eval_l1 = 0.25368
random step 120 eval_l1 = 0.14490
```

Interpretation: the old run did benefit strongly from appearance init. The
random-color run still learns, but it is far from matching the image-init run
after 120 steps. For TokenGS -> PowerFoam, this random-color mode is the better
diagnostic unless the token decoder is explicitly meant to predict initial
appearance.

## Follow-up: Sections Per Pixel / Saturation Diagnostic

Added:

```text
research_experiments/dynamic_foam/diagnose_powerfoam_sections.py
```

Command:

```bash
PYTHONPATH=src/train rtk uv run python research_experiments/dynamic_foam/diagnose_powerfoam_sections.py \
  --config src/train_configs/local_mac_powerfoam_metal_oriented_texel_surface_random_color_video_1024_smoke.jsonc \
  --checkpoint outputs/powerfoam_metal/powerfoam_metal_material_frame_random_color_1024_120step/checkpoint_final.pt \
  --frames 0,5,10,15 \
  --device mps
```

Result across sampled frames:

```text
active sections / pixel:
    mean ~= 1.34
    p50  = 1
    p90  = 2
    p95  = 2
    p99  = 3
    max  = 4

final alpha:
    mean ~= 0.926
    p50  ~= 0.974
    over alpha 0.95 ~= 73.5% of pixels
    over alpha 0.99 ~= 10.2% of pixels
    under alpha 0.10 ~= 2.7% of pixels
    under alpha 0.50 ~= 3.3% of pixels

early stop by transmittance threshold:
    0% of pixels
```

Interpretation: this run is geometrically under-layered but optically
over-opaque. Most pixels are explained by one or two foam sections, and those
sections are often individually near max alpha. The rasterizer still scans all
1024 sorted cells because transmittance rarely crosses the `1e-4` early-stop
threshold, but meaningful hit count is low.

Action implication: before increasing cell count, reduce per-section opacity
or density init/regularization so more than one or two sections can share a
pixel. If TokenGS decodes foam directly, it should probably predict density or
opacity scale carefully; otherwise tokens may learn a single opaque billboard
per ray instead of a useful foam tessellation.
