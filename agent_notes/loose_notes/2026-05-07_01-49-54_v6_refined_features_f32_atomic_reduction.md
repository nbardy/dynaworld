# v6 Refined Features F32 Reduction Fork

## Context

The active training bottleneck was the multicam feature-splatting path at
`512x512`, `B=16`, `G=8192`, `F=32`. Earlier probes showed that batch and
feature dimension inflated memory/time much more than simply increasing
Gaussian count. The specific red flag was that `v6_refined_features F=3` was
much slower than RGB `v6`, even though both have three output channels.

Important correction from the session:

- Do not mutate the stable `variants/v6_refined_features` kernel while probing.
- Stable baselines need to remain available exactly as baselines.
- The optimization work was moved into an isolated fork:
  `variants/v6_refined_features_f32_reduce`.
- `git diff -- variants/v6_refined_features` is empty after the restore.

## Current Model

Observed fact:

- RGB `v6` uses `float3` color staging/reduction in backward.
- Stable `v6_refined_features` uses a generic loop and direct per-pixel atomics:
  `atomic_add_feature_grads(g_colors, grad_features, pix, g, scale, mi)`.

Inference:

- F-channel backward is dominated by the color-gradient atomic stream, not by
  pure splat math.
- At `F=32`, every contributing pixel/Gaussian pair issues 32 device atomics.
  That makes `F` and `B` multiply the slowest part of the backward path.

## Forked Code

New fork:

- `third_party/fast-mac-gsplat/variants/v6_refined_features_f32_reduce/`
- Python package: `torch_gsplat_bridge_v6_refined_features_f32_reduce`
- custom op namespace: `torch.ops.gsplat_metal_v6_refined_features_f32_reduce`
- output API: `(features, accumulated_alpha)`

Implementation inside the fork only:

- Added `F == 3` specialized backward reduction using `float3`, mirroring the RGB
  reduction structure.
- Added `reduce_atomic_add_feature_grads(...)` for generic `F`, using
  `simd_sum` into threadgroup memory and one reduced atomic per
  Gaussian/channel/threadgroup.
- Added a `reserved0` metadata flag in backward to skip color-gradient atomics
  when `ctx.needs_input_grad[2]` is false.
- Avoided dense grad clones on the common no-overflow path.
- Added `--freeze-colors` to the fork benchmark script.

No trainer dispatch or checked-in config currently points at this fork. That is
intentional until a trainer smoke and full phase timing pass says it is ready.

## Metal Notes From Docs

Official docs referenced:

- https://developer.apple.com/documentation/metal/creating-threads-and-threadgroups
- https://developer.apple.com/documentation/metal/calculating-threadgroup-and-grid-sizes
- https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/setthreadgroupmemorylength(_:index:)
- https://developer.apple.com/documentation/apple-silicon/porting-your-metal-code-to-apple-silicon
- https://developer.apple.com/metal/resources/

Relevant takeaways:

- Threadgroups share `threadgroup` memory, and SIMD groups execute lanes
  together. That makes per-threadgroup gradient reduction the right shape for
  high-contention per-Gaussian gradients.
- Barriers must remain on uniform control flow. The feature reduction helper is
  only called under tile-uniform metadata conditions.
- Apple warns not to assume SIMD width in production. This fork still follows
  the existing `GSP_SIMD_WIDTH=32` convention, so a future portability pass
  should plumb runtime thread-execution width where practical.
- Threadgroup memory helps only if it reduces global traffic or contention. Do
  not stage large feature payloads without a measured global-memory win.

## Validation

Build command:

```bash
( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/v6_refined_features_f32_reduce
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

Contract check:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/v6_refined_features_f32_reduce/tests/feature_contract_check.py
```

Result:

```text
shape contract active_policy=off: ok
F=3 v5 parity active_policy=off max_abs=0
shape contract active_policy=on: ok
F=3 v5 parity active_policy=on max_abs=0
F=3 feature grad active_policy=off max_abs=1.8626451e-09
F=8 feature grad active_policy=off max_abs=9.3132257e-10
F=32 feature grad active_policy=off max_abs=2.3283064e-10
F=32 feature grad active_policy=on max_abs=2.3283064e-10
F=32 no-NaN smoke active_policy=off: ok
F=32 no-NaN smoke active_policy=on: ok
```

Alpha check:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/v6_refined_features_f32_reduce/tests/alpha_output_check.py
```

Result:

```text
Test A passed.
Test B passed.
Test C passed.
Test D passed.
Test E passed.
Test F passed.
```

## Benchmark Evidence

Local MPS, same-session comparison, `512x512`, `B=16`, `G=8192`, `F=32`,
`case=medium_sigma_3_8`, `batch_strategy=flatten`, `active_policy=off`,
`warmup=1`, `iters=3`.

| Variant | Forward ms | Backward ms | Total ms |
| --- | ---: | ---: | ---: |
| stable `v6_refined_features` | 216.6 | 1712.1 | 1928.7 |
| fork `v6_refined_features_f32_reduce` | 245.0 | 624.7 | 869.7 |

Interpretation:

- The fork is about `2.2x` faster on this F32 raster-only microbenchmark.
- Most of the win is backward: reduced global `g_colors` atomics are the likely
  mechanism.
- Forward is not improved and was slightly slower in this short run, so this is
  not a universal raster improvement.
- This is still a raster-only probe. The trainer needs phase timing before this
  is promoted, because dense `[B,H,W,F]` tensors and loss/backward retention can
  dominate wall time.

## Open Work

- Add a trainer selection hook only after the fork passes a 1-step F32 trainer
  smoke and a phase trace with `sample/encode/project/raster_forward/loss/backward/optimizer`.
- Run a broader matrix over `B`, `F`, `G`, and resolution to check whether the
  win survives beyond this one high-contention case.
- Stage `colors/features` in threadgroup memory only as a separate forked
  experiment, not inside either stable `v6_refined_features` or this reduction
  fork.
- Consider trainer-side framewise/microbatch backward for camera-swap relpose so
  dense feature images do not all stay live until one giant backward.
- If the intended follow-up freezes splat/features and trains only the camera
  offset head, use the skipped-color-gradient path and verify full trainer phase
  timings, not just raster-only timing.
