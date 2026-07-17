# Renderer Scaling Matrix: STAR UVT vs Dynamic GSplat vs F32 Feature Splatting

Date: 2026-05-18
Workspace: `/Users/nicholasbardy/git/gsplats_browser/dynaworld`

## Question

We wanted the same 64-frame / 32,768-primitive stress case across the current
renderer families:

- screen-space STAR UVT direct/compact backward modes
- per-frame dynamic Gaussian splat raster variants
- F32 feature-splat forks, including the remembered optimized paths

The user specifically asked whether the older feature-splatting speed work was
the bounding/tile-support trick that avoids exploding as `gaussians * pixels`
grow, and whether STAR UVT is missing the same idea.

## Artifacts

Summary report:

```text
outputs/benchmarks/2026-05-18_renderer_scaling_report.md
outputs/benchmarks/2026-05-18_renderer_scaling_report.csv
```

Inputs behind that report:

```text
outputs/benchmarks/2026-05-18_star_uvt_scale_128_64f_32768_top/
outputs/benchmarks/2026-05-18_star_uvt_scale_256_64f_32768_top/
outputs/benchmarks/2026-05-18_star_uvt_scale_512_64f_32768_top/
outputs/benchmarks/2026-05-18_fastmac_rgb_dynamic_B64_G32768_res128_256_512.jsonl
outputs/benchmarks/2026-05-18_feature_f32_B64_G32768_res128.jsonl
outputs/benchmarks/2026-05-18_feature_f32_B64_G32768_res256.jsonl
outputs/benchmarks/2026-05-18_feature_f32_B64_G32768_res512.jsonl
```

Helper changes:

```text
research_experiments/star_uvt_backward_kernel_matrix.py
research_experiments/renderer_scaling_report.py
```

`star_uvt_backward_kernel_matrix.py` now accepts `--v0-mode-cases`, so future
scale probes can avoid re-running known huge memory/failure modes.

## Key Timing Results

All rows below are median-ish or mean from tiny local probes, not long-run
quality gates.

STAR UVT screen-space backward, 64f / 32768 tubes:

```text
128px: direct_atomic 110.4 ms, direct_fixedpoint 112.9 ms
256px: direct_fixedpoint 174.9 ms, direct_atomic 182.4 ms
512px: direct_fixedpoint 507.1 ms, direct_atomic 521.5 ms
```

Dynamic RGB projected raster, B=64 / G=32768:

```text
128px best: v6_refined 3074.5 ms total, overflow tiles ~3842
256px best: v8 252.7 ms total, backward 172.2 ms, overflow 0
512px best: v8 693.0 ms total, backward 541.3 ms, overflow 0
```

The 128px dynamic row is not a typo: with 32,768 splats squeezed into 128px,
the tile occupancy explodes and many kernels hit overflow/fallback behavior.
At 256/512 the projected load is much cleaner.

F32 feature projected raster, B=64 / G=32768 / F=32:

```text
128px: stable v6_refined_features 8022.7 ms; fixedbin/v11 fail on tile overflow
256px: v11 fixedbin 1582.2 ms vs stable 3642.2 ms
512px: f32_fixedbin 5920.8 ms vs stable 41036.8 ms
```

This confirms the remembered feature-splatting improvement is real in the
high-pixel/no-overflow regime. The optimization family is not "STAR UVT";
it lives in fast-mac feature raster forks and primarily reduces the expensive
F32/color-gradient path plus fixed/no-overflow binning overhead. STAR UVT's
current RGB screen-space direct modes do not have an equivalent F32 feature
tube renderer or that fixedbin/gradcache feature-color path.

## Interpretation

For RGB at 512px/64f/32768, STAR UVT direct backward is in the same ballpark as
the better dynamic RGB raster backward, but the comparison is not a full trainer
row:

```text
STAR UVT direct_atomic kernel-only: 521.5 ms
Dynamic GSplat v8 projected raster total: 693.0 ms, backward 541.3 ms
```

For F32 features, STAR UVT is simply missing the comparable optimized feature
path. The current feature forks show exactly why the user remembered a better
pixel-scaling story:

```text
512px stable F32 backward: 38331.1 ms
512px f32_fixedbin backward: 4010.8 ms
```

That is a shader/kernel-path difference, not a data-loader issue.

The full `fixed_step_speed_compare.py` STAR-vs-free-dynamic trainer harness was
started for 128/256/512x64 with 32768 primitives, but it did not emit buffered
output in an interactive window and was stopped. Do not cite it as evidence.
Use the completed kernel/raster matrices above until a smaller unbuffered
trainer-step harness is added.

## Next Useful Work

1. Add a STAR UVT feature-tube fork rather than mutating `star_uvt_v0` RGB
   kernels. It needs an explicit `feature_dim` contract and parity checks.
2. Port the useful F32 ideas into that fork: no-overflow/fixedbin where valid,
   gradcache/accum variants, and a strict overflow fallback path.
3. Add an unbuffered full-step micro-harness for only STAR UVT vs free dynamic
   at one case at a time, so trainer-step comparison failures are observable.
4. Keep direct_atomic as the current quality default; `direct_fixedpoint` is
   faster in short rows but still has prior nonfinite/quality caveats.
