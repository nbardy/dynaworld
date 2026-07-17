# STAR UVT Backward Kernel Matrix

Date: 2026-05-18
Workspace: `/Users/nicholasbardy/git/gsplats_browser/dynaworld`

## User Goal

Dig into STAR UVT backward, add diagnostics/benchmarks, run the available
kernel families across STAR UVT variants, and find whether any kernel drops
backward time versus the current `direct_atomic + index_add` path.

## Script Added

Reusable matrix runner:

```text
research_experiments/star_uvt_backward_kernel_matrix.py
```

It launches each kernel/mode in a separate Python process so Metal tile
constants and custom-op initialization do not leak between cases. It writes:

```text
manifest.json
cases/*.json
logs/*.log
summary.csv
summary.md
```

## Runnable Variant Inventory

```text
star_uvt_v0:
  Runnable screen-space UVT backward modes. This is the current first-class
  STAR UVT overfit family.

star_uvt_prt_v0:
  Runnable projective-rational PRT kernels, but verified at much smaller tube
  counts and not apples-to-apples with the 32768 screen-space UVT case.

star_prt_v0:
  Importable Python scaffold, but no built custom op in this checkout. A tiny
  compact-backward call reports: "star_prt_v0 custom ops not found".

spacetime_v0:
  Reference/skeleton only. README says it is not buildable yet and pass-0
  projection/backward is not implemented.
```

## Main Screen-Space Matrix

Artifact:

```text
outputs/benchmarks/2026-05-18_star_uvt_backward_kernel_matrix_512_64f/
```

Case:

```text
target_size=512
frames=64
tube_count=32768
tile_t=1
tile_capacity=256
warmups=0
repeats=1
```

Fastest successful cold single-run rows:

| mode | reducer | sample unit | total sample+reduce |
| --- | --- | --- | ---: |
| `direct_split_fixedpoint` | `index_add` | `direct_tube_grad` | 570.68 ms |
| `direct_fixedpoint` | `index_add` | `direct_tube_grad` | 597.78 ms |
| `direct_atomic` | `index_add` | `direct_tube_grad` | 601.55 ms |
| `tile_pair_atomic` | `index_add` | `direct_tube_grad` | 666.94 ms |
| `tile_pair_suffix_reduced` | `index_add` | `direct_tube_grad` | 817.48 ms |
| `tile_pair_suffix` | `key_sort_segmented_metal` | `tile_pair` | 974.46 ms |

Large/failed rows:

```text
atomic_append + index_add failed: 16.00 GiB buffer request
with_keys + key_sort_segmented_metal failed: 16.00 GiB buffer request
tile_pair_scanline + key_sort_segmented_metal failed: 12.00 GiB buffer request
tile_pair + key_sort_segmented_metal: 4190.90 ms
tile_pair_parallel + key_sort_segmented_metal: 16127.20 ms
```

Interpretation:

```text
There is no hidden exact/sample-table kernel that beats direct atomic at the
current 512px/64f/32768 setting. The only rows near or better than direct atomic
are the direct per-tube fixedpoint/split-fixedpoint family.
```

## Warmed Top-Kernel Rerun

Artifact:

```text
outputs/benchmarks/2026-05-18_star_uvt_backward_top_warm3/
```

Case:

```text
target_size=512
frames=64
tube_count=32768
warmups=1
repeats=3
```

| mode | total sample+reduce median |
| --- | ---: |
| `direct_fixedpoint` | 718.20 ms |
| `direct_atomic` | 729.76 ms |
| `direct_split_fixedpoint` | 836.11 ms |
| `tile_pair_atomic` | 1008.42 ms |
| `tile_pair_suffix + key_sort_segmented_metal` | 1037.64 ms |

Interpretation:

```text
The cold `direct_split_fixedpoint` lead did not survive the warmed rerun.
`direct_fixedpoint` is the only warmed direct-kernel candidate that beat
direct_atomic, and the margin was small at the kernel-only level.
```

## Full Train-Step Top-Kernel Rerun

Artifact:

```text
outputs/benchmarks/2026-05-18_star_uvt_trainstep_top_warm/
```

Case:

```text
target_size=512
frames=64
tube_count=32768
warmup_steps=1
steps=3
```

| mode | total step median | forward | loss | backward | optimizer |
| --- | ---: | ---: | ---: | ---: | ---: |
| `direct_atomic` | 1043.80 ms | 68.95 ms | 20.47 ms | 852.76 ms | 7.00 ms |
| `direct_fixedpoint` | 835.31 ms | 95.94 ms | 21.03 ms | 705.63 ms | 11.07 ms |
| `direct_split_fixedpoint` | 946.70 ms | 71.24 ms | 23.32 ms | 834.22 ms | 6.92 ms |

Interpretation:

```text
Yes, one mode dropped full backward time in this run:
direct_fixedpoint reduced full backward by about 147 ms versus direct_atomic
at 512px/64f/32768, roughly 17% of backward and 20% of total step.
```

However, this is not a promotion by itself. Existing local benchmark notes say
`direct_fixedpoint` is exact-repeatable but was rejected as a training path:
default scale went nonfinite around step 160, `scale1e4` around step 350, and
quality was far below deterministic tile-pair rows. `direct_split_fixedpoint`
has the same warning: exact-repeatable, but earlier 200-step STAR-only artifact
went nonfinite around step 120 with poor heldout PSNR.

So the current practical conclusion is:

```text
direct_atomic:
  Still the practical quality/overfit default.

direct_fixedpoint:
  Real backward-speed candidate, but only as a new stability investigation.
  Do not switch the first-class 512px overfit config to it without a short
  stability bracket and quality check.

direct_split_fixedpoint:
  Not faster after warmup/full-step timing, and already has negative stability
  evidence.
```

## PRT Variant Probe

Artifacts:

```text
outputs/benchmarks/2026-05-18_star_uvt_prt_backward_kernel_matrix_256_16f_512t/
outputs/benchmarks/2026-05-18_star_uvt_prt_backward_kernel_matrix_256_16f_512t_rerun/
```

Case:

```text
target_size=256
frames=16
tube_count=512
tile_config=4x4x2:512
```

Useful PRT facts:

```text
direct_serial timed out at 25-45s.
tile_pair_atomic was extremely slow when it completed: about 16.3s backward.
tile_pixel_atomic produced usable timing, but not a pass because the one-step
loss did not decrease in the probe's pass/fail contract.
fused_mse passed and was faster than separate render+loss+backward inside PRT:
  first run: 155.09 ms fused vs 294.10 ms separate, 1.90x speedup
  rerun under heavier local state: 625.68 ms fused vs 1145.10 ms separate, 1.83x speedup
```

Interpretation:

```text
PRT fused-MSE is a real internal PRT optimization, but it is not a drop-in
replacement for the current 32768-tube screen-space STAR UVT path. It runs at
verified 512-tube scale here and solves a different projective-rational problem.
Keep it as a separate PRT lane, not the immediate answer to the current
source-view UVT backward cost.
```

## Bottom Line

The current backward-speed hierarchy for the 512px/64f/32768 screen-space case:

```text
1. direct_fixedpoint can reduce backward time in short timing probes.
2. direct_atomic remains the usable training default because fixedpoint has
   known nonfinite/quality failures in longer training.
3. direct_split_fixedpoint is not currently better after warmup/full-step timing.
4. sample-table exact/deterministic branches remain slower or blow up memory.
5. PRT fused-MSE is promising within PRT, but not comparable to 32768 UVT tubes.
```

Next diagnostic target if we want to exploit the speed win:

```text
Run a short direct_fixedpoint stability bracket at 512px/64f/32768:
  - scale 1e6, 1e5, 1e4
  - lower LR brackets
  - finite-grad / finite-loss checks every step
  - compare final PSNR/SSIM and contact sheet against direct_atomic

Only promote direct_fixedpoint if it survives the same short overfit gate that
direct_atomic already passes.
```
