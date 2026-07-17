# STAR UVT Native Target-Area Rowmajor W^T

Date: 2026-05-20

## Why

The hidden-preact split isolated the native target-area hidden backward cost:
output+GELU prebackward is small, while the F32
`hidden_weight^T @ grad_hidden_pre` reconstruction is the expensive subpiece,
especially at 512px. The cheapest exact follow-up was to keep the math the same
but change W^T accumulation order so Metal reads each hidden-weight row
contiguously.

## What Changed

- Added `target_area_star_only_rowmajor_wt` as an exact full-gradient mode.
- Added `target_area_recompute_only_rowmajor_wt` as a benchmark-only
  recompute-floor mode.
- Extended native target-area hidden VJP `mode_bits` to `0..63`.
- Rebuilt the `star_uvt_v0` extension after the Metal/C++ change.

## Results

Tiny parity passed for F4/F32 full gradients. F32 max feature-gradient error was
`2.33e-10`.

Same-session native timing:

| Mode | 256 total/backward ms | 512 total/backward ms |
|---|---:|---:|
| canonical full | `825.59 / 647.42` | `2513.22 / 2040.47` |
| rowmajor full | `933.49 / 711.50` | `2701.51 / 2161.58` |
| canonical recompute-only | `780.68 / 572.12` | `2533.28 / 1993.01` |
| rowmajor recompute-only | `759.97 / 555.79` | `2492.24 / 1983.42` |

## Decision

Reject row-major W^T as a promoted trainable path. It helps the isolated
recompute floor slightly but slows the full exact-gradient path. The bottleneck
is still dense F32 W^T reconstruction under full-support target-area pixels,
not output-gradient atomics or simple memory-ordering. Next useful work should
reduce/avoid that reconstruction or change the visual objective/support.

Report:
`outputs/benchmarks/2026-05-20_star_uvt_native_target_area_rowmajor_wt_gate.md`
