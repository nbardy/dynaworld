# STAR UVT Vec4 W^T Trainer A/B

Date: 2026-05-20

## Why

The vec4 W^T native target-area kernel looked strong in direct timing, but the
first trainer smoke was compared against an older canonical artifact. This gate
reran canonical native target-area on the current build so we can decide whether
the vec4 path should stay opt-in or become the preferred full-support native
target-area VJP mode.

## Results

Both rows pass, use the same checkpoint/config shape, and keep zero tile
overflow.

| Mode | Mean step | Last step | Mean backward | Last backward | Mean sparse visual backward |
|---|---:|---:|---:|---:|---:|
| canonical | `4262.08ms` | `4800.51ms` | `3700.16ms` | `4120.29ms` | `2546.72ms` |
| vec4 W^T | `4071.01ms` | `3747.06ms` | `3152.61ms` | `3078.14ms` | `1963.54ms` |

Endpoint quality is matched: feature loss `0.625451`, sparse visual PSNR
`5.746`, probe PSNR `22.029`, full RGB PSNR `5.648`.

## Decision

Promote `native_hidden64_target_area_star_only_vec4_wt` as the preferred
full-support native target-area star-only mode for this lane. This is a speed
promotion, not a quality promotion. Keep canonical
`native_hidden64_target_area_star_only` as the fallback/parity reference.

Report:
`outputs/benchmarks/2026-05-20_star_uvt_native_target_area_vec4_wt_trainer_ab_gate.md`
