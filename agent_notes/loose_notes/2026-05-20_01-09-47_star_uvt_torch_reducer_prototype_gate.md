# STAR UVT Torch Reducer Prototype Gate

Date: 2026-05-20 01:09 +07

## Goal

After the colorizer atomic split pinned compact native slowdown on colorizer
parameter-gradient atomics, test a cheaper sidecar reducer before committing to
another native ABI.

## Prototype

Added `--include-torch-reducer-prototype` to
`research_experiments/star_uvt_feature_tubes/sparse_hidden_target_area_kernel_benchmark.py`.

The prototype does:

1. sparse native feature/alpha render for the compact pixels;
2. Torch/MPS hidden64 target-area loss plus hidden/output colorizer parameter
   gradients;
3. native target-area star-only vec4 W^T backward using the cell RGB gradients.

It compares the prototype colorizer and STAR gradients against the existing
native full-colorizer path.

## Results

Tiny gate:

- output JSON:
  `outputs/benchmarks/2026-05-20_star_uvt_native_target_area_torch_reducer_prototype_tiny_gate.json`
- pass true
- prototype colorizer max error `2.14e-08`
- prototype STAR max error `1.16e-09`

Compact `64f/512px/8192t`, `6.25%` support, repeat-3/warmup-1:

- without baseline:
  - native full colorizer atomics: `598.74ms` total, `559.63ms` backward
  - prototype: `434.89ms` total, `251.11ms` reducer, `127.35ms` native
    star-only backward
  - max errors: colorizer `1.47e-08`, STAR `2.76e-10`
- with baseline enabled:
  - sparse-pixel baseline: `276.56ms` total (`11.45ms` render, `238.11ms`
    loss/reduce, `27.00ms` backward)
  - native full colorizer atomics: `752.83ms` total
  - prototype: `390.93ms` total (`21.21ms` sparse render, `230.58ms`
    reducer, `139.14ms` native star-only backward)
  - max errors: colorizer `1.26e-08`, STAR `2.62e-10`

Report:
`outputs/benchmarks/2026-05-20_star_uvt_native_target_area_torch_reducer_prototype_gate.md`

## Decision

Correct but rejected as a keeper route. The sidecar reducer beats naive native
atomics but still loses to the existing sparse-pixel baseline. The duplicate
sparse render plus target-area hidden replay erases the atomic win.

Next native work only makes sense if it reduces colorizer parameters inside the
same pass, for example with per-threadgroup partials. Otherwise, keep compact
autograd/sparse-pixel visual gradients and return to the visual objective
quality blocker.
