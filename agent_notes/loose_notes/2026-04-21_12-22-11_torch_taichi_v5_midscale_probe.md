# Torch / Taichi / v5 Midscale Probe

## Context

User asked why direct Torch was skipped in 4K/64K, whether it OOMs, what size
Torch can handle, and whether we can simplify comparison to only Torch, Taichi,
and fast-mac v5.

Benchmark harness update:

- `src/benchmarks/mac_renderer_stack_compare.py` now supports
  `--renderers torch,taichi,v5`.

## Why Torch Was Skipped At 4K/64K

The benchmark skipped direct Torch because the configured work cap is
`64,000,000` pixel-splat evaluations, while 4K/64K is:

```text
4096 * 4096 * 65536 = 1,099,511,627,776 evaluations
```

That skip is a guardrail, not proof that forward immediately OOMs. The direct
Torch reference is a loop over splats and only materializes image-sized tensors
per splat, so forward can sometimes run much larger than a fully vectorized
`G*H*W` dense implementation. But it is far too slow for the key case, and
backward keeps enough autograd state that it hits MPS memory limits quickly.

## 1024x512 / 2048 Splats Forward

Command shape:

```bash
python src/benchmarks/mac_renderer_stack_compare.py \
  --height 1024 --width 512 --gaussians 2048 --batch-size 1 \
  --warmup 1 --iters 3 \
  --include-torch --check-outputs \
  --torch-max-work-items 2000000000 \
  --renderers torch,taichi,v5
```

Forward result:

| Case | Torch | Taichi | v5 | v5 vs Torch | v5 vs Taichi |
| --- | ---: | ---: | ---: | ---: | ---: |
| sparse sigma 1-5 | `2763.382 ms` | `12.609 ms` | `2.943 ms` | `938.85x` | `4.28x` |
| medium sigma 3-8 | `2920.559 ms` | `14.048 ms` | `3.085 ms` | `946.69x` | `4.55x` |

Forward accuracy against Torch stayed tight:

- sparse max diff: Taichi `2.384e-07`, v5 `1.788e-07`
- medium max diff: Taichi `2.384e-07`, v5 `2.384e-07`

## Torch Backward Capacity At 1024x512

Direct Torch forward+backward at `1024x512 / 2048 splats` OOMed:

```text
MPS backend out of memory
MPS allocated: 27.18 GiB
max allowed: 27.20 GiB
Tried to allocate 6.00 MiB
```

Capacity probes at the same resolution:

| Splats | Torch fwd+bwd status | Sparse | Medium |
| ---: | --- | ---: | ---: |
| 512 | pass | `4499.073 ms` | `6722.370 ms` |
| 768 | pass | `5225.254 ms` | `13275.219 ms` |
| 896 | pass | `6218.306 ms` in torch-only run; `9447.481 ms` in all-renderer run | `24562.823 ms` in torch-only run; `21714.489 ms` in all-renderer run |
| 1024 | OOM | n/a | n/a |
| 2048 | OOM | n/a | n/a |

Practical conclusion: at `1024x512`, the looped direct Torch reference can
handle forward at 2048 splats, but forward+backward tops out below 1024 splats
on the current MPS high-water mark. Around 768-896 splats is the upper practical
range, and it is already seconds to tens of seconds per step.

## 1024x512 / 896 Splats Forward+Backward

Command shape:

```bash
python src/benchmarks/mac_renderer_stack_compare.py \
  --height 1024 --width 512 --gaussians 896 --batch-size 1 \
  --warmup 0 --iters 1 --backward \
  --include-torch --no-check-outputs \
  --torch-max-work-items 2000000000 \
  --renderers torch,taichi,v5
```

| Case | Torch | Taichi | v5 | v5 vs Torch | v5 vs Taichi |
| --- | ---: | ---: | ---: | ---: | ---: |
| sparse sigma 1-5 | `9447.481 ms` | `1856.394 ms` | `238.593 ms` | `39.60x` | `7.78x` |
| medium sigma 3-8 | `21714.489 ms` | `413.941 ms` | `45.800 ms` | `474.12x` | `9.04x` |

## Current Belief

Direct Torch is useful as a correctness reference and a small/midscale
baseline, but not as a training renderer. Forward at `1024x512 / 2048` works
but is around 3 seconds. Backward at that size OOMs; under 1K splats is the
realistic ceiling for this looped reference on this machine.

v5 is already the right comparator for practical use: it is ~900x faster than
Torch forward in the 2K-splat midscale probe and still 40-474x faster than
Torch in the largest backward probe that Torch could run.
