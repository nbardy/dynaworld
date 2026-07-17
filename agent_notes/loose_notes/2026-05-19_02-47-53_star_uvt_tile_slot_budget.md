# STAR UVT fixedbin/tile-slot accumulator budget

Date: 2026-05-19

## Goal

Continue the STAR UVT fast feature-shader plan after the feature-gradient-only
/ naive two-pass split proved correct but slow. The next question was whether a
real fixedbin/tile-slot feature-gradient accumulator has a plausible work and
memory shape before writing a larger Metal fork.

## New Gate

Added:

```text
research_experiments/star_uvt_feature_tubes/tile_slot_accumulator_budget.py
```

The script uses the current forward bins from `render_uvt_feature_tubes(...,
return_bins=True)` and estimates:

- occupied tile slots
- tile count distribution and overflow
- current per-pixel/slot/channel feature-gradient atomic write count
- one-atomic-per-tile-slot/channel write count
- naive prefix-recompute multiplier
- scalar contribution-weight tape memory
- wrong per-channel contribution-weight tape memory
- dense feature-image memory for comparison

This is a feasibility gate, not a new trainer mode.

## Command

```bash
TMPDIR=/Users/nicholasbardy/git/gsplats_browser/dynaworld/.tmp \
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  .venv/bin/python research_experiments/star_uvt_feature_tubes/tile_slot_accumulator_budget.py \
  --sizes 128,256,512 --frames 64 --tubes 32768 --feature-dim 32 \
  --warmup 1 --repeat 3 \
  --out-dir outputs/benchmarks/2026-05-19_star_uvt_feature_tile_slot_budget_128_256_512_64f_32768t_f32
```

Artifact:

```text
outputs/benchmarks/2026-05-19_star_uvt_feature_tile_slot_budget_128_256_512_64f_32768t_f32/summary.md
```

Validation:

- `py_compile` passed for `tile_slot_accumulator_budget.py`.
- Artifact sanity passed: 3 rows, sizes `{128,256,512}`, finite outputs, memory
  columns present.

## Results

| size | occupied slots | p95 slots | max slots | overflow sum | direct feature atomics | tile-slot atomics | prefix recompute | scalar f32 tape | per-channel f32 tape |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1,047,276 | 128 | 128 | 8093 | 4.290B | 33.513M | 64.4x | 0.499GiB | 15.98GiB |
| 256 | 2,455,395 | 98 | 118 | 0 | 10.057B | 78.573M | 39.8x | 1.171GiB | 37.47GiB |
| 512 | 2,505,068 | 28 | 43 | 0 | 10.261B | 80.162M | 10.8x | 1.195GiB | 38.22GiB |

Dense feature-image f32 memory for comparison:

- 128px: `0.125GiB`
- 256px: `0.500GiB`
- 512px: `2.000GiB`

## Interpretation

The good news: a tile-slot feature-gradient accumulator has a real target. It
can theoretically reduce feature-gradient write count by `128x` because it
moves from per-pixel/slot/channel atomics to one atomic per tile slot and
channel.

The bad news: the obvious implementation is wrong in two ways.

1. Recomputing transmittance prefixes per slot is too expensive. The multiplier
   is `39.8x` at 256px and still `10.8x` at 512px.
2. Storing per-channel contribution weights is impossible at the target:
   `37-38GiB` at 256/512px.

The plausible design is narrower:

- store or derive scalar contribution weights/prefixes only, not per-channel
  weights
- keep the scalar tape chunkable and probably f16/compressed if it is stored
- reduce channels from the scalar weights and `grad_feature_image`
- or avoid the tape entirely with a native image-space VJP/handoff

The scalar f32 tape is around `1.2GiB` at 256/512px before chunking/f16. That is
large but in the range of the existing dense 512px feature image (`2.0GiB`), so
it is a possible benchmark fork, not obviously impossible.

## Plan Forward

1. Do not implement a per-channel tile-slot tape.
2. Do not implement a prefix-recompute-per-slot kernel.
3. Prototype only a scalar contribution-weight/prefix tape, ideally chunked or
   f16, then reduce feature channels from that scalar tape.
4. Continue treating native image-space VJP/handoff as the parallel path,
   because it may remove both dense F32 image-gradient backprop and the scalar
   tape.
