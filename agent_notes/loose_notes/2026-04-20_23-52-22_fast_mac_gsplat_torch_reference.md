# fast-mac-gsplat Torch reference comparison

## Context

After the v3 saved-order optimization, we added a direct Torch reference
comparison so README speed claims can say more than just v2 vs v3.

## What changed

- Added `--include-torch-reference` to
  `third_party/fast-mac-gsplat/benchmarks/compare_v2_v3.py`.
- Added a direct Torch reference renderer that sorts splats by detached depth,
  loops over splats, and uses vectorized Torch tensor ops over the image.
- Added `--torch-max-work-items` so the direct Torch reference is skipped when
  `height * width * gaussians` is too large.
- Updated `third_party/fast-mac-gsplat/README.md` with small-scene Torch
  comparison numbers.
- Added
  `third_party/fast-mac-gsplat/docs/torch_reference_comparison.md`.

## Results

128x128 / 512 splats, `--warmup 1 --iters 3`, with active background CPU jobs:

```text
forward:
sparse sigma 1-5 px:  v2 4.654 ms, v3 7.443 ms, torch 163.244 ms
medium sigma 3-8 px:  v2 4.666 ms, v3 6.114 ms, torch 162.016 ms

forward+backward:
sparse sigma 1-5 px:  v2 7.972 ms, v3 8.940 ms, torch 828.056 ms
medium sigma 3-8 px:  v2 10.850 ms, v3 11.928 ms, torch 866.864 ms
```

Interpretation:

- Direct Torch is already much slower at small scale: roughly `22-27x` slower
  than v3 forward and `73-93x` slower than v3 forward+backward.
- v2 beats v3 at tiny 128x128 / 512-splat sizes because v3 has extra
  staging/reduction overhead. This does not contradict the 4K / 65,536-splat v3
  win.
- A direct Torch comparison at 4K / 65,536 splats is not meaningful; it would
  imply about `1.1e12` pixel-splat evaluations.

## Caveat

There were unrelated CPU-heavy jobs running during this pass. The Torch gap is
large enough to be directionally useful, but clean headline numbers should be
rerun after the machine is quiet, with randomized renderer order and multiple
seeds.
