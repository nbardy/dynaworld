# fast-mac-gsplat v3 saved-order ablation

## Context

After v3 beat v2 despite re-sorting tile IDs in fast backward, we tested two
small follow-up optimizations carefully:

1. avoid cloning the full 4K RGB gradient image when there are no overflow
   tiles,
2. save v3 forward's tile-local sorted IDs back into `binned_ids` so fast
   backward can reuse the order instead of sorting again.

## Baseline

Sequential current-v3 baseline, 4096x4096 / 65,536 projected splats,
`--warmup 2 --iters 5`:

```text
forward:
sparse sigma 1-5 px:  v3 11.480 ms
medium sigma 3-8 px:  v3 11.927 ms

forward+backward:
sparse sigma 1-5 px:  v3 52.460 ms
medium sigma 3-8 px:  v3 65.687 ms
```

## Clone-only ablation

Patch: use `grad_out.contiguous()` directly on the common no-overflow path and
clone only when overflow tiles must be zeroed.

```text
forward+backward:
sparse sigma 1-5 px:  52.904 ms  (+0.444 ms / +0.8%)
medium sigma 3-8 px:  63.878 ms  (-1.809 ms / -2.8%)
```

Interpretation: small/noisy. It is still worth keeping because it avoids a
large unnecessary 4K RGB tensor copy when no overflow exists.

## Saved sorted ID ablation

Patch: v3 fast forward writes its sorted tile-local `shared_ids` back into
`binned_ids`; v3 fast backward loads that saved order and skips
`bitonic_sort_ids(...)`.

Combined result with the clone fix:

```text
forward:
sparse sigma 1-5 px:  12.410 ms  (+0.930 ms / +8.1% vs baseline)
medium sigma 3-8 px:  13.702 ms  (+1.775 ms / +14.9% vs baseline)

forward+backward:
sparse sigma 1-5 px:  47.872 ms  (-4.588 ms / -8.7% vs baseline)
medium sigma 3-8 px:  60.738 ms  (-4.949 ms / -7.5% vs baseline)
```

Estimated backward-only by subtracting separate forward timings:

```text
baseline sparse backward: 40.980 ms
final sparse backward:    35.462 ms  (-5.518 ms / -13.5%)

baseline medium backward: 53.760 ms
final medium backward:    47.036 ms  (-6.724 ms / -12.5%)
```

Interpretation: saved sorted IDs are the real speedup. The forward writeback is
not free, but the backward sort removal more than pays for it in training mode.

## Tile stats

The two benchmark cases had no overflow, so the fast path stayed active:

```text
sparse_sigma_1_5:
  total_pairs=273547, max_tile_count=15, mean_tile_count=4.174,
  nonzero_tiles=64485, overflow_tiles=0

medium_sigma_3_8:
  total_pairs=552285, max_tile_count=25, mean_tile_count=8.427,
  nonzero_tiles=65520, overflow_tiles=0
```

## Correctness

After both changes:

```text
v3 reference_check:
image max error: 5.960464477539063e-08
means grad max error: 2.4010660126805305e-10
conics grad max error: 9.313225746154785e-10
colors grad max error: 9.313225746154785e-10
opacities grad max error: 1.862645149230957e-09

v2/v3 small MPS parity:
v2_v3 image max 0.0 mean 0.0
```

## Commit shape

The source-of-truth doc in the submodule is
`third_party/fast-mac-gsplat/docs/v3_saved_order_ablation.md`.
