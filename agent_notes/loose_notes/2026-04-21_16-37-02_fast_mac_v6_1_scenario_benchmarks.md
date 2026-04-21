# fast-mac v6.1 Scenario Benchmark Patch

## Context

The scientist pointed out that the current random projected-splat benchmark is
useful but not representative. That was correct. It is a saturated-screen
microbenchmark: nearly every tile is active, overflow is usually zero, and
stop-count pruning often replays the whole tile. It is good at catching fixed
overhead regressions like the v6 active-tile forward bug, but it should not
decide whether active scheduling helps real training.

## Code Changes

Submodule: `third_party/fast-mac-gsplat`

Touched files:

- `variants/v6/torch_gsplat_bridge_v6/rasterize.py`
- `variants/v6/benchmarks/benchmark_mps.py`
- `variants/v6/benchmarks/benchmark_matrix.py`
- `variants/v6/README.md`
- `docs/v6_field_report.md`

Implemented:

- `RasterConfig.active_policy = "off|on|auto"`
- preserved old `RasterConfig(use_active_tiles=True)` behavior by mapping it to
  active policy `on`
- optional pair-budget training chunks with `max_pairs_per_launch`; default is
  `0` because enabling it requires a pre-binning pass
- real trace replay mode:
  `--case real_trace --trace-file <pt file>`
- synthetic scenario families:
  - `microbench_uniform_random`
  - `sparse_screen`
  - `clustered_hot_tiles`
  - `layered_depth`
  - `overflow_adversarial`
- matrix benchmark knobs:
  - `--active-policies off,auto,on`
  - `--max-pairs-per-launches ...`
  - `--trace-file ...`

I did not add the trainer-side trace dump hook in the parent tree because the
training renderer files are currently dirty/untracked from other work. The v6
benchmark can replay the trace once the trainer saves tensors with these keys:
`means2d`, `conics`, `colors`, `opacities`, `depths`.

## Validation

Syntax:

```bash
python -m py_compile \
  variants/v6/torch_gsplat_bridge_v6/rasterize.py \
  variants/v6/benchmarks/benchmark_mps.py \
  variants/v6/benchmarks/benchmark_matrix.py
```

Correctness:

```bash
cd third_party/fast-mac-gsplat/variants/v6
python tests/reference_check.py
```

Result: direct and active paths still match the CPU reference:

```text
B=1/B=2 direct image max error: 5.960464477539063e-08
B=1/B=2 active image max error: 5.960464477539063e-08
grad max errors stayed around 1e-10 to 2e-09
```

Pair-budget chunk parity smoke:

```text
B=3, H=W=64, G=96
max_pairs_per_launch=0 vs 1000
output Linf: 0.0
means grad Linf: 7.28e-12
conics grad Linf: 9.31e-10
colors grad Linf: 1.16e-10
opacities grad Linf: 2.33e-10
```

## Noisy Machine Caveat

While benchmarking, unrelated long-running Python jobs were active on the
machine. Some were using heavy CPU and likely shared memory bandwidth. The 4K
absolute timings in this pass should not replace the earlier quiet-machine v6
field report.

This was visible because v5 also slowed down materially compared to the previous
baseline:

```text
v5 B=4 4K 64k forward+backward, noisy pass:
mean 363.060 ms, median 373.212 ms, fwd 64.711 ms, bwd 298.349 ms
```

Earlier quiet v5/v6 direct numbers were around 250 ms total for v5 and 229-232
ms total for v6 direct. Since both old and new variants slowed, this was an
environment issue, not automatically a renderer regression.

## 4K Policy Probe

Command family:

```bash
python benchmarks/benchmark_mps.py \
  --height 4096 --width 4096 --gaussians 65536 --batch-size 4 \
  --case microbench_uniform_random --backward --profile \
  --warmup 2 --iters 5 --active-policy <off|auto|on> --json
```

Observed under noisy load:

```text
off:  mean 376.799 ms, median 390.688 ms, fwd 87.254 ms, bwd 289.545 ms
auto: mean 344.586 ms, median 325.484 ms, fwd 72.241 ms, bwd 272.345 ms
on:   mean 454.444 ms, median 432.431 ms, fwd 196.424 ms, bwd 258.020 ms
```

Profile:

```text
active_tile_fraction ~= 0.9997
overflow_tile_count = 0
p95_pairs_per_tile = 13
mean_stop_ratio ~= 0.9997
```

Interpretation:

- forced active scheduling still has a large forward penalty on saturated random
  screens
- `active_policy=auto` correctly resolved to direct/off for this case
- absolute ms are noisy; the policy decision is the important result

Pair-budget probe:

```text
--max-pairs-per-launch 1500000 split B=4 into [2, 2]
mean 344.939 ms, median 337.003 ms, fwd 100.033 ms, bwd 244.906 ms
```

Interpretation: pair-budget chunking can reduce backward pressure but raises
forward/launch overhead. It remains an explicit knob, not a default.

## Scenario Sweep

Command:

```bash
python benchmarks/benchmark_matrix.py \
  --height 1024 --width 512 --gaussians 4096 --batch-sizes 4 \
  --strategies auto --stop-count-modes adaptive --dense-thresholds 64 \
  --cases microbench_uniform_random,sparse_screen,clustered_hot_tiles,layered_depth \
  --active-policies off,auto,on --max-pairs-per-launches 0 \
  --warmup 1 --iters 3 --backward
```

Findings:

```text
microbench_uniform_random:
  active fraction 1.00, overflow 0
  direct/off generally best; auto resolves direct

sparse_screen:
  active fraction ~0.15, overflow 0
  active scheduling mixed and often slower

clustered_hot_tiles:
  active fraction ~0.05, overflow ~38-40 tiles
  auto resolves active; this is the clearest active-friendly synthetic case

layered_depth:
  active fraction ~0.28, overflow 0
  stop-count pruning is meaningful, but active scheduling is mixed
```

The proposed `active_tile_fraction < 0.45` heuristic was too eager for the
current kernels. I tightened v6.1 auto to:

```text
active if active_tile_fraction < 0.10
or overflow_tile_count > 0
or max_pairs_per_tile > 2 * max_fast_pairs

reject active if active_tile_fraction > 0.75 and overflow_tile_count == 0
```

After the threshold change:

```text
sparse_screen active fraction 0.1487 -> auto resolves direct
clustered_hot_tiles active fraction 0.0507 with overflow 38 -> auto resolves active
```

## Current Model

Uniform random projected splats should stay in the suite as
`microbench_uniform_random` because it is excellent at catching fixed overhead
regressions. It should not be used as the default selector for active scheduling
or batch chunking.

Active scheduling is still not a general win. It becomes plausible when the
screen is truly clustered or overflow-heavy. Empty tiles alone are not enough,
because the active path still pays extra scheduling and background/prefill costs.

Pair-budget chunking is promising for backward pressure, but it needs a clean
quiet-machine rerun before becoming a default. The pre-binning pass and extra
launches are real costs.

## Next Falsification Tests

- Dump real Dynaworld projected traces and replay with `--case real_trace`.
- Rerun the scenario matrix on a quiet machine with seeds fixed and no other
  GPU/MPS work.
- Compare active sort on/off only for clustered overflow cases.
- Compare `max_pairs_per_launch=0,750000,1500000,3000000` at B=4 and B=8.
- Profile active forward in Xcode to isolate the background fill and scheduling
  overhead separately.
