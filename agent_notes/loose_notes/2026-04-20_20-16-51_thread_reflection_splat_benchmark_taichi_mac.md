# Thread Reflection: Splat Benchmark, Taichi Mac, Sorting

## Context

This work started as a request to benchmark differentiable splat rendering
across multiple renderers and evolved into a broader investigation of Mac
rendering paths, Taichi portability, raw Metal experiments, sort bottlenecks,
and diagnostic strategy.

The most important theme was a repeated narrowing of the bottleneck:

```text
renderer feels slow
-> benchmark renderer variants
-> separate forward/backward and skip unsupported backends explicitly
-> test Taichi/Metal viability
-> distinguish "can run" from "fast"
-> measure K = tile-overlap keys, not only G = splats
```

## What Got Done

### Benchmark Harness

Implemented and iterated a renderer benchmark under `src/benchmarks/` with JSONC
configs under `src/benchmark_configs/`.

Capabilities added across the thread:

- random baked splat sets reused across renderers
- combinatorial sweeps over resolution and splat count
- forward and backward timing
- warmup/timed iteration controls
- renderer skip reporting
- summary tables including skipped backends
- optional render image saving
- accuracy benchmark for small packed 2D splats against a stable Torch baseline
- large-splat throughput configs

Renderers integrated or accounted for:

- current custom dense/tiled renderer
- copied vectorized sparse rasterizer
- copied memory-efficient sparse rasterizer
- Taichi Splatting fork/path
- gsplat when CUDA is available
- raw Metal/MLX bridge for forward-only throughput

The memory-efficient sparse rasterizer was explicitly removed from defaults
because it was much slower on MPS despite its name.

### Taichi Splatting Mac Path

Cloned/forked Taichi Splatting and made a Mac-usable experimental fork:

```text
https://github.com/nbardy/taichi-gsplat-differentiable-render-mac
```

Submodule path:

```text
third_party/taichi-splatting
```

Committed/pushed fork work included:

- non-CUDA sort/cumsum fallback paths
- `metal_reference` raster variant avoiding CUDA-only Taichi SIMT/warp pieces
- `pixel_reference` backward path for Metal
- experimental Taichi-side sort modes:
  - global Taichi sort
  - compact bucket sort
  - ordered tile mapper for already-depth-sorted inputs
- README updated to say:

```text
Taichi Mac Splatting

Fast splatting on the Mac integrated with Taichi.
```

Parent repo branch:

```text
codex/register-taichi-mac-submodule
```

Parent PR:

```text
https://github.com/nbardy/dynaworld/pull/1
```

### Accuracy

Small packed-2D accuracy checks passed for Taichi/Metal against a direct
CPU-float64 Torch baseline.

Important result:

- forward was clean
- packed Gaussian gradients were clean
- feature gradients were clean
- this validated the simple Metal reference raster/backward math

Scope:
    This validated 2D raster math, not full 3D projection correctness or high
    performance.

### Throughput Observations

Taichi/Metal reference became usable and was not terrible at high splat counts,
but not fast enough for 60 FPS at the target scale.

Representative forward-only observations from this thread:

- 128x128, G=65536: roughly mid-20ms after warmup in one probe
- 1024x1024, G=65536: roughly mid-30ms after warmup in one probe

Older scale tests showed higher counts moving into lower FPS:

- 131k and 262k splats were not 60 FPS
- 4K was not solved

The realistic conclusion is that a browser/game renderer doing static forward
rendering is not the same as this differentiable PyTorch/Taichi/Metal stack.
Backward and framework synchronization make the comparison harsher.

### Raw Metal / MLX Direction

A separate raw Metal/MLX experiment existed and was integrated into benchmarks
as a forward-only renderer bridge. The user explicitly said not to touch the raw
Metal shader while exploring Taichi sorting.

Important raw Metal facts preserved:

- SnugBox-style tight bounds
- exact tile/ellipse test
- global depth sort and pair sort still present
- per-tile forward/backward recurrence in Metal kernels
- depth gradients intentionally zero because sort order is piecewise constant

### Sorting Experiments

Taichi-side sort experiments were added but did not beat the reference path:

- `taichi_field`: correct global sort, too slow
- `bucket_taichi`: correct compact per-tile bucket sort, still slower than
  Torch/MPS sort reference
- `ordered_taichi`: correct only if inputs are already depth sorted, but naive
  tile-major scan is too expensive

Current model:
    Moving sort "to Taichi/Metal" is not enough. The speed path needs fused
    tile-local sort+raster in threadgroup/local memory, or a lower-level native
    Metal/MLX/C++ extension.

### Overlap-Key Diagnostics

Added `K` diagnostics so the benchmark can measure actual sort-key pressure:

- `total_overlap_keys`
- `K/G`
- p95/max tiles per splat
- p95/max splats per tile
- large splat count/fraction
- exact-vs-Taichi and exact-vs-custom ratios

Query variants:

- `custom_rect`
- `taichi_obb`
- `exact_conic`
- `exact_conic_custom_tile` when custom tile size differs

Key result:

For random splats, Taichi's OBB query is already close to exact:

- 64x64/G=1024: Taichi/exact = 1.090x
- 128x128/G=65536: Taichi/exact = 1.093x
- 1024x1024/G=65536: Taichi/exact = 1.099x

Custom rectangular bounds were worse, but after correcting for tile-size mismatch
the checked same-tile ratio was about 1.29x, not the initially misleading 2.8x.

### Training Diagnostics

Extended `debug_metrics.py` so real training runs can log overlap pressure under
the existing `logging.with_metrics.renderer` flow.

This matters because random splats may be too clean. The next decisive question
is whether trained Dynaworld outputs create large transparent floaters that make
`K/G` much worse than random cases.

### Documentation / Notes

Wrote several loose notes and key learnings across the thread. Most relevant:

- `agent_notes/loose_notes/2026-04-20_19-48-41_taichi_mac_fork_submodule.md`
- `agent_notes/loose_notes/2026-04-20_20-10-59_overlap_key_pressure_diagnostics.md`
- `agent_notes/loose_notes/2026-04-20_20-15-23_overlap_sort_reflection.md`

Key learnings were updated with the important surprise:

```text
Random-splat sort pressure is not currently dominated by Taichi's OBB tile test.
```

## What Did Not Get Done

### We Did Not Make Taichi Fast Enough

The Taichi Mac path works as an experimental reference, but it is not the final
high-performance Mac renderer.

Specifically not done:

- no fused tile-local sort+raster kernel
- no threadgroup-memory bitonic sort integrated into raster
- no true high-performance Metal-native backward
- no low-precision optimized kernels
- no per-splat backward memory optimization
- no production-quality macOS upstream-ready Taichi patch

### We Did Not Implement True AccuTile In The Mapper

We added exact-conic telemetry, not a replacement mapper.

Reason:
    The telemetry showed exact intersection only saved about 8-10% keys for
    random Taichi cases. That is useful but probably not enough to explain the
    speed gap.

### We Did Not Prove Real Training Outputs Are Clean

This is the largest remaining unknown.

Random benchmarks showed modest `K/G`, but trained models could still create
floaters. Until we run training with `TileDiag/*`, we do not know whether
pruning/floater control is urgent.

### We Did Not Commit The Latest Overlap-Diagnostic Work

The Taichi fork/submodule work was committed and pushed. The latest overlap
diagnostics, benchmark config changes, and reflection notes are still local
working-tree changes at the time of this note.

Also, the repo has many unrelated dirty/untracked files from broader session
work. Future commits should stage narrowly.

## Backtracks / Corrections

### "Taichi can just use Metal backend"

Weakened.

Taichi can target Metal for some kernels, but the existing Taichi Splatting fast
path depended on CUDA-specific helpers and SIMT/block/warp assumptions. A Mac
path required separate reference kernels and sort/cumsum fallbacks.

### "Bucket sort should make sorting fast"

Weakened.

Bucket sort as an architecture can be right, but our Taichi implementation used
separate global-memory stages. That did not capture the key performance property:
fusing local sort and raster in threadgroup memory.

### "Exact tile intersection may fix the sort bottleneck"

Weakened for random splats.

The exact query saved only ~8-10% keys against Taichi OBB in random probes.
It remains plausible for real trained outputs if floaters appear.

### "Custom tiled is much worse by 2.8x"

Corrected.

The 2.8x ratio mixed 8px custom tiles with 16px exact tiles. Same tile size was
about 1.29x in the checked probe.

## Current Best Next Step

Run a short real training probe with renderer diagnostics enabled.

Suggested JSONC addition:

```jsonc
"logging": {
  "with_metrics": {
    "renderer": true,
    "optimizer": false,
    "every": 25,
    "print_summary": true,
    "wandb": true,
    "fail_fast": true
  }
}
```

Watch:

- `TileDiag/ExactConic_duplication_factor_mean`
- `TileDiag/ExactConic_max_tiles_per_splat_max`
- `TileDiag/ExactConic_max_splats_per_tile_max`
- `TileDiag/CustomRectToExactKeyRatio`

Interpretation:

- If trained `K/G` is low, pursue fused raw Metal/MLX/native renderer work.
- If trained `K/G` is high, pursue pruning/floater/projected-support control.
- If exact saves >25-30% same-tile keys on trained outputs, port exact
  conic/AccuTile-style mapper logic.

## Final Assessment

The thread produced a useful benchmark suite, a working Taichi/Mac reference
fork, accuracy validation, raw Metal integration scaffolding, and most
importantly a diagnostic for the actual sort workload.

It did not produce a final 60 FPS differentiable Mac renderer. But it did reduce
the uncertainty around why the current path is slow. The current strongest model
is:

```text
Random splats are not key-count pathological.
Taichi's current OBB culling is close to exact.
The next renderer speed step is likely fusion/lower-level Metal, unless trained
outputs show floater-driven K/G pathologies.
```
