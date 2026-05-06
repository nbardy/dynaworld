# Free-Splats Frame Throughput Benchmark

## Context

User asked to include a single-frame case, try different resolutions, and report
frames/sec because step/sec hides clip-length batching.

I added:

```text
research_experiments/vjepa_performance/benchmark_free_splats_throughput.py
```

It benchmarks the direct `free_splats` path from:

```text
src/train_configs/local_mac_compare_free_splats_16f_implicit_camera_128_fast_mac_8192splats.jsonc
```

The harness times:

- sample clip
- model input
- forward/decode
- regularizers
- background sampling
- render/compose
- reconstruction loss compute
- backward
- optimizer step

It reports both `steps/s` and `frames/s`.

Compile gate:

```bash
PYTHONPATH=src/train uv run python -m py_compile \
  research_experiments/vjepa_performance/benchmark_free_splats_throughput.py
```

## Commands

Short matrix including true single-source-frame:

```bash
PYTHONPATH=src/train WANDB_MODE=disabled uv run python \
  research_experiments/vjepa_performance/benchmark_free_splats_throughput.py \
  --render-sizes 64,128,256 \
  --clip-lengths 1,4,16 \
  --steps 10 \
  --warmup 3 \
  --include-single-source-frame \
  --output-jsonl outputs/benchmarks/free_splats_throughput_2026-05-02.jsonl
```

Stabilized 50-step 128px run:

```bash
PYTHONPATH=src/train WANDB_MODE=disabled uv run python \
  research_experiments/vjepa_performance/benchmark_free_splats_throughput.py \
  --render-sizes 128 \
  --clip-lengths 1,4,16 \
  --steps 50 \
  --warmup 5 \
  --include-single-source-frame \
  --output-jsonl outputs/benchmarks/free_splats_throughput_128px_50step_2026-05-02.jsonl
```

Stabilized 50-step 64px/256px run:

```bash
PYTHONPATH=src/train WANDB_MODE=disabled uv run python \
  research_experiments/vjepa_performance/benchmark_free_splats_throughput.py \
  --render-sizes 64,256 \
  --clip-lengths 1,4,16 \
  --steps 50 \
  --warmup 5 \
  --output-jsonl outputs/benchmarks/free_splats_throughput_64_256px_50step_2026-05-02.jsonl
```

## 50-Step Results

All rows are local MPS, fast-mac, 8192 splats, direct per-frame Gaussian params.

| Source frames | Render size | Clip frames/step | Steps/s | Frames/s | ms/frame |
|---:|---:|---:|---:|---:|---:|
| all | 64 | 1 | 24.76 | 24.76 | 40.39 |
| all | 64 | 4 | 12.45 | 49.79 | 20.09 |
| all | 64 | 16 | 4.28 | 68.48 | 14.60 |
| all | 128 | 1 | 26.08 | 26.08 | 38.34 |
| all | 128 | 4 | 14.19 | 56.75 | 17.62 |
| all | 128 | 16 | 3.84 | 61.41 | 16.28 |
| all | 256 | 1 | 26.50 | 26.50 | 37.73 |
| all | 256 | 4 | 10.45 | 41.78 | 23.93 |
| all | 256 | 16 | 3.09 | 49.41 | 20.24 |
| 1 | 128 | 1 | 31.82 | 31.82 | 31.42 |

## Main Takeaways

- `frames/s` is the better headline. Step/sec makes the 1-frame case look best,
  but batched clips are more efficient per rendered/trained frame.
- The current direct free-splats path does not reproduce old `15-30 it/s` at
  video clip sizes, but it does reach `~50-70 frames/s` depending on resolution
  and clip length.
- True single-source-frame training is faster than one-frame sampling from the
  46-frame bank at 128px (`31.82` vs `26.08 frames/s`) because the optimizer has
  less inactive direct-splat state.
- 64/128/256 are not cleanly pixel-bound in this path. The dominant costs at
  16-frame clips are forward/decode and backward through the direct splat/camera
  graph, not just raster image size. MPS timings still show outliers, so median
  subphase timings are better than one short run for kernel-level conclusions.
