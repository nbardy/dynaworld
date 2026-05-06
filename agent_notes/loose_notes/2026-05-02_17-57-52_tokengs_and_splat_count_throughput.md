# TokenGS and Free-Splat Count Throughput

## Context

Follow-up to the direct free-splats frame-throughput probe. User asked whether
the benchmark was saved, then asked for the fastest TokenGS splat path with the
same sizes/frame counts and for free-splats at 2k/4k sizes.

I interpreted `2k` and `4k` as total splat counts, not 2K/4K pixel resolution.

## Harness Change

Extended:

```text
research_experiments/vjepa_performance/benchmark_free_splats_throughput.py
```

with:

```text
--splat-counts 2048,4096,8192
```

The knob keeps `model.tokens` fixed and adjusts `model.gaussians_per_token`.
Counts must divide `model.tokens`.

Compile gate:

```bash
PYTHONPATH=src/train uv run python -m py_compile \
  research_experiments/vjepa_performance/benchmark_free_splats_throughput.py
```

## TokenGS Path

Used the fastest TokenGS-style control:

```text
src/train_configs/local_mac_compare_unconditioned_tokens_16f_implicit_camera_128_fast_mac_8192splats.jsonc
```

This is learned tokens + Gaussian decoder + implicit camera, with no video
encoder.

Commands:

```bash
PYTHONPATH=src/train WANDB_MODE=disabled uv run python \
  research_experiments/vjepa_performance/benchmark_free_splats_throughput.py \
  --config src/train_configs/local_mac_compare_unconditioned_tokens_16f_implicit_camera_128_fast_mac_8192splats.jsonc \
  --render-sizes 128 \
  --clip-lengths 1,4,16 \
  --splat-counts 8192 \
  --steps 50 \
  --warmup 5 \
  --output-jsonl outputs/benchmarks/token_gs_unconditioned_throughput_128px_8192splats_50step_2026-05-02.jsonl

PYTHONPATH=src/train WANDB_MODE=disabled uv run python \
  research_experiments/vjepa_performance/benchmark_free_splats_throughput.py \
  --config src/train_configs/local_mac_compare_unconditioned_tokens_16f_implicit_camera_128_fast_mac_8192splats.jsonc \
  --render-sizes 64,256 \
  --clip-lengths 1,4,16 \
  --splat-counts 8192 \
  --steps 50 \
  --warmup 5 \
  --output-jsonl outputs/benchmarks/token_gs_unconditioned_throughput_64_256px_8192splats_50step_2026-05-02.jsonl
```

## TokenGS 50-Step Results

Local MPS, fast-mac, 8192 splats:

| Size | Frames/step | Steps/s | Frames/s | ms/frame |
|---:|---:|---:|---:|---:|
| 64 | 1 | 24.34 | 24.34 | 41.08 |
| 64 | 4 | 18.44 | 73.76 | 13.56 |
| 64 | 16 | 7.43 | 118.94 | 8.41 |
| 128 | 1 | 29.92 | 29.92 | 33.43 |
| 128 | 4 | 17.58 | 70.33 | 14.22 |
| 128 | 16 | 6.78 | 108.46 | 9.22 |
| 256 | 1 | 26.01 | 26.01 | 38.44 |
| 256 | 4 | 16.36 | 65.42 | 15.28 |
| 256 | 16 | 4.85 | 77.62 | 12.88 |

So TokenGS does get close to or above `15-20 it/s` for 1-frame and 4-frame
steps. For 16-frame video steps, the correct headline is frames/sec:
`~78-119 frames/s`.

## 128px Bottleneck Snapshot

Mean milliseconds per optimization step:

| Variant | Frames/step | Forward/decode | Render+compose | Recon loss | Backward |
|---|---:|---:|---:|---:|---:|
| TokenGS | 1 | 4.31 | 7.66 | 0.96 | 17.62 |
| TokenGS | 4 | 13.33 | 9.54 | 1.31 | 30.28 |
| TokenGS | 16 | 49.03 | 18.53 | 3.41 | 74.17 |
| Free splats | 1 | 3.42 | 5.98 | 0.81 | 13.75 |
| Free splats | 4 | 10.92 | 9.34 | 1.68 | 29.41 |
| Free splats | 16 | 34.63 | 16.01 | 3.20 | 63.40 |

At 128px/16f, the bottleneck is not just rasterization. TokenGS spends roughly:

- `49 ms` in token/camera/splat decode
- `18.5 ms` in render+compose
- `74 ms` in backward

Free-splats spends less in decode/backward but still has the same general shape.
The meaningful gap at 16f is decoder/autograd overhead, not a pure fast-mac
render bottleneck.

## Free-Splats Splat-Count Sweep

Command:

```bash
PYTHONPATH=src/train WANDB_MODE=disabled uv run python \
  research_experiments/vjepa_performance/benchmark_free_splats_throughput.py \
  --config src/train_configs/local_mac_compare_free_splats_16f_implicit_camera_128_fast_mac_8192splats.jsonc \
  --render-sizes 64,128,256 \
  --clip-lengths 1,4,16 \
  --splat-counts 2048,4096,8192 \
  --steps 20 \
  --warmup 3 \
  --output-jsonl outputs/benchmarks/free_splats_throughput_splat_counts_2048_4096_8192_2026-05-02.jsonl
```

128px rows from that same 20-step sweep:

| Splats | Frames/step | Steps/s | Frames/s | ms/frame |
|---:|---:|---:|---:|---:|
| 2048 | 1 | 32.36 | 32.36 | 30.90 |
| 2048 | 4 | 18.81 | 75.23 | 13.29 |
| 2048 | 16 | 9.51 | 152.21 | 6.57 |
| 4096 | 1 | 38.11 | 38.11 | 26.24 |
| 4096 | 4 | 17.69 | 70.74 | 14.14 |
| 4096 | 16 | 9.97 | 159.54 | 6.27 |
| 8192 | 1 | 48.02 | 48.02 | 20.82 |
| 8192 | 4 | 16.40 | 65.62 | 15.24 |
| 8192 | 16 | 8.13 | 130.12 | 7.69 |

A separate 50-step 128px rerun for 8192 splats produced steadier comparable
rows:

| Splats | Frames/step | Steps/s | Frames/s | ms/frame |
|---:|---:|---:|---:|---:|
| 8192 | 1 | 33.11 | 33.11 | 30.20 |
| 8192 | 4 | 17.72 | 70.88 | 14.11 |
| 8192 | 16 | 8.25 | 132.00 | 7.58 |

Interpretation: reducing to 2k/4k splats helps most at 16-frame batches, but the
system is not simply linear in splat count. Decode, loss, MPS scheduling, and
backward overhead are significant enough that 2k/4k/8k can be close at 1f/4f.
