# Fast-mac TokenGS/free-splats phase and precision probe

## Why

User asked to get back to speed after the multicamera notes: compare current
TokenGS/free-splats speed, split fast-mac forward/backward phases, and check
whether low precision between model/projection/render is already usable or needs
a renderer fork.

## New artifact

- Added profiler:
  `research_experiments/vjepa_performance/profile_fast_mac_render_phases.py`
- Compile check:
  `PYTHONPATH=src/train uv run python -m py_compile research_experiments/vjepa_performance/profile_fast_mac_render_phases.py`
- Smoke:
  `PYTHONPATH=src/train WANDB_MODE=disabled uv run python research_experiments/vjepa_performance/profile_fast_mac_render_phases.py --config src/train_configs/local_mac_compare_unconditioned_tokens_16f_implicit_camera_128_fast_mac_8192splats.jsonc --render-size 128 --clip-length 16 --splat-count 8192 --steps 1 --warmup 0 --output-jsonl outputs/benchmarks/fast_mac_phase_profile_tokengs_smoke_128px_16f_8192splats_2026-05-02.jsonl`

The profiler adds extra standalone projection/raster/backward probes inside a
profiled iteration, so its phase means are for attribution only. Do not sum all
phase means and call that training throughput.

## Actual throughput already on disk

Current real training-throughput artifacts:

- `outputs/benchmarks/free_splats_throughput_splat_counts_2048_4096_8192_2026-05-02.jsonl`
- `outputs/benchmarks/free_splats_throughput_128px_8192splats_rerun_2026-05-02.jsonl`
- `outputs/benchmarks/token_gs_unconditioned_throughput_128px_8192splats_50step_2026-05-02.jsonl`
- `outputs/benchmarks/token_gs_unconditioned_throughput_64_256px_8192splats_50step_2026-05-02.jsonl`

Selected 8192-splat, 16-frame cases:

| Variant | Render | Steps/s | Frames/s | ms/frame |
| --- | ---: | ---: | ---: | ---: |
| free_splats | 64 | 6.872 | 109.9 | 9.10 |
| free_splats | 128 | 8.250 | 132.0 | 7.58 |
| free_splats | 256 | 5.814 | 93.0 | 10.75 |
| unconditioned_tokens | 64 | 7.434 | 118.9 | 8.41 |
| unconditioned_tokens | 128 | 6.779 | 108.5 | 9.22 |
| unconditioned_tokens | 256 | 4.851 | 77.6 | 12.88 |

Selected single-frame, 8192-splat cases:

| Variant | Render | Steps/s | Frames/s | ms/frame |
| --- | ---: | ---: | ---: | ---: |
| free_splats | 128 | 33.111 | 33.1 | 30.20 |
| unconditioned_tokens | 128 | 29.917 | 29.9 | 33.43 |

This is why the old "15-30 it/s" memory is plausible for single-frame or very
small frame-count cases, but not for the current 16-frame training step. For
video runs, frames/s is the cleaner comparison.

## Phase split, 128px, 16 frames, 8192 splats

Command:

```bash
PYTHONPATH=src/train WANDB_MODE=disabled uv run python \
  research_experiments/vjepa_performance/profile_fast_mac_render_phases.py \
  --render-size 128 --clip-length 16 --splat-count 8192 --steps 5 --warmup 2 \
  --output-jsonl outputs/benchmarks/fast_mac_phase_profile_token_vs_free_128px_16f_8192splats_2026-05-02.jsonl
```

Mean ms:

| Variant | forward_decode | project | raster fwd | raster fwd grad | raster bwd projected | objective render fwd | recon loss | full bwd to model |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| free_splats | 94.8 | 34.9 | 7.4 | 7.1 | 9.4 | 28.8 | 7.2 | 115.2 |
| unconditioned_tokens | 129.1 | 31.3 | 6.7 | 7.9 | 10.6 | 27.5 | 3.9 | 178.5 |

Interpretation: projected raster forward/backward is not the dominant wall at
this size. Projection is material at about 31-35 ms. The big cost is full
backward through the rendered objective into model parameters.

## Model autocast probe

Command:

```bash
PYTHONPATH=src/train WANDB_MODE=disabled uv run python \
  research_experiments/vjepa_performance/profile_fast_mac_render_phases.py \
  --config src/train_configs/local_mac_compare_unconditioned_tokens_16f_implicit_camera_128_fast_mac_8192splats.jsonc \
  --render-size 128 --clip-length 16 --splat-count 8192 \
  --amp-mode off --amp-mode fp16 --amp-mode bf16 --steps 5 --warmup 2 \
  --output-jsonl outputs/benchmarks/fast_mac_phase_profile_tokengs_amp_modes_128px_16f_8192splats_2026-05-02.jsonl
```

Mean ms:

| AMP | forward_decode | project | raster fwd | raster bwd projected | objective render fwd | full bwd to model |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 131.3 | 36.8 | 8.7 | 11.5 | 29.6 | 204.6 |
| fp16 | 186.4 | 54.3 | 7.5 | 11.3 | 26.6 | 313.4 |
| bf16 | 173.2 | 57.7 | 6.6 | 12.9 | 35.8 | 301.8 |

On this MPS path, model autocast alone was slower. It also does not make the
raster inputs low precision because the wrapper and bridge force float32.

## Low-precision render boundary

Current wrapper behavior:

- `src/train/renderers/fast_mac.py` casts all Gaussian tensors to `.float()` in
  both single and batch fast-mac render paths before projection/raster.
- `third_party/fast-mac-gsplat/variants/v5/torch_gsplat_bridge_v5/rasterize.py`
  rejects any non-float32 projected tensor.
- `third_party/fast-mac-gsplat/variants/v5_features/torch_gsplat_bridge_v5_features/rasterize.py`
  has the same float32 check.
- Both v5 and v5_features C++/Metal bridge checks also assert float32 for
  means2d, conics, colors/features, and opacities.

Profiler acceptance probe result for both free_splats and unconditioned_tokens:

```text
float16  -> rejected: ValueError: means2d must be float32
bfloat16 -> rejected: ValueError: means2d must be float32
```

Conclusion: a real low-precision render experiment is not a config flip. It
requires a forked fast-mac variant that changes Python checks, C++ checks,
Metal buffer interpretation, and probably accumulation policy. Given projected
raster fwd+bwd is about 18-19 ms while full model backward is 115-205 ms here,
that fork is not the first speed lever unless larger resolutions/splat counts
move the bottleneck.

## Followups

1. Use the throughput harness, not the phase profiler, for headline it/s and
   frames/s.
2. If optimizing next without renderer surgery, focus on backward-to-model and
   projection:
   - smaller conditioning tensors,
   - chunking/backward strategy,
   - fewer train frames per optimization step with frame accumulation,
   - possible projection kernel changes.
3. Only fork fast-mac low precision after measuring a regime where projected
   raster fwd/bwd dominates total step time.

## Single-frame high-resolution followup

User asked what happens if the same single-frame 8192-splat setup goes from
128px to 512px, 2k, and 4k. The existing throughput harness couples
`model.size` and `render.render_size`, so these rows measure the full training
step at that target size: frame sampling/resize, render, pixel loss, backward,
and optimizer. This is not a render-only upscale from a fixed 128px model input.

Commands:

```bash
PYTHONPATH=src/train WANDB_MODE=disabled uv run python \
  research_experiments/vjepa_performance/benchmark_free_splats_throughput.py \
  --config src/train_configs/local_mac_compare_free_splats_16f_implicit_camera_128_fast_mac_8192splats.jsonc \
  --render-sizes 512,2048,4096 --clip-lengths 1 --splat-counts 8192 \
  --steps 10 --warmup 2 \
  --output-jsonl outputs/benchmarks/free_splats_single_frame_512_2k_4k_8192splats_2026-05-02.jsonl

PYTHONPATH=src/train WANDB_MODE=disabled uv run python \
  research_experiments/vjepa_performance/benchmark_free_splats_throughput.py \
  --config src/train_configs/local_mac_compare_unconditioned_tokens_16f_implicit_camera_128_fast_mac_8192splats.jsonc \
  --render-sizes 512,2048,4096 --clip-lengths 1 --splat-counts 8192 \
  --steps 10 --warmup 2 \
  --output-jsonl outputs/benchmarks/token_gs_unconditioned_single_frame_512_2k_4k_8192splats_2026-05-02.jsonl
```

Combined with the earlier 128px rows:

| Variant | Size | Steps/s | ms/frame | render ms | loss ms | backward ms | sample ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| free_splats | 128 | 33.111 | 30.2 | 6.0 | 0.8 | 13.7 | 0.6 |
| free_splats | 512 | 16.892 | 59.2 | 11.3 | 4.1 | 30.0 | 1.4 |
| free_splats | 2048 | 2.737 | 365.3 | 19.4 | 64.9 | 257.8 | 5.1 |
| free_splats | 4096 | 0.400 | 2499.1 | 51.9 | 711.9 | 833.0 | 880.2 |
| unconditioned_tokens | 128 | 29.917 | 33.4 | 7.7 | 1.0 | 17.6 | 0.7 |
| unconditioned_tokens | 512 | 20.581 | 48.6 | 8.1 | 3.8 | 26.6 | 1.3 |
| unconditioned_tokens | 2048 | 2.411 | 414.7 | 25.8 | 68.6 | 295.3 | 5.4 |
| unconditioned_tokens | 4096 | 0.281 | 3556.2 | 459.3 | 905.4 | 917.1 | 1232.9 |

Interpretation:

- 512px is still trainable locally: roughly 17-21 steps/s for single-frame
  8192-splat fitting.
- 2k is slow but usable for probes: roughly 2.4-2.7 steps/s. The wall is mostly
  pixel loss and backward, not render alone.
- 4k is not a practical local training target with this harness. It drops below
  0.5 steps/s, with huge frame sampling/resize and image-loss/backward costs.
  Free-splats render itself is only about 52ms, but the full step is about
  2.5s. TokenGS saw a noisy 4k render/composition spike and a 3.6s full step.

## Rasterizer version and 4k backward suspicion

After the high-res numbers, user asked what rasterizer version the trainer is
actually using because backward looked too slow.

Verified current dispatch:

- `src/train/renderers/fast_mac.py` hard-codes `FAST_MAC_V5_DIR` and
  `FAST_MAC_V5_FEATURES_DIR`.
- RGB `F == 3` dispatches to
  `third_party/fast-mac-gsplat/variants/v5/torch_gsplat_bridge_v5`.
- Feature splatting `F != 3` dispatches to
  `third_party/fast-mac-gsplat/variants/v5_features/torch_gsplat_bridge_v5_features`.
- The current free-splats and unconditioned-token configs decode RGB
  `feature_dim=3`, so the single-frame 8192-splat rows above used v5, not
  v5_features and not any v6/v8/v9 fork.

Live import probe:

```text
local_mac_compare_free_splats_16f_implicit_camera_128_fast_mac_8192splats.jsonc variant= free_splats feature_dim= 3 renderer= fast_mac
local_mac_compare_unconditioned_tokens_16f_implicit_camera_128_fast_mac_8192splats.jsonc variant= unconditioned_tokens feature_dim= 3 renderer= fast_mac
v5 module: /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/v5/torch_gsplat_bridge_v5/__init__.py
v5_features module: /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/v5_features/torch_gsplat_bridge_v5_features/__init__.py
```

To separate the full-step `backward` bucket from raster backward, ran:

```bash
PYTHONPATH=src/train WANDB_MODE=disabled uv run python \
  research_experiments/vjepa_performance/profile_fast_mac_render_phases.py \
  --config src/train_configs/local_mac_compare_free_splats_16f_implicit_camera_128_fast_mac_8192splats.jsonc \
  --render-size 4096 --clip-length 1 --splat-count 8192 --steps 3 --warmup 1 \
  --output-jsonl outputs/benchmarks/fast_mac_phase_profile_free_splats_single_frame_4k_8192splats_2026-05-02.jsonl
```

4k free-splats / v5 phase means and medians:

| Phase | Mean ms | Median ms |
| --- | ---: | ---: |
| fastmac_project_forward | 41.9 | 40.7 |
| fastmac_raster_forward_projected | 66.0 | 59.5 |
| fastmac_raster_forward_projected_grad | 304.5 | 170.4 |
| fastmac_raster_backward_projected | 483.0 | 180.7 |
| objective_render_forward | 109.0 | 57.9 |
| recon_loss_compute | 750.3 | 701.1 |
| full_backward_render_to_model | 915.7 | 836.9 |

This supports the user's suspicion enough to make the next goal a real
rasterizer audit. The full training backward is still dominated by image loss
and downstream model/projection work, but the isolated v5 projected raster
backward at 4k is much worse than the remembered target (`~2x forward`, and
roughly `30ms`-class backward for larger point counts). The immediate next
thread should compare v5 against the newer v6/v8/v9 trainable variants at fixed
projected inputs, fixed point counts, and fixed high resolutions before changing
trainer code.
