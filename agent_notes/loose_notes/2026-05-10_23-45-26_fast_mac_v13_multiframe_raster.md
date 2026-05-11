# Fast-Mac v13 Multiframe Raster Exploration

## Context

The prompt asked whether K-Planes/HexPlane/4DGS-style temporal sharing helps
our immediate pain point. The useful split is representation sharing versus
duplicated raster/autograd work. This session explored the second problem:
multi-frame feature splatting still pays dense per-frame raster/loss backward
costs even when splat parameters are shared.

We used three worker lanes:

- v13a: exact recompute of raster tile state in backward.
- v13b: RGB-gradient handoff scaffold to avoid dense F32 image-gradient input.
- v13c: temporal active-set diagnostics before any approximate pruning.

## Implemented

### v13a temporal recompute state

Created:

```text
third_party/fast-mac-gsplat/variants/v13a_temporal_recompute_state
research_notes/fast_mac_v13a_temporal_recompute_state.md
```

The fork is copied from `v11_features_gradcache_zero_bg_hostmeta_fixedbin` and
renamed to its own Python package/op namespace. It adds:

```python
RasterConfig.backward_state_strategy = "save" | "recompute"
```

`"save"` is the v11 behavior. `"recompute"` drops long-lived saved tile state
from autograd context and recreates it in backward by rerunning `bin` plus the
selected forward-state kernel before calling the inherited backward kernel.

Worker parity smoke found exact agreement:

```text
direct output/alpha/color-grad max diff: 0.0
active output/alpha/color-grad max diff: 0.0
```

Timing was slower, as expected for a memory valve:

```text
B=2 G=4096 F=32 H=W=128 direct save     13.353 ms
B=2 G=4096 F=32 H=W=128 direct recompute 16.026 ms
B=2 G=4096 F=32 H=W=128 active save     15.240 ms
B=2 G=4096 F=32 H=W=128 active recompute 19.437 ms
```

Main-thread benchmark at a larger small row:

```text
B=2 G=8192 F=32 H=W=128 save median      20.326 ms
B=2 G=8192 F=32 H=W=128 recompute median 31.075 ms
```

This is not a speed candidate. Keep it as an opt-in memory escape hatch until a
target-shape MPS memory trace proves the saved fixedbin state is the peak issue.

### v13b RGB-gradient handoff

Created:

```text
third_party/fast-mac-gsplat/variants/v13b_rgb_grad_handoff
research_notes/fast_mac_v13b_rgb_grad_handoff.md
```

The normal raster API is currently a v11-compatible renamed fork. The v13b
specific work adds the target API boundary:

```python
rgb_grad_handoff_backward(...)
```

and registers:

```text
torch.ops.gsplat_metal_v13b_rgb_grad_handoff.render_fast_backward_rgb_grad_handoff
```

That op intentionally raises because the Metal streaming VJP kernel is not
implemented yet. The intended kernel consumes `grad_composed_rgb[B,H,W,3]`,
applies sigmoid-linear colorizer VJP per pixel, and streams `g_feature/g_alpha`
into the inherited reverse raster contributor loop without allocating
`grad_features[B,H,W,F]`.

Bandwidth accounting:

```text
B=16 H=W=256 F=32 current dense backward input: 132 MiB
B=16 H=W=256 F=32 handoff RGB input:            12 MiB
avoided:                                       120 MiB / 90.9%

B=16 H=W=512 F=32 current dense backward input: 528 MiB
B=16 H=W=512 F=32 handoff RGB input:            48 MiB
avoided:                                       480 MiB / 90.9%
```

This remains the main speed/memory candidate, but it still needs the actual
Metal kernel plus parity against the unfused PyTorch colorizer path.

### v13c temporal active masks

Created:

```text
src/benchmarks/temporal_raster_overlap_profile.py
tests/test_temporal_raster_overlap_profile.py
research_notes/fast_mac_v13c_temporal_active_masks.md
```

The profiler currently uses synthetic projected state, not real trainer
projection tensors. It reports visible-Gaussian, active-tile, and
Gaussian-tile-pair overlap/Jaccard/retention.

Synthetic 128px result:

```text
G=256 motion=0px active tile fraction 0.9844 pair Jaccard ~0.9508
G=256 motion=8px active tile fraction 0.9844 pair Jaccard ~0.7633
G=512 motion=0px active tile fraction 1.0000 pair Jaccard ~0.9462
G=512 motion=8px active tile fraction 1.0000 pair Jaccard ~0.7504
```

Interpretation: active tiles saturate quickly in dense cases, so tile-only
temporal masks do not look promising. Pair-level masks are more plausible but
motion-sensitive; no pruning should be enabled before real projected-state
profiles, parity, and timing.

## Shared Renderer Wiring

Main thread wired these as opt-in feature variants in:

```text
src/train/renderers/fast_mac.py
```

Supported new config values:

```json
"fast_mac": {
  "feature_variant": "v13a_temporal_recompute_state",
  "backward_state_strategy": "recompute"
}
```

```json
"fast_mac": {
  "feature_variant": "v13b_rgb_grad_handoff"
}
```

`backward_state_strategy` defaults to `"save"`, so existing configs are
unchanged. v13b dispatch currently selects only the renamed v11-compatible
raster path; it does not enable the scaffolded RGB-gradient handoff kernel.

Shared-dispatch smoke:

```text
v13a_temporal_recompute_state save      loss 0.7124469876289368
v13a_temporal_recompute_state recompute loss 0.7124469876289368
v13b_rgb_grad_handoff save              loss 0.7124469876289368
```

## Baseline Measurements

Quick v11 direct-raster comparisons from this session:

```text
128px B=1 G=8192  F32 median 14.119 ms
128px B=4 G=8192  F32 median 24.299 ms
128px B=2 G=8192  F32 median 19.483 ms
128px B=1 G=16384 F32 median 15.456 ms
128px B=2 G=8192  F3  median  7.513 ms
128px B=1 G=16384 F3  median 10.586 ms

256px B=1 G=8192 F32 median 32.752 ms
256px B=2 G=8192 F32 median 37.756 ms
256px B=2 G=8192 F3  median 13.426 ms
```

Batch and feature dimension both still matter. The F32 path is the pressure
point for multiframe training, and B-vs-G is not equivalent once dense
per-frame feature gradients enter.

Late same-row direct checks after v13 integration:

```text
128px B=2 G=8192 F32 v11 median        18.736 ms
128px B=2 G=8192 F32 v13a save median  20.208 ms
128px B=2 G=8192 F32 v13b median       15.712 ms
```

Do not treat the v13b number as proof of a new kernel win: the normal v13b
raster path is intended to be v11-compatible, and this was a small noisy local
row. It does show the scaffolded fork is runnable and not obviously broken.

## Validation

Commands run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/renderers/fast_mac.py \
  src/benchmarks/temporal_raster_overlap_profile.py \
  tests/test_temporal_raster_overlap_profile.py
```

```bash
uv run --with pytest python -m pytest tests/test_temporal_raster_overlap_profile.py -q
```

Result: `1 passed`.

Shared renderer MPS smoke passed for v13a save, v13a recompute, and v13b
v11-compatible dispatch.

## Next Work

- Implement the v13b Metal kernel for
  `render_fast_backward_rgb_grad_handoff(...)`.
- Add parity against unfused colorizer/autograd for features, alpha,
  colorizer params, and splat grads.
- Add real projected-state input mode to the v13c profiler using
  `project_for_fast_mac_batch(...)` from trainer samples.
- Run target-shape memory traces for v13a recompute before using it in config.
- Do not promote v13c pruning until real profiles pass the thresholds in the
  v13c research note.

## 2026-05-11 Continuation: v13d-v13j Matrix

Added:

```text
src/benchmarks/fast_mac_v13_iteration_matrix.py
research_notes/fast_mac_v13d_to_v13j_iteration_matrix.md
```

The continuation names seven more versions without forking seven redundant
shader trees:

```text
v13d_v11_serial_batch
v13e_v11_active_auto
v13f_v11_active_on
v13g_v11_frozen_features
v13h_v13a_recompute_state
v13i_v13b_renamed_baseline
v13j_rgb_grad_accounting
```

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  src/benchmarks/fast_mac_v13_iteration_matrix.py \
  --height 128 --width 128 --gaussians 8192 --batch-size 2 \
  --feature-dim 32 --warmup 2 --iters 5
```

Results:

```text
v13d serial batch         median 30.968 ms
v13e active auto          median 20.249 ms
v13f active on            median 21.992 ms
v13g frozen features      median 27.869 ms
v13h v13a recompute       median 23.630 ms
v13i v13b renamed         median 19.316 ms
v13j RGB-grad accounting  528MiB -> 48MiB dense backward input at B16/512/F32
```

Readout:

- Serial batch is a memory fallback, not a speed path.
- Forced active scheduling loses on dense rows.
- Active auto is okay but not transformative.
- v13a recompute remains a slower exact memory valve.
- v13b is still the only structural memory/bandwidth win worth a real kernel.

## 2026-05-11 Full-Trainer Probe

After the v13d-v13j matrix, I cloned the 256px F32 multicam relpose goodset
config into `/tmp/dynaworld_v13_fulltrain_configs/` and measured the current
runnable variants with `src/benchmarks/trainer_phase_benchmark.py`.

Important correction: the `528MiB -> 48MiB` v13b result is still memory
accounting for the intended RGB-gradient handoff, not a measured runtime drop in
the current training graph. The runnable `v13b_rgb_grad_handoff` variant imports
and trains through the inherited v11-compatible feature raster path; the fused
handoff op is still a scaffold that intentionally raises until the Metal kernel
exists.

Measured medians from `/tmp/dynaworld_v13_fulltrain_bench/`:

```text
fixed render:
  v5_features                         1182.8ms total, 818.1ms backward
  v11 fixedbin                         861.6ms total, 671.6ms backward
  v13a recompute                       708.5ms total, 554.4ms backward
  v13b renamed/handoff scaffold        839.2ms total, 611.2ms backward

full train:
  v11 fixedbin                        1953.6ms total, 818.1ms backward, 1524.6MiB sampled current peak
  v13a recompute                      1743.7ms total, 924.0ms backward, 1417.7MiB sampled current peak
  v13b renamed/handoff scaffold       1721.3ms total, 929.1ms backward, 1525.1MiB sampled current peak
```

The full-train timings are noisy because encode/model and MPS timing move around
between processes. A prior 2-iter v11 pass hit `1557.8ms`, while the first v13b
pass had a bad encode/project outlier at `2459.6ms`. The reliable conclusion is
that v13b is not yet a proven full-train speedup; v13a can lower sampled current
allocator peak a bit but does not lower driver peak and pays extra backward
work.
