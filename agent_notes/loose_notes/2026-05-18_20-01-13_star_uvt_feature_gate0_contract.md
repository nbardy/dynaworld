# STAR UVT Feature Tubes Gate 0 Contract

Date: 2026-05-18

## Goal

Repeat the STAR UVT feature-shader plan, fill missing implementation details,
and execute the first gate with benchmark evidence instead of leaving the plan
as chat context.

## Starting Point

Earlier today we recorded the renderer scaling matrix:

- STAR UVT RGB direct atomic at 64f/32768 tubes:
  `110.4/182.4/521.5 ms` for `128/256/512px`.
- Dynamic RGB projected raster best rows:
  `252.7 ms` at 256px and `693.0 ms` at 512px.
- F32 projected feature raster best rows:
  `1582.2 ms` at 256px and `5920.8 ms` at 512px, with fixedbin/gradcache
  much better than stable but still far slower than RGB STAR.

Interpretation remains unchanged: STAR has the better time-tube representation,
fast-mac feature forks have the F32 shader tricks, and STAR UVT needs a
separate feature-valued renderer instead of mutating RGB `star_uvt_v0`.

## RGB Kernel Boundary Audit

`star_uvt_v0` is RGB-hardcoded in all relevant layers:

- Python wrapper validates `color.shape == [N,3]`.
- Python backward wrappers validate `grad_image == [T,H,W,3]`.
- Metal forward uses `float3`, `load3(color, tube_id)`, and `out_rgb`.
- Direct and compact backward use `float3 grad_rgb`, `float3 color`, and
  `atomic_add3` for `grad_color`.
- C++ bindings register RGB signatures and allocate `grad_color` as `[N,3]`.

So the F32 feature path is a real kernel/API fork:

```text
feature: [N,F]
forward -> feature_image [T,F,H,W] plus alpha [T,H,W]
backward <- grad_feature_image [T,F,H,W] plus grad_alpha [T,H,W]
```

This is not a safe one-line shape relaxation.

## Gate 0 Code Added

Updated:

```text
research_experiments/star_uvt_feature_tubes/dense_feature_tube_prototype.py
```

New Gate 0 capabilities:

- Optional frame-index rendering so chunks use the same absolute STAR time
  coordinates as the full render.
- `--gate0-benchmark` CLI with CPU/MPS device selection.
- JSON output for shapes, finite checks, split timings, gradient norms,
  tiny-overfit loss trace, and frame-chunked-vs-full backward parity.
- Chunked backward accumulates gradients chunk by chunk, so it is the parity
  gate we need before trainer render/loss microbatching.

## Commands

```bash
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  research_experiments/star_uvt_feature_tubes/dense_feature_tube_prototype.py
```

```bash
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/dense_feature_tube_prototype.py --smoke
```

```bash
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/dense_feature_tube_prototype.py \
  --gate0-benchmark --frames 5 --height 16 --width 16 --tubes 24 \
  --feature-dim 32 --steps 8 --chunk-size 2 --device cpu \
  --out-json outputs/benchmarks/2026-05-18_star_uvt_feature_gate0_dense_contract.json
```

```bash
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/dense_feature_tube_prototype.py \
  --gate0-benchmark --frames 5 --height 16 --width 16 --tubes 24 \
  --feature-dim 32 --steps 8 --chunk-size 2 --device mps \
  --out-json outputs/benchmarks/2026-05-18_star_uvt_feature_gate0_dense_contract_mps_repeat.json
```

## Results

CPU:

```text
artifact: outputs/benchmarks/2026-05-18_star_uvt_feature_gate0_dense_contract.json
pass: true
feature_image_shape: [5,32,16,16]
alpha_shape: [5,16,16]
rgb_shape: [5,3,16,16]
grad seen: raw_feature, center_uv, velocity_uv, colorizer
full dense timing: render 3.33ms, colorize+compose 0.44ms, backward 17.55ms
chunked parity: loss diff 7.45e-09, max grad diff 3.73e-09
tiny overfit: 0.20710 -> 0.11964 in 8 steps
```

MPS:

```text
artifact: outputs/benchmarks/2026-05-18_star_uvt_feature_gate0_dense_contract_mps_repeat.json
pass: true
feature_image_shape: [5,32,16,16]
alpha_shape: [5,16,16]
rgb_shape: [5,3,16,16]
grad seen: raw_feature, center_uv, velocity_uv, colorizer
warmed full parity timing: render 4.87ms, colorize+compose 0.82ms, backward 11.00ms
chunked parity: loss diff 3.73e-09, max grad diff 1.21e-08
tiny overfit: 0.18395 -> 0.11739 in 8 steps
```

The MPS dense prototype is launch/JIT-overhead dominated. Do not use these tiny
numbers as a real renderer-speed claim; use them as a contract and parity gate.

## Docs Updated

```text
research_experiments/star_uvt_feature_tubes/2026-05-18_fast_shader_port_plan.md
research_experiments/star_uvt_feature_tubes/README.md
EXPERIMENTS.md
PROJECT_INDEX.md
TODO/README.md
agent_notes/key_learnings.md
```

## Current State

Gate 0 is complete:

- dense feature tube output contract passes
- feature/geometry/colorizer gradients are nonzero
- tiny overfit decreases
- frame-chunked backward matches full backward

Gate 1 is not implemented yet. The next code step is a feature-specific Metal
module/fork with distinct import names. The first direct kernel should:

- reuse STAR tile/bin/order logic
- emit feature image and alpha
- accept upstream `grad_feature_image` and `grad_alpha`
- compute direct atomic gradients for geometry, opacity, and `[N,F]` features
- report tile counts, overflows, unstable tiles, and feature dimension

After direct feature parity, add a benchmark matrix with modes:

```text
feature_direct_atomic
feature_direct_gradcache
feature_direct_accum_f32
feature_direct_fixedbin
```

Fixedbin must stay opt-in with overflow fallback.

## Gate 1 Direct Feature Metal Follow-Up

After Gate 0 passed, I added the first direct feature Metal path with distinct
feature names. Existing RGB calls stay untouched.

Code added/changed:

```text
third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/metal/star_uvt_kernels.metal
third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/metal/star_uvt_metal.mm
third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/shared/common.h
third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/bindings.cpp
third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/feature_rasterize.py
third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/__init__.py
research_experiments/star_uvt_feature_tubes/direct_feature_kernel_benchmark.py
```

New API:

```text
render_uvt_feature_tubes(...) -> feature_image [T,F,H,W], alpha [T,H,W]
direct_atomic_feature_backward(..., grad_feature_image [T,F,H,W], grad_alpha [T,H,W])
```

Build:

```bash
( cd third_party/fast-mac-gsplat/variants/star_uvt_v0
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

The build succeeded.

Direct Metal tiny parity:

```text
artifact: outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_tiny_parity.json
pass: true
F=4 forward feature error 2.98e-08, alpha error 1.19e-07
F=4 backward errors: feature 7.15e-07, ma 2.38e-07, opacity 1.19e-06, q 4.17e-07
F=32 forward feature error 2.98e-08, alpha error 1.19e-07
F=32 backward errors: feature 1.43e-06, ma 7.15e-07, opacity 9.54e-06, q 1.91e-06
```

Direct Metal timing:

```text
artifact: outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_64f_128_32768_f32.json
pass: true/finite
64f/128px/32768/F32: 259.9ms total, 75.5ms forward, 184.4ms backward
overflow tiles: 8093
interpretation: stress row only; tile overflow means it is not a full-support quality row

artifact: outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_64f_256_32768_f32.json
pass: true/finite
64f/256px/32768/F32: 757.9ms total, 190.1ms forward, 567.8ms backward
overflow tiles: 0
interpretation: first usable direct feature stress row
```

Compared with the earlier renderer matrix, the direct feature STAR row at
256px is faster than projected F32 stable (`3642.2ms`) and v11 fixedbin
(`1582.2ms`) for this synthetic support pattern, but still about `4.16x`
slower than STAR RGB direct atomic at 256px (`182.4ms`). The expected next
bottlenecks are dense `[T,H,W,F]` output memory, per-channel feature writes, and
per-channel feature-gradient atomics.

Do not run full 64f/512px/F32 blindly: the feature image alone is about 2GB
before gradients. Add render/loss chunking or a chunked benchmark harness first.

## Gate 3 Mini Autograd / Video Overfit Smoke

I added a direct-feature autograd wrapper:

```text
torch_gsplat_bridge_star_uvt.feature_rasterize.render_uvt_feature_tubes_autograd
```

It returns:

```text
feature_image [T,F,H,W]
alpha [T,H,W]
```

and its backward routes upstream `FeatureToColor` and alpha-composition
gradients into `direct_atomic_feature_backward`. Depth tensors are still
order-only/non-differentiated, matching the current RGB direct-backward scope.

New benchmark script:

```text
research_experiments/star_uvt_feature_tubes/feature_autograd_overfit_benchmark.py
```

Synthetic autograd smoke:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/feature_autograd_overfit_benchmark.py \
  --frames 4 --size 32 --tubes 64 --feature-dim 32 --steps 12 \
  --out-json outputs/benchmarks/2026-05-18_star_uvt_feature_autograd_overfit_4f_32px_64t_f32.json
```

Result:

```text
pass: true
autograd-vs-manual max errors: feature 2.98e-07, ma 0, opacity 0, q 1.79e-07
loss: 0.22818 -> 0.10965 in 12 steps
mean step: 24.45ms
last step: 5.18ms
overflow tiles: 0
```

Real-video mini overfit:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/feature_autograd_overfit_benchmark.py \
  --video-path test_data/test_video_small_128_4fps.mp4 \
  --frames 8 --size 64 --tubes 512 --feature-dim 32 --steps 20 --lr 0.02 \
  --out-json outputs/benchmarks/2026-05-18_star_uvt_feature_autograd_video_overfit_8f_64px_512t_f32_20step.json
```

Result:

```text
pass: true
autograd-vs-manual max errors: feature 1.19e-07, ma 1.86e-08,
  opacity 5.96e-08, q 1.79e-07
loss: 0.18671 -> 0.04197 in 20 steps
PSNR: 7.29 -> 13.77
mean step: 24.70ms
last step: 14.82ms
overflow tiles: 0
grads seen: raw_feature, center_uv, velocity_uv, raw_opacity, raw_precision, colorizer
```

This closes the "trainable through FeatureToColor" smoke. It does not yet close
the first-class trainer/config acceptance gate, nor the 64f source-video quality
bracket.

## Gate 3 First-Class Trainer And Frame Chunking

Added:

```text
src/train/train.py
src/train/train_star_uvt_feature_overfit.py
src/train_configs/star_uvt_feature_testvideo_8f_64_directatomic_20step.jsonc
third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/feature_rasterize.py
```

The first-class route is:

```text
arch=star_uvt_feature_overfit
```

The trainer uses the direct feature Metal autograd wrapper, `FeatureToColor`,
and the same `test_data/test_video_small_128_4fps.mp4` loader path. It records
loss, PSNR, split timings, gradient-flow checks, and tile overflow.

First-class full-frame smoke:

```text
artifact: outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_8f_64px_512t_f32_20step.json
pass: true
8f/64px/512t/F32: 0.18602 -> 0.04167 loss in 20 steps
PSNR: 7.30 -> 13.80
mean step: 43.32ms
last step: 22.29ms
overflow tiles: 0
grads seen: raw_feature, center_uv, center_t, velocity_uv, raw_precision, raw_opacity, colorizer
```

Then I added frame chunking for the feature path. The important trick is that a
local chunk must keep the full-clip STAR time coordinate. For a chunk starting
at frame `s` with `C` frames inside a global `G` frame clip, the wrapper shifts
`ma.z` by:

```text
offset = s - 0.5*(G - 1) + 0.5*(C - 1)
ma_chunk.z = ma.z - offset
```

so the local renderer's `k - 0.5*(C-1)` time gives the same `delta_t` as the
global renderer's `(s+k) - 0.5*(G-1)`.

Frame-chunk parity:

```text
artifact: outputs/benchmarks/2026-05-18_star_uvt_feature_autograd_overfit_4f_32px_64t_f32_chunkparity.json
pass: true
chunk size: 2
max errors vs full autograd:
  feature 8.35e-07
  ma 1.64e-07
  opacity 2.24e-07
  q 2.98e-07
```

First-class chunked smoke:

```text
artifact: outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_8f_64px_512t_f32_chunk2_20step.json
pass: true
frame_chunk_size: 2
8f/64px/512t/F32: 0.18602 -> 0.04167 loss in 20 steps
PSNR: 7.30 -> 13.80
mean step: 76.79ms
last step: 59.51ms
overflow tiles: 0
tile max/p95: 76 / 74
fixedbin eligible: true
contact sheet: outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_8f_64px_512t_f32_chunk2_contact.png
side-by-side video: outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_8f_64px_512t_f32_chunk2_side_by_side.mp4
```

The chunked smoke is slower at this tiny size because it launches four chunks
per step. Its value is memory: it avoids requiring a full `[T,F,H,W]` feature
image for the entire clip. The next real bracket should scale this route to
64f/256px, keep media on quality candidates, then attempt 512px only through
frame chunks.

## Gate 3 Scale Probes

Added first-class scale configs:

```text
src/train_configs/star_uvt_feature_testvideo_64f_256_directatomic_chunk4_3step.jsonc
src/train_configs/star_uvt_feature_testvideo_64f_512_directatomic_chunk2_2step.jsonc
```

64f/256px probe:

```text
artifact: outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_8192t_f32_chunk4_3step.json
pass: true
source: test_data/test_video_384_128_6fps.mp4
tubes/features: 8192/F32
frame_chunk_size: 4
loss: 0.32612 -> 0.31141 in 3 steps
mean step: 964.66ms
mean forward: 120.89ms
mean colorize/loss: 71.80ms
mean backward: 736.03ms
overflow tiles: 0
tile max/p95: 80 / 63
fixedbin eligible: true
```

64f/512px probe:

```text
artifact: outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_512px_2048t_f32_chunk2_2step.json
pass: true
source: test_data/test_video_384_128_6fps.mp4
tubes/features: 2048/F32
frame_chunk_size: 2
loss: 0.34517 -> 0.34406 in 2 steps
mean step: 4020.73ms
mean forward: 586.93ms
mean colorize/loss: 281.59ms
mean backward: 3070.12ms
overflow tiles: 0
tile max/p95: 11 / 5
fixedbin eligible: true
```

64f/256px higher-capacity diagnostics:

```text
artifact: outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_16384t_f32_chunk4_2step.json
pass: false because tile overflow is nonzero
tubes/features: 16384/F32
frame_chunk_size: 4
loss: 0.32039 -> 0.30803 in 2 steps
mean step: 1075.03ms
mean forward: 133.41ms
mean colorize/loss: 76.67ms
mean backward: 815.62ms
overflow tiles: 736
tile max/p95: 151 / 123
overflow excess refs: 4528
fixedbin eligible: false

artifact: outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_2step.json
pass: false because tile overflow is nonzero
tubes/features: 32768/F32
frame_chunk_size: 4
loss: 0.31908 -> 0.30823 in 2 steps
mean step: 1142.94ms
mean forward: 142.82ms
mean colorize/loss: 87.67ms
mean backward: 863.23ms
overflow tiles: 8160
tile max/p95: 274 / 238
overflow excess refs: 753104
fixedbin eligible: false

artifact: outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_64_2step.json
pass: false because tile overflow is nonzero
tubes/features: 32768/F32
alpha_threshold: 1/64
frame_chunk_size: 4
loss: 0.32037 -> 0.31078 in 2 steps
mean step: 1778.87ms
mean forward: 403.40ms
mean colorize/loss: 122.31ms
mean backward: 1058.12ms
overflow tiles: 5814
tile max/p95: 230 / 191
overflow excess refs: 317382
fixedbin eligible: false

artifact: outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_32_2step.json
pass: false because tile overflow is nonzero
tubes/features: 32768/F32
alpha_threshold: 1/32
frame_chunk_size: 4
loss: 0.32264 -> 0.31420 in 2 steps
mean step: 1309.78ms
mean forward: 161.80ms
mean colorize/loss: 106.00ms
mean backward: 921.20ms
overflow tiles: 5538
tile max/p95: 205 / 168
overflow excess refs: 188460
fixedbin eligible: false
```

Cap-256 follow-up:

```text
env for these rows: STAR_UVT_TILE_CAPACITY=256

artifact: outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_16384t_f32_chunk4_cap256_2step.json
pass: true
tubes/features: 16384/F32
loss: 0.32038 -> 0.30802 in 2 steps
mean step: 1215.41ms
mean backward: 921.74ms
overflow tiles: 0
tile max/p95: 152 / 123
fixedbin eligible: true

artifact: outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_cap256_2step.json
pass: false because unpruned 32768 still overflows
tubes/features: 32768/F32
loss: 0.31651 -> 0.30624 in 2 steps
mean step: 1325.92ms
mean backward: 1035.58ms
overflow tiles: 216
tile max/p95: 275 / 238
fixedbin eligible: false

artifact: outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_64_cap256_20step.json
pass: true
tubes/features: 32768/F32
alpha_threshold: 1/64
loss: 0.31921 -> 0.29350 in 20 steps
PSNR: 4.96 -> 5.32
mean step: 1159.18ms
mean backward: 926.09ms
overflow tiles: 0
tile max/p95: 248 / 204
fixedbin eligible: true
contact sheet: outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_64_cap256_20step_contact.png
side-by-side video: outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_64_cap256_20step_side_by_side.mp4

artifact: outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_32_cap256_20step.json
pass: true
tubes/features: 32768/F32
alpha_threshold: 1/32
loss: 0.32217 -> 0.29861 in 20 steps
PSNR: 4.92 -> 5.25
mean step: 1174.10ms
mean backward: 915.24ms
overflow tiles: 0
tile max/p95: 213 / 175
fixedbin eligible: true
contact sheet: outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_32_cap256_20step_contact.png
side-by-side video: outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_32_cap256_20step_side_by_side.mp4
```

Interpretation:

- Frame chunking works as a memory valve at 64 frames and 512px.
- The current direct feature kernel is backward-dominated at scale.
- The 512px row is only a feasibility probe because it uses 2048 tubes; do not
  compare it as quality parity against the 32768-tube RGB STAR row.
- Real-video 256px support is stricter than the synthetic direct-feature timing
  row: 8192 tubes is currently the largest zero-overflow first-class bracket,
  while 16384 is near the 128-entry cap and 32768 is far over it.
- Cap 256 converts 16384 into a valid row and converts 32768 into a valid row
  only when paired with support pruning. Unpruned 32768 still overflows at cap
  256.
- At this point in the session, `alpha>=1/64/cap256` looked better than
  `alpha>=1/32/cap256`; the cap256 alpha bracket below supersedes this with
  `alpha>=1/72/cap256` as the best passing row. These are still low-PSNR
  validity/speed candidates rather than quality replacements.
- Next useful work is feature-specific gradcache/accum/fixedbin modes and
  explicit overflow fallback.

## Media And Scale Report Follow-Up

Added:

```text
research_experiments/star_uvt_feature_tubes/firstclass_scale_report.py
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_scale_report.md
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_scale_summary.json
```

The first-class trainer now accepts explicit media output paths and writes:

```text
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_8f_64px_512t_f32_chunk2_contact.png
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_8f_64px_512t_f32_chunk2_side_by_side.mp4
```

Validation:

```text
contact sheet: PNG, 526x130, target row over prediction row
side-by-side MP4: 128x64, 8 frames, 4 fps, 2.0s
scale report: summarizes 8f/64px, 64f/256px default, alpha-pruned, and cap256
  rows, plus 64f/512px rows with tile max/p95/fixedbin eligibility
```

The visual media confirms the 8f proof is an overfit smoke, not a quality row:
the prediction row has learned coarse motion/color but is still blurry and
low-detail.

## Cap256 Alpha Bracket Follow-Up

Ran the cap-256 support bracket around the prior `alpha>=1/64` candidate.
Important detail: 2-step tile stats were not enough, because the optimizer can
grow support and overflow later in the 20-step row.

```text
artifact: outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_96_cap256_2step.json
pass: true
alpha_threshold: 1/96
loss: 0.31830 -> 0.30875
mean step/backward: 1255.92ms / 960.53ms
overflow tiles: 0
tile max/p95: 238 / 204

artifact: outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_80_cap256_2step.json
pass: true
alpha_threshold: 1/80
loss: 0.31865 -> 0.30922
mean step/backward: 1213.18ms / 941.09ms
overflow tiles: 0
tile max/p95: 236 / 199

artifact: outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_72_cap256_2step.json
pass: true
alpha_threshold: 1/72
loss: 0.31889 -> 0.30955
mean step/backward: 1621.19ms / 1082.21ms
overflow tiles: 0
tile max/p95: 232 / 195

artifact: outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_96_cap256_20step.json
pass: false due late tile overflow
alpha_threshold: 1/96
loss: 0.31830 -> 0.29150
PSNR: 4.97 -> 5.35
mean step/backward: 1182.75ms / 944.77ms
overflow tiles: 12
tile max/p95: 269 / 220
contact sheet: outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_96_cap256_20step_contact.png
side-by-side video: outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_96_cap256_20step_side_by_side.mp4

artifact: outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_80_cap256_20step.json
pass: false due late tile overflow
alpha_threshold: 1/80
loss: 0.31865 -> 0.29237
PSNR: 4.97 -> 5.34
mean step/backward: 1173.14ms / 931.32ms
overflow tiles: 6
tile max/p95: 261 / 213
contact sheet: outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_80_cap256_20step_contact.png
side-by-side video: outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_80_cap256_20step_side_by_side.mp4

artifact: outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_72_cap256_20step.json
pass: true
alpha_threshold: 1/72
loss: 0.31889 -> 0.29290
PSNR: 4.96 -> 5.33
mean step/backward: 1320.92ms / 1021.20ms
overflow tiles: 0
tile max/p95: 252 / 209
contact sheet: outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_72_cap256_20step_contact.png
side-by-side video: outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_72_cap256_20step_side_by_side.mp4
```

Interpretation update:

- `alpha>=1/72/cap256` is now the quality-max passing 32768-tube 20-step row.
- `alpha>=1/64/cap256` remains the safer fallback because `1/72` only has four
  refs of headroom under the 256 cap.
- `alpha>=1/80` and `alpha>=1/96` prove the quality frontier but cannot be
  called fixed-bin candidates until overflow fallback or higher-cap memory is
  implemented.
- The report labeler now parses `alpha1_N` result filenames and the trainer
  now writes `alpha_threshold`/`max_alpha` into future JSON rows.

## Matched Renderer Table Follow-Up

Extended `research_experiments/renderer_scaling_report.py` so the existing
STAR/dynamic/F32 scaling report also ingests
`outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_scale_summary.json`.
Regenerated:

```text
outputs/benchmarks/2026-05-18_renderer_scaling_report.md
outputs/benchmarks/2026-05-18_renderer_scaling_report.csv
```

The new report has `90` rows and a focused `64f/256px/32768` section. The
current same-shape speed picture is:

```text
STAR UVT RGB direct_atomic: 182.4ms backward-only kernel probe
STAR UVT RGB direct_fixedpoint: 174.9ms kernel probe, but not promoted by training stability
Dynamic RGB projected v8: 252.7ms total / 172.2ms backward
STAR UVT F32 feature alpha>=1/72/cap256: 1320.9ms total / 1021.2ms backward, zero overflow
STAR UVT F32 feature alpha>=1/80/cap256: 1173.1ms total / 931.3ms backward, late overflow
Projected F32 v11 fixedbin: 1582.2ms total / 1135.0ms backward, synthetic projected raster
Projected F32 stable: 3642.2ms total / 3117.7ms backward
```

This keeps the next shader decision crisp: the first feature STAR row is
already in the range of the projected F32 fixedbin/v11 trick, but it gets there
by support pruning and direct atomics rather than by porting the feature
gradcache/fixedbin ideas. The next useful implementation target remains a
feature-specific STAR UVT gradcache or fixedbin-with-overflow-fallback mode,
then rerun this same report.

## Feature Direct Fixedbin Mode Contract

Added `feature_uvt.render_mode` to the first-class feature trainer with two
accepted values:

```text
feature_direct_atomic
feature_direct_fixedbin
```

This is deliberately a contract surface, not a separate optimized kernel yet.
The current implementation still uses the direct feature Metal path, then
records `requested_render_mode`, `effective_render_mode`, and
`mode_fallback_required` from final tile overflow stats. The purpose is to stop
future docs/reports from silently promoting a fixed-bin run that overflowed.

Unpruned fallback probe:

```text
command:
STAR_UVT_TILE_CAPACITY=256 PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  .venv/bin/python src/train/train.py \
  src/train_configs/star_uvt_feature_testvideo_64f_256_featurefixedbin_chunk4_32768t_cap256_2step.jsonc

artifact:
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_featurefixedbin_cap256_2step.json

result:
pass: false
requested_render_mode: feature_direct_fixedbin
effective_render_mode: feature_direct_atomic
mode_fallback_required: true
overflow tiles: 216
max tile / p95: 275 / 238
loss: 0.31651 -> 0.30624
PSNR: 4.996 -> 5.139
mean step: 1603.07ms
forward: 183.32ms
color/loss: 104.12ms
backward: 1146.56ms
```

Passing fixedbin-eligible probe:

```text
command:
STAR_UVT_TILE_CAPACITY=256 PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  .venv/bin/python src/train/train.py \
  src/train_configs/star_uvt_feature_testvideo_64f_256_featurefixedbin_chunk4_32768t_alpha1_72_cap256_20step.jsonc

artifact:
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_featurefixedbin_alpha1_72_cap256_20step.json

media:
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_featurefixedbin_alpha1_72_cap256_20step_contact.png
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_featurefixedbin_alpha1_72_cap256_20step_side_by_side.mp4

result:
pass: true
requested_render_mode: feature_direct_fixedbin
effective_render_mode: feature_direct_fixedbin
mode_fallback_required: false
overflow tiles: 0
max tile / p95: 252 / 209
loss: 0.31889 -> 0.29290
PSNR: 4.964 -> 5.333
mean step: 1252.83ms
forward: 163.88ms
color/loss: 79.12ms
backward: 991.68ms
media render: 226.27ms
```

The scale report was regenerated with these rows and now has 92 entries. The
renderer matrix labels the unpruned row as
`feature_direct_fixedbin->feature_direct_atomic` and the pruned row as
`alpha>=1/72/feature_direct_fixedbin`.

Important interpretation: the fixedbin-eligible row is a useful benchmark and a
promotion guard, but the speed difference versus the earlier
`alpha>=1/72/cap256` direct-atomic row should be treated as rerun variance until
the Metal feature backward changes. The next actual shader work remains a
feature-specific gradcache/accum or optimized fixedbin backward mode, followed
by the same matched renderer report.

## Feature Direct Gradcache Kernel Pass

Ported the first real fast-mac feature-shader lesson into STAR UVT feature
backward: the direct feature kernel can now cache each pixel's F32
`grad_feature_image[pixel]` vector in thread-local storage when
`feature_dim <= 64`. The mode is opt-in through `feature_uvt.render_mode =
feature_direct_gradcache`; plain `feature_direct_atomic` keeps the non-cache
path, and `feature_direct_fixedbin` remains the overflow/promotion guard.

Code surfaces:

```text
third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/metal/star_uvt_kernels.metal
third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/feature_rasterize.py
src/train/train_star_uvt_feature_overfit.py
research_experiments/star_uvt_feature_tubes/direct_feature_kernel_benchmark.py
src/train_configs/star_uvt_feature_testvideo_64f_256_gradcache_chunk4_32768t_alpha1_72_cap256_20step.jsonc
```

Serial synthetic A/B, same command shape, `warmup=2`, `repeat=5`:

```text
direct_atomic artifact:
outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_directatomic_serial_64f_256_32768_f32.json

direct_atomic result:
pass: true
64f/256px/32768t/F32
forward: 144.59ms
backward: 485.63ms
total: 630.22ms
overflow: 0

gradcache artifact:
outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_serial_64f_256_32768_f32.json

gradcache result:
pass: true
64f/256px/32768t/F32
forward: 150.13ms
backward: 471.29ms
total: 621.42ms
overflow: 0
```

The synthetic win is real but small: about `14.34ms` backward, or `2.95%` of
the direct-atomic backward row. It proves the mode works and removes repeated
global loads of the same pixel grad vector, but it does not address the larger
per-feature-channel atomic cost.

First-class real-video gate:

```text
command:
STAR_UVT_TILE_CAPACITY=256 PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  .venv/bin/python src/train/train.py \
  src/train_configs/star_uvt_feature_testvideo_64f_256_gradcache_chunk4_32768t_alpha1_72_cap256_20step.jsonc

artifact:
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_gradcache_alpha1_72_cap256_20step.json

media:
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_gradcache_alpha1_72_cap256_20step_contact.png
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_gradcache_alpha1_72_cap256_20step_side_by_side.mp4

result:
pass: true
requested_render_mode: feature_direct_gradcache
effective_render_mode: feature_direct_gradcache
mode_fallback_required: false
loss: 0.31889 -> 0.29290
PSNR: 4.964 -> 5.333
mean step: 1226.04ms
forward: 159.01ms
color/loss: 73.12ms
backward: 973.24ms
overflow: 0
max tile / p95: 252 / 209
```

The regenerated reports now include 94 renderer rows and a first-class scale row
for `alpha>=1/72/feature_direct_gradcache`. Use gradcache as the current
fastest valid STAR UVT feature mode, but the next shader gate should attack the
dominant cost: feature-gradient atomics/reduction or an RGB-grad handoff, plus a
real optimized fixedbin backward if the tile cap is satisfied.

## Feature-Gradient Atomic Cost Diagnostic

Added a benchmark-only mode to isolate the cost of per-channel feature-gradient
atomic writes:

```text
direct_atomic_skip_feature_grad
gradcache_skip_feature_grad
```

These modes intentionally return incorrect/zero `grad_feature`; they are not
valid train modes and are not accepted by `feature_uvt.render_mode`. The useful
property is that geometry/opacity gradients still match the dense reference, so
the timing isolates the `grad_feature[N,F]` atomic writes.

Command:

```text
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  .venv/bin/python research_experiments/star_uvt_feature_tubes/direct_feature_kernel_benchmark.py \
  --feature-dims 4,32 --timing-frames 64 --timing-size 256 \
  --timing-tubes 32768 --timing-feature-dim 32 \
  --backward-mode gradcache_skip_feature_grad --timing-warmup 2 --timing-repeat 5 \
  --out-json outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_skip_feature_grad_serial_64f_256_32768_f32.json
```

Result:

```text
artifact:
outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_skip_feature_grad_serial_64f_256_32768_f32.json

pass: true
feature_grad_skipped: true
64f/256px/32768t/F32
forward: 187.25ms
backward: 327.71ms
total: 514.96ms
overflow: 0

tiny parity:
F4 ma/q/opacity max errors: 1.19e-07 / 4.17e-07 / 1.19e-06
F32 ma/q/opacity max errors: 9.54e-07 / 2.86e-06 / 9.54e-06
feature grad errors are intentionally large because feature grad is skipped
```

Nearby full-gradcache rerun:

```text
artifact:
outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_serial_rerun2_64f_256_32768_f32.json

pass: true
64f/256px/32768t/F32
forward: 193.99ms
backward: 592.54ms
total: 786.53ms
overflow: 0
```

Interpretation:

- The exact ratio is MPS-noisy; earlier full-gradcache serial was `471.29ms`
  backward, and the nearby rerun was `592.54ms`.
- The skip-feature-gradient diagnostic still cuts backward to `327.71ms`, so
  feature-gradient atomics are a large enough cost to justify the next real
  shader port.
- Next useful implementation is not another grad-vector cache. It should be a
  feature-gradient reduction/accum path, RGB-grad handoff, or optimized fixedbin
  backward that reduces per-channel atomics while preserving real `grad_feature`.

## Trainable Reduced Feature-Gradient Prototype

Ported a v11-style per-tile/simd feature-gradient reduction into STAR UVT
feature direct backward:

```text
backward mode:
gradcache_reduce_feature_grad

trainer render mode:
feature_direct_gradcache_reduce

config:
src/train_configs/star_uvt_feature_testvideo_64f_256_gradcache_reduce_chunk4_32768t_alpha1_72_cap256_20step.jsonc
```

Implementation details:

- `star_uvt_kernels.metal` adds mode bit `4` and a
  `reduce_atomic_add_feature_grads_cached(...)` helper.
- The helper uses `simd_sum` plus `STAR_SIMDGROUPS * STAR_FEATURE_GRAD_CACHE_CAP`
  threadgroup scratch to reduce `grad_feature[tube, channel]` once per stable
  tile/tube/channel instead of doing every pixel/channel atomic directly.
- The path only enables when `feature_dim <= 64`, gradcache is active, and the
  tile is depth-order stable. Unstable/unsupported cases keep the direct atomic
  behavior.
- Edge pixels remain in the barrier path with zero contribution so threadgroup
  barriers stay uniform.

Parity/timing commands:

```text
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  .venv/bin/python research_experiments/star_uvt_feature_tubes/direct_feature_kernel_benchmark.py \
  --feature-dims 4,32 --backward-mode gradcache_reduce_feature_grad \
  --skip-timing \
  --out-json outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_reduce_tiny_parity.json

PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  .venv/bin/python research_experiments/star_uvt_feature_tubes/direct_feature_kernel_benchmark.py \
  --feature-dims 4,32 --timing-frames 64 --timing-size 256 \
  --timing-tubes 32768 --timing-feature-dim 32 \
  --backward-mode gradcache_reduce_feature_grad --timing-warmup 2 --timing-repeat 5 \
  --out-json outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_reduce_serial_64f_256_32768_f32.json

PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  .venv/bin/python research_experiments/star_uvt_feature_tubes/direct_feature_kernel_benchmark.py \
  --feature-dims 4,32 --timing-frames 64 --timing-size 256 \
  --timing-tubes 32768 --timing-feature-dim 32 \
  --backward-mode gradcache --timing-warmup 2 --timing-repeat 5 \
  --out-json outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_serial_rerun3_64f_256_32768_f32.json
```

Synthetic results:

```text
gradcache_reduce_feature_grad:
  pass: true
  64f/256px/32768t/F32
  forward: 186.04ms
  backward: 523.77ms
  total: 709.80ms
  overflow: 0

same-session gradcache rerun3:
  pass: true
  forward: 162.98ms
  backward: 491.07ms
  total: 654.04ms
  overflow: 0
```

First-class 20-step gate:

```text
command:
STAR_UVT_TILE_CAPACITY=256 \
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  .venv/bin/python src/train/train.py \
  src/train_configs/star_uvt_feature_testvideo_64f_256_gradcache_reduce_chunk4_32768t_alpha1_72_cap256_20step.jsonc

artifact:
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_gradcache_reduce_alpha1_72_cap256_20step.json

media:
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_gradcache_reduce_alpha1_72_cap256_20step_contact.png
outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_gradcache_reduce_alpha1_72_cap256_20step_side_by_side.mp4

pass: true
requested/effective mode: feature_direct_gradcache_reduce
loss: 0.3188934475 -> 0.2928974822
PSNR: 4.9635 -> 5.3328
mean step: 1260.62ms
forward: 165.10ms
colorize/loss: 72.49ms
backward: 1000.33ms
optimizer: 11.32ms
overflow: 0
max tile / p95: 252 / 209
```

Media check:

```text
contact PNG: 2062 x 514
MP4: 512x256, 64 frames, 6 fps, duration 10.666667s
```

Interpretation:

- The prototype is trainable and correct, but it is slower than plain
  gradcache in both the synthetic same-session comparison and the first-class
  20-step row.
- The negative result is useful: STAR UVT's direct feature kernel already has
  high per-tile contributor count, and adding per-contributor threadgroup
  barriers eats the feature-atomic savings.
- Do not promote `feature_direct_gradcache_reduce` as the default. Keep
  `feature_direct_gradcache` as the fastest valid mode for now.
- The next attempt should use a different shape: optimized fixedbin feature
  backward, a two-pass/accum feature-gradient route, or RGB-grad handoff. A
  simple v11-style barrier reduction is not enough.

## Narrow RGB Handoff Prototype

Added a benchmark-only STAR feature backward mode:

```text
fused_first3_sigmoid_mse
```

Contract:

```text
rendered feature image: [T,F,H,W]
rendered alpha: [T,H,W]
RGB objective: alpha * sigmoid(feature_image[:, 0:3]) -> mean MSE(target_rgb)
target_rgb carrier: channels 0..2 of the existing grad_feature_image argument
colorizer grads: not returned
learned F32 FeatureToColor: not supported
```

The point is to test the RGB-gradient handoff shape without changing the C++
custom-op signature: the Metal backward reconstructs the first three pixel
feature channels and alpha while replaying contributors, computes the local
sigmoid/MSE VJP, and feeds that thread-local gradient into the normal STAR
reverse contributor loop. This avoids the full F32 `grad_feature_image` load in
the kernel, but it is not a production colorizer path yet.

Parity command:

```text
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  .venv/bin/python research_experiments/star_uvt_feature_tubes/direct_feature_kernel_benchmark.py \
  --feature-dims 4,32 --backward-mode fused_first3_sigmoid_mse \
  --skip-timing \
  --out-json outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_fused_first3_sigmoid_mse_tiny_parity.json
```

Timing commands:

```text
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  .venv/bin/python research_experiments/star_uvt_feature_tubes/direct_feature_kernel_benchmark.py \
  --feature-dims 4,32 --timing-frames 64 --timing-size 256 \
  --timing-tubes 32768 --timing-feature-dim 32 \
  --backward-mode fused_first3_sigmoid_mse --timing-warmup 2 --timing-repeat 5 \
  --out-json outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_fused_first3_sigmoid_mse_serial_64f_256_32768_f32.json

PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  .venv/bin/python research_experiments/star_uvt_feature_tubes/direct_feature_kernel_benchmark.py \
  --feature-dims 4,32 --timing-frames 64 --timing-size 256 \
  --timing-tubes 32768 --timing-feature-dim 32 \
  --backward-mode gradcache --timing-warmup 2 --timing-repeat 5 \
  --out-json outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_serial_rerun4_after_fused_64f_256_32768_f32.json

PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  .venv/bin/python research_experiments/star_uvt_feature_tubes/direct_feature_kernel_benchmark.py \
  --feature-dims 4,32 --timing-frames 64 --timing-size 256 \
  --timing-tubes 32768 --timing-feature-dim 32 \
  --backward-mode gradcache_skip_feature_grad --timing-warmup 2 --timing-repeat 5 \
  --out-json outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_skip_feature_grad_serial_rerun2_after_fused_64f_256_32768_f32.json
```

Results:

```text
tiny parity:
F4 max errors: feature 1.40e-09, ma 2.33e-09, opacity 5.22e-08, q 1.49e-08
F32 max errors: feature 6.98e-10, ma 3.73e-09, opacity 1.12e-08, q 1.49e-08

64f/256px/32768t/F32 fused_first3_sigmoid_mse:
  pass: true
  forward: 159.37ms
  backward: 309.31ms
  total: 468.68ms
  overflow: 0

same-session full gradcache rerun4:
  pass: true
  forward: 169.80ms
  backward: 547.58ms
  total: 717.39ms
  overflow: 0

same-session skip-feature-gradient rerun2:
  pass: true for geometry/opacity parity; feature grad intentionally zero
  forward: 165.31ms
  backward: 351.58ms
  total: 516.89ms
  overflow: 0
```

Interpretation:

- The fused first3 gate is a positive shape test: local RGB objective VJP at the
  raster boundary is much better than the barrier-heavy reduction attempt.
- It is deliberately narrow. It only proves `alpha * sigmoid(feature[:3])`
  MSE, not the real learned 32-channel `FeatureToColor`.
- The next handoff gate should pass linear colorizer weights/biases into the
  kernel and return colorizer parameter gradients. After that, trainer
  integration can be considered for `hidden_dim=null`, `pre_norm=false`,
  `view_condition=none`.
- The regenerated renderer scaling report now includes direct STAR feature
  kernel rows, including this fused handoff row.
