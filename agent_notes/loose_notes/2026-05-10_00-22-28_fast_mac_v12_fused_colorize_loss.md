# Fast-Mac v12 Fused Colorize/Loss Pass

## Context

The slowdown looked like raster backward at first, but phase probes showed the
big red flags were the image-space colorize/loss path:

- `standard_gs` means `0.8 * L1 + 0.2 * DSSIM`; DSSIM is the SSIM loss term.
- `FeatureToColor(pre_norm=true)` uses per-pixel `LayerNorm(F)` by permuting
  `[N,F,H,W] -> [N,H,W,F]`, applying `LayerNorm`, then permuting back.
- The active F32 colorizer for the tested config is already a 1x1 `Conv2d`
  when `hidden_dim=null`; there was no hidden 3x3 conv to remove.

Important measured probes:

- `benchmark_outputs/fast_mac_feature_kernels/2026-05-09_loss_colorize_breakdown_v11_features_512.json`
  - `standard_gs`, `pre_norm=true`
  - `loss_colorize_backward_probe`: `1562.77ms`
  - total step median: `5237.34ms`
  - backward median: `2573.68ms`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-09_loss_colorize_breakdown_v11_512_l1.json`
  - `l1`, `pre_norm=true`
  - `loss_colorize_backward_probe`: `1148.16ms`
  - total step median: `2389.68ms`
  - backward median: `1354.17ms`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-09_loss_colorize_breakdown_v11_512_no_prenorm.json`
  - `standard_gs`, `pre_norm=false`
  - `loss_colorize_backward_probe`: `464.44ms`
  - total step median: `1895.63ms`
  - backward median: `715.07ms`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_loss_colorize_breakdown_v11_512_l1_no_prenorm.json`
  - `l1`, `pre_norm=false`
  - `loss_colorize_backward_probe`: `67.66ms`
  - total one-shot step: `1441.14ms`
  - backward total: `407.17ms`

Full-step 512 comparisons:

- `benchmark_outputs/fast_mac_feature_kernels/2026-05-09_fullstep_v5_features_512_512_warm1_iters20.json`
  - total median `3899.35ms`
  - image-targets/s `8.21`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-09_fullstep_v11_features_512_512_warm1_iters20.json`
  - total median `2944.17ms`
  - image-targets/s `10.87`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_fullstep_v11_512_no_prenorm_warm1_iters20.json`
  - total median `2292.05ms`
  - image-targets/s `13.96`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_fullstep_v11_512_l1_no_prenorm_warm1_iters20.json`
  - total median `2027.04ms`
  - image-targets/s `15.79`

So the fastest proven trainer-side timing ablation is `v11 + pre_norm=false +
L1`: `1.45x` faster than v11 baseline and `1.92x` faster than v5 features.
This is speed evidence only; held-out quality/W&B is still needed before
promoting no-prenorm or no-DSSIM as a training default.

## v12 Branches

Three isolated branches were created under `third_party/fast-mac-gsplat/variants/`.
Stable v11/v6 variants were not edited for this experiment.

### v12a: fused no-norm colorize/alpha-compose/L1 producer

Files:

- `research_notes/fast_mac_v12a_fused_colorize_l1_no_norm.md`
- `third_party/fast-mac-gsplat/variants/v12a_fused_colorize_l1_no_norm/`
- `third_party/fast-mac-gsplat/variants/v12a_fused_colorize_l1_no_norm/benchmarks/benchmark_fused_colorize_l1.py`

v12a adds an actual new Metal op:

```text
torch.ops.gsplat_metal_v12a_fused_colorize_l1_no_norm.fused_no_norm_l1_grad(
    features[N,H,W,F],
    alpha[N,H,W],
    target_rgb[N,3,H,W],
    background_rgb[N,3,H,W],
    weight[3,F],
    bias[3],
) -> loss_per_image[N], grad_features[N,H,W,F], grad_alpha[N,H,W], grad_weight[3,F], grad_bias[3]
```

It is not a full fused raster backward. It replaces PyTorch's
`permute -> Conv2d -> sigmoid -> alpha compose -> L1 backward` gradient
producer, but still writes dense `grad_features` for the existing raster
backward to consume.

Validation:

- v12a extension builds with `setup.py build_ext --inplace`.
- inherited `feature_contract_check.py` passes:
  - F=3 v5 parity max_abs `0`
  - F32 feature grad max_abs `2.3283064e-10`
  - no-NaN smokes pass
- inherited `alpha_output_check.py` passes Tests A-F.
- fused producer parity at `N=32,512x512,F=32`:
  - loss abs `5.96e-08`
  - feature grad max abs `3.33e-15`
  - alpha grad max abs `2.13e-14`
  - weight grad max abs `3.26e-08`
  - bias grad max abs `1.92e-09`

Benchmarks:

- `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_v12a_fused_colorize_l1_N16_256_F32_convfair.json`
  - v12a fused producer median `8.79ms`
  - current trainer-style Torch Conv2d path median `52.05ms`
  - speedup `5.92x`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_v12a_fused_colorize_l1_N32_512_F32_convfair.json`
  - v12a fused producer median `61.49ms`
  - current trainer-style Torch Conv2d path median `568.69ms`
  - speedup `9.25x`

The old NHWF `einsum` comparison is kept only as a reference artifact:

- `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_v12a_fused_colorize_l1_N16_256_F32.json`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_v12a_fused_colorize_l1_N32_512_F32.json`

The convfair benchmark is the more relevant one because it mirrors the current
trainer boundary where fast-mac emits NHWF, then Python permutes to NCHW before
the 1x1 colorizer.

### v12b: RMSNorm branch

Files:

- `research_notes/fast_mac_v12b_fused_colorize_rmsnorm_l1.md`
- `third_party/fast-mac-gsplat/variants/v12b_fused_colorize_rmsnorm_l1/`
- `third_party/fast-mac-gsplat/variants/v12b_fused_colorize_rmsnorm_l1/torch_gsplat_bridge_v12b_fused_colorize_rmsnorm_l1/fused_colorize_l1.py`
- `third_party/fast-mac-gsplat/variants/v12b_fused_colorize_rmsnorm_l1/tests/fused_colorize_l1_reference_check.py`

v12b is a design/reference branch, not a Metal fused-kernel implementation yet.
It adds a pure PyTorch RMSNorm/no-norm colorize-alpha-L1 reference plus manual
closed-form gradient checks.

Validation:

- v12b extension builds.
- `fused_colorize_l1_reference_check.py` passes for F=3/8/32 RMSNorm gradients,
  no-norm identity activation, L1 kink zero-gradient behavior, and alpha-zero
  colorizer gradient blocking.
- inherited feature/alpha raster checks pass when run by the subagent.

RMSNorm may be a useful compromise if no-prenorm quality fails, but it is not
equivalent to LayerNorm. It normalizes scale but does not remove per-pixel
channel mean. Quality must be selected by held-out camera metrics, not speed.

### v12c: full fused raster-color-loss backward prototype

Files:

- `research_notes/fast_mac_v12c_fused_raster_color_loss_backward.md`
- `third_party/fast-mac-gsplat/variants/v12c_fused_raster_color_loss_backward/`
- `third_party/fast-mac-gsplat/variants/v12c_fused_raster_color_loss_backward/tests/fused_linear_sigmoid_mse_check.py`
- `third_party/fast-mac-gsplat/variants/v12c_fused_raster_color_loss_backward/benchmarks/benchmark_fused_linear_sigmoid_mse.py`

v12c is the hard target: compute the pixel colorize/compose/loss VJP inside the
tile raster backward, then immediately apply the contributor-loop VJP. That is
the branch that should avoid materializing `grad_features[N,H,W,F]` at all.

The implemented prototype is intentionally narrow: fixedbin/no-overflow fast
tiles, MPS only, `F <= 32`, linear 1x1 sigmoid colorizer, mean MSE, no active
tiles, no overflow fused path, no LayerNorm/RMSNorm, no hidden colorizer, no
view conditioning, no L1, and no DSSIM.

Correctness:

- `fused_linear_sigmoid_mse_check.py` passes B1/F3, B2/F8, and B2/F32 parity
  against unfused raster + PyTorch colorize/compose/MSE autograd.
- Worst observed drift in that parity script was around `1e-9`.

Benchmarks:

- `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_v12c_fused_linear_sigmoid_mse_B4_G2048_F32_128.json`
  - fused median `33.52ms`
  - unfused trainer-style median `35.87ms`
  - speedup `1.07x`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_v12c_fused_linear_sigmoid_mse_B16_G8192_F32_256.json`
  - fused median `653.64ms`
  - unfused trainer-style median `430.33ms`
  - speedup `0.66x`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_v12c_fused_linear_sigmoid_mse_B16_G8192_F32_256_freeze_colorizer.json`
  - fused median `652.77ms`
  - unfused trainer-style median `436.56ms`
  - speedup `0.67x`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_v12c_fused_linear_sigmoid_mse_B16_G8192_F32_512.json`
  - fused median `893.01ms`
  - unfused trainer-style median `539.58ms`
  - speedup `0.60x`

So v12c as implemented is a negative branch. Freezing the colorizer parameter
gradients did not recover the win, so the first suspicion about `[3,F]` global
atomics being the only blocker was wrong. The current full-fused kernel still
materializes the forward feature image and then runs scalar per-pixel colorize
math inside the tile kernel. PyTorch's optimized image-space conv path is fast
enough that removing `grad_features` write/read does not pay for that scalar
work.

This is the closest analogue to the FasterGS lesson: the expensive image-space
intermediate gradient is not a meaningful model state, so the best kernel
should avoid writing and rereading it. v12a still writes it; v12c is the design
that tries to remove it. The measured result says the boundary needs another
iteration: likely keep colorize/loss as a fast image-space kernel and fuse only
the `grad_features -> raster backward` consumption, or feed a compact RGB
gradient image into raster backward rather than moving the whole colorizer into
the tile loop.

SSIM/DSSIM complicates v12c. L1/MSE give local per-pixel `dL/dpred`; DSSIM
uses 11x11 local-window statistics, so `dL/dpred[p]` depends on neighboring
pixels. Correct DSSIM support probably needs a separate RGB-gradient image
kernel or a much larger fused image-stat path. Do not approximate DSSIM inside
the raster tile without calling it an objective change.

## Raster Copy Benchmarks

The v12 variants inherit the v11 raster lineage. Direct raster matrix checks
show no major copied-path regression.

256px, `B=16,G=8192,F=32`, `--backward --alpha-loss`:

- v11 median `478.49ms`, artifact
  `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_v11_features_gradcache_zero_bg_hostmeta_fixedbin_raster_B16_G8192_F32_256.jsonl`
- v12a median `456.15ms`, artifact
  `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_v12a_fused_colorize_l1_no_norm_raster_B16_G8192_F32_256.jsonl`
- v12b median `481.88ms`, artifact
  `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_v12b_fused_colorize_rmsnorm_l1_raster_B16_G8192_F32_256.jsonl`
- v12c median `437.06ms`, artifact
  `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_v12c_fused_raster_color_loss_backward_raster_B16_G8192_F32_256.jsonl`

512px, same case, 3 iterations:

- v11 median `580.23ms`, artifact
  `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_v11_features_gradcache_zero_bg_hostmeta_fixedbin_raster_B16_G8192_F32_512.jsonl`
- v12a median `579.44ms`, artifact
  `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_v12a_fused_colorize_l1_no_norm_raster_B16_G8192_F32_512.jsonl`
- v12b median `555.96ms`, artifact
  `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_v12b_fused_colorize_rmsnorm_l1_raster_B16_G8192_F32_512.jsonl`
- v12c median `562.59ms`, artifact
  `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_v12c_fused_raster_color_loss_backward_raster_B16_G8192_F32_512.jsonl`

The raster copy timings are noisy and not the promotion evidence. The promotion
evidence from this pass is the v12a fused producer parity plus speedup.

## Answers To The Open Questions

- FasterGS is relevant by analogy, but the current slowdown is not only the
  splat contributor loop. It is also the boundary where dense pixel feature
  gradients are materialized. v12c should push the color/loss VJP into the
  raster backward so the kernel never writes `grad_features[N,H,W,F]`.
- LayerNorm is not quadratic in pixels. It is linear in `N*H*W*F`, but it does
  multiple full feature-map passes and layout changes, and those passes dominate
  on MPS at 512.
- The pre-norm probe was run by toggling `colorize.pre_norm=false` in temporary
  benchmark configs and using `src/benchmarks/trainer_phase_benchmark.py
  --backward-breakdown`.
- Dropping DSSIM is a speed ablation, not automatically a quality fix. It must
  be W&B/heldout-tested separately.
- The current colorizer in this config is already 1x1. There is no 3x3 conv
  hidden in the active path.
- Pixel-space feature normalization is expensive because it sits at full
  resolution after splatting. If normalization is needed for quality, RMSNorm or
  learned scale control should be tested against heldout quality rather than
  assuming LayerNorm must stay.

## Remaining Work

1. Wire v12a into an opt-in trainer path that can keep fast-mac output as NHWF,
   call `fused_no_norm_l1_grad`, then feed the returned dense gradients to the
   existing raster backward. This is not done yet.
2. Redesign v12c after the negative benchmark. The current MSE prototype is
   correct but slower; the next boundary should avoid scalar per-pixel
   colorizer work in the tile loop.
3. Run W&B quality A/Bs:
   - v11 standard baseline quality run if not already done
   - no-prenorm quality run
   - L1/no-DSSIM quality run
   - RMSNorm quality run only after a real implementation path exists
4. If no-prenorm quality is bad, use v12b as the next stabilization branch
   rather than returning immediately to full LayerNorm.
5. Do not promote v12b/v12c based on copied raster timings; they need their own
   fused implementation and parity tests first.

## v12a Trainer Opt-In Follow-Up

After the prototype pass, I added an opt-in trainer-benchmark path rather than
changing the normal objective. The new wrapper lives in
`src/train/objective/v12a_fused_l1.py` and exposes
`fused_no_norm_l1_mean_loss(...)` as a custom autograd function around the
v12a Metal producer. The path is intentionally narrow:

- MPS only
- `colorize.pre_norm=false`
- no hidden MLP
- sigmoid activation
- no view conditioning
- `FeatureToColor` must be a single 1x1 Conv2d from `F -> 3`
- explicit target and background tensors

`src/benchmarks/trainer_phase_benchmark.py` now has `--v12a-fused-l1`. When
set, it bypasses `compose_rasterized(...)` and the normal reconstruction-loss
path for the benchmark chunk, calls the fused producer, and lets the custom
autograd function return precomputed gradients for raster features, alpha, and
colorizer weights/bias. This is a benchmark integration path, not a promoted
trainer config.

Smoke/parity:

- `py_compile` passed for the new wrapper and benchmark script.
- Small MPS smoke produced valid feature, alpha, weight, and bias gradients.
- Small parity against PyTorch no-norm L1 matched loss exactly and gradients to
  about `1e-9` or better.

### 128px 20-Iteration Benchmarks

Temporary configs under `/tmp/dynaworld_v12d_128/` were derived from the
current 3-camera/heldout F32 relpose config and forced to:

- `render_size=128`
- `feature_variant=v11_features_gradcache_zero_bg_hostmeta_fixedbin`
- `losses.type=l1`
- W&B/logging disabled for timing
- `train.steps=20`

Full train-step artifacts:

- Torch no-norm:
  `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_v12d_128_fullstep_torch_no_norm_l1_iters20.json`
- Torch LayerNorm:
  `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_v12d_128_fullstep_torch_layer_norm_l1_iters20.json`
- v12a fused no-norm:
  `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_v12d_128_fullstep_fused_no_norm_l1_iters20.json`

Full-step medians:

- Torch no-norm: `346.16ms`
- Torch LayerNorm: `390.14ms`
- v12a fused no-norm: `334.34ms`

The fused path is about `3.4%` faster than Torch no-norm by median and about
`14.3%` faster than LayerNorm by median on the full 128px trainer step. Mean is
noisy because encode/sample time had outliers.

Fixed-render artifacts:

- Torch no-norm:
  `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_v12d_128_fixedrender_torch_no_norm_l1_iters20.json`
- Torch LayerNorm:
  `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_v12d_128_fixedrender_torch_layer_norm_l1_iters20.json`
- v12a fused no-norm:
  `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_v12d_128_fixedrender_fused_no_norm_l1_iters20.json`

Fixed-render medians:

- Torch no-norm: `174.73ms`
- Torch LayerNorm: `265.58ms`
- v12a fused no-norm: `188.87ms`

This is the important caveat: at 128px, the fused producer is not a fixed-render
win over the already-cheap Torch no-norm path. It is still much faster than the
LayerNorm path, but the earlier `N32/512` 9.25x producer benchmark does not
automatically imply a 128px end-to-end trainer win. The producer overhead matters
when the image-space path is small.

Current read:

- v12a is useful as a high-resolution no-norm L1 fast path candidate.
- Removing LayerNorm/pre-norm remains the large speed lever.
- The opt-in path should stay benchmark-only until a 256/512 fixed-render or
  training A/B shows a clear end-to-end win.
- For 128px fast iteration, plain no-norm Torch is still the simpler baseline.

## Metal DSSIM Cost Probe

I added a standalone DSSIM forward+gradient probe to the experimental v12a
extension:

- C++/Metal op:
  `torch.ops.gsplat_metal_v12a_fused_colorize_l1_no_norm.dssim_forward_grad`
- Python wrapper:
  `third_party/fast-mac-gsplat/variants/v12a_fused_colorize_l1_no_norm/torch_gsplat_bridge_v12a_fused_colorize_l1_no_norm/fused_colorize_l1.py`
- Benchmark:
  `third_party/fast-mac-gsplat/variants/v12a_fused_colorize_l1_no_norm/benchmarks/benchmark_metal_dssim.py`

The op takes `prediction,target` in `[N,C,H,W]` float32 MPS layout and returns:

- `loss_per_image`: DSSIM per image
- `grad_prediction`: dense RGB gradient scaled for `loss_per_image.mean()`

It matches the current PyTorch DSSIM semantics closely: reflect padding,
`avg_pool2d`-style local means, 11x11 default window, and
`0.5 * (1 - mean(ssim_map))`. It intentionally does not touch raster kernels or
trainer dispatch.

Validation:

- Extension rebuilt with the canonical variant build command.
- `py_compile` passed for the new benchmark/wrapper.
- Existing v12a `fused_colorize_l1_check.py` still passes.
- Small DSSIM parity:
  - `N=2,H=W=32`: loss diff `6.26e-07`, grad max `2.12e-09`

Benchmarks versus PyTorch DSSIM autograd:

- `N=16,H=W=128`, artifact
  `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_v12a_metal_dssim_N16_128.json`
  - Metal median `7.60ms`
  - Torch median `19.52ms`
  - speedup `2.57x`
  - grad max `2.11e-11`
- `N=16,H=W=256`, artifact
  `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_v12a_metal_dssim_N16_256.json`
  - Metal median `18.67ms`
  - Torch median `64.80ms`
  - speedup `3.47x`
  - grad max `6.93e-12`
- `N=16,H=W=512`, artifact
  `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_v12a_metal_dssim_N16_512.json`
  - Metal median `70.45ms`
  - Torch median `234.66ms`
  - speedup `3.33x`
  - grad max `2.94e-12`

Read:

- Torch DSSIM really is slow enough on MPS to justify a native image-space VJP.
- This does not need to be fused into raster traversal first. The clean next
  boundary is `prediction,target -> Metal DSSIM + grad_rgb`, then feed
  `grad_rgb` into the colorize/alpha/raster VJP.
- The current Metal implementation is a straightforward one-kernel scatter with
  atomics, not the final optimized SSIM implementation. It is already favorable,
  so a tiled two-pass image kernel could plausibly improve further.

## Toggleable Objective Integration

I wired the DSSIM probe into the shared RGB objective rather than forking a new
trainer. The config toggle is:

```json
"losses": {
  "type": "standard_gs",
  "dssim_backend": "metal"
}
```

Default remains `"torch"`. The integration path is in:

- `src/train/objective/metal_dssim.py`
- `src/train/objective/loss.py`
- `src/train/objective/types.py`
- `src/train/train_video_token_implicit_dynamic.py`

This covers the current implicit/video trainer family because
`train_precomputed_feature_implicit_dynamic.py`,
`train_multicam_precomputed_feature_implicit_dynamic.py`, and
`train_multicam_relative_pose_implicit_dynamic.py` inherit from the base video
trainer and use `RGBReconObjective`. PowerFoam trainers still use their own
older `losses.py`/SSIM path.

Important implementation detail: the Metal op returns per-image DSSIM values
and a dense `grad_prediction` scaled for the mean DSSIM. In
`reconstruction_loss_per_image(...)`, the Metal backend expands the mean DSSIM
scalar across the per-image vector so existing `sum()/frame_count` and
`mean()` call sites get the right gradient scale. This preserves the active
training math, but it is not a true per-image DSSIM breakdown for logging.

Validation:

- Objective parity smoke on MPS:
  - loss diff `2.38e-07`
  - grad max diff `6.40e-10`
  - grad mean diff `7.53e-11`
- `py_compile` passed for touched objective/trainer files.
- `PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_rgb_recon_objective.py -q`
  passed (`5 passed`).

128px integration benchmarks using the current 3-cam relpose F32 config, with
`colorize.pre_norm=false`, `standard_gs`, and v11 feature rasterizer:

- Fixed-render Torch DSSIM:
  `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_metal_dssim_integration_fixedrender_torch_128_iters20.json`
  - median total `269.87ms`
  - median backward `190.48ms`
  - median loss `14.13ms`
- Fixed-render Metal DSSIM:
  `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_metal_dssim_integration_fixedrender_metal_128_iters20.json`
  - median total `229.01ms`
  - median backward `153.97ms`
  - median loss `17.89ms`

Full-step benchmarks:

- Torch DSSIM:
  `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_metal_dssim_integration_fullstep_torch_128_iters20.json`
  - median total `482.32ms`
  - median backward `195.25ms`
- Metal DSSIM:
  `benchmark_outputs/fast_mac_feature_kernels/2026-05-10_metal_dssim_integration_fullstep_metal_128_iters20.json`
  - median total `434.64ms`
  - median backward `168.53ms`

Read:

- The toggle is real and favorable at 128px: fixed-render median improves about
  `15.1%`; full-step median improves about `9.9%`.
- The win lands mostly in backward, as expected.
- The `loss` phase can look slightly higher because the custom op computes and
  stores the gradient during forward; the saved work appears in backward.
- We do not need a forked trainer for this path. A fork would just add more
  trainer surface area without isolating a different model contract.
