# Owner-Run Lean Recompute Shader Diagnostic

## What changed

Continued the owner-run/segment-tape fork after the ownerkeep-i16 clean reject. The useful idea was narrower than the owner side-stream variants: keep the compact owner-run tape, but stop carrying `segment_alpha`, `weights`, and `segment_rgb` local arrays in the RGB-only fused-MSE VJP kernels. The reverse pass can recompute those values from `owner`, `trans_before`, `segment_trans`, `length`, and `site_rgba`.

Touched only the RGB-only MSE VJP hot paths in:

- `wf2_endpoint_run_mse_vjp_direct_atomic_rgb_only_tensor`
- `wf2_segment_tape_mse_vjp_direct_atomic_rgb_only_tensor`

I left the RGBA/depth VJP paths alone because they still use the stored per-segment values for alpha/depth gradient structure.

## Fixes

The first lean patch removed the arrays from the segment-tape MSE path but left writes in the adjacent endpoint-run MSE path, causing Metal compile errors for undeclared `segment_alpha`, `weights`, and `segment_rgb`. I fixed that by applying the same reverse-pass recompute pattern to `wf2_endpoint_run_mse_vjp_direct_atomic_rgb_only_tensor`.

## Verification

Focused endpoint/segment tape unit gate:

```bash
PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools:src/train \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_probe_endpoint_run_tape -v
```

Result: 5/5 passed, including endpoint-run fused MSE and segment-tape fused MSE matching replay plus RGB VJP references.

Focused train/eval smoke:

```bash
PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:src/train \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --tape-mode owner-run-fused-mse-nomid \
  --frame-counts 2 \
  --render-size 16 \
  --site-count 8 \
  --optimizer-mode manual-vjp \
  --steps 1 \
  --warmup-steps 0 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-19_owner_run_fused_mse_nomid_lean_smoke_2f_render16_site8.json
```

Result: status ok, finite loss, nonzero gradients, parameters updated. Environment was contended, so this is correctness-only.

## Timing diagnostic

Scale artifact:

`research_experiments/world_foam_lane2/results/2026-05-19_owner_run_fused_mse_nomid_lean_scale_2_4_8_16_render16_site24_step3_warm1_attempt_clean2.json`

The run started with `benchmark_environment.start.status=background` but ended `contended` because the unrelated `ai_trader` verifier started before the final environment snapshot. Treat this as a strong diagnostic, not a promotion gate.

Step means by frame count:

| frames | total ms | backward ms | selected tape bytes |
| --- | ---: | ---: | ---: |
| 2 | 1.451 | 0.988 | 19,028 |
| 4 | 1.340 | 1.059 | 44,012 |
| 8 | 1.409 | 1.117 | 90,684 |
| 16 | 1.381 | 1.084 | 183,668 |

Scale: total `0.952x`, backward `1.098x`, selected tape storage `9.653x` over an 8x frame-count increase. All row acceptance checks passed.

Against the old contended owner-run-nomid artifact, lean is much faster:

- total ratios lean/old: `0.50 / 0.47 / 0.48 / 0.31x`
- backward ratios lean/old: `0.43 / 0.46 / 0.48 / 0.33x`

Against the clean coeff16 sample control (`2026-05-19_gate4_ownerkeep_i16_zero_length_promotion_pair_sample_attempt8.json`), lean is faster despite the end-contended caveat:

- total ratios lean/sample: `0.57 / 0.53 / 0.65 / 0.67x`
- backward ratios lean/sample: `0.51 / 0.50 / 0.64 / 0.64x`

Against the existing matched small STAR UVT direct-atomic speed gate, lean median timing is also faster at this tiny render/site shape:

- STAR/lean total median ratios: `5.94 / 4.32 / 4.23 / 4.26x`
- STAR/lean backward median ratios: `3.81 / 2.84 / 2.90 / 2.89x`

That STAR comparison is only the existing small-MPS speed gate, not quality/capacity parity.

## Clean repeat

The fully clean repeat succeeded after two blocked environment attempts:

`research_experiments/world_foam_lane2/results/2026-05-19_owner_run_fused_mse_nomid_lean_scale_2_4_8_16_render16_site24_step3_warm1_clean_repeat_attempt3.json`

Both `benchmark_environment.start.status` and `benchmark_environment.end.status` are `background`, and the top-level status is `ok`.

Clean repeat step means:

| frames | total ms | backward ms | total median ms | backward median ms |
| --- | ---: | ---: | ---: | ---: |
| 2 | 1.571 | 1.256 | 1.540 | 1.173 |
| 4 | 1.610 | 1.237 | 1.525 | 1.188 |
| 8 | 1.502 | 1.170 | 1.466 | 1.150 |
| 16 | 1.782 | 1.479 | 1.488 | 1.181 |

Clean repeat scales:

- total mean first-to-last: `1.1345x`
- backward mean first-to-last: `1.1769x`
- selected tape storage first-to-last: `9.6525x`

Clean repeat versus clean coeff16 sample control:

- total mean ratios: `0.612 / 0.631 / 0.690 / 0.858x`
- backward mean ratios: `0.649 / 0.587 / 0.665 / 0.870x`

Clean repeat versus old owner-run-nomid:

- total mean ratios: `0.542 / 0.564 / 0.516 / 0.404x`
- backward mean ratios: `0.546 / 0.536 / 0.507 / 0.449x`

Clean repeat versus the existing small STAR UVT direct-atomic speed gate:

- STAR/lean total median ratios: `5.11 / 3.86 / 4.06 / 3.96x`
- STAR/lean backward median ratios: `3.18 / 2.56 / 2.73 / 2.69x`

The STAR comparison is still only the existing small-MPS speed gate, not quality/capacity parity.

## Decision

Keep the lean owner-run recompute fork as the first positive shader cut after the ownerkeep/side-stream rejects. The clean repeat promotes the shader-side recompute change for this local owner-run-nomid path. It does not solve storage/prep scaling: selected tape storage still grows slightly worse than frame count (`9.65x` over an 8x frame increase), so the next useful fork should target owner-run tape construction/storage rather than another local RGB-only VJP array trim.
