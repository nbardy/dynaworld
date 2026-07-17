# STAR UVT native handoff matched 512 gate

## Goal

Continue the STAR UVT fast feature-shader plan by checking whether the existing
native handoff prototypes still look useful at the active `64f/512px/8192t/F32`
scale. Previous direct-kernel handoff evidence was mostly `256px/32768t`, which
left a scale mismatch against the current cached V-JEPA target-grid route.

## Runs

All runs used:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 rtk .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/direct_feature_kernel_benchmark.py \
  --feature-dims 4,32 --timing-frames 64 --timing-size 512 \
  --timing-tubes 8192 --timing-feature-dim 32 \
  --backward-mode <mode> --timing-warmup 1 --timing-repeat 3 \
  --out-json outputs/benchmarks/2026-05-19_star_uvt_feature_direct_metal_<mode>_matched_64f_512_8192_f32.json
```

Artifacts:

- `outputs/benchmarks/2026-05-19_star_uvt_feature_direct_metal_gradcache_matched_64f_512_8192_f32.json`
- `outputs/benchmarks/2026-05-19_star_uvt_feature_direct_metal_gradcache_reduce_vec4_matched_64f_512_8192_f32.json`
- `outputs/benchmarks/2026-05-19_star_uvt_feature_direct_metal_gradcache_skip_feature_grad_matched_64f_512_8192_f32.json`
- `outputs/benchmarks/2026-05-19_star_uvt_feature_direct_metal_fused_first3_sigmoid_mse_matched_64f_512_8192_f32.json`
- `outputs/benchmarks/2026-05-19_star_uvt_feature_direct_metal_linear_sigmoid_mse_matched_64f_512_8192_f32.json`
- `outputs/benchmarks/2026-05-19_star_uvt_feature_direct_metal_logit_handoff_reduce_vec4_matched_64f_512_8192_f32.json`
- Summary report:
  `outputs/benchmarks/2026-05-19_star_uvt_native_handoff_matched_512_gate.md`

## Results

| Mode | Pass | Forward ms | Prep ms | Backward ms | Total ms |
| --- | --- | ---: | ---: | ---: | ---: |
| `gradcache` | true | `1150.70` | `0.00` | `522.02` | `1672.72` |
| `gradcache_reduce_feature_grad_vec4` | true | `673.79` | `0.00` | `627.11` | `1300.90` |
| `gradcache_skip_feature_grad` | true | `692.25` | `0.00` | `533.42` | `1225.67` |
| `fused_first3_sigmoid_mse` | true | `658.49` | `0.00` | `494.09` | `1152.58` |
| `linear_sigmoid_mse` | true | `748.37` | `0.00` | `918.09` | `1666.46` |
| `logit_handoff_reduce_vec4` | true | `637.12` | `421.89` | `386.26` | `1445.26` |

All rows passed tiny F4/F32 parity, finite timing checks, and zero
overflow/unstable-tile checks.

## Read

The narrow fused-first3 path remains a useful proof that moving RGB/MSE VJP
inside the native backward can help at the matched 512px scale, but it is not a
learned `FeatureToColor` route. Generalized linear in-kernel VJP is still not
promotable: it passes parity but is slower than gradcache on backward and the
linear decoder is visually weak in the trainer. The most useful signal is
`logit_handoff_reduce_vec4`: the native backward is only `386.26ms`, but the
Torch prep to produce `grad_logits` and `grad_alpha_handoff` costs `421.89ms`.

Next implementation should fuse that prep or build a visibility/prefix tape,
then compare against both this native handoff table and the current
sparse-forward batched target/probe trainer baseline.

## Validation

The direct-kernel benchmark itself ran tiny parity for `F=4` and `F=32` before
each timing row. Full validation/doc checks are part of the active thread goal
after docs are updated.
