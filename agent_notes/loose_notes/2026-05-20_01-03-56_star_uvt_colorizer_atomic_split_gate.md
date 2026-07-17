# STAR UVT Colorizer Atomic Split Gate

Date: 2026-05-20 01:03 +07

## Goal

After native `target_area_colorizer_vec4_wt` proved correct but slower than
compact autograd, isolate whether the slowdown is STAR feature/geometry VJP or
the colorizer parameter-gradient path.

## Change

Exposed a benchmark-only native mode:

```text
target_area_colorizer_grad_only = 144
```

That combines colorizer-gradient mode bit `128` with skip-hidden-feature VJP bit
`16`. It computes hidden/logit backward and the colorizer parameter gradients,
but intentionally leaves STAR feature/geometry gradients at zero.

Files touched:

- `third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/feature_rasterize.py`
- `research_experiments/star_uvt_feature_tubes/sparse_hidden_target_area_kernel_benchmark.py`

No native rebuild was needed because this is an existing native bit combination.

## Gates

Tiny parity:

- command output JSON:
  `outputs/benchmarks/2026-05-20_star_uvt_native_target_area_colorizer_grad_only_tiny_gate.json`
- pass: true
- F32 checked max errors: hidden weight `1.75e-10`, hidden bias `9.31e-10`,
  output weight `5.82e-10`, output bias `0.0`, loss `0.0`
- STAR feature/geometry errors are ignored by design because this mode disables
  those gradients

Compact-support direct kernel timing:

```text
64f, 512px, 8192 tubes, F32, grid_side=64, patch_size=2,
tile_capacity=128, 6.25% dense pixel support, repeat=3, warmup=1
```

| Mode | Backward ms | Total ms | Overflow |
| --- | ---: | ---: | ---: |
| `target_area_star_only_vec4_wt` | `88.89` | `146.62` | `0` |
| `target_area_colorizer_grad_only` | `536.57` | `727.32` | `0` |
| `target_area_colorizer_vec4_wt` | `531.40` | `571.21` | `0` |

The full report lives at
`outputs/benchmarks/2026-05-20_star_uvt_native_target_area_colorizer_atomic_split_gate.md`.

## Interpretation

This pins the compact native colorizer failure on the parameter-gradient atomic
envelope. Colorizer-only backward is already as expensive as full colorizer
backward and about `6x` the star-only native backward. The next useful native
port is a colorizer-gradient reduction strategy, not another STAR W^T memory
shuffle.

Current practical decision remains unchanged:

- keep compact autograd as the selected visual route;
- keep native target-area as the exact full-support speed/memory baseline;
- only promote a compact native visual route if it preserves colorizer
  gradients and beats compact autograd in the trainer gate.
