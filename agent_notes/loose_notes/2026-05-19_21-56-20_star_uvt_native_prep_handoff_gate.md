# STAR UVT Native Prep Handoff Gate

Date: 2026-05-19

## Why

The matched 512px native handoff gate showed the important split:
`logit_handoff_reduce_vec4` had the best native backward (`386.26ms`), but the
Torch-side image prep was almost as expensive (`421.89ms`). That made the next
gate straightforward: keep the existing fast reverse traversal and move only
the linear sigmoid-MSE prep to Metal.

## What Changed

- Added `linear_sigmoid_mse_logit_handoff_prep(...)` in the STAR UVT feature
  rasterize wrapper.
- Added a `linear_sigmoid_mse_handoff_prep` Torch op and Metal kernel.
- Let `direct_logit_handoff_backward(...)` accept `grad_logits_layout="thw3"` so
  native prep can feed the reverse pass without a Python permute/copy.
- Added benchmark modes:
  `logit_handoff_native_prep`, `logit_handoff_reduce_native_prep`, and
  `logit_handoff_reduce_vec4_native_prep`.

This is intentionally benchmark-only. It does not yet wire into the trainer or
support hidden `FeatureToColor`.

## Results

Built the STAR UVT extension successfully with:

```bash
( cd third_party/fast-mac-gsplat/variants/star_uvt_v0
  rtk uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

Tiny F4/F32 parity passed. Native prep output errors are tiny:
`prep_grad_logits <= 8.74e-11`, `prep_grad_alpha <= 9.32e-10`.

Serial matched timing at `64f/512px/8192t/F32`, warmup 3, repeat 5:

| Mode | Forward | Prep | Backward | Total |
| --- | ---: | ---: | ---: | ---: |
| Torch prep `logit_handoff_reduce_vec4` | 620.18ms | 413.64ms | 412.72ms | 1446.53ms |
| Native prep `logit_handoff_reduce_vec4_native_prep` | 679.52ms | 37.29ms | 391.69ms | 1108.50ms |

Native prep scale rows:

| Res | Forward | Prep | Backward | Total |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 29.23ms | 3.00ms | 124.32ms | 156.55ms |
| 256 | 149.51ms | 12.09ms | 272.15ms | 433.75ms |
| 512 | 679.52ms | 37.29ms | 391.69ms | 1108.50ms |

## Read

This validates the specific hypothesis that the 512px handoff was paying a
removable Torch prep tax. Prep plus backward drops from `826.35ms` to
`428.98ms`; total drops about `23%` even though forward averaged slightly
slower in this serial rerun.

It is still not the final STAR UVT fast trainer path. The current real
objective uses target-grid V-JEPA and hidden/frozen-probe visual losses, while
this gate only covers linear sigmoid-MSE. The next useful gate is native hidden
RGB/loss prep or a visibility/prefix tape for the selected objective.

## Validation

- `py_compile` passed for the modified Python files.
- Extension rebuild passed.
- Tiny parity and serial timing JSONs pass and have zero overflow.

Report:
`outputs/benchmarks/2026-05-19_star_uvt_native_prep_handoff_gate.md`
