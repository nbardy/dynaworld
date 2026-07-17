# STAR UVT Target-Area Native Gate

Date: 2026-05-19

## Goal

Move the expensive full-cell8 target-area hidden64 sparse-visual VJP out of
Python/Torch feature materialization and into a compact native STAR UVT path,
then benchmark it before treating it as a trainer option.

## What Changed

- Added a bin-only native feature-tube bridge:
  `bin_feature_tubes` / `bin_uvt_feature_tubes`.
- Added native cached-bin target-area hidden64 forward sums and backward:
  `sparse_hidden_sigmoid_target_area_forward_sums_cached_bins` and
  `direct_hidden_sigmoid_target_area_backward_cached_bins`.
- Added
  `research_experiments/star_uvt_feature_tubes/sparse_hidden_target_area_kernel_benchmark.py`.
- Added first-class trainer mode
  `native_hidden64_target_area_star_only`.
- Added matched 5-step trainer config:
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_fullcell8_nativehidden_from1500_lr001_5step_media.jsonc`.

## Results

Benchmark report:
`outputs/benchmarks/2026-05-19_star_uvt_sparse_hidden_target_area_native_gate.md`

Tiny parity passes against CPU autograd for F32/H64. The largest tiny error is
`2.24e-08` on q/opacity gradients.

Synthetic timing says the path is support-dependent:

- `8f/64px/1024t/H64`: native loses (`28.12ms` vs `25.74ms`).
- `64f/128px/8192t/H64`: native wins (`386.55ms` vs `509.50ms`, `1.32x`).
- `64f/256px/8192t/H64`: native wins clearly (`620.70ms` vs `1405.69ms`,
  `2.26x`).
- `64f/512px/8192t/H64`: native-only full support passes at `1874.41ms`; the
  all-at-once Torch baseline OOMs while trying another 4 GiB hidden-grad
  allocation.

The first-class trainer gate passes:

```text
outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_nativehidden_from1500_lr001_5step_media.json
wandb/offline-run-20260519_230044-yk6grg2f
```

Compared with the matched manual hidden64 star-only full-cell8 row, native
target-area star-only cuts mean step `5801.7ms -> 3496.0ms` and last step
`6007.7ms -> 3206.3ms`. It preserves the same endpoint class: sparse visual
PSNR `5.746`, full RGB PSNR `5.648`, no colorizer gradients, zero overflow.

The split changed as expected: native removes the expensive sparse render plus
Python hidden loss construction (`609.3 + 3405.1ms` becomes
`75.0 + 863.6ms`), while native backward grows (`1027.7 -> 2056.1ms`) because
it recomputes hidden/color state while reversing STAR.

## Read

This is the first positive full-support target-area native gate. It fixes the
speed/memory shape of the previous Python full-cell8 diagnostic, but it does
not fix visual quality. The right next move is to use this native target-area
path as the speed baseline for any dense visual-support work, then either:

1. reduce the native backward recompute cost, or
2. change the visual objective/support so dense RGB quality actually improves.

Do not promote this as the selected final overfit route yet. The selected fast
cached-V-JEPA helper is still sparse-forward batched VJP when the objective is
feature/probe movement; this target-area native path is the new full-support
visual-VJP candidate.

## Validation

- `py_compile` passed for the trainer and benchmark.
- `star_uvt_v0` extension rebuilt cleanly.
- Native trainer config resolved.
- `tests/test_star_uvt_feature_rgb_probe.py` and
  `tests/test_star_uvt_feature_target_adapter.py` passed: `32 passed`.
- Post-doc-sync checks passed:
  - `git diff --check`
  - `git -C third_party/fast-mac-gsplat diff --check`
