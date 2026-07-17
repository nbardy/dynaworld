# STAR UVT Native Target-Area Hidden32 Gate

Date: 2026-05-19

## Goal

Try the first concrete follow-up after the native target-area hidden64 gate:
reduce native reverse recompute by running the same full-cell target-area path
with the hidden32 RGB probe.

## What Changed

- Added generic alias `native_hidden_target_area_star_only`.
- Kept `native_hidden64_target_area_star_only` as a compatibility alias.
- Added config:
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_fullcell8_nativehidden32_from1500_lr001_5step_media.jsonc`.

## Results

Report:
`outputs/benchmarks/2026-05-19_star_uvt_native_target_area_hidden32_gate.md`

Kernel parity passes. The 512px native-only H32 path is faster than H64:
`1874.41 -> 1331.00ms` total and `1534.45 -> 1103.68ms` backward.

The trainer row is faster but rejected:

```text
outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_nativehidden32_from1500_lr001_5step_media.json
wandb/offline-run-20260519_231211-l0shgcx9
```

Compared with native hidden64 target-area star-only:

- mean step: `3496.0 -> 2464.6ms`
- sparse visual backward: `2056.1 -> 1321.7ms`
- sparse visual loss construction: `863.6 -> 662.0ms`
- final full RGB PSNR: `5.648 -> 5.632`
- final probe PSNR: `22.029 -> 19.481`
- pass: `true -> false`

## Read

Hidden32 confirms native reverse recompute can be reduced by shrinking the
decoder, but the capacity loss is too expensive for this gate. Do not promote
hidden32 native target-area. Keep hidden64 native target-area as the current
full-support visual-VJP speed baseline and look next for either hidden64
recompute reduction or a different visual objective/support.

## Validation

- `py_compile` passed for trainer and benchmark.
- Hidden32 native config resolved.
- `tests/test_star_uvt_feature_rgb_probe.py` and
  `tests/test_star_uvt_feature_target_adapter.py` passed: `32 passed`.
- Post-doc-sync checks passed:
  - `git diff --check`
  - `git -C third_party/fast-mac-gsplat diff --check`
