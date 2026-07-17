# STAR UVT Alpha-Only Visibility Profile

## Context

The dense-alpha support gate was quality-negative, but it also exposed an
implementation waste: the trainer rendered dense F32 feature images only to
use `render.alpha`, then sent a zero F32 `grad_feature_image` through the
feature backward.

This note records the follow-up diagnostic. It does not promote dense alpha as
a visual objective. It only checks whether alpha-only visibility/support
diagnostics can avoid dense `[T,H,W,F]` feature images.

## Code added

- Added `UVTFeatureAlphaRenderResult`.
- Added `render_uvt_feature_alpha_all_pixels_with_bins(...)` in
  `third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/feature_rasterize.py`.
- Added
  `research_experiments/star_uvt_feature_tubes/alpha_only_visibility_profile.py`.

The wrapper reuses the existing sparse-pixel Metal path with:

- all pixel ids for the current chunk,
- a dummy F1 feature tensor,
- cached tile bins returned for backward,
- F1 zero feature-gradient image for alpha-only backward.

This is intentionally not a new Metal ABI. It is a fast diagnostic shape using
the kernels we already have.

## Validation commands

```bash
PYTHONPATH=src/train rtk .venv/bin/python -m py_compile \
  third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/feature_rasterize.py \
  research_experiments/star_uvt_feature_tubes/alpha_only_visibility_profile.py
```

```bash
PYTHONPATH=src/train rtk .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/alpha_only_visibility_profile.py \
  --config src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_densealpha075_from1500_lr001_5step_media.jsonc \
  --max-chunks 1 \
  --out-json outputs/benchmarks/2026-05-20_star_uvt_alpha_only_visibility_profile_1chunk.json \
  --out-md outputs/benchmarks/2026-05-20_star_uvt_alpha_only_visibility_profile_1chunk.md
```

```bash
PYTHONPATH=src/train rtk .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/alpha_only_visibility_profile.py \
  --config src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_densealpha075_from1500_lr001_5step_media.jsonc \
  --out-json outputs/benchmarks/2026-05-20_star_uvt_alpha_only_visibility_profile_full.json \
  --out-md outputs/benchmarks/2026-05-20_star_uvt_alpha_only_visibility_profile_full.md
```

## Results

One-chunk smoke:

- pass true
- alpha parity max abs diff `0`
- dense cached vs alpha-F1 grad max abs:
  - ma `5.68e-12`
  - q `1.79e-7`
  - opacity `2.91e-11`
- dense-current render+backward `52.2ms`
- sparse-F1 alpha render+backward `23.9ms`
- ratio dense/alpha `2.186x`

Full 32 chunks:

- pass true
- alpha parity max abs diff `0`
- dense cached vs alpha-F1 grad max abs:
  - ma `3.85e-11`
  - q `4.62e-7`
  - opacity `4.51e-10`
- zero overflow
- max/p95 tile `68/53`
- dense-current render+rebin backward total `1100.8ms`
- dense cached F32 total `1067.4ms`
- sparse-F1 alpha render+cached F1 backward total `634.6ms`
- per-chunk means:
  - dense-current total `34.4ms`
  - dense cached total `33.4ms`
  - sparse-F1 alpha total `19.8ms`
- ratio dense-current/sparse-F1 `1.735x`
- per-chunk dense feature image estimate `67.1MiB`
- per-chunk sparse-F1 feature-values plus pixel ids estimate `4.2MiB`

## Interpretation

This is a useful implementation detail: if we need alpha-only visibility
diagnostics, do not render dense F32 feature images. Existing sparse-pixel F1
render plus cached F1 backward is correct and materially faster.

It does not change the dense-alpha quality result. The dense-alpha objective
still regressed weighted loss, feature loss, probe PSNR, dense RGB, and alpha
coverage. The next visual gate still needs a support-changing visibility/model
bridge, not another same-support alpha loss.

## Artifacts

- `outputs/benchmarks/2026-05-20_star_uvt_alpha_only_visibility_profile_1chunk.json`
- `outputs/benchmarks/2026-05-20_star_uvt_alpha_only_visibility_profile_1chunk.md`
- `outputs/benchmarks/2026-05-20_star_uvt_alpha_only_visibility_profile_full.json`
- `outputs/benchmarks/2026-05-20_star_uvt_alpha_only_visibility_profile_full.md`
