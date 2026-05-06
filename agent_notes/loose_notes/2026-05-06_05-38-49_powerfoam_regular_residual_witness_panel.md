# PowerFoam Regular Residual Witness Panel

Date: 2026-05-06

Scope: close the missing A1-style diagnostic for the selected clean DeepView
regular-triangulation PowerFoam row. This is diagnostic-only; no training was
run.

## Code Change

Extended
`research_experiments/dynamic_foam/diagnose_powerfoam_heldout_error.py` with:

- heldout-only mode to avoid rerendering all train views when we only need
  heldout evidence
- optional normal-distance rendering from the existing model forward path
- pixel-level residual buckets by alpha and high-residual mask
- a decoded-geometry ray/sphere support proxy for the worst heldout sample
- a saved panel with columns
  `GT | render | alpha | residual_l1 | normal_distance | log_support_hit_count | nearest_power_support`

The support panel is a proxy from decoded centers/radii and heldout rays. It is
not true COLMAP per-track support, because the current PLY and checkpoint do
not preserve per-point track ids or unique-camera counts for sampled cells.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/dynamic_foam/diagnose_powerfoam_heldout_error.py
```

The selected config uses `regular_triangulation`, so the diagnostic must run
with SciPy:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with scipy python \
  research_experiments/dynamic_foam/diagnose_powerfoam_heldout_error.py \
  src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_appearanceonly_wandboffline_init_raytrace_regular_128_16f_1024cells_40step_noaux.jsonc \
  --batch-size 4 \
  --heldout-only
```

Outputs:

```text
outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_appearanceonly_wandboffline_init_raytrace_regular_128_16f_1024cells_40step_noaux/heldout_error_diagnostics.json
outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_appearanceonly_wandboffline_init_raytrace_regular_128_16f_1024cells_40step_noaux/heldout_error_diagnostics_panel.png
```

The panel is `896 x 128`, matching seven 128px columns.

## Result

The selected regular heldout row is not failing because the heldout view is
blank or unsupported:

- heldout PSNR / SSIM / L1 remain `12.5099 / 0.1169 / 0.1794`
- heldout alpha mean: `0.9776`
- alpha `> 0.9` pixel fraction: `0.9708`
- alpha `< 0.05` pixel fraction: `0.0174`
- selected worst sample: `camera_0040`, frame `0`
- worst-frame support-hit fraction: `0.99994`
- high-residual support-hit fraction: `0.99969`
- mean support hit count on worst frame: `13.05`

The dominant residual bucket is high-alpha pixels:

- alpha `>= 0.5` pixels contain `95.67%` of total residual
- top-20%-residual alpha `>= 0.5` pixels contain `43.61%` of total residual
- that high-alpha/high-residual bucket has mean L1 `0.4301`
- low-alpha/high-residual pixels contribute only `3.69%` of total residual

## Interpretation

The immediate paper-quality blocker has moved past blank coverage. Regular
topology and near-plane starts give almost opaque heldout renderings with
nonzero geometric support, but the rendered structure is still wrong enough to
miss SSIM. The next useful mechanisms should target spatial/depth alignment,
normal/material transport, or a heldout-improving objective. Another
coverage-only support patch is unlikely to close the gate.
