# STAR UVT Selected Visual Quality Gate

Date: 2026-05-20 01:31 +07

## Goal

After the shader diagnostic phase and matched dynamic-gsplat smoke, evaluate
whether the selected compact STAR UVT visual route is good enough to scale.

## Evidence

Selected route:

`src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_compact_currentbuild_from1500_lr001_50step_media.jsonc`

Result:

`outputs/benchmarks/2026-05-20_star_uvt_feature_targetgrid_sparsevisual_targetarea64_compact_currentbuild_from1500_lr001_50step_media.json`

Media inspected:

- `outputs/media/2026-05-20_star_uvt_feature_targetgrid_sparsevisual_targetarea64_compact_currentbuild_from1500_lr001_50step_contact.jpg`
- `outputs/media/2026-05-20_star_uvt_feature_targetgrid_sparsevisual_targetarea64_compact_currentbuild_from1500_lr001_50step_probe_contact.jpg`

The selected route passes mechanics:

- pass true
- zero tile overflow
- mean/last step `930.62/924.55ms`
- mean/last backward `581.32/578.04ms`
- feature target loss `0.625418 -> 0.625345`
- sparse visual loss `0.270538 -> 0.247529`
- RGB probe PSNR `22.028 -> 22.045`

But it fails visual quality:

- dense full RGB PSNR `6.023`
- sparse visual PSNR `6.064`
- RGB STAR same-clip bracket is `12.444`
- dense contact sheet output is sparse/streaked
- RGB-probe contact sheet is still blurry

## Decision

Do not scale this route to the 300-video set yet. The selected compact route is
the right fast helper for local iteration, but it is not a quality route.

The next experiment should change the visual objective/support/model bridge.
More colorizer atomics work is not the lever. The dynamic-gsplat smoke is also
not the fast escape hatch at this scale.

Report:
`outputs/benchmarks/2026-05-20_star_uvt_selected_visual_quality_gate.md`
