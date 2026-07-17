# STAR UVT Full-Cell Visual Gate

Date: 2026-05-19

## Context

The phased target-area64 gate rejected fixed sparse support position as the
visual-quality blocker. The next bounded question was whether STAR UVT was
missing the dynamic-gsplat-style dense support trick: push gradients through
every rendered pixel inside each target-area cell, then reduce against compact
area targets.

## Implementation

Added and ran:

`src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_fullcell8_from1500_lr001_5step_media.jsonc`

The run resumes from:

`outputs/checkpoints/2026-05-19_star_uvt_feature_targetgrid_sparseforward_batchedvjp_lr001_resume50_from1450_lr005sparse_1500step.pt`

Settings:

- `frames=64`, `size=512`, `tubes=8192`
- `feature_target_image_vjp_mode=analytic_sparse_grid_forward_batched`
- `render_mode=feature_direct_gradcache_reduce_vec4`
- `sparse_visual.pixel_source=stratified_patch_grid`
- `sparse_visual.loss_basis=target_area_mean`
- `sparse_visual.sample_grid_shape=[64,64,64]`
- `sparse_visual.patch_shape=[8,8]`
- `rgb_probe_loss_weight=40`, `feature_target_loss_weight=1`,
  `sparse_visual_loss_weight=1`

This means one full step renders all `16,777,216` dense pixels as sparse visual
support, but still compares them through `262,144` target-area cells.

## Runs

The first strict run used W&B offline id `ir5cof8q` and exited nonzero because
the loss-decrease requirement correctly failed. The row had already written
the negative metrics and media.

The config was then changed to `require_loss_decrease=false` so the nonpassing
result could be recorded cleanly. The final recorded W&B offline id is
`kkeofuxf`.

Result JSON:

`outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_from1500_lr001_5step_media.json`

Report:

`outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_from1500.md`

Media:

- `outputs/media/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_from1500_lr001_5step_contact.jpg`
- `outputs/media/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_from1500_lr001_5step_probe_contact.jpg`
- `outputs/media/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_from1500_lr001_5step_sbs.mp4`
- `outputs/media/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_from1500_lr001_5step_probe_sbs.mp4`

## Result

This gate is negative:

- `pass=false`
- weighted loss worsens `1.1429818263 -> 1.1491296147`
- feature target loss worsens `0.6254179478 -> 0.6268119812`
- frozen RGB-probe PSNR worsens `22.027719 -> 21.860199`
- sparse visual loss improves `0.2667866340 -> 0.2616782044`
- sparse visual PSNR improves `5.738359 -> 5.822324`
- dense full RGB PSNR is only `5.722436`
- mean step/backward/render is `7526.73 / 6569.12 / 109.33 ms`
- sparse visual render/loss/backward is `456.50 / 5702.60 / 746.57 ms`
- zero tile overflow, max/p95 tile splats `68 / 46`

## Read

Full dense pixel support through the current sparse-pixel path is the wrong
route. It improves the sampled visual loss, but the actual feature/probe/total
objectives and dense RGB media move backward. It is also dominated by the
Python/Torch loss construction path: `5.70s` of the `7.53s/step` mean is sparse
visual loss materialization/reduction, while sparse visual render is only
`456.5ms` and sparse visual backward is `746.6ms`.

The current next move should be a fused visibility/prefix tape or fused
RGB/loss/gradient path that avoids materializing all dense RGB samples through
Python-side tensors. Do not spend more time rearranging sparse support patterns
until there is a real fused dense visual-gradient route.
