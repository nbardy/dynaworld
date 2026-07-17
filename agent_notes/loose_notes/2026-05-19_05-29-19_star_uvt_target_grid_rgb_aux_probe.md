# STAR UVT Target-Grid RGB-Aux Probe

Date: 2026-05-19 05:29 +0700

## Goal

The pure target-grid media gate proved feature-space overfit and media plumbing,
but not RGB quality because `rgb_loss_weight=0`. This probe adds an RGB
auxiliary loss so the colorizer trains and the output JSON can report feature
loss and RGB loss separately.

## Implementation

- Added per-component loss logging to
  `src/train/train_star_uvt_feature_overfit.py`:
  `rgb_losses`, `feature_target_losses`, start/end RGB loss, start/end feature
  target loss, and RGB PSNR computed from RGB MSE rather than total mixed loss.
- Added config
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbaux1_lr005_20step_media.jsonc`.
- The config keeps `feature_target.materialization=target_grid`,
  `feature_target.loss_weight=1.0`, and sets `rgb_loss_weight=1.0`.

## Command

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbaux1_lr005_20step_media.jsonc
```

W&B offline run: `wandb/offline-run-20260519_052919-91df79vw`.

## Result

- output:
  `outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbaux1_lr005_20step_media.json`
- pass: `true`
- total loss: `1.338106 -> 1.332599`
- feature target loss: `0.999935 -> 0.997336`
- RGB loss: `0.338171 -> 0.335263`
- RGB PSNR: `4.709 -> 4.746`
- colorizer gradients present
- target grid: `[32,32,16,16]`, `1.0MiB`
- target load/prep: `254.69ms`
- mean step: `1999.63ms`
- mean render forward: `586.33ms`
- mean target/loss: `51.75ms`
- mean backward: `1113.59ms`
- zero tile overflow; max tile count `33`, p95 `18`
- media:
  `outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbaux1_lr005_20step_contact.jpg`
  and
  `outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbaux1_lr005_20step_sbs.mp4`

## Interpretation

The RGB auxiliary path is trainable and the new component logging is necessary:
the total mixed loss alone would hide whether RGB or V-JEPA target loss moved.
Both component losses decrease, and the colorizer receives gradients.

This is not enough visual quality improvement. RGB PSNR only improves by
`0.0375dB` in 20 steps, ending at `4.746`, while the row slows to about
`2.000s/step`. The next visual gate should be stronger or longer: a larger RGB
auxiliary weight/schedule, a warm-started colorizer, or a trained/frozen
feature-to-RGB probe. Keep the pure target-grid row as the speed/memory
diagnostic.

## Updated Reports

- `outputs/benchmarks/2026-05-19_star_uvt_vjepa_vs_gaussian_comparison.md`
  now includes the RGB-aux1 target-grid row.
- `outputs/benchmarks/2026-05-19_star_uvt_precomputed_vjepa_bridge_audit.md`
  now records the RGB-aux1 probe and the conclusion that it is a control, not a
  quality fix.
- `BASELINES.md`, `README.md`, `PROJECT_INDEX.md`, `EXPERIMENTS.md`,
  `TODO/README.md`, the feature-tube README, the fast-shader plan, and
  `agent_notes/key_learnings.md` were updated.
