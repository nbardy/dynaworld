# STAR UVT Manual Linear VJP Gate

Date: 2026-05-19

## Original Goal

The thread goal was to turn the STAR UVT performance concern into a measured
execution plan: rerun the relevant STAR UVT, dynamic-gsplat, and feature-splat
benchmarks; break backward into real buckets; identify whether the rasterizer,
feature loss, data loader, or feedforward model was slow; port only the fast
feature-shader tricks that survive measurement; and keep the docs updated after
each gate.

By this point the selected fast STAR UVT feature/probe path was already
`sparse-forward + batched target/probe VJP` on the cached V-JEPA target-grid
objective. The open side question was whether full-cell8 sparse visual RGB
support could become practical if the Python/Torch hidden64 `FeatureToColor`
loss-side VJP was simplified.

## What Changed

- Added `sparse_visual.loss_vjp_mode=manual_linear` to
  `src/train/train_star_uvt_feature_overfit.py`.
- Added a strict `_linear_colorizer_layer(...)` validation path: no hidden
  layer, no pre-norm, no view conditioning, single 1x1 linear conv, sigmoid or
  identity activation.
- Added manual linear sparse visual RGB loss/VJP that computes the linear
  logits, sigmoid, alpha-weighted RGB, target-area loss, colorizer parameter
  grads, feature grads, and alpha grads without autograd through a hidden64
  colorizer.
- Added focused parity coverage in
  `tests/test_star_uvt_feature_target_adapter.py`.
- Extended
  `research_experiments/star_uvt_feature_tubes/sparse_visual_loss_vjp_profile.py`
  to profile `manual_linear` as well as hidden64 modes.

## Results

Linear target-grid RGB probe:

- Config:
  `src/train_configs/star_uvt_feature_targetgrid_vjepa_rgb_probe_linear_lr01_1000step.jsonc`
- Result:
  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_vjepa_rgb_probe_linear_lr01_1000step.json`
- W&B offline run: `09x636sp`
- Checkpoint:
  `outputs/checkpoints/2026-05-19_star_uvt_feature_targetgrid_vjepa_rgb_probe_linear_lr01_1000step.pt`
- Pass: true
- Grid loss `0.058400 -> 0.020778`
- Final grid PSNR `16.824`; final full-video upsample PSNR `16.980`
- Mean step `3.21ms`, backward `1.24ms`

Full-cell8 sparse visual trainer gate:

- Config:
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_fullcell8_linear_from1500_lr001_5step_media.jsonc`
- Result:
  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_linear_from1500_lr001_5step_media.json`
- Report:
  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_linear_from1500.md`
- W&B offline run: `jzewxvlh`
- Pass: true as a narrow mechanics gate
- Mean step/backward/render `2064.4/1230.2/93.7ms`
- Sparse visual render/loss/backward `458.9/383.3/749.9ms`
- Weighted loss `1.142108 -> 1.141392`
- Feature loss `0.625418 -> 0.625434`
- Frozen probe PSNR `22.028 -> 22.029`
- Sparse visual PSNR `5.753 -> 5.763`
- Final full RGB PSNR `5.668`

Manual linear subphase profile:

- Report:
  `outputs/benchmarks/2026-05-19_star_uvt_sparse_visual_manual_linear_subphase_profile_fullstep.md`
- Full-step extrapolated sparse render `484.9ms`
- Full-step extrapolated loss VJP `522.4ms`
- Full-step extrapolated pixel-id build `124.8ms`
- Largest loss-side phases: `linear_param_feature_grad_ms` `189.6ms`,
  `target_area_loss_grad_pred_ms` `158.4ms`, `linear_fc_ms` `91.5ms`

## Read

Manual linear is the first full-cell8 result that makes dense visual support
look mechanically affordable: it cuts the manual hidden64 row from `6414.0ms`
to `2064.4ms` per step and sparse visual loss construction from `3803.6ms` to
`383.3ms`.

It is still not the route to promote. The linear decoder is much weaker than
the hidden64 target-grid oracle (`16.98` full PSNR versus `20.07`), feature
loss slightly worsens in the STAR trainer, and dense full RGB remains only
`5.668`. The lesson is not "use linear"; the lesson is that hidden64
Python-side colorizer/loss VJP was a large avoidable cost, while visual capacity
still matters.

## Current State

Recorded and checked now:

- focused tests pass: `32 passed`
- `py_compile` passes for the STAR UVT trainer/profile/comparison files
- linear probe, linear full-cell8 benchmark, and manual-linear profile all have
  JSON/markdown artifacts

Still active:

- Update/re-run the normalized comparison report with the linear row.
- Keep the selected speed baseline as sparse-forward batched target/probe VJP,
  not full-cell8 dense support.
- Next useful implementation work is native/fused RGB/loss/VJP or
  visibility/prefix tape, plus objective/model changes for visual quality.
