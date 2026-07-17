# STAR UVT probe-init RGB-aux bridge negative

Date: 2026-05-19 18:28

## What changed

Added an opt-in trainer bridge so we can resume the STAR feature model but not
the old trainable colorizer:

- `train.resume_colorizer=false`
- `colorize.init_checkpoint=<checkpoint with colorizer state>`
- guardrail: cannot skip checkpoint colorizer while also resuming the old
  optimizer state

Focused test:

`PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_star_uvt_feature_target_adapter.py -q`

Result: `12 passed`.

## Gate

Ran a 20-step quality diagnostic from the selected sparse-forward 1500
checkpoint:

`src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_autograd_rgbaux1_probeinit_from1500_20step_media.jsonc`

This resumes the model from:

`outputs/checkpoints/2026-05-19_star_uvt_feature_targetgrid_sparseforward_batchedvjp_lr001_resume50_from1450_lr005sparse_1500step.pt`

It initializes a hidden64 trainable colorizer from the standalone target-grid
RGB probe:

`outputs/checkpoints/2026-05-19_star_uvt_feature_targetgrid_vjepa_rgb_probe_hidden64_lr01_1000step.pt`

It uses `image_vjp_mode=autograd` because the selected sparse-forward batched
VJP path correctly rejects full-resolution RGB reconstruction loss.

## Result

Negative promotion result:

- pass: false
- weighted loss: `1.148821 -> 1.146293`
- feature loss: `0.625418 -> 0.626799`
- RGB loss: `0.272626 -> 0.259968`
- RGB PSNR: `5.644 -> 5.851`
- probe loss: `0.006269 -> 0.006488`
- probe PSNR: `22.028 -> 21.879`
- mean step/back/render: `5206.6 / 2959.4 / 1117.7 ms`
- zero overflow, max/p95 `68/46`

Compared with the selected sparse 1500 endpoint, this is `16.48x` slower on mean
step, worsens feature loss by `+0.001371`, and gives back `0.149 dB` probe PSNR.

## Visual read

The trainable-colorizer contact sheet is not a sharper version of the dog video.
It collapses into high-frequency black/white grid-like artifacts while scalar
RGB MSE decreases. The frozen-probe contact sheet remains blurry and slightly
regresses.

## Interpretation

The target-grid RGB probe is not directly reusable as a full-resolution decoder.
It was trained on `[T, F, 16, 16]` target-grid features, but the trainable STAR
colorizer sees full-resolution rendered feature images with a different
distribution. A future bridge needs to train on the rendered feature-image
distribution, add a multiscale visual target, or implement a native sparse visual
loss. Do not promote the autograd RGB-aux probe-init path.

## Artifacts

- Report:
  `outputs/benchmarks/2026-05-19_star_uvt_autograd_rgbaux_probeinit_from1500_negative.md`
- Result:
  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_autograd_rgbaux1_probeinit_from1500_20step_media.json`
- Checkpoint:
  `outputs/checkpoints/2026-05-19_star_uvt_feature_targetgrid_autograd_rgbaux1_probeinit_from1500_1520step.pt`
- Trainable-colorizer media:
  `outputs/media/2026-05-19_star_uvt_feature_targetgrid_autograd_rgbaux1_probeinit_from1500_20step_contact.jpg`
- Frozen-probe media:
  `outputs/media/2026-05-19_star_uvt_feature_targetgrid_autograd_rgbaux1_probeinit_from1500_20step_probe_contact.jpg`
