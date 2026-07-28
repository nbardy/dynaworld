# Browser metric history and sampler defaults

- Removed the browser chart's 180-point rolling-window truncation. Loss, PSNR,
  and SSIM proxy charts now retain the complete current run and label their
  full step extent.
- Reorganized reconstruction metrics into a four-card Loss/MAE/PSNR/SSIM grid,
  with train and heldout values in each card. Added the heldout MAE value that
  the multicamera validator already returned but the UI did not expose.
- Found a sampler-control mismatch: the UI displayed a capped effective motion
  share while the train call passed the uncapped slider value. The previous
  95% motion plus 8% static settings therefore executed as 95% motion and 5%
  static. The UI and train call now agree on the trainer's 90% motion, 8%
  static, and 2% uniform default.
- Browser smoke reached step 2176 with full-history labels (`0-2176` loss,
  `0-2048` validation metrics), populated all four metric cards, and emitted no
  console or WebGPU validation errors.
- These controls are demo heuristics, not paper-ablation results. Motion/static
  mix biases ray selection; support guard is a motion-ray alpha-coverage target;
  temporal support is the normalized-time Gaussian visibility width.
