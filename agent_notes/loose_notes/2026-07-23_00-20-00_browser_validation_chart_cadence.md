# Browser validation chart cadence

- The train/heldout curves looked polygonal because deterministic validation
  was requested only every ten seconds. A 34k-step screenshot had only a
  handful of PSNR/SSIM samples, joined by long straight chords.
- Validation now targets a 1.5-second cadence. The existing pending flag still
  permits only one snapshot/evaluation at a time, so this does not accumulate
  work or block the optimizer; validation remains in its dedicated worker.
- The asset version was bumped so existing tabs load the new cadence. Chart
  strokes use round caps/joins but retain measured piecewise-linear values; no
  spline interpolation or fabricated points were introduced.
- Live Apple browser QA retained about `878 steps/s`, `60 FPS` UI, and showed
  dense train/heldout loss, PSNR, and SSIM histories by roughly step 28k.
