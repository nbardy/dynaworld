# Browser Background, Seed, And Preview Follow-Up

The user reported that the browser result remained blurrier than historical
TokenGS/STAR/World-Tube overfits and that splats looked like gridded circles.

The historical comparison was mostly apples-to-oranges. The strongest TokenGS
and STAR rows are one-source-view memorization with 8,192-32,768 primitives;
STAR's video-sample initializer also starts on target pixels with target RGB.
The browser shares at most 4,096 3D primitives across 17 train cameras and
holds out cam06. The closest recorded multicamera World-Tube result is much
weaker than the headline source-view rows.

Three concrete browser issues were still actionable:

- the recent 2,048-plus-growth default discarded half of the available 4,096
  XYZRGB seed scaffold and replaced it with deterministic duplicate splits;
- the display filter used the 72-pixel target height on a much taller canvas,
  turning subpixel anisotropic lobes into visibly round blobs;
- training composited over black while display cleared to near-black and drew
  alpha below the training raster threshold.

A shader audit then found a fourth and larger issue: with all 4,096 SfM slots
already active, proxy maintenance still replaced 16 slots every 512 steps. By
step 119,808 it had rewritten 3,744 slots with deterministic largest-axis
copies, reset their Adam moments against the global bias-correction step, and
created exactly the repeated chains visible in the preview. Post-fill recycling
is now disabled; lower-count runs may fill initially hidden capacity, then stop.

The SPA now defaults to all 4,096 seeds. Its live render uses the panel height
for anti-aliasing, exact black, and the training alpha threshold/cap. A
optional 2,048-step static warmup rotates through train-camera temporal means
with motion frozen, then restarts the real camera/time schedule. Heldout pixels
are never used by that warmup or composited as a background.

This does not make the current representation World Tubes or native 4DGS.
Residual/depth-guided births, view-dependent appearance, and a matched
independent-per-time oracle remain the important research work after the
corrected baseline is measured.

The full-frame motion-weighted objective was also tested rather than promoted
on intuition. At step 16,384, the standard objective reached `15.3/14.6 dB`
train/heldout, normalized 2x weighting reached `14.9/13.9 dB`, and an earlier
4x run reached `15.3/14.3 dB`. The implementation remains available as an
explicit ablation, but it is off by default.

After fixing topology maintenance and clamping RGB to the target range, the
6:1 run reached `16.2/15.5 dB` and `0.514/0.261` SSIM at step 16,384. A 12:1
aspect cap reached `16.3/15.4 dB` and `0.518/0.254`: a tiny train-only gain,
slightly worse heldout quality, and more raster work. The 6:1 guard remains.

Background remains fixed black and target alpha is one. Learned RGB had been
allowed to reach 1.4 even though decoded targets are clamped to 1; it is now
clamped to the target range so opacity cannot be traded for overbright color.
Random-background training was not added: it is a separate opaque-coverage
regularizer, not a free correctness fix. Softmax Splatting was also rejected as
the primary compositor because it solves 2D forward-warp collisions rather
than depth-ordered 3D transmittance.
