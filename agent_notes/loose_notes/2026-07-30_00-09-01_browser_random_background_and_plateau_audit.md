# Browser Random Background And Plateau Audit

## Why this changed

The tiled browser trainer still composited optimizer predictions over exact
black. A live plateaued run at step 50,544 reported:

- 59.9% mean train alpha coverage;
- 57.7% heldout alpha coverage;
- 4,096 active slots out of 4,096;
- zero topology operations;
- 16% of splats at the 6:1 scale-aspect cap.

Roughly 40% residual transmittance therefore exposed literal black during
training. This was a real opacity/color shortcut, not just a dark preview.

## Implementation

- The SPA exposes a default-on `Random Train BG` toggle for tiled full-frame
  training. The sampled-ray control leaves it disabled.
- JavaScript hashes the step once, quantizes one RGB value to 10 bits per
  channel, and packs it with an enable bit into the previously unused config
  word at byte offset 124. No per-pixel hash or extra GPU buffer was added.
- Tiled forward adds `T_end * background` to accumulated RGB while preserving
  output alpha as `1 - T_end`.
- Validation, snapshots, and live preview remain black-backed.
- Backward needs no new branch: reconstructing the suffix from final rendered
  RGB automatically includes the same background.
- The low-level trainer and worker defaults remain fixed black. The SPA passes
  the checked option explicitly, which preserves controlled benchmark/parity
  callers.

No camera image, temporal mean, or heldout pixel enters the background. A
camera-specific background would be a 2D novel-view leak.

## Verification

The Node browser-trainer suite passed 80/80 after the change.

The live Apple WebGPU parity harness was extended to use random train
compositing in both GPU and CPU finite differences. It passed with:

- forward RGB max absolute error: `1.19e-7`;
- objective absolute error: `7.50e-7`;
- 9/9 active gradient families passing.

The largest rotation-gradient candidate crossed the hard `q <= 9` raster
support boundary, making its finite difference unstable. The harness now
rejects unstable probes and selects the highest-magnitude stable candidate in
the same family. It did not loosen error tolerances; the selected rotation
gradient matched at `9.42e-8` absolute error.

## What this does not prove

This removes one objective mismatch. It does not prove that a long run will
escape the plateau. The next matched experiment should compare black versus
random training with identical initialization, camera/time order, step budget,
and black-eval metrics.

If random backgrounds improve coverage but not structure, the next tight
intervention is fixed-budget residual relocation: recycle low-contribution
slots into high-residual, low-coverage train regions at predetermined steps.
Do not add a coverage penalty in the same experiment; that would hide which
intervention caused the change and could inflate large opaque floaters.
