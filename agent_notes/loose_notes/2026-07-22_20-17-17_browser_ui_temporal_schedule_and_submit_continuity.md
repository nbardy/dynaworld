# Browser UI Temporal Schedule And Submit Continuity

## Scope

This pass stayed inside the browser demo UI boundary. It did not change WGSL,
the Python trainer hierarchy, dataset contracts, or paper-protocol files.

## Decisions

- Preserve the corrected sampler semantics: 90% motion, 8% static, and 2%
  uniform. The earlier matched Static Mix 0% versus 8% run was a near-tie, so
  the UI does not present the static reserve as a demonstrated quality win.
- Preserve fixed temporal support 0.30 as the measured default. Earlier live
  probes found 0.18-0.22 starved motion support and found fixed 0.30 better than
  fixed 0.26 in the longer probe.
- Add an opt-in browser-demo heuristic that holds temporal support at 0.30
  through step 256, smoothstep-narrows to 0.26 through step 2048, and then
  holds. The control explicitly says `(demo)` and its tooltip says the schedule
  is unmeasured. It is not a paper or ablation claim.
- Keep manual `Step` metric readbacks synchronous, but make non-forced sample
  loss telemetry fire-and-forget behind one `lossReadBusy` guard. Metric points
  retain the step at which their readback was requested, and stale trainer
  epochs cannot update the current run.
- Wrap the three result labels and render canvas in one responsive strip. The
  canvas is 4:1, which gives three equal 4:3 camera panels without stretching.

## Verification

- `git diff --check` passed for `app.js`, `index.html`, and `styles.css`.
- `node --check` passed for `app.js`, `dataset.js`, and
  `trainerWebGpu3d.js`.
- The local server returned the `ui101` assets.
- Browser schedule smoke:
  - default: unchecked, manual fixed 0.300, slider enabled
  - step 576: narrowing 21%, effective support 0.296
  - after step 2048: settled, effective support 0.260
  - disabling the schedule restored fixed 0.300 immediately
- Browser layout smoke:
  - desktop canvas: 469.5 x 117.375, ratio 4.0
  - 390px mobile canvas: 390 x 97.5, ratio 4.0
  - label and canvas widths matched; label cells did not overlap
- Continuous training produced increasing step observations across repeated
  sample-loss intervals. A paused forced step advanced 24272 to 24273 and
  refreshed both sample loss and train-grid loss.
- Browser warning/error log was empty.

The schedule mechanics are verified. Its quality effect remains intentionally
unclaimed until a matched fixed-0.30 versus scheduled-0.30-to-0.26 run exists.
