# Browser Trajectory-3DGS Plateau Audit

Audited the active SPA after the user reported a long plateau near one million
steps. The active tiled backend is a custom trajectory-gated dynamic 3DGS, not
native 4DGS, SpacetimeGS, Dynamic3DGS, STAR, or World Tubes.

The useful surprise was that the plateau was observable in parameter structure,
not just loss:

- old late run: train/heldout 19.7/15.3 dB, but large 8,192-step parameter
  deltas remained;
- initialization: zero dynamic splats, 94.5% endpoint support, and 91% of
  splats at the former 3:1 anisotropy cap;
- ordinary training pushed median static mix above 0.98 and left zero dynamic
  splats;
- around 1.2k-1.3k of the 4,096 allocated slots remained raster-dead;
- a 25% dynamic reserve preserved dynamic slots but slightly hurt early
  train/heldout quality, so the runtime branch was removed;
- a 0.02 split-opacity floor did not materially reduce dead slots and was
  removed.

Implemented observable, reversible changes: explicit model naming, beta2/eps
correction, 120k LR decay toggle, longer density-stat memory, 2k-to-4k growth,
recycling through 120k, 6:1 bounded anisotropy, and asynchronous parameter
diagnostics. Clean uncontended throughput returned to roughly 213-246 completed
steps/s; earlier 113-154 readings were invalid because three trainer tabs were
running simultaneously.

The next real model work is loss-map/depth-guided spatial-temporal birth,
followed by an independent-per-time 3DGS oracle and view-dependent appearance.
Browser SfM is not the next move because calibrated cameras already exist and
the canonical Python export adapter owns initialization/calibration semantics.

See `research_notes/browser_trajectory_3dgs_plateau_audit_2026-07-29.md`.
