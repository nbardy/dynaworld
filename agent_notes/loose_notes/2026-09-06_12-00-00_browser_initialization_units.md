# Browser initialization: source-unit counterexample

Date: 2026-09-06. Filename time is an ordering placeholder, not run timing.

The user asked to challenge the previous explanation of sparse Deep3D Jump
initialization. This pass tested the existing implementation before changing
defaults or generating denser data. The working tree already contains unrelated
paper changes and an ongoing browser World Tubes integration; none was reverted.

## Experiment

Run from the repository root:

```sh
node web/dynaworld_browser_trainer/tests/auditInitializationUnits.mjs agent_notes/loose_notes/2026-09-06_12-00-00_browser_initialization_units.json
```

Scale source XYZ and camera translations together by 0.1, 1, and 10. Normalize
using the actual browser function, then initialize all 4096 splats with the
actual initializer. Intrinsics, colors, point count, ordering, and camera roles
are unchanged. This is a coordinate-unit change, not a new scene or seed cloud.

The accompanying JSON records six CPU measurements across Deep3D and Coffee
Martini. Coverage here is mean composed alpha, NOT a percentage of pixels with
nonzero support. An independent pixel statistic measures alpha above 0.5.
These are initialization diagnostics, not training quality or GPU performance.
Temporal gating and Mip opacity compensation are excluded and stated in JSON.

## Measured result

Deep3D source scale 1: 3541/4096 splats have all three axes at the initializer
maximum, median aspect is 1, median projected sigma is 0.344917 px at 96x72,
mean alpha is 0.049240, and only 0.004919 of pixels exceed alpha 0.5.

The geometrically identical scale-0.1 input: only 6 splats have all axes capped,
median aspect is 3, median projected sigma is 1.013392 px, mean alpha is
0.262334, and 0.243345 of pixels exceed alpha 0.5. Maximum normalized center
disagreement is 1.192093e-7. Scaling source units by 10 instead makes all 4096
splats hit the ceiling. Coffee Martini also fails this invariance check.

## Root cause and wider scope

`normalizeDatasetGeometry` normalizes points and camera translations by an
inverse median depth. `localGaussianFrames` then sets minimum/maximum sizes
to `0.03 * geometryScale` and `0.60 * geometryScale`. Its local covariance is
already computed from normalized points. Consequently these source-unit caps
erase local geometry differently for physically identical reconstructions.

The tiled optimizer packs min/max scale as `0.03 * geometryScale` and
`geometryScale`. Velocity and harmonic displacement are capped at
`2 * geometryScale` and `1.5 * geometryScale`. Geometry consistency residuals
also divide by this conversion factor. An initializer-only fix would leave
unit-dependent restrictions active during fitting. A whole training-step
invariance test is needed, not just a prettier initialization screenshot.

## Corrections to the previous explanation

- Sparse feature points and low opacity remain real facts, but the claim that
  denser SfM is the first necessary fix was premature. Existing geometry is
  being visibly damaged by a unit-dependent clamp.
- A coherent-looking screenshot does not establish camera correctness. This
  audit has not independently checked source calibrations or reprojections.
- The report calls 5517 points `box_filtered_count`; do not describe all
  11872-to-5517 removals as geometric reliability rejection without tracing
  the exporter and filters.
- Frame-0-only seeds cannot encode the person's complete trajectory. That
  limitation is distinct from the measured stationary-background size bug.
- Mean alpha near 4.9% is not the fraction of pixels covered.

## Next bounded correction

Keep raw-to-normalized conversion exclusively for coordinates and preview
camera conversion. Define geometric initialization, scale/motion trust regions,
and geometric loss normalization from a characteristic length in normalized
training geometry. Choose and measure that definition before promoting it.
Apply it consistently across the actual backend and initializer. Preserve the
current route as a matched experimental control, then compare identical-source
unit variants and short fixed-frame fits before full dynamic training.

Acceptance: identical projections and initialization under source-unit changes;
matching optimizer steps and regularization; finite local anisotropic shapes;
improved matched RGB metrics, with heldout RGB used only for evaluation. Larger
footprints can increase tile occupancy, so check overflow and drained GPU step
timing. No default, shader, dataset, or deployed build changed in this pass.
