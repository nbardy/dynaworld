# World Tubes / camera-gauge candid review

User asked whether the World Tubes paper and camera-gauge idea are interesting
or slop. This was a read-only scientific review apart from this journal entry.
No training, native builds, tests, benchmark promotions, or manuscript edits.
Two independent read-only reviews covered mathematics and retained evidence.

## Assessment

The promising contribution is compiling one known camera program into shared
projected trajectories, support/tile membership, visibility events, and an
adjoint to the same world parameters. Practical novelty depends on measured
benefit at realistic scene complexity, including compile/repair/fallback costs.
Gauge change of variables, Gaussian conditioning/Schur complements, and the
fixed-topology chain rule are useful foundations, not new mathematical results
by themselves. The manuscript mostly acknowledges their standard status.

## Concrete mathematical concerns

- `01_camera_gauge_choices/README.md:3-11` attributes low-order orbit traces to
  a ray-fiber coordinate choice. At fixed sensor-time base coordinates, a depth
  reparameterization preserves the exact marginal and projected trajectory.
  Homogeneous projection and base/time reparameterization (e.g. tan(theta/2))
  must be distinguished from fiber-depth changes. Coordinates can improve
  evaluation/conditioning without removing physical visibility events.
- `04_revolving_camera_atlas/README.md:59-70` writes z_hat_b=h(z_hat_a).
  If z_hat denotes the conditional mean from the Gaussian packet, this is
  generally false for nonlinear h: E[h(Z)] != h(E[Z]). Restrict to affine
  transformations or explicitly transport a representative point; nonlinear
  transforms do not preserve a Gaussian conditional family. Monotonicity
  preserves order of physical points, not arbitrary distribution means.
- A static perspective camera is still nonlinear in ray coordinates. Exact
  affine Gaussian statements must identify the affine camera/approximation,
  integration domain, and measure; rational center traces alone do not close
  the covariance or nonlinear fiber integral.

## Evidence read

- Current generated schema-v2 ledger accepts theorem correctness and the
  bounded clean variable-camera curve. Frozen learned-world scaling, learned
  moving-camera density, and seven public-context rows remain missing.
  Older master-plan language calling the camera curve absent is stale.
- The accepted camera curve uses three primitives and 64 samples: 11 closure
  rows through 170-degree half-span (340-degree total open yaw), with an
  unresolved endpoint-chart termination at 179.5-degree half-span. This is
  meaningful bounded correctness, not learned-scene speed/quality.
- The visibility fixture reduces order-crossing image error 0.186742 to zero;
  exposure/rolling-shutter and fallback forward/VJP checks are useful behavior
  evidence. Gauge agreement itself verifies change of variables.
- `WORLD_TUBES_PAPER_DRAFT.md:1490-1497` explicitly says the 32x logical-volume
  comparison double-counts shared replay tensors and omits topology, bins, and
  transients. It is not measured memory savings. Tiny historical timing rows
  are diagnostic and do not establish real-scene total acceleration.
- `BASELINES.md:268` invalidates prior Neural3D calibration comparisons: the
  LLFF axis correction reduced epipolar error 90.54px -> 0.54px. Historical
  6.3863dB / 5.9153dB World Tubes rows cannot support current quality claims.
- The 199-atom retained-fiber hybrid stress falls back on 64/64 tiles. It is a
  scoped certificate-coverage warning, not proof every compiled path fails.

## Literature checked (targeted, not exhaustive novelty clearance)

- EWA Volume Splatting (2001) already derives Gaussian footprints through a
  locally affine ray-space mapping:
  https://www.cs.umd.edu/~zwicker/publications/EWAVolumeSplatting-VIS01.pdf
- Spacetime Gaussian Feature Splatting already uses temporal opacity and
  polynomial motion/rotation: https://arxiv.org/abs/2312.16812
- Kinetic data structures already maintain geometric properties through
  motion certificates and certificate-failure events:
  https://graphics.stanford.edu/~comba/papers/socg.pdf
- Neo reuses and updates Gaussian depth ordering between frames and pairs the
  algorithm with specialized hardware: https://arxiv.org/abs/2511.12930
- GSReuse reuses screen-space content via motion vectors, warping, and tile
  filtering: https://pubmed.ncbi.nlm.nih.gov/41921174/
- Targeted search found no Neo/GSReuse/kinetic references in the two manuscript
  sources or bibliography. These works constrain broad reuse novelty claims;
  their existence does not establish equivalence to the compiled dynamic-world
  evaluator and adjoint proposed here.

## Recommended decision experiment

Use one correctly calibrated, visually plausible frozen learned world. Compare
the same renderer by competent per-frame replay/batching and the compiled route
over a fixed physical interval with increasing sample density. Include a simple
temporal-reuse baseline when feasible. Measure image/world-gradient agreement,
compile and refresh costs, total forward/backward time, actual peak memory,
topology-inclusive retained state, and fallback. A speed break-even or a real
memory-enabled workload would justify developing the compiler paper. Another
formalism or package-verifier layer does not resolve this question.
