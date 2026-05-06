# PowerFoam Math Backlog Beyond Sweeps

Date: 2026-05-05

Scope: audit the current PowerFoam completion blockers and define mathematical
next levers that are not another LR, cell-count, or plane-sweep rerun. This is
not a code-change plan. It is a backlog for the next mechanism work.

## Current Completion Blockers

Current verifier command:

```bash
PYTHONPATH=src/train .venv/bin/python research_experiments/dynamic_foam/verify_powerfoam_completion_audit.py --allow-local-tests-unrun --allow-incomplete
```

Result: `ok=false`.

The failed gates are:

- `official_cuda_warp_fixture_present`
- `official_direct_parity_test_ran_passed`
- `official_metal_parity_test_ran_passed`
- `paper_acceptance_verifier`
- `paper_scale_heldout_quality`

The local Metal side is no longer the main unknown. The audit accepts the
saved 4K height+SV raytrace benchmark, the saved 4K optimizer-step trainability
artifact, the local fixture/backward gate if not rerun in this read-only audit,
and the raytrace parity gate if not rerun. The official parity blocker is still
external: generate
`research_experiments/dynamic_foam/fixtures/powerfoam_tiny_height_sv_official_camera_official_v1.json`
on a CUDA/Warp host, copy it back, then run the two skip-until-present official
Direct/Metal parity tests.

The quality blocker is narrower and more interesting:

- selected clean DeepView row:
  `outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_appearanceonly_wandboffline_init_raytrace_128_16f_1024cells_40step_noaux`
- selected init:
  `research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_sift_wide_minucam2.ply`
- heldout PSNR / SSIM: `10.8536 / 0.0766`
- required PSNR / SSIM: `13.0 / 0.15`
- clean point cloud: `2821` points, reproj median/p90 `2.72/5.19 px`,
  track mean/p90 `6.20/8`, verified pairs `496`
- weakness: unique-camera support is still mostly two-camera
  (`unique_camera_track_mean=2.0007`, `unique_camera_track_p90=2.0`)

The clean-init coverage verifier says the selected geometry is visible enough
to be worth diagnosing, not simply invisible:

- train visible fraction: `1.0`
- heldout center coverage: `0.1008`
- heldout visible point ratio: `0.7667`
- heldout alpha previews are nonblank
- but the 1024-cell selected row keeps only `0.3773` of train-visible filtered
  points from the selected clean artifact

Negative controls already narrow the search:

- all-filtered/cell-count increases did not solve heldout quality
- top-ranked point sampling did not solve it
- render-ray fisheye correction alone did not solve it before the PLY builder
  became distortion-consistent
- close-overlap heldout split did not solve it by itself
- all-train-camera plane sweep and stricter photometric inlier plane sweep did
  not beat the selected OPENCV_FISHEYE pycolmap row
- affine/covariant SIFT produced a weaker artifact locally
- ALIKED/LightGlue remains an external ONNX-host candidate, not a local result

Conclusion: stop treating the missing lever as "try one more LR/cell-count/
plane-sweep setting." The next work should target geometry support, topology
stability, and heldout-specific failure attribution.

## Working Model

Let each PowerFoam cell have center `p_i`, support radius `r_i`, density
`sigma_i`, local surface/detail state `theta_i`, and train cameras `C_t`.

The paper-friendly topology relation is the Cech overlap graph:

```text
E_cech = {(i, j): ||p_i - p_j|| <= r_i + r_j}
```

The regular-triangulation graph is closer to the minimal power-cell face graph,
but it changes combinatorially as `p_i` and `r_i` move. Cech/AABB is a superset:
it can include false edges, but false edges should mainly cost traversal/clipping
work rather than remove true cell faces.

Current evidence says Cech/AABB is fast and locally correct enough on synthetic
4K, but clean heldout quality remains weak. That points to one of three
mathematical failures:

1. Geometry support is wrong: the cells cover train and heldout pixels, but not
   with the correct surface intersections, depth ordering, or normals.
2. Topology is unstable or too loose: Cech false edges, radius changes, or
   adjacency refreshes bias traversal/gradients in ways that overfit source
   views before heldout improves.
3. Appearance basis is absorbing geometry error: SV/color state improves train
   views while the cell complex remains a poor cross-view object.

## Backlog A: Heldout-Camera Diagnostics First

### A1. Heldout Residual Stratification

Hypothesis: aggregate heldout PSNR hides a small number of geometric failure
modes. We need per-pixel buckets before inventing another optimizer schedule.

Measure for every heldout pixel:

- alpha bucket: `alpha < .01`, `.01 <= alpha < .5`, `alpha >= .5`
- ray event count / traversal depth
- rendered median depth or depth quantile if available
- nearest contributing cell id and its train-view vote count
- projected init-track unique-camera count for that cell
- local parallax angle between nearest train view and heldout ray
- residual RGB/L1/SSIM-window contribution
- foreground/background mask if available from alpha or simple color threshold

Falsification:

- If high residual pixels are mostly `alpha < .01`, the next lever is coverage
  or radius/topology, not appearance.
- If high residual pixels are `alpha >= .5` with bad color but plausible depth,
  the next lever is appearance regularization/view basis.
- If high residual pixels concentrate at low unique-camera support or high
  parallax, the next lever is clean geometry support.
- If high residual pixels concentrate at topology churn or high traversal depth,
  the next lever is adjacency/topology stabilization.

Acceptance for this diagnostic:

- one saved heldout panel with columns
  `GT | render | alpha | residual | depth/normal | support`
- one JSON with residual bucket totals and top failure bucket
- no training run required

### A2. Train-View Holdout Rotation

Hypothesis: the chosen heldout camera is not the only issue; source cameras may
already fail cross-view when one train camera is temporarily held out.

Run the same selected clean geometry with leave-one-train-camera-out evaluation
inside the 8 train cameras, without using the official heldout camera for
selection.

Falsification:

- If train-camera cross-holdout is also poor, geometry/topology is bad before
  the heldout camera enters the story.
- If train-camera cross-holdout is good but `camera_0040` is poor, the blocker
  is baseline/parallax/visibility relative to the official heldout.

## Backlog B: Clean-Geometry Alternatives

### B1. Track-Support-Aware Cell Selection

Hypothesis: random or rank-only sampling is throwing away the wrong cells. The
selected run keeps only `37.7%` of filtered clean points. The cell subset should
maximize heldout-relevant train support under no-heldout-RGB constraints.

Mathematical objective:

```text
score_i =
    a * log(1 + track_len_i)
  + b * unique_camera_count_i
  + c * unique_frame_count_i
  + d * train_view_coverage_i
  - e * reprojection_error_i
  - f * redundancy_i
```

Choose a subset by farthest-point or facility-location selection in 3D and
screen space, not by simple top-K. `redundancy_i` should penalize points whose
nearest selected neighbor is close in both 3D and projected train pixels.

Falsification:

- If support-aware selection does not improve step-0 heldout over the selected
  random 1024-cell row, the issue is not merely which 1024 clean points survive.

### B2. Multi-Camera Track-Lifted Surfels

Hypothesis: current points are track centers, but PowerFoam cells need local
surface orientation and extent. A point-only init leaves normals/radii/heights
to be inferred from weak photometric gradients.

For each COLMAP/track point, estimate a local surfel:

```text
normal_i = principal normal from multi-view bearing intersection covariance
radius_i = k * sqrt(lambda_tangent_1 + lambda_tangent_2)
height_i = bounded function of reprojection/depth uncertainty
```

Initialize quaternion normals, tangent frame, radius, and texel height from this
surfel rather than generic defaults.

Falsification:

- If step-0 heldout alpha/depth improves but RGB does not, surfel geometry helps
  and appearance can be addressed separately.
- If step-0 heldout does not improve and residual buckets stay geometry-heavy,
  the track geometry itself is too weak.

### B3. Static-Only Masked Geometry

Hypothesis: DeepView `03_Dog` has dynamic/low-texture/object-region ambiguity,
and clean SfM points are biased toward background or repetitive texture. Current
heldout failure may be a static/dynamic support mismatch rather than a renderer
issue.

Build a train-only static support mask:

- high multi-view consistency over frames
- low optical-flow residual after known camera motion
- exclude deforming dog/foreground if it is not reconstructable as one static
  PowerFoam

Then train/evaluate two rows:

- static-mask-only geometry and loss
- complementary dynamic/foreground diagnostic row

Falsification:

- If static-mask-only PSNR/SSIM improves sharply on masked heldout pixels, the
  full-scene row is mixing static PowerFoam acceptance with dynamic content.
- If masked heldout remains poor, the geometry/topology representation is still
  the blocker.

### B4. External-Geometry Differential Diagnosis

Hypothesis: EX4DGS-init beating clean rows means geometry support, not Metal
primitive math, is the main quality gap.

Use external/pretrained geometry only as a diagnostic oracle:

- project EX4DGS points into train and heldout
- compute the same support metrics as clean points
- compare point support, depth distribution, alpha coverage, and residual
  buckets against the clean selected row

This must not be counted as paper-clean acceptance. It is a map of what the
clean geometry is missing.

## Backlog C: Differentiable Topology And Adjacency

### C1. Soft Cech Edge Energy

Hypothesis: hard adjacency refreshes create discontinuous optimization. We can
regularize the topology before changing the renderer.

Define soft overlap weight:

```text
d_ij = ||p_i - p_j||
s_ij = (r_i + r_j - d_ij) / tau
w_ij = sigmoid(s_ij)
```

Use `w_ij` for losses, not necessarily for Metal traversal:

- overlap budget: `sum_j w_ij`
- topology entropy: `-w log w - (1-w) log(1-w)` to avoid indecisive edges
- birth/death penalty: `|w_ij(t) - w_ij(t - k)|`
- heldout-safe constraint: cells visible from many train cameras get lower
  topology-change budget

Falsification:

- If topology churn drops but heldout does not improve, topology discontinuity
  is not the main failure.
- If heldout improves while train PSNR changes little, topology stabilization is
  a real lever.

### C2. Straight-Through Edge Refresh

Hypothesis: the graph decision should be discrete for rendering but smooth for
optimization pressure.

Use hard Cech/AABB edges in rendering, but maintain soft pre-edge scores for
regularization. Treat edge creation/deletion like a straight-through estimator:

```text
hard_edge_ij = 1[s_ij > 0]
loss_edge_ij = hard_edge_ij + w_ij - stopgrad(w_ij)
```

Do not put this in the renderer first. Add a diagnostic topology ledger:

- edges born/dead per validation interval
- mean degree and p95 degree
- missing required overlap edges
- high-residual cells with edge churn

### C3. Differentiable Radius-Topology Coupling

Hypothesis: radii are doing two incompatible jobs: visual coverage and graph
connectivity. Large radii improve alpha coverage but increase false Cech degree
and clipping complexity.

Split radius roles:

```text
r_render_i   controls support sphere used by the primitive
r_topology_i controls edge candidate graph
loss = lambda * ||log r_render_i - log r_topology_i||^2
```

This can reveal whether heldout wants different support and traversal scales.
If the split helps, later collapse it with a learned/calibrated relation.

## Backlog D: Cech/AABB Versus Regular Triangulation

### D1. False-Edge Sensitivity Test

Hypothesis: Cech/AABB is correct as a superset, but false edges may still alter
finite-precision traversal, normal-distance gradients, or training stability.

Compare the same frozen trained state under:

- `cech_aabb`
- `regular_triangulation`
- `cech_aabb` with random false-edge dropout that preserves all regular edges
- `regular_triangulation` plus sampled false edges

Measure:

- render PSNR delta at train and heldout cameras
- alpha/depth delta
- traversal step counts
- gradient cosine similarity for centers/radii/density/heights/SV axes
- adjacency degree distribution

Falsification:

- If frozen-render and gradient deltas are tiny, Cech false edges are not the
  quality blocker.
- If heldout deltas are larger than train deltas, topology choice is view
  generalization-relevant.

### D2. Regular-Triangulation Teacher, Cech Student

Hypothesis: regular triangulation is too slow or non-Metal-friendly to be the
selected production path, but it can teach which Cech edges matter.

For periodic CPU-side diagnostics:

1. Build regular-triangulation edges with SciPy/Qhull.
2. Label Cech edges as `true_face_like` or `false_superset`.
3. Train a cheap edge-priority score from local features:
   distance, radii, power-distance margin, view visibility, support votes.
4. Keep Cech correctness, but order/limit traversal by priority where safe.

Falsification:

- If edge priority does not reduce traversal steps or gradient noise without
  changing renders, abandon it.
- If it changes heldout materially, inspect whether the "false" edges were
  actually needed under finite precision.

### D3. Alpha-Complex Proxy

Hypothesis: the true minimal graph is closer to an alpha complex of support
spheres than either raw Cech or full regular triangulation. A cheap proxy could
reduce false topology while preserving correctness.

Candidate proxy:

```text
edge survives if
    Cech overlap is true
and mutual k-nearest in power-distance space is true
and radical-plane slab intersects both bounded spheres with margin
```

Use this only as an ablation graph. It may be unsafe as a renderer graph unless
the missing-overlap diagnostic proves no required edge was dropped.

## Backlog E: Heldout-Aware Training Without Heldout Leakage

### E1. Internal Heldout Selector

Hypothesis: current best checkpoint selection can still overfit source views
because the real heldout metric is sparse and late. We need an internal
selector that predicts heldout without using heldout RGB.

Use train cameras only:

- rotate one train camera as validation
- stratify by parallax and unique-camera support
- select checkpoints by the worst internal heldout bucket

Then evaluate once on the real heldout camera. This is not a new LR schedule; it
is a no-leakage selection signal.

### E2. Geometry-First Then Appearance-Only Gate

Hypothesis: appearance gradients hide geometry failure. Before color/SV
training, require geometry to pass a source-only multi-view consistency gate.

Gate candidates:

- projected depth ordering agrees across camera pairs
- alpha support overlaps train masks but does not flood background
- normal-distance quantiles are stable across views
- topology degree/churn is bounded

Only after the geometry gate passes should SV RGB/color state be optimized.

Falsification:

- If geometry gate cannot be passed from clean points, the backlog should move
  back to geometry construction.
- If geometry gate passes but heldout remains poor, the appearance model or
  camera/ray contract is the suspect.

## Priority Order

1. Build the heldout residual/support diagnostic panel and JSON (`A1`). This is
   the cheapest way to decide whether the next lever is geometry, topology, or
   appearance.
2. Add train-camera holdout rotation (`A2`) so the quality blocker is not tied
   to one external heldout view.
3. Try support-aware cell selection (`B1`) on the existing selected clean PLY.
   This directly tests the `37.7%` kept-point weakness without changing SfM.
4. Add topology ledger and soft Cech edge losses (`C1`, `C2`) as diagnostics
   before changing Metal traversal.
5. Run frozen-state Cech/AABB vs regular-triangulation sensitivity (`D1`) before
   treating regular triangulation as a production alternative.
6. If diagnostics say geometry is the blocker, implement track-lifted surfel
   initialization (`B2`) and static-only masked geometry (`B3`).

## Non-Goals For The Next Pass

- Do not run another blind LR sweep until residual buckets identify a failure
  class.
- Do not run another cell-count-only control; all-filtered controls were
  negative.
- Do not run another naive plane-sweep row as the main plan; current train-only
  plane-sweep variants did not beat the selected OPENCV_FISHEYE pycolmap row.
- Do not claim completion from local Metal parity alone. Official CUDA/Warp
  fixture parity and paper-scale heldout quality remain separate blockers.

