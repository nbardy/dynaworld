# PowerFoam Heldout Failure Diagnostic Backlog

Date: 2026-05-05

Scope: next experiments for diagnosing why the current local PowerFoam Metal
DeepView clean rows still fail heldout quality. This is a planning note only;
no code or shared status docs were changed with this note.

Current failure boundary:

- Local Metal/Torch/raytrace/4K trainability gates are strong enough that the
  next local question is not "does the renderer run?"
- The selected clean DeepView row is the OPENCV_FISHEYE pycolmap artifact plus
  frozen-geometry RGB-only appearance training, selected at step 40 with heldout
  PSNR about 10.85 and SSIM about 0.077.
- Paper acceptance still needs at least PSNR 13 and SSIM 0.15 on the clean row,
  plus the official CUDA/Warp fixture path.
- Recent negative controls already weakened simple explanations:
  all-filtered-point capacity was worse, first1024 ordering was worse,
  render-side fisheye rays alone were worse, close-overlap heldout alone was
  worse, raw plane sweep was worse, photometric-inlier plane sweep improved but
  remained worse, and material-only training did not rescue the selected clean
  artifact.

Working model:

The failure is probably not one scalar knob. Treat it as a diagnostic split:

1. Alpha/coverage: the heldout view may not be physically covered, or coverage
   may be wrong even when preview alpha is nonblank.
2. Color/material: geometry may be adequate but the learned material field does
   not transfer across view angle, exposure, or occlusion.
3. Geometry: points/cells may be in the wrong 3D places, too sparse, too
   track-local, or concentrated on two-camera support.
4. Topology/support: Cech/AABB adjacency may be correct locally but too coarse,
   too disconnected, or missing the cells that matter for heldout rays.
5. Camera support: intrinsics/extrinsics/lens handling may be internally
   consistent but still misaligned with the point-cloud builder, train rays, or
   heldout rays.

The backlog below is ordered to minimize wasted training. Add cheap probes that
read existing artifacts first, then run short paired experiments only after a
branch survives the metric gate.

## P0: One JSON Diagnostic Per Candidate

Goal: make every candidate comparable before another training sweep.

Add a small script:

```text
research_experiments/dynamic_foam/diagnose_powerfoam_candidate.py
```

Inputs:

```text
--config <train_config.jsonc>
--output-dir <run output dir>
--artifact-json <point cloud summary json>
--heldout-preview-glob "heldout_preview_step_*.png"
--write <diagnostic.json>
```

Expected output shape:

```json
{
  "candidate": {
    "config": "...",
    "output_dir": "...",
    "artifact": "...",
    "best_step": 40,
    "heldout_psnr": 10.85,
    "heldout_ssim": 0.077
  },
  "alpha_coverage": {...},
  "color_transfer": {...},
  "geometry_support": {...},
  "topology": {...},
  "camera_support": {...},
  "diagnosis": {
    "primary_failure": "alpha_coverage|color|geometry|topology|camera|mixed",
    "confidence": "low|medium|high",
    "next_experiment": "..."
  }
}
```

This script should mostly compose existing helpers from
`verify_powerfoam_clean_init_coverage.py`, `verify_powerfoam_paper_acceptance.py`,
camera projection utilities, and run output JSONs. It should not launch
training.

Decision rule:

- If the script cannot classify the selected clean row, do not start more
  training; first add the missing metric.
- If it classifies the failure, run only the next experiment for that branch.

## P1: Alpha/Coverage Branch

Hypothesis:

Heldout failure is mostly missing or misplaced opacity support. A render can
have nonblank alpha previews yet still cover the wrong pixels, cover only easy
background, or miss foreground silhouettes.

Small metrics to add:

1. Heldout alpha occupancy:
   - `alpha_mean`, `alpha_p10/p50/p90`, fraction over `0.1/0.5/0.9`.
   - Compute per heldout frame and aggregate mean/min.
2. Target-conditioned alpha coverage:
   - Convert heldout RGB to a crude foreground mask by difference from
     per-frame border/background median, or by Sobel/gradient magnitude.
   - Report `alpha_on_target_foreground_mean`,
     `foreground_pixels_with_alpha_gt_0p5`, and
     `background_pixels_with_alpha_gt_0p5`.
3. Error-conditioned alpha:
   - For pixels in top 20 percent absolute RGB error, report alpha quantiles.
   - For low-alpha pixels, report RGB error quantiles.
4. Center projection vs rendered alpha:
   - From the init point cloud and heldout camera rays, compute center-pixel
     coverage as the verifier already does.
   - Compare projected center support with actual alpha support.

Expected output:

```json
"alpha_coverage": {
  "heldout_alpha_mean": 0.42,
  "heldout_alpha_gt_0p5": 0.31,
  "foreground_alpha_recall_gt_0p5": 0.22,
  "background_alpha_fp_gt_0p5": 0.18,
  "top_error_alpha_p50": 0.07,
  "center_pixel_coverage_mean": 0.08,
  "center_to_alpha_iou": 0.19
}
```

Interpretation:

- Low foreground alpha recall and low top-error alpha means support a coverage
  bottleneck. Next experiment: build stronger heldout-visible support before
  changing color or LR.
- High alpha recall but poor PSNR/SSIM weakens coverage as the primary blocker.
  Next experiment: go to the color/material branch.
- High background alpha false positives suggest opacity is smeared or topology
  is letting cells over-cover. Next experiment: run topology/support diagnostics
  before adding cells.

Concrete next experiment if supported:

Run an alpha-only or silhouette-proxy training control on the current selected
artifact:

- Freeze color/material to neutral gray.
- Optimize only density/height/SV/center/radius terms needed for alpha.
- Loss: foreground-mask BCE/Dice proxy plus low background alpha penalty.
- Stop after 40-80 steps.

Expected outcomes:

- If alpha improves but heldout RGB stays poor, geometry can cover the view and
  the next branch is color/material.
- If alpha cannot improve without severe train loss damage, support/topology or
  camera alignment is likely wrong.
- If alpha improves only by filling background, topology/regularization is the
  next blocker.

## P2: Color/Material Branch

Hypothesis:

The heldout view is covered, but surface color/material is view-dependent,
underfit, or trained to source-view appearance shortcuts that do not transfer.

Small metrics to add:

1. Alpha-masked color error:
   - Compute heldout PSNR/SSIM/L1 only where `alpha > 0.5`.
   - Also compute only where both projected center support and alpha are true.
2. Source-vs-heldout color gap:
   - For each frame, report source alpha-masked RGB L1 and heldout
     alpha-masked RGB L1.
   - Report ratio `heldout_masked_l1 / source_masked_l1`.
3. Exposure/color-bias fit:
   - Fit a per-channel affine correction from rendered heldout to target
     heldout on alpha-supported pixels.
   - Report PSNR before and after correction.
4. View-angle bucket error:
   - Use per-pixel/cell normals or camera ray direction if available.
   - Bucket errors by approximate normal-dot-view or ray angle.

Expected output:

```json
"color_transfer": {
  "masked_heldout_psnr": 12.4,
  "masked_heldout_l1": 0.18,
  "source_masked_l1": 0.07,
  "heldout_source_l1_ratio": 2.57,
  "affine_color_corrected_psnr": 12.9,
  "affine_gain": [1.08, 0.97, 0.91],
  "angle_bucket_l1": {"front": 0.12, "grazing": 0.27}
}
```

Interpretation:

- Large gain after affine correction points to color calibration/exposure, not
  geometry. Next experiment: per-camera color affine or exposure-normalized
  training control.
- High alpha-masked error with little affine gain points to material/view
  dependence. Next experiment: view-conditioned color basis or stronger SV
  material parameterization, keeping geometry frozen.
- Good alpha-masked PSNR but poor full-image PSNR means uncovered pixels or
  silhouette/background dominates; return to alpha/coverage.

Concrete next experiment if supported:

Run three matched 40-step rows on the selected clean artifact:

1. `rgb_only`: current frozen-geometry RGB-only control.
2. `rgb_plus_camera_affine`: per-camera affine color post-transform.
3. `rgb_plus_viewdir_basis`: small view-direction color basis on the texel/SV
   material, with geometry frozen.

Expected outcomes:

- If `rgb_plus_camera_affine` closes most of the gap, add calibration as a
  controlled baseline and do not pursue geometry yet.
- If `rgb_plus_viewdir_basis` improves heldout but source overfits, the
  material model is too view-local; add regularized view basis or cross-view
  color consistency.
- If neither improves, color is not the primary blocker; move to geometry.

## P3: Geometry Branch

Hypothesis:

The init support is visible enough in aggregate, but its 3D geometry is too
weak: mostly two-camera tracks, wrong depths, missing surfaces, or high
uncertainty around heldout-visible regions.

Small metrics to add:

1. Track support vs heldout error:
   - For rendered pixels/cells, bucket by source track unique-camera count,
     unique-frame count, reprojection error, and track length.
   - Report heldout error contribution per bucket.
2. Heldout-visible point support:
   - Count how many train-visible init points project into heldout foreground
     and top-error regions.
   - Report unique-camera support for those points only.
3. Depth consistency residual:
   - For each point, reproject into all supporting train cameras and report
     reproj error already in the artifact.
   - Add coarse heldout ray distance to nearest point/cell center for target
     foreground pixels.
4. Geometry-frozen vs geometry-trainable delta:
   - Compare identical artifact rows with frozen geometry, material-only, and
     low-geom unfrozen training.
   - Report heldout metric delta and center/radius/height movement norms.

Expected output:

```json
"geometry_support": {
  "heldout_foreground_projected_point_ratio": 0.38,
  "top_error_nearest_center_px_p50": 9.2,
  "top_error_points_unique_camera_p90": 2,
  "track_bucket_l1": {
    "ucam_2": 0.25,
    "ucam_3plus": 0.14
  },
  "geometry_trainable_psnr_delta": -0.2,
  "center_update_norm_p90_scene_frac": 0.006
}
```

Interpretation:

- Error concentrated in two-camera tracks or far-from-center pixels supports
  stronger geometry as the next lever.
- Three-plus-camera support with low reprojection error but poor heldout color
  weakens geometry and points back to color/material.
- Geometry-trainable rows moving little while quality stays poor suggests
  gradients are weak or topology/camera is wrong.
- Geometry-trainable rows moving a lot while heldout worsens suggests train
  views are pulling geometry into view-local explanations.

Concrete next experiment if supported:

Run the external-host clean-geometry branch before local model changes:

- Generate ALIKED/LightGlue known-pose artifact on an ONNX-capable host.
- Prefer artifacts with `unique_camera_track_p90 > 2` and equal or better
  reprojection p90 than the current selected artifact.
- Train matched frozen-geometry RGB-only and low-geom-unfrozen rows.

Expected outcomes:

- Better artifact plus frozen RGB improves: geometry support was primary.
- Better artifact only improves with geometry unfrozen: init is better but
  needs surface adaptation.
- Better artifact does not improve: geometry support is not sufficient; inspect
  camera/topology.

## P4: Topology/Support Branch

Hypothesis:

Cell centers are reasonable, but the selected adjacency/support graph and
ray traversal expose the wrong topology to heldout rays: disconnected cells,
overly local neighbors, bad Cech/AABB radius effects, or too few active
contributors per pixel.

Small metrics to add:

1. Adjacency health per candidate:
   - Existing adjacency stats at step 0 are logged; persist them into the
     candidate diagnostic JSON.
   - Add connected component counts, isolated-cell fraction, degree p10/p50/p90.
2. Heldout active contributor stats:
   - For sampled heldout rays, report active cell count per ray, transmittance
     end value, and effective number of contributors:
     `1 / sum(normalized_weight^2)`.
3. Topology-vs-alpha relation:
   - Bucket pixels by active contributor count and compare alpha/error.
4. Topology mode control:
   - Same artifact, same frozen material, compare `cech_aabb`, current fast
     selected mode if different, and a dense/all-pairs small-cell subset at
     128px.

Expected output:

```json
"topology": {
  "degree_p50": 18,
  "isolated_cell_fraction": 0.03,
  "heldout_active_cells_p50": 4,
  "heldout_effective_contributors_p50": 1.6,
  "low_contributor_top_error_fraction": 0.71,
  "dense_subset_psnr_delta": 0.4
}
```

Interpretation:

- Low active contributors in high-error pixels means support traversal/topology
  is starving heldout rays. Next experiment: topology mode/radius/support
  control.
- Dense/all-pairs subset improving heldout means selected topology is the
  bottleneck, not the point cloud alone.
- Dense/all-pairs subset not improving pushes back to geometry/camera.
- High contributors plus wrong color/low PSNR points to color/material or
  occlusion ordering, not graph sparsity.

Concrete next experiment if supported:

Create a 128px, 1024-cell matched topology triad:

1. current `cech_aabb`.
2. expanded Cech/AABB radius or neighbor cap.
3. dense/all-pairs reference for a small fixed subset.

Expected outcomes:

- If only dense/all-pairs works, the local selected adjacency is too sparse or
  wrong for heldout rays.
- If expanded topology works but dense is similar, tune topology support and
  update acceptance diagnostics.
- If none work, topology is not primary.

## P5: Camera Support Branch

Hypothesis:

The run uses lens-aware projection and OPENCV_FISHEYE metadata, but some
boundary remains inconsistent: pycolmap known-pose builder, DeepView
intrinsics, train rays, heldout rays, scaling, frame/view indexing, or
coordinate convention.

Small metrics to add:

1. Round-trip camera projection check:
   - For each train and heldout camera, project known 3D points through the
     builder camera model and the trainer camera model.
   - Report pixel deltas p50/p90/max.
2. Ray-vs-pycolmap consistency check:
   - Given a 2D feature observation and reconstructed 3D point, compare its
     ray direction from trainer camera against vector from camera origin to
     point.
   - Report angular residuals.
3. Heldout camera sensitivity:
   - Render selected artifact under small perturbations:
     `fx/fy +/-1%`, principal point `+/-2px`, rotation `+/-0.25deg`,
     translation `+/-1% scene_radius`.
   - Report whether PSNR improves under any perturbation.
4. Source camera sanity:
   - Same perturbation on source views; ensure source PSNR degrades at zero
     perturbation. If not, the metric is too insensitive.

Expected output:

```json
"camera_support": {
  "trainer_vs_builder_projection_delta_p90_px": 0.42,
  "ray_point_angular_residual_p90_deg": 0.08,
  "best_heldout_camera_perturbation": {
    "kind": "rot_y",
    "value": 0.25,
    "psnr_delta": 0.9
  },
  "source_zero_is_best": true
}
```

Interpretation:

- Projection/ray residual over about 1 px or camera perturbation improving
  heldout by more than 0.5 PSNR supports a camera mismatch.
- Source zero not best means the source-camera path itself is suspect; stop
  geometry work and audit camera conventions.
- Tiny residuals and no beneficial perturbation weaken camera support as the
  blocker; return to geometry/topology/material.

Concrete next experiment if supported:

Add a camera-refinement diagnostic run, not a model feature:

- Freeze all foam parameters.
- Optimize a small heldout-camera correction on heldout RGB for analysis only.
- Separately optimize train-camera corrections on source views.
- Compare learned correction magnitudes to the known calibration scale.

Expected outcomes:

- Small heldout correction greatly improves PSNR: camera calibration is a
  plausible blocker.
- Large correction needed, or train correction also improves source: camera
  model/convention mismatch likely.
- No meaningful correction helps: camera support is not primary.

## Decision Tree

Run order:

1. Generate P0 diagnostic for the selected clean row and all recent negative
   controls.
2. If heldout foreground alpha recall is low, run P1 alpha-only control.
3. Else if alpha-masked color is poor and affine/viewdir correction helps, run
   P2 color/material triad.
4. Else if error correlates with two-camera/low-track support or top-error
   pixels are far from projected centers, run P3 external clean-geometry branch.
5. Else if active heldout contributor counts are low or dense topology helps,
   run P4 topology triad.
6. Else if camera perturbation improves heldout or projection residuals are
   nontrivial, run P5 camera support audit.
7. If none fires, the current metrics are insufficient; add occlusion/depth
   ordering diagnostics before changing representation.

Stop rules:

- Do not run more cell-count sweeps unless P1 or P4 says coverage/topology is
  the bottleneck and predicts exactly which count/support should change.
- Do not run more plane-sweep scoring variants unless P3 says geometric support
  improves in a way the selected artifact lacks, especially unique-camera
  support above p90 2.
- Do not add a new material model unless P2 alpha-masked and color-corrected
  metrics show the current view-independent material is the bottleneck.
- Do not treat close-overlap split performance as acceptance; it is a sanity
  branch, not the canonical DeepView heldout gate.

## Minimal New Artifacts To Add Later

These are deliberately small and script-shaped:

1. `diagnose_powerfoam_candidate.py`
   - Reads one candidate and writes one JSON.
   - No training, no W&B dependency.
2. `compare_powerfoam_candidates.py`
   - Reads multiple diagnostic JSONs and prints a markdown table sorted by
     heldout PSNR and primary failure.
3. `probe_powerfoam_camera_perturbations.py`
   - Renders existing checkpoint/artifact under camera perturbations and writes
     a JSON plus optional CSV.
4. `probe_powerfoam_topology_subset.py`
   - Runs a tiny no-train or frozen-material render comparison for topology
     modes on the same cells/rays.
5. Optional trainer logging:
   - Persist alpha quantiles, alpha-masked metrics, and heldout active
     contributor stats into `eval_metrics_history.jsonl` once the scripts prove
     these metrics discriminate candidates.

## Acceptance For The Diagnostic Backlog

This backlog is successful when the next experiment can be selected by a JSON
metric instead of taste. A useful next report should be able to say:

```text
Primary failure: geometry_support
Evidence: top-error pixels are far from projected centers, and their points are
mostly unique-camera-count 2; alpha recall is adequate and color affine gives
only +0.1 PSNR.
Next run: external ALIKED/LightGlue clean artifact, matched frozen-RGB and
low-geom-unfrozen 40-step rows.
Stop condition: if unique-camera p90 stays 2 or heldout PSNR remains under
11.0, stop geometry-builder tweaks and audit camera/topology.
```

The main discipline is to keep the branches separate. A bad heldout render can
look like "geometry is bad" even when the measurable issue is alpha, color,
topology, or camera convention. The next useful work is the splitter, not
another broad sweep.
