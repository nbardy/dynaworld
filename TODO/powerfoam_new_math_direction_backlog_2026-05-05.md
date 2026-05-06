# PowerFoam New Math Direction Backlog

Date: 2026-05-05

Scope: ranked next experiments for PowerFoam heldout quality after the local
Metal core, local 4K gates, and optimizer-step trainability are no longer the
main blockers. This is explicitly not a hyperparameter-sweep queue. Each item
changes the geometry, topology, material model, motion model, visibility prior,
or diagnostic contract.

Current blocker split:

- Local Metal forward/backward and saved 4K gates are treated as locally
  validated unless a future audit invalidates the artifacts.
- Official CUDA/Warp parity remains blocked until a CUDA/Warp host generates
  the official fixture and the skip-until-present parity tests actually run.
- Paper acceptance is blocked by heldout quality, not by another obvious local
  timing gate. The selected clean DeepView row is around `10.85` PSNR / `0.08`
  SSIM against a `13.0` / `0.15` acceptance target.

Global gates before ranking any experiment as a candidate:

- It must report train PSNR/SSIM, heldout PSNR/SSIM, alpha coverage, and a
  residual/coverage panel on the same train/heldout split as the current clean
  PowerFoam row.
- It must keep the official-fixture blocker separate from heldout-quality
  blocker language.
- It must pass the focused local Metal gate before any quality run is compared:

```bash
PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with pytest python -m pytest \
  tests/test_powerfoam_direct.py tests/test_multicam_video_data.py -q
```

## P0. Topological Coverage Witnesses

Hypothesis: heldout failure is dominated by missing or badly selected support,
not by the optimizer. A cell set can be train-visible and still fail the
heldout camera if its Cech/regular-triangulation coverage leaves parallax
holes, low-support cells, or false local adjacency around the heldout rays.

Experiment:

- Build a no-heldout-RGB witness score per cell and per heldout ray using only
  train cameras, init tracks, depths, Cech edges, and projected support.
- Record which pixels have zero or low confidence intersection witnesses before
  training, after step 0 render, and after a short optimization run.
- Add a support-aware selector that prefers multi-camera track support,
  parallax diversity, nonredundant projected coverage, and adjacency-stable
  cells over simple top-K or random cell retention.

Acceptance criteria:

- Witness buckets explain at least one dominant heldout residual mode:
  uncovered alpha, wrong depth/order, low unique-camera support, or unstable
  adjacency.
- A support-aware 1024-cell init improves step-0 heldout alpha coverage or
  heldout PSNR over the current selected clean 1024-cell row without using
  heldout RGB.
- The diagnostic can reject itself: if witness confidence is high where
  residuals are high, this is not the next mechanism to scale.

Minimal smoke/benchmark gate:

- A diagnostic-only run writes `GT | render | alpha | residual | witness`
  panels plus JSON bucket totals for the selected clean row.
- A 1-step and 40-step quality probe compare current selection vs witness-aware
  selection at the same cell count, same cameras, same renderer path.

## P1. Visibility And Occlusion Priors

Hypothesis: the foam can cover pixels but blend the wrong material because the
model has no strong prior for which cell should win along a ray under heldout
viewpoint changes. Heldout SSIM can stay low even when alpha is nonblank if
occlusion order and empty-space consistency are weak.

Experiment:

- Add a train-only visibility prior over cells: per-camera visible/support
  votes, frontmost-depth consistency, empty-space penalties before the first
  observed surface, and behind-surface attenuation constraints.
- Track an occlusion confusion matrix: cells that are frontmost in source views
  but become hidden, cells that become newly exposed, and rays whose selected
  contributors disagree across train cameras.
- Prefer losses that constrain ray ordering and free space, not global color.

Acceptance criteria:

- Heldout residuals in high-alpha pixels decrease without reducing alpha
  coverage or train-view quality.
- Depth/order disagreement buckets shrink on the diagnostic panel.
- The prior does not simply make the scene more transparent; alpha mean and
  coverage must stay within a small tolerance of the baseline unless the
  baseline is explicitly over-opaque.

Minimal smoke/benchmark gate:

- Synthetic two-layer occlusion fixture passes forward and backward checks on
  Metal.
- Same 1-step and 40-step DeepView probe logs alpha, residual, and ordering
  buckets; compare against no-prior baseline.

## P2. Camera-Conditioned Material Transport

Hypothesis: current per-cell appearance state is absorbing view mismatch as
repainting. Heldout quality needs material state transported through the cell
frame and camera ray, not a view-agnostic color/SV field that overfits source
projections.

Experiment:

- Give each cell a small local material frame: normal, tangent basis, albedo or
  feature state, and a low-rank view-dependent residual conditioned on the
  camera ray in that local frame.
- Tie material changes across cameras by penalizing unexplained per-view color
  drift for the same witness-supported cell.
- Keep camera conditioning local to the cell frame so it cannot become a hidden
  heldout image encoder.

Acceptance criteria:

- Train-view quality does not improve while heldout stays flat; the useful
  signal is heldout improvement at similar train PSNR.
- Diagnostic buckets shift from "high-alpha wrong color" toward lower residual
  without worsening uncovered-alpha buckets.
- Local-frame material parameters remain smooth over adjacent support, not
  arbitrary per-camera lookup behavior.

Minimal smoke/benchmark gate:

- Unit smoke confirms camera-ray conditioning changes output for oblique rays
  and preserves identical output for identical local rays.
- 40-step DeepView probe compares baseline appearance vs local-frame material
  transport with identical geometry init and logs train/heldout delta.

## P3. Dynamic Support Motion Vs Repainting

Hypothesis: the best-looking dynamic foam runs may repaint a near-fixed support
instead of moving coherent geometry. If heldout failure is dynamic/support
misalignment, more appearance capacity will not fix it.

Experiment:

- Split state into persistent support trajectories and transported material:
  cell center/radius/frame motion should be low-dimensional and temporally
  coherent; appearance residuals should not be able to explain all motion.
- Add motion witnesses: per-cell displacement, local ARAP/stretch energy,
  material-coordinate drift, and whether the same surface patch remains
  responsible for the same track over time.
- Run a static-mask control to separate static-scene heldout acceptance from
  dynamic-object failure.

Acceptance criteria:

- Motion evidence shows actual support displacement or frame rotation where the
  video demands it, not just color changes on fixed cells.
- Static-mask heldout quality improves separately if the dynamic foreground is
  the blocker; if static-mask heldout is still weak, return to topology/support.
- Repaint budget ablation reduces train shortcutting without collapsing
  heldout alpha coverage.

Minimal smoke/benchmark gate:

- Two-frame synthetic translating-surface fixture: fixed-support repainting
  should fail the motion witness; moving support should pass.
- 40-step dynamic DeepView probe logs support displacement, appearance drift,
  static-mask metrics, and heldout metrics.

## P4. Gauge And Fluid Regularizers

Hypothesis: PowerFoam cells are a Lagrangian representation; they need local
gauge constraints that preserve coherent material patches. Without a gauge, the
optimizer can shear, shrink, rotate, or recolor neighboring cells in ways that
fit train views but destroy heldout structure.

Experiment:

- Add SE(3)/Sim(3)-style gauge regularizers over the foam graph: neighbor
  frame compatibility, bounded stretch, temporal acceleration, density/mass
  conservation, and optional divergence/curl penalties for fluid-like motion.
- Start with graph-local penalties on existing Cech neighbors; only move to a
  stronger fluid model if diagnostics show support motion is the bottleneck.
- Treat gauge terms as representation constraints, not generic L2 weight decay.

Acceptance criteria:

- Topology churn, frame flips, or neighbor stretch metrics decrease on the
  short run.
- Heldout PSNR/SSIM improves or at least the dominant geometry residual bucket
  improves at equal train quality.
- The regularizer does not freeze all support motion; displacement metrics
  should remain nonzero on dynamic regions when motion is expected.

Minimal smoke/benchmark gate:

- Graph fixture with known rigid motion has near-zero gauge loss; known shear
  or frame flip has nonzero loss and finite gradients.
- 1-step gradient smoke verifies finite gradients for centers, radii, frames,
  and material state; 40-step probe compares topology churn and heldout quality.

## P5. Data And Pose Diagnostics

Hypothesis: the model may be blamed for a split, pose, calibration, or clean
point support issue. This lane is not a dataset sweep; it is a falsification
gate before deeper math work.

Experiment:

- Run leave-one-train-camera-out evaluation inside the training cameras to
  measure cross-view failure before the official heldout camera is involved.
- Audit pose/calibration residuals by camera: reprojection residual, support
  count, parallax angle, fisheye distortion consistency, track uniqueness, and
  visible support.
- Compare clean point support against the selected cell subset and report what
  fraction of useful multi-camera tracks were discarded.

Acceptance criteria:

- The diagnostic classifies the blocker as one of: model representation, heldout
  camera geometry, pose/calibration, weak clean point support, or dynamic
  foreground mismatch.
- If leave-one-train-camera-out is poor, do not spend effort on official
  heldout-specific tricks yet.
- If train-camera holdout is good and official heldout is poor, rank topology,
  visibility, and pose coverage above appearance.

Minimal smoke/benchmark gate:

- Diagnostic-only script writes per-camera JSON and one compact panel; no
  training required.
- One 40-step leave-one-train-camera-out probe on the selected clean init uses
  the same renderer and logs comparable metrics.

## Ranking Summary

1. Topological coverage witnesses: highest leverage because it can explain
   whether the low heldout score is coverage, support selection, or topology.
2. Visibility/occlusion priors: next if alpha is present but depth/order or
   high-alpha residuals dominate.
3. Camera-conditioned material transport: next if geometry buckets look sane
   and the failure is high-alpha wrong color.
4. Dynamic support motion vs repainting: required before calling dynamic
   PowerFoam solved, especially if static-mask controls are much better than
   full-scene controls.
5. Gauge/fluid regularizers: use after motion/support diagnostics identify
   frame incoherence, topology churn, or material drift as the failure mode.
6. Data/pose diagnostics: run early as a falsification gate, but do not let it
   become an open-ended dataset search.

Do not update `BASELINES.md` from diagnostic-only smokes. Append there only
after a deliberate paper-scale or baseline-quality run with saved artifacts.
