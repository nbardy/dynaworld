# Gauge Field Next Plan

## Current State

The gauge-field experiment is now a representation-ablation harness, not a
finished primitive. It can compare:

```text
screen_disk
oriented_slab
rank_adaptive_metric
free_dynamic_3dgs
```

It has:

```text
source-view support-mode benchmarks
DeepView held-out-camera benchmarks
X-map/projection diagnostics for gauge modes
direct free-dynamic splat baseline
active/effective parameter accounting
coarse end-to-end speed timings
mathematical web-of-thought prompt for next representation search
```

The key experimental result so far is that source-view ranking and held-out-view
ranking disagree. That means source-view PSNR must not select the primitive.

## Goals

### Goal 1: Make The Evaluation Gate Real

The benchmark should answer:

```text
Does a representation survive novel-view/view-stress better than the projected
disk control and direct splat baseline?
```

Required metrics:

- held-out-camera PSNR/L1/SSIM/LPIPS if available
- source-view PSNR/L1 only as context
- X-map occupancy and entropy
- X-map flow or track consistency
- camera perturbation stress
- RGB-near-null cheat probes
- witness-rank / multi-ray concurrence when multiview rays are available
- train-step and render-only wall time

### Goal 2: Run Fair Budgets

Run both fairness regimes:

```text
same rendered primitive count
same active parameter budget
```

Current active-param matched counts against 2048 free dynamic 3DGS splats/frame:

```text
screen_disk: about 8192 elements
oriented_slab: about 7516 elements
rank_adaptive_metric: about 7516 elements
free_dynamic_3dgs: 2048 splats/frame
```

Do not collapse these into one number. Same primitive count tests rendering
semantics; same parameter count tests capacity fairness.

### Goal 3: Remove Camera-Model Ambiguity

The first DeepView lane uses a pinhole approximation for fisheye cameras. Before
strong novel-view claims:

- implement DeepView fisheye projection in the gauge renderer, or
- restrict the first paper-quality claim to calibrated pinhole/synthetic data.

The practical next step is to add a fisheye camera path behind an explicit
config switch and rerun the same source/target pair.

### Goal 4: Add Structural Diagnostics

Add metrics that directly attack the chief-scientist axioms.

#### Witness Rank

For a material event/element with assigned rays:

```text
A_q = sum_k w_kq (I - d_k d_k^T)
```

Report:

```text
eig_1, eig_2, eig_3
eig_3 / eig_1
condition number
valid assigned ray count
```

Use this first as a diagnostic, not a loss.

#### X-map Flow Consistency

Given flow `F_t(p)`:

```text
X_t(p) ~= X_{t+1}(p + F_t(p))
```

Report:

```text
mean X error on valid alpha/flow pixels
error quantiles
occupancy/entropy beside consistency
```

Consistency without occupancy is not trusted.

#### Camera Perturbation Stress

Render the same fitted field under small camera deltas:

```text
translation: +/- small fraction of scene depth
rotation: +/- 1 to 5 degrees
```

Record qualitative previews and self-consistency metrics. This is not a hidden
camera substitute, but it catches billboards, holes, and camera-glued material.

### Goal 5: Tune Representation-Specific Knobs

Run small targeted sweeps instead of changing everything at once.

For `screen_disk`:

- element count
- radius
- alpha logit
- opacity transfer

For `oriented_slab`:

- thickness prior
- in-plane radius
- anisotropy clamp
- whether thickness stays fixed or learned

For `rank_adaptive_metric`:

- eigenvalue sparsity / MDL pressure
- trace penalty
- condition-number penalty
- KNN count for local Jacobian
- fixed vs learned off-diagonal strength

For `free_dynamic_3dgs`:

- per-frame vs static splat bank
- temporal smoothness
- same active parameter budget
- same wall-clock budget

### Goal 6: Make Timing Trustworthy

Current speed notes are 5-step end-to-end timings. Replace with:

```text
train_step_ms
render_forward_ms
render_backward_ms
eval_render_ms
media_write_ms
peak_memory
```

The current rough finding is:

```text
screen_disk at active-param-matched count is plausible
slab/rank are much slower in pure Torch
free_dynamic_3dgs is between screen_disk and slab/rank in the coarse timing
```

Do not make renderer conclusions from the current coarse table.

## First Concrete Run Matrix

Run this before inventing a new primitive:

| family | budget | scenes | target cameras | steps |
| --- | --- | --- | --- | --- |
| same primitive count | 2048 each | 3 DeepView scenes | 2 targets each | 250 |
| same active params | 8192/7516/7516/2048 | 3 DeepView scenes | 2 targets each | 250 |
| longer best candidates | best 2-3 rows | 3 DeepView scenes | 2 targets each | 1000 |

Sort primary tables by held-out-camera quality, not source-view quality.

## First Implementation Tasks

1. Add an `outputs/` artifact policy.
   - Either ignore generated outputs and commit only summaries, or commit a
     curated small results folder.

2. Add a clean timing script.
   - Do not use trainer wall time as renderer evidence.

3. Add camera perturbation stress.
   - Start with saved previews and simple metrics.

4. Add X-map flow consistency.
   - Use existing or lightweight optical flow if available.

5. Add witness-rank diagnostics.
   - Start with DeepView source/target rays and rendered element weights.

6. Add a fisheye camera path or switch to a pinhole-calibrated dataset for the
   next claim.

7. Run the mathematical web-of-thought prompt once with the ready-to-paste first
   invocation.
   - Convert only the strongest resulting diagnostic or primitive into code.

## New Incidence Candidate: Compact Ray-Integrated Ellipsoid

The latest theory pass produced a narrower candidate than the broad
ray-Gaussian line integral:

```text
kappa(x) = beta * [1 - (x - mu)^T A (x - mu)]_+^k
```

This keeps world-space ray integration but gives finite compact support, a conic
footprint, and constant per-pixel polynomial evaluation. The local theorem-level
claim is:

```text
projected-splat screen-covariance nulls are removed;
the radial depth / opacity-support gauge remains.
```

Implemented gate:

```text
research_experiments/gauge_fields/incidence.py::compact_poly_ellipsoid_optical_depth
tests/test_gauge_incidence.py
```

Verified:

```text
closed form vs numeric quadrature
radial gauge invariance
projected covariance-null perturbation is not null under compact ray integration
```

Next integration, if pursued:

```text
render.incidence_mode = compact_poly_ellipsoid
support_mode = derived_support_metric first
```

Use `A = inv(Sigma)` from the existing world support covariance and calibrate
`beta` so the center-ray optical depth matches current alpha initialization. Do
not promote it based on source-view fit. It must beat
`derived_support_metric/projected_conic` on the 3-camera DeepView held-out
selector without the broad-coverage failure mode seen in
`ray_gaussian_line_peak`.

## Kill Criteria

Kill or demote a support mode if:

- it improves source-view PSNR but consistently loses held-out cameras,
- it needs far more coverage/alpha mass to match source-view fit,
- it has low X-map occupancy or collapsed X-map entropy,
- camera perturbation shows billboard sheets or camera-glued structure,
- it remains much slower after a clean renderer/timing path and does not buy
  structural metrics,
- a simpler baseline matches it at equal active parameter and wall-clock budget.

## Claims We Can Make Now

Safe:

```text
The repo now has a modular harness for support-mode and splat baseline ablation.
Held-out camera evaluation already changes the ranking versus source-view PSNR.
The current projected-disk model is a control, not a final geometry primitive.
```

Unsafe:

```text
oriented_slab solves surfaces
rank_adaptive_metric solves universal geometry
free_dynamic_3dgs is better overall
source-view overfit proves 3D
X-map consistency alone proves material identity
```

## Repo Hygiene Left

The gauge-thread commits are scoped. The checkout still contains unrelated dirty
work in dataset intake, video-token configs/models, generated outputs, web
viewer work, tests, and the fast-mac submodule.

Before a global cleanup:

```text
inventory dirty files by workstream
decide whether outputs are artifacts or ignored scratch
commit fast-mac inside third_party/fast-mac-gsplat before updating parent pointer
then commit dynaworld and parent gsplats_browser pointer separately
```
