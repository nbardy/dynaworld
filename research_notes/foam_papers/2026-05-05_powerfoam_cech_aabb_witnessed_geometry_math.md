# PowerFoam Cech/AABB, Witnessed Complexes, And Clean Geometry Math

Date: 2026-05-05

Scope: new math proposals only. This note is not an implementation claim and
does not update the canonical TODO/status files. It assumes the current local
boundary: Metal has a partial trainable bounded-cell raster/raytrace/backward
core with `cech_aabb` adjacency and synthetic 4K artifacts, while full official
PowerFoam remains blocked by official CUDA/Warp parity fixture generation and
paper-scale clean heldout quality.

The goal is to turn the next PowerFoam quality work away from another
hyperparameter sweep and toward verifiable geometry/topology mechanisms:

- Cech/AABB as a fast conservative support graph.
- witnessed power complexes as train-only topology evidence.
- differentiable adjacency signals that do not make Metal traversal soft.
- regular triangulation as a teacher/verifier rather than the first production
  path.
- uncertainty-weighted clean geometry that improves heldout without using
  heldout RGB.

## Current Working Model

Local PowerFoam quality is probably not blocked by primitive Metal math alone.
The stronger local evidence says:

- `cech_aabb` is the fast selected synthetic 4K path.
- SciPy-backed regular triangulation has been render-verified on the small
  parity path.
- Clean DeepView heldout remains below paper-acceptance thresholds even after
  distortion-consistent OPENCV_FISHEYE pycolmap and several negative controls.
- The selected clean point cloud is not invisible; it has train visibility and
  nonblank heldout alpha previews, but unique-camera support is mostly two-view.

Provisional hypothesis:

```text
heldout failure = weak clean geometry support
               + topology instability / false-edge side effects
               + appearance absorbing geometric error
```

This note proposes mechanisms that produce local verifier artifacts before any
long schedule is allowed.

## Symbols

PowerFoam cells:

```text
i, j                    cell indices
p_i in R^3              cell center / weighted site
r_i > 0                 support radius and power weight radius
u_i >= 0                geometry uncertainty scalar
sigma_i                 density
theta_i                 local material/detail state
```

Power distance:

```text
pi_i(x) = ||x - p_i||^2 - r_i^2
```

Uncertainty-adjusted power distance, used only for diagnostics/selection unless
an ablation admits it:

```text
pi_i^u(x) = ||x - p_i||^2 - r_i^2 + lambda_u * u_i
```

Interpretation: high-uncertainty cells should need more evidence before they
win topology or support decisions.

Cech overlap edge:

```text
E_cech = {(i, j): ||p_i - p_j|| <= r_i + r_j}
```

AABB implementation detail:

```text
AABB_i = [p_i - r_i, p_i + r_i]
candidate(i, j) iff AABB_i intersects AABB_j
edge(i, j) iff candidate(i, j) and Cech overlap passes
```

Regular-triangulation edge:

```text
E_reg = weighted-Delaunay / regular-triangulation 1-skeleton
```

`E_reg` should be a sparse face-neighbor graph. `E_cech` is a conservative
superset candidate: cheaper and safer for local Metal traversal, but it can
contain false edges.

Train-only witnesses:

```text
w in W_train
W_train = track points, ray samples, support-mask samples, or source-view
          residual-free geometry probes derived from train observations only
```

Heldout RGB, heldout residuals, and heldout feature maps are not witnesses.

## Proposal A: Witnessed Power Complex

### Hypothesis

Raw Cech adjacency says two support balls overlap, but it does not say any
observed train geometry actually witnesses their shared boundary. A witnessed
power complex can distinguish useful topology from false overlap using only
train observations.

### Definition

For each train-only witness `w`, compute soft ownership under
uncertainty-adjusted power distances:

```text
q_i(w) = softmax_i(-pi_i^u(w) / tau_w)
```

For a hard diagnostic version, let `N_k(w)` be the `k` cells with smallest
`pi_i^u(w)`.

Witnessed edge score:

```text
S_ij = sum_{w in W_train} rho(w) * q_i(w) * q_j(w) * g_ij(w)
```

where:

```text
rho(w)      train-only confidence of witness w
g_ij(w)     local gate: w lies near the radical slab between i and j
```

Radical slab gate:

```text
n_ij = p_j - p_i
h_ij = 0.5 * (||p_j||^2 - ||p_i||^2 + r_i^2 - r_j^2)
slack_ij(w) = |w dot n_ij - h_ij| / (||n_ij|| + eps)
g_ij(w) = exp(-slack_ij(w)^2 / (2 * tau_slab^2))
```

Hard witnessed edge:

```text
E_wit = {(i, j): S_ij >= eta_edge and Cech(i, j)}
```

Do not replace `E_cech` with `E_wit` in Metal first. Start by logging edge
labels:

```text
Cech edge classes:
  witnessed_true       Cech edge with high S_ij
  unwitnessed_false?   Cech edge with near-zero S_ij
  unstable             S_ij crosses eta_edge repeatedly over training
```

### Why This Could Help

The current clean DeepView blocker is not just "more cells." More cells and
top-ranked sampling did not solve heldout. A witness complex asks whether the
cells kept by the clean artifact form topology that source observations can
actually support.

Expected useful signals:

- high heldout residuals concentrate on cells with low witnessed degree
- high traversal counts concentrate on unwitnessed false Cech edges
- source cross-holdout errors correlate with low `S_ij` around contributing
  cells

### Falsifier

If `S_ij` does not correlate with heldout-free internal validation, traversal
cost, gradient noise, or source leave-one-camera-out residuals, the witness
complex is decorative. Drop it before writing a trainer path.

## Proposal B: Differentiable Adjacency Without Soft Metal Traversal

### Hypothesis

Hard graph refreshes create discontinuous optimization, but the renderer should
stay hard/CSR for speed and correctness. The differentiable object should be an
auxiliary topology ledger and regularizer, not a soft renderer.

### Soft Cech Weight

```text
d_ij = ||p_i - p_j||
a_ij = (r_i + r_j - d_ij) / tau_cech
w_ij = sigmoid(a_ij)
```

Useful losses:

```text
L_degree = mean_i relu(sum_j w_ij - degree_budget_i)^2
L_churn  = mean_ij |w_ij(t) - stopgrad(w_ij(t - K))|
L_margin = mean_ij exp(-abs(a_ij) / tau_margin)
```

`L_margin` discourages edges living exactly at birth/death threshold. It should
be logged before it is optimized.

### Witnessed Edge Probability

Use train witnesses to define a smoother edge confidence:

```text
P_ij = 1 - product_{w in W_train} (1 - rho(w) * q_i(w) * q_j(w) * g_ij(w))
```

Candidate regularizer:

```text
L_unwitnessed_cech = sum_{(i,j) in E_cech} (1 - stopgrad(P_ij)) * w_ij
```

This says: keep Cech correctness, but make unsupported overlaps expensive.

### Radius Split Diagnostic

Radii currently serve at least two roles:

```text
r_render_i     support sphere / alpha coverage / culling
r_topology_i   adjacency and radical-plane neighborhood
```

Diagnostic ablation:

```text
L_radius_tie = ||log r_render_i - log r_topology_i||^2
```

If a split improves internal cross-view quality, the single-radius parameter is
overloaded. If it only improves train alpha and hurts cross-view, it is a
capacity leak.

### Metal Implication

Keep Metal input as hard CSR adjacency:

```text
row_ptr, col_idx, optional_edge_priority, optional_edge_class
```

The differentiable pieces live in Python/Torch prepasses:

- compute soft weights and witness scores
- emit graph diagnostics JSON
- optionally sort Cech neighbors by priority
- optionally cap false edges only after a missing-edge verifier proves safety

No soft adjacency should enter the shader until a frozen-state parity verifier
shows it changes neither forward renders nor gradients beyond tolerance.

## Proposal C: Regular Triangulation As Teacher And Guardrail

### Hypothesis

Regular triangulation is mathematically closer to the minimal power-cell face
graph, but Cech/AABB is better aligned with the current Metal performance path.
Use regular triangulation as a teacher, not the first selected production path.

### Teacher Labels

For a frozen state:

```text
E_cech = fast conservative graph
E_reg  = SciPy/Qhull regular-triangulation graph
```

Label:

```text
regular_edge      (i,j) in E_cech and (i,j) in E_reg
cech_extra        (i,j) in E_cech and (i,j) not in E_reg
reg_missing       (i,j) in E_reg  and (i,j) not in E_cech   # should be zero or explained
```

`reg_missing` is the verifier-critical category. If it is nonzero, either the
Cech/AABB graph is not conservative under the current decoding, or numerical
tolerances are wrong.

### Edge-Priority Distillation

Train or fit a simple edge priority score from local features, but only as a
diagnostic table first:

```text
features_ij = [
  d_ij / (r_i + r_j + eps),
  abs(r_i - r_j) / (r_i + r_j + eps),
  witnessed score S_ij,
  radical slab support quantiles,
  min unique-camera support of i/j,
  uncertainty u_i + u_j,
  Cech margin a_ij,
]

target_ij = 1[(i,j) in E_reg]
```

If this predicts `E_reg` well, the same score can order Cech neighbors for
traversal and diagnostics. It should not drop edges until safety is proven.

### Frozen-State Sensitivity Verifier

For each saved state, render and backprop under four graphs:

```text
G0 = cech_aabb
G1 = regular_triangulation
G2 = cech_aabb with cech_extra edges shuffled later in neighbor order
G3 = cech_aabb with sampled cech_extra edge dropout, preserving all regular edges
```

Measurements:

```text
max_abs_rgb_delta
max_abs_alpha_delta
mean_depth_delta
center_grad_cosine
radius_grad_cosine
density_grad_cosine
height_grad_cosine
sv_axis_grad_cosine
mean/p95 traversal steps
mean/p95 degree
```

Interpretation:

- If `G0` vs `G1` is forward/gradient equivalent, Cech false edges are likely
  a speed issue, not a quality issue.
- If `G2` changes gradients, neighbor order is a hidden numerical variable.
- If `G3` improves internal heldout without changing train, false-edge noise is
  cross-view relevant.

## Proposal D: Uncertainty-Weighted Clean Geometry

### Hypothesis

Clean geometry is not binary. A point from pycolmap/SfM has support,
reprojection error, parallax, and track covariance. PowerFoam should treat
uncertain cells as uncertain in selection, initialization, adjacency, and
geometry regularization.

### Train-Only Uncertainty

For each clean track/cell candidate:

```text
track_len_i               number of source observations
unique_camera_count_i     number of source cameras
unique_frame_count_i      number of source frames
reproj_med_i, reproj_p90_i
parallax_med_i            source-view bearing angle spread
triang_cov_i              approximate 3D covariance from bearings/Jacobian
static_score_i            source-only temporal consistency
```

Define normalized uncertainty:

```text
u_i =
    a / sqrt(track_len_i + 1)
  + b / sqrt(unique_camera_count_i + 1)
  + c * reproj_med_i / reproj_scale
  + d * cond(triang_cov_i)
  + e * relu(parallax_min - parallax_med_i)
  + f * (1 - static_score_i)
```

Use `u_i` in four places:

1. Selection:

```text
score_i = coverage_i - lambda_u * u_i - lambda_red * redundancy_i
```

2. Radius initialization:

```text
r_i = clamp(k * sqrt(lambda_tangent_1 + lambda_tangent_2), r_min, r_max)
r_i *= exp(gamma_u * u_i)      # uncertain points get cautious coverage
```

3. Topology confidence:

```text
pi_i^u(x) = pi_i(x) + lambda_u * u_i
```

4. Geometry regularization:

```text
L_motion_i, L_radius_i, L_normal_i weighted by 1 / (u_i + eps)
```

High-confidence clean cells should be stable. Low-confidence cells should be
allowed to adapt but should not dominate topology.

### Surfel Lift

Point-only init asks photometric gradients to learn normals and heights. Lift
each track into a local surfel from train-only geometry:

```text
normal_i = smallest-eigenvector of local track covariance
tangent radii = sqrt(two larger covariance eigenvalues)
height prior = bounded function of depth/reprojection uncertainty
```

Verifier-first claims:

- step-0 alpha/depth improves without color optimization
- internal leave-one-train-camera-out improves before real heldout is touched
- topology degree/churn decreases for high-confidence cells

## No-Leakage Heldout Rules

These proposals are only useful if they preserve the Dynaworld heldout
contract. Rules:

1. Build `W_train`, uncertainty, witness scores, graph priorities, checkpoint
   selectors, and geometry gates from source/train observations only.
2. Do not use heldout RGB, heldout residuals, heldout features, heldout masks,
   or heldout depth estimates to choose cells, tune weights, select
   checkpoints, or accept a mechanism.
3. Heldout camera intrinsics/extrinsics may be used only for the final
   evaluation protocol already defined by the dataset split. If a stress test
   uses heldout pose without RGB, label it `pose-only diagnostic`, not a
   selector.
4. Use internal source leave-one-camera-out validation for mechanism selection:

```text
train cameras -> split into source-fit cameras and internal-query camera
real heldout camera -> untouched until final report
```

5. Any external geometry oracle, pretrained reconstruction, EX4DGS init, or
   learned depth/normal prior is diagnostic unless it is explicitly admitted as
   non-clean. It cannot satisfy paper-clean acceptance.
6. Report behavior by query support: train view, internal heldout, official
   heldout, and unsupported extrapolation are separate buckets.

## Verifier Ideas

### V0: Static Graph Equivalence

Input: one frozen saved state.

Output JSON:

```text
cech_edges
regular_edges
reg_missing_edges
cech_extra_edges
forward_delta_rgb
forward_delta_alpha
gradient_cosines
degree_quantiles
traversal_step_quantiles
```

Pass condition for using Cech/AABB as correctness graph:

```text
reg_missing_edges == 0
forward_delta_rgb <= tolerance
forward_delta_alpha <= tolerance
gradient_cosines >= threshold
```

### V1: Witness Correlation Report

Input: clean init point cloud, train cameras, saved render state.

Compute:

```text
S_ij witnessed scores
cell witnessed degree
cell uncertainty u_i
cell train visibility/support
source leave-one-camera-out residual buckets
```

Pass condition for continued work:

```text
low witnessed degree or high u_i predicts internal heldout residual
```

If not, witnessed topology is not the next lever.

### V2: No-Leakage Internal Holdout Selector

Run source-camera rotation:

```text
for c in train_cameras:
    fit/score using train_cameras - {c}
    evaluate c as internal heldout
```

Forbidden:

```text
official heldout RGB in selection
official heldout residual in checkpoint choice
```

Pass condition:

```text
internal heldout bucket improves before official heldout is evaluated
```

### V3: Uncertainty Ablation Matrix

Small matrix, same seed and schedule:

```text
baseline clean init
uncertainty-weighted selection only
uncertainty-weighted surfel init only
witnessed topology regularizer only
selection + surfel + witnessed regularizer
```

Required outputs:

```text
step-0 train/internal-heldout alpha/depth
final train/internal-heldout PSNR/SSIM/L1
official heldout final metrics, evaluated once per selected candidate
graph churn and degree quantiles
coverage verifier result
```

### V4: Metal Safety Gate

Before adding any shader-facing topology feature:

```text
1. CPU/Torch reference graph emits CSR plus optional edge_class/priority.
2. Metal consumes the same CSR and produces identical forward/backward to the
   old path when priority/class is ignored.
3. Sorted-neighbor or capped-neighbor modes have frozen-state forward and
   gradient parity against uncapped Cech within tolerance.
4. 4K verifier records traversal benefit without quality regression.
```

## Proposed Order Of Work

1. Build V0 on frozen states: prove whether Cech/AABB false edges matter
   relative to regular triangulation.
2. Build V1: compute witness/uncertainty ledgers with no training changes.
3. Run V2 source-camera rotation on the selected clean DeepView artifact.
4. Only if V1/V2 correlate, add the cheapest training ablation:
   uncertainty-weighted selection or surfel lift.
5. Add differentiable adjacency regularization only after frozen graph
   equivalence and witness correlation are both positive.
6. Move anything into Metal only after the CPU/Torch verifier says the graph
   change is both safe and useful.

## Kill Criteria

Stop this line if any two are true:

- Cech vs regular-triangulation frozen gradients are equivalent.
- Witness degree/uncertainty do not predict internal heldout residuals.
- Source leave-one-camera-out does not predict official heldout direction.
- Uncertainty-weighted selection does not improve step-0 internal heldout alpha
  or depth.
- Surfel lift improves train only while internal heldout worsens.

If killed, return to geometry construction: stronger clean multi-camera tracks,
ONNX-backed ALIKED/LightGlue on an external host, or a separate clean static
mask, rather than adding topology machinery.

