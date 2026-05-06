# PowerFoam, Gauge Fields, And Novel-View Math Directions

Date: 2026-05-05

Purpose: propose rigorous next math directions that connect the local
PowerFoam cell machinery, the gauge-field representation harness, and the
DynaWorld novel-view predictive-quotient contract. This is a research note,
not an implementation plan or claim that PowerFoam proper is complete.

## Source Context Read

Docs inspected for this note:

- `research_notes/README.md`
- `research_notes/framing_the_problem/README.md`
- `research_notes/framing_the_problem/framing_3.md`
- `research_notes/training_contract_v1.md`
- `research_notes/foam_papers/foam_implementation_status_2026-05-02.md`
- `research_notes/foam_papers/powerfoam_mathematical_aspects_deep_dive.md`
- `research_notes/foam_papers/powerfoam_rasterizer_notes.md`
- `research_notes/foam_papers/powerfoam_reproduction_audit.md`
- `TODO/powerfoam_full_reproduction_todo.md`
- `research_experiments/gauge_fields/README.md`
- `research_experiments/gauge_fields/SUPPORT_MODE_ABLATION_HANDOFF.md`
- `research_experiments/gauge_fields/NEXT_PLAN.md`
- `research_notes/incidence_kernels_and_material_objects.md`
- `research_experiments/gauge_fields/incidence.py`
- `research_experiments/gauge_fields/train.py`

Current constraints that matter:

- The DynaWorld selector is held-out predictive behavior on supported query
  cameras/times, modulo gauge. Internal token equality is not the target.
- Query camera must enter only through the fixed renderer/rasterizer boundary.
- PowerFoam Metal is a strong local partial core, not the full official
  PowerFoam system. It gives bounded power cells, replay backward, strict
  quaternion height+SV primitive coverage, raytrace paths, and local verifier
  artifacts, but paper-scale heldout quality and official CUDA/Warp parity
  remain incomplete.
- The gauge-field lane already has a support/incidence harness and learned that
  source-view PSNR is a bad selector. Held-out camera quality and structural
  diagnostics must lead.
- Existing incidence work frames a useful ray-event law as optical depth
  `kappa(ell, e)`, with admissibility constraints K1-K7. Existing diagnostics
  include Pluecker witness metrics and X-map/projection health.

## Current Model

The connection between PowerFoam and gauge theory should not be "cells are
gauges" as a metaphor. The concrete bridge is:

```text
PowerFoam cell complex = local coordinate charts and bounded supports.
Gauge choice           = internal chart/frame/radius/ordering convention.
Connection             = how neighboring cell charts agree across radical faces.
Curvature/holonomy     = inconsistency of those agreements around cell cycles.
Observable             = rendered predictive map over supported query rays.
Selector               = held-out novel-view behavior plus rate/minimality.
```

The useful mathematical object is therefore a **bounded cell complex with
gauge-covariant ray incidence**:

```text
cell i:
    g_i = (p_i, r_i, R_i, material_i)
    B_i = { x : ||x-p_i|| <= r_i and pow_i(x) <= pow_j(x) for j in N(i) }

ray:
    ell = (o, d, s0, s1)

observable:
    tau_i(ell) = integral_{ell cap B_i} sigma_i(x; g_i, material_i) ds
    alpha_i(ell) = 1 - exp(-tau_i(ell))
```

The internal variables may be reparameterized. The rendered family
`Pi_S(W) = {R(W, q) : q in S}` should not change under benign gauges.

## Direction 1: Cell-Complex Gauge Connection

### Hypothesis

PowerFoam's adjacency graph can carry a discrete connection that measures
whether neighboring cells form a coherent world object instead of independent
screen-painting supports.

Each cell has a local frame:

```text
F_i(xi) = p_i + r_i R_i xi
xi in R^3
R_i = [t_i b_i n_i] in SO(3)
```

For adjacent cells `i,j`, the radical face is:

```text
H_ij = { x : x dot n_ij = h_ij }
n_ij = p_j - p_i
h_ij = 0.5 (||p_j||^2 - ||p_i||^2 + r_i^2 - r_j^2)
```

A gauge-covariant edge transport should map tangent directions from cell `i`
to cell `j` across this shared face. A simple first transport is the relative
frame:

```text
U_ij = R_j^T R_i
```

but that ignores the face. A face-aware transport should preserve the radical
face tangent plane. Let:

```text
f_ij = normalize(n_ij)
P_ij = I - f_ij f_ij^T
T_i = orthonormal_basis(P_ij R_i[:,0:2])
T_j = orthonormal_basis(P_ij R_j[:,0:2])
U_ij^face = T_j^T T_i       # SO(2) if both projected bases are valid
```

Then a cell loop `C = (i0,i1,...,ik=i0)` has holonomy:

```text
H_C = U_{i0,i1}^face U_{i1,i2}^face ... U_{i_{k-1},i_k}^face
curv(C) = ||log_SO2(H_C)||^2
```

For a locally coherent surface/material sheet, small loops on well-witnessed
cell neighborhoods should have low holonomy. For billboard/camera-glued
solutions, frames can disagree arbitrarily while still fitting the source
view.

### Mechanism Sketch

Start as diagnostics only:

```text
edge_weight_ij = mean over query rays of contribution mass crossing/near face ij
loop_weight_C  = min edge_weight on C

report:
    weighted_holonomy_mean
    weighted_holonomy_p90
    holonomy_vs_heldout_error_correlation
```

Only if the diagnostic predicts held-out failure, consider a regularizer:

```text
L_conn = lambda_conn sum_{(i,j)} edge_weight_ij
         || log_SO2(U_ij^face) ||_Huber
```

Do not add this permanently unless it improves held-out cameras at matched
rate/capacity. Under framing 3, a connection loss is an escape hatch or
representative pressure, not the supervised object.

### Falsification

Cheap diagnostic:

1. For a trained PowerFoam Metal heldout run, compute face-frame holonomy on
   the active `cech_aabb` graph.
2. Bucket cells by held-out residual contribution.
3. Test whether high-residual regions have higher holonomy than low-residual
   regions at matched alpha.

Supports if:

```text
high heldout residual cells have significantly higher weighted holonomy,
and source-view residual does not show the same relation.
```

Weakens if:

```text
holonomy is uncorrelated with heldout residual,
or the metric mostly tracks low alpha / bad visibility.
```

Kill as a loss if:

```text
L_conn improves holonomy but lowers heldout PSNR/SSIM or only improves source view.
```

## Direction 2: Witnessed Power Faces

### Hypothesis

The main PowerFoam/NVS failure is not just weak point cloud initialization; it
is that many power faces are under-witnessed by multi-view rays. Unwitnessed
faces are gauge-free: the optimizer can move radical planes and internal
surfaces without changing source renders.

Existing Pluecker witness metrics use:

```text
W_i = sum_k w_ki (I - d_k d_k^T)
```

For one ray, `W_i` has eigenvalues `{1,1,0}`, so depth is not witnessed. Extend
this from elements to **faces**:

```text
face ij has representative point x_ij
ray weight w_kij = contribution mass whose clipped interval endpoint is face ij

W_ij = sum_k w_kij (I - d_k d_k^T)
lambda_min_ij = eigmin(W_ij)
face_witness_ij = lambda_min_ij / trace(W_ij)
```

Here `w_kij` should come from replay metadata if available: the endpoint winner
for `t_near` or `t_far` was neighbor `j`. If winner metadata is not exposed,
approximate by rays where the active interval endpoint lies within epsilon of
the radical face.

### Mechanism Sketch

Diagnostic metrics:

```text
cell_witness_i      = eigmin(sum_k w_ki (I-dd^T)) / trace(...)
face_witness_ij     = eigmin(W_ij) / trace(W_ij)
unwitnessed_alpha   = sum_i alpha_i * 1[cell_witness_i < eps]
unwitnessed_faces   = mean_ij 1[face_witness_ij < eps]
```

Training candidate, only after diagnostic evidence:

```text
L_support_budget =
    lambda_unwitnessed * sum_i alpha_i stopgrad(1[cell_witness_i < eps])
```

This is not a geometry teacher. It is a rate/minimality pressure: do not spend
opacity on cells whose 3D position is not witnessed by the current `D_var`
support.

### Falsification

Run on current DeepView PowerFoam artifacts:

```text
plot heldout residual vs cell_witness and face_witness
plot selected checkpoint PSNR vs unwitnessed_alpha over steps
compare random init, pycolmap init, plane-sweep init, EX4DGS init
```

Supports if:

```text
best heldout rows have lower unwitnessed_alpha / higher face_witness
independent of source PSNR.
```

Weakens if:

```text
witness scores mostly track number of points/cells,
or high witness does not separate EX4DGS/pycolmap/plane-sweep quality.
```

## Direction 3: Power-Cell Incidence As A Compact Ellipsoid Limit

### Hypothesis

The gauge-field compact polynomial ellipsoid and PowerFoam bounded power cells
are two endpoints of the same incidence family:

```text
compact ellipsoid:
    tau_i(ell) = integral beta_i [1 - (x-mu_i)^T A_i (x-mu_i)]_+^k ds

PowerFoam cell:
    tau_i(ell) = integral_{ell cap B_i} sigma_i(x) ds
```

The ellipsoid has a smooth compact support but no neighbor-owned faces. The
PowerFoam cell has exact neighbor-owned faces but harder topology. A
mathematically clean bridge is:

```text
soft power ownership:
    omega_i(x) = softmax_j( -gamma pow_j(x) )_i

bounded density:
    sigma_i^gamma(x) =
        beta_i [1 - ||x-p_i||^2/r_i^2]_+^k omega_i(x)

limit:
    gamma -> infinity gives hard power-cell ownership inside sphere
```

Ray optical depth:

```text
tau_i^gamma(ell) =
    integral_{s0}^{s1} sigma_i^gamma(o+s d) ds
```

This gives a differentiable continuation from compact gauge-field events to
hard PowerFoam cells. The hard local Metal renderer remains the desired fast
endpoint, but the soft version is useful for math tests and topology
annealing.

### Mechanism Sketch

Three-stage ablation:

```text
A. compact_poly_ellipsoid only
B. soft_power_ellipsoid with gamma schedule 1 -> 32
C. hard PowerFoam raytrace with same initialized p_i, r_i, beta_i
```

All use the same held-out camera split, same cell count, and same rate budget.

Key equations for `B`:

```text
pow_i(x) = ||x-p_i||^2 - r_i^2
omega_i(x) = exp(-gamma pow_i(x)) / sum_{j in N(i) union i} exp(-gamma pow_j(x))
phi_i(x) = [1 - ||x-p_i||^2/r_i^2]_+^k
sigma_i(x) = beta_i phi_i(x) omega_i(x)
```

Candidate regularity diagnostic:

```text
partition_error(x) = |sum_i omega_i(x) - 1|
boundary_entropy(x) = -sum_i omega_i(x) log omega_i(x)
```

For hard PowerFoam, boundary entropy should collapse near faces; for soft
incidence it measures ambiguity.

### Falsification

Supports if:

```text
soft_power_ellipsoid improves heldout over compact ellipsoid at matched source
fit and transitions smoothly to hard PowerFoam without a quality cliff.
```

Weakens if:

```text
soft ownership merely increases source fit, saturates alpha coverage, or
loses to compact ellipsoid and hard PowerFoam on heldout.
```

Kill if:

```text
gamma annealing creates unstable topology or high broad-coverage failure like
the suspicious ray_gaussian_line_peak row.
```

## Direction 4: Gauge-Covariant Dynamic PowerFoam Transport

### Hypothesis

Dynamic feature foam can fit by repainting a mostly fixed lattice. The missing
math is a transport law that distinguishes material motion from appearance
change. Gauge theory gives the right object: a connection over time for each
cell's local chart.

Cell chart at time `tau`:

```text
g_i(tau) = (p_i(tau), r_i(tau), R_i(tau), material_i(tau))
```

Temporal connection:

```text
A_i(tau) = g_i(tau)^{-1} d g_i(tau) / d tau
```

Discrete version:

```text
Delta_p_i = (p_i(t+1) - p_i(t)) / r_i(t)
Delta_R_i = log_SO3(R_i(t)^T R_i(t+1))
Delta_r_i = log(r_i(t+1) / r_i(t))
```

Appearance residual:

```text
Delta_c_i = c_i(t+1) - c_i(t)
```

The dynamic representation should not be judged by motion magnitude alone.
Instead measure whether held-out predictive behavior depends on transported
material motion rather than repainting:

```text
motion_to_repaint_ratio =
    mean_i ||Delta_p_i, Delta_R_i, Delta_r_i||_w /
    mean_i ||Delta_c_i||_w
```

where weights are contribution mass on held-out rays.

### Mechanism Sketch

Diagnostic first:

```text
freeze transport, train appearance only
freeze appearance, train transport only
train both
wrong-time transport swap
wrong-cell material swap
```

Use held-out camera/time metrics:

```text
L_time_dep = PSNR(full) - PSNR(zero transport)
L_material_dep = PSNR(full) - PSNR(shuffled material)
```

Candidate connection regularizer:

```text
L_temporal_conn =
    sum_i w_i ||Delta_R_i||^2
  + sum_i w_i ||Delta_r_i||^2
  + sum_(i,j) w_ij ||Delta_p_i - transported_ij(Delta_p_j)||^2
```

Do not penalize motion magnitude directly. Penalize non-materially coherent
motion only when it predicts held-out failure.

### Falsification

Supports if:

```text
transport-only or material-preserving variants retain heldout quality better
than appearance-only repainting, especially under target camera changes.
```

Weakens if:

```text
appearance-only matches full heldout quality, or transport metrics do not
change wrong-world/wrong-time probes.
```

## Direction 5: Predictive Quotient With Gauge-Canonical Assets

### Hypothesis

A PowerFoam asset can be a good DynaWorld world token only if its gauge choices
are irrelevant to supported rendered behavior and its rate pressure selects a
compact representative. The right target is not canonical cell identity, but
predictive equivalence:

```text
W equiv_S W' iff R(W, q) = R(W', q) for all q in S
```

PowerFoam-specific benign gauges:

```text
G_perm:   cell permutation
G_frame:  local tangent/bitangent rotation around normal when material is isotropic
G_texel:  detail-site permutation inside a cell
G_scale:  coupled density/radius rescalings that preserve tau on S
G_topo:   insertion/splitting of zero-contribution cells
```

Non-benign transformations are those that change held-out renders:

```text
camera-glued frame changes
source-ray-specific color caches
cell splits that only preserve source camera
radius/density changes that preserve alpha from source but fail at query views
```

### Mechanism Sketch

Define a no-grad gauge audit:

```text
for asset W:
    W_perm      = permute cells and texels
    W_rot_iso   = rotate isotropic local frames
    W_split     = split low-alpha cells preserving source alpha approximately
    W_rescale   = r_i <- a r_i, density_i <- density_i / a along fitted ray

measure:
    Delta_source = ||R(W,q_source)-R(W_g,q_source)||
    Delta_holdout = ||R(W,q_holdout)-R(W_g,q_holdout)||
```

Benign gauges should produce both deltas near zero. Source-only invariance is a
failure signal:

```text
gauge_leak_score = Delta_holdout / (Delta_source + eps)
```

This directly matches the predictive quotient view: internal equivalence is
real only if the rendered predictive family agrees on supported queries.

### Falsification

Supports if:

```text
gauge_leak_score predicts heldout underperformance across PowerFoam configs
and detects known repainting / source-camera fits.
```

Weakens if:

```text
all reasonable assets show high leak because the perturbations are too strong,
or all show low leak despite bad heldout quality.
```

## Direction 6: Curvature-Regularized Capacity Growth

### Hypothesis

Grow/prune/resample should not be driven only by contribution/error EMAs. In a
cell-complex view, capacity should be allocated where the current gauge
connection has high predictive curvature and sufficient witness support.

Define a local residual curvature score:

```text
K_i =
    heldout_residual_i
    * witness_confidence_i
    * (holonomy_i + face_boundary_entropy_i)
```

Interpretation:

- high residual + high witness means the scene is observable but not modeled;
- high residual + low witness means the data support is insufficient, so growth
  may memorize;
- high holonomy/boundary entropy means the cell complex is geometrically
  inconsistent or topologically underresolved.

### Mechanism Sketch

Growth proposal:

```text
if K_i high and witness_i high:
    split cell i along the dominant residual/witness direction
elif residual_i high and witness_i low:
    do not grow; report D_var/support insufficiency
elif contribution_i low:
    prune or lower rate budget
```

Dominant split direction can come from the witness matrix:

```text
eigvec_max(W_i) gives most constrained transverse direction
eigvec_min(W_i) gives least witnessed depth direction
```

Do not split along `eigvec_min` unless new query support is added; that is the
gauge-free depth direction.

### Falsification

Compare three growth policies in a small static posed-camera run:

```text
A. contribution/error EMA only
B. residual * witness
C. residual * witness * holonomy
```

Supports if:

```text
C improves heldout PSNR/SSIM or lowers heldout residual at equal cell count
without source-only overfit.
```

Weakens if:

```text
C mostly avoids growth and underfits, or B/C fail to beat the simpler EMA.
```

## Common Diagnostics Table

Any proposed mechanism above should log these before becoming a training loss:

| diagnostic | object | supports direction if | failure mode caught |
| --- | --- | --- | --- |
| `cell_witness_min_eig` | cells | high values correlate with heldout quality | source-ray depth gauge |
| `face_witness_min_eig` | radical faces | under-witnessed faces explain residual | missing multi-view topology |
| `weighted_holonomy_p90` | adjacency loops | high values localize heldout errors | incoherent cell frames |
| `gauge_leak_score` | asset perturbations | source invariance without heldout invariance predicts failure | source-camera gauge cheat |
| `unwitnessed_alpha` | rendered opacity | lower values correlate with heldout quality | opacity spent on gauge-free cells |
| `motion_to_repaint_ratio` | dynamic cells | heldout depends on transport, not only feature changes | repainting fixed lattice |
| `boundary_entropy` | soft power ownership | annealing sharpens without heldout cliff | broad alpha coverage / topology blur |

Primary selector:

```text
heldout PSNR / L1 / SSIM / LPIPS by camera delta, time delta, and observation budget
```

Diagnostics should only graduate into losses when they predict this selector.

## Minimal Experiment Ladder

### Ladder A: No-Grad Audits On Existing Artifacts

No training changes.

```text
1. cell/face witness on PowerFoam raytrace runs
2. cell-frame holonomy on active Cech/AABB graph
3. gauge perturbation leak score
4. dynamic motion-vs-repaint dependency probes
```

Pass condition:

```text
at least one diagnostic correlates with heldout residual across existing
configs better than source PSNR does.
```

### Ladder B: Synthetic Known-Truth Cell Complex

Create small scenes with known topology:

```text
plane sheet
two crossing planes
thin cylinder
transparent/fog volume
moving rigid patch
```

Each scene should have:

```text
known camera poses
train/heldout split
known cell adjacency or sampled surface
known expected witness degeneracies
```

Pass condition:

```text
diagnostics identify the intended degeneracy before heldout quality collapses.
```

### Ladder C: Held-Out DeepView Probe

Use the current DeepView heldout selector, but keep scope narrow:

```text
same data split
same cell count
same rate/export cap
same optimizer budget
one diagnostic-derived loss or policy at a time
```

Pass condition:

```text
improves heldout metrics at matched source fit and rate, not just source PSNR.
```

## Branches And Backtracks

### Branch: This Is Just Another Regularizer

Could be true. A connection/holonomy loss might merely smooth frames and reduce
capacity. If it does not predict heldout errors before training, do not add it.

### Branch: The Bottleneck Is Still Initialization, Not Gauge

Could be true. The current PowerFoam acceptance audit says clean DeepView
geometry remains weak and mostly two-camera supported. If witness metrics just
restate that fact, the next lever is better `D_var` support or better point
clouds, not a new gauge loss.

### Branch: Gauge Diagnostics Need Multi-View Support To Mean Anything

Likely true. Single-camera clips have genuine depth ambiguity. A low witness
score should route to "unsupported query/gauge-free direction," not "bad
representation."

### Branch: Power Cells Are Too Hard For Early World Tokens

Possible. Compact ellipsoids or soft power ownership may be a better
curriculum, with hard PowerFoam only after the world asset has enough support.
The soft-to-hard direction above is meant to test this without inventing an
unrelated representation.

## Decision Implications

If witness and holonomy diagnostics predict heldout residual:

```text
implement them as logged first-class metrics in the PowerFoam/gauge harness;
then test one light representative-pressure loss.
```

If diagnostics do not predict heldout behavior:

```text
do not promote gauge math to training; focus on data support, camera model,
and existing PowerFoam acceptance blockers.
```

If soft power incidence beats compact ellipsoid and bridges to hard PowerFoam:

```text
use it as the differentiable topology curriculum for future cell-complex
world tokens.
```

If soft ownership fails:

```text
keep compact ellipsoid as the gauge-field incidence candidate and hard
PowerFoam as the separate renderer/system path.
```

## Open Questions

- Can replay backward expose enough endpoint-winner metadata to compute
  face-level witness without expensive recomputation?
- What are the shortest reliable cycles in a `cech_aabb` graph? Triangle
  cycles may be noisy if edges are false positives; bounded face cycles may
  require more structure.
- Which gauge perturbations are truly benign for height+SV materials? Local
  tangent rotation is not benign when texel coordinates or anisotropic SV axes
  are fixed.
- Can the same diagnostics work for both static PowerFoam and dynamic feature
  foam, or do dynamic appearance shortcuts require separate probes?
- Does held-out quality improve more from better camera/model support than
  from any cell-complex regularizer? This must remain an active backtrack.
