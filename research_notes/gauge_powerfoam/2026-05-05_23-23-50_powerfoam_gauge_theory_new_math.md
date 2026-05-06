# PowerFoam Gauge Theory New Math

Date: 2026-05-05 23:23:50 +0700

Scope: research math note only. This does not claim full official PowerFoam on
Metal. It assumes the current local state: fast/trainable Metal core and 4K
local gates exist, official CUDA/Warp fixture is absent on this Mac, and paper
acceptance is blocked by low heldout quality.

Goal: define gauge-theoretic directions that can become concrete PowerFoam /
dynamic foam experiments. Keep implementation-adjacent tests separate from
speculative geometry.

## Working Object

PowerFoam gives a cell complex with local charts:

```text
cell i:
    p_i in R3                 center
    r_i > 0                   radius / power weight scale
    R_i in SO3                local frame [t_i, b_i, n_i]
    z_i                       texel / height / SV / material state
    N_i                       selected neighbors from Cech/AABB or raytrace graph

power value:
    P_i(x) = ||x - p_i||^2 - r_i^2

owned bounded cell:
    B_i = { x : ||x-p_i|| <= r_i and P_i(x) <= P_j(x) for j in N_i }
```

The gauge variables are not metaphors. They are internal choices that should
not change the rendered predictive family when they are benign:

```text
cell permutation
local tangent-frame rotation when material is isotropic
texel chart reparameterization
zero-contribution cell insertion
equivalent neighbor ordering
source-camera-only depth shifts are NOT benign
```

The observable is still ray rendering:

```text
ray ell = (o, d, s_min, s_max)
tau_i(ell) = integral over s where x=o+s*d lies in B_i of sigma_i(x,z_i) ds
alpha_i(ell) = 1 - exp(-tau_i(ell))
render(ell) = front_to_back_compose_i color_i(ell) alpha_i(ell)
```

## Near-Term Implementable Tests

These are intended to be small scripts or trainer-side diagnostics before any
new architecture fork.

### 1. Cech/AABB Topological Witness Audit

Hypothesis: the selected Cech/AABB graph may be fast at 4K but expose the
wrong topology to heldout rays. Some radical faces and cells are source-fit
degrees of freedom because no multi-view ray family actually witnesses them.

For ray `k` with direction `d_k`, define the ordinary point witness matrix:

```text
W_i = sum_k m_ki (I - d_k d_k^T)
```

where `m_ki` is contribution mass from cell `i` on ray `k`. Low `eigmin(W_i)`
means the cell has mostly one-view support and depth is gauge-free.

Face witness should be the next diagnostic, because PowerFoam topology lives
at radical faces. For adjacent cells `i,j`:

```text
face normal:
    f_ij = normalize(p_j - p_i)

radical plane:
    H_ij = { x : dot(x, p_j-p_i) = 0.5*(||p_j||^2-||p_i||^2+r_i^2-r_j^2) }

face event weight:
    m_kij = contribution mass on ray k whose clipped interval endpoint is H_ij

face witness:
    W_ij = sum_k m_kij (I - d_k d_k^T)
    witness_ij = eigmin(W_ij) / (trace(W_ij) + eps)
```

If endpoint-winner metadata is missing, approximate `m_kij` by testing whether
`abs(P_i(x_end)-P_j(x_end)) < eps_face` at the active interval endpoint.

Topological witness for Cech/AABB:

```text
edge_alive_ij =
    aabb_overlap(B_i_box, B_j_box)
    and dist(p_i,p_j) <= cech_scale*(r_i+r_j)

edge_witness_ij = edge_alive_ij * witness_ij
unwitnessed_topology =
    sum_ij edge_alive_ij * contribution_ij * 1[witness_ij < eps_w]
    / (sum_ij edge_alive_ij * contribution_ij + eps)
```

Implementation test:

```text
compute on existing candidate outputs, no training:
    cell_witness_p10/p50
    face_witness_p10/p50
    unwitnessed_topology
    heldout_residual_by_low_witness_cell
    heldout_residual_by_low_witness_face
```

Supports the direction if low face witness predicts heldout residual better
than source residual or alpha mass alone. If it only tracks cell count or empty
alpha, it is not a useful gauge metric yet.

Metal gradient implication:

```text
diagnostic mode:
    no gradients required; replay endpoint data is enough.

loss mode:
    needs d interval_endpoint / d(p_i,r_i,p_j,r_j)
    needs d m_kij / d alpha/contribution if used as soft weight
    should stopgrad witness gates initially to avoid eigenvalue instability.
```

Do not make Cech/AABB selection itself differentiable at first. Treat topology
as piecewise constant per step or per refresh window.

### 2. Gauge-Covariant Transport Of Texels And Surface Variables

Hypothesis: current dynamic foam can repaint fixed supports. A useful dynamic
PowerFoam representation should transport material variables across local cell
charts, not merely refit color at each frame.

Each cell has a local chart:

```text
x_i(u,t) = p_i(t) + r_i(t) R_i(t) [u1,u2,h_i(u,t)]
u in local texel domain
z_i(u,t) = material/color/feature at texel u
```

For adjacent cells `i,j`, define a face-aware chart transport. Let `Q_i_ij` be
a 2D basis for the radical face tangent plane expressed in cell `i` local
coordinates:

```text
F_ij = I - f_ij f_ij^T
T_i_ij = orthonormal_basis(F_ij R_i[:,0:2])
T_j_ij = orthonormal_basis(F_ij R_j[:,0:2])
U_ij = T_j_ij^T T_i_ij       # SO2 edge transport after polar projection
```

Transport a texel coordinate from chart `i` to chart `j` through a face point:

```text
x = p_i + r_i R_i [u_i1,u_i2,h_i(u_i)]
u_j_hat = first2( R_j^T (x - p_j) / r_j )
z_i_to_j(u_j_hat) = sample(z_i, u_i)
```

Candidate covariant residual:

```text
R_tex_ij =
    integral over shared face support
        w_ij(u) || z_j(u_j) - T_z_ij z_i(u_i) ||_robust du
```

`T_z_ij` is identity for scalar color/features, an SO2/SO3 action for tangent
vectors, and a pullback for height gradients:

```text
scalar:          T_z_ij z = z
tangent vector:  v_j = U_ij v_i
height gradient: grad_j h = U_ij grad_i h * (r_j/r_i)
normal:          n_j should match R_j[:,2], not a transported scalar
```

Temporal transport for cell `i`:

```text
A_i(t) = R_i(t)^T R_i(t+dt)
u_hat(t+dt) =
    first2( R_i(t+dt)^T (x_i(u,t) + v_i(u,t) dt - p_i(t+dt)) / r_i(t+dt) )

R_time_i = z_i(u_hat,t+dt) - T_time_i z_i(u,t)
```

Near-term implementation:

```text
1. no-grad texel transport consistency on trained static height+SV runs
2. wrong-neighbor transport swap negative control
3. wrong-time transport swap for dynamic feature foam
4. freeze appearance, allow transport; freeze transport, allow appearance
```

The selector must be heldout camera/time metrics, not smaller transport loss.

Metal gradient implication:

```text
needs forward replay:
    contributing cell id
    local u per sample or enough data to reconstruct it
    face/endpoint neighbor id when a segment is clipped by a face

needs backward:
    d color sample / d texels
    d local u / d p_i,r_i,R_i,height
    d face transport U_ij / d R_i,R_j,p_i,p_j,r_i,r_j if trained as loss
    d temporal u_hat / d p_i(t),p_i(t+dt),R_i(t),R_i(t+dt),r_i(t),r_i(t+dt)
```

First loss variant should stopgrad `U_ij` and optimize only `z_i`; second can
open gradients into frame/center/radius after finite-difference parity exists.

### 3. Camera-Ray Visibility As Connection And Holonomy

Hypothesis: a camera ray does not just integrate density; it transports
visibility state through a sequence of cell charts. Bad heldout views may be
high-holonomy paths through the cell complex: source rays traverse a coherent
order, while heldout rays traverse inconsistent chart transitions.

For a ray, the replay gives a path:

```text
gamma(ell) = [(i_0, s_0,s_1), (i_1, s_1,s_2), ...]
```

Each transition `i_a -> i_{a+1}` carries a visibility transport:

```text
V_ij(ell) =
    [ transmittance update ] x [ chart transport U_ij ] x [ material phase ]

log visibility increment:
    A_ij(ell) =
        - integral_segment sigma_i ds       # opacity connection
        + log_SO2(U_ij)                     # tangent chart connection
        + phase/material transport residual # optional
```

A closed camera-ray loop can be built without a literal physical loop by using
two cameras and two neighboring rays that hit the same cell cycle:

```text
loop C:
    source ray path segment i->j->k
    heldout ray path segment k->j->i

holonomy:
    H_C = product_edges exp(A_edge)
    curvature_score = || log(H_C) ||^2
```

Practical version: use graph cycles, weighted by ray co-visibility:

```text
co_visible_ij =
    count of train+heldout rays that contribute to both i and j

cycle_weight_C =
    min_edges co_visible_edge * min_edges face_witness_edge

ray_holonomy_C =
    wrap_pi(sum_edges theta_ij) ^ 2
  + robust(sum_edges tau_edge_disagreement)
```

Visibility curvature should be reported separately for source-only rays and
heldout rays:

```text
source_path_curvature_p90
heldout_path_curvature_p90
curvature_gap = heldout_p90 - source_p90
```

Supports the direction if `curvature_gap` rises before heldout PSNR collapses
or localizes bad heldout pixels.

Metal gradient implication:

```text
diagnostic:
    replay cell sequence, segment t_near/t_far, alpha/contribution.

trainable:
    d tau_edge / d density,height,SV,center,radius,frame
    d theta_ij / d frame and face geometry
    stable small-cycle enumeration on current Cech/AABB graph
```

Do not differentiate through path topology initially. If topology changes,
refresh cycles between optimizer steps and treat them as sampled structure.

### 4. Topology-Respecting Capacity Growth

Hypothesis: adding cells where heldout residual is high can waste capacity if
the residual lies along an unwitnessed gauge direction. Growth should require
residual, witness, and curvature.

Score:

```text
K_i =
    heldout_residual_i
  * clamp(cell_witness_i / target_witness, 0, 1)
  * (1 + holonomy_i)
  * contribution_i
```

Split direction:

```text
W_i eigenvectors:
    e_max = most transverse witnessed direction
    e_min = least witnessed depth direction

if witness_i high:
    split along projected residual gradient or e_max
else:
    do not split along e_min; report support insufficiency
```

Cech/AABB update:

```text
after split:
    rebuild local AABB for parent and children
    update only edges within expanded radius box
    preserve old topology elsewhere
```

Test:

```text
A. existing contribution/error growth
B. residual*witness growth
C. residual*witness*holonomy growth
```

Pass only if heldout metrics improve at equal cell count and equal render
budget. Source PSNR-only improvement is a failure.

## Gradient Checklist For Metal Training

Before any gauge math becomes a trainer loss, the Metal path needs explicit
gradient coverage for the variables the loss touches.

Already adjacent to current local lanes:

```text
center p_i:
    affects power faces, segment endpoints, local texel coordinate, ray length

radius r_i:
    affects sphere bound, power face offset, local coordinate scale, density scale

frame/quaternion R_i:
    affects texel local coordinate, tangent/normal, SV/view basis, transport U_ij

height h_i(u):
    affects surface point, normal proxy, sample coordinate, opacity/color query

SV/material z_i:
    affects view-dependent color and feature decode
```

New gradients needed by gauge losses:

```text
face endpoint:
    d s_face / d p_i,p_j,r_i,r_j

face tangent transport:
    d polar_SO2(T_j^T T_i) / d R_i,R_j,p_i,p_j,r_i,r_j

cycle holonomy:
    d wrap_pi(sum theta_edges) / d theta_edges

texel pullback:
    d sample(z_i,u_i(x)) / d z_i,u_i,p_i,r_i,R_i,height

visibility path:
    d tau_segment / d segment endpoints and density parameters
```

Risk controls:

```text
1. implement all as no-grad diagnostics first
2. finite-difference tiny fixtures before trainer use
3. stopgrad topology, stopgrad witness gates, stopgrad cycle selection
4. open gradients in order: material -> frame -> center/radius -> topology proxy
5. keep heldout metrics as selector; never accept a loss because it lowers itself
```

## Speculative Math

These are plausible but not near-term unless diagnostics above show signal.

### A. Cech Nerve As The Actual World Token Skeleton

The Cech/AABB graph can be treated as a nerve approximation to a covered
surface/volume:

```text
cover U_i = support ball/cell around p_i
nerve simplex [i0,...,ik] exists if intersection(U_i0,...,U_ik) nonempty
```

Instead of only pairwise neighbors, use low-order simplices:

```text
0-simplex: cell support
1-simplex: shared face / overlap
2-simplex: triple overlap / local sheet patch
```

Topological witness:

```text
witness(simplex S) =
    eigmin(sum rays hitting intersection(S) of (I-dd^T)) / trace(...)
```

Potential payoff: holes, disconnected supports, and false sheet topology become
measurable before rendering fails. Main blocker: robust simplex enumeration at
4K scale without destroying the fast selected traversal path.

### B. Discrete Bundle Over Camera Space

Camera space itself can be the base manifold. Each camera ray samples a fiber:

```text
base point q = camera pose + pixel ray
fiber F_q = ordered visible cell/material states along ray
connection = how F_q changes under small camera motion
```

Novel-view failure is high curvature of this camera-space bundle:

```text
parallel transport source ray fiber to heldout ray
compare with actual heldout ray fiber
```

Implementable proxy:

```text
nearby source rays predict heldout path order poorly
=> high camera-visibility curvature
```

This is speculative because the fiber changes discontinuously at occlusion
boundaries. It may still be the right language for diagnosing view-dependent
path order rather than cell geometry.

### C. Gauge-Fixed Material Coordinates Via Harmonic Charts

Instead of arbitrary per-cell texel charts, solve local harmonic coordinates on
the witnessed cell complex:

```text
Delta_graph u = 0
boundary u fixed by high-witness anchor cells
```

Then texels live in a smoother material atlas. Gauge freedom becomes the choice
of anchors and global atlas transforms, not random per-cell tangent rotations.

Main blocker: if current geometry is poor, harmonic charts canonically encode
the wrong surface. This should wait until witness diagnostics indicate that
the cell complex is actually supported.

### D. Curvature As Birth/Death Rate For Foam

Dynamic foam needs real topology change. A source term can be tied to visibility
curvature:

```text
d mass_i / dt + div(material_flux)_i = source_i
source_i prior proportional to positive curvature/residual only where witnessed
```

This could distinguish real foam birth/death from repainting. It is speculative
because current acceptance is static/heldout quality limited; adding birth/death
math before static support is solved likely hides the failure.

## Decision Rules

Promote to code only in this order:

```text
1. no-grad witness/holonomy/transport diagnostics on existing artifacts
2. synthetic finite-difference fixtures for endpoint and transport gradients
3. one stopgrad representative-pressure loss, material-only if possible
4. open geometry gradients only after local parity and heldout signal
5. consider topology/capacity changes only after diagnostics beat source PSNR
```

Kill or pause a direction if:

```text
diagnostic does not correlate with heldout residual
loss improves source view but worsens heldout
metric mostly tracks alpha emptiness or cell count
finite-difference gradients are unstable near topology changes
implementation requires differentiating Cech/AABB selection immediately
```

The near-term value is not "add gauge regularization." The near-term value is
to find whether topological witness, covariant transport, and ray visibility
holonomy explain current heldout failure better than scalar knobs do.
