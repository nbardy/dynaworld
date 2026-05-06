# PowerFoam Wave And Fluid Math For Moving-Surface Representations

Date: 2026-05-05 23:24:04 +0700

## Scope

This is a theory note only. It does not edit the PowerFoam Metal implementation
and does not claim the current local path is a full physical fluid solver. The
goal is to preserve implementable mechanisms that could make dynamic foam or
moving-surface representations less like per-frame repainting and more like a
compact material surface with transport, conservation, and falsifiable motion
state.

Useful constraint from the current lane:

```text
PowerFoam proper on Metal needs fast/accurate forward+backward, 4K behavior,
and trainability. Any wave/fluid idea should first be a cheap diagnostic,
small synthetic ablation, or isolated loss toggle before becoming a renderer or
trainer commitment.
```

Working belief:

```text
Do not bolt a full water solver onto Dynaworld.
Do expose material coordinates, conserved quantities, transport residuals, and
wave priors that can tell "moved surface" apart from "deleted/repainted cells."
```

## Representation Target

A PowerFoam-like dynamic element should be able to carry both render state and
material state:

```text
cell i at time t:
    x_i(t)       world anchor in R^3
    E_i(t)       tangent frame [e1_i, e2_i] in R^{3x2}
    n_i(t)       normal
    A_i(t)       represented surface area or area proxy
    rho_i(t)     material density / opacity per area
    m_i(t)       material amount = rho_i A_i
    h_i(t)       optional height/free-surface displacement
    v_i^T(t)     tangential/material velocity in local E_i coordinates
    w_i(t)       normal velocity
    k_i(t)       local wave vector in tangent coordinates
    phi_i(t)     phase or velocity-potential-like scalar
    omega_i(t)   vorticity/curl residual
    S_i(t)       local SV/support axes or covariance frame
    f_i(t)       material features/color payload
```

Not all fields should ship. The point is to split the state into:

- Render payload: `x, E, n, S, rho, f`, used by Metal/Torch rasterization.
- Transport payload: `v^T, w`, used to advect material coordinates and features.
- Wave payload: `h, k, phi`, used for phase-coherent normal/tangent motion.
- Defect payload: `omega`, used only where potential/wave priors fail.

The model should pay for every extra field through held-out camera/time gains,
not source-view PSNR.

## Mechanism 1: Material Coordinates And Identity Persistence

### Hypothesis

The current failure mode to fight is not just bad geometry; it is a latent
surface that changes identity every frame. Material coordinates give the model a
stable label space:

```text
u_i(t)          material coordinate, usually 2D
X(u_i, t)       world embedding
F(u_i, t)       feature payload attached to material label
rho(u_i, t)     density attached to material label
```

If a surface moves, `u` should be mostly persistent and `X(u,t)` changes. If
the model repaints, rendered RGB can remain good while `F(u,t)` and `rho(u,t)`
lose advection consistency.

### Implementable Version

Keep cell centers as existing render anchors, but add an optional material id
state:

```text
u_i(t+dt) = u_i(t)                      # Lagrangian material labels
x_i(t+dt) = x_i(t) + dt * v_i(t)
f_i(t+dt) = f_i(t) + explicit_source_i
rho_i(t+dt) = rho_i(t) + explicit_source_i
```

or, for Eulerian texture-site fields:

```text
u_i(t+dt) = u_i(t) - dt * v^T_i(t)      # backtrace in chart space
f_i(t+dt) approx sample(f(t), u_i - dt v^T_i)
rho_i(t+dt) approx sample(rho(t), u_i - dt v^T_i)
```

This can start entirely outside the renderer as a diagnostic over trained
per-frame states:

```text
feature_advection_error_i =
    || f_i(t+dt) - bary_sample_N(i,t)(f(t), x_i(t+dt) - dt v_i) ||

density_advection_error_i =
    | rho_i(t+dt) - bary_sample_N(i,t)(rho(t), x_i(t+dt) - dt v_i) |
```

### Renderer Coupling

No Metal kernel change is needed for the first pass. The renderer consumes
`x,E,S,rho,f` as usual. Training adds an auxiliary residual on the state that
produced those render inputs:

```text
L = L_rgb_or_feature_render
  + lambda_adv_f   * robust(feature_advection_error)
  + lambda_adv_rho * robust(density_advection_error)
  + lambda_source  * explicit_source_rate
```

Do not enforce this through invisible cells at first. Weight by a detached
visibility/coverage estimate so occlusion does not look like failed physics:

```text
weight_i = stopgrad(clamp(visible_i * track_confidence_i, 0, 1))
```

### Diagnostic

Report:

```text
advection_feature_l1_p50/p90/p99
advection_density_l1_p50/p90/p99
source_rate_per_visible_area
identity_persistence = corr(f_i(t), f_match(i,t+dt))
heldout_error_vs_advection_error_corr
```

Supports the mechanism if source-view PSNR stays similar but held-out camera or
held-out time error correlates with high advection/source residual.

## Mechanism 2: Surface Area Preservation And Incompressibility

### Continuous Surface Law

For a moving material surface:

```text
X(u,t) in R^3
e_a = d_a X
g_ab = e_a dot e_b
dA = sqrt(det g) du1 du2
d_t X = v^a e_a + w n
```

Surface area evolves as:

```text
d_t log sqrt(det g) = div_T v - 2 H w
```

where `H` is mean curvature. Material density obeys:

```text
d_t(rho sqrt(det g)) + d_a(rho v^a sqrt(det g)) = s_rho sqrt(det g)
```

Interpretation:

```text
normal motion over curved geometry changes area;
tangential divergence changes area in the chart;
rho changes should be explained by transport plus explicit birth/death source.
```

### Discrete PowerFoam Version

Use the existing adjacency, Cech/AABB graph, or a diagnostic kNN graph:

```text
m_i = rho_i A_i

flux_ij =
    l_ij * rho_ij * dot(v_ij^T, nu_ij)

R_mass_i =
    (m_i(t+dt) - m_i(t)) / dt
  + sum_{j in N(i)} flux_ij
  - A_i s_i
```

For nearly incompressible surface flow:

```text
R_area_i =
    (A_i(t+dt) - A_i(t)) / dt
  + A_i * div_T(v_i)
  - 2 A_i H_i w_i

L_incomp_surface = robust(R_area_i / (A_i/dt + eps))
```

For visible foam/spray, strict conservation is wrong. The better constraint is:

```text
most density changes must be either advective flux or paid source term.
```

### Renderer Coupling

Use `rho_i` as density per represented area, not arbitrary alpha, then derive
opacity consistently:

```text
alpha_i = 1 - exp(-sigma_i * rho_i * path_length_i)
```

For feature splatting:

```text
feature_mass_i = rho_i A_i f_i
```

Then density transport constrains both opacity and feature payload without
requiring RGB to be constant under illumination/view changes.

### Diagnostic

Report:

```text
mass_residual_visible_p90
area_residual_visible_p90
net_mass_drift_per_frame
mass_source_budget = sum |s_i| A_i dt
mass_residual_occluded_vs_visible
heldout_psnr_delta_vs_mass_residual
```

If mass residual only spikes at occlusion boundaries, do not turn it into a
global loss. If it spikes in stable visible regions with good source PSNR, it is
evidence of repainting.

## Mechanism 3: Kelvin/Helmholtz-Style Circulation Constraints

### Hypothesis

A moving surface can be material without being curl-free. Waves and sheets can
carry circulation. A useful weak invariant is circulation around a material
loop:

```text
Gamma_C(t) = integral_C v . dl
```

Kelvin-style idea:

```text
In inviscid/barotropic/no-forcing regions, Gamma_C should be approximately
constant along material loops.
```

Helmholtz-style idea:

```text
vorticity should move with the material rather than appear everywhere
independently every frame.
```

This is not a claim that all videos satisfy ideal fluid assumptions. It is a
way to define a sparse "defect" budget: where circulation is not conserved, the
model should mark forcing/breaking/spray instead of hiding it in arbitrary
framewise motion.

### Discrete Loop Constraint

On small graph loops `C = i0 -> i1 -> ... -> i0`:

```text
Gamma_C =
    sum_edges dot(v_ij, x_j - x_i)

R_gamma_C =
    (Gamma_C(t+dt) - Gamma_C(t)) / dt - forcing_C
```

Vorticity payload version:

```text
omega_i = curl_T v_i
R_omega_i =
    (omega_i(t+dt) - advect(omega(t), x_i, v_i)) / dt
  - stretch_i
  - source_omega_i
```

### Renderer Coupling

This should not directly affect rasterization. It regularizes motion state that
then updates render anchors/support axes. Suggested update split:

```text
v_i = grad_T phi_i + curl_T psi_i + v_defect_i
```

or simpler:

```text
v_i = v_potential_i + v_residual_i
L_residual_rate = ||v_residual|| weighted by visibility and heldout benefit
```

The renderer sees only the resulting `x,E,S,rho,f`. The diagnostic asks whether
cells that need large circulation/defect sources are exactly the regions where
held-out prediction fails or breaking/spray appears.

### Diagnostic

Report:

```text
loop_circulation_residual_p90
curl_residual_p90
v_residual_energy / v_total_energy
defect_budget_spatial_sparsity
defect_budget_vs_render_error_corr
```

Supports the mechanism if coherent waves/sheets need little defect budget while
rolling/breaking regions need localized defects.

## Mechanism 4: Shallow-Water And Wave-Equation Priors

### Low-Cost Prior

For surfaces that can be locally parameterized as a base surface plus height:

```text
X(u,t) = B(u) + h(u,t) n_B(u)
```

Use a wave equation or shallow-water residual as a prior:

```text
wave:          d_tt h - c(u)^2 Laplace_T h = source - damping * d_t h

shallow mass:  d_t h + div_T(h v^T) = source_h

shallow mom:   d_t v^T + (v^T . grad_T) v^T + g grad_T(h + b) = forcing - drag
```

The wave-equation prior is the lowest-risk version because it needs only
heights and time derivatives. Shallow-water needs a more meaningful depth or
thickness channel.

### Implementable Fields

Per cell:

```text
h_i(t)        height or normal displacement
dh_i(t)       velocity along normal, can be w_i
c_i(t)        learned local wave speed, positive and bounded
damp_i(t)     learned damping, positive and bounded
source_h_i    sparse source for impact/breaking/occlusion
```

Residual:

```text
R_wave_i =
    (h_i(t+dt) - 2 h_i(t) + h_i(t-dt)) / dt^2
  - c_i^2 * Laplace_graph(h)_i
  + damp_i * (h_i(t+dt) - h_i(t-dt)) / (2 dt)
  - source_h_i
```

Bounds:

```text
c_i dt / edge_length_i <= CFL_max
damp_i >= 0
source_h sparse unless render evidence demands it
```

### Renderer Coupling

Use the wave state to update only the geometric part:

```text
x_i(t) = base_x_i(t) + h_i(t) n_i(t)
normal_i from graph gradient of h or learned E_i update
support axis S_i normal scale can grow with |grad h| or breaking source
```

Training:

```text
L = L_render
  + lambda_wave * robust(R_wave)
  + lambda_cfl  * relu(c dt / edge_len - CFL_max)^2
  + lambda_src  * |source_h|
```

Do not let `h` become a hidden color channel. It should influence geometry,
normal, opacity path length, or support axes.

### Diagnostic

Report:

```text
wave_residual_p90
cfl_violation_count
learned_c_quantiles
source_h_spatial_sparsity
phase_error_on_sinusoid
amplitude_decay_or_growth
```

Good sign: lower held-out temporal phase error without source-view-only gains.
Bad sign: learned `c` saturates, CFL violations spike, or `source_h` explains
all motion.

## Mechanism 5: Differentiable Advection Of Texel Sites, Heights, And SV Axes

### Texel-Site Advection

Instead of letting features be arbitrary per frame:

```text
u_texel(t+dt) = u_texel(t) + dt * v^T(u_texel,t)
f_texel(t+dt) = f_texel(t) + source_f
```

For training stability, use semi-Lagrangian backtracing:

```text
f_hat(u,t+dt) = sample(f(t), u - dt v^T(u,t))
L_tex_adv = || f(t+dt) - f_hat ||
```

This is differentiable if `sample` is bilinear over local chart neighborhoods or
soft kNN over cells.

### Height Advection

Height should be transported by tangential flow and changed by normal dynamics:

```text
d_t h + v^T . grad_T h = w - h_source_or_damping
```

Residual:

```text
R_h_adv =
    (h_i(t+dt) - h_i(t)) / dt
  + dot(v_i^T, grad_T h_i)
  - w_i
  - source_h_i
```

This catches the case where the surface phase jumps without material or wave
motion explaining it.

### SV/Support-Axis Advection

For PowerFoam support axes or covariance/SV frames:

```text
S_i(t)      support shape in local tangent/normal frame
E_i(t)      tangent frame
F_i = grad_T v^T      local deformation gradient
```

A material support should evolve approximately by the local flow map:

```text
S_i(t+dt) approx (I + dt F_i) S_i(t) (I + dt F_i)^T + Q_i
```

where `Q_i` is explicit diffusion/splitting/noise. For an incompressible
surface patch:

```text
det_T(S_i) or represented area should not arbitrarily collapse/expand.
```

Loss:

```text
L_sv_adv =
    robust(log_eigs(S_i(t+dt)) - log_eigs(S_adv_i))
  + robust(frame_transport_error(E_i(t+dt), E_adv_i))
```

This is directly relevant to dynamic foam because support axes can otherwise
rotate/scale each frame to fit screen-space artifacts.

### Renderer Coupling

The renderer already consumes support geometry. A differentiable advection loss
would constrain the inputs before rasterization, while render loss decides
whether the constrained motion actually helps:

```text
state(t) --advection update--> predicted_state(t+dt)
predicted_state(t+dt) --renderer--> image/features
loss = render_loss(predicted, target) + transport_residuals
```

Important split:

```text
teacher-forced diagnostic:
    compare actual optimized state(t+dt) to advected state(t)

rollout loss:
    render advected/predicted state(t+dt) directly and backprop through renderer
```

Start with teacher-forced diagnostics. Rollout losses are more informative but
can destabilize early PowerFoam training.

## Mechanism 6: Gauge-Clean Loss Design

Fluid variables have gauge freedoms. Losses should not pin arbitrary choices:

```text
material labels:       invariant to cell permutation
potential phi:         invariant to phi -> phi + constant
pressure p:            invariant to p -> p + constant
phase:                 invariant to phase -> phase + 2 pi
tangent frame angle:   invariant to E_i -> E_i R(theta_i) when render state is equivalent
camera/world gauge:    do not force a canonical world origin unless sampler supports it
```

Safer residuals:

```text
grad phi, not phi
grad p, not p
sin/cos phase or k = grad phase, not raw phase
loop circulation, not absolute vector-potential gauge
mass flux over graph cuts, not cell ids alone
rendered held-out error, not source-view fit
```

Gauge bug diagnostic:

```text
Apply an equivalent transformation to material ids, phase offsets, or tangent
frames. The diagnostic/loss should be unchanged up to numerical tolerance.
```

If this fails, the loss may improve training by imposing a convention rather
than by adding real predictive structure.

## Ablation Matrix

Run only after the current PowerFoam Metal forward/backward path is stable
enough for repeatable tiny schedules.

```text
A0 repaint baseline:
    dynamic cells with no transport/wave losses

A1 material advection diagnostic only:
    log feature/rho advection residuals, no loss

A2 material advection loss:
    L_adv_f + L_adv_rho + source budget

A3 surface mass/area:
    A2 + finite-volume mass residual + area/incompressibility residual

A4 wave height prior:
    A2 + h,w,c,damping and wave-equation residual

A5 shallow-water prior:
    A4 + h v^T mass/momentum residuals

A6 circulation/defect budget:
    A2/A4 + loop circulation or curl residual + sparse defect source

A7 support-axis advection:
    A2 + SV/covariance/frame transport residual

A8 full coupled candidate:
    best of A2-A7, same renderer/kernel path, no new source-view-only knobs
```

Required controls:

```text
same dataset split
same frame count
same element count
same renderer variant
same train steps
same random seeds where practical
same evaluation cameras/times
held-out-camera selection, not source PSNR selection
```

## Synthetic Gates Before Real Runs

### Gate 1: Translating Material Sheet

```text
X(u,t) = X(u,0) + c t
rho(u,t) constant
f(u,t) constant in material coordinates
```

Expected:

```text
low advection residual
low mass residual
near-zero source budget
```

Failure means the diagnostic graph/interpolator is wrong.

### Gate 2: Area-Preserving Stretch

```text
X(u,t) = [a(t) u1, u2 / a(t), 0]
det grad X = 1
```

Expected:

```text
area/mass residual stays low if flow gradient is estimated correctly
SV axes rotate/scale consistently with deformation
```

Failure means support-axis advection or finite-volume flux is not measuring the
right invariant.

### Gate 3: Compressible Negative Control

```text
X(u,t) = [a(t) u1, a(t) u2, 0]
rho either changes as 1/a^2 or violates conservation
```

Expected:

```text
conservative sequence passes if rho compensates
repaint/compress cheat fails mass residual
```

### Gate 4: Sinusoidal Wave

```text
h(x,t) = A sin(k x - omega t)
omega^2 approx g |k| tanh(|k| depth)
```

Expected:

```text
wave residual low at correct c
phase error exposes wrong learned wave speed
source_h not needed except at boundaries
```

### Gate 5: Rolling Vortex Sheet

```text
v = grad phi + curl psi, with nonzero localized curl
```

Expected:

```text
potential-only model needs high residual or high render error
defect/curl model localizes omega budget
```

### Gate 6: Repaint Cheat

Render-identical frames are generated by shuffling material payloads each
frame.

Expected:

```text
source PSNR can be high
identity/advection residual high
held-out temporal rollout degrades
```

This is the most important negative control because it matches the bad shortcut
the representation is trying to eliminate.

## Real-Data Diagnostics

For each train/eval run that uses these ideas, log a compact table:

```text
render:
    train PSNR/SSIM/L1
    heldout-camera PSNR/SSIM/L1
    heldout-time PSNR/SSIM/L1 where available

motion:
    median/p90/p99 |dx| per frame
    normal vs tangential velocity energy
    displacement / projected_radius

transport:
    feature_advection_l1 p50/p90/p99
    density_advection_l1 p50/p90/p99
    source budget per visible area

conservation:
    mass residual p50/p90/p99
    area residual p50/p90/p99
    net mass drift per frame

wave:
    wave residual p90
    learned c quantiles
    CFL violation rate
    phase error on synthetic or identifiable periodic regions

curl/defect:
    circulation residual p90
    curl residual p90
    defect energy / total motion energy
    defect sparsity

support:
    SV axis advection residual
    log-det support drift
    tangent-frame transport residual
```

Selection rule:

```text
Promote a mechanism only if it improves held-out camera/time or reduces a known
repaint diagnostic at matched render quality. Do not promote a mechanism merely
because it improves source-view PSNR or makes motion fields look smoother.
```

## Coupling To Current PowerFoam Work

Shortest path:

1. Keep the Metal renderer/backward path focused on correct, fast, trainable
   bounded-cell rasterization.
2. Add diagnostics in Python around the dynamic state before touching kernels.
3. Use existing saved tensors or trainer state to compute material/advection
   residuals after a run.
4. If diagnostics correlate with held-out failures, add one auxiliary loss
   toggle at a time.
5. Only after a loss proves useful, consider whether any state needs to become
   first-class in the renderer or export path.

Likely no-kernel first pass:

```text
state tensors:
    positions over time
    features/colors over time
    opacity/density over time
    support axes/covariances over time

derived tensors:
    kNN/adjacency graph
    velocity from finite differences
    graph gradients/divergence/laplacian
    visibility weights from alpha/coverage
```

Potential kernel-facing later pass:

```text
density-per-area alpha model
support-axis evolution state
material-feature payload for feature splatting
normal/height perturbation path
```

Do not let wave/fluid code block the core PowerFoam acceptance path. Treat it as
an overlay that either proves held-out value or gets discarded.

## Risks And Backtracks

- Visibility can masquerade as mass violation. Occlusion-aware weighting is
  mandatory before interpreting conservation residuals.
- Real foam has birth/death. Strict conservation may punish the right behavior;
  use explicit source budgets rather than zero-source assumptions.
- Tangential motion is partly gauge: relabeling material coordinates can look
  like tangential flow. Prefer rendered rollout and invariant flux/loop
  diagnostics over absolute label losses.
- Wave priors may overfit water-like data and hurt generic dynamic scenes. Keep
  them config-gated and measure against non-water moving surfaces.
- Learned wave speed can become a hidden cheat knob. Bound CFL and inspect
  quantiles.
- Curl/defect channels can absorb everything. Penalize defect rate and require
  spatial sparsity or held-out improvement.
- Support-axis advection can be numerically stiff. Start as a teacher-forced
  residual before doing multi-step rollout.

## First Concrete Experiment To Queue

Most useful first item:

```text
teacher_forced_material_transport_probe
```

Inputs:

```text
saved dynamic state over F frames from any stable small PowerFoam/dynamic-foam run
positions, opacity/density, features/colors, support axes if available
camera visibility/alpha summary
```

Compute:

```text
finite-difference velocity
kNN graph in world or material space
semi-Lagrangian feature/rho advection error
mass residual over graph cuts
support log-det/SV-axis drift
correlation with held-out render residual
```

No training change, no kernel change, no baseline claim. This tells whether the
current dynamic representation is actually carrying persistent material or just
winning by per-frame repainting. If the probe has no correlation with held-out
failure, pause the fluid-prior lane and return to renderer/backward/4K work.
