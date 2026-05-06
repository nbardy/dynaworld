# Moving-Surface Gauge And Fluid Math Directions For Dynamic PowerFoam

Date: 2026-05-05 23:17:48 +0700

## Scope

This is a new research note only. It does not edit code, rerun baselines, or
claim that local PowerFoam is a full physical fluid solver. The purpose is to
propose falsifiable mathematical directions for dynamic PowerFoam and gauge
field representations, focused on waves, fluid-like moving surfaces, and gauge
constraints.

Local constraints used here:

- Select representation ideas by held-out camera/time behavior, not source-view
  fit.
- Treat the current PowerFoam Metal path as a partial trainable bounded-cell
  core, not proof of full official PowerFoam or physical motion.
- Do not add physics losses that pin arbitrary latent gauges; losses should be
  invariant to cell permutation, tangent-frame rotation, pressure/potential
  offsets, phase wraps, and benign world reparameterizations.
- Start with diagnostic helpers and tiny synthetic experiments before a trainer
  fork.

## Current Belief

The useful object is not "a water simulator inside PowerFoam." The useful
object is a rendered, compact, moving surface measure whose internal coordinates
are gauge variables:

```text
material surface:     U, with local coordinates u = (u1, u2)
world embedding:      X(u, t) in R^3
surface tangent map:  e_a = d_a X,  a in {1,2}
metric:               g_ab = e_a dot e_b
unit normal:          n = normalize(e_1 x e_2)
second fundamental:   b_ab = -e_a dot d_b n
density/opacity:      rho(u,t)
appearance/features:  f(u,t)
velocity split:       d_t X = v^a e_a + w n
```

The split into tangential velocity `v^a` and normal velocity `w` is gauge-aware:
tangential motion partly changes material labels, while normal motion changes
the visible surface. A screen-space repainting solution can fit frames while
violating the induced metric, area, mass, and phase transport constraints.

## Direction 1: Intrinsic Moving-Surface Conservation

### Hypothesis

Dynamic foam cells should preserve material surface measure under the
surface's own geometry, not under an arbitrary 3D grid. The conserved object is
not raw opacity per cell; it is opacity times surface area.

Continuous surface area element:

```text
dA = sqrt(det g) du1 du2
```

Kinematic metric evolution under `d_t X = v^a e_a + w n`:

```text
d_t g_ab = nabla_a v_b + nabla_b v_a - 2 w b_ab
```

Area evolution:

```text
d_t log sqrt(det g) = nabla_a v^a - 2 H w
H = 0.5 g^ab b_ab
```

Conservative mass law:

```text
d_t(rho sqrt(det g)) + d_a(rho v^a sqrt(det g)) = s_rho sqrt(det g)
```

Equivalent local residual:

```text
R_mass =
    d_t rho + v^a nabla_a rho
  + rho (nabla_a v^a - 2 H w)
  - s_rho
```

For non-birth/death material, `s_rho = 0`. For foam/spray, `s_rho` should be a
small explicit source term with a rate penalty, not hidden in arbitrary
opacity changes.

### Discrete PowerFoam State

Per cell:

```text
x_i(t)       center / surface anchor in R^3
E_i(t)       tangent frame [t_i, b_i] in R^{3x2}
n_i(t)       normal
A_i(t)       represented surface area or area proxy
rho_i(t)     opacity density per area
m_i(t)       material mass = rho_i A_i
v_i^T(t)     tangent velocity in E_i coordinates
w_i(t)       normal velocity
f_i(t)       material appearance or feature payload
```

Discrete residuals on a neighbor graph `N(i)`:

```text
div_T(rho v)_i =
    (1 / A_i) sum_{j in N(i)} l_ij rho_ij (v_ij dot nu_ij)

R_mass_i =
    (m_i(t+dt) - m_i(t)) / dt
  + sum_{j in N(i)} l_ij rho_ij (v_ij dot nu_ij)
  - A_i s_i
```

Here `l_ij` is a shared edge/face length proxy and `nu_ij` is the outward
tangent co-normal in cell `i`'s surface tangent plane. The exact graph can be
PowerFoam adjacency, Cech/AABB, or a diagnostic kNN graph; the first goal is
not perfect finite volume accuracy, it is to detect delete/recreate motion.

### Stability Losses

Diagnostic first:

```text
L_mass_diag = mean_i robust(R_mass_i / (|m_i|/dt + eps))
```

Candidate loss only if diagnostic predicts held-out failure:

```text
L_mass = lambda_mass mean_i stopgrad(visible_i) robust(R_mass_i)
```

Do not apply this to all opacity. Use it on a material-density channel and
leave an explicit `s_i` source for foam birth/death:

```text
L_source_rate = beta_s sum_i |s_i| A_i dt
```

### Tiny Experiment

Name: `moving_surface_mass_negative_control`

Generate three 2D material sheets rendered from two cameras:

```text
A. translate:        X(u,t) = X(u,0) + c t
B. stretch:          X(u,t) = diag(1+a t, 1/(1+a t), 1) X(u,0)
C. repaint cheat:    each frame matches RGB but randomly reassigns rho_i
```

Fit equal-rate cells with and without `L_mass_diag` / `L_mass`. Measure:

```text
source PSNR
held-out camera PSNR
mass residual quantiles
cell identity persistence: corr(f_i(t), f_i(t+1) after advection)
```

Supports the direction if the repaint cheat has similar source PSNR but worse
held-out camera/time and much higher mass residual.

Weakens it if mass residual is dominated by visibility/occlusion rather than
real material instability.

## Direction 2: Gauge-Covariant Tangent Frames And Holonomy

### Hypothesis

Moving surfaces need tangent frames, but tangent-frame angle is a gauge. A
regularizer on absolute `R_i` or quaternion values can punish equivalent worlds.
The invariant object is the connection between neighboring tangent frames and
its holonomy around loops.

For each adjacent pair `(i,j)`, define tangent bases:

```text
E_i = [t_i, b_i]
E_j = [t_j, b_j]
```

Project the center-to-center direction onto each tangent plane:

```text
a_i = normalize(E_i^T (x_j - x_i))
a_j = normalize(E_j^T (x_i - x_j))
```

Let `Q_i(a_i)` be the 2D rotation that maps local x-axis to `a_i`. A simple
edge transport from frame `i` to frame `j` is:

```text
U_ij = Q_j(a_j)^T E_j^T E_i Q_i(a_i)     in O(2)
```

For orientable coherent sheets, use the nearest `SO(2)` factor:

```text
U_ij^+ = polar_SO2(U_ij)
theta_ij = atan2(U_ij^+[2,1], U_ij^+[1,1])
```

Loop holonomy:

```text
Theta_C = wrap_pi(sum_{(i,j) in C} theta_ij)
L_holonomy_diag = mean_C weight_C Theta_C^2
```

This should be invariant under local tangent-frame gauge rotations:

```text
E_i -> E_i R(alpha_i)
theta_ij -> theta_ij + alpha_i - alpha_j
sum loop theta_ij unchanged mod 2 pi
```

### Stability Losses

Start as a diagnostic:

```text
holonomy_residual_p90
holonomy_vs_heldout_residual_corr
holonomy_vs_source_residual_corr
```

If it predicts held-out errors, use a visibility-weighted representative
pressure:

```text
L_conn =
    lambda_conn sum_{(i,j)} w_ij robust(wrap_pi(theta_ij - theta_ij_ref))
```

or loop-only:

```text
L_holo =
    lambda_holo sum_{C in small_cycles} w_C robust(Theta_C)
```

The loop version is more gauge-clean but harder to compute. The edge version
requires a reference transport convention and is easier to over-pin.

### Tiny Experiment

Name: `rotating_tangent_frame_gauge_invariance`

Use a fixed rendered sheet and randomly rotate each cell's tangent basis:

```text
E_i' = E_i R(alpha_i)
```

Expected:

```text
render unchanged
absolute-frame loss changes       -> bad loss
loop holonomy unchanged           -> acceptable diagnostic
edge angle changes by gauge term  -> only acceptable with fixed convention
```

A second test should make a twisted ribbon with true curvature. Supports the
direction if loop holonomy separates the physical twist from random per-cell
frame noise and correlates with held-out view failure when the frame field is
scrambled.

## Direction 3: Surface Wave State As Phase, Momentum, And Curvature

### Hypothesis

Coherent waves are phase systems. A position-only dynamic cell state invites
repainting or frame interpolation. A better compact state for wave-like
surfaces is:

```text
eta_i(t)      normal displacement from a slowly moving base surface
pi_i(t)       conjugate momentum or surface potential
k_i(t)        local phase gradient / wave vector in tangent coordinates
A_i(t)        local amplitude
theta_i(t)    phase stored as (cos theta, sin theta)
```

Linear deep-water local dispersion:

```text
omega_i^2 = g |k_i|
```

Finite-depth shallow-water dispersion:

```text
omega_i^2 = g |k_i| tanh(|k_i| h_i)
```

Surface signal:

```text
eta_i(t) = A_i cos(theta_i(t))
d_t theta_i + omega_i = 0
d_t A_i + gamma_i A_i = source_i
```

Hamiltonian local form:

```text
H_i = 0.5 pi_i^2 + 0.5 omega_i^2 eta_i^2
d_t eta_i =  dH_i/dpi_i = pi_i
d_t pi_i  = -dH_i/deta_i - gamma_i pi_i + forcing_i
```

### Losses

Phase residual:

```text
R_phase_i =
    wrap_pi(theta_i(t+dt) - theta_i(t) + omega_i dt)
```

Energy residual with damping/source:

```text
R_E_i =
    (H_i(t+dt) - H_i(t)) / dt
  + gamma_i pi_i(t)^2
  - W_i_external
```

Gauge checks:

```text
theta_i -> theta_i + 2 pi n          leaves loss unchanged
potential psi -> psi + const         leaves velocity/loss unchanged
permutation of cells                 leaves aggregate loss unchanged
global rigid transform               leaves intrinsic loss unchanged
```

Use the wave residual only where a local wave confidence gate is high:

```text
L_wave =
    lambda_wave sum_i q_i [
        robust(R_phase_i)
      + alpha_E robust(R_E_i / (H_i/dt + eps))
    ]
  + beta_q sum_i q_i                 # rate cost for declaring wave mode
```

The `q_i` gate prevents potential-flow math from being forced onto breaking,
contact, or specular regions.

### Tiny Experiment

Name: `phase_state_vs_position_state`

Generate:

```text
traveling sine sheet:
    eta(x,t) = A sin(k x - omega t)

standing wave:
    eta(x,t) = A cos(k x) cos(omega t)

wrong-dispersion negative control:
    eta(x,t) = A sin(k x - 1.7 omega t)
```

Compare equal-rate payloads:

```text
A. x, feature
B. x, velocity, feature
C. x, eta, pi, feature
D. x, eta, phase(cos,sin), k, sparse residual, feature
```

Train on frames `0,2,4`, test on `1,3,5` and one held-out camera. Supports the
direction if phase/momentum state improves held-out time at equal source loss
and rejects the wrong-dispersion control through high `R_phase`.

## Direction 4: Potential-Plus-Vorticity Split For Breaking Surfaces

### Hypothesis

Before breaking, wave motion is often low-curl/potential-like on the surface.
Breaking, rolling lips, spray, and foam are localized vorticity residuals. A
rate-controlled split should beat a uniform motion MLP if the dataset contains
both coherent wave sheets and localized rotational failure.

Surface velocity Hodge split:

```text
v^a = nabla^a phi + epsilon^{ab} nabla_b psi + h^a
```

where:

```text
phi        scalar potential
psi        stream function / vorticity source
h^a        harmonic component on nontrivial topology
curl v     = Delta psi
div v      = Delta phi
```

Gauge freedoms:

```text
phi -> phi + const
psi -> psi + const
```

Losses:

```text
L_potential =
    mean_i q_pot_i robust(curl_T(v)_i)

L_vort_sparse =
    beta_psi sum_i |Delta psi_i|

L_hodge_recon =
    mean_i ||v_i - grad phi_i - rotgrad psi_i - h_i||^2
```

The key is the gate:

```text
q_vort_i = sigmoid(score_i)
L_rate_vort = beta_vort sum_i q_vort_i
```

Potential mode should be cheap; vorticity residual should be paid for.

### Tiny Experiment

Name: `curl_budget_breaking_sheet`

Generate matched RGB difficulty:

```text
A. traveling sine sheet              low curl
B. translating checker cloth         low curl but high texture
C. rolling vortex sheet              high curl localized
D. random repaint sequence           high apparent motion, no transport
```

Compare:

```text
uniform velocity MLP
potential-only velocity
potential + sparse vorticity residual
```

Metrics:

```text
held-out camera/time PSNR per payload byte
curl residual on true low-curl cases
vorticity token usage on high-curl case
false-positive vorticity usage on repaint negative control
```

Supports if sparse vorticity is used only on the rolling region and improves
held-out prediction per rate. Kill if the vorticity path becomes a generic
frame cache.

## Direction 5: Ray-Invariant Surface Incidence With Dynamic Thickness

### Hypothesis

Moving water/foam surfaces are not zero-thickness splats. They need a compact
surface layer whose optical depth depends on ray incidence angle, thickness,
and transported density, while remaining invariant under rigid transforms and
surface reparameterization.

Per cell:

```text
x_i          surface anchor
n_i          unit normal
r_i^T        tangent radius
h_i          half-thickness along normal
rho_i        volume density
```

Local signed coordinates:

```text
z_i(x) = n_i dot (x - x_i)
y_i(x) = E_i^T (x - x_i)
```

Compact layer density:

```text
sigma_i(x) =
    rho_i [1 - ||y_i||^2/(r_i^T)^2]_+^p
          [1 - z_i^2/h_i^2]_+^q
```

Ray optical depth:

```text
tau_i(ell) = integral_{s0}^{s1} sigma_i(o + s d) ds
alpha_i = 1 - exp(-tau_i)
```

Thin-layer approximation:

```text
tau_i(ell) approx
    rho_i h_i C_q [1 - ||y_i(s*)||^2/(r_i^T)^2]_+^p
    / max(|d dot n_i|, eps_inc)
```

This approximation is dangerous near grazing angles, so it must be clamped or
replaced by finite-segment integration. The diagnostic is:

```text
grazing_ratio_i = tau_exact_i / tau_thin_i
```

### Losses

Thickness stability:

```text
L_thick = mean_i robust(log h_i(t+dt) - log h_i(t))
```

Mass consistency with thickness:

```text
m_i = rho_i A_i 2 h_i
L_mass_volume = robust((m_i(t+dt)-m_i(t))/dt + flux_i - source_i)
```

Incidence sanity:

```text
L_grazing_diag only:
    report p95 tau at |d dot n| < eps
```

Do not use a learned camera-angle color shortcut as the first fix. If view
dependence is needed, separate material transport from view-conditioned shading.

### Tiny Experiment

Name: `grazing_surface_layer_incidence`

Render a moving transparent/opaque sheet from cameras with varying incidence
angles. Compare:

```text
projected screen disk
thin-layer approximation
finite segment compact layer
PowerFoam hard cell raytrace
```

Supports if finite segment incidence improves held-out grazing cameras without
source-view overfitting. Weakens if the improvement is only from extra opacity
coverage and disappears under matched alpha/mass.

## Direction 6: Symmetry And Gauge Test Suite Before Training

Every proposed physics residual should pass symmetry tests before it can become
a loss:

```text
T1. cell permutation:
    permute all cells and adjacency labels consistently -> same scalar losses

T2. global SE(3):
    transform x, E, n, velocities, rays together -> same incidence/losses

T3. tangent-frame SO(2):
    E_i -> E_i R(alpha_i), local vector coordinates rotate accordingly
    -> same intrinsic conservation losses

T4. pressure/potential offset:
    phi, psi, pressure -> + const -> same velocity/residual losses

T5. phase wrap:
    theta -> theta + 2 pi n -> same phase losses

T6. material reparameterization:
    split one cell into two identical subcells with conserved total mass
    -> aggregate conservation diagnostics approximately unchanged
```

If a candidate fails these tests, it might still be useful as a deliberate
canonicalization, but it must be labeled that way. Do not let it masquerade as
a physical invariant.

## Minimal Test Ladder

1. Pure tensor diagnostics on synthetic states:
   `surface_area`, `mass_residual`, `tangent_divergence`, `curl_T`,
   `phase_residual`, `holonomy`, `CFL_surface`.

2. Symmetry tests:
   run T1-T6 above before any image rendering.

3. Synthetic render tests:
   `moving_surface_mass_negative_control`,
   `phase_state_vs_position_state`,
   `curl_budget_breaking_sheet`,
   `grazing_surface_layer_incidence`.

4. Diagnostic-only pass on existing dynamic PowerFoam artifacts:
   log mass residual, holonomy, CFL, thickness/grazing statistics, and
   correlation with held-out residual buckets.

5. One-loss ablation:
   add exactly one low-weight term, with matched exported payload/rate, and
   select by held-out camera/time.

6. Representation ablation:
   compare equal-rate payload families:

```text
P0: x, support, opacity, feature
P1: P0 + tangent/normal velocity
P2: P1 + surface mass conservation diagnostic/loss
P3: P1 + phase/momentum wave state
P4: P3 + sparse vorticity residual
P5: P1 + finite-thickness incidence
```

## Decision Criteria

A direction should advance only if it clears all three gates:

```text
1. symmetry gate:
   residual is invariant to the intended gauges

2. negative-control gate:
   residual catches repaint/delete-recreate behavior that RGB source loss misses

3. predictive gate:
   held-out camera/time improves at matched rate/capacity, or the diagnostic
   strongly predicts held-out failure before training loss is added
```

Anything that only improves source PSNR, only reduces a pretty physics metric,
or only works by adding unpriced payload capacity should stay a note, not become
the next mainline representation.

