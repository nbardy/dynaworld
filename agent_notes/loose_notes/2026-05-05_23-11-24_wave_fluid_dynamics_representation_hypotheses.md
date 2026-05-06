# Wave And Fluid Dynamics Math As Dynaworld Representation Hints

Date: 2026-05-05 23:11:24 +0700

## Context

This is a theory/process note only. It does not claim that Dynaworld or the
local PowerFoam lane already implements a physical fluid solver. The goal is to
extract falsifiable representation ideas from wave dynamics and fluid
simulation math that could later be wired into Dynaworld/PowerFoam dynamic-scene
experiments.

Relevant local framing:

- Dynaworld should select representations by held-out camera/time prediction,
  not source-view fit alone.
- PowerFoam-style cells are interesting because they can carry oriented local
  surface/material state, but a trainable raster/backward path is not the same
  as a physically constrained simulator.
- Any new dynamics idea should first land as a helper, diagnostic, or synthetic
  local test before becoming a trainer-wide architectural commitment.

Working question:

```text
Can a dynamic scene representation borrow the conserved variables,
constraints, and variational structure of wave/fluid systems without pretending
that every video is a literal Navier-Stokes solve?
```

## Current Model

The useful transfer is not "simulate water." The useful transfer is:

```text
represent state variables whose evolution has local conservation laws,
project out gauge freedoms that cameras cannot identify,
penalize impossible time evolution with cheap residuals,
and expose diagnostics that distinguish motion from per-frame repainting.
```

For dynamic PowerFoam-like cells, the lowest-risk path is to treat each cell as
a compact local material chart with optional dynamic state:

```text
cell_i(t) = {
    x_i(t)          # world position or chart anchor
    q_i(t)          # orientation frame / quaternion
    Sigma_i(t)      # support metric or shape
    a_i(t)          # opacity / density / material amount
    f_i(t)          # appearance or feature payload
    v_i(t)          # velocity
    m_i(t)          # mass-like weight or confidence
    optional: phi_i(t), p_i(t), omega_i(t), k_i(t), E_i(t)
}
```

The optional variables are not all meant to ship. They are hypothesis slots:

- `phi`: velocity potential for mostly irrotational motion.
- `p`: pressure or Lagrange multiplier enforcing incompressibility/support.
- `omega`: vorticity or rotational residual for splashes, cloth-like swirls,
  and non-potential motion.
- `k`: local wave vector or phase gradient for propagating ripples.
- `E`: local energy/action budget for stability diagnostics.

## PDE State Variables Worth Stealing

### Shallow Water Variables

State:

```text
h(x,y,t)      water/surface height or thickness
u(x,y,t)      horizontal velocity
b(x,y)        rest bed / base surface
eta = h + b   free-surface elevation
```

Equations:

```text
mass:      d_t h + div(h u) = 0
momentum:  d_t u + (u . grad) u + g grad eta = forcing - drag
```

Representation hint:

- For videos with coherent sheets, a dynamic cell field can carry
  `thickness/density + tangential velocity` instead of only per-frame position.
- A mass residual can prevent "evaporate here, repaint there" dynamics:

```text
R_mass = d_t a_i + div_cell(a_i * v_i)
```

Small local test later:

- Make a 2D Gaussian sheet with known translation and compression.
- Fit dynamic cells with and without `R_mass`.
- Failure signal: without changing held-out RGB loss much, the unconstrained
  model can delete/recreate opacity while the constrained model must advect it.

### Free-Surface Potential Flow

State:

```text
eta(x,y,t)        free-surface height
psi(x,y,t)        velocity potential evaluated at the surface
v = grad phi      irrotational 3D velocity
laplacian(phi)=0  inside the fluid domain
```

Canonical free-surface pair:

```text
d_t eta = G(eta) psi
d_t psi = -g eta - 0.5 |grad psi|^2
          + 0.5 ((G(eta) psi + grad eta . grad psi)^2)/(1 + |grad eta|^2)
```

`G(eta)` is the Dirichlet-Neumann operator. We do not need to implement it to
borrow the structure.

Representation hint:

- Dynamic surfaces often need a conjugate pair: "where the surface is" and
  "how it wants to move."
- A model that stores only positions must infer momentum from adjacent frames;
  a model that stores a potential-like scalar can make smooth extrapolation
  easier.

Small local test later:

- Synthetic sinusoidal heightfield:

```text
eta(x,t) = A sin(k x - omega t)
psi(x,t) approx (A omega / |k|) cos(k x - omega t)
omega^2 = g |k| tanh(|k| depth)
```

- Give the model frames 0 and 2, predict frame 1 and 3.
- Compare position-only dynamics against `(eta, psi)` or `(x, v)` state.
- Failure signal: phase drift, amplitude damping, or a hidden repaint route
  beats real advection on source views but loses held-out time/camera.

### Incompressible Flow Variables

State:

```text
u(x,t)             velocity
p(x,t)             pressure multiplier
rho(x,t)           density/material indicator
div u = 0          incompressibility constraint
d_t rho + u.grad rho = diffusion/source
```

Representation hint:

- Pressure is less useful as a rendered payload than as a projection mechanism:
  update unconstrained velocities, then project to a constrained field.
- In a neural representation, the equivalent can be a loss or layer that removes
  local expansion in material coordinates.

Possible loss:

```text
L_div = mean_i (trace(J_v_i))^2
```

where `J_v_i = d v / d x` in a local cell neighborhood. For surface sheets, use
only tangential divergence unless volume conservation is actually supported by
data.

Small local test later:

- Use a known divergence-free 2D vortex field:

```text
u = (-d_y psi, d_x psi)
psi = exp(-(x^2 + y^2)/sigma^2)
```

- Advect points/features for a few steps.
- Verify that a learned local velocity residual with `L_div` preserves density
  better than an unconstrained MLP at the same reconstruction loss.

### Vorticity And Curl

State:

```text
omega = curl u
d_t omega + (u.grad) omega = (omega.grad) u + viscosity * laplacian omega
```

Representation hint:

- Potential flow explains many waves but not breaking, spray, rolling edges, or
  turbulent foam.
- A small `omega_i` payload can act as a "this region cannot be explained by a
  scalar potential" escape valve.

Hypothesis:

```text
Use potential-like state for coherent sheets; allocate vorticity-like residual
tokens only where held-out prediction improves enough to justify the rate.
```

Small local test later:

- Generate two synthetic sequences with identical RGB difficulty:
  `traveling_sine` and `rolling_vortex_sheet`.
- Compare:
  1. position + velocity
  2. potential scalar only
  3. potential scalar + sparse vorticity residual
- Decision metric: held-out time/camera error per exported byte/token.

## Gauge Constraints

The fluid math is full of gauge freedoms. Dynaworld already has its own gauge
problem: many latent worlds render the same supported queries. The goal is not
to identify a unique physical state; it is to choose a stable representative
that predicts held-out cameras/times.

Candidate gauge constraints:

```text
center-of-mass gauge:
    subtract global translation unless camera support identifies it

material-label gauge:
    cells are permutation-invariant; only rendered fields and local invariants
    matter

potential gauge:
    phi and psi are defined up to additive constants; penalize gradients or
    time differences, not absolute scalar offset

pressure gauge:
    pressure is defined up to a constant; only grad p or projection residual
    should matter

phase gauge:
    wave phase can shift by 2*pi; use sin/cos phase or local wave vector
    instead of raw unwrapped phase when possible

camera/world gauge:
    do not use a dynamics loss to secretly pin a camera convention unless that
    convention is supported by the observation/query sampler
```

Actionable rule:

```text
Every proposed physics variable needs an invariance check:
if an additive constant, permutation, global rigid transform, or phase wrap
changes the loss while leaving all renders equivalent, the loss is probably
punishing gauge instead of behavior.
```

## Hamiltonian And Lagrangian Views

### Hamiltonian View

For many wave systems, state is a canonical pair:

```text
q = eta or cell position
p = psi or momentum
H(q,p) = kinetic(q,p) + potential(q)

d_t q =  dH/dp
d_t p = -dH/dq
```

Representation hint:

- Store a conjugate motion variable, not just a time index.
- Penalize energy drift over short windows when the scene is approximately
  conservative.
- Use learned damping/source terms explicitly rather than letting the model
  hide all non-Hamiltonian behavior in appearance changes.

Candidate loss:

```text
L_energy =
    mean_t robust((H_t+1 - H_t) - W_external_t + D_damping_t)
```

This should be optional. Real videos have forcing, occlusions, camera errors,
contact, and non-water dynamics. The loss is valuable only when used as a local
diagnostic or low-weight stabilizer.

Small local test later:

- Fit a harmonic oscillator or linear wave packet:

```text
H = 0.5 * p^2 + 0.5 * omega^2 * q^2
```

- Verify that a Hamiltonian-style update extrapolates phase better than an MLP
  at equal parameter count.

### Lagrangian View

Lagrangian coordinates label material parcels:

```text
X              material coordinate
x(X,t)         world position
F = d x / d X  deformation gradient
J = det(F)     local area/volume change
```

Representation hint:

- PowerFoam-like cells naturally look Lagrangian: a cell can carry its material
  identity and move through world space.
- Rendering is Eulerian/camera-space; learning can be material-space.

Useful constraints:

```text
mass/opacity:        a_t * J_t approx a_0
orientation transport:
    q_t+1 approx integrate(q_t, local_angular_velocity)
support transport:
    Sigma_t+1 approx F_t Sigma_t F_t^T + process_noise
feature transport:
    f_t+1 approx f_t for material color, except lighting/view-dependent heads
```

Small local test later:

- Make a textured deforming sheet with known deformation gradient.
- Give cells material coordinates and compare against free per-frame cells.
- Failure signal: free cells win source reconstruction but lose when rendering
  an omitted camera because texture/material identity is unstable.

## Stability Losses Worth Trying First

These are cheap, local, and falsifiable. They should be toggles or diagnostic
terms, not permanent doctrine.

### 1. Material Advection Loss

```text
L_advect_feature =
    mean_i || stopgrad(f_i(t)) - sample_feature_at(x_i(t+1) - v_i dt, t+1) ||
```

Use for material-like features only. Do not apply blindly to specular highlights
or view-conditioned features.

### 2. Mass / Opacity Continuity

```text
L_mass =
    mean_cells robust(a_i(t+1) - a_i(t) + dt * div(a v)_i)
```

For splats/cells without a mesh, estimate `div(a v)` from k-nearest neighbors.
The first implementation can be a pure diagnostic logged on synthetic data.

### 3. Divergence Or Area Regularity

```text
L_area =
    robust(log det(Sigma_t+1) - log det(Sigma_t) - dt * trace(J_v_t))
```

This checks whether support volume changes are consistent with velocity
divergence. It can catch arbitrary scale pumping.

### 4. Acceleration Smoothness In Material Coordinates

```text
L_accel = mean_i ||x_i(t+1) - 2 x_i(t) + x_i(t-1)||_Huber
```

This is the least physics-specific baseline. It is still useful because it
detects per-frame repaint routes. It should be compared against stronger
PDE-inspired losses rather than assumed sufficient.

### 5. Symplectic / Reversibility Residual

```text
forward:  z_t -> z_t+1
backward: z_t+1 -> z_t'
L_rev = ||z_t' - z_t||
```

Only use on reversible synthetic or near-conservative scenes. Breaking waves
and dissipative foam should fail this test for physical reasons.

### 6. CFL-Like Step Bound

For a cell radius/support `r_i`:

```text
C_i = ||v_i|| dt / max(r_i, eps)
```

Log quantiles of `C_i`. If many cells move several radii per frame, local
advection losses become ambiguous and neighbor estimates become unstable.

## Actionable Representation Hypotheses

### H1: Dynamic Cells Need Momentum-Like State

Hypothesis:

```text
Adding velocity or potential-like state improves held-out time prediction more
than increasing appearance capacity at the same exported rate.
```

Why it might be true:

- Waves are phase systems; position-only state forgets where the phase is going.
- Held-out time queries punish phase drift more than source reconstruction does.

What would make it false:

- The sampler does not ask for temporal extrapolation/interpolation.
- Appearance motion is dominated by lighting, occlusion, or camera error rather
  than material transport.

Cheap test:

- Synthetic moving surface with held-out intermediate/future frames.
- Compare equal-rate `(x, f)` vs `(x, v, f_reduced)`.

If supported:

- Add a velocity payload path and a material-coordinate smoothness diagnostic.

If invalidated:

- Keep motion implicit and spend capacity on camera/time support or features.

### H2: A Potential Plus Sparse Vorticity Split Is Better Than One Generic MLP

Hypothesis:

```text
Most coherent sheet motion is low-curl/potential-like, while hard regions need
sparse vorticity residuals. A split representation gives better rate-distortion
than a uniform dynamic MLP.
```

Why it might be true:

- Ocean waves before breaking are often modeled well by potential flow.
- Breaking/spray/foam are localized exceptions, not the whole field.

What would make it false:

- Real target videos are mostly contact, articulation, or camera-induced motion.
- The vorticity residual becomes a generic hidden feature cache.

Cheap test:

- Train on two synthetic fields: traveling sine and rolling vortex.
- Measure held-out render quality and residual token usage.

If supported:

- Use a rate penalty on vorticity residual allocation.

If invalidated:

- Avoid adding specialized curl/potential channels to the main model.

### H3: Gauge-Aware Physics Losses Beat Absolute-State Losses

Hypothesis:

```text
Losses on gradients, residuals, rendered effects, and conservation laws are
more robust than losses on absolute potential, pressure, phase, or token order.
```

Why it might be true:

- Fluids have pressure/potential gauges.
- Dynaworld worlds are only identifiable through supported queries.

What would make it false:

- The synthetic task fixes a canonical state and the future architecture must
  export exactly that state.

Cheap test:

- Apply constant shifts to potential/pressure and permutations to cells.
- Assert identical loss/diagnostic values for gauge-invariant terms.

If supported:

- Require gauge invariance checks before adding physics terms to training.

If invalidated:

- Document the canonicalization requirement explicitly; do not let it leak in
  accidentally.

### H4: Conservation Diagnostics Can Detect Repainting Before RGB Metrics Do

Hypothesis:

```text
Mass, feature-advection, and CFL diagnostics catch per-frame delete/recreate
behavior that source RGB loss tolerates.
```

Why it might be true:

- A renderer can match each frame independently while carrying no stable world.
- Held-out camera/time prediction should need stable transport.

What would make it false:

- Occlusion or topology changes dominate, making conservation misleading.
- The model lacks enough support to infer material identity.

Cheap test:

- Build a synthetic "teleporting opacity" baseline that matches frames but
  violates continuity.
- Confirm that diagnostics flag it and held-out queries are worse.

If supported:

- Log these diagnostics on dynamic PowerFoam smoke runs before adding gradients.

If invalidated:

- Treat conservation as domain-specific, not a generic video prior.

## Minimal Local Test Ladder

Do not start with a full trainer fork. Start with small artifacts:

1. `math_helper` level:
   implement pure tensor helpers for `div`, `curl`, `CFL`, `mass_residual`,
   and gauge-invariant potential/pressure normalization on toy point sets.

2. Synthetic state generator:
   create in-memory trajectories for:
   `traveling_sine`, `shallow_water_bump`, `divergence_free_vortex`,
   `compressing_sheet`, and `teleporting_opacity_negative_control`.

3. Diagnostic-only pass:
   run existing dynamic cell outputs or synthetic fitted cells through the
   helpers. No training loss yet.

4. One-loss ablation:
   add exactly one low-weight term, preferably `L_mass` or `L_accel`, and test
   whether held-out time/camera improves at equal render loss.

5. Representation ablation:
   compare payloads:

```text
A: x, Sigma, opacity, feature
B: A + velocity
C: A + potential scalar
D: A + potential scalar + sparse vorticity residual
```

6. Export/rate check:
   report held-out quality per payload byte/token. If a physics variable only
   helps by increasing rate, it is not yet a better representation.

## Failure Modes To Watch

- Physics loss becomes an implementation-detail test that never predicts better
  held-out views.
- The model learns camera convention through dynamics constraints and appears
  "more physical" only because the gauge was over-pinned.
- Conservation penalties punish real visibility changes, disocclusion, foam
  birth/death, or specular appearance changes.
- Potential-flow variables are forced onto rotational/breaking regions where
  the right answer is a residual or a different representation.
- Momentum variables become a hidden per-frame cache if the sampler does not
  require time extrapolation/interpolation.
- Energy losses silently assume no forcing/damping when the video has wind,
  contact, camera exposure changes, or learned appearance sources.

## Process Recommendation

The first useful artifact should be a diagnostic notebook/script or unit-test
fixture, not a new architecture. The strongest near-term deliverable would be:

```text
synthetic dynamic-cell trajectories
plus gauge-invariant residual helpers
plus a small report showing which residuals distinguish:
    real advection
    smooth but non-conservative motion
    delete/recreate repainting
    potential flow
    vortex flow
```

Only after those diagnostics correlate with held-out camera/time behavior should
they become training losses or state variables in PowerFoam/Dynaworld.

