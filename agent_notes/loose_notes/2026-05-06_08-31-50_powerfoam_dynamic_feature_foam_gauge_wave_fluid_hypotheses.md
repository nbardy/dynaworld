# PowerFoam Dynamic Feature Foam: Gauge/Wave/Fluid Hypotheses

Date: 2026-05-06 08:31:50 +0700

## Context

This is a theory note only. It does not edit code, configs, verifiers, or
baselines.

Current objective:

```text
Full PowerFoam proper on Metal:
    accurate forward/backward
    official CUDA/Warp parity controls
    synthetic 4K trainability
    paper-scale heldout PSNR/SSIM
    trainable moving foams, not repainting a near-fixed grid
```

Current gate status used here:

```text
Renderer/parity/4K:
    local Metal/Torch/official-fixture parity and saved synthetic 4K optimizer
    step artifacts are no longer the main blocker.

Paper acceptance:
    selected clean row still misses the gate:
        heldout PSNR: 12.6689  (< 13.0)
        heldout SSIM:  0.1000  (< 0.15)
    regular appearance row:
        heldout PSNR/SSIM: 12.5099 / 0.1169
    many rows select step 0 or improve source while drifting heldout.

Dynamic feature foam:
    high-quality F32 feature foam can fit by low-motion repainting.
    a motion-honesty probe can force large motion, but quality collapses.

CUDA control:
    dynamic feature-foam CUDA micro exists as a cheap control. The fixed-black
    strict micro row has a rendered time-causality signal, but it is not a
    paper-quality row.
```

Do not blur these:

```text
trainable renderer      != paper-quality PowerFoam
screen motion           != material motion
feature delta           != coherent world transport
CUDA micro smoke        != quality benchmark
source-view PSNR        != representation selector
```

## Current Model

Provisional belief:

```text
The next useful theory target is not "add fluid simulation."
It is a gauge-aware material-transport test for whether dynamic feature foam
is moving persistent world support or deleting/repainting appearance.
```

Define "moving foam" strictly:

```text
support motion:
    cells/foam supports change world position/orientation/depth coherently.

material identity:
    density/features attached to a material label persist under transport,
    except for explicit source/sink terms.

view consistency:
    motion improves heldout cameras/times at matched capacity and rate.

control consistency:
    the same qualitative signal survives CUDA/Metal and fixed-background
    controls, or the result is implementation-specific.
```

Failure signature of repainting:

```text
source PSNR rises
heldout PSNR/SSIM stagnates or falls
cell centers/surfaces move little
features/densities change a lot
identity matching across time is weak
time-shuffle/reverse-time controls do not hurt much
```

## Assumptions

- Heldout camera/time metrics are the selector. Source-view fit is only a
  debugging signal.
- The current paper gate is mostly a structural/material/coverage problem, not
  a missing low-level Metal gradient.
- Dynamic F32 feature payloads are expressive enough to hide motion failures.
- Foam can have birth/death. Therefore conservation laws must expose explicit
  source terms instead of forbidding opacity changes.
- Gauge losses must be invariant to cell permutation, local tangent-frame
  rotations, potential offsets, phase wraps, and benign internal
  reparameterizations.
- CUDA is a control for mechanism validity, not the target runtime for the
  local Metal lane.

## Branches

### Branch A: Feature Repainting Dominates

Hypothesis:
    F32 features and colorizer capacity give the model a cheap route: keep a
    near-fixed support lattice and repaint per-frame features.

Why it might be true:
    The good F32 run had low temporal screen motion. The forced-motion probe
    moved, but quality fell. Static appearance-only PowerFoam rows can improve
    source while heldout barely moves or degrades.

What would make it false:
    A matched run with high identity persistence and nontrivial world support
    motion improves heldout PSNR/SSIM, not just source PSNR.

Decision implication:
    If supported, new math should first penalize hidden source terms and
    measure advection residuals before adding more feature/color capacity.

### Branch B: Geometry Is Under-Witnessed, Not Dynamically Wrong

Hypothesis:
    The paper gate misses because the static cell complex is still weak:
    mostly two-camera support, weak depth/material coverage, and hard heldout
    parallax. Dynamic/wave theory will not fix the static acceptance row.

Why it might be true:
    Distortion-consistent regular traversal improved heldout strongly, but all
    filtered points, stronger normals, support thaw, color affine, plane sweep,
    and nearby-heldout controls did not reach the gate.

What would make it false:
    A diagnostic shows high heldout residual is explained by temporal
    transport/identity failures even on well-witnessed, high-alpha regions.

Decision implication:
    If supported, keep dynamic theory as diagnostics only until clean geometry
    and track support move the static paper gate.

### Branch C: Gauge Incoherence Breaks Novel Views

Hypothesis:
    Neighboring PowerFoam cells can satisfy source rays while disagreeing in
    tangent frame, material chart, or support transport. The renderer sees
    alpha/RGB; heldout views see broken chart transitions.

Why it might be true:
    The gauge-field lane already found source PSNR can rank the wrong
    representation. Cell adjacency and regular triangulation changed heldout
    behavior materially, so topology/connection is an active variable.

What would make it false:
    Face-frame holonomy, transport mismatch, and witness uncertainty are
    uncorrelated with heldout residual after controlling for alpha/depth.

Decision implication:
    If supported, try gauge-covariant diagnostics/losses on the existing cell
    graph. Do not regularize absolute quaternions or raw frame angles.

### Branch D: Fluid/Wave Priors Help Only On Synthetic Motion

Hypothesis:
    Fluid/wave-inspired priors are useful for synthetic traveling sheets or
    foam-like clips, but not for the current static DeepView paper gate.

Why it might be true:
    Current paper rows are static posed-camera multiview acceptance. The
    strongest blockers mention structural/depth/material/track quality more
    than time extrapolation.

What would make it false:
    Cheap dynamic diagnostics explain heldout residual on real rows, or a
    motion-prior toggle improves heldout camera/time without source overfit.

Decision implication:
    Keep wave/fluid work in a separate dynamic feature-foam lane until it shows
    a heldout selector win.

### Branch E: Metal-Specific Dynamics Artifact

Hypothesis:
    A motion or time-causality signal could come from local Metal plumbing,
    background handling, or schedule artifacts rather than the representation.

Why it might be true:
    Dynamic CUDA exists only as micro/control evidence; quality rows are local
    Metal. Background differences can hide or expose time signals.

What would make it false:
    Fixed-black CUDA micro, matched Metal micro, and local synthetic cases show
    the same time-causality and motion/identity diagnostics.

Decision implication:
    Any claimed dynamic feature-foam mechanism needs a CUDA/Metal control
    before becoming a paper-acceptance argument.

## Cheap Falsification Tests

### Test 1: Identity-Persistence Diagnostic

Measure:

```text
For trained dynamic feature foam:
    match cells i(t) -> j(t+1) by nearest transported center and visibility.
    report corr(f_i(t), f_j(t+1)), density change, alpha contribution overlap.
```

Supports repainting if:

```text
source PSNR high
cell support motion low
feature/density identity persistence low
heldout residual high where persistence is low
```

Invalidates repainting if:

```text
features/densities persist under transport and high-persistence regions carry
the heldout improvement.
```

Decision:
    If supported, add source-rate/advection metrics before adding losses.

### Test 2: Freeze-Motion vs Freeze-Feature Matched Ablation

Measure:

```text
A. freeze geometry/support, train feature payload only
B. freeze feature payload/colorizer, train support motion only
C. train both
same cells, schedule, cameras, background, and eval cadence
```

Supports repainting if:

```text
A recovers most source PSNR and heldout does not improve;
B moves support but quality collapses;
C looks like A.
```

Invalidates repainting if:

```text
B or C gives post-initial heldout PSNR/SSIM improvement tied to support motion.
```

Decision:
    If A dominates, prioritize material identity/transport losses. If B helps,
    prioritize motion priors and lower feature capacity.

### Test 3: Time-Shuffle / Reverse-Time Control

Measure:

```text
Train or evaluate with frame order permuted/reversed while keeping per-frame
appearance distribution similar.
```

Supports repainting if:

```text
metrics and motion diagnostics barely change under time disorder.
```

Invalidates repainting if:

```text
time disorder sharply hurts heldout time/camera and increases transport
residuals.
```

Decision:
    If shuffle is cheap, the current dynamic model is not learning temporal
    mechanism. Do not cite it as moving foam.

### Test 4: CUDA/Metal Fixed-Black Dynamic Micro Pair

Measure:

```text
Run strict fixed-black CUDA dynamic feature-foam micro and matched Metal micro.
Compare:
    warm-step sanity
    rendered time RGB delta
    train/eval L1/MSE/PSNR/SSIM
    qualitative frame difference
```

Supports a real mechanism if:

```text
both runtimes show the same time-causality direction and no background leak.
```

Invalidates runtime-independent claims if:

```text
the signal appears in only one runtime or disappears under fixed black.
```

Decision:
    Use CUDA as a control before treating a dynamic Metal result as evidence.

### Test 5: Occlusion-Weighted Mass/Source Residual

Measure:

```text
m_i(t) = rho_i(t) * A_i(t)
R_i = (m_i(t+1) - m_i(t)) / dt
      + graph_flux_i(t)
      - explicit_source_i(t)
weight by detached visibility and alpha contribution
```

Supports material-transport priors if:

```text
high visible residual predicts heldout residual better than source residual.
```

Invalidates if:

```text
residual mostly tracks occlusion, low alpha, or bad visibility estimates.
```

Decision:
    If supported, turn source-rate into a first-class logged metric; only later
    consider a weak robust loss.

### Test 6: Cell-Graph Holonomy vs Heldout Error

Measure:

```text
For active PowerFoam adjacency:
    define face-aware tangent-frame transport U_ij.
    compute loop holonomy or edge frame mismatch.
    bucket by alpha and heldout residual contribution.
```

Supports gauge-connection direction if:

```text
high-residual high-alpha regions have high holonomy/mismatch, while source
residual does not explain it.
```

Invalidates if:

```text
holonomy is noise, visibility proxy, or uncorrelated with heldout.
```

Decision:
    If supported, try connection regularization on witnessed faces only.

### Test 7: Traveling-Sheet Synthetic Gate

Measure:

```text
Generate a simple material sheet:
    translate
    sinusoidal traveling wave
    local compression/expansion
Train current dynamic feature foam and a variant with velocity/phase state.
Hold out camera and intermediate/future time.
```

Supports wave/transport state if:

```text
current model gets source frames but drifts phase/identity on heldout time;
velocity/phase state preserves heldout time/camera at matched rate.
```

Invalidates if:

```text
current model solves the synthetic with low residual and no repaint signature.
```

Decision:
    If invalidated, do not add wave state for real rows yet.

### Test 8: Feature-Capacity Pressure Test

Measure:

```text
Run matched F3/F8/F32 payloads or colorizer bottlenecks.
Track:
    support motion
    feature delta
    source PSNR
    heldout PSNR/SSIM
    identity persistence
```

Supports repainting if:

```text
higher F raises source PSNR while support motion and heldout stagnate.
```

Invalidates if:

```text
higher F improves heldout while preserving identity and motion evidence.
```

Decision:
    If supported, treat feature dimension as an appearance shortcut knob.

## New Math Directions

### Direction 1: Gauge-Covariant Material Transport On Cell Graphs

Object:

```text
cell i:
    x_i(t), E_i(t), n_i(t), rho_i(t), f_i(t)
edge i-j:
    radical face or graph face
    transport map T_ij between local tangent/material charts
```

Transport law:

```text
u_j(t+dt) ~= T_ij(u_i(t) + dt * v_i^T)
f_j(t+dt) ~= transported(f_i(t)) + source_f
rho_j(t+dt) ~= transported(rho_i(t)) + source_rho
```

Gauge rule:

```text
absolute tangent-frame angle is arbitrary;
only transported quantities around edges/loops are observable.
```

First deliverable:
    diagnostics for edge transport mismatch and loop holonomy. Loss only if
    mismatch predicts heldout residual.

### Direction 2: Conservative Surface Measure With Explicit Source Terms

State:

```text
A_i(t)       represented surface area proxy
rho_i(t)     material density per area
m_i(t)       rho_i(t) * A_i(t)
s_i(t)       explicit birth/death source
```

Residual:

```text
R_mass_i =
    (m_i(t+dt) - m_i(t)) / dt
  + sum_j flux_ij
  - A_i s_i
```

Why it matters:
    It converts hidden alpha/feature repainting into an explicit paid source
    term while allowing real foam birth/death.

Gauge rule:
    The conserved quantity is measure over material surface, not raw cell
    index or arbitrary alpha.

First deliverable:
    visibility-weighted residual histogram and correlation with heldout error.

### Direction 3: Tangent-Chart Wave Phase / Velocity Potential

State:

```text
phi_i(t)     phase or velocity-potential-like scalar
k_i(t)       local tangent wave vector
h_i(t)       height/free-surface displacement
v_i^T        tangent velocity, optionally grad_T phi
w_i          normal velocity
```

Local residuals:

```text
phase transport:
    wrap(phi_i(t+dt) - phi_i(t) + omega_i dt)

dispersion prior:
    omega_i^2 ~= g |k_i| tanh(|k_i| depth_i)

height/velocity consistency:
    h_i(t+dt) - h_i(t) ~= dt * w_i
```

Why it matters:
    A traveling wave should preserve phase and momentum. A repainting model can
    match frames while losing phase identity.

Gauge rule:
    `phi` is modulo phase wraps and arbitrary global offsets.

First deliverable:
    synthetic traveling-sheet gate. Do not wire this into real PowerFoam until
    it beats the current model on heldout time/camera.

### Direction 4: Helmholtz-Style Motion Split

State:

```text
v = grad_T phi + curl_T psi + r
```

Interpretation:

```text
grad_T phi:
    coherent potential/wave motion
curl_T psi:
    vortex/rolling/swirling residual
r:
    sparse defect/source escape hatch
```

Why it matters:
    Potential-only priors can overconstrain breaking, rolling foam, cloth-like
    motion, and spray. A split gives an auditable way to allocate complexity.

First deliverable:
    synthetic pair:
        traveling_sine should prefer potential
        rolling_vortex_sheet should require curl residual

Decision:
    If the split does not improve heldout synthetic behavior at matched rate,
    do not add it to real training.

### Direction 5: Witness-Weighted Priors

Object:

```text
W_i or W_ij = sum over contributing rays of w_k (I - d_k d_k^T)
witness = eigmin(W) / trace(W)
```

Why it matters:
    Under-witnessed cells/faces are gauge-free. Strong priors there can create
    plausible but wrong geometry; no priors there can allow repainting.

Rule:

```text
diagnostics and losses should be weighted by visibility, alpha, and witness.
```

First deliverable:
    compare heldout residual against witness-weighted transport/holonomy
    residuals, not raw residuals.

## Decision Implications

If Branch A is supported:
    Treat F32 feature foam as an appearance shortcut until it passes identity
    persistence and time-shuffle tests. Add source-rate/advection diagnostics
    before adding representation capacity.

If Branch B is supported:
    Do not sell wave/fluid priors as the path to paper acceptance. Keep focus
    on clean geometry, track support, material/depth structure, and the current
    completion audit.

If Branch C is supported:
    The next math should be cell-graph transport/holonomy on witnessed faces,
    not absolute quaternion/frame smoothing.

If Branch D is supported:
    Keep synthetic wave/fluid benchmarks as a separate dynamic lane. They can
    be valuable without being relevant to the static paper gate.

If Branch E is supported:
    Dynamic claims need CUDA/Metal fixed-background controls before promotion.

Minimum bar for a future "moving foam" claim:

```text
1. post-initial heldout camera/time improvement
2. nontrivial world/support motion
3. feature/density identity persistence under transport
4. source-rate/repaint residual not doing the work
5. CUDA/Metal or equivalent implementation control
6. paper-gate PSNR/SSIM tracked separately from motion diagnostics
```

## Open Questions

- Is the current selected clean paper row's high residual concentrated in
  areas where material identity diagnostics would even be meaningful, or is it
  dominated by static support/coverage error?
- Can replay metadata expose enough face/endpoint ownership to compute
  witness-weighted transport without a kernel change?
- Does a weak source-rate penalty improve heldout, or does it only suppress
  necessary foam birth/death?
- Is F32 feature capacity too high for diagnosing motion, or is the support
  parameterization too weak to move coherently even when features are limited?
- Does CUDA dynamic feature foam reproduce the same identity and time-causality
  signals as local Metal when backgrounds, schedules, and seeds are matched?
