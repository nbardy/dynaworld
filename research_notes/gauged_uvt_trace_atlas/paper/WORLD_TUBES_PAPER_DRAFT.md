---
title: "World Tubes in Gauged Camera Space: Frame-Amortized Dynamic Gaussian Rendering"
author: Anonymous
date: 2026-07-28
bibliography: research_notes/gauged_uvt_trace_atlas/paper/WORLD_TUBES_REFERENCES.bib
link-citations: true
---

## Abstract

Dynamic Gaussian splatting methods render a time-varying scene by evaluating
or deforming primitives at a target timestamp, projecting the resulting
Gaussians, binning them into tiles, resolving visibility, and compositing the
visible contributors. This pipeline is efficient for isolated frames, but it
repeats the same world-side work when a renderer must produce many temporally
nearby samples from a known camera program: video playback, finite exposure,
rolling shutter, dense temporal supervision, or repeated training/evaluation
views.

We introduce **World Tubes in Gauged Camera Space**, a camera-path compiler for
dynamic Gaussian primitives. The core object is a sensor-time trace atlas over
`(u,v,t)`: each spacetime primitive is pulled back through the camera-ray
bundle and pushed forward along the ray fiber to produce a reusable
viewport-time footprint, conditional depth model, support certificate, and
adjoint structure. In a nonsingular affine local gauge, with an untruncated
fiber and locally constant measure factor, a spacetime Gaussian yields a UVT
footprint by exact Schur-complement fiber marginalization. Perspective camera
programs instead require a certified local approximation or the implemented
projective trace family; the affine closure is not asserted globally. Camera
gauges and event-certified domains make the underlying trace integral invariant
to depth coordinates on bounded camera-path chart segments, including the
tested orbit segments, finite exposure, and rolling shutter. The renderer
evaluates frames as slices of the compiled atlas, while training accumulates
gradients through a fixed-topology compiled interval VJP.

On our current STAR UVT / projective interval implementation, a bounded
same-representation fixture over `F={4,8,16,32,64,128}` keeps fixed logical
tensor-element volume growth at `1.0x` while per-frame replay grows `32.0x`;
the final fixed/per-frame trace-count ratio is `0.03125`. A broad real-video
audit covers 10 source-distinct cases, 20 projective-interval trainer
payloads, and gradient-preserving compiled-adjoint replacement. These results
establish structural reuse and bounded correctness, not a publication timing
claim. Warming and repeating replay versus compiled evaluation on one frozen
learned world is the required runtime experiment. The method does not claim
an information-theoretic sublinear bound in the number of output pixels: the
remaining per-pixel shading term is real, while projection, support, binning,
visibility metadata, and backward replay are the work targeted for
camera-program amortization.

## 1. Introduction

3D Gaussian Splatting (3DGS) made real-time neural rendering practical by
replacing expensive volumetric ray marching with visibility-aware anisotropic
splatting [@kerbl2023]. Dynamic extensions such as 4D Gaussian Splatting,
Deformable 3D Gaussians, Dynamic 3D Gaussians, and Spacetime Gaussian Feature
Splatting extend the representation through deformation fields, persistent
motion, or spacetime primitives
[@wu2024_4dgs; @yang2024_deformable; @luiten2024; @li2024_spacetime].
These methods improve dynamic scene modeling, but they usually retain a
per-target-view rendering loop: evaluate the primitive state at the requested
timestamp, project to screen, build tile bins, estimate or sort depth order,
shade, composite, and backpropagate through that target render.

This is the wrong computational unit for several common workloads. A video
renderer needs many temporally adjacent frames. A finite-exposure renderer
needs many shutter samples per output image. A rolling-shutter camera couples
image row to capture time. Training often revisits the same camera path or a
small family of nearby paths many times. In these settings, the output image
samples still scale as `O(F H W)`, but the world-side work should not have to
scale linearly with frame count `F`. The paper's claim is therefore not that
pixels disappear. It is that the expensive training bottlenecks that dominate
dynamic Gaussian pipelines--projection, support, tile membership, ordering
metadata, and backward replay--can scale with trace/event complexity instead
of frame count. The bounded fixture verifies structural reuse; the warmed,
repeated frozen-world experiment is required before making a measured
runtime-scaling claim.

We propose to compile the dynamic world through the camera program itself. A
world primitive is not projected independently into each frame; it becomes a
**world tube** observed as a **sensor-time trace**:

```text
world spacetime primitive -> alpha_i(u,v,t), c_i(u,v,t), z_i(u,v,t).
```

The compiled object is a **trace atlas** over the sensor-time base:

```text
B = Omega x T,      y = (u, v, tau).
```

Each atlas domain stores active primitive traces, support bounds, local depth
models, visibility/order certificates, fallback metadata, and differentiable
state. Frames are slices of this atlas, and finite-exposure or rolling-shutter
images are integrals or row-coupled samples through the same object.

The method keeps four layers distinct:

| Layer | Object | Contract |
|---|---|---|
| World representation | `W_theta` | Camera-independent spacetime primitives and appearance. |
| Camera-program compiler | `phi = C_Gamma(theta;kappa)` | Lower one world and camera program into trace coefficients under certified discrete topology `kappa`. |
| Atlas evaluator | `I = R(phi,kappa;y)` | Evaluate one sensor-time sample or batch without changing the learned world. |
| Compiled adjoint | `D_theta C_Gamma(theta;kappa)^T D_phi R(phi,kappa)^T` | Map image residuals through the evaluator and compiler back to the same world parameters. |

This separation matters experimentally. A faster evaluator with a different
learned world is not evidence for compilation; the causal comparison freezes
`W_theta` and changes only whether `C_Gamma` is replayed per frame or shared
over the camera program.

![System overview. The causal comparison starts from the same learned world and
known camera program. Per-frame replay rebuilds projection, support, binning,
and order at each requested time; World Tubes compiles shared projective traces
and certified tile-time cells, evaluates them with an interval Metal forward,
and maps residuals through the interval and compiler VJPs to the same world
parameters. Output production remains linear in the requested sensor samples.](research_notes/gauged_uvt_trace_atlas/paper/figures/world_tubes_system_overview.svg)

This construction is not merely a renamed STAR UVT renderer. STAR UVT supplies
the spacetime Gaussian representation and the sparse Metal execution lineage;
the gauged camera-ray formulation supplies the new compiler semantics. In
particular, it pulls primitives through a moving camera program, marginalizes
the ray-depth fiber without discarding conditional depth, and partitions
support-overlap regions at certified depth-order events. A single raw interval
can therefore fail under a large-motion order crossing while the corresponding
visibility-stratified gauge domains remain replay-equivalent. This distinction
is part of the method and must not be removed by implementation cleanup.

Our contributions are:

1. **A gauged camera-program compiler for dynamic Gaussian rendering.** We
   define the invariant trace as `pi_* Gamma^* world_primitive`, then derive its
   exact affine-Gaussian UVT marginal and conditional depth packet by a Schur
   complement. The gauge Jacobian is part of the physical fiber measure, not an
   optional implementation correction.

2. **An event-certified projective trace and visibility atlas.** Homogeneous
   camera traces are lowered into bounded rational/polynomial interval records.
   Denominator, approximation, support, and local depth-order tests determine
   whether an interval is emitted, split, or routed to fallback. This retains
   the gauged large-motion/order-crossing mathematics rather than reducing the
   method to a faster STAR kernel.

3. **A fixed-topology compiled evaluator and adjoint.** We implement interval-
   compressed Metal forward and direct VJP paths. Trace coefficients, opacity,
   temporal opacity, spatial precision, and color remain
   differentiable, while support, bin membership, event topology, order, and
   fallback choices are explicitly held fixed within an adjoint block.

4. **A causal frame-amortization evaluation contract.** Per-frame replay and
   compiled evaluation share the exact learned world, camera samples, renderer,
   loss, and world-parameter targets. We separately report unavoidable
   sensor-sample work and the projection/support/binning/visibility/backward
   metadata that the camera-program compiler is designed to share.

## 2. Related Work

**Gaussian splatting.** Classical surface and EWA volume splatting formulate
point/volume primitives as filtered screen footprints, including analytic
integration of elliptical Gaussian reconstruction kernels in a locally affine
ray-space approximation [@zwicker2001_surface; @zwicker2001_ewa_volume]. 3DGS
introduced learned anisotropic 3D Gaussian primitives and a visibility-aware
rasterizer that supports real-time rendering and optimization [@kerbl2023].
Mip-Splatting later makes the sampling/filtering boundary explicit for learned
Gaussians [@yu2024_mipsplatting]. Our work keeps this rasterization lineage but
changes the unit of compilation from one screen at one time to a sensor-time
camera path.

**Dynamic Gaussian representations.** 4D-GS combines 3D Gaussians with 4D
neural voxels and lightweight deformation prediction. Deformable 3D Gaussians
learn a canonical Gaussian scene plus deformation field. Dynamic 3D Gaussians
track persistent Gaussians over time. Spacetime Gaussian Feature Splatting
adds temporal opacity and parametric motion/rotation to Gaussian primitives.
Native 4DGS instead models the spacetime volume directly with anisotropic 4D
Gaussian primitives
[@wu2024_4dgs; @yang2024_deformable; @luiten2024; @li2024_spacetime;
@yang2024_native]. Our strict SPD(4) source uses the same standard
full-covariance Gaussian mathematics; the contribution here is its
camera-program lowering, event certification, reuse, and adjoint, not a new
Gaussian family.
These methods primarily address the dynamic scene representation. We instead
target the repeated rendering work induced by known camera paths and many
temporal samples.

**Nonlinear cameras, rolling shutter, and ray-space splatting.** Gaussian
Splatting on the Move models blur and rolling shutter under natural camera
motion. 3DGUT replaces the EWA projection approximation with an unscented
transform to support nonlinear cameras, rolling shutter, and secondary rays.
Our method is complementary: sigma/projective projection helps define or test
gauge domains, while the trace atlas amortizes camera-path work over many
samples [@seiskari2024; @wu2025_3dgut].

**Compositing and order alternatives.** Weighted blended order-independent
transparency and Weighted Sum Rendering avoid exact sorting by changing the
rendering law, while Gaussian Blending replaces scalar per-pixel
alpha/transmittance with spatial distributions
[@mcguire2013_weighted_oit; @hou2025_sortfree; @koo2026_blending]. StopThePop
instead improves view consistency with hierarchical per-ray depth sorting and
makes the limitations of one representative splat depth especially relevant
under camera rotation [@radl2024_stopthepop]. World Tubes preserves the ordered
front-to-back transfer law of the frozen STAR representation. Its visibility
strata compile where that noncommutative ordering is stable and fall back where
it is not; this is necessary for a causal replay-versus-compiled comparison.

**Dynamic view-synthesis datasets.** Neural 3D Video, D-NeRF, and HyperNeRF
provide public real-multiview, controlled synthetic, and monocular dynamic-scene
benchmarks. Our paper evaluates Neural 3D Video as its positive public
multiview setting and uses D-NeRF only as a separately labelled posed-frame
control. The central experiments also include a controlled synthetic trace
suite where exact ray-fiber integration and visibility events are known
[@li2022_neural3dvideo; @pumarola2021_dnerf; @park2021_hypernerf].

## 3. Method

We use the following notation throughout. Script letters denote discrete
compiler objects; lowercase functions denote fields evaluated at a sensor-time
sample. In particular, `T` denotes the time domain only, while `Trans_m` denotes
front-to-back transmittance. We write `tau` for sensor-program time and use `t`
as its scalar shorthand when no exposure or rolling-shutter offset is being
distinguished.

| Symbol | Meaning |
|---|---|
| `B = Omega x T`, `y=(u,v,tau)` | Sensor-time base and one sensor-time sample. |
| `E_Gamma`, `pi:E_Gamma->B`, `F_y` | Camera-ray bundle, bundle projection, and ray-depth fiber over `y`. |
| `Gamma:E_Gamma->R^3 x R` | Known camera program, including its association between sensor time and world time. |
| `w_i`, `rho_i` | Camera-independent world primitive and its density/opacity field. |
| `bar_rho_i = pi_* Gamma^* rho_i` | Gauge-invariant sensor-time trace of primitive `i`. |
| `z_a`, `J_a` | Local ray-depth coordinate in gauge `a` and its physical fiber-measure factor. |
| `C_l`, `S_l`, `Phi_l`, `Pi_l` | Certified atlas cell, active primitive set, trace functions, and compiled order/partial order. |
| `phi=C_Gamma(theta;kappa)` | Differentiable atlas coefficients compiled from world parameters `theta` with discrete topology `kappa` fixed. |
| `I=R(phi,kappa)` | Atlas evaluator and rendered sensor-time samples. |
| `lambda(y)=partial L/partial I(y)` | Image-space adjoint. |

### 3.1 Sensor-time base and camera-ray bundle

Let the sensor-time base be:

```text
B = Omega x T,      y = (u, v, tau).
```

Let the camera program define a ray bundle:

```text
pi: E_Gamma -> B,
pi^{-1}(y) = F_y.
```

The fiber `F_y` is the ray-depth domain over the sensor sample `y`. A camera
program maps bundle points into world spacetime:

```text
Gamma: E_Gamma -> M,      M = R^3 x R.
```

For a world primitive `w_i`, the invariant sensor-time trace is:

```text
Trace_Gamma[w_i] = pi_* Gamma^* w_i.
```

For a density-like primitive, this becomes:

```text
bar_rho_i(y) = integral_{F_y} rho_i(Gamma(e)) dmu_y(e).
```

This equation is the renderer's invariant object. It does not depend on a
particular depth coordinate or local chart.

### 3.2 Gauges and event-certified domains

A gauge is a local trivialization of the ray bundle over a domain `C_a`:

```text
chi_a: E_Gamma | C_a -> C_a x D_a,
chi_a(e) = (y, z_a).
```

In gauge `a`, the trace is:

```text
bar_rho_i^a(y)
  = integral_{D_a} rho_i(Gamma_a(y, z_a)) J_a(y, z_a) dz_a.
```

The Jacobian `J_a` is not optional. It is the measure correction that makes
ordinary depth, log depth, inverse depth, orbit-angle, and projective gauges
represent the same physical trace. In our current gauge-invariance artifact,
ordinary-depth and log-depth integration of the same revolving-camera
spacetime Gaussian agree to `3.50e-13` relative error with the Jacobian; without
it, the value error is at least `0.600`. The matching gradient artifact agrees
to `2.33e-12` relative error with the Jacobian; without it, gradient error is
at least `0.592`.

**Proposition 1 (fiber-gauge invariance).** Let gauges `a` and `b` describe the
same physical fiber segment over `y`, with a continuously differentiable,
one-to-one coordinate change

```text
z_b = h_ab(y,z_a),
Gamma_a(y,z_a) = Gamma_b(y,h_ab(y,z_a)).
```

Assume the primitive is integrable on that segment and the physical line
measure is represented in the two coordinates by

```text
J_a(y,z_a) dz_a
  = J_b(y,h_ab(y,z_a)) |partial h_ab / partial z_a| dz_a.
```

Then `bar_rho_i^a(y)=bar_rho_i^b(y)`. The proof is the one-dimensional
change-of-variables formula on `F_y`. For an orientation-preserving gauge the
absolute value can be dropped. If the coordinate map ceases to be one-to-one,
its denominator reaches zero, or the two coordinates cover different clipped
fiber segments, the proposition no longer applies; the compiler must end the
cell, introduce a transition, or fall back. This is a coordinate-invariance
statement, not a claim that one chart remains valid across a physical or
projective event.

We use the term **gauge domain** instead of weak chart. A gauge domain certifies:

```text
projection denominator regularity
trace representation error
conservative support bounds
tile-time active set
conditional depth model
stable order, commutable order, or fallback
interval gates
backward support matching forward support
```

For revolving cameras, a projective/orbit gauge such as `r = tan(theta / 2)`
can make traces rational or low-order over longer intervals than naive frame
time. The domain still ends at real events: denominator zeros, behind-camera
transitions, near/far crossings, support entering/leaving tiles, order swaps,
and disocclusions.

**Scope of the current implementation.** Static cameras use the exact affine
gauge lowering, and moving-camera training has a differentiable one-chart
first-order compiler. Separately, the projective compiler and experiments
cover bounded, event-certified orbit segments inside one regular projective
chart. The first-order moving chart is not asserted to equal a nonlinear
pinhole program over arbitrarily long windows, and the projective path does not
yet implement chart transitions for complete `360°` or repeated `720°`
revolutions. We therefore make no exact long-window or full-orbit multi-gauge
claim.

### 3.3 Local Gaussian fiber pushforward

#### Strict SPD(4) source object

The representation-level source object is a Gaussian measure or splat-shaped
atom in world spacetime:

```text
x = (X,Y,Z,T) in R^4
mu_x in R^4
Sigma_x in Sym++(4)
rho_i(x) = a_i exp[-1/2 (x - mu_x)^T Sigma_x^{-1} (x - mu_x)].
```

Thus all ten independent covariance entries are available; an explicit
velocity parameter is not required. For example, conditioning the native
spacetime atom on time gives:

```text
E[X_xyz | T=t]
  = mu_xyz + Sigma_xyz,t Sigma_tt^{-1} (t - mu_t).
```

The space-time cross covariance therefore produces affine motion as a derived
slice of one 4D object. This does not produce arbitrary curved motion from one
Gaussian; curved trajectories require a richer primitive, a nonlinear
coordinate map, or multiple atoms.

Let a nonsingular affine local camera-ray gauge map world spacetime into ordered
coordinates `s=(u,v,z,t)`:

```text
s = G x + b
mu_s = G mu_x + b
Sigma_s = G Sigma_x G^T.
```

Reorder `s` as `(r,z)`, where `r=(u,v,t)`, and partition:

```text
Sigma_s =
  [ Sigma_rr  Sigma_rz
    Sigma_zr  Sigma_zz ].
```

Then the exact UVT marginal and conditional fiber packet are:

```text
r ~ N(mu_r, Sigma_rr)
Q_r = Sigma_rr^{-1}

E[z | r]
  = mu_z + Sigma_zr Q_r (r - mu_r)

nu_z = Var(z | r)
  = Sigma_zz - Sigma_zr Q_r Sigma_rz > 0.
```

The last strict inequality follows from the Schur complement of
`Sigma_s in Sym++(4)`. The packet is therefore a full anisotropic UVT
precision, an affine depth plane over `(u,v,t)`, and a positive conditional
depth variance. These are all consequences of one strict SPD(4) source, not
separate motion laws.

**Proposition 2 (standard conditional-Gaussian equivalence).** Fix a
nonsingular affine gauge `s=Gx+b`. A strict SPD(4) Gaussian in `s=(r,z)` is
equivalent to the tuple

```text
mu_r, mu_z,
Sigma_rr in Sym++(3),
beta in R^3,
nu_z > 0,
```

with conditional law

```text
z | r ~ N(mu_z + beta^T(r-mu_r), nu_z).
```

The forward direction is the partition above, with
`beta^T = Sigma_zr Sigma_rr^{-1}`. Conversely, the tuple reconstructs the
unique joint covariance

```text
Sigma_s =
  [ Sigma_rr             Sigma_rr beta
    beta^T Sigma_rr      nu_z + beta^T Sigma_rr beta ],
```

which is strict SPD because its Schur complement is `nu_z`; transforming with
`Sigma_x=G^{-1}Sigma_sG^{-T}` recovers the world covariance. This is the
ordinary marginal/conditional parameterization of a multivariate Gaussian.
We use it to specify a compiler ABI, not to claim a new probability
distribution or a new native-4D representation.

Two amplitude conventions must not be conflated. A peak-preserving splat keeps
`a_i` as the amplitude of the maximum joint density along the conditional
depth fiber and is compatible with STAR's factorized
`alpha = opacity exp(-q_r/2)` convention. A physical, untruncated
fiber-integrated density with locally constant fiber-measure factor `J_0`
instead produces an optical-depth coefficient multiplied by:

```text
J_0 sqrt(2 pi nu_z).
```

The compiler ABI must label which convention it emits. These are not two names
for the same alpha. If

```text
tau(r) = tau_0 exp(-q_r/2),
```

then physical Beer--Lambert opacity is:

```text
alpha_phys(r) = 1 - exp[-tau(r)].
```

The production interface now exposes both laws rather than silently
identifying them. `peak_splat` retains the historical factorized alpha, while
`beer_lambert` evaluates the physical map above with its exact alpha VJP and
cutoff-support equation. The orthogonal amplitude axis is
`fiber_integrated` versus `peak_density`. In the latter case the affine-gauge
compiler applies the fiber measure and `sqrt(2 pi nu_z)` factor to obtain
projected optical thickness; in the former the trainable value already has
that projected meaning. The retained-fiber renderer additionally keeps
`nu_z` and the conditional Gaussian depth profile instead of collapsing the
primitive to one alpha/depth event.

#### Equivalent local precision form

In a local gauge, linearize the camera map around `(y0, z0)`:

```text
Gamma_a(y,z) ~= x0 + J eta,
eta = [delta_y, delta_z]^T.
```

Let `Lambda_i = Sigma_x^{-1}`, `delta = mu_x - x0`, and
`g = J^T Lambda_i delta`, partitioned as
`g = [g_y, g_z]^T`. Then the local exponent is
`eta^T H eta - 2 g^T eta + delta^T Lambda_i delta`.

Plugging the linearized map into the primitive gives a Gaussian over
`(y,z)` with precision:

```text
H = J^T Lambda_i J
  = [ H_yy  H_yz
      H_zy  H_zz ].
```

Marginalizing the fiber coordinate `z` yields the UVT precision:

```text
S = H_yy - H_yz H_zz^{-1} H_zy.
```

The conditional fiber/depth model is:

```text
z_hat_i(y) = z0 + H_zz^{-1}(g_z - H_zy (y - y0)),
Var(z | y) = H_zz^{-1}.
```

For a scalar fiber with `H_zz > 0`, an untruncated local fiber, and locally
constant fiber-measure factor `J_0`, the marginal also has amplitude:

```text
bar_rho_i(y)
  ~= J_0 a_i sqrt(2 pi / H_zz) exp[-1/2 q_y(delta_y)],

q_y(delta_y)
  = delta_y^T S delta_y
    - 2 (g_y - H_yz H_zz^{-1} g_z)^T delta_y
    + delta^T Lambda_i delta - g_z^T H_zz^{-1} g_z.
```

This local closed form must be replaced by certified quadrature or a residual
bound when fiber clipping or a varying Jacobian is not negligible.

Thus a pulled-back 4D world Gaussian becomes a 3D sensor-time footprint plus
conditional depth and uncertainty. These quantities are exactly what a
rasterizer needs for support, tile-time binning, visibility ordering, and
gradient propagation. The covariance-block and precision-block derivations
above are equivalent Gaussian identities; the former defines the canonical
full-SPD(4) source contract, while the latter is convenient for a locally
linearized camera map.

### 3.4 Homogeneous projective traces and certified lowering

The affine Schur calculation gives the exact local Gaussian object, but a
moving pinhole camera should not be approximated first in divided screen
coordinates. Let `X_i^h(t)` be a homogeneous world point associated with
primitive `i`, and let the camera program supply the projective matrix `P(t)`.
We form the homogeneous camera trace

```text
h_i(t) = P(t) X_i^h(t)
       = (h_u,i(t), h_v,i(t), h_z,i(t)).

u_i(t) = h_u,i(t) / h_z,i(t),
v_i(t) = h_v,i(t) / h_z,i(t),
d_i(t) = h_z,i(t).
```

The quotient is taken only after the denominator has been certified. On a
candidate time interval `I_l=[t_l^-,t_l^+]`, define normalized local time

```text
t_l^c = (t_l^- + t_l^+) / 2,
t_l^s = (t_l^+ - t_l^-) / 2 > 0,
s     = (t - t_l^c) / t_l^s,       s in [-1,1].
```

For the implemented degree-one or degree-two trace family, the compiler stores

```text
h_k,i(s) = sum_{r=0}^p a_{k,i,r} s^r,
k in {u,v,z},       p in {1,2}.
```

These coefficients may be exact for a supported camera/motion family or fitted
from the declared camera samples. Keeping the homogeneous numerator and
denominator separate exposes projective boundaries and avoids fitting a smooth
polynomial through a pole in `u` or `v`.

For every primitive and candidate interval, the compiler evaluates four kinds
of conditions:

1. **Projective validity.** `h_z,i` must retain the declared physical sign and
   satisfy `min_{s in [-1,1]} |h_z,i(s)| >= epsilon_z`, together with the
   near/far-plane constraints. For `p<=2`, the denominator range and minimum
   margin are checked from the endpoints and stationary point; a sign/range
   test detects a real root. A between-frame pole therefore cannot pass merely
   because sampled frames are valid.
2. **Trace approximation.** The divided trace and conditional-depth record must
   meet the declared UV/depth tolerance against the camera-program samples used
   by the certificate. A supplied analytic interval bound may replace this
   sampled residual.
3. **Support validity.** The projected footprint plus declared approximation
   padding must conservatively determine active tiles and interval gates.
4. **Visibility validity.** Every locally support-overlapping pair must have a
   stable order, an accepted commutation error, or an explicit fallback label.

The present implementation has a continuous quadratic denominator test. Its
general UV/depth fit residual is verified on the declared bounded probe/sample
set; it is not presented as a continuous-time approximation theorem. The
variable-camera experiment likewise compares 64 fixed physical-time samples
against exact rational centers and live per-sample order. This distinction is
why the paper claims bounded tested camera-program segments rather than exact
arbitrary trajectories.

An accepted interval is lowered into one STAR-compatible trace record:

```text
R_i,l = {
  homogeneous center coefficients a_u, a_v, a_z,
  opacity and optional temporal-opacity coefficients,
  spatial precision / support padding,
  cell depth and optional affine depth-plane coefficients,
  color or feature payload,
  active time interval [t_l^-,t_l^+),
  active tiles and compiled local order
}.
```

The lowering policy is deterministic and fail-closed:

```text
compile_camera_program(world, Gamma, tolerances):
    queue <- initial bounded camera-program intervals
    atlas <- empty

    while queue is not empty:
        I <- pop(queue)
        H <- fit_or_form_homogeneous_traces(world, Gamma, I)
        Dcert <- certify_denominators_and_physical_depth(H, I)
        Ecert <- measure_trace_residual_and_support(H, I)

        if Dcert or Ecert fails:
            if I can be split:
                push split_at_midpoint(I)
            else:
                atlas.emit_fallback(I, reason=Dcert or Ecert)
            continue

        G <- build_local_support_overlap_graph(H, I)
        Ocert <- certify_cell_depth_order(G, I)
        if Ocert is unresolved:
            if I can be split:
                push split_at_certified_order_root_or_midpoint(I)
            else:
                atlas.emit_fallback(I, reason=Ocert)
            continue

        atlas.emit(lower_to_interval_records(H, Ecert, Ocert, I))

    return atlas
```

Splitting changes chart/event complexity, not the requested output sampling
density by definition. In an easy bounded program, the same accepted interval
can serve many requested times. Near a pole, support change, order crossing, or
large residual, the interval subdivides or dies into fallback rather than
silently extrapolating.

**Production depth boundary.** The compiler can carry an affine local section

```text
z_i(u,v,t) = z_c,i(t)
           + z_u,i(t) (u-u_c,i(t))
           + z_v,i(t) (v-v_c,i(t)).
```

That field is useful for diagnostics, source gradients, and tighter compiler-
side certificates. When image/tile dimensions are supplied, the current
fallback marker bounds this affine section over a tile and flags a UV depth-line
event. After certification, however, the interval Metal compositor consumes
one precompiled order per tile-time cell; it does not perform arbitrary pixel-
varying live sorting. A cell for which one cell order cannot be certified must
be split or routed to fallback. The implemented claim is therefore affine
tile-depth certification followed by scalar cell ordering, not a general
per-pixel visibility solver.

![Projective compiler. Homogeneous camera traces remain undivided while the
compiler certifies the projective denominator, trace approximation, support,
and visibility on a bounded interval. A certified interval is lowered with a
precompiled cell order; an unresolved interval is deterministically split or,
at the minimum interval, sent to an explicit reason-labelled fallback. The
current continuous certificate covers the quadratic denominator, while general
UV/depth residuals use the declared bounded sample set.](research_notes/gauged_uvt_trace_atlas/paper/figures/world_tubes_projective_compiler.svg)

### 3.5 Trace atlas representation

The compiled atlas is:

```text
K_Gamma = { C_l, S_l, Phi_l, Pi_l, E_l }_{l=1}^L.
```

where:

```text
C_l       gauge domain / event cell in (u,v,t)
S_l       active primitive set
Phi_l     trace functions: alpha_i,l(y), c_i,l(y), z_i,l(y)
Pi_l      stable total order, partial order, or commutation certificate
E_l       error, support, fallback, and backward metadata
```

Rendering at `y in C_l` evaluates active traces and composites them in the
compiled order:

```text
I(y) = sum_m Trans_m(y) alpha_{pi_m,l}(y) c_{pi_m,l}(y).
```

The transmittance is:

```text
Trans_m(y) = product_{n<m} (1 - alpha_{pi_n,l}(y)).
```

A frame is a slice:

```text
I_k(u,v) = I(u,v,t_k).
```

Finite exposure is an integral through the atlas:

```text
I_k(u,v) = integral w_k(u,v,tau) I(u,v,tau) d tau.
```

Rolling shutter replaces `tau` with a row/time-coupled sensor program.

### 3.6 Visibility gauge atlas

Depth marginalization alone does not make alpha compositing linear in depth.
The footprint trace:

```text
pi_* Gamma^* w_i
```

is sufficient for support and shading, but visibility is a relation between
lifted traces before the ray-depth structure is fully marginalized. We
therefore compile a second object, a **visibility gauge atlas**:

```text
O_Gamma = { C_l, G_l, Delta_l, Pi_l, R_l }.
```

Here `G_l` is a local support-overlap graph; it contains only primitive pairs
whose sensor-time footprints overlap inside the same tile-time cell. `Delta_l`
stores certified depth/order predicates, `Pi_l` stores the induced total order
or partial-order DAG, and `R_l` stores commutation residuals and fallback
metadata. This makes certification output-sensitive to local support overlap
and event complexity instead of constructing irrelevant global pairs. Dense
overlap remains quadratic in the worst case and is measured rather than hidden.

For each primitive we keep conditional depth:

```text
z_hat_i(y),    sigma_z,i(y).
```

With `sigma_z,i = sqrt(nu_z,i)`, a concrete conservative confidence band is:

```text
D_i(y) =
  [z_hat_i(y) - k sigma_z,i - delta_fit,i,
   z_hat_i(y) + k sigma_z,i + delta_fit,i].
```

Here `k` is the declared tail width and `delta_fit,i` bounds camera-chart or
trace approximation error. This is a sufficient order certificate, not a
necessary one. Unlike a mean-depth test, it rejects thick overlapping
conditional fibers even when their means are separated.

For support-overlapping pairs, define either a center-depth difference:

```text
Delta_ij(y) = z_i(y) - z_j(y)
```

or a conservative interval predicate:

```text
Delta_ij^-(y) = z_i^-(y) - z_j^+(y)
Delta_ij^+(y) = z_i^+(y) - z_j^-(y).
```

Then:

```text
if Delta_ij^+(y) < 0:  i definitely in front of j
if Delta_ij^-(y) > 0:  j definitely in front of i
otherwise:             split, commute, or fallback
```

The gauge should be chosen so these depth-difference fields are cheap to
certify. For orbit/projective cameras, ordinary frame time may make depth
curves high-curvature, while projective time, inverse depth, log depth, or
denominator gauges can make them rational, low-degree, or interval-friendly.

If unresolved pairs remain, a general certificate can bound the effect of
swapping two translucent contributors:

```text
|Delta I_ij(y)| <= alpha_i(y) alpha_j(y) |c_i(y) - c_j(y)|.
```

The implemented hybrid currently uses the stricter depth-band separation
certificate rather than this color-commutation residual. In the general
criterion, unresolved pairs below tolerance could be marked commutable;
important unresolved pairs induce an event boundary or fallback. Fallback is
part of the theorem of the implementation: hard regions can be rendered by
local live sorting or a reference path, while the rest of the atlas retains
shared metadata.

The production distinction is important. Mean-depth sorting remains the fast
baseline-compatible path, but the hybrid renderer now consumes the conditional
variance bands above. Certified tiles use the fast Metal compositor; rejected
tiles keep the Gaussian depth profiles and use an integrated retained-fiber
Metal forward/VJP. This closes the static affine-Gaussian fallback path. It
does not yet close the exact projective/nonlinear-camera case: that route still
needs projective retained-depth records, approximation-error propagation, and
a certified quadrature policy.

**Proposition 3 (fixed-cell compositing correctness).** Fix an atlas cell
`C_l`. Assume its active set contains every primitive whose baseline support
intersects the cell, and let `epsilon_trace,l` bound the accumulated image
effect of trace approximation and support padding on that cell. Suppose the
compiled order differs from the baseline live order only through a sequence
`Q_l` of adjacent swaps, where every swapped support-overlapping pair satisfies

```text
sup_{y in C_l}
  alpha_i(y) alpha_j(y) ||c_i(y)-c_j(y)|| <= epsilon_ij.
```

Then the baseline-relative image error is bounded by

```text
sup_{y in C_l} ||I_compiled(y)-I_replay(y)||
  <= epsilon_trace,l + sum_{(i,j) in Q_l} epsilon_ij.
```

In particular, exact traces plus a certified identical order give exact
baseline replay on `C_l`. The proof applies the two-layer swap identity above
to each adjacent transposition and uses the triangle inequality; preceding
transmittance can only reduce the contribution because it lies in `[0,1]`.
Cells that fail the active-set, order, or error assumptions are not covered by
this proposition and must be split or evaluated by the declared fallback.

This proposition is intentionally baseline-relative. It reproduces the chosen
Gaussian-splat compositing semantics. It does not claim that center-depth alpha
compositing is a physically exact solution of radiative transfer. The current
projective production path primarily uses certified scalar/cell order and
fallback; the commutation bound states the admissible extension but is not
silently credited as implemented dense-scene selectivity.

### 3.7 Noncommutation boundary and ordered ray transfer

The visibility gauge atlas preserves baseline Gaussian-splat semantics. A more
radical sibling method retains the lifted ray-fiber opacity field itself:

```text
sigma_l(y,z) = sum_i rho_i(Gamma_l(y,z)).
```

Instead of sorting primitive centers, one renders by Beer-Lambert
transmittance:

```text
Omega_y(z) = integral_{z_front}^{z} sigma_l(y,s) ds
Trans_y(z) = exp(-Omega_y(z))
I(y)       = integral Trans_y(z) sigma_l(y,z) c_l(y,z) dz.
```

The split is algebraic, not merely architectural. One thin alpha/color event
acts on background radiance with:

```text
G_i =
  [ (1-alpha_i) I_C    alpha_i c_i ]
  [ 0                  1           ].
```

For two contributors:

```text
[G_i,G_j]_color = alpha_i alpha_j (c_i-c_j).
```

Thus order is irrelevant only when one contributor is transparent, their
colors agree, or a declared residual tolerance makes the swap immaterial.
More strongly, total opacity, constant color, and one representative depth do
not determine an extended colored depth profile: two primitives can have the
same summaries while interleaving their optical mass differently and producing
different images. This is why confidence-band overlap rejected by Section 3.6
cannot be repaired in general by sorting conditional means.

**World Tubes + Ordered Ray Transfer** resolves that scoped Gaussian case by
retaining the ambiguous ray-depth fiber and evaluating the ordered optical
transfer:

```text
A_y(z) =
  [ -sigma_y(z) I_C    sigma_y(z)c_y(z) ]
  [ 0                  0                 ]

M_y = P exp integral A_y(z) dz.
```

The product integral is invariant to an orientation-preserving change of the
ray-depth coordinate when the physical-length Jacobian is included. Its
piecewise-constant implementation is an associative visibility-monoid scan.
The current World Tubes extension is a bounded static-affine fallback for
native Gaussian depth fibers. Selectivity has been demonstrated only on the
small static-affine fixture; the dense fixture falls back everywhere.
**WorldFoam** is the broader, separate representation contract for general
cellular fields, cell/event words, and richer finite-element material laws;
the scoped Gaussian fallback is not a claim that WorldFoam has been absorbed
into STAR UVT.

Material-basis selection for general cellular fields is evaluated in the
separate WorldFoam study and is outside the scope of this paper.

### 3.8 Compiled adjoints

Inside a gauge domain with fixed support and visibility metadata, rendering is
differentiable with respect to trace parameters. For a primitive `i`, the
local derivative of compositing is:

```text
dI/dc_i     = Trans_i alpha_i,
dI/dalpha_i = Trans_i (c_i - I_behind,i).
```

The gradient of the loss is:

```text
dL/dtheta_i =
  sum_l integral_{C_l} lambda(y)^T dI(y)/dtheta_i dy,
```

where `lambda(y) = dL/dI(y)` is the image adjoint. The compiled implementation
uses interval Metal forward and direct VJP with topology, active intervals,
and visibility cells held as compiled constants. Trace coefficients, opacity,
temporal opacity, spatial precision, and colors remain differentiable.

More explicitly, let `phi=C_Gamma(theta;kappa)` denote the differentiable atlas
coefficients emitted from world parameters `theta`, and let the event-cell
topology and replay tape be `kappa`. For an image residual
`lambda(y)=dL/dI(y)`, the backward chain is:

```text
image residual lambda
  -> g_phi = D_phi R(phi, kappa)^T lambda
  -> g_theta = D_theta C_Gamma(theta; kappa)^T g_phi
  -> gradients of world means, SPD(4) factors, opacity, and appearance.
```

The first arrow is the interval evaluator VJP. The second accumulates trace
coefficient gradients through gauge transforms, projective/local trace
coefficients, and Schur-complement lowering into the shared world source.
Cell topology, order decisions, and fallback choices are piecewise-constant
during this VJP; event-boundary derivatives require a separate estimator and
are not silently included. An implementation that stops at `g_phi`, or
optimizes a detached atlas directly, is an atlas-fitting baseline rather than
the claimed compiled-world adjoint.

**Proposition 4 (fixed-topology compiled adjoint).** Fix `kappa` and suppose
`C_Gamma(theta;kappa)` and `R(phi,kappa)` are differentiable in a neighborhood
of the current world parameters. For

```text
L(theta;kappa) = ell(R(C_Gamma(theta;kappa),kappa), I_star),
```

the world gradient is

```text
grad_theta L
  = D_theta C_Gamma(theta;kappa)^T
    D_phi R(phi,kappa)^T
    grad_I ell.
```

This is the ordinary chain rule for the compiler/evaluator composition. It is
valid within a structural stratum where perturbing `theta` does not change the
active support, interval split, tile membership, order, or fallback decision.
At a structural boundary, the implementation must recompile or use a separate
boundary estimator; Proposition 4 does not assign a derivative to the discrete
change in `kappa`.

### 3.9 Work model and claim boundary

Let `F` be the number of requested times, `P=H W` the pixels per time, `N` the
world primitive count, `L` the number of accepted chart/event cells, `N_tr` the
compiled primitive-trace records across those cells, `B_int` the interval-bin
entries, and `K` the materialized primitive-pixel interactions. A useful work
decomposition is

```text
W_replay(F)
  = sum_{f=1}^F [W_project(f) + W_support(f)
                 + W_bin(f) + W_visibility(f)]
    + W_shade(K),

W_compiled(F)
  = W_compile(N,L,N_tr,B_int)
    + W_trace_eval(F,N_tr,B_int)
    + W_shade(K).
```

A coarse dense-world accounting isolates the persistent metadata terms as

```text
W_meta,replay   = O(F N + B_replay),
W_meta,compiled = O(N_tr + B_int),       with N_tr <= N L.
```

These are accounting identities for the declared records, not lower bounds;
spatial culling can reduce both routes.

Both routes must write `F P` output samples, and both pay for the actual
shading/compositing interactions represented by `K`. World Tubes targets the
first bracket in `W_replay`: projection, conservative support, tile/bin
membership, stable visibility metadata, and the corresponding world-to-trace
backward tape. `W_trace_eval` denotes evaluating the already compiled trace and
interval gates at requested times, excluding the separately counted
primitive-pixel shading/compositing work. For a fixed bounded camera program
whose chart/event structure does not grow when time sampling is densified,
`L`, `N_tr`, and the persistent
interval metadata can remain fixed while the replay metadata grows with `F`.
This is the regime measured by the fixed-chart structural experiment.

No universal asymptotic statement follows without an assumption on event
complexity. A near-plane singularity, rapid support churn, dense order
crossings, or conservative fallback can make `L`, `N_tr`, `B_int`, or live replay
work grow proportionally to `F`. The compiled evaluator also still samples the
trace and accumulates image residuals at the requested times. Accordingly, our
claim is conditional frame amortization of world-side work, not sublinear
materialization of images and not end-to-end sublinear training through
topology changes.

For timing, compile cost must be reported separately. If `c_replay` is the
median per-time replay cost and `c_eval` the median per-time atlas-evaluation
cost under the same frozen world, the idealized amortization point is

```text
F_break_even = W_compile / (c_replay - c_eval),
```

defined only when `c_replay > c_eval`. The publication experiment measures the
complete warmed, repeated route rather than treating this formula or logical
tensor volume as a timing, storage, or peak-memory result.

## 4. Implementation

The implementation retains the historical STAR UVT / projective interval
backend and adds a parallel native-SPD(4) source plus physical transfer paths.
The production experiment surface has four explicit axes:

| Axis | Implemented values |
|---|---|
| World source | `legacy_tube`, `full_spd4` |
| Renderer | `dense`, `metal_tile`, `retained_fiber_metal`, `hybrid_retained_fiber` |
| Alpha law | `peak_splat`, `beer_lambert` |
| Amplitude convention | `fiber_integrated`, `peak_density` |

`legacy_tube` remains the default and has 10 geometry / 14 total trainable
scalars per atom. `full_spd4` is opt-in and has 14 geometry / 18 total
scalars. Its lossless chart stores a spacetime mean, a conditional spatial
Cholesky factor, a space-time tilt, and a positive temporal precision. Static
cameras use the exact affine-gauge pushforward. `dynamic_first_order` and
`projective_first_order` use the same differentiable one-chart first-order
moving-camera compiler; this compiler matches the camera-program value and
Jacobian at the chart point but is not an exact long-window nonlinear
projection.

Both sources lower into the established q-UVT footprint and affine
conditional-depth fields. The full-SPD(4) lowering additionally retains
conditional-depth variance and the declared amplitude semantics. The fast
`metal_tile` path consumes the footprint and affine mean depth. The
`hybrid_retained_fiber` path also consumes conditional-depth variance: a
tile-level confidence-band certificate selects either the fast compositor or
the retained-depth fallback. `retained_fiber_metal` runs the physical
retained-depth transfer everywhere and serves as the integrated oracle for the
hybrid path.

`peak_splat` preserves the historical
`opacity exp(-q/2)` behavior. `beer_lambert` implements
`1-exp[-tau_0 exp(-q/2)]`, its exact VJP, and its alpha-cutoff support
equation in the Metal kernel. `fiber_integrated` treats the trainable amplitude
as projected peak optical thickness. For `full_spd4 + beer_lambert`,
`peak_density` instead compiles world peak density through the affine
fiber-measure factor and conditional variance. Invalid combinations fail
closed: the legacy source accepts only `fiber_integrated`, `peak_density`
requires `full_spd4 + beer_lambert`, and retained-fiber renderers require
Beer--Lambert semantics.

The retained path evaluates the combined Gaussian extinction and emission
field along depth, performs front-to-back Beer--Lambert transfer, and provides
a native Metal VJP for q-UVT mean/precision, conditional depth mean/slope and
variance, optical thickness, and color. The production entry points are
`render_retained_fiber_metal(...)` and
`render_variance_certified_hybrid_metal(...)`. They are called from the same
multicamera trainer and unified paper runner as the fast STAR path.

The primary projective interval backend stores
homogeneous/projective time coefficients, opacity, optional temporal opacity
coefficients, optional spatial precision, optional depth-affine terms, and
color. The affine depth-plane terms can be carried through the trace ABI, but
they are used compiler-side to bound tile depth and flag UV depth-line events.
The production Metal compositor then consumes the accepted scalar cell order
instead of solving a pixel-varying order field. Tile-time cells store active
intervals and visibility metadata. The hot
path packs accepted cells once into spatial tile bins and uses per-entry
`[active_start, active_stop)` checks in the Metal kernel. Its forward path is:

```text
render_projective_trace_cell_interval_atlas_metal(...)
```

The direct VJP path is:

```text
direct_backward_projective_trace_cell_interval_atlas_metal(...)
```

The trainer route calls:

```text
_render_projective_interval_feature_tubes_autograd(...)
```

via a custom autograd function:

```text
_ProjectiveCellIntervalBackward
```

The physical retained-depth route is currently integrated with the affine
q-UVT producer, not with the full projective trace family. Current limitations
are therefore:

```text
MPS/Metal backend
RGB / feature_dim=3 route for the projective interval trainer
compiled visibility/order and tile membership held fixed during direct VJP
legacy/restricted producer remains the default; full-SPD(4) is explicit opt-in
moving-camera SPD(4) compilation is first-order within one chart
projective compositor consumes precompiled cell order;
arbitrary per-pixel live sorting is not implemented
exact retained-depth transfer for nonlinear/projective traces is not implemented
retained-fiber quadrature is fixed rather than adaptive/error-certified
the present variance certificate can conservatively route every dense tile to fallback
event topology and fallback choices remain fixed during their VJPs
STAR UVT support/visibility/composition quality is a separate active research lane
```

The strict-SPD(4) reference/compiler, synthetic capacity gate, trainable
producer, and Metal forward/VJP checks now pass. In the
rank-six three-camera capacity fixture, full SPD(4) reaches `1.16e-13` MSE
while the restricted source retains `2.07e-4` MSE from a matched initial loss.
The RGB parameterization uses 18 trainable scalars per full-SPD(4)
atom versus 14 per restricted atom, a `18/14 = 1.2857x` per-atom capacity
mismatch. The fixture therefore isolates expressivity; the bounded experiment
in Section 6 separately includes a 199-atom / 3,582-parameter full-SPD(4) row
against the 256-atom / 3,584-parameter restricted source.

For Beer--Lambert, CPU analytic/autograd/finite-difference tests and native
Metal forward/direct-VJP parity pass, including cutoff and clamp branches.
The retained-fiber Metal gate also passes forward and all source VJPs, and the
hybrid path is exercised through the training seam. The final focused
validation suite reports `143 passed, 4 skipped`. These are implementation and
bounded-mechanical checks. They do not turn the short single-seed rows below
into paper-quality convergence evidence.

## 5. Experiments

The final paper should separate four questions:

1. **Correctness:** Does the trace integral match dense ray/fiber evaluation?
2. **Sublinear scaling:** Does world-side work grow sublinearly with frame
   count or camera-family size?
3. **Renderer equivalence:** Does the compiled route preserve images, losses,
   media, and gradients relative to the baseline renderer?
4. **Usefulness:** Does the speed/memory benefit hold on public dynamic-scene
   workloads without excessive fallback?

### 5.1 Synthetic trace correctness

Controlled scenes of analytic 4D Gaussians with known camera programs:

```text
camera paths: static, dolly, orbit, fast orbit, rolling shutter
FOV: 30, 60, 90, 120 degrees
frames/samples: 4, 8, 16, 32, 64, 128
primitive anisotropy: isotropic, elongated spatial, elongated temporal
visibility: isolated, crossing pair, thin foreground occluder, dense translucent
```

Metrics:

```text
trace L1 / KL against dense fiber integration
image PSNR / SSIM / LPIPS
alpha error
depth-order error
fallback fraction
support over/under-coverage
gradient relative error
```

### 5.2 Frame-count scaling

Compare per-frame replay against compiled atlas for fixed camera paths:

```text
F = 4, 8, 16, 32, 64, 128
```

Metrics:

```text
logical tensor-element bytes (not topology-inclusive storage)
trace count
tile/bin entries
interval entries
CPU compile time
GPU forward time
GPU backward time
total step time
route-scoped sampled peak memory (required, not supplied by logical volume)
```

Current internal evidence:

```text
bounded-orbit F: 4, 8, 16, 32, 64, 128
fixed logical-volume growth: 1.0x vs per-frame replay 32.0x
final fixed/per-frame trace-count ratio: 0.03125
publication timing: pending warmed repeated frozen-world sweep
```

### 5.3 Camera-family scaling

For a low-dimensional camera family `Q x Omega x T`, compare one shared family
atlas against replaying one atlas per camera parameter sample:

```text
Q1: orbit phase offset
Q2: orbit phase + height
grid: 3x3, 5x5, 7x7
```

Current internal evidence:

```text
Q2 shared logical-volume growth: 1.0x
Q2 replay logical-volume growth: 64.0x
Q2 final logical-volume ratio: 0.0625
Q2 final chart ratio: 0.015625
Q2 max UV fit residual: 0.111 px
```

The camera compiler must also emit a diagnostic vector per gauge domain:

```text
minimum projective denominator margin
maximum UV reprojection residual (pixels)
maximum depth-model residual
support-certificate slack / under-coverage count
minimum certified order margin
conditional-depth variance range
chart and event-cell counts
fallback fraction and reason histogram
```

We will sweep orbit span, FOV, camera-to-primitive distance, primitive
anisotropy, and motion magnitude with matched world parameters. Each sweep
compares the affine/local trace closure, the projective trace family, and dense
per-frame replay. The resulting affine-versus-projective **closure/death
curves** plot image/gradient error, chart count, and fallback fraction against
camera nonlinearity. They locate where the affine approximation ceases to meet
its declared tolerance and whether projective lowering extends that range
before certification correctly routes samples to a new cell or fallback.
The bounded runner and verifier are implemented at
`projective_variable_camera_closure_death_curve.py`. It fixes one synthetic
world, physical interval, and sample count; uses exact rational centers with
per-sample live depth ordering as the oracle; and binds chart/event/trace
counts, fallback, image error, and fixed-topology world VJPs. Its runtime
curve remains pending and no boundary value is claimed here.

```{=latex}
\begin{table*}[t]
\centering
\caption{Bounded variable-camera closure/death curve.}
\label{tab:variable-camera-closure}
\input{research_notes/gauged_uvt_trace_atlas/paper/generated/schema_v2/variable_camera_table.tex}
\end{table*}
```

### 5.4 Real-video renderer equivalence

Use source-distinct real videos and compare compiled projective interval
renderer against the cadence/per-frame route.

The publication comparison uses a frozen identical-world replay protocol:

1. Train one declared world representation to checkpoint `theta_star`.
2. Freeze `theta_star`, target camera samples, precision, shading settings,
   loss, background, and pixel batch.
3. Route A lowers that same checkpoint independently for each target frame.
4. Route B compiles that same checkpoint once for the complete camera program
   and evaluates identical targets.
5. Compare images, losses, world-parameter VJPs, payload, and timing; do not
   train separate route-specific worlds.

The lane-isolated runner for this causal protocol is implemented at
`run_frozen_world_replay_compiled.py`. It snapshots and hashes the final
learned world, repeats one-frame STAR projection/bin/render for every selected
heldout target, compiles one event-stratified interval atlas from the same
state, and reports image/loss/world-VJP parity, payload, timing, and fallback.
One invocation can now train/save once and evaluate
`F={4,8,16,32,64,128,full}` from that exact checkpoint. Each `F` uses an
ordered integer time grid spanning the same full physical interval rather than
a growing prefix, and the artifact binds the selected indices and centered
times. Each `F` compiles its own atlas from the same frozen world/program; the
experiment does not claim that one identical atlas object is reused across
different sampling densities.
The frozen report's current payload ratio covers route tensor payload only; it
excludes interval/cell and replay-bin topology, allocator overhead, and
transient working memory, and is explicitly ineligible as a storage or full
interaction-memory claim. Route-scoped peak memory and topology-inclusive
bytes remain required for that stronger comparison.
Its non-unit selected-time chunk parity, warmed/repeated timing, and
publication-scale result are still pending on an approved host. Existing
broad-video and frame-scaling artifacts remain renderer-equivalence and
same-representation evidence at their recorded scopes; this implementation
does not retroactively relabel them as a fully frozen public-checkpoint result.

Metrics:

```text
loss curve delta
RGB loss curve delta
end PSNR delta
contact-sheet pixel delta
PNG hash match
gradient flags present
support rebins
stale refreshes
fallback fraction
```

Current internal evidence:

```text
10 source-distinct broad quality pairs
10 source-distinct broad media pairs
20 broad10 trainer case payloads
all cases projective interval main path
all renderer gradient flags present
zero support rebins / stale refreshes in accepted rows
compiled trainer replacement gap: 0
```

![Qualitative real-video tether for the accepted projective decisive-demo
artifact. Columns compare the target, compiled World Tubes render, and error
views at sampled times; the figure is a correctness/media tether rather than a
public-dataset quality claim.](research_notes/gauged_uvt_trace_atlas/paper/figures/real_video_equivalence.jpg)

### 5.5 Public dataset comparison

The paper separates three kinds of public evidence:

1. **Compiler-causal evaluation.** Freeze one learned World Tubes checkpoint
   and compare per-frame replay with one compiled atlas on identical public
   targets. This is the public evaluation of the paper's compiler
   contribution.

2. **Representation and context evaluation.** Train World Tubes, WorldFoam,
   and dynamic 3DGS under the shared progressive, fixed, and sampler-control
   protocols. These selected-time trainer rows compare quality, cost, and
   stored state; they do not evaluate the compiled projective atlas.

3. **External comparison.** Report published or reproduced dynamic-scene
   baselines as contextual quality and efficiency references, not as
   substitutes for the same-representation causal comparison.

Neural 3D Video is the positive multiview setting. D-NeRF is a labelled
one-frame-per-chart negative/control under the current posed-frame adapter and
is not evidence for bounded-chart sublinear scaling.

### 5.6 Finite exposure and rolling shutter

Compare:

```text
baseline per-shutter-sample rendering
compiled atlas sampled at same shutter samples
compiled atlas with adaptive temporal quadrature
high-sample reference
```

Metrics:

```text
quality vs high-sample reference
unique time samples
logical tensor-volume growth
forward/backward time
rolling row-time correctness
```

### 5.7 Visibility stress tests

Create scenes where the compiler should fail locally:

```text
crossing translucent slabs
thin foreground occluders
near-camera large splats
fast orbit around high-depth-variance geometry
disocclusion boundary
```

Metrics:

```text
fallback fraction
order-strata count
commutation-bound accepted pairs
image error near event boundaries
speed lost to fallback
```

## 6. Tables and figures

One provisional figure is packaged with the manuscript: the verified bounded
real-video equivalence contact sheet above. Schema-v1 public-data figures are
excluded. The artifact generator emits fail-closed schema-v2 public, frozen,
and variable-camera SVGs, but those remain placeholders until their complete
components verify. The remaining submission figures are:

1. **Concept figure:** per-frame dynamic GS replay vs world-tube trace atlas.
2. **Bundle diagram:** `B = Omega x T`, ray fibers, `Gamma`, pullback, pushforward.
3. **Schur complement diagram:** 4D primitive -> `(u,v,t,z)` Gaussian -> depth-marginalized UVT footprint.
4. **Gauge-domain/event diagram:** orbit camera with projective domains split by denominator/support/order events.
5. **System diagram:** compile, atlas, Metal interval forward, direct VJP.
6. **Scaling chart:** frame count vs logical-volume/bin/forward/backward ratio.
7. **Camera-family chart:** Q-grid replay vs shared family atlas.
8. **Fallback stress chart:** fallback fraction vs visibility density.
9. **Closure/death curves:** affine and projective error/fallback versus camera
   nonlinearity, annotated by compiler diagnostics and certification events.

Core tables:

```text
Table 1: synthetic correctness vs dense fiber reference
Table 2: frame-count scaling, internal + public sequences
Table 3: ablation of trace representation and gauge domains
Table 4: training/backward replacement and gradient errors
Table 5: public dynamic-scene comparison
Table 6: finite exposure / rolling shutter
Table 7: limitations and fallback-heavy cases
```

### 6.1 Certified correctness and theorem table

The table is injected from the submission artifact bundle. Its retained source
reports are byte-pinned and the rows are rederived by the Torch-free
submission generator. Bounded-fixture timing rows are forbidden here; speed
belongs to the frozen-world table below. The scope is bounded
event-certified projective chart segments and does not assert an unimplemented
full `360/720` multi-chart transition.

```{=latex}
\begin{table*}[t]
\centering
\caption{Certified bounded correctness and structural trace reuse.}
\label{tab:certified-correctness}
\input{research_notes/gauged_uvt_trace_atlas/paper/generated/schema_v2/theorem_table.tex}
\end{table*}
```

### 6.2 Exact same-representation frame scaling

The verified `F={4,8,16,32,64,128}` fixture compares per-frame STAR replay
with the compiled projective atlas at identical representation settings.
Fixed logical tensor-element volume growth is `1.0x` while replay grows
`32.0x`; at `F=128`, the fixed/per-frame trace-count ratio is `0.03125`. The
fixture's historical single-shot timings are diagnostics and are excluded
from the submission timing claim. Logical-volume accounting double-counts
shared replay tensors and excludes topology, packed bins, and transients, so
it is not a storage or peak-memory claim.

The submission timing table below is generated only from a
publication-eligible frozen-checkpoint sweep with raw samples, at least one
warmup, and at least three repeats. Until that artifact exists it renders an
explicit non-submission-ready placeholder.

```{=latex}
\begin{table*}[t]
\centering
\caption{Frozen identical-world replay versus compiled-atlas scaling.}
\label{tab:frozen-world-scaling}
\input{research_notes/gauged_uvt_trace_atlas/paper/generated/schema_v2/frozen_scaling_table.tex}
\end{table*}
```

This experiment tests whether replay equivalence and reuse survive a learned
real scene. The three-lane public training matrix uses selected-time rendering
and is reported only as representation-and-cost context; it is not compiler
evidence.

### 6.3 Bounded SPD(4), alpha-law, and fallback integration

A single-seed, 16-frame, 40-step Coffee Martini fixture exercises the
production selector end to end. The parameter-matched rows differ by only two
trainable scalars. Times are synchronized training wall times, and driver
memory is the sampled peak reported by the isolated run.

| Source / transfer | Atoms | Parameters | Heldout PSNR | Train wall (s) | Peak driver bytes |
|---|---:|---:|---:|---:|---:|
| `legacy_tube + peak_splat` | 256 | 3,584 | `5.9865` | `4.9020` | `63,356,928` |
| `full_spd4 + peak_splat` | 199 | 3,582 | `7.0054` | `4.7512` | `46,596,096` |
| `full_spd4 + beer_lambert + fiber_integrated` | 199 | 3,582 | `7.1333` | `4.6758` | `46,596,096` |

An equal-count 256-atom full-SPD(4) diagnostic reached heldout PSNR `7.0888`,
but its evaluation reported tile overflow. It is retained as a failure-bearing
diagnostic and excluded from matched-quality claims.

The hybrid certificate also has a bounded stress result. With 16 atoms it
routed 10 of 64 tiles to retained-fiber transfer and matched the recorded
all-retained heldout metrics. With 199 atoms it routed 64 of 64 tiles to
fallback. This verifies the integrated branch and its separate native VJP
gate, but it is not an explicit full-image/VJP parity artifact and also shows
that the current dense-scene confidence bands are too conservative to
establish a selective-performance win.

These rows are short, single-seed integration evidence. They do not establish
convergence, scene breadth, or paper-quality superiority. A less conservative
selective certificate and adaptive/error-certified retained quadrature are
future work for this extension, not submission blockers for the central
projective interval-atlas result. The submission still requires the declared
public controls and scene breadth.

### 6.4 Public representation and cost context

The shared progressive, fixed, and global-shuffle protocols, evidence schema,
and matrix generator are implemented, pending focused behavior verification.
These rows train World Tubes with
selected-time STAR rendering, WorldFoam, and dynamic 3DGS under matched
optimizer, target-frame, and target-pixel budgets. They compare learned
representation quality, runtime, and stored state; they do not compare
per-frame replay with the compiled projective atlas. Evidence schema v2 is
source-complete but not yet runtime-verified. It binds exact schedule, raw and
decoded data, canonical evaluation, runtime/native binaries, retained
artifacts, and finalized W&B files. None of the seven core or 21 full-breadth
rows is currently accepted. The canonical evaluator clamps predictions to
L1/MSE over all RGB elements, derives PSNR once from the global MSE, averages
SSIM/LPIPS over the full declared image set, and uses a fixed black background
with no color calibration. The seven core rows must be rerun for the minimum
paper cut; the remaining 14 are breadth targets. Storage and parameter counts
are not matched: World Tubes shares temporal trace state, while dynamic 3DGS
and WorldFoam retain substantial per-frame state. The table below is generated
only when all seven schema-v2 controls pass. It emits no partial numeric rows.

```{=latex}
\begin{table*}[t]
\centering
\caption{Public representation and cost context under the declared schema-v2 controls.}
\label{tab:public-context}
\input{research_notes/gauged_uvt_trace_atlas/paper/generated/schema_v2/public_context_table.tex}
\end{table*}
```

The causal public compiler experiment is specified separately through the
frozen identical-world protocol in Section 5.4 and remains pending. It uses a
static heldout Neural3D camera; bounded moving-camera scaling is currently
synthetic.

Schema-v1 numbers and their plot are intentionally absent from the manuscript.
Schema-v2 reruns of the progressive rows, pixel-matched fixed control, and
global-shuffle control form the seven-row minimum public context table.
Additional camera triplets, two additional Neural3D scenes, the controlled
D-NeRF row, and the deterministic timing audit are the stronger 21-row breadth
target, not blockers for the narrow compiler claim.

The next fixed-512 run was killed after severe unified-memory compression and
swap pressure destabilized the local workstation. Its partial outputs are
excluded. The runner now isolates representations in child processes and is
fail-closed on local MPS, but full-scale local execution remains unauthorized
until targets/rays/evaluation are streamed or a larger machine is used.

## 7. Discussion and limitations

The method is not a universal replacement for dynamic Gaussian scene
representations. It is a renderer/compiler layer for known or low-dimensional
camera programs. It is most useful when many temporal samples reuse a smooth
camera path or path family.

It is also not an information-theoretic claim that total work is sublinear in
materialized output pixels. The sharper claim is empirical and architectural:
in fixed-topology regimes where a dynamic Gaussian training step is dominated
by world-side
projection, support, binning, visibility, and backward replay, compiling those
events into world tubes is designed to make the dominant bottlenecks scale
with trace/event complexity instead of frame count. The structural reuse is
verified, while the warmed repeated frozen-world runtime result remains
pending. We do not yet claim sublinear end-to-end training growth under
structural invalidation and recompilation.

Failure modes:

```text
fallback-heavy visibility chaos can erase speedups
very wide FOV / near-camera splats require more gauge domains
single random novel views do not amortize compile cost
current implementation is RGB/MPS/STAR-UVT scoped
first-order moving-camera charts require bounded approximation error
fixed retained-fiber quadrature lacks an adaptive error certificate
dense scenes can make the current variance certificate fall back everywhere
current broad real-video evidence proves renderer equivalence, not general SOTA quality
general cellular WorldFoam fields remain a different representation contract
```

The main research claim is narrower and stronger:

```text
known camera programs expose repeated world-side rendering work;
camera-gauged world tubes compile that work into reusable sensor-time traces.
```

The hierarchy is:

```text
World Tubes:
  sublinear camera-path compilation for dynamic Gaussian sources, with
  selectable peak-splat or Beer--Lambert transfer.

Visibility Gauge Atlas:
  certified depth/order compilation; accepted tiles use fast compositing and
  rejected affine-Gaussian tiles can use retained-fiber transfer.

World Foam:
  general cellular opacity/transmittance fields and finite-element materials,
  beyond the Gaussian retained-fiber fallback implemented here.
```

## 8. Conclusion

We presented World Tubes in Gauged Camera Space, a trace-atlas renderer for
dynamic Gaussian splatting. By defining rendering as a camera-ray bundle
pushforward, deriving local Schur-complement UVT footprints, and compiling
event-certified gauge domains into interval tile-time metadata, the method
shares projection, support, binning, visibility, and backward work across time.
The implementation now includes parallel restricted and native-SPD(4)
sources, static affine and first-order moving-camera compilation, selectable
peak-splat and Beer--Lambert laws, explicit amplitude conventions, and a
bounded retained-fiber extension. It also demonstrates cross-frame structural
state reuse and a compiled-adjoint training route on STAR UVT/projective
interval traces. The remaining submission work is evidence closure: the frozen
fixed-interval same-checkpoint sweep, one bounded variable-camera
closure/death curve, the seven schema-v2 Coffee Martini control rows, and final
citation/figure/table packaging. The remaining 14 public rows are a
post-minimum breadth target. Exact
retained-depth transfer for nonlinear/projective traces, adaptive quadrature,
and dense-scene certificate calibration remain follow-up work rather than
requirements for the central claim.

## References

::: {#refs}
:::
