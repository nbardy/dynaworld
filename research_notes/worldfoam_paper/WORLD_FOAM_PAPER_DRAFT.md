# WorldFoam in Gauged Camera Space:
# Ray-Fiber Transmittance Fields for Dynamic Rendering

Draft date: 2026-07-05

Status: second-paper working draft. This is a theory/prototype manuscript
skeleton, not a final quality claim. It should be read after the World Tubes
paper draft because it deliberately branches away from baseline-compatible
Gaussian-splat compositing into a lifted opacity/transmittance representation.

## Abstract

Dynamic Gaussian splatting renders a scene by projecting primitives to the
image plane, assigning them to tiles, sorting or approximating visibility, and
alpha compositing the resulting screen-space contributors. This design is fast,
but its visibility model is discrete: primitive order must be resolved, order
changes cause discontinuities, and repeated renders from a known camera path
repeat cell/projection/intersection work that is coherent across time.

We introduce **WorldFoam in Gauged Camera Space**, a camera-compiled
ray-fiber transmittance representation for dynamic rendering. Instead of
rendering a frame as a sorted list of splats, we represent world matter as a
bounded cell complex with local opacity and radiance fields. A camera program
pulls each world cell back to a ray bundle, producing a lifted foam density
over sensor time and ray depth. Rendering evaluates a Beer-Lambert
transmittance integral along the ray fiber:

```text
I(y) = integral T(y,z) sigma(y,z) c(y,z) dz,
T(y,z) = exp(- integral_{z_front}^{z} sigma(y,s) ds).
```

The core computational object is not a video and not a per-frame splat list. It
is a **camera-gauged world foam atlas**: a collection of local domains storing
cell-ray intersection structure, lifted opacity/color bases, transmittance
prefix data, support certificates, and adjoint accumulation rules over
`(u,v,t,z)`. Frames are slices or exposure integrals through the atlas.
Equivalently, the atlas factors a gauge-covariant optical-transfer product
through a compiled cell-path word: each ray induces a front-to-back sequence of
cell/event runs, each run maps to a visibility-monoid element `(beta,m)`, and
the rendered color is the decoded monoid product.

WorldFoam is related to, but distinct from, World Tubes. World Tubes keeps
Gaussian-splat semantics and compiles primitive footprints plus certified
order. WorldFoam instead lifts visibility into a volumetric opacity field where
depth order is replaced by cumulative optical depth. This makes it attractive
for translucent ambiguity, finite exposure, rolling shutter, and fast camera
paths, but also makes it a more radical model requiring its own quality and
parity evidence.

Our current implementation evidence supports the prototype direction rather
than a completed SOTA claim: Metal Gate4/native-cutwalk microgates show
favorable frame-count scaling and speed against matched STAR UVT comparators,
and real32 loader smokes show trainability, but public benchmark quality,
official CUDA/Warp parity, and clean heldout acceptance remain open. This paper
therefore proposes the mathematical and experimental program for WorldFoam and
separates it from the stronger baseline-compatible World Tubes paper.

## 1. Introduction

The central problem is visibility. Dynamic Gaussian methods can make geometry
and motion cheap enough to render in real time, but visibility is still usually
handled as a sorted set of discrete contributors. Even when sorting is
optimized or approximated, the renderer must repeatedly answer a question of
the form:

```text
which primitives affect this pixel/time sample, and in what order?
```

This is awkward under fast camera motion, finite exposure, rolling shutter,
and translucent crossings. It is also computationally wasteful when the camera
program is known and many nearby frames or shutter samples are rendered.

World Tubes answer this by compiling dynamic Gaussian primitives through the
camera program into reusable sensor-time traces, then preserving Gaussian-splat
visibility through a lifted order atlas. WorldFoam asks a more aggressive
question:

```text
Can visibility be represented as ray-fiber optical depth rather than as a
discrete primitive order?
```

The answer is yes mathematically. Let `y = (u,v,t)` be a sensor-time sample and
`z` be a ray-fiber coordinate. If world matter induces a nonnegative opacity
density `sigma(y,z)` and a radiance/color field `c(y,z)` in the camera gauge,
then the rendered color is:

```text
tau(y,z) = integral_{z_front}^{z} sigma(y,s) ds
T(y,z)   = exp(-tau(y,z))
I(y)     = integral T(y,z) sigma(y,z) c(y,z) dz.
```

No global primitive sort appears in this equation. Occlusion is expressed by
the prefix integral `tau`. A discrete sorted splat renderer can be seen as a
quadrature/atomization of this integral. WorldFoam chooses to make the lifted
opacity field the primary representation.

The paper's goal is not to claim that output pixels vanish. The output still
has `O(F H W)` samples if we render `F` frames. The goal is to make the
dominant world-side work scale with **cell/intersection/event complexity** over
the camera program rather than with frame count:

```text
per-frame replay:
    F * (cell/ray intersection + active set + prefix/order + backward replay)

camera-compiled foam:
    compile(cell/ray events over B x Z) + FHW * local eval
```

This is the same amortization idea as World Tubes, but the visibility object is
different: a lifted transmittance field instead of a piecewise-stable sort.

### Contributions

1. **Ray-fiber WorldFoam formulation.** We define dynamic matter as a bounded
   cell complex in world spacetime and render by pulling cell densities through
   a camera-ray bundle into a lifted `(u,v,t,z)` opacity field.

2. **Camera-gauged foam atlas.** We define local gauge domains storing
   cell-ray intersection structure, lifted opacity/radiance bases,
   transmittance prefix summaries, and fallback metadata.

3. **Optical-transfer event algebra.** We formulate rendering as a
   depth-ordered product in the associative visibility monoid
   `(beta,m) otimes (beta',m') = (beta beta', m + beta m')`. This makes
   alpha compositing the atomic-measure case, continuous foam the dense case,
   and owner-run event intervals the sparse camera-compiled case.

4. **Cell-path atlas correctness.** We define the implementation-facing
   renderer as a compiled cell/event word evaluated in the visibility monoid.
   The same-representation replay theorem states that if the compiled atlas and
   per-frame replay emit the same certified word and run lengths, then they
   emit the same image.

5. **Commutator visibility criterion.** The optical-transfer commutator shows
   that order error is controlled by opacity overlap times color contrast.
   This gives a principled split/compression signal and unifies WorldFoam
   interval compression with the World Tubes swap bound.

6. **Cell-complex geometry diagnostics.** We connect foam quality to Cech/AABB
   support graphs, witnessed power complexes, gauge connections across radical
   faces, and holonomy diagnostics that test whether cells form coherent world
   matter rather than camera-glued paint.

7. **Prototype evidence and acceptance gates.** We record the current Metal
   Gate4/native-cutwalk speed evidence, but keep quality/parity acceptance
   explicit: real public datasets, matched STAR/GS baselines, official
   CUDA/Warp parity, and clean heldout metrics are still required before a
   full rendering claim.

### Proof Spine

The current cleaned proof scaffold lives in:

```text
research_notes/worldfoam_paper/proofs/depth_fiber_operator_ordering.md
research_notes/worldfoam_paper/WORLD_FOAM_MATH_APPENDIX.md
research_notes/worldfoam_paper/WORLD_FOAM_OPTICAL_TRANSFER_PAPER_PLAN.md
```

The paper should promote these claims in this order:

```text
1. Gauge-invariant lifted opacity pullback: depth/log-depth/projective gauges
   are equivalent only with the fiber-measure Jacobian.

2. Visibility/marginalization non-commutation: a pure UVT opacity/color
   marginal cannot determine all translucent renderings.

3. Lifted WorldFoam opacity: bounded cells pull back to sparse
   sigma(u,v,t,z) event structure rather than dense UVTZ grids.

4. Transmittance-prefix rendering: visibility is cumulative optical depth
   along the retained camera fiber.

5. Visibility monoid: each event is an optical transfer element `(beta,m)`;
   rendering is an associative depth-ordered scan and decodes as
   `I = m + beta I_bg`.

6. Alpha compositing as atomic optical transfer: sorted splat compositing is
   the monoid scan of atomic opacity events.

7. Cell-path word rasterization: a camera ray induces a certified front-to-back
   cell/event word; WorldFoam evaluates that word in the visibility monoid.

8. Same-representation replay equivalence: on certified domains, compiled
   WorldFoam matches per-frame WorldFoam replay when the emitted word and run
   lengths match, up to declared basis/quadrature/support/fallback error.

9. Monoid-scan VJP: direct backward over `(beta,m)` elements gives prefix/suffix
   adjoints, and constant owner-run elements give closed-form `DeltaTau`,
   length, density, and color gradients.

10. Commutator theorem: visibility/order error is controlled by opacity overlap
   times color contrast; this becomes the interval split/compression signal.

11. Fixed-topology VJP boundary: direct VJPs are exact inside fixed compiled
   topology; endpoint/topology changes require refresh, split, smoothing,
   fallback, or future boundary calculus.

12. Event closure versus Schur closure: World Tubes gets Gaussian Schur
    marginalization; WorldFoam gets sparse event intervals plus optical-transfer
    composition.

13. Event-complexity scaling: structural work scales with camera-path event
   records rather than with sampled frame count when event complexity is
   sublinear.
```

### Math Appendix Summary

The appendix-level identity is:

```text
WorldFoam is gauge-covariant optical transfer factored through a compiled
cell-path atlas.
```

In appendix notation:

```text
lambda_y(z) dz = Gamma_y^* dmu
eta_y(z) dz    = Gamma_y^* dnu

A_y(z) =
    [ -lambda_y(z) I_C   eta_y(z) ]
    [ 0                  0        ]

M_y = P exp int A_y(z) dz
```

The practical rasterizer is the cell-path factorization:

```text
G(y) = otimes_{r in w_y} Phi_r(y)
I(y) = m(y) + beta(y) I_bg(y), where G(y) = (beta(y),m(y)).
```

The paper should promote the monoid, product integral, cell-path replay
theorem, direct VJP, and commutator criterion. Hessians, interface flux
geometry, witness flux, feature-gauge transfer, and universal ray-space
transfer remain appendix branches until finite-difference or correlation tests
support them.

## 2. Related Work

**3D Gaussian splatting and dynamic splats.** 3DGS made real-time radiance
field rendering practical with anisotropic splats and visibility-aware
rasterization. Dynamic variants add temporal deformation, persistent motion,
or spacetime primitives. WorldFoam borrows the goal of fast differentiable
rendering but replaces sorted primitive compositing with lifted optical depth.

**World Tubes and camera-path compilation.** World Tubes compile dynamic
Gaussian primitives through a known camera program into reusable sensor-time
traces. They are the compatibility paper: same primitive semantics, bounded
visibility order, compiled adjoints. WorldFoam is the second paper: it keeps
the same camera-bundle language but changes visibility semantics from discrete
order to cumulative opacity.

**Sort-free and blending alternatives.** Recent work also questions classical
alpha blending and sorting. Sort-free Gaussian Splatting uses weighted-sum
rendering to remove explicit sorting and reports speed gains. Gaussian
Blending treats alpha and transmittance as spatially varying distributions
rather than scalar values. These papers are important context: they show the
community is already probing the limitations of sorted scalar-alpha
compositing. WorldFoam differs by making the lifted ray-fiber transmittance
field and camera-path compiler the central object.

**Volumetric rendering.** NeRF-style emission-absorption rendering already
uses the Beer-Lambert equation. WorldFoam should not pretend this equation is
new. The novelty is the compiled bounded-cell representation and GPU raster
strategy: reuse cell/ray event structure across sensor time while retaining a
compact differentiable primitive family.

**Cell complexes and power diagrams.** PowerFoam-like methods use bounded
cells, adjacency graphs, and cell-local material state. Cech/AABB graphs,
regular triangulations, witnessed complexes, and uncertainty-weighted power
distances are natural tools for making these cells coherent under novel
views. In WorldFoam they become part of the representation and its diagnostics,
not just implementation scaffolding.

## 3. Method

### 3.1 Sensor-time base and ray-fiber bundle

Let:

```text
B = Omega x T,        y = (u,v,t)
```

be the sensor-time base. A camera program defines a ray bundle:

```text
pi: E_Gamma -> B,
pi^{-1}(y) = F_y.
```

Each fiber `F_y` is the ray-depth domain for sensor sample `y`. A camera map:

```text
Gamma: E_Gamma -> M,       M = R^3 x R
```

maps ray-bundle points into world spacetime.

Choose a local camera gauge over a domain `C_l subset B`:

```text
chi_l: E_Gamma | C_l -> C_l x Z_l,
chi_l(e) = (y,z).
```

In this gauge, the camera map is:

```text
Gamma_l(y,z) = Gamma(chi_l^{-1}(y,z)).
```

The ray-fiber coordinate `z` may be ordinary depth, inverse depth, log depth,
orbit angle, or a projective parameter. A valid gauge carries a measure
Jacobian:

```text
dmu_y(e) = J_l(y,z) dz.
```

The physical integral is gauge invariant only if this Jacobian is included.

### 3.2 WorldFoam cells

A WorldFoam scene is a dynamic bounded cell complex:

```text
W = { F_j, theta_j }_{j=1}^N.
```

Each cell has a support region in world spacetime:

```text
F_j subset M
```

and local fields:

```text
sigma_j(x; theta_j) >= 0       opacity density
c_j(x, omega; theta_j)         radiance/color/features
```

For a simple bounded power cell:

```text
cell j:
    p_j(t) in R^3          center
    r_j(t) > 0             radius / power weight
    R_j(t) in SO(3)        local frame
    material_j             density/color basis

B_j(t) = { x : ||x - p_j(t)|| <= r_j(t)
                and pow_j(x,t) <= pow_k(x,t) for k in N(j) }
```

with power distance:

```text
pow_j(x,t) = ||x - p_j(t)||^2 - r_j(t)^2.
```

More generally, `F_j` may be an ellipsoid, slab, tetrahedral/power cell, or
learned bounded support with conservative intersection bounds.

### 3.3 Pullback to lifted camera foam

The camera pulls cell density into the ray bundle:

```text
rho_j^Gamma(e) = sigma_j(Gamma(e)).
```

In gauge coordinates:

```text
rho_{j,l}(y,z) =
    1_{Gamma_l(y,z) in F_j}
    sigma_j(Gamma_l(y,z))
    J_l(y,z).
```

The lifted foam density on a gauge domain is the sum:

```text
sigma_l(y,z) = sum_{j in A_l(y,z)} rho_{j,l}(y,z).
```

The lifted color numerator is:

```text
q_l(y,z) = sum_j rho_{j,l}(y,z) c_j(Gamma_l(y,z)).
```

When `sigma_l(y,z) > 0`, the local premultiplied color field can be written:

```text
c_l(y,z) = q_l(y,z) / sigma_l(y,z).
```

This makes WorldFoam a lifted opacity/color field over `(u,v,t,z)`.

### 3.4 Rendering by transmittance

For a sensor sample `y`, define optical depth:

```text
tau_l(y,z) = integral_{z_front}^{z} sigma_l(y,s) ds.
```

Transmittance:

```text
T_l(y,z) = exp(-tau_l(y,z)).
```

Rendered color:

```text
I(y) = integral_{Z_l} T_l(y,z) sigma_l(y,z) c_l(y,z) dz.
```

Rendered alpha:

```text
alpha(y) = 1 - exp(- integral_{Z_l} sigma_l(y,z) dz).
```

This is the key semantic difference from sorted splatting. A discrete
front-to-back alpha compositor is recovered by approximating `sigma_l` with
depth-localized atoms and applying a quadrature rule. WorldFoam keeps the
continuous/layered optical-depth object as primary.

### 3.5 Foam atlas

A compiled WorldFoam atlas is:

```text
K_Foam = { C_l, Z_l, A_l, H_l, S_l, P_l, E_l }_{l=1}^L.
```

where:

```text
C_l     sensor-time gauge domain in (u,v,t)
Z_l     ray-fiber interval or depth-layer partition
A_l     active cells and cell-ray intersection records
H_l     local lifted opacity/radiance bases
S_l     support/intersection certificates
P_l     transmittance prefix summaries or depth-layer prefix scans
E_l     error/fallback metadata
```

The atlas is accepted on a domain if:

```text
trace/intersection error <= epsilon_trace
opacity-basis error      <= epsilon_sigma
prefix/transmittance err <= epsilon_T
color-basis error        <= epsilon_c
fallback fraction        <= budget
```

Otherwise the domain splits in sensor-time, depth, or cell-support space.

### 3.6 Local basis choices

A foam domain may use several lifted bases:

**Depth layers**

```text
sigma_l(y,z) ~= sum_{k=1}^K sigma_{lk}(y) 1_{z in [z_k,z_{k+1})}
c_l(y,z)     ~= c_{lk}(y) in each layer
```

This is GPU-friendly and close to a k-buffer, but with prefix optical depth
rather than pairwise primitive sort.

**Gaussian/radial depth bases**

```text
sigma_l(y,z) ~= sum_k a_k(y) exp[-0.5 (z-mu_k(y))^2 / s_k(y)^2]
```

This handles soft translucent slabs and crossing semi-transparent matter.

**Cell-intersection events**

For bounded cells, the ray/cell interval is:

```text
J_{j,l}(y) = [z^-_{j,l}(y), z^+_{j,l}(y)].
```

The cell contributes:

```text
tau_{j,l}(y) = integral_{z^-}^{z^+}
    sigma_j(Gamma_l(y,z)) J_l(y,z) dz.
```

For constant density inside the cell, this is density times chord length in
the physical measure. For low-order local density, it is a low-order
quadrature or analytic moment.

### 3.7 Cell-complex gauge connection

Foam quality depends on whether cells form stable world matter or merely
screen-paint source views. Let each cell have a local frame:

```text
F_i(xi) = p_i + r_i R_i xi,
R_i in SO(3).
```

For adjacent cells `i,j`, the radical face has normal:

```text
n_ij = p_j - p_i.
```

A face-aware transport compares tangent frames across the face. Let:

```text
f_ij = normalize(n_ij)
P_ij = I - f_ij f_ij^T
T_i  = orthonormal_basis(P_ij R_i[:,0:2])
T_j  = orthonormal_basis(P_ij R_j[:,0:2])
U_ij = T_j^T T_i.
```

For a loop `C = (i0,i1,...,ik=i0)`, define holonomy:

```text
H_C = U_{i0,i1} U_{i1,i2} ... U_{i_{k-1},i_k}
curv(C) = ||log(H_C)||^2.
```

This is not a mandatory loss. It is first a diagnostic:

```text
Do high-holonomy, well-witnessed cell neighborhoods correlate with heldout
residuals or topology churn?
```

If yes, it becomes a candidate regularizer. If not, it stays out of the model.

### 3.8 Witnessed power complex

Cech/AABB adjacency is a fast conservative support graph:

```text
E_cech = {(i,j): ||p_i - p_j|| <= r_i + r_j}.
```

But an edge can exist geometrically without being supported by training rays.
Define train-only witnesses `w in W_train` and soft power ownership:

```text
q_i(w) = softmax_i(-pow_i^u(w) / tau_w).
```

A witnessed edge score is:

```text
S_ij = sum_{w in W_train} rho(w) q_i(w) q_j(w) g_ij(w),
```

where `g_ij(w)` gates witnesses near the radical slab between cells.

This gives a topology ledger:

```text
witnessed_true        Cech edge with high S_ij
unwitnessed_candidate Cech edge with low S_ij
unstable              edge whose S_ij crosses threshold repeatedly
```

Again, the first use is diagnostic. A paper-quality claim requires showing
that witnessed topology predicts or improves heldout behavior without looking
at heldout RGB.

### 3.9 Differentiation

Within a gauge domain with fixed cell-intersection structure and fixed
depth-layer topology, the renderer is differentiable.

For:

```text
I(y) = integral T(y,z) sigma(y,z) c(y,z) dz,
T(y,z) = exp(- integral_{z_front}^{z} sigma(y,s) ds),
```

the local variation is:

```text
delta I(y)
  = integral T sigma delta c dz
  + integral T (c(z) - I_behind(y,z)) delta sigma(z) dz,
```

where:

```text
I_behind(y,z) =
    integral_{z}^{z_back}
        exp(- integral_z^r sigma(y,s) ds)
        sigma(y,r)c(y,r) dr.
```

This is the continuous analog of the front-to-back alpha-gradient identity:

```text
dI/d alpha_i = T_i (c_i - I_behind_i).
```

A compiled backward pass stores or recomputes:

```text
front transmittance prefix
behind radiance suffix
local basis derivatives
cell-parameter Jacobians
```

The gradient for cell parameters `theta_j` is:

```text
dL/d theta_j =
    sum_l integral_{C_l x Z_l}
        A(y)^T [
            dI/d sigma_l(y,z) * d sigma_l(y,z)/d theta_j
          + dI/d c_l(y,z)     * d c_l(y,z)/d theta_j
        ] dy dz.
```

This is one of the strongest reasons to keep the lifted foam explicit: forward
and backward share the same prefix/suffix transmittance structure across time.

## 4. Implementation Sketch

### 4.1 GPU data structures

```text
FoamTile {
    bounds_u_v_t
    depth_interval_or_layer_range
    active_cell_offset
    active_cell_count
    opacity_basis_offset
    prefix_record_offset
    fallback_flag
    error_budget
}

CellTrace {
    cell_id
    intersection_model
    z_enter_coeffs
    z_exit_coeffs
    density_basis
    color_basis
    jacobian_basis
    validity_error
}

PrefixRecord {
    layer_count
    tau_prefix_offset
    transmittance_prefix_offset
    optional_suffix_radiance_offset
}
```

### 4.2 Forward kernel

```text
for each output sample y:
    tile = locate_foam_tile(y)

    if tile.fallback:
        return local_reference_render(y)

    load active cell traces / depth layers
    evaluate sigma_k(y), c_k(y), dz_k(y)

    tau = 0
    I = 0

    for k in front_to_back_depth_layers:
        alpha_k = 1 - exp(-sigma_k * dz_k)
        T = exp(-tau)
        I += T * alpha_k * c_k
        tau += sigma_k * dz_k

        if exp(-tau) < transmittance_cutoff:
            break
```

This loop is not sorting primitives per frame. It is scanning a compiled depth
or cell-intersection structure.

### 4.3 Backward kernel

```text
for each output sample y:
    replay/evaluate layer sigma, color, alpha
    compute front transmittance prefix
    compute behind-radiance suffix
    accumulate gradients into layer/cell basis parameters
```

The important implementation question is tape versus recompute:

```text
large tape:
    faster backward, bad memory scaling

recompute prefix/suffix:
    lower memory, more ALU

compact scalar prefix/weight tape:
    likely sweet spot if it avoids per-channel storage
```

Current local lessons from STAR/WorldFoam suggest avoiding per-channel tapes
and avoiding duplicate traversal unless the fused loss makes the recompute
floor acceptable.

## 5. Current Evidence

This section is intentionally conservative.

### 5.1 Positive prototype evidence

Internal Gate4/native-cutwalk rows show that a foam-like Metal route can be
fast in focused settings:

```text
WorldFoam 2/4/8/16f mean total:
    3.008 / 3.014 / 3.323 / 4.095 ms

WorldFoam 2/4/8/16f backward:
    2.739 / 2.517 / 2.561 / 3.796 ms

Total/backward scale over 8x frame increase:
    1.361x / 1.386x
```

Matched STAR UVT in the same microgate:

```text
STAR total:
    5.003 / 5.943 / 8.092 / 9.794 ms

STAR backward:
    2.629 / 3.411 / 5.083 / 6.768 ms
```

A repeated-fixture `2/4/8/16/32f` speed smoke also showed:

```text
WorldFoam total:
    2.829 / 3.248 / 4.414 / 4.643 / 6.371 ms

WorldFoam backward:
    2.557 / 2.965 / 4.054 / 4.254 / 6.001 ms

Scale over requested 2 -> 32f:
    2.252x total / 2.347x backward
```

This is useful speed-scaling evidence, but the 32f row repeats loaded frames
and should not be treated as a real 32f quality benchmark.

The real32 loader smoke is valuable as a data/optimizer gate:

```text
loaded_frame_count = 32
no repeats
loss decrease
nonzero gradient
parameter update
```

It is not a quality benchmark.

### 5.2 Current weak points

The quality bridge is not solved:

```text
best current WorldFoam train PSNR:   about 12.248
best current WorldFoam heldout PSNR: about 12.857
gap to solid same-source baseline:   about 9.112 dB
gap to STAR UVT source route:        about 17.575 dB
```

PowerFoam DeepView-style rows remain below paper acceptance:

```text
heldout PSNR around 12.5-12.7
SSIM around 0.10-0.13
acceptance target roughly PSNR >= 13 and SSIM >= 0.15
```

Official CUDA/Warp parity and fixture acceptance remain incomplete. Therefore
the current paper should not claim full replacement of Gaussian splatting or
state-of-the-art dynamic novel-view quality.

## 6. Experiments Required for a Paper

### E1. Synthetic transmittance correctness

Scenes:

```text
single constant-density sphere
two crossing translucent slabs
thin foreground occluder
dense semi-transparent cloud
moving cell complex under fast orbit
finite-exposure motion blur
rolling-shutter camera motion
```

Compare:

```text
dense raymarch reference
per-frame cell rendering
compiled WorldFoam atlas
sorted splat approximation
World Tubes compatibility mode
```

Metrics:

```text
RGB PSNR / SSIM / LPIPS
alpha error
transmittance L1 / L_inf
optical-depth error
cell-intersection false positive / false negative rate
fallback fraction
```

### E2. Frame-count scaling

Sweep:

```text
F = 2, 4, 8, 16, 32, 64
```

Measure:

```text
compile time
forward time
backward time
total optimizer step
cell intersection records
prefix records
GPU memory
quality delta vs per-frame reference
```

Pass condition:

```text
world-side cell/intersection/prefix work grows sublinearly with F,
while output quality stays within tolerance.
```

### E3. Sort pathology benchmark

Use crossing translucent slabs/Gaussian sheets. Compare:

```text
baseline sorted alpha compositing
sort-free weighted-sum-style approximation
Gaussian-blending-style alpha/transmittance distribution
World Tubes visibility gauge atlas
WorldFoam transmittance atlas
```

This is the cleanest experiment for showing why foam deserves to exist as a
second paper. It should measure not only PSNR but flicker, order flips, and
gradient stability near crossings.

### E4. Cell topology diagnostics

For trained foam scenes, log:

```text
Cech/AABB degree
witnessed edge score S_ij
edge churn
face holonomy
holonomy/residual correlation
unwitnessed edge traversal mass
```

Falsification:

```text
If these diagnostics do not predict heldout residuals, traversal instability,
or source leave-one-camera-out error, they are not paper contributions.
```

### E5. Public quality benchmark

Minimum public datasets:

```text
DeepView / similar calibrated multicam scene
Neural 3D Video or Technicolor-style dynamic scene
D-NeRF synthetic dynamic scenes
```

Baselines:

```text
per-frame dynamic 3DGS
STAR UVT / World Tubes compatibility route
PowerFoam/WorldFoam per-frame replay
compiled WorldFoam atlas
external 4DGS/STG numbers where practical
```

Acceptance:

```text
Do not promote until heldout quality is at least competitive with the relevant
same-representation replay baseline and no longer fails the clean DeepView
thresholds.
```

## 7. Claim Boundaries

A safe near-term claim:

```text
WorldFoam provides a camera-gauged ray-fiber transmittance formulation and a
Metal prototype whose cell/intersection work can be reused across time. Focused
microgates show favorable frame-count scaling, while synthetic experiments
show reduced sorting pathologies under translucent crossings.
```

An unsafe current claim:

```text
WorldFoam is already a SOTA replacement for dynamic Gaussian splatting on real
public dynamic novel-view benchmarks.
```

A future full-paper claim after gates clear:

```text
WorldFoam matches or improves dynamic Gaussian rendering quality while replacing
per-frame primitive sorting with compiled ray-fiber transmittance, achieving
lower frame-count scaling for known camera programs and better behavior under
finite exposure, rolling shutter, and translucent ambiguity.
```

## 8. Discussion

WorldFoam is mathematically cleaner than sorted splatting for visibility
because it renders density through transmittance. It is also riskier. The
discrete splat ecosystem has strong optimized baselines, mature quality
benchmarks, and clear compatibility semantics. Foam must prove not just that it
is elegant, but that it can train stable geometry and match heldout views.

The right paper strategy is therefore two-stage:

```text
Paper 1: World Tubes
    compatibility with dynamic Gaussian splats
    camera-path compiler
    visibility gauge atlas
    compiled adjoints
    sublinear dominant world-side scaling

Paper 2: WorldFoam
    lifted opacity/transmittance semantics
    bounded cell complex
    ray-fiber prefix rendering
    topology/witness diagnostics
    speed evidence first, full quality claim only after gates
```

This split keeps the ideas sharp. World Tubes can be accepted or rejected on
whether camera-path compilation helps dynamic splats. WorldFoam can be accepted
or rejected on whether lifted transmittance plus cell topology produces a
better renderer/model family.

## References To Include

- Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering."
- Wu et al., "4D Gaussian Splatting for Real-Time Dynamic Scene Rendering."
- Yang et al., "Deformable 3D Gaussians for High-Fidelity Monocular Dynamic
  Scene Reconstruction."
- Li et al., "Spacetime Gaussian Feature Splatting for Real-Time Dynamic View
  Synthesis."
- Hou et al., "Sort-free Gaussian Splatting via Weighted Sum Rendering."
- Koo et al., "Gaussian Blending: Rethinking Alpha Blending in 3D Gaussian
  Splatting."
- PowerFoam upstream paper/implementation notes already collected under
  `research_notes/foam_papers/`.
