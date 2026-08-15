# WorldFoam in Gauged Camera Space:
# Gauge-Invariant Ordered Ray Transfer for Moving Cameras

Draft date: 2026-08-03

Status: second-paper working draft. This is a theory/prototype manuscript
skeleton, not a final quality claim. It should be read after the World Tubes
paper draft because it deliberately branches away from baseline-compatible
Gaussian-splat compositing into a lifted opacity/transmittance representation.

**Evidence status as of 2026-08-15.** The native training-memory ablation has
`0/21` measured rows and the public heldout-quality ablation has `0/36`
measured rows. Consequently `native_memory_fit=false` and
`public_quality=false` currently mean *not yet measured*, not measured
failure. The fixed representation state is analytically small at `S=1024`
(`114,688 B` live, `81,920 B` checkpoint, `278,528 B` for the conservative
live-plus-checkpoint-clone bound), but this excludes allocator, driver, native
scratch, compiler, decoder, and process RSS. The frozen full-pixel public
schedule is also compute-intractable in its present exact implementation: it
would cold-compile roughly `113--115` million `(view,pixel)` tracks per seed.
That schedule remains the correctness reference; publication training must
not silently relabel it.  A separately versioned matched selected-ray G4-v2 is
now source-implemented for all four comparison routes: 300 steps, four
spacetime samples per step, 1024 shared selected pixels per sample, and one
RGB-MSE contract (`1,228,800` target pixels per row), followed by the unchanged
full 300-frame heldout camera.  WorldFoam and Gaussian rasterized-pixel work is
reported separately rather than claimed equal.  The WorldFoam heldout route
compiles each spatial track across time once and uses bounded dual raw spools;
this removes the old frame-major compile explosion without turning temporary
disk into a model-memory claim.  Its bounded native pilot and all 36 measured
rows remain absent. Unit and source gates below are preflight only and must not
be reported as ablation rows.

The Pass-4 finite-element reference and fixed-tape Metal material microkernel
now pass their bounded CPU and tiny-Metal parity gates. They validate local
segment algebra and its VJP only, not native-4D model scaling, trained quality,
or a paper result. A newer fixed-word P0 CPU reference adds the physical fiber
Jacobian, constant-state recomputed VJP, sparse geometry lowering, and bounded
track/time scratch; a matching suffixed Metal source bridge exists but has not
been rebuilt or executed. The CPU systems path now also includes an exact
active kinetic owner-chart compiler, multi-chart right-continuous dispatch,
continuous primal/referenced-material-action certification, and a
frozen-program stable-stratum VJP for positions, velocities, quadratic weights,
affine rays, density, and RGB. A bounded frame-free CPU artifact store,
dense-observation replay source, executor-sealed full word VJP, and fenced
node-length geometry reduction now meet in one source-level request/step
candidate. A subsequent static lifecycle audit made the generic/full-geometry
compatibility path keep any full-frame decode in CPU-only scratch and transfer
only bounded selected-pixel chunks. The material-only schema-v3 procedural
driver is narrower and stronger: it directly generates the requested pixels,
preserves request order and duplicates, enforces its source budget before
allocation, and reports zero full-frame materializations. A source-only public
candidate now stores per-camera uint8 RGB in pixel-time `[H,W,F,3]` order,
maps at most one payload at a time during a selected-pixel read, explicitly
closes it, content-hashes strict cache metadata at construction, and delegates
ordinary full-frame evaluation to the existing decoder. The construction scan
and OS page-cache pressure are not accelerator-tensor measurements. No cache,
populated dataset binding, or runtime evidence exists yet; a strict source-only
binding validator is implemented but unrun. The procedural row therefore remains
a mechanical memory gate rather than public-data evidence. The audited
5.41-TiB decode amplification is an avoided source-level counterfactual, not a
measured allocation result. Completion fences are
required at sample, active-block, and request-delta release boundaries. Error
paths are also explicit: poison alone retains executor-owned state, abort
releases it only after a successful completion fence, and failed
construction/abort/release fences quarantine the live roots on the poisoned
step until process restart. The older standalone full-geometry assembly is
CPU/fake-native-only and rejects accelerator tensors. The schema-v3 producer
now hash-binds the transitive Python/native source manifest, direct-pixel
capability, raw MPS-limit receipt, and per-trial sampled parent-watchdog receipt.
Those latest transaction, backpressure, evidence, native, and test edits have
not been run or rebuilt and are not native-runtime evidence; decoder internals,
real fence semantics, and allocator peak remain unmeasured. The current safe
update policy reuses structure for material-only changes and fully recompiles
after geometry or camera-ray changes. Native kinetic execution, bounded-cell
events, and derivatives through event/chart/rank choices remain open.

The checked-in procedural `F=8/64/300` configuration contains two sites at
`384x384`. It is intentionally only a retention/mechanical scaling fixture.
It cannot support a claim that the paper trainer fits in memory. That stronger
claim requires the spatial-block route with `1024` global sites at `384x512`,
a fenced device VJP, an actual CPU optimizer mutation, and measured bridge,
allocator, and process-RSS peaks.

The fixed-size state itself is already small by construction, but this is an
analytic/source accounting result rather than the missing allocator result. For
the current P0 material plus affine kinetic geometry with two weight
coefficients, the combined CPU training state is `112 B/site` and the raw
restart checkpoint is `80 B/site`: respectively `112 KiB` and `80 KiB` at
`S=1024`. During a fenced material update, the bridge-owned device material
snapshot and copied CPU material-gradient receipt total `32 B/site` plus a
three-float background and one-float copied loss. The separately owned single
global device material bar adds another `16 B/site`; thus the dominant
site-proportional terms are about `48 KiB` at `S=1024`, before the coordinator's
loss, diagnostics, and geometry scratch. None of these terms contains `F`.
Active compiled-lane state,
full-geometry reduction scratch, target/ray streaming, framework allocation,
and command-buffer lifetime can still dominate the real peak; the distinct
training-memory ablation must measure them rather than extrapolating from these
byte formulas.

Constant peak memory is not, by itself, a differentiator against a fair
per-frame baseline.  The same WorldFoam representation can replay one frame at
a time, accumulate one global material/geometry bar, release that frame's
scratch, and thereby keep peak memory independent of `F`.  That sequential
control repeats owner/run evaluation, prefix transfer, and reverse lowering for
every frame, however.  The compiled shared-adjoint claim is therefore narrower
and stronger: both routes must fit under the same measured peak budget, while
the compiled route should keep certified world-side structure and reverse work
sublinear in `F`; selected-pixel target reads, ray/sample evaluation, and the
corresponding camera slice remain linear.  The paper must report peak memory,
world-side work counters, sample-side work counters, memory traffic or its
closest measured proxy, and wall time together.  A dense all-site/all-frame
retained tape is only an optional stress ablation, not the per-frame baseline.

## Abstract

Dynamic Gaussian splatting renders a scene by projecting primitives to the
image plane, assigning them to tiles, sorting or approximating visibility, and
alpha compositing the resulting screen-space contributors. This design is fast,
but its visibility model is discrete: primitive order must be resolved, order
changes cause discontinuities, and repeated renders from a known camera path
repeat cell/projection/intersection work that is coherent across time.

We introduce **WorldFoam in Gauged Camera Space**, a camera-compiled
ordered ray-transfer representation for dynamic rendering. Instead of rendering
a frame as a sorted list of splats, we represent world matter as a bounded
spacetime cell complex with local extinction and radiance fields. A camera
program defines a moving family of ray fibers; local gauges choose depth
coordinates on those fibers. Pulling world matter through the camera program
produces a lifted optical generator over sensor time and ray depth. Rendering
is its path-ordered product integral:

```text
A_y(z) =
  [ -lambda_y(z) I_C    eta_y(z) ]
  [ 0                   0        ]

M_y = P exp integral A_y(z) dz
I(y) = decode(M_y, I_bg).
```

For scalar extinction with `eta=lambda c`, this is exactly the familiar
Beer--Lambert transmittance integral. Under an orientation-preserving depth
change, `A(z) dz` transforms as a matrix-valued one-form, so `M_y` is invariant
when the physical ray-length Jacobian is included.

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
depth order is encoded by the placement of color mass along cumulative optical
depth rather than erased by a depth marginal. This makes it attractive
for translucent ambiguity, finite exposure, rolling shutter, and fast camera
paths, but also makes it a more radical model requiring its own quality and
parity evidence.

Our current implementation evidence supports the prototype direction rather
than a completed SOTA claim. Metal Gate4/native-cutwalk microgates show
favorable requested-density scaling against matched STAR UVT comparators on
their tested fixed programs. A separate
parameterized M0--M5 segment-material shader now matches its float64 reference
and explicit VJP on a bounded 12-record gate, but it is intentionally not a
renderer fork or throughput benchmark. Public benchmark quality, compact
native kinetic execution, official CUDA/Warp parity, and clean heldout
acceptance remain open. The memory-light CPU/source contract is stronger:
prepared native tokens own no global or chart-local time clone, requested
samples reduce into `O(sum J_c)` node cotangents, and the verified block-major
material path performs one material-word VJP per active compiled block in
`O(sum J_c R_c)` rather than per-frame replay. The integrated source-only
full-geometry path still fences and immediately reduces one bounded
`[J,R_run]`
physical-length cotangent per active block. A newer unselected fused
fixed-camera VJP removes that cotangent entirely by lowering adjacent word bars
directly into material and kinetic-world bars; it is source-written but remains
unbuilt, runtime-unverified, and gated on global validation and staged-oracle
parity. The executor receipt
proves only `fenced_and_reduced_not_globally_committed`; the request and step
receipts separately prove their scatter and commit boundaries. It remains
unrun after integration, and the installed native extension predates the
required compiled schemas. The fused entry still consumes the frame-independent
compiled primal `[J,R_run]` length table, exposes no ray cotangent, and is excluded
from selected renderer routing and the integrated coordinator. Its source now
has a dry-run numerical validation kernel, one four-byte reason mask, and
status-gated accumulation; shared-status phases are intended to validate every
active block before the first write. Native build/parity, high-level
all-block integration, and a bound preventing an adversarial sum of individually
finite atomics from overflowing one destination remain open. This paper
therefore separates the ordered-transfer method from the stronger
baseline-compatible World Tubes paper and keeps its systems claims conditional
on native and end-to-end gates.

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

We use **ordered ray transfer**, **vertical parallel transport**, and
**ray-fiber product integral** as the precise terms for this open-ray operator.
The memorable phrase "ray holonomy" describes the same geometric intuition,
but holonomy conventionally denotes transport around a closed loop. We reserve
unqualified **holonomy** below for cell-complex loop diagnostics, avoiding a
collision between two mathematically different objects.

The paper's goal is not to claim that output pixels vanish. The output still
has `O(F H W)` samples if we render `F` frames. The goal is to make the
dominant world-side work scale with **cell/intersection/event complexity** over
the camera program rather than with frame count:

```text
per-frame replay:
    F * (cell/ray intersection + active set + prefix/order + backward replay)

camera-compiled foam:
    reusable topology/camera compile + per-step transfer rebuild + FHW local eval
```

This is the same amortization idea as World Tubes, but the visibility object is
different: a lifted transmittance field instead of a piecewise-stable sort.

### Contributions

1. **Gauge-invariant ordered ray transfer.** We distinguish the camera program,
   which defines the measurement bundle, from a gauge, which selects a local
   depth coordinate. Dynamic matter is pulled into a lifted `(u,v,t,z)`
   extinction/emission generator whose ordered product integral is invariant
   to valid depth reparameterization.

2. **Camera-gauged foam atlas.** We define local gauge domains storing
   cell-ray intersection structure, lifted opacity/radiance bases,
   transmittance prefix summaries, and fallback metadata.

3. **Translated optical-depth measure and compact transfer quotient.** We lift
   each ordered word to the semidirect measure object `(kappa,nu)`, where rear
   color measure is translated by the front optical depth under concatenation.
   Its Laplace image is the associative four-scalar visibility monoid
   `(beta,m) otimes (beta',m') = (beta beta', m + beta m')`. This explains
   noncommutativity, boundary tangent masses, and why the runtime can remain
   compact without marginalizing away depth order. Alpha compositing is the
   atomic-measure case, continuous foam the dense case, and owner-run event
   intervals the sparse camera-compiled case. This formulation was derived in
   this project; external literature novelty remains unestablished.

4. **Cell-path atlas correctness.** We define the implementation-facing
   renderer as a compiled cell/event word evaluated in the visibility monoid.
   The same-representation replay theorem states that if the compiled atlas and
   per-frame replay emit the same independently validated word and run lengths, then they
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
   Gate4/native-cutwalk speed evidence and the fixed-tape material parity
   result, but keep quality acceptance explicit: compact native-4D state, real
   public datasets, matched STAR/GS baselines, official CUDA/Warp parity, and
   clean heldout metrics are still required before a full rendering claim.

The parameterized M0--M5 Metal forward/VJP microkernel is **validation
infrastructure, not a scientific contribution**. It keeps constant,
affine-color, polynomial, and log-polynomial segment laws on one fixed tape
and one `(beta,m)` ABI so their later material-value comparison is not
confounded by different geometry, dispatch, or adjoint implementations.

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

7. Cell-path word rasterization: a camera ray induces a compiler-validated
   front-to-back cell/event word; WorldFoam evaluates that word in the
   visibility monoid.

8. Same-representation replay equivalence: on certified domains, compiled
   WorldFoam matches per-frame WorldFoam replay when the emitted word and run
   lengths match, up to declared basis/quadrature/support/fallback error.

9. Monoid-scan VJP: a prefix-only second replay over `(beta,m)` elements gives
   a constant-state exact adjoint, and constant owner-run elements give closed-
   form `DeltaTau`, physical length, density, and color gradients.

10. Commutator theorem: visibility/order error is controlled by opacity overlap
   times color contrast; this becomes the interval split/compression signal.

11. Fixed-topology VJP boundary: direct VJPs are exact inside fixed compiled
   topology; endpoint/topology changes require refresh, split, smoothing,
   fallback, or future boundary calculus.

12. Retained depth versus Schur closure: World Tubes gets Gaussian Schur
    marginalization. WorldFoam cannot eliminate depth without erasing the
    ordered-overlap phenomenon; its mathematical core is the translated
    optical-depth measure, whose sufficient runtime quotient is affine
    transfer `(beta,m)`. No new Schur-like foam formulation is required.

13. Fixed-program requested-density scaling: for one fixed world, camera,
    physical interval, chart partition, owner-word set, and temporal ranks,
    node forward and ordered-word reverse state/work are independent of the
    number of requested temporal samples. Camera evaluation, target access,
    sample-to-node reduction, and full output remain linear in the requested
    observations. Longer-duration sequences may grow event, chart, word, or
    rank complexity and are not covered by this claim.
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
new. The proposed contribution is the compiled bounded-cell representation and
GPU raster strategy: reuse cell/ray event structure across sensor time while
retaining a compact differentiable primitive family. Its literature novelty
still requires a formal comparison against prior ordered-transfer and compiled
volume-rendering work.

**Tetrahedral radiance fields and differentiable volume meshes.** Tetra-NeRF
uses a Delaunay tetrahedralization as an adaptive NeRF representation around
sparse scene points. Radiance Meshes use constant-density tetrahedral cells for
exact volume rendering through rasterization and ray tracing, while explicitly
handling learned-position topology flips. DiffTetVR differentiates
tetrahedral volume rendering with respect to material and vertex positions and
supports local subdivision. These methods establish that tetrahedral
parameterization, analytic cell integration, differentiable cell geometry, and
adaptive refinement are prior art. Radiance Meshes' constant-density cells are
also a direct reason M0/M1 must remain in every WorldFoam material comparison.

**Voronoi and power foams.** Radiant Foam represents a scene as a
non-overlapping Voronoi volumetric mesh and traverses it through cell
adjacency for differentiable ray tracing. Power Foam replaces the unbounded
Voronoi cells with bounded power cells and adds rasterization-oriented
locality, surface modeling, and decoupled appearance. WorldFoam therefore does
not claim cell foams, neighbor traversal, bounded power diagrams, or their
rasterization as new. Its narrower proposed delta is the gauge-invariant
ordered-transfer formulation and amortized cell/event compilation over a
known moving-camera program. That delta remains provisional until the
same-representation replay and native-4D scaling gates pass.

**Cell-complex diagnostics.** Cech/AABB graphs, regular triangulations,
witnessed complexes, and uncertainty-weighted power distances are natural
tools for making cells coherent under novel views. In WorldFoam they are
candidate representation diagnostics, not established contributions, until
their residual/topology-churn correlation tests pass.

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

**Theorem (translated optical-depth-measure quotient).** Let a finite
front-to-back P0 word have optical depths `tau_r >= 0` and colors
`c_r in R^d` with `||c_r|| <= C < infinity`. Define

```text
K_0 = 0,
K_r = sum_(q<=r) tau_q,
kappa = K_R,
d nu(u) = c_r du,  u in [K_{r-1}, K_r).
```

For two such objects, let `shift(a)(u)=u+a` and define

```text
(kappa_A, nu_A) odot (kappa_B, nu_B)
  = (kappa_A + kappa_B,
     nu_A + shift(kappa_A)_# nu_B).
```

Then these objects form a monoid with identity `(0,0)`. The map

```text
L(kappa,nu) = (beta,m),
beta = exp(-kappa),
m = integral_[0,infinity) exp(-u) d nu(u)
```

is a monoid homomorphism into affine optical transfer:

```text
L(A odot B)
  = (beta_A beta_B, m_A + beta_A m_B)
  = L(A) compose L(B).
```

For any differentiable parameter path that keeps the finite owner-word
combinatorics fixed, the distributional tangent is

```text
dot nu =
    sum_r dot(c_r) 1_[K_{r-1},K_r) du
  + sum_{r<R} (c_r-c_{r+1}) dot(K_r) delta_{K_r}
  + c_R dot(kappa) delta_kappa,

dot beta = -beta dot(kappa),
dot m = integral exp(-u) d(dot nu)(u).
```

*Proof.* The identity law is immediate. Both parenthesizations of three words
produce

```text
nu_A
  + shift(kappa_A)_# nu_B
  + shift(kappa_A+kappa_B)_# nu_C,
```

so `odot` is associative. A change of variables in the shifted rear integral
gives

```text
integral exp(-u) d(shift(kappa_A)_# nu_B)(u)
  = exp(-kappa_A) m_B,
```

which proves the homomorphism. Differentiating each moving interval contributes
its interior color derivative, a positive atom at its moving right endpoint,
and a negative atom at its moving left endpoint. Adjacent endpoint terms
combine into `(c_r-c_{r+1}) dot(K_r) delta_{K_r}`, leaving the terminal atom
shown above. Applying the Laplace functional yields the final two tangent
identities. QED.

**Corollary (zero-width seam).** Inserting or deleting a segment of optical
depth `tau=0` leaves `L` unchanged. As `tau -> 0+`, its local transfer obeys

```text
|1-beta| <= tau,
||m|| <= C tau,
```

so the primal transfer glues continuously to the word with that segment
removed. Its one-sided tangent can remain nonzero through the boundary atoms;
there is no general `C^1` seam and no claimed classical derivative through a
topology change.

The map `L` is intentionally not injective: different ordered color profiles
can share the same final `(beta,m)`. Thus `(kappa,nu)` is the order-explicit
proof object, while `(beta,m)` is the sufficient runtime quotient for the
declared rendered action.

**Proposition (weighted-total-variation certificate).** Let `kappa` and
`kappa_tilde` be nonnegative and extend two finite vector Radon measures with
finite exponentially weighted variation by zero to `[0,infinity)`. Define

```text
||mu||_(e,TV) = integral exp(-u) d|mu|(u).
```

Then, under any fixed vector norm and its induced vector-measure variation,

```text
||m - m_tilde|| <= ||nu - nu_tilde||_(e,TV),

|beta - beta_tilde|
  <= exp(-min(kappa,kappa_tilde)) |kappa-kappa_tilde|.
```

For a common background `b`, rendered color `I=m+beta b` therefore satisfies

```text
||I-I_tilde||
  <= ||nu-nu_tilde||_(e,TV)
     + ||b|| exp(-min(kappa,kappa_tilde))
       |kappa-kappa_tilde|.
```

The same first inequality applied to the signed tangent measures bounds
`||dot(m)-dot(m_tilde)||`. This is a proof/certification norm, not a proposed
native payload.

**Corollary (opacity-tail truncation).** Let `A` be a retained prefix, `B` a
discarded rear word with `||c_r|| <= C_B`, and `b` the background. For the
tangent statement, assume this prefix/tail split and its primitive order stay
fixed under the admitted perturbation. Since

```text
I(A odot B;b) - I(A;b)
  = beta_A (m_B + (beta_B-1)b),
```

the primal tail error is bounded by

```text
||I(A odot B;b)-I(A;b)||
  <= beta_A (1-beta_B) (C_B+||b||)
  <= exp(-kappa_A) (C_B+||b||).
```

Primal opacity alone is not a training-safe truncation rule. For a parameter
direction, let `||dot(nu_B)||_(e,TV)` bound the rear tangent measure and permit
a background tangent `dot(b)`. Differentiating the exact tail difference gives
the sufficient directional bound

```text
||dot(I_full-I_prefix)|| <= beta_A [
    ||dot(nu_B)||_(e,TV)
  + beta_B |dot(kappa_B)| ||b||
  + (1-beta_B) ||dot(b)||
  + |dot(kappa_A)| (1-beta_B) (C_B+||b||)
].
```

A runtime may truncate a tail during training only after choosing a parameter
norm and bounding this quantity uniformly over its admitted unit directions,
not merely after observing large prefix opacity. Converting the rendered-color
JVP bound into a loss-gradient/VJP certificate additionally requires a bounded
and Lipschitz output cotangent. A finite optimizer-step guarantee must keep the
split fixed and control `C_B`, tangent variation, and these loss constants
uniformly over the admitted neighborhood. The current executor does not enable
this optimization. Proof-oracle regressions comparing both bounds with exact
P0 transfer and tangent differences are source-written but unrun.

Density and geometry changes therefore appear as boundary masses in cumulative
optical depth. This explains why a low-rank primal transfer can still require a
higher-rank tangent certificate.

This measure view is used for ordering, tangent, and error arguments; the
executor still stores only `(beta,m)` at each compiled node. It therefore
strengthens the proof contract without expanding the native runtime state.
The proof-only CPU oracle checks associativity, the Laplace homomorphism,
commutators, distributional tangents, and P0 VJP parity; it is validation
machinery rather than a new executor.

This is the selected mathematical formulation, not another public method
name. The measure makes order and moving optical-depth boundaries explicit for
proofs and error analysis. Native execution should continue to store compact
owner words and affine transfers, not a discretized measure or an expanded
per-frame depth object.

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

#### 3.6.1 Shared segment-material ABI

Material comparisons must reuse one cell word, one endpoint tape, one camera
gauge, and one front-to-back scan. A material evaluator receives a normalized
segment coordinate `xi in [0,1]`, physical segment length `L`, a material mode,
three fixed coefficient slots, and appearance coefficients. It emits:

```text
SegmentMaterial -> (tau, beta, m, density_bounds, branch_code)

beta = exp(-tau)
g = (beta,m)
```

The scan sees only the optical-transfer element `g`; material-specific
quadrature, series, or special-function code stays behind this ABI. Its VJP
returns coefficient, appearance, and length gradients. A physical `L` is used
after the gauge Jacobian has been applied; a parameter-coordinate length
without that measure correction is invalid.

The frozen material matrix is:

| ID | Extinction on one segment | Appearance | Role |
| --- | --- | --- | --- |
| M0 | P0 constant | constant RGB | Current constant-material reference |
| M1 | P0 constant | affine RGB | Appearance-only counterbaseline |
| M2 | positive Bernstein P1 | constant RGB | Cheapest richer density |
| M3 | positive Bernstein P2 | constant RGB | Mandatory polynomial counterbaseline |
| M4 | log-P1 | constant RGB | Positive exponential-linear density |
| M5 | convex log-P2 | constant RGB | Gaussian-like `erf` branch |

M3 is mandatory, not a weak ablation. On a normalized ray segment with
nonnegative Bernstein controls:

```text
sigma_P1(xi) = d0 (1-xi) + d1 xi
tau_P1 = L (d0+d1) / 2

sigma_P2(xi) =
    d0 (1-xi)^2 + 2 d1 xi(1-xi) + d2 xi^2
tau_P2 = L (d0+d1+d2) / 3.
```

Thus positive P2 and log-P2 both use three scalar segment controls, while
positive P2 has a cheaper exact optical-depth integral. Nonnegative quadratic
Lagrange samples are not an acceptable substitute: they do not guarantee
nonnegativity between samples.

For constant source color:

```text
m = (1-beta)c.
```

M1 still returns the same `(beta,m)` ABI, but its affine color requires the
exact emission moment. The implementation stores endpoint colors
`c_front,c_back`. If `tau=sigma L`, then:

```text
m =
  [(1-exp(-tau))-h1(tau)] c_front
  + h1(tau) c_back,

h1(tau) = [1-(1+tau)exp(-tau)]/tau,
```

with `h1` evaluated by its small-`tau` series. In the equivalent slope
notation `c(xi)=c0+c1 xi`, the same formula is
`m=(1-exp(-tau))c0+h1(tau)c1`.

For M5 write the negative log extinction as:

```text
q(xi) = a xi^2 + b xi + c,    a >= 0
sigma(xi) = sigma_star exp(-q(xi)).
```

The three-slot implementation fixes `sigma_star=1`; a positive reference scale
is absorbed into `c <- c-log(sigma_star)`.

The fixed-tape implementation must report which numerical branch was used:

```text
near-zero curvature   -> constant/linear moment series
regular a > 0         -> completed-square erf difference
large same-sign tails -> scaled erfcx difference using endpoint densities
a < 0                 -> rejected/fallback in M5; no implicit erfi path
nonfinite/overflow    -> rejected/fallback
```

Branch thresholds are selected by float64/Metal parity sweeps. A shader
constant chosen without that sweep is not part of the method. Fixed GL16 is not
an accepted fallback because it misses sufficiently narrow legal peaks.

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

A general continuous backward may evaluate:

```text
front transmittance prefix
behind radiance action
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

For the landed finite-P0 word, no per-run suffix or reverse tape is required.
One exact forward obtains the final affine transfer; a second front-to-back
scan keeps only the current prefix and derives each local cotangent from the
final output. This is one of the strongest reasons to keep the lifted foam
explicit: forward and backward share an associative transfer structure across
time without an `F x R` interaction tape.

### 3.10 Fixed-surrogate frame-density factorization

For one certified track-local chart, let `g_j` be the exact affine-Lie transfer
at compiler node `t_j`, let `w_j(t)` be the fixed interpolation weights, and
let `G_J(t)` be the decoded interpolant. Let `R_run,p,c` denote the stored
owner-run/word-incidence count for track `p`, chart `c`; it is distinct from a
rotation matrix and from compiler root-discovery counters. Streaming sample
cotangents gives

```text
bar_g_j = sum_f w_j(t_f)
          D decode(g_J(t_f))^T bar_y_f.
```

After that reduction, one exact ordered-word reverse at each node gives

```text
bar_theta = sum_j D_theta g_j^T bar_g_j.
```

This proves the desired separation for the **fixed compiled surrogate**:
expensive ordered-word forward/reverse work is
`Theta(sum_(p,c) J_(p,c) R_run,p,c)`, independent of requested temporal sample
density. The sample slice is not rank-free. Its honest common-path cost is

```text
Theta(sum_(p,c) F_(p,c) J_(p,c)
      + sum_(p,c) N_fb,p,c J_(p,c)^2
      + PF),
```

plus chart lookup and bounded target/output traffic. Calling this `O(PF)` is
valid only when the world, camera program, physical interval, tolerance,
charts, ranks, interpolation rules, and fallback behavior are fixed while
only `F` changes.

The current continuous direct-kinetic certificate bounds the primal transfer
and referenced-material derivative actions. The exact node-length VJP also
differentiates the fixed surrogate with respect to site trajectories, weights,
and affine rays. What is still missing for a physical full-world gradient
claim is a uniform geometry/ray tangent interpolation bound. The theorem
ledger now proves the conditional sparse composition lemma: local primal and
tangent errors bound the global normalized-loss VJP by adding both the direct
Jacobian-action error and the loss-cotangent change caused by primal error,
without a dense global dual or an artificial frame-count factor. The
translated-measure tangent supplies the correct local boundary-mass formula;
it does not by itself supply the remaining temporal tangent bound.

Arbitrary frame densification also needs exact comparison of rational sample
times with algebraic event roots. Until native dispatch closes that seam, the
claim must assume requested samples avoid unresolved isolator neighborhoods.
Query-dependent root refinement would make compilation depend on `F` and is
not evidence for the strong scaling theorem.

Structural reuse remains separate from this factorization. The current
simple-root routine certifies a restricted continuation after reconstructing
the full predicate registry; it does not repair charts, ranks, payloads, or
dispatch. Output-sensitive local repair is open, so geometry and camera-ray
updates safely trigger full structural recompilation and recertification.

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
    forward scan to final affine transfer
    second front-to-back scan with one current prefix state
    accumulate local transfer cotangents
    reduce them into sparse cell/face parameters
```

The important implementation question is tape versus recompute:

```text
large tape:
    faster backward, bad memory scaling

constant-state two-pass P0 replay:
    current exact memory-light route; no per-run suffix/reverse array
```

Current local lessons from STAR/WorldFoam favor the exact two-pass P0 replay
over per-channel or per-run reverse tapes. Native promotion still requires the
second scan to beat the avoided memory traffic on realistic words.

The first Metal stage is deliberately a **fixed-tape material microkernel**.
It forks the segment evaluator, not the renderer: M0--M5 receive identical
precomputed owner words and endpoints and lower to the same `(beta,m)` scan.
Passing it does not show that a WorldFoam model is compact across time. Shader
arithmetic cannot repair a representation whose parameters or owner tape are
still duplicated per frame. The CPU direct-kinetic compiler now proves the
shared chart/transfer/reverse contract without a frame tape. The sealed
multi-chart program now reaches a source-only native-shaped request/step path;
the missing step is to run its focused CPU/fake-native gates, rebuild and attest
the native extension, then measure coefficient/topology/allocator growth
against requested time density.

The fork is intentionally isolated at:

```text
research_experiments/world_foam_lane2/finite_element_material_transfer.metal
research_experiments/world_foam_lane2/finite_element_material_metal.py
```

It exports one parameterized forward kernel and one VJP kernel. Selected and
attested `world_foam_lane2_fused_slab_v0` renderer routing remains unchanged;
an unselected suffixed fixed-camera fused-v1 source entry point has been added.
Only a material law and full-geometry route that clear matched parity,
quality, byte, and allocator gates should be promoted into its owner-run tape.

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

The bounded Pass-4 segment-material gate is also green. One shared evaluator
covers M0 constant, M1 affine-color P0, positive Bernstein P1/P2, log-P1, and
convex log-P2, returning:

```text
(tau, beta, m, density_bounds, branch_status)
```

with an explicit coefficient/color/length VJP. The accepted 12-record artifact
reports:

```text
CPU independent-quadrature max abs error:  5.96e-15
CPU explicit-VJP normalized error:         5.55e-17
Metal forward normalized error:            7.51e-8
Metal VJP normalized error:                5.96e-8
invalid rows:                              0
current MPS allocation:                    4,608 bytes
sampled MPS driver allocation:             28,016,640 bytes
```

Artifact:

```text
artifacts/foundation_gates/worldfoam_material_m0_m5_cpu_metal_20260727.json
```

This proves fixed-segment numerical parity, including the sharp convex
log-quadratic regression that invalidated the earlier fixed-GL16 fallback. It
does not prove training quality, material superiority, event reuse, throughput,
or compact native-4D model scaling.

The next controlled material-value gate is also complete, but it gives a
selection result rather than a universal promotion. A single complete
constant-color segment observes only total optical depth, so the gate shares
one material field across partial chords and evaluates on a disjoint held-out
chord set. Targets are integrated by an oracle independent of the fitted
segment evaluator. Across seeds `17/29/43`, the matched six-scalar M3 and M5
laws separate by generating family:

```text
held-out positive-P2 target:
    M3 loss 5.26e-17
    M5 loss 8.80e-5

held-out convex-log-P2 target:
    M3 loss 1.33e-3
    M5 loss 6.19e-15
```

M3 and M5 both beat the M0/M1 controls by more than `100x` on their own
families, and each beats the other by more than `100x` there. The scientifically
relevant conclusion is therefore **basis complementarity**, not “P2 wins” or
“log-P2 wins.” The saved 36-row CPU artifact is independently verified and
explicitly marks both `winner=null` and
`eligible_for_native_4d_integration=false`:

```text
artifacts/foundation_gates/worldfoam_material_value_fit_cpu_20260727.json
```

This is synthetic held-out material-capacity evidence. It is not image
training, camera-program compilation, Metal throughput, or real-scene quality
evidence.

The follow-on adaptive M3/M5 CPU ablation is now also complete. It fits both
matched six-scalar bases per cell, selects the one with lower loss on a
disjoint selection split, and evaluates that frozen choice on a disjoint
heldout split. Its independent verifier recomputes `72` candidate rows and
`36` selection rows across seeds `17/29/43` and twelve target cells. The
result has `1.0` pure-family selection accuracy, `1.0` heldout-oracle
agreement, adaptive/best-fixed mean loss ratio `0.313405`, and
adaptive/oracle ratio `1.0`:

```text
outputs/benchmarks/2026-08-15_worldfoam_adaptive_material_basis_cpu/summary.json
```

This supports a one-bit per-cell M3/M5 basis tag as the controlled Paper-B
material-selection ablation. It does **not** authorize replacement of P0 in
the native systems path: it is float64 synthetic chord evidence, not native
integration, trained public-image quality, renderer speed, or memory evidence.
The next material promotion decision must use real heldout material or image
observations after the P0 native path is sound.

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

The Pass-4 parity, material-capacity, and adaptive-selection results therefore
belong in the numerical-foundation row, not the quality or systems rows. The
production retained-fiber shader and CPU direct-kinetic compiler exist, while
real-data/native adaptive-material promotion, built native kinetic multi-chart
parity, structural recertification, allocator evidence, production trainer
routing, checkpoint restore, and dataset-scale evidence remain missing. Staged
source coordinators, the CPU combined transaction, and the unselected fused-v1
source exist, but are not evidence for any of those runtime claims.

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

### E2. Requested-density scaling on one fixed compiled program

Sweep:

```text
F = 8, 64, 300
P = 512 fixed pixel tracks
N_observation = P F
```

The three rows reuse the exact same 300-frame physical grid, 512 pixel ids,
camera program, target provider, 1024-site world, all-competitor owner
certificates, charts, and active-word lowering. Only the endpoint-including
requested-time subset changes. Cold CPU compilation may inspect all 1024
competitors and is charged separately; compact device blocks may contain only
post-certification active sites, never a heuristic spatial crop.

Run the full-geometry reverse ablation in two layers. First, at `F=8`, compare
the staged sparse `[J,R_run]` reduction against the fused union-local reverse
on identical inputs and report loss, material-gradient, geometry-gradient, and
post-update parameter parity over three fresh-process repeats. Then run the
accepted fused route alone at `F=8/64/300`. Include a per-frame replay control
where it remains inside the incident guard; a guard-triggered or allocator OOM
is recorded as a censored resource result rather than silently dropping the
row.

Measure:

```text
compile time
forward time
backward time
total optimizer step
cell intersection records
prefix records
fresh-process RSS peak and delta
sampled MPS current/driver allocation high-water and hard working-set ceiling
bridge-visible, active-lane, and combined-state logical bytes
quality delta vs per-frame reference
streamed observation/sample interactions
target bytes read and transferred
provider/replay metadata bytes
the complete structural fingerprint `(E,J,R_run,active blocks)`
```

Pass condition:

```text
the dataset grid, physical interval, world, charts, ranks, owner words, and
active blocks are identical across rows; node-forward and ordered-word-reverse
launch/interaction counts are exactly invariant; no retained tensor scales as
`F x N`, `F x R_run`, or `F x J x R_run`; measured peak follows the bounded streamed
envelope; and output quality stays within tolerance. Observation count,
selected-target traffic, and sample-to-node work must be reported as the
expected linear terms rather than folded into the invariant claim.
```

Run streamed static PowerFoam (`batch=1`) as a memory control in addition to
the repository's per-frame-parameter `MetalPowerFoamVideo` baseline. The former
can also have frame-independent peak memory; the latter exposes the `F x N`
dynamic-state table that WorldFoam is designed to remove.

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
WorldFoam provides gauge-invariant ordered ray transfer factored through a
camera-compiled cell-path atlas. Its visibility monoid, replay theorem,
exact constant-state P0 VJP, active kinetic CPU chart compiler, multi-chart
transfer, frozen-program stable-stratum VJP, and M0--M5 fixed-segment Metal
microkernel are executable; focused Gate4/native-cutwalk microgates show a
promising temporal-reuse signal. This does not include derivatives through
event, chart, rank, node-time, or recompilation choices.
```

An unsafe current claim:

```text
WorldFoam is already a SOTA replacement for dynamic Gaussian splatting on real
public dynamic novel-view benchmarks.
```

A future full-paper claim after gates clear:

```text
WorldFoam matches or improves dynamic Gaussian rendering quality while replacing
per-frame primitive sorting with compiled ray-fiber transmittance. For denser
sampling of a fixed known camera program, its expensive node/ordered-world
forward and reverse are frame-density invariant while sample/target work is
streamed and linear; it also improves behavior under finite exposure, rolling
shutter, and translucent ambiguity.
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
    fixed-program requested-density-invariant compiled world-side work

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
- Kulhanek and Sattler,
  ["Tetra-NeRF: Representing Neural Radiance Fields Using Tetrahedra"](https://arxiv.org/abs/2304.09987).
- Govindarajan et al.,
  ["Radiant Foam: Real-Time Differentiable Ray Tracing"](https://arxiv.org/abs/2502.01157).
- Govindarajan et al.,
  ["Power Foam: Unifying Real-Time Differentiable Ray Tracing and Rasterization"](https://arxiv.org/abs/2604.24994).
- Mai et al.,
  ["Radiance Meshes for Volumetric Reconstruction"](https://arxiv.org/abs/2512.04076).
- Neuhauser,
  ["DiffTetVR: Differentiable Tetrahedral Volume Rendering"](https://arxiv.org/abs/2601.00114).
- Power Foam implementation/reproduction notes collected under
  `research_notes/foam_papers/`; these local notes do not substitute for the
  upstream method citation.
