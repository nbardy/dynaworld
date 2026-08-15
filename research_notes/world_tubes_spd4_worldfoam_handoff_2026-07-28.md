# World Tubes, Native SPD(4), Ordered Ray Transfer, and WorldFoam
## Shareable Research and Engineering Handoff

- Date: 2026-07-28 KST
- Repository: `/Users/nicholasbardy/git/gsplats_browser/dynaworld`
- Intended audience: a researcher or engineer joining without the originating
  conversation
- Source status: implemented in a dirty working tree and dirty nested STAR
  repository; not yet a clean, committed, publication-reproducible revision
- Evidence status: strong mathematical/reference and bounded engineering
  evidence; incomplete public causal and breadth evidence

This document is deliberately self-contained. It consolidates the
representation debate, recovered mathematics, camera-gauge formulation,
production shader work, bounded training, WorldFoam finite-element branch,
paper decisions, failures, safety constraints, and exact next work.

---

## 1. Executive summary

We started from a concern that the existing STAR UVT implementation had become
an inelegant velocity-driven splat with fixed position, fixed color, weak
covariance semantics, and potentially explosive per-frame memory. Repository
archaeology and derivation established a more precise result:

1. A strict native spacetime Gaussian is

   \[
   \mathcal G_i=(\mu_{4,i},\Sigma_{4,i},\alpha_i,a_i),
   \qquad
   \mu_{4,i}\in\mathbb R^4,\quad
   \Sigma_{4,i}\in\operatorname{SPD}(4).
   \]

2. A full linear Gaussian tube with full spatial covariance is exactly the
   same object in conditional coordinates. Its apparent velocity is not an
   independently imposed trajectory law; it is the regression coefficient
   derived from the space-time cross covariance.

3. The core world object should remain in world spacetime \(XYZT\). UVT is a
   camera-compiled local coordinate expression produced by pulling the world
   through a camera-ray bundle and eliminating the ray-depth fiber.

4. The original production scaffold was genuinely narrower than full SPD(4):
   it lacked full world depth extent and full spatial covariance/orientation.
   We therefore added a parallel, opt-in `full_spd4` source while preserving
   the historical `legacy_tube` default.

5. The production renderer now supports:

   ```text
   world source:
     legacy_tube
     full_spd4

   alpha:
     peak_splat
     beer_lambert

   amplitude convention:
     fiber_integrated
     peak_density

   renderer:
     dense
     metal_tile
     retained_fiber_metal
     hybrid_retained_fiber
   ```

6. Conditional depth variance now drives a conservative order certificate.
   Certified tiles use fast STAR compositing; ambiguous tiles may use a
   retained-depth Metal emission-absorption forward/VJP.

7. At matched trainable-scalar count on one bounded Coffee Martini run,
   native SPD(4) was slightly faster, used less sampled driver memory, and
   reached higher heldout PSNR than the legacy source. This is promising
   single-seed engineering evidence, not a publication-quality win.

8. The retained/hybrid branch is correct on bounded fixtures, but the current
   exact depth-band certificate becomes nonselective at dense same-depth
   initialization: the 199-atom stress run falls back on every tile.

9. A finite-element WorldFoam material matrix M0--M5 was implemented behind
   one segment-transfer ABI and one Metal shader family. Positive Bernstein
   P2 and convex log-P2 are complementary, not a universal winner. Compact
   native-4D WorldFoam integration remains intentionally gated.

10. The paper contribution is not “we invented SPD(4) splats.” Native 4D
    Gaussians have prior art. The defensible World Tubes contribution is a
    camera-program compiler and shared adjoint whose structural
    projection/support/binning/visibility work scales with trace/event
    complexity rather than blindly replaying every primitive for every frame.

11. The central missing publication result is now explicit: freeze one learned
    world and compare per-frame STAR replay against one compiled projective
    interval atlas on identical heldout samples, images, loss, and world
    gradients. The executor is implemented and statically hardened, but has
    not yet produced an accepted runtime artifact.

---

## 2. What problem we were actually solving

The user wanted a spacetime-native representation that:

- does not define the world as unrelated per-frame splat states;
- does not make velocity the ontology;
- can move position through time as a property of a 4D volume;
- supports changing cameras, occlusion, and depth ordering;
- reuses dominant forward and backward structural work across many temporal
  samples;
- remains cheap enough to train with Gaussian-like kernels;
- can be compared fairly against per-frame dynamic Gaussian baselines;
- and can eventually generalize beyond Gaussian splats to cellular or
  finite-element spacetime matter.

The output image stack still has an unavoidable lower bound:

\[
\Omega(FHW).
\]

The intended sublinear claim is narrower:

```text
projection
+ support construction
+ binning
+ visibility/order compilation
+ structural backward replay
```

should grow with camera-path trace/event complexity rather than necessarily
with the number of requested frame samples \(F\).

That distinction is central. “Sublinear rasterization” without qualification
is too broad. “Sublinear structural world-side work for bounded-complexity
camera programs” is the defensible claim.

---

## 3. Questions resolved from the research discussion

| Question | Resolved answer |
| --- | --- |
| Is a normal 3D Gaussian described only by width, height, and depth? | Its geometry has a 3D mean and a full SPD(3) covariance. A common implementation stores three log scales and a unit quaternion, which together represent the six covariance degrees of freedom. |
| Why four stored quaternion numbers but three effective degrees of freedom? | Unit normalization removes one degree of freedom. The sign pair \(q\) and \(-q\) also represents the same rotation. |
| What are log scales? | Unconstrained real parameters exponentiated to positive principal standard deviations. They enforce positivity and give multiplicative scale behavior. |
| Does normal 3DGS have opacity? | Yes. Covariance determines the footprint falloff, while a separate learned amplitude/opacity controls the peak contribution. |
| Does a full SPD(4) covariance have a missing rotation parameter? | No. Its ten covariance degrees of freedom already include four principal widths and six orientation degrees of freedom in \(SO(4)\). A pair of unit quaternions can parameterize \(SO(4)\), but an octonion is not required. |
| Does the 4D covariance change through time? | One fixed joint SPD(4) does not contain a time-varying covariance field. Conditioning it at \(t\) produces an affine-moving center and a constant conditional spatial covariance. |
| Where does motion come from if the object is one fixed 4D ellipsoid? | The ellipsoid may be tilted across space and time. Its space-time cross covariance means later time slices intersect it at shifted spatial centers. |
| Is velocity still stored? | The lossless optimization chart may store \(v\), but \(v=\Sigma_{xt}\Sigma_{tt}^{-1}\). It is a coordinate of the same joint covariance, not extra geometry. |
| Is position fixed? | No in the native full-SPD(4) model. The conditioned center moves affinely with time. The old restricted implementation created confusion because its world covariance was incomplete. |
| Can one SPD(4) atom rotate, curve, or change scale over time? | No. One atom gives affine center motion and fixed conditional covariance. Curvature or rotating/changing covariance needs a mixture of atoms or a richer swept object. |
| Should the core object be UVT? | No. The canonical object is in world \(XYZT\). UVT is a camera-gauge compiler output used by the renderer. |
| Is time slicing the right operation? | A physical frame is a raw time slice. The efficient renderer need not materialize a new 3D splat bank; it can compile the world through the camera-ray bundle and evaluate sensor-time traces. |
| Does normalizing \(p(x\mid t)\) preserve birth/death? | No. Normalizing erases the temporal amplitude envelope. Rendering uses the unnormalized time slice or the equivalent compiled transfer. |
| Is the camera-gauge approach sloppy? | No, provided the camera-specific atlas is treated as a cache/gauge compiled from a camera-independent world, not as the world representation itself. |
| Can camera paths change occlusion and depth ordering? | Yes. Projective interval charts, support events, depth-order roots, visibility strata, and explicit fallback are already part of the World Tubes framework. |
| Is `STAR` an established acronym? | No authoritative expansion was recovered. Treat STAR as an internal backend name; do not present a new backronym as history. |
| Do we need a completely new object before continuing? | Not for Paper A. Full SPD(4) plus the camera compiler is a coherent and now implemented first object. Richer cells, swept volumes, and convex-potential atoms remain secondary research branches. |

---

## 4. Core spacetime Gaussian mathematics

### 4.1 Standard 3DGS geometry

A standard anisotropic 3D Gaussian has:

| Geometry parameter | Effective DOF |
| --- | ---: |
| Mean \(\mu_3\) | 3 |
| Symmetric SPD(3) covariance \(\Sigma_3\) | 6 |
| Geometry total | 9 |

A common parameterization is:

\[
\Sigma_3
=
R(q)\operatorname{diag}\!\left(e^{2s_1},e^{2s_2},e^{2s_3}\right)R(q)^\top,
\]

where \(s_i\) are log standard deviations and \(q\) is a normalized
quaternion.

Opacity and appearance are additional:

| Simple appearance parameter | DOF |
| --- | ---: |
| Opacity/amplitude | 1 |
| RGB | 3 |

Practical 3DGS often uses spherical harmonics rather than only RGB.

### 4.2 Strict native spacetime atom

Let \(z=(x,y,z,t)\). One peak-normalized atom is:

\[
\rho(z)
=
\alpha
\exp\!\left[
-\frac12(z-\mu_4)^\top\Sigma_4^{-1}(z-\mu_4)
\right].
\]

Its simple parameter count is:

| Parameter | Effective DOF |
| --- | ---: |
| Mean \(\mu_4=(x_0,y_0,z_0,t_0)\) | 4 |
| SPD(4) covariance | 10 |
| Geometry | 14 |
| Amplitude | 1 |
| RGB | 3 |
| Total | 18 |

### 4.3 Motion from space-time cross covariance

Partition:

\[
\Sigma_4
=
\begin{bmatrix}
A & b\\
b^\top & c
\end{bmatrix},
\qquad c>0.
\]

Then:

\[
v=\frac{b}{c},
\qquad
C=A-\frac{bb^\top}{c}.
\]

The unnormalized spatial time slice at
\(\tau=t-t_0\) is:

\[
\rho_t(x)
=
\alpha e^{-\tau^2/(2c)}
\exp\!\left[
-\frac12
(x-x_0-v\tau)^\top
C^{-1}
(x-x_0-v\tau)
\right].
\]

Therefore:

\[
x_{\mathrm{center}}(t)=x_0+v(t-t_0),
\]

while \(C\) is the constant conditional spatial covariance.

This proves the exact equivalence:

\[
\operatorname{SPD}(4)
\longleftrightarrow
\operatorname{SPD}(3)\times\mathbb R^3\times\mathbb R_{>0}.
\]

The structured chart \((C,v,\sigma_t)\) is not less native than a raw
4D Cholesky factor. It is a time-adapted coordinate chart of the same joint
Gaussian.

### 4.4 Exact limits of one atom

One full-rank SPD(4) atom cannot intrinsically represent:

- curved center motion;
- changing conditional spatial scale;
- rotating conditional covariance;
- multiple simultaneous spatial modes;
- splitting or merging;
- periodic motion;
- arbitrary temporal presence envelopes.

The initial recommendation is to let mixtures of strict SPD(4) atoms absorb
these residuals before introducing a higher-order primitive. This preserves
the Gaussian compiler and avoids making every possible motion degree of
freedom part of the first implementation.

---

## 5. World, compiler, evaluator, and adjoint

The clean architecture has four layers:

```text
A. camera-independent world W_theta
        |
        | pull back through known camera program Gamma
        v
B. camera/gauge compiler C_Gamma
        |
        | emits traces, support, tile/time cells, visibility strata
        v
C. evaluator R
        |
        | materializes requested pixels/times/exposures
        v
D. compiled adjoint
        |
        | reduces image residuals into trace coefficients,
        | then differentiates once into world parameters
        v
   dL / d theta
```

The invariant expression is:

\[
\text{sensor-time trace}
=
\pi_*\Gamma^*(\text{world primitive}),
\]

where:

- \(\Gamma^*\) pulls the world onto the camera-ray bundle;
- \(\pi_*\) eliminates or transports along the ray-depth fiber;
- the resulting UVT record is a local gauge expression, not the world.

The backward chain is:

\[
r
\longmapsto
D_\phi R(\phi,\kappa)^\top r
\longmapsto
D_\theta C_\Gamma(\theta)^\top g_\phi.
\]

Here \(\kappa\) contains compiled topology/order decisions. The current VJP
holds those discrete decisions fixed. Generic event-boundary derivatives are
not yet claimed.

### Canonical naming

| Name | Role |
| --- | --- |
| Gauged UVT / camera-ray bundle | Retained mathematical framework |
| World Tubes | Paper A method |
| STAR UVT / projective STAR UVT | Implementation/backend |
| Ordered Ray Transfer | Bounded World Tubes robustness ablation |
| WorldFoam | Paper B retained-depth/cellular method |
| PowerFoam | Existing implementation lineage, not identical to native-4D WorldFoam |

---

## 6. Camera-gauge and conditional-depth compilation

For an affine local gauge:

\[
y=(u,v,d,t)=Gz+b,
\]

the Gaussian transforms exactly:

\[
\mu_y=G\mu_z+b,
\qquad
\Sigma_y=G\Sigma_zG^\top.
\]

Let \(r=(u,v,t)\). Then:

\[
\Sigma_r=\Sigma_y[r,r],
\qquad
Q_r=\Sigma_r^{-1},
\]

\[
\mathbb E[d\mid r]
=
\mu_d+\Sigma_{dr}Q_r(r-\mu_r),
\]

\[
\operatorname{Var}(d\mid r)
=
\Sigma_{dd}-\Sigma_{dr}Q_r\Sigma_{rd}.
\]

This yields:

- a UVT Gaussian footprint;
- affine conditional mean depth;
- positive conditional depth variance;
- an explicit camera/gauge identity;
- enough information to certify or reject representative-depth ordering.

### Camera-path scope

- Static affine camera compilation is exact relative to the affine gauge.
- `dynamic_first_order` and `projective_first_order` use a differentiable
  one-chart linearization that matches the camera-program value and Jacobian
  at the chart point.
- The latest focused CPU gate covers float64 reference parity,
  camera-program finite differences, source and camera VJPs, a one-step
  moving-camera training update, and fail-loud `segmented` rejection:
  `22 passed`, with one Metal test explicitly deselected.
- Intrinsics and extrinsics remain connected to autograd in this route.
  However, for full SPD(4), `dynamic_first_order` and
  `projective_first_order` currently lower through the same first-order
  compiler.
- The established projective interval atlas handles broader camera programs
  with event-certified local charts for the historical World Tubes route.
- The new physical retained-depth branch has not yet been lowered through the
  full nonlinear/projective trace family.
- The current D-NeRF paper policy requests `segmented`, so it deliberately
  rejects full SPD(4) rather than silently substituting the one-chart mode.
- Long-window approximation remainders, rolling shutter with native SPD(4),
  moving lens distortion, and exact full-orbit retained transfer remain open.

This is why the camera compiler is part of the method rather than a cosmetic
preprocessing step.

---

## 7. Physical amplitude and opacity semantics

### 7.1 Historical peak-splat law

The historical fast law is:

\[
\alpha_{\mathrm{splat}}(r)
=
o\exp[-q(r)/2],
\]

usually capped below one.

### 7.2 Beer--Lambert law

For projected optical thickness:

\[
\tau(r)=\tau_0\exp[-q(r)/2],
\]

the physical single-primitive opacity is:

\[
\alpha(r)=1-\exp[-\tau(r)].
\]

The implemented derivatives are:

\[
\frac{\partial\alpha}{\partial\tau_0}
=
e^{-\tau}e^{-q/2},
\]

\[
\frac{\partial\alpha}{\partial q}
=
-\frac12\tau e^{-\tau}.
\]

Support culling uses:

\[
\tau_{\mathrm{threshold}}
=
-\log(1-\alpha_{\mathrm{threshold}}).
\]

### 7.3 Fiber-integrated versus world peak density

If the trainable amplitude is already a projected peak optical thickness,
the convention is `fiber_integrated`.

If it is a world peak extinction density \(\rho_{\mathrm{peak}}\), the camera
compiler produces:

\[
\tau_0
=
\rho_{\mathrm{peak}}
\left\|\frac{\partial x_{\mathrm{world}}}{\partial d}\right\|
\sqrt{2\pi\,\operatorname{Var}(d\mid u,v,t)}.
\]

For affine gauge spatial rows \(r_u,r_v,r_d\):

\[
\left\|\frac{\partial x_{\mathrm{world}}}{\partial d}\right\|
=
\frac{\|r_u\times r_v\|}
{|\langle r_d,r_u\times r_v\rangle|}.
\]

This reciprocal-frame formula removed a hot-path general matrix inverse.

Important comparison caveat: the same raw `peak_density` initialization does
not yield the same center alpha in every view. The first bounded
peak-density row is therefore not a fair center-alpha-matched baseline.

---

## 8. Retained-depth optical transfer and ordering certificates

For one atom at fixed sensor-time coordinate \(a=(u,v,t)\):

\[
d\mid a
\sim
\mathcal N\!\left(
d_0+\beta^\top(a-\mu_a),
\sigma_d^2
\right).
\]

The retained path evaluates:

\[
\lambda_i(d,a)
=
\tau_i(a)
\frac{
\exp[-(d-\bar d_i(a))^2/(2\sigma_{d,i}^2)]
}{
\sqrt{2\pi\sigma_{d,i}^2}
},
\]

sums extinction and colored emission over active atoms, and integrates
front-to-back Beer--Lambert transfer along depth.

The native Metal VJP covers:

- UVT mean and precision;
- conditional depth intercept and slope;
- conditional depth variance;
- optical thickness;
- color.

### Variance-aware certificate

For each tile-time cell, the compiler:

1. computes conservative optical-support boxes;
2. records at most 256 active atoms;
3. computes exact affine extrema of pairwise depth-band gaps over overlap
   boxes;
4. certifies fast hard ordering only when confidence bands remain separated;
5. routes ambiguity, invalid records, or overflow to retained transfer.

Reason bits:

```text
1 = active-set overflow
2 = invalid record
4 = ambiguous depth bands
```

This certificate is sufficient, not necessary. The dense 199-atom result
shows that exact band separation can be too conservative for performance.

### Ordered Ray Transfer naming

The correct paper label is:

```text
World Tubes + Ordered Ray Transfer
```

An open camera ray has ordered parallel transport or a product integral.
Holonomy ordinarily refers to a closed loop and is already used by a separate
WorldFoam diagnostic. Do not rename the backend `ray_holonomy`.

The registered identities are:

| ID | World | Alpha | Renderer |
| --- | --- | --- | --- |
| WT-OT0 | `legacy_tube` | `peak_splat` | `metal_tile` |
| WT-OT1 | `full_spd4` | `beer_lambert` | `metal_tile` |
| WT-OT2 | `full_spd4` | `beer_lambert` | `retained_fiber_metal` |
| WT-OT3 | `full_spd4` | `beer_lambert` | `hybrid_retained_fiber` |

---

## 9. What was implemented

### 9.1 Native SPD(4) reference/compiler

Directory:

[`research_experiments/spd4_world_tubes/`](../research_experiments/spd4_world_tubes/)

Key files:

- [`model.py`](../research_experiments/spd4_world_tubes/model.py):
  lossless block-Cholesky SPD(4) chart and atom types.
- [`compiler.py`](../research_experiments/spd4_world_tubes/compiler.py):
  affine gauge pushforward, UVT marginal, conditional depth, amplitude
  conventions, confidence certificates, and STAR lowering.
- [`retained_fiber.py`](../research_experiments/spd4_world_tubes/retained_fiber.py):
  float64/CPU retained-depth reference.
- [`retained_fiber_transfer.metal`](../research_experiments/spd4_world_tubes/retained_fiber_transfer.metal):
  Metal retained-depth forward and VJP.
- [`retained_fiber_metal.py`](../research_experiments/spd4_world_tubes/retained_fiber_metal.py):
  native wrapper/autograd boundary.
- [`hybrid_transfer.py`](../research_experiments/spd4_world_tubes/hybrid_transfer.py):
  variance-certified fast/retained routing.
- [`run_capacity_gate.py`](../research_experiments/spd4_world_tubes/run_capacity_gate.py):
  synthetic full-covariance capacity gate.
- [`run_retained_fiber_gate.py`](../research_experiments/spd4_world_tubes/run_retained_fiber_gate.py):
  CPU/Metal retained-transfer gate.
- [`summarize_bounded_training.py`](../research_experiments/spd4_world_tubes/summarize_bounded_training.py):
  standard-library-only bounded-report validator and hash packager.
- [`README.md`](../research_experiments/spd4_world_tubes/README.md):
  implementation, commands, evidence, and limits.

### 9.2 Production STAR integration

Nested implementation:

[`third_party/fast-mac-gsplat/variants/star_uvt_v0/`](../third_party/fast-mac-gsplat/variants/star_uvt_v0/)

Important files:

- [`research_project/trainer_harness/spd4_world_atom.py`](../third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/trainer_harness/spd4_world_atom.py):
  trainable full-SPD(4) atom, initialization, static and moving-camera
  compilation, amplitude semantics.
- [`research_project/benchmarks/multicam_heldout_compare.py`](../third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/benchmarks/multicam_heldout_compare.py):
  parallel source selection, renderer routing, training/evaluation, report
  metadata, fallback counters.
- [`torch_gsplat_bridge_star_uvt/rasterize.py`](../third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/rasterize.py):
  peak/Beer alpha semantics, support, CPU/dense reference, native dispatch.
- [`csrc/metal/star_uvt_kernels.metal`](../third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/metal/star_uvt_kernels.metal):
  Metal Beer--Lambert forward/VJP and production STAR updates.
- [`csrc/metal/star_uvt_metal.mm`](../third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/metal/star_uvt_metal.mm):
  native configuration plumbing.
- [`tests/test_beer_lambert_alpha.py`](../third_party/fast-mac-gsplat/variants/star_uvt_v0/tests/test_beer_lambert_alpha.py):
  physical alpha/support/VJP and live Metal behavior.

Both world representations lower into the same core STAR record:

```text
ma, q_uvt, depth0, depth_beta, opacity, color
```

Full SPD(4) additionally provides `depth_variance` and, when needed,
`peak_to_fiber_scale` sidecars. This shared back-half ABI is why a
same-renderer legacy/SPD(4) A/B is possible without forking the rasterizer.

The historical path remains default. The new path is opt-in, so code-level
parallel A/B testing is possible without deleting or silently redefining the
original representation.

On the local 24 GiB Mac, “parallel” must mean separate comparable runs, not
simultaneous MPS processes.

Important implementation limits:

- the normal fast STAR route still hard-orders by conditional mean depth;
  depth variance is consumed only by the hybrid certificate;
- retained transfer currently loops over every atom for every selected pixel
  and every fixed midpoint depth sample rather than using a retained-depth
  active list;
- retained quadrature is capped at 64 samples and has detached bounds;
- topology, tile membership, certificate decisions, and order swaps are not
  differentiated;
- the production sequence helper's alpha output is currently an all-ones
  placeholder, so its reliable contract is RGB;
- full-SPD(4) Beer paper runs are currently restricted to `static_view`;
  the multi-chart projective-atlas route rejects Beer rather than changing
  semantics silently.

### 9.3 Unified runner and report identity

- [`run_unified_paper_ablation.py`](../research_experiments/paper_runner_suite/run_unified_paper_ablation.py)
  threads representation, alpha, amplitude, renderer, quadrature,
  certificate, and fallback settings through commands, stale-report identity,
  W&B metadata, and summaries.
- [`tests/test_unified_paper_ablation.py`](../tests/test_unified_paper_ablation.py)
  covers config propagation, invalid combinations, report identity, and
  command construction.
- Separate output roots preserve legacy, corrected, and failed rows rather
  than overwriting them.

The direct benchmark defaults to `dense`; the unified single-run paper runner
defaults to `metal_tile`. Both make the choice explicit in report identity.
The higher-level
[`run_unified_paper_matrix.py`](../research_experiments/paper_runner_suite/run_unified_paper_matrix.py)
does **not** yet expose representation, alpha, amplitude, or renderer axes in
its `MatrixRun` record. Until that is repaired, matrix launches inherit the
single-run defaults and cannot be treated as a complete WT-OT0--3 launcher.

### 9.4 Frozen identical-world replay-versus-compiled executor

The central causal paper test is implemented in:

[`run_frozen_world_replay_compiled.py`](../research_experiments/paper_runner_suite/run_frozen_world_replay_compiled.py)

It is designed to:

1. train and hash one final World Tubes checkpoint;
2. replay per-frame STAR projection/bin/render on heldout samples;
3. compile the identical world once into a projective interval atlas;
4. evaluate identical targets and robust-L1 normalization;
5. compare images, losses, nonzero world-parameter VJPs, payload, timing,
   interval complexity, and fallback;
6. bind the report to protocol, source, checkpoint, native extension, and
   report hashes;
7. preserve threshold failure as a complete negative result.

Current status:

```text
implementation: present
static hardening and py_compile: completed
focused behavior tests after latest changes: pending
live MPS artifact: missing
paper claim: not yet available
```

This distinction is crucial. The existing 21-row matrix compares
representations under selected-time rendering; it is not causal compiled-atlas
evidence.

### 9.5 WorldFoam M0--M5 material matrix

Directory:

[`research_experiments/world_foam_lane2/`](../research_experiments/world_foam_lane2/)

Implemented material modes:

| ID | Extinction | Appearance |
| --- | --- | --- |
| M0 | P0 constant | constant RGB |
| M1 | P0 constant | affine RGB |
| M2 | positive Bernstein P1 | constant RGB |
| M3 | positive Bernstein P2 | constant RGB |
| M4 | log-P1 | constant RGB |
| M5 | convex log-P2 | constant RGB |

They share one physical segment ABI:

\[
(\beta,m)
=
\left(
e^{-\tau},
(1-e^{-\tau})c
\right).
\]

Key files:

- [`finite_element_material_transfer.py`](../research_experiments/world_foam_lane2/finite_element_material_transfer.py):
  float64 reference and analytic VJPs.
- [`finite_element_material_transfer.metal`](../research_experiments/world_foam_lane2/finite_element_material_transfer.metal):
  parameterized Metal material evaluator.
- [`finite_element_material_metal.py`](../research_experiments/world_foam_lane2/finite_element_material_metal.py):
  fail-loud wrapper.
- [`finite_element_material_fit.py`](../research_experiments/world_foam_lane2/finite_element_material_fit.py):
  shared-field partial-chord fitting.
- [`verify_finite_element_material_fit.py`](../research_experiments/world_foam_lane2/verify_finite_element_material_fit.py):
  independent artifact verifier.

What is not implemented:

- a compact native 4D cell field;
- a 4D cell/face event compiler;
- M0--M5 integrated into the full owner-tape production renderer;
- evidence that one material law wins on real heldout imagery;
- removal of current WorldFoam per-frame parameter scaling.

---

## 10. Evidence and measured results

### 10.1 Synthetic full-covariance capacity

Artifact:

[`spd4_native_multiview_capacity_cpu.json`](../artifacts/foundation_gates/spd4_native_multiview_capacity_cpu.json)

On a rank-six, three-camera synthetic covariance fixture with matched initial
loss:

```text
full SPD(4) final MSE:  1.16e-13
restricted source MSE: 2.07e-4
```

This proves capacity on the designed full-anisotropy fixture, not natural-scene
quality.

### 10.2 Bounded Coffee Martini A/B

Protocol:

```text
scene: coffee_martini
train cameras: cam04, cam09
heldout camera: cam06
frames: 16
steps: 40
targets per step: 4
seed: 17
backward: direct_atomic + index_add
```

Canonical aggregate:

- [`summary.md`](../artifacts/spd4_bounded_16f_40step/summary.md)
- [`summary.json`](../artifacts/spd4_bounded_16f_40step/summary.json)

| Row | Atoms | Parameters | Heldout PSNR | SSIM | LPIPS | L1 | Train wall | Peak driver bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Legacy peak | 256 | 3,584 | 5.9865 | 0.02154 | 0.89915 | 0.45313 | 4.9020 s | 63,356,928 |
| Full SPD(4) peak, parameter matched | 199 | 3,582 | 7.0054 | 0.03438 | 0.84708 | 0.37022 | 4.7512 s | 46,596,096 |
| Full SPD(4) Beer/fiber, parameter matched | 199 | 3,582 | 7.1333 | 0.03239 | 0.84321 | 0.36105 | 4.6758 s | 46,596,096 |

Clean bounded deltas:

```text
SPD(4) peak versus legacy:
  +1.0189 dB heldout
  -3.1% train wall
  -26.5% sampled peak driver bytes

SPD(4) Beer/fiber versus legacy:
  +1.1467 dB heldout
  -4.6% train wall
  -26.5% sampled peak driver bytes
```

Additional diagnostics:

- Equal-count 256-atom SPD(4) reached 7.0888 dB but produced 42 evaluation
  overflow tiles on `cam09`; it is not the clean comparison.
- Beer/world-peak-density reached 6.5838 dB and was fast, but its
  initialization was not center-alpha matched across views.
- Raw reports include train/heldout previews and videos.
- The memory columns are sampled MPS process allocator peaks, not isolated
  representation-payload measurements.
- These runs predate a durable clean-source commit. The aggregate hashes the
  reports, records
  `execution_source_state="uncommitted_working_tree"`, and explicitly labels
  them bounded engineering evidence.

### 10.3 Retained/hybrid smokes

| Row | Atoms | Train wall | Mean steady forward | Fallback tiles | Invalid/overflow |
| --- | ---: | ---: | ---: | ---: | ---: |
| All retained | 16 | 0.8753 s | 0.05706 s | 64/64 by design | 0/0 |
| Certified hybrid | 16 | 0.9559 s | 0.07375 s | 10/64 | 0/0 |
| Hybrid dense stress | 199 | 3.0194 s | 0.29124 s | 64/64 | 0/0 |

The 16-atom hybrid and all-retained rows match heldout PSNR to the recorded
precision (`5.868593`); that is not yet pixel- or VJP-parity evidence. The
displayed steady-forward means are the recorded cumulative steady time divided
by 39 calls. The 199-atom result is a negative selectivity control, not a
performance result. A full-resolution hybrid run was intentionally not
launched.

### 10.4 Retained-fiber CPU/Metal parity

Artifact:

[`spd4_retained_fiber_cpu_metal.json`](../artifacts/foundation_gates/spd4_retained_fiber_cpu_metal.json)

```text
forward max absolute error:        2.682209e-7
worst normalized VJP error:        1.266599e-7
32 versus 1024 sample error:       2.459586e-4
depth-variance VJP:                nonzero
sampled MPS driver allocation:     27,754,496 bytes
fixed quadrature:                  <= 64 midpoint samples
adaptive quadrature:               not implemented
projective fallback integration:   not implemented
```

This artifact is a static-camera correctness gate. It does not certify the
new first-order moving-camera compiler or a projective retained atlas.

### 10.5 Combined test state

Before the later frozen-world executor additions:

```text
focused combined suite: 142 passed, 4 skipped
live Beer Metal gate:   15 passed
WorldFoam relevant CPU: 57 passed, 3 skipped
WorldFoam opt-in Metal: 40 passed
```

Both CPython 3.14 and project-venv CPython 3.11 STAR native extensions were
built successfully.

Important freshness boundary: the later frozen identical-world code has only
syntax/static validation in its latest form. Do not treat the `142 passed`
count as coverage of those newest changes.

After the first-order moving-camera SPD(4) addition, its focused CPU lane
separately reported:

```text
moving-camera SPD(4) gate: 22 passed
Metal cases in that gate:  1 explicitly deselected
```

This result covers the new gauge/compiler behavior, not the unrun frozen-world
executor or a long-window projective error bound.

### 10.6 WorldFoam material correctness

Artifact:

[`worldfoam_material_m0_m5_cpu_metal_20260727.json`](../artifacts/foundation_gates/worldfoam_material_m0_m5_cpu_metal_20260727.json)

```text
CPU independent-quadrature max error:    5.96e-15
CPU finite-difference VJP max error:      6.86e-10
Metal forward normalized error:           7.51e-8
Metal VJP normalized error:               5.96e-8
tiny-optical-depth branch:                covered
invalid-domain parity:                    covered
```

### 10.7 WorldFoam matched-byte material value

Canonical artifact:

[`worldfoam_material_value_fit_cpu_20260727.json`](../artifacts/foundation_gates/worldfoam_material_value_fit_cpu_20260727.json)

The fixture shares one global material field across partial chords and uses
disjoint heldout chords with an independent target oracle.

Median heldout loss across seeds 17/29/43:

| Target family | M3 positive P2 | M5 convex log-P2 |
| --- | ---: | ---: |
| Positive-P2 target | \(5.26\times10^{-17}\) | \(8.80\times10^{-5}\) |
| Convex-log-P2 target | \(1.33\times10^{-3}\) | \(6.19\times10^{-15}\) |

Both use six float32 material scalars, or 24 bytes. The conclusion is:

```text
winner = null
eligible_for_native_4d_integration = false
```

M3 and M5 are complementary bases. The next scientific gate is adaptive
per-cell selection or real heldout material evidence.

### 10.8 Existing World Tubes causal scaling evidence

The accepted same-representation synthetic/bounded-orbit frame-scaling
experiment compares per-frame replay with compiled interval evaluation for:

\[
F\in\{4,8,16,32,64,128\}.
\]

At \(F=128\):

```text
compiled / replay payload ratio:   0.03125
compiled / replay compile ratio:   0.047677
compiled / replay forward ratio:   0.181323
compiled / replay backward ratio:  0.392235
```

This is the strongest causal systems evidence currently in the paper. The
missing public frozen-world result is intended to establish the same
distinction on one real trained checkpoint.

### 10.9 Selected-time public representation context

Three progressive Coffee Martini protocols are accepted. Each contains World
Tubes, WorldFoam, and dynamic 3DGS, so this is nine accepted method/seed lane
rows rather than a completed matrix. Across the accepted progressive seeds,
the manuscript currently reports approximately:

| Method | Parameters | Peak sampled driver | Mean train wall | Heldout PSNR |
| --- | ---: | ---: | ---: | ---: |
| World Tubes | 14,336 | 3.114 GB | 78.33 s | 5.9153 |
| WorldFoam | 28,569,600 | 15.794 GB | 361.82 s | 5.6159 |
| Dynamic 3DGS | 4,300,800 | 20.557 GB | 79.44 s | 4.9110 |

These rows motivate the compact-spacetime thesis but are not a causal
compiled-atlas comparison. In particular, current WorldFoam stores:

\[
1024\times300\times93=28{,}569{,}600
\]

trainable scalars.

Public matrix accounting:

```text
accepted protocol rows: 3 / 21
missing selected-time rows: 18
missing frozen causal run: 1
total current runtime debt: 19 jobs
```

The accepted protocols are progressive seeds 17/29/43. Fixed seeds 17/29/43
and global-shuffle seed 17 remain absent. The partial fixed seed-17 directory
from the memory-pressure incident is invalid and must not be promoted.

All declared Neural3D scenes and the controlled D-NeRF inputs are local. The
blocker is safe execution and evidence closure, not data acquisition.

---

## 11. Important failures and backtracks

These failures are preserved because they materially changed the design.

### 11.1 “Velocity means the object is not native 4D”

Status: refuted.

The full tube chart and SPD(4) are bijective when spatial covariance is full.
The real narrowing was incomplete spatial covariance and camera compilation,
not the existence of a stored regression coordinate \(v\).

### 11.2 “A fixed 4D ellipsoid cannot move”

Status: refuted.

Space-time tilt means time slices intersect the ellipsoid at changing spatial
centers. One fixed joint quadratic can therefore encode affine motion without
a separately evaluated trajectory function.

### 11.3 Initial native-SPD(4) 10x slowdown

Observed:

```text
first 199-atom SPD(4) row: 40.50 s
legacy row:                4.90 s
```

Root cause:

- unused batched 4x4 inverse;
- unused determinant;
- repeated `.item()` calls forcing MPS synchronization.

Repair:

- skip unused amplitude conversion;
- use the reciprocal-frame fiber Jacobian;
- keep structural device checks but defer scalar numeric synchronization to
  the trainer nonfinite/rollback boundary;
- preserve eager fail-loud checks on CPU.

Projection microprobe:

```text
approximately 119 ms -> 4.4 ms
```

Conclusion: the apparent 10x cost was an engineering bug, not intrinsic
SPD(4) complexity.

Superseded pre-fix outputs remain in the non-`_optimized` directories.

### 11.4 Retained Metal VJP under-dispatch

The Metal compiler inferred its dispatch grid from the first tensor argument.
The first VJP tensor was `[N,3]`, so only `3N` pixel threads launched. Tiny
fixtures accidentally passed.

Repair:

- put the `[F,H,W]` fallback mask first for forward and VJP;
- keep bounds checks;
- add a test whose selected pixel lies beyond the old accidental extent.

Conclusion: tiny shader fixtures must test the actual production domain
extent, not only algebra.

### 11.5 Mean depth is not enough for thick colored overlap

Status: proved by ordered-transfer counterexamples.

Sorting representative means can fail when differently colored conditional
depth profiles overlap. This motivated retained-depth transfer and the
variance/order certificate.

### 11.6 Hybrid certificate nonselectivity

At 16 atoms, the hybrid selects only 10/64 fallback tiles. At 199 same-depth
atoms, it selects 64/64. Weakening the confidence radius without changing the
retained integration extent would invalidate the guarantee.

Next alternatives:

- better physical depth initialization;
- bounded color-commutator acceptance;
- error-certified support;
- eventually adaptive quadrature.

### 11.7 WorldFoam material identifiability

The initial material fit used complete constant-color segments. Such
observations depend only on total optical depth, so density shape is
unidentifiable.

Repair:

- share one global field across multiple partial chords;
- use disjoint heldout chords;
- use independent target integrators;
- fit all modes/seeds with identical optimization policy.

### 11.8 Adam made log-P2 look representationally weak

Short Adam failed the log-P2 threshold. Equal deterministic strong-Wolfe
L-BFGS polishing recovered the exact log-quadratic controls. The failure was
optimization conditioning, not missing capacity.

### 11.9 GL16 sharp-Gaussian fallback

For:

\[
q(x)=1000(x-1/2)^2,
\]

the first GL16 path returned `0.01983175` instead of `0.05604991`.

It was replaced by:

- sign-aware analytic erf evaluation across the peak;
- scaled-erfcx tail differences;
- stable small-curvature/log-linear series;
- explicit invalid-row rejection.

The failed implementation was not silently relabeled.

### 11.10 Six WorldFoam renderer forks

Rejected. Six clones would confound material cost with topology, traversal,
scan, and adjoint differences. The accepted implementation is one material
selector behind a shared segment ABI.

### 11.11 “Holonomy” as the open-ray backend name

Rejected. The mathematics is ordered parallel transport on an open path.
Holonomy is reserved for closed loops. The implemented name remains
`hybrid_retained_fiber`.

### 11.12 The 21-row matrix was described as compiler evidence

Status: corrected.

The matrix compares representation quality/cost under selected-time rendering.
The new frozen identical-world route is the actual causal compiler test.

---

## 12. Mac memory incident and execution policy

The original `kernel_task` concern occurred in the context of severe unified
memory pressure, compression/swap, MPS driver allocation, and large
per-frame-state workloads. `kernel_task` was macOS protecting the system
through resource/thermal management; killing it was neither possible nor the
correct response.

Observed risk factors included:

- a 24 GiB unified-memory workstation;
- WorldFoam's per-frame parameter tensor over 300 frames;
- large MPS driver allocations;
- eager target/video residency in parts of the paper runner;
- concurrent unrelated Node/CPU load;
- a prior operator-killed fixed paper row after memory pressure.

Consequences now encoded in the workflow:

- no concurrent local MPS training jobs;
- each representation runs in an isolated process;
- completed lane reports can be resumed rather than repeated;
- host-side targets and bounded frame chunks are used in new paper paths;
- source-only/dry-run work is allowed while the host is unsafe;
- publication-scale MPS requires an adequate clean host and live-resource
  preflight;
- full 300-frame or 512-wide work must not be inferred as authorized merely
  because a CLI flag exists.

For this repository, “test in parallel” means parallel code paths and
side-by-side experimental identities. On the incident Mac they should be
executed sequentially.

---

## 13. Paper and novelty classification

### 13.1 Paper A: World Tubes

Defensible core contribution:

```text
camera-independent spacetime world
  -> known camera program
  -> event-certified sensor-time trace atlas
  -> reusable projection/support/binning/visibility records
  -> compiled forward and shared adjoint
```

Do not claim:

- invention of full SPD(4) Gaussians;
- information-theoretically sublinear output generation;
- universal replacement for dynamic Gaussian methods;
- exact generic camera paths or topology derivatives;
- public superiority from the bounded seed-17 SPD(4) row.

Current manuscript:

- [`WORLD_TUBES_PAPER_DRAFT.md`](gauged_uvt_trace_atlas/paper/WORLD_TUBES_PAPER_DRAFT.md)
- [`WORLD_TUBES_PAPER.tex`](gauged_uvt_trace_atlas/paper/WORLD_TUBES_PAPER.tex)
- [`WORLD_TUBES_EXPERIMENT_PLAN.md`](gauged_uvt_trace_atlas/paper/WORLD_TUBES_EXPERIMENT_PLAN.md)
- [`REPRODUCE.md`](gauged_uvt_trace_atlas/paper/REPRODUCE.md)
- [`WORLD_TUBES_REFERENCES.bib`](gauged_uvt_trace_atlas/paper/WORLD_TUBES_REFERENCES.bib)

The retained-fiber path is now labeled a bounded extension rather than an
unfinished main contribution. The generated Pandoc TeX still needs venue
conversion, integrated citations, clean tables/figures, and a visually
verified PDF.

### 13.2 World Tubes + Ordered Ray Transfer

This is a robustness/physics ablation inside Paper A, not a rename of the
paper. It is not submission-critical until the hybrid becomes selective at
realistic density.

Registration:

[`TODO/world_tubes_ordered_transfer_ablation.md`](../TODO/world_tubes_ordered_transfer_ablation.md)

### 13.3 Paper B: WorldFoam

WorldFoam keeps ray depth as the physical transmittance axis rather than
marginalizing it early. Its paper already contains:

- optical transfer monoid;
- product integral/path ordering;
- commutator/swap criterion;
- compiled cell/event words;
- same-representation replay theorem;
- prefix/suffix VJP;
- event-complexity caveats.

Finite-element M0--M5 material laws are a WorldFoam extension first. They
become a separate follow-on paper only if a compact native-4D field produces a
decisive quality/byte, memory, or speed result.

Key docs:

- [`GAUSSIAN_FINITE_ELEMENT_WORLD_FOAM.md`](worldfoam_paper/GAUSSIAN_FINITE_ELEMENT_WORLD_FOAM.md)
- [`PAPER_METHOD_CLASSIFICATION_AND_METAL_GATES.md`](worldfoam_paper/PAPER_METHOD_CLASSIFICATION_AND_METAL_GATES.md)
- [`WORLD_FOAM_MATH_APPENDIX.md`](worldfoam_paper/WORLD_FOAM_MATH_APPENDIX.md)
- [`worldfoam_material_basis_selection_gate.md`](worldfoam_material_basis_selection_gate.md)

### 13.4 Distinct incubating objects

#### Self-normalized convex-potential atom

\[
\sigma(x,t)
=
\alpha(t)
\left(
1-q(x,t)+\min_y q(y,t)
\right)_+^p,
\qquad
\nabla_x^2q\succeq\lambda I.
\]

Useful properties:

- unique derived ridge;
- convex connected spatial slices;
- one ray interval;
- derived local motion/orientation/scale.

Open blockers:

- nontrivial per-time minimizer;
- generally algebraic rather than polynomial ridge;
- overlap transfer cost;
- not automatically a foam partition;
- crowded prior art;
- no matched evidence against mixtures of simpler atoms.

Status: genuinely distinct candidate, not paper-ready.

#### Bernstein swept-volume atom

The July 28 meta-review proposed compact spline-controlled swept volumes as a
serious independent lane for curved motion and changing affine cross-section.
It has attractive semialgebraic support and ray-root structure, but no current
implementation. It is post-submission research, not a reason to reopen Paper A
architecture.

#### Native 4D finite-element cells

Promising because they directly target current WorldFoam per-frame state
growth. Missing:

- view-consistent 4D basis storage;
- cell/face event compiler;
- full VJP through cell geometry/topology;
- a selected material basis;
- real quality/byte evidence.

---

## 14. Fair baseline hierarchy

### Primary causal baseline

Freeze one trained world and compare:

```text
A. per-frame replay:
   project + support + bin + order + render for every requested frame

B. compiled:
   compile one event-stratified camera atlas
   evaluate the same requested frames from that atlas
```

Hold fixed:

- checkpoint/world parameters;
- camera program;
- requested times;
- targets;
- raster resolution;
- alpha/compositing law;
- active primitive set;
- loss normalization;
- gradient parameter names.

This is the purpose of `run_frozen_world_replay_compiled.py`.

### Representation baseline

Compare legacy and native SPD(4) at both:

- equal atom count;
- matched trainable scalars/bytes.

The clean bounded comparison is the 256 legacy versus 199 SPD(4)
matched-scalar row.

### Ordered-transfer ablation

Compare WT-OT0 through WT-OT3 with identical source initialization,
quadrature, atom budget, target pixels, optimizer steps, seed, and
background.

### Contextual external baselines

Dynamic 3DGS, native 4DGS, deformable 3DGS, and other official methods are
context, not substitutes for the same-world causal baseline. Official native
4DGS still needs a repository adapter under the canonical split/metric
contract before becoming an authoritative local baseline.

---

## 15. Current source and provenance state

The work is present on disk but not cleanly landed.

Observed:

- superproject has many modified and untracked files;
- nested `third_party/fast-mac-gsplat` is also modified/untracked;
- bounded SPD(4), retained-fiber, WorldFoam, paper-runner, docs, tests, and
  artifacts are among the dirty changes;
- unrelated browser and other user changes coexist in the same worktree;
- no commit or push was made by this work.

Consequences:

- preserve unrelated changes;
- do not reset or overwrite the worktree;
- split and review commits intentionally;
- rerun publication evidence from a clean source revision;
- current bounded report hashes prove artifact identity, not clean-source
  reproducibility.

---

## 16. What is complete and what is not

| Area | Status |
| --- | --- |
| Native \(\mu_4+\operatorname{SPD}(4)\) reference | Implemented and verified |
| Lossless conditional block chart | Implemented and verified |
| Motion from space-time covariance | Derived, implemented, tested |
| Static affine camera gauge | Implemented |
| First-order moving-camera one-chart compiler | Implemented and CPU tested |
| Full nonlinear/projective native-SPD(4) retained path | Missing |
| Parallel legacy/full-SPD(4) trainer axis | Implemented |
| Single-run paper-runner physical axes and report identity | Implemented and tested |
| Matrix-runner propagation of those physical axes | Missing |
| Peak-splat and Beer--Lambert alpha/VJP | Implemented |
| Fiber-integrated and peak-density amplitude conventions | Implemented |
| Variance-aware order certificate | Implemented |
| Retained-fiber Metal forward/VJP | Implemented |
| Hybrid fast/retained production seam | Implemented |
| Dense-scene hybrid selectivity | Failed current stress gate |
| Adaptive/error-certified retained quadrature | Missing |
| Event/boundary derivatives | Missing |
| Bounded single-seed SPD(4) A/B | Completed |
| Multi-seed, multi-scene SPD(4) A/B | Missing |
| M0--M5 material reference and Metal parity | Completed |
| Universal WorldFoam material winner | No; refuted by complementary targets |
| Compact native-4D WorldFoam field/compiler | Missing |
| Frozen identical-world executor | Implemented, runtime unverified |
| Public frozen replay-versus-compiled result | Missing |
| Selected-time public matrix | 3/21 protocol rows accepted |
| Paper Markdown and generated TeX | Present |
| Venue-ready paper and verified PDF | Missing |
| Clean committed source provenance | Missing |

---

## 17. Exact next actions

### Submission P0

1. Review and intentionally land the source changes in clean commits without
   touching unrelated user work.
2. Run the focused CPU behavior gates for the latest frozen-world code.
3. Propagate representation, alpha, amplitude, and renderer identity through
   `run_unified_paper_matrix.py` before using it for WT-OT experiments.
4. On an adequate clean host, run a tiny frozen-world comparison and require:
   - image/loss parity;
   - nonzero matching world VJPs;
   - checkpoint and source hash agreement;
   - bounded residency.
5. Run the all-300-frame frozen identical-world comparison.
6. Run a same-checkpoint frame-count sweep to test structural scaling on public
   trained data.
7. Execute the 18 missing selected-time matrix protocols.
8. Regenerate the authoritative aggregate, tables, figures, Markdown, TeX,
   bibliography, and rendered PDF.

### Post-paper P1: ordered transfer

1. Run physical-depth initialization ablations.
2. Derive a bounded color-commutator or error-certified support criterion.
3. Require hybrid image and VJP parity against all-retained on stress cases.
4. Require ordinary-scene fallback below the declared 20% target.
5. Require hybrid end-to-end wall time below all-retained.
6. Add adaptive forward/VJP quadrature with an explicit error estimator.
7. Only then consider public WT-OT0--3 runs.

### Post-paper P2: WorldFoam and richer objects

1. Test adaptive M3/M5 selection or real heldout material evidence.
2. Select a material basis before native allocation.
3. Implement a camera-independent 4D cell basis and its ray restriction/VJP.
4. Reuse one owner/event tape and optical-element ABI.
5. Compare same geometry, matched bytes, and quality-matched capacity.
6. Only after those gates revisit convex-potential atoms, swept volumes, or
   more general curved world tubes.

---

## 18. Safe commands and entry points

Run commands from the DynaWorld root.

### Verify the bounded SPD(4) aggregate without MPS

```bash
PYTHONPATH=. python3 -m \
  research_experiments.spd4_world_tubes.summarize_bounded_training \
  --verify-report artifacts/spd4_bounded_16f_40step/summary.json
```

### Inspect the frozen causal command without executing it

```bash
PYTHONPATH=src/train python3 \
  research_experiments/paper_runner_suite/run_frozen_world_replay_compiled.py
```

The command prints a dry-run identity. Do not add `--execute` on the incident
host.

### Full paper reproduction routing

Use:

[`research_notes/gauged_uvt_trace_atlas/paper/REPRODUCE.md`](gauged_uvt_trace_atlas/paper/REPRODUCE.md)

on a clean, adequately provisioned host. The reproduction document is not a
safety override.

---

## 19. Do not repeat these dead ends

- Do not create another full renderer merely to expose full SPD(4); the STAR
  back half already consumes anisotropic UV precision and affine depth.
- Do not create six WorldFoam renderer forks; keep material laws behind one
  segment ABI.
- Do not claim SPD(4) itself as the paper novelty.
- Do not equate selected-time representation rows with causal compiler
  evidence.
- Do not use complete constant-color segments to identify density shape.
- Do not promote a raw peak-density initialization as center-alpha matched.
- Do not weaken the depth-band certificate without preserving an explicit
  image/quadrature error guarantee.
- Do not call open-ray transfer holonomy.
- Do not run multiple MPS lanes simultaneously on the 24 GiB incident Mac.
- Do not start native-4D WorldFoam before a material/field gate selects what to
  integrate.
- Do not reopen object-theory proliferation before completing Paper A's frozen
  causal and public breadth evidence.

---

## 20. Recommended reading order

For a new collaborator:

1. This handoff.
2. [`PROJECT_INDEX.md`](../PROJECT_INDEX.md)
3. [`research_notes/spacetime_gaussian_representation/README.md`](spacetime_gaussian_representation/README.md)
4. [`research_experiments/spd4_world_tubes/README.md`](../research_experiments/spd4_world_tubes/README.md)
5. [`agent_notes/loose_notes/2026-07-27_17-36-51_spd4_physical_renderer_and_bounded_training.md`](../agent_notes/loose_notes/2026-07-27_17-36-51_spd4_physical_renderer_and_bounded_training.md)
6. [`agent_notes/loose_notes/2026-07-28_15-44-33_world_tubes_paper_closure_audit.md`](../agent_notes/loose_notes/2026-07-28_15-44-33_world_tubes_paper_closure_audit.md)
7. [`WORLD_TUBES_PAPER_DRAFT.md`](gauged_uvt_trace_atlas/paper/WORLD_TUBES_PAPER_DRAFT.md)
8. [`TODO/unified_paper_ablation_pipeline.md`](../TODO/unified_paper_ablation_pipeline.md)
9. [`TODO/world_tubes_ordered_transfer_ablation.md`](../TODO/world_tubes_ordered_transfer_ablation.md)
10. [`worldfoam_material_basis_selection_gate.md`](worldfoam_material_basis_selection_gate.md)

Deep mathematical archive:

- [`spacetime_gaussian_representation/`](spacetime_gaussian_representation/)
- [`gauged_uvt_trace_atlas/`](gauged_uvt_trace_atlas/)
- [`worldfoam_paper/`](worldfoam_paper/)
- [`camera_program_compiler_and_cellular_backend_synthesis.md`](camera_program_compiler_and_cellular_backend_synthesis.md)
- [`meta_review_jul_28th.md`](meta_review_jul_28th.md)

---

## 21. Claim language for sharing

Safe:

> We implemented a parallel native-SPD(4) World Tubes source, physical
> Beer--Lambert alpha/VJP, conditional-depth uncertainty certificates, and a
> retained-depth Metal fallback. On one bounded single-seed Coffee Martini
> comparison at matched trainable scalars, native SPD(4) improved heldout
> metrics while slightly reducing wall time and sampled driver memory. The
> result is engineering evidence, not yet a public benchmark claim.

Safe:

> The central World Tubes claim is camera-program compilation and shared
> structural work across time, not the novelty of 4D Gaussian primitives.

Safe:

> The retained/hybrid path is correctness-green on bounded fixtures, but its
> current dense-scene certificate falls back everywhere and does not yet
> establish a performance win.

Unsafe:

> We invented native 4D Gaussian splats.

Unsafe:

> Rendering all frames is sublinear in the number of output pixels.

Unsafe:

> The SPD(4) method beats dynamic 3DGS or native 4DGS generally.

Unsafe:

> WorldFoam's finite-element formulation is production-ready or has a selected
> universal material law.

Unsafe:

> The public replay-versus-compiled causal result is complete.

---

## 22. Final handoff state

The research question is no longer “what is the object?” The first coherent
object and compiler boundary are established:

```text
native world:
  mu4 + SPD(4) + amplitude + appearance

camera compiler:
  gauged UVT footprint
  + conditional depth mean/variance
  + support/order certificates

evaluator:
  fast STAR where certified
  retained ordered transfer where required

adjoint:
  native renderer VJP
  + compiler/world chain rule
  with discrete topology held fixed
```

The next World Tubes work is evidence closure, not another representation
cycle. The next representation cycle belongs after the paper and should be
triggered by a measured residual: dense fallback nonselectivity, failure of a
single SPD(4) mixture budget, or a demonstrated WorldFoam quality/byte gap
that compact 4D cells can attack.
