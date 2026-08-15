---
title: "World Tubes in Gauged Camera Space: Frame-Amortized Dynamic Gaussian Rendering"
author: Anonymous
bibliography: research_notes/gauged_uvt_trace_atlas/paper/WORLD_TUBES_REFERENCES.bib
link-citations: true
---

<!--
SUBMISSION-SOURCE POLICY

This is the concise venue-facing manuscript source. The long-form mathematical
and engineering dossier remains in WORLD_TUBES_PAPER_DRAFT.md and must not be
trimmed to keep this file short.

The submission build must fail, rather than render a warning box or an empty
table, unless all required evidence components are publication eligible:

1. theorem_correctness: accepted in the current schema-v2 evidence ledger;
2. variable_camera_closure_death: accepted from the clean paper-freeze
   schema-v2 artifact through its fixed-SHA compatibility receipt; the newer
   dirty schema-v1 178/179-degree diagnostic is excluded;
3. frozen_world_scaling: requires the frozen identical-world F sweep with raw
   warmed/repeated timing and route-scoped peak-memory samples;
4. public_context: requires all seven core schema-v2 rows;
5. venue_package: requires the official ICLR style, portable figure assets,
   an author-approved AI-use statement, and a clean rendered-PDF audit.

Do not replace a missing component with "NOT SUBMISSION-READY", zero-filled
rows, smoke-test numbers, or a result copied from the long-form dossier.
-->

## Abstract

Dynamic Gaussian renderers repeatedly evaluate, project, bin, order, and
differentiate scene primitives for each requested time. This is efficient for
an isolated frame but duplicates world-side work for video playback, finite
exposure, rolling shutter, and repeated supervision along a known camera path.
We introduce **World Tubes**, a camera-program compiler that maps dynamic
Gaussian primitives into reusable traces over sensor space and time. The
construction pulls a spacetime primitive through the camera-ray bundle and
pushes it forward along the ray-depth fiber. A local Schur complement gives an
exact Gaussian footprint and conditional-depth packet in an affine chart;
homogeneous projective traces, gauge Jacobians, and event-certified domains
extend the construction to bounded moving-camera segments without silently
crossing projective poles or depth-order events. A fixed-topology adjoint maps
image residuals through the compiled evaluator to the original world
parameters. On certified fixtures, ordinary- and log-depth gauges agree to
$3.50\times10^{-13}$ in value and $2.33\times10^{-12}$ in gradient, while
omitting the fiber Jacobian produces errors above $0.59$. Visibility
stratification reduces a deliberate order-crossing error from $0.187$ to zero,
and fixed trace state reaches a $0.03125$ trace-count ratio relative to
per-frame replay at $F=128$. These results establish bounded correctness and
structural reuse while retaining the unavoidable cost of producing the
requested sensor samples. They do not establish learned-scene runtime,
peak-memory scaling, or public-data quality, whose matched artifacts remain
separate gates.

## 1. Introduction

Three-dimensional Gaussian splatting replaces neural ray marching with
visibility-aware anisotropic splats and makes high-quality novel-view rendering
interactive [@kerbl2023]. Dynamic extensions add deformation fields,
persistent trajectories, temporal features, or native spacetime covariance
[@wu2024_4dgs; @yang2024_deformable; @luiten2024; @li2024_spacetime;
@yang2024_native]. These representations differ in how they model the world,
but their renderers generally retain the same computational unit: for every
target time, evaluate the dynamic primitive state, project it, determine
support and tile membership, establish a visibility order, composite, and
backpropagate through that render.

That frame-by-frame unit repeats work when many samples follow a known camera
program. A video renderer requests neighboring times. A finite-exposure image
uses multiple shutter samples. Rolling shutter couples row position to capture
time. Training repeatedly visits the same path or a low-dimensional family of
paths. The image values still require $O(FHW)$ output samples, but the
world-side projection, conservative support, binning, visibility metadata, and
their backward tape need not be rebuilt independently at every time.

World Tubes changes the unit of compilation from a frame to a camera program.
Let $B=\Omega\times\mathcal T$ be the sensor-time base and let $y=(u,v,\tau)$.
A dynamic world primitive is compiled into trace fields

$$
  \bigl(\alpha_i(y),\,c_i(y),\,\widehat z_i(y),\,\nu_{z,i}(y)\bigr),
$$

together with support bounds, visibility certificates, fallback metadata, and
a differentiable map back to the original world parameters. Frames are slices
of this atlas. Exposure and rolling-shutter images are integrals or
row-dependent samples of the same object.

The key difficulty is not drawing a smooth curve through projected centers.
Perspective division can cross a pole; projected support can enter a tile;
and two overlapping translucent primitives can exchange depth order. A trace
may therefore remain algebraically smooth while a renderer compiled under one
support or order topology becomes incorrect. We retain the camera-gauge
mathematics that detects these events. Homogeneous traces are divided only on
certified projective domains, the ray-fiber measure includes its coordinate
Jacobian, and support-overlapping traces are partitioned where their order
certificate fails. Rejected domains are split or routed to an explicit
fallback rather than extrapolated.

Our contributions are:

1. A camera-ray-bundle formulation in which the observable trace is the
   fiber pushforward $\pi_*\Gamma^*w_i$, with an exact local Gaussian
   marginal and conditional-depth packet obtained by a Schur complement.
2. A projective trace and visibility atlas whose denominator, approximation,
   support, and order certificates define bounded reuse domains under moving
   cameras.
3. A fixed-topology evaluator and compiled adjoint that reuse trace and
   tile-time state while returning gradients to the same learned world.
4. A causal evaluation contract that compares per-frame replay and compilation
   from one frozen representation, separating world-side reuse from the
   unavoidable cost of materializing output pixels.

![World Tubes compiles projective traces, certified tile--time cells, and their
adjoint once for a bounded camera program. Per-frame replay repeats the same
world-side lowering. Both routes still shade and materialize every requested
sensor sample.](research_notes/gauged_uvt_trace_atlas/paper/figures/world_tubes_system_overview.svg){#fig:world-tubes-system-overview width=100%}

## 2. Related work

**Gaussian and dynamic splatting.** Classical surface and volume splatting
derive filtered screen footprints in a locally affine ray-space approximation
[@zwicker2001_surface; @zwicker2001_ewa_volume]. Modern 3D Gaussian splatting
learns anisotropic primitives with differentiable visibility-aware
rasterization [@kerbl2023], and Mip-Splatting makes the sampling boundary
explicit [@yu2024_mipsplatting]. Dynamic methods extend the representation
through deformation, motion, temporal opacity, or four-dimensional covariance
[@wu2024_4dgs; @yang2024_deformable; @luiten2024; @li2024_spacetime;
@yang2024_native]. World Tubes is complementary: it compiles repeated rendering
work induced by a camera program rather than proposing another dynamic scene
family.

**Nonlinear cameras and temporal sensors.** Gaussian Splatting on the Move
models blur and rolling shutter, while 3DGUT replaces a first-order projection
with an unscented transform for nonlinear cameras and secondary rays
[@seiskari2024; @wu2025_3dgut]. These methods improve projection at a requested
sample. Our projective domains instead determine when a collection of samples
can share trace, support, and visibility state.

**Visibility and compositing.** Order-independent transparency and sort-free
Gaussian renderers change the transfer law to reduce sorting
[@mcguire2013_weighted_oit; @hou2025_sortfree; @koo2026_blending]. StopThePop
improves view consistency through finer per-ray sorting [@radl2024_stopthepop].
World Tubes preserves the frozen baseline's ordered front-to-back law on
certified cells; it splits or falls back when that noncommutative order cannot
be reused.

## 3. World Tubes

### 3.1 Camera-ray pushforward and gauges

Let $\pi:E_\Gamma\rightarrow B$ be the ray bundle of a known camera program.
The fiber $F_y=\pi^{-1}(y)$ contains the physical ray-depth points associated
with sensor-time sample $y$. The camera program maps bundle points into world
spacetime,

$$
  \Gamma:E_\Gamma\rightarrow\mathbb R^3\times\mathbb R.
$$

For a world primitive $w_i$, its sensor-time trace is

$$
  T_i=\pi_*\Gamma^*w_i.
  \tag{1}
$$

Choose a local depth coordinate $z_a$ on a regular bundle chart $C_a$. For a
density-like primitive $\rho_i$, Eq. (1) becomes

$$
  T_i(y)=\int_{D_a}
  \rho_i\!\left(\Gamma_a(y,z_a)\right)
  J_a(y,z_a)\,\mathrm dz_a,
  \tag{2}
$$

where $J_a$ converts coordinate length to the physical fiber measure. If
$z_b=h(y,z_a)$ is an orientation-preserving reparameterization of the same
fiber segment, the one-dimensional change-of-variables formula makes Eq. (2)
invariant. A chart is invalid when the coordinate map becomes singular or the
two charts cover different clipped fiber segments; invariance does not justify
crossing a pole, support event, or visibility event without a transition.

This distinction is operational. Ordinary depth, log depth, inverse depth, or
an orbit-adapted coordinate may describe the same fiber, but only with the
corresponding $J_a$. Our certified gauge fixture obtains value and gradient
agreement near numerical precision when the Jacobian is present and large
errors when it is removed (Section 4.1).

### 3.2 Exact local Gaussian packet

Consider a strict spacetime Gaussian in a nonsingular local affine camera-ray
chart $s=(r,z)$, with $r=(u,v,\tau)$, mean $(\mu_r,\mu_z)$, and covariance
blocks $\Sigma_{rr},\Sigma_{rz},\Sigma_{zr},\Sigma_{zz}$.

Gaussian marginalization gives the UVT footprint

$$
  r\sim\mathcal N(\mu_r,\Sigma_{rr}),
  \qquad Q_r=\Sigma_{rr}^{-1},
  \tag{3}
$$

while conditioning retains the depth information needed for visibility:

$$
\begin{aligned}
  \widehat z(r)
    &=\mu_z+\Sigma_{zr}Q_r(r-\mu_r),\\
  \nu_z
    &=\Sigma_{zz}-\Sigma_{zr}Q_r\Sigma_{rz}>0.
\end{aligned}
\tag{4}
$$

The strict inequality is the Schur complement of a positive-definite
covariance. Thus one source object yields an anisotropic sensor-time footprint,
an affine conditional-depth plane, and a positive conditional-depth variance.
World Tubes marginalizes depth for fast footprint evaluation but does not
discard the conditional packet used to certify order or invoke a retained-depth
fallback.

Equation (3) is exact only for an affine chart with the declared fiber measure
and integration domain. A moving pinhole camera is nonlinear after perspective
division, so long windows require either certified local approximation or the
projective construction below. Complete covariance and equivalent precision
derivations belong in the supplement.

### 3.3 Projective trace domains

For homogeneous primitive center $X_i^h(\tau)$ and camera matrix $P(\tau)$,
retain the undivided trace and divide only after certification:

$$
\begin{aligned}
  h_i(\tau)&=P(\tau)X_i^h(\tau)
  =\bigl(h_{u,i}(\tau),h_{v,i}(\tau),h_{z,i}(\tau)\bigr),\\
  (u_i,v_i,d_i)&=\left(\frac{h_{u,i}}{h_{z,i}},
          \frac{h_{v,i}}{h_{z,i}},h_{z,i}\right).
\end{aligned}
  \tag{5}
$$

On a bounded interval $I_\ell$, normalized time $s\in[-1,1]$ parameterizes
the supported degree-one or degree-two trace family

$$
  h_{k,i}(s)=\sum_{q=0}^{p}a_{k,i,q}s^q,
  \qquad k\in\{u,v,z\}.
  \tag{6}
$$

The compiler accepts an interval only when four contracts hold:

1. **Projective validity:** $h_{z,i}$ retains the required physical sign and
   remains separated from zero over the complete interval.
2. **Trace accuracy:** divided UV and conditional depth remain within their
   declared residual tolerances.
3. **Support validity:** the footprint plus approximation padding
   conservatively fixes tile membership and active-time gates.
4. **Visibility validity:** every support-overlapping pair has a certified
   order, an accepted commutation residual, or an explicit fallback label.

The implemented denominator certificate is continuous for its quadratic
family. General UV and depth residuals are bounded on the declared probe set;
we do not promote that sampled test into an unrestricted continuous-time
theorem. A failed interval is split at a certified event when available and at
a deterministic midpoint otherwise. A minimum-size interval that still fails
is routed to a reason-labelled fallback.

![A projective trace is divided only after denominator, residual, support, and
visibility certificates all pass. Failure splits the bounded interval or
routes it to an explicit reference fallback; the compiler never extrapolates
through an unresolved pole or order event.](research_notes/gauged_uvt_trace_atlas/paper/figures/world_tubes_projective_compiler.svg){#fig:world-tubes-projective-compiler width=100%}

### 3.4 Visibility-stratified trace atlas

The compiler emits cells

$$
  \mathcal A_\Gamma=
  \{(C_\ell,S_\ell,\Phi_\ell,\Pi_\ell,E_\ell)\}_{\ell=1}^{L},
  \tag{7}
$$

where $C_\ell\subset B$ is a certified sensor-time domain, $S_\ell$ its active
primitive set, $\Phi_\ell$ the trace records, $\Pi_\ell$ the compiled order or
partial-order certificate, and $E_\ell$ the support, error, and fallback
metadata.

For primitive $i$, let $[z_i^-(y),z_i^+(y)]$ conservatively bound conditional
depth. A pair has a fixed order when

$$
  z_i^+(y)<z_j^-(y)
  \quad\text{or}\quad
  z_j^+(y)<z_i^-(y)
  \qquad\forall y\in C_\ell.
  \tag{8}
$$

Otherwise the cell is split, the pair is shown to commute within tolerance, or
the cell falls back. Swapping two alpha-composited contributors changes their
local color by

$$
  \delta_{ij}(y)
  =\alpha_i(y)\alpha_j(y)\bigl(c_i(y)-c_j(y)\bigr).
  \tag{9}
$$

This gives a baseline-relative error certificate for accepted adjacent swaps.
It does not claim that representative-depth alpha compositing is physically
exact radiative transfer. It states when the chosen frozen renderer can reuse
one visibility topology.

The final color inside a cell is evaluated in compiled front-to-back order:

$$
  I(y)=\sum_m
  \left[\prod_{n<m}(1-\alpha_{\pi_n}(y))\right]
  \alpha_{\pi_m}(y)c_{\pi_m}(y).
  \tag{10}
$$

Finite exposure integrates Eq. (10) over shutter time; rolling shutter changes
the sampling map from row position to time. Neither requires recompiling the
world while the requested samples remain in the same certified cells.

### 3.5 Compiled adjoint

Let $\theta$ denote the camera-independent world parameters, let $\kappa$
collect discrete support, bin, event, order, and fallback decisions, and let

$$
  \phi=C_\Gamma(\theta;\kappa),\qquad
  I=R(\phi,\kappa).
$$

Within a fixed-topology stratum, the world gradient is

$$
  \nabla_\theta\mathcal L
  =D_\theta C_\Gamma(\theta;\kappa)^\top
   D_\phi R(\phi,\kappa)^\top
   \nabla_I\ell(I,I^\star).
  \tag{11}
$$

The first reverse map is the interval-renderer VJP; the second accumulates
through projective coefficients, gauge transformations, and the local Gaussian
packet into the same world parameters used by replay. Event-boundary
derivatives are not hidden inside Eq. (11). When a parameter update changes
$\kappa$, the compiler must refresh the affected structure or use a separately
declared boundary estimator.

### 3.6 Work boundary

For $F$ requested times and materialized interaction count $K$, a coarse
decomposition is

$$
\begin{aligned}
W_{\mathrm{replay}}&=W_{\mathrm{shade}}(K)+\sum_{f=1}^{F}W_{\mathrm{world}}^f,\\
W_{\mathrm{tube}}&=W_{\mathrm{shade}}(K)+W_{\mathrm{compile}}+W_{\mathrm{eval}}(F).
\end{aligned}
\tag{12}
$$

Both routes produce $FHW$ samples and pay for real shading interactions.
World Tubes targets the repeated projection, conservative support, tile
membership, visibility metadata, and corresponding backward tape. For a fixed
bounded camera program whose chart and event structure remains stable as time
sampling is densified, the persistent atlas state can remain fixed while replay
metadata grows with $F$. Near poles, support churn, dense order crossings, or
fallback-heavy regions can instead make the compiled structure grow. We claim
conditional camera-program amortization, not a universal sublinear bound on
end-to-end rendering.

## 4. Implementation

Our implementation retains the STAR UVT/projective-interval execution lineage
but changes the compiler semantics. The compiler stores homogeneous trace
coefficients, optional temporal-opacity and spatial-precision terms,
conditional-depth information, active-time intervals, tile membership, and
certified cell order. Accepted cells are packed once into tile-time bins and
evaluated by an interval Metal forward pass. A custom autograd boundary applies
the direct interval VJP and then the differentiable compiler map of Eq. (11).

Support, bin membership, interval splits, order decisions, and fallback labels
are discrete during one adjoint block. Trace coefficients, Gaussian source
parameters, opacity, precision, temporal terms, and appearance remain
differentiable. The implemented moving-camera source is bounded to one
first-order or projective chart at a time; complete repeated-orbit chart
transitions are outside the claim. The retained-depth branch currently covers
the scoped affine Gaussian fallback rather than nonlinear projective fibers.

The release supplement will record source parameterizations, amplitude
conventions, Beer--Lambert support and VJPs, compiler tolerances, kernel entry
points, and the exact fixed-topology refresh policy. These details are omitted
here because they do not change the causal comparison: replay and compilation
start from the same world and differ only in whether world-to-trace work is
repeated per frame.

## 5. Experiments

Verification tests and package checks guard the artifact contracts; they are
not ablation rows and cannot substitute for the missing frozen-world,
moving-camera-density, or public runs. The bounded variable-camera sweep below
is a real accepted ablation imported from unchanged clean source, not a test.

### 5.1 Certified bounded correctness

The certified suite derives its rows from byte-pinned source artifacts without
loading the training runtime. It tests gauge changes, replay equivalence,
visibility crossings, exposure and rolling-shutter evaluation, mixed fallback,
and fixed-topology gradients. Table 1 reports the accepted bounded scope; no
timing row is inferred from these fixtures.

<!-- GENERATED-TABLE:theorem_table.tex
The venue build inputs the verifier-produced theorem table at this exact
location. Do not hand-copy numeric rows into the manuscript source.
-->

The gauge ablation is especially diagnostic. Across five sensor-time points,
ordinary- and log-depth integration agree to at most
$3.50\times10^{-13}$ relative error with the Jacobian; removing it gives at
least $0.600$ relative error. Gradients agree to $2.33\times10^{-12}$ with the
Jacobian, while the no-Jacobian variants err by at least $0.592$. The
visibility ablation isolates a different failure: one unstratified interval
produces $0.186742$ error at an order crossing, whereas event stratification
reproduces the live-order oracle exactly on the fixture.

<!-- GENERATED-TABLE:variable_camera_table.tex
The venue build inputs the verifier-produced variable-camera table at this
exact location. Its compatibility receipt binds the clean schema-v2 raw SHA,
paired source commits, implementation-source manifest, and complete handoff
SHA receipts. Do not decode it with the current schema-v1 runner.
-->

The clean-source bounded sweep certifies all 11 evaluated camera programs
through a $170^\circ$ half-span ($340^\circ$ total open-path yaw), with zero
fallback and invalid-sample fractions. Across those closure rows, the minimum
image PSNR is $81.848$ dB and the maximum world-VJP relative error is
$4.854\times10^{-4}$. At a $179.5^\circ$ half-span ($359^\circ$ total yaw),
the compiler leaves two endpoint charts unresolved under its depth-residual
certificate and stops before lowering or rendering them. This is a witnessed
compiler boundary, not a fabricated quality measurement and not a claim of a
closed $360^\circ$ transition.

![Clean-source bounded variable-camera closure/death sweep. The dashed line is
the terminal compiler-certificate death at a $179.5^\circ$ half-span; image
and VJP values are intentionally absent for that unresolved
row.](research_notes/gauged_uvt_trace_atlas/paper/figures/world_tubes_variable_camera_closure_death_publication.svg){#fig:variable-camera-closure-death width=100%}

### 5.2 Same-representation frame scaling

The structural fixture evaluates $F\in\{4,8,16,32,64,128\}$. Its fixed trace
tensor volume remains constant while the per-frame replay construction grows
by $32\times$ from the first to the last row; the final trace-count ratio is
$0.03125$. This is evidence of structural reuse, not a runtime, storage, or
peak-memory claim: logical tensors exclude packed topology, allocator
overhead, transient buffers, and unavoidable pixel interactions.

The pending causal learned-scene protocol requires one checkpoint
$\theta^\star$, camera program, target sample set, precision, background,
shading law, loss, and pixel batch to remain fixed. Replay must lower that
world independently for every selected time, whereas World Tubes must compile
the same world once for the complete bounded program. Both routes must
evaluate identical targets over the same physical interval and return
gradients to the same world parameters.

<!-- ARTIFACT-GATE:frozen_world_scaling
Insert the main scaling figure and numeric table only when one canonical report
contains accepted rows for F={4,8,16,32,64,128,full}, a shared checkpoint hash,
selected-time parity, at least one warmup and three raw timing repeats per row,
route-scoped peak-memory samples, topology-inclusive retained bytes, fallback
fractions, and publication eligibility. Report compile, forward, backward,
total, break-even F, and uncertainty. Do not use historical single-shot fixture
timings or logical tensor volume as a memory surrogate.
-->

### 5.3 Public evaluation contract

Neural 3D Video supplies the positive calibrated multiview setting
[@li2022_neural3dvideo]; D-NeRF is a labelled posed-frame control
[@pumarola2021_dnerf], not evidence for multi-camera amortization. The primary
public compiler comparison is the frozen same-world protocol above. A separate
matched-schedule matrix compares World Tubes, WorldFoam, and dynamic 3DGS as
representation-and-cost context, not compiler causality. Its evaluator fixes
inputs, split, background, and metric aggregation, and reports storage,
rasterized samples, peak memory, and wall time alongside image quality.

<!-- ARTIFACT-GATE:public_context
Insert public tables/figures only when all seven core schema-v2 rows are
accepted: progressive seeds 17/29/43, fixed pixel-matched seeds 17/29/43, and
global-shuffle seed 17. The main table should show mean and dispersion for the
three-seed protocols plus the labelled sampler control. Additional camera
triplets, Neural3D scenes, and D-NeRF controls belong in the supplement. Never
render partial rows or schema-v1 values in this source.
-->

### 5.4 Central ablations

The bounded evidence separates three mechanisms from a generic “faster STAR”
result. Retaining the gauge Jacobian changes value and gradient invariance by
more than eleven orders of magnitude; event stratification removes the
observed depth-order-crossing error; and fixed trace state reduces the
$F=128$ trace count to $3.125\%$ of replay without claiming that output
shading disappears. The still-required clean-source closure and learned-scene
artifacts will add the affine/projective, runtime, backward, peak-memory, and
break-even ablations only after their corresponding gates accept them.
Ordered transfer counts as a positive result only where its certificate is
selective; all-retained routing remains a negative control.

## 6. Discussion and limitations

World Tubes targets known or low-dimensional camera programs with repeated
temporal samples. Random views offer no amortization; visibility chaos,
near-camera splats, wide fields of view, or support churn may erase it.

The claim is local: the exact Schur complement requires a regular affine chart,
projective traces cover bounded certified segments rather than one complete
$360^\circ/720^\circ$ chart, and the adjoint is exact inside a fixed discrete
topology. Event boundaries require refresh or a declared estimator. Conditional
Gaussian depth does not represent arbitrary interleaved colored density;
ambiguous cells may use ordered transfer, while general cellular fields and
adaptive retained-depth quadrature remain outside the central claim.

## 7. Conclusion

World Tubes compiles dynamic Gaussian scenes into gauge-correct,
event-certified traces whose fixed-topology adjoint is designed to share
world-side work across time. Certified fixtures establish invariance, failure
detection, gradient parity, and structural reuse.

## Reproducibility statement

The repository records checked-in protocols, deterministic seed lists,
dataset and split contracts, and Torch-free evidence verifiers. Each accepted
runtime bundle is additionally required to bind source and native-binary
hashes, isolated-run summaries, raw timing samples, and retained artifacts.
The pending causal comparison must bind replay and compiled routes to one
checkpoint and one camera program. The supplement will enumerate compiler
tolerances, parameter conventions, evaluation formulas, hardware, warmup and
repeat policy, and the commands that regenerate every accepted table and
figure.

<!-- REQUIRED-GATE:ai_use_statement
ICLR 2027 requires a separate AI-use statement. Before submission, add
author-approved wording that accurately discloses the use of generative-AI
tools for code assistance, experiment orchestration, artifact verification,
mathematical review, and manuscript editing, together with the authors'
responsibility for all code, proofs, experiments, and claims. This comment must
cause the submission-package verifier to fail until the statement exists.
-->

## References

::: {#refs}
:::

```{=latex}
\appendix
```

## A. Ordered ray transfer at the noncommutation boundary

One scalar depth order is inadequate when thick, differently colored density
profiles interleave along a ray. The scoped retained-depth extension represents
this case by the ordered optical-transfer product

$$
  M_y=\mathcal P\exp\!\left(\int A_y(z)\,\mathrm dz\right),\qquad
  A_y(z)=
  \begin{bmatrix}
    -\sigma_y(z)I_3 & \sigma_y(z)c_y(z)\\
    0 & 0
  \end{bmatrix}.
  \tag{13}
$$

The product integral is invariant under an orientation-preserving depth
reparameterization when its physical-length Jacobian is included. Here it is a
bounded fallback for Gaussian depth fibers, not a replacement for the fast
certified compositor and not a claim about a general cellular field. The full
supplement must include the transfer derivation, VJP, certificate, selective
fixture, and dense-scene negative control.

## Supplementary-material roadmap

The supplement should preserve, rather than delete, the details removed from
the nine-page main-paper budget:

1. notation, camera-ray bundle construction, and gauge-invariance proof;
2. full SPD(4) covariance and equivalent precision derivations;
3. amplitude conventions and Beer--Lambert support/VJP derivations;
4. projective trace coefficients, continuous denominator certificate, and
   compiler pseudocode;
5. visibility commutation bound and ordered ray-transfer derivation;
6. fixed-topology adjoint proof and complete work accounting;
7. source parameterizations, kernel interfaces, topology-refresh policy, and
   numerical tolerances;
8. dataset contracts, splits, schedules, timing and memory methodology;
9. full closure/death diagnostics, all public rows, additional scenes and
   seeds, qualitative examples, and failure-bearing negative controls.
