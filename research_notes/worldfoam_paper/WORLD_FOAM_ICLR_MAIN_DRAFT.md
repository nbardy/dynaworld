---
title: "WorldFoam: Camera-Compiled Ordered Ray Transfer for Dynamic Cellular Radiance Fields"
author: Anonymous
bibliography: research_notes/worldfoam_paper/WORLD_FOAM_REFERENCES.bib
link-citations: true
---

<!--
SUBMISSION-SOURCE POLICY

This is the concise venue-facing Paper-B source. The long mathematical and
engineering dossier remains in WORLD_FOAM_PAPER_DRAFT.md; proofs and speculative
branches remain in WORLD_FOAM_MATH_APPENDIX.md and the dated synthesis notes.
Do not expand this manuscript by copying the lab notebook into it.

The submission build must fail, rather than print an empty or smoke-derived
result, unless the following evidence components are publication eligible:

G0 numerical foundation:
  the independently verified fixed-segment value/VJP artifact is source-bound
  and its stated scope remains fixed-segment numerical correctness only;
G1 native systems parity and work:
  staged_sparse and fused_union_v2 agree at F=8, and native
  same-representation timing reports end-to-end wall time and ordered-world
  work;
G2 native full-geometry trainability:
  material and geometry gradients plus one optimizer update agree at F=8 in
  three fresh processes; requested-density rows share one world, camera
  program, track set, physical interval, compiler ranks, owner words, and
  active-block fingerprint;
G3 representation-level visibility stress:
  the accepted float64 CPU S1--S8/C1--C7 suite covers dense reference,
  retained-depth evaluation, depth-collapsing controls, crossing diagnostics,
  and adaptive fallback; it does not certify the kinetic compiler or native
  execution;
G4 public quality:
  held-out public-data results exist against same-representation replay, World
  Tubes, dynamic Gaussian, and relevant cellular baselines;
G5 backend scope:
  official CUDA/Warp parity is not claimed; any future parity statement must
  use the official fixture and independently accepted backend evidence;
G6 measured memory/work:
  outputs/worldfoam_training_memory_ablation/
  worldfoam_training_memory_ablation.json contains all 21 required measured
  fresh-process rows (seven mode/frame rows times three repeats), passes the
  fail-closed verifier, and reports process RSS, public MPS allocator peaks,
  logical state, work counters, traffic proxies, and wall time together.

The venue package is a separate submission requirement: the official style,
portable figure exports, citation audit, author-approved AI-use statement, and
rendered-PDF inspection must all be complete.

G6 is the gate for saying that WorldFoam fits the target memory envelope. Byte
formulas, source inspection, the two-site material fixture, a stale schema-v2
artifact, or a sequential-control OOM may not substitute for it. The fair
control is same-representation sequential per-frame replay: it may also have
O(1) peak memory, but it repeats ordered world work O(F).

Do not insert native-memory, native-speed, or public-quality numbers in the
abstract, tables, captions, or conclusion until the corresponding gate passes.
Do not call open ray transport "holonomy"; reserve holonomy for closed loops.
-->

## Abstract

Dynamic volume renderers must preserve depth order when differently colored
matter overlaps. Marginalizing the ray-depth coordinate can be exact for a
Gaussian footprint, but it discards the ordered optical profile required by a
general cellular medium. We introduce **WorldFoam**, a dynamic cellular
radiance representation compiled along a known camera program. A camera gauge
pulls extinction and emitted radiance onto ordered ray-depth fibers with the
physical ray-length Jacobian. Each stable cell path becomes a front-to-back
word whose exact action on rear radiance is the affine transfer
$(\beta,m)$, while a translated optical-depth measure retains the ordered
profile needed for proofs and geometry gradients. A certified atlas reuses
owner words and transfer nodes only within stable event strata; chart splits
and explicit fallback preserve semantics when order or support cannot be
certified. Its adjoint first reduces sample cotangents into compiler-node
cotangents and then reverses each ordered word once, separating shared
world-side work from the unavoidable sample and output slice. This formulation
defines a precise same-representation ablation against sequential per-frame
replay and a fail-closed boundary for event, chart, and rank changes. On a
source-bound float64 CPU suite spanning eight procedural scenes and seven
camera programs, a 128-layer retained-depth evaluator attains 37.9252 dB
fifth-percentile PSNR against a dense reference; separate analytic receipts
cover the constant-density and gauge-change checks. On the two crossing
families, its mean RGB MSE is 82.2477$\times$ and 528.953$\times$ lower than
representative-depth sorting and depth marginalization, respectively; using
the physical gauge Jacobian reduces maximum RGB error from 0.305335 to
$3.32998\times10^{-7}$. These CPU ablations do not establish end-to-end
kinetic-compiler parity, native runtime or memory scaling, or trained
public-data quality.

<!-- ARTIFACT-GATE:abstract-result
The CPU G0/G3 correctness result above is source-bound and publication
eligible. Add a matched memory/work result only after G1, G2, and G6 are
publication eligible. Add public quality only after G4.
-->

## 1. Introduction

Three-dimensional Gaussian splatting is effective because it combines a
compact learned world with projection, tile culling, ordered alpha compositing,
and differentiable rasterization [@kerbl2023]. Dynamic variants attach motion,
deformation, or native spacetime structure to the primitives
[@wu2024_4dgs; @yang2024_deformable; @li2024_spacetime]. Their image formation
still encounters a basic visibility problem: two translucent contributors with
different colors do not generally commute, and their front-to-back order can
change under object or camera motion. Finer sorting improves consistency
[@radl2024_stopthepop], while weighted or modified blending changes the
transfer approximation [@mcguire2013_weighted_oit; @hou2025_sortfree;
@koo2026_blending].

WorldFoam takes a different representation boundary. The world is a moving,
bounded cell complex with extinction and radiance fields. A ray intersects a
sequence of cell intervals, not an unordered set of screen-space footprints.
Beer--Lambert transport composes these intervals as noncommuting affine actions
on rear radiance [@tagliasacchi2022_volume]. This retains the exact phenomenon
that a depth marginal would erase.

The challenge is to retain order without storing a dense frame-by-cell tape.
For a fixed world and camera program, neighboring sensor-time samples often
share event topology and admit a common temporal surrogate. We compile that
structure into an atlas of stable owner words, transfer nodes, certificates,
and sparse lowering rules. Requested samples evaluate the atlas and accumulate
into node cotangents; the expensive ordered-word reverse is performed at the
nodes rather than replayed independently at every requested time. This is a
structural work claim, not permission to hide output, target, ray, or sample
costs.

![World Tubes marginalizes ray depth when Gaussian closure is useful; WorldFoam retains an ordered depth profile and its exact affine transfer. The figure states a target contract, not a measured result.](research_notes/worldfoam_paper/figures/worldfoam_representation_split.svg){#fig:representation-split width=100%}

Our contributions are:

1. A gauge-covariant ray-fiber formulation for dynamic cellular matter in
   which the physical coefficient $\lambda\,dz$ is invariant under admissible
   depth reparameterization.
2. An order-explicit translated optical-depth measure and its exact compact
   affine quotient $(\beta,m)$ for emission--absorption transfer.
3. A certified camera-program atlas that shares owner words and transfer nodes
   only inside stable strata and makes split/fallback behavior part of the
   method.
4. A shared adjoint factorization that reduces sample cotangents to temporal
   nodes before one ordered-word reverse per node, together with a fair
   same-representation sequential replay control.
5. An artifact-gated evaluation protocol that separates algebraic correctness,
   topology correctness, structural work, measured memory, and public quality.

## 2. Related work

**Gaussian and dynamic representations.** Gaussian splatting learns
anisotropic primitives rendered by projection and ordered alpha compositing
[@kerbl2023]. Dynamic Gaussian methods add deformation, trajectories, temporal
features, or spacetime covariance [@wu2024_4dgs; @yang2024_deformable;
@li2024_spacetime]. WorldFoam is not another Gaussian parameterization. It
keeps a cellular partition and integrates optical matter along ray intervals.

**Cellular radiance fields.** Tetra-NeRF uses a Delaunay tetrahedralization as
an explicit radiance-field scaffold [@kulhanek2023_tetranerf]. Radiant Foam
uses a Voronoi partition and adjacency-based differentiable ray traversal
[@govindarajan2025_radiantfoam]. Power Foam introduces bounded power cells that
support both ray tracing and rasterization [@govindarajan2026_powerfoam]. These
works establish important static cellular backends. WorldFoam asks a different
question: when a cellular world and a camera program are dynamic, which ordered
ray structure can be compiled and differentiated across requested times?

**Visibility and transfer.** Accurate front-to-back sorting and
order-independent approximations address different failure modes
[@radl2024_stopthepop; @mcguire2013_weighted_oit; @hou2025_sortfree;
@koo2026_blending]. WorldFoam keeps the ordinary emission--absorption law and
therefore preserves its noncommutative color ordering. It may split a temporal
chart at an order event; it does not declare the event irrelevant.

**Dynamic view synthesis.** Public evaluation follows calibrated multi-view
video and controlled dynamic-scene protocols such as Neural 3D Video and
D-NeRF [@li2022_neural3dvideo; @pumarola2021_dnerf]. Matching their quality is
an empirical gate, not a consequence of the transfer algebra.

## 3. WorldFoam

### 3.1 Ray bundle, camera gauge, and physical measure

Let $B=\Omega\times\mathcal T$ be the sensor-time base, with
$b=(u,v,t)$. A known camera program defines a clipped ray bundle
$\pi:E_\Gamma\rightarrow B$ and a world map

$$
\Gamma:E_\Gamma\rightarrow\mathbb R^3\times\mathcal T.
\tag{1}
$$

A local gauge $a$ identifies a regular bundle domain with coordinates
$(b,z_a)$. Writing $\Gamma_{x,a}$ for the spatial part, a world extinction
density $\rho$ pulls back to the coordinate extinction

$$
\lambda_a(b,z_a)
=\rho\!\left(\Gamma_a(b,z_a)\right)
 \left\|\partial_{z_a}\Gamma_{x,a}(b,z_a)\right\|.
\tag{2}
$$

If $z_b=h(b,z_a)$ is orientation preserving and covers the same clipped
physical segment, then

$$
\lambda_b(b,z_b)\,dz_b=\lambda_a(b,z_a)\,dz_a.
\tag{3}
$$

The optical transfer is therefore independent of the chosen depth coordinate.
Equation (3) does not license crossing a singular gauge, changing the clipped
segment, or reusing an owner word across an uncertified event. Those cases
require a chart transition, split, or fallback.

### 3.2 Dynamic cells and the ordered owner word

Let a dynamic foam contain bounded cells $C_i(t)$ with parameters $\theta_i$,
extinction $\rho_i(x,t)\ge0$, and color $c_i(x,t)\in\mathbb R^3$. A useful
instance is a kinetic power diagram intersected with bounded supports, but the
transfer construction only requires certified ray--cell intervals and owner
identity.

On a regular ray fiber, intersections produce a near-to-far word

$$
w_b=\bigl((i_r,z_r^-,z_r^+,\rho_r,c_r)\bigr)_{r=1}^{R_b}.
\tag{4}
$$

For a piecewise-constant segment, its physical length is
$\ell_r=\int_{z_r^-}^{z_r^+}
\|\partial_z\Gamma_x\|\,dz$ and its optical depth is
$\tau_r=\rho_r\ell_r$. Higher-order positive segment laws may replace this
local material model without changing the word-level transfer ABI. Owner
identity, interval endpoints, and provenance remain attached because equal
rendered transfers can still have different geometry derivatives.

### 3.3 Ordered affine transfer

Represent the action of a front interval on rear radiance $q$ by

$$
T(\beta,m):q\mapsto m+\beta q,
\qquad 0<\beta\le1,
\tag{5}
$$

where $m\in\mathbb R^3$. For a constant-color segment,

$$
\beta_r=e^{-\tau_r},
\qquad
m_r=(1-\beta_r)c_r.
\tag{6}
$$

Near-to-far composition is

$$
(\beta_1,m_1)\otimes(\beta_2,m_2)
=\left(\beta_1\beta_2,\;m_1+\beta_1m_2\right).
\tag{7}
$$

This associative product is generally noncommutative in $m$. The final pixel
is $I=m+\beta I_{\rm bg}$. A balanced scan is possible, but an exact two-pass
front-to-back implementation is important for memory: the backward can
recompute one current prefix from the final transfer instead of retaining a
suffix or reverse record for every run.

### 3.4 Translated optical-depth measure

The compact transfer in Eq. (7) acts correctly on any rear radiance but forgets
which ordered word generated it. To make ordering explicit, define

$$
K_0=0,\qquad K_r=\sum_{q\le r}\tau_q,\qquad \kappa=K_R,
\tag{8}
$$

and the vector measure

$$
d\nu(s)=c_r\,ds,
\qquad s\in[K_{r-1},K_r).
\tag{9}
$$

Concatenating a rear word translates its measure by the front optical depth.
The Laplace map

$$
\mathcal L(\kappa,\nu)
=\left(e^{-\kappa},\int_0^\infty e^{-s}\,d\nu(s)\right)
=(\beta,m)
\tag{10}
$$

is a monoid homomorphism into Eq. (7). It is deliberately non-injective:
$(\beta,m)$ is the exact four-scalar RGB executor state, whereas
$(\kappa,\nu)$ and the owner word retain the depth-resolved information needed
to reason about color-order changes and parameter attribution.

For a stable word, moving an interface creates boundary-supported tangent mass
proportional to the color jump across that interface. Thus a compact primal
transfer does not imply that geometry gradients can discard the owner program.
WorldFoam compresses the executor state while preserving the sparse program
needed for its adjoint.

### 3.5 Stable-stratum camera-program atlas

Over a bounded camera program, interval endpoints and owner order vary until an
event predicate reaches a root. The compiler partitions sensor-time tracks into
charts on which the owner word, support, event orientation, and selected
temporal surrogate are certified. Each compiled block stores:

$$
\begin{aligned}
&\text{tracks, chart bounds, temporal nodes, and owner words},\\
&\text{sparse world attribution and fallback metadata}.
\end{aligned}
\tag{11}
$$

A simple root can terminate one stable chart and begin another. Simultaneous,
ill-conditioned, or unresolved events fail closed. WorldFoam does not
differentiate through the discrete selection of event roots, charts, temporal
ranks, or fallback routes; it differentiates the fixed compiled surrogate
inside an accepted stratum.

![A camera gauge lifts sensor-time samples to ordered ray-depth fibers. Stable owner words lower through an order-explicit measure to an exact affine transfer. Event roots split charts; unresolved domains fall back. Sample cotangents reduce to compiler nodes before ordered-word reversal. This schematic states the compiler contract; the accepted CPU suite does not certify end-to-end kinetic compilation.](research_notes/worldfoam_paper/figures/worldfoam_ray_fiber_atlas.svg){#fig:ray-fiber-atlas width=88%}

### 3.6 Shared adjoint and honest work boundary

Let block $b$ have $J_b$ temporal nodes and $W_b$ ordered owner/run entries
across its tracks. Let $g_j$ be the transfer at node $j$ and $w_j(t_f)$ the
fixed temporal reconstruction weights. Sample cotangents reduce to node
cotangents as

$$
\bar g_j
=\sum_f w_j(t_f)
  D\operatorname{decode}(g(t_f))^\top\bar I_f.
\tag{12}
$$

The compiler then performs one ordered-word VJP per node and scatters the
result through sparse owner attribution. The dominant ordered world work is

$$
W_{\rm world}=\Theta\!\left(\sum_b J_bW_b\right),
\tag{13}
$$

for a fixed certified program. Requested sample evaluation, ray generation,
target reads, residuals, outputs, and the sample-to-node reduction remain
linear in the number of requested observations. No theorem turns an
$F\times H\times W$ video into constant work.

The fair control is not a dense all-frame tape. It is the same WorldFoam world
and transfer law replayed one requested frame at a time, with each frame's tape
released after its gradient reaches one global bar. That control can also keep
peak memory $O(1)$ in $F$, but it repeats the ordered world forward and reverse
$O(F)$ times. The paper therefore evaluates memory and world work separately.

### 3.7 Event and update boundary

Within a fixed accepted chart, gradients flow through material parameters,
segment lengths, cell trajectories, weights, and camera rays supported by the
compiled surrogate. A geometry or camera update that invalidates a certificate
requires recompilation. Reusing a stale owner word across a crossing would
produce a smooth but wrong gradient; recertification is part of the optimizer
contract, not an implementation detail.

A flow-covariant optical connection is a possible future compression of
coherent temporal change. It is not part of the main runtime claim here. It
must first beat the direct transfer atlas under equal capacity, equal primal
error, and equal tangent error; small curvature alone is not a systems result.

## 4. Implementation

The source implementation separates four interfaces:

1. a CPU reference compiler that discovers and certifies kinetic owner charts;
2. compact native-shaped blocks containing post-certification active owners,
   node transfers, and sparse world attribution;
3. a selected-sample evaluator and sample-to-node reduction; and
4. staged and fused ordered-word VJPs that lower node bars to material and
   geometry bars.

Material segments share a parameterized ABI returning optical depth, affine
transfer, bounds, and an explicit VJP. Constant P0 segments define the primary
paper path. Richer positive-polynomial and log-polynomial segments are basis
ablations, not separate renderer forks.

The fused reverse removes an exposed node-by-run length-cotangent buffer by
assembling material and kinetic-world bars inside the accepted block. It does
not remove the primal owner words or node lengths. Targets and rays are read in
bounded selected-sample chunks. Completion fences delimit when block-local
device state, readback receipts, and request state may be released.

These interfaces are present in source, but their rebuilt native
full-geometry parity, end-to-end work, and peak-memory behavior remain the
unmeasured G1, G2, and G6 experiments.

<!-- ARTIFACT-GATE:implementation-status
Visible implementation claims may say which source interfaces exist. They may
not say native parity, memory-fit, speedup, or public-data readiness until the
corresponding source-bound artifacts pass G1--G6.
-->

## 5. Experiments

The experiments isolate five questions that are often conflated: Is the local
transfer exact? Is a compiled chart topologically correct? Does the shared
adjoint avoid repeated ordered world work? Does the complete trainer fit under
a measured memory envelope? Does the representation learn competitive public
novel-view quality?

Verification tests and package checks guard these contracts; they are not
ablation rows and cannot fill missing G1, G2, G4, or G6 evidence.

### 5.1 Numerical and representation-level visibility correctness

The numerical foundation compares every selected segment material against an
independent high-precision integral and compares analytic VJPs against both an
explicit oracle and finite differences. The accepted representation-level
suite below uses constant density, differently colored crossings, thin
occluders, moving bounded cells, finite exposure, and rolling shutter. It
compares retained-depth integration against dense ray integration and two
depth-collapsing baselines. End-to-end kinetic chart discovery, compiled-atlas
parity, and native execution are separate experiments and are not certified by
this artifact.

The accepted G0/G3 CPU suite evaluates all $8\times7=56$ scene--camera
conditions at 13 times and 17 sensor locations. Its dense oracle uses 2,048
depth samples per ray; the constant-density sphere and gauge-change checks also
have analytic references. Table 1 reports aggregates recomputed by the
independent verifier from the source-bound JSON, rather than training metrics
or diagnostic CPU timing.

<!-- GENERATED-TABLE:synthetic_visibility_table.tex
The venue build inputs the compact verifier-derived G0/G3 table here. Do not
hand-copy numeric result cells into this manuscript source.
-->

![For four representative synthetic scenes (S2--S5), each point is the median over seven camera programs. RGB error against the 2,048-sample float64 CPU oracle decreases with retained-depth resolution; this is not a public-data quality or native-performance curve.](outputs/benchmarks/2026-08-15_worldfoam_synthetic_visibility_cpu/figures/worldfoam_synthetic_depth_convergence.svg){#fig:synthetic-depth-convergence width=92%}

![For each of seven camera programs, each curve averages the two differently colored crossing families. The 128-layer retained-depth evaluator has lower temporal flicker than representative-depth sorting and depth marginalization in this synthetic float64 CPU suite.](outputs/benchmarks/2026-08-15_worldfoam_synthetic_visibility_cpu/figures/worldfoam_synthetic_crossing_flicker.svg){#fig:synthetic-crossing-flicker width=92%}

![The representation-level coarse-to-fine policy routes a substantial, camera-dependent fraction of synthetic rays to the 128-layer fallback; each point averages all eight scenes. The nominal-speed ordering is descriptive rather than evidence of a monotone speed effect. Fallback is reported work, and this plot does not certify an end-to-end kinetic compiler.](outputs/benchmarks/2026-08-15_worldfoam_synthetic_visibility_cpu/figures/worldfoam_synthetic_adaptive_fallback.svg){#fig:synthetic-adaptive-fallback width=92%}

The accepted
[source-bound synthetic visibility artifact](outputs/benchmarks/2026-08-15_worldfoam_synthetic_visibility_cpu/summary.json)
records the complete protocol and source SHA-256 digests. Their short display
prefixes are `e9e184ca56a5` and `dd77886d4979`; the verifier checks all 64 hex
digits rather than these manuscript abbreviations.

### 5.2 Same-representation memory/work ablation

The predeclared, not-yet-measured G6 protocol fixes one $384\times512$
camera program, a $32\times32$ kinetic world, 512 selected pixel tracks, four
compiler nodes, and the 300-time physical grid. Requested subsets use
$F\in\{8,64,300\}$.
All 1024 sites remain eligible during certification; compact device blocks are
formed only after active owners are witnessed.

The current evidence ledger contains $0/21$ measured G6 rows. The protocol
below is therefore an experimental contract, not a memory-fit result.

The protocol requires seven mode/frame combinations, each repeated in three
fresh processes. They form three route families:

| Route | Requested frames | Purpose |
|---|---:|---|
| staged sparse reverse | 8 | parity oracle for the fused reverse |
| fused shared reverse | 8, 64, 300 | matched parity and requested-density scaling |
| sequential same-representation replay | 8, 64, 300 | fair repeated-world-work control |

Each accepted row must perform one real optimizer mutation. The artifact must
report compile, forward, backward, optimizer, and wall time; process RSS; public allocator
current and driver peaks; retained logical bytes; target/ray traffic; sample
interactions; node-forward work; ordered-word forward/reverse work; structural
fingerprints; loss; gradient parity; and parameter-update parity. A failed
resource guard is a failed measured row, not a silently censored control.

The hypothesis is deliberately two-part:

$$
\begin{aligned}
\text{compiled:}&\quad
  W_{\rm world}\propto\sum_bJ_bW_b,\\
\text{sequential replay:}&\quad
  W_{\rm world}\propto F,\\
\text{both:}&\quad
  M_{\rm peak}=O(1)\text{ if frame scratch is released}.
\end{aligned}
\tag{14}
$$

<!-- ARTIFACT-GATE:G1-G2-G6-TABLE
The final systems table must be generated from
outputs/worldfoam_training_memory_ablation/
worldfoam_training_memory_ablation.json after the strict verifier passes.
Insert all 21 measured rows or a documented aggregate whose raw rows and
fresh-process receipts remain linked. Never replace missing allocator samples
with logical byte formulas.
-->

### 5.3 Material-basis ablation

Constant, affine-color, positive-polynomial, and convex log-polynomial segment
families use one payload/ABI and disjoint train/held-out chords. The experiment
tests whether one basis dominates or whether basis selection must be adaptive.
It does not justify native integration merely because a basis fits the family
that generated its synthetic target.

![At the same six-scalar payload, M3 and M5 each fit its own generating family
while losing decisively on the other held-out family. The result establishes
complementary material capacity, not renderer speed, public quality, or native
memory.](research_notes/worldfoam_paper/generated/foundation_v1/material_family_loss.svg){#fig:worldfoam-material-family-loss width=88%}

The matched held-out gate therefore rejects a universal material winner.
Adaptive per-cell selection or real held-out material observations are required
before either richer law can replace the constant-density path in a native
renderer.

### 5.4 Public quality and visibility pathology

The not-yet-measured G4 public protocol requires full calibrated multi-view
sequences with held-out cameras, at least one controlled synthetic dynamic
set, seeds fixed before inspection, and identical image metrics across routes.
Its baselines will include dynamic Gaussian replay, World Tubes,
static/streamed cellular controls where applicable, same-representation
WorldFoam replay, and compiled WorldFoam.

The native crossing-opacity extension must separately measure transmittance
error, RGB error, temporal flicker, order flips, gradient continuity inside
strata, split frequency, fallback fraction, active word length, and event
density. Together with G4, it decides whether the accepted CPU benefit
survives the complete compiler and earns its extra complexity.

<!-- ARTIFACT-GATE:G4-PUBLIC-TABLE
Do not create a public-quality row from a smoke, training-view metric, repeated
frames, old microgate, or external paper number with a mismatched protocol.
Require scene/split/seed/config/checkpoint/artifact provenance for every row.
-->

## 6. Discussion and limitations

WorldFoam is useful only where retaining ordered depth changes the answer or
the cost structure. For thin opaque surfaces or scenes already modeled well by
Gaussian splats, World Tubes is the simpler primary method. For thick,
differently colored overlap, a single representative depth can be inadequate,
and the cell-path transfer gives a more faithful object.

The formulation has hard boundaries. A point partition does not bound the
number of cells stabbed by a ray. Event and chart complexity can grow with
motion duration, cell count, and degeneracy. The compiler currently treats
event, chart, rank, and fallback choices as discrete. Geometry steps may force
recertification. The exact four-scalar transfer is not an exact replacement for
owner topology in geometry gradients. Finally, frame-density-independent
ordered world work does not remove the linear camera, target, sample, shading,
and output slice.

These boundaries make the ablations decisive. If compiled and sequential
replay have comparable world work, the compiler has not earned its complexity.
If measured memory does not remain inside the declared guard, analytic state
counts are insufficient. If retained depth does not improve the visibility
stress case or public held-out quality, WorldFoam should remain a reference
renderer rather than become the production lane.

## 7. Conclusion

WorldFoam formulates dynamic cellular rendering as gauge-covariant ordered
transfer on camera ray fibers. The translated optical-depth measure preserves
the ordered profile; its affine quotient gives an exact compact executor; and
a stable-stratum atlas exposes where temporal structure can be shared without
crossing visibility events. The shared adjoint targets repeated ordered world
work while retaining the unavoidable linear sample slice. The paper's systems
and quality conclusions remain tied to matched, source-bound ablations rather
than to representation-level byte formulas.

<!-- ARTIFACT-GATE:conclusion-result
Add one sentence of measured evidence only when its exact table, artifact, and
verifier are included in the submission package.
-->

## Reproducibility statement

The release package will include fixed configs, dataset/split manifests,
compiler and native-source digests, fresh-process receipts, raw per-repeat
rows, artifact verifiers, and scripts that regenerate every table and figure.
Concept figures are deterministic and result-free. Measured figures and tables
must be generated from accepted JSON rather than edited by hand.

<!-- AI-USE-GATE
Replace this comment with the exact author-approved venue disclosure. Do not
guess the venue policy or auto-generate a disclosure on the authors' behalf.
-->

## Appendix A. Proof sketches

### A.1 Gauge covariance

For $z_b=h(b,z_a)$ with positive derivative, physical arclength satisfies

$$
\left\|\partial_{z_b}\Gamma_x\right\|dz_b
=\left\|\partial_{z_a}\Gamma_x\right\|dz_a.
$$

Multiplying by the scalar world extinction yields Eq. (3). The path-ordered
product is therefore unchanged under a regular reparameterization of the same
oriented clipped fiber. A singular chart or changed interval violates the
assumptions and must not be merged by this argument.

### A.2 Affine transfer homomorphism

Applying a front transfer $(\beta_1,m_1)$ to the result of a rear transfer
$(\beta_2,m_2)$ gives

$$
m_1+\beta_1(m_2+\beta_2q)
=(m_1+\beta_1m_2)+(\beta_1\beta_2)q,
$$

which proves Eq. (7). Translating the rear measure by front optical depth
multiplies its Laplace contribution by $e^{-\kappa_1}=\beta_1$, proving Eq.
(10) respects the same product.

### A.3 Shared-adjoint scope

Equation (12) is the chain rule for a fixed temporal reconstruction. Once all
sample contributions have reached $\bar g_j$, reversing the node word once is
exact for that fixed surrogate. The statement does not cover derivatives of
the compiler's discrete choices or prove a physical tangent approximation
bound between nodes; those require separate certificates and ablations.

## References

::: {#refs}
:::
