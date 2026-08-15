# Full concern ledger and build plan

**Date:** 2026-07-23

**Status:** consolidated design and implementation plan; no claim of a completed
full-SPD(4) renderer

**Scope:** canonical spacetime primitive, gauged camera-ray compilation,
depth/visibility, moving cameras, baselines, and memory safety

## Executive verdict

The clean target is not another competing tube representation. It is one
camera-independent world atom and one explicit compilation boundary:

\[
\boxed{
(\mu_{XYZT},\Sigma_4,\text{typed amplitude},\text{appearance})
\longrightarrow
\text{camera-ray pullback}
\longrightarrow
\text{UVT marginal + conditional depth law}
\longrightarrow
\text{certified visibility atlas}
\longrightarrow
\text{Metal replay/fallback}.
}
\]

Much of the **back half** exists. The STAR projective interval path can follow
a known moving/revolving camera program, use pixel- and time-dependent depth,
split at depth-order events, subdivide spatial crossings, and use live-depth
fallback when **visibility order** cannot be certified. Invalid camera/chart
cells are instead subdivided and then rejected/fail-closed. Current
forward/backward and visibility tests pass.

The missing piece is the **front half and its end-to-end continuous derivative
path**: an actively trained full \(XYZT+\operatorname{SPD}(4)\) world atom
lowered into a depth-complete trace, including conditional depth variance and
amplitude normalization, then differentiated back to the world parameters
under an explicit piecewise-visibility policy.

Two qualifications matter:

1. A **depth gauge** is a monotone coordinate change along the same physical
   ray. A moving camera changes the rays themselves, so correct new occlusion
   comes from recompiling/evaluating visibility, not from gauge invariance.
2. Current correctness is relative to a compiled local trace and its candidate
   set, inside certified bounded domains with explicit fallback. It is not a
   one-chart theorem for every arbitrary camera path, and it is not exact
   retained-depth volumetric optical transfer for overlapping colored media.

## Status language

| Label | Meaning here |
|---|---|
| **Exact** | Algebraically proved under the stated assumptions. |
| **Implemented/tested** | Present in source and exercised by focused tests. |
| **Conditional** | Correct inside an explicit chart, certificate, support, or approximation contract. |
| **Prototype** | Real code, but not the canonical trainer/browser path or not broadly accepted. |
| **Missing** | Required for the intended full world-to-image system and not currently present. |

## The remembered moving-camera result

### What works on disk

The strongest implementation is the integrated projective interval lineage in
[`star_uvt_v0`](../../third_party/fast-mac-gsplat/variants/star_uvt_v0), chiefly:

- [`projective_trace.py`](../../third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/projective_trace.py)
  constructs trace cells, support/denominator certificates, temporal roots,
  UV depth-line events, order strata, and fallback decisions;
- [`star_uvt_kernels.metal`](../../third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/metal/star_uvt_kernels.metal)
  evaluates the current affine depth at each pixel/time, selects the live
  front-to-back order, and replays the same order in backward;
- [`tile_metal_autograd.py`](../../third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/trainer_harness/tile_metal_autograd.py)
  owns refresh, staleness checks, rebinning, split/fallback state, and mixed
  rendering;
- [`star_uvt_projective_interval_backend.py`](../../src/train/star_uvt_projective_interval_backend.py)
  is the trainer-side adapter, disabled unless explicitly selected.

For a compiled cell, depth is not one frozen scalar. The Metal path evaluates

\[
d_i(u,v,t)=d_{0,i}(t)
+\beta_{u,i}(t)(u-u_i)
+\beta_{v,i}(t)(v-v_i),
\]

then chooses the order at the current pixel and time. Time crossings are root
split; a depth line inside a tile can cause UV subdivision; unresolved regions
can use live-depth reference fallback. Exposure and rolling-shutter paths use
the same event/fallback model.

### What “correct” means, precisely

The current path is:

- **exact** for monotone reparameterization of a fixed ray when the fiber
  Jacobian is included;
- **implemented/tested** for per-pixel conditional-mean depth sorting and alpha
  compositing relative to supplied trace functions and candidate lists;
- **conditional** for a camera path represented by local polynomial/rational
  charts with support, denominator, residual, and order certificates;
- **piecewise differentiable** while atlas topology, tile membership, and
  discrete order are held fixed;
- **piecewise constant in hard-order depth metadata**: conditional depth affects
  the discrete permutation, so its ordinary image derivative is zero inside a
  fixed order stratum and undefined at a swap; nonzero depth-law gradients need
  a retained/smoothed fiber path or an auxiliary loss;
- **fallback-based** for unresolved visibility ambiguity; denominator,
  residual, near-plane, or support-validity failures are subdivided and then
  rejected/fail-closed, while stale candidate sets are refreshed/rebinned;
- **not** yet an exact full \((\mu_4,\Sigma_4)\)-to-camera compiler;
- **not** a derivative through the instant at which a hard visibility order
  swaps; that event is nonsmooth;
- **not** an exact continuous-depth optical-transfer solver when thick colored
  Gaussian fibers overlap.

The current orbit helper also uses local camera approximations in some paths.
Its certificates establish the compiled approximation's contract; they do not
turn an arbitrary camera program into an exact global rational chart.

### Current verification in this audit

No MPS workload was launched. CPU-only current-source checks produced:

- **46 passed** across bundle gauge invariance, gauge-gradient invariance,
  projective visibility, and visibility-stress tests;
- **44 passed, 3 deselected** across orbit windows, exposure/rolling-shutter
  quadrature/backward, and mixed-fallback backward, with Metal/MPS cases
  explicitly excluded.

Total for these two invocations: **90 passed, 3 deselected**. This supports the
source-level claim above; it does not close broad scene quality, global camera
coverage, or MPS memory acceptance.

## Versions and lineages on disk

The outer repository was at `281471d` and the nested fast-mac tree at
`64a4e0a` during this audit; both worktrees contained existing user changes.

| Path/version | What it actually provides | Honest status and missing boundary |
|---|---|---|
| [`v5`](../../third_party/fast-mac-gsplat/variants/v5), [`v5_features`](../../third_party/fast-mac-gsplat/variants/v5_features) | Ordinary anisotropic 3DGS Metal forward/backward used by current RGB/feature defaults. | Good 3DGS renderer; no intrinsic time. |
| [`v6_refined`](../../third_party/fast-mac-gsplat/variants/v6_refined) and `v7`–`v13` families | Renderer/hardware/feature-gradient optimization experiments. | Mostly renderer experiments, not distinct spacetime scene representations. |
| `FreeDynamic3DGS` in [`train_splat_baseline.py`](../../research_experiments/gauge_fields/train_splat_baseline.py) | Independent trainable XYZ, log scales, quaternion, opacity, and RGB at each frame, reprojected and sorted for that camera/frame. | Fair generous per-frame quality ceiling; \(O(TN)\) resident model state. |
| [`spacetime_v0`](../../third_party/fast-mac-gsplat/variants/spacetime_v0) | Intended full world `float4` mean/full `4x4` precision and CPU affine ray-depth reference. | Canonical mathematical ancestor; world-to-sensor Metal pass, trainer, and backward never landed. |
| [`star_uvt_v0`](../../third_party/fast-mac-gsplat/variants/star_uvt_v0), affine base | Full symmetric UVT quadratic, tile-time Metal, alpha compositing, and extensive forward/backward. Affine packet carries UVT geometry plus `depth0` and `depth_beta`. | Active back end; consumes camera-conditioned records and omits conditional depth variance in the compact affine contract. Its opening README describes old Gate 0 and is stale about backward/training. |
| `WorldTubeBatch` in [`world_tube.py`](../../third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/trainer_harness/world_tube.py) | \(x_0,v,t_0\), two screen/fronto-parallel spatial precisions, temporal precision, opacity, RGB. | Restricted producer: no world-z width, full SPD(3), or spatial orientation. It is not full SPD(4). |
| `FeatureScreenTimeTubeModel` in [`star_uvt_feature_tube_model.py`](../../src/train/star_uvt_feature_tube_model.py) | Trainable full SPD(3) in sensor-time UVT. | Camera-specific projected model, not a reusable world asset; depth metadata is not a complete trainable conditional law. |
| `star_uvt_v0` projective interval path | Local projective traces, camera-family charts, event-certified order strata, per-pixel depth, interval Metal forward/VJP, mixed fallback, exposure and rolling shutter. | Strongest moving-camera path; begins from compiled UVT/projective records and treats atlas topology/order as constants in backward. Its lack of a nonzero direct hard-order depth VJP is mathematically consistent inside a fixed stratum, not by itself an implementation bug. |
| [`star_uvt_prt_v0`](../../third_party/fast-mac-gsplat/variants/star_uvt_prt_v0) | Dedicated Projective Rational Tube research fork: restricted world tube + camera polynomial to rational image traces, dense/tiled Metal and backward gates. | Important completion lineage, not product default and not full SPD(4). Some timing/novel-camera gates remain open. |
| [`star_prt_v0`](../../third_party/fast-mac-gsplat/variants/star_prt_v0) | Earlier dense PyTorch scaffold. | Metal entry points are placeholders; superseded/reference only. |
| [`star_uvt_projective_interval_backend.py`](../../src/train/star_uvt_projective_interval_backend.py) | Explicit trainer adapter for prebuilt projective trace atlases. | Disabled by default; no canonical world producer or full world/camera VJP. |
| [`world_foam_lane2_v0`](../../third_party/fast-mac-gsplat/variants/world_foam_lane2_v0) and fused CSR/direct/slab descendants | Retain depth, compile owner/cell runs, and perform explicit fiber transfer prototypes. | Best retained-depth research lineage; topology/geometry gradients and broad native optical-transfer/quality acceptance remain incomplete. The paper-runner `worldfoam` label can route to an ancestor/proxy instead. |
| [`trainerWebGpu3d.js`](../../web/dynaworld_browser_trainer/trainerWebGpu3d.js) | Active browser dynamic anisotropic 3DGS: moving/harmonic centers, log scales, quaternion, pinhole projection, camera-specific depth sort, analytic geometry gradients. | Useful browser trainer, but not STAR, no ray-bundle atlas, no full SPD(4), and no continuous camera-program certificate. |
| [`trainerWebGpuStar.js`](../../web/dynaworld_browser_trainer/trainerWebGpuStar.js) | Standalone affine UVT/STAR WebGPU microprototype with conditional-depth sort. | Small fixed-capacity prototype; no moving-camera atlas/certifier/fallback, and its raw precision update is not a production-safe SPD parameterization. |
| [`trainerWebGpuDynamicGs.js`](../../web/dynaworld_browser_trainer/trainerWebGpuDynamicGs.js) | Small explicit per-frame 3DGS browser baseline with per-frame depth order. | Geometry is fixed and only RGB/opacity train in this microbenchmark; not the full paper baseline. |
| [`browser_4dgs_baseline.md`](../browser_4dgs_baseline.md) | Contract/roadmap for an external native 4DGS baseline. | No integrated external full-4DGS renderer/trainer is present. |

In the repository's Python 3.11 environment, the version audit found and
loaded local CPython 3.11 native artifacts for `v5`, `v5_features`,
`star_uvt_v0`, and `star_uvt_prt_v0`; CPython 3.14 artifacts are also present
for the two STAR lineages. `star_prt_v0` has no native library and intentionally
raises from its Metal entry points. The WorldFoam fused CSR/direct/slab native
libraries load under Python 3.11; the base WorldFoam artifact on disk is
Python-3.14-only. These are build/import checks, not broad numerical or memory
acceptance.

The main paper runner does not currently exercise the strongest projective
interval atlas as its general moving-camera path. Its accepted World Tubes row
uses the restricted model and narrower projection policies. The browser also
does not use the atlas. “It exists on disk” and “it is the active product/paper
path” are therefore different claims.

## The mathematical object we should freeze

### Canonical world atom

For finite-lifetime dynamic content, use

\[
\mathcal G_i=(\mu_i,\Sigma_i,a_i,\psi_i),
\qquad
\mu_i\in\mathbb R^4,
\quad
\Sigma_i\in\operatorname{SPD}(4).
\]

Geometry has \(4+10=14\) effective degrees of freedom. A scalar amplitude and
constant RGB add four, for 18 total. Position is the mean in ordinary 3DGS and
in this 4D model; covariance describes spatial/spacetime extent, not uncertainty
about the learned position.

The best optimization chart is the exact conditional block chart

\[
x_0\in\mathbb R^3,\quad t_0\in\mathbb R,\quad
C=L_xL_x^\top\in\operatorname{SPD}(3),\quad
v\in\mathbb R^3,\quad c=e^{2\ell_t}>0,
\]

with

\[
\Sigma_4=
\begin{bmatrix}
C+c vv^\top & cv\\
cv^\top & c
\end{bmatrix}.
\]

This is a lossless parameterization of every strict SPD(4) covariance. Under a
peak-density amplitude convention, writing \(\tau=t-t_0\), its raw fixed-time
field is

\[
\rho_t(x)=
a\,e^{-\tau^2/(2c)}
\exp\!\left[-\frac12
(x-x_0-v\tau)^\top C^{-1}(x-x_0-v\tau)
\right].
\]

After normalization at fixed time, its conditional shape is
\(X\mid T=t\sim\mathcal N(x_0+v\tau,C)\), but that normalized conditional omits
the temporal multiplier and must not replace the raw rendered slice. Thus
“velocity” is exactly the space-time cross covariance expressed in physical
coordinates; it is not a separate spline assumption. One strict Gaussian has
an affine conditional center and constant conditional covariance. Curved motion
or changing physical rotation/scale requires a mixture/piecewise chain or a
later generalized tube.

The six orientation degrees of freedom of a 4D ellipsoid are already in
\(\Sigma_4\): three are spatial orientation and three are spacetime tilt. A
pair of unit quaternions can parameterize \(SO(4)\), but an octonion is not the
right rotation object and is unnecessary for the first implementation.

### Compiled camera-ray object

Let \(y=(u,v,t)\) and let \(z\) parameterize depth along a ray. The camera
program defines a local ray chart

\[
\Gamma(y,z)=(X(y,z),t).
\]

The mathematically correct transform is the pullback to the ray bundle followed
by pushforward along the depth fiber:

\[
\mathcal T_\Gamma=\pi_*\Gamma^*.
\]

In an affine chart, write the pulled-back precision in UVT/depth blocks as

\[
H=
\begin{bmatrix}P&r\\r^\top&h\end{bmatrix}.
\]

Completing the square gives the exact factorization

\[
S=P-\frac{rr^\top}{h},\qquad
\beta=-\frac{r^\top}{h},\qquad
s_z^2=\frac1h,
\]

\[
Z\mid Y=y
\sim
\mathcal N\!\left(m_z+\beta(y-m_y),s_z^2\right).
\]

Therefore a joint UVTZ Gaussian is losslessly represented by:

| Compiled field | Scalars |
|---|---:|
| UVT mean | 3 |
| UVT SPD(3) covariance/precision | 6 |
| Conditional depth center | 1 |
| Conditional depth slopes | 3 |
| Conditional depth variance | 1 |
| Geometry total | 14 |
| Typed trace amplitude + RGB | 4 |
| Total | 18 |

The current compact affine STAR record has all of these geometry fields except
the final conditional-depth variance. Carrying it is what makes the affine
factorization lossless and permits uncertainty-aware order/fallback decisions.

For a perspective ray that is affine in its chosen depth coordinate,

\[
X(y,z)=a(y)+z d(y),
\]

interpret \(a\) and \(d\) as their lifted 4D spacetime vectors (with zero time
component in the depth direction), require the local ray chart to be full rank,
and assume the fiber-measure Jacobian is independent of \(z\) in this affine
depth coordinate. For world precision \(\Lambda\), define

\[
h=d^\top\Lambda d,\quad
b=d^\top\Lambda(a-\mu),\quad
q_\perp=(a-\mu)^\top\Lambda(a-\mu)-\frac{b^2}{h}.
\]

Then, exactly along each ray,

\[
\widehat z=-b/h,\qquad
\operatorname{Var}(z\mid y)=1/h,
\]

and the unbounded trace contains

\[
A\,J(y)\sqrt{2\pi/h}\,e^{-q_\perp/2}.
\]

Near/far clipping adds a Gaussian-CDF factor. Perspective therefore does not
destroy the one-dimensional Gaussian fiber law; it makes its trace amplitude,
mean, and variance nonlinear functions of \(y\). Those functions can be
compiled locally with rational/polynomial coefficients and residual
certificates—the role the existing atlas is already designed to serve. In a
nonlinear log/inverse-depth gauge, its coordinate Jacobian generally depends on
depth and must remain inside the integral; the physical trace stays invariant,
but the coordinate density need not retain this simple Gaussian form.

### Rendering policy

The compiled object should be named explicitly, for example:

```text
FiberTrace = {
  uvt_marginal,
  conditional_depth,
  trace_amplitude_semantics,
  appearance,
  gauge_id_and_orientation,
  support_and_fit_certificate
}
```

Rendering then has three modes:

1. stable, disjoint/certified depth intervals: fast interval Metal alpha replay;
2. small ambiguity with a proved swap-error bound: declared commutation
   approximation;
3. unresolved overlap/event cells: live conditional-depth or retained-fiber
   fallback.

The future retained-fiber fallback should differ from today's live mean-depth
sort. It should use the conditional Gaussian depth laws and optical thickness
to integrate overlapping colored fibers to a declared tolerance. This is the
clean bridge to the WorldFoam idea without making every pixel pay its cost.

## Exhaustive concern ledger from this thread

### Memory, batching, splat counts, and fair baselines

| # | Concern | Worked out in theory | Worked out in engineering/evidence | Still missing |
|---:|---|---|---|---|
| 1 | Why was `kernel_task` huge; could it kill the Mac? | It is consistent with macOS managing severe unified-memory/compression/swap pressure, not proof that `kernel_task` itself was the faulty model process. | The dangerous row was killed and full-scale local MPS work is fail-closed. | Phase-resolved externally observed memory attribution. Do not rerun the full row locally. |
| 2 | What exactly was the process doing? | Training memory is parameters + optimizer + autograd/render workspaces + targets/cameras/caches + allocator/compressor behavior. | The invalid fixed-512 Coffee Martini incident used two train plus one heldout view, four sampled frame/view items per step, 1,024 primitives, and full resident per-frame banks/caches. The machine reached severe compression/swap pressure and destabilized, but that killed row has no admissible process peak. | A safe one-lane microprofile that separates each term. Do not substitute the accepted progressive row's 20.557 GB mean for this incident. |
| 3 | Were all 300 frames trained at once? | Active batch size and resident temporal storage are different. | No: four structured samples were rendered per optimizer step. But dynamic 3DGS kept 300 parameter banks resident and earlier evaluation/caches were too eager. | True disk streaming and bounded accounting for every cache/model/optimizer allocation. |
| 4 | How many splats? | Compare active primitives at a queried time separately from total clip states. | Accepted setup: 1,024 active dynamic splats at a frame and 1,024 active shared tubes; dynamic stores 307,200 frame-splat states. | Every result table must report active N, total state, optimizer bytes, tile/cache entries, and peak host/driver memory separately. |
| 5 | Why could memory explode if Metal kernels were lean? | Lean shader scratch does not bound the whole training process. | Direct-atomic backward reduces one workspace; host targets/bounded evaluation were improved after the incident. | Static review is not runtime memory acceptance. Safely profile post-incident changes before reopening MPS. |
| 6 | Does dynamic 3DGS's large memory prove tubes? | It supports the temporal-amortization/storage hypothesis, not visual-model superiority or a causal explanation of all driver memory. | Three-seed means: about 3.114 GB peak driver and 0.060 MB checkpoint for restricted tubes versus 20.557 GB and 17.206 MB for per-frame dynamic 3DGS. | Broader scenes and causal memory breakdown. |
| 7 | What is a fair baseline; is per-frame doing better with similar compute? | Fairness requires several axes: same active N, same bytes, same wall/steps, same representation compiled vs replay, generous per-frame ceiling, and model-class peer. | Current quality is mixed: tubes lead PSNR/L1; per-frame leads SSIM/LPIPS; walls are about 78.3 s vs 79.4 s because only four frames are sampled per step. Absolute quality is poor. | Full-SPD4 tubes, equal-byte/quality curves, convergence curves, multiple scenes/cameras, and a native 4DGS peer. The current row does not establish a winner. |

The exact accepted three-seed means behind that interpretation are:

| Model | PSNR | SSIM | LPIPS | L1 | Wall | Peak driver | Checkpoint | Trainable scalars |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Restricted World Tubes | 5.9153 | 0.03549 | 0.98305 | 0.45120 | 78.33 s | 3.114 GB | 0.060 MB | 14,336 |
| Per-frame dynamic 3DGS | 4.9110 | 0.28267 | 0.90228 | 0.52139 | 79.44 s | 20.557 GB | 17.206 MB | 4,300,800 |

These values establish neither visual adequacy nor a clean winner. They show a
large learned-state/observed-memory difference and a metric tradeoff in a
restricted, poorly fit experiment. The per-frame bank itself is about 17.2 MB
of raw float parameters, or roughly 51.6 MB for parameters plus two Adam
moments. That cannot directly explain a roughly 20.6 GB driver peak; render and
autograd allocations, allocator caching, targets/media, compilation caches, and
system compression must be measured rather than inferred from parameter count.

### Primitive definition and free parameters

| # | Concern | Worked out in theory | Worked out in engineering/evidence | Still missing |
|---:|---|---|---|---|
| 8 | What are ordinary 3DGS free parameters? | Mean 3 + SPD(3) covariance 6 = 9 geometry; peak opacity 1 + constant RGB 3 gives 13 effective DOF. | Normally stored as 3 log scales + quaternion4, so 14 floats with one quaternion constraint. | Only appearance choice (RGB vs SH) varies. |
| 9 | Is full symmetric covariance more than width/height/depth? | Yes. Three principal widths plus three 3D rotation DOF, equivalently any SPD(3) matrix. | Standard anisotropic renderer uses \(R(q)\operatorname{diag}(e^{2\ell})R(q)^\top\). | None for ordinary 3DGS. |
| 10 | What are log scales and why quaternion4/3 DOF? | \(s=e^\ell>0\); a unit quaternion has four stored reals, one norm constraint, and \(q\sim-q\). | Implemented in standard/browser 3D paths. | A full 4D production parameterization is missing; use the block-Cholesky chart first. |
| 11 | Does normal 3DGS really have opacity; isn't opacity edge-dependent through covariance? | Pixel alpha is a separate peak opacity multiplied by Gaussian falloff. Covariance controls the edge falloff, not the peak scalar. | All principal paths carry opacity. | Freeze world amplitude semantics: peak alpha, conserved mass, or extinction. The compiler must apply its Jacobian/normalization consistently. |
| 12 | Why was color fixed; should it be free? | Fixed color was an experimental isolation choice, not a 4D constraint. | Per-frame baseline frees RGB per frame; current tubes have one RGB/tube; feature paths train appearance. | Decide the canonical first and final appearance contracts (constant RGB first; SH/view/time basis later). |
| 13 | Was spatial position fixed; should it be free? | The world mean is trainable, and the strict 4D slice center moves affinely through cross covariance. | Restricted tubes train \(x_0\) and \(v\); only the small browser DynamicGs microbenchmark freezes geometry. | Canonical trainable \(\mu_4,\Sigma_4\) code. |
| 14 | Why velocity, and why spline velocity? | \(v=\Sigma_{xt}/\Sigma_{tt}\) is exactly a coordinate of SPD(4). A spline is not part of the minimal atom. | Restricted tube exposes trainable velocity. | Restore full \(C\in\mathrm{SPD}(3)\); add a mixture/piecewise chain only after measuring curved-motion residuals. |
| 15 | Is the full 4D object only eight parameters? | No. Axis-aligned mean4 + scales4 is eight but loses rotations/tilts. Full geometry is mean4 + SPD(4)10 = 14. | `spacetime_v0` preserves the reference object. | Active safe parameterization/trainer. |
| 16 | Is “position is the mean” only possible in 4D; do we need covariance of position separately? | Position is the mean in 3D and 4D. Covariance is support/extent. Parameter uncertainty would be a different Bayesian object. | All Gaussian paths store means. | Nothing foundational. |
| 17 | What do \(t_0\) and temporal covariance mean? | \(t_0\) is the peak-time mean; temporal variance controls lifetime/activity, and cross covariance controls affine motion. | Restricted tube stores `t0`, temporal precision, velocity, and temporal opacity machinery exists in STAR. | Connect the full world temporal extent to the active compiler with frozen amplitude semantics. |
| 18 | Should position, covariance, scale, and rotation vary over time? | One strict joint Gaussian gives affine mean and constant conditional SPD(3). A moving camera can still change projected covariance. Physical \(C(t)\) or curved motion is a richer model. | Per-frame baseline can change everything; prototypes explore richer motion. | Empirically test strict SPD(4) first, then mixtures, then only justified generalized tubes. |
| 19 | Where is 4D rotation; do we need an octonion? | SPD(4) already contains four widths + six orientation DOF. \(\mathrm{Spin}(4)\) can use two unit quaternions; no octonion is needed. | Full matrix exists only in reference code. | Production block-Cholesky state; pair quaternions are optional later, not a blocker. |
| 20 | Should this be a true 4D Gaussian rather than a Gaussian cylinder? | Yes for the minimal dynamic atom: strict SPD(4) is a bounded spacetime ellipsoid. A cylinder is the singular infinite-persistence boundary. | Restricted code can approximate persistent tubes. | Type persistent background explicitly: static 3DGS, bounded persistent tube, or declared PSD limit. |

### Slicing, XYZT/UVT, shaders, gauges, depth, and visibility

| # | Concern | Worked out in theory | Worked out in engineering/evidence | Still missing |
|---:|---|---|---|---|
| 21 | Is taking a time slice the right operation? | Raw fixed-time slice gives the instantaneous world field; conditioning would erase lifetime and time-marginalization would blur motion. Rendering may evaluate the compiled trace at time without materializing a 3D bank. | STAR queries sensor-time traces and integrates exposure/rolling shutter. | Preserve the exact operator order when visibility and shutter integration do not commute. |
| 22 | Should the core primitive be XYZT or UVT; why did UVT appear? | Core scene state should be camera-independent XYZT. UVT is a camera/chart-specific compiled output because raster support lives there. | Gate 0 deliberately began after projection to isolate the renderer; that boundary later became the active model. | Restore the world producer and prevent sensor-space models from being described as reusable multiview world assets. |
| 23 | Did we fail to implement the math in shaders? | The shader should consume compiled records; discrete atlas certification can remain host-side. | The back half is substantial: UVT quadratics, projective coefficients, support/binning, per-pixel depth/order, compositing, exposure, backward. | The missing front half is world SPD(4) → complete FiberTrace + normalization + VJP. |
| 24 | Is “camera compiler” sloppy; weren't these camera gauges and depth rays? | “Compiler” is sound engineering shorthand for specializing \(\pi_*\Gamma^*\) over a known camera program. Gauges are fiber coordinates; the ray bundle is the physical object. | Gauge IDs, local projective charts, support/denominator certificates, and interval lowering exist. | Exact camera-program input contract, residual guarantees, and full world/camera gradient path. |
| 25 | Do changing camera paths alter occlusion/depth ordering correctly? | Yes inside event-certified local domains: moving view changes fibers, order functions acquire roots/UV lines, and strata/fallback handle them. | Implemented/tested in projective STAR for compiled traces; per-pixel/time depth is consumed in Metal. | Canonical SPD(4) source, broad pathological acceptance, distortion/general lenses, and a full 360/720 multi-chart integration. |
| 26 | Is there a more elegant depth formulation? | Yes: UVT marginal + affine conditional Gaussian depth is exactly equivalent to joint UVTZ in an affine chart. Perspective retains an exact Gaussian along each ray. | STAR carries conditional mean depth and uses its pixel slopes; WorldFoam prototypes retain depth longer. | Add conditional variance, typed trace amplitude, uncertainty-aware order, and retained-fiber optical fallback. |
| 27 | What did STAR stand for; did we lose an older elegant representation? | No authoritative expansion was found; do not invent a backronym. Repository-wide archaeology recovered full SPD(4), fiber pushforward, gauges, projective traces, transported covariance, mixtures, and splines. The elegant object was narrowed in implementation, not deleted in the visible history. | Relevant lineages remain on disk. | Consolidate them under one world/FiberTrace boundary instead of introducing another representation name. Imported third-party prehistory remains incomplete. |

## What remains in mathematics

### M0 — Freeze semantics before coding

Write one normative contract for:

- world space units and the explicit time-to-space scale used in conditioning;
- strict finite SPD(4) dynamic atoms and the separate persistent/static type;
- amplitude as exactly one of peak alpha, conserved mass, or optical extinction;
- constant RGB for the first geometry experiment and the later appearance hook;
- Gaussian support truncation/tail tolerance, antialiasing, near/far clipping,
  shutter and rolling-shutter operator order;
- gauge orientation: only orientation-preserving depth charts may retain the
  same front-to-back order without reversal.

Until M0 is frozen, two algebraically correct implementations can disagree by
a scale/Jacobian factor while both appear plausible.

### M1 — Canonical SPD(4) parameterization theorem and implementation spec

Document and property-test the bijection between strict SPD(4) and
\((C,v,c)\), including the inverse, log-Cholesky constraints, units, determinant,
and gradient conditioning. Prove that no covariance DOF is lost.

### M2 — Time-slice and persistence semantics

Freeze the raw-slice formula, temporal activity normalization under the chosen
amplitude convention, and the typed static/persistent boundary. Do not silently
normalize to \(p(x\mid t)\).

### M3 — Ray-bundle disintegration and gauge theorem

Turn the existing derivations into the normative `FiberTrace` contract:

\[
\Gamma^*\nu_i(dy,dz)
=
(\pi_*\Gamma^*\nu_i)(dy)\,K_i(y,dz).
\]

State exactly how both factors transform under an orientation-preserving depth
gauge and prove value and VJP invariance away from topology events.

### M4 — Exact affine lowering and inverse reconstruction

Prove/test the Schur factorization above, its inverse reconstruction of the
joint UVTZ covariance, and the amplitude normalization for the chosen M0
semantics. This gives the first lossless end-to-end reference.

### M5 — Perspective ray functions and clipping

Specify exact functions \(q_\perp(y)\), \(\widehat z(y)\), \(h(y)\), Jacobian,
and truncated-CDF mass. Define how a local polynomial/rational approximation is
certified against those functions and against the input camera program—not only
against its own fitted trace.

### M6 — Visibility theorem and declared approximation policy

Define finite support intervals from the Gaussian tail tolerance and use depth
variance in separation certificates. Prove:

- stable-order replay within a certified cell;
- root isolation and atlas subdivision/termination under stated regularity and
  resource bounds;
- adjacent-swap image bounds, such as the two-layer
  \(\alpha_i\alpha_j\lVert c_i-c_j\rVert\) control;
- conditions for fast alpha compositing versus retained conditional-fiber
  quadrature;
- a deterministic fallback when the certificate cannot close.

A universal finite atlas for every arbitrary camera program is not required;
the contract may return fallback rather than make an unjustified claim.

### M7 — Differentiability at topology events

State the objective as piecewise smooth. Within a fixed atlas, use the ordinary
VJP. At support/order/topology events, specify rebuild and one-sided/subgradient
or retained-smooth fallback behavior. Do not claim a classical derivative of a
hard sorting permutation exactly at a swap.

### M8 — Capacity extensions only after falsification

First test one strict SPD(4) atom. If it fails controlled curved/rotating
sequences, use a short mixture or piecewise chain of strict atoms. Add spline
centers or SPD-valued covariance trajectories only after a measured residual
shows that the mixture is inadequate at the desired budget.

## What remains in code

### C0 — Build an isolated, auditable reference front end

Start outside the production variants, for example:

```text
research_experiments/spd4_world_tubes/
  model.py              # WorldAtom and safe block-Cholesky parameters
  camera_program.py     # calibrated ray bundle and shutter-time contract
  fiber_trace.py        # affine exact and perspective raywise lowering
  reference.py          # slow retained UVTZ / conditional-fiber renderer
  compiler.py           # local charts, residuals, support certificates
  visibility.py         # intervals, event roots, swap bounds, fallback
  train_smoke.py        # tiny end-to-end optimizer
  benchmark.py          # quality/time/memory/fallback accounting
```

Add property/acceptance tests rather than a new fork first:

```text
tests/test_spd4_world_tubes_algebra.py
tests/test_spd4_world_tubes_fiber_trace.py
tests/test_spd4_world_tubes_gauge.py
tests/test_spd4_world_tubes_visibility.py
tests/test_spd4_world_tubes_gradients.py
tests/test_spd4_world_tubes_end_to_end.py
```

### C1 — Introduce a typed trace ABI

The affine trace schema must carry:

- UVT mean and constraint-safe SPD(3) covariance/precision;
- trace amplitude under the frozen M0 convention;
- conditional depth center, three slopes, and log variance;
- appearance;
- gauge ID/orientation, source atom ID, chart domain, support/tail threshold,
  approximation residual, and fallback mode.

Projective cells should carry approximations/certificates for trace amplitude,
depth mean, depth precision, UVT support, clipping, and camera-fit residual.
Version this ABI rather than silently adding one float to every legacy packet.

### C2 — Connect the full world producer

Implement trainable \(x_0,t_0,L_x,v,\ell_t\), reconstruct strict SPD(4), and
lower it through the exact affine reference. Replace the current two-precision
`WorldTubeBatch` restriction with full SPD(3) spatial shape before adding any
spline or extra neural field.

### C3 — Reuse, do not rewrite, the STAR back end

Adapt `FiberTrace` into `star_uvt_v0`:

- retain the current UVT quadratic and per-pixel depth mean path;
- extend interval certificates with conditional depth variance;
- keep event splitting, UV subdivision, stale-candidate refresh, exposure, and
  rolling-shutter machinery;
- add a retained conditional-fiber fallback for cells that cannot safely use
  mean-depth alpha ordering;
- preserve the existing reference fallback as a diagnostic mode.

### C4 — Complete the derivative chain

The native/host VJP must return gradients for every **continuous image field**
used by the fast path—UVT footprint/trace coefficients, trace amplitude, and
appearance—and differentiate the compiler back to world mean,
block-Cholesky covariance, amplitude, appearance, and optionally camera
parameters.

Do not demand a fictitious nonzero derivative of hard order. Conditional depth
center/slopes/variance only select a permutation in ordinary sorted-alpha
replay: their image derivative is zero inside a stable stratum and undefined at
a swap. Test that policy explicitly. Nonzero gradients for depth-law shape
belong to retained/smooth conditional-fiber fallback, a soft-order surrogate,
or an auxiliary depth loss. Treat atlas topology as frozen between rebuilds
and finite-difference only smooth parameters away from event surfaces.

### C5 — Integrate a minimal trainer before the browser

First gates:

1. one atom, affine camera, analytic target;
2. several atoms with a known temporal depth crossing;
3. two calibrated moving cameras and heldout-view reconstruction;
4. perspective orbit segments with denominator/near-plane/UV-order events;
5. finite exposure and rolling shutter;
6. ambiguous thick-fiber overlap that forces retained fallback.

Only after these pass should the paper runner gain a `full_spd4_world_tubes`
lane. Keep the restricted lane as a labeled ablation. Port to the browser last;
the current browser dynamic 3DGS remains useful as a separate baseline.

### C6 — Reopen MPS only through a bounded microprofile

The first native run must be a single child process with tiny \(N,F,H,W\), hard
timeouts, external resident/driver-memory observation, chunked evaluation, and
an artifact written before scaling. Increase one axis at a time. Never infer
memory safety from shader scratch estimates alone, and never start with the
300-frame fixed-512 row.

### C7 — Repair documentation drift

After the new ABI is real, update stale claims in the root STAR README,
renderer taxonomy/indexes, and older learning notes. In particular, do not
repeat either false extreme:

- “STAR has no backward/trainer” — stale Gate-0 history;
- “the full 4D-to-camera compiler is implemented” — overstates the current
  UVT/restricted producer.

## Acceptance ladder

| Gate | Required evidence | Failure meaning |
|---|---|---|
| A — algebra | Random SPD(4) block roundtrip, slice equality, affine UVTZ reconstruction, determinant/sign/units checks. | Primitive/compiler algebra is wrong. |
| B — gauge | Ordinary/log or two monotone depth charts agree in values and world-parameter gradients with Jacobians; reversed gauge is rejected/reorders. | Fiber coordinate semantics are wrong. |
| C — reference image | Direct world-ray integration, joint UVTZ, and factorized FiberTrace agree for affine cameras and clipping. | Lowering or amplitude is wrong. |
| D — native parity | CPU/Torch reference and Metal agree in forward and VJP for fixed certified atlas cells. | ABI/shader/VJP is wrong. |
| E — visibility | Time crossings, in-tile UV depth lines, disocclusion, thin occluders, and dense overlaps either certify or use the declared visibility fallback; invalid denominator/residual/near-plane/support charts subdivide then reject/fail-closed; stale candidates refresh/rebin. | Atlas/validity/fallback contract is incomplete. |
| F — multiview world semantics | One learned world state renders several moving cameras and a heldout camera without camera-specific source UVT parameters. | The model is still a sensor-space representation. |
| G — capacity | Full SPD(4) beats or explains the restricted fronto-parallel tube on controlled rotation/depth/tilt scenes. | Restore/fix G0 before adding curves. |
| H — systems | Phase-resolved host/driver peak, optimizer/model/cache bytes, fallback fraction, compile/forward/backward time, and quality are reported under a safe monitor. | No scale or memory claim is admissible. |
| I — baselines | Same-representation replay, equal-active-N, equal-byte, equal-time, per-frame ceiling, native 4DGS peer, and retained-depth peer are reported separately. | “Better” is under-specified. |

## Fair baseline matrix

No single comparison answers every question:

| Baseline | Controls | Question answered |
|---|---|---|
| STAR compiled atlas vs per-frame replay of the same full SPD(4) atoms | Same learned world state and images | Does compilation amortize storage/compute without changing the model? |
| Full SPD(4) tubes vs per-frame dynamic 3DGS at equal active N | Same active raster work | What quality is gained/lost by temporal sharing? |
| Same pair at equal model+optimizer bytes | Same learned-state budget | Which representation uses memory more effectively? |
| Same pair at equal wall time/steps and convergence curves | Same optimization budget | Is either easier to optimize? |
| Restricted tube vs full SPD(4) tube | Same temporal sharing, extra spatial DOF isolated | Did fronto-parallel narrowing cause the current quality deficit? |
| Native full-4DGS peer | Comparable 4D parameter capacity | How does the proposed compiler compare with a known native 4D representation? |
| WorldFoam/conditional-fiber fallback | Retained depth | When does center-order splatting fail due to overlapping depth support? |

Every row should report quality metrics separately, not collapse them: PSNR,
SSIM, LPIPS, and L1; active primitives; total trainable scalars; checkpoint,
optimizer, host, and driver bytes; compiler metadata; compile/forward/backward
time; event-cell and fallback fractions; and heldout camera/time coverage.

## Shortest honest implementation sequence

1. Freeze M0 amplitude, units, support, clipping, and persistence semantics.
2. Implement the strict block-Cholesky world atom and exact CPU affine
   `FiberTrace`, including depth variance and inverse reconstruction.
3. Prove/test gauge and gradient invariance plus direct ray-integration parity.
4. Feed that trace into the existing STAR affine path and complete trace/world
   VJPs.
5. Extend projective cells/certificates to depth variance and camera-fit
   residuals; reuse existing event/order/fallback machinery.
6. Add retained conditional-fiber fallback only for ambiguous overlaps.
7. Pass tiny multiview/moving-camera gates, then a safely monitored native
   microprofile.
8. Run the full baseline matrix. Add mixtures before splines if strict SPD(4)
   fails controlled curvature/rotation tests.
9. Integrate the paper runner, then the browser.

That sequence preserves the elegant object the research already found while
reusing the strongest code already on disk. It also separates three claims
that were previously getting conflated: compact world state, correct moving
camera visibility, and exact thick-fiber optical transfer.
