# World Tubes math, implementation, evidence, and next experiments

Date: 2026-09-06. Follow-up to the September 5 candid paper review.
User challenged the assessment of existing speed evidence and requested a
review of mathematics, code, bugs, alternative formulations, and experiments.

## Assessment and correction of the previous review

The initial speed evidence is stronger than my previous answer conveyed.
Retained Metal benchmarks show substantial forward and backward improvements,
and real-video artifacts cover ten sources. These justify taking the method
seriously. Pending realistic experiments are not failed experiments.

However, the next step is not simplI'my to launch every pending run. This audit
found reproducible correctness defects in particular paths, a timing asymmetry
in one early benchmark family, and a mismatch between the claimed ordering
implementation and the current native kernel. None establishes that the
overall method fails. Each has a bounded correction or a narrower valid claim.

Keep gauge invariance as a coordinate-consistency property. The interesting
research contribution would be compiling a camera program and dynamic world
into reusable rendering work, with a corresponding world-parameter adjoint.
Change of variables and the chain rule support that implementation; they do
not independently establish novelty or acceleration.

## Scope and reproducibility

Read-only implementation review covered the World Tubes/STAR UVT camera
lowering, SPD4 affine and pinhole projection, projective fitting/support and
visibility, interval binning, native Metal forward/backward, fallback, cache
refresh, trainer integration, early benchmarks, and frozen-world runners.
Two independent subreviews covered math and native/backend behavior. This is
not an exhaustive audit of every renderer, browser path, or file in Dynaworld.

No implementation or manuscript was edited, no GPU job/build was launched, and
no benchmark or paper-evidence count was promoted. The checkout contains
pre-existing and concurrent modifications; they were preserved.

Retained review artifacts:

`outputs/reviews/2026-09-06_world_tubes_math_code/`

- Five CPU reproduction scripts, their stdout, and `reproduction_results.json`.
- `source_sha256.json`, identifying reviewed source snapshots.
- `frozen_world_dry_run.json`, including the current resource gate.

The existing focused CPU gate completed with **75 passed, 20 skipped**:
`test_star_uvt_projective_trace.py`, `test_star_uvt_projective_binning.py`,
`test_star_uvt_projective_visibility.py`, `test_star_uvt_projective_correctness.py`,
and `test_spd4_world_tubes.py`. MPS availability was explicitly masked for this
CPU-only run; no skipped native check is counted as passing. The local venv
lacked pytest, so cached pytest/pluggy/iniconfig packages were added to the
Python path without installation; plugin autoload was disabled. Torch and
BLAS thread counts were bounded to one. The retained reproducers expose cases
the passing fixtures do not cover; they print counterexamples, not passing
regression assertions.

Line references below are to source inspected in this session. Prefix `STAR/`
means `third_party/fast-mac-gsplat/variants/star_uvt_v0/`.

## 1. Existing evidence: credit, limitations, and falsifiers

### 1.1 The orbit benchmark is meaningful preliminary evidence

`outputs/benchmarks/2026-05-25_star_uvt_projective_shared_work_goal_audit/summary.md`
retains the fixed-chart orbit comparison at F = 8,16,32,64. At the final point:

| Metric | Compiled / replay | Equivalent speedup or reduction |
| --- | ---: | ---: |
| Forward time | 0.117417 | 8.52x faster |
| Backward time | 0.158417 | 6.31x faster |
| CPU compile time | 0.090765 | 11.02x faster |
| Trace/payload count | 0.0625 | 16x fewer |

The underlying F=64 row reports about 25.78 versus 219.53 ms forward, and
22.86 versus 144.30 ms backward. The physical orbit remains fixed as its sample
density increases. Both routes use the interval evaluator with different
chart decompositions. This supports the benefit of sharing chart work; it is
not yet a comparison against every competent rendering/reuse baseline.

Unlike the high-motion script discussed next, this orbit timing keeps scalar
summaries outside the timed loops. Do not discard this result because a
different benchmark contains a synchronization asymmetry.

### 1.2 Learned high-motion timings have a specific confound

Three retained small-scene artifacts report final forward ratios at most
0.2657 and backward ratios at most 0.0939. Shared interval entries grow at most
1.462x over their sweep, versus at least 9.852x for replay. Structural sharing
is real evidence even if the exact speed ratios need a corrected run.

In `research_experiments/star_uvt_feature_tubes/`
`projective_trained_high_motion_trace_scaling_benchmark.py:327-393`, the
per-frame timed loop calls `image.sum().detach().cpu().item()` each frame and
three gradient reductions/copies each backward frame. The interval comparator
at lines 136-197 gathers summaries outside timing. The asymmetry adds work and
host synchronization to replay. Its contribution to the reported ratios is
unknown until measured symmetrically. The same script increases a prefix of
physical time with F; that mixes interval length with sample density.

Cheap falsifier: move both routes' diagnostic reductions outside timing,
match transfer policy and cotangent fields, and sweep sample density over one
fixed interval. Retain the original artifacts as historical measurements.

### 1.3 The real-video lane exists, with a narrower comparison

`.../2026-05-25_star_uvt_projective_real_video_compiled_adjoint_replacement/summary.md`
records ten sources, twenty cases, four frame counts, interval-main-path use,
renderer gradients, and accepted cache reuse.

`.../2026-05-25_star_uvt_projective_real_video_timing_protocol_acceptance/summary.md`
records a fresh-process median projective-total ratio of 0.835659, a no-first
ratio of 0.564512, and two retained strict warm-state failures. This is a cache
policy comparison within compiled training, not the pending frozen-world
compiled-versus-replay experiment. It supports implementation viability, not
heldout novel-view quality or a universal total-step speedup.

The old Neural3D calibration error invalidates the affected quality comparisons
listed in `BASELINES.md`; it does not erase independent orbit and source-video
timings. Current publication evidence still lacks the frozen learned-world
scaling and public-context results. The previously accepted bounded camera
curve and theorem fixtures should keep their actual, limited status.

### 1.4 What "sublinear" can defensibly mean

The observations can support sublinear growth over a measured range and
amortization of world projection, chart storage, and some backward work.
Explicitly materializing F images of HxW pixels requires Omega(FHW) output work.
The claim should identify which cost grows with sample count and which grows
with visibility/support events. It should not imply indefinitely sublinear
total rendering cost with fixed resolution and fully materialized outputs.

## 2. Confirmed defects and their scope

### A. P1: moving-camera lowering resets the temporal envelope

Source: `STAR/research_project/trainer_harness/world_tube.py:316-321,422-427`.
Both legacy moving-camera projectors replace the tube's time center `t0` with
`camera.chart_time` while leaving its temporal precision and opacity unchanged.

CPU counterexample: zero camera/object motion, t0=2, chart time=0, lambda_t=2,
opacity=0.62. At t=0 and projected center, returned alpha is 0.62; the original
tube requires 0.62 exp(-4) = 0.011355696. The lifetime dependency on t0 is lost.
Existing motion smoke initializes t0=chart_time=0, hiding this case.

Correct first-order anchoring, with projected center p_c and velocity w at
chart time t_c:

```text
ma.t = t0
ma.uv = p_c + w (t0 - t_c)
depth0 = z_c + zdot (t0 - t_c)
```

Then `ma.uv + w(t-t0) = p_c + w(t-t_c)` while the envelope remains centered
at t0. Keep the original temporal precision and amplitude. This is an
algebraic correction, not a representation change. SPD4 affine pushforward
already preserves the original mean and does not share this exact defect.
Live legacy callers include `multicam_heldout_compare.py:951-965`.

### B. P1: broadening covariance escapes cached support

Source: `STAR/torch_gsplat_bridge_star_uvt/projective_trace.py:2340-2345` and
`STAR/research_project/trainer_harness/tile_metal_autograd.py:1578-1582,1630-1668`.
Coverage and omitted-alpha enumeration inspect center +/- fixed `uv_padding`.
Initial automatic support, however, depends on learned precision.

CPU counterexample: center (4.5,4.5), opacity 0.5, tile size 8, original spatial
precision diag(1,1), cached cell (0,0). Broaden precision to diag(0.01,0.01)
without moving the center. Refresh returns `rebinned=false`, `stale=false`,
and omitted-alpha bound 0. Pixel (12,4) stays black although required alpha is
0.36307454. This is a correctness issue during shape updates, not only a loose
performance bound. Proposal in section 4 repairs the support criterion.

### C. P1: factory defaults disagree about tile size

Source: `src/train/star_uvt_projective_interval_backend.py:18,924-928` defaults
compiler tiles to 16; `src/train/star_uvt_render_configs.py:26-34` leaves native
tile dimensions at `UVTRenderConfig` defaults of 8 (`STAR/.../rasterize.py:42`).

CPU counterexample: a splat centered at (12.5,12.5) on a 16x16 image is packed
only into compiler cell (0,0). Native tile8 semantics look for cell (1,1) at
that pixel and find no entry. The reference center is [0.5,0.5,0.5]. This
reproduction verifies the conflicting configuration/bin interpretation; it
does not execute the Metal kernel. Explicit tile8 benchmark configurations
mask this default-path defect.

Use one geometry configuration for compiler and native renderer and reject
inconsistent image/tile dimensions when constructing the trainer state.

### D. P1: mixed fallback uses a different compositing contract

Source: `STAR/research_project/trainer_harness/tile_metal_autograd.py:1017-1029`
does not forward max_alpha/background to the cell-atlas reference;
`STAR/.../projective_trace.py:4447,4483-4488` caps alpha at 1 and omits residual
background. Native metadata supplies the configured cap/background, and
`STAR/csrc/metal/star_uvt_kernels.metal:2276-2278` adds residual background.

CPU counterexample: red splat, raw center opacity 0.995, configured cap 0.99.
Fallback yields [0.995,0,0] and red-opacity derivative 1. Native formula gives
[0.99,0,0] and derivative 0. On white background native formula gives
[1,0.01,0.01]; fallback remains [0.995,0,0]. This compares executed CPU fallback
with the inspected native equation, not a new native runtime result.

The ordinary tile-reference branch also terminates when all pixels pass the
transmittance threshold, whereas native terminates individual pixels. Restore
the same per-pixel mask and `C = sum_i T_i alpha_i color_i + T_final background`.

Scope: cell-atlas reference/mixed fallback; the separate UVT bridge reference
already accepts cap/background. Current peak-splat model parameterization
limits opacity below 0.99 and black-background fixtures hide two differences,
so this does not retroactively falsify every accepted mixed-fallback fixture.

### E. P1: static SPD4 pinhole clamps behind-camera atoms into view

Source: `STAR/research_project/trainer_harness/spd4_world_atom.py:894,638`.
Camera depth is clamped to positive min_depth before projection and lowering.
The reviewed static caller and renderer do not apply a separate validity mask.

CPU counterexample: mean (0,0,-2,0), spatial precision 100, identity camera,
fx=fy=40, principal point (0.5,0.5), opacity 0.3. Returned center is
(0.5,0.5,0), depth approximately 0.0001, spatial precision 6.25e-10, opacity
0.3: a huge foreground footprint from a tightly concentrated atom behind the
camera. Positive-domain division stabilization is not near-plane clipping.

Define an explicit near-plane/support policy. A Gaussian crossing the plane
needs truncated integration or a declared approximation/fallback; rejecting
all partially intersecting Gaussians is not exact. CPU moving-SPD4 guards
reject invalid center depths, but MPS skips these value guards, so a production
policy cannot rely on CPU exceptions alone.

### F. P2: rotated legacy sheets omit spatial depth slope

Source: `STAR/research_project/trainer_harness/world_tube.py:233-234,319-320,`
`425-426,496-497`. Both spatial components of depth_beta are zero although the
world-XY sheet's projection is rotated.

CPU example: yaw 45 degrees, center at world origin, camera translation z=3,
fx=40. Code gives beta_u=0; the first-order plane requires beta_u=-0.075.
This matters where overlapping sheets change depth order across screen space.

For sheet-to-screen Jacobian J, sheet-to-depth row r, screen velocity w, and
center depth velocity zdot:

```text
b_uv = r J^-1
b_t = zdot - b_uv w
```

Assume J is nonsingular; near edge-on projections need conditioning/fallback.
The SPD4 conditional-depth formulation is the general version and does not
share this legacy restriction.

### G. P2: sampled residuals do not certify all intermediate times

Source: `STAR/.../projective_trace.py:780-795,918-934`. Fitting checks supplied
samples, then expands a continuous polynomial range by the sampled max error.

CPU counterexample: u(t)=1/(1.1+t^2), t in [-1,1], sampled only at endpoints.
The degree-one fit passes 1e-4 tolerances with residual 2.38e-7 and reports
u in [0.47619003,0.47619051]. At t=0 the true value is 0.90909088. The correctly
certified denominator stays above 1.1; denominator safety does not bound fit
error. Rendering only the supplied sample set remains within the stated
sample contract. Arbitrary-time queries, densification, or shutter integration
need recertification or a continuous residual bound. The current manuscript
already acknowledges sampled residuals in places; stronger atlas notes should
be brought into agreement.

## 3. Math that holds, and math that needs precise wording

### 3.1 Affine Gaussian lowering is a sound foundation

For affine camera coordinates and a positive-definite joint Gaussian precision
on (y,z), where y=(u,v,t), partition precision as `[A b; b^T c]`. Completing
the square gives marginal precision `A-b b^T/c`, conditional depth mean
`mu_z - b^T(y-mu_y)/c`, and variance `1/c`. The infinite-fiber amplitude factor
is `sqrt(2 pi/c)`, with the appropriate coordinate/physical-measure factor.
The SPD4 covariance pushforward/conditioning is consistent with this algebra.
Global perspective projection is not affine jointly in u,v,z: its inverse ray
map contains zu and zv. Exact affine closure must not be claimed globally.

Inspected native smooth-branch derivatives also have the expected form. For
delta=(dx,dy), alpha=a exp(-delta^T Q delta/2), and
`s=(dL/dalpha) alpha`, center gradient is `s Q delta`, packed precision gradient
is `-s [dx^2,2 dx dy,dy^2]/2`, and temporal exponent-coefficient gradient is
`-s [1,t,t^2]/2`. This source check is not numerical native validation.
Zero depth gradients inside fixed hard ordering are expected; derivatives at
visibility boundaries are a separate issue.

### 3.2 Keep gauge invariance, separate fiber and base coordinates

For extinction density per world length, the physical quantity is

```text
tau = integral rho(r(z),t) ||dr/dz|| dz.
```

A differentiable, invertible depth chart z=h(zeta) preserves tau when the
integrand includes `|h'(zeta)|`. This is useful: alternative depth coordinates
must agree physically, and invariance tests catch measure errors. It is a
standard change-of-variables result, not an independent novelty claim.

A depth-fiber change leaves sensor coordinates (u,v,t) fixed. It cannot
straighten the physical screen trajectory. Reparameterizing camera time or
angle changes base coordinates and needs transformed derivatives and shutter
weights. For t=g(s), `w(t)dt` becomes `w(g(s)) |g'(s)| ds`. Those changes can
make traces easier to approximate, but are different operations.

The depth/log-depth probes currently establish invariance of their selected
dz measure. Camera-forward depth generally has
`ds = ||R^T K^-1 [u,v,1]^T|| dz`. Two charts can agree while sharing an omitted
ray-length factor. This is a limitation of the probe, not proof of a missing
factor everywhere: SPD4 peak-density affine lowering explicitly includes its
fiber-length correction (`spd4_world_atom.py:562-594`).

### 3.3 Conditional means do not transform as points

`research_notes/gauged_uvt_trace_atlas/04_revolving_camera_atlas/README.md:63`
writes a transformed representative depth as h of the old one. If the
representative is a conditional mean, the correct relation is

```text
E[z_b | y] = E[h_y(z_a) | y], generally not h_y(E[z_a | y]).
```

Even order of means can reverse under increasing h: A is equally likely to be
1 or 9, B=4. E[A]=5>4, but E[log A]=log 3<log 4. Restrict mean transport to
affine charts, transport an explicitly chosen physical point, or use quantiles
and support endpoints. Monotonicity preserves point/quantile order, not
arbitrary expectation order. Keep visibility comparisons in one physical
depth convention where possible.

### 3.4 Exposure does not commute with alpha composition

Correct exposure integrates the composited image at each shutter sample.
Even with fixed depth order, averaging individual alphas before composition
is generally wrong: E[alpha_1 alpha_2] differs from E[alpha_1]E[alpha_2].
The reviewed quadrature path composites samples before integration; retain
that behavior. Order stability simplifies certificates, not this product.

## 4. More useful formulations and implementation improvements

These are proposals with standard mathematical ingredients. No speedup or
accuracy improvement below has been measured in this session.

### 4.1 Apply the shared world-to-trace adjoint once

Let theta be frozen world/camera parameters, phi=C(theta;kappa) all continuous
compiled fields, and kappa a fixed topology. Split rendered samples into chunks:

```text
L = sum_b L_b(R_b(phi;kappa))
g_phi = sum_b dL_b/dphi
g_theta = (D_theta C)^T g_phi
```

The current frozen compiled timing path at
`STAR/research_project/benchmarks/multicam_heldout_compare.py:5880-5960`
builds the shared graph once but calls backward with retain_graph on each
chunk, traversing the world-to-trace graph repeatedly. Accumulate chunk
cotangents into detached leaf copies of every compiled field, then invoke one
VJP through the original compiler graph. This is exactly equivalent for fixed
parameters/topology, with correctly weighted losses. Include direct theta
dependencies separately and regularization once. Opacity, temporal envelope,
footprint, and appearance fields must all participate; omitting a detached
field silently loses gradients. No optimizer step or topology mutation may
occur between chunks.

Cheap test: compare world gradients from chunked ordinary autograd and the
factored version for multiple chunk sizes and nontrivial image cotangents.
Then profile world-lowering backward time separately. If already negligible,
do not expect a large total improvement. This is a plausible systems gain,
not a new chain rule.

### 4.2 Bound the changing footprint, not just its center

For a 2D Gaussian alpha field with SPD Q,

```text
alpha(delta) = a exp(-delta^T Q delta/2)
alpha >= epsilon  iff  delta^T Q delta <= 2 log(a/epsilon)
radius_j = sqrt(2 log(a/epsilon) (Q^-1)_jj)
```

These are exact axis-aligned bounds on the alpha-superlevel ellipse for
a>=epsilon>0; support is empty when a<epsilon. For UVT use the conditional
spatial precision block, shifted center at each time, and effective temporal
amplitude. Recompute or conservatively bound these when precision or opacity
changes. For Beer-Lambert opacity use optical-thickness threshold
`-log(1-epsilon)` instead of the peak-splat threshold.

Use these bounds both for cache coverage and omitted-support enumeration.
For a whole time cell, certify the time-dependent envelope/shape extrema or
restrict the contract to the actual sample set. Per-atom epsilon is not a
global image-error budget: many omitted contributions can accumulate.

### 4.3 Certify rational extrema and approximation residual separately

For u(t)=n(t)/d(t) with quadratic n,d and nonzero denominator, extrema occur
at interval endpoints and real roots of `n'd - nd'`. Cubic terms cancel, so
the derivative numerator is at most quadratic. This gives cheap exact center
bounds for this rational trace family.

For a polynomial approximation p(t), bound the different quantity

```text
|u-p| = |n-p d| / |d| <= sup |n-p d| / delta,
where |d| >= delta > 0.
```

Use interval/Bernstein polynomial bounds and subdivision for the numerator.
This adds a continuous residual certificate where continuous queries are
needed. It does not certify footprint shape or visibility by itself. If the
paper is only about a fixed sample set, label the certificate accordingly and
avoid adding this complexity until arbitrary-time behavior is required.

### 4.4 Use exact finite-ray Gaussian integration as an oracle

At a fixed pixel and shutter sample t*, a world-space ray is affine in depth
even under pinhole projection. For spacetime X(z)=a+bz, with a=(o,t*) and
b=(d,0), and density `rho0 exp(-(X-m)^T Lambda (X-m)/2)`, define

```text
A = b^T Lambda b > 0
B = b^T Lambda (m-a)
C = (m-a)^T Lambda (m-a)
tau = rho0 ||d|| sqrt(2 pi/A) exp(-(C-B^2/A)/2)
      * [Phi(sqrt(A) (z_far-B/A)) - Phi(sqrt(A) (z_near-B/A))].
```

Completing the square proves this exact finite-ray optical thickness. Phi is
the standard normal CDF. Ray norm is one for arc-length depth and generally
not one for camera-forward depth. rho0 is peak extinction density, not peak
alpha or normalized mass. Use stable CDF differences in extreme tails and
stable residual evaluation when C and B^2/A nearly cancel.

This is a useful independent reference for pinhole footprint and clipping
errors, including off-axis and near-plane cases. It does not make the entire
screen-time footprint Gaussian, nor solve differently colored overlapping
volume transfer. Existing affine retained-fiber evaluation is not this full
pinhole finite-ray oracle. Do not change peak-splat to Beer-Lambert during a
frozen renderer comparison: that changes the rendering law being compared.
Analytic Gaussian ray integration has prior art, including
[Volumetrically Consistent 3D Gaussian Rasterization](https://openaccess.thecvf.com/content/CVPR2025/html/Talegaonkar_Volumetrically_Consistent_3D_Gaussian_Rasterization_CVPR_2025_paper.html).

### 4.5 Consume compiled order and retain packed device buffers

The native interval forward loop at `STAR/csrc/metal/star_uvt_kernels.metal:2264`
calls `select_projective_cell_order_id_interval` for each rank. That helper
at lines 650-692 scans packed candidates to find the next pixel/time depth.
Backward at 2480-2489 constructs the full order before alpha termination;
the family path repeats the pattern around 2634. Work is proportional to
active_count * packed_interval_count per pixel/frame, quadratic when all
candidates are active. This is not direct consumption of a precompiled order.

`WORLD_TUBES_PAPER_DRAFT.md:669-677` currently says the interval compositor
consumes one precompiled order and does not perform pixel-varying live sorting.
That description needs correction or the certified-order kernel needs to be
implemented. The present kernel can still be faster through other sharing;
metadata compression alone does not establish amortized ordering work.

Use certified lists directly where one order is valid; split or explicitly
sort unresolved strata. A retained certificate must match all native depth
fields and be invalidated on relevant updates. Test high tile occupancy and
occlusion; three-primitive scenes cannot expose quadratic ordering costs.

Separately, `STAR/.../projective_trace.py:5755-5771,6201-6217` packs CPU bins and
copies them to device in both forward and backward. Feature and alpha renders
can repeat this four times per step (`star_uvt_feature_overfit_trainer.py:324,365`).
Cache immutable packed device buffers by topology/config version and reuse
across those calls. Cache invalidation is part of correctness, especially after
the support-growth defect above. Profile packing, upload, ordering, shading,
and world adjoint separately before selecting optimization priorities.

### 4.6 Image error alone does not bound gradient error

For fixed topology, images I_c,I_r, Jacobians J_c,J_r, and a loss whose image
gradient is L_l-Lipschitz, one useful decomposition is

```text
||grad_theta L_c - grad_theta L_r||
 <= ||J_c-J_r|| ||grad_I L_r|| + ||J_c|| L_l ||I_c-I_r||.
```

Thus small image error alone does not establish a useful world-gradient
approximation. Retain explicit VJP comparisons with identical cotangents;
do not substitute nonzero-gradient checks for agreement. Hard visibility
boundaries are outside this smooth fixed-topology identity. General
visibility derivative issues are established in
[Differentiable Monte Carlo Ray Tracing through Edge Sampling](https://cseweb.ucsd.edu/~tzli/diffrt/).

## 5. Experiment order and decision rules

1. **Repair demonstrated correctness gaps on CPU, then tiny native cases.**
   Preserve lifetime under zero motion with t0 != chart time; test shape
   broadening, default tile geometry, saturated/fallback/background semantics,
   rotated depth slope, and near-plane handling. Add continuous-query checks
   only if that contract is claimed. Match RGB and all participating gradients.
   The counterexamples above are the regression inputs; no new audit framework
   is needed. Fix only paths participating in the frozen contract before its
   run; other findings remain separately tracked.

2. **Repair and repeat the early timing comparison cheaply.**
   Same frozen inputs, physical interval, rendering law, precision, buffers,
   cotangents, and transfer policy. Warm both routes, alternate their order,
   and keep diagnostics outside timing. Report compile, forward, backward,
   refresh/fallback, and total; retain variation, not only the best value.
   No retraining is needed to isolate this measurement issue.

3. **Run the existing static frozen-world sweep first.**
   Use one correctly calibrated checkpoint and F=4,8,16,32,64,128,300 over the
   same physical interval. Current entry point is
   `research_experiments/paper_runner_suite/run_frozen_world_replay_compiled.py`
   with `coffee_martini_full_300f_progressive_512_v1.jsonc`, seed 17. The
   retained dry run resolves the exact command; its `0` frame-count sentinel
   means the full 300-frame interval. The protocol presently trains its shared
   state before the sweep; do not assume an unrelated checkpoint is accepted.
   Compare images/world VJPs and actual peak process/device memory, count
   topology and transients, and include compile cost in break-even reporting.

4. **Then use that checkpoint for bounded moving-camera density.**
   Existing `run_frozen_world_moving_camera.py` validates the static summary
   and checkpoint identity and performs checkpoint-only evaluation. Use its
   bounded yaw and F=8,16,32,64 contract after correcting participating moving
   projection paths. A synthetic camera perturbation with reused target pixels
   can test replay/compiled agreement and gradient workload; it is not ground
   truth novel-view reconstruction quality for the perturbed camera.

5. **Expand to the public-context/quality matrix after the central result.**
   Complete the seven public contexts and 21 lane records with the corrected
   calibration and protocol-defined seeds. Separate rendering/compiler
   acceleration from reconstruction-model quality. Keep explicit baseline
   rows in BASELINES.md for accepted reruns. Do not rerun already accepted
   theorem/camera probes unless a source change affects their contract.

6. **Ablate optimizations separately.**
   Packed-buffer reuse, certified-order consumption, and a single compiler VJP
   are separate changes. Their measurements should reveal which mechanism
   produces improvement. Do not mix a new rendering law or representation
   into the frozen compiled-versus-replay evidence run.

Support for the paper: correct world gradients and a repeatable total-cost
break-even at realistic occupancy, or an actual memory-enabled workload that
replay cannot run under the same budget. Weaker outcome: only metadata shrinks
while sorting/shading dominates; frame the contribution as storage/setup
amortization. Strong negative result: fair batching/reuse eliminates the total
benefit in the intended workloads; narrow the method's target rather than
adding more formalism. Near-term uncertainty is empirical, after the concrete
correctness corrections, not a need to invent a replacement theory.

## 6. GPU blocker: current state, not old hardware folklore

The fresh dry run reports a streamed estimated peak of 8.655 GiB on this
24 GiB host, `high_risk=false`, but `source_complete_runtime_unverified`.
Old eager estimates around 18.745 GiB and statements requiring a larger host
do not describe this updated source path. An estimate is not proof it runs.

The actual live gate still fails: available memory was 7,464,452,096 bytes
(about 6.95 GiB), below the required 10 GiB; load per logical CPU was 0.987,
above 0.75. Free disk passed the 8 GiB threshold. The swap probe failed with
CalledProcessError; its max-integer sentinel is unavailable data, not measured
swap consumption. Clean source is required for execution, and this checkout
currently contains modifications. No gate was bypassed and no accelerator
process was started. The runtime lane is pending a clear host, functioning
resource checks, and a clean accepted source snapshot.

## 7. Positioning and open questions

The strongest framing remains a camera-program compiler with reusable dynamic
scene lowering and a world adjoint. Prior work already covers local affine
Gaussian footprints ([EWA Volume Splatting](https://www.cs.umd.edu/~zwicker/publications/EWAVolumeSplatting-VIS01.pdf)),
temporal Gaussian primitives ([Spacetime Gaussian Feature Splatting](https://arxiv.org/abs/2312.16812)),
event certificates ([kinetic data structures](https://graphics.stanford.edu/~comba/papers/socg.pdf)),
and temporal reuse of Gaussian sorting ([Neo](https://arxiv.org/abs/2511.12930)).
These constrain broad novelty claims; they do not establish equivalence to
the proposed combination or its training adjoint. This was a targeted prior-art
check, not exhaustive novelty clearance.

Open empirical questions: how many cells and fallback strata survive learned
scene complexity; whether total speed gains survive matched synchronization
and batching; whether world VJP error stays controlled; whether shape changes
make repairs frequent; and whether corrected-calibration reconstructions are
good enough to represent the intended workload. The early evidence makes
those questions worth answering. Calling the whole project slop would not
reflect that evidence; calling it an established result would also be premature.
