# Decision, Implementation Plan, Baselines, and Falsification Gates

## 1. Final representation hierarchy

### G0 — exact finite-lifetime reference atom

\[
\mu_4\in\mathbb R^4,
\qquad
\Sigma_4\in\operatorname{SPD}(4),
\qquad
\alpha,
\qquad
\text{appearance}.
\]

Purpose: the clean mathematical reference, algebra tests, literature parity,
and native 4D Gaussian baseline.

### T0 — recommended practical minimal World Tube

Choose a fixed reference time \(t_{\rm ref}\) and store

\[
p_{\rm ref}\in\mathbb R^3,
\quad
Q\in\operatorname{SPD}(3),
\quad
v\in\mathbb R^3,
\quad
\text{activity},
\quad
\alpha,
\quad
\text{appearance}.
\]

The spatial field is

\[
\rho(x,t)=
\alpha a(t)
\exp\!\left[-\frac12
(x-p_{\rm ref}-v(t-t_{\rm ref}))^\top
Q
(x-p_{\rm ref}-v(t-t_{\rm ref}))\right].
\]

Activity is typed:

```text
persistent: a(t) = 1 on an explicit support interval
localized:  a(t) = exp(-0.5 * lambda_t * (t - t0)^2), lambda_t > 0
```

The localized case is exactly G0. The persistent case is its
\(\lambda_t=0\) precision-space boundary and is needed for long-lived or static
content. Keeping \(p_{\rm ref}\) at a fixed global reference time avoids the
optimization coupling in which changing activity center \(t_0\) also changes
the represented path unless the center is compensated.

For a localized primitive, geometry has the same 14 effective DOF as G0:

| Field | DOF |
|---|---:|
| Reference position | 3 |
| Full spatial precision | 6 |
| Spacetime tilt / velocity | 3 |
| Activity center | 1 |
| Temporal precision | 1 |
| Total | 14 |

For a persistent primitive, the activity center/precision are unnecessary;
store an interval or global persistence tag instead.

### M1 — first extension

Use adaptive mixtures or piecewise chains of T0/G0 atoms. Split when measured
center curvature, covariance residual, projection denominator risk, temporal
activity error, or visibility-event density exceeds a declared threshold.
Share appearance/identity across child pieces where the opacity convention
allows it.

### D1 — later generalized dynamic tube

Only if M1 is inefficient at equal bytes, introduce low-knot functions
\(m(t)\), \(R(t)\), and log-scales \(\ell(t)\). This explicitly permits curved
motion and changing spatial orientation/size, but it is not one spacetime
Gaussian and does not compile to one exact Gaussian trace.

## 2. Frozen semantic choices required before code

The first implementation should not begin until a short contract freezes:

1. **Coordinate chart:** spatial origin/scale, temporal origin/scale, and
   conversion to physical units.
2. **Activity:** persistent versus localized Gaussian; explicit finite support
   policy.
3. **Amplitude:** peak-preserving 3DGS alpha, normalized Gaussian mass, or
   extinction/optical depth.
4. **Spatial filter:** covariance floor and screen-space antialiasing
   convention.
5. **Depth:** conditional mean, variance/interval sidecar, and sorting rule.
6. **Visibility:** order certificate, chart subdivision, and fallback behavior.
7. **Appearance:** fixed RGB for the first geometry isolation; SH or temporal
   appearance is a separate capacity ablation.

**Recommended first semantics:** peak-preserving alpha, simple RGB, localized
or persistent activity, full spatial SPD(3), conditional depth sidecar, and the
existing sorted-alpha renderer. Run mass/fiber-normalized amplitude separately
so geometry changes are not confounded with a renderer change.

## 3. Safe trainable parameterization

### G0 reference

Store the full covariance as a lower Cholesky factor:

```text
mu4            [N, 4]
raw_cholesky   [N, 10]
opacity_logit  [N, 1]
rgb_logits     [N, 3]
```

Construct positive diagonal entries with softplus or exponentiation plus a
small floor and set \(\Sigma_4=LL^\top\). This is ideal for CPU-double algebra
and literature parity.

### T0 production chart

Store:

```text
p_ref              [N, 3]
spatial_cholesky   [N, 6]    # covariance or precision, one convention only
spacetime_tilt     [N, 3]    # v
activity_mode      typed     # persistent or localized
activity_center    [N, 1]    # localized only
log_temporal_scale [N, 1]    # localized only
opacity_logit      [N, 1]
appearance         [N, A]
```

Use Cholesky solves, not explicit matrix inverses, in the hot path. Never
optimize six/ten packed symmetric entries with only diagonal clamping; positive
diagonals do not imply a positive-definite matrix.

Record coordinate normalization in checkpoints. Under \(t'=kt\), the exact
physical transforms include

\[
v'=v/k,
\qquad
\lambda_t'=\lambda_t/k^2,
\qquad
t_0'=kt_0.
\]

Equivalent scenes at 24, 30, and 60 fps must render equivalently after this
conversion.

## 4. Compile to the existing UVT ABI

For a local camera chart, define

\[
F(X,t)=(u,v,t_{\rm ABI},d).
\]

At the primitive center, compute \(m=F(\mu_4)\) and the chart Jacobian
\(A=DF(\mu_4)\). Under the local affine approximation,

\[
C'=A\Sigma_4A^\top.
\]

Partition \(a=(u,v,t)\) and depth \(d\):

\[
C'=
\begin{bmatrix}
C_{aa}&C_{ad}\\
C_{da}&C_{dd}
\end{bmatrix}.
\]

Compile

\[
Q_{uvt}=C_{aa}^{-1},
\qquad
\beta_d=C_{da}C_{aa}^{-1},
\qquad
\operatorname{Var}(d\mid a)
=C_{dd}-C_{da}C_{aa}^{-1}C_{ad}.
\]

Emit the existing renderer fields:

```text
ma          = m[:3]                 # 3 floats
q_uvt       = packed(Q_uvt)         # 6
depth0      = m[3]                  # 1
depth_beta  = beta_d                # 3
opacity                              # 1
color                                # 3
```

This remains 17 emitted floats per local trace. Carry conditional depth
variance, gauge/chart ID, support bounds, fit error, and order/fallback status
as compiler sidecars. The existing Metal ABI can consume any valid SPD UVT
precision but does not itself validate SPD.

The covariance pushforward above and the precision-space ray-fiber Schur
complement must agree numerically. That equivalence is a required test, not an
assumption.

## 5. Exact embedding of the current restricted model

The current model can be embedded as a controlled subfamily. Let

\[
C_3=\operatorname{diag}(1/p_x,1/p_y,\varepsilon_z^2),
\qquad
s_t^2=1/\lambda_t.
\]

Then construct

\[
\mu_4=(x_0,t_0),
\]

\[
\Sigma_4=
\begin{bmatrix}
C_3+s_t^2vv^\top&s_t^2v\\
s_t^2v^\top&s_t^2
\end{bmatrix}.
\]

It follows exactly that

\[
X\mid t\sim
\mathcal N(x_0+v(t-t_0),C_3).
\]

As \(\varepsilon_z\to0\), this approaches the current rank-two/fronto-parallel
support. Orthographic projection supplies the cleanest parity test. Pinhole
parity additionally depends on how finite depth thickness is approximated.

This embedding enables a decisive control: initialize the full model from the
restricted one, freeze every newly added covariance DOF, and require the same
render/loss before unlocking anything.

## 6. Minimal staged implementation

Use an isolated experiment lane first, for example:

```text
research_experiments/spd4_world_tubes/
  model.py
  reference.py
  compiler.py
  benchmark.py
```

Stages:

1. Freeze the semantic contract in Section 2.
2. Implement CPU-double G0 and the exact G0↔block round trip.
3. Implement CPU-double affine camera compilation and numerical depth
   integration reference.
4. Add the restricted embedding and orthographic parity fixture.
5. Emit the existing UVT ABI and compare brute-force UVT rendering.
6. Verify Metal forward parity and compiler-chain gradients.
7. Train a synthetic restricted ground truth with new DOF frozen.
8. Train analytic full-SPD scenes containing spatial correlation, depth
   thickness, and all three space-time cross terms.
9. Add the persistent T0 activity mode and long-horizon stability fixture.
10. Run real static-camera multiview/heldout comparisons.
11. Add moving-camera derivatives and projective chart splitting.
12. Compare adaptive M1 mixtures against D1 spline/dynamic-covariance tubes.
13. Promote into production only after all correctness and representation
    gates pass.
14. Do not port to the browser until the native model establishes the desired
    quality/memory Pareto point.

Unlock trainable capacity in this order:

1. means, opacity, RGB;
2. spatial diagonal including \(z\)-width;
3. spatial correlations/orientation;
4. space-time cross terms/tilt;
5. temporal activity parameters;
6. mixture splitting;
7. only then time-varying covariance or appearance.

## 7. Correctness and falsification suite

### Algebra

1. Random SPD(4) → block → SPD(4) round trip.
2. Direct 4D exponent equals completed-square tube exponent.
3. Slice conditional mean is affine and conditional covariance is constant.
4. Random trainable parameters always produce valid SPD spatial/full
   covariances in the appropriate mode.
5. Cholesky pack/checkpoint round trip.
6. Exact time/spatial-unit rescaling invariance.

### Compiler

7. Affine covariance pushforward equals precision Schur complement.
8. Monte Carlo affine samples match UVT mean/covariance and conditional depth.
9. Numerical ray integration matches the analytic depth marginal and chosen
   amplitude convention.
10. Affine slice-then-project equals project-then-slice.
11. CPU double, CPU float, brute-force UVT, and Metal agree within declared
    tolerances.
12. Finite differences cover mean, all covariance coordinates, opacity, and
    appearance.

### Counterexamples

13. Static/persistent splat over 16–1024 frames: strict SPD should expose its
    conditioning/activity limitation; T0 should remain stable.
14. Constant acceleration: measure G0/M1 segment count versus trajectory error.
15. Rotating anisotropic splat: one G0 must fail; compare M1 against D1 at equal
    bytes.
16. Orthographic source-depth nullspace: vary unseen depth width, then score a
    side camera.
17. Near-plane support: require denominator/support certificate or fallback.
18. Two translucent layers crossing in depth: detect order failure.
19. Order swap during exposure: show that preintegrated traces cannot replace
    time-sampled compositing without event handling.
20. Split one primitive into children: quantify opacity non-invariance under
    peak-alpha semantics.
21. Many individually tiny omitted tails: bound aggregate rather than only
    per-primitive error.
22. Perturb across chart/support/order events and compare fixed-atlas VJP to
    full finite differences.

### Diagnostics to log

- minimum eigenvalues and condition numbers;
- spatial/temporal principal scales in physical units;
- dimensionless one-sigma displacement
  \(\lVert v\rVert\sigma_t/s_{\rm spatial}\);
- projected UVT principal minors;
- conditional depth mean and variance;
- tile/primitive pairs and temporal support occupancy;
- chart splits, order events, fallbacks, and overflows;
- compiler forward/backward versus raster forward/backward time;
- peak memory split by parameters, optimizer, intermediates, atlas, and data.

## 8. There is no single fair baseline

Fairness has at least two independent axes:

1. **stored scene state over the entire clip**;
2. **active rendering work for one requested frame/output pixel set**.

Per-frame 3DGS may have similar active raster work per frame while storing a
separate bank for every time. A compact tube may store far less but touch broad
temporal/tile support in its atlas. Report both; do not collapse them to “splat
count.”

### Baseline A — causal compiler isolation (paper-primary)

Use the **same learned world primitives and renderer semantics** in two paths:

1. evaluate/project/bin/sort them independently for every queried frame;
2. compile them once into UVT/projective trace records and replay temporal
   queries.

This isolates the compiler's claim: projection/binning reuse, memory, backward
reuse, and numerical equivalence. It does not test whether one representation
class fits video better than another.

The repository already has credible bounded synthetic evidence for this narrow
claim. At \(F=128\), the accepted same-representation chart reports
fixed/replay ratios of 0.03125 for payload, about 0.0477 for CPU compilation,
0.181 for forward, and 0.392 for backward. This establishes amortization on
that path; it does not establish public-scene quality, universal camera/event
scaling, or native-resolution readiness.

Report the break-even query count

\[
F^*=\frac{C_{\rm compile}}
{c_{\rm replay}-c_{\rm compiled}},
\]

and distinguish offline compilation, where the future camera program is known,
from causal-prefix compilation, where the atlas is incrementally extended. An
offline compiler with future-camera lookahead is not a causal replacement for
online replay.

### Baseline B — equal active primitives/raster entries

Compare T0/G0 with independent per-frame 3DGS using the same active primitive
count \(N\) and output workload at each frame. This is informative for quality
and raster cost, but per-frame storage is unmatched.

### Baseline C — equal total learned scalars/bytes

With simple RGB/opacity:

```text
restricted World Tube: 14 floats / primitive
full G0/T0:             18 floats / primitive
per-frame 3DGS:         14 floats / splat / frame
```

At equal total stored scalars,

\[
G_{\rm perframe}=
\left\lfloor\frac{18N_{G0}}{14T}\right\rfloor.
\]

For \(T=300\), this permits only about \(N_{G0}/233\) splats per frame. Such a
single point is often too starved to be useful, which is why the honest result
is a storage-quality Pareto curve rather than one allegedly fair run.

At equal learned scalar budget between current restricted and full G0,

\[
N_{G0}=\left\lfloor\frac{14}{18}N_{\rm restricted}\right\rfloor.
\]

Also compare equal \(N\), because four extra floats are a small model-capacity
increment and equal-parameter matching changes coverage.

### Baseline D — generous per-frame upper bound

Give independent per-frame 3DGS enough splats to fit each frame well. Treat it
as a quality ceiling with expensive storage, not as a compression-matched
competitor.

### Baseline E — same representation, compiled versus replayed

For the strongest systems result, initialize a G0/T0 scene once and compare
the exact same state through:

- ordinary per-frame time slice + 3D projection/bin/sort;
- compiled gauged UVT trace atlas.

This is stricter than comparing against an independently trained per-frame
model because quality differences should vanish within approximation/error
certificates.

### Baseline F — model-class peers

At equal bytes and training data, include:

- native full 4D Gaussian with four scales + quaternion pair;
- STG-style temporal Gaussian with polynomial center/rotation;
- Dynamic 3D Gaussians-style persistent moving/rotating 3DGS;
- neural deformation-field 4D-GS;
- M1 piecewise full-Gaussian mixture;
- D1 spline center/rotation/scale tube;
- direct screen-UVT full Gaussian as a source-chart upper bound, clearly not a
  camera-portable world baseline.

## 9. Required benchmark controls

Hold fixed or report:

- train/validation/heldout-camera split;
- input frames and camera coverage;
- samples per optimizer step and temporal sampling policy;
- rendered pixels/resolution per sample;
- optimizer steps, wall time, and convergence curves;
- initialization and densification/splitting policy;
- appearance basis and opacity convention;
- screen-space filter and compositing rule;
- total learned scalars/bytes and optimizer bytes;
- active primitives, tile pairs, and pixels touched per frame;
- compile/rebuild/fallback counts;
- random seeds.

Score source-view and heldout-camera PSNR/SSIM/LPIPS separately. A model that
uses newly free depth covariance to improve training views while degrading
heldout cameras has exploited the exact depth nullspace, not learned a better
world representation.

## 10. Current memory/result interpretation

The paper's free dynamic baseline allocates tensors shaped approximately
\([T,N,\cdot]\) for position, scales, quaternion, opacity, and RGB. It does not
render all 300 frames in one optimizer batch: the current protocol samples a
small structured set of frames/views per step. Nevertheless, every per-frame
parameter bank is resident and optimizer/autograd/renderer allocations add to
it.

For 1,024 simple-RGB splats over 300 frames, raw parameters are

\[
14\times1024\times300=4{,}300{,}800\text{ floats}
\approx17.2\text{ MB (FP32)}.
\]

That raw parameter table alone cannot explain tens of gigabytes of peak unified
memory. Render intermediates, autograd graphs, optimizer state, image/camera
data, MPS allocations, caching, and allocator behavior dominate the observed
peak. Conversely, the browser's explicitly retained per-sample gradient tape
can be hundreds of MiB by itself. “Lean Metal kernel” does not imply lean
end-to-end training memory if the surrounding tape and batches are large.

In the accepted run, parameters plus Adam account for only about 51.6 MB for
the bank versus about 0.17 MB for the restricted tube, far below the roughly
17.4 GB driver-peak gap. The compiled World Tubes runtime can itself hold a
large atlas/cache (roughly 455 MiB in the audited full runtime), despite its
56 KiB checkpoint. Always report checkpoint bytes, optimizer bytes, compiler
cache, renderer scratch/intermediates, and driver/process peak separately.

The current accepted three-seed progressive Coffee Martini row in
[`WORLD_TUBES_PAPER_DRAFT.md`](../gauged_uvt_trace_atlas/paper/WORLD_TUBES_PAPER_DRAFT.md)
reports:

| Current implementation | Heldout PSNR | SSIM | LPIPS | Peak driver | Checkpoint |
|---|---:|---:|---:|---:|---:|
| Restricted World Tubes | 5.9153 | 0.03549 | 0.98305 | 3.114 GB | 0.060 MB |
| Independent per-frame 3DGS | 4.9110 | 0.28267 | 0.90228 | 20.557 GB | 17.206 MB |

These rows are incomplete and all absolute quality is poor. Metrics disagree:
the restricted tube has better PSNR/L1, while per-frame 3DGS has better
SSIM/LPIPS. Thus the evidence does **not** show that per-frame 3DGS is simply
better, nor does it establish that World Tubes are already superior. It shows
a large storage/peak-memory difference and a representation/optimization
comparison that remains unresolved. Fixed pixel controls, sampler controls,
more camera triplets/scenes, and better absolute fits are still required.
The budgeted wall times happened to be similar (about 78.3 versus 79.4 seconds)
because only four frames were sampled/rasterized per optimizer step; this does
not make the \(O(TN)\) resident bank a constant-storage representation.

## 11. Starting acceptance and kill gates

Thresholds below are predeclared starting points; change them only before a run
matrix, not after seeing outcomes.

### Correctness gates

- no invalid SPD construction over random stress and 1,000 optimizer steps;
- CPU-double Schur/numerical-integration relative error below \(10^{-5}\);
- restricted orthographic render max error below \(10^{-4}\);
- Metal/reference agreement within the existing renderer tolerance;
- gradient relative error below \(10^{-3}\), away from discrete events;
- physical render invariance under exact time/spatial unit conversion.

### Implementation gates

- no \(O(TN)\) world-state materialization in the compact path;
- peak memory at equal \(N\) no more than 1.25× the restricted path, excluding
  separately reported support/tile-pair growth;
- compiler under 10% of the target step time or else a fused/checkpointed VJP
  is required before scaling;
- frozen restricted embedding within 0.05 dB of the current implementation.

### Representation gates

- full covariance must materially beat the restricted embedding on analytic
  scenes designed to require the missing DOF;
- on a predeclared real-scene/seed matrix, target either ≥0.25 dB median
  heldout-PSNR gain at equal \(N\), or equal quality within 0.1 dB with ≥15%
  fewer primitives;
- reject source-view gains paired with >0.2 dB heldout loss;
- if learned support produces >2× tile pairs without reaching the quality
  threshold, kill that unconstrained-support configuration;
- if one-chart nonlinear projection exceeds 0.25-pixel support residual for
  significant active mass, require splitting/fallback rather than hiding the
  approximation;
- promote D1 over M1 only if it delivers the same correctness/quality with a
  material byte or atlas-event reduction across several curvature/rotation
  fixtures.

Failures are scoped. Failure of one opacity convention does not kill the
geometry. Failure of one camera chart does not kill the world primitive.
Failure of the current `precision_xy` model does not kill full spacetime
Gaussians.

## 12. Decision summary

1. Restore full spatial SPD(3); that is the missing capacity in the current
   world scaffold.
2. Keep spacetime tilt/velocity; it is the exact coordinate form of the three
   space-time covariances.
3. Do not add a separate 4D rotation alongside a full covariance.
4. Support localized strict SPD(4) atoms and a typed persistent
   \(\lambda_t=0\) boundary.
5. Keep world geometry separate from camera-specific UVT compiler output.
6. Use time slicing semantically, but compile/project through the ray bundle
   rather than rebuilding a per-frame bank.
7. Try adaptive mixtures before a fully dynamic \(C(t)\), while retaining the
   latter as the falsifying matched baseline.
8. Decide peak-opacity versus mass/extinction semantics explicitly.
9. Require novel-view pressure when unfixing depth covariance.
10. Report storage and active render work separately; no single “splats per
    frame” comparison is fair.
