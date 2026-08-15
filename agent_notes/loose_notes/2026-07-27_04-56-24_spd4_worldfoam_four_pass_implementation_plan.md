# Four-pass implementation review: full SPD(4) World Tubes and finite-element WorldFoam

- Time: 2026-07-27 04:56:24 +0900
- Role: primary implementation/research coordinator
- Method: deep-critical-thought skill, with independent STAR, WorldFoam-shader,
  and paper/test audits
- Objective: turn the representation discussion into a bounded, checkable code,
  shader, paper, and experiment change set
- Why attempted: the research notes now contain several mathematically plausible
  objects, but the next engineering step was still ambiguous and risked either
  duplicating an already implemented STAR back end or proliferating WorldFoam
  renderer forks

## Inputs used

- Current source and tests under `star_uvt_v0`, especially
  `projective_trace.py`, `star_uvt_kernels.metal`, and the projective interval
  tests.
- Current WorldFoam owner-segment tape, fused-slab Metal source, optical-transfer
  fixture, experiment notes, and paper drafts.
- The representation audits under
  `research_notes/spacetime_gaussian_representation/`.
- The finite-element and method-classification notes under
  `research_notes/worldfoam_paper/`.
- The post-incident local-MPS safety rules in `TODO/README.md`.
- Three independent code/paper audits performed during this session.

## Invariants frozen before planning

1. A strict native spacetime Gaussian is a mean in \(\mathbb R^4\) plus a
   strict SPD(4) covariance. Velocity is a derived coordinate of the space-time
   cross covariance, not a separate trajectory law.
2. For an affine camera-ray gauge, marginalizing the ray fiber gives an exact
   UVT Gaussian. The conditional depth mean is affine in \((u,v,t)\), and the
   conditional depth variance is a positive scalar.
3. STAR already evaluates anisotropic UV precision and pixel-dependent affine
   depth in its interval Metal shader. Reimplementing that renderer would be
   duplication.
4. A certified hard-order cell may use the existing fast alpha compositor. A
   cell with overlapping conditional depth bands and materially different
   colors needs a retained-fiber optical fallback; merely sorting means is not
   exact.
5. Every WorldFoam material law must lower one physical segment to the same
   optical element. For constant source color this is
   \[
     (\beta,m)=\left(e^{-\tau},(1-e^{-\tau})c\right).
   \]
6. Positive Bernstein P2 extinction is a mandatory counterbaseline to log-P2:
   both use three ray-segment controls, while the positive polynomial has a
   cheaper exact integral.
7. No publication-scale local MPS run is authorized in this session.

## Pass 1 — broad architecture proposal

### Proposed work

- Replace the restricted `WorldTubeBatch` with a full SPD(4) trainable object.
- Extend the STAR shader ABI with full covariance, conditional depth mean, and
  conditional depth variance.
- Implement P0, affine-color P0, positive P1/P2, log-P1/P2, and the convex
  potential atom as separate WorldFoam shader variants.
- Add full public 300-frame comparisons after implementation.

### Review and rejection

This plan was too broad and contained two false premises.

- `star_uvt_kernels.metal` already consumes `spatial_precision_uv` and
  `depth_affine_uv` in interval forward and backward. A full-covariance shader
  fork would move camera/compiler algebra into the wrong layer.
- Six WorldFoam renderer forks would confound material cost with geometry,
  event-tape, and dispatch differences.
- Replacing `WorldTubeBatch` would break a large historical caller surface.
- The convex-potential atom has unresolved minimizer/root compilation cost and
  is not ready for native allocation.
- Full 300-frame MPS runs would violate the post-incident safety gate.

### Result of pass 1

Backtrack. Preserve both existing renderers. Split the work into two vertical
slices with explicit source-to-ABI boundaries.

## Pass 2 — two vertical slices

### Proposed work

#### Slice A: full SPD(4) World Tubes source

- Add an isolated strict SPD(4) batch and exact affine camera-ray lowering.
- Produce the existing UVT fields plus conditional depth variance.
- Add variance-aware separation certificates.
- Adapt certified traces to the unchanged STAR interval atlas.

#### Slice B: WorldFoam material matrix

- Reuse one fixed owner-segment tape.
- Implement M0--M5 behind one material-mode ABI.
- Return a transfer element and analytic coefficient VJP.
- Compare material laws before integrating any winner into optimized endpoint
  record kernels.

### Mathematical review

The split is correct, but the initial parameter semantics were still
underspecified.

For Slice A, let the affine gauge map world spacetime into
\(y=(u,v,z,t)\). With
\[
 \mu_y=G\mu_x+b,\qquad \Sigma_y=G\Sigma_xG^\top,
\]
write \(r=(u,v,t)\) and depth \(z\). Then
\[
 \Sigma_r=\Sigma_y[r,r],\qquad Q_r=\Sigma_r^{-1},
\]
\[
 E[z\mid r]=\mu_z+\Sigma_{zr}Q_r(r-\mu_r),
\]
\[
 \operatorname{Var}(z\mid r)
 =\Sigma_{zz}-\Sigma_{zr}Q_r\Sigma_{rz}.
\]
This is the exact packet expected by the existing UVT/conditional-depth path.
The opacity convention must be explicit: peak-preserving splat amplitude and
physical fiber-integrated density differ by
\(\sqrt{2\pi\,\operatorname{Var}(z\mid r)}\) times the fiber-measure Jacobian.

For Slice B, direct positive P1/P2 controls must be Bernstein controls, not
ordinary nonnegative Lagrange nodal values. Nonnegative quadratic Lagrange
samples do not guarantee positive density between nodes. Convex log-P2 must
initially enforce nonnegative quadratic coefficient in negative log density;
otherwise Metal needs a stable `erfi`/Dawson branch.

### Result of pass 2

Accept the two slices, but freeze typed semantics and install counterbaselines
before any production-trainer integration.

## Pass 3 — ABI, VJP, and test design

### Proposed interfaces

#### SPD(4) reference/compiler

- `WorldAtomBatch`: `mean_xyzt`, `covariance_xyzt`, opacity, color.
- `AffineRayGauge`: a nonsingular affine map to ordered coordinates
  `(u,v,depth,t)`, plus a fiber-measure scale.
- `FiberTrace`: UVT mean and precision, affine conditional-depth coefficients,
  conditional depth variance, amplitude scale, and source/gauge metadata.
- A block-Cholesky constructor for all ten covariance degrees.
- Confidence-band order certification:
  \[
  [\hat z-k\sigma_z-\delta_{\rm fit},
   \hat z+k\sigma_z+\delta_{\rm fit}].
  \]

#### WorldFoam material ABI

- Normalized segment coordinate \(\xi\in[0,1]\), physical length \(L\).
- Three fixed coefficient slots and one material mode.
- Constant RGB first; affine RGB P0 as the appearance counterbaseline.
- Forward returns \(\tau,\beta,m\), density bounds, and a numerical branch code.
- VJP returns gradients with respect to coefficient slots, colors, and length.

### Engineering review

- Do not append unversioned fields to the legacy six-tensor UVT tuple.
- Do not require variance in the fast shader. It belongs in the compiler
  certificate; a certified non-overlap cell renders identically with the
  existing shader.
- The existing fallback only re-sorts conditional means. Name it honestly; it
  is not the retained-fiber optical fallback.
- The material microkernel should be a new small source loaded by the existing
  WorldFoam experimental extension or by a tiny dynamic-compile wrapper. It
  should not clone the PowerFoam or fused-slab renderer.
- Metal on this host does not expose `erf`, `erfc`, `expm1`, or `log1p`.
  Log-P1 therefore needs a local limiting series. Convex log-P2 needs a local
  erf approximation plus a stable small-curvature series or a clearly marked
  fixed-quadrature fallback.
- For nonuniform extinction, expected depth requires an additional absorption
  moment. The first native comparison should therefore use the existing
  RGB-only fixed-tape loss seam; it must not pretend that `(beta,m)` alone
  preserves the old expected-depth output.
- The existing PowerFoam `max_alpha` path has a clamp/update inconsistency.
  Exact transfer tests must use one consistent `log_beta=-tau` update.

### Test design

1. Float64 SPD(4) algebra, joint-factorization, gauge, reconstruction, and
   gradient tests.
2. UVT adapter parity with the current reference renderer.
3. Thick-depth overlap tests where separated means but overlapping confidence
   bands force fallback.
4. Float64 material integrals against independent quadrature.
5. Analytic VJP against autograd and central differences.
6. Tiny Metal compile/forward/VJP parity only; no training or public matrix.
7. Existing STAR and WorldFoam CPU regression subsets.

### Result of pass 3

The interfaces are coherent. The remaining risk is claiming an end-to-end
paper method from foundation kernels alone.

## Pass 4 — final scoped implementation contract

### Implement now

1. An isolated, differentiable SPD(4) reference package under
   `research_experiments/spd4_world_tubes/`:
   - lossless block-Cholesky construction;
   - exact affine gauge pushforward;
   - exact UVT marginal and conditional depth mean/variance;
   - explicit peak-preserving versus fiber-integrated amplitude semantics;
   - confidence-band order certificates;
   - adapter to the existing STAR UVT atlas;
   - dense retained-fiber reference for falsification, not a production claim.
2. An isolated WorldFoam finite-element material reference under
   `research_experiments/world_foam_lane2/`:
   - M0 P0, M1 affine-color P0, positive Bernstein P1/P2, log-P1, and convex
     log-P2;
   - shared `(beta,m)` element;
   - analytic/series segment integrals and VJPs;
   - explicit numerical branch reporting.
3. One parameterized Metal segment-material source and a thin dynamic-compile
   wrapper for tiny side-by-side forward/VJP parity. This is the shader fork:
   fork the *segment material evaluator*, not the renderer.
4. Unit tests, a safe fixed-tape benchmark runner, and schema-rich JSON output.
5. Paper updates that distinguish:
   - existing implemented STAR back half;
   - newly implemented SPD(4) source/reference/compiler slice;
   - newly implemented WorldFoam material microkernel;
   - still-missing retained-fiber production fallback, compact native-4D
     WorldFoam field/compiler, trainer integration, and public evidence.

### Explicitly defer

- Curved/spline world tubes.
- Convex-potential atoms.
- A production retained-fiber STAR Metal fallback.
- Porting every material mode into endpoint-record/framegroup kernels.
- A compact global 4D FEM mesh/trainer.
- Publication-scale local MPS and public multi-scene claims.

These are not silently dropped. They become continuation gates after the two
foundational slices pass parity and show value.

## Acceptance thresholds for this session

- SPD(4) float64 algebra and reconstruction: max error \(10^{-10}\) on
  well-conditioned cases.
- Material reference versus independent quadrature: max error \(10^{-10}\) in
  float64 on the declared domain.
- VJP versus autograd/finite differences: relative error \(10^{-5}\).
- Tiny Metal forward: max absolute error \(10^{-5}\).
- Tiny Metal VJP: normalized error \(10^{-4}\).
- No NaN/Inf in accepted-domain fixtures; all fallback/approximation branches
  counted.
- Existing CPU regressions remain green.

## Promotion and kill criteria

- Full SPD(4) advances to a trainable STAR producer only if the adapter is exact
  and a controlled tilt/depth-width scene exposes a real capacity gain over the
  restricted model.
- A WorldFoam material advances to optimized owner-tape kernels only if it beats
  M0 and the affine-color control at matched bytes or matched quality.
- Log-P2 is killed as the default if positive Bernstein P2 matches its quality
  with materially lower segment/VJP cost.
- No GFEM paper claim is promoted until a compact native-4D field/compiler
  removes the current per-frame parameter scaling.
- No result from a tiny Metal parity run is described as a systems-speed or
  quality result.

## Claim status at plan freeze

- Full SPD(4) affine FiberTrace algebra: known linear-Gaussian identity; to be
  independently tested in code.
- Motion from spacetime cross covariance: proved linear-Gaussian identity.
- Existing STAR per-pixel affine depth shader: implementation fact verified by
  source inspection.
- Variance-aware band certificate: sufficient conservative criterion; not a
  necessary condition.
- Retained-fiber fallback requirement for thick differently colored overlap:
  proved by ordered optical-transfer counterexamples; production implementation
  absent.
- Positive Bernstein P2 as cheaper counterbaseline: derived cost/DOF argument;
  empirical winner unknown.
- GFEM memory/quality advantage: question, not a result.

## Next actions after this session

1. If both references and the tiny shader parity gate pass, integrate the
   winning material law into the existing RGB-only segment-tape fused loss.
2. Add the strict SPD(4) trainable producer behind a new explicit config value,
   retaining the restricted producer as the default.
3. Build the retained-fiber fallback only for cells rejected by uncertainty
   certificates.
4. Run controlled synthetic capacity and material-value scenes.
5. Request explicit approval before any bounded local-MPS training run.
6. Only then add schema-versioned paper-runner axes and public experiments.

## Validation log

- Pre-change CPU regression, 2026-07-27:
  `test_star_uvt_projective_uvt_producer.py`,
  `test_star_uvt_projective_visibility.py`, and
  `test_cell_path_optical_transfer_fixture.py`: **60 passed** in 35.82 s.
- No MPS work was included in this baseline.

## Implementation closeout

### Slice A: strict SPD(4) World Tubes source

Implemented under `research_experiments/spd4_world_tubes/`:

- lossless ten-DOF block-Cholesky SPD(4) chart;
- exact affine ray-gauge pushforward;
- exact UVT marginal and affine conditional depth mean/variance;
- explicit peak-density and fiber-integrated amplitude coefficients;
- exact affine-box confidence-band order certificate;
- dense retained-fiber emission-absorption oracle;
- structural six-field STAR adapter with explicit `peak_preserving` and
  `thin_fiber_optical_depth` mappings.

Important correction found during final audit: the geometric STAR lowering is
exact, but physical fiber opacity is not representable by the unchanged
factorized alpha law. STAR evaluates

\[
\alpha_{\rm STAR}(r)=o\exp[-q(r)/2],
\]

whereas physical optical depth gives

\[
\alpha_{\rm phys}(r)=1-\exp\{-\tau_0\exp[-q(r)/2]\}.
\]

Using `tau_0` as STAR opacity is only first-order exact. The default adapter is
therefore peak-preserving, and the physical thin-limit mapping is named
explicitly. A Beer--Lambert alpha mode/VJP and retained-depth colored-overlap
fallback remain production work.

CPU validation: `tests/test_spd4_world_tubes.py` is **11 passed**, including
actual lowering through the existing projective interval-atlas producer.

### Slice B: WorldFoam M0--M5 material transfer

Implemented under `research_experiments/world_foam_lane2/`:

- one shared fixed-segment `(tau,beta,m,density_bounds,branch_status)` ABI;
- M0/M1 P0, positive Bernstein P1/P2, log-P1, and convex log-P2;
- explicit coefficient/color/length VJP;
- one parameterized Metal forward/VJP source, not renderer forks;
- a fail-loud dynamic Metal wrapper;
- a schema-rich CPU/default and opt-in tiny-Metal gate runner.

An independent red team falsified the first GL16 fallback. For
`q(x)=1000(x-1/2)^2`, GL16 returned `0.01983175` instead of `0.05604991`.
That implementation was not preserved or silently relabeled. It was replaced
by:

1. sign-aware analytic erf evaluation for intervals straddling the Gaussian
   peak;
2. scaled `erfcx` endpoint-tail differences for same-sign tails;
3. endpoint-scaled log-linear moments that avoid separately forming
   overflowing `exp(-b)` and cancelling `exp(-c)`;
4. explicit invalid-row rejection in the host wrapper.

The sharp interior cases `a=1000` and `a=10000` are now regression tests.
M1 accepts either two explicit endpoint colors or one aliased color. In the
aliased case, the VJP now accumulates both endpoint contributions into the one
supplied color gradient. Forward/VJP share the same finite-domain rejection.

Final focused CPU validation is **37 passed, 2 skipped** across the SPD(4) and
material suites; the skips are only the opt-in Metal tests.

Including the pre-existing STAR UVT producer/visibility and WorldFoam
optical-transfer regression subsets, the combined CPU result is **97 passed, 2
skipped**.

The explicitly authorized 12-segment mechanical Metal gate then passed:

```text
CPU independent-quadrature max abs error  5.96e-15
CPU explicit-VJP normalized error         5.55e-17
Metal forward normalized error            7.51e-8
Metal VJP normalized error                5.96e-8
invalid rows                              0
current MPS allocation                    4,608 bytes
sampled MPS driver allocation             28,016,640 bytes
```

Artifact:
`artifacts/foundation_gates/worldfoam_material_m0_m5_cpu_metal_20260727.json`.
This was a sub-kilobyte record set plus compiler/runtime overhead, not training,
throughput evidence, or a publication-scale run.

Post-red-team validation tightened same-sign M5 evaluation further: approximate
`erf` subtraction is now used only when the transformed interval straddles the
Gaussian peak; every same-sign interval uses the scaled-`erfcx` identity.
Material-only validation is **37 passed, 3 skipped** on CPU and **40 passed**
with the explicitly enabled tiny Metal fixtures. The refreshed 12-record gate
retains the errors above and records one nonzero small-\(\tau\) series row.
A current-source one-seed material-value smoke is saved at
`artifacts/foundation_gates/worldfoam_material_value_fit_cpu_postfix_smoke_20260727.json`;
all preregistered smoke checks pass, but it does not replace the existing
three-seed material-value artifact.

### Claim status after implementation

- strict SPD(4) affine compilation: **implemented reference; CPU verified**;
- affine motion from space-time cross covariance: **implemented and tested**;
- variance-aware ordering: **implemented sufficient certificate; CPU tested**;
- exact physical STAR opacity from a density atom: **missing production mode**;
- retained-fiber differently colored overlap: **implemented dense oracle only**;
- M0--M5 local material algebra/VJP: **CPU and tiny Metal parity verified**;
- compact native-4D WorldFoam field/compiler: **missing**;
- trained quality, quality-per-byte, and end-to-end speed wins: **open
  questions**.
