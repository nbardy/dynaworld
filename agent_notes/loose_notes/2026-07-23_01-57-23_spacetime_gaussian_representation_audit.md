# Spacetime Gaussian representation audit

- **Time:** 2026-07-23 01:57 +0900
- **Author/role:** Codex coordinator with six specialist work waves
  (repository archaeology, Gaussian geometry, implementation/code audit,
  mathematical red team, implementation/experiment design, and
  literature/baseline audit)
- **Objective:** recover all DynaWorld/STAR/World Tubes spacetime-Gaussian
  formulations; answer the exact parameter/rotation/slicing questions; locate
  where the active implementation narrowed; compare 10–20 candidate model
  classes; and freeze a falsifiable representation/experiment recommendation.
- **Why attempted:** current discussion was treating a restricted
  `x0 + velocity + precision_xy + lambda_t` scaffold as if it were the final
  definition of a World Tube. The user remembered a cleaner one-center plus
  one-spacetime-covariance formulation and asked for a full audit before adding
  splines or more time-varying parameters.

## Inputs used

Repository-wide targeted Markdown inventory and full reads of relevant hits,
including:

- `third_party/fast-mac-gsplat/variants/spacetime_v0/docs/handoff.md`
- `third_party/fast-mac-gsplat/variants/star_uvt_v0/README.md`
- `third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/phases/phase_2_world_tube_projection.md`
- `research_experiments/star_uvt_notes.md`
- `research_notes/gauged_uvt_trace_atlas/**`
- `research_notes/gauged_uvt_trace_atlas/paper/WORLD_TUBES_PAPER_DRAFT.md`
- `research_notes/paper_training_protocol_v1.md`
- April material-flow, polynomial, low-rank, and architecture notes
- current July browser anisotropic/dynamic/STAR notes
- active standard 3DGS, per-frame dynamic, World Tube, STAR UVT, projective,
  Metal ABI, and browser implementation paths
- Git/submodule history available in the checkout

Primary external references used:

- Kerbl et al. 3DGS paper and official code
- Yang et al. native 4D Gaussian Splatting (ICLR 2024)
- Li et al. Spacetime Gaussian Feature Splatting (CVPR 2024)
- Luiten et al. Dynamic 3D Gaussians
- Wu et al. deformation-field 4D-GS

The repository had 1,559 Markdown files before the durable notes from this
audit were added. Every Markdown path was included in the recursive inventory;
representation-related hits were deeply read. Unrelated UI, dataset, product,
and infrastructure notes were not linearly reread. No input-isolation constraint
applied to this context-rich audit.

## Central recovered evidence

### Status: implementation fact

The elegant native object was not deleted or rejected. The spacetime handoff
stores one `mu: float4` and one full `Q: float4x4`, then ray-integrates the world
Gaussian into a sensor-time Gaussian. Its acceptance tests explicitly require
constant-velocity 3D support to equal one tilted 4D Gaussian.

### Status: implementation fact

Phase 2 writes the exact precision normal form

\[
\Lambda_{xx}=A,
\quad
\Lambda_{xt}=-Av,
\quad
\Lambda_{tt}=v^\top Av+\lambda_t.
\]

It calls its implemented fronto-parallel projection intentionally weaker and
states that full anisotropic 3D covariance integration is missing. Active
`WorldTubeBatch` collapses \(A\) to `precision_xy[2]`; world \(z\)-thickness,
full spatial correlations, and spatial orientation are absent.

### Status: implementation fact

Feature STAR already has the remembered clean \(m_{uvt}\) plus full symmetric
\(Q_{uvt}\) representation in screen-time. This is a full Gaussian in
\((u,v,t)\), but it is view/chart-specific and has already summarized the ray
depth fiber. It is compiler output, not a portable world asset.

### Status: implementation fact

Later STAR notes restore a gauge-charted world primitive with
\(\eta\in\mathbb R^4\) and \(\Lambda\in\mathbb S_{++}^4\), and the gauged atlas
notes derive the camera-ray pullback/depth pushforward
\(\pi_*\Gamma^*\rho\). The active restriction was an implementation gate, not
a theory reversal.

## Derivations

### Status: proved lemma — full SPD(4) equals a full affine Gaussian tube

Let

\[
\Sigma_4=
\begin{bmatrix}A&b\\b^\top&c\end{bmatrix}\succ0,
\qquad
v=b/c,
\qquad
C=A-bb^\top/c.
\]

The Schur complement gives \(C\succ0\), and block inversion yields

\[
\Sigma_4^{-1}=
\begin{bmatrix}
C^{-1}&-C^{-1}v\\
-v^\top C^{-1}&c^{-1}+v^\top C^{-1}v
\end{bmatrix}.
\]

Therefore

\[
\rho(x,t)=
\alpha e^{-(t-t_0)^2/(2c)}
e^{-\frac12(x-x_0-v(t-t_0))^\top
C^{-1}(x-x_0-v(t-t_0))}.
\]

Conversely,

\[
\Sigma_4=
\begin{bmatrix}
C+cvv^\top&cv\\cv^\top&c
\end{bmatrix}
\]

is SPD for every \(C\succ0,c>0,v\). Hence the mapping is bijective. Velocity
is exactly the three space-time covariance DOF, not an additional ad hoc law.

### Status: proved lemma — rigidity of a single joint Gaussian

At fixed time, a joint Gaussian precision has spatial quadratic coefficient
\(Q\) independent of time. Completing the square makes the center affine and
leaves covariance \(Q^{-1}\) constant. One joint Gaussian cannot intrinsically
accelerate, rotate its 3D conditional ellipsoid, or change its spatial scales.

### Status: proved lemma — strict persistence lies on the PSD boundary

Constant temporal activity requires \(\lambda_t=0\). Then

\[
\Lambda_4(v,1)^\top=0,
\]

so the precision is semidefinite. This is a persistent Gaussian cylinder/tube,
not a normalizable SPD(4) ellipsoid over an unbounded timeline. It is valid on
a bounded clip with explicit support. Practical T0 should support this typed
boundary instead of representing static content with an ill-conditioned huge
temporal covariance.

### Status: known theorem / checked application — rotation

SPD(4) has ten DOF: four eigen-scales plus six \(SO(4)\) orientation DOF. A
general \(SO(4)\) rotor is represented by a pair of unit quaternions:

\[
p\mapsto q_Lpq_R^{-1}.
\]

Eight stored components minus two unit constraints give six effective DOF.
An octonion is neither necessary nor a minimal \(SO(4)\) parameterization. The
fixed 4D orientation is not a time-varying physical 3D quaternion \(q(t)\).

### Status: proved counterexample — full depth covariance is not monocularly identified

For orthographic projection along \(z\),

\[
\operatorname{diag}(s_x^2,s_y^2,\varepsilon^2)
\quad\text{and}\quad
\operatorname{diag}(s_x^2,s_y^2,M^2)
\]

have identical training-view footprints and arbitrarily different side views.
Restoring the missing \(z\) covariance is mathematically correct but must be
paired with heldout-camera pressure or shape/rate priors.

### Status: proved distinction — slicing

- Raw fixed-time restriction preserves temporal activity and is the correct
  instantaneous field.
- Normalized conditioning erases activity.
- Time marginalization makes a static trajectory smear.
- Shutter integration must generally happen after rendering/compositing;
  visibility makes it nonlinear.

For affine camera charts, slice and Gaussian pushforward commute. Perspective,
support/chart changes, depth order, and exposure require approximations,
certificates, subdivision, or fallback.

## Parameter answers frozen by the audit

### Standard 3DGS

```text
mean                 3 DOF
SPD(3) covariance    6 DOF
geometry             9 DOF
log-scales           3 stored/effective
quaternion           4 stored, 3 effective
peak opacity         1
simple RGB           3
total                14 stored, 13 effective
```

Log-scales are unconstrained logs of principal standard deviations;
covariance eigenvalues are `exp(2 * log_scale)`. Opacity is an independent peak
multiplier, while covariance controls position-dependent falloff.

### Full spacetime Gaussian

```text
mean4                4 DOF
SPD(4) covariance   10 DOF
geometry            14 DOF
peak opacity         1
simple RGB           3
total                18 effective
```

The equivalent physical block is full spatial SPD(3) \(C\) (6), tilt/velocity
\(v\) (3), temporal variance (1), spatial reference center (3), and temporal
center (1).

## Representation recommendation

### Status: recommendation

1. Use strict full \((\mu_4,\Sigma_4)\) as G0, the exact finite-lifetime
   mathematical atom and literature/control baseline.
2. Use practical T0 as full spatial SPD(3) + affine tilt + typed persistent or
   localized Gaussian activity. This is full SPD(4) when localized and its
   semidefinite closure when persistent.
3. Optimize in a Cholesky-safe conditional precision/covariance chart with
   declared physical space/time normalization.
4. Compile world state to gauged UVT records with conditional depth variance,
   chart/support validity, and order/fallback sidecars.
5. Use adaptive mixtures/piecewise chains as the first curvature/rotation
   extension.
6. Keep spline \(m(t),R(t),\ell(t)\) as a matched falsifier and promote it only
   if mixtures require materially more bytes or atlas events.
7. Freeze peak-opacity versus mass/extinction semantics before implementation.

## Representation families considered

Twenty families were compared in the durable catalog: static 3DGS,
independent/per-track per-frame banks, polynomial STG, low-rank bases, grouped
SE(3), neural deformation fields, transported covariance, diagonal/full 4D
Gaussians, persistent PSD tubes, current restricted World Tubes, direct UVT,
piecewise UVT, projective traces, mixtures, B-spline centers, Lie-spline
covariance tubes, gauge-charted Gaussians, and the general ray-bundle trace
atlas.

## Fair-baseline result

### Status: recommendation

There is no single fair baseline because total stored clip state and active
per-frame rendering work are different axes.

Required suite:

1. same learned world primitives, compiled atlas versus per-frame replay —
   isolates the compiler claim;
2. same active primitives per frame — quality/raster comparison, storage
   unmatched;
3. same total scalars/bytes — storage comparison, often leaves per-frame banks
   too few splats;
4. generous independent per-frame 3DGS — quality upper bound;
5. native full 4DGS, STG, Dynamic 3D Gaussians, deformation 4D-GS, adaptive
   mixture, and spline/dynamic-covariance peers at equal bytes.

For simple RGB, full G0/T0 stores 18 floats/primitive versus 14 for one 3DGS
state. A per-frame bank over \(T\) frames stores \(14TN\). At equal total
scalars, its splats per frame are

\[
G=\left\lfloor\frac{18N_{G0}}{14T}\right\rfloor.
\]

Current training samples four structured frames/views per optimizer step, not
all 300 frames simultaneously, but all per-frame banks and Adam state are
resident and some losses span all \(T\). For 1,024 splats over 300 frames, raw
parameters are about 17.2 MB; that does not explain a ~20 GB MPS peak by itself.
Autograd/render intermediates, optimizer/data state, allocator behavior, and in
the browser an explicit sample-gradient tape dominate. Current mixed quality
rows do not prove per-frame 3DGS is the better representation.

## Failed or rejected shortcuts preserved

- **Refuted:** “velocity proves the primitive is not a full 4D Gaussian.” With
  a full spatial block, it is an exact coordinate decomposition.
- **Refuted:** “four axes plus four widths are the full 4D geometry.” That
  diagonal model has no motion cross-covariance.
- **Refuted:** “add an octonion for 4D rotation.” A pair of quaternions or
  covariance/Cholesky is correct.
- **Refuted:** “full 4D covariance lets the 3D covariance change with time.” A
  single joint Gaussian's conditional covariance is constant.
- **Refuted:** “UVT center/covariance is the canonical world object.” It is a
  camera-specific compiled object with depth removed/summarized.
- **Refuted:** “all 300 frames are trained/rendered at once.” Four structured
  samples are rasterized per step in the current protocol, though all state is
  resident.
- **Not authorized:** blindly unfixing depth covariance, rotation, scale,
  curved motion, appearance, and opacity together. That would confound
  representation, identifiability, compiler, and renderer changes.
- **Preserved failure:** a strict SPD(4)-only universal primitive cannot express
  exact persistent opacity; the persistent boundary must be explicit.

## Assumptions and unresolved questions

- **Open:** peak alpha versus mass-normalized density versus extinction
  semantics for ray-depth pushforward.
- **Open:** thin/surfel prior versus free depth thickness for real scenes.
- **Open:** minimal sufficient visibility sidecar under overlapping depth
  distributions.
- **Open:** whether mixtures or a spline \(C(t)\) win at equal bytes on real
  rotating/curved motion.
- **Open:** how to preserve alpha exactly under primitive temporal splitting in
  standard sorted-alpha semantics.
- **Open:** chart split thresholds and differentiating discrete
  support/order events.
- **Assumption:** a declared bounded clip/time chart is part of the asset
  contract.
- **Assumption:** fixed RGB is acceptable only for the first geometry-isolation
  experiment, not as a final appearance conclusion.

## Independent verification still required

### Status: computational evidence completed in this session

A 100-case NumPy float64 stress check generated random SPD(4) covariances and
verified:

- covariance → \((C,v,c)\) → covariance max error
  \(1.78\times10^{-15}\);
- direct inverse versus block precision max error
  \(5.77\times10^{-15}\);
- direct 4D exponent versus completed-square tube exponent max error
  \(3.20\times10^{-14}\);
- covariance marginal precision versus precision Schur complement max error
  \(1.61\times10^{-12}\).

For \(\lambda_t=0\), the tested precision annihilated \((v,1)\) to
\(6.94\times10^{-18}\) and had one numerical zero eigenvalue, confirming the
persistent semidefinite boundary.

### Still required

- production CPU-double test-harness coverage at extreme scales/condition
  numbers and through checkpoint pack/unpack;
- covariance-pushforward versus precision-Schur equivalence.
- numerical depth integration including amplitude factors.
- restricted embedding parity through CPU and Metal.
- time/FPS/spatial-unit invariance.
- source-depth nullspace heldout-camera fixture.
- persistent horizon, acceleration, rotating covariance, visibility swap,
  exposure swap, opacity split, support-tail, and event-gradient fixtures.
- primary-paper reproduction of native 4DGS/STG/D3DG parameter counts in the
  exact benchmark implementation chosen.

## Durable outputs

- [`research_notes/spacetime_gaussian_representation/README.md`](../../research_notes/spacetime_gaussian_representation/README.md)
- [`01_foundations.md`](../../research_notes/spacetime_gaussian_representation/01_foundations.md)
- [`02_slicing_projection_and_opacity.md`](../../research_notes/spacetime_gaussian_representation/02_slicing_projection_and_opacity.md)
- [`03_repository_archaeology.md`](../../research_notes/spacetime_gaussian_representation/03_repository_archaeology.md)
- [`04_formulation_catalog.md`](../../research_notes/spacetime_gaussian_representation/04_formulation_catalog.md)
- [`05_decision_and_experiments.md`](../../research_notes/spacetime_gaussian_representation/05_decision_and_experiments.md)

## Precise next actions

1. Freeze the six semantic choices in the decision document.
2. Build the isolated CPU-double G0/T0 reference and tests.
3. Embed the restricted current model and require orthographic parity with all
   new DOF frozen.
4. Compile to the existing UVT ABI and verify Schur/depth/Metal parity.
5. Run analytic missing-capacity fixtures before any real-video claim.
6. Run equal-\(N\), equal-byte, same-state compiler/replay, and generous
   per-frame Pareto baselines.
7. Only then choose M1 mixtures versus D1 dynamic covariance.
