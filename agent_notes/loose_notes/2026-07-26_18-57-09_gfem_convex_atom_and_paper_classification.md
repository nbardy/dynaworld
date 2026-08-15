# GFEM, Convex Atom, and Paper Classification Session

**Time:** 2026-07-26 18:57 +0900

**Role:** primary research auditor/deriver, with three independent
mathematical, camera-transfer, and publication/engineering audits.

## Trigger

The user supplied two external-model follow-up dumps and asked to preserve the
best information as durable research notes, decide whether the proposals were
new papers/methods or extensions of existing work, and identify the remaining
paper engineering—especially whether side-by-side Metal shaders were needed.

Inputs:

1. `/Users/nicholasbardy/.codex/attachments/ed37275e-39f1-43bb-b4a7-4f692d8ceb02/pasted-text.txt`
   - SHA-256:
     `9297e456845d7c285e705962631aff2f20c30bf40d56627d10b2215b163407c5`
2. `/Users/nicholasbardy/.codex/attachments/a41b0a04-7c57-44ba-aa3d-95e748406c02/pasted-text.txt`
   - SHA-256:
     `bbafd893ee7579e8b07934b7df355b3a8fbcec970f92f505ce320afb8cf82a01`
3. Existing World Tubes, Gauged UVT, spacetime Gaussian, WorldFoam, renderer
   taxonomy, experiment, and baseline notes.
4. Primary-paper novelty probes listed below.

No training or shader implementation was attempted. This was a theory,
classification, and experiment-design session.

## Why this work was attempted

The external dumps mixed:

- already-developed repository mathematics;
- valid corrections to how that mathematics should be described;
- a substantial new finite-element direction;
- a genuinely distinct compact convex primitive;
- several overstatements about exactness, gauge theory, and sublinear cost.

Without a split audit, the project risked:

- opening redundant paper lanes;
- calling standard/existing transfer theory novel;
- implementing an expensive atom before cheaper counterbaselines;
- confusing a compact parameterization with compact raster work;
- measuring multiple changes in one shader comparison.

## Existing evidence inspected

The audit confirmed that the repository already contains:

- native \(\mu_4+\operatorname{SPD}(4)\) spacetime Gaussians and the exact
  conditional linear-tube chart;
- camera program / ray bundle / depth gauge separation;
- ray-depth pullback and UVT pushforward;
- World Tubes continuous screen-time trace/event compilation;
- WorldFoam optical-transfer product integral;
- visibility monoid \((\beta,m)\);
- commutator/swap theorem;
- cell/event words and same-representation replay;
- prefix/suffix VJP;
- fixed-topology event-complexity caveats.

Key files:

- `research_notes/spacetime_gaussian_representation/`
- `research_notes/gauged_uvt_trace_atlas/`
- `research_notes/world_foam_reformulation.md`
- `research_notes/worldfoam_paper/WORLD_FOAM_MATH_APPENDIX.md`
- `research_notes/worldfoam_paper/WORLD_FOAM_PAPER_DRAFT.md`
- `research_notes/worldfoam_paper/proofs/depth_fiber_operator_ordering.md`
- `research_notes/renderer_lane_taxonomy.md`
- `BASELINES.md`

## Current model after the audit

### Paper A: World Tubes

The first follow-up's “push the atom into screen-time, compile support/events,
reuse backward” is World Tubes, not a new paper. Its valuable additions are:

- shared parameters are not shared geometric work;
- explicit output is \(\Omega(PT)\);
- structural scaling must be stated in trace/event records;
- exact Gaussian UVT closure requires an affine observation chart;
- shared backward reuses coefficient/tape construction but still evaluates
  output samples.

### Paper B: WorldFoam

The second dump's:

- vertical transfer connection;
- path-ordered exponential;
- gauge invariance;
- alpha/color commutator;
- prefix/suffix/Duhamel backward;
- event compiler;

mostly restate WorldFoam's existing method. They improve exposition and supply
some useful continuous formulas but do not justify a new renderer paper.

### Extension B1: finite-element WorldFoam

The strongest immediate representation direction is a native 4D cell complex
with compact P1/P2 extinction fields and exact segment transfer.

The decisive red-team result is:

> Positive direct-density Bernstein P1/P2 is the mandatory baseline and may be
> better than log/Gaussian FEM.

On a 4-simplex, both P2 branches use 15 scalar coefficients. The direct
positive polynomial has an elementary polynomial ray integral, while log-P2
needs exp/erf and possibly erfi. Log-FEM earns its cost only if relative dynamic
range improves quality or reduces cell count.

### Candidate Paper C: self-normalized convex-potential atom

The distinct object is:

\[
\sigma(x,t)
=
\alpha(t)
\big(
1-q(x,t)+\min_yq(y,t)
\big)_+^p,
\qquad
\nabla_x^2q\succeq\lambda I.
\]

It gives:

- unique derived ridge;
- bounded connected convex slices;
- derived motion/curvature/local orientation/scale;
- at most one interval per affine ray;
- polynomial segment integration after support roots;
- \(C^p\) integrated support birth under generic tangency.

It remains only a candidate because:

- it is an overlapping splat, not a foam partition;
- colored overlap needs the existing continuous transfer solve;
- \(r(t)\) and \(\mu(t)\) can be rational/algebraic even for polynomial \(q\);
- support is nonempty at every time unless \(\alpha(t)\) supplies lifetime;
- quadratic \(q\) is just a moving compact ellipsoid;
- higher-degree capacity raises root/minimizer cost;
- it is foliation-native, not fully symmetric in 4D;
- nearby convex/compact polynomial prior art is dense.

## Mathematical work preserved

### Log-quadratic segment integral

For:

\[
\ell(s)=as^2+bs+c,
\qquad
\tau=\sigma_\star\int_{s_-}^{s_+}e^{-\ell(s)}ds,
\]

the \(a>0\) result is:

\[
\tau
=
\sigma_\star e^{-c+b^2/(4a)}
\frac{\sqrt\pi}{2\sqrt a}
\left[
\operatorname{erf}
\left(
\sqrt a(s+b/(2a))
\right)
\right]_{s_-}^{s_+}.
\]

Linear, constant, and bounded negative-curvature/erfi branches were also
recorded. Stable evaluation needs series, `expm1`, scaled-tail, log-domain, and
negative-curvature policies.

### Moment VJP

\[
M_n=\int s^ne^{-(as^2+bs+c)}ds,
\]

\[
\partial_aM_0=-M_2,\qquad
\partial_bM_0=-M_1,\qquad
\partial_cM_0=-M_0.
\]

For \(a\ne0\):

\[
M_{n+1}
=
\frac{
nM_{n-1}-bM_n-[s^ne^{-(as^2+bs+c)}]_{s_-}^{s_+}
}{2a}.
\]

### Moving endpoints

\[
d\tau
=
\int\partial_\theta\sigma\,ds
+\sigma(s_+)ds_+
-\sigma(s_-)ds_-.
\]

These terms are generally nonzero at finite-element cell faces. The convex
atom's support-root terms vanish only because its density is zero there.

### FEM approximation bound

If:

\[
\|\ell-\ell_h\|_\infty\le\epsilon,
\]

then:

\[
e^{-\epsilon}\le\sigma_h/\sigma\le e^\epsilon,
\qquad
e^{-\epsilon}\tau\le\tau_h\le e^\epsilon\tau.
\]

This requires strictly positive target density; exact vacuum is singular in
log coordinates.

### Convex-atom ridge derivatives

For \(H=q_{xx}(r(t),t)\):

\[
\dot r=-H^{-1}q_{xt},
\]

\[
\ddot r
=
-H^{-1}
\left(
q_{xtt}+2q_{xxt}\dot r+q_{xxx}[\dot r,\dot r]
\right).
\]

\[
\dot\mu=q_t(r,t),
\qquad
\ddot\mu=q_{tt}-q_{tx}H^{-1}q_{xt}.
\]

### Correct extended-profile no-go example

The thin-layer commutator does not alone prove that
\((\alpha,c,\widehat z)\) fails, because exact thin-layer depths can be sorted.

For full optical depth \(\tau\), set \(r=e^{-\tau/2}\). Put red half-layers
around one full blue layer, then swap red and blue roles. Both primitive
summaries retain the same total alpha, color, and mean depth, but the images
differ by:

\[
(1-r)^2(1-r^2)(R-B)\ne0.
\]

This is the correct representative-depth counterexample.

## Backtracks and corrections

### Backtrack 1: “ray holonomy”

**Status:** weakened/replaced.

Open-ray ordered transport is not ordinarily holonomy. Holonomy is a
closed-loop object and already names a cell-complex diagnostic in WorldFoam.
Use:

- ordered ray transfer;
- ray-fiber product integral;
- open-ray parallel transport.

### Backtrack 2: gauge “invariance”

**Status:** narrowed.

Depth-coordinate reparameterization keeps the one-form \(A(s)ds\) and physical
transfer invariant. A true state-basis gauge transform is:

\[
A'=G^{-1}AG-G^{-1}dG,
\]

and open transport is endpoint-covariant:

\[
U'=G(s_1)^{-1}UG(s_0).
\]

### Backtrack 3: “full 4D Gaussian becomes exact screen-time Gaussian”

**Status:** hypothesis-limited.

Exact Gaussian closure holds for an affine observation map or exact Gaussian
fiber marginal. A general moving perspective camera produces a
projective/rational object and needs an exact ray construction, segmented fit,
or fallback certificate.

### Backtrack 4: “polynomial potential gives polynomial trace”

**Status:** false in general.

Self-normalization introduces:

\[
\mu(t)=\min_xq(x,t),
\]

which can be rational or algebraic. The ray polynomial may be simple at fixed
time while the compiled sensor-time coefficients are not.

### Backtrack 5: “Gaussian FEM is the obvious rich cell basis”

**Status:** unresolved and actively challenged.

Positive Bernstein FEM is cheaper, exactly integrable, and equally compact at
P2 on a simplex. The Gaussian/log exponent is now a challenger, not the
default.

## Existing quantitative motivation

Full-300-frame `coffee_martini`:

| Method | Parameters | Peak MPS | Train wall | Heldout PSNR |
|---|---:|---:|---:|---:|
| World Tubes | 14,336 | 3.114 GB | 78.33 s | 5.9153 |
| WorldFoam | 28,569,600 | 15.794 GB | 361.82 s | 5.6159 |
| Dynamic 3DGS | 4,300,800 | 20.557 GB | 79.44 s | 4.9110 |

This motivates compact native 4D WorldFoam state. It does not prove GFEM
quality.

Focused Gate4 fused-MSE microgates show useful WorldFoam kernel scaling versus
matched STAR, but are not native optical-transfer or RGB-quality parity.

## External novelty anchors checked

Primary pages:

- Tetra-NeRF: <https://arxiv.org/abs/2304.09987>
- Radiance Meshes: <https://arxiv.org/abs/2512.04076>
- DiffTetVR: <https://arxiv.org/abs/2601.00114>
- Don't Splat Your Gaussians: <https://arxiv.org/abs/2405.15425>
- 3D Convex Splatting: <https://arxiv.org/abs/2411.14974>
- From ex(p) to poly: <https://arxiv.org/abs/2603.18707>
- Deformable Beta Splatting: <https://arxiv.org/abs/2501.18630>
- Splat the Net: <https://arxiv.org/abs/2510.08491>
- PhysConvex: <https://arxiv.org/abs/2602.18886>

This was a targeted novelty-risk probe, not a complete systematic literature
review.

## Branches

### Branch A: positive polynomial FEM wins

Hypothesis:
    Cells already supply compact support, so positive P1/P2 extinction gives
    enough capacity with much cheaper integration.

What would make it false:
    It needs substantially more cells or cannot model extinction dynamic range.

Cheap test:
    Same cell tape, bytes, color, camera rays, and optimizer; compare M2/M3 with
    M4/M5.

If supported:
    Rename the method Finite-Element WorldFoam; drop Gaussian from the
    headline.

### Branch B: log-FEM wins

Hypothesis:
    Multiplicative/relative density approximation gives higher quality per
    cell, enough to offset exp/erf cost.

What would make it false:
    Same quality/cell and worse Metal throughput or stability.

Cheap test:
    Fixed-tape float64 then Metal P2 comparison.

If supported:
    Promote log-P2 as the material law, with convexity/stability constraints.

### Branch C: appearance is the actual current WorldFoam bottleneck

Hypothesis:
    P0 density is adequate; constant appearance causes the quality gap.

What would make it false:
    Affine appearance M1 fails while richer extinction wins.

Cheap test:
    M1 P0 density + affine RGB before changing geometry.

If supported:
    Do not over-engineer density.

### Branch D: convex atom wins

Hypothesis:
    Nonquadratic connected convex shapes use far fewer primitives than cell or
    Gaussian bases and compile with low event complexity.

What would make it false:
    Ridge/root/overlap cost or simple mixtures erase the advantage.

Cheap test:
    Quadratic X1 against one/two/four compact polynomial/SPD atoms at matched
    bytes, before quartic X2.

If supported:
    Open a separate primitive-paper lane only after multi-scene validation.

### Branch E: temporal event complexity kills every compiled variant

Hypothesis:
    Real camera paths and visibility make trace records effectively
    \(\Theta(NT)\).

What would make it false:
    Stable measured record/event/tape ratios fall with \(T\) across real
    scenes.

Cheap test:
    Death curves for \(T=2,4,8,16,32,64,128,300\).

If supported:
    Preserve the representation research but stop claiming temporal structural
    sublinearity.

## Decisions

1. Do not open a new screen-time compiler paper.
2. Do not open a new ordered-transfer/“ray holonomy” paper.
3. Add the external clarifications to the existing paper theory.
4. Treat finite-element WorldFoam as the immediate new method extension.
5. Require positive polynomial/Bernstein FEM as the primary counterbaseline.
6. Treat the self-normalized convex atom as a separate incubating primitive.
7. Build one shared segment-to-\((\beta,m)\) Metal ABI and a controlled
   material matrix rather than independent renderer forks.
8. Keep constant color initially; isolate appearance in M1.
9. Require CPU math/VJP parity before Metal and same-representation compiler
   parity before training claims.
10. Use trace-record/event death curves and explicit output lower bounds in all
    scaling claims.

## Durable files produced

- `research_notes/spacetime_gaussian_representation/09_screen_time_compiler_followup_audit.md`
- `research_notes/worldfoam_paper/GAUSSIAN_FINITE_ELEMENT_WORLD_FOAM.md`
- `research_notes/worldfoam_paper/SELF_NORMALIZED_CONVEX_ATOM_AND_RAY_TRANSFER_AUDIT.md`
- `research_notes/worldfoam_paper/PAPER_METHOD_CLASSIFICATION_AND_METAL_GATES.md`
- updated navigation in the two folder READMEs and `research_notes/README.md`

## Next actions

1. Implement the float64 fixed-cell material fixture M0–M5.
2. Verify ray integrals, coefficient VJPs, endpoint VJPs, and gauge Jacobians.
3. Implement the same fixed-tape material matrix in one Metal integration
   surface.
4. Run same-tape, matched-byte, and quality-matched regimes.
5. Select positive polynomial versus log-FEM.
6. Integrate only the winner into the native WorldFoam compiler.
7. Re-run full-300-frame and event/memory death curves.
8. Start X1 only after the finite-element result identifies a residual.

## Independent verification still required

- full equation-by-equation literature novelty review;
- symbolic/numeric verification of all special-function stability branches;
- finite-difference audit at active-face swaps and tangencies;
- exact current Metal ABI feasibility;
- measured optimizer conditioning of Bernstein versus log coefficients;
- full multiple-scene training;
- external reviewer check before claiming a new method/paper.
