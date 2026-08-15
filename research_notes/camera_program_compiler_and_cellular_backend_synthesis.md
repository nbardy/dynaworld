# Camera-Program Compiler And Cellular Backend Synthesis

Date: 2026-07-26

Status: paper-positioning synthesis and bounded research recommendation; no new
renderer or training result

## Source And Scope

This note audits the ChatGPT Pro export at:

```text
/Users/nicholasbardy/.codex/attachments/
59da4296-f678-4d80-a5cb-8ec2526c3360/pasted-text.txt
```

The export argues that STAR should be understood as a sensor-time
forward/adjoint compiler, that its Gaussian tube is a genuine but non-novel 4D
field, and that a 4D simplicial/corner representation is the strongest
high-risk cellular lane. It also supplies a long standalone research prompt.

The audit compares those claims with:

- `research_notes/renderer_lane_taxonomy.md`;
- `research_notes/spacetime_gaussian_representation/README.md`;
- `research_notes/spacetime_gaussian_representation/08_native_motion_bundles_and_shared_raster.md`;
- `research_notes/gauged_uvt_trace_atlas/DEPTH_FIBER_CROSS_TRACK_NOTE.md`;
- `research_notes/gauged_uvt_trace_atlas/paper/WORLD_TUBES_PAPER_DRAFT.md`;
- `research_notes/worldfoam_paper/WORLD_FOAM_PAPER_DRAFT.md`;
- `agent_notes/loose_notes/2026-07-26_17-37-10_fiberwise_log_quadratic_and_gaussian_fem_worldfoam_audit.md`;
- `BASELINES.md` and the current theorem/public-paper artifacts.

No new benchmark was run. Repository measurements below are inherited evidence,
not new observations.

## Executive Verdict

The export's central separation is right, but its preferred naming is one
level off:

```text
canonical umbrella:
    camera-program forward/adjoint compilation

primary paper method:
    World Tubes in Gauged Camera Space

current implementation:
    projective STAR UVT interval atlas

retained-depth sibling method:
    WorldFoam / cellular optical-transfer compilation
```

The useful abstraction is not literally "`STAR` with interchangeable Gaussian
and cell backends." World Tubes and WorldFoam share the camera-ray bundle,
event-stratification objective, and compiled-adjoint architecture, but they
lower different operator orderings into different intermediate
representations. World Tubes integrates/summarizes ray depth early. WorldFoam
retains ray depth and composes optical-transfer elements. They can share an
interface and paper vocabulary without pretending that one existing compiler
accepts both unchanged.

The export correctly proves that a full linear Gaussian tube is exactly a
strict SPD(4) Gaussian in conditional coordinates. That result is already
canonical in the repository. The current STAR implementation remains a
structured subfamily because its spatial conditional precision is restricted.

The strongest genuinely useful additions are:

1. a crisp four-layer decomposition of world, compiler, evaluator, and
   adjoint;
2. the recommendation to make **same-world, different-renderer** the decisive
   systems ablation;
3. a camera-pushforward complexity concept that measures chart/event/rank and
   traversal complexity rather than primitive count alone;
4. a direct nonnegative P1 corner-extinction cellular backend, which is a
   distinct and possibly cleaner first experiment than the previously proposed
   log-extinction FEM.

The current paper should remain the World Tubes paper. It should not be renamed
around a universal STAR framework until a second backend uses a real shared
compiler interface and adjoint. Cellular spacetime remains a bounded parallel
research lane and possible second method/paper, not a replacement for the
current submission spine.

## 1. The Correct Four-Layer Decomposition

The export's two-axis grid is directionally right but collapses two distinct
operations. Use four layers:

| Layer | Question | World Tubes answer | Cellular/WorldFoam answer |
| --- | --- | --- | --- |
| World | What exists in spacetime? | overlapping Gaussian atoms/tubes | a partitioned 4D cell complex with local fields |
| Compiler/lowering | What reusable camera-program object is built? | UVT/projective traces, support intervals, conditional depth, order strata | ordered ray-cell intervals and local optical-transfer words |
| Evaluator | What remains per requested sensor-time sample? | footprint evaluation, compositing, output write | transfer-word evaluation/scan, output write |
| Adjoint | How are residuals returned to world parameters? | reduce into trace coefficients, then direct per-trace/world VJP | prefix/suffix transfer VJP, local field/endpoint VJP, then world/cell VJP |

The shared abstract factorization is:

\[
W_\theta
\xrightarrow{\mathcal C_\Gamma}
A_{\theta,\Gamma}
\xrightarrow{S_T}
I_T
\xrightarrow{}
L,
\]

\[
\nabla_\theta L
=
J_{\mathcal C,\theta}^{\mathsf T}
J_{S,A}^{\mathsf T}
\nabla_I L.
\]

This factorization does not remove output-linear work. Materializing \(P\)
pixels at \(T\) requested samples is still \(\Omega(PT)\). The target is to
avoid repeating expensive world projection, support construction, binning,
visibility-event discovery, and world-gradient accumulation \(T\) times.

### Important correction to the export

The phrase:

```text
STAR is the compiler architecture;
Gaussian tubes and spacetime cells are alternative world backends.
```

is too strong as a current implementation claim. A safer statement is:

> The camera-program compiler/adjoint architecture is the umbrella. World
> Tubes/STAR and cellular WorldFoam are sibling lowerings with potentially
> shared interfaces, certification machinery, and evaluation protocols.

This preserves the repository's established operator-order distinction:
visibility generally does not commute with early depth marginalization.

## 2. SPD(4) And Linear Gaussian Tubes

Let \(\tau=t-t_0\), \(x_0\in\mathbb R^3\),
\(C\in\operatorname{SPD}(3)\), \(v\in\mathbb R^3\), and \(s_t>0\). The field

\[
g(x,t)=
\alpha\exp\left[
-\frac12\left(
(x-x_0-v\tau)^{\mathsf T}C^{-1}(x-x_0-v\tau)
+\frac{\tau^2}{s_t^2}
\right)\right]
\]

is one global 4D density, not a bank of frame states. Its precision is:

\[
Q=
\begin{bmatrix}
C^{-1} & -C^{-1}v\\
-v^{\mathsf T}C^{-1} & s_t^{-2}+v^{\mathsf T}C^{-1}v
\end{bmatrix}.
\]

Conversely, for:

\[
\Sigma=
\begin{bmatrix}
\Sigma_{xx} & \Sigma_{xt}\\
\Sigma_{tx} & \Sigma_{tt}
\end{bmatrix}
\in\operatorname{SPD}(4),
\]

the exact conditional chart is:

\[
s_t^2=\Sigma_{tt},\qquad
v=\Sigma_{xt}\Sigma_{tt}^{-1},\qquad
C=\Sigma_{xx}
-\Sigma_{xt}\Sigma_{tt}^{-1}\Sigma_{tx}.
\]

Therefore a full conditional Gaussian tube and a strict SPD(4) Gaussian are
the same function family under a bijective reparameterization.

Consequences:

- STAR has a real spacetime object.
- That fact is not a new Gaussian family or a defensible headline novelty.
- Storing velocity does not make the family less "4D"; velocity is the
  space-time cross covariance in physical coordinates.
- One strict SPD(4) atom has affine conditional motion and constant
  conditional spatial covariance. Curvature, rotating covariance, and changing
  scale require mixtures/piecewise atoms or a richer swept field.
- The current STAR source is narrower than full SPD(4) because it does not
  expose a general six-parameter conditional spatial covariance.

This fully agrees with the July 23 representation audit and the July 23 native
motion derivation. It should be cited as recovered standard mathematics, not
new theory.

## 3. What The Export Gets Right, Repeats, Or Gets Wrong

| Export claim | Audit status | Reason |
| --- | --- | --- |
| World ontology and observation algorithm are separate choices. | Supported, with refinement. | The repository already separates representation from camera-ray lowering, but World Tubes and WorldFoam also differ in operator ordering. |
| STAR's tube is a real 4D object but not a new Gaussian family. | Supported and already canonical. | Exact SPD(4)-conditional-tube equivalence is already proved in the representation notes. |
| The headline should be camera-program compilation and shared adjoint, not "we invented 4D Gaussians." | Supported and already the current paper spine. | The manuscript already claims camera-ray pushforward, event-certified domains, projective interval atlas, and direct backward. |
| Changing \(K(t)\) or \(w2c(t)\) within a sequence is not yet compiled. | Stale/invalid as a statement of current state. | The current projective/orbit route compiles moving camera programs, low-dimensional camera families, finite exposure, and rolling shutter on bounded certified domains. It does not claim arbitrary global camera paths or a universal 360/720-degree chart. |
| Learned tubes can make nearly every tile overflow/fallback. | Historically supported but overbroad. | This describes older restricted/source-view pathologies. Current accepted projective artifacts include fallback-free bounded cases and broad replacement gates; visibility chaos remains a limitation, not a universal current state. |
| Same learned world, per-time replay versus compiled atlas is the fairest baseline. | Strongly supported. | The manuscript already calls in-representation replay its cleanest evaluation, and the theorem table reports fixed/replay ratios through \(F=128\). The next improvement is to make the learned-world identity and parameter/checkpoint reuse unmistakable in the public table. |
| Cellular spacetime should replace STAR. | Rejected. | It has no matched quality/speed result and changes visibility semantics. It remains a bounded sibling lane. |
| Cellular spacetime is useless because line-stabbing depth can be large. | Rejected. | That is a real failure mode and kill criterion, not proof against the representation. |
| Heavy work can be independent of requested temporal density. | Proposal/conditional claim. | Only inside a fixed bounded interval with a compact certified event/trace representation. Output writes and residual shading remain sample-linear. |

### Metric provenance warning

The export quotes older fixed-camera microbenchmarks such as
`0.024 -> 0.033 s` at \(128^2\), a `0.067 s` STAR row at \(256^2,F=32\),
and a `0.099 s` backward phase. These numbers were not located as current
canonical rows in `BASELINES.md` during this audit. Do not copy them into the
paper without linking them to their exact saved artifacts and protocol.

The current manuscript has stronger and better-routed evidence:

```text
bounded-orbit F = 4..128:
    fixed/replay payload ratio at F=128   = 0.03125
    fixed/replay forward ratio at F=128   = 0.181323
    fixed/replay backward ratio at F=128  = 0.392235

camera-family Q2:
    shared/replay payload ratio           = 0.0625
    shared/replay chart ratio             = 0.015625
    max UV fit residual                   = 0.111 px
```

The public progressive Coffee Martini table is newer still, but its low
absolute quality, metric disagreement, unmatched storage/parameter counts, and
missing fixed/sampler/scene-breadth controls mean it cannot replace the
same-representation theorem row.

## 4. Comparison With The Current World Tubes Paper

### Already present in the manuscript

The current draft already contains the export's strongest conceptual points:

- known or low-dimensional camera programs expose reusable world-side work;
- the compiled object is a sensor-time trace atlas, not a list of frames;
- the camera-ray bundle transform is
  \(\operatorname{Trace}_\Gamma[w]=\pi_*\Gamma^*w\);
- local depth elimination is a Schur complement;
- visibility is preserved through conditional depth/order certificates and
  event-certified gauge domains;
- projective interval forward and direct VJP are implemented;
- per-frame replay of the same representation is the central comparator;
- the main claim is sublinear **world-side** scaling, not sublinear output
  materialization;
- WorldFoam is explicitly a separate retained-depth representation/paper lane.

The imported thread therefore validates the paper spine; it does not discover
it from scratch.

### What should be sharpened

1. Add the four-layer table above near the end of the introduction or at the
   start of method. It prevents reviewers from conflating world representation
   novelty with renderer/compiler novelty.
2. State the SPD(4)-tube equivalence in one compact proposition and immediately
   disclaim Gaussian-family novelty.
3. Lead the experimental narrative with the same-world renderer ablation:
   identical world parameters/checkpoint, per-time replay versus compiled
   atlas, identical requested samples and loss.
4. Name the shared adjoint as a contribution only at the operational level:
   sparse compiled incidence, atlas-space residual reduction, direct
   world-gradient accumulation, and measured work/memory. The chain rule itself
   is not new.
5. Add camera-pushforward diagnostics—events, chart count, trace rank or
   coefficient count, line/layer depth, fallback area, and recompile
   frequency—to the camera-motion stress table.
6. Keep WorldFoam and the simplicial backend out of the main method. Mention
   them as the noncommuting retained-depth alternative and future backend
   experiment.

### What should not change

- Keep the public method name **World Tubes in Gauged Camera Space**.
- Keep STAR UVT/projective interval as the implementation name.
- Do not retitle the current paper as a universal STAR framework before a
  second backend exists behind a shared interface.
- Do not claim arbitrary camera trajectories, universal finite atlases, exact
  visibility-boundary differentiation, or frame-independent total rendering.
- Do not mix the renderer paper's claim with the broader world-token
  predictive-quotient contract. The latter concerns exportable,
  camera-portable learned worlds; the former concerns efficient observation of
  a world under a known camera program.

## 5. Camera-Pushforward Complexity

Primitive count and byte count do not predict compiler difficulty. A useful
provisional diagnostic is:

\[
\kappa_\epsilon(W,\Gamma)
=
E_\epsilon
+A_\epsilon
+\sum_{\ell\in\mathcal L_\epsilon}
r_{\ell,\epsilon}d_\ell,
\]

where:

- \(E_\epsilon\) counts certified support, denominator, tangency, order,
  visibility, and topology events over the camera program;
- \(A_\epsilon\) counts active compiled references/interval entries;
- \(r_{\ell,\epsilon}\) is local trace or transfer approximation rank in chart
  \(\ell\);
- \(d_\ell\) is compositing/traversal or line-stabbing depth.

This is a diagnostic, not yet an invariant or theorem. It depends on:

- the tolerance and norm used for certification;
- the camera and sensor-time domain;
- the chosen chart/basis and splitting policy;
- visibility/compositing semantics;
- whether compile bytes, work, or runtime is the target.

A practical logged vector is safer than prematurely collapsing to one scalar:

```text
K_push = {
  atlas_bytes,
  active_references,
  support_events,
  denominator_events,
  order_or_face_events,
  chart_count,
  local_rank_mean_p95_max,
  traversal_depth_mean_p95_max,
  fallback_area,
  recompile_frequency
}
```

Only after correlation with compile, forward, backward, memory, and quality
should a differentiable regularizer be chosen. Otherwise the representation
may learn to game a proxy by increasing error, fallback, or hidden evaluator
work.

## 6. Cellular Backend: Direct P1 Extinction Versus Log-FEM

The imported simplicial construction adds a useful option not cleanly
separated in the prior log-FEM note.

Let \(K\) be a 4D simplex with barycentric coordinates
\(\lambda_i(Z)\), \(i=0,\ldots,4\).

### Option A: direct nonnegative P1 extinction

\[
\sigma_K(Z)=\sum_{i=0}^{4}\lambda_i(Z)\sigma_i,
\qquad \sigma_i\ge 0.
\]

Because barycentric coordinates are nonnegative inside a simplex and sum to
one, nonnegative nodal values guarantee \(\sigma_K\ge0\). Along an affine
ray-cell segment \(Z(s)\):

\[
\sigma(s)=a+bs.
\]

The transmittance from the segment entrance is:

\[
T(s)=\exp\left(-as-\frac12bs^2\right).
\]

For polynomial premultiplied emission
\(j(s)=\sum_{n=0}^{q}c_ns^n\), the segment contribution is:

\[
\Delta C
=
\sum_{n=0}^{q}c_n
\int_0^L s^n
\exp\left(-as-\frac12bs^2\right)\,ds.
\]

These are bounded quadratic-exponential moments: use erf/recurrences for
\(b>0\), erfi-scaled formulas for \(b<0\), and stable linear/constant limits
for \(b\to0\). Nonnegative endpoint extinction constrains the apparently
growing \(b<0\) branch over the finite segment, but it does not turn that
branch into an ordinary decaying Gaussian.

Advantages:

- exact convex support and five active corner bases in a 4-simplex;
- \(C^0\) extinction across conforming shared faces;
- exact affine ray clipping and affine extinction;
- true zeros are representable;
- clean moment-based constant/polynomial emission;
- simple nodal gradients away from topology events.

Risks:

- positivity requires constrained/activated nodal parameters;
- large extinction dynamic range can be awkward in direct coordinates;
- the quadratic coefficient along a segment can have either sign, requiring
  stable bounded-interval formulas rather than assuming a globally decaying
  Gaussian;
- 4D connectivity, line-stabbing depth, remeshing, and topology gradients
  remain hard.

### Option B: P1/P2 log-extinction

\[
\sigma_K(Z)=\exp[-\ell_K(Z)],
\qquad
\ell_K(Z)=\sum_a\theta_aN_a(Z).
\]

P1 makes \(\ell(s)\) affine and \(\sigma(s)\) exponential-linear. P2 makes
\(\ell(s)\) quadratic and \(\sigma(s)\) Gaussian-like over the bounded
segment.

Advantages:

- strict positivity and large dynamic range;
- unconstrained nodal log parameters;
- \(C^0\) log-density on a conforming mesh;
- P2 includes a flexible bounded log-quadratic field.

Risks:

- the transfer law differs from direct P1; do not reuse the
  exponential-of-quadratic emission claim without re-derivation;
- strict positivity cannot represent exact empty interior without a limit;
- varying-emission integrals may lose the simplest polynomial-moment form;
- nonlinear conditioning may be worse.

### Recommendation

The first cellular fixture should compare:

```text
P0 direct extinction
P1 direct nonnegative extinction
P1 log-extinction
P2 log-extinction (only after the first three pass)
```

Hold fixed:

- the same 4D simplex chain and ray-cell words;
- constant RGB first;
- camera path and requested times;
- total learned bytes where possible;
- endpoint and topology treatment.

Measure:

- analytic versus quadrature optical-depth/RGB error;
- coefficient and endpoint VJP error;
- forward/backward time;
- conditioning and gradient range;
- compiled tape bytes and event count;
- quality per learned byte.

This A/B is more informative than immediately committing to "Gaussian FEM."

## 7. Exactness And Gradient Boundary

Inside one fixed event stratum:

- convex simplex intersection is an exact interval;
- P1 direct extinction is affine along an affine depth ray;
- constant-color transfer is exact from total optical depth;
- polynomial-emission transfer under direct P1 extinction has analytic
  truncated-moment formulas;
- coefficient and active-endpoint derivatives are analytic.

For a moving segment:

\[
\Delta\tau(\theta)
=
\int_{a(\theta)}^{b(\theta)}
\sigma(s,\theta)\,ds,
\]

\[
\frac{d\Delta\tau}{d\theta}
=
\int_a^b\partial_\theta\sigma\,ds
+\sigma(b)b_\theta
-\sigma(a)a_\theta.
\]

The last two terms are boundary flux and must not be dropped. Face-winner
swaps, interval births/deaths, connectivity changes, and remeshing are
discrete event boundaries. The honest first contract is:

```text
exact fixed-topology forward/VJP
+ active-endpoint derivatives
+ certified refresh/fallback at topology events
```

The general sensor-time shape derivative:

\[
\frac{d}{d\theta}\int_B f_\theta(y)\,dy
=
\int_{B\setminus\Sigma_\theta}\partial_\theta f_\theta(y)\,dy
+\int_{\Sigma_\theta}[f_\theta](y)v_n(y)\,dS
\]

is a useful research target, but no current implementation proves complete
visibility-boundary differentiation. Ordinary pathwise AD inside fixed
strata captures the first term and generally misses the moving discontinuity
term.

## 8. Decisive Experiments

### Experiment A: same world, different renderer

Use one learned full conditional Gaussian/tube checkpoint.

```text
A1: condition/slice at every time and use ordinary per-frame Gaussian replay
A2: compile once into the World Tubes sensor-time atlas and sample it
```

Require identical:

- world parameters, appearance, lifetime, and learned bytes;
- cameras, times, loss, and requested pixels;
- optimizer state when comparing a training step;
- output semantics within stated tolerance.

Report compile, forward, backward, total step, peak memory, atlas bytes,
candidate references, fallback, gradient error, and quality.

This isolates the paper contribution better than comparing against a different
dynamic-GS architecture.

### Experiment B: same compiler interface, different world producer

Only after the common boundary is real:

```text
B1: current structured STAR tube
B2: full SPD(4) Gaussian
B3: low-knot swept Gaussian
```

Compare quality/byte against trace rank, event count, and compile/recompile
cost. This asks whether extra producer expressivity reduces total compiled
complexity enough to pay for harder traces.

### Experiment C: cellular transfer toy

Use two or a short chain of fixed 4D simplices, one moving camera, and one true
occlusion/order event. Implement direct P0/P1 and log-P1 extinction, constant
RGB, exact segment transfer, compiled events, and one corner-level adjoint.

Initial correctness gates:

```text
analytic vs dense quadrature relative error <= 1e-6
compiled vs direct RGB max error            <= 1e-6
coefficient VJP relative error              <= 1e-5
endpoint VJP relative error                 <= 1e-5
```

Promotion requires all three:

\[
\frac{\text{compiled cell events}}
{\text{summed independent-frame events}}\ll1,
\]

\[
\text{reverse memory}
=O(\text{cells}+\text{events}+\text{small reduction state}),
\]

and a quality-per-byte or quality-adjusted traversal advantage over the
constant-cell and World Tubes alternatives. Do not build a production 4D
mesher before this gate.

### Experiment D: camera-motion stress

Vary independently:

- frame density over a fixed physical interval;
- physical duration;
- translation and angular velocity;
- orbit angle;
- exposure and rolling-shutter slope;
- support/order/disocclusion event frequency.

This separation is essential. Increasing frame density should mainly stress
evaluation. Increasing duration or camera complexity can legitimately increase
charts and events.

## 9. Prior-Art Boundary

Primary-source checks support the following boundaries:

- [Native 4D Gaussian Splatting](https://arxiv.org/abs/2412.20720) already
  models scenes with anisotropic 4D Gaussian primitives.
- [Spacetime Gaussian Feature Splatting](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Spacetime_Gaussian_Feature_Splatting_for_Real-Time_Dynamic_View_Synthesis_CVPR_2024_paper.html)
  already combines temporal opacity with parametric motion/rotation and
  feature splatting.
- [Disentangled4DGS](https://arxiv.org/abs/2503.22159) explicitly attacks
  repeated 4D slicing/matrix work and projects temporal/spatial deformation
  into dynamic ray-space 2D Gaussians.
- [Radiant Foam](https://arxiv.org/abs/2502.01157) already establishes a
  differentiable Voronoi-foam representation with efficient neighbor
  traversal.
- [Power Foam](https://arxiv.org/abs/2604.24994) already introduces bounded
  power cells intended to unify differentiable ray tracing and rasterization.
- [Radiance Meshes](https://arxiv.org/abs/2512.04076) already uses Delaunay
  tetrahedral cells for exact fast volume rendering and explicitly confronts
  topology flips under learned vertex positions.
- [DiffTetVR](https://arxiv.org/abs/2601.00114) already provides
  differentiable tetrahedral volume rendering, vertex optimization, and local
  subdivision.
- [Simplex space-time meshes](https://arxiv.org/abs/2210.09831) are
  established in moving-domain numerical PDE work.
- [3DGUT](https://arxiv.org/abs/2412.12507) already supports nonlinear camera
  projection, rolling shutter, and secondary rays through an unscented
  projection formulation.

Therefore do not headline:

- a 4D Gaussian;
- velocity-tube coordinates;
- 4D simplices or pentatopes;
- bounded differentiable cells;
- exact tetrahedral segment rendering;
- rolling shutter or nonlinear cameras alone.

The defensible novelty target remains the combination:

> Compile a canonical dynamic world and bounded continuous camera program into
> an event-certified sensor-time forward representation and operational shared
> adjoint, so world-side work and reverse storage scale with compiled
> representation/event complexity rather than independent frame replay.

For cells, narrow this further to:

> exact retained-depth spacetime-cell transfer, sensor-time event sharing, and
> one compiled field/geometry adjoint.

These are plausible novelty targets, not completed literature-exhaustive
novelty proofs.

## 10. Relationship To The Broader DynaWorld Notes

The renderer/compiler result and the world-token training contract solve
different problems:

```text
predictive-quotient/world-token lane:
    Can observations be compressed into a self-sufficient,
    camera-portable world asset that predicts held-out queries?

World Tubes/compiler lane:
    Given a world and a known camera program, can repeated rendering
    and differentiation reuse world-side work across sensor time?
```

The first is about representation sufficiency, leakage, rate, and novel-view
generalization. The second is about observation-algorithm complexity and
correctness. A fast compiler does not make a camera-leaking world token valid.
A valid world token does not imply frame-amortized rendering.

The clean integration point is the exported world object:

```text
observations -> W0
W0 + query camera program -> compiled atlas
compiled atlas + requested samples -> images/loss
```

The renderer paper should not absorb the training-contract claims until a
world-token model actually exports a canonical world object consumed by the
compiler.

## 11. Decision And Next Actions

Current decision:

```text
Keep World Tubes as the primary paper.
Keep STAR UVT as its implementation family.
Keep camera-program forward/adjoint compilation as the umbrella abstraction.
Keep WorldFoam/cellular transfer as a sibling retained-depth lane.
Promote direct P1 corner extinction to the first cellular A/B.
Use same-world per-frame replay as the decisive paper baseline.
```

Highest-value paper action:

1. add the four-layer decomposition and the explicit no-new-Gaussian claim;
2. make identical-checkpoint replay versus compiled atlas the first systems
   ablation;
3. report the camera-pushforward diagnostic vector alongside runtime;
4. keep the public breadth and fixed/sampler controls as submission blockers.

Highest-value bounded research action:

1. fixed two-simplex or short-chain fixture;
2. P0 direct, P1 direct, and P1 log extinction;
3. constant RGB and exact segment transfer;
4. coefficient plus endpoint finite-difference VJP;
5. direct-per-time versus compiled-event scaling through \(T=256\);
6. stop if event sharing, reverse storage, or quality-per-byte does not improve.

## Open Questions

1. Can the current paper runner reuse literally identical learned Gaussian
   parameters between the ordinary per-frame renderer and the compiled atlas,
   or does an adapter still alter semantics?
2. Which parts of \(K_{\text{push}}\) best predict measured compile, forward,
   backward, and memory across camera speeds?
3. Is direct P1 extinction better conditioned than log-P1 at the density ranges
   needed by real scenes?
4. Can constant cell appearance isolate the geometry/extinction question, or
   will appearance capacity dominate even the toy?
5. Can active-face endpoint gradients be implemented while treating topology
   changes as certified refresh events?
6. Does a simplicial complex reduce total compiled event complexity relative
   to power-cell owner traversal at matched quality and bytes?
7. What camera-program domain is broad enough for a useful claim but narrow
   enough for honest finite certification?
