# Paper/Method Classification and Native Metal Gates

## Where the two external dumps belong, and what is still required

**Status:** publication and engineering decision note.

**Date:** 2026-07-26

## 1. Direct answer

The new material does **not** form one new paper.

It separates into four research objects:

| Object | Current classification | Paper destination |
|---|---|---|
| Continuous screen-time shared compiler | Existing method; useful clarification | World Tubes paper |
| Camera-gauged ordered ray transfer, product integral, commutator, VJP | Existing method core; useful refinement | WorldFoam paper |
| Gaussian/log finite-element spacetime cells | New WorldFoam representation/material extension | WorldFoam section first; follow-on only if it wins |
| Self-normalized strongly-convex spacetime atom | Genuinely distinct local primitive candidate | Incubating possible third method; not paper-ready |

So the short answer to “is the last dump a new paper?” is:

> **No as a whole.** Most of its renderer theory is already WorldFoam. Its
> convex-potential atom is a new candidate object, but it does not yet have the
> implementation, prior-art separation, or evidence required to call it a new
> paper.

The Gaussian finite-element WorldFoam construction is more mature
mathematically than the convex atom and attacks a measured failure of the
current implementation. It should be the first new engineering branch.

## 2. Existing Paper A: World Tubes

### Core contribution

\[
\text{world spacetime primitives}
\longrightarrow
\text{continuous sensor-time traces}
\longrightarrow
\text{compiled support/visibility/adjoint atlas}.
\]

Its meaningful novelty is not merely using 4D Gaussians. Native spacetime
Gaussian representations already exist, including [4D Gaussian Splatting with
native 4D primitives](https://arxiv.org/abs/2412.20720). The method claim is
the compilation of a known camera program into reusable projection, binning,
visibility, and backward records.

### What the first external follow-up adds

It adds good exposition:

- compact parameters do not imply compact raster work;
- explicit \(P\times T\) output has an \(\Omega(PT)\) lower bound;
- structural work should be stated in trace/event counts;
- a shared backward shares coefficient/tape construction, not every output
  interaction.

### What it does not add

It is not a new primitive, compiler family, or paper. Its exact Gaussian
screen-time closure also needs the affine-camera hypothesis that the dump
sometimes omits.

### Remaining Paper A work

The current lane already has the stronger local implementation/evidence.
Remaining work is principally:

- multiple public scenes and camera triplets;
- final matched runtime/cost tables;
- longer and more varied camera programs;
- event/fallback death curves;
- final figures/manuscript;
- optional non-Metal portability.

Do not delay Paper A for a new WorldFoam material family.

## 3. Existing Paper B: WorldFoam

### Core contribution

\[
\text{bounded spacetime matter}
\xrightarrow{\Gamma^*}
\text{ray-fiber extinction/emission}
\longrightarrow
\text{ordered optical-transfer event atlas}.
\]

The paper already contains:

- camera programs, ray fibers, and depth gauges;
- the physical-length Jacobian;
- visibility monoid \((\beta,m)\);
- product-integral/path-ordered transfer;
- cell/event words;
- same-representation replay theorem;
- commutator/swap criterion;
- prefix/suffix VJP;
- event-complexity caveats.

The second external dump improves terminology and adds a useful continuous
Duhamel derivative/support-discriminant proposal. It does not create another
ray-transfer paper.

### Current measured failure that motivates compact spacetime cells

The matched full-300-frame `coffee_martini` rows in `BASELINES.md` report:

| Method | Parameters | Peak sampled MPS driver | Mean train wall | Heldout PSNR |
|---|---:|---:|---:|---:|
| World Tubes | 14,336 | 3.114 GB | 78.33 s | 5.9153 |
| WorldFoam | 28,569,600 | 15.794 GB | 361.82 s | 5.6159 |
| Dynamic 3DGS | 4,300,800 | 20.557 GB | 79.44 s | 4.9110 |

The WorldFoam row stores:

\[
1024\ \text{cells}\times300\ \text{frames}\times93
=28{,}569{,}600
\]

trainable scalars. This is exactly the failure a native 4D finite-element field
could attack. It does not prove the new field will preserve quality or reduce
event cost.

### Current positive evidence

The focused Gate4/native-cutwalk fused-MSE microgate reported:

\[
\text{WorldFoam total at }2/4/8/16\text{ frames}
=3.008/3.014/3.323/4.095\text{ ms},
\]

versus matched STAR:

\[
5.003/5.943/8.092/9.794\text{ ms}.
\]

This proves a useful kernel/scale signal, not native optical-transfer parity or
real-RGB quality dominance.

## 4. New extension B1: finite-element WorldFoam

### Narrow method claim

The defensible candidate is:

> A compact native 4D spacetime cell complex with P1/P2 extinction
> coefficients, exact or certified camera-fiber integration, compiled event
> reuse over a camera program, and an analytic ordered-transfer backward.

“Finite-element radiance field,” “tetrahedral rendering,” and “exact constant
cell integration” are not sufficient novelty claims. Relevant prior art
includes:

- Tetra-NeRF: <https://arxiv.org/abs/2304.09987>
- Radiance Meshes: <https://arxiv.org/abs/2512.04076>
- DiffTetVR: <https://arxiv.org/abs/2601.00114>
- Radiant Foam: <https://arxiv.org/abs/2502.01157>
- compact volumetric kernels: <https://arxiv.org/abs/2405.15425>
- polynomial splat kernels: <https://arxiv.org/abs/2603.18707>

The possible distinction is the combination of a **4D dynamic cell field**,
camera-program event compilation, and a shared differentiable transfer tape.

### Is this its own paper?

Not yet. Initially it is the most useful representation section and ablation
inside WorldFoam. It justifies a follow-on method paper only if the compact 4D
state produces a decisive result such as:

- materially better heldout quality at matched bytes;
- materially lower bytes at matched quality;
- lower training memory/time at matched quality;
- a measured structural temporal-reuse win unavailable to static cell
  renderers.

If positive polynomial FEM beats log/Gaussian FEM, preserve the method as
**Finite-Element WorldFoam** and drop “Gaussian” from the headline.

## 5. Candidate Paper C: self-normalized convex-potential atoms

### Candidate object

\[
\sigma(x,t)
=
\alpha(t)
\left(
1-q(x,t)+\min_yq(y,t)
\right)_+^p,
\qquad
\nabla_x^2q\succeq\lambda I.
\]

This provides:

- a unique derived ridge;
- compact, convex, connected spatial slices;
- derived motion, curvature, local orientation, and scale;
- empty-or-one-interval intersection with an affine ray;
- polynomial segment integrals for low-degree polynomial \(q\), after roots.

### Why it is not yet a paper

1. It is an overlapping compact primitive, not automatically a foam owner
   partition.
2. Differently colored overlap generally needs integration of the summed
   transfer generator.
3. The per-time minimizer is not free. Even polynomial \(q\) can give rational
   or algebraic \(r(t)\) and \(\mu(t)\), requiring nonlinear solves or
   certified approximations.
4. Self-normalization makes the support nonempty at every time; finite lifetime
   must be supplied by the amplitude/support law for \(\alpha(t)\).
5. Quadratic \(q\) reduces to a moving compact ellipsoidal kernel. New shape
   capacity begins at higher degree, exactly where root/minimizer cost rises.
6. The construction is native to the physical time foliation, not fully
   symmetric in 4D: it minimizes over space separately at each \(t\).
7. Prior art is crowded:
   [3D Convex Splatting](https://arxiv.org/abs/2411.14974),
   [dynamic convex fields](https://arxiv.org/abs/2602.18886),
   [deformable beta kernels](https://arxiv.org/abs/2501.18630),
   [polynomial kernels](https://arxiv.org/abs/2603.18707), and
   [bounded neural primitives with analytic line
   integrals](https://arxiv.org/abs/2510.08491).

### Required comparison

Before a paper claim, it must beat:

- one quadratic compact atom;
- two and four simpler quadratic/SPD4/polynomial atoms;
- positive P2 WorldFoam;
- log-P2 WorldFoam;
- current World Tubes and Dynamic 3DGS;

at matched parameter and serialized-byte budgets.

## 6. Yes, more Metal is needed—but one controlled matrix

The immediate need is not several independent renderer forks. Add specialized
segment integrators behind one owner-run tape and one optical-element ABI:

\[
\boxed{
\text{segment record}
\longmapsto
g=(\beta,m),
\quad
\beta=e^{-\Delta\tau},
\quad
m=(1-\beta)c.
}
\]

Keep geometry, camera path, cell word, endpoints, appearance, and scan
identical while varying only the material law.

### Required material matrix

| ID | Extinction inside one cell | Appearance | Purpose |
|---|---|---|---|
| M0 | P0 constant | constant RGB | Current keeper/reference |
| M1 | P0 constant | affine RGB | Tests whether appearance, not density, is the gap |
| M2 | positive P1 direct density | constant RGB | Cheapest richer density |
| M3 | positive Bernstein P2 density | constant RGB | Main polynomial counterbaseline |
| M4 | log-P1 density | constant RGB | Automatic positivity; exp-linear integral |
| M5 | convex log-P2 density | constant RGB | Gaussian-like erf branch |
| X1 | quadratic convex-potential atom | constant RGB | Separate CPU then Metal candidate |
| X2 | quartic convex-potential atom | constant RGB | Only after X1 identifies a residual |

Important constraints:

- use Bernstein/Bézier coefficients when positivity is required; nonnegative
  P2 Lagrange nodal values do not guarantee positivity between nodes;
- use physical ray length or carry the gauge Jacobian;
- keep M5 convex initially to avoid unstable `erfi`;
- keep constant color until density transfer wins;
- reuse endpoint records, owner words, scan, and prefix/suffix VJP;
- do not mix atom overlap and foam ownership in one benchmark.

## 7. Three comparison regimes

Every material variant needs all three:

### Regime A: same geometry and tape

Identical cells, ray words, endpoints, frame samples, and colors. This isolates
the cost and expressivity of the material integrator.

### Regime B: matched trainable/checkpoint bytes

Allow cell count to vary so every method has the same:

- trainable parameter bytes;
- optimizer-state bytes;
- serialized checkpoint bytes.

This tests representation efficiency.

### Regime C: quality-matched capacity

Allow cell count to vary until heldout quality is within a declared tolerance.
Then compare:

- forward/backward time;
- compile time;
- memory;
- cell runs and events;
- tape bytes.

This tests system efficiency at equal output quality.

## 8. Engineering ladder

### Gate 0: float64 reference

Required before Metal:

- exact/analytic optical depth versus high-accuracy quadrature;
- coefficient VJP versus central finite differences;
- endpoint VJP versus finite differences under fixed active faces;
- constant, near-linear, near-zero-curvature, grazing, and high-optical-depth
  cases;
- depth-coordinate reparameterization with the physical Jacobian;
- explicit failure/fallback at topology changes.

### Gate 1: fixed-tape Metal material parity

Feed precomputed owner words and endpoints. Suggested starting acceptance:

\[
\max|I_{\mathrm{Metal}}-I_{\mathrm{ref}}|\le10^{-5},
\]

\[
\frac{\|\nabla_{\mathrm{Metal}}-\nabla_{\mathrm{ref}}\|}
{\max(1,\|\nabla_{\mathrm{ref}}\|)}
\le10^{-4}.
\]

Use absolute tolerances near zero and publish the actual tolerance sweep. Also
require:

- zero NaN/Inf;
- branch counters for series/direct-erf/scaled-erfcx-tail/reject;
- M0 timing does not materially regress;
- synchronized GPU timing, not dispatch-only timing.

### Gate 2: compiled same-representation parity

Direct per-time traversal and compiled traversal must agree on:

- owner/event words;
- active entry/exit faces;
- endpoints/physical lengths;
- RGB/transmittance;
- coefficient and endpoint gradients.

Test:

\[
T=2,4,8,16,32,64,128,300
\]

on static, linear, accelerating, orbiting, visibility-crossing,
rolling-shutter, and finite-exposure camera programs.

### Gate 3: material-value gate

The richer density must beat both M0 and M1. A useful preregistered threshold
for deciding whether to continue is either:

\[
\ge0.5\text{ dB heldout PSNR at matched serialized bytes},
\]

or:

\[
\ge2\times\text{ lower representation+optimizer bytes within }0.2\text{ dB},
\]

repeated beyond one scene and reported with SSIM, LPIPS, and L1. These are
engineering continuation thresholds, not universal scientific significance
tests.

### Gate 4: systems/event gate

Record:

- compile, forward, loss, backward, and optimizer time;
- current/peak driver and framework memory;
- checkpoint and optimizer bytes;
- target and rasterized pixels;
- owner runs per ray;
- tile incidences;
- support/order/active-face events;
- fallback fraction;
- atlas refresh rate;
- tape bytes per output pixel;
- amortization point versus replay.

Plot the death curves rather than one favorable frame count.

### Gate 5: paper breadth

At minimum:

- three public dynamic scenes;
- multiple camera triplets/trajectories;
- three seeds;
- full requested temporal extent;
- heldout PSNR, SSIM, LPIPS, L1, and temporal stability;
- matched target/raster budget;
- parameter-matched and byte-matched tables;
- visual failure cases;
- official or clearly scoped contextual baselines.

## 9. Additional gate for the convex-potential atom

Before native shader promotion, demonstrate that ridge and support compilation
do not silently restore a per-atom-per-frame nonlinear solve:

\[
C_{\min}+C_{\mathrm{roots}}+E_{\mathrm{support}}
=o(NT)
\]

on the intended camera/time programs, or state a weaker measured claim.

Also require:

- minimizer continuation/certification error;
- root bracketing and tangency fallback rates;
- summed-medium overlap transfer parity;
- comparison with two/four simpler atoms;
- finite-lifetime amplitude/support behavior;
- spatial-Hessian conditioning and eigenvalue collision diagnostics.

## 10. Literature/novelty gate

Before drafting contribution claims, compare the actual equations and kernels,
not abstracts alone, against:

- native/spacetime 4D Gaussian splatting;
- 3D Convex Splatting;
- compact beta/Epanechnikov/polynomial splats;
- analytic ray-traced volumetric primitives;
- tetrahedral/radiance-mesh renderers;
- differentiable tetrahedral volume rendering;
- dynamic convex radiance fields;
- spacetime finite-element and space-time mesh methods.

For each proposed contribution, maintain:

| Proposed claim | Closest prior equation/system | Exact difference | Evidence |
|---|---|---|---|
| Native 4D FEM extinction | TBD | camera-compiled dynamic event reuse? | proof + benchmark needed |
| Exact log-P2 segment VJP | TBD | shared spacetime/event tape? | reference + Metal needed |
| Self-normalized convex atom | TBD | spatial-minimum normalization and derived ridge? | literature + experiments needed |
| Shared ordered backward | Existing WorldFoam/local + standard transfer | compiler/tape implementation | parity + scaling needed |

No row is a contribution until the “exact difference” survives review.

## 11. What is enough for each publication level?

### Enough to mention GFEM in the current WorldFoam paper

- the complete mathematical definition;
- float64 reference and finite-difference VJP;
- fixed-tape native Metal forward/backward;
- P0/P1/P2/log side-by-side ablation;
- one honest matched real-scene result;
- explicit “prototype/material extension” claim boundary.

### Enough to headline GFEM as a new method paper

- native spacetime compiler and geometry VJP;
- clear novelty against cell/tetrahedral renderers;
- multi-scene breadth;
- decisive quality/bytes or quality/speed win;
- measured event-scaling advantage;
- stable full training and public reproducibility.

### Enough to headline the convex-potential atom

All GFEM-level systems evidence, plus:

- cheap/certified minimizer and root compiler;
- overlap transfer solution;
- matched simpler-atom defeat;
- novelty separation from convex and compact polynomial splatting;
- an advantage attributable specifically to self-normalization/nonquadratic
  shape, not merely more parameters.

## 12. Recommended sequence

1. Finish/publish the existing World Tubes breadth work independently.
2. [x] Build the CPU reference for M0–M5 on one fixed cell tape.
3. [x] Build one parameterized Metal material-integrator kernel/dispatch, not
   separate renderer forks.
4. [x] Test positive polynomial and log-Gaussian FEM at equal payload. The
   result is complementary family-specific wins, not a universal winner.
5. Test adaptive per-cell M3/M5 selection or real held-out material
   observations before integrating a rich material law. Keep P0 as the systems
   oracle meanwhile.
6. Re-run the 300-frame matched protocol and event/memory death curves.
7. Only then introduce quadratic X1.
8. Open a separate convex-atom paper lane only after X1 beats simpler mixtures.

## 13. Final decision

Yes, there is substantial engineering work. The controlled same-tape material
matrix is complete and found complementary M3/M5 wins. More shader forks are
not the next gate. The decisive systems work is native lowering of the landed
kinetic multi-chart compiler/VJP, structural recertification, and matched
full-training evidence; rich material should advance only through adaptive or
real-heldout selection.

The most promising immediate advance is compact finite-element WorldFoam,
because it directly targets the measured 28.6-million-parameter, 15.8-GB,
361.8-second failure. The most intellectually new object is the
self-normalized convex atom, but it is also the least validated and should
remain a separate research candidate for now.
