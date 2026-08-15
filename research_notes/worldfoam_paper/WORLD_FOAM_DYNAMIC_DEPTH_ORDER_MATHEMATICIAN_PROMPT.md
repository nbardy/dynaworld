# Research Handoff Prompt: Dynamic WorldFoam Depth-Order Closure

Date: 2026-08-03

Purpose: hand a strong mathematician or mathematical-research model the exact
WorldFoam problem we need solved. This is a paste-ready research brief, not a
claim that every requested theorem is already true.

## How To Use This File

Paste the section from `BEGIN RESEARCH PROMPT` through `END RESEARCH PROMPT`
into the research model. If attachments are supported, attach the repository
files listed in the prompt. Ask for one rigorous pass first; use its unresolved
theorem ledger to drive later passes.

Provenance warning: this brief describes the live 2026-08-03 working tree.
Several kinetic compiler, exact-root, oracle, and test files listed below are
currently dirty or untracked rather than commit-pinned. Before an external
reproducibility handoff, either commit them intentionally or bundle a manifest
with content hashes. Do not treat a path name alone as immutable evidence.
The independently tested step-scoped target-frame cache is also new worktree
state. The newer lazy native material-step coordinator is still under active
integration and is **not** evidence for a completed production trainer; do not
infer runtime or memory claims from that file until its focused lifecycle tests
and a rebuilt native gate pass.

The desired result is not more terminology. It is either:

1. a mathematically justified, implementable closure that preserves ordered
   depth transfer while moving expensive world work off the requested-frame
   axis; or
2. a precise negative result showing which restrictions are required and when
   exact per-time replay is unavoidable.

---

## BEGIN RESEARCH PROMPT

You are an expert mathematical researcher working at the intersection of:

- computational and kinetic geometry;
- power diagrams, regular triangulations, and lower envelopes;
- approximation theory and certified numerics;
- Lie groups, semigroups, product integrals, and radiative transfer;
- differentiable rendering and shape derivatives;
- reverse-mode automatic differentiation; and
- GPU-oriented algorithm design.

You are not being asked for a brainstorming list or a grand unifying name. You
are being asked to solve, narrow, or falsify one concrete mathematical problem.

Do not reveal private raw chain-of-thought. Produce an auditable research note:
definitions, equations, proof sketches, counterexamples, explicit assumptions,
branch-local conclusions, algorithm pseudocode, complexity bounds, and cheap
falsification tests.

### Core Reformulation: Retain Depth, Eliminate Repeated Replay

The problem is **not** to make a foam order-blind. It is to retain every
physically relevant depth-order effect while avoiding independent reconstruction
and reverse replay of the same ordered ray/cell interaction at every requested
frame.

For sensor/ray track \(p\) and physical time \(t\), let

\[
\mathcal W_p(t)
=((i_1,L_1),\ldots,(i_R,L_R))
\]

be the front-to-back positive-length cell word. A P0 scalar-extinction/RGB
segment has the exact affine action

\[
T_r=(\beta_r,m_r),
\qquad
\beta_r=e^{-\rho_{i_r}L_r},
\qquad
m_r=(1-\beta_r)c_{i_r},
\]

with associative, generally noncommutative composition

\[
(\beta_1,m_1)\star(\beta_2,m_2)
=(\beta_1\beta_2,m_1+\beta_1m_2).
\]

Thus a whole ordered word at one time collapses **exactly** to one four-scalar
transfer element, even though its differently colored segments cannot be
reordered or marginalized. The desired Schur-like breakthrough for foam is
therefore not a second Schur complement. It is the composition of:

1. a **kinetic geometry closure** that partitions continuous sensor time into
   event-free charts with certified owner words;
2. an **ordered-transfer closure** that evaluates each chart's exact word only
   at adaptive compiler nodes and represents total transfer between them;
3. a **sparse operator closure** that certifies the primal transfer and only the
   required world-parameter JVP/VJP actions, never a dense world Jacobian; and
4. a **streamed adjoint** that reduces all requested-frame residuals into
   bounded node cotangents before executing one word/world VJP per active
   compiled block.

Item 4 is the target factorization. The currently integrated block-major
CPU/fake-native coordinator executes one **material-word** VJP per active block.
The stable-stratum geometry and node-length VJP pieces exist separately, but a
complete native geometry/world reverse remains an explicit integration and
certification gate.

Formally, given persistent world parameters \(\theta\), a continuous camera
program \(\Gamma\), fixed physical interval \(I\), tolerance \(\varepsilon\),
and any finite query set \(T_F\subset I\), construct a program
\(\mathcal K(\theta,\Gamma,I,\varepsilon)\) independent of the density of
\(T_F\) such that every query either returns certified transfer and requested
sparse derivative actions or enters a precisely identified exact-replay
fallback.

The target complexity separation is:

\[
\begin{aligned}
W_{\rm structural+word+worldVJP}
&=O\!\left(\operatorname{Topology}(S,E,Q)
+\sum_{(p,c)}J_{p,c}R_{p,c}\right),\\
W_{\rm sample+residual}
&=O\!\left(\sum_{(p,c)}F_{p,c}J_{p,c}
+N_{\rm fb}J_{\max}^2\right)+\Omega(PF),
\end{aligned}
\]

with peak reverse interaction state bounded by persistent world state plus the
largest live spatial/native bundle and bounded target/sample blocks, not by an
\(F\times R\) tape. The \(\Omega(PF)\) term for targets, residuals, and output
is unavoidable and is not the optimization target.

This leaves three genuinely mathematical bottlenecks:

- **event complexity:** can the kinetic lower envelope/regular complex be
  compiled and repaired output-sensitively, with exact degeneracy semantics?;
- **operator rank:** can total transfer and the required sparse derivative
  actions be certified with rank controlled by physical event distance,
  motion, optical depth, and tolerance rather than requested frame count?; and
- **training reuse:** can predicate margins give useful optimizer trust regions
  and affected-chart repair, or does geometry training force near-global
  recompilation?

A useful negative theorem is acceptable. If any one of these cannot be bounded
without a representation restriction, identify the weakest restriction that
makes the theorem true: shared motion bases, bounded algebraic degree, separated
events, bounded line-stabbing depth, restricted material commutators, or an
explicit exact-replay route.

### 0. First-Pass Work Order

This brief includes a larger theorem and literature ledger so that you can see
the boundary of the problem. Do **not** try to solve every branch in the first
pass. A restricted CPU reference certificate was implemented after this brief
was first drafted. The first pass now has exactly one work package:

> Independently audit, strengthen, or falsify the implemented multichart
> **simple-root persistence and re-isolation certificate**, then determine
> whether its whole-registry proof can support an output-sensitive affected-
> chart repair or whether full recompilation should remain the production
> algorithm.

Unless you explicitly prove the stronger statement, the update domain is one
directional homotopy

\[
\theta(\eta)=\theta_0+\eta\,\Delta\theta,
\qquad \eta\in[0,r].
\]

Treat \(\theta_0\) and \(\Delta\theta\) as exact rational data: either supplied
as rationals or obtained from the exact values of rounded binary64 endpoints
and their exact difference. A certificate on this segment is **not** a
certificate for a norm ball around \(\theta_0\); a norm-ball theorem is an
optional stronger result and must introduce its own quantified perturbation
domain.

Starting from the supplied event-free certificate, active-owner compiler,
restricted re-isolation implementation, behavioral tests, and proof note:

1. define the complete canonical predicate registry with three typed classes:
   topology-event candidates; root-bearing analytic/representation guards
   such as pair-denominator roots; and non-root validity,
   positivity, or noncollapse guards;
2. group base roots by exact algebraic equality, assign disjoint rational
   neighborhoods to **distinct supported simple root groups**, and prove one
   continued event root per group using sign and nonvanishing-derivative
   bounds. In the first pass, singleton simple groups are sufficient; shared,
   repeated, persistent-zero, or ambiguous simultaneous groups must fail
   closed unless a precise joint-event semantics is proved;
3. do not equate polynomial-root persistence with semantic event persistence.
   At every continued root, exactly reclassify activity/co-minimality and the
   certified left/right owner words. A root is a topology event only when the
   ordered owner word genuinely changes. Either preserve the needed activity
   margins along the root graph or reclassify and merge the isolated roots
   after continuation;
4. continue or re-isolate root-bearing analytic guards without promoting them
   to topology events; split a representation/cut chart only when its own
   validity requires it, and do not charge such a split to semantic event
   count \(E\). Certify non-root guards—especially \(\|d(t)\|^2>0\)—uniformly
   separated from zero over the full update tube, then prove complementary
   time cells root-free, preserve the order of the **semantically active**
   event groups, and retain one certified owner word between consecutive moved
   groups;
5. audit the current exact complexity
   (O(U S R)+O((K+M)2^D)+M\,T_{\mathrm{Sturm}}+(M+1)O(S\log S)), then give
   an output-sensitive algorithm that re-isolates/refits moved roots and
   updates only affected chart payloads, or a counterexample/lower bound that
   justifies full recompilation; and
6. provide the smallest counterexample when any assumption is removed.

Treat the fixed-SPD(4) slice theorem, generic quartic predicate theorem,
active-owner closure, node-length seam, and event-free trust certificate as
supplied lemmas; the new simple-root certificate itself is the object to audit.
Defer bounded-cell sphere/vacuum events, global 4D meshing, new material bases,
and image-wide compiler engineering to later passes. The answer must finish
with exactly one next mathematical implementation and one kill diagnostic. A
valid answer may be a negative theorem showing that only recompile-from-scratch
is sound under the current predicate registry.

### 1. Mission In One Sentence

Find the best exact or certified representation of a dynamic foam's
time-dependent **ordered ray transfer and sparse world-parameter JVP/VJP
actions** such that increasing the requested frame density changes only cheap
sample evaluation and residual reduction, while topology discovery, cell-word
traversal, world differentiation, and reverse interaction storage scale with
physical event/rank complexity rather than independent frame replay. Do not
materialize a dense world Jacobian.

### 2. Frozen Conclusions You Must Preserve

These are current decisions, not questions to reopen without a counterexample.

1. **Do not seek another Gaussian depth marginal.** World Tubes can eliminate
   ray depth analytically because a pulled-back Gaussian is closed under
   marginalization; its conditional covariance is a Schur complement.
   WorldFoam intentionally retains depth order. An order-blind marginal would
   erase the differently colored overlap phenomenon it exists to model.

2. **The target is not sublinear image output.** Writing or comparing \(PF\)
   RGB samples has an unavoidable \(\Omega(PF)\) cost. The target is that the
   expensive geometry, topology, ordered-word, and world-VJP work is invariant
   to requested temporal sample density \(F\) over a fixed physical interval,
   fixed camera program, fixed world, and fixed accuracy tolerance.

3. **An open camera ray uses ordered parallel transport, not holonomy.** You
   may use holonomy as geometric inspiration for closed loops, but call the
   forward object an ordered transfer, product integral, or path-ordered
   exponential.

4. **Do not rename the project.** `World Tubes` remains the primary Gaussian
   paper. `WorldFoam` remains the retained-depth/cellular sibling. `Power Foam`
   and `Radiant Foam` are implementation/prior-art lineages, not synonyms for
   the dynamic compiler proposed here.

5. **Do not assume a universal material basis.** The repository's equal-byte
   M3/M5 experiment found complementary exact-family wins. Material selection
   is separate from the structural depth-order problem.

6. **Distinguish three meanings of dynamic.** They are not interchangeable:

   - physical scene time changes the slice/ray through a persistent world;
   - optimizer time moves sites and weights and may invalidate topology; and
   - static-foam densification, pruning, or connectivity changes during
     optimization do not by themselves constitute a dynamic-scene model.

7. **The fixed shared-SPD(4) power world is a restricted exact route.** In a
   fixed world-coordinate gauge, its slices are one common translation of
   fixed anisotropic 3D sites with affine relative weights and constant
   candidate-face normals. A common time-dependent coordinate gauge can align
   one chosen normal at the level of coordinates, but admissibility depends on
   the metric: Euclidean rigid co-rotation preserves an isotropic spatial power
   form, whereas a general anisotropic metric is preserved only by its own
   isometry group; time-dependent scale/affine/projective changes also require
   transformed weights, density, and optical line element. Even when admissible,
   one common gauge cannot generically make several independently changing
   relative face orientations time-independent. State the gauge group and
   preservation law before using it; the restriction is not invariant under an
   unspecified arbitrary coordinate change. The
   selected general geometry candidate therefore composes a shared
   camera/scene gauge for bulk motion with direct affine kinetic 3D residual
   sites and quadratic weights. Its exact fixed-time algebra and guarded
   quartic predicate isolation, exhaustive continuous owner-chart compilation,
   an independent exact oracle, an active-owner closure, exact multi-chart
   dispatch, continuous material-action certification, and a stable-stratum
   sparse geometry/material VJP are implemented at CPU scope. Predicate
   construction is \(O(U S R_{\max})\) over unique witnessed owner words; the
   current closure additionally pays
   \(O(W(S\log S+S R_{\max}))\) for cumulative root-complement discoveries and
   all-site certificates. A provenance-sealed CPU lowering now emits CSR owner
   words plus positive physical lengths at compiler nodes, with a matching CPU
   Lie oracle, stable-stratum length-to-geometry VJP, and source-only native
   forward/VJP and ragged sample-reduction ABIs. A CPU equal-rank lowerer now
   packs heterogeneous `(track,chart)` rows into bounded actual-`J` buckets
   without a global temporal refinement or `J_max` padding, and a CPU outer
   coordinator merges compact view/block material bars into one caller-owned
   global bar under one loss denominator and one optimizer authorization. A
   CPU union-local assembler now joins all heterogeneous native blocks touched
   by one spatial request into one caller-owned compact union bar before that
   global scatter, with exact missing/duplicate/foreign coverage checks. A
   CPU/fake-native block-major paper step now holds each spatial bundle across
   every temporal request, accumulates one bounded node cotangent per native
   block, runs each material-only word VJP once, and only then scatters the
   union bar. It allocates no `[J,W]` geometry bar. `K=1/4` matches a direct-
   autograd oracle; densifying `F=5` to `F=41` leaves node-forward/word-VJP
   counts and retained runtime bytes invariant; sequential spatial bundles
   peak at the largest bundle rather than their sum. This is CPU/fake-native
   integration evidence, not rebuilt-Metal invocation or allocator evidence. An
   exact-rational directional trust certificate proves a nonzero reuse radius
   for one strict event-free chart over the full closed time interval; active,
   endpoint-event, or multichart programs deliberately return zero in that
   *event-free* gate. A separate CPU reference now accepts a restricted
   multichart stratum: it rebuilds the complete rooted/rootless predicate
   registry, proves separated singleton simple-root tubes and root-free
   complements over the whole exact binary64 directional homotopy, re-isolates
   candidate endpoint roots, and reclassifies analytic roots versus semantic
   owner changes. Repeated/shared/persistent-zero/endpoint/collapse/ambiguous
   cases fail closed. It is a whole-registry certificate, not output-sensitive
   warm payload repair or an event-time derivative. The extension has not been rebuilt or
   executed after these edits, and row packing is not yet a production
   image-wide kinetic compiler. It is a certificate/oracle, not a patcher: it
   returns no repaired program and rebuilds no charts, ranks, payloads, or
   dispatch. Production must currently recompile and recertify after every
   geometry or camera-ray update; only material-only updates may reuse the
   sealed structure. Bounded-cell sphere/vacuum events, supported
   persistent/simultaneous-event semantics, output-sensitive local repair, and
   derivatives through event/chart/schedule changes remain open.

### 3. Why Existing Static Foams Are Fast

The relevant prior art is fast for a narrower reason than the phrase
"constant-time ray traversal" may suggest.

#### 3.1 Radiant Foam

Radiant Foam partitions \(\mathbb R^3\) into non-overlapping Voronoi cells:

\[
V_i=\left\{x:\operatorname*{arg\,min}_j\|x-p_j\|^2=i\right\}.
\]

Once the first cell is known, a ray walks face-to-face through the dual
Delaunay adjacency. The implementations rely on small observed or average
local degree, so the transition cost is treated as amortized constant. An
arbitrary 3D Voronoi/Delaunay cell does not have a worst-case degree bound that
depends only on dimension. This does **not** make the whole ray constant time:
if it crosses \(R\) cells, traversal is still \(\Theta(R)\) under bounded or
amortized local degree, plus first-cell lookup; otherwise the actual tested
neighbor counts must be included.

P0 density/color lets each ray-cell segment be integrated exactly:

\[
\alpha_i=1-\exp(-\sigma_i L_i),
\]

so there is no dense sample march inside a segment. Non-overlap also prevents
the redundant overlap tests suffered by unstructured primitives.

#### 3.2 Power Foam

Power Foam uses power cells intersected with their controlling spheres:

\[
P_i=\left\{x:
\operatorname*{arg\,min}_j
\left(\|x-p_j\|^2-r_j^2\right)=i
\right\}\cap B(p_i,r_i),
\]

with a spherical bound controlled by the same radius. The bound enables tile
culling and rasterization while adjacency still supports ray walking. During
training it avoids repeatedly building the exact regular triangulation by using
the Čech graph of overlapping spheres, a conservative superset of the required
alpha-complex adjacency. False candidate faces do not change the exact cell
intersection, although they add tests.

#### 3.3 What this does not solve

Static foam speed comes from:

- one spatial owner instead of many overlapping primitives;
- neighbor-to-neighbor traversal rather than a global search at every crossing;
- exact segment integration instead of dense ray marching;
- spatial bounds and tile culling; and
- early transmittance termination.

It does **not** provide:

- shallow line-stabbing depth;
- a time-varying cell complex;
- certified camera-ray order events;
- reuse of one ordered word across a continuous camera program;
- a cross-frame shared adjoint; or
- an amortized topology-maintenance theorem under learned geometry updates.

The current literature search found no published neural-rendering foam that
unifies all six. Treat that as a literature-audit result to verify, not as a
license to overclaim novelty.

### 4. Formal World And Observation Model

#### 4.1 Native 4D power world

Let physical spacetime be

\[
X=(x,t)\in\mathbb R^3\times I.
\]

For sites \(q_i\in\mathbb R^4\), weights \(w_i\in\mathbb R\), and a shared
metric \(M\in\operatorname{SPD}(4)\), define

\[
\Pi_i(X)=(X-q_i)^TM(X-q_i)-w_i,
\]

\[
C_i=\{X:\Pi_i(X)\leq\Pi_j(X)\ \forall j\}.
\]

Because the quadratic form is shared, every pairwise difference

\[
H_{ij}(X)=\Pi_i(X)-\Pi_j(X)
\]

is affine in \(X\). The world is a static partition in spacetime; physical
motion appears when it is sliced by \(t\).

This lifting has an exact and important expressivity restriction. Write the
shared metric as

\[
M=
\begin{bmatrix}
A&b\\
b^T&c
\end{bmatrix},
\qquad
\lambda=c-b^TA^{-1}b>0,
\qquad
v=A^{-1}b.
\]

For a static 4D site \(q_i=(a_i,\tau_i)\), completing the spatial square gives

\[
\Pi_i(x,t)
=
\left\|x-\left(p_i^0-vt\right)\right\|_A^2
+\lambda(t-\tau_i)^2-w_i,
\qquad
p_i^0=a_i+v\tau_i.
\]

Every effective 3D site therefore has the same velocity \(-v\). After the common
translation \(y=x+vt\), the centers are fixed and the relative power weights are
affine:

\[
\widehat w_i(t)
=2\lambda\tau_i t+w_i-\lambda\tau_i^2,
\]

because the common \(-\lambda t^2\) term cancels from ownership. Conversely,
any fixed anisotropic 3D sites with affine relative weights and one common
translation lift to such a fixed shared-SPD(4) world. Thus this is exactly a
restricted translating-face kinetic 3D family, not merely an informal
limitation.

Every pairwise face can be written

\[
H_{ij}(x,t)=n_{ij}^{T}x+\eta_{ij}t+b_{ij}.
\]

Consequently, the spatial normal \(n_{ij}\) of a sliced face is constant in
time; only its offset changes. For the executable \(M=I\) specialization,
\(v=0\): effective centers do not move at all and dynamics comes from affine
relative weights selecting and resizing cells. In this fixed coordinate gauge,
a persistent rotating face is impossible with one fixed site pair. A common
time-dependent coordinate change can freeze one chosen orientation, so the
fixed-gauge statement is not an absolute coordinate-invariant impossibility
theorem. For isotropic spatial power distance, rigid co-rotation is admissible;
for general anisotropic \(A\), only its isometries preserve the fixed metric
without transforming it. Scale, affine, projective, or diffeomorphic gauges
have more freedom but must separately transform and preserve power ownership,
ray order, optical line element, and the compact parameterization. Even under
an admissible common gauge, several independently changing *relative* face
orientations cannot generically all be made time-independent. The fixed
shared-SPD(4) representation therefore remains a restricted exact special case
and possible bulk-motion gauge, rather than the whole general WorldFoam motion
model.

The repository now contains the minimal direct kinetic 3D alternative:

\[
p_i(t)=p_{i,0}+t v_i,
\qquad
\omega_i(t)=\omega_{i,0}+\omega_{i,1}t+\omega_{i,2}t^2.
\]

For the current affine ray program, each pairwise power difference is

\[
D_i-D_j=A_{ij}(t)z+B_{ij}(t),
\]

with \(\deg A_{ij},\deg B_{ij}\leq2\); adjacent-cut concurrence

\[
B_{ij}A_{jk}-B_{jk}A_{ij}=0
\]

has degree at most four. After factoring shared camera/scene motion into one
common gauge, \(p_j(t)-p_i(t)\) can still change direction, so this frontend
represents residual rotating faces that one common gauge cannot generally
freeze simultaneously. Its persistent geometry uses only
\(p_{i,0}\), \(v_i\), and at most three weight coefficients per site, so
parameter bytes are independent of requested frames. The executable frontend
uses the Euclidean metric \(A=I\): six affine-position scalars plus three
weight coefficients and P0 density/RGB total thirteen scalars per site. A
general shared \(A\in\operatorname{SPD}(3)\) is only a derived/proposed
extension unless it is explicitly whitened together with rays and the optical
line element. The current exact CPU
frontend derives these polynomials and exact fixed-time owner words. An exact
rational square-free/Sturm primitive isolates roots through quartics. An
exhaustive CPU reference compiler now enumerates all pair near/far and finite
triple candidates, keeps denominator roots as analytic guards, emits certified
half-open continuous owner charts, and is cross-checked against an independent
all-pair/all-triple oracle. This is an \(O(S^3)\) small-world correctness route,
and a separate active-owner compiler now derives candidates from witnessed
endpoint owners and active cuts, caches predicate sources by unique word,
closes newly discovered root complements monotonically, and differentially
matches the exhaustive compiler on its supported strata. Its honest bound is
\(O(U S R_{\max})\) predicate construction plus
\(O(W(S\log S+S R_{\max}))\) closure/certification work, not an unqualified
\(O(SR)\) claim. A CPU multi-chart bridge and stable-stratum sparse VJP now
cover positions, velocities, quadratic weights, affine rays, density, and RGB
with fixed program choices. A provenance-sealed single-ray lowerer and CPU
oracle now realize the exact CSR-owner-plus-node-length interface; source-only
Metal/C++/Python ABIs realize Lie-node forward/VJP and a row-ragged sample
reducer, but remain unbuilt and runtime-unverified. A CPU equal-rank packer
streams canonical `(track,chart)` rows in bounded actual-rank buckets with
union-compacted site IDs and concatenated `[J,W_total]` physical lengths; it
does not pad to a global rank or retain an `F` axis. A CPU multi-view outer
coordinator preserves one global denominator, one caller-owned material bar,
and exactly one optimizer authorization. A narrow exact-rational trust-region
certificate proves continuous owner-word reuse for one strict event-free
chart and returns zero for eventful/multichart programs. Production image-wide
kinetic compilation, multichart root persistence and re-isolation, bounded-cell
events, full-fiber/simultaneous-event support beyond fail-closed behavior, and
derivatives through structural choices remain open.

The selected kinetic theorem scope is currently an **unbounded Euclidean power
partition clipped by global near/far**, closer to the Radiant-Foam geometric
core than full bounded Power-Foam parity. Moving controlling spheres would add
sphere entry/exit, vacuum-gap, radius-positivity, adjacency/culling, and VJP
events. None of those are covered by the present pair-endpoint/triple theorem;
they are a later extension, not an implicit feature of the word "power."

The degree bounds above hold in a chart where site trajectories and ray
origin/direction are affine and weights are degree at most two. Factoring a
non-affine time-dependent rigid rotation into the camera may turn those
coefficients into trigonometric or rational functions. A shared gauge is
therefore useful only if its transformed trajectory class is stated and its
event predicates are rederived or piecewise certified; it does not inherit the
quadratic/quartic theorem automatically.

The metric also requires units. Introduce characteristic scales
\(\ell_0>0\) and \(\tau_0>0\), use dimensionless coordinates

\[
\bar x=x/\ell_0,
\qquad
\bar t=t/\tau_0,
\]

and state the equivalent time-to-length scale encoded by \(M\). The current
executable reference uses \(M=I\) only in its supplied normalized coordinates.
Allowing a general shared \(M\) is a proposed theorem scope, not an implemented
feature. Analyze sensitivity to \(\ell_0/\tau_0\); mixing unscaled meters and
seconds inside a Euclidean norm is not meaningful.

Verify or sharpen the exact slice characterization and its approximation lower
bound for multiple independently rotating faces modulo one common scene gauge.
Then solve the remaining direct-kinetic production problem: audit the active
compiler's \(U/W\)-dependent closure against the exhaustive reference and
independent oracle, determine whether a certified neighbor/event queue can
improve it, complete the degeneracy semantics, and give an output-sensitive
maintenance, invalidation, and native-lowering contract.

#### 4.2 Camera-ray strip and gauge

For sensor track \(p\), use the current affine chart

\[
\Gamma_p(t,z)
=
\bigl(o_p(t)+z\,d_p(t),t\bigr),
\]

\[
o_p(t)=o_{p,0}+t\,o_{p,1},
\qquad
d_p(t)=d_{p,0}+t\,d_{p,1}.
\]

The physical line element is

\[
ds=\|\partial_z\Gamma_p(t,z)\|\,dz=\|d_p(t)\|\,dz.
\]

Under any orientation-preserving depth reparameterization
\(z'=\phi_{p,t}(z)\), the optical measure must remain invariant:

\[
\sigma(\Gamma_p(t,z))\,\|\partial_z\Gamma_p(t,z)\|\,dz.
\]

Gauge/chart changes may improve conditioning but must not alter physical
optical depth, order, or the rendered observable.

#### 4.3 Fixed-time lower-envelope reduction

After separating terms common to every site, the pulled-back 4D power distance
has the form

\[
\Pi_i(\Gamma_p(t,z))
=
Q_p(t,z)+a_{p,i}(t)z+b_{p,i}(t),
\]

where \(Q_p\) is independent of the site. It cancels from ownership. Therefore
the exact owner word is the lower envelope in \(z\) of the lines

\[
\ell_{p,i}(t,z)=a_{p,i}(t)z+b_{p,i}(t).
\]

For the current common-metric 4D sites and affine ray program,
\(a_{p,i}(t)\) and \(b_{p,i}(t)\) are affine in \(t\). Pairwise face cuts are
Möbius functions:

\[
z_{p,ij}(t)
=
-\frac{B_{p,ij}(t)}{A_{p,ij}(t)},
\]

with affine \(A,B\). Near/far crossings are linear predicates only for fixed
near/far coordinates in this affine depth chart. Time-varying clipping,
log-depth coordinates, and projective gauges require rederived endpoint/event
predicates and their gauge Jacobians; do not silently reuse the linear result.
Adjacent-cut concurrence is the quadratic predicate

\[
B_{ij}(t)A_{jk}(t)-B_{jk}(t)A_{ij}(t)=0.
\]

Equivalently, the owner word is a kinetic lower hull of the dual points
\((a_i(t),b_i(t))\). This special structure is likely more useful than a
generic 4D arrangement and should be exploited before proposing cylindrical
algebraic decomposition.

#### 4.4 Ordered owner word

At time \(t\), write the positive-length ray-cell sequence as

\[
\mathcal W_p(t)=
\bigl((i_1,z_0,z_1),\ldots,(i_R,z_{R-1},z_R)\bigr),
\]

in front-to-back order. A valid topology chart
\(\mathcal T_c=[\alpha_c,\alpha_{c+1})\) keeps owner identities, order, and endpoint
identities fixed, keeps every required denominator away from zero, and excludes
unresolved full-fiber ties.

A zero-length finite-density segment has identity transfer, so a birth/death
event can use a deterministic right-continuous seam convention. A positive-
length full-fiber tie between differently materialized cells is not harmless;
it requires an explicit physical tie rule or fail-closed dispatch.

### 5. Ordered Optical Transfer Is The Candidate Closure

The immediate renderer scope is primary-ray emission--absorption with scalar
extinction and RGB emission. It does not include multiple scattering,
polarization, or a general matrix-valued participating medium. Generalize only
after solving or sharply characterizing this smaller problem.

For a P0 segment \(r\) of owner \(i_r\), let

\[
L_r(t)
=
\|d_p(t)\|\,[z_r(t)-z_{r-1}(t)],
\qquad
\tau_r(t)=\rho_{i_r}L_r(t),
\]

\[
\beta_r(t)=e^{-\tau_r(t)},
\qquad
m_r(t)=(1-\beta_r(t))c_{i_r}.
\]

Represent the action on incoming background radiance \(c\) by

\[
G_r(c)=m_r+\beta_r c.
\]

Front-to-back composition is associative:

\[
(\beta_1,m_1)\star(\beta_2,m_2)
=
(\beta_1\beta_2,\ m_1+\beta_1m_2).
\]

Equivalently,

\[
G(\beta,m)=
\begin{bmatrix}
\beta I_3&m\\
0&1
\end{bmatrix}.
\]

Within this P0, scalar-extinction, view-independent RGB emission--absorption
scope, at a **single time** a word of any length collapses exactly to four
transfer scalars: one \(\beta\) and RGB \(m\). View-dependent bases, P1 or
higher-order material variation, scattering, and matrix-valued transport may
require larger sufficient statistics. The real problem in the stated P0 scope
is not the dimension of the pointwise output. It is to represent, certify,
evaluate, and differentiate the functions

\[
t\mapsto G_p(t),
\qquad
t\mapsto [D_\theta G_p(t)]v_\theta,
\qquad
t\mapsto [D_\theta G_p(t)]^T\lambda,
\]

for specified sparse parameter directions \(v_\theta\) and output cotangents
\(\lambda\), over topology-stable charts without replaying every cell word at
every requested time.

The current coordinate candidate is

\[
\kappa=-\log\beta,
\qquad
v=\frac{\kappa}{1-\beta}m,
\]

with inverse

\[
\beta=e^{-\kappa},
\qquad
m=\frac{1-e^{-\kappa}}{\kappa}v.
\]

The \(\kappa=0\) singularity is removable. For RGB colors in \([0,1]^3\), the
physical cone is

\[
\kappa\geq0,
\qquad
0\leq v_j\leq\kappa.
\]

This affine-transfer Lie chart is the current **systems analogue** of the
World Tubes Schur closure: exact ordered composition happens at \(J\) compiler
nodes, then total transfer is represented across time. It is not another depth
marginal and is not yet proven optimal.

### 6. Noncommutativity Is The Essential Obstruction

Two differently colored segments generally do not commute:

\[
(\beta_1,m_1)\star(\beta_2,m_2)
\neq
(\beta_2,m_2)\star(\beta_1,m_1).
\]

Their emitted-moment difference is

\[
(1-\beta_2)m_1-(1-\beta_1)m_2.
\]

Attenuation alone does commute because total optical depth is additive:

\[
\kappa_{\mathrm{total}}=\sum_r\tau_r.
\]

If all segments share the same source color, order may also collapse. For
arbitrary colors, an order-blind statistic cannot be exact. Your proposed
closure must therefore do at least one of the following explicitly:

- preserve the order event structure;
- prove a restricted commuting material family;
- bound the error from approximate commutation; or
- fail closed to exact ordered replay.

Do not conceal this obstruction inside a learned decoder.

### 7. Desired Compiler And Adjoint Factorization

Let:

- \(S\): number of persistent world sites;
- \(P\): number of sensor/ray tracks processed by the full observation;
- \(\mathcal I=\{(p,c)\}\): the ragged set of stored, nonempty track-local chart
  incidences; a global common refinement may be used semantically for a proof,
  but it must never be materialized as a dense \(P\times C\) table;
- \(F_{p,c}\): requested samples for track \(p\) on its local chart \(c\), with
  \(PF=\sum_{(p,c)\in\mathcal I}F_{p,c}\) for a uniform \(F\)-sample query;
- \(K\): bounded streamed temporal block size;
- \(B_p\): bounded spatial track block size;
- \(N_B=\lceil P/B_p\rceil\): number of spatial track blocks;
- \(R_{p,c}\): word length for track \(p\) on chart \(c\);
- \(W=\sum_{(p,c)\in\mathcal I}R_{p,c}\): total stored run-chart incidences;
- \(E\): genuine topology/order event records;
- \(Q\): total active track-face incidences over the ragged program;
- \(J_{p,c}\): compiler nodes/rank on one stored track-local chart;
- \(N_{\mathrm{fb}}\): total sample rows that take the row-local dense
  interpolation fallback; and
- \(\varepsilon\): requested forward-and-derivative-action tolerance.

The formal input is a compact continuous world parameter vector \(\theta\), a
continuous camera program \(\Gamma\), an interval \([t_0,t_1]\), a requested
tolerance \(\varepsilon\), and an arbitrary finite query set
\(T_F\subset[t_0,t_1]\). The compiler must construct \(\mathcal K\) from the
continuous world/camera program rather than from the density of \(T_F\). For
every query it must either (a) return the exact owner dispatch and a transfer
plus specified sparse derivative actions within \(\varepsilon\), or (b) mark a
precise conflict region for exact ordered replay. Its structural state and
world-side reverse state may depend on \(S,E,Q,W,J,\varepsilon\), but not on
\(F\) alone. Sample/output work and streamed residency remain separate and may
scale with \(F\).

The desired map is

\[
(\theta,\Gamma)
\xrightarrow{\text{structural compile}}
\mathcal K
\xrightarrow{\text{numeric compile at }J_c\text{ nodes}}
a
\xrightarrow{\text{sample evaluate}}
\{\widehat C_{p,f}\},
\]

with reverse

\[
\{\bar C_{p,f}\}
\longrightarrow
\bar a
\longrightarrow
(\bar\theta,\bar\Gamma).
\]

For P0 material, there is now a sharper exact interface between geometry and
transfer. At compiler node (t_{p,c,j}), let the certified ordered word be
((o_{p,c,1},\ldots,o_{p,c,R_{p,c}})), let its consecutive ray cuts be
(z_0<\cdots<z_{R_{p,c}}), and define physical lengths

\[
\ell_{p,c,j,r}
=\lVert d_p(t_{p,c,j})\rVert
\bigl(z_{r+1}(t_{p,c,j})-z_r(t_{p,c,j})\bigr).
\]

Then the exact node transfer factors as

\[
(\theta_{\rm geom},\Gamma)
\longrightarrow
\bigl(o_{p,c,r},\ell_{p,c,j,r}\bigr)
\longrightarrow
T_{p,c,j}
=\prod_{r=1}^{R_{p,c}}
T\!\left(\rho_{o_r}\ell_{p,c,j,r},c_{o_r}\right)
\longrightarrow
G_{p,c,j}=\log_{\rm Aff}(T_{p,c,j}).
\]

Its frozen-program reverse factors in the opposite direction:

\[
\bar G_{p,c,j}
\longrightarrow
(\bar\rho,\bar c,\bar\ell_{p,c,j,r})
\longrightarrow
(\bar p_0,\bar v,\bar w,\bar\Gamma).
\]

This is the systems analogue of the World Tubes Schur closure, but it is not a
Schur complement and does not marginalize depth. It quotients away only the
within-cell continuous coordinate after preserving the entire ordered word.
The noncommutative product—and therefore every depth-order change—remains
explicit. The sufficient native node payload is consequently CSR owners plus
`[J,R]` physical lengths, not a frame-by-run tape.

This interface is also why the legacy static-site native refresh cannot be
reused honestly for general kinetic sites. For
(p_i(t)=p_{i,0}+t v_i) and quadratic (w_i(t)), a pair face has a
time-linear normal and time-quadratic bias. Substitution of an affine camera
ray gives, generically,

\[
z_{ij}(t)=\frac{N_{ij,2}(t)}{D_{ij,2}(t)},
\]

a quadratic-over-quadratic cut rather than the old Mobius
linear-over-linear cut. The selected lowering therefore compiles certified
node lengths outside the sample loop. A future native kinetic-boundary ABI is
an alternative, not a prerequisite for the node-length route.

Keep three derivative contracts separate:

1. The **frozen-program VJP/JVP** differentiates the numeric transfer with
   structural charts, event intervals, owner dispatch, adaptive nodes/rank,
   interpolation weights, and certificate decisions fixed. This is the scope
   of the current executable sparse adjoint.
2. The **physical rendered-loss derivative** differentiates the exact physical
   observable. For fixed point samples inside a regular topology stratum, the
   owner labels are locally constant and no derivative of the discrete compiler
   choice exists; endpoint geometry derivatives still matter. Exposure or
   other time-integrated losses additionally acquire event-boundary terms when
   the one-sided physical integrands differ.
3. The **full compiled-algorithm derivative** additionally differentiates the
   adaptive approximation procedure: selected charts, event isolators,
   interpolation nodes/weights, rank, fallback, and dispatch decisions. It is
   distinct from the physical derivative and may intentionally be stopped or
   replaced by a certified surrogate.

The frame-density/memory theorem below applies immediately to the first
contract. A claim about end-to-end geometry training must specify which
physical objective in contract 2 is intended and whether contract 3 is
differentiated, bounded as an approximation, or stopped deliberately.

The intended heavy world work is approximately

\[
O\!\left(
\operatorname{Topology}(S,E,Q)
+\sum_{(p,c)\in\mathcal I} J_{p,c}R_{p,c}
\right)
\]

for compilation and again for the world VJP. The requested-sample slice may be

\[
O\!\left(\sum_{(p,c)\in\mathcal I}F_{p,c}J_{p,c}
+N_{\mathrm{fb}}J_{\max}^2\right)
\]

plus unavoidable target reads, residual formation, and writes. The common path
is linear in the local \(J_{p,c}\); each exceptional fallback row may cost
quadratic work in its local rank, so any unconditional \(O(FJ)\) claim is false
unless \(N_{\mathrm{fb}}=0\) is proved. Peak interaction state should be bounded by
\(B_p,K,J_{\max}\), local runs/incidences, global world gradients, and event
records—not by an \(F\times R\) reverse tape or a resident \(P\times F\)
target/ray tensor.

For the current special case where charts, nodes, and query times are shared
across spatial blocks, sample-weight construction is still repeated per block:

\[
O\!\left(N_B\sum_c F_cJ_c+N_{\mathrm{fb}}J_{\max}^2\right).
\]

A future validated global weight cache can reduce the first factor from
\(N_B\) to one. The general ragged formula above must not be implemented by
padding every track into a global common refinement.

For one equal-rank native row bucket with \(P_b\) track-chart rows,
\(W_b\) flattened ordered runs, and rank \(J\), the new precompiled-length
forward/reverse seam has the following dominant float32 payload terms:

\[
\underbrace{4JW_b}_{\text{node lengths}}
+\underbrace{16P_bJ}_{\text{node Lie charts}}
+\underbrace{4JW_b}_{\text{node-length bars}}
+\underbrace{16P_bJ}_{\text{node Lie bars}}
+\underbrace{32S_b}_{\text{compact material plus bars}},
\]

plus small CSR/configuration arrays. For \(N\le B_pK\) sample rows, the
native-prepared reducer block retains \(4NJ+16N+O(1)\) logical bytes: float32
weights, RGB targets, and device row IDs. The CPU/source bridge block instead
retains \(4NJ+24N+O(1)\): the same weights/targets, CPU row IDs, and an 8-byte
flat provenance index. If both wrappers coexist during handoff, their unique
logical payload is at least \(4NJ+28N+O(1)\) before bridge-only row/provenance
storage is released; allocator reservations and transfer temporaries remain
unmeasured. The training-only reducer need not allocate predictions. Crucially,
none of these terms contains the total requested frame count. Adaptive ranks
should be bucketed or streamed; padding every row to \(J_{\max}\) is not part
of the claim.

The theorem we ultimately need should say:

> If only the requested temporal sample set becomes denser while the world,
> continuous camera program, physical interval, and tolerance remain fixed,
> then structural charts, owner words, event records, numeric rank, world-side
> operations, and reverse interaction memory remain invariant.

Increasing physical duration, camera curvature, scene-motion frequency,
material contrast, or accuracy may legitimately increase charts, events, or
rank. Do not confuse those variables with frame density.

### 8. Current Repository Evidence

Treat the status words literally:

- **CPU-executable/tested** means the Python reference or adapter behavior ran
  in focused tests.
- **Fake-native/CPU-lifecycle-tested source ABI** means Python adapter behavior
  agrees with an injected CPU implementation of the native call contract; it
  still has not been rebuilt or run on Metal after these edits.
- **Source-contract-only native ABI** means Metal/C++/Python schemas, source
  invariants, and arithmetic contracts agree, but no fake-native lifecycle or
  rebuilt-device execution has been demonstrated.
- **Open** means neither a theorem nor a complete implementation exists, even
  when a lower-level primitive already does.

The following is current evidence and reusable machinery, not
publication-scale proof:

- exact fixed-time owner discovery via an \(O(S\log S)\) lower envelope;
- only adjacent active owner pairs emitted, without an \(S^2\) boundary table;
- dyadic-rational near/far and quadratic triple-event predicates that are exact
  relative to the supplied binary64 inputs, not to unknown latent geometry;
- rational roots and certified isolating intervals for irrational roots;
- exact P0 ordered transfer and a constant-state prefix-only word VJP;
- sparse track-boundary coefficient reduction and once-per-incidence lowering;
- boundary-to-4D-site/weight scatter;
- affine-transfer Lie encode/decode and physical-cone checks;
- adaptive, separate primal and required tangent/VJP-action rank gates;
- continuous fixed-topology transfer, owner, and derivative certificates;
- compact track-local schedules rather than a full global dual state;
- exact streamed \(B_p\times K\) target staging and one global loss
  normalization; strict/evaluation replay can also stage explicit rays, while
  material training validates one bounded reference ray row and then carries
  targets only;
- source-level piecewise-topology dispatch with right-continuous seams;
- a loss-only native source ABI that reduces directly into loss and node
  cotangents without allocating a discarded \(B_p\times K\times3\)
  prediction tensor;
- a source-only material reverse ABI, selected only by the non-paper
  `training_owner_topology_only` binding, that skips geometry reverse and omits
  the per-block gradient tensors \([Q_b,4]\), \([B_b,5]\), and \([S_b,5]\)
  while retaining the RGBA gradient \([S_b,4]\); strict/evaluation bindings
  keep the full geometry reverse;
- owner-only P0 material optimization with manual chain-rule VJPs, zero
  retained compiled CPU atlases per spatial block, zero per-step CPU atlas
  compiles, and a program retaining lightweight topology, compact schedules,
  and owner bindings rather than compiled CPU atlases; the session additionally
  retains the bounded native-token cache below;
- a session-owned, fail-closed **explicit byte-bounded LRU** cache of validated
  native topology tokens, keyed by immutable program/binding, schedule, device,
  and native-ops identity. Its policy separately bounds cached entry count,
  cached tensor bytes, and cached-plus-one-live-token bytes; it preflights and
  evicts before allocation, checks actual token bytes afterward, and may retain
  zero, one, or several tokens depending on the caller's declared budget. A
  cold step prepares a missing block and a matching later step may reuse it;
- fit-derived second-form barycentric sample weights with an \(O(KJ)\)
  common path per staged spatial/temporal block, exact-node rows, and explicit
  \(O(J^2)\) row-local fallback. The current material trainer repeats the
  weight construction over spatial blocks, so its measured full-step contract
  is \(O(N_BFJ+N_{\mathrm{fb}}J_{\max}^2)\), not an unqualified \(O(FJ)\);
- an exact direct affine kinetic 3D CPU frontend with
  \(p_i(t)=p_{i,0}+t v_i\), degree-\(\leq2\) weights, exact
  degree-\(\leq2\) ray-cut coefficients, degree-\(\leq4\) concurrence
  polynomials, fixed-gauge rotating-face coverage, exact fixed-time words,
  guarded exact rational root isolation through quartics, and parameter bytes
  independent of requested frame count;
- an exhaustive \(O(S^3)\) continuous CPU chart compiler with exact algebraic
  root grouping, right-continuous half-open charts, all-site owner witnesses,
  and fail-closed degeneracy records, plus an independent global-product/Sturm
  oracle that exposed and helped repair a production Sturm-sign normalization
  bug; neither component samples requested frames;
- an active-owner CPU compiler that matches the exhaustive/oracle routes on
  supported strata, fails closed on inactive full-fiber ties, and reports its
  unique-word and cumulative-discovery work rather than hiding it behind final
  chart count;
- a CPU multi-chart transfer program with exact right-continuous binary sample
  dispatch, no dense sample-by-chart table, actual second-form barycentric
  primal/material-action certification, and an \(O(\sum_c J_c)\) streamed
  residual-to-node reduction;
- a frozen-program stable-stratum VJP for positions, velocities, quadratic
  weights, affine rays, density, and RGB. Its world reverse is
  \(O(\sum_c J_cR_c)\) and retains no frame tape, but it deliberately stops
  chart endpoints, event times, dispatch, node times, rank, interpolation
  weights, and compiler choices;
- a provenance-sealed, frame-independent CPU lowering from one kinetic chart
  to compact CSR owners plus positive \([J,R]\) physical lengths, together
  with an independent Lie-node forward/VJP oracle. The lowering retains no
  requested times, targets, frame count, or sample count;
- a tensor-free CPU batch descriptor and bounded materializer that assigns one
  canonical row to each `(track,chart)`, buckets rows by their actual rank,
  union-compacts global site IDs, remaps CSR owner words, and emits
  `[J,W_total]` node lengths without global temporal refinement or `J_max`
  padding. Cold content/provenance checks are separated from warm
  identity/layout/version checks. Its byte report includes canonical descriptor
  metadata and explicitly excludes unmeasured Python allocator and runtime
  peaks;
- a source-only native precompiled-length forward/VJP seam. It maps compact P0
  material plus \([J,R]\) lengths to \([1,J,4]\) affine-Lie node charts,
  accepts arbitrary Lie cotangents, accumulates compact/global material bars,
  and returns bounded \([J,R]\) length bars. It computes
  \(\kappa=\sum_r\rho_r\ell_r\) directly rather than reconstructing it from a
  potentially rounded or underflowed transmittance product; CPU regressions
  cover optical depths from \(10^{-18}\) through \(10^4\). A CPU bridge maps
  those length bars once per compiler node to positions, velocities, quadratic
  weights, and affine-ray bars on the certified stable stratum;
- a warm-safe equal-rank runtime adapter for that existing native ABI. Cold
  preparation validates the bounded batch payload once; warm refresh/VJP uses
  only sealed identity, layout, mutation version, and callable identity. It
  accepts caller-owned compact material and compact-gradient buffers, can
  return the bounded `[J,W]` length bar for full geometry reverse, or can use a
  separate material-only ABI that allocates no length bar; both can omit any
  global material bar. This is fake-native/CPU-tested and still unbuilt on
  Metal;
- a source-only row-ragged Lie sample reducer for general track-chart rows. It
  consumes node charts \([R_b,J,4]\), selected row ids \([N]\), row-local
  weights \([N,J]\), and targets \([N,3]\), then accumulates loss and node
  cotangents in \(O(NJ)\) without a row-by-global-time table or mandatory
  prediction tensor;
- a CPU/source paper-to-kinetic sample bridge that dispatches arbitrary
  view/frame/pixel observations exactly and right-continuously, partitions
  them by true-`J` native block, and emits bounded row ids, row-local weights,
  targets, and the one global loss scale. The streamed request path does not
  rehash world tensors or read accelerator target scalars. It also exposes the
  heterogeneous-block seam explicitly: several native compact site unions may
  contribute to one coordinator spatial request;
- an independently tested CPU/source step-scoped target-frame cache that binds
  exact provider/source generations, owns contiguous float32 frame clones,
  decodes each requested `(view,frame)` at most once across spatial bundles,
  and fails before source access when one more frame would exceed an explicit
  resident-byte budget. It has no eviction or unbounded fallback, clears on
  close, and reports exact cache-owned logical bytes; allocator/Python peaks
  remain unmeasured. This cache is not yet integrated into a verified
  production native trainer;
- a CPU/source union-local assembler that cold-seals the sorted site union and
  compact-to-union maps for those heterogeneous native blocks, uses one
  caller-owned `[S_union,4]` request bar, accepts each expected compact-only VJP
  exactly once, rejects missing/duplicate/foreign work, and seals exactly one
  compact coordinator result without a per-request global `[S,4]` bar;
- a CPU/fake-native block-major material step that makes the spatial bundle,
  not the temporal request, the lifetime of compiled node state. It streams
  all `K` chunks into the same node cotangents, invokes one material-only
  ordered-word VJP per active native block, scatters one union bar, and releases
  spatial bundles sequentially. A direct-autograd oracle proves `K` partition
  invariance, an `F=5/41` gate proves frame-density-invariant compiled work and
  runtime bytes, and a two-bundle gate proves max-bundle live node state. The
  native shaders expose the needed accumulate-only/material-only ABIs, but
  actual native launch frequency and allocator peaks remain unmeasured;
- a CPU paper-batch adapter that groups arbitrary sampled observations by view,
  preserves one global loss denominator and original batch order, and stages
  targets one frame at a time without constructing a view/time Cartesian
  product, plus an outer coordinator that accepts compact view/block bars,
  index-adds repeated source IDs into one caller-owned global `[S,4]` bar,
  proves exact disjoint coverage with bounded cursors, and issues exactly one
  optimizer authorization only after the global `P*B*3` denominator is fully
  covered;
- an exact-rational directional geometry trust certificate for one strict
  event-free chart. It proves ray noncollapse, active denominator signs,
  positive ordered cuts, and all-site endpoint owner gaps continuously over
  time with Bernstein bounds. It has `O(RS)` predicate count, no frame axis,
  and deliberately returns zero for active-event, endpoint-event, or
  multichart programs because old numeric event roots generically move under
  every nonzero update; and
- a separate restricted multichart simple-root certificate. It reconstructs
  rooted and rootless topology, analytic, and noncollapse predicates from all
  base owner words; isolates separated singleton roots; uses exact tensor-
  Bernstein boxes to prove one monotone root graph per tube and no roots in
  complements; re-isolates the rounded candidate endpoint; and exactly
  reclassifies left/right owner words. It rejects root births in complements,
  repeated/shared/persistent-zero/endpoint roots, ray collapse, and ambiguous
  semantics. Its current API rebuilds the whole registry and proof, implements
  no warm affected-payload patch, and differentiates no event time; and
- explicit host-memory and exact-versus-compiled route accounting.

The separate matched-24-byte material gate found that M3 fits its positive-P2
generating family at \(5.26\times10^{-17}\) heldout loss while M5 gives
\(8.80\times10^{-5}\); on the convex-log-P2 family, M5 gives
\(6.19\times10^{-15}\) while M3 gives \(1.33\times10^{-3}\). A complete
constant-color chord identifies only total optical depth, so material shape
requires shared partial chords or richer observations. This is why the P0
systems oracle may proceed while universal rich-material selection remains
closed.

The implementation also found a critical negative result: a low-rank temporal
atlas can fit the primal transfer almost exactly while badly approximating a
dormant material or geometry tangent. Forward error alone cannot select \(J\).
The compiler must certify or test both \(G_p(t)\) and the required sparse
JVP/VJP actions; it must not construct a dense \(D_\theta G_p(t)\).

For any predicate claimed exact, distinguish arithmetic exactness on binary64
inputs from geometric correctness under calibration/parameter uncertainty.
Attach a conditioning or perturbation margin showing how much each camera or
site coefficient may move before the certified sign/order can change.

Still open:

- rebuilt native Metal parity and real allocator measurements;
- a production dataset-bound world/topology initializer;
- unified-runner training, evaluation, checkpoint, and artifact integration;
- general projective/rational camera charts and log-depth gauge parity;
- native exact dispatch at irrational algebraic event times;
- a sharper output-sensitive replacement for the current
  \(O(U S R_{\max})+O(W(S\log S+S R_{\max}))\) active closure, and a separately
  certified neighbor supergraph if an \(O(\delta R)\) claim is desired;
- an output-sensitive batched per-pixel compiler/lowerer; the landed lowerer is
  an exact single-ray compiler plus a bounded packer of already compiled rows,
  not production image-wide kinetic compilation;
- rebuilt real-Metal parity for precompiled-length forward/VJP and the ragged
  sample reducer, replacement of the CPU/fake-native step executors with those
  native launches, and native session/trainer integration;
- supported persistent, simultaneous, and full-fiber-tie semantics beyond the
  current explicit fail-closed policy;
- event-time and topology-change derivatives;
- independent audit and differential stress of the landed restricted
  multichart singleton-simple-root certificate, then sealed affected-source
  incidence plus output-sensitive native chart-payload repair; full
  recompilation remains mandatory outside that stratum; and
- public-scene fixed-duration \(F=4,\ldots,256\) timing/memory evidence.

The native tensor design has no intrinsic 32-GB requirement. For the audited
choice \(B_p=8192,K=8,J=16\), node state plus node cotangent is about 4 MiB.
One float32 target block is 0.75 MiB. The loss-only ABI allocates no prediction
block; forward media/evaluation may explicitly add a separate 0.75 MiB
prediction block. Material training no longer stages an explicit
\([B_p,K,6]\) ray tensor: one bounded \([B_p,1,6]\) reference row validates
the immutable affine camera coefficients exactly, after which sample blocks
carry targets only. Strict/evaluation replay retains its explicit-ray route.
Prepared native topology/sample state owns no global \([F]\) or chart-local
\([F_c]\) time clone. Each sample launch receives only its live CPU-float64
\([K]\) time block and releases it after synchronization. Host/provider paths
may still retain cheap \(O(F)\) sample identities, and the current block-first
barycentric weight construction remains \(O(N_BFJ)\); a global \(FJ\) cache is
an explicit memory/order trade, not free temporal sharing.
The material-only reverse avoids lifecycle
allocations of exactly \(16Q_b+20B_b+20S_b\) bytes for one active block by
omitting incidence Möbius bars, boundary bars, and compact geometry gradients.
Here \(Q_b\), \(B_b\), \(S_b\), and \(W_b\) are that block's incidence,
boundary, compact-site, and active-word-run counts, respectively.

After that omission, the audited dominant source-tensor payload lower bounds
for the current material path are

\[
M_{\mathrm{sample},b}
=76S_b+112B_p+40B_b+20Q_b+12W_b
+32B_pJ+12B_pK+4KJ,
\]

\[
M_{\mathrm{finalize},b}
=76S_b+64B_p+40B_b+20Q_b+12W_b.
\]

They omit allocator reservations/metadata, Python objects, optimizer state,
and any native transient not visible in the source tensor ledger; they are not
upper bounds.

Its topology-token cache can remove repeat preparation, but retention is an
explicit LRU policy tradeoff rather than one token per spatial block. Entry,
cached-byte, and cached-plus-live-token limits are enforced before allocation
and checked against actual token bytes. A zero-entry policy minimizes residency;
a larger bounded policy amortizes preparation. These are source tensor-payload
and lifecycle counts, not allocator measurements or a rebuilt-Metal result.
The catastrophic memory estimate came from a dense interval-dual CPU proof
oracle whose state is quadratic in a large global derivative dimension. That
oracle is intentionally restricted to tiny fixtures and must not shape the
production algorithm.

### 9. Required Lower Bounds And Counterexamples

Any positive theorem must coexist with these obstructions. Prove, sharpen, or
correct them.

#### 9.1 Output lower bound

Producing or comparing \(PF\) colors requires \(\Omega(PF)\) work. Streaming
can bound residency but not erase I/O.

#### 9.2 Line-stabbing lower bound

Pointwise overlap one does not bound ray depth. Arrange \(S\) disjoint cells
sequentially along one ray and obtain \(R=\Theta(S)\). A partition alone does
not guarantee a shallow word.

#### 9.3 Event-output lower bound

An exact compiler must spend at least \(\Omega(R+E)\) to emit its positive-
length owner intervals and genuine topology events. Unrestricted moving-order
families can have superlinear event complexity. Derive the sharp bound for the
special kinetic lower envelope of affine lines instead of assuming it is small.

#### 9.4 No universal fixed linear/polynomial temporal atlas

At a single \(t\), total transfer is four-dimensional. Over time, arbitrary
word depth, Möbius poles, nearby topology events, high material contrast, and
motion frequency can produce unbounded complexity for a fixed linear or
fixed-degree polynomial approximation space. This does **not** by itself rule
out every finite-dimensional nonlinear parameterization: for example,
\(\exp[-\rho(1+t)]\) is nonlinear in one scalar \(\rho\) despite spanning an
unbounded linear family as \(\rho\) varies. Define the allowed representation
and evaluation model before claiming a universal nonlinear-state lower bound.
For the actual linear/Chebyshev atlas, give a Kolmogorov-width-style bound or
an explicit adversarial family under stated parameter ranges and tolerance.

#### 9.5 Primal rank need not equal tangent rank

Construct and analyze a family in which \(G(t)\) has a tiny accurate basis but
some world-parameter derivative \(D_\theta G(t)\) does not. Explain what local
operator norm, Fréchet derivative, randomized tangent probe, or deterministic
certificate is sufficient without building a dense global dual tensor.

#### 9.6 Topology differentiability obstruction

Classical derivatives exist inside a fixed combinatorial stratum. At a zero-run
birth/death, forward transfer may remain continuous while its derivative jumps.
A full-fiber differently colored tie may be discontinuous. Exposure-integrated
losses may acquire boundary terms. Classify the appropriate one-sided, Clarke,
distributional, shape-derivative, or explicitly nondifferentiable semantics.

#### 9.7 Training invalidation obstruction

If nearly every geometry update invalidates most certificates, frame-density-
independent evaluation may still hold but amortized training does not. Derive a
trust-region or local-repair condition, or state an inference-only theorem.

### 10. Research Branches To Adjudicate

For every branch, assign exactly one status:

```text
implement_now
diagnostic_only
baseline_only
defer
kill
```

Do not end by combining every branch.

#### B1. Kinetic lower envelope / convex hull

At each \(t\), dualize \(\ell_i(t,z)=a_i(t)z+b_i(t)\) to a moving planar point
or line. Hull edges encode adjacent owners; triple concurrency encodes run
birth/death; endpoint tournaments encode near/far ownership. The fixed-4D
special case has affine \(a_i,b_i\) and quadratic concurrence. The implemented
direct affine kinetic 3D frontend permits degree-\(\leq2\) coefficients and
quartic concurrence.

Determine:

- whether the proved generic event predicate set for quadratic coefficients is
  complete under the stated exclusions, or provide a minimal counterexample;
- whether the implemented unique-word/root-complement active closure can be
  sharpened beyond its current \(U/W\)-dependent bound without losing its
  exhaustive/oracle agreement and fail-closed degeneracy policy;
- the output-sensitive event bound for the direct affine kinetic family;
- exact handling of equal slopes, simultaneous roots, persistent ties, and
  symbolic perturbation;
- whether a kinetic data structure is better than repeated midpoint discovery;
- whether the relevant event set can be restricted to currently active hull
  certificates; and
- how local updates behave when sites or weights move during optimization.

This is the leading structural branch.

#### B2. Global 4D regular triangulation versus track-local compilation

A global 4D power complex may prune candidate neighbors, but its combinatorial
size can be quadratic in \(S\), and its fixed shared-metric world is only the
restricted common-translation/fixed-normal family. Compare:

1. global 4D regular-triangulation construction/traversal;
2. track-local kinetic lower envelopes;
3. a hybrid conflict/adjacency graph plus exact track-local hulls; and
4. the now-implemented continuous direct kinetic 3D reference frontend with
   explicit affine \(p_i(t)\), quadratic \(w_i(t)\), and exact half-open owner
   charts, then reduced to a production sweep and native representation.

Characterize representation equivalence, worst-case complexity, update cost,
GPU suitability, and which variant actually serves the frame-density theorem.
Do not recommend the global 4D route as the general representation unless it
overcomes the proved fixed-normal restriction.

#### B3. Exact geometry plus certified numerical transfer atlas

Keep topology/event charts exact while approximating total ordered transfer on
each chart. Derive a composable certificate for both primal transfer and the
required sparse world-parameter JVP/VJP actions. It must not allocate a global
derivative dimension squared.

Candidate tools include vector-valued Chebyshev or Bernstein approximation,
rational approximation near Möbius poles, adaptive interval splitting,
empirical interpolation, and local sparse operator-norm bounds.

#### B4. Affine semigroup / Lie / product-integral structure

The continuous generator can be written

\[
A(z)=
\begin{bmatrix}
-\sigma(z)I_3&j(z)\\
0&0
\end{bmatrix},
\]

and total transfer is a path-ordered exponential. Analyze whether the solvable
affine algebra yields stronger closure than generic matrix Magnus/BCH theory.

Investigate:

- exact exponential-polynomial or rational function classes on a stable chart;
- structure-preserving group interpolation;
- Duhamel/Fréchet formulas for world derivatives;
- commutator bounds as an adaptive exact-versus-approximate order test;
- exact P0/P1 material moments; and
- cone-preserving error bounds at tiny and very large optical depth.

Do not promote Magnus terminology unless it reduces rank, work, or certificate
cost on an explicit counterexample suite.

#### B5. Hierarchical ordered-product data structures

Associativity permits a segment tree or balanced product tree over a stable
word. Determine whether kinetic order changes and geometry updates can be
localized to \(O(\log R)\) or output-sensitive tree edits, and whether shared
prefix/suffix summaries improve the cross-time VJP without reintroducing an
\(F\times R\) tape.

Compare against the current constant-state prefix-only reverse and against
compiler-node replay. A tree that only accelerates one time sample does not
solve the temporal problem by itself.

#### B6. Joint sensor-time approximation

The current reference compiles time per track. Neighboring pixels may share
topology and transfer structure. Evaluate hierarchical \((u,v,t)\) patches,
low-rank tensors, tensor trains, visibility complexes, or wavefront/event
fronts—but require a precise degree-of-freedom restriction and a cheaper
falsification test.

This branch is optional for frame-density independence and should not block a
correct per-track temporal compiler.

#### B7. Geometry-update trust regions and generalized derivatives

Use exact predicate margins and parameter Lipschitz bounds to derive a
sufficient optimizer-step radius that preserves the structural program. When a
certificate fails, identify the minimal conflict region and local repair.

Separately classify event gradients. Do not blur:

- differentiation inside a stable stratum;
- the derivative of an event time;
- a boundary contribution to an integrated loss; and
- the derivative of a discrete topology dispatch.

#### B8. Restricted commuting or low-dimensional material families

Find the largest useful material family for which ordered composition has an
exact smaller closure. Examples may include shared source color, scalar-only
attenuation, a low-dimensional color cone, or bounded commutators.

This may define a selective fast path, not a universal replacement. It must be
compared against the current certified fallback and the M3/M5 identifiability
result.

#### B9. Canonical deformation and shared motion bases

Dynamic neural fields often use a canonical world plus deformation, a higher-
dimensional slice, or shared trajectory bases. Determine whether any of these
can reduce kinetic event/rank complexity for foam cells without smuggling in
per-frame state, losing partition validity, or making topology maintenance
harder.

Treat this as a representation restriction to prove useful, not an automatic
answer.

### 11. Concrete Theorem Requests

Attempt these in priority order. A negative theorem or sharp counterexample is
valuable.

0. **Representation-adequacy theorem audit.** Verify or correct the supplied
   exact result: fixed shared-SPD(4) cells are precisely one common translation
   of fixed anisotropic 3D sites with affine relative weights, hence constant
   candidate-face normals in a fixed world-coordinate gauge. State the allowed
   gauge group before claiming what one common time-dependent scene gauge can
   remove; at minimum distinguish rigid/similarity, affine/projective, and
   general diffeomorphic changes and which preserve the representation and
   optical measure. Verify separation for several independently rotating faces
   modulo the admissible group, and audit the
   \(\Omega(\Theta L/\varepsilon)\) lower bound for **active fixed-normal face
   pieces or temporal chart switches** in the stated one-piece-per-chart
   approximation class. Do not reinterpret it as a general linear lower bound
   on site count without proving how arbitrary polyhedral/staircase
   approximations map sites and candidate pairs to those pieces. Treat
   the shared gauge plus direct affine kinetic residual frontend as the selected
   general candidate unless this theorem is refuted.

1. **Direct-kinetic owner-word theorem.** For affine camera-ray tracks,
   affine 3D site trajectories, and degree-\(\leq2\) weights, verify the proved
   generic event set: active near/far owner equalities, co-minimal finite triple
   concurrences, full-fiber ties already visible as common near/far roots, and
   separate ray-collapse events. Pair-denominator roots alone are analytic cut
   guards, not topology events. Audit the degree-\(\leq2\) pair/ray and
   degree-\(\leq4\) concurrence bounds, the exact guarded root isolator, the
   exhaustive continuous reference compiler, and the remaining
   identically-zero, repeated, simultaneous, grazing, inactive-root, and
   full-fiber production semantics. Then audit the implemented active-boundary
   closure, prove or correct its
   \(O(U S R_{\max})+O(W(S\log S+S R_{\max}))\) work accounting, and state why
   \(O(\delta R)\) needs a separately certified neighbor supergraph.

2. **Output-sensitive compiler theorem.** Give the best work/storage bound in
   \(S,R,E\) for one track and then many tracks. If near-output-sensitive
   compilation is impossible, exhibit the obstruction.

3. **4D-slice equivalence corollaries.** Starting from the supplied exact
   common-translation/affine-relative-weight characterization, determine its
   non-unique generator gauges, conditioning under spacetime rescaling, and the
   sharp approximation cost for motions outside the family. Compare its
   quadratic event specialization against the direct kinetic quartic family.

4. **Algebraic seam theorem.** Give an exact representation and comparison
   algorithm for rational and irrational event times, with half-open,
   right-continuous dispatch for dyadic query times and no rounded-seam
   ambiguity on a GPU/native boundary. State exactness relative to the supplied
   binary64 coefficients and separately certify robustness to bounded camera,
   site, weight, and calibration perturbations.

5. **Stable-chart analyticity theorem.** Under denominator, owner,
   positive-length, and ray-speed margins
   \(\|d_p(t)\|\geq\delta_d>0\), prove analyticity of total transfer and
   characterize its nearest real or complex singularities, including poles,
   zeros, and square-root branch points introduced by \(\|d_p(t)\|\).

6. **Primal-and-tangent rank theorem.** Bound the required \(J(\varepsilon)\)
   for \(G_p(t)\) and specified sparse JVP/VJP action families using word
   depth, distance to events/poles, optical-depth bounds, material contrast,
   and camera motion. State any operator norm being certified; do not represent
   a dense global \(D_\theta G_p(t)\).

7. **Sparse certificate-composition theorem.** Show how track-local transfer
   and jet bounds plus sparse incidence-to-site operator norms imply a global
   world-VJP error bound without a dense global dual construction.

8. **Frozen-program constant-state adjoint theorem.** Formalize a prefix-only
   ordered-word VJP and its product-integral extension with the structural
   program held fixed, including zero density, \(\kappa=0\), high opacity,
   repeated faces, and optional camera gradients. List every stopped structural
   dependency explicitly.

9. **Physical/event and compiled-algorithm derivative theorem.** First classify
   zero-length birth/death, endpoint crossing, triple concurrency, diagram
   flip, grazing incidence, and full-fiber tie events for the exact physical
   point-sampled and exposure-integrated objectives, including any event-time
   boundary terms. Separately analyze derivatives of adaptive compiler choices
   such as interpolation nodes/weights and rank. State which compiler terms are
   irrelevant to the exact physical objective, which are deliberately stopped,
   and which need an approximation-error or surrogate-objective argument.

10. **Structural trust-region and moving-root theorem.** Audit the landed
    exact-rational directional certificate for one strict event-free chart:
    its ray, denominator, ordered-cut, and all-site endpoint inequalities; its
    Bernstein perturbation bound; and its `O(RS)` predicate count. Then solve
    the still-open eventful case. Prove simple-root persistence and event-order
    preservation inside disjoint rational neighborhoods, prove the complement
    root-free, and give an output-sensitive re-isolation/refit and local-repair
    algorithm. Do not claim the old numeric seam stays fixed: generically
    `dt_event/dtheta` is nonzero, so frozen event endpoints have zero radius.

11. **Worst-case constructions.** Construct explicit families exhibiting:

    - \(\Theta(S)\) line stabbing with pointwise overlap one;
    - superlinear owner/order event count;
    - unbounded required rank for the declared linear/Chebyshev atlas at fixed
      tolerance, or a stronger lower bound only after defining the nonlinear
      representation model;
    - primal rank much smaller than tangent rank;
    - widespread topology invalidation; and
    - a case where 4D global topology is much larger than track-local output.

12. **Break-even theorem.** Refine the current conservative route model. For
    word work \(W\), rank \(J\), tracks \(P\), and frames \(F\), the simplified
    exact proxy is

    \[
    C_{\mathrm{exact}}=3FW.
    \]

    With the current per-spatial-block construction of verified linear
    barycentric sample weights, a simplified compiled proxy is

    \[
    C_{\mathrm{compiled}}
    =2JW+PJ^2+PFJ+N_BFJ+N_{\mathrm{fb}}J^2.
    \]

    A future validated global weight cache replaces \(N_BFJ\) by \(FJ\).

    Derive the correct hardware-aware policy including bytes moved, sparse
    incidence reuse, block sizes, validation/certificate amortization, and
    fallback probability. Here \(N_{\mathrm{fb}}\) counts exceptional sample
    row executions requiring the row-local dense solve across all blocks. A
    two-run word may correctly
    prefer exact replay; the compiler is expected to win only when word/event
    reuse justifies it.

### 12. Literature Starting Point And Precise Boundary

Read and verify primary sources. Do not cite a paper merely by keyword. This is
a seed bibliography, not an exhaustive or frozen novelty search; recheck each
scope claim and search explicitly for work published after the date of this
brief before asserting that no dynamic neural foam exists.

The literature currently supplies **separate pieces**, not the requested
composition:

| Lineage | What it actually contributes | What it does not close |
| --- | --- | --- |
| Radiant Foam | Static non-overlapping Voronoi cells, neighbor-to-neighbor ray walking, and exact P0 segment integration | Physical scene time, kinetic event charts, cross-time word reuse, or a shared world adjoint |
| Power Foam | Bounded power cells, conservative sphere-overlap/Čech candidates, oriented interfaces, and a rasterization route | A continuously moving cell complex or frame-density-independent training |
| Exact kinetic data structures / kinetic regular triangulations | Polynomial certificates, event queues, exact root ordering, and local combinatorial updates for moving weighted points | Ray-specific optical transfer, learned losses, sparse cross-time VJPs, or proof that event count is small |
| STBVH and spacetime ray tracing | Spatial-temporal bounds and temporal splitting that avoid duplicating one acceleration structure per motion segment | Ordered cell-word compression, differentiable topology maintenance, or shared parameter gradients |
| D-NeRF, K-Planes, DynMF, and related dynamic fields | Canonical deformations, factorized 4D fields, or shared motion bases that reduce persistent world bytes | Exact cell ownership/order events or moving expensive ray/world reverse work off the requested-frame axis |
| Associative alpha/volume compositing | Exact scan/tree composition and constant-state word VJPs at one fixed ordered word | Discovery, certification, approximation, and repair of the word as camera and scene time vary |

The closest reusable formulations are therefore kinetic lower envelopes/regular
triangulations for **structure**, affine transfer semigroups/product integrals
for **physics**, certified polynomial/rational approximation for **time**, and
sparse reverse-mode factorization for **training**. Treat a proposal that solves
only one row of the table as a component or baseline, not as the WorldFoam
closure.

#### Static neural foams

- [Radiant Foam: Real-Time Differentiable Ray Tracing](https://arxiv.org/abs/2502.01157)
  establishes a static 3D Voronoi radiance representation, exact ray-cell
  segment integration, and adjacency walking.
- [Power Foam: Unifying Real-Time Differentiable Ray Tracing and Rasterization](https://arxiv.org/abs/2604.24994)
  adds bounded power cells, sphere-overlap/Čech candidate adjacency, oriented
  surfaces, and a rasterization path.
- [Semantic Foam: Unifying Spatial and Semantic Scene Decomposition](https://arxiv.org/abs/2604.26262)
  adds per-cell semantic structure; it does not add physical scene time.
- [SDFoam: Signed-Distance Foam for Explicit Surface Reconstruction](https://arxiv.org/abs/2512.16706)
  adds an SDF-based static cell field and lists dynamics as future work.
- [Scalable GPU Construction of 3D Voronoi and Power Diagrams](https://research.zenseact.com/publications/paragram/)
  supplies a highly parallel static construction route and demonstrates the
  importance of geometry-build cost at large site counts.

#### Cellular/tetrahedral rendering neighbors

- [Tetra-NeRF](https://arxiv.org/abs/2304.09987) uses a static Delaunay
  tetrahedral representation for adaptive neural-field support.
- [Radiance Meshes](https://arxiv.org/abs/2512.04076) uses static Delaunay
  tetrahedra and exact fast volume rendering; its attribute backbone smooths
  over optimization-time topology flips rather than compiling scene time.
- [DiffTetVR](https://arxiv.org/abs/2601.00114) derives differentiable
  tetrahedral volume rendering and vertex gradients for a static volume.
- [Simplex space-time meshes in engineering applications with moving domains](https://arxiv.org/abs/2210.09831)
  shows that 4D pentatope meshes can represent complex moving domains. It does
  not supply an ordered optical-transfer compiler or shared rendering adjoint.

#### Kinetic geometry and spacetime acceleration

- [On Kinetic Delaunay Triangulations](https://arxiv.org/abs/1312.2194)
  demonstrates near-quadratic change counts for a restricted planar,
  unweighted, unit-speed linear-motion Delaunay problem. It is cautionary
  evidence that kinetic topology need not stay small, not a bound for this
  weighted 3D per-ray lower-envelope family.
- [Kinetic Voronoi Diagrams and Delaunay Triangulations under Polygonal Distance Functions](https://arxiv.org/abs/1404.4851)
  provides an event/certificate viewpoint and Davenport--Schinzel-style bounds
  for algebraic trajectories in a related setting.
- [A Package for Exact Kinetic Data Structures and Sweepline Algorithms](https://pmc.ncbi.nlm.nih.gov/articles/PMC3001684/)
  is directly relevant to exact polynomial certificates and root-driven
  topology maintenance, including regular triangulations.
- [CGAL kinetic 3D regular triangulation documentation](https://doc.cgal.org/4.11/Kinetic_data_structures/classCGAL_1_1Kinetic_1_1Regular__triangulation__3.html)
  is an implementation precedent for moving weighted points, not a rendering
  or differentiable-training solution.
- [STBVH: A Spatial-Temporal BVH for Efficient Multi-Segment Motion Blur](https://www.embree.org/papers/2017-HPG-msmblur.pdf)
  demonstrates practical time sharing in an acceleration structure, but not a
  learned cell partition or shared world adjoint.
- [Spacetime Ray Tracing for Animation](https://glassner.com/computer-graphics/graphics-research/spacetime-ray-tracing/)
  and [An Efficient Spatio-Temporal Architecture for Animation Rendering](https://dcgi.fel.cvut.cz/home/havran/ARTICLES/havran03egsr.pdf)
  are prior art for lifting animation into spacetime and reusing work across
  frames. Position the new claim beyond, not in ignorance of, this lineage.

#### Dynamic neural representations

- [D-NeRF](https://arxiv.org/abs/2011.13961) uses a canonical field plus a
  time-conditioned deformation.
- [HyperNeRF](https://arxiv.org/abs/2106.13228) uses higher-dimensional slices
  to accommodate topology-varying observations.
- [DynMF](https://arxiv.org/abs/2312.00112) shares a small basis of temporal
  trajectories across dynamic primitives.
- [Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting](https://arxiv.org/abs/2310.10642)
  conditions a 4D Gaussian at time using the familiar Schur complement, then
  renders sampled times; it does not retain a foam cell word.

#### Visibility derivatives

- [A Simple Approach to Differentiable Rendering of SDFs](https://arxiv.org/abs/2405.08733)
  explicitly treats visibility derivatives as boundary-integral phenomena.
  Use this and primary differentiable-rendering work to distinguish stable-
  stratum derivatives from event/boundary terms.

The plausible novelty target is the combination:

> exact or certified kinetic cell/ray topology over a camera program, retained
> ordered emission--absorption transfer, a frame-density-independent compiled
> atlas, and a reusable cross-time sparse world adjoint.

Associativity of alpha compositing, power diagrams, 4D cells, kinetic data
structures, or temporal interpolation alone are not novel claims.

### 13. Repository Attachments And Code Anchors

For the first-pass moving-root task, read **only these required core
attachments** before proposing a theorem or replacement:

- `agent_notes/loose_notes/2026-08-03_16-35-33_kinetic_power_word_event_sufficiency_red_team.md`
- `agent_notes/loose_notes/2026-08-03_20-40-32_worldfoam_multichart_simple_root_reisolation.md`
- `research_experiments/world_foam_lane2/kinetic_simple_root_reisolation.py`
- `research_experiments/world_foam_lane2/test_kinetic_simple_root_reisolation.py`
- `research_experiments/world_foam_lane2/kinetic_power_word_compiler.py`
- `research_experiments/world_foam_lane2/test_kinetic_power_word_compiler.py`
- `research_experiments/world_foam_lane2/kinetic_active_owner_chart_compiler.py`
- `research_experiments/world_foam_lane2/test_kinetic_active_owner_chart_compiler.py`
- `research_experiments/world_foam_lane2/rational_polynomial_roots.py`
- `research_experiments/world_foam_lane2/test_rational_polynomial_roots.py`
- `research_experiments/world_foam_lane2/kinetic_geometry_trust_region.py`
- `research_experiments/world_foam_lane2/test_kinetic_geometry_trust_region.py`

Everything below is an **optional evidence appendix**. Consult a file only when
the core task exposes a concrete dependency or contradiction. Do not read the
whole appendix before answering the first-pass work order.

Optional background and status documents:

- `research_notes/meta_review_jul_28th.md`
- `research_notes/world_tubes_spd4_worldfoam_handoff_2026-07-28.md`
- `research_notes/worldfoam_paper/WORLD_FOAM_PAPER_DRAFT.md`
- `research_notes/worldfoam_paper/WORLD_FOAM_MATH_APPENDIX.md`
- `research_notes/worldfoam_paper/GAUSSIAN_FINITE_ELEMENT_WORLD_FOAM.md`
- `research_notes/worldfoam_paper/WORLD_FOAM_MEMORY_LIGHT_THEOREM_LEDGER_2026-08-03.md`
- `research_notes/worldfoam_paper/WORLD_FOAM_NATIVE_MEMORY_SOURCE_AUDIT_2026-08-03.md`
- `artifacts/foundation_gates/worldfoam_material_value_fit_cpu_20260727.json`
- `TODO/worldfoam_memory_light_native4d.md`
- `agent_notes/loose_notes/2026-08-03_03-35-19_worldfoam_memory_light_shared_adjoint.md`
- `agent_notes/loose_notes/2026-08-03_15-24-30_worldfoam_kinetic3d_memory_state_sync.md`
- `agent_notes/loose_notes/2026-08-03_17-58-19_worldfoam_production_kinetic_compiler_and_bounded_native_time_state.md`
- `agent_notes/loose_notes/2026-08-03_19-09-46_worldfoam_node_length_native_seam_and_mathematician_handoff.md`
- `agent_notes/loose_notes/2026-08-03_19-24-32_worldfoam_geometry_update_trust_region.md`

Optional executable references:

- `research_experiments/world_foam_lane2/sparse_power_word_compiler.py`
- `research_experiments/world_foam_lane2/kinetic_owner_chart_compiler.py`
- `research_experiments/world_foam_lane2/test_kinetic_owner_chart_compiler.py`
- `research_experiments/world_foam_lane2/kinetic_owner_chart_oracle.py`
- `research_experiments/world_foam_lane2/test_kinetic_owner_chart_oracle.py`
- `research_experiments/world_foam_lane2/kinetic_chart_transfer_bridge.py`
- `research_experiments/world_foam_lane2/test_kinetic_chart_transfer_bridge.py`
- `research_experiments/world_foam_lane2/kinetic_multichart_transfer_program.py`
- `research_experiments/world_foam_lane2/test_kinetic_multichart_transfer_program.py`
- `research_experiments/world_foam_lane2/kinetic_continuous_transfer_acceptance.py`
- `research_experiments/world_foam_lane2/test_kinetic_continuous_transfer_acceptance.py`
- `research_experiments/world_foam_lane2/kinetic_stable_stratum_vjp.py`
- `research_experiments/world_foam_lane2/test_kinetic_stable_stratum_vjp.py`
- `research_experiments/world_foam_lane2/kinetic_multichart_stable_stratum_vjp.py`
- `research_experiments/world_foam_lane2/test_kinetic_multichart_stable_stratum_vjp.py`
- `research_experiments/world_foam_lane2/kinetic_native_topology_lowering.py`
- `research_experiments/world_foam_lane2/test_kinetic_native_topology_lowering.py`
- `research_experiments/world_foam_lane2/kinetic_native_precompiled_length_oracle.py`
- `research_experiments/world_foam_lane2/test_kinetic_native_precompiled_length_oracle.py`
- `research_experiments/world_foam_lane2/kinetic_native_precompiled_length_adapter.py`
- `research_experiments/world_foam_lane2/test_kinetic_native_precompiled_length_adapter.py`
- `research_experiments/world_foam_lane2/kinetic_native_equal_rank_lowering.py`
- `research_experiments/world_foam_lane2/test_kinetic_native_equal_rank_lowering.py`
- `research_experiments/world_foam_lane2/kinetic_native_equal_rank_runtime_adapter.py`
- `research_experiments/world_foam_lane2/test_kinetic_native_equal_rank_runtime_adapter.py`
- `research_experiments/world_foam_lane2/kinetic_ragged_paper_step_cpu_fake_native.py`
- `research_experiments/world_foam_lane2/test_kinetic_ragged_paper_step_cpu_fake_native.py`
- `research_experiments/world_foam_lane2/test_kinetic_ragged_lie_sample_source_contract.py`
- `src/train/paper_ragged_track_staging.py`
- `tests/test_paper_ragged_track_staging.py`
- `src/train/paper_kinetic_ragged_sample_plan.py`
- `tests/test_paper_kinetic_ragged_sample_plan.py`
- `src/train/paper_kinetic_union_local_bar_assembly.py`
- `tests/test_paper_kinetic_union_local_bar_assembly.py`
- `src/train/paper_ragged_material_bar_coordinator.py`
- `tests/test_paper_ragged_material_bar_coordinator.py`
- `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/csrc/metal/world_foam_lane2_shared_replay_tensor.metal`
- `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/csrc/metal/world_foam_lane2_metal.mm`
- `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/csrc/bindings.cpp`
- `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/ops.py`
- `research_experiments/world_foam_lane2/power_topology_event_predicates.py`
- `research_experiments/world_foam_lane2/compiled_transfer_adjoint.py`
- `research_experiments/world_foam_lane2/compiled_lie_world_adjoint.py`
- `research_experiments/world_foam_lane2/compact_lie_schedule.py`
- `research_experiments/world_foam_lane2/continuous_lie_jet_certificate.py`
- `research_experiments/world_foam_lane2/continuous_owner_identity_certificate.py`
- `research_experiments/world_foam_lane2/native_piecewise_topology_adapter.py`
- `research_experiments/world_foam_lane2/material_training_step.py`
- `research_experiments/world_foam_lane2/host_memory_contract.py`
- `research_experiments/world_foam_lane2/compiled_route_cost_gate.py`

### 14. Required Output Format

For the **first pass**, independently audit the landed restricted certificate
and finish the missing output-sensitive-maintenance decision. Return Markdown
in this exact section order. Do not produce the broader B1--B9 survey, full
literature review, material-basis analysis, or paper benchmark plan unless it
is needed to refute the selected moving-root theorem.

#### 1. Executive Decision

State in at most 250 words:

- whether the landed certificate is sound or the smallest exact counterexample;
- whether simple event roots and their order persist under a bounded update;
- the load-bearing assumptions;
- whether local re-isolation can beat complete recompilation; and
- the cheapest counterexample or diagnostic that would kill the route.

#### 2. Evidence Versus Inference

Separate:

- repository-implemented facts;
- literature-established facts;
- deductions you prove;
- plausible conjectures; and
- unsupported claims you reject.

#### 3. Canonical Predicate Registry

Define the exact base/update domains and enumerate the complete predicate
registry required to rule out missing, disappearing, merged, or reordered
events. Use the directional homotopy above unless you explicitly prove a
stronger quantified domain. Use three typed classes: (i) topology-event
candidates, (ii) root-bearing analytic/representation guards such as
pair-denominator roots, and (iii) non-root validity/positivity/noncollapse
guards. Group coincident roots by exact algebraic equality. Also separate an
algebraic root from a semantically active event: for each continued root
specify the co-minimality/activity test and certified left/right owner-word
classification. State when a class-(ii) root forces only a cut/refit chart and
must not increment semantic topology-event count \(E\). For each predicate or
group give its polynomial degree, degeneracy semantics, and whether the current
compiler retains enough provenance to reconstruct it. If the registry is
incomplete, stop and exhibit the minimal missing source.

#### 4. Theorem Ledger

Use a table with columns:

```text
claim | status | assumptions | proof/counterexample | implementation consequence
```

Allowed status values:

```text
proved
proof_sketch
conjecture
refuted
known_prior_art
```

#### 5. Persistence Or Counterexample

Prove the theorem under explicit assumptions, or give the smallest exact
counterexample. A positive result must establish:

- one and only one continued event root inside every supported simple root-
  group neighborhood;
- no roots on every complementary interval;
- continued or re-isolated analytic-guard roots kept distinct from topology
  events, including any representation-only chart splits they require;
- uniform nonvanishing margins for every non-root guard, including ray speed;
- preserved ordering and separation of continued root groups;
- exact reclassification of every continued root as active, inactive, or
  ambiguous from co-minimality and certified left/right owner words, rather
  than treating root continuation itself as topology persistence;
- a valid owner-word witness between every consecutive root-group pair; and
- a declared policy for endpoint, repeated, simultaneous, grazing, and
  identically-zero predicates.

#### 6. Re-isolation And Local-Repair Algorithm

Give executable pseudocode and honest work/storage bounds in terms of sites,
active word depth, predicate sources, events, affected tracks, exact-arithmetic
bit complexity, and update radius. State exactly what can be reused, what must
be recomputed, and when the algorithm falls back to full compilation. Include
the activity/co-minimality reclassification or certified activity margins at
continued roots and the host/native provenance boundary, but do not design a
new renderer.

#### 7. Falsification Suite

Give small exact fixtures for: one rational moving root, one irrational moving
root, two nearly colliding roots, two predicates sharing one algebraic root, a
new root born in the complement, a repeated or grazing root, a ray-collapse
guard, an endpoint event, an inactive third-site undercut, and a step just
inside/outside the certified radius. For each, state the invariant and kill
threshold.

#### 8. One Next Implementation And One Kill Diagnostic

Choose exactly one implementation slice. Name the repository inputs and output
certificate it should consume/produce. Then choose exactly one cheap diagnostic
whose failure means we should stop pursuing local event maintenance and use
full recompilation. Do not recommend publication-scale training in this pass.

### 15. Quality Bar And Forbidden Moves

Your response fails if it:

- proposes another Gaussian Schur complement without preserving order;
- says "4D" as if dimensional lifting alone gives temporal reuse;
- treats amortized constant cost per cell transition as constant total ray cost;
- ignores \(\Omega(PF)\) output and target work;
- certifies only the primal transfer and ignores required sparse derivative
  actions;
- assumes topology is fixed while claiming dynamic geometry training;
- rounds algebraic event times without a fail-closed correctness argument;
- hides a dense global derivative tensor inside a certificate;
- invokes gauge, holonomy, Magnus, tensors, or category language without an
  operational map and a falsification test;
- proposes a universal material basis despite the complementary M3/M5 result;
- conflates a static foam's training-time flips with physical scene time;
- ends with an unranked combination of every branch; or
- recommends publication-scale training before the mathematical route gate.

The ideal first-pass answer is simple but deep: one adversarial audit of the
landed complete registry and persistence certificate, one minimal counterexample
or strengthened theorem, one genuinely output-sensitive re-isolation/repair
algorithm, one fail-closed fallback, and one diagnostic that decides whether
local maintenance is worth implementing.

## END RESEARCH PROMPT

---

## Maintainer Interpretation

This prompt deliberately directs new theory at the remaining gap rather than
reopening solved pieces. The representation-adequacy decision is now narrow and
explicit: in a fixed coordinate gauge the fixed shared-SPD(4) world is an exact
restricted common-translation/fixed-normal special case. One shared
camera/scene gauge remains valuable for bulk motion only when its transformed
trajectory class and optical-measure law are explicit and certified; direct
affine kinetic 3D residual sites are the selected general frontend candidate.
The latter is green for exact fixed-time words, degree-bounded event-polynomial
generation, guarded exact quartic predicate isolation, an exhaustive
continuous \(O(S^3)\) owner-chart compiler, an independent adversarial oracle,
and a differential-tested active-owner closure at CPU scope. That closure is
not a proved flat \(O(SR)\) sweep: it reports
\(O(U S R_{\max})+O(W(S\log S+S R_{\max}))\). Exact multi-chart dispatch,
continuous primal/material-action certification, streamed node-cotangent
reduction, and a stable-stratum sparse VJP for geometry, rays, weights, and P0
materials are also green on CPU. A provenance-sealed single-ray lowering,
independent CPU Lie oracle, node-length-to-geometry VJP, source-only native
Lie-node forward/VJP, and row-ragged source reducer now implement the
frame-independent seam. A bounded equal-rank batch packer and an outer
multi-view one-global-bar coordinator now close two host-side production
contracts without introducing a global time refinement. A block-major
CPU/fake-native paper step closes their invocation lifecycle at integration-
proof scope: each active compiled block runs one forward and material-only VJP
across all temporal chunks, with no `[J,W]` bar. An exact-rational event-free
trust certificate supplies a sound nonzero directional reuse radius. A
separate restricted multichart reference now proves whole-segment persistence
and endpoint re-isolation for separated singleton simple roots after rebuilding
the complete rooted/rootless registry. The extension is still unbuilt and
runtime-unverified; production image-wide kinetic compilation, output-
sensitive affected-payload repair, and the total derivative of recompilation
remain unimplemented.
The leading current hypothesis is:

```text
direct affine kinetic 3D sites with frame-independent parameters
+ active-owner kinetic compiler with explicit U/W closure accounting
+ exact ordered transfer at adaptive J nodes
+ CSR owner words plus physical [J,R] node lengths
+ affine-transfer Lie atlas
+ separate primal and sparse-tangent certification
+ row-ragged streamed residual-to-node reduction
+ one frozen-program node-level word/world VJP
```

The leading alternative is an exact-geometry/certified-transfer hybrid using a
different structure-preserving approximation. The fixed 4D family remains an
oracle and useful restricted route, not the default general-motion answer. A
full global 4D mesher should advance only if it beats track-local
output-sensitive compilation on measured event count, bytes, and rebuild cost.
The native ABI is still unbuilt and unmeasured. The source now accepts the
lowered precompiled-length chart and ragged row-local sample blocks, and the
CPU outer coordinator merges compact view/block bars into one update. A
warm-safe equal-rank node adapter and a bounded paper-to-native sample plan now
exist at CPU/source scope. The CPU/fake-native block-major session and union-
local compact-bar assembly are now present. Production batched image-wide
compilation, replacement by actual ragged native launches, rebuilt runtime
parity, and the unified-runner lane are still absent.
Recompiling after a geometry perturbation is a different derivative from the
landed frozen-program VJP. The event-free case has a conservative exact trust
radius; the restricted separated-singleton multichart case has a whole-
direction re-isolation certificate. All other eventful cases still require a
full recompile or a generalized-event treatment, and none of these certificates
adds event-time derivatives.

This work is a WorldFoam second-paper/future lane. It must not delay the
publication-scale World Tubes experiments and paper packaging already identified
as the primary project goal.
