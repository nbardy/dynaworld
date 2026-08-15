# WorldFoam minimality, kinetic-event, and repair-math intake

Date: 2026-08-05 KST

## Question

Audit the external mathematical handoff in:

```text
/Users/nicholasbardy/.codex/attachments/
690cab9b-e1e8-484e-be04-1962444f01be/pasted-text.txt
```

Decide which claims are new and correct, which are already present, which need
repair, and whether any of them should change the memory-light WorldFoam
runtime direction.

This pass was deliberately source- and document-only. The host remained under
severe memory pressure, so no Python import, test, build, Metal/MPS/CUDA launch,
dataset decode, or training run was attempted.

## Executive conclusion

The handoff helps, but it does **not** justify a sixth formulation or a new
shader family.

The strongest additions are:

1. a scoped proof that `(beta,m)` is the exact contextual quotient and that
   RGB P0 transfer needs at least four smooth real coordinates generically;
2. a generic per-track semantic-event output bound
   `O(S^2 beta_6(S))` for the actual direct-kinetic moving-line family; and
3. a corrected distinction between event-local structural repair and the
   changed-entry cost of refreshing a materialized `[J,R]` numeric payload.

The first-order jet is correct but already implicit in the repository's dual
arithmetic and explicit transfer VJP. The proposed per-run opacity sensitivity
is useful, but its advertised general training stop rule omits prefix and
background tangent terms; the existing translated-measure tail theorem is
safer. Discriminants and resultants are useful as an independent exact CPU
oracle, not as a replacement for the stronger constructive Bernstein proof or
for full recompilation. Commutation is already covered. Same-color coalescing
is only a forward equivalence unless color parameters are genuinely tied and
all original provenance is retained.

The engineering direction is therefore unchanged: finish and validate the
fused fixed-camera full-geometry route, keep full structural recompilation
after geometry/ray updates, and treat local repair and tail skipping as later
optimizations gated by measurement and independent certificates.

## 1. Four-scalar contextual quotient and smooth minimality

### Exact statement

For a `C`-channel finite P0 word `W`, write its action on a rear radiance as

```text
Phi_W(q) = m_W + beta_W q.
```

Define contextual equivalence by equality for every admissible rear radiance
`q`. Then

```text
W ~ W'
iff
(beta_W,m_W) = (beta_W',m_W').
```

The forward direction follows by evaluating at `q=0` to identify `m`, then at
one nonzero `q` to identify the scalar `beta`. The reverse direction is
immediate, and composition shows that the equivalence persists inside every
front and rear word context.

For the open physical interior

```text
O_C = {
  0 < beta < 1,
  0 < m_j < 1-beta for every channel j
},
```

every point is realized by one segment:

```text
tau = -log(beta),
c = m / (1-beta).
```

If `C^1` maps `E:O_C -> R^d` and `D` on a neighborhood of `E(O_C)` satisfy
`D(E(beta,m))=(beta,m)`, differentiating gives a rank-`C+1` identity. Hence

```text
d >= C+1.
```

RGB therefore needs at least four smooth real coordinates on the generic
physical interior.

### Scope and non-claims

- Arbitrary rear contexts are load-bearing. With only one fixed black
  background, `beta` is not separately observable from the emitted moment.
- "Unique quotient" means unique up to isomorphism, not unique coordinates.
- The lower bound assumes a smooth, robust exact encoder/decoder. It does not
  rule out discontinuous bit packing, quantization, priors on a lower-dimensional
  data manifold, or approximation.
- It does not prove that four float32 values are bit-minimal.
- It does not lower-bound the owner word, charts, temporal rank `J`, geometry,
  or complete executor memory.
- It is scoped to scalar extinction and the declared P0 `C`-channel action.
- External-literature novelty has not been established. Present it as a useful
  minimality proposition, not a novelty headline.

### Decision

Adopt in the theorem ledger and paper. It validates the existing four-scalar
native ABI and strengthens the proof spine without changing runtime state.

### Falsification/oracle plan

- one-segment realizability over random interior points;
- rank/Jacobian witness on the one-segment parameterization;
- contexts that distinguish each perturbed transfer coordinate;
- a negative fixed-black-only fixture demonstrating why contextual scope is
  necessary.

No runtime oracle is required for the algebraic proof.

## 2. Exact first-order transfer jet

For `g_i=(beta_i,m_i)` and a selected direction
`dot(g_i)=(dot(beta_i),dot(m_i))`, differentiating

```text
g_1 star g_2
  = (beta_1 beta_2, m_1 + beta_1 m_2)
```

gives

```text
dot(beta)
  = dot(beta_1) beta_2 + beta_1 dot(beta_2),

dot(m)
  = dot(m_1) + dot(beta_1) m_2 + beta_1 dot(m_2).
```

The tangent lift is associative because it is the derivative of an
associative product. For `k` fixed directions, its pointwise state has
`(C+1)(1+k)` scalars.

This is correct, but it is not a new runtime closure:

- `continuous_lie_jet_certificate.py` already propagates value and tangent
  tuples with dual arithmetic;
- the paper appendix and compiled transfer adjoint already contain the exact
  cotangent pullback;
- it does not bound `k`, the temporal transfer rank, or an all-parameter
  Jacobian;
- storing it in the native executor would usually increase state relative to
  streamed cotangents followed by one word reverse.

### Decision

Use at most as an appendix bridge between the existing JVP certificate and
VJP. Do not call it a new renderer or make it a contribution headline.

## 3. Opaque-tail specialization

Let

```text
C_r = (1-beta_r)c_r + beta_r C_(r+1),
T_0 = 1,
T_r = product_(q<=r) beta_q,
C_(R+1) = b.
```

On one fixed word, the exact differential is

```text
dC = T_R db
   + sum_r [
       T_r (c_r-C_(r+1)) d(tau_r)
       + T_(r-1)(1-beta_r) d(c_r)
     ].
```

For colors/background in a convex radiance set of infinity-diameter `D`,

```text
||partial C / partial tau_r||_inf <= D T_r,
||D_(c_r) C||_(inf <- inf) = T_(r-1)(1-beta_r),
|bar(tau_r)| <= D T_r ||bar(C)||_1.
```

For `tau_r=rho_r L_r`, the density and length bounds multiply by `L_r` and
`rho_r`, respectively. This is a useful sharp per-run diagnostic.

### Defect in the supplied training-stop claim

For prefix `A` and removed rear word `B`,

```text
Delta = C_full-C_prefix = T_A(C_B-b),

dDelta = T_A[-d(kappa_A)(C_B-b) + (dC_B-db)].
```

The handoff's tail-only sum omits both the prefix-transmittance derivative and
the background tangent. A counterexample holds every tail parameter fixed and
varies one prefix optical depth: the proposed right-hand side is zero while
`dDelta` is generally nonzero. A varying background is a second counterexample.

A corrected fixed-split cube/diameter bound is

```text
||dDelta||_inf <= T_A [
    D alpha_B |d(kappa_A)|
  + D sum_(r in B) T_r^B |d(tau_r)|
  + sum_(r in B) T_(r-1)^B alpha_r ||d(c_r)||_inf
  + alpha_B ||db||_inf
].
```

It still requires a uniform admitted parameter norm, fixed topology/order and
split, and loss-cotangent bounds before it becomes a training-gradient
guarantee. It also needs a certificate obtainable without traversing the
suffix one hoped to skip.

### Decision

Keep the existing translated-measure T0b theorem as canonical. Optionally add
the per-run specialization later, but do not enable runtime tail termination.

## 4. Generic direct-kinetic semantic-event bound

### Mapping to the actual compiler family

On one coefficient chart, the direct kinetic frontend reduces each site to a
moving line in depth:

```text
ell_i(t,z) = a_i(t) z + b_i(t),
deg(a_i), deg(b_i) <= 2.
```

The exact implementation is in
`research_experiments/world_foam_lane2/kinetic_power_word_compiler.py`.
For a triple, concurrence is the determinant

```text
C_ijk(t)
  = (b_i-b_j)(a_j-a_k)
  - (b_j-b_k)(a_i-a_j),
```

of degree at most four. Outside an identically-zero or other unsupported
degeneracy, each triple can therefore become concurrent at most four times.

Alexandron, Kaplan, and Sharir's moving-line envelope argument bounds external
upper/lower-envelope changes by

```text
O(S^2 beta_(s+2)(S)).
```

Using `s<=4` gives

```text
E_interior = O(S^2 beta_6(S)).
```

At each fixed clipped endpoint `z=z_-` or `z=z_+`, the competing values are
total quadratics in `t`. Each pair crosses at most twice, so the endpoint
lower-envelope sequence has Davenport--Schinzel order two and at most `2S-1`
pieces. Thus

```text
E_clipped = O(S^2 beta_6(S)+S)
          = O(S^2 beta_6(S)).
```

Standalone denominator guards have degree at most two per pair. If proof-only
analytic guards are counted, their `O(S^2)` total is absorbed; they are not
semantic positive-length owner-word changes.

Primary source:

- Giora Alexandron, Haim Kaplan, and Micha Sharir, *Kinetic and Dynamic Data
  Structures for Convex Hulls and Upper Envelopes*, Computational Geometry
  36(2), 2007, especially pp. 8--9:
  https://www.cs.tau.ac.il/~haimk/papers/cgta.pdf

The 2024 Agarwal--Ezra--Sharir bivariate-surface result supplies a weaker
`O(S^2 log^(11+epsilon) S)` backup under its favorable-cross-section and patch
hypotheses. Our non-full-fiber total-line pair intersections satisfy the
one-point fixed-time cross-section condition, but that theorem remains backup
only and needs more care for partial patches and birth/death boundaries:

- https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ESA.2024.6

### Load-bearing assumptions

- one fixed coefficient chart and finite physical interval;
- total degree-at-most-two line coefficients;
- constant-description semialgebraic motion;
- no pair full-fiber coincidence;
- no identically-zero triple determinant;
- isolated generic events, with four-way/simultaneous/persistent active strata
  rejected or handled separately;
- semantic positive-length owner changes are counted separately from inactive
  candidate roots and analytic guards.

### What it does not prove

- It does not bound the raw `Theta(S^3)` universe of triple predicates.
- It does not turn the current active-owner closure into an
  `O(E polylog S)` compiler.
- It does not bound repeated all-site witnesses, root-isolation bit complexity,
  or chart-wise duplicated word storage.
- It does not make event count independent of physical duration or trajectory
  degree.

### Decision

Adopt as a generic **semantic output-size theorem** and cite the primary paper.
Keep the current compiler-work caveat verbatim. A kinetic envelope data
structure is a possible later compiler redesign, not part of the present
native-memory closure.

## 5. Structural repair versus numeric payload refresh

The handoff's conclusion is right but its original proof conflates one changed
node with all `J` changed nodes.

Define the exact changed-coordinate set

```text
Delta_L(U) = {
  (p,c,j,r): the materialized stored node length changes under update U
}.
```

Any exact eager updater that returns the current materialized node-length ABI
must perform

```text
Omega(|Delta_L(U)|)
```

scalar output work. If `R(U)` merely means incidences for which **some** node
changes, the unconditional bound is only `Omega(|R(U)|)`.

There are, however, stable-topology families with no event change and all node
values changing. A minimal two-site example uses a fixed ray over `[0,2]`,
sites at `0` and `2`, and constant power weights. The stable cut is

```text
z* = 1 + (w_0-w_1)/4.
```

A small update to `w_0` moves both run lengths at every compiler node while
the owner word remains `(0,1)` and `E_U=0`. Replicating the track yields

```text
|Delta_L(U)|
  = Theta(sum_(p,c,r in R(U)) J_(p,c)).
```

Equal-rank blocks therefore have a valid **worst-case**
`Omega(J|R(U)|)` refresh lower bound.

The current immutable contiguous block materializer and full-content digests
can impose a stronger engineering rebuild granularity than this representation
lower bound. That is an implementation choice, not the mathematical theorem.

### Decision

Adopt the corrected changed-entry theorem. It proves that `E_U=0` does not make
numeric refresh free and that an `O(E_U polylog S)` complete warm-update claim
is false for the current eager ABI. It does not show local structural repair is
useless; it identifies the two supports that must be tracked:

```text
parameter/site -> affected predicate sources,
parameter/site -> affected active run/chart/node payload coordinates.
```

Full recompilation remains the production rule until a measured affected-
support updater is independently certified and beats rebuilding.

## 6. Discriminants and resultants

For a complete rooted and rootless predicate registry with
`h_k(t,eta) in Q[t,eta]`, fixed nonzero `t` degree, nonzero discriminant,
nonzero endpoint values, and simple pairwise-distinct base roots over a compact
optimizer homotopy interval:

- every real base root continues uniquely and real-analytically;
- the number of interior real roots is constant;
- the root cannot become multiple, enter, or leave through an endpoint;
- nonzero resultants between adjacent base root groups from distinct sources
  prevent the first cross-source ordering collision;
- same-source collisions are caught by the discriminant.

This is mathematically sound but incomplete as a program certificate. It does
not establish:

- complete registry closure;
- semantic activity or co-minimality;
- left/right owner words;
- full-fiber companion nonvanishing;
- ray noncollapse and other nonroot guards;
- support for initially shared or simultaneous groups.

Whole-polynomial resultants can also reject because of irrelevant complex-root
collisions. Symbolic expressions may grow substantially.

### Decision

Record as an independent exact CPU-oracle/proof-compression option only. The
current tensor-Bernstein root-tube and complement proof is more constructive
and already covers more of the accepted stratum. Do not replace it, do not
change the full-recompile policy, and do not implement the oracle before the
native fused route and paper evidence gates.

## 7. Commutation and same-color coalescing

For physical P0 segments,

```text
g_1 star g_2 - g_2 star g_1
  = (0, (1-beta_1)(1-beta_2)(c_1-c_2)).
```

Two nonidentity segments commute iff their colors agree. The maximal
pairwise-commuting physical families are transparent identities plus transfers
with one fixed source color. This is already present in the paper appendix and
oracle; it is wording cleanup, not a new result.

Adjacent equal-color segments coalesce in the forward P0 action:

```text
g(tau_1,c) star g(tau_2,c)
  = g(tau_1+tau_2,c).
```

The handoff's gradient claim is too broad. If `c_1` and `c_2` are independent
parameters that merely have equal current values, their two material
Jacobians differ from one merged-color Jacobian. Exact full VJP preservation
requires a literally tied/shared color parameter plus retained chain-rule
provenance for every original optical-depth/geometry parameter. Expected-depth
outputs, owner diagnostics, non-P0 material, and floating-point bitwise parity
also invalidate the broad claim.

### Decision

Do not put coalescing on the critical memory-light path. Measure adjacent tied-
color hit rate before considering it as a guarded micro-optimization.

## Branches considered

### Branch A: redesign the runtime around jets

Rejected. Pointwise tangent closure is already represented by existing dual
oracles, and native jet storage grows with selected directions without solving
temporal rank.

### Branch B: enable opacity termination immediately

Rejected. The supplied general gradient criterion is incomplete, D6's
continuous geometry/ray tangent bound remains open, and obtaining a suffix
certificate may itself traverse the suffix.

### Branch C: replace the compiler with a kinetic-envelope KDS now

Deferred. The output bound makes this mathematically plausible, but the current
paper blocker is native fused integration and evidence, not CPU compile
asymptotics. First measure compiler share and semantic/raw-event ratios.

### Branch D: use discriminants/resultants as the production repairer

Rejected. They do not certify semantic program validity and are less
constructive than the existing Bernstein reference.

### Branch E: keep the existing runtime and strengthen the theorem contract

Selected. Add minimality, the generic semantic-event bound, and the corrected
numeric refresh lower bound; retain current runtime and safety policy.

## Promotion tests before any systems optimization

1. **Event-bound instrumentation:** log `S`, semantic `E`, analytic guards,
   inactive candidate roots, unique witnessed words, all-site witness work,
   `R`, and compiler wall time. The theorem is useful only if semantic output
   and compiler overhead can be separated empirically.
2. **Tail certificate:** exact JVP/VJP comparisons including prefix directions,
   background tangents, HDR diameter, loss-cotangent effects, and the cost of
   computing the certificate itself.
3. **Affected repair:** stable `E_U=0` fixtures with dense numeric change,
   exact changed-entry accounting, fresh-compile equality, and a broad-support
   fallback threshold.
4. **Resultant oracle:** every accepted result must agree with both the current
   Bernstein certificate and a fresh exact compilation; any disagreement kills
   promotion.
5. **Coalescing:** tied-color and independent-equal-color gradient fixtures,
   depth-auxiliary counterexamples, and measured realistic hit rate.

## Final priority

The attachment improves the paper's theorem story, not the current shader
architecture. The priority remains:

```text
source-complete fused request
-> coordinator selection and commit authority
-> quiet-host native build/parity/allocator gates
-> F=8/64/300 requested-density experiment
-> only then optional local repair or tail skipping
```

No new formulation, public name, or renderer fork should be created from this
intake.
