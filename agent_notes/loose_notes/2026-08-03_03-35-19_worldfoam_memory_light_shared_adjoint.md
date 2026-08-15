# WorldFoam memory-light shared adjoint: exact replay, transfer coefficients, and fixed-topology geometry

Date: 2026-08-03 KST

Status: CPU reference and behavior-test pass completed. This note records a bounded
implementation advance, not a production Metal trainer or publication-scale
benchmark claim.

## Context

The immediate question was whether WorldFoam can have the same architectural
memory property as World Tubes:

```text
expensive world-side raster/intersection/backward work shared across time
cheap camera/sample evaluation linear in requested frames
no frame axis in world parameters or reverse interaction storage
```

The concern was legitimate because the current unified `worldfoam` paper row
uses `MetalPowerFoamVideo`, whose geometry, density, texture, and spherical-
Voronoi state have an explicit frame dimension. At the selected `K=4`, `M=3`
configuration it stores 93 floats per cell per frame. For 1024 cells and 300
frames that is 28,569,600 parameters before gradients and Adam state. That
row is a per-frame dynamic PowerFoam baseline, not native WorldFoam.

Static PowerFoam itself has the useful memory structure:

```text
one world state + sparse neighbor graph + direct/recomputed backward
```

Its persistent storage is independent of camera count and frame count. The
current WorldFoam Gate4 prototypes already recover part of that structure:

- shared 4D sites and one `[site_count,4]` RGBA field;
- affine camera-ray tracks;
- rational boundary depths;
- base owner words plus sparse change records;
- fused RGB MSE and shared site-gradient accumulation.

But the promoted Metal path freezes geometry and still scans the cell word at
every sampled frame. The current native cutwalk compiler also discovers its
structure by looping frames, pixels, all pairwise boundaries, and sites. That
is why the existing evidence is a good prototype signal but not the final
memory-light trainer.

## Current model

There are two distinct targets. They must not be conflated.

### Layer A: exact chunked fixed-word replay

Given a caller-supplied owner word, process requested frames in a bounded block
and replay the word exactly. Accumulate only shared site and active-boundary
gradients; camera-ray gradients are optional because cameras are normally
fixed. This gives frame-independent logical reverse interaction memory,
although work still includes one run scan per sampled frame. The CPU API still
receives resident targets/times, so actual data streaming remains open.

### Layer B: compiled total-transfer coefficients

On each checked fixed-word chart, evaluate the *whole ordered transfer word* at
`J` temporal nodes and fit a compact basis for

```text
G(t) = (beta(t), m(t)).
```

Requested frames then perform only basis evaluation, RGB loss, and reduction
into `J` coefficient adjoints. The cell-word scan and world VJP run at the
`J` compile nodes, not at all `F` samples.

This second layer is the closest WorldFoam analog of World Tubes trace
coefficients. It is approximate in general and currently uses only a blockwise
sampled forward-error guard. It therefore needs adaptive rank, chart splitting,
and continuous forward/Jacobian certification before promotion.

## Do we need another Schur complement?

No new world/depth marginal is indicated.

World Tubes uses Gaussian closure under marginalization:

```text
Gaussian pullback + eliminate depth -> Schur-complement UVT trace
```

WorldFoam deliberately retains depth and order:

```text
cell-path events
-> exact local segment transfer
-> ordered transfer product
-> event-stable temporal coefficients
-> shared coefficient/world adjoint
```

Differently colored ordered segments do not have a universal exact fixed-size
temporal closure. The transfer product is noncommutative whenever colors and
opacity overlap. For general projective endpoint functions, the honest route
is event closure plus an adaptive certified basis, not a forced second Schur
identity.

New foundational math should be opened only if one of these implementation
tests fails:

1. sparse dynamic adjacency cannot reproduce the required owner words;
2. topology invalidation forces nearly complete recompilation every step;
3. temporal basis rank grows with frame sampling density at fixed physical
   interval and tolerance;
4. material transfer has no acceptably compact forward-and-VJP closure.

## Exact ordered-transfer algebra

One constant-color P0 segment has

```text
L_phys = ||d(t)|| (z_right-z_left)
tau = sigma L_phys
beta = exp(-tau)
m = (1-beta)c
g = (beta,m).
```

Front-to-back composition is

```text
(beta_1,m_1) tensor (beta_2,m_2)
  = (beta_1 beta_2, m_1 + beta_1 m_2).
```

The old reverse form uses prefix and suffix state. It is correct, but a more
useful shader identity eliminates all per-run reverse arrays.

Let total word transfer be

```text
G = (beta_G,m_G),
```

the prefix before the current segment be

```text
P = (T,F),
```

and the local segment be

```text
g = (beta,m).
```

For total-transfer cotangent `(bar_beta_G, bar_m_G)` and any local parameter
`p`, the exact second-pass identity is

```text
bar_p
  = bar_m_G dot (T partial_p m)
    + partial_p log(beta)
      [bar_m_G dot (m_G - F - T m) + bar_beta_G beta_G].
```

Then advance only

```text
F <- F + T m
T <- T beta.
```

For P0 density/color, this reduces to

```text
bar_tau = bar_m_G dot (F + T c - m_G) - beta_G bar_beta_G
bar_sigma = L_phys bar_tau
bar_L_phys = sigma bar_tau
bar_c = T (1-beta) bar_m_G.
```

For a decoded RGB pixel `I=m_G+beta_G background` with cotangent `v`:

```text
bar_tau = v dot (F + T c - I).
```

This needs the final transfer plus one front-to-back replay and constant local
state. It removes the current Metal shader's thread-private arrays:

```text
owners[129]
lengths[129]
trans_before[129].
```

That is a genuine memory/bandwidth improvement, but it is an algebraic
rearrangement of ordered transfer, not a new world representation.

The `||d(t)||` factor is the ordinary-depth fiber Jacobian. It is essential:
the first draft of the CPU reference omitted it and therefore changed opacity
under a harmless rescaling of the ray coordinate. The corrected forward and
VJP pass an orientation-preserving affine depth-rescaling test, including the
gradient into `d_0,d_1` when camera gradients are requested. Ordinary-versus-
log-depth parity and general nonlinear ray charts remain production gates.

## Factorized moving-boundary adjoint

For a 4D affine face

```text
h(X)=n dot x + n_t t + b = 0
```

and an affine ray track

```text
o(t)=o_0+t o_1
d(t)=d_0+t d_1,
```

the cut depth is

```text
z(t) = (A+B t)/(C+D t),

A = -o_0 dot n - b
B = -o_1 dot n - n_t
C =  d_0 dot n
D =  d_1 dot n.
```

For endpoint cotangent `bar_z`, first reduce into the four depth coefficients:

```text
partial_A z = 1/q
partial_B z = t/q
partial_C z = -p/q^2
partial_D z = -t p/q^2,

p=A+Bt, q=C+Dt.
```

All requested times accumulate into one sparse active track-boundary
coefficient adjoint. The once-per-incidence VJP is then

```text
bar_n  += -bar_A o_0 - bar_B o_1 + bar_C d_0 + bar_D d_1
bar_n_t += -bar_B
bar_b   += -bar_A

bar_o_0 += -bar_A n
bar_o_1 += -bar_B n
bar_d_0 +=  bar_C n
bar_d_1 +=  bar_D n.
```

The implementation originally used a dense
`[track_count,boundary_count,4]` buffer. That would have defeated the memory
goal. It was rejected during red-team and replaced by an explicit sparse
`[active_track_boundary_incidence,4]` buffer.

## Sparse power-boundary scatter

The all-pairs `make_boundaries_4d` reference is `O(site_count^2)` and cannot be
the production world structure. The new reference accepts explicit active
neighbor pairs only.

For sites `a=(x_a,t_a,w_a)` and `c=(x_c,t_c,w_c)`:

```text
N = 2(c_xyzt-a_xyzt)
b = ||a_xyzt||^2 - ||c_xyzt||^2 - w_a + w_c.
```

The once-per-face scatter is

```text
bar_a_xyzt = -2 bar_N + 2 a_xyzt bar_b
bar_c_xyzt =  2 bar_N - 2 c_xyzt bar_b
bar_w_a = -bar_b
bar_w_c =  bar_b.
```

Repeated site ids are accumulated with sparse index addition.

## Analytic supplied-word ordering check

For affine faces and affine ray tracks, each referenced cut has a linear
denominator and every coordinate segment length is quadratic-over-quadratic.
The CPU reference uses float64 roots/extrema to check over a chart:

- no referenced denominator zero;
- the supplied word connects near to far;
- adjacent runs share cut ids;
- every rational coordinate segment length has a positive analytic minimum;
- the product of that minimum and the minimum fiber speed gives a conservative
  physical-length lower bound.

This is intentionally an **analytic ordering check**, not a certificate. It is
not outward-rounded interval arithmetic and does not prove that the compiler
discovered the correct or complete power-cell owner word. The power wrapper
also checks each supplied adjacent owner transition against the oriented active
pair, but a third cell can still undercut both. Independent owner inequalities,
event witnesses, or a compiler proof token remain required.

## Compiled temporal transfer adjoint

The stronger prototype uses a Chebyshev basis on each stable interval.

At `J` roots, evaluate the exact total transfer:

```text
g_j = G(t_j).
```

With interpolation matrix `A`, store

```text
a = A g.
```

For requested sample basis row `phi(t)`:

```text
G_hat(t) = phi(t) a.
```

The reverse is factored exactly for this compiled representation:

```text
bar_a = sum_samples phi(t)^T bar_G_hat(t)
bar_g = A^T bar_a
```

followed by the constant-state word VJP at only `J` nodes and one sparse
boundary/ray/site scatter.

For `P` ray tracks, spatial track block `B_p`, average run count `R`, `F`
requested samples, temporal block `K`, and fixed rank `J`:

```text
compile + world VJP: O(P J R)
sample + residual reduction: O(P F J)
peak per-step atlas/scratch: O(B_p J + B_p K + block incidences + world)
```

There is no `F R` term in the compiled world-side reverse. The unavoidable
linear slice reads targets/residuals and evaluates the small temporal basis.
The low-level full-track function still materializes `O(PJ)` coefficients; the
recommended CPU wrapper tiles tracks so that this does not become gigabytes at
native resolution. Caller-resident rays, target tensors, and Python owner words
remain `O(P)`, `O(PF)`, and `O(PR)` respectively and are not yet streamed/CSR.

The current approximation check is blockwise sampled forward error, not a
formal continuous transfer or Jacobian certificate. A hard smooth chart in the
tests produces about `2.9e-3` max transfer error at `J=16`, proving fixed rank
is not universal; raw interpolation can also leave the physical transfer cone.
Promotion therefore requires:

- adaptive `J` selected by tolerance rather than a fixed value;
- a comparison against structure-preserving total-transfer coordinates such as
  `kappa=-log(beta)` and `c_eff=m/(1-beta)`, whose decoder
  `(exp(-kappa),(1-exp(-kappa))c_eff)` can enforce the physical cone;
- continuous or interval forward error bound;
- Jacobian/VJP error bound;
- physical beta/range safeguards;
- event splitting when rank or certificate cost grows.

## Implemented code

New isolated CPU reference:

```text
research_experiments/world_foam_lane2/compiled_transfer_adjoint.py
```

It implements:

- lowering and boundary-to-site VJP for caller-supplied sparse active pairs;
- sparse track-boundary Mobius coefficient lowering and one shared VJP;
- analytic denominator and rational-length ordering checks for supplied words;
- the ordinary-depth fiber Jacobian and affine gauge-rescaling parity;
- exact fixed-word chunked MSE/VJP with constant reverse state;
- optional camera gradients, disabled by default for fixed-camera training;
- compact Chebyshev total-transfer compilation;
- time- and track-blocked residual reduction into coefficient adjoints;
- recomputed node transfer VJP;
- selected logical tensor-payload accounting (not measured allocator peak).

Behavior tests:

```text
research_experiments/world_foam_lane2/test_compiled_transfer_adjoint.py
```

They cover 16 behavior contracts, including supplied-word ordering and
homogeneous-plane scaling, full-transfer sampled error, the fixed-rank hard
chart, compiled/exact/manual VJPs versus autograd, frame- and track-block
invariance, affine depth-gauge rescaling, sparse power scatter and composed
site-geometry VJP, no retained autograd graph, fail-closed chart extrapolation,
and denominator events.

## Verification

No MPS or CUDA workload was launched.

CPU results:

```text
16 passed
    new compiled-transfer/shared-adjoint behavior suite

34 passed
    new suite + existing cell-path transfer, coefficient-factorization,
    and Gate4 moving-ray compiler contracts

53 passed, 7 deselected
    non-Metal M0--M5 material transfer and fit suites

9 passed + native source verifier status ok
    suffixed constant-state packed-framegroup Metal source contract

ruff check: all checks passed
```

The Metal extension was intentionally not rebuilt and no MPS dispatch ran, so
the installed binary does not yet expose the new schema.

The smooth synthetic chart reached compiled-versus-exact forward and all-world
VJP agreement at the test tolerance with `J=16`. This is a bounded reference
result, not evidence that `J=16` suffices on public scenes.

## Branches tried, rejected, or narrowed

### Rejected: another Gaussian-style Schur marginal

It would erase the retained-depth/order contribution and does not close
differently colored noncommuting transfer words.

### Rejected: dense track-by-boundary coefficient adjoints

They are independent of frame count but can be enormous. The implementation
now stores only referenced sparse incidences.

### Rejected: suffix/per-run reverse arrays as the production target

They are correct but consume thread-private storage proportional to maximum
run count. The prefix-only second pass is exact and constant-state.

### Narrowed: “topology/continuous certificate”

Float64 analytic denominator and length checks reject many bad supplied words,
but they are neither outward-rounded nor an owner-word proof. The stronger
name was removed to prevent an invalid claim.

### Narrowed: Chebyshev “certificate”

The current validation is sampled. It is a useful falsification gate, not a
continuous forward/Jacobian proof.

### Rejected: all-pairs boundaries in the native world path

Production must consume sparse active adjacency. All-pairs construction stays
only as a tiny-fixture oracle.

## What this reference establishes and what remains unproved

Implemented and tested by the bounded CPU reference:

- shared native parameters need no frame axis;
- exact logical reverse interaction memory can be independent of total frame
  count for a fixed caller-resident word program;
- per-run reverse arrays are unnecessary;
- moving endpoint gradients reduce into sparse camera/boundary coefficients;
- boundary gradients scatter analytically to shared 4D power sites;
- the experimental total-transfer coefficient factorization has the desired
  `J R + F J` work split under a supplied fixed word and passing sampled gate;
- its analytic shared VJP matches the compiled representation's autograd.

Not proved:

- correct continuous owner-word discovery;
- sparse dynamic adjacency at public-scene scale;
- topology refresh during geometry optimization;
- continuous forward and VJP approximation bounds;
- complete owner correctness (including third-cell undercuts);
- actual target/ray streaming or flat CSR topology;
- Metal runtime parity or speed;
- current packing-cap removal;
- full trainer integration;
- safe native-resolution memory envelope;
- public multi-scene quality.

## Production shader and trainer sequence

Do this in order:

1. Rebuild and run bounded parity for the new suffixed packed-framegroup Metal
   source op. It has the constant-state two-pass P0 VJP, physical fiber length,
   and direct `grad_boundary[B,5]` beside `grad_site_rgba[S,4]`, but it has not
   been compiled or executed.
2. Replace its sample-scale boundary atomics with sparse per-track Mobius
   coefficient accumulation and a once-per-incidence boundary VJP.
3. Add the sparse power-boundary-to-site scatter op and verify site/weight
   finite differences under fixed topology.
4. Change the trainer API to spatial blocks `B_p` and selected frame blocks
   `K`; do not retain
   `explicit_rays[P,F,6]`, eager full targets, or `P(F-1)` frame selectors.
5. Replace all-pairs 4D boundaries with sparse active adjacency and a small
   all-pairs correctness oracle.
6. Implement continuous owner/event discovery and explicit refresh/fallback
   rules. Topology must not depend on a density cutoff that changes during
   training.
7. Replace per-track Python dictionaries with flat CSR/int32 topology, then
   remove the current 32-frame bitmask, 256-site, 4093-boundary, and 129-run
   production caps or shard them explicitly.
8. Wire P0 through native geometry first. Rich M3/M5 promotion can reuse the
   existing `(bar_tau,bar_beta,bar_m)` material VJP ABI after P0 parity is green.
9. Port the total-transfer `J`-node atlas only if exact chunked replay shows
   that the per-frame run scan is the remaining bottleneck. Add adaptive rank
   and forward/Jacobian certificates before calling it a paper method.
10. Replace the unified paper row's per-frame `MetalPowerFoamVideo` object with
   this native shared world/compiler path. Keep the former row as the honest
   per-frame PowerFoam baseline.
11. Profile parameter bytes, optimizer bytes, structural atlas bytes, reverse
    interaction bytes, target/ray blocks, allocator peak, and wall time across
    fixed-duration `F=4..256` unique samples before any large public run.

## Stop rules

Stop or narrow the WorldFoam systems claim if:

- active adjacency or event records grow approximately with frame sampling
  density over a fixed physical interval;
- geometry updates invalidate most words every optimizer step;
- adaptive temporal rank grows with requested frame count rather than physical
  camera/world complexity;
- the compiled world VJP cannot match exact replay gradients at the declared
  tolerance;
- public quality requires so many cells/material coefficients that the shared
  representation loses its memory advantage.

## Decision

The reference demonstrates a viable frame-memory-light fixed-word formulation
without replacing WorldFoam's retained-depth math. The experimental coefficient
atlas shows the World-Tubes-like work split on bounded charts, but production,
resolution-scale memory, complete topology, and continuous error control remain
unproven.

No new Schur complement is currently indicated. Gauge, event-density, or
adaptive-rank failures may still require revised lowering, but they do not yet
justify a wholesale representation restart. The next engineering action is to
build/parity the bounded P0 constant-state Metal bridge and then replace its
direct boundary atomics with sparse incidence reduction—not a publication-
scale run on this host.

## Expansion pass: the World-Tubes-shaped work split

The preceding decision was too weak about arithmetic and bandwidth. Exact
constant-state replay plus sparse incidence reduction removes a frame-sized
reverse tape, but it still scans every ordered word at every requested sample:

```text
exact fixed-word replay world work = O(P F R)
```

That is frame-memory-light, but it is not the World Tubes compiler shape. The
strong systems target requires the expensive world/word traversal to happen at
adaptive compiler nodes `J`, while the requested-sample axis performs only
basis evaluation, target reads, residual formation, and coefficient-adjoint
reduction:

```text
world compile                       O(P J R)
cheap sample evaluation/reduction  O(P F J)
world reverse replay               O(P J R) after retaining compiled node totals
```

The linear `F` term is unavoidable because the renderer must write and compare
`P F` output samples. The claim is that it no longer rereads and differentiates
the full world word for every one of those samples.

### Transfer-level analogue of the Gaussian Schur closure

World Tubes gets its closure by marginalizing Gaussian depth with a Schur
complement. WorldFoam cannot do that without deleting the retained depth order
that distinguishes it. Its corresponding compiled object is the *total ordered
affine transfer* of a stable ray word:

```text
G(c) = m + beta c
front(G_back(c)) = (m_front + beta_front m_back)
                  + (beta_front beta_back) c
```

For temporal fitting, the current candidate uses the affine-transfer logarithm

```text
kappa = -log(beta)
v     = kappa m / (1 - beta)
```

and decodes with

```text
beta = exp(-kappa)
m    = ((1 - exp(-kappa)) / kappa) v.
```

The physical cone is simple in these coordinates:

```text
kappa >= 0
0 <= v_rgb <= kappa
```

This operation is applied only after exact ordered composition at each compiler
node, so it does not discard moving-camera depth swaps or differently colored
overlap. It is the transfer-level analogue of the Schur factorization in the
systems architecture, but not another Gaussian Schur complement.

The source-only native pipeline now has the intended stages under a suffixed,
unbuilt ABI:

1. lower active track/boundary incidences to sparse Mobius coefficients;
2. replay exact ordered words at `J` compiler nodes;
3. encode total transfers as `(kappa,v)`;
4. evaluate selected samples from caller-supplied basis weights and immediately
   reduce residuals to node cotangents;
5. replay the exact word VJP at the same `J` nodes;
6. finalize sparse incidence, boundary, and eventually 4D-site gradients.

The direct and staged exact paths remain useful oracles and fallbacks. They are
not sufficient evidence for sublinear world-side work.

### The necessary correction: primal rank is not tangent rank

The hard two-cell opacity fixture exposed a failure that a forward-only gate
cannot see. With two Lie nodes, rendered transfer is accurate at roughly
floating-point error, and gradients of the currently active red cell agree.
However, a green cell with zero current density has a nonzero perturbation
direction outside that rank-two primal curve. Its compiled density gradient is
wrong by about `1.91e-3`, and a sparse depth-coefficient cotangent is wrong by
about `1.40e-2`.

Across `J = 2,4,8,16,32`, primal transfer error stays near `1e-15`, while the
maximum exact-world MSE-VJP error falls approximately as:

```text
1.40e-2, 1.43e-3, 3.26e-5, 2.78e-6, 1.27e-8
```

The deterministic sampled tangent diagnostic follows the same death curve,
from about `1.02e-2` to `4.57e-9`. Therefore the affine-log chart is a strong
primal representation, but forward fit error is not a sufficient rank rule.

The minimum defensible formulation is now:

```text
piecewise affine-log transfer atlas
+ adaptive primal rank/splitting
+ adaptive tangent/VJP rank/splitting
+ event/topology validity
```

This does not yet justify inventing another world primitive or a different
transfer algebra. It does require a `C1`-style compiler contract: certify or
falsify both the total transfer and its world-parameter derivative. Sampled
tangent probes are a useful compile/refresh-time diagnostic, not a continuous
certificate and not hidden per-step work.

### Red-team fixes incorporated

- Manual-VJP entry points detach inputs and return tensors without retained
  autograd graphs.
- Tiny optical depths use `-expm1(-tau)` in CPU and source-only native forward
  and reverse paths; `1-exp(-tau)` loses the color direction near zero.
- Interpolated samples, not only fit nodes, receive fail-closed Lie-cone checks.
- Sampled forward/tangent validation is disabled by default and separately
  accounted when requested. A 257-sample exact validation inside every train
  step would reintroduce `O(P V R)` work and can dominate small `J`.
- The initial CPU prefix-only node reverse rescanned the word to recover its
  total transfer. Retaining the already compiled node total removes that scan,
  so the current CPU/native-shaped reverse is one `J R` replay.

CPU verification after these corrections:

```text
38 passed
13 source-contract tests + 4 subtests passed
native source verifier status=ok
```

This combined exact replay, sparse-incidence, Lie-chart, compiled-world,
tiny-opacity, graph-retention, and separate tangent-gate suites. No MPS or CUDA
workload was launched. The native ABI is still unbuilt and has no runtime
parity evidence.

### Revised decision

Yes, WorldFoam needs a closure that plays the same architectural role as the
Schur-derived UVT trace in World Tubes. No, the current evidence does not call
for another Gaussian Schur complement or a wholesale new representation. The
best current path is the ordered affine-transfer Lie atlas with joint
primal/tangent adaptive rank and chart splitting.

Before claiming the memory-bandwidth result, finish native source/runtime
parity, sparse boundary-to-site scatter, streamed target/ray/topology blocks,
and continuous owner/event plus forward/Jacobian validity. Then measure fixed-
duration `F=4..256` scaling on an approved clean host. More GPU memory can make
experiments convenient, but it must not substitute for this factorization.

## Continuation: staged/compact lifecycle and continuous certificate

The CPU implementation now exposes the intended production schedule rather
than only a monolithic reference call:

```text
refresh compact world/node atlas once
-> accumulate arbitrary K-frame target blocks into J node bars
-> replay/finalize the world and active boundaries once
-> lower/scatter active-face bars into global 4D sites once
```

The accumulator retains no targets, predictions, residuals, or `F x R` tape.
One global loss denominator makes `K=1`, intermediate `K`, and `K=F`
partitions equivalent. Compact blocks use flat CSR topology and gather only
referenced tracks, boundaries, and sites. Active power faces are derived from
the same compact site snapshot used by the site VJP; independent boundary
input is not accepted. Tensor-version signatures cover the world, source, and
all topology tensors, and result tokens cannot be scattered through a
different prepared block.

The verified fixed-duration CPU artifact is:

```text
artifacts/foundation_gates/worldfoam_compiled_lie_frame_density_cpu_20260803.json
```

For `F={16,64,256,1024}` over one fixed interval it retains the selection
signature `J={16,2,2}`, refresh work `40`, CPU reverse work `40`, and selected
logical reverse state `2536` bytes. Sample-basis interactions alone grow
`88 -> 5632`. The toy has only two word runs, so its full interaction proxy is
not a practical win after retaining node totals: compiled/exact is
`1.750x, 1.125x, 0.969x, 0.930x`.
This artifact proves the expensive/cheap split and flat reverse state, not
native speed or bandwidth. Realistic large-`R` rows are mandatory.

A separate outward-rounded continuous interval implementation now certifies
fixed-word P0 affine-ray transfer and its first derivatives, with optional
propagation of the boundary Jacobian error bound through the power-boundary to
site/weight map. It is deliberately honest about scope: complete owner
identity, topology discovery, and runtime floating-point roundoff are not
certified. The difficult rank-16 chart needs more than the current `10,000`
work-unit budget; the low-rank charts fit the budget but fail strict transfer
and jet tolerances. This is a certificate-cost/rank-death result, not a reason
to rename the representation or invent another marginalization.

The current asymptotic target is therefore:

```text
exact replay:       about 3 F R interactions
compiled WorldFoam: about 2 J R + F J interactions
resident reverse:  O(B_p J + active incidence/site state + B_p K)
```

This is the World-Tubes-shaped contract. New foundational math is justified
only if measured fixed-duration `J`, chart count, or event count grows with
sample density or physical complexity enough to erase the `R >> J` regime.
The native source token follow-up is now green (`28 passed`, `10` subtests;
source verifier `116` schemas/implementations and `104` kernels). It separates
global normalization from local completion and rejects stale, mixed,
overlapping, missing, and duplicate chart/`K` tokens. The immediate work is
trainer-level spatial-block tiling/reduction, build/runtime parity, complete
owner/event compilation, and measured `F/R/J` allocator/bandwidth death
curves.

The selected-frame MP4 provider seam subsequently closed at CPU/source scope
(`17 passed`). Opt-in bundle construction performs no decode, and train or
heldout requests seek only their selected logical frames. This bounds target
residency but does not remove linear decode/I/O: exact dataset identity is
recomputed in bounded chunks at final reporting. The compiled trainer still
needs pixel-aware `B_p x K` gathering before accelerator transfer and affine
camera-ray fits in place of full `P x F` explicit ray storage.

## Expansion Pass: complete owner identity rather than supplied-word faith

### Trigger and backtrack

The continuous Lie-jet certificate originally assumed that the supplied owner
word was the true power-cell word. This was weaker than the intended fixed-
topology contract: positive ordered pairwise cuts do not exclude an unlisted
third site from owning an interior depth interval.

Status of the old assumption: invalidated as a production certificate boundary.
It remains acceptable only for isolated transfer fixtures.

### Continuous all-competitor proof

For power sites `i,j`, define

```text
Delta_ij(x,t) = power_i(x,t) - power_j(x,t)
              = n_ij . x + n_t,ij t + b_ij.
```

Owner `i` is valid exactly when `Delta_ij <= 0` for every competitor `j`.
Along the affine ray program

```text
x(t,z) = o0 + t o1 + z (d0 + t d1),
```

`Delta_ij` is affine in `z` at fixed `t`. Therefore it suffices to prove the
inequality at both endpoints of every claimed word segment. Finite endpoints
are the same Mobius cuts used by rendering; near/far endpoints are constants.
Exact-rational outward interval arithmetic and adaptive time bisection now
bound those endpoint inequalities continuously.

At a finite endpoint, directly substituting an interval Mobius depth loses the
dependency that the stored cut plane is zero and can make an exact tie look
positive forever. The corrected evaluation subtracts an exact rational
multiple of the stored zero plane from the site-derived owner plane before
interval evaluation. This changes no endpoint value and removes the dominant
dependency without pretending independently rounded planes are bit-identical.

File:

```text
research_experiments/world_foam_lane2/continuous_owner_identity_certificate.py
```

The adaptive wrapper now optionally requires this certificate for every chart.
`owner_identity_certified=true` means every word run was checked against every
site over the full closed time interval. It remains false unless the caller
supplies the exact site snapshot, boundary pairs, an ownership tolerance, and
every chart passes.

### Cost and claim boundary

```text
work ~ O(leaves * tracks * runs * sites)
frame-sampling dependence = none
```

This is compile/refresh work, not a warm-step tape. It is a complete small-
fixture oracle and a fail-closed certificate, not yet a scalable discovery
algorithm. Large scenes still need spatial competitor pruning whose excluded
sites carry a conservative lower-bound witness.

Certified: continuous supplied-word owner identity, including third-cell
undercuts, for affine rays and P0 power cells. Excluded: discovering replacement
words, topology-event roots, runtime Metal roundoff, and differentiation through
owner changes.

Falsification gates:

- a true three-site word passes continuously;
- a pairwise ordered word omitting the middle owner fails with a concrete
  third-cell witness;
- mutating the site snapshot invalidates the previous word;
- the API has no frame/sample-count input;
- bounded work fails closed.

Focused result: `7 passed` for the standalone and integrated owner nodes used
during development; the broader continuous suite is rerun after concurrent
streaming/native work settles.

## Expansion Pass: sparse owner discovery, bounded data blocks, and sealed native ownership

### A useful new formulation, but not another Schur complement

At one ray and camera time, the 4D power distance to site `i` is

```text
q_i(z) = ||d||^2 z^2 + a_i z + b_i.
```

The quadratic term is common to every site. The complete depth-owner word is
therefore the lower envelope of `S` lines, not the result of enumerating all
`S(S-1)/2` faces. Exact `Fraction` arithmetic over the binary64 inputs, slope
sorting, and a monotone lower hull now discover the fixed-time word in
`O(S log S)` work and `O(S)` scratch. Only adjacent owners emit active face
pairs. Random depth witnesses match brute power argmin, equal-slope dominated
sites disappear deterministically, and the three-site fixture emits two faces
rather than all three pairs.

An adaptive wrapper discovers at a rational midpoint, runs the continuous
all-site owner certificate over the entire interval, and bisects a failed
interval. A red-team counterexample found that certifying both closed sides is
not sufficient: a global temporal owner swap selected site 0 under the exact
fixed-time tie rule but site 1 under downstream right-continuous chart
dispatch. The compiler now detects that disagreement and emits a zero-width
unresolved seam rather than claiming coverage. If the split budget is zero,
the whole interval also remains explicitly unresolved. Irrational event roots,
zero-length segment birth/death strata, and production topology-chart
streaming remain open; this is not yet the trainer topology compiler.

Files:

```text
research_experiments/world_foam_lane2/sparse_power_word_compiler.py
research_experiments/world_foam_lane2/test_sparse_power_word_compiler.py
```

Focused result: `11 passed`; the combined continuous owner/adaptive suite is
`14 passed`.

### Bounded `B_p x K` data and global spatial reduction

The provider seam now stages exact selected targets `[B_p,K,3]` and rays
`[B_p,K,6]`, decoding at most one MP4 frame at a time and gathering pixels on
CPU before transfer. Fixed cameras emit exact affine ray rows; moving cameras
retain exact selected rays but fail closed when an affine program is required.
One global RGB denominator is carried through every spatial/time partition.
Rectangular multi-view blocks now move view onto the track axis exactly:
`P x (V K)` becomes `(V P) x K`, so the native one-affine-program-per-track
contract is met without changing the `P V K 3` loss denominator. View order is
canonical and view-local spatial blocks remain contiguous in the expanded
track axis.
At 512 square, `K=8`, the full-image targets+rays would be about `72 MiB`;
`B_p=8192,K=8` is `2.25 MiB` plus roughly `0.375 MiB` of fixed-camera program.

The compiled CPU reverse now allocates global site geometry, weight, density,
and color bars once. Each compact spatial block scatters immediately into
those buffers. Exact block ids, half-open global track coverage, atlas/source
identity, and normalization identity reject duplicate, overlapping, missing,
or mixed blocks. Loss and all gradients match `B_p=1`, intermediate, and full
spatial partitions while buffer pointers and resident bytes remain invariant.

Files:

```text
src/train/powerfoam_track_staging.py
research_experiments/world_foam_lane2/prepared_track_block.py
research_experiments/world_foam_lane2/staged_compiled_lie_adjoint.py
```

Independent CPU results: provider/staging/streaming `34 passed`; prepared and
staged adjoint `18 passed, 9 subtests passed`.

### Owner-aware native capability binding

The source-only native lifecycle no longer accepts an opaque promise that a
fixed word is valid. Binding reruns the actual continuous adaptive acceptance,
requires `owner_identity_certified=true`, and seals canonical facts covering
the exact topology/world tensor digests, policies, chart intervals/ranks/work,
per-chart owner certificate digests and bounds, and aggregate owner evidence.
Fabricated, failed, stale, mutated, and digest- or fact-tampered bindings fail
closed. Callers no longer inject `K x J` interpolation weights; weights derive
from the certified chart and local float64 sample times. Native runtime
floating-point roundoff remains explicitly uncertified.

Independent source/token result: `21 passed, 9 subtests passed`; Ruff and the
native source verifier are green. No extension build, MPS launch, CUDA run, or
training workload was performed.

### Revised remaining seam

The expensive-versus-cheap cost factorization is now represented at every
layer except one integrated trainer call:

```text
compile/refresh: O(B_p J R + certified topology work)
sample slice:    O(B_p K J) and linear target/ray I/O
world reverse:   O(B_p J R + active incidence/site state)
live step state: O(B_p J + B_p K + active topology + global site bars)
```

The next honest gate is the adapter from a view-local staged track block into
the sealed native topology/world/chart/sample/reverse tokens. Only after that
source path is complete should an approved quiet host rebuild the extension
and measure live allocator/bandwidth death curves.

## Expansion Pass: topology events are polynomial guards, not dyadic guesses

Three counterexamples broke the naive “bisect until each fixed word passes”
story:

1. A full-fiber temporal tie can make the standalone site-id tie rule disagree
   with the downstream right-continuous chart dispatcher.
2. A run whose length is zero exactly at birth/death cannot satisfy a uniform
   positive-length bound on either adjacent open interval.
3. A triple-cut event can occur at an irrational time such as `sqrt(2)`, so no
   amount of binary midpoint splitting can put a float endpoint exactly on it.

The new event layer derives exact binary64-real predicates. For a pair face,

```text
Delta_ij(t,z) = A_ij(t) z + B_ij(t),
```

with affine `A,B`. Near/far crossings are linear roots. Coincidence of
adjacent cuts is the quadratic

```text
B_ij A_jk - B_jk A_ij = 0.
```

Exact rational roots are retained exactly; irrational quadratic roots receive
rational Sturm isolators while retaining their polynomial guard. The
`sqrt(2)` counterexample is pinned. A rational isolator endpoint is explicitly
not relabelled as the real seam.

For finite P0 density/color, a zero-length segment has transfer `(1,0)`, the
ordered-product identity. It therefore needs no new material payload, but the
event must be marked, adjacent products must agree after deleting the zero
run, and the geometry VJP is one-sided or nondifferentiable at the event.
Positive-length full-fiber ties still fail closed until a material/tie policy
is declared.

Files:

```text
research_experiments/world_foam_lane2/power_topology_event_predicates.py
research_experiments/world_foam_lane2/test_power_topology_event_predicates.py
```

Independent event+sparse result: `20 passed`; Ruff and `py_compile` are green.
The next compiler task is to retain these algebraic guards through dispatch
and stream separate topology tokens on the two sides. This is new event-
compiler math, not a new WorldFoam representation or Schur marginal.

## Native source memory audit

For one resident spatial block (`P=B_p`), chart rank `J`, local sample block
`K`, runs `R`, active incidences `I`, faces `B`, and sites `S`, the source
implementation has the intended core shape:

```text
topology                           8(P+1) + 12R + 4I + 20B bytes
world refresh                      36S + 48P + 20B + 16I + 32 bytes
node chart + node cotangent        32 P J bytes
world-gradient state               16S + 16I + 20B bytes
one K block                         24 P K + 4 K J + 68 bytes
site-gradient output               20S bytes
```

The Metal reverse has scalar prefix state and no `F x R` tape. Intended work
is

```text
O(2 P sum_c J_c R_c + P sum_c F_c J_c + I + B).
```

At `P=8192,K=8,J=16`, node state plus bar is about `4 MiB`; native target plus
prediction is about `1.5 MiB`. There is no intrinsic 32-GB requirement.

The audit also found honest non-core gaps: certificate bindings still retain
their full CPU prepared atlas, current staging carries explicit `B_p x K` rays
beside affine rows, Python sample/block metadata remains `O(F)`/`O(F/K)`, all
chart tokens were retained together, and production training does not yet call
this lifecycle. A state-init validation also built and discarded a full
`F_c x J` weight matrix. That allocation is now removed: state init validates
only finite chart-local times, while interpolation weights remain `K x J`.

The remaining runtime contract must explicitly cap in-flight command buffers;
Python reference drops are not allocator evidence. Camera/program checks
belong once per spatial/view block, outside the warm `K` loop. No allocator,
runtime, or native parity result exists until the extension is rebuilt on an
approved quiet host.

## Sample-to-node reduction is now linear in chart rank

### Context and corrected derivation

The staged and native shared-adjoint paths still formed each `K x J`
sample-to-node slice as

```text
W = T(t) @ fit_matrix,
```

which costs `O(K J^2)`. This does not reintroduce an `F x R` world tape, but it
is the wrong sample-side slope for the claimed compiled schedule. The desired
cardinal row is the second-form barycentric interpolant

```text
q_j(t) = lambda_j / (x(t) - x_j)
W_j(t) = q_j(t) / sum_k q_k(t),
```

which costs `O(J)` per sample and yields the same pullback
`bar_node = W^T bar_sample`.

An important false start was rejected. For ideal Gauss-Chebyshev roots one can
write `lambda_j proportional to (-1)^j sin(theta_j)`, but those rank-only
weights are not necessarily the weights of the *stored floating-point nodes*.
For `[1e12,1e12+1]`, rank 32, the analytic-root vector misses the existing
dense oracle by more than `1e-3`. The implementation instead uses

```text
lambda = fit_matrix[-1, :] / max(abs(fit_matrix[-1, :])).
```

Column `j` of the inverse Chebyshev Vandermonde is cardinal polynomial `L_j`;
its final row is the highest-order Chebyshev coefficient of every `L_j` and is
therefore proportional to the barycentric weight for the actual rounded node.
At binding/schedule construction, `V @ fit_matrix ~= I` is checked once. The
per-`K` path then uses only the cached `J`-vector.

### Numerical and fail-closed contract

- Normalization uses the exact operation order used by `chebyshev_basis`.
- An exact normalized-node collision produces a bit-exact one-hot row before
  division; duplicate normalized nodes are rejected.
- A nonexact sample within `16 eps` scaled ULPs of a node uses the retained
  dense oracle rather than being snapped to the node.
- Barycentric terms are row-scaled before summation. A row falls back when
  intermediates are nonfinite or
  `abs(sum(q))/sum(abs(q)) <= 64 eps J`.
- Dense fallback must be finite and preserve the cardinal partition of unity;
  otherwise launch fails closed.
- Samples outside the chart interval and barycentric vectors not derived from
  the certified fit are rejected.

Assumptions: sample times are non-trainable schedule data; chart rank is at
least two; the small `J x J` fit remains resident as setup evidence and a rare
row oracle. This change removes quadratic *sample evaluation*, not the
`O(J^2)` schedule storage or one-time inverse construction.

### Implementation and falsification evidence

The compact schedule now owns the fit-derived `J`-vector and includes it in
its digest. The staged CPU adjoint constructs weights once per bounded frame
block, outside the spatial-track loop. Both strict frozen-evaluation and
material-training native bindings use the same helper; native sample tokens
and adapter results carry method and fallback interaction counts. The dense
path remains an oracle/fallback, not the production common path.

Behavior tests cover ranks `2,3,4,7,8,16,32`; ordinary, shifted, and
large-offset intervals; endpoints, exact nodes, and nextafter neighbors;
forward interpolation; node cotangent reduction; and autograd through node
values. The analytic-root large-offset counterexample is pinned.

Verified CPU/source results:

```text
compact interpolation behavior       31 passed
staged compiled Lie adjoint           13 passed, 6 subtests passed
native track adapter                   9 passed
native source/binding suite           22 passed, 11 subtests passed
native source verifier                status=ok
```

No extension build, MPS launch, CUDA launch, or training run was performed.
The next falsification step is a rebuilt native CPU/Metal parity run in an
approved quiet window, including a block containing exact and near-node
sample times, then measured timing to replace interaction proxies.

The combined material-training regression initially exposed a stale lifecycle
double that omitted the new sample-weight token fields. The fake now derives
real weights from the sealed training binding, and material-step accounting
aggregates method, linear interactions, exact-node rows, and dense-fallback
work across spatial blocks. The final combined compact/adapter/material/source
gate is `65 passed, 11 subtests passed`.

## Subsequent integrated source status

The earlier open-adapter and full-template statements are now historical. The
source path has the intended bounded lifecycle:

```text
compact schedule and one B_p topology
-> one chart token
-> one B_p x K target/ray block
-> immediate residual-to-node reduction
-> one J-node ordered-word reverse
-> global site-gradient scatter
```

The schedule can be created directly from chart specs in `O(sum J_c^2)` bytes
without a full-`P` atlas. The strict proof path can select
`track_local_sparse`, which streams only one track's referenced boundaries,
incidences, sites, and 12 ray coefficients and enforces a local-dual dimension
cap. The dense certificate remains a tiny-fixture oracle: for
`D=5B+12P+4I+4S`, its pointer-slot lower bound is
`max(16D^2,64 P J_max D)`. At `P=8192,J=16`, even the impossible
`B=I=0,S=1` case exceeds `768 GiB` before interval and `Fraction` objects.
That oracle, not the native representation, caused the catastrophic host
estimate.

The native source adapter enforces one in-flight sample block, global
`P*F*3` normalization, view-major rectangular multicamera factoring, immediate
prediction release, and caller-owned site bars across spatial blocks. A
piecewise adapter streams one topology/chart at a time, retains exact
polynomial guards for binary sample times, uses right-continuous seams, and
marks irrational algebraic endpoints non-paper until the native domain can
represent them exactly. Event-time and discrete-dispatch VJPs remain
unresolved; the reported geometry derivative is one-sided and fixed-topology.

Material optimization now has a deliberately narrower capability than strict
evaluation. It certifies owner/topology identity, freezes sites, weights, rays,
words, and schedules, and permits only P0 density/color refresh. Caller-owned
raw density is decoded through thresholded softplus and raw RGB through
sigmoid. Native physical bars use exact manual chain-rule VJPs; no clamp or
projected optimizer update is used. Prepared compact blocks and gradient
storage are reused across steps with zero per-step CPU atlas recompiles. These
results are explicitly `paper_evidence_eligible=false`; a frozen checkpoint
must rerun strict transfer/Jacobian certification.

The audited native tensor payload at `B_p=8192,K=8,J=16` is about `4 MiB` for
node state plus node bars and `1.5 MiB` for target plus prediction. There is no
intrinsic 32-GB requirement. This is not allocator evidence: the extension is
unbuilt, the checked-in binary predates the ABI, and command-buffer/driver
peaks are unmeasured.

The complete route-cost gate also corrected the old `0.930x` story. With the
now-verified linear barycentric weights, the two-run `J=16/2/2`, `F=1024`
fixture still costs `11608` compiled proxy interactions versus `6144` exact
and has no temporal break-even. It proves flat reverse state, not speed. A
high-run fixture does produce a compiled break-even. Runtime routing and paper
claims therefore require measured large-`R` `F/R/J` death curves.

Final safe source gates in this continuation include:

```text
provider/staging/streaming              34 passed
native verifier + adapter + material    34 passed, plus 11 verifier subtests
piecewise native adapter                 4 passed
compact interpolation                   31 passed
staged compiled adjoint                 13 passed, 6 subtests
```

### Why this is not yet the unified paper lane

The green optimizer is a hand-built fixed-topology rectangular fixture. The
Coffee Martini paper sampler is usually ragged across views and changes each
step; in the audited seed-17 schedule, `550/600` batches are non-rectangular.
There is also no production dataset-bound world/topology compiler, credible
frozen `[S,5]` native-4D initializer, forward-only streamed evaluator, native
checkpoint schema, or strict final-certificate artifact. Progressive stages
change both resolution and primitive count and therefore invalidate frozen
rays/topology unless each stage recompiles and recertifies. The native loss is
normalized MSE, while the legacy per-frame paper lane uses L1 plus `0.1 MSE`.

The honest order is: rebuild and establish bounded runtime parity; define a
versioned frozen-world asset; compile/serialize per-view/per-`B_p` programs;
support ragged per-view observations under one denominator and optimizer step;
add forward-only strict evaluation/checkpointing/accounting; then register a
fixed-resolution `worldfoam_native4d` side lane. Keep the existing
`worldfoam` lane labelled as per-frame `MetalPowerFoamVideo`.

### Revised formulation decision

No second Gaussian Schur complement is indicated. The WorldFoam analogue is
already present: common-quadratic cancellation for sparse power words,
piecewise ordered affine transfer, an affine-Lie total-transfer atlas, linear
sample-to-node reduction, and one shared sparse world adjoint. New foundational
math should be opened only if realistic fixed-duration measurements show rank,
event count, certificate cost, or topology refresh growing with requested
frame density or destroying quality/gradient parity. The current blockers are
compiler, orchestration, native runtime, and evidence—not the core shared-
backward factorization.
