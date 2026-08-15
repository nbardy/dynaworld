# Kinetic power-word event sufficiency: proof and red-team audit

Date: 2026-08-03

Status: source-grounded mathematical audit. This note gives a complete
generic-event theorem for one affine ray track, a fail-closed contract for
degenerate strata, and an implementable all-competitor active-hull sweep. It
does not claim that the continuous kinetic compiler is implemented.

## 1. Why this note exists

The memory-light WorldFoam route needs to compile a finite number of physical
owner/order changes over continuous time. Requested video frames must not be
used as topology probes. The immediate question is which exact event
polynomials suffice for the positive-length lower-envelope owner word of one
ray, and which apparent events are merely rational-cut denominator artifacts.

Inspected source:

- `research_experiments/world_foam_lane2/kinetic_power_word_compiler.py`
- `research_experiments/world_foam_lane2/rational_polynomial_roots.py`
- `research_experiments/world_foam_lane2/sparse_power_word_compiler.py`
- `research_experiments/world_foam_lane2/power_topology_event_predicates.py`
- the three corresponding focused test files

Observed source truth:

1. Fixed-time discovery is already an exact lower envelope of lines and
   deletes zero-length runs.
2. Kinetic pair differences `A_ij(t) z + B_ij(t)` are exact rational
   polynomials of degree at most two.
3. Adjacent-cut concurrence is represented by a degree-at-most-four cross
   product and general exact Sturm isolation exists.
4. The current wrappers deliberately reject full-fiber roots and candidate
   concurrence roots sharing denominator roots. That is safe but incomplete:
   a production event compiler must classify such roots rather than reject an
   entire time interval.
5. No current source enumerates the complete active-owner event set or emits
   the resulting half-open kinetic charts.

## 2. Setup and conventions

Fix one normalized affine ray track

```text
x(t,z) = o(t) + z d(t)
o(t)   = o0 + t o1
d(t)   = d0 + t d1
```

on the compact rectangle

```text
t in T = [tau_0, tau_1],
z in Z = [n, f],                 n < f.
```

Direct kinetic power sites are

```text
p_i(t) = p_i0 + t v_i,
w_i(t) = w_i0 + w_i1 t + w_i2 t^2.
```

After dropping the common `||d(t)||^2 z^2` term, site `i` contributes one
line in depth,

```text
L_i(t,z) = a_i(t) z + b_i(t),
```

where `a_i,b_i in Q[t]` have degree at most two when the binary64 inputs are
interpreted as exact rationals. For an ordered pair define

```text
Delta_ij(t,z) = L_i(t,z) - L_j(t,z)
              = A_ij(t) z + B_ij(t).
```

The fixed-time positive-length word `W(t)` is the ordered sequence of site ids
on the lower envelope over `[n,f]`, with every zero-length run removed. At
ordinary times, ties exist only at adjacent cuts and do not require a material
tie rule. Event-time ties require the explicit seam policy in Section 9.

The theorem assumes a valid fiber:

```text
d(t) != 0 for every t in the open chart.
```

This assumption is load-bearing. It is not implied by any pair or triple
predicate.

## 3. The universal finite candidate set

For every unordered pair `{i,j}`, define the two endpoint gaps

```text
N_ij(t) = Delta_ij(t,n) = A_ij(t) n + B_ij(t),
F_ij(t) = Delta_ij(t,f) = A_ij(t) f + B_ij(t).
```

Both have degree at most two. For every unordered triple `{i,j,k}`, choose one
orientation and define

```text
C_ijk(t) = B_ij(t) A_jk(t) - B_jk(t) A_ij(t).
```

Changing the orientation changes at most the sign. Its degree is at most four.
Finally define the ray-validity polynomial

```text
Q_d(t) = d(t) dot d(t).
```

It has degree at most two and is nonnegative over the reals.

### Claim E1: generic sufficiency

Assume:

1. `Q_d(t) > 0` on the supported time domain;
2. globally identical line functions have been quotiented by a deterministic
   tie rank;
3. any endpoint or triple predicate that is identically zero is either handled
   as a persistent stratum as in Section 8 or reported unresolved; and
4. the near/far depths are constant and finite.

Let `E` contain:

```text
{tau_0, tau_1}
union roots of every nonzero N_ij and F_ij in T
union roots of every nonzero C_ijk in T
union roots of Q_d in T.
```

Then `W(t)` is constant on every connected open component of `T \ E`.

The raw generic root bound, before coincident roots and activity are filtered,
is

```text
4 * binom(S,2) + 4 * binom(S,3) + 2
```

distinct roots counted with the degree bound rather than actual multiplicity.
This is an `O(S^3)` universal oracle, not the intended production sweep.

### Claim E2: what is actually minimal

The minimal geometric event characterization is:

1. a change of the active minimum at `z=n` or `z=f`;
2. a co-minimal interior concurrence of at least three lines;
3. an active full-fiber tie that changes the selected positive-length owner;
4. an invalid ray time `d(t)=0`; and
5. the time-domain endpoints.

Items 1--3 must be filtered by exact activity and by the one-sided words.
Raw roots involving only dominated lines are not genuine events. A universal
polynomial list cannot be both minimal and activity-oblivious.

An isolated full-fiber tie does not need a separate raw predicate. Because
`n != f`,

```text
N_ij(t*) = F_ij(t*) = 0
iff
A_ij(t*) = B_ij(t*) = 0.
```

The reverse direction follows by subtracting the endpoint equations:

```text
F_ij - N_ij = (f-n) A_ij.
```

Thus pair full-fiber events are exactly the common roots of the two endpoint
gaps. They still need a distinct event classification because their material
semantics differ from a zero-length run event.

### Claim E3: equal slope alone is not a topology event

Suppose

```text
A_ij(t*) = 0,    B_ij(t*) != 0.
```

At `t*`, `Delta_ij` is the same nonzero value for every `z in [n,f]`. By
continuity and compactness, it retains one strict sign on the whole depth
interval for all time sufficiently near `t*`. Therefore this pair cannot
exchange lower-envelope ownership there.

Consequences:

- roots of `A_ij` alone are not required topology seams;
- active `A_ij` roots are useful analytic guards for rational cut evaluation;
- a compiler must not confuse an infinite pair intersection with a physical
  compact-depth event.

## 4. Proof of generic sufficiency

Take a time `t*` that is not in the candidate set.

### 4.1 Endpoint owners are locally fixed

No endpoint gap vanishes at `t*`, so the minimum line at each of `n` and `f`
is unique. Every endpoint gap is continuous. The two endpoint owners therefore
remain unchanged in a time neighborhood of `t*`.

### 4.2 Every active interior cut is finite and simple

At an interior lower-envelope cut, two adjacent owner lines `i,j` satisfy

```text
Delta_ij(t*,z*) = 0.
```

If `A_ij(t*)=0`, then also `B_ij(t*)=0`; the pair would tie at both endpoints,
contradicting the absence of endpoint candidates. Therefore `A_ij(t*) != 0`,
and the cut is locally the continuous rational graph

```text
z_ij(t) = -B_ij(t) / A_ij(t).
```

No third line is co-minimal there. If `k` also equalled `i,j`, the corresponding
triple determinant would vanish at `t*`. Hence each active cut has a local
neighborhood in which the same two lines remain the only minimizers at that
cut.

### 4.3 No inactive line can first appear inside a run

Let `i` own a depth segment `[l(t),r(t)]`, and let `k` be any competitor. The
competitor gap

```text
G_ki(t,z) = L_k(t,z) - L_i(t,z)
```

is affine in `z`. If it is nonnegative at both segment endpoints, then for
`z=(1-lambda)l+lambda r`,

```text
G_ki(t,z)
  = (1-lambda) G_ki(t,l) + lambda G_ki(t,r) >= 0.
```

Therefore an inactive line cannot first undercut strictly inside an owner
segment while staying above both endpoints. Its first contact must occur at:

- the physical near/far endpoint, producing a pair endpoint root; or
- an existing active cut, producing a co-minimal triple root; or
- the whole owner line at once, producing a full-fiber tie, already a common
  near/far root.

### 4.4 Cuts cannot reorder silently

An active cut can leave `[n,f]` only by meeting an endpoint. Two consecutive
active cuts can exchange order only by meeting, at which time the intervening
run has zero length and at least three lines concur. Both cases are in `E`.

Thus the same owners remain active in the same order in a neighborhood of
`t*`. The word is locally constant at every point outside `E`. A locally
constant map into a discrete set is constant on each connected component,
which proves Claim E1.

## 5. Active-boundary certificate: the implementable production reduction

The `O(S^3)` set is a proof oracle. A chart-local compiler can use the current
word and all-site certificates.

Let a witness time have positive-length owners

```text
u_0, u_1, ..., u_{R-1}.
```

For every competitor `k`, certify:

1. `u_0` remains minimal at `n`;
2. `u_{R-1}` remains minimal at `f`; and
3. `k` remains no lower than both owners at every active cut.

For the active cut between `i` and `j`,

```text
z_ij = -B_ij / A_ij,
```

and the oriented owner-minus-competitor difference is

```text
Delta_ik(t,z_ij)
  = [B_ik A_ij - A_ik B_ij] / A_ij.
```

Define

```text
H_{k|ij}(t) = B_ik(t) A_ij(t) - A_ik(t) B_ij(t).
```

The certificate for owner `i` versus competitor `k` is

```text
Delta_ik(t,z_ij) <= 0.
```

Equivalently:

```text
if A_ij > 0: H_{k|ij} <= 0,
if A_ij < 0: H_{k|ij} >= 0,
or without division: H_{k|ij} A_ij <= 0.
```

The sign of `A_ij` must be carried with the numerator. Testing `H` without its
denominator orientation is not an owner certificate.

Per chart this requires:

```text
2(S-1) endpoint comparisons
+ (R-1)(S-2) active-boundary competitor comparisons
+ (R-1) active-denominator guards.
```

This is `O(SR)` and should be called an **active-boundary certificate**, not a
fully output-sensitive algorithm: every active run is still checked against
all `S` sites.

The route to `O(delta R)` is a certified conservative candidate-neighbor
graph of maximum degree `delta`, for example a kinetic regular/Delaunay graph
or a proven conservative Cech-style supergraph. Only certified neighbors could
then be checked at each active boundary. Constructing and maintaining that
graph, and proving it cannot omit a future owner, is a separate cost and a
separate theorem. It must not be hidden inside the `O(delta R)` claim.

### First-event sweep algorithm

```text
input: exact kinetic sites, one ray, [tau_0,tau_1], [n,f]

0. Certify Q_d(t)>0. If Q_d has a root, emit an invalid/unresolved fiber seam.
1. Choose a rational witness in the current open time cell.
2. Build the exact fixed-time positive-length word W.
3. Emit endpoint-owner versus all-competitor predicates.
4. For every active cut, emit H_{k|ij} for all competitors and an A_ij guard.
5. Isolate and exactly cluster all roots after the current time.
6. Take the earliest root bucket; certify W up to that bucket.
7. Recompute the complete fixed-time word on rational witnesses immediately
   to the left and right. Never apply simultaneous pair swaps sequentially.
8. If the one-sided words agree and no event stratum needs storage, coalesce.
   Otherwise emit a half-open seam and its event classification.
9. Repeat to tau_1.
```

Why the next genuine event cannot be missing: until the first word failure,
the current word owns every segment. Section 4.3 forces the first competitor
contact to one of the endpoints being certified. Hence the active-boundary
set contains a polynomial vanishing at that first failure.

## 6. Topology events versus analytic denominator guards

These must be separate in the ABI and in cost reports.

| Condition at `t*` | Geometric meaning | Required action |
| --- | --- | --- |
| `A_ij=0`, `B_ij!=0` | distinct parallel lines; cut at infinity | no topology seam; guard rational division |
| `A_ij=B_ij=0` | full-fiber pair tie | classify activity/material; endpoint predicates also vanish |
| `C_ijk=0`, all required `A` nonzero | finite triple concurrence candidate | check depth and all-site co-minimality |
| `C_ijk=0` because two denominators vanish but intercepts do not | false concurrence at infinity | discard after exact classification |
| active `A_ij` approaches zero while its cut stays compact | necessarily `B_ij` also approaches zero | full-fiber candidate, not a harmless pole |

The current `isolate_kinetic_adjacent_cut_concurrence` rejects a whole interval
when the concurrence polynomial shares any real root with the denominator
product. That prevents a false finite-cut claim, but it is not the final event
algorithm. At a shared root, exact gcd/remainder relations must distinguish:

1. a harmless parallel/infinite-cut artifact;
2. an active or inactive full-fiber tie; and
3. a simultaneous higher-order event bucket.

Likewise, `_reject_kinetic_full_fiber_ties` currently rejects an interval
containing an isolated full-fiber root. The complete compiler should emit and
classify the root. Only an unsupported material/tie policy should make the
result unresolved.

## 7. Ray-direction degeneracy is a separate event family

The pair/triple set is incomplete unless ray validity is an explicit
precondition or event family. Consider one site and

```text
d(t) = (t,0,0).
```

There are no pair or triple predicates, yet at `t=0` the map `z -> x(t,z)`
collapses to one point. Depth order and physical segment length are undefined.

For an affine direction,

```text
Q_d(t) = ||d0+t d1||^2.
```

If `d1 != 0`, any real ray-collapse root is a double root. If `d0=d1=0`,
`Q_d` is identically zero and the entire track is invalid. A production
compiler should either:

- prove `Q_d>0` over every chart; or
- split at its roots and emit an uncovered/unresolved event stratum.

It must not silently bridge the root. Normalized camera directions should make
this rare, but normalization in sampled code is not a continuous-time proof.

## 8. Degenerate and persistent predicates

### 8.1 Globally duplicate lines

If

```text
A_ij identically 0 and B_ij identically 0,
```

the two sites induce the same line for every `(t,z)` on this ray. Choose one
canonical representative under a declared tie rank. If their materials differ,
this is not a geometry-only merge: the tie rule selects which material owns the
fiber, and optimization at this state is nonsmooth.

### 8.2 Persistent parallel families

If `A_ij` is identically zero but `B_ij` is not, the lines are parallel for all
time. Roots of `B_ij` are isolated full-fiber ties and are also roots of both
endpoint gaps. A denominator that is identically zero is therefore not by
itself an unsupported topology: `L_0=0`, `L_1=-t` is the simplest global owner
swap.

### 8.3 Identically zero endpoint predicate

If `N_ij` is identically zero, the pair is tied at `z=n` for all time. This is
a persistent boundary stratum, not infinitely many topology changes. Inward
ownership is decided by the sign of `A_ij` because

```text
Delta_ij(t,z) = A_ij(t) (z-n).
```

If that sign can change, the root is a full-fiber tie and the far predicate
also vanishes there. If both `N_ij` and `F_ij` are identically zero, then
`A_ij` and `B_ij` are identically zero and the lines are global duplicates.

The far-endpoint case is symmetric.

### 8.4 Identically zero triple determinant

`C_ijk` identically zero means a persistent coefficient-space collinearity:
where the cuts are finite, the three lines remain concurrent. It does not mean
the word changes at every time. Example:

```text
L_0=z,  L_1=0,  L_2=-z
```

has persistent triple concurrence at `z=0`, while `L_1` is a permanent
zero-length middle line and the positive-length word is always `[0,2]`.

A correct implementation has two honest choices:

1. support a persistent-concurrence stratum, retain only the extreme-slope
   co-minimal lines, and split when endpoint/full-fiber/other-competitor events
   change that pencil; or
2. report an active persistent concurrence as unresolved.

Blindly asking a root isolator for the roots of the zero polynomial is invalid.
Blindly rejecting every `C identically 0`, including dominated pencils, is safe
but unnecessarily incomplete.

### 8.5 Repeated and grazing roots

Every distinct root must be retained regardless of multiplicity. Even
multiplicity often means a grazing event and unchanged one-sided words, but
multiplicity alone does not establish inactivity, especially in a simultaneous
bucket.

After exact left/right word evaluation:

- coalesce a root whose one-sided positive-length words and event semantics
  agree;
- retain an exact event stratum if its tie/material rule matters; and
- never infer “no event” merely because the polynomial sign did not flip.

### 8.6 Simultaneous roots

Roots from different polynomials must be grouped as exact algebraic event
buckets. Overlapping floating or rational isolating intervals do not prove the
roots equal. Conversely, close numerical estimates do not prove they differ.

For two rational polynomials:

1. compute their exact polynomial gcd;
2. determine whether that gcd has a root in the overlapping isolating region;
3. if not, refine until their intervals are disjoint;
4. if yes, attach both predicates to one algebraic event object.

At a simultaneous event, recompute the whole envelope on each side. Applying
pair swaps one at a time creates an arbitrary, order-dependent result for a
four-way concurrence or endpoint-plus-full-fiber event.

## 9. Right-continuous seams and exact event times

Use event buckets

```text
tau_0 = e_0 < e_1 < ... < e_M = tau_1
```

with ordinary chart words on `(e_r,e_{r+1})`. For dispatch, use the declared
right-continuous convention `[e_r,e_{r+1})`, with the final endpoint handled
separately.

At an exact algebraic seam, the right word can be defined by symbolic
`t=e+epsilon` ordering: inspect the first nonzero Taylor coefficient of each
relevant polynomial, then apply the deterministic site tie rank only if all
orders remain tied. This chooses an exact minimizer at the seam because every
right-limit owner must meet the old envelope continuously there.

Two qualifications are essential:

1. A newborn/dead run has zero length exactly at the event. Either compact it
   out of the formal positive-length seam word or leave it in the transfer
   program only under the proved identity-transfer equivalence. Do not call an
   un-compacted right-limit list the positive-length word.
2. An active full-fiber tie can exchange a positive-length colored region.
   Site-id tie-breaking at the exact time need not equal the right-limit owner.
   The material result may be discontinuous and the geometry derivative may be
   one-sided or undefined. The seam policy must explicitly choose the
   increasing-time owner rather than compare against the fixed-time site-id
   word.

For an irrational event, retain the square-free polynomial and its rational
isolating interval. Comparing a rational request time to the event is done by
refining the interval or evaluating exact signs; converting the root to one
float is not certified dispatch.

## 10. Cheapest counterexamples and falsifiers

The following line fixtures are smaller and more diagnostic than a renderer
smoke. `W_-`, `W_0`, and `W_+` denote exact words at `t=-0.1,0,+0.1`.

### F1: omit near crossings and miss a birth

On `z in [0,1]`:

```text
L_0=0,  L_1=z-t.
W_-=[0], W_0=[0], W_+=[1,0].
```

Only the near endpoint is touched at `t=0`.

### F2: omit far crossings and miss a birth

On `z in [0,1]`:

```text
L_0=0,  L_1=1-z-t.
W_-=[0], W_0=[0], W_+=[0,1].
```

Only the far endpoint is touched at `t=0`.

### F3: check endpoints only and miss an inactive-site insertion

On `z in [-1,1]`:

```text
L_0=z,  L_1=-z,  L_2=-t.
W_-=[0,1], W_0=[0,1], W_+=[0,2,1].
```

`L_2` first touches at the existing interior vertex. The endpoint minima do
not tie. A sweep that checks triples only among already active owners also
misses this event; every competitor must be checked at every active boundary
unless a conservative neighbor graph has been proved.

### F4: treat every equal-slope root as topology and over-split

On `z in [-1,1]`, for `|t|<1`:

```text
L_0=0,  L_1=1+t z.
W(t)=[0], including at A_01(0)=0.
```

The pair is distinct and uniformly ordered at the slope-zero time.

### F5: reject an identically zero denominator and lose a valid owner swap

On any finite depth interval:

```text
L_0=0,  L_1=-t.
W_-=[0], W_0=[0] under site-id ties, W_+=[1].
```

`A_01` is identically zero and `B_01=t`. This is a finite full-fiber event set,
not an interval-wide failure. It also proves that fixed-time site-id tie
selection is not automatically right-continuous.

### F6: accept every cross-product root and invent a finite concurrence

Use the existing three-site fixture in
`test_cross_product_root_at_zero_denominators_fails_closed`:

```text
A_01=A_12=2t,
C=2t(2t^2+7).
```

At `t=0`, both cuts are at infinity and the cross-product root is not a finite
triple cut. The current test correctly prevents a false positive, although a
complete compiler should discard/classify this bucket rather than reject the
whole surrounding interval.

### F7: reject every persistent concurrence and lose a stable valid word

```text
L_0=z,  L_1=0,  L_2=-z.
C_012 identically 0,
W(t)=[0,2] for all t.
```

The middle line is a persistent zero-length stratum.

### F8: discard even-multiplicity roots without checking activity

On `z in [0,1]`:

```text
L_0=0,  L_1=z+t^2.
```

The near predicate has a double root at zero, but the positive-length word is
`[0]` on both sides and at the seam. This fixture should coalesce only after
the exact one-sided check. In contrast, on any depth interval

```text
L_0=0,  L_1=-t^2
```

the one-sided owner is site 1 on both sides, while a lowest-site-id exact tie
would select site 0 at zero. The same even multiplicity therefore carries
material seam semantics unless the declared right-continuous rule selects the
one-sided owner. Together the two fixtures show why multiplicity is a hint,
not a decision.

### F9: omit ray validity

With one site and `d(t)=(t,0,0)`, the pair/triple candidate set is empty but
the fiber collapses at zero. Any theorem missing `Q_d` is false.

### F10: process simultaneous roots sequentially

Generate four lines `L_i=s_i z+c_i t` that all meet at `(t,z)=(0,0)`.
Permute the order in which local swaps are applied. If the resulting right
word changes with permutation, the updater is invalid. A full fixed-time
rebuild on a rational right witness is the oracle.

### Property-based falsifier

For `S=3..6`, generate small integer kinetic site/ray coefficients and:

1. enumerate the universal pair-endpoint/triple candidate roots exactly;
2. cluster algebraically equal roots;
3. choose several rational witness times inside every open root cell;
4. run `discover_kinetic_power_word_at_time` at every witness; and
5. fail if any two words in one cell differ.

Then compare the active-boundary sweep against the global `O(S^3)` oracle. The
first global event that changes the one-sided word must be present in the
current active certificate. This is the cheapest direct falsifier of the
production reduction.

A source-only microcheck using `discover_sparse_line_envelope_word` reproduced
the exact words in F1--F5 and F8. No Metal/MPS/CUDA work was required.

## 11. Implementation consequences for the current source

### Keep

- exact binary64-to-rational pair polynomial construction;
- general square-free/Sturm root isolation and multiplicity;
- exact fixed-time lower-envelope discovery;
- zero-positive-length-run deletion; and
- explicit polynomial retention for irrational root dispatch.

### Change in the future continuous compiler

1. Add `Q_d` validation before topology compilation.
2. Represent roots as shared algebraic event buckets, not independent floats.
3. Generate endpoint-owner and active-boundary all-competitor certificates.
4. Treat active denominator roots as analytic guards, not automatically as
   physical topology events.
5. Replace interval-wide rejection of isolated full-fiber roots with an event
   record and explicit right-limit material policy.
6. Classify shared concurrence/denominator roots instead of rejecting the
   whole interval.
7. Rebuild the entire exact word on both sides of simultaneous events.
8. Preserve an explicit unresolved result for ray collapse or unsupported
   persistent active strata.

### Do not claim yet

- complete continuous-time kinetic chart compilation;
- total handling of persistent active degeneracies;
- `O(delta R)` event discovery without a certified neighbor graph;
- differentiability through event times or topology selection; or
- native/runtime parity from these CPU proofs.

## 12. Current belief and decision

Current belief: the finite-event mathematical spine is now sufficiently
specified for a generic exact compiler. Confidence is high for the universal
pair-endpoint/triple theorem and the `O(SR)` all-competitor certificate, and
medium for the engineering difficulty of algebraic event clustering and
persistent-stratum support.

The important correction is that pair cut denominators are not themselves the
missing physical event family. The missing work is active lower-envelope
certification, exact event bucketing, and explicit seam/degeneracy semantics.
No new frame axis and no per-frame topology table is mathematically required.

The next implementation gate should be the exact global oracle plus F1--F10,
followed by the active-boundary sweep checked against that oracle. Only after
those agree should the kinetic charts be lowered into the native ordered-
transfer backend.
