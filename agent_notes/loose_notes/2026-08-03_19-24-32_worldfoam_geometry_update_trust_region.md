# WorldFoam Geometry-Update Trust Region: Exact Event-Free Theorem

## Context

The direct affine kinetic WorldFoam frontend can compile exact continuous
owner charts and the stable-stratum VJP can differentiate node physical
lengths while a supplied owner word, chart partition, node schedule, and rank
remain fixed.  What was missing was a sound answer to a narrower optimizer
question:

> For how far along one geometry/ray update direction can an existing
> structural result be reused without silently crossing an owner-topology
> event?

The existing node-local margins in
`research_experiments/world_foam_lane2/kinetic_stable_stratum_vjp.py` do not
answer this.  Positive margins at finitely many compiler nodes cannot exclude
a denominator zero, competitor contact, or ray collapse between those nodes.

This session added:

- `research_experiments/world_foam_lane2/kinetic_geometry_trust_region.py`
- `research_experiments/world_foam_lane2/test_kinetic_geometry_trust_region.py`

The implementation is CPU-only and changes no native or shared shader source.

## Backtrack: Frozen Event Endpoints Do Not Have a Generic Nonzero Radius

Status:
    prior broad trust-region idea weakened

Suppose an active chart seam is the simple root `t_*` of an event predicate
`P(t,theta)=0`.  Under a parameter update `theta(e)=theta_0+e delta`, implicit
differentiation gives, when defined,

```text
dt_*/de = -P_e(t_*,theta_0) / P_t(t_*,theta_0).
```

Generically this is nonzero.  Therefore the old numeric event time is stale
for every nonzero update, however small.  Reusing the old numeric seam as if
it remained exact has zero generic radius.

This does **not** imply that the combinatorial sequence of owner words has
zero radius.  A simple event can persist and move while retaining its order
relative to neighboring events.  But that broader result must re-isolate or
refit the event endpoint after the update.  It is not a frozen-program
derivative and is not implemented here.

Decision:

- active or multichart programs return an explicit zero radius and request
  event-root re-isolation;
- endpoint events return zero because a one-sided structural policy is not
  implemented;
- the nonzero theorem covers exactly one event-free owner chart over the
  complete closed time domain.

## Current Model

### Parameter path

For step scalar `e`:

```text
p_i(t,e) = p_i0 + t v_i + e (delta_p_i0 + t delta_v_i)
w_i(t,e) = sum_k (w_ik + e delta_w_ik) t^k,  k <= 2
o(t,e)   = o0 + t o1 + e (delta_o0 + t delta_o1)
d(t,e)   = d0 + t d1 + e (delta_d0 + t delta_d1).
```

Every binary64 input is converted to its exact rational value.  The theorem
concerns the exact rational path above, not an unspecified sequence of
floating-point optimizer operations.

For a rounded optimizer proposal, the safe adapter reads both stored
binary64 endpoints, forms their exact rational difference, and certifies the
segment between those actual endpoints.  If a smaller suggested line-search
step is rounded into a new endpoint, that endpoint must be certified again.

### Pair powers

Along `x=o+z d`, the common quadratic term in `z` cancels between sites:

```text
power_i - power_j = A_ij(t,e) z + B_ij(t,e),
A_ij = 2 (p_j-p_i) dot d,
B_ij = 2 (p_j-p_i) dot o
       + ||p_i||^2 - ||p_j||^2 - w_i + w_j.
```

Both `A` and `B` have bidegree at most `(2,2)` in `(t,e)` for the current
affine kinetic family.

Let the certified word be `o_0,...,o_{R-1}`.  Its internal cuts are

```text
z_q = -B_{o_{q-1},o_q} / A_{o_{q-1},o_q},  q=1,...,R-1.
```

## Exact Sufficient Conditions

The new certificate constructs strict positive polynomials for all of the
following conditions over `(t,e) in [t_min,t_max] x [0,r]`.

### 1. Ray noncollapse

```text
||d(t,e)||^2 > 0.
```

### 2. Active cut denominators

At the base time anchor, record the exact sign `s_q in {-1,+1}` of every
active denominator.  Prove

```text
s_q A_q(t,e) > 0
```

continuously.  This excludes poles and makes every rational cut well-defined
with fixed denominator sign.

### 3. Positive and ordered runs

For the first and last cuts:

```text
z_1 - near = -(B_1 + near A_1) / A_1 > 0,
far - z_last = (B_last + far A_last) / A_last > 0.
```

After using the certified denominator signs, these become strict polynomial
sign tests.

For consecutive cuts:

```text
z_{q+1} - z_q
  = (B_q A_{q+1} - B_{q+1} A_q) / (A_q A_{q+1}) > 0.
```

Again the known denominator signs reduce this to one strict polynomial sign.

### 4. All-site owner dominance at run endpoints

For run owner `i` and competitor `k`, define

```text
D_ik(z) = A_ik z + B_ik = power_i-power_k.
```

At `near` and `far`, prove `D_ik < 0` directly.  At an internal cut
`z_c=-B_c/A_c`:

```text
D_ik(z_c) = (B_ik A_c - A_ik B_c) / A_c < 0.
```

The defining adjacent owner is the only allowed equality at that endpoint.
Every other site must have a strict positive gap.  Since `D_ik` is affine in
depth, nonpositivity at both endpoints implies nonpositivity throughout the
run; strictness at the non-defining endpoint gives a unique positive-length
owner interior.  Ordered positive cuts plus these all-site inequalities prove
that the word is the complete lower envelope.

This direct word theorem also excludes hidden full-fiber ties.  An adjacent
tie would contradict its nonzero cut denominator; any other positive-length
tie would contradict a strict endpoint gap.

### Theorem boundary

Claim:
    If every constructed strict polynomial is positive on the time/step
    rectangle, then the same unique owner word is valid for every time and
    every step in the certified interval.

Assumptions:

- one affine ray track;
- affine 3D site positions, degree-at-most-two time weights;
- finite fixed near/far with near < far;
- exactly one base owner chart and no active or endpoint event;
- strict unique-owner topology on the closed time domain;
- exact rational interpretation of binary64 inputs and the supplied path.

Not claimed:

- unchanged numeric endpoints of an active/multichart program;
- event-time, chart-endpoint, node-time, rank, or compiler-choice derivatives;
- projective or nonlinear ray tracks;
- bounded-cell sphere/vacuum events;
- a maximum possible topology radius;
- safety of a differently rounded optimizer endpoint.

## Exact Bernstein Perturbation Certificate

Every target predicate is represented as

```text
P(t,e) = P_0(t) + e P_1(t) + ... + e^m P_m(t).
```

The time interval is recursively subdivided.  On every leaf, exact
power-to-Bernstein conversion gives coefficients whose convex hull contains
the polynomial range.  If every Bernstein coefficient of `P_0` is positive,
then that leaf has an exact lower bound.  Across all leaves define

```text
m_0 = minimum base Bernstein coefficient > 0.
```

For each update coefficient, use the same leaves and define

```text
M_b = maximum absolute Bernstein coefficient of P_b.
```

The Bernstein hull property gives, for every time and `0 <= e <= r`,

```text
P(t,e) >= m_0 - sum_{b=1}^m M_b r^b.
```

Therefore the exact sufficient radius condition is

```text
sum_b M_b r^b < m_0.
```

The left side is monotone for `r >= 0`.  The code first checks the requested
radius.  If it fails, a strictly positive seed radius follows from

```text
r <= 1,
r <= m_0 / (2 sum_b M_b),
```

because then `sum_b M_b r^b <= m_0/2`.  Exact rational bisection expands this
seed toward the largest radius accepted by this coefficient bound.  The
returned radius is not claimed to be the true maximal topological radius.

Polynomial bidegrees are small:

- ray speed, fixed endpoint gaps, and denominators: at most `(2,2)`;
- internal-cut competitor gaps and adjacent-cut ordering numerators: at most
  `(4,4)`.

The number of predicates is `O(R S)` for `R` word runs and `S` sites.  There
is no requested-frame axis.

## Adversarial Cases

### Near optimizer root

A dominated site lies only `1/1024` above the active envelope and its weight
update lowers it at unit speed.  The exact certificate returns a positive
radius strictly below `1/1024`; replay inside preserves the word, while replay
beyond the root changes it.

### Active denominator collapse

Two sites define a valid cut at the base point.  Moving one site onto the
other collapses the cut denominator and creates a full-fiber tie at step one.
The certified radius is positive but strictly below one.  Replay after the
collapse has a different owner word.

### Inactive competitor

Every run endpoint is checked against every site, not merely the active
neighbors.  A previously dominated third site moving into the envelope limits
the radius and is present in the emitted predicate records.

### Interior grazing root

`L_0=0` and `L_1=z+t^2` have the same one-owner word on both sides of `t=0`,
but tie at `(t,z)=(0,0)`.  Compiler nodes away from zero can have positive
margins.  The exact continuous certificate rejects even a zero update because
the strict near-owner predicate has an interior root.  This is intentional:
node margins are not continuous proof.

### Simultaneous event

The existing active compiler rejects an ambiguous simultaneous active event.
The trust-region layer propagates that failed base proof as zero radius; it
does not invent an event policy.

### Ray collapse

An update driving the ray direction to zero at step one gets a positive radius
strictly below one from `||d||^2`; the singular step is never certified.

## Branches Not Implemented

### Branch A: Combinatorial multi-chart persistence

Hypothesis:
    A useful nonzero radius can preserve event identity/order while allowing
    numeric event times to move.

Required proof components:

1. reconstruct every relevant active-certificate polynomial under the update;
2. place each base event in a disjoint rational neighborhood;
3. prove opposite endpoint signs and a fixed nonzero `dP/dt` throughout each
   event neighborhood and update interval, giving exactly one simple root;
4. prove every source is root-free on the complement neighborhoods, so no new
   event appears;
5. prove root neighborhoods remain disjoint and ordered;
6. prove one stable witness word in every root-complement cell;
7. re-isolate/refit each moved root before dispatch.

Why it was not implemented:
    The current program output does not retain a canonical complete source
    registry for this two-parameter proof, and a partial active-source check
    would be unsound.  Exact frozen endpoints are not a substitute.

Cheap falsification test:
    Perturb a scene with one simple active event.  If the old event time is
    reused, evaluate its defining polynomial after the update; generically it
    is nonzero immediately.

### Branch B: Fixed-query sample stability

Individual query times away from seams can receive much larger local radii by
certifying only their fixed-time words.  That can be useful for a sampled
training batch, but it is not a continuous compiler-program certificate and
must not be labelled as one.

### Branch C: Rounded optimizer arithmetic envelope

One could add IEEE-754 multiplication/addition error intervals around the
directional path.  The simpler sound implementation instead certifies the
actual stored binary64 candidate endpoint.  A future fused optimizer needs
either that post-rounding endpoint check or a formally validated floating
interval implementation.

## API Semantics and Stop Rules

- `passed=True` means a positive radius was proved.
- `requested_radius_certified=True` means the entire requested update is safe.
- If `passed=True` but the requested radius was limited, backtrack; do not
  apply the original full proposal.
- `recompile_required` is true whenever the requested radius is not certified.
- Any newly rounded backtracked endpoint must be certified again.
- Active/multichart programs must re-isolate/refit roots; do not weaken this
  gate to a node-margin check.

## Validation

Focused adversarial coverage exercises:

- exact zero-direction reuse;
- near-root radius limitation and directional replay;
- denominator collapse and replay beyond the radius;
- inactive competitor entry;
- active/multichart and simultaneous-event rejection;
- interior grazing tie rejection;
- requested-radius monotonicity;
- exact stored-binary64 candidate certification; and
- ray-direction collapse.

Final CPU gate:

```text
PYTHONPATH=research_experiments/world_foam_lane2 \
  uv run --with pytest python -m pytest \
  research_experiments/world_foam_lane2/test_kinetic_*.py -q

101 passed
```

Ruff check and format-check passed for the new module and tests.

No Metal, MPS, CUDA, dataset, or trainer workload was launched.
