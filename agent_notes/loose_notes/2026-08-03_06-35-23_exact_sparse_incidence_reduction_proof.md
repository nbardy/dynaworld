# Exact Sparse-Incidence Reduction Proof And Red-Team Note

## Context

This note records the CPU proof/oracle added in:

- `research_experiments/world_foam_lane2/exact_sparse_incidence_oracle.py`
- `research_experiments/world_foam_lane2/test_exact_sparse_incidence_oracle.py`

The immediate question is narrower than temporal atlas approximation: given
exact per-sample ordered-transfer endpoint cotangents, can the native path
accumulate four Mobius coefficient cotangents per referenced `(ray track,
boundary)` and lower them once, instead of atomically scattering five plane
parameter gradients at every endpoint?

The answer is yes under the fixed-word affine-ray/P0 contract. This is an exact
factorization. It does not certify the supplied owner word, and it does not turn
the sampled Chebyshev forward-error gate into a continuous or gradient
certificate.

## Current Model

Current belief:
    Exact sparse-incidence reduction is the right native boundary-VJP seam.

Confidence:
    High for the stated affine-ray, affine-plane, constant-state P0 contract.

Evidence:
    Independent direct implicit-plane and sparse coefficient VJPs match each
    other, PyTorch autograd, and the existing exact streamed word VJP under
    shared cuts, repeated events, track/time blocking, and ray-depth gauge
    rescaling.

Could be wrong if:
    Native topology is stale, endpoint events are emitted with the wrong sign,
    the ray uses a different depth parameterization, or a material model makes
    optical depth depend on more than physical segment length.

## Definitions And Assumptions

For track `p` and chart time `t`, use ordinary affine ray depth `z`:

```text
o_p(t) = o0_p + t o1_p
d_p(t) = d0_p + t d1_p
x_p(z,t) = o_p(t) + z d_p(t)
s_p(t) = ||d_p(t)||
```

For boundary `b = [n, n_t, bias]`:

```text
n . x + n_t t + bias = 0
z_pb(t) = (A_pb + B_pb t) / (C_pb + D_pb t)

A_pb = -(o0_p . n + bias)
B_pb = -(o1_p . n + n_t)
C_pb = d0_p . n
D_pb = d1_p . n
```

Assumptions:

1. The caller supplies a correct, ordered owner word for every chart.
2. Every referenced denominator stays safely nonzero over evaluated samples.
3. Segment coordinate lengths are positive.
4. Density is constant within each P0 owner segment.
5. Camera/ray gradients are optional; disabling them must not remove the
   physical fiber-speed factor from world gradients.
6. Endpoint events and sparse incidence rows are CPU float64 in the oracle;
   native cost accounting separately assumes float32 scalar atomics.

## Derivation 1: Transfer Cotangent To Physical Endpoint Cotangents

For one segment with owner density `rho`, coordinate length
`Delta z = z_right-z_left`, and physical length `L`:

```text
L = s(t) Delta z
tau = rho L
beta = exp(-tau)
```

Let `tau_bar = d loss / d tau`, obtained by the exact prefix-only ordered
transfer VJP. Then:

```text
rho_bar     += L tau_bar
s_bar       += rho Delta z tau_bar
z_left_bar  += -s rho tau_bar
z_right_bar += +s rho tau_bar
```

The `s=||d(t)||` multiplier is required even with `compute_ray_grad=false`.
Without it, a harmless rescaling of ray depth would change boundary and density
gradients.

For total transfer `[beta_total,m_total]`, prefix transfer
`[beta_prefix,m_prefix]`, owner color `c`, and incoming cotangent
`[g_beta,g_m]`, the constant-state prefix identity used by the oracle is:

```text
tau_bar = g_m . (m_prefix + beta_prefix c - m_total)
          - beta_total g_beta
```

No per-frame suffix or per-run reverse tape is needed.

## Derivation 2: Endpoint Cotangent To Mobius Coefficients

Write:

```text
N = A + B t
q = C + D t
z = N/q
```

The exact row Jacobian is:

```text
dz/d[A,B,C,D] = [1/q, t/q, -N/q^2, -tN/q^2]
```

Therefore every finite endpoint event with scalar `z_bar` contributes:

```text
[A_bar,B_bar,C_bar,D_bar] +=
    z_bar [1/q, t/q, -N/q^2, -tN/q^2]
```

All samples, repeated uses, and both sides of a shared cut reduce by addition
into one four-scalar row for each canonical `(track,boundary)` incidence.

## Derivation 3: One Sparse Lowering Equals Direct Plane Scatter

After temporal/event reduction, lower one coefficient cotangent row:

```text
n_bar += -A_bar o0 - B_bar o1 + C_bar d0 + D_bar d1
n_t_bar += -B_bar
bias_bar += -A_bar
```

Substitute the endpoint row from the previous section and use `z=N/q`:

```text
n_bar = -z_bar [o0 + t o1 + z(d0 + t d1)] / q
       = -z_bar [o(t) + z d(t)] / [n . d(t)]

n_t_bar = -z_bar t / [n . d(t)]
bias_bar = -z_bar / [n . d(t)]
```

These are exactly the derivatives obtained by implicit differentiation of:

```text
n . [o(t)+z d(t)] + n_t t + bias = 0
```

The ray part also agrees:

```text
o0_bar += -z_bar n/q
o1_bar += -t z_bar n/q
d0_bar += -z z_bar n/q
d1_bar += -t z z_bar n/q
```

When requested, add the independent fiber-speed cotangent to the direction
coefficients. In no-ray-gradient mode, neither the coefficient ray gradient nor
the speed Jacobian is allocated.

## Shared Cuts And Repeated Incidences

A shared internal cut appears twice per sample: once as the right endpoint of
the front run and once as the left endpoint of the back run. Those cotangents
must not be overwritten or deduplicated before summation. Likewise, a boundary
shared by multiple tracks has multiple canonical incidence rows that finally
scatter into the same five boundary parameters.

The oracle enforces unique incidence rows but intentionally accepts arbitrarily
repeated event references to each row. This matches a CSR-style native program:

```text
event -> incidence id -> four-scalar coefficient adjoint -> boundary id
```

## Ray-Depth Gauge Check

Apply the orientation-preserving reparameterization:

```text
d'(t) = d(t)/lambda
z' = lambda z
near' = lambda near
far' = lambda far
```

Then:

```text
s' = s/lambda
Delta z' = lambda Delta z
L' = s' Delta z' = L
z_bar' = z_bar/lambda
dz'/d boundary = lambda dz/d boundary
```

So prediction, density/color gradients, and boundary gradients remain
unchanged. The adversarial test uses `lambda=13` and also checks that no-ray
mode ignores an intentionally NaN ray-metric tensor, proving it is not read or
retained.

## Logical Atomic And Payload Model

Let:

```text
E = number of finite endpoint events
I = number of canonical (track,boundary) incidences
w = scalar bytes (4 for native float32)
```

Direct boundary scatter:

```text
scalar atomics = 5E
minimum scalar payload = 5Ew
intermediate incidence adjoint = 0
```

Sparse coefficient reduction:

```text
endpoint coefficient atomics = 4E
boundary finalize atomics = 5I
total scalar atomics = 4E + 5I
minimum scalar payload = (4E+5I)w
incidence adjoint payload = 4Iw
```

The sparse path wins this narrow scalar-atomic count exactly when:

```text
E/I > 5
```

This is not a hardware bandwidth prediction. Real atomics move cache lines,
contend, and may be aggregated within a threadgroup. It is a falsifiable logical
count. The tests include both a high-reuse winning case and a low-reuse case
where sparse reduction is correctly reported as more expensive.

## Branches And Backtracks

### Branch A: Direct Five-Scalar Scatter Is Already Best

Why it might be true:
    Short charts or low endpoint reuse make the finalize pass pure overhead.

What would make it false:
    Measured reuse `E/I` is comfortably above five and boundary atomic
    contention dominates.

Cheap test:
    Record `E`, `I`, direct/sparse atomic counts, and kernel timing on one
    frozen topology without changing rendering.

### Branch B: Four-Scalar Global Atomics Are Still Too Expensive

Why it might be true:
    At large frame count, `4E` is still frame-linear.

What would make it false:
    Threadgroup-local or frame-block reduction lowers the global writes to
    approximately `4 * incidence rows touched per block`.

Cheap test:
    Implement the same CPU reduction with event blocks, then compare a native
    local-reduce kernel against per-event coefficient atomics.

### Backtrack: This Does Not Finish The Compiled Atlas

Status:
    Explicitly unresolved.

Evidence:
    The reduction is exact once endpoint events exist. Exact native replay can
    still emit events at `O(P F R)`. Frame-independent expensive world replay
    additionally requires the fixed-`K` compiled transfer atlas or another
    certified temporal chart.

Replacement model:
    Treat sparse incidence reduction as the exact world-VJP backend shared by
    both exact streamed replay and a future compiled-atlas front end.

## Falsification Tests Added

1. Shared internal cuts across two tracks and eleven times: sparse reduction,
   direct implicit scatter, and existing exact streamed VJP agree.
2. Repeated incidence events crossing block boundaries: block sizes `1`, `4`,
   and full-event agree, including ray gradients.
3. Independent PyTorch autograd objective over event depths agrees with both
   boundary and ray reductions.
4. Ray-depth gauge rescaling by `13x` preserves prediction and world VJP.
5. No-ray mode returns no ray tensor and reads no optional ray-metric input.
6. Atomic accounting reports both the high-reuse win and low-reuse loss.

## Decision Implications

If native measurements confirm high reuse and atomic contention:
    Implement four-scalar incidence accumulation plus one boundary finalize.

If reuse is below five:
    Keep direct boundary scatter for that chart or block; the factorization is
    still correct but not cheaper.

If coefficient atomics remain dominant:
    Add local event-block aggregation before global incidence accumulation.

If direct and sparse native results disagree:
    Check endpoint sign, physical fiber speed, incidence identity, and stale
    topology before changing the mathematics.

## Open Questions

1. What is the measured distribution of endpoint events per incidence on real
   Coffee Martini rows?
2. Can native threadgroups aggregate both sides of a shared cut before any
   global atomic?
3. Should topology carry a compiler-issued ownership/certification token so an
   invalid word cannot silently produce a gradient?
4. How should M3/M5 material integrals expose endpoint and material-basis
   cotangents to the same incidence reducer?
5. What continuous or derivative-aware gate should replace the current sampled
   atlas forward-error diagnostic before publication claims?

## Addendum: compiled front end and capability layer now exist

The earlier “future compiled-atlas front end” and missing-token questions are
superseded at source level. Compact template-free schedules, continuous strict
and owner-only training bindings, `J`-node affine-Lie compilation, verified
`O(KJ)` sample-to-node weights, staged sparse incidence finalization, and
right-continuous piecewise topology streaming are implemented and behavior-
tested. The training binding permits mutable P0 material only; strict frozen
evaluation retains transfer/Jacobian certification.

The remaining questions are now narrower:

1. Does rebuilt native timing confirm the interaction model and high-reuse
   incidence benefit on realistic large-run words?
2. Can the production compiler serialize exact algebraic endpoints and reuse
   topology across real camera/sample batches without excessive refresh?
3. What geometry trust region or recertification schedule supports site motion?
4. What are the measured barycentric dense-fallback and certificate-fallback
   fractions?
5. Do adaptive M3/M5 materials improve real heldout quality enough to justify
   their integration?

No accelerator workload was used for this addendum; native parity, allocator
peaks, and public quality remain open.
