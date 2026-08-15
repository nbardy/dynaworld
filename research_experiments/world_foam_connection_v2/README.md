# WorldFoam Connection V2 Reference Implementation

This directory is an isolated, pure-Torch implementation for the constrained
Lagrangian ray-fiber optical connection. It does not import or modify the
production WorldFoam or World Tubes renderers.

It is "full" at the bounded research-oracle level: algebra, the stable-P0
theorem kernel, shared scalar flow, endpoint integration, selected
directions, temporal representations, fitting, analytic fixtures, end-to-end
comparison, and a JSON CLI are present. It is not a native renderer or a
publication performance result.

## Convention and implemented equations

An affine transfer is `U=(beta,m)` and acts as `q -> m + beta q`. Repository
order is near to far:

```text
U(b,a) = U(s,a) U(b,s)
compose(front, back) = front(back(q)).
```

A P0 generator is

```text
A_z = X(-lambda, eta),
lambda = rho ||partial_z Gamma_x||,
eta = lambda c.
```

The caller must include this physical ray-speed Jacobian in `lambda` and
`eta`. The core evaluates the exact P0 segment exponential and exact analytic
time derivative of the complete moving word.

For explicit sampled scalar flow `w`, the constrained connection is

```text
A_t = -w A_z,
F_bulk = partial_t A_z + (partial_z w) A_z.
```

At an internal moving interface `r(t)`, both one-sided flow traces are kept:

```text
F_singular = ([w A_z] - r_dot [A_z]) delta_r.
```

Bulk terms use configurable-order Gauss--Legendre integration. Every singular
term uses its exact repository-order sandwich:

```text
U(r,a) ([w A_z] - r_dot [A_z]) U(b,r).
```

For moving clipped endpoints,

```text
B_near = (a_dot - w(a+)) A_z(a+),
B_far  = (b_dot - w(b-)) A_z(b-),
dU/dt  = U B_far - B_near U + K_raw.
```

The endpoint histories scan exact piecewise-constant generators according to
`dot H = H B`. The corrected transfer, curvature source, and reconstruction
are

```text
U_tilde = H_near U H_far^{-1},
K_F     = H_near K_raw H_far^{-1},
U       = H_near^{-1} U_tilde H_far.
```

The implementation returns both:

- the theorem prediction above; and
- an autograd-independent analytic derivative of the exact moving P0 word.

Their difference is the primary convention/BV correctness residual. A second
algebraic residual compares `K_F` with the transformed covariant derivative.
Because that second path shares algebra with the theorem, the oracle also
finite-differences a freshly rebuilt neighboring-time
`H_near U H_far^{-1}`; that is the independent endpoint-sign/order check.

## Public tensor shapes

For `R` ordered runs, `Q` quadrature nodes, and `K` endpoint-history steps:

| Tensor | Shape |
| --- | --- |
| `P0Ray.cuts` | `[R+1]` |
| `P0Ray.extinction` | `[R]` |
| `P0Ray.emission_density` | `[R,3]` |
| `P0RayRate.cut_velocity` | `[R+1]` |
| `P0RayRate.extinction_time` | `[R]` |
| `P0RayRate.emission_density_time` | `[R,3]` |
| `P0FlowSamples.bulk_value`, `bulk_d_dz` | `[R,Q]` |
| `P0FlowSamples.cell_left_value`, `cell_right_value` | `[R]` |
| endpoint duration/scalar arrays | `[K]` |
| endpoint source arrays | `[K,3]` |
| transfer/generator/tangent vectors | `[...,4]` |

Use `sample_shared_chebyshev_flow(...)` for a complete chain rule through
moving cuts, quadrature nodes, time, and shared-flow coefficients. The lower
theorem core still accepts evaluated tensors rather than a callback.

## Main API

- `evaluate_connection(...)`: validates input and returns the complete core,
  physical cone reports, group-completion conditioning, and optional flow
  admissibility report.
- `evaluate_connection_core(...)`: differentiable theorem calculation without
  discrete pass/fail policy.
- `evaluate_selected_direction(...)`: one explicit `torch.func.jvp` through
  every primary four-vector observable.
- `ordered_p0_transport(...)`, `direct_p0_time_derivative(...)`,
  `integrate_bulk_curvature(...)`, `integrate_singular_curvature(...)`, and
  `scan_endpoint_transports(...)`: individually testable mathematical seams.
- `diagnose_sensor_depth_lift(...)`: solves the full
  `[Gamma_u Gamma_v Gamma_z]` lift and compares it with the best and supplied
  scalar axial flows.
- `derive_constant_endpoint_history(...)` and
  `derive_piecewise_constant_endpoint_history(...)`: derive endpoint state
  from the same flow/cut inputs and return an explicit byte/state receipt.
- `SharedChebyshevFlow`: a bounded, globally shared flow with no frame or ray
  axis. `evaluate_shared_flow_selected_direction(...)` differentiates through
  its cut-dependent resampling.
- `compile_equal_family_representation(...)`: compares physical `U`, group
  `U_tilde`, and signed `K_F` only after reconstructing the same physical
  transfer. Endpoint values, the `K_F` base, and shared-flow bytes are charged.
- `TrainableConnectionAtlas`: differentiable fixed-node A0/A1/A2 fitting;
  additive `K_F` fails closed before `beta_tilde<=0`.
- `run_reference_oracle(...)`: all bounded correctness fixtures, independent
  neighboring-time `U_tilde` derivative, JVP/finite-difference checks,
  holonomy orientation, lift diagnostics, and the A0/A0c/A1/A2 comparison.

## Representation experiment

The reference keeps four named controls separate:

| Variant | Stored temporal object | Extra state |
| --- | --- | --- |
| A0 | physical `U` | none |
| A0c | physical `U` | capacity-matched shared flow, intentionally unused |
| A1 | unrestricted `U_tilde` | flow plus near/far endpoint transports |
| A2 | signed `K_F=dU_tilde/dt` | flow, endpoints, and one base `U_tilde(t0)` |

Every candidate uses one adaptive piecewise-linear family and is scored after
physical reconstruction. The current certificate is explicitly a
primal-plus-temporal-secant probe certificate. It records
`selected_parameter_tangents_certified=false`,
`complete_work_accounting=false`, and `promotion_eligible=false`; the core JVP
tests do not silently upgrade the atlas certificate.

## Safe-host commands

Do not run these on the incident-prone local Mac. On a quiet CPU/CUDA host:

```bash
PYTHONPATH=research_experiments uv run --with pytest python -m pytest \
  research_experiments/world_foam_connection_v2/tests -q

PYTHONPATH=research_experiments uv run python -m \
  world_foam_connection_v2.run_oracle \
  --require-reference-gates \
  --output outputs/world_foam_connection_v2/reference_oracle.json
```

The CLI is deterministic float64 CPU work. It does not launch MPS, CUDA,
training, or a native build.

## Fail-closed distinctions

- Direct and reconstructed `U` are checked against the bounded-RGB physical
  cone `0<beta<=1`, `0<=m_c<=1-beta`.
- `U_tilde` and endpoint transports are checked only as unrestricted affine
  group elements with `beta>0`; they need not be physical.
- `K_F` is a signed tangent and is never passed through a transfer cone.
- Flow provenance/capacity is an explicit declaration. It checks temporal
  degrees of freedom and, when a source-motion byte budget is supplied,
  retained bytes. The reference cannot infer whether upstream code lied about
  target conditioning or answer tables.
- Discontinuous flow is allowed for the general BV diagnostic but fails the
  continuous-flow admissibility report.

## Deliberate boundary

- The implemented connection is one stable P0 owner word on one scalar-depth
  track. Birth/death, root isolation, simultaneous events, and chart stitching
  remain external compiler responsibilities.
- The full `(v_u,v_v,w)` sensor-depth lift is diagnosed, but only its scalar
  depth component drives this track kernel. Generic transverse camera/object
  motion needs a future sensor-time patch implementation.
- Gauss--Legendre and temporal-atlas certificates are finite probe checks, not
  continuous interval proofs or certified polynomial extrema.
- The shared-flow JVP includes cut/time/coefficient resampling. A compact
  multi-time endpoint atlas with all training gradients still needs a native
  implementation; a per-request P0 endpoint history is explicitly charged
  and disqualified from memory-light claims.
- There is no event compiler, renderer integration, Metal/CUDA kernel,
  production trainer, measured timing, or paper claim here.
- Native promotion stays closed until complete selected-tangent certificates,
  complete bytes/work, at least `2x` improvement against both A0 and A1, and
  at least 20% measured request-time improvement survive real charts.

No runtime validation was executed on the memory-constrained local host. The
source and test contracts exist; their first execution belongs on the safe
host above.
