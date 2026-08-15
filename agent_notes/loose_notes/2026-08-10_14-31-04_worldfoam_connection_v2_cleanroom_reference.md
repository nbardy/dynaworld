# WorldFoam Connection V2 Clean-Room Reference

Date: 2026-08-10 KST

Status: source-complete bounded reference; static-audited locally; runtime
execution intentionally deferred to a safe host

## Context and decision

The user made two explicit decisions:

1. Stop changing the nearly finished World Tubes paper and hand it to another
   owner.
2. Start a separate implementation of the latest constrained-Lagrangian
   ray-fiber optical-connection algorithm from scratch.

The paper split is now durable in:

- `research_notes/world_tubes_paper_completion_handoff_2026-08-10.md`;
- the `research_notes/README.md` index; and
- the new isolated source namespace
  `research_experiments/world_foam_connection_v2/`.

No existing World Tubes, Projective STAR, WorldFoam, Gate4, or PowerFoam
runtime was imported into the new implementation. The new branch is not part
of Paper A and cannot silently alter its claim or experiment queue.

The local Mac has suffered previous memory/resource incidents. This session
therefore ran no Python interpreter, import, pytest, build, Metal, MPS, CUDA,
training, or oracle workload. Only source reads, patches, and shell-level
static scans were allowed. Every numerical statement below is an implemented
contract or a gate to run later, not a newly observed result.

## Current mathematical model

An optical transfer is the affine map

\[
T(\beta,m):q\mapsto m+\beta q,
\qquad \beta>0,\quad m\in\mathbb R^3.
\]

Repository composition order is near to far:

\[
T_1T_2
=T(\beta_1\beta_2,m_1+\beta_1m_2),
\qquad
U(b,a)=U(s,a)U(b,s).
\]

For one P0 run,

\[
A_z=X(-\lambda,\eta),
\qquad
\exp(\ell A_z)
=T\left(e^{-\lambda\ell},
\frac{1-e^{-\lambda\ell}}{\lambda}\eta\right).
\]

`lambda` and `eta` must already include the physical camera-ray speed
Jacobian. The reference does not guess or hide that conversion.

Choose a bounded shared scalar depth flow `w(t,z)` and constrain the horizontal
connection by

\[
A_t=-wA_z.
\]

Then, inside a P0 run,

\[
F^R_{tz}
=\partial_tA_z+\partial_z(wA_z)
=\partial_tA_z+(\partial_zw)A_z.
\]

At a moving interface `z=r(t)`, with one-sided traces retained,

\[
F_{\rm sing}
=\left([wA_z]-\dot r[A_z]\right)\delta_{z=r}.
\]

For moving clips `a(t),b(t)`, define

\[
B_a=(\dot a-w(a^+))A_z(a^+),
\qquad
B_b=(\dot b-w(b^-))A_z(b^-).
\]

The corrected repository-order theorem implemented by the core is

\[
\frac{dU}{dt}
=UB_b-B_aU
+\int_a^b U(s,a)F^R_{tz}(s)U(b,s)\,ds.
\]

Endpoint transports solve right-acting ODEs

\[
\dot H_a=H_aB_a,
\qquad
\dot H_b=H_bB_b,
\]

and define

\[
\widetilde U=H_aUH_b^{-1},
\qquad
K_F=\frac{d\widetilde U}{dt}
=H_a\left[\int U(s,a)F(s)U(b,s)\,ds\right]H_b^{-1}.
\]

Reconstruction is

\[
U=H_a^{-1}\widetilde U H_b.
\]

The physical contraction/color cone applies to direct and reconstructed `U`:

\[
0<\beta\le1,
\qquad
0\le m_c\le1-\beta.
\]

`U_tilde` lives only in the affine group completion `beta_tilde>0`; it may
amplify and have signed moments. `K_F` is a signed tangent, not a transfer.

## What was implemented

### Exact algebra and stable-P0 theorem core

`affine.py` implements:

- immutable affine transfer, generator, and tangent records;
- repository-order compose, scan, inverse, and radiance application;
- stable affine-group exponential and its exact time derivative;
- associative first jets;
- order-sensitive generator and tangent sandwiches;
- physical-cone and unrestricted-group diagnostics.

`connection.py` implements:

- exact P0 segment/prefix/suffix/total transfer;
- an analytic moving-word `dU/dt` independent of autograd;
- arbitrary-order Gauss--Legendre bulk curvature integration;
- the full two-trace BV atom `[wA]-r_dot[A]`;
- moving near/far endpoint flux;
- exact piecewise-constant endpoint scans `H <- H exp(dt B)`;
- `U_tilde`, `K_F`, reconstruction, and theorem residuals;
- sampled flow/capacity/noncrossing diagnostics; and
- full sensor-depth lift diagnostics against the scalar axial approximation.

The field formerly called an "independent covariant derivative" was renamed
`algebraically_transported_covariant_derivative`. It shares algebra with the
theorem and is not enough to catch an endpoint scan sign bug by itself.

### Shared-flow chain rule

`shared_flow.py` implements a bounded global Chebyshev flow with coefficient
shape `[D_t+1,D_z+1]`. It has no requested-frame or ray axis. Queries now fail
outside the declared `(t,z)` domain.

`shared_flow_connection.py` supplies the missing end-to-end chain rule:

1. cuts define Gauss--Legendre depths;
2. those depths move under a geometry direction;
3. the shared flow is resampled at bulk and one-sided endpoint sites;
4. `torch.func.jvp` covers cuts, material, rates, time, flow coefficients,
   and endpoint-history tensors in one transform.

This closes the scalar-track resampling omission in the lower explicit-sample
ABI. It does not implement the full transverse `(v_u,v_v,w)` sensor-time patch.

### Endpoint provenance

The generic tensor core must accept an explicit endpoint history to remain a
pure map, but arbitrary history tensors can hide a per-frame answer.

`endpoint_history.py` therefore adds two sealed builders:

- a one-step exact constant-generator history derived from the same
  ray/rate/flow inputs; and
- a charged left-sampled P0 time history with one explicit snapshot per
  interval.

Both return scalar counts and retained bytes. A multi-step history is labelled
`uses_requested_frame_table=true`, so it cannot support a memory-light claim.

### Three ABIs and the required controls

`temporal_atlas.py` provides the common adaptive piecewise-linear family and
stable charts:

- physical log-cone chart for `U`;
- unrestricted `beta>0` log chart for `U_tilde`;
- raw signed four-vector for `K_F`.

`representation_benchmark.py` evaluates four named rows:

| Row | Object | Charged auxiliary state |
| --- | --- | --- |
| A0 | direct physical `U` | none |
| A0c | direct physical `U` | capacity-matched flow, intentionally unused |
| A1 | group `U_tilde` | flow plus near/far endpoint values |
| A2 | signed `K_F` | flow, endpoints, and base `U_tilde(t0)` |

All rows use the same node family and are evaluated only after reconstructing
the same physical `U`. The receipt includes node/flow/base/endpoint bytes and
separate ordered-run, flow, endpoint, reconstruction, cone, and group work
components.

The certificate was deliberately narrowed after red-team review. It says:

```text
probe_grid_only=true
probe_primal_secant_verified=<runtime result>
selected_parameter_tangents_certified=false
complete_work_accounting=false
canonical_primal_tangent_verified=false
promotion_eligible=false
```

It cannot be presented as the paper's continuous primal/tangent certificate.

`fitting.py` supplies a small Adam reference optimizer for A0/A1/A2. Additive
`K_F` integration now fails before reconstruction if any represented or
endpoint `beta` crosses the declared positive group boundary. Crossing is not
silently treated as a cone penalty.

### Fixtures and independent paths

`fixtures.py`, `holonomy.py`, and `oracle.py` cover:

- front-red/back-blue noncommuting order;
- the pinned moving interface `r(t)=1+t`, `w=0`,
  `A^-=X(-1,e_R)`, `A^+=X(-2,2e_B)`;
- advected differently colored slabs with moving clips;
- `r_dot!=w` boundary mismatch;
- material evolution with bulk-only curvature;
- discontinuous flow requiring the full `[wA]` atom;
- flat translation with fixed clips, zero curvature, and nonzero endpoint
  flux;
- near-only and far-only endpoint motion;
- noncommuting multi-step endpoint scan order;
- vacuum;
- a constant noncommuting closed plaquette with an orientation receipt;
- sideways pinhole motion that has a full sensor-plane lift but no scalar
  depth-flow lift; and
- nonzero cosine-depth curvature whose ordered transported integral cancels.

The cosine fixture is intentionally not a flat connection. It uses

\[
A_z(t,z)=\left(1+\epsilon t\cos(2\pi z)\right)X_0,
\qquad w=0,
\]

so

\[
F=\epsilon\cos(2\pi z)X_0\ne0
\]

pointwise while the complete ordered transported integral over `[0,1]`
vanishes. This catches the false converse "constant total transfer implies
zero curvature."

The endpoint-sign/order check does not reuse the algebraic covariant path. It
rebuilds `H_a U H_b^{-1}` at neighboring times and central-differences that
fresh corrected transfer against `K_F`.

The oracle also central-differences all named selected-direction observables,
including a second JVP through moving flow-sampling sites.

### CLI

`run_oracle.py` emits one deterministic JSON report and optionally exits 2
when a bounded reference gate fails. It never promotes a native runtime. The
promotion record has `measured_time_improvement=null` and remains closed even
if all correctness fixtures pass.

## Branches considered and rejected

### Rename the method around holonomy

Rejected. An open camera ray is parallel transport. Holonomy is used only for
the explicit closed ray-time rectangle.

### Replace `U` immediately with `K_F`

Rejected. Flat coherent motion can make `K_F=0`, but A1 `U_tilde` may already
be equally compact and cheaper to compile/reconstruct. A2 must beat A0 and A1
after all state and gradients are charged.

### Let a free per-ray flow flatten the answer

Rejected. The only trainable flow here is a small global Chebyshev field with
an explicit capacity receipt. Per-ray, per-frame, target-conditioned, or
transfer-conditioned answer tables fail provenance.

### Call a dense probe certificate continuous

Rejected. Every atlas result is labelled probe-grid only. Continuous bounds
and selected parameter tangents remain separate gates.

### Treat observed flow slopes as certified extrema

Rejected. `FlowAdmissibilityReport` now records
`probe_grid_only=true` and `continuous_bound_certified=false`.

### Hide endpoint time state outside the payload

Rejected. Endpoint nodes are stored inside A1/A2 representation values, and
generic P0 histories have an explicit retained-state receipt.

### Integrate signed `K_F` through `beta_tilde=0`

Rejected. Both the fitting and representation paths fail the group gate.

## What is genuinely new here

The ray-fiber connection theorem and its geometric motivation came from the
scientist memo and the subsequent repo-order mathematical audit. This session
does not claim independent invention of that theorem.

The implementation work did add project-specific engineering formulations:

- the exact repository-order differentiable P0 ABI;
- a sealed endpoint-history derivation and state receipt;
- the A0/A0c/A1/A2 equal-family comparison after physical reconstruction;
- complete scalar shared-flow resampling JVPs;
- a neighboring-time independent corrected-transfer derivative; and
- a fail-closed distinction between physical transfer, group element, and
  signed tangent throughout fitting and certification.

Those are implementation/research-design contributions, not yet paper claims.

## Falsification and promotion rules

The reference branch fails if any of these occur on the safe host:

- theorem or reconstructed-transfer residual above `1e-9` in conditioned
  float64 fixtures;
- wrong front/back order or plaquette sign;
- failure to reject discontinuous flow under the continuous policy;
- scalar flow falsely explains sideways pinhole motion;
- additive A2 crosses `beta_tilde<=0` without failure;
- selected material or shared-flow/cut JVP differs from central difference by
  more than `1e-6` in the bounded tests;
- A1/A2 reconstructed physical transfer violates the cone; or
- a source/API test cannot import and execute.

Native runtime promotion has stricter gates:

- the same continuous primal and selected-tangent norm for all variants;
- complete endpoint/flow/reconstruction/conditioning/gradient bytes and work;
- at least `2x` retained payload and ordered work improvement against both A0
  and A1;
- at least 20% measured request-time improvement;
- no per-frame or per-ray flow/endpoint answer table;
- benefit survives geometry/camera directions; and
- representative real nontrivial stable charts, not only the flat identity
  fixture.

Failure keeps curvature as a useful theorem/correspondence diagnostic and
kills the native A2 runtime branch. It does not reopen Paper A.

## Safe-host execution gate

On a quiet host, from the dynaworld root:

```bash
PYTHONPATH=research_experiments uv run --with pytest python -m pytest \
  research_experiments/world_foam_connection_v2/tests -q
```

Then:

```bash
PYTHONPATH=research_experiments uv run python -m \
  world_foam_connection_v2.run_oracle \
  --require-reference-gates \
  --output outputs/world_foam_connection_v2/reference_oracle.json
```

Review the JSON before increasing probe count or connecting a renderer. Do not
launch MPS on the local incident-prone Mac. CUDA/B200 is appropriate only
after the CPU float64 oracle passes and a separately validated CUDA/native
implementation exists.

## Remaining work after the first runtime gate

1. Fix any bounded source/test failure without touching Paper A.
2. Add a continuous interval certificate for reconstructed physical `U` and
   selected parameter actions.
3. Replace per-query endpoint histories with a compact differentiated endpoint
   atlas and complete state/work accounting.
4. Extend from one scalar-depth track to a sensor-time patch using the full
   `(v_u,v_v,w)` lift, or explicitly limit the method to axial-compatible
   charts.
5. Integrate one isolated native kernel only if A1/A2 passes the mathematical
   and payload/work gates.
6. Measure real request time and gradients on a clean host.
7. Start a separate WorldFoam/connection paper only if the empirical quotient
   survives those gates.

## Final static-audit closure

The last source-only audit found five bounded issues, all repaired before
handoff:

- endpoint generator construction now validates `[R,Q]` flow tensors directly
  and no longer builds a Gauss--Legendre eigensystem merely to inspect shape;
- flow admissibility reports both temporal-capacity and retained-byte ratios,
  and fails capacity when an explicitly supplied source-motion byte budget is
  exceeded;
- affine tangent composition/actions now reject mixed dtype or device metadata
  consistently with the generator ABI;
- negative-time calls to the two endpoint-history fixtures fail at the fixture
  boundary; and
- tiny A0 and A2 optimizer contracts now require finite, decreasing fits after
  scoring reconstructed physical transfer.

These are source and test-contract changes only. They do not alter the earlier
statement that no Python import, pytest run, build, or accelerator workload was
executed on this host.
