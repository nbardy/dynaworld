# WorldFoam Memory-Light Theorem and Counterexample Ledger

Date: 2026-08-03

Status: mathematical audit of the current CPU/source design. This note proves
some restricted statements, rejects several broader readings, and selects a
production representation direction. It does not claim native Metal parity,
allocator measurements, publication-scale training, or a complete kinetic
native/trainer topology pipeline with runtime evidence. An exhaustive continuous CPU reference,
independent oracle, and active-boundary CPU compiler now exist. The active
compiler constructs predicates once per unique owner word, but its all-site
closure cost must still be reported against cumulative root-complement
witnesses rather than final chart count.

## Executive Decision

The memory-light architecture is mathematically sound, but the present world
parameterization is not a general dynamic foam.

Keep the exact ordered-transfer monoid, constant-state word VJP, sparse
track-face reduction, certified temporal atlas, `B_p x K` streaming, and
exact-versus-compiled route selection. Together they can make expensive
ordered-word forward/backward work independent of requested frame density on a
fixed physical interval, while unavoidable output and residual work remains
linear in the number of requested samples.

The mathematical core of that transfer layer is the translated optical-depth
measure `(kappa,nu)`: concatenation shifts the rear color measure by the front
optical depth, and the practical affine transfer `(beta,m)` is its Laplace
homomorphic image. The measure is an order-explicit proof and tangent object;
the runtime should continue to store only `(beta,m)` plus the compiled word.
No new Schur-like elimination is needed or justified. Gaussian Schur closure
works by marginalizing a Gaussian variable; eliminating WorldFoam depth would
erase the ordered-overlap phenomenon that the method is intended to retain.

Restrict the current fixed shared-metric 4D power world to an exact special
case and proof fixture. Its time slices are, after one common translation,
ordinary anisotropic 3D power diagrams with fixed sites and affine weights.
Every candidate face has a constant spatial normal. It cannot economically
represent a persistent rotating boundary or independently moving sites unless
many spacetime sites and topology events approximate that motion.

For the general dynamic renderer, replace only the geometry frontend with a
direct kinetic 3D power diagram whose sites and weights use a small temporal
basis independent of requested frames. The existing ray-cut, event, ordered
transfer, sparse VJP, temporal compilation, and streaming layers remain the
backend contract. Affine site trajectories increase event-polynomial degree
from at most quadratic to at most quartic for the basic predicates, but they
do not introduce a frame axis in persistent state.

The first production claim should be a frozen-structural-program P0 theorem.
Full differentiation through event times, chart/rank choices, and topology
changes remains a separate research problem and must not be implied by that
claim.

The present production update policy is correspondingly conservative. A
sealed structural program may survive material-only changes. Every geometry,
weight-trajectory, or camera-ray change must trigger a fresh full structural
compile and recertification. The simple-root continuation code is a restricted
whole-registry certificate, not a local program-repair algorithm.

## 1. Evidence Boundary

### 1.1 Inspected implementation evidence

The following files are authoritative for this audit:

- `TODO/worldfoam_memory_light_native4d.md`
- `research_experiments/world_foam_lane2/compiled_transfer_adjoint.py`
- `research_experiments/world_foam_lane2/exact_sparse_incidence_oracle.py`
- `research_experiments/world_foam_lane2/transfer_lie_chart.py`
- `research_experiments/world_foam_lane2/compiled_lie_world_adjoint.py`
- `research_experiments/world_foam_lane2/compact_lie_schedule.py`
- `research_experiments/world_foam_lane2/staged_compiled_lie_adjoint.py`
- `research_experiments/world_foam_lane2/piecewise_topology_staged_adjoint.py`
- `research_experiments/world_foam_lane2/sparse_power_word_compiler.py`
- `research_experiments/world_foam_lane2/power_topology_event_predicates.py`
- `research_experiments/world_foam_lane2/rational_polynomial_roots.py`
- `research_experiments/world_foam_lane2/kinetic_power_word_compiler.py`
- `research_experiments/world_foam_lane2/kinetic_owner_chart_compiler.py`
- `research_experiments/world_foam_lane2/kinetic_owner_chart_oracle.py`
- `research_experiments/world_foam_lane2/kinetic_active_owner_chart_compiler.py`
- `research_experiments/world_foam_lane2/kinetic_chart_transfer_bridge.py`
- `research_experiments/world_foam_lane2/kinetic_multichart_transfer_program.py`
- `research_experiments/world_foam_lane2/kinetic_continuous_transfer_acceptance.py`
- `research_experiments/world_foam_lane2/kinetic_stable_stratum_vjp.py`
- `research_experiments/world_foam_lane2/kinetic_multichart_stable_stratum_vjp.py`
- `research_experiments/world_foam_lane2/kinetic_native_topology_lowering.py`
- `research_experiments/world_foam_lane2/kinetic_native_precompiled_length_oracle.py`
- `research_experiments/world_foam_lane2/kinetic_native_precompiled_length_adapter.py`
- `research_experiments/world_foam_lane2/kinetic_native_equal_rank_lowering.py`
- `research_experiments/world_foam_lane2/kinetic_native_equal_rank_runtime_adapter.py`
- `research_experiments/world_foam_lane2/kinetic_native_equal_rank_geometry_reduction.py`
- `research_experiments/world_foam_lane2/kinetic_compiled_cpu_artifact_store.py`
- `research_experiments/world_foam_lane2/kinetic_geometry_trust_region.py`
- `research_experiments/world_foam_lane2/kinetic_simple_root_reisolation.py`
- `research_experiments/world_foam_lane2/kinetic_ragged_paper_step_cpu_fake_native.py`
- `research_experiments/world_foam_lane2/material_parameterization.py`
- `research_experiments/world_foam_lane2/test_kinetic_ragged_lie_sample_source_contract.py`
- `research_experiments/world_foam_lane2/host_memory_contract.py`
- `research_experiments/world_foam_lane2/compiled_route_cost_gate.py`
- `src/train/paper_ragged_track_staging.py`
- `src/train/paper_kinetic_ragged_sample_plan.py`
- `src/train/paper_kinetic_replayable_observations.py`
- `src/train/paper_kinetic_union_local_bar_assembly.py`
- `src/train/paper_ragged_material_bar_coordinator.py`
- `src/train/paper_kinetic_world_initializer.py`
- `src/train/paper_kinetic_active_track_program_factory.py`
- `src/train/paper_kinetic_fixed_site_material_state.py`
- `src/train/paper_kinetic_fixed_site_material_step.py`
- `research_notes/worldfoam_paper/WORLD_FOAM_DYNAMIC_DEPTH_ORDER_MATHEMATICIAN_PROMPT.md`
- `agent_notes/loose_notes/2026-08-03_03-35-19_worldfoam_memory_light_shared_adjoint.md`
- `agent_notes/loose_notes/2026-08-03_16-35-33_kinetic_power_word_event_sufficiency_red_team.md`

Provenance: this ledger describes the live 2026-08-03 working tree. Several
kinetic/root/compiler/oracle files are dirty or untracked rather than
commit-pinned. External reproducibility requires an intentional commit or a
content-hash manifest; the paths alone are not immutable evidence.

Observed in those sources:

1. The fixed-time compiler lowers Euclidean 4D power sites to a lower envelope
   of lines along each ray and retains only adjacent active owner pairs.
2. The P0 renderer composes exact affine RGB transfers and implements a
   two-pass, prefix-only manual VJP with no suffix or per-run reverse array.
3. Moving face cuts are reduced to sparse per-track/per-boundary Möbius
   coefficients, then lowered once to boundary, site, weight, and optionally
   camera gradients.
4. The compiled Lie route evaluates exact ordered words at `J` nodes, streams
   sample residuals into node cotangents, and scans the words again only at the
   nodes.
5. The compact schedule has an `O(KJ)` barycentric common path and an explicit
   row-local `O(J^2)` fallback.
6. The current host audit deliberately marks the global full-track CPU atlas,
   dense continuous-dual certificate, and resident full-video reference API as
   outside the bounded production contract.
7. The current native source has not been rebuilt and runtime-validated after
   its latest ABI changes. Logical tensor accounting is not allocator proof.
8. The exact continuous CPU reference enumerates all pair near/far and finite
   triple candidates, while the active compiler derives predicates only from
   observed endpoint owners and active cuts. Cached predicate construction is
   `O(U S R_max)` for `U` unique owner words. The current monotone closure still
   performs `W` root-complement discoveries and all-site certificates at
   `O(W (S log S + S R_max))`; describing the entire implementation as merely
   `O(SR)` or `O(C S R)` for final chart count `C` is false.
9. The independent oracle exposed a Sturm-chain sign bug: making a negative
   remainder monic can flip its sign and corrupt variation counts. Production
   root isolation now normalizes Sturm members only by positive scalars and has
   a rootless \(x^2+1\) regression.
10. A provenance-bound CPU bridge now binds every active kinetic chart to
    exact ordered P0 transfer at fixed `J_c` nodes, dispatches binary samples
    right-continuously, and returns only `O(sum J_c)` node cotangents before one
    sparse VJP to positions, velocities, quadratic weights, affine rays,
    density, and RGB. Continuous primal and referenced-material Jacobian/action
    error are certified for the actual second-form barycentric evaluator.
    Geometry approximation, chart endpoints, node times, interpolation
    weights, rank, dispatch, event times, and runtime roundoff remain outside
    the derivative/certificate claim.
11. A provenance-sealed single-ray lowerer now emits native-shaped CSR owners
    and positive `[J,R]` physical lengths with no requested-frame/sample axis.
    An independent CPU affine-Lie oracle and fake-native adapter test forward,
    arbitrary Lie cotangents, compact/global material scatter, and bounded
    length bars. A CPU geometry bridge maps the latter to frozen-stratum site,
    trajectory, weight, and ray bars.
12. A CPU/source equal-rank lowerer packs real `(track,chart)` rows into bounded
    actual-`J` native blocks without common temporal refinement or `J_max`
    padding. Its warm runtime adapter is lifecycle-tested against an injected
    CPU fake-native implementation; Metal build/runtime parity remains open.
    A row-ragged paper sampler joins arbitrary view/frame/pixel observations to
    those blocks, and a backend-independent outer coordinator preserves one
    global loss denominator and one optimizer authorization.
13. A CPU/source union-local assembler closes the heterogeneous-native-block
    join for one spatial request: cold-sealed compact-to-union maps feed one
    caller-owned `[S_union,4]` bar, each expected compact native VJP contributes
    exactly once, and the result seals directly into the outer coordinator.
    This is coverage/provenance evidence, not native VJP numerical parity.
14. An exact rational directional trust certificate proves a nonzero reuse
    radius for one strict event-free single chart. A separate exact CPU
    reference handles a restricted eventful stratum: it reconstructs the full
    rooted and rootless predicate registry from every base owner word, proves
    separated singleton simple-root tubes and a root-free complement over the
    complete binary64-rational directional homotopy, re-isolates each endpoint,
    and exactly reclassifies the left/right owner words. Repeated, shared,
    persistent-zero, endpoint, ray-collapse, and ambiguous cases fail closed.
    This is a correctness reference, not an output-sensitive warm-repair
    algorithm, and it provides no derivative through event time or structure.
15. A CPU/fake-native block-major paper step now exposes and closes a lifecycle
    bug that the request-local assembler alone did not close. For each spatial
    bundle it refreshes every active equal-rank node world once, streams every
    temporal request into the same bounded node cotangents, invokes one
    material-only ordered-word VJP per active native block, performs one union
    scatter, and only then releases the bundle. Varying temporal request size
    changes sample-launch count but not node-forward or word-VJP count. The
    material path allocates no `[J,W]` geometry-length bar. CPU tests match a
    direct-autograd oracle for `K=1` versus `K=4`, keep word work and retained
    runtime bytes fixed for `F=5` versus `F=41`, and show that two spatial
    bundles peak at the larger bundle rather than their sum. This is still
    fake-native/source integration, not rebuilt Metal or allocator evidence.
16. A byte- and entry-bounded CPU LRU now retains only observation-invariant
    compiler programs, topology/equal-rank descriptors, and a frame-free
    ragged sampler. It rejects targets, observations, rays, device maps, native
    payloads, and runtimes. The new fixed-site material coordinator uses this
    store directly; the legacy coordinator remains separate.
17. A replayable dense-observation source retains zero observation, target,
    ray, or device tensors, uses `24(F+V)` logical metadata bytes, and emits
    bounded track requests and observation chunks with exact coverage receipts.
    A separate fenced reduction maps one full bounded native `[J,W_b]` length
    bar to site, trajectory, polynomial-weight, and affine-ray bars. At this
    stage these closed source-level prerequisites but did not alone form a
    unified geometry-training step.
18. A source-integrated first fixed-site lifecycle now joins a bounded
    point-cloud initializer, an exact static-camera track factory, the bounded
    compiled-artifact store, built-in streamed target decode, a material-only
    authorization coordinator, canonical sigmoid/thresholded-softplus
    decode/VJP, manual SGD, and raw-only restart checkpoints. Material state is
    `48 B/site`, checkpoint state is `16 B/site`, both require explicit logical
    byte admission, and the optimizer lifecycle fails closed on non-CPU
    devices. This source has not yet run its focused gate and is not native or
    dataset-training evidence.
19. A newer source-only fixed-camera full-geometry coordinator now joins dense
    replay, one executor-sealed material-plus-length VJP per active block,
    immediate bounded `[J,W_b]` owner-local reduction, request/step accumulators,
    and a combined CPU-SGD/recompile transaction. It emits no ray bar and
    authorizes no mutation until exact full-manifest coverage. This is an
    integrated **source contract**, not rebuilt native evidence: its focused
    CPU/fake-native gate is unrun, the installed extension predates the ABI,
    and the material-only schema-v3 lane cannot certify it.
20. An unselected source-only fixed-camera fused-v1 transaction removes the
    `[J,W_b]` length cotangent and now owns fresh exactly-zero one-shot scratch.
    It clears one shared four-byte reason mask, enqueues validate-all,
    guarded-accumulate-all, and postwrite-finalize-all on one serialized stream, then
    fences/reads once. Each compact ledger and the shared global ledgers are
    scanned before and after accumulation. A zero receipt certifies the actual
    final float32 ledgers contain no NaN or infinity; postwrite failure is
    quarantined rather than rolled back, and failed completion proof retains
    live roots for restart. This is not a prospective bound, an exact or
    deterministic summation claim, exact active-manifest coverage, or optimizer
    fail-atomicity. Build/parity, native ordering/IEEE evidence, integration,
    allocator evidence, and the out-of-place optimizer boundary remain open.
21. The dense native sample executor now has a source-written, test-written,
    unrun single-outstanding launch-lifetime/settlement protocol. It installs
    the lifetime before native prepare, roots the prepared payload and every
    native sample argument, while the dense caller leases transfer/materialization
    predecessors. It permits at most one outstanding lifetime and makes
    settlement the sole completion fence and release authority. Every
    successful sealed session requires exact
    `prepare_count = launch_count = completion_fence_count`, zero outstanding
    lifetimes at seal, and no retained launch history. The dense request adds
    CPU-transfer and sample-materialization predecessor leases and expands
    restart-required quarantine across current target, sample, reverse,
    material/full-geometry, VJP, reduction, and completion roots. The lower-
    level request now rejects arbitrary target callables and accepts a sealed
    loader that installs one transfer lifetime before provider work, roots the
    CPU source before enqueue, retains returned device tensors, and carries a
    post-enqueue failure into the same quarantine. The lazy source route now
    has its own bounded no-retry carrier for visible lane-construction,
    sparse-transfer, sample, reverse, and lane-release roots. This closes those
    source ownership holes, not the runtime gate. A caller-installed two-phase
    lifetime now covers pre-lane union-local map transfers, and a separate
    carrier roots compact-material gather before its contiguous result; both
    retire duplicate predecessors only after a proven fence and leave no
    success-path strong-reference cycle. Earlier/lower top-level device
    allocation/`zero_`, cold device-to-host union-map receipt, native-forward
    enqueue-failure ownership, and canonical backend/stream fence contracts
    remain unresolved, so the route is still fail-closed on every non-CPU
    device, and no latest CPU/fake-native, rebuilt Metal, MPS allocator, or RSS
    gate has run.
22. The source-written sealed all-block union-v2 transaction is now an exact
    request-union geometry implementation candidate, but all blocks and compact
    material ledgers still coexist and no native/runtime gate has run. Its
    construction is now two phase: a caller-visible lifetime precedes every
    raw/native return and output allocation, publishes each returned root
    immediately, includes the four-byte sticky status in its exact scratch
    budget, and clears bulky construction roots only after accepted fenced
    execution. Partial construction either fences before release or enters the
    bounded fail-stop quarantine. The outer lifetime cannot publish individual
    raw-preparer config temporaries before the aggregate raw token returns, so
    failure is fenced but native allocator/runtime evidence remains mandatory
    for that internal seam. The
    all-block transaction is not mathematically required for request-level fail
    atomicity. A source-audited bounded-batch derivation keeps all
    output ledgers fresh and private until exact full-manifest acceptance, then
    prepares/fences/releases at most `q` reverse blocks at a time. At `q=1`,
    the prepared/output term changes from `sum_b(A_b+C_b)` to
    `max_b(A_b+C_b)` while the shared union ledgers remain live. This is a
    symbolic logical-tensor result and transaction proof sketch for bounded
    `q`, not an implementation, allocator measurement, or claim that extra
    per-batch fences improve wall time.

### 1.2 What is derived here

This note adds:

- an exact characterization of all time slices produced by a fixed shared-
  metric 4D power diagram;
- a minimal counterexample and approximation lower bound for rotating faces;
- a direct kinetic 3D replacement with bounded-degree event predicates;
- an exact pointwise transfer-closure theorem and constant-state VJP proof;
- a counterexample to universal exact fixed-dimensional linear/polynomial
  temporal atlases, not to every nonlinear parameterization;
- a repo-ordered ray-fiber optical-connection identity, normalized coherent-
  flow flatness theorem, general BV interface-curvature term, and counterexample
  showing that one constant whole-ray transfer need not imply flatness;
- corrected one-sided transfer-jet seam defects, the cumulative event-
  regularity filtration, and trivial real-order monodromy for separated simple
  roots;
- a clean separation of frozen-program and full derivatives; and
- work and memory bounds that include `B_p`, `K`, `J`, `E`, `R`, and dense
  fallback rows.

### 1.3 Units

All spacetime power expressions require dimensionless coordinates. Let

```text
x_bar = x / ell_0
t_bar = t / tau_0
```

for declared characteristic length `ell_0` and time `tau_0`. Equivalently, a
metric cross block encodes a time-to-length scale. Writing `M = I` while mixing
raw meters and seconds is not a coordinate-invariant physical statement.

The derivations below use normalized coordinates and omit bars.

## 2. Symbol Table

| Symbol | Meaning |
| --- | --- |
| `S` | persistent world-site count |
| `P` | total ray/sensor tracks in the logical observation |
| `F` / `F_requested` | requested temporal sample count over a fixed physical interval |
| `F_dataset` | calibrated camera/time records in the fixed physical dataset grid |
| `F_stored` | frames physically stored by the target source/cache |
| `B_p` | maximum simultaneously resident spatial-track block |
| `K` | maximum simultaneously resident temporal sample block |
| `I_tc` | ragged set of stored nonempty track-local chart incidences `(p,c)`; a semantic global refinement is never materialized as `P x C` |
| `C` | total stored track-local charts, `|I_tc|`, not a dense global chart axis |
| `E` | genuine track-local topology/order event records |
| `J_{p,c}` | transfer nodes/rank on track-local chart `(p,c)`; `J = max J_{p,c}` |
| `F_{p,c}` | requested sample count assigned to track-local chart `(p,c)` |
| `r_{p,c}` | ordered cell-run count for track `p` on local chart `c` |
| `R` | `sum_{(p,c) in I_tc} r_{p,c}`, stored run-chart incidences |
| `R_b` | resident run-chart incidences for one spatial block |
| `I_b` | resident unique track-face incidences for one spatial block |
| `N_fb,p,c` | exceptional temporal weight rows using dense `J_{p,c}^2` fallback |
| `N_fb` | sum of fallback-row executions over stored incidences/blocks |
| `N_B` | number of spatial blocks, `ceil(P/B_p)` |
| `B` | exact active native reverse-block count in one request transaction |
| `q` | bounded number of prepared reverse blocks retained in one transaction batch |
| `A_b` | source-visible logical bytes unique to prepared reverse block `b`, excluding already-counted active/lane state |
| `C_b` | compact material-output bytes of block `b`, currently `16s_b` for float32 RGBA |
| `L` | temporal basis size for a direct kinetic site's trajectory/weight |
| `epsilon` | requested primal and derivative-action approximation tolerance |

`F_requested` is the sampling-density variable. `F_dataset` and `F_stored`
belong to the fixed data/source contract for the narrow sweep. `E`, `R`, and
`J` are physical/compiler complexities. The desired theorem fixes the world,
camera program, physical interval, dataset grid, storage identity, and
`epsilon`, then varies only `F_requested`.

## 3. Theorem and Counterexample Ledger

| ID | Claim | Status | Load-bearing assumptions | Consequence |
| --- | --- | --- | --- | --- |
| G1 | A fixed shared-SPD(4) power diagram slices to a common translation of fixed anisotropic 3D sites with affine relative weights. | proved below | one shared constant metric; static 4D sites/weights | Current native-4D geometry is a restricted kinetic family. |
| G2 | Every candidate sliced face has a constant spatial normal. | proved below | same as G1 | A persistent rotating face is impossible with one fixed site pair. |
| G3 | In the one-fixed-normal-piece-per-temporal-chart approximation class, a rotating plane needs active face pieces/chart switches proportional to rotation divided by angular tolerance. | proof sketch below | one fixed-normal active face piece per chart | It does not by itself lower-bound arbitrary polyhedral site count; compactness still needs measurement. |
| G4 | Direct Euclidean affine kinetic 3D sites with degree-at-most-two weights retain bounded-degree ray/event predicates. | proved below | `A=I`, affine sites/rays, quadratic weights on one knot span | Near/far and analytic denominator predicates are degree at most two; triple concurrence is degree at most four. General shared `SPD(3)` is proposed/derived, not implemented. |
| G5 | Active-boundary predicate construction avoids exhaustive all-triple enumeration. | implemented and differential-tested | exact affine kinetic frontend; nondegenerate/fail-closed strata | Predicate work is `O(U S R_max)` across unique words, but closure certification is additionally `O(W (S log S + S R_max))`; final chart count alone is not the bound. |
| T0 | Ordered P0 words form a translated optical-depth-measure monoid whose Laplace image is the affine transfer monoid. | proved algebraically; small CPU certificate oracle | finite nonnegative optical depths; fixed front-to-back word; bounded RGB | The measure retains order for proofs and tangents, while runtime remains the four-scalar `(beta,m)` quotient. This is not a Schur marginalization. |
| T0a | Exponentially weighted total variation bounds affine-moment error, while the exponential mean-value bound controls attenuation error. | proved algebraically; paper theorem stated; proof-oracle regression source-written/unrun | finite vector measures extended by zero; declared vector norm; common background for the rendered-color bound | Supplies a representation-independent primal/tangent certificate without storing the measure at runtime. |
| T0b | A rear opacity tail is primal-safe after enough prefix optical depth, but training-safe truncation additionally requires a directional tangent bound. | proved algebraically; paper corollary stated; proof-oracle regression source-written/unrun; runtime optimization disabled | bounded rear colors; finite tangent measure; fixed prefix/tail membership and order; declared parameter norm with a uniform admitted-unit-direction bound; bounded/Lipschitz loss cotangent for a loss-VJP claim | Prevents high-opacity primal heuristics from silently dropping large geometry/material gradients. |
| T1 | A P0 scalar-extinction/RGB-emission word collapses pointwise to four transfer scalars. | proved below | primary-ray emission-absorption; view-independent RGB; finite P0 segments | Exact depth order is retained through an associative affine-transfer product. |
| T2 | No order-blind material summary is exact for arbitrary differently colored segments. | proved by two-segment counterexample | nonzero opacity and unequal colors | Depth order cannot be marginalized away. |
| T3 | The exact fixed-word VJP needs final transfer plus one prefix state, not a suffix tape. | proved below | frozen positive-length word; finite P0 material | Per-sample reverse interaction scratch is `O(1)` in word length, excluding output gradient buffers and the word itself. |
| T4 | Pointwise four-scalar closure does not imply one exact universal fixed-dimensional linear or fixed-degree polynomial temporal atlas. | proved by exponential family | linear/polynomial atlas class; arbitrary density/length parameters | Current linear/Chebyshev `J` must be adaptive. This does not rule out every compact nonlinear representation. |
| T5 | Small primal rank does not imply small tangent rank. | proved by a Chebyshev perturbation family | trainable temporal parameter direction | Rank selection must certify required JVP/VJP actions, not only RGB. |
| C0 | Finitely many simple separated owner cuts with strict competitor gaps admit a local owner-preserving ray-fiber trivialization. | proved by the implicit-function theorem in the 2026-08-05 connection audit | regular chart; continuous endpoints; uniform strict gaps; no simultaneous/full-fiber event | Supplies a canonical local fiber coordinate. Cross-pixel sensor-time patch reuse remains an open census, not a compiler or speed result. |
| C1 | In the repository's near-to-far/right-ordered convention, the covariant time derivative of one stable-chart transfer equals the depth integral of prefix-curvature-suffix plus explicit moving-endpoint flux. | proved algebraically in the 2026-08-05 connection audit; oracle not implemented | finite optical depth; `C^1` interior connection or BV one-sided extension; fixed scan convention | Supplies an exact connection/correspondence diagnostic. The scientist attachment's left-ordered sandwiches and holonomy sign cannot be copied into executable WorldFoam. |
| C2 | For an independently specified normalized `C^1` depth flow, constrained flatness `F^R_tz=0` is equivalent to exact reuse of every orientation-preserving transported subinterval. | proved in the 2026-08-05 connection audit | `A_t=-wA_z`; `phi_t0=id`; non-crossing flow; physical ray Jacobian included; endpoints transported for a whole-clip claim; a merely Lipschitz flow uses the corresponding almost-everywhere/pulled-back-measure statement | Flatness is sufficient for coherent reuse. It is not necessary for equality of one total transfer, where nonzero curvature can cancel in depth. |
| C3 | P0 interface curvature is `([w A_z]-r_dot[A_z]) delta_r`; it reduces to `(w-r_dot)[A_z] delta_r` only for a continuous flow trace. | proved distributionally; oracle not implemented | BV generator and flow with one-sided traces; endpoint flux treated separately | Prevents discontinuous flow, clipping, or reversed prefix/suffix order from being mistaken for a valid visibility residual. |
| C4 | A scalar depth flow on one fixed pixel represents a 3D scene flow only when `V(Gamma,t)=partial_t Gamma+w partial_z Gamma`. | proved by the pushforward compatibility condition | regular ray parameterization | Generic transverse camera/object motion needs a tracked pixel or full `(u,v,t,z)` horizontal field; a free per-ray `w` can hide the answer. |
| C5 | Compiling transported curvature lowers temporal state/work relative to the existing transfer atlas. | open and explicitly unclaimed | same continuous primal/tangent norm; independently budgeted flow; endpoint transports, reconstruction, gradients, cone, and conditioning all counted | Compare physical `U`, group-completion `U_tilde` in an unrestricted `beta>0` chart, and signed-tangent `K_F`. Neither new object fits the current physical-transfer ABI unchanged. Do not create a runtime unless total payload/word work improves at least `2x` and measured request time at least 20%. |
| D2a | At a one-sided order-`q` vanishing segment seam, matching exterior germs and signed width coefficients determine the first possible transfer-jet defect: `q! P_0(a_+X_+^0-a_-X_-^0)S_0`. | proved under corrected `C^q` hypotheses | integer `q>=1`; common-domain one-sided Taylor germs through order `q`; one-sided `C^q` widths/generators | Cumulative defect sets obey `Sigma_<=0 subseteq Sigma_<=1 ... subseteq Delta_owner`; the attachment's reverse inclusion is false. |
| D2b | Exact owner subcharts can be covered by fewer transfer charts whenever primal transfer and every required selected tangent action glue within tolerance. | open compiler hypothesis | exact owner/provenance atlas retained; declared direction set/norm; certified seam defects; heterogeneous/ragged transfer evaluation | Census `delta_e^(0)` and `delta_e^(1)(D)` before implementation. Fewer chart labels alone is not a win; `sum J_d^2`, `sum F_d J_d`, and `sum J_d R_d` must fall after certificate cost. |
| R1 | Interior real simple separated event roots with fixed count have global sorted labels and trivial real-order monodromy on a connected regular region. | proved by implicit continuation and order separation; restricted CPU reference already exercises one update path | complete relevant registry; roots remain real, simple, interior, and distinct; no degree loss | Braid/2-stack machinery is unnecessary for ordinary supported roots. Closed-loop warm-program comparison is deferred until an in-place patcher exists. |
| D1 | Away from event strata, the frozen-word geometry/material derivative equals the classical derivative of exact sampled rendering. | proved locally and CPU-certified for one directional single-chart update | strict owner/order/denominator/length margins and fixed sample time | Frozen-program training is valid inside the implemented narrow event-free trust region; active or multichart updates still recompile. |
| D2 | At zero-length birth/death, finite-P0 forward transfer is continuous but its geometry derivative can jump. | proved below | finite density/color; identity at zero length | Use one-sided/generalized derivatives or avoid seam samples. |
| D3 | A moving event boundary contributes a jump term to a continuous exposure/integral objective. | proved by Leibniz rule | simple event root | The term vanishes when the two forward traces agree at the seam, but not for a discontinuous tie/material rule. |
| D4 | The present manual VJP is the full derivative through topology, event times, adaptive nodes, rank, and dispatch. | refuted by source scope | none | Paper/code must call it a frozen-program VJP. |
| D5 | After streamed residual-to-node reduction, the frozen-program kinetic geometry/material reverse needs a frame-indexed tape. | refuted by implementation | fixed charts/endpoints/node times/weights/rank; strict margins | Reverse state is `O(sum J_c)` plus world gradients and work is `O(sum J_c R_c)`; recompiling after a perturbation is a different derivative. |
| D6 | The current direct-kinetic certificate bounds continuous geometry/ray tangent error. | open | actual affine-Lie barycentric evaluator; fixed surrogate; declared sparse action norms | Exact node geometry VJPs prove the compiled-surrogate derivative, but not its uniform error against the physical geometry/ray derivative. |
| D6a | A continuous Jacobian bound in lowered active-cut coefficients and affine ray direction, composed through the exact sparse world-to-cut Jacobian, is sufficient for D6. | proved conditionally; certifier not implemented | fixed stable word; denominator/speed/length margins; stored-node primal linearization; bounded sparse induced norm | Certify one track-local cut jet rather than allocate a dense global world dual; runtime representation remains unchanged. |
| D7 | Uniform local primal/tangent bounds imply a sparse global normalized-loss VJP bound without a dense global dual. | proved conditionally below | D6-style local bounds; bounded/Lipschitz output cotangent; sealed sparse gather/scatter; globally normalized nonnegative sample weights | Once D6 supplies `epsilon_0,epsilon_1`, loss-gradient error follows algebraically and does not gain a frame-count factor merely from denser normalized sampling. |
| D8 | The fixed-word physical-length VJP is supported only on owners in that word plus the ray. | proved algebraically; owner-local sparse CPU bridge implemented source-only/unrun; fixed-camera fused-v1 source and all-block transaction adapter implemented but unbuilt; the fixed-camera coordinator can select it explicitly and the combined updater binds that mode in its receipt, all source-only | sealed continuous owner certificate; immutable world/ray/chart snapshot; distinct owners; fixed near/far, node times, rank, and dispatch | Warm row scratch can be `O(JR+R(6+L_w))` and scatter directly into caller-owned world bars; non-owner competition remains a cold certificate obligation. Fixed-camera fused v1 has no ray-cotangent surface and removes only the `[J,W]` physical-length cotangent tape, not the compiled primal lengths. Its postwrite transaction rejects actual nonfinite aggregate destinations; prospective bounds, native build/parity, allocator evidence, and production trainer routing remain open. |
| D9 | A zero receipt from the sealed fused transaction certifies finite final output ledgers and fail-closed optimizer admission for the supplied block sequence. | proved conditionally from the source protocol; raw finalizer and one-shot/fail-stop adapter implemented; unbuilt and runtime-unverified | fresh token-owned exact-zero storage-distinct scratch; immutable sealed inputs; no hidden aliases/concurrent mutation; exactly `B` validations, `B` guarded accumulations, and `B` finalizers on one serialized stream; compact scans per block and shared-global scans exactly once in each scan phase; one successful completion fence/status read; IEEE-compatible NaN/infinity classification; no persistent commit before acceptance | Prewrite rejection leaves zero scratch; postwrite rejection may leave mutated/nonfinite scratch but cannot authorize commit and is quarantined. Zero proves actual final float32 finiteness, not exactness, deterministic atomic order, underflow freedom, a prospective overflow bound, exact active-manifest coverage, or optimizer-step fail-atomicity. |
| M1 | Compiled expensive word forward/reverse work can be independent of `F`. | proved conditionally for a fixed compiled surrogate; dense source telemetry corrected to separate row-node threads from run-node interactions, unrun | `E`, ragged run incidences `R`, local ranks `J`, camera program, interval, tolerance, nodes, weights, and dispatch fixed | Each direction scans `sum_(p,c) J_(p,c) r_(p,c)` run-node incidences, so a material forward plus reverse scans twice that amount; the prior dense report's `row_count*J` was only dispatch-thread count. Sampling is still rank-weighted and linear in requested observations. |
| M2 | Total rendering/training work can be independent of `F`. | impossible | output has `PF` colors | At least `Omega(PF)` output/residual work remains. |
| M3 | Peak reverse interaction memory can omit both `F x R` and resident `P x F` tensors. | proved for the staged CPU/source lifecycle; native allocator unverified | one `B_p x K` block in flight; compact schedules; streamed targets; bounded union-local material bars | Target design is memory-light and the ragged coordination seam is implemented at CPU/source scope; current CPU APIs/oracles are not evidence for native peak. |
| M4 | Compiled replay is always faster than exact replay. | refuted by current cost gate | none | Route per chart; low-run/high-rank charts should use exact streamed replay. |
| M5 | The previous schema-v3 source candidate varied only requested sampling density and retained no hidden frame-scaled artifact metadata. | refuted by source red-team; source repair unrun | none | Old artifacts retained `Theta(F_dataset)` camera signatures and warm checks rescanned frame metadata. The repaired gate fixes one 300-frame provider and varies endpoint-including `F_requested=8,64,300`; only cross-row receipts may establish invariance. |
| M6 | Fixed-program requested-density invariance: increasing only `F_requested` leaves compiled node-forward and ordered-word-reverse state, interaction counts, and frame-invariant logical tensor bytes unchanged. | proved conditionally for the fixed compiled surrogate; implemented at CPU/source lifecycle scope; rebuilt native allocator evidence open | fixed world and temporal basis; continuous camera; physical interval; dataset grid and target-storage identity; event/chart partition; owner words and stored run incidences `R`; ranks `J`; interpolation, fallback, and dispatch decisions; bounded `B_p x K` streaming; nonresident selected-pixel targets; exact sample/event dispatch or exclusion of unresolved isolator neighborhoods | No selected-route tensor needs an `F_requested x S`, `F_requested x R`, or `F_requested x J x R` axis. Camera/target access, sample-to-node reduction, and output remain the separate `Theta(sum F_(p,c) J_(p,c) + sum N_fb,p,c J_(p,c)^2 + PF)` slice. Streamed static PowerFoam with `batch=1` is an essential memory control because it can also have frame-independent peak. |
| M7 | M6 implies end-to-end sublinear memory/work for arbitrary longer-duration sequences. | refuted for current source and impossible for full-coverage work | none | The replay source currently retains exactly `24(F_requested+V)` logical metadata bytes, and provider camera/time records are `O(V F_dataset)`/`O(F_dataset)`; these memory terms require identity/camera streaming to remove. Reading, comparing, or emitting all requested colors costs `Omega(PF)`. If `F` grows by extending physical duration rather than densifying one fixed program, `E`, stored word/run incidence `R`, chart count, ranks `J`, and exact-root bit lengths may grow with duration, so no sublinear structural-memory theorem follows. |
| M8 | On a fixed certified atlas, compiling the ordered-word world reverse replaces the exact replay term `Theta(sum F_(p,c) r_(p,c))` by `Theta(sum J_(p,c) r_(p,c))`, while retaining only bounded streamed sample state. | proved conditionally from the compiled objective and prefix VJP; native launch/allocator evidence open | all M6 assumptions; spatial-bundle outer/temporal-chunk inner lifecycle; one node reverse after complete sample reduction; no rank or chart reselection from the requested grid | The expensive depth-order/world slice is sublinear—and in fact invariant—in requested frame density when `J` is fixed. Total training remains at least `Omega(PF)` and includes rank-weighted sample reduction, so the paper must report world-side and end-to-end scaling separately. No Schur-complement-like depth elimination is required for this result. |
| M9 | Removing the staged `[J,W_b]` length cotangent guarantees a lower whole-request peak. | refuted as an implication; fused v1 and raw/sealed all-block union-v2 source have explicit logical bounds, but both are unbuilt/unrun and allocator evidence is open | none | Fused v1 retains all prepared active blocks, compact outputs, one global float32 geometry output, the CPU float64 bridge destination, and the step accumulator concurrently. Union v2 now source-writes request-union geometry with exact source/bridge saving `12(S-U)(6+C_w)` before any new `8U` CPU commit map, while retaining compact material ledgers; executor/request bridging and bounded fail-atomic block streaming remain separate. |
| M10 | With `N <= K`, one outstanding sample lifetime, fenced settlement, and no retained lifetime history bound successful-path retained sample-axis source tensor payload by `O(KJ+K)`, hence `O(K)` for fixed `J`, rather than by launch count or `F`. | proved conditionally from the source protocol; implementation and focused tests source-written but unrun; native allocator unverified | prepared native payload fully roots all native arguments; CPU-transfer and materialization predecessor leases survive to the sole completion fence; exact `prepare=launch=fence` accounting; successful settlement releases roots immediately; arbitrary loader partial failure and unknown-completion failure quarantine excluded from the successful path | This is a lifetime/retention theorem, not a total-work theorem: full sampled rendering remains `Omega(PF)`. Unknown completion intentionally retains one bounded lifetime until restart. Python heap, allocator slabs, command buffers, driver storage, and actual RSS remain outside the logical tensor bound. |
| M11 | Request-level fail atomicity can coexist with at most `q` live prepared reverse blocks because partially written transaction scratch is disposable until exact full-manifest acceptance. | proved conditionally as a transaction recurrence; not implemented or runtime-verified | tensor-free exact manifest preflight; fresh storage-distinct union outputs; no persistent write before acceptance; sticky per-batch validation/finiteness status; canonical batch order; compact material is scattered into fresh union material scratch and that **post-scatter union ledger** is finalized before release; the one-fence form requires a status-gated native scatter, while current primitives require a first fence/status acceptance before unconditional `index_add_` and a second fence after the union-material finite check; unknown completion quarantines current batch, shared outputs, and session/lane roots | For `q=1`, the all-block prepared/output term `sum_b(A_b+C_b)` becomes `max_b(A_b+C_b)` without changing `P_U sum_b Q_b g_b = sum_b P_b g_b`. Finite compact bars do not prove their union sum finite, and an unconditional pre-status scatter can consume an invalid map. Fence count is `nu ceil(B/q)` with `nu=1` only after the gated-scatter ABI exists and `nu=2` for the safe current-primitive fallback; it is structural rather than frame-density scaling on a fixed atlas but may regress wall time. |
| P1 | The current fixed 4D representation should remain the general WorldFoam world. | rejected | general moving/deforming scenes required | Keep it as a special case/oracle; use direct kinetic 3D sites for the general frontend. |

## 4. Exact Expressivity of Fixed Shared-Metric 4D Power Cells

### 4.1 Setup

Write one shared spacetime metric in block form:

```text
M = [ A   b ]
    [ b^T c ] in SPD(4),
```

where `A in SPD(3)`, `b in R^3`, and the Schur complement

```text
lambda = c - b^T A^{-1} b > 0.
```

Let static 4D site `i` be

```text
q_i = (a_i, tau_i),       a_i in R^3,
```

with scalar power weight `w_i`. Its power function is

```text
Pi_i(x,t) = ([x;t] - q_i)^T M ([x;t] - q_i) - w_i.
```

### 4.2 Slice-equivalence theorem

**Theorem G1.** For every fixed `t`, the owner diagram induced by the static
4D sites is exactly an anisotropic 3D power diagram whose sites all share one
translation and whose relative weights are affine in time.

**Derivation.** Define

```text
v = A^{-1} b,
p_i^0 = a_i + v tau_i.
```

Completing the spatial square gives

```text
Pi_i(x,t)
  = ||x - (p_i^0 - v t)||_A^2
    + lambda (t - tau_i)^2 - w_i,
```

where `||y||_A^2 = y^T A y`. Therefore the effective moving 3D site and weight
are

```text
p_i(t) = p_i^0 - v t,
omega_i(t) = w_i - lambda (t - tau_i)^2.
```

Every site has the same velocity `-v`. In the translated coordinate

```text
y = x + v t,
```

the centers are fixed at `p_i^0`. Expanding `omega_i` gives

```text
omega_i(t)
  = -lambda t^2
    + 2 lambda tau_i t
    + w_i - lambda tau_i^2.
```

The first term is common to all sites and cancels from every ownership
comparison. Hence the same diagram is generated by fixed centers and affine
weights

```text
omega_hat_i(t)
  = 2 lambda tau_i t + w_i - lambda tau_i^2.
```

Conversely, given fixed anisotropic centers `p_i^0`, arbitrary affine weights
`alpha_i t + gamma_i`, and a chosen common velocity `-v`, choose

```text
tau_i = alpha_i / (2 lambda),
a_i = p_i^0 - v tau_i,
w_i = gamma_i + lambda tau_i^2,
b = A v,
c = lambda + b^T A^{-1} b.
```

These parameters reproduce the requested sliced diagrams. Thus, modulo
non-unique power-diagram generator gauges, this characterizes the family.

### 4.3 Constant-normal corollary

The pairwise difference is

```text
H_ij(x,t) = Pi_i(x,t) - Pi_j(x,t)
          = n_ij^T x + eta_ij t + gamma_ij,
```

with

```text
n_ij = 2 [A, b] (q_j - q_i),
```

where the expression takes the spatial three rows. `n_ij` is constant in
time. A sliced candidate face can translate and appear/disappear, but a
persistent pairwise face cannot rotate.

For the executable `M = I` specialization, `b = 0`: the effective spatial
sites do not move at all. Dynamics arise entirely from affine relative power
weights selecting and resizing cells.

This is not the claim that every sliced cell is static. Intersections of many
independently translating fixed-normal halfspaces can change size, vertices,
and topology. The restriction is that the available face orientations come
from a finite, time-independent bank of site pairs.

### 4.4 Minimal exact counterexample

Consider two kinetic 3D sites in a 2D spatial subspace:

```text
p_1(t) = (-1, -t/2),
p_2(t) = ( 1,  t/2),
w_1(t) = w_2(t) = 0.
```

Their Euclidean bisector passes through the origin and has normal proportional
to

```text
p_2(t) - p_1(t) = (2,t).
```

Its orientation varies continuously with `t`. No one pair of fixed 4D shared-
metric sites can induce that persistent face because G2 requires a constant
spatial normal.

An even clearer target is the rotating partition

```text
cos(omega t) x_1 + sin(omega t) x_2 = 0.
```

One fixed 4D pair cannot represent any nonzero angular interval exactly.

### 4.5 Approximation lower bound for fixed-normal refinement

Restrict the rotating target line to a disk of radius `L`. Approximate it on
each temporal chart by a line with fixed normal. If the angular error is
`delta`, the endpoint displacement is at least

```text
L |sin(delta)|.
```

Uniform Hausdorff error at most `epsilon < L` therefore requires

```text
|delta| <= asin(epsilon / L).
```

Covering a total rotation angle `Theta` with one active fixed-normal face piece
per temporal chart requires at
least

```text
N >= Theta / (2 asin(epsilon/L))
  = Omega(Theta L / epsilon)       as epsilon/L -> 0.
```

Each switch of that active approximating face requires a combinatorial event or
a material/owner handoff. This is a lower bound on active face pieces/chart
switches in the stated approximation class. It is **not** automatically a
linear lower bound on total site count for arbitrary polyhedral or staircase
approximations: \(S\) sites can expose up to quadratically many candidate
pairs, and a broader site/event bound would need a separate construction. It
still proves that this simple refinement is not a free substitute for a kinetic
orientation degree of freedom.

### 4.6 Decision implied by G1--G3

The fixed 4D formulation is useful for:

- exact affine-cut event predicates;
- spacetime birth/death through affine relative weights;
- a compact special-case scene family;
- proof fixtures and same-representation scaling tests; and
- scenes well described by a finite bank of fixed face orientations.

It must not be sold as an unrestricted moving-cell model. If real data needs
rotating, independently translating, or deforming persistent partitions, the
site count and event count may grow with approximation accuracy and erase the
memory/quality advantage.

## 5. Direct Kinetic 3D Replacement

### 5.1 Smallest useful generalization

The current executable frontend uses the Euclidean time-dependent 3D power
field

```text
D_i(x,t) = ||x - p_i(t)||^2 - omega_i(t),
```

with

```text
p_i(t)     = p_i0 + t v_i,
omega_i(t) = omega_i0 + omega_i1 t + omega_i2 t^2.
```

This is nine structural scalars per site. Adding P0 density and RGB gives
thirteen scalars per site, independent of `F`. A shared fixed
`A in SPD(3)` is a derived/proposed generalization, not an implemented input.
It can be reduced to the Euclidean form only by explicitly whitening sites and
rays and carrying the transformed optical line element.

The broader proposed family uses a small declared basis:

```text
p_i(t)     = sum_{ell=0}^{L-1} a_{i,ell} psi_ell(t),
omega_i(t) = sum_{ell=0}^{L-1} s_{i,ell} psi_ell(t).
```

`L` follows physical motion complexity, not requested frame count. Persistent
geometry state is `O(SL)` rather than `O(SF)`. The current quadratic weight is
one coefficient richer than a common two-function affine basis; do not call it
an affine-weight implementation.

Piecewise polynomial or B-spline bases should expose their knot spans as
structural charts. Do not hide a per-frame table behind the word "basis."

### 5.2 Ray-cut equations

For the proposed shared-metric notation below, set `A=I` for the current
implementation. For pair `(i,j)`, define

```text
n_ij(t) = 2 A (p_j(t) - p_i(t)),
b_ij(t) = p_i(t)^T A p_i(t)
          - p_j(t)^T A p_j(t)
          - omega_i(t) + omega_j(t).
```

The face is

```text
h_ij(x,t) = n_ij(t)^T x + b_ij(t) = 0.
```

For affine camera ray

```text
r(t,z) = o(t) + z d(t),
```

the cut is

```text
z_ij(t) = -B_ij(t) / A_ij(t),
A_ij(t) = n_ij(t)^T d(t),
B_ij(t) = n_ij(t)^T o(t) + b_ij(t).
```

This preserves the downstream interface already used by the transfer backend:
an implicit face, a ray pullback, a cut evaluator, and a sparse VJP.

### 5.3 Event-degree theorem for affine motion

Assume `p_i`, `o`, and `d` are affine and `omega_i` has degree at most two on
one chart. Then:

- `n_ij(t)` has degree at most one;
- `b_ij(t)` has degree at most two;
- `A_ij(t)` and `B_ij(t)` have degree at most two;
- a fixed near/far crossing `A_ij(t) z_fixed + B_ij(t) = 0` has degree at
  most two;
- a finite-cut denominator event `A_ij(t) = 0` has degree at most two; and
- adjacent-cut concurrence

  ```text
  B_ij(t) A_jk(t) - B_jk(t) A_ij(t) = 0
  ```

  has degree at most four.

Thus changing the geometry frontend does not require generic symbolic
algebra. It requires certified degree-two and degree-four root isolation,
degeneracy handling, and sparse coefficient VJPs. The current affine 4D case
is the lower-degree specialization where `A_ij` and `B_ij` are affine and the
concurrence predicate is quadratic.

### 5.4 What direct kinetic 3D does not solve

It does not bound:

- ray stabbing depth `R`;
- kinetic event count `E`;
- topology invalidation under large optimizer steps;
- temporal transfer rank `J` near events or high contrast; or
- material identifiability.

It fixes representation adequacy without changing those independent
problems. A cited near-quadratic result concerns a restricted planar,
unweighted, unit-speed linear-motion Delaunay setting. It is cautionary prior
art, not a bound for this weighted 3D per-ray lower envelope. The present route
must derive or measure its own event complexity and remain output-sensitive in
`E`, not assume constancy in `S` or physical time.

The current theorem is also for an unbounded Euclidean power partition clipped
by global near/far. It does not yet reproduce Power Foam's moving controlling
spheres. Bounded-cell parity adds sphere entry/exit and vacuum-gap events,
radius positivity, conservative adjacency/culling, and their sparse VJPs.
Those predicates are a later obligation and are not covered by the pair
near/far/triple completeness theorem below.

## 6. Exact Ordered-Transfer Closure

### 6.1 Pointwise closure theorem

For one finite P0 segment, let

```text
tau  = rho L >= 0,
beta = exp(-tau),
m    = (1 - beta) c in R^3.
```

The segment acts on radiance behind it as

```text
g(q) = m + beta q.
```

Represent it by `g = (beta,m)`. If `g_1` is in front of `g_2`, composition is

```text
g_1 star g_2
  = (beta_1 beta_2, m_1 + beta_1 m_2).
```

This operation is associative because it is composition of affine maps:

```text
(g_1 star g_2) star g_3
  = g_1 star (g_2 star g_3)
  = (beta_1 beta_2 beta_3,
     m_1 + beta_1 m_2 + beta_1 beta_2 m_3).
```

The identity is `(1,0)`. By induction, any finite ordered word collapses
exactly at one time to one scalar attenuation and one RGB moment: four real
scalars.

This theorem is restricted to scalar extinction, RGB emission, P0 material,
and primary-ray emission-absorption. View-dependent bases, polarization,
matrix-valued transport, scattering, or spatially varying segment material can
require a larger sufficient statistic.

The order-explicit object behind this quotient is obtained by setting

```text
K_0 = 0,
K_r = sum_(q<=r) tau_q,
d nu(u) = c_r du on [K_(r-1),K_r).
```

For front word `A` and rear word `B`, define

```text
(kappa_A,nu_A) odot (kappa_B,nu_B)
  = (kappa_A+kappa_B,
     nu_A + shift(kappa_A)_# nu_B).
```

Translations compose, so `odot` is associative. Its Laplace image is

```text
L(kappa,nu)
  = (exp(-kappa), integral exp(-u) d nu(u))
  = (beta,m),

L(A odot B) = L(A) star L(B).
```

The translated measure is therefore the mathematical proof object that makes
order and moving optical-depth boundaries explicit. It is intentionally not a
runtime representation: the compiled executor still stores the sufficient
affine quotient `(beta,m)` and replays the retained word only at compiler
nodes. This closes the algebraic question without inventing a Schur-like foam
marginalization.

### 6.2 Order cannot be marginalized

Take

```text
beta_1 = beta_2 = 1/2,
c_1 = (1,0,0),
c_2 = (0,0,1).
```

Then `m_1 = (1/2,0,0)` and `m_2 = (0,0,1/2)`. The two orders emit

```text
m_12 = (1/2,0,1/4),
m_21 = (1/4,0,1/2).
```

Their attenuation is identical, but their RGB is not. More generally,

```text
m_12 - m_21
  = (1 - beta_2) m_1 - (1 - beta_1) m_2.
```

Therefore any statistic that retains only total optical depth or an unordered
set of materials is not exact for the target phenomenon.

### 6.3 Constant-state word VJP theorem

Let the prefix before a current segment be

```text
P = (T,F),
```

the local segment be `g = (beta,m)`, and the final total be

```text
G = (B,M).
```

For a local perturbation,

```text
delta B = B delta log(beta),

delta M
  = T delta m
    + (M - F - T m) delta log(beta).
```

The second identity follows because `M - F - Tm` is exactly the suffix moment
after attenuation by the prefix and current segment. For cotangent
`(bar_B,bar_M)` and local parameter `q`,

```text
bar_q
  = T bar_M dot partial_q m
    + [bar_M dot (M - F - Tm) + bar_B B]
      partial_q log(beta).
```

One first pass computes `(B,M)`. A second front-to-back pass retains only
`(T,F)` and advances

```text
F <- F + T m,
T <- T beta.
```

No suffix array is necessary.

For P0 `m=(1-beta)c`, `log(beta)=-tau`, the code's local forms are

```text
bar_tau = bar_M dot (F + T c - M) - B bar_B,
bar_rho = L bar_tau,
bar_L   = rho bar_tau,
bar_c   = T (1-beta) bar_M.
```

Shared endpoint cotangents from neighboring segments are summed, then reduced
through one sparse track-face coefficient row and one boundary-to-site scatter.

Time is `Theta(R)` per exact word sample, with two scans. Local reverse scratch
is `Theta(1)` in `R`; persistent gradient arrays are `O(S + I)` and the input
word itself is `O(R)`.

### 6.4 Pointwise closure is not one universal linear temporal atlas

Consider a one-run stable chart on `t in [0,1]` with positive length

```text
L(t) = 1 + t
```

and variable density `rho > 0`. Its attenuation family is

```text
beta_rho(t) = exp(-rho (1+t)).
```

For distinct `rho_1,...,rho_N`, these exponentials are linearly independent;
their Wronskian is a nonzero Vandermonde factor times a positive exponential.
Consequently, no universal finite-dimensional linear temporal space contains
all valid transfers, even before multiple colored segments are composed.

A polynomial atlas of any fixed finite degree also cannot represent one
nonconstant exponential exactly on an interval. The exact four-scalar
pointwise monoid therefore does not imply an exact fixed-`J` linear or
fixed-degree polynomial atlas.

This is **not** a lower bound against every nonlinear representation: the same
family `exp(-rho(1+t))` has an exact one-parameter nonlinear description by
`rho`. Any stronger impossibility claim must first define the admissible
nonlinear state, evaluation operations, accuracy metric, and parameter range.
The result here justifies adaptive rank for the actual linear/Chebyshev atlas.

### 6.5 Certified approximate temporal closure

On a topology-stable chart, suppose:

- every cut denominator has magnitude at least `delta_A > 0`;
- every physical segment length has a positive margin;
- `||d(t)|| >= delta_d > 0`;
- densities and colors are bounded;
- the camera/site functions extend analytically to a complex neighborhood;
  and
- the nearest cut pole, direction-norm zero, square-root branch point, or
  topology singularity lies outside a Bernstein ellipse `E_rho`, `rho > 1`.

Then total transfer and any fixed sparse parameter-direction JVP are analytic
on that ellipse. Standard Chebyshev bounds give an error of the form

```text
||f - f_J||_infinity
  <= constant * M_rho * rho^{-J} / (rho - 1).
```

Thus

```text
J = O(log(M_rho/epsilon) / log(rho))
```

is independent of requested frame density `F`. It still depends on physical
motion, distance to events, opacity, material contrast, interval length, and
the derivative directions being certified.

The affine-transfer Lie chart is a structure-preserving coordinate choice for
this approximation. It does not make the semigroup commutative and it does not
remove the need for rank/error certification.

### 6.6 Primal-versus-tangent counterexample

Let

```text
kappa_theta(t) = kappa_0 + theta T_N(t),
```

where `T_N` is the degree-`N` Chebyshev polynomial and choose a neighborhood of
`theta=0` small enough that `kappa_theta >= 0`. With one constant source color,
the primal transfer at `theta=0` is constant and has rank one. Its parameter
derivative contains `T_N(t)` and requires degree at least `N` for exact
polynomial representation.

Therefore a forward-only rank gate can accept a catastrophically wrong world
gradient. The current separate primal/tangent gates are mathematically
necessary, not optional diagnostics.

## 7. Physical, Frozen-Program, and Compiled-Algorithm Derivatives

### 7.1 Three derivative objects that must not be conflated

Let the structural compiler return

```text
s(theta) = {
  owner words,
  event/chart intervals,
  dispatch rules,
  adaptive nodes and ranks,
  interpolation weights,
  certificate decisions
}.
```

Let `R(theta)` denote the exact physical renderer for the declared observation
objective, and let the numerical compiled renderer be `G(theta; s)`. There are
three objects:

```text
frozen program:
    D_theta G(theta; s_0) with s_0 held fixed,

physical objective:
    D_theta R(theta), point-sampled or exposure-integrated as declared,

full compiled algorithm:
    D_theta G(theta; s(theta)).
```

The inspected manual VJP implements the first contract. It differentiates
densities, colors, physical lengths, finite cut coefficients, active
boundaries, shared sites/weights, and optionally affine rays within the
supplied program. It does not differentiate discrete owner discovery, event
isolation/dispatch, chart endpoints, adaptive split/rank decisions, or all
parameter dependence of sample nodes and weights.

The physical derivative is not automatically the full compiled-algorithm
derivative. Inside a regular stratum the frozen exact renderer can equal it;
an exposure objective may add seam-boundary terms; adaptive approximation
choices belong to the third object and may be stopped or bounded separately.

### 7.2 Regular-stratum theorem

Fix a physical query `(p,t)`. Suppose there is an open parameter neighborhood
where:

- every owner inequality has the same strict sign;
- the word and adjacent face identities do not change;
- all cut denominators stay nonzero;
- segment lengths stay positive; and
- the ray speed stays nonzero.

Then the discrete word is locally constant and every retained endpoint is a
smooth implicit function of parameters. On this neighborhood, the frozen-word
derivative of exact direct rendering equals the ordinary classical derivative
of the physical sampled renderer.

This is stronger than saying topology derivatives are always missing. Away
from an event, there is no derivative of a locally constant discrete owner
label to include. The endpoint geometry derivative is required and is already
inside the current fixed-word VJP.

### 7.3 Sufficient structural trust region

Let `q_k(theta,t)` be every sign predicate needed by a certified program and
suppose

```text
mu_k = inf_t |q_k(theta,t)| > 0,
L_k  >= sup_{t, s in [0,1]}
        ||D_theta q_k(theta + s delta theta,t)||_*.
```

Then any perturbation satisfying

```text
||delta theta|| < min_k mu_k / L_k
```

preserves all certified signs by the mean-value bound. Similar margins apply
to positive segment lengths and ray speed. This is conservative but gives a
precise optimizer-step radius for safe structural reuse.

The derivative supremum must cover the entire proposed parameter segment (or
a pre-certified parameter ball), not only the derivative at the current
parameter. If only a pointwise derivative is available, add a Hessian/Lipschitz
remainder and solve the resulting bound before accepting the step.

If the step exceeds the radius, the correct operation is local or global
recompile/recertification, not pretending the old word's derivative is full.

No local patcher currently satisfies that obligation. Existing root records
are insufficient because a previously rootless predicate can create a pair of
roots during an update. The current simple-root routine reconstructs the full
rooted and rootless registry and proves a restricted continuation theorem, but
returns neither a repaired compiler program nor rebuilt chart/rank/native
payloads. Therefore the executable policy today is full structural
recompilation after every geometry or ray update, with reuse reserved for
material-only updates.

The implemented CPU certificate is narrower and exact relative to its rational
inputs. It certifies one directional homotopy
`theta(eta)=theta_0+eta*delta_theta`, `eta in [0,r]`, for one strict
event-free single chart by Bernstein-bounding ray noncollapse, denominator
signs, ordered positive cuts, and all-site endpoint owner gaps over the whole
closed time/update rectangle. It returns zero for active-event, endpoint-event,
or multichart programs. This proves safe reuse on that segment only; it is not
a norm-ball certificate, and it does not continue moving event roots.

### 7.4 Birth/death seam

For finite density and color, a new run of length `ell` has

```text
beta(ell) = exp(-rho ell),
m(ell) = (1-beta(ell)) c.
```

At `ell=0` it is the identity `(1,0)`, so insertion/deletion leaves the
forward ordered product unchanged. However, if locally

```text
ell(theta) = max(0, a(theta)),
```

then one side has `D ell = 0` and the other has `D ell = D a`. Unless the
newborn generator is optically indistinguishable from what it replaces, the
render derivative jumps. The forward map can be continuous and still lack a
two-sided classical geometry derivative.

### 7.5 Event-time and exposure term

For a simple event root

```text
h(tau(theta), theta) = 0,
partial_t h != 0,
```

the implicit derivative is

```text
D_theta tau = -D_theta h / partial_t h.
```

For an exposure/integral objective split at the event,

```text
L(theta)
  = integral_a^tau f_-(t,theta) dt
    + integral_tau^b f_+(t,theta) dt,
```

Leibniz' rule gives

```text
D_theta L
  = integral_a^tau D_theta f_- dt
    + integral_tau^b D_theta f_+ dt
    + [f_-(tau,theta)-f_+(tau,theta)] D_theta tau.
```

At an identity birth/death seam, the forward values agree and the explicit
boundary term vanishes, though the interior derivatives may differ. At a
full-fiber material tie or another discontinuous dispatch, it need not vanish.

For a finite set of point samples away from event times, no exposure boundary
term appears; the sampled objective is simply piecewise differentiable. At a
sample exactly on a seam, use a declared one-sided/generalized convention or
fail closed.

### 7.6 Adaptive compiler decisions

Rank selection, chart splitting, and topology dispatch are discrete algorithmic
choices and generally do not possess ordinary derivatives. The practical
training contract should be:

1. compile and certify a structural program;
2. stop-gradient its discrete decisions for one bounded optimizer interval;
3. compute the exact VJP of the compiled surrogate;
4. bound primal and required derivative-action error versus exact replay;
5. enforce the trust region; and
6. recompile/reselect rank when the certificate or trust margin fails.

This can give a controlled surrogate gradient without claiming to
differentiate the compiler itself.

### 7.7 Missing geometry/ray tangent and loss-composition theorem

The current direct-kinetic continuous certificate proves the primal transfer
and referenced-material Jacobian/JVP/VJP actions for the actual cleared
second-form barycentric evaluator. The stable-stratum geometry code separately
differentiates exact node lengths and words. Those facts do not yet imply that
the compiled geometry or ray gradient approximates the exact physical
gradient between nodes.

For every accepted track-local chart, the missing local statement is a
declared sparse action-norm bound of the form

```text
sup_t ||G(t)-G_J(t)||                         <= epsilon_0,
sup_t sup_(v in V, ||v||<=1)
      ||D_theta G(t)[v]-D_theta G_J(t)[v]||   <= epsilon_1,
```

where `V` contains the supported local site-position, velocity,
weight-trajectory, and affine-ray directions; the interpolation nodes,
weights, rank, and dispatch are fixed. The translated-measure tangent supplies
the exact local optical-depth boundary terms, but it does not itself bound
their temporal interpolation error.

A conditional local-to-global composition lemma can already be closed. Let
sample `s` have local parameters `theta_s=A_s Theta`, exact output `y_s`,
compiled output `yhat_s`, exact local Jacobian `J_s`, and compiled local
Jacobian `Jhat_s`. Assume on the certified output domain:

```text
||yhat_s-y_s||                         <= epsilon_0,s
||Jhat_s-J_s||_op                      <= epsilon_1,s
||J_s||_op                             <= B_s
||grad phi_s(y)||                      <= Q_s
||grad phi_s(y')-grad phi_s(y)||       <= H_s ||y'-y||.
```

Then the sample's global-parameter gradient error obeys

```text
||A_s^T [Jhat_s^T grad phi_s(yhat_s)
         - J_s^T grad phi_s(y_s)]||
 <= ||A_s||_op [
      epsilon_1,s Q_s
      + (B_s+epsilon_1,s) H_s epsilon_0,s
    ].
```

Proof: add and subtract
`Jhat_s^T grad phi_s(y_s)`, apply the triangle inequality, use the tangent
bound on the first term, the cotangent Lipschitz bound on the second, and
`||Jhat_s|| <= B_s+epsilon_1,s`. The outer `A_s^T` is the actual sparse local-
to-global scatter, so no dense global dual is introduced.

For nonnegative normalized sample weights `a_s` with `sum_s a_s=1`, summing
the lemma gives

```text
||grad_Theta Lhat-grad_Theta L||
 <= sum_s a_s ||A_s||_op [
      epsilon_1,s Q_s
      + (B_s+epsilon_1,s) H_s epsilon_0,s
    ].
```

Uniform bounds therefore do not acquire an artificial requested-frame-count
factor under the paper's globally normalized loss. Mean RGB MSE has an
explicit constant Hessian; only a residual/output bound is needed for `Q_s`.
An incidence-overlap argument may sharpen this triangle bound, but is not
needed for correctness.

Thus the local-to-global algebra is conditionally proved. The genuinely open
mathematical gate is the direct-kinetic local `epsilon_1` certificate (D6),
including its supported parameter norm and denominator/fiber-speed margins.
Runtime float32 roundoff and exceptional dense interpolation rows require
either a separate bound or an explicitly measured non-theorem gate.

A sufficient lowered certificate is now derived. For each active internal cut
write `z_r(t)=-B_r(t)/A_r(t)`, with quadratic-or-lower `A_r,B_r`, and include
the affine ray-direction coefficients controlling `||d_0+t d_1||`. Let `q`
collect these `6(R-1)+6` or fewer track-local coordinates. Outward interval
dual arithmetic can bound continuously

```text
e_q = sup_t max_(output k, coordinate h)
      |partial_qh G_k(t)-partial_qh G_J,k(t)|.
```

The compiled branch must use the stored float64 node-length primal, attach the
exact real-arithmetic node-length tangent, apply the stored fixed fit, and use
the actual second-form barycentric evaluator. It must not differentiate node
times, rank, chart endpoints, or dispatch. If `H=D_theta q` is the existing
sparse world-to-cut Jacobian and

```text
L_qtheta = sup_(||v||_theta<=1) ||H v||_1,
```

then the chain rule gives `epsilon_1 <= e_q L_qtheta`. This closes the
mathematical sufficiency reduction without a global dual or new runtime state.
What remains open is a bounded track-local implementation, proof receipt, and
finite-difference/interval-witness gate. The detailed derivation and failure
branches are in Expansion Pass 8 of
`agent_notes/loose_notes/2026-08-04_03-19-39_worldfoam_scientist_feedback_and_fixed_site_source_closure.md`.

Until these statements exist, the geometry reverse is exact for the fixed
compiled surrogate, not continuously accuracy-certified against the physical
renderer. This is a quantitative error/certificate problem, not evidence that
another representation or Schur-like closure is required.

### 7.8 Owner-local support of the fixed-word length VJP

The current native geometry reducer is temporally bounded but spatially
conservative: each row emits dense bars over all `S` sites and repeats an
all-competitor node audit. The dense row result is not required by the
derivative itself.

For a certified word of distinct owners `i_1,...,i_R`, every internal cut is
the equality root of one adjacent owner pair. Reversing physical lengths first
gives the internal cut bars

```text
bar_z_r = ||d|| (bar_ell_r - bar_ell_(r+1)).
```

The implicit derivative of `z_r` touches only sites `i_r`, `i_(r+1)`, and the
ray. The direct physical-speed term touches only the ray. Taking the union
over cuts therefore gives support contained in the `R` word owners plus the
ray. Non-owner sites determine whether this frozen word remains valid, but
they do not enter its classical fixed-word differential.

A production sparse row result can consequently retain only owner ids,
`[R,3]` position/velocity bars, `[R,L_w]` weight bars, optional `[12]` ray
bars, and `[J,R]` recomputed lengths. Better, it can scatter those owner-local
bars directly into request-owned world buffers and return a tensor-free
receipt. This changes warm row scratch and dense-add work from `O(S)` per row
to output-sensitive `O(R)`, while the one caller-owned world gradient remains
`O(S)`.

This optimization is sound only after certificate authority is separated
from the warm VJP. Cold compilation/recertification must prove the all-site
owner/event statement and seal it to the exact world, ray, interval, word,
and chart contents. The warm VJP may then validate that identity plus local
denominator/speed/length margins and evaluate only the owner-supported
differential. Every geometry, weight-trajectory, or camera-ray update still
invalidates the certificate and triggers full recompilation. The current
dense audited reducer remains the independent oracle until sparse parity and
stale-certificate gates pass. Expansion Pass 9 in the dated source-closure
note contains the ABI, bound, and falsification matrix.

There is already native precedent for this split in the older fixed-word
boundary backend: node bars are accumulated into sparse Mobius incidences,
then finalized into active boundary bars and scattered only through the site
pairs that generated those boundaries. That code establishes a reusable
staged-adjoint lifecycle. The owner-local CPU lowering from `[J,W_b]`
physical-length cotangents to compact `positions0`, velocity, and
polynomial-weight bars (plus optional affine-ray bars) now exists source-only,
as does a suffixed fixed-camera fused-v1 native source path. The latter is
unbuilt, unrun, unselected, and not integrated into the coordinator. A
source-only all-block adapter now owns fresh zero scratch, clears one shared
four-byte reason mask, validates every block before the first write, enqueues
every status-gated accumulation, then finalizes every output ledger before one
fence/status read. This closes partial cross-block writes and rejects an actual
nonfinite aggregate destination as a source contract. It does not supply a
prospective bound or an exact/deterministic summation theorem. The fused route
must still match both the staged sparse and dense float64 oracles
without a row-by-global-site temporary or warm inactive-competitor scan. If
float32 owner-cut recomputation cannot match the float64 oracle under certified
margins, the compiler should lower compact per-node cut jets; this remains
`O(JR)` and frame-independent.

The stronger fixed-camera production form does not need to materialize the
intermediate `[J,W_b]` length cotangent at all. The ordered-word kernel already computes
`bar_ell_r`. Keeping only the previous/current pair yields each internal cut
bar `||d||(bar_ell_r-bar_ell_(r+1))`, while a scalar accumulation of
`(ell_r/||d||)bar_ell_r` yields the ray-speed bar. Thus the material reverse
and owner-local kinetic scatter can be fused in one `(row,node)` thread with
constant local reverse state. The staged one-length-bar path remains the
currently integrated source coordinator and correct parity oracle. The
suffixed fixed-camera fused-v1 source removes the length-cotangent allocation,
output, and copy, while retaining the frame-independent compiled primal
`[J,W_b]` lengths. It exposes no ray cotangent; trainable-camera reverse needs a
future, separately certified ABI. The final native memory gate should require
zero length-cotangent allocation and zero device-to-host cotangent copies.

### 7.9 Postwrite finite-ledger admission theorem

Consider one supplied ordered sequence of `B >= 1` fused blocks. Let `S_b` be
block `b`'s compact-site count, `S` the shared global-site count, `J_b` its node
count, `W_b` its total stored word incidences, and `L_w in {1,2,3}` its shared
weight-coefficient count. The source-level admission theorem requires all of
the following invariants:

1. one sealed cold token rejects duplicate block generations, binds their order
   and node cotangents, and owns fresh, pairwise storage-distinct, exactly-zero
   compact and global output scratch;
2. those outputs do not alias any supplied primal or cotangent, no hidden alias
   or concurrent writer exists, and every sealed input stays immutable through
   acceptance;
3. one fresh int32 status is cleared once, reason bits are monotone ORs, and the
   same status is passed to every phase;
4. one serialized stream enqueues all `B` validation kernels, then all `B`
   status-gated accumulation kernels, then all `B` finalizers, with no host
   read/fence between phases;
5. validation scans every compact ledger and the shared globals exactly once;
   finalization repeats those coverage counts after every accumulation; and
6. the coordinator accepts only after one successful completion fence/status
   read, revalidates the sealed identities, returns bars bound by signatures,
   and permits no persistent optimizer mutation before that acceptance.

Under those premises, a zero receipt proves every stored entry of the actual
float32 compact-material, global-position, global-velocity, and global-weight
ledgers is finite. If validation finds an invalid input, nonfinite proposed
contribution, or nonzero scratch entry, stream order makes every accumulation
observe a nonzero status before its first atomic, so the fresh scratch remains
zero. Otherwise validation has dry-run the same per-thread arithmetic under an
immutable input snapshot. Every finalizer runs after all atomic writes and
`isfinite`-checks its complete ledger coverage; the final fence makes all of
those status ORs visible before the host read. Thus a zero read implies finite
final ledgers for the execution that actually occurred.

For IEEE binary32 addition, `+Inf`, `-Inf`, and NaN remain nonfinite after later
finite additions (`+Inf + -Inf` becomes NaN), so an actual intermediate
overflow cannot silently heal before the final scan. This strengthening is a
backend premise, not yet native evidence: Metal fast-math, atomic-float, and
same-current-stream visibility still need rebuilt poisoned fixtures. The
minimal source theorem is only final-ledger finiteness. Signed zero is accepted;
subnormal flushing, finite underflow, cancellation/rounding error, and a
nondeterministic but finite atomic order are not rejected. A run that accepts
therefore does not prove correct rounding, staged-oracle parity, or that every
possible atomic order would accept.

Failure is fail-stop admission, not rollback. A nonzero postwrite receipt may
leave mutated or nonfinite scratch, but the sealed result is never constructed.
A phase exception after any enqueue triggers one abort fence and quarantine; if
that fence cannot establish completion, live roots are retained and the process
requires restart. A missing finalizer or count mismatch likewise cannot produce
the sealed accepted receipt. The token is one-shot. Raw phase APIs, hidden
aliases, a caller that bypasses the sealed result, and mutation after acceptance
are outside this theorem. The adapter also does not prove that its supplied
sequence is the executor's complete active manifest, and acceptance merely
authorizes a later commit: fail-atomic optimizer mutation still requires the
separate out-of-place candidate updater.

The exact logical output-scratch payload during the transaction is

```text
M_fused_tx_output
  = 16 sum_b S_b + (24 + 4 L_w) S + 4 bytes,
```

where the last term is the status. This excludes prepared primals, node
cotangents, allocator padding, and persistent optimizer/candidate state. All
compact bars coexist until finalization, so the term is a sum over blocks, not
the staged route's maximum one compact bar. A failed completion proof may retain
one such payload in restart-required quarantine; retrying in that process is
forbidden. Ownership transfer after acceptance can avoid a second copy, while
copy/add into separate persistent bars costs another output-sized pass/buffer.

Validation and accumulation each execute the full fused reverse, so reverse
arithmetic runs twice and is
`Theta(2 sum_b J_b W_b (1+L_w))` (with fixed `L_w <= 3`,
`Theta(sum_b J_b W_b)`). Pre- and post-scans inspect exactly
`8 sum_b S_b + 2(6+L_w)S` float entries in total. The protocol uses `3B` phase
launches, one status initialization, and one final fence/read. Fresh scratch
creation additionally zero-fills `4 sum_b S_b + (6+L_w)S` float entries and may
issue framework allocation/clear launches; those are not included in the `3B`
native phase count. It allocates no `[J_b,W_b]` length-cotangent tape. A later
optimizer commit is separate work.

## 8. Work Bounds

### 8.1 Exact streamed replay

Using `c` as a semantic shared-chart shorthand only, not a required dense
storage layout, the current conservative route scans every run at every
requested time but uses constant local reverse state:

```text
T_exact = Theta(sum_c F_c R_c),
M_exact_reverse_interaction
  = O(S + I + B_p K),
```

plus the resident compact word/topology. It is memory-light in `F` but does not
remove frame-linear word work.

### 8.2 Compiled route

For the current cost proxy, exact forward plus reverse is

```text
C_exact = 3 sum_c F_c R_c.
```

For legibility, first take the current special case with one semantic chart
set shared across tracks and define `R_c=sum_p r_{p,c}`. With exact word
evaluation at nodes, dense coefficient fitting, verified linear barycentric
sample weights, and row-local fallback, the **current** compiled proxy is

```text
C_compiled_current
  = 2 sum_c J_c R_c
    + P sum_c J_c^2
    + P sum_c F_c J_c
    + N_B sum_c F_c J_c
    + sum_c N_fb,c J_c^2.
```

The fourth term is repeated sample-weight construction in every spatial
block. A future validated global weight cache would replace `N_B` by one:

```text
C_compiled_global_weight_cache
  = C_compiled_current
    - (N_B-1) sum_c F_c J_c.
```

For a genuinely ragged track-local program, do not materialize the semantic
common refinement. The structural/node and sample terms become sums only over
stored `(p,c)` incidences, schematically

```text
2 sum_(p,c) J_(p,c) r_(p,c)
+ sum_(p,c) J_(p,c)^2
+ sum_(p,c) F_(p,c) J_(p,c)
+ fallback executions.
```

Interpretation:

- `2 J_c R_c`: one exact node-word forward and one node-word reverse;
- `P J_c^2`: current per-track coefficient fit;
- `P F_c J_c`: transfer evaluation and sample-to-node cotangent reduction;
- `N_B F_c J_c`: current per-spatial-block barycentric weight construction;
  the proposed global cache changes `N_B` to one; and
- `N_fb,c J_c^2`: exceptional dense interpolation rows.

The `2 J_c R_c` term requires a step-scoped lifecycle. A superficially
bounded request-local implementation that refreshes and reverses the compiled
word program once per `K` chunk actually pays

```text
2 ceil(F/K) sum_c J_c R_c,
```

which is still linear in frame density. The landed CPU/fake-native paper step
uses spatial-bundle outer / temporal-chunk inner order: node charts are built
once, every chunk accumulates into the same node cotangents, and the material
word VJP runs once after complete bundle coverage. The native shader ABI has
the required accumulate-only semantics, but this invocation frequency is not
yet verified on rebuilt Metal.

The newest dense source candidate applies the same factorization to the
full-geometry reverse: it retains one node cotangent per active block, runs one
executor-sealed material-plus-length word VJP, immediately reduces the bounded
`[J,W]` length bar into request-local kinetic bars, and merges only a sealed
request delta into a world-bound step accumulator. It requests a completion
fence after every sample launch, after every active-block scatter, and after
every request-delta commit, rather than trusting Python reference release to
bound an asynchronous command queue. The executor releases `[J,W]` only after
an exact sealed geometry-reduction receipt; its telemetry intentionally proves
`fenced_and_reduced_not_globally_committed`, while the request and step layers
separately own the scatter and commit fences. MPS accepts only the canonical
`torch.mps.synchronize()` fence/provenance pair; CPU fake-native tests retain an
injected fence.

The source ownership protocol now closes the earlier local-prepare gap. Before
native prepare begins, the session installs one sealed sample lifetime rooted
in the sample block, world token, background, loss state, node bar, cone, and,
once returned, the prepared native payload. No second launch is allowed while
that lifetime is outstanding. Settlement owns the only completion fence,
validates the launch receipt, commits coverage/counters, and then releases all
completion-safe roots immediately. A successful session seal requires

```text
native_sample_prepare_count
  = native_sample_launch_count
  = native_sample_completion_fence_count,
maximum_simultaneous_sample_lifetime_count = 1,
outstanding_sample_lifetime_count_at_seal = 0,
sample_lifetime_history_retained = false.
```

The dense caller separately leases the CPU decode-transfer source and the
sample-materialization predecessors until settlement. Its current-lifetime
reference set also includes target and sample blocks, the executor lifetime,
retained active reverse state/compact bar, material or full-geometry execution, native
VJP, geometry reduction, and geometry completion. Those references feed the
accumulator-owned restart-required quarantine. If the completion fence fails,
completion is unknown, the one lifetime remains rooted, and retry/abort release
is forbidden. If the fence succeeds but later receipt validation fails, the
safe roots are released without issuing a redundant second fence.

The lazy native material-step route now owns one bounded restart-required
carrier around its visible lane construction, sparse transfer, sample,
reverse, and lane-release lifetimes; it no longer clears those roots after an
unknown completion fence. Union-local map transfer and compact-material gather
now have source-written predecessor lifetimes installed before their first
device operation and retired after a proven fence without a bundle/lifetime
cycle. Four earlier/lower gaps still block an accelerator claim: top-level
device allocation/`zero_` has no preinstalled transaction owner, cold union-map
validation chains a device-to-host receipt conversion, native-forward
enqueue-failure ownership is uncertified, and an arbitrary callable is not a
canonical backend/stream completion capability.
The route therefore still rejects every non-CPU device. Target decode otherwise keeps each
full frame on CPU, gathers only the selected `[N,3]` chunk, and transfers that
bounded chunk once. The older standalone full-geometry finalizer remains
CPU/fake-native-only. All of these latest executor, dense-request, and test
edits are source-written but unrun; the installed extension is stale, and real
device-fence, decoder, command-buffer, allocator, and RSS behavior remain
unmeasured.

Add structural compilation/certification and sparse finalization:

```text
T_step
  = T_structural(S,P,E,R)
    + C_compiled_current
    + O(E + I + S + PF).
```

For the landed active compiler, predicate construction is
`O(U S R_max)` across unique witnessed owner words and the current monotone
closure/certification loop is `O(W (S log S + S R_max))` across cumulative
root-complement discoveries. This is independent of requested frames but is
not yet a final output-sensitive `E/R` theorem. The remaining target is a
certified event queue or neighbor structure with initialization plus event
processing in `E`, active run/incidence output, and logarithmic/local update
factors.

### 8.3 Frame-density theorem

Fix:

- world parameters and their temporal basis;
- the continuous camera program;
- the physical interval;
- error tolerance `epsilon`;
- structural event set and chart partition;
- active owner words; and
- certified ranks `J_c`.

Then increasing only `F` leaves

```text
T_structural,
sum_c J_c R_c,
E,
R,
J_c,
and world-gradient buffer size
```

unchanged. Only

```text
Theta((P+N_B) sum_c F_c J_c + PF)
```

in the current shared-chart implementation; the proposed global weight cache
changes `P+N_B` to `P+1`. In the ragged formulation, use
`Theta(sum_(p,c) F_(p,c) J_(p,c) + PF)` plus explicitly counted weight-builder
and fallback executions. Only stored incidences may appear.

Consequently, `O(PF)` is valid only as shorthand for the frame-density limit
in which all `J_(p,c)`, charts, and dispatch rules are fixed constants. With
rank visible, the common sample slice is

```text
Theta(sum_(p,c) F_(p,c) J_(p,c)
      + sum_(p,c) N_fb,p,c J_(p,c)^2
      + PF),
```

plus chart lookup. The theorem is about moving ordered-word/world work off the
requested-frame axis, not removing rank-weighted temporal evaluation.

This separation is now exercised end to end at CPU/fake-native scope: changing
`K` from one to four changes request and sample-launch counts without changing
the number of compiled node forwards or material word VJPs, and increasing
`F` from five to 41 leaves those counts and retained runtime bytes invariant.

This is the desired World-Tubes-shaped split. It is not a sublinear total
renderer theorem because producing/comparing `PF` colors costs `Omega(PF)`.

The theorem fails if rank/event selection is itself based on the requested
frame grid. Continuous certificates or a frame-independent probe policy are
therefore part of the claim, not merely test infrastructure.

It also needs a seam condition. The current CPU program fails closed for a
rational sample strictly inside a nonexact algebraic root-isolator
neighborhood. Arbitrarily densifying a sample grid can eventually enter such a
neighborhood. A full-interval theorem therefore needs exact
rational-sample-versus-algebraic-event comparison in runtime dispatch, or must
explicitly restrict accepted sample sets to those outside all unresolved
neighborhoods. Refining isolators in response to the requested grid would make
part of compilation depend on `F` and is not evidence for the strong theorem.

### 8.4 Break-even condition

Ignoring constant compile/finalize terms and fallback, a shared chart in the
**current** material trainer has a temporal break-even only if its exact
per-sample slope exceeds its compiled slope:

```text
3 R_c > (P+N_B) J_c.
```

Since `R_c = P r_bar_c`, this is approximately

```text
3 r_bar_c > (1 + 1/B_p) J_c
```

for full blocks and large `P`. With a validated global weight cache the exact
condition becomes `3 R_c > (P+1)J_c`, approximately `3 r_bar_c > J_c`.
Dense fallback increases the compiled slope. A two-run word
with rank sixteen can correctly have no break-even. The correct renderer is a
hybrid:

- exact streamed replay for shallow/low-rank-unfriendly charts; and
- compiled replay when run reuse exceeds rank and compile cost.

Frame-density-independent heavy work is an architectural capability, not a
promise that compilation wins every chart.

## 9. Peak-Memory Bounds

### 9.1 Desired asymptotic bound

With one spatial block and one temporal block in flight, a template-free
implementation should satisfy

```text
M_peak
  = O(
      M_world(S,L)
      + M_optimizer(S,L)
      + P
      + E
      + sum_(p,c) J_(p,c)^2
      + B_structural_store
      + M_cold_compile_scratch
      + max_b [
          M_live_structural,b
          + S_b + R_b + I_b
          + B_p sum_c J_c
          + C I_b
          + B_p K
          + K J
        ]
    ).
```

`B_structural_store` is the explicit byte cap of the implemented CPU
entry/byte-bounded LRU. Cached artifacts retain only observation-invariant
programs, topology/equal-rank descriptors, and ragged samplers; they retain no
targets, observations, native tokens, runtimes, or gradients. The current warm
step prepares one native token/lane under `M_live_structural,b` and releases it
after the spatial request. `M_cold_compile_scratch` remains an explicit gap:
the store evicts before compiling a candidate and caps the admitted artifact,
but the compiler's temporary Python/tensor peak is not yet preflight-bounded or
measured. It is independent of requested `F` by construction, but it still
belongs in any honest absolute-memory claim.

For a pipeline that streams one chart at a time, `B_p sum_c J_c` can be
reduced to `B_p J`. `I_b` and referenced sites are `O(R_b)` for compact sparse
words, although constants and cross-chart duplication must be measured.

There is no `F R` interaction tape and no resident `P F` target/ray tensor.
Some host/provider paths may retain cheap `O(F)` sample identity/time metadata.
The native prepared/replay state no longer owns a global `[F]` or chart-local
`[F_c]` time clone: each launch receives only its live float64 `[K]` block.
Strict end-to-end `O(K)` time metadata therefore still requires the host
sampler/provider to generate or stream identities rather than retain them.

`M_optimizer` may multiply model bytes by gradients and optimizer moments, but
it remains independent of `F` when the world uses `O(SL)` parameters.

The displayed `+ P` term is the current affine ray-program storage, not a
fundamental per-pixel world requirement. A calibrated camera plus pixel lattice
can generate the next `B_p` ray coefficients procedurally, reducing resident
ray state to the camera program plus `O(B_p)` staging.

### 9.2 Exact current source tensor-payload formulas

For the current fixed-topology compact reference, the audited per-block tensor
payloads are:

```text
global source parameters              36 S + 48 P bytes
global site-gradient buffers          36 S bytes

compact topology
  16 B_p + 8 + 12 R_b + 4 I_b + 16 B_b + 8 S_b bytes

compact world and all resident charts
  72 S_b + 96 B_p + 40 B_b
  + 64 B_p sum_c J_c
  + 8 sum_c (J_c + J_c^2)
  + 48 C I_b + 24 R_b bytes

strict/eval target-plus-ray block      36 B_p K bytes
material target-only block             12 B_p K bytes
bounded material camera check          24 B_p bytes once per spatial block.
```

As a scale check, float32 node state plus node cotangents at
`B_p=8192, J=16` occupy

```text
2 * 8192 * 16 * 4 channels * 4 bytes = 4 MiB.
```

At `K=8`, material training carries a `0.75 MiB` target block and no prediction
or explicit sample-ray block. Its one-row exact camera validation scratch is
`0.1875 MiB`; target-only staging saves `1.5 MiB` relative to the bounded
strict target-plus-ray payload. Topology, world state, optimizer state,
allocator overhead, and command buffers are additional, but these core
payloads do not imply a 32-GB machine requirement.

`B_b` is the active boundary-row count and `S_b` the referenced site count.
These formulas count tensor payload only, not Python objects, allocator
fragmentation, command buffers, framework caches, or driver memory.

For `N` ragged observations in one actual-rank `J` launch, the sealed
CPU/source sample block retains

```text
4 N J + 24 N bytes.
```

That count is `4 N J` float32 weights, `12 N` RGB targets, `4 N` CPU row ids,
and `8 N` flat provenance ids. The current production wrapper no longer makes
the redundant CPU int64 row conversion. Native prepare adds one MPS int32 row
copy (`4 N`) and 20 bytes of config tensors. The minimum named public-tensor
payload simultaneously rooted at that prepared boundary is therefore

```text
4 N J + 28 N + 20 bytes,
```

before adding the live node chart/background aliases and any retained bounded
CPU/device target chunk. The loss-only compiled launch allocates no explicit
prediction output.

The dense caller applies a separate conservative preflight to source-visible
sample materialization:

```text
max(
  N (8 J + 12) + interpolation_scratch + 16 K_sub,
  N (16 J + 32)
),
```

where

```text
interpolation_scratch
  <= 4096 + 512 J + 8 J^2 + K_sub (1024 + 512 J).
```

Its materialization lease keeps the float64 weight source, positions/row ids,
chunk targets, and the sample block alive until the completion fence; the CPU
decode object likewise retains its transfer source. These are predecessor
lifetime bounds, not measured allocator peaks.

Because each launch has `N <= K`, the session permits exactly one outstanding
sample lifetime, and successful settlement releases it immediately without
retaining history, the retained sample-axis source tensor payload is
`O(KJ+K)`, or `O(K)` when `J` is fixed. It is not multiplied by launch count or
requested frames. This says nothing about total sampled work, which remains at
least `Omega(PF)`. Float64 implementation temporaries beyond the declared
preflight, decoder internals, Python objects, driver/command-buffer storage,
allocator slabs, and measured RSS remain outside these formulas and must be
reported separately.

For one heterogeneous spatial request, the union-local assembler retains

```text
8 S_union + 8 sum_b S_b bytes    source union plus compact-to-union maps
16 S_union bytes                 caller-owned request material bar
```

and allocates no per-request global `16 S` material bar. These are logical
payload counts; transfer temporaries, allocator storage, and peak remain
unmeasured.

The step-scoped block-major material runner changes the correct live-state
bound from a request-local maximum to a maximum over spatial bundles. For one
bundle `q`, before its sequential compact scatters finish, the dominant
logical float32 payload is

```text
M_live,q
  = 16 S_union,q
    + sum_(b in q) (16 S_b + 32 R_b J_b)
    + max_(b in q) 16 S_b
    + O(1) bytes.
```

Here `16 S_b` is the compact material snapshot, `32 R_b J_b` is node chart
plus node cotangent, and the final maximum is one compact material-bar scratch.
Spatial bundles execute sequentially, so the step peak is `max_q M_live,q`,
not `sum_q M_live,q`. This sum *within* a bundle is necessary because all
heterogeneous node cotangents must survive across its `K` chunks; the older
`32 B_p J_max` formula is valid only for a separately proved one-native-block-
at-a-time replay schedule. Material-only reverse adds zero `[J,W]` bytes. The
source-level full geometry candidate adds at most one sequential `4 J_b W_b`
length bar to this logical bound, plus its geometry-chain scratch; it has not
yet passed the quiet-host CPU gate or a rebuilt-native gate.
Per-active-block and per-request commit fences now prevent compact bars and
request deltas from queuing across their ownership boundaries. Target staging
and one fenced ragged sample lifetime are counted separately. The successful
path retains no lifetime history. Dense quarantine now roots the CPU transfer,
materialization predecessor, current target/sample, outstanding executor
lifetime, retained active reverse state/compact bar, material/full-geometry
execution, native VJP, reduction, and completion state until completion is
proved. A source-written, unrun fake-native test now reaches sample settlement,
injects a failed completion fence, and inspects those retained roots.
Failure-path quarantine is deliberately excluded from the successful-step peak:
it is a bounded one-lifetime fail-stop leak after loss of completion proof,
forbids optimizer authorization, and requires process restart rather than
unsafe release. The exact sealed target loader now covers post-enqueue failure
by retaining its selected-read, CPU source, and returned device roots in the
request quarantine when abort fencing fails. The lazy route now has equivalent
bounded quarantine for every root visible after lane construction begins, and
the source-written union-map/gather lifetimes cover two predecessors that were
previously outside it. Accelerator execution remains rejected because the
top-level allocation/zeroing transaction, cold device-to-host union-map
receipt, native-forward output exception, and backend/stream fence contracts
listed above remain open.
None of these statements is an allocator-peak measurement.

They are exact formulas for the listed source tensors, not an exact process
peak and not a max-only whole-session bound. The session's persistent block
state is the sum of each retained block's topology, owner binding, schedule,
and cached-token payload; sum those blocks before adding the maximum one-block
live staging/scratch term.

#### Fused-v1 versus staged full-geometry live set

The suffixed fixed-camera fused-v1 path removes the **cotangent**
`[J_b,W_b]` tensor. It does not remove the compiled primal lengths, and its
current all-block fail-atomic transaction introduces other live tensors.

For active native block `b`, let:

```text
R_b = lowered row count,
J_b = node count,
W_b = ordered-word count,
s_b = compact referenced-site count,
rho_b = maximum word count of one row in the block,
C_w = weight coefficient count (currently 1..3),
S   = global world-site count,
U   = request-union site count (U <= S).
```

The source preflight derives the exact owned fused-preparation payload

```text
M_fused_prepared
  = 4 sum_b [R_b (J_b + 14) + s_b (6 + C_w) + 13].
```

The terms are row node times, near/far values, affine rays, compact
position/velocity/weight values, and two small config tensors. Resident
topology and material tensors are aliases and are not counted again.

The one-shot transaction then owns

```text
M_fused_output
  = 16 sum_b s_b             compact RGBA bars
    + 4 S (6 + C_w)          global float32 geometry bars.
```

Converting the accepted global geometry bars to the public CPU-float64 ABI
adds

```text
M_fused_bridge_destination = 8 S (6 + C_w),
```

while the source's bridge-visible accounting correctly reports source plus
destination as `12 S (6+C_w)`. The destination is the request geometry delta
already present in the common request bound; it must not be double-counted.

The earlier comparison counted only the staged mode's native physical-length
cotangent,

```text
M_staged_length_bar = max_b 4 J_b W_b
```

but that is not its complete reverse phase.  For the fixed-camera path, the
sparse reducer's deterministic source-visible tensor upper bound for block
`b` is

```text
H_b
  = 4 J_b W_b
    + (56 + 8 C_w) s_b
    + 16 J_b rho_b
    + 8 J_b
    + 8 (37 + 2 C_w) rho_b
    + 608
    + V_b,

V_b = 1 + max(J_b rho_b, 3s_b, C_w s_b)
      one-byte finite-mask plus scalar validation scratch.
```

`H_b` includes the native `[J_b,W_b]` input, the maximum CPU row copy, compact
site ids and geometry outputs, row source and parameter bars, and the
conservative one-node scratch.  Enabling trainable camera rays adds
`96(T_b+1)` bytes, where `T_b` is the block's distinct track count, and adds
`12T_b` as a candidate inside `V_b`.  The
sequential compact RGBA commit bar adds another `16 s_b` bytes outside this
reducer preflight.

The state common to both reverse modes is useful to state explicitly.  Let

```text
A  = (64 + 8 C_w) S + 4
     whole-step material/loss/CPU-f64 geometry accumulator,

G  = 8 S (6 + C_w)
     request-local CPU-float64 geometry delta,

X0 = sum_b [32 R_b J_b + 16 s_b + 4] + 16 U + 16
     active node bars/primals/losses plus union material bar and scalars,

L  = frame-free resident native lane payload.
```

Fenced one-block reduction therefore gives the tighter staged reverse-phase
logical envelope

```text
M_staged_phase
  <= L + A + G + X0 + max_b [16 s_b + H_b].
```

The implemented fail-before-work admission is deliberately a little looser:
it uses `16 max_b s_b + max_b H_b`, because the two maxima may come from
different blocks.  Before this audit, the dense request included only
`max_b 4J_bW_b` in its active-state cap and enforced `H_b` only as a separate
non-additive cap.  That was a real accounting gap.  The source now exposes the
reducer's tensor-allocation-free preflight and composes the complete maximum `H_b`
with the still-live request and step state.  These edits are source-only and
unrun. The helper still allocates ordinary Python tuples/maps and does not
measure allocator storage.

Current fused v1 instead has the source-visible reverse/bridge envelope

```text
M_fused_phase
  <= L + A + X0
     + M_fused_prepared
     + 16 sum_b s_b
     + 12 S (6 + C_w).
```

The last term is the simultaneous global float32 geometry source and its
CPU-float64 request destination.  Thus removing `[J_b,W_b]` does **not** make
current fused v1 the likely memory winner.  At `C_w=3`, that pair alone is
`108S` bytes, and all prepared blocks and compact material outputs coexist
until transaction acceptance. Fused v1 can still win bandwidth or launch
economics, but its memory case must be measured rather than presumed. These
are logical source-tensor envelopes, not allocator or process peaks; target
decode/sample phases must be composed separately by lifetime, not merely
listed under independent caps.
The source now reports the reverse-only `L + active` sum and the sum of its two
existing policy caps. This aligns reverse reporting with the displayed
formula; it does not compose the target/sample phases or prove a whole-request
system peak.
The current fused admission is additionally conservative by one unused
`16 max_b s_b` sequential-compact-scratch term inherited from the shared active
helper; removing that accounting-only charge is safe after the mode-specific
phase helpers are split, but it is not the main memory optimization.

The mathematically natural memory-v2 output is request-union-local:

```text
all blocks atomically accumulate geometry into [U, 6+C_w]
  through their sealed compact_to_union_i64 maps
  -> validate/finalize/fence the complete request transaction
  -> bridge only [U, 6+C_w] to CPU float64
  -> index-add once through the union's global source-site ids at commit.
```

This is an exact index-space factorization, not new optical-transfer math. Let
`P_b in {0,1}^{S x s_b}` scatter block-local bars to global sites,
`Q_b in {0,1}^{U x s_b}` scatter them to the request union, and
`P_U in {0,1}^{S x U}` scatter the union to global sites. The existing cold
bundle certificate proves

```text
P_b = P_U Q_b.
```

For block-local geometry bars `g_b`, current v1 and union v2 therefore satisfy

```text
G_v1 = sum_b P_b g_b
     = P_U (sum_b Q_b g_b)
     = G_v2
```

in exact arithmetic. Floating atomic accumulation order remains a numerical
parity tolerance issue, not a representation difference.

The current Metal kernel already computes `left_owner` and `right_owner` in
compact block coordinates before translating them through
`source_site_ids_i64`.  That tensor must remain the global provenance map.
Memory v2 adds a separate sealed `compact_to_union_i64` destination map and
sets the geometry output site count to `U`; it must not reinterpret the global
ids. Device admission should retain and cross-check three distinct identities:
block compact-to-global provenance, block compact-to-union destination, and
union-to-global provenance, proving
`union_ids[compact_to_union] == source_ids` before any write. The
mode-specific source/destination pair then becomes

```text
12 U (6+C_w)
```

instead of `12 S(6+C_w)`.  The exact **geometry source/bridge** saving versus
current fused v1 is

```text
Delta_memory_v2 = 12 (S-U) (6+C_w),
```

or `108(S-U)` bytes at `C_w=3`. Existing MPS union ids and compact-to-union
maps already cost `8U + 8 sum_b s_b` in lane accounting, so v2 must alias them.
If the CPU request delta needs a new explicit union-to-global `int64` commit
map, charge another `8U`; the net counted improvement is then

```text
12(S-U)(6+C_w) - 8U.
```

Consequently v2 promotion must measure `U/S`; the bridge is never larger, but
the complete source-tensor delta is not automatically positive for a nearly
global union. It preserves the existing all-block fail-atomic transaction.
Per-block streaming is a later option only if preparation
dominates: naïve persistent block commits violate request atomicity, while a
sound streamed design needs transaction-local union scratch, a presealed
manifest, bounded block batches, fences, one final status, and exactly one
persistent commit.

The global full-track CPU atlas costs

```text
64 P sum_c J_c
+ 8 sum_c (J_c + J_c^2)
+ 48 C I_global
+ 24 R_global bytes
```

and is explicitly not the production path.

### 9.3 Dense certificate counterexample

The current strict dense forward-dual oracle has derivative dimension

```text
D = 5 B_b + 12 B_p + 4 I_b + 4 S_b.
```

Its pointer-slot lower bound is

```text
Omega(max(16 D^2, 64 B_p J D)) bytes,
```

before any rational interval objects. This is the source of catastrophic
memory, not an intrinsic requirement of WorldFoam. It must remain a tiny-
fixture oracle. Production certification must use sparse directional actions
and local operator bounds.

### 9.4 Event and global-chart caution

`O(E)` storage is frame-density independent but not automatically small. If
different tracks have staggered events, taking one global union of all event
times and storing every track at every global chart can create an avoidable
`O(P E J)` atlas. Use ragged track-local or block-local chart schedules and
store only active chart-track incidences. A global event partition is valid
only when its measured duplication is bounded.

## 10. Keep, Restrict, Replace

### 10.1 Keep

1. **P0 affine-transfer monoid.** It is the exact pointwise closure of the
   current physical renderer.
2. **Two-pass prefix-only VJP.** It removes `O(R)` thread-private reverse
   arrays without changing gradients.
3. **Sparse track-face incidence.** Never restore a dense
   `track x all-boundaries` buffer.
4. **Exact word evaluation at compiler nodes.** Approximate only the temporal
   total transfer, not depth order inside each node.
5. **Joint primal and sparse derivative-action certification.** Forward-only
   validation is mathematically insufficient.
6. **`B_p x K` streaming and one global loss normalization.** This is the
   correct memory shape.
7. **Exact-versus-compiled routing.** Exact replay is a legitimate fast path,
   not a failure of the compiler.
8. **Event-scaled structural contracts.** Frame samples must not define the
   topology program.

### 10.2 Restrict

1. Call the current `[S,5]` fixed 4D Euclidean world an
   **affine-weight sliced power special case**, not general moving 4D foam.
2. Keep the first native training theorem to P0 scalar extinction and
   view-independent RGB.
3. State the current reverse as a **frozen-program VJP**.
4. Keep ordinary depth, affine cameras, fixed near/far coordinates, isolated
   event roots, and finite nonzero ray speed as initial theorem assumptions.
5. Condition frame-density independence on fixed physical interval, tolerance,
   event set, and rank.
6. Keep arithmetic exactness relative to supplied binary64 values distinct
   from robustness to calibration uncertainty.

### 10.3 Replace

1. Replace the general production world frontend with direct kinetic 3D sites
   and weights using a small shared temporal basis.
2. Replace the dense global derivative/certificate representation with sparse
   JVP/VJP actions and track-local operator bounds.
3. Replace the global full-`P` atlas template with template-free block-local
   compilation and ragged event/chart schedules.
4. Replace stored global sample-time/id arrays with a streamed iterator if
   strict `O(1)`-in-`F` metadata is required.
5. Replace any unconditional "compiled is faster" rule with measured,
   chart-local route selection including fallback frequency and bytes moved.

### 10.4 Do not replace

Do not discard the current compiler and shader work wholesale. The direct
kinetic frontend changes

```text
site parameters
-> active implicit face coefficients
-> face-coefficient VJP
```

but preserves the rest:

```text
kinetic event charts
-> ordered owner words
-> exact segment transfers
-> J-node total-transfer atlas
-> B_p x K residual reduction
-> node cotangents
-> constant-state word VJP
-> sparse face/site accumulation.
```

## 11. Proof and Implementation Obligations

### Gate A: representation adequacy

Implement only a CPU fixture first:

1. one translating boundary representable by fixed 4D slices;
2. one rotating boundary from two affine kinetic 3D sites;
3. fit the fixed 4D model with increasing site count;
4. plot error, active fixed-normal face-piece/chart-switch count, candidate
   pair count, site count, and event count versus tolerance; and
5. kill the fixed 4D general-world claim if the predicted
   `Omega(Theta L/epsilon)` active-piece trend appears. Do not call this a
   linear site-count lower bound without a separate pair/site argument.

### Gate B: native kinetic face ABI

The CPU frontend and stable-stratum reverse now return, per active track-face
incidence:

```text
cut value z(t),
cut parameter VJP,
denominator margin,
event predicates,
owner certificate facts.
```

The source lowering is now implemented: real `(track,chart)` rows are packed
into bounded actual-rank blocks, a fake-native CPU lifecycle adapter exercises
the existing precompiled-length ABI, and ragged samples plus heterogeneous
compact bars reach one outer coordinator result. The remaining gate is to
bind dataset-generated programs to this path, rebuild the extension, verify
real runtime forward/VJP parity, and measure allocator/bandwidth behavior.
Keep structural choices sealed and fail closed when a CPU
certificate/provenance digest no longer matches.

### Gate C: bounded-degree event completeness

The exhaustive CPU reference is now green for all pair near/far roots, all
finite triple concurrences, algebraic root grouping, separate denominator-only
analytic guards, exact left/right word filtering, ray-collapse rejection, and
fail-closed persistent/full-fiber/simultaneous degeneracies.

The implemented active-boundary closure uses endpoint owners against all sites,
every active cut against all competitors, and active denominator guards. Cache
predicate construction by unique owner word `U`, giving `O(U S R_max)` source
work. Do not hide the current `W` root-complement word discoveries and all-site
certificates: they add `O(W (S log S + S R_max))`. An `O(delta R)` neighbor-only
route additionally requires a certified kinetic regular/Delaunay or
conservative Cech-style supergraph; its construction and maintenance cost is
separate. Never infer events from requested frames.

### Gate D: frozen-program gradient contract

For samples separated from every event by a declared margin:

- compare direct kinetic exact replay and compiled forward;
- compare material, trajectory, weight, and camera VJPs;
- report both primal and normalized directional-gradient error;
- vary `K` and `B_p` and require invariant loss/gradients; and
- perturb within and just beyond the certified trust radius.

The material-only `K` partition check is green through the complete
CPU/fake-native block-major coordinator (`K=1` versus `K=4`, including an
independent direct-autograd loss/bar oracle and one optimizer authorization).
The final bullet is green on CPU for the exact directional event-free
single-chart certificate and for the restricted multichart separated-
singleton-simple-root reference. The latter reconstructs all predicate sources,
proves root-free complements, re-isolates endpoints, and checks semantic
left/right owner words; a fixed-seed differential suite compares every accepted
candidate against a fresh exact compile. Warm/output-sensitive affected-source
repair and derivatives through event/chart/rank choices remain open; unsupported
eventful strata still recompile.

### Gate E: memory contract

On a quiet approved runtime, measure rather than infer:

```text
world parameter bytes,
optimizer bytes,
event/topology bytes,
atlas/node bytes,
target/ray staging bytes,
residual/prediction bytes,
temporary kernel bytes,
allocator current and peak bytes,
command-buffer in-flight count.
```

For one fixed 300-frame dataset/camera grid, fixed world/interval/tolerance,
and the checked-in schema-v3 endpoint-including requested subsets
`F_requested = 8,64,300`, require:

```text
world parameter bytes                         invariant,
world/node reverse work                       invariant,
reverse interaction peak excluding I/O        ratio <= 1.10,
resident target/ray block                     O(B_p K),
per-sample run tape                            zero,
dense certificate/global atlas                absent.
```

The structural version of this gate is green at CPU/fake-native scope for
`F=5` versus `F=41`: retained runtime bytes, node-forward count, material-word
VJP count, and ordered-run/node interactions are invariant while streamed
sample interactions grow. That evidence predates the latest lifetime edits.
The current source/test contract additionally requires exactly one outstanding
sample lifetime, no retained lifetime history, zero outstanding lifetimes at
seal, and exact

```text
native sample prepares = native sample launches = sample completion fences
```

on every successful sealed session. It also requires CPU-transfer and
sample-materialization predecessor leases to survive until the settlement
fence and the dense failure quarantine to retain all current sample/reverse
roots when completion is unknown. The lazy source route now supplies the same
bounded fail-stop rule for visible lane/sparse-transfer/reverse roots. Its
pre-lane union-map transfer and compact gather are now source-covered by
caller-visible lifetimes and proven-fence retirement, while top-level
allocation/zeroing, cold device-to-host map validation, native-forward
exception, and backend-fence contracts remain open. These latest assertions
are unrun. The
material-only native source path also has no `[J,W]` output allocation. Actual
allocator current/peak, command-buffer overlap, launch frequency, bandwidth,
and RSS remain unverified until a safe native runtime window. The lower-level
target-loader lease and lazy-route quarantine are source-written but unrun;
the four revised earlier/lower lazy lifetime gaps remain blockers. The opt-in gate now binds direct selected-pixel
receipts, a transitive Python/native source manifest, an applied effective
`<=2 GiB` MPS allocator limit, lower-bound public counters from the configured
5.0-ms sampler, and a separate 0.25-second sampled process-group watchdog that
terminates above 4 GiB. These latest producer/verifier/native/test changes are
source-only and unrun; the sampled counters and watchdog are not exact allocator
peaks.

A source red-team found that the earlier unrun producer changed
`F_dataset=F_requested`, retained one camera signature per dataset frame in
every artifact, and repeatedly rebuilt those signatures in warm validation.
The repaired source removes that artifact payload, makes provider warm
generation checks constant-size in `F_dataset`, hoists static-camera equality
to one per-view cold certificate, and requires the verifier—not a hardcoded
step flag—to derive structural invariance across the fixed-dataset matrix.
Explicit requested-sample manifests and target/output work remain linear in
`F_requested`; that is outside the expensive word-reverse invariance claim.

### Gate F: route economics

Sweep realistic `R/J/F` charts. Confirm:

- low-run charts choose exact replay;
- high-run charts cross the predicted break-even;
- dense fallback frequency is reported;
- the route accounts for bytes and synchronization, not only interaction
  counts; and
- `J` and `E` stay invariant when only the output frame grid is densified.

## 12. Stop Rules

Narrow or stop the general memory-light training claim if any of these holds:

1. required trajectory basis size `L` grows proportionally with training
   frame count rather than physical motion complexity;
2. event count `E` or certified rank `J` grows when only requested sampling
   density changes;
3. most geometry optimizer steps invalidate most tracks despite a practical
   trust region and local repair;
4. real scenes need enough fixed-normal 4D refinement to erase the site-state
   advantage;
5. sparse derivative-action certification cannot avoid dense global dual
   state;
6. compiled gradient error exceeds tolerance even when primal RGB passes;
7. the native allocator retains multiple `B_p x K` blocks or full-video
   targets; or
8. realistic `R/J` never yields a compiled break-even and exact replay is not
   fast enough.

## 13. Final Narrow Claim

The defensible theorem target is:

> For a dynamic 3D power diagram whose site trajectories and weights have
> bounded temporal degree, a continuous camera program over a fixed physical
> interval, finite P0 emission-absorption, a certified finite event
> stratification, a continuously certified primal and referenced-material-
> action transfer rank on each stable chart, and a fixed compiled node/schedule
> program whose requested rational samples are exactly dispatchable (or avoid
> every unresolved algebraic isolator neighborhood), WorldFoam can evaluate
> streamed requested samples and accumulate a
> frozen-program world VJP with no frame-by-run tape. The stable-stratum
> geometry VJP differentiates that fixed compiled objective. A sufficient
> track-local lowered cut-jet certificate for continuous geometry/ray Jacobian
> error has been derived, but its outward-rounded certifier and runtime parity
> gate are not yet implemented.
> Structural and world-word work scales with events, active runs, and compiler
> rank, while requested-frame dependence is confined to rank-weighted
> bounded-block temporal basis evaluation, target I/O, residual reduction, and
> output writes.

The corresponding non-claim is equally important:

> Neither total work nor all metadata is automatically sublinear in frame
> count; event complexity, ray depth, approximation rank, and topology refresh
> are not universally bounded; and the current fixed shared-metric 4D site
> family is not a general moving-cell representation.
