# WorldFoam memory-light native-4D completion

## Evidence truth: tests are not ablations

The source/unit gates in this document are preflight verification only.  They
must never be reported as paper ablations or as evidence that WorldFoam fits in
memory.

- **G6 native memory/work ablation:** `0/21` measured evidence rows.  The dry
  plan contains 12 shared-adjoint primary rows and 9 same-representation
  sequential-replay controls at `F=8/64/300`, with three additional restart
  processes.  It emits no evidence until every row runs in a fresh real MPS
  process and the independent verifier accepts the resulting artifact.
- **G4 public held-out quality ablation:** `0/36` measured training rows.  The
  frozen G4-v2 matrix is three public scenes x three seeds x four matched
  routes.  It uses 300 optimizer steps, four spacetime samples per step, 1024
  identical selected pixels per sample, and one common RGB-MSE contract:
  `1,228,800` target pixels / `3,686,400` RGB loss scalars per row.  The final
  evaluation remains the complete 300-frame `384x512` heldout camera.  Its
  worker, source-read ownership, process-group watchdog, spatial-major heldout
  evaluator, collector, independent verifier, and paper-asset generator are
  source-implemented.  None is a quality or memory result until real rows run
  and verify.
- **G4-v1 reference remains compute-intractable:** the exact all-pixel executor
  cold-compiles one continuous program per unique `(view,pixel)` after every
  full-geometry update.  The frozen v1 scheduler therefore asks
  for `115,015,680`, `112,852,992`, and `113,442,816` cold track compiles for
  seeds `17`, `29`, and `43`, respectively (`117,964,800` is the exact
  two-distinct-view-per-step upper bound).  The framewise control also asks for
  exactly `1,843,200` native step calls per seed.  This is bounded in peak
  residency but not a tractable publication run.  It remains an unchanged
  correctness reference and is not silently relabelled as v2.
- **G4-v2 tractable replacement is implemented, not measured:** all four routes
  consume the identical selected target schedule.  WorldFoam rasterizes those
  `1,228,800` selected pixels; the two Gaussian controls still full-rasterize
  exactly `235,929,600` training pixels.  This is equal loss-pixel budget, not
  equal raster work, and both counts are mandatory paper columns.  WorldFoam
  heldout evaluation is spatial-major: `196,608` cold track compiles and
  `58,982,400` camera-record validations per heldout camera, issued as `1,536`
  128-track host calls / `15,360` bounded native bundles.  Predictions and
  RGB8 targets use two non-mmap, no-cache-when-available raw spools totalling
  `843.75 MiB` temporary disk, not resident model memory; both are deleted and
  content/digest checked.  Run the bounded one-step pilot before any full row.
- **G4 fairness/accounting repair:** both WorldFoam routes now apply the frozen
  stage learning-rate multiplier to material and geometry updates.  Training
  memory/time covers optimizer step 0 through checkpoint; the shared row
  sampler separately covers setup and the complete heldout evaluation.
  Heldout output allocation is independently capped by the sealed chunk plan,
  and an unknown completion fence quarantines roots instead of releasing them.
- **G4 next executable gate:** produce and verify the source-bound bounded
  Coffee Martini seed-17 pilot for both WorldFoam routes.  It must execute one
  real selected optimizer step, prove old-vs-spatial-major bitwise parity on a
  bounded track set, record compiler/work/memory/timing receipts, and remain
  explicitly `pilot_only=true`, `public_quality_evidence=false`.  Only then run
  the 36 publication rows in fresh processes.
- **Current G6 implementation blocker:** native source registers exactly
  `133/133` schemas and implementations, but the installed extension exposes
  only `103/133`.  Rebuild and attest the
  `world_foam_lane2_fused_slab_v0` extension on a quiet host before executing
  G6.
- **Canonical G6 handoff:** run the allocation-free plan, then the real
  guarded ablation on a quiet Mac:

  ```bash
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
    research_experiments/world_foam_lane2/run_worldfoam_g6_clean_host_bundle.py
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
    research_experiments/world_foam_lane2/run_worldfoam_g6_clean_host_bundle.py \
    --execute
  ```

  The first command self-checks that Torch remains absent and emits `0` rows.
  The second host-guards, force-rebuilds and attests the 133-schema Metal
  extension, checks both G6 ABI seals, runs all 21 evidence rows plus three
  restart processes, and independently verifies the artifact.  See
  `../research_experiments/world_foam_lane2/WORLDFOAM_G6_CLEAN_HOST_RUNBOOK.md`.
- **Current memory claim:** `native_memory_fit=false`.  The bounded logical
  state is small (`114,688` bytes live, `81,920` bytes checkpoint, and
  `278,528` bytes for the conservative live-plus-checkpoint-clone bound at
  `S=1024`), but allocator, native scratch, compiler, and Python working sets
  remain unmeasured.  Promotion requires `<2 GiB` MPS working set and `<4 GiB`
  sampled process-group RSS for the frozen matrix.

The 8-GiB free-host-RAM launch guard is incident headroom, not a claim that the
representation needs 8, 32, or any other large number of GiB.  At the latest
2026-08-15 audit this Mac had only about 4.1 GiB disk free and 18.3/19.0 GiB of
swap in use, so neither a native rebuild nor MPS row is safe here yet.  Before
the publication run, freeze the exact G6 source set in Git: the artifact binds
repository-relative source bytes, but most of the new implementation is still
untracked and a hash-only local checkout is not a durable release revision.

## 2026-08-15 training-memory ablation closeout

The paper-scale training-memory **implementation** is now routed end to end.
This supersedes older status text below that says the full-geometry adapter,
live restore, fair control, or evidence writer is absent.  The current paths
are:

- staged sparse and fused union-v2 shared-adjoint rows;
- a same-representation compile-once/framewise-replay control;
- full material, position, velocity, and weight CPU-SGD updates;
- sparse selected-track checkpoint/live restore and repeat-matched restart;
- fresh-process orchestration, MPS/RSS watchdogs, exact source/native/hardware
  bindings, and staged/fused plus fused/control `F=8` parity payloads.

The frozen matrix is 12 primary rows, 9 compiled-framewise controls, and 3
restart processes: 21 evidence rows across 24 sequential fresh processes.
It uses `S=1024`, `P=512`, `384x512`, and `F=8/64/300`.  The expensive
world-side tensor state is frame-bounded; samples, camera/ray slicing, and
small receipt telemetry remain deliberately linear and are reported rather
than hidden.

All primary scaling rows now have the same one-step scope.  In particular,
fused `F=8` no longer includes checkpoint creation or an uninterrupted second
step.  Each of the three auxiliary lifecycle processes performs fresh step 1,
checkpoint, uninterrupted step 2, full world release, restore, and restored
step 2, then binds its step-1 content back to the independent primary row.
The compiled-framewise logical peak also now conservatively sums overlapping
coordinator and geometry-bridge state plus the per-frame material/geometry bars
and loss scalar; the earlier incomplete `max(...)` bound is gone.

This does **not** yet prove WorldFoam fits the memory target.  No real MPS row
has run.  The logical combined live state is `114,688` bytes, the combined
checkpoint is `81,920` bytes, and the live-plus-checkpoint payload-clone bound
is `278,528` bytes at `S=1024`, but compiler objects, Python storage, native
scratch, and allocator behavior require fresh-process measurements.  Promotion
requires the full artifact to stay below the 2-GiB MPS working-set ceiling and
4-GiB sampled process-group RSS watchdog, with the frozen cross-`F` growth
limits and all numerical/lifecycle parity gates green.

Current execution blocker: rebuild and attest the Metal extension, then run on
a quiet host.  The local host at closeout had about 11 GiB disk free and
16.9/17 GiB swap in use, so it deliberately fails the incident guard.  The
guard asks for 8 GiB available RAM and <=2 GiB swap use; this is safety
headroom, not a 32-GiB representation requirement.  Do not weaken it.

Runbook and exact evidence boundary:
[2026-08-15 source closeout](../agent_notes/loose_notes/2026-08-15_04-37-03_worldfoam_training_memory_ablation_source_closeout.md).

Status: previously run CPU/source gates cover the exact and compiled math,
compact template-free schedules, exact pixel-aware `B_p x K` staging, a
backpressured native block adapter, caller-owned cross-`B_p` site bars, sparse
track-local certification, right-continuous piecewise-topology streaming, and
an owner-only P0 material optimizer with physical softplus/sigmoid parameters.
Earlier deterministic route-cost and host-memory artifacts were also green.
The latest 2026-08-04 mapped-source, binding, full-geometry, native-attestation,
producer, verifier, and test edits remain source-only and unrun. The loss-only
source ABI no longer allocates a discarded
`B_p x K x 3` prediction tensor. Material-only reverse skips Mobius, boundary,
and compact-geometry gradient buffers; target-only staging removes the
temporary `B_p x K x 6` ray block after one bounded exact camera-row check;
and a policy-bounded LRU caps cached native-topology entries and bytes while
preflighting the one live token; an evicted block may be prepared again.
Material training retains lightweight topology plus compact schedules rather
than a compiled CPU atlas per spatial block. The extension
has not been rebuilt or run. A newer source-only dense request/step candidate
now owns world-bound material/loss/geometry bars, consumes executor-sealed full
word VJPs, immediately reduces bounded `[J,W]` length bars, and authorizes the
optimizer only after exact full-manifest replay. Its sample executor now
source-writes a single-outstanding launch-lifetime/settlement contract: the
lifetime is installed before native prepare, roots the prepared payload and
all native sample arguments, while the dense caller leases transfer and
materialization predecessors; both are released only after the sole successful
completion fence rather than retained in launch history. A successfully sealed
session requires exact
`native_sample_prepare_count = native_sample_launch_count = native_sample_completion_fence_count`,
reports a maximum of one outstanding lifetime, and retains zero lifetime
history. The dense caller also leases the CPU transfer source and sample
materialization predecessors through settlement, and its restart-required
quarantine now includes current target, sample, material/reverse, reduction,
and completion roots. Active-block scatter and request-delta commit remain
separately fenced so Python reference release is not mistaken for
command-buffer backpressure. Full frames remain on CPU and only bounded
selected-pixel chunks cross to the device. The executor releases `[J,W]` only
after a sealed fenced-reduction receipt and explicitly does not claim the
higher-layer global commit. Poisoning retains asynchronous state until an
explicit abort fence succeeds. The lower-level target seam now accepts only a
sealed exact loader capability: it installs a one-chunk lifetime before target
work, publishes the CPU source before transfer enqueue, retains every returned
device tensor, and carries post-enqueue failures into request quarantine. The
lazy route now also owns one bounded restart-required no-retry quarantine for
its visible lane-construction, sparse-transfer, sample, reverse, and
lane-release roots. Two additional source-only lifetimes now install before
union-local map transfer and compact-material gathering, publish every raw
transfer/gather result before a later contiguous operation, and retire their
duplicate predecessors only after a proven lane-release or abort fence. The
spent bundle/lane is non-reusable and no reverse strong-reference cycle is
left on success. It still rejects non-CPU devices because the top-level device
allocation/`zero_` transaction has no preinstalled owner, cold union-map
validation chains a device-to-host receipt conversion, native-forward
enqueue-failure ownership is not certified, and the completion fence is not
yet a sealed backend/stream capability. The older standalone full-geometry path
is CPU/fake-native-only. These source and test edits have not been rerun, and
the native extension has not been rebuilt. The unified paper runner does
not yet expose `worldfoam_native4d`, and projective cameras, exact irrational
native chart endpoints, event-time derivatives, and geometry-update
recertification beyond the narrow event-free directional certificate remain
open.

The latest source-only full-geometry request defaults to fixed cameras and no
longer allocates either the request-local or whole-step float64
``[(view,pixel),12]`` ray cotangent. Camera calibration remains an explicit
opt-in mode. At ``2704x2028`` this removes about ``502 MiB`` of tensor payload
per selected view before counting Python key objects. This edit is unrun. A
separate material-only schema-v3 scaling verifier now requires real-native
fresh-process ``F=8/64/300`` rows over denser requested samples on one fixed
physical interval and fixed compiled world; it does not certify the unfinished
native full-geometry optimizer. Its MPS-only producer has a checked-in real
coordinator trial driver/config and coordinator-owned structural/work receipts,
but no row has run. A source red-team found that the previous trial incorrectly
set ``F_dataset=F_requested``, retained one per-frame camera signature in every
CPU artifact, and repeatedly rebuilt frame-scaled signatures during warm
validation. The repaired source fixes one 300-frame provider/camera grid and
varies only endpoint-including requested subsets ``8/64/300``; the verifier
reconstructs the subset and requires the dataset grid to remain identical.
Artifacts no longer retain the signature tuple, provider warm generation
checks no longer serialize all frame times, and static-camera equality is
certified once per view rather than scanned per track. Only the cross-row
verifier may now claim structural invariance; the step does not hardcode it.
Schema v3 seals an AST-resolved transitive closure of local
Python imports together with the declared native sources. Its capability
manifest now claims only the MPS backend and the direct
``PowerFoamSelectedPixelRead/v1`` contract; measurements come from bound runtime
receipts, not static capability claims. The producer is written to apply an MPS
per-process working-set limit whose effective bound is ``<=2 GiB`` and bind its
raw receipt and digest. Separately, the parent polls process-group RSS at a
configured 0.25-second interval and terminates on a sampled value above 4 GiB;
that is a sampled watchdog, not an exact allocator-hard RSS cap. The public MPS
counter sampler has a producer-bound configured interval of exactly 5.0 ms, and
its maxima remain labelled lower bounds. Each raw watchdog receipt is bound to
the child execution evidence and must report a clean exit, an empty process
group, and no watchdog termination. Resource attestation covers exactly node
forward, loss-only sample accumulation, and material-only word VJP, and the
query itself does not prove execution. Metal private/register/spill bytes remain
unobservable and are not inferred. The launch guard requires 8 GiB of host
availability as incident headroom, not because the representation needs 32 GB.
All producer/driver/verifier/native/test edits remain unrun and the extension
remains unbuilt. CUDA needs a separately bound native port and producer.

A new source-only fixed-camera full-geometry step coordinator now reaches the
existing full native VJP/reduction path, requires exactly one full VJP and one
fenced reduction per active block, rejects every ray bar, and returns sealed
material/geometry/loss authorization. Its geometry generation is now derived
from and checked against the live provider world rather than accepted as a
caller label. A combined CPU-SGD transaction now
builds out-of-place candidates, retires the old generation, streams exact cold
recompilation of the complete explicit next-step manifest through a bounded
LRU, permits only fresh cold compilation for later misses, and promotes only
the fresh sealed world; checkpoint cloning is a separate bounded operation. A
separate suffixed fixed-camera fused-v1 source entry point now combines the
material word VJP with owner-local kinetic scatter and removes the
`[J,W_b]` physical-length **cotangent** output/copy. It still consumes the
frame-independent compiled primal `[J,W_b]` lengths. The fixed-camera
coordinator now keeps staged sparse as its default but can explicitly select
`fused_direct_v1`; the selected mode is digest-bound, checked in request and
step accounting, retained in the combined-update receipt, and handed to the
same out-of-place CPU-SGD/cold-recompile lifecycle. The previous per-thread partial-write
hazard and actual-run nonfinite destination hazard are now closed at
source-contract level by one split all-block transaction. A cold token owns
fresh exactly-zero scratch and one int32 reason mask; the adapter enqueues
validate-all, guarded-accumulate-all, and finalize-all on one serialized stream, scans
each compact ledger and the shared global ledgers both before and after
accumulation, and accepts only after one completion fence/status read. A final
NaN or either infinity rejects and quarantines scratch before optimizer
authorization; it is not byte rollback or a prospective bound on every atomic
order. The token is one-shot and post-enqueue failure is fence-settled or
retained in restart-required quarantine. Exact active-manifest coverage is now
checked against the executor session before the fused receipt is accepted.
Completion-callback reentry is rejected, and pre/post callback snapshots bind
the sampled manifest, bars, status, and transaction state so callback mutation
cannot silently authorize scratch. Focused adversarial CPU/fake-native tests
for these contracts have been written but not run. This remains source-only,
unbuilt evidence. Staged-versus-fused numeric parity, two-step loss/checkpoint
behavior, live checkpoint restore, rebuilt-native ordering/IEEE parity, allocator
evidence, trainer integration, and production routing remain open.

Removing `[J,W_b]` does not yet prove that fused v1 has the lower whole-request
peak. The current transaction can overlap every prepared active block, all
compact outputs, one global float32 geometry output, and its global CPU-float64
bridge with the whole-step accumulator. That state is frame-independent, but
it can exceed the one-block-at-a-time staged peak. Measure both modes before
promotion. If global geometry scratch dominates, the next candidate is a
direct request-union geometry output and one post-fence global scatter; if
prepared blocks dominate, investigate a block-streamed fail-atomic transaction
instead. The staged reducer's complete fixed-camera per-block preflight is

```text
H_b
  = 4 J_b W_b
    + (56+8C_w) s_b
    + 16 J_b rho_b
    + 8 J_b
    + 8(37+2C_w) rho_b
    + 608
    + V_b,

V_b = 1 + max(J_b rho_b, 3s_b, C_w s_b).
```

`V_b` is the explicit one-byte finite-validation mask plus scalar; trainable
rays add `96(T_b+1)` to `H_b` and candidate `12T_b` inside `V_b`.

The dense request previously added only `max_b 4J_bW_b` to its active-state
cap while checking `H_b` separately. That failed to compose simultaneously
live tensors. A source-only correction now reuses the reducer's public
tensor-allocation-free preflight, enforces the existing bridge cap before lane work,
and adds `max_b H_b` to the active request/step state without double-counting
`[J,W]`. Its admission is conservatively
`X0 + 16 max_b s_b + max_b H_b`; the tighter block-correlated expression is
`X0 + max_b(16s_b+H_b)`. No allocator peak is claimed.

For fused v1, the geometry source plus CPU-float64 bridge is
`12S(6+C_w)`. A v2 transaction should instead write directly to a sealed
request-union `[U,6+C_w]` ledger through a separate
`compact_to_union_i64` map, preserve `source_site_ids_i64` as global
provenance, bridge only the union, and global-index-add once at request commit.
The exact geometry source/bridge saving is

```text
12(S-U)(6+C_w) bytes = 108(S-U) bytes when C_w=3.
```

Existing MPS union ids/maps are already lane-accounted and v2 must alias them.
If request commit needs a new CPU `int64` union-to-global map, charge `8U`, so
the net counted improvement is `12(S-U)(6+C_w)-8U`. Measure `U/S` before
promotion; do not claim the complete request is always smaller.

The suffixed raw union-v2 Python/Torch/Objective-C++/Metal split ABI and a
sealed all-block adapter are now source-written and statically inspected. The
adapter binds the exact union bundle and canonical block manifest, requires
resident map aliases and `P_b=P_U Q_b`, allocates compact material plus
`[U,6+C_w]` geometry scratch, performs `B/B/B`
validate/accumulate/finalize launches with one sticky status and one fence, and
returns single-use union-local bars without a global or optimizer write. It
explicitly leaves material in block-compact space and does not certify a union
material sum. Construction is now two phase: a caller-visible lifetime is
installed before the first raw/native return or output allocation, each return
is published immediately, partial construction fences once before release,
and accepted execution drops transaction, raw, output, prepared-block, bundle,
map, threshold, and signature roots before marking that lifetime released.
A known rejection retains exactly one fail-stop carrier and forbids later
fused work; unknown completion requires restart. No Python test, native rebuild,
Metal compile, parity, failure injection, allocator, request bridge, executor,
or trainer route has run. The outer lifetime cannot publish the raw preparer's
individual internal config temporaries before that preparer returns its
aggregate token; failure is fenced, but native allocator/runtime evidence is
still required for that seam. Bounded `q=1` remains separate and needs either a
status-gated material scatter/finalizer or the documented two-fence fallback.

This makes direct union-local geometry output the first memory-v2 direction,
not per-block geometry outputs and not another temporal marginalization.
Block streaming comes later because a naïve blockwise persistent commit breaks
request-level fail atomicity.

The lazy provider also drops the point-cloud initializer after cold world
construction and retains only its tensor-free provenance/generation receipt.
For degree-2 weights this makes the intended fixed-site base exactly `120*S`
bytes at steady state and `136*S+4` during an active material-only step, before
compiled artifacts, decoding/interpolation, native scratch, allocator storage,
or optional optimizer moments. Callers must still release their own initializer
and physical seed after constructing the live material state. These lifecycle
and accounting edits are source-only and unrun.

The geometry scope has also narrowed. In a fixed world-coordinate gauge, a
fixed shared-SPD(4) power world slices exactly to a restricted kinetic 3D
family: after one common translation its sites are fixed, its relative weights
are affine in time, and every candidate face has a constant spatial normal. A
time-dependent global scene gauge can freeze one rotating face, so the two-site
rotating-face fixture is a fixed-gauge separation test; one common gauge cannot
generally freeze several independently rotating faces. The intended general
frontend therefore uses the camera/scene gauge for shared bulk motion and
direct affine kinetic residual sites `p_i(t)=p_i0+t v_i` with degree-`<=2`
weights. It proves quadratic-or-lower ray-cut coefficients,
quartic-or-lower adjacent concurrence, rotating residual faces, and parameter
bytes independent of requested frames. An exact rational square-free/Sturm
isolation primitive through quartics and a guarded finite-cut concurrence
wrapper are implemented at CPU scope. An exhaustive \(O(S^3)\) continuous CPU
reference compiler and an independent exact oracle now emit/check half-open
owner charts without requested-frame sampling. The active-boundary compiler
now constructs predicates in `O(U S R_max)` across `U` unique witnessed owner
words, with a separately reported
`O(W (S log S + S R_max))` cost for `W` cumulative root-complement discoveries
and all-site certificates. This is an honest output-sensitive improvement over
the all-triples oracle, not an unqualified `O(SR)` total-time theorem. Exact
multi-chart dispatch, fixed-rank ordered P0 transfer, continuous primal and
referenced-material action certification, and an end-to-end frozen-program
site/trajectory/weight/ray/material VJP now run on CPU with `O(sum J_c)` reverse
state and no frame tape. The source native replay also retains sample times only
for the live `K` block; it owns no global `[F]` or chart-local `[F_c]` clone.
A provenance-sealed CPU lowerer now packs real `(track,chart)` rows into
bounded actual-`J` native blocks with compact CSR owners and positive `[J,R]`
node lengths. Its independent Lie oracle, fake-native CPU lifecycle adapter,
node-length-to-geometry VJP, row-ragged paper sampler, union-local
heterogeneous-block bar assembly, and outer multi-view/global-denominator
coordinator are green under CPU/source tests. A block-major CPU/fake-native
paper step now holds each spatial bundle across all temporal chunks, runs each
node forward and material-only word VJP exactly once, allocates no `[J,W]`
geometry bar, and releases bundles sequentially. Its loss/material bar matches
a direct-autograd oracle across `K=1/4`; `F=5/41` changes only streamed sample
work, not compiled-word work or retained runtime bytes. An exact directional
certificate proves reuse for one strict event-free single chart. A separate
exact CPU reference proves whole-direction persistence, endpoint re-isolation,
and semantic left/right owner-word agreement for separated singleton simple
roots after rebuilding the complete rooted/rootless predicate registry. The
simple-root routine is a certificate/oracle, not a program patcher. The current
production rule is therefore strict: material-only updates may reuse a sealed
compiled program, while every geometry or camera-ray update receives a fresh
structural compile and recertification. A bounded CPU artifact store and a
replayable dense-observation source now meet in a source-only block-major
request/step candidate. Its upstream source still retains `O(F)` scalar frame
metadata, and its latest transaction and full-geometry edits remain
runtime-unverified. A bounded frame-independent point-cloud initializer, an
exact static-camera active-track factory, a fixed-site raw-material/manual-SGD
lifecycle, and a raw-only restart checkpoint now exist in production source.
The initializer starts with zero velocity, and the factory rejects moving,
projective, and gauged camera paths. The extension remains unbuilt and
runtime-unverified. Production-scale image-wide compilation/serialization,
warm/output-sensitive affected-chart repair, total derivatives through
structural choices, a fenced accelerator updater/complete train loop, and
streamed evaluation remain open. A caller-owned exact-coverage material-only
authorization coordinator now exists in source and remains unrun.

Primary handoff:

```text
agent_notes/loose_notes/2026-08-03_03-35-19_worldfoam_memory_light_shared_adjoint.md
agent_notes/loose_notes/2026-08-03_16-35-33_kinetic_power_word_event_sufficiency_red_team.md
agent_notes/loose_notes/2026-08-03_17-58-19_worldfoam_production_kinetic_compiler_and_bounded_native_time_state.md
agent_notes/loose_notes/2026-08-03_19-09-46_worldfoam_node_length_native_seam_and_mathematician_handoff.md
agent_notes/loose_notes/2026-08-03_20-40-32_worldfoam_multichart_simple_root_reisolation.md
```

Reference implementation:

```text
research_experiments/world_foam_lane2/compiled_transfer_adjoint.py
research_experiments/world_foam_lane2/test_compiled_transfer_adjoint.py
research_experiments/world_foam_lane2/exact_sparse_incidence_oracle.py
research_experiments/world_foam_lane2/test_exact_sparse_incidence_oracle.py
research_experiments/world_foam_lane2/transfer_lie_chart.py
research_experiments/world_foam_lane2/test_transfer_lie_chart.py
research_experiments/world_foam_lane2/compiled_lie_world_adjoint.py
research_experiments/world_foam_lane2/test_compiled_lie_world_adjoint.py
research_experiments/world_foam_lane2/compact_lie_schedule.py
research_experiments/world_foam_lane2/native_track_adapter.py
research_experiments/world_foam_lane2/material_parameterization.py
research_experiments/world_foam_lane2/material_training_step.py
research_experiments/world_foam_lane2/piecewise_topology_staged_adjoint.py
research_experiments/world_foam_lane2/native_piecewise_topology_adapter.py
research_experiments/world_foam_lane2/host_memory_contract.py
research_experiments/world_foam_lane2/verify_worldfoam_memory_scaling_acceptance.py
research_experiments/world_foam_lane2/worldfoam_memory_scaling_acceptance_v3.json
research_experiments/world_foam_lane2/run_worldfoam_memory_scaling_acceptance.py
research_experiments/world_foam_lane2/test_verify_worldfoam_memory_scaling_acceptance.py
research_experiments/world_foam_lane2/compiled_route_cost_gate.py
research_experiments/world_foam_lane2/kinetic_power_word_compiler.py
research_experiments/world_foam_lane2/test_kinetic_power_word_compiler.py
research_experiments/world_foam_lane2/rational_polynomial_roots.py
research_experiments/world_foam_lane2/test_rational_polynomial_roots.py
research_experiments/world_foam_lane2/kinetic_owner_chart_compiler.py
research_experiments/world_foam_lane2/test_kinetic_owner_chart_compiler.py
research_experiments/world_foam_lane2/kinetic_owner_chart_oracle.py
research_experiments/world_foam_lane2/test_kinetic_owner_chart_oracle.py
research_experiments/world_foam_lane2/kinetic_chart_transfer_bridge.py
research_experiments/world_foam_lane2/test_kinetic_chart_transfer_bridge.py
research_experiments/world_foam_lane2/kinetic_active_owner_chart_compiler.py
research_experiments/world_foam_lane2/test_kinetic_active_owner_chart_compiler.py
research_experiments/world_foam_lane2/kinetic_multichart_transfer_program.py
research_experiments/world_foam_lane2/test_kinetic_multichart_transfer_program.py
research_experiments/world_foam_lane2/kinetic_continuous_transfer_acceptance.py
research_experiments/world_foam_lane2/test_kinetic_continuous_transfer_acceptance.py
research_experiments/world_foam_lane2/kinetic_stable_stratum_vjp.py
research_experiments/world_foam_lane2/test_kinetic_stable_stratum_vjp.py
research_experiments/world_foam_lane2/kinetic_multichart_stable_stratum_vjp.py
research_experiments/world_foam_lane2/test_kinetic_multichart_stable_stratum_vjp.py
research_experiments/world_foam_lane2/kinetic_native_topology_lowering.py
research_experiments/world_foam_lane2/test_kinetic_native_topology_lowering.py
research_experiments/world_foam_lane2/kinetic_native_precompiled_length_oracle.py
research_experiments/world_foam_lane2/test_kinetic_native_precompiled_length_oracle.py
research_experiments/world_foam_lane2/kinetic_native_precompiled_length_adapter.py
research_experiments/world_foam_lane2/test_kinetic_native_precompiled_length_adapter.py
research_experiments/world_foam_lane2/kinetic_native_equal_rank_lowering.py
research_experiments/world_foam_lane2/test_kinetic_native_equal_rank_lowering.py
research_experiments/world_foam_lane2/kinetic_native_equal_rank_runtime_adapter.py
research_experiments/world_foam_lane2/test_kinetic_native_equal_rank_runtime_adapter.py
research_experiments/world_foam_lane2/kinetic_geometry_trust_region.py
research_experiments/world_foam_lane2/test_kinetic_geometry_trust_region.py
research_experiments/world_foam_lane2/kinetic_simple_root_reisolation.py
research_experiments/world_foam_lane2/test_kinetic_simple_root_reisolation.py
research_experiments/world_foam_lane2/kinetic_ragged_paper_step_cpu_fake_native.py
research_experiments/world_foam_lane2/test_kinetic_ragged_paper_step_cpu_fake_native.py
research_experiments/world_foam_lane2/test_kinetic_ragged_lie_sample_source_contract.py
src/train/paper_ragged_track_staging.py
tests/test_paper_ragged_track_staging.py
src/train/paper_kinetic_ragged_sample_plan.py
tests/test_paper_kinetic_ragged_sample_plan.py
src/train/paper_kinetic_union_local_bar_assembly.py
tests/test_paper_kinetic_union_local_bar_assembly.py
src/train/paper_ragged_material_bar_coordinator.py
tests/test_paper_ragged_material_bar_coordinator.py
src/train/paper_kinetic_world_initializer.py
tests/test_paper_kinetic_world_initializer.py
src/train/paper_kinetic_active_track_program_factory.py
tests/test_paper_kinetic_active_track_program_factory.py
src/train/paper_kinetic_fixed_site_material_state.py
tests/test_paper_kinetic_fixed_site_material_state.py
src/train/paper_kinetic_fixed_site_material_step.py
tests/test_paper_kinetic_fixed_site_material_step.py
src/train/paper_kinetic_replayable_observations.py
tests/test_paper_kinetic_replayable_observations.py
research_experiments/world_foam_lane2/kinetic_compiled_cpu_artifact_store.py
research_experiments/world_foam_lane2/kinetic_dense_cached_native_material_request.py
research_notes/worldfoam_paper/WORLD_FOAM_DYNAMIC_DEPTH_ORDER_MATHEMATICIAN_PROMPT.md
```

## Target contract

```text
persistent world parameters       no frame axis
camera/ray structural program     event-scaled, not sample-density-scaled
reverse interaction memory        independent of F_requested for one fixed compiled program
per-step atlas/scratch             bounded in spatial blocks B_p and time K
targets/residuals                  exact bounded B_p x K target-only material adapter green
camera rays                        bounded block affine rows + one exact validation row;
                                   fixed-camera global cotangent is zero
sample evaluation/reduction        O(sum F_pc J_pc + sum N_fb,pc J_pc^2 + PF)
world/intersection VJP             shared across time
```

Mathematical decision:

```text
proof/tangent object               translated optical-depth measure (kappa,nu)
runtime ordered-transfer state     affine quotient (beta,m) + compact owner word
heavy fixed-surrogate work         Theta(sum_(p,c) J_(p,c) r_(p,c))
ragged sample work                 Theta(sum F_(p,c) J_(p,c)
                                         + sum N_fb,p,c J_(p,c)^2
                                         + PF)
```

Do not search for a foam analogue of Gaussian Schur marginalization. The
translated measure already gives the exact order-explicit monoid and tangent
formula, while its Laplace image gives the existing four-scalar runtime. Depth
elimination would erase the order changes WorldFoam is intended to model.
This translated optical-depth measure is the scientist review's one strong
formulation newly derived in this project and is already integrated as a
proof/tangent object; the canonical runtime remains the same compact `(beta,m)`
owner-word program. External literature novelty remains open.
`O(PF)` is only shorthand for varying `F` with chart ranks and fallback
behavior fixed; keep `J` and exceptional-row work visible in theorem and
benchmark tables.

Do not use the current per-frame `MetalPowerFoamVideo` paper row as evidence
for this contract. Keep it as the per-frame PowerFoam baseline until the native
path replaces it.

## Closed in the CPU reference

- [x] Exact ordered-transfer composition.
- [x] Translated optical-depth-measure proof object, associative shifted-measure
  concatenation, Laplace homomorphism to `(beta,m)`, distributional
  boundary-mass tangent, and proof-only CPU parity oracle. Runtime remains the
  affine quotient; no measure payload is added to native execution.
- [x] State the paper-grade translated-measure theorem with explicit finite-P0
  assumptions, identity/associativity proof, Laplace homomorphism, zero-width
  seam corollary, weighted-total-variation transfer bound, and a tangent-aware
  opacity-tail criterion. The tail optimization remains disabled.
- [x] Source-write proof-oracle regressions for the weighted-total-variation
  and tangent-aware opacity-tail inequalities. They compare the stated bounds
  with exact P0 transfer/tangent differences and remain unrun on this host.
- [ ] Run those focused proof-oracle regressions before using either inequality
  as an adaptive compiler/runtime certificate. A primal opacity threshold alone
  is not an optimizer-safe truncation rule.
- [x] Constant-state two-pass word VJP; no per-run suffix/reverse arrays.
- [x] Sparse active track-boundary Mobius coefficient adjoint.
- [x] Once-per-incidence boundary/ray VJP.
- [x] Boundary lowering/VJP for caller-supplied sparse active pairs.
- [x] Once-per-face boundary-to-4D-site/weight VJP.
- [x] Float64 analytic denominator/order check for a supplied word.
- [x] Physical ordinary-depth fiber Jacobian and affine gauge-rescaling gate.
- [x] Exact chunked-frame loss/VJP with chunk invariance.
- [x] Track-blocked per-step coefficient/adjoint scratch.
- [x] Optional camera-ray gradients; fixed-camera default avoids the full buffer.
- [x] Optional Chebyshev total-transfer coefficients and shared coefficient VJP.
- [x] Affine-transfer Lie chart
  `kappa=-log(beta), v=kappa*m/(1-beta)` with stable encode/decode and analytic
  VJPs, including `kappa=0` and high-opacity cases.
- [x] Physical Lie-cone gate `kappa>=0, 0<=v_c<=kappa` and a negative control
  proving that neither raw nor Lie coordinates are a universal fixed-rank
  winner.
- [x] End-to-end `J`-node Lie compilation, streamed sample-to-node cotangent
  reduction, one-scan prefix-only word VJP from retained node totals, and
  sparse boundary/site gradients.
- [x] Separate sampled forward and tangent/VJP rank gates. Validation is off by
  default and counted explicitly when run at compile/refresh time.
- [x] Stable tiny-optical-depth forward/VJP using `-expm1(-tau)` and a manual
  reverse boundary that retains no autograd graph.
- [x] Selected logical tensor-payload accounting (not measured peak memory).
- [x] Staged `K`-block sample-to-node reduction with one global loss
  denominator and one world/boundary finalize.
- [x] Compact CSR track blocks whose active faces derive from the same compact
  4D sites, with boundary-to-site VJP and global site-gradient scatter.
- [x] Tensor-version and prepared-token provenance guards across the CPU
  refresh, accumulate, finalize, and scatter lifecycle.
- [x] CPU autograd, forward, gradient, and fail-closed behavior tests.
- [x] Template-free compact schedules with `O(sum_c J_c^2)` global metadata;
  no full-`P` atlas is required by the source training route.
- [x] Strict track-local sparse Lie certification with a hard local-dual cap;
  the dense certificate is retained only as a tiny-fixture oracle.
- [x] Source-only owner/topology material binding that permits live P0 material
  refresh while keeping sites, rays, topology, and schedules immutable.
- [x] Source-only piecewise-topology streaming with exact polynomial guards for
  binary sample times, right-continuous seams, and one global gradient ledger.
- [x] Explicit host-memory formulas and an exact-versus-compiled route gate.
- [x] Fit-derived second-form barycentric sample weights for the actual rounded
  nodes: `O(KJ)` common-path work, exact-node one-hot rows, and explicit
  row-local dense fallback/fail-closed accounting.
- [x] Exact fixed shared-SPD(4) slice characterization in a fixed coordinate
  gauge: one common translation of fixed anisotropic 3D sites, affine relative
  weights, and constant spatial face normals. A shared scene gauge remains the
  bulk-motion factor; it cannot generally freeze independently rotating faces.
- [x] Exact CPU direct affine kinetic 3D frontend with
  `p_i(t)=p_i0+t v_i`, degree-`<=2` weights, degree-`<=2` ray-cut coefficients,
  degree-`<=4` adjacent concurrence, a fixed-gauge rotating-face separation
  fixture, exact fixed-time words, and no frame-indexed parameters.
- [x] Exact rational square-free/Sturm root isolation through quartics and a
  guarded finite-cut concurrence predicate at CPU scope.
- [x] Repair Sturm-chain sign preservation by allowing only positive
  normalization, with an independent oracle and rootless `x^2+1` regression.
- [x] Exhaustive continuous CPU owner-chart compiler: all pair near/far and
  finite triple candidates, analytic-only denominator guards, exact algebraic
  root grouping, all-site witnesses, right-continuous charts, and explicit
  fail-closed degeneracies. This is the small-world `O(S^3)` reference, not the
  production sweep.
- [x] Independent global-product/Sturm chart oracle with all-pair depth-cut
  words and adversarial inactive, simultaneous, close-root, full-fiber, and
  ray-collapse coverage.
- [x] CPU one-chart kinetic P0 transfer bridge with exact fixed-`J` node words,
  compact Lie sample reduction, prefix-only material VJP, and structural/
  reverse state invariant to requested sample count. Geometry/event VJPs and
  native seam dispatch remain outside its claim.
- [x] Active-boundary exact compiler with cached per-owner-word predicate
  construction, exact algebraic clustering, all-site cell witnesses,
  differential parity against the exhaustive compiler/oracle, and honest
  `U`-word versus `W`-witness work accounting. Inactive and endpoint
  full-fiber ties fail closed because open-cell word equality does not make
  their material transfer unambiguous.
- [x] Provenance-bound multi-chart CPU dispatch with right-continuous binary
  search, bounded `K`-sample reduction to `O(sum J_c)` node cotangents, and no
  dense sample-by-chart state.
- [x] Continuous, outward-rounded rank acceptance for the actual second-form
  barycentric evaluator: primal transfer plus complete referenced-material
  Jacobian bounds imply declared-norm material JVP/VJP bounds without using
  requested sample times. Float roundoff, dense fallback, irrational seam
  neighborhoods, and geometry/event Jacobians remain excluded.
- [x] End-to-end frozen-program node VJP for affine kinetic positions,
  velocities, quadratic weights, affine rays, density, and RGB. It includes
  implicit cut and physical fiber-speed terms, uses stable `expm1` opacity,
  and keeps world reverse work `O(sum_c J_c R_c)` with no frame tape. It is a
  stop-gradient compiler VJP: chart endpoints, node times, sample weights,
  dispatch, rank, and event times are fixed.
- [x] Loss-only native source accumulation that does not allocate or write a
  `B_p x K x 3` prediction tensor. Forward media/evaluation may request that
  optional output explicitly.
- [x] Material-only native reverse that retains RGBA bars but omits Mobius,
  boundary, and compact geometry-gradient tensors. Strict/evaluation bindings
  retain the complete geometry VJP.
- [x] Session-owned, fail-closed native topology-token cache bounded by the
  number of spatial blocks. Immutable CSR/source-id state is reused; live
  material/world values still refresh every block and step.
- [x] Target-only material staging. One bounded fixed-camera reference row is
  checked exactly against the immutable affine program, then each hot block
  carries only `[B_p,K,3]` targets. Strict/evaluation and piecewise paths keep
  their explicit-ray validation route.
- [x] Lightweight material-training ownership: each block retains compact
  topology, a compact spec schedule, and an owner binding, with zero retained
  compiled CPU atlases and zero per-step CPU atlas compiles.
- [x] Provenance-sealed single-ray kinetic lowering to native-shaped CSR owners
  and positive `[J,R]` physical node lengths, with identical retained bytes for
  small and arbitrarily large requested frame counts.
- [x] Independent CPU affine-Lie node forward/VJP oracle plus a source-only
  native adapter that accumulates compact/global P0 material bars and returns
  bounded `[J,R]` physical-length bars.
- [x] Frozen-stratum node-length geometry bridge for positions, velocities,
  quadratic weights, and affine rays; discrete compiler choices remain stopped.
- [x] Row-ragged native source sample reducer using `[R_b,J,4]` node charts,
  selected row ids, and `[N,J]` local weights, with no global common-refinement
  table and no mandatory prediction allocation.
- [x] Ragged paper-batch staging grouped by view with one global loss
  denominator, no view/time Cartesian padding, and one-frame-at-a-time target
  decode.
- [x] Bounded equal-rank batching over real `(track,chart)` rows, with no
  global temporal refinement or `J_max` padding.
- [x] Row-ragged paper-to-kinetic sample planning and a backend-independent
  outer coordinator with exact coverage and one optimizer authorization.
- [x] Union-local heterogeneous-native-block material-bar assembly using one
  caller-owned `[S_union,4]` request bar and no per-request global `[S,4]` bar.
- [x] Exact directional trust certificate for one strict event-free
  single-chart update.
- [x] Restricted exact multichart update certificate for separated singleton
  simple roots, including complete predicate-source reconstruction, root-free
  complements, endpoint re-isolation, semantic owner-word reclassification,
  and fail-closed unsupported strata.
- [ ] Replace full-registry/full-compile reference work with a certified
  output-sensitive affected-source repair only if measurements justify it.
  This is the open simple-root local-repair question, not a completed warm
  update path. Until it returns rebuilt charts, ranks, payloads, and dispatch
  under a complete root-birth/collision certificate, full structural
  recompilation after geometry or camera-ray updates is the safe production
  rule.

The direct kinetic CPU route now spans active-boundary chart compilation,
multi-chart ordered transfer, continuously certified primal/material rank, one
frozen-program geometry/material reverse, and a source-native node-length seam.
Requested samples participate only in bounded interpolation/residual reduction;
compiler-node cotangents and world gradients have no frame axis. The exhaustive
compiler and independent oracle remain differential references. The landed
lowerer batches real track-chart rows and the ragged coordination seam is green,
but dataset-bound compilation and the native runtime are absent, so this is not
yet an image-wide trainer. Persistent/simultaneous events remain
fail-closed, every full-fiber material tie now fails closed, and total
derivatives through changing chart endpoints/schedules remain open.

## Phase 1: exact P0 Metal geometry VJP

- [x] Add a suffixed source-only packed-framegroup op with the prefix-only
  constant-state second pass, physical fiber length, and no per-run reverse
  arrays. Existing ABI/routing remains untouched.
- [x] Extend that suffixed source ABI from `(loss, grad_site_rgba)` to also
  return direct `grad_boundary[B,5]`.
- [ ] Rebuild the extension and run the bounded MPS forward/site/boundary
  parity gate; the checked-in binary predates the new schemas. The real MPS
  adapter now fails cold unless one unambiguous `_C` library is newer than all
  compiled C++/Objective-C++/Metal sources and exactly registers the four
  kinetic forward, loss-only sample, full-VJP, and material-only-VJP schemas
  with `CompositeExplicitAutograd` implementations. CPU fake-native seams
  remain callable-only.
- [x] Add a source-only staged sparse path that lowers only active
  `(track,boundary)` incidences to Mobius `[A,B,C,D]` rows once, removes
  boundary/world reads from the framegroup replay, accumulates coefficient
  cotangents, and applies the plane VJP once per incidence. It is not built or
  runtime-verified yet.
- [ ] Coalesce adjacent shared endpoints and reduce coefficient cotangents
  within framegroups when measurement shows sample-scale coefficient atomics
  dominate. Sparse scalar atomics beat direct scatter only when endpoint reuse
  per incidence exceeds five.
- [x] Add source-only sparse boundary-to-site/weight scatter derived from the
  exact resident sites/pairs used by the forward refresh.
- [x] Add a separate fenced CPU/source reduction from one full bounded native
  equal-rank `[J,W_b]` physical-length bar to frozen-stratum site position,
  velocity, polynomial-weight, and affine-ray bars. This closes the numeric
  bridge, not native trainer integration or allocator proof. Its word reverse
  is frame-independent, but current CPU certification still pays all-site
  `O(J S R)` validation and per-row dense global-site accumulation; do not call
  it a complete sparse `O(sum J R)` geometry reverse.
- [x] Replace the warm per-row dense-site result with a source-only,
  certificate-bound owner-local VJP/scatter. The sparse bridge consumes one
  fenced native `[J,W_b]` length bar, copies only one row at a time, recomputes
  adjacent certified cuts, and emits compact position, velocity,
  polynomial-weight, and optional affine-ray bars. It performs zero all-site
  owner scans and creates no row-by-global-site temporary; one compact
  `index_add_` per row/request feeds the caller-owned global bar. The older
  all-site audited reducer remains the parity oracle. Geometry/ray updates
  still force full recompilation. Source/tests are unrun.
- [x] Add source-only direct-kinetic native geometry lowering behind a sealed
  fixed-camera adapter. It revalidates compiler-issued certificates against
  live source/program contents, aliases already-resident MPS topology where
  layout permits, rejects stale provenance before launch, and exposes no ray
  cotangent surface. It is unbuilt, unrun, unselected, and not production ABI
  evidence. Device numerical guards in the raw single-block entry remain
  defensive; the all-block transaction below supplies the source-level
  validate-all/write-all boundary. Promotion still requires rebuilt parity and
  the postwrite/fail-stop transaction gates below; the source no longer treats
  a prospective aggregate-destination bound as the only safe closure.
- [x] Add a suffixed source-only fixed-camera fused node-word/kinetic reverse:
  consume each material-produced `bar_ell` in the owning
  `(row,node)` thread, form adjacent cut bars from the previous/current pair,
  use the fixed ray-speed primal, and scatter owner-local kinetic bars directly.
  Its acceptance target is **zero** `[J,W_b]` physical-length cotangent
  tape/output bytes and zero device-to-host cotangent copies; the compiled
  primal length table remains resident and frame-independent. Keep the one-bar
  sparse CPU route as the default coordinator mode, parity oracle, and
  failure-localization bridge.
- [x] Reject fused-v1 numeric thresholds that underflow, overflow, or otherwise
  become invalid when converted from host binary64 values to the Metal
  binary32 config. This is only scalar admission; it does not certify dynamic
  row/node/material/cotangent arithmetic.
- [x] Make fused atomic output admission fail closed at source-contract level.
  A cold one-shot token allocates fresh exactly-zero, storage-distinct compact
  and global scratch plus one four-byte status. It rejects duplicate block
  generations, enqueues all validations before all status-gated accumulations
  and all postwrite finalizers, scans every compact ledger twice and the shared
  global ledgers twice, and accepts only a zero status after one completion
  fence. A final NaN or either infinity therefore cannot authorize an optimizer
  commit. Postwrite rejection does not roll back scratch; any post-enqueue
  exception is abort-fenced and quarantined, and a failed abort fence retains
  live roots and requires restart. Zero certifies the actual final ledgers are
  finite, not exact, deterministic, underflow-free, or prospectively safe for
  every atomic order.
- [x] Make fused v1 an explicit opt-in source mode in the fixed-camera
  coordinator, prove exact active-manifest coverage at the executor boundary,
  bind mode-specific counts and memory receipts, and preserve the selected
  mode through the out-of-place CPU updater/cold-recompile receipt. Staged
  sparse remains the default. Adversarial callback-mutation/reentry and
  coordinator fake-native tests exist but are unrun.
- [x] Source-write the suffixed union-v2 raw ABI and sealed all-block adapter.
  It preserves fused v1 as the oracle, binds the exact request-union spatial
  bundle and `P_b=P_U Q_b`, aliases resident identities, validates all blocks
  before any write, accumulates geometry directly into `[U,6+C_w]`, finalizes
  every compact material ledger and the shared union ledgers, and returns one
  single-use accepted result after one status/fence. It has no executor,
  persistent commit, optimizer, or bounded-`q` route; source tests are unrun.
- [x] Split union-v2 construction into caller-installed lifetime preparation
  and device materialization. Every raw return and scratch allocation is
  published immediately; partial construction fences before release or enters
  the bounded fail-stop quarantine, and accepted execution clears bulky
  construction roots after its proven fence. Exact scratch accounting includes
  the four-byte sticky status. The raw preparer's internal pre-return config
  temporaries remain opaque to the outer carrier and require native
  allocator/failure validation. This remains source-only and unrun.
- [ ] Rebuild and falsify union v2 before integration: require `U=S` parity
  against fused v1, `U<S` parity against staged sparse, duplicate-union-site
  accumulation, poisoned-map/status/finalizer/fence/callback cases, exact
  `U/S` and logical-byte accounting, and allocator/RSS evidence. Only then add
  the executor receipt, `[U,*]` CPU bridge, request delta, union-to-global
  commit, and explicit trainer mode.
- [ ] Build and attest the fused-v1 source, require active-compiler provenance,
  and match it against both staged sparse and dense float64 oracles before any
  promotion. Add a **separate** selected full-geometry ABI/attestation rather
  than broadening the sealed material-only v3 evidence: node forward,
  loss-only ragged sample accumulation, and the split fused
  validate/accumulate/finalize transaction are the exact five selected kernel
  resources. Bind the rebuilt library hash, schemas, observed call counts, and
  completion-fence provenance in an outer native execution receipt. Measure
  allocator peak for both complete request lifecycles,
  including all prepared blocks, compact/global output scratch, CPU-float64
  bridge, and whole-step bars. Native gates must poison final ledgers with
  NaN/`+Inf`/`-Inf` and finite-add overflow, exercise a skipped/failed finalizer
  and abort fence, and verify same-stream visibility. A future trainable-camera
  ABI must separately add ray/speed cotangents and their parity gate; do not
  infer them from fixed-camera v1.
- [ ] Match the CPU reference for forward, site RGBA, boundary, site geometry,
  site weight, and camera-ray coefficient gradients.
- [ ] Add ordinary/log-depth gauge parity and camera-fit residual gates. The
  current reference implements ordinary depth plus affine rescaling only.
- [ ] Preserve an exact/fail-closed fixed-topology scope; do not imply
  derivatives through owner-word changes.

Acceptance:

```text
forward max abs error       <= 1e-5
gradient normalized error   <= 1e-4
no frame-by-run allocation
no dense track-by-boundary allocation
physical optical depth invariant under declared gauge changes
```

## Phase 2: streamed trainer contract

The CPU staged adjoint and paper target-provider seam preserve one global loss
normalization across arbitrary `K` partitions and avoid retaining accelerator
targets/predictions in the accumulator. The source material trainer now uses a
loss-only native accumulation ABI, so its hot path never creates a discarded
prediction block. The unified production runner still does not use this route.
A source-only, unrun Neural3D/PyAV offline converter is designed to seek selected
frames rather than decode the whole clip onto host RAM. No populated mapped
cache or production trainer binding exists yet, and codec/PyAV/Pillow allocator
peaks remain unmeasured.

- [x] Source-route selected frame blocks `K`, reduce immediately through the
  loss-only ABI without allocating a prediction block, and enforce one
  in-flight block. Built-runtime proof is still open.
- [x] Tile material targets in exact bounded `B_p x K` blocks and consume them
  through the compiled native source lifecycle. Strict/evaluation replay can
  still stage bounded target/ray blocks.
- [x] Factor rectangular multi-view observations exactly from
  `pixel x (view*time)` into `(view,pixel) x time`, preserving the global RGB
  denominator and one affine ray program per native track.
- [x] Adapt arbitrary paper `SpacetimeBatch` observations into canonical
  view-local groups without a view/time Cartesian product. Preserve original
  batch positions and one global `P*B*3` denominator; assign disjoint logical
  sample ranges to the existing native scalar contract.
- [x] Remove retained `explicit_rays[P,F,6]` and temporary `[B_p,K,6]` rays
  from material training. A one-row `[B_p,1,6]` exact validation scratch
  remains once per spatial block.
- [x] Remove `P(F-1)` frame-selector materialization from this route.
- [x] Remove the source-only native state-init allocation of a discarded
  `F_c x J` interpolation matrix and the retained chart-local `F_c` time clone.
  State init owns no sample-time tensor; each live block constructs weights
  from one CPU float64 `[K]` time slice and releases it after synchronization.
- [x] Keep targets on host/streaming storage and gather only selected `B_p x K`
  target pixels before a material accelerator transfer. The procedural memory
  fixture now supplies those pixels directly, eliminating the source-audited
  `5.41 TiB` full-frame decode amplification that its compatibility fallback
  would have caused. Strict/evaluation replay may additionally gather rays.
- [x] Add the source-only public selected-pixel backend. The explicit
  `MappedRgb8PowerFoamTargetSource` uses one per-camera raw uint8
  `[H,W,F,3]` pixel-time payload, maps only within one read, copies exact
  requested pixels in caller order, closes the mapping before return, and
  reports mapped address space plus requested page coverage separately from
  logical tensor scratch. Its cache manifest is strict, every payload is
  content-hashed through the opened file at construction, and the caller must
  provide explicit per-payload mapped-address-space and total construction-
  verification byte bounds. That cold
  full-payload verification scan and OS page-cache/readahead remain separately
  reportable system costs; requested-page coverage is not a residency
  measurement. Full-frame evaluation delegates to the existing path/MP4
  source. This is deliberately a standalone source primitive: the
  current unified/per-frame trainer still requests full frames, so registering
  a config key there would be a false integration claim. No cache has been
  generated and no runtime test has run.
- [x] Add the strict source-only public `target_dataset_binding/v1` validator.
  It binds raw-input identities, strict cache-manifest/payload identities,
  matching **declared** raw/cache decoded-float hashes, the actual static-
  intrinsics/dynamic-extrinsics camera grid, and the common stored cache plus
  exact logical frame maps. It can rehash the exact cache files without
  importing Torch or a dataset decoder; it does not decode either side and
  therefore cannot recompute decoded equality. The source and tests remain
  unrun.
- [x] Repair the requested-frame scaling experiment so every row uses one
  fixed 300-frame dataset/provider/camera program and only the
  endpoint-including requested subset changes (`8/64/300`). Bind the dataset
  count, subset rule, and exact index digest; remove per-artifact frame
  signatures; make provider warm generation checks constant-size in
  `F_dataset`; hoist static-camera equality to one cold per-view certificate;
  and let only the cross-row verifier derive structural invariance. These
  source/test changes are unrun.
- [ ] Replace the explicit `SpacetimeBatch.samples` and canonical-position
  tuples with a procedural range/stride manifest only if the paper needs a
  strict `O(K)` requested-observation metadata claim. They remain
  `O(F_requested)` small host metadata today; target/output/residual work is
  unavoidably `Omega(P F_requested)`.
- [x] Add the bounded offline cache converter. The source-only converter uses
  the same open raw handle for identity verification and bounded frame decode,
  transposes through capped frame-major/tiled spools into exact
  `[H,W,F_stored,3]` RGB8 payloads, independently recomputes raw-decoded and
  cache-decoded float32 hashes, and atomically publishes the strict manifest
  and binding without overwrite. Its checked-in CLI supports an exact raw-RGB8
  fixture backend; source/tests are unrun.
- [x] Add the source-only real Neural3D/MP4 decoder and camera-plan adapter.
  Selected CFR frames decode through one converter-owned bounded open handle;
  the descriptor binds the exact logical/native frame maps, MP4 header,
  PyAV/libav/Pillow runtime, source files, poses, adapter/converter source, and
  calibrated camera tensors. Source/target camera paths and start times mirror
  the existing paper loader's role-specific contract; other Neural3D rig views
  share the source start. Sparse selections seek while nearby selections decode
  sequentially, and a successful close certifies exact plan consumption. The
  source-visible RGB conversion cap conservatively includes native and resized
  buffers, while allocator/RSS peaks remain a separate companion measurement.
  Source/tests are statically clean but unrun.
- [ ] Generate a populated public binding and run the public-target companion
  gate. The procedural schema-v3 mechanical gate and the unrun MP4 adapter are
  not public-data evidence. The companion must record cumulative requested
  pages and host/system memory pressure across cache verification, not infer
  file-backed residency from process RSS alone. PyAV/libav/Pillow allocator
  and codec-state peaks remain unmeasured even though visible buffers and input
  reads have explicit logical caps.
- [ ] Stream evaluation and cap retained media.
- [ ] Record target, prediction, residual, atlas, gradient, optimizer, and
  allocator bytes separately; measure live/peak memory rather than relying on
  logical tensor formulas.
- [x] Bound the live compiled CPU chart/sample payload to one spatial block at
  a time. Native topology retention is an entry/byte-bounded LRU and can be
  disabled while still allowing one preflight-bounded live token. Report the
  current cached-token sum separately from the maximum live `B_p x K` scratch;
  evicted blocks may trade memory for re-prepare work. The strict frozen proof
  path remains separate.
- [x] Add an entry- and byte-bounded observation-invariant CPU artifact store.
  It retains programs/lowerings/frame-free samplers, never targets, rays,
  device maps, native payloads, or runtimes.
- [x] Add a replayable dense-observation source with zero retained observation,
  target, ray, or device tensors, bounded track/chunk requests, and exact
  coverage receipts. Its retained scalar metadata is `24(F+V)` logical bytes;
  unavoidable sample work remains `O(PF)`.
- [x] Compose the artifact store and dense replay source at source level into
  one bounded block-major request/step. Each spatial lane spans all observation
  chunks, accumulates one node bar, selects exactly one material-only or
  full-geometry word reverse after exact request coverage, returns one sealed
  request delta, and merges into world-bound step bars. Exact full-manifest
  replay is required before optimizer authorization. Active-block and
  request-delta release boundaries are completion-fenced; the last one returns
  a sealed tensor-free commit receipt. The sample path now returns one sealed
  launch-lifetime token that is installed before native prepare and roots the
  prepared payload, sample block, world/background/loss state, and node/cone
  predecessors until settlement. Settlement owns the only completion fence;
  after successful verification it commits accounting and releases the roots
  immediately. A successful session seal requires
  `prepare_count = launch_count = completion_fence_count`, a maximum of one
  outstanding lifetime, zero outstanding lifetimes at seal, and no retained
  lifetime history. This is source-written and test-written but unrun.
- [x] Source-write the dense caller side of the sample async-lifetime proof.
  CPU decode transfer ownership and sample-materialization predecessors now
  remain leased through the sample completion fence. Current target/sample
  roots, the outstanding launch lifetime, retained active reverse state,
  material/full-geometry execution, native VJP, reduction, and completion
  roots participate in the poisoned restart-required quarantine. A fence
  failure with unknown completion intentionally retains the one bounded
  lifetime and forbids retry or abort release; a successful fence followed by
  validation failure releases completion-safe roots without issuing a second
  fence. The source-written dense failure test reaches a real fake-native
  sample launch and checks those quarantined roots; it remains unrun.
- [x] Source-write the lower-level target-loader partial-failure contract. The
  request rejects arbitrary callables and accepts only a source/request/device/
  budget-bound sealed loader. Its one active lifetime is installed before
  target-provider work, roots the selected read and CPU source before the
  transfer call, adds each returned device tensor, and is released only after
  a sample-completion or abort fence. A sealed finite-state test fault injects
  failure after transfer but before target sealing without admitting arbitrary
  callback code. Failed abort fencing retains loader plus lifetime in the
  accumulator quarantine. Production coordinators now construct this exact
  object. Source and focused tests are written but unrun.
- [x] Give the lazy native material-step route an accumulator-owned,
  restart-required failure quarantine for every root visible after lane
  construction begins. The source-only route now installs lane-runtime and
  sparse target/weight transfer lifetimes before their transfers, publishes raw
  transfer results before later contiguous operations, settles each sparse
  lifetime explicitly, fences reverse scratch before release, and retains one
  bounded fail-stop carrier after an unknown sample/reverse/lane fence. Focused
  behavior tests are source-written but unrun.
- [x] Extend the lazy carrier through union-local map construction and compact
  material gathering. A caller-visible construction slot now exists before the
  first `.to`; raw transfer and `index_select` results are rooted before later
  contiguous operations; CPU duplicate predecessors are dropped only after a
  proven fence; and success retirement leaves no bundle/lifetime reference
  cycle. Source behavior tests are written but unrun.
- [ ] Close the remaining earlier/lower lazy accelerator lifetime gaps. The
  top-level device allocation/`zero_` transaction has no preinstalled owner;
  cold union-map validation chains a device-to-host receipt conversion; native
  forward output ownership after an enqueue exception is uncertified; and the
  supplied fence is not yet a sealed backend/stream-specific capability. Keep
  every non-CPU device fail-closed until those contracts plus native
  allocator/parity/peak gates are proved.
- [ ] Verify the one-in-flight command-buffer policy against the real allocator.
  The dense source now requires its canonical `torch.mps.synchronize()` fence
  after every MPS sample launch, active-block scatter, and request-delta commit,
  but only executing that path with allocator telemetry can prove that queued
  command buffers do not retain `O(F)` K-local inputs or `O(P/B_p)` deltas.
  After proof, test a measured small fixed-depth queue for throughput.
- [x] Remove the free target-loader callback from the canonical coordinator.
  It constructs the built-in decoder bound to source, request, device, target
  generation, and explicit decode/chunk/bridge budgets. The lower-level
  request seam now requires the same sealed exact-type object.
- [ ] Measure decoder-internal host/non-tensor peak. Sealing the loader closes
  the ownership/quarantine hole but does not measure Python or allocator peak.
- [x] Conservatively bound source-visible sample-materialization tensors before
  calling the native prepare seam. The request now derives an effective sample
  cap from an explicit byte budget, evaluates float64 barycentric weights in
  bounded row subchunks, releases each subchunk before the next, and preflights
  `max(N*(8*J+12)+interpolation_scratch+16*K_sub, N*(16*J+32))`, where
  `interpolation_scratch <= 4096+512*J+8*J^2+K_sub*(1024+512*J)`. The
  rank-squared term covers validation of the stored `J x J` fit matrix. The public native
  prepare seam adds `4N+20` bytes after removal of the redundant CPU-int64 row
  copy. Focused source/CPU tests cover subchunk parity and accounting, but were
  not rerun in the 2026-08-04 source-only audit.
- [ ] Measure actual allocator/RSS and Python-object peaks for sample
  materialization. The conservative tensor envelope is not a measurement of
  PyTorch internals, allocator slabs, asynchronous transfer retention, driver
  scratch, the `entries`/`by_row` Python containers, or decoder objects. Keep
  `sample_materialization_float64_scratch_measured=false`,
  `whole_step_python_object_peak_measured=false`, and the separate real-native
  allocator gate closed until the quiet-host scaling run supplies that proof.

Acceptance:

```text
reverse interaction bytes F=256 / F=16 <= 1.10
world parameter bytes invariant in F
selected block bytes scale in K, not F
successful seal has prepare = launch = sample completion fence
maximum outstanding sample lifetimes = 1; outstanding at seal = 0
no retained sample-lifetime history
loss/gradients invariant across K=1, intermediate, F
peak per-step atlas/scratch scales in B_p, not full P
loss/gradients invariant across B_p=1, intermediate, P
no resident full target O(PF), explicit-ray O(PF), or Python-word O(PR)
object graph in the production training step
```

The opt-in paper-mode provider source constructs MP4-backed bundles without
video decode and is designed to seek selected logical frames in bounded blocks.
The material track-staging source contract produces exact `[B_p,K,3]` targets
without explicit sample rays, preserves one global RGB denominator across
arbitrary spatial/temporal partitions, and checks the exact fixed-camera affine
program using one bounded reference row. Strict/evaluation staging still
supports `[B_p,K,6]` rays.
Moving-camera affine compilation fails
closed rather than fitting endpoints approximately. The existing path/MP4
source recomputes its own decoded-content identity by bounded streaming at
final reporting; this does not compare raw-decoder output with mapped-cache
decoded output. CPU/I/O is still linear while residency remains bounded. The experimental source lane
adapts these
blocks into certified native tokens, partitions view-major multicamera samples,
preserves one global selected-sample loss denominator, rebases compact topology
per spatial block, uses loss-only accumulation without a prediction output,
and scatters compact bars into one global site ledger. A source-only row-ragged
Lie reducer now removes the need for a global time refinement inside the sample
kernel. The CPU/source join now also plans arbitrary paper observations, merges
heterogeneous native blocks through one union-local compact bar, and coordinates
all view/block bars under one global denominator and optimizer authorization.
The source boundary now includes a bounded point-cloud initializer, a direct
dataset-bound static-camera track compiler, fixed-site raw material/manual SGD,
and a raw-only restart checkpoint. These are runtime-unverified and deliberately
start with zero velocity/P0 material. A caller-owned one-step material-only
coordinator now performs exact canonical coverage and returns an unconsumed
authorization plus its exact accumulator/replay receipt without mutating
parameters. The remaining integration boundary is quiet-host coordinator-to-
CPU-SGD/two-step/resume evidence, production-scale compiler evidence, a fenced
accelerator updater and built native binary, streamed evaluation, and a
distinct unified-runner lane.

For `B_p=8192`, `K=8`, and `J=16`, the audited native node state plus node bar
is about `4 MiB`. One float32 material target block is `0.75 MiB`; the
loss-only ABI adds no prediction or explicit-ray block. The target-only route
saves `1.5 MiB` versus bounded target-plus-ray staging at this block size.
Forward media/evaluation may explicitly add a separate `0.75 MiB` prediction
and retain bounded rays. There is no intrinsic 32-GB requirement. These are
source tensor-payload counts, not a measured allocator peak.

## Phase 3: sparse continuous event compiler and kinetic geometry frontend

- [x] Replace fixed-time all-pairs `O(S^2)` discovery with an exact rational
  lower envelope of `S` power-distance lines: `O(S log S)` work, `O(S)`
  scratch, and only adjacent active faces. Supplied piecewise charts stream
  natively at source level; complete continuous projective compilation remains
  open.
- [x] Keep the all-pairs path only as a small-fixture oracle.
- [x] Discover complete owner words over continuous affine camera charts in
  the exhaustive CPU reference. The universal seam set is pair near/far roots,
  finite triple concurrence, domain endpoints, full-fiber classification, and
  separate ray-collapse events; denominator-only roots are analytic guards.
- [x] Add an independent continuous all-site owner witness; denominator/order
  checks alone are insufficient.
- [x] Prove/check every run endpoint against every competitor so a third cell
  cannot undercut a pairwise-valid supplied transition.
- [x] Derive exact rational near/far crossing and triple-concurrence event
  polynomials, including certified rational isolators for irrational quadratic
  roots, a right-continuous seam policy, and zero-run transfer equivalence.
- [x] Characterize the fixed shared-SPD(4) world exactly, in a fixed coordinate
  gauge, as the restricted common-translation/fixed-normal kinetic 3D family;
  retain one shared camera/scene gauge for bulk motion.
- [x] Add the direct affine kinetic 3D CPU frontend with degree-`<=2` weights,
  exact degree-`<=2` pair/ray coefficients, exact degree-`<=4` concurrence
  polynomials, fixed-gauge rotating-face coverage, and frame-independent
  parameter bytes.
- [x] Add exact rational square-free/Sturm isolation through quartics and guard
  candidate concurrence roots against full-fiber ties and zero cut
  denominators at CPU scope.
- [x] Add exhaustive half-open continuous chart emission, exact left/right
  activity filtering, algebraic root clustering, explicit ray-collapse and
  full-fiber rejection, and independent oracle parity for small worlds.
- [x] Implement active-boundary certificate closure using endpoint owners
  versus all sites plus every active cut versus all competitors. Predicate
  generation is `O(U S R_max)` across unique words; repeated root-complement
  discovery/certification is separately `O(W (S log S + S R_max))`. Do not
  collapse that to `O(SR)` or claim `O(delta R)` until an event queue and a
  kinetic-neighbor supergraph are separately certified.
- [x] Add sparse frozen-program kinetic site/trajectory/weight/ray/material
  VJPs and exact multi-chart binary-sample dispatch on CPU. The end-to-end
  reverse consumes only `O(sum J_c)` node cotangents after blocked sampling.
- [ ] Add supported persistent/simultaneous/full-fiber material semantics or
  retain the current fail-closed policy. The single-ray direct kinetic topology
  now lowers to the source-native node-length seam and its length bars feed a
  CPU frozen-geometry VJP; bounded equal-rank track-chart lowering is green,
  while dataset-bound program generation and real Metal parity remain open.
  Full-fiber guards fail even when their left/right open owner words match.
- [x] Retain polynomial guards through source chart dispatch for binary sample
  times, bind zero-run identity seams, and use a right-continuous/one-sided
  fixed-topology convention. Irrational algebraic endpoints remain non-paper
  until native domains represent them exactly; full-fiber ties and event-time
  derivatives still fail closed.
- [x] Define and implement the narrow event-free directional geometry-update
  reuse rule.
  Split the long-lived geometry/ray/owner certificate from mutable material
  refresh. A full frozen-world transfer/Jacobian certificate is a publication
  gate, not something to rerun over every `B_p` block after every color/density
  optimizer step. The current exact certificate covers one strict single chart
  along one rational directional homotopy. The restricted multichart reference
  covers separated singleton simple roots through complete-registry
  recertification and endpoint re-isolation, but it does not return a patched
  program or rebuild charts, ranks, payloads, or dispatch. Until that machinery
  exists, **all geometry and camera-ray updates recompile**; only material-only
  updates reuse a sealed structural artifact. Warm affected-source repair
  remains optional and unbuilt.
- [x] Keep the source training topology independent of learned-density early
  termination.
- [ ] Remove or explicitly shard the 32-frame, 256-site, 4093-boundary, and
  129-run caps.

Acceptance:

```text
fixed physical interval, F=4..256 unique samples
active adjacency, word records, events invariant/nearly invariant in F
small fixtures exactly match all-pairs/per-frame owner words
refresh/fallback fraction reported during optimization
```

## Phase 4: required World-Tubes-shaped total-transfer atlas

The exact constant-state/sparse-incidence path is frame-memory-light, but its
word scan is still `O(P F R)`. That is not the requested World Tubes shape.
The compiled atlas is therefore required for the strong systems claim: exact
world/word work at `J` nodes, followed by a cheap linear sample slice.

- [x] Implement the CPU `J`-node total-transfer compilation and shared world
  adjoint with no `F x R` tape.
- [x] Bind the active kinetic owner program to exact ordered P0 transfer over
  every safe chart, dispatch binary samples right-continuously, reduce streamed
  residuals to `O(sum J_c)` node cotangents, and run one sparse frozen-program
  geometry/material reverse. Structural/reverse state is invariant in `F`;
  irrational seam neighborhoods and total compiler/event derivatives remain
  outside the claim.
- [x] Use joint primal-and-tangent adaptive rank rather than a fixed `J`.
- [x] Compare raw `(beta,m)` interpolation against the affine-transfer Lie
  logarithm
  `kappa=-log(beta), v=kappa*m/(1-beta)`, with stable analytic encode/decode
  VJPs and physical-cone checks. On the hard moving-opacity fixture the Lie
  chart is at floating-point error from `J=2`, while raw transfer has
  `8.21e-2` error at `J=2` and `2.88e-3` at `J=16`. A raw-linear negative
  control correctly favors raw coordinates, so chart/rank selection must be
  adaptive.
- [x] Integrate the Lie chart through the exact word/world VJP. The hard chart
  falsifies forward-only rank selection: forward error stays near `1e-15`, but
  maximum world-VJP error falls from `1.40e-2` at `J=2` to `1.27e-8` at `J=32`.
- [x] Add a source-only bounded fixed-word native node compile, streamed sample
  reducer, node VJP, and sparse-incidence finalize under a suffixed ABI.
- [x] Add the general-kinetic precompiled-length source seam: compact CSR
  owners plus `[J,R]` physical lengths to affine-Lie node charts, and the
  reverse from arbitrary Lie bars to compact P0 material bars plus length bars.
- [x] Add row-ragged source reduction over selected `(track,chart)` rows using
  row-local `[N,J]` weights. Repeated rows atomically share node bars; unselected
  rows remain untouched; loss-only launch allocates no prediction tensor.
- [x] Close the step-scoped invocation lifecycle at CPU/fake-native scope.
  Execute spatial bundle outer / temporal chunk inner, accumulate every chunk
  into the same node cotangents, run each material-only ordered-word VJP once,
  union-scatter once, and issue one global optimizer authorization. `K=1/4`
  and `F=5/41` invariance tests guard against reintroducing
  `ceil(F/K) * J * R` word work. Material-only reverse allocates no `[J,W]`
  length bar; geometry reverse remains a separate explicit path.
- [x] Implement the executor-bound full-VJP request/step source path. The
  executor seals exact ordered sample manifests, normalization, node/loss
  accumulator identity, and a single-use full reverse receipt. Dense replay
  can now select `full_geometry`, fence and reduce each returned `[J,W]` length
  bar immediately, and merge one combined request delta into step-owned site,
  trajectory, weight, ray, material, and loss bars. No free-standing caller
  coverage string or third coordinator was added. The executor consumes the
  native result only against the sealed geometry reduction and labels its proof
  `fenced_and_reduced_not_globally_committed`; higher layers separately fence
  each block scatter and request-delta commit. Executor poisoning retains
  native references, abort is fence-before-release with retry after a failed
  fence, and the legacy standalone finalizer is hard CPU/fake-native-only.
- [x] Add the top-level fixed-camera full-geometry authorization coordinator as
  a sibling of the material-only coordinator. It forces
  `full_geometry=True`, disables camera-ray optimization and storage, and now
  accepts an explicit staged-sparse or fused-v1 reverse policy. Mode-specific
  accounting requires exactly one selected reverse per active block: staged
  requires one fenced sparse reduction per block, while fused requires one
  all-active-block transaction and zero `[J,W]` cotangent. It rejects
  material-only VJP launches and forged ray claims and returns sealed site
  position/velocity/weight plus material/loss bars. It performs no mutation and
  is source-only/unrun; staged remains the default.
- [x] Add the source-only combined material/geometry updater and explicit
  checkpoint. Geometry candidates are out-of-place; the old artifact store and
  provider are retired/poisoned; a fresh immutable world/provider is built;
  and the exact manifest is cold-compiled and digest-bound before promotion.
  Recompile streams through the configured bounded LRU even when the manifest
  is larger than the cache. A full cloned checkpoint is an explicit
  post-promotion operation rather than retained beside every live generation.
  The policy/receipt conservatively adds old-store residency and fresh-store
  capacity to state, duplicated immutable geometry, candidate, authorization,
  and validation scratch; compiler scratch and real allocator peak remain
  unmeasured. The scope is store-owned/tracked memory and requires an explicit
  zero-retained-prior-generation attestation; caller-retained retired objects
  are outside the bound. Stateless CPU SGD and fixed cameras are enforced.
  The receipt now binds the consumed full-geometry result generation and its
  selected reverse mode. Checkpoint creation and strict policy-bounded payload
  parsing now exist; rebuilding the live provider/store/material generation
  and cold-recompiling it on restore do not. The combined parser now rejects
  schema/digest drift, derived logical-byte lies, oversized backing storage,
  clone-coexistence overflow, and oversized request-local track ranges before
  full geometry scans or clones. Nested material policy/parameterization still
  needs current-config authority at live restore. Source/tests remain unrun.
- [ ] Add two sealed, frame-independent production bridges while leaving raw
  optimizer/material state and geometry on CPU. A staged-material snapshot
  must bind the CPU state identity and tensor version, material generation,
  exact MPS `[S,4]` destination/signature/bytes, and canonical completion
  fence. A post-step material-gradient bridge must bind the exact
  authorization/result/accumulator generation, MPS bar identity/version, one
  fenced CPU `[S,4]` clone, and single-use consumption. Do not hash the full
  live material every step; the live chain deliberately uses identity/version
  signatures. This adds only `O(S)` transfer state and avoids moving the CPU
  optimizer lifecycle onto MPS or introducing implicit `.item()` syncs in its
  invariants.
- [ ] Implement live combined-generation restore under current-config
  authority. Rebuild the immutable world/provider from checkpoint geometry,
  restore the CPU material generation, create a fresh bounded artifact store,
  cold-recompile the exact manifest, and require the semantic recompile seal
  digest to match before returning a fresh coordinator. Then bump the
  checkpoint schema once so its current hard-coded
  `combined_checkpoint_restore_integrated=False` evidence becomes true. Use
  the existing atomic checkpoint filesystem helpers in the trainer.
- [ ] Promote that full-geometry path only after its focused CPU/fake-native
  tests run on a quiet host, followed by rebuilt-Metal parity and allocator
  evidence. It currently remains source-only and runtime-unverified.
- [x] Split immutable topology, world-refresh, chart, `K`-block, and
  world-gradient tokens; bind tensor versions, certificate generations,
  global `P*F*3` normalization, and half-open chart/`K` partitions.
- [x] Bind native tokens to a canonical rerun of continuous transfer,
  world-Jacobian, site-geometry, and all-competitor owner certification;
  reject fabricated/stale/mutated bindings and derive `K x J` interpolation
  weights from certified chart times rather than caller input.
- [ ] Build that source and establish bounded Metal runtime parity; the current
  installed extension predates it and contains none of the required kinetic
  schemas. Run `verify_worldfoam_native_variant_imports.py` after rebuilding;
  wrapper callability is no longer accepted as compiled-ABI evidence.
- [x] Certify continuous forward transfer error for fixed-topology P0 affine-ray
  charts with bounded work budgets.
- [x] Certify the continuous world Jacobian/VJP error for the older supplied-
  word/fixed-4D reference, including an optional conservative
  boundary-to-power-site operator bound. Do not transfer this status to the
  direct-kinetic geometry route.
- [x] For the direct kinetic frontend, certify the actual cleared second-form
  barycentric evaluator continuously against exact word replay for primal and
  referenced-material Jacobian error, then convert the entrywise bound into
  declared-norm material JVP/VJP bounds. Continuous geometry approximation,
  floating-point roundoff, and runtime dense fallback remain uncertified.
- [ ] Prove and implement the missing direct-kinetic uniform geometry/ray
  tangent bound for the actual affine-Lie barycentric evaluator:
  `sup_t ||D_theta G(t)[v]-D_theta G_J(t)[v]|| <= epsilon_geometry` for a
  declared bounded family of sparse site-position, velocity,
  weight-trajectory, and affine-ray directions. Include zero-opacity/high-
  opacity limits and keep nodes, weights, rank, charts, and dispatch frozen.
- [x] Prove the conditional sparse local-to-global normalized-loss VJP lemma.
  It includes both the Jacobian-action error and the output-cotangent change
  induced by primal approximation error, and it introduces no artificial
  frame-count factor or dense global dual.
- [ ] Instantiate that lemma with certified direct-kinetic per-track
  `epsilon_0/epsilon_1` bounds and the actual sparse gather/scatter norms. This
  remains blocked on the preceding uniform geometry/ray tangent certificate,
  not on the loss-composition algebra.
- [x] Enforce the affine-Lie physical cone in sampled and continuous gates.
- [ ] Lower every certified visibility/topology birth/death event end to end in
  the native backend. Exact active CPU compilation and multi-chart dispatch no
  longer use requested frames, and the existing native source adapter streams
  caller-supplied piecewise charts. Exact irrational native endpoints,
  projective/rational camera programs, supported full-fiber material rules,
  and event-time/discrete-dispatch derivatives remain unresolved.
- [ ] Close arbitrary-density seam dispatch by comparing every rational sample
  time exactly against algebraic event roots in the native/runtime path. Until
  then, state the weaker theorem assumption that requested samples avoid all
  unresolved isolator neighborhoods. Do not refine isolators from the
  requested frame grid and then call compilation frame-independent.
- [x] Compare exact `O(FR)` replay against compiled `O(JR + FJ)` evaluation
  on the identical CPU world over `F=16..1024`. The two-run fixture proves
  frame-independent world work/state, not practical speed; realistic large-`R`
  native rows remain mandatory.

The older `0.930x` interaction proxy omitted coefficient fitting and
sample-weight construction. A verified fit-derived barycentric route now gives
`O(sum F_c J_c)` common-path weight work per spatial block and reports
exact-node/fallback rows; the current full material step therefore constructs
weights in `O(N_B sum F_c J_c)` rather than once globally;
the dense `O(sum F_c J_c^2)` path is only a row-local oracle. Even with linear
weights, the low-run `J=16/2/2`, `F=1024` fixture routes to exact replay
(`11608` compiled proxy interactions versus `6144` exact) and has no temporal
break-even. Realistic high-`R` native rows remain mandatory.

The deterministic sampled rank selector now uses multiple held-out directions
and scale-normalized parameter-block errors, and the separate continuous
wrapper bounds fixed-word P0 transfer and first derivatives without sampling.
It now also certifies supplied fixed-word owner identity against all competitor
sites, while runtime floating-point roundoff remains explicitly uncertified.
The hard rank-16 chart also exceeds the current bounded
certificate budget, so certificate cost/rank death must be reported rather
than hidden.

The dense continuous certificate is an oracle, not a production option. With
`D=5B+12P+4I+4S`, its pointer-slot lower bound is
`max(16D^2, 64 P J_max D)` bytes. At `P=8192,J=16`, even the impossible
`B=I=0,S=1` case exceeds `768 GiB` before any Python interval or `Fraction`
objects. Production strict certification must use the bounded
`track_local_sparse` mode; runtime allocator measurement remains separate.

## Phase 5: materials and unified paper lane

P0 geometry/systems integration is not blocked on a universal M3/M5 winner.
Only promotion of the richer material basis waits for adaptive or real-data
evidence.

- [x] Wire a source-level owner-topology-only P0 material optimizer through the
  native lifecycle. It retains only lightweight compact topology, a compact
  spec schedule, and an owner binding per block; it retains zero compiled CPU
  atlases and performs zero per-step CPU atlas compiles. It decodes density with
  softplus and RGB with sigmoid and applies exact manual transform VJPs. It is
  explicitly non-paper until a frozen checkpoint passes strict
  transfer/Jacobian recertification.
- [x] Implement the narrow production-source fixed-site material lifecycle:
  bounded zero-velocity point-cloud initialization, exact static-camera
  programs, canonical material decode/VJP, CPU-only `48 B/site` manual-SGD
  state, `16 B/site` raw checkpoint, and a caller-owned exact-coverage
  authorization coordinator with a built-in target decoder. This is source
  status only; no focused gate ran on the saturated host.
- [x] Implement a fail-closed material-only memory-scaling acceptance verifier.
  Schema v3 requires three fresh-process repeats at `F=8/64/300` over one fixed
  physical interval, real native coordinator integration, complete cold
  dense-track compilation, zero saved-autograd state, direct selected-pixel
  target receipts, fresh-process RSS, lower-bound public MPS maxima from the
  configured 5.0-ms sampler under an applied `<=2 GiB` MPS allocator limit, and
  a separate 0.25-second sampled process-group watchdog that terminates above
  `4 GiB`. Raw limit/watchdog receipts and the transitive Python/native source
  manifest are hash-bound. Exact observable resource attestation covers node
  forward, loss-only sample accumulation, and material-only word VJP; the query
  alone is not execution evidence. The gate rejects fake-native and any
  full-geometry claim, while private/register/spill bytes remain explicitly
  unobservable.
- [x] Replace dense one-shot construction with caller-owned two-phase source
  ownership. The lane now installs and retains the union-local construction
  lifetime plus one runtime-construction lifetime per block before device
  materialization; partial failures quarantine the aggregate current source,
  transferred destinations, partial bundle/runtimes, and executor. The lane
  retains those roots through its outer release boundary. Its conservative
  lane cap now charges both the fresh CPU union/map predecessors and their
  device destinations, plus one retained CPU epsilon scalar per runtime;
  CPU aliasing is not used to understate an accelerator peak. Focused
  fake-native and fault-injection tests are written but unrun. This closes the
  Python-root hole, not accelerator release authority: the exact sealed receipt
  remains required by the next item.
- [x] Replace the dense route's return-allocating node forward with the
  existing caller-preallocated `launch_node_forward_into` ABI. A bounded
  gather lifetime is installed before `index_select`; the compact predecessor,
  caller-owned node output, world, and token remain rooted until that block's
  reverse fence, the fused transaction fence, or a proven abort/outer fence.
  Enqueue-then-raise plus failed-fence quarantine coverage is written. Logical
  active-state bytes do not increase because these carriers alias the already
  charged compact/node tensors. Source/tests remain unrun.
- [x] Replace arbitrary completion callbacks and free provenance strings on the
  public CPU-only lazy-material entrypoint with an internally minted exact
  completion capability. Every supported phase now registers one monotone
  launch epoch before work, fences that exact epoch, consumes one exact
  capability/owner/device/stage/launch/sequence-bound receipt, and only then
  releases roots. Capability and receipt identities are clone-resistant;
  failed fences poison one bounded quarantine without retry. Bundle
  materialization is pre-launched, over-coverage is retired then rejected, and
  one separately charged terminal probe proves exact iterable exhaustion.
  Bundle/sample wrappers drop the previous yielded payload before requesting
  the next, preserving the intended one-lane/one-sample peak. Source and
  behavior tests are written but deliberately unrun under the host-safety
  pause. This checkbox covers only the CPU-lazy source contract, not native
  runtime evidence or the legacy low-level/dense APIs.
- [ ] Complete sealed accelerator and dense-path promotion. Accelerator
  capability minting remains fail-closed until canonical native ABI/build,
  device, and launch-domain attestation is bound. Before opening it, bind every
  releasable lifetime to its installing capability/owner and pre-launch epoch;
  make authority-free convenience release methods CPU-only; move sparse
  transfer plus executor settlement into one post-fence revalidate ->
  consume-once -> commit-both transaction so a GIL-releasing synchronize cannot
  swap the active subject; and put any accelerator optimizer/update work under
  its own registered epoch. Do not partially label the dense path sealed: dense
  still has construction, delta commit, staged/fused geometry, post-accept
  commit, abort, optimizer, and lane-release callback seams.
  - [x] Source-seal the lazy sparse-sample composite. One stable slot is now
    installed before sample materialization and binds the exact capability,
    owner, subject, session, frozen plan, stream, transfer lifetime, sample
    block, executor lifetime, launch epoch, pending completion, and receipt.
    The fence returns the receipt unconsumed; the outer coordinator revalidates
    the complete relation, consumes once, then applies one sealed executor
    commit plan plus exact stream/slot assignment commits. Subject bindings
    strongly retain their exact subject, clone/ABA substitutions fail closed,
    and any precommit failure retains the bounded composite in the trainer's
    restart-required quarantine. Legacy callback release paths and public
    authority-free convenience releases are CPU-only. Focused behavior tests
    are source-written and statically aligned but deliberately unrun under the
    host-safety pause.
  - [ ] Bind lane construction, node-forward/gather, reverse, lane release,
    bundle construction/exhaustion, dense request-delta/post-accept commits,
    and optimizer/update launches to analogous stable prelaunch subjects.
    Direct authority-free accelerator launch APIs remain a promotion blocker;
    do not open accelerator capability minting until every listed lifetime has
    the same exact settlement proof.
- [x] Correct source-only work telemetry so `node_forward_thread_count` reports
  `sum_b R_row,b J_b`, while `node_forward_interaction_count` reports the real
  `sum_b W_run,b J_b` CSR scan and must equal the material word-VJP interaction
  count. The previous value understated absolute forward world work by the
  mean runs per row; fixed-`F` invariance was unaffected. Tests remain unrun.
- [ ] Produce the measured rows that make the material-only memory verifier
  pass. The checked-in real MPS coordinator driver/config and the v3 source
  contract now exist. Rebuild and attest the Metal extension, then run the
  opt-in rows on a quiet host satisfying the 8-GiB-availability launch guard.
  That guard is safety headroom, not a 32-GB representation requirement. The
  verifier, producer, driver, native attestation, and behavior tests are
  source-only and remain unrun on this host.
  The checked-in v3 fixture currently has only two world sites at `384x384`.
  Even a passing `F=8/64/300` artifact is therefore a mechanical
  sample-density/retention result, not evidence that the paper trainer fits.
  The distinct training-memory acceptance row must use the spatial-block
  route at `384x512` with `1024` global sites, execute a fenced device VJP and
  real CPU optimizer mutation, and report bridge, allocator, and process-RSS
  peaks. Do not merge those two claims.
- [ ] Add the distinct native full-geometry memory/scaling gate. The production
  source coordinator now exposes staged/fused geometry authorization and the
  CPU updater consumes it, but the dense coordinator exposes only staged
  sparse and fused-direct v1. After rebuilt-native parity, route the existing
  fused-union v2 transaction so geometry scratch is request-union `O(U)` rather
  than global `O(S)`. None of these paths has real-native or allocator evidence;
  do not infer it from the material-only result.
- [ ] Run the real sealed coordinator-to-CPU-updater, two-step loss-decrease,
  checkpoint/restore, and restart-history gate on a quiet host.
- [ ] Reuse the existing M0--M5 `(bar_tau,bar_beta,bar_m)` material VJP ABI.
- [ ] Test adaptive M3/M5 selection or real heldout material observations.
- [ ] Register `worldfoam_native4d` as a distinct unified runner lane.
- [ ] Keep `MetalPowerFoamVideo` explicitly labelled per-frame.
- [ ] Run bounded same-world replay-versus-compiled scaling before public
  multi-scene training.

The G4 execution/evidence boundary is now checked in separately at
`src/train_configs/paper_protocols/worldfoam_native4d_g4_public_quality_v1.jsonc`,
`research_experiments/world_foam_lane2/run_worldfoam_public_quality_ablation.py`,
and `verify_worldfoam_public_quality_ablation.py`. It freezes the honest
3-scene x 3-seed x 4-route, 36-row matrix and aborts before the first baseline
while the public native4d worker is unavailable. This is not lane registration
or public evidence: the current blockers are the train-only mapped-cache
binding, missing heldout target/evaluator bridge, procedural-only production
adapter, missing public row worker/capability receipt, and unverified native
binary. The existing unified `worldfoam` lane remains ineligible and must not
be relabelled.

### Unified-runner blockers

Do not register `worldfoam_native4d` as paper evidence by wrapping the current
fixture. The source material session is a hand-built fixed-topology rectangular
program, while the paper protocol changes its sampled view/time shape every
step. The dense full-geometry authorization now passes through an explicit
staged/fused fixed-camera coordinator into an out-of-place CPU combined updater
that retires stale structure and cold-recompiles a fresh bounded working set.
A bounded production-source world initializer, exact static-camera track
factory, raw-only checkpoint creator/parser, and next-step claim also exist,
but they are narrow and unrun; live restore is absent. There is still no
fenced accelerator updater, streamed prediction evaluator, evidence writer, or
runner dispatch.
The minimum honest bridge is:

1. rebuild the stale extension, add the separate exact five-kernel selected
   full-geometry attestation, and run same-input staged-versus-fused parity;
2. add the sealed CPU-to-MPS material snapshot and MPS-to-CPU material-gradient
   bridges, preserving one forward/VJP per active block and one synchronized
   `K` launch in flight while keeping optimizer state CPU-resident;
3. implement semantic live restore and pass uninterrupted two-step versus
   checkpoint/restart loss/state parity. Every geometry update must discard
   and rebuild/reseal the program state;
4. build the first trainer for the existing single-stage
   `coffee_martini_full_300f_fixed_512_pixel_matched_v1` control only: fixed
   `384x512`, fixed `1024` sites, fixed `512` pixel tracks, and fixed optimizer
   policy. The `F=8/64/300` memory sweep must reuse the exact same track ids,
   300-frame physical grid, camera program, target provider, and compiled
   world; only the endpoint-including requested-time subset may change, giving
   exactly `512*F` streamed observations and a `512*F*3` loss denominator. Use
   the existing all-competitor active-owner compiler with
   `maximum_sites_per_track_compile=1024`, record its cold CPU cost, and lower
   only certified active owner words into compact device blocks. Never replace
   that proof with an uncertified 64-site crop. Fail closed on a resolution,
   site-count, track-manifest, camera/target generation, or policy stage change.
   The 600-step progressive
   row needs a separate stage-transition transaction for resolution changes,
   `256->512->1024` site growth, new compiler/provider state, and authorized LR
   multipliers; it does not block the first trainer;
5. add forward-only streamed reconstruction/evaluation, strict frozen-
   checkpoint recertification, metrics/media, and cost/allocator accounting;
6. run a separate full-geometry gate: staged/fused parity at bounded `F=8`,
   then fused-only fresh-process `F=8/64/300` under the existing incident
   guards; and
7. only then register a distinct `worldfoam_native4d` lane.

The existing unified `worldfoam` lane remains the explicitly per-frame
`MetalPowerFoamVideo` baseline. It must not be relabelled as this path.

### Deferred connection/curvature falsification gate

The audited fiber/connection proposal is recorded in
`research_notes/worldfoam_paper/WORLD_FOAM_STRATIFIED_LAGRANGIAN_CONNECTION_AUDIT_2026-08-05.md`.
It does not preempt the fused-v1 build, parity, allocator, trainer-lifecycle,
or `F=8/64/300` gates above.

- [ ] After those gates, first compare direct physical transfer `U` with the
  flow-corrected group-completion transfer `U_tilde`, using the existing
  approximation family but an oracle-local unrestricted `beta>0` affine chart.
  Do not feed `U_tilde` through the current physical Lie-cone ABI: endpoint
  inverses can produce `beta_tilde>1` and signed moments. Only if the
  independently specified flow materially lowers certified primal and tangent
  node counts should the same CPU/source oracle add transported curvature
  source `K_F` as a signed four-component tangent, not as a transfer. All three
  routes must reconstruct the same physical `U` under identical continuous
  primal and selected-tangent tolerances. Charge the compact shared flow,
  endpoint transports, reconstruction, cone checks on reconstructed `U`,
  conditioning, and gradients through the flow.
- [ ] Run a separate neighboring-track topology-template reuse census before
  considering sensor-time patch compilation.
- [ ] Separately census generic-event seam defects
  `delta_e^(0)=||G_+-G_-||` and
  `delta_e^(1)(D)=sup_{v in D, ||v||<=1}||D_vG_+-D_vG_-||` before considering
  a two-level exact-owner/coarser-transfer atlas. Park that branch when the
  required geometry/material action defects remain large at nearly every
  owner seam.

Do not add a curvature runtime unless total retained payload and
`sum_b J_b W_run,b` ordered-run work improve by at least `2x` against both
direct alternatives and measured request time improves by at least 20%.
Per-frame flow state, hidden per-ray
answer tables, tangent-rank regression, unphysical reconstruction, or failed
conditioning kills the runtime branch. Keep curvature as a theorem/diagnostic
if it fails. Do not implement sheaf, stack, or monodromy software for this
gate.

## Stop rules

Stop or narrow the systems claim if event/adjacency/rank grows with frame
sampling density at fixed physical duration, most words rebuild every step,
compiled gradients miss exact replay tolerance, or quality requires enough
cells/material state to erase the shared-memory advantage.

## Execution safety

CPU/source gates are permitted only after a live resource preflight. On
2026-08-04 the host again reached load averages near `80--100`, so even focused
Python/pytest gates were paused. The final preflight improved to
`9.37/12.33/27.22` but exposed only about `68 MiB` of free VM pages, so the
pause remains in force. No broad MPS run is authorized by this TODO.
Metal parity must be a tightly bounded opt-in smoke after the host is quiet and
has real memory/swap headroom. Publication training belongs on an approved
clean host.
