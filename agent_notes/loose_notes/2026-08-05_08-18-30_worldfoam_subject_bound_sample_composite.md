# WorldFoam subject-bound sample composite

Date: 2026-08-05 KST

## Scope

This session continued the memory-light WorldFoam lane under the machine-safety
pause. No Python process, import, pytest node, build, Metal/MPS/CUDA operation,
or training run was launched. Work was limited to source edits, shell reads,
static scans, a test-source migration, and a read-only red team.

The immediate objective was narrower than accelerator promotion: close the
sample-level ownership hole that existed between sparse target materialization,
the native sample executor, and the asynchronous completion fence. Accelerator
capability minting remains deliberately disabled.

## Relation to the scientist's new mathematics

The latest fiber/connection/curvature proposal was fully audited separately in:

- `research_notes/worldfoam_paper/WORLD_FOAM_MEASURE_CONNECTION_SYNTHESIS_2026-08-05.md`
- `research_notes/worldfoam_paper/WORLD_FOAM_STRATIFIED_LAGRANGIAN_CONNECTION_AUDIT_2026-08-05.md`

The useful mathematical result is the constrained Lagrangian ray-fiber optical
connection and its transported-curvature variation identity. It distinguishes
coherent temporal motion from residual optical evolution. It does **not** yet
replace direct ordered transfer in the runtime. Direct physical `U` remains the
production ABI; `U_tilde` and `K_F` remain oracle ablations until they reduce
certified node count and end-to-end time under capacity-matched comparison.

The current memory win therefore still comes from the already-derived kinetic
ordered program:

```text
world-side forward + material backward
  = 2 sum_b J_b W_run,b
```

at fixed compiled program complexity, independent of requested frame count.
The unavoidable sample/output slice remains linear in requested observations.

## Failure found

The sealed sample executor previously consumed its receipt internally before
the outer sparse stream could prove that its exact active transfer was still the
one covered by that fence. A device synchronization may release the GIL. That
made this sequence unsafe for accelerator promotion:

```text
register/fence executor sample
consume executor receipt
re-read mutable stream.active_transfer
release stream transfer
```

Another thread or callback could in principle substitute the outer subject
between the fence and release. Identity digests alone were also insufficient:
the binding did not strongly retain the subject, permitting ID-reuse ABA if the
outer root was lost.

## Implemented source contract

### Exact subject binding

`kinetic_sealed_completion_fence.py` now has a frozen, slotted,
clone-resistant `PaperKineticCompletionSubjectBinding` that binds:

- exact capability identity and generation;
- owner generation and private capability nonce digest;
- exact strongly retained subject object;
- subject kind, identity, and immutable generation digest.

Launch epochs and receipts propagate that exact binding. A subject-bound
receipt cannot use the legacy relation-string `consume_for`; it must use
`consume_for_subject` with the exact binding and exact subject. Sample stages
and every accelerator stage require a subject binding. Accelerator capability
construction remains fail-closed.

### Stable outer sample slot

`kinetic_lazy_native_material_step.py` now installs one
`_SampleCompositeSettlementSlot` before advancing the sparse sample iterator.
The immutable slot generation binds:

- logical step and bundle generation;
- exact frozen plan identity/generation;
- exact native session identity/generation;
- exact sparse stream identity;
- launch ordinal and prior covered-sample count;
- the slot's own identity.

The slot strongly retains the plan, session, and stream. Ordered publications
then add the exact subject binding, launch epoch, materialized sample block,
active transfer lifetime, native sample lifetime, and pending completion.
Publication and preconsume checks prove all cross-relations, including:

```text
stream.plan is slot.plan
transfer.sample_block is slot.sample_block
executor_lifetime.sample_block is slot.sample_block
executor_lifetime belongs to slot.session
pending._sample_lifetime is slot.executor_lifetime
pending.subject_binding is slot.subject_binding
pending._launch_epoch is slot.launch_epoch
```

The launch epoch is registered before `next(sample_iterator)`, so target/device
materialization and the subsequent native sample launch share one conservative
completion domain.

### Deferred receipt consumption

Sealed `settle_sample_accumulate` now fences once and returns a
`KineticNativePendingSampleLaunchCompletion`. It deliberately keeps the exact
receipt unconsumed and retains executor roots. The coordinator then performs:

```text
slot relation validation
exact stream active-transfer validation
exact executor/pending/subject validation
preallocate sealed executor commit plan
consume the one exact subject-bound receipt
executor assignment commit
exact stream assignment commit
slot assignment commit
```

The executor commit plan is frozen, slotted, sealed, exact-session/pending
bound, clone-resistant, and one-shot. It is allocated before consumption and
authorized only after successful receipt consumption. The public commit keeps
only constant-time exact authorization guards after consumption; its release
tail does not call native code or allocate. The final sample completion receipt
remains unsealed and assert-invalid while roots are pending, then is sealed only
after executor root clearing.

### Failure behavior

Any exception while a sample slot is installed now enters the trainer-owned
bounded restart-required quarantine. It does not invent a second epoch, retry a
fence, consume a partial abort, or close the suspended stream. The quarantine
retains the slot, pending completion, exact receipt reachability, native
session/lifetime, transfer, sample block, lane roots, capability, and traceback.

This is intentionally fail-stop. It prevents use-after-free and unbounded retry
history; it does not promise in-process recovery after an asynchronous failure.

## Authority-free paths closed

Public sparse-stream, sparse-lifetime, union-transfer, partial-bundle, and
node-forward convenience releases now reject non-CPU use. Legacy callback
settlement/release in sample, abort, fused geometry, and staged geometry paths
also rejects accelerator use before invoking the callback. This does not make
the dense route sealed; it makes unsupported accelerator authority fail closed.

## Static verification written, not run

Focused source tests now cover:

- strong subject retention and clone/foreign/ABA rejection;
- subject-bound epoch/receipt consumption and one-shot behavior;
- sealed settle returning an unconsumed pending completion;
- no second sample launch while a pending completion exists;
- exact prelaunch plan/session/stream/slot binding;
- post-fence validation failure retaining all roots with no optimizer call;
- consume-once and executor -> exact-stream -> slot commit order;
- forged/preconsume/reused commit-plan rejection;
- metadata-only non-CPU legacy callback rejection.

Static whitespace/conflict checks were clean. None of these tests were executed
on this host.

## Red-team result

The read-only audit found no remaining normal-path sample-composite P0 after the
cross-relations and commit-plan authorization were added. One narrow P1 remains:
the postconsume composite crosses three Python method calls. An asynchronous
exception between those assignment commits can make later quarantine
self-validation reject the partially committed shape. Exact roots are still
held and completion is known, so this is a bounded fail-stop leak/restart case,
not a use-after-free. It does not justify opening accelerator minting.

## Still open before accelerator promotion

1. Bind lane construction/release, node-forward/gather, reverse, bundle
   construction/exhaustion, dense request-delta/post-accept, and optimizer/update
   work to analogous stable prelaunch subjects.
2. Remove or subject-bind remaining direct authority-free accelerator launch
   APIs.
3. Rebuild and attest the native ABI on an approved clean host.
4. Run the focused source tests, CPU updater integration, and fault injection.
5. Run MPS/CUDA parity plus allocator/RSS evidence in a guarded quiet window.
6. Only then open accelerator capability minting or claim native memory parity.

The sample slice is source-complete and statically audited. The full accelerator
and dense-path promotion is not complete.
