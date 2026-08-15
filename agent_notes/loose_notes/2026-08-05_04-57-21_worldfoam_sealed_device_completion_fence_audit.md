# WorldFoam sealed device-completion fence audit

## Context

The lazy kinetic WorldFoam route has now gained explicit lifetimes around the
top-level device transaction, pre-lane union construction, lane construction,
compact material gathering, forward-into output, sparse sample transfer, and
reverse scratch.  Those lifetimes are useful only if their release is backed
by a completion operation that actually covers the producer backend and
device.

Current source still accepts:

```text
device_completion_fence: Callable[[], None]
device_completion_fence_provenance: str
```

The callback identity and provenance string are hashed into the step, but this
proves only that the same Python object/string was reused.  It does not prove
that the callback synchronized the device or dispatch domain that produced the
live tensors.  A callback can return `None` after synchronizing nothing, the
wrong CUDA device, or a sibling stream.  Callback-state snapshots cannot
repair that semantic hole.

Source inspected:

- `research_experiments/world_foam_lane2/kinetic_lazy_native_material_step.py`
- `research_experiments/world_foam_lane2/kinetic_native_material_step_executor.py`
- `research_experiments/world_foam_lane2/kinetic_native_lazy_bundle_lane.py`
- `src/train/paper_kinetic_lazy_program_bundles.py`
- `src/train/paper_kinetic_sparse_sample_blocks.py`
- `src/train/paper_kinetic_union_local_bar_assembly.py`
- the fused-slab Metal bridge, which dispatches through PyTorch's
  `DynamicMetalShaderLibrary` / `MetalKernelFunction`

No Python, import, test, build, Metal, MPS, CUDA, or training command was run.

## Current model

The smallest safe interface is not a better callback.  It is one exact-type,
module-minted capability that contains no injected synchronization function.
It is bound to:

```text
(native_ops identity,
 native ABI identities,
 backend provenance,
 normalized device,
 completion scope,
 owner generation digest,
 creating thread)
```

Its backend operation is selected in its own source:

```text
CPU   -> call return is completion (synchronous producers only)
MPS   -> torch.mps.synchronize()                 # all MPS work
CUDA  -> torch.cuda.synchronize(bound_device)    # all streams on device d
```

A device-wide fence is deliberately stronger than a stream-local fence.  The
completion domain is the complete set of streams on the bound device, so a
caller cannot accidentally fence stream `s1` while the native producer used
`s0`.  CUDA requires an explicit device index.  MPS is normalized to `mps:0`.

The isolated source contract is:

```text
research_experiments/world_foam_lane2/kinetic_sealed_completion_fence.py
research_experiments/world_foam_lane2/test_kinetic_sealed_completion_fence.py
```

Accelerator minting remains hard closed.  The MPS/CUDA implementations exist,
but the public factory refuses those devices until the canonical native module
has a verified launch-domain/build attestation and safe-host runtime evidence.
There is intentionally no `allow_unverified=True`, callback injection, or
caller-declared stream name.

Confidence: high that this closes callback-semantic forgery at the Python
boundary; low that accelerator release is ready, because the module is not yet
integrated and no native evidence was run.  The focused CPU/exact-type/gate
tests are source-written but deliberately unrun under the host-safety rule.

## Exact lifecycle

Let `C` be one capability and `n` its next sequence number.

```text
healthy(C, n)
  -> prebuild and validate unpublished receipt R_n
  -> mark invocation active
  -> attempt exactly one source-selected device-wide synchronize
     -> success: publish R_n, advance n := n + 1
     -> exception: completion_unknown := true, poison C forever,
                   retain error/traceback/stage/generation, publish no receipt
```

Properties:

1. `type(C) is PaperKineticSealedCompletionFence`; subclasses are rejected.
2. `type(R) is PaperKineticCompletionFenceReceipt`; subclasses are rejected.
3. A private seal and SHA-256 generation digest bind immutable fields.
4. ABI identities use the underlying `__func__` for bound methods, avoiding
   unstable temporary bound-method identities.
5. The bound lazy ABI includes the caller-owned-output
   `kinetic_precompiled_length_p0_lie_node_forward_into_launch_only_v1`, not
   the unsafe return-allocating forward oracle.
6. Calls must occur on the creating thread.
7. The invocation lock is nonblocking; concurrent or recursive entry fails
   without a second backend call.
8. There is no receipt history; counters and the sole failure record are O(1).
9. A failed synchronization is never retried.  The capability must be retained
   inside the trainer's sole restart-required quarantine.
10. A receipt is fully built before synchronization and remains unpublished
   until success, so receipt construction cannot fail after known completion.
11. The receipt states `completion_domain_drained`; it does not falsely call
    CPU call-return semantics a device-wide fence.
12. Provenance is derived from the capability.  The caller no longer supplies
    a completion-provenance string.

## Every lazy-route completion seam

### S0: top-level output allocation and initial global-bar zero

Current source installs `_TopLevelDeviceTransactionLifetime` before the first
step-local device allocation.  This closes root visibility, but release still
depends on the arbitrary callback later in the step.

Required integration:

1. Compute a step-owner SHA-256 digest from provider generation, step index,
   material/background generations, immutable tensor signatures, native ABI,
   device, and backend provenance.
2. Mint `PaperKineticSealedCompletionFence` inside the coordinator before the
   first device allocation.
3. Retain it on the top-level transaction and trainer quarantine.
4. Include its immutable generation digest, not `id(callback)` or a caller
   string, in `step_generation_id`.

Failure cleanup is a separate launch epoch.  If cleanup performs
`global_bar.zero_()` or `loss.zero_()` on an accelerator, it must receive its
own final receipt before the trainer clears active state or reuses storage.
The safe failure sequence is:

```text
fence failed work
  -> if unknown: quarantine, enqueue no cleanup
  -> if known: enqueue cleanup zeros
  -> fence cleanup zeros
  -> if unknown: quarantine
  -> if known: release transaction and fail the logical step cleanly
```

Merely enqueueing cleanup zeros and returning is not a completed cleanup.

### S1: cold union-local bundle construction

The bundle lifetime slot roots CPU sources, raw transfer results, and the
partially materialized bundle.  Replace the direct callback in the outer
exception path with:

```text
receipt = capability.fence(
    stage="bundle-construction-abort",
    launch_generation_digest=bundle_lifetime_generation,
)
slot.release_active_after_completion_fence(receipt)
```

The release method should require the exact receipt and verify capability,
owner, device, sequence, and launch-generation binding.  A no-argument method
named `release_after_completion_fence` is only a convention, not proof.

### S2: native lane/runtime construction

Runtime `.to(...).contiguous()` predecessors are now rooted by construction
lifetimes.  Partial materialization must use the same capability.  A foreign
capability cannot be supplied to the lane; it must match `native_ops`, device,
backend provenance, and the step owner.

### S3: sparse target/weight transfer

`PaperKineticSparseSampleMaterializationLifetime` roots the raw weight and
target transfers before contiguity.  Its release currently follows the sample
completion callback.  Bind its active generation digest into that receipt and
make `release_active_after_completion_fence(sample_block, receipt)` validate
it.  The sample/native launch fence is device-wide and therefore also covers
the earlier transfer on the same device.

### S4: compact material gather and forward-into

The current source now publishes an external `node_chart_out` into
`KineticNativeNodeForwardIntoLifetime` before enqueue.  This is the right
ownership shape.  Its first later sample receipt covers the gather, contiguity,
forward, and sample launch.  If forward/sample preparation fails before that
receipt, the session-abort receipt must cover them before any forward lifetime
is retired.

### S5: each bounded sample launch

Change executor settlement to:

```text
settle_sample_accumulate(
    lifetime,
    *,
    completion_fence: PaperKineticSealedCompletionFence,
    launch_generation_digest: str,
) -> KineticNativeSampleLaunchCompletionReceipt
```

The executor must use an exact type check, call `completion_fence.fence(...)`,
and embed the capability digest, fence sequence, scope, normalized device, and
sealed receipt digest in its own completion receipt.  Remove both callable and
free-form provenance parameters.

One sample fence per launch is expensive but currently enforces the `q=1`
sample-memory bound.  Future batching may settle `q>1` launches with one
device-wide receipt only when the explicit memory policy accounts for all `q`
live sample lifetimes.

### S6: each reverse scratch

Each material VJP uses one compact bar and currently fences before deleting
its active block.  Keep that order, but bind the stage to the reverse result,
runtime generation, compact-bar signature, and global-bar generation.  The
exact receipt authorizes retirement of the compact bar, forward lifetime, and
block state.

### S7: successful lane release

The current successful lane executes one additional device-wide lane-release
fence after every reverse has already been fenced.  Static source shows no
device enqueue between the last reverse receipt and lane retirement; session
sealing and telemetry construction are host ledger work.  Therefore this
extra fence appears redundant.

Promotion choice:

- Preferred: prove no post-reverse device enqueue and let the final reverse
  receipt authorize lane/bundle retirement.
- Conservative fallback: keep the lane-release receipt until a safe-host
  trace confirms the proof.

Do not silently remove it solely for timing.  First encode the no-post-reverse
enqueue invariant in the executor receipt and a behavioral failure test.

### S8: session abort and failed lane release

`session.abort(device_completion_fence=lambda: ...)` wraps one arbitrary
callback in another.  Replace it with the exact capability and a launch
generation.  The session must preserve the current one-fence rules:

- if sample settlement already completed but later validation rejected, do
  not fence again;
- if a fused transaction already returned a known-settled rejection, do not
  fence again;
- if completion is unknown, never call abort again;
- otherwise consume exactly one new capability sequence.

The executor should store capability generation and last accepted sequence,
not just a provenance string and count.

### S9: target-cache/bundle/forward root release

Every method named `release_*_after_completion_fence` or
`retire_after_completion_fence` should eventually require a sealed receipt.
The receipt need not be retained after release; validate then drop it to keep
O(1) metadata.  Until these APIs consume receipts, the capability closes the
fence-call seam but not the entire release-authorization seam.

### S10: optimizer authorization

The optimizer may run only after:

```text
all sample receipts accepted
all reverse receipts accepted
all bundle construction/transfer roots safely retired
top-level transaction completion known
capability healthy and successful_count == expected sequence count
```

If an optimizer callback starts and fails, poisoning remains correct because
external parameter mutation may already have occurred.  Completion receipts
cannot make an optimizer fail-atomic.

## Integration patch plan

1. Import the isolated capability in the lazy coordinator.
2. Remove `device_completion_fence` and
   `device_completion_fence_provenance` from the public step signature.
3. Mint the capability internally before the first device operation.
4. Pass the exact capability through `_execute_native_bundle` and executor
   settlement/abort; never pass `capability.fence` as a callback.
5. Replace every direct callback invocation with a named stage and exact launch
   generation digest.
6. Replace accounting provenance with:

   ```text
   completion_capability_generation_digest
   completion_scope
   normalized_completion_device
   successful_completion_fence_count
   final_completion_fence_sequence
   accelerator_completion_runtime_attested=false
   ```

7. Store the capability itself in `_LazyAsyncFailureQuarantine`; validate it
   with `require_healthy=False` and require `completion_unknown=True`.
8. Update sample/executor receipts to nest the sealed capability receipt.
9. Then update release/retire APIs to consume and validate receipts instead of
   trusting method names.
10. Preserve the non-CPU gate until native attestation and runtime evidence are
    recorded.  Source implementation alone is not promotion.

## Required accelerator attestation

### MPS

The canonical fused-slab extension must attest that the selected launch-only
operators dispatch through PyTorch MPS tensor/Metal infrastructure on `mps:0`.
The inspected source uses `DynamicMetalShaderLibrary` and
`MetalKernelFunction.dispatch`, which is encouraging but not runtime proof.

Required safe-host evidence:

1. enqueue each selected operator;
2. settle through the capability-owned `torch.mps.synchronize()`;
3. verify output and status visibility;
4. inject pre-enqueue, post-enqueue, and fence failures;
5. show one bounded quarantine and no retry after unknown completion;
6. measure allocator/RSS release only after the receipt;
7. verify no callback can substitute a no-op fence.

### CUDA

CUDA requires a real canonical native ops build first.  The capability binds an
explicit `cuda:d` and calls `torch.cuda.synchronize(cuda:d)`, which covers all
streams on that device.  Required evidence additionally includes two CUDA
devices or an explicit wrong-device negative fixture, plus a non-default-stream
producer fixture showing that device-wide completion covers it.

## Falsification tests

The design is wrong if any of these succeeds:

1. Constructing a capability subclass and passing exact-type validation.
2. Mutating a required native op after capability construction without
   invalidating the capability.
3. Invoking from another thread or re-entering during one fence.
4. Retrying after one synchronization exception.
5. Minting MPS/CUDA from a caller boolean, callback, or free-form provenance.
6. Releasing a sample, forward, lane, bundle, or top-level lifetime without a
   receipt bound to its launch generation.
7. Authorizing the optimizer after fewer receipts than launch epochs.
8. Synchronizing `cuda:1` for work bound to `cuda:0`.
9. Treating a failed device synchronize as known completion.
10. Claiming accelerator runtime verification from CPU/source tests.

## Open questions

1. Can the successful lane-release fence be eliminated using the final reverse
   receipt, or does a hidden tensor signature/read enqueue work on a backend?
2. Should the top-level cleanup perform two fences (failed work then cleanup),
   or should a known-failed step retain dirty global bars until an explicit
   reset transaction?
3. What exact native build identifier should the future promotion table bind:
   extension binary hash, selected-op resource attestation digest, or both?
4. Should all lower release APIs consume the full receipt or a smaller
   one-shot release authorization derived from it?  The latter reduces coupling
   but adds another capability type.
5. Can `q>1` sample and reverse settlement materially reduce device-wide sync
   overhead without breaking the explicit peak-memory budget?

## Decision

Keep the new capability module and integrate it before enabling accelerators.
Do not promote MPS or CUDA merely because device-wide synchronization code is
present.  The immediate source blocker is no longer discovering a fence API;
it is replacing every callback/release convention with the exact capability
and receipt, then producing native failure and allocator evidence on a safe
host.
