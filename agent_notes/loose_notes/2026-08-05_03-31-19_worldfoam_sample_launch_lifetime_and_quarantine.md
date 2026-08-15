# WorldFoam sample-launch lifetime and quarantine

Date: 2026-08-05 (KST)

Status: source written and statically inspected only. No Python process,
import, test, build, Metal/MPS launch, CUDA launch, or training run was started
on the incident-affected Mac.

## Why this work existed

The stratified/Lagrangian connection audit and union-local fused-geometry
design made the desired time-sharing theorem precise, but the native sample
reducer still had an ownership hole. `launch_sample_accumulate` prepared an
opaque native payload, enqueued work, returned `None`, and let its local
`prepared` reference die before the caller's completion fence. The production
prepared object owns the MPS sample-row copy and the two small config tensors,
so this was a real asynchronous-lifetime defect, not documentation polish.

The dense path had three adjacent versions of the same error:

- the CPU source of a CPU-to-device target transfer was explicitly deleted
  before the comment's promised first fence;
- CPU float64 interpolation weights and device sample positions were dropped
  after materialization commands were enqueued;
- an active reverse block was removed from the active dictionary before its
  native reverse, union scatter, loss add, and block-commit fence completed.

## Assumptions

1. A Python return from a native launch is an enqueue proof, not a device
   completion proof.
2. A successful caller-supplied device completion fence covers every command
   enqueued before it on the admitted device/backend.
3. If that fence raises or returns the wrong type, completion is unknown. It
   is not safe to guess that it either did or did not synchronize.
4. One bounded sample has at most `K` observations. The native preparer adds
   the already-accounted `4K + 20` logical bytes; a lifetime carrier only owns
   aliases and adds zero logical tensor bytes.
5. Python-object/allocator/RSS overhead is still unmeasured and must not be
   reported as zero physical memory.

## Selected lifecycle

For a session `s`, let `L(s)` be its outstanding sample-lifetime count. The
source contract is

```text
L(s) in {0, 1}

launch:
  L = 0
  -> install provisional roots before native prepare
  -> attach opaque prepared payload immediately after prepare
  -> enqueue exactly one native launch
  -> return the same session-owned lifetime with L = 1

settle:
  L = 1
  -> invoke exactly one completion fence while all roots remain owned
  -> revalidate session/prepared/read-only/output bindings
  -> commit sample coverage and manifest
  -> clear every lifetime root immediately
  -> L = 0
```

On every sealable block and session,

```text
native prepare count
= native sample launch count
= native sample completion-fence count.
```

The session retains no lifetime list, receipt list, flat-index tuple, target
history, or sample history. The lifetime records only count/first/last and a
digest for flat sample identities. Therefore peak sample-lifetime retention is
`O(K)` and lifetime history is `O(1)`.

## Why the fence is inside settlement

A method accepting `fenced=True` or a caller-authored receipt cannot prove a
fence happened while the prepared payload was alive. Settlement therefore
accepts the fence callback and calls it itself. It snapshots the session
ledger before the callback, rejects reentry during the callback, and compares
the post-callback ledger before committing coverage.

Counters, the sample manifest, streamed observation count, and first/last
sample IDs remain pending until settlement. A launch that was merely enqueued
never becomes accepted paper telemetry.

## Failure matrix

| Failure point | Completion knowledge | Retention/release rule |
|---|---|---|
| Before native prepare | no native sample enqueue | provisional inputs stay session-owned until abort cleanup |
| Native prepare or launch raises | possibly enqueued | prepared/input roots stay owned; abort must fence before release |
| Completion fence raises or returns non-`None` | unknown | retain everything, prohibit retry/abort release, require restart |
| Fence succeeds, then signature/provenance check fails | complete | release sample roots immediately, poison session, do not issue a second sample fence |
| Successful settlement | complete | commit metadata, return tensor-free receipt, release all sample roots |
| Reverse block commit fence fails | unknown at request layer | keep block in `active` plus explicit reverse scratch/results; quarantine if abort fence also fails |

This intentionally follows the existing fused-transaction rule: once the
canonical fence itself fails, a second fence is not treated as evidence. That
is stricter than assuming synchronization is idempotent, but it avoids
releasing a payload on an unproven timeline.

## Dense ownership changes

- `PaperKineticDenseChunkTargets` now seals the CPU transfer source into the
  bounded chunk target object. The request holds that object through every
  sample fence in the chunk. This realizes memory already present in the
  target bridge preflight rather than raising the cap.
- The lower-level request no longer accepts an arbitrary target-loader
  callable. `PaperKineticDenseChunkTargetLoader` is sealed to one source,
  request, device, target generation, and the three decode/chunk/bridge caps.
  It installs one `PaperKineticDenseChunkTargetLoadLifetime` before provider
  work, publishes the selected-read/CPU roots before `Tensor.to`, and retains
  each returned transfer/contiguous tensor. Success releases the carrier only
  after the chunk's sample-completion fence; a post-enqueue exception is
  released only after abort fencing or retained in restart-required
  quarantine if that fence fails. The exact source-test fault is a sealed
  finite-stage value, not an arbitrary callback. This closes the earlier
  ownership hole at source level without claiming allocator measurement.
- Dense sample materialization returns an internal lease owning the CPU-f64
  weight source, device positions, chunk-target source, and sealed sample
  block. The caller releases it only after executor settlement succeeds.
- The reverse loop uses `active.get`, not `active.pop`, before reverse work.
  It deletes the block only after the block-commit fence and exposes bounded
  current reverse scratch/results to failure quarantine.
- Failure quarantine explicitly retains current chunk targets, their CPU
  source, sample materialization, sample block, native lifetime, session's
  authoritative outstanding lifetime, and one current reverse block/scratch
  chain. Its validator calls the lifetime/lease retention assertions rather
  than hashing only object IDs.

The resulting bounded live-set shape is

```text
O(one target chunk)
+ O(one K-sample materialization/native launch)
+ O(active block node/loss state)
+ O(one reverse block scratch/result chain),
```

with no frame history and no sample-token history.

## Branches considered and backtracked

### Keep `prepared` in a caller local

Rejected. The old bug occurs precisely because the executor returns before
the caller fences; a local in the executor cannot cross that return.

### Return `prepared` directly

Rejected. It would expose the native ABI and still would not bind the sample,
world, outputs, manifest, and exact session generation into one lifecycle.

### Retain every launch token until reverse

Rejected. This would make host/object retention linear in sample-launch count
and obscure the intended one-launch backpressure theorem.

### Allow a failed completion fence to be retried by abort

Rejected for the production contract. The fence may have failed before or
after synchronization; treating a second callback as canonical proof would
silently choose one interpretation. Completion-unknown now means quarantine
and restart.

### Drop CPU transfer/materialization predecessors because copies requested
`non_blocking=False`

Rejected. A requested transfer mode is not a backend completion receipt, and
the surrounding design already requires a device fence before releasing
K-local inputs.

## Falsification targets

Source-written tests (unrun) are intended to falsify the design if any of these
occur:

- a weak reference to the native prepared payload dies before the settlement
  callback runs;
- a second launch/reverse/seal enters while one lifetime is outstanding;
- settlement calls a fence more than once;
- a failed fence permits abort release or loses its roots;
- successful settlement leaves tensor/sample references on the lifetime;
- telemetry does not prove prepare = launch = completion fence;
- repeated successful settlements retain a token history or exceed one
  simultaneous lifetime;
- dense failure quarantine loses the current transfer/materialization/reverse
  roots.

The dense focused suite now reaches an actual fake-native prepare and launch,
injects failure in the sample-settlement fence, proves abort does not issue a
second fence after completion becomes unknown, and inspects the quarantined CPU
target source, materialization lease, sample block, executor lifetime, session
authority, and prepared payload. Its happy path checks exact
prepare/launch/settle equality, maximum lifetime one, zero retained history,
and zero additional logical tensor bytes. The lazy focused expectations were
also migrated from the obsolete one-fence-per-bundle model to one fence per
sample plus one lane-release fence per bundle. These are source assertions,
not executed evidence.

## Remaining blockers

1. The older lazy material-step caller now settles each sample, but its failure
   path still lacks a durable completion-unknown quarantine. It is therefore
   source-gated to CPU contract execution and explicitly rejects accelerator
   devices until that quarantine (including reverse/lane roots) is closed.
2. Runtime syntax/import, CPU fake-native (including the written target-loader
   post-enqueue and sample-settlement failure injections), native Metal/MPS,
   allocator, and full training gates remain unrun by deliberate host policy.
3. The native backend itself remains source/runtime-unverified; these changes
   establish an ownership theorem, not measured device behavior.

## Current interpretation

This is not a new mathematical representation. It is the systems lemma needed
for the existing WorldFoam formulation to realize its intended asymptotics:
the expensive ordered-word reverse remains once per active spatial block,
while the linear observation slice is bounded, cheap, and retired one sample
launch at a time. The scientist's stratified/Lagrangian formulation supplies
the correct event/order geometry; this lifetime work prevents the native
implementation from invalidating the memory theorem at the asynchronous
boundary.
