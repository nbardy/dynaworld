# WorldFoam bounded block-streamed fail-atomic transaction

Date: 2026-08-05 (KST)

Status: source-audited design and proof sketch only. No Python process, import,
test, build, Metal/MPS launch, CUDA launch, allocator measurement, or training
run was started on the incident-affected Mac.

## Context

The fixed-camera fused-v1 reverse removes the staged `[J_b,W_b]` physical-
length cotangent, but its one-shot transaction retains every prepared active
block and every compact material output until one final fence. Union-local
fused geometry v2 removes the global `S`-row geometry output, but by itself it
does not remove that all-block overlap.

The current v1 transaction is intentionally conservative:

```text
prepare every block and every output
-> validate every block
-> accumulate every block
-> finalize every block and shared global ledgers
-> one fence/status read
-> accept disposable transaction outputs
-> only later commit to persistent optimizer state.
```

The question in this note is narrower than a new renderer:

> Can request-level fail atomicity be preserved while only `q` prepared
> reverse blocks are live, with `q=1` as the minimum-memory case?

Current belief: yes, because fail atomicity is required at the persistent
state boundary, not inside fresh transaction-local scratch. Confidence is
medium until the native phase/status semantics and allocator lifetime are run.

## Evidence inspected

- `kinetic_native_equal_rank_runtime_adapter.py` owns the current all-block
  prepared-token tuple, all node bars, all compact material bars, global
  geometry ledgers, and one sticky status until a final fence.
- On accepted completion it clears prepared/node/output references from the
  mutable transaction state and returns only the accepted output receipt.
- `kinetic_dense_cached_native_material_request.py` already owns a fresh
  request-local `U x 4` material ledger and performs no persistent step commit
  until a sealed request delta is consumed.
- `WORLD_FOAM_UNION_LOCAL_FUSED_GEOMETRY_V2_DESIGN_2026-08-05.md` proves the
  exact geometry factorization `P_b = P_U Q_b` and requires a fresh `U`-row
  geometry ledger.

These are source facts, not runtime evidence.

## Symbols

For active blocks `b=1,...,B`, define:

```text
B       exact active block count in canonical order,
q       maximum prepared blocks retained in one transaction batch,
A_b     source-visible logical bytes unique to prepared reverse block b,
N_b     active node-chart/node-bar/loss state for b,
s_b     compact site count of b,
C_b     16 s_b bytes for one compact float32 RGBA cotangent,
U       request-union site count,
C_w     weight-coefficient count,
M_U     16 U bytes for the request-union float32 material ledger,
G_U     4 U (6+C_w) bytes for the request-union float32 geometry ledger,
D_U     8 U (6+C_w) bytes for the accepted CPU-float64 geometry bridge,
Z       four-byte sticky status plus bounded scalar metadata.
```

`A_b` deliberately excludes tensors already charged in `N_b` or immutable
lane residency. It must be measured from the actual token; this note does not
pretend its value is known from Python references alone.

## Existing all-block union-v2 live-set shape

Before its sole transaction fence, an all-block union-v2 candidate has the
source-visible shape

```text
M_all_pre
  = sum_b N_b
  + sum_b (A_b + C_b)
  + M_U + G_U + Z
  + other already-declared lane/request roots.                 (1)
```

After successful acceptance, prepared blocks can be released before the CPU
geometry bridge, leaving a separate bridge phase approximately shaped as

```text
M_all_bridge
  = accepted device outputs + D_U
  + step/request destinations.                                 (2)
```

The whole-request peak is the maximum of phases, not their sum. Neither (1)
nor (2) is an allocator/RSS formula.

## Proposed bounded-batch transaction

### Phase 0: tensor-free exact-manifest preflight

Before allocating or preparing a reverse block, seal:

```text
canonical active block generation order,
active block count B,
node-bar identities/signatures,
global source-site provenance,
union ids and every compact-to-union mapping digest,
global S and request-union U,
weight width C_w,
per-block and shared byte budgets,
backend/stream/fence capability identity.
```

This preflight may retain `O(B)` scalar identities/digests because `B` is
structural program complexity, not requested-frame density. It must not retain
an `F`-axis table or prepare all native tokens.

### Phase 1: allocate fresh request-local outputs

Allocate exactly once:

```text
H_material in R^(U x 4), initialized to zero,
H_geometry in R^(U x (6+C_w)), initialized to zero,
one sticky validation/finiteness reason mask.
```

These outputs are disposable transaction scratch. They are not the persistent
optimizer bars and cannot authorize an update.

### Phase 2: process canonical batches

For consecutive batches `I_k` with `|I_k| <= q`:

```text
prepare only blocks in I_k
-> bind them to their preflight manifest entries
-> validate their immutable source ids, compact-to-union maps, node bars,
   prepared ABI, and current shared ledgers
-> status-guarded ordered reverse
-> status-guarded compact-material scatter and geometry accumulation into the
   shared U-row ledgers
-> postwrite-finalize the touched outputs and shared ledgers
-> one completion fence while all I_k roots remain owned
-> read/revalidate the sticky status and callback snapshot
-> release I_k prepared/output predecessor roots immediately.
```

The status is cleared once for the whole transaction and remains sticky. A
nonzero status rejects the entire transaction even if earlier batches were
successfully fenced.

### Phase 3: accept only exact complete coverage

After the last successful batch, require:

```text
settled block count = B,
settled block manifest digest = preflight digest,
no duplicate/omitted/reordered generation,
sticky reason mask = 0,
every output finite,
no outstanding prepared batch,
exactly `nu` fences per admitted batch, where `nu=1` requires the
status-gated native material scatter above and `nu=2` is the safe fallback
using only the current raw primitives.
```

Only then may the U-row outputs be bridged/sealed into one request delta. The
existing later request-delta commit remains the sole persistent write.

## Exact cotangent proof

Let `g_b` be the exact compact geometry cotangent, `Q_b` compact-to-union, and
`P_U` union-to-global. Define the batch recurrence

```text
H_0 = 0,
H_k = H_(k-1) + sum_(b in I_k) Q_b g_b.                       (3)
```

Induction over canonical batches gives

```text
H_K = sum_(k=1)^K sum_(b in I_k) Q_b g_b
    = sum_(b=1)^B Q_b g_b.                                   (4)
```

The final global cotangent is therefore

```text
P_U H_K
  = sum_b P_U Q_b g_b
  = sum_b P_b g_b,                                           (5)
```

which is exactly fused-v1/staged geometry over real arithmetic. The same
argument applies to material cotangents with the `U x 4` ledger.

Fences do not change the mathematical order. Float32 atomics and different
kernel parenthesization can change rounding, so native parity is tolerance-
based rather than a bitwise theorem.

## Fail-atomicity proof obligation

Let `X` be persistent optimizer/step state and `H` transaction scratch. The
only required state transition is

```text
(X,H=0) -> (X,H_partial) -> either (X,discard) or (X,accepted H)
                              -> commit(X,H).                 (6)
```

No intermediate batch writes `X`. Consequently a later invalid block can
leave `H` partially changed without violating optimizer fail atomicity: `H` is
discarded and no authorization receipt exists.

This weakens one current v1 diagnostic:

```text
all_blocks_validated_before_first_write = false
```

and replaces it with the stronger operational invariant:

```text
no_persistent_write_before_full_manifest_acceptance = true.  (7)
```

Treating partial disposable scratch as forbidden would force all-block
retention without protecting any additional persistent state.

## Source-visible memory bound

For `q=1`, the prepared/output part of (1) becomes

```text
M_stream_pre
  = sum_b N_b
  + max_b(A_b + C_b)
  + M_U + G_U + Z
  + other declared lane/request roots.                        (8)
```

Thus the exact symbolic reduction relative to all-block preparation is

```text
Delta_stream
  = sum_b(A_b + C_b) - max_b(A_b + C_b).                      (9)
```

For general queue depth `q`, replace the maximum with the maximum sum of any
admitted live batch:

```text
M_q_pre
  = sum_b N_b
  + max_k sum_(b in I_k)(A_b + C_b)
  + M_U + G_U + Z + ...                                      (10)
```

`q` is therefore a memory/launch-latency policy, not model semantics. Start at
`q=1`; promote a larger bounded `q` only from measured allocator and wall-time
evidence.

If the v2 kernel writes both material and geometry directly to union rows,
`C_b` disappears from (8). If it still produces one compact material bar,
that bar must remain owned through the batch fence and be scattered into the
fresh union material ledger before release. Finiteness of the compact bar is
not sufficient: two finite compact contributions can overflow when
`index_add_` combines them at one union destination. The one-fence batch
acceptance sequence must therefore be

```text
finalize compact bar
-> status-gated compact-to-union scatter into fresh H_material
-> finalize H_material together with the shared union geometry ledgers
-> completion fence/status acceptance
-> release compact bar.
```

The scatter itself must read the same sticky status. An unconditional PyTorch
`index_add_` before the first status read is unsafe: if device validation
rejects a tampered compact-to-union map, raw fused writes are gated but the
unconditional scatter could still consume the invalid map. The current raw
union-v2 wrapper also shape-locks its first finalizer input to `[s_b,4]`.
Consequently a one-fence bounded-batch adapter needs both a status-gated
compact-material-to-union scatter and a suffixed shared-output finalizer for
`[U,4]`.

Using only the current primitives, the safe fallback is two fences per batch:

```text
raw validate/accumulate/finalize
-> fence and read zero status
-> compact-to-union index_add_ into fresh H_material
-> independently finalize/check H_material
-> second fence and accept
-> release compact bar.
```

This preserves the `q=1` memory theorem and request fail atomicity but doubles
the synchronization term. It must be costed honestly rather than described as
the one-fence design.

These equations exclude allocator bins, Metal private/register/spill storage,
command buffers, Python objects, decoder state, and process RSS.

## Time-scaling consequence

The minimum-memory route pays `nu ceil(B/q)` transaction fences rather than
one, with `nu=1` only for the status-gated native-scatter design and `nu=2`
for the safe current-primitive fallback.
For a fixed certified atlas, `B` is structural block complexity and does not
grow merely because requested frame density `F_requested` changes. Therefore
the extra synchronization does not reintroduce per-frame ordered-word reverse,
although it can materially hurt wall time.

This remains separate from unavoidable sampled work and output:

```text
world/ordered reverse:  O(sum_b J_b R_b), independent of F_requested
sample/output slice:    Omega(P F_requested)
fence count:            nu ceil(B/q), not F_requested.        (11)
```

If longer physical duration creates more charts/blocks, `B` can grow. No
duration-independent claim follows.

## Failure matrix

| Failure | Completion knowledge | Required action |
| --- | --- | --- |
| Preflight rejection | no batch enqueue | reject without quarantine |
| Prepare raises before any possible enqueue | no new batch work | discard fresh transaction scratch after settling prior batches |
| Prepare/launch raises after possible enqueue | uncertain until canonical batch fence | retain current batch/shared scratch; fence once |
| Batch fence raises or violates contract | unknown | quarantine current batch, shared outputs, status, session/lane roots; no retry; restart |
| Fence succeeds, status nonzero | complete and rejected | release current batch; discard all transaction scratch; no persistent commit |
| Fence succeeds, callback/provenance mutation found | complete but invalid | release completion-safe batch roots; poison/reject transaction; no second batch fence |
| All batches accepted, bridge fails after enqueue | governed by request bridge quarantine | retain accepted union outputs and bridge roots until its completion proof |
| Persistent request-delta commit fails | governed by existing step quarantine | never replay the consumed/unknown commit |

An ordinary exception must never cause a `finally` block to clear roots after
an unknown fence result.

## Branches and backtracks

### Branch A: all blocks must validate before any scratch write

Status: weakened.

That rule is useful when outputs alias persistent state. Here outputs are fresh
and disposable. Exact preflight plus per-batch validation protects provenance;
full-manifest acceptance protects the persistent boundary.

Could be wrong if the native kernel has an undisclosed alias from transaction
scratch into persistent model/optimizer storage. Hidden-alias absence must be
proved or conservatively enforced before promotion.

### Branch B: retain all prepared blocks but only shrink geometry output

Status: valid intermediate, not the minimum-memory endpoint.

Union-local geometry v2 gives an exact `S -> U` saving and is simpler to land.
It should remain an oracle/intermediate route. If measured prepared-token plus
compact-output overlap is small, streaming may not repay its fences.

### Branch C: directly write material and geometry to one union namespace

Status: promising v3 simplification.

It removes per-block compact output ledgers, but requires the kernel to keep
global source provenance and union write destinations distinct. Cross-block
duplicate union destinations must atomically accumulate. This is not required
for the first union-geometry v2.

### Branch D: per-block persistent commit

Status: rejected.

It lowers scratch but destroys request-level all-or-nothing admission. A later
block failure would leave optimizer bars partially changed.

### Branch E: queue depth greater than one

Status: deferred policy.

A small `q` may recover throughput while maintaining a hard byte cap. It must
be selected from measured token bytes and retain every live batch until the
oldest canonical fence. Never infer safety from Python reference count.

## Falsification gates

Reject bounded streaming if any of these fail:

1. `q=1`, `U=S` identity parity against fused v1 and staged sparse for
   forward, loss, material, position, velocity, and weight gradients.
2. `U<S` parity with duplicate global sites across blocks.
3. A later invalid block leaves persistent step/optimizer bars unchanged.
4. Prepare/launch/fence failure retains the current batch and shared roots;
   unknown completion cannot retry or release.
5. Exact active-manifest coverage rejects omission, duplication, and reorder.
6. Compact-to-union mismatch is rejected before the corresponding write.
7. NaN, positive infinity, negative infinity, and finite-add overflow in any
   batch make the sticky status reject the whole transaction.
8. Measured maximum prepared-token count is `<=q` and zero after acceptance.
9. Complete fresh-process allocator/RSS peak improves over all-block union v2;
   logical source bytes alone are insufficient.
10. Wall-time regression from `nu ceil(B/q)` fences is reported. Promote only if
    memory improves materially without making the paper trainer impractical.
11. Repeating `F_requested=8,64,300` on one fixed atlas leaves `B`, reverse
    interactions, prepared-byte cap, and fence count invariant.

## Decision

Land and measure union-local geometry v2 first because it is exact and changes
one output index space. Preserve fused v1 as the oracle. If prepared blocks or
compact outputs dominate the complete request peak, implement the bounded-
batch transaction with `q=1` and fresh union scratch. Do not attempt per-block
persistent commits.

The key systems conclusion is:

```text
request-level fail atomicity does not require all request scratch to remain
unmodified until every future block has validated; it requires all scratch to
remain private and disposable until exact full-manifest acceptance.
```

That observation gives a plausible path from frame-independent memory to a
small absolute peak without changing WorldFoam's ordered-depth mathematics.
