# WorldFoam fused-union-v2 raw native slice

Date: 2026-08-05

Status: source-written and statically inspected only. No Python import, test,
build, Metal compilation, MPS launch, CUDA launch, or training was run because
the host remains under the incident-safe no-runtime policy.

## Context

The exact union-local design in
`research_notes/worldfoam_paper/WORLD_FOAM_UNION_LOCAL_FUSED_GEOMETRY_V2_DESIGN_2026-08-05.md`
factors the fused-v1 geometry scatter through the request union. The canonical
v1 native implementation was found in
`third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0`, not in
the older `world_foam_lane2_fused_direct_v0` fork. Those variant files already
contained extensive uncommitted work before this slice and were preserved.

## Current model

For block `b`, with compact geometry cotangent `g_b`, global-site scatter
`P_b`, compact-to-union scatter `Q_b`, and union-to-global scatter `P_U`, the
cold mapping certificate gives

```text
P_b = P_U Q_b.
```

Thus one transaction-local union ledger

```text
H = sum_b Q_b g_b in R^(U x (6+C))
```

commits exactly to the old global result:

```text
P_U H = sum_b P_U Q_b g_b = sum_b P_b g_b.
```

The new native slice changes only the output index space. It preserves v1's
ordered optical-product reverse, kinetic cut derivative, material ledger,
continuous-owner inputs, and fixed-camera limitation.

Confidence: medium at source level, zero for native runtime until rebuild and
parity. The algebra is exact; source wiring can still contain a compiler or ABI
mistake because nothing was executed.

## Three identities

The raw token and native ABI keep these as separate int64 tensors:

```text
source_site_ids_i64
    compact k -> immutable global site s;

compact_to_geometry_output_i64
    compact k -> request-union destination u;

geometry_output_source_site_ids_i64
    request-union row u -> immutable global site s.
```

Cold Python admission and device prevalidation require

```text
geometry_output_source_site_ids_i64[
    compact_to_geometry_output_i64[k]
] == source_site_ids_i64[k].
```

The union global ids are sorted, unique, and in `[0,S)`. Compact destinations
are in `[0,U)`. All three identity tensors must be storage-distinct, including
the `U=S` identity-map case, because an aliased tensor would erase the ABI's
semantic distinction and weaken mutation/provenance checks.

When either cold map starts on CPU, the raw preparer copies it to MPS but also
retains the original CPU tensor and its mutation signature as an asynchronous
transfer predecessor. Its bytes are charged in retained logical accounting.
The raw token exposes no early release: a higher-level adapter may release that
predecessor only after a proven copy/launch completion fence. This closes the
immediate use-after-free hole but is intentionally non-promotable lifetime
behavior until the sealed transaction adapter owns the fence.

## ABI and buffer audit

The first slice exposes only three suffixed split operators:

```text
kinetic_fused_union_full_vjp_validate_shared_status_launch_only_v2
kinetic_fused_union_full_vjp_accumulate_shared_status_launch_only_v2
kinetic_fused_union_full_vjp_finalize_shared_status_launch_only_v2
```

There is deliberately no combined raw operation and no persistent commit.
The current coordinator policy can validate all admitted blocks, accumulate
all blocks, and finalize all blocks under one status. The split ABI also stays
compatible with a future bounded-block transaction-local coordinator: all
scratch can remain uncommitted until full-manifest acceptance, so all-block
validation-before-first-write is not claimed as the only fail-atomic design.

For validation, buffers 0--25 retain the v1 order. Buffer 26 is the shared-ledger
scan flag, 27 is compact-to-union, 28 is union-to-global, and 29 is `U`.
For accumulation, buffers 0--25 retain v1 order, 26 is compact-to-union, 27 is
union-to-global, and 28 is `U`. Finalization binds four bars, the shared status,
compact count, `U`, coefficient count, and the one-shared-ledger scan flag.

The v2 config is int32[7]:

```text
[row_count, node_count, compact_site_count, word_count, C, S, U].
```

Host constants provide trusted bounds. Device validation checks `config[6]`
against trusted `U` before atomics. The write grid rechecks it and rechecks the
two identities for every adjacent-cut destination before writing.

## Fail-atomic boundary

The validation kernel first calls the existing v1 dry-run reverse validator,
then checks the union factorization and zero/finite compact and union ledgers.
Any reason bit is ORed into the one shared int32 status. The accumulation grid
is ordered after validation on the same stream and reads the completed status
before its first atomic. Invalid preflight input therefore leaves output
scratch untouched.

The finalizer scans compact material bars for every block and scans shared
union geometry bars exactly once. A nonfinite postwrite sum rejects and
quarantines disposable scratch; it cannot roll the atomics back. Raw result
acceptance requires both locally proven accumulation and finalization, so an
isolated validate/finalize receipt cannot authorize bars. A higher-level sealed
transaction token remains necessary for multi-block single-use ownership,
fencing, manifest admission, and quarantine.

## Memory equation

The geometry source/destination bridge changes from

```text
12 S (6+C)
```

to

```text
12 U (6+C),
```

before any optional CPU commit-map charge. For `C=3`, the exact logical saving
is `108(S-U)` bytes. This source slice does not prove allocator or RSS savings.
It also temporarily retains the raw v1 prepared token as the structural oracle,
including its tiny six-int config; promotion can remove redundant config
retention after parity, but doing so before verification would weaken the
oracle boundary for negligible memory benefit.

## Branches and backtracks

### Branch A: all-block transaction coordinator

Current first policy: validate all blocks, accumulate all blocks, finalize all
blocks, fence once, read one status, then construct one request delta.

Support: simplest direct extension of v1 and easiest parity oracle.

Falsifier: prepared-block residency dominates peak even after U-sized outputs.

### Branch B: bounded block batches

Later policy: admit a presealed manifest, process `q` prepared blocks at a
time into the same transaction-local U ledger, fence/release each batch, then
perform one final status/manifest acceptance before persistent commit.

Support: reduces prepared-block peak while preserving request-level atomicity.

Falsifier: repeated fences dominate latency or the platform cannot provide the
required completion/quarantine proof.

### Backtrack: reuse source ids as union destinations

Rejected. Global source ids can be sparse in `[0,S)` and are not destinations
in `[0,U)`. Reinterpreting them would be out-of-bounds when `U<S` and would
erase the exact `P_b=P_UQ_b` certificate.

### Backtrack: expose a combined convenience op first

Rejected. A combined one-block route encourages premature bar acceptance and
does not model request-level multi-block admission. Split phases are the safer
minimal ABI and remain separable for later streaming.

## Falsification gates

1. Rebuild the extension in an approved quiet environment.
2. Run `U=S` identity-map parity against unchanged fused v1.
3. Run `U<S` parity against staged sparse geometry reduction.
4. Exercise cross-block duplicate union destinations.
5. Tamper each map, union ordering, bounds, config[6], and identity equality;
   verify nonzero status and byte-identical zero ledgers.
6. Force finite-contribution overflow; verify finalizer rejection and scratch
   quarantine before any request/optimizer commit.
7. Measure full request peaks and `U/S`; reject v2 if allocator/RSS does not
   improve despite logical tensor savings.
8. Add a sealed runtime transaction adapter binding union generation, tensor
   identities/versions/digests, exact active block order, fence provenance,
   and restart-required quarantine.

## Remaining seams

- The runtime adapter now has an isolated all-block union-v2 transaction, but
  the request executor still does not route it.
- No request-delta `[U,*]` type or CPU union `index_add_` commit is connected.
- The isolated adapter receipt proves exact bundle-manifest coverage and one
  shared union namespace, but no executor/combined-step receipt consumes it.
- The raw prepared token has source-level accounting only; allocator, private
  Metal storage, registers, spills, Python heap, and RSS are unmeasured.
- Native syntax, binding arity, Metal compilation, numerical parity, and
  asynchronous fail-atomic behavior are all unverified.

## Follow-on: sealed all-block runtime adapter

The source-only adapter now has an isolated union-v2 all-block transaction in
`kinetic_native_equal_rank_runtime_adapter.py`. It deliberately reuses each
sealed fused-v1 block as the mathematical/provenance oracle and requires the
existing `PaperKineticUnionLocalSpatialBundle` as its only union authority.
The exact canonical bundle order is the active manifest; the bundle generation,
union tensor identity/signature, per-block mapping generation/signature, and
the equality `union_ids[compact_to_union[k]] == source_ids[k]` are cold-bound.

The raw union tokens must alias the already-resident bundle union and maps:
`mapping_tensor_owned_by_preparer == (False, False)`. This removes an ambiguous
asynchronous map lifetime from the admitted adapter route. The raw token still
supports CPU-map predecessors in isolation, but the sealed transaction does
not admit those copies; all inherited direct-v1 launch tensors must retain the
same `(False,)*12 + (True,True)` ownership contract as the v1 adapter.

The transaction allocates fresh `[s_b,4]` compact material bars and shared
`[U,3]`, `[U,3]`, `[U,C]` geometry bars. It enqueues every validation, every
accumulation, and every finalizer under one four-byte status, then invokes one
completion fence. Failure consumes the token and retains every root in one
bounded fail-stop quarantine carrier; all later fused work is rejected until a
process restart rather than accumulating an unbounded rejection history.
Unknown completion is also retained by the existing restart-required carrier.
Success releases prepared/raw/bundle/status roots only after the fence and
returns a receipt whose union ids and bars can be consumed once.
There is no global scatter, request-delta commit, optimizer write, executor
route, trainer route, or bounded `q=1` route.

A focused source-contract test was added for exact manifest/factorization
binding, global phase order, single-fence lifetime, quarantine, single-use
bars, and explicit `q=1` exclusion. Nothing was imported or executed on this
host. Native build, runtime behavior, parity, and memory remain unverified.

## Follow-on: partial-construction lifetime repair

The initial adapter preparation accumulated raw tokens and output tensors in
local variables. That was not strong enough for an asynchronous allocator: a
later construction exception could outlive an earlier device return without a
caller-visible owner.

Union-v2 preparation is now two phase. The cold phase performs every
tensor-free rejection before device work and returns a caller-owned lifetime
with fixed raw/output publication slots. The materializer assigns each raw
token or zero tensor to its slot immediately after the returning call and
before validation or the next allocation. On any later exception, exactly one
construction completion fence is invoked and the lifetime becomes the bounded
fail-stop quarantine root; retry and later fused work are forbidden.

On success the transaction owns the lifetime until the execution fence and
receipt seal. Acceptance transfers the bars to the result, clears every raw
and output slot, block/node/bundle root, map/threshold/signature tuple, breaks
the lifetime-to-transaction and state-to-lifetime links, and marks the lifetime
`released`. Rejected paths do none of that cleanup. Total transaction scratch
now explicitly includes the sticky four-byte status; the accepted receipt
retains no status tensor.
