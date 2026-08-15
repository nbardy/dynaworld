# WorldFoam Union-Local Fused Geometry V2

Date: 2026-08-05

Status: exact design plus source-written raw v2 and sealed all-block adapter
slices. The suffixed Python
preparer/split wrapper, Torch schemas/dispatch, Objective-C++ validation and
launch boundary, three Metal kernels, and a union-local all-block transaction
now exist while fused v1 remains the oracle. This work is statically inspected
only: no Python import/test, native
rebuild, Metal compilation, runtime parity, allocator measurement, request
executor routing, or trainer evidence exists yet.

## Decision

The first memory-v2 change should be a request-union geometry cotangent, not a
new depth marginalization, per-block persistent commits, or curvature runtime.

WorldFoam must retain ordered depth because noncommuting optical transfer is
the phenomenon being modeled. The existing compiled `J`-node adjoint already
makes expensive ordered-word/world reverse independent of requested frame
density on a fixed certified atlas. The remaining fused-v1 memory problem is
an output-index-space problem: v1 writes geometry bars for every global world
site even when one request touches only a smaller union.

V2 changes only where exact geometry cotangents are accumulated. It preserves
the kinetic owner/cut math, ordered-transfer VJP, active-manifest transaction,
and global optimizer state.

## 1. Symbols and assumptions

For active native block `b`, let

```text
S     global world-site count,
U     request-union site count, U <= S,
s_b   compact referenced-site count of block b,
C     polynomial weight coefficient count,
d     geometry parameter width = 6+C,
g_b   block-local geometry cotangent in R^(s_b x d).
```

Assume the existing cold union bundle is current and proves:

1. each block compact site id is a valid global id;
2. union global ids are sorted and unique;
3. each compact-to-union map is in `[0,U)`;
4. `union_ids[compact_to_union_b[k]] == block_source_ids_b[k]`;
5. every mapping tensor and union tensor remains identity/version bound through
   transaction completion.

The equality proof below is exact over real arithmetic. Parallel float32
atomics can change summation order and therefore require a tolerance-based
native parity gate.

## 2. Exact factorization theorem

Define binary scatter matrices

```text
P_b in {0,1}^{S x s_b}    compact block b -> global world,
Q_b in {0,1}^{U x s_b}    compact block b -> request union,
P_U in {0,1}^{S x U}      request union -> global world.
```

The cold mapping certificate gives, column by column,

```text
P_b = P_U Q_b.                                           (1)
```

Current fused v1 accumulates

```text
G_v1 = sum_b P_b g_b.                                   (2)
```

Union-local v2 accumulates one transaction-local union ledger

```text
H = sum_b Q_b g_b in R^(U x d),                         (3)
```

then commits it once:

```text
G_v2 = P_U H
     = P_U sum_b Q_b g_b
     = sum_b P_U Q_b g_b
     = sum_b P_b g_b
     = G_v1.                                             (4)
```

Therefore v2 is exactly the same cotangent in a factored index space. It is
not an approximation, a new rendering formulation, or a loss of depth order.

Cross-block duplicate union destinations are required and must accumulate.
Duplicate global site ids inside one block compact table remain invalid.

## 3. Memory theorem and caveat

Let v1's global float32 geometry output coexist with its CPU-float64 request
destination. Its source/destination pair is

```text
M_v1_geometry_bridge = 12 S d.                           (5)
```

V2 uses

```text
M_v2_geometry_bridge = 12 U d.                           (6)
```

so the exact geometry source/bridge saving is

```text
Delta_bridge = 12 (S-U) (6+C).                          (7)
```

For the current `C=3`, this is `108(S-U)` bytes.

The existing lane already charges the MPS union and block maps:

```text
8U + 8 sum_b s_b.                                       (8)
```

V2 must alias those tensors; copying them into every prepared block would
erase part of the win. If the CPU request delta needs a new explicit `int64`
union-to-global commit map, charge another `8U`, giving net counted change

```text
Delta_net = 12 (S-U)(6+C) - 8U.                         (9)
```

Thus the bridge is never larger, but complete source-tensor memory is not
guaranteed smaller when `U` is nearly `S`. Promotion requires measured `U/S`
and complete request peaks. At `C=3`, a newly allocated CPU map still gives a
positive counted win only when `U/S < 108/116`, unless the map can be reused or
the identity/global case avoids it.

This is logical tensor accounting. It excludes allocator bins, Metal private
storage/registers/spills, Python objects, target decoder allocations, and
process RSS.

## 4. Three index identities must remain distinct

The v2 raw ABI must receive and bind all three identities:

```text
block source_site_ids_i64
    compact -> global provenance;

block compact_to_geometry_output_i64
    compact -> request-union write destination;

shared geometry_output_source_site_ids_i64
    request union -> global provenance.
```

Do not reinterpret the existing block `source_site_ids_i64` as a union index.
The kernel still uses it to validate the owner against the immutable world.
Before any atomic write, device validation must prove

```text
geometry_output_source_site_ids_i64[
    compact_to_geometry_output_i64[k]
] == source_site_ids_i64[k].                            (10)
```

It must also prove global ids are in `[0,S)`, destinations are in `[0,U)`, the
union is sorted/unique, and the output shapes are `[U,3]`, `[U,3]`, and
`[U,C]`.

The v2 config needs both `S` and `U`; it cannot infer the global world size
from an output whose first dimension is now `U`.

## 5. Fail-atomic transaction

V2 preserves the current request transaction:

```text
cold prepare and bind exact active manifest/maps
-> allocate fresh zero compact-material and union-geometry ledgers
-> clear one shared status
-> validate all blocks and the shared union
-> status-guarded accumulate all blocks
-> finalize all compact ledgers and the U-sized union ledgers
-> one completion fence and status read
-> accept one sealed request delta or quarantine every live root.
```

No persistent optimizer bar may change before acceptance. On rejection or a
failed abort fence, prepared blocks, both maps, union ids, output ledgers,
status, callback snapshots, and current request/sample roots remain retained
for restart-required quarantine.

Naive per-block persistent commits are not a substitute. They violate
request-level all-or-nothing admission. A future streamed transaction would
need transaction-local union scratch, a presealed manifest, bounded block
batches, a fence after each batch, one final status, and exactly one persistent
commit.

## 6. Source implementation map

Keep fused v1 intact as an oracle and add a suffixed `fused_union_v2` route.

Current source-only progress: items 1--4 now have an implementation. The
Python token carries both maps, `S`, `U`, and retained CPU transfer
predecessors; Metal has separate validate/accumulate/finalize kernels; and the
Objective-C++/Torch boundary registers three split-phase suffixed operators.
The runtime adapter binds the existing sealed spatial bundle, requires exact
canonical all-block order, aliases one union identity and each resident map,
owns fresh union-local scratch, fences once, and returns single-use accepted
bars without a commit path. These sources are unbuilt and unrun. Items 5--9
remain unimplemented seams.

Construction is now explicitly two phase. A caller first retains a cold
construction lifetime after every count, map, threshold, `U/C`, and total
scratch-budget check has passed. Materialization uses fixed publication slots:
each returned raw token or zero tensor is installed into the caller-visible
lifetime before the next potentially throwing validation/allocation. A partial
failure consumes one construction fence and retains the lifetime in the same
bounded fail-stop quarantine used by transaction rejection. Success transfers
the lifetime into the transaction. Only after the later execution fence and
accepted receipt does the adapter clear raw tokens, outputs, block/bundle
roots, and bulky per-block Python metadata and break both lifetime/transaction
links.

One source boundary remains opaque: the outer lifetime can publish only the
aggregate raw token after the raw preparer returns. Config temporaries allocated
inside that preparer before its return cannot be individually installed in the
outer carrier. An exception is completion-fenced, but native fault injection
and allocator telemetry are still required to certify retention at that seam.

The scratch budget is total transaction scratch, not bars alone:

```text
M_transaction = M_compact_material_bars + M_union_geometry_bars + 4 bytes
```

The four bytes are the one sticky int32 validation status. Accepted receipts
retain zero status bytes.

Material bars remain one compact `[s_b,4]` ledger per block. Every compact
ledger is validated and finalized once. The shared validation/finalization
flag applies only to the three union geometry ledgers and is true only for the
first canonical block. The receipt therefore records
`material_output_index_space=block_compact` and explicitly does **not** certify
any union-material finiteness property; a bounded `q=1` material-union design
is a separate later problem.

1. Raw Python ops: add prepared-v2 fields for compact-to-union, shared union
   ids, `U`, and global `S`; preserve global source ids.
2. Metal: factor common kinetic reverse arithmetic and add v2
   validate/accumulate/finalize entry points whose geometry ledgers scan `U`.
3. Objective-C++ and bindings: add three suffixed split ops with explicit
   `S`, `U`, both maps, and `[U,*]` output validation.
4. Runtime adapter: bind spatial-bundle generation, union tensor
   identity/version, each mapping digest, output index-space digest, and all
   roots through the fence/callback/quarantine lifecycle.
5. Executor receipt: certify exact active block order and a single shared union
   namespace under a distinct `geometry_output_index_space_certified` flag.
6. Dense request: bridge accepted `[U,*]` float32 bars to `[U,*]` CPU float64
   bars; keep compact material union assembly unchanged.
7. Request delta: add explicit geometry index-space, mapping provenance, union
   row count, and a bounded CPU union-to-global map. Staged/global deltas remain
   `[S,*]`.
8. Commit: use CPU `index_add_` for union geometry and the existing `.add_`
   path for global staged geometry. Optimizer authorization remains global and
   unchanged after commit.
9. Fixed-camera step and combined receipt: route and digest-bind
   `fused_union_v2` explicitly; never silently change `fused_direct_v1`.

The current lane's cold bundle relation must be rechecked before sealing v2
tokens. Warm identity/layout checks alone do not prove mapping contents.

## 7. Required falsification gates

V2 is rejected unless all of these pass:

1. `U=S` identity-map parity against fused v1;
2. `U<S` forward/loss/material/position/velocity/weight parity against staged
   sparse reverse;
3. cross-block duplicate-site accumulation;
4. compact-map duplicate, out-of-range, content-tamper, wrong-union, and
   source/union mismatch rejection before writes;
5. shared-status validation/finalization failure and nonfinite postwrite
   quarantine;
6. callback mutation/reentry rejection with both maps in the snapshot;
7. no request-delta or optimizer commit before accepted fence/status;
8. CPU union `index_add_` parity, digest binding, single-use consumption, and
   complete release;
9. exact logical accounting including any new `8U` CPU map;
10. fresh-process allocator/RSS/MPS evidence and observed `U/S` distribution.

Only after parity and measured whole-request memory improve should v2 replace
v1. If `U` is routinely near `S`, keep v1 or select between output spaces per
request rather than claiming a universal union win.

## 8. Relation to the scientist's connection proposal

The stratified Lagrangian optical connection remains a legitimate future
hypothesis for transporting or certifying transfer across moving ray fibers.
It is orthogonal to this change. Union v2 does not alter ray transport,
curvature, camera gauge, event structure, or ordered optical products; it
factors an already-derived world cotangent after the ordered reverse.

Accordingly, the dependency order is:

```text
correct staged/fused memory accounting
-> union-local fused geometry v2
-> rebuilt native parity and allocator evidence
-> CPU<->MPS material bridges, live restore, fixed-512 trainer/evaluator
-> only then U / flow-corrected U_tilde / curvature K_F falsification.
```

No new Schur-complement analogue is needed for this systems result. The
noncommuting depth coordinate remains alive; compiled adjoints and sparse
index-space factorization provide the reuse.
