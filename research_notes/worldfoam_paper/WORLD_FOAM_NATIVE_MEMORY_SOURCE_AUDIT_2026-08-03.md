# WorldFoam Native Memory And Lifecycle Source Audit

Date: 2026-08-03

Status: source contract verified on CPU; native Metal build, allocator peak, and
optimizer-step invocation counts remain unverified.

## Dimensions

- `F`: requested observations/frames per pixel track in one logical step.
- `K`: observations per track in one streamed coordinator request.
- `B_p`: pixel tracks in one spatial coordinator bundle.
- `N <= B_p K`: row-selected observations in one native ragged launch.
- `R_b`: `(track, chart)` rows in equal-rank native block `b`.
- `J_b`: actual temporal rank of block `b`; there is no `J_max` padding.
- `W_b`: ordered owner-word entries in block `b`.
- `S_b`: compact material sites referenced by block `b`.
- `S_u`: union-local sites across the native blocks of one spatial bundle.
- `S_g`: global model sites.

## Exact Logical Tensor Payloads

For one materialized equal-rank CPU payload, the checked layouts occupy

```text
8 S_b + 57 R_b + 4 W_b + 4 J_b W_b + 20 bytes.
```

This includes source-site ids, row/domain identity, CSR offsets/owners,
physical node lengths, and the four-int config. On an MPS device the current
runtime owns another launch copy of

```text
8 S_b + 4 R_b + 4 W_b + 4 J_b W_b + 24 bytes.
```

The runtime presently retains the CPU payload as its provenance object, so
these CPU and device payloads coexist. This is bounded independently of `F`,
but it is still a removable host/device duplication in a future production
seal.

The expensive material state for a live native block is:

```text
node chart             16 R_b J_b
node cotangent         16 R_b J_b
compact material       16 S_b       (caller-owned)
compact material bar   16 S_b       (caller-owned scratch)
geometry length bar     0           (material-only ABI)
```

The optional geometry-training reverse still returns `4 J_b W_b` bytes. It
must not be used for material-only training.

For a row-ragged sample launch, the bridge and native wrapper coexist at a
logical minimum of

```text
weights + target + CPU row id + flat provenance + device row id
= 4 N J_b + 28 N bytes,
```

before allocator/transient storage. The outer staged target rectangle may
also remain live at `12 B_p K` bytes. Training allocates no `12 N` prediction
output; evaluation may request that optional output explicitly.

The union-local accumulator adds:

```text
source union ids       8 S_u
compact-to-union maps  8 sum_b S_b
union material bar    16 S_u
loss scalar             4
```

There is no per-request `[S_g,4]` bar.

## What The Native Source Now Proves

1. Precompiled kinetic geometry is `[J_b,W_b]`, never `[F,W_b]`.
2. Node state and its cotangent are `[R_b,J_b,4]`.
3. Ragged sampling uses `[N]` row ids, `[N,J_b]` weights, and `[N,3]`
   targets; it does not form a row-by-global-time Cartesian product.
4. The loss-only sampler writes into caller-owned loss/node-cotangent state
   and allocates no prediction tensor.
5. The dedicated kinetic material VJP returns the caller-owned `[S_b,4]` bar
   and allocates no discarded `[J_b,W_b]` geometry bar.

The source verifier records these as source-only facts. It does not claim a
built Metal runtime or measured allocator behavior.

## Load-Bearing Lifecycle Gap

Bounded tensor shapes alone do **not** prove sublinear expensive work.

The current union-local assembly is request-local. If it is invoked once for
every `B_p x K` coordinator request, it can execute node forward/VJP roughly
`ceil(F/K)` times per active native block:

```text
wrong expensive reverse work = O(ceil(F/K) sum_b J_b W_b).
```

The required optimizer-step lifecycle is:

```text
for spatial bundle:                         # outer loop
    refresh each active native block once
    zero one [R_b,J_b,4] cotangent per block
    for K-sized observation request:        # inner loop
        stream targets/rows/weights
        accumulate loss and node cotangents
    VJP each active native block once
    scatter each compact bar once into one union-local bar
```

With a single pass over target requests, heterogeneous blocks inside a bundle
remain live together. Therefore the honest node-state peak is

```text
32 sum_(b active in bundle) R_b J_b bytes,
```

and the global peak is the **maximum over spatial bundles**, not the sum over
all bundles. The older `32 B_p J_max` estimate is valid only if a target-
replay-safe sequential-native-block schedule is implemented and measured.

Until a step-scoped session enforces this loop order and call count, the
memory shape is bounded but the principal `F`-sublinear backward-work claim is
not integrated.

## Forbidden State

The production material-training path must reject or report any of:

- persistent `[F,W]`, `[F,R]`, or `[R,F,*]` compiled/reverse tensors;
- resident full-video targets or full-step predictions;
- a per-request global `[S_g,4]` bar;
- a material-only `[J,W]` geometry cotangent;
- node forward or node VJP invocation counts growing with `ceil(F/K)`.

## Required Runtime Measurement Schema

For every optimizer step, spatial bundle, and native-block generation digest,
record:

- `R_b`, `J_b`, `W_b`, `S_b`, `S_u`, `B_p`, `K`, `N`, and logical `F`;
- node-forward, ragged-sample, material-node-VJP, and union-scatter call counts;
- `sum_b J_b W_b` word interactions and `sum_launches N J_b` sample interactions;
- live tensor payload bytes at post-refresh, peak-sampling, pre-VJP, post-VJP,
  and post-release phases;
- allocator current/peak/reserved bytes and command-buffer completion points;
- whether CPU compiler payloads and device launch payloads coexist;
- prediction bytes and geometry-length-bar bytes, both required to be zero for
  material training.

An `F = 4, 8, 16, 32, 64, 128, 300` sweep at fixed spatial/compiler state must
show constant node-forward and material-VJP counts per block, constant
expensive-state bytes, bounded `K`-sample peak, and only the expected
`O(F J)` cheap sample interpolation/reduction work.

## Source Change In This Audit

The new material-only kinetic ABI is
`kinetic_precompiled_length_p0_lie_material_node_vjp_accumulate_launch_only`.
It shares the exact reverse arithmetic with the geometry-capable kernel but
disables length writes and aliases the otherwise unused kernel argument to the
caller-owned compact material bar. No `[J,W]` output is allocated.

No Metal/MPS build or execution was performed during this audit.

## Subsequent CPU Integration Closure

The lifecycle gap identified above is now closed at CPU/fake-native scope by
`research_experiments/world_foam_lane2/kinetic_ragged_paper_step_cpu_fake_native.py`.
It holds one spatial bundle across all temporal chunks, invokes node forward
and the material-only word VJP once per active block, performs one union
scatter, and releases bundles sequentially. Tests show `K=1/4` agreement with
an independent direct-autograd oracle and `F=5/41` invariance of retained
runtime bytes and compiled-word work. This does not revise the audit's native
conclusion: rebuilt Metal invocation telemetry and allocator measurements are
still absent.
