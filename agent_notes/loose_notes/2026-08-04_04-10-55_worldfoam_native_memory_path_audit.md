# WorldFoam native memory-path source audit

## Scope and safety

This was a source-only audit. I did not import Python, run tests, build an
extension, launch Metal/MPS/CUDA, or run training. The repository and nested
`third_party/fast-mac-gsplat` worktree were already heavily dirty, so I made
only one bounded allocation/accounting fix and preserved all other work.

The question was narrower than "which renderer already produces images?":
which native implementation is closest to the intended WorldFoam theorem,
where expensive ordered-word forward/backward work is independent of the
requested frame count and the frame/sample slice is streamed.

## Decision

The closest implementation is the **kinetic precompiled equal-rank path inside
`world_foam_lane2_fused_slab_v0`**, specifically:

1. `wf2_kinetic_precompiled_length_p0_lie_node_forward_tensor`;
2. `wf2_kinetic_ragged_p0_lie_sample_mse_vjp_accumulate_only_tensor`;
3. `wf2_kinetic_precompiled_length_p0_lie_material_node_vjp_tensor`;
4. the block-major lifecycle in
   `kinetic_dense_cached_native_material_request.py` and
   `kinetic_native_material_step_executor.py`.

This is not merely the least-bad existing shader. Its decomposition is the
desired one:

```text
sealed ordered word + J chart nodes
  -> one node-chart forward per active block
  -> stream bounded K-local samples directly into one node cotangent
  -> one ordered-word VJP per active block
```

The material-only VJP deliberately aliases the disabled length-bar argument to
the caller material bar, so it does not allocate the otherwise unnecessary
`[J,W]` geometry adjoint. The loss-only sample kernel does not allocate or
write `[N,3]` predictions. The executor checks that node forward and reverse
each occur exactly once per active block and that no frame/sample axis remains
in the sealed telemetry.

The important qualification is that the implementation is still embedded in
a very large research variant and its real native allocator/lifecycle behavior
has not been measured after the latest edits. Source structure supports the
memory theorem; it does not yet prove the runtime claim.

## Persistent and live storage of the selected path

For an equal-rank active block with `S_b` compact sites, `R_b` rows/tracks,
`J_b` chart nodes, and `W_b` word entries, source-visible storage is:

| Lifetime | Tensor family | Bytes |
| --- | --- | ---: |
| cached CPU artifact | offsets, owner word, row metadata, node lengths | `8 S_b + 57 R_b + 4 W_b + 4 J_b W_b + 20` |
| resident native launch copy | same compiled program plus configs | `8 S_b + 4 R_b + 4 W_b + 4 J_b W_b + 24` |
| active node forward | Lie node chart | `16 R_b J_b` |
| active sample reduction | node cotangent | `16 R_b J_b` |
| active material refresh | compact material | `16 S_b` |
| active material reverse | compact material bar | `16 S_b` |
| material-only geometry bar | disabled | `0` |
| one ragged sample block | row id, `J_b` weights, target, identity | `4 N J_b + 24 N` retained by the Python block |
| public native sample preparation | device row IDs plus two tiny configs | `4 N + 20` after this audit |

The sample count `N` is bounded by `maximum_samples_per_launch`; it is not the
entire requested frame set. Target decoding holds one CPU frame and one bounded
chunk transfer. The conservative current policy fences after every native
sample launch, bounding asynchronous launch depth to one.

The request owns a scalar loss and one node cotangent per active block while
streaming all of its temporal chunks. After coverage, it performs one reverse
per active block, scatters into a world-bounded union bar, fences, and releases
block scratch. A whole optimizer step owns one world-sized material bar and
loss scalar; it does not own an `F`-axis tensor.

There are nevertheless explicit non-frame scaling axes to preserve in every
claim:

- lane preparation converts the materialized-block iterator to a tuple and
  builds every runtime in the cached artifact, so lane residency grows with
  compiled block count and the sum of all `J_b W_b` programs;
- the request's `active` dictionary retains chart, cotangent, compact material,
  and scalar loss for every block touched before the reverse loop, giving the
  source upper bound
  `32 sum_b(R_b J_b) + 16 sum_b(S_b)` before union/commit scratch;
- the union material bar grows with unique sites in the spatial request, and
  the one-at-a-time compact commit bar grows with `max_b S_b`;
- full geometry adds world-sized CPU float64 position/velocity/weight bars,
  optional request-track ray bars, and one-at-a-time `4 J_b W_b` native length
  adjoint; material-only avoids all of those except the world material bar;
- target-gather dictionaries, destination/pixel index vectors, selected RGB,
  sample identity lists, and ragged interpolation weights grow with the bounded
  decode/sample chunk, not with the full requested sequence;
- cached artifacts outside the request may be byte-budgeted, but a cache budget
  is not the same as the live native-lane bound and must be measured separately.

If spatial block count itself becomes the next memory limiter, the next
reordering is block outer / temporal chunks inner so one chart/cotangent pair
can be reused and reversed before moving to the next block. That requires a
block-keyed observation iterator (or cheap replayable target source) so it does
not regain memory by buffering all targets. It is an optimization of the
spatial bound, not a prerequisite for the current frame-sublinear theorem.

Remaining unmeasured storage includes Metal driver/allocator scratch, compiler
temporaries, command-buffer retention, Python-object heap peaks, decoder
allocator peaks, and private-register spills. Those omissions are already
fail-closed in request/result metadata (`native_runtime_verified=false`,
`allocator_peak_measured=false`, and the more specific measurement flags).

## Why the other native paths are not the target backend

### Retained-fiber / ordered-transfer shader

`research_experiments/spd4_world_tubes/retained_fiber_transfer.metal` is a
valuable physical oracle and selective fallback, not a compact WorldFoam
trainer:

- forward writes a full `[F,H,W,3]` image;
- the wrapper creates or retains a full `[F,H,W]` fallback mask;
- autograd saves seven atom-field tensors, all times, and the full mask;
- backward owns at least 2,048 source-visible bytes of private arrays per
  pixel thread (`lambdas[64]`, `betas[64]`, `source_colors[64]`, and
  `behind[64]`), which may spill;
- its certificate thread owns `active_ids[256]` (1,024 bytes), scans every
  atom, then examines active pairs;
- certificate expansion materializes a full `[F,H,W]` mask.

`hybrid_transfer.py` simultaneously holds full fast RGB, full retained RGB,
the mask, and the `torch.where` result. It belongs in the ordered-transfer
ablation/oracle lane, not in a claimed sublinear temporal training path.

### Finite-element M0--M5 material shader

`finite_element_material_transfer.metal` is a good evaluator and derivative
oracle, but its Python bridge materializes per-segment intermediates:

- forward: tau, beta, moment, bounds, status = `32 N` output bytes;
- VJP outputs: controls, two colors, length, status = `44 N` bytes;
- VJP cotangent inputs add `20 N` bytes;
- `_raise_on_invalid` copies the full status vector to CPU and synchronizes;
- non-contiguous cotangents may be copied by `.contiguous()`.

Do not put that bridge in the ordered-word hot loop. Reuse/inline its shared
`evaluate_material` arithmetic in the word forward/VJP kernels, or add a fused
evaluate-compose and compose-VJP ABI that writes only the caller-owned node
chart/material bar. Branch diagnostics should be device-reduced to a bounded
counter, not copied as a status vector during every warm call.

### Original PowerFoam Metal and dynamic-powerfoam-metal

These are the closest **frame-local streaming-rasterizer** ancestors. Their
backward recomputes/replays ray-cell work from compact per-pixel checkpoints
(`log_t` plus `pixel_stop` or `tile_stop`) instead of saving every intersection.
That is the right local replay idea.

They are not temporally sublinear implementations:

- rays and image outputs are `[B,H,W,*]`;
- training saves full `[B,H,W]` transmittance/checkpoint tensors;
- autograd saves rays, candidates/topology, and those checkpoints;
- tiled candidate arrays scale with tile-cell incidences, and emit-sort adds
  keys, unsorted IDs, sort order, and sorted IDs;
- auxiliary outputs add `[B,H,W,Q]` and `[B,cell_count]` tensors;
- backward allocates world gradients and may allocate missing full-image
  cotangents.

`dynamic-powerfoam-metal` has the same decisive shape: its train forward emits
out, alpha, `log_t`, and `pixel_stop`, and its autograd context saves thirteen
tensors including rays/checkpoints. If batch is used as time, storage and
expensive word replay both remain linear in frame count.

The useful inheritance is therefore PowerFoam's bounded per-ray replay and
spatial candidate discipline, not its batch-shaped autograd boundary.

## Allocation fixed in this audit

`prepare_kinetic_ragged_p0_lie_sample_block` accepted CPU int32/int64 row IDs,
but unconditionally converted them to a CPU int64 tensor for bounds checks and
then converted them back to MPS int32. Production supplies CPU int32, so this
created an avoidable `8 N` host tensor on every streamed sample block.

The public ABI now:

- requires the caller's CPU row vector to be contiguous;
- checks min/max in its supplied integer width;
- narrows only during the device copy;
- accounts public preparation as `4 N + 20`, down from `12 N + 20`.

The request preflight and source-contract assertions were updated to the same
formula. No runtime verification was performed in this source-only session.

## Float64 sample-weight scratch closure

The dense sample materializer previously budgeted the returned ragged sample
block but did not bound the temporary tensor graph created by
`sample_to_node_weights`. That evaluator is row-separable, yet one call over a
large row group can create several float64 and boolean `[K,J]` intermediates,
including the dense exceptional-row fallback. A bounded `N` output alone was
therefore not a sufficient source-level peak argument.

The request now has a dedicated sample-materialization byte budget. Before
building a native lane it derives the largest admissible returned block, and
the materializer evaluates each interpolation row group in bounded
subchunks. It keeps the existing float64 arithmetic and exact/dense-fallback
counters, writes each subchunk into the same preallocated destination, and
releases the evaluator result, times, and destination indices before the next
subchunk. This changes lifetime, not interpolation semantics.

The conservative public-tensor envelope is:

```text
interpolation_scratch <= 4096 + 512 J + 8 J^2 + K_sub (1024 + 512 J)
materialization_peak <= max(
    N (8 J + 12) + interpolation_scratch + 16 K_sub,
    N (16 J + 32)
)
```

The rank-squared term covers the boolean validation temporary created from the
stored `J x J` fit matrix. The other constants intentionally overcount
source-visible expression temporaries; they are an admission-control envelope,
not an allocator model. The returned
block remains `4 N J + 24 N` bytes and the following native preparation adds
`4 N + 20` bytes. Request preflight accounts target-transfer residency,
materialization, and native preparation before target decoding or lane work.
Whole-step accounting records the requested and effective sample caps, maximum
interpolation rows per subchunk, interpolation scratch upper bound, and total
materialization upper bound.

This closes the unbounded source-visible interpolation lifetime, but it does
not measure PyTorch-internal temporaries, allocator slabs, Python containers,
asynchronous transfer retention, or RSS. Accordingly
`sample_materialization_float64_scratch_measured`,
`whole_step_python_object_peak_measured`, and `allocator_peak_measured` remain
false. A later native improvement can evaluate row/time weights directly in a
caller-owned device buffer and eliminate the CPU `[N,J]` staging path, but that
is an optimization rather than a missing bound.

## Precise remaining native implementation work

1. **Build and run the selected ABI in a quiet window.** Prove extension/source
   freshness, operator registration, forward/VJP parity, exactly-one reverse,
   finite gradients, and real fence behavior.
2. **Measure actual allocator peaks.** Capture process RSS and Metal/MPS
   allocator deltas around cold compile, lane prepare, node forward, each
   bounded sample launch, reverse, fence, and release. Separate retained,
   allocated, and driver/compiler peaks.
3. **Run the frame-scaling gate.** Hold world, spatial bundle, `J`, and `K`
   fixed while increasing requested `F`. Expensive node/word forward and VJP
   counts must be invariant; sample work may be linear; peak bytes must follow
   the bounded analytic envelope rather than `F`.
4. **Extract a slim canonical native variant after parity.** The selected
   kernels live inside a roughly 44k-line fused-slab research collection.
   Extract only compiled topology upload, kinetic node forward, ragged loss-only
   reduction, material/full geometry VJPs, and their small bridge. Do not delete
   the source variant until the extracted library passes byte-for-byte fixtures
   and source-freshness gates.
5. **Fuse material families at the word boundary.** Port M0--M5 evaluation into
   the node word program without materializing per-segment arrays. Preserve the
   current P0 fast path as the first paper row; gate adaptive M3/M5 use on real
   held-out evidence.
6. **Reuse fixed-capacity session scratch if allocator measurements justify
   it.** Let the caller own node chart, node cotangent, scalar loss, diagnostic,
   compact material/bar, sample row/config buffers, and target/weight staging.
   Kernels should mutate those buffers; no autograd `save_for_backward` boundary
   should surround the temporal request.
7. **Integrate geometry only after material-only proof.** Full geometry already
   has a `[J,W]` length bar and CPU reduction path. Keep this a separately
   labeled row until its lifecycle and peak memory are proven; it is not needed
   for the compact material-training claim.

No new Schur-complement-like mathematical discovery is indicated by this
source audit. The required reformulation already exists: compile changing
depth order into a bounded ordered word at chart nodes, stream camera samples
into a shared node cotangent, then differentiate the word once. The remaining
gap is native integration and measurement, not a missing representation-level
formula.

The external scientist's strongest new optical-depth translated-measure
formulation fits this backend without increasing runtime memory. The measure
`(kappa, nu)` is the proof/certificate object; its Laplace image is exactly the
four-scalar `(beta, m)` transfer already compressed into the node chart (stored
in Lie coordinates as `(kappa, velocity)`). It should strengthen the paper's
composition, seam, and tangent arguments, but the native path should not
materialize `nu` or a tangent measure.
