# Device batching helps; interval capacity limits the physical time span

This goal continuation first finished the validated loader integration as root
`47d5a71`, then ran a new execution-layout control. The prior user-facing review
restated retained source-convergence evidence rather than adding accepted runs.
This work advances evidence through five independently accepted Metal layouts
and a reproduced capacity failure. One lead, no subagents, sequential jobs,
offline W&B and all existing host/RSS/MPS/swap/disk/time guards remain.

## Experiment and implementation

The same 2048-tube, 800-update Coffee Martini world is frozen. Checkpoint
SHA da6eac63e2497a47fe7bcfa5be645b7b665199c318e17384b31cbf493f4e0c77,
logical world SHA 2ac9be4f45e1304d9ed2e8208303d3cd1ab89be6b46233434e.
Cam06, 96x128, full physical interval, float32, source alpha/depth law and all
numerical tolerances are unchanged. Configured tile_t stays 1, capacity 256,
CPU cache 8. Test requested device chunks 1/2/4/8 at F4 and F32. F4 additionally
checks non-unit selected-time slicing; F32 uses all 32 sample times. Each row
has one warmup and three alternating replay/compiled timing trials.

The existing frozen report now accepts an optional positive integer
resident_chunk_frames. Its default preserves the old tile_t-derived behavior;
explicit values are clamped only to the actual frame count. It controls how
many outputs/graphs are retained for a loss/backward chunk. Replay continues
projecting and rendering each frame individually inside that chunk. Compiled
rendering sends the chunk to the existing native interval evaluator.

Important correction to the pre-experiment interpretation: unchanged configured
tile_t does NOT imply unchanged native interval bins. The native preparation
passes tile_t=config.frames to its packer, unioning intervals across the whole
execution chunk. Changing chunk size changes packed candidate work and output/
graph batching together. The parent atlas remains byte-identical. This is an
execution-layout control, not an isolated launch-overhead or kernel-only test.

## Capacity result and failed first attempt

The first run completed four correct rows, then failed at F4/chunk4 with
"packed projective interval atlas tile capacity overflow". The new script also
used unsupported keyword arguments to W&B SummaryDict.update while finalizing;
it now passes a dictionary. This secondary logging error obscured the final
exception and prevented the outer launcher from saving a successful resource
receipt. Those first rows are retained diagnostics, not the accepted resource-
backed result. All exact first-run sources are archived in
outputs/benchmarks/2026-09-13_frozen_device_chunks/after/; offline ID `20j7gs6o`.

An exact CPU packing probe reconstructs verified retained atlases, slices them
with the production iterator, and calls the same interval packer at capacity 256.
This explains the failure without raising capacity or truncating contributors:

| Sample count | Chunk1 max slots | Chunk2 | Chunk4 | Chunk8 |
| --- | ---: | ---: | ---: | ---: |
| F4 across the full interval |119|181|334 (33 overflow tiles)|334 (same effective chunk4)|
| F32 across the full interval |149|172|193|260 (4 overflow tiles)|

Four adjacent dense samples fit while four widely spaced samples do not.
Capacity depends on unioned active intervals over physical time, not merely
image count or MPS memory. CPU probe RSS 498,941,952 bytes, zero new swap,
unchanged guards. It is mechanical packing evidence with no W&B/timing claim.
Its exact source receipts are retained in capacity_sources/, including the
intermediate logging-only fix verified by SHA against the run receipt.

The final experiment retains the three capacity-negative requested cases and
runs only the five geometrically admissible layouts. It does not claim an
all-green matrix. The runner checks every rendered atlas SHA against the CPU
packing input; existing numerical gates still reject any unexpected mismatch.
The fixed logging path completes and retains the final outer resource receipt.

## Accepted Metal results

| F | Device chunk | Compile s | Compiled E+B s | Compiled full s | Replay E+B s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 1 | 0.28225 | 0.14469 | 0.43130 | 0.03432 |
| 32 | 1 | 1.85824 | 1.87755 | 4.85160 | 0.33562 |
| 4 | 2 | 0.29350 | 0.11942 | 0.41461 | 0.04175 |
| 32 | 2 | 1.87581 | 1.64620 | 4.65392 | 0.32242 |
| 32 | 4 | 1.89969 | 1.21665 | 4.23945 | 0.32963 |

At F32, chunk1->4 cuts compiled E+B by 35.20% (1.88->1.22s), full cost by 12.62%
(4.85->4.24s). Compilation stays about 1.86-1.90s; target loading about 1.12s.
Replay E+B stays about 0.32-0.34s and remains faster. Sizes run sequentially,
not in randomized crossover order; the earlier chunk1/2 attempt also favored
chunk 2, but small differences still include run-order variation. E+B excludes
I/O/transfer/loss and includes CPU evaluator setup; full cost includes compilation
and measured forward/backward segments, excluding optimizer work and cleanup.

Five rendered rows pass the existing independent image/world-VJP, checkpoint,
identity, payload, retained-storage, route-memory and repeated-timing validators.
Both F4 rows pass full-vs-one-frame non-unit slicing. Targets, camera, world,
acceptance and parent atlas bytes match the prior one-frame sweep. Max RGB
error 8.64267349e-7; max per-parameter normalized VJP 1.84535141e-6. The focused
regression gate remains 189 passing tests (4.04s). Peak sampled tree+launcher
RSS 2,146,926,592 bytes (~2.00GiB), no new swap. The F32 compiled route sampled
driver peaks 29.34/29.51/37.93MB for chunks1/2/4; the unchanged 2-GiB allocator
cap is not the observed chunk limit. No resource guard tripped.

Final artifacts: outputs/benchmarks/2026-09-13_frozen_device_chunks_capacity_checked/,
offline W&B `0yon96y9`. report_validation.json distinguishes five accepted rendered
rows from three capacity rejections. Tests/RSS/source/native/data/atlas backing
are verified; native SHA 6995d093694fda0e54c85926aa94bd3f3aeab58019c4121427b54dd2b8559cf5.
No optimizer fit ran: train/heldout quality remains 20.84/15.30 dB. At fixed
chunk 2, F4->F32 still grows 13.78x compiled E+B for 8x frames. No sublinear
claim, no completed full-layout matrix, and no publication promotion follow.
Public counts stay 0/7 contexts and 0/21 lanes; BASELINES remains unchanged.

## Source ownership and next action

Root owns compare_frozen_world_device_chunks.py, its JSONC, this note, scoped
status edits and frozen_world_device_chunks.patch. The small STAR hook applies
to the retained streaming preimage, not clean STAR HEAD; actual temporary
apply/check and byte equality passed, with patch_identity.json retained.
STAR remains `f735898`; no blind staging of unrelated streaming/native/browser WIP.
Run-bound sources are archived and hash-checked. One post-run comment clarifies
that native bins span the chunk; Python AST equality proves no executable
change (source_comment_correction.json). No tests were rerun for that comment.

Next refresh the existing bounded Python profile at the valid F32/chunk4 layout,
including all native-input packing inside evaluator time, to locate the remaining
1.22 s E+B versus replay 0.33 s gap. Larger uniform chunks are not a free lever;
any future capacity-aware subdivision must preserve all contributors and charge
its selection/packing work. Do not retry the stopped shape-recording profiler,
raise capacity to bypass this result, or infer a complexity exponent from mixed
chunk sizes. The broad overnight goal remains active.
