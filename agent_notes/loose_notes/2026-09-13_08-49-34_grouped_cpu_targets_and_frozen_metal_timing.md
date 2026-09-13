# Grouped CPU target loading preserves the fitted-world contract

The preceding user-facing turn clarified old source fitting versus shared-world
quality, using already retained evidence; it did not advance accepted counts.
This continuation returns to the completed but uncommitted loader integration.
Prior committed state is root b444982 / STAR f735898. One lead, no subagents,
sequential accelerator jobs, offline W&B and unchanged resource guards apply.

## Change and validation

The existing PaperMulticamTargetProvider is adopted as an owned root module.
Its new load_grouped_frozen_target_frames helper prefetches the largest whole
multiple of the device chunk that fits the existing CPU cache. With cache8
and device chunk1, eight selected frames share a decoder request; only the
current frame reaches Metal. Prefetch and current selection both remain inside
target_cpu_load timing in the common loader used by replay and compilation.
Order, duplicates, canonical decoding, cameras, target bytes, alpha law,
precision, renderer and learned world are unchanged. No shader rebuild occurred.

Five real-provider cases cover chunk/cache (1,8), (3,8), (8,8), (16,8), (1,1),
including skipped and duplicate selected times over two passes. They check exact
output, bounded batches/cache and grouped decode counts. The final focused gate
passes 189 tests in 4.02 seconds. The existing independent scientific validators
accept all four F4/8/16/32 rows, with byte-identical retained atlases and unchanged
world, camera, target and numerical contracts. Max RGB error is 8.64267349e-7;
max per-parameter normalized VJP error is 1.72261932e-6. Non-unit time slicing passes.

Seconds, separate processes; E+B excludes I/O, transfer and loss but includes
host evaluator setup. Full includes compilation plus forward/backward measured
segments, not optimizer updates or inter-segment cleanup. Separate-process
variation does not causally attribute changes in other phases to this loader.

| Frames | Compile before / after | E+B before / after | Full before / after |
| --- | ---: | ---: | ---: |
| 4 | 0.29421 / 0.28983 | 0.14639 / 0.13348 | 0.44385 / 0.42411 |
| 8 | 0.72904 / 0.74382 | 0.32414 / 0.29893 | 1.06859 / 1.04633 |
| 16 | 1.24308 / 1.29556 | 0.99235 / 0.76828 | 5.37078 / 2.69158 |
| 32 | 1.84470 / 1.89020 | 2.36214 / 1.93107 | 10.55377 / 4.96755 |

F32 compiled CPU target loading falls 6.30112 -> 1.12123 seconds. Replay also
benefits: its final F32 E+B is 0.32979 seconds, full forward/backward 1.46586.
This is shared data-loading progress, not a World Tubes rasterizer speed win.
Eightfold frame growth still costs 8.754x replay and 14.468x compiled E+B.
No sublinear claim follows. No optimizer fit ran; shared-world quality remains
20.84/15.30 dB train/heldout. Local numerical acceptance stays 4/4; publication
eligibility false, public context/lane counts 0/7 and 0/21, BASELINES unchanged.

## Memory, artifacts and source ownership

The CPU cache stays at eight frames / 1,179,648 bytes, request/decode batches
at most eight. Conservative transient accounting includes old cache, decoder
batch, clones and returned stack: 32 frames / 4,718,592 bytes. This is not a
claim that total CPU storage equals cache bytes. No full-video tensor is made.
Peak process-tree plus launcher RSS is 1,990,328,320 bytes (1.854 GiB), tests
920,174,592 bytes; both have zero new swap and no guard trips. The unchanged
3-GiB RSS, 2-GiB MPS allocator, 256-MiB swap-growth and 600-second limits apply.

Authoritative final artifacts: outputs/benchmarks/2026-09-13_grouped_target_provider/,
offline W&B vkf33zkl. An earlier placement of the same helper in the benchmark
also passed 189 tests and all rows (F32 4.97101 seconds, offline 8wtmghh6), retained
in outputs/benchmarks/2026-09-13_grouped_target_loading/. Moving the helper to its
own provider motivated the final-source repeat; earlier sources are archived.
Final run-bound sources were rehashed against receipts and copied to after/.
Native SHA: 6995d093694fda0e54c85926aa94bd3f3aeab58019c4121427b54dd2b8559cf5.

The STAR benchmark contains substantial preexisting streaming/frozen work that
is not cleanly separable from HEAD. Its two owned hookup hunks are retained in
research_experiments/paper_runner_suite/grouped_frozen_target_loading.patch.
They apply to the retained before/multicam_heldout_compare.py, not clean STAR
HEAD. git apply --check plus an actual temporary apply and exact comparison
passed; patch_identity.json binds preimage, patch and current source hashes.
The live worktree already contains the hook. Do not apply it twice. Root owns
provider, focused test, patch, this note and scoped status edits; unrelated
streaming WIP stays unstaged and STAR HEAD remains f735898. A clean checkout
alone does not reproduce this whole dirty-harness diagnostic.

## Next experiment

Test bounded multi-frame device batching with the same learned world, interval,
CPU cache, numerical tolerances and hard resource guards. The current harness
couples residency to temporal tile size. Separate those two controls for a
causal batching test: keep tile_t=1 and capacity256, report actual chunk sizes
explicitly, and measure replay as well as compiled routes. A throughput gain
would not by itself prove asymptotic sublinear rasterization. Keep the full-world
shape-recording profiler stopped. The broad goal remains active.
