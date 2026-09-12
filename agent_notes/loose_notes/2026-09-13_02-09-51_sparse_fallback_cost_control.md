# Sparse fallback reduces rendering cost without changing the saved atlas

The preceding turn was progress: it fixed whole-interval fallback and repeated
native slots, passing the saved F4/F3 local gates. This continuation addresses
the demonstrated cost of rendering the complete Torch reference before patching
only fallback tiles. STAR `a7ec561` is the tested production change. The broad
iteration goal and public evidence requirements remain open.

## Correction and behavioral contract

`render_projective_trace_cell_atlas_reference` has an optional
`fallback_tiles_only` mode. It gathers contributions from the complete atlas,
then skips ordering/compositing for tile samples without a fallback flag.
`ProjectiveCellIntervalTrainerState.render_mixed_fallback` uses that mode.
Other reference calls retain the old complete-image behavior by default.
The reference tensor retains its output shape; unused tiles are zero and the
native result supplies them. This removes unnecessary compositing and graph
construction, not the dense output-size lower bound or all metadata work.

A flagged tile must include every contributing trace, including cells that
were not themselves flagged. A new CPU/MPS regression puts an unflagged green
trace behind a flagged red one and a blue trace in another tile. It checks the
selected image against the full reference masked at the output, preserves the
green contributor's gradient, and gives the unrelated blue trace zero sparse
reference gradient. The actual mixed Metal renderer restores the complete
image and all parameter gradients. Existing centered-source-depth, stable-id,
real-gap, inherited-fallback, and cached-update checks remain passing.

The source change is 11 insertions and 3 deletions across the reference renderer
and trainer bridge. There was no native build, shader change, model update,
new alpha law, tolerance relaxation, or support/visibility approximation.

## Before/after local Metal control

`src/train_configs/frozen_fallback_cost_20260913.jsonc` fixes one warmup and
three retained timing trials. Each uses the existing device-synchronized,
alternating paired replay/compiled timing implementation. Compilation includes
world projection; forward includes target transfer; backward excludes optimizer
work. Correctness passes and inter-segment cleanup are outside timed segments.

Both processes strict-load the same four-frame world checkpoint (256 tubes,
96x128, cam06, peak-splat, capacity128, resident two-frame chunks). Checkpoint,
world/camera/target/loss contract hashes, frame/time selection, thresholds,
atlas statistics, and the actual serialized atlas SHA are unchanged. Existing
timing and retained-storage validators pass. This is a local implementation
cost control on one small world, not the full clean-source scaling experiment.

| Median seconds | Before | Sparse fallback | Before / after |
| --- | ---: | ---: | ---: |
| Compiled forward | 5.68597 | 2.12509 | 2.68x |
| Compiled backward | 5.07568 | 1.80433 | 2.81x |
| Compiled forward + backward | 10.76165 | 3.90343 | 2.76x |
| Compilation | 7.98407 | 8.01556 | 1.00x |
| Compilation + forward + backward | 18.74572 | 11.90172 | 1.58x |
| Replay forward + backward | 0.04441 | 0.03680 | 1.21x |

Medians of summed samples need not equal sums of component medians. The three
compiled forward/backward totals are [10.7110,10.8732,10.7616] s before and
[3.9034,3.9745,3.8774] s after. Replay also sped up between processes, so do not
treat the before/after ratio as a hardware-independent constant. The rendered
work reduction, unchanged atlas, and separated phase timings support this
local improvement. The compiled route remains substantially slower than replay
on this tiny case; this is not evidence of sublinear end-to-end runtime.

Offline W&B runs are `sr5un7i8` (before) and `ahjazger` (after), with resolved
config, tags, numerical results, report files, and timing samples retained.
There was no new training or image-quality improvement in this control.

F4 image maximum error remains 6.8545341e-7; after-change global world-VJP error
is 7.5579362e-7 and maximum parameter-group error 7.2330414e-7. Fallback stays
12.7778%, all eight local checks pass, and the retained F4 atlas is byte-identical
(SHA 4c4379490011322f187c70dfe3639cf86ff0bcd313b8ff66c2ca0495375007ea).
The F3 [0,2,3] check also passes all gates at 17.0370% fallback, max RGB
6.8545341e-7, global VJP 1.0089654e-6, max parameter VJP 1.1288501e-6.
Non-unit chunk slicing produces identical RGB, global VJP error 4.2225722e-7,
and max parameter error 5.1701967e-7. The existing independent slice validator
accepts it, and every learned-world identity remains unchanged.

## Resources, tests, and artifacts

The focused CPU suite passed 63 tests with 33 MPS-dependent skips. The guarded
actual Metal depth/fallback suite passed all 16 tests. Before/after benchmark
peak process-tree plus launcher RSS was 2,006,794,240 / 1,553,219,584 bytes.
Every process completed with zero new swap and no resource guard trip. The
3-GiB RSS, 2-GiB MPS, host reserve, disk, and 600-second bounds were unchanged.
No application was killed, no online W&B upload occurred, and no agent was
spawned. Mechanical regression/slicing/profile checks omit W&B; the actual
repeated cost control logs to offline W&B.

`outputs/benchmarks/2026-09-13_sparse_fallback/` retains `before/` and `after/`
launchers, scientific scripts, source hashes, reports, atlas binaries, W&B
runs, logs, and resource receipts. `comparison.json` checks the common inputs
and records existing-validator results and raw timing distributions. The
changed files were checked against the source hashes after execution. The
baseline source hashes were checked before editing the renderer.

## Profiled next cost

One additional guarded actual MPS compilation was instrumented with cProfile.
It took 8.3345 s including profiler overhead; these values are for attribution,
not a separate speed benchmark:

- visibility fallback marking: 7.2376 s cumulative;
- UV visibility event report: 5.7001 s;
- depth-line evaluation: 274,620 calls, 4.4974 s cumulative;
- per-tile corner depth ranges: 17,128 calls, 1.4817 s;
- time-event stratification: 0.5711 s.

The prior verified saved-atlas inspection establishes all spatial depth slopes
are zero in this same byte-identical atlas. It therefore has z(u,v,t)=z(t):
checking spatial order boundaries and four identical corner depths is redundant.
An exact zero-spatial-slope specialization is the next target. Preserve scalar
near-depth fallback, invalid-data checks, centered source-depth/tie semantics,
and the general spatially varying path. Do not discard a nonzero polynomial
coefficient merely because its value happens to vanish at one sampled time.
Use retained native RGB/all-world-VJP evidence to validate that specialization.

No paper context or lane count changes: 0/7 and 0/21 remain. BASELINES is
unchanged. The successful source fits and newer 19.69/14.96 dB quality controls
remain separate from this renderer cost result. The goal stays active.
