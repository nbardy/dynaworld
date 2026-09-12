# Batched UVT lowering removes the dominant autograd graph cost

## Continuation and scope

The preceding goal turn was progress: STAR f98b288 fixed temporal-adjoint
conditioning on the retained 800-update world and produced a warmed timing
negative. This continuation targets that demonstrated cost while preserving
the frozen checkpoint, cameras, targets, alpha law, tolerances and local
resource contract. One lead, no subagents, sequential accelerator jobs, no
new token budget. The earlier GPU/Kineto record_shapes profiler stays stopped.
The goal remains broad overnight scientific/code iteration, not completion of
this single optimization.

## Change and mechanism

STAR commit 470a36d replaces per-tube coefficient construction in
uvt_tubes_to_projective_trace_cell_atlas with batched tensor operations. Active
membership is transferred to CPU once, retained source rows and active spans
are derived from that boolean mask, and each coefficient/opacity/color/precision
field is gathered and stacked in one graph. Source-id order, per-trace support
padding, centered temporal opacity, centered source depth and spatial affine
depth remain intact. The function loses 51 lines (49 additions, 100 deletions).
No native kernel, API signature, loss or numerical threshold changed.

A separate CPU control uses the exact old function extracted from the retained
source snapshot and the new function against the same saved projected tensors.
It counts the actual autograd graph, applies the same deterministic cotangent
to all seven retained tensor fields, and compares the lowering VJP. This is a
mechanism diagnostic, not a permanent test requiring a particular graph size.

| Measure on the retained projected inputs | Scalar construction | Batched construction |
| --- | ---: | ---: |
| Autograd graph nodes | 84,235 | 65 |
| CPU lowering VJP, single shot | 0.23255 s | 0.000399 s |
| CPU full atlas construction, single shot | 1.7132 s | 1.5356 s |

All coefficient tensors, cells, source ids and active spans are exactly equal.
Five of six input-gradient tensors are exact; the q_uvt normalized difference
is 1.433e-8. CPU timings are contextual; native renderer timing below is the
real performance control. The graph reduction explains why repeated native
chunk backward previously had to traverse tens of thousands of tiny nodes.

## Actual Metal result

The same 2048-tube legacy world trained for 800 updates is loaded without an
optimizer. It uses Coffee Martini cam06, selected frame indices [0,10,21,31]
over the full 32-frame interval, 96x128 images, one resident target frame, and
one warmup plus three measured alternating paired replay/compiled trials.

| Warmed median segment | Before | After |
| --- | ---: | ---: |
| Atlas construction including world projection | 4.23359 s | 1.67010 s |
| Compiled forward | 0.37468 s | 0.31689 s |
| Compiled backward | 8.62019 s | 0.25802 s |
| Compiled forward + backward | 8.97529 s | 0.57491 s |
| Compiled construction + forward + backward | 13.26454 s | 2.24501 s |
| Replay forward + backward | 0.07841 s | 0.04544 s |

The observed before/after factors are 33.4x for compiled backward, 15.6x for
compiled forward/backward, and 5.91x including construction. These are separate
processes and replay also improved by 1.73x; do not attribute every timing
factor solely to this code change or call it a kernel-only isolated speedup.
The exact graph/value control provides separate mechanism evidence. All three
post-change compiled forward/backward samples are 0.56485/0.57491/0.60286 s;
all three replay samples are 0.04496/0.04585/0.04544 s. No samples were dropped.
The compiler still loses to replay by about 12.7x without construction and
49.4x with construction at this frame count. This does not establish sublinear
scaling or speed superiority over replay.

All eight main correctness checks and nine same-parent slice checks pass at
the unchanged 1e-5 tolerances. Maximum RGB difference from replay is 8.34465e-7,
global world-gradient normalized error 2.26582e-6, maximum per-parameter error
1.70109e-6, and raw_lambda_t error 3.22180e-7. Slicing images/losses remain exact,
with maximum per-parameter gradient difference 6.38715e-7. Fallback is zero.

Most strongly, the serialized atlas is byte-for-byte identical to the prior
centered result: 2,900,089 bytes, SHA
caa194e6235c1de7020803eaaf2b9b8ed2b8a812e67d416a4c0f0bb136c4cbb8.
It still has 1958 active traces and 1853 cells. The shared checkpoint and all
camera/target/evaluation-contract hashes match. All common bound source files
except this producer have identical hashes. This is an implementation change
to the same lowered representation, not a different or smaller fitted scene.

## Remaining cost: bounded CPU profile

A small CPU-only cProfile run on saved projected tensors completes under the
same resource guard. It profiles Python metadata construction with no GPU,
Kineto or tensor-shape recorder. Profiling adds overhead; its times are for
attribution and are not substituted into the Metal benchmark.

Total profiled CPU time is 2.2554 s across 5,024,072 calls. Visibility event
stratification accounts for 1.6616 s cumulative (0.9614 s self); support-event
rebinning takes 0.4005 s and fallback marking 0.1926 s. The midpoint depth helper
runs 49,668 times and costs 0.3682 s cumulative. There are approximately
1.495 million calls each to min and max, while actual pair-depth-root solving
runs only 3,710 times and costs 0.0198 s.

Source inspection locates the pairwise nested loop: it computes active overlap
for every pair even when the containing cell has fewer than two samples, then
immediately skips that pair. A future control can skip those impossible-root
loops at cell level. Midpoint sorting also evaluates small CPU tensors per
trace rather than caching depth values by sample span. Preserve float32 order,
source-id ties, continuous root boundaries, exact-root singleton intervals and
all general spatial-depth cases when testing either change. This profile
changes the next action; no further visibility optimization is claimed here.

## Verification, artifacts and resources

Focused CPU gates passed 80 tests with 23 MPS skips, and the guarded actual
Metal depth/support/temporal gate passed all 43 cases. Four new CPU regressions
ensure empty input and entirely filtered scenes still render black with both
polynomial and centered temporal encodings. Existing moving/filtering,
anisotropic, depth, fallback, cached-update, feature and slicing tests remain
part of the focused gate. The fitted-world report exercises the actual native
forward/backward after the source change. No shader rebuild was needed.

All outputs are in outputs/benchmarks/2026-09-13_batched_uvt_lowering/:
projective_trace.before.py binds the scalar control to the previous run;
report.json and its binary atlas retain the new actual measurement;
compare_lowering_graphs.py retains old/new raw coefficient values and gradients;
metadata_cpu.prof and metadata_cpu_profile.json retain the Python profile;
validate_results.py applies the existing independent storage, timing, route-
memory and slicing validators and records exact before/after identities in
report_validation.json. No new scientific verifier contract was introduced.
The retained scripts and launcher have source-hash sidecars and resource receipts.

The run reuses src/train_configs/frozen_world_quality_timing_control_20260913.jsonc,
changing only output directory and W&B tags/name in its saved evaluation script.
From repo root, run PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:. WANDB_MODE=offline
.venv/bin/python outputs/benchmarks/2026-09-13_batched_uvt_lowering/launch.py
evaluate_saved_world.py. The same guarded launcher accepts compare_lowering_graphs.py
or profile_metadata_cpu.py for the CPU diagnostics; run jobs sequentially.

Offline W&B is yu2nbadb. Mechanical tests and CPU mechanism/profile diagnostics
omit W&B because they make no training or native benchmark claim. All jobs are
confirmed terminal. Peak tree/launcher RSS for the warmed Metal run falls from
2,657,337,344 to 1,245,200,384 bytes. CPU graph/profile peaks are 363,266,048 and
257,982,464 bytes. All receipts show zero new swap and no tripped resource guard;
3-GiB process, 2-GiB MPS allocator and 600-second limits remain unchanged on the
physical 24-GiB host. No applications were killed and no online upload occurred.

The retained world quality stays 20.84/15.30 dB train/heldout. Public evidence
counts remain 0/7 contexts and 0/21 lanes; no full frozen scaling sweep is now
complete. BASELINES standings stay unchanged. The STAR commit owns only the
producer; the root integration owns the four empty-scene tests and scoped
notes/status updates. All unrelated benchmark/root WIP remains unstaged.
