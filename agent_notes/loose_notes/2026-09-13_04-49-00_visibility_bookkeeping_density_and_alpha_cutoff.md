# Visibility bookkeeping improves compilation; the density sweep exposes a cutoff discontinuity

## Continuation and scope

The immediately preceding user-answer turn clarified historical source fits;
it added no experimental evidence count and is no progress by that narrower
goal criterion. This continuation revalidated the pending sweep using exact
session 42470, which returned terminal exit zero. Its actual saved four rows
and resource receipt are authoritative. The earlier optimization work was
uncommitted and its completed tests/results were checked before integration.
One lead, no subagents, sequential accelerator jobs, unchanged resource guards,
offline logging and no newly specified token budget. The broad overnight goal
remains active. The full-world Kineto/record_shapes lane stays stopped.

## Source change and matched F4 evidence

STAR commit e1c33c1 changes only visibility-event bookkeeping in
stratify_projective_trace_cell_atlas_visibility_events. It precomputes each
trace's active span intersected with the containing cell and excludes spans
shorter than two samples from continuous pair-root searches. This preserves
the old pair condition: an intersection cannot contain two samples unless
both member spans do. Singleton traces still participate in output cells,
depth sorting and compositing. Actual pair-root calls remain exactly 3,710
on the fitted F4 input.

Midpoint depth evaluation is batched per output cell. It retains float32
arithmetic order, including (z2*t)*t rather than z2*(t*t), and the original
trace-id tie ordering. Root solving, support bounds, fallback policy, native
shaders, alpha law, world parameters and tolerances are unchanged.

The new behavioral regression combines continuous crossings, exact ties and
single-sample contributors on nonuniform times. Its first reference fixture
incorrectly kept every trace active in one unsplit reference cell. The
renderer trusts cell membership; the oracle therefore rendered nonexistent
contributors. The fixture was corrected to declare each trace's active span
and use live-depth reference sorting. No production change or tolerance
relaxation was made to accommodate that failed test. Both logs are retained.

The corrected focused CPU gate passes 99 tests with 23 MPS skips. The actual
guarded Metal depth/visibility gate passes all 62 tests. The fitted F4 report
passes all eight main and nine slice checks at unchanged 1e-5 tolerances:
max RGB error 8.34465e-7; max per-parameter world-gradient error 1.64077e-6;
zero fallback. The serialized atlas is byte-identical to the previous run:
2,900,089 bytes, SHA caa194e6235c1de7020803eaaf2b9b8ed2b8a812e67d416a4c0f0bb136c4cbb8.

One warmup and three measured alternating paired trials give these medians:

| Segment | Prior batched lowering | Visibility bookkeeping |
| --- | ---: | ---: |
| Compilation including projection | 1.67010 s | 0.97880 s |
| Compiled forward | 0.31689 s | 0.30388 s |
| Compiled backward | 0.25802 s | 0.25815 s |
| Compiled forward + backward | 0.57491 s | 0.57076 s |
| Compilation + forward + backward | 2.24501 s | 1.55038 s |
| Replay forward + backward | 0.04544 s | 0.04778 s |

These are separate-process local measurements. Exact atlas identity and the
CPU profile support the construction-cost attribution; they do not establish
a renderer speedup over replay. Compilation falls by about 41.4% (1.71x).
The bounded Python-only profile falls 2.2554 -> 1.1534 s, with total calls
5,024,072 -> 2,029,429. Visibility stratification falls 1.6616 -> 0.5657 s
cumulative; support rebinning and fallback marking remain around 0.3975 and
0.1895 s. The 3,710 actual root solves still take only about 0.0194 s.

Artifacts: outputs/benchmarks/2026-09-13_visibility_bookkeeping/ contains the
pre-edit source, CPU and Metal logs, warm report, exact binary atlas, Python
profile, source hashes, receipts and report_validation.json. Its existing
validate_results.py applies the independent storage/timing/memory/slicing
validators and compares checkpoint, camera/target contracts and exact atlas.
Offline W&B: 204zxu3g. Warm-run peak tree/launcher RSS: 1,186,545,664 bytes.

## Fixed-world local density curve

The new checked-in frozen_world_local_density_control_20260913.jsonc retains
the same 800-update, 2048-tube learned world, Coffee Martini cam06 and 96x128
images. Each F={4,8,16,32} selection spans the same 32-frame physical interval;
the sampled integer sets need not be nested. Resident render/target chunks
stay at one frame, with the same eight-frame CPU target LRU. Each row has one
warmup and three measured alternating paired repetitions. There is no fitting
or change in the world as F changes.

| F | Compile s | Compiled forward s | Compiled backward s | Compiled fwd+bwd s | Replay fwd+bwd s | Retained atlas bytes | Checks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4 | 0.98195 | 0.31779 | 0.26420 | 0.57802 | 0.04412 | 2,900,089 | pass |
| 8 | 2.52755 | 0.73313 | 0.63685 | 1.38490 | 0.07615 | 6,486,712 | pass |
| 16 | 5.02036 | 4.88385 | 1.45287 | 6.34042 | 3.41366 | 12,470,567 | pass |
| 32 | 8.04311 | 10.60139 | 2.77363 | 13.37502 | 6.84725 | 20,357,803 | image fails |

Sum-of-segment medians need not equal the median measured sum. The table uses
each retained summary, with no omitted trials. All rows have zero fallback.
Maximum per-parameter gradient errors are 1.7841e-6, 1.1273e-6, 9.0287e-7 and
9.7061e-7 respectively. Only F4 runs the separate selected-time slice gate,
as specified by the existing sweep helper, and that gate passes.

For eightfold denser output, compilation grows 8.19x, backward 10.50x,
and retained atlas storage 7.02x. Cells grow 1,853 -> 48,753 (26.31x), while
interval trace entries grow 49,668 -> 272,274 and dense trace samples grow
51,016 -> 519,505. The interval/dense ratio improves 0.974 -> 0.524; this
does not remove the growth in actual topology, traversal, or output work.
Trace coefficients alone stabilize once all 2048 tubes are active at F8,
but the serialized evaluator includes much more than coefficients.
This local curve does not establish sublinear end-to-end differentiable
rasterization. It remains slower than replay, and the full curve also has a
failed numerical gate. No asymptotic exponent is established from four rows.

The existing independent logical-payload, storage, timing and route-memory
validators pass for every retained row, including the failed numerical row.
All row/artifact hashes, fixed checkpoint, selected-time schedules, source
hashes and retained offline W&B report were checked. report_validation.json
preserves all_rows_accepted=false, with only image_matches failing at F32.
Local accepted rows are now 3/4; this is not a full public-protocol sweep.
Public evidence counts and BASELINES standings remain unchanged.

Artifacts: outputs/benchmarks/2026-09-13_fitted_world_density_sweep/ contains
the final report, incrementally retained rows and atlases, progress, source
identities, launcher, validator application and receipt. Offline W&B: 82zpsgqs.
Maximum tree/launcher RSS is 2,080,866,304 bytes, with zero new swap and no
guard trip. The exact pending session returned terminal exit zero; the negative
acceptance is a numerical result, not a crashed or timed-out job.

## Why the forward curve jumps above eight frames

Source inspection shows load_target_chunk inside both forward timers, followed
by host-to-device transfer and loss construction. PaperMulticamTargetProvider
holds eight CPU frames in an LRU. Sequential passes over more than eight
unique frames can evict every next-needed frame before its reuse.

A bounded CPU probe used the same camera source, indices, resolution, one-frame
requests and eight-frame cache, with one cold and two warmed passes per F.
Both warmed F4 passes had four cache hits and zero decodes, taking 0.00051 and
0.00031 s; F8 had eight hits and zero decodes, taking 0.00055 and 0.00051 s.
F16 had zero hits and sixteen decodes on each pass, taking 3.10279/3.07503 s.
F32 had zero hits and thirty-two decodes, taking 6.23887/6.22510 s. Content
hashes match across cold/warm repetitions. Probe timings include target hashing.
These measurements explain most of the shared forward jump. They are not
subtracted from separate-process end-to-end timings to invent renderer-only
results. The next performance report must expose target-loading cost explicitly.

## F32 failure: one pixel crosses the opacity cutoff

A guarded no-grad Metal diagnosis rerendered all 32 frames from the exact
checkpoint. It reproduces the saved maximum RGB error 0.0006172060966491699.
Only one pixel exceeds 1e-5: frame 1, y=43, x=44 (all three channels). All other
frames have max error below 9e-7. Evaluating the pre-edit visibility function
on the identical incoming F32 atlas gives exactly the same output cells as
e1c33c1. This error was not introduced by the visibility bookkeeping change.

CPU scalar terms at that pixel locate trace/source id 771. The source evaluates
the original spacetime quadratic; the atlas completes the spatial square and
separates spatial and temporal exponentials. The real-arithmetic expressions
agree, but float32 gives:

| Quantity | Source | Compiled |
| --- | ---: | ---: |
| Quadratic exponent argument | 9.336318969726562 | 9.336320877075195 |
| Alpha | 0.003921571187674999 | 0.003921567928045988 |

The two alphas differ by only about 3.26e-9, but straddle 1/255. Source keeps
the contributor; compiled drops it. CPU recomposition matches each saved Metal
RGB within 6e-8. Keeping compiled continuous values but imposing the source's
inclusion mask lowers max pixel discrepancy to 1.19209e-7. This intervention
isolates the branch decision rather than world fitting, depth order or gradient
noise. Since the omitted tube occludes later contributions, dropping it can
increase final RGB; it is not simply a missing positive-color term.

For a prefix transmittance T, primitive alpha a/color c, and the suffix color S
composited over the same background, including versus omitting that primitive
changes RGB by T*a*(c-S). A tiny error in a near the cutoff can therefore cause
an image jump of order a, rather than order of the alpha rounding error.
Mathematical change-of-variables equality alone does not preserve this discrete
branch. The fixed 1e-5 image gate remains failed; no threshold was loosened.

No production alpha repair was applied in this turn. Next preserve the source
cutoff decision using a source-compatible evaluation/ambiguity path, including
its backward behavior and retained byte cost. First freeze this one-pixel case
as a behavioral regression and check neighboring values on both sides of the
cutoff. Merely raising the image tolerance, nudging opacities, or omitting this
frame would not fix the demonstrated contract. Then rerun the same fixed-world
curve with target-loading cost separately measured.

Diagnostics live in outputs/benchmarks/2026-09-13_f32_parity_diagnostic/:
compare_routes.py and route_tensors.pt retain both images, projected tensors,
atlas tensors/cells and top error coordinates; pixel_terms.py/json retain the
alpha terms and mask intervention; target_cache_probe.py/json retain actual
decode/hit counters and times. diagnostic_validation.json records tensor/source
hash checks, CPU-to-Metal agreement, the intervention and resource receipts.
These mechanical probes do not claim training or renderer speed, and omit W&B.

The initial sandboxed pixel-term launch could not read the host swap probe and
stopped before execution. Its log is retained as .sandbox_probe_failure.log.
An authorized escalation enabled that probe without weakening any resource
threshold. Diagnostic peak tree/launcher RSS: Metal localization 761,004,032;
CPU terms 389,824,512; CPU decode probe 922,435,584 bytes. Every completed job
has zero new swap and no guard trip. All exact session handles are terminal.
The 3-GiB process, 2-GiB MPS and 600-second guards remain unchanged.

The root integration owns the single regression, local density config, this
note and scoped status updates. All unrelated source/browser/paper WIP is
preserved. The successful source overfit and latest 20.84/15.30-dB world fit
remain valid quality evidence, separate from this failed F32 compiler check.
