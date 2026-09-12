# Centered temporal gradients pass on the fitted world; warmed cost remains high

## Continuation and evidence target

The previous user-facing turn clarified the historical source fits; it did not
advance scientific evidence beyond the retained results. This continuation
resumed the pending centered-temporal patch and produced new fitted-world
correctness, identical-backward, and warmed timing evidence. One lead, no
subagents, sequential accelerator jobs, unchanged resource gates, no new token
budget. The 41-case guarded Metal regression that was running before the
interruption was confirmed terminal and passing before another job launched.

The same 800-update legacy world remains frozen: Coffee Martini, 2048 tubes,
32 dataset frames, train cam04/cam09 and heldout validation cam06. Its retained
quality is still 20.84216/15.30286 dB train/heldout. No optimizer ran in these
checks. The historical source overfits are not invalidated by these compiler
measurements. The selected frame indices are [0,10,21,31], with centered times
[-15.5,-5.5,5.5,15.5] and one resident target frame.

## Numerical cause and repair

The previous lowering stored a temporal envelope as
q(t) = c0 + c1*t + c2*t^2, with (c0,c1,c2) = (lambda*t0^2,-2*lambda*t0,lambda).
Its world-lambda derivative reconstructed t0^2*g0 - 2*t0*g1 + g2 after separate
float32 native accumulations. Retained gradients showed substantial cancellation:
median per-trace cancellation ratio about 39.7, p90 about 2829, and p99 about
210,000 among non-negligible results. Recontracting the retained coefficient
gradients in float64 changed the projected temporal derivative only about
1.1e-7 to 1.5e-7 relative, so final-contraction precision alone did not repair
upstream rounding. The unweighted derivative and raw_lambda_t have different
conditioning; do not confuse their reported relative errors.

STAR commit f98b288 adds an opt-in centered encoding in the same [N,3] buffer:
q(t) = c0 + c2*(t-c1)^2, with producer entries (0,t0,lambda). The native
coefficient derivative is grad_q * (1,-2*lambda*(t-t0),(t-t0)^2). Accumulating
in this basis avoids recovering lambda by subtracting large global-time terms.
This is an algebraically equivalent numerical formulation, not new gauge theory.
It does not eliminate all float32 error or make the spatial polynomial stable
under arbitrarily large time-origin shifts.

The frozen-world compiler opts into centered mode at all three producer sites
(main check, slice check, timing trial). Default producers and family kernels
retain their old polynomial semantics. The flag is carried through reference,
interval forward/row rendering, native backward, cached live UVT updates,
refresh, frame slicing and detached copies. Native cell metadata uses its
previously unused reserved0 slot; family kernels explicitly keep their own
existing reserved0 semantics. No C++ ABI or tensor buffer changed. The Metal
source is compiled at runtime and was exercised by actual Metal tests.

The serialized atlas hashes the encoding flag. Existing polynomial artifacts
remain valid. Compared with the old negative fitted-world report, all topology
except this flag and all six non-temporal tensor records are byte-identical.
Only the temporal coefficient tensor changes. Metadata adds 29 bytes, making
retained atlas size 2,900,089 bytes. The independent existing storage verifier
accepts both encodings; two CPU regressions protect preservation of the flag.

## Fitted-world correctness and repeatability

| Check | Old expanded encoding | Centered encoding | Gate |
| --- | ---: | ---: | ---: |
| Maximum image difference from replay | 1.400709e-6 | 8.344650e-7 | 1e-5 |
| Global world-gradient normalized difference | 4.262453e-6 | 2.256811e-6 | 1e-5 |
| raw_lambda_t normalized difference | 3.235681e-5 | 3.141388e-7 | 1e-5 |
| Maximum per-parameter normalized difference | 3.235681e-5 | 1.695002e-6 | 1e-5 |
| Same-parent slicing maximum per-parameter difference | 1.677275e-5 | 8.197792e-7 | 1e-5 |
| Fallback fraction | 0 | 0 | <=0.2 |

All eight main checks and all nine slice checks pass. Same-parent images and
losses are exact. The atlas still has 1958 active traces and 1853 cells; its
interval/dense trace-sample ratio is 0.97358. Checkpoint, target, camera,
evaluation-contract hashes and every tolerance match the previous negative.
This is accepted local F4 correctness, not a completed scaling sweep or paper.

Three additional identical backward passes reused one parent atlas and loss.
All images/losses are exact. The three pairwise raw_lambda_t relative differences
are 1.2278e-7, 1.2613e-7 and 1.2623e-7, versus 9.7363e-6, 1.0358e-5 and
1.2180e-5 previously. Every parameter's pairwise difference is below 8.15e-7.
All pairs and raw gradients are retained; no favorable-repeat selection or
threshold relaxation was used. Atomic backward remains nondeterministic.

## Warmed timing: a clear remaining negative

A separate process executes the existing alternating paired timing routine,
with one warmup and three measured repetitions on the identical fixed world,
four times, source code and atlas bytes. Its independent correctness/slicing
checks also pass (maximum per-parameter error 1.6134e-6). The measured medians:

| Segment | Seconds |
| --- | ---: |
| Replay forward + backward | 0.078415 |
| Compiled atlas construction including projection | 4.233592 |
| Compiled forward | 0.374682 |
| Compiled backward | 8.620190 |
| Compiled forward + backward | 8.975295 |
| Compiled construction + forward + backward | 13.264536 |

Replay total samples are 0.0784/0.0655/0.1696 s; compiled forward/backward
samples are 8.9753/9.1429/8.5851 s. Retain the slow replay sample too. Summed
medians need not equal the median of summed trial segments. Warmup does not
remove the large cost. These are synchronized segments excluding optimizer
and inter-segment cleanup. The report's timing publication_ready flag means
its minimum repetition contract passes, not that this one local F4 diagnostic
completes public-scene evidence or supports sublinear scaling.

Source inspection finds a concrete next candidate: the UVT producer loops over
all 2048 tubes, performs per-row active-index readbacks and many tiny tensor
stacks, then concatenates them into one atlas. Each resident-chunk backward
traverses this large shared lowering graph. Test batched coefficient construction
or the existing vectorized cached live updater against the exact retained atlas
and world gradients, then repeat this same timing control. This is a candidate
bottleneck, not profiler-confirmed attribution. The earlier record_shapes
profiler lane remains stopped after its resource failure.

## Artifacts, checks, resources and reproduction

All centered correctness/repeatability artifacts live under
outputs/benchmarks/2026-09-13_temporal_adjoint/. The warmed comparison is under
outputs/benchmarks/2026-09-13_frozen_quality_warm_timing/. Both retain guarded
launchers, exact evaluation scripts, source hashes, reports, serialized atlases,
resource receipts, logs and offline W&B. validate_results.py in the first
folder applies the existing independent storage, slicing and timing validators
to both reports and writes report_validation.json beside each. No new
scientific acceptance contract was introduced.

Configs are src/train_configs/frozen_world_quality_control_20260913.jsonc and
src/train_configs/frozen_world_quality_timing_control_20260913.jsonc. The first
centered evaluation changes only its output path/tags in the retained local
script. The timing config changes output/tags and warmups/repeats, retaining
all scientific data/checkpoint/tolerance settings. From the repo root, use
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:. WANDB_MODE=offline .venv/bin/python
with the desired folder's launch.py evaluate_saved_world.py. The first folder's
launch.py gradient_repeatability.py runs the three-pass control. Launchers
check live resources before starting and enforce the existing 600-second bound.

The focused pre-existing CPU gates passed 74 tests with 23 MPS skips; actual
Metal depth/support/temporal gates passed 41. The new time-translation cases
cover CPU/MPS, origin 0/8192, native/fallback, cached updates, and frame slicing
against an analytic centered Gaussian value and center/precision/opacity/color
adjoint. The two additional CPU retained-encoding cases passed. Mechanical
tests omit W&B. The first CPU storage-test invocation hit sandbox access to
the uv cache before collecting tests; the authorized cached-runtime retry
passed and both logs are retained. No test threshold was changed.

Offline W&B: fitted-world correctness tpv64jg8, identical backward 4fv80w4y,
warmed timing fyqtqtbf. Peak process-tree/launcher RSS was 2,434,826,240 bytes,
2,232,565,760 bytes and 2,657,337,344 bytes respectively; all show zero new swap,
no tripped guard, and unchanged 3-GiB process / 2-GiB allocator limits on this
physical 24-GiB host. All jobs are terminal. No online upload or remote compute.

Correctness report SHA f07c984df8efd33d1dbebc983f3e1a74fcef23cc0312609f522a0c72f152573c.
Atlas SHA caa194e6235c1de7020803eaaf2b9b8ed2b8a812e67d416a4c0f0bb136c4cbb8.
Raw repeated gradients SHA f76109020e8ff9a444e96ff76a10022f1cf450067bfe6085b7e31bb14e879ef8.
The source commit contains only four scoped benchmark insertions alongside the
owned STAR files; unrelated benchmark WIP remains unstaged. Root integration
contains the cached updater, storage validation, behavioral tests, timing config,
and these scoped notes/status updates. BASELINES and public acceptance counts
remain unchanged at 0/7 contexts and 0/21 lanes. The broad goal remains active.
