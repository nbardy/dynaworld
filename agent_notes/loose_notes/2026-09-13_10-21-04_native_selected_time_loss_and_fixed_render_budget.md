# Selected-time losses: a fixed native render budget is a strong finite-grid control

The previous turn was progress: commit c2835b6 established three native batching
layouts on the fitted world and changed the static-camera comparator. This
turn extends that same runner to selected losses over F=4/8/16/32, preserving
all target times and charging unused native images. Eight routes (four pairs)
complete and pass independent verification. No existing renderer/native code
changed. One lead, no subagents; sequential jobs and unchanged resource guards.

## Experiment and implementation

Same 800-update, 2048-tube Coffee Martini world, static cam06, 96x128,
peak_splat, tile_t=1, capacity 256, direct_atomic/index_add. Checkpoint SHA
is da6eac63e2497a47fe7bcfa5be645b7b665199c318e17384b31cbf493f4e0c77.
The selected integer times match the existing full-interval contract exactly:
F4=[0,10,21,31], F8=[0,4,9,13,18,22,27,31], F16=[0,2,4,6,8,10,12,14,
17,19,21,23,25,27,29,31], and F32 is the whole integer lattice.

Batch1 projects/renders each requested time. Batch32 projects once and renders
all 32 native frames; a tensor selector sends only requested frames into the
same globally normalized robust-L1 loss. Each actual render records its frame
IDs, RGB element count and projection call. Counters are retained for both
correctness and every timing trial. Native batch32 always materializes
1,179,648 RGB values (4,718,592 float32 bytes), even when F=4. The two routes
retain the same CPU target LRU 8 and request/decode limit 8; neither materializes
a full target video tensor. The native route's output/residual allocations
are included in sampled memory.

The runner now accepts a frame-count sweep via the checked-in config and
retains one report and offline W&B run per F. Its historical single-F config
remains supported. Report-preservation guards reject existing output reports
before loading the model; a direct call verified the original report SHA is
unchanged after the expected FileExistsError. No renderer behavior changed.
The previous raw-tensor checker was extended and promoted to the canonical
research_experiments/paper_runner_suite/verify_native_uvt_control.py; it is
not a competing verifier with different tolerances. Historical verifier
snapshots remain part of their original run artifacts.

## Measurements

All values below are milliseconds, medians of three alternating route trials
after one warmup per route. F values run sequentially, not randomized. E+B
includes host projection/bin/render and actual backward; full also includes
CPU target loading, transfer, selection and robust-L1 construction. Selection
is charged to the loss phase. Startup, correctness copies/hashes, W&B logging,
inter-segment cleanup and optimizer updates are outside these phase sums.

| Selected F | Replay E+B | Native32 E+B | Replay full | Native32 full | Offline W&B |
| ---: | ---: | ---: | ---: | ---: | --- |
| 4 | 35.13 | 48.07 | 37.46 | 48.88 | j0qlwycs |
| 8 | 65.01 | 42.55 | 69.73 | 43.59 | y2knefe4 |
| 16 | 162.57 | 60.45 | 769.61 | 663.09 | dhavo0h0 |
| 32 | 332.40 | 63.98 | 1471.27 | 1189.84 | 8bmiezac |

Over-rendering loses at F4 (48.07 vs 35.13 ms E+B); it wins at F8/F16/F32 by
1.53x/2.69x/5.20x E+B. Preserve the negative F4 outcome rather than claiming
native batching always wins. The full timings jump beyond F8 because the
selected target set no longer fits the eight-frame CPU cache; loading costs
~0.60 s at F16 and ~1.11 s at F32. At F4/F8 the warmed target set fits in cache.

For separately measured interval-atlas context, the latest capacity-safe
F4/chunk2 is 124.73 ms E+B / 418.48 ms full; F32/chunk4 is 1187.09 ms E+B /
4181.26 ms full. Their accepted report hashes and exact target/camera hashes
match this control. They were not rerun in the same timing process; do not
present those ratios as a randomized direct comparison or compare against an
invented F8/F16 optimized interval row. The native-vs-replay pairs above are
measured together, with identical selected losses and the same frozen world.

## Independent acceptance and resource evidence

The canonical verifier reconstructs selected times independently, reloads
seven-frame CPU chunks, recomputes image and robust-L1 metrics in NumPy
float64, checks all seven parameter shapes/coverage/nonzero gradients, and
recomputes normalized VJP differences. All selected outputs are bit-identical
between the two routes and to selection from the preceding full-native RGB
artifact. Global normalized gradient error peaks at 7.52151182e-6 and maximum
per-parameter error at 5.73887744e-6, within unchanged 1e-5 limits. These are
one correctness VJP per route, not nondeterministic-reduction stability bounds.

The checker also validates target and camera hashes against the prior accepted
frozen interval sweep, native/checkpoint identity, world nonmutation, recorded
rendered work in all trials, phase arithmetic, CPU cache accounting, MPS sample
counts, source archives, process receipts and offline W&B artifact backing.
Camera reconstruction uses Metal arithmetic for exact identity, as established
last turn. All run-bound and verifier-bound source sets are archived in after/.

Peak tree-plus-launcher RSS is 1,146,732,544 bytes (1.068 GiB); independent
verification peaks at 848,576,512 bytes. Both have zero new swap and no guard
trip. The 3-GiB RSS, 2-GiB MPS allocator, host/swap/disk and 600-second limits
are unchanged. This is the physical 24-GiB Mac, not physical 8-GiB hardware.
All GPU processes started this turn are terminal. No upload or native rebuild.

An optional plot helper could not import matplotlib from the local .venv;
that plotting lane stopped without installing packages. No figure is claimed.
The tables and raw JSON remain the evidence. The actual eight-route Metal run
and independent artifact checker exercise the changed runtime call graph;
no broad renderer regression suite was rerun for this runner-only extension.

## Mathematical interpretation and next action

For fixed selected indices J, let S_J select rows from the full native image
sequence I(theta). The loss is L_J=ell(S_J I(theta),Y_J). Ordinary chain rule
within the renderer's differentiable stratum gives

    grad_theta L_J = D_theta I(theta)^T S_J^T grad_z ell(z,Y_J).

S_J^T places zeros in unselected image gradients. It does not require uniform
times, a new gauge, an event atlas, or a changed objective. The parity results
check this execution route on the retained fitted world.

Native E+B grows only 1.331x as selected F grows 8x, but the underlying rendered
lattice R stays 32 throughout. Nearly flat selected-loss cost is therefore
available from an ordinary fixed-R baseline. It does not establish an
asymptotic sublinear-in-F renderer, nor attribute an advantage to the interval
compiler. Report requested F, rendered R, primitive count, output/residual
storage and complete cost separately. When the underlying time grid or output
count grows, this baseline must pay for those additional samples.

The static affine experiment now has a useful baseline and a crossover, and
the interval atlas has not earned its cost on this case. Stop repeating tiny
static compiler optimizations without a new measured need. Return next to the
independent convergence question: the successful source fitter uses full-clip
MSE while the shared-world trainer uses sampled robust L1. Isolate that
photometric objective at fixed world parameterization, initialization, sample
schedule, update budget and evaluator before another capacity sweep. That
control has not been run here; it must preserve heldout-as-development-view
semantics and not quietly alter every other protocol setting.

Artifacts: outputs/benchmarks/2026-09-13_native_uvt_selected_times/.
Config: src/train_configs/frozen_world_native_selected_times_20260913.jsonc.
The retained launch.py invokes evaluate_native_batches.py; launch_verification.py
invokes validate_results.py, which calls the canonical verifier. Run from repo
root with PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:. .venv/bin/python,
using fresh output paths for reruns. Prior run artifacts remain preserved.

Selected-loss pairs advance 0/4->4/4 (eight verified routes). Earlier contiguous
batch results, interval 5-pass/3-capacity-negative layouts and frozen-density 4/4
checks remain retained. Public counts stay 0/7 contexts and 0/21 lanes; no PSNR
or publication claim changes. BASELINES gets dated local rows; the canonical
paper work model now distinguishes selected F from the rendered lattice R.
The broad overnight goal remains active.
