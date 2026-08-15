# WorldFoam training-memory ablation: source closeout and evidence boundary

## Decision

The WorldFoam memory-light implementation is now an ablation implementation,
not merely a collection of unit tests.  The tests are preflight gates for a
separate fresh-process experiment.  No paper-scale MPS memory artifact has run,
so the current claim is **source-complete and logically bounded**, not
**empirically memory-fit**.

Do not promote this lane, update `BASELINES.md`, or say that WorldFoam fits the
machine until the guarded `F=8/64/300` artifact verifies.

## What the ablation asks

The fixed workload is:

- image contract `384 x 512`;
- `S=1024` trainable kinetic power sites;
- `P=512` fixed selected camera tracks;
- one fixed 300-frame physical grid, with endpoint-including requested subsets
  `F in {8,64,300}`;
- direct selected-pixel targets, never an eager `H x W x F` target tensor;
- physical P0 Beer--Lambert material;
- trainable material, position, velocity, and weight coefficients;
- one stateless CPU manual-SGD mutation per evidence row.

The matrix has 21 evidence rows executed by 24 **sequential fresh processes**:

1. staged sparse, `F=8`, repeats `0/1/2`;
2. fused union-v2, `F=8/64/300`, repeats `0/1/2`;
3. same-representation compiled-framewise replay, `F=8/64/300`, repeats
   `0/1/2`;
4. three additional auxiliary lifecycle processes for repeat-matched fused
   `F=8`.  Each auxiliary process runs fresh step 1, writes a checkpoint, runs
   uninterrupted step 2, releases that world, restores the checkpoint, and
   replays step 2.  These processes are not scaling rows.

At `F=8`, the artifact must bind both staged-versus-fused parity and
fused-versus-compiled-framewise parity.  The latter is the central systems
control: both lanes compile the same continuous selected-track representation
once; the control replays one frame at a time and accumulates four global bars,
while fused WorldFoam shares the compiled world adjoint across the temporal
slice.

The primary fused `F=8` row is exactly one optimizer step, just like `F=64`
and `F=300`.  Checkpoint creation and both step-2 paths live only in the
auxiliary worker, so they cannot inflate the F=8 process peak or timing and
artificially flatten the scaling curve.  The auxiliary step-1 result is bound
back to the independent primary F=8 row by loss, gradient digest, complete
post-step parameter digest, all five deltas, and a portable update-content
digest.

## Logical state accounting

For `S` sites, the persistent trainable state used by this gate is

```text
material live                    = 48 S bytes
trainable geometry               = 64 S bytes
combined live                    = 112 S bytes
combined checkpoint              = 80 S bytes
live + checkpoint               = 192 S bytes
live + checkpoint clone bound   = 272 S bytes
```

At `S=1024`, these are respectively:

```text
material live                    =  49,152 bytes
trainable geometry               =  65,536 bytes
combined live                    = 114,688 bytes
combined checkpoint              =  81,920 bytes
live + checkpoint                = 196,608 bytes
live + checkpoint clone bound    = 278,528 bytes
```

The per-step global bars are `16,384` bytes for material and `65,536` bytes
for geometry.  Optimizer history is zero only because this gate uses stateless
SGD; material and geometry are never counted as zero.

The compiled-framewise frame-local logical upper bound is now the conservative
live union

```text
coordinator-visible peak
+ overlapping geometry-bridge scratch
+ 64 S bytes of per-frame geometry bars
+ material CPU readback
+ one CPU loss scalar
```

The coordinator component already includes its frame material bar, which is
recorded explicitly to prevent double counting.  An earlier implementation
used `max(coordinator, bridge) + readback`, omitted the geometry bars, and did
not prove the two large components disjoint; that formula was rejected.

Direct target decode is capped by the actual selected observation bound:

```text
4096 observations * 3 channels * 4 bytes = 49,152 bytes
```

An earlier preflight incorrectly treated the entire 64-MiB bridge ceiling as
if it were an allocated target scratch tensor.  The adapter now uses the exact
49,152-byte bound for both decode and selected-target scratch.

## What is frame-bounded and what is allowed to be linear

The expensive world-side tensor working set is bounded by sites, active owner
blocks, selected tracks per request, and observations per chunk.  It does not
retain full frames, predictions, sample tensors, targets, or an autograd graph.
The compiled-framewise control permits at most one live frame.

This is not a claim that every scalar or Python object is independent of `F`.
The camera/time sampling slice and streamed-work receipts are deliberately
cheap linear metadata.  In particular, the primary lane retains tensor-free
geometry reduction receipts per spatial bundle, while the control records
per-frame timing/digest telemetry.  The paper claim must remain:

> dominant world-side raster/backward tensor state is frame-bounded; camera
> slicing, samples, rays, and small receipt telemetry scale linearly with the
> requested observations.

The exact allowed linear work is:

```text
streamed samples                 = 512 F
direct selected observations    = 512 F
camera ray slices               = 512 F
camera ray scalars              = 6 * 512 F
sample-to-node interactions     = 4 * 512 F
```

The bounded observation schedule produces `4`, `8`, and `40` spatial bundles
for `F=8`, `64`, and `300`.  Native launches therefore are not claimed to be
frame-invariant.

## Measurements required for promotion

Each evidence row must come from a fresh process and bind config, acceptance
contract, transitive source manifest, native sources, native extension,
hardware, producer, and driver hashes.  The parent samples the full process
group and terminates above 4 GiB RSS.  The child applies an MPS working-set
ceiling no greater than 2 GiB and samples public MPS current/driver counters.
Public samples are lower bounds; Metal private/register/spill storage is not
invented from logical accounting.

Promotion requires all of the following:

- every required row is measured, finite, and native;
- four gradient families and five parameter families update nontrivially;
- staged/fused and fused/framewise `F=8` parity pass;
- uninterrupted and checkpoint/restart step 2 agree;
- no row crosses the 2-GiB MPS or 4-GiB sampled process-group watchdog;
- measured fused memory growth from `F=8` to `F=300` stays inside the frozen
  absolute and scale thresholds;
- compiler work receipts remain fixed where required and sampled/ray work is
  exactly linear;
- no full-frame target, dense fallback, fake-native, dry-run, or simulated row
  enters the artifact.

## Current implementation status

Implemented:

- the production shared-adjoint primary adapter;
- staged sparse and fused union-v2 routing;
- the fair compile-once/framewise replay control;
- complete material plus geometry CPU-SGD update receipts;
- sparse selected-track checkpoint and live restore;
- uninterrupted-versus-restart lifecycle receipts;
- matched one-step F=8 scaling measurement with all checkpoint/step-2 work
  isolated in the auxiliary process;
- compiler-work aggregation and mode-specific launch accounting;
- fresh-process orchestration, parent RSS watchdog, public MPS sampling, source
  manifests, and fail-closed artifact production;
- staged/fused and fused/framewise `F=8` parity payloads.

The producer evidence-hash loop was also repaired to hash only the 21 row
receipts.  The previous loop indexed auxiliary restart receipts as if they
contained a `row` object and would have crashed after all expensive workers
finished.

Unfinished evidence, not unfinished mathematics:

- rebuild the stale Metal extension against the current bound sources;
- attest its exact exported ABI;
- execute the 24 guarded processes on a quiet host;
- verify the resulting 21-row artifact;
- only then write a baseline/paper result.

The dry plan must emit zero evidence and report only
`native_extension_older_than_bound_native_sources` before the rebuild.

Final safe source verification on this tree:

- `55 passed` across the combined control, adapter/lifecycle, producer,
  verifier, lazy bridge/step, and combined-state focused gate;
- Python compilation and both JSON parses pass;
- dry plan reports 12 primary rows, 9 controls, 3 auxiliary processes, zero
  evidence, and only the stale-native blocker;
- no measured artifact exists under
  `outputs/worldfoam_training_memory_ablation/`.

## Host safety at closeout

At this closeout the local machine had about 11 GiB disk free and
`16.9/17 GiB` swap in use.  The launch contract requires at least 8 GiB
available host memory, at most 2 GiB swap use, and bounded load.  This machine
therefore fails the intended safety preflight, and no Metal build or MPS row
was launched.

The 8-GiB availability requirement is incident headroom.  It is not a 32-GiB
representation requirement and must not be described as one.  A B200 cannot
run this Metal evidence path without a separately implemented and bound CUDA
backend.

## Safe-host handoff

The canonical handoff is now one fail-closed launcher rather than a manually
spliced build/attest/run sequence:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/run_worldfoam_g6_clean_host_bundle.py

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/run_worldfoam_g6_clean_host_bundle.py \
  --execute
```

The first command is a source/config plan with zero subprocesses, zero
Torch/native imports, zero MPS work, zero writes, and zero evidence rows. The
second applies the host guard, force-rebuilds and attests the exact 133-schema
extension using the preserved virtual-environment Python path, checks both G6
ABI seals, runs the 21 evidence rows plus three restart processes, and invokes
the independent verifier. See
`research_experiments/world_foam_lane2/WORLDFOAM_G6_CLEAN_HOST_RUNBOOK.md`.

The preserved executable path is important: resolving `.venv/bin/python` to
the Homebrew framework binary discards the venv and its Torch installation.
The producer now avoids that failure and installs the bound fused-slab variant
directory before native import.

Correction recorded 2026-08-15: an earlier version of the command below named
the legacy `world_foam_lane2_v0` directory.  That was wrong for G6.  The
producer imports `torch_world_foam_lane2_fused_slab.ops` and binds the
`world_foam_lane2_fused_slab_v0` source tree, so rebuilding the legacy variant
would leave the 30-schema stale-binary blocker unchanged.

From the Dynaworld root, after host preflight and a current native rebuild:

```bash
( cd third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld \
    python setup.py build_ext --inplace )

PYTHONDONTWRITEBYTECODE=1 python3 \
  research_experiments/world_foam_lane2/run_worldfoam_training_memory_ablation.py

PYTHONDONTWRITEBYTECODE=1 python3 \
  research_experiments/world_foam_lane2/run_worldfoam_training_memory_ablation.py \
  --execute

PYTHONDONTWRITEBYTECODE=1 python3 \
  research_experiments/world_foam_lane2/verify_worldfoam_training_memory_ablation.py \
  outputs/worldfoam_training_memory_ablation/worldfoam_training_memory_ablation.json
```

The first command is a dry plan only.  It must never create an evidence row.
The second command is the real ablation.  Do not weaken the host guard or
memory thresholds to force it through.

## Falsifiers and stop rules

- If the fused `F=300` measured peak materially grows with `F`, the current
  implementation has not delivered the intended memory behavior even if its
  logical tensors look small.
- If compiled-framewise and fused `F=8` disagree, the systems comparison is
  invalid until representation/update parity is restored.
- If compiler/receipt objects dominate RSS, reduce or stream metadata rather
  than hiding it outside tensor accounting.
- If event density or active owner words grow with sampled frame density at a
  fixed physical interval, narrow the sublinear claim.
- If the bridge or native allocator exceeds the gate, profile that exact path;
  do not infer a requirement for 32 GiB from a failed saturated-host run.
- If publication quality requires enough sites/material state to erase the
  shared-memory advantage, retain WorldFoam as a bounded systems result rather
  than promoting a universal renderer claim.
