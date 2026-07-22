# World Tubes Paper Closure And Code Retention

Date: 2026-07-19

## Context

Several DynaWorld tasks independently reached compatible conclusions but left
the appearance of multiple competing projects. The publication target is now
singular:

> World Tubes in Gauged Camera Space, implemented by projective STAR UVT,
> evaluated against same-representation per-frame replay, WorldFoam, and
> dynamic 3DGS under a shared public-data protocol.

WorldFoam is a retained-depth operator-ordering challenger and explanatory
ablation. Dynamic 3DGS is the conventional per-frame baseline. The browser,
V-JEPA/world-token, Softmax-GS, feature-tube tuning, and 300-clip scale lanes
are not on the paper's critical path.

## Observed Current State

Observed in the live tree:

- The staged unified smoke is complete for all three MPS lanes.
- The one-step full-temporal smoke loads 300 Coffee Martini frames, creates the
  600 train camera-time pairs, and evaluates the heldout camera.
- The three full paper protocols dry-run successfully and resolve the expected
  manifest, camera split, kernels, steps, frame cost, and pixel cost.
- A fresh focused gate over protocol, runner, PowerFoam, and multicamera data
  tests passed `70 passed, 1 skipped`. This is a different selected test set
  from the previously reported `78 passed, 1 skipped`, not a failure.
- The progressive protocol is 600 steps and 235,929,600 target pixels.
- The fixed-512 protocol is 300 steps and the same 235,929,600 target pixels.
- The global-shuffle protocol is configured but not run as paper evidence.
- The runner, protocols, manifest, and several supporting changes are still
  uncommitted in a dirty worktree shared with an active browser conversion.

Observed metric/report gaps in the unified runner:

- PSNR, SSIM, L1, parameter count/bytes, optimizer bytes, optimizer steps,
  target/raster frames and pixels, and optimizer-update elapsed time exist.
- LPIPS is absent from the unified summary.
- peak device memory is absent.
- serialized checkpoint/model storage bytes are absent.
- paper-facing compile/projection/binning/forward/backward phase timing is not
  normalized into the unified summary.
- active trace/interval counts and visibility fallback fractions are not
  normalized into the unified summary.
- there is no matrix runner or aggregate verifier for
  `protocol x seed x scene x camera split`.
- there is no final CSV/JSON/LaTeX table generator over the new unified run
  summaries.

Observed mathematical boundary:

- The continuous quadratic denominator certificate is implemented by checking
  normalized interval endpoints and the stationary point.
- Orbit-window and moving-camera tests exist, including a bounded elevated
  orbit segment.
- No checked code was found for the proposed full `360 degree / 720 degree`
  multi-gauge orbit acceptance test. The paper must implement that result or
  narrow the claim from full revolving-camera programs to certified bounded
  orbit segments.

## Narrow Paper Claim

The defensible claim is not a new universal dynamic-NVS model and not a claim
that World Tubes always beats every representation in PSNR.

The paper should claim:

1. A known or low-dimensional camera program can be compiled into reusable
   sensor-time traces of dynamic Gaussian primitives.
2. Projection, support, binning, visibility metadata, and backward replay can
   scale with trace/event complexity rather than repeated per-frame world-side
   work.
3. Gauge domains and continuous event certificates make that compilation valid
   across nonlinear camera motion, with explicit fallback at genuine support,
   near-plane, denominator, order, and disocclusion events.
4. At matched target-pixel and storage/capacity accounting, the compiled method
   retains useful same-representation quality while reducing the targeted
   repeated work.

The strongest baseline is always:

```text
per-frame STAR replay vs compiled projective STAR UVT
```

WorldFoam and dynamic 3DGS contextualize representation tradeoffs; they cannot
replace the same-representation causal comparison.

## Submission Dependency Chain

### P0: Land A Reproducible Runner Commit

Before expensive runs:

1. Separate the paper-runner changes from the active browser changes.
2. Commit the runner, typed contracts, manifest, protocols, tests, and docs as
   one reproducible paper-pipeline unit.
3. Record the exact fast-mac/STAR and PowerFoam submodule states.
4. Rerun the focused gate from the clean commit.
5. Dry-run all paper protocols and save the resolved commands/config hashes.

Failure condition: any paper row depends on an unrecorded dirty submodule or a
locally edited config.

### P1: Complete The Unified Evidence Contract

Add to every lane summary:

- heldout and train PSNR, SSIM, LPIPS, and L1;
- optimizer-update time and full wall time;
- compile, projection/support/binning, forward, backward, and evaluation time
  where the backend can expose them;
- peak device memory;
- trainable and total parameters;
- parameter, optimizer-state, and serialized checkpoint bytes;
- target and rasterized frame/pixel counts;
- active trace/interval/event counts and fallback fraction for World Tubes;
- active cells/events for WorldFoam;
- per-frame stored state for dynamic 3DGS.

Do not force a fake common statistic when a representation has no equivalent.
Use common cost columns plus representation-specific diagnostic columns.

### P2: Run The Minimum Coffee Martini Matrix

Required first matrix:

```text
progressive 512: seeds 17, 29, 43
fixed 512 pixel-matched: seeds 17, 29, 43
global shuffle: seed 17
deterministic World Tubes correctness timing: seed 17, separately labeled
```

Repeat global shuffle only if seed 17 shows a material effect. Do not spend
three seeds proving a null sampler result.

Each row must fail closed unless it has:

- exact 300-frame dataset contract;
- train `cam04/cam09`, heldout `cam06`;
- all configured steps completed;
- finite metrics;
- exact target-pixel budget;
- local media and W&B provenance;
- a complete cost/storage record.

### P3: Run The Causal Scaling And Theorem Tables

The central systems table is:

```text
per-frame STAR replay vs compiled projective STAR UVT
F = 4, 8, 16, 32, 64, 128
```

Measure compile, projection/support/binning, forward, backward, total step,
payload, peak memory, break-even frame count, and quality delta.

The synthetic theorem table should contain:

- dense fiber reference vs compiled trace;
- affine vs projective vs multi-gauge camera path;
- certified residual bound vs observed maximum error;
- denominator and physical near-plane events;
- visibility crossing and fallback;
- finite exposure and rolling shutter;
- forward and gradient parity.

For full revolving-camera claims, add the missing 360/720-degree multi-gauge
test. Otherwise narrow the text and table to the bounded camera programs that
are actually verified.

### P4: Add Public Breadth

Minimum defensible breadth after Coffee Martini:

- multiple heldout camera triplets on Coffee Martini;
- at least two additional Neural3D scenes;
- at least one controlled D-NeRF sequence;
- seeds 17/29/43 for the promoted protocol, with narrower repeats for expensive
  diagnostic controls.

This is the gate that decides whether World Tubes is a broader method or a
useful one-scene systems result.

### P5: Aggregate And Package

Build one aggregate verifier that consumes the required run matrix and emits:

- machine-readable JSON;
- tidy CSV;
- paper LaTeX tables;
- plot inputs for quality/cost, frame scaling, memory/storage, and fallback;
- an explicit missing/failed-row report.

Only accepted aggregate rows enter `BASELINES.md`. Then convert the draft to a
real LaTeX manuscript, lock configs/manifests, generate figures from saved
JSON, and provide one end-to-end reproduction command.

## What Existing Work Unifies Into The Paper

| Existing work | Paper role | Action |
| --- | --- | --- |
| Gauged camera-ray bundle and denominator certificates | Mathematical validity | Keep and compress into theory/correctness sections. |
| World Tubes / projective STAR UVT | Primary method | Keep active. |
| Per-frame STAR paths | Same-representation causal baseline | Keep active and make first comparison. |
| Unified paper protocol/runner | Experimental spine | Finish and land. |
| Dynamic 3DGS / fast-mac | Conventional baseline | Keep healthy; no redesign. |
| WorldFoam/PowerFoam | Retained-depth challenger and visibility ablation | Keep one reproducible lane; park native expansion. |
| Decisive demo, exposure, rolling, visibility, orbit reports | Synthetic theorem table inputs | Consolidate into one generator/table. |
| Frame-scaling and compiled-adjoint reports | Systems table inputs | Consolidate into one scaling suite. |
| Existing Coffee Martini matched sweep | Pilot and route check | Retain, label as pilot, do not present as final breadth. |

## What To Ignore Until Submission

- Browser WebGPU optimization and UI work.
- V-JEPA/F32 multicamera world-token training.
- Gaussian 300-clip scale training and its 512px NaN.
- Further STAR feature-tube opacity/support sweeps.
- Softmax-GS ports or architecture expansion.
- Direct-serial promotion beyond an optional bounded parity result.
- Native WorldFoam optical-transfer Metal expansion.
- New gauge/fiber vocabulary or umbrella theory.
- Native 2704x2028 training. Streaming is valuable follow-up engineering, but
  a defensible 512-wide paper does not depend on it.
- Full external SOTA reproduction before the same-representation result is
  convincing.

## Code Retention Policy

### Keep Active

- shared paper protocol types/sampler/cost contracts;
- unified runner and aggregate verifier;
- STAR projective interval forward/backward and per-frame replay;
- the minimal PowerFoam/WorldFoam adapter;
- the fast-mac dynamic 3DGS adapter;
- theorem/scaling generators that produce paper rows;
- behavior-level tests for the above.

### Freeze In Place

- feature-tube quality experiments;
- Softmax-GS;
- V-JEPA performance experiments;
- gauge-field/world-token training not used by the paper;
- old PowerFoam reproduction/CUDA exploration;
- browser trainer;
- intermediate STAR and WorldFoam audit/report layers.

Frozen means no new architecture work. It does not mean delete evidence from a
dirty working tree.

### Consolidate Then Delete From The Active Checkout

After accepted JSONs, commands, and artifact references are preserved in one
registry, the following are deletion candidates:

- report-on-report audit modules superseded by a final theorem/scaling
  verifier;
- paired tests that only verify the formatting or existence of superseded
  intermediate reports;
- rejected one-knob STAR support/alpha sweep launchers once their negative
  result and selected artifacts are indexed;
- duplicate WorldFoam Gate4 launch wrappers that do not own a distinct kernel
  or accepted fixture;
- duplicate browser bundle exporters after the active browser task selects one
  canonical adapter;
- the old 2D browser trainer path after the multicamera 3D path passes its
  bundle/camera parity gate;
- generated caches, `__pycache__`, temporary outputs, and local smoke debris.

Do not mass-delete `research_experiments/` today. Many tests import those
modules directly, and several are the only executable provenance for accepted
theory artifacts. First build the paper dependency closure, retain its
generators, snapshot the rest in git history/artifact indexes, then delete in
one reviewable archival commit.

## Falsification And Stop Rules

Current belief: World Tubes is paper-worthy as a camera-program compiler.

Confidence: medium until public scaling and breadth are run.

Evidence that strengthens the paper:

- same-representation quality within roughly 0.1-0.3 dB PSNR;
- at least 3x reduction in the targeted projection/binning work;
- roughly 1.3x or better end-to-end gain on the intended dense-frame regime;
- controlled memory/storage and fallback behavior;
- consistent results across scenes/camera splits.

Evidence that weakens or changes the paper:

- no break-even before impractically large F;
- quality loss dominated by compiled approximation rather than training;
- fallback grows with frame count nearly as fast as per-frame work;
- wins disappear after storage/capacity matching;
- results hold only on Coffee Martini.

If those fail, do not reopen umbrella theory. Narrow the paper to a certified
renderer/compiler result, publish a negative scaling boundary if still useful,
or stop.

## Immediate Next Action

The next action is not another model idea. Isolate and commit the verified
unified paper runner, then add the missing evidence schema and matrix aggregate
before launching the expensive 512-wide rows.
