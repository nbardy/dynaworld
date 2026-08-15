# World Tubes Paper Finish Master Plan

Date: 2026-08-13 KST

Paper: **World Tubes in Gauged Camera Space: Frame-Amortized Dynamic Gaussian Rendering**

Scope: Paper A only

Status: implementation-rich, evidence-incomplete, not submission-ready

Supersedes for execution planning: the Paper A portions of older broad project handoffs

Does not supersede: generated evidence ledgers, retained run reports, source history, or the WorldFoam research lane

## 0. Executive answer

### Do we need more paper code?

**Yes, but the remaining code work is bounded and mostly packaging, verification,
and failure repair.** The main method, renderer backend, typed training protocol,
unified runner, matrix runner, frozen same-world comparator, variable-camera
stress runner, theorem-table generator, evidence-schema-v2 validator, and
submission artifact generator already exist.

The required code work is:

1. preserve the exact Paper A source slice in clean superproject and STAR
   commits;
2. add one exact dataset-family/pose-source acceptance gate so Neural3D
   evidence cannot self-consistently hash the superseded camera convention;
3. run the expanded focused behavioral suite against that clean slice;
4. fix only failures that block an already-declared paper contract;
5. complete one bounded end-to-end evidence smoke;
6. package one genuinely runnable demo command and a machine-verifiable demo
   manifest;
7. finish venue-template/PDF packaging and any small artifact-export glue the
   selected venue requires.

No new renderer family, optimizer, gauge formalism, WorldFoam kernel, CUDA
port, or native-resolution loader is required to finish the narrow paper.

### Do we need to run the real experiments?

**Yes. This is the dominant remaining work.** The current schema-v2 submission
ledger accepts only the bounded theorem component. It has:

```text
accepted theorem component:                  yes
accepted public contexts:                    0 / 7 minimum
accepted public lane records:                0 / 21 minimum
accepted frozen same-world scaling sweep:    no
accepted bounded variable-camera curve:      no
strict evidence artifact bundle:             incomplete
venue-ready, visually checked PDF:           no
```

The minimum queue contains **nine runtime jobs**:

1. one frozen identical-world replay-versus-compiled sweep;
2. one bounded variable-camera closure/death curve;
3. seven schema-v2 Coffee Martini public contexts.

The public matrix summary is derived after the seven contexts validate; it is
not a tenth runtime job. Each public context runs the three declared lanes—
World Tubes, retained-depth WorldFoam context, and dynamic 3DGS context—under
one protocol and evaluator, yielding 21 lane records.

### Do we need more math?

**No new foundational math is required for Paper A.** The paper should freeze:

\[
T_i = \pi_*\Gamma^*w_i,
\]

the gauged camera-ray pullback/pushforward construction; Gaussian fiber
marginalization by Schur complement on valid local charts; conditional-depth
and visibility-order strata; bounded projective event domains; and the
fixed-topology compiled adjoint.

The remaining mathematical work is editorial and evidentiary:

- check assumptions and notation consistently across the manuscript;
- state the fixed-topology and bounded-chart hypotheses next to every theorem
  and empirical claim that uses them;
- ensure the synthetic theorem table is a faithful executable witness;
- distinguish structural sublinear world-side work from necessarily linear
  output sampling;
- describe failure/death outside the certified camera domain;
- verify citations and novelty wording.

There is also one **method-exposition gap** that requires writing equations and
pseudocode, not inventing a new algorithm: the draft must promote the actual
homogeneous projective trace and interval-certificate construction from the
research notes into the method section. Likewise, the visibility and adjoint
claims need short formal statements and an explicit complexity model. Section
13.2 specifies this work.

Do **not** add the new WorldFoam connection formulation, full-orbit holonomy,
new gauge names, retained-depth transport kernels, or an SPD(4)-novelty claim
to this paper. Those are separate research directions.

## 1. Authority and truth hierarchy

When documents disagree, use this order:

1. retained, verifier-accepted JSON and its source hashes;
2. `research_notes/gauged_uvt_trace_atlas/paper/generated/schema_v2/evidence_ledger.md`;
3. this master finish plan;
4. `research_notes/world_tubes_paper_completion_handoff_2026-08-10.md`;
5. `TODO/unified_paper_ablation_pipeline.md`;
6. `research_notes/gauged_uvt_trace_atlas/paper/WORLD_TUBES_EXPERIMENT_PLAN.md`;
7. `EXPERIMENTS.md` and `BASELINES.md`;
8. dated loose notes and older handoffs.

Two important supersessions are already settled:

- Historical schema-v1 progressive runs are diagnostics, not paper evidence.
  They cannot be relabelled because they lack schema-v2 provenance and used a
  superseded Neural3D camera conversion.
- The extra 14 public breadth rows are desirable after the minimum cut, but do
  not block the narrow compiler paper. Older plans that describe every breadth
  dataset as mandatory are superseded by the 2026-08-10 completion handoff.

## 2. Fact, inference, proposal, and speculation

### Observed facts

- The theorem component is accepted by the schema-v2 artifact generator.
- The seven minimum public contexts, frozen same-world sweep, and
  variable-camera curve are absent from the current evidence ledger.
- Three-lane two-step and all-300-frame one-step smokes have executed on MPS.
- Historical 512-wide runs exist but do not satisfy the current evidence
  contract.
- The repository and STAR submodule are currently dirty, including both Paper
  A changes and unrelated WorldFoam/browser work.
- Core Paper A runners and several tests are currently untracked or modified;
  publication commands require clean source.
- The generated TeX is a generic Pandoc article, not a venue-ready manuscript.
- The current 512-wide publication protocol uses enough eager target/baseline
  state that the incident-calibrated guard rejects the 24 GiB workstation.
- Native 2704x2028 training is not supported by the current eager full-video
  path and is not part of the narrow claim.

### Strong inference

- The shortest route to a defensible paper is evidence closure, not another
  method cycle.
- The frozen same-world experiment is the most causally important result
  because it isolates compilation from representation and training quality.
- A clean 32 GiB-or-larger Apple host is the least disruptive currently
  supported publication host; this is a systems envelope, not an intrinsic
  memory requirement of World Tubes.
- Porting to CUDA/B200 before submission would add a second native backend and
  new parity burden, so it is unlikely to shorten the critical path.

### Proposed execution policy

- Freeze the Paper A algorithm.
- Preserve a clean source slice before spending compute.
- Execute one evidence-producing job at a time.
- Preserve failed reports as negative results.
- Narrow claims when a gate fails; reopen implementation only for a diagnosed
  correctness or evidence-contract defect.
- Stop once the narrow paper meets the final definition of done.

### Speculation that must not become a claim without evidence

- Exact wall-clock speedup and break-even frame count on the publication host.
- Cross-scene generality beyond the minimum Coffee Martini context.
- End-to-end sublinear training cost when recompilation or topology changes
  dominate.
- Native-resolution scaling.
- Full-orbit camera-chart closure.
- Superiority to external state of the art.

## 3. Frozen paper contribution and non-claims

### 3.1 Contribution to defend

Known or low-dimensional camera programs allow dynamic Gaussian primitives to
be compiled into reusable sensor-time world tubes. Within certified camera
charts and fixed topology, the dominant world-side projection, support,
binning, visibility-metadata, and backward replay work scales with trace/event
complexity instead of being independently reconstructed for every requested
frame.

The method hierarchy is:

```text
Gauged camera-ray bundle and pushforward
  -> local Gaussian fiber marginalization
  -> conditional-depth and visibility strata
  -> bounded projective trace atlas
  -> fixed-topology compiled forward and adjoint
  -> Projective STAR UVT Metal implementation
```

The gauged layer is core method mathematics. Projective STAR UVT is its primary
implementation, not a replacement name for the method.

### 3.2 Required claim distinctions

The manuscript must separately report:

- structural trace/event scaling;
- measured compiler and renderer timing;
- output-sample cost, which remains at least linear in requested pixels;
- checkpoint/world storage;
- optimizer state;
- rasterized samples and target samples;
- fallback rate and certified-domain coverage;
- compile amortization and break-even behavior.

Equal active primitive counts are not an equal-memory proof. World Tubes shares
temporal state, while dynamic 3DGS and the retained-depth context lane may store
substantial per-frame or per-event state. Report the actual bytes and work
units.

### 3.3 Explicit non-claims

The paper must not claim:

- sublinear generation of `F * H * W` output samples;
- universally sublinear end-to-end training;
- state-of-the-art novel-view reconstruction;
- generic arbitrary-camera exactness;
- complete `360°/720°` chart transition or closed-loop holonomy;
- derivatives through visibility boundaries or topology changes;
- native-resolution 2704x2028 training;
- a universal replacement for 4DGS, deformable Gaussians, or WorldFoam;
- novelty of SPD(4) Gaussian parameterization by itself;
- completion of retained-depth ordered transfer or WorldFoam connection math.

## 4. Definition of done

Paper A is finished only when every checkbox below is true.

### Source and environment

- [ ] The exact Paper A superproject slice is intentionally reviewed and
      preserved in a clean commit.
- [ ] The exact STAR submodule slice is intentionally reviewed and preserved
      in a clean commit.
- [ ] Both commit hashes are recorded in every evidence report.
- [ ] No unrelated WorldFoam, browser, V-JEPA, or user change is accidentally
      included.
- [ ] The Python environment, native binary identities, dataset identities,
      calibration convention, and LPIPS weights are pinned.
- [ ] Submission validation explicitly requires
      `neural_3d_llff_opencv_relative_pinhole_v2` for `neural_3d_video` rows
      and rejects the superseded identity even when its hash is internally
      consistent.

### Behavioral verification

- [ ] The expanded focused paper test suite passes from the clean source.
- [ ] The focused LLFF-to-OpenCV axis-conversion regression passes.
- [ ] The honest incomplete artifact bundle independently verifies.
- [ ] A bounded schema-v2 three-lane evidence smoke passes.
- [ ] Reuse, stale-source, mismatched-evaluator, mismatched-data, and wrong-W&B
      rejection paths remain fail-closed.

### Required evidence

- [ ] The frozen same-world `F={full,4,8,16,32,64,128}` sweep is accepted.
- [ ] Non-unit full-atlas versus sliced-atlas forward and VJP parity passes.
- [ ] The bounded variable-camera closure/death curve is accepted.
- [ ] All seven minimum Coffee Martini public contexts are accepted.
- [ ] The derived matrix summary has exactly seven ordered keys and 21 lane
      records.
- [ ] Every paper number is produced from accepted schema-v2 JSON.

### Manuscript and release

- [ ] The strict artifact generator succeeds without `--allow-incomplete`.
- [ ] The Markdown manuscript contains no obsolete schema-v1 claim.
- [ ] Related work and bibliography are complete and verified.
- [ ] Required figures and tables are generated, captioned, and legible.
- [ ] One runnable demo command and machine-readable manifest work on clean
      source.
- [ ] The manuscript is converted to the selected venue template.
- [ ] A clean PDF builds reproducibly.
- [ ] Every PDF page is visually inspected.
- [ ] The strict manuscript-package verifier passes.
- [ ] Accepted baseline rows are appended to `BASELINES.md`.
- [ ] Reproduction instructions name all source, data, artifact, and hardware
      constraints.

## 5. Critical-path dependency graph

```text
P0 preserve clean Paper A source
  -> P1 focused tests + schema-v2 smoke
      -> P2 frozen same-world causal sweep
      -> P3 variable-camera closure/death curve
      -> P4 seven public contexts, one at a time
          -> P5 strict artifact generation
              -> P6 manuscript/figures/citations/venue conversion
                  -> P7 demo, reproducibility, PDF QA, baseline update, release
```

P2 and P3 may run in parallel on separate clean hosts after P1. P4 may also
begin after P1, but one expensive process per host remains mandatory. P5 cannot
complete until P2, P3, and all seven P4 contexts validate.

## 6. Phase P0 — preserve a clean executable Paper A slice

### Goal

Turn the current mixed dirty checkout into an auditable, clean, reproducible
Paper A source state without losing unrelated work.

### Why this blocks everything else

The evidence runners correctly reject dirty source. More importantly, an
artifact produced from an unreviewed mixed tree cannot identify what code made
the result. Running first and cleaning later would invalidate the expensive
jobs.

### Required inspection

Record, before any manipulation:

```bash
git status --short --branch
git rev-parse HEAD
git diff --stat
git diff --name-status
git -C third_party/fast-mac-gsplat status --short --branch
git -C third_party/fast-mac-gsplat rev-parse HEAD
git -C third_party/fast-mac-gsplat diff --stat
```

Do not use `git reset --hard`, blanket staging, blanket commits, or deletion of
untracked research. The current checkout contains extensive unrelated work.

### Paper A source slice to review and preserve

At minimum, review these files or families explicitly:

#### Protocol, data, camera, and evaluator contract

- `src/train/paper_training_types.py`
- `src/train/paper_training_protocol.py`
- `src/train/camera.py`
- `src/train/multicam_video_data.py`
- `src/train/multicam_val_data.py`
- any canonical evaluator code imported by the unified runner
- `research_notes/data_contract.md`

One bounded code patch is mandatory in this group. Today the decoded-bundle
hash includes `pose_source`, and cross-lane merging requires every lane to use
the same string, but a consistently wrong legacy string can still pass. Add a
dataset-family-aware acceptance function at the single-run validation boundary
and ensure the matrix/submission generator transitively requires its result.

For the current manifests the minimum mapping is:

```text
manifest record dataset == neural_3d_video
  required pose_source == neural_3d_llff_opencv_relative_pinhole_v2

manifest record dataset == dnerf
  required pose_source == dnerf_matched_time_blender_to_opencv_relative_pinhole
```

Do not infer the expected identity from the reported bundle itself. Derive it
from the validated manifest dataset family, then compare the lane metadata and
hashed decoded-bundle field against that expected value. Retain the expected
value and pass/fail result in schema-v2 evidence.

Add behavioral regressions proving:

- the required Neural3D v2 value is accepted;
- `neural_3d_llff_relative_pinhole` is rejected even when all three lanes agree
  and its decoded-bundle hash is recomputed consistently;
- a mismatch between lane metadata and decoded-bundle `pose_source` is
  rejected;
- the D-NeRF expected value remains accepted;
- matrix reuse and strict generation reject a retained child that fails this
  acceptance result.

#### Evidence runners

- `research_experiments/paper_runner_suite/run_unified_paper_ablation.py`
- `research_experiments/paper_runner_suite/run_unified_paper_matrix.py`
- `research_experiments/paper_runner_suite/run_frozen_world_replay_compiled.py`
- `research_experiments/paper_runner_suite/world_tubes_theorem_table.py`
- `research_experiments/paper_runner_suite/generate_world_tubes_paper_artifacts.py`
- `research_experiments/paper_runner_suite/PAPER_ARTIFACTS.md`
- `research_experiments/star_uvt_feature_tubes/projective_variable_camera_closure_death_curve.py`

#### Projective STAR implementation

- the exact Paper A changes beneath
  `third_party/fast-mac-gsplat/variants/star_uvt_v0/`
- the superproject wrappers that import and identify that native variant
- build metadata needed to reproduce the exact loaded binary

#### Minimum protocols

- `src/train_configs/paper_protocols/world_tubes_submission_matrix_v1.jsonc`
- `src/train_configs/paper_protocols/world_tubes_evidence_smoke_matrix.jsonc`
- `src/train_configs/paper_protocols/coffee_martini_full_300f_progressive_512_v1.jsonc`
- `src/train_configs/paper_protocols/coffee_martini_full_300f_fixed_512_pixel_matched_v1.jsonc`
- `src/train_configs/paper_protocols/coffee_martini_full_300f_progressive_global_shuffle_512_v1.jsonc`
- any directly referenced lane configs that are part of those resolved
  protocols

#### Focused tests

- `tests/test_paper_training_protocol.py`
- `tests/test_unified_paper_ablation.py`
- `tests/test_unified_paper_matrix.py`
- `tests/test_frozen_world_replay_compiled.py`
- `tests/test_star_uvt_projective_variable_camera_closure_death_curve.py`
- `tests/test_world_tubes_paper_artifacts.py`

#### Manuscript package

- `research_notes/gauged_uvt_trace_atlas/paper/WORLD_TUBES_PAPER_DRAFT.md`
- `research_notes/gauged_uvt_trace_atlas/paper/WORLD_TUBES_EXPERIMENT_PLAN.md`
- `research_notes/gauged_uvt_trace_atlas/paper/WORLD_TUBES_REFERENCES.bib`
- `research_notes/gauged_uvt_trace_atlas/paper/REPRODUCE.md`
- generated schema-v2 placeholders and their manifest
- this finish plan and its TODO index entry

### Review checklist for each selected source file

- [ ] It is required by the declared Paper A call graph or documentation.
- [ ] Its diff is understood line by line at the behavior-contract level.
- [ ] It does not import the new WorldFoam connection lane.
- [ ] It does not silently fall back to schema v1.
- [ ] It records clean superproject and submodule identities.
- [ ] It binds loaded native binary hashes, not only source paths.
- [ ] It uses the corrected Neural3D calibration convention:
      `neural_3d_llff_opencv_relative_pinhole_v2`.
- [ ] It preserves exact ordered sample identities.
- [ ] It uses the canonical evaluator and fixed black background.
- [ ] It fails closed on missing or mismatched retained artifacts.
- [ ] It is exercised by a behavior test or the bounded evidence smoke.

### Exit gate

```text
superproject porcelain output == empty
STAR submodule porcelain output == empty
both exact commits recorded
paper slice review documented
no unrelated lane included
```

## 7. Phase P1 — environment, dependency, data, and behavior verification

### 7.1 Pin the execution environment

Record in a retained preflight report:

- operating system and build;
- host model and physical/unified memory;
- Python version and environment lock identity;
- PyTorch version and MPS availability;
- compiler/Xcode/Metal toolchain identity;
- main repository commit;
- STAR submodule commit;
- every loaded native extension path, byte count, and hash;
- W&B mode, entity/project, client version, and connectivity result;
- dataset raw inputs and decoded-bundle hashes;
- evaluator source identity;
- LPIPS AlexNet trunk and linear-weight hashes.

Materialize LPIPS weights before a publication window so a network download
cannot invalidate or interrupt a row:

```bash
.venv/bin/python -c 'import lpips; lpips.LPIPS(net="alex", verbose=False).eval().cpu()'
```

The matrix preflight is expected to bind the AlexNet trunk and LPIPS v0.1
linear weights. Preserve the resulting hashes.

### 7.2 Focused source/contract gate

Run from a clean DynaWorld root on a quiet development host:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
.venv/bin/python -m pytest \
  tests/test_paper_training_protocol.py \
  tests/test_unified_paper_ablation.py \
  tests/test_unified_paper_matrix.py \
  tests/test_unified_paper_matrix_lightweight_import.py \
  tests/test_frozen_world_replay_compiled.py \
  tests/test_star_uvt_projective_variable_camera_closure_death_curve.py \
  tests/test_world_tubes_paper_artifacts.py -q
```

These seven files currently contain 114 test functions before the new exact
pose-source regressions. The count is not an
acceptance criterion; every selected behavior must pass.

Run the corrected Neural3D axis-conversion regression separately because it is
not in that seven-file command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train \
.venv/bin/python -m pytest \
  tests/test_multicam_video_data.py::test_neural_3d_raw_llff_axes_convert_to_opencv -q
```

Also audit `research_notes/data_contract.md` for any surviving operational
reference to “schema v1”; documentation must describe the current schema-v2
runner without erasing historical provenance.

### 7.3 Verify the honest incomplete bundle

```bash
python3 \
  research_experiments/paper_runner_suite/generate_world_tubes_paper_artifacts.py \
  --verify-dir \
  research_notes/gauged_uvt_trace_atlas/paper/generated/schema_v2 \
  --allow-incomplete
```

Expected state before runtime evidence:

- theorem component accepted;
- public component missing;
- frozen component missing;
- variable-camera component missing;
- placeholder tables and SVGs present;
- no partial numeric public/frozen/variable table emitted.

### 7.4 Candidate-host preflight

```bash
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/paper_runner_suite/run_unified_paper_matrix.py \
  --preflight-only \
  --matrix src/train_configs/paper_protocols/world_tubes_submission_matrix_v1.jsonc \
  --out-dir outputs/benchmarks/2026-07-28_world_tubes_submission_matrix_schema2 \
  --device mps \
  --wandb-mode online \
  --check-wandb-connectivity
```

Inspect rather than merely trusting the exit code:

- exact seven ordered run keys;
- exactly three lane plans per context;
- corrected camera calibration;
- expected 300-frame source/heldout identities;
- progressive and fixed target-pixel accounting;
- native binary discovery;
- LPIPS weights;
- W&B connectivity;
- memory, swap, disk, and load guard results;
- clean-source requirement.

### 7.5 Bounded schema-v2 evidence smoke

On an operator-approved quiet MPS host:

```bash
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/paper_runner_suite/run_unified_paper_matrix.py \
  --execute \
  --require-clean-source \
  --matrix src/train_configs/paper_protocols/world_tubes_evidence_smoke_matrix.jsonc \
  --out-dir outputs/benchmarks/2026-08-10_world_tubes_schema2_evidence_smoke \
  --device mps \
  --wandb-mode offline \
  --allow-local-mps-execution
```

The smoke must exercise:

- all three actual training lanes;
- K-frame selected-time sampling;
- primitive growth;
- forward, backward, and optimizer steps;
- heldout evaluation;
- exact target/rasterized accounting;
- retained configs, checkpoints, media, and summary hashes;
- native-route identity;
- schema-v2 data/evaluator/source identity;
- reuse and stale-artifact rejection.

It is mechanical evidence only. Never cite its quality or timing as a paper
result.

### Failure policy for P1

- A contract/test failure authorizes a narrow fix to the broken contract.
- A dirty-source failure requires source cleanup, not an override.
- A dataset/calibration mismatch invalidates the row before training.
- A W&B identity failure must be fixed before online publication runs.
- A host guard failure requires a different/quiet host; never bypass it.
- Do not start P2–P4 until the schema-v2 smoke passes.

## 8. Phase P2 — frozen identical-world causal sweep

### Why this is the central result

Comparing different learned representations confounds compilation with model
capacity, initialization, and optimization. The frozen experiment trains one
World Tubes world once, hashes it, and evaluates two execution strategies:

```text
A. per-frame STAR replay
B. one compiled projective interval atlas
```

The checkpoint, world parameters, heldout targets, camera program, evaluator,
physical interval, and requested sample identities remain identical. This is
the clean experiment for the paper's amortization claim.

### Required sample schedule

```text
seed = 17
F = full, 4, 8, 16, 32, 64, 128
CLI frame counts = 0,4,8,16,32,64,128
0 means the full 300-frame interval
every F spans the same full physical interval
indices are ordered and explicitly retained
timing warmups >= 1
timing repeats >= 3
checked command uses 1 warmup and 5 repeats
```

Do not use growing prefixes. A prefix changes both sample count and physical
time span and weakens the scaling interpretation.

### Dry run

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
.venv/bin/python \
  research_experiments/paper_runner_suite/run_frozen_world_replay_compiled.py \
  --protocol \
  src/train_configs/paper_protocols/coffee_martini_full_300f_progressive_512_v1.jsonc \
  --seed 17 \
  --max-frames 0 \
  --frame-counts 0,4,8,16,32,64,128 \
  --timing-warmups 1 \
  --timing-repeats 5 \
  --require-clean-source
```

Inspect the printed plan, resource estimate, output root, source identities,
and target schedule. On an approved publication MPS host add both `--execute`
and `--allow-local-mps-execution`; the latter is an explicit incident-safety
acknowledgement, not permission to bypass a failed live-resource gate.

### Mandatory retained inputs

- resolved protocol and its hash;
- training seed;
- clean main/STAR commits;
- loaded binary hashes;
- trained checkpoint path, bytes, and SHA-256;
- canonical world-state hash before and after every evaluation;
- camera and decoded-target bundle identity;
- exact ordered frame indices and physical times for every F;
- evaluator identity;
- timing warmup/repeat count;
- host/runtime identity.

### Mandatory retained outputs per F and route

- rendered images or retained image digest sufficient for exact validation;
- robust-L1 loss;
- image max absolute delta and loss delta between routes;
- world-parameter VJP tensor digests and norms;
- global and per-parameter normalized VJP error;
- trace, interval, tile, bin, support, and fallback counts;
- compile time;
- projection/support/binning time;
- forward time;
- backward time;
- total step or execution time;
- raw timing samples, not only medians;
- logical tensor work volume, explicitly labelled ineligible as storage or
  peak-memory evidence;
- topology-inclusive serialized retained evaluator bytes, with the atlas
  binary and its tensor/topology hashes retained;
- route-scoped sampled current/driver allocator baselines, peaks, increments,
  and phase samples for replay and compiled correctness routes;
- explicit exclusion of interleaved parity replay from compiled-route memory;
- explicit non-unit selected-time full-atlas/chunk-slice parity result.

### Acceptance gates

```text
same checkpoint/world across all F                    required
same full physical interval                           required
exact ordered sample identities                       required
image max absolute error                              <= 1e-5
loss absolute delta                                   <= 1e-5
global world-VJP normalized L2 error                  <= 1e-5
maximum per-parameter VJP normalized L2 error         <= 1e-5
minimum replay and compiled world-VJP norm            > 1e-12
fallback fraction                                     <= 0.20
non-unit atlas/chunk slice parity                      accepted
timing warmups                                         >= 1
timing repeats                                         >= 3
raw timing sample algebra                              verifier-rederived
topology-inclusive serialized compiled atlas           required
route-scoped replay/compiled allocator peaks            required
logical tensor proxy not promoted as memory/storage     required
source/data/evaluator/native identities               exact
```

### Claim-level success targets

These are desirable claims, not permission to reject an honest valid result:

```text
projection/support/binning metadata reduction at F>=32   >= 3x
many-sample end-to-end speedup                            >= 1.3x
quality loss versus replay                                <= 0.1-0.3 dB
ordinary-scene fallback                                   <= 10-20%
break-even                                                 16-32 samples
```

### Decision branches

#### Branch P2-A — parity and speed pass

- Promote the same-world scaling table and plot.
- State the measured break-even range.
- Keep structural and timing claims separate.
- Report compilation cost and amortized cost explicitly.

#### Branch P2-B — parity passes, speed target fails

- Publish the verified structural amortization result.
- Report the negative timing honestly.
- Narrow the title/abstract wording from speedup to compiled or
  frame-amortized world-side work if necessary.
- Profile only enough to explain the result; do not start a new architecture.

#### Branch P2-C — parity fails

- Stop all publication timing interpretation.
- Localize the first failing F and first failing parameter family.
- Check selected-time slicing, fallback routing, checkpoint mutation,
  evaluator identity, and native binary identity.
- Fix correctness only if the failure is an implementation defect in the
  already-declared method.
- Rerun the complete sweep after the fix; never splice pre- and post-fix rows.

#### Branch P2-D — fallback exceeds the gate

- Preserve the report as a bounded-domain negative result.
- Determine whether the fixed camera program is outside the certified atlas or
  the certificate is overly conservative.
- Narrow the supported domain before extending theory.

## 9. Phase P3 — bounded variable-camera closure/death curve

### Purpose

This experiment defends the statement that gauged projective charts handle a
bounded family of moving cameras and fail explicitly once their certified
domain closes. It is not a full-orbit theorem.

### Fixed experiment contract

```text
one fixed synthetic world
one fixed physical interval
64 ordered samples
increasing yaw half-span
exact rational live-depth-order oracle
compiled bounded camera program under test
same forward loss and world-VJP comparison
```

### Execute and verify

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
.venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_variable_camera_closure_death_curve.py \
  --execute \
  --out-dir \
  outputs/benchmarks/2026-07-28_world_tubes_variable_camera_closure_death_curve

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_variable_camera_closure_death_curve.py \
  --verify-report \
  outputs/benchmarks/2026-07-28_world_tubes_variable_camera_closure_death_curve/summary.json \
  --require-current-source
```

### Acceptance gates for a closure row

```text
image PSNR                                    >= 50 dB
image p99.9 absolute error                    <= 2/255
image max absolute error                      <= 4/255
world-VJP relative L2 maximum                 <= 0.02
fallback cell fraction                        <= 0.20
fallback sample fraction                      <= 0.20
invalid sample fraction                       == 0
trace/replay ratio                            < 0.50
interval/dense ratio                          < 0.80
unresolved or stale chart state               none
```

### Whole-curve gate

- [ ] At least one camera span is accepted as closure.
- [ ] At least one larger span is classified as death.
- [ ] Accepted rows form one prefix.
- [ ] Death rows form one suffix.
- [ ] No camera span returns to closure after first death.
- [ ] Every row has retained exact-oracle and compiled-route identities.

### Decision branches

- If the curve passes, claim bounded moving-camera closure and explicit death.
- If small spans fail, diagnose projective chart or oracle correctness before
  proceeding.
- If the death boundary appears earlier than hoped, publish the measured
  bounded range and limitation.
- If the sequence is non-monotone, treat it as a certificate/validator defect;
  do not interpret it as evidence.
- Do not respond by implementing `360°/720°` multi-gauge orbit transitions for
  this paper.

## 10. Phase P4 — seven schema-v2 public contexts

### Purpose

These runs show how the method behaves inside one honest public-data training
and cost context. They are not the causal proof of compilation; P2 is.

### Minimum matrix

| # | Role | Protocol | Seed | World Tubes policy |
|---:|---|---|---:|---|
| 1 | Primary progressive | `coffee_martini_full_300f_progressive_512_v1` | 17 | `fast_exploration` |
| 2 | Primary progressive | `coffee_martini_full_300f_progressive_512_v1` | 29 | `fast_exploration` |
| 3 | Primary progressive | `coffee_martini_full_300f_progressive_512_v1` | 43 | `fast_exploration` |
| 4 | Pixel-matched fixed | `coffee_martini_full_300f_fixed_512_pixel_matched_v1` | 17 | `fast_exploration` |
| 5 | Pixel-matched fixed | `coffee_martini_full_300f_fixed_512_pixel_matched_v1` | 29 | `fast_exploration` |
| 6 | Pixel-matched fixed | `coffee_martini_full_300f_fixed_512_pixel_matched_v1` | 43 | `fast_exploration` |
| 7 | Global-shuffle control | `coffee_martini_full_300f_progressive_global_shuffle_512_v1` | 17 | `fast_exploration` |

### Frozen dataset/split contract

```text
dataset: Neural 3D Video / Coffee Martini
frames: all 300 synchronized frames
training cameras: cam04 and cam09
heldout camera: cam06
calibration: neural_3d_llff_opencv_relative_pinhole_v2
background: fixed black
color calibration: none
evaluator: canonical schema-v2 evaluator
```

### Schedule contract

- Progressive: 600 steps across `128w -> 256w -> 512w`.
- Fixed control: 300 steps at 512w.
- Progressive and fixed must have exactly matched target-pixel budgets.
- Global shuffle changes only the sampling-order control declared by its
  protocol.
- Every selected target/rasterized sample is retained in the exact accounting.

### Exact run keys

```text
coffee_martini_full_300f_progressive_512_v1/seed_17/fast_exploration
coffee_martini_full_300f_progressive_512_v1/seed_29/fast_exploration
coffee_martini_full_300f_progressive_512_v1/seed_43/fast_exploration
coffee_martini_full_300f_fixed_512_pixel_matched_v1/seed_17/fast_exploration
coffee_martini_full_300f_fixed_512_pixel_matched_v1/seed_29/fast_exploration
coffee_martini_full_300f_fixed_512_pixel_matched_v1/seed_43/fast_exploration
coffee_martini_full_300f_progressive_global_shuffle_512_v1/seed_17/fast_exploration
```

### Run one context at a time

Begin with the progressive seed-17 key after a dry run. Substitute each exact
key above on subsequent invocations:

```bash
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/paper_runner_suite/run_unified_paper_matrix.py \
  --execute \
  --require-clean-source \
  --reuse-existing \
  --max-new-runs 1 \
  --run-key \
  coffee_martini_full_300f_progressive_512_v1/seed_17/fast_exploration \
  --matrix \
  src/train_configs/paper_protocols/world_tubes_submission_matrix_v1.jsonc \
  --out-dir \
  outputs/benchmarks/2026-07-28_world_tubes_submission_matrix_schema2 \
  --device mps \
  --wandb-mode online \
  --allow-local-mps-execution
```

`--reuse-existing` is safe only because the runner revalidates source, data,
evaluator, native binary, retained artifacts, and exact summary content. Never
manually copy a summary into this output root.

### Required record for each of the three lanes

#### Identity and provenance

- clean superproject commit;
- clean STAR submodule commit;
- exact resolved config and hash;
- exact native route and loaded-binary hashes;
- raw input file identities;
- decoded target/camera bundle identity;
- calibration convention;
- exact W&B run and finalized-file identity;
- hardware/runtime identity.

#### Training and sampling

- seed and initialization identity;
- all scheduled steps completed;
- exact ordered frame/time samples;
- target frame count and target pixel count;
- rasterized frame count and rasterized pixel count;
- primitive growth history and final active counts;
- loss history and finite-gradient status;
- checkpoint identity and bytes.

#### Evaluation

- all declared heldout frames;
- global RGB L1;
- global RGB MSE and one derived PSNR;
- mean SSIM over the declared set;
- mean LPIPS over the declared set;
- fixed black background and no hidden calibration;
- retained train and heldout media.

#### Cost and memory

- compile time;
- projection/support/binning time;
- forward time;
- backward time;
- optimizer time;
- total wall time;
- parameter count and bytes;
- optimizer-state bytes;
- checkpoint bytes;
- sampled current and driver peak memory;
- rasterized samples;
- representation-specific trace/event/fallback counts.

### Per-context acceptance

- [ ] All three declared lanes completed.
- [ ] All metrics are finite.
- [ ] All exact schedule/data/evaluator identities validate.
- [ ] Progressive/fixed target-pixel accounting is exact where applicable.
- [ ] Each summary has synchronized timing categories.
- [ ] Checkpoint, media, config, and W&B artifacts exist and hash-match.
- [ ] The matrix embeds an exact copy of the retained child summary, modulo the
      runner-added path field.

### Matrix acceptance

```text
matrix name                    world_tubes_submission_matrix_v1
ordered public contexts       exactly 7
lane records                  exactly 21
partial numeric table         forbidden
stale/mixed-source reuse      forbidden
missing child                 rejected
extra child                   rejected
```

### Public-result interpretation branches

#### Quality is competitive

- Report mean and dispersion across the three primary/fixed seeds.
- Keep the causal speed claim tied to P2.
- Describe three-lane results as representation-and-cost context.

#### Quality is weaker but stable

- Publish as a compiler/method paper rather than an SOTA-quality paper.
- Report exact tradeoffs and visual failure modes.
- Do not retune all lanes after seeing heldout results unless a predeclared
  protocol defect is found.

#### One seed or lane crashes

- Preserve the failed artifact and cause.
- Fix only deterministic implementation/resource faults.
- Restart the entire affected context from a clean process.
- Do not aggregate a partial context or mix source commits.

#### Memory guard rejects the job

- Move the unchanged job to a compliant host.
- Do not lower evidence requirements or bypass the guard.
- Streaming targets may be implemented only if no compliant MPS host is
  available and the same schema-v2 contract can be preserved.

## 11. Host safety and experiment operations

### Current incident-calibrated envelope

```text
progressive/global-shuffle estimated requirement: 18.745 GiB
fixed-512 estimated requirement:                  17.303 GiB
guard ceiling:                                    60% physical memory
currently supported MPS publication target:       clean Apple host >=32 GiB
```

The 32 GiB host recommendation is not a World Tubes representation theorem.
It covers eager decoded targets, baseline state, optimizer state, allocator
headroom, and evidence artifacts in the current unified protocol.

### Fresh check before every expensive child

```text
reclaimable memory       >= 10 GiB
swap in use              <= 2 GiB
free disk                >= 32 GiB
1-minute load / CPUs     <= 0.75
```

Also require:

- one publication process per host;
- no unrelated high-CPU or high-memory processes;
- no concurrent MPS job;
- one resumable context at a time;
- retained stdout/stderr and failure receipt;
- operator approval for every expensive launch;
- resource check repeated between lanes/contexts, not only once per day.

### Why not immediately use a B200?

The publication implementation is Metal/MPS. A B200 is useful only after a
separately verified CUDA implementation has forward/VJP, topology, evaluator,
timing, and evidence parity. That port is real work and introduces a second
implementation into the paper. Unless a validated CUDA route already exists,
use a sufficiently provisioned Apple host and keep CUDA out of the critical
path.

### Native resolution

Do not attempt 2704x2028 with the eager full-video path. Native-resolution
support requires:

1. selected-frame or selected-pixel decode;
2. bounded host cache;
3. selected-sample ray staging;
4. streamed evaluation;
5. new memory validation.

Those changes are valuable systems work but are not required for the 512-wide
narrow paper.

## 12. Phase P5 — strict evidence artifact generation

### Submission-facing generator

The canonical generator is:

`research_experiments/paper_runner_suite/generate_world_tubes_paper_artifacts.py`

It is CPU-only, does not import Torch or a renderer, and must rederive—not
trust—paper-facing numbers from retained reports.

### Inputs that must all be accepted

1. accepted theorem summary and its pinned retained reports;
2. accepted frozen same-world wrapper summary;
3. accepted variable-camera summary;
4. exact seven-row matrix summary;
5. each exact schema-v2 public child summary.

### Strict generation

```bash
python3 \
  research_experiments/paper_runner_suite/generate_world_tubes_paper_artifacts.py
```

Do not pass `--allow-incomplete` for the submission bundle.

### Required generated outputs

- `theorem_table.tex` and Markdown equivalent;
- `public_context_table.tex` and Markdown equivalent;
- `frozen_scaling_table.tex` and Markdown equivalent;
- `variable_camera_table.tex` and Markdown equivalent;
- theorem SVG;
- public-context SVG;
- frozen-scaling SVG;
- variable-camera SVG;
- evidence ledger JSON/Markdown;
- missing-runtime-input ledger, which must be empty;
- artifact manifest with byte counts and SHA-256 for every generated file.

### Generator invariants

- No partial numeric component is allowed.
- Frozen timing statistics are rederived from raw samples.
- Timing key sets and sample lengths are exact.
- Theorem values are rederived from pinned source reports.
- The selected matrix name and output root must match.
- The matrix has exactly the declared ordered children and lane count.
- Embedded child summaries exactly match retained child summaries.
- Schema-v1 artifacts are rejected.
- Placeholder bytes are absent from the strict bundle.

### Failure response

- Missing input: rerun or recover the exact required job; do not hand-fill.
- Hash mismatch: find the changed producer or artifact; do not update the
  expected hash without understanding it.
- Algebra mismatch: fix producer or generator and regenerate from raw samples.
- Partial matrix: finish the missing context; do not publish a partial table.

## 13. Phase P6 — manuscript completion

### 13.1 Manuscript source policy

The editable source is:

`research_notes/gauged_uvt_trace_atlas/paper/WORLD_TUBES_PAPER_DRAFT.md`

The current `WORLD_TUBES_PAPER.tex` is generated working output. Do not edit it
as the source of truth. After Markdown changes, regenerate it until venue
conversion creates the final venue-owned TeX tree.

### 13.2 Required mathematical edit pass

This is a consistency/proof-exposition pass, not a new-theorem pass.

- [ ] Define the sensor-time base, ray-depth fiber, camera program, pullback,
      and pushforward before first use.
- [ ] Use one notation for world primitive, camera chart, trace, and depth
      variable throughout.
- [ ] State positivity/SPD, affine-chart, denominator, near-plane, and bounded
      interval assumptions exactly where needed.
- [ ] Show the Schur complement and identify retained conditional-depth mean
      and variance.
- [ ] Separate gauge-invariant fiber value from coordinate/Jacobian factors.
- [ ] Define visibility/order strata and what is held fixed in the VJP.
- [ ] State what happens at event boundaries and that derivatives through
      topology changes are not claimed.
- [ ] Define finite exposure and rolling-shutter sampling without overstating
      continuous exactness.
- [ ] Use “ordered parallel transport” or “path-ordered transfer” for open
      rays; reserve “holonomy” for closed loops.
- [ ] Keep ordered retained-depth transfer as a boundary/extension, not the
      main method.
- [ ] Cross-check every stated tolerance with the generated theorem table.

Add the missing implementation-facing mathematics explicitly:

- [ ] Move the homogeneous projective trace into the main method:
      \(u(t)=h_u(t)/h_z(t)\), \(v(t)=h_v(t)/h_z(t)\), with the chosen local
      polynomial/rational record.
- [ ] Define the continuous denominator margin, physical near-plane event,
      trace residual, recursive interval splitting rule, accepted chart,
      unresolved chart, and fallback route.
- [ ] Include compact pseudocode for compile → certify → split → lower →
      fallback. The existing derivation source is
      `research_notes/gauged_uvt_trace_atlas/03_projective_rational_traces/README.md`.
- [ ] Add a short gauge-change invariance statement whose scope matches the
      theorem artifact.
- [ ] Add a fixed-cell visibility proposition: on a certified stable-order
      cell, the compiled result matches frozen replay up to declared trace and
      commutation error.
- [ ] Add a fixed-topology adjoint-validity statement that names all detached
      decisions: topology, support, bin membership, order, chart selection,
      and fallback choice.
- [ ] Define a cost model with requested samples \(F\), primitives/traces
      \(N\), chart/event complexity \(J\), interval/bin entries, image size
      \(H\times W\), one-time compilation, world-side replay, and unavoidable
      output shading. State which terms are measured and which are asymptotic.
- [ ] Add a notation table. In particular, stop overloading `T` for time and
      transmittance, `A_l` for active sets and adjoints, and `tau` for sensor
      time and optical depth.

Do not promote standard conditional-Gaussian algebra to the headline theorem.
The Schur result supports the compiler; the novel contribution is the
certified camera-program trace atlas and reusable fixed-topology execution.

One implementation boundary must be stated precisely: the compiler can carry
a pixel-varying conditional-depth plane, but the production interval Metal
visibility sorter currently consumes scalar/cell depth metadata. The paper
must not imply production pixel-varying projective depth-order certification.

### 13.3 Related-work and novelty audit

The current bibliography is compact. Before submission:

- [ ] Verify every bibliographic field against the authoritative publication.
- [ ] Ensure every cited key resolves and every bibliography entry is cited or
      intentionally retained.
- [ ] Cover static 3D Gaussian splatting.
- [ ] Cover dynamic, deformable, native-4D, and spacetime Gaussian methods.
- [ ] Cover moving-camera/rolling-shutter Gaussian work.
- [ ] Cover alternative camera/ray parameterizations where directly relevant.
- [ ] Cover Neural3D and D-NeRF dataset provenance.
- [ ] Add trace-atlas, pushforward/marginalization, or compiled-rendering prior
      art only where it directly supports or limits novelty.
- [ ] Cover classical splatting/EWA foundations where the renderer lineage
      depends on them.
- [ ] Cover visibility, sorting, and order-independent/approximate compositing
      work relevant to the stable-order certificate.
- [ ] Cover dynamic-Gaussian compression or acceleration and camera/temporal
      amortization where they provide the nearest systems comparisons.
- [ ] Cover certified interval or approximation methods used to justify the
      residual/denominator terminology.
- [ ] Cite appropriate change-of-variables or fiber-integration foundations
      for standard mathematical machinery rather than implying novelty.
- [ ] Avoid claiming invention of Schur marginalization, SPD(4), fiber
      integration, or path ordering.
- [ ] State the specific novelty as compiling a known dynamic Gaussian world
      and bounded camera program into reusable sensor-time traces and a
      fixed-topology adjoint.

Use primary papers or official project publications for technical citations.
Do not rely on memory for publication year, author list, venue, or precise
method characterization.

### 13.4 Results-writing pass

- [ ] Replace all public/frozen/variable placeholders only through generated
      `\input` tables and accepted figures.
- [ ] State the theorem table as bounded executable correctness evidence.
- [ ] Present P2 as the central causal experiment.
- [ ] Present P3 as bounded camera-domain closure/death.
- [ ] Present P4 as public representation-and-cost context.
- [ ] Report seed aggregation and individual values where informative.
- [ ] Report compile cost and break-even, not only steady-state timing.
- [ ] Report memory and bytes with measurement definitions.
- [ ] Explain rasterized versus target sample accounting.
- [ ] Include negative results and fallback-heavy regimes.
- [ ] Do not use smoke results as benchmark evidence.
- [ ] Do not use old schema-v1 plots or the historical `paper_ready=true` bit.
- [ ] Remove the internal manuscript status block before submission.
- [ ] Rewrite the current prospective/checklist language in Sections 5–6 as
      Experimental Setup, Results, Ablations, and Limitations.
- [ ] Compress the current seven-item contribution list to roughly four
      defensible contributions: gauged compiler; certified projective and
      visibility atlas; compiled adjoint/implementation; causal evaluation.

### 13.5 Required figures

Minimal main-paper figure package:

1. one combined method/system figure: world primitive → gauged camera-ray
   bundle → Schur marginalization → sensor-time tube → replay-versus-compiled
   execution;
2. one projective chart/event figure showing homogeneous traces, certified
   windows, visibility strata, fallback, and closure/death;
3. generated frozen same-world causal scaling figure;
4. generated bounded variable-camera closure/death figure;
5. generated public-context figure;
6. one publication-quality labeled qualitative comparison with consistent
   crops and useful error panels.

The generated theorem figure and a dedicated Schur/conditional-depth diagram
may move to the appendix if the venue page budget is tight. A fallback-heavy
example may be combined with figure 2 or placed in the appendix.

The four evidence figures must come from the strict generator. Concept/system
figures may be authored separately, but their source must be retained and
their claims must match the method. Avoid ornamental figures that consume page
budget without explaining a contribution.

The existing `real_video_equivalence.jpg` is only a draft asset: it has
unlabelled rows and does not contain all panels described by its caption. It
must be regenerated or replaced, not merely resized.

### 13.6 Required tables

- theorem/correctness table;
- frozen same-world scaling/parity/cost table;
- public three-lane context table;
- bounded variable-camera closure/death table;
- implementation/complexity or memory-accounting table if not clear in prose;
- limitations/fallback table or compact appendix table.

Every numeric table must trace to the artifact manifest. No hand-copied result
cell is allowed.

### 13.7 Venue conversion

After the target venue is chosen:

- [ ] Acquire the official current template and author kit. ICLR 2027 is now
      confirmed from the official author guide, but on 2026-08-15 its stated
      `iclr2027.zip` URL returned `404` and the official Master-Template
      repository still exposed only through `iclr2026`. Do not relabel the
      2026 style as 2027; retry the official URL when it is published.
- [ ] Record the template archive/style hashes and retrieval date once that
      official archive exists.
- [ ] Create the venue-owned TeX entry point.
- [ ] Port title, abstract, sections, equations, tables, figures, bibliography,
      and appendix without editing generated evidence tables.
- [ ] Apply anonymous/authorship rules correctly.
- [ ] Meet page, font, margin, figure, color, and supplemental limits.
- [ ] Create an explicit main-paper/appendix/supplement page-budget plan.
- [ ] Export SVG figures to the venue-supported PDF/PNG formats.
- [ ] Ensure fonts are embedded.
- [ ] Resolve overfull boxes, clipped tables, missing glyphs, and broken links.
- [ ] Preserve an arXiv-compatible build if venue source restrictions differ.

The required generative-AI disclosure is drafted and verifier-clean at
`research_notes/gauged_uvt_trace_atlas/paper/venue/iclr2027/AI_USE_STATEMENT.md`;
the author must still approve its wording before submission. The official
ICLR 2027 author guide requires anonymous review, at most nine main-text pages,
and a separate AI-use statement.

A fail-closed venue scaffold now also exists in that directory. Its manifest
binds the current concise draft, bibliography, evidence manifest, and AI-use
statement, but declares `scaffold_blocked` and records null archive/style
hashes while the official kit is unavailable. Its TeX entry point intentionally
errors and consumes only the accepted theorem fragment; it does not promote the
placeholder frozen/public fragments or the dirty-source variable-camera CPU
candidate. This scaffold does not close any venue-conversion checkbox above.

The final strict verifier is expected to reject the generic Pandoc `article`
class. Venue conversion is therefore mandatory, not cosmetic.

### 13.8 PDF visual QA

Inspect every rendered page at normal size and zoomed:

- no clipped equations, figures, legends, or tables;
- labels readable in grayscale and color;
- consistent notation and font sizes;
- all cross-references resolve;
- no placeholder text or draft status survives;
- citations render and bibliography is complete;
- figure/table numbering matches prose;
- captions state dataset, split, seeds, metric direction, and error bars;
- appendix and supplemental references resolve;
- hyperlinks do not cover unrelated text;
- author/anonymity state matches submission mode.

### 13.9 Manuscript package gate

After venue conversion and PDF QA:

```bash
python3 \
  research_experiments/paper_runner_suite/generate_world_tubes_paper_artifacts.py \
  --verify-manuscript
```

The generator's `submission_ready` field covers the evidence bundle only. It
does not certify visual layout. Record PDF QA separately.

## 14. Phase P7 — demo, reproducibility, baselines, and release

### 14.1 One-command demo

The paper needs one small command that a clean user can run without launching
publication training. The demo should:

- load a retained bounded fixture or tiny public subset;
- run replay and compiled routes on the same world;
- emit images and a small JSON report;
- check forward and VJP parity;
- report trace/replay structural counts;
- complete within a documented modest resource envelope;
- require no W&B account;
- fail clearly if the STAR native binary is absent or incompatible.

The existing “one-command” full matrix in `REPRODUCE.md` is an experiment
launcher, not this demo.

If the existing theorem or frozen runner can expose this contract through a
small preset, prefer a thin checked-in config/entry point over a new renderer.

### 14.2 Demo manifest

Retain a machine-readable manifest containing:

- command and arguments;
- source commits;
- environment identity;
- input files and hashes;
- native binary hash;
- expected output files and hashes or numeric tolerances;
- expected runtime/resource class;
- expected forward/VJP gates;
- license/data-use notes.

### 14.3 Reproduction guide

Update `REPRODUCE.md` so a new operator can perform, in order:

1. environment construction;
2. native STAR build/verification;
3. data acquisition and decoded-bundle validation;
4. LPIPS weight materialization;
5. focused tests;
6. bounded evidence smoke;
7. frozen same-world sweep;
8. variable-camera curve;
9. each exact public run key;
10. strict artifact generation;
11. manuscript/PDF build;
12. independent artifact and manuscript verification.

Commands must say which are safe/read-only, which allocate MPS memory, which
need network/W&B, and which require operator approval.

### 14.4 Baselines and project indexes

Only after verifier acceptance:

- append dated rows to `BASELINES.md`—never overwrite old history;
- include exact config, split, steps, source hashes, W&B identity, timing,
  metrics, and retained JSON path;
- update `EXPERIMENTS.md` from “planned/running” to exact accepted state;
- update `TODO/README.md` and `PROJECT_INDEX.md` to point to the final package;
- preserve failed and historical rows with explicit labels;
- do not mark smoke or schema-v1 rows as baselines.

### 14.5 Release bundle

Prepare:

- clean source commit/tag or archival hash;
- clean STAR submodule commit;
- venue/arXiv source tree;
- final PDF;
- bibliography;
- generated tables and figures;
- artifact manifest;
- accepted JSON summaries and their manifests;
- demo manifest and instructions;
- data preparation instructions and licenses;
- limitations and hardware note.

External submission, arXiv upload, repository publication, or artifact upload
requires explicit user authorization. Preparing the local package does not
authorize publishing it.

## 15. Exact code work inventory

### 15.1 Mandatory now

- cleanly preserve the current Paper A source slice;
- add exact manifest-family-to-pose-source validation to single-run and
  submission acceptance, with legacy-Neural3D rejection tests;
- build/verify the exact STAR native extension for the chosen host;
- run the expanded focused gate, including lightweight matrix import and the
  LLFF/OpenCV conversion regression;
- repair actual behavior-contract failures;
- run and validate the schema-v2 evidence smoke;
- ensure the frozen runner's non-unit selected-time slice parity executes in
  the live report;
- ensure the variable-camera live report verifies current source;
- ensure each matrix child retains exact W&B/config/checkpoint/media identity;
- create or finalize the bounded one-command demo and manifest;
- export generated SVGs to venue-supported formats;
- wire the official venue template and final PDF build;
- keep the strict generator/manuscript verifier green.

### 15.2 Contingent only on a demonstrated failure

- fix selected-time atlas slicing if live forward/VJP parity fails;
- fix source/data/native identity binding if the smoke exposes a hole;
- fix calibrated camera decoding if v2 data validation fails;
- fix timing synchronization if raw timing algebra fails;
- implement streamed target decode only if no safe compliant host exists;
- tighten or narrow the projective certificate if the bounded camera curve is
  invalid or non-monotone;
- adjust manuscript claim wording if timing or public quality is weaker than
  target.

### 15.3 Optional after minimum submission cut

- run the other 14 full-breadth public contexts;
- add more heldout camera triplets;
- add `cook_spinach` and `cut_roasted_beef` seeds 17/29/43;
- add one D-NeRF control;
- add the separately labelled deterministic timing audit;
- improve demo ergonomics;
- add native-resolution streaming;
- port to CUDA after Paper A is frozen.

### 15.4 Forbidden on the critical path

- a new World Tubes architecture;
- a new public alias or gauge formalism;
- full-orbit chart theory;
- new WorldFoam connection or native kernels;
- adaptive M3/M5 WorldFoam material work;
- retained-depth projective integration;
- ordered-transfer multi-seed sweeps;
- SPD(4) novelty work;
- browser/V-JEPA/world-token work;
- feature/Softmax/opacity/support sweeps;
- `direct_serial` promotion;
- external SOTA reproduction;
- native 2704x2028 promotion;
- CUDA as a prerequisite;
- manual editing of generated result tables or working TeX.

## 16. Mathematical work inventory

### 16.1 Complete enough for submission

- camera-ray bundle formulation;
- gauged pullback and depth-fiber pushforward;
- affine local Gaussian marginalization;
- Schur complement for UVT footprint;
- retained conditional-depth mean/variance;
- event-certified projective chart domains;
- visibility/order stratification;
- bounded trace atlas and fallback semantics;
- fixed-topology compiled adjoint;
- finite-exposure/rolling-shutter bounded sampling evidence.

### 16.2 Required editorial math work

- notation normalization;
- assumption placement;
- proof sketches tied to executable witnesses;
- complexity definitions with all variables defined;
- distinction between logical structural volume and measured allocator memory;
- distinction between fixed topology and topology-changing optimization;
- explicit boundary/fallback/death behavior;
- open-path ordered transfer terminology;
- limitations wording.

### 16.3 Not required and actively deferred

- full closed-loop holonomy theorem;
- complete multi-gauge 360°/720° atlas;
- differentiation through chart/visibility events;
- retained-depth analytical marginalization for WorldFoam;
- stratified Lagrangian connection theorem in Paper A;
- adaptive-rank foam transfer;
- new Gaussian parameterization theorem.

## 17. Post-minimum breadth plan

Only after the narrow stop condition is satisfied, the full 21-context
manifest adds 14 contexts:

| Breadth role | Protocol | Seeds | Context count |
|---|---|---|---:|
| Alternate Coffee triplet | `coffee_martini_triplet_cam13_cam18_holdout_cam00_progressive_512_v1` | 17,29,43 | 3 |
| Alternate Coffee triplet | `coffee_martini_triplet_cam02_cam07_holdout_cam12_progressive_512_v1` | 17,29,43 | 3 |
| Additional Neural3D scene | `cook_spinach_full_300f_progressive_512_v1` | 17,29,43 | 3 |
| Additional Neural3D scene | `cut_roasted_beef_full_300f_progressive_512_v1` | 17,29,43 | 3 |
| Controlled D-NeRF | `dnerf_bouncingballs_matched_20f_progressive_512_v1` | 17 | 1 |
| Deterministic audit | `coffee_martini_full_300f_smoke_1step` | 17 | 1 |

These produce 42 additional ordinary lane records if every context has three
lanes, for 63 total lane records across all 21 contexts. The deterministic
one-step audit must remain separately labelled and
must never be aggregated with 512-wide quality results.

Use the distinct full-breadth output root:

`outputs/benchmarks/2026-07-28_world_tubes_full_public_matrix_schema2`

Never mix it with the minimum output root:

`outputs/benchmarks/2026-07-28_world_tubes_submission_matrix_schema2`

## 18. Risk register and mitigations

| Risk | Detection | Required response | Forbidden response |
|---|---|---|---|
| Mixed dirty source | clean-source gate or diff audit | transplant exact Paper A slice | blanket commit |
| Wrong Neural3D axes | calibration identity/hash mismatch | rebuild with v2 conversion | relabel old rows |
| Stale W&B run | finalized-file identity mismatch | rerun/finalize exact child | edit run ID by hand |
| MPS memory incident | preflight/live guard | move to quiet compliant host | bypass guard |
| Frozen parity failure | verifier threshold | diagnose first failing F/parameter | cite timing anyway |
| No measured speedup | valid raw timing | narrow to structural amortization | invent new method cycle |
| Camera curve dies early | closure/death verifier | publish bounded range | implement full orbit now |
| Public quality weak | heldout metrics/visuals | compiler-paper framing | post-hoc unbounded sweeps |
| Partial matrix | strict generator | finish missing context | hand-build table |
| Hash drift | artifact manifest | locate changed producer/input | update expected hash blindly |
| Generic Pandoc TeX | manuscript verifier | official venue conversion | waive final verifier |
| Citation error | primary-source audit | correct bibliography/prose | rely on remembered metadata |
| Native resolution OOM | eager-path guard | keep 512 claim or stream later | force 2704x2028 |
| CUDA distraction | missing parity/backend | keep MPS critical path | port before evidence |

## 19. Failure triage protocol

For every failure, record:

1. exact command;
2. start/end time;
3. clean source hashes;
4. host and live resource state;
5. stdout/stderr path;
6. partial artifact paths;
7. first violated invariant;
8. whether failure is correctness, evidence integrity, resource, environment,
   or scientific outcome;
9. smallest authorized next action;
10. whether all earlier rows remain valid.

Never merge rows across code commits after a behavioral change. If a fix can
affect results, invalidate and rerun every dependent component. Documentation-
only or venue-layout changes do not invalidate numeric evidence if the strict
artifact manifest still verifies.

## 20. Suggested ownership and parallel work

### Paper/source owner

- preserve clean source;
- own method/claim boundary;
- review code changes;
- maintain this plan and evidence ledger links.

### Experiment operator

- own host preflight and safety;
- launch one job at a time;
- retain logs and reports;
- run independent verifiers immediately after each job.

### Manuscript owner

- finish citations and exposition without changing method scope;
- integrate only generated evidence;
- own venue conversion and PDF QA.

### Independent verifier

- run artifact verification from retained files;
- compare source/data/native hashes;
- audit every numeric paper claim back to JSON;
- reject partial or hand-entered results.

These roles may be held by fewer people, but the verification pass should be
performed from a fresh context rather than by trusting the launcher's memory.

## 21. Day-by-day execution template

Actual durations depend on the chosen host and observed lane timing; do not
invent a schedule before preflight. Use this dependency-oriented sequence:

### Work block 1 — source freeze

- audit dirty main and STAR trees;
- preserve/transplant Paper A files;
- record clean commits;
- build/identify native extension.

### Work block 2 — cheap gates

- materialize dependencies;
- run focused tests;
- verify incomplete bundle;
- run preflight;
- run bounded schema-v2 smoke.

### Work block 3 — causal evidence

- dry-run and execute frozen same-world sweep;
- independently verify report;
- inspect parity before timing.

### Work block 4 — camera evidence

- execute variable-camera curve;
- independently verify closure/death monotonicity;
- freeze exact claim range.

### Work blocks 5–11 — public contexts

- one exact key per work block;
- fresh host check before every key;
- verify child immediately;
- inspect artifacts/W&B identity;
- only then move to next key.

### Work block 12 — strict evidence bundle

- validate all seven children and matrix summary;
- generate strict tables/plots/manifest;
- run independent artifact verification.

### Work blocks 13+ — paper package

- finish math exposition and related work;
- integrate generated evidence;
- create concept/system figures;
- finalize demo;
- convert venue template;
- build and visually inspect PDF;
- run strict manuscript verifier;
- update baselines/indexes;
- prepare local release bundle.

## 22. First commands for the next execution session

Run from `/Users/nicholasbardy/git/gsplats_browser/dynaworld`.

### Read-only orientation

```bash
git status --short --branch
git rev-parse HEAD
git -C third_party/fast-mac-gsplat status --short --branch
git -C third_party/fast-mac-gsplat rev-parse HEAD
sed -n '1,220p' \
  research_notes/gauged_uvt_trace_atlas/paper/generated/schema_v2/evidence_ledger.md
sed -n '1,240p' \
  TODO/world_tubes_paper_finish_master_plan_2026-08-13.md
```

### Do not launch training until

```text
clean source exists
focused tests pass
incomplete bundle verifies
candidate-host preflight passes
schema-v2 smoke passes
operator approves the launch window
```

Then execute P2, P3, and the seven P4 keys exactly as defined above.

## 23. Final claim audit

Before declaring completion, create a claim-to-evidence table and check every
sentence in the abstract, introduction contributions, results, and conclusion.
At minimum:

| Claim family | Required evidence |
|---|---|
| Gauge/fiber correctness | accepted theorem component |
| Schur marginalization | derivation plus executable theorem rows |
| Visibility crossing repair | raw-crossing and repaired theorem rows |
| Fixed-topology forward/VJP parity | theorem and frozen same-world reports |
| Structural frame amortization | trace/event scaling in theorem/frozen reports |
| Measured speed or break-even | warmed/repeated frozen raw timings only |
| Bounded moving cameras | accepted closure/death curve |
| Public behavior | seven schema-v2 contexts and 21 lane records |
| Memory/storage | explicitly defined byte/counter fields |
| Reproducibility | clean commits, manifests, demo, reproduction guide |
| Limitations | fallback/death/native-resolution/topology disclosures |

Delete or narrow any claim without its required accepted evidence. Adding a
sentence to the limitations section does not repair an unsupported headline
claim.

## 24. Final stop rule

Stop Paper A work when the definition of done in Section 4 is satisfied.

Do not wait for:

- the other 14 breadth contexts;
- WorldFoam native memory-light kernels;
- the connection/curvature formulation;
- retained-depth ordered-transfer promotion;
- full-orbit gauges;
- CUDA portability;
- native resolution;
- external SOTA reproduction.

At that point, the World Tubes compiler paper is a complete, honest narrow
submission. Move all subsequent scientific development into a separately
named Paper B/WorldFoam plan so that Paper A cannot reopen by drift.
