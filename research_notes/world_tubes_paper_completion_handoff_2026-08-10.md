# World Tubes Paper Completion Handoff

Date: 2026-08-10 KST

Scope: finish Paper A without absorbing the new WorldFoam connection algorithm

Owner contract: evidence closure, manuscript packaging, and submission only

## 1. Executive directive

The World Tubes paper has a coherent method, implementation spine, working
manuscript, and fail-closed evidence pipeline. It does not need another theory
cycle. The next paper owner should preserve the current source, run the frozen
experiments on an approved host, populate the schema-v2 evidence bundle, and
finish the manuscript.

The new stratified Lagrangian/measure-connection work is a separate research
lane. It may eventually become a WorldFoam paper or a follow-on ablation, but
it must not change this paper's method, code path, experiment queue, or title.
The split is deliberate:

```text
Paper A being handed off:
  World Tubes in Gauged Camera Space
  -> dynamic Gaussian world
  -> known camera-program compiler
  -> reusable projective sensor-time trace atlas
  -> fixed-topology compiled adjoint

Separate research lane:
  WorldFoam constrained Lagrangian optical connection
  -> retained depth and ordered optical transfer
  -> new oracle/compression hypotheses
  -> no role in closing Paper A
```

Current submission evidence is not nearly complete despite the mature source
and manuscript:

```text
accepted theorem component:         yes
accepted public context rows:       0 / 7 minimum, 0 / 21 breadth
accepted frozen compiler sweep:     no
accepted variable-camera curve:     no
strict evidence bundle:             incomplete
venue-ready and visually checked PDF: no
```

The authoritative current ledger is
`research_notes/gauged_uvt_trace_atlas/paper/generated/schema_v2/evidence_ledger.md`.
It names ten unresolved evidence records: one matrix summary, seven public run
summaries, one frozen-world summary, and one variable-camera summary. These are
nine runtime jobs because the matrix summary is derived after all seven public
runs validate.

## 2. Frozen paper claim

Keep the title:

> **World Tubes in Gauged Camera Space: Frame-Amortized Dynamic Gaussian
> Rendering**

Keep the primary claim:

> Known or low-dimensional camera programs let dynamic Gaussian primitives be
> compiled into reusable sensor-time world tubes. In the tested
> fixed-topology training-step/world-VJP regime, the dominant world-side
> projection, support, binning, visibility-metadata, and backward-replay work
> scales with trace/event complexity rather than being blindly repeated for
> every requested frame.

The invariant mathematical object remains

\[
T_i = \pi_*\Gamma^* w_i,
\]

where the camera program pulls a camera-independent world primitive onto its
ray bundle and the ray-depth fiber is pushed forward into a sensor-time trace.
In a nonsingular affine local gauge, a strict spacetime Gaussian remains
Gaussian under depth marginalization, and the UVT footprint follows from a
Schur complement. Conditional depth mean and variance are retained for
support and visibility certification even though depth is marginalized from
the fast footprint.

The gauged math is not merely a STAR optimization. It is the part that makes a
moving camera and a large-motion depth-order crossing well-defined:

```text
Gauged camera-ray pushforward     = method
conditional depth/order strata   = method
bounded projective chart domains = method
Projective STAR UVT               = primary implementation backend
```

Do not weaken this to “STAR with a faster kernel.” The raw crossing fixture
has quality error `0.186742`; visibility stratification repairs it to `0` in
the accepted theorem artifact.

The paper does **not** claim:

- sublinear materialization of `F * H * W` output samples;
- end-to-end sublinear training under topology invalidation and recompilation;
- state-of-the-art dynamic novel-view quality;
- a universal replacement for 4DGS or deformable-Gaussian methods;
- exact generic camera paths, complete `360/720` chart transitions, or
  visibility-boundary derivatives;
- that SPD(4) Gaussians are novel;
- that retained-depth WorldFoam or the new connection algorithm is complete.

The strongest causal baseline is per-frame replay of the **same learned
World Tubes world**. External 3DGS/4DGS methods are contextual baselines, not a
substitute for that comparison.

## 3. Method and naming boundary

| Name | Paper role | Status for this handoff |
| --- | --- | --- |
| Gauged UVT / camera-ray bundle | Mathematical framework | Keep central. |
| World Tubes | Paper A method | Finish now. |
| Projective STAR UVT | Sparse Metal implementation | Primary backend. |
| Native SPD(4), Beer--Lambert | Bounded source/physics extensions | Optional scoped evidence only. |
| World Tubes + Ordered Ray Transfer | Retained-fiber robustness ablation | Post-paper unless already selective. |
| WorldFoam | General retained-depth/cellular representation | Separate paper lane. |
| Constrained Lagrangian optical connection | New WorldFoam theorem/compression hypothesis | Explicitly out of scope. |

An open ray uses ordered parallel transport or a path-ordered transfer
product. Reserve “holonomy” for a closed loop. Do not rename the paper or
shader around ray holonomy.

## 4. Exact implementation truth

### 4.1 Implemented mainline

- The historical STAR UVT/projective interval backend has event-certified
  bounded camera charts, interval compression, Metal forward, and direct VJP.
- The VJP differentiates trace coefficients, opacity, temporal opacity,
  spatial precision, depth-affine fields, and color while compiled topology,
  support, tile membership, visibility order, and fallback choices remain
  fixed.
- The typed protocol and exact-coverage sampler are implemented in
  `src/train/paper_training_types.py` and
  `src/train/paper_training_protocol.py`.
- `run_unified_paper_ablation.py` runs World Tubes, WorldFoam, and dynamic
  3DGS through one protocol/cost/evaluator contract.
- `run_unified_paper_matrix.py` expands the seven-row minimum or 21-row
  breadth manifest, runs one resumable row at a time, and rejects partial or
  stale evidence.
- Evidence schema v2 binds the exact ordered sample schedule, all consumed raw
  inputs, decoded targets and cameras, the canonical evaluator, runtime and
  loaded native binaries, retained checkpoints/media/configs, and the actual
  finalized W&B file.
- The canonical evaluator clamps to `[0,1]`, uses fixed black background and
  no color calibration, computes global RGB L1/MSE, derives PSNR once from
  global MSE, and averages SSIM/LPIPS over the full declared image set.
- `run_frozen_world_replay_compiled.py` is implemented to train and hash one
  world once, then evaluate per-frame replay and compiled-atlas routes at
  `F={4,8,16,32,64,128,full}` from that same checkpoint and physical interval.
- `projective_variable_camera_closure_death_curve.py` is implemented to
  compare one compiled bounded camera program with an exact rational,
  per-sample live-depth-order oracle.
- `generate_world_tubes_paper_artifacts.py` is a Torch-free, fail-closed
  submission generator. It emits placeholders rather than partial numbers.
- The Markdown manuscript, generated standalone TeX, bibliography, and
  reproduction guide exist.

### 4.2 Implemented but not accepted evidence

- A staged four-frame/two-step MPS smoke and an all-300-frame one-step MPS
  smoke completed all three lanes. They prove mechanics, K-frame accounting,
  growth, gradients, evaluation, and W&B plumbing, not benchmark quality.
- Three progressive-512 Coffee Martini seeds completed all three lanes under
  schema v1. They are historical diagnostics only.
- The frozen-world and schema-v2 code received documented static hardening,
  but the latest source has not produced an accepted runtime artifact.
- The variable-camera runner has contract-test coverage, but its real CPU
  curve is missing.
- The generated schema-v2 bundle accepts theorem correctness only; the public,
  frozen, and variable-camera tables are explicit placeholders.
- The manuscript's current standalone Pandoc TeX is not a venue manuscript
  and has no clean, visually inspected PDF.

### 4.3 Two independent reasons old Neural3D results are non-authoritative

The July progressive rows cannot be promoted or relabelled:

1. They lack evidence-schema-v2 identities that were never recorded.
2. The July 31 LLFF camera-axis correction superseded the old calibration.

Every new Neural3D paper artifact must carry
`neural_3d_llff_opencv_relative_pinhole_v2`. The old
`neural_3d_llff_relative_pinhole` conversion misread raw LLFF
`[down,right,back]` axes and is diagnostic only.

### 4.4 Source is not clean or durably landed

At handoff time:

```text
superproject HEAD: cb0a904514658a0d4d5b0c2f9f9b8759ddabf448
STAR submodule HEAD: 64a4e0a2414c3b70d881d1c51632c73985c74ba4
superproject state: dirty
STAR submodule state: dirty
```

Core paper runners, protocol code, configs, manuscript files, and indexes are
modified. The frozen runner, artifact generator, variable-camera runner,
bibliography, generated ledger, and focused tests are among the untracked
files. Publication execution requires clean main and STAR trees and will
reject this checkout.

The first owner action is therefore an intentional paper-slice preservation
and review. Do not blanket-commit the dirty tree: it also contains unrelated
WorldFoam, browser, and user work. Land or transplant only the exact Paper A
source into a clean branch/worktree, then record both commits in every run.

## 5. Accepted evidence today

The only accepted schema-v2 submission component is the bounded theorem table:

| Claim | Recorded value | Gate |
| --- | ---: | ---: |
| Fiber value gauge invariance | `3.50087e-13` relative error | `<=1e-10` |
| Fiber gradient gauge invariance | `2.32523e-12` relative error | `<=1e-9` |
| Compiled atlas vs dense/replay image | `0` max abs error | `<=1e-5` |
| Raw order crossing exposes failure | `0.186742` quality error | expected `>1e-5` |
| Visibility stratification repairs crossing | `0` quality error | `<=1e-5` |
| Exposure/rolling-shutter forward parity | `5.96046e-8` | `<=1e-5` |
| Exposure/rolling-shutter VJP parity | `6.37738e-7` | `<=1e-5` |
| Mixed fallback VJP parity | `7.40632e-7` | `<=1e-5` |
| Fixed/replay trace-count ratio at `F=128` | `0.03125` | `<0.25` |

The bounded `F=4..128` fixture also records fixed logical tensor volume growing
`1x` while replay grows `32x`. This is structural reuse evidence. Logical
tensor volume excludes topology, packed bins, allocator overhead, and
transient working memory; historical single-shot timings are not publication
timing.

Bounded native-SPD(4), Beer--Lambert, retained-fiber, decisive-demo,
visibility-stress, and short quality rows may be retained as clearly labelled
engineering/appendix evidence. They do not replace the missing public causal
experiment or the schema-v2 context table.

## 6. Minimum experiment queue

Execute this queue in order. A failure should produce a preserved negative
artifact, not another architecture branch.

### P0. Preserve a clean executable source

1. Review the current paper diff and STAR submodule diff.
2. Select only the Paper A runners, protocols, backend changes, tests,
   manuscript, bibliography, and generated-placeholder pipeline.
3. Land them in intentional commits or transplant them into a clean worktree.
4. Confirm both source trees are clean and record both exact commits.
5. Confirm all Neural3D protocols decode with the v2 LLFF/OpenCV calibration.

### P1. Behavior and evidence-plumbing verification

1. Run the focused paper protocol, schema-v2, matrix, frozen-world,
   variable-camera, and artifact-generator tests on a quiet host.
2. Verify the current incomplete generated bundle.
3. Run one bounded three-lane schema-v2 evidence smoke. Require real W&B file
   discovery, decoded-bundle equality, evaluator equality, route-native
   identity, retained-artifact identity, and stale/reuse rejection.
4. Do not use the smoke in a paper table.

### P2. Frozen identical-world causal sweep

Run the lane-isolated World Tubes executor before the three-representation
matrix. It trains once and evaluates the same world over the same full physical
interval at:

```text
F = full, 4, 8, 16, 32, 64, 128
seed = 17
timing = at least 1 warmup and 3 repeats; checked command uses 1/5
```

This is the central compiler result. It must bind one checkpoint, world-state
hash, target grid, evaluator, native extension, and clean source. Every `F`
uses ordered samples spanning the full interval, not a growing prefix. Verify
non-unit full-atlas versus one-frame chunk-slice forward and VJP parity before
using any speed number.

### P3. Bounded variable-camera closure/death curve

Run the CPU/Torch bounded camera stress with one world, 64 samples, and one
physical interval fixed while yaw span increases. The accepted report must
show a monotone closure prefix followed by a death suffix. This is the evidence
for the moving-camera claim. It remains a bounded open-path result, not a full
orbit transition or continuous boundary theorem.

### P4. Seven-row schema-v2 public context table

Run exactly one previously unaccepted key per matrix invocation:

| Role | Protocol | Seeds |
| --- | --- | --- |
| Primary progressive | `coffee_martini_full_300f_progressive_512_v1` | `17,29,43` |
| Pixel-matched fixed | `coffee_martini_full_300f_fixed_512_pixel_matched_v1` | `17,29,43` |
| Global-shuffle sampler | `coffee_martini_full_300f_progressive_global_shuffle_512_v1` | `17` |

The progressive schedule has 600 steps over `128w -> 256w -> 512w`; the fixed
control has 300 steps at 512w with the exact same target-pixel budget. Every
protocol declares 300 synchronized frames, trains `cam04/cam09`, and holds out
`cam06`.

The matrix is selected-time representation-and-cost context. It produces 21
lane records—World Tubes, WorldFoam, and dynamic 3DGS for each of seven runs.
It is **not** compiled-atlas evidence.

### P5. Generate strict paper artifacts

After the frozen report, variable-camera report, seven run summaries, and
canonical matrix summary all verify, run the submission generator without
`--allow-incomplete`. It must produce a complete hash manifest, theorem table,
public-context table, frozen scaling table, variable-camera table, and plots.
Never copy partial training-runner tables into the paper.

### P6. Finish the manuscript

1. Replace no claim with schema-v1 numbers.
2. Finish integrated citations and bibliography.
3. Add the concept/system diagrams and accepted generated plots.
4. Package one runnable demo command and artifact manifest.
5. Convert the standalone Pandoc output to the target venue template.
6. Build a clean PDF and visually inspect every page, figure, table, equation,
   citation, cross-reference, and appendix.
7. Run the strict manuscript-package verifier.

### Post-minimum breadth

Only after the minimum cut is complete, consider the other 14 matrix rows:
six alternate Coffee Martini triplet rows, six rows on two additional
Neural3D scenes, one D-NeRF control, and one separately labelled deterministic
timing audit. They strengthen breadth but do not block the narrow compiler
paper.

## 7. Acceptance gates

### 7.1 Frozen compiler sweep

Every row and the aggregate must satisfy:

```text
same checkpoint and unchanged world across all F
same full physical interval and exact ordered sample identities
image max abs error                         <= 1e-5
loss absolute delta                        <= 1e-5
global world-VJP normalized L2 error       <= 1e-5
max per-parameter VJP normalized L2 error  <= 1e-5
minimum replay/compiled world-VJP norm     >  1e-12
fallback fraction                          <= 0.20
non-unit atlas/chunk slice parity          accepted
timing warmups                             >= 1
timing repeats                             >= 3
all raw timing algebra                     verifier-rederived
```

If parity passes but timing does not improve, report structural amortization
without an end-to-end speedup claim. Do not hide the result or retune the
method through an unrelated architecture cycle.

### 7.2 Variable-camera curve

An accepted closure row requires:

```text
image PSNR                   >= 50 dB
image p99.9 absolute error   <= 2/255
image max absolute error     <= 4/255
world-VJP relative L2 max    <= 0.02
fallback cell fraction       <= 0.20
fallback sample fraction     <= 0.20
invalid sample fraction      == 0
trace/replay ratio           < 0.50
interval/dense ratio         < 0.80
no unresolved/stale chart state
```

The complete curve must contain at least one accepted closure row, at least one
death row, and no return to closure after the first death.

### 7.3 Each public context row

Require:

- clean and exact main/STAR source commits;
- current v2 calibration and exact raw/decoded dataset identities;
- exact ordered schedule, all declared steps, and all 300 frames;
- exact target and rasterized frame/pixel accounting;
- finite heldout PSNR, SSIM, LPIPS, and L1 from the canonical evaluator;
- synchronized compile/forward/backward/optimizer timing;
- parameters, parameter bytes, optimizer bytes, checkpoint bytes, and sampled
  current/driver peak memory;
- representation-specific trace/event/fallback diagnostics;
- train and heldout media;
- finalized exact W&B file identity;
- exact run-summary equality when embedded into the matrix summary.

The matrix gate requires exactly seven ordered run keys and 21 lane records.
No partial numeric manuscript table is acceptable.

### 7.4 Claim-level targets

The experiment plan's strong targets are useful success criteria, not reasons
to fabricate acceptance:

```text
>= 3x less projection/support/binning metadata work at F>=32
>= 1.3x end-to-end speedup on a many-sample workload
<= 0.1-0.3 dB loss versus same-representation replay
gradient relative error <= 1e-5
ordinary-scene fallback <= 10-20%
break-even by 16-32 samples on smooth camera paths
```

The minimum arXiv result is narrower: verified bounded correctness, clear
sublinear structural world-side scaling, same-representation quality parity,
one honest public context table, measured limitations, and one runnable demo.

## 8. Host safety and resource policy

The 2026-07-22 fixed-512 attempt was killed during severe unified-memory,
compressor, swap, and `kernel_task` pressure. Its partial outputs are invalid.
The 24 GiB incident workstation is not authorized for publication-scale MPS.
Do not bypass the guard or run paper lanes concurrently there.

Current incident-calibrated selected-matrix estimates are:

```text
progressive/global-shuffle: 18.745 GiB
fixed-512:                  17.303 GiB
guard ceiling:              60% of physical unified memory
```

That makes a clean Apple host with at least 32 GiB unified memory the minimum
currently supported MPS target for this eager 512-wide three-lane protocol.
This is an operational safety envelope, **not** an intrinsic World Tubes
memory requirement. The method and bounded kernels have run locally at much
smaller scale. Target/video residency, baseline per-frame state, optimizer
state, allocator headroom, and the three-lane evidence contract drive this
guard.

Every expensive child must pass a fresh live-resource check:

```text
reclaimable memory:                  >= 10 GiB
swap in use:                         <= 2 GiB
free disk:                           >= 32 GiB
one-minute load / logical CPU:       <= 0.75
```

Other rules:

- run one representation/process at a time;
- use resumable one-row matrix execution;
- never infer authorization from the existence of `--execute` flags;
- keep publication execution off the incident Mac;
- do not claim native 2704x2028 support—the current 512-wide path still has
  eager host-decoded targets;
- native resolution requires on-demand K-frame decode, bounded caching,
  selected-sample rays, and streamed evaluation first.

A B200 is not a drop-in execution substitute. The publication implementation
is STAR/PowerFoam/fast-mac Metal/MPS. Moving the paper matrix to CUDA requires a
separately verified native port and would become a new engineering dependency.
Do not let a CUDA port block Paper A; use a sufficiently provisioned Apple
host unless an already validated CUDA path exists.

## 9. Do not reopen while finishing the paper

- the new WorldFoam measure/connection/curvature algorithm;
- new gauge theories, aliases, or full-orbit formalisms;
- native WorldFoam or adaptive M3/M5 material selection;
- WT-OT0--3 multi-seed experiments or dense retained-fiber shader work;
- projective retained-depth integration or adaptive quadrature;
- SPD(4) novelty claims or another Gaussian representation branch;
- browser training, V-JEPA/world tokens, 300-clip feature sweeps, Softmax
  variants, or `direct_serial` promotion;
- native 2704x2028 quality runs;
- external SOTA reproduction or a CUDA port as a prerequisite;
- support/opacity sweeps that do not close a declared paper gate;
- manual promotion of the old `paper_ready=true`, `3/7`, or `3/21` artifacts;
- manual edits to `WORLD_TUBES_PAPER.tex`; regenerate it from Markdown.

Do not delete these branches or their artifacts. Preserve provenance, but do
not route paper time into them.

## 10. Code and document map

| Purpose | Canonical path |
| --- | --- |
| This completion handoff | `research_notes/world_tubes_paper_completion_handoff_2026-08-10.md` |
| Manuscript source | `research_notes/gauged_uvt_trace_atlas/paper/WORLD_TUBES_PAPER_DRAFT.md` |
| Experiment/claim plan | `research_notes/gauged_uvt_trace_atlas/paper/WORLD_TUBES_EXPERIMENT_PLAN.md` |
| Reproduction commands | `research_notes/gauged_uvt_trace_atlas/paper/REPRODUCE.md` |
| Bibliography | `research_notes/gauged_uvt_trace_atlas/paper/WORLD_TUBES_REFERENCES.bib` |
| Generated working TeX | `research_notes/gauged_uvt_trace_atlas/paper/WORLD_TUBES_PAPER.tex` |
| Active paper TODO | `TODO/unified_paper_ablation_pipeline.md` |
| Current evidence ledger | `research_notes/gauged_uvt_trace_atlas/paper/generated/schema_v2/evidence_ledger.md` |
| Artifact contract | `research_experiments/paper_runner_suite/PAPER_ARTIFACTS.md` |
| Typed protocol | `src/train/paper_training_types.py` |
| Sampler/protocol resolver | `src/train/paper_training_protocol.py` |
| Single three-lane runner | `research_experiments/paper_runner_suite/run_unified_paper_ablation.py` |
| Matrix runner | `research_experiments/paper_runner_suite/run_unified_paper_matrix.py` |
| Frozen causal executor | `research_experiments/paper_runner_suite/run_frozen_world_replay_compiled.py` |
| Variable-camera gate | `research_experiments/star_uvt_feature_tubes/projective_variable_camera_closure_death_curve.py` |
| Submission generator | `research_experiments/paper_runner_suite/generate_world_tubes_paper_artifacts.py` |
| Synthetic theorem generator | `research_experiments/paper_runner_suite/world_tubes_theorem_table.py` |
| Minimum seven-row manifest | `src/train_configs/paper_protocols/world_tubes_submission_matrix_v1.jsonc` |
| Full 21-row manifest | `src/train_configs/paper_protocols/world_tubes_full_public_matrix_v1.jsonc` |
| Mechanical evidence smoke | `src/train_configs/paper_protocols/world_tubes_evidence_smoke_matrix.jsonc` |
| Projective STAR backend | `third_party/fast-mac-gsplat/variants/star_uvt_v0/` |
| Core contract tests | `tests/test_paper_training_protocol.py`, `tests/test_unified_paper_ablation.py`, `tests/test_unified_paper_matrix.py` |
| Frozen executor tests | `tests/test_frozen_world_replay_compiled.py` |
| Variable-camera tests | `tests/test_star_uvt_projective_variable_camera_closure_death_curve.py` |
| Artifact tests | `tests/test_world_tubes_paper_artifacts.py` |
| Data/calibration contract | `research_notes/data_contract.md` |
| Full July synthesis | `research_notes/world_tubes_spd4_worldfoam_handoff_2026-07-28.md` |

The July handoff is valuable archaeology but contains a now-superseded
`3/21` acceptance count. Use this handoff, the current TODO, and the generated
schema-v2 ledger for operational truth.

## 11. First commands for the next paper owner

Run all commands from the DynaWorld root. The first block is read-only and safe
on the current checkout:

```bash
git status --short --branch
git rev-parse HEAD
git -C third_party/fast-mac-gsplat status --short --branch
git -C third_party/fast-mac-gsplat rev-parse HEAD
sed -n '1,220p' \
  research_notes/gauged_uvt_trace_atlas/paper/generated/schema_v2/evidence_ledger.md
```

After intentionally preserving the Paper A slice, move to a clean clone or
worktree. Confirm both trees are clean before any publication evidence:

```bash
test -z "$(git status --porcelain)"
test -z "$(git -C third_party/fast-mac-gsplat status --porcelain)"
```

On a quiet development host, run the focused contract suite:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
.venv/bin/python -m pytest \
  tests/test_paper_training_protocol.py \
  tests/test_unified_paper_ablation.py \
  tests/test_unified_paper_matrix.py \
  tests/test_frozen_world_replay_compiled.py \
  tests/test_star_uvt_projective_variable_camera_closure_death_curve.py \
  tests/test_world_tubes_paper_artifacts.py -q
```

Verify the honest incomplete bundle before producing new evidence:

```bash
python3 \
  research_experiments/paper_runner_suite/generate_world_tubes_paper_artifacts.py \
  --verify-dir \
  research_notes/gauged_uvt_trace_atlas/paper/generated/schema_v2 \
  --allow-incomplete
```

Run the non-launching candidate-host audit:

```bash
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/paper_runner_suite/run_unified_paper_matrix.py \
  --preflight-only \
  --matrix \
  src/train_configs/paper_protocols/world_tubes_submission_matrix_v1.jsonc \
  --out-dir \
  outputs/benchmarks/2026-07-28_world_tubes_submission_matrix_schema2 \
  --device mps \
  --wandb-mode online \
  --check-wandb-connectivity
```

On an operator-approved quiet MPS host, first run the bounded evidence smoke:

```bash
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/paper_runner_suite/run_unified_paper_matrix.py \
  --execute \
  --require-clean-source \
  --matrix \
  src/train_configs/paper_protocols/world_tubes_evidence_smoke_matrix.jsonc \
  --out-dir outputs/benchmarks/2026-08-10_world_tubes_schema2_evidence_smoke \
  --device mps \
  --wandb-mode offline \
  --allow-local-mps-execution
```

Dry-run the frozen causal sweep before adding `--execute` on the approved host:

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

Run and verify the bounded variable-camera gate on a quiet host:

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

Then run one exact public matrix key per invocation. Copy keys from dry-run
output; begin with:

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

After all required summaries verify, regenerate the strict evidence bundle and
the working TeX:

```bash
python3 \
  research_experiments/paper_runner_suite/generate_world_tubes_paper_artifacts.py

pandoc --citeproc \
  research_notes/gauged_uvt_trace_atlas/paper/WORLD_TUBES_PAPER_DRAFT.md \
  --standalone --from markdown --to latex \
  --output \
  research_notes/gauged_uvt_trace_atlas/paper/WORLD_TUBES_PAPER.tex
```

Convert that working TeX to the selected venue template, build and visually
inspect the PDF, then run the final manuscript-package gate:

```bash
python3 \
  research_experiments/paper_runner_suite/generate_world_tubes_paper_artifacts.py \
  --verify-manuscript
```

The last verifier must run after venue conversion and PDF visual QA. The
generated evidence bundle's `submission_ready` field covers evidence only; it
does not certify the manuscript layout.

## 12. Stop condition

Paper A is complete only when all of the following are true:

```text
clean, recorded superproject and STAR commits
focused source/behavior gates pass
frozen same-world sweep accepted
bounded variable-camera closure/death curve accepted
seven schema-v2 public context rows accepted
strict artifact generator accepted
paper tables and plots generated only from accepted JSON
citations and venue conversion complete
one demo and reproducibility manifest complete
clean PDF built and visually inspected
BASELINES.md receives dated rows only after verifier acceptance
```

At that point, stop. Do not wait for the other 14 breadth rows, WorldFoam
connection experiments, native resolution, CUDA portability, or external SOTA
reproduction to call the narrow World Tubes compiler paper complete.
