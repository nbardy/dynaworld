# World Tubes paper reproduction

The submission evidence contract is intentionally smaller than the historical
research tree. Run from the DynaWorld repository root.

## Environment

```bash
uv sync --group experiments
```

Before a publication row, materialize the canonical LPIPS/AlexNet weights once
in the same environment. This is the only step allowed to download them:

```bash
.venv/bin/python -c \
  'import lpips; lpips.LPIPS(net="alex", verbose=False).eval().cpu()'
```

The matrix preflight then hashes both the 244 MB AlexNet trunk and the packaged
LPIPS v0.1 linear weights. Training is rejected if either asset is absent or
drifted, so evaluation cannot trigger a late download or silently change the
metric.

Submission rows must run from a clean superproject plus clean STAR submodule.
The runner records both exact commits and now requires clean source by
default; `--require-clean-source` remains as an explicit compatibility flag.
Only a labelled mechanical smoke may opt out with `--allow-dirty-source`, and
such a run is ineligible for paper aggregation.

Evidence schema v2 additionally binds every consumed raw input, decoded target
and camera tensor, the exact ordered sample schedule, one canonical evaluator,
hardware/runtime and loaded native binaries, retained artifacts, and the
finalized W&B file. The three schema-v1 progressive rows do not contain those
identities and are historical diagnostics only. The canonical evaluator clamps
predictions to `[0,1]`, uses a fixed black background and no color calibration,
computes L1/MSE over all RGB elements, derives PSNR from global MSE, and
averages SSIM/LPIPS over the full declared image set. No selected-time row is
currently accepted under schema v2.

## Small local Metal playground

This runs World Tubes, dynamic 3DGS, and WorldFoam sequentially on the real
Coffee Martini train `cam04/cam09`, heldout `cam06` split. The checked-in
`coffee_martini_local_playground.jsonc` uses 32 frames, 80 updates, 128 to 256
primitives, and 64-wide to 128-wide images. Run from the repository root:

```bash
PYTHONPATH=src/train PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python research_experiments/paper_runner_suite/run_unified_paper_ablation.py \
  --execute --protocol src/train_configs/paper_protocols/coffee_martini_local_playground.jsonc \
  --seed 17 --device mps --out-dir outputs/local_playground \
  --worldfoam-initializer video --wandb-mode offline \
  --allow-local-mps-execution --allow-dirty-source
```

Use `--reuse-existing` to validate and reuse matching completed lanes. Edit the
JSONC for experiments. Each lane has a 3-GiB process-tree RSS guard, a 2-GiB
PyTorch MPS allocator cap, and 2 GiB reserved for the OS; RSS and device memory
can overlap, so their sum is conservative. The guard also stops on new host
swap growth above 256 MiB, low available memory, or output growth beyond 2 GiB.
It accounts for the launcher and descendants, including W&B. Old desktop swap
is recorded rather than treated as proof that the small job cannot fit.
The publication profiles retain their original resource gates.

The September 11 run completed all three 80-step lanes under these limits on
a **24-GiB Mac**. Peak sampled tree/launcher RSS was 1.75 GiB and the largest
reported MPS driver allocation was 1.08 GiB; no new swap was observed. The
selected runtime, backend checkouts, source, LPIPS weights, and whole scene
occupied 4.73 GiB; outputs including offline W&B were about 11.2 MB. Unrelated
datasets, historical outputs, Git history, and download caches are excluded
from that installation footprint. This is a measured small-job budget, not a
test on physical 8-GiB hardware or a fresh-install packaging claim.

Results are diagnostic: heldout PSNR was 6.50 / 5.96 / 6.64 dB for World Tubes /
dynamic 3DGS / WorldFoam. All use the same sample/pixel budget, but have different
parameter counts and initialization models. Quality is poor at this setting.
The tree is dirty and these runs do not enter the paper acceptance ledger or
`BASELINES.md` standings. W&B remains offline because automatic approval review
rejected uploading the generated media/configuration; artifacts are retained
locally. The run summary, checkpoints, previews, per-child memory receipts, and
offline W&B records are under:

```text
outputs/benchmarks/2026-09-11_local_playground/coffee_martini_local_playground/seed_17/
```

The separate two-step/four-frame smoke also ran `--frozen-world-replay-compiled`.
It **failed** the existing image, world-gradient, and fallback gates; its cold
compiled route was slower. Do not count it as sublinear scaling evidence or
launch a larger compiler sweep before diagnosing this failure. Its retained
row is in the sibling `coffee_martini_local_playground_smoke` directory.

## Lightweight one-command demo

Run the bounded paper demo without a dataset, training, W&B, MPS, or the
publication host:

```bash
.venv/bin/python \
  research_experiments/paper_runner_suite/run_world_tubes_lightweight_demo.py
```

This executes the existing decisive-demo implementation on one unchanged
synthetic world. The replay route materializes one trace cell per frame; the
compiled route uses one shared interval cell. Both are rendered by the actual
STAR UVT CPU reference renderer and differentiated with `torch.autograd` under
the same image adjoint. The command reports forward and world-VJP parity plus
trace, interval-entry, cell, dense-sample, and fallback counts. It writes:

```text
outputs/demos/world_tubes_lightweight/
  summary.json
  demo_manifest.json
  replay_compiled_error.png  # PPM fallback if Pillow is unavailable
```

`demo_manifest.json` hashes the report, image, source files, and ABI-matched
STAR UVT native binary. The bounded computation itself is deliberately the CPU
oracle path—the native projective kernels require MPS—but the command fails
before rendering when the packaged native binary is absent or cannot load. It
prints the repository-local build command in that failure. Verify an existing
demo independently with:

```bash
.venv/bin/python \
  research_experiments/paper_runner_suite/run_world_tubes_lightweight_demo.py \
  --verify-dir outputs/demos/world_tubes_lightweight
```

This is a runnable correctness and structural-reuse demonstration, not public
quality, warmed publication timing, or a replacement for the frozen-world
experiment below.

## One-command full-breadth public matrix

The complete selected-time public-data workload is fixed in
`src/train_configs/paper_protocols/world_tubes_full_public_matrix_v1.jsonc`.
It contains 21 independent protocol/seed rows: the seven Coffee Martini
progressive/control rows, six alternate-camera-triplet rows, six rows across
two additional Neural3D scenes, one separately labelled D-NeRF control, and
one 64-wide deterministic STAR correctness/timing audit that must not be
aggregated with the 512-wide quality rows.

The 21-row matrix uses selected-time training and evaluation to compare
representation quality, cost, and stored state. It is the full-breadth target,
not the minimum paper gate, and it does not replace the frozen
replay-versus-compiled experiment below.

On a sufficiently provisioned machine, the single reproduction command is:

```bash
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/paper_runner_suite/run_unified_paper_matrix.py \
  --execute \
  --require-clean-source \
  --reuse-existing \
  --matrix src/train_configs/paper_protocols/world_tubes_full_public_matrix_v1.jsonc \
  --out-dir outputs/benchmarks/2026-07-28_world_tubes_full_public_matrix_schema2 \
  --device mps \
  --wandb-mode online \
  --allow-local-mps-execution
```

Do not run this command on the workstation involved in the 2026-07-22 memory
incident. The manifest is a reproducibility contract, not a safety override.
The first acknowledgement is present because this command is specifically for
an operator-approved MPS execution host. The high-risk acknowledgement is
intentionally omitted, so the incident-calibrated preflight still refuses the
command on an undersized host. Moving to an adequately provisioned host is the
supported path.

## Minimum Coffee Martini control subset

The seven-run Coffee Martini control subset is source-complete for sequential,
bounded execution. Targets, rays, evaluation, Metal statistics, retained
media, and WorldFoam video initialization no longer require an eager full-video
tensor. The old `18.745/17.303 GiB` estimates and the inferred 32-GB minimum are
superseded. A quiet 16-GB Apple-Silicon host is now a candidate, not a promise:
the source-derived estimate must remain under the enforced 60% ceiling, the
live guard must pass, and each isolated child records peak process RSS and is
terminated if it crosses that ceiling. Start with one row and stop on any
memory/swap/resource violation.

Audit a candidate host without loading data or importing a renderer:

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

The command exits nonzero and reports every failed source/resource check until
the host is eligible. A passing preflight does not launch a run.

Run one previously-unaccepted row per invocation. Copy the exact key from the
runner's dry-run output, then replace `--run-key` for each remaining row:

```bash
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/paper_runner_suite/run_unified_paper_matrix.py \
  --execute \
  --require-clean-source \
  --reuse-existing \
  --max-new-runs 1 \
  --run-key coffee_martini_full_300f_progressive_512_v1/seed_17/fast_exploration \
  --matrix src/train_configs/paper_protocols/world_tubes_submission_matrix_v1.jsonc \
  --out-dir outputs/benchmarks/2026-07-28_world_tubes_submission_matrix_schema2 \
  --device mps \
  --wandb-mode online \
  --allow-local-mps-execution
```

Run the execution command only after the preflight passes on the
operator-approved host. Bounded execution writes a compact
`matrix_progress.json` and exits cleanly
without claiming matrix completion. Only after all seven rows validate does
the runner write the complete matrix summary and final artifacts. The MPS
acknowledgement does not bypass the live resource or incident-calibrated
memory gates. Do not add the high-risk acknowledgement on the incident
workstation.

The command must end with `matrix_summary.json`, `paper_rows.json`,
`paper_rows.csv`, `paper_table.md`, `paper_table.tex`, and
`heldout_psnr.svg`. Missing lane metrics fail the run instead of producing a
partial table.

Existing complete clean-source summaries can be aggregated without launching
any renderer or touching MPS:

```bash
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/paper_runner_suite/run_unified_paper_matrix.py \
  --aggregate-existing \
  --matrix src/train_configs/paper_protocols/world_tubes_full_public_matrix_v1.jsonc \
  --out-dir outputs/benchmarks/2026-07-28_world_tubes_full_public_matrix_schema2
```

This emits `existing_evidence_summary.json` and, when valid schema-v2 rows
exist, an `accepted_existing_evidence/` bundle. It reopens child identities,
raw and decoded data, evaluator/runtime/native contracts, W&B files, and
retained artifacts, so a legacy summary or partial lane cannot enter the
table.

The repository currently has `0/7` schema-v2-accepted minimum controls and
`0/21` full-breadth rows. The older `3/21` and seven-run `3/7` summaries are
historical schema-v1 artifacts, not authoritative ledgers. Their missing
identities cannot be reconstructed by aggregation; rerun the seven-row
submission subset on an adequate clean host, then extend to the other 14
breadth rows when resources permit.
Use the schema-v2 directory above rather than writing into the retained
2026-07-22 schema-v1 bundle.

## Same-representation scaling and theorem table

The verified bounded-fixture scaling artifact is:

```text
outputs/benchmarks/2026-07-22_world_tubes_same_representation_scaling_f4_128_cap256/summary.json
```

Regenerate the theorem table after changing a certified source artifact:

```bash
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/paper_runner_suite/world_tubes_theorem_table.py \
  --out-dir outputs/benchmarks/2026-07-22_world_tubes_theorem_table
```

## Public data

```bash
PYTHONPATH=src/train .venv/bin/python src/dataset_pipeline/neural_3d_video.py all \
  --config src/dataset_configs/neural_3d_video_paper_breadth.jsonc

PYTHONPATH=src:src/train .venv/bin/python src/dataset_pipeline/dnerf.py all \
  --config src/dataset_configs/dnerf_paper_breadth.jsonc
```

D-NeRF uses the posed-frame adapter described in
`research_notes/data_contract.md`. Official matched-time train/test poses are
discontinuous, so the current honest policy is a separately labelled
one-frame-per-chart gauged fallback; it must not be presented as the
sublinear bounded-chart result or injected into the synchronized multicamera
matrix.
Accordingly, the D-NeRF row is a distinct heterogeneous-manifest
negative/control. It is not part of synchronized multicamera aggregation and
does not satisfy the Neural3D frozen replay-versus-compiled scaling
requirement.

## Manuscript

### Frozen identical-world replay versus compiled atlas

Dry-run the exact lane-isolated command:

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

On an approved adequately sized MPS host, add both `--execute` and
`--allow-local-mps-execution`. Local MPS remains fail-closed unless that
incident-safety acknowledgement is explicit.
Execution also rechecks live resources before each expensive child: at least
10 GiB reclaimable memory, at most 2 GiB swap in use, at least 8 GiB free
disk, and one-minute load no greater than 0.75 per logical CPU. These are
safety and timing-integrity gates, not flags to bypass.
The accepted run must include `--max-frames 0` and
`--frame-counts 0,4,8,16,32,64,128`. It trains/saves once and evaluates every
sampling density from the same world; every `F` spans the full physical
interval and binds its exact integer frame indices and centered times. Before
publication, verify non-unit selected-time full-atlas versus chunk-slice
forward/VJP parity. The integrated runner preserves its original single-shot
route timings as correctness diagnostics, then collects independent,
alternating paired trials; publication eligibility requires at least one
warmup and three reported repeats, and the frozen command above requests five.
The report must include the checkpoint hash, image/loss/VJP parity, tensor
payload, raw timing samples and summaries, and fallback statistics. The tensor
payload excludes topology and transient working memory and must not be cited
as storage.

### Frozen learned-world bounded moving-camera density

After the static frozen sweep is accepted, run the checkpoint-only bounded-yaw
extension. It reuses the exact checkpoint and does not train again:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  .venv/bin/python \
  research_experiments/paper_runner_suite/run_frozen_world_moving_camera.py \
  --execute \
  --protocol \
  src/train_configs/paper_protocols/coffee_martini_full_300f_progressive_512_v1.jsonc \
  --seed 17 \
  --device mps \
  --wandb-mode online \
  --require-clean-source \
  --allow-local-mps-execution
```

The contract is fixed at direct $256\times256$ decode, bounded yaw from
$-22.5^\circ$ to $+22.5^\circ$, `F=8,16,32,64`, one midpoint first-order chart,
one warmup, and five paired repeats. It reports compiled-versus-replay parity,
not ground-truth quality at synthetic yaw poses. A mechanically complete failed
predeclared gate is retained as `complete_negative`; do not retune it.

### Bounded variable-camera closure/death curve

The submission currently consumes the already accepted clean paper-freeze
schema-v2 curve. Its exact raw digest is
`118f26857a1c51262f6d8b0a33d55ee037dc19a07713ce318aaab9878d5df198`,
with clean superproject source `33a64aa44efd430f56eb284915aa47b3e5ec2b7d`
and STAR source `6c9945258fb1b31c43418857eb5ead98e588fd77`.
Import and verify it with:

```bash
.venv/bin/python \
  research_experiments/paper_runner_suite/import_world_tubes_variable_camera_schema_v2.py
.venv/bin/python \
  research_experiments/paper_runner_suite/import_world_tubes_variable_camera_schema_v2.py \
  --verify-local
```

The importer uses a pinned legacy-schema compatibility decoder and explicitly
does not call the current schema-v1 runner. It accepts the 12-row curve ending
at the `170`-degree closure row and `179.5`-degree terminal compiler death. It
must not import the dirty 15-row `178/179`-degree schema-v1 diagnostic.

The following command is retained for a future new-schema rerun, not for
re-validating the accepted schema-v2 bytes.

Run the small CPU/Torch camera-program stress gate on a quiet host:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
.venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_variable_camera_closure_death_curve.py \
  --execute \
  --out-dir \
  outputs/benchmarks/2026-07-28_world_tubes_variable_camera_closure_death_curve
```

It keeps one world, one physical interval, and 64 requested samples fixed while
increasing bounded yaw motion. Verify the emitted report and its current source
bindings:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_variable_camera_closure_death_curve.py \
  --verify-report \
  outputs/benchmarks/2026-07-28_world_tubes_variable_camera_closure_death_curve/summary.json \
  --require-current-source
```

Any future accepted schema-v1 report must contain a closure prefix and a death suffix. The
reference uses exact rational trace centers and live per-sample depth order.
The result is empirical at the declared samples and covers a bounded open
camera path; it is not a continuous residual certificate, a `360/720` chart
transition, a closed-loop holonomy result, or a visibility-boundary gradient.

### Deterministic concept and system figures

Generate the two result-free vector figures referenced by the manuscript:

```bash
python3 \
  research_experiments/paper_runner_suite/generate_world_tubes_concept_figures.py
```

The generator is standard-library-only and reads no benchmark artifacts. It
writes `figures/world_tubes_system_overview.svg` and
`figures/world_tubes_projective_compiler.svg` with deterministic UTF-8 bytes.
Verify the checked-in files without rewriting them:

```bash
python3 \
  research_experiments/paper_runner_suite/generate_world_tubes_concept_figures.py \
  --verify-dir \
  research_notes/gauged_uvt_trace_atlas/paper/figures
```

### Submission-facing tables and figures

Generate the current honest artifact state:

```bash
python3 \
  research_experiments/paper_runner_suite/generate_world_tubes_paper_artifacts.py \
  --allow-incomplete
```

This writes explicit placeholders and an exact missing-input ledger while any
component is absent. After the seven-row matrix, static frozen sweep,
checkpoint-only moving-camera sweep, and variable-camera report all verify,
omit `--allow-incomplete`; the command then
fails unless every declared component is submission-ready. The selected
matrix's `output_root` supplies the default run root, so minimum and
full-breadth matrices cannot silently share summaries. Verify the current
placeholder bytes with:

```bash
python3 \
  research_experiments/paper_runner_suite/generate_world_tubes_paper_artifacts.py \
  --verify-dir \
  research_notes/gauged_uvt_trace_atlas/paper/generated/schema_v2 \
  --allow-incomplete
```

Omit `--allow-incomplete` for the strict evidence-bundle gate. Its readiness
field is scoped to generated evidence only; the manuscript/venue gate below is
still required. The theorem table is rederived from byte-pinned retained
reports and excludes old single-shot forward/backward timing ratios; warmed
repeated speed evidence comes only from the frozen-world component.

See `research_experiments/paper_runner_suite/PAPER_ARTIFACTS.md` for the exact
manuscript-facing files. Do not copy partial tables emitted by training
runners into the paper.

The real-video correctness tether retains its verified bounded-fixture source.
The Coffee progressive plot retains only historical schema-v1 provenance and
is not referenced by the manuscript:

```text
figures/real_video_equivalence.jpg
  <- outputs/benchmarks/2026-07-08_star_uvt_projective_decisive_demo_fixture/real_video_media/contact_sheet.jpg

figures/coffee_progressive_heldout_psnr.png
  <- outputs/benchmarks/2026-07-22_world_tubes_submission_matrix_clean_v1/accepted_existing_evidence/heldout_psnr.svg
```

The Coffee progressive figure is an archival diagnostic and must not be
presented as schema-v2 paper evidence. The manuscript's generated public table
emits no numbers until all seven schema-v2 controls are accepted.

```bash
pandoc --citeproc \
  research_notes/gauged_uvt_trace_atlas/paper/WORLD_TUBES_PAPER_DRAFT.md \
  --standalone --from markdown --to latex \
  --output research_notes/gauged_uvt_trace_atlas/paper/WORLD_TUBES_PAPER.tex
```

The authored concept figures are SVGs, so Pandoc's standalone TeX uses the
LaTeX `svg` package. A network-independent or venue build may instead convert
those deterministic SVGs to PDF and rewrite only a temporary manuscript copy;
do not replace the checked-in SVG sources.

Then run the Torch-free package gate:

```bash
python3 \
  research_experiments/paper_runner_suite/generate_world_tubes_paper_artifacts.py \
  --verify-manuscript \
  --allow-incomplete
```

`WORLD_TUBES_PAPER.tex` is generated: do not edit it manually. This Pandoc
output is a standalone article, not the venue package. After venue conversion,
omit `--allow-incomplete` from the package gate, build the PDF, convert the
accepted generated SVG plots to the venue-supported PDF/PNG form, and visually
inspect every page, table, figure, citation, and cross-reference before calling
the manuscript submission-ready.

The paper deliberately claims bounded tested chart segments. Do not restore
full `360/720` multi-gauge language unless the chart-transition implementation
and orbit test are both present.
