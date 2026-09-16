# Unified Paper Ablation Pipeline

## 2026-08-25 Execution Handoff

The authoritative schema-v2 ledger currently has two accepted components and
three missing runtime components:

- accepted: theorem/correctness and variable-camera closure/death;
- missing: frozen identical-world scaling;
- missing: checkpoint-only bounded moving-camera density;
- missing: all seven public Coffee Martini protocol rows (`0/21` lane rows).

This is no longer a data-acquisition or new-mathematics blocker. The
bounded-memory/runner integration is source-complete and has passed two
bounded read-only review passes with no remaining P0/P1 finding. Per the
resource stop, no Python, tests, builds, training, or accelerator work was run
in this preparation pass. The current worktree is still uncommitted; run the
focused behavior gates, then commit one clean superproject/STAR pair before
evidence execution.
The moving-camera route is a recovery/adaptation of the previously reviewed
`a290862` / STAR `6c99452` implementation, not a new camera formalism. Its
frozen contract is direct `256x256`, yaw `-22.5..+22.5` degrees,
`F={8,16,32,64}`, one midpoint first-order chart, and checkpoint-only replay.
The gauged camera-ray, conditional-depth, event/order, and projective-trace
math remains part of World Tubes; “holonomy” is not used as a literal method
name for this open-path experiment.

The current source preparation streams Neural3D targets, selected rays,
evaluation, Metal statistics, retained media, and WorldFoam video
initialization. Frozen selected-time parity and exact p99.9 accumulation must
also stay chunk-bounded. Each expensive child now has an independent process
RSS guard in addition to renderer allocator counters. A quiet 16-GB host is a
candidate for the sequential run, not a guaranteed fit; the first row is the
runtime calibration and the campaign stops on a live gate, RSS guard, swap
growth, or any failed frozen contract.

Run order after a clean source-only review:

1. Static frozen learned-world sweep: train once, then evaluate
   `F={full,4,8,16,32,64,128}` with one warmup/five repeats.
2. Moving-camera density: reuse that exact checkpoint; no training;
   `F={8,16,32,64}` at `256x256`.
3. Seven public rows, one invocation/row: progressive seeds `17/29/43`,
   pixel-matched fixed seeds `17/29/43`, global-shuffle seed `17`.
4. Generate the schema-v2 artifact bundle, regenerate TeX, convert to the
   selected venue package, and visually inspect the PDF.

Do not add another audit, queue wrapper, venue scaffold, renderer family, or
math branch before these evidence counts move. Exact commands and gates live
in `research_notes/gauged_uvt_trace_atlas/paper/REPRODUCE.md`.

## Implemented And Verified

- One typed protocol resolves dataset identity, camera split, full temporal
  count, progressive stages, K, grouping, target-frame budget, and target-pixel
  budget.
- One coverage-exact sampler visits every train `(camera, time)` pair once per
  epoch and supports same-time plus local-time grouping.
- World Tubes uses STAR UVT Metal selected-time rendering with
  `direct_atomic + index_add` for the throughput row.
- Dynamic 3DGS uses fast-mac Metal and active-prefix capacity stages.
- WorldFoam uses PowerFoam raytrace Metal and optimizer-state-preserving cell
  growth.
- All lanes report target/raster frames and pixels, parameters, parameter
  bytes, optimizer bytes, serialized checkpoint bytes, optimizer steps,
  device-synchronized compile/forward/backward/optimizer timing, sampled peak
  current/driver memory, LPIPS, and representation-specific trace/event/fallback
  diagnostics.
- The 4-frame staged MPS smoke and all-300-frame MPS smoke are complete.
- `run_unified_paper_matrix.py` expands a declarative protocol/seed/policy
  matrix, validates every lane fail-closed, and emits row JSON, CSV, Markdown,
  LaTeX, and SVG plot artifacts.
- The bounded synthetic same-representation replay-versus-compiled scaling row
  is complete
  for `F={4,8,16,32,64,128}`. The compiled payload remains fixed while replay
  grows `32x`; the checked theorem table consumes this result and the existing
  certified synthetic fixtures.
- The paper claim is explicitly bounded to tested camera-chart segments; no
  unimplemented `360/720` multi-gauge transition is claimed.
- Submission runs can require a clean superproject and STAR submodule state;
  both exact commits are recorded. W&B code/diff upload is disabled for these
  runs because resolved configs plus clean commit hashes are the reproducible
  source contract.

Evidence schema v2 is implemented and source-reviewed, but the new bounded
residency path is not yet runtime-verified. It binds the exact ordered sample schedule, all raw
inputs, decoded targets and camera programs, the canonical evaluator, runtime
and loaded native binaries, every retained lane artifact, and finalized W&B
files. It also fixes cross-lane PSNR aggregation to derive PSNR from global
MSE. This audit invalidates schema-v1 acceptance: the three completed
progressive rows remain historical numerical diagnostics, but one WorldFoam
W&B identity is stale/mismatched and none of the rows carries the full v2
contract. The current accepted ledger is therefore `0/21`.
The minimum submission-control subset is separately `0/7`; the other 14 rows
are the stronger breadth target.

## P0: Produce Paper Rows

1. On the approved quiet host, use the first real row as the bounded runtime
   calibration. It must verify actual W&B remote/file identity, decoded-bundle
   equality, route-native identity, evaluator equality, reuse rejection,
   process peak RSS, and matrix aggregation. A smoke does not count as an
   ablation or advance the ledger.
2. Run the implemented lane-isolated frozen identical-world comparison via
   `run_frozen_world_replay_compiled.py` with
   `--frame-counts 0,4,8,16,32,64,128`. It now trains and saves once, evaluates
   every `F` from the exact same world, and samples each row across the same
   full physical interval. First verify non-unit selected-time full-atlas
   versus chunk-slice forward/VJP parity. The integrated sweep retains
   single-shot correctness timings and separately records alternating paired,
   synchronized timing trials; use at least one warmup and three repeats
   (`1/5` is the publication-runner default).
3. Run the checkpoint-only bounded learned-world moving-camera density gate
   from the accepted static checkpoint. The separate variable-camera
   closure/death component is already accepted.
4. Rerun the structured progressive 512-wide protocol for seeds 17/29/43
   under evidence schema v2.
5. Run the exact target-pixel-matched fixed-512 control for seeds 17/29/43.
6. Run the global-shuffle sampler control; broaden repeats only if the first
   seed shows a meaningful effect.
7. Verify every core summary has all 300 frames, the exact camera split,
   completed steps, exact target-pixel cost, finite metrics, train/heldout
   media, and schema-v2 identities before adding a paper row to
   `BASELINES.md`.
8. Run `generate_world_tubes_paper_artifacts.py` without
   `--allow-incomplete`; only its verified, submission-ready bundle may feed
   the final manuscript tables and plots.

These seven Coffee Martini rows are the minimum selected-time
representation-and-cost context table. The separate frozen identical-world
lane supplies public causal replay-versus-compiled evidence for the projective
atlas. The remaining 14 full-matrix rows are breadth work after the minimum
paper cut: alternate triplets, two additional Neural3D scenes, D-NeRF, and the
separately labelled deterministic audit.

Launch the seven-row queue resumably with `--reuse-existing`,
`--max-new-runs 1`, and one exact `--run-key` copied from the dry-run output.
The runner records partial progress without emitting a false complete matrix;
final tables are generated only once all seven rows validate.

Primary clean-source command:

```bash
PYTHONPATH=src/train:third_party/powerfoam-metal .venv/bin/python \
  research_experiments/paper_runner_suite/run_unified_paper_ablation.py \
  --execute \
  --protocol src/train_configs/paper_protocols/coffee_martini_full_300f_progressive_512_v1.jsonc \
  --require-clean-source \
  --wandb-mode online \
  --allow-local-mps-execution
```

The MPS acknowledgement is required on an operator-approved Apple execution
host. It does not bypass the source-derived memory estimate, process-RSS guard,
or any live memory, swap, disk, or load gate. A freshly quiet 16-GB host may
qualify; eligibility is decided at launch and continuously during the child.

Publication-scale execution is fail-closed on the incident workstation. The
fixed-512 attempt was killed under severe unified-memory pressure; its partial
outputs are invalid. The full 21-row selected-time public workload is fixed in
`src/train_configs/paper_protocols/world_tubes_full_public_matrix_v1.jsonc`,
with zero schema-v2 rows accepted. The seven-row submission subset is fixed in
`world_tubes_submission_matrix_v1.jsonc` and is the minimum rerun queue; the
other 14 rows remain the full-breadth target. The older `3/21` count and
seven-run aggregate are historical schema-v1 artifacts, not an authoritative
publication ledger.

The two matrices have distinct canonical schema-v2 roots and must never share
or overwrite a `matrix_summary.json`:

- seven-row submission subset:
  `outputs/benchmarks/2026-07-28_world_tubes_submission_matrix_schema2`;
- 21-row full-breadth matrix:
  `outputs/benchmarks/2026-07-28_world_tubes_full_public_matrix_schema2`.

Both roots are declared in their matrix configs. Omit `--out-dir` to use that
declaration, or pass the matching path explicitly.

## P1: Bounded Residency Before Any Native-Resolution Promotion

The eager path is not acceptable for 2704x2028 because all-frame float targets
and per-sample ray grids scale to tens of gigabytes. Implement in this order:

1. **Implemented, runtime-unverified:** camera/calibration tensors can live on
   the compute device while paper targets remain disk-backed.
2. **Implemented, runtime-unverified:** decode only selected source frames at
   the requested resolution with a bounded CPU LRU and bounded identity pass.
3. **Implemented, runtime-unverified:** generate calibrated PowerFoam ray grids
   only for selected samples.
4. **Implemented, runtime-unverified:** stream train/heldout evaluation in
   bounded chunks and retain only capped media frames.
5. Reuse the current/driver device-memory sampler already present in the common
   cost ledger.
6. Pass a one-step all-300-frame native-resolution MPS smoke before creating a
   native progressive quality protocol.

## P2: Breadth And Paper Tables

After the seven primary/control rows pass, add camera-triplet breadth on Coffee
Martini, then `cook_spinach` and `cut_roasted_beef`. The public-data ingest is
checked in and all three declared Neural3D scenes are present locally. Every
breadth row must choose an explicit scene-specific WorldFoam point cloud or
the labelled video initializer; the runner will not silently reuse Coffee
Martini geometry.

The controlled D-NeRF ingest/validator is checked in for `bouncingballs` and
`mutant`. D-NeRF is a separately labelled posed-frame negative/control. Under
the current one-frame-per-chart adapter it may test correctness and fallback
behavior, but it must not be aggregated with synchronized multicamera rows or
cited as sublinear bounded-chart scaling.

Report quality against active render count and total stored state; equal nominal
primitive count is not equal capacity for shared tubes versus per-frame state.

## P3: Submission Package

1. Regenerate the authoritative seven-row submission ledger,
   JSON/CSV/Markdown/LaTeX tables, and figures from accepted artifacts. Extend
   it to the 21-row breadth ledger when those rows exist.
2. Add the frozen identical-world public result and public same-checkpoint
   frame-count sweep.
3. Lock clean commits, native-binary identity, configs, dataset checksums,
   evaluator contracts, W&B run IDs, and reproduction commands.
4. Convert the generated standalone TeX into the venue template, build a clean
   PDF, and visually inspect every page.
5. Package one runnable demo command and a concise artifact manifest.

## Explicit Stop List

Until the matrix and manuscript are submission-complete, do not spend paper
time on browser training, V-JEPA/world-token work, 300-clip feature sweeps,
Softmax variants, `direct_serial` promotion, new gauge theories, native
WorldFoam shader expansion, native 2704x2028 quality runs, or external SOTA
reproduction. Retain their code and artifacts for provenance, but do not route
new work into those lanes.
