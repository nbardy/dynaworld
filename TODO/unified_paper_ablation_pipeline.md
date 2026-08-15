# Unified Paper Ablation Pipeline

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

Evidence schema v2 is additionally implemented and statically checked, but not
yet behavior-verified. It binds the exact ordered sample schedule, all raw
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

1. Run focused schema-v2 behavior tests and a bounded, quiet-host three-lane
   evidence smoke. This must verify actual W&B file discovery, decoded-bundle
   equality, route-native identity, evaluator equality, reuse rejection, and
   matrix aggregation without launching a publication row.
2. Run the implemented lane-isolated frozen identical-world comparison via
   `run_frozen_world_replay_compiled.py` with
   `--frame-counts 0,4,8,16,32,64,128`. It now trains and saves once, evaluates
   every `F` from the exact same world, and samples each row across the same
   full physical interval. First verify non-unit selected-time full-atlas
   versus chunk-slice forward/VJP parity. The integrated sweep retains
   single-shot correctness timings and separately records alternating paired,
   synchronized timing trials; use at least one warmup and three repeats
   (`1/5` is the publication-runner default).
3. Run the implemented bounded variable-camera closure/death gate with its
   fixed world, fixed physical interval, and exact rational live-depth-order
   oracle. The static public causal row cannot by itself support the
   moving-camera claim.
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
host. It does not bypass the incident-calibrated memory estimate or any live
memory, swap, disk, or load gate; the incident workstation remains ineligible.

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

1. **Partial:** camera/calibration tensors can live on the compute device while
   paper targets remain host-resident.
2. **Missing:** decode only the sampled K source frames at the active stage
   resolution with a bounded CPU cache. Current decoded target videos are
   still host-eager.
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
