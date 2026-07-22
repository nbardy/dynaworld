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
  diagnostics under evidence schema v1.
- The 4-frame staged MPS smoke and all-300-frame MPS smoke are complete.
- `run_unified_paper_matrix.py` expands a declarative protocol/seed/policy
  matrix, validates every lane fail-closed, and emits row JSON, CSV, Markdown,
  LaTeX, and SVG plot artifacts.
- The exact same-representation replay-versus-compiled scaling row is complete
  for `F={4,8,16,32,64,128}`. The compiled payload remains fixed while replay
  grows `32x`; the checked theorem table consumes this result and the existing
  certified synthetic fixtures.
- The paper claim is explicitly bounded to tested camera-chart segments; no
  unimplemented `360/720` multi-gauge transition is claimed.
- Submission runs can require a clean superproject and STAR submodule state;
  both exact commits are recorded. W&B code/diff upload is disabled for these
  runs because resolved configs plus clean commit hashes are the reproducible
  source contract.

## P0: Produce Paper Rows

1. Finish the structured progressive 512-wide protocol for seeds 17/29/43.
   **Complete:** all three clean-source seeds are accepted across World Tubes,
   WorldFoam, and dynamic 3DGS.
2. Run the exact target-pixel-matched fixed-512 control for the same seeds.
3. Run the global-shuffle sampler control; broaden repeats only if the first
   seed shows a meaningful effect.
4. Run a separately labeled deterministic World Tubes correctness/timing audit;
   do not substitute it for the normal throughput kernel.
5. Verify every summary has all 300 frames, the exact camera split, completed
   steps, exact target-pixel cost, finite metrics, train/heldout media, and W&B
   provenance before adding a row to `BASELINES.md`.

Primary clean-source command:

```bash
PYTHONPATH=src/train:third_party/powerfoam-metal .venv/bin/python \
  research_experiments/paper_runner_suite/run_unified_paper_ablation.py \
  --execute \
  --protocol src/train_configs/paper_protocols/coffee_martini_full_300f_progressive_512_v1.jsonc \
  --require-clean-source \
  --wandb-mode online
```

Publication-scale execution is fail-closed on the incident workstation. The
fixed-512 attempt was killed under severe unified-memory pressure; its partial
outputs are invalid. The full 21-row public workload is frozen in
`src/train_configs/paper_protocols/world_tubes_full_public_matrix_v1.jsonc`,
with 3 accepted and 18 results missing.

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

After the primary rows pass, add camera-triplet breadth on Coffee Martini,
then `cook_spinach` and `cut_roasted_beef`. The public-data ingest is checked in
and the two Neural3D archives are being acquired. Every breadth row must choose
an explicit scene-specific WorldFoam point cloud or the labelled video
initializer; the runner will not silently reuse Coffee Martini geometry.

The controlled D-NeRF ingest/validator is checked in for `bouncingballs` and
`mutant`. D-NeRF is monocular and asynchronous rather than synchronized
multicamera data, so it needs a separate controlled posed-frame adapter; do not
mislabel it as a row from the three-lane multicamera protocol.

Report quality against active render count and total stored state; equal nominal
primitive count is not equal capacity for shared tubes versus per-frame state.

## Explicit Stop List

Until the matrix and manuscript are submission-complete, do not spend paper
time on browser training, V-JEPA/world-token work, 300-clip feature sweeps,
Softmax variants, `direct_serial` promotion, new gauge theories, native
WorldFoam shader expansion, native 2704x2028 quality runs, or external SOTA
reproduction. Retain their code and artifacts for provenance, but do not route
new work into those lanes.
