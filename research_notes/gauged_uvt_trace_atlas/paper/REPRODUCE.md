# World Tubes paper reproduction

The submission evidence contract is intentionally smaller than the historical
research tree. Run from the DynaWorld repository root.

## Environment

```bash
uv sync --group experiments
```

Submission rows must run from a clean superproject plus clean STAR submodule.
The runner records both exact commits and fails early when
`--require-clean-source` is set.

## One-command public matrix

The complete public-data workload is frozen in
`src/train_configs/paper_protocols/world_tubes_full_public_matrix_v1.jsonc`.
It contains 21 independent protocol/seed rows: the seven Coffee Martini
progressive/control rows, six alternate-camera-triplet rows, six rows across
two additional Neural3D scenes, one separately labelled D-NeRF control, and
one 64-wide deterministic STAR correctness/timing audit that must not be
aggregated with the 512-wide quality rows.

On a sufficiently provisioned machine, the single reproduction command is:

```bash
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/paper_runner_suite/run_unified_paper_matrix.py \
  --execute \
  --require-clean-source \
  --matrix src/train_configs/paper_protocols/world_tubes_full_public_matrix_v1.jsonc \
  --out-dir outputs/benchmarks/world_tubes_full_public_matrix_v1 \
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

## Frozen Coffee Martini control matrix

The seven-run Coffee Martini control subset is currently blocked on this 24GB unified-memory
workstation after an operator-killed memory-pressure incident. The runner is
fail-closed on local MPS and the command below will refuse to start unless an
operator explicitly supplies both safety acknowledgements. Do not do that on
the incident machine. Use streamed targets/rays/evaluation or a sufficiently
provisioned Apple host.

```bash
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/paper_runner_suite/run_unified_paper_matrix.py \
  --execute \
  --require-clean-source \
  --matrix src/train_configs/paper_protocols/world_tubes_submission_matrix_v1.jsonc \
  --out-dir outputs/benchmarks/world_tubes_submission_matrix_v1 \
  --device mps \
  --wandb-mode online
```

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
  --matrix src/train_configs/paper_protocols/world_tubes_submission_matrix_v1.jsonc \
  --out-dir outputs/benchmarks/2026-07-22_world_tubes_submission_matrix_clean_v1
```

This emits `existing_evidence_summary.json` and an
`accepted_existing_evidence/` bundle. It accepts only complete
`run_summary.json` files with clean repository and STAR provenance, so partial
lane debris from an interrupted run cannot enter the table.

## Same-representation scaling and theorem table

The accepted scaling artifact is:

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

## Manuscript

```bash
pandoc research_notes/gauged_uvt_trace_atlas/paper/WORLD_TUBES_PAPER_DRAFT.md \
  --standalone --from gfm --to latex \
  --output research_notes/gauged_uvt_trace_atlas/paper/WORLD_TUBES_PAPER.tex
```

The paper deliberately claims bounded tested chart segments. Do not restore
full `360/720` multi-gauge language unless the chart-transition implementation
and orbit test are both present.
