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

## Frozen Coffee Martini matrix

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

D-NeRF rows require the posed-frame adapter described in
`research_notes/data_contract.md`; they must not be injected into the
synchronized multicamera matrix.

## Manuscript

```bash
pandoc research_notes/gauged_uvt_trace_atlas/paper/WORLD_TUBES_PAPER_DRAFT.md \
  --standalone --from gfm --to latex \
  --metadata title='World Tubes in Gauged Camera Space: Sublinear Frame Scaling for Dynamic Gaussian Splatting' \
  --metadata author='Anonymous' \
  --output research_notes/gauged_uvt_trace_atlas/paper/WORLD_TUBES_PAPER.tex
```

The paper deliberately claims bounded tested chart segments. Do not restore
full `360/720` multi-gauge language unless the chart-transition implementation
and orbit test are both present.
