# Next Run Handoff — Evidence Execution Only

This file is subordinate to `GOAL.md`. Use the exact commands and acceptance
gates in the Paper A and Paper B master plans. Do not create the previously
proposed `src/train/kinetic_core/` abstraction as a prerequisite:
consolidation is a separate cleanup task and must not block evidence.

## 1. Paper A preflight

Run the focused source/contract gate in
`TODO/world_tubes_paper_finish_master_plan_2026-08-13.md`, then:

```bash
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/paper_runner_suite/run_unified_paper_matrix.py \
  --preflight-only \
  --matrix src/train_configs/paper_protocols/world_tubes_submission_matrix_v1.jsonc \
  --out-dir outputs/benchmarks/2026-07-28_world_tubes_submission_matrix_schema2 \
  --device mps --wandb-mode online --check-wandb-connectivity
```

If this fails a host or external prerequisite, stop Paper A for the night. If
it exposes a reproduced correctness defect, fix only that frozen-contract
defect and rerun the same gate.

## 2. Paper A bounded evidence smoke

On an operator-approved quiet MPS host:

```bash
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/paper_runner_suite/run_unified_paper_matrix.py \
  --execute --require-clean-source \
  --matrix src/train_configs/paper_protocols/world_tubes_evidence_smoke_matrix.jsonc \
  --out-dir outputs/benchmarks/2026-08-10_world_tubes_schema2_evidence_smoke \
  --device mps --wandb-mode offline --allow-local-mps-execution
```

Only after the smoke verifies should the Paper A operator follow master-plan
phases P2–P4 for the frozen same-world sweep, bounded variable-camera curve,
and seven public contexts.

## 3. Paper B dry plan and guarded execution

Start with the allocation-free G6 plan:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/run_worldfoam_g6_clean_host_bundle.py
```

Run the G4-v2 pilot dry plan separately. If and only if both dry plans have no
external blockers and the live host guards pass, use the Paper B master-plan
commands for one guarded rebuild, the real G4 two-route pilot, and then the
G6/G4 matrices. Never run Paper A and Paper B accelerator jobs concurrently.

## 4. Accepted tables and manuscripts

Regenerate the existing Paper A bundle from verified artifacts:

```bash
python3 research_experiments/paper_runner_suite/generate_world_tubes_paper_artifacts.py \
  --verify-dir research_notes/gauged_uvt_trace_atlas/paper/generated/schema_v2
```

Update the existing drafts only after newly accepted data exists. The `ICLR`
substring in historical filenames does not select a venue. Do not build a
project page, venue template, additional manuscript, new verifier, or another
status audit overnight.

## Goal invocation

Use:

> Execute `GOAL.md` evidence-first. Use at most two non-recursive
> medium-reasoning subagents with disjoint Paper A/Paper B artifact ownership;
> run accelerator work sequentially; stop at the first resource/external
> blocker; create no new audits, verifiers, plans, methods, venue scaffolds, or
> cleanup work. Hard token cap 2M; stop new work at 1.6M.
