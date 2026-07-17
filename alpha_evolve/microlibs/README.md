# Microlib Contract

A microlib is the smallest unit the evolver should mutate. It is not
necessarily a Python package. It is a bounded problem package: code surface,
prompt, evaluator, and score schema.

## Required Fields

Every microlib spec should answer:

- Problem: what behavior should improve?
- Why now: why is this a current DynaWorld bottleneck?
- Allowed edits: which files can Codex touch?
- Baseline: what is the current measured row?
- Evaluator cascade: what commands return JSON scores?
- Primary metrics: what gets optimized?
- Hard rejects: what candidate behavior is invalid?
- Promotion gate: what proof is needed before touching canonical docs or
  claiming a lane changed?

## Microlib Types

### Implementation Microlib

Edits production or experiment code. Requires runtime smoke after signature,
config, dataclass, or return-shape changes.

Examples:

- STAR UVT feature RGB-gradient handoff
- mixed scheduler
- Gaussian promotion guard

### Evaluator Microlib

Builds or tightens evaluators before implementation evolution. The output is a
script/test that future implementation candidates cannot edit.

Examples:

- source/camera-disjoint manifest validator
- heldout leakage probe
- tile-overflow JSON schema checker

### Search-Heuristic Microlib

Evolves a script that searches config/implementation variants inside a fixed
budget, rather than directly evolving the final patch.

Examples:

- short STAR support-pruning schedule search
- Gaussian promotion guard parameter search

## Candidate Patch Rules

- Candidate patches should be easy to review and revert.
- Candidate patches should not modify evaluator files unless the microlib is an
  evaluator microlib.
- Candidate patches should not write large artifacts into tracked directories.
- Candidate patches should keep checked-in JSONC configs as the source of
  hyperparameter truth.

## Score Schema

Use consistent names so cross-problem dashboards work:

```json
{
  "status": "passed",
  "correct": true,
  "finite": true,
  "scope_ok": true,
  "primary_score": 0.0,
  "loss_initial": null,
  "loss_final": null,
  "psnr_final": null,
  "total_step_ms": null,
  "forward_ms": null,
  "backward_ms": null,
  "overflow_tile_count": null,
  "changed_loc": null,
  "notes": []
}
```

Microlibs can add lane-specific fields, but these names should stay stable.
