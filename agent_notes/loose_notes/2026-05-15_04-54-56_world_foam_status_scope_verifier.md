# World Foam status scope verifier

## Context

The canonical fused slab mixed summary had all evidence checklist items true
while still keeping `completion_claim=false` and
`star_uvt_competitive_claim=false`. That is intentional, but easy to misread:
the current shader gate is verified, while the STAR-competitive/full-completion
claim is still out of scope.

## Change

Added:

```text
research_experiments/world_foam_lane2/verify_fused_slab_status_summary.py
```

The verifier checks that the status summary:

- remains `ok_current_shader_gate_with_structural_gap`
- keeps `completion_claim=false`
- keeps `star_uvt_competitive_claim=false`
- records explicit open items before completion
- records `direct_atomic_grad_only` as winner
- keeps matched-frame PSNR spread below tolerance
- records owner-update and ordered-append as rejected
- includes a STAR speed reference with an explicit non-matched scope note

Regenerated:

```text
research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_status_summary.json
research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_status_summary_verifier.json
```

## Evidence

Ran:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_fused_slab_status_summary.py \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_status_summary_verifier.json
```

Result:

```text
status: ok
failures: []
```

## Takeaway

Future agents should use the verifier before treating the fused World Foam
summary as authoritative. It is a scoped success artifact for the current shader
gate, not a completion or STAR-competitive acceptance gate.
