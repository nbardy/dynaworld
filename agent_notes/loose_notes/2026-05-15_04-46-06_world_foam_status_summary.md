# World Foam fused slab status summary

## Context

After the fused mixed shader work, the evidence was scattered across verifier
JSON, train/eval artifacts, raw/autograd VJP smokes, and negative probe
artifacts. That made it too easy to answer from memory and too hard for a
future agent to see which result is canonical.

## Change

Added a status/audit script:

```text
research_experiments/world_foam_lane2/summarize_fused_slab_mixed_results.py
```

It loads the current aggregate verifier, depth-order probe, and owner-update
failure artifact, then writes a single JSON summary with:

- explicit prompt-to-artifact checklist
- winner and per-mode speed table
- matched-frame PSNR spread across modes
- RGB and RGBA/depth smoke coverage
- raw plus autograd VJP coverage
- rejected owner-update and ordered-append variants
- explicit `completion_claim=false`
- explicit `star_uvt_competitive_claim=false`

Also added a short pointer section near the top of:

```text
research_experiments/world_foam_lane2/README.md
```

## Evidence

Ran:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/summarize_fused_slab_mixed_results.py \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_status_summary.json
```

Output status:

```text
status: ok_current_shader_gate_with_structural_gap
completion_claim: false
star_uvt_competitive_claim: false
missing_checklist_items: []
winner: direct_atomic_grad_only
total 2f->16f: 7.17 ms -> 9.32 ms
total scale 2f->16f: 1.30x
render scale 2f->16f: 1.06x
backward scale 2f->16f: 1.74x
max matched-frame PSNR spread across modes: 2.06e-6
```

Validation:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/summarize_fused_slab_mixed_results.py \
  research_experiments/world_foam_lane2/verify_fused_slab_mixed_scaling.py \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_fused_slab_affine_realray_mps.py

git diff --check -- \
  research_experiments/world_foam_lane2/README.md \
  research_experiments/world_foam_lane2/summarize_fused_slab_mixed_results.py \
  research_experiments/world_foam_lane2/verify_fused_slab_mixed_scaling.py \
  agent_notes/loose_notes \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_fused_slab_affine_realray_mps.py
```

Both passed.

## Takeaway

The current shader gate is now easy to verify from one artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_status_summary.json
```

This should be treated as "current World Foam fused shader path verified with a
known structural gap", not as a STAR-UVT-competitive completion claim.
