# Softmax-GS Plan Docs Refresh

## Context

User asked what work remains now and asked for short-term and long-term plan
docs. Existing plan docs already lived under
`research_notes/gaussian_splatting_papers/`, but they needed to reflect the
latest recompute-backward and contribution-tape state.

## What Changed

Updated:

- `research_notes/gaussian_splatting_papers/2026-05-25_short_term_softmax_gs_plan.md`
- `research_notes/gaussian_splatting_papers/2026-05-25_long_term_splats_vs_worldfoam_plan.md`
- `research_experiments/softmax_gs/REFERENCE_REPORT.md`
- `EXPERIMENTS.md`

Also fixed the contribution-tape test assertion so it checks
`final_contribution_weight`, not the immediate insertion-time contribution
weight. Future Softmax-GS prefix rescaling can change earlier weights, so the
final tape row is the native-backward contract.

## Current Model

Short term:
    Dynamic GS is the first target. Finish native/tape backward for
    `v5_softmax_gs`, then run one meaningful matched quality row. Do not port
    Softmax-GS to STAR until dynamic GS shows a real measured gain.

Long term:
    Better splats/STAR remain the mainline for now. WorldFoam is the serious
    challenger, not the default future by aesthetics. It needs a matched
    heldout-quality/trainability win before replacing the splat-time route.

## Evidence

Focused reference gate:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_softmax_gs_reference.py -q
```

Result:

```text
8 passed
```

## Next Work

1. Implement native/tape backward for the Softmax-GS scalar update.
2. Add tiny MPS parity against the Torch recompute scaffold.
3. Run one-step enabled train smoke without the Torch recompute path.
4. Run matched dynamic-GS quality diagnostic with W&B on if it is no longer a
   purely mechanical local smoke.
5. Only then decide whether a STAR CPU diagnostic is worth doing.
