# World Tubes Paper Draft

Date: 2026-07-04 16:11:57 Asia/Seoul

## Context

The user selected the title direction:

```text
World Tubes in Gauged Camera Space:
Sublinear Frame Scaling for Dynamic Gaussian Splatting
```

and asked to write out the paper plus the ablations, charts, datasets, and
baselines needed to compare it.

## Work Done

Created a paper folder:

```text
research_notes/gauged_uvt_trace_atlas/paper/
```

Added:

```text
WORLD_TUBES_PAPER_DRAFT.md
WORLD_TUBES_EXPERIMENT_PLAN.md
```

Updated:

```text
research_notes/gauged_uvt_trace_atlas/README.md
```

to point new agents at the paper folder.

## Evidence Anchors Used

The draft is grounded in the current verified projective/gauged evidence:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_goal_completion_promotion_audit/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_goal_final_completion_audit/summary.json
research_notes/gauged_uvt_trace_atlas/GOAL_META_KEY_MATH.md
research_notes/gauged_uvt_trace_atlas/README.md
```

Important current internal metrics included in the draft:

```text
orbit payload-growth ratio: 0.125
trained interval-entry growth ratio: 0.148
final trained trace-count ratio: 0.1
final trained forward ratio: <= 0.266
final trained backward ratio: <= 0.094
fresh-process median no-first ratio: 0.565
fresh-process median projective-total ratio: 0.836
compiled trainer case payload count: 20
broad trainer/quality/media distinct sources: 10
```

## Claim Boundary

The draft intentionally frames the contribution as a renderer/compiler layer:

```text
known or low-dimensional camera programs expose repeated world-side work;
camera-gauged world tubes compile that work into reusable sensor-time traces.
```

It does not claim universal replacement for all 4DGS methods, arbitrary
single-frame novel-view speedup, or broad SOTA dynamic-view quality.

## Next Steps

1. Convert the Markdown draft to LaTeX once the first public dataset table is
   available.
2. Build a canonical paper demo command:

```text
compile atlas -> render frame stack -> run backward -> emit JSON + contact sheet
```

3. Run the synthetic trace suite first. This is the cleanest validation of the
   Schur/fiber/gauge math.
4. Run same-representation public subset experiments before spending time on
   external SOTA baselines.
5. Generate the paper charts from JSON summaries rather than copying numbers
   by hand.
