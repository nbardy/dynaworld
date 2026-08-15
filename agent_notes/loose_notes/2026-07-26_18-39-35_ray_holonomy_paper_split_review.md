# Ray Holonomy Paper-Split Review

Date: 2026-07-26 KST

## Trigger

Reviewed the user-provided proposal for a "gauge-invariant ray holonomy
renderer for moving cameras" and reconciled it against the current World
Tubes / STAR UVT and WorldFoam paper lanes.

Source attachment:

```text
/Users/nicholasbardy/.codex/attachments/c3f6b522-fd32-4797-941d-8fc2ed5722e2/pasted-text.txt
```

## Files Read

```text
PROJECT_INDEX.md
README.md progress section
TODO/README.md
research_notes/README.md
research_notes/renderer_lane_taxonomy.md
research_notes/framing_the_problem/README.md
research_notes/gauged_uvt_trace_atlas/DEPTH_FIBER_CROSS_TRACK_NOTE.md
research_notes/gauged_uvt_trace_atlas/08_worldfoam_bridge/README.md
research_notes/worldfoam_paper/README.md
research_notes/worldfoam_paper/WORLD_FOAM_OPTICAL_TRANSFER_PAPER_PLAN.md
research_notes/worldfoam_paper/WORLD_FOAM_MATH_APPENDIX.md
agent_notes/loose_notes/2026-07-17_23-59-22_world_tubes_lane_closeout_and_repo_integration.md
agent_notes/loose_notes/2026-07-19_03-46-49_renderer_lane_taxonomy_correction.md
```

The repository had pre-existing dirty research-note changes. This pass did not
rewrite or discard them; it added a dedicated intake note and narrow links /
dated taxonomy clarification.

## Decision

The proposal belongs primarily to the existing WorldFoam second-paper lane,
not as a wholesale extension of the current STAR UVT paper.

```text
Shared:
    gauged camera-ray bundle
    fiber-coordinate invariance
    certified camera domains

World Tubes / STAR UVT Paper A:
    Gaussian-compatible early depth pushforward
    conditional depth/order certificates
    visibility strata/fallback
    compiled interval direct adjoint

WorldFoam / Ray-Holonomy Paper B:
    retained depth fiber
    path-ordered optical transfer
    noncommutative visibility
    convex-potential atom branch
    discriminant-certified fiber compiler
```

## Important Terminology

The proposal sharpened an existing ambiguity:

```text
camera program = the measurement, so changing it changes the observation
gauge          = a coordinate/trivialization choice inside that measurement
ray            = the one-dimensional fiber
```

"Ray holonomy" is the ordered transfer along that fiber. It is distinct from
the older cell-graph loop-holonomy diagnostic.

## Strongest New Material

```text
self-normalized strongly convex polynomial spacetime atoms
guaranteed compact/connected/smooth slices
derived ridge and kinematics
one support interval per straight ray
exact single-atom optical depth
endpoint-free first derivatives
adaptive discriminant-certified trace compilation
Duhamel transfer derivative and VJP-error bounds
```

These are untested representation proposals, not current implementation facts.

## Resulting Durable Note

Created:

```text
research_notes/worldfoam_paper/scientist_notes/
2026-07-26_gauge_invariant_ray_holonomy_intake_and_paper_split.md
```

It preserves the full proposal in normalized form, states what is already in
WorldFoam, identifies genuinely new content, gives the paper split, records
critical risks, and defines a falsification ladder.

## Next Work

Do not interrupt the frozen World Tubes paper matrix for another renderer
rewrite. The first future Paper-B experiment should be either:

```text
matched convex-potential atom capacity/conditioning against Gaussian, cell,
and capped-determinant baselines

or

a decisive moving-camera colored-overlap row where measured STAR commutator
energy/fallback predicts a quality gap and retained transfer can close it
```

